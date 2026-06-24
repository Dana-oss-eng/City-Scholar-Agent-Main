"""本模块作用：CityScholar-Agent 核心编排 —— 知识库、检索、问答、分析与多工具调度。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import OUTPUT_DIR
from core.memory import ConversationMemory
from core.prompts import (
    RESEARCH_GAP_PROMPT, ENHANCED_ANALYSIS_PROMPT, LLM_COMPARE_PROMPT,
    LLM_OUTLINE_PROMPT,
    build_answer_suffix, build_answer_system_prompt, build_answer_task_prompt,
    build_empty_library_message, build_no_result_message,
    build_memory_enhanced_context,
)
from core.workflow import (
    WorkflowRunResult, WorkflowState,
    build_default_workflow_plan, export_workflow_result,
    format_workflow_run_result, mark_step_status,
)
from rag.embedder import (
    EmbeddingIndex,
    build_embedding_index as run_build_embedding_index,
    get_default_embedding_index_path,
    is_embedding_index_compatible,
    load_embedding_index,
)
from rag.loader import list_pdf_files
from rag.parser import (
    PaperMetadata, ParsedDocument,
    parse_pdf_files, enrich_documents_with_metadata,
)
from rag.retriever import (
    RetrievedChunk,
    extract_search_terms,
    retrieve_by_keyword,
    retrieve_hybrid,
    expand_query,
    rerank_with_llm,
)
from rag.splitter import build_knowledge_base_records
from rag.graphrag import (
    EntityGraph, build_entity_graph, export_graph_png,
    graph_enhanced_retrieval, format_graph_context, get_graph_stats,
)
from tools.analyze_tool import (
    StructuredPaperAnalysis, analyze_single_paper, format_analysis_result,
)
from tools.compare_tool import (
    MultiPaperComparison, compare_papers as run_compare_papers, format_comparison_result,
)
from tools.outline_tool import (
    ReviewOutline, format_review_outline, generate_review_outline,
)
from tools.summarize_tool import generate_summary, format_summary
from tools.translate_tool import translate
from tools.recommend_tool import recommend_similar_papers, format_recommendations
from tools.bibtex_tool import export_all_bibtex, export_single_bibtex


# ====== 响应数据结构 ======

@dataclass
class KnowledgeBaseState:
    raw_papers_dir: str
    pdf_files: list[str] = field(default_factory=list)
    documents: list[ParsedDocument] = field(default_factory=list)
    chunk_records: list[dict] = field(default_factory=list)
    parse_errors: list[dict[str, str]] = field(default_factory=list)


@dataclass
class AgentAnswer:
    question: str
    model_answer: str
    sources: list[dict]
    retrieved_count: int
    used_prompt: str


@dataclass
class PaperAnalysisResponse:
    target: str
    status_message: str
    analysis: StructuredPaperAnalysis | None
    formatted_output: str


@dataclass
class PaperComparisonResponse:
    targets: list[str]
    status_message: str
    comparison: MultiPaperComparison | None
    formatted_output: str


@dataclass
class ReviewOutlineResponse:
    topic: str
    status_message: str
    outline: ReviewOutline | None
    formatted_output: str


@dataclass
class WorkflowResponse:
    topic: str
    status_message: str
    workflow_result: WorkflowRunResult | None
    formatted_output: str


@dataclass
class EmbeddingIndexResponse:
    status_message: str
    index_path: str
    vector_count: int
    loaded_from_cache: bool


@dataclass
class SummarizeResponse:
    file_name: str
    formatted_output: str


@dataclass
class TranslateResponse:
    original: str
    translated: str
    direction: str


@dataclass
class RecommendResponse:
    target: str
    recommendations: list[dict]
    formatted_output: str


# ====== 辅助函数 ======

def _flatten_llm_value(val) -> str:
    """将 LLM 返回的各种值格式化为可读文本。"""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, (list, tuple)):
        return "；".join(str(v).strip() for v in val if str(v).strip())
    if isinstance(val, dict):
        parts = []
        for k, v in val.items():
            if v and str(v).strip():
                parts.append(str(v).strip())
        return "；".join(parts)
    return str(val).strip()


def _parse_llm_json(raw: str) -> dict:
    """从 LLM 返回的文本中提取 JSON 对象，处理各种包裹和噪声情况。"""
    text = raw.strip()
    # 去掉 markdown 代码块包裹 (```json ... ``` 或 ``` ... ```)
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl >= 0:
            text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    # 尝试找到 JSON 对象的起始和结束位置
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


# ====== 主 Agent ======

class CityScholarAgent:
    """城市研究论文学术助手 —— 统一管理建库、检索、分析与科研工具。"""

    def __init__(
        self,
        raw_papers_dir: str | Path,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        top_k: int = 5,
    ) -> None:
        self.raw_papers_dir = Path(raw_papers_dir).expanduser().resolve()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.knowledge_base: KnowledgeBaseState | None = None
        self.embedding_index: EmbeddingIndex | None = None
        self.embedding_client = None
        self.embedding_model_name: str = ""
        self.embedding_dimensions: int = 0
        # 多 LLM 客户端（由外部注入）
        self.chat_client = None      # 主对话模型
        self.analysis_client = None  # 分析模型（长上下文）
        # 对话记忆
        self.memory = ConversationMemory()
        # 元数据提取是否已完成
        self._metadata_enriched: bool = False
        # GraphRAG 实体图
        self.entity_graph: EntityGraph | None = None

    # ====== 知识库构建 ======

    def build_knowledge_base(self) -> KnowledgeBaseState:
        pdf_paths = list_pdf_files(self.raw_papers_dir)
        documents, parse_errors = parse_pdf_files(pdf_paths)
        chunk_records = build_knowledge_base_records(
            documents, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,
        )
        self.knowledge_base = KnowledgeBaseState(
            raw_papers_dir=str(self.raw_papers_dir),
            pdf_files=[str(p) for p in pdf_paths],
            documents=documents, chunk_records=chunk_records, parse_errors=parse_errors,
        )
        return self.knowledge_base

    def enrich_metadata(self) -> None:
        """为所有已解析论文提取元数据（使用 chat_client）。"""
        if self._metadata_enriched or self.knowledge_base is None:
            return
        if self.chat_client is None:
            return

        def _chat_fn(messages, **kwargs):
            return self.chat_client.chat(messages, **kwargs)

        enrich_documents_with_metadata(self.knowledge_base.documents, _chat_fn)
        # 元数据更新后重建 chunk records 以携带新元数据
        self.knowledge_base.chunk_records = build_knowledge_base_records(
            self.knowledge_base.documents,
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,
        )
        self._metadata_enriched = True

    def ensure_knowledge_base_ready(self) -> KnowledgeBaseState:
        if self.knowledge_base is None:
            return self.build_knowledge_base()
        return self.knowledge_base

    def list_available_papers(self) -> list[dict]:
        kb = self.ensure_knowledge_base_ready()
        items: list[dict] = []
        for i, doc in enumerate(kb.documents, 1):
            items.append({
                "index": i, "file_name": doc.file_name,
                "document_id": doc.document_id,
                "total_pages": doc.total_pages,
                "total_characters": doc.total_characters,
                "title": doc.metadata.title or "",
                "authors": doc.metadata.authors,
                "year": doc.metadata.year,
                "journal": doc.metadata.journal,
            })
        return items

    def find_document(self, target: str | None = None) -> ParsedDocument | None:
        kb = self.ensure_knowledge_base_ready()
        if not kb.documents:
            return None
        if target is None or not target.strip():
            return kb.documents[0]
        nt = target.strip().lower()
        if nt.isdigit():
            idx = int(nt) - 1
            if 0 <= idx < len(kb.documents):
                return kb.documents[idx]
            return None
        for doc in kb.documents:
            if nt == doc.file_name.lower() or nt == doc.document_id.lower():
                return doc
        for doc in kb.documents:
            if nt in doc.file_name.lower() or nt in doc.document_id.lower():
                return doc
        return None

    def find_documents(self, targets: list[str] | None = None, default_count: int = 2) -> list[ParsedDocument]:
        kb = self.ensure_knowledge_base_ready()
        if not kb.documents:
            return []
        if not targets:
            return kb.documents[:min(default_count, len(kb.documents))]
        selected: list[ParsedDocument] = []
        seen: set[str] = set()
        for t in targets:
            doc = self.find_document(t)
            if doc is None:
                raise ValueError(f"未找到目标论文：{t}")
            if doc.document_id not in seen:
                seen.add(doc.document_id)
                selected.append(doc)
        return selected

    # ====== 检索管道 ======

    def _get_chat_fn(self):
        """获取 chat 函数，无客户端时返回 None。"""
        if self.chat_client is None:
            return None
        return lambda messages, **kw: self.chat_client.chat(messages, **kw)

    def retrieve_chunks_for_question(self, question: str, chunk_records: list[dict]) -> list[RetrievedChunk]:
        chat_fn = self._get_chat_fn()

        # Step 1: 查询扩展
        expanded_queries = expand_query(question, chat_fn)

        # Step 2: 检索（混合优先，回退关键词）
        all_results: list[RetrievedChunk] = []
        for q in expanded_queries:
            if (
                self.embedding_index is not None
                and self.embedding_client is not None
                and self.embedding_model_name
                and self.embedding_dimensions > 0
            ):
                try:
                    qv = self.embedding_client.embed_texts(
                        model=self.embedding_model_name,
                        texts=[q], dimensions=self.embedding_dimensions,
                    )[0]
                    results = retrieve_hybrid(
                        q, chunk_records, self.embedding_index.vectors, qv, top_k=self.top_k,
                    )
                except Exception:
                    results = retrieve_by_keyword(q, chunk_records, top_k=self.top_k)
            else:
                results = retrieve_by_keyword(q, chunk_records, top_k=self.top_k)
            all_results.extend(results)

        # 去重
        seen = set()
        deduped: list[RetrievedChunk] = []
        for r in sorted(all_results, key=lambda x: -x.score):
            if r.chunk_id not in seen:
                seen.add(r.chunk_id)
                deduped.append(r)

        # Step 3: 精排
        reranked = rerank_with_llm(question, deduped[:15], chat_fn, top_k=self.top_k)
        return reranked

    # ====== 问答 ======

    def answer(self, question: str) -> AgentAnswer:
        q = question.strip()
        if not q:
            raise ValueError("问题不能为空。")

        # 指代消解（增强版）
        resolved_doc_id = self.memory.resolve_reference(q, self.list_available_papers())
        if resolved_doc_id:
            self.memory.last_referenced_doc = resolved_doc_id

        kb = self.ensure_knowledge_base_ready()
        if not kb.chunk_records:
            self.memory.add_user(q)
            self.memory.add_assistant(build_empty_library_message())
            return AgentAnswer(question=q, model_answer=build_empty_library_message(),
                              sources=[], retrieved_count=0, used_prompt=build_answer_system_prompt())

        chunks = self.retrieve_chunks_for_question(q, kb.chunk_records)
        ctx_blocks = _build_context_blocks(chunks)
        chat_hist = self.memory.build_context_for_llm()
        prompt = build_answer_system_prompt() + "\n\n" + build_answer_task_prompt(q, ctx_blocks, chat_hist)

        if not chunks:
            msg = build_no_result_message(q)
            self.memory.add_user(q)
            self.memory.add_assistant(msg)
            return AgentAnswer(question=q, model_answer=msg, sources=[], retrieved_count=0, used_prompt=prompt)

        # 使用 LLM 生成回答
        if self.chat_client:
            try:
                sources_for_llm = _build_llm_context(chunks)
                llm_answer = self.chat_client.chat(
                    messages=[
                        {"role": "system", "content": build_answer_system_prompt()},
                        {"role": "user", "content": build_answer_task_prompt(q, sources_for_llm, chat_hist)},
                    ],
                    temperature=0.2, max_tokens=1200,
                )
            except Exception:
                llm_answer = _synthesize_answer(q, chunks)
        else:
            llm_answer = _synthesize_answer(q, chunks)

        # 后处理：验证回答中引用的 [来源N] 是否都有对应的原文依据
        llm_answer = _verify_and_enhance_citations(llm_answer, chunks)

        sources = _build_source_entries(chunks)

        # 记录对话（含丰富的论文上下文）
        top_chunk = chunks[0] if chunks else None
        doc_id = top_chunk.document_id if top_chunk else ""
        file_name = str(top_chunk.metadata.get("file_name", "")) if top_chunk else ""
        self.memory.add_user(q)
        self.memory.add_assistant(llm_answer, doc_id=doc_id, file_name=file_name)

        # 将论文详细信息写入记忆上下文
        if doc_id and file_name:
            title = str(top_chunk.metadata.get("title", file_name)) if top_chunk else file_name
            authors_raw = top_chunk.metadata.get("authors", []) if top_chunk else []
            authors = "、".join(authors_raw[:3]) if isinstance(authors_raw, list) else str(authors_raw)
            year = str(top_chunk.metadata.get("year", "")) if top_chunk else ""
            # 尝试从回答中提取研究方法关键词
            methods_hint = _extract_methods_from_answer(llm_answer)
            # 从问题中提取研究主题
            topic_hint = q[:80]
            self.memory.remember_paper_detail(
                doc_id=doc_id, file_name=file_name,
                title=title, authors=authors, year=year,
                methods=methods_hint, research_topic=topic_hint,
                key_topics=_extract_key_topics_from_chunks(chunks),
            )

        return AgentAnswer(question=q, model_answer=llm_answer,
                          sources=sources, retrieved_count=len(chunks), used_prompt=prompt)

    # ====== 向量索引 ======

    def prepare_embedding_index(self, *, client=None, model_name: str = "", dimensions: int = 0,
                                processed_data_dir: str | Path = "", build_if_missing: bool = False,
                                force_rebuild: bool = False) -> EmbeddingIndexResponse:
        kb = self.ensure_knowledge_base_ready()
        if not kb.chunk_records:
            raise ValueError("当前没有可用于构建向量索引的切块记录。")

        index_path = get_default_embedding_index_path(processed_data_dir, model_name, dimensions)
        self.embedding_client = client
        self.embedding_model_name = model_name
        self.embedding_dimensions = dimensions

        if not force_rebuild:
            cached = load_embedding_index(index_path)
            if cached is not None and is_embedding_index_compatible(cached, kb.chunk_records, model_name, dimensions):
                self.embedding_index = cached
                return EmbeddingIndexResponse(
                    status_message="已加载本地向量索引。", index_path=str(index_path),
                    vector_count=len(cached.vectors), loaded_from_cache=True,
                )

        if not build_if_missing:
            self.embedding_index = None
            return EmbeddingIndexResponse(
                status_message="当前未找到可用向量索引，可执行 build_index 构建。",
                index_path=str(index_path), vector_count=0, loaded_from_cache=False,
            )

        if client is None:
            raise ValueError("构建向量索引需要可用的 embedding 客户端。")

        idx = run_build_embedding_index(
            chunk_records=kb.chunk_records, client=client,
            model_name=model_name, dimensions=dimensions, index_path=index_path,
        )
        self.embedding_index = idx
        return EmbeddingIndexResponse(
            status_message="已完成本地向量索引构建。", index_path=str(index_path),
            vector_count=len(idx.vectors), loaded_from_cache=False,
        )

    # ====== 单篇分析 ======

    def analyze_paper(self, target: str | None = None) -> PaperAnalysisResponse:
        kb = self.ensure_knowledge_base_ready()
        if not kb.documents:
            return PaperAnalysisResponse(target=target or "默认", status_message="当前没有可分析的论文。",
                                         analysis=None, formatted_output="当前没有可分析的论文。")

        doc = self.find_document(target)
        if doc is None:
            return PaperAnalysisResponse(target=target or "默认", status_message="未找到匹配的论文。",
                                         analysis=None, formatted_output="未找到匹配的论文。")

        # 先规则分析为底
        analysis = analyze_single_paper(doc.full_text, file_name=doc.file_name, document_id=doc.document_id)

        # LLM 增强分析
        if self.analysis_client:
            try:
                enhanced = self._llm_enhanced_analysis(doc)
                if enhanced:
                    analysis = enhanced
            except Exception:
                pass

        # 将论文详细信息写入记忆上下文
        self.memory.remember_paper_detail(
            doc_id=doc.document_id, file_name=doc.file_name,
            title=doc.metadata.title or doc.file_name,
            authors="、".join(doc.metadata.authors[:3]) if doc.metadata.authors else "",
            year=doc.metadata.year or "",
            methods=analysis.methods if analysis.methods != "未识别" else "",
            research_topic=analysis.research_question if analysis.research_question != "未识别" else "",
            key_topics=_extract_keywords_simple(
                f"{analysis.research_question} {analysis.research_object} {analysis.methods}"
            ),
        )

        return PaperAnalysisResponse(
            target=target or doc.file_name,
            status_message=f"已完成对《{doc.file_name}》的结构化分析。",
            analysis=analysis, formatted_output=format_analysis_result(analysis),
        )

    def _llm_enhanced_analysis(self, doc: ParsedDocument) -> StructuredPaperAnalysis | None:
        if self.analysis_client is None:
            return None
        text_input = self._truncate_for_analysis(doc.full_text)
        try:
            raw = self.analysis_client.chat(
                messages=[
                    {"role": "system", "content": "你是城市研究论文分析专家。只基于给定内容提取，不要编造。"},
                    {"role": "user", "content": ENHANCED_ANALYSIS_PROMPT.format(text=text_input)},
                ],
                temperature=0.1, max_tokens=1500,
                response_format={"type": "json_object"},
            )
            data = json.loads(raw)
        except Exception:
            return None

        def _get(key: str, fallback: str = "") -> str:
            v = data.get(key, "")
            return str(v).strip() if isinstance(v, str) and str(v).strip() else fallback

        return StructuredPaperAnalysis(
            file_name=doc.file_name, document_id=doc.document_id,
            research_question=_get("research_question", "未识别"),
            research_object=_get("research_object", "未识别"),
            methods=_get("methods", "未识别"),
            data_source=_get("data_source", "未识别"),
            key_findings=_get("key_findings", "未识别"),
            limitations=_get("limitations", "未识别"),
            implications=_get("implications", "未识别"),
            evidence_map={
                "research_type": [str(data.get("research_type", ""))],
                "methodology_tags": data.get("methodology_tags", []),
                "quality_notes": [str(data.get("quality_notes", ""))],
            },
        )

    def _truncate_for_analysis(self, full_text: str, max_chars: int = 20000) -> str:
        text = full_text.strip()
        lowered = text.lower()
        for marker in ["\nreferences", "\nreference", "\nbibliography", "\nacknowledgments"]:
            idx = lowered.find(marker)
            if idx >= 0:
                text = text[:idx].strip()
                break
        if len(text) <= max_chars:
            return text
        return text[:int(max_chars * 0.6)] + "\n\n[...中间内容已省略...]\n\n" + text[-int(max_chars * 0.4):]

    # ====== 多篇比较 ======

    def compare_papers(self, targets: list[str] | None = None, topic_hint: str = "未指定") -> PaperComparisonResponse:
        kb = self.ensure_knowledge_base_ready()
        if len(kb.documents) < 2:
            return PaperComparisonResponse(targets=targets or [], status_message="论文数量不足 2 篇。",
                                           comparison=None, formatted_output="论文数量不足 2 篇。")
        docs = self.find_documents(targets, default_count=2)
        if len(docs) < 2:
            return PaperComparisonResponse(targets=targets or [], status_message="至少需要 2 篇论文。",
                                           comparison=None, formatted_output="至少需要 2 篇论文。")
        paper_inputs = [
            {"file_name": d.file_name, "document_id": d.document_id, "full_text": d.full_text}
            for d in docs
        ]

        # 基础规则比较（兜底）
        comparison = run_compare_papers(paper_inputs, topic_hint=topic_hint)
        formatted = format_comparison_result(comparison)

        # LLM 增强比较
        if self.analysis_client and len(docs) >= 2:
            try:
                llm_formatted = self._llm_compare_papers(docs, topic_hint)
                if llm_formatted:
                    formatted = llm_formatted
            except Exception:
                pass  # 回退到规则比较结果

        return PaperComparisonResponse(
            targets=[d.file_name for d in docs],
            status_message=f"已完成 {len(docs)} 篇论文的比较。",
            comparison=comparison, formatted_output=formatted,
        )

    def _llm_compare_papers(self, docs, topic_hint: str = "未指定") -> str | None:
        """使用 LLM 进行深度多维度论文比较。"""
        if self.analysis_client is None:
            return None

        texts_block = ""
        for i, d in enumerate(docs, 1):
            truncated = self._truncate_for_analysis(d.full_text, max_chars=8000)
            texts_block += (
                f"论文{i}：《{d.file_name}》\n"
                f"标题：{d.metadata.title or '未知'}\n"
                f"作者：{', '.join(d.metadata.authors) if d.metadata.authors else '未知'}\n"
                f"年份：{d.metadata.year or '未知'}\n"
                f"内容摘要：{truncated[:6000]}\n\n"
            )

        try:
            raw = self.analysis_client.chat(
                messages=[{
                    "role": "user",
                    "content": LLM_COMPARE_PROMPT.format(texts=texts_block),
                }],
                temperature=0.2, max_tokens=4000,
            )
            data = _parse_llm_json(raw)
        except Exception:
            return None

        # 格式化 LLM 比较结果
        lines = [
            "📊 多篇论文深度比较结果（LLM 增强分析）",
            f"比较主题：{topic_hint}",
            f"纳入论文：{len(docs)} 篇",
            "",
        ]
        for i, d in enumerate(docs, 1):
            lines.append(f"  [{i}] 《{d.file_name}》")
            if d.metadata.title:
                lines.append(f"      完整标题：{d.metadata.title}")
        lines.append("")

        # 各维度比较
        dimension_labels = {
            "research_questions": "🔬 研究问题异同",
            "theoretical_frameworks": "📖 理论基础对比",
            "methodology": "⚙️ 研究方法对比",
            "data_and_scale": "📊 数据来源与尺度对比",
            "key_findings": "💡 核心发现共识与分歧",
            "contributions": "🏆 理论/实践贡献对比",
            "limitations": "⚠️ 局限性对比",
            "complementarity": "🔗 论文间互补性",
            "synthesis": "📝 综合评述",
        }

        for key, label in dimension_labels.items():
            val = data.get(key, "")
            if val and str(val).strip():
                lines.append(f"{label}：")
                if isinstance(val, list):
                    for item in val:
                        lines.append(f"  • {_flatten_llm_value(item)}")
                elif isinstance(val, dict):
                    # 如果 LLM 返回了嵌套对象，将其扁平化为可读文本
                    for sub_key, sub_val in val.items():
                        if sub_val and str(sub_val).strip():
                            lines.append(f"  {_flatten_llm_value(sub_val)}")
                else:
                    lines.append(f"  {_flatten_llm_value(val)}")
                lines.append("")

        return "\n".join(lines)

    # ====== 综述提纲 ======

    def _llm_generate_outline(self, docs, topic: str) -> str | None:
        """使用 LLM 生成结构化综述提纲。"""
        if self.analysis_client is None:
            return None

        # 构建论文分析文本块
        analyses_block = ""
        for i, d in enumerate(docs, 1):
            truncated = self._truncate_for_analysis(d.full_text, max_chars=6000)
            analyses_block += (
                f"论文{i}：《{d.file_name}》\n"
                f"标题：{d.metadata.title or '未知'}\n"
                f"作者：{', '.join(d.metadata.authors) if d.metadata.authors else '未知'}\n"
                f"年份：{d.metadata.year or '未知'}\n"
                f"内容摘要：{truncated[:4000]}\n\n"
            )

        try:
            raw = self.analysis_client.chat(
                messages=[{
                    "role": "user",
                    "content": LLM_OUTLINE_PROMPT.format(analyses=analyses_block),
                }],
                temperature=0.3, max_tokens=4000,
            )
            data = _parse_llm_json(raw)
        except Exception:
            return None

        # 格式化 LLM 提纲
        lines = [
            "综述提纲（LLM 增强生成）",
            f"主题：{topic}",
            f"参考论文：{len(docs)} 篇",
            "",
        ]

        sections = data.get("sections", [])
        if not sections:
            return None

        for idx, section in enumerate(sections, 1):
            title = section.get("title", f"第{idx}章")
            bullets = section.get("bullets", [])
            lines.append(f"{title}")
            for bullet in bullets:
                if bullet and str(bullet).strip():
                    lines.append(f"- {str(bullet).strip()}")
            lines.append("")

        return "\n".join(lines)

    def generate_review_outline(self, topic: str, targets: list[str] | None = None) -> ReviewOutlineResponse:
        ct = topic.strip()
        if not ct:
            raise ValueError("综述主题不能为空。")
        kb = self.ensure_knowledge_base_ready()
        if len(kb.documents) < 2:
            return ReviewOutlineResponse(topic=ct, status_message="论文数量不足。",
                                         outline=None, formatted_output="论文数量不足。")
        docs = self.find_documents(targets, default_count=3)
        if len(docs) < 2:
            return ReviewOutlineResponse(topic=ct, status_message="至少需要 2 篇论文。",
                                         outline=None, formatted_output="至少需要 2 篇论文。")
        paper_inputs = [
            {"file_name": d.file_name, "document_id": d.document_id, "full_text": d.full_text}
            for d in docs
        ]
        # 规则引擎生成（兜底）
        outline = generate_review_outline(ct, paper_inputs)
        formatted = format_review_outline(outline)

        # LLM 增强提纲
        if self.analysis_client and len(docs) >= 2:
            try:
                llm_formatted = self._llm_generate_outline(docs, ct)
                if llm_formatted:
                    formatted = llm_formatted
            except Exception:
                pass  # 回退到规则提纲

        return ReviewOutlineResponse(
            topic=ct, status_message=f"已基于 {len(docs)} 篇论文生成综述提纲。",
            outline=outline, formatted_output=formatted,
        )

    # ====== 多步工作流 ======

    def run_review_workflow(self, topic: str, targets: list[str] | None = None,
                            output_dir: str | Path | None = None) -> WorkflowResponse:
        ct = topic.strip()
        if not ct:
            raise ValueError("工作流主题不能为空。")
        kb = self.ensure_knowledge_base_ready()
        if len(kb.documents) < 2:
            return WorkflowResponse(topic=ct, status_message="论文不足 2 篇。",
                                    workflow_result=None, formatted_output="论文不足 2 篇。")
        docs = self.find_documents(targets, default_count=8)
        if len(docs) < 2:
            return WorkflowResponse(topic=ct, status_message="至少需要 2 篇论文。",
                                    workflow_result=None, formatted_output="至少需要 2 篇论文。")

        plan = build_default_workflow_plan(topic=ct, targets=targets or [])
        state = WorkflowState(topic=ct, targets=targets or [])
        mark_step_status(plan, "select_papers", "completed")
        state.selected_papers = [d.file_name for d in docs]
        state.step_logs.append(f"步骤 1：已选中 {len(docs)} 篇论文。")

        cr = self.compare_papers(targets=[d.document_id for d in docs], topic_hint=ct)
        mark_step_status(plan, "compare_papers", "completed")
        state.comparison_text = cr.formatted_output
        state.step_logs.append("步骤 2：已完成多篇论文比较。")

        or_ = self.generate_review_outline(topic=ct, targets=[d.document_id for d in docs])
        mark_step_status(plan, "generate_outline", "completed")
        state.outline_text = or_.formatted_output
        state.step_logs.append("步骤 3：已生成综述提纲。")

        od = str(Path(output_dir).expanduser().resolve()) if output_dir else str(OUTPUT_DIR)
        artifact = export_workflow_result(
            output_dir=od, topic=ct, selected_papers=state.selected_papers,
            comparison_text=state.comparison_text, outline_text=state.outline_text,
            step_logs=state.step_logs,
        )
        mark_step_status(plan, "export_markdown", "completed")
        state.export_artifact = artifact
        state.step_logs.append(f"步骤 4：已导出 Markdown 报告到 {artifact.output_path}")

        wr = WorkflowRunResult(
            status_message=f"已完成主题「{ct}」的多步工作流。",
            plan=plan, state=state, formatted_output="",
        )
        wr.formatted_output = format_workflow_run_result(wr)
        return WorkflowResponse(topic=ct, status_message=wr.status_message,
                                workflow_result=wr, formatted_output=wr.formatted_output)

    # ====== 新增科研工具 ======

    def summarize_paper(self, target: str | None = None) -> SummarizeResponse:
        """使用 LLM 生成论文结构化摘要。"""
        kb = self.ensure_knowledge_base_ready()
        if not kb.documents:
            return SummarizeResponse(file_name="", formatted_output="没有可用的论文。")
        doc = self.find_document(target)
        if doc is None:
            return SummarizeResponse(file_name="", formatted_output="未找到匹配的论文。")

        if self.chat_client:
            summary = generate_summary(doc.full_text, doc.file_name, self._get_chat_fn())
        else:
            from tools.summarize_tool import PaperSummary
            summary = PaperSummary(file_name=doc.file_name, background="LLM 未启用，无法生成摘要。")

        # 将论文详细信息写入记忆上下文
        self.memory.remember_paper_detail(
            doc_id=doc.document_id, file_name=doc.file_name,
            title=doc.metadata.title or doc.file_name,
            authors="、".join(doc.metadata.authors[:3]) if doc.metadata.authors else "",
            year=doc.metadata.year or "",
            methods=getattr(summary, 'methods', '') or '',
            research_topic=getattr(summary, 'background', '')[:100] or '',
            key_topics=getattr(summary, 'keywords', []) or [],
        )

        return SummarizeResponse(file_name=doc.file_name, formatted_output=format_summary(summary))

    def translate_text(self, text: str, direction: str = "auto") -> TranslateResponse:
        """中英学术文本互译。"""
        result = translate(text, self._get_chat_fn(), direction)
        return TranslateResponse(original=text, translated=result, direction=direction)

    def recommend_papers(self, target: str | None = None) -> RecommendResponse:
        """基于 embedding 相似度推荐相关论文。"""
        kb = self.ensure_knowledge_base_ready()
        doc = self.find_document(target)
        if doc is None:
            return RecommendResponse(target=target or "", recommendations=[], formatted_output="未找到目标论文。")

        emb_vectors = self.embedding_index.vectors if self.embedding_index else None
        recs = recommend_similar_papers(doc.document_id, kb.chunk_records, emb_vectors, top_n=5)
        self.memory.remember_paper(doc.document_id, doc.file_name)
        return RecommendResponse(
            target=doc.file_name, recommendations=recs,
            formatted_output=format_recommendations(recs),
        )

    def export_bibtex(self, target: str | None = None) -> str:
        """导出单篇或全部论文的 BibTeX。"""
        kb = self.ensure_knowledge_base_ready()
        if target:
            doc = self.find_document(target)
            if doc is None:
                return "未找到目标论文。"
            return export_single_bibtex(doc)
        return export_all_bibtex(kb.documents)

    def identify_research_gaps(self, targets: list[str] | None = None) -> str:
        """基于多篇论文分析识别研究空白。"""
        kb = self.ensure_knowledge_base_ready()
        docs = self.find_documents(targets, default_count=3)
        if len(docs) < 2:
            return "研究空白分析至少需要 2 篇论文。"

        analyses_text = ""
        for d in docs:
            analysis = analyze_single_paper(d.full_text, file_name=d.file_name, document_id=d.document_id)
            analyses_text += (
                f"《{d.file_name}》：\n"
                f"  研究问题：{analysis.research_question}\n"
                f"  方法：{analysis.methods}\n"
                f"  局限性：{analysis.limitations}\n"
                f"  启示：{analysis.implications}\n\n"
            )

        if self.chat_client:
            try:
                raw = self.chat_client.chat(
                    messages=[{"role": "user", "content": RESEARCH_GAP_PROMPT.format(analyses=analyses_text)}],
                    temperature=0.3, max_tokens=1000,
                    response_format={"type": "json_object"},
                )
                data = json.loads(raw)
                lines = [
                    "研究空白与未来方向分析：",
                    f"已有共识：{data.get('common_approaches', '')}",
                    "",
                    "研究空白：",
                ]
                for g in data.get("gaps", []):
                    lines.append(f"- {g}")
                lines.append("")
                lines.append("未来方向：")
                for d_ in data.get("future_directions", []):
                    lines.append(f"- {d_}")
                lines.append(f"\n综合洞见：{data.get('cross_cutting_insight', '')}")
                return "\n".join(lines)
            except Exception:
                pass

        lines = ["研究空白分析（规则模式，建议启用 LLM 获得更深入分析）：", ""]
        for d in docs:
            lines.append(f"《{d.file_name}》局限：{analyze_single_paper(d.full_text).limitations}")
        return "\n".join(lines)

    def chat(self, message: str) -> str:
        """快捷对话接口：一句话输入 → 回答文本。"""
        result = self.answer(message)
        return result.model_answer

    # ====== GraphRAG 图增强检索 ======

    def build_entity_graph(self) -> str:
        """构建论文实体共现图（关键词/方法/城市/技术）。"""
        kb = self.ensure_knowledge_base_ready()
        if not kb.chunk_records:
            return "知识库为空，无法构建实体图。"
        self.entity_graph = build_entity_graph(kb.chunk_records)
        return get_graph_stats(self.entity_graph)

    def graph_search(self, question: str) -> AgentAnswer:
        """图增强检索问答：基础检索 + 实体图邻居游走 → 增强上下文 → LLM 回答。

        Args:
            question: 用户问题
        Returns:
            增强后的问答结果
        """
        q = question.strip()
        if not q:
            raise ValueError("问题不能为空。")

        kb = self.ensure_knowledge_base_ready()
        if not kb.chunk_records:
            return AgentAnswer(question=q, model_answer=build_empty_library_message(),
                              sources=[], retrieved_count=0, used_prompt="")

        # Step 1: 基础检索
        base_chunks = self.retrieve_chunks_for_question(q, kb.chunk_records)

        # Step 2: 图增强（如果图已构建）
        graph_ctx = ""
        if self.entity_graph:
            base_indices = {
                i for i, rec in enumerate(kb.chunk_records)
                if rec.get("chunk_id") in {c.chunk_id for c in base_chunks}
            }
            graph_indices = graph_enhanced_retrieval(
                q, self.entity_graph, kb.chunk_records, base_indices, top_k=self.top_k,
            )
            graph_ctx = format_graph_context(q, self.entity_graph)

            # 用图索引补充 chunk
            all_chunks = list(base_chunks)
            existing_ids = {c.chunk_id for c in all_chunks}
            for idx in graph_indices:
                if len(all_chunks) >= self.top_k:
                    break
                if idx < len(kb.chunk_records):
                    rec = kb.chunk_records[idx]
                    if rec.get("chunk_id") not in existing_ids:
                        meta = rec.get("metadata", {}) if isinstance(rec.get("metadata"), dict) else {}
                        from rag.retriever import RetrievedChunk as RC
                        all_chunks.append(RC(
                            chunk_id=str(rec.get("chunk_id", "")),
                            document_id=str(rec.get("document_id", "")),
                            text=str(rec.get("text", "")),
                            snippet=str(rec.get("text", ""))[:200],
                            score=0.5, matched_terms=["graph"], metadata=meta,
                        ))

            chunks = all_chunks[:self.top_k]
        else:
            chunks = base_chunks

        # Step 3: LLM 回答（附图谱上下文）
        ctx_blocks = _build_llm_context(chunks)
        if graph_ctx:
            ctx_blocks.insert(0, graph_ctx)

        chat_hist = self.memory.build_context_for_llm()
        if self.chat_client and chunks:
            try:
                llm_answer = self.chat_client.chat(
                    messages=[
                        {"role": "system", "content": build_answer_system_prompt()},
                        {"role": "user", "content": build_answer_task_prompt(q, ctx_blocks, chat_hist)},
                    ],
                    temperature=0.2, max_tokens=1200,
                )
            except Exception:
                llm_answer = _synthesize_answer(q, chunks)
        else:
            llm_answer = _synthesize_answer(q, chunks)

        sources = _build_source_entries(chunks)
        self.memory.add_user(q)
        self.memory.add_assistant(llm_answer)

        # 将论文详细信息写入记忆上下文
        if chunks:
            top_c = chunks[0]
            doc_id = top_c.document_id
            file_name = str(top_c.metadata.get("file_name", ""))
            if doc_id and file_name:
                title = str(top_c.metadata.get("title", file_name))
                authors_raw = top_c.metadata.get("authors", [])
                authors = "、".join(authors_raw[:3]) if isinstance(authors_raw, list) else str(authors_raw)
                year = str(top_c.metadata.get("year", ""))
                self.memory.remember_paper_detail(
                    doc_id=doc_id, file_name=file_name,
                    title=title, authors=authors, year=year,
                    methods="", research_topic=q[:80],
                    key_topics=[],
                )

        return AgentAnswer(question=q, model_answer=llm_answer,
                          sources=sources, retrieved_count=len(chunks),
                          used_prompt=build_answer_system_prompt())

    def export_graph_visualization(self, fmt: str = "png", output_path=None) -> str:
        """导出实体共现图的可视化表示。

        Args:
            fmt: 输出格式，"png"（直接生成图片）、"mermaid"、"dot"
            output_path: PNG 文件路径（fmt="png" 时必须提供）
        Returns:
            文件路径（png）或可视化代码文本（mermaid/dot）
        """
        from rag.graphrag import export_graph_mermaid, export_graph_dot

        if self.entity_graph is None:
            return "⚠️ 图谱尚未构建，请先执行 graph build。"

        if fmt == "png":
            if output_path is None:
                return "⚠️ PNG 导出需要提供 output_path。"
            return export_graph_png(self.entity_graph, output_path) or "⚠️ 缺少 matplotlib/networkx，请执行 pip install matplotlib networkx"
        if fmt == "dot":
            return export_graph_dot(self.entity_graph)
        return export_graph_mermaid(self.entity_graph)


# ====== 辅助函数 ======

def _build_context_blocks(chunks: list[RetrievedChunk]) -> list[str]:
    blocks: list[str] = []
    for i, c in enumerate(chunks, 1):
        pn = c.metadata.get("page_numbers", [])
        pt = "、".join(str(n) for n in pn) if pn else "?"
        fn = str(c.metadata.get("file_name", "?"))
        blocks.append(f"[来源{i}] {fn} | 页码:{pt} | {c.snippet}")
    return blocks


def _build_llm_context(chunks: list[RetrievedChunk]) -> list[str]:
    blocks: list[str] = []
    for i, c in enumerate(chunks, 1):
        pn = c.metadata.get("page_numbers", [])
        pt = "、".join(str(n) for n in pn) if pn else "?"
        fn = str(c.metadata.get("file_name", "?"))
        sec = str(c.metadata.get("section", "正文"))
        title = str(c.metadata.get("title", fn))
        authors = "、".join(c.metadata.get("authors", []))
        year = str(c.metadata.get("year", ""))
        header = f"[来源{i}] 《{title}》"
        if authors:
            header += f" | 作者: {authors}"
        if year:
            header += f" | {year}"
        header += f" | 页码:{pt} | 章节:{sec}"
        blocks.append(f"{header}\n片段：{c.text[:800]}")
    return blocks


def _build_source_entries(chunks: list[RetrievedChunk]) -> list[dict]:
    sources: list[dict] = []
    for c in chunks[:5]:
        sources.append({
            "file_name": str(c.metadata.get("file_name", "?")),
            "source_path": str(c.metadata.get("source_path", "")),
            "page_numbers": c.metadata.get("page_numbers", []),
            "chunk_id": c.chunk_id,
            "score": round(c.score, 2),
            "snippet": c.snippet,
            "matched_terms": c.matched_terms,
            "section": str(c.metadata.get("section", "")),
            "title": str(c.metadata.get("title", "")),
            "authors": c.metadata.get("authors", []),
            "year": str(c.metadata.get("year", "")),
        })
    return sources


def _synthesize_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    lines = ["根据当前召回的论文片段，整理如下："]
    for i, c in enumerate(chunks[:3], 1):
        lines.append(f"{i}. {c.snippet}")
    lines.append(build_answer_suffix())
    return "\n".join(lines)


def _verify_and_enhance_citations(answer: str, chunks: list[RetrievedChunk]) -> str:
    """后处理：验证 LLM 回答中的 [来源N] 引用是否都有实际来源支撑。

    如果检测到引用了不存在的来源编号，会在答案末尾追加提示。
    同时尝试为没有提供原文片段的论断补充依据提示。
    """
    import re as _re
    cited = set()
    for m in _re.finditer(r"\[来源(\d+)\]", answer):
        cited.add(int(m.group(1)))

    missing = [n for n in cited if n > len(chunks)]
    if missing:
        answer += (
            f"\n\n⚠️ 注意：以下来源编号在当前检索结果中无对应片段："
            + "、".join(f"[来源{n}]" for n in sorted(missing))
            + "。可能是模型生成错误，请结合下方「来源依据」核验。"
        )

    # 检查是否有「依据片段」小节，如无则追加来源提示
    if "依据片段" not in answer and chunks:
        answer += "\n\n📋 依据片段：\n"
        for i, c in enumerate(chunks[:5], 1):
            answer += f"  [来源{i}] {c.snippet[:200]}\n"

    return answer


def _extract_methods_from_answer(answer: str) -> str:
    """从 LLM 回答中尝试提取研究方法关键词。"""
    import re as _re
    method_patterns = [
        r'(?:方法[：:])[^\n]{0,100}',
        r'(?:采用|使用|运用)[^\n]{0,80}(?:方法|分析|模型|回归|调查|实验)',
    ]
    for pat in method_patterns:
        m = _re.search(pat, answer)
        if m:
            return m.group(0).strip()[:120]
    return ""


def _extract_key_topics_from_chunks(chunks: list[RetrievedChunk]) -> list[str]:
    """从检索到的 chunk 中提取关键主题词。"""
    topics: list[str] = []
    seen: set[str] = set()
    for c in chunks[:3]:
        for term in c.matched_terms:
            if len(term) >= 3 and term not in seen:
                seen.add(term)
                topics.append(term)
    return topics[:10]


def _extract_keywords_simple(text: str) -> list[str]:
    """简单提取中英文关键词。"""
    import re as _re
    # 中文词（2-4字）
    cn = _re.findall(r"[一-鿿]{2,4}", text)
    # 英文词（3+字母）
    en = _re.findall(r"[a-zA-Z]{3,}", text)
    stop = {"研究", "分析", "本文", "论文", "基于", "通过", "进行", "以及", "对于", "不同",
            "the", "and", "for", "with", "that", "this", "from", "into", "what", "which"}
    seen: set[str] = set()
    result: list[str] = []
    for w in cn + en:
        wl = w.lower()
        if wl not in stop and wl not in seen:
            seen.add(wl)
            result.append(w)
    return result[:12]
