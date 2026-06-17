"""本模块作用：多策略检索 + LLM Reranker + 查询扩展，支撑高质量论文问答。"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


COMMON_STOP_WORDS = {
    "的", "了", "和", "是", "在", "与", "及", "对", "中", "及其",
    "一个", "一种", "如何", "什么", "哪些", "这个", "那个", "我们", "你们",
    "the", "and", "for", "with", "that", "this", "from", "into", "what", "which",
}


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    text: str
    snippet: str
    score: float
    matched_terms: list[str]
    metadata: dict


# ====== 关键词提取与检索 ======

def normalize_search_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_search_terms(text: str) -> list[str]:
    """从问题中提取中英文检索词（含 bigram/trigram）。"""
    nt = normalize_search_text(text)
    en = re.findall(r"[a-z0-9]{2,}", nt)
    cn = re.findall(r"[一-鿿]{2,}", text)
    terms: list[str] = [t for t in en if t not in COMMON_STOP_WORDS]
    for phrase in cn:
        if phrase not in COMMON_STOP_WORDS:
            terms.append(phrase)
        for i in range(len(phrase) - 1):
            bg = phrase[i:i + 2]
            if bg not in COMMON_STOP_WORDS and len(bg) >= 2:
                terms.append(bg)
        for i in range(len(phrase) - 2):
            tg = phrase[i:i + 3]
            if tg not in COMMON_STOP_WORDS:
                terms.append(tg)
    # 去重保序
    seen = set()
    out = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _calc_keyword_score(question: str, chunk_record: dict) -> tuple[float, list[str]]:
    qt = normalize_search_text(question)
    terms = extract_search_terms(question)
    ct = normalize_search_text(str(chunk_record.get("text", "")))
    meta = chunk_record.get("metadata", {}) if isinstance(chunk_record.get("metadata"), dict) else {}
    fn = normalize_search_text(str(meta.get("file_name", "")))
    title = normalize_search_text(str(meta.get("title", "")))

    matched: list[str] = []
    score = 0.0
    if qt and qt in ct:
        score += 8.0
    for term in terms:
        tc = ct.count(term) + fn.count(term) + title.count(term)
        if tc <= 0:
            continue
        matched.append(term)
        w = 2.4 if len(term) >= 6 else (1.8 if len(term) >= 4 else 1.1)
        score += min(tc, 3) * w
    if terms:
        score += (len(matched) / len(terms)) * 3.0
    return score, matched


def retrieve_by_keyword(question: str, chunk_records: list[dict], top_k: int = 5, min_score: float = 1.0) -> list[RetrievedChunk]:
    """纯关键词检索。"""
    results: list[RetrievedChunk] = []
    for rec in chunk_records:
        sc, mt = _calc_keyword_score(question, rec)
        if sc < min_score:
            continue
        meta = rec.get("metadata", {}) if isinstance(rec.get("metadata"), dict) else {}
        ct = str(rec.get("text", ""))
        results.append(RetrievedChunk(
            chunk_id=str(rec.get("chunk_id", "")),
            document_id=str(rec.get("document_id", "")),
            text=ct,
            snippet=_build_snippet(ct, mt),
            score=sc, matched_terms=mt, metadata=meta,
        ))
    results.sort(key=lambda x: (-x.score, str(x.metadata.get("file_name", ""))))
    return results[:top_k]


def _build_snippet(text: str, matched_terms: list[str], max_len: int = 200) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    if len(t) <= max_len:
        return t
    for term in matched_terms:
        pos = t.lower().find(term.lower())
        if pos >= 0:
            s = max(pos - 50, 0)
            e = min(s + max_len, len(t))
            snip = t[s:e].strip()
            return ("..." if s > 0 else "") + snip + ("..." if e < len(t) else "")
    return t[:max_len].rstrip() + "..."


# ====== 向量检索 ======

def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def retrieve_hybrid(
    question: str,
    chunk_records: list[dict],
    embedding_vectors: dict[str, list[float]],
    query_vector: list[float],
    top_k: int = 5,
    lexical_weight: float = 0.35,
    semantic_weight: float = 0.65,
) -> list[RetrievedChunk]:
    """混合检索：关键词 + 向量语义。"""
    kw_results = retrieve_by_keyword(question, chunk_records, top_k=max(top_k * 4, 15), min_score=0.0)
    kw_map = {r.chunk_id: (r.score, r.matched_terms) for r in kw_results}

    results: list[RetrievedChunk] = []
    for rec in chunk_records:
        cid = str(rec.get("chunk_id", ""))
        cv = embedding_vectors.get(cid, [])
        sem = cosine_similarity(query_vector, cv)
        kw_sc, kw_mt = kw_map.get(cid, (0.0, []))
        nkw = min(kw_sc / 20.0, 1.0)
        hybrid = nkw * lexical_weight + sem * semantic_weight
        if sem < 0.08 and kw_sc <= 0:
            continue
        meta = rec.get("metadata", {}) if isinstance(rec.get("metadata"), dict) else {}
        ct = str(rec.get("text", ""))
        results.append(RetrievedChunk(
            chunk_id=cid, document_id=str(rec.get("document_id", "")),
            text=ct, snippet=_build_snippet(ct, kw_mt),
            score=hybrid, matched_terms=kw_mt, metadata=meta,
        ))
    results.sort(key=lambda x: (-x.score, str(x.metadata.get("file_name", ""))))
    return results[:top_k]


# ====== 查询扩展 ======

_QUERY_EXPAND_PROMPT = """将以下学术问题扩展为 3 条检索查询（中英文均可），以提升论文检索覆盖率。
每条查询应是关键词组合或短语，用换行分隔。只输出查询文本，不要编号。

原始问题：{question}

扩展查询："""


def expand_query(question: str, chat_fn=None) -> list[str]:
    """使用 LLM 将用户问题扩展为多条检索查询。"""
    if chat_fn is None:
        return [question]
    try:
        raw = chat_fn(
            messages=[{"role": "user", "content": _QUERY_EXPAND_PROMPT.format(question=question)}],
            temperature=0.3, max_tokens=200,
        )
        queries = [q.strip() for q in raw.strip().split("\n") if q.strip()]
        return queries[:3] if queries else [question]
    except Exception:
        return [question]


# ====== LLM Reranker ======

_RERANK_PROMPT = """评估以下论文片段对用户问题的相关性，仅输出 0-10 的分数（数字）。

用户问题：{question}

论文片段（来源：{source}）：
{text}

相关性分数（0=完全不相关, 10=高度相关）："""


def rerank_with_llm(question: str, chunks: list[RetrievedChunk], chat_fn=None, top_k: int = 5) -> list[RetrievedChunk]:
    """使用 LLM 做精排：对粗召回结果逐条打分后重新排序。"""
    if chat_fn is None or len(chunks) <= top_k:
        return chunks[:top_k]

    for ch in chunks:
        fn = str(ch.metadata.get("file_name", ""))
        try:
            raw = chat_fn(
                messages=[{"role": "user", "content": _RERANK_PROMPT.format(
                    question=question, source=fn, text=ch.text[:1200],
                )}],
                temperature=0.1, max_tokens=10,
            )
            num = re.search(r"(\d+(?:\.\d+)?)", raw)
            if num:
                ch.score = float(num.group(1)) / 10.0
        except Exception:
            pass

    chunks.sort(key=lambda x: -x.score)
    return chunks[:top_k]


# ====== 元数据过滤 ======

def filter_by_metadata(chunks: list[RetrievedChunk], *, year_from: str = "", year_to: str = "", journal: str = "") -> list[RetrievedChunk]:
    """按元数据字段过滤检索结果。"""
    out = chunks
    if year_from:
        out = [c for c in out if str(c.metadata.get("year", "")) >= year_from]
    if year_to:
        out = [c for c in out if str(c.metadata.get("year", "")) <= year_to]
    if journal:
        jl = journal.lower()
        out = [c for c in out if jl in str(c.metadata.get("journal", "")).lower()]
    return out
