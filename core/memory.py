"""本模块作用：多轮对话记忆管理，支持上下文追踪、论文指代消解、主题关联与会话摘要。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ConversationTurn:
    role: str       # "user" | "assistant"
    content: str
    referenced_doc_id: str = ""   # 本轮引用的论文 ID
    referenced_file: str = ""     # 本轮引用的论文文件名


@dataclass
class PaperContext:
    """当前会话中讨论过的某篇论文的上下文记忆。"""
    doc_id: str
    file_name: str
    title: str = ""
    authors: str = ""
    year: str = ""
    methods: str = ""
    research_topic: str = ""       # 研究主题（从分析/问答中提取）
    key_topics: list[str] = field(default_factory=list)  # 关键词题列表
    last_discussed_turn: int = 0   # 最后讨论该论文的对话轮次


@dataclass
class ConversationMemory:
    """管理单次会话的多轮对话上下文，支持论文级上下文追踪。"""
    history: list[ConversationTurn] = field(default_factory=list)
    last_referenced_doc: str = ""      # 最近引用的论文 document_id
    last_referenced_file: str = ""     # 最近引用的论文文件名
    pinned_docs: list[str] = field(default_factory=list)  # 用户当前锁定的论文 ID 列表
    paper_contexts: dict[str, PaperContext] = field(default_factory=dict)  # doc_id → PaperContext

    def add_user(self, content: str) -> None:
        self.history.append(ConversationTurn(role="user", content=content))

    def add_assistant(self, content: str, doc_id: str = "", file_name: str = "") -> None:
        if doc_id:
            self.last_referenced_doc = doc_id
        if file_name:
            self.last_referenced_file = file_name
        self.history.append(ConversationTurn(
            role="assistant", content=content,
            referenced_doc_id=doc_id, referenced_file=file_name,
        ))

    def resolve_reference(self, text: str, available_docs: list[dict]) -> str:
        """解析用户输入中的指代（"这篇""第一篇""刚才那篇"等），返回对应的 document_id。

        增强策略：
        1. 数字指代（"第1篇"）
        2. 指示词指代（"这篇""那篇"）
        3. 论文标题/关键词匹配（如用户提到了上篇论文的研究方法或主题）
        4. 对话历史回溯

        Args:
            text: 用户输入
            available_docs: [{"document_id": ..., "file_name": ..., "index": ...}, ...]
        Returns:
            解析到的 document_id，未命中返回空字符串
        """

        # 1. 数字指代："第1篇""第一篇""论文1"
        m = re.search(r"第\s*(\d+)\s*篇|论文\s*(\d+)", text)
        if m:
            idx = int(m.group(1) or m.group(2)) - 1
            if 0 <= idx < len(available_docs):
                return str(available_docs[idx].get("document_id", ""))

        # 2. 指示词指代："这篇""刚才""上一篇""那篇""该论文""这篇文章"
        if any(w in text for w in ["这篇", "刚才", "上一篇", "那篇", "这个", "该论文", "这篇文章", "这篇论文",
                                     "这个研究", "上述论文", "前面那篇", "刚刚那篇"]):
            if self.last_referenced_doc:
                return self.last_referenced_doc

        # 3. 主题匹配：用户可能提到了上一篇文章的关键主题词
        if self.last_referenced_doc and self.last_referenced_doc in self.paper_contexts:
            ctx = self.paper_contexts[self.last_referenced_doc]
            match_topics = ctx.key_topics + ([ctx.research_topic] if ctx.research_topic else [])
            for topic in match_topics:
                if topic and len(topic) >= 3 and topic.lower() in text.lower():
                    return self.last_referenced_doc

        # 4. 标题关键词匹配
        if self.last_referenced_doc and self.last_referenced_doc in self.paper_contexts:
            ctx = self.paper_contexts[self.last_referenced_doc]
            if ctx.title and len(ctx.title) >= 6:
                title_words = re.findall(r"[一-鿿\w]{3,}", ctx.title)
                match_count = sum(1 for w in title_words if w.lower() in text.lower())
                if match_count >= 2:  # 至少2个标题词匹配
                    return self.last_referenced_doc

        # 5. 上一轮助手的回复中引用了哪篇
        for turn in reversed(self.history):
            if turn.role == "assistant" and turn.referenced_doc_id:
                return turn.referenced_doc_id

        return ""

    def build_context_for_llm(self, max_turns: int = 8) -> str:
        """将最近 N 轮对话 + 当前论文上下文构建为 LLM 可用的上下文字符串。"""
        recent = self.history[-max_turns * 2:]  # user + assistant 各算一轮
        parts: list[str] = []

        # 当前论文上下文
        paper_ctx = self._build_paper_context_block()
        if paper_ctx:
            parts.append(paper_ctx)

        if recent:
            lines = ["[对话历史]"]
            for t in recent:
                role = "用户" if t.role == "user" else "助手"
                lines.append(f"{role}：{t.content[:400]}")
            parts.append("\n".join(lines))

        return "\n\n".join(parts)

    def _build_paper_context_block(self) -> str:
        """构建当前正在讨论的论文的上下文块。"""
        if not self.last_referenced_doc or self.last_referenced_doc not in self.paper_contexts:
            # 回退到基本文件名信息
            if self.last_referenced_file:
                return (
                    f"[当前论文上下文]\n"
                    f"最近讨论的论文：{self.last_referenced_file}\n"
                    f"如果用户的问题涉及「这篇文章」「该论文」等指代，请理解为指上述论文。\n"
                )
            return ""

        ctx = self.paper_contexts[self.last_referenced_doc]
        lines = ["[当前论文上下文 —— 最近讨论的论文]"]
        lines.append(f"标题：{ctx.title or ctx.file_name}")
        if ctx.authors:
            lines.append(f"作者：{ctx.authors}")
        if ctx.year:
            lines.append(f"年份：{ctx.year}")
        if ctx.methods:
            lines.append(f"研究方法：{ctx.methods}")
        if ctx.research_topic:
            lines.append(f"研究主题：{ctx.research_topic}")
        if ctx.key_topics:
            lines.append(f"关键主题词：{'、'.join(ctx.key_topics[:8])}")
        lines.append("如果用户的问题涉及「这篇文章」「该论文」「刚才那篇」等指代，请理解为指上述论文。")
        lines.append("如果用户问「用了什么研究方法」「数据来源是什么」等，优先基于上述论文信息回答。\n")
        return "\n".join(lines)

    def remember_paper(self, doc_id: str, file_name: str) -> None:
        """记住当前操作的论文（基本信息版）。"""
        self.last_referenced_doc = doc_id
        self.last_referenced_file = file_name
        if doc_id not in self.pinned_docs:
            self.pinned_docs.append(doc_id)
        # 确保有基本上下文记录
        if doc_id not in self.paper_contexts:
            self.paper_contexts[doc_id] = PaperContext(
                doc_id=doc_id, file_name=file_name,
                last_discussed_turn=self.turn_count,
            )

    def remember_paper_detail(
        self, doc_id: str, file_name: str,
        title: str = "", authors: str = "", year: str = "",
        methods: str = "", research_topic: str = "",
        key_topics: list[str] | None = None,
    ) -> None:
        """记住当前操作的论文（详细信息版），用于增强后续对话的指代消解与上下文感知。

        在完成 analyze/summarize/answer 操作后调用，将论文的关键信息写入记忆。
        """
        self.last_referenced_doc = doc_id
        self.last_referenced_file = file_name
        if doc_id not in self.pinned_docs:
            self.pinned_docs.append(doc_id)

        ctx = self.paper_contexts.get(doc_id, PaperContext(doc_id=doc_id, file_name=file_name))
        ctx.file_name = file_name
        if title:
            ctx.title = title
        if authors:
            ctx.authors = authors
        if year:
            ctx.year = year
        if methods:
            ctx.methods = methods
        if research_topic:
            ctx.research_topic = research_topic
        if key_topics:
            ctx.key_topics = list(set(ctx.key_topics + key_topics))
        ctx.last_discussed_turn = self.turn_count
        self.paper_contexts[doc_id] = ctx

    def get_last_paper_context(self) -> PaperContext | None:
        """获取最近讨论的论文上下文。"""
        if self.last_referenced_doc and self.last_referenced_doc in self.paper_contexts:
            return self.paper_contexts[self.last_referenced_doc]
        return None

    def clear(self) -> None:
        """清空对话历史（保留论文引用记忆）。"""
        self.history.clear()

    @property
    def turn_count(self) -> int:
        return len([t for t in self.history if t.role == "user"])
