"""MCP 工具定义与处理函数 —— 描述可用的工具接口及其调用逻辑。"""

from __future__ import annotations

from typing import Any


# ====== 工具定义（JSON Schema） ======

TOOL_DEFINITIONS = [
    {
        "name": "search_papers",
        "description": "在本地论文知识库中按关键词搜索论文，返回匹配的论文列表（含标题、作者、年份、期刊）。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "搜索关键词（支持中英文，可匹配标题、期刊名）"},
            },
            "required": ["keyword"],
        },
    },
    {
        "name": "answer_question",
        "description": "基于本地 43 篇城市研究论文进行检索增强问答（RAG），返回带来源引用的回答。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "需要回答的问题"},
            },
            "required": ["question"],
        },
    },
    {
        "name": "analyze_paper",
        "description": "对指定论文进行深度结构化分析（研究问题、方法、数据、发现、局限、启示）。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "论文序号（如1）或文件名关键词"},
            },
            "required": ["target"],
        },
    },
    {
        "name": "summarize_paper",
        "description": "对指定论文生成结构化摘要（背景-方法-发现-意义）。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "论文序号（如1）或文件名关键词"},
            },
            "required": ["target"],
        },
    },
    {
        "name": "compare_papers",
        "description": "比较两篇或多篇论文的研究方法、数据来源、发现与启示。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "targets": {"type": "string", "description": "论文序号列表，用逗号分隔，如 1,3,5"},
            },
            "required": ["targets"],
        },
    },
    {
        "name": "list_papers",
        "description": "列出本地知识库中所有论文的索引、标题、作者和年份。",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "graph_search",
        "description": "图增强检索问答（GraphRAG）：利用论文实体共现图扩展检索上下文。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "需要回答的问题"},
            },
            "required": ["question"],
        },
    },
]


# ====== 工具处理函数 ======

def handle_tool_call(tool_name: str, arguments: dict[str, Any], agent: Any) -> list[dict]:
    """根据工具名分发调用，返回 MCP 格式的结果内容列表。"""
    try:
        if tool_name == "list_papers":
            papers = agent.list_available_papers()
            text = _format_paper_list(papers)
            return [{"type": "text", "text": text}]

        elif tool_name == "search_papers":
            keyword = str(arguments.get("keyword", ""))
            papers = agent.list_available_papers()
            matched = [
                p for p in papers
                if keyword.lower() in str(p.get("title", "")).lower()
                or keyword.lower() in p["file_name"].lower()
                or keyword.lower() in str(p.get("journal", "")).lower()
            ]
            text = _format_paper_list(matched[:10])
            return [{"type": "text", "text": text}]

        elif tool_name == "answer_question":
            question = str(arguments.get("question", ""))
            result = agent.answer(question)
            text = f"{result.model_answer}\n\n（共召回 {result.retrieved_count} 个来源片段）"
            return [{"type": "text", "text": text}]

        elif tool_name == "analyze_paper":
            target = str(arguments.get("target", ""))
            result = agent.analyze_paper(target)
            return [{"type": "text", "text": result.formatted_output}]

        elif tool_name == "summarize_paper":
            target = str(arguments.get("target", ""))
            result = agent.summarize_paper(target)
            return [{"type": "text", "text": result.formatted_output}]

        elif tool_name == "compare_papers":
            targets_str = str(arguments.get("targets", ""))
            targets = [t.strip() for t in targets_str.replace("，", ",").split(",") if t.strip()]
            result = agent.compare_papers(targets, topic_hint="用户指定比较")
            return [{"type": "text", "text": result.formatted_output}]

        elif tool_name == "graph_search":
            question = str(arguments.get("question", ""))
            if agent.entity_graph is None:
                return [{"type": "text", "text": "实体图尚未构建，请先在 CLI 中执行 graph build。"}]
            result = agent.graph_search(question)
            return [{"type": "text", "text": result.model_answer}]

        else:
            return [{"type": "text", "text": f"未知工具：{tool_name}"}]

    except Exception as exc:
        return [{"type": "text", "text": f"工具调用异常：{exc}"}]


def _format_paper_list(papers: list[dict]) -> str:
    if not papers:
        return "未找到匹配论文。"
    lines = [f"共 {len(papers)} 篇论文："]
    for p in papers:
        title = p.get("title", "") or p["file_name"]
        year = f" ({p.get('year', '')})" if p.get("year") else ""
        authors = p.get("authors", [])
        author_str = f" — {authors[0]}" if authors else ""
        lines.append(f"  [{p['index']}] {title}{year}{author_str}")
    return "\n".join(lines)
