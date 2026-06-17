"""本模块作用：使用 LLM 对论文生成结构化摘要（背景-方法-结果-意义）。"""

from __future__ import annotations

from dataclasses import dataclass, field


_SUMMARIZE_PROMPT = """你是一个学术论文摘要生成助手。请基于以下论文内容生成结构化中文摘要，严格输出 JSON。

格式要求：
{{
  "background": "研究背景与问题（2-3句）",
  "methods": "研究方法与数据（2-3句）",
  "findings": "主要发现与结果（2-3句）",
  "implications": "研究意义与启示（1-2句）",
  "keywords": ["关键词1", "关键词2", "关键词3"]
}}

论文全文：
{text}

请仅输出 JSON 对象："""


@dataclass
class PaperSummary:
    file_name: str
    background: str = ""
    methods: str = ""
    findings: str = ""
    implications: str = ""
    keywords: list[str] = field(default_factory=list)


def generate_summary(full_text: str, file_name: str, chat_fn) -> PaperSummary:
    """为单篇论文生成结构化摘要。

    Args:
        full_text: 论文全文
        file_name: 论文文件名
        chat_fn: LLM chat 函数
    """
    if chat_fn is None:
        return PaperSummary(file_name=file_name, background="LLM 未启用，无法生成摘要。")

    import json
    max_chars = 15000
    text_input = full_text[:max_chars]
    if len(full_text) > max_chars:
        text_input = full_text[:int(max_chars * 0.6)] + "\n\n[...中间内容已省略...]\n\n" + full_text[-int(max_chars * 0.4):]

    try:
        raw = chat_fn(
            messages=[
                {"role": "system", "content": "你是一个精确的学术论文摘要生成器。只基于给定内容，不要编造。"},
                {"role": "user", "content": _SUMMARIZE_PROMPT.format(text=text_input)},
            ],
            temperature=0.2, max_tokens=800,
            response_format={"type": "json_object"},
        )
        data = json.loads(raw)
    except Exception:
        return PaperSummary(file_name=file_name, background="LLM 调用失败，摘要生成中断。")

    return PaperSummary(
        file_name=file_name,
        background=str(data.get("background", "")).strip(),
        methods=str(data.get("methods", "")).strip(),
        findings=str(data.get("findings", "")).strip(),
        implications=str(data.get("implications", "")).strip(),
        keywords=[str(k).strip() for k in data.get("keywords", []) if str(k).strip()],
    )


def format_summary(s: PaperSummary) -> str:
    lines = [
        f"论文摘要：《{s.file_name}》",
        f"研究背景：{s.background}",
        f"研究方法：{s.methods}",
        f"主要发现：{s.findings}",
        f"研究意义：{s.implications}",
    ]
    if s.keywords:
        lines.append(f"关键词：{', '.join(s.keywords)}")
    return "\n".join(lines)
