"""本模块作用：集中管理所有 LLM 提示模板与兜底文本。"""

# ====== 问答提示 ======

def build_answer_system_prompt() -> str:
    return (
        "你是 CityScholar-Agent，一个城市研究领域的论文学术助手。"
        "请严格依据召回的论文片段作答，每条关键论断必须标注证据来源编号 [来源N]，"
        "并在答案末尾附上「依据片段」小节，逐条列出每条论断对应的原文片段原文（至少30字）。"
        "不要编造未在来源中出现的结论。回答应简洁清晰。"
    )


def build_answer_task_prompt(question: str, context_blocks: list[str], chat_history: str = "") -> str:
    ctx = "\n\n".join(context_blocks) if context_blocks else "当前没有可用来源片段。"
    hist = f"\n\n对话历史：\n{chat_history}" if chat_history else ""
    return (
        f"用户问题：{question}{hist}\n\n"
        f"来源片段：\n{ctx}\n\n"
        "请基于来源片段整理中文回答，严格按以下结构输出：\n"
        "1) 直接回答（含 [来源N] 编号标注）\n"
        "2) 关键依据（逐条列出，每条附 [来源N] 编号与论文片段原文）\n"
        "3) 不确定性说明（如有）\n"
        "重要：每条论断必须能从对应来源片段中找到原文支持，不可跨来源编造。"
    )


def build_empty_library_message() -> str:
    return (
        "当前论文库中还没有可用内容，暂时无法完成检索与回答。"
        "请先将 PDF 放入 raw_papers/ 并重新启动。"
    )


def build_no_result_message(question: str) -> str:
    return (
        f"当前未在本地论文中检索到与「{question}」足够相关的内容。"
        "建议换一种问法、缩短问题或补充相关论文后重试。"
    )


def build_answer_suffix() -> str:
    return "以上内容基于当前召回片段整理，建议结合下方来源依据核对。"


# ====== 研究空白分析 ======

RESEARCH_GAP_PROMPT = """你是一个城市研究领域的文献综述专家。请基于以下多篇论文的分析结果，识别研究空白和未来研究方向。

论文分析：
{analyses}

请输出 JSON：
{{
  "common_approaches": "已有研究共同采用的方法/视角（2-3句）",
  "gaps": ["空白点1", "空白点2", "空白点3"],
  "future_directions": ["未来方向1", "未来方向2"],
  "cross_cutting_insight": "跨论文的综合洞见（2-3句）"
}}
"""


# ====== 增强分析提示 ======

ENHANCED_ANALYSIS_PROMPT = """你是城市研究领域的论文分析专家。请对以下论文进行深度结构化分析，输出 JSON。

字段要求（每项 2-4 句）：
- research_question: 研究问题
- research_object: 研究对象（城市/区域/人群）
- methods: 研究方法与模型
- data_source: 数据来源与样本量
- key_findings: 核心发现与统计结果
- limitations: 局限性
- implications: 对城市治理/安全的启示
- research_type: 研究类型（实证/综述/理论/案例研究）
- methodology_tags: 方法标签数组，如 ["回归分析", "问卷调查", "空间分析"]
- quality_notes: 研究方法严谨性简评

论文内容：
{text}

请仅输出 JSON 对象："""


# ====== LLM 多论文比较提示 ======

LLM_COMPARE_PROMPT = """你是城市研究领域的论文比较分析专家。请对以下多篇论文进行深度结构化比较，输出 JSON。

比较维度（每项为一个字符串，2-5 句中文，需明确论文间的异同，引用具体论文内容）：
- research_questions: 各论文研究问题的异同（纯文本字符串，不要嵌套对象）
- theoretical_frameworks: 理论基础与概念框架的异同（纯文本字符串）
- methodology: 研究方法、模型、技术的异同（纯文本字符串）
- data_and_scale: 数据来源、样本量、时空尺度的异同（纯文本字符串）
- key_findings: 核心发现的共识与分歧（纯文本字符串）
- contributions: 理论/实践贡献的对比（纯文本字符串）
- limitations: 局限性的对比（纯文本字符串）
- complementarity: 论文间如何互补（纯文本字符串）
- synthesis: 综合评述——这些论文共同构建了怎样的知识图景（纯文本字符串）

重要：每个字段的值必须是一个纯文本字符串，不要嵌套 JSON 对象或列表，直接写分析文字。

论文内容：
{texts}

请仅输出 JSON 对象："""


# ====== LLM 综述提纲生成 ======

LLM_OUTLINE_PROMPT = """你是城市研究领域的文献综述专家。请基于以下多篇论文的分析结果，生成一份结构化的综述提纲。

论文分析：
{analyses}

请生成一份包含 6 个章节的综述提纲，输出 JSON 格式：
{{
  "sections": [
    {{
      "title": "1. 研究背景与问题提出",
      "bullets": ["要点1（至少15字，结合具体论文内容）", "要点2", "要点3", "要点4"]
    }},
    {{
      "title": "2. 核心概念与理论基础",
      "bullets": ["要点1（结合论文涉及的理论框架）", "要点2", "要点3", "要点4"]
    }},
    {{
      "title": "3. 研究方法与数据来源",
      "bullets": ["要点1（对比各论文的方法选择）", "要点2", "要点3", "要点4"]
    }},
    {{
      "title": "4. 主要发现与研究进展",
      "bullets": ["要点1（综合各论文的核心结论）", "要点2", "要点3", "要点4"]
    }},
    {{
      "title": "5. 对城市治理、规划与安全的政策启示",
      "bullets": ["要点1（从论文发现中提炼实践建议）", "要点2", "要点3", "要点4"]
    }},
    {{
      "title": "6. 研究局限与未来方向",
      "bullets": ["要点1（指出现有研究的不足）", "要点2（提出未来研究建议）", "要点3", "要点4"]
    }}
  ]
}}

要求：
1. 每个要点必须言之有物，至少 15 字，结合具体论文的研究内容
2. 不要泛泛而谈，要具体到论文中的方法、发现或观点
3. 不同章节之间要有逻辑递进
4. 要点要体现跨论文的比较与综合，不是逐篇罗列

请仅输出 JSON 对象："""


# ====== 记忆增强的问答上下文 ======

def build_memory_enhanced_context(
    last_paper_title: str = "",
    last_paper_authors: str = "",
    last_paper_methods: str = "",
    last_paper_topic: str = "",
) -> str:
    """构建当前会话中最近讨论论文的上下文提示。"""
    parts: list[str] = []
    if last_paper_title:
        parts.append(f"当前正在讨论的论文：《{last_paper_title}》")
    if last_paper_authors:
        parts.append(f"作者：{last_paper_authors}")
    if last_paper_methods:
        parts.append(f"研究方法：{last_paper_methods}")
    if last_paper_topic:
        parts.append(f"研究主题：{last_paper_topic}")
    if parts:
        parts.insert(0, "[当前论文上下文]")
        parts.append("如果用户的问题涉及「这篇文章」「该论文」等指代，请理解为指上述论文。\n")
    return "\n".join(parts)
