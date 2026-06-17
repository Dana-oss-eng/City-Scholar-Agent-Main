"""本模块作用：在整个智能体中负责将多步流程的中间结果导出为 Markdown 文件，支撑第三周的最小工作流闭环。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExportArtifact:
    """本数据结构作用：保存一次导出任务的结果信息。"""

    output_path: str
    title: str
    content_length: int


def sanitize_file_name(name: str, fallback_name: str = "workflow_report") -> str:
    """将任意标题清洗为适合文件名使用的字符串。

    输入：
        name: 原始标题文本。
        fallback_name: 当标题为空时使用的默认文件名。
    输出：
        清洗后的文件名主体。
    异常：
        无。
    """

    cleaned_name = re.sub(r"[^\w\u4e00-\u9fff\- ]+", "_", name.strip())
    cleaned_name = re.sub(r"\s+", "_", cleaned_name)
    cleaned_name = cleaned_name.strip("_")
    return cleaned_name or fallback_name


def ensure_output_dir(output_dir: str | Path) -> Path:
    """确保导出目录存在。

    输入：
        output_dir: 导出目录路径。
    输出：
        已确保存在的目录对象。
    异常：
        当目录创建失败时，抛出 OSError。
    """

    path = Path(output_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _transform_comparison_to_markdown(comparison_text: str, paper_count: int) -> str:
    """将终端格式的比较文本转换为适合 Markdown 报告的格式。

    输入：
        comparison_text: 原始比较输出文本。
        paper_count: 纳入论文数量。
    输出：
        Markdown 格式的比较章节内容。
    异常：
        无。
    """

    lines = [
        "## 多篇论文比较分析",
        "",
        f"本部分对纳入的 {paper_count} 篇论文进行了多维度结构化比较，涵盖研究问题、理论基础、"
        "研究方法、数据来源、核心发现、局限性及互补性等关键维度，旨在揭示论文间的共识、分歧与协同关系。",
        "",
    ]

    # 判断是否为 LLM 增强格式（包含 emoji 维度标题）
    llm_dimension_emojis = ["🔬", "📖", "⚙️", "💡", "🏆", "⚠️", "🔗", "📝"]
    is_llm_format = any(
        line.strip().startswith(emo) for line in comparison_text.split("\n") for emo in llm_dimension_emojis
    )

    for line in comparison_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue

        if is_llm_format:
            # ── LLM 增强格式 ──
            # 跳过容器标题行
            if stripped.startswith("📊") and ("比较结果" in stripped or "深度比较" in stripped):
                continue
            if stripped.startswith("比较主题：") or stripped.startswith("纳入论文："):
                continue

            # emoji 维度标题 → Markdown 三级标题
            is_emoji_dim = any(stripped.startswith(emo) for emo in llm_dimension_emojis)
            if is_emoji_dim and "：" in stripped:
                label = stripped.split("：", 1)[0].strip()
                for emo in llm_dimension_emojis:
                    if label.startswith(emo):
                        label = label[len(emo):].strip()
                        break
                lines.append(f"### {label}")
                lines.append("")
                continue

            # 论文列表项
            if re.match(r"^\s*\[\d+\]", stripped):
                lines.append(f"- **{stripped.strip()}**")
                continue
            if stripped.startswith("完整标题："):
                lines.append(f"  {stripped}")
                continue
        else:
            # ── 规则比较格式 ──
            # 转换中文段落标题为 Markdown 子标题
            rule_section_headers = {
                "多篇论文比较结果：": None,  # 跳过
                "纳入论文：": "### 纳入论文概览",
                "共同主题：": "### 共同主题",
                "方法比较：": "### 研究方法比较",
                "数据来源比较：": "### 数据来源比较",
                "主要发现比较：": "### 主要发现比较",
                "综合启示：": "### 综合启示",
            }
            if stripped in rule_section_headers:
                header = rule_section_headers[stripped]
                if header:
                    lines.append(header)
                    lines.append("")
                continue
            # 跳过顶层信息行
            if stripped.startswith("比较主题：") or stripped.startswith("纳入论文数量："):
                continue
            # 论文条目（如 "1. 《file.pdf》"）
            if re.match(r"^\d+\.\s*《", stripped):
                lines.append(f"**{stripped}**")
                continue

        # ── 共同处理：保留原有格式 ──
        if stripped.startswith("•") or stripped.startswith("-") or stripped.startswith("·"):
            lines.append(stripped)
        else:
            lines.append(stripped)

    lines.append("")
    return "\n".join(lines)


def _transform_outline_to_markdown(outline_text: str, topic: str) -> str:
    """将终端格式的提纲文本转换为适合 Markdown 报告的格式。

    输入：
        outline_text: 原始提纲输出文本。
        topic: 综述主题。
    输出：
        Markdown 格式的提纲章节内容。
    异常：
        无。
    """

    lines = [
        "## 综述提纲",
        "",
        f"基于多篇论文的比较分析结果，围绕「{topic}」这一主题，"
        "以下综述提纲为撰写系统性的文献综述提供了结构化框架。各章节逻辑递进，"
        "从研究背景与问题提出出发，逐步深入到方法、数据、核心发现与研究展望。",
        "",
    ]

    for line in outline_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue

        # 跳过容器标题行和元信息行
        if stripped in ("综述提纲：",):
            continue
        if stripped.startswith("主题：") or stripped.startswith("参考论文：") or stripped.startswith("来源论文"):
            continue
        if stripped.startswith("=") or stripped.startswith("---"):
            continue

        # 章节条目（如 "1. Section Title" 或 "第一章 xxx"）
        if re.match(r"^\d+\.\s+", stripped):
            lines.append(f"### {stripped}")
            lines.append("")
            continue
        if re.match(r"^第[一二三四五六七八九十\d]+章", stripped):
            lines.append(f"### {stripped}")
            lines.append("")
            continue

        # 要点 / 项目符号
        if stripped.startswith("•") or stripped.startswith("-") or stripped.startswith("·"):
            lines.append(stripped)
        elif stripped:
            lines.append(f"- {stripped}")

    lines.append("")
    return "\n".join(lines)


def build_workflow_markdown(
    topic: str,
    selected_papers: list[str],
    comparison_text: str,
    outline_text: str,
    step_logs: list[str],
) -> str:
    """将工作流结果整理为内容丰富的 Markdown 文本。

    输入：
        topic: 工作流主题。
        selected_papers: 纳入论文列表。
        comparison_text: 多篇比较展示文本。
        outline_text: 综述提纲展示文本。
        step_logs: 工作流步骤日志。
    输出：
        Markdown 格式文本，每个板块均包含实质性文字内容。
    异常：
        无。
    """

    lines = [
        f"# {topic}",
        "",
        "## 综述摘要",
        "",
        f"本报告围绕「{topic}」这一主题，基于对 {len(selected_papers)} 篇城市研究领域论文的系统性分析，",
        "从研究问题、理论基础、研究方法、数据来源、核心发现等多个维度进行了深度比较与综合评述。",
        "报告进一步提供了结构化的综述提纲，为后续撰写完整的文献综述提供了框架参考。",
        "",
        "## 纳入论文",
        "",
        f"本次综述共纳入 {len(selected_papers)} 篇论文：",
        "",
    ]

    for i, paper in enumerate(selected_papers, 1):
        lines.append(f"{i}. **{paper}**")

    lines.append("")

    # ── 多篇论文比较章节（转换格式，去除代码块） ──
    lines.append(_transform_comparison_to_markdown(comparison_text, len(selected_papers)))

    # ── 综述提纲章节（转换格式，去除代码块） ──
    lines.append(_transform_outline_to_markdown(outline_text, topic))

    # ── 综合小结 ──
    lines.extend([
        "## 综合小结",
        "",
        f"本工作流通过对 {len(selected_papers)} 篇论文的多维度比较与提纲梳理，",
        f"系统呈现了「{topic}」领域的研究现状与知识结构。",
        "上述比较分析与综述提纲可作为文献综述撰写的基础框架，",
        "建议结合具体研究需求进一步深化各章节内容。",
        "",
    ])

    # ── 工作流执行日志 ──
    lines.extend([
        "## 工作流执行日志",
        "",
    ])
    for step_log in step_logs:
        lines.append(f"- {step_log}")

    lines.append("")
    return "\n".join(lines)


def export_markdown_report(
    output_dir: str | Path,
    title: str,
    markdown_text: str,
) -> ExportArtifact:
    """将 Markdown 内容写入输出目录。

    输入：
        output_dir: 输出目录。
        title: 导出标题。
        markdown_text: 待写入的 Markdown 内容。
    输出：
        导出结果对象。
    异常：
        当文件写入失败时，抛出 OSError。
    """

    target_dir = ensure_output_dir(output_dir)
    file_name = sanitize_file_name(title) + ".md"
    output_path = target_dir / file_name
    output_path.write_text(markdown_text, encoding="utf-8")
    return ExportArtifact(
        output_path=str(output_path),
        title=title,
        content_length=len(markdown_text),
    )


def run_export_demo() -> None:
    """执行 export_tool 模块的最小演示。

    输入：
        无。
    输出：
        无。函数会直接打印导出结果。
    异常：
        无。
    """

    markdown_text = build_workflow_markdown(
        topic="城市韧性研究综述",
        selected_papers=["paper_a.pdf", "paper_b.pdf"],
        comparison_text=(
            "多篇论文比较结果：\n"
            "比较主题：城市韧性\n"
            "纳入论文数量：2\n"
            "纳入论文：\n"
            "1. 《paper_a.pdf》\n"
            "   研究问题：城市韧性如何评估？\n"
            "   研究对象：沿海城市群\n"
            "   方法：指标评价与空间分析\n"
            "共同主题：\n"
            "- 多篇论文都涉及「城市韧性」相关议题\n"
            "方法比较：\n"
            "- 论文A采用指标评价法，论文B采用问卷回归法\n"
            "主要发现比较：\n"
            "- 基础设施韧性与治理协同能力显著相关\n"
            "综合启示：\n"
            "- 提升跨区域协同治理能力\n"
        ),
        outline_text=(
            "综述提纲：\n"
            "主题：城市韧性研究综述\n"
            "参考论文：paper_a.pdf, paper_b.pdf\n"
            "1. 研究背景与问题提出\n"
            "- 城市韧性研究缘起与发展\n"
            "- 当前城市安全治理的现实需求\n"
            "2. 核心概念与理论基础\n"
            "- 韧性理论的演进与流派\n"
            "- 城市治理与规划的交叉视角\n"
            "3. 常用方法与数据来源\n"
            "- 指标评价与空间分析方法\n"
            "- 问卷调查与回归分析方法\n"
            "4. 主要发现与研究进展\n"
            "- 基础设施韧性的关键影响因素\n"
            "- 公共服务可达性与城市安全感知\n"
            "5. 研究局限与未来方向\n"
            "- 样本覆盖范围与外部可推广性\n"
            "- 跨学科方法融合的可能性\n"
        ),
        step_logs=[
            "步骤 1：已选中 2 篇论文。",
            "步骤 2：已完成多篇比较。",
            "步骤 3：已生成综述提纲。",
        ],
    )
    artifact = export_markdown_report("outputs", "城市韧性研究综述", markdown_text)
    print(f"导出成功：{artifact.output_path}")


if __name__ == "__main__":
    run_export_demo()
