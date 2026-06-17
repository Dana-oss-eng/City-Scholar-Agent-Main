"""CityScholar-Agent 命令行主入口 —— 本地论文知识库、检索问答、多工具科研助手。"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# ── Windows 终端 UTF-8 编码修复 ──
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 通过环境变量注入 API Key
# 方式一：在系统环境变量中设置 DASHSCOPE_API_KEY 和 DEEPSEEK_API_KEY
# 方式二：将下方占位符替换为你的真实 Key（注意：不要将真实 Key 提交到公开仓库）
if "DASHSCOPE_API_KEY" not in os.environ:
    os.environ["DASHSCOPE_API_KEY"] = "your-dashscope-api-key-here"
if "DEEPSEEK_API_KEY" not in os.environ:
    os.environ["DEEPSEEK_API_KEY"] = "your-deepseek-api-key-here"

from config import get_app_config
from core.agent import (
    AgentAnswer, CityScholarAgent, EmbeddingIndexResponse,
    KnowledgeBaseState, PaperAnalysisResponse, PaperComparisonResponse,
    ReviewOutlineResponse, WorkflowResponse, SummarizeResponse,
    TranslateResponse, RecommendResponse,
)
from llm.factory import create_llm_clients
from rag.graphrag import get_graph_stats
from security.guard import SecurityGuard


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          欢迎横幅与交互式提示                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

WELCOME_BANNER = r"""
   ██████╗██╗████████╗██╗   ██╗███████╗ ██████╗██╗  ██╗ ██████╗ ██╗      █████╗ ██████╗
  ██╔════╝██║╚══██╔══╝╚██╗ ██╔╝██╔════╝██╔════╝██║  ██║██╔═══██╗██║     ██╔══██╗██╔══██╗
  ██║     ██║   ██║    ╚████╔╝ ███████╗██║     ███████║██║   ██║██║     ███████║██████╔╝
  ██║     ██║   ██║     ╚██╔╝  ╚════██║██║     ██╔══██║██║   ██║██║     ██╔══██║██╔══██╗
  ╚██████╗██║   ██║      ██║   ███████║╚██████╗██║  ██║╚██████╔╝███████╗██║  ██║██║  ██║
   ╚═════╝╚═╝   ╚═╝      ╚═╝   ╚══════╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝
                          🏙️  城市研究论文学术助手  v2.0  🏙️
"""

WELCOME_GREETING = """
👋 欢迎使用 CityScholar-Agent！

  我是你的城市研究论文智能助手，可以帮你：
  📖 检索问答 —— 基于本地论文库回答你的研究问题
  📊 论文分析 —— 单篇深度分析、多篇对比、研究空白识别
  📝 综述辅助 —— 结构化摘要、综述提纲、一键工作流
  🌐 学术翻译 —— 中英双向学术翻译
  🔗 文献管理 —— BibTeX 导出、相似论文推荐
  🕸️ GraphRAG —— 实体共现图谱增强检索

  💡 随时输入 help 查看完整命令列表，输入 exit 退出程序。
"""


# ====== 工具函数 ======

def ensure_directories(paths: list[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def format_page_numbers(page_numbers: list[int]) -> str:
    return "、".join(str(n) for n in page_numbers) if page_numbers else "?"


def format_elapsed(start: float) -> str:
    """格式化耗时。"""
    elapsed = time.time() - start
    if elapsed < 1:
        return f"{elapsed * 1000:.0f}ms"
    elif elapsed < 60:
        return f"{elapsed:.1f}s"
    else:
        return f"{elapsed / 60:.1f}min"


# ====== 启动信息 ======

def show_startup_info(config: dict, kb: KnowledgeBaseState, embedding_status: EmbeddingIndexResponse | None,
                      has_chat: bool, has_embed: bool) -> None:
    """显示系统启动后的状态面板。"""
    print("━" * 58)
    print(f"  📂 论文目录：{config['raw_papers_dir']}")
    print(f"  📄 发现 PDF：{len(kb.pdf_files)} 篇 | 解析成功：{len(kb.documents)} 篇 | 可检索片段：{len(kb.chunk_records)} 个")
    if kb.parse_errors:
        print(f"  ⚠️  解析失败：{len(kb.parse_errors)} 篇")
    print(f"  🤖 对话模型：{'✅ 已启用' if has_chat else '❌ 未启用'} | 向量模型：{'✅ 已启用' if has_embed else '❌ 未启用'}")
    if embedding_status and embedding_status.vector_count > 0:
        print(f"  🔢 向量索引：{embedding_status.vector_count} 条 | 加载方式：{'📦 缓存' if embedding_status.loaded_from_cache else '🆕 新建'}")
    print("━" * 58)


def show_help_panel() -> None:
    """显示详细帮助面板。"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           📋 命令一览                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  💬 直接输入问题          → 检索问答（自动混合检索 + Reranker）               ║
║  📄 summarize [N/关键词]  → 生成论文结构化摘要                                ║
║  🔬 analyze [N/关键词]    → 单篇论文深度分析                                  ║
║  ⚖️  compare [1,2]        → 多篇论文深度对比（9维度LLM增强分析）               ║
║  📋 outline [1,2]::主题   → 综述提纲生成                                      ║
║  🔄 workflow [1,2]::主题  → 一键工作流（对比→提纲→导出Markdown）              ║
║  🔍 gaps [1,2,3]          → 研究空白与未来方向分析                             ║
║  📎 recommend [N]          → 相似论文推荐                                     ║
║  🌐 translate 文本         → 中英学术翻译                                     ║
║  📎 bibtex [N]             → 导出 BibTeX 引用                                 ║
║  🕸️  graph build           → 构建论文实体共现图（GraphRAG）                    ║
║  🕸️  graph <问题>          → 图增强检索问答                                   ║
║  🕸️  graph stats           → 查看图谱统计                                     ║
║  🕸️  graph export [png/mermaid/dot] → 导出可视化图谱（默认 PNG 图片）       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  📋 papers / list          → 列出所有可用论文                                 ║
║  🔎 search <关键词>         → 按关键词搜索论文                                 ║
║  🔧 build_index            → 构建向量索引                                     ║
║  🔧 rebuild_index          → 强制重建向量索引                                 ║
║  🧠 memory                 → 查看对话记忆状态                                 ║
║  🧹 clear                  → 清空对话记忆                                     ║
║  📊 stats                  → 查看系统详细状态                                 ║
║  🔒 security               → 查看安全防护状态                                 ║
║  ❓ help / 帮助             → 显示此帮助面板                                  ║
║  🚪 exit / quit / 退出     → 退出程序                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


# ====== 展示函数 ======

def display_answer(result: AgentAnswer) -> None:
    """展示检索问答结果。"""
    print(f"\n📝 回答：\n{result.model_answer}")
    if result.sources:
        print("\n📚 来源依据：")
        for i, s in enumerate(result.sources, 1):
            title = s.get("title", "") or s.get("file_name", "?")
            authors = s.get("authors", [])
            author_str = f" | {' '.join(authors[:3])}" if authors else ""
            year = f" ({s.get('year', '')})" if s.get("year") else ""
            print(f"  [{i}] {title}{year}{author_str}")
            print(f"      页码：{format_page_numbers(s.get('page_numbers', []))} | 章节：{s.get('section', '?')} | 相关度：{s.get('score', 0)}")
            print(f"      片段：{s.get('snippet', '')[:150]}")
        print(f"\n  📊 共召回 {result.retrieved_count} 个相关片段")


def display_analysis(result: PaperAnalysisResponse) -> None:
    """展示论文分析结果。"""
    print(f"\n📊 {result.status_message}")
    print(result.formatted_output)


def display_comparison(result: PaperComparisonResponse) -> None:
    """展示论文对比结果。"""
    print(f"\n📊 {result.status_message}")
    print(result.formatted_output)


def display_outline(result: ReviewOutlineResponse) -> None:
    """展示综述提纲。"""
    print(f"\n📋 {result.status_message}")
    print(result.formatted_output)


def display_workflow(result: WorkflowResponse) -> None:
    """展示工作流执行结果。"""
    print(f"\n🔄 {result.status_message}")
    print(result.formatted_output)


def display_embedding_status(result: EmbeddingIndexResponse) -> None:
    """展示向量索引状态。"""
    print(f"\n🔢 {result.status_message}")
    print(f"   路径：{result.index_path} | 向量数：{result.vector_count} | 缓存：{'是' if result.loaded_from_cache else '否'}")


def display_stats(agent: CityScholarAgent, config: dict, guard: SecurityGuard, start_time: float) -> None:
    """展示系统详细状态。"""
    kb = agent.ensure_knowledge_base_ready()
    mem = agent.memory
    print("\n" + "━" * 58)
    print("  📊 系统状态总览")
    print("━" * 58)
    print(f"  📂 论文目录：{config['raw_papers_dir']}")
    print(f"  📄 论文数量：{len(kb.pdf_files)} 篇 PDF | {len(kb.documents)} 篇已解析 | {len(kb.chunk_records)} 个片段")
    print(f"  🤖 对话模型：{'DeepSeek / Qwen' if agent.chat_client else '未启用'}")
    print(f"  🔢 向量模型：{agent.embedding_model_name or '未启用'}（{agent.embedding_dimensions}维）")
    if agent.embedding_index:
        print(f"      向量索引：{len(agent.embedding_index.vectors)} 条")
    if agent.entity_graph:
        print(f"  🕸️  实体图谱：已构建")
        print(f"      {get_graph_stats(agent.entity_graph)}")
    else:
        print(f"  🕸️  实体图谱：未构建（执行 graph build 构建）")
    print(f"  🧠 对话记忆：{mem.turn_count} 轮")
    print(f"  🔒 安全审计：{len(guard.audit_logs)} 条日志 | 拦截：{guard._blocked_count} 次")
    uptime = format_elapsed(start_time)
    print(f"  ⏱️  运行时间：{uptime}")
    print("━" * 58)


# ====== 命令解析 ======

def parse_target(user_input: str) -> str | None:
    """解析单目标参数（论文编号或关键词）。"""
    parts = user_input.strip().split(maxsplit=1)
    return parts[1].strip() if len(parts) >= 2 and parts[1].strip() else None


def parse_target_list(text: str) -> list[str]:
    """解析多目标参数列表（支持逗号或空格分隔）。"""
    t = text.replace("，", ",").strip()
    if not t:
        return []
    if "," in t:
        return [x.strip() for x in t.split(",") if x.strip()]
    return [x.strip() for x in t.split() if x.strip()]


def parse_topic(user_input: str) -> tuple[list[str] | None, str]:
    """解析「论文编号::主题」格式。"""
    parts = user_input.strip().split(maxsplit=1)
    if len(parts) < 2:
        return None, ""
    payload = parts[1].strip()
    if "::" not in payload:
        return None, payload
    target_text, topic = payload.split("::", maxsplit=1)
    targets = parse_target_list(target_text) or None
    return targets, topic.strip()


# ====== 交互式提示 ======

def get_user_input() -> str | None:
    """获取用户输入，带友好的交互式提示。"""
    try:
        ui = input("\n💬 请输入你的问题或命令（输入 help 查看帮助）：").strip()
        return ui
    except (EOFError, KeyboardInterrupt):
        print("\n\n👋 感谢使用 CityScholar-Agent，欢迎再次回来！")
        return None


def confirm_action(prompt: str) -> bool:
    """需要用户确认的操作。"""
    try:
        answer = input(f"⚠️  {prompt} (y/n)：").strip().lower()
        return answer in {"y", "yes", "是", "确认"}
    except (EOFError, KeyboardInterrupt):
        return False


# ====== 主循环 ======

def run_cli(agent: CityScholarAgent, config: dict) -> None:
    """交互式命令行主循环 —— 处理所有用户命令与问答请求。"""
    embed_dim = int(config.get("ds_embed_dim", 512))
    processed_dir = Path(str(config["processed_data_dir"]))
    guard = SecurityGuard()
    start_time = time.time()

    # ── 显示欢迎信息 ──
    print(WELCOME_BANNER)
    print(WELCOME_GREETING)

    while True:
        # ── 获取用户输入 ──
        ui = get_user_input()
        if ui is None:
            break
        if not ui:
            continue

        # ── 安全检查：频率限制 ──
        allowed, rate_msg = guard.check_rate_limit()
        if not allowed:
            print(f"⛔ {rate_msg}")
            print("   💡 请稍等片刻再继续提问。")
            continue

        # ── 安全检查：输入消毒 + 注入检测 ──
        ui, warnings = guard.check_input(ui)
        for w in warnings:
            print(f"⚠️ {w}")

        nl = ui.lower()

        # ========== 退出命令 ==========
        if nl in {"exit", "quit", "q", "退出"}:
            print("\n👋 感谢使用 CityScholar-Agent！欢迎再次回来探讨城市研究。")
            # 最终安全摘要
            if guard.audit_logs:
                print(f"📋 本次会话安全记录：{len(guard.audit_logs)} 条审计日志，{guard._blocked_count} 次高风险拦截。")
            break

        # ========== 帮助命令 ==========
        if nl in {"help", "帮助", "?"}:
            show_help_panel()
            continue

        # ========== 论文列表 ==========
        if nl in {"papers", "list", "论文"}:
            items = agent.list_available_papers()
            if not items:
                print("\n📭 当前论文库中没有可用的论文。")
                print("   💡 请将 PDF 文件放入 data/raw_papers/ 目录后重新启动程序。")
            else:
                print(f"\n📄 当前可用论文（共 {len(items)} 篇）：")
                print("━" * 58)
                for it in items:
                    title = it.get("title", "")
                    authors = it.get("authors", [])
                    year = f" ({it.get('year', '')})" if it.get("year") else ""
                    author_short = f" | {authors[0]}..." if authors else ""
                    print(f"  [{it['index']}] {title or it['file_name']}{year}{author_short}")
                    print(f"      ID: {it['document_id']} | 页数: {it['total_pages']} | 字数: {it['total_characters']}")
                print("━" * 58)
                print("  💡 使用编号（如 1）或文件名来指定某篇论文。")
            continue

        # ========== 论文搜索 ==========
        if nl.startswith("search ") or nl.startswith("搜索 "):
            keyword = ui.strip().split(maxsplit=1)[1].strip() if ui.strip().split(maxsplit=1)[1:] else ""
            if not keyword:
                print("💡 请输入搜索关键词，如：search 城市韧性")
                continue
            items = agent.list_available_papers()
            matched = [it for it in items if keyword.lower() in str(it.get("title", "")).lower()
                      or keyword.lower() in it["file_name"].lower()
                      or keyword.lower() in str(it.get("journal", "")).lower()]
            if matched:
                print(f"\n🔍 搜索「{keyword}」结果（共 {len(matched)} 篇）：")
                for it in matched[:10]:
                    print(f"  [{it['index']}] {it.get('title', '') or it['file_name']} ({it.get('year', '')})")
                if len(matched) > 10:
                    print(f"  ... 还有 {len(matched) - 10} 篇匹配（请缩小搜索范围）。")
            else:
                print(f"\n🔍 搜索「{keyword}」没有找到匹配的论文。")
                print("   💡 试试其他关键词，或使用 papers 命令查看全部论文。")
            continue

        # ========== 向量索引构建 ==========
        if nl in {"build_index", "index"}:
            print("\n🔧 正在构建向量索引，这可能需要一些时间...")
            t0 = time.time()
            try:
                result = agent.prepare_embedding_index(
                    client=agent.embedding_client, model_name=agent.embedding_model_name,
                    dimensions=embed_dim, processed_data_dir=processed_dir,
                    build_if_missing=True, force_rebuild=False,
                )
                display_embedding_status(result)
                print(f"   ⏱️  耗时：{format_elapsed(t0)}")
            except Exception as e:
                print(f"❌ 向量索引构建失败：{e}")
                print("   💡 请检查 embedding 客户端是否已正确配置。")
            continue

        if nl == "rebuild_index":
            print("\n🔧 正在强制重建向量索引...")
            if not confirm_action("这将重新计算所有向量，可能需要较长时间，确定继续？"):
                print("   已取消。")
                continue
            t0 = time.time()
            try:
                result = agent.prepare_embedding_index(
                    client=agent.embedding_client, model_name=agent.embedding_model_name,
                    dimensions=embed_dim, processed_data_dir=processed_dir,
                    build_if_missing=True, force_rebuild=True,
                )
                display_embedding_status(result)
                print(f"   ⏱️  耗时：{format_elapsed(t0)}")
            except Exception as e:
                print(f"❌ 向量索引重建失败：{e}")
            continue

        # ========== 对话记忆 ==========
        if nl in {"memory", "记忆"}:
            mem = agent.memory
            print("\n🧠 对话记忆状态：")
            print(f"   对话轮数：{mem.turn_count}")
            print(f"   最近引用的论文：{mem.last_referenced_file or '无'}")
            print(f"   锁定的论文：{mem.pinned_docs if mem.pinned_docs else '无'}")
            if mem.history:
                print(f"   最近对话：")
                for t in mem.history[-4:]:
                    role = "👤 你" if t.role == "user" else "🤖 助手"
                    print(f"     {role}：{t.content[:80]}{'...' if len(t.content) > 80 else ''}")
            continue

        if nl in {"clear", "清空", "重置"}:
            agent.memory.clear()
            print("\n🧹 对话记忆已清空。论文引用记忆已保留。")
            continue

        # ========== 系统状态 ==========
        if nl in {"stats", "状态"}:
            display_stats(agent, config, guard, start_time)
            continue

        # ========== 安全状态 ==========
        if nl in {"security", "安全"}:
            print(f"\n{guard.get_summary()}")
            continue

        # ========== 科研工具：结构化摘要 ==========
        if nl.startswith("summarize") or nl.startswith("摘要"):
            target = parse_target(ui)
            print(f"\n📄 正在生成论文摘要...")
            t0 = time.time()
            try:
                result = agent.summarize_paper(target)
                print(f"\n📄 {result.formatted_output}")
                print(f"   ⏱️  耗时：{format_elapsed(t0)}")
            except Exception as e:
                print(f"❌ 摘要生成失败：{e}")
                print("   💡 请确认目标论文存在（使用 papers 查看编号），或检查 LLM 连接。")
            continue

        # ========== 科研工具：单篇分析 ==========
        if nl.startswith("analyze") or nl.startswith("分析"):
            target = parse_target(ui)
            print(f"\n🔬 正在深度分析论文...")
            t0 = time.time()
            try:
                result = agent.analyze_paper(target)
                display_analysis(result)
                print(f"   ⏱️  耗时：{format_elapsed(t0)}")
            except Exception as e:
                print(f"❌ 分析失败：{e}")
                print("   💡 请确认目标论文存在（使用 papers 查看编号）。")
            continue

        # ========== 科研工具：多篇对比 ==========
        if nl.startswith("compare") or nl.startswith("对比"):
            parts = ui.strip().split(maxsplit=1)
            targets = parse_target_list(parts[1]) if len(parts) >= 2 else None
            if not targets:
                print("💡 请输入要对比的论文编号，如：compare 1,2 或 compare 1 2 3")
                continue
            print(f"\n⚖️  正在对比 {len(targets)} 篇论文...")
            t0 = time.time()
            try:
                result = agent.compare_papers(targets)
                display_comparison(result)
                print(f"   ⏱️  耗时：{format_elapsed(t0)}")
            except Exception as e:
                print(f"❌ 比较失败：{e}")
            continue

        # ========== 科研工具：综述提纲 ==========
        if nl.startswith("outline") or nl.startswith("提纲"):
            targets, topic = parse_topic(ui)
            if not topic:
                print("💡 请使用格式：outline [论文编号]::主题")
                print("   例如：outline 1,2 :: 城市韧性研究综述")
                continue
            print(f"\n📋 正在为「{topic}」生成综述提纲...")
            t0 = time.time()
            try:
                result = agent.generate_review_outline(topic, targets)
                display_outline(result)
                print(f"   ⏱️  耗时：{format_elapsed(t0)}")
            except Exception as e:
                print(f"❌ 提纲生成失败：{e}")
            continue

        # ========== 科研工具：一键工作流 ==========
        if nl.startswith("workflow") or nl.startswith("流程"):
            targets, topic = parse_topic(ui)
            if not topic:
                print("💡 请使用格式：workflow [论文编号]::主题")
                print("   例如：workflow 1,2,3 :: 城市韧性研究综述")
                continue
            print(f"\n🔄 正在执行「{topic}」的完整工作流...")
            print("   这将依次完成：论文对比 → 综述提纲 → 导出 Markdown 报告")
            t0 = time.time()
            try:
                result = agent.run_review_workflow(topic, targets)
                display_workflow(result)
                print(f"   ⏱️  总耗时：{format_elapsed(t0)}")
            except Exception as e:
                print(f"❌ 工作流执行失败：{e}")
            continue

        # ========== 科研工具：研究空白分析 ==========
        if nl.startswith("gaps") or nl.startswith("空白") or nl.startswith("研究空白"):
            parts = ui.strip().split(maxsplit=1)
            targets = parse_target_list(parts[1]) if len(parts) >= 2 else None
            print(f"\n🔬 正在识别研究空白与未来方向...")
            t0 = time.time()
            try:
                result = agent.identify_research_gaps(targets)
                print(f"\n🔬 {result}")
                print(f"   ⏱️  耗时：{format_elapsed(t0)}")
            except Exception as e:
                print(f"❌ 研究空白分析失败：{e}")
            continue

        # ========== 科研工具：相似论文推荐 ==========
        if nl.startswith("recommend") or nl.startswith("推荐"):
            target = parse_target(ui)
            print(f"\n📎 正在查找相似论文...")
            t0 = time.time()
            try:
                result = agent.recommend_papers(target)
                print(f"\n📎 针对《{result.target}》的{result.formatted_output}")
                print(f"   ⏱️  耗时：{format_elapsed(t0)}")
            except Exception as e:
                print(f"❌ 推荐失败：{e}")
                print("   💡 请确认向量索引已构建（执行 build_index），且目标论文存在。")
            continue

        # ========== 科研工具：学术翻译 ==========
        if nl.startswith("translate") or nl.startswith("翻译"):
            text = ui.strip().split(maxsplit=1)[1].strip() if len(ui.strip().split(maxsplit=1)) >= 2 else ""
            if not text:
                print("💡 请输入待翻译文本，如：translate Urban resilience refers to...")
                continue
            print(f"\n🌐 正在翻译...")
            t0 = time.time()
            try:
                result = agent.translate_text(text)
                print(f"\n🌐 翻译结果（{result.direction}）：")
                print(f"   原文：{result.original[:200]}{'...' if len(result.original) > 200 else ''}")
                print(f"   译文：{result.translated}")
                print(f"   ⏱️  耗时：{format_elapsed(t0)}")
            except Exception as e:
                print(f"❌ 翻译失败：{e}")
            continue

        # ========== 科研工具：BibTeX 导出 ==========
        if nl.startswith("bibtex") or nl.startswith("引用"):
            target = parse_target(ui)
            try:
                result = agent.export_bibtex(target)
                print(f"\n📎 BibTeX 引用：")
                print("━" * 58)
                print(result)
                print("━" * 58)
                if not target:
                    print("  💡 以上为全部论文的 BibTeX，使用 bibtex [编号] 导出单篇。")
            except Exception as e:
                print(f"❌ BibTeX 导出失败：{e}")
            continue

        # ========== GraphRAG 图谱 ==========
        if nl.startswith("graph") or nl.startswith("图谱"):
            parts = ui.strip().split(maxsplit=1)
            sub = (parts[1].strip() if len(parts) >= 2 else "").lower()
            if sub in {"build", "构建", "rebuild"}:
                print(f"\n🕸️  正在构建论文实体共现图...")
                t0 = time.time()
                try:
                    stats = agent.build_entity_graph()
                    print(f"\n🕸️  实体图已构建完成！")
                    print(stats)
                    print(f"   ⏱️  耗时：{format_elapsed(t0)}")
                    print("   💡 现在可以使用 graph <问题> 进行图增强检索，或 graph export 导出可视化图谱。")
                except Exception as e:
                    print(f"❌ 图谱构建失败：{e}")
            elif sub in {"stats", "统计"}:
                if agent.entity_graph:
                    print(f"\n🕸️  图谱统计：\n{get_graph_stats(agent.entity_graph)}")
                else:
                    print("\n📭 图谱尚未构建。")
                    print("   💡 请先执行 graph build 构建实体共现图。")
            elif sub.startswith("export") or sub.startswith("导出"):
                if not agent.entity_graph:
                    print("\n📭 图谱尚未构建，无法导出。")
                    print("   💡 请先执行 graph build 构建实体共现图。")
                    continue
                # 解析导出格式：graph export [png|mermaid|dot]
                exp_parts = sub.split()
                fmt = exp_parts[1] if len(exp_parts) > 1 else "png"
                if fmt not in {"png", "mermaid", "dot"}:
                    print(f"\n⚠️  不支持的导出格式「{fmt}」，可选：png（默认）、mermaid、dot")
                    continue
                try:
                    output_dir = Path(str(config.get("output_dir", "outputs")))
                    output_dir.mkdir(parents=True, exist_ok=True)
                    if fmt == "png":
                        png_path = output_dir / "entity_graph.png"
                        result = agent.export_graph_visualization(fmt="png", output_path=str(png_path))
                        if result and result.startswith("⚠️"):
                            print(f"\n{result}")
                        else:
                            print(f"\n🕸️  图谱已导出为 PNG 图片：{png_path}")
                            print(f"   💡 直接双击打开即可查看，节点标签清晰可读。")
                    elif fmt == "mermaid":
                        result = agent.export_graph_visualization(fmt="mermaid")
                        print(f"\n🕸️  图谱 Mermaid 代码（可复制到支持 Mermaid 的编辑器中渲染）：")
                        print("━" * 58)
                        print(result)
                        print("━" * 58)
                        print("   💡 将上方代码粘贴到 VS Code / Typora / GitHub 的 Markdown 文件中即可渲染。")
                    else:  # dot
                        result = agent.export_graph_visualization(fmt="dot")
                        dot_path = output_dir / "entity_graph.dot"
                        dot_path.write_text(result, encoding="utf-8")
                        print(f"\n🕸️  图谱已导出到：{dot_path}")
                        print(f"   💡 安装 Graphviz 后运行：dot -Tpng {dot_path} -o graph.png")
                except Exception as e:
                    print(f"❌ 图谱导出失败：{e}")
            elif sub:
                # graph + 问题 = 图增强检索
                if agent.entity_graph:
                    print(f"\n🕸️  正在进行图增强检索...")
                    t0 = time.time()
                    try:
                        answer = agent.graph_search(sub)
                        print(f"\n🕸️ [GraphRAG 增强] 📝 回答：\n{answer.model_answer}")
                        if answer.sources:
                            print(f"\n📚 来源（共 {answer.retrieved_count} 条）：")
                            for i, s in enumerate(answer.sources[:3], 1):
                                print(f"  [{i}] {s.get('title', s.get('file_name', '?'))} | 相关度：{s['score']}")
                        print(f"   ⏱️  耗时：{format_elapsed(t0)}")
                    except Exception as e:
                        print(f"❌ 图谱搜索失败：{e}")
                else:
                    print("\n📭 请先执行 graph build 构建实体图。")
            else:
                print("\n💡 graph 命令用法：")
                print("  graph build            → 构建论文实体共现图")
                print("  graph stats            → 查看图谱统计")
                print("  graph export            → 导出为 PNG 图片（默认）")
                print("  graph export mermaid    → 导出 Mermaid 可视化")
                print("  graph export dot        → 导出 Graphviz DOT 图谱")
                print("  graph <问题>           → 图增强检索问答")
            continue

        # ========== 默认：检索问答 ==========
        print(f"\n🔍 正在检索相关知识并生成回答...")
        t0 = time.time()
        try:
            answer = agent.answer(ui)
        except Exception as e:
            print(f"❌ 问答出错：{e}")
            print("   💡 请检查问题表述，或使用 help 查看支持的格式。")
            continue

        # LLM 增强（如果 agent.chat_client 可用且原回答为规则生成）
        if agent.chat_client and answer.sources:
            try:
                from core.agent import _build_llm_context
                from core.prompts import build_answer_system_prompt, build_answer_task_prompt
                ctx = _build_llm_context(
                    agent.retrieve_chunks_for_question(ui, agent.ensure_knowledge_base_ready().chunk_records)
                )
                hist = agent.memory.build_context_for_llm()
                enhanced = agent.chat_client.chat(
                    messages=[
                        {"role": "system", "content": build_answer_system_prompt()},
                        {"role": "user", "content": build_answer_task_prompt(ui, ctx, hist)},
                    ],
                    temperature=0.2, max_tokens=1200,
                )
                answer.model_answer = enhanced
                agent.memory.add_user(ui)
                agent.memory.add_assistant(enhanced, doc_id=answer.sources[0].get("document_id", "") if answer.sources else "")
            except Exception:
                pass

        # ── 安全检查：输出审查 ──
        output_text, output_warnings = guard.check_output(answer.model_answer)
        answer.model_answer = output_text
        for w in output_warnings:
            print(f"⚠️ {w}")

        display_answer(answer)
        print(f"   ⏱️  耗时：{format_elapsed(t0)}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              主入口 main()                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main() -> None:
    """CityScholar-Agent 启动入口 —— 初始化配置、LLM 客户端、知识库与交互循环。"""
    print("\n🚀 正在启动 CityScholar-Agent，请稍候...\n")

    # ── 加载配置 ──
    config = get_app_config()
    ensure_directories([
        Path(str(config["data_dir"])),
        Path(str(config["raw_papers_dir"])),
        Path(str(config["processed_data_dir"])),
        Path(str(config["output_dir"])),
    ])

    # ── 创建多 LLM 客户端 ──
    providers = create_llm_clients(
        dashscope_api_key=str(config.get("ds_api_key", "")),
        dashscope_base_url=str(config.get("ds_base_url", "")),
        dashscope_chat_model=str(config.get("ds_chat_model", "qwen-plus")),
        dashscope_analysis_model=str(config.get("ds_analysis_model", "qwen-max")),
        dashscope_embedding_model=str(config.get("ds_embed_model", "text-embedding-v3")),
        dashscope_timeout=int(config.get("ds_timeout", 45)),
        deepseek_api_key=str(config.get("deepseek_api_key", "")),
        deepseek_base_url=str(config.get("deepseek_base_url", "https://api.deepseek.com")),
        deepseek_model=str(config.get("deepseek_model", "deepseek-chat")),
    )

    # ── 初始化 Agent ──
    agent = CityScholarAgent(
        raw_papers_dir=Path(str(config["raw_papers_dir"])),
        chunk_size=int(config.get("chunk_size", 800)),
        chunk_overlap=int(config.get("chunk_overlap", 150)),
        top_k=int(config.get("top_k", 5)),
    )

    # 注入 LLM 客户端
    agent.chat_client = providers.chat_client
    agent.analysis_client = providers.analysis_client

    # ── 构建知识库 ──
    print("📂 正在构建本地论文知识库...")
    try:
        kb = agent.build_knowledge_base()
    except ImportError as e:
        print(f"❌ 依赖缺失：{e}")
        print("   💡 请运行 pip install -r requirements.txt 安装所需依赖。")
        return
    except Exception as e:
        print(f"❌ 知识库构建失败：{e}")
        return

    # ── 元数据提取 ──
    if agent.chat_client and kb.documents:
        print("🏷️  正在提取论文元数据...")
        try:
            agent.enrich_metadata()
            enriched = sum(1 for d in kb.documents if d.metadata.title)
            print(f"   ✅ 元数据提取完成：{enriched}/{len(kb.documents)} 篇")
        except Exception as e:
            print(f"   ⚠️  元数据提取遇到问题（不影响核心功能）：{e}")

    # ── 向量索引加载 ──
    embedding_status: EmbeddingIndexResponse | None = None
    if providers.embedding_client and kb.chunk_records:
        agent.embedding_client = providers.embedding_client
        agent.embedding_model_name = str(config.get("ds_embed_model", "text-embedding-v3"))
        agent.embedding_dimensions = int(config.get("ds_embed_dim", 512))
        print("🔢 正在加载向量索引...")
        try:
            embedding_status = agent.prepare_embedding_index(
                client=providers.embedding_client,
                model_name=agent.embedding_model_name,
                dimensions=agent.embedding_dimensions,
                processed_data_dir=Path(str(config["processed_data_dir"])),
                build_if_missing=False, force_rebuild=False,
            )
        except Exception as e:
            print(f"   ⚠️  向量索引加载失败（不影响关键词检索）：{e}")

    has_chat = providers.chat_client is not None
    has_embed = providers.embedding_client is not None

    # ── 显示启动状态 ──
    show_startup_info(config, kb, embedding_status, has_chat, has_embed)

    # ── 解析失败提示 ──
    if kb.parse_errors:
        print(f"\n⚠️  以下 {len(kb.parse_errors)} 篇 PDF 解析失败（已跳过）：")
        for err in kb.parse_errors[:5]:
            print(f"  - {err['file_name']}: {err['error_message'][:100]}")
        if len(kb.parse_errors) > 5:
            print(f"  ... 还有 {len(kb.parse_errors) - 5} 篇解析失败。")

    # ── 空库保护 ──
    if not kb.pdf_files:
        print("\n📭 当前未在 data/raw_papers/ 发现任何 PDF 文件。")
        print("   💡 请将论文 PDF 放入该目录后重新运行程序。")
        print("   程序仍可启动，但检索功能将不可用。\n")
        # 不直接 return，允许用户探索其他功能
    if not kb.documents and kb.pdf_files:
        print("\n⚠️  发现了 PDF 文件但未能成功解析任何一篇。")
        print("   💡 请检查 PDF 格式是否正确（是否为扫描版、加密等）。\n")

    # ── 启动交互式主循环 ──
    run_cli(agent, config)


if __name__ == "__main__":
    main()
