"""重新生成 outputs/ 下的工作流报告和知识图谱，基于当前 raw_papers/ 中的 19 篇论文。"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    RAW_PAPERS_DIR, OUTPUT_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
    DS_API_KEY, DS_BASE_URL, DS_CHAT_MODEL, DS_ANALYSIS_MODEL,
    DS_EMBED_MODEL, DS_TIMEOUT,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
)
from llm.factory import create_llm_clients
from core.agent import CityScholarAgent


def main():
    print("=" * 60)
    print("CityScholar-Agent 输出文件重新生成")
    print(f"论文目录: {RAW_PAPERS_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)

    # ── 1. 创建 LLM 客户端 ──
    print("\n[1/5] 初始化 LLM 客户端...")
    providers = create_llm_clients(
        dashscope_api_key=DS_API_KEY,
        dashscope_base_url=DS_BASE_URL,
        dashscope_chat_model=DS_CHAT_MODEL,
        dashscope_analysis_model=DS_ANALYSIS_MODEL,
        dashscope_embedding_model=DS_EMBED_MODEL,
        dashscope_timeout=DS_TIMEOUT,
        deepseek_api_key=DEEPSEEK_API_KEY,
        deepseek_base_url=DEEPSEEK_BASE_URL,
        deepseek_model=DEEPSEEK_MODEL,
    )
    print(f"  Chat:     {'可用' if providers.chat_client else '不可用'}")
    print(f"  Analysis: {'可用' if providers.analysis_client else '不可用'}")
    print(f"  Embed:    {'可用' if providers.embedding_client else '不可用'}")

    # ── 2. 初始化 Agent 并构建知识库 ──
    print("\n[2/5] 构建知识库（解析 PDF + 分块）...")
    agent = CityScholarAgent(
        raw_papers_dir=str(RAW_PAPERS_DIR),
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    # 注入 LLM 客户端（和 App.py 一样的模式）
    agent.chat_client = providers.chat_client
    agent.analysis_client = providers.analysis_client

    kb = agent.build_knowledge_base()
    print(f"  论文数量: {len(kb.documents)}")
    print(f"  分块数量: {len(kb.chunk_records)}")
    if kb.parse_errors:
        print(f"   [!]解析错误 ({len(kb.parse_errors)}):")
        for err in kb.parse_errors[:3]:
            print(f"    - {err}")

    # ── 3. 元数据增强 ──
    if agent.chat_client:
        print("\n[3/5] LLM 元数据增强（提取标题/作者/DOI 等）...")
        try:
            agent.enrich_metadata()
            enriched = sum(1 for d in kb.documents if d.metadata.title)
            print(f"  完成：{enriched}/{len(kb.documents)} 篇")
        except Exception as e:
            print(f"   [!]出错（继续）: {e}")
    else:
        print("\n[3/5] 跳过元数据增强（无可用 LLM）。")

    # ── 4. 运行工作流生成 Markdown 报告 ──
    print("\n[4/5] 运行综述工作流...")

    topic = "城市韧性与可持续发展：方法、概念与实证的多维审视"
    print(f"  主题: {topic}")

    try:
        result = agent.run_review_workflow(
            topic=topic,
            targets=None,          # 自动选 8 篇
            output_dir=str(OUTPUT_DIR),
        )
        print(f"  状态: {result.status_message}")
        if result.workflow_result:
            for log in result.workflow_result.state.step_logs:
                print(f"    {log}")
    except Exception as e:
        print(f"[ERROR]工作流出错: {e}")

    # ── 5. 导出知识图谱 ──
    print("\n[5/5] 导出知识图谱...")
    try:
        from tools.graph_tool import build_knowledge_graph, export_graph_html, export_graph_png

        docs = agent.knowledge_base.documents
        if len(docs) >= 2:
            selected = docs[:8]
            print(f"  基于 {len(selected)} 篇论文构建图谱...")

            # 比较 + 提纲
            comp_resp = agent.compare_papers(
                targets=[d.document_id for d in selected],
                topic_hint="城市韧性与可持续发展",
            )
            outline_resp = agent.generate_review_outline(
                topic="城市韧性与可持续发展",
                targets=[d.document_id for d in selected],
            )

            if comp_resp.comparison is None:
                print("[ERROR]比较结果为空，跳过高谱生成。")
                return

            # 构建知识图谱
            graph = build_knowledge_graph(
                comparison=comp_resp.comparison,
                outline=outline_resp.outline,
                title="城市韧性与可持续发展 — 知识图谱",
            )

            # 导出 HTML
            html_path = str(OUTPUT_DIR / "knowledge_graph_demo.html")
            export_graph_html(graph, html_path)
            print(f"  [OK] HTML: {html_path}")

            # 导出 PNG
            png_path = str(OUTPUT_DIR / "knowledge_graph_demo.png")
            png_result = export_graph_png(graph, png_path)
            if png_result:
                print(f"  [OK] PNG:  {png_path}")
            else:
                print("   [!]PNG 导出失败（可能缺少 matplotlib/networkx）")
        else:
            print("  论文不足 2 篇，跳过。")
    except Exception as e:
        import traceback
        print(f"[ERROR]图谱出错: {e}")
        traceback.print_exc()

    # ── 完成 ──
    print("\n" + "=" * 60)
    print("重新生成完成！输出文件：")
    for f in sorted(OUTPUT_DIR.iterdir()):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name} ({size_kb:.1f} KB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
