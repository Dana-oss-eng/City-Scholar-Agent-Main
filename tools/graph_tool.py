"""本模块作用：在整个智能体中负责将论文比较与提纲结果构建为知识图谱，
并导出为可交互的 HTML 可视化图谱或静态 PNG 图片。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from tools.compare_tool import MultiPaperComparison, ComparisonPaperSummary
from tools.outline_tool import ReviewOutline, ReviewOutlineSection

# ====== 图谱数据结构 ======

@dataclass
class GraphNode:
    """图谱中的单个节点。"""
    id: str
    label: str
    node_type: str  # "paper", "theme", "method", "data", "finding", "implication"
    group: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class GraphEdge:
    """图谱中的一条边。"""
    source: str
    target: str
    label: str = ""
    weight: float = 1.0


@dataclass
class KnowledgeGraph:
    """完整的知识图谱数据结构。"""
    title: str
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)


# ====== 图谱构建 ======

_COLORS = {
    "paper": "#4A90D9",
    "theme": "#E87332",
    "method": "#50B86C",
    "data": "#9B59B6",
    "finding": "#F1C40F",
    "implication": "#E74C3C",
    "section": "#1ABC9C",
}

_GROUPS = {
    "paper": "论文",
    "theme": "共同主题",
    "method": "研究方法",
    "data": "数据来源",
    "finding": "研究发现",
    "implication": "综合启示",
    "section": "提纲章节",
}


def _sanitize_id(text: str) -> str:
    """将任意文本清洗为合法的节点 ID。"""
    cleaned = re.sub(r"[^\w一-鿿]+", "_", text.strip())
    return cleaned.strip("_") or "node"


def _shorten(text: str, max_len: int = 60) -> str:
    """截断过长文本，用于节点标签。"""
    t = re.sub(r"\s+", " ", text).strip()
    if len(t) <= max_len:
        return t
    return t[:max_len].rstrip() + "…"


def build_knowledge_graph(
    comparison: MultiPaperComparison,
    outline: ReviewOutline | None = None,
    title: str = "知识图谱",
) -> KnowledgeGraph:
    """根据多篇论文比较结果和可选提纲构建知识图谱。

    输入：
        comparison: 多篇论文比较结果。
        outline: 可选，综述提纲。
        title: 图谱标题。
    输出：
        完整的知识图谱对象。
    异常：
        无。
    """

    graph = KnowledgeGraph(title=title)
    seen_nodes: set[str] = set()
    paper_ids: list[str] = []

    # 1. 论文节点
    for summary in comparison.paper_summaries:
        pid = _sanitize_id(summary.file_name)
        paper_ids.append(pid)
        if pid not in seen_nodes:
            seen_nodes.add(pid)
            graph.nodes.append(GraphNode(
                id=pid,
                label=_shorten(summary.file_name.replace(".pdf", ""), 50),
                node_type="paper",
                group="paper",
                metadata={
                    "file_name": summary.file_name,
                    "research_question": _shorten(summary.research_question, 80),
                    "research_object": _shorten(summary.research_object, 80),
                },
            ))

    # 2. 共同主题节点 + 边
    for i, theme in enumerate(comparison.common_themes):
        tid = f"theme_{i}"
        if tid not in seen_nodes:
            seen_nodes.add(tid)
            graph.nodes.append(GraphNode(
                id=tid,
                label=_shorten(theme, 50),
                node_type="theme",
                group="theme",
            ))
        for pid in paper_ids:
            graph.edges.append(GraphEdge(
                source=pid, target=tid,
                label="涉及", weight=0.6,
            ))

    # 3. 方法比较 → 方法节点
    for i, method_text in enumerate(comparison.method_comparison):
        # 提取方法关键词
        mid = f"method_{i}"
        label = _shorten(method_text.split("：", 1)[-1] if "：" in method_text else method_text, 45)
        if mid not in seen_nodes:
            seen_nodes.add(mid)
            graph.nodes.append(GraphNode(
                id=mid,
                label=label,
                node_type="method",
                group="method",
            ))
        # 连接到对应论文
        if i < len(paper_ids):
            graph.edges.append(GraphEdge(
                source=paper_ids[i], target=mid,
                label="使用", weight=0.7,
            ))

    # 4. 数据来源节点
    for i, data_text in enumerate(comparison.data_comparison):
        did = f"data_{i}"
        label = _shorten(data_text.split("：", 1)[-1] if "：" in data_text else data_text, 45)
        if did not in seen_nodes:
            seen_nodes.add(did)
            graph.nodes.append(GraphNode(
                id=did,
                label=label,
                node_type="data",
                group="data",
            ))
        if i < len(paper_ids):
            graph.edges.append(GraphEdge(
                source=paper_ids[i], target=did,
                label="基于", weight=0.7,
            ))

    # 5. 研究发现节点
    for i, finding_text in enumerate(comparison.finding_comparison):
        fid = f"finding_{i}"
        label = _shorten(finding_text.split("：", 1)[-1] if "：" in finding_text else finding_text, 45)
        if fid not in seen_nodes:
            seen_nodes.add(fid)
            graph.nodes.append(GraphNode(
                id=fid,
                label=label,
                node_type="finding",
                group="finding",
            ))
        if i < len(paper_ids):
            graph.edges.append(GraphEdge(
                source=paper_ids[i], target=fid,
                label="得出", weight=0.8,
            ))

    # 6. 综合启示节点
    for i, impl_text in enumerate(comparison.integrated_implications):
        iid = f"implication_{i}"
        label = _shorten(impl_text.split("提示：", 1)[-1] if "提示：" in impl_text else impl_text, 45)
        if iid not in seen_nodes:
            seen_nodes.add(iid)
            graph.nodes.append(GraphNode(
                id=iid,
                label=label,
                node_type="implication",
                group="implication",
            ))
        # 连接到所有论文
        for pid in paper_ids:
            graph.edges.append(GraphEdge(
                source=pid, target=iid,
                label="启示", weight=0.5,
            ))

    # 7. 提纲章节节点（如有）
    if outline is not None:
        for i, section in enumerate(outline.sections):
            sid = f"section_{i}"
            if sid not in seen_nodes:
                seen_nodes.add(sid)
                graph.nodes.append(GraphNode(
                    id=sid,
                    label=_shorten(section.title, 50),
                    node_type="section",
                    group="section",
                    metadata={"bullets": section.bullets},
                ))
            # 章节连接到相关论文
            for pid in paper_ids:
                graph.edges.append(GraphEdge(
                    source=sid, target=pid,
                    label="参考", weight=0.4,
                ))

    # 8. 论文间共现边（共享主题的论文互连）
    for i, pid_a in enumerate(paper_ids):
        for pid_b in paper_ids[i + 1:]:
            graph.edges.append(GraphEdge(
                source=pid_a, target=pid_b,
                label="同主题", weight=0.3,
            ))

    return graph


# ====== 自包含 HTML 可视化导出（无需额外依赖） ======

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — 知识图谱</title>
<script src="https://unpkg.com/vis-network@9.1.6/dist/vis-network.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif; background: #f5f7fa; }}
  .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); color: white; padding: 18px 24px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; }}
  .header h1 {{ font-size: 20px; font-weight: 600; }}
  .header .stats {{ font-size: 13px; opacity: 0.85; }}
  .container {{ display: flex; height: calc(100vh - 80px); }}
  #graph {{ flex: 1; background: white; }}
  .sidebar {{ width: 280px; background: white; border-left: 1px solid #e0e4e8; padding: 16px; overflow-y: auto; font-size: 13px; }}
  .sidebar h3 {{ font-size: 15px; margin-bottom: 12px; color: #1a1a2e; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 12px; }}
  .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }}
  .search-box {{ width: 100%; padding: 8px 12px; border: 1px solid #d0d5dd; border-radius: 6px; font-size: 13px; margin-bottom: 12px; outline: none; }}
  .search-box:focus {{ border-color: #4A90D9; box-shadow: 0 0 0 3px rgba(74,144,217,0.15); }}
  .node-list {{ list-style: none; }}
  .node-list li {{ padding: 6px 8px; cursor: pointer; border-radius: 4px; margin-bottom: 2px; transition: background 0.15s; }}
  .node-list li:hover {{ background: #f0f4ff; }}
  .node-list li .dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }}
  .toolbar {{ display: flex; gap: 6px; margin-bottom: 12px; flex-wrap: wrap; }}
  .toolbar button {{ padding: 6px 12px; border: 1px solid #d0d5dd; border-radius: 6px; background: white; cursor: pointer; font-size: 12px; transition: all 0.15s; }}
  .toolbar button:hover {{ background: #f0f4ff; border-color: #4A90D9; }}
  .toolbar button.active {{ background: #4A90D9; color: white; border-color: #4A90D9; }}
  #detail-panel {{ margin-top: 12px; padding: 10px; background: #f8f9fb; border-radius: 6px; font-size: 12px; display: none; }}
  #detail-panel h4 {{ margin-bottom: 6px; color: #1a1a2e; }}
  #detail-panel p {{ color: #555; line-height: 1.5; }}
</style>
</head>
<body>
<div class="header">
  <div>
    <h1>📊 {title} — 知识图谱</h1>
    <div class="stats">节点: {node_count} &nbsp;|&nbsp; 边: {edge_count} &nbsp;|&nbsp; 论文: {paper_count}</div>
  </div>
  <div class="toolbar" id="toolbar">
    <button onclick="resetView()" title="重置视图">🔄 重置</button>
    <button onclick="togglePhysics()" title="暂停/恢复布局动画">⏯️ 物理引擎</button>
    <button onclick="exportPNG()" title="导出为 PNG 图片">📸 截图</button>
    <button onclick="fitAll()" title="适应窗口">🔍 全部显示</button>
  </div>
</div>
<div class="container">
  <div id="graph"></div>
  <div class="sidebar">
    <h3>📋 图例</h3>
    <div class="legend">
      {legend_items}
    </div>
    <h3>🔍 搜索节点</h3>
    <input type="text" class="search-box" id="search" placeholder="输入节点名称筛选…" oninput="filterNodes()">
    <ul class="node-list" id="node-list"></ul>
    <div id="detail-panel">
      <h4 id="detail-title"></h4>
      <p id="detail-content"></p>
    </div>
  </div>
</div>
<script>
  var nodes = new vis.DataSet({nodes_json});
  var edges = new vis.DataSet({edges_json});

  var container = document.getElementById('graph');
  var data = {{ nodes: nodes, edges: edges }};
  var options = {{
    nodes: {{
      shape: 'dot',
      font: {{ size: 13, face: 'Microsoft YaHei, sans-serif', color: '#333' }},
      borderWidth: 2,
      shadow: {{ enabled: true, size: 8 }},
    }},
    edges: {{
      font: {{ size: 10, face: 'Microsoft YaHei, sans-serif', color: '#999', strokeWidth: 0 }},
      arrows: {{ to: {{ enabled: true, scaleFactor: 0.5 }} }},
      smooth: {{ type: 'continuous' }},
      color: {{ opacity: 0.45 }},
    }},
    physics: {{
      solver: 'forceAtlas2Based',
      forceAtlas2Based: {{
        gravitationalConstant: -40,
        centralGravity: 0.005,
        springLength: 180,
        springConstant: 0.08,
        damping: 0.4,
      }},
      stabilization: {{ iterations: 200 }},
    }},
    interaction: {{
      hover: true,
      tooltipDelay: 150,
      zoomView: true,
      dragView: true,
      navigationButtons: true,
    }},
    layout: {{ improvedLayout: true }},
  }};
  var network = new vis.Network(container, data, options);

  // 节点点击：高亮邻居并显示详情
  network.on('click', function(params) {{
    if (params.nodes.length > 0) {{
      var nodeId = params.nodes[0];
      var node = nodes.get(nodeId);
      var connected = network.getConnectedNodes(nodeId);
      var allNodes = nodes.get();
      allNodes.forEach(function(n) {{
        nodes.update({{ id: n.id, opacity: (n.id === nodeId || connected.includes(n.id)) ? 1.0 : 0.2 }});
      }});
      var panel = document.getElementById('detail-panel');
      document.getElementById('detail-title').textContent = node.label;
      var meta = node.metadata || {{}};
      var info = [];
      if (meta.file_name) info.push('文件: ' + meta.file_name);
      if (meta.research_question && meta.research_question !== 'undefined') info.push('研究问题: ' + meta.research_question);
      if (meta.research_object && meta.research_object !== 'undefined') info.push('研究对象: ' + meta.research_object);
      if (meta.bullets && meta.bullets.length) info.push('要点: ' + meta.bullets.slice(0, 3).join('; '));
      document.getElementById('detail-content').textContent = info.join('\\n') || '无额外信息';
      panel.style.display = 'block';
    }} else {{
      resetHighlight();
    }}
  }});

  // 双击重置高亮
  network.on('doubleClick', resetHighlight);

  function resetHighlight() {{
    nodes.forEach(function(n) {{ nodes.update({{ id: n.id, opacity: 1.0 }}); }});
    document.getElementById('detail-panel').style.display = 'none';
  }}

  // 搜索过滤
  function filterNodes() {{
    var q = document.getElementById('search').value.toLowerCase();
    var list = document.getElementById('node-list');
    var allNodes = nodes.get();
    list.innerHTML = '';
    allNodes.forEach(function(n) {{
      if (n.label && n.label.toLowerCase().includes(q)) {{
        var li = document.createElement('li');
        li.innerHTML = '<span class="dot" style="background:' + (n.color && n.color.background ? n.color.background : '#999') + '"></span>' + n.label;
        li.onclick = function() {{ network.focus(n.id, {{ scale: 1.5, animation: true }}); network.selectNodes([n.id]); }};
        list.appendChild(li);
      }}
    }});
  }}

  // 初始化节点列表
  filterNodes();

  // 工具栏功能
  var physicsOn = true;
  function togglePhysics() {{
    physicsOn = !physicsOn;
    network.setOptions({{ physics: physicsOn }});
    var btn = document.querySelectorAll('#toolbar button')[1];
    btn.textContent = physicsOn ? '⏯️ 物理引擎' : '▶️ 启动布局';
    btn.classList.toggle('active', !physicsOn);
  }}

  function resetView() {{
    resetHighlight();
    network.fit({{ animation: true }});
    if (!physicsOn) togglePhysics();
  }}

  function fitAll() {{
    network.fit({{ animation: {{ duration: 600 }} }});
  }}

  function exportPNG() {{
    var canvas = document.querySelector('#graph canvas');
    if (canvas) {{
      var link = document.createElement('a');
      link.download = '{title_safe}_知识图谱.png';
      link.href = canvas.toDataURL('image/png');
      link.click();
    }} else {{
      alert('无法获取画布，请稍后重试。');
    }}
  }}

  // 响应式
  window.addEventListener('resize', function() {{ network.fit(); }});

  // 加载完成
  network.once('stabilizationIterationsDone', function() {{
    network.fit({{ animation: true }});
  }});
</script>
</body>
</html>"""


def _build_node_json(node: GraphNode) -> dict:
    """将 GraphNode 转为 vis.js 节点 JSON。"""
    color = _COLORS.get(node.node_type, "#999999")
    size_map = {
        "paper": 32,
        "theme": 22,
        "method": 18,
        "data": 18,
        "finding": 18,
        "implication": 16,
        "section": 20,
    }
    return {
        "id": node.id,
        "label": node.label,
        "group": node.group,
        "color": {"background": color, "border": _darken(color), "highlight": {"background": color, "border": _darken(color)}},
        "size": size_map.get(node.node_type, 16),
        "font": {"size": 14 if node.node_type == "paper" else 12, "bold": node.node_type == "paper"},
        "metadata": {k: str(v) if not isinstance(v, list) else v for k, v in node.metadata.items()},
    }


def _darken(hex_color: str, factor: float = 0.8) -> str:
    """将 hex 颜色加深。"""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r, g, b = int(r * factor), int(g * factor), int(b * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def _build_edge_json(edge: GraphEdge) -> dict:
    """将 GraphEdge 转为 vis.js 边 JSON。"""
    width_map = {0.8: 3, 0.7: 2.5, 0.6: 2, 0.5: 1.5, 0.4: 1, 0.3: 0.5}
    return {
        "from": edge.source,
        "to": edge.target,
        "label": edge.label,
        "width": width_map.get(round(edge.weight, 1), max(0.5, edge.weight * 3)),
        "title": f"{edge.label} (权重: {edge.weight:.2f})" if edge.label else "",
    }


def export_graph_html(graph: KnowledgeGraph, output_path: str | Path) -> str:
    """将知识图谱导出为自包含的交互式 HTML 可视化文件（无需额外依赖）。

    输入：
        graph: 知识图谱对象。
        output_path: 输出 HTML 文件路径。
    输出：
        实际写入的文件路径字符串。
    异常：
        当文件写入失败时，抛出 OSError。
    """

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    nodes_json = json.dumps(
        [_build_node_json(n) for n in graph.nodes],
        ensure_ascii=False,
    )
    edges_json = json.dumps(
        [_build_edge_json(e) for e in graph.edges],
        ensure_ascii=False,
    )

    paper_count = sum(1 for n in graph.nodes if n.node_type == "paper")

    legend_items = "\n".join(
        f'<div class="legend-item"><span class="legend-dot" style="background:{_COLORS[t]}"></span>{_GROUPS[t]}</div>'
        for t in ["paper", "theme", "method", "data", "finding", "implication", "section"]
    )

    title_safe = re.sub(r"[^\w一-鿿]+", "_", graph.title).strip("_")

    html = _HTML_TEMPLATE.format(
        title=graph.title,
        title_safe=title_safe,
        node_count=len(graph.nodes),
        edge_count=len(graph.edges),
        paper_count=paper_count,
        legend_items=legend_items,
        nodes_json=nodes_json,
        edges_json=edges_json,
    )

    path.write_text(html, encoding="utf-8")
    return str(path)


# ====== 静态 PNG 导出（可选，需 matplotlib + networkx） ======

def export_graph_png(graph: KnowledgeGraph, output_path: str | Path) -> str | None:
    """将知识图谱导出为静态 PNG 图片（需要 matplotlib 和 networkx）。"""

    try:
        import networkx as nx
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
    except ImportError:
        return None

    # 中文字体检测
    for f in fm.findSystemFonts():
        fl = f.lower()
        if any(k in fl for k in ("msyh", "microsoft yahei", "simhei", "simsun", "noto sans cjk sc", "wqy")):
            fm.fontManager.addfont(f)
            prop = fm.FontProperties(fname=f)
            plt.rcParams["font.sans-serif"] = [prop.get_name(), "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            break

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    G = nx.Graph()
    for node in graph.nodes:
        G.add_node(node.id, label=node.label, node_type=node.node_type)
    for edge in graph.edges:
        G.add_edge(edge.source, edge.target, weight=edge.weight, label=edge.label)

    node_colors = [_COLORS.get(G.nodes[n].get("node_type", ""), "#999") for n in G.nodes]
    node_sizes = [1400 if G.nodes[n].get("node_type") == "paper" else 600 for n in G.nodes]

    plt.figure(figsize=(22, 16))
    plt.title(graph.title, fontsize=18, fontweight="bold", pad=12)

    pos = nx.spring_layout(G, k=2.8, iterations=50, seed=42)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                           alpha=0.9, edgecolors="#333", linewidths=1)
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color="#999", width=1.0)

    # 标签 — 白色圆角背景，确保中文清晰可读
    labels = {n: G.nodes[n].get("label", n)[:30] for n in G.nodes}
    bbox = dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.85, edgecolor="#ccc", linewidth=0.5)
    nx.draw_networkx_labels(G, pos, labels, font_size=9, bbox=bbox)

    # 图例
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=_COLORS[t], markersize=10, label=_GROUPS[t])
        for t in ["paper", "theme", "method", "data", "finding", "implication", "section"]
    ]
    plt.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.9)

    plt.tight_layout(pad=1)
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()

    return str(path)


# ====== 图谱摘要文本 ======

def format_graph_summary(graph: KnowledgeGraph) -> str:
    """将知识图谱整理为可读的文本摘要。

    输入：
        graph: 知识图谱对象。
    输出：
        图谱摘要文本。
    异常：
        无。
    """

    type_counts: dict[str, int] = {}
    for node in graph.nodes:
        type_counts[node.node_type] = type_counts.get(node.node_type, 0) + 1

    lines = [
        f"知识图谱：「{graph.title}」",
        f"节点总数：{len(graph.nodes)}",
        f"边总数：{len(graph.edges)}",
        "节点分布：",
    ]
    for t in ["paper", "theme", "method", "data", "finding", "implication", "section"]:
        if t in type_counts:
            lines.append(f"  - {_GROUPS.get(t, t)}：{type_counts[t]} 个")

    lines.append("\n核心论文：")
    for node in graph.nodes:
        if node.node_type == "paper":
            lines.append(f"  - {node.label}")
            if node.metadata.get("research_question"):
                lines.append(f"    研究问题：{node.metadata['research_question']}")

    return "\n".join(lines)


# ====== 演示 ======

def run_graph_demo() -> None:
    """执行 graph_tool 模块的最小演示。

    输入：
        无。
    输出：
        无。函数会直接打印图谱摘要并导出 HTML。
    异常：
        无。
    """

    from tools.compare_tool import MultiPaperComparison, ComparisonPaperSummary
    from tools.outline_tool import ReviewOutline, ReviewOutlineSection

    # 构造模拟比较结果
    comparison = MultiPaperComparison(
        topic_hint="城市韧性与安全治理",
        paper_summaries=[
            ComparisonPaperSummary(
                file_name="paper_a.pdf",
                document_id="paper_a",
                research_question="城市韧性如何评估？",
                research_object="沿海城市群",
                methods="指标评价与空间分析",
                data_source="统计年鉴和遥感数据",
                key_findings="基础设施韧性与治理协同能力显著相关",
                limitations="样本集中于中心城区",
                implications="提升跨区域协同治理能力",
            ),
            ComparisonPaperSummary(
                file_name="paper_b.pdf",
                document_id="paper_b",
                research_question="城市安全治理如何影响韧性？",
                research_object="内陆都市圈",
                methods="问卷调查与回归分析",
                data_source="问卷、POI 数据与统计资料",
                key_findings="公共服务可达性影响城市安全感知",
                limitations="外部可推广性受限",
                implications="公共服务规划与安全治理协同推进",
            ),
            ComparisonPaperSummary(
                file_name="paper_c.pdf",
                document_id="paper_c",
                research_question="数字化如何提升城市韧性？",
                research_object="智慧城市试点区域",
                methods="数字孪生建模与仿真分析",
                data_source="IoT 传感器和城市大数据平台",
                key_findings="数字孪生技术显著提升应急响应效率",
                limitations="技术成熟度与数据隐私",
                implications="加速城市数字化转型与韧性建设融合",
            ),
        ],
        common_themes=[
            "多篇论文都涉及「城市韧性」相关议题。",
            "多篇论文都涉及「治理协同」相关议题。",
            "多篇论文都涉及「数据分析方法」相关议题。",
        ],
        method_comparison=[
            "《paper_a.pdf》：指标评价与空间分析方法",
            "《paper_b.pdf》：问卷调查与回归分析方法",
            "《paper_c.pdf》：数字孪生建模与仿真分析",
        ],
        data_comparison=[
            "《paper_a.pdf》：统计年鉴和遥感数据",
            "《paper_b.pdf》：问卷、POI 数据与统计资料",
            "《paper_c.pdf》：IoT 传感器和城市大数据平台",
        ],
        finding_comparison=[
            "《paper_a.pdf》：基础设施韧性与治理协同能力显著相关",
            "《paper_b.pdf》：公共服务可达性影响城市安全感知",
            "《paper_c.pdf》：数字孪生显著提升应急响应效率",
        ],
        integrated_implications=[
            "《paper_a.pdf》提示：提升跨区域协同治理能力",
            "《paper_b.pdf》提示：公共服务规划与安全治理协同",
            "《paper_c.pdf》提示：加速城市数字化转型与韧性建设融合",
        ],
    )

    outline = ReviewOutline(
        topic="城市韧性与安全治理研究综述",
        source_papers=["paper_a.pdf", "paper_b.pdf", "paper_c.pdf"],
        sections=[
            ReviewOutlineSection(
                title="研究背景与问题提出",
                bullets=["城市韧性研究缘起", "安全治理现实需求"],
            ),
            ReviewOutlineSection(
                title="常用方法与数据来源",
                bullets=["指标评价法", "问卷回归法", "数字孪生法"],
            ),
            ReviewOutlineSection(
                title="主要发现与启示",
                bullets=["韧性与治理协同", "公共服务可达性", "数字化转型"],
            ),
        ],
    )

    graph = build_knowledge_graph(comparison, outline, title="城市韧性与安全治理 — 知识图谱")

    print("=" * 60)
    print(format_graph_summary(graph))
    print("=" * 60)

    # 导出 HTML
    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = export_graph_html(graph, output_dir / "knowledge_graph_demo.html")
    print(f"\n[OK] 交互式 HTML 图谱已导出：{html_path}")

    # 尝试导出 PNG
    png_path = export_graph_png(graph, output_dir / "knowledge_graph_demo.png")
    if png_path:
        print(f"[OK] 静态 PNG 图谱已导出：{png_path}")
    else:
        print("[WARN] 未安装 matplotlib/networkx，跳过 PNG 导出。请执行 pip install matplotlib networkx 以启用。")

    print("\n[TIP] 用浏览器打开 HTML 文件即可交互式浏览知识图谱。")
    print("   支持节点拖拽、缩放、搜索、点击高亮邻居、截图导出等功能。")


if __name__ == "__main__":
    run_graph_demo()
