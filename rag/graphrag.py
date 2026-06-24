"""本模块作用：构建论文实体共现图，提供图增强检索能力（最小 GraphRAG 实现）。

核心思路：
1. 从所有 chunk 中抽取关键实体（方法名、城市名、概念词、技术词）
2. 构建实体共现图：同一篇论文中共同出现的实体建立边
3. 检索时：从问题中提取实体 → 在图上游走邻居 → 召回邻居实体关联的 chunk
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field


# ====== 实体抽取规则 ======

ENTITY_PATTERNS: dict[str, str] = {
    "method": r"(?:随机森林|支持向量机|神经网络|深度学习|机器学习|回归分析|空间分析"
              r"|TOPSIS|层次分析|AHP|主成分分析|PCA|聚类|随机森林|XGBoost|LSTM|CNN|GNN"
              r"|random forest|svm|neural network|deep learning|machine learning"
              r"|regression|spatial analysis|time series|agent.?based)",
    "concept": r"(?:韧性|resilience|可持续|sustainable|气候变化|climate|脆弱性|vulnerability"
               r"|适应性|adaptation|城市群|agglomeration|碳排|carbon|洪水|flood"
               r"|热岛|heat island|可达性|accessibility|安全|safety|perception)",
    "tech": r"(?:数字孪生|digital twin|IoT|物联网|6G|大语言模型|LLM|large language model"
             r"|遥感|remote sensing|GIS|地理信息|POI|GPS|轨迹|trajectory)",
    "city": r"(?:长三角|珠三角|京津冀|成渝|Yangtze|Pearl River|Beijing|上海|北京|深圳|广州"
            r"|成都|重庆|European|Europe|China|中国|美国|US)",
}

STOP_ENTITIES = {"研究", "分析", "模型", "方法", "数据", "结果", "影响", "基于",
                 "used", "using", "based", "study", "paper", "analysis", "model", "data"}


def extract_entities(text: str) -> dict[str, list[str]]:
    """从文本中按类别抽取实体。"""
    found: dict[str, list[str]] = {}
    for category, pattern in ENTITY_PATTERNS.items():
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        cleaned = [m.strip().lower() for m in matches if m.strip().lower() not in STOP_ENTITIES]
        if cleaned:
            # 去重并取前 10 个
            seen = set()
            unique = []
            for e in cleaned:
                if e not in seen:
                    seen.add(e)
                    unique.append(e)
            found[category] = unique[:10]
    return found


# ====== 图结构 ======

@dataclass
class EntityGraph:
    """实体共现图。"""
    nodes: dict[str, dict] = field(default_factory=dict)       # entity_name → {type, papers: set, frequency}
    edges: dict[str, dict[str, float]] = field(default_factory=dict)  # entity_a → {entity_b: weight}
    entity_to_chunks: dict[str, list[int]] = field(default_factory=dict)  # entity → chunk indices

    def add_node(self, name: str, etype: str, paper_id: str) -> None:
        if name not in self.nodes:
            self.nodes[name] = {"type": etype, "papers": set(), "frequency": 0}
        self.nodes[name]["papers"].add(paper_id)
        self.nodes[name]["frequency"] += 1

    def add_edge(self, a: str, b: str) -> None:
        if a == b:
            return
        if a not in self.edges:
            self.edges[a] = {}
        self.edges[a][b] = self.edges[a].get(b, 0.0) + 1.0

    def get_neighbors(self, entity: str, top_k: int = 10) -> list[tuple[str, float]]:
        """获取某实体的邻居实体及权重。"""
        if entity not in self.edges:
            return []
        neighbors = sorted(self.edges[entity].items(), key=lambda x: -x[1])
        return neighbors[:top_k]

    def get_chunks_for_entity(self, entity: str) -> list[int]:
        """获取某实体关联的 chunk 索引列表。"""
        return self.entity_to_chunks.get(entity, [])


def build_entity_graph(chunk_records: list[dict]) -> EntityGraph:
    """从知识库 chunk 记录构建实体共现图。

    复杂度 O(N * E)，N=chunk数，E=每chunk实体数（小）。
    """
    graph = EntityGraph()

    for idx, rec in enumerate(chunk_records):
        text = str(rec.get("text", ""))
        meta = rec.get("metadata", {}) if isinstance(rec.get("metadata"), dict) else {}
        doc_id = str(meta.get("document_id", str(rec.get("document_id", ""))))

        # 抽取实体
        entities_by_type = extract_entities(text)
        all_entities: list[str] = []
        for etype, entities in entities_by_type.items():
            for e in entities:
                all_entities.append(e)
                graph.add_node(e, etype, doc_id)
                if e not in graph.entity_to_chunks:
                    graph.entity_to_chunks[e] = []
                if idx not in graph.entity_to_chunks[e]:
                    graph.entity_to_chunks[e].append(idx)

        # 同一 chunk 内共现的实体建边
        for i, ea in enumerate(all_entities):
            for eb in all_entities[i + 1:]:
                graph.add_edge(ea, eb)
                graph.add_edge(eb, ea)

    return graph


def graph_enhanced_retrieval(
    question: str,
    graph: EntityGraph,
    chunk_records: list[dict],
    base_chunk_indices: set[int],
    top_k: int = 5,
) -> list[int]:
    """图增强检索：从问题提取实体 → 找邻居 → 召回关联 chunk。

    Args:
        question: 用户问题
        graph: 已构建的实体图
        chunk_records: 全部 chunk 记录
        base_chunk_indices: 基础检索已召回的 chunk 索引集合
        top_k: 最终返回数量

    Returns:
        增强后的 chunk 索引列表
    """
    # 抽取问题中的实体
    q_entities_map = extract_entities(question)
    q_entities: list[str] = []
    for entities in q_entities_map.values():
        q_entities.extend(entities)

    if not q_entities:
        return list(base_chunk_indices)[:top_k]

    # 在图上游走：找到问题实体及其邻居
    expanded_entities: set[str] = set(q_entities)
    for entity in q_entities:
        neighbors = graph.get_neighbors(entity, top_k=5)
        for neighbor, weight in neighbors:
            if weight >= 2:  # 至少共现 2 次才算可靠关联
                expanded_entities.add(neighbor)

    # 召回邻居实体关联的 chunk
    graph_chunk_indices: set[int] = set()
    for entity in expanded_entities:
        chunk_indices = graph.get_chunks_for_entity(entity)
        graph_chunk_indices.update(chunk_indices)

    # 合并：基础检索结果在前，图检索结果补充
    merged = list(base_chunk_indices) + [i for i in graph_chunk_indices if i not in base_chunk_indices]
    return merged[:top_k]


def format_graph_context(question: str, graph: EntityGraph) -> str:
    """将图检索到的实体上下文格式化为文本。"""
    q_entities_map = extract_entities(question)
    q_entities: list[str] = []
    for entities in q_entities_map.values():
        q_entities.extend(entities)

    if not q_entities:
        return ""

    lines = ["[知识图谱上下文 — 相关问题域实体]"]
    seen = set(q_entities)
    for entity in q_entities:
        etype = graph.nodes.get(entity, {}).get("type", "?")
        freq = graph.nodes.get(entity, {}).get("frequency", 0)
        lines.append(f"  [{etype}] {entity}（出现 {freq} 次）")

        neighbors = graph.get_neighbors(entity, top_k=4)
        if neighbors:
            n_str = "、".join(f"{n}（{w:.0f}次共现）" for n, w in neighbors if w >= 2)
            if n_str:
                lines.append(f"    ↳ 关联：{n_str}")
                for n, _ in neighbors:
                    seen.add(n)
    return "\n".join(lines)


def get_graph_stats(graph: EntityGraph) -> str:
    """返回图统计信息。"""
    lines = [
        f"实体节点数：{len(graph.nodes)}",
        f"边数量：{sum(len(v) for v in graph.edges.values()) // 2}",
        f"实体类型分布：",
    ]
    type_counts: Counter[str] = Counter()
    for node_data in graph.nodes.values():
        type_counts[node_data["type"]] += 1
    for etype, count in type_counts.most_common():
        lines.append(f"  - {etype}: {count}")
    return "\n".join(lines)


# ====== 图可视化导出 ======

# 实体类型的 Mermaid 颜色映射
_ETYPE_COLORS: dict[str, str] = {
    "method": "#4A90D9",   # 蓝色 — 方法
    "concept": "#7B68EE",  # 紫色 — 概念
    "tech": "#2ECC71",     # 绿色 — 技术
    "city": "#E67E22",     # 橙色 — 城市
}

# Mermaid 子图标题
_ETYPE_LABELS: dict[str, str] = {
    "method": "方法 Methods",
    "concept": "概念 Concepts",
    "tech": "技术 Tech",
    "city": "城市 Cities",
}


def export_graph_mermaid(graph: EntityGraph, top_n: int = 60, min_weight: int = 2) -> str:
    """将实体共现图导出为 Mermaid.js flowchart 格式，可嵌入 Markdown 渲染为可视化图谱。

    Args:
        graph: 已构建的实体图
        top_n: 最多展示的节点数量（防止图过于密集）
        min_weight: 边权重下限，低于此值的边不展示

    Returns:
        Mermaid flowchart 代码块（可直接嵌入 .md 文件）
    """
    if not graph.nodes:
        return "%% 图谱为空，无可用数据。"

    # 按频率排序，取 top_n 节点
    sorted_nodes = sorted(graph.nodes.items(), key=lambda x: -x[1].get("frequency", 0))
    visible_nodes = sorted_nodes[:top_n]
    visible_set = {name for name, _ in visible_nodes}

    lines = [
        "```mermaid",
        "graph TB",
        "%% ── 样式定义 ──",
        "classDef method fill:#4A90D9,color:#fff,stroke:#2C5F8A",
        "classDef concept fill:#7B68EE,color:#fff,stroke:#4B3E9E",
        "classDef tech fill:#2ECC71,color:#fff,stroke:#1F8C4D",
        "classDef city fill:#E67E22,color:#fff,stroke:#B85D14",
        "",
        "%% ── 子图：按类型分组 ──",
    ]

    # 生成子图（按实体类型分组）
    for etype in ["method", "concept", "tech", "city"]:
        etype_nodes = [(n, d) for n, d in visible_nodes if d.get("type") == etype]
        if not etype_nodes:
            continue
        label = _ETYPE_LABELS.get(etype, etype)
        lines.append(f"subgraph {etype}[{label}]")
        for name, data in etype_nodes:
            freq = data.get("frequency", 0)
            node_id = _sanitize_node_id(name)
            lines.append(f"    {node_id}[{name}<br/>×{freq}]")
        lines.append("end")
        lines.append("")

    # 生成边（仅包含可见节点之间的边，且权重大于等于 min_weight）
    lines.append("%% ── 共现边 ──")
    edge_added: set[tuple[str, str]] = set()
    for node_a, neighbors in graph.edges.items():
        if node_a not in visible_set:
            continue
        for node_b, weight in neighbors.items():
            if node_b not in visible_set:
                continue
            if weight < min_weight:
                continue
            # 去重（无向边只画一次）
            edge_key = tuple(sorted([node_a, node_b]))
            if edge_key in edge_added:
                continue
            edge_added.add(edge_key)
            a_id = _sanitize_node_id(node_a)
            b_id = _sanitize_node_id(node_b)
            # 边越粗表示共现越多
            stroke = min(int(weight), 5)
            lines.append(f"{a_id} ---|{int(weight)}| {b_id}")
            if stroke >= 4:
                lines[-1] = f"linkStyle {len(edge_added) - 1} stroke-width:{stroke}px"

    lines.append("```")
    lines.append("")
    lines.append(f"<!-- 图谱包含 {len(graph.nodes)} 个实体节点、{sum(len(v) for v in graph.edges.values()) // 2} 条边。")
    lines.append(f"  展示前 {len(visible_nodes)} 个高频节点。权重 ≥ {min_weight} 的边被显示。")
    lines.append(f"  在支持 Mermaid 的编辑器（如 VS Code、Typora、GitHub）中可直接渲染为可视化图谱。 -->")

    if len(graph.nodes) > top_n:
        lines.append(f"<!-- ⚠️ 共有 {len(graph.nodes)} 个实体，仅展示前 {top_n} 个高频节点。可通过调整 top_n 参数查看更多。 -->")

    return "\n".join(lines)


def export_graph_dot(graph: EntityGraph, top_n: int = 60, min_weight: int = 2) -> str:
    """将实体共现图导出为 Graphviz DOT 格式。

    Args:
        graph: 已构建的实体图
        top_n: 最多展示的节点数量
        min_weight: 边权重下限

    Returns:
        Graphviz DOT 源码
    """
    if not graph.nodes:
        return "// 图谱为空"

    sorted_nodes = sorted(graph.nodes.items(), key=lambda x: -x[1].get("frequency", 0))
    visible_nodes = sorted_nodes[:top_n]
    visible_set = {name for name, _ in visible_nodes}

    # DOT 颜色映射
    etype_colors_dot = {
        "method": "#4A90D9",
        "concept": "#7B68EE",
        "tech": "#2ECC71",
        "city": "#E67E22",
    }

    lines = [
        "// CityScholar-Agent 实体共现图 — Graphviz DOT",
        "// 用法：dot -Tpng graph.dot -o graph.png  或  dot -Tsvg graph.dot -o graph.svg",
        "digraph EntityGraph {",
        "    rankdir=LR;",
        '    node [shape=box, style="rounded,filled", fontname="sans-serif", fontsize=11];',
        '    edge [fontname="sans-serif", fontsize=9, color="#999999"];',
        "",
        "    // ── 实体节点 ──",
    ]

    for name, data in visible_nodes:
        node_id = _sanitize_node_id(name)
        etype = data.get("type", "")
        freq = data.get("frequency", 0)
        color = etype_colors_dot.get(etype, "#CCCCCC")
        fontsize = 9 + min(freq, 5)  # 频率越高字号越大
        lines.append(
            f'    {node_id} [label="{name}\\n(×{freq})", '
            f'fillcolor="{color}", fontsize={fontsize}];'
        )

    lines.append("")
    lines.append("    // ── 共现边 ──")

    edge_added: set[tuple[str, str]] = set()
    for node_a, neighbors in graph.edges.items():
        if node_a not in visible_set:
            continue
        for node_b, weight in neighbors.items():
            if node_b not in visible_set:
                continue
            if weight < min_weight:
                continue
            edge_key = tuple(sorted([node_a, node_b]))
            if edge_key in edge_added:
                continue
            edge_added.add(edge_key)
            a_id = _sanitize_node_id(node_a)
            b_id = _sanitize_node_id(node_b)
            penwidth = min(weight * 0.8, 5.0)
            lines.append(f'    {a_id} -> {b_id} [label="{int(weight)}", penwidth={penwidth:.1f}];')

    lines.append("}")
    return "\n".join(lines)


def export_graph_png(graph: EntityGraph, output_path, top_n: int = 60, min_weight: int = 2):
    """将实体共现图直接导出为 PNG 图片，中文标签清晰可读。

    Args:
        graph: 已构建的实体图
        output_path: 输出 PNG 文件路径
        top_n: 最多展示节点数
        min_weight: 边权重下限
    Returns:
        输出路径字符串；依赖缺失返回 None
    """
    try:
        import networkx as nx
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
    except ImportError:
        return None

    # 中文字体：扫描系统字体，优先用微软雅黑/黑体
    for f in fm.findSystemFonts():
        fl = f.lower()
        if any(k in fl for k in ("msyh", "microsoft yahei", "simhei", "simsun", "noto sans cjk sc", "wqy")):
            fm.fontManager.addfont(f)
            prop = fm.FontProperties(fname=f)
            plt.rcParams["font.sans-serif"] = [prop.get_name(), "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            break

    G = nx.Graph()
    sorted_nodes = sorted(graph.nodes.items(), key=lambda x: -x[1].get("frequency", 0))
    visible = {n for n, _ in sorted_nodes[:top_n]}

    for name, data in sorted_nodes[:top_n]:
        G.add_node(name, etype=data.get("type", ""), freq=data.get("frequency", 0))

    for a, neighbors in graph.edges.items():
        if a not in visible:
            continue
        for b, w in neighbors.items():
            if b not in visible or w < min_weight:
                continue
            G.add_edge(a, b, weight=w)

    if not G.nodes:
        return None

    n = len(G.nodes)
    fig, ax = plt.subplots(figsize=(max(16, n * 0.4), max(12, n * 0.3)))
    pos = nx.spring_layout(G, k=3.5, iterations=60, seed=42)

    _PNG_COLORS = {"method": "#4A90D9", "concept": "#7B68EE", "tech": "#2ECC71", "city": "#E67E22"}
    for etype, color in _PNG_COLORS.items():
        nodelist = [n for n in G.nodes if G.nodes[n]["etype"] == etype]
        if nodelist:
            sizes = [400 + G.nodes[n]["freq"] * 100 for n in nodelist]
            nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=color,
                                   node_size=sizes, alpha=0.9, edgecolors="#333", linewidths=1)

    widths = [max(0.3, G[u][v]["weight"] * 0.5) for u, v in G.edges]
    nx.draw_networkx_edges(G, pos, alpha=0.25, edge_color="#888", width=widths)

    labels = {n: f"{n}\n(×{G.nodes[n]['freq']})" for n in G.nodes}
    bbox = dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.85, edgecolor="#ccc", linewidth=0.5)
    nx.draw_networkx_labels(G, pos, labels, font_size=8, bbox=bbox)

    plt.tight_layout(pad=1)
    plt.savefig(str(output_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return str(output_path)


def _sanitize_node_id(name: str) -> str:
    """将实体名转换为合法的 Mermaid/DOT 节点 ID。"""
    import re
    # 只保留字母、数字和下划线
    safe = re.sub(r"[^a-zA-Z0-9_一-鿿]", "_", name)
    if not safe:
        safe = "node_" + str(abs(hash(name)) % 10000)
    # Mermaid 不允许以数字开头
    if safe[0].isdigit():
        safe = "n_" + safe
    return safe
