"""本模块作用：基于论文相似度推荐知识库内相关论文。"""

from __future__ import annotations

import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def recommend_similar_papers(
    target_doc_id: str,
    chunk_records: list[dict],
    embedding_vectors: dict[str, list[float]] | None,
    top_n: int = 5,
) -> list[dict]:
    """基于 embedding 相似度推荐与目标论文最相似的其他论文。

    Args:
        target_doc_id: 目标论文的 document_id
        chunk_records: 全部知识库块记录
        embedding_vectors: chunk_id → vector 映射（为 None 时退化为关键词重叠推荐）
        top_n: 推荐数量
    Returns:
        [{"document_id": ..., "file_name": ..., "title": ..., "score": ...}, ...]
    """
    # 收集目标论文的向量
    target_vectors: list[list[float]] = []
    for rec in chunk_records:
        if str(rec.get("document_id", "")) == target_doc_id:
            if embedding_vectors:
                v = embedding_vectors.get(str(rec.get("chunk_id", "")), [])
                if v:
                    target_vectors.append(v)

    # 按文档聚合相似度
    doc_scores: dict[str, tuple[float, int, str, str]] = {}  # doc_id → (total_score, count, file_name, title)
    for rec in chunk_records:
        did = str(rec.get("document_id", ""))
        if did == target_doc_id:
            continue
        meta = rec.get("metadata", {}) if isinstance(rec.get("metadata"), dict) else {}
        fn = str(meta.get("file_name", ""))
        title = str(meta.get("title", fn))

        if did not in doc_scores:
            doc_scores[did] = (0.0, 0, fn, title)

        if embedding_vectors and target_vectors:
            cv = embedding_vectors.get(str(rec.get("chunk_id", "")), [])
            if cv:
                # 取该 chunk 与目标论文所有 chunk 的最大相似度
                best = max(cosine_similarity(cv, tv) for tv in target_vectors)
                ts, cnt, _, _ = doc_scores[did]
                doc_scores[did] = (ts + best, cnt + 1, fn, title)
        else:
            # 无向量：纯关键词重叠退化为最低可用模式
            ts, cnt, _, _ = doc_scores[did]
            doc_scores[did] = (ts + 0.5, cnt + 1, fn, title)

    # 按平均相似度排序
    ranked: list[tuple[float, str, str, str]] = []
    for did, (total, cnt, fn, title) in doc_scores.items():
        avg = total / cnt if cnt > 0 else 0.0
        ranked.append((avg, did, fn, title))
    ranked.sort(key=lambda x: -x[0])

    return [
        {"document_id": did, "file_name": fn, "title": title, "score": round(sc, 3)}
        for sc, did, fn, title in ranked[:top_n]
    ]


def format_recommendations(recs: list[dict]) -> str:
    if not recs:
        return "未找到相似论文推荐。"
    lines = ["相似论文推荐："]
    for i, r in enumerate(recs, 1):
        title = r.get("title", r.get("file_name", ""))
        lines.append(f"{i}. {title}（{r['file_name']}）[相似度: {r['score']}]")
    return "\n".join(lines)
