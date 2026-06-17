"""本模块作用：将论文文本切分为适合检索的语义块，支持章节感知切分与元数据传递。"""

from __future__ import annotations

import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from rag.parser import ParsedDocument


# 常见论文章节标题模式（使用 flags=re.IGNORECASE 处理英文大小写）
SECTION_PATTERNS = [
    (r"\babstract\b", "abstract", True),
    (r"\bintroduction\b|\bbackground\b", "introduction", True),
    (r"\brelated\s*work\b|\bliterature\s*review\b", "related_work", True),
    (r"\bmethods?\b|\bmethodology\b|\bexperimental\b", "methods", True),
    (r"\bresults?\b|\bfindings?\b", "results", True),
    (r"\bdiscussions?\b", "discussion", True),
    (r"\bconclusions?\b|\bsummary\b", "conclusion", True),
    (r"摘要|引言|前言|绪论|背景", "introduction", False),
    (r"方法|研究方法|实验设计|数据与方法", "methods", False),
    (r"结果|实验结果|研究发现", "results", False),
    (r"讨论", "discussion", False),
    (r"结论|总结|结语", "conclusion", False),
    (r"参考文献|References|Bibliography", "references", False),
]


def detect_sections(text: str) -> list[tuple[int, str]]:
    """检测章节边界：返回 [(字符位置, 章节名), ...]，按位置排序去重。"""
    boundaries: list[tuple[int, str]] = []
    for pattern, name, ignore_case in SECTION_PATTERNS:
        flags = re.IGNORECASE if ignore_case else 0
        for m in re.finditer(pattern, text, flags=flags):
            pos = m.start()
            if pos == 0 or text[pos - 1] in "\n":
                boundaries.append((pos, name))
    boundaries.sort(key=lambda x: x[0])
    # 近距离去重
    out: list[tuple[int, str]] = []
    for pos, name in boundaries:
        if out and pos - out[-1][0] < 60:
            continue
        out.append((pos, name))
    return out


def split_paragraphs(text: str) -> list[str]:
    """按段落切分：优先双换行，长段落按单换行再切。"""
    raw = re.split(r"\n\s*\n", text)
    paragraphs: list[str] = []
    for block in raw:
        s = block.strip()
        if not s:
            continue
        if len(s) > 600:
            for sub in re.split(r"\n(?=[A-Z一-鿿])", s):
                if sub.strip():
                    paragraphs.append(sub.strip())
        else:
            paragraphs.append(s)
    return paragraphs


def assign_section(pos: int, boundaries: list[tuple[int, str]]) -> str:
    """确定某字符位置属于哪个章节。"""
    sec = "body"
    for bp, bn in boundaries:
        if pos >= bp:
            sec = bn
        else:
            break
    return sec


def _find_page_nums(pages, start_char: int, end_char: int) -> list[int]:
    nums: list[int] = []
    for p in pages:
        if start_char < p.end_char and end_char > p.start_char:
            if p.page_number not in nums:
                nums.append(p.page_number)
    return nums or [1]


def _make_metadata(doc: ParsedDocument, chunk_idx: int, start_char: int, end_char: int, section: str) -> dict:
    return {
        "file_name": doc.file_name,
        "source_path": doc.source_path,
        "chunk_index": chunk_idx,
        "start_char": start_char,
        "end_char": end_char,
        "page_numbers": _find_page_nums(doc.pages, start_char, end_char),
        "section": section,
        "title": doc.metadata.title or doc.file_name,
        "authors": doc.metadata.authors,
        "year": doc.metadata.year,
        "journal": doc.metadata.journal,
        "doi": doc.metadata.doi,
    }


def semantic_chunk_document(doc: ParsedDocument, chunk_size: int = 800, chunk_overlap: int = 150) -> list[dict]:
    """按章节/段落语义切分单篇论文。"""
    text = doc.full_text
    boundaries = detect_sections(text)
    paragraphs = split_paragraphs(text)
    chunks: list[dict] = []
    buf, buf_start, buf_sec, idx = "", 0, "body", 0

    def flush():
        nonlocal buf, buf_start, idx
        if not buf.strip():
            return
        cs = text.find(buf[:80]) if len(buf) >= 80 else buf_start
        if cs < 0:
            cs = buf_start
        ce = cs + len(buf)
        chunks.append({
            "chunk_id": f"{doc.document_id}_chunk_{idx:04d}",
            "document_id": doc.document_id,
            "text": buf.strip(),
            "metadata": _make_metadata(doc, idx, cs, ce, buf_sec),
        })
        idx += 1

    for para in paragraphs:
        ps = text.find(para[:60]) if len(para) >= 60 else buf_start
        if ps < 0:
            ps = buf_start
        psec = assign_section(ps, boundaries)

        if buf and len(buf) + len(para) > chunk_size + 300:
            flush()
            if len(para) < chunk_size:
                buf, buf_start, buf_sec = para + "\n\n", ps, psec
                continue
            buf, buf_start, buf_sec = "", ps, psec

        buf += para + "\n\n"
        if buf_sec == "body":
            buf_sec = psec
        buf_start = min(buf_start, ps) if buf_start else ps

    flush()
    return chunks


def fixed_chunk_document(doc: ParsedDocument, chunk_size: int = 500, chunk_overlap: int = 100) -> list[dict]:
    """固定大小切分（回退方案）。"""
    text = doc.full_text
    chunks: list[dict] = []
    idx, start = 0, 0
    step = chunk_size - chunk_overlap
    while start < len(text):
        end = min(start + chunk_size, len(text))
        ct = text[start:end].strip()
        if ct:
            chunks.append({
                "chunk_id": f"{doc.document_id}_chunk_{idx:04d}",
                "document_id": doc.document_id,
                "text": ct,
                "metadata": _make_metadata(doc, idx, start, end, "body"),
            })
            idx += 1
        if end >= len(text):
            break
        start += step
    return chunks


def build_knowledge_base_records(
    documents: list[ParsedDocument],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    *,
    semantic: bool = True,
) -> list[dict]:
    """将已解析文档切分为可检索记录列表。

    Args:
        documents: 论文列表
        chunk_size: 块大小（字符数），语义模式默认 800
        chunk_overlap: 固定切分时的重叠量
        semantic: 是否使用章节语义切分
    """
    all_chunks: list[dict] = []
    for doc in documents:
        if semantic:
            chunks = semantic_chunk_document(doc, chunk_size, chunk_overlap)
        else:
            chunks = fixed_chunk_document(doc, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)
    return all_chunks
