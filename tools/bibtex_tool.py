"""本模块作用：从论文元数据生成 BibTeX 引用条目。"""

from __future__ import annotations

from rag.parser import PaperMetadata, ParsedDocument


def metadata_to_bibtex(meta: PaperMetadata, file_name: str) -> str:
    """将单篇论文元数据转为 BibTeX 条目。

    Args:
        meta: 论文元数据对象
        file_name: 论文文件名（用作 cite key 的备选）
    """
    # 生成 cite key
    cite_key = _make_cite_key(meta, file_name)

    # 判断类型
    entry_type = "article"
    if "conference" in meta.journal.lower() or "proc." in meta.journal.lower():
        entry_type = "inproceedings"

    lines = [f"@{entry_type}{{{cite_key},"]

    if meta.title:
        lines.append(f"  title = {{{meta.title}}},")
    if meta.authors:
        lines.append(f"  author = {{{' and '.join(meta.authors)}}},")
    if meta.year:
        lines.append(f"  year = {{{meta.year}}},")
    if meta.journal:
        lines.append(f"  journal = {{{meta.journal}}},")
    if meta.doi:
        lines.append(f"  doi = {{{meta.doi}}},")

    lines.append("}")
    return "\n".join(lines)


def _make_cite_key(meta: PaperMetadata, file_name: str) -> str:
    """生成 BibTeX cite key: 第一作者LastNameYear 格式。"""
    if meta.authors:
        first_author = meta.authors[0].split()[-1] if " " in meta.authors[0] else meta.authors[0]
        first_author = "".join(c for c in first_author if c.isalpha())
    else:
        first_author = "Unknown"
    year = meta.year or "0000"
    return f"{first_author}{year}"


def export_all_bibtex(documents: list[ParsedDocument]) -> str:
    """为所有已解析论文生成 BibTeX 文件内容。"""
    entries: list[str] = []
    for doc in documents:
        if doc.metadata.title:
            entries.append(metadata_to_bibtex(doc.metadata, doc.file_name))
        else:
            # 无元数据时生成最小条目
            entries.append(f"@misc{{{doc.document_id},\n  title = {{{doc.file_name}}},\n}}")
    return "\n\n".join(entries)


def export_single_bibtex(doc: ParsedDocument) -> str:
    """为单篇论文生成 BibTeX 条目。"""
    return metadata_to_bibtex(doc.metadata, doc.file_name)
