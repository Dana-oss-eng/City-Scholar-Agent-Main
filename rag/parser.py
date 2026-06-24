"""本模块作用：解析本地 PDF 论文为统一文档结构，并提取学术元数据。"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


@dataclass
class ParsedPage:
    """单页解析结果，便于定位文本来源页码。"""
    page_number: int
    text: str
    start_char: int
    end_char: int


@dataclass
class PaperMetadata:
    """论文元数据 —— 从 PDF 正文中提取的结构化学术信息。"""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: str = ""
    journal: str = ""
    doi: str = ""
    abstract: str = ""
    keywords: list[str] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """单篇论文的完整解析结果。"""
    document_id: str
    file_name: str
    source_path: str
    total_pages: int
    total_characters: int
    full_text: str
    pages: list[ParsedPage]
    metadata: PaperMetadata = field(default_factory=PaperMetadata)


# ====== PDF 基础解析 ======

def get_pdf_reader_class():
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("缺少 pypdf 依赖，请先执行 `pip install -r requirements.txt`。") from exc
    return PdfReader


def build_document_id(pdf_path: str | Path) -> str:
    path = Path(pdf_path)
    normalized_name = re.sub(r"\W+", "_", path.stem.strip().lower()).strip("_")
    return normalized_name or "unknown_document"


def normalize_text(text: str) -> str:
    cleaned_text = text.replace("\r", "\n")
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
    return cleaned_text.strip()


def parse_pdf_file(pdf_path: str | Path) -> ParsedDocument:
    """解析单个 PDF 文件（文本提取 + 结构整理）。"""
    path = Path(pdf_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"不是 PDF 文件：{path}")

    PdfReader = get_pdf_reader_class()
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise RuntimeError(f"无法打开 PDF 文件：{path.name}") from exc

    pages: list[ParsedPage] = []
    text_segments: list[str] = []
    current_char = 0

    for page_index, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception as exc:
            raise RuntimeError(f"第 {page_index} 页文本提取失败：{path.name}") from exc

        page_text = normalize_text(raw_text)
        if text_segments and page_text:
            current_char += 2

        start_char = current_char
        if page_text:
            text_segments.append(page_text)
            current_char += len(page_text)
        end_char = current_char

        pages.append(ParsedPage(
            page_number=page_index, text=page_text,
            start_char=start_char, end_char=end_char,
        ))

    full_text = "\n\n".join(text_segments).strip()
    if not full_text:
        raise RuntimeError(
            f"未能从 PDF 中提取到有效文本：{path.name}。该文件可能是扫描版、加密文件或内容异常。"
        )

    return ParsedDocument(
        document_id=build_document_id(path),
        file_name=path.name,
        source_path=str(path),
        total_pages=len(pages),
        total_characters=len(full_text),
        full_text=full_text,
        pages=pages,
    )


def parse_pdf_files(pdf_paths: list[str | Path]) -> tuple[list[ParsedDocument], list[dict[str, str]]]:
    """批量解析 PDF，单文件失败不中断整体流程。"""
    documents: list[ParsedDocument] = []
    errors: list[dict[str, str]] = []
    for pdf_path in pdf_paths:
        current_path = Path(pdf_path).expanduser().resolve()
        try:
            documents.append(parse_pdf_file(current_path))
        except Exception as exc:
            errors.append({
                "file_name": current_path.name,
                "source_path": str(current_path),
                "error_message": str(exc),
            })
    return documents, errors


# ====== 元数据提取 ======

_METADATA_PROMPT = """你是一个学术论文元数据提取助手。请从以下论文开头文本中提取元数据，严格输出 JSON。

字段说明：
- title: 论文完整标题
- authors: 作者姓名列表
- year: 发表年份（4位数字字符串）
- journal: 发表的期刊或会议名称
- doi: DOI 号（如有）
- abstract: 摘要全文
- keywords: 关键词列表

如果某个字段无法从文本中识别，用空字符串或空数组表示。

论文文本：
{text}

请仅输出 JSON 对象："""


def extract_metadata_with_llm(front_text: str, chat_fn) -> PaperMetadata:
    """使用 LLM 从论文开头文本中提取结构化学术元数据。

    Args:
        front_text: 论文前若干页文本（建议前 5000 字符）
        chat_fn: 签名为 (messages: list[dict], *, model: str) -> str 的 LLM 调用函数
    """
    if not front_text.strip() or chat_fn is None:
        return PaperMetadata()

    try:
        raw = chat_fn(
            messages=[
                {"role": "system", "content": "你是一个精确的学术论文元数据提取器。只从给定文本中提取，不要编造。"},
                {"role": "user", "content": _METADATA_PROMPT.format(text=front_text[:6000])},
            ],
            temperature=0.1,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        data = json.loads(raw)
    except Exception:
        return PaperMetadata()

    return PaperMetadata(
        title=str(data.get("title", "")).strip(),
        authors=[str(a).strip() for a in data.get("authors", []) if str(a).strip()],
        year=str(data.get("year", "")).strip(),
        journal=str(data.get("journal", "")).strip(),
        doi=str(data.get("doi", "")).strip(),
        abstract=str(data.get("abstract", "")).strip(),
        keywords=[str(k).strip() for k in data.get("keywords", []) if str(k).strip()],
    )


def enrich_documents_with_metadata(
    documents: list[ParsedDocument],
    chat_fn=None,
) -> list[ParsedDocument]:
    """为已解析文档批量补充元数据（原地修改并返回）。"""
    for doc in documents:
        if doc.metadata.title:
            continue  # 已有元数据，跳过
        front_text = doc.full_text[:5000] if doc.full_text else ""
        doc.metadata = extract_metadata_with_llm(front_text, chat_fn)
    return documents


# ====== 演示 ======

def run_parser_demo() -> None:
    from rag.loader import list_pdf_files
    pdf_files = list_pdf_files()
    print("Parser Demo")
    print(f"待解析 PDF 数量：{len(pdf_files)}")
    if not pdf_files:
        print("提示：请先将 PDF 文件放入 raw_papers/。")
        return

    documents, errors = parse_pdf_files(pdf_files)
    print(f"解析成功数量：{len(documents)}")
    print(f"解析失败数量：{len(errors)}")
    if documents:
        sample = documents[0]
        print(f"示例文档：{sample.file_name} | 页数：{sample.total_pages} | 字符数：{sample.total_characters}")
        print(f"文本预览：{sample.full_text[:120]}")
    if errors:
        print("失败示例：", errors[0])


if __name__ == "__main__":
    run_parser_demo()
