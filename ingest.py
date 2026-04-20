"""
Build a FAISS vector index from a larger multi-format knowledge base.
Supported inputs in sample_docs/: .txt, .md, .pdf, .csv, .tsv, .xlsx, .xls, .html, .htm

Usage:
    python ingest.py
"""

import math
import os
import re
from pathlib import Path

import fitz
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embeddings import get_embeddings, save_provider_marker

load_dotenv()

DOCS_DIR = Path(__file__).parent / "sample_docs"
INDEX_DIR = Path(__file__).parent / "faiss_index"
MAX_CHUNKS_PER_SOURCE = int(os.getenv("MAX_CHUNKS_PER_SOURCE", "120"))
MAX_TOTAL_CHUNKS = int(os.getenv("MAX_TOTAL_CHUNKS", "600"))
MAX_ROWS_PER_TABULAR_SECTION = int(os.getenv("MAX_ROWS_PER_TABULAR_SECTION", "250"))
SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".pdf",
    ".csv",
    ".tsv",
    ".xlsx",
    ".xls",
    ".html",
    ".htm",
}


def clean_text(text: str) -> str:
    """Normalize whitespace so chunking behaves consistently across file types."""
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def dataframe_to_text(df: pd.DataFrame, label: str) -> str:
    """Convert a dataframe slice into a retrieval-friendly text block."""
    working = df.fillna("")
    if working.empty:
        return f"{label}\n\n[empty table]"

    header = " | ".join(str(col) for col in working.columns)
    rows = [
        " | ".join(str(value) for value in row)
        for row in working.astype(str).itertuples(index=False, name=None)
    ]
    body = "\n".join(rows)
    return clean_text(f"{label}\n\nColumns: {header}\n\n{body}")


def summarize_dataframe(df: pd.DataFrame, label: str) -> list[tuple[str, str]]:
    """Create preview and row-window sections for large structured tables."""
    working = df.fillna("")
    if working.empty:
        return [(label, f"{label}\n\n[empty table]")]

    sections: list[tuple[str, str]] = []
    preview = dataframe_to_text(working.head(20), f"{label}\nPreview rows")
    sections.append((f"{label} Preview", preview))

    total_rows = len(working)
    for start in range(0, total_rows, MAX_ROWS_PER_TABULAR_SECTION):
        stop = min(start + MAX_ROWS_PER_TABULAR_SECTION, total_rows)
        batch = working.iloc[start:stop]
        batch_text = dataframe_to_text(
            batch,
            f"{label}\nRows {start + 1}-{stop} of {total_rows}",
        )
        sections.append((f"{label} Rows {start + 1}-{stop}", batch_text))

    return sections


def load_text_file(path: Path) -> list[tuple[str, str]]:
    return [("Text", clean_text(path.read_text(encoding="utf-8")))]


def load_markdown_file(path: Path) -> list[tuple[str, str]]:
    return [("Markdown", clean_text(path.read_text(encoding="utf-8")))]


def load_pdf_file(path: Path) -> list[tuple[str, str]]:
    """Extract PDF text page by page so citations can still point to coarse sections."""
    sections: list[tuple[str, str]] = []
    document = fitz.open(path)
    try:
        for page_index, page in enumerate(document, start=1):
            text = clean_text(page.get_text("text"))
            if text:
                sections.append((f"Page {page_index}", text))
    finally:
        document.close()
    return sections


def load_delimited_file(path: Path, separator: str) -> list[tuple[str, str]]:
    df = pd.read_csv(path, sep=separator)
    return summarize_dataframe(df, f"Dataset: {path.name}")


def load_excel_file(path: Path) -> list[tuple[str, str]]:
    workbook = pd.read_excel(path, sheet_name=None)
    sections: list[tuple[str, str]] = []
    for sheet_name, df in workbook.items():
        sections.extend(
            summarize_dataframe(
                df,
                f"Workbook: {path.name}\nSheet: {sheet_name}",
            )
        )
    return sections


def load_html_file(path: Path) -> list[tuple[str, str]]:
    """Index both visible page text and any HTML tables embedded in the document."""
    html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    sections: list[tuple[str, str]] = []

    title = soup.title.get_text(strip=True) if soup.title else path.name
    body_text = clean_text(soup.get_text("\n"))
    if body_text:
        sections.append((f"HTML: {title}", body_text))

    for table_index, table in enumerate(soup.find_all("table"), start=1):
        frames = pd.read_html(str(table))
        for frame_index, df in enumerate(frames, start=1):
            sections.extend(
                summarize_dataframe(
                    df,
                    f"HTML table from {path.name} #{table_index}.{frame_index}",
                )
            )

    return sections


def load_source_sections(path: Path) -> list[tuple[str, str]]:
    """Dispatch file loading based on extension."""
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return load_text_file(path)
    if suffix == ".md":
        return load_markdown_file(path)
    if suffix == ".pdf":
        return load_pdf_file(path)
    if suffix == ".csv":
        return load_delimited_file(path, ",")
    if suffix == ".tsv":
        return load_delimited_file(path, "\t")
    if suffix in {".xlsx", ".xls"}:
        return load_excel_file(path)
    if suffix in {".html", ".htm"}:
        return load_html_file(path)
    raise ValueError(f"Unsupported file type: {path}")


def split_markdown_sections(text: str) -> list[tuple[str, str]]:
    """Preserve markdown headings as explicit retrieval sections."""
    sections: list[tuple[str, str]] = []
    current_title = "Introduction"
    current_lines: list[str] = []

    for line in text.splitlines():
        if re.match(r"^#{1,6}\s+", line.strip()):
            if current_lines:
                sections.append((current_title, clean_text("\n".join(current_lines))))
                current_lines = []
            current_title = line.lstrip("#").strip()
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, clean_text("\n".join(current_lines))))

    return [section for section in sections if section[1]]


def choose_chunk_params(text: str, suffix: str) -> tuple[int, int]:
    """Tune chunk size and overlap by content type and overall document density."""
    length = len(text)
    avg_paragraph = max(1, math.ceil(length / max(1, text.count("\n\n") + 1)))

    if suffix == ".md":
        base_size = 1200
        overlap = 180
    elif suffix == ".pdf":
        base_size = 1400
        overlap = 220
    elif suffix in {".xlsx", ".xls", ".csv", ".tsv"}:
        base_size = 1500
        overlap = 160
    elif suffix in {".html", ".htm"}:
        base_size = 1300
        overlap = 180
    else:
        base_size = 1000
        overlap = 150

    if avg_paragraph > 900:
        base_size += 300
        overlap += 40
    if length < 2000:
        base_size = min(base_size, 800)
        overlap = min(overlap, 120)

    return base_size, overlap


def recursive_split_text(text: str, suffix: str, chunk_size: int, overlap: int) -> list[str]:
    """Use true recursive chunking with separators chosen for each file family."""
    if suffix == ".md":
        separators = ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""]
    elif suffix in {".html", ".htm"}:
        separators = ["\n\n", "\n", ". ", " ", ""]
    elif suffix in {".xlsx", ".xls", ".csv", ".tsv"}:
        separators = ["\nRows ", "\n\n", "\n", " | ", " ", ""]
    else:
        separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )
    return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]


def normalize_sections(path: Path, sections: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Standardize sections before recursive chunking."""
    if path.suffix.lower() == ".md" and len(sections) == 1:
        return split_markdown_sections(sections[0][1])
    if len(sections) == 1:
        return [("Content", sections[0][1])] if sections[0][1] else []
    return [(title, text) for title, text in sections if text]


def chunk_document(path: Path, sections: list[tuple[str, str]]) -> list[Document]:
    """Convert loaded sections into chunked LangChain documents with rich metadata."""
    suffix = path.suffix.lower()
    strategy = (
        "section_aware_markdown"
        if suffix == ".md"
        else "table_aware"
        if suffix in {".xlsx", ".xls", ".csv", ".tsv", ".html", ".htm"}
        else "recursive_prose"
    )
    normalized_sections = normalize_sections(path, sections)
    chunk_size, overlap = choose_chunk_params(
        "\n\n".join(text for _, text in normalized_sections), suffix
    )

    chunks: list[Document] = []
    for section_index, (section_title, section_text) in enumerate(normalized_sections, start=1):
        # Split inside each section instead of across the whole file so sources remain inspectable.
        chunk_texts = recursive_split_text(section_text, suffix, chunk_size, overlap)
        for chunk_index, chunk_text in enumerate(chunk_texts, start=1):
            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "source": path.name,
                        "path": str(path),
                        "section": section_title,
                        "section_index": section_index,
                        "chunk_index": chunk_index,
                        "chunk_strategy": strategy,
                        "chunk_size": chunk_size,
                        "chunk_overlap": overlap,
                        "file_type": suffix,
                    },
                )
            )

    if len(chunks) > MAX_CHUNKS_PER_SOURCE:
        print(
            f"Capping {path.name} from {len(chunks)} to {MAX_CHUNKS_PER_SOURCE} chunks "
            "to control embedding cost."
        )
        return chunks[:MAX_CHUNKS_PER_SOURCE]

    return chunks


def load_and_split() -> list[Document]:
    """Load every supported source file under sample_docs and chunk it."""
    chunks: list[Document] = []
    for path in sorted(DOCS_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        sections = load_source_sections(path)
        sections = [(title, text) for title, text in sections if text]
        if not sections:
            continue

        doc_chunks = chunk_document(path, sections)
        chunks.extend(doc_chunks)
        print(f"Loaded {path.name}: {len(doc_chunks)} chunks")

    if len(chunks) > MAX_TOTAL_CHUNKS:
        print(
            f"Capping total corpus from {len(chunks)} to {MAX_TOTAL_CHUNKS} chunks "
            "to stay within embedding budget."
        )
        chunks = chunks[:MAX_TOTAL_CHUNKS]

    return chunks


def main():
    """Build and persist the FAISS index for the current local corpus."""
    print("Loading and chunking knowledge base...")
    chunks = load_and_split()
    print(f"Total chunks prepared: {len(chunks)}")

    print("Embedding and building FAISS index...")
    embeddings, provider = get_embeddings()
    try:
        vectorstore = FAISS.from_documents(chunks, embeddings)
    except Exception as exc:
        if provider == "google":
            print(f"Google embeddings failed ({exc}). Falling back to local hash embeddings...")
            embeddings, provider = get_embeddings("local")
            vectorstore = FAISS.from_documents(chunks, embeddings)
        else:
            raise
    vectorstore.save_local(str(INDEX_DIR))
    save_provider_marker(INDEX_DIR, provider)

    print(f"Index saved to {INDEX_DIR}/ using provider: {provider}")


if __name__ == "__main__":
    main()
