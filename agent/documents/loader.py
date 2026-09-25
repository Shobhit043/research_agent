import logging
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PyPdfError

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".txt", ".md"})


class DocumentLoadError(Exception):
    """Raised when a file cannot be turned into searchable text."""


_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SPACES = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _SPACES.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def load_document(path: str | Path, name: str | None = None) -> list[Document]:
    """Load a supported file. `name` overrides the source label used in citations
    (uploads are saved under a temporary filename)."""
    path = Path(path).expanduser()
    name = name or path.name
    if not path.is_file():
        raise DocumentLoadError(f"File not found: {path}")

    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DocumentLoadError(f"Unsupported file type '{suffix or name}'. Supported: {supported}")
    if suffix == ".pdf":
        return _load_pdf(path, name)
    return _load_text(path, name)


def _load_pdf(path: Path, name: str) -> list[Document]:
    """One Document per non-empty page, keeping 1-based page numbers for citations."""
    try:
        reader = PdfReader(path)
        if reader.is_encrypted and not reader.decrypt(""):
            raise DocumentLoadError(f"{name} is password protected")
        pages = [
            Document(
                page_content=clean_text(_page_text(page, number, name)),
                metadata={"source": name, "page": number},
            )
            for number, page in enumerate(reader.pages, start=1)
        ]
    except PyPdfError as exc:
        raise DocumentLoadError(f"Could not read {name}: {exc}") from exc

    pages = [page for page in pages if page.page_content]
    if not pages:
        raise DocumentLoadError(f"{name} has no extractable text (scanned PDFs need OCR first)")
    return pages


def _page_text(page, number: int, name: str) -> str:
    try:
        return page.extract_text() or ""
    except Exception as exc:
        # pypdf raises assorted errors on malformed pages; one bad page shouldn't sink the file.
        logger.warning("Skipping unreadable page %d of %s: %s", number, name, exc)
        return ""


def _load_text(path: Path, name: str) -> list[Document]:
    try:
        text = clean_text(path.read_text(encoding="utf-8-sig"))
    except UnicodeDecodeError as exc:
        raise DocumentLoadError(f"{name} is not UTF-8 text") from exc
    if not text:
        raise DocumentLoadError(f"{name} is empty")
    # Plain text has no pages; page=None makes citations read [notes.md] rather than [notes.md p.1].
    return [Document(page_content=text, metadata={"source": name, "page": None})]


def chunk_documents(pages: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split pages into overlapping chunks. Chunks never span pages, so every chunk cites one page."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(pages)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata['source']}#{index}"
    return chunks


def ingest_document(
    path: str | Path, chunk_size: int, chunk_overlap: int, name: str | None = None
) -> list[Document]:
    return chunk_documents(load_document(path, name), chunk_size, chunk_overlap)


def citation_tag(source: str, page: int | None) -> str:
    return f"[{source} p.{page}]" if page is not None else f"[{source}]"
