import logging
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from agent.documents.ocr import OCRUnavailableError
from agent.documents.ocr import engine as ocr_engine

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"})
SUPPORTED_EXTENSIONS = frozenset({".pdf", ".txt", ".md"}) | IMAGE_EXTENSIONS
# A page with less extracted text than this is treated as scanned and OCR'd.
_MIN_TEXT_CHARS = 20


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


def load_document(
    path: str | Path, name: str | None = None, ocr: bool = True, ocr_max_pages: int = 30
) -> list[Document]:
    """Load a supported file. `name` overrides the source label used in citations
    (uploads are saved under a temporary filename). With `ocr`, scanned PDF pages and
    images are OCR'd; OCR'd documents carry `metadata["ocr"] = True`."""
    path = Path(path).expanduser()
    name = name or path.name
    if not path.is_file():
        raise DocumentLoadError(f"File not found: {path}")

    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DocumentLoadError(f"Unsupported file type '{suffix or name}'. Supported: {supported}")
    if suffix == ".pdf":
        return _load_pdf(path, name, ocr, ocr_max_pages)
    if suffix in IMAGE_EXTENSIONS:
        return _load_image(path, name, ocr)
    return _load_text(path, name)


def _load_pdf(path: Path, name: str, ocr: bool, ocr_max_pages: int) -> list[Document]:
    """One Document per non-empty page, keeping 1-based page numbers for citations."""
    try:
        reader = PdfReader(path)
        if reader.is_encrypted and not reader.decrypt(""):
            raise DocumentLoadError(f"{name} is password protected")
        texts = [clean_text(_page_text(page, number, name)) for number, page in enumerate(reader.pages, start=1)]
    except PyPdfError as exc:
        raise DocumentLoadError(f"Could not read {name}: {exc}") from exc

    pages, ocr_used, ocr_skipped = [], 0, 0
    for number, text in enumerate(texts, start=1):
        metadata = {"source": name, "page": number}
        if len(text) < _MIN_TEXT_CHARS and ocr:
            if ocr_used >= ocr_max_pages:
                ocr_skipped += 1
            else:
                scanned = _ocr_pdf_page(path, number, name)
                ocr_used += 1
                if len(scanned) > len(text):
                    text, metadata["ocr"] = scanned, True
        if text:
            pages.append(Document(page_content=text, metadata=metadata))

    if ocr_skipped:
        logger.warning("%s: OCR limit of %d pages reached; %d scanned pages skipped", name, ocr_max_pages, ocr_skipped)
    if not pages:
        hint = "it may be a scan and OCR is disabled" if not ocr else "OCR found no readable text either"
        raise DocumentLoadError(f"{name} has no extractable text ({hint})")
    return pages


def _ocr_pdf_page(path: Path, number: int, name: str) -> str:
    try:
        return clean_text(ocr_engine.read_pdf_page(path, number - 1))
    except OCRUnavailableError:
        raise
    except Exception as exc:
        logger.warning("OCR failed on page %d of %s: %s", number, name, exc)
        return ""


def _load_image(path: Path, name: str, ocr: bool) -> list[Document]:
    if not ocr:
        raise DocumentLoadError(f"{name} is an image; enable OCR to index images")
    try:
        text = clean_text(ocr_engine.read_image(path))
    except OCRUnavailableError as exc:
        raise DocumentLoadError(f"Cannot read {name}: {exc}") from exc
    except ValueError as exc:
        raise DocumentLoadError(f"Could not read {name}: {exc}") from exc
    if not text:
        raise DocumentLoadError(f"No readable text found in {name}")
    return [Document(page_content=text, metadata={"source": name, "page": None, "ocr": True})]


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
    path: str | Path, chunk_size: int, chunk_overlap: int, name: str | None = None,
    ocr: bool = True, ocr_max_pages: int = 30,
) -> list[Document]:
    return chunk_documents(load_document(path, name, ocr, ocr_max_pages), chunk_size, chunk_overlap)


def citation_tag(source: str, page: int | None) -> str:
    return f"[{source} p.{page}]" if page is not None else f"[{source}]"
