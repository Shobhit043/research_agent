from langchain_core.documents import Document
from langchain_core.tools import BaseTool, tool

from agent.config import Settings
from agent.documents import DocumentStore, citation_tag

_READ_CHARS = 6000


def build_document_search_tool(store: DocumentStore, settings: Settings) -> BaseTool:
    @tool
    def search_documents(query: str) -> str:
        """
        Search the documents the user has uploaded and return the most relevant
        passages, each labelled with its source file (and page, for PDFs).

        Use this for any question about the uploaded documents. Search with
        specific keywords; call again with different wording if nothing relevant
        comes back.

        Args:
            query: Keywords or a question describing the information to find.
        """
        hits = store.search(query, k=settings.doc_search_k)
        if not hits:
            return "No matching passages found in the uploaded documents."
        return "\n\n".join(
            f"{citation_tag(doc.metadata['source'], doc.metadata['page'])}\n{doc.page_content}"
            for doc, _ in hits
        )

    return search_documents


def merge_chunks(chunks: list[Document]) -> str:
    """Rebuild a page's text from overlapping chunks using their start offsets."""
    text = ""
    end = 0
    for chunk in sorted(chunks, key=lambda c: c.metadata.get("start_index", 0)):
        start = chunk.metadata.get("start_index", end)
        overlap = max(0, end - start)
        text += chunk.page_content[overlap:] if overlap < len(chunk.page_content) else ""
        end = max(end, start + len(chunk.page_content))
    return text


def build_read_document_tool(store: DocumentStore) -> BaseTool:
    @tool
    def read_document(name: str, page: int | None = None) -> str:
        """
        Read the full text of an uploaded document, or one page of a PDF. Use this
        to summarise a document or when search results are too fragmentary.
        Without a page, returns the beginning of the document and its page count.

        Args:
            name: The document's file name exactly as listed, e.g. "report.pdf".
            page: Optional 1-based page number (PDFs only).
        """
        chunks, _ = store.export(name)
        if not chunks:
            return f"No uploaded document named '{name}'. Available: {', '.join(store.sources) or 'none'}."
        pages = sorted({c.metadata.get("page") for c in chunks}, key=lambda p: p or 0)
        if page is not None:
            selected = [c for c in chunks if c.metadata.get("page") == page]
            if not selected:
                return f"{name} has no page {page}. Pages with text: {', '.join(map(str, pages))}."
            return f"{citation_tag(name, page)}\n{merge_chunks(selected)[:_READ_CHARS]}"

        sections, used = [], 0
        for number in pages:
            body = merge_chunks([c for c in chunks if c.metadata.get("page") == number])
            if used + len(body) > _READ_CHARS and sections:
                sections.append(f"(Truncated: {len(pages)} pages in total. Continue with page={number}.)")
                break
            sections.append(f"{citation_tag(name, number)}\n{body[:_READ_CHARS]}")
            used += len(body)
        return "\n\n".join(sections)

    return read_document
