import pytest
from langchain_core.documents import Document

from agent.documents import (
    DocumentLoadError,
    DocumentStore,
    chunk_documents,
    clean_text,
    ingest_document,
    load_document,
)
from fakes import KeywordEmbedder


def test_clean_text_joins_hyphenated_breaks_and_collapses_whitespace():
    assert clean_text("retrie-\nval   augmented\n\n\n\ngeneration") == (
        "retrieval augmented\n\ngeneration"
    )


def test_pdf_keeps_page_numbers_and_skips_blank_pages(make_pdf):
    path = make_pdf("paper.pdf", ["Intro to transformers", "", "Results section"])

    pages = load_document(path)

    assert [p.metadata["page"] for p in pages] == [1, 3]
    assert all(p.metadata["source"] == "paper.pdf" for p in pages)
    assert "transformers" in pages[0].page_content


def test_name_override_sets_citation_source(make_pdf):
    path = make_pdf("tmp8f3k2.pdf", ["Some text"])

    [page] = load_document(path, name="Annual Report.pdf")

    assert page.metadata["source"] == "Annual Report.pdf"


@pytest.mark.parametrize("filename", ["notes.txt", "README.md"])
def test_text_files_load_without_page_numbers(tmp_path, filename):
    path = tmp_path / filename
    path.write_text("# Title\n\nSome   useful   content", encoding="utf-8")

    [doc] = load_document(path)

    assert doc.metadata == {"source": filename, "page": None}
    assert doc.page_content == "# Title\n\nSome useful content"


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(DocumentLoadError, match="not found"):
        load_document(tmp_path / "missing.pdf")


def test_unsupported_extension_is_rejected(tmp_path):
    path = tmp_path / "sheet.xlsx"
    path.write_bytes(b"data")

    with pytest.raises(DocumentLoadError, match="Unsupported file type"):
        load_document(path)


def test_corrupt_pdf_is_rejected(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"garbage")

    with pytest.raises(DocumentLoadError, match="Could not read"):
        load_document(path)


def test_pdf_without_text_suggests_ocr(make_pdf):
    with pytest.raises(DocumentLoadError, match="OCR"):
        load_document(make_pdf("scan.pdf", [""]))


def test_non_utf8_and_empty_text_files_are_rejected(tmp_path):
    binary = tmp_path / "binary.txt"
    binary.write_bytes(b"\xff\xfe\x00\x81")
    empty = tmp_path / "empty.md"
    empty.write_text("   \n")

    with pytest.raises(DocumentLoadError, match="UTF-8"):
        load_document(binary)
    with pytest.raises(DocumentLoadError, match="empty"):
        load_document(empty)


def test_chunks_respect_size_and_carry_citation_metadata():
    page = Document(page_content=" ".join(["token"] * 600), metadata={"source": "a.pdf", "page": 7})

    chunks = chunk_documents([page], chunk_size=500, chunk_overlap=50)

    assert len(chunks) > 1
    assert all(len(c.page_content) <= 500 for c in chunks)
    assert all(c.metadata["page"] == 7 and c.metadata["source"] == "a.pdf" for c in chunks)
    assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)


def test_ingest_document_end_to_end(make_pdf):
    path = make_pdf("report.pdf", ["Revenue grew 12 percent", "Headcount stayed flat"])

    chunks = ingest_document(path, chunk_size=1000, chunk_overlap=100)

    assert [c.metadata["page"] for c in chunks] == [1, 2]


def _doc(text, source="a.pdf", page=1):
    return Document(page_content=text, metadata={"source": source, "page": page})


def test_store_ranks_most_relevant_chunk_first():
    store = DocumentStore()
    store.add([
        _doc("The quarterly revenue grew strongly in Europe", page=1),
        _doc("Employee headcount remained flat this year", page=2),
        _doc("Revenue revenue revenue forecast for next year", page=3),
    ])

    hits = store.search("revenue forecast")

    assert hits[0][0].metadata["page"] == 3
    assert all(doc.metadata["page"] != 2 for doc, _ in hits)


def test_store_finds_match_in_single_chunk_corpus():
    store = DocumentStore()
    store.add([_doc("Attention is all you need")])

    assert store.search("attention")


def test_store_reupload_replaces_previous_chunks():
    store = DocumentStore()
    store.add([_doc("old content about llamas")])
    store.add([_doc("new content about alpacas")])

    assert len(store) == 1
    assert not store.search("llamas")
    assert store.search("alpacas")


def test_store_remove_and_chunk_counts():
    store = DocumentStore()
    store.add([_doc("alpha", "a.pdf"), _doc("beta", "a.pdf"), _doc("gamma", "b.md", None)])

    assert store.chunk_counts() == {"a.pdf": 2, "b.md": 1}
    assert store.remove("a.pdf") is True
    assert store.remove("a.pdf") is False
    assert store.sources == ["b.md"]
    assert not store.search("alpha")


def test_store_handles_empty_index_and_stopword_queries():
    store = DocumentStore()
    assert store.search("anything") == []
    store.add([_doc("some text")])
    assert store.search("the of and") == []


def _hybrid_store():
    store = DocumentStore(embedder=KeywordEmbedder())
    store.add([
        _doc("Quarterly sales climbed sharply in Europe", page=1),
        _doc("The office moved to a new building", page=2),
        _doc("Supplier concentration is a major threat", page=3),
    ])
    return store


def test_hybrid_search_finds_synonyms_that_bm25_misses():
    keyword_only = DocumentStore()
    keyword_only.add([_doc("Quarterly sales climbed sharply in Europe", page=1)])
    assert keyword_only.search("how much did turnover rise") == []

    hits = _hybrid_store().search("how much did turnover rise", k=1)

    assert hits[0][0].metadata["page"] == 1


def test_hybrid_fuses_keyword_and_semantic_rankings():
    hits = _hybrid_store().search("supplier risk", k=2)

    assert hits[0][0].metadata["page"] == 3  # top on both BM25 ("supplier") and dense ("risk")


def test_vectors_survive_export_and_restore_without_re_embedding():
    original = _hybrid_store()
    chunks, vectors = original.export("a.pdf")

    class NoEmbedding(KeywordEmbedder):
        def embed_documents(self, texts):
            raise AssertionError("restore must reuse stored vectors")

    restored = DocumentStore(embedder=NoEmbedding())
    restored.add(chunks, vectors)

    assert restored.hybrid
    assert restored.search("turnover", k=1)[0][0].metadata["page"] == 1


def test_remove_keeps_vectors_aligned():
    store = _hybrid_store()
    store.remove("a.pdf")
    store.add([_doc("Headcount grew", source="b.pdf")])

    assert store.search("employees", k=1)[0][0].metadata["source"] == "b.pdf"


def test_embedding_failure_falls_back_to_keyword_search():
    class Broken(KeywordEmbedder):
        def embed_documents(self, texts):
            raise OSError("model download failed")

    store = DocumentStore(embedder=Broken())
    store.add([_doc("revenue grew")])

    assert not store.hybrid
    assert store.search("revenue")
