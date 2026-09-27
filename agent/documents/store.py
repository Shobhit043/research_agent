import logging
import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from langchain_core.documents import Document

from agent.documents.embeddings import Embedder

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset([
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "with",
])
# Reciprocal rank fusion constant from Cormack et al. (2009); 60 is the standard choice.
RRF_K = 60
_DENSE_CANDIDATES = 20

# (query vector, k) -> [(chunk_id, similarity)], e.g. a pgvector query scoped to one session.
VectorSearch = Callable[[np.ndarray, int], list[tuple[str, float]]]


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


@dataclass(frozen=True)
class _Index:
    """Immutable snapshot, swapped in one assignment so searches never see a half-built index."""

    chunks: list[Document] = field(default_factory=list)
    term_freqs: list[Counter] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    doc_freq: Counter = field(default_factory=Counter)
    avg_length: float = 0.0
    vectors: np.ndarray | None = None

    @classmethod
    def build(cls, chunks: list[Document], vectors: np.ndarray | None) -> "_Index":
        term_freqs = [Counter(tokenize(c.page_content)) for c in chunks]
        lengths = [sum(freqs.values()) for freqs in term_freqs]
        return cls(
            chunks=chunks,
            term_freqs=term_freqs,
            lengths=lengths,
            doc_freq=Counter(term for freqs in term_freqs for term in freqs),
            avg_length=(sum(lengths) / len(lengths)) if lengths else 0.0,
            vectors=vectors if chunks else None,
        )


class DocumentStore:
    """In-memory hybrid index: BM25 keyword scoring fused with dense-vector similarity.

    BM25 uses the Lucene IDF variant, which stays positive even for tiny corpora
    (a one-chunk file would score zero under the classic Okapi formula). Without an
    embedder, or if embedding fails, it degrades to BM25 only.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        vector_search: VectorSearch | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self._embedder = embedder
        # When set, dense ranking is delegated (to pgvector) instead of using in-memory vectors.
        self._vector_search = vector_search
        self._k1 = k1
        self._b = b
        self._index = _Index()

    def __len__(self) -> int:
        return len(self._index.chunks)

    @property
    def hybrid(self) -> bool:
        if self._embedder is None:
            return False
        return self._vector_search is not None or self._index.vectors is not None

    @property
    def sources(self) -> list[str]:
        return sorted({chunk.metadata["source"] for chunk in self._index.chunks})

    def chunk_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(c.metadata["source"] for c in self._index.chunks).items()))

    def ocr_sources(self) -> set[str]:
        return {c.metadata["source"] for c in self._index.chunks if c.metadata.get("ocr")}

    def export(self, source: str) -> tuple[list[Document], np.ndarray | None]:
        """Chunks and their vectors for one source, for persistence."""
        index = self._index
        positions = [i for i, c in enumerate(index.chunks) if c.metadata["source"] == source]
        vectors = index.vectors[positions] if index.vectors is not None else None
        return [index.chunks[i] for i in positions], vectors

    def add(self, chunks: list[Document], vectors: np.ndarray | None = None, embed: bool = True) -> None:
        """Index chunks, replacing any existing chunks from the same source files.

        Pass `vectors` when restoring from storage to skip re-embedding, or `embed=False`
        when the vectors live in the database (pgvector) and needn't be held in memory.
        """
        if vectors is None and embed and self._embedder is not None and chunks:
            vectors = self._embed([c.page_content for c in chunks])

        incoming = {chunk.metadata["source"] for chunk in chunks}
        index = self._index
        keep = [i for i, c in enumerate(index.chunks) if c.metadata["source"] not in incoming]
        merged = [index.chunks[i] for i in keep] + list(chunks)

        merged_vectors = None
        if vectors is not None and (index.vectors is not None or not keep):
            kept_vectors = index.vectors[keep] if keep else np.empty((0, vectors.shape[1]), np.float32)
            merged_vectors = np.vstack([kept_vectors, vectors]).astype(np.float32)
        self._index = _Index.build(merged, merged_vectors)

    def remove(self, source: str) -> bool:
        index = self._index
        keep = [i for i, c in enumerate(index.chunks) if c.metadata["source"] != source]
        if len(keep) == len(index.chunks):
            return False
        vectors = index.vectors[keep] if index.vectors is not None else None
        self._index = _Index.build([index.chunks[i] for i in keep], vectors)
        return True

    def search(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        index = self._index
        lexical = self._bm25(index, query)
        dense = self._dense(index, query) if self.hybrid else []
        if not dense:
            return [(index.chunks[i], score) for i, score in lexical[:k]]

        fused: Counter = Counter()
        for ranking in (lexical, dense):
            for rank, (position, _) in enumerate(ranking):
                fused[position] += 1 / (RRF_K + rank + 1)
        return [(index.chunks[i], score) for i, score in fused.most_common(k)]

    def _embed(self, texts: list[str]) -> np.ndarray | None:
        try:
            return self._embedder.embed_documents(texts)
        except Exception as exc:
            # A missing model download shouldn't make documents unsearchable.
            logger.warning("Embedding failed (%s); falling back to keyword search only", exc)
            self._embedder = None
            return None

    def _bm25(self, index: _Index, query: str) -> list[tuple[int, float]]:
        terms = set(tokenize(query))
        n = len(index.chunks)
        if not n or not terms:
            return []

        idf = {
            term: math.log(1 + (n - index.doc_freq[term] + 0.5) / (index.doc_freq[term] + 0.5))
            for term in terms
            if index.doc_freq[term]
        }
        if not idf:
            return []

        scored = []
        for position, (freqs, length) in enumerate(zip(index.term_freqs, index.lengths, strict=True)):
            norm = self._k1 * (1 - self._b + self._b * length / index.avg_length)
            score = sum(
                weight * freqs[term] * (self._k1 + 1) / (freqs[term] + norm)
                for term, weight in idf.items()
                if freqs[term]
            )
            if score > 0:
                scored.append((position, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored

    def _dense(self, index: _Index, query: str) -> list[tuple[int, float]]:
        try:
            query_vector = self._embedder.embed_query(query)
        except Exception as exc:
            logger.warning("Query embedding failed (%s); using keyword search only", exc)
            return []
        if self._vector_search is not None:
            positions = {c.metadata.get("chunk_id"): i for i, c in enumerate(index.chunks)}
            try:
                hits = self._vector_search(query_vector, _DENSE_CANDIDATES)
            except Exception as exc:
                logger.warning("Vector search failed (%s); using keyword search only", exc)
                return []
            return [(positions[chunk_id], score) for chunk_id, score in hits if chunk_id in positions]
        if index.vectors is None:
            return []
        similarities = index.vectors @ query_vector
        top = np.argsort(-similarities)[:_DENSE_CANDIDATES]
        return [(int(i), float(similarities[i])) for i in top]
