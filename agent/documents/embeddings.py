import logging
import os
import threading
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Returns L2-normalised float32 vectors, so a dot product is cosine similarity."""

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


class FastEmbedEmbedder:
    """Local ONNX embeddings via fastembed (no PyTorch). The model loads on first use."""

    def __init__(self, model_name: str, cache_dir: str | None = None):
        self.model_name = model_name
        self._cache_dir = cache_dir or os.getenv("FASTEMBED_CACHE_PATH")
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding

                logger.info("Loading embedding model %s", self.model_name)
                self._model = TextEmbedding(self.model_name, cache_dir=self._cache_dir)
        return self._model

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return normalize(np.array(list(self._load().embed(texts))))

    def embed_query(self, text: str) -> np.ndarray:
        return normalize(np.array(list(self._load().query_embed(text))))[0]
