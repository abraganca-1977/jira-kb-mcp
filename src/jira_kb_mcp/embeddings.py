"""Local, offline embedding generation via fastembed (ONNX Runtime).

Default model is multilingual (paraphrase-multilingual-MiniLM-L12-v2) since
Jira tickets are frequently written in languages other than English. The
model downloads once from Hugging Face on first use and is cached locally
under ~/.cache/fastembed (or fastembed's own default cache dir).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    @property
    def dimension(self) -> int:
        return TextEmbedding.get_embedding_size(self.model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.embed(texts)
        return [np.asarray(v, dtype=np.float32).tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        # fastembed's query_embed applies model-specific query prefixes
        # (e.g. "query: ") when the underlying model expects them.
        vec = next(self._model.query_embed([text]))
        return np.asarray(vec, dtype=np.float32).tolist()


@lru_cache(maxsize=4)
def get_embedder(model_name: str = DEFAULT_MODEL) -> Embedder:
    """Cached factory so we don't reload the ONNX model repeatedly."""
    return Embedder(model_name=model_name)
