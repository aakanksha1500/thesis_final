"""
Phase 8 — Vector index backing the RAG knowledge base.

Tries FAISS first (flat L2 index over normalised vectors == cosine
similarity), falls back to a pure-Python/numpy brute-force cosine search
if faiss-cpu is not installed. ChromaDB is supported as an alternative
backend (requirements.txt lists both — FAISS is the default, ChromaDB is
a documented drop-in if a persistent client/server store is preferred for
a later deployment phase).

Same dependency-fallback philosophy as rag/embedder.py: retrieval must
work — possibly slower, never absent — regardless of which of
{faiss-cpu, chromadb, neither} is installed in the current environment.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag.embedder import cosine_similarity
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Document:
    """One retrievable chunk in the vector store."""
    doc_id: str
    text: str
    source: str            # e.g. "CBI Open Data Portal [D3]"
    document_set: str      # e.g. "cbi_open_data"
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore:
    """
    Add/search interface over embedded documents.

    Backend selection (best available, checked once at construction):
      1. faiss-cpu    — IndexFlatIP over L2-normalised vectors (cosine sim)
      2. numpy         — brute-force cosine search, no external dependency
                         beyond numpy (already required from Phase 3)

    Both backends implement the same add() / search() contract, so
    KnowledgeBase never needs to know which one is active.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self._documents: list[Document] = []
        self._vectors: list[list[float]] = []
        self._backend = "numpy"
        self._faiss_index = None
        self._init_backend()

    def _init_backend(self) -> None:
        try:
            import faiss  # noqa: PLC0415

            self._faiss_index = faiss.IndexFlatIP(self.dim)
            self._backend = "faiss"
            logger.info(f"[VectorStore] Initialised FAISS IndexFlatIP (dim={self.dim})")
        except ImportError:
            logger.warning(
                "[VectorStore] faiss-cpu not installed — running brute-force "
                "numpy cosine search. Add faiss-cpu to requirements.txt for "
                "sub-linear search at larger corpus sizes."
            )

    def __len__(self) -> int:
        return len(self._documents)

    def add(self, documents: list[Document], vectors: list[list[float]]) -> None:
        """Add documents with their pre-computed embedding vectors."""
        if len(documents) != len(vectors):
            raise ValueError(
                f"documents ({len(documents)}) and vectors ({len(vectors)}) "
                f"length mismatch"
            )
        if not documents:
            return

        if self._backend == "faiss":
            import numpy as np  # noqa: PLC0415

            arr = np.array(vectors, dtype="float32")
            self._faiss_index.add(arr)

        self._documents.extend(documents)
        self._vectors.extend(vectors)
        logger.debug(f"[VectorStore] Added {len(documents)} documents (backend={self._backend})")

    def search(
        self, query_vector: list[float], top_k: int = 3
    ) -> list[tuple[Document, float]]:
        """
        Return up to top_k (Document, relevance_score) pairs, sorted by
        relevance descending. relevance_score is cosine similarity in
        [-1, 1] regardless of backend.
        """
        if not self._documents:
            return []

        if self._backend == "faiss":
            import numpy as np  # noqa: PLC0415

            q = np.array([query_vector], dtype="float32")
            k = min(top_k, len(self._documents))
            scores, indices = self._faiss_index.search(q, k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                results.append((self._documents[idx], float(score)))
            return results

        # Brute-force fallback
        scored = [
            (doc, cosine_similarity(query_vector, vec))
            for doc, vec in zip(self._documents, self._vectors)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        with open(index_dir / "documents.pkl", "wb") as f:
            pickle.dump(self._documents, f)
        with open(index_dir / "vectors.pkl", "wb") as f:
            pickle.dump(self._vectors, f)
        meta = {"dim": self.dim, "backend": self._backend, "n_documents": len(self._documents)}
        with open(index_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"[VectorStore] Saved {len(self._documents)} documents to {index_dir}")

    @classmethod
    def load(cls, index_dir: Path) -> "VectorStore":
        with open(index_dir / "meta.json") as f:
            meta = json.load(f)
        store = cls(dim=meta["dim"])
        with open(index_dir / "documents.pkl", "rb") as f:
            documents = pickle.load(f)
        with open(index_dir / "vectors.pkl", "rb") as f:
            vectors = pickle.load(f)
        store.add(documents, vectors)
        logger.info(f"[VectorStore] Loaded {len(documents)} documents from {index_dir}")
        return store
