"""
Phase - 8: Text embedding layer for the RAG knowledge base.

REAL-MODE: sentence-transformers/all-MiniLM-L6-v2
FALLBACK MODE: hashing-trick bag-og-words embedding - deterministic,
    dependency-free, same output dimensionality, cosine-comparable.
    Not semantically strong, but sufficient to exercise retrieval, ranking,
    and ablation-condition plumbing without the model download.
    
The fallback is intentional and is flagged loudly, same as LLMClient - a reader
of the logs should never wonder which mode ran.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Sequence

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "based", "be", "by", "for", "from",
    "has", "have", "if", "in", "into", "is", "it", "its", "of", "on", "or",
    "should", "than", "that", "the", "this", "to", "was", "we", "were",
    "will", "with", "your", "you",
})

class Embedder:
    """
    The wrapper around a sentence-embedding model.
    
    Args:
        model_name: sentence-transformers model id. Defaults to 
                    settings.rag.embedding_model.
        dim: output vector dimensionality. Defaults to settings.
            rag.embedding_dim. The fallback embedder always produces
            vectors of exactly this length so REAL and FALLBACK vectors
            are interchangeable in a persisted index.
    """

    def __init__(
        self,
        model_name: str | None = None,
        dim: int | None = None,
    ):
        self.model_name = model_name or settings.rag.embedding_model
        self.dim = dim or settings.rag.embedding_dim
        self._model = None
        self._mode = "fallback"
        self._init_model()

    def _init_model(self) -> None:

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self.dim = self._model.get_sentence_embedding_dimension()
            self._mode = "sentence-transformers"
            logger.info(
                f"[Embedder] Initialised in REAL mode — model={self.model_name} "
                f"dim={self.dim}"
            )
        except ImportError:
            logger.warning(
                "[Embedder] sentence-transformers not initialised - running in "
                "FALLBACK mode (hashing embedder). Add sentence-transformers "
                "to requirements.txt and pip install for real embeddings."
            )
        except Exception as exc:
            logger.warning(
                f"[Embedder] Failed to load '{self.model_name}': {exc} - "
                f"falling back to hashing embedder."
            )

    @property
    def mode(self) -> str:
        return self._mode

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Embed a batch of texts. Returns one L2-normalised vector per input,
        in input order, regardless of which mode is active.
        """
        if not texts:
            return []

        if self._mode == "sentence-transformers" and self._model is not None:
            vectors = self._model.encode(
                list(texts), normalize_embeddings=True, show_progress_bar=False
            )
            return [v.tolist() for v in vectors]

        return [self._hash_embed(t) for t in texts]

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]

    # Fallback embedder - deterministic hashing trick

    def _hash_embed(self, text: str) -> list[float]:
        """
        Deterministic bag-of-words hashing embedding.
        
        Each token is hashed into one of self.dim` buckets (sign determined
        by a second hash, standard hashing-trick practice to reduce
        collision bias). The resulting vector is L2-normalised so cosine
        similarity behaves the same way it would for a real model's output.

        Not a substitute for semantic embeddings — two paraphrases with no
        shared tokens will not match. This is a structural stand-in so the
        retrieval pipeline (chunking, indexing, top-k, relevance filtering)
        is fully testable without the model dependency.
        """
        vec = [0.0] * self.dim
        tokens = [
            t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS
        ]
        if not tokens:
            return vec

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            bucket = int(digest[:8], 16) % self.dim
            sign = 1.0 if int(digest[8:9], 16) % 2 == 0 else -1.0
            vec[bucket] += sign

        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]

def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two equal-length vectors, in [-1, 1]."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
