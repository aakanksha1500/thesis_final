"""
Phase - 8: Text embedding layer for the RAG knowledge base.

REAL-MODE: sentence-transformers/all-MiniLM-L6-v2
FALLBACK MODE: IDF-weighted hashing-trick bag-of-words embedding -
    deterministic, dependency-free, same output dimensionality,
    cosine-comparable. Not semantically strong (no paraphrase/synonym
    matching), but a corpus-fitted IDF weighting materially improves
    lexical discrimination over unweighted hashing: common cross-topic
    words (e.g. boilerplate financial vocabulary shared by every
    document_set) are down-weighted, while words that are rare across
    the corpus and therefore topically distinctive are up-weighted.
    Still not a substitute for a real semantic model — call fit_idf()
    with the full corpus before encoding to activate the weighting;
    without it, this degrades gracefully to unweighted hashing.

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
        force_fallback: bool = False,
    ):
        self.model_name = model_name or settings.rag.embedding_model
        self.dim = dim or settings.rag.embedding_dim
        self._requested_dim = dim          # None means "no explicit preference"
        self._model = None
        self._mode = "fallback"
        self._force_fallback = force_fallback
        self._idf: list[float] | None = None
        self._idf_n_docs: int = 0
        self._init_model()

    def _init_model(self) -> None:
        if self._force_fallback:
            logger.info(
                f"[Embedder] force_fallback=True — using hashing embedder "
                f"(dim={self.dim}) regardless of environment."
            )
            return

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            model_dim = (
                self._model.get_embedding_dimension()
                if hasattr(self._model, "get_embedding_dimension")
                else self._model.get_sentence_embedding_dimension()
            )

            if self._requested_dim is not None and self._requested_dim != model_dim:
                logger.warning(
                    f"[Embedder] dim={self._requested_dim} was requested but "
                    f"'{self.model_name}' emits {model_dim}-dim vectors. Using "
                    f"{model_dim}. Pass force_fallback=True if you need the "
                    f"hashing embedder at an exact dimensionality."
                )
            self.dim = model_dim
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

    # IDF fitting (fallback mode only) - persistence support

    def fit_idf(self, corpus_texts: Sequence[str]) -> None:
        """
        Compute per-bucket inverse document frequency over `corpus_texts`
        and activate IDF weighting for subsequent _hash_embed() calls.

        Only meaningful in fallback mode - a no-op call in sentence-
        transformers mode does no harm but has no effect on encode().
        Must be called with the SAME corpus used to build the index
        (typically all document chunks), and the resulting weights must
        be persisted (get_idf) and restored (set_idf) before encoding
        queries against a loaded index, or query and document vectors
        drift into different weightings.
        """
        n_docs = len(corpus_texts)
        if n_docs == 0:
            self._idf = None
            self._idf_n_docs = 0
            return

        doc_freq = [0] * self.dim
        for text in corpus_texts:
            tokens = {
                t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS
            }
            buckets = {
                int(hashlib.sha256(t.encode("utf-8")).hexdigest()[:8], 16) % self.dim
                for t in tokens
            }
            for bucket in buckets:
                doc_freq[bucket] += 1

        # Smoothed IDF, sklearn-style: log((n+1)/(df+1)) + 1 - always
        # positive, and a bucket that appears in every document (df == n)
        # still gets weight 1 rather than collapsing to 0.
        self._idf = [
            math.log((n_docs + 1) / (df + 1)) + 1.0 for df in doc_freq
        ]
        self._idf_n_docs = n_docs
        logger.info(
            f"[Embedder] Fitted IDF weighting over {n_docs} documents "
            f"(dim={self.dim})."
        )

    def get_idf(self) -> list[float] | None:
        """Return the fitted IDF vector, for persistence alongside the index."""
        return self._idf

    def set_idf(self, idf: list[float] | None) -> None:
        """Restore a previously-fitted IDF vector (e.g. after loading a
        persisted index), so query-time encoding matches document-time
        encoding."""
        if idf is not None and len(idf) != self.dim:
            logger.warning(
                f"[Embedder] set_idf() vector length {len(idf)} != dim "
                f"{self.dim} - ignoring, falling back to unweighted hashing."
            )
            return
        self._idf = idf

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
            weight = self._idf[bucket] if self._idf is not None else 1.0
            vec[bucket] += sign * weight

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
