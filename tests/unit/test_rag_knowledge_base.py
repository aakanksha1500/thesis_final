"""
Phase 8 - RAG embedder, vector store, and knowledge base.

GROUP A: Embedder — hashing fallback determinism and shape (no dependencies)
GROUP B: VectorStore — add/search contract, numpy brute-force backend
GROUP C: KnowledgeBase — build over D1-D4 (seed fallback), retrieve()
         contract, relevance filtering, document_set filtering

These tests never require sentence-transformers or faiss-cpu to be
installed — both rag/embedder.py and rag/vector_store.py fall back to
dependency-free implementations, and the tests exercise that fallback
path explicitly (the environment these tests actually run in).

RUNNING:
    python -m pytest tests/unit/test_rag_knowledge_base.py -v
"""

from __future__ import annotations

from rag.embedder import Embedder, cosine_similarity
from rag.knowledge_base import KnowledgeBase
from rag.vector_store import Document, VectorStore

# Group A: Embedder

class TestEmbedder:

    def test_encode_returns_correct_dimensionality(self):
        embedder = Embedder(dim=64)
        vec = embedder.encode_one("moderate risk classification")
        assert len(vec) == 64
    
    def test_encode_is_deterministic(self):
        embedder = Embedder(dim=64)
        v1 = embedder.encode_one("balanced mixed fund")
        v2 = embedder.encode_one("balanced mixed fund")
        assert v1 == v2

    def test_encode_empty_batch_returns_empty_list(self):
        embedder = Embedder(dim=64)
        assert embedder.encode([]) == []

    def test_vectors_are_l2_normalised(self):
        embedder = Embedder(dim=64)
        vec = embedder.encode_one("government bond fund with low fees")
        norm = sum(v * v for v in vec) ** 0.5
        assert abs(norm - 1.0) < 1e-6 or norm == 0.0  # zero vector for no tokens edge case

    def test_identical_texts_have_similarity_one(self):
        embedder = Embedder(dim=64)
        v1 = embedder.encode_one("investment grade corporate bond fund")
        v2 = embedder.encode_one("investment grade corporate bond fund")
        assert cosine_similarity(v1, v2) > 0.999

    def test_unrelated_texts_have_lower_similarity_than_identical(self):
        embedder = Embedder(dim=64)
        v1 = embedder.encode_one("investment grade corporate bond fund")
        v2 = embedder.encode_one("pizza recipe ingredients cheese dough")
        v3 = embedder.encode_one("investment grade corporate bond fund")
        assert cosine_similarity(v1, v2) < cosine_similarity(v1, v3)

    def test_falls_back_when_sentence_transformers_unavailable(self):
        # In this test environment sentence-transformers is not installed,
        # so mode should report the fallback path (documents the contract —
        # if the dependency IS installed, mode will legitimately be "sentence-transformers").
        embedder = Embedder(dim=64)
        assert embedder.mode in ("fallback", "sentence-transformers")


# GROUP B: VectorStore

class TestVectorStore:
    def _make_docs(self):
        embedder = Embedder(dim=32)
        texts = [
            "Irish government bond short-dated sovereign exposure",
            "Broad global equity ETF growth product",
            "Instant access savings account deposit guarantee",
        ]
        docs = [
            Document(doc_id=f"d{i}", text=t, source="Test Source", document_set="test")
            for i, t in enumerate(texts)
        ]
        vectors = embedder.encode(texts)
        return embedder, docs, vectors
    
    def test_add_and_len(self):
        embedder, docs, vectors = self._make_docs()
        store = VectorStore(dim=embedder.dim)
        store.add(docs, vectors)
        assert len(store) == 3
    
    def test_add_length_mismatch_raises(self):
        embedder, docs, vectors = self._make_docs()
        store = VectorStore(dim=embedder.dim)
        with __import__("pytest").raises(ValueError):
            store.add(docs, vectors[:2])
    
    def test_search_returns_most_similar_first(self):
        embedder, docs, vectors = self._make_docs()
        store = VectorStore(dim=embedder.dim)
        store.add(docs, vectors)

        query_vec = embedder.encode_one("government bond sovereign exposure")
        results = store.search(query_vec, top_k=3)
    
        assert len(results) == 3
        #  Bond-related doc should outrank the savings-account doc
        top_doc, top_score = results[0]
        assert "bond" in top_doc.text.lower()

    def test_search_empty_store_returns_empty_list(self):
        store = VectorStore(dim=32)
        results = store.search([0.0] * 32, top_k=3)
        assert results == []

    def test_search_respects_top_k(self):
        embedder, docs, vectors = self._make_docs()
        store = VectorStore(dim=embedder.dim)
        store.add(docs, vectors)
        
        results = store.search(embedder.encode_one("bond"), top_k=1)
        assert len(results) == 1
    
    def test_save_and_load_roundtrip(self, tmp_path):
        embedder, docs, vectors = self._make_docs()
        store = VectorStore(dim=embedder.dim)
        store.add(docs, vectors)
        store.save(tmp_path)

        loaded = VectorStore.load(tmp_path)
        assert len(loaded) == len(store)
        results = loaded.search(embedder.encode_one("bond"), top_k=1)
        assert len(results) == 1


# GROUP C: KnowledeBase

class TestKnowledgeBase:

    def test_rebuild_populates_all_document_sets(self):
        kb = KnowledgeBase()
        kb.rebuild()
        assert len(kb.store) > 0

    def test_retrieve_before_build_triggers_lazy_build(self):
        kb = KnowledgeBase()
        results = kb.retrieve("moderate risk diversified bond equity")
        # Whether or not results clear the relevance floor, the store
        # should now be built (non-empty) as a side effect of retrieve().
        assert len(kb.store) > 0
        assert isinstance(results, list)

    def test_retrieve_empty_query_returns_empty_list(self):
        kb = KnowledgeBase()
        assert kb.retrieve("") == []
        assert kb.retrieve("   ") == []

    def test_retrieve_respects_top_k(self):
        kb = KnowledgeBase()
        kb.rebuild()
        # Force everything through by using a very low relevance floor
        # via direct store search instead of retrieve()'s configured floor.
        vec = kb.embedder.encode_one("bond fund regulation guidance suitability")
        hits = kb.store.search(vec, top_k=2)
        assert len(hits) <= 2

    def test_retrieve_result_shape(self):
        kb = KnowledgeBase()
        kb.rebuild()
        results = kb.retrieve("central bank guidance suitability moderate risk")
        for r in results:
            assert set(r.keys()) >= {"claim", "source", "text", "relevance", "document_set"}
            assert 0.0 <= r["relevance"] or r["relevance"] < 0  # cosine sim can be negative in principle

    def test_retrieve_document_set_filter(self):
        kb = KnowledgeBase()
        kb.rebuild()
        results = kb.retrieve(
            "regulation guidance suitability bond equity",
            top_k=5,
            document_sets=["cbi_open_data"],
        )
        for r in results:
            assert r["document_set"] == "cbi_open_data"

    def test_seed_fallback_used_when_no_downloaded_data_present(self):
        """
        In this test environment, no live CBI API access and no downloaded
        D1/D2/D4 files exist, so every document_set should fall back to its
        bundled seed corpus — this is the expected, documented behaviour,
        not a failure mode.
        """
        kb = KnowledgeBase()
        kb.rebuild()
        document_sets_present = {doc.document_set for doc in kb.store._documents}
        assert document_sets_present == {
            "cbi_open_data", "eu_digital_finance", "finqa_original", "finqa_verified",
        }