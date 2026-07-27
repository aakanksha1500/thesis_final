"""
Phase 8 - Replaces the Phase 6 ExplainabilityAgent layer B stub
with a real FAISS vector store over four document sets.

The bundled seed corpora are explicitly syenthetic/illustrative - same
disclosure pattern as IRISH_PRODUCT_CATALOGUE in agents/investment_agent.py.
They exist so 'retrieved()' always returns something real to score against.

Knowledge.retrieve() is the single entry point Explainability Agent 
Layer B calls.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

from config.settings import ROOT_DIR, settings
from rag.embedder import Embedder
from rag.vector_store import Document, VectorStore
from utils.logger import get_logger

logger = get_logger(__name__)

CBI_API_URL = "https://www.centralbank.ie/opendata/api/rates"
CBI_API_TIMEOUT_SECONDS = 5

DATA_RAW = ROOT_DIR / "data" / "raw"

# Bundled seed corpora — synthetic/illustrative, used when live/downloaded
# data is unavailable. Kept small and clearly labelled per document_set.

_SEED_CBI_OPEN_DATA: list[dict[str, str]] = [
    {
        "doc_id": "cbi-seed-001",
        "text": (
            "The Central Bank of Ireland's Consumer Protection Code requires "
            "regulated firms to ensure that any information provided to a "
            "consumer is clear, accurate, and not misleading, and that "
            "product suitability is assessed against the consumer's "
            "financial situation, needs, and objectives before a "
            "recommendation is made."
        ),
    },
    {
        "doc_id": "cbi-seed-002",
        "text": (
            "Deposit Guarantee Scheme (DGS) protection in Ireland covers "
            "eligible deposits up to EUR 100,000 per depositor per credit "
            "institution. Term deposits and instant-access savings accounts "
            "issued by DGS-participating institutions fall within this cover."
        ),
    },
    {
        "doc_id": "cbi-seed-003",
        "text": (
            "Central Bank of Ireland guidance on suitability assessments "
            "states that a moderate risk tolerance classification should "
            "map to a diversified mix of government and investment-grade "
            "corporate bonds, broad equity exposure, and balanced multi-asset "
            "funds, avoiding concentration in single-issuer or high-yield "
            "instruments."
        ),
    },
    {
        "doc_id": "cbi-seed-004",
        "text": (
            "Retail investors should be informed that past performance of "
            "a financial product is not a reliable indicator of future "
            "results, and that all investment products carry a risk of "
            "capital loss except where explicitly covered by a statutory "
            "guarantee scheme."
        ),
    },
]

_SEED_EU_DIGITAL_FINANCE: list[dict[str, str]] = [
    {
        "doc_id": "eu-seed-001",
        "text": (
            "The Digital Operational Resilience Act (DORA) requires "
            "financial entities operating in the EU to implement a "
            "comprehensive ICT risk management framework, including "
            "incident reporting, resilience testing, and oversight of "
            "critical third-party ICT providers, effective from January 2025."
        ),
    },
    {
        "doc_id": "eu-seed-002",
        "text": (
            "Under MiFID II, firms providing investment advice must act "
            "honestly, fairly, and professionally in accordance with the "
            "best interests of the client, and must assess the suitability "
            "of a recommendation based on the client's knowledge, financial "
            "situation, and investment objectives."
        ),
    },
    {
        "doc_id": "eu-seed-003",
        "text": (
            "ECB macroeconomic projections for the euro area point to "
            "policy interest rate movements as the primary transmission "
            "channel affecting short-duration money market and term-deposit "
            "yields across member states, with government bond yields more "
            "sensitive to longer-horizon rate expectations."
        ),
    },
    {
        "doc_id": "eu-seed-004",
        "text": (
            "The EU Sustainable Finance Disclosure Regulation (SFDR) "
            "requires asset managers to classify financial products by the "
            "degree to which environmental or social characteristics are "
            "promoted, informing how mixed and equity funds may be "
            "represented to retail investors."
        ),
    },
]

_SEED_FINQA_ORIGINAL: list[dict[str, str]] = [
    {
        "doc_id": "finqa-orig-seed-001",
        "text": (
            "Context: Total revenue increased from $1,204 million in the "
            "prior year to $1,389 million in the current year. Question: "
            "what was the percentage increase in revenue year over year? "
            "Reasoning: (1389 - 1204) / 1204 = 15.4%."
        ),
    },
    {
        "doc_id": "finqa-orig-seed-002",
        "text": (
            "Context: Operating expenses were $342 million, comprising "
            "$210 million in personnel costs and the remainder in "
            "administrative costs. Question: what portion of operating "
            "expenses was administrative? Reasoning: (342 - 210) / 342 = 38.6%."
        ),
    },
]

_SEED_FINQA_VERIFIED: list[dict[str, str]] = [
    {
        "doc_id": "finqa-verif-seed-001",
        "text": (
            "Verified ground truth: given a portfolio with an expected "
            "annual return of 6.0% and an expense ratio of 0.6%, the net "
            "expected annual return is 5.4%, computed as 6.0% - 0.6%."
        ),
    },
    {
        "doc_id": "finqa-verif-seed-002",
        "text": (
            "Verified ground truth: a term deposit paying 2.5% annually, "
            "compounded once, on a principal of EUR 10,000 yields EUR 250 "
            "in interest after one year, before any deposit interest "
            "retention tax (DIRT) is applied."
        ),
    },
]

_SEED_BY_SET: dict[str, list[dict[str, str]]] = {
    "cbi_open_data": _SEED_CBI_OPEN_DATA,
    "eu_digital_finance": _SEED_EU_DIGITAL_FINANCE,
    "finqa_original": _SEED_FINQA_ORIGINAL,
    "finqa_verified": _SEED_FINQA_VERIFIED,
}

_SOURCE_LABEL: dict[str, str] = {
    "cbi_open_data": "CBI Open Data Portal [D3]",
    "eu_digital_finance": "EU Digital Finance Platform [D4]",
    "finqa_original": "FinQA Original [D2]",
    "finqa_verified": "FinQA Verified [D1]",
}

def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Fixed-size character chunking with overlap. Sentence-aware where possible."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= size:
        return [text] if text else []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks

class KnowledgeBase:
    """
    RAG Knowledge base - embeds and indexes D1-D4, exposes retrieve().
    
    Lazily builds its index on first use (from live/downloaded data where
    available, seed corpora otherwise) unless a persisted index already
    exists at settings.rag.index_dir, in which case it loads that instead.
    Call `rebuild()` to force a fresh build (used by
    scripts/build_knowledge_base.py after new data is downloaded).
    """
    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or Embedder()
        self.store = VectorStore(dim=self.embedder.dim)
        self._built = False

    # Build / Load

    def ensure_built(self) -> None:
        if self._built:
            return
        index_dir = settings.rag.index_dir
        if (index_dir / "meta.json").exists():
            try:
                self.store = VectorStore.load(index_dir)
                self._built = True
                logger.info(
                    f"[KnowledgeBase] Loaded persisted index - "
                    f"{len(self.store)} documents"
                )
                return
            except Exception as exc:
                logger.warning(
                    f"[KnowledgeBase] Failed to load persisted index: {exc} "
                    f"— rebuilding."
                )
        self.rebuild()

    def rebuild(self) -> None:
        """Fetch/load all document sets, embed, and index them from scratch."""
        logger.info("[KnowledgeBase] Building index over D1-D4...")
        documents: list[Document] = []

        for document_set in settings.rag.document_sets:
            raw_records = self._load_document_set(document_set)
            for record in raw_records:
                chunks = _chunk_text(
                    record["text"],
                    settings.rag.chunk_size_chars,
                    settings.rag.chunk_overlap_chars,
                )
                for i, chunk in enumerate(chunks):
                    documents.append(Document(
                        doc_id=f"{record['doc_id']}-{i}",
                        text=chunk,
                        source=_SOURCE_LABEL[document_set],
                        document_set=document_set,
                        metadata={"origin_doc_id": record["doc_id"]},
                    ))

        if documents:
            vectors = self.embedder.encode([d.text for d in documents])
            self.store = VectorStore(dim=self.embedder.dim)
            self.store.add(documents, vectors)

        self._built = True
        logger.info(
            f"[KnowledgeBase] Index built — {len(self.store)} chunks across "
            f"{len(settings.rag.document_sets)} document sets "
            f"(embedder mode={self.embedder.mode}, store backend={self.store._backend})"
        )

    def persist(self) -> None:
        self.store.save(settings.rag.index_dir)

    # Per-set loading — live fetch / downloaded file / seed fallback
    def _load_document_set(self, document_set: str) -> list[dict[str, str]]:
        if document_set == "cbi_open_data":
            return self._load_cbi_open_data()
        if document_set == "eu_digital_finance":
            return self._load_eu_digital_finance()
        if document_set == "finqa_original":
            return self._load_finqa_split("finqa_original_train.json", "finqa_original")
        if document_set == "finqa_verified":
            return self._load_finqa_verified()
        logger.warning(f"[KnowledgeBase] Unknown document_set '{document_set}' — skipping")
        return []

    def _load_cbi_open_data(self) -> list[dict[str, str]]:
        """
        [D3] Attempt a live fetch from the CBI Open Data Portal API.
        Falls back to the bundled seed corpus if unreachable — the API
        contract for opendata.centralbank.ie is not guaranteed stable
        enough to hard-depend on for reproducible test runs.
        """
        try:
            req = urllib.request.Request(
                CBI_API_URL, headers={"Accept": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=CBI_API_TIMEOUT_SECONDS) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            records = self._parse_cbi_payload(payload)
            if records:
                logger.info(f"[KnowledgeBase] Fetched {len(records)} live CBI records")
                return records
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            logger.info(
                f"[KnowledgeBase] CBI Open Data API unreachable ({exc}) — "
                f"using bundled seed corpus for [D3]."
            )
        return list(_SEED_CBI_OPEN_DATA)

    @staticmethod
    def _parse_cbi_payload(payload: Any) -> list[dict[str, str]]:
        """Best-effort parse of the CBI API's JSON shape into {doc_id, text}."""
        records = []
        items = payload.get("data", payload) if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            return records
        for i, item in enumerate(items):
            if isinstance(item, dict) and "description" in item:
                records.append({
                    "doc_id": f"cbi-live-{i:04d}",
                    "text": str(item["description"]),
                })
        return records

    def _load_eu_digital_finance(self) -> list[dict[str, str]]:
        """
        [D4] EU Digital Finance Platform requires manual CSV download per
        scripts/download_datasets.py. Loads any CSVs found under
        data/raw/eu_digital_finance/; falls back to seed corpus otherwise.
        """
        directory = DATA_RAW / "eu_digital_finance"
        if not directory.exists():
            return list(_SEED_EU_DIGITAL_FINANCE)

        records: list[dict[str, str]] = []
        try:
            import csv  # noqa: PLC0415

            for csv_path in sorted(directory.glob("*.csv")):
                with open(csv_path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for i, row in enumerate(reader):
                        text = " ".join(
                            f"{k}: {v}" for k, v in row.items() if v
                        )
                        if text.strip():
                            records.append({
                                "doc_id": f"{csv_path.stem}-{i:05d}",
                                "text": text,
                            })
        except Exception as exc:
            logger.warning(f"[KnowledgeBase] Failed reading EU Digital Finance CSVs: {exc}")

        return records if records else list(_SEED_EU_DIGITAL_FINANCE)

    def _load_finqa_split(self, filename: str, document_set: str) -> list[dict[str, str]]:
        """[D2] FinQA Original — loaded from data/raw/ if download_datasets.py --phase 8 has run."""
        path = DATA_RAW / filename
        if not path.exists():
            return list(_SEED_BY_SET[document_set])
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            records = []
            for i, item in enumerate(raw[:500]):  # cap for index build time
                pre = " ".join(item.get("pre_text", []))
                post = " ".join(item.get("post_text", []))
                question = item.get("qa", {}).get("question", "")
                text = f"{pre} {post} Question: {question}".strip()
                if text:
                    records.append({"doc_id": f"finqa-orig-{i:05d}", "text": text})
            return records if records else list(_SEED_BY_SET[document_set])
        except Exception as exc:
            logger.warning(f"[KnowledgeBase] Failed reading FinQA Original: {exc}")
            return list(_SEED_BY_SET[document_set])

    def _load_finqa_verified(self) -> list[dict[str, str]]:
        """[D1] FinQA Verified — loaded from data/raw/finqa_verified/ (HF dataset dir) if present."""
        path = DATA_RAW / "finqa_verified"
        if not path.exists():
            return list(_SEED_FINQA_VERIFIED)
        try:
            from datasets import load_from_disk  # noqa: PLC0415

            ds = load_from_disk(str(path))
            records = []
            for i, item in enumerate(ds.select(range(min(500, len(ds))))):
                question = item.get("question", "")
                answer = item.get("answer", "")
                text = f"Question: {question} Verified answer: {answer}".strip()
                if text:
                    records.append({"doc_id": f"finqa-verif-{i:05d}", "text": text})
            return records if records else list(_SEED_FINQA_VERIFIED)
        except Exception as exc:
            logger.warning(f"[KnowledgeBase] Failed reading FinQA Verified: {exc}")
            return list(_SEED_FINQA_VERIFIED)

    # Retrieval - the entry point ExplainabilityAgent Layer B calls

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_sets: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the top_k most relevant chunks for 'query'
        
        Args:
            query: Free-text query - typically buily by the caller
                    from the claim/synthesis text needing grounding.
            top_k: Defaults to settings.rag.top_k_citations.
            document_sets: Optional filter, e.g: ["cbi_open_data"] to restrict
                            citations to Irish regulatory sources only.

        Returns:
            List of {"class": query, "source":str, "text": srr
                "relevance": float, "document_set": str} dicts,
            filtered by settings.rag.min_relevance_score and stored by
            relevance descending. Matches the Phase 8 stub's documented return
            shape ({"claim", "source", "relevance"}) plus the retrieved "text"
            and "document_Set" fields Phase 8 adds.
        """
        self.ensure_built()
        if not query or not query.strip() or len(self.store) == 0:
            return []

        top_k = top_k or settings.rag.top_k_citations
        query_vector = self.embedder.encode_one(query)

        # Over-fetch when filtering by document_set so top_k is still met
        # after filtering, then trim back down.
        raw_top_k = top_k * 4 if document_sets else top_k
        hits = self.store.search(query_vector, top_k=raw_top_k)

        results = []
        for doc, score in hits:
            if document_sets and doc.document_set not in document_sets:
                continue
            if score < settings.rag.min_relevance_score:
                continue
            results.append({
                "claim": query,
                "source": doc.source,
                "text": doc.text,
                "relevance": round(float(score), 4),
                "document_set": doc.document_set,
                "doc_id": doc.doc_id,
            })
            if len(results) >= top_k:
                break

        return results

# Mosule level singleton - ExplainabilityAgent imports this directly so
# every call reuses the same in-memory index instead of rebuilding per call.
knowledge_base = KnowledgeBase()
