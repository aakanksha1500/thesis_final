# Orchestrated Multi-Agent Conversational AI for Transparent Financial Decision Support.

**MSC Computer Science (AI) - University of Galway**
**Student:** Aakanksha Shyam Deshpande (25238088)
**Supervisor:** Adrian Clear

---

## System Overview

A hierarchical multi-agent conversational financial advisor with 5 specialist agents coordinated by a central Orchestrator. The system implements HALO-Style orchestration, in-pipeline explainability, runtie hallucination detection, and TRiSM-compliant audit logging.

---

## Evaluation data: three layers, kept separate

Every dataset in this project belongs to exactly one of three layers. The
separation is enforced in code (`evaluation/datasets.py`), not by
convention, because the failure it prevents — computing an accuracy
number on data that has no labels — produces a result that looks
perfectly valid and is worthless.

| Layer | What it is | n | Used for | Never used for |
|---|---|---|---|---|
| **1 — Real / operational** | `data/processed/customers.csv`: the population the running system serves (3 hand-completed `DEMO_*`, 1 GMSC-derived, 250 `SYNTH_*` load-test, 1 session row) | 254 | robustness, coverage, edge cases, real-data behaviour | accuracy, precision, recall, F1 |
| **2 — Synthetic stress** | `data/evaluation/stress_customers.json`: full factorial over 5 risk classes × 6 coverage tiers, plus 100 tagged edge cases | 1,000 | stress and robustness across the whole feature space | accuracy, precision, recall, F1 |
| **3 — Gold standard** | `data/evaluation/gold_risk_profiles.json`: 40 profiles per risk class, labelled by an independent rubric, plus the 15 original hand-labelled profiles kept separate | 200 (+15) | accuracy, precision, recall, F1, confusion matrix, CIs | — |

**251 of the 254 Layer-1 customers have no risk label, and none was
invented for them.** The three that do are hand-completed demo profiles;
they are stored under `demo_label_if_present`, deliberately *not* under a
`ground_truth_*` key, so no accuracy computation can pick them up by
field-name match.

**Layer 2 profiles carry `construction_target_class`** — a record of
which rubric cell each profile was *built* from. That is a construction
record, not an observation. `load_stress_customers()` returns a dataset
whose layer has `has_ground_truth=False`; calling `ground_truth()` on it
raises `GroundTruthUnavailable`.

### The labelling instrument

`evaluation/risk_rubric.py` assigns every Layer-3 label. It is a two-axis
MiFID II / CBI suitability band matrix:

- **risk capacity** (objective: income, debt, dependents, employment,
  horizon) → integer points → low / medium / high
- **risk tolerance** (self-reported: loss tolerance, financial knowledge)
  → integer points → low / medium / high
- a 3×3 matrix in which **capacity caps tolerance** — a low-capacity
  client cannot be labelled above `moderately_conservative` regardless of
  stated willingness

It imports nothing from `agents/`, never loads `risk_model.pkl`, never
computes a SHAP value, and never observes a model prediction. Those
properties are enforced by AST-level tests
(`tests/unit/test_evaluation_layers.py::TestRubricIndependence`), not by
comment.

**Shared structure, stated up front.** The rubric and the agent's rule
layer both encode the same regulatory constructs, because the regulation
names those inputs. They differ in functional form (ordinal band matrix
vs. continuous weighted score with equal-width binning), and the rubric
shares nothing at all with the ML layer. Measured agreement on the gold
set: **rubric vs rule layer 0.295**, **rubric vs ML layer 0.360** —
barely above the 0.20 chance rate for five balanced classes. The rubric
also reproduces **10/15** of the original hand labels exactly and
**15/15** within one tier. Both diagnostics are reported in every RQ1
results file rather than left for a reader to ask about.

### Construction rules that were followed, and can be checked

1. Ground truth is assigned **before** the model is run, by an instrument
   that cannot see the model.
2. No profile is ever discarded for being predicted incorrectly.
3. No label is ever revised in response to model output.
4. Layer-2 construction targets are never promoted to ground truth.
5. Every dataset carries its generator, seed, git SHA and build time.
6. Every label in the gold file can be re-derived from its stored
   features — a test does exactly that for all 200.

---

## Running the evaluations

```bash
# Datasets (deterministic, seeded, no LLM)
python scripts/build_gold_risk_dataset.py        # Layer 3 — 200 profiles
python scripts/build_stress_customer_base.py     # Layer 2 — 1,000 profiles

# RQ1 — the headline accuracy result. No API key needed: the
# classification path is fully deterministic.
python scripts/eval_rq1_gold.py

# Layers 1 + 2 — robustness only. Computes no accuracy metric, and cannot.
python scripts/eval_robustness_layers.py

# RQ5 — all 91 FinQA Verified test items, 3 grounding conditions.
# Needs a live API key; without one the results are written with
# is_placeholder=true.
python scripts/eval_rq5_finqa_verified.py

# Banking77 — stratified n=500, 1 LLM call per item.
EVAL_LIVE_API=1 python scripts/eval_banking77_stratified.py --n 500
python scripts/eval_banking77_stratified.py --dry-run   # sampling only

# Everything, in dependency order
python scripts/regenerate_evidence.py
```

### Confidence intervals

`evaluation/bootstrap.py` provides 1,000-resample percentile bootstrap
CIs at 95%, seeded and reproducible. Paired metrics resample
`(prediction, label)` positions so pairing is preserved; ablation
comparisons over identical items use a paired difference, so the interval
answers "does grounding help" rather than "are these two samples
different".

---

## RQ5: what changed and why

The previous RQ5 evaluation
(`tests/unit/test_rq5_finqa_evaluation.py`) reported exact match 1.00
with RAG against 0.40 without, over 10 items. Reading that file's own
code: `_with_rag_predict` is a hand-written function that computes the
correct answer from the context — with a unit test asserting it scores
10/10 — and `_no_rag_predict` is documented as "always subtracts,
regardless of what the question actually asks for". Neither condition
ever called a model, and the 10 items were synthetic FinQA-*style*
questions rather than FinQA.

`scripts/eval_rq5_finqa_verified.py` replaces it: all **91 items of the
real FinQA Verified test split**, the same questions in the same order in
every condition, answered by the actual model.

| Condition | Grounding |
|---|---|
| `no_rag` | none |
| `rag_retrieved` | top-3 chunks retrieved from a contexts-only corpus of all 91 documents |
| `rag_oracle` | the item's own gold context |

Three conditions instead of two because the old binary design conflated
two different claims. `rag_oracle − no_rag` measures whether grounding
helps; `rag_oracle − rag_retrieved` measures what retrieval costs.
Observed own-document retrieval rate with the fallback hashing embedder:
**0.780** — i.e. 20% of items cannot be answered from grounding no matter
how good the reader is.

**Train data is not mixed in.** `finqa_original_train.json` (6,251 items)
is present for RAG corpus construction and is never loaded by
`evaluation/finqa_data.py::load_finqa_verified`. Two verified questions
also occur as strings in the train split (FinQA Verified re-annotates
original FinQA data); zero train *items* are evaluated, and the check
that establishes this is reported in every RQ5 results file.

**A leakage trap worth knowing about.**
`rag/knowledge_base.py::_load_finqa_verified` indexes verified items as
`"Question: … Verified answer: …"` — the answer goes into the retrievable
corpus. The RQ5 harness therefore builds its own in-memory corpus from
the `context` field only and never touches the shared index.

---

## Banking77: what changed and why

Two things.

**The harness had been lost.** In the snapshot this work started from,
`tests/unit/test_banking77_evaluation.py` was byte-identical to
`tests/unit/test_approval_gate.py` (same md5). The Banking77 evaluation
had been overwritten by a copy of the approval-gate tests, and pytest was
collecting and running that duplicate twice under two names — which is
why nothing failed. Only the outputs survived. The harness is rebuilt in
`scripts/eval_banking77_stratified.py` + `evaluation/banking77_data.py`.

**The split is `train`, and the results say so.** Only one arrow file is
cached (9,993 rows, `state.json` says `_split: "train"`); the 3,076-row
test split is described in `dataset_info.json` but is not in the repo.
Using the train split is sound *here* because the intent classifier is
**zero-shot** — no Banking77 data is used for training, fine-tuning or
few-shot prompting anywhere in the project, so train utterances are
unseen text to it. `assert_no_prompt_leakage()` verifies that claim
against the actual prompt (0 sampled utterances found in it) rather than
asserting it. If the test split is ever downloaded, the loader prefers it
automatically.

Sampling is **stratified over all 77 fine-grained intents** rather than
over the 3 reachable buckets: this preserves bucket balance automatically
(population 0.841 / 0.147 / 0.012 → sample 0.842 / 0.146 / 0.012) *and*
guarantees no individual intent drops out, which bucket-level
stratification would allow. Allocation is proportional with
largest-remainder rounding; the seed is fixed; **no label was changed**.

Five of the eight buckets have zero Banking77 ground truth by
construction, so the relevant signal for them is **false-positive leakage
into advisory buckets** — a "where is my card" question routed to the
InvestmentAgent — which is reported explicitly. Those five buckets are
evaluated properly by the 110-item hand-authored corpus in
`tests/unit/test_core_intent_evaluation.py`.

---

## Known limitations

Stated here rather than discovered in a viva.

- **RQ1's headline numbers are low, and are reported unchanged.**
  Accuracy 0.380 [0.315, 0.445] and macro F1 0.345 [0.281, 0.402] on 200
  balanced profiles, against RAR 0.940 and AUC-ROC 0.803. The pattern is
  diagnosable, not mysterious: `aggressive` recall is 0.025 (1 of 40) and
  `conservative` recall is 0.325, because `hybrid_score` is compressed
  toward the centre while `_score_to_class` cuts [0,1] into five
  equal-width bands — so the outer tiers are close to unreachable. High
  AUC (ranking) with low per-class recall (thresholding) is the
  signature of a **calibration** problem, not a ranking one. The
  `score_distribution_diagnostic` block in the results file contains the
  evidence. Nothing was retuned to improve these numbers.
- **RQ5 and Banking77 numbers in `results/mock/` are placeholders.** They
  were generated without an API key; the mock client does no numerical
  reasoning and returns a fixed intent bucket. Both files carry
  `is_placeholder: true`. Re-run with `GROQ_API_KEY` set before reporting
  anything from them. (The mock Banking77 run is instructive anyway:
  accuracy 0.842 with macro F1 0.114 — a degenerate always-majority
  classifier that raw accuracy alone would have flattered.)
- **RQ1 numbers here were produced under scikit-learn 1.7.2**, while
  `risk_model.pkl` was pickled under 1.9.0 (which needs Python ≥3.11).
  Unpickling emits `InconsistentVersionWarning`. Both versions are
  recorded in the results file; regenerate on your own machine before
  final reporting.
- **The gold set is synthetic-but-independently-labelled**, not real
  customers. It establishes that the classifier's behaviour is measured
  against an instrument it cannot influence. It does not establish
  external validity against a real client population, and no claim to
  that effect should be made.
- **RQ3's transparency and trust scores remain a synthetic survey
  fixture.** No amount of extra data fixes this; it needs the n≥20 pilot
  the results file already flags, or a reframing around
  machine-checkable proxies.
- **RQ2 remains n=5 queries** over a 29-product catalogue. Untouched by
  this work — the bottleneck there is relevance labelling, not sample
  size.
- **Age-neutrality was an open design question — it has since been
  resolved, and this section previously didn't say so.**
  `test_trained_model_age_effect_is_empirical_not_assumed` no longer
  exists in `tests/unit/test_risk_profiling_agent.py`; it was replaced
  by `test_age_neutral_capacity_is_invariant_to_age`, whose own
  docstring records why: age carries 0.48 feature importance in the
  GMSC distress model, and letting a protected characteristic drive
  nearly half of a suitability assessment isn't defensible under EU AI
  Act / CBI model-risk expectations. `scripts/
  recalibrate_risk_model_age_neutral.py` removes age's own SHAP
  contribution from the capacity score; after that, P(distress) for
  ages 25 and 65 differ by 0.002 and land in the same percentile
  bucket, so capacity is identical by design. That's still measured
  behaviour worth stating plainly in a results chapter, but it is no
  longer an open question with a test left failing to mark it — the
  decision was made (age-neutral) and the test suite was updated to
  match, before this section was. If a viva question raises this,
  the answer is "resolved in favour of age-neutrality, for the fairness
  reason above" — not "still undecided."