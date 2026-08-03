#!/bin/bash
# Checks one marker string per commit from this session — run from repo root:
#   bash verify_markers.sh
# "MISSING" means that file needs re-copying from the corresponding message
# in the conversation (search for the filename I gave there).

check() {
  if grep -q "$2" "$1" 2>/dev/null; then
    echo "OK      $3"
  else
    echo "MISSING $3  (expected '$2' in $1)"
  fi
}

check ".env.example" "your-groq-api-key-here" "Day 0: .env.example placeholder"
check "utils/llm_client.py" "_NO_AUTH_PROVIDERS" "Ollama provider"
check "explainability/explainability_agent.py" "_attribution_method_label" "Day 1a: SHAP tagging"
check "scripts/compare_risk_model_variants.py" "def compare" "Day 1b: comparison script"
check "tests/unit/test_hallucination_detector.py" "TestPremiseSanityCheck" "Day 1c: RQ5 fix + sanity test"
check "rag/hallucination_detector.py" "init_error" "HHEM diagnostics"
check "tests/unit/test_rq5_finqa_evaluation.py" "test_with_rag_predict_matches_every_fixture_item" "RQ5 comma-parsing fix"
check "scripts/generate_transactions.py" "SCENARIOS" "Day 2a: transaction generator"
check "agents/budget_agent.py" "_aggregate_transactions" "Day 2a: BudgetAgent aggregation"
check "config/settings.py" "min_aggregation_window_months" "12-month floor"
check "data/transaction_store.py" "class TransactionStore" "TransactionStore"
check "data/banking77_bucket_map.json" "bucket_coverage_summary" "Day 2b: bucket map"
check "tests/unit/test_banking77_evaluation.py" "TestBanking77BucketEvaluation" "Day 2b: evaluation test"
check "config/prompts.py" "INTENT_CLASSIFIER_SYSTEM" "classify_only() token fix"
check "agents/base_agent.py" "system_override" "classify_only() token fix (base_agent)"
check "config/settings.py" "specialist_provider" "Per-role provider routing"
check "orchestrator/orchestrator.py" "provider: str | None = None" "Per-role provider routing (orchestrator)"