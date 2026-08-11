#!/bin/bash
# Run from your project root (Documents/thesis_final).
# bash-3.2-compatible (macOS default /bin/bash) -- no associative arrays.

files=(
  "orchestrator/orchestrator.py"
  "orchestrator/audit_log.py"
  "orchestrator/planner.py"
  "config/settings.py"
  "data/psychometric_proxy.py"
  "tests/unit/test_banking77_evaluation.py"
  "requirements.txt"
  "utils/market_data_client.py"
  "conftest.py"
  "api/app.py"
  "api/session_registry.py"
  "agents/conversational_agent.py"
  "explainability/explainability_agent.py"
  "scripts/build_regulatory_corpus.py"
  "run_demo.py"
)
markers=(
  "SKIP-EXPLAIN"
  "record_citations"
  "specialists_present"
  "log_citations_to_audit"
  "confidence: str"
  "skipif"
  "pypdf"
  "noqa: F401"
  "pyarrow.dataset"
  "SECURITY"
  "G2, open"
  "behind on payments"
  "Grounded in:"
  "mabs-5-steps"
  "store.mark_delivered"
)

fail=0
i=0
while [ "$i" -lt "${#files[@]}" ]; do
  f="${files[$i]}"
  marker="${markers[$i]}"
  if [ ! -f "$f" ]; then
    echo "MISSING FILE: $f"
    fail=1
  elif ! grep -q "$marker" "$f"; then
    echo "STALE:  $f  (missing marker: '$marker')"
    fail=1
  else
    echo "OK:     $f"
  fi
  i=$((i + 1))
done

echo ""
if [ "$fail" -eq 0 ]; then
  echo "All files in sync."
else
  echo "^^ fix the STALE/MISSING files above before running."
fi
exit "$fail"