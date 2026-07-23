#!/usr/bin/env bash
# Expanded (12-pair, family-labeled) defect/clean gate cohort across the phase-gradient
# arms: control, one-shot diagnostics at revision, acceptance gate at submission, and
# after-every-edit volunteered feedback (noisy). Local model, no API spend.
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
REVISION="${REVISION:-b7ca741b86de18df552fd2cc952861e04621a4bd}"
DRAFTS="${DRAFTS:-runs/protocol/checker_gate_v3_validation.json}"
OUT="${OUT:-runs/pilot/checker_gate_v3_qwen35_27b_pilot.json}"
ARMS="${ARMS:-control,diagnostics,gate,noisy}"

if [ -e "$OUT" ]; then
  echo "refusing to overwrite checker gate v3 output: $OUT" >&2
  exit 2
fi

"$PY" scripts/experiments/checker_paired.py revise "$DRAFTS" "$OUT" \
  --model "$MODEL" --revision "$REVISION" \
  --arms "$ARMS" \
  --temperature 0 --seed 0 --seeds 1 \
  --max-new 1400 --max-turns 12 --max-reads 4 --gpu-only

"$PY" scripts/analysis/analyze_checker_paired.py --drafts "$DRAFTS" --revisions "$OUT"
