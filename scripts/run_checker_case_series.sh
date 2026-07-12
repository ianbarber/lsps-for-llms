#!/usr/bin/env bash
# Zero-cost opportunity-conditioned case series over frozen checker-positive drafts.
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

MODEL="Qwen/Qwen3.6-27B"
REVISION="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
DRAFTS="runs/protocol/checker_opportunity_case_series_v4.json"
OUT="runs/pilot/checker_case_series_qwen36_27b_6a9e13bd_v2_s1.json"

if [ -e "$OUT" ]; then
  echo "refusing to overwrite checker case-series output: $OUT" >&2
  exit 2
fi

"$PY" scripts/experiments/checker_paired.py revise "$DRAFTS" "$OUT" \
  --model "$MODEL" --revision "$REVISION" \
  --names auth_fold_callable,auth_histogram_counter \
  --arms control,diagnostics,gate --checker-positive-only \
  --temperature 0.7 --seed 0 --seeds 1 \
  --max-new 800 --max-turns 6 --max-reads 4 --gpu-only

"$PY" scripts/analysis/analyze_checker_paired.py --drafts "$DRAFTS" --revisions "$OUT"
