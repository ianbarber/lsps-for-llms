#!/usr/bin/env bash
# C36: conditional vs blanket read suppression, on the read-required `readreq` split.
#
# Order of work (each stage refuses to proceed past a failed gate):
#   1. hash-gate the frozen protocol sources
#   2. validate 6 pilot + 12 main instances, both flavours (36 builds)
#   3. untrained PILOT FLOOR on arm 1 + arm 2 (12 rollouts); exit 2 on failure
#   4. the 2x3 main grid, untrained then trained (72 rollouts)
#   5. analysis by pre-registered category
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
# the project venv unless PYTHON says otherwise (pyrefly, peft and transformers live there)
[ -z "${PYTHON:-}" ] && [ -x "$ROOT/.venv-streams/bin/python" ] && PY="$ROOT/.venv-streams/bin/python"

MODEL="${MODEL:-Qwen/Qwen3.6-27B}"
# the base revision the adapter was trained on and that C33/C35 pinned; never resolve `main`
REVISION="${REVISION:-6a9e13bd6fc8f0983b9b99948120bc37f49c13e9}"
REVISION_ARG=(--revision "$REVISION")
ADAPTER="${ADAPTER:-runs/sft/substitution_lora_27b}"
RUN_ID="${RUN_ID:?set RUN_ID to a run tag}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "run IDs may contain only letters, digits, dot, underscore, and hyphen" >&2
  exit 64
fi
STAGE="${STAGE:-all}"          # all | validate | pilot | grid | analyse
OUT_PREFIX="runs/readreq/navigation_v2_${RUN_ID}"
PILOT_VALIDATION="runs/protocol/navigation_v2_readreq_pilot_validation.json"
MAIN_VALIDATION="runs/protocol/navigation_v2_readreq_validation.json"
mkdir -p runs/readreq runs/protocol

# ---------------------------------------------------------------- 1. hash gate
"$PY" - <<'PY'
import hashlib, pathlib, sys
root = pathlib.Path.cwd()
frozen = {
    "scripts/experiments/navigation_tasks.py":
        "f860fd07fdeb2d4f78d89a047c6804d79cc3babd60fab7e0a06e839679692d97",
    "scaffold/stream_agent.py":
        "0267afa17a22c5f0eea77bce82927b5d25890dea0500312163d2a1e2e1f40b79",
}
for rel, expected in frozen.items():
    actual = hashlib.sha256((root / rel).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"frozen protocol hash mismatch: {rel}\n  want {expected}\n  got  {actual}")
print("frozen protocol hashes verified")
PY

# ---------------------------------------------------------------- 2. validate
if [ "$STAGE" = all ] || [ "$STAGE" = validate ]; then
  [ -e "$PILOT_VALIDATION" ] || "$PY" scripts/experiments/navigation_readreq_tasks.py validate \
      --split readreq_pilot --out "$PILOT_VALIDATION"
  [ -e "$MAIN_VALIDATION" ] || "$PY" scripts/experiments/navigation_readreq_tasks.py validate \
      --split readreq --out "$MAIN_VALIDATION"
fi
for artifact in "$PILOT_VALIDATION" "$MAIN_VALIDATION"; do
  "$PY" - "$artifact" <<'PY'
import json, sys
data = json.loads(open(sys.argv[1]).read())
if not data.get("passed") or not all(row["passed"] for row in data["rows"]):
    raise SystemExit(f"{sys.argv[1]}: instances did not clear every gate")
print(f"{sys.argv[1]}: {len(data['rows'])}/{len(data['rows'])} instances validated")
PY
done

# ---------------------------------------------------------------- 3. pilot floor
if [ "$STAGE" = all ] || [ "$STAGE" = pilot ]; then
  "$PY" scripts/experiments/run_navigation_readreq.py "${OUT_PREFIX}_pilot_untrained.json" \
    --model "$MODEL" "${REVISION_ARG[@]}" --split readreq_pilot \
    --arms push_insufficient,push_sufficient --validation "$PILOT_VALIDATION" --gpu-only
fi

# ---------------------------------------------------------------- 4. main grid
if [ "$STAGE" = all ] || [ "$STAGE" = grid ]; then
  "$PY" scripts/experiments/run_navigation_readreq.py "${OUT_PREFIX}_main_untrained.json" \
    --model "$MODEL" "${REVISION_ARG[@]}" --split readreq \
    --arms push_insufficient,push_sufficient,push_chained --validation "$MAIN_VALIDATION" --gpu-only
  "$PY" scripts/experiments/run_navigation_readreq.py "${OUT_PREFIX}_main_trained.json" \
    --model "$MODEL" "${REVISION_ARG[@]}" --split readreq --adapter "$ADAPTER" \
    --arms push_insufficient,push_sufficient,push_chained --validation "$MAIN_VALIDATION" --gpu-only
fi

# ---------------------------------------------------------------- 5. analysis
if [ "$STAGE" = all ] || [ "$STAGE" = analyse ]; then
  "$PY" scripts/analysis/analyze_navigation_readreq.py \
    "${OUT_PREFIX}_main_untrained.json" "${OUT_PREFIX}_main_trained.json"
fi
