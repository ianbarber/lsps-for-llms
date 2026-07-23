#!/usr/bin/env bash
# SUBSTITUTION training (follow-up to the reread-after-span null, C31).
#
# Election was trainable on a 7B (C2, run_relabel2.sh): the model rolls out, chooses <read>, a rule
# oracle redirects it to <defn>, and we DROP the read-attempt+redirect prefix so the model's own
# <defn> is the first trained action from a CLEAN prompt. This script asks the same question for
# SUBSTITUTION: when a definition span is delivered automatically, can training stop the model from
# reopening the defining file (prompting could not: 0/12 for the 27B, 2/36 across three models)?
#
#   harvest (12 FRESH substrain instances, disjoint seeds AND templates) -> LoRA SFT -> retest on
#   the 12 HELD-OUT apparatus instances (auto_neutral arm), against the untrained baseline in
#   runs/pilot/navigation_v2_reread_qwen36_27b_apparatus.json.
#
# The reserved 12 confirmation instances (41xxx) are never built or touched.
set -u
PY=${PYTHON:-.venv-streams/bin/python}
M=${MODEL:-Qwen/Qwen3.6-27B}
REV=${REVISION:-6a9e13bd6fc8f0983b9b99948120bc37f49c13e9}
ADAPTER=runs/sft/substitution_lora_27b
HARVEST=runs/agent/substitution_harvest_qwen36_27b.json
RETEST=runs/pilot/navigation_v2_reread_qwen36_27b_apparatus_trained.json
VALID=runs/protocol/navigation_v2_substrain_validation.json

if [ ! -f "$VALID" ]; then
  echo "[validate-start]"
  $PY scripts/experiments/run_substitution_train.py validate --out "$VALID" \
    && echo "[validate-done]" || { echo "[validate-FAIL]"; exit 1; }
else echo "[validate-skip]"; fi

if [ ! -f "$HARVEST" ]; then
  echo "[harvest-start]"
  $PY scripts/experiments/run_substitution_train.py harvest "$HARVEST" \
    --model "$M" --revision "$REV" --temp 0.7 --seeds 4 --gpu-only \
    --max-new 1000 --max-turns 12 --max-reads 12 \
    && echo "[harvest-done]" || { echo "[harvest-FAIL]"; exit 1; }
else echo "[harvest-skip]"; fi

if [ ! -d "$ADAPTER" ]; then
  echo "[sft-start]"
  $PY scripts/sft_lora.py --harvest "$HARVEST" --model "$M" --revision "$REV" \
    --filter sft_keep --out "$ADAPTER" --epochs 3 --lr 1e-4 --accum 4 --max-len 4096 \
    && echo "[sft-done]" || { echo "[sft-FAIL]"; exit 1; }
else echo "[sft-skip]"; fi

if [ ! -f "$RETEST" ]; then
  echo "[retest-start]"
  $PY scripts/experiments/run_navigation_reread.py "$RETEST" \
    --model "$M" --revision "$REV" --adapter "$ADAPTER" \
    --split apparatus --arms auto_neutral --temperature 0 --seed 0 \
    --max-new 1000 --max-turns 12 --max-reads 12 --gpu-only \
    --tmp-root /tmp/streams_nav_reread_trained \
    && echo "[retest-done]" || { echo "[retest-FAIL]"; exit 1; }
else echo "[retest-skip]"; fi

echo "[ALL-SUBSTITUTION-DONE]"
