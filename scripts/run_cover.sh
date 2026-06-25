#!/usr/bin/env bash
# E1 v2: the coverage suite with NON-DEFN-CHAINABLE insufficiency (read is forced when the defn is
# insufficient). Run base + cost-trained Qwen3.6-27B; the question is whether the trained model
# READS exactly when needed (J_read >> 0), generalizing to the held-out F2 (attribute-injection)
# mechanism — or whether, with the cheap defn-chain escape removed, it stubbornly defn's and fails.
set -u
cd /home/ianbarber/Projects/Streams
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/mnt/nas/hf-cache
PY=.venv-streams.system/bin/python
M="Qwen/Qwen3.6-27B"
C="--suite cover2 --model $M --gpu-only --conds A --lsp-tools --temp 0.7 --max-reads 4 --max-turns 12 --seeds 3"

pkill -9 -x pyrefly 2>/dev/null || true
echo "[verify-start]"; $PY scripts/synth_tasks_cover2.py 2>&1 | tail -3; echo "[verify-done]"
pkill -9 -x pyrefly 2>/dev/null || true

echo "[cover2-base-start]"
$PY scripts/synth_mf.py runs/agent/cover2_base.json $C \
  && echo "[cover2-base-done]" || { echo "[cover2-base-FAIL]"; exit 1; }

echo "[cover2-sft-start]"
$PY scripts/synth_mf.py runs/agent/cover2_sft.json $C --adapter runs/sft/effic_lora_relabel2_27b \
  && echo "[cover2-sft-done]" || { echo "[cover2-sft-FAIL]"; exit 1; }

echo "[COVER2-DONE]"
$PY scripts/analysis/coverage_j.py base=runs/agent/cover2_base.json trained=runs/agent/cover2_sft.json
