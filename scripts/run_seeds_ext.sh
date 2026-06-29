#!/usr/bin/env bash
# Extend the 27B + frontier ablation arms from 2 seeds (0-1) to 4 (add seeds 2-3), so the
# headline token ratios are not 2-seed. Writes *_s23.json; merge into the base files after.
# Strictly sequential (the 27B and gapd-with-check use pyrefly; the frontier effic arms are
# pyrefly-free) so at most one pyrefly process runs at a time. Resumable (skip existing).
set -u
cd /home/ianbarber/Projects/Streams
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/mnt/nas/hf-cache
PY=.venv-streams.system/bin/python
SS="--seed-start 2 --seeds 2"
pkill -9 -f "[p]yrefly" 2>/dev/null

# --- 27B local effic_real2 tool ablation (pyrefly) ---
C27="--suite effic_real2 --model Qwen/Qwen3.6-27B --gpu-only --conds A --temp 0.7 --max-reads 4 --max-turns 14 --max-new 3000 $SS"
[ -f runs/agent/er2_27b_base_s23.json ]     || { echo "[27b-wd $(date +%T)]"; $PY scripts/synth_mf.py runs/agent/er2_27b_base_s23.json $C27 --lsp-tools; }
[ -f runs/agent/er2_27b_readonly_s23.json ] || { echo "[27b-ro $(date +%T)]"; $PY scripts/synth_mf.py runs/agent/er2_27b_readonly_s23.json $C27 --no-defn; }

# --- frontier effic_real2 (pyrefly-free, API) ---
FE="--suite effic_real2 --max-reads 6 --max-turns 14 --temperature 0.7 $SS"
for spec in "deepseek/deepseek-chat-v3.1:deepseek:1:3" "anthropic/claude-sonnet-4.5:sonnet45:3:12"; do
  IFS=: read -r MODEL TAG WDB ROB <<< "$spec"
  echo "[fr-$TAG $(date +%T)]"
  [ -f runs/agent/fr_${TAG}_withdefn_s23.json ] || $PY scripts/api_agent.py runs/agent/fr_${TAG}_withdefn_s23.json $FE --model "$MODEL" --budget-usd "$WDB"
  [ -f runs/agent/fr_${TAG}_readonly_s23.json ] || $PY scripts/api_agent.py runs/agent/fr_${TAG}_readonly_s23.json $FE --model "$MODEL" --no-defn --budget-usd "$ROB"
done

# --- Gap D frontier (with-check uses pyrefly, API) ---
GE="--suite gapd --max-reads 6 --max-turns 14 --temperature 0.7 $SS"
for spec in "deepseek/deepseek-chat-v3.1:deepseek:2" "anthropic/claude-sonnet-4.5:sonnet45:6"; do
  IFS=: read -r MODEL TAG BUD <<< "$spec"
  echo "[gd-$TAG $(date +%T)]"
  [ -f runs/agent/gd_${TAG}_nocheck_s23.json ]   || $PY scripts/api_agent.py runs/agent/gd_${TAG}_nocheck_s23.json $GE --model "$MODEL" --budget-usd "$BUD"
  [ -f runs/agent/gd_${TAG}_withcheck_s23.json ] || $PY scripts/api_agent.py runs/agent/gd_${TAG}_withcheck_s23.json $GE --model "$MODEL" --with-check --budget-usd "$BUD"
done
echo "[SEEDS-EXT-DONE $(date +%T)]"
