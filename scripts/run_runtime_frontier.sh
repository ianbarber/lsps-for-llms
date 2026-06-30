#!/usr/bin/env bash
# RUNTIME-FEEDBACK test: does EXECUTION feedback help a strong agent (unlike the language
# server's static information, which §3 shows is redundant)? Three arms on the `runtime`
# suite, scored HELD-OUT in every arm (`resolved` = held-out, `visible_pass` = the test the
# agent could run). The visible test FAILS for the wrong fix, so RUNNING is the detector.
#
#   R0  --no-test            no run_tests tool: reason from source + shown spec, then done().
#   R1  (default)            run_tests available; the agent elects to run it.
#   R2  --auto-feedback      env volunteers the visible-test result after every edit.
#
# Always --no-hint: the visible test is informative here (not a partial spec), so the gapd2
# "partial spec" note is off and auto-stop-on-visible-pass is the natural R1/R2 behaviour.
# Every wrong fix is type-clean, so check_types is useless on this suite (execution-only).
# Needs the OpenRouter key (repo-root .orkey or OPENROUTER_API_KEY).
set -u
cd /home/ianbarber/Projects/Streams
PY=.venv-streams.system/bin/python
SEEDS="${SEEDS:-3}"
COMMON="--suite runtime --seeds $SEEDS --max-reads 3 --max-turns 10 --temperature 0.7 --no-hint"

run() {  # tag model arm extra budget
  local tag="$1" model="$2" arm="$3" extra="$4" bud="$5"
  local out="runs/agent/rt_${tag}_${arm}.json"
  if [ -f "$out" ]; then echo "[rt-$tag-$arm-skip]"; return; fi
  echo "[rt-$tag-$arm-start $(date +%T)]"
  $PY scripts/api_agent.py "$out" $COMMON --model "$model" $extra --budget-usd "$bud"
  echo "[rt-$tag-$arm-done exit $? $(date +%T)]"
}

# spec = MODEL:TAG:BUDGET_PER_ARM
for spec in "deepseek/deepseek-chat-v3.1:deepseek:3" "anthropic/claude-sonnet-4.5:sonnet45:8"; do
  IFS=: read -r MODEL TAG BUD <<< "$spec"
  run "$TAG" "$MODEL" r0_notest "--no-test"        "$BUD"
  run "$TAG" "$MODEL" r1_run    ""                 "$BUD"
  run "$TAG" "$MODEL" r2_auto   "--auto-feedback"  "$BUD"
done
echo "[ALL-RUNTIME-DONE $(date +%T)]"
