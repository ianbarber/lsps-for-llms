#!/usr/bin/env python3
"""Self-distillation harvester (R2). Run the agent in the DEPLOYMENT config (D-tuned by
default) over the synthetic tasks, keep the RESOLVED trajectories, and emit them as
{input_ids, labels} for d_sft.py — labels train ONLY on the model's own action tokens
(reasoning + edits), masking every spliced observation (‹diag›, <test_result>, file
views, turn scaffolding). This is on-policy, rejection-sampled self-distillation matched
to the live-‹diag› deployment (the OLD i_sft_data was forward-‹info›, wrong task).

Usage: harvest_sft.py [out.jsonl] [--conds D] [--seeds K] [--seed-start S] [--temp T]
                      [--debounce N] [--pause-align] [--announce-lsp] [--c-eager]
                      [--names ...] [--min-train-tokens N] [--max-new T]
"""
import os, sys, json, argparse, time
os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scaffold.stream_agent import StreamAgent
from scaffold.mock_env import MockEnv
from scripts.synth_tasks import TASKS

ap = argparse.ArgumentParser()
ap.add_argument("out", nargs="?", default="runs/i_sft_data_v2/data.jsonl")
ap.add_argument("--conds", default="D", help="which conditions to harvest resolved trajs from")
ap.add_argument("--seeds", type=int, default=16)
ap.add_argument("--seed-start", type=int, default=0)
ap.add_argument("--temp", type=float, default=0.7)
ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
ap.add_argument("--max-new", type=int, default=2200)
ap.add_argument("--debounce", type=int, default=24)
ap.add_argument("--pause-align", action="store_true", default=True)
ap.add_argument("--announce-lsp", action="store_true", default=True)
ap.add_argument("--c-eager", action="store_true")
ap.add_argument("--names", default=None)
ap.add_argument("--min-train-tokens", type=int, default=10, help="drop trivially-short trajectories")
A = ap.parse_args()

tasks = TASKS if not A.names else [t for t in TASKS if t["name"] in set(A.names.split(","))]
conds = A.conds.split(",")

print(f"[load] {A.model}  conds={conds} seeds={A.seed_start}..{A.seed_start+A.seeds-1} temp={A.temp}", flush=True)
tok = AutoTokenizer.from_pretrained(A.model)
model = AutoModelForCausalLM.from_pretrained(A.model, dtype=torch.bfloat16, device_map="auto").eval()

def build_prompt(task):
    numbered = "\n".join(f"{i+1:>3}| {ln}" for i, ln in enumerate(task["code"].splitlines()))
    return (f"Fix the bug(s) in this Python module so the test below passes.\n\n"
            f"Module `sol.py`:\n{numbered}\n\n"
            f"The test that must pass (do NOT edit the test; it is the spec):\n"
            f"```python\n{task['test']}\n```\n"
            f"Make line-range edits, then run <test/>.")

os.makedirs(os.path.dirname(A.out), exist_ok=True)
kept, attempted = [], 0
per_task = {}
for task in tasks:
    for c in conds:
        for seed in range(A.seed_start, A.seed_start + A.seeds):
            env = MockEnv(task["code"], task["test"], task["entry"])
            agent = StreamAgent(model, tok, env, condition=c, max_new_tokens=A.max_new,
                                edit_mode="line", temperature=A.temp, seed=seed,
                                debounce=A.debounce, pause_align=A.pause_align,
                                announce_lsp=A.announce_lsp, c_eager=A.c_eager)
            r = agent.run(build_prompt(task), "sol.py")
            attempted += 1
            env.close()
            ok = r["resolved"] and r["n_train_tokens"] >= A.min_train_tokens
            if ok:
                kept.append({"input_ids": r["sft_input_ids"], "labels": r["sft_labels"],
                             "task": task["name"], "cond": c, "seed": seed,
                             "n_train_tokens": r["n_train_tokens"]})
                per_task[task["name"]] = per_task.get(task["name"], 0) + 1
            print(f"  {task['name']:24} {c} s{seed}: resolved={r['resolved']} "
                  f"train_tok={r['n_train_tokens']} {'KEPT' if ok else ''}", flush=True)
        # checkpoint after each task
        with open(A.out, "w") as f:
            for ex in kept:
                f.write(json.dumps(ex) + "\n")

tot_train = sum(ex["n_train_tokens"] for ex in kept)
print(f"\n=== harvested {len(kept)} resolved trajectories from {attempted} attempts "
      f"({len(kept)/max(attempted,1):.0%}) ===", flush=True)
print(f"  total train tokens: {tot_train}  (mean {tot_train//max(len(kept),1)}/traj)")
print(f"  per task: {per_task}")
print(f"  -> {A.out}", flush=True)
