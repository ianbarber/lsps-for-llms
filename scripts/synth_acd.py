#!/usr/bin/env python3
"""A/C/D efficiency eval on the synthetic multi-site type-signal tasks, through the
SAME non-blocking agent harness (MockEnv + real pyrefly + line edits).

The mechanism under test: a multi-site type-error cascade. `pytest` reveals broken
sites one at a time (A must grind serially: fix->retest->next crash); `pyrefly`
reveals all sites at once (C batches at the yield, D splices live). Headline = resolve
rate + efficiency (input/output tokens, wall-clock, #test round-trips) per condition.

Usage: synth_acd.py [out.json] [--conds A,C,D] [--names n1,n2] [--seeds K]
                    [--temp T] [--adapter DIR] [--model ID] [--max-new T]
"""
import os, sys, json, time, argparse
os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scaffold.stream_agent import StreamAgent
from scaffold.mock_env import MockEnv
from scripts.synth_tasks import TASKS

ap = argparse.ArgumentParser()
ap.add_argument("out", nargs="?", default="runs/agent/synth_acd.json")
ap.add_argument("--conds", default="A,C,D")
ap.add_argument("--names", default=None, help="comma subset of task names")
ap.add_argument("--seeds", type=int, default=1, help="sampled rollouts per (task,cond) when temp>0")
ap.add_argument("--seed-start", type=int, default=0, help="first seed index (offset for fresh seeds)")
ap.add_argument("--temp", type=float, default=0.0, help="0 = greedy (deterministic, seeds ignored)")
ap.add_argument("--adapter", default=None)
ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
ap.add_argument("--max-new", type=int, default=1400)
ap.add_argument("--latency", type=int, default=8)
ap.add_argument("--debounce", type=int, default=0, help="D: settle tokens before re-querying (0=immediate)")
ap.add_argument("--pause-align", action="store_true", help="D: deliver at a newline/pause")
ap.add_argument("--announce-lsp", action="store_true", help="D: tell the model LSP feedback is inline")
ap.add_argument("--c-eager", action="store_true", help="C: post-edit hook (deliver diag immediately) vs batched at yield")
ap.add_argument("--syntax-gate", action="store_true", help="D: only deliver live diag when the file parses (suppress self-inflicted syntax squiggles)")
A = ap.parse_args()

tasks = TASKS if not A.names else [t for t in TASKS if t["name"] in set(A.names.split(","))]
conds = A.conds.split(",")
n_seeds = 1 if A.temp == 0 else A.seeds

print(f"[load] {A.model}{' + '+A.adapter if A.adapter else ''}  temp={A.temp} seeds={n_seeds}", flush=True)
tok = AutoTokenizer.from_pretrained(A.model)
model = AutoModelForCausalLM.from_pretrained(A.model, dtype=torch.bfloat16, device_map="auto")
if A.adapter:
    from peft import PeftModel; model = PeftModel.from_pretrained(model, A.adapter)
model = model.eval()

def build_prompt(task):
    code = task["code"]
    numbered = "\n".join(f"{i+1:>3}| {ln}" for i, ln in enumerate(code.splitlines()))
    return (f"Fix the bug(s) in this Python module so the test below passes.\n\n"
            f"Module `sol.py`:\n{numbered}\n\n"
            f"The test that must pass (do NOT edit the test; it is the spec):\n"
            f"```python\n{task['test']}\n```\n"
            f"Make line-range edits, then run <test/>.")

agg = {c: {"rows": []} for c in conds}
os.makedirs(os.path.dirname(A.out), exist_ok=True)
def checkpoint():
    json.dump({"model": A.model, "temp": A.temp, "config": vars(A),
               "rows": {c: agg[c]["rows"] for c in conds}}, open(A.out + ".partial", "w"))
for task in tasks:
    for c in conds:
        for seed in range(A.seed_start, A.seed_start + n_seeds):
            env = MockEnv(task["code"], task["test"], task["entry"])
            agent = StreamAgent(model, tok, env, condition=c, latency_tokens=A.latency,
                                max_new_tokens=A.max_new, edit_mode="line",
                                temperature=A.temp, seed=seed,
                                debounce=A.debounce, pause_align=A.pause_align,
                                announce_lsp=A.announce_lsp, c_eager=A.c_eager,
                                syntax_gate=A.syntax_gate)
            t0 = time.time()
            r = agent.run(build_prompt(task), "sol.py")
            dt = time.time() - t0
            m = r["metrics"]
            row = {"task": task["name"], "cond": c, "seed": seed, "resolved": bool(r["resolved"]),
                   "bailed": r.get("bailed"), "in_tokens": r["in_tokens"], "out_tokens": r["out_tokens"],
                   "sec": round(dt, 1), "rework_ratio": m.get("rework_ratio"), "n_edits": m.get("n_edits"),
                   "n_tests": r["n_tests"], "turns": r["turns"],
                   "stream_tail": r["stream"][-3000:], "events": r["events"]}
            agg[c]["rows"].append(row)
            env.close()
            print(f"  [{task['name']:26}] {c} s{seed}: resolved={row['resolved']} "
                  f"in={row['in_tokens']} out={row['out_tokens']} tests={row['n_tests']} "
                  f"edits={row['n_edits']} rework={row['rework_ratio']} ({row['sec']}s)", flush=True)
    checkpoint()   # incremental: partial results survive an interrupted long run

def mean(xs): return round(sum(xs)/len(xs), 1) if xs else 0.0
print("\n=== aggregate (resolve + efficiency) ===", flush=True)
summary = {}
for c in conds:
    rows = agg[c]["rows"]; res = [r for r in rows if r["resolved"]]
    summary[c] = {"resolve_rate": round(len(res)/len(rows), 3) if rows else 0, "n": len(rows),
                  "mean_in": mean([r["in_tokens"] for r in rows]),
                  "mean_out": mean([r["out_tokens"] for r in rows]),
                  "mean_tests": mean([r["n_tests"] for r in rows]),
                  "mean_sec": mean([r["sec"] for r in rows]),
                  # efficiency among RESOLVED only (matched correctness):
                  "resolved_mean_out": mean([r["out_tokens"] for r in res]),
                  "resolved_mean_tests": mean([r["n_tests"] for r in res])}
    print(f"  {c}: resolve={summary[c]['resolve_rate']} ({len(res)}/{len(rows)})  "
          f"out={summary[c]['mean_out']} tests={summary[c]['mean_tests']}  | "
          f"resolved-only out={summary[c]['resolved_mean_out']} tests={summary[c]['resolved_mean_tests']}", flush=True)

os.makedirs(os.path.dirname(A.out), exist_ok=True)
json.dump({"model": A.model, "adapter": A.adapter, "temp": A.temp, "summary": summary,
           "rows": {c: agg[c]["rows"] for c in conds}}, open(A.out, "w"), indent=2)
print(f"-> {A.out}", flush=True)
