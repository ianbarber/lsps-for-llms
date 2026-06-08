#!/usr/bin/env python3
"""Condition runner for the MULTI-FILE suite (non-redundant-channel setting).

The prompt shows ONLY the target file (numbered) + the names of the other
workspace files (readable via <read path=.../>) + the behavioural test (the spec).
Type definitions the target misuses live in the unshown files, so the checker's
diagnostics carry information the model does not have in context.

PAPER CONDITION -> INVOCATION (same flags as synth_acd.py):
  A        : --conds A
  C-eager  : --conds C --c-eager
  D-gate   : --conds D --debounce 24 --pause-align --syntax-gate
  +rich    : add --rich-signal   (appends remote definitions — the key arm here)

Usage: synth_mf.py [out.json] [--conds ...] [--seeds K] [--seed-start S] [...]
"""
import os, sys, json, time, argparse
os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scaffold.stream_agent import StreamAgent
from scaffold.mock_env import MultiFileEnv
from scripts.synth_tasks_mf import TASKS_MF

ap = argparse.ArgumentParser()
ap.add_argument("out", nargs="?", default="runs/agent/mf_run.json")
ap.add_argument("--conds", default="A,C,D")
ap.add_argument("--names", default=None)
ap.add_argument("--seeds", type=int, default=1)
ap.add_argument("--seed-start", type=int, default=0)
ap.add_argument("--temp", type=float, default=0.7)
ap.add_argument("--adapter", default=None)
ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
ap.add_argument("--max-new", type=int, default=2200)
ap.add_argument("--latency", type=int, default=8)
ap.add_argument("--debounce", type=int, default=0)
ap.add_argument("--pause-align", action="store_true")
ap.add_argument("--announce-lsp", action="store_true")
ap.add_argument("--c-eager", action="store_true")
ap.add_argument("--syntax-gate", action="store_true")
ap.add_argument("--rich-signal", action="store_true")
ap.add_argument("--preread", action="store_true",
                help="put ALL workspace files in the prompt (context-saturation ablation: "
                     "the diagnostic channel becomes informationally redundant)")
ap.add_argument("--clean-delivery", action="store_true",
                help="D: deliver live diagnostic as a clean user turn + file view (isolates format from timing)")
ap.add_argument("--diag-filter", default=None, choices=[None, "type", "syntax"],
                help="deliver only cross-file TYPE errors, or only self-inflicted SYNTAX/scope errors")
A = ap.parse_args()

tasks = TASKS_MF if not A.names else [t for t in TASKS_MF if t["name"] in set(A.names.split(","))]
conds = A.conds.split(",")
n_seeds = 1 if A.temp == 0 else A.seeds

print(f"[load] {A.model}{' + '+A.adapter if A.adapter else ''}  temp={A.temp} "
      f"seeds={A.seed_start}..{A.seed_start+n_seeds-1}", flush=True)
tok = AutoTokenizer.from_pretrained(A.model)
model = AutoModelForCausalLM.from_pretrained(A.model, dtype=torch.bfloat16, device_map="auto")
if A.adapter:
    from peft import PeftModel; model = PeftModel.from_pretrained(model, A.adapter)
model = model.eval()

def build_prompt(task):
    src = task["files"][task["target"]]
    numbered = "\n".join(f"{i+1:>3}| {ln}" for i, ln in enumerate(src.splitlines()))
    others = [f for f in sorted(task["files"]) if f != task["target"]]
    if A.preread:
        # context-saturation: show every other file's full contents up front, so all
        # type information the checker reasons over is already in the model's context.
        ctx = "\n".join(f"`{f}`:\n```python\n{task['files'][f]}```" for f in others)
        whereabouts = (f"The rest of the workspace (read-only, shown in full):\n{ctx}\n\n")
    else:
        whereabouts = (f"The workspace also contains: {', '.join(others)} — you have NOT seen "
                       f"their contents; use <read path=\"...\"/> if you need them.\n\n")
    return (f"Fix the bug(s) in `{task['target']}` so the test below passes.\n\n"
            f"`{task['target']}`:\n{numbered}\n\n"
            f"{whereabouts}"
            f"The test that must pass (do NOT edit it; it is the spec):\n"
            f"```python\n{task['test']}\n```\n"
            f"Make line-range edits to `{task['target']}`, then run <test/>.")

agg = {c: {"rows": []} for c in conds}
os.makedirs(os.path.dirname(A.out), exist_ok=True)
def checkpoint():
    json.dump({"model": A.model, "config": vars(A),
               "rows": {c: agg[c]["rows"] for c in conds}}, open(A.out + ".partial", "w"))

for task in tasks:
    for c in conds:
        for seed in range(A.seed_start, A.seed_start + n_seeds):
            env = MultiFileEnv(task["files"], task["target"], task["test"])
            agent = StreamAgent(model, tok, env, condition=c, latency_tokens=A.latency,
                                max_new_tokens=A.max_new, edit_mode="line",
                                temperature=A.temp, seed=seed,
                                debounce=A.debounce, pause_align=A.pause_align,
                                announce_lsp=A.announce_lsp, c_eager=A.c_eager,
                                syntax_gate=A.syntax_gate, rich_signal=A.rich_signal,
                                clean_delivery=A.clean_delivery, diag_filter=A.diag_filter)
            t0 = time.time()
            r = agent.run(build_prompt(task), task["target"])
            dt = time.time() - t0
            m = r["metrics"]
            row = {"task": task["name"], "group": task["group"], "cond": c, "seed": seed,
                   "resolved": bool(r["resolved"]), "bailed": r.get("bailed"),
                   "in_tokens": r["in_tokens"], "out_tokens": r["out_tokens"],
                   "sec": round(dt, 1), "rework_ratio": m.get("rework_ratio"),
                   "n_edits": m.get("n_edits"), "n_tests": r["n_tests"],
                   "n_reads": r["n_reads"], "turns": r["turns"],
                   "stream_tail": r["stream"][-3000:], "events": r["events"]}
            agg[c]["rows"].append(row)
            env.close()
            print(f"  [{task['name']:22}] {c} s{seed}: resolved={row['resolved']} "
                  f"reads={row['n_reads']} tests={row['n_tests']} edits={row['n_edits']} "
                  f"out={row['out_tokens']} ({row['sec']}s)", flush=True)
    checkpoint()

print("\n=== aggregate ===", flush=True)
summary = {}
for c in conds:
    rs = agg[c]["rows"]; res = [r for r in rs if r["resolved"]]
    bygrp = {}
    for g in ("plain", "rich", "control"):
        sub = [r for r in rs if r["group"] == g]
        bygrp[g] = f"{sum(r['resolved'] for r in sub)}/{len(sub)}"
    summary[c] = {"resolve_rate": round(len(res)/len(rs), 3) if rs else 0, "n": len(rs),
                  "by_group": bygrp,
                  "mean_reads": round(sum(r['n_reads'] for r in rs)/max(len(rs),1), 2)}
    print(f"  {c}: resolve={summary[c]['resolve_rate']} ({len(res)}/{len(rs)})  "
          f"by_group={bygrp}  mean_reads={summary[c]['mean_reads']}", flush=True)

json.dump({"model": A.model, "adapter": A.adapter, "config": vars(A), "summary": summary,
           "rows": {c: agg[c]["rows"] for c in conds}}, open(A.out, "w"), indent=2)
print(f"-> {A.out}", flush=True)
