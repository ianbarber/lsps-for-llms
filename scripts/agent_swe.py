#!/usr/bin/env python3
"""Real-harness A/C/D efficiency eval on SWE-rebench tasks. For each provisioned
task and each feedback condition, run the non-blocking StreamAgent against a fresh
TaskEnv (clone+venv+install+test_patch) and record the efficiency metrics:
  resolved, input tokens, output tokens, wall-clock, rework_ratio, #edits/#tests.

Conditions: A (no LSP) / C (sync: diagnostics batched at the next yield) /
D (live: diagnostics spliced mid-stream as edits complete, non-blocking).

Usage: agent_swe.py [out.json] [--conds A,C,D] [--limit N] [--ids id,..]
                    [--adapter DIR] [--model ID] [--max-new T] [--latency K]
"""
import os, sys, json, time, argparse
os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scaffold.stream_agent import StreamAgent
from harness.task_env import TaskEnv

ap = argparse.ArgumentParser()
ap.add_argument("out", nargs="?", default="runs/agent/swe_base.json")
ap.add_argument("--conds", default="A,C,D")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--ids", default=None)
ap.add_argument("--adapter", default=None)
ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
ap.add_argument("--max-new", type=int, default=2200)
ap.add_argument("--latency", type=int, default=8)
ap.add_argument("--tasks-file", default="runs/rebench/provisioned.jsonl")
ap.add_argument("--scope", default="region", choices=["region", "full"],
                help="region = present the enclosing buggy function (oracle localization, "
                     "isolates the feedback-delivery variable); full = whole file")
A = ap.parse_args()
import re as _re

def first_hunk_span(patch, src):
    """1-based (start, end) line span on the BASE (buggy) side of the first hunk
    that touches `src`, from the gold patch. Used only for oracle localization."""
    cur = None
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            cur = line.split(" b/")[-1]
        elif line.startswith("@@ ") and cur == src:
            m = _re.search(r"@@ -(\d+),?(\d+)?", line)
            if m:
                s = int(m.group(1)); n = int(m.group(2) or 1)
                return s, s + max(n, 1) - 1
    return None

def enclosing_region(contents, start, end, pad=6):
    """Expand [start,end] to the enclosing top-level def/class (so the model sees a
    coherent unit), else a padded window. 1-based inclusive line numbers."""
    lines = contents.splitlines()
    n = len(lines)
    lo = max(1, start - pad)
    # walk up to a def/class at column 0-ish above the hunk
    for i in range(min(start, n) - 1, -1, -1):
        s = lines[i]
        if _re.match(r"\s{0,4}(def|class|async def)\b", s):
            lo = i + 1; break
        if i <= start - 40:
            break
    hi = min(n, end + pad)
    # extend down until a line at the same/lower indent starts a new top-level block
    base_indent = len(lines[lo - 1]) - len(lines[lo - 1].lstrip()) if lo - 1 < n else 0
    for j in range(end, n):
        s = lines[j]
        if s.strip() and (len(s) - len(s.lstrip())) <= base_indent and \
           _re.match(r"\s*(def|class|async def|@)\b", s) and j + 1 > end:
            hi = j; break
        hi = min(n, j + 1)
        if j > end + 80:
            break
    return lo, hi

tasks = [json.loads(l) for l in open(A.tasks_file)]
if A.ids:
    keep = set(A.ids.split(","))
    tasks = [t for t in tasks if t["instance"]["instance_id"] in keep]
if A.limit:
    tasks = tasks[:A.limit]
conds = A.conds.split(",")
WORKROOT = os.path.abspath("runs/agent/workdirs")

print(f"[load] {A.model}{' + '+A.adapter if A.adapter else ''}", flush=True)
tok = AutoTokenizer.from_pretrained(A.model)
model = AutoModelForCausalLM.from_pretrained(A.model, dtype=torch.bfloat16, device_map="auto")
if A.adapter:
    from peft import PeftModel; model = PeftModel.from_pretrained(model, A.adapter)
model = model.eval()

def build_prompt(problem, src, contents, inst, scope):
    all_lines = contents.splitlines()
    n = len(all_lines)
    region_note = ""
    lo, hi = 1, n
    if scope == "region":
        span = first_hunk_span(inst.get("patch", ""), src)
        if span:
            lo, hi = enclosing_region(contents, span[0], span[1])
            region_note = (f"\n(The bug is within lines {lo}-{hi}, shown below. You may "
                           f"<read path=\"{src}\"/> for the full file if needed.)")
    numbered = "\n".join(f"{i+1:>4}| {all_lines[i]}" for i in range(lo - 1, hi))
    return (f"{problem.strip()}\n\n"
            f"The bug is in `{src}`.{region_note}\n{numbered}\n\n"
            f"Make the minimal line-range edit(s) so the failing tests pass, then run <test/>.")

agg = {c: {"rows": []} for c in conds}
for ti, t in enumerate(tasks):
    inst = t["instance"]; src = t["src_file"]; iid = inst["instance_id"]
    for c in conds:
        env = TaskEnv(inst, workroot=os.path.join(WORKROOT, c))
        try:
            env.reset(install=True)
            contents = env.read_file(src)
            prompt = build_prompt(inst.get("problem_statement", ""), src, contents, inst, A.scope)
            agent = StreamAgent(model, tok, env, condition=c, latency_tokens=A.latency,
                                max_new_tokens=A.max_new, edit_mode="line")
            t0 = time.time()
            r = agent.run(prompt, src)
            dt = time.time() - t0
            m = r["metrics"]
            row = {"id": iid, "cond": c, "resolved": bool(r["resolved"]), "bailed": r.get("bailed"),
                   "in_tokens": r["in_tokens"], "out_tokens": r["out_tokens"], "sec": round(dt, 1),
                   "rework_ratio": m.get("rework_ratio"), "edit_count": m.get("edit_count"),
                   "failed_edits": m.get("failed_edit_count"), "n_tests": r["n_tests"],
                   "n_reads": r["n_reads"], "turns": r["turns"],
                   "stream_tail": r["stream"][-3500:], "events": r["events"]}
        except Exception as e:
            import traceback; traceback.print_exc()
            row = {"id": iid, "cond": c, "error": f"{type(e).__name__}: {e}", "resolved": False}
        finally:
            try: env.close(remove_workdir=True)
            except Exception: pass
        agg[c]["rows"].append(row)
        print(f"  [{iid:42}] {c}: resolved={row.get('resolved')} "
              f"in={row.get('in_tokens')} out={row.get('out_tokens')} "
              f"rework={row.get('rework_ratio')} edits={row.get('edit_count')} "
              f"tests={row.get('n_tests')} ({row.get('sec')}s)", flush=True)

def mean(xs): return round(sum(xs)/len(xs), 1) if xs else 0.0
print("\n=== aggregate (efficiency headline = in/out tokens + sec at matched resolve) ===", flush=True)
summary = {}
for c in conds:
    rows = [r for r in agg[c]["rows"] if "error" not in r]
    res = sum(r["resolved"] for r in rows)
    summary[c] = {
        "resolved": res, "n": len(agg[c]["rows"]),
        "mean_in": mean([r["in_tokens"] for r in rows]),
        "mean_out": mean([r["out_tokens"] for r in rows]),
        "mean_sec": mean([r["sec"] for r in rows]),
        "mean_rework": mean([r["rework_ratio"] for r in rows if r.get("rework_ratio") is not None]),
    }
    print(f"  {c}: resolved={res}/{len(agg[c]['rows'])}  in={summary[c]['mean_in']}  "
          f"out={summary[c]['mean_out']}  sec={summary[c]['mean_sec']}  "
          f"rework={summary[c]['mean_rework']}", flush=True)

os.makedirs(os.path.dirname(A.out), exist_ok=True)
json.dump({"model": A.model, "adapter": A.adapter, "summary": summary,
           "rows": {c: agg[c]["rows"] for c in conds}}, open(A.out, "w"), indent=2)
print(f"-> {A.out}", flush=True)
