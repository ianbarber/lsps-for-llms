#!/usr/bin/env python3
"""Realistic A/C/D run: drive the StreamAgent over real buggy single-file tasks with
REAL pyrefly LSP, under each feedback condition, and aggregate the in-flight metrics
(headline = rework_ratio). This is the first 'go realistic' datapoint — the toy
C-vs-D efficiency result (R0c) carried over to a genuine edit-test-fix agent loop.

Tasks are chosen so a *type/name-level* error fires pyrefly (so delivery TIMING can
matter): a tempting wrong edit leaves an undefined name / bad attribute / type error
that the static analyzer flags immediately, while the test also fails. Live (D)
surfaces that squiggle mid-stream; sync (C) at the next turn boundary; A never.

Usage: agent_acd.py [out.json] [--adapter DIR] [--model ID] [--conds A,C,D]
                    [--task N] [--max-new T] [--latency K]
"""
import os, sys, json, argparse, time
os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scaffold.stream_agent import StreamAgent
from scaffold.mock_env import MockEnv

ap = argparse.ArgumentParser()
ap.add_argument("out", nargs="?", default="runs/agent/acd_base.json")
ap.add_argument("--adapter", default=None)
ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
ap.add_argument("--conds", default="A,C,D")
ap.add_argument("--task", type=int, default=None, help="run only task index N (smoke test)")
ap.add_argument("--max-new", type=int, default=2000)
ap.add_argument("--latency", type=int, default=8)
ap.add_argument("--edit-mode", default="rewrite", choices=["search", "rewrite"])
A = ap.parse_args()

# Each task: a buggy single file with a TYPE/NAME-level defect pyrefly catches, plus a
# behavioral test. The fix is small; a plausible wrong edit keeps/introduces a squiggle.
TASKS = [
    dict(
        name="totals_undefined",
        ep="totals",
        prompt=(
            "The function `totals` should return the running sums of a list of ints "
            "(e.g. totals([1,2,3]) == [1,3,6]). It is buggy. Fix it.\n\n"
            "```python\n"
            "def totals(xs: list[int]) -> list[int]:\n"
            "    out = []\n"
            "    for x in xs:\n"
            "        acc = acc + x\n"          # acc undefined on first iter -> pyrefly + NameError
            "        out.append(acc)\n"
            "    return out\n"
            "```"
        ),
        buggy=("def totals(xs: list[int]) -> list[int]:\n"
               "    out = []\n"
               "    for x in xs:\n"
               "        acc = acc + x\n"
               "        out.append(acc)\n"
               "    return out\n"),
        test="assert totals([1,2,3])==[1,3,6] and totals([])==[] and totals([5])==[5]",
    ),
    dict(
        name="label_type",
        ep="label",
        prompt=(
            "`label` should return a string like 'item-3'. It is buggy (it returns the "
            "wrong type and crashes). Fix it so label(3)=='item-3'.\n\n"
            "```python\n"
            "def label(n: int) -> str:\n"
            "    return 'item-' + n\n"          # int+str TypeError, pyrefly flags
            "```"
        ),
        buggy="def label(n: int) -> str:\n    return 'item-' + n\n",
        test="assert label(3)=='item-3' and label(0)=='item-0'",
    ),
    dict(
        name="top_attr",
        ep="top",
        prompt=(
            "`top` should return the largest value in a list. It is buggy. Fix it so "
            "top([3,1,2])==3.\n\n"
            "```python\n"
            "def top(xs: list[int]) -> int:\n"
            "    return xs.max()\n"            # list has no .max() -> pyrefly + AttributeError
            "```"
        ),
        buggy="def top(xs: list[int]) -> int:\n    return xs.max()\n",
        test="assert top([3,1,2])==3 and top([5])==5",
    ),
    dict(
        name="avg_div",
        ep="avg",
        prompt=(
            "`avg` should return the average of a list as a float. It is buggy. Fix it "
            "so avg([2,4])==3.0.\n\n"
            "```python\n"
            "def avg(xs: list[int]) -> float:\n"
            "    return sum(xs) / len\n"        # len (the builtin) used as int -> pyrefly + TypeError
            "```"
        ),
        buggy="def avg(xs: list[int]) -> float:\n    return sum(xs) / len\n",
        test="assert avg([2,4])==3.0 and avg([10])==10.0",
    ),
]

print(f"[load] {A.model}{' + '+A.adapter if A.adapter else ''}", flush=True)
tok = AutoTokenizer.from_pretrained(A.model)
model = AutoModelForCausalLM.from_pretrained(A.model, dtype=torch.bfloat16, device_map="auto")
if A.adapter:
    from peft import PeftModel; model = PeftModel.from_pretrained(model, A.adapter)
model = model.eval()

conds = A.conds.split(",")
tasks = [TASKS[A.task]] if A.task is not None else TASKS
agg = {c: {"resolved": 0, "rework": [], "n_edits": [], "tokens": [], "cycles": [], "rows": []} for c in conds}

for ti, task in enumerate(tasks):
    for c in conds:
        env = MockEnv(task["buggy"], task["test"], task["ep"])
        agent = StreamAgent(model, tok, env, condition=c, latency_tokens=A.latency,
                            max_new_tokens=A.max_new, edit_mode=A.edit_mode)
        t0 = time.time()
        r = agent.run(task["prompt"], "sol.py")
        dt = time.time() - t0
        m = r["metrics"]
        row = {"task": task["name"], "cond": c, "resolved": bool(r["resolved"]),
               "rework_ratio": m["rework_ratio"], "n_edits": m["n_edits"],
               "edit_error_cycles": m["edit_error_cycles"], "n_tokens": r["n_tokens"],
               "n_events": len(r["events"]), "sec": round(dt, 1)}
        agg[c]["resolved"] += row["resolved"]
        agg[c]["rework"].append(m["rework_ratio"]); agg[c]["n_edits"].append(m["n_edits"])
        agg[c]["tokens"].append(r["n_tokens"]); agg[c]["cycles"].append(m["edit_error_cycles"])
        agg[c]["rows"].append({**row, "events": r["events"], "stream_tail": r["stream"][-2000:]})
        print(f"  [{task['name']:>16}] {c}: resolved={row['resolved']} "
              f"rework={m['rework_ratio']:.3f} edits={m['n_edits']} "
              f"cycles={m['edit_error_cycles']} tok={r['n_tokens']} ({dt:.0f}s)", flush=True)
        env.close()

def mean(xs): return round(sum(xs)/len(xs), 3) if xs else 0.0
print("\n=== aggregate ===", flush=True)
for c in conds:
    a = agg[c]; n = len(a["rows"])
    print(f"  {c}: resolved={a['resolved']}/{n}  mean_rework={mean(a['rework'])}  "
          f"mean_edits={mean(a['n_edits'])}  mean_cycles={mean(a['cycles'])}  "
          f"mean_tok={mean(a['tokens'])}", flush=True)

os.makedirs(os.path.dirname(A.out), exist_ok=True)
json.dump({"model": A.model, "adapter": A.adapter, "agg": {
    c: {"resolved": agg[c]["resolved"], "n": len(agg[c]["rows"]),
        "mean_rework": mean(agg[c]["rework"]), "mean_edits": mean(agg[c]["n_edits"]),
        "mean_cycles": mean(agg[c]["cycles"]), "mean_tokens": mean(agg[c]["tokens"]),
        "rows": agg[c]["rows"]} for c in conds}}, open(A.out, "w"), indent=2)
print(f"-> {A.out}", flush=True)
