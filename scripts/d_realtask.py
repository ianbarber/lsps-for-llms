#!/usr/bin/env python3
"""Single REAL SWE-bench task end-to-end through the continuous-stream agent — the
de-risk before the full pilot: does the agent produce valid edits + resolve on an
actual task (not a toy), and does the live-diagnostic path fire on a hard bug?

Usage: d_realtask.py [instance_id] [source] [condition]
"""
import os, sys, re
os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")
sys.path.insert(0, "/home/ianbarber/Projects/Streams")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from harness.task_env import TaskEnv, load_instance
from scaffold.stream_agent import StreamAgent

IID = sys.argv[1] if len(sys.argv) > 1 else "sympy__sympy-23950"
SRC = sys.argv[2] if len(sys.argv) > 2 else "verified"
COND = sys.argv[3] if len(sys.argv) > 3 else "D"

print(f"[load instance] {IID} ({SRC})", flush=True)
inst = load_instance(IID, source=SRC)
env = TaskEnv(inst)
print("[reset] clone+venv+install (slow) ...", flush=True)
env.reset()

# target file(s) from the gold patch header (oracle localization — fair across A/C/D)
files = re.findall(r'^\+\+\+ b/(.+)$', inst.get("patch", ""), re.M)
files = [f for f in files if not f.startswith(("test", "tests")) and f.endswith(".py")] or files
tgt = files[0] if files else None
file_blob = ""
if tgt:
    try:
        content = env.read_file(tgt)
        numbered = "\n".join(f"{i+1:4d}| {ln}" for i, ln in enumerate(content[:6000].splitlines()))
        file_blob = (f"\n\nThe bug is in `{tgt}`. Current content (line-numbered for reference; "
                     f"your SEARCH text must NOT include the line numbers):\n{numbered}")
    except Exception as e:
        file_blob = f"\n\n(could not read {tgt}: {e})"

TASK = (f"Resolve this issue by editing the repository.\n\n## Problem\n"
        f"{inst.get('problem_statement','')[:3000]}{file_blob}\n\n"
        f"Make minimal search-replace edits to fix it, then emit <done/>.")

print(f"[load model]", flush=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct",
        torch_dtype=torch.bfloat16, device_map="auto").eval()

print(f"[run agent] condition={COND}", flush=True)
agent = StreamAgent(model, tok, env, condition=COND, latency_tokens=12, max_new_tokens=1600)
r = agent.run(TASK, tgt)
ev = r["events"]
print(f"\n=== {IID} / {COND} ===")
print(f"resolved={r['resolved']}  tests={r['tests']}")
print(f"edits={sum(1 for e in ev if e['type']=='edit')}  "
      f"diag_injections={sum(1 for e in ev if e['type'].startswith('diag'))}  tokens={r['n_tokens']}")
print(f"metrics={r['metrics']}")
os.makedirs("runs/d_realtask", exist_ok=True)
with open(f"runs/d_realtask/{IID}_{COND}_stream.txt", "w") as f:
    f.write(r["stream"])
print(f"\n--- STREAM (first 2000 chars) ---\n{r['stream'][:2000]}")
print(f"\n--- agent patch ---\n{env.current_patch()[:1500]}")
for e in ev:
    if e["type"].startswith("diag"):
        print(f"  [diag {e['type']} @tok{e['tok']}] {e['text'][:80]!r}")
env.close()
