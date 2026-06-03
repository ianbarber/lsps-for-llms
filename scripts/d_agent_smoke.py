#!/usr/bin/env python3
"""Smoke-test the continuous-stream StreamAgent in A/C/D on a controlled type-bug
(real pyrefly diagnostic). Validates: edits parse+apply mid-stream, C injects sync,
D splices async, run_tests resolves. Reports per-condition resolved + metrics +
the diagnostic-injection events."""
import os, sys
os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")
sys.path.insert(0, "/home/ianbarber/Projects/Streams")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scaffold.stream_agent import StreamAgent
from scaffold.mock_env import MockEnv

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-Coder-7B-Instruct"

BUGGY = ('def total(xs: list[int]) -> int:\n'
         '    s: int = ""\n'            # BUG: str assigned to int-annotated var
         '    for x in xs:\n'
         '        s += x\n'
         '    return s\n')
TEST = "assert total([1,2,3])==6\nassert total([])==0"
TASK = ("The file sol.py contains a buggy function:\n\n```python\n" + BUGGY +
        "```\n\nThe variable s is annotated int but initialized to a string, so "
        "summation fails. Fix sol.py so the tests pass. Edit sol.py.")

print(f"[load] {MODEL}", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto").eval()
print("[load] done", flush=True)

for cond in ("A", "C", "D"):
    env = MockEnv(BUGGY, TEST, "total")
    agent = StreamAgent(model, tok, env, condition=cond, latency_tokens=8, max_new_tokens=700)
    r = agent.run(TASK)
    ev = r["events"]
    nedit = sum(1 for e in ev if e["type"] == "edit")
    ndiag = sum(1 for e in ev if e["type"].startswith("diag"))
    print(f"\n=== {cond} === resolved={r['resolved']}  edits={nedit}  diag_injections={ndiag}  "
          f"tokens={r['n_tokens']}", flush=True)
    print(f"  metrics: {r['metrics']}", flush=True)
    if ndiag:
        d = next(e for e in ev if e["type"].startswith("diag"))
        print(f"  first diag ({d['type']} @tok {d['tok']}): {d['text'][:90]!r}", flush=True)
    env.close()
print("\n[smoke done]", flush=True)
