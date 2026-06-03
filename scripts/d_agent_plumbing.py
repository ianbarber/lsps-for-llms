#!/usr/bin/env python3
"""Validate the diagnostic-injection PLUMBING deterministically: force pyrefly to
return a canned diagnostic, then confirm C injects it synchronously (at the edit)
and D splices it asynchronously (edit_token + latency), and the text lands in the
stream. (The toy model fixes bugs first-try, so the real injection path only fires
on hard Goldilocks tasks — this isolates the mechanism.)"""
import os, sys
os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")
sys.path.insert(0, "/home/ianbarber/Projects/Streams")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scaffold.stream_agent import StreamAgent
from scaffold.mock_env import MockEnv

MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
FORCED = "[error] L2 bad-assignment: Literal[''] is not assignable to int"
BUGGY = 'def total(xs: list[int]) -> int:\n    s: int = ""\n    for x in xs:\n        s += x\n    return s\n'
TEST = "assert total([1,2,3])==6"
TASK = ("sol.py has a bug: variable s is annotated int but set to a string.\n```python\n"
        + BUGGY + "```\nFix sol.py with one edit, then emit <done/>.")

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto").eval()

LAT = 8
for cond in ("C", "D"):
    env = MockEnv(BUGGY, TEST, "total", force_diag=FORCED)
    r = StreamAgent(model, tok, env, condition=cond, latency_tokens=LAT, max_new_tokens=400).run(TASK)
    ev = r["events"]
    edit = next((e for e in ev if e["type"]=="edit"), None)
    diag = next((e for e in ev if e["type"].startswith("diag")), None)
    in_stream = FORCED in r["stream"]
    print(f"\n=== {cond} ===")
    print(f"  edit@tok={edit['tok'] if edit else None}  diag={diag['type'] if diag else None}@tok={diag['tok'] if diag else None}")
    if cond=="C": print(f"  PLUMBING {'OK' if (diag and diag['type']=='diag_sync' and in_stream) else 'FAIL'} (sync inject at edit, text in stream={in_stream})")
    if cond=="D":
        ok = bool(diag and diag['type']=='diag_async' and in_stream)
        mode = "on-done" if (diag and diag.get('ondone')) else ("async@latency" if diag else "none")
        offset = (diag['tok']-edit['tok']) if (diag and edit) else None
        print(f"  delivered={mode}  offset={offset}  text_in_stream={in_stream}  PLUMBING {'OK' if ok else 'FAIL'}")
    env.close()
print("\n[plumbing done]")
