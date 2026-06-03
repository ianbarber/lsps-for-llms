#!/usr/bin/env python3
"""Option-D interleaved-async CANARY (v0.5 §0.8, new G2).

End-to-end integration test of the mechanism + model: does an inline diagnostic
CAUSALLY help a real coder? Each fixture is a function with a bug that is hard to
spot from a quick read but a diagnostic pinpoints. We ask the model to return the
corrected function in two conditions and run the unit test:
  A: code only (no diagnostic)
  D: code + inline ‹diag›...‹/diag› block (the real-ish pyrefly diagnostic)
Canary PASSES if D fix-rate > A fix-rate (the diagnostic is used). Zero-shot on the
base instruct coder (SFT on the ‹diag› format will only sharpen this later).
"""
import os, sys, json, re, io, contextlib, multiprocessing as mp
os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("model", nargs="?", default="Qwen/Qwen2.5-Coder-7B-Instruct")
_ap.add_argument("out", nargs="?", default="runs/d_canary/canary_7b.json")
_ap.add_argument("--adapter", default=None, help="optional LoRA adapter dir to load on top")
_a = _ap.parse_args()
MODEL, OUT, ADAPTER = _a.model, _a.out, _a.adapter

# Each: buggy code, a diagnostic pointing at the real bug, a test, entry point.
FIX = [
 dict(name="off_by_one_slice",
   code="def last_n(xs: list, n: int) -> list:\n    return xs[len(xs)-n+1:]\n",
   diag="[error] L2 list-slice: returns n-1 elements, off-by-one; expected xs[len(xs)-n:]",
   test="assert last_n([1,2,3,4,5],2)==[4,5]\nassert last_n([1,2,3],3)==[1,2,3]",
   ep="last_n"),
 dict(name="wrong_accumulator_init",
   code="def product(xs: list) -> int:\n    p = 0\n    for x in xs:\n        p *= x\n    return p\n",
   diag="[error] L2 logic: accumulator p initialized to 0 makes product always 0; init to 1",
   test="assert product([1,2,3,4])==24\nassert product([5])==5",
   ep="product"),
 dict(name="mutates_default",
   code="def add_item(x, acc: list = []) -> list:\n    acc.append(x)\n    return acc\n",
   diag="[warning] L1 mutable-default: shared list across calls; use acc=None then acc=[]",
   test="assert add_item(1)==[1]\nassert add_item(2)==[2]  # must NOT be [1,2]",
   ep="add_item"),
 dict(name="int_div",
   code="def average(xs: list) -> float:\n    return sum(xs) // len(xs)\n",
   diag="[error] L2 bad-return-type: // returns int, function annotated -> float; use /",
   test="assert average([1,2])==1.5",
   ep="average"),
 dict(name="wrong_compare",
   code="def all_positive(xs: list) -> bool:\n    return any(x > 0 for x in xs)\n",
   diag="[error] L2 logic: uses any() but should be all() for 'all positive'",
   test="assert all_positive([1,2,3]) is True\nassert all_positive([1,-1,3]) is False",
   ep="all_positive"),
 dict(name="key_error",
   code="def get_or_zero(d: dict, k) -> int:\n    return d[k]\n",
   diag="[error] L2 possibly-undefined-key: d[k] raises KeyError when k absent; use d.get(k,0)",
   test="assert get_or_zero({'a':5},'a')==5\nassert get_or_zero({},'x')==0",
   ep="get_or_zero"),
 dict(name="str_vs_int",
   code="def double(n: int) -> int:\n    return n + n + ''\n",
   diag="[error] L2 bad-operand-type: '' (str) added to int n; remove the + ''",
   test="assert double(3)==6\nassert double(0)==0",
   ep="double"),
 dict(name="reversed_range",
   code="def countdown(n: int) -> list:\n    return list(range(n))\n",
   diag="[error] L2 logic: range(n) is ascending 0..n-1; countdown needs range(n,0,-1)",
   test="assert countdown(3)==[3,2,1]\nassert countdown(1)==[1]",
   ep="countdown"),
]

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
if ADAPTER:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER)
    print(f"[adapter] loaded {ADAPTER}", flush=True)
model = model.eval()

def prompt(f, with_diag):
    body = f"```python\n{f['code']}```"
    if with_diag:
        body += f"\n\nStatic analysis reported:\n‹diag›\n{f['diag']}\n‹/diag›"
    msg = [{"role":"user","content":
            "Fix the bug in this function and return ONLY the corrected function "
            "in a ```python code block:\n\n"+body}]
    return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

def extract(g):
    m=re.search(r"```(?:python)?\s*(.*?)```",g,re.S); return m.group(1) if m else g

def _w(code,test,ep,q):
    G={}
    try:
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            exec("from typing import *\n"+code,G); exec(test,G)
        q.put(True)
    except Exception: q.put(False)
def runtest(code,test,ep):
    q=mp.Queue(); p=mp.Process(target=_w,args=(code,test,ep,q)); p.start(); p.join(8)
    if p.is_alive(): p.terminate(); p.join(); return False
    try: return q.get_nowait()
    except Exception: return False

def gen_batch(prompts):
    enc=tok(prompts,return_tensors="pt",padding=True).to(model.device)
    with torch.no_grad():
        o=model.generate(**enc,max_new_tokens=320,do_sample=False,pad_token_id=tok.pad_token_id)
    return tok.batch_decode(o[:,enc["input_ids"].shape[1]:],skip_special_tokens=True)

rows=[]; A_pass=D_pass=0
ga=gen_batch([prompt(f,False) for f in FIX])
gd=gen_batch([prompt(f,True) for f in FIX])
for f,a,d in zip(FIX,ga,gd):
    ap=runtest(extract(a),f["test"],f["ep"]); dp=runtest(extract(d),f["test"],f["ep"])
    A_pass+=ap; D_pass+=dp
    rows.append(dict(name=f["name"],A_pass=ap,D_pass=dp))
    print(f"{f['name']:22s} A={'P' if ap else 'F'}  D={'P' if dp else 'F'}",flush=True)

n=len(FIX)
res=dict(model=MODEL,adapter=ADAPTER,n=n,A_fix_rate=A_pass/n,D_fix_rate=D_pass/n,
         canary_pass=D_pass>A_pass,rows=rows)
os.makedirs(os.path.dirname(OUT),exist_ok=True); json.dump(res,open(OUT,"w"),indent=2)
print(f"\nA fix-rate={A_pass}/{n}={A_pass/n:.2f}  D fix-rate={D_pass}/{n}={D_pass/n:.2f}")
print(f"CANARY {'PASS (D>A: diagnostic causally helps)' if D_pass>A_pass else 'INCONCLUSIVE (D<=A)'}")
