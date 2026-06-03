#!/usr/bin/env python3
"""Diagnostic-DEPENDENCE probe (sharper than A-vs-D): does the model actually READ
the live diagnostic, or just emit the obvious fix regardless?

3-way per fixture: fix-rate under
  NONE    : no diagnostic
  CORRECT : the real diagnostic (points at the bug)         [condition D]
  WRONG   : a plausible-but-incorrect diagnostic (mis-points) [D-adversarial]

If the model uses the diagnostic: CORRECT > NONE > WRONG (a wrong diag misleads).
If it ignores it: the three are ~equal. Run on base and on each LoRA adapter to
see whether SFT taught genuine diagnostic-use.

Usage: d_diaguse.py [model] [out.json] [--adapter DIR]
"""
import os, sys, json, re, io, contextlib, argparse, multiprocessing as mp
os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("model", nargs="?", default="Qwen/Qwen2.5-Coder-7B-Instruct")
ap.add_argument("out", nargs="?", default="runs/d_diaguse/base.json")
ap.add_argument("--adapter", default=None)
A = ap.parse_args()

# fixture: buggy code, CORRECT diag (real bug), WRONG diag (mis-points), test, ep
FIX = [
 dict(name="off_by_one_slice",
   code="def last_n(xs: list, n: int) -> list:\n    return xs[len(xs)-n+1:]\n",
   correct="[error] L2 list-slice: off-by-one, returns n-1 elements; use xs[len(xs)-n:]",
   wrong="[error] L1 bad-param-type: n should be annotated Optional[int]",
   test="assert last_n([1,2,3,4,5],2)==[4,5]\nassert last_n([1,2,3],3)==[1,2,3]", ep="last_n"),
 dict(name="product_init",
   code="def product(xs: list) -> int:\n    p = 0\n    for x in xs:\n        p *= x\n    return p\n",
   correct="[error] L2 logic: accumulator p=0 makes product always 0; init p=1",
   wrong="[error] L4 unused: loop variable x flagged; rename to _",
   test="assert product([1,2,3,4])==24\nassert product([5])==5", ep="product"),
 dict(name="avg_intdiv",
   code="def average(xs: list) -> float:\n    return sum(xs) // len(xs)\n",
   correct="[error] L2 bad-return-type: // returns int but -> float; use /",
   wrong="[error] L2 zero-div: len(xs) may be 0; guard empty list",
   test="assert average([1,2])==1.5", ep="average"),
 dict(name="all_vs_any",
   code="def all_positive(xs: list) -> bool:\n    return any(x > 0 for x in xs)\n",
   correct="[error] L2 logic: uses any() but 'all positive' needs all()",
   wrong="[error] L2 bad-return-type: generator not bool; wrap in list()",
   test="assert all_positive([1,2,3]) is True\nassert all_positive([1,-1,3]) is False", ep="all_positive"),
 dict(name="get_or_zero",
   code="def get_or_zero(d: dict, k) -> int:\n    return d[k]\n",
   correct="[error] L2 key: d[k] raises KeyError when absent; use d.get(k,0)",
   wrong="[error] L1 bad-param-type: k should be annotated str",
   test="assert get_or_zero({'a':5},'a')==5\nassert get_or_zero({},'x')==0", ep="get_or_zero"),
 dict(name="str_plus_int",
   code="def double(n: int) -> int:\n    return n + n + ''\n",
   correct="[error] L2 bad-operand: '' (str) added to int; remove the + ''",
   wrong="[error] L1 missing-return: function may implicitly return None",
   test="assert double(3)==6\nassert double(0)==0", ep="double"),
 dict(name="countdown",
   code="def countdown(n: int) -> list:\n    return list(range(n))\n",
   correct="[error] L2 logic: range(n) ascends 0..n-1; countdown needs range(n,0,-1)",
   wrong="[error] L2 bad-return-type: range object not list; already wrapped — ok",
   test="assert countdown(3)==[3,2,1]\nassert countdown(1)==[1]", ep="countdown"),
 dict(name="first_even",
   code="def first_even(xs: list) -> int:\n    for x in xs:\n        if x % 2 == 1:\n            return x\n    return -1\n",
   correct="[error] L3 logic: x%2==1 selects ODD; for first even use x%2==0",
   wrong="[error] L5 unreachable: final return -1 flagged as dead code",
   test="assert first_even([1,3,4,5])==4\nassert first_even([1,3])==-1", ep="first_even"),
]

tok = AutoTokenizer.from_pretrained(A.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(A.model, torch_dtype=torch.bfloat16, device_map="auto")
if A.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, A.adapter); print(f"[adapter] {A.adapter}", flush=True)
model = model.eval()

def prompt(f, diag):
    body = f"```python\n{f['code']}```"
    if diag: body += f"\n\nStatic analysis reported:\n‹diag›\n{diag}\n‹/diag›"
    msg=[{"role":"user","content":"Fix the bug in this function and return ONLY the corrected "
          "function in a ```python code block:\n\n"+body}]
    return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

def extract(g):
    m=re.search(r"```(?:python)?\s*(.*?)```",g,re.S); return m.group(1) if m else g
def _w(code,test,q):
    G={}
    try:
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            exec("from typing import *\n"+code,G); exec(test,G)
        q.put(True)
    except Exception: q.put(False)
def rt(code,test):
    q=mp.Queue(); p=mp.Process(target=_w,args=(code,test,q)); p.start(); p.join(8)
    if p.is_alive(): p.terminate(); p.join(); return False
    try: return q.get_nowait()
    except Exception: return False
def gen(prompts):
    enc=tok(prompts,return_tensors="pt",padding=True).to(model.device)
    with torch.no_grad():
        o=model.generate(**enc,max_new_tokens=320,do_sample=False,pad_token_id=tok.pad_token_id)
    return tok.batch_decode(o[:,enc["input_ids"].shape[1]:],skip_special_tokens=True)

arms={"none":[prompt(f,None) for f in FIX],
      "correct":[prompt(f,f["correct"]) for f in FIX],
      "wrong":[prompt(f,f["wrong"]) for f in FIX]}
out={"model":A.model,"adapter":A.adapter,"n":len(FIX),"rates":{},"rows":[]}
res={}
for arm,ps in arms.items():
    gens=gen(ps); res[arm]=[rt(extract(g),f["test"]) for g,f in zip(gens,FIX)]
    out["rates"][arm]=sum(res[arm])/len(FIX)
for i,f in enumerate(FIX):
    out["rows"].append({"name":f["name"],**{a:res[a][i] for a in arms}})
os.makedirs(os.path.dirname(A.out),exist_ok=True); json.dump(out,open(A.out,"w"),indent=2)
r=out["rates"]
print(f"\n{A.adapter or 'BASE'}: none={r['none']:.2f}  correct={r['correct']:.2f}  wrong={r['wrong']:.2f}")
print(f"  uses-diagnostic signal: correct-none=+{r['correct']-r['none']:.2f}, wrong-none={r['wrong']-r['none']:+.2f}")
