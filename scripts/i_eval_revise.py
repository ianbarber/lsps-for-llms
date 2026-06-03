#!/usr/bin/env python3
"""Backward-REVISION interleaving eval (R0b) — the discriminating, LSP-relevant test.
The model is shown a buggy first attempt; a ‹diag› about the bug is injected
mid-stream; we measure whether its CONTINUATION emits a corrected version that
passes the test (revision) vs leaving the bug.

Arms: no_diag (no injection — model just wrote it, likely leaves the bug) vs
      diag (inject ‹diag›). Reaction = the LAST function in the output passes the test.

Usage: i_eval_revise.py [out.json] [--adapter DIR] [--model ID]
"""
import os, sys, re, json, argparse, io, contextlib, multiprocessing as mp
os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("out", nargs="?", default="runs/i_eval/revise_base.json")
ap.add_argument("--adapter", default=None)
ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
A = ap.parse_args()
DIAG_OPEN, DIAG_CLOSE = "\n‹diag›\n", "\n‹/diag›\n"

FIX = [
 dict(ep="add", buggy="def add(a, b):\n    return a - b\n", diag="line 2: '-' should be '+' (function must add)", test="assert add(2,3)==5 and add(0,0)==0"),
 dict(ep="is_even", buggy="def is_even(n):\n    return n % 2 == 1\n", diag="line 2: returns True for ODD n; for is_even use n % 2 == 0", test="assert is_even(4) and not is_even(3)"),
 dict(ep="first", buggy="def first(xs):\n    return xs[1]\n", diag="line 2: index 1 is the second element; first should return xs[0]", test="assert first([10,20])==10"),
 dict(ep="square", buggy="def square(x):\n    return x * 2\n", diag="line 2: x*2 doubles; square should be x*x", test="assert square(3)==9 and square(5)==25"),
 dict(ep="count_pos", buggy="def count_pos(xs):\n    return sum(1 for x in xs if x < 0)\n", diag="line 2: 'x<0' counts negatives; for positives use x>0", test="assert count_pos([1,-2,3])==2"),
 dict(ep="last", buggy="def last(xs):\n    return xs[0]\n", diag="line 2: xs[0] is the first; last should return xs[-1]", test="assert last([1,2,3])==3"),
 dict(ep="maxof", buggy="def maxof(a, b):\n    return a if a < b else b\n", diag="line 2: 'a<b -> a' returns the smaller; for max use a if a>b else b", test="assert maxof(3,7)==7 and maxof(9,2)==9"),
 dict(ep="dbl_list", buggy="def dbl_list(xs):\n    return [x + 2 for x in xs]\n", diag="line 2: 'x+2' adds 2; to double use x*2", test="assert dbl_list([1,2,3])==[2,4,6]"),
 dict(ep="strip_neg", buggy="def strip_neg(xs):\n    return [x for x in xs if x < 0]\n", diag="line 2: keeps negatives; to remove them keep x>=0", test="assert strip_neg([1,-2,3])==[1,3]"),
 dict(ep="join_words", buggy="def join_words(ws):\n    return ','.join(ws)\n", diag="line 2: joins with comma; should join with a single space", test="assert join_words(['a','b'])=='a b'"),
]

def last_func(text, ep):
    # the LAST `def ep(` block in the text (corrected version if the model revised)
    starts = [m.start() for m in re.finditer(rf"\bdef {re.escape(ep)}\s*\(", text)]
    if not starts: return None
    code = text[starts[-1]:]
    lines = code.splitlines(); out = [lines[0]]
    for ln in lines[1:]:
        if ln.strip() == "" or ln.startswith((" ", "\t")): out.append(ln)
        else: break
    return "\n".join(out)

def _w(code, test, q):
    G = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec("from typing import *\n"+code, G); exec(test, G)
        q.put(True)
    except Exception: q.put(False)
def run_test(code, test):
    if not code: return False
    q = mp.Queue(); p = mp.Process(target=_w, args=(code, test, q)); p.start(); p.join(6)
    if p.is_alive(): p.terminate(); p.join(); return False
    try: return q.get_nowait()
    except Exception: return False

print(f"[load] {A.model}{' + '+A.adapter if A.adapter else ''}", flush=True)
tok = AutoTokenizer.from_pretrained(A.model)
model = AutoModelForCausalLM.from_pretrained(A.model, torch_dtype=torch.bfloat16, device_map="auto")
if A.adapter:
    from peft import PeftModel; model = PeftModel.from_pretrained(model, A.adapter)
model = model.eval(); dev = model.device; eos = tok.eos_token_id

def run(ins, inject):
    msgs = [{"role": "system", "content": "You are a coding assistant fixing a bug. If the static "
             "analyzer reports a problem, output a corrected version of the function."},
            {"role": "user", "content": f"Implement and verify this function:\n{ins['buggy']}"}]
    head = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    # assistant starts with the buggy attempt (the model's "first draft"), then the diag
    assistant = ins["buggy"] + (DIAG_OPEN + ins["diag"] + DIAG_CLOSE if inject else "")
    ids = tok(head + assistant, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=160, do_sample=False, pad_token_id=eos)
    cont = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    full = assistant + cont
    return run_test(last_func(full, ins["ep"]), ins["test"]), cont

results = []
for arm in ("no_diag", "diag"):
    n = 0; rows = []
    for ins in FIX:
        ok, cont = run(ins, inject=(arm == "diag")); n += ok
        rows.append({"ep": ins["ep"], "revised": ok})
    results.append({"arm": arm, "n": len(FIX), "revise_rate": round(n/len(FIX), 3), "rows": rows})
    print(f"  {arm}: revised={n}/{len(FIX)}", flush=True)

os.makedirs(os.path.dirname(A.out), exist_ok=True)
json.dump({"model": A.model, "adapter": A.adapter, "arms": results}, open(A.out, "w"), indent=2)
nd = next(r for r in results if r["arm"] == "no_diag"); d = next(r for r in results if r["arm"] == "diag")
print(f"\nREVISION lift (diag - no_diag) = {d['revise_rate'] - nd['revise_rate']:+.3f}  -> {A.out}")
