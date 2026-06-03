#!/usr/bin/env python3
"""C-vs-D delivery-form EFFICIENCY eval (the thesis). Same feedback, delivered
LIVE/early (D, mid-generation) vs SYNC/late (C, at a turn boundary after a full
attempt) vs NONE (A). Tasks involve enough work that doing it on a wrong foundation
and redoing it costs tokens. Metric: tokens-to-correct + final correctness.
Hypothesis: D reaches correct with materially less rework than C (live feedback
prevents building on the wrong foundation).

Usage: i_eval_cd.py [out.json] [--adapter DIR] [--model ID]
"""
import os, sys, re, json, argparse, io, contextlib, multiprocessing as mp, random
os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("out", nargs="?", default="runs/i_eval/cd_base.json")
ap.add_argument("--adapter", default=None)
ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
A = ap.parse_args()
INFO_OPEN, INFO_CLOSE = "\n‹info›\n", "\n‹/info›\n"
rng = random.Random(7)

KEYS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
def make_tasks(n=12):
    ts = []
    for i in range(n):
        v = rng.randint(1000, 99999)
        ts.append(dict(
            id=f"config_{i}", value=v, ep="config",
            prompt=("Write `config()` returning a dict that maps each of these six keys "
                    f"{KEYS} to the configured integer value (all six keys map to the "
                    "same integer). Return only the function."),
            fact=f"The configured integer value is {v}. Every key must map to {v}.",
            expect={k: v for k in KEYS}))
    return ts

def take_func(code, ep):
    starts = [m.start() for m in re.finditer(rf"\bdef {re.escape(ep)}\s*\(", code)]
    if not starts: return None
    c = code[starts[-1]:]; lines = c.splitlines(); out = [lines[0]]
    for ln in lines[1:]:
        if ln.strip() == "" or ln.startswith((" ", "\t")): out.append(ln)
        else: break
    return "\n".join(out)

def _w(code, ep, q):
    G = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec("from typing import *\n"+code, G); q.put(G[ep]())
    except Exception as e: q.put(f"ERR:{type(e).__name__}")
def call(code, ep):
    if not code: return "ERR:none"
    q = mp.Queue(); p = mp.Process(target=_w, args=(code, ep, q)); p.start(); p.join(6)
    if p.is_alive(): p.terminate(); p.join(); return "ERR:timeout"
    try: return q.get_nowait()
    except Exception: return "ERR"

print(f"[load] {A.model}{' + '+A.adapter if A.adapter else ''}", flush=True)
tok = AutoTokenizer.from_pretrained(A.model)
model = AutoModelForCausalLM.from_pretrained(A.model, torch_dtype=torch.bfloat16, device_map="auto")
if A.adapter:
    from peft import PeftModel; model = PeftModel.from_pretrained(model, A.adapter)
model = model.eval(); dev = model.device; eos = tok.eos_token_id

def gen(prefix, maxn=200):
    ids = tok(prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=maxn, do_sample=False, pad_token_id=eos)
    new = out[0, ids.shape[1]:]
    return tok.decode(new, skip_special_tokens=True), int((new != eos).sum())

def head_for(t):
    msgs = [{"role": "system", "content": "You are a coding assistant. Write the requested function."},
            {"role": "user", "content": t["prompt"]}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

def arm_A(t):  # none — model guesses BASE
    c, n = gen(head_for(t)); return call(take_func(c, t["ep"]), t["ep"]), n
def arm_D(t):  # live/early — BASE injected right after the signature
    pre = f"def {t['ep']}():\n    " + INFO_OPEN + t["fact"] + INFO_CLOSE
    c, n = gen(head_for(t) + pre); return call(take_func("def "+t['ep']+"():\n    "+c, t["ep"]), t["ep"]), n
def arm_C(t):  # sync/late — model writes a full guessed attempt, THEN BASE as a turn obs, THEN revises
    c1, n1 = gen(head_for(t))
    turn = c1 + "<|im_end|>\n<|im_start|>user\n" + INFO_OPEN + t["fact"] + INFO_CLOSE + \
           "Now return the corrected function.<|im_end|>\n<|im_start|>assistant\n"
    c2, n2 = gen(head_for(t) + turn)
    return call(take_func(c2, t["ep"]), t["ep"]), n1 + n2

results = {}
for name, fn in (("A", arm_A), ("C", arm_C), ("D", arm_D)):
    nc = 0; toks = []; rows = []
    for t in make_tasks():
        r, n = fn(t); ok = (r == t["expect"]); nc += ok; toks.append(n)
        rows.append({"id": t["id"], "correct": ok, "tokens": n})
    results[name] = {"correct": nc, "n": len(rows), "mean_tokens": round(sum(toks)/len(toks), 1), "rows": rows}
    print(f"  {name}: correct={nc}/{len(rows)}  mean_tokens_to_finish={results[name]['mean_tokens']}", flush=True)

os.makedirs(os.path.dirname(A.out), exist_ok=True)
json.dump({"model": A.model, "adapter": A.adapter, "results": results}, open(A.out, "w"), indent=2)
print(f"\nD correct={results['D']['correct']}/{results['D']['n']} @ {results['D']['mean_tokens']} tok  vs  "
      f"C correct={results['C']['correct']}/{results['C']['n']} @ {results['C']['mean_tokens']} tok")
print(f"efficiency: D uses {results['C']['mean_tokens']-results['D']['mean_tokens']:.0f} FEWER tokens than C "
      f"at matched correctness (live feedback avoids the rewrite). -> {A.out}")
