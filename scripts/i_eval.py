#!/usr/bin/env python3
"""Interleaving-consumption eval (Rung 0): can the model USE information injected
mid-generation (react) without echoing it (parrot)? Headroom is guaranteed — the
injected value is random, so the model cannot guess it without reading the inject.

Per instance, two arms:
  no-inject : generate normally (model can't know the value -> reaction ~0)
  inject    : splice ‹info›FACT‹/info› mid-generation -> trained model should react
Metrics: REACTION (generated code runs and returns the injected value),
         PARROTING (model echoes the ‹info›/‹/info› delimiters into its own output).

Usage: i_eval.py [out.json] [--adapter DIR] [--model ID]
"""
import os, sys, re, json, argparse, io, contextlib, multiprocessing as mp, random
os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("out", nargs="?", default="runs/i_eval/base.json")
ap.add_argument("--adapter", default=None)
ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
ap.add_argument("--inject-at", type=int, default=24, help="token offset to splice the inject")
A = ap.parse_args()

INFO_OPEN, INFO_CLOSE = "\n‹info›\n", "\n‹/info›\n"
rng = random.Random(0)

def make_instances(n=16):
    inst = []
    names = ["limit", "threshold", "max_size", "timeout", "port", "retries",
             "buffer_len", "seed", "offset", "capacity", "window", "batch"]
    for i in range(n):
        nm = names[i % len(names)]
        val = rng.randint(1000, 99999)
        inst.append(dict(
            id=f"return_{nm}_{i}",
            prompt=(f"Write a Python function `get_{nm}()` that takes no arguments and "
                    f"returns the configured {nm} as an integer. Return only the function."),
            fact=f"The configured {nm} is {val}. get_{nm}() must return exactly {val}.",
            ep=f"get_{nm}", value=val))
    return inst

def take_func(code):
    # keep the def line + its indented body, stop at the first dedented non-blank line
    lines = code.splitlines()
    if not lines: return code
    out = [lines[0]]
    for ln in lines[1:]:
        if ln.strip() == "" or ln.startswith((" ", "\t")):
            out.append(ln)
        else:
            break
    return "\n".join(out)

def _w(code, ep, q):
    G = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec("from typing import *\n"+code, G)
            r = G[ep]()
        q.put(r)
    except Exception as e:
        q.put(f"ERR:{type(e).__name__}")
def run_fn(code, ep):
    q = mp.Queue(); p = mp.Process(target=_w, args=(code, ep, q)); p.start(); p.join(6)
    if p.is_alive(): p.terminate(); p.join(); return "ERR:timeout"
    try: return q.get_nowait()
    except Exception: return "ERR:noresult"

print(f"[load] {A.model}{' + '+A.adapter if A.adapter else ''}", flush=True)
tok = AutoTokenizer.from_pretrained(A.model)
model = AutoModelForCausalLM.from_pretrained(A.model, torch_dtype=torch.bfloat16, device_map="auto")
if A.adapter:
    from peft import PeftModel; model = PeftModel.from_pretrained(model, A.adapter)
model = model.eval(); dev = model.device
eos = tok.eos_token_id

def gen_with_optional_inject(prompt, fact, ep, inject):
    # Force the assistant to start with the signature, inject ‹info› right before the
    # body (matches the training layout exactly), then let the model complete.
    msgs = [{"role": "system", "content": "You are a coding assistant. Write the requested function."},
            {"role": "user", "content": prompt}]
    head = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    preamble = f"def {ep}() -> int:\n    "
    prefix = head + preamble + (INFO_OPEN + fact + INFO_CLOSE if inject else "")
    ids = tok(prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=64, do_sample=False, pad_token_id=eos)
    completion = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    full_code = preamble + completion          # what the model produced as the function
    return full_code, completion               # (code-to-run, model-completion-for-parrot-check)

results = []
for arm in ("no_inject", "inject"):
    n_react = n_parrot = 0
    rows = []
    for ins in make_instances():
        full_code, completion = gen_with_optional_inject(ins["prompt"], ins["fact"], ins["ep"], inject=(arm == "inject"))
        r = run_fn(take_func(full_code), ins["ep"])
        react = (r == ins["value"])
        parrot = ("‹info›" in completion or "‹/info›" in completion)
        n_react += react; n_parrot += parrot
        rows.append({"id": ins["id"], "react": react, "parrot": parrot, "ret": str(r)[:20]})
    results.append({"arm": arm, "n": len(rows), "reaction_rate": round(n_react/len(rows), 3),
                    "parrot_rate": round(n_parrot/len(rows), 3), "rows": rows})
    print(f"  {arm}: reaction={n_react}/{len(rows)}  parrot={n_parrot}/{len(rows)}", flush=True)

os.makedirs(os.path.dirname(A.out), exist_ok=True)
json.dump({"model": A.model, "adapter": A.adapter, "arms": results}, open(A.out, "w"), indent=2)
ni = next(r for r in results if r["arm"] == "no_inject"); inj = next(r for r in results if r["arm"] == "inject")
print(f"\nREACTION lift (inject - no_inject) = {inj['reaction_rate'] - ni['reaction_rate']:+.3f}")
print(f"PARROT rate (inject) = {inj['parrot_rate']:.3f}  -> {A.out}")
