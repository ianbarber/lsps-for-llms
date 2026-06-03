#!/usr/bin/env python3
"""Option-D substrate capability baseline: does the chosen CODER actually code?

Standard HF instruct model (model.generate works — no stream API), batched greedy
HumanEval. This is the de-risk we never had for the stream model: confirm the coder
is genuinely strong before building the interleaved-async mechanism on it.

Usage:
  d_capability_eval.py <model_id> <out.json> [--limit N] [--batch B]
"""
import os, sys, json, argparse, re, io, contextlib, signal, multiprocessing as mp
os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("model_id"); ap.add_argument("out")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--max-new", type=int, default=640)
args = ap.parse_args()

print(f"[load] {args.model_id}", flush=True)
tok = AutoTokenizer.from_pretrained(args.model_id)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    args.model_id, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()
print("[load] done", flush=True)

ds = load_dataset("openai/openai_humaneval", split="test")
items = list(ds)
if args.limit: items = items[:args.limit]

def build_prompt(p):
    msg = [{"role": "user", "content":
            "Complete this Python function. Return ONLY the complete function "
            "implementation inside a single ```python code block:\n\n" + p["prompt"]}]
    return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

def extract(gen, entry_point):
    # Keep the WHOLE fenced block (helper functions defined before the entry
    # point must be retained — slicing from `def entry_point` drops them and
    # causes spurious NameErrors). Fall back to from-first-def if no fence.
    m = re.search(r"```(?:python)?\s*(.*?)```", gen, re.S)
    if m:
        return m.group(1)
    idx = gen.find("def ")
    return gen[idx:] if idx != -1 else gen

def _worker(code, test, entry_point, q):
    g = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(code, g)
            exec(test, g)
            g["check"](g[entry_point])
        q.put(("pass", None))
    except Exception as e:
        q.put(("fail", f"{type(e).__name__}: {e}"))

def run_test(code, test, entry_point, prompt="", timeout=10):
    # Canonical HumanEval convention: the completion relies on the prompt's
    # imports/preamble. Instruct models regenerate the def but NOT the imports,
    # so prepend the prompt's import lines + a typing safety-net before exec.
    imports = "\n".join(l for l in prompt.splitlines()
                        if l.startswith(("import ", "from ")))
    code = ("from typing import *\nimport math, re, collections, itertools, functools\n"
            + imports + "\n" + code)
    q = mp.Queue()
    p = mp.Process(target=_worker, args=(code, test, entry_point, q))
    p.start(); p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join(); return "fail", "timeout"
    try: return q.get_nowait()
    except Exception: return "fail", "no-result"

rows, n_pass = [], 0
for i in range(0, len(items), args.batch):
    chunk = items[i:i+args.batch]
    prompts = [build_prompt(p) for p in chunk]
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    gens = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    for p, gen in zip(chunk, gens):
        code = extract(gen, p["entry_point"])
        verdict, err = run_test(code, p["test"], p["entry_point"], prompt=p["prompt"])
        ok = verdict == "pass"; n_pass += ok
        rows.append({"task_id": p["task_id"], "pass": ok, "err": err,
                     "code_head": code[:200]})
    done = min(i+args.batch, len(items))
    print(f"[eval] {done}/{len(items)}  running pass@1={n_pass/done:.3f}", flush=True)

res = {"model": args.model_id, "n": len(items), "pass_at_1": n_pass/len(items),
       "n_pass": n_pass, "rows": rows}
with open(args.out, "w") as f: json.dump(res, f, indent=2)
print(f"\n[DONE] {args.model_id}: pass@1 = {n_pass}/{len(items)} = {res['pass_at_1']:.3f}", flush=True)
