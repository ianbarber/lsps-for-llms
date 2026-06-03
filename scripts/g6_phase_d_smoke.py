#!/usr/bin/env python3
"""Phase D smoke + D2 baseline repro: load model, run Phase D decoder EAGER
(no compile) for a few rows, confirm output identity vs Phase B eager, and
confirm it runs. Writes baseline_repro.json.
"""
from __future__ import annotations
import importlib.util, json, os, sys, time
from pathlib import Path

os.environ["HF_HOME"] = "/home/ianbarber/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/home/ianbarber/.cache/huggingface/hub"

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "JonasGeiping/stream-qwen3-8b"
OUT = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_d")
PATCH_D = OUT / "patched" / "stream_inference_phase_d.py"
PATCH_B = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_b/patched/stream_inference_phase_b.py")
PROMPTS = [
    "Write a Python function that reverses a linked list in place.",
    "Explain how a B-tree differs from a binary search tree.",
]

def load_patched(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def greedy(gen_fn, model, tok, silence, prompt, n, extra):
    rows = []
    for ri, row, isp in gen_fn(model, tok, prompt, silence, max_rows=n+5,
                               warm_start=False, temperature=0.0, **extra):
        if isp: continue
        rows.append(list(row))
        if len(rows) >= n: break
    return rows

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    snap = snapshot_download(MODEL_ID)
    if snap not in sys.path: sys.path.insert(0, snap)
    modD = load_patched(PATCH_D, "si_d"); modB = load_patched(PATCH_B, "si_b")
    print("[load]...", flush=True); t0=time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto"); model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    silence = modD.detect_silence_token(tok)
    print(f"[load] {time.perf_counter()-t0:.1f}s silence={silence}", flush=True)

    res = {"eager_identity": [], "all_match": True}
    extra = {"max_context_rows": 256}
    t0 = time.perf_counter()
    for i,p in enumerate(PROMPTS):
        d = greedy(modD.generate, model, tok, silence, p, 20, extra)
        b = greedy(modB.generate, model, tok, silence, p, 20, {})
        n = min(len(d), len(b)); match = d[:n]==b[:n] and len(d)==len(b)
        res["eager_identity"].append({"prompt": i, "match": bool(match),
            "d_rows": len(d), "b_rows": len(b)})
        res["all_match"] = res["all_match"] and match
        print(f"[smoke] prompt {i}: match={match} d={len(d)} b={len(b)}", flush=True)
    res["elapsed_s"] = time.perf_counter() - t0
    (OUT/"baseline_repro.json").write_text(json.dumps(res, indent=2))
    print(f"[smoke] ALL_MATCH={res['all_match']}  (Phase D eager == Phase B eager)", flush=True)

if __name__ == "__main__":
    main()
