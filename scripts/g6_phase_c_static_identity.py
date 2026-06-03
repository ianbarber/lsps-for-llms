#!/usr/bin/env python3
"""Phase C C4 — output identity gate.

Greedy (T=0.0) decode of 5 prompts x 30 rows with BOTH the Phase B decoder and
the Phase C static-shape decoder. The static decoder MUST produce bit-identical
token sequences. Divergence => mask/padding leak => invalid optimisation.

Writes runs/g6_phase_c_static/identity.json.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")

MODEL_ID = "JonasGeiping/stream-qwen3-8b"
OUT_DIR = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_c_static")
PATCH_B = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_b/patched/stream_inference_phase_b.py")
PATCH_C = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_c_static/patched/stream_inference_static.py")

PROMPTS = [
    "Write a Python function that reverses a linked list in place.",
    "Explain how a B-tree differs from a binary search tree.",
    "Refactor this code to use a context manager: open('f.txt'); read(); close().",
    "What is the time complexity of merge sort, and why?",
    "Sketch a unit test for a function that adds two integers.",
]
IDENTITY_ROWS = 30


def load_patched(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_rows(model, tok, silence, gen_fn, prompt, max_rows, **extra):
    rows = []
    g = gen_fn(model, tok, prompt, silence, max_rows=max_rows, warm_start=False,
               temperature=0.0, **extra)
    for row_idx, row, is_prefill in g:
        if is_prefill:
            continue
        rows.append(list(row))
        if len(rows) >= max_rows:
            break
    return rows


def first_divergence(a, b):
    R = min(len(a), len(b))
    for r in range(R):
        ar, br = a[r], b[r]
        Cn = min(len(ar), len(br))
        for c in range(Cn):
            if ar[c] != br[c]:
                return (r, c, ar[c], br[c])
        if len(ar) != len(br):
            return (r, -1, len(ar), len(br))
    if len(a) != len(b):
        return (R, -1, len(a), len(b))
    return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    snap = snapshot_download(MODEL_ID)
    if snap not in sys.path:
        sys.path.insert(0, snap)
    mod_b = load_patched(PATCH_B, "si_phase_b")
    mod_c = load_patched(PATCH_C, "si_static")

    print("[load] loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    silence = mod_b.detect_silence_token(tok)
    print(f"[load] silence={silence}", flush=True)

    out = {"identity_rows": IDENTITY_ROWS, "per_prompt": []}
    for i, p in enumerate(PROMPTS):
        torch.manual_seed(42)
        t0 = time.perf_counter()
        b_rows = run_rows(model, tok, silence, mod_b.generate, p, IDENTITY_ROWS)
        t_b = time.perf_counter() - t0

        torch.manual_seed(42)
        t0 = time.perf_counter()
        # small static buffer is fine for 30 rows; use 256 rows headroom.
        c_rows = run_rows(model, tok, silence, mod_c.generate, p, IDENTITY_ROWS,
                          max_context_rows=256)
        t_c = time.perf_counter() - t0

        div = first_divergence(b_rows, c_rows)
        verdict = "identical" if div is None else (
            f"divergent@row={div[0]},col={div[1]} B={div[2]} C={div[3]}")
        out["per_prompt"].append({
            "prompt_idx": i,
            "phase_b_rows": len(b_rows),
            "static_rows": len(c_rows),
            "phase_b_seconds": t_b,
            "static_seconds": t_c,
            "verdict": verdict,
            "divergence": div,
        })
        print(f"[identity] prompt {i}: {verdict}  (B={t_b:.1f}s, C={t_c:.1f}s)", flush=True)

    all_identical = all(p["verdict"] == "identical" for p in out["per_prompt"])
    out["all_identical"] = all_identical
    (OUT_DIR / "identity.json").write_text(json.dumps(out, indent=2))
    print(f"\n[done] all_identical={all_identical}; wrote {OUT_DIR}/identity.json", flush=True)
    sys.exit(0 if all_identical else 1)


if __name__ == "__main__":
    main()
