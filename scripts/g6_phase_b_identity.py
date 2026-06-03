#!/usr/bin/env python3
"""Phase B identity check (B7) — compare Phase A vs Phase B outputs.

For each of the 5 prompts, run the Phase A silence-fix generator AND the
Phase B tensorized+vectorized-mask generator under T=0.0 (greedy). Save
both decoded token-grids and check identity. Optionally also compare
Phase B + compile.

The point: Phase B optimisations (sampling tensorization, mask vectorization)
MUST not change outputs in greedy mode. If they do, the patch is invalid.

Writes runs/g6_phase_b/identity.json with per-prompt diff info.
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
OUT_DIR = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_b")
PATCH_A = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_a/patched/stream_inference_silence_fix.py")
PATCH_B = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_b/patched/stream_inference_phase_b.py")

PROMPTS = [
    "Write a Python function that reverses a linked list in place.",
    "Explain how a B-tree differs from a binary search tree.",
    "Refactor this code to use a context manager: open('f.txt'); read(); close().",
    "What is the time complexity of merge sort, and why?",
    "Sketch a unit test for a function that adds two integers.",
]

# Short identity decode — 30 rows is plenty to surface any divergence.
IDENTITY_ROWS = 30


def load_patched_module(patch_path: Path, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, patch_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_rows(model, tok, silence_token, gen_fn, prompt, max_rows):
    rows = []
    g = gen_fn(model, tok, prompt, silence_token, max_rows=max_rows,
               warm_start=False, temperature=0.0)
    for row_idx, row, is_prefill in g:
        if is_prefill:
            continue
        rows.append(list(row))
        if len(rows) >= max_rows:
            break
    return rows


def first_divergence(a_rows, b_rows):
    """Return (row, col) of first differing entry, or None if identical."""
    R = min(len(a_rows), len(b_rows))
    for r in range(R):
        ar, br = a_rows[r], b_rows[r]
        C = min(len(ar), len(br))
        for c in range(C):
            if ar[c] != br[c]:
                return (r, c, ar[c], br[c])
        if len(ar) != len(br):
            return (r, -1, len(ar), len(br))
    if len(a_rows) != len(b_rows):
        return (R, -1, len(a_rows), len(b_rows))
    return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    snap = snapshot_download(MODEL_ID)
    if snap not in sys.path:
        sys.path.insert(0, snap)

    mod_a = load_patched_module(PATCH_A, "si_phase_a")
    mod_b = load_patched_module(PATCH_B, "si_phase_b")

    print("[load] loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="auto")
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    silence_token = mod_a.detect_silence_token(tok)
    print(f"[load] silence={silence_token}", flush=True)

    out = {"identity_rows": IDENTITY_ROWS, "per_prompt": []}
    for i, p in enumerate(PROMPTS):
        torch.manual_seed(42)
        t0 = time.perf_counter()
        a_rows = run_rows(model, tok, silence_token, mod_a.generate, p, IDENTITY_ROWS)
        t_a = time.perf_counter() - t0

        torch.manual_seed(42)
        t0 = time.perf_counter()
        b_rows = run_rows(model, tok, silence_token, mod_b.generate, p, IDENTITY_ROWS)
        t_b = time.perf_counter() - t0

        div = first_divergence(a_rows, b_rows)
        verdict = "identical" if div is None else f"divergent@row={div[0]},col={div[1]} A={div[2]} B={div[3]}"
        out["per_prompt"].append({
            "prompt_idx": i,
            "phase_a_rows": len(a_rows),
            "phase_b_rows": len(b_rows),
            "phase_a_seconds": t_a,
            "phase_b_seconds": t_b,
            "verdict": verdict,
            "divergence": div,
        })
        print(f"[identity] prompt {i}: {verdict}  (A={t_a:.1f}s, B={t_b:.1f}s)",
              flush=True)

    all_identical = all(p["verdict"] == "identical" for p in out["per_prompt"])
    out["all_identical"] = all_identical
    (OUT_DIR / "identity.json").write_text(json.dumps(out, indent=2))
    print(f"\n[done] all_identical={all_identical}; wrote {OUT_DIR}/identity.json",
          flush=True)


if __name__ == "__main__":
    main()
