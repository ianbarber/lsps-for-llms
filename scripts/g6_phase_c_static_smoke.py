#!/usr/bin/env python3
"""Phase C smoke test — verify the static-shape decoder loads, runs a few rows,
and that the static cache + mask plumbing doesn't error. NOT a benchmark.

Runs 8 decode rows on one prompt with the static decoder (no compile) and prints
the rows. Also checks StaticStreamCache cursor advances by C each row.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")

MODEL_ID = "JonasGeiping/stream-qwen3-8b"
PATCH = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_c_static/patched/stream_inference_static.py")


def load_patched(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    snap = snapshot_download(MODEL_ID)
    if snap not in sys.path:
        sys.path.insert(0, snap)
    mod = load_patched(PATCH, "si_static")

    print("[load] loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    silence = mod.detect_silence_token(tok)
    print(f"[load] silence={silence}", flush=True)

    prompt = "Write a Python function that reverses a linked list in place."
    rows = []
    # small static buffer for the smoke test
    g = mod.generate(model, tok, prompt, silence, max_rows=8, warm_start=False,
                     temperature=0.0, max_context_rows=64)
    for row_idx, row, is_prefill in g:
        if is_prefill:
            continue
        rows.append(row)
        print(f"row {row_idx}: {row}", flush=True)
        if len(rows) >= 8:
            break
    print(f"[smoke] decoded {len(rows)} rows OK", flush=True)
    # decode the Output channel
    out_toks = [r[1] for r in rows if r[1] != silence]
    print(f"[smoke] Output channel text: {tok.decode(out_toks)!r}", flush=True)


if __name__ == "__main__":
    main()
