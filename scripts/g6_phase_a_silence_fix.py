#!/usr/bin/env python3
"""Phase A4 — apply silence_penalty to Analytical too, re-measure multi-stream.

Uses the patched stream_inference at runs/g6_phase_a/patched/.
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
OUT_DIR = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_a")
PATCH = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_a/patched/stream_inference_silence_fix.py")

PROMPTS = [
    "Write a Python function that reverses a linked list in place.",
    "Explain how a B-tree differs from a binary search tree.",
    "Refactor this code to use a context manager: open('f.txt'); read(); close().",
    "What is the time complexity of merge sort, and why?",
    "Sketch a unit test for a function that adds two integers.",
]


def load_patched_module():
    spec = importlib.util.spec_from_file_location("stream_inference_patched", PATCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def measure_prompt(model, tok, silence_token, gen_fn, prompt, n_output_tokens,
                   channels_productive):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    productive = 0
    total = 0
    rows_run = 0
    silence_per_channel = [0] * 10
    max_rows = max(n_output_tokens * 10, 500)
    g = gen_fn(model, tok, prompt, silence_token, max_rows=max_rows,
               warm_start=False, temperature=0.0)
    for row_idx, row, is_prefill in g:
        if is_prefill:
            continue
        rows_run += 1
        total += len(row)
        for c in range(len(row)):
            if row[c] == silence_token:
                silence_per_channel[c] += 1
        for c in channels_productive:
            if c < len(row) and row[c] != silence_token:
                productive += 1
        if productive >= n_output_tokens:
            break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed, rows_run, productive, total, silence_per_channel


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    snap = snapshot_download(MODEL_ID)
    if snap not in sys.path:
        sys.path.insert(0, snap)

    patched = load_patched_module()
    gen_fn = patched.generate
    silence_fn = patched.detect_silence_token

    print(f"[load] loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="auto")
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    silence_token = silence_fn(tok)
    print(f"[init] silence token id: {silence_token}", flush=True)

    # Warmup
    print(f"[warmup] 50 rows...", flush=True)
    rows = 0
    g = gen_fn(model, tok, PROMPTS[0], silence_token, max_rows=50,
               warm_start=False, temperature=0.0)
    for r in g:
        rows += 1
        if rows >= 50:
            break

    torch.cuda.reset_peak_memory_stats()
    per_prompt = []
    for i, p in enumerate(PROMPTS):
        elapsed, rows, prod, total, sil_ch = measure_prompt(
            model, tok, silence_token, gen_fn, p, 256,
            channels_productive=[1, 2])
        tps = prod / elapsed if elapsed > 0 else 0.0
        rps = rows / elapsed if elapsed > 0 else 0.0
        sil_ratio_ch1 = sil_ch[1] / rows if rows else 0.0
        sil_ratio_ch2 = sil_ch[2] / rows if rows else 0.0
        per_prompt.append({
            "prompt_idx": i,
            "elapsed_s": elapsed,
            "rows": rows,
            "productive_tokens": prod,
            "total_tokens_emitted": total,
            "productive_tokens_per_sec": tps,
            "rows_per_sec": rps,
            "silence_ratio_ch1_output": sil_ratio_ch1,
            "silence_ratio_ch2_analytical": sil_ratio_ch2,
        })
        print(f"[multi_fix] {i}: {elapsed:.2f}s {rows} rows {prod} prod "
              f"{tps:.2f} tok/s | ch1_sil {sil_ratio_ch1:.2%} ch2_sil {sil_ratio_ch2:.2%}",
              flush=True)

    speeds = [m["productive_tokens_per_sec"] for m in per_prompt]
    mean = sum(speeds) / len(speeds)
    var = sum((s - mean) ** 2 for s in speeds) / max(len(speeds) - 1, 1)
    std = var ** 0.5
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    out = {
        "label": "multi_stream_silence_fix",
        "channels_productive": [1, 2],
        "per_prompt": per_prompt,
        "mean_productive_tokens_per_sec": mean,
        "std_productive_tokens_per_sec": std,
        "peak_memory_gb": peak_gb,
    }
    (OUT_DIR / "silence_fix.json").write_text(json.dumps(out, indent=2))
    print(f"[multi_fix] MEAN {mean:.2f} +/- {std:.2f} tok/s | peak {peak_gb:.2f} GB",
          flush=True)


if __name__ == "__main__":
    main()
