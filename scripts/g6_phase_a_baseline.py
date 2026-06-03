#!/usr/bin/env python3
"""Phase A baseline — clean GPU re-run of the G6 throughput benchmark.

Same 5 prompts x 256 productive Output tokens, 50-row warmup.
Captures: tok/s mean+std, per-prompt wall-clock, peak GPU memory,
elapsed warm-up time.

Writes runs/g6_phase_a/baseline_clean.json
"""
from __future__ import annotations

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

PROMPTS = [
    "Write a Python function that reverses a linked list in place.",
    "Explain how a B-tree differs from a binary search tree.",
    "Refactor this code to use a context manager: open('f.txt'); read(); close().",
    "What is the time complexity of merge sort, and why?",
    "Sketch a unit test for a function that adds two integers.",
]


def setup_paths():
    snap = snapshot_download(MODEL_ID)
    if snap not in sys.path:
        sys.path.insert(0, snap)
    return snap


def load_model_and_tokenizer():
    print(f"[load] loading {MODEL_ID} (bf16, device_map=auto)...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"[load] done in {time.time() - t0:.1f}s", flush=True)
    return model, tok


def measure_prompt(model, tok, silence_token, gen_fn, prompt, n_output_tokens,
                   channels_productive):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    productive = 0
    total = 0
    rows_run = 0
    max_rows = max(n_output_tokens * 10, 500)
    g = gen_fn(model, tok, prompt, silence_token, max_rows=max_rows,
               warm_start=False, temperature=0.0)
    for row_idx, row, is_prefill in g:
        if is_prefill:
            continue
        rows_run += 1
        total += len(row)
        for c in channels_productive:
            if c < len(row) and row[c] != silence_token:
                productive += 1
        if productive >= n_output_tokens:
            break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed, rows_run, productive, total


def bench(model, tok, silence_token, gen_fn, channels_productive, label,
          n_output_tokens=256, prompts=PROMPTS):
    print(f"\n[{label}] productive channels: {channels_productive}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    per_prompt = []
    for i, p in enumerate(prompts):
        elapsed, rows, prod, total = measure_prompt(
            model, tok, silence_token, gen_fn, p, n_output_tokens,
            channels_productive)
        tps = prod / elapsed if elapsed > 0 else 0.0
        rps = rows / elapsed if elapsed > 0 else 0.0
        per_prompt.append({
            "prompt_idx": i,
            "elapsed_s": elapsed,
            "rows": rows,
            "productive_tokens": prod,
            "total_tokens_emitted": total,
            "productive_tokens_per_sec": tps,
            "rows_per_sec": rps,
        })
        print(f"[{label}] {i}: {elapsed:.2f}s {rows} rows {prod} prod "
              f"{tps:.2f} tok/s {rps:.2f} rows/s", flush=True)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    speeds = [m["productive_tokens_per_sec"] for m in per_prompt]
    rows_speeds = [m["rows_per_sec"] for m in per_prompt]
    mean = sum(speeds) / len(speeds)
    var = sum((s - mean) ** 2 for s in speeds) / max(len(speeds) - 1, 1)
    std = var ** 0.5
    mean_rows = sum(rows_speeds) / len(rows_speeds)
    summary = {
        "label": label,
        "channels_productive": channels_productive,
        "n_output_tokens_target": n_output_tokens,
        "per_prompt": per_prompt,
        "mean_productive_tokens_per_sec": mean,
        "std_productive_tokens_per_sec": std,
        "mean_rows_per_sec": mean_rows,
        "peak_memory_gb": peak_gb,
    }
    print(f"[{label}] MEAN {mean:.2f} +/- {std:.2f} tok/s | "
          f"{mean_rows:.2f} rows/s | peak {peak_gb:.2f} GB", flush=True)
    return summary


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    snap = setup_paths()
    from stream_inference import generate as gen_fn
    from stream_inference import detect_silence_token

    model, tok = load_model_and_tokenizer()
    silence_token = detect_silence_token(tok)
    print(f"[init] silence token id: {silence_token}", flush=True)

    # Warm-up — time it
    print(f"[warmup] 50 rows...", flush=True)
    torch.cuda.synchronize()
    t_warm0 = time.perf_counter()
    rows = 0
    g = gen_fn(model, tok, PROMPTS[0], silence_token, max_rows=50,
               warm_start=False, temperature=0.0)
    for r in g:
        rows += 1
        if rows >= 50:
            break
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - t_warm0
    print(f"[warmup] {rows} rows in {warmup_s:.2f}s", flush=True)

    # Single stream
    single = bench(model, tok, silence_token, gen_fn,
                   channels_productive=[1], label="single_stream")
    multi = bench(model, tok, silence_token, gen_fn,
                  channels_productive=[1, 2], label="multi_stream")

    out = {
        "warmup_rows": rows,
        "warmup_elapsed_s": warmup_s,
        "single_stream": single,
        "multi_stream": multi,
        "snapshot_dir": snap,
    }
    (OUT_DIR / "baseline_clean.json").write_text(json.dumps(out, indent=2))
    print(f"\n[done] wrote {OUT_DIR}/baseline_clean.json")


if __name__ == "__main__":
    main()
