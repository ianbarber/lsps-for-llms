#!/usr/bin/env python3
"""Phase A3 — torch.compile probe.

Tries compile modes default -> reduce-overhead -> max-autotune.
Falls back to compiling forward/attention module in isolation if full compile fails.

Records: did it compile, graph breaks, tok/s, compile overhead, peak memory.
Writes runs/g6_phase_a/compile_probe.json
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")

MODEL_ID = "JonasGeiping/stream-qwen3-8b"
OUT_DIR = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_a")

# Use 2 prompts and 128 tokens to keep each compile-mode trial budgeted
PROMPTS = [
    "Write a Python function that reverses a linked list in place.",
    "Explain how a B-tree differs from a binary search tree.",
]
N_TOKENS = 128


def load_model():
    snap = snapshot_download(MODEL_ID)
    if snap not in sys.path:
        sys.path.insert(0, snap)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="auto")
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def measure(model, tok, silence_token, gen_fn, prompts, n_tokens):
    """Return (mean_tok_per_sec, first_call_s, peak_mem_gb)."""
    torch.cuda.reset_peak_memory_stats()
    first_call_s = None
    per_prompt = []
    for i, p in enumerate(prompts):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        productive = 0
        total = 0
        rows_run = 0
        g = gen_fn(model, tok, p, silence_token, max_rows=n_tokens * 10,
                   warm_start=False, temperature=0.0)
        for row_idx, row, is_prefill in g:
            if is_prefill:
                continue
            rows_run += 1
            total += len(row)
            if row[1] != silence_token:
                productive += 1
            if productive >= n_tokens:
                break
        torch.cuda.synchronize()
        el = time.perf_counter() - t0
        per_prompt.append({"i": i, "elapsed_s": el, "rows": rows_run,
                           "productive": productive,
                           "tok_s": productive / el if el else 0.0})
        if first_call_s is None:
            first_call_s = el
        print(f"  prompt {i}: {el:.2f}s {rows_run} rows {productive} prod "
              f"{productive/el:.2f} tok/s", flush=True)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    speeds = [m["tok_s"] for m in per_prompt]
    mean = sum(speeds) / len(speeds) if speeds else 0.0
    return mean, first_call_s, peak_gb, per_prompt


def attempt_mode(model, tok, silence_token, gen_fn, mode, results):
    print(f"\n=== Trying torch.compile(mode='{mode}') on full model ===", flush=True)
    res = {"mode": mode, "scope": "full_model", "compiled": False}
    t_comp_start = time.perf_counter()
    try:
        compiled = torch.compile(model, mode=mode)
        # Single-shot smoke probe with one prompt; this triggers the compile.
        torch.cuda.synchronize()
        t_first0 = time.perf_counter()
        g = gen_fn(compiled, tok, PROMPTS[0], silence_token, max_rows=20,
                   warm_start=False, temperature=0.0)
        rows_seen = 0
        for r in g:
            rows_seen += 1
            if rows_seen >= 5:
                break
        torch.cuda.synchronize()
        t_first1 = time.perf_counter()
        res["compiled"] = True
        res["compile_first_call_s"] = t_first1 - t_first0
        res["smoke_rows"] = rows_seen
        # Real measurement
        print(f"  smoke passed in {t_first1 - t_first0:.2f}s, running measurement...", flush=True)
        mean, first_s, peak_gb, per_prompt = measure(
            compiled, tok, silence_token, gen_fn, PROMPTS, N_TOKENS)
        res["mean_tok_per_sec"] = mean
        res["first_call_s"] = first_s
        res["peak_memory_gb"] = peak_gb
        res["per_prompt"] = per_prompt
        print(f"  MEAN {mean:.2f} tok/s | peak {peak_gb:.2f} GB", flush=True)
    except Exception as e:
        res["compiled"] = False
        res["error_type"] = type(e).__name__
        res["error"] = str(e)[:2000]
        res["traceback"] = traceback.format_exc()[-4000:]
        print(f"  FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
    res["total_elapsed_s"] = time.perf_counter() - t_comp_start
    results.append(res)
    return res


def attempt_module_compile(model, tok, silence_token, gen_fn, results):
    """Try compiling the inner self_attn module of one layer in isolation."""
    print(f"\n=== Trying torch.compile on layer[0].self_attn only ===", flush=True)
    res = {"mode": "default", "scope": "self_attn_layer0", "compiled": False}
    try:
        layer0 = model.model.layers[0]
        original = layer0.self_attn
        layer0.self_attn = torch.compile(original, mode="default")
        torch.cuda.synchronize()
        t_first0 = time.perf_counter()
        # smoke
        g = gen_fn(model, tok, PROMPTS[0], silence_token, max_rows=20,
                   warm_start=False, temperature=0.0)
        rows_seen = 0
        for r in g:
            rows_seen += 1
            if rows_seen >= 5:
                break
        torch.cuda.synchronize()
        res["compile_first_call_s"] = time.perf_counter() - t_first0
        res["compiled"] = True
        print(f"  smoke passed in {res['compile_first_call_s']:.2f}s", flush=True)
        mean, first_s, peak_gb, per_prompt = measure(
            model, tok, silence_token, gen_fn, PROMPTS, N_TOKENS)
        res["mean_tok_per_sec"] = mean
        res["peak_memory_gb"] = peak_gb
        res["per_prompt"] = per_prompt
        # restore
        layer0.self_attn = original
        print(f"  MEAN {mean:.2f} tok/s | peak {peak_gb:.2f} GB", flush=True)
    except Exception as e:
        try:
            layer0.self_attn = original
        except Exception:
            pass
        res["error_type"] = type(e).__name__
        res["error"] = str(e)[:2000]
        res["traceback"] = traceback.format_exc()[-4000:]
        print(f"  FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
    results.append(res)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch._dynamo.config.cache_size_limit = 64

    model, tok = load_model()
    from stream_inference import generate as gen_fn
    from stream_inference import detect_silence_token
    silence_token = detect_silence_token(tok)
    print(f"[init] silence token id: {silence_token}", flush=True)

    # baseline warmup so first-call latency below is purely the compile cost
    print(f"[warmup] 20 rows eager...", flush=True)
    g = gen_fn(model, tok, PROMPTS[0], silence_token, max_rows=20,
               warm_start=False, temperature=0.0)
    rows = 0
    for _ in g:
        rows += 1
        if rows >= 20:
            break

    results = []
    # Try increasingly aggressive full-model modes. Each one with a fresh compile.
    for mode in ["default", "reduce-overhead", "max-autotune"]:
        attempt_mode(model, tok, silence_token, gen_fn, mode, results)
        # If we can't even do default, no point going further.
        if mode == "default" and not results[-1]["compiled"]:
            print("  default failed; trying isolated module compile as fallback", flush=True)
            break
        # Reset dynamo so next mode is a clean compile run
        try:
            torch._dynamo.reset()
        except Exception:
            pass

    if results and not results[0]["compiled"]:
        attempt_module_compile(model, tok, silence_token, gen_fn, results)

    (OUT_DIR / "compile_probe.json").write_text(json.dumps(results, indent=2))
    print(f"\n[done] wrote {OUT_DIR}/compile_probe.json")


if __name__ == "__main__":
    main()
