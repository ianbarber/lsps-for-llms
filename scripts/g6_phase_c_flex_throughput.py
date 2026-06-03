#!/usr/bin/env python3
"""Phase C Route 2 — F2 baseline repro + F5 throughput-by-context.

Modes:
  --mode baseline_repro : run Phase B (SDPA) generator, confirm ~4.95 tok/s multi.
  --mode flex           : run FlexAttention generator at multiple context lengths.

For flex mode, measures combined multi-stream packing-2 (channels [1,2]) at
productive-token targets 256, 1024, 4096, 8192. tok/s, per-row latency, peak mem.

Writes runs/g6_phase_c_flex/{baseline_repro.json | throughput_by_context.json}.
Idempotent.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "JonasGeiping/stream-qwen3-8b"
OUT_DIR = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_c_flex")
PATCH_B = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_b/patched/stream_inference_phase_b.py")
FLEX_DIR = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_c_flex/patched")

PROMPTS = [
    "Write a Python function that reverses a linked list in place.",
    "Explain how a B-tree differs from a binary search tree.",
    "Refactor this code to use a context manager: open('f.txt'); read(); close().",
    "What is the time complexity of merge sort, and why?",
    "Sketch a unit test for a function that adds two integers.",
]


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
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
    max_rows = max(n_output_tokens * 12, 500)
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
          n_output_tokens, prompts=PROMPTS):
    torch.cuda.reset_peak_memory_stats()
    per_prompt = []
    for i, p in enumerate(prompts):
        elapsed, rows, prod, total = measure_prompt(
            model, tok, silence_token, gen_fn, p, n_output_tokens, channels_productive)
        tps = prod / elapsed if elapsed > 0 else 0.0
        rps = rows / elapsed if elapsed > 0 else 0.0
        ms_per_row = 1000.0 * elapsed / rows if rows else 0.0
        per_prompt.append({
            "prompt_idx": i, "elapsed_s": elapsed, "rows": rows,
            "productive_tokens": prod, "total_tokens_emitted": total,
            "productive_tokens_per_sec": tps, "rows_per_sec": rps,
            "ms_per_row": ms_per_row,
        })
        print(f"[{label} n={n_output_tokens}] {i}: {elapsed:.2f}s {rows}r {prod}p "
              f"{tps:.2f} tok/s {ms_per_row:.1f} ms/row", flush=True)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    speeds = [m["productive_tokens_per_sec"] for m in per_prompt]
    mean = sum(speeds) / len(speeds)
    var = sum((s - mean) ** 2 for s in speeds) / max(len(speeds) - 1, 1)
    std = var ** 0.5
    ms = [m["ms_per_row"] for m in per_prompt]
    mean_ms = sum(ms) / len(ms)
    summary = {
        "label": label, "channels_productive": channels_productive,
        "n_output_tokens_target": n_output_tokens, "per_prompt": per_prompt,
        "mean_productive_tokens_per_sec": mean, "std_productive_tokens_per_sec": std,
        "mean_ms_per_row": mean_ms, "peak_memory_gb": peak_gb,
    }
    print(f"[{label} n={n_output_tokens}] MEAN {mean:.2f}+/-{std:.2f} tok/s | "
          f"{mean_ms:.1f} ms/row | peak {peak_gb:.2f} GB", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["baseline_repro", "flex"])
    ap.add_argument("--contexts", default="256,1024,4096,8192")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    snap = snapshot_download(MODEL_ID)
    if snap not in sys.path:
        sys.path.insert(0, snap)
    b = load_mod(PATCH_B, "si_phase_b")

    print(f"[load] loading model (mode={args.mode})...", flush=True)
    t_load = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    silence_token = b.detect_silence_token(tok)
    print(f"[load] done in {time.perf_counter()-t_load:.1f}s", flush=True)

    if args.mode == "baseline_repro":
        gen_fn = b.generate
        # warmup
        print("[warmup] 50 rows...", flush=True)
        for ri, _ in enumerate(b.generate(model, tok, PROMPTS[0], silence_token,
                                          max_rows=50, warm_start=False, temperature=0.0)):
            if ri >= 50:
                break
        single = bench(model, tok, silence_token, gen_fn, [1], "single_stream", 256)
        multi = bench(model, tok, silence_token, gen_fn, [1, 2], "multi_stream", 256)
        out = {
            "mode": "baseline_repro",
            "harness": "5 prompts x 256 productive tokens, greedy T=0, Phase B SDPA generator.",
            "single_stream": single, "multi_stream": multi,
            "phase_b_published_multi_tok_s": 4.95,
        }
        out_path = OUT_DIR / "baseline_repro.json"
    else:
        flex = load_mod(FLEX_DIR / "stream_inference_flex.py", "si_flex")
        import torch._dynamo as dynamo
        dynamo.config.cache_size_limit = 64
        flex.install_flex_attention(model)
        gen_fn = flex.generate
        # warmup compiles the flex kernel (first call expensive)
        print("[warmup] 60 rows (compiles flex kernel)...", flush=True)
        tw = time.perf_counter()
        first_row_t = None
        twf = time.perf_counter()
        cnt = 0
        for ri, _ in enumerate(flex.generate(model, tok, PROMPTS[0], silence_token,
                                             max_rows=60, warm_start=False, temperature=0.0)):
            if first_row_t is None:
                torch.cuda.synchronize()
                first_row_t = time.perf_counter() - twf
            cnt += 1
            if cnt >= 60:
                break
        torch.cuda.synchronize()
        warm_s = time.perf_counter() - tw
        print(f"[warmup] {cnt} rows in {warm_s:.1f}s; first-row {first_row_t:.1f}s", flush=True)

        contexts = [int(x) for x in args.contexts.split(",")]
        by_context = {}
        for n in contexts:
            by_context[str(n)] = bench(model, tok, silence_token, gen_fn, [1, 2],
                                       "multi_stream", n)
        out = {
            "mode": "flex",
            "harness": "FlexAttention BlockMask generator, multi-stream packing-2 "
                       "(channels [1,2]), greedy T=0. Context = productive-token target.",
            "warmup_seconds": warm_s, "first_row_seconds": first_row_t,
            "by_context": by_context,
        }
        out_path = OUT_DIR / "throughput_by_context.json"

    out_path.write_text(json.dumps(out, indent=2))
    print(f"[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
