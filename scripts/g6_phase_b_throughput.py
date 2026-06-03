#!/usr/bin/env python3
"""Phase B throughput (B6) — measure post-Phase-B speedup on the 5-prompt harness.

Modes:
  - phase_b_tensorize_only: B3 + B4 patches only (no torch.compile).
  - phase_b_compile_default: B3 + B4 + torch.compile(mode='default').
  - phase_b_compile_reduce_overhead: B3 + B4 + torch.compile(mode='reduce-overhead').

For each mode, runs single_stream and multi_stream (same as Phase A baseline_b).
Records compile telemetry: time to first forward, time to second forward (graph
reuse), graph-break log if any.

CLI:
    python g6_phase_b_throughput.py --mode {tensorize_only,compile_default,compile_reduce_overhead}

Writes runs/g6_phase_b/throughput_<mode>.json.
"""
from __future__ import annotations

import argparse
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
PATCH_B = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_b/patched/stream_inference_phase_b.py")

PROMPTS = [
    "Write a Python function that reverses a linked list in place.",
    "Explain how a B-tree differs from a binary search tree.",
    "Refactor this code to use a context manager: open('f.txt'); read(); close().",
    "What is the time complexity of merge sort, and why?",
    "Sketch a unit test for a function that adds two integers.",
]


def load_patched_module(patch_path: Path, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, patch_path)
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


def bench(model, tok, silence_token, gen_fn, channels_productive, label,
          n_output_tokens=256, prompts=PROMPTS):
    print(f"\n[{label}] productive channels: {channels_productive}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    per_prompt = []
    for i, p in enumerate(prompts):
        elapsed, rows, prod, total, sil_ch = measure_prompt(
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
            "silence_per_channel": sil_ch,
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["tensorize_only", "compile_default",
                             "compile_reduce_overhead", "compile_max_autotune_no_cg"])
    ap.add_argument("--out-suffix", default=None,
                    help="override output filename suffix")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    snap = snapshot_download(MODEL_ID)
    if snap not in sys.path:
        sys.path.insert(0, snap)

    patched = load_patched_module(PATCH_B, "si_phase_b")
    gen_fn = patched.generate
    silence_fn = patched.detect_silence_token

    print(f"[load] loading model (mode={args.mode})...", flush=True)
    t_load = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="auto")
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    silence_token = silence_fn(tok)
    print(f"[load] done in {time.perf_counter()-t_load:.1f}s", flush=True)

    compile_info = {"mode": args.mode}
    if args.mode != "tensorize_only":
        # Map mode -> torch.compile mode + flags.
        if args.mode == "compile_default":
            tc_mode = "default"
        elif args.mode == "compile_reduce_overhead":
            tc_mode = "reduce-overhead"
        elif args.mode == "compile_max_autotune_no_cg":
            tc_mode = "max-autotune-no-cudagraphs"

        # Enable verbose graph-break logging.
        import torch._dynamo as dynamo
        dynamo.config.suppress_errors = False
        dynamo.config.cache_size_limit = 128
        # Don't recompile on dynamic shapes — let dynamo handle it.
        # For our workload the only dynamic is cached_len in the mask.
        dynamo.config.automatic_dynamic_shapes = True

        print(f"[compile] wrapping model.forward with torch.compile(mode='{tc_mode}')", flush=True)
        t_c0 = time.perf_counter()
        # Compile only the inner model forward — the wrapper class drives the
        # python control flow and the dict-form attention_mask. The actual hot
        # path is `self.model(...)` (Qwen3Model.forward), which is what we want
        # CUDA graphs to capture.
        try:
            model.forward = torch.compile(model.forward, mode=tc_mode, fullgraph=False, dynamic=True)
        except Exception as e:
            compile_info["compile_wrap_error"] = repr(e)
            print(f"[compile] wrap FAILED: {e!r}", flush=True)
        compile_info["compile_wrap_seconds"] = time.perf_counter() - t_c0
        print(f"[compile] wrapped in {compile_info['compile_wrap_seconds']:.2f}s", flush=True)

    # Warmup — first row triggers compile (if enabled). Time it.
    print(f"[warmup] 50 rows (first row may include compile time)...", flush=True)
    torch.cuda.synchronize()
    t_warm0 = time.perf_counter()
    first_row_t = None
    rows = 0
    g = gen_fn(model, tok, PROMPTS[0], silence_token, max_rows=50,
               warm_start=False, temperature=0.0)
    for r in g:
        rows += 1
        if first_row_t is None:
            torch.cuda.synchronize()
            first_row_t = time.perf_counter() - t_warm0
        if rows >= 50:
            break
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - t_warm0
    print(f"[warmup] {rows} rows in {warmup_s:.2f}s; first-row {first_row_t:.2f}s",
          flush=True)
    compile_info["warmup_seconds"] = warmup_s
    compile_info["first_row_seconds"] = first_row_t

    single = bench(model, tok, silence_token, gen_fn,
                   channels_productive=[1], label="single_stream")
    multi = bench(model, tok, silence_token, gen_fn,
                  channels_productive=[1, 2], label="multi_stream")

    suffix = args.out_suffix or args.mode
    out = {
        "mode": args.mode,
        "compile_info": compile_info,
        "patch": str(PATCH_B),
        "snapshot_dir": snap,
        "warmup_rows": rows,
        "warmup_elapsed_s": warmup_s,
        "single_stream": single,
        "multi_stream": multi,
    }
    out_path = OUT_DIR / f"throughput_{suffix}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
