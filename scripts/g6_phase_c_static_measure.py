#!/usr/bin/env python3
"""Phase C combined GPU measurement — one model load, one lock acquisition.

Does, in order:
  1. compile-verify: wrap model.forward with reduce-overhead, run 60 rows at a
     SMALL buffer (max_context_rows=512), record recompile count, first-row vs
     steady-state latency, graph-break counters. -> compile.json
  2. throughput-by-context: multi-stream packing-2 at productive-token targets
     {256,1024,4096,8192}, each with a per-context-sized static buffer
     (target*1.3+128 rows). Records tok/s, ms/row, peak mem. -> throughput_by_context.json

Assumes the caller holds the GPU lock. Reads HF_HOME from env.
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
PATCH_C = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_c_static/patched/stream_inference_static.py")

PROMPTS = [
    "Write a Python function that reverses a linked list in place.",
    "Explain how a B-tree differs from a binary search tree.",
    "Refactor this code to use a context manager: open('f.txt'); read(); close().",
    "What is the time complexity of merge sort, and why?",
    "Sketch a unit test for a function that adds two integers.",
]


def load_patched(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def measure_prompt(model, tok, silence, gen_fn, prompt, n_tokens, channels, extra):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    productive = total = rows_run = 0
    max_rows = max(n_tokens * 10, 500)
    g = gen_fn(model, tok, prompt, silence, max_rows=max_rows, warm_start=False,
               temperature=0.0, **extra)
    for row_idx, row, is_prefill in g:
        if is_prefill:
            continue
        rows_run += 1
        total += len(row)
        for c in channels:
            if c < len(row) and row[c] != silence:
                productive += 1
        if productive >= n_tokens:
            break
    torch.cuda.synchronize()
    return time.perf_counter() - t0, rows_run, productive, total


def bench(model, tok, silence, gen_fn, channels, label, n_tokens, extra, n_prompts=5):
    print(f"\n[{label}] channels={channels} target={n_tokens} n_prompts={n_prompts}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    per_prompt = []
    for i, p in enumerate(PROMPTS[:n_prompts]):
        el, rows, prod, total = measure_prompt(
            model, tok, silence, gen_fn, p, n_tokens, channels, extra)
        per_prompt.append({
            "prompt_idx": i, "elapsed_s": el, "rows": rows,
            "productive_tokens": prod, "total_tokens_emitted": total,
            "productive_tokens_per_sec": prod / el if el else 0.0,
            "rows_per_sec": rows / el if el else 0.0,
            "ms_per_row": 1000.0 * el / rows if rows else 0.0,
        })
        print(f"[{label}] {i}: {el:.2f}s {rows} rows {prod} prod "
              f"{prod/el if el else 0:.2f} tok/s {1000*el/max(rows,1):.1f} ms/row", flush=True)
    peak = torch.cuda.max_memory_allocated() / 1e9
    sp = [m["productive_tokens_per_sec"] for m in per_prompt]
    rp = [m["rows_per_sec"] for m in per_prompt]
    ms = [m["ms_per_row"] for m in per_prompt]
    mean = sum(sp) / len(sp)
    var = sum((s - mean) ** 2 for s in sp) / max(len(sp) - 1, 1)
    summ = {
        "label": label, "channels_productive": channels, "n_output_tokens_target": n_tokens,
        "per_prompt": per_prompt, "mean_productive_tokens_per_sec": mean,
        "std_productive_tokens_per_sec": var ** 0.5,
        "mean_rows_per_sec": sum(rp) / len(rp), "mean_ms_per_row": sum(ms) / len(ms),
        "peak_memory_gb": peak,
    }
    print(f"[{label}] MEAN {mean:.2f} tok/s | {summ['mean_rows_per_sec']:.2f} rows/s | "
          f"{summ['mean_ms_per_row']:.1f} ms/row | peak {peak:.2f} GB", flush=True)
    return summ


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    snap = snapshot_download(MODEL_ID)
    if snap not in sys.path:
        sys.path.insert(0, snap)
    mod = load_patched(PATCH_C, "si_static")
    gen_fn = mod.generate

    print("[load] loading model...", flush=True)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    silence = mod.detect_silence_token(tok)
    print(f"[load] done in {time.perf_counter()-t0:.1f}s silence={silence}", flush=True)

    # ── wrap with reduce-overhead (CUDA graphs) ─────────────────────────────
    import torch._dynamo as dynamo
    dynamo.config.suppress_errors = False
    dynamo.config.cache_size_limit = 64
    tc0 = time.perf_counter()
    model.forward = torch.compile(model.forward, mode="reduce-overhead",
                                  fullgraph=False, dynamic=False)
    compile_info = {"mode": "reduce-overhead",
                    "compile_wrap_seconds": time.perf_counter() - tc0}
    print(f"[compile] wrapped in {compile_info['compile_wrap_seconds']:.3f}s", flush=True)

    # ── 1. compile-verify at small buffer (512 rows) ────────────────────────
    extra512 = {"max_context_rows": 512}
    torch.cuda.synchronize()
    tw = time.perf_counter()
    first_row = None
    row_times = []
    prev = tw
    rows = 0
    for _ in gen_fn(model, tok, PROMPTS[0], silence, max_rows=60, warm_start=False,
                    temperature=0.0, **extra512):
        torch.cuda.synchronize()
        now = time.perf_counter()
        if first_row is None:
            first_row = now - tw
        else:
            row_times.append(now - prev)
        prev = now
        rows += 1
        if rows >= 60:
            break
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - tw
    tail = sorted(row_times[-30:]) if len(row_times) >= 30 else sorted(row_times)
    steady = tail[len(tail) // 2] if tail else None
    try:
        recompiles = dict(dynamo.utils.counters.get("recompiles", {}))
    except Exception:
        recompiles = {}
    try:
        gb = dict(dynamo.utils.counters.get("graph_break", {}))
    except Exception:
        gb = {}
    compile_info.update({
        "verify_buffer_rows": 512, "warmup_seconds": warmup_s,
        "first_row_seconds": first_row, "steady_state_row_seconds": steady,
        "row_times_tail": row_times[-30:],
        "unique_graph_breaks": len(gb), "graph_break_counters": gb,
        "recompile_counters": recompiles,
    })
    (OUT_DIR / "compile.json").write_text(json.dumps(compile_info, indent=2))
    print(f"[compile] first_row={first_row:.2f}s steady={steady} "
          f"graph_breaks={len(gb)}", flush=True)
    print(f"[compile] steady-state row latency: "
          f"{steady*1000 if steady else 'n/a'} ms", flush=True)

    # ── 2. throughput-by-context (per-context buffer) ───────────────────────
    # The static buffer size sets the K-length SDPA attends over (the whole
    # point of C6: does full-context attention cost dominate?). We measure
    # steady-state per-row latency at each buffer size over a short run, which
    # directly gives the cost-vs-context curve. tok/s is then derived from the
    # measured productive-fraction × (1 / ms_per_row). This avoids running full
    # multi-hour trajectories while still characterizing the curve faithfully:
    # at a given buffer size the per-row cost is constant (static shape), so a
    # short run's steady-state == the full trajectory's per-row cost.
    contexts = [256, 1024, 4096, 8192]
    SHORT_ROWS = 80          # rows to time per buffer size (after warmup)
    results = {}
    for n in contexts:
        mcr = int(n * 1.3) + 128   # buffer holds the full target trajectory
        extra_n = {"max_context_rows": mcr}
        # warmup/capture
        print(f"[ctx n={n}] buffer_rows={mcr}: capture...", flush=True)
        torch.cuda.synchronize()
        tw = time.perf_counter()
        rr = 0
        for _ in gen_fn(model, tok, PROMPTS[0], silence, max_rows=60, warm_start=False,
                        temperature=0.0, **extra_n):
            rr += 1
            if rr >= 60:
                break
        torch.cuda.synchronize()
        capture_s = time.perf_counter() - tw
        # timed steady-state run
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        prev = t0
        row_times = []
        prod = total = rr = 0
        for row_idx, row, is_prefill in gen_fn(
                model, tok, PROMPTS[1], silence, max_rows=SHORT_ROWS,
                warm_start=False, temperature=0.0, **extra_n):
            if is_prefill:
                continue
            torch.cuda.synchronize()
            now = time.perf_counter()
            row_times.append(now - prev)
            prev = now
            rr += 1
            total += len(row)
            for c in (1, 2):
                if row[c] != silence:
                    prod += 1
            if rr >= SHORT_ROWS:
                break
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e9
        # steady-state = median of last 40 row deltas (drops any warmup tail)
        tail = sorted(row_times[-40:]) if len(row_times) >= 40 else sorted(row_times)
        steady_ms = (tail[len(tail) // 2] * 1000.0) if tail else None
        prod_frac = prod / rr if rr else 0.0   # productive tokens per row (ch 1,2)
        derived_tps = (prod_frac / (steady_ms / 1000.0)) if steady_ms else 0.0
        results[str(n)] = {
            "target_tokens": n, "buffer_rows": mcr, "capture_seconds": capture_s,
            "rows_timed": rr, "steady_state_ms_per_row": steady_ms,
            "productive_per_row_ch12": prod_frac,
            "derived_multi_tok_s": derived_tps,
            "peak_memory_gb": peak,
        }
        print(f"[ctx n={n}] steady {steady_ms:.1f} ms/row | "
              f"prod/row {prod_frac:.2f} | derived {derived_tps:.2f} tok/s | "
              f"peak {peak:.2f} GB", flush=True)

    out = {"decoder": "static_compile_reduce_overhead",
           "buffer_sizing": "per_context (target*1.3+128 rows)",
           "method": "steady-state per-row latency over short run; tok/s derived",
           "short_rows": SHORT_ROWS, "compile_info": compile_info,
           "by_context": results}
    (OUT_DIR / "throughput_by_context.json").write_text(json.dumps(out, indent=2))
    print("[done] wrote compile.json + throughput_by_context.json", flush=True)


if __name__ == "__main__":
    main()
