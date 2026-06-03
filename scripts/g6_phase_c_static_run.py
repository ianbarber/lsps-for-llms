#!/usr/bin/env python3
"""Phase C Route 1 — baseline repro (C2), CUDA-graph compile (C5), and
throughput-by-context (C6) for the static-shape decoder.

Subcommands (via --task):
  baseline   : reproduce Phase B (4.95 multi tok/s) with the Phase B decoder.
  static     : static decoder, NO compile (sanity / overhead measurement).
  compile    : static decoder + torch.compile(mode='reduce-overhead'); record
               compile time, graph-break info, first-call vs steady-state row
               latency. Writes compile.json.
  throughput : static decoder + compile, multi-stream packing-2, at multiple
               productive-token context lengths {256,1024,4096,8192}. Writes
               throughput_by_context.json.

All GPU timing must run under the cooperative lock (handled by the caller's
shell wrapper). This script assumes it already holds the lock.
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


def load_patched(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def measure_prompt(model, tok, silence, gen_fn, prompt, n_output_tokens,
                   channels_productive, extra):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    productive = 0
    total = 0
    rows_run = 0
    max_rows = max(n_output_tokens * 10, 500)
    g = gen_fn(model, tok, prompt, silence, max_rows=max_rows, warm_start=False,
               temperature=0.0, **extra)
    for row_idx, row, is_prefill in g:
        if is_prefill:
            continue
        rows_run += 1
        total += len(row)
        for c in channels_productive:
            if c < len(row) and row[c] != silence:
                productive += 1
        if productive >= n_output_tokens:
            break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed, rows_run, productive, total


def bench(model, tok, silence, gen_fn, channels, label, n_output_tokens, extra):
    print(f"\n[{label}] channels={channels} target={n_output_tokens}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    per_prompt = []
    for i, p in enumerate(PROMPTS):
        elapsed, rows, prod, total = measure_prompt(
            model, tok, silence, gen_fn, p, n_output_tokens, channels, extra)
        tps = prod / elapsed if elapsed > 0 else 0.0
        rps = rows / elapsed if elapsed > 0 else 0.0
        per_prompt.append({
            "prompt_idx": i, "elapsed_s": elapsed, "rows": rows,
            "productive_tokens": prod, "total_tokens_emitted": total,
            "productive_tokens_per_sec": tps, "rows_per_sec": rps,
            "ms_per_row": 1000.0 * elapsed / rows if rows else 0.0,
        })
        print(f"[{label}] {i}: {elapsed:.2f}s {rows} rows {prod} prod "
              f"{tps:.2f} tok/s {rps:.2f} rows/s {1000*elapsed/max(rows,1):.1f} ms/row",
              flush=True)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    speeds = [m["productive_tokens_per_sec"] for m in per_prompt]
    rps_l = [m["rows_per_sec"] for m in per_prompt]
    mspr = [m["ms_per_row"] for m in per_prompt]
    mean = sum(speeds) / len(speeds)
    var = sum((s - mean) ** 2 for s in speeds) / max(len(speeds) - 1, 1)
    summary = {
        "label": label, "channels_productive": channels,
        "n_output_tokens_target": n_output_tokens, "per_prompt": per_prompt,
        "mean_productive_tokens_per_sec": mean, "std_productive_tokens_per_sec": var ** 0.5,
        "mean_rows_per_sec": sum(rps_l) / len(rps_l),
        "mean_ms_per_row": sum(mspr) / len(mspr),
        "peak_memory_gb": peak_gb,
    }
    print(f"[{label}] MEAN {mean:.2f} tok/s | {summary['mean_rows_per_sec']:.2f} rows/s | "
          f"{summary['mean_ms_per_row']:.1f} ms/row | peak {peak_gb:.2f} GB", flush=True)
    return summary


def load_model(tok_only=False):
    snap = snapshot_download(MODEL_ID)
    if snap not in sys.path:
        sys.path.insert(0, snap)
    print("[load] loading model...", flush=True)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"[load] done in {time.perf_counter()-t0:.1f}s", flush=True)
    return model, tok, snap


def setup_compile(model):
    """Wrap model.forward with reduce-overhead (CUDA graphs). Return compile info."""
    import torch._dynamo as dynamo
    dynamo.config.suppress_errors = False
    dynamo.config.cache_size_limit = 64
    # Static shapes => no need for dynamic; force static so inductor specializes.
    info = {"mode": "reduce-overhead"}
    t0 = time.perf_counter()
    model.forward = torch.compile(model.forward, mode="reduce-overhead",
                                  fullgraph=False, dynamic=False)
    info["compile_wrap_seconds"] = time.perf_counter() - t0
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["baseline", "static", "compile", "throughput"])
    ap.add_argument("--max-context-rows", type=int, default=8192)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.task == "baseline":
        mod = load_patched(PATCH_B, "si_phase_b")
        model, tok, snap = load_model()
        silence = mod.detect_silence_token(tok)
        gen_fn = mod.generate
        extra = {}
        # warmup
        list_warm(model, tok, silence, gen_fn, extra)
        single = bench(model, tok, silence, gen_fn, [1], "single_stream", 256, extra)
        multi = bench(model, tok, silence, gen_fn, [1, 2], "multi_stream", 256, extra)
        out = {"decoder": "phase_b", "single_stream": single, "multi_stream": multi}
        (OUT_DIR / "baseline_repro.json").write_text(json.dumps(out, indent=2))
        print(f"[done] wrote baseline_repro.json", flush=True)
        return

    # static / compile / throughput all use the static decoder.
    mod = load_patched(PATCH_C, "si_static")
    model, tok, snap = load_model()
    silence = mod.detect_silence_token(tok)
    gen_fn = mod.generate
    mcr = args.max_context_rows
    extra = {"max_context_rows": mcr}

    compile_info = {}
    if args.task in ("compile", "throughput"):
        compile_info = setup_compile(model)
        print(f"[compile] wrapped in {compile_info['compile_wrap_seconds']:.3f}s", flush=True)

    if args.task == "static":
        list_warm(model, tok, silence, gen_fn, extra)
        single = bench(model, tok, silence, gen_fn, [1], "single_stream", 256, extra)
        multi = bench(model, tok, silence, gen_fn, [1, 2], "multi_stream", 256, extra)
        out = {"decoder": "static_no_compile", "max_context_rows": mcr,
               "single_stream": single, "multi_stream": multi}
        (OUT_DIR / "static_no_compile.json").write_text(json.dumps(out, indent=2))
        print(f"[done] wrote static_no_compile.json", flush=True)
        return

    if args.task == "compile":
        # Time first-call (compile + capture) and steady-state row latency.
        import torch._dynamo as dynamo
        try:
            explain = dynamo.explain(model.forward)
        except Exception:
            explain = None
        torch.cuda.synchronize()
        t_warm0 = time.perf_counter()
        first_row_t = None
        row_times = []
        prev = t_warm0
        rows = 0
        g = gen_fn(model, tok, PROMPTS[0], silence, max_rows=60, warm_start=False,
                   temperature=0.0, **extra)
        for r in g:
            torch.cuda.synchronize()
            now = time.perf_counter()
            if first_row_t is None:
                first_row_t = now - t_warm0
            else:
                row_times.append(now - prev)
            prev = now
            rows += 1
            if rows >= 60:
                break
        torch.cuda.synchronize()
        warmup_s = time.perf_counter() - t_warm0
        # steady-state = median of last 30 row deltas
        tail = sorted(row_times[-30:]) if len(row_times) >= 30 else sorted(row_times)
        steady = tail[len(tail) // 2] if tail else None
        compile_info.update({
            "warmup_seconds": warmup_s,
            "first_row_seconds": first_row_t,
            "steady_state_row_seconds": steady,
            "row_times_tail": row_times[-30:],
            "max_context_rows": mcr,
        })
        # graph-break / recompile diagnostics
        try:
            stats = torch._dynamo.utils.compile_times(repr_type="csv")
            compile_info["dynamo_compile_times_csv"] = stats
        except Exception as e:
            compile_info["dynamo_compile_times_err"] = repr(e)
        try:
            compile_info["frame_count"] = torch._dynamo.utils.counters.get("frames", {})
            compile_info["graph_break_counters"] = dict(
                torch._dynamo.utils.counters.get("graph_break", {}))
            compile_info["unique_graph_breaks"] = len(
                torch._dynamo.utils.counters.get("graph_break", {}))
        except Exception as e:
            compile_info["counters_err"] = repr(e)
        (OUT_DIR / "compile.json").write_text(json.dumps(compile_info, indent=2))
        print(f"[compile] first_row={first_row_t:.2f}s steady={steady}", flush=True)
        print(f"[compile] graph_breaks={compile_info.get('unique_graph_breaks')}", flush=True)
        print("[done] wrote compile.json", flush=True)
        return

    if args.task == "throughput":
        # The static buffer (MAX_CONTEXT_ROWS) sets the attention K-length the
        # CUDA graph attends over. To characterize cost-vs-context honestly, we
        # size the buffer PER context: each target gets its own buffer just big
        # enough to hold the trajectory (target*1.3 rows headroom), recapturing
        # the decode graph once per context (4 captures total, NOT per-row).
        contexts = [256, 1024, 4096, 8192]
        results = {}
        warm_info = {}
        for n in contexts:
            mcr_n = int(n * 1.3) + 128  # headroom for silent rows + prefill
            extra_n = {"max_context_rows": mcr_n}
            # Warmup at this buffer size to trigger compile/capture for this shape.
            print(f"[warmup] n={n} buffer_rows={mcr_n}: capture...", flush=True)
            torch.cuda.synchronize()
            tw = time.perf_counter()
            rr = 0
            for _ in gen_fn(model, tok, PROMPTS[0], silence, max_rows=60,
                            warm_start=False, temperature=0.0, **extra_n):
                rr += 1
                if rr >= 60:
                    break
            torch.cuda.synchronize()
            warm_info[str(n)] = {"buffer_rows": mcr_n,
                                 "warmup_seconds": time.perf_counter() - tw}
            print(f"[warmup] n={n} {rr} rows in "
                  f"{warm_info[str(n)]['warmup_seconds']:.1f}s", flush=True)
            res = bench(model, tok, silence, gen_fn, [1, 2], f"multi_n{n}", n, extra_n)
            res["buffer_rows"] = mcr_n
            results[str(n)] = res
        out = {"decoder": "static_compile_reduce_overhead",
               "buffer_sizing": "per_context (target*1.3+128 rows)",
               "compile_info": compile_info, "warmup_per_context": warm_info,
               "by_context": results}
        (OUT_DIR / "throughput_by_context.json").write_text(json.dumps(out, indent=2))
        print("[done] wrote throughput_by_context.json", flush=True)
        return


def list_warm(model, tok, silence, gen_fn, extra):
    print("[warmup] 50 rows...", flush=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    rows = 0
    for _ in gen_fn(model, tok, PROMPTS[0], silence, max_rows=50, warm_start=False,
                    temperature=0.0, **extra):
        rows += 1
        if rows >= 50:
            break
    torch.cuda.synchronize()
    print(f"[warmup] {rows} rows in {time.perf_counter()-t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
