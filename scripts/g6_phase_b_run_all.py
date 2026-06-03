#!/usr/bin/env python3
"""Phase B end-to-end runner: identity (B7) + tensorize-only throughput (B6)
+ torch.compile throughput (B5+B6), all in one Python process so we only pay
the 2-minute model-load cost once.

Sub-steps (gated by CLI flags so we can skip / resume):
  --do-identity         : run B7 identity check (Phase A vs Phase B greedy)
  --do-tensorize        : run B3+B4 throughput (no compile)
  --do-compile          : run B5 torch.compile probe + throughput
  --compile-mode MODE   : torch.compile mode (default | reduce-overhead | max-autotune-no-cudagraphs)

Outputs:
  runs/g6_phase_b/identity.json
  runs/g6_phase_b/throughput_tensorize_only.json
  runs/g6_phase_b/throughput_compile_<mode>.json
  runs/g6_phase_b/compile.json
"""
from __future__ import annotations

import argparse
import importlib.util
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
    R = min(len(a_rows), len(b_rows))
    for r in range(R):
        ar, br = a_rows[r], b_rows[r]
        C = min(len(ar), len(br))
        for c in range(C):
            if ar[c] != br[c]:
                return (r, c, int(ar[c]), int(br[c]))
        if len(ar) != len(br):
            return (r, -1, len(ar), len(br))
    if len(a_rows) != len(b_rows):
        return (R, -1, len(a_rows), len(b_rows))
    return None


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


def warmup(gen_fn, model, tok, silence_token, n=50, label="warmup"):
    print(f"[{label}] {n} rows...", flush=True)
    torch.cuda.synchronize()
    t_warm0 = time.perf_counter()
    first_row_t = None
    rows = 0
    g = gen_fn(model, tok, PROMPTS[0], silence_token, max_rows=n,
               warm_start=False, temperature=0.0)
    for r in g:
        rows += 1
        if first_row_t is None:
            torch.cuda.synchronize()
            first_row_t = time.perf_counter() - t_warm0
        if rows >= n:
            break
    torch.cuda.synchronize()
    warm_s = time.perf_counter() - t_warm0
    print(f"[{label}] {rows} rows in {warm_s:.2f}s; first-row {first_row_t:.2f}s",
          flush=True)
    return rows, warm_s, first_row_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--do-identity", action="store_true")
    ap.add_argument("--do-tensorize", action="store_true")
    ap.add_argument("--do-compile", action="store_true")
    ap.add_argument("--compile-mode", default="reduce-overhead",
                    choices=["default", "reduce-overhead", "max-autotune-no-cudagraphs"])
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    snap = snapshot_download(MODEL_ID)
    if snap not in sys.path:
        sys.path.insert(0, snap)

    mod_a = load_patched_module(PATCH_A, "si_phase_a")
    mod_b = load_patched_module(PATCH_B, "si_phase_b")

    print("[load] loading model...", flush=True)
    t_load = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="auto")
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    silence_token = mod_a.detect_silence_token(tok)
    print(f"[load] done in {time.perf_counter()-t_load:.1f}s; silence={silence_token}",
          flush=True)

    # ---------------- B7: identity ----------------
    if args.do_identity:
        print("\n[step] identity check (B7)", flush=True)
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
        out["all_identical"] = all(p["verdict"] == "identical" for p in out["per_prompt"])
        (OUT_DIR / "identity.json").write_text(json.dumps(out, indent=2))
        print(f"[identity] all_identical={out['all_identical']}", flush=True)
        if not out["all_identical"]:
            print("[ERROR] identity broken — aborting before throughput runs.",
                  flush=True)
            return

    # ---------------- B6: tensorize-only throughput ----------------
    if args.do_tensorize:
        print("\n[step] tensorize-only throughput (B6)", flush=True)
        _, warm_s, first_t = warmup(mod_b.generate, model, tok, silence_token, 50)
        single = bench(model, tok, silence_token, mod_b.generate,
                       channels_productive=[1], label="single_stream")
        multi = bench(model, tok, silence_token, mod_b.generate,
                      channels_productive=[1, 2], label="multi_stream")
        out = {
            "mode": "tensorize_only",
            "patch": str(PATCH_B),
            "warmup_elapsed_s": warm_s,
            "warmup_first_row_s": first_t,
            "single_stream": single,
            "multi_stream": multi,
        }
        (OUT_DIR / "throughput_tensorize_only.json").write_text(json.dumps(out, indent=2))
        print(f"[done] wrote {OUT_DIR}/throughput_tensorize_only.json", flush=True)

    # ---------------- B5: torch.compile ----------------
    if args.do_compile:
        print(f"\n[step] torch.compile (B5) mode={args.compile_mode}", flush=True)
        compile_info = {"mode": args.compile_mode}
        import torch._dynamo as dynamo
        dynamo.config.suppress_errors = False
        dynamo.config.cache_size_limit = 1024
        dynamo.config.automatic_dynamic_shapes = True
        # Reduce inductor compile parallelism to avoid swamping CPU/memory on
        # rapid recompiles. Default is 20 workers.
        try:
            import torch._inductor.config as ic
            ic.compile_threads = 4
        except Exception:
            pass
        # Reset to drop any earlier dynamo state from --do-tensorize.
        try:
            dynamo.reset()
        except Exception:
            pass

        # Log recompiles loudly so we know if we're churning.
        import logging
        logging.getLogger("torch._dynamo.guards").setLevel(logging.INFO)
        # `torch._dynamo.config.report_guard_failures = True` is the
        # programmatic equivalent.
        try:
            dynamo.config.report_guard_failures = True
        except Exception:
            pass

        t_c0 = time.perf_counter()
        try:
            model.forward = torch.compile(
                model.forward, mode=args.compile_mode, fullgraph=False, dynamic=True)
            compile_info["wrap_seconds"] = time.perf_counter() - t_c0
            print(f"[compile] wrapped in {compile_info['wrap_seconds']:.2f}s",
                  flush=True)
        except Exception as e:
            compile_info["wrap_error"] = repr(e) + "\n" + traceback.format_exc()
            print(f"[compile] wrap FAILED: {e!r}", flush=True)
            (OUT_DIR / f"compile_{args.compile_mode}.json").write_text(json.dumps(compile_info, indent=2))
            return

        # Tiny compile probe — 2-row decode to measure first-call latency.
        # Persist compile.json *before* the long warmup so we have telemetry
        # even if warmup OOMs.
        print("[compile] probe 2-row decode...", flush=True)
        torch.cuda.synchronize()
        t_p0 = time.perf_counter()
        rows = 0
        for _ in mod_b.generate(model, tok, PROMPTS[0], silence_token,
                                 max_rows=2, warm_start=False, temperature=0.0):
            rows += 1
            if rows >= 2:
                break
        torch.cuda.synchronize()
        compile_info["probe_2row_seconds"] = time.perf_counter() - t_p0
        print(f"[compile] probe done in {compile_info['probe_2row_seconds']:.2f}s",
              flush=True)
        (OUT_DIR / f"compile_{args.compile_mode.replace('-','_')}_probe.json").write_text(
            json.dumps(compile_info, indent=2))

        # 3-row second probe — should be MUCH faster than first probe if the
        # graph is reused (dynamic shapes) and not recompiled per row.
        torch.cuda.synchronize()
        t_p1 = time.perf_counter()
        rows = 0
        for _ in mod_b.generate(model, tok, PROMPTS[1], silence_token,
                                 max_rows=3, warm_start=False, temperature=0.0):
            rows += 1
            if rows >= 3:
                break
        torch.cuda.synchronize()
        compile_info["probe_3row_seconds_p2"] = time.perf_counter() - t_p1
        print(f"[compile] second probe (3 rows) done in {compile_info['probe_3row_seconds_p2']:.2f}s",
              flush=True)

        # Short warmup — 10 rows only, to confirm steady-state row time.
        print(f"[compile-warmup] 10 rows...", flush=True)
        torch.cuda.synchronize()
        t_w0 = time.perf_counter()
        per_row_times = []
        rows = 0
        prev_t = t_w0
        for _ in mod_b.generate(model, tok, PROMPTS[2], silence_token,
                                 max_rows=10, warm_start=False, temperature=0.0):
            rows += 1
            torch.cuda.synchronize()
            now = time.perf_counter()
            per_row_times.append(now - prev_t)
            prev_t = now
            if rows >= 10:
                break
        warm_s = time.perf_counter() - t_w0
        first_t = per_row_times[0] if per_row_times else None
        compile_info["warmup_seconds"] = warm_s
        compile_info["warmup_first_row_seconds"] = first_t
        compile_info["warmup_per_row_seconds"] = per_row_times
        print(f"[compile-warmup] 10 rows in {warm_s:.2f}s; per-row={['%.2f'%t for t in per_row_times]}",
              flush=True)
        (OUT_DIR / f"compile_{args.compile_mode.replace('-','_')}.json").write_text(
            json.dumps(compile_info, indent=2))
        print(f"[compile] telemetry -> {OUT_DIR}/compile_{args.compile_mode.replace('-','_')}.json", flush=True)

        # Decide whether to proceed to full bench based on steady-state row time.
        # If per-row > 1s at row 10, we're probably recompiling — skip the
        # full 256-token bench (it would take an hour).
        if per_row_times and per_row_times[-1] > 1.0:
            print(f"[compile] steady-state row time {per_row_times[-1]:.2f}s is "
                  f"WORSE than tensorize-only (~0.37s); skipping full bench.",
                  flush=True)
            return

        single = bench(model, tok, silence_token, mod_b.generate,
                       channels_productive=[1], label="single_stream")
        multi = bench(model, tok, silence_token, mod_b.generate,
                      channels_productive=[1, 2], label="multi_stream")
        out = {
            "mode": "compile_" + args.compile_mode,
            "compile_info": compile_info,
            "patch": str(PATCH_B),
            "single_stream": single,
            "multi_stream": multi,
        }
        out_path = OUT_DIR / f"throughput_compile_{args.compile_mode.replace('-', '_')}.json"
        out_path.write_text(json.dumps(out, indent=2))
        print(f"[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
