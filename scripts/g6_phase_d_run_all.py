#!/usr/bin/env python3
"""Phase D combined run — one model load.

Order:
  D4  recompiles: wrap forward with reduce-overhead, decode 30 rows at a small
      buffer, count recompiles via dynamo counters (+ TORCH_LOGS=recompiles to a
      file). MAKE-OR-BREAK gate. -> compile.json
  D5  identity: greedy (temperature=0) 5 prompts x 30 rows, compiled Phase D vs
      reference (Phase B decoder, eager). Assert bit-identical token sequences.
      -> identity.json
  D6  throughput-by-context: multi-stream packing-2 (ch 1,2 productive) at
      {256,1024,4096,8192}; per-context static buffer. tok/s, ms/row, peak mem,
      capture overhead. -> throughput_by_context.json

Uses local HF cache (NAS down). Idempotent: rewrites the JSONs each run.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ["HF_HOME"] = "/home/ianbarber/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/home/ianbarber/.cache/huggingface/hub"

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "JonasGeiping/stream-qwen3-8b"
OUT = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_d")
PATCH_D = OUT / "patched" / "stream_inference_phase_d.py"
PATCH_B = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_b/patched/stream_inference_phase_b.py")

PROMPTS = [
    "Write a Python function that reverses a linked list in place.",
    "Explain how a B-tree differs from a binary search tree.",
    "Refactor this code to use a context manager: open('f.txt'); read(); close().",
    "What is the time complexity of merge sort, and why?",
    "Sketch a unit test for a function that adds two integers.",
]


def load_patched(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def greedy_tokens(gen_fn, model, tok, silence, prompt, n_rows, extra):
    """Run greedy decode, collect the first n_rows non-prefill rows as token lists."""
    rows = []
    g = gen_fn(model, tok, prompt, silence, max_rows=n_rows + 5, warm_start=False,
               temperature=0.0, **extra)
    for row_idx, row, is_prefill in g:
        if is_prefill:
            continue
        rows.append(list(row))
        if len(rows) >= n_rows:
            break
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    snap = snapshot_download(MODEL_ID)
    if snap not in sys.path:
        sys.path.insert(0, snap)
    modD = load_patched(PATCH_D, "si_phase_d")
    modB = load_patched(PATCH_B, "si_phase_b")
    genD = modD.generate
    genB = modB.generate

    print("[load] loading model...", flush=True)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    silence = modD.detect_silence_token(tok)
    print(f"[load] done in {time.perf_counter()-t0:.1f}s silence={silence}", flush=True)

    import torch._dynamo as dynamo
    dynamo.config.suppress_errors = False
    dynamo.config.cache_size_limit = 64

    # ---- reference (eager Phase B) greedy tokens for identity (before compile) ----
    print("\n[ref] computing eager Phase B reference tokens (5 prompts x 30 rows)...", flush=True)
    ref = {}
    for i, p in enumerate(PROMPTS):
        ref[i] = greedy_tokens(genB, model, tok, silence, p, 30, {})
        print(f"[ref] prompt {i}: {len(ref[i])} rows", flush=True)

    # ---- compile wrap ----
    eager_forward = model.forward
    tc0 = time.perf_counter()
    model.forward = torch.compile(model.forward, mode="reduce-overhead",
                                  fullgraph=False, dynamic=False)
    compile_info = {"mode": "reduce-overhead",
                    "compile_wrap_seconds": time.perf_counter() - tc0}
    print(f"[compile] wrapped in {compile_info['compile_wrap_seconds']:.3f}s", flush=True)

    # ---- D4: recompiles over 30 decode rows (small buffer) ----
    dynamo.utils.counters.clear()
    print("\n[D4] decoding 30 rows (buffer=512) to count recompiles...", flush=True)
    extra512 = {"max_context_rows": 512}
    tw = time.perf_counter()
    first_row = None
    row_times = []
    prev = tw
    rows = 0
    for _ in genD(model, tok, PROMPTS[0], silence, max_rows=30, warm_start=False,
                  temperature=0.0, **extra512):
        torch.cuda.synchronize()
        now = time.perf_counter()
        if first_row is None:
            first_row = now - tw
        else:
            row_times.append(now - prev)
        prev = now
        rows += 1
        if rows >= 30:
            break
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - tw
    recompiles = dict(dynamo.utils.counters.get("recompiles", {}))
    n_recompiles = sum(recompiles.values()) if recompiles else 0
    # also count via the frames recorded
    frames = dict(dynamo.utils.counters.get("frames", {}))
    tail = sorted(row_times[-15:]) if len(row_times) >= 15 else sorted(row_times)
    steady = tail[len(tail) // 2] if tail else None
    compile_info.update({
        "verify_buffer_rows": 512, "rows_decoded": rows, "warmup_seconds": warmup_s,
        "first_row_seconds": first_row, "steady_state_row_seconds": steady,
        "row_times": row_times,
        "recompile_counters": recompiles,
        "total_recompiles": n_recompiles,
        "frames_counters": frames,
    })
    (OUT / "compile.json").write_text(json.dumps(compile_info, indent=2))
    print(f"[D4] total_recompiles={n_recompiles}  first_row={first_row:.2f}s  "
          f"steady={steady*1000 if steady else 'n/a'} ms/row", flush=True)
    print(f"[D4] recompile_counters={recompiles}", flush=True)

    # ---- D5: identity (compiled Phase D vs eager reference) ----
    print("\n[D5] identity: compiled Phase D vs eager Phase B (greedy, 30 rows)...", flush=True)
    identity = {"per_prompt": [], "all_match": True}
    for i, p in enumerate(PROMPTS):
        got = greedy_tokens(genD, model, tok, silence, p, 30, extra512)
        exp = ref[i]
        n = min(len(got), len(exp))
        match = got[:n] == exp[:n] and len(got) == len(exp)
        first_div = None
        for r in range(n):
            if got[r] != exp[r]:
                first_div = r
                break
        identity["per_prompt"].append({
            "prompt_idx": i, "match": bool(match),
            "ref_rows": len(exp), "got_rows": len(got),
            "first_divergence_row": first_div,
        })
        identity["all_match"] = identity["all_match"] and match
        print(f"[D5] prompt {i}: match={match} (ref={len(exp)} got={len(got)} "
              f"div={first_div})", flush=True)
    (OUT / "identity.json").write_text(json.dumps(identity, indent=2))
    print(f"[D5] ALL_MATCH={identity['all_match']}", flush=True)

    # ---- D6: throughput-by-context ----
    print("\n[D6] throughput-by-context (multi-stream packing-2)...", flush=True)
    contexts = [256, 1024, 4096, 8192]
    SHORT_ROWS = 80
    results = {}
    for n in contexts:
        mcr = int(n * 1.3) + 128
        extra_n = {"max_context_rows": mcr}
        print(f"[ctx n={n}] buffer_rows={mcr}: capture...", flush=True)
        torch.cuda.synchronize()
        tw = time.perf_counter()
        rr = 0
        for _ in genD(model, tok, PROMPTS[0], silence, max_rows=40, warm_start=False,
                      temperature=0.0, **extra_n):
            rr += 1
            if rr >= 40:
                break
        torch.cuda.synchronize()
        capture_s = time.perf_counter() - tw
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        prev = t0
        row_times = []
        prod = total = rr = 0
        for row_idx, row, is_prefill in genD(
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
        tail = sorted(row_times[-40:]) if len(row_times) >= 40 else sorted(row_times)
        steady_ms = (tail[len(tail) // 2] * 1000.0) if tail else None
        prod_frac = prod / rr if rr else 0.0
        derived_tps = (prod_frac / (steady_ms / 1000.0)) if steady_ms else 0.0
        rows_per_s = (1000.0 / steady_ms) if steady_ms else 0.0
        results[str(n)] = {
            "target_tokens": n, "buffer_rows": mcr, "capture_seconds": capture_s,
            "rows_timed": rr, "steady_state_ms_per_row": steady_ms,
            "rows_per_sec": rows_per_s,
            "productive_per_row_ch12": prod_frac,
            "derived_multi_tok_s": derived_tps,
            "peak_memory_gb": peak,
        }
        print(f"[ctx n={n}] steady {steady_ms:.1f} ms/row | rows/s {rows_per_s:.2f} | "
              f"prod/row {prod_frac:.2f} | derived {derived_tps:.2f} tok/s | "
              f"peak {peak:.2f} GB | capture {capture_s:.1f}s", flush=True)

    out = {"decoder": "phase_d_static_compile_reduce_overhead",
           "buffer_sizing": "per_context (target*1.3+128 rows)",
           "method": "steady-state per-row latency over short run; tok/s derived",
           "short_rows": SHORT_ROWS, "compile_info": compile_info,
           "by_context": results}
    (OUT / "throughput_by_context.json").write_text(json.dumps(out, indent=2))
    print("\n[done] wrote compile.json, identity.json, throughput_by_context.json", flush=True)


if __name__ == "__main__":
    main()
