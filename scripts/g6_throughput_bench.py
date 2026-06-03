#!/usr/bin/env python3
"""G6 throughput micro-benchmark for stream-qwen3-8b on GB10.

Measures:
  1. Single-stream throughput  — count Output-channel tokens only as productive.
  2. Multi-stream throughput   — count Output + Analytical (packing factor 2)
                                 as productive.
  3. gen.send() viability      — inject a token into the User stream mid-decode.

Architectural note: the stream-qwen3 model produces all 10 channel tokens
per forward pass via a block-causal attention mask. Per-row latency is
*identical* regardless of how many channels you treat as productive — the
"packing factor" gain is therefore amortisation of one decode step across
multiple useful channels, not a parallel-decode trick.

Idempotent: writes JSON measurements + a text gen.send check.

Usage:
    python scripts/g6_throughput_bench.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "JonasGeiping/stream-qwen3-8b"
OUT_DIR = Path("/home/ianbarber/Projects/Streams/runs/g6_throughput")


PROMPTS = [
    "Write a Python function that reverses a linked list in place.",
    "Explain how a B-tree differs from a binary search tree.",
    "Refactor this code to use a context manager: open('f.txt'); read(); close().",
    "What is the time complexity of merge sort, and why?",
    "Sketch a unit test for a function that adds two integers.",
]


def setup_paths():
    """Add the snapshot dir to sys.path so we can import stream_inference."""
    snap = snapshot_download(MODEL_ID)
    if snap not in sys.path:
        sys.path.insert(0, snap)
    return snap


def load_model_and_tokenizer():
    """Load model + tokenizer with the bf16 / device_map='auto' settings the plan calls for."""
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


def run_warmup(model, tok, silence_token, gen_fn, n_rows: int = 50):
    """Burn n_rows of generation to amortise kernel autotune + lazy init."""
    print(f"[warmup] {n_rows} rows...", flush=True)
    rows = []
    g = gen_fn(model, tok, PROMPTS[0], silence_token, max_rows=n_rows,
               warm_start=False, temperature=0.0)
    for r in g:
        rows.append(r)
        if len(rows) >= n_rows:
            break
    print(f"[warmup] consumed {len(rows)} rows", flush=True)


def measure_prompt(model, tok, silence_token, gen_fn, prompt: str,
                   n_output_tokens: int, channels_productive: list[int],
                   warm_start: bool = False):
    """Run generation for one prompt and stop when we've collected n_output_tokens
    *non-silence* tokens across `channels_productive`.

    Returns (elapsed_seconds, rows_run, productive_tokens, total_tokens_emitted).

    The model emits C=10 tokens/row regardless; productive_tokens counts only
    non-silence tokens in the requested channels.
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    productive = 0
    total = 0
    rows_run = 0
    # Hard ceiling so we can't hang. 10x headroom over needed rows.
    max_rows = max(n_output_tokens * 10, 500)

    g = gen_fn(model, tok, prompt, silence_token, max_rows=max_rows,
               warm_start=warm_start, temperature=0.0)
    for row_idx, row, is_prefill in g:
        if is_prefill:
            continue
        rows_run += 1
        for c in range(len(row)):
            total += 1
        for c in channels_productive:
            if c < len(row) and row[c] != silence_token:
                productive += 1
        if productive >= n_output_tokens:
            break

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed, rows_run, productive, total


def bench(model, tok, silence_token, gen_fn, channels_productive: list[int],
          label: str, n_output_tokens: int = 256, prompts=PROMPTS,
          per_meas_timeout_s: float = 1800.0):
    """Run the per-prompt benchmark loop. Returns a dict of measurements."""
    print(f"\n[{label}] {len(prompts)} prompts x {n_output_tokens} productive tokens", flush=True)
    print(f"[{label}] productive channels: {channels_productive}", flush=True)
    torch.cuda.reset_peak_memory_stats()

    per_prompt = []
    t_start = time.perf_counter()
    for i, p in enumerate(prompts):
        if time.perf_counter() - t_start > per_meas_timeout_s:
            print(f"[{label}] TIMEOUT hit, stopping", flush=True)
            break
        elapsed, rows, prod, total = measure_prompt(
            model, tok, silence_token, gen_fn, p, n_output_tokens,
            channels_productive,
        )
        # tokens/sec — productive channel tokens; rows/sec also recorded
        toks_per_sec = prod / elapsed if elapsed > 0 else 0.0
        rows_per_sec = rows / elapsed if elapsed > 0 else 0.0
        # all-channels tokens/sec (10 channels per row, but we count only the productive set above)
        per_prompt.append({
            "prompt_idx": i,
            "elapsed_s": elapsed,
            "rows": rows,
            "productive_tokens": prod,
            "total_tokens_emitted": total,
            "productive_tokens_per_sec": toks_per_sec,
            "rows_per_sec": rows_per_sec,
        })
        print(f"[{label}] prompt {i}: {elapsed:.2f}s | {rows} rows | "
              f"{prod} prod toks | {toks_per_sec:.2f} prod-tok/s | "
              f"{rows_per_sec:.2f} rows/s",
              flush=True)

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    # aggregate stats
    if per_prompt:
        speeds = [m["productive_tokens_per_sec"] for m in per_prompt]
        rows_speeds = [m["rows_per_sec"] for m in per_prompt]
        mean = sum(speeds) / len(speeds)
        var = sum((s - mean) ** 2 for s in speeds) / max(len(speeds) - 1, 1)
        std = var ** 0.5
        mean_rows = sum(rows_speeds) / len(rows_speeds)
    else:
        mean = std = mean_rows = 0.0
    summary = {
        "label": label,
        "channels_productive": channels_productive,
        "n_output_tokens_target": n_output_tokens,
        "per_prompt": per_prompt,
        "mean_productive_tokens_per_sec": mean,
        "std_productive_tokens_per_sec": std,
        "mean_rows_per_sec": mean_rows,
        "peak_memory_gb": peak_mem_gb,
    }
    print(f"[{label}] MEAN {mean:.2f} +/- {std:.2f} prod-tok/s | "
          f"{mean_rows:.2f} rows/s | peak {peak_mem_gb:.2f} GB",
          flush=True)
    return summary


def gen_send_check(model, tok, silence_token, gen_fn, out_path: Path):
    """Drive an interactive (empty-prompt) generator and inject a token via .send()."""
    lines = []
    lines.append(f"gen.send() viability check")
    lines.append(f"model: {MODEL_ID}")
    lines.append(f"silence_token: {silence_token}")
    lines.append("")
    try:
        g = gen_fn(model, tok, "", silence_token, max_rows=30, warm_start=False,
                   temperature=0.0)
        # advance past the seed row
        first = next(g)
        lines.append(f"first yield: row_idx={first[0]} is_prefill={first[2]}")
        # decide on a token to inject — use the tokenization of " hello"
        inject_ids = tok.encode(" hello", add_special_tokens=False)
        inject_tok = inject_ids[0]
        lines.append(f"injecting token id {inject_tok} ('{tok.decode([inject_tok])}') via gen.send()")
        # advance a few rows with silence on User
        row = g.send(silence_token)
        lines.append(f"after send(silence): row_idx={row[0]} user_tok={row[1][0]} output_tok={row[1][1]}")
        row = g.send(inject_tok)
        lines.append(f"after send(inject): row_idx={row[0]} user_tok={row[1][0]} output_tok={row[1][1]}")
        # continue with silence and see Output reacts
        for step in range(5):
            row = g.send(silence_token)
            lines.append(f"step {step} after inject: row_idx={row[0]} user_tok={row[1][0]} output_tok={row[1][1]}")
        # try a follow-up word
        more_ids = tok.encode(" world", add_special_tokens=False)
        if more_ids:
            row = g.send(more_ids[0])
            lines.append(f"after send(world): row_idx={row[0]} user_tok={row[1][0]} output_tok={row[1][1]}")
        lines.append("")
        lines.append("RESULT: SUCCESS — gen.send() accepts injected token IDs without breaking the loop.")
    except Exception as e:
        lines.append(f"RESULT: FAILED — {type(e).__name__}: {e}")
        import traceback
        lines.append(traceback.format_exc())

    out_path.write_text("\n".join(lines))
    print(f"[gen.send] wrote {out_path}", flush=True)


def write_env(snap_dir: str):
    """Dump installed package versions + torch/CUDA + device info."""
    import subprocess
    env_path = OUT_DIR / "env.txt"
    lines = []
    lines.append("# G6 throughput env")
    lines.append("")
    import platform
    lines.append(f"platform: {platform.platform()}")
    lines.append(f"python: {platform.python_version()}")
    lines.append(f"machine: {platform.machine()}")
    lines.append("")
    lines.append(f"torch: {torch.__version__}")
    lines.append(f"torch CUDA build: {torch.version.cuda}")
    lines.append(f"cuda available: {torch.cuda.is_available()}")
    lines.append(f"device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        lines.append(f"device: {p.name}")
        lines.append(f"device total_memory_gb: {p.total_memory / 1e9:.2f}")
        lines.append(f"device sm: {p.major}.{p.minor}")
    lines.append("")
    lines.append(f"model snapshot: {snap_dir}")
    lines.append("")
    lines.append("# pip freeze")
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=60,
        )
        lines.append(out.stdout.strip())
    except Exception as e:
        lines.append(f"# pip freeze failed: {e}")
    env_path.write_text("\n".join(lines) + "\n")
    print(f"[env] wrote {env_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tokens", type=int, default=256)
    parser.add_argument("--n-warmup-rows", type=int, default=50)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--skip-single", action="store_true")
    parser.add_argument("--skip-multi", action="store_true")
    parser.add_argument("--skip-send", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    snap = setup_paths()
    write_env(snap)

    from stream_inference import generate as gen_fn  # noqa: E402
    from stream_inference import detect_silence_token  # noqa: E402

    model, tok = load_model_and_tokenizer()
    silence_token = detect_silence_token(tok)
    print(f"[init] silence token id: {silence_token}", flush=True)

    if not args.skip_warmup:
        run_warmup(model, tok, silence_token, gen_fn, n_rows=args.n_warmup_rows)

    if not args.skip_single:
        # "single-stream" — productive = Output channel only (index 1)
        single = bench(
            model, tok, silence_token, gen_fn,
            channels_productive=[1],
            label="single_stream",
            n_output_tokens=args.n_tokens,
        )
        (OUT_DIR / "single_stream.json").write_text(json.dumps(single, indent=2))

    if not args.skip_multi:
        # "multi-stream" — productive = Output + Analytical (packing factor 2)
        multi = bench(
            model, tok, silence_token, gen_fn,
            channels_productive=[1, 2],
            label="multi_stream",
            n_output_tokens=args.n_tokens,
        )
        (OUT_DIR / "multi_stream.json").write_text(json.dumps(multi, indent=2))

    if not args.skip_send:
        gen_send_check(model, tok, silence_token, gen_fn,
                       OUT_DIR / "gen_send_check.txt")

    print("\n[done]")


if __name__ == "__main__":
    main()
