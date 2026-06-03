#!/usr/bin/env python3
"""Phase F — throughput for LONG contexts (4096, 8192) only, appended to the
existing throughput_by_context.json (256/1024 already measured). Standalone reload
because the combined run_all was interrupted. Idempotent: merges into by_context.
"""
from __future__ import annotations
import importlib.util, json, os, sys, time
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REVISION = "54c7451bfcccecc233fad91affa68563d1de9d66"
SNAP = os.path.expanduser(
    f"~/.cache/huggingface/hub/models--JonasGeiping--stream-qwen3-8b/snapshots/{REVISION}")
OUT_DIR = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_f")
PATCH_B = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_b/patched/stream_inference_phase_b.py")
FDIR = OUT_DIR / "patched"

PROMPTS = [
    "Write a Python function that reverses a linked list in place.",
    "Explain how a B-tree differs from a binary search tree.",
    "Refactor this code to use a context manager: open('f.txt'); read(); close().",
]
CTX = [int(x) for x in os.environ.get("PHASE_F_LONG_CTX", "4096,8192").split(",")]
N_PROMPTS_AT = {4096: 3, 8192: 1}
MAX_ROWS_FOR = {4096: 5200, 8192: 9000}


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_rows(gen_fn, model, tok, silence_token, prompt, n_rows, channels, **gkw):
    rows = []; prod = 0
    g = gen_fn(model, tok, prompt, silence_token, max_rows=max(n_rows*12, 500),
               warm_start=False, temperature=0.0, **gkw)
    for ri, row, isp in g:
        if isp: continue
        rows.append(list(row))
        for c in channels:
            if c < len(row) and row[c] != silence_token: prod += 1
        if prod >= n_rows: break
    return rows


def bench(model, tok, silence_token, gen_fn, channels, n_output, n_prompts, **gkw):
    torch.cuda.reset_peak_memory_stats()
    per_prompt = []
    for i, p in enumerate(PROMPTS[:n_prompts]):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        rows = run_rows(gen_fn, model, tok, silence_token, p, n_output, channels, **gkw)
        torch.cuda.synchronize(); elapsed = time.perf_counter() - t0
        prod = sum(1 for r in rows for c in channels if c < len(r) and r[c] != silence_token)
        nrows = len(rows)
        per_prompt.append({"prompt_idx": i, "elapsed_s": elapsed, "rows": nrows,
                           "productive_tokens": prod,
                           "productive_tokens_per_sec": prod/elapsed if elapsed else 0,
                           "rows_per_sec": nrows/elapsed if elapsed else 0,
                           "ms_per_row": 1000*elapsed/nrows if nrows else 0})
        print(f"[multi n={n_output}] {i}: {elapsed:.1f}s {nrows}r {prod}p "
              f"{prod/elapsed:.2f} tok/s {1000*elapsed/nrows:.1f} ms/row", flush=True)
    peak = torch.cuda.max_memory_allocated()/1e9
    sp = [m["productive_tokens_per_sec"] for m in per_prompt]
    mean = sum(sp)/len(sp); std = (sum((s-mean)**2 for s in sp)/max(len(sp)-1,1))**0.5
    ms = [m["ms_per_row"] for m in per_prompt]; rps = [m["rows_per_sec"] for m in per_prompt]
    s = {"label": "multi_stream", "channels_productive": channels, "n_prompts": n_prompts,
         "n_output_tokens_target": n_output, "per_prompt": per_prompt,
         "mean_productive_tokens_per_sec": mean, "std_productive_tokens_per_sec": std,
         "mean_rows_per_sec": sum(rps)/len(rps), "mean_ms_per_row": sum(ms)/len(ms),
         "peak_memory_gb": peak}
    print(f"[multi n={n_output}] MEAN {mean:.2f}+/-{std:.2f} tok/s | "
          f"{s['mean_ms_per_row']:.1f} ms/row | peak {peak:.2f}GB", flush=True)
    return s


def main():
    if SNAP not in sys.path: sys.path.insert(0, SNAP)
    b = load_mod(PATCH_B, "si_phase_b")
    f = load_mod(FDIR / "stream_inference_inplace.py", "si_inplace")
    print("[load]", SNAP, flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        SNAP, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    tok = AutoTokenizer.from_pretrained(SNAP, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    silence_token = b.detect_silence_token(tok)
    import torch._dynamo as dynamo; dynamo.config.cache_size_limit = 64
    f.install_flex_attention(model)
    # warmup
    for k, _ in enumerate(f.generate(model, tok, PROMPTS[0], silence_token,
                                     max_rows=60, warm_start=False, temperature=0.0,
                                     max_context_rows=512)):
        if k >= 60: break
    torch.cuda.synchronize()

    tp_path = OUT_DIR / "throughput_by_context.json"
    tp = json.loads(tp_path.read_text())
    by_ctx = tp["by_context"]
    for n in CTX:
        npr = N_PROMPTS_AT.get(n, 1); mcr = MAX_ROWS_FOR.get(n, n+1000)
        by_ctx[str(n)] = bench(model, tok, silence_token, f.generate, [1, 2], n, npr,
                               max_context_rows=mcr)
        tp["by_context"] = dict(sorted(by_ctx.items(), key=lambda kv: int(kv[0])))
        tp_path.write_text(json.dumps(tp, indent=2))
    print("[done] long-ctx throughput complete.", flush=True)


if __name__ == "__main__":
    main()
