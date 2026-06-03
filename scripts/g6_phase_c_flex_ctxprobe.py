#!/usr/bin/env python3
"""Phase C Route 2 — lightweight per-row-latency-vs-context probe.

Running greedy decode all the way to 8192 productive tokens takes ~1 h/prompt at
~2.5 tok/s, which blows the timebox. Instead we measure steady-state ms/row at a
range of CACHE DEPTHS directly: build a synthetic prefill of D rows (so the cache
holds D*C tokens), warm up, then time N_MEASURE decode rows. The per-row time at
cache depth D*C is what the L4 integral needs.

We do this for BOTH the flex (BlockMask) and SDPA (dense) paths so the context
curve is directly comparable. Cache depths chosen to bracket the L4 trajectory
(productive-token targets map ~1:1 to rows for the Output channel under silence
penalty, and total flat context = rows*C).

Writes runs/g6_phase_c_flex/context_latency.json. Idempotent.

Usage: python g6_phase_c_flex_ctxprobe.py   (acquire GPU lock externally)
"""
from __future__ import annotations

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
FLEX_DIR = OUT_DIR / "patched"
C = 10
# Cache depths in ROWS (flat tokens = rows*C). Brackets L4's 256..8192-token trajs.
DEPTH_ROWS = [25, 100, 400, 800]   # ~250, 1000, 4000, 8000 flat tokens
N_MEASURE = 40


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_prefill(model, tok, silence_token, depth_rows, device):
    """Run a synthetic block-causal prefill of depth_rows rows; return past_kv +
    all_cached_ids. Uses a repeating plausible token pattern (silence + a word)."""
    base_word = tok.encode(" think", add_special_tokens=False)[0]
    rows = []
    for r in range(depth_rows):
        row = [silence_token, base_word] + [base_word] * (C - 2)
        rows.append(row)
    flat = [t for row in rows for t in row]
    N = depth_rows * C
    input_ids = torch.tensor([flat], device=device, dtype=torch.long)
    position_ids = torch.tensor([[r for r in range(depth_rows) for _ in range(C)]],
                                device=device, dtype=torch.long)
    channel_ids = torch.tensor([[c for _ in range(depth_rows) for c in range(C)]],
                               device=device, dtype=torch.long)
    return input_ids, position_ids, channel_ids, N


def measure_path(model, tok, silence_token, device, flex_mod, use_flex, fap):
    """For each depth, prefill then time N_MEASURE single-row decode steps."""
    results = []
    base_word = tok.encode(" think", add_special_tokens=False)[0]
    _peer = torch.where(torch.eye(C, dtype=torch.bool, device=device),
                        torch.tensor(0.0, device=device, dtype=torch.bfloat16),
                        torch.tensor(-1e4, device=device, dtype=torch.bfloat16))
    _cids = torch.arange(C, device=device).unsqueeze(0)
    for depth in DEPTH_ROWS:
        torch.cuda.reset_peak_memory_stats()
        ii, pi, ci, N = build_prefill(model, tok, silence_token, depth, device)
        if use_flex:
            bm = fap.build_block_mask(C, q_len=N, kv_len=N, q_offset=0, device=device)
            am = {"full_attention": bm, "sliding_attention": bm}
        else:
            ridx = torch.arange(N, device=device) // C
            allow = (ridx.unsqueeze(0) < ridx.unsqueeze(1)) | torch.eye(N, dtype=torch.bool, device=device)
            dm = torch.where(allow, torch.tensor(0.0, device=device),
                             torch.tensor(-1e4, device=device)).to(torch.bfloat16).view(1, 1, N, N)
            am = {"full_attention": dm, "sliding_attention": dm}
        with torch.no_grad():
            out = model(input_ids=ii, attention_mask=am, position_ids=pi,
                        use_cache=True, channel_ids=ci)
        past_kv = out.past_key_values

        row = [silence_token, base_word] + [base_word] * (C - 2)
        # warmup (compile for flex)
        for w in range(5):
            cached = past_kv.get_seq_length()
            iid = torch.tensor([row], device=device, dtype=torch.long)
            posid = torch.full((1, C), (cached // C), device=device, dtype=torch.long)
            if use_flex:
                bm = fap.build_block_mask(C, q_len=C, kv_len=cached + C, q_offset=cached, device=device)
                am = {"full_attention": bm, "sliding_attention": bm}
            else:
                cache_block = torch.zeros(C, cached, device=device, dtype=torch.bfloat16)
                m = torch.cat([cache_block, _peer], dim=-1).view(1, 1, C, cached + C)
                am = {"full_attention": m, "sliding_attention": m}
            with torch.no_grad():
                out = model(input_ids=iid, attention_mask=am, position_ids=posid,
                            past_key_values=past_kv, use_cache=True, channel_ids=_cids)
            past_kv = out.past_key_values

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for s in range(N_MEASURE):
            cached = past_kv.get_seq_length()
            iid = torch.tensor([row], device=device, dtype=torch.long)
            posid = torch.full((1, C), (cached // C), device=device, dtype=torch.long)
            if use_flex:
                bm = fap.build_block_mask(C, q_len=C, kv_len=cached + C, q_offset=cached, device=device)
                am = {"full_attention": bm, "sliding_attention": bm}
            else:
                cache_block = torch.zeros(C, cached, device=device, dtype=torch.bfloat16)
                m = torch.cat([cache_block, _peer], dim=-1).view(1, 1, C, cached + C)
                am = {"full_attention": m, "sliding_attention": m}
            with torch.no_grad():
                out = model(input_ids=iid, attention_mask=am, position_ids=posid,
                            past_key_values=past_kv, use_cache=True, channel_ids=_cids)
            past_kv = out.past_key_values
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        ms_per_row = 1000.0 * elapsed / N_MEASURE
        peak = torch.cuda.max_memory_allocated() / 1e9
        results.append({"depth_rows": depth, "cache_tokens_start": N,
                        "n_measure": N_MEASURE, "ms_per_row": ms_per_row,
                        "rows_per_sec": N_MEASURE / elapsed,
                        "peak_memory_gb": peak})
        print(f"[{'flex' if use_flex else 'sdpa'}] depth={depth}r ({N}tok): "
              f"{ms_per_row:.1f} ms/row  {N_MEASURE/elapsed:.2f} rows/s  peak {peak:.2f}GB",
              flush=True)
        del past_kv
        torch.cuda.empty_cache()
    return results


def main():
    snap = snapshot_download(MODEL_ID)
    if snap not in sys.path:
        sys.path.insert(0, snap)
    b = load_mod(PATCH_B, "si_phase_b")
    flex = load_mod(FLEX_DIR / "stream_inference_flex.py", "si_flex")
    fap = flex._flex if hasattr(flex, "_flex") else load_mod(FLEX_DIR / "flex_attention_patch.py", "fap")

    print("[load] loading model...", flush=True)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    silence_token = b.detect_silence_token(tok)
    device = model.get_input_embeddings().weight.device
    print(f"[load] done in {time.perf_counter()-t0:.1f}s", flush=True)

    import torch._dynamo as dynamo
    dynamo.config.cache_size_limit = 64

    print("\n=== SDPA path ===", flush=True)
    sdpa = measure_path(model, tok, silence_token, device, flex, False, fap)

    print("\n=== Flex path ===", flush=True)
    flex.install_flex_attention(model)
    flexr = measure_path(model, tok, silence_token, device, flex, True, fap)

    out = {
        "method": f"Steady-state ms/row at fixed cache depths {DEPTH_ROWS} rows "
                  f"(={[d*C for d in DEPTH_ROWS]} tokens), {N_MEASURE} timed decode rows each, "
                  "after warmup. Greedy single-row decode.",
        "sdpa": sdpa, "flex": flexr,
    }
    (OUT_DIR / "context_latency.json").write_text(json.dumps(out, indent=2))
    print(f"\n[done] wrote {OUT_DIR}/context_latency.json", flush=True)


if __name__ == "__main__":
    main()
