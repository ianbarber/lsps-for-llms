#!/usr/bin/env python3
"""Phase C Route 2 — combined GPU runner. Loads the 8B model ONCE and runs:
  1. numdiff  (single-forward flex-vs-dense logits diff; bug-vs-noise)
  2. identity (5 prompts x 30 rows greedy; SDPA vs flex bit-identity)
  3. baseline_repro (Phase B SDPA throughput, single+multi @256)
  4. throughput_by_context (flex multi @ 256/1024/4096/8192)

This minimizes GPU-lock contention with the concurrent static route: one model
load, one lock acquisition. Each stage writes its own JSON. Idempotent.

Order matters: SDPA stages (baseline_repro, the ref side of identity, numdiff
dense side) run BEFORE install_flex_attention monkeypatches the attention class;
flex stages run after. We run all SDPA-side work first, then install flex, then
all flex work.
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


def run_rows(gen_fn, model, tok, silence_token, prompt, n_rows, channels=None,
             stop_on_productive=None):
    rows = []
    prod = 0
    max_rows = n_rows + 5 if stop_on_productive is None else max(n_rows * 12, 500)
    g = gen_fn(model, tok, prompt, silence_token, max_rows=max_rows,
               warm_start=False, temperature=0.0)
    for row_idx, row, is_prefill in g:
        if is_prefill:
            continue
        rows.append(list(row))
        if stop_on_productive is not None:
            for c in channels:
                if c < len(row) and row[c] != silence_token:
                    prod += 1
            if prod >= stop_on_productive:
                break
        elif len(rows) >= n_rows:
            break
    return rows


def bench(model, tok, silence_token, gen_fn, channels, label, n_output):
    torch.cuda.reset_peak_memory_stats()
    per_prompt = []
    for i, p in enumerate(PROMPTS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rows = run_rows(gen_fn, model, tok, silence_token, p, n_output,
                        channels=channels, stop_on_productive=n_output)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        prod = sum(1 for r in rows for c in channels if c < len(r) and r[c] != silence_token)
        nrows = len(rows)
        tps = prod / elapsed if elapsed else 0.0
        ms_per_row = 1000.0 * elapsed / nrows if nrows else 0.0
        per_prompt.append({"prompt_idx": i, "elapsed_s": elapsed, "rows": nrows,
                           "productive_tokens": prod,
                           "productive_tokens_per_sec": tps,
                           "rows_per_sec": nrows / elapsed if elapsed else 0.0,
                           "ms_per_row": ms_per_row})
        print(f"[{label} n={n_output}] {i}: {elapsed:.2f}s {nrows}r {prod}p "
              f"{tps:.2f} tok/s {ms_per_row:.1f} ms/row", flush=True)
    peak = torch.cuda.max_memory_allocated() / 1e9
    sp = [m["productive_tokens_per_sec"] for m in per_prompt]
    mean = sum(sp) / len(sp)
    std = (sum((s - mean) ** 2 for s in sp) / max(len(sp) - 1, 1)) ** 0.5
    ms = [m["ms_per_row"] for m in per_prompt]
    rps = [m["rows_per_sec"] for m in per_prompt]
    s = {"label": label, "channels_productive": channels,
         "n_output_tokens_target": n_output, "per_prompt": per_prompt,
         "mean_productive_tokens_per_sec": mean, "std_productive_tokens_per_sec": std,
         "mean_rows_per_sec": sum(rps) / len(rps),
         "mean_ms_per_row": sum(ms) / len(ms), "peak_memory_gb": peak}
    print(f"[{label} n={n_output}] MEAN {mean:.2f}+/-{std:.2f} tok/s | "
          f"{s['mean_ms_per_row']:.1f} ms/row | peak {peak:.2f}GB", flush=True)
    return s


def main():
    snap = snapshot_download(MODEL_ID)
    if snap not in sys.path:
        sys.path.insert(0, snap)
    b = load_mod(PATCH_B, "si_phase_b")
    flex = load_mod(FLEX_DIR / "stream_inference_flex.py", "si_flex")

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
    C = 10
    print(f"[load] done in {time.perf_counter()-t0:.1f}s", flush=True)

    import torch._dynamo as dynamo
    dynamo.config.cache_size_limit = 64

    # ============ SDPA-side work (before flex patch) ============
    # numdiff dense logits
    prefill_rows = b.build_system_prompt_prefill(tok, silence_token, num_channels=C)[:8]
    n_prefill = len(prefill_rows)
    flat = [t for row in prefill_rows for t in row]
    N = n_prefill * C
    input_ids = torch.tensor([flat], device=device, dtype=torch.long)
    position_ids = torch.tensor([[r for r in range(n_prefill) for _ in range(C)]],
                                device=device, dtype=torch.long)
    channel_ids = torch.tensor([[c for _ in range(n_prefill) for c in range(C)]],
                               device=device, dtype=torch.long)
    rows_idx = torch.arange(N, device=device) // C
    allowed = (rows_idx.unsqueeze(0) < rows_idx.unsqueeze(1)) | torch.eye(N, dtype=torch.bool, device=device)
    dense_mask = torch.where(allowed, torch.tensor(0.0, device=device),
                             torch.tensor(-1e4, device=device)).to(torch.bfloat16).view(1, 1, N, N)
    with torch.no_grad():
        od = model(input_ids=input_ids,
                   attention_mask={"full_attention": dense_mask, "sliding_attention": dense_mask},
                   position_ids=position_ids, use_cache=False, channel_ids=channel_ids)
    logits_dense = od.logits[0].float()

    # baseline_repro (Phase B SDPA throughput)
    print("\n=== baseline_repro (SDPA) ===", flush=True)
    for ri, _ in enumerate(b.generate(model, tok, PROMPTS[0], silence_token,
                                      max_rows=50, warm_start=False, temperature=0.0)):
        if ri >= 50:
            break
    bl_single = bench(model, tok, silence_token, b.generate, [1], "single_stream", 256)
    bl_multi = bench(model, tok, silence_token, b.generate, [1, 2], "multi_stream", 256)
    (OUT_DIR / "baseline_repro.json").write_text(json.dumps({
        "mode": "baseline_repro",
        "harness": "5 prompts x 256 productive tokens, greedy T=0, Phase B SDPA generator.",
        "single_stream": bl_single, "multi_stream": bl_multi,
        "phase_b_published_multi_tok_s": 4.95}, indent=2))

    # identity reference rows (SDPA)
    print("\n=== identity reference (SDPA) ===", flush=True)
    ref_rows = {}
    for i, p in enumerate(PROMPTS):
        torch.manual_seed(0)
        ref_rows[i] = run_rows(b.generate, model, tok, silence_token, p, 30)

    # ============ install flex, then flex-side work ============
    print("\n=== installing FlexAttention patch ===", flush=True)
    flex.install_flex_attention(model)

    # numdiff flex logits
    block_mask = flex.build_block_mask(C, q_len=N, kv_len=N, q_offset=0, device=device)
    with torch.no_grad():
        of = model(input_ids=input_ids,
                   attention_mask={"full_attention": block_mask, "sliding_attention": block_mask},
                   position_ids=position_ids, use_cache=False, channel_ids=channel_ids)
    logits_flex = of.logits[0].float()
    diff = (logits_dense - logits_flex).abs()
    am_d = logits_dense.argmax(-1); am_f = logits_flex.argmax(-1)
    flips = (am_d != am_f)
    numdiff = {
        "context": f"{n_prefill} prefill rows x {C} = {N} tokens, full prefill",
        "logits_max_abs_diff": diff.max().item(),
        "logits_mean_abs_diff": diff.mean().item(),
        "argmax_flips": int(flips.sum().item()),
        "argmax_positions": int(flips.numel()),
        "argmax_flip_rate": flips.float().mean().item()}
    if flips.any():
        fidx = flips.nonzero(as_tuple=True)[0]
        gaps = []
        for pos in fidx.tolist()[:30]:
            t2 = logits_dense[pos].topk(2).values
            gaps.append((t2[0] - t2[1]).item())
        numdiff["flipped_top1_top2_gaps"] = gaps
        numdiff["flipped_max_gap"] = max(gaps)
    (OUT_DIR / "numdiff.json").write_text(json.dumps(numdiff, indent=2))
    print(f"[numdiff] max_abs_diff={numdiff['logits_max_abs_diff']:.4f} "
          f"flips={numdiff['argmax_flips']}/{numdiff['argmax_positions']} "
          f"(gaps<={numdiff.get('flipped_max_gap')})", flush=True)

    # identity candidate rows (flex) — warmup first to compile
    print("\n=== identity candidate (flex) ===", flush=True)
    tw = time.perf_counter()
    fr = None; twf = time.perf_counter(); cnt = 0
    for ri, _ in enumerate(flex.generate(model, tok, PROMPTS[0], silence_token,
                                         max_rows=60, warm_start=False, temperature=0.0)):
        if fr is None:
            torch.cuda.synchronize(); fr = time.perf_counter() - twf
        cnt += 1
        if cnt >= 60:
            break
    torch.cuda.synchronize()
    warm_s = time.perf_counter() - tw
    print(f"[flex warmup] {cnt} rows in {warm_s:.1f}s; first-row {fr:.1f}s", flush=True)

    cand_rows = {}
    for i, p in enumerate(PROMPTS):
        torch.manual_seed(0)
        cand_rows[i] = run_rows(flex.generate, model, tok, silence_token, p, 30)

    per_prompt = []; all_id = True; tot = 0; mm_tot = 0
    for i in range(len(PROMPTS)):
        r = ref_rows[i]; c = cand_rows[i]; n = min(len(r), len(c)); mism = []
        for ri in range(n):
            for ci in range(len(r[ri])):
                tot += 1
                if ci < len(c[ri]) and r[ri][ci] != c[ri][ci]:
                    mm_tot += 1
                    mism.append({"row": ri, "channel": ci, "ref": r[ri][ci], "flex": c[ri][ci]})
        identical = (len(r) == len(c)) and not mism
        all_id = all_id and identical
        # first divergence row
        first_div = mism[0]["row"] if mism else None
        per_prompt.append({"prompt_idx": i, "ref_rows": len(r), "flex_rows": len(c),
                           "mismatches": len(mism), "first_divergence_row": first_div,
                           "first_mismatches": mism[:8],
                           "verdict": "identical" if identical else "DIVERGED"})
        print(f"[id] prompt {i}: {'IDENTICAL' if identical else 'DIVERGED'} "
              f"({len(mism)} mism, first@row {first_div})", flush=True)
    (OUT_DIR / "identity.json").write_text(json.dumps({
        "harness": "5 prompts x 30 rows, greedy T=0, seed 0, Phase B SDPA vs Flex BlockMask.",
        "n_rows": 30, "total_tokens_compared": tot, "total_mismatches": mm_tot,
        "argmax_flip_rate": mm_tot / max(tot, 1),
        "per_prompt": per_prompt, "all_identical": all_id,
        "verdict": "PASS" if all_id else "DIVERGED_NUMERICAL"}, indent=2))

    # throughput by context (flex)
    print("\n=== throughput_by_context (flex) ===", flush=True)
    by_ctx = {}
    for n in [256, 1024, 4096, 8192]:
        by_ctx[str(n)] = bench(model, tok, silence_token, flex.generate, [1, 2],
                               "multi_stream", n)
    (OUT_DIR / "throughput_by_context.json").write_text(json.dumps({
        "mode": "flex",
        "harness": "FlexAttention BlockMask generator, multi-stream packing-2 [1,2], greedy T=0.",
        "warmup_seconds": warm_s, "first_row_seconds": fr,
        "by_context": by_ctx}, indent=2))

    print("\n[done] all stages complete.", flush=True)


if __name__ == "__main__":
    main()
