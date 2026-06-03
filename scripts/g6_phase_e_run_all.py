#!/usr/bin/env python3
"""Phase E — combined GPU runner. Loads the 8B model ONCE (from local ~/.cache,
offline) and runs:
  1. baseline_repro      (Phase B SDPA throughput, single+multi @256)
  2. numdiff             (SDPA-dense vs GQA-flex logits diff + argmax flip rate)
  3. profile_one_row     (torch.profiler on ONE GQA-flex decode row -> profile.txt;
                          the proof aten::copy_/repeat_kv dropped from 47.7%)
  4. throughput_by_context (GQA-flex multi @ 256/1024/4096[/8192], tok/s, ms/row, peak)

SDPA-side work runs BEFORE install_flex_attention; flex work after. Idempotent;
each stage writes its own JSON.
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
OUT_DIR = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_e")
PATCH_B = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_b/patched/stream_inference_phase_b.py")
GQA_DIR = OUT_DIR / "patched"

PROMPTS = [
    "Write a Python function that reverses a linked list in place.",
    "Explain how a B-tree differs from a binary search tree.",
    "Refactor this code to use a context manager: open('f.txt'); read(); close().",
    "What is the time complexity of merge sort, and why?",
    "Sketch a unit test for a function that adds two integers.",
]

# Context lengths to measure. 8192 optionally via env PHASE_E_CTX="256,1024,4096,8192".
CTX = [int(x) for x in os.environ.get("PHASE_E_CTX", "256,1024,4096").split(",")]
N_PROMPTS_AT = {256: 5, 1024: 5, 4096: 3, 8192: 1}  # fewer prompts at long ctx


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


def bench(model, tok, silence_token, gen_fn, channels, label, n_output, n_prompts=5):
    torch.cuda.reset_peak_memory_stats()
    per_prompt = []
    prompts = PROMPTS[:n_prompts]
    for i, p in enumerate(prompts):
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
                           "productive_tokens": prod, "productive_tokens_per_sec": tps,
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
    s = {"label": label, "channels_productive": channels, "n_prompts": len(prompts),
         "n_output_tokens_target": n_output, "per_prompt": per_prompt,
         "mean_productive_tokens_per_sec": mean, "std_productive_tokens_per_sec": std,
         "mean_rows_per_sec": sum(rps) / len(rps),
         "mean_ms_per_row": sum(ms) / len(ms), "peak_memory_gb": peak}
    print(f"[{label} n={n_output}] MEAN {mean:.2f}+/-{std:.2f} tok/s | "
          f"{s['mean_ms_per_row']:.1f} ms/row | peak {peak:.2f}GB", flush=True)
    return s


def profile_one_row(model, tok, silence_token, gqa):
    """Profile a single GQA-flex decode row (after warmup) -> profile.txt.
    Confirms aten::copy_ / repeat_kv share dropped from Phase D's 47.7%."""
    from torch.profiler import profile, ProfilerActivity
    # warm the generator a bit so we profile a steady-state decode row, not compile.
    g = gqa.generate(model, tok, PROMPTS[0], silence_token, max_rows=400,
                     warm_start=False, temperature=0.0)
    for k, (ri, row, isp) in enumerate(g):
        if not isp and k > 60:
            break
    torch.cuda.synchronize()
    # profile the next ~6 rows
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        cnt = 0
        for ri, row, isp in g:
            if isp:
                continue
            cnt += 1
            if cnt >= 6:
                break
        torch.cuda.synchronize()
    tbl = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30)
    (OUT_DIR / "profile.txt").write_text(tbl)
    # extract copy share
    copy_share = None
    total = None
    for ev in prof.key_averages():
        if ev.key == "aten::copy_":
            copy_self = ev.self_cpu_time_total
    # total self cpu
    tot = sum(ev.self_cpu_time_total for ev in prof.key_averages())
    cs = next((ev.self_cpu_time_total for ev in prof.key_averages() if ev.key == "aten::copy_"), 0)
    share = cs / tot if tot else 0.0
    print(f"[profile] aten::copy_ self_cpu share = {100*share:.2f}% "
          f"(Phase D was 47.7%)", flush=True)
    return {"aten_copy_self_cpu_share": share, "total_self_cpu_us": tot,
            "copy_self_cpu_us": cs}


def main():
    if SNAP not in sys.path:
        sys.path.insert(0, SNAP)
    b = load_mod(PATCH_B, "si_phase_b")
    gqa = load_mod(GQA_DIR / "stream_inference_gqa.py", "si_gqa")

    print("[load] loading model from", SNAP, flush=True)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        SNAP, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    tok = AutoTokenizer.from_pretrained(SNAP, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    silence_token = b.detect_silence_token(tok)
    device = model.get_input_embeddings().weight.device
    C = 10
    print(f"[load] done in {time.perf_counter()-t0:.1f}s", flush=True)

    import torch._dynamo as dynamo
    dynamo.config.cache_size_limit = 64

    # ===== numdiff dense (SDPA) logits =====
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

    # ===== baseline_repro (Phase B SDPA) =====
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

    # ===== install GQA flex, then flex-side work =====
    print("\n=== installing FlexAttention + enable_gqa patch (Phase E) ===", flush=True)
    gqa.install_flex_attention(model)

    # numdiff flex (GQA) logits
    block_mask = gqa.build_block_mask(C, q_len=N, kv_len=N, q_offset=0, device=device)
    with torch.no_grad():
        of = model(input_ids=input_ids,
                   attention_mask={"full_attention": block_mask, "sliding_attention": block_mask},
                   position_ids=position_ids, use_cache=False, channel_ids=channel_ids)
    logits_flex = of.logits[0].float()
    diff = (logits_dense - logits_flex).abs()
    am_d = logits_dense.argmax(-1); am_f = logits_flex.argmax(-1)
    flips = (am_d != am_f)
    numdiff = {
        "context": f"{n_prefill} prefill rows x {C} = {N} tokens, full prefill, "
                   f"SDPA(repeat_kv) vs FlexAttention(enable_gqa).",
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

    # ===== profile one row (the key deliverable) =====
    print("\n=== profile_one_row (GQA flex) ===", flush=True)
    prof_summary = profile_one_row(model, tok, silence_token, gqa)
    (OUT_DIR / "profile_summary.json").write_text(json.dumps(prof_summary, indent=2))

    # ===== throughput by context (GQA flex) =====
    print("\n=== throughput_by_context (GQA flex) ===", flush=True)
    # warmup compile
    tw = time.perf_counter(); cnt = 0; fr = None; twf = time.perf_counter()
    for ri, row, isp in gqa.generate(model, tok, PROMPTS[0], silence_token,
                                     max_rows=60, warm_start=False, temperature=0.0):
        if not isp and fr is None:
            torch.cuda.synchronize(); fr = time.perf_counter() - twf
        cnt += 1
        if cnt >= 60:
            break
    torch.cuda.synchronize()
    warm_s = time.perf_counter() - tw
    print(f"[flex warmup] {cnt} rows in {warm_s:.1f}s; first-row {fr:.1f}s", flush=True)

    by_ctx = {}
    for n in CTX:
        npr = N_PROMPTS_AT.get(n, 5)
        by_ctx[str(n)] = bench(model, tok, silence_token, gqa.generate, [1, 2],
                               "multi_stream", n, n_prompts=npr)
        # write incrementally so a long-ctx timeout still leaves partial results
        (OUT_DIR / "throughput_by_context.json").write_text(json.dumps({
            "mode": "flex_gqa",
            "harness": "FlexAttention enable_gqa BlockMask, dynamic KV, multi-stream "
                       "packing-2 [1,2], greedy T=0.",
            "warmup_seconds": warm_s, "first_row_seconds": fr,
            "by_context": by_ctx}, indent=2))

    print("\n[done] all stages complete.", flush=True)


if __name__ == "__main__":
    main()
