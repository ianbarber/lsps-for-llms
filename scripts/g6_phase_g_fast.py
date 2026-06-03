#!/usr/bin/env python3
"""Phase G — FAST decisive runner (skips the slow 5-prompt baseline bench; that is
already reproduced: copy 56.26%, ~4.78 tok/s). Does, in one model load:
  1. baseline profile  (ONE Phase F sliced row -> confirm aten::copy_ ~56%)
  2. swap to Phase G full-buffer flex
  3. prefill numdiff vs SDPA dense
  4. decode numdiff: Phase G full-buffer vs Phase F sliced (20 rows, same kernel)
  5. profile ONE Phase G row -> profile.txt (THE KEY CHECK: copy 56% -> small)
  6. throughput (Phase G) at PHASE_G_CTX (default 256,1024) multi packing-2
Writes profile_summary.json, numdiff.json, throughput_by_context.json, profile.txt,
baseline_repro_profile.txt. Idempotent.
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
OUT_DIR = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_g")
PATCH_B = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_b/patched/stream_inference_phase_b.py")
PATCH_F = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_f/patched/stream_inference_inplace.py")
GDIR = OUT_DIR / "patched"

PROMPTS = [
    "Write a Python function that reverses a linked list in place.",
    "Explain how a B-tree differs from a binary search tree.",
    "Refactor this code to use a context manager: open('f.txt'); read(); close().",
    "What is the time complexity of merge sort, and why?",
    "Sketch a unit test for a function that adds two integers.",
]
CTX = [int(x) for x in os.environ.get("PHASE_G_CTX", "256,1024").split(",")]
N_PROMPTS_AT = {256: 3, 1024: 3, 4096: 2, 8192: 1}
MAX_ROWS_FOR = {256: 512, 1024: 1500, 4096: 5200, 8192: 9000}


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_rows(gen_fn, model, tok, silence_token, prompt, n_rows, channels=None,
             stop_on_productive=None, **gkw):
    rows = []; prod = 0
    max_rows = n_rows + 5 if stop_on_productive is None else max(n_rows * 12, 500)
    g = gen_fn(model, tok, prompt, silence_token, max_rows=max_rows,
               warm_start=False, temperature=0.0, **gkw)
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


def bench(model, tok, silence_token, gen_fn, channels, n_output, n_prompts, **gkw):
    torch.cuda.reset_peak_memory_stats()
    per_prompt = []
    for i, p in enumerate(PROMPTS[:n_prompts]):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        rows = run_rows(gen_fn, model, tok, silence_token, p, n_output,
                        channels=channels, stop_on_productive=n_output, **gkw)
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


def profile_row(model, tok, silence_token, gen_fn, out_name, label, **gkw):
    from torch.profiler import profile, ProfilerActivity
    g = gen_fn(model, tok, PROMPTS[0], silence_token, max_rows=400,
               warm_start=False, temperature=0.0, **gkw)
    for k, (ri, row, isp) in enumerate(g):
        if not isp and k > 60:
            break
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        cnt = 0
        for ri, row, isp in g:
            if isp:
                continue
            cnt += 1
            if cnt >= 6:
                break
        torch.cuda.synchronize()
    tbl = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30)
    (OUT_DIR / out_name).write_text(tbl)
    tot = sum(ev.self_cpu_time_total for ev in prof.key_averages())
    cs = next((ev.self_cpu_time_total for ev in prof.key_averages() if ev.key == "aten::copy_"), 0)
    ic = next((ev.self_cpu_time_total for ev in prof.key_averages() if ev.key == "aten::index_copy_"), 0)
    mm = next((ev.self_cpu_time_total for ev in prof.key_averages() if ev.key == "aten::mm"), 0)
    print(f"[profile {label}] copy_={100*cs/tot if tot else 0:.2f}% "
          f"index_copy_={100*ic/tot if tot else 0:.2f}% mm={100*mm/tot if tot else 0:.2f}%",
          flush=True)
    return {"label": label, "aten_copy_self_cpu_share": cs/tot if tot else 0,
            "aten_index_copy_self_cpu_share": ic/tot if tot else 0,
            "aten_mm_self_cpu_share": mm/tot if tot else 0,
            "total_self_cpu_us": tot, "copy_self_cpu_us": cs}


def main():
    if SNAP not in sys.path:
        sys.path.insert(0, SNAP)
    b = load_mod(PATCH_B, "si_phase_b")
    fbase = load_mod(PATCH_F, "si_inplace_f")
    g = load_mod(GDIR / "stream_inference_inplace.py", "si_inplace_g")
    print("[load]", SNAP, flush=True)
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
    import torch._dynamo as dynamo; dynamo.config.cache_size_limit = 64

    # SDPA dense prefill reference (before flex install)
    prefill_rows = b.build_system_prompt_prefill(tok, silence_token, num_channels=C)[:8]
    n_prefill = len(prefill_rows); flat = [t for row in prefill_rows for t in row]
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

    inner = model.model if hasattr(model, "model") else model
    AttnClass = type(inner.layers[0].self_attn)
    TRUE_ORIG = AttnClass.forward

    # ---- baseline profile (Phase F sliced) + capture 20-row reference decode ----
    print("\n=== Phase F baseline (profile + ref decode) ===", flush=True)
    fbase.install_flex_attention(model)
    for k, _ in enumerate(fbase.generate(model, tok, PROMPTS[0], silence_token,
                                         max_rows=60, warm_start=False, temperature=0.0,
                                         max_context_rows=512)):
        if k >= 60: break
    torch.cuda.synchronize()
    bp = profile_row(model, tok, silence_token, fbase.generate,
                     "baseline_repro_profile.txt", "phase_f_sliced", max_context_rows=512)
    (OUT_DIR / "baseline_profile_summary.json").write_text(json.dumps(bp, indent=2))
    rows_ref = run_rows(fbase.generate, model, tok, silence_token, PROMPTS[0], 20,
                        max_context_rows=512)

    # ---- swap to Phase G ----
    print("\n=== install Phase G full-buffer flex ===", flush=True)
    AttnClass.forward = TRUE_ORIG; AttnClass._flex_installed = False
    g.install_flex_attention(model)

    # ---- prefill numdiff ----
    bm = g.build_block_mask(C, q_len=N, kv_len=N, q_offset=0, kv_valid=N, device=device)
    with torch.no_grad():
        of = model(input_ids=input_ids,
                   attention_mask={"full_attention": bm, "sliding_attention": bm},
                   position_ids=position_ids, use_cache=False, channel_ids=channel_ids)
    logits_flex = of.logits[0].float()
    diff = (logits_dense - logits_flex).abs()
    am_d = logits_dense.argmax(-1); am_f = logits_flex.argmax(-1); flips = (am_d != am_f)
    numdiff = {"prefill": {
        "context": f"{n_prefill}x{C}={N} tokens, SDPA(repeat_kv) vs Phase G flex(enable_gqa,full-buffer).",
        "logits_max_abs_diff": diff.max().item(), "logits_mean_abs_diff": diff.mean().item(),
        "argmax_flips": int(flips.sum().item()), "argmax_positions": int(flips.numel()),
        "argmax_flip_rate": flips.float().mean().item()}}
    if flips.any():
        gaps = []
        for pos in flips.nonzero(as_tuple=True)[0].tolist()[:30]:
            t2 = logits_dense[pos].topk(2).values; gaps.append((t2[0]-t2[1]).item())
        numdiff["prefill"]["flipped_top1_top2_gaps"] = gaps
        numdiff["prefill"]["flipped_max_gap"] = max(gaps)

    # ---- decode numdiff: G full-buffer vs F sliced reference ----
    print("\n=== decode numdiff (G full-buffer vs F sliced, 20 rows) ===", flush=True)
    rows_g = run_rows(g.generate, model, tok, silence_token, PROMPTS[0], 20, max_context_rows=512)
    n_cmp = min(len(rows_ref), len(rows_g))
    mism = [i for i in range(n_cmp) if rows_ref[i] != rows_g[i]]
    numdiff["decode_vs_reference"] = {
        "note": "G full-buffer+future-mask vs F sliced (same flex kernel). Identical => "
                "in-place offset + future-mask correct.",
        "reference": "phase_f_sliced_flex", "rows_compared": n_cmp,
        "mismatch_rows": mism, "n_mismatch_rows": len(mism),
        "all_rows_identical": len(mism) == 0}
    if mism:
        numdiff["decode_vs_reference"]["first_mismatch"] = {
            "row": mism[0], "reference_row": rows_ref[mism[0]], "phase_g_row": rows_g[mism[0]]}
    (OUT_DIR / "numdiff.json").write_text(json.dumps(numdiff, indent=2))
    print(f"[numdiff prefill] max_abs={numdiff['prefill']['logits_max_abs_diff']:.4f} "
          f"flips={numdiff['prefill']['argmax_flips']}/{numdiff['prefill']['argmax_positions']}", flush=True)
    print(f"[numdiff decode] {n_cmp} rows, {len(mism)} mismatch -> identical={len(mism)==0}", flush=True)

    # ---- profile ONE Phase G row (THE KEY DELIVERABLE) ----
    print("\n=== profile_one_row (Phase G full-buffer) — KEY CHECK ===", flush=True)
    pf = profile_row(model, tok, silence_token, g.generate, "profile.txt",
                     "phase_g_full_buffer", max_context_rows=512)
    pf["baseline_copy_share_phase_f"] = bp["aten_copy_self_cpu_share"]
    (OUT_DIR / "profile_summary.json").write_text(json.dumps(pf, indent=2))
    print(f"[profile] copy share: F={100*bp['aten_copy_self_cpu_share']:.1f}% "
          f"-> G={100*pf['aten_copy_self_cpu_share']:.1f}%", flush=True)

    # ---- throughput ----
    print("\n=== throughput (Phase G) ===", flush=True)
    by_ctx = {}
    tp_path = OUT_DIR / "throughput_by_context.json"
    if tp_path.exists():
        try: by_ctx = json.loads(tp_path.read_text()).get("by_context", {})
        except Exception: by_ctx = {}
    for n in CTX:
        by_ctx[str(n)] = bench(model, tok, silence_token, g.generate, [1, 2], n,
                               N_PROMPTS_AT.get(n, 2), max_context_rows=MAX_ROWS_FOR.get(n, n+1000))
        tp_path.write_text(json.dumps({
            "mode": "full_buffer_flex_gqa_blockmask_future_mask",
            "harness": "FlexAttention enable_gqa over FULL pre-grown KV buffer, BlockMask "
                       "future-mask (kv_idx<cursor+C), FORCE_USE_FLEX_ATTENTION, NO slice/contiguous, "
                       "multi-stream packing-2 [1,2], greedy T=0.",
            "by_context": dict(sorted(by_ctx.items(), key=lambda kv: int(kv[0])))}, indent=2))
    print("\n[done] fast Phase G complete.", flush=True)


if __name__ == "__main__":
    main()
