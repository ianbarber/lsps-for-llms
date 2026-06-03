#!/usr/bin/env python3
"""Phase G — combined GPU runner. Loads the 8B model ONCE (local ~/.cache, offline):

  1. baseline_repro   (Phase F in-place-flex VALID-REGION-SLICE decoder, multi @256)
                      -> confirm ~4.7 tok/s
  2. baseline_profile (torch.profiler on ONE Phase F sliced decode row ->
                      baseline_repro_profile.txt; confirm aten::copy_ ~56%)
  3. numdiff          (prefill: SDPA-dense vs Phase G full-buffer-flex logits; PLUS a
                      teacher-forced 20-decode-row check that the in-place write
                      offset + future-mask are correct vs a DynamicCache flex
                      reference) -> numdiff.json
  4. profile_one_row  (torch.profiler on ONE Phase G full-buffer decode row ->
                      profile.txt; THE KEY CHECK: aten::copy_ 56% -> small)
  5. throughput_by_context (Phase G full-buffer flex multi @ 256/1024/4096[/8192])
                      -> throughput_by_context.json

Idempotent; each stage writes its own JSON.
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
PATCH_E = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_e/patched/stream_inference_gqa.py")
PATCH_F = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_f/patched/stream_inference_inplace.py")
GDIR = OUT_DIR / "patched"

PROMPTS = [
    "Write a Python function that reverses a linked list in place.",
    "Explain how a B-tree differs from a binary search tree.",
    "Refactor this code to use a context manager: open('f.txt'); read(); close().",
    "What is the time complexity of merge sort, and why?",
    "Sketch a unit test for a function that adds two integers.",
]

CTX = [int(x) for x in os.environ.get("PHASE_G_CTX", "256,1024,4096").split(",")]
N_PROMPTS_AT = {256: 5, 1024: 5, 4096: 3, 8192: 1}
MAX_ROWS_FOR = {256: 512, 1024: 1500, 4096: 5200, 8192: 9000}


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_rows(gen_fn, model, tok, silence_token, prompt, n_rows, channels=None,
             stop_on_productive=None, **gkw):
    rows = []
    prod = 0
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


def bench(model, tok, silence_token, gen_fn, channels, label, n_output, n_prompts=5,
          **gkw):
    torch.cuda.reset_peak_memory_stats()
    per_prompt = []
    prompts = PROMPTS[:n_prompts]
    for i, p in enumerate(prompts):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rows = run_rows(gen_fn, model, tok, silence_token, p, n_output,
                        channels=channels, stop_on_productive=n_output, **gkw)
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


def profile_row(model, tok, silence_token, gen_fn, out_name, label, **gkw):
    from torch.profiler import profile, ProfilerActivity
    g = gen_fn(model, tok, PROMPTS[0], silence_token, max_rows=400,
               warm_start=False, temperature=0.0, **gkw)
    for k, (ri, row, isp) in enumerate(g):
        if not isp and k > 60:
            break
    torch.cuda.synchronize()
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
    (OUT_DIR / out_name).write_text(tbl)
    tot = sum(ev.self_cpu_time_total for ev in prof.key_averages())
    cs = next((ev.self_cpu_time_total for ev in prof.key_averages() if ev.key == "aten::copy_"), 0)
    ic = next((ev.self_cpu_time_total for ev in prof.key_averages() if ev.key == "aten::index_copy_"), 0)
    share = cs / tot if tot else 0.0
    print(f"[profile {label}] aten::copy_ self_cpu share = {100*share:.2f}%  "
          f"index_copy_ = {100*ic/tot if tot else 0:.2f}%", flush=True)
    return {"label": label, "aten_copy_self_cpu_share": share,
            "aten_index_copy_self_cpu_share": ic / tot if tot else 0.0,
            "total_self_cpu_us": tot, "copy_self_cpu_us": cs}


def main():
    if SNAP not in sys.path:
        sys.path.insert(0, SNAP)
    b = load_mod(PATCH_B, "si_phase_b")
    fbase = load_mod(PATCH_F, "si_inplace_f")     # Phase F sliced decoder (baseline)
    g = load_mod(GDIR / "stream_inference_inplace.py", "si_inplace_g")  # Phase G

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

    # ===== prefill numdiff reference (SDPA dense) — BEFORE installing flex =====
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

    # capture the TRUE original attn forward ONCE (before any flex install), so the
    # F->G swap is bulletproof (chaining _flex_orig_forward breaks across installs).
    inner = model.model if hasattr(model, "model") else model
    AttnClass = type(inner.layers[0].self_attn)
    TRUE_ORIG_FORWARD = AttnClass.forward

    # ===== baseline_repro (Phase F sliced-valid-region in-place flex = 56% copy) ====
    print("\n=== baseline_repro (Phase F sliced in-place flex) ===", flush=True)
    fbase.install_flex_attention(model)  # installs the flex patch (Phase F variant)
    for k, _ in enumerate(fbase.generate(model, tok, PROMPTS[0], silence_token,
                                         max_rows=60, warm_start=False, temperature=0.0,
                                         max_context_rows=512)):
        if k >= 60:
            break
    torch.cuda.synchronize()
    bl_multi = bench(model, tok, silence_token, fbase.generate, [1, 2], "multi_stream",
                     256, max_context_rows=512)
    # capture Phase F's 20-row reference decode NOW (F installed) for the numdiff. F is
    # bit-identical to Phase E (Phase F log) and uses the same flex kernel, so it is a
    # valid decode reference; comparing G vs F directly tests the future-mask + offset.
    rows_ref = run_rows(fbase.generate, model, tok, silence_token, PROMPTS[0], 20,
                        max_context_rows=512)
    (OUT_DIR / "baseline_repro.json").write_text(json.dumps({
        "mode": "baseline_repro_phase_f_sliced_inplace_flex",
        "harness": "5 prompts x 256 productive tokens, greedy T=0, Phase F in-place "
                   "cache returning a valid-region SLICE (the .contiguous()-on-slice "
                   "path that profiles at 56% copy).",
        "multi_stream": bl_multi,
        "phase_f_published_multi_tok_s_256": 4.73}, indent=2))

    # ===== baseline profile (confirm copy ~56% on the Phase F sliced path) =====
    print("\n=== baseline_profile (Phase F sliced row) ===", flush=True)
    bp = profile_row(model, tok, silence_token, fbase.generate,
                     "baseline_repro_profile.txt", "phase_f_sliced", max_context_rows=512)
    (OUT_DIR / "baseline_profile_summary.json").write_text(json.dumps(bp, indent=2))

    # ----- swap to Phase G flex patch (full-buffer, future-mask, no .contiguous) ----
    # Restore the TRUE original forward, clear the install flag, then install G clean.
    print("\n=== installing Phase G full-buffer flex patch ===", flush=True)
    AttnClass.forward = TRUE_ORIG_FORWARD
    AttnClass._flex_installed = False
    g.install_flex_attention(model)

    # ===== prefill numdiff: Phase G full-buffer flex vs SDPA dense =====
    block_mask = g.build_block_mask(C, q_len=N, kv_len=N, q_offset=0, kv_valid=N, device=device)
    with torch.no_grad():
        of = model(input_ids=input_ids,
                   attention_mask={"full_attention": block_mask, "sliding_attention": block_mask},
                   position_ids=position_ids, use_cache=False, channel_ids=channel_ids)
    logits_flex = of.logits[0].float()
    diff = (logits_dense - logits_flex).abs()
    am_d = logits_dense.argmax(-1); am_f = logits_flex.argmax(-1)
    flips = (am_d != am_f)
    numdiff = {
        "prefill": {
            "context": f"{n_prefill} prefill rows x {C} = {N} tokens, full prefill, "
                       f"SDPA(repeat_kv) vs Phase G FlexAttention(enable_gqa, full-buffer).",
            "logits_max_abs_diff": diff.max().item(),
            "logits_mean_abs_diff": diff.mean().item(),
            "argmax_flips": int(flips.sum().item()),
            "argmax_positions": int(flips.numel()),
            "argmax_flip_rate": flips.float().mean().item()}}
    if flips.any():
        fidx = flips.nonzero(as_tuple=True)[0]
        gaps = []
        for pos in fidx.tolist()[:30]:
            t2 = logits_dense[pos].topk(2).values
            gaps.append((t2[0] - t2[1]).item())
        numdiff["prefill"]["flipped_top1_top2_gaps"] = gaps
        numdiff["prefill"]["flipped_max_gap"] = max(gaps)

    # ===== decode numdiff: Phase G full-buffer vs Phase F sliced flex (reference) =====
    # The make-or-break risks are (a) a WRONG in-place write offset, (b) a future-mask
    # that LEAKS unwritten KV. Either diverges from the reference at a wide logit gap.
    # Reference = Phase F sliced-decode rows captured above (F is bit-identical to E and
    # uses the same flex kernel; F vs G differ ONLY in slice-vs-full-buffer+future-mask,
    # so identical rows prove the future-mask + offset are correct).
    print("\n=== decode numdiff (Phase G full-buffer vs Phase F sliced flex, 20 rows) ===", flush=True)
    rows_g = run_rows(g.generate, model, tok, silence_token, PROMPTS[0], 20,
                      max_context_rows=512)
    rows_e = rows_ref
    n_cmp = min(len(rows_e), len(rows_g))
    mismatch_rows = [i for i in range(n_cmp) if rows_e[i] != rows_g[i]]
    numdiff["decode_vs_reference"] = {
        "note": "Greedy T=0, 20 decode rows. Phase G full-buffer+future-mask flex vs "
                "Phase F sliced-valid-region flex (same kernel; differ only in "
                "slice-vs-full-buffer+future-mask). Identical rows => in-place write "
                "offset AND future-mask both correct (a leak or wrong offset diverges).",
        "reference": "phase_f_sliced_flex",
        "rows_compared": n_cmp,
        "mismatch_rows": mismatch_rows,
        "n_mismatch_rows": len(mismatch_rows),
        "all_rows_identical": len(mismatch_rows) == 0,
    }
    if mismatch_rows:
        i = mismatch_rows[0]
        numdiff["decode_vs_reference"]["first_mismatch"] = {
            "row": i, "reference_row": rows_e[i], "phase_g_row": rows_g[i]}
    (OUT_DIR / "numdiff.json").write_text(json.dumps(numdiff, indent=2))
    print(f"[numdiff prefill] max_abs={numdiff['prefill']['logits_max_abs_diff']:.4f} "
          f"flips={numdiff['prefill']['argmax_flips']}/{numdiff['prefill']['argmax_positions']}",
          flush=True)
    print(f"[numdiff decode] {n_cmp} rows, {len(mismatch_rows)} mismatch -> "
          f"identical={len(mismatch_rows)==0}", flush=True)

    # ===== profile one Phase G row (THE KEY DELIVERABLE) =====
    print("\n=== profile_one_row (Phase G full-buffer) — KEY CHECK ===", flush=True)
    pf = profile_row(model, tok, silence_token, g.generate,
                     "profile.txt", "phase_g_full_buffer", max_context_rows=512)
    pf["baseline_copy_share_phase_f"] = bp["aten_copy_self_cpu_share"]
    (OUT_DIR / "profile_summary.json").write_text(json.dumps(pf, indent=2))
    print(f"[profile] copy share: baseline(F)={100*bp['aten_copy_self_cpu_share']:.1f}% "
          f"-> Phase G full-buffer={100*pf['aten_copy_self_cpu_share']:.1f}%", flush=True)

    # ===== throughput by context (Phase G full-buffer flex) =====
    print("\n=== throughput_by_context (Phase G full-buffer flex) ===", flush=True)
    tw = time.perf_counter(); cnt = 0; fr = None; twf = time.perf_counter()
    for ri, row, isp in g.generate(model, tok, PROMPTS[0], silence_token,
                                   max_rows=60, warm_start=False, temperature=0.0,
                                   max_context_rows=512):
        if not isp and fr is None:
            torch.cuda.synchronize(); fr = time.perf_counter() - twf
        cnt += 1
        if cnt >= 60:
            break
    torch.cuda.synchronize()
    warm_s = time.perf_counter() - tw
    print(f"[G warmup] {cnt} rows in {warm_s:.1f}s; first-row {fr:.1f}s", flush=True)

    by_ctx = {}
    for n in CTX:
        npr = N_PROMPTS_AT.get(n, 5)
        mcr = MAX_ROWS_FOR.get(n, n + 1000)
        by_ctx[str(n)] = bench(model, tok, silence_token, g.generate, [1, 2],
                               "multi_stream", n, n_prompts=npr, max_context_rows=mcr)
        (OUT_DIR / "throughput_by_context.json").write_text(json.dumps({
            "mode": "full_buffer_flex_gqa_blockmask_future_mask",
            "harness": "FlexAttention enable_gqa over FULL pre-grown KV buffer with a "
                       "BlockMask future-mask (kv_idx<cursor+C), NO slice / NO contiguous, "
                       "multi-stream packing-2 [1,2], greedy T=0.",
            "warmup_seconds": warm_s, "first_row_seconds": fr,
            "max_context_rows_per_ctx": MAX_ROWS_FOR,
            "by_context": by_ctx}, indent=2))

    print("\n[done] all stages complete.", flush=True)


if __name__ == "__main__":
    main()
