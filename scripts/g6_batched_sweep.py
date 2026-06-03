#!/usr/bin/env python3
"""g6 batched decode sweep — does batching independent sequences amortize the
16 GB weight read and push BW toward the 273 GB/s roofline?

The microbench (runs/g6_microbench) proved single-sequence decode is matmul/
weight-read bound (92% of GPU time) at only 39.8 GB/s = 15% of the 273 GB/s
roofline — i.e. batch-starved (a decode "row" is one task's 10 channels). The
L4 eval is a throughput workload (~900 independent trajectories), so the
standard lever is BATCHED decode: advance B independent task-streams in
parallel so the weight read amortizes across B*C tokens.

This drives model.forward directly in a manual batched decode loop (we control
the batch dim). Each step: [B, C] token ids, batched DynamicCache, batched
cross-stream mask [B,1,C,total], greedy argmax. BF16, no_grad. cuda.Event for
per-step GPU time, wall-clock for aggregate throughput.

Usage: g6_batched_sweep.py <ctx> <out.json> <B1,B2,...>
  e.g. g6_batched_sweep.py 512  runs/g6_batched/sweep_ctx512.json  1,4,8,16,32,64
"""
from __future__ import annotations
import json, os, sys, time, traceback
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.cache_utils import DynamicCache

REVISION = "54c7451bfcccecc233fad91affa68563d1de9d66"
SNAP = os.path.expanduser(
    f"~/.cache/huggingface/hub/models--JonasGeiping--stream-qwen3-8b/snapshots/{REVISION}")

DEV = "cuda"
C = 10                     # channels per task-stream (one decode "row")
PRODUCTIVE = 2             # Output + Analytical channels (packing-2 convention)
N_WARMUP = 12
N_TIMED = 25
BW_ROOFLINE = 273.0        # GB/s unified-memory roofline

torch.manual_seed(0)


def sync():
    torch.cuda.synchronize()


def evt():
    return torch.cuda.Event(enable_timing=True)


def load_model():
    print("[load] loading stock Qwen3-8B (sdpa) ...", flush=True)
    t0 = time.time()
    cfg = AutoConfig.from_pretrained(SNAP, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        SNAP, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(DEV).eval()
    print(f"[load] done in {time.time()-t0:.1f}s", flush=True)
    return model, cfg


def weight_bytes(cfg):
    D = cfg.hidden_size; I = cfg.intermediate_size; Hd = cfg.head_dim
    nH = cfg.num_attention_heads; nKV = cfg.num_key_value_heads
    per_layer = (D*nH*Hd + D*nKV*Hd + D*nKV*Hd + nH*Hd*D  # q,k,v,o
                 + D*I + D*I + I*D)                        # gate,up,down
    return per_layer * cfg.num_hidden_layers * 2  # bf16


def make_batched_cache(model, cfg, ctx_len, B):
    """Prefill a DynamicCache with [B, ctx_len] random tokens -> batched KV."""
    cache = DynamicCache()
    ids = torch.randint(0, cfg.vocab_size, (B, ctx_len), device=DEV)
    pos = torch.arange(ctx_len, device=DEV).unsqueeze(0).expand(B, -1)
    causal = torch.triu(torch.full((ctx_len, ctx_len), -1e4, device=DEV,
                                   dtype=torch.bfloat16), 1)
    cmask = causal.view(1, 1, ctx_len, ctx_len).expand(B, 1, -1, -1).contiguous()
    channel_ids = (torch.arange(C, device=DEV)
                   .repeat((ctx_len + C - 1) // C)[:ctx_len]
                   .unsqueeze(0).expand(B, -1).contiguous())
    with torch.no_grad():
        model.model(
            input_ids=ids,
            attention_mask={"full_attention": cmask, "sliding_attention": cmask},
            position_ids=pos,
            past_key_values=cache,
            use_cache=True,
            channel_ids=channel_ids,
        )
    return cache


def clone_cache(cache):
    new = DynamicCache()
    for li in range(len(cache.layers)):
        new.update(cache.layers[li].keys.clone(),
                   cache.layers[li].values.clone(), li, {})
    return new


def trim_cache(cache, keep_len):
    """Truncate each layer's K/V back to keep_len along the sequence dim.
    A decode step appends C tokens; we pop them so the cache is reusable for the
    next timed step WITHOUT a full-cache clone (the clone is O(B*ctx) and was
    dominating wall time at large B). Slicing is a cheap view + contiguous of
    only the kept region (same length every step, so per-step work is constant).
    """
    for li in range(len(cache.layers)):
        cache.layers[li].keys = cache.layers[li].keys[:, :, :keep_len, :].contiguous()
        cache.layers[li].values = cache.layers[li].values[:, :, :keep_len, :].contiguous()


def row_inputs(ctx_len, B):
    """Steady-state batched row: [B,C] tokens, cross-stream mask [B,1,C,total].
    Cross-stream mask: within the C new tokens, channel i may not attend to
    channel j != j (i.e. each channel sees only its own new token + all past).
    """
    total = ctx_len + C
    input_ids = torch.randint(0, 1000, (B, C), device=DEV)
    position_ids = torch.full((B, C), ctx_len, device=DEV, dtype=torch.long)
    channel_ids = torch.arange(C, device=DEV).unsqueeze(0).expand(B, -1).contiguous()
    # base mask [C, total]: rows attend to all past (0..ctx_len), and only own
    # column among the C new ones.
    m = torch.zeros(C, total, device=DEV, dtype=torch.bfloat16)
    block = torch.full((C, C), -1e4, device=DEV, dtype=torch.bfloat16)
    block.fill_diagonal_(0.0)
    m[:, ctx_len:] = block
    mask = m.view(1, 1, C, total).expand(B, 1, -1, -1).contiguous()
    return input_ids, position_ids, channel_ids, mask


def sweep(model, cfg, ctx_len, batch_sizes):
    wbytes = weight_bytes(cfg)
    wgb = wbytes / 1e9
    results = {}
    for B in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"\n[B={B} ctx={ctx_len}] building cache ...", flush=True)
        try:
            base = make_batched_cache(model, cfg, ctx_len, B)
            input_ids, position_ids, channel_ids, mask = row_inputs(ctx_len, B)
        except torch.cuda.OutOfMemoryError as e:
            print(f"[B={B}] OOM during cache build: {e}", flush=True)
            results[str(B)] = {"oom": True, "stage": "cache_build"}
            torch.cuda.empty_cache()
            break

        # Persistent working cache: clone base ONCE, then each step appends C
        # tokens (forward) and trims back to ctx_len. No per-step full clone.
        work = clone_cache(base)

        def one_step():
            with torch.no_grad():
                model(
                    input_ids=input_ids,
                    attention_mask={"full_attention": mask,
                                    "sliding_attention": mask},
                    position_ids=position_ids,
                    past_key_values=work,
                    use_cache=True,
                    channel_ids=channel_ids,
                )
            trim_cache(work, ctx_len)

        def trim_only():
            # re-grow by C then trim, to measure the trim's per-step cost alone
            trim_cache(work, ctx_len)

        try:
            # warmup
            for _ in range(N_WARMUP):
                one_step()
            sync()

            # trim-only cost (probe artifact: real decode keeps tokens). Measure
            # by timing a no-op trim on the already-ctx_len cache.
            trim_times = []
            for _ in range(8):
                s, e = evt(), evt(); sync(); s.record(); trim_only(); e.record(); sync()
                trim_times.append(s.elapsed_time(e))
            trim_ms = sorted(trim_times)[len(trim_times)//2]

            # GPU-event per-step (forward incl the in-loop trim)
            gpu_times = []
            for _ in range(N_TIMED):
                s, e = evt(), evt(); sync(); s.record(); one_step(); e.record(); sync()
                gpu_times.append(s.elapsed_time(e))
            gpu_times.sort()
            gpu_step_ms = gpu_times[len(gpu_times)//2] - trim_ms

            # wall-clock aggregate: time a contiguous block of N_TIMED steps
            sync()
            t0 = time.perf_counter()
            for _ in range(N_TIMED):
                one_step()
            sync()
            wall_total_s = time.perf_counter() - t0
            wall_step_s = wall_total_s / N_TIMED
            wall_step_s_net = wall_step_s - trim_ms / 1000.0
            clone_ms = trim_ms  # keep field name for downstream compat

            peak_mem = torch.cuda.max_memory_allocated()

            # throughput: B sequences advance C tokens each per step; productive
            # = PRODUCTIVE of C channels per task.
            all_tok_per_step = B * C
            prod_tok_per_step = B * PRODUCTIVE
            agg_all_tps = all_tok_per_step / wall_step_s_net
            agg_prod_tps = prod_tok_per_step / wall_step_s_net

            # achieved weight bandwidth: weights read once per step (one forward
            # over all B*C tokens), gpu_step_ms is per-step GPU time.
            achieved_bw = wgb / (gpu_step_ms / 1000.0)

            results[str(B)] = {
                "B": B,
                "gpu_step_ms": gpu_step_ms,
                "wall_step_ms_net": wall_step_s_net * 1000.0,
                "wall_step_ms_raw": wall_step_s * 1000.0,
                "clone_ms": clone_ms,
                "agg_all_tok_s": agg_all_tps,
                "agg_productive_tok_s": agg_prod_tps,
                "peak_mem_gb": peak_mem / 1e9,
                "achieved_bw_gb_s": achieved_bw,
                "bw_efficiency": achieved_bw / BW_ROOFLINE,
                "tokens_per_step_all": all_tok_per_step,
                "tokens_per_step_productive": prod_tok_per_step,
            }
            print(f"[B={B}] gpu_step={gpu_step_ms:.1f}ms wall_step={wall_step_s_net*1000:.1f}ms "
                  f"agg_all={agg_all_tps:.1f}tok/s agg_prod={agg_prod_tps:.1f}tok/s "
                  f"peak={peak_mem/1e9:.1f}GB bw={achieved_bw:.1f}GB/s "
                  f"({achieved_bw/BW_ROOFLINE*100:.0f}%)", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            print(f"[B={B}] OOM during timing: {e}", flush=True)
            results[str(B)] = {"oom": True, "stage": "timing",
                               "peak_mem_gb": torch.cuda.max_memory_allocated()/1e9}
            del base
            try:
                del work
            except NameError:
                pass
            torch.cuda.empty_cache()
            break
        del base, work
        torch.cuda.empty_cache()
    return {
        "ctx_len": ctx_len,
        "weight_gb": wgb,
        "C": C, "productive": PRODUCTIVE,
        "roofline_gb_s": BW_ROOFLINE,
        "by_batch": results,
    }


def main():
    ctx_len = int(sys.argv[1])
    out_path = Path(sys.argv[2])
    batch_sizes = [int(x) for x in sys.argv[3].split(",")]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    assert torch.cuda.is_available()
    torch.set_grad_enabled(False)
    print(f"[env] torch {torch.__version__} device {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[env] free/total GB: {[round(x/1e9,1) for x in torch.cuda.mem_get_info()]}", flush=True)
    model, cfg = load_model()
    print(f"[weights] total matmul weight bytes = {weight_bytes(cfg)/1e9:.2f} GB", flush=True)

    res = sweep(model, cfg, ctx_len, batch_sizes)
    out_path.write_text(json.dumps(res, indent=2))
    print(f"\n[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
