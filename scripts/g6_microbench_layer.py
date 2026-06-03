#!/usr/bin/env python3
"""g6 microbench — clean GPU-time decomposition with torch.cuda.Event.

CUPTI/CUDA-event profiling via torch.profiler is broken on this aarch64 GB10 box,
so EVERY GPU-time number here comes from torch.cuda.Event(enable_timing=True) with
explicit torch.cuda.synchronize() barriers. No profiler is used for timing.

Decoder substrate: the STOCK Qwen3 model (SDPA attention, DynamicCache) from the
local HF cache. We replicate the Streams steady-state decode row exactly:
  - input_ids [1, C] (C=10 channels, one token per stream)
  - growing KV cache (DynamicCache) -> the unit Streams calls a "row"
  - cross-stream additive mask [1,1,C,total]
The brief states the conclusion should hold for stock and the patched in-place-flex
decoder alike; stock is simpler and avoids compile/BlockMask artifacts muddying the
per-stage clock. Choice documented in the report.

Tasks: M1 whole-row, M2 per-stage layer breakdown, M3 dispatch-vs-GPU gap,
M4 matmul-only floor + bandwidth. Idempotent; writes JSON under runs/g6_microbench.
"""
from __future__ import annotations
import json, os, sys, time
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
OUT = Path("/home/ianbarber/Projects/Streams/runs/g6_microbench")
OUT.mkdir(parents=True, exist_ok=True)

DEV = "cuda"
C = 10                 # channels per row (steady-state Streams row width)
CONTEXTS = [512, 4096] # cached KV length before the timed row
N_WARMUP = 15
N_TIMED = 30

torch.manual_seed(0)


def evt():
    return torch.cuda.Event(enable_timing=True)


def sync():
    torch.cuda.synchronize()


class Timer:
    """cuda.Event timer; returns mean ms over n reps of fn (each rep synchronized)."""
    @staticmethod
    def time(fn, n_warmup, n_timed):
        for _ in range(n_warmup):
            fn()
        sync()
        times = []
        for _ in range(n_timed):
            s, e = evt(), evt()
            sync()
            s.record()
            fn()
            e.record()
            sync()
            times.append(s.elapsed_time(e))
        times.sort()
        return {
            "mean_ms": sum(times) / len(times),
            "median_ms": times[len(times) // 2],
            "min_ms": times[0],
            "max_ms": times[-1],
            "n": len(times),
        }


def load_model():
    print("[load] loading stock Qwen3 8B (sdpa) ...", flush=True)
    t0 = time.time()
    cfg = AutoConfig.from_pretrained(SNAP, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        SNAP, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(DEV).eval()
    print(f"[load] done in {time.time()-t0:.1f}s", flush=True)
    return model, cfg


def make_cache(model, cfg, ctx_len):
    """Build a DynamicCache pre-filled with ctx_len tokens by running a prefill."""
    cache = DynamicCache()
    ids = torch.randint(0, cfg.vocab_size, (1, ctx_len), device=DEV)
    pos = torch.arange(ctx_len, device=DEV).unsqueeze(0)
    cmask = torch.zeros(1, 1, ctx_len, ctx_len, device=DEV, dtype=torch.bfloat16)
    # causal
    causal = torch.triu(torch.full((ctx_len, ctx_len), -1e4, device=DEV, dtype=torch.bfloat16), 1)
    cmask[0, 0] = causal
    channel_ids = torch.arange(C, device=DEV).unsqueeze(0).repeat(1, (ctx_len + C - 1) // C)[:, :ctx_len]
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


def row_inputs(ctx_len):
    """Build the steady-state row inputs: [1,C] tokens, cross-stream mask over total."""
    total = ctx_len + C
    input_ids = torch.randint(0, 1000, (1, C), device=DEV)
    position_ids = torch.full((1, C), ctx_len, device=DEV, dtype=torch.long)
    channel_ids = torch.arange(C, device=DEV).unsqueeze(0)
    mask = torch.zeros(1, 1, C, total, device=DEV, dtype=torch.bfloat16)
    for i in range(C):
        for j in range(C):
            if i != j:
                mask[0, 0, i, ctx_len + j] = -1e4
    return input_ids, position_ids, channel_ids, mask


def clone_cache(cache):
    """Deep copy a DynamicCache so repeated timed rows don't grow it."""
    new = DynamicCache()
    for li in range(len(cache.layers)):
        k = cache.layers[li].keys
        v = cache.layers[li].values
        new.update(k.clone(), v.clone(), li, {})
    return new


# ---------------- M1: whole-row wall time + GPU-event ----------------
def m1_whole_row(model, cfg):
    print("[M1] whole-row timing", flush=True)
    res = {}
    for ctx in CONTEXTS:
        base = make_cache(model, cfg, ctx)
        input_ids, position_ids, channel_ids, mask = row_inputs(ctx)

        def one_row():
            cache = clone_cache(base)
            with torch.no_grad():
                model(
                    input_ids=input_ids,
                    attention_mask={"full_attention": mask, "sliding_attention": mask},
                    position_ids=position_ids,
                    past_key_values=cache,
                    use_cache=True,
                    channel_ids=channel_ids,
                )

        # GPU-event timing (includes the clone overhead; we subtract clone separately)
        def clone_only():
            clone_cache(base)

        clone_t = Timer.time(clone_only, 5, 15)
        gpu_t = Timer.time(one_row, N_WARMUP, N_TIMED)

        # wall-clock (python perf_counter, with sync) over the same one_row
        for _ in range(5):
            one_row()
        sync()
        walls = []
        for _ in range(N_TIMED):
            sync()
            t0 = time.perf_counter()
            one_row()
            sync()
            walls.append((time.perf_counter() - t0) * 1000.0)
        walls.sort()
        wall_mean = sum(walls) / len(walls)

        res[str(ctx)] = {
            "gpu_event_ms_incl_clone": gpu_t["mean_ms"],
            "clone_ms": clone_t["mean_ms"],
            "gpu_event_ms_row": gpu_t["mean_ms"] - clone_t["mean_ms"],
            "wall_ms_incl_clone": wall_mean,
            "wall_ms_row": wall_mean - clone_t["mean_ms"],
            "gpu_event_detail": gpu_t,
        }
        print(f"  ctx={ctx}: gpu_row={res[str(ctx)]['gpu_event_ms_row']:.2f}ms "
              f"wall_row={res[str(ctx)]['wall_ms_row']:.2f}ms clone={clone_t['mean_ms']:.2f}ms",
              flush=True)
        del base
        torch.cuda.empty_cache()
    (OUT / "whole_row.json").write_text(json.dumps(res, indent=2))
    return res


# ---------------- M2: per-stage layer breakdown ----------------
def m2_layer_breakdown(model, cfg):
    print("[M2] per-stage layer breakdown", flush=True)
    from importlib import import_module
    mq = sys.modules[type(model.model.layers[0]).__module__]
    apply_rope = mq.apply_rotary_pos_emb

    res = {}
    for ctx in CONTEXTS:
        base = make_cache(model, cfg, ctx)
        input_ids, position_ids, channel_ids, mask = row_inputs(ctx)

        layer = model.model.layers[0]
        attn = layer.self_attn
        mlp = layer.mlp

        # Build the hidden_states input to layer 0 (post-embedding) for this row.
        # Mirrors Qwen3Model.forward: embed_tokens + additive channel_embedding, then RoPE on local_y.
        with torch.no_grad():
            hs = model.model.embed_tokens(input_ids)  # [1,C,D]
            if getattr(model.model, "channel_embedding_method", "none") != "none":
                ce = model.model.channel_embedding(channel_ids.clamp(0, model.model.num_channels - 1))
                hs = hs + ce
            cos, sin = model.model.rotary_emb(hs, position_ids)
            pos_emb = (cos, sin)

        D = cfg.hidden_size
        Hd = cfg.head_dim
        nH = cfg.num_attention_heads
        nKV = cfg.num_key_value_heads
        hidden_shape = (1, C, -1, Hd)

        # snapshot cache K/V for layer 0 (so KV-write timing is realistic length)
        k0 = base.layers[0].keys.clone()
        v0 = base.layers[0].values.clone()

        # Pre-norm'd input to attention
        with torch.no_grad():
            normed = layer.input_layernorm(hs)

        stages = {}

        # --- input RMSNorm ---
        def s_input_norm():
            layer.input_layernorm(hs)
        stages["input_rmsnorm"] = Timer.time(lambda: s_input_norm(), N_WARMUP, N_TIMED)

        # gate_in_norm (role gating disabled but norm still runs in fwd)
        def s_gate_norm():
            attn.gate_in_norm(normed)
        stages["gate_in_rmsnorm"] = Timer.time(lambda: s_gate_norm(), N_WARMUP, N_TIMED)

        # --- QKV projection (3 matmuls + q/k per-head norm + view/transpose) ---
        def s_qkv():
            with torch.no_grad():
                q = attn.q_norm(attn.q_proj(normed).view(hidden_shape)).transpose(1, 2)
                k = attn.k_norm(attn.k_proj(normed).view(hidden_shape)).transpose(1, 2)
                v = attn.v_proj(normed).view(hidden_shape).transpose(1, 2)
            return q, k, v
        stages["qkv_proj"] = Timer.time(lambda: s_qkv(), N_WARMUP, N_TIMED)
        with torch.no_grad():
            q, k, v = s_qkv()

        # qkv matmul-only (no per-head norm) for matmul-floor cross-check
        def s_qkv_mm():
            with torch.no_grad():
                attn.q_proj(normed); attn.k_proj(normed); attn.v_proj(normed)
        stages["qkv_matmul_only"] = Timer.time(lambda: s_qkv_mm(), N_WARMUP, N_TIMED)

        # --- RoPE apply ---
        def s_rope():
            with torch.no_grad():
                apply_rope(q, k, cos, sin)
        stages["rope_apply"] = Timer.time(lambda: s_rope(), N_WARMUP, N_TIMED)
        with torch.no_grad():
            qr, kr = apply_rope(q, k, cos, sin)

        # --- KV cache write (DynamicCache.update -> torch.cat append) ---
        def s_kv_write():
            cache = DynamicCache()
            cache.update(k0.clone(), v0.clone(), 0, {})  # seed full-length
            with torch.no_grad():
                cache.update(kr, v, 0, {})
        stages["kv_write"] = Timer.time(lambda: s_kv_write(), N_WARMUP, N_TIMED)

        # Build full K/V (cached + new) for attention timing
        with torch.no_grad():
            k_full = torch.cat([k0, kr], dim=-2)  # [1,nKV,total,d]
            v_full = torch.cat([v0, v], dim=-2)

        # --- attention (SDPA) incl repeat_kv ---
        sdpa_mod = sys.modules.get(attn.__class__.__module__)
        repeat_kv = getattr(mq, "repeat_kv", None)

        def s_attention():
            with torch.no_grad():
                torch.nn.functional.scaled_dot_product_attention(
                    qr, k_full, v_full, attn_mask=mask, scale=attn.scaling, enable_gqa=True)
        stages["attention_sdpa"] = Timer.time(lambda: s_attention(), N_WARMUP, N_TIMED)

        with torch.no_grad():
            ao = torch.nn.functional.scaled_dot_product_attention(qr, k_full, v_full, attn_mask=mask, scale=attn.scaling, enable_gqa=True)
            ao = ao.transpose(1, 2).reshape(1, C, -1).contiguous()

        # --- O projection ---
        def s_oproj():
            with torch.no_grad():
                attn.o_proj(ao)
        stages["o_proj"] = Timer.time(lambda: s_oproj(), N_WARMUP, N_TIMED)

        # --- post-attn RMSNorm ---
        with torch.no_grad():
            after_attn = hs + attn.o_proj(ao)
        def s_postnorm():
            layer.post_attention_layernorm(after_attn)
        stages["post_attn_rmsnorm"] = Timer.time(lambda: s_postnorm(), N_WARMUP, N_TIMED)
        with torch.no_grad():
            mlp_in = layer.post_attention_layernorm(after_attn)

        # --- MLP gate/up matmul ---
        def s_gate_up():
            with torch.no_grad():
                mlp.gate_proj(mlp_in); mlp.up_proj(mlp_in)
        stages["mlp_gate_up"] = Timer.time(lambda: s_gate_up(), N_WARMUP, N_TIMED)
        with torch.no_grad():
            g_ = mlp.gate_proj(mlp_in); u_ = mlp.up_proj(mlp_in)

        # --- SiLU + mul ---
        def s_silu():
            with torch.no_grad():
                mlp.act_fn(g_) * u_
        stages["silu_mul"] = Timer.time(lambda: s_silu(), N_WARMUP, N_TIMED)
        with torch.no_grad():
            inter = mlp.act_fn(g_) * u_

        # --- MLP down matmul ---
        def s_down():
            with torch.no_grad():
                mlp.down_proj(inter)
        stages["mlp_down"] = Timer.time(lambda: s_down(), N_WARMUP, N_TIMED)

        # --- full single-layer forward (ground truth to compare sum-of-stages) ---
        layer_cache_seed = (k0, v0)
        def s_full_layer():
            cache = DynamicCache()
            cache.update(k0.clone(), v0.clone(), 0, {})
            with torch.no_grad():
                layer(
                    hidden_states=hs,
                    attention_mask=mask,
                    position_ids=position_ids,
                    past_key_value=cache,
                    use_cache=True,
                    cache_position=torch.arange(ctx, ctx + C, device=DEV),
                    position_embeddings=pos_emb,
                )
        full_layer = Timer.time(lambda: s_full_layer(), N_WARMUP, N_TIMED)
        # subtract the cache-seed clone overhead (2 clones of full-length K/V)
        def s_seed_only():
            cache = DynamicCache(); cache.update(k0.clone(), v0.clone(), 0, {})
        seed_t = Timer.time(lambda: s_seed_only(), 5, 15)

        stage_sum = sum(v["mean_ms"] for kk, v in stages.items()
                        if kk not in ("qkv_matmul_only",))
        full_layer_row = full_layer["mean_ms"] - seed_t["mean_ms"]

        # matmul stages
        matmul_ms = (stages["qkv_matmul_only"]["mean_ms"] + stages["o_proj"]["mean_ms"]
                     + stages["mlp_gate_up"]["mean_ms"] + stages["mlp_down"]["mean_ms"])
        cast_ms = (stages["input_rmsnorm"]["mean_ms"] + stages["gate_in_rmsnorm"]["mean_ms"]
                   + stages["post_attn_rmsnorm"]["mean_ms"] + stages["rope_apply"]["mean_ms"])
        attn_ms = stages["attention_sdpa"]["mean_ms"]
        kv_ms = stages["kv_write"]["mean_ms"]

        res[str(ctx)] = {
            "stages_ms": {k: v["mean_ms"] for k, v in stages.items()},
            "stages_detail": stages,
            "stage_sum_ms": stage_sum,
            "full_layer_ms": full_layer_row,
            "full_layer_raw_incl_seed": full_layer["mean_ms"],
            "seed_clone_ms": seed_t["mean_ms"],
            "x36_stage_sum_ms": stage_sum * 36,
            "x36_full_layer_ms": full_layer_row * 36,
            "rollup": {
                "matmul_ms": matmul_ms, "matmul_pct": matmul_ms / stage_sum,
                "cast_ms": cast_ms, "cast_pct": cast_ms / stage_sum,
                "attention_ms": attn_ms, "attention_pct": attn_ms / stage_sum,
                "kv_write_ms": kv_ms, "kv_write_pct": kv_ms / stage_sum,
            },
        }
        print(f"  ctx={ctx}: stage_sum={stage_sum:.3f}ms full_layer={full_layer_row:.3f}ms "
              f"x36_sum={stage_sum*36:.1f}ms | matmul={matmul_ms/stage_sum*100:.0f}% "
              f"cast={cast_ms/stage_sum*100:.0f}% attn={attn_ms/stage_sum*100:.0f}% "
              f"kv={kv_ms/stage_sum*100:.0f}%", flush=True)
        del base, k0, v0, k_full, v_full
        torch.cuda.empty_cache()
    (OUT / "layer_breakdown.json").write_text(json.dumps(res, indent=2))
    return res


# ---------------- M3: dispatch-vs-GPU gap ----------------
def m3_dispatch_gap(m1_res, m2_res):
    print("[M3] dispatch vs GPU gap", flush=True)
    res = {}
    for ctx in CONTEXTS:
        c = str(ctx)
        # (a) pure GPU work: sum of per-stage GPU time x36 layers
        gpu_stage_x36 = m2_res[c]["x36_stage_sum_ms"]
        # also: cuda.Event whole-row GPU time (kernels back-to-back, no python gaps measured)
        gpu_event_row = m1_res[c]["gpu_event_ms_row"]
        # (b) whole-row wall-clock (python loop + dispatch + GPU)
        wall_row = m1_res[c]["wall_ms_row"]
        res[c] = {
            "gpu_event_whole_row_ms": gpu_event_row,
            "sum_stage_gpu_x36_ms": gpu_stage_x36,
            "wall_row_ms": wall_row,
            "ratio_wall_over_gpu_event": wall_row / gpu_event_row,
            "ratio_wall_over_stagesum": wall_row / gpu_stage_x36,
            "gap_ms_wall_minus_gpu_event": wall_row - gpu_event_row,
        }
        print(f"  ctx={ctx}: wall={wall_row:.1f}ms gpu_event={gpu_event_row:.1f}ms "
              f"stagesum_x36={gpu_stage_x36:.1f}ms ratio_wall/gpu={wall_row/gpu_event_row:.2f}",
              flush=True)
    (OUT / "dispatch_gap.json").write_text(json.dumps(res, indent=2))
    return res


# ---------------- M4: matmul floor + bandwidth ----------------
def m4_matmul_floor(model, cfg, m2_res):
    print("[M4] matmul floor + bandwidth", flush=True)
    D = cfg.hidden_size; I = cfg.intermediate_size; Hd = cfg.head_dim
    nH = cfg.num_attention_heads; nKV = cfg.num_key_value_heads
    # weight bytes per layer (bf16 = 2 bytes)
    q_w = D * nH * Hd; k_w = D * nKV * Hd; v_w = D * nKV * Hd; o_w = nH * Hd * D
    gate_w = D * I; up_w = D * I; down_w = I * D
    layer_weight_params = q_w + k_w + v_w + o_w + gate_w + up_w + down_w
    layer_weight_bytes = layer_weight_params * 2
    total_weight_bytes = layer_weight_bytes * cfg.num_hidden_layers

    res = {"weights": {
        "layer_weight_params": layer_weight_params,
        "layer_weight_bytes": layer_weight_bytes,
        "total_matmul_weight_bytes_x36": total_weight_bytes,
        "total_matmul_weight_gb": total_weight_bytes / 1e9,
    }, "by_context": {}}

    BW_ROOFLINE = 273.0  # GB/s

    for ctx in CONTEXTS:
        c = str(ctx)
        s = m2_res[c]["stages_ms"]
        # 7 matmuls per layer: qkv(3) + o + gate + up + down
        matmul_layer_ms = (s["qkv_matmul_only"] + s["o_proj"] + s["mlp_gate_up"] + s["mlp_down"])
        matmul_x36_ms = matmul_layer_ms * 36
        # bytes moved by matmuls per row = weights (read once) + activations (small)
        # activation bytes: input/output of each matmul, C tokens
        # dominant term is weights; report weight-bandwidth-achieved.
        gb_moved = total_weight_bytes / 1e9
        achieved_bw = gb_moved / (matmul_x36_ms / 1000.0)
        theoretical_ms = gb_moved / BW_ROOFLINE * 1000.0
        res["by_context"][c] = {
            "matmul_layer_ms": matmul_layer_ms,
            "matmul_x36_ms": matmul_x36_ms,
            "weight_gb_moved": gb_moved,
            "achieved_bandwidth_gb_s": achieved_bw,
            "roofline_gb_s": BW_ROOFLINE,
            "bandwidth_efficiency": achieved_bw / BW_ROOFLINE,
            "theoretical_matmul_ms_at_roofline": theoretical_ms,
        }
        print(f"  ctx={ctx}: matmul_x36={matmul_x36_ms:.1f}ms achieved_bw={achieved_bw:.1f}GB/s "
              f"({achieved_bw/BW_ROOFLINE*100:.0f}% of {BW_ROOFLINE}) theo={theoretical_ms:.1f}ms",
              flush=True)
    (OUT / "matmul_floor.json").write_text(json.dumps(res, indent=2))
    return res


def main():
    assert torch.cuda.is_available()
    torch.set_grad_enabled(False)  # inference-only; avoids autograd graphs skewing RMSNorm/RoPE timing
    print(f"[env] torch {torch.__version__} device {torch.cuda.get_device_name(0)}", flush=True)
    model, cfg = load_model()
    m1 = m1_whole_row(model, cfg)
    m2 = m2_layer_breakdown(model, cfg)
    m3 = m3_dispatch_gap(m1, m2)
    m4 = m4_matmul_floor(model, cfg, m2)
    summary = {"m1_whole_row": m1, "m3_dispatch": m3, "m4_matmul": m4,
               "m2_rollup": {c: m2[c]["rollup"] for c in m2},
               "m2_full": m2}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    write_summary_md(m1, m2, m3, m4)
    print("[done] wrote runs/g6_microbench/*.json + summary.md", flush=True)


def write_summary_md(m1, m2, m3, m4):
    L = []
    L.append("# g6 microbench — clean GPU-time decomposition (torch.cuda.Event)\n")
    L.append("Substrate: STOCK Qwen3-8B, SDPA attention, DynamicCache. Steady-state row = [1,10] (C channels) "
             "through 36 layers with cross-stream mask + growing KV. BF16. All GPU times via cuda.Event "
             "(CUPTI/profiler timing broken on this box).\n")
    for ctx in CONTEXTS:
        c = str(ctx)
        L.append(f"\n## Context {ctx}\n")
        L.append(f"- Whole row: GPU-event **{m1[c]['gpu_event_ms_row']:.1f} ms** vs wall-clock "
                 f"**{m1[c]['wall_ms_row']:.1f} ms**\n")
        r = m2[c]["rollup"]
        L.append(f"- Single layer (sum of stages): {m2[c]['stage_sum_ms']:.2f} ms; "
                 f"full-layer ground truth {m2[c]['full_layer_ms']:.2f} ms; x36 = {m2[c]['x36_stage_sum_ms']:.0f} ms\n")
        L.append(f"- Rollup: matmul {r['matmul_pct']*100:.0f}% | cast(RMSNorm+RoPE) {r['cast_pct']*100:.0f}% | "
                 f"attention {r['attention_pct']*100:.0f}% | KV-write {r['kv_write_pct']*100:.0f}%\n")
        L.append("- Per-stage GPU ms:\n")
        for k, v in m2[c]["stages_ms"].items():
            L.append(f"  - {k}: {v:.4f} ms\n")
        dg = m3[c]
        L.append(f"- Dispatch gap: wall/gpu_event ratio **{dg['ratio_wall_over_gpu_event']:.2f}** "
                 f"(wall {dg['wall_row_ms']:.1f} - gpu_event {dg['gpu_event_whole_row_ms']:.1f} = "
                 f"gap {dg['gap_ms_wall_minus_gpu_event']:.1f} ms)\n")
        mf = m4["by_context"][c]
        L.append(f"- Matmul floor: x36 matmul {mf['matmul_x36_ms']:.1f} ms; achieved BW "
                 f"**{mf['achieved_bandwidth_gb_s']:.1f} GB/s** ({mf['bandwidth_efficiency']*100:.0f}% of 273); "
                 f"roofline-ideal {mf['theoretical_matmul_ms_at_roofline']:.1f} ms\n")
    (OUT / "summary.md").write_text("".join(L))


if __name__ == "__main__":
    main()
