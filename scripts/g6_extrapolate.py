#!/usr/bin/env python3
"""Extrapolate L4 wall-clock from G6 throughput measurements.

Reads runs/g6_throughput/{single,multi}_stream.json and writes
runs/g6_throughput/extrapolation.md.

This is the *honest* version: the eval extrapolation uses the measured TPS
directly (constant-TPS assumption, with a separate note on the KV-cache
linear-growth caveat); the SFT estimate uses a FLOPs-based model since
training and inference have very different cost structures.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT_DIR = Path("/home/ianbarber/Projects/Streams/runs/g6_throughput")


def fmt_dur(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}min"
    if seconds < 86400:
        return f"{seconds/3600:.1f}h"
    return f"{seconds/86400:.2f}d ({seconds/86400/7:.2f}w)"


def main():
    single = json.loads((OUT_DIR / "single_stream.json").read_text())
    multi = json.loads((OUT_DIR / "multi_stream.json").read_text())

    single_tps = single["mean_productive_tokens_per_sec"]
    single_rps = single["mean_rows_per_sec"]
    multi_tps = multi["mean_productive_tokens_per_sec"]
    multi_rps = multi["mean_rows_per_sec"]
    single_peak = single["peak_memory_gb"]
    multi_peak = multi["peak_memory_gb"]

    # L4 plan: 200 tasks x 9 seeds x 5 conditions = 9000 trajectories
    n_tasks = 200
    n_seeds = 9
    traj_tokens = 8000
    n_single_conditions = 4   # A, B, C, C'
    n_multi_conditions = 1    # D
    rollouts_per_condition = n_tasks * n_seeds  # 1800

    # ----- Constant-TPS eval estimate (lower bound) -----
    # NB: this is OPTIMISTIC because attention cost grows linearly with cache.
    output_tps_D = multi_rps  # Output channel produces one tok/row when non-silent.
    per_traj_single_s = traj_tokens / single_tps
    per_traj_D_s = traj_tokens / output_tps_D

    eval_single_s = rollouts_per_condition * per_traj_single_s * n_single_conditions
    eval_D_s = rollouts_per_condition * per_traj_D_s * n_multi_conditions
    eval_total_s = eval_single_s + eval_D_s

    # ----- KV-cache-aware (more realistic) eval estimate -----
    # Model per-row time as tau_0 + tau_a * N (N = cache rows).
    # From bench: 270 rows in 219s avg => avg row time 0.81s.
    # Sanity (no warmup) gave ~0.48 s/row at row<20. Use linear fit through these
    # two points to get tau_0 (intercept at N=0) and tau_a (slope per cache-row).
    # Time for trajectory of N_max rows: integral_0^N (tau_0 + tau_a * n) dn
    # = tau_0 * N_max + tau_a * N_max^2 / 2.
    tau_at_0 = 0.48   # s/row at cache size ~0
    tau_at_270 = 0.81 # average over rows [0, 270]
    # The average over [0, N] of (a + b*n) is a + b*N/2 = tau_at_N
    # so b = 2*(tau_at_N - a) / N at N=270:
    tau_a = 2 * (tau_at_270 - tau_at_0) / 270
    tau_0 = tau_at_0
    # rows for a single-stream trajectory of 8000 Output tokens (~95% non-silent):
    # 270 rows -> 256 prod tokens => ratio 0.95, so 8000 prod tokens => ~8421 rows.
    n_rows_traj = int(traj_tokens / (single_tps / single_rps))  # ~8421
    traj_kvc_s = tau_0 * n_rows_traj + tau_a * n_rows_traj**2 / 2
    eval_total_kvc_s = (
        rollouts_per_condition * traj_kvc_s * n_single_conditions
        + rollouts_per_condition * traj_kvc_s * n_multi_conditions
    )

    # ----- FLOPs-based SFT (training) estimate -----
    # Model: 8B params, BF16. Per-token training FLOPs ~6 * params (fwd 2x params
    # via 2N matmul, bwd 4x params via 2 backward passes through). LoRA rank 128
    # adds <1% to compute (matmuls of 4096 x 128); we ignore the LoRA delta.
    n_params = 8e9
    per_token_train_flops = 6 * n_params  # 4.8e10
    # GB10 BF16 dense: NVIDIA states ~1 PFLOP (1e15 FLOPS) dense BF16 (approx).
    # Realistic MFU with HF Trainer + LoRA at seq 32k: 25-35%. Use 30% as midpoint.
    gb10_peak_bf16_flops = 1.0e15
    mfu = 0.30
    achieved_flops = gb10_peak_bf16_flops * mfu  # 3e14
    n_sft_trajs = 2000
    sft_seq_len = 32_768
    n_epochs = 2
    sft_tokens = n_sft_trajs * sft_seq_len * n_epochs  # 1.31e8
    sft_s = sft_tokens * per_token_train_flops / achieved_flops
    # Sanity: 1.31e8 tokens * 4.8e10 FLOPs / 3e14 FLOPS = 20,950 s = 5.8 hours.

    # ----- Totals -----
    total_optimistic_s = eval_total_s + sft_s
    total_realistic_s = eval_total_kvc_s + sft_s

    threshold_s = 21 * 86400

    lines = []
    lines.append("# G6 → L4 wall-clock extrapolation")
    lines.append("")
    lines.append("## Inputs (from bench)")
    lines.append(f"- Single-stream Output throughput: **{single_tps:.2f} tok/s** "
                 f"(σ {single['std_productive_tokens_per_sec']:.2f}), "
                 f"{single_rps:.2f} rows/s, peak {single_peak:.2f} GB")
    lines.append(f"- Multi-stream (Output+Analytical) combined: **{multi_tps:.2f} tok/s** "
                 f"(σ {multi['std_productive_tokens_per_sec']:.2f}), "
                 f"{multi_rps:.2f} rows/s, peak {multi_peak:.2f} GB")
    lines.append(f"- Packing-factor-2 reality check: multi/single = "
                 f"{multi_tps/single_tps:.2f}× (NOT 2×). Analytical is silent for "
                 f"most rows because only Output has a silence_penalty in the "
                 f"bundled inference loop.")
    lines.append(f"- Measurement window: 256 productive tokens (~270 rows) per prompt.")
    lines.append("")
    lines.append("## L4 evaluation — OPTIMISTIC (constant-TPS)")
    lines.append(f"Plan: {n_tasks} tasks × {n_seeds} seeds × 5 conditions = "
                 f"{n_tasks * n_seeds * 5} trajectories at ~{traj_tokens} Output tokens each.")
    lines.append("")
    lines.append(f"- Per-trajectory, A/B/C/C′ (single): {traj_tokens}/{single_tps:.2f} = "
                 f"{per_traj_single_s:.0f}s = **{fmt_dur(per_traj_single_s)}**")
    lines.append(f"- Per-trajectory, D (multi): {traj_tokens}/{output_tps_D:.2f} = "
                 f"{per_traj_D_s:.0f}s = **{fmt_dur(per_traj_D_s)}**")
    lines.append(f"- 4 single-stream conditions × {rollouts_per_condition} rollouts: "
                 f"**{fmt_dur(eval_single_s)}**")
    lines.append(f"- 1 D condition × {rollouts_per_condition} rollouts: "
                 f"**{fmt_dur(eval_D_s)}**")
    lines.append(f"- **Total L4 eval (optimistic): {fmt_dur(eval_total_s)}**")
    lines.append("")
    lines.append("## L4 evaluation — REALISTIC (linear KV-cache growth)")
    lines.append(f"At the bench window (~270 rows) the avg row time is {tau_at_270:.2f}s; ")
    lines.append(f"at the sanity-test window (~20 rows) it was ~{tau_at_0:.2f}s. ")
    lines.append(f"Linear fit: τ(N) = {tau_0:.2f} + {tau_a:.5f}·N seconds/row.")
    lines.append("")
    lines.append(f"At 8000-token trajectories ({n_rows_traj} rows), total per-trajectory ")
    lines.append(f"time ≈ {tau_0:.2f}·N + {tau_a:.5f}·N²/2 = **{fmt_dur(traj_kvc_s)}** "
                 f"(compared to {fmt_dur(per_traj_single_s)} under constant-TPS).")
    lines.append("")
    lines.append(f"- **Total L4 eval (realistic): {fmt_dur(eval_total_kvc_s)}**")
    lines.append("")
    lines.append("Caveat: the linear-fit is from 2 data points and the slope likely ")
    lines.append("under-represents true cost beyond a few thousand rows (where the ")
    lines.append("quadratic-attention cost truly bites). Treat this as a lower bound on ")
    lines.append("the realistic figure.")
    lines.append("")
    lines.append("## L4 SFT pass — FLOPs-based estimate (more credible)")
    lines.append(f"- Plan: {n_sft_trajs} trajectories × {sft_seq_len} seq-len × "
                 f"{n_epochs} epochs = {sft_tokens:,} tokens.")
    lines.append(f"- Per-token training FLOPs (8B params, fwd+bwd, LoRA rank 128 "
                 f"adds <1%): 6·N = {per_token_train_flops:.1e}.")
    lines.append(f"- GB10 BF16 peak ≈ {gb10_peak_bf16_flops:.0e} FLOPS, assume MFU "
                 f"{mfu*100:.0f}% → {achieved_flops:.1e} FLOPS achieved.")
    lines.append(f"- **Estimated SFT wall-clock: {fmt_dur(sft_s)}** "
                 f"(±2× depending on real MFU).")
    lines.append("")
    lines.append("Note: SFT and eval costs are different physics. SFT batches 32 examples ")
    lines.append("per step at the cost of FLOPs; eval is autoregressive serial decode that ")
    lines.append("is *memory-bandwidth-bound* (KV-cache reads dominate). Don't blend their ")
    lines.append("throughput numbers without care.")
    lines.append("")
    lines.append("## TOTAL L4 wall-clock")
    lines.append(f"- **Optimistic** (constant-TPS eval + flops SFT): "
                 f"{fmt_dur(total_optimistic_s)}")
    lines.append(f"- **Realistic** (KV-aware eval + flops SFT): "
                 f"{fmt_dur(total_realistic_s)}")
    lines.append("")
    lines.append("## Kill-switch check (threshold = 3 weeks = 21 days)")
    lines.append("")
    if total_optimistic_s > threshold_s:
        lines.append(f"**KILL-SWITCH TRIGGERED.**")
        lines.append("")
        lines.append(f"Even the optimistic estimate ({fmt_dur(total_optimistic_s)}) exceeds "
                     f"3 weeks. The realistic estimate ({fmt_dur(total_realistic_s)}) is "
                     f"vastly worse.")
        lines.append("")
        lines.append("### Why")
        lines.append("The bundled `stream_inference.generate()` decoder achieves ~1.2 rows/s "
                     "at 256 productive tokens. The architecture decodes all 10 channels per "
                     "row from a single forward pass, but the bundled implementation has no "
                     "Flash-attention path, no torch.compile, no kernel fusion, and rebuilds "
                     "the attention mask in Python every step. The result is a reference "
                     "implementation, not a production decoder.")
        lines.append("")
        lines.append("Even at the FLOPs/memory-bandwidth ceiling for an 8B model on GB10, "
                     "naive autoregressive decode is ~30–60 tok/s with a real kernel stack. "
                     "We are 30–50× off that.")
        lines.append("")
        lines.append("### Knobs that *don't* fix it (rounded, vs optimistic eval)")
        scenarios_eval = [
            ("Seeds 9 → 6 (eval)",
             eval_total_s * 6/9 + sft_s),
            ("Seeds 9 → 5 (eval)",
             eval_total_s * 5/9 + sft_s),
            ("Held-out skip (200 → 100 tasks)",
             eval_total_s * 0.5 + sft_s),
            ("Trajectory cap 8k → 4k tokens",
             eval_total_s * 0.5 + sft_s),
            ("Drop ablation condition (5 → 4)",
             (eval_single_s * 3/4 + eval_D_s) + sft_s),
            ("All of: seeds 9→5, tasks 200→100, traj 8k→4k",
             eval_total_s * (5/9) * 0.5 * 0.5 + sft_s),
        ]
        for name, new_total in scenarios_eval:
            cut = (total_optimistic_s - new_total) / total_optimistic_s * 100
            verdict = "UNDER 3w" if new_total <= threshold_s else "still over 3w"
            lines.append(f"  - {name}: → {fmt_dur(new_total)} ({cut:.0f}% cut; {verdict})")
        lines.append("")
        lines.append("### Knobs that *would* fix it (decoder engineering)")
        # If we had 30 tok/s instead of 1.16:
        speedup_30 = 30 / single_tps
        eval_at_30 = eval_total_s / speedup_30
        lines.append(f"- **Reach 30 tok/s decode** (~25× speedup over bench): "
                     f"eval drops to {fmt_dur(eval_at_30)}. **Combined L4 ≈ "
                     f"{fmt_dur(eval_at_30 + sft_s)}**, well under 3 weeks.")
        lines.append("  - Routes: torch.compile the model, Flash-attention path for the ")
        lines.append("    block-causal mask (custom kernel), batched per-prompt decode, ")
        lines.append("    eliminate Python mask construction from the per-row hot path, ")
        lines.append("    move from CPU-driven generator to a TorchScript/CUDA-graphs loop.")
        lines.append("- **Reduce L4 scope to L3** (~50 tasks × 6 seeds × 4 conditions = ")
        lines.append("    1200 rollouts vs 9000): with current decoder, eval = "
                     f"{fmt_dur(eval_total_s * 1200 / (n_tasks * n_seeds * 5))} — still ~3 months, ")
        lines.append("    still over budget unless paired with decoder speedup.")
    else:
        lines.append(f"**UNDER BUDGET.**")
    lines.append("")
    lines.append("## Assumptions (review for honesty)")
    lines.append("- Mean trajectory length: 8000 Output-channel tokens (per task spec). ")
    lines.append("  Real SWE-bench Verified trajectories run 1k–24k+; 8k is a midpoint.")
    lines.append("- Throughput measured at temperature=0 with the bundled "
                 "`stream_inference.generate()`; production decoding may use sampling "
                 "(negligible cost diff).")
    lines.append("- Multi-stream productive-tokens here = Output + Analytical combined. ")
    lines.append("  Analytical is silent ~85% of rows because only Output has a "
                 "silence_penalty applied. To realise true packing-factor-2 we would have "
                 "to add a think_silence_penalty for the diagnostic channel (and verify "
                 "downstream quality).")
    lines.append("- D output-channel TPS is taken as `rows/s` (proxy: Output is non-silent ")
    lines.append("  on ~95% of rows once the silence_penalty has ramped up).")
    lines.append("- KV-cache linear-fit uses 2 data points; quadratic effects beyond ~5k ")
    lines.append("  rows will make the realistic figure even worse.")
    lines.append("- SFT FLOPs estimate assumes 30% MFU. Real MFU on GB10 for HF Trainer + ")
    lines.append("  LoRA at 32k context could be 15–40%; corresponding SFT time would scale.")
    lines.append("- Eval includes ONLY model decode. LSP daemon, file I/O, patch ")
    lines.append("  application, pytest runs add ~30–100% overhead. Not modelled here.")
    lines.append("- Single GB10. No parallelism.")
    lines.append("")
    (OUT_DIR / "extrapolation.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_DIR / 'extrapolation.md'}")
    print(f"optimistic: {fmt_dur(total_optimistic_s)}")
    print(f"realistic:  {fmt_dur(total_realistic_s)}")
    print(f"kill-switch: {'TRIGGERED' if total_optimistic_s > threshold_s else 'CLEAR'}")


if __name__ == "__main__":
    main()
