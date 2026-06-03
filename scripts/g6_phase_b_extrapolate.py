#!/usr/bin/env python3
"""Phase B L4 wall-clock extrapolation (B8).

Reads Phase B throughput JSONs and reports the L4 estimate at:
  - phase_b_tensorize_only
  - phase_b_compile_default (= max post-Phase-B throughput we got)

For each, both:
  - optimistic constant-TPS
  - realistic KV-cache linear-growth, reusing the Phase A 2.4 us/row slope (a
    placeholder — Phase B didn't fit a new slope; the per-row time is dominated
    by Python/launch overhead, not attention, so the slope is roughly the same).
"""
from __future__ import annotations

import json
from pathlib import Path

OUT_DIR = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_b")


def fmt_dur(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}min"
    if seconds < 86400:
        return f"{seconds/3600:.1f}h"
    days = seconds / 86400
    return f"{days:.2f}d ({days/7:.2f}w)"


def estimate(single_tps, multi_tps, single_rps, multi_rps,
             # L4 plan
             n_tasks=200, n_seeds=9, traj_tokens=8000,
             n_single_conditions=4, n_multi_conditions=1):
    rollouts_per_condition = n_tasks * n_seeds  # 1800

    # ---- Optimistic: constant-TPS ----
    # Single conditions emit `traj_tokens` Output tokens at single_tps.
    # Multi (D) condition emits the same number of Output tokens but the
    # decoder simultaneously emits Analytical; bench reports combined
    # tokens/sec for [1,2] so per-Output rate is multi_rps (rows/s) when the
    # Output channel is non-silent ~90% of the time, ≈ multi_rps.
    per_traj_single = traj_tokens / single_tps
    output_tps_D = multi_rps  # ~1 Output tok/row
    per_traj_multi = traj_tokens / output_tps_D
    eval_single = rollouts_per_condition * per_traj_single * n_single_conditions
    eval_multi = rollouts_per_condition * per_traj_multi * n_multi_conditions
    eval_optim = eval_single + eval_multi

    # ---- Realistic: linear-KV-cache slope (Phase A placeholder) ----
    # Phase A diagnosis: "2.4 µs/row slope" is mentioned in the plan. Below
    # we use Phase A's empirical 2-point fit from the row=20 (~0.48 s/row)
    # vs row=270 (~0.81 s/row) sanity check. tau_a = 2*(0.81-0.48)/270 =
    # 0.00244 s/(row^2) — i.e. 2.44 ms/row extra per cache-row, NOT µs.
    # That's the "2.4 …/row" referenced in the plan.
    tau_a_phaseA = 2 * (0.81 - 0.48) / 270  # 0.00244 s per row per cache-row
    # In Phase B, steady-state row time at row=10 is ~0.36 s; in Phase A it
    # was ~0.40s. Take τ_0 from row=10 directly (per-row times in compile
    # warmup were 0.36 s after the row-3 recompile).
    tau_0_phaseB = 1.0 / multi_rps - 0  # avg row time ~ 0.36s

    # Trajectory has N rows ~= traj_tokens / (single_tps / single_rps)
    # (the ratio is "rows per productive Output token" = ~272/256 = 1.06).
    rows_per_tok = single_rps / single_tps
    n_rows = int(traj_tokens / (single_tps / single_rps))  # ~ traj_tokens * 1.06
    # Time = int_0^N (tau_0 + tau_a * n) dn = tau_0*N + tau_a*N^2/2
    traj_kvc_s = tau_0_phaseB * n_rows + tau_a_phaseA * n_rows ** 2 / 2
    eval_real = (rollouts_per_condition * traj_kvc_s *
                 (n_single_conditions + n_multi_conditions))
    return {
        "single_tps": single_tps,
        "multi_tps": multi_tps,
        "single_rps": single_rps,
        "multi_rps": multi_rps,
        "n_rows_per_traj": n_rows,
        "tau_0_s": tau_0_phaseB,
        "tau_a_s_per_row_per_cache_row": tau_a_phaseA,
        "per_traj_optim_single_s": per_traj_single,
        "per_traj_optim_multi_s": per_traj_multi,
        "per_traj_real_s": traj_kvc_s,
        "eval_optim_seconds": eval_optim,
        "eval_real_seconds": eval_real,
        "eval_optim_weeks": eval_optim / (7 * 86400),
        "eval_real_weeks": eval_real / (7 * 86400),
    }


def main():
    phase_a_single = 2.36
    phase_a_multi = 4.52
    phase_a_single_rps = 2.51
    phase_a_multi_rps = 2.54

    runs = {}
    # Phase A reproduction (baseline_b)
    bl = json.loads((OUT_DIR / "baseline_b.json").read_text())
    runs["phase_a_repro"] = {
        "single_tps": bl["single_stream"]["mean_productive_tokens_per_sec"],
        "multi_tps": bl["multi_stream"]["mean_productive_tokens_per_sec"],
        "single_rps": bl["single_stream"]["mean_rows_per_sec"],
        "multi_rps": bl["multi_stream"]["mean_rows_per_sec"],
    }
    # Phase B tensorize-only
    t = json.loads((OUT_DIR / "throughput_tensorize_only.json").read_text())
    runs["tensorize_only"] = {
        "single_tps": t["single_stream"]["mean_productive_tokens_per_sec"],
        "multi_tps": t["multi_stream"]["mean_productive_tokens_per_sec"],
        "single_rps": t["single_stream"]["mean_rows_per_sec"],
        "multi_rps": t["multi_stream"]["mean_rows_per_sec"],
    }
    # Phase B + torch.compile default
    cpath = OUT_DIR / "throughput_compile_default.json"
    if cpath.exists():
        c = json.loads(cpath.read_text())
        runs["compile_default"] = {
            "single_tps": c["single_stream"]["mean_productive_tokens_per_sec"],
            "multi_tps": c["multi_stream"]["mean_productive_tokens_per_sec"],
            "single_rps": c["single_stream"]["mean_rows_per_sec"],
            "multi_rps": c["multi_stream"]["mean_rows_per_sec"],
        }
    # Phase B + reduce-overhead, if present
    rpath = OUT_DIR / "throughput_compile_reduce_overhead.json"
    if rpath.exists():
        r = json.loads(rpath.read_text())
        runs["compile_reduce_overhead"] = {
            "single_tps": r["single_stream"]["mean_productive_tokens_per_sec"],
            "multi_tps": r["multi_stream"]["mean_productive_tokens_per_sec"],
            "single_rps": r["single_stream"]["mean_rows_per_sec"],
            "multi_rps": r["multi_stream"]["mean_rows_per_sec"],
        }

    out = {
        "phase_a_published": {
            "single_tps": phase_a_single, "multi_tps": phase_a_multi,
            "single_rps": phase_a_single_rps, "multi_rps": phase_a_multi_rps,
        },
        "runs": {},
    }
    for name, v in runs.items():
        est = estimate(v["single_tps"], v["multi_tps"], v["single_rps"], v["multi_rps"])
        out["runs"][name] = est
        print(f"\n=== {name} ===")
        print(f"  single {v['single_tps']:.2f} tok/s, multi {v['multi_tps']:.2f} tok/s")
        print(f"  per-traj optimistic: single {fmt_dur(est['per_traj_optim_single_s'])}, "
              f"multi {fmt_dur(est['per_traj_optim_multi_s'])}")
        print(f"  per-traj realistic (KV-aware): {fmt_dur(est['per_traj_real_s'])}")
        print(f"  L4 eval optimistic: {fmt_dur(est['eval_optim_seconds'])} "
              f"({est['eval_optim_weeks']:.2f} weeks)")
        print(f"  L4 eval realistic:  {fmt_dur(est['eval_real_seconds'])} "
              f"({est['eval_real_weeks']:.2f} weeks)")

    (OUT_DIR / "extrapolation.json").write_text(json.dumps(out, indent=2))
    print(f"\n[done] wrote {OUT_DIR}/extrapolation.json")


if __name__ == "__main__":
    main()
