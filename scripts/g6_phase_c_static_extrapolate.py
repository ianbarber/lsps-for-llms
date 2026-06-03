#!/usr/bin/env python3
"""Phase C C7 — L4 wall-clock extrapolation from the context-length curve.

Unlike Phase B (which reused a placeholder slope), this fits the per-row time as
a function of context length directly from throughput_by_context.json, then
integrates over an 8000-token trajectory.

L4 plan: 200 tasks x 9 seeds x 5 conditions x 8000 productive tokens.
  - 4 single-stream conditions (C', baselines): emit 8000 Output tokens.
  - 1 multi-stream condition (D): emit 8000 Output tokens while also producing
    Analytical; per-Output rate ~ rows/s (Output non-silent ~1 tok/row).

We model per-row latency t(n) where n = current context length in *rows*. From
the measured points {256,1024,4096,8192} productive tokens we recover the row
count and mean ms/row at each, then fit t(n) = t0 + tau * n (linear in context),
and integrate over a trajectory of N rows: T_traj = sum_{r=0}^{N} t(r).

Writes runs/g6_phase_c_static/extrapolation.json.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT_DIR = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_c_static")

N_TASKS = 200
N_SEEDS = 9
TRAJ_TOKENS = 8000
N_SINGLE_COND = 4
N_MULTI_COND = 1


def fmt_dur(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}min"
    if seconds < 86400:
        return f"{seconds/3600:.1f}h"
    days = seconds / 86400
    return f"{days:.2f}d ({days/7:.2f}w)"


def linfit(xs, ys):
    """Least-squares y = a + b x. Returns (a, b)."""
    n = len(xs)
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return sy / n, 0.0
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    return a, b


def main():
    tbc = json.loads((OUT_DIR / "throughput_by_context.json").read_text())
    by_ctx = tbc["by_context"]

    # Each context point gives steady_state_ms_per_row measured at a buffer that
    # holds the full target trajectory. The representative context length (in
    # rows, the K-length SDPA attends over) is the buffer size for that target,
    # i.e. the full trajectory length. We fit ms/row vs that K-length.
    pts = []
    for k in sorted(by_ctx.keys(), key=int):
        r = by_ctx[k]
        ctx_rows = r["buffer_rows"]            # K-length the kernel attends over
        ms_per_row = r["steady_state_ms_per_row"]
        prod_frac = r["productive_per_row_ch12"]
        pts.append((int(k), ctx_rows, ms_per_row, r["derived_multi_tok_s"], prod_frac))

    xs = [p[1] for p in pts]
    ys = [p[2] / 1000.0 for p in pts]  # seconds/row
    t0, tau = linfit(xs, ys)  # t(n) = t0 + tau * n   (n = K-length in rows)

    # Trajectory: 8000 productive Output tokens. productive-per-row (ch1 only ~1)
    # → rows ≈ tokens (Output non-silent ~ once/row after warmup). Use the
    # measured ch1+ch2 prod/row halved as a per-channel proxy, floored at ~1.
    largest = pts[-1]
    # rows per Output token ≈ 1 / (Output prod fraction). Output is ch1; the
    # measured prod_frac counts ch1+ch2, so Output alone ≈ prod_frac/2 but never
    # below the empirical ~1.05 rows/token from Phase A/B.
    rows_per_tok = 1.06
    n_rows = TRAJ_TOKENS * rows_per_tok

    # Integrate t(r) over r=0..n_rows : T = t0*N + tau*N^2/2
    def traj_time(N):
        return t0 * N + tau * N * N / 2.0

    T_traj = traj_time(n_rows)

    rollouts_per_cond = N_TASKS * N_SEEDS
    # All 5 conditions decode a full trajectory of ~n_rows rows (the multi
    # condition produces extra channels but the decoder cost is per-row, shared).
    eval_total = rollouts_per_cond * T_traj * (N_SINGLE_COND + N_MULTI_COND)

    # Also a constant-rate optimistic bound using the 8k multi tok/s.
    multi_tps_8k = by_ctx[max(by_ctx.keys(), key=int)]["derived_multi_tok_s"]
    per_traj_opt = TRAJ_TOKENS / multi_tps_8k if multi_tps_8k else float("inf")
    eval_opt = rollouts_per_cond * per_traj_opt * (N_SINGLE_COND + N_MULTI_COND)

    out = {
        "plan": {"n_tasks": N_TASKS, "n_seeds": N_SEEDS, "traj_tokens": TRAJ_TOKENS,
                 "n_single_conditions": N_SINGLE_COND, "n_multi_conditions": N_MULTI_COND,
                 "rollouts_per_condition": rollouts_per_cond},
        "context_curve_points": [
            {"target_tokens": p[0], "k_length_rows": p[1], "ms_per_row": p[2],
             "derived_multi_tps": p[3], "productive_per_row_ch12": p[4]}
            for p in pts
        ],
        "fit": {"t0_seconds_per_row": t0, "tau_seconds_per_row_per_ctx_row": tau,
                "model": "t(n) = t0 + tau * n  (n = context length in rows)"},
        "trajectory": {"rows_per_output_token": rows_per_tok, "n_rows_8k_traj": n_rows,
                       "per_traj_seconds_kv_aware": T_traj,
                       "per_traj_seconds_constant_rate": per_traj_opt},
        "L4": {
            "eval_seconds_kv_aware": eval_total,
            "eval_weeks_kv_aware": eval_total / (7 * 86400),
            "eval_seconds_constant_rate": eval_opt,
            "eval_weeks_constant_rate": eval_opt / (7 * 86400),
        },
        "verdict_vs_budget_weeks": 3.0,
    }
    (OUT_DIR / "extrapolation.json").write_text(json.dumps(out, indent=2))
    print(f"fit: t(n) = {t0*1000:.1f}ms + {tau*1e6:.2f}us/ctx-row * n")
    print(f"8k traj rows={n_rows:.0f}  per-traj kv-aware={fmt_dur(T_traj)}  "
          f"const-rate={fmt_dur(per_traj_opt)}")
    print(f"L4 kv-aware: {fmt_dur(eval_total)} ({out['L4']['eval_weeks_kv_aware']:.2f} w)")
    print(f"L4 const-rate: {fmt_dur(eval_opt)} ({out['L4']['eval_weeks_constant_rate']:.2f} w)")
    print(f"[done] wrote {OUT_DIR}/extrapolation.json")


if __name__ == "__main__":
    main()
