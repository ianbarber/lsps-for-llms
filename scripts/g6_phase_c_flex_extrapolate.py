#!/usr/bin/env python3
"""Phase C Route 2 — F6 L4 wall-clock extrapolation from the FlexAttention
context-length curve.

Unlike Phase B (which borrowed Phase A's KV slope), here we FIT the per-row time
as an affine function of cache length directly from the measured 256/1024/4096/
8192 curve:  t_row(n_cache) = tau_0 + tau_a * n_cache.

L4 plan: 200 tasks x 9 seeds x 5 conditions (4 single + 1 multi) x 8000 productive
tokens. Integrate the affine per-row time over the trajectory.

Writes runs/g6_phase_c_flex/extrapolation.json.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT_DIR = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_c_flex")


def fmt_dur(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}min"
    if seconds < 86400:
        return f"{seconds/3600:.1f}h"
    days = seconds / 86400
    return f"{days:.2f}d ({days/7:.2f}w)"


def fit_affine(xs, ys):
    """Least-squares t = a + b*x. Returns (a, b)."""
    n = len(xs)
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return sy / n, 0.0
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    return a, b


def main():
    tp = json.loads((OUT_DIR / "throughput_by_context.json").read_text())
    by_ctx = tp["by_context"]

    # For each context target, derive (mean_cache_len_during_traj, ms_per_row).
    # We approximate the mean cache length over a trajectory that reaches
    # `rows` rows as rows*C/2 (linear ramp from 0). ms_per_row is the measured
    # mean. Build (x=mean_cache, y=s_per_row) points and fit affine.
    pts = []
    rows_at = {}
    for n_str, summ in by_ctx.items():
        n = int(n_str)
        # rows reached ~ from per_prompt; use mean rows
        rows = sum(p["rows"] for p in summ["per_prompt"]) / len(summ["per_prompt"])
        ms = summ["mean_ms_per_row"]
        C = 10
        mean_cache = rows * C / 2.0  # average flat cache length over the ramp
        pts.append((mean_cache, ms / 1000.0))
        rows_at[n] = {"rows": rows, "ms_per_row": ms,
                      "tok_s": summ["mean_productive_tokens_per_sec"],
                      "peak_gb": summ["peak_memory_gb"]}

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    tau_0, tau_a = fit_affine(xs, ys)  # s/row, s/row per cache-token

    # L4 trajectory: 8000 productive Output tokens. rows_per_output ~ measured.
    # Use the 256-context point's rows/productive ratio as rows-per-token.
    C = 10
    ref = by_ctx["256"]
    ref_rows = sum(p["rows"] for p in ref["per_prompt"]) / len(ref["per_prompt"])
    ref_prod = sum(p["productive_tokens"] for p in ref["per_prompt"]) / len(ref["per_prompt"])
    rows_per_output = ref_rows / max(ref_prod, 1)

    traj_tokens = 8000
    n_rows = int(traj_tokens * rows_per_output)
    # integrate t(row) = tau_0 + tau_a * (row*C)   [cache grows by C each row]
    #   sum_{r=0}^{N-1} (tau_0 + tau_a*C*r) = tau_0*N + tau_a*C*N*(N-1)/2
    traj_s = tau_0 * n_rows + tau_a * C * n_rows * (n_rows - 1) / 2.0

    n_tasks, n_seeds = 200, 9
    n_conditions = 5  # 4 single + 1 multi; flex multi is the same decoder
    rollouts = n_tasks * n_seeds
    eval_s = rollouts * traj_s * n_conditions

    out = {
        "method": "Affine per-row fit t_row = tau_0 + tau_a*cache_len from the "
                  "flex context-length curve (256/1024/4096/8192).",
        "fit": {"tau_0_s_per_row": tau_0, "tau_a_s_per_row_per_cache_token": tau_a},
        "context_points": rows_at,
        "rows_per_output_token": rows_per_output,
        "traj_tokens": traj_tokens, "n_rows_per_traj": n_rows,
        "per_traj_seconds": traj_s, "per_traj_human": fmt_dur(traj_s),
        "L4_plan": {"n_tasks": n_tasks, "n_seeds": n_seeds,
                    "n_conditions": n_conditions, "rollouts_per_condition": rollouts},
        "eval_seconds": eval_s, "eval_weeks": eval_s / (7 * 86400),
        "eval_human": fmt_dur(eval_s),
    }
    (OUT_DIR / "extrapolation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
