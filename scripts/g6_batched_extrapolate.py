#!/usr/bin/env python3
"""g6 batched — L4 wall-clock projection from the batched-decode sweep.

Reads sweep JSONs, picks the best feasible batch (highest aggregate productive
tok/s that fits in memory at the realistic ctx), and projects L4 wall-clock for:
  - full:     200 tasks x 9 seeds x 5 conditions x 8000 tokens
  - descoped:  50 tasks x 6 seeds x 5 conditions x 8000 tokens
against the 3-week (504h, 21d) budget.

L4 is a throughput workload: total productive tokens / aggregate productive
tok/s. "tokens" here = generated rows; a trajectory of N rows produces
PRODUCTIVE productive tokens per row. We express runtime in terms of the
aggregate PRODUCTIVE tok/s the sweep measured, so the trajectory length is the
productive-token count.

Convention: 8000 tokens per run = 8000 *rows* per trajectory (the realistic L4
decode length). Productive tokens per trajectory = 8000 * PRODUCTIVE... but the
sweep's agg_productive_tok_s already counts B*PRODUCTIVE per step, so the
natural unit is "rows/s aggregate". We therefore project on ROWS: a step
advances every sequence one row, so aggregate rows/s = B / wall_step_s. Total
rows = n_traj * 8000. This is the cleanest, batch-fair accounting and avoids
double-counting channels.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

OUT = Path("/home/ianbarber/Projects/Streams/runs/g6_batched")
BUDGET_DAYS = 21.0
BUDGET_HOURS = BUDGET_DAYS * 24
TOKENS_PER_RUN = 8000  # rows per trajectory (realistic L4 decode length)

FULL = dict(tasks=200, seeds=9, conditions=5)
DESCOPED = dict(tasks=50, seeds=6, conditions=5)


def load(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def feasible_entries(sweep):
    """Return list of (B, entry) that did not OOM, sorted by B."""
    out = []
    for k, e in sweep["by_batch"].items():
        if e.get("oom"):
            continue
        out.append((int(k), e))
    return sorted(out, key=lambda x: x[0])


def rows_per_s(entry):
    """Aggregate rows advanced per wall-second = B / wall_step_s."""
    B = entry["B"]
    wall_step_s = entry["wall_step_ms_net"] / 1000.0
    return B / wall_step_s


def project(sweep, label):
    feas = feasible_entries(sweep)
    if not feas:
        return None
    # The throughput curve plateaus once compute-bound, so the largest B often
    # gives marginally higher rows/s at 2x the latency+memory. Pick the SMALLEST
    # B whose aggregate rows/s is within 3% of the observed max -- that is the
    # knee (best throughput per unit memory/latency), the right operating point.
    max_rps = max(rows_per_s(e) for _, e in feas)
    best_B, best_e = min((b_e for b_e in feas
                          if rows_per_s(b_e[1]) >= 0.97 * max_rps),
                         key=lambda x: x[0])
    rps = rows_per_s(best_e)

    proj = {"ctx_len": sweep["ctx_len"], "best_B": best_B,
            "best_rows_per_s": rps,
            "best_agg_productive_tok_s": best_e["agg_productive_tok_s"],
            "best_peak_mem_gb": best_e["peak_mem_gb"],
            "best_achieved_bw_gb_s": best_e["achieved_bw_gb_s"],
            "best_bw_efficiency": best_e["bw_efficiency"],
            "scopes": {}}
    for name, sc in (("full", FULL), ("descoped", DESCOPED)):
        n_traj = sc["tasks"] * sc["seeds"] * sc["conditions"]
        total_rows = n_traj * TOKENS_PER_RUN
        wall_s = total_rows / rps
        wall_h = wall_s / 3600.0
        wall_d = wall_h / 24.0
        proj["scopes"][name] = {
            "n_trajectories": n_traj,
            "tokens_per_run": TOKENS_PER_RUN,
            "total_rows": total_rows,
            "wall_hours": wall_h,
            "wall_days": wall_d,
            "vs_budget_days": wall_d / BUDGET_DAYS,
            "fits_budget": wall_d <= BUDGET_DAYS,
        }
    return proj


def main():
    s512 = load(OUT / "sweep_ctx512.json")
    s4096 = load(OUT / "sweep_ctx4096.json")
    res = {"budget_days": BUDGET_DAYS, "budget_hours": BUDGET_HOURS,
           "tokens_per_run": TOKENS_PER_RUN,
           "full_scope": FULL, "descoped_scope": DESCOPED,
           "projections": {}}
    if s512:
        res["projections"]["ctx512"] = project(s512, "ctx512")
    if s4096:
        res["projections"]["ctx4096"] = project(s4096, "ctx4096")
    (OUT / "extrapolation.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
