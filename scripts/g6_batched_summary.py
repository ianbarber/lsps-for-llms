#!/usr/bin/env python3
"""g6 batched — assemble the scaling table + summary.md from the sweep JSONs
and the extrapolation. Pure reporting; no GPU."""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path("/home/ianbarber/Projects/Streams/runs/g6_batched")


def load(p):
    p = OUT / p
    return json.loads(p.read_text()) if p.exists() else None


def table(sweep):
    if not sweep:
        return "(no data)\n"
    L = ["| B | gpu_step ms | wall_step ms | agg all tok/s | agg prod tok/s | rows/s | peak GB | achieved GB/s | % roofline |\n",
         "|---|---|---|---|---|---|---|---|---|\n"]
    for k in sorted(sweep["by_batch"], key=int):
        e = sweep["by_batch"][k]
        if e.get("oom"):
            L.append(f"| {k} | OOM ({e.get('stage','')}) | | | | | {e.get('peak_mem_gb','')} | | |\n")
            continue
        rps = e["B"] / (e["wall_step_ms_net"] / 1000.0)
        L.append(f"| {e['B']} | {e['gpu_step_ms']:.1f} | {e['wall_step_ms_net']:.1f} | "
                 f"{e['agg_all_tok_s']:.1f} | {e['agg_productive_tok_s']:.1f} | {rps:.2f} | "
                 f"{e['peak_mem_gb']:.1f} | {e['achieved_bw_gb_s']:.1f} | "
                 f"{e['bw_efficiency']*100:.0f}% |\n")
    return "".join(L)


def main():
    s512 = load("sweep_ctx512.json")
    s4096 = load("sweep_ctx4096.json")
    extra = load("extrapolation.json")

    L = ["# g6 batched decode sweep — does batching solve the L4 throughput problem?\n\n",
         "Substrate: stock Qwen3-8B BF16 SDPA, manual batched `model.forward` decode loop. ",
         "Each step advances B independent task-streams, each [B,C] (C=10 channels) tokens, ",
         "batched DynamicCache + batched cross-stream mask. cuda.Event per-step GPU time; ",
         "wall-clock for aggregate throughput. Productive = Output+Analytical = 2 of 10 channels.\n\n",
         "Weight bytes/step (matmul read) = 13.89 GB. Roofline = 273 GB/s. ",
         "Single-seq baseline (microbench) = 39.8 GB/s = 15%.\n\n",
         "## ctx 512\n\n", table(s512), "\n",
         "## ctx 4096\n\n", table(s4096), "\n"]

    if extra:
        L.append("## L4 projection (rows/s aggregate; 8000 rows/run; 3-week=21d budget)\n\n")
        L.append("| ctx | best B | rows/s | peak GB | GB/s (% roofline) | scope | wall days | vs budget | fits |\n")
        L.append("|---|---|---|---|---|---|---|---|---|\n")
        for cx, pr in extra["projections"].items():
            if not pr:
                continue
            for scope, sc in pr["scopes"].items():
                L.append(f"| {cx} | {pr['best_B']} | {pr['best_rows_per_s']:.2f} | "
                         f"{pr['best_peak_mem_gb']:.1f} | {pr['best_achieved_bw_gb_s']:.1f} "
                         f"({pr['best_bw_efficiency']*100:.0f}%) | {scope} | {sc['wall_days']:.2f} | "
                         f"{sc['vs_budget_days']:.2f}x | {'YES' if sc['fits_budget'] else 'no'} |\n")
    (OUT / "summary.md").write_text("".join(L))
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    main()
