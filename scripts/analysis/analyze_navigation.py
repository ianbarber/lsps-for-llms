#!/usr/bin/env python3
"""Task-level summaries and paired bootstrap contrasts for navigation-v1."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import random
import statistics
from pathlib import Path


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def task_values(rows: list[dict], variant: str, arm: str, metric: str) -> dict[str, float]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row["variant"] == variant and row["arm"] == arm:
            grouped.setdefault(row["task"], []).append(row)

    def value(row: dict) -> float:
        if metric == "pass":
            return float(bool(row["held_out_pass"]))
        if metric == "total_tokens":
            return float(row["in_tokens"] + row["out_tokens"])
        return float(row[metric])

    return {task: statistics.mean(value(row) for row in group) for task, group in grouped.items()}


def repetition_grid(rows: list[dict], variant: str, arm: str) -> Counter:
    return Counter(
        (row["task"], row.get("seed"))
        for row in rows if row["variant"] == variant and row["arm"] == arm
    )


def paired_effect(
    rows: list[dict], base: tuple[str, str], treatment: tuple[str, str], metric: str,
    bootstrap: int, seed: int,
) -> dict:
    before = task_values(rows, *base, metric)
    after = task_values(rows, *treatment, metric)
    if repetition_grid(rows, *base) != repetition_grid(rows, *treatment):
        return {
            "n_tasks": len(set(before) & set(after)), "estimable": False,
            "reason": "incomplete paired task/seed grid",
            "baseline_tasks": sorted(before), "treatment_tasks": sorted(after),
        }
    tasks = sorted(set(before) & set(after))
    if not tasks:
        return {"n_tasks": 0}

    def effect(sample: list[str]) -> float:
        return statistics.mean(after[task] - before[task] for task in sample)

    observed = effect(tasks)
    rng = random.Random(seed)
    samples = [effect([rng.choice(tasks) for _ in tasks]) for _ in range(bootstrap)]
    return {
        "n_tasks": len(tasks), "mean_delta": observed,
        "ci95": [quantile(samples, 0.025), quantile(samples, 0.975)],
    }


def paired_ratio(
    rows: list[dict], base: tuple[str, str], treatment: tuple[str, str], metric: str,
    bootstrap: int, seed: int, margin: tuple[float, float] = (0.90, 1.10),
) -> dict:
    before = task_values(rows, *base, metric)
    after = task_values(rows, *treatment, metric)
    if repetition_grid(rows, *base) != repetition_grid(rows, *treatment):
        return {
            "n_tasks": len(set(before) & set(after)), "estimable": False,
            "reason": "incomplete paired task/seed grid",
            "baseline_tasks": sorted(before), "treatment_tasks": sorted(after),
        }
    tasks = sorted(set(before) & set(after))
    if not tasks:
        return {"n_tasks": 0}
    if any(before[task] <= 0 for task in tasks):
        return {"n_tasks": len(tasks), "undefined": "baseline metric is non-positive"}

    def effect(sample: list[str]) -> float:
        control_mean = statistics.mean(before[task] for task in sample)
        if control_mean <= 0:
            return math.inf
        return statistics.mean(after[task] for task in sample) / control_mean

    ratios = [after[task] / before[task] for task in tasks]
    rng = random.Random(seed)
    samples = [effect([rng.choice(tasks) for _ in tasks]) for _ in range(bootstrap)]
    interval = [quantile(samples, 0.025), quantile(samples, 0.975)]
    within_margin = interval[0] >= margin[0] and interval[1] <= margin[1]
    return {
        "n_tasks": len(tasks),
        "ratio_of_task_weighted_means": effect(tasks),
        "mean_task_ratio_descriptive": statistics.mean(ratios),
        "median_task_ratio": statistics.median(ratios),
        "ci95": interval,
        "equivalence_margin": list(margin),
        "exploratory_margin_compatible": within_margin,
        "interpretation": (
            "exploratory_within_margin_not_population_equivalence"
            if within_margin else "inconclusive_or_different"
        ),
    }


def interaction(rows: list[dict], metric: str, bootstrap: int, seed: int) -> dict:
    keys = (
        ("typed", "baseline"), ("typed", "semantic_auto"),
        ("erased", "baseline"), ("erased", "semantic_auto"),
    )
    grids = [repetition_grid(rows, *key) for key in keys]
    if not grids[0] or any(grid != grids[0] for grid in grids[1:]):
        return {
            "n_tasks": 0, "estimable": False,
            "reason": "incomplete factorial task/seed grid",
        }
    cells = {
        key: task_values(rows, *key, metric)
        for key in keys
    }
    tasks = sorted(set.intersection(*(set(values) for values in cells.values())))
    if not tasks:
        return {"n_tasks": 0}

    def effect(sample: list[str]) -> float:
        deltas = []
        for task in sample:
            typed = cells[("typed", "semantic_auto")][task] - cells[("typed", "baseline")][task]
            erased = cells[("erased", "semantic_auto")][task] - cells[("erased", "baseline")][task]
            deltas.append(typed - erased)
        return statistics.mean(deltas)

    rng = random.Random(seed)
    samples = [effect([rng.choice(tasks) for _ in tasks]) for _ in range(bootstrap)]
    return {
        "n_tasks": len(tasks), "difference_in_differences": effect(tasks),
        "ci95": [quantile(samples, 0.025), quantile(samples, 0.975)],
    }


def summarize_cell(rows: list[dict], variant: str, arm: str) -> dict:
    cell = [row for row in rows if row["variant"] == variant and row["arm"] == arm]
    grouped: dict[str, list[dict]] = {}
    for row in cell:
        grouped.setdefault(row["task"], []).append(row)
    if not grouped:
        return {}

    def means(field: str, transform=float) -> list[float]:
        return [statistics.mean(transform(row[field]) for row in group) for group in grouped.values()]

    passes = means("held_out_pass", bool)
    total_tokens = [statistics.mean(row["in_tokens"] + row["out_tokens"] for row in group)
                    for group in grouped.values()]
    adjusted_wall = [
        statistics.mean(
            row["wall_sec"] + (
                row["server_latency_ms"] / 1000 if row["arm"] == "semantic_auto" else 0
            )
            for row in group
        )
        for group in grouped.values()
    ]
    success_mass = sum(passes)
    return {
        "tasks": len(grouped),
        "pass": statistics.mean(passes),
        "input_tokens_mean": statistics.mean(means("in_tokens")),
        "input_tokens_median": statistics.median(means("in_tokens")),
        "output_tokens_mean": statistics.mean(means("out_tokens")),
        "expected_total_tokens_per_success": (
            sum(total_tokens) / success_mass if success_mass else math.inf
        ),
        "localized": statistics.mean(means("correct_file_localized_before_first_edit", bool)),
        "wrong_file_edits": statistics.mean(means("wrong_file_edits")),
        "reads": statistics.mean(means("n_reads")),
        "turns": statistics.mean(means("turns")),
        "wall_sec_mean": statistics.mean(means("wall_sec")),
        "wall_sec_median": statistics.median(means("wall_sec")),
        "adjusted_end_to_end_sec_mean": statistics.mean(adjusted_wall),
        "adjusted_end_to_end_sec_median": statistics.median(adjusted_wall),
        "server_latency_ms_mean": statistics.mean(means("server_latency_ms")),
        "server_latency_ms_median": statistics.median(means("server_latency_ms")),
        "semantic_election": statistics.mean(
            [statistics.mean(bool(row["n_lsp"]) for row in group) for group in grouped.values()]
        ),
        "semantic_then_target_read": statistics.mean(
            means("semantic_then_target_read", bool)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()
    rows = []
    for path in args.runs:
        rows.extend(json.loads(Path(path).read_text())["rows"])
    if not rows:
        raise ValueError("no navigation rows")

    print("NAVIGATION OUTCOMES (task is the generalization unit)")
    for variant, arm in sorted({(row["variant"], row["arm"]) for row in rows}):
        print(f"{variant}/{arm}")
        print(json.dumps(summarize_cell(rows, variant, arm), indent=2, allow_nan=True))

    contrasts = {}
    for variant in ("typed", "erased"):
        for metric in ("pass", "total_tokens"):
            contrasts[f"{variant}_semantic_auto_minus_baseline_{metric}"] = paired_effect(
                rows, (variant, "baseline"), (variant, "semantic_auto"), metric,
                args.bootstrap, args.seed,
            )
        contrasts[f"{variant}_semantic_auto_over_baseline_total_tokens"] = paired_ratio(
            rows, (variant, "baseline"), (variant, "semantic_auto"), "total_tokens",
            args.bootstrap, args.seed,
        )
    for metric in ("pass", "total_tokens"):
        contrasts[f"typed_x_semantic_interaction_{metric}"] = interaction(
            rows, metric, args.bootstrap, args.seed
        )
    print("PAIRED TASK-LEVEL CONTRASTS")
    print(json.dumps(contrasts, indent=2, allow_nan=True))
    core = [row for row in rows if row["arm"] in ("baseline", "semantic_auto")]
    if not core:
        print("Causal matrix absent; controls alone estimate competence/actionability, not a treatment effect.")
    elif not any(row["held_out_pass"] for row in core):
        print("All causal cells are at floor; pass contrasts are non-identifying, not equivalence evidence.")
        print("Token contrasts in failed cells measure failure cost, not useful-work efficiency.")
    elif core and all(row["held_out_pass"] for row in core):
        print("All causal cells are at ceiling; pass contrasts are non-identifying, not equivalence evidence.")
    else:
        print("Nulls are not equivalence unless the pass CI is inside +/-0.10 and the preregistered token-ratio margin is met.")


if __name__ == "__main__":
    main()
