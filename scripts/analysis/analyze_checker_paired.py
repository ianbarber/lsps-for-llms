#!/usr/bin/env python3
"""Opportunity summaries and task-bootstrap contrasts for checker-paired-v1."""

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


def revision_eligible(draft: dict) -> bool:
    return bool(
        draft["coherent"]
        and (draft.get("draft_submitted") or draft.get("revision_case_series_eligible"))
    )


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["task"], []).append(row)

    def task_mean(value) -> float:
        return statistics.mean(
            statistics.mean(float(value(row)) for row in task_rows)
            for task_rows in grouped.values()
        )

    def task_median(value) -> float:
        return statistics.median(
            statistics.mean(float(value(row)) for row in task_rows)
            for task_rows in grouped.values()
        )

    draft_cost_available = all(
        row.get("draft_in_tokens") is not None and row.get("draft_out_tokens") is not None
        for row in rows
    )
    result = {
        "n_tasks": len(grouped), "n_revision_trajectories": len(rows),
        "final_held_pass": task_mean(lambda row: row["held_pass"]),
        "accepted_rate": task_mean(lambda row: row["accepted"]),
        "accepted_correct_rate": task_mean(lambda row: row["accepted"] and row["held_pass"]),
        "type_clean_rate": task_mean(lambda row: row["type_clean"]),
        "semantic_clean_rate": task_mean(lambda row: row["semantic_clean"]),
        "accepted_type_clean_correct_rate": task_mean(
            lambda row: row["accepted_type_clean_correct"]
        ),
        "accepted_behavioral_defect_rate": task_mean(
            lambda row: row["accepted"] and not row["held_pass"]
        ),
        "accepted_semantic_defect_rate": task_mean(
            lambda row: row["accepted"] and not row["semantic_clean"]
        ),
        "accepted_any_checker_defect_rate": task_mean(
            lambda row: row["accepted"] and not row["type_clean"]
        ),
        "unsubmitted_rate": task_mean(lambda row: row["unsubmitted"]),
        "gate_checked_rate": task_mean(lambda row: bool(row["gate_invocations"])),
        "gate_rejected_rate": task_mean(lambda row: bool(row["gate_rejections"])),
        "gate_accepted_rate": task_mean(lambda row: bool(row["gate_acceptances"])),
        "edited_diagnosed_location_rate": task_mean(lambda row: row["edited_diagnosed_location"]),
        "diagnostics_eliminated_mean": task_mean(lambda row: row["diagnostics_eliminated"]),
        "diagnostics_retained_mean": task_mean(lambda row: row["diagnostics_retained"]),
        "diagnostics_introduced_mean": task_mean(lambda row: row["diagnostics_introduced"]),
        "revision_tokens_mean": task_mean(lambda row: row["in_tokens"] + row["out_tokens"]),
        "revision_tokens_median": task_median(lambda row: row["in_tokens"] + row["out_tokens"]),
        "turns_mean": task_mean(lambda row: row["turns"]),
        "checker_latency_ms_mean": task_mean(lambda row: row["checker_latency_ms"]),
        "wall_sec_mean": task_mean(lambda row: row["wall_sec"]),
    }
    result["draft_plus_revision_tokens_estimable"] = draft_cost_available
    result["draft_plus_revision_tokens_mean"] = (
        task_mean(lambda row: row["draft_in_tokens"] + row["draft_out_tokens"]
                  + row["in_tokens"] + row["out_tokens"])
        if draft_cost_available else None
    )
    result["draft_plus_revision_tokens_median"] = (
        task_median(lambda row: row["draft_in_tokens"] + row["draft_out_tokens"]
                    + row["in_tokens"] + row["out_tokens"])
        if draft_cost_available else None
    )
    return result


def end_to_end_summary(drafts: list[dict], rows: list[dict], arm: str) -> dict:
    by_draft: dict[str, list[dict]] = {}
    for row in rows:
        if row["arm"] == arm:
            by_draft.setdefault(row["draft_id"], []).append(row)
    eligible = [draft for draft in drafts if revision_eligible(draft)]
    missing = [draft["draft_id"] for draft in eligible if not by_draft.get(draft["draft_id"])]
    if missing:
        return {
            "estimable": False, "reason": "missing revision trajectories for eligible drafts",
            "missing_draft_ids": missing,
        }
    if any(draft.get("in_tokens") is None or draft.get("out_tokens") is None for draft in drafts):
        return {"estimable": False, "reason": "natural draft token cost is missing"}

    attempts: dict[str, list[dict]] = {}
    for draft in drafts:
        draft_tokens = draft["in_tokens"] + draft["out_tokens"]
        revisions = by_draft.get(draft["draft_id"], [])
        if not revision_eligible(draft):
            attempts.setdefault(draft["task"], []).append({
                "held": 0.0, "accepted": 0.0, "accepted_correct": 0.0,
                "accepted_clean_correct": 0.0, "accepted_behavioral_defect": 0.0,
                "accepted_semantic_defect": 0.0, "pre_revision_failure": 1.0,
                "gate_checked": 0.0, "gate_rejection": 0.0, "gate_acceptance": 0.0,
                "unsubmitted": 1.0, "draft_tokens": draft_tokens,
                "revision_tokens": 0.0, "total_tokens": draft_tokens,
            })
            continue
        for row in revisions:
            revision_tokens = row["in_tokens"] + row["out_tokens"]
            attempts.setdefault(draft["task"], []).append({
                "held": float(row["held_pass"]), "accepted": float(row["accepted"]),
                "accepted_correct": float(row["accepted"] and row["held_pass"]),
                "accepted_clean_correct": float(
                    row["accepted_type_clean_correct"]
                ),
                "accepted_behavioral_defect": float(row["accepted"] and not row["held_pass"]),
                "accepted_semantic_defect": float(row["accepted"] and not row["semantic_clean"]),
                "pre_revision_failure": 0.0,
                "gate_checked": float(bool(row["gate_invocations"])),
                "gate_rejection": float(bool(row["gate_rejections"])),
                "gate_acceptance": float(bool(row["gate_acceptances"])),
                "unsubmitted": float(row["unsubmitted"]),
                "draft_tokens": draft_tokens, "revision_tokens": revision_tokens,
                "total_tokens": draft_tokens + revision_tokens,
            })

    def task_mean(field: str) -> float:
        return statistics.mean(
            statistics.mean(attempt[field] for attempt in task_attempts)
            for task_attempts in attempts.values()
        )

    total_cost = task_mean("total_tokens")
    accepted_correct = task_mean("accepted_correct")
    accepted_clean_correct = task_mean("accepted_clean_correct")
    return {
        "estimable": True, "n_tasks": len(attempts), "n_generated_drafts": len(drafts),
        "n_revision_eligible_drafts": len(eligible),
        "n_pre_revision_failures": len(drafts) - len(eligible),
        "final_held_pass_yield": task_mean("held"),
        "accepted_yield": task_mean("accepted"),
        "accepted_correct_yield": accepted_correct,
        "accepted_type_clean_correct_yield": accepted_clean_correct,
        "accepted_behavioral_defect_rate": task_mean("accepted_behavioral_defect"),
        "accepted_semantic_defect_rate": task_mean("accepted_semantic_defect"),
        "pre_revision_failure_rate": task_mean("pre_revision_failure"),
        "unsubmitted_rate": task_mean("unsubmitted"),
        "gate_checked_rate": task_mean("gate_checked"),
        "gate_rejection_rate": task_mean("gate_rejection"),
        "gate_acceptance_rate": task_mean("gate_acceptance"),
        "draft_tokens_mean": task_mean("draft_tokens"),
        "revision_tokens_mean_including_pre_revision_zeros": task_mean("revision_tokens"),
        "total_tokens_mean": total_cost,
        "expected_tokens_per_accepted_correct_patch": (
            total_cost / accepted_correct if accepted_correct else math.inf
        ),
        "expected_tokens_per_accepted_type_clean_correct_patch": (
            total_cost / accepted_clean_correct if accepted_clean_correct else math.inf
        ),
    }


def expected_cost_per_accepted_correct(
    drafts: list[dict], rows: list[dict], arm: str, opportunity_only: bool,
    bootstrap: int, seed: int, require_type_clean: bool = False,
) -> dict:
    selected = [draft for draft in drafts if not opportunity_only or (
        draft["coherent"] and any(
            item["classification"] == "semantic" for item in draft["draft_diagnostics"]
        )
    )]
    if not selected:
        return {"n_tasks": 0, "estimable": False, "reason": "no selected natural drafts"}
    if any(draft.get("in_tokens") is None or draft.get("out_tokens") is None for draft in selected):
        return {"n_tasks": 0, "estimable": False, "reason": "natural draft token cost is missing"}
    by_draft: dict[str, list[dict]] = {}
    for row in rows:
        if row["arm"] == arm:
            by_draft.setdefault(row["draft_id"], []).append(row)
    missing = [
        draft["draft_id"] for draft in selected
        if revision_eligible(draft)
        and not by_draft.get(draft["draft_id"])
    ]
    if missing:
        return {
            "n_tasks": len({draft["task"] for draft in selected}),
            "estimable": False,
            "reason": "missing revision trajectories for coherent drafts",
            "missing_draft_ids": missing,
        }
    unexpected = [
        draft["draft_id"] for draft in selected
        if not revision_eligible(draft)
        and by_draft.get(draft["draft_id"])
    ]
    if unexpected:
        return {
            "n_tasks": 0, "estimable": False,
            "reason": "revision trajectories exist for ineligible drafts",
            "unexpected_draft_ids": unexpected,
        }

    attempts: dict[str, list[tuple[float, float]]] = {}
    n_revisions = 0
    for draft in selected:
        draft_cost = float(draft["in_tokens"] + draft["out_tokens"])
        revisions = by_draft.get(draft["draft_id"], [])
        if not revision_eligible(draft):
            attempts.setdefault(draft["task"], []).append((draft_cost, 0.0))
            continue
        for row in revisions:
            n_revisions += 1
            success = bool(row["accepted"] and row["held_pass"])
            if require_type_clean:
                success = success and bool(row["type_clean"])
            cost = draft_cost + float(row["in_tokens"] + row["out_tokens"])
            attempts.setdefault(draft["task"], []).append((cost, float(success)))

    tasks = sorted(attempts)

    def estimate(sample: list[str]) -> tuple[float, float, float]:
        task_costs = [statistics.mean(cost for cost, _ in attempts[task]) for task in sample]
        task_success = [statistics.mean(success for _, success in attempts[task]) for task in sample]
        mean_cost = statistics.mean(task_costs)
        success_rate = statistics.mean(task_success)
        return mean_cost, success_rate, mean_cost / success_rate if success_rate else math.inf

    mean_cost, success_rate, expected = estimate(tasks)
    rng = random.Random(seed)
    samples = [estimate([rng.choice(tasks) for _ in tasks])[2] for _ in range(bootstrap)]
    return {
        "n_tasks": len(tasks), "n_natural_drafts": len(selected),
        "n_revision_trajectories": n_revisions, "estimable": True,
        "mean_draft_plus_revision_tokens": mean_cost,
        "accepted_correct_rate": success_rate,
        "expected_tokens_per_accepted_correct_patch": expected,
        "ci95": [quantile(samples, 0.025), quantile(samples, 0.975)],
        "success_requires_type_clean": require_type_clean,
    }


def paired_contrast(
    rows: list[dict], arm: str, field: str, opportunity_only: bool,
    bootstrap: int, seed: int,
) -> dict:
    selected = [row for row in rows if not opportunity_only or row["opportunity"]]
    by_arm: dict[str, dict[str, list[dict]]] = {}
    for row in selected:
        by_arm.setdefault(row["arm"], {}).setdefault(row["task"], []).append(row)
    if "control" not in by_arm or arm not in by_arm:
        return {"n_tasks": 0}
    control_grid = Counter(
        (row["draft_id"], row.get("seed")) for row in selected if row["arm"] == "control"
    )
    treatment_grid = Counter(
        (row["draft_id"], row.get("seed")) for row in selected if row["arm"] == arm
    )
    if control_grid != treatment_grid:
        return {
            "n_tasks": 0, "estimable": False,
            "reason": "incomplete paired draft/seed grid",
        }
    tasks = sorted(set(by_arm["control"]) & set(by_arm[arm]))
    if not tasks:
        return {"n_tasks": 0}

    def task_mean(which: str, task: str) -> float:
        return statistics.mean(float(row[field]) for row in by_arm[which][task])

    def effect(sample: list[str]) -> float:
        return statistics.mean(task_mean(arm, task) - task_mean("control", task) for task in sample)

    rng = random.Random(seed)
    samples = [effect([rng.choice(tasks) for _ in tasks]) for _ in range(bootstrap)]
    return {
        "n_tasks": len(tasks), "mean_delta": effect(tasks),
        "ci95": [quantile(samples, 0.025), quantile(samples, 0.975)],
    }


def end_to_end_contrast(
    drafts: list[dict], rows: list[dict], arm: str, field: str,
    bootstrap: int, seed: int,
) -> dict:
    """Contrast arms by task while retaining failed natural drafts as zero outcomes."""
    if arm == "control":
        raise ValueError("treatment arm must differ from control")
    eligible = {
        draft["draft_id"] for draft in drafts
        if revision_eligible(draft)
    }
    selected = [row for row in rows if row["arm"] in {"control", arm}]
    grids = {
        which: Counter(
            (row["draft_id"], row.get("seed"))
            for row in selected if row["arm"] == which
        )
        for which in ("control", arm)
    }
    expected_ids = {draft_id for draft_id, _ in grids["control"]}
    if grids["control"] != grids[arm] or expected_ids != eligible:
        return {
            "n_tasks": 0, "estimable": False,
            "reason": "incomplete paired revision grid for coherent drafts",
        }

    outcome = {
        "held": lambda row: row["held_pass"],
        "accepted_correct": lambda row: row["accepted"] and row["held_pass"],
        "accepted_type_clean_correct": lambda row: row["accepted_type_clean_correct"],
        "accepted_behavioral_defect": lambda row: row["accepted"] and not row["held_pass"],
        "accepted_semantic_defect": lambda row: row["accepted"] and not row["semantic_clean"],
        "gate_checked": lambda row: bool(row["gate_invocations"]),
        "gate_rejection": lambda row: bool(row["gate_rejections"]),
        "gate_acceptance": lambda row: bool(row["gate_acceptances"]),
        "unsubmitted": lambda row: row["unsubmitted"],
    }
    if field not in outcome:
        raise ValueError(f"unsupported end-to-end field: {field}")
    by_arm_draft = {
        which: {
            draft_id: [row for row in selected
                       if row["arm"] == which and row["draft_id"] == draft_id]
            for draft_id in eligible
        }
        for which in ("control", arm)
    }
    task_drafts: dict[str, list[dict]] = {}
    for draft in drafts:
        task_drafts.setdefault(draft["task"], []).append(draft)

    def task_value(which: str, task: str) -> float:
        values = []
        for draft in task_drafts[task]:
            if draft["draft_id"] not in eligible:
                values.append(0.0)
                continue
            values.append(statistics.mean(
                float(outcome[field](row)) for row in by_arm_draft[which][draft["draft_id"]]
            ))
        return statistics.mean(values)

    tasks = sorted(task_drafts)

    def effect(sample: list[str]) -> float:
        return statistics.mean(
            task_value(arm, task) - task_value("control", task) for task in sample
        )

    rng = random.Random(seed)
    samples = [effect([rng.choice(tasks) for _ in tasks]) for _ in range(bootstrap)]
    return {
        "n_tasks": len(tasks), "estimable": True, "mean_delta": effect(tasks),
        "ci95": [quantile(samples, 0.025), quantile(samples, 0.975)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drafts", required=True)
    parser.add_argument("--revisions")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()
    payload = json.loads(Path(args.drafts).read_text())
    drafts = payload["drafts"]
    coherent = [draft for draft in drafts if draft["coherent"]]
    submitted = [draft for draft in drafts if draft.get("draft_submitted")]
    opportunities = [draft for draft in coherent if any(
        item["classification"] == "semantic" for item in draft["draft_diagnostics"]
    )]

    print(f"CHECKER DRAFT AUDIT: {payload.get('kind', 'unknown')}")
    if payload.get("selection_is_not_a_natural_opportunity_sample"):
        print("CASE SERIES: opportunity-conditioned selection; prevalence and unconditional value are not estimable")
    print(f"drafts={len(drafts)} submitted={len(submitted)} coherent={len(coherent)} "
          f"semantic-opportunity={len(opportunities)}")
    print(f"unconditional opportunity rate={len(opportunities)/len(drafts):.3f}")
    if coherent:
        print(f"opportunity rate among coherent={len(opportunities)/len(coherent):.3f}")
    for draft in drafts:
        semantic = sum(item["classification"] == "semantic"
                       for item in draft["draft_diagnostics"])
        partial = len(draft["draft_diagnostics"]) - semantic
        print(f"  {draft['task']}: submitted={draft.get('draft_submitted')} "
              f"coherent={draft['coherent']} visible={draft['visible_pass']} "
              f"held={draft['held_pass']} semantic={semantic} syntax/partial={partial}")
    coherent_by_task: dict[str, list[dict]] = {}
    for draft in coherent:
        coherent_by_task.setdefault(draft["task"], []).append(draft)
    task_weighted_opportunity = (
        statistics.mean(
            statistics.mean(any(
                item["classification"] == "semantic" for item in draft["draft_diagnostics"]
            ) for draft in task_drafts)
            for task_drafts in coherent_by_task.values()
        ) if coherent_by_task else math.nan
    )
    print("COHORT AUDIT " + json.dumps({
        "tasks_generated": len({draft["task"] for draft in drafts}),
        "drafts_generated": len(drafts),
        "drafts_submitted": len(submitted),
        "drafts_unsubmitted": len(drafts) - len(submitted),
        "drafts_coherent_submitted": sum(
            bool(draft.get("draft_submitted") and draft["coherent"]) for draft in drafts
        ),
        "drafts_submitted_incoherent": sum(
            bool(draft.get("draft_submitted") and not draft["coherent"]) for draft in drafts
        ),
        "drafts_checker_positive": len(opportunities),
        "task_weighted_opportunity_rate_among_coherent": task_weighted_opportunity,
    }, allow_nan=True))
    if not args.revisions:
        return

    rows = json.loads(Path(args.revisions).read_text())["rows"]
    eligible_ids = {
        draft["draft_id"] for draft in drafts
        if revision_eligible(draft)
    }
    unexpected = sorted({row["draft_id"] for row in rows} - eligible_ids)
    if unexpected:
        raise ValueError(f"revision rows exist for ineligible drafts: {unexpected}")
    heading = ("SELECTED RECOVERED-WORKSPACE REVISION EFFICACY"
               if payload.get("selection_is_not_a_natural_opportunity_sample")
               else "REVISION EFFICACY AMONG COHERENT SUBMITTED DRAFTS")
    print(f"\n{heading}")
    for opportunity_only, subset in (
        (False, "coherent_submitted_revision"),
        (True, "checker_positive_revision"),
    ):
        selected = [row for row in rows if not opportunity_only or row["opportunity"]]
        print(subset)
        for arm in sorted({row["arm"] for row in selected}):
            print(f"  {arm}: {json.dumps(summarize_rows([row for row in selected if row['arm'] == arm]), allow_nan=True)}")
        contrasts = {}
        for arm in sorted({row["arm"] for row in selected} - {"control"}):
            for field in ("held_pass", "accepted_type_clean", "accepted_type_clean_correct"):
                contrasts[f"{arm}_minus_control_{field}"] = paired_contrast(
                    rows, arm, field, opportunity_only, args.bootstrap, args.seed
                )
        print(json.dumps(contrasts, indent=2, allow_nan=True))
    print("\nEND-TO-END OUTCOMES ACROSS ALL NATURAL DRAFTS")
    for arm in sorted({row["arm"] for row in rows}):
        print(f"  {arm}: {json.dumps(end_to_end_summary(drafts, rows, arm), allow_nan=True)}")
        print("    deployment_cost: " + json.dumps(
            expected_cost_per_accepted_correct(
                drafts, rows, arm, False, args.bootstrap, args.seed
            ), allow_nan=True,
        ))
        print("    type_clean_deployment_cost: " + json.dumps(
            expected_cost_per_accepted_correct(
                drafts, rows, arm, False, args.bootstrap, args.seed,
                require_type_clean=True,
            ), allow_nan=True,
        ))
    end_to_end_contrasts = {}
    for arm in sorted({row["arm"] for row in rows} - {"control"}):
        for field in (
            "held", "accepted_correct", "accepted_type_clean_correct",
            "accepted_behavioral_defect", "accepted_semantic_defect", "gate_checked",
            "gate_rejection", "gate_acceptance", "unsubmitted",
        ):
            end_to_end_contrasts[f"{arm}_minus_control_{field}"] = end_to_end_contrast(
                drafts, rows, arm, field, args.bootstrap, args.seed
            )
    print("END-TO-END TASK-BOOTSTRAP ARM CONTRASTS")
    print(json.dumps(end_to_end_contrasts, indent=2, allow_nan=True))
    print("A null is not equivalence unless its task-bootstrap pass CI lies inside +/-0.10.")


if __name__ == "__main__":
    main()
