#!/usr/bin/env python3
"""C36: is the substitution adapter's read suppression CONDITIONAL or BLANKET?

Reads one or more run artifacts from `scripts/experiments/run_navigation_readreq.py` (untrained
and trained), builds the 2x3 condition x arm table, checks the pre-registered validity gates, and
reports the pre-registered verdict. NO verdict is printed unless every validity gate holds.

Usage:
  python scripts/analysis/analyze_navigation_readreq.py \
      runs/readreq/navigation_v2_<id>_main_untrained.json \
      runs/readreq/navigation_v2_<id>_main_trained.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ARMS = ("push_insufficient", "push_sufficient", "push_chained")
CONDITIONS = ("untrained", "trained")
METRICS = (
    ("read_after_span", "read_after_span"),
    ("retrieved_missing_information", "retrieved (read|grep)"),
    ("sought_missing_information_before_first_edit", "sought (incl. defn/probe)"),
    ("repair_correct", "repair_correct (visible AND held-out)"),
    ("held_out_pass", "held_out_pass (raw, C33/C35-comparable)"),
)
# pre-registered; all must hold or no verdict is reported.
# V1 is a CONJUNCTION: "reading recovers the information" is only established if the untrained
# model both retrieved and passed. Held-out pass alone can be reached without any retrieval, so
# on its own it does not establish recoverability.
VALIDITY = (
    ("V1a", "untrained", "push_insufficient", "repair_correct", ">=", 9),
    ("V1b", "untrained", "push_insufficient", "retrieved_missing_information", ">=", 9),
    ("V2", "untrained", "push_sufficient", "read_after_span", ">=", 9),
    ("V3a", "trained", "push_sufficient", "read_after_span", "<=", 2),
    ("V3b", "trained", "push_sufficient", "repair_correct", ">=", 9),
    ("V4", "trained", "push_chained", "repair_correct", ">=", 9),
)
MAX_CATEGORY_C = 2          # V5: rows that passed with no seeking at all -> insufficiency violated
MAX_CATEGORY_E = 2          # V6: rows that passed off the involuntary post-edit view
# The verdict keys on RETRIEVAL, not on the broader `sought` disjunction: a cell that never reads
# but emits one refused <defn> and then guesses would otherwise score sought=12/12 and be reported
# as CONDITIONAL, which is the exact behaviour the experiment exists to detect.
DECISION = {
    "conditional": {"retrieved": 8, "repair_correct": 8},
    "blanket": {"retrieved": 2, "sought": 2, "repair_correct": 3},
}


def load(paths: list[str]) -> tuple[list[dict], list[dict]]:
    rows, metas = [], []
    for path in paths:
        data = json.loads(Path(path).read_text())
        condition = data.get("model_meta", {}).get("condition") or (
            "trained" if data.get("model_meta", {}).get("adapter") else "untrained"
        )
        for row in data["rows"]:
            rows.append({**row, "condition": row.get("condition") or condition,
                         "artifact": Path(path).name})
        metas.append({"artifact": Path(path).name, "model": data.get("model"),
                      "adapter": data.get("model_meta", {}).get("adapter"),
                      "condition": condition, "split": data.get("split"),
                      "n_rows": len(data["rows"])})
    return rows, metas


def cell(rows: list[dict], condition: str, arm: str) -> list[dict]:
    return [r for r in rows if r["condition"] == condition and r["arm"] == arm]


def fraction(cell_rows: list[dict], key: str) -> str:
    if not cell_rows:
        return "   -   "
    return f"{sum(1 for r in cell_rows if r.get(key)):>2}/{len(cell_rows):<2}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+")
    args = parser.parse_args()
    rows, metas = load(args.artifacts)

    print("=== artifacts ===")
    for meta in metas:
        print(f"  {meta['artifact']}  condition={meta['condition']} split={meta['split']} "
              f"model={meta['model']} adapter={meta['adapter']} n={meta['n_rows']}")

    print("\n=== 2x3 grid (condition x arm) ===")
    header = "  " + " " * 12 + "".join(f"{arm:>26s}" for arm in ARMS)
    for key, label in METRICS:
        print(f"\n  {label}")
        print(header)
        for condition in CONDITIONS:
            line = f"  {condition:>12s}"
            for arm in ARMS:
                line += f"{fraction(cell(rows, condition, arm), key):>26s}"
            print(line)

    print("\n=== seeking components (which action, if any, the cell actually took) ===")
    for condition in CONDITIONS:
        for arm in ARMS:
            sub = cell(rows, condition, arm)
            if not sub:
                continue
            print(f"  {condition:>9s} {arm:18s} "
                  f"read={fraction(sub, 'sought_component_read')} "
                  f"grep={fraction(sub, 'sought_component_grep')} "
                  f"defn_blocked={fraction(sub, 'sought_component_defn_blocked')} "
                  f"probe={fraction(sub, 'sought_component_probe')}")

    print("\n=== per-template (decisive cell: trained x push_insufficient) ===")
    decisive = cell(rows, "trained", "push_insufficient")
    for template in sorted({r["template"] for r in decisive}):
        sub = [r for r in decisive if r["template"] == template]
        print(f"  {template:24s} n={len(sub)} "
              f"read={fraction(sub, 'read_after_span')} "
              f"retrieved={fraction(sub, 'retrieved_missing_information')} "
              f"sought={fraction(sub, 'sought_missing_information_before_first_edit')} "
              f"repair={fraction(sub, 'repair_correct')} "
              f"held={fraction(sub, 'held_out_pass')}")

    print("\n=== outcome categories ===")
    for condition in CONDITIONS:
        for arm in ARMS:
            sub = cell(rows, condition, arm)
            if not sub:
                continue
            counts = Counter(r.get("outcome_category") for r in sub)
            print(f"  {condition:>9s} {arm:18s} " +
                  "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    print("\n=== hygiene ===")
    stale = [r for r in rows if r.get("stale_bytecode_suspected")]
    attempts = sum(r.get("n_heldout_read_attempts", 0) for r in rows)
    bad_gen = [r for r in rows if r.get("repair_correct") and r.get("generalization_pass") is False]
    # a held-out pass on a workspace that fails the visible test is the disclosed xor-family
    # artifact, not a repair; it must never be counted as a solve
    spurious = [r for r in rows if r.get("held_out_pass") and not r.get("visible_pass")]
    # a row whose stream contains the hidden line while the ledger recorded no retrieval and no
    # post-edit view is a DETECTION failure, not a blanket signature; it must never be binned as B
    silent = [r for r in rows
              if r.get("hidden_line_in_context")
              and not r.get("n_target_reads_total_normalised")
              and not r.get("n_target_greps_total")
              and r.get("first_target_read_channel") != "post_edit_view"]
    print(f"  stale_bytecode_suspected       {len(stale)}/{len(rows)}")
    print(f"  heldout read attempts (denied) {attempts}")
    print(f"  post-edit redactions applied   {sum(r.get('n_post_edit_redactions', 0) for r in rows)}")
    print(f"  passed but did NOT generalise  {len(bad_gen)}")
    print(f"  server errors                  {sum(1 for r in rows if r.get('server_errors'))}")
    print(f"  held-out pass w/o visible pass {len(spurious)}"
          + (f"  {[r['task'] for r in spurious]}" if spurious else ""))
    print(f"  SILENT detection failures      {len(silent)}"
          + (f"  {[r['task'] for r in silent]}" if silent else ""))

    print("\n=== pre-registered validity gates ===")
    ok = True
    for name, condition, arm, key, op, bound in VALIDITY:
        sub = cell(rows, condition, arm)
        value = sum(1 for r in sub if r.get(key))
        if op == "<=":
            passed = bool(sub) and value <= bound
        else:
            passed = bool(sub) and value >= bound
        ok &= passed
        print(f"  {name:4s} {condition:>9s} x {arm:18s} {key:>12s} {value}/{len(sub)} "
              f"(need {op} {bound})  {'OK' if passed else 'FAIL'}")
    # V5/V6 are per CONDITION, not pooled: instance validity is a property of the rollouts in a
    # cell, so three lucky untrained rows must not void the trained cell (or vice versa).
    for name, category, bound, why in (
        ("V5", "C_no_seeking_pass", MAX_CATEGORY_C, "insufficiency violated"),
        ("V6", "E_post_edit_view_then_repair", MAX_CATEGORY_E,
         "passed off the involuntary post-edit view"),
    ):
        counts = {
            condition: sum(1 for r in cell(rows, condition, "push_insufficient")
                           if r.get("outcome_category") == category)
            for condition in CONDITIONS
        }
        worst = max(counts.values(), default=0)
        passed = worst <= bound
        ok &= passed
        detail = " ".join(f"{c}={v}" for c, v in counts.items())
        print(f"  {name:4s} category {category:28s} {detail} (need <= {bound} per condition)  "
              f"{'OK' if passed else 'FAIL — ' + why + ', the run is void'}")

    print("\n=== verdict ===")
    if not ok:
        print("  NO VERDICT: a pre-registered validity gate failed. Report descriptively only.")
        return 1
    retrieved = sum(1 for r in decisive if r.get("retrieved_missing_information"))
    sought = sum(1 for r in decisive if r.get("sought_missing_information_before_first_edit"))
    passes = sum(1 for r in decisive if r.get("repair_correct"))
    raw_held = sum(1 for r in decisive if r.get("held_out_pass"))
    n = len(decisive)
    if (retrieved >= DECISION["conditional"]["retrieved"]
            and passes >= DECISION["conditional"]["repair_correct"]):
        verdict = ("CONDITIONAL — the adapter goes and gets the missing information and repairs; "
                   "the trained policy is sufficiency-conditional, not blanket")
    elif (retrieved <= DECISION["blanket"]["retrieved"]
          and sought <= DECISION["blanket"]["sought"]
          and passes <= DECISION["blanket"]["repair_correct"]):
        verdict = ("BLANKET — the adapter neither retrieves nor seeks, and fails; the learned "
                   "policy is actively harmful when the span does not suffice")
    else:
        verdict = "MIXED — report descriptively; pre-registered follow-up is 3 seeds at temp 0.7"
    print(f"  trained x push_insufficient: retrieved={retrieved}/{n} sought={sought}/{n} "
          f"repair_correct={passes}/{n} (raw held_out_pass={raw_held}/{n})")
    print(f"  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
