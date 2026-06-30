#!/usr/bin/env python3
"""Analyze the runtime-feedback matrix (scripts/run_runtime_frontier.sh output).

Three arms per model on the `runtime` suite, scored HELD-OUT (`resolved`):
  R0 no-test   : the agent cannot run the code (reason from source + shown spec).
  R1 run       : the agent elects to run the visible test.
  R2 auto-fb   : the env volunteers the visible-test result after each edit.

The contrast that answers "does runtime feedback help a strong agent":
  R1 - R0  = the value of having execution at all (the headline).
  R2 - R1  = whether the agent under-uses execution (handing it over for free).
Stratified by simulation difficulty (hard vs easy) to test whether execution's value
rises with how hard the code is to mentally execute.

Run: .venv-streams.system/bin/python scripts/analyze_runtime.py
"""
import json
import glob
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = lambda p: os.path.join(ROOT, "runs", "agent", p)

ARMS = [("R0 no-test", "r0_notest"), ("R1 run", "r1_run"), ("R2 auto-fb", "r2_auto")]
MODELS = [("deepseek-v3.1", "deepseek"), ("sonnet-4.5", "sonnet45")]


def rows(path):
    if not os.path.exists(path):
        return []
    return json.load(open(path)).get("rows", [])


def pct(rs, key="resolved"):
    rs = [r for r in rs if r.get(key) is not None]
    return (100.0 * sum(bool(r[key]) for r in rs) / len(rs)) if rs else float("nan")


def mean(rs, f):
    rs = [r for r in rs if f(r) is not None]
    return (sum(f(r) for r in rs) / len(rs)) if rs else float("nan")


def main():
    # task -> sim difficulty, from the suite
    import sys
    sys.path.insert(0, ROOT)
    from scripts.synth_tasks_runtime import TASKS_RUNTIME
    sim = {t["name"]: t["group"] for t in TASKS_RUNTIME}

    print("=" * 78)
    print("RUNTIME-FEEDBACK TEST — held-out pass@1 by model x arm (all tasks)")
    print("=" * 78)
    print(f"{'model':16}", end="")
    for label, _ in ARMS:
        print(f"{label:>13}", end="")
    print(f"{'R1-R0':>9}{'R2-R1':>8}")
    grand = defaultdict(list)
    for mlabel, mkey in MODELS:
        print(f"{mlabel:16}", end="")
        vals = {}
        for _, akey in ARMS:
            rs = rows(A(f"rt_{mkey}_{akey}.json")) or rows(A(f"rt_{mkey}_{akey}.json.partial"))
            grand[akey] += rs
            vals[akey] = pct(rs)
            print(f"{vals[akey]:>12.1f}%", end="")
        d10 = vals["r1_run"] - vals["r0_notest"]
        d21 = vals["r2_auto"] - vals["r1_run"]
        print(f"{d10:>+8.1f}{d21:>+8.1f}")
    print("-" * 78)
    # pooled
    print(f"{'POOLED':16}", end="")
    pv = {}
    for _, akey in ARMS:
        pv[akey] = pct(grand[akey])
        print(f"{pv[akey]:>12.1f}%", end="")
    print(f"{pv['r1_run']-pv['r0_notest']:>+8.1f}{pv['r2_auto']-pv['r1_run']:>+8.1f}")

    # ---- stratified by simulation difficulty ----
    print("\n" + "=" * 78)
    print("STRATIFIED by simulation difficulty (pooled over both models)")
    print("=" * 78)
    print(f"{'difficulty':16}", end="")
    for label, _ in ARMS:
        print(f"{label:>13}", end="")
    print(f"{'R1-R0':>9}")
    for diff in ("hard", "easy"):
        print(f"{diff:16}", end="")
        vals = {}
        for _, akey in ARMS:
            rs = [r for r in grand[akey] if sim.get(r["task"]) == diff]
            vals[akey] = pct(rs)
            print(f"{vals[akey]:>12.1f}%", end="")
        print(f"{vals['r1_run']-vals['r0_notest']:>+8.1f}")

    # ---- behaviour: did the agent actually run tests when it could? ----
    print("\n" + "=" * 78)
    print("EXECUTION BEHAVIOUR (R1: did the agent elect to run; how many edits/turns)")
    print("=" * 78)
    for mlabel, mkey in MODELS:
        for alabel, akey in ARMS:
            rs = rows(A(f"rt_{mkey}_{akey}.json")) or rows(A(f"rt_{mkey}_{akey}.json.partial"))
            if not rs:
                continue
            ran = mean(rs, lambda r: 1.0 if (r.get("n_test", 0) + r.get("n_auto_test", 0)) > 0 else 0.0)
            print(f"  {mlabel:14} {alabel:12} ran_test={100*ran:5.0f}%  "
                  f"edits={mean(rs, lambda r: r.get('n_edit',0)):.2f}  "
                  f"turns={mean(rs, lambda r: r.get('turns',0)):.2f}  "
                  f"visible_pass={pct(rs,'visible_pass'):5.1f}%  held_out={pct(rs):5.1f}%")

    # ---- per-task held-out pass@1 (R0 vs R1), pooled ----
    print("\n" + "=" * 78)
    print("PER-TASK held-out pass@1 (pooled over models): R0 -> R1  (n per cell)")
    print("=" * 78)
    for t in TASKS_RUNTIME:
        nm = t["name"]
        r0 = [r for r in grand["r0_notest"] if r["task"] == nm]
        r1 = [r for r in grand["r1_run"] if r["task"] == nm]
        print(f"  [{t['group']:4}] {nm:20} R0={pct(r0):5.1f}%  R1={pct(r1):5.1f}%  "
              f"(n={len(r0)}/{len(r1)})  delta={pct(r1)-pct(r0):+.1f}")

    # ---- spend ----
    tot = 0.0
    for f in glob.glob(A("rt_*.json")):
        try:
            tot += json.load(open(f)).get("total_cost_usd", 0.0)
        except Exception:
            pass
    print(f"\ntotal API spend (final jsons): ${tot:.4f}")


if __name__ == "__main__":
    main()
