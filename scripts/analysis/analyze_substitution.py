#!/usr/bin/env python3
"""Baseline vs substitution-trained comparison on the held-out reread instances.

Primary outcome: read_after_span (did the model reopen the file the auto-delivered span came
from). Secondary: held-out pass, total tokens, edits. Both artifacts are `auto_neutral` rows on
the SAME 12 apparatus instances at temperature 0, so the comparison is paired by task.

Usage:
  python scripts/analysis/analyze_substitution.py \
      --baseline runs/pilot/navigation_v2_reread_qwen36_27b_apparatus.json \
      --trained  runs/pilot/navigation_v2_reread_qwen36_27b_apparatus_trained.json \
      [--harvest runs/agent/substitution_harvest_qwen36_27b.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(path: str, arm: str = "auto_neutral") -> dict:
    d = json.loads(Path(path).read_text())
    return {r["task"]: r for r in d["rows"] if r["arm"] == arm}, d


def summarize(name: str, rs: dict) -> None:
    n = len(rs)
    if not n:
        print(f"  {name}: no rows")
        return
    v = list(rs.values())
    print(f"  {name:10s} n={n:2d}  reread={sum(r['read_after_span'] for r in v)}/{n}  "
          f"pass={sum(r['resolved'] for r in v)}/{n}  "
          f"mean_in={sum(r['in_tokens'] for r in v)/n:7.1f}  "
          f"mean_out={sum(r['out_tokens'] for r in v)/n:6.1f}  "
          f"mean_total={sum(r['in_tokens'] + r['out_tokens'] for r in v)/n:7.1f}  "
          f"mean_edits={sum(r['n_edits'] for r in v)/n:4.2f}  "
          f"mean_reads={sum(r['n_reads'] for r in v)/n:4.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--trained", required=True)
    ap.add_argument("--harvest", default=None)
    ap.add_argument("--arm", default="auto_neutral")
    a = ap.parse_args()

    base, bd = rows(a.baseline, a.arm)
    trn, td = rows(a.trained, a.arm)
    shared = sorted(set(base) & set(trn))

    print("model  base:", bd.get("model"), bd.get("model_meta", {}).get("revision"))
    print("model trained:", td.get("model"), td.get("model_meta", {}).get("adapter"))
    print(f"\n=== arm {a.arm}, {len(shared)} shared held-out instances ===")
    summarize("baseline", {k: base[k] for k in shared})
    summarize("trained", {k: trn[k] for k in shared})

    print("\n=== paired reread transitions (baseline -> trained) ===")
    cells = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
    for k in shared:
        cells[(bool(base[k]["read_after_span"]), bool(trn[k]["read_after_span"]))] += 1
    print(f"  persisted (reread -> reread)   {cells[(True, True)]}")
    print(f"  REMOVED   (reread -> none)     {cells[(True, False)]}")
    print(f"  induced   (none  -> reread)    {cells[(False, True)]}")
    print(f"  absent    (none  -> none)      {cells[(False, False)]}")

    both = [k for k in shared if base[k]["resolved"] and trn[k]["resolved"]]
    if both:
        bt = sum(base[k]["in_tokens"] + base[k]["out_tokens"] for k in both) / len(both)
        tt = sum(trn[k]["in_tokens"] + trn[k]["out_tokens"] for k in both) / len(both)
        print(f"\n=== tokens on {len(both)} instances resolved in BOTH arms ===")
        print(f"  baseline {bt:.1f} -> trained {tt:.1f}   (ratio base/trained {bt/tt:.3f}, "
              f"delta {tt-bt:+.1f})")

    print("\n=== per-instance ===")
    print(f"  {'task':24s} {'base(reread,pass,tok)':26s} {'trained(reread,pass,tok)':26s}")
    for k in shared:
        b, t = base[k], trn[k]
        print(f"  {k:24s} {str(b['read_after_span']):5s} {str(b['resolved']):5s} "
              f"{b['in_tokens']+b['out_tokens']:6d}      "
              f"{str(t['read_after_span']):5s} {str(t['resolved']):5s} "
              f"{t['in_tokens']+t['out_tokens']:6d}")

    if a.harvest:
        h = json.loads(Path(a.harvest).read_text())
        hr = h["rows"]
        n = len(hr)
        print(f"\n=== harvest ({h.get('split')}, {len({r['task'] for r in hr})} instances) ===")
        print(f"  rollouts {n}  held-out pass {sum(r['held_out_pass'] for r in hr)}/{n}  "
              f"redirect fired {sum(r['relabel_keep_index'] is not None for r in hr)}/{n}  "
              f"clean SFT demos {sum(r['sft_keep'] for r in hr)}/{n}")
        print(f"  harvest tasks: {sorted({r['task'] for r in hr})}")
        overlap = {r["task"] for r in hr} & set(shared)
        print(f"  harvest/retest overlap: {sorted(overlap) or 'NONE (disjoint)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
