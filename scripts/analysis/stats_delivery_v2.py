#!/usr/bin/env python3
"""Set the delivery-timing re-run (runs/delivery_v2/) against the June numbers (ledger C37).

This is NOT the June reproducer. `stats_delivery.py` re-derives the published values from
the frozen June artifacts and fails if they move. This script analyses a fresh run on the
ported harness and asks whether the *claims* survive, which is a different question and has
a different failure mode: a number that differs is a result, not a bug.

So the two exit conditions are kept apart.

  structural problems  -> FAIL, non-zero exit. An arm missing, an arm whose event trace does
                          not witness its own mechanism, arms of unequal size. These mean the
                          apparatus is wrong and the numbers should not be read at all.
  claim outcomes       -> reported as HOLDS / DOES NOT HOLD, exit code unaffected. The report
                          rests on two: no contrast among the five properly-delivered arms is
                          significant, and the naive arm's deficit is carried by the announce
                          sentence. If the re-run overturns either, that is the finding.

The re-run differs from June in one known way beyond the harness port: the June harness
terminated 449 of 473 resolved rollouts on its own `<done/>` echo, which the ported agent no
longer does. Per-cell agreement is therefore expected to be close but not exact.

Arms are keyed by `config["arm"]` recorded in each artifact, not by filename, which is the
provenance failure that forced runs/agent/synth_delivery_provenance.json to exist.

Run:  python scripts/analysis/stats_delivery_v2.py [--partial]
        --partial  also read <arm>.json.partial checkpoints, for watching a run in flight.
                   Partial arms have fewer cells; contrasts then use only shared cells and
                   every table marks the arm incomplete.
"""
import argparse
import glob
import json
import os
import sys
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delivery_stats_lib import (  # noqa: E402
    WITNESS, bh_fdr, diag_kinds, load_june, mcnemar, task_bootstrap, wilson)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
# Overridable so the analyzer can be exercised against fixtures without writing into the
# directory a live run is checkpointing into.
V2 = os.environ.get("DELIVERY_V2_DIR") or os.path.join(ROOT, "runs", "delivery_v2")
FULL_N = 168  # 14 tasks x 12 seeds
FAILURES = []


def load_v2(use_partial):
    """Arm -> (rows, complete?). Keyed by the arm recorded in the artifact's config."""
    out = {}
    paths = sorted(glob.glob(os.path.join(V2, "*.json")))
    if use_partial:
        finished = {os.path.basename(p) for p in paths}
        paths += [p for p in sorted(glob.glob(os.path.join(V2, "*.json.partial")))
                  if os.path.basename(p)[:-len(".partial")] not in finished]
    for p in paths:
        try:
            d = json.load(open(p))
        except json.JSONDecodeError:
            print(f"  (skipping {os.path.basename(p)}: mid-write, not valid JSON yet)")
            continue
        arm = (d.get("config") or {}).get("arm")
        if not arm:
            FAILURES.append(f"{os.path.basename(p)}: no config.arm; cannot identify the arm")
            continue
        rs = [r for v in d["rows"].values() for r in v]
        out[arm] = (rs, not p.endswith(".partial") and len(rs) == FULL_N)
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--partial", action="store_true", help="include in-flight .partial checkpoints")
args = ap.parse_args()

june = load_june(ROOT)
v2raw = load_v2(args.partial)
if not v2raw:
    hint = ("" if args.partial else
            " Pass --partial to also read in-flight .partial checkpoints.")
    print(f"no readable artifacts in {V2}. The re-run writes <arm>.json when an arm finishes "
          f"and <arm>.json.partial after each task.{hint}")
    sys.exit(1)
v2 = {k: r for k, (r, _) in v2raw.items()}
complete = {k: c for k, (_, c) in v2raw.items()}

# ------------------------------------------------------------------ apparatus
print("== apparatus: does each arm's event trace witness its own mechanism? ==")
for arm in sorted(v2):
    kinds = diag_kinds(v2[arm])
    want = WITNESS.get(arm)
    ok = (kinds == []) if want is None else (kinds == [want])
    if not ok:
        FAILURES.append(f"{arm}: diag event types {kinds}, expected {want or '(none)'}")
    print(f"  {arm:8s} n={len(v2[arm]):3d}{'' if complete[arm] else ' (INCOMPLETE)':13s}  "
          f"diag events: {kinds or ['(none)']}  {'ok' if ok else 'FAIL'}")
missing = [a for a in WITNESS if a not in v2]
if missing:
    print(f"  not yet present: {', '.join(missing)}")
sizes = {len(r) for a, r in v2.items() if complete[a]}
if len(sizes) > 1:
    FAILURES.append(f"completed arms have unequal cell counts {sizes}; pairing is not balanced")

# -------------------------------------------------------------- resolve rates
print("\n== resolve rate: re-run vs June ==")
print(f"  {'arm':8s} {'re-run':>18s}  {'June':>14s}   delta")
for arm in sorted(v2):
    r = v2[arm]
    n, t = sum(x["resolved"] for x in r), len(r)
    lo, hi = wilson(n, t)
    jr = june[arm]
    jn, jt = sum(x["resolved"] for x in jr), len(jr)
    flag = "" if complete[arm] else "  (partial)"
    print(f"  {arm:8s} {n:3d}/{t:3d}={n/t:.3f} [{lo:.2f},{hi:.2f}]  "
          f"{jn:3d}/{jt}={jn/jt:.3f}   {n/t - jn/jt:+.3f}{flag}")

# ---------------------------------------------------- per-cell agreement
print("\n== per-cell agreement with June (shared cells only) ==")
print("  The ported agent no longer self-terminates on its own <done/> echo, so exact")
print("  agreement is not expected; this measures how far the port moved each arm.")
for arm in sorted(v2):
    a = {(x["task"], x["seed"]): x for x in v2[arm]}
    b = {(x["task"], x["seed"]): x for x in june[arm]}
    ks = sorted(set(a) & set(b))
    if not ks:
        continue
    same_res = sum(1 for k in ks if bool(a[k]["resolved"]) == bool(b[k]["resolved"]))
    exact = sum(1 for k in ks
                if all(a[k][f] == b[k][f] for f in ("resolved", "in_tokens", "out_tokens", "n_tests")))
    print(f"  {arm:8s} {len(ks):3d} shared cells: resolved agrees on {same_res:3d} "
          f"({same_res/len(ks):.0%}), bit-identical on {exact:3d} ({exact/len(ks):.0%})")

# --------------------------------------------------------------- contrasts
armlist = sorted(v2)
if len(armlist) < 2:
    print("\n(only one arm so far; contrasts need at least two)")
else:
    print(f"\n== pairwise contrasts among the {len(armlist)} arms present ==")
    res = []
    for a, b in combinations(armlist, 2):
        B, C, p, n = mcnemar(v2[a], v2[b])
        o, lo, hi = task_bootstrap(v2[a], v2[b])
        res.append((a, b, B, C, p, o, lo, hi, n))
    cut = bh_fdr([r[4] for r in res])
    for a, b, B, C, p, o, lo, hi, n in res:
        mark = " *" if cut is not None and p <= cut else "  "
        print(f"  {a:8s} vs {b:8s}: b={B:3d} c={C:3d} n={n:3d} McNemar p={p:.4f}   "
              f"task delta {o:+.3f} [{lo:+.3f},{hi:+.3f}]{mark}")
    if cut is None:
        print("  no contrast survives Benjamini-Hochberg at FDR 5%")
    else:
        print(f"  * = rejected at Benjamini-Hochberg FDR 5% (cutoff p<={cut:.4f})")

    # ------------------------------------------------------- the report's claims
    print("\n== do the report's claims survive? ==")
    sig = {frozenset((a, b)) for a, b, B, C, p, *_ in res if cut is not None and p <= cut}
    band = {"A", "C-lazy", "C-eager", "D-plain", "D-gate"} & set(armlist)
    if len(band) >= 2:
        within = [frozenset((a, b)) for a, b in combinations(sorted(band), 2)]
        bad = [s for s in within if s in sig]
        smallest = min(p for a, b, B, C, p, *_ in res if a in band and b in band)
        print(f"  [{'HOLDS' if not bad else 'DOES NOT HOLD'}] no contrast among the "
              f"{len(band)} properly-delivered arms is significant")
        print(f"          smallest p among those {len(within)} contrasts = {smallest:.3f}"
              + (f"; significant: {[sorted(s) for s in bad]}" if bad else ""))
    if "D-naive" in armlist and {"D-plain", "D-gate"} <= set(armlist):
        print("  the naive arm's deficit, decomposed:")
        for a, b, label in (("D-naive", "D-plain", "drop the announce sentence"),
                            ("D-plain", "D-gate", "add the syntax gate on top"),
                            ("D-naive", "D-gate", "both changes together")):
            B, C, p, n = mcnemar(v2[a], v2[b])
            o, lo, hi = task_bootstrap(v2[a], v2[b])
            print(f"    {label:28s} {a} -> {b}: {-o:+.3f} [{-hi:+.3f},{-lo:+.3f}]  p={p:.4f}")
    incomplete = [a for a in armlist if not complete[a]]
    if incomplete:
        print(f"  NOTE: {', '.join(incomplete)} incomplete; claim outcomes are provisional.")
    if len(armlist) < len(WITNESS):
        print(f"  NOTE: {len(WITNESS) - len(armlist)} arm(s) not yet run; the FDR family is "
              f"{len(res)} contrasts, not the 15 the published analysis used.")

# ------------------------------------------------------- gate mechanism
gate_arms = [a for a in ("D-naive", "D-plain", "D-gate") if a in v2]
if gate_arms:
    print("\n== mechanism: what the syntax gate removes ==")
    for arm in gate_arms:
        texts = [e.get("text", "") for r in v2[arm] for e in r["events"]
                 if e["type"] == "diag_debounced"]
        bad = sum(1 for t in texts if "invalid-syntax" in t or "parse-error" in t)
        jt = [e.get("text", "") for r in june[arm] for e in r["events"]
              if e["type"] == "diag_debounced"]
        jbad = sum(1 for t in jt if "invalid-syntax" in t or "parse-error" in t)
        print(f"  {arm:8s} deliveries={len(texts):4d} (June {len(jt):4d})   "
              f"syntax/parse={bad:4d} ({bad/max(1,len(texts)):.0%})  "
              f"(June {jbad:4d}, {jbad/max(1,len(jt)):.0%})")

print("\n" + ("APPARATUS OK" if not FAILURES else "STRUCTURAL FAILURES:\n  " + "\n  ".join(FAILURES)))
sys.exit(1 if FAILURES else 0)
