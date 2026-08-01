#!/usr/bin/env python3
"""Reproduce the delivery-timing table in REPORT.md (ledger C37) from the committed rollouts.

The question: WHEN should a type diagnostic reach a coding agent? Six delivery arms over the
14-task synthetic authoring suite (`scripts/synth_tasks_delivery.py`), Qwen2.5-Coder-7B-Instruct
with real Pyrefly, temp 0.7, 14 tasks x 12 seeds = 168 paired (task, seed) rollouts per arm.

  A        no feedback
  C-lazy   batched, delivered at the model's next yield
  C-eager  batched, delivered immediately after each edit (the production hook)
  D-naive  live mid-stream, debounce 24 + pause-align + an `announce_lsp` prompt telling the
           model to fix each diagnostic before moving on
  D-plain  live mid-stream, debounce 24 + pause-align, no announce sentence
  D-gate   D-plain plus a syntax gate: deliver only when the file `ast.parse`s

Produced by `scripts/synth_delivery.py --arm <name>`, one artifact per arm. Arms are keyed by
the `arm` each artifact records in its own `config`, never by filename, and each arm's
mechanism is corroborated independently by the diagnostic events in its rows: an arm that
claims to be live but emits no `diag_debounced` is a broken arm whatever its config says.

Every figure the report prints is re-derived here and checked against RECORDED_* below. A
mismatch prints FAIL and exits non-zero, so REPORT.md cannot drift from the artifacts.

Two tests are reported for every contrast. Exact McNemar treats each (task, seed) as an
independent pair, which over-counts because seeds within a task are correlated. The
task-clustered bootstrap resamples the 14 tasks, which is the convention REPORT.md uses ("the
unit is the task; intervals are task-level bootstraps"). Fifteen contrasts are run, so a
Benjamini-Hochberg FDR is applied across the family.

Run:  python scripts/analysis/stats_delivery.py   (from the repo root)
"""
import glob
import json
import math
import os
import random
import sys
from itertools import combinations

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.abspath(ROOT))
from scripts.synth_delivery import ARM_WITNESS  # noqa: E402  (the runner owns the arm table)

RUNS = os.environ.get("DELIVERY_DIR") or os.path.join(ROOT, "runs", "delivery")
FULL_N = 168  # 14 tasks x 12 seeds
FAILURES = []

# Every figure REPORT.md prints, re-checked against the artifacts on every run, at the
# precision the report prints it. Printing a number without checking it is how a typo in the
# report survives a green reproducer, so the rule here is: if the report states it, it is
# below.
RECORDED_RESOLVE = {"C-lazy": 0.500, "C-eager": 0.494, "D-gate": 0.452,
                    "D-plain": 0.452, "A": 0.440, "D-naive": 0.310}
RECORDED_DELIVERIES = {"D-naive": 624, "D-plain": 504, "D-gate": 163}
# The report's "Difference vs no feedback" column: arm minus A, so the negation of the
# A-vs-arm contrast, with the interval endpoints swapped as well as negated.
RECORDED_VS_NONE = {"C-lazy":  (+0.060, -0.000, +0.119),
                    "C-eager": (+0.054, -0.024, +0.143),
                    "D-naive": (-0.131, -0.196, -0.065),
                    "D-plain": (+0.012, -0.083, +0.083),
                    "D-gate":  (+0.012, -0.030, +0.060)}
RECORDED_FDR_CUTOFF = 0.0071     # footnote
RECORDED_BAND_SMALLEST_P = 0.143  # footnote: smallest p among the properly-delivered arms
# The paragraph below the table: (delta, lo, hi, p)
RECORDED_DECOMP = {("D-naive", "D-plain"): (+0.143, +0.054, +0.238, 0.0022),
                   ("D-plain", "D-gate"):  (+0.000, -0.089, +0.113, 1.0000)}
RECORDED_BROKEN_SHARE = {"D-plain": 0.74, "D-gate": 0.25}   # share of live deliveries


def check(label, got, want, tol):
    ok = abs(got - want) <= tol
    if not ok:
        FAILURES.append(f"{label}: got {got}, report says {want}")
    return "ok  " if ok else "FAIL"


def wilson(n, t, z=1.96):
    p = n / t
    d = 1 + z * z / t
    c = (p + z * z / (2 * t)) / d
    h = z * math.sqrt(p * (1 - p) / t + z * z / (4 * t * t)) / d
    return c - h, c + h


def mcnemar(X, Y):
    """Exact two-sided McNemar over paired (task, seed) units."""
    ix = {(r["task"], r["seed"]): r["resolved"] for r in X}
    iy = {(r["task"], r["seed"]): r["resolved"] for r in Y}
    keys = [k for k in ix if k in iy]
    b = sum(1 for k in keys if ix[k] and not iy[k])
    c = sum(1 for k in keys if iy[k] and not ix[k])
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
    return b, c, p, len(keys)


def by_task(rs):
    d = {}
    for r in rs:
        d.setdefault(r["task"], []).append(bool(r["resolved"]))
    return {k: sum(v) / len(v) for k, v in d.items()}


def task_bootstrap(X, Y, B=20000, seed=0):
    """Mean per-task resolve difference, 95% CI over tasks resampled with replacement."""
    x, y = by_task(X), by_task(Y)
    ts = sorted(set(x) & set(y))
    rnd = random.Random(seed)
    obs = sum(x[t] - y[t] for t in ts) / len(ts)
    ds = []
    for _ in range(B):
        s = [ts[rnd.randrange(len(ts))] for _ in ts]
        ds.append(sum(x[t] - y[t] for t in s) / len(s))
    ds.sort()
    return obs, ds[int(0.025 * B)], ds[int(0.975 * B)]


def bh_fdr(ps, q=0.05):
    """Benjamini-Hochberg: the largest p still rejected at FDR q, or None."""
    m = len(ps)
    cut = None
    for i, p in enumerate(sorted(ps), 1):
        if p <= i * q / m:
            cut = p
    return cut


def load():
    """Arm -> rows, from each artifact's own recorded config."""
    out = {}
    for p in sorted(glob.glob(os.path.join(RUNS, "*.json"))):
        d = json.load(open(p))
        arm = (d.get("config") or {}).get("arm")
        if not arm:
            FAILURES.append(f"{os.path.basename(p)}: no config.arm; the arm cannot be identified")
            continue
        out[arm] = [r for v in d["rows"].values() for r in v]
    return out


ARMS = load()
if not ARMS:
    print(f"no rollouts in {RUNS}")
    sys.exit(1)

# --------------------------------------------------------------------- apparatus
print("== apparatus: each arm's delivery mechanism, witnessed by its own event trace ==")
for arm in sorted(ARMS):
    kinds = sorted({e["type"] for r in ARMS[arm] for e in r["events"]
                    if e["type"].startswith("diag")})
    want = ARM_WITNESS.get(arm)
    ok = (kinds == []) if want is None else (kinds == [want])
    if not ok:
        FAILURES.append(f"{arm}: diag event types {kinds}, expected {want or '(none)'}")
    print(f"  {arm:8s} n={len(ARMS[arm]):3d}  diag events: {kinds or ['(none)']}  "
          f"{'ok' if ok else 'FAIL'}")
for arm in ARM_WITNESS:
    if arm not in ARMS:
        FAILURES.append(f"{arm}: no artifact in {RUNS}; the arm is missing")
    elif len(ARMS[arm]) != FULL_N:
        FAILURES.append(f"{arm}: {len(ARMS[arm])} rollouts, expected {FULL_N}")

# ------------------------------------------------------------------ resolve rates
print(f"\n== resolve rate, 14 tasks x 12 seeds, n={FULL_N} per arm ==")
for arm in sorted(ARMS, key=lambda a: -sum(r["resolved"] for r in ARMS[a])):
    v = ARMS[arm]
    n, t = sum(r["resolved"] for r in v), len(v)
    lo, hi = wilson(n, t)
    st = ("    " if arm not in RECORDED_RESOLVE
          else check(f"resolve {arm}", round(n / t, 3), RECORDED_RESOLVE[arm], 0.001))
    print(f"  {arm:8s} {n:3d}/{t} = {n/t:.3f}  [{lo:.2f},{hi:.2f}]   {st}")

# --------------------------------------------------------------------- contrasts
print("\n== all 15 pairwise contrasts ==")
res = []
for a, b in combinations(sorted(ARMS), 2):
    B, C, p, n = mcnemar(ARMS[a], ARMS[b])
    o, lo, hi = task_bootstrap(ARMS[a], ARMS[b])
    res.append((a, b, B, C, p, o, lo, hi))
cut = bh_fdr([r[4] for r in res])
for a, b, B, C, p, o, lo, hi in res:
    mark = " *" if cut is not None and p <= cut else "  "
    print(f"  {a:8s} vs {b:8s}: b={B:3d} c={C:3d} McNemar p={p:.4f}   "
          f"task delta {o:+.3f} [{lo:+.3f},{hi:+.3f}]{mark}")
print(f"  * = rejected at Benjamini-Hochberg FDR 5% over all {len(res)} contrasts"
      + (f" (cutoff p<={cut:.4f})" if cut is not None else "; nothing rejected"))
if cut is not None and RECORDED_FDR_CUTOFF is not None:
    check("FDR cutoff", round(cut, 4), RECORDED_FDR_CUTOFF, 0.0)

# The report's difference column, which is the negation of the A-vs-arm contrast.
vs_a = {(b if a == "A" else a): (-o, -hi, -lo) if a == "A" else (o, lo, hi)
        for a, b, B, C, p, o, lo, hi in res if "A" in (a, b)}
for arm, (want_d, want_lo, want_hi) in RECORDED_VS_NONE.items():
    if arm not in vs_a:
        FAILURES.append(f"vs-none {arm}: no contrast against A to check the report against")
        continue
    got_d, got_lo, got_hi = vs_a[arm]
    check(f"vs-none {arm} delta", round(got_d, 3), want_d, 0.0)
    check(f"vs-none {arm} CI lo", round(got_lo, 3), want_lo, 0.0)
    check(f"vs-none {arm} CI hi", round(got_hi, 3), want_hi, 0.0)

# The report's claim: the five properly-delivered arms are indistinguishable, and every
# significant contrast involves the naive arm.
sig = {frozenset((a, b)) for a, b, B, C, p, *_ in res if cut is not None and p <= cut}
band = {"A", "C-lazy", "C-eager", "D-plain", "D-gate"} & set(ARMS)
within = [frozenset(pair) for pair in combinations(sorted(band), 2)]
bad = [sorted(s) for s in within if s in sig]
if bad:
    FAILURES.append(f"contrasts among the properly-delivered arms came out significant: {bad}")
if any("D-naive" not in s for s in sig):
    FAILURES.append("a significant contrast does not involve D-naive: "
                    f"{[sorted(s) for s in sig if 'D-naive' not in s]}")
if len(band) > 1:
    smallest = min(p for a, b, B, C, p, *_ in res if a in band and b in band)
    print(f"  none of the {len(within)} contrasts among the {len(band)} properly-delivered "
          f"arms is significant (smallest p={smallest:.3f}).")
    if RECORDED_BAND_SMALLEST_P is not None:
        check("smallest p among the properly-delivered arms",
              round(smallest, 3), RECORDED_BAND_SMALLEST_P, 0.0)

# -------------------------------------------------- decomposing the naive deficit
if {"D-naive", "D-plain", "D-gate"} <= set(ARMS):
    print("\n== decomposing the naive arm's deficit ==")
    for a, b, label in (("D-naive", "D-plain", "drop the announce sentence"),
                        ("D-plain", "D-gate", "add the syntax gate on top"),
                        ("D-naive", "D-gate", "both changes together")):
        B, C, p, n = mcnemar(ARMS[a], ARMS[b])
        o, lo, hi = task_bootstrap(ARMS[a], ARMS[b])
        print(f"  {label:28s} {a} -> {b}: {-o:+.3f} [{-hi:+.3f},{-lo:+.3f}]  McNemar p={p:.4f}")
        want = RECORDED_DECOMP.get((a, b))
        if want:
            check(f"decomposition {a}->{b} delta", round(-o, 3), want[0], 0.0)
            check(f"decomposition {a}->{b} CI lo", round(-hi, 3), want[1], 0.0)
            check(f"decomposition {a}->{b} CI hi", round(-lo, 3), want[2], 0.0)
            check(f"decomposition {a}->{b} p", round(p, 4), want[3], 0.0)

# ------------------------------------------------------------- the gate mechanism
gate_arms = [a for a in ("D-naive", "D-plain", "D-gate") if a in ARMS]
if gate_arms:
    print("\n== mechanism: what the syntax gate removes ==")
    for arm in gate_arms:
        texts = [e.get("text", "") for r in ARMS[arm] for e in r["events"]
                 if e["type"] == "diag_debounced"]
        broken = sum(1 for t in texts if "invalid-syntax" in t or "parse-error" in t)
        st = ("    " if arm not in RECORDED_DELIVERIES
              else check(f"deliveries {arm}", len(texts), RECORDED_DELIVERIES[arm], 0))
        if arm in RECORDED_BROKEN_SHARE and texts:
            check(f"broken-parse share {arm}", round(broken / len(texts), 2),
                  RECORDED_BROKEN_SHARE[arm], 0.0)
        print(f"  {arm:8s} live deliveries={len(texts):4d}  "
              f"mentioning a syntax/parse error={broken:4d} "
              f"({broken/max(1,len(texts)):.0%})   {st}")

print("\n" + ("ALL CHECKS PASS" if not FAILURES else "FAILURES:\n  " + "\n  ".join(FAILURES)))
sys.exit(1 if FAILURES else 0)
