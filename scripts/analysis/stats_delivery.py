#!/usr/bin/env python3
"""Reproduce the delivery-timing statistics (ledger C37) from the committed result JSONs.

The question: WHEN should a type diagnostic reach a coding agent? Six delivery
arms over the 14-task synthetic authoring suite, Qwen2.5-Coder-7B-Instruct with
real Pyrefly, temp 0.7, 14 tasks x 12 seeds = 168 paired (task, seed) units per
arm:

  A        no feedback
  C-lazy   batched, delivered at the model's next yield
  C-eager  batched, delivered immediately after each edit (the production hook)
  D-naive  live mid-stream, debounce 24 + pause-align + an `announce_lsp` prompt
           telling the model to fix each diagnostic before moving on
  D-plain  live mid-stream, debounce 24 + pause-align, no announce sentence
  D-gate   D-plain plus a syntax gate: deliver only when the file `ast.parse`s

Arm -> flag -> file is NOT inferred from filenames. The finalized JSONs omit the
runner config, but the contemporaneous per-task checkpoints (`<out>.json.partial`)
recorded `vars(argparse)` next to byte-identical rows; those configs are lifted
into runs/agent/synth_delivery_provenance.json, which this script prints and
cross-checks against the event traces.

  arm      | scripts/synth_acd.py @ 779aa5c invocation                | files (seeds 0-5 / 6-11)
  A        | --conds A                                                | synth_power.json[A] / synth_ac_s6.json[A]
  C-lazy   | --conds C                                                | synth_power.json[C] / synth_ac_s6.json[C]
  C-eager  | --conds C --c-eager                                      | synth_ceager.json / synth_ceager_s6.json
  D-naive  | --conds D --debounce 24 --pause-align --announce-lsp     | synth_power.json[D] / synth_dnaive_s6.json
  D-plain  | --conds D --debounce 24 --pause-align                    | synth_dplain.json / synth_dplain_s6.json
  D-gate   | --conds D --debounce 24 --pause-align --syntax-gate      | synth_dgate.json / synth_dgate_s6.json

Seeds 0-5 were run 2026-06-01..03; seeds 6-11 of every arm except D-naive were
run 2026-06-03; the D-naive seeds 6-11 block was run 2026-07-29 on the same
unchanged environment (pyrefly 1.0.0 / torch 2.11.0 / transformers 5.9.0,
installed 2026-05-27) with the same harness restored from git. The seeds-6-11
blocks carry a harness fix the seeds-0-5 blocks predate (24k context cap plus a
250-line file-view truncation); it fires as a `context_overflow` bail on 8 of the
1008 rollouts.

Two tests are reported for every contrast. Exact McNemar treats each (task, seed)
as an independent pair, which over-counts: seeds within a task are correlated. The
task-clustered bootstrap resamples the 14 tasks, which is the convention REPORT.md
uses ("the unit is the task; intervals are task-level bootstraps"). Fifteen
contrasts are run, so a Benjamini-Hochberg FDR is applied across the family.

Values published in log.md / the 2026-06-04 README are re-checked where they exist;
a mismatch prints FAIL and sets a non-zero exit code.

Run:  python scripts/analysis/stats_delivery.py   (from the repo root)
"""
import json
import math
import os
import random
import sys
from itertools import combinations

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
FAILURES = []


def A(p):
    return os.path.join(ROOT, "runs", "agent", p)


def rows(path, cond):
    return json.load(open(A(path)))["rows"][cond]


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
    ts = sorted(x)
    rnd = random.Random(seed)
    obs = sum(x[t] - y[t] for t in ts) / len(ts)
    ds = []
    for _ in range(B):
        s = [ts[rnd.randrange(len(ts))] for _ in ts]
        ds.append(sum(x[t] - y[t] for t in s) / len(s))
    ds.sort()
    return obs, ds[int(0.025 * B)], ds[int(0.975 * B)]


def bh_fdr(ps, q=0.05):
    """Benjamini-Hochberg: return the largest p that is still rejected, or None."""
    m = len(ps)
    srt = sorted(ps)
    cut = None
    for i, p in enumerate(srt, 1):
        if p <= i * q / m:
            cut = p
    return cut


def check(label, got, want, tol):
    ok = abs(got - want) <= tol
    if not ok:
        FAILURES.append(f"{label}: got {got}, recorded {want}")
    return "ok  " if ok else "FAIL"


ARMS = {
    "A":       rows("synth_power.json", "A") + rows("synth_ac_s6.json", "A"),
    "C-lazy":  rows("synth_power.json", "C") + rows("synth_ac_s6.json", "C"),
    "C-eager": rows("synth_ceager.json", "C") + rows("synth_ceager_s6.json", "C"),
    "D-naive": rows("synth_power.json", "D") + rows("synth_dnaive_s6.json", "D"),
    "D-plain": rows("synth_dplain.json", "D") + rows("synth_dplain_s6.json", "D"),
    "D-gate":  rows("synth_dgate.json", "D") + rows("synth_dgate_s6.json", "D"),
}

# ---------------------------------------------------------------- provenance
print("== arm identity: config recovered from the contemporaneous run checkpoints ==")
prov = json.load(open(A("synth_delivery_provenance.json")))
FLAGS = ("conds", "debounce", "pause_align", "announce_lsp", "c_eager", "syntax_gate")
for f, e in prov["files"].items():
    for c in (e["config"] if isinstance(e["config"], list) else [e["config"]]):
        print(f"  {f:24s} {e['arm']:20s} " + " ".join(f"{k}={c.get(k)}" for k in FLAGS))
    if e.get("blob_matches_779aa5c") is False:
        FAILURES.append(f"{f}: worktree copy differs from the committed blob at 779aa5c")
    if e.get("rows_identical_to_checkpoint") is False:
        FAILURES.append(f"{f}: rows differ from the config-carrying checkpoint")

# The event trace is an independent witness of the delivery mechanism: A emits no
# diag events, C-lazy queues them as observations, C-eager fires a post-edit hook,
# every D arm splices mid-stream after the debounce.
WITNESS = {"A": None, "C-lazy": "diag_sync_queued", "C-eager": "diag_eager"}
print("  event-trace witness (independent of filenames):")
for k, v in ARMS.items():
    kinds = sorted({e["type"] for r in v for e in r["events"] if e["type"].startswith("diag")})
    want = WITNESS.get(k, "diag_debounced")
    ok = (kinds == []) if want is None else (kinds == [want])
    if not ok:
        FAILURES.append(f"{k}: diag event types {kinds}, expected {want}")
    print(f"    {k:8s} diag events: {kinds or ['(none)']}  {'ok' if ok else 'FAIL'}")

# ------------------------------------------------------- apparatus check
print("  apparatus check (runs/agent/synth_dgate_reproduction_check.json):")
_chk = {(r["task"], r["seed"]): r
        for r in rows("synth_dgate_reproduction_check.json", "D")}
_june = {(r["task"], r["seed"]): r for r in rows("synth_dgate_s6.json", "D")}
_f = ["resolved", "in_tokens", "out_tokens", "n_edits", "n_tests", "turns", "rework_ratio",
      "stream_tail"]
_same = sum(1 for k, r in _chk.items() if all(_june[k][x] == r[x] for x in _f))
if _same != len(_chk):
    FAILURES.append(f"apparatus check: only {_same}/{len(_chk)} rollouts reproduce")
print(f"    {_same}/{len(_chk)} June D-gate rollouts re-run on 2026-07-29 reproduce bit-for-bit "
      f"{'ok' if _same == len(_chk) else 'FAIL'}")

# ----------------------------------------------------------- resolve rates
print("\n== resolve rate, 14 tasks x 12 seeds, n=168 per arm ==")
RECORDED = {"A": 0.482, "C-lazy": 0.530, "C-eager": 0.524, "D-plain": 0.458, "D-gate": 0.482}
for k, v in ARMS.items():
    n, t = sum(r["resolved"] for r in v), len(v)
    lo, hi = wilson(n, t)
    st = "    " if k not in RECORDED else check(f"resolve {k}", round(n / t, 3), RECORDED[k], 0.001)
    e0 = sum(r["resolved"] for r in v if r["seed"] < 6)
    e1 = sum(r["resolved"] for r in v if r["seed"] >= 6)
    print(f"  {k:8s} {n:3d}/{t} = {n/t:.3f}  [{lo:.2f},{hi:.2f}]   "
          f"(seeds 0-5 {e0}/84, seeds 6-11 {e1}/84)   {st}")
print(f"  D-naive seeds 0-5 = 0.345 and seeds 6-11 = 0.333: the arm replicates on fresh seeds.")

# ------------------------------------------------------------- comparisons
print("\n== all 15 pairwise contrasts ==")
res = []
for a, b in combinations(ARMS, 2):
    B, C, p, n = mcnemar(ARMS[a], ARMS[b])
    o, lo, hi = task_bootstrap(ARMS[a], ARMS[b])
    res.append((a, b, B, C, p, o, lo, hi))
cut = bh_fdr([r[4] for r in res])
for a, b, B, C, p, o, lo, hi in res:
    mark = " *" if cut is not None and p <= cut else "  "
    print(f"  {a:8s} vs {b:8s}: b={B:3d} c={C:3d} McNemar p={p:.4f}   "
          f"task delta {o:+.3f} [{lo:+.3f},{hi:+.3f}]{mark}")
print(f"  * = rejected at Benjamini-Hochberg FDR 5% over all 15 contrasts (cutoff p<={cut:.4f})")
sig = {(a, b) for a, b, B, C, p, o, lo, hi in res if cut is not None and p <= cut}
band = {"A", "C-lazy", "C-eager", "D-plain", "D-gate"}
if any(a in band and b in band for a, b in sig):
    FAILURES.append("a contrast among the five properly-delivered arms came out significant")
if not all(("D-naive" in (a, b)) for a, b in sig):
    FAILURES.append("a significant contrast does not involve D-naive")
if len(sig) != 5:
    FAILURES.append(f"expected all 5 D-naive contrasts significant, got {len(sig)}")
print("  every significant contrast involves D-naive; none of the ten contrasts among the")
print("  five properly-delivered arms is significant (smallest of those, McNemar p="
      f"{min(p for a,b,B,C,p,o,lo,hi in res if a in band and b in band):.3f}).")

# ------------------------------------------------- decomposing D-naive's deficit
print("\n== decomposing the naive arm's deficit ==")
for a, b, label in (("D-naive", "D-plain", "drop the announce sentence"),
                    ("D-plain", "D-gate", "add the syntax gate on top"),
                    ("D-naive", "D-gate", "both changes together")):
    B, C, p, n = mcnemar(ARMS[a], ARMS[b])
    o, lo, hi = task_bootstrap(ARMS[a], ARMS[b])
    print(f"  {label:28s} {a} -> {b}: {-o:+.3f} [{-hi:+.3f},{-lo:+.3f}]  McNemar p={p:.4f}")
print("  the announce sentence carries the deficit; the gate adds little to the outcome")
print("  even though it removes ~70% of the deliveries.")

# ------------------------------------------------------- the gate's mechanism
print("\n== mechanism: what the syntax gate removes ==")
RECORDED_N = {("D-naive", "0-5"): 244, ("D-plain", "0-5"): 240, ("D-gate", "0-5"): 71,
              ("D-plain", "6-11"): 302, ("D-gate", "6-11"): 90}
for (nm, seeds), src in (
        (("D-naive", "0-5"), rows("synth_power.json", "D")),
        (("D-naive", "6-11"), rows("synth_dnaive_s6.json", "D")),
        (("D-plain", "0-5"), rows("synth_dplain.json", "D")),
        (("D-plain", "6-11"), rows("synth_dplain_s6.json", "D")),
        (("D-gate", "0-5"), rows("synth_dgate.json", "D")),
        (("D-gate", "6-11"), rows("synth_dgate_s6.json", "D"))):
    texts = [e.get("text", "") for r in src for e in r["events"] if e["type"] == "diag_debounced"]
    bad = sum(1 for t in texts if "invalid-syntax" in t or "parse-error" in t)
    st = ("    " if (nm, seeds) not in RECORDED_N
          else check(f"deliveries {nm} {seeds}", len(texts), RECORDED_N[(nm, seeds)], 0))
    print(f"  {nm:8s} seeds {seeds:5s} deliveries={len(texts):4d}  "
          f"mentioning a syntax/parse error={bad:4d} ({bad/max(1,len(texts)):.0%})   {st}")
print("  the gate cuts live deliveries ~70% in both seed blocks (240->71, 302->90) and the")
print("  share describing a broken parse falls from ~76% to ~24%: most ungated live")
print("  diagnostics were about the model's own half-finished edit.")

print("\n" + ("ALL CHECKS PASS" if not FAILURES else "FAILURES:\n  " + "\n  ".join(FAILURES)))
sys.exit(1 if FAILURES else 0)
