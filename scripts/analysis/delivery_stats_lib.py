#!/usr/bin/env python3
"""Statistics shared by the delivery-timing analyzers (ledger C37).

`stats_delivery.py` reproduces the published June numbers; `stats_delivery_v2.py`
analyses a re-run on the ported harness and sets it against June. Those two must
compute resolve rates, McNemar and the task-clustered bootstrap the same way, or a
June-vs-v2 difference could come from the analysis rather than from the experiment.
Hence one implementation, imported by both, rather than a copy in each.

The unit conventions match REPORT.md: exact McNemar pairs on (task, seed), which
over-counts because seeds within a task are correlated, so the task-clustered
bootstrap resampling the 14 tasks is the interval that is quoted.
"""
import json
import math
import os
import random

# Arm -> the June artifacts holding its rows, as (file, cond-key-within-file) pairs.
# Seeds 0-5 first, 6-11 second. Not inferable from filenames: `synth_power.json` holds
# three different arms under three cond keys, and D-naive's second block lives in a file
# named for the arm while C-lazy's does not. The identity of each entry rests on the
# configs recovered into runs/agent/synth_delivery_provenance.json, not on this table;
# this table only records the lookup. Shared so the June reproducer and the v2 comparison
# cannot disagree about which rows constitute an arm.
JUNE_SOURCES = {
    "A":       [("synth_power.json", "A"), ("synth_ac_s6.json", "A")],
    "C-lazy":  [("synth_power.json", "C"), ("synth_ac_s6.json", "C")],
    "C-eager": [("synth_ceager.json", "C"), ("synth_ceager_s6.json", "C")],
    "D-naive": [("synth_power.json", "D"), ("synth_dnaive_s6.json", "D")],
    "D-plain": [("synth_dplain.json", "D"), ("synth_dplain_s6.json", "D")],
    "D-gate":  [("synth_dgate.json", "D"), ("synth_dgate_s6.json", "D")],
}

# The delivery event each arm must emit, as an independent witness of the mechanism:
# A emits nothing, C-lazy queues an observation, C-eager fires a post-edit hook, and
# every D arm splices mid-stream after the debounce.
WITNESS = {"A": None, "C-lazy": "diag_sync_queued", "C-eager": "diag_eager",
           "D-naive": "diag_debounced", "D-plain": "diag_debounced", "D-gate": "diag_debounced"}


def load_june(root):
    """Arm -> rows, from the committed June artifacts."""
    out = {}
    for arm, srcs in JUNE_SOURCES.items():
        rs = []
        for fn, cond in srcs:
            path = os.path.join(root, "runs", "agent", fn)
            rs += json.load(open(path))["rows"][cond]
        out[arm] = rs
    return out


def diag_kinds(rs):
    """The distinct diagnostic event types appearing in an arm's rows."""
    return sorted({e["type"] for r in rs for e in r["events"] if e["type"].startswith("diag")})


def wilson(n, t, z=1.96):
    """Wilson score interval for n successes of t."""
    p = n / t
    d = 1 + z * z / t
    c = (p + z * z / (2 * t)) / d
    h = z * math.sqrt(p * (1 - p) / t + z * z / (4 * t * t)) / d
    return c - h, c + h


def mcnemar(X, Y):
    """Exact two-sided McNemar over paired (task, seed) units.

    Returns (b, c, p, n_pairs). Only keys present in both arms are paired, so an
    arm that is short a few cells contributes only its overlap.
    """
    ix = {(r["task"], r["seed"]): r["resolved"] for r in X}
    iy = {(r["task"], r["seed"]): r["resolved"] for r in Y}
    keys = [k for k in ix if k in iy]
    b = sum(1 for k in keys if ix[k] and not iy[k])
    c = sum(1 for k in keys if iy[k] and not ix[k])
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
    return b, c, p, len(keys)


def by_task(rs):
    """Per-task resolve fraction."""
    d = {}
    for r in rs:
        d.setdefault(r["task"], []).append(bool(r["resolved"]))
    return {k: sum(v) / len(v) for k, v in d.items()}


def task_bootstrap(X, Y, B=20000, seed=0):
    """Mean per-task resolve difference, 95% CI over tasks resampled with replacement.

    Resamples only tasks the two arms share, so a partial arm cannot silently
    compare a task present in one against a task absent from the other.
    """
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
