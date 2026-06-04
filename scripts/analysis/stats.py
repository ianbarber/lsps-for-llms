#!/usr/bin/env python3
"""Reproduce the paper's headline statistics from the committed result files.

Prints:
  1. PAPER §5.1 — the n=168 fix-rate table (14 tasks x 12 seeds) with Wilson 95% CIs,
     and all ten pairwise exact McNemar tests among the five proper-delivery arms.
  2. PAPER §5.2 — the D-naive (n=84) harm-mode comparisons, incl. the hygiene effect.
  3. PAPER §6.1 — rich-signal vs plain, paired.
  4. PAPER §6.2 — SFT adapter vs base on held-out vs train tasks, with the A control.

Condition -> file map (paper name = log/repo name where they differ):
  A        : synth_power.json[A]   + synth_ac_s6.json[A]
  C-lazy   : synth_power.json[C]   + synth_ac_s6.json[C]
  C-eager  : synth_ceager.json[C]  + synth_ceager_s6.json[C]     (--c-eager)
  D-naive  : synth_power.json[D]   (n=84; called "D-tuned" in log.md;
                                     --debounce 24 --pause-align --announce-lsp)
  D-plain  : synth_dplain.json[D]  + synth_dplain_s6.json[D]     (no announce)
  D-gate   : synth_dgate.json[D]   + synth_dgate_s6.json[D]      (+ --syntax-gate)

Run:  python scripts/analysis/stats.py   (from the repo root)
"""
import json, math, os, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
A = lambda p: os.path.join(ROOT, "runs", "agent", p)

def rows(path, cond):
    return json.load(open(path))["rows"][cond]

def wilson(n, t, z=1.96):
    p = n / t; d = 1 + z * z / t
    c = (p + z * z / (2 * t)) / d
    h = z * math.sqrt(p * (1 - p) / t + z * z / (4 * t * t)) / d
    return c - h, c + h

def mcnemar(X, Y):
    """Exact two-sided McNemar on paired (task, seed) units."""
    ix = {(r["task"], r["seed"]): r["resolved"] for r in X}
    iy = {(r["task"], r["seed"]): r["resolved"] for r in Y}
    keys = [k for k in ix if k in iy]
    b = sum(1 for k in keys if ix[k] and not iy[k])
    c = sum(1 for k in keys if iy[k] and not ix[k])
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
    return b, c, p, len(keys)

BAND = {
    "A":       rows(A("synth_power.json"), "A") + rows(A("synth_ac_s6.json"), "A"),
    "C-lazy":  rows(A("synth_power.json"), "C") + rows(A("synth_ac_s6.json"), "C"),
    "C-eager": rows(A("synth_ceager.json"), "C") + rows(A("synth_ceager_s6.json"), "C"),
    "D-plain": rows(A("synth_dplain.json"), "D") + rows(A("synth_dplain_s6.json"), "D"),
    "D-gate":  rows(A("synth_dgate.json"), "D") + rows(A("synth_dgate_s6.json"), "D"),
}
DNAIVE = rows(A("synth_power.json"), "D")   # n=84; "D-tuned" in log.md

print("== PAPER 5.1: proper-delivery arms, n=168 ==")
for k, v in BAND.items():
    n, t = sum(r["resolved"] for r in v), len(v)
    lo, hi = wilson(n, t)
    print(f"  {k:8} {n:3}/{t} = {n/t:.3f}  [{lo:.2f},{hi:.2f}]")
print("  all pairwise exact McNemar:")
ks = list(BAND)
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        b, c, p, n = mcnemar(BAND[ks[i]], BAND[ks[j]])
        print(f"    {ks[i]:8} vs {ks[j]:8}: b={b:2} c={c:2} p={p:.3f}")

print("\n== PAPER 5.2: D-naive harm mode (n=84, seeds 0-5) ==")
n = sum(r["resolved"] for r in DNAIVE)
print(f"  D-naive  {n}/84 = {n/84:.3f}")
sub = lambda v: [r for r in v if r["seed"] < 6]
for name, other in (("C-eager", sub(BAND["C-eager"])), ("C-lazy", sub(BAND["C-lazy"])),
                    ("A", sub(BAND["A"])), ("D-gate (hygiene)", sub(BAND["D-gate"]))):
    b, c, p, _ = mcnemar(DNAIVE, other)
    print(f"  D-naive vs {name:17}: b={b:2} c={c:2} p={p:.4f}")

print("\n== PAPER 6.1: rich signal (n=84, paired vs plain counterpart) ==")
for rich, plain, cond, label in ((A("synth_dgate_rich.json"), A("synth_dgate.json"), "D", "D-gate"),
                                 (A("synth_ceager_rich.json"), A("synth_ceager.json"), "C", "C-eager")):
    R, P = rows(rich, cond), rows(plain, cond)
    b, c, p, _ = mcnemar(R, P)
    print(f"  {label}+rich {sum(r['resolved'] for r in R)}/84 vs plain "
          f"{sum(r['resolved'] for r in P)}/84:  b={b} c={c} p={p:.3f}")

print("\n== PAPER 6.2: SFT (held-out = odd-indexed tasks in scripts/synth_tasks.py) ==")
sys.path.insert(0, ROOT)
from scripts.synth_tasks import TASKS
names = [t["name"] for t in TASKS]
hold = {nm for i, nm in enumerate(names) if i % 2 == 1}
arms = {"D-gate base": rows(A("synth_dgate.json"), "D"),
        "D-gate +SFT": rows(A("synth_dgate_sft.json"), "D"),
        "A base":      sub(BAND["A"]),
        "A +SFT":      rows(A("synth_a_sft.json"), "A")}
for k, v in arms.items():
    h = [r for r in v if r["task"] in hold]; tr = [r for r in v if r["task"] not in hold]
    print(f"  {k:12} held-out {sum(r['resolved'] for r in h):2}/{len(h)}   "
          f"train {sum(r['resolved'] for r in tr):2}/{len(tr)}")
for x, y in (("D-gate +SFT", "D-gate base"), ("A +SFT", "A base")):
    hx = [r for r in arms[x] if r["task"] in hold]; hy = [r for r in arms[y] if r["task"] in hold]
    tx = [r for r in arms[x] if r["task"] not in hold]; ty = [r for r in arms[y] if r["task"] not in hold]
    _, _, ph, _ = mcnemar(hx, hy); _, _, pt, _ = mcnemar(tx, ty)
    print(f"  {x} vs {y}: held-out p={ph:.3f} | train p={pt:.3f}")
