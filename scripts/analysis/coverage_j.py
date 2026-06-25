#!/usr/bin/env python3
"""Coverage-judging score J for the `cover` suite (does the agent read based on what
`<defn>` returned, or on task shape?).

For a result JSON produced by `synth_mf.py --suite cover`, each task is one of three
byte-identical-surface variants of a topic:
  cover_<topic>_suf    coverage SUFFICIENT (defn returns the value)        -> should NOT read
  cover_<topic>_f1ins  INSUFFICIENT, mechanism F1 (forwarding alias)       -> must read
  cover_<topic>_f2ins  INSUFFICIENT, mechanism F2 (value externalised)     -> must read

The coverage-judging score:
    J = P(read | insufficient) - P(read | sufficient)
A model that JUDGES coverage has J >> 0 (reads exactly when the defn it saw was inadequate).
A SHAPE-KEYED model has J ~ 0 on a held-out mechanism, because its read-decision does not
depend on the content `<defn>` returned. We report J separately by mechanism:
  J_F1 = read(f1ins) - read(suf)   (F1 = training mechanism, if F1 was trained on)
  J_F2 = read(f2ins) - read(suf)   (F2 = HELD-OUT mechanism — the generalization test)

Usage:  python scripts/analysis/coverage_j.py runs/agent/cover_base.json [runs/agent/cover_trained.json ...]
        (label each file PRE/POST etc. by passing name=path, e.g. base=runs/agent/cover_base.json)
"""
import json, sys, statistics as st

def load(path):
    d = json.load(open(path)); r = d["rows"]
    return r["A"] if isinstance(r, dict) else r

def variant(task):
    if task.endswith("_suf"):   return "suf"
    if task.endswith("_f1ins"): return "f1ins"
    if task.endswith("_f2ins"): return "f2ins"
    return "?"

def rate(rows, key):
    return sum(1 for r in rows if (r.get(key) or 0) > 0) / len(rows) if rows else float("nan")

def esc_rate(rows):
    """Escalation = the model PROBED (>=1 defn) and then got MORE — a 2nd defn (cheap defn-chain,
    since the insufficient variants name their target) OR a read. This is the true coverage-judging
    signal: on a sufficient variant one defn suffices; on an insufficient one a judge escalates."""
    if not rows: return float("nan")
    return sum(1 for r in rows
               if (r.get("n_lsp") or 0) >= 1
               and ((r.get("n_lsp") or 0) >= 2 or (r.get("n_reads") or 0) >= 1)) / len(rows)

def report(label, rows):
    by = {"suf": [], "f1ins": [], "f2ins": []}
    for r in rows:
        v = variant(r["task"])
        if v in by: by[v].append(r)
    print(f"\n=== {label}  (suf n={len(by['suf'])}, f1ins n={len(by['f1ins'])}, f2ins n={len(by['f2ins'])}) ===")
    print(f"  {'variant':20}{'read':>6}{'defn>=2':>8}{'escalate':>9}{'solved':>8}{'in_tok':>8}")
    for v in ("suf", "f1ins", "f2ins"):
        g = by[v]
        if not g: continue
        dchain = sum(1 for r in g if (r.get("n_lsp") or 0) >= 2)/len(g)
        cov = "sufficient" if v == "suf" else "insufficient"
        print(f"  {v+' ('+cov+')':20}{rate(g,'n_reads'):6.2f}{dchain:8.2f}{esc_rate(g):9.2f}"
              f"{rate(g,'resolved'):8.2f}{st.mean(r.get('in_tokens',0) for r in g):8.0f}")
    es = esc_rate(by["suf"])
    j1 = esc_rate(by["f1ins"]) - es if by["f1ins"] else float("nan")
    j2 = esc_rate(by["f2ins"]) - es if by["f2ins"] else float("nan")
    # read-only variant of J, for comparison
    rs = rate(by["suf"], "n_reads")
    j1r = rate(by["f1ins"], "n_reads") - rs if by["f1ins"] else float("nan")
    j2r = rate(by["f2ins"], "n_reads") - rs if by["f2ins"] else float("nan")
    print(f"  J_escalate  F1(train)={j1:+.2f}   F2(HELD-OUT)={j2:+.2f}   <-- primary (read or defn-chain)")
    print(f"  J_read-only F1={j1r:+.2f}   F2={j2r:+.2f}   (reads specifically; lower if it defn-chains)")
    print(f"  floor check: suf solved={rate(by['suf'],'resolved'):.2f} (must be high, else suite still floors)")
    return j1, j2

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__); sys.exit(0)
    print("Coverage-judging score J = P(read|insufficient) - P(read|sufficient).")
    print("J>>0 = judges coverage from defn content; J~0 on held-out F2 = shape-keying.")
    rows_by_label = {}
    for a in args:
        label, path = (a.split("=", 1) if "=" in a else (a, a))
        rows_by_label[label] = load(path)
    js = {lab: report(lab, rows) for lab, rows in rows_by_label.items()}
    if len(js) >= 2:
        labs = list(js)
        print(f"\n=== read-out: does training instill coverage-judging that GENERALIZES? ===")
        print(f"  J_F2 across arms: " + ", ".join(f"{l}={js[l][1]:+.2f}" for l in labs))
        print(f"  CLOSED if a trained arm has J_F2 >> 0 while base ~ 0 (judges unseen mechanism);")
        print(f"  SHAPE-KEYING if J_F2 ~ 0 for the trained arm (read-decision ignores defn content).")
