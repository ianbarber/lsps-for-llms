#!/usr/bin/env python3
"""Synthetic 'type-system-signal' tasks for the A/C/D delivery-form eval — v2, rebuilt
after subagent cross-review. Centered on MULTI-SITE type-error cascades.

THE MECHANISM these are built to exercise:
  A real type-level change (renamed field, signature drift, return-type ripple) breaks
  SEVERAL call sites. `pytest` reveals them ONE AT A TIME — it crashes at the first bad
  site; only after you fix it does the next crash. So condition A (test feedback only)
  must grind site-by-site: fix, re-test, next crash, fix, ... (round-trips + rework).
  `pyrefly` reports ALL broken sites at once → C/D can fix them together; D sees them
  live mid-stream. That serial-vs-all-at-once gap is the live-vs-batched signal.

DESIGN BAR (per cross-review):
- Natural bug (refactor drift / rename / ripple), not constructed-to-trip-the-checker.
- Intended behaviour lives in the TEST, not in giveaway docstrings.
- Partial/natural typing (don't annotate everything just to surface a diagnostic).
- Distractor: a plausible-but-wrong fix that passes a naive reader but fails the test.
- Solvable by reading too (A>0) — the checker aids efficiency, isn't the only way in.
- Difficulty calibrated EMPIRICALLY (base pass@k, keep suite mean ~30-60%), not a priori.

Each task: name, bug_class, sites (#broken call sites), entry, code, test, gold_note
(reference only — NEVER shown to the model), fix_tokens (for the loud-traceback flag).
"""

TASKS = [
    dict(
        name="grid_field_rename", bug_class="renamed field, 3 sites", sites=3,
        entry="bounding_area",
        code='''\
from dataclasses import dataclass

@dataclass
class Cell:
    row: int
    x: int            # renamed from `col` during a refactor

def corners(cells: list[Cell]):
    rows = [c.row for c in cells]
    cols = [c.col for c in cells]
    return (min(rows), min(cols), max(rows), max(cols))

def width(cells: list[Cell]) -> int:
    return max(c.col for c in cells) - min(c.col for c in cells) + 1

def bounding_area(cells: list[Cell]) -> int:
    top, left, bot, right = corners(cells)
    return (bot - top + 1) * (right - left + 1)
''',
        test='''\
cs = [Cell(0, 0), Cell(2, 3), Cell(1, 1)]
assert bounding_area(cs) == 3 * 4
assert width(cs) == 4
''',
        gold_note="field was renamed col->x; update the three `.col` usages to `.x`",
        fix_tokens=["col", "x"],
    ),
    dict(
        name="fmt_signature_drift", bug_class="signature drift (added param), 3 sites", sites=3,
        entry="render",
        code='''\
def fmt_row(label: str, value: float, unit: str) -> str:
    return f"{label}: {value:.2f}{unit}"

def render(data):
    rows = []
    for k, v in data.items():
        rows.append(fmt_row(k, v))
    rows.append(fmt_row("subtotal", sum(data.values())))
    rows.append(fmt_row("items", float(len(data))))
    return rows
''',
        test='''\
assert render({"a": 1.0, "b": 2.5}) == ["a: 1.00", "b: 2.50", "subtotal: 3.50", "items: 2.00"]
''',
        gold_note="fmt_row gained a `unit` param; pass \"\" at all three call sites",
        fix_tokens=["unit"],
    ),
    dict(
        name="records_arity_drift", bug_class="tuple arity change (schema), 2 sites", sites=2,
        entry="summary",
        code='''\
def totals(records: list[tuple[str, int, float]]):
    out = {}
    for name, qty in records:
        out[name] = out.get(name, 0) + qty
    return out

def grand_total(records: list[tuple[str, int, float]]) -> float:
    s = 0.0
    for name, qty in records:
        s += qty
    return s

def summary(records: list[tuple[str, int, float]]):
    return totals(records), grand_total(records)
''',
        test='''\
recs = [("apple", 3, 0.5), ("apple", 2, 0.5), ("pear", 1, 1.0)]
t, g = summary(recs)
assert t == {"apple": 5, "pear": 1}
assert g == 6.0
''',
        gold_note="records are 3-tuples now; unpack (name, qty, price) at both loops",
        fix_tokens=["price", "_"],
    ),
    dict(
        name="lookup_optional_cascade", bug_class="unguarded Optional, 2 sites", sites=2,
        entry="invoice_total",
        code='''\
def find(catalog, sku):
    for item in catalog:
        if item["sku"] == sku:
            return item
    return None

def line_price(catalog, sku, qty):
    item = find(catalog, sku)
    return item["price"] * qty

def invoice_total(catalog, order):
    # order: list of (sku, qty). Unknown skus are skipped.
    total = 0.0
    for sku, qty in order:
        item = find(catalog, sku)
        total += item["price"] * qty
    return total
''',
        test='''\
cat = [{"sku": "A", "price": 2.0}, {"sku": "B", "price": 5.0}]
assert invoice_total(cat, [("A", 3), ("B", 1)]) == 11.0
assert invoice_total(cat, [("A", 2), ("ZZ", 9)]) == 4.0
assert line_price(cat, "B", 2) == 10.0
''',
        gold_note="find() may return None; guard both line_price and invoice_total (skip unknown)",
        fix_tokens=["None", "is none"],
    ),
    dict(
        name="config_truthiness_distractor", bug_class="Optional + truthiness distractor", sites=1,
        entry="effective_limits",
        code='''\
def effective_limits(cfg, keys, default):
    out = {}
    for k in keys:
        v = cfg.get(k)
        out[k] = v if v else default
    return out
''',
        test='''\
# a configured value of 0 is meaningful and must be preserved; only ABSENT keys default
assert effective_limits({"a": 5, "b": 0}, ["a", "b", "c"], 99) == {"a": 5, "b": 0, "c": 99}
''',
        gold_note="`v if v else default` drops a legit 0; use `v if v is not None else default` "
                  "(or cfg.get(k, default)). pyrefly flags cfg.get -> Optional.",
        fix_tokens=["none", "get"],
    ),
    dict(
        name="parse_branch_ripple", bug_class="wrong return type in a branch, ripples to 2 callers", sites=1,
        entry="stats",
        code='''\
def parse_amount(s):
    s = s.strip()
    if not s:
        return ""
    return float(s)

def total(rows) -> float:
    return sum(parse_amount(r) for r in rows)

def stats(rows):
    vals = [parse_amount(r) for r in rows]
    return total(rows), max(vals)
''',
        test='''\
assert stats(["1", "", "3"]) == (4.0, 3.0)
assert stats(["2.5", "2.5"]) == (5.0, 2.5)
''',
        gold_note="empty parses to '' (str) — should be 0.0; breaks sum() and max() over mixed types",
        fix_tokens=["0.0", "float"],
    ),
    dict(
        name="return_container_ripple", bug_class="return type changed dict->list, 2 callers", sites=2,
        entry="report",
        code='''\
def load_scores(pairs):
    # returns a list of (name, score); was previously a dict
    return [(name, score) for name, score in pairs]

def best(pairs) -> str:
    scores = load_scores(pairs)
    return max(scores, key=lambda kv: scores[kv])

def report(pairs):
    scores = load_scores(pairs)
    names = [n for n in scores]
    return names, best(pairs)
''',
        test='''\
ps = [("ann", 3), ("bo", 5), ("cy", 1)]
names, top = report(ps)
assert names == ["ann", "bo", "cy"]
assert top == "bo"
''',
        gold_note="load_scores returns list[tuple]; `scores[kv]` and `for n in scores` (then "
                  "expecting keys) assume a dict — index/iterate the list of pairs instead",
        fix_tokens=["dict", "kv", "[1]"],
    ),
    dict(
        name="method_rename_cascade", bug_class="renamed method, 2 sites", sites=2,
        entry="process",
        code='''\
class Account:
    def __init__(self, balance: int):
        self.balance = balance

    def deposit(self, amt: int) -> None:
        self.balance += amt

    def withdraw(self, amt: int) -> None:
        self.balance -= amt

def process(acct: Account, txns: list[int]) -> int:
    for t in txns:
        if t >= 0:
            acct.credit(t)
        else:
            acct.debit(-t)
    return acct.balance
''',
        test='''\
a = Account(100)
assert process(a, [50, -30, 10]) == 130
''',
        gold_note="methods were renamed credit->deposit, debit->withdraw; update both call sites",
        fix_tokens=["deposit", "withdraw", "credit", "debit"],
    ),
    dict(
        name="dict_key_type_drift", bug_class="dict key type changed int->str lookups", sites=1,
        entry="lookup_all",
        code='''\
def build_index(items: list[tuple[int, str]]) -> dict[int, str]:
    return {k: v for k, v in items}

def lookup_all(items: list[tuple[int, str]], keys: list[int]) -> list[str]:
    idx = build_index(items)
    return [idx[str(k)] for k in keys]
''',
        test='''\
items = [(1, "a"), (2, "b"), (3, "c")]
assert lookup_all(items, [2, 1]) == ["b", "a"]
''',
        gold_note="idx is keyed by int; `idx[str(k)]` looks up a str -> KeyError; use idx[k]",
        fix_tokens=["str", "int"],
    ),
    dict(
        name="ctor_param_added", bug_class="constructor gained a field, 2 sites", sites=2,
        entry="make_points",
        code='''\
from dataclasses import dataclass

@dataclass
class Vec:
    x: int
    y: int
    z: int

def make_points(pairs: list[tuple[int, int]]) -> list[Vec]:
    out = []
    for a, b in pairs:
        out.append(Vec(a, b))
    out.append(Vec(0, 0))
    return out
''',
        test='''\
ps = make_points([(1, 2), (3, 4)])
assert [(v.x, v.y, v.z) for v in ps] == [(1, 2, 0), (3, 4, 0), (0, 0, 0)]
''',
        gold_note="Vec gained a `z` field; both Vec(...) calls miss it — pass z=0",
        fix_tokens=["z"],
    ),
    dict(
        name="renamed_return_key", bug_class="returned dict key renamed, 2 readers", sites=2,
        entry="summarize",
        code='''\
from typing import TypedDict

class Stats(TypedDict):
    total: int
    count: int

def analyze(xs: list[int]) -> Stats:
    return {"total": sum(xs), "count": len(xs)}

def mean_of(xs: list[int]) -> float:
    r = analyze(xs)
    return r["sum"] / r["count"]

def summarize(xs: list[int]) -> str:
    r = analyze(xs)
    return f"{r['sum']} over {r['count']}"
''',
        test='''\
assert mean_of([2, 4, 6]) == 4.0
assert summarize([2, 4, 6]) == "12 over 3"
''',
        gold_note="analyze returns 'total', not 'sum'; both readers use r['sum'] -> KeyError",
        fix_tokens=["sum", "total"],
    ),
    dict(
        name="optional_two_helpers", bug_class="two Optional-returning helpers unguarded", sites=2,
        entry="route_cost",
        code='''\
def node_at(grid, idx):
    if 0 <= idx < len(grid):
        return grid[idx]
    return None

def cost_between(grid, i, j):
    a = node_at(grid, i)
    b = node_at(grid, j)
    return abs(a - b)

def route_cost(grid, hops):
    total = 0
    for i, j in hops:
        total += cost_between(grid, i, j)
    return total
''',
        test='''\
g = [5, 8, 2, 10]
assert route_cost(g, [(0, 1), (2, 3)]) == 3 + 8
assert route_cost(g, [(0, 99)]) == 0
''',
        gold_note="node_at may return None; cost_between must skip/handle out-of-range (return 0)",
        fix_tokens=["none", "is none"],
    ),
    dict(
        name="tuple_return_widened", bug_class="return widened 2-tuple->3-tuple, 1 reader", sites=1,
        entry="describe",
        code='''\
def minmax(xs: list[int]):
    return min(xs), max(xs), sum(xs) / len(xs)

def describe(xs: list[int]) -> str:
    lo, hi = minmax(xs)
    return f"range {lo}..{hi}"
''',
        test='''\
assert describe([3, 1, 4, 1, 5]) == "range 1..5"
''',
        gold_note="minmax now returns 3 values (min, max, avg); the reader unpacks 2 -> ValueError",
        fix_tokens=["avg", "_", "mean"],
    ),
    dict(
        name="mutable_default_none", bug_class="None default used as collection, 2 paths", sites=2,
        entry="collect",
        code='''\
def add_tags(tags=None, *more):
    for m in more:
        tags.append(m)
    return tags

def collect(groups):
    # groups: list of tag-lists; flatten, but start fresh each call
    acc = add_tags()
    for g in groups:
        acc = add_tags(acc, *g)
    return acc
''',
        test='''\
assert collect([["a", "b"], ["c"]]) == ["a", "b", "c"]
assert collect([]) == []
''',
        gold_note="tags=None then tags.append crashes; default to [] (if tags is None: tags=[])",
        fix_tokens=["none", "[]"],
    ),
]

if __name__ == "__main__":
    import os, json, subprocess, tempfile, io, contextlib, multiprocessing as mp
    PYREFLY = "/home/ianbarber/Projects/Streams/.venv-streams/bin/pyrefly"

    def verify_pyrefly(code):
        ws = tempfile.mkdtemp(prefix="synthchk_")
        fp = os.path.join(ws, "m.py")
        open(fp, "w").write(code)
        open(os.path.join(ws, "pyrefly.toml"), "w").write('[tool.pyrefly]\nproject-includes = ["*.py"]\n')
        try:
            r = subprocess.run([PYREFLY, "check", "--output-format", "json", fp],
                               cwd=ws, capture_output=True, text=True, timeout=30)
            errs = json.loads(r.stdout or "{}").get("errors", [])
        except Exception:
            errs = []
        return [(e["line"], e.get("severity"), e.get("name"), (e.get("concise_description") or "")) for e in errs]

    def buggy_exc(code, test):
        q = mp.Queue()
        def w(q):
            g = {}
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    exec("from typing import *\n" + code, g); exec(test, g)
                q.put(("PASS", ""))
            except Exception as e:
                q.put((type(e).__name__, str(e)))
        p = mp.Process(target=w, args=(q,)); p.start(); p.join(8)
        if p.is_alive(): p.terminate(); return ("timeout", "")
        try: return q.get_nowait()
        except Exception: return ("?", "")

    print(f"{'task':28} {'sites':5} {'pyflerr':7} {'behaviour':14} loud?")
    for t in TASKS:
        diags = verify_pyrefly(t["code"])
        errs = [d for d in diags if d[1] == "error"]
        exc, msg = buggy_exc(t["code"], t["test"])
        loud = any(tok.lower() in msg.lower() for tok in t.get("fix_tokens", []))
        print(f"{t['name']:28} {t['sites']:<5} {len(errs):<7} {exc:14} {'LOUD' if loud else 'ok'}  | {msg[:60]}")
        for d in diags:
            print(f"     L{d[0]} [{d[1]}] {d[2]}: {d[3][:80]}")
