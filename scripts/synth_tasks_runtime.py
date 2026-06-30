#!/usr/bin/env python3
"""RUNTIME-FEEDBACK suite — does EXECUTION feedback help a strong agent, the way the
language server's static INFORMATION does not?

THE QUESTION. Our headline result (REPORT.md §3) is that a language server's information
is redundant for a self-retrieving agent: the facts it serves (a symbol's type, its
definition, its references) are already in the source, so a capable reader derives them
and pass@1 does not move. The natural boundary question is what the agent CANNOT derive by
reading. Source code is a program; its behaviour on an input is that program executed. To
get behaviour from source the model must mentally EXECUTE it, and that simulation can be
wrong. Runtime feedback (a failing test) delivers exactly what the model's own simulation
got wrong, which a static read cannot.

So this suite is the mirror of gapd2. There the wrong fix PASSED the visible test (the
checker was the only possible detector); here the wrong fix FAILS the visible test, so
RUNNING is the detector. The agent is run under three arms (api_agent.py flags):

  R0  --no-test            no run_tests tool: the agent must reason about behaviour from
                           the source and the shown spec, then call done(). (The spec is
                           still shown as static text, so R0 knows WHAT is checked, it just
                           cannot RUN it: this isolates "can it simulate" from "does it
                           know the target".)
  R1  (default)            run_tests available; the agent elects to run the visible test.
  R2  --auto-feedback      the env volunteers the visible-test result after every edit
                           (execution handed over for free, no election needed).

Scoring is HELD-OUT in every arm (`resolved` = score() on held_out, `visible_pass` = the
test the agent could run). Held-out scoring does real work in all three arms: in R0 it
catches a mis-simulated fix; in R1/R2 it catches a fix that overfits the one shown assert
but is wrong on other inputs. Run all three with --no-hint (the visible test is informative
here, NOT a partial spec to be suspicious of, so the gapd2 "partial spec" note is omitted
and auto-stop-on-visible-pass is the natural behaviour for R1/R2).

EVERY wrong fix is WELL-TYPED (the verifier checks pyrefly is clean on it, W4). The bug is
pure runtime logic, so a type checker is NOT a detector and check_types would be useless on
this whole suite: the discriminator is execution, period. This is the deliberate contrast
with gapd2, where the checker COULD see the latent bug.

SIMULATION-DIFFICULTY STRATIFICATION (the `sim`/`group` field, "hard" | "easy"). The
secondary hypothesis is that the value of execution rises with how hard the code is to
mentally execute. HARD tasks are multi-step / stateful / boundary / off-by-one logic where
a careful read is genuinely unreliable (sliding windows, the forgotten final flush, tier
boundaries, run counting). EASY tasks are one-line specs where the fix is obvious from
reading (a missing abs(), a one-sided clamp, a missing empty guard); a strong model should
solve those in R0 too, so execution should add little there.

THE RUNTIME-SUITE INVARIANT (enforced by __main__):
  W1  stub:  held-out score() FAILS (NotImplementedError).
  W2  gold:  visible run_tests() PASSES and held-out score() PASSES.
  W3  wrong: visible run_tests() FAILS and held-out score() FAILS, and wrong != gold.
             (KEY, the opposite of gapd2 V3: here RUNNING the visible test exposes the bug.)
  W4  wrong is WELL-TYPED: target-scoped pyrefly is CLEAN, so execution is the only
             in-budget detector.

Schema (single-file tasks; everything needed is in target.py + the shown test, so R0 is not
penalised for a file it cannot read):
  name, group("hard"|"easy"), target("target.py"), files{"target.py": stub},
  test(VISIBLE), held_out(HELD-OUT), gold_target, wrong_guess, sim, note.

Run the verifier with .venv-streams.system/bin/python; one pyrefly process at a time.
api_agent.py loads it via --suite runtime / TASKS_RUNTIME.
"""


def _task(name, sim, head, gold_body, wrong_body, test, held_out, note):
    """Assemble a single-file task. `head` is import(s)+def+docstring ending in a newline;
    `gold_body`/`wrong_body` are the full indented function body (4-space indent, trailing
    newline). The stub raises NotImplementedError."""
    stub = head + "    raise NotImplementedError\n"
    return dict(
        name=name, group=sim, target="target.py",
        files={"target.py": stub},
        test=test, held_out=held_out,
        gold_target=head + gold_body, wrong_guess=head + wrong_body,
        sim=sim, note=note)


# ===================================================================================
# HARD — multi-step / stateful / boundary logic; a careful read is genuinely unreliable,
# so running the test is expected to carry information.
# ===================================================================================

# H1 — sliding window, off-by-one drops the FINAL window. --------------------------
_H1_WINDOW = _task(
    "rt_window_sum", "hard",
    head=(
        "def max_window_sum(nums: list[int], k: int) -> int:\n"
        '    """Return the largest sum of any k consecutive elements of nums.\n'
        '    Assumes 1 <= k <= len(nums)."""\n'
    ),
    gold_body=(
        "    best = cur = sum(nums[:k])\n"
        "    for i in range(k, len(nums)):\n"
        "        cur += nums[i] - nums[i - k]\n"
        "        if cur > best:\n"
        "            best = cur\n"
        "    return best\n"
    ),
    wrong_body=(
        "    best = sum(nums[:k])\n"
        "    for i in range(len(nums) - k):\n"
        "        best = max(best, sum(nums[i:i + k]))\n"
        "    return best\n"
    ),
    test=(
        "from target import max_window_sum\n"
        "assert max_window_sum([1, 2, 1, 1, 5, 6], 2) == 11\n"
    ),
    held_out=(
        "from target import max_window_sum\n"
        "assert max_window_sum([3, 1, 2, 2, 9], 2) == 11\n"
    ),
    note=("Sliding window: the wrong fix's loop range stops one short (range(len-k) instead "
          "of range(len-k+1)), dropping the FINAL k-window. It only differs when the maximum "
          "window is the last one. The off-by-one is easy to miss by reading; running the "
          "test exposes it."),
)

# H2 — even split: leftover cents to FIRST r people, not last r. --------------------
_H2_SPLIT = _task(
    "rt_split_bill", "hard",
    head=(
        "def split_bill(total_cents: int, n: int) -> list[int]:\n"
        '    """Split total_cents as evenly as possible among n people. Each person gets\n'
        '    total_cents // n; the leftover total_cents % n cents go one each to the FIRST\n'
        '    that many people. The returned list sums to total_cents."""\n'
    ),
    gold_body=(
        "    base, rem = divmod(total_cents, n)\n"
        "    return [base + (1 if i < rem else 0) for i in range(n)]\n"
    ),
    wrong_body=(
        "    base, rem = divmod(total_cents, n)\n"
        "    return [base + (1 if i >= n - rem else 0) for i in range(n)]\n"
    ),
    test=(
        "from target import split_bill\n"
        "assert split_bill(100, 3) == [34, 33, 33]\n"
    ),
    held_out=(
        "from target import split_bill\n"
        "assert split_bill(14, 4) == [4, 4, 3, 3]\n"
    ),
    note=("Both candidate fixes distribute the leftover cents and both sum to total; they "
          "differ only in WHICH people get the extra cent (first r vs last r). The spec says "
          "the first r. A sum check passes both fixes; only the positional test discriminates."),
)

# H3 — run-length encode: the classic forgotten final flush. -----------------------
_H3_RLE = _task(
    "rt_run_lengths", "hard",
    head=(
        "def run_lengths(items: list[str]) -> list[tuple[str, int]]:\n"
        '    """Encode items as (value, run_length) pairs for each maximal run of equal\n'
        '    consecutive values, in order. E.g. [\'a\',\'a\',\'b\'] -> [(\'a\',2),(\'b\',1)]."""\n'
    ),
    gold_body=(
        "    out: list[tuple[str, int]] = []\n"
        "    i, n = 0, len(items)\n"
        "    while i < n:\n"
        "        j = i\n"
        "        while j < n and items[j] == items[i]:\n"
        "            j += 1\n"
        "        out.append((items[i], j - i))\n"
        "        i = j\n"
        "    return out\n"
    ),
    wrong_body=(
        "    out: list[tuple[str, int]] = []\n"
        "    if not items:\n"
        "        return out\n"
        "    cur, count = items[0], 1\n"
        "    for x in items[1:]:\n"
        "        if x == cur:\n"
        "            count += 1\n"
        "        else:\n"
        "            out.append((cur, count))\n"
        "            cur, count = x, 1\n"
        "    return out\n"
    ),
    test=(
        "from target import run_lengths\n"
        "assert run_lengths(['a', 'a', 'b', 'b', 'b']) == [('a', 2), ('b', 3)]\n"
    ),
    held_out=(
        "from target import run_lengths\n"
        "assert run_lengths(['x', 'y', 'y']) == [('x', 1), ('y', 2)]\n"
    ),
    note=("Single-pass run-length encoders famously forget to FLUSH the final run after the "
          "loop ends, so the last group is dropped. It is invisible unless you trace what is "
          "still in the accumulator at loop exit, or run the test."),
)

# H4 — discount tiers: inclusive boundary (>= vs >). --------------------------------
_H4_TIERS = _task(
    "rt_discount_tiers", "hard",
    head=(
        "def final_price(cents: int) -> int:\n"
        '    """Apply a volume discount to an order of `cents`: under 1000, no discount;\n'
        '    1000 to 4999 inclusive, 10% off; 5000 and above, 20% off. Return the integer\n'
        '    discounted price (cents * 9 // 10 for 10% off, cents * 8 // 10 for 20%)."""\n'
    ),
    gold_body=(
        "    if cents >= 5000:\n"
        "        return cents * 8 // 10\n"
        "    if cents >= 1000:\n"
        "        return cents * 9 // 10\n"
        "    return cents\n"
    ),
    wrong_body=(
        "    if cents > 5000:\n"
        "        return cents * 8 // 10\n"
        "    if cents > 1000:\n"
        "        return cents * 9 // 10\n"
        "    return cents\n"
    ),
    test=(
        "from target import final_price\n"
        "assert final_price(5000) == 4000\n"
    ),
    held_out=(
        "from target import final_price\n"
        "assert final_price(1000) == 900\n"
    ),
    note=("Inclusive tier boundaries: the spec applies the discount AT 1000 and AT 5000 "
          "(>=), but the plausible '> threshold' fix drops the exact-boundary order to the "
          "lower tier. It differs only exactly on the boundary values; a boundary test "
          "catches it."),
)

# H5 — longest strictly-increasing run: step-vs-element off-by-one + single floor. --
_H5_RUN = _task(
    "rt_increasing_run", "hard",
    head=(
        "def longest_increasing_run(nums: list[int]) -> int:\n"
        '    """Return the length (number of elements) of the longest contiguous run of\n'
        '    STRICTLY increasing values. A single element has length 1; an empty list 0."""\n'
    ),
    gold_body=(
        "    if not nums:\n"
        "        return 0\n"
        "    best = cur = 1\n"
        "    for i in range(1, len(nums)):\n"
        "        cur = cur + 1 if nums[i] > nums[i - 1] else 1\n"
        "        if cur > best:\n"
        "            best = cur\n"
        "    return best\n"
    ),
    wrong_body=(
        "    if not nums:\n"
        "        return 0\n"
        "    best = cur = 0\n"
        "    for i in range(1, len(nums)):\n"
        "        cur = cur + 1 if nums[i] > nums[i - 1] else 0\n"
        "        if cur > best:\n"
        "            best = cur\n"
        "    return best\n"
    ),
    test=(
        "from target import longest_increasing_run\n"
        "assert longest_increasing_run([1, 3, 2, 4, 6, 5]) == 3\n"
    ),
    held_out=(
        "from target import longest_increasing_run\n"
        "assert longest_increasing_run([2, 2, 2]) == 1\n"
    ),
    note=("Run-length counting: counting increasing STEPS (init 0) yields one less than the "
          "element count and also breaks the single-element floor (the answer should be 1, "
          "not 0). The off-by-one is invisible by inspection unless you trace a concrete run."),
)


# ===================================================================================
# EASY — one-line specs where the fix is obvious from reading; a strong model should
# solve these without running, so execution is expected to add little.
# ===================================================================================

_E1_ABS = _task(
    "rt_abs_gap", "easy",
    head=(
        "def gap(a: int, b: int) -> int:\n"
        '    """Return the non-negative distance between a and b."""\n'
    ),
    gold_body="    return abs(a - b)\n",
    wrong_body="    return a - b\n",
    test=(
        "from target import gap\n"
        "assert gap(2, 5) == 3\n"
    ),
    held_out=(
        "from target import gap\n"
        "assert gap(1, 9) == 8\n"
    ),
    note=("A signed subtraction where the spec asks for a non-negative distance; the missing "
          "abs() is obvious from the one-line spec."),
)

_E2_CLAMP = _task(
    "rt_clamp", "easy",
    head=(
        "def clamp(x: int, lo: int, hi: int) -> int:\n"
        '    """Return x constrained to the inclusive range [lo, hi]: values below lo become\n'
        '    lo, values above hi become hi."""\n'
    ),
    gold_body="    return max(lo, min(x, hi))\n",
    wrong_body="    return min(x, hi)\n",
    test=(
        "from target import clamp\n"
        "assert clamp(-5, 0, 10) == 0\n"
    ),
    held_out=(
        "from target import clamp\n"
        "assert clamp(-1, 2, 8) == 2\n"
    ),
    note=("Two-sided clamp; the one-sided min(x, hi) misses the lower bound, which is plain "
          "from the spec naming both bounds."),
)

_E3_FIRST = _task(
    "rt_first_or", "easy",
    head=(
        "def first_or(items: list[int], default: int) -> int:\n"
        '    """Return the first element of items, or `default` if items is empty."""\n'
    ),
    gold_body="    return items[0] if items else default\n",
    wrong_body="    return items[0]\n",
    test=(
        "from target import first_or\n"
        "assert first_or([], 7) == 7\n"
    ),
    held_out=(
        "from target import first_or\n"
        "assert first_or([], -1) == -1\n"
    ),
    note=("Missing empty-list guard; the spec explicitly names the empty case, so the "
          "IndexError is evident from reading."),
)


# ===================================================================================
# TRAP — the wrong fix is a PLAUSIBLE reimplementation that mis-simulates a real Python
# semantic (str.lstrip is char-set not prefix; int(x/k) truncates toward zero while //
# floors; round() is banker's; slice composition; multi-wrap modulo). Designed to defeat
# mental execution: a careful reader who knows the exact semantic writes the gold; one who
# mis-remembers it writes the wrong fix and only RUNNING reveals the divergence. Same W1-W4
# invariant (visible test fails the wrong fix; wrong fix is type-clean).
# ===================================================================================

# T1 — str.lstrip strips a CHARACTER SET, not a prefix. -----------------------------
_T1_LSTRIP = _task(
    "rt_strip_prefix", "trap",
    head=(
        "def strip_prefix(s: str, prefix: str) -> str:\n"
        '    """Return s with `prefix` removed from its start if s starts with it; else\n'
        '    return s unchanged. Only a single leading exact `prefix` is removed."""\n'
    ),
    gold_body="    return s[len(prefix):] if s.startswith(prefix) else s\n",
    wrong_body="    return s.lstrip(prefix)\n",
    test=(
        "from target import strip_prefix\n"
        "assert strip_prefix('aabbcc', 'ab') == 'aabbcc'\n"
    ),
    held_out=(
        "from target import strip_prefix\n"
        "assert strip_prefix('abab', 'ab') == 'ab'\n"
    ),
    note=("str.lstrip(prefix) strips any leading characters in the SET {prefix chars}, not "
          "the prefix string: 'aabbcc'.lstrip('ab') == 'cc'. The gold checks startswith and "
          "slices. A reader who mis-remembers lstrip writes the wrong fix; running reveals it."),
)

# T2 — int(v / size) truncates toward zero; v // size floors. -----------------------
_T2_FLOORDIV = _task(
    "rt_bucket_floor", "trap",
    head=(
        "def bucket(v: int, size: int) -> int:\n"
        '    """Return the index of the size-wide bucket v falls into, using FLOOR division\n'
        '    so negatives go to lower buckets. size 10: 5->0, -5->-1, -15->-2."""\n'
    ),
    gold_body="    return v // size\n",
    wrong_body="    return int(v / size)\n",
    test=(
        "from target import bucket\n"
        "assert bucket(-5, 10) == -1\n"
    ),
    held_out=(
        "from target import bucket\n"
        "assert bucket(-15, 10) == -2\n"
    ),
    note=("int(v / size) truncates toward zero (int(-5/10) == 0) while v // size floors "
          "(-5 // 10 == -1). They agree on non-negative v and diverge on negatives; the "
          "negative test exposes the wrong fix."),
)

# T3 — round() is banker's rounding (ties to even), not int(x + 0.5). ---------------
_T3_BANKERS = _task(
    "rt_round_even", "trap",
    head=(
        "def to_nearest(x: float) -> int:\n"
        '    """Round x to the nearest integer using Python\'s built-in round (banker\'s\n'
        '    rounding: exact halves go to the nearest EVEN integer). 2.5->2, 3.5->4, 0.5->0."""\n'
    ),
    gold_body="    return round(x)\n",
    wrong_body="    return int(x + 0.5)\n",
    test=(
        "from target import to_nearest\n"
        "assert to_nearest(2.5) == 2\n"
    ),
    held_out=(
        "from target import to_nearest\n"
        "assert to_nearest(0.5) == 0\n"
    ),
    note=("int(x + 0.5) rounds halves UP (int(2.5+0.5)==3) while round() rounds halves to "
          "even (round(2.5)==2). The half-integer test exposes the reimplementation."),
)

# T4 — max(dict) maxes over KEYS; needs key= to max over values. --------------------
_T4_MAXKEY = _task(
    "rt_top_label", "trap",
    head=(
        "def top_label(scores: dict[str, int]) -> str:\n"
        '    """Return the label with the highest score. Assume a unique maximum."""\n'
    ),
    gold_body="    return max(scores, key=lambda k: scores[k])\n",
    wrong_body="    return max(scores)\n",
    test=(
        "from target import top_label\n"
        "assert top_label({'a': 5, 'b': 1}) == 'a'\n"
    ),
    held_out=(
        "from target import top_label\n"
        "assert top_label({'apple': 9, 'zebra': 1}) == 'apple'\n"
    ),
    note=("max(scores) iterates the dict's KEYS and returns the largest key string, not the "
          "highest-scoring label; the gold passes key=. The test picks data where the "
          "alphabetically-last key is not the top scorer."),
)

# T5 — slice composition: xs[1::2] is not xs[::2][1:]. ------------------------------
_T5_SLICE = _task(
    "rt_every_other", "trap",
    head=(
        "def every_other_from_second(xs: list[int]) -> list[int]:\n"
        '    """Return every other element starting from the SECOND element (index 1):\n'
        '    [10,20,30,40,50] -> [20,40]."""\n'
    ),
    gold_body="    return xs[1::2]\n",
    wrong_body="    return xs[::2][1:]\n",
    test=(
        "from target import every_other_from_second\n"
        "assert every_other_from_second([10, 20, 30, 40, 50]) == [20, 40]\n"
    ),
    held_out=(
        "from target import every_other_from_second\n"
        "assert every_other_from_second([1, 2, 3, 4, 5, 6]) == [2, 4, 6]\n"
    ),
    note=("xs[::2][1:] takes every other from the FIRST element then drops one ([30,50]), "
          "which is not xs[1::2] ([20,40]). Two slice expressions that look interchangeable; "
          "evaluating both in your head is exactly what running checks."),
)

# T6 — modulo wrap: a single subtract handles only one wrap, and not negatives. -----
_T6_MODWRAP = _task(
    "rt_ring_positions", "trap",
    head=(
        "def positions(steps: list[int], n: int) -> list[int]:\n"
        '    """A token starts at 0 on a ring of n cells and moves by each step in turn;\n'
        '    positions wrap modulo n and stay in [0, n). Return the position after each\n'
        '    step. Steps may be negative; Python\'s % gives a non-negative result here."""\n'
    ),
    gold_body=(
        "    pos = 0\n"
        "    out: list[int] = []\n"
        "    for s in steps:\n"
        "        pos = (pos + s) % n\n"
        "        out.append(pos)\n"
        "    return out\n"
    ),
    wrong_body=(
        "    pos = 0\n"
        "    out: list[int] = []\n"
        "    for s in steps:\n"
        "        pos = pos + s\n"
        "        if pos >= n:\n"
        "            pos -= n\n"
        "        out.append(pos)\n"
        "    return out\n"
    ),
    test=(
        "from target import positions\n"
        "assert positions([-3], 5) == [2]\n"
    ),
    held_out=(
        "from target import positions\n"
        "assert positions([12], 5) == [2]\n"
    ),
    note=("The wrong fix's single 'if pos >= n: pos -= n' handles one positive wrap only: it "
          "leaves negative steps negative (-3 stays -3, not 2) and under-wraps large steps "
          "(12 -> 7, not 2). Only the modulo handles both; running with a negative or "
          "double-wrap step exposes it."),
)


TASKS_RUNTIME = [
    _H1_WINDOW,
    _H2_SPLIT,
    _H3_RLE,
    _H4_TIERS,
    _H5_RUN,
    _E1_ABS,
    _E2_CLAMP,
    _E3_FIRST,
    _T1_LSTRIP,
    _T2_FLOORDIV,
    _T3_BANKERS,
    _T4_MAXKEY,
    _T5_SLICE,
    _T6_MODWRAP,
]


# ===================================================================================
# VERIFIER (__main__) — enforce the RUNTIME-SUITE INVARIANT W1..W4.
# ===================================================================================
if __name__ == "__main__":
    import os as _os
    import sys
    import json
    import subprocess
    sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
    from scaffold.mock_env import MultiFileEnv, PYREFLY

    # Kill any stale pyrefly daemon before we start; we run strictly sequentially.
    # (list-form pkill: the literal "[p]yrefly" in argv never matches the regex itself.)
    subprocess.run(["pkill", "-9", "-f", "[p]yrefly"], capture_output=True)

    def pyrefly_target_errs(env, target):
        """Target-scoped pyrefly errors (full dicts) for the files currently in env.ws."""
        try:
            r = subprocess.run([PYREFLY, "check", "--output-format", "json"], cwd=env.ws,
                               capture_output=True, text=True, timeout=90)
            errs = json.loads(r.stdout or "{}").get("errors", [])
        except Exception as e:
            return [{"name": "INVOKE_FAIL", "description": str(e)}]
        return [e for e in errs if _os.path.basename(e.get("path", "") or "") == target]

    def _diag(e):
        return (e.get("concise_description") or e.get("description") or "")[:160]

    def _splice(t, body_src):
        return {**t["files"], t["target"]: body_src}

    print(f"{'task':22} {'sim':5} {'W1':6} {'W2vis':6} {'W2held':7} "
          f"{'W3vis':6} {'W3held':7} {'W4type':10}")
    allok = True
    n_hard = n_easy = 0

    for t in TASKS_RUNTIME:
        tgt = t["target"]
        n_hard += t["group"] == "hard"
        n_easy += t["group"] == "easy"

        # W1 — stub: held-out score() FAILS (unimplemented).
        e_stub = MultiFileEnv(t["files"], tgt, t["test"], held_out_src=t["held_out"],
                              skip_pyrefly=True)
        w1 = not e_stub.score()["resolved"]
        e_stub.close()

        # W2 — gold: visible PASSES and held-out PASSES.
        e_gold = MultiFileEnv(_splice(t, t["gold_target"]), tgt, t["test"],
                              held_out_src=t["held_out"], skip_pyrefly=True)
        w2_vis = e_gold.run_tests()["resolved"]
        w2_held = e_gold.score()["resolved"]
        e_gold.close()

        # W3 — wrong: visible FAILS and held-out FAILS, and wrong != gold (the KEY:
        #       running the visible test exposes the bug).
        e_wrong = MultiFileEnv(_splice(t, t["wrong_guess"]), tgt, t["test"],
                               held_out_src=t["held_out"], skip_pyrefly=True)
        w3_vis_fail = not e_wrong.run_tests()["resolved"]
        w3_held_fail = not e_wrong.score()["resolved"]
        e_wrong.close()
        w3 = w3_vis_fail and w3_held_fail and (t["wrong_guess"] != t["gold_target"])

        # W4 — wrong fix is WELL-TYPED: target-scoped pyrefly is clean (so a type checker is
        #       NOT a detector here; execution is the only in-budget one).
        e_typ = MultiFileEnv(_splice(t, t["wrong_guess"]), tgt, t["test"],
                             held_out_src=t["held_out"])   # skip_pyrefly=False -> init pyrefly
        werrs = pyrefly_target_errs(e_typ, tgt)
        e_typ.close()
        w4 = len(werrs) == 0

        ok = w1 and w2_vis and w2_held and w3 and w4
        allok = allok and ok
        print(f"{t['name']:22} {t['group']:5} "
              f"{'FAIL' if w1 else 'PASS!':6} "
              f"{'PASS' if w2_vis else 'FAIL!':6} "
              f"{'PASS' if w2_held else 'FAIL!':7} "
              f"{'FAIL' if w3_vis_fail else 'PASS!':6} "
              f"{'FAIL' if w3_held_fail else 'PASS!':7} "
              f"{('clean' if w4 else f'{len(werrs)}err!'):10}"
              f"{'' if ok else '   <-- PROBLEM'}")
        if not w3:
            print(f"     ! W3 violated: wrong visible_fail={w3_vis_fail} "
                  f"held_fail={w3_held_fail} differs={(t['wrong_guess'] != t['gold_target'])}")
        if not w4:
            print(f"     ! W4 violated: wrong fix is NOT type-clean: "
                  f"{[(e.get('name'), _diag(e)) for e in werrs][:3]}")

    print("\n--- per-task note (sim difficulty / what the test exposes) ---")
    for t in TASKS_RUNTIME:
        print(f"  [{t['group']:4}] {t['name']}:")
        print(f"      {t['note']}")

    print(f"\nALL OK ({len(TASKS_RUNTIME)} tasks: {n_hard} hard, {n_easy} easy)" if allok
          else "\nPROBLEMS — fix before running the matrix")
