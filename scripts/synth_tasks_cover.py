#!/usr/bin/env python3
"""COVERAGE-JUDGING suite — does a coding agent decide to <read> based on the CONTENT that a <defn>
returns, NOT on the task's surface shape? Sibling of synth_tasks_effic_nodel.py (the NODEL suite); built
to isolate ONE decision: "is the value I need PRESENT in the span <defn sym> just gave me, or must I
read on?" — judged purely from retrieved content, with surface cues and the delegation escape removed.

WHY A NEW SUITE (what the three prior attempts got wrong, and the fix each constraint encodes):
  - DIFFICULTY FLOOR (the failure that killed the last suite): the needed content was a multi-entry table,
    so even a model that DID read could not transcribe it and _suf was unsolvable -> no signal. FIX (R-small):
    the needed content is ONE arbitrary constant; the gold fix is a single changed number (<=3 changed lines),
    transcribable by a 7B/27B. The suite measures the READ DECISION, not transcription stamina.
  - DELEGATION escape: a forwarding gold (`return grade(score)`) is coverage-blind — it works whether the
    defn returned the body or a stub, because the agent never needs the body's CONTENT. FIX (R-nodel): target.py
    imports only a VERSION SENTINEL (`from biglib import GRADE_VERSION`); `grade` itself is NOT imported and the
    prompt forbids adding the import (hot-path / circular-import rationale). There is no callable to forward to,
    so the fix must INLINE-correct the constant.
  - GUESSABILITY: an idiomatic default (cutoff 60, factor 100, ...) is a free correct guess. FIX (R-arb): every
    constant is arbitrary (2-3 digits, no idiomatic default; the obvious guess differs) and tests are
    checksum/hash-only (INPUTS + sha256 of expected outputs; no expected literal in the test), so guess-then-fit
    cannot converge.
  - SURFACE LEAKAGE: anything in the prompt/target/test that predicts coverage lets a surface-keyed model
    cheat. FIX (R-surface): within a topic, `_suf` / `_f1ins` / `_f2ins` have BYTE-IDENTICAL target.py, prompt,
    and test. The ONLY difference is biglib.py's definition of the symbol -> the only way to know coverage is to
    call <defn sym> and inspect the span.

THE DESIGN — three variants per TOPIC, differing only in what <defn sym> returns:
  _suf  (coverage SUFFICIENT, mechanism none): biglib has `def sym(x): return "P" if x >= 53 else "F"` — the
        needed constant (53) is INLINE in the body. <defn sym> returns it -> solvable from the defn alone.
  _f1ins (INSUFFICIENT, mechanism F1 = forwarding alias): biglib has `sym = _sym_impl` (a one-line alias) plus
        >=5 decoy `_*_impl` functions with DIFFERENT cutoffs. <defn sym> returns ONLY the alias line — the value
        is absent and the alias names the impl, so the agent must <read> to find `_sym_impl`'s body.
  _f2ins (INSUFFICIENT, mechanism F2 = split definition): biglib has `_SYM_CUT = 53` (among decoy `_*_CUT`
        constants) and a REAL function `def sym(x): return "P" if x >= _SYM_CUT else "F"`. <defn sym> returns the
        function BODY — which shows the STRUCTURE but EXTERNALISES the value -> the agent must <read> for _SYM_CUT.
        F2 is the subtle case: the returned defn LOOKS like a normal function (not an obvious stub), so a model
        keyed on "stub-shape -> read" will NOT read; only a model that judges "the VALUE I need isn't in what I
        got back -> read" will. F1 vs F2 thus separates shape-judging from content-judging.

  A coverage-judging agent solves ALL THREE. A surface/shape-keyed agent solves _suf, and on _f2ins (and often
  _f1ins) emits the idiomatic-but-WRONG constant and FAILS, because it never read the externalised value.

NO-DELEGATION, concretely (R-nodel): biglib exposes only the VERSION SENTINEL `GRADE_VERSION` for target.py to
import. `grade` is never imported into target.py and the prompt states the inline copy must stand alone (biglib
is a heavy/circular import at call time). The gold fix changes ONLY the inline constant; it adds no import and
no call to `grade`. The verifier asserts the gold neither imports nor calls the symbol.

SCHEMA (per task dict) — same shape as the NODEL suite plus the coverage meta:
  name (cover_<topic>_suf | _f1ins | _f2ins), topic, group("rich"), target("target.py"),
  symbol (the function the agent inspects with <defn>, e.g. "grade" — SAME string across the three variants),
  meta {"coverage": "sufficient"|"insufficient", "mechanism": "none"|"f1"|"f2"} (also encoded in the name),
  defn_sufficient (bool), requires_read (bool; True on _f1ins/_f2ins),
  files {target.py(buggy inline copy), biglib.py(symbol defined per-variant + decoys)},
  test (checksum, passes on gold), gold_target (inline-corrected copy), inputs, real_body (the gold target body,
  exec'able in-process for the digest), local (the public fn the test imports),
  wrong_guess (idiomatic-WRONG gold splice -> R-arb), wrong_kind("value"), wrong_note,
  needed_value (the distinctive arbitrary constant the fix must transcribe, as a string -> R-defn checks spans),
  delegate_syms (symbols a forwarding fix COULD try -> R-nodel asserts the gold calls none of them).

VERIFIER (__main__, drives mock_env.MultiFileEnv.goto_definition for the spans). Per task assert:
  R1  buggy target.py FAILs the test;
  R2  gold PASSes AND is pyrefly-clean (nerr==0; pyrefly run SEQUENTIALLY; kill daemon first);
  R-small  the gold diff vs buggy is <=3 changed lines (a small fact, transcribable by a 7B/27B);
  R-nodel  the gold neither imports a callable form of `sym` nor calls it (no `from biglib import grade`,
           no `grade(`); a delegation attempt is impossible at the fix site;
  R-arb  the idiomatic-guess value != the gold value on the pinned inputs (so the constant is not name-derivable);
  R5  no answer-literal leak: no expected-output literal and no needed_value token in the prompt/test (only
      INPUTS + the 64-hex hash);
  R-surface  within each topic the three variants' target.py + test + symbol are BYTE-IDENTICAL (no surface cue);
  R-defn  goto_definition(sym): _suf's span CONTAINS needed_value; _f1ins's span is the bare alias line (value
          absent, names the impl) and the value is recoverable only by reading biglib; _f2ins's span is the
          function body (value absent from the span) and the value is recoverable only by reading biglib.
Prints "ALL OK (<n> tasks)" only when every task passes. Run (pyrefly runs SEQUENTIALLY; kill daemon first):
  pkill -9 -x pyrefly; \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/mnt/nas/hf-cache \
  .venv-streams.system/bin/python scripts/synth_tasks_cover.py
"""

# ---- filler so biglib.py is genuinely big (a <read> must scan it; <defn> returns just one node) ----
def _filler(n, base=0):
    out = []
    for i in range(base, base + n):
        out.append(
            f"class _Aux{i}:\n"
            f"    \"\"\"Internal helper {i} — unrelated to the task.\"\"\"\n"
            f"    def __init__(self, seed: int = {i}) -> None:\n"
            f"        self._s = seed\n"
            f"    def mix(self, x: int) -> int:\n"
            f"        return (x * {i + 3}) ^ self._s\n"
            f"    def label(self) -> str:\n"
            f"        return f\"aux{i}:{{self._s}}\"\n")
    return "\n".join(out)


def _scatter(blocks, gaps=(13, 11, 12, 10, 13)):
    """Interleave real/decoy `blocks` between filler bands so every symbol is buried deep in a big file —
    a <read> must scan it, while <defn> on a known name returns just the one node, wherever it sits."""
    parts = []
    base = 0
    for i, blk in enumerate(blocks):
        g = gaps[i % len(gaps)]
        parts.append(_filler(g, base)); base += g
        parts.append(blk)
    parts.append(_filler(gaps[-1], base))
    return "\n\n".join(parts)


# ==================================================================================================
# Each TOPIC is a single-constant transform `sym(x)` that target.py re-implements INLINE with the WRONG
# constant. The authoritative constant lives in biglib's `sym`; the three variants differ ONLY in how
# biglib defines `sym`, which changes what <defn sym> returns (see module docstring). Every constant is
# arbitrary (2-3 digits, no idiomatic default) so it is non-guessable; the idiomatic guess differs.
#
# Per-topic fields:
#   topic, local (public fn name in target.py the test imports), arg name + annotation, return ann,
#   tmpl: a format template for the ONE-LINE body using {C} (the constant) and {arg}; it must read
#         `return <expr involving {arg} and {C}>`. SAME template renders the buggy copy (with wrong C),
#         the gold (with the real C), biglib's _suf body, and F2's `def sym` (with _SYM_CUT in place of C).
#   gold_c   : the real arbitrary constant (the value the agent must retrieve),
#   buggy_c  : the WRONG constant the inline copy currently has (a plausible-but-wrong number),
#   guess_c  : the IDIOMATIC default a non-reader would emit (differs from gold_c on the inputs -> R-arb),
#   inputs   : a fixed call set spanning both sides of the transform,
#   const_name : the F2 module-level constant name (e.g. "_GRADE_CUT"),
#   doc      : the one-line public docstring/prose for the target function.
# The constant is a 2-3 digit number with NO idiomatic default; `guess_c` is the obvious-but-wrong value.
# ==================================================================================================
_TOPICS = [
    dict(topic="grade", local="report", arg="score", argann="int", ret="str",
         sym="grade", const_name="_GRADE_CUT",
         tmpl='return "P" if {arg} >= {C} else "F"',
         gold_c=53, buggy_c=60, guess_c=60,            # idiomatic pass mark 60 is WRONG; real cutoff 53
         inputs=[(53,), (52,), (60,), (59,), (0,), (100,)],
         doc='Return "P"/"F" for `score` (must match biglib.grade).'),

    dict(topic="discount", local="net_cents", arg="cents", argann="int", ret="int",
         sym="discount", const_name="_DISC_OFF",
         tmpl='return {arg} - {C}',
         gold_c=147, buggy_c=100, guess_c=0,           # idiomatic "no discount" 0 is WRONG; real off 147
         inputs=[(500,), (200,), (147,), (1000,), (148,), (3000,)],
         doc="Return `cents` minus the fixed discount (must match biglib.discount)."),

    dict(topic="retry", local="attempts_left", arg="used", argann="int", ret="int",
         sym="retry", const_name="_RETRY_MAX",
         tmpl='return {C} - {arg}',
         gold_c=37, buggy_c=3, guess_c=3,              # idiomatic max-retries 3 is WRONG; real cap 37
         inputs=[(0,), (10,), (37,), (36,), (1,), (20,)],
         doc="Return retries remaining after `used` (must match biglib.retry)."),

    dict(topic="threshold", local="is_hot", arg="temp", argann="int", ret="bool",
         sym="threshold", const_name="_HOT_AT",
         tmpl='return {arg} >= {C}',
         gold_c=212, buggy_c=100, guess_c=100,         # idiomatic boiling 100C is WRONG; real cut 212
         inputs=[(212,), (211,), (100,), (300,), (0,), (213,)],
         doc="Return whether `temp` is hot (must match biglib.threshold)."),

    dict(topic="shipfee", local="fee_for", arg="grams", argann="int", ret="int",
         sym="shipfee", const_name="_FREE_OVER",
         tmpl='return 0 if {arg} >= {C} else 599',
         gold_c=864, buggy_c=1000, guess_c=1000,       # idiomatic free-over-1000 is WRONG; real 864
         inputs=[(864,), (863,), (1000,), (0,), (865,), (500,)],
         doc="Return the shipping fee for `grams` (must match biglib.shipfee)."),

    dict(topic="token", local="bucket_of", arg="code", argann="int", ret="int",
         sym="token", const_name="_TOK_MOD",
         tmpl='return {arg} % {C}',
         gold_c=83, buggy_c=100, guess_c=10,           # idiomatic mod-10 is WRONG; real modulus 83
         inputs=[(0,), (83,), (84,), (200,), (412,), (83 * 5,)],
         doc="Return the bucket for `code` (must match biglib.token)."),
]


# ---------------------------------------------------------------------------------------------------
# target.py — the ONLY thing the model sees; BYTE-IDENTICAL across the three variants of a topic. It
# imports ONLY the version sentinel `<SYM>_VERSION` (NOT the callable `sym`): the realistic reason the
# inline copy exists, and the reason there is no symbol to delegate to at the fix site. The docstring
# states the inline copy is a denormalised hot-path copy of biglib.<sym> and that biglib is intentionally
# not imported at call time (heavy module / would be a circular import), so the constant below must be
# transcribed to match biglib.<sym> for this version — and the import must NOT be added.
# ---------------------------------------------------------------------------------------------------
def _ver_name(p):  # the version sentinel symbol, e.g. grade -> GRADE_VERSION
    return p["sym"].upper() + "_VERSION"


def _emit_target(p, c):
    ver = _ver_name(p)
    body = p["tmpl"].format(arg=p["arg"], C=c)
    return (
        f"from biglib import {ver}  # version sentinel ONLY; biglib.{p['sym']} is NOT imported (see below)\n\n"
        f"def {p['local']}({p['arg']}: {p['argann']}) -> {p['ret']}:\n"
        f"    \"\"\"{p['doc']}\"\"\"\n"
        f"    # Denormalised INLINE copy of biglib.{p['sym']}, kept here for the hot path: biglib is\n"
        f"    # intentionally NOT imported at call time (heavy module / would be a circular import), so do\n"
        f"    # NOT add `from biglib import {p['sym']}` — the constant below must be transcribed to match\n"
        f"    # biglib.{p['sym']} for {ver}.\n"
        f"    assert {ver} >= 1\n"
        f"    {body}\n"
    )


def _emit_buggy(p):
    return _emit_target(p, p["buggy_c"])


def _emit_gold(p):
    return _emit_target(p, p["gold_c"])


def _emit_wrong(p):
    # idiomatic-but-WRONG: the inline copy with the idiomatic-default constant (type-clean, value-wrong).
    return _emit_target(p, p["guess_c"])


def _gold_body_src(p):
    """The gold target function as a BARE def (no `from biglib import` header, no version assert) so it
    can be exec'd in-process to derive the digest / expected outputs without importing biglib."""
    body = p["tmpl"].format(arg=p["arg"], C=p["gold_c"])
    return (f"def {p['local']}({p['arg']}):\n"
            f"    {body}\n")


def _gold_fn(p):
    ns: dict = {}
    exec(_gold_body_src(p), ns)
    return ns[p["local"]]


def _gold_hash(p):
    import hashlib
    fn = _gold_fn(p)
    got = "|".join(repr(fn(*args)) for args in p["inputs"])
    return hashlib.sha256(got.encode()).hexdigest()


def _emit_test(p):
    """CHECKSUM test (no expected-output literal): call the function under test over a fixed INPUT set and
    assert sha256(joined repr()s) == precomputed gold digest. A failing run reveals only an opaque hash
    mismatch, so 'guess then fit to the revealed expected output' cannot converge — combined with the
    arbitrary constant, the ONLY way to pass is to retrieve the constant. Digest derived in-process."""
    return (f"from target import {p['local']}\n"
            "import hashlib\n"
            f"INPUTS = {p['inputs']!r}\n"
            f"got = \"|\".join(repr({p['local']}(*args)) for args in INPUTS)\n"
            f"assert hashlib.sha256(got.encode()).hexdigest() == \"{_gold_hash(p)}\", \"wrong\"\n")


# ---------------------------------------------------------------------------------------------------
# biglib.py — three variants. In ALL three the version sentinel is defined first (so target.py's import
# type-checks); it is NOT the answer. The symbol `sym` is then defined differently per variant.
# ---------------------------------------------------------------------------------------------------
def _ver_line(p):
    return f"{_ver_name(p)} = 1\n"


# Decoy constants used in F1 (different cutoffs on the impls) and F2 (different _*_CUT constants), so the
# real one is not the only candidate a reader sees — they must resolve WHICH one `sym` actually uses.
_DECOY_DELTAS = [11, -7, 23, -3, 41, -17, 5]


def _biglib_suf(p):
    """VARIANT _suf: `sym` is a real function with the constant INLINE in the body -> <defn sym> returns
    the body and the value is right there."""
    body = p["tmpl"].format(arg="x", C=p["gold_c"])
    fn = (f"def {p['sym']}(x):\n"
          f"    \"\"\"Authoritative {p['sym']} (constant inline).\"\"\"\n"
          f"    {body}\n")
    return _ver_line(p) + "\n" + _scatter([fn])


def _f1_impls(p):
    """F1 decoy impls + the real `_sym_impl`. Each is a real function over the SAME template but a DIFFERENT
    cutoff; only `_<sym>_impl` carries the gold constant. The alias `sym = _<sym>_impl` names the impl, so
    the agent must READ to see which impl the alias points at AND read off its constant."""
    real_name = f"_{p['sym']}_impl"
    blocks = []
    # 5 decoys with different (wrong) constants, named distinctly so they are NOT the alias target.
    for i, d in enumerate(_DECOY_DELTAS[:5]):
        dc = p["gold_c"] + d
        b = p["tmpl"].format(arg="x", C=dc)
        blocks.append(f"def _{p['sym']}_v{i}(x):\n"
                      f"    \"\"\"Decoy variant {i} — NOT the authoritative impl.\"\"\"\n"
                      f"    {b}\n")
    # the REAL impl (gold constant inline).
    real_body = p["tmpl"].format(arg="x", C=p["gold_c"])
    real = (f"def {real_name}(x):\n"
            f"    \"\"\"The authoritative impl `{p['sym']}` aliases.\"\"\"\n"
            f"    {real_body}\n")
    return blocks, real, real_name


def _biglib_f1(p):
    """VARIANT _f1ins: `sym = _<sym>_impl` (a one-line forwarding alias) + >=5 decoy impls. <defn sym>
    returns ONLY the alias line (value absent) -> must <read> to resolve the impl and read its constant."""
    blocks, real, real_name = _f1_impls(p)
    alias = f"{p['sym']} = {real_name}"
    # decoys, then the real impl, then the alias, all buried in filler (real impl is NOT first or last).
    body = _ver_line(p) + "\n" + _scatter(blocks + [real, alias],
                                          gaps=(12, 10, 9, 11, 10, 9, 12))
    return body


def _alias_f1(p):
    _, _, real_name = _f1_impls(p)
    return f"{p['sym']} = {real_name}"


def _biglib_f2(p):
    """VARIANT _f2ins: a module-level constant `_<SYM>_CUT = <gold>` (among decoy `_*_CUT` constants) and a
    REAL function `def sym(x): return ... _<SYM>_CUT ...`. <defn sym> returns the function BODY — which shows
    the STRUCTURE but EXTERNALISES the value -> must <read> for `_<SYM>_CUT`."""
    cn = p["const_name"]
    # decoy constants with the SAME naming family but different values (none equals the gold constant).
    decoy_consts = []
    for i, d in enumerate(_DECOY_DELTAS[:5]):
        decoy_consts.append(f"{cn}_ALT{i} = {p['gold_c'] + d}")
    real_const = f"{cn} = {p['gold_c']}"
    body = p["tmpl"].format(arg="x", C=cn)
    fn = (f"def {p['sym']}(x):\n"
          f"    \"\"\"Authoritative {p['sym']} (cutoff externalised to {cn}).\"\"\"\n"
          f"    {body}\n")
    # decoys + the real const + the function, buried in filler (the real const is not first/last).
    blocks = decoy_consts[:2] + [real_const] + decoy_consts[2:] + [fn]
    return _ver_line(p) + "\n" + _scatter(blocks, gaps=(12, 10, 9, 11, 10, 9, 12))


def _defn_f2(p):
    """The function body <defn sym> returns in F2 — used only for documentation/oracle; the verifier
    recomputes it live via goto_definition."""
    body = p["tmpl"].format(arg="x", C=p["const_name"])
    return (f"def {p['sym']}(x):\n"
            f"    \"\"\"Authoritative {p['sym']} (cutoff externalised to {p['const_name']}).\"\"\"\n"
            f"    {body}\n")


TASKS_COVER = []
for _p in _TOPICS:
    _buggy = _emit_buggy(_p)
    _gold = _emit_gold(_p)
    _wrong = _emit_wrong(_p)
    _test = _emit_test(_p)
    _real_body = _gold_body_src(_p)
    _needed = str(_p["gold_c"])          # the distinctive arbitrary constant the fix must transcribe
    _delegate = [_p["sym"]]              # a forwarding fix COULD try to call `sym`; the gold must not

    # _suf — coverage sufficient (defn has the value inline)
    TASKS_COVER.append(dict(
        name=f"cover_{_p['topic']}_suf", topic=_p["topic"], group="rich", target="target.py",
        symbol=_p["sym"], meta={"coverage": "sufficient", "mechanism": "none"},
        defn_sufficient=True, requires_read=False,
        files={"target.py": _buggy, "biglib.py": _biglib_suf(_p)},
        test=_test, gold_target=_gold, inputs=_p["inputs"], real_body=_real_body, local=_p["local"],
        wrong_guess=_wrong, wrong_kind="value",
        wrong_note=f"idiomatic {_p['guess_c']} (real: {_p['gold_c']}) -> type-clean, value-wrong",
        needed_value=_needed, delegate_syms=_delegate))

    # _f1ins — insufficient, F1 forwarding alias (defn returns the bare alias line)
    TASKS_COVER.append(dict(
        name=f"cover_{_p['topic']}_f1ins", topic=_p["topic"], group="rich", target="target.py",
        symbol=_p["sym"], meta={"coverage": "insufficient", "mechanism": "f1"},
        defn_sufficient=False, requires_read=True,
        files={"target.py": _buggy, "biglib.py": _biglib_f1(_p)},
        test=_test, gold_target=_gold, inputs=_p["inputs"], real_body=_real_body, local=_p["local"],
        wrong_guess=_wrong, wrong_kind="value",
        wrong_note=f"idiomatic {_p['guess_c']} (real: {_p['gold_c']}) -> type-clean, value-wrong",
        needed_value=_needed, delegate_syms=_delegate))

    # _f2ins — insufficient, F2 split definition (defn returns body; value externalised to a constant)
    TASKS_COVER.append(dict(
        name=f"cover_{_p['topic']}_f2ins", topic=_p["topic"], group="rich", target="target.py",
        symbol=_p["sym"], meta={"coverage": "insufficient", "mechanism": "f2"},
        defn_sufficient=False, requires_read=True,
        files={"target.py": _buggy, "biglib.py": _biglib_f2(_p)},
        test=_test, gold_target=_gold, inputs=_p["inputs"], real_body=_real_body, local=_p["local"],
        wrong_guess=_wrong, wrong_kind="value",
        wrong_note=f"idiomatic {_p['guess_c']} (real: {_p['gold_c']}) -> type-clean, value-wrong",
        needed_value=_needed, delegate_syms=_delegate))


def _gold_output_literals(t):
    """Expected-output literals (repr of each gold result over INPUTS) that MUST NOT appear in the test."""
    ns: dict = {}
    exec(t["real_body"], ns)
    fn = ns[t["local"]]
    return [repr(fn(*args)) for args in t["inputs"]]


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from scaffold.mock_env import MultiFileEnv

    def diag(files, target, test):
        e = MultiFileEnv(files, target, test); d = e.pyrefly_diagnostics(); e.close(); return d
    def passes(files, target, test):
        e = MultiFileEnv(files, target, test); ok = e.run_tests()["resolved"]; e.close(); return ok
    def gotodef(files, target, test, sym):
        e = MultiFileEnv(files, target, test); span, path = e.goto_definition(sym); e.close()
        return span, path

    print(f"{'task':24} {'cov':5} {'mech':5} {'R1buggy':8} {'R2gold':7} {'pyfl':5} "
          f"{'Rsmall':8} {'Rnodel':7} {'Rarb':6} {'R5leak':7} {'Rdefn':14}")
    allok = True
    by_topic = {}
    for t in TASKS_COVER:
        tgt = t["target"]; sym = t["symbol"]; local = t["local"]
        by_topic.setdefault(t["topic"], {})[t["meta"]["mechanism"]] = t
        gold_all = {**t["files"], tgt: t["gold_target"]}
        full_big = t["files"]["biglib.py"]
        needed = t["needed_value"]

        # R1: buggy target FAILs the test
        r1 = not passes(t["files"], tgt, t["test"])

        # R2: gold PASSes AND is pyrefly-clean
        r2_pass = passes(gold_all, tgt, t["test"])
        nerr_gold = diag(gold_all, tgt, t["test"]).count("[error]")
        r2_clean = nerr_gold == 0

        # R-small: the gold diff vs the buggy target is a SMALL fact (<=3 changed lines), transcribable.
        gold_lines = t["gold_target"].splitlines()
        buggy_lines = t["files"]["target.py"].splitlines()
        changed = [ln for ln in gold_lines if ln not in buggy_lines]
        rsmall = len(changed) <= 3
        rsmall_str = f"{len(changed)}<=3" + ("=y" if rsmall else "=N!")

        # R-nodel: the gold neither imports a callable form of `sym` nor calls it (no forward target).
        gold_src = t["gold_target"]
        no_import = (f"from biglib import {sym}\n" not in gold_src) and \
                    (f"from biglib import {sym} " not in gold_src) and \
                    (f"import biglib" not in gold_src)
        no_call = not any((ds + "(" in gold_src) for ds in t["delegate_syms"])
        # the gold's import header (everything before the def) is byte-identical to the buggy target's,
        # i.e. the fix added NO import to forward through:
        same_header = gold_src.split("def ", 1)[0] == t["files"]["target.py"].split("def ", 1)[0]
        rnodel = no_import and no_call and same_header

        # R-arb: the idiomatic-guess value != the gold value on the pinned inputs (not name-derivable).
        wrong_all = {**t["files"], tgt: t["wrong_guess"]}
        wrong_pass = passes(wrong_all, tgt, t["test"])
        _pair = next(pp for pp in _TOPICS if pp["topic"] == t["topic"])
        _gns: dict = {}; exec(t["real_body"], _gns); _gfn = _gns[local]
        _guess_ns: dict = {}
        exec(f"def _g({_pair['arg']}):\n    {_pair['tmpl'].format(arg=_pair['arg'], C=_pair['guess_c'])}\n",
             _guess_ns)
        _guess_fn = _guess_ns["_g"]
        _gold_list = [repr(_gfn(*a)) for a in t["inputs"]]
        _guess_list = [repr(_guess_fn(*a)) for a in t["inputs"]]
        rarb = (not wrong_pass) and (t["wrong_guess"] != t["gold_target"]) and \
               any(g != w for g, w in zip(_gold_list, _guess_list))

        # R5: no-leak — strip the legitimate INPUTS line + the 64-hex hash, then assert no needed_value
        # token and no distinctive expected-output literal survives; also assert no leak in the PROMPT
        # (the buggy target.py the model sees) of the needed value.
        import re as _re5
        _resid = t["test"]
        _resid = _re5.sub(r"^INPUTS = .*$", "", _resid, flags=_re5.MULTILINE)
        _resid = _re5.sub(r"[0-9a-f]{64}", "", _resid)
        _out_lits = _gold_output_literals(t)
        _lit_leaks = [lit for lit in _out_lits if len(lit) >= 4 and lit in _resid]
        needed_in_test = needed in _resid
        # the needed value must NOT appear in the prompt the model sees (the buggy target.py); the buggy
        # constant is deliberately != gold so this holds.
        needed_in_prompt = needed in t["files"]["target.py"]
        r5 = (len(_lit_leaks) == 0) and (not needed_in_test) and (not needed_in_prompt)

        # R-defn: goto_definition(sym) over the LIVE workspace.
        span, _path = gotodef(t["files"], tgt, t["test"], sym)
        if t["meta"]["mechanism"] == "none":           # _suf: value INLINE in the defn span
            rdefn = (span is not None) and (needed in span)
            rdefn_str = "suf:has" if rdefn else "suf:MISS!"
        elif t["meta"]["mechanism"] == "f1":           # _f1ins: bare alias, value absent, names the impl
            is_alias = (span is not None) and (len(span.strip().splitlines()) == 1) and \
                       (f"{sym} = " in span) and (f"_{sym}_impl" in span) and (needed not in span)
            read_has = needed in full_big
            rdefn = is_alias and read_has
            rdefn_str = "f1:alias" if rdefn else "f1:LEAKY!"
        else:                                          # _f2ins: function body, value externalised
            body_no_value = (span is not None) and (t["files"]["biglib.py"] != "") and \
                            (needed not in span) and (t["const_name"] if False else True)
            # the span IS the function body (multi-line def) referencing the externalised const name,
            # and does NOT contain the value; the value is recoverable only by reading biglib.
            cn = _pair["const_name"]
            is_body = (span is not None) and span.strip().startswith("def ") and \
                      (cn in span) and (needed not in span) and (len(span.strip().splitlines()) > 1)
            read_has = needed in full_big
            rdefn = is_body and read_has
            rdefn_str = "f2:body" if rdefn else "f2:LEAKY!"

        ok = r1 and r2_pass and r2_clean and rsmall and rnodel and rarb and r5 and rdefn
        if not ok:
            allok = False
        print(f"{t['name']:24} "
              f"{t['meta']['coverage'][:3]:5} "
              f"{t['meta']['mechanism']:5} "
              f"{'FAIL' if r1 else 'PASS!':8} "
              f"{'PASS' if r2_pass else 'FAIL!':7} "
              f"{nerr_gold:<5} "
              f"{rsmall_str:8} "
              f"{'ok' if rnodel else 'DELEG!':7} "
              f"{'ok' if rarb else 'GUESS!':6} "
              f"{'ok' if r5 else 'LEAK!':7} "
              f"{rdefn_str:14}"
              f"{'' if ok else '  <-- PROBLEM'}")
        if not rnodel:
            print(f"     ! R-nodel: no_import={no_import} no_call={no_call} same_header={same_header}")
        if not rarb:
            print(f"     ! R-arb: wrong_pass={wrong_pass} guess==gold on inputs? "
                  f"{_gold_list == _guess_list}  ({t['wrong_note']})")
        if not r5:
            print(f"     ! R5 leak: lit_leaks={_lit_leaks} needed_in_test={needed_in_test} "
                  f"needed_in_prompt={needed_in_prompt}")
        if not rdefn:
            print(f"     ! R-defn ({t['meta']['mechanism']}): span={span!r} needed_in_span="
                  f"{(span is not None and needed in span)} needed_in_big={needed in full_big}")
        if not r2_clean:
            print(f"     ! gold not clean: {diag(gold_all, tgt, t['test']).splitlines()[:2]}")

    # R-surface: within each topic the three variants' target.py + test + symbol are BYTE-IDENTICAL.
    print("\n--- R-surface within-topic identity (the crux: the 3 variants are indistinguishable) ---")
    for topic, variants in sorted(by_topic.items()):
        suf, f1, f2 = variants.get("none"), variants.get("f1"), variants.get("f2")
        complete = suf is not None and f1 is not None and f2 is not None
        tgts = complete and (suf["files"]["target.py"] == f1["files"]["target.py"] ==
                             f2["files"]["target.py"])
        tests = complete and (suf["test"] == f1["test"] == f2["test"])
        syms = complete and (suf["symbol"] == f1["symbol"] == f2["symbol"])
        rsurf = tgts and tests and syms
        if not rsurf:
            allok = False
        print(f"  topic {topic:11} target-identical={tgts}  test-identical={tests}  "
              f"symbol-identical={syms}  {'OK' if rsurf else '<-- PROBLEM'}")

    print(f"\nALL OK ({len(TASKS_COVER)} tasks)" if allok else "PROBLEMS — fix before review")
