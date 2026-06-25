#!/usr/bin/env python3
"""COVERAGE-JUDGING suite v2 — does a coding agent decide to <read> based on the CONTENT that a <defn>
returns, when the needed value is GENUINELY read-only (no second cheap <defn> can recover it)? Sibling of
scripts/synth_tasks_cover.py (v1); same single decision ("is the value I need PRESENT in the span <defn sym>
just gave me, or must I read on?"), same small-constant / no-delegation / non-guessable / surface-invisible
design — but with v1's exploitable escape removed.

WHY v2 (the flaw in v1 this fixes):
  In v1 the two INSUFFICIENT variants were DEFN-CHAINABLE: _f1ins aliased `sym = _<sym>_impl` (so a second
  `<defn _<sym>_impl>` returned the impl body with the value inline) and _f2ins externalised the value to a
  top-level `_<SYM>_CUT = 53` Assign (so a second `<defn _<SYM>_CUT>` returned the value). A cost-trained model
  exploited this: it defn-chained EVERYTHING and almost never read, so we could not measure read-when-needed.
  v2 makes the needed value reachable ONLY by `<read>` — it lives where goto_definition cannot return it.

  HOW goto_definition (mock_env.MultiFileEnv) resolves a name (the mechanism v2 exploits): it walks `tree.body`
  (module top level only) and returns a span ONLY for a node that is a ClassDef / FunctionDef / AsyncFunctionDef
  (matched by `node.name`) OR an Assign whose TARGET is an `ast.Name` equal to the symbol. Therefore a value that
  is NOT the content of any single named top-level node is invisible to EVERY `<defn>`:
    - a value carried by a module-level CALL statement `_reg("grade_cut", 53)` is an Expr/Call node (no name) ->
      no `<defn>` returns it;
    - a value carried by a module-level Assign whose target is an Attribute `_CFG.grade_cut = 53` has target
      `ast.Attribute`, not `ast.Name` -> goto_definition's Assign branch skips it -> no `<defn>` returns it.

THE THREE VARIANTS PER TOPIC (target.py + prompt + test BYTE-IDENTICAL across the three; only biglib.py differs):
  _suf  (coverage SUFFICIENT, mechanism none): biglib has `def sym(x): return "P" if x >= 53 else "F"` — the
        constant (53) is INLINE in the body. `<defn sym>` returns it -> solvable from the defn alone, no read.
  _f1ins (INSUFFICIENT, mechanism F1 = REGISTRY CALL, NON-DEFN-CHAINABLE): biglib has
        `_REG = {}` ; a `def _reg(k, v): _REG[k] = v` ; module-level CALLS `_reg("<topic>_cut", 53)` (+ >=5 decoy
        `_reg("<other>_cut", <different N>)` scattered in filler) ; and `def sym(x): return "P" if x >= _REG["<topic>_cut"] else "F"`.
        `<defn sym>` shows the lookup `_REG["<topic>_cut"]` but NOT the value; `<defn _REG>` returns `_REG = {}`
        (empty); `<defn _reg>` returns the function (no value). The value 53 lives ONLY in the module-level CALL
        `_reg("<topic>_cut", 53)` — an Expr/Call statement, NOT a named def/assign — so NO `<defn>` returns it.
        Must `<read>` biglib and find the right `_reg(...)` among the decoys.
  _f2ins (INSUFFICIENT, mechanism F2 = ATTRIBUTE INJECTION, NON-DEFN-CHAINABLE, HELD-OUT): biglib has
        `class _Cfg: pass` ; `_CFG = _Cfg()` ; module-level attribute ASSIGNMENTS `_CFG.<topic>_cut = 53` (+ >=5
        decoy `_CFG.<other>_cut = <different N>` scattered) ; and `def sym(x): return "P" if x >= _CFG.<topic>_cut else "F"`.
        `<defn sym>` shows `_CFG.<topic>_cut`; `<defn _CFG>` returns `_CFG = _Cfg()` (no value); `<defn _Cfg>`
        returns the empty class. The value lives ONLY in the module-level Assign whose TARGET is the Attribute
        `_CFG.<topic>_cut` (an ast.Attribute, not ast.Name) — goto_definition matches Assign only by top-level
        NAME, so it returns nothing for `<topic>_cut`. Must `<read>`.

  F1 and F2 are DIFFERENT non-defn-chainable mechanisms (registry call vs attribute injection) so F2 is a genuine
  HELD-OUT: a model keyed on F1's `_REG[...]` reference style must STILL read on F2's `_CFG.attr` style; a
  content-judging model reads on both ("the value isn't in what I got back -> read").

NO-DELEGATION, GUESSABILITY, SURFACE — identical to v1 (R-nodel: target imports only the VERSION SENTINEL, `sym`
is never imported and the prompt forbids adding the import; R-arb: arbitrary 2-3 digit constants + checksum-only
tests; R-surface: the 3 variants share BYTE-IDENTICAL target.py/prompt/test). Reuses v1's topics/constants/inputs.

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
  R-defn-UNREACHABLE (the NEW v2 gate): for each INSUFFICIENT variant, drive goto_definition for EVERY name an
      agent might probe — `sym`, and the referenced names (`_REG`,`_reg`,`_CFG`,`_Cfg`, the bare topic key, and
      `<topic>_cut`) — and assert NONE of their returned spans contains needed_value, while read_file("biglib.py")
      DOES contain it. I.e. prove the value is genuinely read-only: no single `<defn>` reveals it. (For _suf,
      conversely, assert `<defn sym>` DOES contain needed_value.)
Prints "ALL OK (<n> tasks)" only when every task passes. Run (pyrefly runs SEQUENTIALLY; kill daemon first):
  pkill -9 -x pyrefly; \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/mnt/nas/hf-cache \
  .venv-streams.system/bin/python scripts/synth_tasks_cover2.py
"""

# ---- filler so biglib.py is genuinely big (a <read> must scan it; <defn> returns just one node) ----
def _filler(n, base=0):
    # All numeric literals offset to >=90000 so a 2-3 digit needed value can never appear standalone
    # in a defn-able filler span (the value must live ONLY in the read-only node -> R-defn-UNREACHABLE).
    out = []
    for i in range(base, base + n):
        j = 90000 + i
        out.append(
            f"class _Aux{j}:\n"
            f"    \"\"\"Internal helper {j} — unrelated to the task.\"\"\"\n"
            f"    def __init__(self, seed: int = {j}) -> None:\n"
            f"        self._s = seed\n"
            f"    def mix(self, x: int) -> int:\n"
            f"        return (x * {j + 3}) ^ self._s\n"
            f"    def label(self) -> str:\n"
            f"        return f\"aux{j}:{{self._s}}\"\n")
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
# (Topics/constants/inputs are reused verbatim from v1 — only the two insufficient biglib emitters change.)
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
# inline copy exists, and the reason there is no symbol to delegate to at the fix site. (Verbatim from v1.)
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
# type-checks); it is NOT the answer. The symbol `sym` is then defined differently per variant. The two
# INSUFFICIENT variants carry the value in a node goto_definition cannot return (NON-DEFN-CHAINABLE).
# ---------------------------------------------------------------------------------------------------
def _ver_line(p):
    return f"{_ver_name(p)} = 1\n"


# Decoy deltas: the decoy registry entries / attributes use DIFFERENT (wrong) values so the real one is
# not the only candidate a reader sees — they must resolve WHICH key/attr `sym` actually uses.
_DECOY_DELTAS = [11, -7, 23, -3, 41, -17, 5]


def _key(p):
    """The registry key / attribute name `sym` actually consults, e.g. grade -> 'grade_cut'."""
    return f"{p['topic']}_cut"


def _decoy_keys(p):
    """>=5 decoy key names (a DIFFERENT topic family) so the real `_<topic>_cut` is buried among siblings."""
    others = [tt["topic"] for tt in _TOPICS if tt["topic"] != p["topic"]]
    return [f"{o}_cut" for o in others[:5]]


def _biglib_suf(p):
    """VARIANT _suf: `sym` is a real function with the constant INLINE in the body -> <defn sym> returns
    the body and the value is right there. (Identical mechanism to v1's _suf.)"""
    body = p["tmpl"].format(arg="x", C=p["gold_c"])
    fn = (f"def {p['sym']}(x):\n"
          f"    \"\"\"Authoritative {p['sym']} (constant inline).\"\"\"\n"
          f"    {body}\n")
    return _ver_line(p) + "\n" + _scatter([fn])


def _biglib_f1(p):
    """VARIANT _f1ins, mechanism F1 = REGISTRY CALL (non-defn-chainable). biglib holds an EMPTY registry
    `_REG = {}`, a registrar `def _reg(k, v): _REG[k] = v`, module-level CALLS `_reg("<key>", <val>)` (the
    real one carrying the gold constant, among >=5 decoys with different keys/values), and a real `sym`
    that does `return ... _REG["<key>"] ...`. The value lives ONLY in the module-level CALL statement (an
    Expr/Call node, NOT a named def/assign), so NO `<defn>` returns it: `<defn sym>` shows the LOOKUP but
    not the value; `<defn _REG>` returns the empty dict; `<defn _reg>` returns the registrar function. Must
    <read> to find the right `_reg(...)` among the decoys."""
    key = _key(p)
    reg_decl = "_REG = {}"
    reg_fn = ("def _reg(k, v):\n"
              "    \"\"\"Register a config value (module-init side effect; value lives in the CALL site).\"\"\"\n"
              "    _REG[k] = v\n")
    # decoy registry CALLS (different keys + different values) — scattered so the real one is buried.
    decoy_calls = []
    for dk, d in zip(_decoy_keys(p), _DECOY_DELTAS[:5]):
        decoy_calls.append(f'_reg("{dk}", {p["gold_c"] + d})')
    real_call = f'_reg("{key}", {p["gold_c"]})'
    body = p["tmpl"].format(arg="x", C=f'_REG["{key}"]')
    fn = (f"def {p['sym']}(x):\n"
          f"    \"\"\"Authoritative {p['sym']} (cutoff in the _REG registry under \\\"{key}\\\").\"\"\"\n"
          f"    {body}\n")
    # _REG decl + registrar + (decoys, real call interleaved) + sym, all buried; real call not first/last.
    blocks = [reg_decl, reg_fn] + decoy_calls[:2] + [real_call] + decoy_calls[2:] + [fn]
    return _ver_line(p) + "\n" + _scatter(blocks, gaps=(12, 10, 9, 11, 10, 9, 12, 11))


def _biglib_f2(p):
    """VARIANT _f2ins, mechanism F2 = ATTRIBUTE INJECTION (non-defn-chainable, HELD-OUT). biglib holds an
    empty `class _Cfg: pass`, an instance `_CFG = _Cfg()`, module-level ATTRIBUTE assignments
    `_CFG.<key> = <val>` (the real one carrying the gold constant, among >=5 decoys with different attrs/
    values), and a real `sym` that does `return ... _CFG.<key> ...`. The value lives ONLY in the module-
    level Assign whose TARGET is the Attribute `_CFG.<key>` (an ast.Attribute, not ast.Name), which
    goto_definition's Assign branch skips — so NO `<defn>` returns it: `<defn sym>` shows `_CFG.<key>`;
    `<defn _CFG>` returns `_CFG = _Cfg()` (no value); `<defn _Cfg>` returns the empty class. Must <read>."""
    key = _key(p)
    # `__setattr__` / `__getattr__` (-> int) make the dynamically-injected attributes type-clean for pyrefly
    # (which can't see the runtime `_CFG.x = N` injections statically): __setattr__ clears the assign-site
    # `missing-attribute`, __getattr__ clears the read site. Both dunder bodies carry NO value, so the class
    # span `<defn _Cfg>` still reveals nothing — the value lives only in the `_CFG.<key> = N` Assign-to-Attribute.
    cls = ("class _Cfg:\n"
           "    \"\"\"Config holder; attributes injected at module init below.\"\"\"\n"
           "    def __setattr__(self, name: str, value: int) -> None:\n"
           "        object.__setattr__(self, name, value)\n"
           "    def __getattr__(self, name: str) -> int:\n"
           "        raise AttributeError(name)\n")
    inst = "_CFG = _Cfg()"
    decoy_attrs = []
    for dk, d in zip(_decoy_keys(p), _DECOY_DELTAS[:5]):
        decoy_attrs.append(f"_CFG.{dk} = {p['gold_c'] + d}")
    real_attr = f"_CFG.{key} = {p['gold_c']}"
    body = p["tmpl"].format(arg="x", C=f"_CFG.{key}")
    fn = (f"def {p['sym']}(x):\n"
          f"    \"\"\"Authoritative {p['sym']} (cutoff injected as _CFG.{key}).\"\"\"\n"
          f"    {body}\n")
    # class + instance + (decoys, real attr interleaved) + sym, all buried; real attr not first/last.
    blocks = [cls, inst] + decoy_attrs[:2] + [real_attr] + decoy_attrs[2:] + [fn]
    return _ver_line(p) + "\n" + _scatter(blocks, gaps=(12, 10, 9, 11, 10, 9, 12, 11))


TASKS_COVER2 = []
for _p in _TOPICS:
    _buggy = _emit_buggy(_p)
    _gold = _emit_gold(_p)
    _wrong = _emit_wrong(_p)
    _test = _emit_test(_p)
    _real_body = _gold_body_src(_p)
    _needed = str(_p["gold_c"])          # the distinctive arbitrary constant the fix must transcribe
    _delegate = [_p["sym"]]              # a forwarding fix COULD try to call `sym`; the gold must not

    # _suf — coverage sufficient (defn has the value inline)
    TASKS_COVER2.append(dict(
        name=f"cover2_{_p['topic']}_suf", topic=_p["topic"], group="rich", target="target.py",
        symbol=_p["sym"], meta={"coverage": "sufficient", "mechanism": "none"},
        defn_sufficient=True, requires_read=False,
        files={"target.py": _buggy, "biglib.py": _biglib_suf(_p)},
        test=_test, gold_target=_gold, inputs=_p["inputs"], real_body=_real_body, local=_p["local"],
        wrong_guess=_wrong, wrong_kind="value",
        wrong_note=f"idiomatic {_p['guess_c']} (real: {_p['gold_c']}) -> type-clean, value-wrong",
        needed_value=_needed, delegate_syms=_delegate))

    # _f1ins — insufficient, F1 registry call (value lives ONLY in a module-level _reg(...) call)
    TASKS_COVER2.append(dict(
        name=f"cover2_{_p['topic']}_f1ins", topic=_p["topic"], group="rich", target="target.py",
        symbol=_p["sym"], meta={"coverage": "insufficient", "mechanism": "f1"},
        defn_sufficient=False, requires_read=True,
        files={"target.py": _buggy, "biglib.py": _biglib_f1(_p)},
        test=_test, gold_target=_gold, inputs=_p["inputs"], real_body=_real_body, local=_p["local"],
        wrong_guess=_wrong, wrong_kind="value",
        wrong_note=f"idiomatic {_p['guess_c']} (real: {_p['gold_c']}) -> type-clean, value-wrong",
        needed_value=_needed, delegate_syms=_delegate))

    # _f2ins — insufficient, F2 attribute injection (value lives ONLY in a _CFG.<key> = N attribute assign)
    TASKS_COVER2.append(dict(
        name=f"cover2_{_p['topic']}_f2ins", topic=_p["topic"], group="rich", target="target.py",
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


# The full set of names an agent might probe with <defn> on an insufficient variant — the symbol itself
# plus every name referenced in the biglib mechanism. R-defn-UNREACHABLE asserts NONE of these reveal the
# value via goto_definition (while read_file does). Built per-task from the topic key.
def _probe_names(t):
    p = next(pp for pp in _TOPICS if pp["topic"] == t["topic"])
    key = _key(p)
    return [t["symbol"], "_REG", "_reg", "_CFG", "_Cfg", key, t["topic"]]


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

    print(f"{'task':26} {'cov':5} {'mech':5} {'R1buggy':8} {'R2gold':7} {'pyfl':5} "
          f"{'Rsmall':8} {'Rnodel':7} {'Rarb':6} {'R5leak':7} {'Rdefn-UNREACH':16}")
    allok = True
    by_topic = {}
    for t in TASKS_COVER2:
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
        # token and no distinctive expected-output literal survives; also assert no leak in the PROMPT.
        import re as _re5
        _resid = t["test"]
        _resid = _re5.sub(r"^INPUTS = .*$", "", _resid, flags=_re5.MULTILINE)
        _resid = _re5.sub(r"[0-9a-f]{64}", "", _resid)
        _out_lits = _gold_output_literals(t)
        _lit_leaks = [lit for lit in _out_lits if len(lit) >= 4 and lit in _resid]
        needed_in_test = needed in _resid
        needed_in_prompt = needed in t["files"]["target.py"]
        r5 = (len(_lit_leaks) == 0) and (not needed_in_test) and (not needed_in_prompt)

        # R-defn-UNREACHABLE (the NEW v2 gate): drive goto_definition over the LIVE workspace.
        read_has = needed in full_big   # the value IS in biglib (so <read> recovers it)
        if t["meta"]["mechanism"] == "none":           # _suf: value INLINE in the <defn sym> span
            span, _ = gotodef(t["files"], tgt, t["test"], sym)
            rdefn = (span is not None) and (needed in span) and read_has
            rdefn_str = "suf:has" if rdefn else "suf:MISS!"
            probe_report = ""
        else:                                          # insufficient: NO probe-name's defn reveals the value
            leaky = []
            for nm in _probe_names(t):
                span, _ = gotodef(t["files"], tgt, t["test"], nm)
                if span is not None and needed in span:
                    leaky.append(nm)
            rdefn = (len(leaky) == 0) and read_has
            mech = t["meta"]["mechanism"]
            rdefn_str = (f"{mech}:read-only" if rdefn else f"{mech}:LEAKY!")
            probe_report = f"leaky={leaky}"

        ok = r1 and r2_pass and r2_clean and rsmall and rnodel and rarb and r5 and rdefn
        if not ok:
            allok = False
        print(f"{t['name']:26} "
              f"{t['meta']['coverage'][:3]:5} "
              f"{t['meta']['mechanism']:5} "
              f"{'FAIL' if r1 else 'PASS!':8} "
              f"{'PASS' if r2_pass else 'FAIL!':7} "
              f"{nerr_gold:<5} "
              f"{rsmall_str:8} "
              f"{'ok' if rnodel else 'DELEG!':7} "
              f"{'ok' if rarb else 'GUESS!':6} "
              f"{'ok' if r5 else 'LEAK!':7} "
              f"{rdefn_str:16}"
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
            print(f"     ! R-defn-UNREACHABLE ({t['meta']['mechanism']}): {probe_report} "
                  f"read_has={read_has} (value must be defn-invisible but read-recoverable)")
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

    print(f"\nALL OK ({len(TASKS_COVER2)} tasks)" if allok else "PROBLEMS — fix before review")
