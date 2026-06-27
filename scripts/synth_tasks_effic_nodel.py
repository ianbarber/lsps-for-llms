#!/usr/bin/env python3
"""NODEL suite — does the agent JUDGE retrieval-coverage by USING the retrieved BODY CONTENT, with the
DELEGATION escape hatch removed? Sibling of synth_tasks_effic_decouple.py (the DECOUPLE suite), built to
fix that suite's fatal confound and re-run the SAME coverage A/B experiment cleanly.

WHY A NEW SUITE (the confound we are killing). The DECOUPLE suite's gold fix was DELEGATION —
`return combine(a, b)`: the target just forwards to the imported symbol. That works whether `<defn sym>`
returned the FULL body (A) or the opaque stub (B), because the agent never needed the body's CONTENT,
only a callable to forward to. Both variants export a working callable, so A and B were behaviourally
EQUIVALENT for a delegating agent and the coverage test was destroyed. The fix here: the correct edit
must TRANSCRIBE the CONTENT of the retrieved definition INLINE at the bug site, and there is NO in-scope
callable (and no usable data structure) to forward to.

THE DESIGN — "denormalised inlined table, completeness/transcription". Each pair_id has TWO variants
whose prompt / bug / function-to-fix / test are the SAME TEMPLATE (only the topic noun differs ACROSS
pairs, never WITHIN a pair between A and B). Nothing in the task text tells the agent which case it is.

  target.py holds its OWN, LOCAL, INLINED copy of a lookup written as explicit `if key == ...: return
  <const>` branches (a denormalised hot-path copy — REALISTIC: the docstring says the table is inlined
  here for the hot path and biglib is deliberately NOT imported at call time to avoid a heavy/circular
  import). The local copy is BUGGY: it is MISSING cases and/or has WRONG arbitrary constants. The
  authoritative reference lives in biglib.py as the DATA symbol `SPEC` (a flat denormalised array, see
  below). To fix, the agent must inspect `SPEC`, read off the per-key arbitrary constants, and write the
  missing/correct branches INLINE.

  VARIANT A (defn_sufficient=True):  `SPEC` is a FULL top-level Assign with all literal entries.
    goto_definition("SPEC") (mock_env.py:198-200 Assign branch) returns the WHOLE multi-line literal, so
    the per-key constants are right there in the span and the fix is derivable from the defn alone.

  VARIANT B (defn_sufficient=False, requires_read=True): `SPEC` is a top-level ONE-LINE alias
    `SPEC = _TBL[pos]` into a SHUFFLED table of decoy arrays. goto_definition("SPEC") returns ONLY that
    one opaque line — it names no `_SPEC_*` directly (the subscript position != the impl's name suffix),
    so a single follow-up `<defn>` cannot resolve it. The real array is one of >=6 similarly-named decoy
    arrays (each a DIFFERENT arbitrary table); it is identifiable only by READING biglib.py. The agent
    must: <defn SPEC> -> see it is an opaque alias -> <read biglib.py> -> find the real array -> transcribe.

  A coverage-judging agent solves BOTH; a surface-keyed / delegating agent fails B (it has no callable to
  forward to and the stub array reveals nothing, so it emits the idiomatic-but-WRONG table and fails).

WHY THE DELEGATION ESCAPE IS GONE (the whole point — enumerated in the module docstring; the verifier
also asserts R-nodel):
  - There is NO callable in biglib to forward to. biglib exposes only DATA (`SPEC` and the decoy arrays)
    plus unrelated filler classes. `from biglib import SPEC; return SPEC[k]` does NOT type-check or work:
    SPEC is a FLAT array `(k0, v0, k1, v1, ...)` (a denormalised hot-path layout), not a mapping — there
    is no key-indexed lookup to forward to. Reconstructing the answer from it requires writing the same
    branch logic anyway, i.e. transcribing the content.
  - The target deliberately does NOT import SPEC (REALISTIC hot-path / no-startup-import reason in the
    docstring), and the gold fix adds NO import — it INLINES the constants. The verifier's R-nodel gate
    asserts the gold diff introduces NO call to any biglib symbol and adds no `from biglib import` of a
    data/callable used as a forward target; the new content is literal constants transcribed from SPEC.

GUESS-RESISTANCE (kills confound #1): every per-key value is an ARBITRARY constant (e.g. an affine
i*9-4 of an opaque code, a 3-digit checksum, a reversed-with-offset string) NOT inferable from the key
name, the function name, or the docstring. The idiomatic guess (identity, +1, the key itself) is
type-valid but TEST-REJECTED. The behavioural test pins the EXACT arbitrary values via a checksum
(sha256 of the joined repr()s over a fixed INPUT set), so a FAILING run reveals nothing (opaque hash) and
'guess then fit to the revealed expected output' cannot converge — the only way to pass is to READ the
constants. The digest is computed IN-PROCESS from the gold body at construction time (never hand-written).

SCHEMA (per task dict) — a SUPERSET of synth_tasks_effic.py's schema plus the decouple/nodel keys:
  name (effic_nd_<topic>_a | _b), pair_id, group("rich"), target("target.py"),
  symbol ("SPEC" — the DATA symbol the agent inspects; SAME string in A and B of a pair),
  defn_sufficient (bool), requires_read (bool; True only on B),
  files{target.py(buggy inlined table), biglib.py(SPEC + decoys)}, test(checksum, passes on gold),
  gold_target(corrected inlined table), inputs, real_body (the gold target body used for the digest),
  symbol_defns={"defn": {SPEC: <what goto_definition returns>}, ...},
  wrong_guess (idiomatic-wrong gold splice — proves non-guessability), wrong_kind("value"),
  wrong_note, real_api (the substring the FIX must transcribe — a distinctive arbitrary constant/branch;
  the verifier asserts A's defn CONTAINS it and B's defn-stub does NOT but the full read DOES),
  local (the public function name in target.py the test imports),
  delegate_syms (the set of biglib symbols a delegating fix COULD try to forward to — verifier asserts
  the gold introduces NO call to any of them: R-nodel).

VERIFIER (__main__): per task assert (mirrors effic R1-R5 + decouple R6/R7 + the new R-nodel / R-arb):
  R1 buggy target FAILs the test; R2 gold PASSes AND is pyrefly-clean (nerr==0);
  R3 biglib big (defn_lines*5 < biglib_n for A; >=200 lines for B);
  R4 NON-GUESSABLE (idiomatic wrong_guess spliced into gold => test FAIL) AND hash-unique (none of the
     decoy arrays reproduces the gold checksum);
  R5 no leak — no expected-output literal and no API-surface token in the test (only INPUTS + the hash);
  R6 DECOUPLING — goto_definition(SPEC): A's span CONTAINS real_api; B's span is a one-line opaque alias
     that does NOT contain real_api (and names no `_SPEC_`), while the full read DOES contain it;
  R7 PAIR-SURFACE-IDENTITY — A and B target.py + test + symbol are byte-identical (no prompt cue);
  R-nodel (NEW) — the gold diff vs the buggy target adds NO call to any delegate_sym and adds no
     `from biglib import` line: the fix is pure inline transcription, with no content-free forward;
  R-arb (NEW) — the pinned test values are arbitrary: for each input the gold output differs from the
     idiomatic/derivable-from-name guess output, so the values cannot be reconstructed from names.
Prints "ALL OK (<n> tasks)" only when every task passes. Run (pyrefly runs SEQUENTIALLY; kill daemon first):
  pkill -9 -f pyrefly; \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/mnt/nas/hf-cache \
  .venv-streams.system/bin/python scripts/synth_tasks_effic_nodel.py
"""

# ---- filler (verbatim style from synth_tasks_effic_decouple.py:82-94) ----------------------------
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


def _scatter(blocks, gaps=(14, 12, 12, 12, 14)):
    """Interleave real/decoy `blocks` between filler bands so every symbol is buried deep in a big file
    — a <read> must scan it, while <defn> on a known name returns just the one node."""
    parts = []
    base = 0
    for i, blk in enumerate(blocks):
        g = gaps[i % len(gaps)]
        parts.append(_filler(g, base)); base += g
        parts.append(blk)
    parts.append(_filler(gaps[-1], base))
    return "\n\n".join(parts)


# ==================================================================================================
# Each pair is a denormalised KEY -> ARBITRARY-CONSTANT table that target.py re-implements INLINE as
# explicit branches. The reference lives in biglib.py as the DATA symbol `SPEC`, a FLAT array
# (k0, v0, k1, v1, ...) — a denormalised hot-path layout, NOT a mapping (so there is no callable and no
# clean SPEC[k] lookup to delegate to). The buggy target is MISSING some keys and/or has WRONG constants;
# the gold transcribes the full correct table inline. The constants are arbitrary -> non-guessable.
#
# Per-pair fields:
#   topic, local (public fn name), arg name, key list (the dispatch keys, as python literals),
#   gold values (arbitrary), an idiomatic-but-WRONG value rule (for wrong_guess / R-arb), the buggy
#   subset/perturbation, inputs (a fixed call set), and a distinctive `real_api` substring the fix needs.
# ==================================================================================================


def _make_dispatch(argname, keys, vals, indent="    "):
    """Render an inlined if/elif dispatch over string `keys` returning python literal `vals`, with a
    final `raise KeyError(...)`. This is the form both the buggy target and the gold target take."""
    out = []
    for i, (k, v) in enumerate(zip(keys, vals)):
        kw = "if" if i == 0 else "elif"
        out.append(f"{indent}{kw} {argname} == {k!r}:\n{indent}    return {v!r}")
    out.append(f"{indent}raise KeyError({argname})")
    return "\n".join(out)


# Deterministic scramble of the 5 entries so the values-only tuple is NOT in key order (defeats any
# positional guess) yet is stable across A and the real decoy in B.
_SCRAMBLE = [2, 4, 0, 3, 1]


def _flat_spec(keys, vals):
    """biglib SPEC as a VALUES-ONLY tuple in SCRAMBLED order, with the owning key on each line as an
    INLINE COMMENT. This is the denormalised reference the agent must READ and TRANSCRIBE:
      - At RUNTIME the value is a BARE tuple of the values only (no keys) in scrambled order, so
        `from biglib import SPEC` yields NO key->value mapping — the data-delegation attack
        `dict(zip(SPEC[0::2], SPEC[1::2]))[key]` cannot reconstruct the answer (there are no key labels
        and the order is scrambled). The key->value correspondence exists ONLY in the source comments.
      - In the SOURCE SPAN that <defn SPEC>/<read> return, each line carries `# <key>`, so a reader can
        recover the mapping and transcribe the per-key constants INLINE. The comments are stripped by the
        interpreter, so the mapping is recoverable ONLY by reading the definition text — not by
        introspecting the runtime object. This blocks both DELEGATION and INTROSPECTION on this channel."""
    n = len(keys)
    order = [i % n for i in _SCRAMBLE[:n]] if n <= len(_SCRAMBLE) else list(range(n))
    # ensure a valid permutation even if n != len(_SCRAMBLE)
    if sorted(order) != list(range(n)):
        order = list(range(n))
    lines = []
    for i in order:
        lines.append(f"    {vals[i]!r},   # {keys[i]}")
    return "(\n" + "\n".join(lines) + "\n)"


# ---------------------------------------------------------------------------------------------------
# PAIR 1 — "status" : map a status code string to an ARBITRARY numeric token. The real value for key
# 'k' is an arbitrary affine of an opaque per-key code (NOT the key, NOT an index). Buggy target omits
# two keys and gets one constant wrong. Idiomatic guess (len of key, or 0) is value-wrong.
# ---------------------------------------------------------------------------------------------------
_P1_KEYS = ["red", "amber", "green", "blue", "violet"]
_P1_VALS = [317, 158, 904, 542, 671]          # arbitrary 3-digit tokens, not derivable from the key
_P1_BUGVALS = [317, 158, 904]                  # buggy: only first three keys, missing 'blue'/'violet'
_P1_BUGKEYS = ["red", "amber", "green"]
_P1_WRONGVALS = [3, 5, 5, 4, 6]                # idiomatic guess: len(key) -> value-wrong & type-clean
_P1_REAL_API = "542"                           # 'blue' -> 542, a distinctive arbitrary value in SPEC


# ---------------------------------------------------------------------------------------------------
# PAIR 2 — "region" : map a region tag to an ARBITRARY string label (reversed key + arbitrary suffix).
# Buggy target has the WRONG suffix on every entry and is missing one key. Idiomatic guess (the key
# itself, or upper()) is value-wrong.
# ---------------------------------------------------------------------------------------------------
_P2_KEYS = ["nw", "ne", "sw", "se", "mid"]
_P2_VALS = ["wn#7", "en#7", "ws#7", "es#7", "dim#7"]   # arbitrary: reversed key + "#7"
_P2_BUGVALS = ["wn#3", "en#3", "ws#3", "es#3"]          # buggy: wrong suffix '#3', missing 'mid'
_P2_BUGKEYS = ["nw", "ne", "sw", "se"]
_P2_WRONGVALS = ["NW", "NE", "SW", "SE", "MID"]         # idiomatic guess: key.upper() -> value-wrong
_P2_REAL_API = "'dim#7'"                                 # 'mid' -> 'dim#7' (distinctive value in SPEC)


# ---------------------------------------------------------------------------------------------------
# PAIR 3 — "priority" : map a level name to an ARBITRARY weight (a non-monotone arbitrary integer set).
# Buggy target is missing two keys and has one wrong weight. Idiomatic guess (rank 0..n) is value-wrong.
# ---------------------------------------------------------------------------------------------------
_P3_KEYS = ["low", "med", "high", "crit", "fatal"]
_P3_VALS = [22, 91, 13, 784, 46]               # arbitrary non-monotone weights, NOT a 0..n rank
_P3_BUGVALS = [22, 91, 13]                      # buggy: missing 'crit'/'fatal'
_P3_BUGKEYS = ["low", "med", "high"]
_P3_WRONGVALS = [0, 1, 2, 3, 4]                 # idiomatic guess: positional rank -> value-wrong
_P3_REAL_API = "784"                            # 'crit' -> 784 (distinctive arbitrary value in SPEC)


# ---------------------------------------------------------------------------------------------------
# PAIR 4 — "currency" : map a 3-letter currency code to an ARBITRARY integer minor-unit factor that is
# NOT the usual 100 (deliberately un-idiomatic). Buggy target uses the idiomatic 100 everywhere and is
# missing one key. Idiomatic guess (100) is value-wrong.
# ---------------------------------------------------------------------------------------------------
_P4_KEYS = ["usd", "jpy", "kwd", "btc", "xau"]
_P4_VALS = [1000, 1, 1000, 100000000, 31]      # arbitrary factors (un-idiomatic: usd=1000 not 100)
_P4_BUGVALS = [100, 100, 100, 100]              # buggy: idiomatic 100 everywhere, missing 'xau'
_P4_BUGKEYS = ["usd", "jpy", "kwd", "btc"]
_P4_WRONGVALS = [100, 100, 100, 100, 100]       # idiomatic guess: 100 -> value-wrong
_P4_REAL_API = "100000000"                     # 'btc' -> 1e8 (distinctive value in SPEC)


# ---------------------------------------------------------------------------------------------------
# PAIR 5 — "shortcut" : map an action name to an ARBITRARY key-chord string. Buggy target is missing
# two actions and one chord is wrong. Idiomatic guess (first letter, or 'Ctrl+'+first) is value-wrong.
# ---------------------------------------------------------------------------------------------------
_P5_KEYS = ["save", "open", "quit", "find", "undo"]
_P5_VALS = ["M-9", "C-x7", "K-2", "F-q", "Z-0"]    # arbitrary chords, NOT 'Ctrl+S' etc.
_P5_BUGVALS = ["M-9", "C-x7", "K-9"]                # buggy: 'quit' wrong ('K-9' vs 'K-2'), missing find/undo
_P5_BUGKEYS = ["save", "open", "quit"]
_P5_WRONGVALS = ["Ctrl+S", "Ctrl+O", "Ctrl+Q", "Ctrl+F", "Ctrl+Z"]  # idiomatic -> value-wrong
_P5_REAL_API = "'F-q'"                               # 'find' -> 'F-q' (distinctive value in SPEC)


# ---------------------------------------------------------------------------------------------------
# PAIR 6 — "opcode" : map a mnemonic to an ARBITRARY opcode integer. Buggy target is missing two
# mnemonics and one opcode is wrong. Idiomatic guess (hash/ord of first char) is value-wrong.
# ---------------------------------------------------------------------------------------------------
_P6_KEYS = ["nop", "add", "sub", "jmp", "ret"]
_P6_VALS = [0x90, 0x2B, 0x55, 0xE9, 0xC3]      # arbitrary opcodes (x86-ish but un-guessable mapping)
_P6_BUGVALS = [0x90, 0x2B, 0x99]                # buggy: 'sub' wrong (0x99 vs 0x55), missing jmp/ret
_P6_BUGKEYS = ["nop", "add", "sub"]
_P6_WRONGVALS = [0, 1, 2, 3, 4]                 # idiomatic guess: positional -> value-wrong
_P6_REAL_API = "233"                            # 'jmp' -> 0xE9 == 233 (repr of the int literal in SPEC)


# ==================================================================================================
# Per-pair registry. `_emit_*` build the buggy target, gold target, wrong-guess target, test, and the
# two biglib variants from these so A and B share an identical SURFACE (everything the model sees).
# ==================================================================================================
_PAIRS = [
    dict(pair_id="status", local="status_token", arg="code",
         keys=_P1_KEYS, vals=_P1_VALS, bugkeys=_P1_BUGKEYS, bugvals=_P1_BUGVALS,
         wrongvals=_P1_WRONGVALS, real_api=_P1_REAL_API,
         ret="int", argann="str",
         prose='    """Return the numeric token for status `code` (must match biglib.SPEC)."""',
         inputs=[("red",), ("amber",), ("green",), ("blue",), ("violet",)],
         wrong_note="idiomatic len(code) (real: arbitrary 3-digit tokens) -> type-clean, wrong -> VALUE"),

    dict(pair_id="region", local="region_label", arg="tag",
         keys=_P2_KEYS, vals=_P2_VALS, bugkeys=_P2_BUGKEYS, bugvals=_P2_BUGVALS,
         wrongvals=_P2_WRONGVALS, real_api=_P2_REAL_API,
         ret="str", argann="str",
         prose='    """Return the label for region `tag` (must match biglib.SPEC)."""',
         inputs=[("nw",), ("ne",), ("sw",), ("se",), ("mid",)],
         wrong_note="idiomatic tag.upper() (real: reversed key + '#7') -> type-clean, wrong -> VALUE"),

    dict(pair_id="priority", local="priority_weight", arg="level",
         keys=_P3_KEYS, vals=_P3_VALS, bugkeys=_P3_BUGKEYS, bugvals=_P3_BUGVALS,
         wrongvals=_P3_WRONGVALS, real_api=_P3_REAL_API,
         ret="int", argann="str",
         prose='    """Return the weight for priority `level` (must match biglib.SPEC)."""',
         inputs=[("low",), ("med",), ("high",), ("crit",), ("fatal",)],
         wrong_note="idiomatic positional rank 0..n (real: arbitrary non-monotone) -> type-clean -> VALUE"),

    dict(pair_id="currency", local="minor_units", arg="ccy",
         keys=_P4_KEYS, vals=_P4_VALS, bugkeys=_P4_BUGKEYS, bugvals=_P4_BUGVALS,
         wrongvals=_P4_WRONGVALS, real_api=_P4_REAL_API,
         ret="int", argann="str",
         prose='    """Return the minor-unit factor for currency `ccy` (must match biglib.SPEC)."""',
         inputs=[("usd",), ("jpy",), ("kwd",), ("btc",), ("xau",)],
         wrong_note="idiomatic 100 everywhere (real: arbitrary per-ccy factors) -> type-clean -> VALUE"),

    dict(pair_id="shortcut", local="chord_for", arg="action",
         keys=_P5_KEYS, vals=_P5_VALS, bugkeys=_P5_BUGKEYS, bugvals=_P5_BUGVALS,
         wrongvals=_P5_WRONGVALS, real_api=_P5_REAL_API,
         ret="str", argann="str",
         prose='    """Return the key-chord for `action` (must match biglib.SPEC)."""',
         inputs=[("save",), ("open",), ("quit",), ("find",), ("undo",)],
         wrong_note="idiomatic Ctrl+<X> (real: arbitrary chords) -> type-clean, wrong -> VALUE"),

    dict(pair_id="opcode", local="opcode_for", arg="mnem",
         keys=_P6_KEYS, vals=_P6_VALS, bugkeys=_P6_BUGKEYS, bugvals=_P6_BUGVALS,
         wrongvals=_P6_WRONGVALS, real_api=_P6_REAL_API,
         ret="int", argann="str",
         prose='    """Return the opcode for mnemonic `mnem` (must match biglib.SPEC)."""',
         inputs=[("nop",), ("add",), ("sub",), ("jmp",), ("ret",)],
         wrong_note="idiomatic positional (real: arbitrary opcodes) -> type-clean, wrong -> VALUE"),
]


# The target file imports ONLY a version sentinel from biglib (NOT SPEC) — the realistic reason the
# inlined table exists and the reason there is no symbol to delegate to at the fix site. The docstring
# states the inlined table must match biglib.SPEC for this version and is denormalised here for the hot
# path (biglib is heavy / would cause a circular import if pulled in at call time).
_HEADER = (
    "from biglib import SPEC_VERSION  # version sentinel only; SPEC itself is NOT imported (see below)\n\n"
)
_DOC_TAIL = (
    "    # NOTE: this table is a denormalised INLINE copy of biglib.SPEC, kept here for the hot path —\n"
    "    # biglib is intentionally not imported at call time (heavy module / would be a circular import),\n"
    "    # so the entries below must be transcribed to match biglib.SPEC for SPEC_VERSION.\n"
    "    assert SPEC_VERSION >= 1\n"
)


def _emit_target(p, keys, vals):
    return (
        f"{_HEADER}"
        f"def {p['local']}({p['arg']}: {p['argann']}) -> {p['ret']}:\n"
        f"{p['prose']}\n"
        f"{_DOC_TAIL}"
        f"{_make_dispatch(p['arg'], keys, vals)}\n"
    )


def _emit_buggy(p):
    return _emit_target(p, p["bugkeys"], p["bugvals"])


def _emit_gold(p):
    return _emit_target(p, p["keys"], p["vals"])


def _emit_wrong(p):
    # idiomatic-but-WRONG: full key set (so it is not "incomplete"), but the idiomatic value rule.
    return _emit_target(p, p["keys"], p["wrongvals"])


def _gold_body_src(p):
    """The gold target function as a BARE def (no `from biglib import` header) so it can be exec'd
    in-process to derive the digest / expected outputs without importing biglib. `real_body` uses this;
    the full file the agent edits is `gold_target` (which DOES carry the SPEC_VERSION import header)."""
    return (f"def {p['local']}({p['arg']}):\n"
            f"{_make_dispatch(p['arg'], p['keys'], p['vals'])}\n")


def _gold_fn(p):
    """Build the gold target function in-process so the test digest is derived from it (no drift)."""
    ns: dict = {}
    exec(_gold_body_src(p), ns)
    return ns[p["local"]]


def _gold_hash(p):
    import hashlib
    fn = _gold_fn(p)
    got = "|".join(repr(fn(*args)) for args in p["inputs"])
    return hashlib.sha256(got.encode()).hexdigest()


def _emit_test(p):
    """CHECKSUM test (no expected-output literal): call the function under test over a fixed INPUT set
    and assert sha256(joined repr()s) == precomputed gold digest. A failing run reveals only an opaque
    hash mismatch, so 'guess then fit to the revealed expected output' cannot converge — combined with
    the arbitrary constants, the ONLY way to pass is to READ the table. Digest derived in-process."""
    inputs_repr = repr(p["inputs"])
    return (f"from target import {p['local']}\n"
            "import hashlib\n"
            f"INPUTS = {inputs_repr}\n"
            f"got = \"|\".join(repr({p['local']}(*args)) for args in INPUTS)\n"
            f"assert hashlib.sha256(got.encode()).hexdigest() == \"{_gold_hash(p)}\", \"wrong\"\n")


# The version sentinel the target imports (the ONLY biglib name target.py pulls in). Defined at the top
# of biglib in BOTH variants so the import type-checks; it is NOT the answer (just a version gate).
_VERSION_LINE = "SPEC_VERSION = 1\n"


# biglib SPEC as a flat denormalised array; in A it is a full literal Assign, in B an opaque alias.
def _biglib_a(p):
    """VARIANT A biglib: SPEC is a FULL top-level Assign with all literal entries -> <defn SPEC>
    returns the whole table; the per-key constants are right there."""
    spec = f"SPEC = {_flat_spec(p['keys'], p['vals'])}\n"
    return _VERSION_LINE + "\n" + _scatter([spec], gaps=(16, 12, 14, 16))


# B: shuffled table of decoy arrays + opaque subscript alias. Each `_SPEC_k` is a DIFFERENT arbitrary
# array; the real one is at index impl_idx. `_TBL[pos] == _SPEC_{_PERM[pos]}`, so SPEC = _TBL[pos] with
# pos = _PERM.index(impl_idx). The alias names no `_SPEC_` -> no single follow-up <defn> resolves it.
_PERM = [3, 6, 1, 4, 0, 5, 2]


def _decoy_arrays(p):
    """7 decoy flat arrays (incl. the real one at index 6). Each is a DIFFERENT arbitrary table over the
    SAME keys so none reproduces the gold checksum; the real one (index 6) is byte-equal to SPEC_A."""
    keys = p["keys"]
    base = p["vals"]
    # build 6 perturbed decoys + the real one; perturbations are arbitrary but all DIFFER from gold
    def perturb(shift, mul):
        out = []
        for v in base:
            if isinstance(v, int):
                out.append(v * mul + shift if mul != 1 or shift != 0 else v + 7)
            else:
                out.append(str(v) + f"~{shift}")
        return out
    decoy_vals = {
        0: perturb(0, 3),
        1: perturb(11, 1),
        2: perturb(-5, 1),
        3: perturb(0, 2),
        4: perturb(100, 1),
        5: perturb(-1, 1),
        6: list(base),   # THE REAL ONE
    }
    # Emit the `_SPEC_k` blocks in SCRAMBLED source order (_PERM), so the REAL array (index 6) is NOT
    # first or last in the file. This defeats any uniform runtime/source heuristic (first-match,
    # last-match, majority-vote over the comment-labelled lines): with the real values appearing in
    # exactly ONE of 7 arrays and that array at a non-extremal source position, no positional or
    # majority rule selects it — the ONLY way to know which `_SPEC_k` is authoritative is to resolve the
    # `SPEC = _TBL[pos]` indirection (i.e. judge that the defn-stub is insufficient and read on).
    blocks = [f"_SPEC_{k} = {_flat_spec(keys, decoy_vals[k])}" for k in _PERM]
    return blocks, 6


def _biglib_b(p):
    blocks, impl_idx = _decoy_arrays(p)
    tbl_line = "_TBL = [" + ", ".join(f"_SPEC_{k}" for k in _PERM) + "]"
    alias = f"SPEC = _TBL[{_PERM.index(impl_idx)}]"
    # version sentinel -> decoys -> _TBL (references them) -> alias, all buried in filler.
    body = _VERSION_LINE + "\n" + _scatter(blocks + [tbl_line, alias],
                                           gaps=(12, 10, 9, 11, 10, 9, 11, 10, 9))
    return body, impl_idx


def _alias_b(p):
    _, impl_idx = _decoy_arrays(p)
    return f"SPEC = _TBL[{_PERM.index(impl_idx)}]"


TASKS_EFFIC_ND = []
for _p in _PAIRS:
    _buggy = _emit_buggy(_p)
    _gold = _emit_gold(_p)
    _wrong = _emit_wrong(_p)
    _test = _emit_test(_p)
    _real_body = _gold_body_src(_p)   # BARE gold def (no import) -> exec'able in-process for the digest
    # delegate symbols a forwarding fix COULD try: SPEC (data) and SPEC_VERSION (sentinel). The gold
    # must NOT introduce a call to SPEC (it is not callable anyway) nor a forwarding import of SPEC.
    _delegate = ["SPEC"]

    # VARIANT A — defn-sufficient: SPEC is a full literal table; <defn SPEC> returns it.
    TASKS_EFFIC_ND.append(dict(
        name=f"effic_nd_{_p['pair_id']}_a", pair_id=_p["pair_id"], group="rich", target="target.py",
        symbol="SPEC", defn_sufficient=True, requires_read=False,
        files={"target.py": _buggy, "biglib.py": _biglib_a(_p)},
        test=_test, gold_target=_gold, inputs=_p["inputs"], real_body=_real_body, local=_p["local"],
        symbol_defns={"defn": {"SPEC": _biglib_a(_p)},  # placeholder; verifier uses LIVE goto_definition
                    "members": {"SPEC": "SPEC: flat (key, value, ...) reference table"},
                    "refs": {"SPEC": ["biglib.py"]}},
        wrong_guess=_wrong, wrong_kind="value", wrong_note=_p["wrong_note"],
        real_api=_p["real_api"], delegate_syms=_delegate))

    # VARIANT B — defn-insufficient: SPEC is an opaque subscript alias; must <read> to find the array.
    _biglib_b_src, _ = _biglib_b(_p)
    TASKS_EFFIC_ND.append(dict(
        name=f"effic_nd_{_p['pair_id']}_b", pair_id=_p["pair_id"], group="rich", target="target.py",
        symbol="SPEC", defn_sufficient=False, requires_read=True,
        files={"target.py": _buggy, "biglib.py": _biglib_b_src},
        test=_test, gold_target=_gold, inputs=_p["inputs"], real_body=_real_body, local=_p["local"],
        symbol_defns={"defn": {"SPEC": _alias_b(_p)}},
        wrong_guess=_wrong, wrong_kind="value", wrong_note=_p["wrong_note"],
        real_api=_p["real_api"], delegate_syms=_delegate))


def _gold_output_literals(t):
    """Expected-output literals (repr of each gold result over INPUTS) that MUST NOT appear in the test."""
    ns: dict = {"SPEC_VERSION": 1}
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

    print(f"{'task':22} {'suf':4} {'R1buggy':8} {'R2gold':7} {'pyfl':5} {'R3big':12} "
          f"{'R4guess':8} {'R5leak':7} {'R6decouple':12} {'Rnodel':7} {'Rarb':6}")
    allok = True
    by_pair = {}
    for t in TASKS_EFFIC_ND:
        tgt = t["target"]; sym = t["symbol"]; local = t["local"]
        by_pair.setdefault(t["pair_id"], {})[("a" if t["defn_sufficient"] else "b")] = t
        gold_all = {**t["files"], tgt: t["gold_target"]}

        # R1: buggy target FAILs the test
        r1 = not passes(t["files"], tgt, t["test"])

        # R2: gold PASSes AND is pyrefly-clean
        r2_pass = passes(gold_all, tgt, t["test"])
        nerr_gold = diag(gold_all, tgt, t["test"]).count("[error]")
        r2_clean = nerr_gold == 0

        # R3: biglib genuinely expensive to read.
        big_n = len(t["files"]["biglib.py"].splitlines())
        if t["defn_sufficient"]:
            span_a, _ = gotodef(t["files"], tgt, t["test"], sym)
            defn_n = len((span_a or "").splitlines())
            r3 = defn_n * 5 < big_n
            r3str = f"{defn_n}*5<{big_n}" + ("=y" if r3 else "=N!")
        else:
            r3 = big_n >= 200
            r3str = f"{big_n}" + (">=200" if r3 else "<200!")

        # R4: NON-GUESSABLE — (a) idiomatic wrong_guess spliced into gold => test FAIL; (b) hash-unique:
        # no decoy array reproduces the gold checksum.
        wrong_all = {**t["files"], tgt: t["wrong_guess"]}
        wrong_pass = passes(wrong_all, tgt, t["test"])
        r4_splice = (not wrong_pass) and (t["wrong_guess"] != t["gold_target"])
        # (b) decoy-hash uniqueness
        _gns: dict = {"SPEC_VERSION": 1}; exec(t["real_body"], _gns)
        _gfn = _gns[local]
        _gold_got = "|".join(repr(_gfn(*a)) for a in t["inputs"])
        _pair = next(pp for pp in _PAIRS if pp["pair_id"] == t["pair_id"])
        _blocks, _impl_idx = _decoy_arrays(_pair)
        _collisions = []
        # recompute the decoy value lists (same logic as _decoy_arrays) and emulate the dispatch on each
        def _decoy_vals_list(pp):
            base = pp["vals"]
            def perturb(shift, mul):
                out = []
                for v in base:
                    if isinstance(v, int):
                        out.append(v * mul + shift if mul != 1 or shift != 0 else v + 7)
                    else:
                        out.append(str(v) + f"~{shift}")
                return out
            return {0: perturb(0, 3), 1: perturb(11, 1), 2: perturb(-5, 1), 3: perturb(0, 2),
                    4: perturb(100, 1), 5: perturb(-1, 1), 6: list(base)}
        _dv = _decoy_vals_list(_pair)
        for _k in range(7):
            _m = dict(zip(_pair["keys"], _dv[_k]))
            try:
                _g = "|".join(repr(_m[a[0]]) for a in t["inputs"])
            except Exception:
                _g = None
            if _g == _gold_got and _k != _impl_idx:
                _collisions.append(_k)
        r4_unique = (len(_collisions) == 0)
        r4 = r4_splice and r4_unique

        # R5: no-leak — strip the legitimate INPUTS line + the 64-hex hash, then assert no distinctive
        # (>=4-char) expected-output literal and no joined-got survives; also no API-surface token.
        import re as _re5
        _resid = t["test"]
        _resid = _re5.sub(r"^INPUTS = .*$", "", _resid, flags=_re5.MULTILINE)
        _resid = _re5.sub(r"[0-9a-f]{64}", "", _resid)
        _out_lits = _gold_output_literals(t)
        _joined = "|".join(_out_lits)
        _lit_leaks = [lit for lit in _out_lits if len(lit) >= 4 and lit in _resid]
        if _joined in t["test"]:
            _lit_leaks.append("<joined-got>")
        # API tokens: a forwarding call SPEC( or a from-biglib import of SPEC in the test would be a leak
        api_leak = [tok for tok in ("SPEC[", "SPEC(", "from biglib import SPEC") if tok in t["test"]]
        r5 = (len(_lit_leaks) == 0) and (len(api_leak) == 0)

        # R6: DECOUPLING — goto_definition(SPEC) over the LIVE workspace.
        span, _path = gotodef(t["files"], tgt, t["test"], sym)
        full_big = t["files"]["biglib.py"]
        real_api = t["real_api"]
        if t["defn_sufficient"]:
            r6 = (span is not None) and (real_api in span)
            r6str = "A:defn-has" if r6 else "A:DEFN-MISS!"
        else:
            stub_line = (span is not None) and (real_api not in span) and ("= _TBL[" in span) \
                        and ("_SPEC_" not in span) and (len(span.strip().splitlines()) == 1)
            read_has = real_api in full_big
            r6 = stub_line and read_has
            r6str = "B:stub-only" if r6 else "B:LEAKY!"

        # R-nodel (NEW): the gold diff vs the buggy target must INLINE content, with NO content-free
        # forward. Concretely: (a) the gold introduces NO call to any delegate_sym (e.g. `SPEC(` / `SPEC[`)
        # — the only added lines are literal `return <const>` branches; (b) the gold adds no
        # `from biglib import` line beyond the buggy target's (the buggy and gold headers are identical),
        # so no new symbol is pulled in to forward to. The new content is the transcribed constants.
        gold_src = t["gold_target"]; buggy_src = t["files"]["target.py"]
        added_lines = [ln for ln in gold_src.splitlines() if ln not in buggy_src.splitlines()]
        nodel_no_callforward = not any(
            (ds + "(" in gold_src) or (ds + "[" in gold_src) for ds in t["delegate_syms"]
        )
        # the import header (everything before the def) is byte-identical between buggy and gold:
        gold_head = gold_src.split("def ", 1)[0]
        buggy_head = buggy_src.split("def ", 1)[0]
        nodel_same_header = (gold_head == buggy_head) and ("from biglib import SPEC\n" not in gold_src) \
            and ("from biglib import SPEC " not in gold_src)
        # every ADDED line is a transcribed branch (`return ...` or `elif`/`if` over the arg) — i.e. the
        # fix is inline content, not a forward.
        nodel_added_are_branches = all(
            ("return " in ln) or ("==" in ln) or (ln.strip() == "") for ln in added_lines
        )
        rnodel = nodel_no_callforward and nodel_same_header and nodel_added_are_branches

        # R-arb (NEW): the pinned values are arbitrary — the gold output differs from the
        # idiomatic/derivable-from-name guess output on EVERY pinned input (so no subset is name-derivable).
        # Build the idiomatic-guess mapping (the pair's wrongvals) directly and compare to gold per input.
        _wrong_map = dict(zip(_pair["keys"], _pair["wrongvals"]))
        _wrong_got = [repr(_wrong_map[a[0]]) for a in t["inputs"]]
        _gold_list = [repr(_gfn(*a)) for a in t["inputs"]]
        rarb = all(g != w for g, w in zip(_gold_list, _wrong_got))

        ok = r1 and r2_pass and r2_clean and r3 and r4 and r5 and r6 and rnodel and rarb
        if not ok:
            allok = False
        print(f"{t['name']:22} "
              f"{('A' if t['defn_sufficient'] else 'B'):4} "
              f"{'FAIL' if r1 else 'PASS!':8} "
              f"{'PASS' if r2_pass else 'FAIL!':7} "
              f"{nerr_gold:<5} "
              f"{r3str:12} "
              f"{'fails' if r4 else 'SOLVES!':8} "
              f"{'ok' if r5 else 'LEAK!':7} "
              f"{r6str:12} "
              f"{'ok' if rnodel else 'DELEG!':7} "
              f"{'ok' if rarb else 'GUESS!':6}"
              f"{'' if ok else '  <-- PROBLEM'}")
        if not r4:
            print(f"     ! R4 fail: splice_ok={r4_splice} decoy_collisions={_collisions} "
                  f"wrong_pass={wrong_pass}  ({t['wrong_note']})")
        if not r5:
            print(f"     ! R5 leak: lit_leaks={_lit_leaks} api_leak={api_leak}")
        if not r2_clean:
            print(f"     ! gold not clean: {diag(gold_all, tgt, t['test']).splitlines()[:2]}")
        if not r6:
            print(f"     ! R6 decouple: span={span!r} real_api_in_span="
                  f"{(span is not None and real_api in span)} real_api_in_big={real_api in full_big}")
        if not rnodel:
            print(f"     ! R-nodel: no_callforward={nodel_no_callforward} same_header={nodel_same_header} "
                  f"added_are_branches={nodel_added_are_branches} added={added_lines[:3]}")
        if not rarb:
            print(f"     ! R-arb: gold outputs equal idiomatic-guess outputs -> name-derivable")

    # R7: PAIR-SURFACE-IDENTITY — A and B target.py + test + symbol byte-identical (no prompt cue).
    print("\n--- R7 pair surface identity (the crux: A and B prompts are indistinguishable) ---")
    for pid, ab in sorted(by_pair.items()):
        a, b = ab.get("a"), ab.get("b")
        complete = a is not None and b is not None
        same_target = complete and (a["files"]["target.py"] == b["files"]["target.py"])
        same_test = complete and (a["test"] == b["test"])
        same_sym = complete and (a["symbol"] == b["symbol"])
        r7 = same_target and same_test and same_sym
        if not r7:
            allok = False
        print(f"  pair {pid:9} target-identical={same_target}  test-identical={same_test}  "
              f"symbol-identical={same_sym}  {'OK' if r7 else '<-- PROBLEM'}")

    print(f"\nALL OK ({len(TASKS_EFFIC_ND)} tasks)" if allok else "PROBLEMS — fix before review")
