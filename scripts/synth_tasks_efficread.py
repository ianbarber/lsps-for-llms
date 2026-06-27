#!/usr/bin/env python3
"""Read-required boundary suite.

This suite guards against a degenerate always-`<defn>` policy. Tasks are designed so that
`<defn>` is genuinely insufficient and the agent must read the whole `biglib.py` to solve.

Two flavors:
  A. Name-hidden (tasks 1-4): the stub describes the goal in prose but does not name the
     required symbol. The real symbol has a non-obvious private name buried among distractors
     with obvious names. The agent must read the file to discover the right symbol.
  B. Many-symbol (tasks 5-6): the stub names four or more required symbols. Issuing separate
     `<defn>` calls is more expensive and error-prone than one `<read>`, so reading is the
     economical move.

Task schema mirrors synth_tasks_effic.py with additional flags:
  defn_sufficient=False, requires_read=True, many_symbol=bool.

Verifier checks R1-R5 from synth_tasks_effic.py plus a read-required condition appropriate
for each flavor.
"""

# ---- filler to make biglib.py genuinely expensive to read (each ~8 lines; many of them) ----
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


def _scatter(blocks: list[str], gaps=(14, 12, 12, 12, 14)) -> str:
    """Interleave real/distractor `blocks` between bands of filler so every real symbol is buried
    deep in a big file (a <read> must scan it), while a <defn> — if you knew the name — would return
    just the one def. `gaps` are the filler-block sizes between/around the placed blocks."""
    parts = []
    base = 0
    for i, blk in enumerate(blocks):
        g = gaps[i % len(gaps)]
        parts.append(_filler(g, base)); base += g
        parts.append(blk)
    parts.append(_filler(gaps[-1], base))
    return "\n\n".join(parts)


# ============================================================================
# FLAVOR A — name-hidden. Target docstring is PROSE only; real symbol has a non-obvious private name
# buried among _filler AND among DISTRACTOR helpers carrying the obvious/idiomatic names. <defn> needs
# the name; the name is undiscoverable from the stub -> must <read biglib.py>.
# ============================================================================

# ---- TASK 1: string normalization ("slug") -------------------------------------------------------
# real:  _canon_token(text)  -> lowercased, non-alnum runs -> single '-', trimmed
# distractors (WRONG): slugify (keeps underscores, no trim), normalize (only lowercases),
#                      to_slug (uses '_' not '-')
_T1_REAL = (
    "def _canon_token(text: str) -> str:\n"
    '    """Canonical URL token: lowercase, every run of non-alphanumerics becomes a single\n'
    '    \'-\', and leading/trailing \'-\' are stripped. (This is the one the pipeline uses.)"""\n'
    "    out: list[str] = []\n"
    "    prev_dash = False\n"
    "    for ch in text.lower():\n"
    "        if ch.isalnum():\n"
    "            out.append(ch); prev_dash = False\n"
    "        elif not prev_dash:\n"
    "            out.append('-'); prev_dash = True\n"
    "    return ''.join(out).strip('-')\n"
)
_T1_DISTRACTORS = (
    "def slugify(text: str) -> str:\n"
    '    """Lowercase and replace spaces with hyphens (keeps underscores/punctuation as-is)."""\n'
    "    return text.lower().replace(' ', '-')\n"
    "\n\n"
    "def normalize(text: str) -> str:\n"
    '    """Lowercase the text. Nothing else."""\n'
    "    return text.lower()\n"
    "\n\n"
    "def to_slug(text: str) -> str:\n"
    '    """Underscore-style slug: non-alnum -> single \'_\'."""\n'
    "    import re\n"
    "    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')\n"
)
_BIGLIB1 = _scatter([_T1_DISTRACTORS, _T1_REAL])
_T1_STUB = (
    "from biglib import *\n\n"
    "def make_slug(title: str) -> str:\n"
    '    """Turn a human title into a canonical URL slug using the appropriate helper in\n'
    "    biglib: lowercase, each run of non-alphanumeric characters collapsed to a single\n"
    "    hyphen, and no leading or trailing hyphen. (Find the helper that does exactly this;\n"
    '    the obvious-named ones do NOT.)"""\n'
    "    raise NotImplementedError\n"
)
_T1_GOLD = (
    "from biglib import _canon_token\n\n"
    "def make_slug(title: str) -> str:\n"
    '    """Turn a human title into a canonical URL slug using the appropriate helper in\n'
    "    biglib: lowercase, each run of non-alphanumeric characters collapsed to a single\n"
    "    hyphen, and no leading or trailing hyphen. (Find the helper that does exactly this;\n"
    '    the obvious-named ones do NOT.)"""\n'
    "    return _canon_token(title)\n"
)
_T1_TEST = (
    "from target import make_slug\n"
    "assert make_slug('Hello,  World!') == 'hello-world'\n"
    "assert make_slug('  A__B  ') == 'a-b'\n"
    "assert make_slug('foo.bar.baz') == 'foo-bar-baz'\n"
)

# ---- TASK 2: integer-checksum validation (Luhn-style, but custom) --------------------------------
# real:  _mod10_ok(digits)  -> doubles every SECOND digit from the LEFT (not right), sum % 10 == 0
# distractors (WRONG): validate (parity check), is_valid (sum % 10 only), check_digit (returns the
#                      would-be check digit, not a bool, and doubles from the right)
_T2_REAL = (
    "def _mod10_ok(digits: list[int]) -> bool:\n"
    '    """House checksum: double every SECOND digit counting from the LEFT (index 1, 3, ...);\n'
    '    if a doubled value exceeds 9 subtract 9; the total must be divisible by 10."""\n'
    "    total = 0\n"
    "    for i, d in enumerate(digits):\n"
    "        if i % 2 == 1:\n"
    "            d *= 2\n"
    "            if d > 9:\n"
    "                d -= 9\n"
    "        total += d\n"
    "    return total % 10 == 0\n"
)
_T2_DISTRACTORS = (
    "def validate(digits: list[int]) -> bool:\n"
    '    """Parity validator: true when the count of odd digits is even."""\n'
    "    return sum(1 for d in digits if d % 2) % 2 == 0\n"
    "\n\n"
    "def is_valid(digits: list[int]) -> bool:\n"
    '    """Plain mod-10: true when the raw digit sum is divisible by 10 (no doubling)."""\n'
    "    return sum(digits) % 10 == 0\n"
    "\n\n"
    "def check_digit(digits: list[int]) -> int:\n"
    '    """Return the trailing check digit that WOULD make a right-doubled Luhn sum valid."""\n'
    "    total = 0\n"
    "    for i, d in enumerate(reversed(digits)):\n"
    "        if i % 2 == 0:\n"
    "            d *= 2\n"
    "            if d > 9:\n"
    "                d -= 9\n"
    "        total += d\n"
    "    return (10 - total % 10) % 10\n"
)
_BIGLIB2 = _scatter([_T2_DISTRACTORS, _T2_REAL])
_T2_STUB = (
    "from biglib import *\n\n"
    "def accepts(digits: list[int]) -> bool:\n"
    '    """Return True iff `digits` pass the house checksum implemented in biglib: every\n'
    "    second digit FROM THE LEFT is doubled (cast back below 10 by subtracting 9), and the\n"
    "    grand total is a multiple of 10. Use the biglib helper that does exactly this; the\n"
    '    plausibly-named validators in biglib use different rules."""\n'
    "    raise NotImplementedError\n"
)
_T2_GOLD = (
    "from biglib import _mod10_ok\n\n"
    "def accepts(digits: list[int]) -> bool:\n"
    '    """Return True iff `digits` pass the house checksum implemented in biglib: every\n'
    "    second digit FROM THE LEFT is doubled (cast back below 10 by subtracting 9), and the\n"
    "    grand total is a multiple of 10. Use the biglib helper that does exactly this; the\n"
    '    plausibly-named validators in biglib use different rules."""\n'
    "    return _mod10_ok(digits)\n"
)
# gold-derived expectations: [1,2,3,4] -> 1 + 4 + 3 + 8 = 16 -> not ok; [0,5,0,5] -> 0+1+0+1=2 -> no;
# [2,4] -> 2 + 8 = 10 -> ok ; [5,5,5,5] -> 5+1+5+1=12 -> no ; [0] -> 0 -> ok
_T2_TEST = (
    "from target import accepts\n"
    "assert accepts([2, 4]) is True\n"
    "assert accepts([1, 2, 3, 4]) is False\n"
    "assert accepts([0]) is True\n"
    "assert accepts([5, 5, 5, 5]) is False\n"
)

# ---- TASK 3: byte-group encoding ------------------------------------------------------------------
# real:  _pack_groups(data)  -> hex of each byte, joined by ':' in groups, UPPERCASE, 2-wide
# distractors (WRONG): encode (base64-ish placeholder), b32encode (returns lowercased no-sep),
#                      to_base32 (joins by '-' lowercase)
_T3_REAL = (
    "def _pack_groups(data: bytes) -> str:\n"
    '    """Wire format: each byte as a 2-wide UPPERCASE hex pair, pairs joined by a single\n'
    '    colon. b\'\\\\x0a\\\\xff\' -> \'0A:FF\'. Empty input -> empty string."""\n'
    "    return ':'.join(f'{b:02X}' for b in data)\n"
)
_T3_DISTRACTORS = (
    "def encode(data: bytes) -> str:\n"
    '    """Lossy placeholder encoder: the byte values joined by commas, as decimals."""\n'
    "    return ','.join(str(b) for b in data)\n"
    "\n\n"
    "def b32encode(data: bytes) -> str:\n"
    '    """Lowercase hex with NO separators (e.g. b\'\\\\x0a\\\\xff\' -> \'0aff\')."""\n'
    "    return ''.join(f'{b:02x}' for b in data)\n"
    "\n\n"
    "def to_base32(data: bytes) -> str:\n"
    '    """Lowercase hex pairs joined by hyphens (e.g. \'0a-ff\')."""\n'
    "    return '-'.join(f'{b:02x}' for b in data)\n"
)
_BIGLIB3 = _scatter([_T3_DISTRACTORS, _T3_REAL])
_T3_STUB = (
    "from biglib import *\n\n"
    "def wire(data: bytes) -> str:\n"
    '    """Render `data` in the on-the-wire format defined in biglib: each byte as an\n'
    "    UPPERCASE two-character hex pair, pairs joined by a single colon (empty input gives\n"
    "    an empty string). Several similarly-named encoders exist in biglib but use lowercase\n"
    '    and/or the wrong separator; pick the one matching this exact format."""\n'
    "    raise NotImplementedError\n"
)
_T3_GOLD = (
    "from biglib import _pack_groups\n\n"
    "def wire(data: bytes) -> str:\n"
    '    """Render `data` in the on-the-wire format defined in biglib: each byte as an\n'
    "    UPPERCASE two-character hex pair, pairs joined by a single colon (empty input gives\n"
    "    an empty string). Several similarly-named encoders exist in biglib but use lowercase\n"
    '    and/or the wrong separator; pick the one matching this exact format."""\n'
    "    return _pack_groups(data)\n"
)
_T3_TEST = (
    "from target import wire\n"
    "assert wire(bytes([10, 255])) == '0A:FF'\n"
    "assert wire(b'') == ''\n"
    "assert wire(bytes([0, 1, 16])) == '00:01:10'\n"
)

# ---- TASK 4: duration parsing ---------------------------------------------------------------------
# real:  _decode_span(text)  -> '1h30m' / '45m' / '90s' -> total SECONDS
# distractors (WRONG): parse_duration (assumes the whole string is minutes),
#                      parse (returns a (h,m,s) tuple, not seconds), to_seconds (treats bare number
#                      as seconds but ignores unit suffixes)
_T4_REAL = (
    "def _decode_span(text: str) -> int:\n"
    '    """Parse a compact duration like \'1h30m\', \'45m\', \'90s\', \'2h\' into TOTAL SECONDS.\n'
    '    Recognised units: h (3600), m (60), s (1). Components may appear in any combination."""\n'
    "    import re\n"
    "    total = 0\n"
    "    for value, unit in re.findall(r'(\\d+)([hms])', text):\n"
    "        total += int(value) * {'h': 3600, 'm': 60, 's': 1}[unit]\n"
    "    return total\n"
)
_T4_DISTRACTORS = (
    "def parse_duration(text: str) -> int:\n"
    '    """Interpret the leading number as a count of MINUTES, ignoring any unit suffix."""\n'
    "    import re\n"
    "    m = re.match(r'(\\d+)', text)\n"
    "    return int(m.group(1)) * 60 if m else 0\n"
    "\n\n"
    "def parse(text: str) -> tuple[int, int, int]:\n"
    '    """Return a (hours, minutes, seconds) tuple parsed from a \'h/m/s\' string."""\n'
    "    import re\n"
    "    g = {u: 0 for u in 'hms'}\n"
    "    for value, unit in re.findall(r'(\\d+)([hms])', text):\n"
    "        g[unit] = int(value)\n"
    "    return (g['h'], g['m'], g['s'])\n"
    "\n\n"
    "def to_seconds(text: str) -> int:\n"
    '    """Treat the whole string as a bare integer number of seconds (no unit handling)."""\n'
    "    return int(text) if text.isdigit() else 0\n"
)
_BIGLIB4 = _scatter([_T4_DISTRACTORS, _T4_REAL])
_T4_STUB = (
    "from biglib import *\n\n"
    "def seconds(spec: str) -> int:\n"
    '    """Convert a compact duration spec such as \'1h30m\', \'45m\' or \'90s\' into a total\n'
    "    number of SECONDS, summing hour/minute/second components. Use the biglib helper that\n"
    "    returns an integer second-count; the other parsers in biglib assume minutes, return a\n"
    '    tuple, or ignore the unit suffixes."""\n'
    "    raise NotImplementedError\n"
)
_T4_GOLD = (
    "from biglib import _decode_span\n\n"
    "def seconds(spec: str) -> int:\n"
    '    """Convert a compact duration spec such as \'1h30m\', \'45m\' or \'90s\' into a total\n'
    "    number of SECONDS, summing hour/minute/second components. Use the biglib helper that\n"
    "    returns an integer second-count; the other parsers in biglib assume minutes, return a\n"
    '    tuple, or ignore the unit suffixes."""\n'
    "    return _decode_span(spec)\n"
)
_T4_TEST = (
    "from target import seconds\n"
    "assert seconds('1h30m') == 5400\n"
    "assert seconds('45m') == 2700\n"
    "assert seconds('90s') == 90\n"
    "assert seconds('2h') == 7200\n"
)

# ============================================================================
# FLAVOR B — many-symbol. The needed symbol NAMES are KNOWN (the docstring lists them), but the task
# needs >=4 distinct ones; >=4 separate <defn> calls cost more than one <read biglib.py>, so reading
# once is the natural / economical move. many_symbol=True.
# ============================================================================

# ---- TASK 5: assemble a price record from 4 named field helpers ----------------------------------
_T5_HELPERS = (
    "def net_amount(cents: int) -> int:\n"
    '    """Net amount in cents (passes the integer cents straight through)."""\n'
    "    return cents\n"
    "\n\n"
    "def tax_amount(cents: int) -> int:\n"
    '    """Tax in cents: 8% of net, rounded DOWN (floor)."""\n'
    "    return cents * 8 // 100\n"
    "\n\n"
    "def gross_amount(cents: int) -> int:\n"
    '    """Gross in cents: net plus tax (tax floored at 8%)."""\n'
    "    return cents + cents * 8 // 100\n"
    "\n\n"
    "def currency_code() -> str:\n"
    '    """ISO currency code used by this ledger."""\n'
    "    return 'USD'\n"
)
_BIGLIB5 = _scatter([_T5_HELPERS], gaps=(16, 16))
_T5_STUB = (
    "from biglib import net_amount, tax_amount, gross_amount, currency_code\n\n"
    "def line_item(cents: int) -> dict:\n"
    '    """Build a price record using the FOUR biglib helpers net_amount, tax_amount,\n'
    "    gross_amount and currency_code. Return a dict with keys \'net\', \'tax\', \'gross\'\n"
    "    (each the cents value from the matching helper) and \'currency\' (the currency code).\n"
    '    Reading biglib once is cheaper than four separate symbol lookups."""\n'
    "    raise NotImplementedError\n"
)
_T5_GOLD = (
    "from biglib import net_amount, tax_amount, gross_amount, currency_code\n\n"
    "def line_item(cents: int) -> dict:\n"
    '    """Build a price record using the FOUR biglib helpers net_amount, tax_amount,\n'
    "    gross_amount and currency_code. Return a dict with keys \'net\', \'tax\', \'gross\'\n"
    "    (each the cents value from the matching helper) and \'currency\' (the currency code).\n"
    '    Reading biglib once is cheaper than four separate symbol lookups."""\n'
    "    return {\n"
    "        'net': net_amount(cents),\n"
    "        'tax': tax_amount(cents),\n"
    "        'gross': gross_amount(cents),\n"
    "        'currency': currency_code(),\n"
    "    }\n"
)
_T5_TEST = (
    "from target import line_item\n"
    "r = line_item(1000)\n"
    "assert r == {'net': 1000, 'tax': 80, 'gross': 1080, 'currency': 'USD'}\n"
    "r2 = line_item(255)\n"
    "assert r2 == {'net': 255, 'tax': 20, 'gross': 275, 'currency': 'USD'}\n"
)

# ---- TASK 6: build a packet header from 4 named encoders -----------------------------------------
_T6_HELPERS = (
    "def version_nibble() -> int:\n"
    '    """Protocol version occupying the high nibble (value 0x4)."""\n'
    "    return 0x4\n"
    "\n\n"
    "def flag_bits(urgent: bool) -> int:\n"
    '    """Flag nibble: 0x1 when urgent, else 0x0."""\n'
    "    return 0x1 if urgent else 0x0\n"
    "\n\n"
    "def length_byte(payload: bytes) -> int:\n"
    '    """Length field: number of payload bytes, capped at 255."""\n'
    "    return min(len(payload), 255)\n"
    "\n\n"
    "def checksum_byte(payload: bytes) -> int:\n"
    '    """Checksum field: XOR of every payload byte (0 for empty payload)."""\n'
    "    acc = 0\n"
    "    for b in payload:\n"
    "        acc ^= b\n"
    "    return acc\n"
)
_BIGLIB6 = _scatter([_T6_HELPERS], gaps=(16, 16))
_T6_STUB = (
    "from biglib import version_nibble, flag_bits, length_byte, checksum_byte\n\n"
    "def header(payload: bytes, urgent: bool) -> tuple[int, int, int]:\n"
    '    """Assemble a 3-int packet header using the FOUR biglib encoders version_nibble,\n'
    "    flag_bits, length_byte and checksum_byte. Return (first, length, checksum) where\n"
    "    `first` packs the version nibble in the high 4 bits and the flag nibble in the low 4\n"
    "    bits ((version << 4) | flags), `length` is the length byte, and `checksum` is the\n"
    '    checksum byte. One read of biglib beats four separate symbol lookups."""\n'
    "    raise NotImplementedError\n"
)
_T6_GOLD = (
    "from biglib import version_nibble, flag_bits, length_byte, checksum_byte\n\n"
    "def header(payload: bytes, urgent: bool) -> tuple[int, int, int]:\n"
    '    """Assemble a 3-int packet header using the FOUR biglib encoders version_nibble,\n'
    "    flag_bits, length_byte and checksum_byte. Return (first, length, checksum) where\n"
    "    `first` packs the version nibble in the high 4 bits and the flag nibble in the low 4\n"
    "    bits ((version << 4) | flags), `length` is the length byte, and `checksum` is the\n"
    '    checksum byte. One read of biglib beats four separate symbol lookups."""\n'
    "    first = (version_nibble() << 4) | flag_bits(urgent)\n"
    "    return (first, length_byte(payload), checksum_byte(payload))\n"
)
# 0x4<<4 | 0x1 = 0x41 = 65 ; len 3 ; xor(1,2,3) = 0
_T6_TEST = (
    "from target import header\n"
    "assert header(bytes([1, 2, 3]), True) == (65, 3, 0)\n"
    "assert header(b'', False) == (64, 0, 0)\n"
    "assert header(bytes([255, 1]), True) == (65, 2, 254)\n"
)


TASKS_EFFICREAD = [
    dict(name="readreq_slug_canon", group="readreq", target="target.py", symbol="_canon_token",
         symbols=["_canon_token"], many_symbol=False,
         files={"target.py": _T1_STUB, "biglib.py": _BIGLIB1}, test=_T1_TEST, gold_target=_T1_GOLD,
         defn_sufficient=False, requires_read=True,
         symbol_defns={"defn": {"_canon_token": _T1_REAL}}),

    dict(name="readreq_checksum_mod10", group="readreq", target="target.py", symbol="_mod10_ok",
         symbols=["_mod10_ok"], many_symbol=False,
         files={"target.py": _T2_STUB, "biglib.py": _BIGLIB2}, test=_T2_TEST, gold_target=_T2_GOLD,
         defn_sufficient=False, requires_read=True,
         symbol_defns={"defn": {"_mod10_ok": _T2_REAL}}),

    dict(name="readreq_encode_packgroups", group="readreq", target="target.py", symbol="_pack_groups",
         symbols=["_pack_groups"], many_symbol=False,
         files={"target.py": _T3_STUB, "biglib.py": _BIGLIB3}, test=_T3_TEST, gold_target=_T3_GOLD,
         defn_sufficient=False, requires_read=True,
         symbol_defns={"defn": {"_pack_groups": _T3_REAL}}),

    dict(name="readreq_duration_decodespan", group="readreq", target="target.py", symbol="_decode_span",
         symbols=["_decode_span"], many_symbol=False,
         files={"target.py": _T4_STUB, "biglib.py": _BIGLIB4}, test=_T4_TEST, gold_target=_T4_GOLD,
         defn_sufficient=False, requires_read=True,
         symbol_defns={"defn": {"_decode_span": _T4_REAL}}),

    dict(name="readreq_price_record_many", group="readreq", target="target.py", symbol="gross_amount",
         symbols=["net_amount", "tax_amount", "gross_amount", "currency_code"], many_symbol=True,
         files={"target.py": _T5_STUB, "biglib.py": _BIGLIB5}, test=_T5_TEST, gold_target=_T5_GOLD,
         defn_sufficient=False, requires_read=True,
         symbol_defns={"defn": {}}),

    dict(name="readreq_packet_header_many", group="readreq", target="target.py", symbol="checksum_byte",
         symbols=["version_nibble", "flag_bits", "length_byte", "checksum_byte"], many_symbol=True,
         files={"target.py": _T6_STUB, "biglib.py": _BIGLIB6}, test=_T6_TEST, gold_target=_T6_GOLD,
         defn_sufficient=False, requires_read=True,
         symbol_defns={"defn": {}}),
]


def _leak_tokens(t: dict) -> list[str]:
    """The real symbol surface that must NOT appear in the test: each real symbol NAME and — when the
    gold accesses it as a member — its `.name(` access form. Derived from the task's `symbols` list so
    we never hand-maintain a brittle string list."""
    toks = set()
    for sym in t["symbols"]:
        toks.add(sym)          # bare name (FLAVOR B: imported names; FLAVOR A: the private helper)
        toks.add("." + sym)    # member-access form, if the gold ever uses one
    return sorted(toks)


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from scaffold.mock_env import MultiFileEnv

    def diag(files, target, test):
        e = MultiFileEnv(files, target, test); d = e.pyrefly_diagnostics(); e.close(); return d
    def passes(files, target, test):
        e = MultiFileEnv(files, target, test); ok = e.run_tests()["resolved"]; e.close(); return ok

    print(f"{'task':30} {'flavor':6} {'R1stub':7} {'R2gold':7} {'pyfl':5} "
          f"{'R3readreq':10} {'R4big':8} {'R5leak':7}")
    allok = True
    for t in TASKS_EFFICREAD:
        tgt = t["target"]
        stub_src = t["files"][tgt]
        big_src = t["files"]["biglib.py"]
        gold_all = {**t["files"], tgt: t["gold_target"]}

        # R1: stub FAILs the test
        r1 = not passes(t["files"], tgt, t["test"])

        # R2: gold PASSes AND is pyrefly-clean
        r2_pass = passes(gold_all, tgt, t["test"])
        nerr_gold = diag(gold_all, tgt, t["test"]).count("[error]")
        r2_clean = nerr_gold == 0

        # R3: READ-REQUIRED — flavor-aware.
        #   FLAVOR A (name-hidden): every real symbol NAME is absent from the stub (cannot be defn'd
        #     straight from the stub) AND present in biglib (discoverable only by reading).
        #   FLAVOR B (many-symbol): >=4 distinct symbols, all present in biglib; names ARE in the stub
        #     by design — reading is required on economic grounds (one read < four defn calls).
        missing_from_stub = [s for s in t["symbols"] if s not in stub_src]
        present_in_big = [s for s in t["symbols"] if s in big_src]
        all_in_big = len(present_in_big) == len(t["symbols"])
        if t["many_symbol"]:
            r3 = all_in_big and len(t["symbols"]) >= 4
        else:
            r3 = all_in_big and (len(missing_from_stub) == len(t["symbols"]))

        # R4: biglib big (>=200 lines) so a <read> is genuinely expensive
        big_n = len(big_src.splitlines())
        r4 = big_n >= 200

        # R5: no-leak — no real symbol name / member-access leaks into the test
        leak_toks = [tok for tok in _leak_tokens(t) if tok in t["test"]]
        r5 = len(leak_toks) == 0

        ok = r1 and r2_pass and r2_clean and r3 and r4 and r5
        if not ok:
            allok = False
        flavor = "many" if t["many_symbol"] else "hidden"
        print(f"{t['name']:30} {flavor:6} "
              f"{'FAIL' if r1 else 'PASS!':7} "
              f"{'PASS' if r2_pass else 'FAIL!':7} "
              f"{nerr_gold:<5} "
              f"{('stub-ok' if r3 else 'NAME!'):10} "
              f"{(str(big_n) + ('>=200' if r4 else '<200!')):8} "
              f"{'ok' if r5 else 'LEAK!':7}"
              f"{'' if ok else '  <-- PROBLEM'}")
        if not r3:
            if t["many_symbol"]:
                print(f"     ! R3 (many) read-required violated: n_symbols={len(t['symbols'])} "
                      f"(want >=4) present_in_big={present_in_big}")
            else:
                print(f"     ! R3 (hidden) read-required violated: missing_from_stub={missing_from_stub} "
                      f"(want all {len(t['symbols'])}) present_in_big={present_in_big}")
        if leak_toks:
            print(f"     ! R5 symbol name leaked into test: {leak_toks}")
        if not r2_clean:
            print(f"     ! gold not clean: {diag(gold_all, tgt, t['test']).splitlines()[:2]}")
        if not r1:
            print(f"     ! R1 stub did NOT fail the test (stub already solves?)")
    print(f"ALL OK ({len(TASKS_EFFICREAD)} tasks)" if allok else "PROBLEMS — fix before review")
