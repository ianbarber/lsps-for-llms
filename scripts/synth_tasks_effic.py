#!/usr/bin/env python3
"""Definition-sufficient efficiency suite.

Each task asks the agent to use one symbol defined in a large `biglib.py`. Reading the whole
file costs thousands of tokens; `<defn sym/>` returns the same definition in a few dozen.
The suite tests whether the agent learns to prefer the cheaper retrieval action.

Tasks are intentionally non-guessable: the correct member names and call signatures are
un-idiomatic, so a model that does not retrieve fails. Retrieval via `<read>` or the cheaper
`<defn>` is required to solve.

Task schema:
  name, group, target, symbol,
  files = {target.py: stub, biglib.py: large source with the symbol buried},
  test, gold_target,
  symbol_defns = {"defn": {sym: src}, "members": ..., "refs": ...},
  wrong_guess, wrong_kind, wrong_note.

Verifier (run as `__main__`) checks for each task:
  R1 the stub fails the test;
  R2 the gold passes and is pyrefly-clean;
  R3 biglib is large enough that `<defn>` is materially cheaper than `<read>`;
  R4 the idiomatic wrong guess fails, so retrieval is required;
  R5 the test does not leak the correct symbol name or signature.
"""

# ---- filler to make biglib.py genuinely expensive to read (each ~8 lines; many of them) ----
def _filler(n):
    out = []
    for i in range(n):
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


def _bury(defn: str, pre: int = 18, post: int = 18) -> str:
    """Bury `defn` between two blocks of filler so the real symbol is far from the top and a
    <read> must scan a big file, while <defn> returns just `defn` regardless of position."""
    return _filler(pre) + "\n\n" + defn + "\n" + _filler(post)


# ============================================================================
# TASK 1 (ANCHOR) — Account class, member-completion.
#   real: credit / worth   wrong guess: deposit / balance (idiomatic, ABSENT) -> attribute error
# ============================================================================
_ACCOUNT = (
    "class Account:\n"
    '    """A balance you deposit into and read."""\n'
    "    def __init__(self) -> None:\n"
    "        self._cents = 0\n"
    "    def credit(self, n: int) -> None:\n"
    "        self._cents += n\n"
    "    def worth(self) -> int:\n"
    "        return self._cents\n"
)
_BIGLIB1 = _bury(_ACCOUNT, 20, 20)
_T1_STUB = (
    "from biglib import Account\n\n"
    "def net_worth(deposits: list[int]) -> int:\n"
    '    """Open an Account, add every deposit, return the resulting worth."""\n'
    "    raise NotImplementedError\n"
)
_T1_GOLD = (
    "from biglib import Account\n\n"
    "def net_worth(deposits: list[int]) -> int:\n"
    '    """Open an Account, add every deposit, return the resulting worth."""\n'
    "    a = Account()\n"
    "    for d in deposits:\n"
    "        a.credit(d)\n"
    "    return a.worth()\n"
)
_T1_TEST = (
    "from target import net_worth\n"
    "assert net_worth([3, 7, 5]) == 15\n"
    "assert net_worth([]) == 0\n"
)
# wrong guess: idiomatic .deposit/.balance — absent from Account -> pyrefly missing-attribute.
_T1_WRONG = (
    "from biglib import Account\n\n"
    "def net_worth(deposits: list[int]) -> int:\n"
    '    """Open an Account, add every deposit, return the resulting worth."""\n'
    "    a = Account()\n"
    "    for d in deposits:\n"
    "        a.deposit(d)\n"
    "    return a.balance()\n"
)

# ============================================================================
# TASK 2 (ANCHOR) — transfer function, call-signature (arg ORDER + keyword-only memo).
#   real: transfer(dest, amount, *, memo='')   wrong guess: transfer(cents, acct) order swap -> type
# ============================================================================
_TRANSFER = (
    "def transfer(dest: str, amount: int, *, memo: str = '') -> tuple[str, int, str]:\n"
    '    """Build a transfer record (dest, amount, memo)."""\n'
    "    return (dest, amount, memo)\n"
)
_BIGLIB2 = _bury(_TRANSFER, 18, 18)
_T2_STUB = (
    "from biglib import transfer\n\n"
    "def pay(acct: str, cents: int) -> tuple[str, int, str]:\n"
    '    """Pay `cents` to account `acct`; return the transfer record."""\n'
    "    raise NotImplementedError\n"
)
_T2_GOLD = (
    "from biglib import transfer\n\n"
    "def pay(acct: str, cents: int) -> tuple[str, int, str]:\n"
    '    """Pay `cents` to account `acct`; return the transfer record."""\n'
    "    return transfer(acct, cents)\n"
)
_T2_TEST = (
    "from target import pay\n"
    "assert pay('A1', 250) == ('A1', 250, '')\n"
)
# wrong guess: idiomatic order swap transfer(amount, dest) -> int where str, str where int -> type.
_T2_WRONG = _T2_GOLD.replace("return transfer(acct, cents)", "return transfer(cents, acct)")

# ============================================================================
# TASK 3 — FIFO queue, member-completion.
#   real: offer / poll / depth   wrong guess: push / pop (idiomatic, ABSENT) -> attribute error
# ============================================================================
_QUEUE = (
    "class WorkQueue:\n"
    '    """A first-in/first-out integer channel. Construct with the initial items."""\n'
    "    def __init__(self, items: list[int] | None = None) -> None:\n"
    "        self._xs: list[int] = list(items or [])\n"
    "    def offer(self, x: int) -> None:\n"
    "        self._xs.append(x)\n"
    "    def poll(self) -> int:\n"
    "        return self._xs.pop(0)\n"
    "    def depth(self) -> int:\n"
    "        return len(self._xs)\n"
)
_BIGLIB3 = _bury(_QUEUE, 16, 18)
_T3_STUB = (
    "from biglib import WorkQueue\n\n"
    "def drain_first_two(items: list[int]) -> list[int]:\n"
    '    """Load `items` into a WorkQueue, then remove and return the two oldest, in order."""\n'
    "    raise NotImplementedError\n"
)
_T3_GOLD = (
    "from biglib import WorkQueue\n\n"
    "def drain_first_two(items: list[int]) -> list[int]:\n"
    '    """Load `items` into a WorkQueue, then remove and return the two oldest, in order."""\n'
    "    q = WorkQueue(items)\n"
    "    return [q.poll(), q.poll()]\n"
)
_T3_TEST = (
    "from target import drain_first_two\n"
    "assert drain_first_two([10, 20, 30]) == [10, 20]\n"
    "assert drain_first_two([5, 6]) == [5, 6]\n"
)
# wrong guess: idiomatic .pop() (stack pop = LIFO, also absent on the class) -> attribute error.
_T3_WRONG = _T3_GOLD.replace("return [q.poll(), q.poll()]", "return [q.pop(), q.pop()]")

# ============================================================================
# TASK 4 — interval Span, member-completion.
#   real: lo / hi / width   wrong guess: start / end (idiomatic, ABSENT) -> attribute error
# ============================================================================
_SPAN = (
    "class Span:\n"
    '    """A closed integer interval. Construct as Span(low, high)."""\n'
    "    def __init__(self, a: int, b: int) -> None:\n"
    "        self._a = a\n"
    "        self._b = b\n"
    "    def lo(self) -> int:\n"
    "        return self._a\n"
    "    def hi(self) -> int:\n"
    "        return self._b\n"
    "    def width(self) -> int:\n"
    "        return self._b - self._a\n"
)
_BIGLIB4 = _bury(_SPAN, 20, 16)
_T4_STUB = (
    "from biglib import Span\n\n"
    "def midpoint(s: Span) -> int:\n"
    '    """Return the integer midpoint of the interval (its low plus half its width)."""\n'
    "    raise NotImplementedError\n"
)
_T4_GOLD = (
    "from biglib import Span\n\n"
    "def midpoint(s: Span) -> int:\n"
    '    """Return the integer midpoint of the interval (its low plus half its width)."""\n'
    "    return s.lo() + s.width() // 2\n"
)
_T4_TEST = (
    "from target import midpoint\n"
    "from biglib import Span\n"
    "assert midpoint(Span(2, 10)) == 6\n"
    "assert midpoint(Span(0, 7)) == 3\n"
)
# wrong guess: idiomatic .start()/.end() -> absent on Span -> attribute error.
_T4_WRONG = _T4_GOLD.replace("return s.lo() + s.width() // 2", "return s.start() + (s.end() - s.start()) // 2")

# ============================================================================
# TASK 5 — key-value Store, member-completion.
#   real: fetch / put / keys_sorted   wrong guess: get / set (idiomatic, ABSENT) -> attribute error
# ============================================================================
_STORE = (
    "class Store:\n"
    '    """A small keyed integer store. Construct with an initial mapping."""\n'
    "    def __init__(self, init: dict[str, int] | None = None) -> None:\n"
    "        self._d: dict[str, int] = dict(init or {})\n"
    "    def fetch(self, k: str) -> int:\n"
    "        return self._d[k]\n"
    "    def put(self, k: str, v: int) -> None:\n"
    "        self._d[k] = v\n"
    "    def keys_sorted(self) -> list[str]:\n"
    "        return sorted(self._d)\n"
)
_BIGLIB5 = _bury(_STORE, 18, 20)
_T5_STUB = (
    "from biglib import Store\n\n"
    "def first_value(s: Store) -> int:\n"
    '    """Return the value stored under the alphabetically-first key of the store."""\n'
    "    raise NotImplementedError\n"
)
_T5_GOLD = (
    "from biglib import Store\n\n"
    "def first_value(s: Store) -> int:\n"
    '    """Return the value stored under the alphabetically-first key of the store."""\n'
    "    return s.fetch(s.keys_sorted()[0])\n"
)
_T5_TEST = (
    "from target import first_value\n"
    "from biglib import Store\n"
    "assert first_value(Store({'b': 7, 'a': 3, 'c': 5})) == 3\n"
    "assert first_value(Store({'z': 9})) == 9\n"
)
# wrong guess: idiomatic dict-style .get(...) -> absent on Store -> attribute error.
_T5_WRONG = _T5_GOLD.replace("return s.fetch(s.keys_sorted()[0])", "return s.get(s.keys_sorted()[0])")

# ============================================================================
# TASK 6 — 2D point geometry, call-signature (arg ORDER: y before x).
#   real: make_point(y, x)   wrong guess: make_point(x, y) idiomatic order -> VALUE (type-clean)
# ============================================================================
_POINT = (
    "def make_point(row: int, col: int) -> tuple[int, int]:\n"
    '    """Construct a grid coordinate as (row, col) — i.e. (y, x), ROW FIRST."""\n'
    "    return (row, col)\n"
)
_BIGLIB6 = _bury(_POINT, 18, 18)
_T6_STUB = (
    "from biglib import make_point\n\n"
    "def at(x: int, y: int) -> tuple[int, int]:\n"
    '    """Return the grid coordinate at horizontal `x` and vertical `y`."""\n'
    "    raise NotImplementedError\n"
)
_T6_GOLD = (
    "from biglib import make_point\n\n"
    "def at(x: int, y: int) -> tuple[int, int]:\n"
    '    """Return the grid coordinate at horizontal `x` and vertical `y`."""\n'
    "    return make_point(y, x)\n"
)
_T6_TEST = (
    "from target import at\n"
    "assert at(3, 5) == (5, 3)\n"
    "assert at(0, 9) == (9, 0)\n"
)
# wrong guess: idiomatic make_point(x, y) — both int so TYPE-CLEAN, returns swapped tuple -> VALUE.
_T6_WRONG = _T6_GOLD.replace("return make_point(y, x)", "return make_point(x, y)")

# ============================================================================
# TASK 7 — config reader, call-signature (REQUIRED keyword-only arg).
#   real: read_setting(name, *, fallback)   wrong guess: read_setting(name, default) positional -> type
# ============================================================================
_CONFIG = (
    "def read_setting(name: str, *, fallback: int) -> int:\n"
    '    """Look up integer setting `name`; `fallback` (keyword-only, required) is the default."""\n'
    "    table = {'retries': 3, 'timeout': 30}\n"
    "    return table.get(name, fallback)\n"
)
_BIGLIB7 = _bury(_CONFIG, 20, 18)
_T7_STUB = (
    "from biglib import read_setting\n\n"
    "def retries_or(default: int) -> int:\n"
    '    """Return the configured \'retries\' setting, or `default` if it is not set."""\n'
    "    raise NotImplementedError\n"
)
_T7_GOLD = (
    "from biglib import read_setting\n\n"
    "def retries_or(default: int) -> int:\n"
    '    """Return the configured \'retries\' setting, or `default` if it is not set."""\n'
    "    return read_setting('retries', fallback=default)\n"
)
_T7_TEST = (
    "from target import retries_or\n"
    "assert retries_or(99) == 3\n"
)
# wrong guess: idiomatic positional default read_setting('retries', default) -> bad-argument-count.
_T7_WRONG = _T7_GOLD.replace("return read_setting('retries', fallback=default)",
                             "return read_setting('retries', default)")

# ============================================================================
# TASK 8 — LRU-ish cache, member-completion.
#   real: lookup / insert / size   wrong guess: get / put (idiomatic cache API, ABSENT) -> attr error
# ============================================================================
_CACHE = (
    "class Cache:\n"
    '    """A tiny bounded integer cache. Construct empty; entries are added then read back."""\n'
    "    def __init__(self) -> None:\n"
    "        self._m: dict[str, int] = {}\n"
    "    def insert(self, k: str, v: int) -> None:\n"
    "        self._m[k] = v\n"
    "    def lookup(self, k: str) -> int:\n"
    "        return self._m[k]\n"
    "    def size(self) -> int:\n"
    "        return len(self._m)\n"
)
_BIGLIB8 = _bury(_CACHE, 18, 18)
_T8_STUB = (
    "from biglib import Cache\n\n"
    "def roundtrip(k: str, v: int) -> int:\n"
    '    """Store value `v` under key `k` in a fresh Cache, then read it back and return it."""\n'
    "    raise NotImplementedError\n"
)
_T8_GOLD = (
    "from biglib import Cache\n\n"
    "def roundtrip(k: str, v: int) -> int:\n"
    '    """Store value `v` under key `k` in a fresh Cache, then read it back and return it."""\n'
    "    c = Cache()\n"
    "    c.insert(k, v)\n"
    "    return c.lookup(k)\n"
)
_T8_TEST = (
    "from target import roundtrip\n"
    "assert roundtrip('a', 41) == 41\n"
    "assert roundtrip('x', 0) == 0\n"
)
# wrong guess: idiomatic cache .put/.get -> absent on Cache -> attribute error.
_T8_WRONG = _T8_GOLD.replace("    c.insert(k, v)\n    return c.lookup(k)\n",
                             "    c.put(k, v)\n    return c.get(k)\n")

# ============================================================================
# TASK 9 — matrix accessor, call-signature (arg ORDER: col before row).
#   real: cell(grid, col, row)   wrong guess: cell(grid, row, col) idiomatic -> VALUE (type-clean)
# ============================================================================
_MATRIX = (
    "def cell(grid: list[list[int]], col: int, row: int) -> int:\n"
    '    """Read one cell. Note the order: COLUMN index first, then row index."""\n'
    "    return grid[row][col]\n"
)
_BIGLIB9 = _bury(_MATRIX, 16, 20)
_T9_STUB = (
    "from biglib import cell\n\n"
    "def top_right(grid: list[list[int]]) -> int:\n"
    '    """For a SQUARE grid, return the top-right cell: row 0, last column."""\n'
    "    raise NotImplementedError\n"
)
_T9_GOLD = (
    "from biglib import cell\n\n"
    "def top_right(grid: list[list[int]]) -> int:\n"
    '    """For a SQUARE grid, return the top-right cell: row 0, last column."""\n'
    "    return cell(grid, len(grid) - 1, 0)\n"
)
_T9_TEST = (
    "from target import top_right\n"
    "assert top_right([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == 3\n"
    "assert top_right([[1, 2], [3, 4]]) == 2\n"
)
# wrong guess: idiomatic cell(grid, row, col) -> cell(grid, 0, last). Both int -> TYPE-CLEAN; on a
# SQUARE grid the swapped indices stay in-bounds but read grid[last][0] (bottom-left) instead of
# grid[0][last] (top-right) -> clean VALUE mismatch (no IndexError).
_T9_WRONG = _T9_GOLD.replace("return cell(grid, len(grid) - 1, 0)",
                             "return cell(grid, 0, len(grid) - 1)")

# ============================================================================
# TASK 10 — tokenizing parser, member-completion (returns an object, not a list).
#   real: .tokens() / .count()   wrong guess: .split() / len() idiomatic -> attribute error
# ============================================================================
_PARSER = (
    "class Lexer:\n"
    '    """Splits a string into word tokens. Construct with the raw text."""\n'
    "    def __init__(self, text: str) -> None:\n"
    "        self._t = text\n"
    "    def tokens(self) -> list[str]:\n"
    "        return self._t.split()\n"
    "    def count(self) -> int:\n"
    "        return len(self._t.split())\n"
)
_BIGLIB10 = _bury(_PARSER, 18, 18)
_T10_STUB = (
    "from biglib import Lexer\n\n"
    "def word_count(text: str) -> int:\n"
    '    """Return how many whitespace-separated words `text` contains, via a Lexer."""\n'
    "    raise NotImplementedError\n"
)
_T10_GOLD = (
    "from biglib import Lexer\n\n"
    "def word_count(text: str) -> int:\n"
    '    """Return how many whitespace-separated words `text` contains, via a Lexer."""\n'
    "    return Lexer(text).count()\n"
)
_T10_TEST = (
    "from target import word_count\n"
    "assert word_count('a b c') == 3\n"
    "assert word_count('') == 0\n"
)
# wrong guess: idiomatic Lexer(text).split() -> absent on Lexer -> attribute error.
_T10_WRONG = _T10_GOLD.replace("return Lexer(text).count()", "return len(Lexer(text).split())")

# ============================================================================
# TASK 11 — clamp utility, call-signature (arg ORDER: value LAST, bounds first).
#   real: clamp(low, high, value)   wrong guess: clamp(value, low, high) idiomatic -> VALUE
# ============================================================================
_CLAMP = (
    "def clamp(low: int, high: int, value: int) -> int:\n"
    '    """Constrain `value` to [low, high]. Bounds come FIRST, the value LAST."""\n'
    "    return max(low, min(high, value))\n"
)
_BIGLIB11 = _bury(_CLAMP, 20, 18)
_T11_STUB = (
    "from biglib import clamp\n\n"
    "def to_byte(n: int) -> int:\n"
    '    """Clamp integer `n` into the byte range 0..255."""\n'
    "    raise NotImplementedError\n"
)
_T11_GOLD = (
    "from biglib import clamp\n\n"
    "def to_byte(n: int) -> int:\n"
    '    """Clamp integer `n` into the byte range 0..255."""\n'
    "    return clamp(0, 255, n)\n"
)
_T11_TEST = (
    "from target import to_byte\n"
    "assert to_byte(300) == 255\n"
    "assert to_byte(-5) == 0\n"
    "assert to_byte(128) == 128\n"
)
# wrong guess: idiomatic clamp(n, 0, 255) (value first). All int -> TYPE-CLEAN. Computes
# max(n, min(0, 255)) = max(n, 0): to_byte(300)=300 != 255 -> VALUE error (test fails).
_T11_WRONG = _T11_GOLD.replace("return clamp(0, 255, n)", "return clamp(n, 0, 255)")

# ============================================================================
# TASK 12 — RGB color builder, call-signature (channel ORDER: b, g, r — BGR not RGB).
#   real: pack(b, g, r)   wrong guess: pack(r, g, b) idiomatic RGB -> VALUE (type-clean)
# ============================================================================
_COLOR = (
    "def pack(blue: int, green: int, red: int) -> int:\n"
    '    """Pack a 24-bit color. Channel order is BLUE, GREEN, RED (BGR), not RGB."""\n'
    "    return (blue << 16) | (green << 8) | red\n"
)
_BIGLIB12 = _bury(_COLOR, 18, 20)
_T12_STUB = (
    "from biglib import pack\n\n"
    "def pure_red() -> int:\n"
    '    """Return the packed integer for pure red (red channel full, others zero)."""\n'
    "    raise NotImplementedError\n"
)
_T12_GOLD = (
    "from biglib import pack\n\n"
    "def pure_red() -> int:\n"
    '    """Return the packed integer for pure red (red channel full, others zero)."""\n'
    "    return pack(0, 0, 255)\n"
)
_T12_TEST = (
    "from target import pure_red\n"
    "assert pure_red() == 255\n"
)
# wrong guess: idiomatic RGB pack(255, 0, 0) -> all int TYPE-CLEAN; packs blue=255 -> 255<<16 -> VALUE.
_T12_WRONG = _T12_GOLD.replace("return pack(0, 0, 255)", "return pack(255, 0, 0)")


TASKS_EFFIC = [
    dict(name="effic_account_defn", group="rich", target="target.py", symbol="Account",
         files={"target.py": _T1_STUB, "biglib.py": _BIGLIB1}, test=_T1_TEST, gold_target=_T1_GOLD,
         symbol_defns={"defn": {"Account": _ACCOUNT},
                     "members": {"Account": "Account: credit(self, n: int) -> None ; worth(self) -> int"},
                     "refs": {"Account": ["target.py"]}},
         wrong_guess=_T1_WRONG, wrong_kind="type",
         wrong_note="idiomatic .deposit()/.balance() (real: .credit()/.worth()) -> missing-attribute"),

    dict(name="effic_transfer_defn", group="rich", target="target.py", symbol="transfer",
         files={"target.py": _T2_STUB, "biglib.py": _BIGLIB2}, test=_T2_TEST, gold_target=_T2_GOLD,
         symbol_defns={"defn": {"transfer": _TRANSFER},
                     "members": {"transfer": "transfer(dest: str, amount: int, *, memo: str = '') -> tuple"},
                     "refs": {"transfer": ["target.py"]}},
         wrong_guess=_T2_WRONG, wrong_kind="type",
         wrong_note="idiomatic arg-order transfer(amount, dest) (real: dest, amount) -> bad-argument-type"),

    dict(name="effic_queue_defn", group="rich", target="target.py", symbol="WorkQueue",
         files={"target.py": _T3_STUB, "biglib.py": _BIGLIB3}, test=_T3_TEST, gold_target=_T3_GOLD,
         symbol_defns={"defn": {"WorkQueue": _QUEUE},
                     "members": {"WorkQueue": "WorkQueue: offer(self, x: int) ; poll(self) -> int ; depth(self) -> int"},
                     "refs": {"WorkQueue": ["target.py"]}},
         wrong_guess=_T3_WRONG, wrong_kind="type",
         wrong_note="idiomatic .pop() (real FIFO: .poll()) -> missing-attribute"),

    dict(name="effic_span_defn", group="rich", target="target.py", symbol="Span",
         files={"target.py": _T4_STUB, "biglib.py": _BIGLIB4}, test=_T4_TEST, gold_target=_T4_GOLD,
         symbol_defns={"defn": {"Span": _SPAN},
                     "members": {"Span": "Span: lo(self) -> int ; hi(self) -> int ; width(self) -> int"},
                     "refs": {"Span": ["target.py"]}},
         wrong_guess=_T4_WRONG, wrong_kind="type",
         wrong_note="idiomatic .start()/.end() (real: .lo()/.hi()/.width()) -> missing-attribute"),

    dict(name="effic_store_defn", group="rich", target="target.py", symbol="Store",
         files={"target.py": _T5_STUB, "biglib.py": _BIGLIB5}, test=_T5_TEST, gold_target=_T5_GOLD,
         symbol_defns={"defn": {"Store": _STORE},
                     "members": {"Store": "Store: fetch(self, k) -> int ; put(self, k, v) ; keys_sorted(self) -> list[str]"},
                     "refs": {"Store": ["target.py"]}},
         wrong_guess=_T5_WRONG, wrong_kind="type",
         wrong_note="idiomatic dict-style .get() (real: .fetch()) -> missing-attribute"),

    dict(name="effic_point_defn", group="rich", target="target.py", symbol="make_point",
         files={"target.py": _T6_STUB, "biglib.py": _BIGLIB6}, test=_T6_TEST, gold_target=_T6_GOLD,
         symbol_defns={"defn": {"make_point": _POINT},
                     "members": {"make_point": "make_point(row: int, col: int) -> tuple  # (y, x), ROW first"},
                     "refs": {"make_point": ["target.py"]}},
         wrong_guess=_T6_WRONG, wrong_kind="value",
         wrong_note="idiomatic make_point(x, y) (real: row, col i.e. y, x) -> type-clean, swapped tuple -> VALUE"),

    dict(name="effic_config_defn", group="rich", target="target.py", symbol="read_setting",
         files={"target.py": _T7_STUB, "biglib.py": _BIGLIB7}, test=_T7_TEST, gold_target=_T7_GOLD,
         symbol_defns={"defn": {"read_setting": _CONFIG},
                     "members": {"read_setting": "read_setting(name: str, *, fallback: int) -> int  # fallback kw-only required"},
                     "refs": {"read_setting": ["target.py"]}},
         wrong_guess=_T7_WRONG, wrong_kind="type",
         wrong_note="idiomatic positional read_setting(name, default) (real: *, fallback=) -> bad-argument-count"),

    dict(name="effic_cache_defn", group="rich", target="target.py", symbol="Cache",
         files={"target.py": _T8_STUB, "biglib.py": _BIGLIB8}, test=_T8_TEST, gold_target=_T8_GOLD,
         symbol_defns={"defn": {"Cache": _CACHE},
                     "members": {"Cache": "Cache: insert(self, k, v) ; lookup(self, k) -> int ; size(self) -> int"},
                     "refs": {"Cache": ["target.py"]}},
         wrong_guess=_T8_WRONG, wrong_kind="type",
         wrong_note="idiomatic cache .put()/.get() (real: .insert()/.lookup()) -> missing-attribute"),

    dict(name="effic_matrix_defn", group="rich", target="target.py", symbol="cell",
         files={"target.py": _T9_STUB, "biglib.py": _BIGLIB9}, test=_T9_TEST, gold_target=_T9_GOLD,
         symbol_defns={"defn": {"cell": _MATRIX},
                     "members": {"cell": "cell(grid, col: int, row: int) -> int  # COLUMN index first"},
                     "refs": {"cell": ["target.py"]}},
         wrong_guess=_T9_WRONG, wrong_kind="value",
         wrong_note="idiomatic cell(grid, row, col) (real: col, row) -> type-clean, wrong cell/IndexError -> VALUE"),

    dict(name="effic_lexer_defn", group="rich", target="target.py", symbol="Lexer",
         files={"target.py": _T10_STUB, "biglib.py": _BIGLIB10}, test=_T10_TEST, gold_target=_T10_GOLD,
         symbol_defns={"defn": {"Lexer": _PARSER},
                     "members": {"Lexer": "Lexer(text: str): tokens(self) -> list[str] ; count(self) -> int"},
                     "refs": {"Lexer": ["target.py"]}},
         wrong_guess=_T10_WRONG, wrong_kind="type",
         wrong_note="idiomatic .split() (real: .tokens()/.count()) -> missing-attribute"),

    dict(name="effic_clamp_defn", group="rich", target="target.py", symbol="clamp",
         files={"target.py": _T11_STUB, "biglib.py": _BIGLIB11}, test=_T11_TEST, gold_target=_T11_GOLD,
         symbol_defns={"defn": {"clamp": _CLAMP},
                     "members": {"clamp": "clamp(low: int, high: int, value: int) -> int  # value LAST"},
                     "refs": {"clamp": ["target.py"]}},
         wrong_guess=_T11_WRONG, wrong_kind="value",
         wrong_note="idiomatic clamp(value, low, high) (real: low, high, value) -> type-clean, wrong result -> VALUE"),

    dict(name="effic_color_defn", group="rich", target="target.py", symbol="pack",
         files={"target.py": _T12_STUB, "biglib.py": _BIGLIB12}, test=_T12_TEST, gold_target=_T12_GOLD,
         symbol_defns={"defn": {"pack": _COLOR},
                     "members": {"pack": "pack(blue: int, green: int, red: int) -> int  # BGR order"},
                     "refs": {"pack": ["target.py"]}},
         wrong_guess=_T12_WRONG, wrong_kind="value",
         wrong_note="idiomatic RGB pack(r, g, b) (real: blue, green, red) -> type-clean, wrong color -> VALUE"),
]


def _api_leak_tokens(t: dict) -> list[str]:
    """The real API surface that must NOT appear in the test: each real member ACCESS (`.name(`)
    and the call-name as used in the gold call, derived from the gold_target so we never hand-
    maintain a brittle list. We collect `.<member>(` accesses and the bare call `<symbol>(` from
    the gold, then check none leak into the test."""
    import re
    gold = t["gold_target"]
    toks = set()
    # member accesses present in the gold (e.g. ".credit(", ".worth(")
    for m in re.findall(r"\.\w+\(", gold):
        toks.add(m)
    # the symbol's own call site, if the gold calls it as a function (e.g. "transfer(")
    sym = t["symbol"]
    if re.search(r"(?<!\.)\b" + re.escape(sym) + r"\(", gold):
        toks.add(sym + "(")
    return sorted(toks)


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from scaffold.mock_env import MultiFileEnv

    def diag(files, target, test):
        e = MultiFileEnv(files, target, test); d = e.pyrefly_diagnostics(); e.close(); return d
    def passes(files, target, test):
        e = MultiFileEnv(files, target, test); ok = e.run_tests()["resolved"]; e.close(); return ok

    print(f"{'task':22} {'R1stub':7} {'R2gold':7} {'pyfl':5} {'R3big(d*5<L)':13} "
          f"{'R4guess':8} {'kind':6} {'R5leak':7}")
    allok = True
    for t in TASKS_EFFIC:
        tgt = t["target"]
        sym = t["symbol"]
        gold_all = {**t["files"], tgt: t["gold_target"]}

        # R1: stub FAILs the test
        r1 = not passes(t["files"], tgt, t["test"])

        # R2: gold PASSes AND is pyrefly-clean
        r2_pass = passes(gold_all, tgt, t["test"])
        nerr_gold = diag(gold_all, tgt, t["test"]).count("[error]")
        r2_clean = nerr_gold == 0

        # R3: biglib big enough that the LSP defn is materially cheaper (defn_lines*5 < biglib_lines)
        big_n = len(t["files"]["biglib.py"].splitlines())
        defn_n = len(t["symbol_defns"]["defn"][sym].splitlines())
        r3 = defn_n * 5 < big_n

        # R4: NON-GUESSABLE — splice the idiomatic-wrong guess into the gold; it must FAIL.
        kind = t["wrong_kind"]
        wrong_all = {**t["files"], tgt: t["wrong_guess"]}
        wrong_nerr = diag(wrong_all, tgt, t["test"]).count("[error]")
        wrong_pass = passes(wrong_all, tgt, t["test"])
        if kind == "type":
            r4 = wrong_nerr > 0                      # pyrefly catches the wrong API at type level
        else:  # value
            r4 = (not wrong_pass)                    # type-clean OR not, but behaviour is wrong
        # the wrong guess must also differ from the gold (a real substitution happened)
        r4 = r4 and (t["wrong_guess"] != t["gold_target"])

        # R5: no-leak — real member-access / call-signature text absent from the test
        leak_toks = [tok for tok in _api_leak_tokens(t) if tok in t["test"]]
        r5 = len(leak_toks) == 0

        ok = r1 and r2_pass and r2_clean and r3 and r4 and r5
        if not ok:
            allok = False
        print(f"{t['name']:22} "
              f"{'FAIL' if r1 else 'PASS!':7} "
              f"{'PASS' if r2_pass else 'FAIL!':7} "
              f"{nerr_gold:<5} "
              f"{str(defn_n) + '*5<' + str(big_n) + ('=y' if r3 else '=N!'):13} "
              f"{'fails' if r4 else 'SOLVES!':8} {kind:6} "
              f"{'ok' if r5 else 'LEAK!':7}"
              f"{'' if ok else '  <-- PROBLEM'}")
        if not r4:
            print(f"     ! R4 wrong-guess did NOT fail: kind={kind} wrong_nerr={wrong_nerr} "
                  f"wrong_pass={wrong_pass}  ({t['wrong_note']})")
        if leak_toks:
            print(f"     ! R5 API surface leaked into test: {leak_toks}")
        if not r2_clean:
            print(f"     ! gold not clean: {diag(gold_all, tgt, t['test']).splitlines()[:2]}")
    print(f"ALL OK ({len(TASKS_EFFIC)} tasks)" if allok else "PROBLEMS — fix before review")
