#!/usr/bin/env python3
"""AUTHORING suite (Exp 2) — does a static type checker help a coding agent when it is
WRITING a new module from a spec, rather than fixing a small bug?

MOTIVATION. Section 5 (gapd2) found the checker's information REDUNDANT on small held-out
bug fixes: the wrong one-line fix that a strong agent produces is usually well-typed, so
pyrefly stays silent and the behavioural test is the real detector. Exp 2 asks the opposite
question at the opposite end of the size axis: when the agent AUTHORS a larger module — several
interacting typed functions/classes that must call a provided typed `lib.py` correctly — it
tends to make ORGANIC type errors as a by-product of writing a lot of code at once:
  undefined names, wrong call signatures / arity, bad imports, attribute typos, wrong
  TypedDict keys, protocol non-conformance, generic/Callable mismatches.
These are exactly the mistakes a static checker names for free, BEFORE any test is run. Exp 2
measures whether surfacing them (arms `check` / `feedback`) changes held-out correctness or the
residual type errors on the final submission, versus a no-checker baseline (`none`).

TASK SHAPE (schema-compatible with gapd2 / synth_mf / api_agent — same keys consumed by the
harness: name, group, target, files, test, held_out, gold_target):
  files["target.py"]  a LARGER stub — several typed signatures with docstring specs, every
                      body `raise NotImplementedError`. The agent implements ALL of them.
  files["lib.py"]     a provided, typed API the target MUST call correctly (misusing it is a
                      static type error). Read-only for the agent (only target.py is editable).
  test                VISIBLE behavioural spec the agent may run via <test>.
  held_out            HELD-OUT behavioural oracle (scores correctness; the agent never runs it).
  gold_target         a correct full implementation, kept for validation, NOT shown to the agent.
  type_wrong          a deliberately type-WRONG implementation used ONLY by GATE A to prove the
                      task has real type-error surface (a checker CAN catch mistakes here).
  note                one-line description of the type surface exercised.

Unlike gapd2 there is no single designed `wrong_body`: correctness is judged purely by the
held-out test and type-cleanliness purely by pyrefly. 12 tasks spanning TypedDict, dataclasses,
exceptions, enums, NamedTuple, Protocols, generics (Generic/TypeVar), Callable, and nested
containers, from easy to genuinely tricky. Stdlib only — tests run on the bare interpreter.

GATE A (no model), enforced by __main__ for every task:
  A  stub:  held-out score() FAILS (nothing is implemented).
  B  gold:  VISIBLE test PASSES and HELD-OUT test PASSES.
  C  gold:  pyrefly is CLEAN (no diagnostics in target.py or lib.py).
  D  the task has type-error surface: pyrefly on the STUB and/or the type_wrong sketch
     surfaces >= 1 diagnostic in target.py (so a checker could catch an organic mistake).

Run the verifier with .venv-streams.system/bin/python (one pyrefly process at a time;
pkill -9 -f "[p]yrefly" between stale daemons if needed).
"""
from textwrap import dedent


def S(s: str) -> str:
    """Dedent a triple-quoted block and normalise to a single trailing newline."""
    return dedent(s).strip("\n") + "\n"


def _task(name, group, lib, stub, gold, type_wrong, test, held_out, note):
    files = {"target.py": stub}
    if lib is not None:
        files["lib.py"] = lib
    return dict(
        name=name, group=group, target="target.py", files=files,
        test=test, held_out=held_out, gold_target=gold, type_wrong=type_wrong,
        note=note)


# ===================================================================================
# T1 (easy) — shopping cart over a TypedDict Item; misuse = bad key / wrong arity.
# ===================================================================================
_T1 = _task(
    "auth_cart_typeddict", "easy",
    lib=S('''
        from typing import TypedDict


        class Item(TypedDict):
            name: str
            qty: int
            price: int


        def line_total(item: Item) -> int:
            """Total cents for one line: quantity times unit price."""
            return item["qty"] * item["price"]
    '''),
    stub=S('''
        from lib import Item, line_total


        def make_item(name: str, qty: int, price: int) -> Item:
            """Build an Item from its fields."""
            raise NotImplementedError


        def cart_total(items: list[Item]) -> int:
            """Total cents across every item's line_total."""
            raise NotImplementedError


        def priciest(items: list[Item]) -> str:
            """Name of the item with the largest line_total. `items` is non-empty;
            on a tie return the earliest such item."""
            raise NotImplementedError
    '''),
    gold=S('''
        from lib import Item, line_total


        def make_item(name: str, qty: int, price: int) -> Item:
            """Build an Item from its fields."""
            return {"name": name, "qty": qty, "price": price}


        def cart_total(items: list[Item]) -> int:
            """Total cents across every item's line_total."""
            return sum(line_total(it) for it in items)


        def priciest(items: list[Item]) -> str:
            """Name of the item with the largest line_total. `items` is non-empty;
            on a tie return the earliest such item."""
            return max(items, key=line_total)["name"]
    '''),
    type_wrong=S('''
        from lib import Item, line_total


        def make_item(name: str, qty: int, price: int) -> Item:
            return {"name": name, "count": qty, "price": price}


        def cart_total(items: list[Item]) -> int:
            return sum(line_total(it, 0) for it in items)


        def priciest(items: list[Item]) -> str:
            return max(items, key=line_total)["name"]
    '''),
    test=S('''
        from target import make_item, cart_total, priciest
        a = make_item("pen", 2, 150)
        b = make_item("mug", 1, 500)
        assert cart_total([a, b]) == 800
        assert priciest([a, b]) == "mug"
    '''),
    held_out=S('''
        from target import make_item, cart_total, priciest
        assert cart_total([]) == 0
        one = make_item("x", 3, 100)
        assert priciest([one]) == "x"
        two = make_item("y", 5, 60)
        three = make_item("z", 3, 100)
        assert priciest([two, three]) == "y"
        assert cart_total([two, three]) == 600
    '''),
    note="TypedDict Item + helper; organic errors: bad TypedDict key, wrong call arity.")


# ===================================================================================
# T2 (easy-med) — a Bank of dataclass Accounts; misuse = attribute/method typos.
# ===================================================================================
_T2 = _task(
    "auth_bank_dataclass", "easy",
    lib=S('''
        from dataclasses import dataclass


        class InsufficientFunds(Exception):
            """Raised when an account cannot cover a debit."""


        @dataclass
        class Account:
            owner: str
            balance: int = 0

            def deposit(self, amount: int) -> None:
                self.balance += amount

            def withdraw(self, amount: int) -> None:
                if amount > self.balance:
                    raise InsufficientFunds(self.owner)
                self.balance -= amount
    '''),
    stub=S('''
        from lib import Account, InsufficientFunds


        class Bank:
            """A collection of named accounts."""

            def __init__(self) -> None:
                raise NotImplementedError

            def open(self, owner: str) -> Account:
                """Create and store a zero-balance account for owner; return it."""
                raise NotImplementedError

            def deposit(self, owner: str, amount: int) -> None:
                """Deposit into an existing account."""
                raise NotImplementedError

            def transfer(self, src: str, dst: str, amount: int) -> bool:
                """Move amount from src to dst. Return False (and change nothing) if
                src has insufficient funds; return True on success."""
                raise NotImplementedError

            def total_assets(self) -> int:
                """Sum of all account balances."""
                raise NotImplementedError
    '''),
    gold=S('''
        from lib import Account, InsufficientFunds


        class Bank:
            """A collection of named accounts."""

            def __init__(self) -> None:
                self._accounts: dict[str, Account] = {}

            def open(self, owner: str) -> Account:
                acct = Account(owner)
                self._accounts[owner] = acct
                return acct

            def deposit(self, owner: str, amount: int) -> None:
                self._accounts[owner].deposit(amount)

            def transfer(self, src: str, dst: str, amount: int) -> bool:
                source = self._accounts[src]
                try:
                    source.withdraw(amount)
                except InsufficientFunds:
                    return False
                self._accounts[dst].deposit(amount)
                return True

            def total_assets(self) -> int:
                return sum(a.balance for a in self._accounts.values())
    '''),
    type_wrong=S('''
        from lib import Account, InsufficientFunds


        class Bank:
            def __init__(self) -> None:
                self._accounts: dict[str, Account] = {}

            def open(self, owner: str) -> Account:
                acct = Account(owner)
                self._accounts[owner] = acct
                return acct

            def deposit(self, owner: str, amount: int) -> None:
                self._accounts[owner].credit(amount)

            def transfer(self, src: str, dst: str, amount: int) -> bool:
                source = self._accounts[src]
                try:
                    source.withdraw(amount)
                except InsufficientFunds:
                    return False
                self._accounts[dst].deposit(amount)
                return True

            def total_assets(self) -> int:
                return sum(a.funds for a in self._accounts.values())
    '''),
    test=S('''
        from target import Bank
        b = Bank()
        b.open("alice")
        b.open("bob")
        b.deposit("alice", 100)
        assert b.transfer("alice", "bob", 40) is True
        assert b.total_assets() == 100
    '''),
    held_out=S('''
        from target import Bank
        b = Bank()
        b.open("alice")
        b.open("bob")
        b.deposit("alice", 30)
        assert b.transfer("alice", "bob", 50) is False
        assert b.total_assets() == 30
        assert b.transfer("alice", "bob", 30) is True
        assert b.total_assets() == 30
    '''),
    note="dataclass Account + custom Exception; organic errors: attribute/method typos.")


# ===================================================================================
# T3 (med) — a generic MultiMap[K, V] built on a provided Pair[K, V]; generics surface.
# ===================================================================================
_T3 = _task(
    "auth_multimap_generic", "med",
    lib=S('''
        from typing import Generic, TypeVar

        K = TypeVar("K")
        V = TypeVar("V")


        class Pair(Generic[K, V]):
            """An immutable key/value pair."""

            def __init__(self, key: K, value: V) -> None:
                self.key = key
                self.value = value
    '''),
    stub=S('''
        from typing import Generic, TypeVar
        from lib import Pair

        K = TypeVar("K")
        V = TypeVar("V")


        class MultiMap(Generic[K, V]):
            """A map from each key to a list of values, in insertion order."""

            def __init__(self) -> None:
                raise NotImplementedError

            def add(self, key: K, value: V) -> None:
                """Append value to key's list (creating the list if key is new)."""
                raise NotImplementedError

            def get(self, key: K) -> list[V]:
                """Values for key in insertion order; empty list if key is absent."""
                raise NotImplementedError

            def pairs(self) -> list[Pair[K, V]]:
                """Every (key, value) as a Pair. Keys in first-insertion order; within a
                key, values in the order they were added."""
                raise NotImplementedError
    '''),
    gold=S('''
        from typing import Generic, TypeVar
        from lib import Pair

        K = TypeVar("K")
        V = TypeVar("V")


        class MultiMap(Generic[K, V]):
            """A map from each key to a list of values, in insertion order."""

            def __init__(self) -> None:
                self._data: dict[K, list[V]] = {}

            def add(self, key: K, value: V) -> None:
                self._data.setdefault(key, []).append(value)

            def get(self, key: K) -> list[V]:
                return list(self._data.get(key, []))

            def pairs(self) -> list[Pair[K, V]]:
                out: list[Pair[K, V]] = []
                for key, values in self._data.items():
                    for value in values:
                        out.append(Pair(key, value))
                return out
    '''),
    type_wrong=S('''
        from typing import Generic, TypeVar
        from lib import Pair

        K = TypeVar("K")
        V = TypeVar("V")


        class MultiMap(Generic[K, V]):
            def __init__(self) -> None:
                self._data: dict[K, list[V]] = {}

            def add(self, key: K, value: V) -> None:
                self._data.setdefault(key, []).append(value)

            def get(self, key: K) -> list[V]:
                return list(self._data.get(key, []))

            def pairs(self) -> list[Pair[K, V]]:
                out: list[Pair[K, V]] = []
                for key, values in self._data.items():
                    for value in values:
                        out.append(Pair(key))
                return out
    '''),
    test=S('''
        from target import MultiMap
        m: MultiMap[str, int] = MultiMap()
        m.add("a", 1)
        m.add("a", 2)
        m.add("b", 3)
        assert m.get("a") == [1, 2]
        assert m.get("z") == []
    '''),
    held_out=S('''
        from target import MultiMap
        m: MultiMap[str, int] = MultiMap()
        m.add("x", 10)
        m.add("y", 20)
        m.add("x", 11)
        ps = m.pairs()
        assert [(p.key, p.value) for p in ps] == [("x", 10), ("x", 11), ("y", 20)]
    '''),
    note="Generic[K, V] container over provided Pair; organic error: missing ctor argument.")


# ===================================================================================
# T4 (med) — concrete shapes conforming to a provided Protocol; structural typing surface.
# ===================================================================================
_T4 = _task(
    "auth_shapes_protocol", "med",
    lib=S('''
        from typing import Protocol


        class Shape(Protocol):
            def area(self) -> float: ...
            def label(self) -> str: ...


        def summary(shapes: list[Shape]) -> str:
            """One 'label=area' term per shape (area to 1 dp), joined by ', '."""
            return ", ".join(f"{s.label()}={s.area():.1f}" for s in shapes)
    '''),
    stub=S('''
        from lib import Shape, summary


        class Circle:
            """A circle of the given radius."""

            def __init__(self, radius: float) -> None:
                raise NotImplementedError

            def area(self) -> float:
                raise NotImplementedError

            def label(self) -> str:
                raise NotImplementedError


        class Square:
            """A square of the given side length."""

            def __init__(self, side: float) -> None:
                raise NotImplementedError

            def area(self) -> float:
                raise NotImplementedError

            def label(self) -> str:
                raise NotImplementedError


        def report(shapes: list[Shape]) -> str:
            """Delegate to lib.summary."""
            raise NotImplementedError


        def default_report() -> str:
            """summary of a unit Circle followed by a unit Square."""
            raise NotImplementedError
    '''),
    gold=S('''
        import math
        from lib import Shape, summary


        class Circle:
            """A circle of the given radius."""

            def __init__(self, radius: float) -> None:
                self.radius = radius

            def area(self) -> float:
                return math.pi * self.radius * self.radius

            def label(self) -> str:
                return "circle"


        class Square:
            """A square of the given side length."""

            def __init__(self, side: float) -> None:
                self.side = side

            def area(self) -> float:
                return self.side * self.side

            def label(self) -> str:
                return "square"


        def report(shapes: list[Shape]) -> str:
            return summary(shapes)


        def default_report() -> str:
            shapes: list[Shape] = [Circle(1.0), Square(1.0)]
            return summary(shapes)
    '''),
    type_wrong=S('''
        import math
        from lib import Shape, summary


        class Circle:
            def __init__(self, radius: float) -> None:
                self.radius = radius

            def size(self) -> float:
                return math.pi * self.radius * self.radius

            def label(self) -> str:
                return "circle"


        class Square:
            def __init__(self, side: float) -> None:
                self.side = side

            def area(self) -> float:
                return self.side * self.side

            def label(self) -> str:
                return "square"


        def report(shapes: list[Shape]) -> str:
            return summary(shapes)


        def default_report() -> str:
            shapes: list[Shape] = [Circle(1.0), Square(1.0)]
            return summary(shapes)
    '''),
    test=S('''
        from target import Circle, Square, report, default_report
        r = report([Circle(1.0), Square(2.0)])
        assert "circle=3.1" in r
        assert "square=4.0" in r
        assert default_report().count("=") == 2
    '''),
    held_out=S('''
        from target import Circle, Square, report
        shapes = [Square(3.0), Circle(2.0)]
        assert report(shapes) == "square=9.0, circle=12.6"
    '''),
    note="Protocol Shape; organic error: method misnamed -> concrete type breaks conformance.")


# ===================================================================================
# T5 (med) — a finite-state Machine over a provided Enum + transition table.
# ===================================================================================
_T5 = _task(
    "auth_machine_enum", "med",
    lib=S('''
        from enum import Enum


        class State(Enum):
            IDLE = "idle"
            RUNNING = "running"
            DONE = "done"


        TRANSITIONS: dict[State, set[State]] = {
            State.IDLE: {State.RUNNING},
            State.RUNNING: {State.DONE, State.IDLE},
            State.DONE: set(),
        }
    '''),
    stub=S('''
        from lib import State, TRANSITIONS


        class Machine:
            """A finite-state machine over State, starting in IDLE."""

            def __init__(self) -> None:
                raise NotImplementedError

            def current(self) -> State:
                """The current state."""
                raise NotImplementedError

            def can(self, target: State) -> bool:
                """True if a transition from the current state to target is allowed."""
                raise NotImplementedError

            def to(self, target: State) -> None:
                """Perform the transition; raise ValueError if it is not allowed."""
                raise NotImplementedError
    '''),
    gold=S('''
        from lib import State, TRANSITIONS


        class Machine:
            """A finite-state machine over State, starting in IDLE."""

            def __init__(self) -> None:
                self._state = State.IDLE

            def current(self) -> State:
                return self._state

            def can(self, target: State) -> bool:
                return target in TRANSITIONS[self._state]

            def to(self, target: State) -> None:
                if target not in TRANSITIONS[self._state]:
                    raise ValueError(f"cannot go {self._state} -> {target}")
                self._state = target
    '''),
    type_wrong=S('''
        from lib import State, TRANSITIONS


        class Machine:
            def __init__(self) -> None:
                self._state = State.IDLE

            def current(self) -> State:
                return self._state

            def can(self, target: State) -> bool:
                return target in TRANSITIONS["running"]

            def to(self, target: State) -> None:
                if target not in TRANSITIONS[self._state]:
                    raise ValueError(f"cannot go {self._state} -> {target}")
                self._state = target
    '''),
    test=S('''
        from target import Machine
        from lib import State
        m = Machine()
        assert m.current() == State.IDLE
        assert m.can(State.RUNNING) is True
        m.to(State.RUNNING)
        assert m.current() == State.RUNNING
    '''),
    held_out=S('''
        from target import Machine
        from lib import State
        m = Machine()
        m.to(State.RUNNING)
        assert m.can(State.IDLE) is True
        assert m.can(State.DONE) is True
        m.to(State.DONE)
        assert m.can(State.RUNNING) is False
        raised = False
        try:
            m.to(State.RUNNING)
        except ValueError:
            raised = True
        assert raised
    '''),
    note="Enum + dict[State, set[State]] table; organic error: str key into a State-keyed dict.")


# ===================================================================================
# T6 (easy-med) — a tokenizer producing a provided NamedTuple Token.
# ===================================================================================
_T6 = _task(
    "auth_tokenizer_namedtuple", "easy",
    lib=S('''
        from typing import NamedTuple


        class Token(NamedTuple):
            kind: str
            value: int
    '''),
    stub=S('''
        from lib import Token


        def tokenize(text: str) -> list[Token]:
            """Split on whitespace. A word of only digits -> Token('num', int(word));
            any other word -> Token('word', len(word))."""
            raise NotImplementedError


        def sum_nums(tokens: list[Token]) -> int:
            """Sum the values of all 'num' tokens."""
            raise NotImplementedError
    '''),
    gold=S('''
        from lib import Token


        def tokenize(text: str) -> list[Token]:
            out: list[Token] = []
            for word in text.split():
                if word.isdigit():
                    out.append(Token("num", int(word)))
                else:
                    out.append(Token("word", len(word)))
            return out


        def sum_nums(tokens: list[Token]) -> int:
            return sum(t.value for t in tokens if t.kind == "num")
    '''),
    type_wrong=S('''
        from lib import Token


        def tokenize(text: str) -> list[Token]:
            out: list[Token] = []
            for word in text.split():
                if word.isdigit():
                    out.append(Token("num", int(word)))
                else:
                    out.append(Token("word", len(word)))
            return out


        def sum_nums(tokens: list[Token]) -> int:
            return sum(t.val for t in tokens if t.kind == "num")
    '''),
    test=S('''
        from target import tokenize, sum_nums
        toks = tokenize("cat 12 dog 30")
        assert sum_nums(toks) == 42
    '''),
    held_out=S('''
        from target import tokenize, sum_nums
        assert tokenize("") == []
        assert sum_nums(tokenize("")) == 0
        toks = tokenize("7 hi 3")
        assert [t.kind for t in toks] == ["num", "word", "num"]
        assert sum_nums(toks) == 10
    '''),
    note="NamedTuple Token; organic error: attribute typo on the tuple field.")


# ===================================================================================
# T7 (tricky) — build small functions on a provided generic Callable fold.
# ===================================================================================
_T7 = _task(
    "auth_fold_callable", "tricky",
    lib=S('''
        from typing import Callable, TypeVar

        T = TypeVar("T")
        U = TypeVar("U")


        def fold(items: list[T], init: U, step: Callable[[U, T], U]) -> U:
            """Left fold: thread step(acc, item) across items starting from init."""
            acc = init
            for item in items:
                acc = step(acc, item)
            return acc
    '''),
    stub=S('''
        from lib import fold


        def total(nums: list[int]) -> int:
            """Sum of nums, computed with fold."""
            raise NotImplementedError


        def concat(words: list[str]) -> str:
            """Join words with no separator, computed with fold from an empty string."""
            raise NotImplementedError


        def count_matching(items: list[int], threshold: int) -> int:
            """How many items are strictly greater than threshold, computed with fold."""
            raise NotImplementedError
    '''),
    gold=S('''
        from lib import fold


        def total(nums: list[int]) -> int:
            return fold(nums, 0, lambda acc, n: acc + n)


        def concat(words: list[str]) -> str:
            return fold(words, "", lambda acc, w: acc + w)


        def count_matching(items: list[int], threshold: int) -> int:
            return fold(items, 0, lambda acc, n: acc + (1 if n > threshold else 0))
    '''),
    type_wrong=S('''
        from lib import fold


        def total(nums: list[int]) -> int:
            return fold(nums, 0, lambda acc, n: acc + str(n))


        def concat(words: list[str]) -> str:
            return fold(words, "", lambda acc, w: acc + w)


        def count_matching(items: list[int], threshold: int) -> int:
            return fold(items, 0, lambda acc, n: acc + (1 if n > threshold else 0))
    '''),
    test=S('''
        from target import total, concat, count_matching
        assert total([1, 2, 3]) == 6
        assert concat(["a", "b", "c"]) == "abc"
        assert count_matching([1, 5, 3, 8], 3) == 2
    '''),
    held_out=S('''
        from target import total, concat, count_matching
        assert total([]) == 0
        assert concat([]) == ""
        assert count_matching([], 0) == 0
        assert count_matching([3, 3, 4], 3) == 1
    '''),
    note="generic Callable[[U,T],U] fold; organic error: step body has a type-mismatched op.")


# ===================================================================================
# T8 (tricky) — a directed weighted Graph built from a provided Edge class.
# ===================================================================================
_T8 = _task(
    "auth_graph_edges", "tricky",
    lib=S('''
        class Edge:
            """A weighted directed edge."""

            def __init__(self, src: str, dst: str, weight: int) -> None:
                self.src = src
                self.dst = dst
                self.weight = weight
    '''),
    stub=S('''
        from lib import Edge


        class Graph:
            """A directed weighted graph accumulated from edges."""

            def __init__(self) -> None:
                raise NotImplementedError

            def add(self, edge: Edge) -> None:
                """Register an edge."""
                raise NotImplementedError

            def neighbors(self, node: str) -> list[str]:
                """Destinations one edge from node, in the order edges were added."""
                raise NotImplementedError

            def cost(self, src: str, dst: str) -> int | None:
                """Weight of the src->dst edge, or None if there is no such edge."""
                raise NotImplementedError

            def out_degree(self, node: str) -> int:
                """Number of edges leaving node."""
                raise NotImplementedError
    '''),
    gold=S('''
        from lib import Edge


        class Graph:
            """A directed weighted graph accumulated from edges."""

            def __init__(self) -> None:
                self._edges: list[Edge] = []

            def add(self, edge: Edge) -> None:
                self._edges.append(edge)

            def neighbors(self, node: str) -> list[str]:
                return [e.dst for e in self._edges if e.src == node]

            def cost(self, src: str, dst: str) -> int | None:
                for e in self._edges:
                    if e.src == src and e.dst == dst:
                        return e.weight
                return None

            def out_degree(self, node: str) -> int:
                return sum(1 for e in self._edges if e.src == node)
    '''),
    type_wrong=S('''
        from lib import Edge


        class Graph:
            def __init__(self) -> None:
                self._edges: list[Edge] = []

            def add(self, edge: Edge) -> None:
                self._edges.append(edge)

            def neighbors(self, node: str) -> list[str]:
                return [e.target for e in self._edges if e.src == node]

            def cost(self, src: str, dst: str) -> int | None:
                for e in self._edges:
                    if e.src == src and e.dst == dst:
                        return e.weight
                return None

            def out_degree(self, node: str) -> int:
                return sum(1 for e in self._edges if e.src == node)
    '''),
    test=S('''
        from target import Graph
        from lib import Edge
        g = Graph()
        g.add(Edge("a", "b", 5))
        g.add(Edge("a", "c", 2))
        assert g.neighbors("a") == ["b", "c"]
        assert g.cost("a", "b") == 5
        assert g.out_degree("a") == 2
    '''),
    held_out=S('''
        from target import Graph
        from lib import Edge
        g = Graph()
        g.add(Edge("a", "b", 5))
        assert g.cost("a", "z") is None
        assert g.neighbors("z") == []
        assert g.out_degree("z") == 0
    '''),
    note="interacting Edge/Graph; organic error: attribute typo on Edge (.target vs .dst).")


# ===================================================================================
# T9 (easy-med) — word histogram over Counter + a provided top_n helper.
# ===================================================================================
_T9 = _task(
    "auth_histogram_counter", "easy",
    lib=S('''
        from collections import Counter


        def top_n(counts: "Counter[str]", n: int) -> list[str]:
            """The n most common keys, most-common first (ties by insertion order)."""
            return [k for k, _ in counts.most_common(n)]
    '''),
    stub=S('''
        from collections import Counter
        from lib import top_n


        def word_counts(words: list[str]) -> "Counter[str]":
            """Count occurrences of each word."""
            raise NotImplementedError


        def most_common_word(words: list[str]) -> str:
            """The single most common word. `words` is non-empty."""
            raise NotImplementedError


        def rank(words: list[str], k: int) -> list[str]:
            """The k most common words, via lib.top_n."""
            raise NotImplementedError
    '''),
    gold=S('''
        from collections import Counter
        from lib import top_n


        def word_counts(words: list[str]) -> "Counter[str]":
            return Counter(words)


        def most_common_word(words: list[str]) -> str:
            return word_counts(words).most_common(1)[0][0]


        def rank(words: list[str], k: int) -> list[str]:
            return top_n(word_counts(words), k)
    '''),
    type_wrong=S('''
        from collections import Counter
        from lib import top_n


        def word_counts(words: list[str]) -> "Counter[str]":
            return Counter(words)


        def most_common_word(words: list[str]) -> str:
            return word_counts(words).most_common(1)[0][0]


        def rank(words: list[str], k: int) -> list[str]:
            return top_n(words, k)
    '''),
    test=S('''
        from target import word_counts, most_common_word, rank
        ws = ["a", "b", "a", "c", "a", "b"]
        assert word_counts(ws)["a"] == 3
        assert most_common_word(ws) == "a"
        assert rank(ws, 2) == ["a", "b"]
    '''),
    held_out=S('''
        from target import word_counts, most_common_word, rank
        assert dict(word_counts([])) == {}
        assert rank(["x"], 5) == ["x"]
        assert most_common_word(["solo"]) == "solo"
    '''),
    note="Counter[str] + top_n helper; organic error: pass list where Counter is required.")


# ===================================================================================
# T10 (tricky) — an int->int Pipeline over a provided Callable Handler + apply_all.
# ===================================================================================
_T10 = _task(
    "auth_pipeline_handler", "tricky",
    lib=S('''
        from typing import Callable

        Handler = Callable[[int], int]


        def apply_all(handlers: list[Handler], value: int) -> int:
            """Apply each handler to value in turn, threading the result."""
            for h in handlers:
                value = h(value)
            return value
    '''),
    stub=S('''
        from lib import Handler, apply_all


        class Pipeline:
            """An ordered pipeline of int->int stages."""

            def __init__(self) -> None:
                raise NotImplementedError

            def register(self, handler: Handler) -> None:
                """Append a stage to the end of the pipeline."""
                raise NotImplementedError

            def run(self, value: int) -> int:
                """Thread value through every registered stage via apply_all."""
                raise NotImplementedError


        def adder(n: int) -> Handler:
            """A stage that adds n to its input."""
            raise NotImplementedError
    '''),
    gold=S('''
        from lib import Handler, apply_all


        class Pipeline:
            """An ordered pipeline of int->int stages."""

            def __init__(self) -> None:
                self._stages: list[Handler] = []

            def register(self, handler: Handler) -> None:
                self._stages.append(handler)

            def run(self, value: int) -> int:
                return apply_all(self._stages, value)


        def adder(n: int) -> Handler:
            def stage(x: int) -> int:
                return x + n
            return stage
    '''),
    type_wrong=S('''
        from lib import Handler, apply_all


        class Pipeline:
            def __init__(self) -> None:
                self._stages: list[Handler] = []

            def register(self, handler: Handler) -> None:
                self._stages.append(handler)

            def run(self, value: int) -> int:
                return apply_all(self._stages, value)


        def adder(n: int) -> Handler:
            def stage(x: int) -> str:
                return str(x + n)
            return stage
    '''),
    test=S('''
        from target import Pipeline, adder
        p = Pipeline()
        p.register(adder(2))
        p.register(adder(3))
        assert p.run(0) == 5
    '''),
    held_out=S('''
        from target import Pipeline, adder
        p = Pipeline()
        assert p.run(7) == 7
        p.register(adder(-1))
        p.register(adder(-1))
        assert p.run(10) == 8
    '''),
    note="Callable Handler alias + apply_all; organic error: stage returns str, breaks Handler.")


# ===================================================================================
# T11 (easy-med) — helpers over a provided fixed-size Grid class.
# ===================================================================================
_T11 = _task(
    "auth_grid_helpers", "med",
    lib=S('''
        class Grid:
            """A fixed-size grid of integers, zero-initialised."""

            def __init__(self, rows: int, cols: int) -> None:
                self.rows = rows
                self.cols = cols
                self._cells: list[list[int]] = [[0] * cols for _ in range(rows)]

            def set(self, r: int, c: int, value: int) -> None:
                self._cells[r][c] = value

            def get(self, r: int, c: int) -> int:
                return self._cells[r][c]
    '''),
    stub=S('''
        from lib import Grid


        def from_pairs(rows: int, cols: int, pairs: list[tuple[int, int, int]]) -> Grid:
            """Build a Grid and set each (r, c, value) triple on it."""
            raise NotImplementedError


        def row_sum(grid: Grid, r: int) -> int:
            """Sum of the values in row r."""
            raise NotImplementedError


        def transpose_get(grid: Grid, r: int, c: int) -> int:
            """The value stored at the transposed coordinate (c, r)."""
            raise NotImplementedError
    '''),
    gold=S('''
        from lib import Grid


        def from_pairs(rows: int, cols: int, pairs: list[tuple[int, int, int]]) -> Grid:
            g = Grid(rows, cols)
            for r, c, value in pairs:
                g.set(r, c, value)
            return g


        def row_sum(grid: Grid, r: int) -> int:
            return sum(grid.get(r, c) for c in range(grid.cols))


        def transpose_get(grid: Grid, r: int, c: int) -> int:
            return grid.get(c, r)
    '''),
    type_wrong=S('''
        from lib import Grid


        def from_pairs(rows: int, cols: int, pairs: list[tuple[int, int, int]]) -> Grid:
            g = Grid(rows, cols)
            for r, c, value in pairs:
                g.set(r, c)
            return g


        def row_sum(grid: Grid, r: int) -> int:
            return sum(grid.get(r, c) for c in range(grid.cols))


        def transpose_get(grid: Grid, r: int, c: int) -> int:
            return grid.get(c, r)
    '''),
    test=S('''
        from target import from_pairs, row_sum, transpose_get
        g = from_pairs(2, 2, [(0, 0, 1), (0, 1, 2), (1, 0, 3)])
        assert row_sum(g, 0) == 3
        assert transpose_get(g, 0, 1) == 3
    '''),
    held_out=S('''
        from target import from_pairs, row_sum, transpose_get
        g = from_pairs(3, 3, [(2, 2, 9), (1, 0, 4)])
        assert row_sum(g, 2) == 9
        assert row_sum(g, 0) == 0
        assert transpose_get(g, 0, 1) == 4
    '''),
    note="tuple-unpacking + Grid API; organic error: Grid.set called with missing argument.")


# ===================================================================================
# T12 (med) — normalise a partial config (total=False TypedDict) into a full Config.
# ===================================================================================
_T12 = _task(
    "auth_config_typeddict", "med",
    lib=S('''
        from typing import TypedDict


        class RawConfig(TypedDict, total=False):
            host: str
            port: int
            debug: bool


        class Config(TypedDict):
            host: str
            port: int
            debug: bool


        DEFAULTS: Config = {"host": "localhost", "port": 8080, "debug": False}
    '''),
    stub=S('''
        from lib import RawConfig, Config, DEFAULTS


        def normalize(raw: RawConfig) -> Config:
            """Return a full Config: each field from raw if present, else DEFAULTS."""
            raise NotImplementedError


        def endpoint(cfg: Config) -> str:
            """'host:port' rendered from a full Config."""
            raise NotImplementedError
    '''),
    gold=S('''
        from lib import RawConfig, Config, DEFAULTS


        def normalize(raw: RawConfig) -> Config:
            return {
                "host": raw.get("host", DEFAULTS["host"]),
                "port": raw.get("port", DEFAULTS["port"]),
                "debug": raw.get("debug", DEFAULTS["debug"]),
            }


        def endpoint(cfg: Config) -> str:
            return f"{cfg['host']}:{cfg['port']}"
    '''),
    type_wrong=S('''
        from lib import RawConfig, Config, DEFAULTS


        def normalize(raw: RawConfig) -> Config:
            return {
                "host": raw.get("host", DEFAULTS["host"]),
                "port": raw.get("port", DEFAULTS["port"]),
                "debug": raw.get("debug", DEFAULTS["debug"]),
            }


        def endpoint(cfg: Config) -> str:
            return f"{cfg['hostname']}:{cfg['port']}"
    '''),
    test=S('''
        from target import normalize, endpoint
        c = normalize({"host": "example.com", "port": 9000})
        assert c["host"] == "example.com"
        assert c["port"] == 9000
        assert c["debug"] is False
        assert endpoint(c) == "example.com:9000"
    '''),
    held_out=S('''
        from target import normalize, endpoint
        c = normalize({})
        assert c == {"host": "localhost", "port": 8080, "debug": False}
        assert endpoint(c) == "localhost:8080"
        c2 = normalize({"debug": True})
        assert c2["debug"] is True
        assert c2["port"] == 8080
    '''),
    note="total=False vs total TypedDict; organic error: read an undefined TypedDict key.")


TASKS_AUTHORING = [
    _T1, _T2, _T3, _T4, _T5, _T6, _T7, _T8, _T9, _T10, _T11, _T12,
]


# ===================================================================================
# GATE A (__main__) — for every task assert A (stub fails held-out), B (gold passes both),
# C (gold pyrefly-clean), D (task has type-error surface: stub and/or type_wrong dirty).
# ===================================================================================
if __name__ == "__main__":
    import os as _os
    import sys
    import json
    import subprocess
    sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
    from scaffold.mock_env import MultiFileEnv, PYREFLY

    subprocess.run(["pkill", "-9", "-f", "[p]yrefly"], capture_output=True)

    def pyrefly_errs(files, scope):
        """UNCAPPED pyrefly errors filtered to basename == scope. Fresh workspace that never
        runs tests, so no _run_tests.py noise pollutes the diagnostics."""
        env = MultiFileEnv(files, "target.py", "", skip_pyrefly=False)
        try:
            r = subprocess.run([PYREFLY, "check", "--output-format", "json"], cwd=env.ws,
                               capture_output=True, text=True, timeout=90)
            errs = json.loads(r.stdout or "{}").get("errors", [])
        except Exception as e:
            return [{"name": "INVOKE_FAIL", "description": str(e)}]
        finally:
            env.close()
        return [e for e in errs if _os.path.basename(e.get("path", "") or "") == scope]

    def _splice(t, src):
        return {**t["files"], t["target"]: src}

    def _diag(e):
        return (f"[{e.get('name')}] L{e.get('line')}: "
                f"{(e.get('concise_description') or e.get('description') or '')[:110]}")

    print(f"{'task':28} {'grp':7} {'A.stubfail':11} {'B.vis':6} {'B.held':7} "
          f"{'C.goldclean':12} {'D.surface':22}")
    allok = True
    detail = []
    for t in TASKS_AUTHORING:
        tgt = t["target"]

        # A — stub fails the held-out oracle (nothing implemented).
        e_stub = MultiFileEnv(t["files"], tgt, t["test"], held_out_src=t["held_out"],
                              skip_pyrefly=True)
        a_fail = not e_stub.score()["resolved"]
        e_stub.close()

        # B — gold passes the VISIBLE test and the HELD-OUT oracle.
        e_gold = MultiFileEnv(_splice(t, t["gold_target"]), tgt, t["test"],
                              held_out_src=t["held_out"], skip_pyrefly=True)
        b_vis = e_gold.run_tests()["resolved"]
        b_held = e_gold.score()["resolved"]
        e_gold.close()

        # C — gold is pyrefly-clean in BOTH target.py and lib.py.
        gold_files = _splice(t, t["gold_target"])
        gold_tgt_errs = pyrefly_errs(gold_files, "target.py")
        gold_lib_errs = pyrefly_errs(gold_files, "lib.py") if "lib.py" in gold_files else []
        c_clean = len(gold_tgt_errs) == 0 and len(gold_lib_errs) == 0

        # D — type-error surface exists: stub and/or the type_wrong sketch flag target.py.
        stub_errs = pyrefly_errs(t["files"], "target.py")
        wrong_errs = pyrefly_errs(_splice(t, t["type_wrong"]), "target.py")
        d_surface = (len(stub_errs) + len(wrong_errs)) >= 1

        ok = a_fail and b_vis and b_held and c_clean and d_surface
        allok = allok and ok
        print(f"{t['name']:28} {t['group']:7} "
              f"{('FAIL ok' if a_fail else 'PASS BAD!'):11} "
              f"{('PASS' if b_vis else 'FAIL!'):6} "
              f"{('PASS' if b_held else 'FAIL!'):7} "
              f"{('clean' if c_clean else 'DIRTY!'):12} "
              f"{(f'stub={len(stub_errs)} wrong={len(wrong_errs)}' + (' ok' if d_surface else ' NONE!')):22}"
              f"{'' if ok else '   <-- PROBLEM'}")
        if not c_clean:
            print(f"     ! C gold NOT clean: target={[_diag(e) for e in gold_tgt_errs][:4]} "
                  f"lib={[_diag(e) for e in gold_lib_errs][:4]}")
        if not d_surface:
            print(f"     ! D no type surface (stub and type_wrong both clean)")
        detail.append((t["name"], t["group"], t["note"], wrong_errs))

    print("\n--- D evidence: the diagnostic a checker surfaces on the type_wrong sketch ---")
    for name, grp, note, werrs in detail:
        print(f"  [{grp}] {name}: {note}")
        for e in werrs[:4]:
            print(f"      {_diag(e)}")
        if not werrs:
            print(f"      (type_wrong clean; surface comes from the stub instead)")

    print(f"\nALL OK ({len(TASKS_AUTHORING)} tasks)" if allok
          else "\nPROBLEMS — fix before smoke")
