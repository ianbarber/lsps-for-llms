#!/usr/bin/env python3
"""G4 fixtures — 10 fixed (prefix, edit) cases for the payload-equivalence audit.

Per experiment_plan §11.1 G4: 10 fixed (prefix, edit) triggers, each a small
typed Python snippet where an edit produces deterministic pyrefly diagnostics.
We use standalone self-contained snippets with NO third-party imports (the
"standalone .py fixtures (simpler, fully controlled — preferred for an audit)"
option in the task) so the diagnostics are fully deterministic and reproducible
without per-task venv setup.

Each case is `(prefix, edit)`:
- `prefix` is the file content before the edit.
- `edit` is the file content after the edit (full-document — matches the
  delivery layers' full-document didChange sync).
- `edited_region` is the 1-indexed line span the edit touched, for top-K
  recency ranking.

The cases span: argument-type errors, assignment-type errors, return-type
errors, multiple diagnostics (top-K ordering), an unused-name, an
attribute-access error, and a clean edit (zero diagnostics → empty payload).
"""

from __future__ import annotations

from dataclasses import dataclass

from lsp.payload import EditedRegion


@dataclass(frozen=True)
class G4Case:
    name: str
    prefix: str
    edit: str
    edited_region: EditedRegion
    note: str


# Line numbers in edited_region are 1-indexed and point at the edited line(s) in
# the *post-edit* (`edit`) content.
CASES: list[G4Case] = [
    G4Case(
        name="01_bad_argument_type",
        prefix=(
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n"
            "\n"
            "x: int = add(1, 2)\n"
        ),
        edit=(
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n"
            "\n"
            'x: int = add(1, "two")\n'
        ),
        edited_region=EditedRegion(4, 4),
        note="pass a str where an int param is expected",
    ),
    G4Case(
        name="02_bad_assignment",
        prefix="y: str = \"hello\"\n",
        edit="y: str = 42\n",
        edited_region=EditedRegion(1, 1),
        note="assign int to a str-annotated name",
    ),
    G4Case(
        name="03_bad_return_type",
        prefix=(
            "def f() -> int:\n"
            "    return 1\n"
        ),
        edit=(
            "def f() -> int:\n"
            "    return \"nope\"\n"
        ),
        edited_region=EditedRegion(2, 2),
        note="return a str from an int-annotated function",
    ),
    G4Case(
        name="04_two_errors_topk_order",
        prefix=(
            "a: int = 1\n"
            "b: int = 2\n"
        ),
        edit=(
            'a: int = "x"\n'
            'b: int = "y"\n'
        ),
        edited_region=EditedRegion(2, 2),
        note="two bad-assignments; recency ranking puts line 2 first",
    ),
    G4Case(
        name="05_attribute_error",
        prefix=(
            "n: int = 5\n"
            "m = n + 1\n"
        ),
        edit=(
            "n: int = 5\n"
            "m = n.nonexistent_attr\n"
        ),
        edited_region=EditedRegion(2, 2),
        note="access a missing attribute on an int",
    ),
    G4Case(
        name="06_undefined_name",
        prefix=(
            "value: int = 10\n"
            "result = value\n"
        ),
        edit=(
            "value: int = 10\n"
            "result = undefined_symbol\n"
        ),
        edited_region=EditedRegion(2, 2),
        note="reference a name that is not defined",
    ),
    G4Case(
        name="07_clean_edit_empty_payload",
        prefix=(
            "def g(x: int) -> int:\n"
            "    return x\n"
        ),
        edit=(
            "def g(x: int) -> int:\n"
            "    return x * 2\n"
        ),
        edited_region=EditedRegion(2, 2),
        note="well-typed edit; expected zero diagnostics → empty payload",
    ),
    G4Case(
        name="08_list_element_type",
        prefix=(
            "xs: list[int] = [1, 2, 3]\n"
        ),
        edit=(
            'xs: list[int] = [1, "two", 3]\n'
        ),
        edited_region=EditedRegion(1, 1),
        note="str element in a list[int]",
    ),
    G4Case(
        name="09_call_missing_arg",
        prefix=(
            "def h(a: int, b: int) -> int:\n"
            "    return a + b\n"
            "\n"
            "z = h(1, 2)\n"
        ),
        edit=(
            "def h(a: int, b: int) -> int:\n"
            "    return a + b\n"
            "\n"
            "z = h(1)\n"
        ),
        edited_region=EditedRegion(4, 4),
        note="call with a missing required argument",
    ),
    G4Case(
        name="10_dict_value_type",
        prefix=(
            'd: dict[str, int] = {"a": 1}\n'
        ),
        edit=(
            'd: dict[str, int] = {"a": "one"}\n'
        ),
        edited_region=EditedRegion(1, 1),
        note="str value in a dict[str, int]",
    ),
]


assert len(CASES) == 10, "G4 requires exactly 10 fixed (prefix, edit) cases"
