#!/usr/bin/env python3
"""C36 read-required navigation-v2 instances (`readreq` / `readreq_pilot` splits).

WHY. The substitution adapter (`runs/sft/substitution_lora_27b`) drove reread of the defining
file from 11/12 -> 0/12 (C33 apparatus) and 12/12 -> 1/12 (C35 confirmation). Every training
instance had a SUFFICIENT span, so the learned policy is either

  (a) CONDITIONAL   "use the span when it suffices, read when it does not", or
  (b) BLANKET       "never read after a span"  -- which would be actively harmful.

Deciding between them needs navigation-v2 instances whose delivered span is GENUINE (the real
LSP go-to-definition result, byte-identical to what the live server returns) but INSUFFICIENT,
and where reading the defining file provably recovers what is missing.

MECHANISM -- HOISTED-HELPER DELEGATION. The instance is exactly what the frozen generator emits,
plus one behaviour-preserving refactor: every override's arithmetic expression is HOISTED into a
module-level private helper in the SAME file, leaving the override as a pure delegation.

    def _hscxnvd(value: int) -> int:                            # line 13
        return value * 25 + 61  # spec: value * 8 + 61          # line 14  <- the bug


    class Cnyeajbl(Browutpe):                                   # line 17
        def mnfokyyg(self, value: int) -> int:
            return _hscxnvd(value)                              # line 19

Go-to-definition at the visible call site still resolves the override, so the live server returns
the COMPLETE, untruncated current source of the method that binds -- and that source no longer
contains the constants the fix needs. Nothing is doctored: the method really is two lines long.

A byte-paired SUFFICIENT twin (`inline`) is the same repository with the target class's body
inlined and the helper left in place as dead code, so line numbers do not move. The two flavours
differ in exactly one file and exactly one line -- the span's own second line. Prompts, tests,
`expected` and `held_expected` are byte-identical.

THE BUG IS IN THE MULTIPLIER, NOT THE OFFSET (C36 revision, and the reason this file was
redesigned). The helper form is `value*A + B` / `value*A - B` / `(value ^ A) + B`. B is the frozen
`_spec` per-class `params[i]`; A is a second per-class multiplier drawn without replacement by a
seed-keyed RNG owned by THIS module. The target helper renders `A + da` with a seed-keyed
`|da| >= 2`, so the residual `gold(v) - buggy(v)` is VALUE-DEPENDENT.

  Why this matters. In the first version of this file the injected bug was the frozen `+1` on the
  OFFSET, inherited from `navigation_tasks._sources`. That made the residual a CONSTANT, so
  rewriting the one line the model is handed -- the delegation inside the span -- as
  `return _hscxnvd(value) - 1` was an exact, fully generalizing repair with zero reads. Measured
  live on all 12 candidate instances of the first split: 12/12 visible-pass, held-out-pass AND
  generalization-pass. A blanket-suppressed adapter would therefore have PASSED, which is exactly
  the outcome the experiment must not confuse with a conditional policy. Perturbing the multiplier
  removes that family: any `buggy(v) + c` that satisfies the visible point is off by `da*7` at the
  held-out point. `no_read_adversary()` enumerates the whole one-free-parameter repair space and
  the split gates on it.

RECOVERABILITY. Because the multiplier is what is wrong, the file's remaining constant plus the
single visible equation does not by itself say WHICH constant to change (repairing the offset also
passes the visible test, and fails held-out). Each helper therefore carries a trailing
`# spec: <intended expression>` comment -- identical to the code on every non-target helper, and
differing exactly in the multiplier on the target. One read of the defining file yields the gold
expression outright; the equation route (solve the multiplier from `expected` with the file's
offset) is verified to agree. This is deliberate: the instance must measure whether the policy
GOES AND LOOKS, not whether the model can do inference under ambiguity.

FROZEN-PROTOCOL CONSTRAINT. `scripts/experiments/navigation_tasks.py` is hash-gated and stays
BYTE-IDENTICAL. New seeds, new templates and the hoist are injected at runtime, following
`run_substitution_train.install_train_split()`, and every patched attribute is restored under
`finally` and asserted restored afterwards.

Usage:
  python scripts/experiments/navigation_readreq_tasks.py select
  python scripts/experiments/navigation_readreq_tasks.py validate --split readreq_pilot \
      --out runs/protocol/navigation_v2_readreq_validation.json
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scaffold.tooling import find_pyrefly  # noqa: E402
from scripts.experiments import navigation_tasks as NT  # noqa: E402
from scripts.experiments.run_navigation import (  # noqa: E402
    _format_method_span,
    _method_from_lsp,
)
from scripts.experiments.run_navigation_reread import _shared_span_and_verify  # noqa: E402


# ---------------------------------------------------------------------------
# split definition (injected at runtime; navigation_tasks.py is never touched)
# ---------------------------------------------------------------------------
EXPERIMENT = "read-required-substitution"
PILOT_SPLIT = "readreq_pilot"
MAIN_SPLIT = "readreq"
SPLITS = (PILOT_SPLIT, MAIN_SPLIT)
FLAVOURS = ("hoisted", "inline")

TEMPLATES = ("delegate_scale_offset", "delegate_scale_sub", "delegate_xor_offset")
A_RANGE = tuple(range(2, 61))     # hidden multiplier, this module's RNG
B_RANGE = tuple(range(11, 90))    # mirrors the frozen _spec value_range for non-multiply templates

# The injected fault perturbs BOTH of the target helper's constants. One perturbed constant is not
# enough: a fault in the offset alone is undone by `helper(value) + c`, and a fault in the
# multiplier alone is undone by `helper(value) + c*value` -- both one-free-parameter repairs the
# visible test pins exactly, so both are exact zero-read solves. With both constants wrong the
# cheapest wrapper that can reach gold is `helper(value) + c1*value + c2`, which one equation does
# not pin. |da| >= 2 so that "the multiplier is off by one" is not a blind repair either.
BUG_DELTA_CHOICES = tuple(d for d in range(-9, 10) if abs(d) >= 2)
BUG_OFFSET_DELTA_CHOICES = tuple(d for d in range(-9, 10) if d != 0)
SPEC_PREFIX = "  # spec: "

PILOT_BAND = range(67001, 68000)
MAIN_BAND = range(71001, 73000)
N_PILOT = 6
N_MAIN = 12

# pre-registered insufficiency thresholds, applied by the deterministic selector AND re-asserted
# per instance in `validate`.
MIN_DISTINCT_HELDOUT = 12
MAX_BLIND_GUESS_CEILING = 0.10

# produced by `select_seeds` (subcommand `select`); re-derived and asserted in `validate`.
PILOT_SEEDS = (67001, 67003, 67009, 67012, 67014, 67017)
MAIN_SEEDS = (
    71005, 71012, 71053, 71057, 71060, 71116,
    71121, 71124, 71175, 71191, 71195, 71262,
)
SPLIT_SEEDS = {PILOT_SPLIT: PILOT_SEEDS, MAIN_SPLIT: MAIN_SEEDS}

# hash gate: the two sources whose bytes define the environment this experiment inherits.
FROZEN_SHA256 = {
    "scripts/experiments/navigation_tasks.py":
        "f860fd07fdeb2d4f78d89a047c6804d79cc3babd60fab7e0a06e839679692d97",
    "scaffold/stream_agent.py":
        "0267afa17a22c5f0eea77bce82927b5d25890dea0500312163d2a1e2e1f40b79",
}

_ORIG_SOURCES = NT._sources
_ORIG_EXPRESSION = NT._expression
_ORIG_EVALUATE = NT._evaluate

# frozen `_sources` passes only the per-class param to `_expression`/`_evaluate`, never the class
# index, so the param value is used as the lookup key for this module's second constant. The
# `table_key_unique` gate rejects any seed where the target's buggy key `params[t]+1` collides
# with another class's param, which is what makes the key well defined.
_TABLE: dict[int, int] = {}
# validation-only: force the target class onto an alternative (A, B) member of the ambiguity set,
# used to materialise the twins that prove the visible surface does not determine the answer.
_PARAM_OVERRIDE: dict | None = None


def _substrain_seeds() -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Read the training split out of run_substitution_train.py without importing it."""
    source = (ROOT / "scripts" / "experiments" / "run_substitution_train.py").read_text()
    tree = ast.parse(source)
    found: dict[str, tuple] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in ("TRAIN_SEEDS", "TRAIN_TEMPLATES"):
                found[target.id] = tuple(ast.literal_eval(node.value))
    if "TRAIN_SEEDS" not in found or "TRAIN_TEMPLATES" not in found:
        raise SystemExit("could not read TRAIN_SEEDS/TRAIN_TEMPLATES from run_substitution_train.py")
    return found["TRAIN_SEEDS"], found["TRAIN_TEMPLATES"]


def install_readreq_splits() -> dict:
    """Register the two read-required splits, asserting non-collision (mirrors install_train_split)."""
    train_seeds, train_templates = _substrain_seeds()
    spent_seeds = {
        seed for name, seeds in NT.SPLIT_SEEDS.items() if name not in SPLITS for seed in seeds
    } | set(train_seeds)
    spent_templates = {
        template for name, templates in NT.SPLIT_TEMPLATES.items() if name not in SPLITS
        for template in templates
    } | set(train_templates)
    new_seeds = set(PILOT_SEEDS) | set(MAIN_SEEDS)
    if len(new_seeds) != len(PILOT_SEEDS) + len(MAIN_SEEDS):
        raise SystemExit("readreq pilot and main seeds overlap each other")
    if new_seeds & spent_seeds:
        raise SystemExit(
            f"readreq seeds collide with a spent split: {sorted(new_seeds & spent_seeds)}"
        )
    if set(TEMPLATES) & spent_templates:
        raise SystemExit(
            f"readreq templates collide with a spent split: {sorted(set(TEMPLATES) & spent_templates)}"
        )
    for split, seeds in SPLIT_SEEDS.items():
        NT.SPLIT_SEEDS[split] = seeds
        NT.SPLIT_TEMPLATES[split] = TEMPLATES
    return {
        "pilot": list(NT.SPLIT_SEEDS["pilot"]),
        "apparatus": list(NT.SPLIT_SEEDS["apparatus"]),
        "confirmation": list(NT.SPLIT_SEEDS["confirmation"]),
        "substrain_training": list(train_seeds),
        "spent_templates": sorted(spent_templates),
    }


# ---------------------------------------------------------------------------
# the delegating arithmetic family
# ---------------------------------------------------------------------------
def form(template: str, a: int, b: int) -> str:
    return {
        "delegate_scale_offset": f"value * {a} + {b}",
        "delegate_scale_sub": f"value * {a} - {b}",
        "delegate_xor_offset": f"(value ^ {a}) + {b}",
    }[template]


def evaluate(template: str, value: int, a: int, b: int) -> int:
    return {
        "delegate_scale_offset": value * a + b,
        "delegate_scale_sub": value * a - b,
        "delegate_xor_offset": (value ^ a) + b,
    }[template]


def _expression(template: str, param: int) -> str:
    """Installed over NT._expression. Non-readreq templates fall through unchanged."""
    if template not in TEMPLATES:
        return _ORIG_EXPRESSION(template, param)
    return form(template, _TABLE[param], param)


def _evaluate(template: str, value: int, param: int) -> int:
    if template not in TEMPLATES:
        return _ORIG_EVALUATE(template, value, param)
    return evaluate(template, value, _TABLE[param], param)


def bug_delta_options(spec: dict, multipliers: list[int]) -> list[tuple[int, int]]:
    """Admissible (multiplier, offset) perturbations for the target.

    Both rendered constants must stay inside the declared support and distinct from every sibling
    class's value, so the buggy line does not masquerade as another class's implementation and the
    file's constants remain unique.
    """
    target = spec["target_idx"]
    a_true, b_true = multipliers[target], spec["params"][target]
    sibling_a = {m for i, m in enumerate(multipliers) if i != target}
    sibling_b = {b for i, b in enumerate(spec["params"]) if i != target}
    return [
        (da, db)
        for da in BUG_DELTA_CHOICES if a_true + da in A_RANGE and a_true + da not in sibling_a
        for db in BUG_OFFSET_DELTA_CHOICES
        if b_true + db in B_RANGE and b_true + db not in sibling_b
    ]


def bug_delta(spec: dict, multipliers: list[int]) -> tuple[int, int] | None:
    """Seed-keyed and variant-independent, so typed and erased render identically."""
    options = bug_delta_options(spec, multipliers)
    if not options:
        return None
    return random.Random(f"{spec['seed']}:bugdelta").choice(options)


def _multipliers(spec: dict) -> list[int]:
    """Per-class hidden multiplier A, drawn without replacement. Keyed on the SEED only, so the
    typed and erased variants render identically (`runtime_files_identical` must hold)."""
    return random.Random(f"{spec['seed']}:multiplier").sample(list(A_RANGE), len(spec["classes"]))


def _helper_names(spec: dict) -> list[str]:
    rng = random.Random(f"{spec['seed']}:helper")
    names: list[str] = []
    seen: set[str] = set()
    while len(names) < len(spec["classes"]):
        candidate = "_h" + "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(6))
        if candidate in seen:
            continue
        seen.add(candidate)
        names.append(candidate)
    return names


def _table(spec: dict, multipliers: list[int]) -> dict[int, int]:
    target = spec["target_idx"]
    table = {param: multipliers[idx] for idx, param in enumerate(spec["params"])}
    table[spec["params"][target] + 1] = multipliers[target]   # the buggy key
    return table


def _spec_for(seed: int, template: str) -> dict:
    """Frozen `_spec`, forced onto one template through a throwaway single-seed split."""
    key = "_readreq_probe"
    NT.SPLIT_SEEDS[key] = (seed,)
    NT.SPLIT_TEMPLATES[key] = (template,)
    try:
        return NT._spec(seed, key)
    finally:
        NT.SPLIT_SEEDS.pop(key, None)
        NT.SPLIT_TEMPLATES.pop(key, None)


# ---------------------------------------------------------------------------
# the hoist
# ---------------------------------------------------------------------------
def helper_body_line(gold_expression: str, code_expression: str, indent: str = "    ") -> str:
    """`return <code>  # spec: <gold>` -- the ONE line that carries the missing information.

    On every non-target helper the two are identical, so the comment is uniform repository
    convention rather than a marker of the target. On the target they differ in the multiplier,
    which is what makes the correct repair uniquely determined for a reader (see module docstring).
    """
    return f"{indent}return {code_expression}{SPEC_PREFIX}{gold_expression}"


def _render_units(spec: dict, multipliers: list[int], names: list[str],
                  delta: tuple[int, int], flavour: str) -> dict[str, str]:
    base, method = spec["base"], spec["method"]
    n_modules = len(spec["modules"])
    da, db = delta
    files: dict[str, str] = {}
    for module_idx, module in enumerate(spec["modules"]):
        blocks = [f"from pkg.base import {base}\n"]
        for idx, class_name in enumerate(spec["classes"]):
            if idx % n_modules != module_idx:
                continue
            offset = spec["params"][idx]
            is_target = idx == spec["target_idx"]
            gold_expression = form(spec["template"], multipliers[idx], offset)
            code_expression = form(spec["template"],
                                   multipliers[idx] + (da if is_target else 0),
                                   offset + (db if is_target else 0))
            helper = names[idx]
            inlined = flavour == "inline" and is_target
            body = (helper_body_line(gold_expression, code_expression, "        ") + "\n" if inlined
                    else f"        return {helper}(value)\n")
            blocks.append(
                f"\n\ndef {helper}(value: int) -> int:\n"
                + helper_body_line(gold_expression, code_expression) + "\n"
                f"\n\nclass {class_name}({base}):\n"
                f"    def {method}(self, value: int) -> int:\n"
                + body
            )
        files[f"pkg/units/{module}.py"] = "".join(blocks)
    return files


def _sources_readreq(flavour: str):
    """A drop-in for NT._sources: frozen scaffolding, hoisted implementation modules.

    Everything except the `pkg/units/*.py` bodies, `target_method_span` and `gold` comes from the
    FROZEN `_sources` -- `expected`, `held_expected`, both test files, the registry contract and
    the +1 bug convention included, because `_expression`/`_evaluate` are patched underneath it.
    """
    if flavour not in FLAVOURS:
        raise ValueError(f"unknown flavour {flavour!r}")

    def _sources(spec: dict, variant: str) -> tuple[dict[str, str], dict]:
        global _TABLE
        if _PARAM_OVERRIDE is not None:
            spec = copy.deepcopy(spec)
            spec["params"] = list(spec["params"])
            spec["params"][spec["target_idx"]] = _PARAM_OVERRIDE["b"]
        multipliers = _multipliers(spec)
        if _PARAM_OVERRIDE is not None:
            multipliers[spec["target_idx"]] = _PARAM_OVERRIDE["a"]
        names = _helper_names(spec)
        delta = bug_delta(spec, multipliers)
        if delta is None:
            raise SystemExit(f"seed {spec['seed']}: no admissible multiplier perturbation")
        _TABLE = _table(spec, multipliers)

        files, meta = _ORIG_SOURCES(spec, variant)
        files.update(_render_units(spec, multipliers, names, delta, flavour))

        target_idx = spec["target_idx"]
        target_class = spec["classes"][target_idx]
        target_path = meta["target_path"]
        helper = names[target_idx]
        start, end, span = NT._method_span(files[target_path], target_class, spec["method"])
        lines = files[target_path].splitlines()
        helper_def_line = next(
            i + 1 for i, line in enumerate(lines) if line.startswith(f"def {helper}(")
        )
        helper_return_line = helper_def_line + 1
        offset = spec["params"][target_idx]
        da, db = delta
        gold_expression = form(spec["template"], multipliers[target_idx], offset)
        buggy_expression = form(spec["template"], multipliers[target_idx] + da, offset + db)
        return_text = helper_body_line(gold_expression, buggy_expression)
        if flavour == "hoisted":
            gold = {"path": target_path, "start": helper_return_line, "end": helper_return_line,
                    "new_text": f"    return {gold_expression}"}
        else:
            gold = {"path": target_path, "start": end, "end": end,
                    "new_text": f"        return {gold_expression}"}
        meta["target_method_span"] = {"start": start, "end": end, "source": span}
        meta["gold"] = gold
        meta["helper"] = {
            "name": helper,
            "def_line": helper_def_line,
            "return_line": helper_return_line,
            "return_text": return_text,
            "source": f"def {helper}(value: int) -> int:\n{return_text}",
            "multiplier": multipliers[target_idx],
            "buggy_multiplier": multipliers[target_idx] + da,
            "offset": offset,
            "buggy_offset": offset + db,
            "delta": da,
            "offset_delta": db,
            "call_line": end,
            "flavour": flavour,
            "gold_expression": gold_expression,
            "buggy_expression": buggy_expression,
            "spec_comment": SPEC_PREFIX + gold_expression,
            "inline_repair": {"path": target_path, "start": end, "end": end,
                              "new_text": f"        return {gold_expression}"},
        }
        meta["readreq"] = {
            "flavour": flavour,
            "multipliers": list(multipliers),
            "buggy_multiplier": multipliers[target_idx] + da,
            "buggy_offset": offset + db,
            "delta": [da, db],
            "helpers": list(names),
            "template_family": spec["template"],
        }
        return files, meta

    return _sources


@contextlib.contextmanager
def patched(flavour: str):
    """Install the three runtime overrides, restoring them unconditionally."""
    NT._expression, NT._evaluate = _expression, _evaluate
    NT._sources = _sources_readreq(flavour)
    try:
        yield
    finally:
        NT._sources = _ORIG_SOURCES
        NT._expression = _ORIG_EXPRESSION
        NT._evaluate = _ORIG_EVALUATE


@contextlib.contextmanager
def param_override(a: int, b: int):
    global _PARAM_OVERRIDE
    _PARAM_OVERRIDE = {"a": a, "b": b}
    try:
        yield
    finally:
        _PARAM_OVERRIDE = None


def assert_frozen_restored() -> dict:
    """Post-build guard: the frozen module is un-patched and its bytes are unchanged."""
    restored = {
        "sources_restored": NT._sources is _ORIG_SOURCES,
        "expression_restored": NT._expression is _ORIG_EXPRESSION,
        "evaluate_restored": NT._evaluate is _ORIG_EVALUATE,
    }
    for rel, expected in FROZEN_SHA256.items():
        actual = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        restored[f"{rel}_unmodified"] = actual == expected
    if not all(restored.values()):
        raise SystemExit(f"frozen protocol was not restored: {restored}")
    return restored


def build_readreq_tasks(root: str | Path, split: str, flavour: str) -> list[dict]:
    """Materialise a readreq split through the FROZEN build_tasks."""
    if split not in SPLITS:
        raise ValueError(f"unknown readreq split {split!r}")
    install_readreq_splits()
    with patched(flavour):
        tasks = NT.build_tasks(Path(root) / flavour / split, split)
    assert_frozen_restored()
    for task in tasks:
        task["flavour"] = flavour
        task["experiment"] = EXPERIMENT
    return tasks


def build_readreq_files(seed: int, template: str, flavour: str,
                        variant: str = "typed") -> tuple[dict[str, str], dict, dict]:
    """In-memory render (no git repo, no disk): (files, meta, spec)."""
    spec = _spec_for(seed, template)
    with patched(flavour):
        files, meta = NT._sources(spec, variant)
    assert_frozen_restored()
    return files, meta, spec


def prompt_for(spec: dict, meta: dict, files: dict[str, str]) -> str:
    """`NT.build_prompt` against an in-memory render."""
    task = {
        **spec, **meta,
        "n_overrides": len(spec["classes"]),
        "variants": {"typed": {"files": files}},
    }
    return NT.build_prompt(task, "typed")


# ---------------------------------------------------------------------------
# insufficiency: the ambiguity set
# ---------------------------------------------------------------------------
def ambiguity_set(spec: dict, multipliers: list[int]) -> dict:
    """F = every (A', B') in the declared support, minus every sibling value, that reproduces the
    single visible equation `f(input) == expected`.

    Sibling A and B values are removed because reading the OTHER three unit modules is cheap,
    allowed, and does not touch the defining file; both constants are sampled without replacement,
    so those values are genuinely excluded for a model that reads everything except the one file.
    """
    target = spec["target_idx"]
    template, value = spec["template"], spec["input"]
    a_true, b_true = multipliers[target], spec["params"][target]
    expected = evaluate(template, value, a_true, b_true)
    held_input = value + 7
    held_expected = evaluate(template, held_input, a_true, b_true)
    sibling_a = {a for i, a in enumerate(multipliers) if i != target}
    sibling_b = {b for i, b in enumerate(spec["params"]) if i != target}
    members = []
    for a in A_RANGE:
        if a in sibling_a:
            continue
        for b in B_RANGE:
            if b in sibling_b:
                continue
            if evaluate(template, value, a, b) == expected:
                members.append({"a": a, "b": b, "held": evaluate(template, held_input, a, b)})
    held = [member["held"] for member in members]
    distinct = sorted(set(held))
    ceiling = max((held.count(h) for h in distinct), default=1) / max(len(members), 1)
    return {
        "input": value,
        "expected": expected,
        "held_input": held_input,
        "held_expected": held_expected,
        "truth": {"a": a_true, "b": b_true},
        "n_members": len(members),
        "n_distinct_heldout": len(distinct),
        "blind_guess_ceiling": round(ceiling, 6),
        "exact_function_ceiling": round(1 / max(len(members), 1), 6),
        "truth_in_set": {"a": a_true, "b": b_true, "held": held_expected} in members,
        "members": members,
        "distinct_heldout": distinct,
    }


# ---------------------------------------------------------------------------
# insufficiency, part 2: the mechanical NO-READ ADVERSARY
#
# `ambiguity_set` bounds hypotheses of the form `value*A' (+|-) B'` / `(value ^ A') + B'`. It does
# NOT bound repairs expressed as a function of the visible `helper(value)` call, and that omission
# is what made the first version of this split solvable with zero reads (`helper(value) - 1` was an
# exact repair on 12/12 instances, because the injected fault was a constant +1 on the offset).
#
# The adversary below is the missing gate. It is a scripted policy that may see the prompt and the
# span, may run the VISIBLE test as often as it likes, and may never read or grep the defining
# file. Every one-free-parameter repair it can write at the span's own line is enumerated; the
# visible test pins the parameter; the gate requires every visible-passing candidate to FAIL
# held-out. Two-free-parameter repairs are not pinned by one equation, so they are bounded by a
# ceiling instead, exactly as the (A', B') family is.
# ---------------------------------------------------------------------------
ADVERSARY_C_RANGE = tuple(range(-500, 501))
# the tightest slope range that still contains every admissible multiplier perturbation, so the
# published two-parameter ceiling is the CONSERVATIVE one a model with a correct prior would face
ADVERSARY_SLOPE_RANGE = tuple(range(-12, 13))


def _one_param_candidates(g, value: int, c: int) -> dict[str, int | None]:
    """Every one-free-parameter expression a non-reading model can put at the span's line."""
    gv = g(value)
    return {
        "g(value) + c": gv + c,
        "g(value) - c": gv - c,
        "g(value) * c": gv * c,
        "c - g(value)": c - gv,
        "g(value) // c": (gv // c) if c else None,
        "g(value) % c": (gv % c) if c else None,
        "g(value) ^ c": (gv ^ c) if (c >= 0 and gv >= 0) else None,
        "g(value) + c*value": gv + c * value,
        "g(value) - c*value": gv - c * value,
        "g(value) * c + value": gv * c + value,
        "value + c": value + c,
        "value - c": value - c,
        "c - value": c - value,
        "value * c": value * c,
        "value ^ c": (value ^ c) if c >= 0 else None,
        "value * value + c": value * value + c,
        "c": c,
    }


def no_read_adversary(spec: dict, multipliers: list[int], delta: tuple[int, int]) -> dict:
    """Enumerate the zero-read repair space at the span's own line. See the block comment above."""
    target = spec["target_idx"]
    template, value = spec["template"], spec["input"]
    a_true, b_true = multipliers[target], spec["params"][target]
    da, db = delta
    a_bug, b_bug = a_true + da, b_true + db
    held_input = value + 7
    expected = evaluate(template, value, a_true, b_true)
    held_expected = evaluate(template, held_input, a_true, b_true)

    def g(v: int) -> int:                       # the buggy helper the model can call but not see
        return evaluate(template, v, a_bug, b_bug)

    escapes: list[dict] = []
    heldout_only: list[dict] = []
    n_visible_pass = 0
    for c in ADVERSARY_C_RANGE:
        here = _one_param_candidates(g, value, c)
        there = _one_param_candidates(g, held_input, c)
        for name, got in here.items():
            visible_ok = got == expected
            held_ok = there[name] == held_expected
            n_visible_pass += visible_ok
            if visible_ok and held_ok:
                escapes.append({"form": name, "c": c, "held": there[name]})
            elif held_ok and not visible_ok:
                # NOT a solve -- the model sees this fail its own <test/> -- but it WOULD be a
                # spurious held_out_pass if a rollout happened to end on it. Cannot be designed
                # away (for the xor family a c matching gold at the single held-out point always
                # exists), so it is disclosed here and neutralised by scoring `repair_correct`
                # = visible_pass AND held_out_pass.
                heldout_only.append({"form": name, "c": c})

    # two-free-parameter wrapper family: `g(value) + c1*value + c2`, with c2 pinned by the visible
    # equation for each c1. Not uniquely determined, so it gets a ceiling, not a hard gate.
    wrapper_members = []
    for c1 in ADVERSARY_SLOPE_RANGE:
        c2 = expected - (g(value) + c1 * value)
        wrapper_members.append(g(held_input) + c1 * held_input + c2)
    wrapper_hits = sum(1 for h in wrapper_members if h == held_expected)
    wrapper_ceiling = (max(wrapper_members.count(h) for h in set(wrapper_members))
                       / len(wrapper_members))
    # the spec comment is what determines the repair: keeping the FILE's offset and re-solving the
    # multiplier from the visible fact is either unsolvable or wrong at the held-out point, so a
    # reader who ignores the comment and trusts the code's constants does not accidentally win.
    code_offset_route = [a for a in A_RANGE if evaluate(template, value, a, b_bug) == expected]
    return {
        "n_one_param_forms": len(_one_param_candidates(g, value, 0)),
        "c_range": [ADVERSARY_C_RANGE[0], ADVERSARY_C_RANGE[-1]],
        "n_visible_passing_one_param": n_visible_pass,
        "escapes": escapes,
        "no_escape": not escapes,
        "n_heldout_only_candidates": len(heldout_only),
        "heldout_only_candidates": heldout_only[:20],
        "buggy_multiplier": a_bug,
        "buggy_offset": b_bug,
        "delta": [da, db],
        "code_offset_route_multipliers": code_offset_route,
        "code_offset_route_passes_heldout": any(
            evaluate(template, held_input, a, b_bug) == held_expected for a in code_offset_route
        ),
        "wrapper_family": "g(value) + c1*value + c2",
        "wrapper_slope_range": [ADVERSARY_SLOPE_RANGE[0], ADVERSARY_SLOPE_RANGE[-1]],
        "wrapper_n_visible_passing": len(wrapper_members),
        "wrapper_n_heldout_passing": wrapper_hits,
        "wrapper_blind_guess_ceiling": round(wrapper_ceiling, 6),
    }


def selection_stats(seed: int, template: str) -> dict:
    spec = _spec_for(seed, template)
    multipliers = _multipliers(spec)
    names = _helper_names(spec)
    amb = ambiguity_set(spec, multipliers)
    target = spec["target_idx"]
    buggy_key = spec["params"][target] + 1
    identifiers = [spec["base"], spec["method"], *spec["classes"], *spec["tokens"], *spec["modules"]]
    delta = bug_delta(spec, multipliers)
    adversary = (no_read_adversary(spec, multipliers, delta) if delta is not None else None)
    gates = {
        # the frozen `_sources` still renders (and discards) a unit body at params[target]+1, so
        # that key must exist in _TABLE and must not shadow a sibling's.
        "table_key_unique": buggy_key not in spec["params"] and buggy_key in B_RANGE,
        "distinct_heldout_at_least_min": amb["n_distinct_heldout"] >= MIN_DISTINCT_HELDOUT,
        "blind_guess_ceiling_within_bound": amb["blind_guess_ceiling"] <= MAX_BLIND_GUESS_CEILING,
        "truth_in_ambiguity_set": amb["truth_in_set"],
        "heldout_varies_over_ambiguity_set": amb["n_distinct_heldout"] > 1,
        "positive_outputs": amb["expected"] > 0 and amb["held_expected"] > 0,
        "helper_names_unique": len(set(names)) == len(names),
        "helper_names_disjoint_from_identifiers": not any(
            helper in ident or ident in helper for helper in names for ident in identifiers
        ),
        "bug_delta_available": delta is not None,
        "no_read_adversary_fails_heldout": bool(adversary and adversary["no_escape"]),
        "wrapper_family_ceiling_within_bound": bool(
            adversary and adversary["wrapper_blind_guess_ceiling"] <= MAX_BLIND_GUESS_CEILING
        ),
        "spec_comment_is_load_bearing": bool(
            adversary and not adversary["code_offset_route_passes_heldout"]
        ),
    }
    return {"seed": seed, "template": template, "spec": spec, "multipliers": multipliers,
            "helpers": names, "ambiguity": amb, "delta": delta, "adversary": adversary,
            "gates": gates, "passed": all(gates.values())}


def select_seeds(band, n: int, templates: tuple[str, ...] = TEMPLATES) -> list[dict]:
    """The published rule: scan the band increasing; the candidate at position p is offered
    `templates[p % len(templates)]` (forced by the frozen `_spec`); accept the first seed that
    clears every selection gate."""
    chosen: list[dict] = []
    for seed in band:
        stats = selection_stats(seed, templates[len(chosen) % len(templates)])
        if stats["passed"]:
            chosen.append(stats)
            if len(chosen) == n:
                break
    if len(chosen) != n:
        raise SystemExit(f"band exhausted after {len(chosen)}/{n} seeds")
    return chosen


def selection_is_reproducible() -> dict:
    pilot = select_seeds(PILOT_BAND, N_PILOT)
    main = select_seeds(MAIN_BAND, N_MAIN)
    return {
        "pilot_seeds": tuple(s["seed"] for s in pilot),
        "pilot_templates": tuple(s["template"] for s in pilot),
        "main_seeds": tuple(s["seed"] for s in main),
        "main_templates": tuple(s["template"] for s in main),
        "pilot_matches": tuple(s["seed"] for s in pilot) == PILOT_SEEDS,
        "main_matches": tuple(s["seed"] for s in main) == MAIN_SEEDS,
        "ceiling_sum_heldout": round(sum(s["ambiguity"]["blind_guess_ceiling"] for s in main), 4),
        "ceiling_sum_exact_function": round(
            sum(s["ambiguity"]["exact_function_ceiling"] for s in main), 4
        ),
    }


# ---------------------------------------------------------------------------
# runtime hygiene: the stale-bytecode hazard
# ---------------------------------------------------------------------------
def purge_pycache(repo: str | Path) -> int:
    """Remove every __pycache__ under `repo`.

    CPython invalidates a .pyc on (source mtime SECONDS, source size). The gold fix
    `+ 61` -> `+ 60` preserves byte length, so an edit landing in the same mtime second as the
    previous compile silently reuses the stale .pyc and records a CORRECT repair as a FAILURE --
    i.e. it manufactures the blanket-suppression signature. Purge before every scoring run.
    """
    removed = 0
    for cache in Path(repo).rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
        removed += 1
    return removed


def run_visible(task: dict, env, variant: str = "typed") -> bool:
    purge_pycache(task["variants"][variant]["repo_dir"])
    return bool(env.run_tests()["resolved"])


def run_heldout(task: dict, variant: str = "typed") -> bool:
    purge_pycache(task["variants"][variant]["repo_dir"])
    return bool(NT.run_heldout(task, variant))


def reset_repo(task: dict, env, variant: str = "typed") -> None:
    env.reset()
    purge_pycache(task["variants"][variant]["repo_dir"])


# ---------------------------------------------------------------------------
# the held-out oracle guard (apparatus deviation, installed identically in every cell)
# ---------------------------------------------------------------------------
HELDOUT_PATHS = ("test_heldout.py",)
GUARDED_READ_FRAMES = ("_file_view", "_read_range", "_grep", "run")
# Deliberately neutral: a refusal with NO pointer to where to look instead. An earlier version
# ended "...the information you need is in the source under pkg/", which is an instruction to go
# and read -- i.e. guidance toward the outcome variable, delivered only to the rollouts that
# happen to try the oracle.
HELDOUT_DENIED_TEXT = "(this file is not readable in this run)"


def _caller_frames(n: int = 3) -> list[str]:
    """Frame names above the instrumentation, skipping any other `read_file` wrapper.

    The guard and the ledger both wrap `env.read_file` with a function literally named
    `read_file`, so whichever is installed outermost would otherwise appear as the inner one's
    caller and defeat both frame tests. Leading `read_file` frames are therefore skipped.
    """
    names: list[str] = []
    depth = 1
    while len(names) < n and depth < 40:
        try:
            frame = sys._getframe(depth)
        except ValueError:                                       # pragma: no cover
            names.append("?")
            depth += 1
            continue
        name = frame.f_code.co_name
        depth += 1
        if name == "read_file" and not names:
            continue
        names.append(name)
    while len(names) < n:                                        # pragma: no cover
        names.append("?")
    return names


def _norm(path: str | None) -> str:
    """`RealRepoEnv.read_file` normalises through `_abspath`, so `./pkg/units/u.py` and
    `pkg/units/u.py` are the same file. Exact string matching would miss the read entirely and
    record a false BLANKET signature."""
    return os.path.normpath(path) if path else ""


def is_heldout_path(path: str | None) -> bool:
    if not path:
        return False
    name = Path(path).name
    return name in HELDOUT_PATHS or "heldout" in name


def install_heldout_guard(env) -> dict:
    """Deny MODEL-issued reads of the held-out oracle.

    `read_file` is not glob-restricted, so `<read path="test_heldout.py"/>` returns the second
    equation, which together with the visible pair determines A and B UNIQUELY for the linear
    families -- a total solve with zero reads of the defining file. Prior splits got away with it
    only because their spans were sufficient. The guard returns a denial STRING rather than
    raising, because `stream_agent` calls `read_file` from paths that do not catch exceptions.
    It is installed identically in every arm and both model conditions, and every attempt is
    published as `n_heldout_read_attempts`.
    """
    state = {"n_attempts": 0, "attempts": []}
    original = env.read_file

    def read_file(path=None, *args, **kwargs):
        if is_heldout_path(path):
            frames = _caller_frames(2)
            if frames[0] in GUARDED_READ_FRAMES:
                state["n_attempts"] += 1
                state["attempts"].append({"path": path, "caller": frames[0],
                                          "caller2": frames[1]})
                return HELDOUT_DENIED_TEXT
        return original(path, *args, **kwargs)

    env.read_file = read_file
    return state


# The pattern that actually substitutes for `<read>`. `return value` does NOT: it misses the
# `delegate_xor_offset` family, whose helper line is `return (value ^ A) + B`. This constant is
# what the published `grep_substitution_note` quotes, so artifact prose and gate cannot drift.
GREP_ARITHMETIC_PATTERN = r"return .*\d"

POST_EDIT_ELISION = (
    '  # (elided from this automatic post-edit view; use <read path="{path}"/> to see this line)'
)


def install_post_edit_redactor(env, target_path: str, line_no: int, pristine: str) -> dict:
    """Close the INVOLUNTARY post-edit information channel. Apparatus deviation, both conditions.

    `StreamAgent.deliver_turn` re-shows the FULL numbered view of every file edited since the last
    turn (scaffold/stream_agent.py, `fv = "\\n\\n".join(self._file_view(f) for f in ...)`), and the
    unit module is ~30 lines, far under the 250-line truncation. The task FORCES an edit in that
    file, so without this wrapper a model that never reads still receives the helper's
    `return <buggy>  # spec: <gold>` line for free on the turn after its first edit -- which is
    measured: the redactor's own self-check asserts the un-redacted view does deliver it.

    That channel would make the instances not read-required at all, and would put the BLANKET
    outcome out of reach (a blanket-suppressed model would blind-edit, be handed the answer, and
    pass with zero reads). It is therefore closed rather than merely detected.

    Scope, deliberately minimal:
      * ONLY the post-edit view channel (`_file_view` reached from `deliver_turn`'s genexpr).
        A deliberate `<read path=...>` (`_file_view` <- `run`), a `<read lines=a-b>`
        (`_read_range`), a `<grep>` and the agent's own edit-normalisation read are untouched and
        all still deliver the line -- each is asserted per instance.
      * ONLY the one line that carries the missing information, and only while it is still the
        pristine text: once the model has itself written that line, it sees its own bytes.
      * Line COUNT and numbering are preserved, so `<edit lines="A-B">` keeps working and the gold
        edit still applies through a redacted view (asserted per instance).
      * The elision is visibly marked and names the read that would reveal it, so nothing is
        hidden from the model -- the information is withheld from a push channel, not from the
        model's own actions.
    """
    state = {"n_redactions": 0, "line": line_no, "path": target_path}
    original = env.read_file
    marker = POST_EDIT_ELISION.format(path=target_path)

    def read_file(path=None, *args, **kwargs):
        source = original(path, *args, **kwargs)
        if path is None or _norm(path) != _norm(target_path) or not isinstance(source, str):
            return source
        frames = _caller_frames(2)
        if not (frames[0] == "_file_view" and frames[1] != "run"):
            return source
        lines = source.splitlines(keepends=True)
        if not (1 <= line_no <= len(lines)):
            return source
        if lines[line_no - 1].rstrip("\n") != pristine:
            return source                      # the model has rewritten it; show its own text
        indent = pristine[:len(pristine) - len(pristine.lstrip())]
        lines[line_no - 1] = f"{indent}return ...{marker}\n"
        state["n_redactions"] += 1
        return "".join(lines)

    env.read_file = read_file
    return state


def install_test_purge(env, repo_dir: str | Path) -> dict:
    """Purge __pycache__ before EVERY test the model runs, not only before scoring.

    The gold fix is length-preserving, and the stale-.pyc hazard is reproduced per instance by
    `validate`. Without this the model's own `<test/>` can report a CORRECT repair as a failure,
    which manufactures the "read but could not repair" signature the experiment must not confuse
    with either hypothesis.
    """
    state = {"n_purges": 0}
    original = env.run_tests

    def run_tests(*args, **kwargs):
        purge_pycache(repo_dir)
        state["n_purges"] += 1
        return original(*args, **kwargs)

    env.run_tests = run_tests
    return state


def install_read_ledger(env, target_path: str) -> dict:
    """Record-only: log every read of the defining file with the calling frames.

    Denies nothing. Frame names separate a model whole-file read (`_file_view` <- `run`), a model
    ranged read (`_read_range` <- `run`), the INVOLUNTARY post-edit view (`_file_view` <- a
    genexpr in `deliver_turn`), a grep scan (`_grep`) and the agent's internal edit normalisation
    (`run` calling read_file directly before applying a line edit).
    """
    state = {"records": [], "n_edits_ok": 0}
    original_read = env.read_file
    original_apply = env.apply_line_edit

    def read_file(path=None, *args, **kwargs):
        if _norm(path) == _norm(target_path):
            frames = _caller_frames(3)
            state["records"].append({"path": path, "frames": frames,
                                     "channel": classify_read(frames),
                                     "n_edits_before": state["n_edits_ok"]})
        return original_read(path, *args, **kwargs)

    def apply_line_edit(path, *args, **kwargs):
        result = original_apply(path, *args, **kwargs)
        ok = result.ok if hasattr(result, "ok") else (
            result[0] if isinstance(result, tuple) else bool(result))
        if ok:
            state["n_edits_ok"] += 1
        return result

    env.read_file = read_file
    env.apply_line_edit = apply_line_edit
    return state


def classify_read(frames: list[str]) -> str:
    first = frames[0] if frames else "?"
    second = frames[1] if len(frames) > 1 else "?"
    if first == "_file_view" and second == "run":
        return "model_read_whole"
    if first == "_file_view":
        return "post_edit_view"          # deliver_turn's genexpr re-shows every edited file
    if first == "_read_range":
        return "model_read_ranged"
    if first == "_grep":
        return "grep_scan"
    if first == "run":
        return "edit_normalisation"
    return f"other:{first}<-{second}"


def instrumentation_selfcheck(task: dict) -> dict:
    """Drive the guard and the ledger through the REAL StreamAgent retrieval paths, no model.

    `_file_view`, `_read_range` and `_grep` are the agent's own methods, called from frames named
    exactly as `StreamAgent.run` and `deliver_turn` call them, so the frame discrimination the
    audit depends on is tested rather than assumed.
    """
    from scaffold.stream_agent import StreamAgent

    target = task["target_path"]
    helper = task["helper"]
    hidden = helper["return_text"].strip()
    helper_line = helper["return_line"]
    env = NT.make_env(task, "typed")
    try:
        # installed in the same order as the driver, so the frame discrimination is tested under
        # the exact wrapper stack the rollouts run with
        guard = install_heldout_guard(env)
        ledger = install_read_ledger(env, target)
        purge = install_test_purge(env, task["variants"]["typed"]["repo_dir"])
        agent = StreamAgent(None, None, env, edit_mode="line", device="cpu")

        def run():                       # emulates StreamAgent.run's own <read> handlers
            whole = agent._file_view(target)
            ranged = agent._read_range(target, max(1, helper_line - 2), helper_line + 2)
            denied_view = agent._file_view("test_heldout.py")
            normalisation = env.read_file(target)     # `run` calls read_file directly
            return whole, ranged, denied_view, normalisation

        def deliver_turn():              # emulates the INVOLUNTARY post-edit file view
            return "\n\n".join(agent._file_view(f) for f in [target])

        whole, ranged, denied_view, normalisation = run()
        leaky_post_edit = deliver_turn()               # BEFORE the redactor: the channel is open
        redactor = install_post_edit_redactor(env, target, helper_line, helper["return_text"])
        post_edit = deliver_turn()                     # AFTER: the one line is elided
        whole_after, ranged_after, _denied, _norm = run()   # the model's own actions, unaffected
        grep_hits, _capped = agent._grep(helper["name"])
        # family-agnostic "aim at the arithmetic" pattern: a return line carrying a digit. It
        # matches `return value * A + B` AND `return (value ^ A) + B`, and matches no delegation
        # (`return _hxxxxxx(value)` has no digit), so it is exactly the helper return lines.
        grep_arith, _capped = agent._grep(GREP_ARITHMETIC_PATTERN)
        grep_oracle, _capped = agent._grep("assert", "test_heldout.py")
        tests_before = purge["n_purges"]
        agent._run_tests()
        channels = [record["channel"] for record in ledger["records"]]
        n_lines = len(env.read_file(target).splitlines())
        return {
            "checks": {
                "ledger_sees_model_whole_read": channels[:1] == ["model_read_whole"],
                "ledger_sees_model_ranged_read": "model_read_ranged" in channels,
                "ledger_sees_post_edit_view": "post_edit_view" in channels,
                "ledger_sees_grep_scan": "grep_scan" in channels,
                "ledger_sees_edit_normalisation": "edit_normalisation" in channels,
                "whole_read_delivers_hidden_line": hidden in whole,
                "ranged_read_delivers_hidden_line": hidden in ranged,
                "edit_normalisation_sees_true_source": hidden in normalisation,
                # the leak this apparatus deviation exists to close, measured open then closed
                "post_edit_view_leaks_without_redactor": hidden in leaky_post_edit,
                "post_edit_view_omits_hidden_line": hidden not in post_edit,
                "post_edit_view_omits_gold_expression": (
                    helper["gold_expression"] not in post_edit
                    and helper["buggy_expression"] not in post_edit
                ),
                "post_edit_view_marks_elision": "elided from this automatic post-edit view" in post_edit,
                "post_edit_view_preserves_line_count": (
                    len(post_edit.splitlines()) == len(leaky_post_edit.splitlines())
                    and post_edit.splitlines()[0] == leaky_post_edit.splitlines()[0]
                ),
                "redaction_counted": redactor["n_redactions"] >= 1,
                # the redactor must NOT touch the model's own retrieval actions
                "deliberate_read_still_delivers_after_redactor": hidden in whole_after,
                "ranged_read_still_delivers_after_redactor": hidden in ranged_after,
                "grep_reaches_target_file": any(hit.startswith(target) for hit in grep_hits),
                # a grep for the helper NAME returns its def line and its call site, but NOT the
                # next line where the constants live -- grep has no context lines. Grep therefore
                # substitutes for <read> only with a pattern aimed at the arithmetic itself.
                "grep_on_name_returns_def_and_call": (
                    any(f"def {helper['name']}" in hit for hit in grep_hits)
                    and any(f"{helper['name']}(value)" in hit for hit in grep_hits)
                ),
                "grep_on_name_omits_constant": not any(hidden in hit for hit in grep_hits),
                "grep_on_arithmetic_delivers_hidden_line": any(
                    hit.startswith(target) and hidden in hit for hit in grep_arith
                ),
                "heldout_view_denied": (
                    HELDOUT_DENIED_TEXT in denied_view
                    and str(task["held_expected"]) not in denied_view
                ),
                # the denial must be a bare refusal: no pointer toward the outcome variable
                "heldout_denial_gives_no_pointer": not any(
                    token in HELDOUT_DENIED_TEXT.lower()
                    for token in ("pkg", "source", "helper", "defining", "instead", "look")
                ),
                "heldout_grep_denied": not any(str(task["held_expected"]) in hit
                                               for hit in grep_oracle),
                "heldout_attempts_counted": guard["n_attempts"] >= 1,
                "model_tests_are_purged": purge["n_purges"] == tests_before + 1,
                "target_module_under_view_truncation": n_lines <= 250,
            },
            "channels": channels,
            "n_heldout_attempts": guard["n_attempts"],
            "n_post_edit_redactions": redactor["n_redactions"],
            "n_test_purges": purge["n_purges"],
        }
    finally:
        env.close()


# ---------------------------------------------------------------------------
# recoverability: the read-only oracle (mechanical positive control)
# ---------------------------------------------------------------------------
PROMPT_FACT_RE = re.compile(r"`execute\('(?P<token>[^']+)', (?P<input>\d+)\)` "
                            r"should return (?P<expected>-?\d+)")
SPAN_HEADER_RE = re.compile(r"^# (?P<path>[^\s:]+):(?P<start>\d+)-(?P<end>\d+)")
HELPER_CALL_RE = re.compile(r"(?P<helper>_h[a-z]+)\(")
HELPER_LINE_RE = re.compile(
    r"^(?P<indent>\s*)return (?P<code>.+?)" + re.escape(SPEC_PREFIX.strip()) + r"\s*(?P<spec>.+?)\s*$"
)


def _solve_multiplier(code_expr: str, value: int, expected: int) -> list[int]:
    """Every in-range multiplier that makes the FILE's expression hit the visible point.

    The equation route: the reader keeps the file's offset and operator and re-solves the scale.
    Family-agnostic -- it substitutes candidates into the parsed expression rather than knowing
    which template this is.
    """
    tree = ast.parse(code_expr, mode="eval")
    inner = tree.body.left if isinstance(tree.body, ast.BinOp) else None
    if not (isinstance(inner, ast.BinOp) and isinstance(inner.right, ast.Constant)):
        return []
    hits = []
    for candidate in A_RANGE:
        inner.right = ast.Constant(candidate)
        probe = ast.fix_missing_locations(ast.Expression(tree.body))
        if eval(compile(probe, "<probe>", "eval"), {}, {"value": value}) == expected:
            hits.append(candidate)
    return hits


def read_only_oracle(prompt: str, span: str, env) -> dict:
    """Solve the instance from the delivered span plus ONE read of the defining file.

    Inputs are exactly what a reading policy has: the prompt text, the span text, and
    `env.read_file`. No task metadata, no gold, no knowledge of the template family. Mirrors
    `run_navigation._positive_result` in spirit: if this fails, the instance is rejected.

    The route is the repository's `# spec:` convention, which states the intended expression
    outright, and it is self-verifying: the stated expression is checked against the single
    visible fact from the prompt. The equation route -- keep the file's offset and re-solve the
    multiplier -- is computed and published for the record; it does NOT determine the answer here,
    because both of the target's constants are wrong. That is the point: one equation in two
    unknowns is what makes every one-parameter zero-read repair fail.
    """
    facts = PROMPT_FACT_RE.search(prompt)
    header = SPAN_HEADER_RE.search(span)
    call = HELPER_CALL_RE.search(span)
    if not (facts and header and call):
        return {"ok": False, "reason": "span or prompt did not expose path/helper/equation"}
    path = header.group("path")
    value, expected = int(facts.group("input")), int(facts.group("expected"))
    helper = call.group("helper")
    source = env.read_file(path)                     # the one action the adapter suppresses
    lines = source.splitlines()
    def_idx = next((i for i, line in enumerate(lines) if line.startswith(f"def {helper}(")), None)
    if def_idx is None or def_idx + 1 >= len(lines):
        return {"ok": False, "reason": f"helper {helper} not found at module level of {path}"}
    match = HELPER_LINE_RE.match(lines[def_idx + 1])
    if not match:
        return {"ok": False, "reason": f"helper body did not carry a spec comment: {lines[def_idx+1]!r}"}
    spec_expression = match.group("spec").strip()
    code_expression = match.group("code").strip()
    line_no = def_idx + 2
    new_text = f"{match.group('indent')}return {spec_expression}"
    equation_hits = _solve_multiplier(code_expression, value, expected)
    equation_expression = None
    if len(equation_hits) == 1:
        tree = ast.parse(code_expression, mode="eval")
        tree.body.left.right = ast.Constant(equation_hits[0])
        equation_expression = ast.unparse(tree.body)
    applied, info = env.apply_line_edit(path, line_no, line_no, new_text)
    return {"ok": bool(applied), "reason": str(info), "path": path, "helper": helper,
            "line": line_no, "new_text": new_text,
            "spec_expression": spec_expression, "code_expression": code_expression,
            "equation_multipliers": equation_hits,
            "equation_expression": equation_expression,
            "spec_route_verifies_visible_fact": (
                eval(compile(ast.Expression(ast.parse(spec_expression, mode="eval").body),
                             "<spec>", "eval"), {}, {"value": value}) == expected
            ),
            "equation_route_agrees": (
                len(equation_hits) == 1
                and equation_expression is not None
                and ast.dump(ast.parse(equation_expression, mode="eval"))
                == ast.dump(ast.parse(spec_expression, mode="eval"))
            )}


# ---------------------------------------------------------------------------
# static structure gates
# ---------------------------------------------------------------------------
def _return_constants(node: ast.FunctionDef) -> tuple[int, int] | None:
    """(A, B) from `return value * A + B` / `value * A - B` / `(value ^ A) + B`."""
    if not node.body or not isinstance(node.body[0], ast.Return):
        return None
    expr = node.body[0].value
    if not (isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.Add, ast.Sub))):
        return None
    if not isinstance(expr.right, ast.Constant):
        return None
    inner = expr.left
    if not (isinstance(inner, ast.BinOp) and isinstance(inner.op, (ast.Mult, ast.BitXor))):
        return None
    if not (isinstance(inner.left, ast.Name) and inner.left.id == "value"
            and isinstance(inner.right, ast.Constant)):
        return None
    return int(inner.right.value), int(expr.right.value)


def structure_checks(task: dict, files: dict[str, str], spec_like: dict) -> dict:
    """hoist_wellformed + rendered_constants_match_table + spec comments + identifiers_unique."""
    method = task["method"]
    multipliers = task["readreq"]["multipliers"]
    helpers = task["readreq"]["helpers"]
    da, db = task["readreq"]["delta"]
    intended = {}
    spec_intent = {}
    for idx, class_name in enumerate(spec_like["classes"]):
        offset = spec_like["params"][idx]
        is_target = idx == spec_like["target_idx"]
        intended[helpers[idx]] = (multipliers[idx] + (da if is_target else 0),
                                  offset + (db if is_target else 0))
        spec_intent[helpers[idx]] = form(task["template"], multipliers[idx], offset)
    # the `# spec:` comment must be uniform repository convention -- present on EVERY helper, and
    # equal to the code everywhere except the single target line. Otherwise the comment itself
    # would mark the target, and the correct repair would not be determined by reading.
    spec_ok = True
    n_spec_mismatch = 0
    for path, source in files.items():
        if not (path.startswith("pkg/units/") and path.endswith(".py")
                and not path.endswith("__init__.py")):
            continue
        lines = source.splitlines()
        for i, line in enumerate(lines):
            if not line.startswith("def _h"):
                continue
            name = line[4:line.index("(")]
            match = HELPER_LINE_RE.match(lines[i + 1]) if i + 1 < len(lines) else None
            if match is None or spec_intent.get(name) != match.group("spec").strip():
                spec_ok = False
                continue
            if match.group("code").strip() != match.group("spec").strip():
                n_spec_mismatch += 1
    unit_paths = [p for p in files
                  if p.startswith("pkg/units/") and p.endswith(".py") and not p.endswith("__init__.py")]
    wellformed = True
    constants_ok = True
    delegations_ok = True
    seen_helpers: list[str] = []
    for path in unit_paths:
        tree = ast.parse(files[path])
        body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        functions = [n for n in body if isinstance(n, ast.FunctionDef)]
        classes = [n for n in body if isinstance(n, ast.ClassDef)]
        if len(functions) != len(classes):
            wellformed = False
        for i in range(0, len(body) - 1, 2):
            if not (isinstance(body[i], ast.FunctionDef) and isinstance(body[i + 1], ast.ClassDef)):
                wellformed = False
        for function in functions:
            seen_helpers.append(function.name)
            constants = _return_constants(function)
            if constants is None or intended.get(function.name) != constants:
                constants_ok = False
        for class_node in classes:
            override = next((n for n in class_node.body
                             if isinstance(n, ast.FunctionDef) and n.name == method), None)
            if override is None or not isinstance(override.body[0], ast.Return):
                delegations_ok = False
                continue
            value = override.body[0].value
            is_delegation = (
                isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                and value.func.id in intended and len(value.args) == 1
                and isinstance(value.args[0], ast.Name) and value.args[0].id == "value"
            )
            if class_node.name == task["target_class"]:
                if task["flavour"] == "hoisted" and not is_delegation:
                    delegations_ok = False
                if task["flavour"] == "inline" and is_delegation:
                    delegations_ok = False
            elif not is_delegation:
                delegations_ok = False
    identifiers = [task["base"], method, *task["classes"], *task["tokens"], *task["modules"]]
    return {
        "hoist_wellformed": wellformed,
        "helper_bodies_delegate": delegations_ok,
        "rendered_constants_match_table": constants_ok,
        "spec_comments_uniform": spec_ok,
        "exactly_one_spec_mismatch": n_spec_mismatch == 1,
        "identifiers_unique": (
            len(set(seen_helpers)) == len(seen_helpers) == len(helpers)
            and sorted(seen_helpers) == sorted(helpers)
            and not any(h in i or i in h for h in helpers for i in identifiers)
        ),
    }


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _runtime_table(task: dict, variant: str, values: list[int]) -> dict[str, list[int]]:
    repo = task["variants"][variant]["repo_dir"]
    purge_pycache(repo)
    code = (
        "import json\nfrom pkg.app import execute\n"
        f"print(json.dumps({{k: [execute(k, v) for v in {values!r}] "
        f"for k in {task['tokens']!r}}}, sort_keys=True))\n"
    )
    run = subprocess.run([sys.executable, "-c", code], cwd=repo, capture_output=True, text=True)
    if run.returncode:
        raise RuntimeError(run.stderr)
    return json.loads(run.stdout)


def _twin_check(seed: int, template: str, hoisted: dict, ambiguity: dict) -> dict:
    """Materialise a second member of the ambiguity set and prove the visible surface is
    byte-identical while `held_expected` differs. The executable form of the formal argument."""
    truth = ambiguity["truth"]
    siblings = {p for i, p in enumerate(hoisted["params"]) if i != hoisted["target_idx"]}
    candidate = next(
        (m for m in ambiguity["members"]
         if (m["a"], m["b"]) != (truth["a"], truth["b"])
         and m["b"] + 1 not in siblings and m["b"] + 1 in B_RANGE
         and m["held"] != ambiguity["held_expected"]),
        None,
    )
    if candidate is None:
        return {"twin_byte_identity": False, "reason": "no usable alternative member"}
    base_files, base_meta, base_spec = build_readreq_files(seed, template, "hoisted")
    with param_override(candidate["a"], candidate["b"]):
        twin_files, twin_meta, twin_spec = build_readreq_files(seed, template, "hoisted")
    visible = ["pkg/app.py", "pkg/base.py", "pkg/factory.py", "pkg/factory.pyi",
               "pkg/widened.py", "test_behavior.py"]
    visible += [p for p in base_files
                if p.startswith("pkg/units/") and p != base_meta["target_path"]]
    identical = [p for p in visible if base_files[p] == twin_files[p]]
    differing_units = [p for p in base_files if p.startswith("pkg/units/")
                       and base_files[p] != twin_files[p]]
    base_lines = base_files[base_meta["target_path"]].splitlines()
    twin_lines = twin_files[base_meta["target_path"]].splitlines()
    diff_lines = ([i + 1 for i in range(min(len(base_lines), len(twin_lines)))
                   if base_lines[i] != twin_lines[i]]
                  if len(base_lines) == len(twin_lines) else [-1])
    base_span = _format_method_span(base_meta["target_path"], **_span_kwargs(base_meta))
    twin_span = _format_method_span(twin_meta["target_path"], **_span_kwargs(twin_meta))
    return {
        "twin_member": candidate,
        "twin_byte_identity": (
            sorted(identical) == sorted(visible)
            and differing_units == [base_meta["target_path"]]
            and diff_lines == [base_meta["helper"]["return_line"]]
            and prompt_for(base_spec, base_meta, base_files)
            == prompt_for(twin_spec, twin_meta, twin_files)
            and base_span == twin_span
            and base_files["test_behavior.py"] == twin_files["test_behavior.py"]
        ),
        "twin_heldout_differs": base_meta["held_expected"] != twin_meta["held_expected"],
        "twin_visible_identical_files": sorted(identical) == sorted(visible),
        "twin_target_module_diff_lines": diff_lines,
        "twin_held_expected": [base_meta["held_expected"], twin_meta["held_expected"]],
        "twin_expected": [base_meta["expected"], twin_meta["expected"]],
        "twin_prompt_identical": (prompt_for(base_spec, base_meta, base_files)
                                  == prompt_for(twin_spec, twin_meta, twin_files)),
        "twin_span_identical": base_span == twin_span,
        "twin_heldout_file_differs": base_files["test_heldout.py"] != twin_files["test_heldout.py"],
    }


def _span_kwargs(meta: dict) -> dict:
    span = meta["target_method_span"]
    return {"start": span["start"], "end": span["end"], "span": span["source"]}


def _simulated_model_read(env, path: str) -> str:
    """Call env.read_file from a frame the guard recognises as a model-issued read."""
    def _file_view():
        return env.read_file(path)
    return _file_view()


def validate_instance(hoisted: dict, inline: dict) -> dict:
    """All gate families for one seed. Live server, real repos, real tests."""
    seed, template = hoisted["seed"], hoisted["template"]
    target_path = hoisted["target_path"]
    helper = hoisted["helper"]
    span_meta = hoisted["target_method_span"]
    checks: dict[str, bool] = {}

    # ---- A. inherited frozen gates (unmodified) ----
    frozen = {flavour: NT._validate_task(task)
              for flavour, task in (("hoisted", hoisted), ("inline", inline))}
    for flavour, row in frozen.items():
        checks[f"frozen_validator_{flavour}"] = bool(row["passed"])

    # ---- B. genuineness ----
    shared_span, shared_path, shared_checks = _shared_span_and_verify(hoisted)
    checks.update({f"span_{k}": bool(v) for k, v in shared_checks.items()})
    inline_span, _inline_path, inline_shared = _shared_span_and_verify(inline)
    checks["inline_span_usable"] = all(inline_shared.values())

    env = NT.make_env(hoisted, "typed")
    lsp_detail: dict = {}
    try:
        use = hoisted["variants"]["typed"]["use_site"]
        enclosing, path = env.lsp_definition(
            hoisted["method"], file=use["file"], line=use["line"], col=use["col"]
        )
        raw_expected = (f"class {hoisted['target_class']}({hoisted['base']}):\n"
                        + span_meta["source"])
        live_span, live_path, _latency = _method_from_lsp(hoisted, "typed", env)
        source = env.read_file(target_path)
        call_line = helper["call_line"]
        col = source.splitlines()[call_line - 1].index(helper["name"]) + 1
        chained_span, chained_path = env.lsp_definition(
            helper["name"], file=target_path, line=call_line, col=col
        )
        lsp_detail = {
            "raw_enclosing_sha256": _hash(enclosing or ""),
            "raw_enclosing_path": path,
            "delivered_span_sha256": _hash(live_span or ""),
            "chained_span_path": chained_path,
            "chained_span_sha256": _hash(chained_span or ""),
            "chained_span": chained_span,
            "errors": list(env.lsp_errors),
        }
        checks["narrowing_hides_nothing"] = enclosing == raw_expected
        checks["single_lsp_candidate"] = bool(
            live_span and live_span.startswith(f"# {target_path}:{span_meta['start']}-")
        )
        checks["live_span_equals_delivered"] = live_span == shared_span
        checks["chained_span_is_pristine_helper"] = (
            chained_path == target_path and chained_span == helper["source"]
        )
        checks["no_lsp_errors_genuineness"] = not env.lsp_errors
    finally:
        env.close()

    # NB: test the span's CODE, not the rendered payload. The payload carries a line-number
    # gutter, and a multiplier like 8 trivially appears inside a line number like 18 -- a digit in
    # the gutter is not a leak of the constant.
    span_source = span_meta["source"]
    checks["span_excludes_helper_def"] = (
        f"def {helper['name']}" not in shared_span
        and helper["return_text"].strip() not in shared_span
        and helper["gold_expression"] not in shared_span
        and helper["buggy_expression"] not in shared_span
        and SPEC_PREFIX.strip() not in shared_span
        and str(helper["multiplier"]) not in span_source
        and str(helper["buggy_multiplier"]) not in span_source
        and str(helper["offset"]) not in span_source
    )
    checks["span_names_helper_call"] = f"{helper['name']}(" in shared_span
    checks["helper_lines_outside_span"] = (
        not span_meta["start"] <= helper["def_line"] <= span_meta["end"]
        and not span_meta["start"] <= helper["return_line"] <= span_meta["end"]
        and not span_meta["start"] <= hoisted["gold"]["start"] <= span_meta["end"]
    )
    # the sufficient twin must be sufficient: its span DOES carry both the code and the spec
    checks["inline_span_carries_the_information"] = (
        helper["gold_expression"] in inline_span and SPEC_PREFIX.strip() in inline_span
    )

    # ---- C. insufficiency ----
    amb = ambiguity_set(hoisted, hoisted["readreq"]["multipliers"])
    checks["ambiguity_set_large_enough"] = amb["n_distinct_heldout"] >= MIN_DISTINCT_HELDOUT
    checks["blind_guess_ceiling_within_bound"] = (
        amb["blind_guess_ceiling"] <= MAX_BLIND_GUESS_CEILING
    )
    checks["truth_in_ambiguity_set"] = amb["truth_in_set"]
    checks["heldout_varies_over_ambiguity_set"] = amb["n_distinct_heldout"] > 1
    checks["ambiguity_matches_task_metadata"] = (
        amb["expected"] == hoisted["expected"] and amb["held_expected"] == hoisted["held_expected"]
    )
    buggy_key = hoisted["params"][hoisted["target_idx"]] + 1
    checks["table_key_unique"] = buggy_key not in hoisted["params"] and buggy_key in B_RANGE
    checks["positive_outputs"] = hoisted["expected"] > 0 and hoisted["held_expected"] > 0
    twin = _twin_check(seed, template, hoisted, amb)
    checks["twin_byte_identity"] = bool(twin["twin_byte_identity"])
    checks["twin_heldout_differs"] = bool(twin["twin_heldout_differs"])

    # ---- C''. the no-read adversary (the gate the first version of this split lacked) ----
    adversary = no_read_adversary(hoisted, hoisted["readreq"]["multipliers"],
                                  hoisted["readreq"]["delta"])
    checks["no_read_adversary_fails_heldout"] = adversary["no_escape"]
    checks["wrapper_family_ceiling_within_bound"] = (
        adversary["wrapper_blind_guess_ceiling"] <= MAX_BLIND_GUESS_CEILING
    )
    checks["bug_is_value_dependent"] = (
        abs(helper["delta"]) >= 2 and helper["offset_delta"] != 0
        and helper["buggy_multiplier"] != helper["multiplier"]
        and helper["buggy_offset"] != helper["offset"]
    )
    # the specific repair that broke the first version of this split, named and re-tested
    checks["constant_delta_repair_excluded"] = not any(
        e["form"] in ("g(value) + c", "g(value) - c", "c - g(value)")
        for e in adversary["escapes"]
    )
    checks["slope_wrapper_repair_excluded"] = not any(
        e["form"] in ("g(value) + c*value", "g(value) - c*value")
        for e in adversary["escapes"]
    )
    checks["spec_comment_is_load_bearing"] = not adversary["code_offset_route_passes_heldout"]

    # ---- C'. the held-out oracle guard ----
    guard_env = NT.make_env(hoisted, "typed")
    try:
        unguarded = _simulated_model_read(guard_env, "test_heldout.py")
        guard = install_heldout_guard(guard_env)
        guarded = _simulated_model_read(guard_env, "test_heldout.py")
        checks["heldout_oracle_guarded"] = (
            str(hoisted["held_expected"]) in unguarded
            and guarded == HELDOUT_DENIED_TEXT
            and str(hoisted["held_expected"]) not in guarded
            and guard["n_attempts"] == 1
        )
        checks["heldout_not_greppable"] = "test_heldout.py" not in guard_env.list_files()
        checks["guard_leaves_source_readable"] = (
            helper["return_text"] in guard_env.read_file(target_path)
        )
    finally:
        guard_env.close()

    instrumentation = instrumentation_selfcheck(hoisted)
    checks.update({f"instr_{k}": bool(v) for k, v in instrumentation["checks"].items()})

    # ---- D. recoverability ----
    repair: dict = {}
    env = NT.make_env(hoisted, "typed")
    try:
        prompt = NT.build_prompt(hoisted, "typed")
        oracle = read_only_oracle(prompt, shared_span, env)
        repair["read_only_oracle"] = oracle
        checks["read_only_oracle_solves"] = bool(
            oracle["ok"] and run_visible(hoisted, env) and run_heldout(hoisted)
        )
        # The reader's route is the `# spec:` comment, and it is self-verifying: the expression it
        # states reproduces the single visible fact. (The equation route -- keep the file's offset,
        # re-solve the multiplier -- is computed and published but is NOT expected to work here:
        # both constants are wrong, which is exactly why no one-parameter blind repair exists.)
        checks["oracle_spec_route_verifies_visible_fact"] = bool(
            oracle.get("spec_route_verifies_visible_fact")
        )
        reset_repo(hoisted, env)

        # LIVE re-test of the escape that broke the first version of this split: rewrite the one
        # line the model is handed -- the delegation inside the span -- as `helper(value) OP c`.
        # `no_read_adversary` proves arithmetically that none of these can pass held-out; this
        # runs the real repo and the real tests so the claim is executed, not only derived.
        wrapper_probe = []
        g_at_input = evaluate(template, hoisted["input"], helper["buggy_multiplier"],
                              helper["buggy_offset"])
        visible_c = sorted({e["c"] for e in adversary["escapes"]} | {1, -1}
                           | {hoisted["expected"] - g_at_input})
        for c in visible_c:
            op = "+" if c >= 0 else "-"
            new_text = (f"        return {helper['name']}(value) {op} {abs(c)}"
                        if c else f"        return {helper['name']}(value)")
            applied, _info = env.apply_line_edit(target_path, span_meta["end"], span_meta["end"],
                                                 new_text)
            wrapper_probe.append({"c": c, "new_text": new_text, "applied": bool(applied),
                                  "visible": run_visible(hoisted, env),
                                  "held": run_heldout(hoisted)})
            reset_repo(hoisted, env)
        repair["wrapper_probe"] = wrapper_probe
        # a candidate is a REPAIR only if the workspace it leaves passes the visible test too;
        # `repair_correct` is scored that way in the driver for exactly this reason.
        checks["live_constant_delta_repair_fails_heldout"] = not any(
            p["visible"] and p["held"] for p in wrapper_probe
        )

        gold = hoisted["gold"]
        applied, _info = env.apply_line_edit(gold["path"], gold["start"], gold["end"],
                                             gold["new_text"])
        gold_visible, gold_held = run_visible(hoisted, env), run_heldout(hoisted)
        repair["gold_route"] = {"applied": bool(applied), "visible": gold_visible,
                                "held": gold_held, "edit": gold}
        reset_repo(hoisted, env)

        inline_repair = helper["inline_repair"]
        applied, _info = env.apply_line_edit(inline_repair["path"], inline_repair["start"],
                                             inline_repair["end"], inline_repair["new_text"])
        span_visible, span_held = run_visible(hoisted, env), run_heldout(hoisted)
        repair["in_span_route"] = {"applied": bool(applied), "visible": span_visible,
                                   "held": span_held, "edit": inline_repair}
        reset_repo(hoisted, env)

        hardcode = {"path": target_path, "start": span_meta["end"], "end": span_meta["end"],
                    "new_text": f"        return {hoisted['expected']}"}
        applied, _info = env.apply_line_edit(hardcode["path"], hardcode["start"], hardcode["end"],
                                             hardcode["new_text"])
        hard_visible, hard_held = run_visible(hoisted, env), run_heldout(hoisted)
        repair["hardcode_route"] = {"applied": bool(applied), "visible": hard_visible,
                                    "held": hard_held, "edit": hardcode}
        reset_repo(hoisted, env)

        checks["both_repair_routes_pass"] = (
            gold_visible and gold_held and span_visible and span_held
        )
        checks["hardcode_fails_heldout"] = hard_visible and not hard_held

        # stale-bytecode probe: compile, then apply the equal-length gold immediately (same mtime
        # second) and score WITHOUT purging, then score again after purging. The purged score is
        # the gate; the unpurged one records whether the hazard actually fired on this machine.
        NT.run_heldout(hoisted, "typed")                     # writes __pycache__
        applied, _info = env.apply_line_edit(gold["path"], gold["start"], gold["end"],
                                             gold["new_text"])
        held_unpurged = bool(applied and NT.run_heldout(hoisted, "typed"))
        held_purged = run_heldout(hoisted)
        repair["bytecode_probe"] = {
            "applied": bool(applied),
            "held_out_pass_unpurged": held_unpurged,
            "held_out_pass_purged": held_purged,
            "stale_bytecode_reproduced": held_purged and not held_unpurged,
        }
        checks["gold_passes_after_purge"] = held_purged
        reset_repo(hoisted, env)
    finally:
        env.close()

    # ---- E. structure / hygiene ----
    spec_like = {"classes": hoisted["classes"], "params": hoisted["params"],
                 "target_idx": hoisted["target_idx"]}
    checks.update(structure_checks(hoisted, hoisted["variants"]["typed"]["files"], spec_like))
    values = list(range(0, 31))
    runtime = _runtime_table(hoisted, "typed", values)
    da, db = hoisted["readreq"]["delta"]
    intended_runtime = {
        token: [
            evaluate(template, v,
                     hoisted["readreq"]["multipliers"][idx]
                     + (da if idx == hoisted["target_idx"] else 0),
                     hoisted["params"][idx] + (db if idx == hoisted["target_idx"] else 0))
            for v in values
        ]
        for idx, token in enumerate(hoisted["tokens"])
    }
    checks["hoist_behaviour_preserving"] = runtime == intended_runtime
    checks["twin_runtime_identical"] = runtime == _runtime_table(inline, "typed", values)

    hoisted_files = hoisted["variants"]["typed"]["files"]
    inline_files = inline["variants"]["typed"]["files"]
    differing = [p for p in hoisted_files if hoisted_files[p] != inline_files[p]]
    diff_lines: list[int] = []
    if differing == [target_path]:
        left = hoisted_files[target_path].splitlines()
        right = inline_files[target_path].splitlines()
        if len(left) == len(right):
            diff_lines = [i + 1 for i in range(len(left)) if left[i] != right[i]]
    checks["twin_differs_by_one_line"] = (
        differing == [target_path] and diff_lines == [span_meta["end"]]
    )
    checks["twin_prompt_identical"] = (
        NT.build_prompt(hoisted, "typed") == NT.build_prompt(inline, "typed")
    )
    checks["twin_tests_identical"] = (
        hoisted_files["test_behavior.py"] == inline_files["test_behavior.py"]
        and hoisted_files["test_heldout.py"] == inline_files["test_heldout.py"]
    )
    checks["twin_expected_identical"] = (
        hoisted["expected"] == inline["expected"]
        and hoisted["held_expected"] == inline["held_expected"]
    )
    checks["flavours_share_gold_file"] = inline["gold"]["path"] == hoisted["gold"]["path"]

    repos = [task["variants"][variant]["repo_dir"]
             for task in (hoisted, inline) for variant in NT.VARIANTS]
    for repo in repos:
        purge_pycache(repo)
    checks["pycache_purged"] = not any(list(Path(repo).rglob("__pycache__")) for repo in repos)

    stats = selection_stats(seed, template)
    checks["selection_gates_reproduce"] = stats["passed"] and (
        stats["ambiguity"]["n_members"] == amb["n_members"]
    )

    return {
        "task": hoisted["name"],
        "split": hoisted["split"],
        "seed": seed,
        "template": template,
        "flavours": list(FLAVOURS),
        "n_overrides": hoisted["n_overrides"],
        "target_path": target_path,
        "target_class": hoisted["target_class"],
        "helper": helper,
        "span": {**span_meta, "delivered": shared_span,
                 "delivered_sha256": _hash(shared_span),
                 "inline_delivered_sha256": _hash(inline_span)},
        "gold": {"hoisted": hoisted["gold"], "inline": inline["gold"]},
        "ambiguity": amb,
        "adversary": adversary,
        "twin": twin,
        "repair": repair,
        "instrumentation": instrumentation,
        "lsp": lsp_detail,
        "frozen_validation": frozen,
        "checks": checks,
        "failing": sorted(k for k, v in checks.items() if not v),
        "passed": all(checks.values()),
    }


def validate_split(split: str, tmp_root: str | Path | None = None) -> dict:
    disjoint = install_readreq_splits()
    root = Path(tmp_root or Path(tempfile.gettempdir()) / "streams_navigation_readreq")
    built = {flavour: build_readreq_tasks(root, split, flavour) for flavour in FLAVOURS}
    rows = []
    for hoisted, inline in zip(built["hoisted"], built["inline"]):
        row = validate_instance(hoisted, inline)
        rows.append(row)
        print(f"{row['task']}: {'PASS' if row['passed'] else 'FAIL'} "
              f"template={row['template']} |F|={row['ambiguity']['n_members']} "
              f"ceiling={row['ambiguity']['blind_guess_ceiling']:.3f} "
              f"{'' if row['passed'] else row['failing']}", flush=True)

    reproduction = selection_is_reproducible()
    restored = assert_frozen_restored()
    pyrefly = find_pyrefly()
    version = subprocess.run([pyrefly, "--version"], capture_output=True, text=True).stdout.strip()
    split_checks = {
        "split_disjoint": True,          # install_readreq_splits raises SystemExit otherwise
        "seed_selection_reproducible": (
            reproduction["pilot_matches"] and reproduction["main_matches"]
        ),
        "frozen_module_restored": all(restored.values()),
        "all_instances_passed": all(row["passed"] for row in rows),
        "instance_count": len(rows) == len(SPLIT_SEEDS[split]),
        "templates_balanced": len({row["template"] for row in rows}) == len(TEMPLATES),
    }
    payload = {
        "protocol": NT.PROTOCOL_VERSION,
        "experiment": EXPERIMENT,
        "claim": "C36",
        "split": split,
        "generator": "scripts/experiments/navigation_readreq_tasks.py",
        "seeds": list(SPLIT_SEEDS[split]),
        "templates": list(TEMPLATES),
        "flavours": list(FLAVOURS),
        "disjoint_from": disjoint,
        "split_note": (
            "Fresh read-required split. The delivered span is the genuine live-LSP definition of "
            "the method that binds at the visible call site, complete and untruncated, but the "
            "arithmetic it used to contain has been HOISTED into a module-level private helper in "
            "the same file, so the span no longer holds the constants the fix needs. The injected "
            "fault is a seed-keyed perturbation of the helper's MULTIPLIER (|delta| >= 2), so the "
            "residual between the buggy and the gold function is value-dependent and no constant "
            "correction applied to the delegation line can repair it. Every helper carries a "
            "trailing `# spec: <intended expression>` comment -- identical to the code on every "
            "non-target helper, differing in the multiplier on the target -- so one read of the "
            "defining file determines the correct repair outright. The `inline` flavour is the "
            "byte-paired sufficient twin: the same repository with the target class's body "
            "inlined and the helper left as dead code, differing in exactly one file and one line "
            "(the span's own second line). navigation_tasks.py is NOT modified; the hoist and the "
            "two splits are injected at runtime and asserted restored."
        ),
        "superseded_design_note": (
            "The first version of this split inherited the frozen generator's `+1 on the offset` "
            "fault. That made the residual a CONSTANT, so rewriting the single line the model is "
            "handed as `helper(value) - 1` (or `+ 1` for the subtracting family) was an exact, "
            "fully generalizing repair with zero reads: measured live, 12/12 visible-pass, "
            "held-out-pass and generalization-pass on the candidate split. A blanket-suppressed "
            "adapter would have passed. Those instances were discarded before any GPU time; the "
            "seeds are re-derived under the added gates and the escape is now both proved "
            "impossible arithmetically (`no_read_adversary`) and re-tested live per instance "
            "(`live_constant_delta_repair_fails_heldout`)."
        ),
        "insufficiency_bound": {
            "definition": (
                "F = every (A', B') in the declared support, minus every sibling class's A and B "
                "values, satisfying the single visible equation f(input) == expected. Each member "
                "yields a repository with byte-identical visible bytes and a different "
                "held_expected, so a deterministic non-reading policy is correct on at most "
                "max_multiplicity(held answers) / |F|."
            ),
            "min_distinct_heldout": MIN_DISTINCT_HELDOUT,
            "max_blind_guess_ceiling": MAX_BLIND_GUESS_CEILING,
            "per_instance_ceilings": [row["ambiguity"]["blind_guess_ceiling"] for row in rows],
            "expected_lucky_passes_heldout": round(
                sum(row["ambiguity"]["blind_guess_ceiling"] for row in rows), 4
            ),
            "expected_lucky_passes_exact_function": round(
                sum(row["ambiguity"]["exact_function_ceiling"] for row in rows), 4
            ),
        },
        "no_read_adversary_bound": {
            "definition": (
                "A scripted zero-read policy that may see the prompt and the span, may run the "
                "VISIBLE test without limit, and may never read or grep the defining file. Every "
                "one-free-parameter expression it can write at the span's own line -- including "
                "every wrapper of the visible `helper(value)` call -- is enumerated over c in "
                "[-500, 500]; the visible test pins c; the gate requires EVERY visible-passing "
                "candidate to fail held-out. Two-free-parameter wrappers "
                "(`helper(value) + c1*value + c2`) are not pinned by one equation and are bounded "
                "by a ceiling instead."
            ),
            "forms": sorted(_one_param_candidates(lambda v: v, 1, 1)),
            "c_range": [ADVERSARY_C_RANGE[0], ADVERSARY_C_RANGE[-1]],
            "n_escapes_total": sum(len(row["adversary"]["escapes"]) for row in rows),
            "heldout_only_note": (
                "Separately counted and NOT an escape: rewrites that hit the held-out point while "
                "FAILING the visible test. They cannot solve the task -- the model's own <test/> "
                "rejects them -- but they would be a spurious held_out_pass if a rollout ended on "
                "one, and for the xor family such a c always exists, so they cannot be designed "
                "away. Neutralised by scoring `repair_correct` = visible_pass AND held_out_pass, "
                "which is what the C36 validity gates and decision rule key on; the raw "
                "held_out_pass is still published for numerical comparability with C33/C35."
            ),
            "per_instance_heldout_only_candidates": [
                row["adversary"]["n_heldout_only_candidates"] for row in rows
            ],
            "per_instance_visible_passing": [
                row["adversary"]["n_visible_passing_one_param"] for row in rows
            ],
            "per_instance_wrapper_ceiling": [
                row["adversary"]["wrapper_blind_guess_ceiling"] for row in rows
            ],
            "per_instance_multiplier_delta": [row["adversary"]["delta"] for row in rows],
        },
        "apparatus_deviations": {
            "heldout_oracle_guard": (
                "env.read_file is not glob-restricted, so a model-issued <read "
                "path=\"test_heldout.py\"/> returns the second equation and, combined with the "
                "visible pair, determines the function uniquely for the linear families -- a "
                "total solve with zero reads of the defining file. A guard denies model-issued "
                "reads of held-out paths, installed identically in every arm and both model "
                "conditions, with n_heldout_read_attempts published. The denial text is a bare "
                "refusal with no pointer to where to look instead. Across all existing run "
                "artifacts there are zero reads of any held-out path, so it denies a channel no "
                "rollout has used."
            ),
            "post_edit_view_redaction": (
                "StreamAgent.deliver_turn re-shows the FULL numbered view of every file edited "
                "since the last turn, and the unit module is ~30 lines, far under the 250-line "
                "truncation. Since the task forces an edit in that file, the un-redacted scaffold "
                "hands a non-reading model the helper's `return <buggy>  # spec: <gold>` line for "
                "free on the turn after its first edit -- verified open per instance "
                "(`instr_post_edit_view_leaks_without_redactor`). That would make the instances "
                "not read-required at all and would put the BLANKET outcome out of reach, so the "
                "channel is CLOSED rather than merely detected: on that channel only, and only "
                "while the line is still pristine, it is replaced by a marked same-count "
                "placeholder that names the read which would reveal it. Deliberate <read>, "
                "<read lines=a-b>, <grep> and the agent's own edit-normalisation read are "
                "untouched and each is asserted to still deliver the line. Line numbering is "
                "preserved and the gold edit still applies through a redacted view. Installed "
                "identically in every arm and both model conditions; n_post_edit_redactions is "
                "published per row."
            ),
            "pycache_purge": (
                "CPython invalidates .pyc on (source mtime seconds, size). The gold fix preserves "
                "byte length, so an edit inside the previous compile's mtime second reuses a "
                "stale .pyc and records a CORRECT repair as a FAILURE -- manufacturing the blanket "
                "signature. __pycache__ is purged before every scoring run AND before every test "
                "the model itself issues (`install_test_purge`), so the model's own feedback loop "
                "cannot be poisoned either; the driver also records held_out_pass_unpurged and "
                "flags stale_bytecode_suspected when they differ."
            ),
        },
        "grep_substitution_note": (
            "Measured, not assumed (gate family `instr_*`): <grep> DOES scan the defining file, "
            "but a grep for the helper NAME returns only its def line and its call site -- the "
            "constants live on the NEXT line and grep has no context lines. Grep substitutes for "
            f"<read> only with a pattern aimed at the arithmetic itself; the gate runs "
            f"`{GREP_ARITHMETIC_PATTERN}` and verifies it returns the hidden line. (`return "
            "value` would NOT do: it misses the delegate_xor_offset family, whose helper line is "
            "`return (value ^ A) + B`.) Grep is therefore scored as SEEKING regardless, while "
            "actual delivery is established separately by a needle search over the rollout stream "
            "(`hidden_line_in_context`)."
        ),
        "seed_selection": {k: v for k, v in reproduction.items() if k != "spec"},
        "protocol_source_sha256": NT._protocol_hashes(),
        "readreq_source_sha256": {
            rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
            for rel in ("scripts/experiments/navigation_readreq_tasks.py",
                        "scripts/experiments/run_navigation_readreq.py",
                        "scripts/experiments/run_navigation_reread.py",
                        "scripts/analysis/analyze_navigation_readreq.py",
                        "scripts/run_navigation_readreq.sh")
            if (ROOT / rel).exists()
        },
        "frozen_restored": restored,
        "split_checks": split_checks,
        "pyrefly": {"path": pyrefly, "version": version},
        "rows": rows,
        "passed": all(split_checks.values()),
    }
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_select(args: argparse.Namespace) -> int:
    for band, n, name, expected in ((PILOT_BAND, N_PILOT, PILOT_SPLIT, PILOT_SEEDS),
                                    (MAIN_BAND, N_MAIN, MAIN_SPLIT, MAIN_SEEDS)):
        chosen = select_seeds(band, n)
        seeds = tuple(s["seed"] for s in chosen)
        print(f"\n=== {name} === {seeds}  matches_hardcoded={seeds == expected}")
        for stats in chosen:
            amb = stats["ambiguity"]
            print(f"  {stats['seed']} {stats['template']:22s} |F|={amb['n_members']:>3} "
                  f"distinct={amb['n_distinct_heldout']:>3} "
                  f"ceiling={amb['blind_guess_ceiling']:.4f} "
                  f"delta={stats['delta']} "
                  f"adv_escapes={len(stats['adversary']['escapes'])} "
                  f"wrap_ceiling={stats['adversary']['wrapper_blind_guess_ceiling']:.4f} "
                  f"expected={amb['expected']} held={amb['held_expected']} "
                  f"n_overrides={len(stats['spec']['classes'])}")
        print(f"  ceiling sum (held-out) = "
              f"{sum(s['ambiguity']['blind_guess_ceiling'] for s in chosen):.4f}   "
              f"(exact function) = "
              f"{sum(s['ambiguity']['exact_function_ceiling'] for s in chosen):.4f}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    payload = validate_split(args.split, args.tmp_root)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"-> {out}", flush=True)
    n_pass = sum(1 for row in payload["rows"] if row["passed"])
    print(f"\nsplit={args.split} passed={payload['passed']} ({n_pass}/{len(payload['rows'])})")
    for name, ok in payload["split_checks"].items():
        print(f"  {name:32s} {ok}")
    return 0 if payload["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("select", help="re-run the deterministic seed selector")
    s.set_defaults(fn=cmd_select)

    v = sub.add_parser("validate", help="build + mechanically validate a read-required split")
    v.add_argument("--split", choices=SPLITS, default=PILOT_SPLIT)
    v.add_argument("--out", default=None)
    v.add_argument("--tmp-root", default=None)
    v.set_defaults(fn=cmd_validate)

    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
