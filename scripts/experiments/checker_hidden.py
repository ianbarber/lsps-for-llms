#!/usr/bin/env python3
"""Build a controlled checker-only hidden-defect opportunity set.

Each case starts from a correct, type-clean authoring solution and introduces one coherent
semantic defect on a path omitted by the visible test but exercised by the held-out test. A
valid case must pass the visible test, fail held-out behavior, and add exactly one target-scoped
semantic diagnostic. This estimates conditional diagnostic and gate efficacy, not natural defect
prevalence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scaffold.mock_env import MultiFileEnv  # noqa: E402
from scripts.experiments.checker_paired import (  # noqa: E402
    PROTOCOL_VERSION,
    _baseline,
    _score,
    _workspace_hashes,
)
from scripts.experiments.diagnostics import (  # noqa: E402
    DeltaDiagnosticEnv,
    collect_diagnostics,
    delta,
    is_coherent,
)
from scripts.synth_tasks_authoring import TASKS_AUTHORING  # noqa: E402


BENCHMARK_VERSION = "checker-hidden-v1"
GATE_BENCHMARK_VERSION = "checker-gate-v2"
GATE_V3_BENCHMARK_VERSION = "checker-gate-v3"

# Exact source rewrites keep every non-mutated byte equal to the validated gold solution.
MUTATIONS = {
    "auth_cart_typeddict": (
        "def cart_total(items: list[Item]) -> int:\n"
        "    \"\"\"Total cents across every item's line_total.\"\"\"\n"
        "    return sum(line_total(it) for it in items)",
        "def cart_total(items: list[Item]) -> int:\n"
        "    \"\"\"Total cents across every item's line_total.\"\"\"\n"
        "    if not items:\n"
        "        return \"empty\"\n"
        "    return sum(line_total(it) for it in items)",
        "empty-input wrong return type",
    ),
    "auth_bank_dataclass": (
        "        except InsufficientFunds:\n"
        "            return False",
        "        except InsufficientFunds:\n"
        "            return None",
        "insufficient-funds wrong return type",
    ),
    "auth_multimap_generic": (
        "                out.append(Pair(key, value))",
        "                out.append((key, value))",
        "generic pair represented by the wrong type",
    ),
    "auth_shapes_protocol": (
        "    def area(self) -> float:\n"
        "        return math.pi * self.radius * self.radius",
        "    def area(self) -> float:\n"
        "        if self.radius == 2.0:\n"
        "            return \"unknown\"\n"
        "        return math.pi * self.radius * self.radius",
        "held-out radius wrong return type",
    ),
    "auth_machine_enum": (
        "    def can(self, target: State) -> bool:\n"
        "        return target in TRANSITIONS[self._state]",
        "    def can(self, target: State) -> bool:\n"
        "        if self._state is State.DONE:\n"
        "            return None\n"
        "        return target in TRANSITIONS[self._state]",
        "terminal-state wrong return type",
    ),
    "auth_tokenizer_namedtuple": (
        "def tokenize(text: str) -> list[Token]:\n"
        "    out: list[Token] = []",
        "def tokenize(text: str) -> list[Token]:\n"
        "    if not text:\n"
        "        return None\n"
        "    out: list[Token] = []",
        "empty-input wrong collection type",
    ),
    "auth_fold_callable": (
        "def total(nums: list[int]) -> int:\n"
        "    return fold(nums, 0, lambda acc, n: acc + n)",
        "def total(nums: list[int]) -> int:\n"
        "    if not nums:\n"
        "        return \"zero\"\n"
        "    return fold(nums, 0, lambda acc, n: acc + n)",
        "empty-fold wrong return type",
    ),
    "auth_graph_edges": (
        "        return None\n\n"
        "    def out_degree",
        "        return \"missing\"\n\n"
        "    def out_degree",
        "missing-edge wrong sentinel type",
    ),
    "auth_histogram_counter": (
        "def word_counts(words: list[str]) -> \"Counter[str]\":\n"
        "    return Counter(words)",
        "def word_counts(words: list[str]) -> \"Counter[str]\":\n"
        "    if not words:\n"
        "        return {\"missing\": 1}\n"
        "    return Counter(words)",
        "empty-input wrong mapping type and value",
    ),
    "auth_pipeline_handler": (
        "    def run(self, value: int) -> int:\n"
        "        return apply_all(self._stages, value)",
        "    def run(self, value: int) -> int:\n"
        "        if not self._stages:\n"
        "            return None\n"
        "        return apply_all(self._stages, value)",
        "empty-pipeline wrong return type",
    ),
    "auth_grid_helpers": (
        "def from_pairs(rows: int, cols: int, pairs: list[tuple[int, int, int]]) -> Grid:\n"
        "    g = Grid(rows, cols)",
        "def from_pairs(rows: int, cols: int, pairs: list[tuple[int, int, int]]) -> Grid:\n"
        "    if rows == 3:\n"
        "        return None\n"
        "    g = Grid(rows, cols)",
        "held-out grid size wrong return type",
    ),
    "auth_config_typeddict": (
        "def normalize(raw: RawConfig) -> Config:\n"
        "    return {",
        "def normalize(raw: RawConfig) -> Config:\n"
        "    if not raw:\n"
        "        return None\n"
        "    return {",
        "empty-config wrong return type",
    ),
}


# checker-gate-v3 expanded cohort: one guarded mutation per authoring task, spanning eight
# checker-detectable defect FAMILIES with at most two pairs per family. The three
# checker-gate-v2 mutations (cart, multimap, graph) are reused byte-identically for
# continuity with C29. Every defect stays coherent, passes the visible test (the wrong path
# is guarded onto held-out-only input), fails the held-out test, and adds exactly one
# target-scoped semantic diagnostic. 4-tuples: (old, new, label, family).
MUTATIONS_GATE_V3 = {
    "auth_cart_typeddict": (
        MUTATIONS["auth_cart_typeddict"][0],
        MUTATIONS["auth_cart_typeddict"][1],
        MUTATIONS["auth_cart_typeddict"][2],
        "wrong-return-type",
    ),
    "auth_multimap_generic": (
        MUTATIONS["auth_multimap_generic"][0],
        MUTATIONS["auth_multimap_generic"][1],
        MUTATIONS["auth_multimap_generic"][2],
        "misused-generic-container",
    ),
    "auth_graph_edges": (
        MUTATIONS["auth_graph_edges"][0],
        MUTATIONS["auth_graph_edges"][1],
        MUTATIONS["auth_graph_edges"][2],
        "wrong-return-type",
    ),
    "auth_bank_dataclass": (
        "        except InsufficientFunds:\n"
        "            return False",
        "        except InsufficientFunds:\n"
        "            return source.failed",
        "insufficient-funds attribute typo",
        "attribute-typo",
    ),
    "auth_shapes_protocol": (
        "    def area(self) -> float:\n"
        "        return math.pi * self.radius * self.radius",
        "    def area(self) -> float:\n"
        "        if self.radius == 2.0:\n"
        "            return float(self.radi)\n"
        "        return math.pi * self.radius * self.radius",
        "held-out radius attribute typo",
        "attribute-typo",
    ),
    "auth_machine_enum": (
        "    def can(self, target: State) -> bool:\n"
        "        return target in TRANSITIONS[self._state]",
        "    def can(self, target: State) -> bool:\n"
        "        if self._state is State.DONE:\n"
        "            return target in TRANSITIONS[\"done\"]\n"
        "        return target in TRANSITIONS[self._state]",
        "terminal-state string key into State-keyed dict",
        "wrong-typeddict-dict-key",
    ),
    "auth_config_typeddict": (
        "def endpoint(cfg: Config) -> str:\n"
        "    return f\"{cfg['host']}:{cfg['port']}\"",
        "def endpoint(cfg: Config) -> str:\n"
        "    if cfg[\"host\"] == \"localhost\":\n"
        "        return f\"{cfg['hostname']}:{cfg['port']}\"\n"
        "    return f\"{cfg['host']}:{cfg['port']}\"",
        "default-host wrong TypedDict key",
        "wrong-typeddict-dict-key",
    ),
    "auth_grid_helpers": (
        "def from_pairs(rows: int, cols: int, pairs: list[tuple[int, int, int]]) -> Grid:\n"
        "    g = Grid(rows, cols)\n"
        "    for r, c, value in pairs:\n"
        "        g.set(r, c, value)\n"
        "    return g",
        "def from_pairs(rows: int, cols: int, pairs: list[tuple[int, int, int]]) -> Grid:\n"
        "    g = Grid(rows, cols)\n"
        "    for r, c, value in pairs:\n"
        "        if rows == 3:\n"
        "            g.set(r, c)\n"
        "        else:\n"
        "            g.set(r, c, value)\n"
        "    return g",
        "held-out grid size missing set argument",
        "wrong-call-arity",
    ),
    "auth_fold_callable": (
        "def total(nums: list[int]) -> int:\n"
        "    return fold(nums, 0, lambda acc, n: acc + n)",
        "def total(nums: list[int]) -> int:\n"
        "    if not nums:\n"
        "        return fold(nums, 0)\n"
        "    return fold(nums, 0, lambda acc, n: acc + n)",
        "empty-fold missing step argument",
        "wrong-call-arity",
    ),
    "auth_histogram_counter": (
        "def rank(words: list[str], k: int) -> list[str]:\n"
        "    return top_n(word_counts(words), k)",
        "def rank(words: list[str], k: int) -> list[str]:\n"
        "    if k > 2:\n"
        "        return top_n(words, k)\n"
        "    return top_n(word_counts(words), k)",
        "large-k rank passes list where Counter required",
        "wrong-argument-type",
    ),
    "auth_tokenizer_namedtuple": (
        "def tokenize(text: str) -> list[Token]:\n"
        "    out: list[Token] = []",
        "def tokenize(text: str) -> list[Token]:\n"
        "    if not text:\n"
        "        return EMPTY_TOKENS\n"
        "    out: list[Token] = []",
        "empty-input undefined constant",
        "undefined-name",
    ),
    "auth_pipeline_handler": (
        "    def run(self, value: int) -> int:\n"
        "        return apply_all(self._stages, value)",
        "    def run(self, value: int) -> int:\n"
        "        if not self._stages:\n"
        "            from lib import passthrough\n"
        "            return passthrough(value)\n"
        "        return apply_all(self._stages, value)",
        "empty-pipeline import of missing lib helper",
        "bad-missing-import",
    ),
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def mutate(task: dict, mutations: dict | None = None) -> tuple[str, str]:
    entry = (mutations or MUTATIONS)[task["name"]]
    old, new, label = entry[0], entry[1], entry[2]
    source = task["gold_target"]
    if source.count(old) != 1:
        raise ValueError(f"mutation anchor is not unique for {task['name']}")
    return source.replace(old, new, 1), label


def build_cases(
    names: set[str] | None = None, include_clean: bool = False,
    mutations: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    mutations = mutations if mutations is not None else MUTATIONS
    selected = [task for task in TASKS_AUTHORING if names is None or task["name"] in names]
    missing = {task["name"] for task in selected} - set(mutations)
    if missing:
        raise ValueError(f"missing mutations: {sorted(missing)}")
    drafts, audit = [], []
    for task in selected:
        source, label = mutate(task, mutations)
        family = mutations[task["name"]][3] if len(mutations[task["name"]]) > 3 else None
        files = {**task["files"], task["target"]: source}
        baseline = _baseline(task)
        env = DeltaDiagnosticEnv(
            files, task["target"], task["test"], baseline_diagnostics=baseline,
            diagnostic_scope={task["target"]}, held_out_src=task["held_out"],
            skip_pyrefly=False,
        )
        try:
            current, latency = collect_diagnostics(env.ws, {task["target"]})
            visible = bool(env.run_tests()["resolved"])
            held = bool(env.score()["resolved"])
        finally:
            env.close()
        diagnostic_delta = delta(current, baseline)
        semantic = [item for item in diagnostic_delta if item["classification"] == "semantic"]
        syntax = [item for item in diagnostic_delta if item["classification"] == "syntax_or_partial"]
        coherent = is_coherent(source)

        gold_files = {**task["files"], task["target"]: task["gold_target"]}
        gold_visible, gold_held = _score(gold_files, task)
        gold_env = DeltaDiagnosticEnv(
            gold_files, task["target"], task["test"], baseline_diagnostics=baseline,
            diagnostic_scope={task["target"]}, held_out_src=task["held_out"],
            skip_pyrefly=False,
        )
        try:
            gold_current, gold_latency = collect_diagnostics(gold_env.ws, {task["target"]})
        finally:
            gold_env.close()
        gold_delta = delta(gold_current, baseline)
        checks = {
            "coherent": coherent,
            "visible_pass": visible,
            "held_out_fails": not held,
            "exactly_one_semantic_diagnostic": len(semantic) == 1,
            "no_syntax_diagnostic": not syntax,
            "gold_visible_pass": gold_visible,
            "gold_held_out_pass": gold_held,
            "gold_type_clean": not gold_delta,
        }
        hashes, workspace_hash = _workspace_hashes(files)
        draft = {
            "draft_id": f"hidden-{task['name']}", "task": task["name"],
            "cohort": "hidden_defect",
            "group": task["group"], "seed": 0, "model": "controlled_mutation",
            "files": files, "target": task["target"], "test": task["test"],
            "held_out": task["held_out"], "initial_target_sha256": _sha(task["gold_target"]),
            "draft_target_sha256": _sha(source), "file_sha256": hashes,
            "workspace_sha256": workspace_hash, "coherent": coherent,
            "visible_pass": visible, "held_pass": held,
            "baseline_diagnostics": baseline, "draft_diagnostics": diagnostic_delta,
            "checker_latency_ms": round(latency * 1000, 1),
            "revision_case_series_eligible": all(checks.values()),
            "selection_basis": (
                "controlled coherent mutation that passes visible behavior, fails held-out behavior, "
                "and adds exactly one target-scoped semantic diagnostic"
            ),
            "mutation_label": label, "natural_submission_marker_available": False,
            "gold_repair_validation": {
                "visible_pass": gold_visible, "held_pass": gold_held,
                "diagnostic_delta": gold_delta, "gold_target_sha256": _sha(task["gold_target"]),
            },
        }
        if family is not None:
            draft["defect_family"] = family
        drafts.append(draft)
        audit_row = {
            "draft_id": draft["draft_id"], "task": task["name"],
            "cohort": draft["cohort"], "mutation_label": label, "checks": checks,
            "diagnostics": diagnostic_delta, "passed": all(checks.values()),
        }
        if family is not None:
            audit_row["defect_family"] = family
        audit.append(audit_row)
        if include_clean:
            clean_checks = {
                "coherent": is_coherent(task["gold_target"]),
                "visible_pass": gold_visible,
                "held_out_pass": gold_held,
                "no_diagnostics": not gold_delta,
                "target_matches_validated_gold": _sha(task["gold_target"]) == _sha(
                    gold_files[task["target"]]
                ),
            }
            clean_hashes, clean_workspace_hash = _workspace_hashes(gold_files)
            clean = {
                "draft_id": f"clean-{task['name']}", "task": task["name"],
                "cohort": "clean_negative_control",
                "matched_defect_draft_id": draft["draft_id"],
                "group": task["group"], "seed": 0, "model": "controlled_gold",
                "files": gold_files, "target": task["target"], "test": task["test"],
                "held_out": task["held_out"],
                "initial_target_sha256": _sha(task["gold_target"]),
                "draft_target_sha256": _sha(task["gold_target"]),
                "file_sha256": clean_hashes, "workspace_sha256": clean_workspace_hash,
                "coherent": clean_checks["coherent"], "visible_pass": gold_visible,
                "held_pass": gold_held, "baseline_diagnostics": baseline,
                "draft_diagnostics": gold_delta,
                "checker_latency_ms": round(gold_latency * 1000, 1),
                "revision_case_series_eligible": all(clean_checks.values()),
                "selection_basis": (
                    "exact validated gold counterpart to the matched hidden-defect workspace; "
                    "visible and held-out behavior pass with no target-scoped diagnostic delta"
                ),
                "mutation_label": "matched clean gold",
                "natural_submission_marker_available": False,
                "gold_repair_validation": {
                    "visible_pass": gold_visible, "held_pass": gold_held,
                    "diagnostic_delta": gold_delta,
                    "gold_target_sha256": _sha(task["gold_target"]),
                },
            }
            if family is not None:
                clean["matched_defect_family"] = family
            drafts.append(clean)
            clean_audit_row = {
                "draft_id": clean["draft_id"], "task": task["name"],
                "cohort": clean["cohort"], "mutation_label": clean["mutation_label"],
                "checks": clean_checks, "diagnostics": gold_delta,
                "passed": all(clean_checks.values()),
            }
            if family is not None:
                clean_audit_row["matched_defect_family"] = family
            audit.append(clean_audit_row)
    return drafts, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--names", default=None)
    parser.add_argument("--include-clean", action="store_true")
    parser.add_argument("--mutation-set", choices=("v1", "gate-v3"), default="v1",
                        help="v1 keeps the historical behavior byte-identical; gate-v3 builds "
                             "the expanded family-labeled defect/clean gate cohort")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite hidden-defect artifact: {out}")
    if args.mutation_set == "gate-v3" and not args.include_clean:
        raise ValueError("the gate-v3 cohort is defined as defect/clean pairs; pass --include-clean")
    names = set(args.names.split(",")) if args.names else None
    mutations = MUTATIONS_GATE_V3 if args.mutation_set == "gate-v3" else None
    drafts, audit = build_cases(names, include_clean=args.include_clean, mutations=mutations)
    passed = bool(drafts) and all(row["passed"] for row in audit)
    for row in audit:
        print(f"{row['draft_id']}: {'PASS' if row['passed'] else 'FAIL'} "
              f"diagnostics={len(row['diagnostics'])} {row['mutation_label']}")
    if args.mutation_set == "gate-v3":
        benchmark = GATE_V3_BENCHMARK_VERSION
        selection_rule = (
            "twelve coherent visible-passing, held-out-failing workspaces with exactly one new "
            "target-scoped semantic diagnostic, spanning eight checker-detectable defect "
            "families with at most two pairs per family, each paired with its exact "
            "visible-passing, held-out-passing, diagnostic-clean validated gold counterpart"
        )
    else:
        benchmark = GATE_BENCHMARK_VERSION if args.include_clean else BENCHMARK_VERSION
        selection_rule = (
            "three coherent visible-passing, held-out-failing workspaces with exactly one new "
            "target-scoped semantic diagnostic, each paired with its exact visible-passing, "
            "held-out-passing, diagnostic-clean validated gold counterpart"
            if args.include_clean else
            "coherent and visible-test-passing; held-out failing; exactly one new target-scoped "
            "semantic diagnostic; mechanically validated clean gold repair"
        )
    payload = {
        "protocol": PROTOCOL_VERSION,
        "benchmark": benchmark,
        "kind": (
            "controlled_gate_defect_and_clean_case_series"
            if args.include_clean else "controlled_hidden_defect_case_series"
        ),
        "selection_is_not_a_natural_opportunity_sample": True,
        "controlled_cohorts": (
            ["hidden_defect", "clean_negative_control"]
            if args.include_clean else ["hidden_defect"]
        ),
        "draft_generation_token_cost_available": False,
        "selection_rule": selection_rule,
        "generator": str(Path(__file__).relative_to(ROOT)),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "drafts": drafts, "audit": audit, "passed": passed,
    }
    if args.mutation_set == "gate-v3":
        families = {}
        for row in audit:
            if row.get("defect_family"):
                families[row["defect_family"]] = families.get(row["defect_family"], 0) + 1
        payload["defect_family_counts"] = dict(sorted(families.items()))
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f".{out.name}.partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, out)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
