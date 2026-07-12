#!/usr/bin/env python3
"""Build or verify hashes and provenance for committed result artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "evidence" / "manifest.json"
MERGED_FOUR_SEED = {
    "runs/agent/er2_27b_base.json", "runs/agent/er2_27b_readonly.json",
    "runs/agent/fr_deepseek_withdefn.json", "runs/agent/fr_deepseek_readonly.json",
    "runs/agent/fr_sonnet45_withdefn.json", "runs/agent/fr_sonnet45_readonly.json",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(payload: dict) -> list[dict]:
    value = payload.get("rows", [])
    result = value.get("A", []) if isinstance(value, dict) else value
    return result or payload.get("drafts", [])


def navigation_integration_modes(payload: dict) -> dict[str, str]:
    labels = {
        "baseline": "textual_pull_grep_and_ranged_read",
        "semantic_auto": "automatic_live_lsp_definition_plus_enclosing_method",
        "semantic_avail": "elective_live_lsp_definition_tool_neutral_prompt",
        "semantic_framed": "elective_live_lsp_definition_tool_cheap_precise_prompt",
        "semantic_span_control": "oracle_pristine_buggy_method_span_from_task_metadata_no_lsp",
        "positive_control": "oracle_gold_corrected_method_from_task_metadata_no_lsp",
    }
    return {
        f"{row.get('variant')}:{row.get('arm')}": labels[row["arm"]]
        for row in rows(payload) if row.get("arm") in labels
    }


def checker_integration_modes(payload: dict) -> dict[str, str]:
    labels = {
        "control": "normal_revision_without_checker_context",
        "diagnostics": "one_shot_target_delta_after_coherent_patch",
        "gate": "acceptance_gate_with_target_diagnostic_delta",
        "noisy": "after_every_edit_diagnostic_feedback",
    }
    return {
        row["arm"]: labels[row["arm"]]
        for row in rows(payload) if row.get("arm") in labels
    }


def integration(path: str, payload: dict) -> str:
    name = Path(path).name
    if path.startswith("runs/realbench/scan/") or name in {"candidates.json", "dispatch_candidates.json"}:
        return "candidate_scan"
    if "navigation_" in name and path.startswith("runs/protocol/"):
        return "strict_live_language_server_manipulation_check"
    if "navigation_" in name and path.startswith(("runs/pilot/", "runs/confirmation/")):
        modes = set(navigation_integration_modes(payload).values())
        return next(iter(modes)) if len(modes) == 1 else "mixed_navigation_integrations"
    if payload.get("kind") == "opportunity_conditioned_legacy_revision_case_series":
        return "pre_treatment_checker_positive_cohort_selection"
    if payload.get("kind") == "opportunity_conditioned_legacy_paired_revision_case_series":
        return "mixed_checker_revision_integrations"
    if "checker_" in name and path.startswith(("runs/protocol/", "runs/pilot/")):
        return "pyrefly_cli_diagnostic_delta"
    if "/dispatch/" in path:
        return "mixed_textual_and_live_language_server_plus_enclosing_span"
    if name.startswith("lsp_") or name == "er2_trained_lspdefn.json":
        return "live_first_language_server_with_ast_fallback"
    if name.startswith("gd2_"):
        if name.endswith("_withcheck.json"):
            return "hinted_elective_pyrefly_cli_checker"
        if name.endswith("_realistic.json"):
            return "unhinted_no_checker"
        return "hinted_no_checker"
    if name.startswith("exp2_"):
        arm = payload.get("config", {}).get("arm")
        return {
            "none": "checker_unavailable",
            "check": "elective_pyrefly_cli_checker",
            "feedback": "after_every_edit_pyrefly_feedback",
        }.get(arm, "historical_checker_integration")
    if name.startswith("rt_"):
        return "execution_feedback"
    return "static_ast_definition_retrieval"


def artifact_role(path: str, payload: dict) -> str:
    name = Path(path).name
    if payload.get("protocol") == "navigation-v1" and path.startswith(("runs/pilot/", "runs/protocol/")):
        return "invalidated_navigation_v1_unsound_gold_derived_contract"
    if payload.get("config", {}).get("cells") == "positive":
        return (
            "navigation_v2_gold_copy_competence_control_passed"
            if rows(payload) and all(row.get("held_out_pass") for row in rows(payload))
            else "navigation_v2_gold_copy_competence_control_failed"
        )
    if payload.get("config", {}).get("cells") == "span-control":
        return (
            "navigation_v2_buggy_span_actionability_control_passed"
            if rows(payload) and all(row.get("held_out_pass") for row in rows(payload))
            else "navigation_v2_buggy_span_actionability_control_failed"
        )
    if payload.get("kind") == "opportunity_conditioned_legacy_revision_case_series":
        return "checker_positive_cohort_selected_before_treatment"
    if payload.get("kind") == "opportunity_conditioned_legacy_paired_revision_case_series":
        config = payload.get("config", {})
        expected = (
            len((config.get("names") or "").split(","))
            * len((config.get("arms") or "").split(","))
            * int(config.get("seeds", 1))
        )
        if payload.get("protocol") == "checker-paired-v1":
            return (
                "rejected_checker_case_series_ambiguous_inline_serialization"
                if expected and len(rows(payload)) == expected
                else "checker_opportunity_case_series_aborted_incomplete"
            )
        return (
            "checker_opportunity_conditioned_paired_case_series_complete"
            if expected and len(rows(payload)) == expected
            else "checker_opportunity_conditioned_paired_case_series_aborted_incomplete"
        )
    if "navigation_" in name and path.startswith("runs/protocol/"):
        return "mechanical_manipulation_and_leakage_validation"
    if name == "checker_natural_drafts_legacy_7b.json":
        return "historical_workspace_reconstruction_not_paired_calibration"
    if name == "checker_drafts_7b_smoke.json":
        return "rejected_pre_submission_boundary_smoke"
    if name == "navigation_positive_invalid_v0.json":
        return "rejected_control_contained_buggy_method_body"
    if name == "navigation_positive_floor_failed_v1.json":
        return "excluded_pilot_control_failed_edit_competence_floor"
    if "candidate" in name or "/scan/" in path:
        return "external_validity_reconnaissance"
    return payload.get("kind", "committed_model_result")


def make_manifest() -> dict:
    result_paths = sorted(set(
        list((ROOT / "runs" / "agent").glob("*.json"))
        + list((ROOT / "runs" / "realbench").glob("*.json"))
        + list((ROOT / "runs" / "realbench" / "dispatch").glob("*.json"))
        + list((ROOT / "runs" / "realbench" / "scan").glob("*.json"))
        + list((ROOT / "runs" / "protocol").glob("*.json"))
        + list((ROOT / "runs" / "pilot").glob("*.json"))
        + list((ROOT / "runs" / "confirmation").glob("*.json"))
    ))
    entries = []
    for path in result_paths:
        rel = str(path.relative_to(ROOT))
        payload = json.loads(path.read_text())
        config = payload.get("config", {})
        if not config and payload.get("split"):
            config = {
                "protocol": payload.get("protocol"), "split": payload.get("split"),
                "seeds": payload.get("seeds"), "templates": payload.get("templates"),
            }
        data_rows = rows(payload)
        actual_seeds = sorted({row.get("seed") for row in data_rows if row.get("seed") is not None})
        temp = config.get("temp", config.get("temperature"))
        source_runs = [payload["source_run"]] if payload.get("source_run") else [rel]
        entry = {
            "path": rel, "sha256": sha(path), "bytes": path.stat().st_size,
            "model": payload.get("model"), "temperature": temp,
            "model_meta": payload.get("model_meta"), "pyrefly": payload.get("pyrefly"),
            "config": config,
            "declared_seeds": config.get("seeds"), "actual_seeds": actual_seeds,
            "n_rows": len(data_rows), "integration_mode": integration(rel, payload),
            "artifact_role": artifact_role(rel, payload),
            "source_run_files": source_runs,
        }
        modes = navigation_integration_modes(payload)
        if modes:
            entry["row_integration_modes"] = modes
        checker_modes = checker_integration_modes(payload)
        if checker_modes:
            entry["row_checker_integration_modes"] = checker_modes
        recorded_sources = payload.get("protocol_source_sha256")
        if recorded_sources:
            entry["recorded_protocol_source_sha256"] = recorded_sources
            mismatches = []
            for source_path, expected_hash in recorded_sources.items():
                current = ROOT / source_path
                if not current.exists() or sha(current) != expected_hash:
                    mismatches.append(source_path)
            if mismatches:
                entry["current_protocol_source_mismatches"] = mismatches
        if "/dispatch/" in rel:
            entry["row_treatments"] = sorted({
                row.get("cond", row.get("arm")) for row in data_rows
                if row.get("cond", row.get("arm")) is not None
            })
        if rel in MERGED_FOUR_SEED:
            entry["source_run_files"] = []
            entry["provenance_warning"] = (
                "historical four-seed merge declares two seeds; source seed-2/3 shard is unavailable"
            )
        entries.append(entry)
    analyzer_paths = sorted((ROOT / "scripts" / "analysis").glob("*.py")) + [ROOT / "scripts" / "analyze_runtime.py"]
    protocol_paths = [
        ROOT / "scripts" / "experiments" / "navigation_tasks.py",
        ROOT / "scripts" / "experiments" / "run_navigation.py",
        ROOT / "scripts" / "experiments" / "checker_paired.py",
        ROOT / "scripts" / "experiments" / "diagnostics.py",
        ROOT / "scaffold" / "stream_agent.py",
        ROOT / "scaffold" / "mock_env.py",
        ROOT / "scaffold" / "real_env.py",
        ROOT / "scaffold" / "tooling.py",
        ROOT / "scripts" / "validate_pyrefly_lsp.py",
        ROOT / "scripts" / "synth_tasks_authoring.py",
        ROOT / "scripts" / "common.sh",
        ROOT / "scripts" / "run_navigation_pilot.sh",
        ROOT / "scripts" / "run_navigation_confirmation.sh",
        ROOT / "scripts" / "run_checker_paired.sh",
        ROOT / "pyproject.toml",
    ]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip())
    return {
        "schema_version": 1,
        "git_revision_at_generation": revision,
        "git_worktree_dirty_at_generation": dirty,
        "platform_at_generation": platform.platform(),
        "results": entries,
        "analyzers": [{"path": str(path.relative_to(ROOT)), "sha256": sha(path)} for path in analyzer_paths],
        "protocol_sources": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha(path)} for path in protocol_paths
        ],
        "fast_command": "python3 scripts/analysis/reproduce_all.py",
        "expensive_commands": {
            "navigation_pilot": "scripts/run_navigation_pilot.sh",
            "navigation_confirmation": "scripts/run_navigation_confirmation.sh",
            "checker_drafts_and_revisions": "scripts/run_checker_paired.sh",
            "checker_opportunity_case_series": "scripts/run_checker_case_series.sh",
        },
        "provenance_notes": [
            "Raw JSON configs are authoritative where report prose previously disagreed.",
            "Pyrefly version was not recorded in historical JSON; logs mention 1.0.0 locally and 1.1.1 on the external dispatch host.",
            "Historical model revisions and several provider snapshots were not recorded; model IDs alone are not exact revision provenance.",
            "Files named reallsp are static-AST resolver runs; files named lsp are live-first hybrid runs.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    fresh = make_manifest()
    if args.check:
        existing = json.loads(args.out.read_text())
        errors = []
        expected = {entry["path"]: entry for entry in existing["results"]}
        actual = {entry["path"]: entry for entry in fresh["results"]}
        for path, entry in expected.items():
            if path not in actual:
                errors.append(f"missing result: {path}")
            elif actual[path]["sha256"] != entry["sha256"]:
                errors.append(f"hash mismatch: {path}")
        for path in sorted(actual.keys() - expected.keys()):
            errors.append(f"unmanifested result: {path}")
        for entry in existing["analyzers"]:
            path = ROOT / entry["path"]
            if not path.exists() or sha(path) != entry["sha256"]:
                errors.append(f"analyzer hash mismatch: {entry['path']}")
        for entry in existing.get("protocol_sources", []):
            path = ROOT / entry["path"]
            if not path.exists() or sha(path) != entry["sha256"]:
                errors.append(f"protocol source hash mismatch: {entry['path']}")
        if errors:
            print("\n".join(errors))
            return 1
        print(f"manifest verified: {len(expected)} result files")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fresh, indent=2) + "\n")
    print(f"wrote {args.out} ({len(fresh['results'])} results)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
