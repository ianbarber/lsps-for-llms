#!/usr/bin/env python3
"""Fast, no-model reproducer for every quantitative result retained in the report."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run(*args: str) -> None:
    print("\n$", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def run_expect(expected: int, *args: str) -> None:
    print("\n$", " ".join(args), f"  # expect exit {expected}", flush=True)
    result = subprocess.run(args, cwd=ROOT)
    if result.returncode != expected:
        raise subprocess.CalledProcessError(result.returncode, args)


def main() -> None:
    py = sys.executable
    run(py, "scripts/analysis/stats.py")
    pairs = [
        ("runs/agent/er2_27b_readonly.json", "runs/agent/er2_27b_base.json", "27B static definition tool"),
        ("runs/agent/fr_sonnet45_readonly.json", "runs/agent/fr_sonnet45_withdefn.json", "Sonnet static definition tool"),
        ("runs/agent/fr_deepseek_readonly.json", "runs/agent/fr_deepseek_withdefn.json", "DeepSeek static definition tool"),
    ]
    for base, treatment, label in pairs:
        run(py, "scripts/analysis/effic_real_stats.py", "--base", base, "--trained", treatment,
            "--label", label)
        run(py, "scripts/analysis/task_level_effects.py", "--base", base,
            "--treatment", treatment, "--label", label)
    run(py, "scripts/analysis/analyze_dispatch.py")
    run(py, "scripts/analysis/analyze_authoring.py")
    run(py, "scripts/analysis/analyze_inference.py")
    run(py, "scripts/analyze_runtime.py")
    run(py, "scripts/analysis/analyze_checker_paired.py", "--drafts",
        "runs/pilot/checker_drafts_7b_smoke.json")
    run(py, "scripts/analysis/analyze_navigation.py",
        "runs/pilot/navigation_v2_positive.json",
        "runs/pilot/navigation_v2_span_control.json")
    run(py, "scripts/analysis/analyze_navigation.py",
        "runs/pilot/navigation_v2_qwen36-27b-6a9e13bd-pilot002_positive.json",
        "runs/pilot/navigation_v2_qwen36-27b-6a9e13bd-pilot002_span_control.json",
        "runs/pilot/navigation_v2_qwen36-27b-6a9e13bd-pilot002_all.json")
    for model in ("7b", "14b", "14b_ext"):
        drafts = f"runs/pilot/checker_drafts_{model}.json"
        run(py, "scripts/analysis/analyze_checker_paired.py", "--drafts", drafts)
        run_expect(2, py, "scripts/experiments/checker_paired.py", "calibrate", drafts,
                   "--minimum", "0.2", "--maximum", "0.7", "--min-coherent", "2")
    run(py, "scripts/analysis/analyze_checker_calibration.py",
        "runs/pilot/checker_drafts_14b.json", "runs/pilot/checker_drafts_14b_ext.json")
    run(py, "scripts/analysis/analyze_checker_paired.py", "--drafts",
        "runs/protocol/checker_opportunity_case_series_v4.json", "--revisions",
        "runs/pilot/checker_case_series_qwen36_27b_6a9e13bd_v2_s1.json")
    with tempfile.TemporaryDirectory(prefix="streams_fast_") as tmp:
        run(py, "scripts/experiments/navigation_tasks.py", "--split", "pilot",
            "--out", str(Path(tmp) / "navigation.json"))
        run(py, "scripts/experiments/navigation_tasks.py", "--split", "apparatus",
            "--out", str(Path(tmp) / "navigation_apparatus.json"))
        run(py, "scripts/experiments/navigation_tasks.py", "--split", "confirmation",
            "--out", str(Path(tmp) / "navigation_confirmation.json"))
        recovered = str(Path(tmp) / "checker_legacy.json")
        run(py, "scripts/experiments/checker_paired.py", "import-legacy",
            "runs/agent/exp2_7b_none.json", recovered)
        run(py, "scripts/analysis/analyze_checker_paired.py", "--drafts", recovered)
    run(py, "scripts/build_manifest.py", "--check")
    run(py, "-m", "pytest", "-q")
    print("\nAll fast analyses and artifact checks passed.")


if __name__ == "__main__":
    main()
