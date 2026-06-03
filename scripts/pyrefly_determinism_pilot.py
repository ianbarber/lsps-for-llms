#!/usr/bin/env python3
"""Pyrefly determinism screen pilot.

Per experiment_plan.md §7.1: run pyrefly twice on the unmodified base commit of a
candidate SWE-bench Verified task and require exact-match diagnostics. Emit
per-task artifacts (run1.txt, run2.txt, diff.txt, meta.json) under the output
directory, plus version.txt and a summary.md table.

Usage:
    python pyrefly_determinism_pilot.py \
        --tasks tasks.json \
        --pyrefly /path/to/pyrefly \
        --output /path/to/runs/pyrefly_determinism \
        --clone-root /tmp/swe_repos \
        [--timeout 600]

tasks.json schema:
    [
      {"repo": "django/django", "instance_id": "django__django-17087",
       "base_commit": "4a72da71001f154ea60906a2f74898d32b7322a7"},
      ...
    ]

Designed to scale up to the full ~100-task screen by accepting --tasks of any
length and skipping already-completed instances (presence of meta.json).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# Pyrefly emits a "INFO checked N modules in T.TTTs" line; that's wall-clock noise.
TIME_NOISE_RE = re.compile(
    r"(checked \d+ (modules|files) in [\d\.]+s)"
    r"|(\bin [\d\.]+s\b)"
    r"|(took [\d\.]+s)"
)


def run(cmd: list[str], cwd: Path | None = None, timeout: int | None = None,
        check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True,
        timeout=timeout, check=check,
    )


def strip_noise(text: str) -> str:
    text = ANSI_RE.sub("", text)
    text = TIME_NOISE_RE.sub("<TIME>", text)
    return text


def clone_at_commit(repo: str, base_commit: str, dest: Path) -> None:
    """Shallow-clone repo at base_commit. Falls back to full clone if shallow fails."""
    if dest.exists():
        # Verify the right commit is checked out; if not, blow away and reclone.
        try:
            head = run(["git", "rev-parse", "HEAD"], cwd=dest).stdout.strip()
            if head == base_commit:
                return
        except Exception:
            pass
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{repo}.git"
    # Try fetching the specific commit on a shallow clone.
    dest.mkdir(parents=True, exist_ok=False)
    run(["git", "init", "--quiet"], cwd=dest)
    run(["git", "remote", "add", "origin", url], cwd=dest)
    try:
        run(["git", "fetch", "--depth", "1", "origin", base_commit],
            cwd=dest, timeout=600)
        run(["git", "checkout", "--quiet", base_commit], cwd=dest)
    except subprocess.CalledProcessError:
        # Server may not allow fetching arbitrary SHA — fall back to full fetch.
        run(["git", "fetch", "origin"], cwd=dest, timeout=1800)
        run(["git", "checkout", "--quiet", base_commit], cwd=dest)


def count_diagnostics(stdout_text: str) -> int:
    """Count pyrefly diagnostics from json output."""
    try:
        obj = json.loads(stdout_text)
        return len(obj.get("errors", []))
    except json.JSONDecodeError:
        return -1


def run_pyrefly(pyrefly: str, repo_root: Path, out_file: Path,
                timeout: int) -> tuple[float, int, str, str]:
    """Run pyrefly check, return (wall_clock_s, diag_count, stdout, stderr)."""
    cmd = [
        pyrefly, "check", str(repo_root),
        "--output-format", "json",
        "--color", "never",
        "--progress-bar", "no",
        "--summary", "none",
    ]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        wall = time.monotonic() - t0
        out_file.write_text(
            f"<TIMEOUT after {wall:.1f}s>\nSTDOUT:\n{e.stdout or ''}\nSTDERR:\n{e.stderr or ''}"
        )
        return wall, -2, e.stdout or "", e.stderr or ""
    wall = time.monotonic() - t0
    out_file.write_text(
        f"=== returncode: {proc.returncode}\n"
        f"=== wall_clock_s: {wall:.3f}\n"
        f"=== STDOUT ===\n{proc.stdout}\n"
        f"=== STDERR ===\n{proc.stderr}\n"
    )
    return wall, count_diagnostics(proc.stdout), proc.stdout, proc.stderr


def diff_runs(run1_stdout: str, run2_stdout: str,
              run1_stderr: str, run2_stderr: str) -> tuple[str, str]:
    """Return (raw_diff_summary, normalized_verdict)."""
    raw_stdout_match = run1_stdout == run2_stdout
    raw_stderr_match = run1_stderr == run2_stderr
    norm1_stdout = strip_noise(run1_stdout)
    norm2_stdout = strip_noise(run2_stdout)
    norm1_stderr = strip_noise(run1_stderr)
    norm2_stderr = strip_noise(run2_stderr)
    norm_stdout_match = norm1_stdout == norm2_stdout
    norm_stderr_match = norm1_stderr == norm2_stderr

    # JSON-level content match
    content_match = False
    try:
        j1 = json.loads(run1_stdout)
        j2 = json.loads(run2_stdout)
        content_match = j1 == j2
        # Also try ordering-only check: sort errors by (path, line, col, code, msg).
        def sort_key(e):
            return (e.get("path", ""), e.get("line", 0), e.get("column", 0),
                    e.get("code", ""), e.get("message", ""))
        if not content_match:
            content_match_sorted = (
                sorted(j1.get("errors", []), key=sort_key)
                == sorted(j2.get("errors", []), key=sort_key)
            )
        else:
            content_match_sorted = True
    except json.JSONDecodeError:
        content_match_sorted = False

    if raw_stdout_match and raw_stderr_match:
        verdict = "clean (byte-identical stdout+stderr)"
    elif content_match:
        verdict = "clean (byte-identical JSON content; stderr noise only)"
    elif content_match_sorted:
        verdict = "ordering-only (same diagnostic set, different order)"
    elif norm_stdout_match:
        verdict = "noise-only (timestamps/ANSI differ; content same after strip)"
    else:
        verdict = "content-flaky"

    # Build a small diff block (line-level unified diff on normalized stdout).
    import difflib
    diff_lines = list(difflib.unified_diff(
        norm1_stdout.splitlines(), norm2_stdout.splitlines(),
        fromfile="run1_stdout_normalized", tofile="run2_stdout_normalized",
        lineterm="", n=2,
    ))
    diff_stderr_lines = list(difflib.unified_diff(
        norm1_stderr.splitlines(), norm2_stderr.splitlines(),
        fromfile="run1_stderr_normalized", tofile="run2_stderr_normalized",
        lineterm="", n=2,
    ))
    raw_summary = (
        f"raw_stdout_match={raw_stdout_match} raw_stderr_match={raw_stderr_match}\n"
        f"normalized_stdout_match={norm_stdout_match} normalized_stderr_match={norm_stderr_match}\n"
        f"json_content_match={content_match} json_content_match_sorted={content_match_sorted}\n"
        f"verdict={verdict}\n"
        "\n=== Normalized stdout diff ===\n"
        + ("\n".join(diff_lines) or "<no differences>")
        + "\n\n=== Normalized stderr diff ===\n"
        + ("\n".join(diff_stderr_lines) or "<no differences>")
    )
    return raw_summary, verdict


def repo_size_mb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        if p.is_file() and not p.is_symlink():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total / (1024 * 1024)


def process_task(task: dict, args) -> dict:
    repo = task["repo"]
    instance_id = task["instance_id"]
    base_commit = task["base_commit"]
    out_dir = Path(args.output) / instance_id
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not args.force:
        print(f"[skip] {instance_id} already done")
        return json.loads(meta_path.read_text())

    repo_dir = Path(args.clone_root) / instance_id
    print(f"[clone] {instance_id} {repo}@{base_commit[:12]} -> {repo_dir}")
    try:
        clone_at_commit(repo, base_commit, repo_dir)
    except subprocess.CalledProcessError as e:
        meta = {
            "instance_id": instance_id, "repo": repo, "base_commit": base_commit,
            "error": "clone_failed",
            "stderr": (e.stderr or "")[:2000],
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        return meta

    size_mb = repo_size_mb(repo_dir)
    print(f"[size] {instance_id}: {size_mb:.1f} MB")

    run1_path = out_dir / "run1.txt"
    run2_path = out_dir / "run2.txt"

    print(f"[run1] {instance_id}")
    wall1, count1, stdout1, stderr1 = run_pyrefly(
        args.pyrefly, repo_dir, run1_path, args.timeout)
    print(f"[run1] {instance_id} wall={wall1:.2f}s diags={count1}")

    print(f"[run2] {instance_id}")
    wall2, count2, stdout2, stderr2 = run_pyrefly(
        args.pyrefly, repo_dir, run2_path, args.timeout)
    print(f"[run2] {instance_id} wall={wall2:.2f}s diags={count2}")

    diff_text, verdict = diff_runs(stdout1, stdout2, stderr1, stderr2)
    (out_dir / "diff.txt").write_text(diff_text)

    meta = {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": base_commit,
        "repo_size_mb": round(size_mb, 1),
        "wall_clock_run1_sec": round(wall1, 3),
        "wall_clock_run2_sec": round(wall2, 3),
        "diagnostic_count_run1": count1,
        "diagnostic_count_run2": count2,
        "determinism_verdict": verdict,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def write_version_file(args) -> None:
    out = Path(args.output) / "version.txt"
    proc = subprocess.run([args.pyrefly, "--version"],
                          capture_output=True, text=True, check=True)
    out.write_text(
        f"pyrefly_version: {proc.stdout.strip()}\n"
        f"install_method: {args.install_method}\n"
        f"git_sha: {args.git_sha}\n"
        f"binary_path: {args.pyrefly}\n"
    )


def write_summary(args, results: list[dict]) -> None:
    lines = [
        "# Pyrefly determinism pilot — summary",
        "",
        "| instance_id | repo size (MB) | run1 (s) | run2 (s) | diags r1 | diags r2 | verdict |",
        "| --- | ---:| ---:| ---:| ---:| ---:| --- |",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['instance_id']} | — | — | — | — | — | ERROR: {r['error']} |")
            continue
        lines.append(
            f"| {r['instance_id']} | {r.get('repo_size_mb','?')} | "
            f"{r['wall_clock_run1_sec']} | {r['wall_clock_run2_sec']} | "
            f"{r['diagnostic_count_run1']} | {r['diagnostic_count_run2']} | "
            f"{r['determinism_verdict']} |"
        )
    (Path(args.output) / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True,
                   help="JSON file with list of {repo, instance_id, base_commit}")
    p.add_argument("--pyrefly", required=True, help="Path to pyrefly binary")
    p.add_argument("--output", required=True, help="Output directory for artifacts")
    p.add_argument("--clone-root", required=True, help="Directory to clone repos into")
    p.add_argument("--timeout", type=int, default=600,
                   help="Per-pyrefly-invocation timeout in seconds (default 600)")
    p.add_argument("--install-method", default="pip",
                   help="For version.txt: how pyrefly was installed")
    p.add_argument("--git-sha", default="",
                   help="For version.txt: pyrefly source commit SHA if known")
    p.add_argument("--force", action="store_true",
                   help="Re-process tasks even if meta.json already exists")
    args = p.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    write_version_file(args)

    tasks = json.loads(Path(args.tasks).read_text())
    results = []
    for t in tasks:
        try:
            r = process_task(t, args)
        except Exception as e:
            r = {"instance_id": t["instance_id"], "repo": t["repo"],
                 "base_commit": t["base_commit"], "error": f"exception: {e!r}"}
            (Path(args.output) / t["instance_id"]).mkdir(parents=True, exist_ok=True)
            (Path(args.output) / t["instance_id"] / "meta.json").write_text(
                json.dumps(r, indent=2))
        results.append(r)
        print(f"[done] {r.get('instance_id')}: {r.get('determinism_verdict', r.get('error'))}")

    write_summary(args, results)
    print(f"\nWrote summary to {Path(args.output) / 'summary.md'}")


if __name__ == "__main__":
    main()
