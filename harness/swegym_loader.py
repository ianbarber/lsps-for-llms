"""SWE-Gym task loader, repo cloner, and test runner.

For G1 baseline-green validation: load a task from runs/g1_prep/swegym_tasks.json,
shallow-clone the target repo at base_commit, run its test suite, and check that
the PASS_TO_PASS tests are green (baseline-green) and FAIL_TO_PASS tests fail
(SWE-bench-style test_PASS criterion measurable).

This is plumbing only — no model is involved. Used in Wave 1 to confirm the harness
is wired before Wave 2 fires the actual G1 run.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from datasets import load_dataset


REPO_URL_TEMPLATE = "https://github.com/{repo}.git"


def load_g1_tasks(path: Path) -> list[dict]:
    return json.loads(path.read_text())["tasks"]


def load_full_record(instance_id: str) -> dict:
    """Re-fetch the full SWE-Gym row to recover PASS_TO_PASS / FAIL_TO_PASS / test_patch."""
    ds = load_dataset("SWE-Gym/SWE-Gym", split="train")
    for ex in ds:
        if ex["instance_id"] == instance_id:
            return dict(ex)
    raise KeyError(instance_id)


def shallow_clone_at_commit(repo: str, base_commit: str, dest: Path) -> dict:
    """Shallow-clone {repo} at {base_commit} into {dest}; returns timing/size info."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = REPO_URL_TEMPLATE.format(repo=repo)
    t0 = time.time()
    # Strategy: init + fetch the single commit (cheap when supported).
    subprocess.run(["git", "init", "-q", str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "remote", "add", "origin", url], check=True)
    try:
        subprocess.run(
            ["git", "-C", str(dest), "fetch", "--depth", "1", "origin", base_commit],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        # Some servers reject by-SHA fetch when uploadpack.allowReachableSHA1InWant=false.
        # Fall back to full unshallow then checkout.
        subprocess.run(["git", "-C", str(dest), "fetch", "origin"], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "-q", base_commit], check=True)
    elapsed = time.time() - t0
    du_out = subprocess.run(["du", "-sm", str(dest)], capture_output=True, text=True)
    size_mb = int(du_out.stdout.split()[0]) if du_out.stdout else -1
    return {"elapsed_s": round(elapsed, 1), "size_mb": size_mb, "url": url}


def apply_test_patch(repo_dir: Path, test_patch: str) -> None:
    """Apply the test_patch so we have the FAIL_TO_PASS tests in the tree."""
    patch_file = repo_dir / ".g1_test_patch.diff"
    patch_file.write_text(test_patch)
    subprocess.run(["git", "-C", str(repo_dir), "apply", "--allow-empty", str(patch_file)], check=True)


# ---------------------------------------------------------------------------
# Per-task venv creation with Python-version pinning (uv-backed).
#
# Promoted to a reusable helper so harness/task_env.py shares one code path for
# clone + venv + install. SWE-bench Verified provisions a contemporaneous Python
# per task; several older SWE-Gym commits (hydra-1456, pydantic-4882) reject
# Python 3.12 (dataclass mutable-default; pydantic-core skew). `uv` already has
# 3.10/3.11/3.12 cached on this box (no network for the interpreter itself).
# ---------------------------------------------------------------------------


def create_venv(venv_dir: Path, python_version: str = "3.12") -> str:
    """Create a per-task venv with the requested Python version via `uv venv`.

    Returns the absolute path to the venv's python interpreter. `uv` resolves the
    interpreter from its managed pool (3.10.19 / 3.11.14 / 3.12 are pre-cached).
    """
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["uv", "venv", "--python", python_version, str(venv_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    py = venv_dir / "bin" / "python"
    if not py.exists():
        raise RuntimeError(f"venv python not found at {py}")
    return str(py)


def pip_install(venv_python: str, repo_dir: Path, *, extras: str = "",
                pre_pins: list[str] | None = None, timeout: float = 1800.0) -> dict:
    """Editable-install the cloned repo into its venv (`pip install -e .[extras]`).

    `pre_pins` are packages installed *before* the editable install (e.g.
    `numpy<1.27` for dask, or a contemporaneous `pydantic-core` pin). Uses
    `uv pip` for speed, falling back to the venv's own pip on failure.
    `--no-build-isolation` is avoided; we let uv resolve build deps.
    Returns {ok, elapsed_s, log_tail}.
    """
    t0 = time.time()
    logs: list[str] = []

    def _run(cmd: list[str]) -> subprocess.CompletedProcess:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           cwd=str(repo_dir))
        logs.append(f"$ {' '.join(cmd)}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
        return r

    env_py = venv_python
    if pre_pins:
        r = _run(["uv", "pip", "install", "--python", env_py, *pre_pins])
        if r.returncode != 0:
            return {"ok": False, "elapsed_s": round(time.time() - t0, 1),
                    "log_tail": "\n".join(logs)[-4000:], "stage": "pre_pins"}

    target = "." + (f"[{extras}]" if extras else "")
    r = _run(["uv", "pip", "install", "--python", env_py, "-e", target])
    if r.returncode != 0 and extras:
        # Retry without extras (some extras don't resolve on aarch64/old commits).
        r = _run(["uv", "pip", "install", "--python", env_py, "-e", "."])
    ok = r.returncode == 0
    # Ensure pytest is present in the venv.
    if ok:
        _run(["uv", "pip", "install", "--python", env_py, "pytest"])
    return {"ok": ok, "elapsed_s": round(time.time() - t0, 1),
            "log_tail": "\n".join(logs)[-4000:], "stage": "editable_install"}


def unified_diff_vs_base(repo_dir: Path) -> str:
    """`git diff` of the workdir vs the base commit, excluding harness artifacts.

    The agent's solution patch. Excludes the `.g1_test_patch.diff` sidecar and the
    test_patch hunks (those are environment, not the agent's edit) by diffing only
    tracked source — callers should reset the test_patch into a separate commit if
    they want it excluded; here we return the raw working-tree diff against HEAD.
    """
    r = subprocess.run(
        ["git", "-C", str(repo_dir), "diff", "HEAD",
         "--", ".", ":(exclude).g1_test_patch.diff"],
        capture_output=True, text=True,
    )
    return r.stdout


def run_pytest(repo_dir: Path, test_ids: list[str], *, timeout: float = 600.0, venv_python: str | None = None) -> dict:
    """Run pytest on the given test IDs (SWE-bench format: path::name).

    Returns counts of passed/failed/errored.
    """
    py = venv_python or sys.executable
    cmd = [py, "-m", "pytest", "-x", "--no-header", "-q", "--tb=short"] + test_ids
    t0 = time.time()
    try:
        r = subprocess.run(
            cmd,
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = r.stdout
        stderr = r.stderr
        rc = r.returncode
    except subprocess.TimeoutExpired:
        return {"elapsed_s": timeout, "rc": -1, "stdout": "", "stderr": "TIMEOUT", "summary": "TIMEOUT"}
    elapsed = time.time() - t0
    # Parse a summary line like "2 passed, 1 failed in 3.4s".
    summary = ""
    for line in reversed(stdout.splitlines()):
        if " passed" in line or " failed" in line or " error" in line:
            summary = line.strip()
            break
    return {
        "elapsed_s": round(elapsed, 1),
        "rc": rc,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-2000:],
        "summary": summary,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks-json", default="/home/ianbarber/Projects/Streams/runs/g1_prep/swegym_tasks.json")
    p.add_argument("--instance-id", required=True, help="One of the 10 task instance_ids to validate")
    p.add_argument("--workdir", default="/home/ianbarber/Projects/Streams/runs/g1_prep/swegym_clones")
    p.add_argument("--venv-python", default=None, help="Python for the per-task venv; if absent, uses sys.executable")
    p.add_argument("--max-tests", type=int, default=10,
                   help="Cap number of PASS_TO_PASS tests to run (smoke check)")
    args = p.parse_args()

    tasks = load_g1_tasks(Path(args.tasks_json))
    chosen = next((t for t in tasks if t["instance_id"] == args.instance_id), None)
    if chosen is None:
        raise SystemExit(f"Instance {args.instance_id} not in {args.tasks_json}")

    full = load_full_record(args.instance_id)
    print(f"[validation] instance={args.instance_id} repo={chosen['repo']} commit={chosen['base_commit'][:12]}")
    print(f"             P2P={len(full['PASS_TO_PASS'])} F2P={len(full['FAIL_TO_PASS'])}")

    clone_dir = Path(args.workdir) / args.instance_id
    info = shallow_clone_at_commit(chosen["repo"], chosen["base_commit"], clone_dir)
    print(f"             clone: {info['size_mb']} MB in {info['elapsed_s']}s -> {clone_dir}")

    # Apply test_patch (so FAIL_TO_PASS tests exist in the tree).
    apply_test_patch(clone_dir, full["test_patch"])
    print("             applied test_patch")

    # Baseline-green check on a sample of PASS_TO_PASS tests.
    p2p = full["PASS_TO_PASS"][: args.max_tests]
    if not p2p:
        print("             no PASS_TO_PASS tests; skipping baseline-green check")
        p2p_result = {"summary": "EMPTY", "rc": 0}
    else:
        print(f"             running {len(p2p)} PASS_TO_PASS tests (capped at --max-tests)")
        p2p_result = run_pytest(clone_dir, p2p, venv_python=args.venv_python)
        print(f"             P2P pytest: {p2p_result['summary']} (rc={p2p_result['rc']}, t={p2p_result['elapsed_s']}s)")

    # Also confirm FAIL_TO_PASS tests fail at the base commit (test_PASS criterion is measurable).
    f2p = full["FAIL_TO_PASS"][: args.max_tests]
    if f2p:
        print(f"             running {len(f2p)} FAIL_TO_PASS tests (expected to fail)")
        f2p_result = run_pytest(clone_dir, f2p, venv_python=args.venv_python)
        print(f"             F2P pytest: {f2p_result['summary']} (rc={f2p_result['rc']}, t={f2p_result['elapsed_s']}s)")
    else:
        f2p_result = {"summary": "EMPTY", "rc": 0}

    report = {
        "instance_id": args.instance_id,
        "repo": chosen["repo"],
        "base_commit": chosen["base_commit"],
        "clone": info,
        "pass_to_pass": {
            "n_run": len(p2p),
            "n_total": len(full["PASS_TO_PASS"]),
            "result": p2p_result,
        },
        "fail_to_pass": {
            "n_run": len(f2p),
            "n_total": len(full["FAIL_TO_PASS"]),
            "result": f2p_result,
        },
    }
    out_path = Path(args.workdir) / f"{args.instance_id}_validation.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"             wrote {out_path}")


if __name__ == "__main__":
    main()
