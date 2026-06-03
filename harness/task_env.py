#!/usr/bin/env python3
"""TaskEnv — the agent-facing coding environment for Streams v0.5 (option D).

One `TaskEnv` wraps one SWE-bench / SWE-Gym instance and gives the coding agent a
small, synchronous action API: reset (clone+venv+install+apply test_patch), read
/ list files, apply search-replace edits, ask pyrefly for diagnostics, run the
task's tests, and read the current solution patch. It also accumulates the
in-flight accounting (chars written / deleted-after-first-write, edit cycles)
that feeds the primary endpoint, rework-ratio (§8).

Design constraints (per the build brief):
- Synchronous and simple — the agent loop owns timing and conditions A/B/C/D.
- Reuses `harness/swegym_loader` for clone / venv / install / pytest, and
  `lsp/pyrefly_client` + `lsp/payload` for diagnostics (the single normalization
  chokepoint G4 audits).
- No GPU, no model here. This is pure apparatus.

The two load-bearing pieces:
  * `.run_tests()` implements the SWE-bench *resolved* criterion exactly:
    all FAIL_TO_PASS pass AND all PASS_TO_PASS still pass.
  * `.apply_edit()` records rework accounting so the agent loop can compute
    rework-ratio = chars_deleted_after_first_write / total_chars_written.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Allow running both as a module and as a script.
_HARNESS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _HARNESS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from harness import swegym_loader as loader  # noqa: E402
from lsp.pyrefly_client import PyreflyDaemon  # noqa: E402
from lsp.payload import EditedRegion, normalize_diagnostics  # noqa: E402


DEFAULT_WORKROOT = _PROJECT_ROOT / "runs" / "task_env" / "workdirs"


# ---------------------------------------------------------------------------
# Per-task environment overrides.
#
# Some older SWE-Gym commits need a pinned Python and/or pinned build deps. These
# are the SWE-bench-Verified-norm provisioning hints, keyed by instance_id and by
# repo (repo-level default, instance override wins). Extend as tasks are added.
# ---------------------------------------------------------------------------

ENV_OVERRIDES: dict[str, dict[str, Any]] = {
    # repo-level defaults
    "repo:dask/dask": {"python": "3.12", "pre_pins": ["numpy<1.27"], "extras": "array,test"},
    "repo:facebookresearch/hydra": {"python": "3.10", "extras": ""},
    "repo:pydantic/pydantic": {"python": "3.10", "extras": ""},
    "repo:python/mypy": {"python": "3.12", "extras": ""},
    "repo:modin-project/modin": {"python": "3.12", "extras": ""},
    "repo:pandas-dev/pandas": {"python": "3.12", "extras": "test"},
    "repo:bokeh/bokeh": {"python": "3.10", "extras": ""},
    "repo:conan-io/conan": {"python": "3.10", "extras": ""},
    # instance-level overrides (win over repo defaults)
    # "facebookresearch__hydra-1456": {"python": "3.10"},
}


def _overrides_for(instance: dict) -> dict[str, Any]:
    repo = instance.get("repo", "")
    merged: dict[str, Any] = dict(ENV_OVERRIDES.get(f"repo:{repo}", {}))
    merged.update(ENV_OVERRIDES.get(instance.get("instance_id", ""), {}))
    return merged


# ---------------------------------------------------------------------------
# In-flight metrics accounting (rework-ratio inputs).
# ---------------------------------------------------------------------------


@dataclass
class RegionStat:
    """Per-(file, region) edit history, used for edit-error-cycle accounting."""

    first_written: bool = False
    edit_cycles: int = 0  # number of times this region was re-edited after first


@dataclass
class EditMetrics:
    """Running totals that feed rework-ratio and the secondary edit metrics (§8).

    rework-ratio = chars_deleted_after_first_write / total_chars_written.

    "chars_written" counts every character the agent emits into a file via
    apply_edit (the `replace` text). "chars_deleted_after_first_write" counts
    characters removed (the `search` text length) from a file region the agent had
    *already* written into during this trajectory — i.e. genuine rework, not the
    first authoring of the gold/base code. The very first edit to a file region is
    authoring; deleting/replacing it later is rework.
    """

    chars_written: int = 0
    chars_deleted_after_first_write: int = 0
    edit_count: int = 0
    failed_edit_count: int = 0
    # key: f"{path}" -> set of touched line spans is overkill; we track per-file
    # "have we written here yet" plus a coarse cycle count.
    _files_written: set[str] = field(default_factory=set)
    region_cycles: int = 0

    @property
    def rework_ratio(self) -> float:
        if self.chars_written == 0:
            return 0.0
        return self.chars_deleted_after_first_write / self.chars_written

    def as_dict(self) -> dict[str, Any]:
        return {
            "chars_written": self.chars_written,
            "chars_deleted_after_first_write": self.chars_deleted_after_first_write,
            "rework_ratio": round(self.rework_ratio, 4),
            "edit_count": self.edit_count,
            "failed_edit_count": self.failed_edit_count,
            "region_edit_cycles": self.region_cycles,
        }


@dataclass
class EditResult:
    ok: bool
    path: str
    reason: str = ""
    # The new region of the file around the edit (for the agent to see what landed).
    new_region: str = ""
    region_start_line: int = 0  # 1-indexed
    region_end_line: int = 0  # 1-indexed


# ---------------------------------------------------------------------------
# TaskEnv
# ---------------------------------------------------------------------------


class TaskEnv:
    """One coding-agent environment for one SWE-bench/SWE-Gym instance.

    Lifecycle:
        env = TaskEnv(instance)
        state = env.reset()                  # clone+venv+install+apply test_patch
        env.read_file(p) / env.list_files()
        env.apply_edit(p, search, replace)   # the agent's edit action
        env.pyrefly_diagnostics(p)           # normalized (severity,line,code,msg)
        result = env.run_tests()             # {f2p_pass, p2p_pass, resolved}
        patch = env.current_patch()          # unified diff vs base (agent solution)
        env.metrics()                        # rework-ratio inputs
        env.close()

    `instance` is a dict with at least: instance_id, repo, base_commit,
    FAIL_TO_PASS, PASS_TO_PASS, test_patch. problem_statement / patch (gold) are
    optional (gold kept only for reference / difficulty proxies, never applied).
    """

    def __init__(self, instance: dict, *, workroot: Path | str = DEFAULT_WORKROOT,
                 diag_timeout: float = 10.0):
        self.instance = instance
        self.instance_id: str = instance["instance_id"]
        self.repo: str = instance["repo"]
        self.base_commit: str = instance["base_commit"]
        self.problem_statement: str = instance.get("problem_statement", "")
        self.test_patch: str = instance.get("test_patch", "")
        self.gold_patch: str = instance.get("patch", "")  # reference only
        self.fail_to_pass: list[str] = _as_list(instance.get("FAIL_TO_PASS", []))
        self.pass_to_pass: list[str] = _as_list(instance.get("PASS_TO_PASS", []))

        self.workroot = Path(workroot)
        self.repo_dir = self.workroot / self.instance_id
        self.venv_dir = self.workroot / f"{self.instance_id}.venv"
        self.diag_timeout = diag_timeout

        self.venv_python: str | None = None
        self.metrics_state = EditMetrics()
        self._daemon: PyreflyDaemon | None = None
        self._reset_done = False
        self._install_log = ""
        self._test_files: list[str] = []  # files added by the test_patch

    # -- reset / provisioning ------------------------------------------------

    def reset(self, *, install: bool = True) -> dict:
        """Clone at base_commit, build the per-task venv, editable-install, and
        apply the test_patch. Returns the initial state the agent sees.

        Returns dict: {instance_id, repo, base_commit, problem_statement,
        fail_to_pass(count), pass_to_pass(count), test_files, install_ok,
        python_version, clone}.
        """
        ov = _overrides_for(self.instance)
        py_version = ov.get("python", "3.12")

        clone_info = loader.shallow_clone_at_commit(self.repo, self.base_commit, self.repo_dir)

        # Snapshot base as a clean commit so the test_patch + agent edits diff
        # cleanly. The shallow clone already has HEAD at base_commit; we commit
        # any clone artifacts away by creating a marker commit is unnecessary —
        # HEAD *is* base. We will exclude the test_patch from current_patch() by
        # committing it (see below), so HEAD advances to base+test_patch.
        self.venv_python = None
        install_ok = None
        if install:
            self.venv_python = loader.create_venv(self.venv_dir, py_version)
            res = loader.pip_install(
                self.venv_python, self.repo_dir,
                extras=ov.get("extras", ""),
                pre_pins=ov.get("pre_pins"),
            )
            install_ok = res["ok"]
            self._install_log = res["log_tail"]

        # Apply the test_patch (adds the new tests) and COMMIT it, so that
        # current_patch() (diff vs HEAD) returns only the agent's edits, not the
        # test scaffolding.
        test_files: list[str] = []
        if self.test_patch.strip():
            loader.apply_test_patch(self.repo_dir, self.test_patch)
            test_files = _patched_files(self.test_patch)
            self._test_files = test_files
            subprocess.run(["git", "-C", str(self.repo_dir), "add", "-A"], check=True,
                           capture_output=True)
            subprocess.run(
                ["git", "-C", str(self.repo_dir), "-c", "user.email=env@streams",
                 "-c", "user.name=TaskEnv", "commit", "-q", "-m", "apply test_patch"],
                check=True, capture_output=True,
            )

        self._reset_done = True
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "problem_statement": self.problem_statement,
            "fail_to_pass": len(self.fail_to_pass),
            "pass_to_pass": len(self.pass_to_pass),
            "test_files": test_files,
            "install_ok": install_ok,
            "python_version": py_version,
            "clone": clone_info,
        }

    # -- file access ---------------------------------------------------------

    def _abspath(self, path: str) -> Path:
        p = (self.repo_dir / path).resolve()
        if not str(p).startswith(str(self.repo_dir.resolve())):
            raise ValueError(f"path escapes repo: {path}")
        return p

    def read_file(self, path: str) -> str:
        return self._abspath(path).read_text()

    def list_files(self, subdir: str = "", *, pattern: str = "*.py",
                   include_tests: bool = True) -> list[str]:
        """Repo-relative file list. Defaults to Python source; pass pattern='*'
        for everything. Skips .git and the per-task patch sidecar."""
        root = self._abspath(subdir) if subdir else self.repo_dir
        out: list[str] = []
        for p in root.rglob(pattern):
            if ".git/" in str(p) or p.name == ".g1_test_patch.diff":
                continue
            if not p.is_file():
                continue
            rel = str(p.relative_to(self.repo_dir))
            if not include_tests and ("/test" in rel or rel.startswith("test")):
                continue
            out.append(rel)
        return sorted(out)

    # -- edit action (with rework accounting) --------------------------------

    def apply_edit(self, path: str, search: str, replace: str) -> EditResult:
        """Apply a unique search-replace edit to a file in the workdir.

        Returns EditResult(ok, path, reason, new_region, region_start/end_line).
        Records rework accounting: every `replace` char counts as written; if the
        edited file region was *already written this trajectory*, the `search`
        chars removed count as deleted-after-first-write (rework). The first edit
        into a given file is authoring, not rework.
        """
        abspath = self._abspath(path)
        if not abspath.exists():
            self.metrics_state.failed_edit_count += 1
            return EditResult(False, path, reason="file does not exist")
        content = abspath.read_text()

        count = content.count(search)
        if count == 0:
            self.metrics_state.failed_edit_count += 1
            return EditResult(False, path, reason="search string not found")
        if count > 1:
            self.metrics_state.failed_edit_count += 1
            return EditResult(False, path, reason=f"search string not unique ({count} matches)")

        idx = content.index(search)
        new_content = content[:idx] + replace + content[idx + len(search):]
        abspath.write_text(new_content)

        # --- rework accounting ---
        already_written = path in self.metrics_state._files_written
        self.metrics_state.chars_written += len(replace)
        if already_written:
            # Replacing/removing previously-agent-written content is rework.
            self.metrics_state.chars_deleted_after_first_write += len(search)
            self.metrics_state.region_cycles += 1
        self.metrics_state._files_written.add(path)
        self.metrics_state.edit_count += 1

        # Compute the 1-indexed line span of the new region for the agent's view.
        start_line = content[:idx].count("\n") + 1
        end_line = start_line + replace.count("\n")
        new_lines = new_content.splitlines()
        ctx = 3
        lo = max(0, start_line - 1 - ctx)
        hi = min(len(new_lines), end_line + ctx)
        new_region = "\n".join(new_lines[lo:hi])

        return EditResult(True, path, new_region=new_region,
                          region_start_line=start_line, region_end_line=end_line)

    def apply_line_edit(self, path: str, start: int, end: int, new_text: str) -> EditResult:
        """Replace 1-indexed lines [start, end] (inclusive) with new_text. Robust
        alternative to search/replace for weak models on large files: no string
        matching, the model picks a line range off the numbered view. Same rework
        accounting (chars of replaced region count as deleted-after-first-write)."""
        abspath = self._abspath(path)
        if not abspath.exists():
            self.metrics_state.failed_edit_count += 1
            return EditResult(False, path, reason="file does not exist")
        content = abspath.read_text()
        lines = content.splitlines(keepends=True)
        n = len(lines)
        if not (1 <= start <= end <= n):
            self.metrics_state.failed_edit_count += 1
            return EditResult(False, path,
                              reason=f"line range {start}-{end} out of bounds (file has {n} lines)")
        removed = "".join(lines[start - 1:end])
        nt = new_text if new_text.endswith("\n") else new_text + "\n"
        new_content = "".join(lines[:start - 1]) + nt + "".join(lines[end:])
        abspath.write_text(new_content)

        already_written = path in self.metrics_state._files_written
        self.metrics_state.chars_written += len(nt)
        if already_written:
            self.metrics_state.chars_deleted_after_first_write += len(removed)
            self.metrics_state.region_cycles += 1
        self.metrics_state._files_written.add(path)
        self.metrics_state.edit_count += 1

        allnew = new_content.splitlines()
        e2 = start + nt.count("\n") - 1
        ctx = 3
        lo = max(0, start - 1 - ctx)
        hi = min(len(allnew), e2 + ctx)
        return EditResult(True, path, new_region="\n".join(allnew[lo:hi]),
                          region_start_line=start, region_end_line=e2)

    # -- pyrefly diagnostics -------------------------------------------------

    def _ensure_daemon(self) -> PyreflyDaemon:
        if self._daemon is None:
            # Point pyrefly at the per-task venv interpreter so it resolves the
            # editable-installed package (per §7.1 — without this, hundreds of
            # spurious missing-import diagnostics). We write a minimal pyrefly
            # config into the repo root if one isn't present.
            self._write_pyrefly_config()
            self._daemon = PyreflyDaemon(str(self.repo_dir), diag_timeout=self.diag_timeout)
        return self._daemon

    def _write_pyrefly_config(self) -> None:
        """Write a pyrefly.toml that resolves the per-task venv's site-packages.

        NOTE: a uv venv's `bin/python` is a bare symlink to the base interpreter,
        so `python-interpreter` alone makes pyrefly query the *base* site-packages
        and miss the venv's installed deps (hundreds of spurious missing-import
        diagnostics, exactly the §7.1 failure mode). We therefore pass the venv's
        site-packages directory explicitly via `site-package-path` — verified to
        resolve numpy/installed deps on dask-10027.
        """
        cfg = self.repo_dir / "pyrefly.toml"
        if cfg.exists():
            return
        lines = ['project-includes = ["**/*.py"]']
        sp = self._venv_site_packages()
        if self.venv_python:
            lines.append(f'python-interpreter = "{self.venv_python}"')
        if sp:
            lines.append(f'site-package-path = ["{sp}"]')
        cfg.write_text("\n".join(lines) + "\n")

    def _venv_site_packages(self) -> str | None:
        if not self.venv_dir.exists():
            return None
        for lib in sorted(self.venv_dir.glob("lib/python*/site-packages")):
            return str(lib.resolve())
        return None

    def pyrefly_diagnostics(self, path: str, *, edited_region: tuple[int, int] | None = None,
                            top_k: int = 10) -> list[dict]:
        """Run pyrefly on the *current on-disk* state of `path`; return normalized
        diagnostics: list of {severity, line, code, message} (1-indexed line),
        ranked top-K by recency-of-edited-region via lsp.payload.

        `edited_region` is an optional (start_line, end_line) 1-indexed span used
        for ranking (defaults to top-of-file order).
        """
        abspath = self._abspath(path)
        text = abspath.read_text()
        daemon = self._ensure_daemon()
        # open is idempotent for ranking; use change to push current text if open.
        raw = daemon.open(str(abspath), text=text)
        # A second push ensures the daemon reflects the latest disk state even if
        # the document was opened earlier in this trajectory.
        raw = daemon.change(str(abspath), text)
        region = EditedRegion(*edited_region) if edited_region else None
        return normalize_diagnostics(raw, region, top_k=top_k)

    # -- tests (SWE-bench resolved criterion) --------------------------------

    def run_tests(self, *, max_f2p: int | None = None, max_p2p: int | None = None,
                  timeout: float = 1200.0) -> dict:
        """Run FAIL_TO_PASS and PASS_TO_PASS in the per-task venv.

        Returns:
          {
            f2p_pass: int, f2p_total: int,
            p2p_pass: int, p2p_total: int,
            resolved: bool,            # all F2P pass AND all P2P pass
            f2p_summary, p2p_summary,  # pytest summary lines
          }

        `resolved` is the SWE-bench success criterion. Caps (`max_*`) are for fast
        smoke checks only; a real resolved verdict requires the full sets.
        """
        py = self.venv_python or sys.executable
        f2p = self.fail_to_pass[:max_f2p] if max_f2p else self.fail_to_pass
        p2p = self.pass_to_pass[:max_p2p] if max_p2p else self.pass_to_pass

        f2p_res = self._pytest(py, f2p, timeout=timeout) if f2p else _empty_pytest()
        p2p_res = self._pytest(py, p2p, timeout=timeout) if p2p else _empty_pytest()

        f2p_pass = f2p_res["passed"]
        p2p_pass = p2p_res["passed"]
        resolved = (
            bool(f2p) and f2p_pass == len(f2p)
            and p2p_pass == len(p2p)
        )
        return {
            "f2p_pass": f2p_pass,
            "f2p_total": len(f2p),
            "p2p_pass": p2p_pass,
            "p2p_total": len(p2p),
            "resolved": resolved,
            "f2p_summary": f2p_res["summary"],
            "p2p_summary": p2p_res["summary"],
            "f2p_failed": f2p_res["failed"] + f2p_res["errored"],
            "p2p_failed": p2p_res["failed"] + p2p_res["errored"],
        }

    def _resolve_test_ids(self, test_ids: list[str]) -> list[str]:
        """Qualify bare test names (SWE-bench sympy/django-style) to pytest
        `path::name`. SWE-Gym/astropy/sklearn already use `path::name`; some repos
        (notably sympy) record bare function names. When the test_patch added a
        single test file, prefix bare names with it. Free-text django test
        descriptions cannot be qualified and are returned unchanged (they will be
        reported as errors — such tasks should use the django runner, not pytest)."""
        out: list[str] = []
        single = self._test_files[0] if len(self._test_files) == 1 else None
        for tid in test_ids:
            if "::" in tid or "/" in tid or tid.endswith(".py"):
                out.append(tid)
            elif single and tid.replace("_", "").replace(" ", "").isalnum() is False:
                # has spaces / punctuation -> django free-text, leave as-is
                out.append(tid)
            elif single:
                out.append(f"{single}::{tid}")
            else:
                out.append(tid)
        return out

    def _pytest(self, py: str, test_ids: list[str], *, timeout: float) -> dict:
        """Run pytest WITHOUT -x (we need exact pass/fail counts, not first-fail)
        and parse per-outcome counts from the summary line."""
        test_ids = self._resolve_test_ids(test_ids)
        cmd = [py, "-m", "pytest", "--no-header", "-q", "--tb=no", "-p", "no:cacheprovider",
               *test_ids]
        try:
            r = subprocess.run(cmd, cwd=str(self.repo_dir), capture_output=True,
                               text=True, timeout=timeout)
            stdout = r.stdout
        except subprocess.TimeoutExpired:
            return {"passed": 0, "failed": 0, "errored": len(test_ids),
                    "summary": "TIMEOUT"}
        return _parse_pytest_counts(stdout, n_requested=len(test_ids))

    # -- solution patch ------------------------------------------------------

    def current_patch(self) -> str:
        """Unified diff of the workdir vs base+test_patch (the agent's solution).

        Because reset() commits the test_patch, `git diff HEAD` returns only the
        agent's own edits — exactly the candidate patch SWE-bench would score.
        """
        return loader.unified_diff_vs_base(self.repo_dir)

    # -- metrics / cleanup ---------------------------------------------------

    def metrics(self) -> dict:
        return self.metrics_state.as_dict()

    def close(self, *, remove_workdir: bool = False) -> None:
        if self._daemon is not None:
            try:
                self._daemon.close()
            except Exception:
                pass
            self._daemon = None
        if remove_workdir:
            import shutil
            for d in (self.repo_dir, self.venv_dir):
                if d.exists():
                    shutil.rmtree(d, ignore_errors=True)

    def __enter__(self) -> "TaskEnv":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _as_list(v: Any) -> list[str]:
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            return [v]
    return list(v)


def _patched_files(patch: str) -> list[str]:
    files: list[str] = []
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            files.append(line[6:].strip())
    return files


def _empty_pytest() -> dict:
    return {"passed": 0, "failed": 0, "errored": 0, "summary": "EMPTY"}


def _parse_pytest_counts(stdout: str, *, n_requested: int) -> dict:
    """Parse pytest's summary line (e.g. '2 failed, 63 passed in 3.4s')."""
    summary = ""
    for line in reversed(stdout.splitlines()):
        s = line.strip()
        if (" passed" in s or " failed" in s or " error" in s
                or "no tests ran" in s) and (" in " in s or s.endswith("s")):
            summary = s
            break

    import re
    counts = {"passed": 0, "failed": 0, "errored": 0}
    for n, word in re.findall(r"(\d+)\s+(passed|failed|error|errors)", summary):
        if word == "passed":
            counts["passed"] = int(n)
        elif word == "failed":
            counts["failed"] = int(n)
        else:
            counts["errored"] = int(n)
    counts["summary"] = summary or stdout.strip().splitlines()[-1:][0] if stdout.strip() else "NO_OUTPUT"
    if isinstance(counts["summary"], list):
        counts["summary"] = "NO_OUTPUT"
    return counts


# ---------------------------------------------------------------------------
# CLI: load an instance from a candidates json and validate baseline-green.
# ---------------------------------------------------------------------------


def load_instance(instance_id: str, *, source: str = "swegym") -> dict:
    """Fetch a full instance record (with FAIL_TO_PASS/PASS_TO_PASS/test_patch)."""
    from datasets import load_dataset
    if source == "swegym":
        ds = load_dataset("SWE-Gym/SWE-Gym", split="train")
    elif source == "verified":
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    else:
        raise ValueError(source)
    for ex in ds:
        if ex["instance_id"] == instance_id:
            return dict(ex)
    raise KeyError(instance_id)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-id", required=True)
    ap.add_argument("--source", choices=["swegym", "verified"], default="swegym")
    ap.add_argument("--max-f2p", type=int, default=None)
    ap.add_argument("--max-p2p", type=int, default=None)
    ap.add_argument("--edit-file", default=None, help="optional file to test apply_edit on")
    args = ap.parse_args()

    inst = load_instance(args.instance_id, source=args.source)
    env = TaskEnv(inst)
    print(f"[reset] {args.instance_id} ({inst['repo']})")
    state = env.reset()
    print(json.dumps({k: v for k, v in state.items() if k != "problem_statement"}, indent=2))
    print(f"[tests] running F2P={state['fail_to_pass']} P2P={state['pass_to_pass']} "
          f"(caps f2p={args.max_f2p} p2p={args.max_p2p})")
    res = env.run_tests(max_f2p=args.max_f2p, max_p2p=args.max_p2p)
    print(json.dumps(res, indent=2))
    env.close()
