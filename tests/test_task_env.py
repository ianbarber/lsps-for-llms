"""Fast unit tests for TaskEnv pure logic — no clone/venv/install.

Covers the load-bearing accounting (rework-ratio inputs) and the pytest-summary
parser the SWE-bench `resolved` verdict depends on. The heavyweight clone+install
end-to-end validation lives in runs/task_env/validation.md.
"""
import subprocess
from pathlib import Path

import pytest

from harness.task_env import TaskEnv, _parse_pytest_counts


def _fake_repo(tmp_path: Path) -> Path:
    d = tmp_path / "repo"
    d.mkdir()
    subprocess.run(["git", "init", "-q", str(d)], check=True)
    (d / "mod.py").write_text("def foo():\n    return 1\n\ndef bar():\n    return 2\n")
    subprocess.run(["git", "-C", str(d), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(d), "-c", "user.email=a@b", "-c",
                    "user.name=t", "commit", "-q", "-m", "base"], check=True)
    return d


def _env(repo: Path) -> TaskEnv:
    inst = {"instance_id": "fake-1", "repo": "x/y", "base_commit": "HEAD",
            "FAIL_TO_PASS": [], "PASS_TO_PASS": [], "test_patch": ""}
    env = TaskEnv(inst)
    env.repo_dir = repo
    env.venv_python = None
    env._reset_done = True
    return env


def test_apply_edit_first_write_is_authoring(tmp_path):
    env = _env(_fake_repo(tmp_path))
    r = env.apply_edit("mod.py", "return 1", "return 42")
    assert r.ok
    m = env.metrics()
    assert m["chars_written"] == len("return 42")
    assert m["chars_deleted_after_first_write"] == 0  # first write != rework
    assert m["rework_ratio"] == 0.0
    assert r.region_start_line == 2


def test_apply_edit_second_write_is_rework(tmp_path):
    env = _env(_fake_repo(tmp_path))
    env.apply_edit("mod.py", "return 1", "return 42")
    env.apply_edit("mod.py", "return 42", "return 99")
    m = env.metrics()
    assert m["chars_deleted_after_first_write"] == len("return 42")
    assert m["region_edit_cycles"] == 1
    assert m["rework_ratio"] == pytest.approx(0.5)


def test_apply_edit_failures(tmp_path):
    env = _env(_fake_repo(tmp_path))
    assert not env.apply_edit("mod.py", "zzz", "X").ok            # not found
    assert not env.apply_edit("mod.py", "return", "X").ok          # non-unique
    assert not env.apply_edit("nope.py", "a", "b").ok              # missing file
    assert env.metrics()["failed_edit_count"] == 3
    assert env.metrics()["chars_written"] == 0


def test_current_patch_only_agent_edit(tmp_path):
    env = _env(_fake_repo(tmp_path))
    env.apply_edit("mod.py", "return 1", "return 99")
    patch = env.current_patch()
    assert "-    return 1" in patch
    assert "+    return 99" in patch


@pytest.mark.parametrize("summary,exp", [
    ("2 failed, 63 passed in 3.41s", (63, 2, 0)),
    ("65 passed in 2.0s", (65, 0, 0)),
    ("1 failed in 0.5s", (0, 1, 0)),
    ("3 errors in 1.0s", (0, 0, 3)),
    ("1 passed, 4 warnings in 0.06s", (1, 0, 0)),
])
def test_pytest_parser(summary, exp):
    c = _parse_pytest_counts(summary, n_requested=99)
    assert (c["passed"], c["failed"], c["errored"]) == exp


def test_resolve_bare_test_ids(tmp_path):
    env = _env(_fake_repo(tmp_path))
    env._test_files = ["sympy/sets/tests/test_contains.py"]
    out = env._resolve_test_ids(["test_as_set"])
    assert out == ["sympy/sets/tests/test_contains.py::test_as_set"]
    # already-qualified ids pass through
    assert env._resolve_test_ids(["a/b.py::test_x"]) == ["a/b.py::test_x"]
