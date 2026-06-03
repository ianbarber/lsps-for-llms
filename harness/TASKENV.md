# TaskEnv — agent-facing coding environment (Streams v0.5, option D)

`harness/task_env.py` is the apparatus every condition (A/B/C/D, §0.5) runs on.
One `TaskEnv` wraps one SWE-bench / SWE-Gym instance and exposes a small,
**synchronous** action API the coding agent calls. The agent loop owns timing and
the sync-vs-async delivery logic; `TaskEnv` owns clone/install/test/diagnostics/
edit-accounting. **No model, no GPU here.**

It reuses:
- `harness/swegym_loader` for clone / venv / `pip install -e .` / pytest;
- `lsp/pyrefly_client.PyreflyDaemon` for diagnostics (one daemon per task);
- `lsp/payload.normalize_diagnostics` — the single G4-audited normalization
  chokepoint (canonical `(severity, line, code, message)`, top-K by
  recency-of-edited-region).

## Lifecycle

```python
from harness.task_env import TaskEnv, load_instance

inst = load_instance("sympy__sympy-23950", source="verified")  # or source="swegym"
env  = TaskEnv(inst)                       # holds repo/base_commit/F2P/P2P/test_patch/gold(ref)
state = env.reset()                        # clone @ base + per-task venv + install + apply test_patch

env.list_files("sympy/sets")               # repo-relative file list (default *.py)
src = env.read_file("sympy/sets/contains.py")
r = env.apply_edit("sympy/sets/contains.py", search, replace)   # the agent's edit action
diags = env.pyrefly_diagnostics("sympy/sets/contains.py",
                                edited_region=(r.region_start_line, r.region_end_line))
result = env.run_tests()                   # {f2p_pass, p2p_pass, resolved, ...}
patch  = env.current_patch()               # unified diff vs base (the agent's solution)
m = env.metrics()                          # rework-ratio inputs
env.close()
```

## Methods

### `reset(install=True) -> dict`
Clone the repo at `base_commit` (via `swegym_loader.shallow_clone_at_commit`),
create a per-task uv venv at the pinned Python (`ENV_OVERRIDES`), `pip install -e
.[extras]` with any pre-pins, then **apply the test_patch and commit it**. The
commit is deliberate: it advances `HEAD` to base+test_patch so `current_patch()`
(`git diff HEAD`) returns **only the agent's edits**, never the test scaffolding.
Returns initial state: `instance_id, repo, base_commit, problem_statement,
fail_to_pass`/`pass_to_pass` (counts), `test_files`, `install_ok`,
`python_version`, `clone`.

### `read_file(path)` / `list_files(subdir="", pattern="*.py", include_tests=True)`
Repo file access. Paths are repo-relative; a guard rejects paths escaping the
repo root.

### `apply_edit(path, search, replace) -> EditResult`
Unique-match search-replace edit (fails if `search` is absent or non-unique —
forces the agent to disambiguate, like Claude Code's Edit). Returns
`EditResult(ok, path, reason, new_region, region_start_line, region_end_line)`
where `new_region` is the post-edit file slice ±3 lines (what the agent sees land)
and the 1-indexed line span feeds pyrefly ranking.
**Rework accounting** (the primary-endpoint input) is recorded here:
- every `replace` char → `chars_written`;
- if the file was **already written this trajectory**, the `search` chars removed
  → `chars_deleted_after_first_write` (genuine rework; first authoring of a file
  is not rework) and a region edit-cycle is counted;
- failed edits → `failed_edit_count`.

### `pyrefly_diagnostics(path, edited_region=None, top_k=10) -> list[dict]`
Runs the per-task pyrefly daemon on the **current on-disk** file state and returns
normalized `{severity, line, code, message}` records (1-indexed line), ranked
top-K nearest the `edited_region` via `lsp.payload`. Lazily starts one
`PyreflyDaemon` per task and writes a `pyrefly.toml` that pins the venv's
site-packages (see Gotchas). Daemon round-trip is G5-fast (6–21 ms p95).

### `run_tests(max_f2p=None, max_p2p=None, timeout=1200) -> dict`
Runs FAIL_TO_PASS and PASS_TO_PASS in the per-task venv (pytest, no `-x`, exact
counts). Returns `f2p_pass/f2p_total/p2p_pass/p2p_total`, `f2p_failed/p2p_failed`,
summaries, and **`resolved`** = *all F2P pass AND all P2P pass* — the SWE-bench
success criterion (load-bearing). `max_*` caps are smoke-check only; a real
`resolved` verdict needs the full sets. Bare-name test IDs (sympy/django-style)
are auto-qualified to `<test_patch file>::<name>` when the test_patch added a
single file (`_resolve_test_ids`); free-text django IDs are left unchanged and
will report as errors (use the django runner for those).

### `current_patch() -> str`
`git diff HEAD` of the workdir — the candidate patch SWE-bench would score
(test_patch excluded by construction; see `reset`).

### `metrics() -> dict`
`{chars_written, chars_deleted_after_first_write, rework_ratio, edit_count,
failed_edit_count, region_edit_cycles}`.
**rework_ratio = chars_deleted_after_first_write / chars_written** — the
pre-registered primary endpoint (§8).

### `close(remove_workdir=False)`
Stops the pyrefly daemon; optionally deletes the clone + venv.

## Per-task env pinning

`ENV_OVERRIDES` (in `task_env.py`) holds repo-level defaults and per-instance
overrides: pinned Python (`uv venv --python`), editable-install `extras`, and
`pre_pins` installed before the editable install. Repo default is overridden by an
instance-level entry. Examples: `dask` → py3.12 + `numpy<1.27`; `hydra`/`pydantic`
→ py3.10. `uv` already has 3.10/3.11/3.12 cached locally (no network for the
interpreter).

## Gotchas (caller contract)

1. **Do not `.resolve()` `env.venv_python`.** A uv venv's `bin/python` is a bare
   symlink to the base interpreter; the **symlink** path carries venv context
   (site-packages, pytest), the realpath does not (`No module named pytest`).
   `create_venv` returns the unresolved path and `DEFAULT_WORKROOT` is absolute,
   so the built-in path is correct — just don't realpath it when reusing an env
   object across processes.
2. **pyrefly needs `site-package-path`, not just `python-interpreter`** — for the
   same symlink reason. `_write_pyrefly_config` sets both; without the explicit
   site-packages, pyrefly emits hundreds of spurious `missing-import` (the §7.1
   noise failure).
3. **Old C-extension repos** (astropy<2, sklearn<0.23, old matplotlib) may not
   `pip install -e .` on aarch64 + py3.12. Prefer pure-Python eval tasks; swap if
   `install_ok` is False.
4. **Django Verified tasks** use free-text test IDs for django's own runner and
   are not pytest-runnable as-is — excluded from the eval pool.
