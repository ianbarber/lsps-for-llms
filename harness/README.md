# harness/ — real-repository pipeline (paper Appendix A only)

Native (no-container) SWE-rebench task environments on aarch64: shallow clone at
base commit, per-task uv venv, editable install, test-patch application, real
pytest F2P/P2P scoring, and Pyrefly diagnostics over the repo (via `lsp/`).

- `task_env.py` — the environment (apply-or-reject unique-match edits, line edits,
  diagnostics, metrics). `swegym_loader.py` — clone/install/patch plumbing.
- Used by `scripts/rebench_*.py` and `scripts/agent_swe.py`.

This pipeline backs Appendix A (the 7B cannot solve real oracle-localized bugs;
the checker is blind to their logic errors). It is retained, working, for future
work with stronger base agents. `lsp/pyrefly_client.py` and `lsp/payload.py` are
its live dependencies.
