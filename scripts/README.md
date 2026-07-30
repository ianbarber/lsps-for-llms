# scripts/

What lives where. Findings are in [`REPORT.md`](../REPORT.md); which artifact backs which claim is in
[`evidence/claim_ledger.md`](../evidence/claim_ledger.md); the commands to run things are in the
[top-level README](../README.md).

| Path | Purpose |
|---|---|
| `analysis/` | Analyzers and reproducers. `reproduce_all.py` is the fast no-model entry point |
| `experiments/` | Task generators and experiment drivers for navigation, substitution, retrieval and the checker |
| `realbench/` | Real-repository scanning and the dispatch ladder |
| `run_*.sh` | Local-model drivers. One experiment each, GPU or API required |
| `synth_tasks_*.py` | Task-suite generators |
| `build_manifest.py` | Records hashes, configs, seeds and provenance into `evidence/manifest.json` |
| `validate_pyrefly_lsp.py` | Checks the AST resolver against a live pyrefly daemon. Kills every pyrefly process on the machine when it finishes |

Conventions worth knowing before you add anything:

- Every `run_*.sh` sources `common.sh`, derives the repository root from its own path, and honours
  `PYTHON`. Cache and offline variables are left to the caller.
- Drivers refuse to overwrite an existing output file. Delete it deliberately or pick a new path.
- Pyrefly is discovered by `scaffold/tooling.py`. The exception is `realbench/pyrefly_nav.py`, which
  stays self-contained because it is copied alone into SWE-bench containers.
- `build_manifest.py` never rewrites historical result JSON; it only records what is there. Adding a
  results directory means adding it to the scan list, or its files are silently unmanifested.
