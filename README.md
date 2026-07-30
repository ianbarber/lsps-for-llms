# When Do Language Servers Help Coding Agents?

Experiments on whether language-server tooling makes a coding agent better, measured against a
capable text baseline of grep, ranged reads and a shell.

**The short answer.** Language servers help when two conditions hold together: the server supplies
something the agent cannot cheaply work out for itself, and the agent's behaviour actually changes
to use it. Most of what we tested failed one or the other. Go-to-definition mostly fails the first,
because the model reads the types it needs in the code already in front of it. Compact definition
spans mostly fail the second, because models handed a span open the file anyway. A type check at
the moment of submission satisfies both, and was the clearest result here.

Installing the tool is not the intervention. When trying any of this, measure whether the service
changes what the agent does, not how often it gets called.

[**REPORT.md**](REPORT.md) is the technical report: what we asked, how we tested it, what happened,
and what follows for someone using a coding agent. Everything below is about the repository.

## Where the evidence lives

| Path | What it holds |
|---|---|
| [`REPORT.md`](REPORT.md) | The report. The single source for findings and their scope. |
| [`evidence/claim_ledger.md`](evidence/claim_ledger.md) | Every material claim mapped to its artifacts and evidence status, including contradicted, superseded and excluded results. |
| [`evidence/protocols.md`](evidence/protocols.md) | Experiment protocols, stopping gates and execution status. |
| [`evidence/manifest.json`](evidence/manifest.json) | Hashes, model metadata, integration modes and provenance warnings. |

The ledger is the place to look before trusting a number. It records what each claim is actually
rated as supporting, and it keeps rows for results that turned out to be wrong.

## Reproduce the analysis

Python 3.10+:

```bash
python3 -m pip install -e '.[dev,analysis]'
python3 scripts/analysis/reproduce_all.py
```

This verifies the manifest, reruns the retained analyzers, recomputes task-level effects and reruns
the navigation mechanical checks. It works from committed artifacts and makes no model or API
calls. Pyrefly is discovered through `STREAMS_PYREFLY`, `PYREFLY_BIN`, `PATH`, `.venv/bin` or
`.venv-streams/bin`.

Model execution is separate and needs a GPU or API credentials. Read
[`evidence/protocols.md`](evidence/protocols.md) first: several drivers refuse to overwrite frozen
artifacts, and one of them spends a pre-registered holdout that cannot be recovered once used.

| Script | Runs |
|---|---|
| `scripts/run_navigation_pilot.sh` | Typed/erased navigation pilot |
| `scripts/run_substitution_train.sh` | Reread harvest, LoRA training and held-out retest |
| `scripts/run_navigation_readreq.sh` | Read-required boundary grid |
| `scripts/run_checker_gate_v3.sh` | Four-arm checker delivery grid |
| `scripts/run_checker_paired.sh`, `run_checker_case_series.sh`, `run_checker_hidden.sh`, `run_checker_gate_v2.sh` | Earlier checker experiments, retained |
| `scripts/realbench/local_dispatch.py` | Dispatch ladder and the hidden-type grid |

## Repository map

| Path | Purpose |
|---|---|
| `evidence/` | Claim ledger, protocols, manifest, hashes and provenance |
| `paper/` | LaTeX build of the report (`make` needs pandoc and tectonic) |
| `scripts/analysis/` | Reproducers and statistical analysis |
| `scripts/experiments/` | Retrieval, navigation, substitution and checker harnesses |
| `scripts/realbench/` | Real-repository candidate scanning and dispatch experiments |
| `scaffold/` | Agent loop, tools and workspace environments |
| `runs/agent/`, `runs/pilot/` | Archived raw model results |
| `runs/protocol/` | Mechanical validation and frozen selection artifacts |
| `runs/confirmation/`, `runs/readreq/` | Pre-registered confirmation and read-required boundary results |
| `docs/real_repo_progress.md` | Chronological research log. Historical, not a claim source. |
