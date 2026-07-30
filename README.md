# When Do Language Servers Help Coding Agents?

Experiments on whether language-server tooling makes a coding agent better, measured against a
capable text baseline of grep, ranged reads and a shell.

**The short answer.** A language server helped where it changed what the agent did. Making one
available changed nothing on its own, and each operation needed its own mechanism before it paid.
Correct, visible type annotations helped regardless of tooling; they are what the agent navigated
on when it had them.
Go-to-definition earned its place only where working out a receiver's type meant leaving the source
already in front of the agent, and then only when the agent was told to use it. A compact
definition span cost less than reading the file only when it replaced the read, which mostly it did
not until the model was fine-tuned. A type checker paid off as a blocking gate at the end of the
turn, and the gain came from refusing the submission rather than from the moment the diagnostic
arrived; a live channel that interrupted the agent and told it to fix each diagnostic on arrival
did worse than saying nothing.

Installing the tool is not the intervention. When trying any of this, measure whether the service
changes what the agent does, not how often it gets called.

**Read the report:** [**REPORT.md**](REPORT.md), or the same thing typeset as a preprint,
[**paper/report.pdf**](paper/report.pdf). What we asked, how we tested it, what happened, and what
follows for someone using a coding agent. Everything below is about the repository.

## Where the evidence lives

| Path | What it holds |
|---|---|
| [`REPORT.md`](REPORT.md) | The report. The single source for findings and their scope. [`paper/report.pdf`](paper/report.pdf) is the same content as a preprint. |
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

This verifies the manifest, reruns the retained analyzers, recomputes task-level effects, rebuilds
and revalidates the navigation, retrieval and checker task splits against a live Pyrefly,
reproduces the delivery-timing statistics and runs the test suite. It works from committed
artifacts and makes no model or API calls. Pyrefly is discovered through `STREAMS_PYREFLY`,
`PYREFLY_BIN`, `PATH`, `.venv/bin` or `.venv-streams/bin`.

Two analyzers take run artifacts as arguments and are outside that path:

```bash
python3 scripts/analysis/analyze_substitution.py \
  --baseline runs/pilot/navigation_v2_reread_qwen36_27b_apparatus.json \
  --trained  runs/pilot/navigation_v2_reread_qwen36_27b_apparatus_trained.json

python3 scripts/analysis/analyze_navigation_readreq.py \
  runs/readreq/navigation_v2_c36_main_untrained.json \
  runs/readreq/navigation_v2_c36_main_trained.json
```

The second one prints no verdict unless every pre-registered validity gate holds.

## Check the definition tool against a live language server

The definition operation in the retrieval experiments is a static AST resolver,
`scaffold/mock_env.py::MultiFileEnv.goto_definition`, not a language server.
`scripts/validate_pyrefly_lsp.py` checks the two agree. For each synthetic task it writes the
workspace to a temp directory with a `pyrefly.toml`, spawns a live `pyrefly lsp` daemon over stdio
JSON-RPC, asks it for `textDocument/definition` at a use site of the task symbol, and compares the
answer with the resolver's. It prints a summary, and writes the artifact only when
`VALIDATE_PYREFLY_OUT` names an output path:

```bash
VALIDATE_PYREFLY_OUT=runs/protocol/ast_resolver_vs_pyrefly_agreement.json \
  python3 scripts/validate_pyrefly_lsp.py
```

It is deliberately not part of `reproduce_all.py`. It spawns one live daemon per task, strictly
sequentially, because concurrent pyrefly daemons have deadlocked in this environment; the script
documents that hazard, gives every JSON-RPC read a timeout so a hang fails loudly, and kills stray
`pyrefly lsp` processes between tasks.

## Run the models

Model execution is separate and needs a GPU or API credentials. Read
[`evidence/protocols.md`](evidence/protocols.md) first: several drivers refuse to overwrite frozen
artifacts, and some are gated shut on purpose. `scripts/run_navigation_confirmation.sh` will not
run at all while its pilot is at ceiling, and the split it reserved was later spent on the
substitution confirmation instead.

| Script | Runs |
|---|---|
| `scripts/run_navigation_pilot.sh` | Typed/erased navigation pilot |
| `scripts/experiments/retrieval_paired.py` | Paired retrieval-cost suite over vendored library source |
| `scripts/experiments/run_navigation_reread.py` | Pushed and self-elected span arms, and the pre-registered substitution confirmation |
| `scripts/run_substitution_train.sh` | Reread harvest, LoRA training and held-out retest |
| `scripts/run_navigation_readreq.sh` | Read-required boundary grid: hash gate, instance validation, an untrained pilot floor, then the three-arm grid untrained and trained |
| `scripts/run_checker_gate_v3.sh` | Four-arm checker delivery grid |
| `scripts/run_checker_paired.sh`, `run_checker_case_series.sh`, `run_checker_hidden.sh`, `run_checker_gate_v2.sh` | Earlier checker experiments, retained |
| `scripts/realbench/local_dispatch.py` | Dispatch ladder and the hidden-source dispatch grid |

The six delivery-timing arms have no driver left in the tree; the one that produced them was
removed in the publication cleanup, and `scripts/synth_mf.py` now offers condition A only.
`scripts/analysis/stats_delivery.py` works from the committed results and records the original
invocations, the per-arm provenance and the self-checks it enforces.

## Repository map

| Path | Purpose |
|---|---|
| `evidence/` | Claim ledger, protocols, manifest, hashes and provenance |
| `paper/` | LaTeX build of the report (`make` needs pandoc and tectonic) |
| `scripts/` | Shell drivers for the model runs, task generators, the manifest builder and `validate_pyrefly_lsp.py` |
| `scripts/analysis/` | Reproducers and statistical analysis, including `reproduce_all.py` and `stats_delivery.py` |
| `scripts/experiments/` | Retrieval, navigation, substitution and checker harnesses |
| `scripts/realbench/` | Real-repository candidate scanning and dispatch experiments |
| `scaffold/` | Agent loop, tools and workspace environments |
| `runs/agent/`, `runs/pilot/` | Archived raw model results; the `synth_*.json` files in `runs/agent/` are the delivery-timing arms |
| `runs/protocol/` | Mechanical validation and frozen selection artifacts, plus the resolver-versus-Pyrefly agreement check |
| `runs/confirmation/`, `runs/readreq/` | Pre-registered confirmation and read-required boundary results |
| `runs/realbench/` | Dispatch grids and the repository candidate scans |
| `tests/` | Unit tests over the harness and the analyzers, run by the reproducer |
| `docs/`, `log.md` | Chronological research logs, bibliography and the external-validity reconnaissance. Historical, not a claim source. |
