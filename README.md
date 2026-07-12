# When Do Language Servers Help Coding Agents?

This repository studies a conditional engineering question rather than defending a universal LSP claim:

> When do semantic retrieval, readable/inferred types, checker feedback, and server-backed operations
> improve a coding agent after election, actionability, latency, and context costs are counted?

The working model is:

`opportunity × correctness/uniqueness × compression × election × actionability − cost`.

LSP is the transport; a language server is the semantic implementation; types are semantic substrate; a
type checker produces consistency diagnostics; and pull tools, pushed context, patch-boundary feedback,
gates, rerankers, constrained decoding, and training rewards are distinct integrations.

## Current findings

**Supported, narrow positive.** A compact definition result cuts pooled input tokens 3.5–4.7x at unchanged
success when the counterfactual is a whole-file library read. After averaging rollout seeds within task,
the treatment is cheaper on 11/11 local-27B tasks, 11/11 Sonnet tasks, and 10/11 DeepSeek tasks. Mean
per-task ratios are 4.02, 3.65, and 6.03; medians are 3.83, 3.01, and 2.70. Most of these runs use a static
AST resolver, so this supports compact retrieval, not LSP transport in general.

**Supported policy result.** Tool election is model- and policy-dependent. Prompting works in some
capable-model runs; on-policy relabeling and cost-reward training change a 7B model’s retrieval choice.
This is not evidence for a monotonic parameter-count law.

**Tentative negative regimes.** A historical dispatch suite shows near-one token ratios when a capable
agent can read the concrete receiver and open the target directly. The suite leaks that information and
has no erased-type arm, so it does not prove semantic navigation is redundant. Historical checker arms
compare unrelated first drafts, and their residual counts were contaminated by an in-workspace test file.
Corrected replay nevertheless finds checker-positive diagnostics in both coherent recoverable historical
workspaces. Execution/checker inference suites are at behavioral ceiling, but checker opportunity was not
measured there; neither fact establishes equivalence.

**New apparatus evidence.** The repaired `navigation-v2` typed/erased apparatus uses sound per-key `Literal`
overloads in `.pyi` files over byte-identical runtime code. It passes base/gold, held-out, all-key runtime
contract, type-cleanliness, leakage, widened-call, and strict live-Pyrefly checks on all four pilot instances and twelve additional
apparatus-audit instances. Those 12 are not confirmation because validator changes followed their first
inspection; a new disjoint 12-instance confirmation split passes mechanical checks and is reserved behind
source hashes without model exposure. The earlier 7B `navigation-v1` pilot used an unsound gold-derived
factory return and is retained only as invalid apparatus history. In v2, 7B fails the buggy-span control
0/2 and a pinned 14B rung reaches 1/2, so neither runs a matrix. After a tested compact-edit parser fix,
pinned Qwen3.6-27B clears both controls 2/2 and runs the sound matrix. All 12 cells pass, making correctness
non-identifying; typed automatic context falls within the token-ratio margin on these two pilot tasks only,
erased automatic context adds cost, and every automatic result is followed by a target read. Automatic
resolution also adds about six seconds of composed harness lookup time per task. This is a
precision/overhead boundary, not the missing useful-compression result, so confirmation remains blocked.
Fresh checker calibration stops correctly: 7B produces 0/3 coherent drafts, while 14B produces 2/8 and
both are type-clean. Neither reaches the 20–70% opportunity band.
A pre-treatment selection artifact therefore freezes the two exact recovered coherent checker-positive
workspaces for a conditional case series. Pinned Qwen3.6-27B completes paired control, patch-boundary, and
gate revisions. The first replay is rejected as outcome evidence: ambiguous same-line edit serialization
introduces syntax errors on the harder task, creating the apparent terminal-cleanliness difference. Every
arm remains 1/2 on joint accepted-clean-correct. The unresolved gate trajectory never invokes the gate, so
the run supplies no gate-prevention evidence.
A protocol-v2 rerun removes that artifact and records completion/gate events separately. One-shot
diagnostics finish type-clean on 2/2 selected workspaces versus 1/2 control, but every arm remains 1/2 on
held pass and accepted-clean-correct; diagnostics add 217 mean revision tokens. The gate accepts the clean
task, while the unresolved task never invokes it. This is intermediate checker-state evidence, not a
correctness, prevalence, or prevention result.

See [REPORT.md](REPORT.md) for the evidence, limitations, related work, and practitioner decision table;
[evidence/claim_ledger.md](evidence/claim_ledger.md) for claim-by-claim status; and
[evidence/protocols.md](evidence/protocols.md) for the frozen protocols and zero-paid-spend run budget.

## Fast reproduction

Python 3.10+ is required. Install the analysis/LSP development dependencies, then run the no-model
reproducer:

```bash
python3 -m pip install -e '.[dev,analysis]'
python3 scripts/analysis/reproduce_all.py
```

The command verifies `evidence/manifest.json`, reruns every analyzer retained in the report, recomputes
task-level effects, and reruns all navigation mechanical gates. It makes no model or API calls.
Pyrefly is discovered from `STREAMS_PYREFLY`, `PYREFLY_BIN`, PATH, `.venv/bin`, or
`.venv-streams/bin`.

Targeted commands:

```bash
python3 scripts/analysis/stats.py
python3 scripts/analysis/effic_real_stats.py \
  --base runs/agent/er2_27b_readonly.json \
  --trained runs/agent/er2_27b_base.json --label 27B
python3 scripts/analysis/task_level_effects.py \
  --base runs/agent/er2_27b_readonly.json \
  --treatment runs/agent/er2_27b_base.json --label 27B
python3 scripts/experiments/navigation_tasks.py --split pilot
```

## Expensive runs

No paid API run is authorized. Local pilots have a monetary cap of `$0`. Navigation model artifacts use
an explicit run tag and refuse overwrites, preserving every model rung and the invalidated v1 evidence.
The calibration driver exactly regenerates the three reported 1,400-token batches; the separate case-series
driver runs the preselected checker-positive cohort:

```bash
RUN_ID=qwen25coder14b-pilot001 \
  MODEL=Qwen/Qwen2.5-Coder-14B-Instruct \
  REVISION=aedcc2d42b622764e023cf882b6652e646b95671 \
  PYTHON=python3 scripts/run_navigation_pilot.sh
PYTHON=python3 scripts/run_checker_paired.sh
PYTHON=python3 scripts/run_checker_case_series.sh
```

The confirmation command is intentionally separate and runs only after the v2 gold-copy and buggy-span
controls and pilot clear the preregistered gates. Against the completed 27B pilot below, it exits before
model loading because the shared parser changed after the run; the at-run pilot is also uniformly at ceiling
and therefore scientifically ineligible for confirmation:

```bash
PILOT_RUN_ID=qwen36-27b-6a9e13bd-pilot002 RUN_ID=qwen36-27b-confirm001 \
  MODEL=Qwen/Qwen3.6-27B \
  REVISION=6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 \
  PYTHON=python3 scripts/run_navigation_confirmation.sh
```

Historical OpenRouter drivers still require `OPENROUTER_API_KEY` or `.orkey` and enforce their own
`--budget-usd` limits. Do not run them merely to reproduce the report; committed JSON is sufficient.

## Decision recipe

1. Use text search and ranged reads for unique, local bindings.
2. Use typed semantic resolution for genuinely non-lexical ambiguity.
3. Keep semantic retrieval only when it replaces substantial reading or prevents wrong-file work.
4. Run target-scoped diagnostic deltas on coherent agent patches; checker cleanliness is not correctness.
5. Gate only when telemetry shows prevention of defects the ungated agent would actually submit.

Compact retrieval has narrow repository support; useful semantic-navigation value remains open. The repaired
checker replay shows a selected terminal-state effect without correctness gain; gate prevention remains open.
The confidence-labeled decision table and claim links are in
[REPORT.md](REPORT.md#6-practitioner-decision-table).

Always measure whether semantic retrieval substitutes for a read, the checker opportunity rate, whether
the model edits diagnosed locations, and accepted-defect rate. Static tooling will not solve dynamic
behavior, logical errors outside the type system, incorrect annotations, `Any` boundaries, or failures the
model cannot act on.

## Repository layout

| path | purpose |
|---|---|
| `REPORT.md` | final conditional argument and practitioner framework |
| `evidence/` | claim ledger, preregistered protocols, hashes, provenance warnings |
| `runs/agent/` | committed historical raw model results |
| `runs/protocol/` | mechanical checks and draft-recovery audit artifacts |
| `scripts/experiments/` | typed/erased navigation and paired-checker harnesses |
| `scripts/analysis/` | historical and task-level analyzers plus the fast reproducer |
| `scripts/realbench/` | SWE-bench scanning and historical dispatch tooling |
| `scaffold/` | agent and workspace environments |
| `docs/real_repo_progress.md` | preserved chronological research log; not the final claim source |

Historical evidence is preserved. Where raw data or exact provenance is missing, the claim ledger marks
the result tentative or unsupported instead of reconstructing certainty from `log.md` summaries.
