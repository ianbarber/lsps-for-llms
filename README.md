# When Do Language Servers Help Coding Agents?

Language-server services help coding agents when repository semantics are the bottleneck and the result
changes the agent's work: resolving an ambiguous binding, replacing broad reading with compact context, or
surfacing a defect that can be repaired or rejected.

This repository turns controlled experiments on those three jobs into practitioner guidance. It contains
raw artifacts, reproducible analysis, experiment harnesses, and a technical report covering semantic
retrieval, typed resolution, checker feedback, tool election, and gates.

## Operational defaults

1. **Use text search and ranged reads for unique, local bindings.** They are cheap and transparent.
2. **Use typed semantic resolution when text cannot identify the exact target.** It is most relevant for
   overloads, inheritance, re-exports, factories, and same-named implementations. The repository demonstrates
   resolver precision here, not an agent-level outcome gain.
3. **Keep semantic retrieval when it changes localization or replaces work.** A correct result that neither
   prevents wrong-target work nor replaces reading is overhead.
4. **Check coherent patches for new relevant type errors.** Cleaner checker state is an intermediate signal,
   not behavioral correctness; the repository does not show a correctness gain.
5. **Treat gates as unproven until they prevent bad submissions.** The local comparison is invalid and
   contains no rejection event.

Measure effect, not invocation: did the service replace reading, prevent a wrong-target edit, repair an
actionable defect, or reject a bad submission? Optimize tool election only after the service demonstrates
value when it is available.

## Evidence at a glance

| Job | What the experiments show | Practitioner reading |
|---|---|---|
| **Resolve** | Navigation is near token-neutral when text already exposes the target. Sound types let Pyrefly distinguish the correct implementation from same-named alternatives, but every automatic result in the two-task agent pilot is followed by a target read. | **Resolver mechanism supported; agent benefit open.** Use semantic resolution for genuine ambiguity and measure wrong-target work as well as tokens. |
| **Compress** | Compact definitions reduce input tokens 3.5-4.7x at unchanged success when they replace whole-file reads. The three-model comparison uses a static AST resolver; a live-first hybrid shows the same direction. | **Supported against a coarse baseline.** Use compact spans when they replace broad retrieval; the advantage over efficient ranged reads remains unproven. |
| **Validate** | On two selected workspaces with checker-detectable errors, diagnostics produce one additional type-clean result, no held-out gain, and 217 extra revision tokens. The gate comparison contains no valid prevention contrast or rejection event. | **Intermediate effect only.** Treat checker cleanliness as a signal; require behavioral improvement or demonstrated prevention for deployment value. |

Prompting, relabeling, and cost-reward training change retrieval-tool election in model-specific runs. That
matters only after the service itself has shown value.

## Read the work

- [REPORT.md](REPORT.md) presents the practitioner guide, evidence, limitations, and related work.
- [evidence/claim_ledger.md](evidence/claim_ledger.md) maps every important claim to its artifacts and evidence status.
- [evidence/protocols.md](evidence/protocols.md) records the experiment protocols, stopping gates, and execution status.
- [evidence/manifest.json](evidence/manifest.json) records hashes, model metadata, integration modes, and provenance warnings.

The claim ledger includes excluded and invalidated results for auditability, and marks evidence as missing,
confounded, or unsupported where appropriate.

## Reproduce the analysis

Python 3.10+ is required:

```bash
python3 -m pip install -e '.[dev,analysis]'
python3 scripts/analysis/reproduce_all.py
```

The reproducer verifies the manifest, reruns the retained analyzers, recomputes task-level effects, and
reruns the navigation mechanical checks. It uses committed artifacts and makes no model or API calls.
Pyrefly is discovered through `STREAMS_PYREFLY`, `PYREFLY_BIN`, `PATH`, `.venv/bin`, or
`.venv-streams/bin`.

Model execution is separate from reproduction. See [evidence/protocols.md](evidence/protocols.md) before
using `scripts/run_navigation_pilot.sh`, `scripts/run_navigation_confirmation.sh`,
`scripts/run_checker_paired.sh`, or `scripts/run_checker_case_series.sh`. No paid API run is authorized by
the protocol.

## Repository map

| Path | Purpose |
|---|---|
| `REPORT.md` | Practitioner guidance and technical report |
| `evidence/` | Claim ledger, protocols, manifest, hashes, and provenance |
| `runs/agent/` | Archived raw model results |
| `runs/pilot/` | Pilot and case-series results |
| `runs/protocol/` | Mechanical validation and frozen selection artifacts |
| `scripts/analysis/` | Reproducers and statistical analysis |
| `scripts/experiments/` | Navigation and paired-checker harnesses |
| `scripts/realbench/` | Real-repository candidate scanning and dispatch experiments |
| `scaffold/` | Agent loop, tools, and workspace environments |
| `docs/real_repo_progress.md` | Preserved chronological research log, not the final claim source |
