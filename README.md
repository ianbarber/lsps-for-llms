# When Do Language Servers Help Coding Agents?

*TL;DR*

Agents do well with with cheap text search and ranged reads. Add language-server features when:
* bindings are genuinely ambiguous
* compact definitions can actually replace search and reading

A type-checker gate can catch a defective submissions, but add it at submission time, not after edits.

When trying it, measure whether the service changes what the agent actually does (in terms of task success or token efficiency), not how often it is called.

## Findings: where do LSPs help

**1. Resolution: mostly no, because readable types already do the work.** A capable model resolves dispatch-ambiguous targets by reading the receiver's type wherever it appears in visible source, then opens the right file directly. Live go-to-definition is token-neutral (ratios 0.94–1.07) even when the type sits behind factory indirection only the language server can statically resolve. 

The exception is where type-bearing source is hidden, so the type must be *retrieved*. An example would be where there are many overrides with the same signature. Then text resolution falls and framed definition increases, sometimes leading to thrash and failed resolution.

**2. Efficiency: yes, when the span substitutes for reading.** Compact definitions cut total tokens 3.5–4.7x against whole-file retrieval and 1.30x against an efficient grep-plus-ranged-read interface. In order to benefit the agent must *elect* the operation and it then must  *not read the file anyway* Prompting lifts election on capable models, and training lifts a 7B from roughly 0% to 100% use. However most models will automatically re-read the definition after recieving it, hence negating any efficiency gains: across Qwen3.6-27B, Sonnet 4.5, and DeepSeek v3.1 the defining file was reread on 35 of 36 pushed spans, and explicitly instructing the model that the span is sufficient only removed the reread on 2 of 36. Training does work: relabel-tuning Qwen 3.6 27B removed the reread on 11 of the 11 held-out instances where it occurred, cutting matched-success tokens 1.59x and reads per task from 3.42 to 0.08.

**3. Checking: deliver it late, and prefer a submission gate.** On twelve identical seeded defects, checker delivery at revision or at submission lifts accepted type-clean and held-out-correct outcomes from 1/12 to 10/12 and 11/12; after-every-edit delivery changes nothing. Type-check as a gate: it rejected 10 of 12 bad submissions with every rejection completing the repair-retest-resubmit-accept cycle, at 0/12 false rejections, and it is basically free on clean work (591 tokens, same as no checker). More regular calls tax every passing draft.

## Evidence strength at a glance

| Claim | Status |
|---|---|
| Semantic navigation helps when types are readable in source | **Not shown** — token-neutral across the dispatch suite and type-location ladder |
| Semantic resolution helps when the type must be retrieved | **Directional, single grid** — 14/15 framed lookup vs. 12/15 text, two localization rescues, n=15 at one seed |
| Typed semantic resolution picks the right target | **Mechanism supported** — resolver precision gain shown; agent-level benefit not shown |
| Compact definitions cut tokens vs. efficient text retrieval | **Supported in a controlled suite** — 1.30x fewer total tokens at equal success across 11 tasks; 3.5–4.7x vs. whole-file reads |
| Election is policy-shapeable | **Supported per-model** — prompting lifts capable models; training lifts a 7B from ~0% to 100% |
| Pushed spans get reread rather than substituted | **Supported across three models** — 35/36 rereads; an explicit sufficiency instruction removed 2/36 |
| Training removes the reread and recovers the saving | **Supported on held-out instances** — 11/12 to 0/12 reread on the 27B, 1.59x fewer tokens; one model, one seed, held-out pass 11/12 to 10/12 |
| Late checker delivery (revision or submission) improves outcomes | **Supported at n=12 pairs** — 10/12 and 11/12 accepted-correct vs. 1/12 control, bootstrap intervals exclude zero |
| Submission gates recover checker-detectable bad completions | **Supported at n=12 pairs** — 10/12 rejected and 10/10 repaired, resubmitted, accepted; 0/12 clean drafts falsely rejected |
| After-every-edit checker feedback helps | **Not shown** — +0.000 [−0.125, +0.125] vs. control, though the channel fired in only 1/12 rows |

## Operational defaults

1. **Start with text search and ranged reads.** A capable agent self-localizes by reading type information in visible source; navigation adds latency.
2. **Add typed semantic resolution where the type is not readable in source the agent already sees.** Retrieved-type situations and genuinely ambiguous bindings (overloads, inheritance, re-exports, factories) are the candidates. Expect the payoff in resolution rate, not tokens, and verify it prevents wrong-target work rather than just getting called.
3. **Use compact definitions when they replace search and reading.** Against grep plus ranged reads, definitions cut mean total tokens assuming zero defining-file rereads. Check for rereads: a span the agent reads *past* is pure overhead.
4. **Prompt for election; train for substitution.** Strong system framing lifted a frontier model's use of the definition tool where mild advertisement did not, and relabel training took a 7B from ~0% to 100% use. But telling a model to trust a pushed span did not stop it rereading the file, while relabel training removed that reread outright and cut tokens 1.59x. If you cannot train the policy, let the agent elect the call rather than pushing it.
5. **Run the checker as a gate at submission, with an explicit repair-and-resubmit loop.** Do not stream diagnostics into authoring. A gate adds tokens only on defective submissions, while unconditional revision diagnostics tax every clean draft. Measure false rejections, resubmission, accepted-correct yield, and total cost on your own workload before trusting it.

## Caveats:

- Agent-level outcome gains from typed resolution: the tasks we used here were just not hard enough to give signal.
- Natural prevalence of gate opportunities and population rejection precision: the benefit will depend on how many bugs are in the codebase as a whole

## Read the work

- [REPORT.md](REPORT.md) presents the findings, evidence, strength, and limits per question.
- [evidence/claim_ledger.md](evidence/claim_ledger.md) maps every important claim to its artifacts and evidence status, including excluded and invalidated results.
- [evidence/protocols.md](evidence/protocols.md) records the experiment protocols, stopping gates, and execution status.
- [evidence/manifest.json](evidence/manifest.json) records hashes, model metadata, integration modes, and provenance warnings.

## Reproduce the analysis

Python 3.10+ is required:

```bash
python3 -m pip install -e '.[dev,analysis]'
python3 scripts/analysis/reproduce_all.py
```

The reproducer verifies the manifest, reruns the retained analyzers, recomputes task-level effects, and reruns the navigation mechanical checks. It uses committed artifacts and makes no model or API calls. Pyrefly is discovered through `STREAMS_PYREFLY`, `PYREFLY_BIN`, `PATH`, `.venv/bin`, or `.venv-streams/bin`.

Model execution is separate from reproduction. See [evidence/protocols.md](evidence/protocols.md) before using `scripts/run_navigation_pilot.sh`, `scripts/run_navigation_confirmation.sh`, `scripts/run_checker_paired.sh`, `scripts/run_checker_case_series.sh`, `scripts/run_checker_hidden.sh`, `scripts/run_checker_gate_v2.sh`, or `scripts/run_checker_gate_v3.sh`. No paid API run is authorized by the protocol.

## Repository map

| Path | Purpose |
|---|---|
| `REPORT.md` | Findings and technical report |
| `evidence/` | Claim ledger, protocols, manifest, hashes, and provenance |
| `runs/agent/` | Archived raw model results |
| `runs/pilot/` | Pilot and case-series results |
| `runs/protocol/` | Mechanical validation and frozen selection artifacts |
| `scripts/analysis/` | Reproducers and statistical analysis |
| `scripts/experiments/` | Retrieval, navigation, and paired-checker harnesses |
| `scripts/realbench/` | Real-repository candidate scanning and dispatch experiments |
| `scaffold/` | Agent loop, tools, and workspace environments |
| `docs/real_repo_progress.md` | Preserved chronological research log, not the final claim source |
