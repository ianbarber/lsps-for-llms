# When Do Language Servers Help Coding Agents?

## Abstract

Language servers are not uniformly useful or redundant for coding agents. Their realized value is a
chain:

> opportunity prevalence × signal correctness and uniqueness × retrieval/compression advantage ×
> agent election × ability to act − tool, latency, and context costs.

The strongest committed result in this repository is narrow and positive. When an agent otherwise reads
a whole library file, a compact definition result preserves success while reducing input tokens by
3.5–4.7x in pooled comparisons across a local 27B model and two frontier models. Task-level reanalysis
confirms the direction on 11/11, 11/11, and 10/11 tasks. This is evidence for compact semantic retrieval
against a whole-file baseline, not for the general value of the Language Server Protocol (LSP).

The negative regimes are equally important. In a historical dispatch suite, a capable agent often read
the concrete receiver type and opened the correct file directly, so semantic navigation was near
token-neutral. That suite leaked the disambiguating class and never erased the type, so it cannot support
the old conclusion that navigation never beats a readable type. Historical checker experiments compared
independently generated trajectories and delivered either elective or after-every-edit diagnostics. Their
residual counts were polluted by type-checking the generated test runner, although corrected replay now
finds genuine checker-positive opportunities in two coherent historical final workspaces.

We therefore replace the old “types, not language servers” conclusion with a decision framework and two
causal protocols. Experiment 1 pairs sound typed and erased factory stubs over byte-identical runtime code
and automatically supplies strict live Pyrefly results to separate semantic information from election. Its
repaired manipulation checks pass. The earlier 7B model pilot is invalid because its typed factory used a
false gold-derived return contract. In the repaired v2 pilot, the gold-copy control passes 2/2 but the
pristine buggy-span actionability control fails 0/2. A pinned Qwen3.6-27B pilot clears both repaired
controls and runs the sound matrix, but all 12 causal/deployment cells pass and automatic results are
followed by target reads, so correctness is non-identifying and useful compression is not demonstrated.
Experiment 2 freezes natural drafts before paired revisions. Fresh 7B and 14B calibration misses the
required coherence/opportunity band. A repaired two-task selected replay finds one additional type-clean
terminal workspace with one-shot diagnostics, but no held-out or accepted-clean-correct gain; its unresolved
gate trajectory never invokes the gate.
The practical conclusion is conditional: use the cheapest integration that supplies unique, actionable
information at an actual opportunity, and measure whether it replaces work rather than merely preceding
it.

## Practitioner recipe

1. **Use text search and ranged reads for unique, local bindings.** They are cheap, transparent, and already
   effective when the relevant fact is nearby ([C6](evidence/claim_ledger.md#c6),
   [C9](evidence/claim_ledger.md#c9)).
2. **Use semantic resolution for genuinely non-lexical ambiguity.** Sound readable or inferred types can
   make an override, overload, re-export, or factory result unique; erased boundaries reduce that precision
   ([C15](evidence/claim_ledger.md#c15), [C24](evidence/claim_ledger.md#c24)).
3. **Keep semantic retrieval only when it replaces substantial reading or prevents wrong-file work.** A
   correct definition followed by the same target read is overhead. Replacement of whole-file reads is
   supported narrowly; replacement of efficient ranged retrieval is not ([C1](evidence/claim_ledger.md#c1),
   [C4](evidence/claim_ledger.md#c4), [C24](evidence/claim_ledger.md#c24)).
4. **Run target-scoped diagnostic deltas on coherent agent patches.** Treat a cleaner checker state as
   intermediate evidence, never as behavioral correctness. For autonomous agents that assemble a patch
   through multiple low-level edits, defer deduplicated semantic deltas to a coherent boundary; this warning
   does not apply when each edit is already an atomic coherent patch ([C11](evidence/claim_ledger.md#c11),
   [C26](evidence/claim_ledger.md#c26)).
5. **Gate only when telemetry shows prevention.** A useful gate must reject a dirty completion the ungated
   agent would have submitted, or turn it into an accepted clean and behaviorally correct patch. Otherwise
   it is rejection cost without demonstrated product value ([C14](evidence/claim_ledger.md#c14),
   [C26](evidence/claim_ledger.md#c26)).

## 1. Terms and causal model

These are different objects and must not be treated as competitors:

- **LSP** is a transport protocol between an editor/client and a semantic service.
- A **language server** is the implementation that parses, infers, indexes, resolves, and validates code.
- **Type annotations and inferred types** are semantic information in or derived from code.
- A **type checker** tests consistency and emits diagnostics. It may be a CLI, a server feature, or a
  compiler phase.
- An **integration mode** determines when an agent receives or uses the information: pull retrieval,
  pushed context, coherent-patch feedback, acceptance gate, reranking, constrained decoding, or training
  reward.

Types are semantic substrate. A language server can infer types, follow bindings, index definitions,
validate a patch, and expose the result compactly. A type may also be readable directly in source; a
checker may run without LSP; an LSP server may provide useful syntax or index operations without a rich
type system. The scientific question is which information and integration changes behavior at acceptable
cost.

For task `t`, a useful decomposition is:

`value(t) ≈ opportunity(t) × correctness(t) × uniqueness(t) × compression(t) × election(t) × actionability(t) − cost(t)`.

A zero at any multiplicative stage explains many nulls. A perfect checker has no revision value on a
type-clean draft. A perfect goto has no deployment value when it is not elected. A compact result has no
compression value when the agent still reads the same file. Correct diagnostics have no outcome value if
the model cannot make a valid edit. Tool latency and repeated context can make a correct signal net
negative.

## 2. What a server can provide

| value | useful mechanism | redundant when | necessary conditions |
|---|---|---|---|
| Semantic information | infer receiver/type/binding unavailable in the visible text | the fact is already visible or cheaply derived | the server resolves the relevant language construct correctly and more uniquely than text search |
| Context compression | return a definition, signature, hover, or references instead of a large read | the baseline already uses a small ranged read, or the agent reads the file after the result | result replaces retrieval and includes enough editable context |
| Disambiguation/indexing | resolve re-exports, overloads, inheritance, factories, or same-named methods | names are lexically unique or the receiver is obvious | correct project configuration, index coverage, and static resolvability |
| Validation | detect inconsistent calls, attributes, keys, returns, or imports | natural drafts are type-clean or errors lie outside the checker | new actionable diagnostics, coherent delivery, and successful revision |
| Gate/reranking | keep dirty patches out or prefer clean candidates | false positives dominate or type cleanliness is irrelevant | delta-scoped diagnostics and an explicit acceptance policy |
| Safe structured operations | rename, references, AST-aware edits, syntax validation | a simple text edit is already safe and unique | operation preserves semantics and its preconditions are checked |
| Supervision/reward | teach tool election or penalize compiler/type failures | the deployed policy already elects and acts correctly | training examples match deployment opportunities and costs |

## 3. Evidence audit

The machine-readable hashes and configurations are in `evidence/manifest.json`; the full claim mapping is
in `evidence/claim_ledger.md`. This section retains the supported historical evidence while separating
static resolvers, live servers, hybrids, checkers, and execution feedback.

### 3.1 Supported: compact retrieval beats whole-file reads

The `effic` and `effic_real2` tasks require an agent to implement a small target using an unfamiliar
library API. The treatment returns a compact definition; the counterfactual removes that action and
drives whole-file retrieval. Most headline files use the repository’s static AST resolver, not a live
language server. All committed tool-value files used temperature 0.7, despite older report prose saying
0.0.

| model / treatment | tasks × seeds | no-definition tokens | compact-definition tokens | success | task-level effect |
|---|---:|---:|---:|---:|---|
| Qwen3.6-27B, static definition tool | 11 × 4 | 4563 | 1302 | 44/44 both arms | cheaper on 11/11 tasks; mean task ratio 4.02, 95% bootstrap 2.85–5.30; median 3.83 |
| Claude Sonnet 4.5, static definition tool | 11 × 4 | 21,985 | 6018 | 44/44 both arms | cheaper on 11/11; mean 3.65, 2.72–5.20; median 3.01 |
| DeepSeek v3.1, static definition tool | 11 × 4 | 36,192 | 7705 | 44/44 both arms | cheaper on 10/11; mean 6.03, 2.32–10.82; median 2.70 |

The pooled token ratios are 3.50x, 3.65x, and 4.70x. The difference between DeepSeek’s mean and median
shows why tool-loop distributions must not be summarized by a mean alone. The expected token cost per
success equals the per-arm token mean here because every attempt succeeds. Seeds are nested repetitions;
the intervals resample tasks after averaging within task.

The local 7B training experiments provide a complementary policy result. A read-trained and a
definition-trained policy both solve 40 matched `(task, seed)` cells, while input tokens fall from 3191
to 684 (4.7x). A clean relabel retest changes definition election from approximately zero to 100%, and a
cost-reward GRPO retest reaches 86% definition use, 663 input tokens, and 36/36 success. These comparisons
show that election can be changed by prompting or training. They do not establish a monotonic relationship
with parameter count because model family, training, and policy differ.

### 3.2 Tentative: live-server transfer

`lsp_base.json` and `lsp_sft.json` use live-first Pyrefly on 12 tasks × two seeds. The observed token mean
falls from 2894 to 689 while success rises from 14/24 to 24/24. The historical implementation uses AST
logic to find a use site and expand the returned span, and it can fall back after an LSP miss. Rows do not
record which backend answered. This is evidence that the composed live-first tool can reproduce the
controlled behavior, but it is not a pure server ablation.

Files named `reallsp_*` are not live LSP runs: they use the static resolver over a live workspace. The
manifest labels this explicitly. Historical local logs mention Pyrefly 1.0.0; the external x86 dispatch
host used 1.1.1. No historical result JSON records a server version, so neither version is assigned
globally.

### 3.3 Tentative negative: realistic retrieval can erase compression advantage

A three-task mini-swe-agent probe found that a capable shell agent used grep and ranged reads rather than
whole-file reads, and definition calls were often additive. Full raw logs are not committed, so this is a
case study only.

The later 15-task dispatch suite compared grep/ranged-read retrieval with a live Pyrefly goto composed
with an enclosing span. For Qwen3.6-27B on the annotated variant, grep, neutral goto, and framed goto solve
15/15, 14/15, and 15/15, with matched-success token ratios 0.972 and 1.041. Stripped and factory-indirection
variants produce descriptive ratios from 0.945 to 1.065. A 7B model solves only 2–4 of 15 depending on the
arm, with no common matched-success set.

These trajectories show a negative regime: when the prompt exposes a concrete receiver and the correct
file is cheap to read, goto may not save tokens. They do **not** prove equivalence or that navigation never
beats a readable type. The prompt explicitly described the static receiver, showed concrete construction,
and even claimed a call-site annotation in variants where it was removed. No erased/`Any` condition existed,
and no equivalence margin was preregistered. The former broad conclusion is unsupported.

### 3.4 Checker evidence: opportunity and integration were confounded

The held-out `gapd2` inference suite is a ceiling observation. For each of two frontier models, all six
rich tasks × three seeds pass in the hinted no-checker, hinted elective-checker, and unhinted no-checker
conditions: 18/18 in each cell. This shows no behavioral headroom. The artifacts do not preserve a common
natural first draft or diagnostic deltas, so checker-opportunity prevalence is unmeasured rather than zero.

The 12-task authoring experiment compares no checker, elective checker, and automatic feedback after
each edit. Qwen3.6-27B passes 12/12 held-out in every independently generated arm. Qwen2.5-Coder-7B passes
6/12 without a checker, 3/12 with an elective checker, and 4/12 with after-every-edit feedback; the noisy
arm uses 2.35x the input tokens. Because first drafts differ across arms, this is evidence only that these
particular live integrations did not help, not that coherent diagnostics or a gate are unusable.

The old residual-diagnostic metric is invalid. Tests wrote `_run_tests.py` into the checked workspace,
and Pyrefly diagnosed that generated test. Replaying five exact uncapped no-checker final trajectories
recovers two coherent workspaces; both are checker-positive, with three and one new target-scoped semantic
diagnostics. Three other recovered workspaces are incoherent, and seven trajectories cannot be recovered
because historical edit bodies were truncated. This proves opportunity existed in a narrow historical
subset, but these are final trajectories without a first-draft submission boundary or paired revisions.

### 3.5 Other boundaries and unsupported history

Execution feedback is also ceilinged on the committed small-task suite. Two frontier models × 14 tasks ×
three seeds pass in all three arms (no execution, elective execution, automatic execution), 252/252 total
attempts. Elective execution is always used and automatic feedback saves turns, but absence of failures is
not equivalence beyond small, simulable functions.

Older claims about correction, completeness, prevention, navigation, 21–86-file scaling, and a 35B model
are backed only by `log.md`, not committed raw JSON and analyzers. The 35B log actually records a positive
direction under a ceiling and calls it inconclusive. Those claims are excluded from the empirical
conclusion rather than deleted from repository history.

## 4. New experiments

### 4.1 Experiment 1: readable types × semantic navigation

`navigation-v2` generates paired repositories with byte-identical runtime `.py` trees, tests, and
gold patches. Each begins at `x.<neutral_method>(...)`, has 8–15 same-named overrides across files, and
constructs the receiver through a remote factory/registry. Neutral names are deterministically randomized.
The prompt and visible failure contain neither the concrete receiver class nor gold path.

Only `factory.pyi` differs. The typed stub soundly overloads every registry key with its actual concrete
return and retains a base-returning `str` fallback; the erased stub returns only the base. The visible call
retains the selected key as a `Literal`. Every declared return is checked against runtime construction, both
variants are type-clean, and a widened `str` call must resolve back to base/null. The textual baseline can
still trace the registry. The automatic treatment
strictly asks live Pyrefly for definition-at-use-site and expands only the enclosing method; it never falls
back to the AST resolver. Thus the four causal cells are typed/textual, typed/automatic result,
erased/textual, and erased/automatic result. Typed deployment cells separately measure neutral election,
cheap/precise framing, and the automatic upper bound.

The automatic result is now neutrally identified as current source context rather than an unexplained XML
appendix. A separate oracle localization/actionability control supplies the byte-identical pristine buggy
span from task metadata with the same framing and budget; it is outside the factorial estimate. The existing
gold-copy control remains a stricter edit-competence floor. Failure of either control blocks interpretation
of a model null.

All four repaired pilot instances across two development templates and twelve additional apparatus-audit instances
across three disjoint templates pass every mechanical check: visible and held-out
base tests fail, both gold tests pass, every runtime `.py` matches, all overloads match construction, both
variants are type-clean, gold is identical, override counts are valid,
typed goto reaches the gold override, erased goto returns a non-discriminating base/null result with an
independent server-health query, and prompts
contain no concrete class/path leakage, and widening removes discrimination. This is a **mechanical
supported result**, not a model outcome. The
apparatus-audit instances are not called confirmation because they were observed while the validator was
still changing. A new disjoint 12-instance, three-template confirmation split is reserved behind protocol
source hashes and passes the same mechanical checks; no model has seen it.

The earlier local 7B `navigation-v1` pilot is invalid for this causal question. Its typed factory promised
the one gold class for every string key and reinforced the false annotation with a cast. The committed 0/12
matrix and 2/2 gold-copy control are preserved as rejected apparatus history, but its localization and token
differences are not evidence about normal typed code. In `navigation-v2`, 7B passes gold-copy 2/2 but fails
buggy-span 0/2. A pinned 14B rung passes gold-copy 2/2 and buggy-span 1/2, so its matrix is withheld. The
first Qwen3.6-27B attempt exposes a compact-edit parser incompatibility: it emits the correct edits on the
same line as the opening tag, but the harness records zero edits. That attempt is preserved. A regression-
tested parser repair accepts both compact and newline forms; under a new run tag, the same pinned 27B
revision passes gold-copy and buggy-span controls 2/2 and opens the matrix.

All 12 two-task matrix cells pass, so pass effects are non-identifying. Typed automatic context has a total-
token ratio of 1.037 versus typed baseline (exploratory task bootstrap 0.988–1.093); the ratio and interval
fall inside the prespecified 0.90–1.10 margin on these two pilot tasks only. This is not population
equivalence and excludes latency. Erased automatic context costs 1.190x baseline (1.119–1.251), for an
observed two-task interaction of -214 tokens (-402 to -26). Every automatic payload is followed by a target-
file read. Composed live resolution takes about 6.03 seconds typed and 6.02 seconds erased per task, dominated
by the harness's fixed indexing wait. Automatic lookup occurs before `wall_sec` starts, unlike elective
calls; descriptively adding it gives 108.89 seconds typed automatic versus 64.49 baseline and 107.34 seconds
erased automatic versus 63.72 baseline. Neutral elective navigation is chosen 0/2; cheap/precise framing
raises election to 1/2. This is a precision/overhead boundary, not a correctness benefit or deployable
navigation win. Confirmation remains blocked because the pilot is at behavioral ceiling.

### 4.2 Experiment 2: paired checker opportunity and integration

`checker-paired-v1` first produces one natural draft with no checker, stores the exact workspace and hash,
then forks control, coherent-patch diagnostics, acceptance gate, and the historical noisy after-every-edit
integration from identical bytes. Diagnostics are target-scoped deltas from the pristine project,
deduplicated, uncapped in raw results, classified as syntax/partial versus semantic, and presented with a
file, range, code, concise message, and code frame. Tests run outside the checked tree.

Opportunity is a new semantic diagnostic on the natural draft. Calibration requires 20–70% opportunity
among coherent submitted drafts while preserving viable edits. Analysis reports revision efficacy among
coherent submissions and on the pre-treatment opportunity subset, then separately reports end-to-end yield
across every generated draft with unsubmitted and incoherent drafts counted as pipeline failures. The gate
rejects `done` while new errors remain; its product
metrics include accepted type-clean correctness, abstention, and task-weighted draft-plus-revision tokens
per accepted held-out-correct patch. Rejected, abstained, and wrong attempts remain in the cost numerator.
One older local 7B smoke artifact
predates the explicit `<submit_draft/>` boundary and contains repeated partial edits; corrected collection
classifies all 14 diagnostics as syntax/partial cascades and rejects the workspace as incoherent. It is an
apparatus smoke, not a compliant calibration.

Fresh current-protocol calibration also stops before revisions. The 7B model submits 3/3 workspaces but
produces 0/3 coherent drafts. Across initial and extended 14B development batches, the model submits 7/8
and produces 2/8 coherent drafts; both pass held-out behavior and have no semantic diagnostic opportunity.
Thus neither regime reaches the
preregistered minimum of two coherent drafts or the 20–70% opportunity band. This is a capability/calibration
gap for the tested local models, not a checker null. Historical replay proves that checker-positive final
workspaces can occur, but it cannot estimate natural opportunity prevalence or same-model end-to-end cost.

The first conditional case-series replay selects two exact recovered checker-positive workspaces before
treatment and forks byte-identical control, one-shot pre-revision diagnostics, and gate arms. It is retained
as rejected apparatus evidence. The relaxed same-line edit parser preserves a separator space before a top-
level import on the harder task, causing every arm to introduce `IndentationError`; the diagnostic arm alone
spends another edit removing it. Its 2/2 versus 1/2 terminal type-clean difference is therefore entangled
with edit serialization. All arms eliminate the original semantic diagnostics and remain 1/2 on
`accepted && type_clean && held_pass`. The unresolved gate trajectory never attempts completion: it has
zero gate checks and zero rejection events. This run supplies no gate-prevention evidence, natural
opportunity estimate, or end-to-end cost because historical draft-generation tokens are unavailable
([C25](evidence/claim_ledger.md#c25)).

Protocol v2 then reruns the exact grid with indentation-anchored inline serialization, explicit completion
and gate events, null draft cost, and final workspace hashes. No ambiguous serialization failure occurs. On
the hard workspace, control eliminates the three initial diagnostics but introduces four new semantic
errors; one-shot diagnostics eliminates the initial diagnostics and introduces none. Both remain
behaviorally wrong and unsubmitted. On the second workspace, all arms produce byte-identical clean,
held-out-correct, accepted targets. Diagnostics therefore end type-clean on 2/2 versus control 1/2, at
1,585 versus 1,368 mean revision tokens, but every arm remains 1/2 on held pass and
`accepted && type_clean && held_pass`. This is descriptive intermediate-state evidence on two selected
workspaces with one seed, not a correctness or population effect. The gate is checked and accepts only the
already-clean task; the unresolved task never attempts completion, so `gate_rejections=0` and prevention is
not estimable ([C26](evidence/claim_ledger.md#c26)).

### 4.3 External validity

The bounded reconnaissance audited the committed SWE-bench scans under a one-hour cap with no new spend.
`django__django-11211` is a partial historical case with a working environment and 21 same-named
overrides, but prompt leakage and a discriminating fix-site goto were not audited; both historical arms
passed and the semantic tool was not elected. Other candidates fail or lack actual-fix
ambiguity, leakage, static-resolution, or environment checks. This is a partial historical case, not an
admissible external-validity result or a population estimate.
The table and rejection reasons are in `docs/external_validity_recon.md`. Constructed tasks remain the
instrumented apparatus; broader external validity is open, with CrossCodeEval as the preferred fallback.

## 5. Related work and reconciliation

Positive prior work generally targets missing context, automatic delivery, coherent drafts, or constrained
generation. Those regimes differ from a self-retrieving repair agent that can cheaply read the same fact.

- **Statically Contextualizing Large Language Models with Typed Holes** (OOPSLA 2024,
  <https://arxiv.org/abs/2409.00921>) pushes hole types, typing context, and remote definitions, then
  iterates with a language server. It supports semantic context injection when information is missing or
  nonlocal; it does not conflict with redundancy when the same fact is already readable in budget.
- **STALL+** (<https://arxiv.org/abs/2406.10018>) compares static analysis in prompting, decoding, and
  post-processing on CrossCodeEval. Effects vary by language and phase, and retrieval plus static analysis
  can be complementary. This motivates separating integration modes rather than asking whether “LSP” wins.
- **CoCoGen** (Findings ACL 2024, <https://arxiv.org/abs/2403.16792>) checks a coherent draft and retrieves
  repository context to repair project/API mismatches. That is precisely the opportunity-conditioned,
  patch-boundary regime omitted by the old after-every-edit study.
- **LSPRAG** (ICSE 2026, <https://arxiv.org/abs/2510.22210>) automatically supplies definitions and
  references for unit-test generation across Java, Go, and Python. Automatic delivery removes election
  failure and directly tests compact semantic retrieval, like Experiment 1’s upper-bound arm.
- **CodeStruct** (ACL 2026, <https://arxiv.org/abs/2604.05407>) studies AST-structured reads/edits and syntax
  validation. It supports safe structured operations and actionability, not a claim specific to types or
  LSP transport.
- **CompCoder** (<https://arxiv.org/abs/2203.05132>) uses compiler feedback for training and candidate
  discrimination; type-constrained generation (<https://arxiv.org/abs/2504.09246>) enforces constraints
  during decoding. These can help when repair-time feedback is not actionable.
- **RLCSF v2** (<https://arxiv.org/abs/2510.22907>) treats compiler/language-server feedback as process
  reward and replay data. It is a training methodology, not direct evidence that live diagnostics raise
  pass@1 in this repository’s task regime.

The reconciliation is conditional. Static tooling helps when it supplies missing, correct, compact,
actionable information at the phase where the model can use it. It is redundant or harmful when the fact
is already cheap to retrieve, the draft has no detectable error, delivery is noisy, the model does not
elect or cannot repair, or latency/context exceeds saved work.

## 6. Practitioner decision table

| situation | recommended integration | evidence / claim | repository confidence | verify in telemetry |
|---|---|---|---|---|
| Fact is visible, lexically unique, and cheap to read | textual grep plus ranged reads | case-study boundary: [C6](evidence/claim_ledger.md#c6), [C9](evidence/claim_ledger.md#c9); counterfactual boundary of C1 | **Tentative guidance**: dispatch leaks and probe logs are incomplete | localization, bytes/tokens read, success |
| Binding is ambiguous or remote; inheritance, overloads, factories, or re-exports defeat text | first test pushed semantic context, then semantic pull navigation | v2 mechanism: [C15](evidence/claim_ledger.md#c15); ceilinged two-task pilot: [C24](evidence/claim_ledger.md#c24); prior work supports missing-context retrieval | **Mechanism supported; useful agent benefit open** | exact resolution, wrong-file edits, substitution, server errors/latency |
| A compact result can replace a large read | automatic or elected definition/method span | direct repository result: [C1](evidence/claim_ledger.md#c1); live-first transfer: [C4](evidence/claim_ledger.md#c4) | **Supported, narrow** for compact retrieval; pure live-server transfer tentative | substitution rather than semantic-then-read; expected cost/success |
| Forced/automatic context helps but the elective tool is rarely chosen | first try explicit cheap/precise framing | model-dependent election: [C3](evidence/claim_ledger.md#c3); historical framed dispatch: C9 | **Tentative and model-specific** | election, substitution, correctness, added context |
| Framing remains insufficient and the automatic upper bound is valuable | policy training or on-policy relabeling | direct repository policy result: [C2](evidence/claim_ledger.md#c2); C3 caveat | **Supported model-specifically**, not a capability law | election and behavioral retention on held-out opportunities |
| Natural coherent drafts contain checker-detectable errors and the model can revise | patch-boundary diagnostic deltas | calibration boundary: [C16](evidence/claim_ledger.md#c16), [C22](evidence/claim_ledger.md#c22); rejected replay: [C25](evidence/claim_ledger.md#c25); repaired selected replay: [C26](evidence/claim_ledger.md#c26); CoCoGen externally | **Intermediate checker-state effect on two selected workspaces; correctness benefit open** | opportunity, diagnosed-location edits, error deltas, held-out correctness, revision cost |
| Self-repair is unreliable and preventing accepted latent defects matters | acceptance gate | local rejection path unexercised: [C26](evidence/claim_ledger.md#c26); broader claim [C14](evidence/claim_ledger.md#c14) | **Prevention benefit remains external/design guidance** | gate invocation, accepted defects, rejection, false-positive cost |
| Several candidates are available before acceptance | offline checker reranking | CompCoder externally; no direct repository arm | **External evidence only** | clean-correct ranking precision, candidate cost |
| Invalid candidates can be excluded during generation or training | constrained decoding or compiler/type reward | CompCoder, type-constrained generation, and RLCSF externally; no direct repository claim | **External evidence only** | validity, search cost, behavioral correctness |
| A structural operation is risky or non-unique as text | server/AST rename, references, or structured edit | CodeStruct externally; no direct repository claim | **External evidence only** | semantic preservation, rollback/failure rate |

Scope limits: static tooling should not be expected to solve dynamic behavior, logical errors outside the
type system, incorrect annotations, `Any`-heavy boundaries, environment/runtime failures, or errors the
model cannot act on. These are design boundaries, not all directly tested nulls in this repository.

## 7. Reproducibility and limitations

Run `python3 scripts/analysis/reproduce_all.py` from a clean clone to verify hashes, recompute every
reported historical table, run the navigation manipulation gates, and print task-level effects. Model runs
are separate: `scripts/run_navigation_pilot.sh`, `scripts/run_navigation_confirmation.sh`, and
`scripts/run_checker_paired.sh`. Drivers are repository-relative and honor `PYTHON`; Pyrefly is discovered
from `STREAMS_PYREFLY`, `PYREFLY_BIN`, PATH, or a local virtual environment.

Six merged four-seed result files still declare two seeds. The combined rows contain seeds 0–3, but the
source seed-2/3 shards are unavailable. The manifest preserves this warning instead of silently rewriting
historical raw data. Closed-model versions and providers date quickly. Historical server backend/fallback
and version data are incomplete. Synthetic task families provide causal control but limited external
validity. Twelve reserved confirmation instances will still be too few for tight equivalence claims; null language
requires the preregistered margin and an interval within it.

The manifest classifies the exploratory `navigation-v1` model rows as invalidated historical artifacts.
They cannot be regenerated from the repaired sound-contract generator and are never consumed by the
`navigation-v2` confirmation gate. Current mechanical artifacts match the v2 source hashes; old raw rows
remain unchanged.

Every v2 row retains its exact at-run source hashes. The 14B and first 27B artifacts intentionally differ
from current `stream_agent.py` because the observed compact-edit parser incompatibility was repaired before
the tagged `pilot002` rerun; they are preserved as pre-repair calibration. After `pilot002`, inline
serialization was tightened again to repair the checker replay, so the manifest records the exact
`stream_agent.py` mismatch; all navigation cells nevertheless pass and their raw edits remain inspectable.
The checker v1 replay is explicitly rejected, while the v2 selection and paired artifact match current
behavior sources. The stopped three-seed attempt remains labeled incomplete rather than analyzed.

## 8. Conclusion

Language servers help coding agents when they expose correct and unique semantics, compress retrieval,
arrive at a real opportunity, are elected or supplied, and lead to a successful action at lower total
cost. They do not help merely because a server exists or a type system is present.

The supported positive result is compact retrieval against whole-file reads. The historical realistic
retrieval, checker, and execution results identify plausible negative regimes but are ceilings, leaky
tasks, noisy integrations, or small case studies rather than universal nulls. The new experiments make
the missing causal distinctions explicit. The repaired 7B and 14B navigation pilots stop at actionability
floors. Qwen3.6-27B clears the controls, but its two-task matrix is behaviorally ceilinged and automatic
context precedes the same target read; confirmation remains blocked by the non-degeneracy gate. Fresh
checker calibration is rejected at its draft-coherence/opportunity gate. The repaired selected replay shows
a terminal checker-state difference without behavioral gain; its gate rejection path is unexercised.

The operational recipe is: use text for unique local facts; use typed semantic resolution for genuinely
non-lexical ambiguity; retain it only when it replaces work; treat patch-boundary checker cleanliness as an
intermediate signal; and gate only when accepted-defect telemetry demonstrates prevention. Useful
ambiguous-navigation value and checker joint-outcome improvement remain open here. Gate prevention and
offline reranking remain external design guidance rather than demonstrated repository results.
