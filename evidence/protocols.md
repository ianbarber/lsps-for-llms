# Frozen protocols and run budget

## Experiment 0: compact retrieval substitution

Protocol `retrieval-paired-v1` retains the eleven `effic_real2` tasks and holds the model, prompt,
line-edit interface, visible tests, seed, and budgets fixed. The `whole` arm exposes whole-file reads;
the `text` arm exposes grep, ranged reads, and a whole-file fallback; the `definition` arm adds compact
static-AST definition lookup while retaining every text fallback. The primary contrast is text versus
definition, so the treatment estimates the incremental value of compact lookup for an agent that already
has an efficient shell-like retrieval interface. The whole arm is a historical-counterfactual calibration,
not the main practitioner baseline.

Before model execution, every task must fail in its base state, pass with the gold target, localize through
text search, resolve an exact AST definition, have a valid ranged-read span, and return a definition at
least five times smaller than its defining file. Outcomes include held-out success, unconditional total
tokens, grep/ranged/whole/definition calls, retrieval-response characters, and whether a definition is
followed by a read of the defining file. Temperature-zero cells use one deterministic rollout per task.
Ratios use task-weighted means and paired task bootstraps over matched-success tasks. Disjoint frozen
result shards may be combined only when their model, revision, budgets, temperature, and seed configuration
match and no `(task, arm, seed)` cell overlaps.

The executed local Qwen3.5-27B run passes both main arms 11/11. Mean total tokens are 1,602 for text and
1,235 for definition, a text/definition ratio of 1.297 (task-bootstrap 95% CI 1.093–1.527); definition is
cheaper on 10/11 tasks and is followed by zero defining-file rereads. This is a controlled-suite estimate,
not a live-LSP or natural-repository prevalence claim.

## Experiment 1: types x semantic navigation

Protocol version `navigation-v2` uses four excluded pilot instances over two development templates, twelve
additional apparatus-audit instances over three templates, and twelve reserved confirmation instances over
three new templates, with deterministic disjoint seeds in `scripts/experiments/navigation_tasks.py`.
The apparatus set is explicitly not confirmation because it was inspected while validator behavior was
still being repaired. Identifiers are neutral;
each task has 8–15 overrides, factory/registry construction, identical runtime `.py` files, tests, and gold
patches between variants, and a one-line gold fix. Only `factory.pyi` differs. The typed stub has a sound
`Literal` overload for every registry key plus a base-returning `str` fallback; the erased stub exposes only
the base return. The visible call retains the selected key's `Literal` precision.

The core cells are typed/textual, typed/automatic semantic result, erased/textual, and erased/automatic
semantic result. Typed deployment cells are textual, neutral elective semantic tool, cheap/precise framed
tool, and automatic semantic result. `semantic_span_control` supplies the same pristine buggy span with the
same neutral framing and budget, but from frozen task metadata; it is an oracle localization/actionability
control outside the factorial estimate. `positive_control` supplies the corrected definition and tests
copy/edit competence. Both controls must pass 2/2 selected pilot instances before a navigation null is
interpretable. Automatic results are composed:
strict Pyrefly LSP definition resolution plus the enclosing method span. There is no AST fallback.

Mechanical exclusions are fixed before model runs: reject a family if base does not fail, gold does not
pass, runtime outputs differ across variants, gold differs, override count is outside 8–15, typed goto
misses the gold override, erased goto is discriminating, the rendered prompt leaks the concrete class or
gold path, any overload disagrees with runtime construction, either variant has type diagnostics, widening
the typed call to `str` remains discriminating, the buggy-span payload contains the gold replacement, or a
control cannot be acted on. Pilot and apparatus-audit instances may be revised;
confirmation families may not. Confirmation is one shot after prompts, generator, metrics, exclusions,
and source hashes are frozen.

Primary outcome is task-level held-out pass@1. Other preregistered outcomes are correct-file localization
before first edit, wrong-file edits, input/output tokens, calls, reads, turns, wall time, server latency,
semantic election, semantic-then-read behavior, and expected tokens per successful task. Seeds are nested
within task. Report task distributions, means and medians, and task bootstrap intervals. The token-ratio
estimand is the treatment/control ratio of task-weighted mean unconditional tokens; paired task bootstrap
recomputes that ratio. Per-task ratios are descriptive only. Equivalence margins, if a null is discussed,
are +/-10 percentage points for pass@1 and a token ratio of 0.90–1.10;
an interval crossing those bounds is inconclusive rather than equivalent.

## Experiment 2: checker opportunity and integration

Protocol version `checker-paired-v1` first generates a natural draft with no checker access, stores every
workspace file and its hash, then forks control, one-shot coherent-patch diagnostics, acceptance gate, and
a low-level-edit-triggered pushed-feedback baseline from those exact bytes. Coherence means the target parses
and contains no remaining explicit `raise NotImplementedError`. Revision efficacy is conditional on a
coherent submitted draft. A separate end-to-end estimand retains failed, unsubmitted, and incoherent drafts
as pipeline failures; they are never silently removed from the denominator.

Diagnostics are target-scoped deltas from the pristine workspace, deduplicated, uncapped in raw JSON, and
presented with path, range, code, concise message, classification, and code frame. Tests execute outside
the checked tree. Opportunity means at least one new semantic diagnostic on the frozen natural draft;
syntax/partial-edit failures are separate. The model regime must put 20–70% of coherent development drafts
in this opportunity set while retaining viable edit behavior. If none does, report that capability gap.

Control and treated revisions share the same token, turn, read, and test budgets. Outcomes separately
include coherent-draft and opportunity-conditioned revision efficacy, plus end-to-end yields across every
natural draft with pre-revision failures counted as failures. Other measures are diagnostics eliminated/retained/introduced,
edits overlapping diagnosed locations, type-clean accepted patches, gate rejection/abstention, tokens,
turns, latency, and expected cost per accepted correct patch. The latter is task-weighted mean total
draft-plus-revision tokens divided by the accepted-and-held-out-correct rate; rejected, abstained, and wrong
attempts remain in the cost numerator. Report the stricter type-clean accepted-correct cost separately. A
+/-10 percentage-point margin applies to
pass@1 or accepted-defect null claims.

Because fresh 7B/14B development calibration did not enter the opportunity band, a separate
**opportunity-conditioned case series** answers the narrower practitioner question: what happens when a
coherent frozen draft already has a checker-detectable semantic error? Selection occurs before treatment
using only the committed frozen workspace, coherence, and target-delta diagnostic state. The source
artifact is not edited. A derived selection record preserves its hash, workspace hashes, selection rule,
and a mechanical gold-repair check. Two exact recovered historical 7B workspaces qualify. A stronger
pinned local model receives byte-identical workspaces under control, one-shot pre-revision diagnostics,
and acceptance-gate arms with common seeds. The primary product outcome is
`accepted && type_clean && held_pass`; accepted semantic/behavioral defects, gate rejection, diagnostic
changes, diagnosed-location edits, revision tokens, and latency are secondary outcomes. This selected
case series estimates conditional revision efficacy only. It cannot estimate natural opportunity
prevalence, same-model end-to-end value, or a population effect, and historical draft-generation token
cost is unavailable.

Protocol `checker-paired-v2` keeps missing draft cost as null and records completion attempts, gate checks,
rejections, and acceptances as separate events. Inline edit bodies are normalized only when their
indentation equals the replaced line or differs by one unambiguous separator space; other inline indentation
is rejected without mutation. Each row stores raw/applied edit-body hashes and final file/workspace hashes.

Protocol `checker-paired-v3` closes the null-prefix pairing hole. Control and gate must have identical
generated prefixes through the first completion attempt. If neither completes, their full generated
trajectory hashes must match because the gate cannot yet affect behavior; one-sided completion is also a
failure. The complete JSON is staged beside the destination, validated in memory, flushed, and atomically
renamed. Validation failure removes the staged file and never creates the advertised result path.

Protocol `checker-paired-v4` removes inline line-edit serialization from paired revisions. Revisions use
an exact SEARCH/REPLACE/END block whose search text must match the live file verbatim; draft generation
continues to use line edits. Diagnosed-location repair is recovered from the exact frozen/final file diff.
The v3 completion-boundary and atomic-publication rules remain unchanged. This transport change was made
after two local v3 hidden-defect attempts were atomically rejected for ambiguous inline serialization;
neither rejected attempt is a result artifact.

The local v4 development attempt was stopped after control and diagnostic cells both exhausted the budget,
made three edits, never submitted, and retained the defect. Protocol `checker-paired-v5` therefore restores
the actionable numbered line-edit interface but advertises only a multiline edit block with a required
newline immediately after the opening tag. That path bypasses inline indentation normalization completely;
any inline deviation remains a recorded serialization failure and atomically rejects the grid. V3's
completion pairing, event accounting, and atomic publication remain in force.

A post-run action-origin audit invalidates v5's completion interpretation. The model emits `<test/>` at
token 3; the passing-test user observation contains a literal `<done/>`; and the first `done_attempt` fires
at token 4 while the new assistant turn contains only `<think>`. Protocol `checker-paired-v6` makes tool
observations an explicit parser boundary by advancing every action cursor beyond delivered user text. Every
completion event records `source=model`, and selected-case publication rejects an uncertified completion.
A successful edit invalidates any earlier passing-test state. After gate rejection, the agent must repair,
run a fresh visible test, and emit a new model-generated `<done/>`; acceptance without that resubmission is
impossible.

`checker-gate-v2` freezes three of the preselected hidden-defect workspaces and pairs each with its exact
validated gold counterpart. Defects are coherent, visible-passing, held-out-failing, and add exactly one
target-scoped semantic diagnostic. Clean controls pass visible and held-out behavior with no diagnostic
delta. Control and gate use identical prompts and deterministic seeds through the first model completion.
Primary defect outcomes are reached bad completions, rejection sensitivity, diagnosed-location repair,
fresh test, resubmission, and accepted type-clean held-out correctness. Primary clean outcomes are false
rejection and first-submission acceptance. Selection is controlled and cannot estimate natural prevalence.

## Run and spend budget

No paid API run is authorized by this protocol. The executed development regimes use locally cached
Qwen2.5-Coder 7B/14B, Qwen3.5-27B, and Qwen3.6-27B models. The pilot is two navigation instances across a gold-copy control, a buggy-span
actionability control, four core cells, and two incremental deployment cells (typed baseline/automatic are
shared), followed by three checker development tasks. Monetary cap is **$0**. The frozen navigation
confirmation is twelve instances across three templates x six unique cells x three nested seeds. It was
never run for navigation: the uniform-ceiling gate in `scripts/run_navigation_confirmation.sh` blocks it
permanently, because every core row of the accepted pilot passes held-out. Those twelve reserved instances
were subsequently spent, once, as the pre-registered confirmation of the substitution-training result
(ledger C35). Any OpenRouter or other paid confirmation requires a separate
model/cell/cost proposal. The v6 controlled extension adds three defect/clean pairs across control and gate
(12 local-model cells) under the same $0 monetary cap.

## Execution status

The historical `navigation-v1` 7B run is invalid because its factory contract is unsound. In repaired v2,
7B fails buggy-span actionability 0/2 and 14B reaches 1/2, so both stop. After a tested compact-edit parser
repair, pinned Qwen3.6-27B clears gold-copy and buggy-span controls 2/2 and runs the matrix. All 12 cells pass;
the typed token ratio falls within the margin on these two pilot tasks only, erased automatic context adds
cost, automatic payloads always precede target reads, and composed resolution adds about six seconds per
task. Confirmation remains blocked by the uniform-ceiling gate; the reserved split it would have used was spent
on the substitution confirmation instead (C35).
Fresh checker calibration still misses the opportunity band. The separately selected two-task checker-
positive case-series replay is invalid for cleanup claims because ambiguous inline edit serialization creates
the observed syntax-state difference. All arms remain 1/2 jointly. The unresolved gate trajectory never
invokes the gate, so no rejection or prevention evidence exists. Protocol `checker-paired-v2` makes inline
serialization explicit relative to current indentation, records actual done/gate events separately, and
preserves missing draft cost as null. Its immutable rerun has no serialization failures. Diagnostics end
type-clean on 2/2 selected workspaces versus 1/2 control but add 217 mean revision tokens; every arm remains
1/2 on held pass and joint accepted-clean-correct. The gate accepts the already-clean task and descriptively
records zero rejections, but the unresolved control and gate trajectories diverge before any completion
attempt. The v2 null-prefix validator misses that divergence, so the gate contrast is excluded.

The controlled `checker-hidden-v1` generator mechanically validates twelve coherent, visible-passing,
held-out-failing workspaces with exactly one semantic diagnostic and a clean gold repair. Two v3 model
attempts are atomically rejected for ambiguous inline serialization, and the v4 search-edit development
attempt is stopped for actionability before publication. V5 runs three preselected cases, but its first
completion is triggered by literal action text in the passing-test observation; model-submission and
one-shot-effect claims are therefore invalidated.

V6 runs three frozen defect/clean pairs with pinned local Qwen3.5-27B. All six control/gate first-completion
prefixes are identical and model-generated. One seeded defect self-repairs in both arms before completion.
Control accepts the other two bad completions; the gate rejects both, after which each trajectory makes one
diagnosed-location edit, retests, resubmits with a second model-generated `<done/>`, and is accepted clean
and held-out-correct. Gate accepted-clean-correct is 3/3 versus 1/3 control on defects. Every matched clean
draft is accepted on the first gate check, for 0/3 false rejections. Mean revision tokens are 1,211 gate
versus 838 control on defects and 643 in both arms on clean controls. This supports controlled conditional
recovery, not natural opportunity prevalence, population rejection precision, or deployment value. No paid
API calls were made.
