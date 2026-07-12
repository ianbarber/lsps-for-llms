# Frozen protocols and run budget

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

## Run and spend budget

No paid API run is authorized by this protocol. The executed development regimes use locally cached
Qwen2.5-Coder 7B/14B models. The pilot is two navigation instances across a gold-copy control, a buggy-span
actionability control, four core cells, and two incremental deployment cells (typed baseline/automatic are
shared), followed by three checker development tasks. Monetary cap is **$0**. The frozen navigation
confirmation is twelve instances across three templates x six unique cells x three nested seeds; it remains
unrun until the pilot clears every gate. Any OpenRouter or other paid confirmation requires a separate
model/cell/cost proposal.

## Execution status

The historical `navigation-v1` 7B run is invalid because its factory contract is unsound. In repaired v2,
7B fails buggy-span actionability 0/2 and 14B reaches 1/2, so both stop. After a tested compact-edit parser
repair, pinned Qwen3.6-27B clears gold-copy and buggy-span controls 2/2 and runs the matrix. All 12 cells pass;
the typed token ratio falls within the margin on these two pilot tasks only, erased automatic context adds
cost, automatic payloads always precede target reads, and composed resolution adds about six seconds per
task. Confirmation remains blocked by the uniform-ceiling gate.
Fresh checker calibration still misses the opportunity band. The separately selected two-task checker-
positive case-series replay is invalid for cleanup claims because ambiguous inline edit serialization creates
the observed syntax-state difference. All arms remain 1/2 jointly. The unresolved gate trajectory never
invokes the gate, so no rejection or prevention evidence exists. Protocol `checker-paired-v2` makes inline
serialization explicit relative to current indentation, records actual done/gate events separately, and
preserves missing draft cost as null. Its immutable rerun has no serialization failures. Diagnostics end
type-clean on 2/2 selected workspaces versus 1/2 control but add 217 mean revision tokens; every arm remains
1/2 on held pass and joint accepted-clean-correct. The gate accepts the already-clean task, is never invoked
on the unresolved task, and records zero rejections. No paid API calls were made.
