# When Squigglies Don't Help: Delivery Hygiene and the Limits of Live Type-Checker Feedback for Coding Agents

**Ian Barber**
*ian.barber@gmail.com*

*Preprint. Code, per-rollout data, and a complete chronological lab log are available
in the project repository.*

## Abstract

Human programmers benefit from *live* editor feedback — type errors surface as
squigglies while they type. We test whether the same holds for an LLM coding agent:
does delivering type-checker diagnostics *live*, spliced into the decode stream
mid-generation, improve bug-fixing over standard synchronous tool-result delivery,
or over no diagnostics at all? On a controlled suite of 14 multi-site Python
type-error tasks (12 paired seeds; exact McNemar tests; Qwen2.5-Coder-7B-Instruct
with the Pyrefly checker in a non-blocking stream agent), we find a strong
asymmetry. **No properly-delivered configuration detectably helps:** synchronous
(eager or lazy), hygiene-gated live, and no-feedback conditions yield fix-rates of
0.458–0.530 with no pairwise difference approaching significance (minimum p = 0.12,
n = 168/condition; the design is powered only for effects ≳12 points, so this is an
upper bound on the channel's value, not proven equivalence). **A naively-delivered
live channel, however, significantly hurts** (0.345; p = 0.0002 vs eager-sync,
surviving multiplicity correction). The damage is mechanistically attributable: 78%
of its delivered diagnostics describe the model's *own half-finished edits*, and of
the 25 paired units where no-feedback succeeded and naive-live failed, 23 (92%)
involved such a self-inflicted diagnostic. Two delivery-hygiene repairs — removing
an urgency-framing prompt sentence and gating delivery on parseable file states —
return live delivery to the level of the other arms (naive-vs-gated improvement,
nominal p = 0.041); gated-live vs no-feedback is then exactly balanced. Enriching
diagnostics with go-to-def-style context yields only a small, non-significant nudge,
and rejection-sampled self-distillation fails to teach feedback-use: its gains
appear at least as strongly in a no-feedback control arm (task memorization), which
we trace to a *circularity* — because the channel adds nothing zero-shot, successful
zero-shot trajectories contain no feedback-use to distill. For current tool-loop
agents, an in-loop diagnostic channel is easy to subtract value with and hard to add
value with; delivery hygiene determines which. We release the harness, tasks, and
full audit trail, including two retracted intermediate claims and a confound we
introduced and later isolated.

## 1. Introduction

Language-server protocols (LSPs) changed how humans write code by making feedback
*ambient*: a type error appears as a squiggly underline within milliseconds of being
typed, while the relevant context is still in the programmer's working memory. LLM
coding agents receive feedback differently — through a blocking act–observe loop in
which diagnostics, if surfaced at all, arrive as tool results between actions. As
single-stream "interaction model" designs emerge that can interleave asynchronous
input events directly into a decoding stream [hooper2026speculative; su2026multistream;
gong2025ghostshell], it is natural to ask whether the human experience transfers:
*should* agent feedback be live?

This paper reports a controlled study of that question and returns a cautionary
answer. We hold the information source fixed — diagnostics from the Pyrefly type
checker [pyrefly] over the agent's evolving file — and vary only the *delivery form*:
never (A), batched at the model's next natural pause (C-lazy), immediately after each
edit as a tool result (C-eager, the production norm), or spliced live into the decode
stream while the model continues generating (D, several variants). The agent always
retains a test-execution loop, so the question is about the *marginal* value of the
diagnostic channel in a realistic try-and-correct setting.

Our findings:

1. **No detectable benefit from any proper delivery.** At n = 168 paired units per
   condition, every properly-delivered configuration — eager sync, lazy sync,
   hygiene-gated live — and the no-feedback baseline yield fix-rates of 0.458–0.530,
   with no pairwise exact McNemar test approaching significance (minimum p = 0.12
   across all ten pairs). With ~40–50 discordant pairs per comparison the design's
   minimum detectable effect is roughly 12 points at 80% power, so we claim an upper
   bound on the channel's marginal value, not equivalence. For an untrained 7B agent
   that already has a test loop, the type-checker channel adds no detectable value,
   regardless of timing.
2. **A significant harm mode.** A naive live implementation — diagnostics spliced
   mid-stream after every edit, plus a prompt instructing the model to treat them
   urgently — falls 14–25 points below every other condition (0.345), significantly
   below both sync variants (p = 0.0002, p = 0.006; both survive multiplicity
   correction). The mechanism is measurable: 78.3% of its delivered diagnostics are
   *self-inflicted* (describing the model's own incomplete edit state), and 23 of
   the 25 paired units where no-feedback succeeds and naive-live fails (92%) involve
   one. The agent chases squigglies about code it has not finished writing.
3. **Hygiene recovers; nothing exceeds.** Removing the prompt and gating delivery on
   parseable file states improves the live channel (naive-vs-gated, nominal
   p = 0.041) back to the level of the other arms — gated-live vs no-feedback is then
   exactly balanced (p = 1.0) — while cutting deliveries by 70%. Two attempts to add
   value *beyond* the no-feedback baseline fail: enriching
   diagnostics with definition context (a small, consistently-positive but
   non-significant nudge), and LoRA self-distillation on the agent's own successful
   trajectories, which produces task memorization — exposed by a no-feedback control
   arm — and zero held-out transfer. We identify a circularity that any
   self-distillation approach to feedback-use must break: trajectories that succeed
   without exploiting the channel contain no signal about exploiting it.
4. **An isolation/integration gap.** The same model consumes interleaved feedback
   *perfectly* in isolation: injected facts are used 16/16, injected bug reports
   trigger correct revision 10/10 (vs 3/10 unprompted), and live injection halves
   tokens-to-correct on a toy task. Capability is not the bottleneck; the agentic
   loop's interaction dynamics are.

We also document our process failures — a prompt confound we introduced, an
efficiency claim that did not survive a power increase, and an unpaired-test error —
because each was caught by an explicit adversarial-audit step that materially changed
the paper's conclusions (§7). We believe the audit pattern is as reusable as the
empirical result.

## 2. Related Work

**Interleaved / real-time interaction models.** Su et al. [su2026multistream]
unblock LLMs with parallel input/output streams; Hooper et al.
[hooper2026speculative] interleave clock tokens and asynchronous events into a single
stream for real-time agents; GhostShell [gong2025ghostshell] streams function calls
for robotics; Ginart et al. [ginart2024async] schedule asynchronous tool use. These
works build the *mechanism* for live input; our study measures whether one canonical
application of that mechanism — live diagnostics for coding — actually pays, and
finds the answer hinges on delivery discipline rather than the mechanism itself.

**Execution and analysis feedback for code agents.** RLEF [gehring2024rlef] and
RL-CLS [zhang2025rlcls] use execution/compiler/language-server signals as *training
rewards*; DeepSWE [deepswe2025] scales RL for agentic software engineering. In
contrast we study the *inference-time* channel: whether and how diagnostics should be
shown to a frozen agent. Production agent frameworks increasingly bundle LSP-derived
context [claudecodelsp2025]; our results suggest concrete hygiene requirements for
such integrations.

**Self-correction, distraction, and inference-time repair.** Our mechanism connects
three established lines. First, the self-correction literature: Self-Refine
[madaan2023selfrefine] and Reflexion [shinn2023reflexion] show models revising with
feedback, while Huang et al. [huang2024cannot] show that without *reliable external*
feedback, self-correction loops degrade performance — our agent chasing diagnostics
about its own unfinished edits is a tool-grounded instance of exactly that failure:
the feedback is externally generated but, mid-edit, describes a transient state and
is therefore *unreliable about the task*. Second, the distraction literature: Shi et
al. [shi2023distracted] show LLMs are measurably degraded by irrelevant context;
self-inflicted diagnostics are irrelevant context injected at the worst possible
moment. Third, inference-time repair: Self-Debugging [chen2023selfdebug] establishes
that *test execution* feedback supports iterative repair — consistent with our
finding that the test loop, which all conditions share, carries essentially all the
usable signal. Finally, our circularity finding (§6.2) is a feedback-channel
analogue of known limitations of rejection-sampled bootstrapping such as STaR
[zelikman2022star]: self-generated successful trajectories under-represent precisely
the behaviours that were not needed to succeed.

**Benchmarks and evaluation.** Our tasks are synthetic by necessity: on real
decontaminated SWE-rebench instances [swerebench2025], the 7B agent cannot solve
even oracle-localized bugs, and a type checker is structurally blind to the logic
errors those bugs comprise (Appendix A) — an instance of the broader observation
that benchmark realism and signal isolation trade off [swebenchillusion2025,
swebenchverified]. Bjarnason et al. [bjarnason2026randomness] document the
seed-variance pathologies of agentic evals; our 6→12-seed power-up, which dissolved
an apparent "eager is best" ordering, is a worked example of exactly that hazard,
and motivates our paired-seed McNemar methodology.

## 3. Apparatus

**Model and stack.** Qwen2.5-Coder-7B-Instruct [hui2024qwen25coder], BF16, run
locally on a single NVIDIA DGX Spark [nvidiadgxspark]; Pyrefly as the type checker
[pyrefly]. Sampling at temperature 0.7; the random seed is shared across conditions
so each (task, seed) pair is a matched unit.

**Non-blocking stream agent.** The agent emits one continuous stream containing
free-form reasoning and three actions: line-range edits
(`<edit path lines="A-B">…</edit>`), test runs (`<test/>`), and file reads. Edits
apply immediately (apply-or-reject); the model is *not* forced to yield after an
edit, which is what makes a live channel distinct from a synchronous one. Test
results and other observations return as user-turn messages. Anti-degeneracy guards
(consecutive-failure bail, hard context cap) and their effects are reported in §6.3.

**Conditions.** All conditions share the identical test loop and differ only in the
diagnostic channel:

| | delivery of pyrefly diagnostics |
|---|---|
| **A** | never |
| **C-lazy** | queued; delivered at the model's next natural yield |
| **C-eager** | post-edit hook: delivered immediately after each edit as a tool result |
| **D-naive** | live: spliced into the stream ≈24 tokens after each edit (debounced, pause-aligned), plus a system-prompt sentence announcing live squigglies and urging immediate fixes |
| **D-plain** | D-naive without the announce sentence |
| **D-gate** | D-plain plus a *syntax gate*: deliver only when the current file parses |

**Tasks.** 14 single-file Python debugging tasks built around *multi-site type-error
cascades*: a type-level change (renamed field or method, signature drift, tuple-arity
change, container-type ripple, unguarded Optional) breaks several call sites, so a
test run reveals breakage one site at a time while the type checker sees all sites at
once. Every task is verified to (i) fail its behavioural test and (ii) fire
bug-relevant pyrefly diagnostics at each broken site; one task is a negative control
with no type signal. The set was hardened by three independent adversarial reviews
(confound, realism, difficulty) before any condition data was collected: intent is
specified by tests rather than docstrings, distractor fixes are planted (e.g. a
truthiness guard that drops a legitimate zero), typing is naturalistic, and tasks
whose runtime traceback names the fix are identified so the live channel cannot be
credited for information the test loop already provides. Difficulty was calibrated
empirically into the 0.3–0.6 fix-rate band.

**Measurement.** Fix = full behavioural suite passes at trajectory end. Primary
statistics are exact two-sided McNemar tests on paired (task, seed) outcomes;
efficiency comparisons use only jointly-resolved pairs (matched-pair analysis) to
remove selection bias; intervals are Wilson 95%.

## 4. The model can consume interleaved feedback (isolation results)

Three single-shot probes establish that the *mechanism* is not the bottleneck
(greedy decoding, disjoint seeds from the main study):

- **Forward use.** A needed constant is spliced mid-stream immediately after a
  function signature. The model uses it in the body 16/16 (0/16 without injection),
  and never parrots the injected span.
- **Backward revision.** The model is shown a buggy attempt; a diagnostic about the
  bug is spliced mid-stream. Its continuation emits a corrected function 10/10,
  versus 3/10 spontaneous correction without the diagnostic (+0.70).
- **Toy efficiency.** When the needed fact arrives live versus at a turn boundary,
  the model reaches the same 12/12 correctness in half the output tokens (70 vs 139).

Interleaved consumption is, zero-shot, a solved problem in isolation. Everything that
follows is therefore about *interaction dynamics in the loop*, not about whether the
model can read a splice.

## 5. Main results

### 5.1 No detectable benefit from any proper delivery

Final fix-rates over 14 tasks × 12 seeds (n = 168 paired units per condition):

| condition | fix-rate | Wilson 95% |
|---|---|---|
| C-lazy | 0.530 | [0.45, 0.60] |
| C-eager | 0.524 | [0.45, 0.60] |
| D-gate | 0.482 | [0.41, 0.56] |
| A (none) | 0.482 | [0.41, 0.56] |
| D-plain | 0.458 | [0.38, 0.53] |

No pairwise exact McNemar test among these five arms approaches significance: the
minimum p across all ten pairs is 0.119 (C-lazy vs D-plain; b/c = 31/19), and the
two arms central to the live-vs-none question are exactly balanced — D-gate vs A at
12 discordant pairs each way (p = 1.0), C-eager vs C-lazy at 27/26 (p = 1.0).
Two qualifications keep this honest. *Power:* with ~40–50 discordant pairs per
comparison, the minimum detectable effect at 80% power is roughly 12–15 points;
absence of significance here bounds the channel's value rather than proving
equivalence, and balanced discordant counts are consistent with parity but cannot
certify it. *Multiplicity:* this section reports a family of ten tests (plus the
four harm-mode tests of §5.2); under Holm–Bonferroni correction across the full
family, no within-band difference survives — and neither does any need to, since we
claim none. At 6 seeds, C-eager had led the table at 0.595 and the ordering looked
monotonic; six fresh seeds regressed it to 0.452. We report this dissolved ordering
deliberately: with fewer than ~10 seeds, agentic fix-rates produce convincing-looking
rankings that are pure seed noise [bjarnason2026randomness]. Per-task rates appear in
Appendix C; task-level variance is large (one task swings from 1/12 under D-plain to
12/12 under C-eager), reinforcing the need for paired, multi-seed designs.

### 5.2 The harm mode and its decomposition

D-naive, the configuration closest to a naive reading of "give the agent live
squigglies," fixes 0.345 (n = 84; it was retired from further seeds once the
confound below was identified). Paired against the other arms:

| comparison | b / c | exact p |
|---|---|---|
| D-naive vs C-eager | 5 / 26 | **0.0002** |
| D-naive vs C-lazy | 9 / 26 | **0.006** |
| D-naive vs A | 14 / 25 | 0.108 |
| **D-naive vs D-gate (hygiene)** | 11 / 24 | **0.041** |

Note the nuances. Naive live delivery is significantly worse than *batched* delivery
but not, at this n, significantly worse than *no* feedback. Under Holm–Bonferroni
correction across the paper's full test family, the harm results survive comfortably
(adjusted p < 0.005); the hygiene comparison below (p = 0.041) does **not** survive
correction and is reported as nominal. The deficit decomposes into two independently
fixable causes:

1. **The announce prompt (−0.107).** D-naive uniquely carried one system-prompt
   sentence describing the live channel and urging immediate fixes. Removing it
   (D-plain) recovers 0.345 → 0.452. This sentence was a confound we introduced and
   initially failed to control; §7 describes how it was caught.
2. **Self-inflicted diagnostics (−0.048 further).** Adding a parseability gate to
   D-plain (yielding D-gate) recovers 0.452 → 0.500 while cutting deliveries from
   240 (D-plain) to 71 (D-gate), −70%. The suppressed deliveries were squigglies
   about code the model was mid-way through writing. (D-naive's own delivery count,
   244, is reported in §5.3.)

The combined repair — D-naive vs D-gate — is the paper's hygiene effect (b/c =
11/24, nominal p = 0.041, uncorrected). It certifies that the two delivery repairs
*improve* the live channel; the separate, balanced D-gate-vs-A comparison (p = 1.0)
locates the repaired channel at the level of no-feedback, not above it.

### 5.3 Mechanism

Three quantitative observations locate the harm:

- **78.3%** (191/244) of D-naive's delivered diagnostics reference self-inflicted
  states — parse errors and unbound names produced by the model's own incomplete
  edit sequences — rather than the task's bug.
- Of the 25 paired units where A solved and D-naive failed, **23 (92%)** had at least
  one self-inflicted diagnostic delivered.
- Alternative explanations fail quantitatively: D-naive triggers the
  consecutive-failure bail *less* than C conditions (0.107 vs 0.167); only 7.3% of
  its failures are output-budget-bound (the budget-bound condition is C-eager, at
  50% of failures); and input-context inflation is a consequence of failing
  trajectories, not their cause.

The qualitative picture is an interruption loop: the model begins a multi-part edit;
the checker truthfully reports the half-finished state as broken; delivered live,
the report redirects the model into "fixing" code it had not finished writing;
the detour spawns new transient states and new squigglies. Humans survive ambient
feedback because they possess a settled-state intuition — you do not chase squigglies
mid-keystroke. The syntax gate is a crude mechanical surrogate for that intuition,
and it removes most of the harm.

### 5.4 An efficiency claim, retracted

At intermediate scale (6 matched pairs) the gated live condition appeared to solve
jointly-solved tasks in roughly half the test round-trips of sync (1.7 vs 3.2). At 20
matched pairs the effect is exactly null (2.50 vs 2.50; token ratio 1.00 ± noise).
We retract the intermediate claim and note the lesson: matched-pair subsets of
agentic runs are small and seductive, and efficiency claims require the same power
discipline as accuracy claims.

## 6. Can anything add value to the channel?

### 6.1 Richer signal content

We enriched every delivered diagnostic with go-to-def/hover-style context: the
current signature or class definition of each symbol the diagnostic names. On the
live arm the effect is small and uniformly positive (0.524 vs 0.500; the only two
discordant pairs both favour enrichment; p = 0.5); on eager-sync it is null
(0.607 vs 0.595, p = 1.0). Task-level texture is informative: rename- and key-type
tasks reach ceiling under enrichment — definition context helps exactly where the
question is "what does this symbol look like *now*" — but the aggregate band holds.
Content is not the binding constraint.

### 6.2 Self-distillation, and the circularity it exposes

We harvested the agent's own *resolved* trajectories on a 7-task training split
(disjoint seeds, deployment configuration), masked all observation tokens so that
loss falls only on the model's actions, and LoRA-tuned [hu2022lora] on 62
demonstrations. Evaluation on the 7 held-out tasks, with a no-feedback control arm:

| arm | held-out | train-tasks |
|---|---|---|
| D-gate, base | 0.524 | 0.476 |
| D-gate, +SFT | 0.524 | 0.667 |
| A, base | 0.619 | 0.333 |
| A, +SFT | 0.500 | 0.595 |

Held-out feedback-use gain is exactly zero (paired p = 1.0). Train-task gains are
substantial — and appear *more strongly in the no-feedback arm* (+0.26, p = 0.007):
the adapter memorized the training tasks' fixes rather than learning to exploit the
channel. The diagnosis is visible in the data itself: harvested demonstrations
average only 364 trained tokens — short, clean solves in which the diagnostic channel
was barely exercised. **Because the channel adds nothing zero-shot, trajectories that
succeed zero-shot contain almost no feedback-use to imitate.** Rejection-sampled
self-distillation therefore selects for easy solves, not channel exploitation — a
circularity that any bootstrap of feedback-use must break, e.g. with demonstrations
constructed so the diagnostic is load-bearing (the isolation probes of §4 sketch the
form) or with RL against feedback-dependent rewards [zhang2025rlcls; gehring2024rlef].

## 7. Methodology: adversarial self-audit as a reusable pattern

We propose a secondary, methodological contribution: before believing our own
intermediate headline ("live feedback hurts"), we ran a structured adversarial audit
— four independent analyses (paired statistics, jackknife robustness, confound
hunting, control validation) followed by a synthesis — against our own conclusions.
Every load-bearing number in this paper was changed or created by that audit:

1. It replaced unpaired two-proportion tests with paired McNemar, which
   *strengthened* the live-vs-batched result (p = 0.0002) while *weakening*
   live-vs-nothing to non-significance — a sign-relevant correction.
2. It identified the announce-prompt confound (§5.2) as the one unresolved threat to
   the then-headline claim, and specified the announce-off arm that subsequently
   halved the measured effect.
3. It proposed the parseability-gate experiment that converted "live feedback is
   harmful" into "self-inflicted feedback is harmful" — the paper's mechanism.
4. Its jackknife established that the D-naive deficit survives any single-task
   deletion (14/14) and concentrates, as the mechanism predicts, on tasks whose
   runtime tracebacks already name the fix and on single-site tasks.

The pattern — independent statistician, robustness analyst, confound adversary, and
control auditor, each given the raw data and an explicit brief to break the claim —
is cheap relative to the experiments it protects and, in this study, was the
difference between publishing a wrong claim and a right one. The full audit memo is
preserved verbatim in the lab log. (Two engineering defects whose correction also
shifted early numbers — an edit-parser bug that silently dropped single-line edits,
and unbounded input-context growth in degenerate trajectories, later capped — are
documented in Appendix B.)

## 8. Practical guidance

For builders attaching linters/type checkers to coding agents:

- **Default to post-edit synchronous delivery.** It is simple, standard, and in our
  data never worse than any alternative.
- **If you build a live channel:** debounce; deliver at a natural pause; **never
  deliver diagnostics about a state the model is mid-way through writing** (gate on
  parseability or equivalent); deduplicate against already-delivered content; and
  avoid urgency-framing instructions, which measurably backfire.
- **Verify your checker carries signal for your bug distribution** before investing
  in delivery: on real-repo logic bugs our type checker produced zero bug-relevant
  diagnostics (Appendix A), and no delivery design can rescue an empty channel.
- **Instrument the channel.** The single most diagnostic number in this study was
  the fraction of delivered diagnostics that were self-inflicted (78%).

## 9. Limitations

One model (7B-instruct), one checker, one language, single-file synthetic tasks, and
n = 168 paired units per condition: parity claims are bounded by this power, and
small true effects (±3–4 points) would be invisible. The agent's actions are short;
the strongest case for live delivery — feedback arriving during long uninterruptible
generations — is structurally under-tested here. The SFT result is specific to
rejection-sampled self-distillation at small scale (62 demonstrations); it bounds
that recipe, not training in general. Real-repo validation awaits either stronger
base agents or bug distributions where type-level signal aligns with the failure
(Appendix A documents why both were binding here). Finally, all conditions retain a
test-execution loop; in settings where running tests is expensive or impossible, the
diagnostic channel's marginal value may be very different.

## 10. Conclusion

In a controlled agentic setting where a type checker demonstrably sees every planted
bug, we could not make its feedback *help* an untrained 7B coding agent — not with
synchronous delivery, not with hygiene-gated live delivery, not with richer content,
and not with self-distilled training. We could, however, easily make it *hurt*: the
naive transplant of the human "live squigglies" experience costs 14–25 points of
fix-rate, for a measurable reason (the agent chases feedback about its own unfinished
work) with a cheap mechanical fix. The asymmetry — delivery can subtract but
struggles to add — together with the isolation/integration gap of §4 suggests that
the value of ambient feedback for humans rests on interaction skills (settled-state
judgment, attention gating) that current agents lack and that successful zero-shot
trajectories cannot teach. Making feedback channels *load-bearing* during training,
rather than ambient during inference, is the open problem this study motivates.

## Appendix A. Real-repository groundwork

We built a native (no-container) pipeline for decontaminated post-2025 SWE-rebench
instances [swerebench2025] (clone, per-task venv, editable install, test-patch
application; aarch64): 25 single-file candidates selected, 8 provisioned, 4 fully
well-formed (failing target test, passing complement). The 7B agent, even given
oracle localization to the buggy function, resolved 0/3 attempted instances and
produced no bug-relevant type diagnostics: the bugs are logic-level, and pyrefly's
output on these repositories consisted of unused imports, missing stubs, and the
agent's own transient parse errors. Both observations — agent capability floor and
checker/bug misalignment — motivated the controlled synthetic suite. The pipeline is
retained in the repository for future work with stronger models.

## Appendix B. Reproducibility and naming

All per-rollout records (including full event traces and delivered-diagnostic texts),
the task suite and its verifier, the agent harness with every delivery condition as
a flag, the harvest/train/eval SFT pipeline, and a chronological lab log containing
every intermediate result, retraction, and audit memo are in the repository. Key
artifacts: `scaffold/stream_agent.py`, `scripts/synth_tasks.py`,
`scripts/synth_acd.py`, `runs/agent/*.json`, `log.md`, `WRITEUP.md`.

**Naming note.** The repository's logs and result files predate this paper's
terminology: the condition called **D-naive** here appears as `D-tuned` in
`log.md`/`synth_power.json` (it was, at the time, the debounce-"tuned" variant —
before the announce confound and gating results reframed it as the naive baseline).
`D-plain` and `D-gate` match their file names.

**Engineering errata.** Two harness defects were found and fixed during the study,
each documented with before/after numbers in the log: (i) the line-edit parser
initially required `lines="A-B"` and silently dropped the model's frequent
single-line `lines="N"` edits, depressing several early fix-rates (all reported
numbers post-date the fix); (ii) degenerate trajectories could grow input context
without bound via repeated file-view re-feeds (observed at 58k tokens against a 32k
context); a hard 24k cap now converts these to recorded bails (4 rollouts in the
final data).

## Appendix C. Per-task fix counts

Resolved counts per task (out of 12 seeds; D-naive out of 6). Tasks marked † are the
negative control (no type signal) and the calibration outlier (no condition solves
it).

| task | A | C-lazy | C-eager | D-plain | D-gate | D-naive |
|---|---|---|---|---|---|---|
| grid_field_rename | 2 | 5 | 3 | 2 | 3 | 1 |
| fmt_signature_drift | 4 | 5 | 3 | 5 | 3 | 0 |
| records_arity_drift † | 0 | 0 | 0 | 0 | 0 | 1 |
| lookup_optional_cascade | 6 | 7 | 7 | 8 | 6 | 2 |
| config_truthiness_distractor † | 3 | 6 | 1 | 2 | 2 | 0 |
| parse_branch_ripple | 4 | 3 | 4 | 4 | 2 | 0 |
| return_container_ripple | 7 | 7 | 6 | 7 | 9 | 3 |
| method_rename_cascade | 10 | 7 | 11 | 9 | 10 | 4 |
| dict_key_type_drift | 10 | 9 | 10 | 10 | 11 | 3 |
| ctor_param_added | 7 | 8 | 8 | 8 | 6 | 2 |
| renamed_return_key | 7 | 8 | 12 | 1 | 8 | 3 |
| optional_two_helpers | 7 | 8 | 8 | 5 | 6 | 4 |
| tuple_return_widened | 3 | 4 | 5 | 4 | 5 | 1 |
| mutable_default_none | 11 | 12 | 10 | 12 | 10 | 5 |

Task-level variance is substantial (e.g. `renamed_return_key`: 1/12 under D-plain vs
12/12 under C-eager), underscoring why unpaired or few-seed comparisons in this
regime are untrustworthy and why all headline statistics are paired across
(task, seed) units.

## References

See `bibliography.md` for full BibTeX. Citation keys used above:
[su2026multistream] [hooper2026speculative] [gong2025ghostshell] [ginart2024async]
[gehring2024rlef] [zhang2025rlcls] [deepswe2025] [swerebench2025]
[swebenchverified] [swebenchillusion2025] [bjarnason2026randomness]
[claudecodelsp2025] [pyrefly] [hui2024qwen25coder] [hu2022lora] [nvidiadgxspark]
[madaan2023selfrefine] [shinn2023reflexion] [huang2024cannot] [shi2023distracted]
[chen2023selfdebug] [zelikman2022star].
