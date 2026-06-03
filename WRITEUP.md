# Streams: How (and whether) to deliver live LSP feedback to a coding agent

*Draft write-up — 2026-06-03. Status: n=84/condition complete and audited; n=168
power-up in flight (see §9). All numbers reproducible from `runs/agent/*.json` via the
analysis snippets in `log.md`.*

## TL;DR

We set out to test an attractive intuition: human programmers benefit from *live* IDE
feedback (squigglies appearing as you type), so a coding LLM agent should too — feedback
spliced mid-generation into its stream should beat the standard synchronous
tool-result delivery. **The intuition is wrong as stated, but the reason is useful.**
On a controlled 14-task × 6-seed agentic benchmark (Qwen2.5-Coder-7B, real
[pyrefly](https://pyrefly.org) type-checker, paired seeds):

1. **A naive live implementation tanks the agent** — fix-rate 0.345 vs 0.548 for
   batched sync delivery (paired McNemar p=0.006), the largest effect we measured.
2. **The harm is delivery hygiene, not the live channel itself.** It decomposes into
   (a) an over-eager instruction prompt and (b) *self-inflicted squiggles*: **78% of
   delivered live diagnostics were about the model's own half-finished edits**, and 92%
   of the cases where no-feedback beat live involved one. Removing the prompt and
   gating diagnostics on "the file currently parses" recovers fix-rate to 0.500 —
   statistically indistinguishable from sync (p=0.56) and from no feedback (p=0.79).
   Naive-live vs cleaned-live is significant (p=0.041).
3. **Once delivered properly, the channel adds little for an untrained model:**
   every well-delivered condition lands in a narrow parity band (≈0.48–0.55). *How* you
   deliver feedback matters more than *whether* you deliver it.
4. The model is fully *capable* of consuming interleaved feedback — in isolation it
   uses mid-stream injections perfectly (16/16 forward use, +0.70 revision lift,
   half-the-tokens efficiency on a toy). The gap between "capable in isolation" and
   "neutral in the loop" is the interesting object: untrained agents don't exploit
   live feedback, and badly delivered live feedback actively derails them.

**Practical recipe:** if you attach a linter/LSP to a coding agent loop, deliver
results synchronously after each edit (a post-edit hook is fine and simplest), or, if
you want a live channel, debounce it, deliver at a natural pause, **never deliver
diagnostics about code the model is mid-way through writing** (gate on parseability),
and don't add aggressive "fix it now" instructions.

---

## 1. Motivation

LSPs transformed human programming by making feedback *ambient*: errors surface while
you type, not after you compile. Coding agents instead get feedback through a
synchronous tool loop — act, wait, observe. We asked whether the *delivery form*
matters at matched information content: does live, mid-generation delivery (the
"squigglies" experience) improve an agent's in-flight decision-making relative to the
same diagnostics delivered synchronously at action boundaries, or not at all?

This is also a question about a capability frontier: single-stream "interaction
models" that interleave input events into the decode stream (cf. clock-token and
interleaved-interaction work) are an emerging design; coding+LSP is a natural,
measurable instantiation.

## 2. Setup

**Model.** Qwen2.5-Coder-7B-Instruct, BF16, greedy/temp-0.7 sampling, run locally
(single GB10/128GB unified). One model throughout; no fine-tuning in any result below.

**Agent harness.** A custom *non-blocking* continuous-stream agent
(`scaffold/stream_agent.py`): the model emits reasoning, line-range edits
(`<edit path lines="A-B">`), `<test/>` and `<read/>` actions in one stream; the
harness applies edits apply-or-reject, runs the real test suite on `<test/>`, and
feeds observations back as user-turn messages. Diagnostics come from the real pyrefly
type-checker on the current file state. Crucially, the model is *not* forced to yield
after an edit — it can keep streaming, which is what makes a live channel meaningful.

**Conditions** (all see identical test feedback; only the *diagnostic channel* differs):

| condition | diagnostic delivery |
|---|---|
| **A** | never |
| **C-lazy** | queued; delivered at the model's next natural yield (turn boundary) |
| **C-eager** | post-edit hook: delivered immediately after each edit as a tool result |
| **D-naive** ("D-tuned" in logs) | live: spliced mid-stream ~24 tokens after each edit (debounced, pause-aligned) + a system-prompt sentence announcing live squigglies |
| **D-plain** | as D-naive, without the announce sentence |
| **D-gate** | as D-plain, plus a *syntax gate*: deliver only when the file currently parses |

**Tasks.** 14 synthetic Python debugging tasks (`scripts/synth_tasks.py`) built around
*multi-site type-error cascades* (renamed fields/methods, signature drift, tuple-arity
changes, unguarded Optionals, container-type ripples): a type-level change breaks
several call sites; pytest reveals them one at a time, the type-checker reveals all at
once. Each task verified to (a) fail behaviourally and (b) fire guiding pyrefly
diagnostics at every broken site; one task is a **negative control** with no type
signal. The set was hardened by adversarial review (three independent critics) against
known confounds: intent lives in tests rather than giveaway docstrings,
plausible-but-wrong distractor fixes, natural partial typing, one bug-class per task.
Difficulty was calibrated empirically to the ~0.3–0.6 resolve band.

Why synthetic? We first provisioned real, decontaminated SWE tasks (SWE-rebench,
post-2025, runs natively on our box — see Appendix), but found two blockers: the 7B
cannot solve real repo bugs even with oracle localization (0/3), and **pyrefly is
structurally blind to real-world logic bugs** — its diagnostics on those tasks were
noise (unused imports, missing stubs, the model's own parse errors), never about the
bug. The LSP-delivery question requires tasks where the LSP actually carries signal.

**Measurement.** Resolve = full test suite passes. 6 seeds per (task, condition) at
temp 0.7; the (task, seed) pair is a paired unit across conditions (same RNG).
Statistics are exact paired McNemar tests; efficiency comparisons use only
jointly-resolved (task, seed) pairs to kill selection bias. Wilson 95% CIs.

## 3. Result 1 — interleaved feedback is perfectly consumable in isolation

Before the agent loop, we tested the *mechanism* in controlled single-shot settings:

- **Forward use:** inject a needed fact mid-stream (`‹info›...‹/info›` after the
  function signature); the model uses it **16/16** (0/16 without injection; zero
  parroting of the injected span).
- **Backward revision:** show a buggy attempt, inject a `‹diag›` about it; the model's
  continuation emits a corrected version **10/10** vs 3/10 unprompted (+0.70).
- **Toy efficiency:** when the needed fact arrives live vs at a turn boundary, live
  reaches the same 12/12 correctness in **half the tokens** (70 vs 139).

So a chat-tuned coder *can* read and react to mid-stream splices zero-shot. Whatever
happens in the loop is not a basic consumption failure.

## 4. Result 2 — in the loop, delivery hygiene dominates

Fix-rates, n = 84 paired units per condition (14 tasks × 6 seeds):

| condition | resolve | Wilson 95% |
|---|---|---|
| C-eager | 0.595 | [0.49, 0.69] |
| C-lazy | 0.548 | [0.44, 0.65] |
| **D-gate** | **0.500** | [0.40, 0.60] |
| A (none) | 0.476 | [0.37, 0.58] |
| D-plain | 0.452 | [0.35, 0.56] |
| **D-naive** | **0.345** | [0.25, 0.45] |

Paired McNemar (exact, two-sided), the comparisons that matter:

| comparison | b / c | p |
|---|---|---|
| D-naive vs C-eager | 5 / 26 | **0.0002** |
| D-naive vs C-lazy | 9 / 26 | **0.006** |
| D-naive vs A | 14 / 25 | 0.108 (n.s.) |
| **D-naive vs D-gate (hygiene)** | 11 / 24 | **0.041** |
| D-gate vs A | 8 / 6 | 0.79 (parity) |
| D-gate vs C-lazy | 11 / 15 | 0.56 (parity) |
| C-eager vs A | 18 / 8 | 0.076 (trend) |

The naive live channel is significantly worse than *batched* delivery (not,
significantly, than nothing — an important nuance). The deficit decomposes:

- **Remove the announce prompt** (D-naive → D-plain): +0.107. An instruction to treat
  squigglies as urgent made the model jumpier, not better. This was a confound we
  introduced ourselves and caught in audit (§7).
- **Add the syntax gate** (D-plain → D-gate): +0.048 more, and the gate cut diagnostic
  deliveries from 240 to **71 (−70%)**. Suppressing two-thirds of the channel's
  traffic *improved* outcomes — the suppressed traffic was noise (§6).

Cleaned-up live delivery sits at parity with everything else. The one significant
effect in the whole delivery matrix is *naive-live vs cleaned-live*: **how you deliver
is the lever; whether you deliver is nearly a wash** for an untrained model.

## 5. Result 3 — the parity band (and a cautionary tale about "best")

At n=84, eager-sync led the table (0.595) and we were tempted to crown it. Fresh seeds
(6–11, in-flight power-up to n=168) regressed C-eager to 0.452; combined estimates so
far: C-lazy 0.530, C-eager 0.524, A 0.482. The eager-vs-lazy adjacency was never
significant (p=0.56), and the regression confirms the audit's warning that the
"monotonic ordering" was point-estimate storytelling. What survives at higher n:
batched sync slightly above none, everything well-delivered within a few points of
everything else. The headline effect remains the hygiene gap.

## 6. Mechanism — the self-inflicted squiggle loop

Why does naive live delivery hurt? The audit quantified it:

- **78.3%** (191/244) of D-naive's delivered diagnostics referenced *self-inflicted*
  states — parse errors and unbound names from the model's own half-finished edit
  sequences, not the task's bug.
- In the 25 paired units where A resolved and D-naive failed, **23 (92%)** had at
  least one self-inflicted diagnostic delivered.
- Artifact checks ruled out the alternatives: D-naive *bails* less than C conditions
  (0.107 vs 0.167), only 7.3% of its failures are token-budget-bound (ironically the
  budget-bound condition is C-eager, whose failures hit the cap 50% of the time), and
  input-token inflation is downstream of failure, not its cause.

The picture is a feedback loop: the model makes a multi-part edit; mid-way, the
checker (correctly!) reports the half-finished state as broken; delivered live, that
report interrupts the model into "fixing" code it hadn't finished writing; the fix
spawns new transient states, new squiggles, and the agent chases its own tail. Humans
filter ambient feedback through attention and a settled-state intuition — they know
not to chase squiggles mid-keystroke. An untrained LLM agent doesn't. The syntax gate
is a crude mechanical version of that intuition, and it removes most of the harm.

## 7. What we got wrong along the way (kept for the record)

Honest methods notes — each materially changed our conclusions when fixed:

1. **An efficiency claim that didn't survive power.** At 6 matched pairs, tuned-live
   appeared to solve jointly-solved tasks in half the round-trips (1.7 vs 3.2). At 20
   matched pairs it vanished (2.50 vs 2.50, ratio 1.00). Retracted; small-n matched
   subsets are seductive.
2. **An announce-prompt confound we introduced.** Only the live condition carried an
   extra instruction sentence; it accounted for roughly half the live deficit. Caught
   by adversarial audit, fixed with an announce-off arm.
3. **Unpaired tests on paired data.** Initial significance claims used two-proportion
   z-tests; correct paired McNemar weakened one headline (live vs nothing: not
   significant) and strengthened another (live vs batched: p=0.0002).
4. **A harness bug suppressing single-line edits.** The line-edit parser required
   `lines="N-M"`; the model often emits `lines="N"`. Every such edit silently dropped —
   the model re-emitted the same (often correct!) fix until budget. Several "0-edit"
   failures were this bug, and several earlier resolve numbers were undercounts.
5. **A task set that could mark its own homework.** The first synthetic batch had
   giveaway docstrings, loud tracebacks that named the fix, and a contrived task; an
   adversarial review (confounders / realism / difficulty critics) caught all three
   and the set was rebuilt around multi-site cascades with intent-in-tests.

## 8. Practical recipe for LSP-augmented coding agents

- **Default: post-edit synchronous delivery** (lint/type-check in the edit tool's
  result). Simple, never worse than alternatives in our data, and the model's
  test-loop already covers the rest.
- **If you want a live channel:** debounce it; deliver at a natural pause; **gate on
  parseable state** (never report on code mid-write); deduplicate against what the
  model has already seen; and skip the motivational prompt engineering — telling the
  model to treat squigglies as urgent made it measurably worse.
- **Check your checker carries signal for your bugs.** A type-checker contributes
  nothing on logic bugs (our real-SWE pilot: zero bug-relevant diagnostics). Delivery
  design can't rescue an empty channel.
- **Instrument the channel.** Count what fraction of delivered diagnostics are about
  the agent's own transient states; if it's high (ours was 78%), gate before you blame
  the model.

## 9. Limitations & ongoing

- **One model, 7B, one checker, synthetic single-file tasks.** Real-repo validation
  needs either stronger models (real SWE bugs were beyond the 7B even with oracle
  localization) or task sources where type signal aligns with the bug.
- **n=84 per condition for the audited results**; an n=168 power-up is completing
  (A/C arms done — see §5; live arms in flight). Parity claims are bounded by these n.
- **Zero-shot only.** Everything above is an *untrained* model. Two follow-ups now
  starting: **(a) self-distillation SFT** — train on the model's own resolved
  trajectories (observation tokens masked) in the deployment format and re-measure
  whether a trained model can pull live delivery *above* the parity band; **(b)
  richer constructive signals** — deliver hover/go-to-def-style context (signatures,
  available fields) alongside diagnostics, testing whether the channel's *content*
  rather than timing is the binding constraint.
- The strongest version of the live-feedback hypothesis — that it pays off when
  feedback arrives *during* long uninterruptible generations — is under-tested here:
  our agent's actions are short. Tasks with genuinely long single-pass generations are
  future work.

## Appendix A — real-SWE substrate (negative groundwork)

We built a native (no-Docker) SWE-rebench pipeline (clone + uv venv + editable install
+ test_patch; aarch64): 25 decontaminated post-2025 single-file tasks selected, 8
provisioned, 4 fully well-formed. The 7B agent engaged (7–12 edits with oracle
localization) but resolved 0; pyrefly produced no bug-relevant diagnostics on these
logic bugs. Conclusion: task *type*, not model scale alone, gates any LSP-delivery
effect on real repos. Pipeline retained: `harness/task_env.py`,
`scripts/rebench_*.py`.

## Appendix B — repository map

- `scaffold/stream_agent.py` — non-blocking agent (all delivery conditions, label-mask
  SFT capture); `scaffold/mock_env.py` — single-file env w/ real pyrefly.
- `scripts/synth_tasks.py` — task set + verifier; `scripts/synth_acd.py` — condition
  runner; `scripts/harvest_sft.py` — self-distillation harvester;
  `scripts/d_sft.py` — LoRA trainer.
- `scripts/i_eval*.py` — isolation evals (R0/R0b/R0c).
- `runs/agent/*.json` — all condition results (per-rollout rows incl. event traces).
- `log.md` — full chronological decision log, including the audit memo and all
  retractions; `experiment_plan.md` — original plan (historical; superseded in parts).
