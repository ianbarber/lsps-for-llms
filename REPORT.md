# Making a Language Server Pay Off for a Coding Agent: Redundant Information, Cheaper Retrieval

## Abstract

Do language servers help coding agents by supplying information, and do they make retrieval cheaper?
We measure both across synthetic and real-repository tasks, a 7B and a 27B open model, and frontier
models in a tool-calling loop, by toggling each language-server capability at fixed model capability.
The payoff is retrieval efficiency: a go-to-definition action cuts input tokens 3.5 to 4.7 times at
equal success, but only when retrieval is required, the alternative is a whole-file read, and the
agent chooses the cheap action. A capable model chooses it from prompt framing alone (a frontier model
on 100% of rollouts); a 7B has to be trained to (0% to 100% use, 4.5 times fewer tokens). The
information itself (diagnostics, references, completion, type inference, and even the program's runtime
behavior) does not raise success when the agent can derive the same fact by reading the source, which
held on every channel and task we tested. This covers a lot of real code, but not all of it: a very
large or highly complex codebase, where the needed fact is not readable in the steps available, may
carry information we did not test.

## Contributions

- **Efficiency is the win, under three conditions.** A go-to-definition action reduces input tokens
  3.5 to 4.7 times at equal success, and only when retrieval is required, the counterfactual is a
  whole-file read, and the agent chooses the cheap action. We show this on synthetic and real
  vendored-library tasks, for a 7B, a 27B, and three frontier models.
- **Election is capability-gated.** A capable model chooses the cheap action when the system
  prompt frames it as cheaper (a 27B at 88 to 93%, a frontier model at 100%); a 7B needs on-policy
  training (0% to 100%). We give the training recipe and show that prompting and offline imitation
  fail for the weak model.
- **Information is redundant when it is readable.** When the fact a language server would supply is
  derivable from the source by reading, handing it to a self-retrieving agent does not raise pass@1.
  This held on every channel we tested (correction, completeness, navigation, prevention, scale, type
  inference) and even on the program's runtime behavior. It covers a lot of real code. We did not test
  very large or highly complex codebases, or facts that appear only at run time, where the fact may not
  be readable in the steps available and a language server may then carry information that helps.

---

## 1. Introduction

Coding agents spend most of their tokens retrieving context. A language server offers a coding
agent two things: information (diagnostics, inferred types, reference lists) and cheaper retrieval
(a go-to-definition that returns one symbol instead of a whole file). We ask which one helps, and
under what conditions.

A capable agent can read files on its own. Whatever a language server computes, it computes from
source the agent can also read, so its information may be redundant. Prior work treats
language-server feedback as a useful reward or context signal (Zhang et al., 2025); we find that
for a self-retrieving agent the information does not raise success wherever the agent can read the
source and derive it, and the residual value is the cost of retrieval.

Whether an agent then uses a cheaper retrieval action is a policy question, not an information one.
An agent will not use a cheaper action because it exists. We separate the information value from
the efficiency value by toggling each capability at fixed model capability: we run the same model
with and without a given language-server action, rather than compare a trained model to an
untrained one. We do this across two synthetic task families, real vendored library code, a 7B and
a 27B open model, and three frontier models driven through a tool-calling API.

The result is a deployable account of when a language server pays off in an agentic loop, and how
to get the agent to use it. On the synthetic efficiency suite a 7B moves from 0% to 100%
go-to-definition use and from 3086 to 688 input tokens after one on-policy training round; an
untrained 27B reaches 88 to 93% use from prompt framing; a frontier model in a tool-calling loop
reaches 100% use and a 3.7 times token reduction at equal success (Figure 1).

![Tool-value ablation across models](docs/figures/fig1.png)

*Figure 1. The tool-value ablation. Removing the go-to-definition action, so the same model can
only read whole files, costs 3.5 to 4.7 times more input tokens at the same (ceiling) success, for
a 27B and two frontier models on the obscure real-code suite. Tokens on a log scale.*

## 2. Setup

**Agent and actions.** The agent fixes a bug in a target file through a try-and-correct loop. It
has five actions: `<read path>` returns a full file, `<defn sym>` returns the definition span of a
symbol, `<findrefs sym>` returns reference sites, `<test>` runs the suite, and `<edit>` applies a
line-range change. We run two harnesses. A local token-stream harness drives open-weight models
(Qwen2.5-Coder-7B-Instruct, Qwen3.6-27B) with logit access. A turn-based tool-calling harness
(`scripts/api_agent.py`) drives frontier models through the OpenRouter API, exposing the same
actions as function tools, the modality production agents use.

**`<defn>` is a real go-to-definition, not an oracle.** Given a symbol name the agent requests, the
tool AST-resolves that symbol's top-level definition against the live workspace and returns its
source span, what a language server's go-to-definition returns, with no privileged knowledge of
which symbol or what the answer is, and returning "(no definition found)" on an unresolvable name.
We validated the static resolver against a production language server. On the synthetic suite a live
`pyrefly lsp` daemon (`textDocument/definition`) resolves all 12 evaluation symbols to the same
definition as the static resolver (12/12), and a run with `<defn>` backed by the live daemon
reproduces the headline (use 0 to 100%, 2894 to 689 tokens, 58 to 100% success). On the real-code
suite the agreement holds wherever pyrefly resolves: 9 of 11 `toolz`/`more_itertools` symbols agree
with the static resolver and none disagree (pyrefly returns null on 2 `more_itertools` re-exports,
where the agent falls back to the static resolver). For real-LSP runs a persistent `pyrefly lsp`
daemon is started once per task workspace and reused across the rollout (`--lsp-defn`), rather than
spawned per query. A trained-7B run on the real-code suite with `<defn>` backed by the per-task daemon
matches the static-resolver run exactly: 18 of 22 solved, 100% go-to-definition use, identical mean
input tokens. The resolver backing the action does not change the result, because both return the same
definition span.

**Environments, synthetic and real.** The
synthetic *effic* suite buries one needed symbol in a ~370-line module, so `<read>` returns ~3500
tokens and `<defn>` returns ~50; tasks are non-guessable, so retrieval is required. A
*read-required* boundary suite inverts this: the needed symbol is unknowable without reading, so
`<defn>` cannot solve it. The real-code suites replace the synthetic module with real vendored
library source (`toolz`, `more_itertools`): `effic_real` uses familiar functions, `effic_real2`
uses obscure ones a model is unlikely to have memorized. An inference suite (`gapd`) requires a
type the type-checker infers (overload resolution, generics, union narrowing, `Protocol`,
`TypedDict`).

**Metrics.** We report go-to-definition use rate, whole-file read rate, input tokens to solve, and
pass@1, with exact McNemar on success and an exact sign test on tokens over paired (task, seed)
units.

## 3. Language-server information is redundant (C1)

We tested whether the information a language server provides, as opposed to a cheaper way to
retrieve it, raises success for a self-retrieving agent. It does not, on any channel we tested. The
agent reads the source and derives the same fact, so the information is redundant wherever it is
readable.

**Correction.** An oracle ladder replaces the diagnostic with progressively stronger feedback: no
feedback, synchronous diagnostics, perfect error localization, and the gold fix. For a 7B, perfect
localization lowers pass@1 (0.45 to 0.25, p<0.001), and the gold fix does not beat no-feedback
(0.39 vs 0.45, p=0.29), a capability floor rather than an information gain. A 35B mixture-of-experts
model ceilings the suite. The diagnostic adds nothing the agent does not already read.

**Completeness, scale, navigation, prevention.** Varying repository size from 21 to 86 files at a
fixed read budget, success stays at 1.00 with 6 to 8 reads, and find-references does not earn its
keep, because reading does not become expensive at tractable scale. When name search fails on a
re-exported symbol, the agent reads the call graph and succeeds without find-references. The agent
reads a library before calling its API and does not emit the hallucinated member, so completion has
nothing to prevent.

**Inference.** The last channel is type inference, and it is the one where a type checker is usually
the *unique* detector: a wrong usage that the test does not exercise. We test it directly with
**held-out scoring**. Each of six tasks (overload resolution, optional `None`, union narrowing,
`TypedDict`) has a plausible wrong fix that *passes a visible test the agent runs but fails a hidden
held-out test*, and that pyrefly flags statically (`sum(int | None)`, an unknown `TypedDict` key,
`NoneType has no attribute`). We give two frontier models a `check_types()` tool that surfaces those
diagnostics, told the visible test is partial, and also run a realistic condition with no such hint.
The checker changes nothing. In every condition both models solve **18/18 with zero latent bugs**,
with and without the checker, and even when they elect it (`deepseek-chat-v3.1` calls it on every
rollout, `claude-sonnet-4.5` on more than half). The models write the correct fix by careful reading
and do not ship the inference bug for the checker to catch.

A language server computes from source the agent can also read. Across correction, completeness,
navigation, prevention, scale, and type inference, a capable agent reads that source and derives the
same fact, so the information is redundant and the value of a language server is the cost of retrieval,
not its content. This holds wherever the fact is readable, which covers most everyday code. It need not
hold where the fact is not readable in the steps available: a codebase too large or too tangled to read
under budget, or an inference a frontier model reliably gets wrong while the visible test misses it. We
did not find such a case, the inference tasks were an adversarial reviewer's plausible-error designs and
the models still did not err, but we did not rule one out (§7).

![A type checker does not reduce latent bugs, even with held-out scoring](docs/figures/fig2.png)

*Figure 2. The held-out-scored inference test. A `check_types()` tool does not raise pass@1: both
frontier models solve 18/18 with zero latent bugs, with the checker, without it, and in a realistic
no-hint condition. The plausible wrong fix passes the visible test the agent runs but fails the hidden
test and is flagged by pyrefly; the models write the correct fix anyway.*

**Execution.** Every channel above serves a fact already in the source. We also tested a fact that is
not in the text: the program's runtime behavior, which the agent can get right only by mentally
executing the code and verify only by running it. Fourteen small bug-fix tasks each have a plausible
wrong fix that fails a visible test (so running is a detector) yet is well-typed (so a checker is
useless and execution is the only in-budget detector), scored held-out. Two frontier models run under
three arms: no execution (the run-tests tool is removed and the agent reasons from the source and the
shown spec), elect-to-run (the agent may run the visible test), and feedback-handed-over (the
environment volunteers the test result after every edit). Five tasks are Python-semantic traps whose
plausible reimplementation mis-simulates a real behavior (`str.lstrip` strips a character set, not a
prefix; `int(x/k)` truncates where `//` floors; `round` is banker's rounding). Across 252 rollouts both
models solve every task in every arm, 100% held-out pass@1, including with no execution at all. The only
quantity that moves is efficiency: handing the result over (1.39 turns) is cheaper than electing to run
it (2.42) at identical success. So execution feedback does not raise correctness here, it only saves
turns, where the code is small enough to read or simulate reliably. Execution feedback does raise success
in other regimes, harder generation and weaker models (§6); we did not find the scale at which it becomes
the only detector for a frontier agent (§7).

![Execution feedback does not change correctness, only efficiency](docs/figures/fig6.png)

*Figure 6. Execution feedback is redundant for a frontier agent on small bug-fixes. Across 14 tasks,
two models, and three seeds, held-out pass@1 is 100% whether the agent cannot run the code, elects to
run it, or is handed the result for free (a). The only effect is efficiency: volunteering the result
finishes in fewer turns (b).*

## 4. Retrieval efficiency is real, under three conditions (C2)

Because `<read X>` and `<defn X>` usually return the same symbol, the cheap action saves tokens
whenever the agent would otherwise read. We isolate this value with a **tool-value ablation**: hold
the model fixed and toggle the action, rather than compare a trained model to an untrained one.

| setting | model | tokens with `<defn>` | tokens read-only | factor | success |
|---|---|---|---|---|---|
| synthetic, trained vs untrained | 7B | 688 | 3086 | 4.5× | 1.00 vs 0.65 |
| real obscure (`effic_real2`) | 27B | 1302 | 4563 | 3.5× | 1.00 = 1.00 |
| real, tool-calling | claude-sonnet-4.5 | 6018 | 21985 | 3.7× | 1.00 = 1.00 |
| real, tool-calling | deepseek-chat-v3.1 | 7705 | 36192 | 4.7× | 1.00 = 1.00 |

*Mean input tokens to solve. For the 7B the comparison is untrained-reading vs trained-`<defn>`
(election by training); for the 27B and frontier models it is the same model with the action removed
vs present (election by framing). The 27B ablation is paired (sign test p=0.0013, cheaper on 33/44
cells); the frontier numbers are larger because a tool-calling turn re-sends the growing context, so
a read-only agent re-pays for the file it loaded.*

The reduction holds only under three conditions, each isolated by a control.

1. **Retrieval is required.** On familiar library functions, the base model guesses the API from
   memory (35 of 41 successes solve with no retrieval at all), so there is no read to replace and the
   saving disappears. On obscure functions the guess fails and retrieval is required. This is why we
   report the clean ratios on the obscure suite.
2. **The counterfactual is a whole-file read.** The saving is `tokens(read) − tokens(defn)`, realized
   only when the agent would otherwise read. A 7B guesses rather than reads even when it cannot (it
   thrashes on 18 of 20 failures without reading), so the per-call saving is absent for it; a 27B and a
   frontier model read, so the saving is present.
3. **The agent chooses the cheap action.** Availability is not use (§5).

A separate control rules out that the saving merely reflects retrieval helping success: a model
trained to retrieve via `<read>` spends 3191 input tokens against a model trained to retrieve via
`<defn>` at 684, on tasks both solve (4.7× cheaper, sign test p=6.8e-4, n=40). Both retrieve and
solve, so the saving is the action choice itself.

The cost-gap question needs a task where the agent must read one file to edit another, so a cheaper
read can pay off; the symbol is consulted, not edited. The vendored-library suites are built for this.
An off-the-shelf option, RefactorBench (Gautam et al., 2025), edits the symbol's own file, so the agent
loads that file to edit it regardless of `<defn>` and efficiency has nothing to save, which is why we
did not use it.

## 5. Election is capability-gated (C3)

Conditions (1) and (2) are properties of the task and the model's reading habit. Condition (3), that
the agent chooses the cheap action, is the practitioner's lever, and it depends on model capability.

| model | `<defn>` use by default | `<defn>` use when framed as cheaper | trained |
|---|---|---|---|
| 7B (Qwen2.5-Coder) | 2% | 2% | 100% (after one on-policy round) |
| 27B (Qwen3.6) | 0% | 88% | not needed |
| deepseek-chat-v3.1 | n/a | 93% | not needed |
| claude-sonnet-4.5 | n/a | 100% | not needed |

![Go-to-definition use by model](docs/figures/fig3.png)

*Figure 3. Election is capability-gated, on the obscure real-code suite. A 7B uses go-to-definition
2% of the time by default and 100% after one on-policy training round; a 27B and two frontier models
reach 93 to 100% from prompt framing alone.*

**A weak model needs training.** The 7B uses `<defn>` 2% of the time by default, and a prompt
instructing it to prefer `<defn>` leaves use near 0%. Offline imitation of cheap `<defn>` trajectories
also fails: use stays near 0% and tokens do not fall, because the demonstrations never show the
expensive action available and the cheap one chosen, so the cloned policy is unconstrained exactly
where the preference must be expressed.

On-policy imitation fixes the distribution mismatch. We roll out the untrained agent with both actions
available; where it emits `<read>` for a non-editable file and the needed symbol is resolvable, we drop
that step and let the agent emit `<defn sym>` itself, keeping the rest of its trajectory; we fine-tune a
LoRA adapter on the relabeled trajectories (one DAgger round). No gold action is injected: we relabel
only the retrieval channel of the agent's own behavior. The 7B then moves to 100% use, success 0.65 to
1.00, and 3086 to 688 input tokens (McNemar success p=1.5e-5, b=17/c=0; sign test tokens p=2.2e-4,
cheaper on 37/48). The preference generalizes to held-out task types (0.42 to 1.00 success, 5.2× tokens),
transfers unchanged to real `toolz`/`more_itertools` code (100% `<defn>` use, correct non-idiomatic calls
from the retrieved signature), and preserves a read-when-needed boundary: on read-required tasks the
trained agent still reads 100% of the time and success rises (0.54 to 0.83). A cost-reward GRPO objective
reaches the same operating point (Appendix A), so an independent training signal instills the same
preference.

**A capable model needs only framing.** The same prompt that leaves the 7B at 2% moves an untrained 27B
to 88% use, with no training. Framing the action in the system prompt as cheaper than a read is what does
it. A control isolates the framing from the rest of the prompt: on the identical synthetic 2-file suite,
an older prompt that mentioned `<defn>` only in the task message gave 0% use and 4058 tokens, while the
current prompt, which presents `<defn>` beside the read instruction as the cheaper option, gives 88% use
and 1237 tokens. Frontier models go further: `claude-sonnet-4.5` chooses `<defn>` on 100% of rollouts and
`deepseek-chat-v3.1` on 93%, with no training.

![The 7B on-policy training win](docs/figures/fig4.png)

*Figure 4. The on-policy training win for a 7B on definition-sufficient tasks (12 seeds):
go-to-definition use rises to 100%, mean input tokens fall, and success rises on all tasks and on
held-out task types.*

![The learned policy is a boundary](docs/figures/fig5.png)

*Figure 5. The learned policy is a boundary, not a collapse. On definition-sufficient tasks the
trained 7B uses go-to-definition; on read-required tasks it still reads.*

## 6. Related work

Our method is on-policy imitation under distribution shift. GKD (Agarwal et al., ICLR 2024) distills a
teacher under the student's own distribution; DAgger (Ross, Gordon & Bagnell, AISTATS 2011) and cost-aware
AggreVaTe (Ross & Bagnell, 2014) ground rolling out a learned policy and relabeling with an expert;
Revisiting DAgger for LLM agents (Li et al., 2026) applies the idea to tool-using language models. STaR
(Zelikman et al., 2022) bootstraps reasoning from the model's own rationales; we bootstrap a cost
preference from the model's own trajectories. The expert here is a deterministic read-to-defn relabel,
not a model.

Cost-aware tool use through RL is the alternative we corroborate but do not require. OTC-PO (Wang et al.,
2025) and IKEA (Huang et al., 2025) reward fewer or cheaper tool calls; their results motivate that a
token-cost reward learns the same preference we obtain by on-policy imitation, which our GRPO
corroboration confirms (Appendix A).

Language servers as a supervision signal. RLCSF (Zhang et al., 2025) takes the complementary view that
compiler and language-server signals, including diagnostics, symbol resolution, type information, and
references, are a valuable supervision signal for a coding agent, and builds a reinforcement-learning
system around exposing them. It is a methodology contribution and reports no empirical comparison of that
information against plain source reading. Our result supplies one: in the channels we tested, a
self-retrieving agent recovers those facts by reading, so supplying them does not raise pass@1, and the
value we measure is retrieval cost.

Redundant retrieval for capable models. Outside code, the same pattern is documented: a tool or retrieval
call can be redundant or net-negative when the model already holds the answer. "To Call or Not to Call"
(arXiv:2605.00737) reports tool calls with negative utility in roughly a third of cases when the model
succeeds without them, and IKEA (Huang et al., 2025) reports search agents over-retrieving rather than
using what they already know. Our finding is the language-server instance of the same effect, measured
against code-agent pass@1.

Feedback that carries new information. Execution-grounded feedback can raise code-generation success:
Self-Debugging (Chen et al., 2024) feeds runtime results back to the model, RLTF (Liu et al., 2023) trains
on unit-test outcomes, and type-constrained decoding (Mündler et al., 2025) enforces well-typedness during
generation; self-repair gives only modest gains once its cost is counted (Olausson et al., 2024). These
signals are not in the source text the agent reads: they are the program's behavior at run time, or a
constraint imposed during decoding. Our §3 execution test is the case where the agent can recover the
behavior from small code by reading or simulating it, and there the feedback only saves turns. Execution
feedback helps in the harder-generation and weaker-model settings these works study, and is redundant in
ours.

## 7. Limitations

- **The cost gap is engineered, but validated.** The synthetic suite sets the read-versus-defn cost gap
  and the non-guessability of tasks. We validate that the effect survives on real vendored library code
  and at the frontier in the tool-calling modality (§4), but the read-required boundary covers two reasons
  a read is needed (name-hidden, many-symbol), not all.
- **Redundancy holds where the fact is readable.** Across the channels we tested, the language server's
  information is redundant for a self-retrieving agent, because the agent reads the source and derives the
  fact. A fact the agent cannot recover by reading, a runtime value it would have to execute for, an
  inference beyond the model's reach, or a codebase too large to read under budget, could make it
  non-redundant. We did not find such a case but did not rule one out.
- **The inference null is bounded by task difficulty.** Our held-out-scored inference test (§3) covers the
  regime where a type checker is usually the unique detector (a wrong usage the visible test misses), and
  frontier models still solve 18/18 with zero latent bugs. We did not find an inference a frontier model
  *reliably gets wrong* while the visible test misses it; the tasks were an adversarial reviewer's
  plausible-error designs and the models did not err, so we cannot rule out that harder inference exists
  where a checker would prove non-redundant. We also did not test the cross-file regime where the relevant
  definitions are too many to read under the budget.
- **The execution null is bounded by task scale.** The execution test (§3) covers small, self-contained
  bug-fixes where a frontier model simulates the code reliably, and there both models solve 14/14 in every
  arm. We deliberately stress-tested simulation with Python-semantic traps and the models still did not err,
  but we did not find the scale or opacity (long programs, behavior behind a dependency the agent did not
  read, genuine nondeterminism) at which execution becomes the unique in-budget detector. Published results
  show execution feedback helps in harder-generation and weaker-model regimes (§6); our null is the
  frontier-on-small-tasks corner, not a claim that execution never helps.
- **Tool-calling token accounting.** The API harness re-sends context each turn, so its absolute token
  counts are not comparable to the local harness; we report within-harness ratios at equal success.

## 8. Conclusion

A language server's payoff for a coding agent is a cheaper retrieval action. A go-to-definition cuts input
tokens 3.5 to 4.7 times at equal success, under three conditions: retrieval is required, the alternative is
a whole-file read, and the agent chooses the cheap action. Election is the practitioner's lever. A capable
model chooses the cheap action when the prompt frames it as cheaper; a weak model learns to through one
round of on-policy imitation, where prompting and offline cloning fail. The information a language server
supplies does not raise success when the agent can read the source and derive the same fact, which held on
every channel and task we tested. A very large or highly tangled codebase, where the needed facts are not
readable in the steps available, is where its information might still pay off; we did not test that regime.

**The recipe.** Expose go-to-definition as an action, not diagnostics as context. Frame it in the system
prompt as cheaper than a whole-file read; a capable model then uses it without training. If the model is
weak enough to ignore the framing, train the preference on-policy by relabeling its own reads to
definitions. Mix in tasks that genuinely need a full read, so the agent learns when the cheap action
suffices.

---

## Appendix A: cost-reward RL corroboration

A GRPO objective (reward: solve at minimum tokens, group-normalized advantage over the model's own action
tokens) reaches the same operating point as the on-policy relabel, over four rounds. A single round
under-trains.

| stage | `<defn>` use | mean input tokens | solved |
|---|---|---|---|
| wild (round 0) | 37% | 2048 | 67% |
| after 1 round | 6% | 3041 | regressed |
| after 4 rounds (harvest) | 86% | 790 | n/a |
| clean held-out retest | 86% | 663 | 100% |

GRPO corroborates that a token-cost objective instills the same preference; the SFT relabel remains the
recipe because it needs one round.

## Appendix B: cross-scale transfer of the training recipe

The same relabel pipeline on Qwen3.6-27B (a different generation and a reasoning model) gives the same
result.

| | `<defn>` use | read use | mean input tokens | success |
|---|---|---|---|---|
| wild 27B | 0% | 96% | 4058 | 0.96 |
| trained 27B | 100% | 0% | 726 | 1.00 |

The token reduction at matched success is 5.5× (n=24). The recipe is not a small-model artifact and
transfers across a roughly 4× scale jump and a model-family change.

---

All code, suites, result JSONs, and reproduction scripts are in the repository; see the README for how to
reproduce each number and how to test a new model.
