# Making a Language Server Pay Off for a Coding Agent: Redundant Information, Cheaper Retrieval

## Abstract

We ask two questions about giving a coding agent a language server: does it help by supplying
information, and does it help by making retrieval cheaper. We measure both across synthetic and
real-repository tasks, a 7B and a 27B open model, and frontier models in a tool-calling loop, by
toggling each language-server capability at fixed model capability. The information (diagnostics,
references, completion, type inference) does not raise success, because a model that can read
files derives the same facts itself. A go-to-definition action does help, cutting input tokens
3.7 to 5.3 times at equal success, but only when retrieval is required, the alternative is a
whole-file read, and the agent chooses the cheap action. A capable model chooses it from prompt
framing alone (a frontier model on 100% of rollouts); a 7B has to be trained to (0% to 100% use,
4.5 times fewer tokens).

## Contributions

- **Information is redundant.** Across correction, completeness, navigation, prevention, scale,
  and type inference, handing a self-retrieving agent the language server's information does not
  raise pass@1. The agent reads the source and derives the fact itself.
- **Efficiency is real, under three conditions.** A go-to-definition action reduces input tokens
  3.7 to 5.3 times at equal success, and only when retrieval is required, the counterfactual is a
  whole-file read, and the agent chooses the cheap action. We show this on synthetic and real
  vendored-library tasks, for a 7B, a 27B, and three frontier models.
- **Election is capability-gated.** A capable model chooses the cheap action when the system
  prompt frames it as cheaper (a 27B at 88 to 95%, a frontier model at 100%); a 7B needs on-policy
  training (0% to 100%). We give the training recipe and show that prompting and offline imitation
  fail for the weak model.

---

## 1. Introduction

Coding agents spend most of their tokens retrieving context. A language server offers a coding
agent two things: information (diagnostics, inferred types, reference lists) and cheaper retrieval
(a go-to-definition that returns one symbol instead of a whole file). We ask which one helps, and
under what conditions.

A capable agent can read files on its own. Whatever a language server computes, it computes from
source the agent can also read, so its information may be redundant. Prior work treats
language-server feedback as a useful reward or context signal (Zhang et al., 2025); we find that
for a self-retrieving agent the information does not raise success, and the residual value is the
cost of retrieval.

Whether an agent then uses a cheaper retrieval action is a policy question, not an information one.
An agent will not use a cheaper action because it exists. We separate the information value from
the efficiency value by toggling each capability at fixed model capability: we run the same model
with and without a given language-server action, rather than compare a trained model to an
untrained one. We do this across two synthetic task families, real vendored library code, a 7B and
a 27B open model, and three frontier models driven through a tool-calling API.

The result is a deployable account of when a language server pays off in an agentic loop, and how
to get the agent to use it. On the synthetic efficiency suite a 7B moves from 0% to 100%
go-to-definition use and from 3086 to 688 input tokens after one on-policy training round; an
untrained 27B reaches 88 to 95% use from prompt framing; a frontier model in a tool-calling loop
reaches 100% use and a 4.0 times token reduction at equal success.

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
We validated the static resolver against a production language server: a live `pyrefly lsp` daemon
(`textDocument/definition`) resolves all 12 evaluation symbols to the same definition (12/12), and
a run with `<defn>` backed by the live daemon reproduces the headline (use 0 to 100%, 2894 to 689
tokens, 58 to 100% success).

**Environments, synthetic and real.** We use both, as two views of the same experiment. The
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
retrieve it, raises success for a self-retrieving agent. It does not, on any channel we tested.

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

**Inference.** The remaining channel is the one localization and enumeration never tested: facts
the type-checker infers that a model might not derive by reading. We built eight tasks whose correct
fix depends on a non-trivially inferred type, and verified that the type-checker names the type that
the runtime test failure does not (for example `Returned type int | None is not assignable to int`,
or `Object of class Plain has no attribute escape`). We then gave two frontier models a
`check_types()` tool that surfaces those diagnostics. It changes nothing: `deepseek-chat-v3.1` and
`claude-sonnet-4.5` each solve 16/16 with and without it, and `claude-sonnet-4.5` never calls it
(0/16 rollouts, `deepseek` 1/16). The models read the library (mean 0.9 and 1.7 reads) and infer the
types themselves.

A language server computes from the same source the agent can read. Across the channels we tested, a
capable agent self-retrieves and self-infers, so the information is redundant and the residual value
of a language server is the cost of retrieval, not its content.

## 4. Retrieval efficiency is real, under three conditions (C2)

Because `<read X>` and `<defn X>` usually return the same symbol, the cheap action saves tokens
whenever the agent would otherwise read. We isolate this value with a **tool-value ablation**: hold
the model fixed and toggle the action, rather than compare a trained model to an untrained one.

| setting | model | tokens with `<defn>` | tokens read-only | factor | success |
|---|---|---|---|---|---|
| synthetic, trained vs untrained | 7B | 688 | 3086 | 4.5× | 1.00 vs 0.65 |
| real obscure (`effic_real2`) | 27B | 1260 | 4644 | 3.7× | 1.00 = 1.00 |
| real, tool-calling | claude-sonnet-4.5 | 6014 | 23811 | 4.0× | 1.00 = 1.00 |
| real, tool-calling | deepseek-chat-v3.1 | 9325 | 49142 | 5.3× | 1.00 = 1.00 |

*Mean input tokens to solve. For the 7B the comparison is untrained-reading vs trained-`<defn>`
(election by training); for the 27B and frontier models it is the same model with the action removed
vs present (election by framing). The 27B ablation is paired (sign test p=0.017, cheaper on 17/22
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

**Off-the-shelf refactoring benchmarks do not isolate this.** We evaluated RefactorBench
(Gautam et al., 2025) for a non-constructed real-code test and found it structurally unsuited to the
cost-gap question: its tasks edit the symbol's own large file, so the agent loads that file to edit it
regardless of `<defn>`; the rename variants exercise find-references, the channel §3 finds redundant;
and three of four candidate symbols sit past the editable-view truncation. The cost-gap question needs
a task where the symbol is consulted, not edited.

## 5. Election is capability-gated (C3)

Conditions (1) and (2) are properties of the task and the model's reading habit. Condition (3), that
the agent chooses the cheap action, is the practitioner's lever, and it depends on model capability.

| model | `<defn>` use by default | `<defn>` use when framed as cheaper | trained |
|---|---|---|---|
| 7B (Qwen2.5-Coder) | 2% | 2% | 100% (after one on-policy round) |
| 27B (Qwen3.6) | 0% | 88% | not needed |
| deepseek-chat-v3.1 | n/a | 95% | not needed |
| claude-sonnet-4.5 | n/a | 100% | not needed |

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
to 88% use. The driver is the system-prompt framing of the action as cheaper than a read, not the
repository and not a trained preference. A prompt-versus-structure control confirms it: on the identical
synthetic 2-file suite, an older prompt that advertised `<defn>` only in the task message gave 0% use and
4058 tokens, while the current prompt, which presents `<defn>` beside the read instruction as the cheaper
option, gives 88% use and 1237 tokens. Frontier models go further: `claude-sonnet-4.5` chooses `<defn>` on
100% of rollouts and `deepseek-chat-v3.1` on 95%, with no training. For a weak model the cheap-retrieval
preference is learned on-policy; for a capable model it follows from framing the tool as the cheaper
option.

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
corroboration confirms (Appendix A). The closest prior work on language servers is RLCSF (Zhang et al.,
2025), which rewards compiler and language-server diagnostics during RL; we find its information redundant
for a self-retrieving agent and locate the residual value in retrieval cost.

## 7. Limitations

- **The cost gap is engineered, but validated.** The synthetic suite sets the read-versus-defn cost gap
  and the non-guessability of tasks. We validate that the effect survives on real vendored library code
  and at the frontier in the tool-calling modality (§4), but the read-required boundary covers two reasons
  a read is needed (name-hidden, many-symbol), not all.
- **Redundancy is a majority result, not an absolute.** Across the six channels we tested, the language
  server's information is redundant for a self-retrieving agent. We do not claim it is redundant for every
  task: a fact the agent cannot recover by reading (a non-readable runtime value, an inference beyond the
  model's reach) could make it non-redundant. We did not encounter such cases often enough to measure them.
- **Tool-calling token accounting.** The API harness re-sends context each turn, so its absolute token
  counts are not comparable to the local harness; we report within-harness ratios at equal success.

## 8. Conclusion

A language server does not raise a coding agent's success by supplying information, because across the
channels we tested the agent reads the source and derives the same facts itself. Its value is a cheaper
retrieval action, which cuts input tokens 3.7 to 5.3 times at equal success, under three conditions:
retrieval is required, the alternative is a whole-file read, and the agent chooses the cheap action.
Election is the practitioner's lever. A capable model chooses the cheap action when the prompt frames it as
cheaper; a weak model learns to through one round of on-policy imitation, where prompting and offline
cloning fail.

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
