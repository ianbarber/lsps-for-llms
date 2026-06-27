# Making a Language Server Pay Off for a Coding Agent: Train It to Retrieve Cheaply

A coding agent that can read files on its own rarely needs a language server for
*information*, but it can use one for *retrieval efficiency*. On our synthetic suites, the
expensive action a capable agent already takes — reading a whole file — returns the same
symbol that a cheap go-to-definition would. The agent does not prefer the cheap action by
default, by prompting, or by offline imitation of it. We show that a lightweight on-policy
relabel teaches the preference, cutting input tokens several-fold while preserving success
and a read-when-needed boundary. All `<defn>` results use a real resolver over the live
workspace; no oracle is consulted in the evaluation loop.

## Contributions

- We show that, on our suites, a language server's information channels do not improve a
  self-retrieving agent's pass@1; the residual value is cheaper retrieval.
- We demonstrate that a cheap go-to-definition preference is not learned from prompting or
  offline cloning, but is learned from on-policy cost-aware imitation.
- We verify that the learned policy preserves a read-when-needed boundary on tasks where
  `<defn>` is genuinely insufficient.
- We corroborate the finding with a token-cost RL baseline and a scale check on a 27B model.

The key caveat is scope: the evidence is on synthetic tasks with a controlled cost gap,
and coverage is labelled during training. Section 5.6 shows that a capable trained model
judges coverage per-instance at test time, but real-repository indirection remains an open
question.

---

## 1. Introduction

Coding agents spend most of their tokens retrieving context. On our synthetic suites the
same symbol is often available two ways: a targeted language-server query
(go-to-definition, ~50 tokens) or a whole-file read (~3500 tokens). A capable agent does
not choose the cheap path on its own.

This is a *policy* problem, not an information one. Prompting does not make the agent
prefer `<defn>`. Offline imitation of cheap `<defn>` trajectories also fails, because the
demonstrations never show the expensive action available and the cheap one chosen. We show
that on-policy imitation fixes the mismatch: relabel the agent's own `<read>` steps to
`<defn>` and fine-tune on those trajectories. On definition-sufficient tasks, a 7B agent
moves from 0% to 100% `<defn>` use, mean input tokens fall from 3086 to 688 (4.5×), and
success rises from 0.65 to 1.00. The policy also preserves a read-when-needed boundary on
tasks where `<defn>` is genuinely insufficient.

The evidence is on synthetic tasks with a controlled cost gap. We treat the result as a
proof of mechanism and a reproducible recipe, not as a claim that the effect holds
automatically on arbitrary real repositories.

## 2. Motivation: information channels are redundant on our suites

The evidence below comes from synthetic suites with oracle channels. It is meant to
motivate why we focus on retrieval cost, not to prove that LSP information is universally
redundant. A reviewer should read these nulls as scoped motivation, not as a headline
result.

We tested four information channels:

- **Correction.** An oracle ladder (no feedback / synchronous diagnostics / perfect
  localization / gold fix) shows localization harms (p<0.001) and gold-fix does not beat
  no-feedback for a 7B. A 35B MoE ceilings the suite. The diagnostic adds nothing the model
  does not already read.
- **Completeness and scale.** Varying repository size from 21 to 86 files at a fixed
  generous read budget, success stays at 1.00 with roughly 6–8 reads. Find-references does
  not earn its keep because reading does not become expensive at tractable scale.
- **Navigation and prevention.** Find-references is redundant on success: the agent reads
  the call graph when name search fails. Prevention fails its precondition, because the
  agent reads the library and never emits the hallucinated symbol.

We also studied how feedback is delivered. In an n=168 zero-shot sweep (14 tasks × 12
seeds), we varied synchronous end-of-turn delivery, interleaved live delivery, eager versus
lazy updates, and hygiene gating. Properly delivered feedback of any timing lands in a
parity band (fix-rates 0.46–0.53, all pairwise differences non-significant). Only naive
live delivery hurts, and that harm is a recoverable format-hygiene artifact: diagnostic
markers leak into the agent's own edits. It is not an intrinsic cost of liveness.

Neither the information channel nor the delivery format is the binding constraint on our
suites. The remaining lever is the cost of retrieval.

## 3. Setup

The agent is a 7B coding model (Qwen2.5-Coder-7B-Instruct) in a try-and-correct loop with
`<read>`, `<defn>` (go-to-definition), `<test>`, and `<edit>` actions, a real `pyrefly`
type-checker, and a non-blocking stream harness.

**`<defn>` is a real go-to-definition, not an oracle.** Given a symbol name the agent
requests, the tool AST-resolves that symbol's top-level definition against the live
workspace and returns its source span — exactly what an LSP go-to-definition does, derived
from the codebase with no privileged knowledge of which symbol or what the answer is, and
returning "(no definition found)" on an unresolvable name. We validated this against a
production language server as a sanity check: driving a live `pyrefly lsp` daemon (JSON-RPC
`textDocument/definition`) resolves all 12 evaluation symbols
to the same definition as the static resolver (12/12), and a full run with `<defn>` backed by
the live daemon reproduces the headline (use 0→100%, 2894→689 tokens, 58→100% success,
~4.2×). The cheap action is a real go-to-definition, equal to pyrefly's, not a
static-resolver artifact. We use the static resolver for bulk runs (hermetic and
validated-equal) and the live daemon to confirm server equivalence.

The cost gap: the needed symbol's definition is buried in a ~370-line module; `<read>`
returns the whole file (~3500 tokens) while `<defn>` returns ~6 lines (~50 tokens) — the
same information at a fraction of the cost. Tasks are non-guessable (idiomatic API guesses
fail), so retrieval is genuinely required. The read-required family inverts this: the needed
symbol is unknowable without reading, so `<defn>` cannot solve it (the boundary control). We
report go-to-definition use rate, tokens-to-solve, and pass@1, with paired exact McNemar on
success and a paired token test, across seen and held-out task types.

## 4. Method: on-policy cost-aware imitation

**The dominance argument.** Because `<read X>` and `<defn X>` return the same information,
the minimum-cost action is `<defn X>` whenever `<defn>` covers the needed symbol. The
"expert" is therefore a free deterministic read→defn relabel, not a model. This dominance
holds only where `<defn>` covers the needed symbol. The relabel is applied exactly on the
suite's definition-sufficient tasks; coverage is supplied by task labels, not learned.
Discovering coverage in an unlabelled repo is out of scope here (§7).

**The on-policy round (DAgger round-0).**

1. Roll out the untrained agent with both `<read>` and `<defn>` available.
2. When the agent emits `<read>` for a non-editable file and the needed symbol is
   resolvable, drop the `<read>` step and let the agent emit `<defn sym>` instead, using
   its own symbol choice.
3. Continue the rollout from that point, keeping the agent's own subsequent actions.
4. Mix in read-first trajectories from read-required tasks so the boundary is represented.
5. LoRA fine-tune on the combined trajectories.

No gold action is injected. We relabel only the retrieval channel of the agent's own
behaviour. An earlier pilot that teacher-forced `<defn>` as the first action reached the
same result; the relabel confirms the effect survives when the action is the agent's own.

**Why on-policy is necessary.** Offline cloning trains on the teacher's state distribution.
The deployment distribution, where the expensive action is still available, is off-support,
so the cloned policy is unconstrained exactly where the preference must be expressed. The
cost preference is a choice the offline data never demonstrate.

## 5. Results

### 5.1 The efficiency win (C1)

Headline (mixed suite, real resolver, untrained PRE vs trained POST), definition-sufficient
tasks, **n=48**: `<defn>` use **0→100%**, `<read>` use **42%→0%**, success **0.65→1.00**
(McNemar exact p=1.5e-5, b=17/c=0), mean input tokens **3086→688 (4.5×)**, paired sign
p=2.2e-4 (POST cheaper on 37/48). This is a genuine on-policy relabel of the agent's own
retrieval; no gold action is injected.

**Relabel-only retest.** The same method run in isolation reproduces the headline: use
0→100%, tokens 3086→724 (4.3×), paired sign p=2.2e-4, n=48.

**Teacher-forced pilot.** An earlier pilot that teacher-forced `<defn>` as the first action
reached the same operating point. Token reduction is measured on the matched-outcome subset
(tasks both policies solve): 2108→675 (3.1×, paired sign p=2.7e-4, n=84). Success is over
all rollouts: 0.60→0.98 (McNemar exact p=6.2e-14, n=144). The pilot agreeing with the
genuine relabel shows the effect is the retrieval preference, not an artifact of which
action was forced.

**Isolation control.** To rule out that the saving merely reflects retrieval helping
success, we compare a model trained to retrieve via `<read>` against our definition-trained
model on the tasks both solve. At matched outcome the read-trained model spends 3191 input
tokens and the definition-trained model 684 (4.7× cheaper, definition cheaper on 31/40,
exact sign p=6.8e-4, n=40). Both models retrieve and solve; the only difference is the
action chosen. The saving is the cost preference itself, not retrieval versus guess.

### 5.2 Non-degeneracy: the boundary (C2)

On read-required tasks the read rate stays 100% and success rises (0.58→0.79 on the
standard suite; 0.54→0.83 with the real resolver). On many-symbol tasks the agent reads
once instead of issuing several `<defn>` calls. Overall `<defn>` use on the boundary is
about 50%, but it is always backed by a read: on name-hidden tasks the agent may emit a
definition first and then read; on many-symbol tasks it reads directly. Token count on
read-required tasks goes up (real-resolver 2632→4844) because the agent correctly pays the
read cost to solve work that genuinely needs it. The efficiency win is on
definition-sufficient tasks, not bought by under-reading the boundary.

### 5.3 What fails: default, prompting, and offline imitation (C3)

Four policies on the same task suite show that only on-policy training moves the operating
point:

- **Default:** 2% `<defn>` use on the definition-only suite, 0% on the mixed suite.
- **Explicit prompt:** still near 0–2% use; a 35B model stays at 0% even when told to prefer
  `<defn>`.
- **Offline rejection-sampling on cheap `<defn>` trajectories:** `<defn>` use stays near 0%
  and tokens do not fall, though general success rises slightly.
- **On-policy imitation:** 100% `<defn>` use and about 4.5× fewer tokens.

Offline demonstrations never show the expensive action available and the cheap one chosen,
so the cloned policy is unconstrained exactly where the preference must be expressed.

### 5.4 Corroboration with cost-reward RL

We also trained a cost-reward GRPO alternative: reward is solve-at-min-tokens, with
group-normalized advantage over the model's own action tokens. It reaches the same
cheap-retrieval operating point as the SFT relabel, but needs several on-policy rounds. A
single round under-trains (use 38%→6%, tokens 2048→3041); after four rounds the policy
converges to 86% use and 790 mean input tokens; on a clean held-out retest it lands at 86%
use, 663 tokens, 100% solved (baseline 38% use, 2048 tokens, 67% solved). The retest is small
(n=36). GRPO corroborates that a token-cost objective instills the same preference, but the
SFT relabel remains the headline recipe because it needs only one round.

### 5.5 Surface transfer

The three definition-sufficient task types never seen in SFT training (queue, cache, clamp;
n=12) behave like the trained types: `<defn>` use 0→100%, success 0.42→1.00, tokens
3775→722 (5.2×). These differ from training only in surface content, so they show
surface-transfer, not coverage-judging.

### 5.6 Coverage-judging

To test whether the agent can judge coverage itself, we built a suite where the task surface
is byte-identical across variants. In the sufficient variant, `<defn>` returns the needed
value inline. In the insufficient variant, `<defn>` returns a definition that references the
value through an indirection (a registry call or attribute assignment). The value is then
reachable only by a full `<read>` of the large module and by no further `<defn>`, which we
verify exhaustively against the resolver. Nothing the agent sees before retrieving
distinguishes the variants; the only way to decide is to call `<defn>` and inspect the
return.

On this suite the cost-trained 27B reads mostly when needed. On sufficient variants it uses
the cheap `<defn>` and reads only 17% of the time (100% solved, ~1.5k tokens). On insufficient
variants it reads 100% of the time and still solves 100%. The conditional read difference is
**J = P(read | insufficient) − P(read | sufficient) = +0.83**. The same J holds on a held-out
indirection mechanism the model never saw (attribute-injection), while an untrained 27B reads
indiscriminately (J = 0).

We rule out a form heuristic — "read whenever the returned definition references any name"
— with an adversarial control. In a sufficient variant the value is present in the returned
span but accessed through a local name, giving the same surface form as the insufficient
cases. The trained model reads on it only 6% of the time (similar to the 17% on plain
sufficient), versus 100% when the value is genuinely absent. So the read decision tracks
whether the value is actually present in the returned definition, not the surface form.

Caveats: one model (27B) on synthetic tasks with modest n (18 per variant); 17% reads on
sufficient variants show imperfect discrimination; and the insufficient signal is fairly
legible (the returned definition visibly references an external name), whereas real-repository
indirection is messier.

## 6. Related work

Our method is closest to on-policy imitation under distribution shift. GKD (Agarwal et al.,
ICLR 2024) distills a teacher under the student's own distribution. DAgger (Ross, Gordon &
Bagnell, AISTATS 2011) and cost-aware AggreVaTe (Ross & Bagnell, 2014) provide the
foundations for rolling out a learned policy and relabeling with an expert. Revisiting
DAgger for LLM agents (Li et al., 2025) applies the same idea to tool-using language models.
STaR (Zelikman et al., 2022) bootstraps reasoning from the model's own generated rationales;
we bootstrap a cost preference from the model's own trajectories instead.

Cost-aware tool use via RL is the alternative we corroborate but do not require. OTC-PO
(Wang et al., 2025) and IKEA (Huang et al., 2025) reward fewer or cheaper tool calls. Their
results motivate that a token-cost reward can learn the same preference we obtain by
on-policy imitation.

The closest prior work on language servers is RLCSF (Zhang et al., 2025), which rewards
compiler and LSP diagnostics during RL. RLCSF treats LSP feedback as a useful signal; we
find that, on our suites, the information in that feedback is redundant for a self-retrieving
agent. The residual value is retrieval efficiency, and we show that a lightweight on-policy
imitation step instills the preference for it where prompting and offline cloning do not.

## 7. Limitations

- **Coverage discovery — labelled in training, but discovered at test.** Definition-sufficiency
  is labelled by the task suite and used in the training mix, so the *training* preference is
  instilled given coverage. But §5.6 shows the trained 27B then *judges* coverage at test time
  on a surface-invisible suite — reading only when the retrieved definition is genuinely
  insufficient (J = +0.83, generalizing to a held-out mechanism) — so for a capable model the
  boundary is discovered per-instance, not merely supplied. The open scope limit is narrower
  than we first thought: it is whether this holds on *real-repository* indirection (messier and
  less legible than our synthetic cases) and below the 27B scale.
- **Synthetic tasks with a controlled cost gap.** The read-required boundary covers two
  reasons a full read is needed (name-hidden, many-symbol), not all, and the suites are not
  a real repository with ambiguous navigation.
- **Cross-scale transfer is a check, not a second headline.** The training was originally
  7B-only; we re-ran the same relabel pipeline on Qwen3.6-27B (a different generation and a
  reasoning model). The wild 27B is capable but reads everything (0% use, 96% read, 4058
  tokens, 96% success), and the relabel flips it to 100% use, 0% read, 100% success, 726
  tokens — 5.5× cheaper at matched success (n=24). So the preference is not a small-model
  artifact and the method transfers across a ~4× scale jump and a model-family change, but
  this is a lighter check (2–4 seeds vs the 7B's 12, default thinking-on config, the same
  definition-sufficient suite), not a fully-powered second headline.
- **Hermetic resolver.** `<defn>` is a real AST resolver over the live workspace with no
  oracle in the evaluation loop, but it is a static resolver rather than a running
  language-server client. The live-daemon equivalence is validated (§3), but bulk deployment
  against a live daemon is engineering we did not run at scale.
- **Statistics.** Token-magnitude and success are both significant on the pooled 12-seed
  sample (paired token p=2.7e-4, n=84; success McNemar p=6.2e-14, n=144). An earlier 4-seed
  sample was token-underpowered (p≈0.15), resolved by the extra-seed run.

## 8. Conclusion

On our synthetic suites, a language server does not help a self-retrieving agent by
providing information it cannot already read. The residual value is cheaper retrieval: a
go-to-definition returns the same symbol as a whole-file read at a fraction of the token
cost. A capable agent does not prefer the cheap action by default, by prompting, or by
offline imitation of it. On-policy imitation of the agent's own relabeled trajectories does
teach the preference, cutting input tokens several-fold while preserving success and a
read-when-needed boundary.

The broader lesson, if the result generalizes, is that whenever an agent has two actions that
return the same information at different cost, the cheaper one may need to be learned
on-policy. We expect this pattern to apply beyond language servers — to index lookups versus
document reads, targeted APIs versus broad scrapes, and similar retrieval choices — but that
remains to be shown empirically.
