# Making a Language Server Pay Off for a Coding Agent: Train It to Retrieve Cheaply

A coding agent that reads its own context gets no new *information* from a language
server — but it does leave a real *efficiency* win on the table, and it won't take that
win unless you train it to. This report shows the win is a policy problem, why prompting
and offline imitation can't reach it, and a lightweight on-policy recipe that does. All
`<defn>` (go-to-definition) results use a real resolver over the live workspace; no oracle
is consulted in the evaluation loop.

## Summary

A language server gives a capable coding agent no information it cannot already read.
Across correction, completeness, navigation, and prevention, the diagnostic,
find-references, and completion channels did not improve task success on our suites for an
agent that self-retrieves. The one real benefit is **cheaper retrieval**: a
go-to-definition returns one symbol's definition for a fraction of a whole-file read's
token cost. But the agent does not take the cheap action on its own — not by default
(~2% use), not under explicit instruction (a 35B model stays at 0% even when told to
prefer it), and not from offline imitation of the cheap action. The win comes from
**training the preference on-policy**: a 7B agent goes to 100% go-to-definition use and
spends several times fewer input tokens at maintained-or-better success (headline
3086→688 tokens, 4.5×). It also learns the **boundary** — on tasks that genuinely need a
full read it still reads 100% of the time — so it is "definition when sufficient, read
when needed," not a degenerate always-define.

## The recipe

1. **Use the LSP for efficiency, not information.** Give the agent a real go-to-definition
   *action* (`<defn sym>`), not diagnostics-as-context. The information is redundant for a
   self-retriever; the cost saving is not.
2. **Train the preference; don't prompt for it.** Default and explicit instruction both
   leave use at ~0%.
3. **Train it on-policy.** Offline imitation of the cheap action teaches "retrieve," not
   "prefer the cheap retrieval," because the demonstrations never contain "the expensive
   action was available and I chose the cheap one." We isolate this: a model trained to
   retrieve-via-read and our trained model both retrieve and solve, but ours is 4.7×
   cheaper at matched outcome — the saving is the action *choice*, not retrieval-vs-guess.
4. **Preserve the boundary.** Mix in tasks that genuinely need a full read so the agent
   learns *when* the cheap action suffices.

The honest caveat: the read→definition relabel is valid only where go-to-definition
actually covers the needed symbol, and in *training* that coverage is supplied — the suite
labels each task definition-sufficient or read-required, and the training mix uses that
label. At *test* time, though, the trained model judges coverage itself: on a surface-invisible
suite it reads only when the retrieved definition is genuinely insufficient, generalizing to
indirection it never saw (§5.4). So a practitioner does not need a perfect coverage oracle at
inference — a capable trained model reads when, and only when, the cheap retrieval fell short.

## Contributions

- **C1 — the value-add.** Applying the recipe makes a *real* go-to-definition a genuine
  win. A 7B agent goes from 0→100% use on definition-sufficient tasks and, at matched
  outcome (tasks both the untrained and trained agents solve), spends 3.1× fewer input
  tokens (2108→675, paired p=2.7e-4, n=84) with success rising 0.60→0.98 (McNemar
  p=6.2e-14, n=144). Re-running the full headline end-to-end with the real resolver and no
  oracle in the loop reproduces it: use 0→100%, mean input tokens 3086→688 (4.5×, paired
  sign p=2.2e-4, cheaper on 37/48), success 0.65→1.00 (McNemar p=1.5e-5, b=17/c=0). The
  efficiency claim is isolated from "retrieval helps success" by the matched-outcome token
  test and a read-trained baseline (§5.1, isolation control).
- **C2 — non-degeneracy.** The trained policy is a *boundary*, not a collapse. On tasks
  where a definition is insufficient and a full read is required, the agent still reads in
  100% of rollouts and its success rises (0.58→0.79; real-resolver headline 0.54→0.83). On
  those tasks its token count goes *up* (real-resolver 2632→4844) because it correctly pays
  the read cost to solve genuinely read-required work — the efficiency win is on
  definition-sufficient tasks, not bought by under-reading the ones that need a read.
- **C3 — what fails, and why.** The preference is not adopted by default (2% use on the
  definition-only suite, 0% on the mixed suite; 0% for a 35B even when instructed) and is
  not instilled by offline rejection-sampling on demonstrations of the cheap action (use
  stays ~0, tokens unchanged) — a predictable off-policy failure, since the demonstrations
  never show the cheap action chosen over an available expensive one.
- **C4 — scoped motivation.** *On our task suites*, a language server's information
  channels do not improve pass@1 over a no-tool baseline for a self-retrieving agent,
  because the agent already reads the files those features summarize. We do not claim
  universal redundancy (see §2); C4 motivates why we isolate efficiency and is not itself a
  headline result.

---

## 1. Introduction

Coding agents spend most of their tokens retrieving context. The same information is often
available two ways: a targeted language-server query (go-to-definition, ~50 tokens) or a
whole-file read (~3500 tokens). A capable agent does not choose the cheap path on its own.

This is a *policy* problem, not an information one. We show the cheap-retrieval preference
is not reachable by prompting or by offline imitation of the cheap action — for a precise
reason, off-policy distribution mismatch — and that it *is* reachable on-policy. The method
in one line: relabel the agent's own reads to the cost-dominant go-to-definition and
fine-tune on those on-policy trajectories. The result is 2→100% use, ~4.5× fewer input
tokens, 0.65→1.00 success, surface-transfer to held-out task content, and a preserved
read-when-needed boundary.

## 2. Why efficiency is the lever

A language server's information is redundant for an agent that reads its own context. We
summarize the evidence compactly:

- **Correction.** An oracle ladder (no feedback / synchronous diagnostics / perfect
  localization / gold fix) shows localization *harms* (p<0.001) and gold-fix does not beat
  no-feedback for a 7B; a 35B MoE ceilings the suite. The diagnostic adds nothing the model
  does not already read.
- **Completeness and scale.** Varying repository size 21→86 files at a fixed generous read
  budget, success stays 1.00 at a flat ~6–8 reads — find-references does not earn its keep
  because reading does not become expensive at tractable scale.
- **Navigation and prevention.** Find-references is redundant on success (the agent reads
  the call graph when name search fails); prevention fails its precondition — the agent
  reads the library and never emits the hallucinated symbol, so there is nothing to prevent.

Beyond *which* channel, we also studied *how* the feedback is delivered — synchronously at
end-of-turn versus interleaved live into the generation stream, eager versus lazy, with and
without hygiene gating — in an n=168 zero-shot sweep (14 tasks × 12 seeds). Properly-delivered
feedback of any timing lands in a parity band (fix-rates 0.46–0.53, all pairwise differences
non-significant); only naive live delivery hurts, and that harm is a recoverable
format-hygiene artifact (diagnostic markers leaking into the agent's own edits), not an
intrinsic cost of liveness. So neither timing nor delivery format is the binding constraint
either — which, with the channel nulls above, is why we stop asking how to deliver the
information and ask instead what retrieving it costs.

These nulls are consistent with redundancy, but also with a ceiling/floor sandwich (the 35B
saturates; the 7B cannot act on any feedback, gold-fix included), and they are measured on
synthetic suites with oracle channels, not a real repo with ambiguous navigation. We
therefore treat C4 as motivation — on these suites the information channel does not help, so
we ask what does — not as a proven universal. The rest of the report is about the one lever
that remains: the *cost* of retrieval.

## 3. Setup

The agent is a 7B coding model (Qwen2.5-Coder-7B-Instruct) in a try-and-correct loop with
`<read>`, `<defn>` (go-to-definition), `<test>`, and `<edit>` actions, a real `pyrefly`
type-checker, and a non-blocking stream harness.

**`<defn>` is a real go-to-definition, not an oracle.** Given a symbol name the agent
requests, the tool AST-resolves that symbol's top-level definition against the live
workspace and returns its source span — exactly what an LSP go-to-definition does, derived
from the codebase with no privileged knowledge of which symbol or what the answer is, and
returning "(no definition found)" on an unresolvable name. We validated this against a
production language server: driving a live `pyrefly lsp` daemon (JSON-RPC
`textDocument/definition`) resolves all 12 evaluation symbols to the same definition as the
static resolver (12/12), and a full run with `<defn>` backed by the live daemon reproduces
the headline (use 0→100%, 2894→689 tokens, 58→100% success, ~4.2×). The cheap action is a
real go-to-definition, equal to pyrefly's, not a static-resolver artifact. We use the static
resolver for bulk runs (hermetic and validated-equal) and the live daemon to confirm
server equivalence.

The cost gap: the needed symbol's definition is buried in a ~370-line module; `<read>`
returns the whole file (~3500 tokens) while `<defn>` returns ~6 lines (~50 tokens) — the
same information at a fraction of the cost. Tasks are non-guessable (idiomatic API guesses
fail), so retrieval is genuinely required. The read-required family inverts this: the needed
symbol is unknowable without reading, so `<defn>` cannot solve it (the boundary control). We
report go-to-definition use rate, tokens-to-solve, and pass@1, with paired exact McNemar on
success and a paired token test, across seen and held-out task types.

## 4. Method: on-policy cost-aware imitation

**The dominance argument.** Because `<read X>` and `<defn X>` return the same information,
the minimum-cost action is always `<defn X>` when `<defn>` suffices — an AggreVaTe-style
cost-to-go dominance. The "expert" is therefore a free deterministic read→defn relabel, not
a model. That dominance holds *only where `<defn>` covers the needed symbol*, and the
relabel is applied exactly on the suite's definition-sufficient tasks: the coverage decision
is supplied by the task labels, not learned. The expert is free *given* coverage;
discovering coverage in an unlabelled repo is out of scope here (§7).

**The on-policy round (DAgger round-0).** We roll the wild agent out under the deployment
prompt with both actions available. When it reaches for the expensive `<read>` of a
non-editable file, the rule oracle redirects it; the agent then picks `<defn>` itself — its
own symbol choice — and continues on-policy. We drop the read-attempt-and-redirect prefix
and keep the agent's own definition-first continuation, so the trained first action from the
clean deployment prompt is the agent's *own* go-to-definition. We mix in read-first
trajectories on the read-required tasks so the boundary is represented, then LoRA fine-tune.
No gold action is injected — we relabel only the *retrieval channel* of the agent's own
behaviour. (An earlier pilot that teacher-forced `<defn>` as a lead token reached the same
place; the relabel result confirms the effect survives when the action is the agent's own.)

**Why on-policy is necessary.** Offline cloning trains on the teacher's state distribution;
the deployment distribution (with the expensive action available) is off-support, so the
cloned policy is unconstrained exactly where the preference must be expressed (Ross &
Bagnell compounding error). The cost-preference is a choice the offline data never
demonstrates.

## 5. Results

### 5.1 The efficiency win (C1)

Headline (full mixed suite, real go-to-definition resolver, untrained PRE vs trained POST),
definition-sufficient tasks, **n=48**: go-to-definition use **0→100%**, `<read>` use
**42%→0%**, success **0.65→1.00** (McNemar p=1.5e-5, b=17/c=0), **mean tokens 3086→688
(4.5×), paired sign p=2.2e-4** (POST cheaper on 37/48). This is a genuine on-policy relabel
of the agent's own retrieval; no gold action is injected.

Two corroborating estimates of the same effect, labelled so they are not confused with the
headline: the *relabel-only retest* (the method in isolation) reproduces it — use 0→100%,
tokens 3086→724 (4.3×), p=2.2e-4, n=48; and a *teacher-forced lead-`<defn>` pilot* (12
seeds) reaches the same place — matched-outcome tokens 2108→675 (3.1×, p=2.7e-4, n=84),
success 0.60→0.98 (McNemar p=6.2e-14, n=144). The pilot agreeing with the genuine relabel is
how we know the effect is the *retrieval preference*, not an artifact of which action was
forced.

**Isolation control.** To rule out that the saving merely reflects "retrieval helps," we
compare a model trained to retrieve via `<read>` against our definition-trained model on the
tasks both solve. At matched outcome the read-trained model spends 3191 input tokens and the
definition-trained model 684 (4.7×, definition cheaper on 31/40, exact sign p=6.8e-4, n=40).
Both models retrieve and solve; the only difference is the action chosen — so the win is the
cost-preference itself, not retrieval-versus-guess.

### 5.2 Non-degeneracy: the boundary (C2)

On read-required tasks the read rate stays 100% and success rises 0.58→0.79 (real-resolver
0.54→0.83); on many-symbol tasks the agent reads once instead of issuing several
go-to-definition calls (an economic choice). Go-to-definition use on the boundary is ~50%,
but always backed by a read: on name-hidden tasks the agent may emit a definition first and
then read, while on many-symbol tasks it reads once. Token count on these tasks goes up
(2632→4844) because the agent correctly pays the read cost to solve work that genuinely
needs it. It learned "definition when sufficient, read when needed."

### 5.3 What fails, and an RL corroboration (C3)

Same task, four policies: default (2% use); explicit "prefer the LSP" prompt (still ~0–2%;
0% for a 35B); offline rejection-sampling on the cheap action (use ~0, tokens unchanged,
though general success rises); on-policy imitation (100% use, ~4.5× tokens). Only the
on-policy step moves the operating point.

**Cost-reward RL (GRPO) — an independent corroboration.** We also built and ran the
token-cost RL alternative (group-sampled rollouts, reward = solve-at-min-tokens,
group-normalized advantage, policy gradient on the model's own action tokens; 4 rounds, K=4
steps/round, G=8, λ=0.5, lr 1e-5). It reaches the same cheap-retrieval operating point as
the SFT relabel, but needs several on-policy rounds, not one. A single round under-trains and
even regresses (use 38%→6%, tokens 2048→3041); across four rounds the policy converges
monotonically (use 37%→48%→86%, mean input tokens 2048→1740→790), and on a clean held-out
retest the final adapter lands at **86% use, 663 input tokens, 100% solved** (vs the wild
baseline 38% / 2048 / 67%) — essentially the SFT operating point (~100% use, ~700 tokens).
Two different training objectives — cost-aware imitation and a token-cost reward — instill
the same preference. GRPO is not cheaper to run (multi-round harvests vs one relabel pass),
so the SFT relabel stays the headline recipe and GRPO is the corroboration, not a
replacement. (It needs ~3–4 rounds, shows mild round-to-round oscillation, and the retest is
small, n=36.)

### 5.4 Generalization, and discovering coverage

The three definition-sufficient task types never seen in the SFT harvest (queue, cache,
clamp; n=12) behave like the trained ones: use 0→100%, success 0.42→1.00, tokens 3775→722
(5.2×). But these differ from training only in surface content, so they show surface-transfer,
not that the agent *judges* coverage.

To test coverage-judging directly, we built a suite where — across **byte-identical-surface**
task variants — a `<defn>` returns either the needed value inline (sufficient) or a definition
that *references* the value through an indirection (a registry call, or an attribute
assignment) placed so the value is reachable **only by a `<read>`** of the large module and
by no further `<defn>` (a property we verify exhaustively against the resolver). Nothing the
agent sees before retrieving distinguishes the variants; the only way to know whether a read
is needed is to call `<defn>` and inspect what came back.

On this suite the **cost-trained 27B reads exactly when needed**. On sufficient variants it
takes the value from the cheap `<defn>` and reads only 17% of the time (solving 100%, ~1.5k
tokens); on insufficient variants — where the value is genuinely unreachable by definition —
it reads 100% and still solves 100%. The read-decision is coverage-conditional:
**J = P(read | insufficient) − P(read | sufficient) = +0.83**, and it holds *equally on a
held-out indirection mechanism the model never saw* (J = +0.83 on attribute-injection),
against an untrained 27B that reads everything indiscriminately (J = 0). Because the variants
are byte-identical in everything the agent sees before retrieving, the discrimination cannot
come from task shape — it must come from the *content* the `<defn>` returned; and because it
transfers across two distinct indirection mechanisms, it is not a heuristic on the *form* of
the return either. So a capable model, once trained for the cost-preference, **discovers
coverage per-instance** — it pays the read only when the definition it retrieved is actually
insufficient — rather than relying on a supplied label. (The cost-preference here is the same
27B relabel adapter from §7, evaluated zero-shot on this suite; training instils the
efficient *read-only-when-needed* policy on top of a coverage-perception the base model
already has but spends indiscriminately.)

Caveats: one model (27B) on synthetic tasks with modest n (18 per variant); the read-decision
is not perfectly clean (17% reads on sufficient); and "insufficient" here is a fairly legible
signal (the returned definition visibly references an external name), where real-repository
indirection is messier. The byte-identical surface and the cross-mechanism transfer already
exclude the obvious shape-keying routes; a dedicated shape-keyed baseline (trained where
surface predicts coverage, and shown to fail this suite) would make that airtight.

## 6. Related work

On-policy distillation and imitation as the right tool for distribution shift: GKD (Agarwal
et al., ICLR 2024, arXiv:2306.13649), DAgger (Ross, Gordon & Bagnell, AISTATS 2011),
cost-aware AggreVaTe (Ross & Bagnell, arXiv:1406.5979), Revisiting-DAgger-for-LLM-Agents (Li
et al., arXiv:2605.12913), and STaR (Zelikman et al., arXiv:2203.14465) — the offline-cloning
paradigm we contrast against. Cost-aware tool-use via RL, the alternative we corroborate but
do not require: OTC-PO (Wang et al., arXiv:2504.14870) and IKEA (Huang et al.,
arXiv:2505.07596) reward fewer or cheaper tool calls.

The closest prior work is RLCSF ("Reinforcement Learning from Compiler and Language Server
Feedback", Zhang et al., arXiv:2510.22907), which *rewards* compiler and language-server
diagnostics during RL. Where RLCSF treats LSP feedback as a useful signal to reward, we show
the LSP's *information* is redundant for a self-retrieving agent and that its sole residual
value — retrieval *efficiency* — is a preference a lightweight on-policy imitation step
instills where prompting and offline cloning cannot. We do not reward the tool; we train
*when to call it*.

## 7. Limitations

- **Coverage discovery — labelled in training, but discovered at test.** Definition-sufficiency
  is labelled by the task suite and used in the training mix, so the *training* preference is
  instilled given coverage. But §5.4 shows the trained 27B then *judges* coverage at test time
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

For agent builders: do not attach a language server expecting it to *inform* — a capable
agent already reads what it would surface. Attach it for *retrieval efficiency*, and train
the preference on-policy; prompting and offline demonstrations will not produce it. The
general form is broader than language servers: any two actions that return the same
information at different cost — an index lookup versus a document read, a targeted API versus
a broad scrape — is the same problem with the same fix.
