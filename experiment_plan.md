# Asynchronous In-Stream LSP Feedback for Local Coding LLMs

**Status:** draft v0.5 — **Interaction-Model Pivot** (single-stream interleaved-async on a real coder; see §0). Supersedes the multi-stream substrate design below.
**Date:** 2026-05-29
**Hardware target:** single NVIDIA GB10 / DGX Spark (128 GB unified)
**Primary author:** Ian Barber

> **Changelog v0.5 → v0.5.1 — execution refined to "validate the interleaving recipe first" (2026-05-31).** Building the single-stream coding agent revealed two things: (a) a custom *continuous-stream* agent fights a 7B chat model at every level (edit format, EOS/turn-ending, and especially **parroting** — the model echoes anything spliced into its own assistant stream); (b) the parroting *is* the multi-stream conceptual problem — a single stream conflates generation and consumption, so **mid-generation live feedback (D) is SFT-gated**: the model must be *trained* to treat a mid-stream `‹delim›` as external input. Geiping multi-stream-on-a-coder (architectural separation) judged too heavy (~weeks). **Decision (Ian): validate Hooper/TM single-stream interleaving via SFT as a clean, isolated recipe FIRST** — a ladder of (R0) a consumption eval [reaction + parroting, headroom via a random injected value], (R1) self-distill data [‹info› interleaved + loss-masked], (R2) LoRA train + monitor reaction↑/parroting↓, (R3) then layer LSP + the coding agent. The §0 design (single-stream interleaved, A/C/D, self-distill) is unchanged; this only reorders execution to de-risk the load-bearing "trained interleaving works" assumption before the LSP+agent complexity. The agent (`scaffold/stream_agent.py`) + TaskEnv are retained for R3.
>
> **Changelog v0.4 → v0.5 — Interaction-Model Pivot.** L0 substrate validation showed `stream-qwen3-8b` is a parallel-cognition/monitorability CHAT model, not a code model: numerically healthy and fluent in conversation, but it goes silent / rambles on code and its SWE-bench-class ability is near the floor (driving it correctly produced correct logic on trivial functions but it cannot carry a coding-agent eval). No specialized coder shares its Qwen3-8B base (weight-merge ruled out), and retraining multi-stream on a coder is a heavy mini-project. **Pivot to option D: drop the Geiping multi-stream substrate; operationalize "in-stream feedback" as single-stream interleaved-async tokens (Hooper 2026; Thinking Machines "interaction models" 2026) on a genuinely strong coder (Qwen2.5-Coder family).** RQ1 is unchanged — "does async feedback delivery beat synchronous tool-call delivery at matched information content?" — only the *operationalization* changes. Single-stream interleaving removes the multi-stream "format" axis, so **C′ dissolves**: C (sync inline) vs D (async inline) isolates synchrony directly. The latency-replay protocol (§7.4) and its causal-validity gate (G3) carry over (layout adapted multi-stream→interleaved); payload-equivalence (G4) and the pyrefly daemon (G5) carry over unchanged. Throughput ceases to be a crisis (standard transformer + standard batched/vLLM inference). Full design in §0; sections §1–§17 below are retained as historical reference for the superseded multi-stream design.
>
> **Changelog v0.3.2 → v0.4 — throughput crisis resolved.** The G6 kill-switch (1.16 tok/s → 100–1370 weeks) was a **measurement + regime artifact**, not a real wall. A 7-phase single-sequence decoder optimization sprint (A–G: torch.compile, Flash/FlexAttention, static/in-place KV, CUDA graphs) moved throughput almost nothing (~5 tok/s flat) because it chased "copies" that were **CPU-self-time profiler artifacts** on this CUPTI-broken aarch64 box. A clean `torch.cuda.Event` microbench finally showed the decode is **matmul/weight-read bound at 15% of peak memory bandwidth — batch-starved** (a "row" is one task's 10 channels). The eval is a **throughput workload (≈900 independent trajectories)**, so the correct lever is **batched decode**: at B=16–32 aggregate throughput reaches **~30 tok/s at realistic ctx≈4096** (≈6×; ~79 tok/s / 15× at short ctx). **L4 is now tractable: descoped (50 tasks × 6 seeds × 5 conditions) ≈ 5–11 days, comfortably within the 21-day budget.** Full-scope L4 (200 × 9 × 5) is ~30–47 days at BF16 — needs INT8 weight-quant (≈2×, now compute-bound so it helps) or extended wall-clock; an L4-era decision, not a blocker. Consequences: §7.5 eval runs **batched at B=16–32**; §10 L4 adopts the **descoped scope** as primary; §6 records the resolution and the reusable in-place GQA-flex decoder; §11.1 G6 marked resolved; §14 R10 downgraded; §16 #8 resolved. **Process lesson (now standing guidance): before optimizing, establish the roofline, the workload shape (latency vs throughput + real batch), and one ground-truth number the profiler must reconcile against — `torch.cuda.Event`, not `torch.profiler` CPU time, on this hardware.**
>
> **Changelog v0.3.1 → v0.3.2.** Engineering-sprint findings + corrected throughput target. Phases A–C established that the substrate decoder is **launch-overhead-bound, not compute-bound**: attention is ~5% of per-row time; the binding cost is the per-row Python decode loop, which `torch.compile` cannot capture because the KV-cache cursor is a Python int (forces a fresh graph every row — 19,760 recompiles observed). Two attention-side routes (static-shape mask, FlexAttention) both left throughput at ~5 tok/s, confirming the diagnosis. **Throughput target corrected from 30 → ~12–16 tok/s** (the BF16 memory-bandwidth ceiling: 16 GB weights / ≈273 GB/s GB10 bandwidth ≈ 59 ms/forward floor); 30 tok/s reclassified as quantization-only and out of v1 scope. Sprint progress: G6 1.16 (contended) → Phase A 4.54 (clean + per-channel silence fix) → Phase B 4.95 (tensorized `sample_top_p` + vectorized mask) → Phase C ~5 (attention confirmed not the bottleneck) → Phase D in progress (cursor-tensorize → full per-row CUDA-graph capture, target the bandwidth ceiling). R10 updated; §16 #8 updated. If Phase D reaches ~12+ tok/s, L4 adopts a modest descope (50 tasks × 6 seeds).
>
> **Changelog v0.3 → v0.3.1.** L0 Wave 1 readouts integrated. G5 (pyrefly daemon partial-file probe): clean — daemon round-trip p95 6–21 ms (~30× margin under 200 ms budget); all 5 partial-file states bounded and recoverable; hard parse-validity gate downgraded to optional soft filter in §7.1; §13/§14 R3 medium → low. G6 (GB10 throughput): **kill-switch triggered** — 1.16 tok/s on contended GPU; L4 over budget by 33–460×; engineering sprint launched (target 30 tok/s via `torch.compile` + FlashAttention/FlexAttention kernel work); new §14 R10 added. G1-prep: harness wired; **chat-template-aware prompting and markdown-fence-aware completion extraction added as pre-G1 plumbing requirements** (stream-qwen3-8b's Output channel is chat-tuned). Multi-stream packing-factor-2 reality is 1.13× (not 2×) due to silence_penalty being Output-only — §7.4 notes per-channel silence-penalty as a packing prerequisite.
>
> **Changelog v0.2 → v0.3.** Substrate collapsed to 8B-only (`JonasGeiping/stream-qwen3-8b`) across all rungs after L0 Wave 0 surfaced that 1.7B/4B variants were never released and that 8B (dense) vs 27B (DeltaNet-hybrid) is an uncontrolled architecture transfer; 27B reserved for v2 follow-up. L4 seed count raised 6 → 9 (8B's lower per-token cost permits more seeds in the same wall-clock budget). Pyrefly daemon (`pyrefly lsp`) committed as the snapshot transport — one-shot CLI invocations measured at 0.4–2.4 s, incompatible with D's 200 ms debounce target. §7.2 criterion 3 (≤20 diagnostics on unmodified repo) reworded to require per-task `pip install -e .` + `pyrefly init` + `--python-interpreter-path` first; raw-CLI counts are dominated by spurious `missing-import` noise. Author list for arXiv 2605.12460 corrected to Su, Yang, Li, Geiping. §15 timeline shortened to ~8 weeks (down from 14–16). New §13/§14 entry: 8B-class result may not transfer to frontier-class — explicitly framed as v1 scope.
>
> **Changelog v0.1 → v0.2.** Added C′ condition (multi-stream + sync) to isolate format from synchrony; added pre-registered statistical analysis section using variance priors from Bjarnason et al. (2026); added L0 single-stream-degradation gate; matched-volume SFT for condition A; strengthened leakage probes with adversarial and counter-factual diagnostics; reframed contribution against newly-surfaced prior art (Ginart 2024; Hooper 2026; Gong 2025); Phase 0 teacher-rollout budgeted; quantization/LoRA specs added; pyrefly determinism screen added.

---

## 0. v0.5 — Interaction-Model Pivot (AUTHORITATIVE current design)

> Sections §1–§17 describe the superseded multi-stream design and are kept for provenance. Where they conflict with §0, §0 wins.

### 0.1 Why the pivot
The async-feedback hypothesis was operationalized via the Geiping multi-stream substrate (`stream-qwen3-8b`), the only released ≤27B multi-stream model. L0 validation found it is a parallel-cognition/monitorability model, not a coder: healthy and fluent in conversation (greedy and sampled), but it goes silent or rambles on code and cannot carry a SWE-bench-class agent. No specialized coder shares its Qwen3-8B base (weight-merge impossible); retraining multi-stream onto a coder is a heavy mini-project with no clean dense Qwen3 base. Meanwhile the field's two most recent takes on "models that consume environment signals mid-generation" — Hooper et al. (arXiv 2605.13360) and Thinking Machines' "interaction models" (2026) — both use **single-stream interleaved input/output**, not parallel channels. We therefore re-operationalize the same scientific question on a single-stream interleaved substrate and a model that can actually code.

### 0.2 Hypothesis (unchanged)
**RQ1:** Does delivering LSP diagnostics *asynchronously, interleaved into the agent's generation at the time they actually become available*, reduce edit-rework and improve task resolution relative to delivering the *same diagnostics synchronously at an edit/tool-call boundary*, holding base model and information content constant? Primary endpoint: rework-ratio (continuous, per-trajectory). Co-primary: pass@1 (paired McNemar).

### 0.3 Substrate
- **Model family: Qwen2.5-Coder (dense)** — `Qwen2.5-Coder-7B-Instruct` (L0/L1 dev), `-14B-Instruct` (L2/L3), `-32B-Instruct` (L4 headline). Apache-2.0, strong open coders (HumanEval ~88/~89/~92), GB10-runnable at BF16. A *single dense architecture across the whole ladder* — eliminates the dense-vs-DeltaNet arch-transfer confound that killed the 8B→27B story. (Qwen3-Coder-30B-A3B MoE is a faster-inference alternative for L4 if throughput demands it; introduces a dense→MoE transfer caveat, so dense is the default.)
- **Standard inference:** ordinary transformer, ordinary KV cache, ordinary batched/vLLM serving. The entire custom multi-stream decoder + its throughput sprint (§14 R10) are retired; throughput is no longer a research risk.
- **Capability gate (replaces G1):** confirm the chosen coder's standalone HumanEval/MBPP pass@1 matches published numbers on GB10 before building on it. (`scripts/d_capability_eval.py`.)
- **External reference:** `agentica-org/DeepSWE-Preview` (Qwen3-32B RL, 42–59% SWE-bench) as before.

### 0.4 The interleaved-async mechanism
A trajectory is a single token stream. Diagnostics are inserted inline as a delimited block, e.g. `‹diag›(severity,line,code,msg)…‹/diag›`, at a token position determined by the condition:
- **A — no-LSP.** No diagnostic blocks. Floor.
- **B — instructed tool-call.** Model emits an explicit "request diagnostics" action; the block is returned at the next position (model-initiated, single-stream). "Do agents use it when given the choice?"
- **C — forced sync post-edit.** After every `Edit`, the diagnostic block is inserted at the **edit boundary** (the next token position). Production-style synchronous hook.
- **D — async interleaved.** After an edit, the diagnostic block is inserted at `edit_position + round(measured_latency / ms_per_token)` — i.e. *mid-generation of whatever the agent did next*, reflecting real pyrefly latency. The model is trained to react to a diagnostic that refers to an edit it has already moved past.

Because every condition delivers identical inline tokens, **there is no "format" axis** — C vs D differ *only* in *when* the diagnostic appears. This is a cleaner synchrony isolation than the old C′ (which needed multi-stream to control format), so **C′ is removed**.

### 0.5 Conditions (revised)
| ID | Name | Delivery | Purpose |
|---|---|---|---|
| **A** | No-LSP | none | floor (matched-volume SFT) |
| **B** | Instructed tool-call | inline, model-initiated | use-vs-presence audit |
| **C** | Forced sync post-edit | inline at edit boundary | production sync baseline |
| **D** | Async interleaved | inline at latency-replayed position | **the hypothesis** |
| **E** *(stretch)* | Distilled async | as D, on-policy distilled from C teacher | RQ4 |

**Central comparison: D vs C** (isolates synchrony; format constant). Floor: A. Use audit: B. Leakage probes (D-noise / D-adversarial / D-counter-factual) unchanged from §11.2.

### 0.6 What carries over vs changes
- **Carries over:** payload normalization + the SHA-256 equivalence gate (G4) — `normalize_payload` is delivery-agnostic; the latency-replay reformat + causal-validity gate (G3) — the masking logic is identical, only the output *layout* changes from multi-stream grid to a single interleaved sequence; the pyrefly daemon client (G5); SWE-bench/SWE-Gym selection, harness, scaffolding; the statistical plan (§8/§9), now without C′.
- **Changes:** `training/reformat.py` emits an interleaved single-stream sequence (diagnostic block at `query_pos + latency_in_tokens`) instead of a `MultiStreamSequence` grid; the delivery layers (`lsp/delivery_*.py`) target an inline insertion offset rather than a side channel; no custom decoder, no channel/silence handling.
- **Retired:** the multi-stream substrate, the cognition-model driving, the decoder throughput sprint, C′, R10.

### 0.7 Related-work positioning
- **Hooper et al. (2026, arXiv 2605.13360):** single-stream clock-token interleaving for time-aware real-time agents (Q&A/voice). Same mechanism family; we differ in domain (SWE-bench/LSP) and the latency-replay training-data protocol.
- **Thinking Machines "interaction models" (2026):** "single interleaved token sequence (input₀ output₀ input₁ output₁…)", "time-aligned micro-turns", and an async background model whose "results stream back as produced and the interaction model interleaves these updates into the conversation." This is our async-LSP pattern with pyrefly as the background process. We differ by being a *controlled comparison* of sync vs async at matched content, in the coding/LSP domain, with the latency-replay construction. The convergence of two serious efforts on single-stream interleaving is evidence the operationalization is right.
- **Contribution restated:** (i) a controlled, information-content-matched comparison of synchronous vs asynchronous LSP feedback for coding agents; (ii) the latency-replay protocol for constructing causally-valid interleaved-async SFT data from synchronous-teacher rollouts; (iii) the diagnostic-stream evaluation methodology (leakage/adversarial/counter-factual). Survives a null result on (i).

### 0.8 Scaling ladder (revised)
| Rung | Model | Tasks | Seeds | Gate to next |
|---|---|---|---|---|
| **L0** | Qwen2.5-Coder-7B | 1 canary + capability baseline | 1 | capability ≈ published HumanEval; G3/G4 green on interleaved layout; canary (D acts on a diagnostic, A cannot) |
| **L1** | Qwen2.5-Coder-7B | 5 easy | 3 | causal-validity + payload audit pass; D-real > D-noise on rework-ratio |
| **L2** | Qwen2.5-Coder-14B | 20 | 3 | σ ≤ 3pp; **C vs D establishes direction or null** |
| **L3** | Qwen2.5-Coder-14B | ~50 filtered | 6 | effect stable; ablations meaningful |
| **L4** | Qwen2.5-Coder-32B | ~50 filtered + held-out | 6 | headline + generalisation |

Standard batched inference → L4 wall-clock is no longer budget-threatening; revisit timing after the L0 capability baseline.

### 0.9 Open questions for the pivot (decide before L1)
1. Interleaved diagnostic-block delimiter + tokenization (reuse Qwen chat special tokens vs new sentinels).
2. How the agent scaffold realizes "insert at token position" at *inference* time for D (the trajectory is generated, not pre-laid-out) — likely: run the agent, debounce-snapshot pyrefly, and splice the diagnostic into the context stream at the live position when it arrives, matching the training layout. Prototype at L0.
3. Whether B (instructed tool-call) and C (forced sync) need the same SFT exposure as D to be fair (matched-volume SFT, per §7.3 logic).
4. ~~Confirm `ms_per_token`~~ **RESOLVED 2026-05-30: 92.6 ms/token (10.8 tok/s single-seq) on Qwen2.5-Coder-7B.** Consequence: pyrefly latency (6–21 ms) is *sub-token* and a 200 ms debounce is only ~2 tokens, so realistic async offsets are small here. **Design implication:** the sync-vs-async contrast is as much about *delivery mode* (D = keep generating and receive; C = effectively pause at the edit) as raw token-offset; run a **latency sweep** (e.g. 0 / 2 / 8 / 32 token offsets) to characterize sensitivity rather than relying on one measured latency. The KV-cache splice mechanism is validated (`scripts/d_splice_prototype.py`).
5. Whether to keep a multi-stream arm at all as a secondary comparison (probably not for v1).

---

## 1. Motivation

Language servers help human programmers because feedback is **live**: a type error surfaces under the cursor while code is being written. For LLM coding agents, the equivalent loop is a tool call — the model finishes a step, requests diagnostics, waits for a response, and reacts. This serializes feedback that, for humans, is parallel.

The Su, Yang, Li, Geiping multi-stream architecture (arXiv 2605.12460) offers a way to make that feedback parallel: a model decoding across multiple time-synchronized streams can receive asynchronous environment signals on a side stream while emitting its primary output. The released artifacts ship 10 fixed channels named for cognitive functions (Analytical, Skeptical, Intuitive, …); the architecture supports environment-driven input via `gen.send(tok)` on the inference generator, which is the hook we use to inject LSP diagnostics into a repurposed channel. The paper lists "Tool Results / Code REPL / Notifications" as motivating use cases but does not implement them.

This project tests whether *asynchronous in-stream LSP diagnostics* — squigglies delivered on a parallel stream as the model writes — measurably outperform *synchronous post-edit tool-call diagnostics* at **matched information content**, on a real software engineering benchmark.

## 2. Positioning against prior work

The closest adjacent work falls in three buckets:

**Asynchronous tool use at the runtime level.** Ginart et al. (2024, arXiv 2410.21620) wrap a single-stream LLM in an event-driven FSM for voice agents; Gong et al. (2025, arXiv 2508.05298) stream outgoing function calls through a multi-channel scheduler for embodied robotics. Both place the asynchrony in an external runtime. We place it in the decoding substrate itself.

**Asynchronous I/O at the model level.** Hooper et al. (2026, arXiv 2605.13360) — the closest prior work in spirit — interleave clock tokens into a single-stream transformer to teach time-aware behaviour for real-time agents, reporting 1.3–2.2× wall-clock speedup on HotpotQA / voice. We differ in (a) substrate (multi-stream decode with a *separate* diagnostic channel vs inline clock tokens), (b) target (task-resolution capability vs human-perceived latency), and (c) domain (SWE-bench Verified vs Q&A/voice). Su et al. (2026, arXiv 2605.12460) provides the substrate but motivates it with cognitive-stream interpretability rather than environment I/O; we adapt one of their 10 cognitive channels as the diagnostic-input channel.

**LSP as RL signal.** Zhang et al. (2025, arXiv 2510.22907) use compiler and LSP diagnostics as RL *reward*. Their contribution is policy shaping; ours is in-context delivery form. The two are orthogonal and composable.

Our distinctive contribution is **(i)** a controlled, information-content-matched comparison of synchronous vs asynchronous feedback delivery on SWE-bench Verified, and **(ii)** the latency-replay protocol for constructing causally-valid multi-stream SFT data from synchronous-teacher rollouts (§7.4). Either is independently publishable; together they survive a null empirical result on (i).

## 3. Research questions

**RQ1 (primary).** Does an asynchronous in-stream diagnostics channel reduce edit-rework and improve task resolution relative to synchronous post-edit tool-call diagnostics, holding base model and information content constant?

**RQ2 (mechanism decomposition).** Of any C → D gap, how much attributes to *format* (multi-stream layout) vs *synchrony* (asynchronous delivery)?

**RQ3 (use vs presence).** Does the model meaningfully *use* the diagnostics stream, or is any effect attributable to additional fine-tuning compute?

**RQ4 (distillation, stretch).** Can a synchronous-LSP teacher be distilled into the asynchronous-stream student without losing the latency advantage?

## 4. Hypotheses (pre-registered)

- **H1.** Condition D achieves lower rework-ratio than Condition C′ on the SWE-bench Verified filtered subset, at matched information content. *(C′ is the matched-format synchronous control — see §5.)*
- **H2.** D's *time-to-first-correct-edit* (in emitted tokens) is shorter than C′'s.
- **H3.** D's pass@1 is no worse than C′'s; we predict equal or better.
- **H4 (falsifier).** D-real outperforms both D-noise and D-adversarial; D-real and D-counter-factual diverge in the expected direction. (If H4 fails, the model is not using the stream and the comparison is uninterpretable.) See §10.2.

Pass@1 is reported alongside **pass^3** (the probability all 3 seeds pass), per Bjarnason et al. (2026) recommendation.

## 5. Conditions

All conditions share substrate, scaffolding, evaluation harness, and a shared SFT ancestor pass.

| ID | Name | Stream usage | Delivery | Purpose |
|---|---|---|---|---|
| **A** | No-LSP baseline | Single-stream | None | Floor (matched-volume SFT, no LSP content) |
| **B** | Instructed tool-call | Single-stream | Model-initiated | "Do agents use LSP when given the choice?" |
| **C** | Forced post-edit hook | Single-stream | Auto after every edit | Production sync baseline (Claude Code-style) |
| **C′** | Multi-stream synchronous | Multi-stream | On side stream, no latency replay | **Isolates format from synchrony** (key v0.2 addition) |
| **D** | Interleaved async | Multi-stream | On side stream, with measured-latency replay | The hypothesis |
| **E** *(stretch)* | Distilled async | Multi-stream | As D, trained by on-policy distillation from C teacher | RQ4 |

**Central comparison:** D vs C′ (isolates synchrony). **Secondary:** C′ vs C (isolates format). **Floor:** A. **Use audit:** B. **Stretch:** E.

## 6. Substrate

- **Base model (all conditions, all rungs):** `JonasGeiping/stream-qwen3-8b` — dense Qwen3-8B backbone, ~16.4 GB BF16, Apache-2.0. Operated in single-stream mode for A/B/C and multi-stream mode for C′/D/E.
- **External reference (outside controlled comparison):** `agentica-org/DeepSWE-Preview` — one run, to anchor absolute pass rates against the published leaderboard. Absolute pass rates at the 8B class will be below DeepSWE; this is expected and the comparison serves as a sanity anchor, not a SOTA claim.
- **Scaffold:** mini-SWE-agent-style edit-test-iterate, frozen across conditions. Only the LSP-delivery layer varies.
- **LSP:** Pyrefly v1.0.0 (commit `2362c071caa576f9112781b5571f9e283cd52920`), **daemon mode (`pyrefly lsp`)** with incremental analysis. Per-task `pip install -e .` and `pyrefly init` at the base commit; `--python-interpreter-path` points at the per-task venv. CLI one-shot mode is not viable for D's snapshot loop (measured 0.4–2.4 s per invocation at L0 Wave 0).
- **Inference API:** `stream_generate_iter` returns a Python generator with `gen.send(tok)` for environment-driven input. `model.generate()` is intentionally disabled by the substrate; we use the streaming API uniformly across single-stream and multi-stream conditions. `transformers>=5.2` required.
- **Decode throughput (resolved 2026-05-29):** single-sequence decode is ~5 tok/s (~377 ms/row), memory-bound at 15% of GB10's ~273 GB/s. **Evaluation runs batched** (B=16–32 independent trajectories) — decode is weight-bandwidth-bound, so batching amortizes the 14 GB weight read and lifts aggregate productive throughput to **~30 tok/s at ctx≈4096 (B=16, 37 GB), ~79 tok/s at short context (B=32)**. A `torch.cuda.Event` microbench established matmul = 92% of GPU time, casts/copies ≈ 2% (the torch-profiler "56% copy" was a CPU-self-time artifact — CUPTI is broken on this aarch64 box; always time with `cuda.Event`). Reusable: an in-place GQA-FlexAttention decoder (`runs/g6_phase_f/patched/`, identity-verified, recompiles bounded) and the batched-sweep harness (`scripts/g6_batched_*.py`). INT8 weight-quant is available as a ~2× stacking lever for full-scope L4 (batching made the regime compute-bound, so quant helps).

### Why the same model across A/B/C/C′/D
Eliminates the capability-floor confound. Differences in performance attribute to *information delivery form*, not weights. The L0 single-stream-degradation gate (§11.1) checks that this is actually a fair fight against vanilla Qwen3-8B.

### Why 8B only, no 27B
Only `stream-qwen3-8b` (dense, 8B) and `stream-qwen3.5-27b` (DeltaNet-hybrid, 27B) are released; smaller variants exist only as training pipelines in the paper's `sec5_efficiency/`. The 8B and 27B substrates differ architecturally, so a scaling claim from 8B → 27B would be uncontrolled (an unknown amount of any observed effect would be attributable to the architecture change, not scale). We restrict v1 to 8B and reserve a 27B follow-up for v2 once the methodology is validated. The v1 contribution is the controlled comparison + the latency-replay protocol, not a SOTA claim — the 8B class is sufficient for both.

## 7. Methodology

### 7.1 LSP integration (pyrefly)

- **Transport:** `pyrefly lsp` daemon, one persistent process per task, incremental analysis. CLI one-shot mode is **not** used at runtime — L0 Wave 0 measured 0.4–2.4 s per invocation, incompatible with D's 200 ms debounce target. The daemon's IPC round-trip is what the 200 ms budget refers to; daemon throughput characterised in G5/G6.
- **Per-task setup:** at the SWE-bench base commit, run `pip install -e .` (or the project's documented install command) into a fresh per-task venv, then `pyrefly init` to generate a config, then start `pyrefly lsp` with `--python-interpreter-path` pointed at that venv. Without this, pyrefly returns hundreds of spurious `missing-import` diagnostics and the entire pipeline is dominated by noise.
- **Snapshot cadence:**
  - A: N/A.
  - B: on-demand (model-initiated) via LSP `textDocument/diagnostic` request.
  - C: after every `Edit` operation; LSP push.
  - C′: after every `Edit` operation, emitted onto the side stream immediately (no latency).
  - D: debounced — snapshot fires after ~200 ms of no token emission OR on stable hunk boundary, with payload delivered to side stream at `(snapshot_time + measured_latency)`. Latency is measured against the daemon, not the CLI.
- **Diagnostic payload (normalized, byte-identical across B/C/C′/D):** `(severity, line, code, message)` tuples, top-K by recency-of-edited-region (K=10 default; sweep at L2).
- **Partial-file handling:** *optional* soft parse-validity filter before snapshot in C′/D. G5 verified pyrefly's daemon parser is error-recovering and tolerant of mid-edit states (5/5 broken-state probes returned bounded diagnostics within ~3 ms and recovered cleanly on revert), so the snapshot loop can forward any state safely. The optional filter is an ergonomics tweak (skip `parse-error` diagnostics during in-progress edits) rather than a correctness gate.
- **Determinism screen:** for each candidate task in §6's selection, run pyrefly twice on the unmodified repo (with per-task env setup as above); require exact-match diagnostics. Drop non-deterministic tasks. L0 Wave 0 pilot (3 tasks): 3/3 byte-identical.
- **Payload-equivalence audit:** SHA-256 of normalized payload at every snapshot in B/C/C′/D; CI check asserts equality across conditions for the same trigger. Hard gate at L1 (G4 is the L0 precursor).

### 7.2 Selection criteria for the ~100-task subset

Filter SWE-bench Verified by:
1. Typed-friendly subprojects (django, sympy, scikit-learn prioritized).
2. Gold patch touches ≤3 files.
3. Baseline pyrefly diagnostic count on unmodified repo ≤20, **measured after per-task `pip install -e .` (or equivalent task-specific env setup) + `pyrefly init` config generation + `--python-interpreter-path` pointing at the per-task venv.** Raw-CLI invocation without env setup is dominated by spurious `missing-import` diagnostics (L0 Wave 0: 111/404/941 raw vs ~75 post-env on django) and is not a valid baseline.
4. **Pyrefly determinism: exact-match diagnostics across two runs on the env-prepared unmodified repo.**

Instance IDs frozen before any model evaluation. **Held-out subset:** an equal-sized 100-task subset drawn from SWE-rebench (post-2025 contamination-free split) for the L4 generalisation check.

### 7.3 Shared SFT ancestor

Before branching, single SFT pass on the stream model using ~2 000 SWE-style edit-test trajectories from SWE-Gym / R2E-Gym training splits (disjoint from Verified). No diagnostics in this pass — teaches scaffold rhythm only. All conditions descend from this checkpoint.

**Per-condition fine-tuning compute is matched.** Each downstream condition receives a LoRA-SFT pass on the *same* 2 000-trajectory teacher corpus, reformatted into its condition-specific layout. Condition A's pass uses the LSP-free version of the same trajectories (zero-content diagnostics token positions remain, payload elided) to control training volume.

**Training hyperparameters (subject to L0 micro-benchmark validation):**
- Quantization: BF16 throughout SFT (8B fits comfortably in GB10's 128 GB unified memory without aggressive quantization); BF16 at eval. NF4 reserved as fallback if observed memory pressure forces it.
- LoRA rank: 128 on `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (raised from 64 — 8B has headroom).
- Sequence length: 32 768 tokens (multi-stream packing factor 2).
- Effective batch: 32 sequences via gradient accumulation.
- LR: 1e-4, cosine, 3% warmup.
- Epochs: 2 (revisit after L1).

### 7.4 Multi-stream training data construction (D)

This is the load-bearing methodological contribution. Procedure:

1. **Teacher rollout (Phase 0).** Run C's scaffold with a strong teacher on ~2 000 training tasks. Default teacher: DeepSWE-Preview locally (cost: ~3–5 days on GB10 at DeepSWE's size class). Alternative: Claude Sonnet-class API (cost: ~$1–3k). Both reported at L2.
2. **Log:** agent tokens with token-level timestamps; LSP queries; LSP responses; **actual measured pyrefly latencies** on this hardware for the repos in question.
3. **Reformat to multi-stream layout:**
   - Agent tokens → `assistant` stream.
   - LSP responses → `diagnostics` stream, placed at `(query_emit_time + measured_latency_sample)`.
   - **Causal-validity gate:** mask the teacher's *synchronous* diagnostic response from the prefix when reformatting — otherwise the student sees both the delayed async signal and the original sync signal in training, guaranteed leakage. Unit-tested at L0.
   - **Tokenizer-rate adjustment:** latency offsets computed in *student-tokenizer* time, not teacher-tokenizer time.
4. **Latency-distribution check:** compare pyrefly latency distribution on the 2 000 training-task repos against the 100 evaluation-task repos. If KS distance > 0.1, reweight or resample to match.
5. LoRA-SFT on the reformatted corpus.

For C′: identical reformat without the latency offset (diagnostics on side stream, timestep = snapshot timestep).

**Per-channel silence-penalty (multi-stream packing prerequisite).** G6 surfaced that the substrate's bundled `stream_inference.generate()` applies `silence_penalty` only to the Output channel; the diagnostic channel (e.g. Analytical) stays silent ~85% of rows, yielding a packing-factor-2 reality of 1.13× rather than 2×. C′/D require silence_penalty to be applied to the diagnostic channel as well — a small patch to `stream_inference.py`. Without it, the multi-stream conditions pay the cross-stream forward-pass cost without realising the throughput benefit.

For E (stretch): on-policy distillation from D-student rollouts; teacher is C on matched prefix; KL-match token distributions.

### 7.5 Evaluation harness

Frozen scaffold; only LSP-delivery layer varies. Per-task: hard wall-clock cap, hard token-budget cap, hard turn cap, identical across conditions. **Token budget cap = task-median completion budget under A × 3**, pre-registered after L2 calibration. **pass@1-vs-budget curves** also reported.

**Batched execution (v0.4):** trajectories are decoded in batches of **B=16–32 independent (task, seed) streams** in lockstep on the shared step index, to escape the batch-starved single-sequence regime (§6). Clean for A/B/C/C′ (independent sequences). For D, each batch element carries its own LSP-diagnostic injection schedule — at each step, pending diagnostics are gathered into each element's side-stream slot (the per-element analog of `gen.send`), idle elements emit silence tokens. Batching is a pure throughput optimization; it does not alter any trajectory's content or the matched-information-content guarantees.

Output captured: full trajectory, all snapshots, all diagnostic events, final patch. Scoring: SWE-bench's own test runner.

## 8. Metrics

**Primary endpoint (pre-registered):** rework-ratio (`chars_deleted_after_first_write / total_chars_written`), continuous, per-trajectory.

**Co-primary (binary):** pass@1 on the filtered subset, with paired McNemar test across matched (task, seed) pairs.

**Secondary:**
- pass@3, pass^3 (per Bjarnason et al.: probability all 3 seeds pass) — consistency metric
- Edit-error cycles per task (edit → diagnostic-with-error → re-edit-same-region)
- Time-to-first-correct-edit (tokens)
- Diagnostic-to-fix latency (tokens between an error and the responsive edit)
- Token budget on solved tasks
- Wall-clock per task; tokens/s; GPU memory peak

**Reporting:**
- Bootstrap 95% CIs over tasks × seeds for all metrics.
- Holm–Bonferroni correction across secondary endpoints.
- Shapiro–Wilk normality check before any t-test.
- pass@1-vs-budget curves on a log-budget axis.

## 9. Statistical analysis plan (pre-registered)

**Variance prior (Bjarnason et al. 2026):** σ ≈ 1.5–2.0 pp on pass@1 even at T=0; single-run estimates vary by 2.2–6.0 pp across runs.

**Power analysis for the primary binary endpoint (pass@1, C′ vs D).** Paired McNemar across 100 tasks × 9 seeds at L4. Assuming p_C′ ≈ 0.20 (8B class, below 27B's anticipated ~0.30) and a target detectable difference of 3 pp, the per-task statistic is mean pass@1 over seeds (bounded discrete in {0, 1/9, 2/9, ...}); paired bootstrap over the 100 tasks yields ≥88% power at α=0.05 (two-sided), conditional on σ ≤ 2 pp. **If observed σ at L2 exceeds 3 pp, raise seeds to 12 at L4** — L2 gate is quantitative on this. Seed count uplift over v0.2 (6 → 9) absorbs the variance budget freed by dropping 27B (8B is ~3× cheaper per token, so 9 seeds at 8B is comparable wall-clock to 6 seeds at 27B).

**Power for rework-ratio (primary continuous endpoint):** estimated at L2 from observed variance on the 20-task × 3-seed L2 grid. Required n derived empirically; reported in L2 gate report.

**Unit of analysis:** per-task pass@1 averaged across seeds (not per-trial), per Bjarnason et al.

**Comparisons explicitly planned:**
- D vs C′ on rework-ratio (primary, RQ1+RQ2)
- D vs C′ on pass@1 (co-primary)
- C′ vs C on both metrics (decomposition: format-only effect)
- A vs all others (floor sanity)
- B vs C (use vs forced)
- D-real vs D-noise vs D-adversarial vs D-counter-factual (H4)

Holm–Bonferroni across the secondary metric family within each comparison.

## 10. Scaling ladder

All rungs use `JonasGeiping/stream-qwen3-8b` (v0.3 decision; see §6).

| Rung | Tasks | Seeds | Wall-clock | Gate to next rung |
|---|---|---|---|---|
| **L0** | 1 canary fixture + 10 SWE-Gym | 1 | ~1 week | All §11.1 gates green |
| **L1** | 5 easy tasks | 3 | ~2 days | Causal-validity unit test passes; payload SHA audit passes; D-real > D-noise on rework-ratio |
| **L2** | 20 tasks | 3 | 3–4 days | Observed σ ≤ 3 pp; **C′ vs D ordering establishes effect direction or null** |
| **L3** | ~50 filtered tasks | 6 | ~5–7 days | Effect direction stable; ablations meaningful |
| **L4** | ~50 filtered + held-out (descoped) | 6 | ~1 week (batched B=16–32) | Headline + generalisation |

**Publish-worthy floor:** L3 with full ablation suite. L4 is the headline.

**L4 scope (v0.4):** the **descoped grid (50 tasks × 6 seeds × 5 conditions ≈ 1,500 trajectories)** is now the primary L4 plan — batched decode puts it at ~5–11 days, comfortably within budget, with adequate statistical power (re-confirm on the L2 variance estimate). The full grid (200 × 9 × 5) is ~30–47 days at BF16; pursue it only if (a) the L2/L3 effect is marginal and needs the extra power, and (b) INT8 weight-quant (~2×) or extra wall-clock is authorized. Pre-register the descoped grid before L4.

**Note on substrate uniformity.** Original v0.2 ladder used 1.7B / 4B / 8B / 27B across rungs to amortise iteration cost. v0.3 collapses to 8B-only because only 8B and 27B variants are released and they have different architectures (see §6). Within-rung iteration cost is higher at 8B than at 1.7B, but eliminating the cross-architecture confound is worth more than the cycle-time savings would have been.

## 11. Correctness gates and probes

### 11.1 L0 — pre-flight gates (all must pass)
- **G1 — Single-stream-degradation check.** Vanilla Qwen3-8B vs stream-qwen3-8b in single-stream mode on HumanEval, MBPP, and 10 SWE-Gym tasks. If stream-qwen3-8b is materially worse (>2 pp degradation on HumanEval), substrate is handicapped — reconsider design (see §13 R4). Highest-information gate; run early. **Prompt-format fairness prerequisite:** stream-qwen3-8b's Output channel is chat-tuned and emits markdown-fenced code blocks; G1 must apply the model's shipped `chat_template.jinja` and the completion extractor must strip markdown fences before scoring. Without this, the dry-run pattern (stream 0/3 vs vanilla 1/3 on a 3-problem sample) reflects format mismatch, not capability, and R4 would trigger spuriously. Also: validate that the stream model's approximated T=0 (`temperature=1e-3, top_k=1`) reliably reproduces greedy behaviour, or wire `stream_inference.sample_top_p` for true T=0.
- **G2 — L0 canary fixture.** Hand-crafted task whose gold solution requires acting on a specific diagnostic. D must solve it; A must fail it. Other patterns → stream wiring broken. Final L0 integration test.
- **G3 — Causal-validity unit test.** For 10 reformatted teacher trajectories, verify the teacher's sync diagnostic response is removed from the D-formatted prefix.
- **G4 — Payload equivalence (SHA-256) audit** across B/C/C′/D for 10 fixed (prefix, edit) cases.
- **G5 — Pyrefly partial-file probe.** ✓ Complete (2026-05-27). Daemon round-trip p95 6.2 ms (121-line file) / 21.3 ms (1 674-line file); ~30× margin under 200 ms budget. All 5 partial-file states (unclosed string, dangling `def`, empty `if:`, mid-statement, unclosed paren) bounded and recoverable. No hard parse-validity gate needed; §7.1 reworded to optional soft filter.
- **G6 — GB10 throughput micro-benchmark.** ✓ RESOLVED (2026-05-29). The 2026-05-27 kill-switch (1.16 tok/s → 100–1370 weeks) was a measurement+regime artifact. A 7-phase single-sequence optimization sprint chased CPU-self-time profiler artifacts and moved nothing (~5 tok/s flat). A `torch.cuda.Event` microbench identified the decode as matmul-bound and **batch-starved**; **batched decode (B=16–32) reaches ~30 tok/s at realistic context**, making descoped L4 a ~1-week run (see §6, §7.5, §10, §14 R10). `transformers>=5.2` and `gen.send()` confirmed working.

### 11.2 Leakage probes (L1; repeat L3)
**Three variants of "fake D":**
- **D-noise.** Diagnostics replaced with random `(severity, line, code, message)` tuples sampled uniformly. Expected: D-real > D-noise.
- **D-adversarial.** Well-formed diagnostics but `line` and `code` permuted across diagnostics within the same trajectory (point to wrong locations). Expected: D-real > D-adversarial; D-adversarial < A (negative information).
- **D-counter-factual.** Plausible false-positive diagnostics (e.g., synthetic `unused-import` warnings at lines where the symbol is actually used). Expected: D-real does *not* respond to the false positive at a rate higher than chance.

**Revised H4:** D-real > D-noise (one-sided) **and** D-real > D-adversarial (one-sided) **and** D-real false-positive-response-rate not significantly different from D-noise on D-counter-factual injection.

### 11.3 L2 — additional probes
- **C-saturation probe.** Sweep C cadence: per-edit / per-2-edits / per-turn. Plot pass@1 and rework-ratio.
- **C′ vs D primary readout.** If C′ ≈ D at L2, the async hypothesis is empirically wrong. Kill or pivot before L3.
- **Variance estimation.** Observed σ on rework-ratio; size L3/L4 seeds.
- **Teacher-source robustness.** Compare D trained from DeepSWE teacher vs Sonnet-class teacher on 20 tasks; check effect-direction preserved.

## 12. Ablations (at L3 unless noted)

- **D with zeroed diagnostics stream** — controls for "is the SFT the source of gains?"
- **D synchronous vs asynchronous** — already covered by C′ vs D, but also run "D async forced to block on diagnostic" as third arm
- **LSP cadence sweep in C** (L2; also serves C-saturation probe)
- **Diagnostic payload sweep:** raw vs `(sev, line, code, msg)` vs severity-histogram only
- **D snapshot cadence sweep:** debounce 50 / 200 / 500 / 1000 ms
- **Stream width:** D with 2 streams vs 3 (+ scratchpad)
- **Joint C-cadence × D-debounce sweep** (the two are not independent)
- **No-shared-ancestor D** — train D from `stream-qwen3.5-27b` directly without the ancestor pass; measures whether the ancestor introduces a diagnostic-naive prior that handicaps D

## 13. Threats to validity and mitigations

| Threat | Mitigation |
|---|---|
| C → D conflates format and synchrony | **C′ condition** isolates format; D vs C′ is the central RQ1 readout |
| Stream-qwen3-8b's single-stream mode degraded vs vanilla Qwen3-8B | **L0 G1 single-stream-degradation gate**; if fails, restrict claim (27B not a fallback — see §6) |
| 8B-class result does not transfer to frontier-class models (27B+) | v1 contribution framed as methodology + 8B-class controlled comparison; 27B reserved for v2 follow-up; transfer not claimed |
| Training-volume confound (B/C/D get extra SFT, A doesn't) | A gets matched-volume no-op-LSP SFT pass |
| Information-content drift between conditions | Byte-identical payloads, SHA-256 audited at L0 (G4) and L1 |
| Pyrefly version skew or non-determinism | Pin version 1.0.0 + SHA `2362c071`; determinism screen on selection (L0 pilot: 3/3 clean) |
| Pyrefly snapshot too slow for D's debounce | Daemon mode (`pyrefly lsp`) committed at v0.3; CLI one-shot mode infeasible (0.4–2.4 s measured); G5 validates daemon-mode throughput |
| Repo-state non-determinism (test flakes) | Containerized envs; flake repeats with disagreement → drop task |
| Cherry-picked task subset | Pre-register subset + seeds before evaluation; **held-out 100-task subset from SWE-rebench** evaluated only at L4 |
| Teacher policy biases D's training distribution | Two-teacher robustness check at L2 (DeepSWE vs Sonnet-class) |
| Async-latency replay causally invalid | Mask teacher sync diagnostic from prefix; unit-tested at L0 (G3) |
| Trajectory-length asymmetry vs fixed token cap | Cap set as `median(A_budget) × 3`; pass@1-vs-budget curves reported |
| Power insufficient at 3 seeds | 6 seeds at L3, 9 at L4; quantitative σ gate at L2 (escalate to 12 if σ > 3pp) |
| SWE-bench contamination | Perplexity audit of Verified vs matched SWE-Gym tasks; SWE-rebench held-out subset |
| Cognitive channel (e.g. "Analytical") repurposed as diagnostic input is a soft architectural fit | Disclosed in §2 framing; G2 canary verifies the model can act on side-channel content |

## 14. Risk register

- **R1 (high).** Multi-stream training-data construction. **Mitigation:** measured empirical latencies, causal-validity unit tests, L1 audit.
- **R2 (low–medium).** GB10 throughput at 8B (downgraded from medium at 27B). **Mitigation:** L0 G6 micro-benchmark; verify `transformers>=5.2` stack on GB10.
- **R3 (low; downgraded from medium 2026-05-27 after G5).** Pyrefly on partial files. **Mitigation:** G5 confirmed pyrefly's parser is error-recovering and tolerant of mid-edit states; optional soft filter only.
- **R4 (medium).** Stream substrate single-stream capability degraded. **Mitigation:** L0 G1 gate; fallback to "given a multi-stream substrate" framing. 27B is *not* a viable fallback (different architecture; see §6).
- **R5 (medium).** Teacher rollout budget. **Mitigation:** budgeted in Phase 0; cheaper local DeepSWE option.
- **R6 (low).** SWE-bench Verified contamination. **Mitigation:** SWE-rebench held-out subset.
- **R7 (low).** Pyrefly v1.0 maturity. **Mitigation:** determinism screen drops unstable tasks; L0 pilot was clean.
- **R8 (medium).** 8B-class result does not transfer to frontier-class. **Mitigation:** v1 framed as methodology + 8B comparison; transfer claim deferred to a v2 follow-up at 27B.
- **R9 (low).** `seal-rg/streaming` repo has no LICENSE file (weights are Apache-2.0). **Mitigation:** clarify licensing with authors before publication; weights-only usage is unambiguous.
- **R10 (RESOLVED 2026-05-29; was high).** Substrate decode throughput. The G6 kill-switch (1.16 tok/s) and the entire 7-phase single-sequence optimization sprint (A–G) were chasing a non-problem: **CPU-self-time profiler artifacts** (CUPTI broken on aarch64) made cast/dispatch accounting look like a 56% "copy" bottleneck. A `torch.cuda.Event` microbench established the decode is **matmul/weight-read bound and batch-starved** (matmul 92%, casts 2%, 15% of peak BW at batch=1). **Resolution: batched decode** — the eval is a throughput workload (~900 independent trajectories), and B=16–32 lifts aggregate throughput to ~30 tok/s at realistic context. Descoped L4 ≈ 1 week (within budget). **Residual:** full-scope L4 (~30–47 days at BF16) would need INT8 weight-quant (~2×, now helps since batching made the regime compute-bound) or extra wall-clock — an L4-era decision. No longer gates downstream rungs.

## 15. Budget and timeline

Indicative, single GB10:

| Phase | Item | Duration | Cost |
|---|---|---|---|
| 0 | Teacher rollout (2k tasks): local DeepSWE OR Sonnet API | 3–5 days OR ~2 days | 3–5 GPU-days OR $1–3k API |
| 0 | Latency measurement on training repos | concurrent with above | — |
| 1 | L0 gates Wave 0+1 (G5/G6/G1-prep + substrate + skeleton + pyrefly) | ~5 days (2026-05-22 → 2026-05-27, complete) | — |
| 1.5 | **Decoder throughput investigation** (7-phase sprint + cudaEvent microbench + batched-decode probe) — RESOLVED via batching; ~5 days actual, mostly wasted on profiler artifacts (see §14 R10, log 2026-05-27→29) | ~5 days (done) | — |
| 1.6 | L0 gates Wave 2+ (G1 actual + G3 + G4 + G2 canary) — runs batched | ~3–4 days | — |
| 2 | L1 leakage probes + audit | 2 days | — |
| 3 | L2 variance estimation + C′-vs-D primary readout | 3–4 days | — |
| 4 | L3 main runs + ablations | ~5–7 days (down from ~2 weeks at 27B) | — |
| 5 | L4 headline + held-out + final ablations | 2–3 weeks (down from 3–4 weeks at 27B) | — |
| 6 | Write-up | 2 weeks | — |

**Total: ~7 weeks** to publish-ready at L3 (with full ablations); **~9 weeks** for L4 headline — *conditional on the engineering sprint reaching ≥30 tok/s*. v0.2 estimate (14–16 weeks at 27B) shortened by the 8B-only substrate decision; v0.3.1 adds ~1 week for the decoder sprint surfaced by G6. If the sprint plateaus at 10–30 tok/s, descope to ~50 tasks × 6 seeds and the timeline returns to ~7–8 weeks; below 10 tok/s the project pivots.

## 16. Open questions (decide before L2)

1. Final filtered subset composition (criteria fixed; specific instance IDs to draw).
2. Whether E (distillation) is in scope for v1 paper or reserved for a follow-up.
3. ~~Pyrefly daemon vs single-shot mode tradeoff for D's snapshot loop.~~ **Resolved v0.3: daemon (`pyrefly lsp`) committed.** CLI one-shot measured at 0.4–2.4 s; incompatible with 200 ms debounce. See §7.1.
4. Whether to include a non-Python language (e.g., TypeScript with `tsserver`) as a cross-language generalisation ablation at L4, or defer.
5. Default teacher: DeepSWE local vs Sonnet API. Two-teacher robustness check at L2 will inform this.
6. **Which of the 10 cognitive channels** (Analytical, Skeptical, Intuitive, Between, Curious, Void, Instinct, Synthesis) to repurpose for diagnostics. Decide before L1 SFT — choice affects pre-training prior on that channel. Default candidate: "Analytical" (least overloaded for code).
7. **When to revisit 27B as v2 follow-up.** Not part of v1 scope; criteria for unlock are (a) v1 published, (b) DeltaNet-hybrid inference stack matured, (c) reviewer ask or independent motivation.
8. ~~**Throughput sufficiency.**~~ **RESOLVED 2026-05-29 (see §6, §14 R10).** Batched decode (B=16–32) gives ~30 tok/s at realistic context; descoped L4 (50 × 6 × 5) fits the budget at ~1 week. Full-scope L4 would need INT8 quant or extra wall-clock — deferred to an L4-era decision. The pre-registered descoped grid is now the primary L4 plan (§10).

## 17. Publishability framing

The paper's contribution is positioned as:

1. **A latency-replay protocol** for constructing causally-valid multi-stream SFT data from synchronous-teacher rollouts — the methodological contribution that survives a null empirical result.
2. **The first controlled comparison** of synchronous-tool-call vs asynchronous-side-stream LSP feedback on a software engineering benchmark, at the 8B class, with matched information content, matched substrate, and matched training compute.
3. **Empirical artifacts:** trained checkpoint(s) for C, C′, D (and optionally E) on stream-qwen3-8b; training-data construction pipeline; evaluation harness; pre-registered protocol — all released.
4. **A diagnostic-stream evaluation methodology** (leakage / adversarial / counter-factual probes) that other researchers can reuse to validate side-channel models.

The contribution is *not* a new SOTA on SWE-bench Verified; absolute numbers will be below DeepSWE (which is larger). The substrate is restricted to 8B class in v1 because only `stream-qwen3-8b` (dense) and `stream-qwen3.5-27b` (DeltaNet-hybrid) are released and they differ architecturally; scaling to frontier-class is reserved for a v2 follow-up so that the v1 claim remains a *controlled* comparison rather than an uncontrolled architecture transfer.

## References

Maintained in `bibliography.md`. Core citations:

- Su, Yang, Li, Geiping. *Multi-Stream LLMs.* arXiv 2605.12460. (Code: github.com/seal-rg/streaming; weights: huggingface.co/JonasGeiping/stream-qwen3-8b and stream-qwen3.5-27b.)
- Hooper et al. *Speculative Interaction Agents.* arXiv 2605.13360.
- Ginart et al. *Asynchronous Tool Usage for Real-Time Agents.* arXiv 2410.21620.
- Gong et al. *GhostShell.* arXiv 2508.05298.
- Bjarnason et al. *On Randomness in Agentic Evals.* arXiv 2602.07150.
- Zhang et al. *RL from Compiler and Language Server Feedback.* arXiv 2510.22907.
- Gehring et al. *RLEF.* arXiv 2410.02089.
- Together AI / Agentica. *DeepSWE.* July 2025.
- Thinking Machines. *On-Policy Distillation.* Oct 2025.
- SWE-Bench Illusion. arXiv 2506.12286.
- seal-rg/streaming.
- Pyrefly (Meta).
