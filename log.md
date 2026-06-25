# Experiment Log

Chronological log of decisions, runs, and findings. Each entry dated; rationale captured for future re-reading.

---

## 2026-05-22 — Project kickoff

**Activity:** Initial scoping and design.

**Decisions:**
- Project framed around RQ1: does async in-stream LSP feedback beat sync post-edit tool-call LSP at matched content?
- Hardware target locked: single GB10 / DGX Spark, 128 GB unified.
- Benchmark: SWE-bench Verified filtered subset (~100 tasks).
- Substrate: `stream-qwen3.5-27b` (Geiping et al., arXiv 2605.12460) used for **all** conditions (single-stream for A/B/C, multi-stream for D). DeepSWE-Preview as external reference only.
- LSP: Pyrefly (selected over pyright for speed and engineering confidence).
- ProgramBench dropped — too hard, no signal at our scale.

**Rationale for same-model-across-conditions:** eliminates the capability-floor confound. Differences attribute to delivery form, not weights.

**Artifacts created:** `experiment_plan.md` v0.1.

---

## 2026-05-22 — Critical review (max-effort agent)

**Activity:** Dispatched Opus subagent for independent critique of v0.1.

**Top findings (severity-ordered):**
1. **C vs D conflates format and synchrony.** Need a C′ condition: multi-stream layout + sync delivery. Without it, a positive C-vs-D result is causally ambiguous.
2. **Statistical power likely insufficient.** 100 tasks × 3 seeds won't reliably detect 2–5pp pass@1 effects given known SWE-bench variance (1.5–3pp single-run SD per arXiv 2602.07150). Recommend ~6 seeds and pre-registered McNemar power analysis.
3. **Novelty framing contestable.** Three adjacent papers missed: Ginart 2410.21620 (Salesforce, async tool use), Hooper 2605.13360 (Berkeley, speculative interaction), GhostShell 2508.05298 (streaming function calls). Reframe contribution as controlled comparison + latency-replay methodology, not "first implementation."

**Other catches:**
- Single-stream-degradation check at L0 (vanilla Qwen3.5-27B vs stream-qwen3.5-27b on single-stream benchmarks).
- A needs matched no-op SFT to control training-volume.
- Stronger leakage probe: adversarial + counter-factual diagnostics, not just noise.
- Causal-validity bug in async-latency replay: mask teacher's sync diagnostic from prefix.
- Teacher rollout cost unbudgeted (~$1–3k API or 1–2 GPU-weeks).
- Pyrefly determinism screen for task selection.
- Trajectory-length asymmetry vs fixed token cap.

**Decision:** Adopt all 10 reviewer recommendations. Revise plan to v0.2.

---

## 2026-05-22 — Lit review of newly-surfaced prior art

**Activity:** Dispatched background agent to review Ginart 2410.21620, Hooper 2605.13360, GhostShell 2508.05298, and Bjarnason et al. 2602.07150.

**Findings:**
- **Ginart (Salesforce, 2024):** runtime FSM wrapping single-stream LLM for voice. System demo. Reframe distance: architectural (we modify the model) vs runtime (they wrap it).
- **Hooper (Berkeley, May 2026):** closest prior work. Clock-token interleaving for inline async I/O; 1.3–2.2× speedup on HotpotQA/voice. Reframe distance: single-stream + clock tokens (theirs) vs multi-stream decode + side channel (ours); latency target (theirs) vs capability target (ours); generic tools vs LSP.
- **GhostShell (Gong et al., Aug 2025):** robotics streaming function calls, multi-channel scheduler. Reframe distance: outgoing commands (theirs) vs incoming diagnostics (ours); external scheduler vs architectural integration.
- **Bjarnason et al. (Feb 2026):** **methodological backbone.** Variance prior σ ≈ 1.5–2 pp on pass@1 even at T=0; single-run range 2.2–6.0 pp. Use paired McNemar; report pass^k alongside pass@k.

**Decisions adopted into v0.2:**
- Variance prior fixed: σ ≈ 2 pp for power calc.
- Primary endpoint changed from pass@1 to **rework-ratio** (continuous, per-trajectory, higher power).
- Paired McNemar for pass@1 between conditions on matched tasks/seeds.
- Add pass^3 as a consistency metric per Bjarnason recommendation.
- L4 seeds raised from 3 → 6 based on σ prior.
- L2 includes explicit variance-estimation pass on rework-ratio before promoting.

**Artifacts updated:** `bibliography.md` v0.2 (full entries with full author lists, summaries, distinguishing dimensions for each reframe target).

---

## 2026-05-22 — Plan revision v0.2

**Activity:** Revised `experiment_plan.md` from v0.1 → v0.2 incorporating all 10 reviewer recommendations and the citation language from the lit review.

**Material changes from v0.1:**
1. **New condition C′** (multi-stream format + sync delivery) to isolate format from synchrony confound.
2. **Pre-registered statistical analysis section** added: McNemar paired test on pass@1; rework-ratio as primary endpoint; bootstrap CIs over tasks × seeds; Holm-Bonferroni across secondaries; pass^3 reported alongside pass@k.
3. **L0 single-stream-degradation gate**: vanilla Qwen3.5-27B vs stream-qwen3.5-27b on HumanEval/MBPP + 10 SWE-Gym tasks. Must pass before any other work.
4. **Condition A matched-volume SFT** (no-op-LSP pass on the same trajectories used for B/C) to eliminate training-volume confound.
5. **Stronger leakage probes**: noise + adversarial (wrong location) + counter-factual (plausible false-positive). H4 revised.
6. **Causal-validity fix**: mask teacher's synchronous diagnostic response from D's training prefix during reformat.
7. **Reframed contribution** as controlled comparison + latency-replay methodology; cites Ginart / Hooper / GhostShell / Bjarnason explicitly; "first implementation" claim removed.
8. **Phase 0 teacher rollout budgeted** in §14: ~$1–3k API or 1–2 GPU-weeks.
9. **Quantization / LoRA / seq-len / batch** specified in §7.3.
10. **Pyrefly determinism screen** added to §6 task-selection criteria.
11. **L4 seeds raised from 3 → 6**; per-task pass@1 averaged over seeds is the unit of analysis.
12. **Token-budget reporting**: pass@1-vs-budget curves alongside fixed-cap pass@1.

**Status:** v0.2 saved. Ready for execution planning at L0.

---

## 2026-05-26 — L0 Wave 0: substrate verification, repo scaffold, pyrefly determinism pilot

**Activity:** Resumed project after weekend; dispatched three Wave 0 subagents in parallel.

**Subagent A — HF substrate availability:**
- `JonasGeiping/stream-qwen3.5-27b` confirmed: 53.8 GB BF16, Apache-2.0, **DeltaNet-hybrid** architecture, fits GB10.
- `JonasGeiping/stream-qwen3-8b` confirmed: 16.4 GB, **dense Qwen3-8B** backbone, Apache-2.0.
- **No `stream-qwen3-1.7b` or `stream-qwen3-4b` released** (paper trains them in `sec5_efficiency/` but no weights shipped). L0/L1/L2 rungs in v0.2 plan assumed those — that assumption is dead.
- Multi-stream inference API works via `stream_generate_iter` returning a Python generator; `gen.send(tok)` is the supported hook for injecting environment signals (LSP diagnostics) into a side stream. `model.generate()` intentionally disabled.
- 10 fixed channels: User, Output, Analytical, Skeptical, Intuitive, Between, Curious, Void, Instinct, Synthesis — named for cognition, not tool-I/O. Repurposing one for diagnostics works via `gen.send()` but is an adaptation, not a clean fit.
- Code repo: github.com/seal-rg/streaming (no LICENSE file — flag for pre-publication resolution).
- Paper authors: Su, Yang, Li, Geiping (not Geiping et al. — bibliography needs author-list correction).
- Stack: `transformers>=5.2` (bleeding-edge — GB10 compatibility to verify in G6).

**Subagent B — Repo skeleton:** Created `configs/`, `harness/`, `scaffold/`, `lsp/`, `eval/`, `training/`, `scripts/`, `runs/`, `tests/` plus `.gitignore` and a stub `README.md`. No code, no package metadata, no git init. Clean.

**Subagent C — Pyrefly determinism pilot:**
- `pyrefly 1.0.0` installs via `pip install pyrefly` (aarch64 manylinux wheel; works on GB10). SHA `2362c071caa576f9112781b5571f9e283cd52920`.
- **Determinism is clean** on django/sympy/scikit-learn (3/3 byte-identical across two runs). §7.1 determinism screen is realistic.
- **Latency: D's 200 ms debounce target is NOT achievable with one-shot CLI invocations.** Cheapest case is ~400 ms; sympy is 2.4 s. Daemon (`pyrefly lsp`) mode is required.
- **§7.2 criterion 3 (≤20 diagnostics) is broken without per-task env setup.** Raw counts were 111/404/941, dominated by `missing-import`. Even after `pip install -e .` + `--python-interpreter-path`, django went 111→75 (still > 20). Screen needs per-task env setup + `pyrefly init` before the filter is meaningful.
- Correct CLI flag is `--python-interpreter-path` (not `--python-interpreter`).
- Reusable pilot script saved to `scripts/pyrefly_determinism_pilot.py`; artifacts under `runs/pyrefly_determinism/`.

**Decisions adopted (revising plan v0.2 → v0.3):**

1. **Substrate ladder collapses to 8B-only.** `JonasGeiping/stream-qwen3-8b` is the substrate for **all rungs**, L0 through L4. 27B dropped entirely as primary; reserved for a v2 follow-up. Rationale: only 8B and 27B are released; they have different architectures (dense vs DeltaNet-hybrid), so a cross-architecture scaling claim would be uncontrolled. Going 8B-only eliminates the transfer risk and makes the controlled comparison cleaner. Cost: paper headline is now an 8B-class result, not a 27B SOTA chase — but the contribution was always the methodology, not absolute numbers.
2. **L4 seeds raised from 6 to 9.** 8B is ~3× cheaper than 27B per-token, so we can afford more seeds in the same wall-clock budget. Quantitative σ gate at L2 may push to 12.
3. **Pyrefly daemon mode (`pyrefly lsp`) is committed for D's snapshot loop.** §16 open question #3 resolved.
4. **§7.2 criterion 3 reworded:** ≤20 diagnostics measured **after per-task `pip install -e .` and `pyrefly init` config generation**, with `--python-interpreter-path` pointing at the per-task venv.
5. **§13 threats updated:** "single-stream degradation vs vanilla Qwen" now compares vanilla **Qwen3-8B** vs stream-qwen3-8b. New threat added: "8B-class result may not transfer to frontier-class models" — mitigated by methodology framing + reserved 27B follow-up.
6. **§14 risk register updated:** R2 (GB10 throughput at 27B) → reworded for 8B (lower risk). R8 (frontier-class transfer) added.
7. **§15 budget/timeline shortened:** at 8B, L3 drops to ~5–7 days and L4 to ~2–3 weeks (down from ~2 weeks and 3–4 weeks at 27B). Total revised estimate ~8 weeks to L4 headline.
8. **Bibliography author-list correction:** Su, Yang, Li, Geiping for arXiv 2605.12460.

**Open follow-ups for Wave 1:**
- G6 throughput micro-benchmark on stream-qwen3-8b (validates `transformers>=5.2` on GB10).
- G5 pyrefly partial-file probe (daemon-mode this time).
- G1 prep (download vanilla Qwen3-8B + stream-qwen3-8b; wire HumanEval/MBPP; pick 10 SWE-Gym tasks).

**Artifacts updated:** `experiment_plan.md` v0.2 → v0.3; `bibliography.md` author list for arXiv 2605.12460.

---

## 2026-05-27 — L0 Wave 1: G5 / G6 / G1-prep complete; throughput kill-switch triggered

**Activity:** Three Wave 1 subagents returned. G5 and G1-prep were strong; G6 surfaced a project-scale throughput problem.

**G5 — pyrefly partial-file probe (daemon mode):**
- Daemon `pyrefly lsp --indexing-mode lazy-blocking` round-trip latency **p95 = 6.2 ms** on a 121-line file, **21.3 ms** on a 1 674-line file. **~30× margin under the 200 ms debounce budget.**
- All 5 partial-file states (unclosed string, dangling `def`, empty `if:`, mid-statement, unclosed paren) returned bounded diagnostics within ~3 ms and recovered cleanly after revert. Pyrefly's parser is error-recovering; type analysis around broken regions continues.
- **No hard parse-validity gate needed.** Plan §7.1 wording softens to "optional soft filter". §13 R3 downgrades medium → low.
- Minimal stdio JSON-RPC LSP client (~150 LOC, no `pygls` dep) at `scripts/g5_pyrefly_partial_file.py`. Reusable for production C′/D.
- One quirk: pyrefly emits a non-LSP-spec `data: "committing-transaction"` field per diagnostic. §7.1 normalizer projects to `(severity, line, code, message)` so it's harmless.

**G1-prep — models, harness, SWE-Gym selection:**
- Both `Qwen/Qwen3-8B` and `JonasGeiping/stream-qwen3-8b` staged at `/mnt/nas/hf-cache/`. **Wave 2 must export `HF_HOME=/mnt/nas/hf-cache` before any model load.**
- Stack working on GB10/aarch64/Python 3.12.3: `torch 2.11.0+cu128`, `transformers 5.9.0` (≥5.2 ✓), `accelerate 1.13.0`. `trust_remote_code=True` required for the stream substrate.
- HumanEval+MBPP runner at `harness/single_stream_eval.py`; dual-API (vanilla `model.generate` and stream `model.stream_generate`).
- SWE-bench-style P2P/F2P harness at `harness/swegym_loader.py`. **dask-10027 baseline-green validated end-to-end.**
- 10 SWE-Gym tasks frozen at `runs/g1_prep/swegym_tasks.json` (bokeh, conan, dask×2, hydra×2, modin, pandas, pydantic, mypy; 3 easy / 4 medium / 3 hard by gold-patch LOC).
- **Critical G1-fairness threat surfaced:** `stream-qwen3-8b`'s Output channel is **chat-tuned** (emits markdown-fenced code blocks), not raw prompt continuation. Dry-run scored 0/3 HumanEval (vs vanilla's 1/3) entirely because of prompt-format mismatch, not capability. **If Wave 2 fires G1 without applying the model's `chat_template.jinja` and stripping markdown fences in the completion extractor, R4 triggers spuriously.**
- Stream model has no exposed T=0 path; approximated with `temperature=1e-3, top_k=1`. Wave 2 should sanity-check determinism, or wire `stream_inference.sample_top_p` directly.
- Python 3.12 incompatibility for older SWE-Gym task commits (`hydra-1456`, `pydantic-4882`, possibly others). Wave 2 needs per-task Python 3.10 envs (SWE-bench Verified harness norm).
- SWE-Gym's typed-Python pool is *not* django/sympy/scikit-learn (those are SWE-bench Verified); the substitution to pandas/pydantic/mypy/dask/modin/hydra/bokeh/conan is correct for G1's purpose.

**G6 — GB10 throughput micro-benchmark — KILL-SWITCH TRIGGERED:**
- Single-stream throughput on `stream-qwen3-8b`: **1.16 ± 0.15 tok/s** (Output channel productive tokens), peak 16.85 GB, BF16.
- Multi-stream (Output + Analytical) throughput: **1.31 ± 0.24 tok/s combined**. Packing-factor-2 reality is **1.13× not 2×** because `silence_penalty` is applied only to Output; Analytical stays silent ~85% of rows. True packing-2 requires per-channel silence policy.
- `gen.send()` injection hook validated — LSP-diagnostic injection plumbing is sound.
- transformers 5.9.0 + torch 2.11.0+cu128 stack works on GB10/aarch64 (BF16 matmul on first call; no friction).
- **L4 extrapolation:** ~100 weeks (optimistic constant-TPS) to ~1370 weeks (realistic linear-KV-cache growth) vs **3-week budget** — off by 33× best case, ~460× worst.
- **Root cause:** `stream_inference.generate()` is a reference Python implementation. Python-built attention mask in the per-row hot path, no `torch.compile`, no Flash-attention path for the block-causal cross-stream mask, no kernel fusion. ~1 row/s is 25–50× under what an 8B model on GB10 should yield.
- **Caveat:** G6 ran while G1-prep was also using the GPU (concurrent dispatch). 1.16 tok/s is a *contended* measurement. Empty-GPU re-measure is the first task of the engineering sprint.

**Decision — engineering sprint, not descope:**
- Stop criterion: **30 tok/s** (closes the L4 budget gap fully; keeps v0.3 plan intact: 100 tasks × 9 seeds × 5 conditions × ~8k tokens).
- Phase A (in progress): empty-GPU re-baseline + `torch.compile` probe + per-channel silence fix + `torch.profiler` to identify the real bottleneck. Phase A target: ≥5 tok/s (proves easy wins moved the needle).
- Phase B (dispatched after Phase A profile): parallel investigation of FlashAttention-3 vs FlexAttention (PyTorch 2.5+ score_mod is purpose-built for non-standard masks like the cross-stream block-causal one). User's intuition: kernel work is plausible.
- Hard fallbacks if 30 tok/s unattainable: 10–30 tok/s → modest descope (50 tasks × 6 seeds); <10 tok/s → larger descope or substrate pivot.
- L0 timeline extends past the 1-week budget; this is the honest call.

**Plan updates queued for v0.3.1:**
- §7.1 LSP: parse-validity gate softens to "optional soft filter" (G5).
- §7.4 / §11.1: per-channel silence-penalty noted as a multi-stream packing prerequisite (G6 sub-finding).
- §11.1 G1: chat-template-aware prompting + markdown-fence-aware completion extraction added as pre-G1 plumbing requirements (G1-prep).
- §11.1 G5: parameters captured (daemon-mode latency p95 6–21 ms; tolerant to mid-edit states).
- §13/§14 R3 (partial-file pyrefly): medium → low.
- §13/§14 new R10: decoder throughput on this substrate is well below hardware ceiling; engineering sprint underway.
- §15 budget: add Phase A/B engineering-sprint line item.
- §16 open questions: add "throughput sufficiency" pending Phase A/B.

**Artifacts:**
- `/home/ianbarber/Projects/Streams/runs/g5_partial_file/` (latency.json, partial_file_probe.json, summary.md, stress_forms_models/)
- `/home/ianbarber/Projects/Streams/runs/g6_throughput/` (env.txt, single_stream.json, multi_stream.json, gen_send_check.txt, extrapolation.md)
- `/home/ianbarber/Projects/Streams/runs/g1_prep/` (weights.txt, swegym_tasks.json, swegym_validation.md, humaneval_dryrun.txt, mbpp_dryrun.txt, dryrun_raw.json, summary.md, swegym_clones/dask__dask-10027/)
- Reusable scripts: `scripts/g5_pyrefly_partial_file.py`, `scripts/g6_throughput_bench.py`, `scripts/g6_extrapolate.py`, `scripts/g1_dryrun.py`, `scripts/select_swegym_tasks.py`
- Harness: `harness/single_stream_eval.py`, `harness/swegym_loader.py`

---

## 2026-05-28 — Decoder sprint Phase C: both attention routes confirm attention is NOT the bottleneck

**Activity:** Ran two parallel Phase C routes (static-shape→CUDA-graphs, and FlexAttention) to break the per-row recompile wall that capped Phase B at 4.95 tok/s. Session was restarted mid-run; both background agents died but both had written their key results first. GPU lock left stale; cleared manually.

**Route 1 — static-shape mask + KV cache (`runs/g6_phase_c_static/`):**
- Padded mask + KV to a fixed MAX_CONTEXT_ROWS, masked padding to −∞. Kept Phase B's tensorized `sample_top_p` + vectorized mask.
- **Output identity: PASS** — all 5 prompts bit-identical to Phase B (SDPA retained; only buffer shapes changed).
- **CUDA graphs still blocked.** Compile log shows **19,760 recompiles**. Root cause moved one level down: `self._cursor` (KV write cursor) is a **Python int**, so dynamo emits a guard `_cursor == 0` and recaptures a new graph every row (`_cursor == 0,1,2,…`). Static tensor *shapes* were necessary but not sufficient; the per-row Python *scalar* defeats graph reuse.
- Throughput unchanged (~366 ms/row, ~4.9 tok/s). No win.

**Route 2 — FlexAttention (`runs/g6_phase_c_flex/`):**
- **Viability: PASS** on GB10 sm_12.1 — `flex_attention` + `create_block_mask` compile and run; output matches SDPA to 0.0039 (bf16 noise). Hardware-support risk cleared.
- **Mask semantics correct.** `maskcheck.json` (teacher-forced, prefill + 5 decode rows) all_match=true. `numdiff.json`: logits max_abs_diff 0.36, **argmax flip 3/80 = 3.75%, all flips at top1–top2 gaps ≤0.125** (near-ties → fp noise, not a mask bug).
- Bit-identity over 30-row rollout FAILs (15% flip, one early near-tie flip then cascade) — **expected for any attention-kernel swap** (different reduction order); bit-identity was the wrong gate. maskcheck + numdiff are the right gates and both pass.
- **Throughput: 4.87 tok/s @256, 5.10 @1024** — statistically identical to Phase B. Per-row ~366 ms unchanged. (4096/8192 contexts not run — agent died at restart; untenable in timebox anyway at ~5 tok/s.)

**Decisive conclusion:** Two independent optimisations of the attention path both leave throughput at ~4.9–5.1 tok/s. This **confirms the Phase A profile**: attention is ~5% of the budget; the ~366 ms/row is **launch overhead from the per-row Python decode loop**, and `torch.compile` cannot capture the full per-row step because of per-row Python scalar guards (the KV cursor). The fix is neither mask nor kernel — it is making the **entire per-row step CUDA-graph-capturable**: cursor as a tensor (no scalar guard), static KV (Route 1 done), static mask via tensor-cursor indexing (Route 1 done), all per-row bookkeeping tensorized, then capture+replay the single-row graph.

**Hardware-ceiling reality check (flagged for the 30 tok/s target):** stream-qwen3-8b is ~16 GB BF16. GB10 unified memory bandwidth is ≈273 GB/s (LPDDR5x). Weight-read floor per forward ≈ 16/273 ≈ 59 ms → ~12–16 rows/s even with zero launch overhead. **30 tok/s at BF16 is likely below the memory-bandwidth floor**; reaching it would require INT8/INT4 quantization (which, if applied uniformly across A/B/C/C′/D, keeps the comparison controlled but lowers absolute capability and interacts with the G1 degradation gate). Phase D at BF16 realistically targets the bandwidth ceiling (~10–16 tok/s), which supports a modest descope (50 tasks × 6 seeds), not the full 100×9.

**Sprint scoreboard (combined multi-stream packing-2, empty GPU):**
- G6 (contended): 1.16 tok/s
- Phase A (clean + per-channel silence fix): 4.54
- Phase B (tensorize sample_top_p + vectorize mask + compile-default): 4.95
- Phase C Route 1 (static shapes, SDPA): ~4.9 (CUDA graphs blocked by cursor scalar)
- Phase C Route 2 (FlexAttention): 4.87–5.10 (attention was never the bottleneck)

**Decision pending (main session):** Phase D (cursor-tensorize → full per-row CUDA-graph capture, BF16, target ~10–16 tok/s) vs quantization path (target ~30 but adds a confound) vs accept ~5 tok/s and hard-descope. Surfaced to Ian.

**Artifacts:** `runs/g6_phase_c_static/{baseline_repro,identity}.json, patched/stream_inference_static.py, patched/ROUTE1_DIFF.md, logs/`; `runs/g6_phase_c_flex/{viability,maskcheck,numdiff,identity,throughput_by_context}.json, patched/{flex_attention_patch,stream_inference_flex}.py, logs/run_all.log`.

---

## 2026-05-28 (later) — Phase D: launch-overhead hypothesis FALSIFIED; real bottleneck is repeat_kv copy; NAS unmounted

**Activity:** Phase D (cursor-tensorize → full per-row CUDA-graph capture). Decisive negative result.

**Infra incident:** `/mnt/nas/hf-cache` (NAS) unmounted mid-sprint — boot mount failed (network unreachable ~19:06), remount needs root. Model *code* was locally cached but the 16 GB weights were not. Agent re-downloaded `stream-qwen3-8b` (same SHA `54c7451…`) to `~/.cache/huggingface/hub` (16 GB; 299 GB free local). **Consequence: vanilla `Qwen/Qwen3-8B` is no longer available locally** (it lived on the NAS) — Wave 2 G1 must re-fetch it, or the NAS must be remounted. Update the G1-prep note's `HF_HOME=/mnt/nas/hf-cache` assumption: until NAS is back, use the default `~/.cache/huggingface`.

**Cursor tensorization (the scoped change): SUCCEEDED.**
- The only per-row Python scalar dynamo guarded on was `self._cursor`, advanced inside the traced `update()`. Made `update()` pure (`index_copy_` at tensor `cache_position`, no scalar), moved the advance to the eager loop (`cache.advance(C)` outside the trace).
- **Recompiles: 19,760 (Phase C, 1/row) → 0 (Phase D, over 30 rows).** The make-or-break check passed.
- **Eager output identity: bit-identical to Phase B**, all 5 prompts × 30 rows. The refactor is provably correct.

**But full-step capture gave ZERO speedup — falsifying the launch-overhead hypothesis.**
- `torch.compile(reduce-overhead)`: recompiles→0 but inductor **skipped cudagraphs** ("mutated inputs (72 instances)" — per-layer KV `index_copy_` mutates externally-held buffers).
- **Explicit `torch.cuda.CUDAGraph`**: captured + replayed cleanly, eliminating ALL kernel-launch + Python overhead. **Per-row latency 409 ms — identical to eager.** This is the clean control: if removing all launch overhead doesn't move latency, launch overhead was never the bottleneck. The Phase A profiler reading was misleading (CUDA timing came back zero on aarch64 + cudaMallocAsync/CUPTI; the 42% `.item()` CPU figure was not the wall-clock driver).

**Real bottleneck identified: the static-buffer `repeat_kv` copy.**
- Profiler (Phase D, static buffer): `aten::copy_` = **47.7% of CPU**, dominant over matmul.
- Context-control experiment: per-row latency tracks the **full padded buffer size**, not the valid region (640 cols → 383 ms; 102,400 cols → 1,139 ms).
- Root cause: Phase C's static cache returns the entire padded KV buffer; because the cross-stream mask is non-None, SDPA's GQA fast-path is disabled, forcing `repeat_kv`'s expand→reshape (a real copy) of the whole padded K/V (8→32 heads) every layer every row. **The static-buffer design that killed the recompiles introduced a full-buffer memcpy that dominates — and CUDA graphs can't help because it's real device work, not launch overhead.**
- vs bandwidth ceiling: ~409 ms/row is ~7× the ~59 ms BF16 weight-read floor. **Memory bandwidth is NOT yet the binding constraint — the copy is.** The ceiling was never reached.

**Reconciling the whole sprint:** the dynamic-KV path (Phase B eager, Phase C Route-2 FlexAttention) sits ~5 tok/s and scales more gracefully than the static buffer; the static buffer (Route 1, Phase D) is worse at long context due to the full-buffer copy. So static-padding was the wrong trade. Note: Route 2's FlexAttention patch deliberately did `repeat_kv` to match SDPA identity, so **FlexAttention's native GQA (`enable_gqa`, no repeat_kv) is an UNTAPPED lever** on the dynamic path.

**Throughput (Phase D, static, multi-stream packing-2):** 256→4.40, 1024→3.86, 4096→2.32, 8192→1.53 tok/s. No gain over Phase C.

**L4 extrapolation (8000 tok/run):** full 200×9×5 ≈ 27–78 wk (39 wk curve-mean); descoped 50×6×5 ≈ 4.5–13 wk (13 wk at realistic 8192). Both blow the 3-week budget.

**Verdict: PARTIAL — change landed, premise was false.** Sprint scoreboard unchanged at ~5 tok/s; we now know *why*. BF16 + this reference decoder will not clear the budget without either (a) KV-copy rework — dynamic/sliced KV to restore GQA SDPA, FlexAttention with native `enable_gqa`, or paged/block KV so repeat_kv touches only live blocks; (b) INT8/INT4 quantization (lifts the still-distant bandwidth ceiling AND shrinks the KV copy; controlled if uniform across A–D, but adds a capability confound + must re-check G1); or (c) hard descope. **Surfaced to Ian — strategic decision.**

**Artifacts:** `runs/g6_phase_d/{patched/stream_inference_phase_d.py, patched/PHASE_D_DIFF.md, compile.json, identity.json, identity_eager.json, throughput_by_context.json, ctx_control.json, profile.txt, extrapolation.json, recompiles.log, logs/}`; scripts `scripts/g6_phase_d_{smoke,run_all,cudagraph,diag,extrapolate}.py`.

---

## 2026-05-29 — Phase E: repeat_kv removed (correct) but cache-cat copy now dominates (69%); throughput still ~5 tok/s

**Activity:** Phase E — FlexAttention with native `enable_gqa=True` (no repeat_kv) on dynamic non-padded KV, the targeted attack on Phase D's identified 47.7% repeat_kv copy.

**The change: correct but insufficient.**
- Integration: reused Route 2's verified cross-stream `mask_mod`, removed the `repeat_kv` expand→reshape, passed GQA-shaped K/V (`[B,8,K,d]`) directly to `flex_attention(..., enable_gqa=True)`. Dynamic Phase B KV base (not Phase D's static buffer).
- **Correctness PASS:** maskcheck `all_match` (prefill + 5 decode rows); numdiff vs SDPA logits max_abs_diff 0.44, **3/80 argmax flips all at top1–top2 gaps ≤0.125** (two exact ties) → GQA head→group mapping correct, differences are benign bf16 reduction-order noise. `dynamic=True` kept recompiles bounded (not per-row).
- **Throughput UNCHANGED:** 4.72 @256, 5.07 @1024 tok/s, ~377–382 ms/row — statistically identical to Phase B/C (<1% vs this run's SDPA baseline).

**Mechanistic finding — the copy share ROSE to 69%.**
- Profiler (steady-state GQA-flex row): `aten::copy_` self-CPU **69.15%** (was 47.7% in Phase D), `aten::mm` 14.3%.
- Removing repeat_kv removed one copy and shrank per-row K/V bytes (8-headed now), but exposed the next: transformers 5.9.0 `DynamicLayer.update` does `self.keys = torch.cat([self.keys, key_states], dim=-2)` **every decode row** — reallocating + copying the entire K/V cache, all 36 layers, per row. repeat_kv had masked part of this; with matmul relatively cheaper, copy's share climbed.
- Still ~6.4× off the ~59 ms BF16 weight-read floor. **Memory bandwidth STILL not binding — stacked per-row copies are.**

**Sprint pattern (5 phases):** onion-peeling — launch overhead (falsified, D) → mask (C1) → attention kernel (C2) → repeat_kv (D diagnosis, E fix) → cache-cat (E diagnosis). Each fix exposes the next copy. Throughput floor stuck at ~5 tok/s because copies stack; we keep removing the current dominant one and the next takes over.

**Why the cache-cat matters most for L4:** the cat cost grows with context length, so the 8192-token regime (the realistic L4 decode length) collapses (~1.5 tok/s on Phase D static at 8192). Removing it flattens the context-scaling curve — which is what actually gates L4 feasibility, more than the 256 headline.

**L4 extrapolation (Phase E, optimistic 256/1024 curve, 8000 tok/run):** full 200×9×5 ≈ 24 wk; descoped 50×6×5 ≈ 4 wk. Both blow the 3-week budget at BF16; 8192 regime would be worse.

**Next lever (proposed Phase F):** pre-grown/ring KV buffer that **appends in place at a cursor** (no realloc, no cat), GQA-shaped (no repeat_kv), attention reads only the valid region `[0:cursor]` (no full-buffer cost). This is the synthesis of D (no realloc) + E (no full-buffer read, no repeat_kv) avoiding both their traps. Plausibly the last BF16 copy lever; should produce a real jump (copies are 69% now) AND flatten the 8192 curve. If Phase F still lands ~5 tok/s → the binding cost is the per-row Python decode loop / kernel-launch granularity, and the honest options become INT8/INT4 quantization or hard descope.

**Decision pending (main session):** Phase F (in-place KV, last BF16 copy lever, explicit stopping rule) vs quantization vs descope-now. Surfaced to Ian.

**Artifacts:** `runs/g6_phase_e/{patched/flex_attention_gqa.py, patched/stream_inference_gqa.py, patched/PHASE_E_DIFF.md, maskcheck.json, numdiff.json, throughput_by_context.json, profile.txt, extrapolation.json, recompiles.log, logs/}`; scripts `scripts/g6_phase_e_{smoke,maskcheck,run_all,extrapolate}.py`. Note: scripts use `HF_HUB_OFFLINE=1` + explicit snapshot SHA path since NAS is down and `snapshot_download` fails offline.

---

## 2026-05-29 (later) — Phase F: in-place KV write works, but contiguous-on-slice copy takes over; STOPPING RULE triggered (~5 tok/s after 6 phases)

**Activity:** Phase F — pre-grown KV buffer, in-place append at tensor cursor, GQA-shaped, attention reading a valid-region slice. The last BF16 copy lever per the agreed stopping rule.

**Result: change worked, throughput did NOT move.**
- In-place write succeeded: `aten::index_copy_` is now **5.1%** (the per-row `torch.cat` full-cache realloc is gone). maskcheck `all_match`.
- But throughput **unchanged**: 4.73 @256, 5.09 @1024 tok/s, **376 ms/row** — statistically identical to Phase E.
- Profiler: `aten::copy_` **still 56.2%** (2688 calls), `aten::mm` 11.9%. New culprit: slicing the valid region `K[:, :, :cursor, :]` from the contiguous `[B,H,MAX,d]` buffer yields a **non-contiguous** tensor (dim-1 H strides over full MAX), so FlexAttention's `.contiguous()` on K/V copies the whole valid region every layer every row. **Same O(context) copy, relocated from `cat` (E) to `contiguous`-on-slice (F).**

**The decisive pattern (6 phases):** throughput is flat at ~5 tok/s / ~376 ms/row across A/B/C/D/E/F. The binding cost is consistently an O(context) bf16 K/V copy that takes ~56–69% of CPU; each phase removes the current dominant copy (repeat_kv → cat → contiguous-on-slice) and the next O(context) copy takes over. We are ~6.4× off the ~59 ms BF16 weight-read floor; **memory bandwidth has never become binding** — copies have, the whole way down. Per the agreed rule, **BF16 copy-chasing is exhausted.**

**Correction to the stopping rule's own premise (important):** the rule said "if F fails → quantization or descope." But the diagnosis shows the bottleneck is bf16 **K/V/activation copies**, not weight reads or matmul (mm is only ~12%). **Weight-only INT8/INT4 quantization would shrink weight reads and matmul — NOT the copies — so it likely would NOT move throughput here.** Quantization only helps if the KV cache is also quantized (extra complexity + a stronger capability confound). So quantization is a weaker fallback than assumed when we set the rule.

**Genuinely differentiated Phase G (not more onion-peeling):** pre-grown buffer + FlexAttention over the **FULL** buffer with a `BlockMask` that masks the unfilled future. flex's block-sparsity **skips the all-masked future blocks** (no compute, no copy), and there is **no slice → no `.contiguous()` copy** and (in-place write) no cat. This removes ALL THREE copies at once (repeat_kv via enable_gqa, cat via in-place write, contiguous via no-slice). It differs from Phase D's failure: D used dense SDPA on the full buffer (full-buffer copy + repeat_kv); G uses flex BlockMask block-skipping (the mechanism BlockMask exists for). The in-place write is only 5%, so if the 56% contiguous copy is removed, throughput could finally fall toward the floor. Plausibly the actual solution — but it is a 7th phase.

**Sprint scoreboard (combined multi-stream packing-2, tok/s):** G6 1.16 → A 4.54 → B 4.95 → C ~5 → D ~5 → E 4.7/5.1 → F 4.7/5.1. ~12 GPU-hours of agent time; throughput flat since Phase A's silence fix.

**Decision (main session):** honoring the stopping rule's spirit, surfaced to Ian with the corrected premise — Phase G (block-skip, hard stop after) vs quantization (now known weaker) vs descope-now. Diminishing returns is real (6 phases flat); descope-now gets us to the actual science. But Phase G is genuinely differentiated and cheap.

**Artifacts:** `runs/g6_phase_f/{patched/inplace_kv_cache.py, patched/stream_inference_inplace.py, patched/flex_attention_gqa.py, patched/PHASE_F_DIFF.md, maskcheck.json, throughput_by_context.json, profile.txt, recompiles.log, logs/}`; scripts `scripts/g6_phase_f_*.py`.

---

## 2026-05-29 (later 2) — Phase F deep-dive overturns the copy diagnosis; Phase G stopped; cudaEvent microbench launched

**Activity:** Phase F's full final report arrived (after a longer instrumented run, superseding its earlier partial notification). It overturns the copy diagnosis that had driven Phases E→G, so Phase G (already running on the now-refuted premise) was stopped and replaced with a clean GPU-timing microbench.

**Phase F final findings (supersede earlier):**
- In-place `InPlaceStreamCache` (pre-allocated GQA-shaped K/V, `index_copy_` at tensor cursor, returns valid-region slice; BlockMask `q_offset=cursor,kv_len=cursor+C`). maskcheck `all_match`; numdiff decode 20 rows vs DynamicCache **0 mismatch** (write offset provably correct); recompiles bounded to **9 events** (vs Phase C's 19,760).
- Throughput-by-context (clean): 256→4.73, 1024→5.09, 4096→5.01 tok/s; ~377–396 ms/row. **Curve essentially FLAT through 4096** (+5% ms/row over 16× context) — the valid-region slice works; 8192 did NOT collapse like Phase D's static buffer (1.53). But flat at ~5 tok/s, not lifted.
- **DECISIVE CORRECTION:** the recurring 56–69% `aten::copy_` was **never the cache cat** (`aten::cat` = 0.16% always), never repeat_kv, never contiguous-on-slice. It is **`aten::_to_copy` = per-layer RMSNorm fp32 round-trips + RoPE casts** — model-internal, cache-independent. And on this aarch64 build (CUPTI broken) **torch-profiler CPU self-time is dispatch/cast accounting, NOT the wall-clock driver** — the same lesson as Phase D's CUDA-graph control (which removed all launch overhead with zero speedup). The "copy" we chased for 3 phases was a profiler artifact.

**Consequence — Phase G stopped:** Phase G's premise was "remove the valid-region slice → kill the 56% contiguous copy." Since the 56% is casts, not the slice, Phase G was testing a refuted hypothesis. Killed its run (and a stale Phase-F `throughput_long` orphan, PID 171479, that had been causing 2× GPU contention — Phase G's contended baseline read 818 ms/row vs the true ~377). maskcheck had passed. No throughput gain expected or needed.

**Sprint conclusion (6 phases, A–G):** throughput flat ~5 tok/s / ~377 ms/row throughout; every phase removed its named cost (launch overhead [falsified D], mask [C1], attention kernel [C2], repeat_kv [E], cat [F]) without moving wall-clock. The binding cost is NOT any KV copy and NOT memory bandwidth (we're ~6.4× off the 59 ms weight floor and ~15% of the 273 GB/s roofline). It is most likely the **per-row Python decode-loop / kernel-launch granularity** at BF16 — but **we have never had clean GPU timing to prove it** (CUPTI broken → all 6 phases trusted CPU self-time, which Phase F showed is misleading).

**Action — cudaEvent microbench (task #23, running):** the measurement that should have anchored the sprint. `torch.cuda.Event` GPU-time decomposition of one decode row: matmul % vs cast % vs attention % vs dispatch-gap (whole-row wall-clock minus summed GPU-event time). Settles: (a) is the RMSNorm/RoPE cast real GPU time or a CPU artifact? (b) dispatch-bound (→ quantization won't fix; fusion/graph-capture needed, hard given per-row sampling/silence/`gen.send`) or GPU-bound (→ INT8/INT4 quantization directly helps, ~2×/4× weight-bandwidth); (c) matmul floor + achieved bandwidth vs 273 GB/s roofline + projected INT8/INT4 tok/s. **This decides quantization vs hard-descope.**

**Process note:** subagents repeatedly spawned detached GPU runs then yielded, and left orphaned watcher/throughput processes that caused GPU contention (corrupting one baseline by 2×). Cleaned up manually. For future GPU phases: have the agent run the bench in-process (not detached) or register a tracked cleanup, and check `nvidia-smi` for orphans before trusting any throughput number.

**Artifacts:** `runs/g6_phase_f/{patched/*, maskcheck.json, numdiff.json, profile.txt, profile_summary.json, throughput_by_context.json, extrapolation.json, recompiles.log}`; Phase G partial under `runs/g6_phase_g/` (baseline_profile_summary.json confirms 56.3% copy share = the cast artifact). Microbench → `runs/g6_microbench/`.

---

## 2026-05-29 (later 3) — cudaEvent microbench: matmul-bound + batch-starved, NOT copies/dispatch. Reframes throughput as a batching problem.

**Activity:** Ran the clean `torch.cuda.Event` GPU-time microbench (CUPTI broken → all prior phases trusted CPU self-time). I ran it directly (the dispatched agent stalled waiting on a GPU held by a relaunched Phase-G orphan; killed the orphan + stale agents, fixed a one-line GQA bug in the bench's isolated-attention timing, re-ran clean).

**Definitive GPU-time breakdown (stock Qwen3-8B, BF16, steady-state row = [1,10] through 36 layers):**
- **ctx 512:** whole row GPU-event **371.9 ms** vs wall **373.0 ms** → dispatch gap **ratio 1.00**. Rollup: **matmul 92% | cast(RMSNorm+RoPE) 2% | attention 3% | KV-write 1%.** Achieved BW **39.8 GB/s = 15% of 273 roofline**; roofline-ideal row ≈ 50.9 ms.
- **ctx 4096:** whole row 399.9 ms GPU vs 402.5 ms wall (ratio 1.01). matmul 75% | cast 2% | attention 18% | KV-write 4%. Achieved BW 39.5 GB/s (14%).
- Per-stage (512), GPU-ms: qkv_proj 3.12, mlp_down 3.00, mlp_gate_up 2.68, o_proj 1.09, attention 0.35, rope 0.05, rmsnorm 0.04, kv_write 0.06.

**Definitive conclusions (these supersede all prior phase diagnoses):**
1. **NOT dispatch/launch-bound** — wall = GPU-event (ratio 1.00). Kills the "per-row Python decode-loop granularity" hypothesis from Phase F.
2. **NOT the casts** — RMSNorm fp32 + RoPE = 2% of real GPU time. The recurring "56–69% aten::copy_" was a **CPU-self-time artifact** (dispatch/cast accounting on this aarch64 box), confirming Phase F's suspicion. Real GPU cast time is negligible.
3. **NOT KV copies** — cat was 0.16% (Phase F), attention 3% at short ctx.
4. **IS matmul/weight-read bound (92%), running at only 15% of peak memory bandwidth** — because the decode batch is tiny (one task's 10 channels). Classic small-batch decode regime.

**The reframe — we optimized the wrong axis for 6 phases.** Phases A–G all attacked single-sequence *latency* and chased copies that were profiler artifacts. But the matmuls read ~16 GB of weights per row at 39.8/273 GB/s = batch-starved, with ~6× headroom to the roofline. **The L4 eval is a throughput workload: 900+ independent trajectories (100 tasks × 9 seeds × 5 conditions).** The correct lever is **batched decode** — run N tasks concurrently so the 16 GB weight read amortizes across N×10 tokens, pushing BW utilization toward the roofline. This is standard eval practice and does not touch experimental validity (each sequence is independent). Quantization (INT8/INT4, fewer weight bytes) stacks on top.

**Caveat to verify:** batched generation with the multi-stream `stream_generate_iter` API is clean for A/B/C/C′ (no async injection); condition D's per-task async LSP injection (`gen.send` at per-task latency offsets) needs padding/masking to batch — doable but to be confirmed.

**Meta-lesson:** this cudaEvent microbench should have run at G6 before any optimization. CPU-self-time profiling on a CUPTI-broken box sent six phases chasing artifacts. Recorded as process guidance.

**Decision (main session):** batched-decode probe (recommended) vs quantization vs descope. Surfaced to Ian.

**Artifacts:** `runs/g6_microbench/{summary.md, summary.json, whole_row.json, layer_breakdown.json, dispatch_gap.json, matmul_floor.json, run.log, run2.log}`; `scripts/g6_microbench_layer.py`.

---

## 2026-05-29 (later 4) — Batched-decode probe: throughput crisis RESOLVED. Decoder sprint CLOSED.

**Activity:** Batched-decode throughput probe (the lever the microbench pointed to). Ran synchronously (no detach) — clean.

**Scaling — aggregate productive tok/s (Output+Analytical), per-step latency, peak mem:**
- **ctx 512:** B=1 → 5.4 (371ms, 16.6GB); B=4 → 20.7 (388ms); B=8 → 33.0 (486ms); B=16 → 54.8 (585ms); B=32 → **78.9** (810ms, 21.6GB); B=64 → 80.6 (1592ms, 26.7GB, **past knee**). Knee at B=32 ≈ 14.5× over single-sequence.
- **ctx 4096 (realistic L4 regime):** B=1 → 5.0 (399ms, 17.7GB); B=8 → 22.0 (728ms); B=16 → **29.9** (1071ms, 36.8GB); B=32 → 35.4 (1810ms, 57.3GB, diminishing). Knee at B=16 ≈ 6×.

**Why it scales:** exactly as the microbench predicted — decode is memory-bound (read 14 GB weights per step); batching B independent sequences amortizes that read across B×10 tokens until compute-bound. Per-step latency stays ~flat at low B (memory-bound) then rises (compute-bound). Operating point: **B=16** at realistic context (~30 tok/s, 37 GB of 128 — headroom for 8k context where KV ~2×).

**L4 wall-clock at realistic ~30 tok/s:**
- **Descoped (50 tasks × 6 seeds × 5 conditions = 1,500 trajectories): ~5–11 days — FITS the 21-day budget comfortably.** This is the pre-registered fallback scope.
- **Full (200 × 9 × 5 = 9,000 trajectories): ~30–47 days — OVER the 21-day budget at BF16.** Needs INT8 quantization (~2×, stacks cleanly → ~3 weeks) or extended wall-clock. L4-era decision, not a blocker now.

**Validity:** batching is clean for A/B/C/C′ (independent sequences). Condition D injects LSP diagnostics per-task via `gen.send` at per-task latency offsets — batchable with per-element injection timing + padding/masking (feasible; to implement at the D-build stage).

**RESOLUTION:** the throughput crisis (G6's 1.16 tok/s → 100–1370 weeks, project-threatening) was a measurement+regime artifact. The real picture: **single-sequence ~5 tok/s, but the eval is a throughput workload and batching at B=16 gives ~30 tok/s, making the (pre-registered) descoped L4 a ~1-week run.** Quantization remains available as additional headroom for full-scope L4.

**Decoder sprint CLOSED.** Net result: 1.16 → (clean baseline 2.4) → batched ~30 tok/s. The 7 optimization phases (A–G) were largely wasted effort chasing CPU-self-time profiler artifacts on a CUPTI-broken aarch64 box; the resolution came from (a) one cuda.Event microbench that correctly identified matmul/batch-bound, and (b) batching — standard eval practice. **Process lesson recorded in memory: roofline + workload-shape + tool-validation BEFORE optimizing.** Reusable artifacts retained: in-place GQA-flex decoder (Phase F, identity-verified), cuda.Event microbench harness, batched-sweep harness.

**Unblocks:** plan update to v0.4 (record resolution, set eval batch=16, adopt descoped L4, note full-scope needs INT8); then resume the science — Wave 2 (G1 single-stream-degradation, now also needs vanilla Qwen3-8B re-fetched since NAS dropped it) and Wave 3 (delivery-layer skeletons + reformat pipeline).

**Artifacts:** `runs/g6_batched/{sweep_ctx512.json, sweep_ctx4096.json, extrapolation.json, _smoke.json}`; `scripts/g6_batched_{sweep,extrapolate,summary}.py`.

---

## 2026-05-29 (later 5) — Resume science: G3 ✓, G4 ✓; G1 surfaces a substrate-usability problem (the model is a cognition/chat model, awkward for code eval)

**Activity:** With throughput resolved (v0.4), launched Wave 2 (G1) + Wave 3 (delivery skeletons+G4, reformat+G3) in parallel against the restored NAS.

**G3 — causal-validity (latency-replay reformat): PASS (10/10).** Built §7.4 pipeline (`training/reformat.py`: `reformat_to_D` with sync-diagnostic masking + student-tokenizer-time latency shift; `reformat_to_Cprime`; `MultiStreamSequence`). Adversarial-leak fixture proves the test is non-vacuous. **Caught a real bug:** back-to-back snapshots' diagnostics latency-shifted onto the same side-stream slot → silent overwrite (info-content violation) → fixed with `next_free`. Artifacts `runs/g3/`, `training/`, `tests/`.

**G4 — payload-equivalence: PASS (10/10 byte-identical across B/C/C′/D)** via live pyrefly daemon. Four delivery layers in `lsp/` over one shared `normalize_payload` (drops non-spec `committing-transaction`, severity int→name, 0→1-index, top-K=10 by recency, canonical JSON). B/C/C′/D differ only in channel+timing descriptors (D has real debounce + latency-offset). pytest regression re-runnable at L1/L3. Artifacts `runs/g4/`, `lsp/`.

**G1 — single-stream-degradation: BLOCKED by a substrate-driving problem (NOT capability degradation, NOT a clean pass either).** The fairness probe did its job and stopped a meaningless run:
- The G1 harness drove `stream-qwen3-8b` as **raw completion** → garbage Output + non-determinism, while vanilla Qwen3-8B passed. Root cause: the model is a **parallel-cognition CHAT model** (README: "monitorability of internal streams"; 8 cognitive channels Analytical/Skeptical/Intuitive/Void/…). Correct API (`stream_inference.generate`): prompt goes on the **User channel** (channel 0, injected one token/tick), **Output** (channel 1) responds, with an 11-row warm-start primer.
- Driven canonically (`scripts/g1_stream_driving_check.py`), it DOES produce correct code on easy problems — `has_close_elements` (stochastic run) gave the right nested-loop algorithm; `add()` gave `return a + b`. **So the substrate is code-capable in principle — the original garbage was a driving bug.**
- **But real generation-control problems remain, blocking any eval:**
  1. **Determinism:** `sample_top_p` only argmaxes at `temperature<=0` (line 142); we (and the harness) passed small-positive temps → stochastic. **Fix: `temperature=0`** → confirmed deterministic (r1==r2==r3).
  2. **Silent-output failure:** at greedy, the model **deterministically stays silent on Output** for harder prompts (`has_close_elements` → 0 Output tokens × 3 runs); `silence_penalty=10` insufficient; the `all_silent_streak >= 1` early-stop (line 361) then halts immediately.
  3. **No stop / chat preamble / formatting:** `add()` → "Sure! Here's the complete function: …" then **repeats endlessly** to `max_rows`, with literal `\n` and spaced tokens; needs EOS/stop criteria, preamble stripping, output cleanup.
  4. Possible token-dropping (a `-` operator missing in one decode) — to confirm.
- **Implication:** the substrate is a chat/cognition model, not a code model. Getting reliable, reproducible, extractable SWE-style trajectories needs a real eval-driver engineering effort (anti-silence forcing, stop criteria, output post-processing), and its capability on non-trivial code is unproven (went mute on a basic function). This is a substrate-validity concern of the kind G1/R4 exist to surface. **Note:** the entire throughput sprint drove this model without ever decoding output — this problem was invisible until the fairness probe forced us to look. (Same meta-lesson as the throughput saga: validate the basic assumption — "can we get usable output?" — before building on it.)
- **Decision surfaced to Ian:** invest in a proper stream-eval driver + validate HumanEval pass rate before trusting G1, vs reconsider substrate/framing if code-capability proves too weak.

**Status:** G3 ✓, G4 ✓ (2 of 4 L0 correctness gates green). G1 pending the eval-driver work. G2 canary (Wave 4) still blocked on G1.

**Artifacts:** `scripts/g1_stream_driving_check.py`, `runs/g1/fairness_probe.json`; `runs/g3/`, `runs/g4/`, `lsp/`, `training/`, `tests/`.

---

## 2026-05-29 (later 6) — Health check: substrate is numerically healthy but is a COGNITION/CHAT model, not a code model

**Activity:** Per Ian's prompts (possible numeric issues? correct calling framing?), checked the substrate's source + ran the README's blessed conversational recipe.

**Framing (source review):** `_tokenize_user` (docstring: "matching the training data pipeline, per-chunk with leading space") tokenizes plain user text onto the User channel — **no chat wrapper**. So `generate(user_text=...)` / `stream_generate(tok, prompt, ...)` is the correct call; **`chat_template.jinja` is inherited Qwen3 boilerplate the stream training did NOT use — applying it would be wrong.** README blessed recipe: `warm_start=True, temperature=0.6, silence_penalty=5.0, skip_silence=True` (we'd omitted skip_silence and used sp=10). **Every model-card example is conversational; zero code examples.**

**Numeric health: HEALTHY (not a bug).** `scripts/g1_sanity_health.py` — driven the blessed way, Output is fluent and coherent, both stochastic AND greedy (temp=0):
- temp=0: *"OK I'm ready. I've been thinking about what it means to be truly present with someone you love… Is that the same as fresh? Or is fresh just attention…"*
- Analytical/Synthesis channels produce coherent parallel "thoughts" — the parallel-cognition behavior the model was built for.
No NaN/corruption. The earlier garbage was purely raw-completion mis-driving. **Greedy works fine for conversation** → the "goes silent on code" failure is **code-specific, not a general greedy bug**.

**Conclusion — substrate–task mismatch:** `stream-qwen3-8b` is a healthy parallel-cognition CHAT model, not a code model. Its "stream-data" fine-tune moved it toward cognition/monitorability and apparently away from structured code generation. This is a substrate-validity issue that should have been probed at Wave 0 (we assumed Qwen3-8B-base → codes; it was cognition-SFT'd).

**Key reframe for the decision:** the experiment never uses this base checkpoint — §7.3 plans a **shared code-SFT pass** (SWE-Gym/R2E-Gym) before all conditions. So the decisive question is NOT "can the cognition-tuned base model code?" (the running probe will say: poorly) but **"does code-SFT on the multi-stream architecture RECOVER coding ability?"** If yes → substrate viable, G1's base gap is an expected recoverable SFT artifact. If even post-SFT the multi-stream format is hostile to structured code → substrate fatal for this project. **Proposed next: a small code-SFT viability probe on the stream model** (LoRA on a few hundred code/edit trajectories → re-measure HumanEval), which is the actually-decisive test. The base-model HumanEval probe (running) is a useful baseline data point but tests the wrong checkpoint for the real question.

**Note:** 27B (`stream-qwen3.5-27b`) is the same monitorability-research family — likely also a cognition model; the code-SFT-recovery question applies there too.

**Artifacts:** `scripts/g1_sanity_health.py`, `scripts/g1_stream_driving_check.py`, `runs/g1/fairness_probe.json`. Base-model code-capability probe running → `runs/g1_probe/`.

---

## 2026-05-29 (later 7) — Coder-availability research: B2-merge dead, D (mechanism-switch) strongest. Substrate fork to Ian.

**Activity:** Web research on Qwen3-8B-compatible / GB10-runnable coding models, to inform the substrate options after the health check showed stream-qwen3-8b is a cognition model that can't code.

**Findings:**
- **B2 (weight-merge stream × coder): NOT VIABLE.** No specialized coder shares `Qwen/Qwen3-8B`'s base — the Qwen3-Coder family (30B-A3B, 480B-A35B, Coder-Next) is all separately-pretrained MoE with incompatible shapes. Only `Qwen3-8B(-instruct)` is mergeable → modest lift, not SWE-bench class. Ruled out.
- **B1 (retrain multi-stream on a coder base): viable, heavy.** No Qwen3 dense coder → a Qwen3-8B retrain inherits weak code. `Qwen2.5-Coder-14B` (HumanEval+ 83.5, SWE-bench ~27%, Apache-2.0, GB10-runnable) is a stronger base but needs the stream recipe ported to Qwen2.5 arch. Days of training; the bundled `finetune.py`+`StreamDataCollator` are the machinery.
- **D (switch to single-stream interleaved-async tokens, use a strong coder directly): strongest, lowest-risk.** GB10-runnable coders: `Qwen3-Coder-30B-A3B-Instruct` (MoE 30B/3.3B-active, ~50% SWE-bench, Apache-2.0), `Qwen2.5-Coder-14B` (dense, ~27%, Apache-2.0), `agentica-org/DeepSWE-Preview` (Qwen3-32B RL, 42–59% SWE-bench, MIT — our external reference).

**Strategic read:** the fork is "keep Geiping multi-stream (A code-SFT cheap/low-ceiling, or B1 retrain heavy) vs switch mechanism to Hooper-style interleaved-async tokens on a real coder (D)." D doesn't kill RQ1 — "async beats sync at matched content" is agnostic to multi-stream-side-channel vs interleaved-tokens; the latency-replay methodology survives; related-work shifts toward Hooper. D buys a ~50% SWE-bench model immediately vs near-floor capability. **Recommended: D** (a week of pain has all traced to forcing a cognition model to code), but the multi-stream architecture was the distinctive contribution → decision surfaced to Ian.

**Also (process):** confirmed root cause of the recurring subagent "generate stalls" — single-seq `generate()` is ~96s/problem (~30 min for a 20-problem probe), agents run it *detached* (setsid, PPID=1 orphans) to survive their turn, and session cleanup reaps detached GPU procs at ~6–7 min → run dies before finishing. Fix: batch the eval (we have the harness; minutes not 30 min) AND run GPU eval **inline from main session (harness-tracked), never delegated-and-detached**. Inline runs (microbench, batched sweep, health check, driving check) all completed; delegated GPU runs nearly all stalled. Recorded in memory.

---

## 2026-05-29 (later 8) — DECISION: pivot to option D (Interaction-Model). Plan v0.5. Overnight autonomous execution started.

**Decision (Ian):** pivot to option D — drop the Geiping multi-stream substrate; operationalize "in-stream async feedback" as **single-stream interleaved-async tokens** (Hooper / Thinking Machines interaction-models style) on a **real coder (Qwen2.5-Coder family)**. Ian stepping away until morning; instructed to proceed with sensible defaults, no check-ins.

**Why (recap):** stream-qwen3-8b is a cognition/chat model that can't carry a coding eval; no Qwen3-8B-base coder exists to merge; retraining multi-stream on a coder is heavy. The TM "interaction models" piece + Hooper both converged on single-stream interleaving — same idea, more tractable, field-validated. RQ1 unchanged; only the operationalization changes. C′ dissolves (no format axis in single-stream → C-sync-inline vs D-async-inline isolates synchrony directly).

**Defaults chosen (justified, reversible):**
- Substrate: **Qwen2.5-Coder dense family** — 7B (L0/L1 dev), 14B (L2/L3), 32B (L4 headline). Apache-2.0, strong (HumanEval ~88/89/92), GB10-runnable, ONE dense arch across the ladder (fixes the arch-transfer confound), standard inference (throughput crisis evaporates — normal transformer/KV/batching/vLLM). Qwen3-Coder-30B-A3B MoE noted as a faster-inference L4 alt (dense default to avoid dense→MoE transfer caveat).
- Conditions: A/B/C/D (C′ removed). Central comparison **D vs C** (synchrony, format constant). Leakage probes unchanged.
- Carries over: payload normalization + G4, latency-replay reformat + G3 (layout multi-stream→interleaved), pyrefly daemon G5, SWE-Gym/harness, stats plan (minus C′). Retired: multi-stream substrate, cognition-model driving, decoder throughput sprint, R10.

**Plan:** experiment_plan.md → **v0.5**, new authoritative **§0 (Interaction-Model Pivot)** written (hypothesis, substrate, interleaved mechanism, conditions, carries/changes, related-work positioning incl. Hooper + TM, revised ladder, open questions). §1–§17 retained as superseded-multi-stream provenance.

**Overnight execution started:**
1. **Capability baseline (the de-risk we never had):** `scripts/d_capability_eval.py` — batched greedy HumanEval on Qwen2.5-Coder-7B (then 14B) via standard `model.generate`. Running INLINE/harness-tracked (not delegated — the GPU-eval stalls were detached-agent artifacts). Confirms the coder actually codes (expect ~88) before building on it.
2. **Reformat → interleaved + G3** (agent, non-GPU): `training/reformat.py` emits interleaved single-stream (diag block spliced at query_pos + latency_in_tokens; sync original masked); re-run causal-validity gate + adversarial-leak fixture → `runs/g3_interleaved/`.
3. **Delivery → inline + G4** (agent, non-GPU): `lsp/delivery_*.py` target inline insertion offset; drop delivery_cprime; re-run SHA-256 payload-equivalence across B/C/D → `runs/g4_inline/`.
4. Pending: interleaved-async inference splicing prototype (§0.9 Q2) + new interleaved canary (G2).

**Process lesson applied:** GPU eval runs inline from main session, batched, harness-tracked background — never delegated-and-detached (every delegated GPU run stalled via setsid+reap; every inline run completed).

**Artifacts:** experiment_plan.md v0.5 §0; `scripts/d_capability_eval.py`; tasks #25–#29. Capability run → `runs/d_capability/`.

---

## 2026-05-30 — D-pivot L0 progress: substrate VALIDATED (Qwen2.5-Coder-7B codes), G3+G4 green on interleaved layout

**Activity:** Overnight autonomous execution of the v0.5 pivot. All three foundational pivot gates landed.

**Capability baseline (the keystone de-risk) — PASS.** `Qwen2.5-Coder-7B-Instruct` HumanEval **pass@1 = 0.829 (136/164)** measured, **≈0.89 true**. Of 28 fails, 10 were extractor artifacts (NameError on helper functions the extractor dropped by slicing from the entry-point `def`; fixed to keep the whole fenced block) and only 17 were genuine wrong answers. Matches published ~88. **The coder genuinely codes — the exact thing the stream model could not do.** The pivot is empirically validated. `scripts/d_capability_eval.py`, `runs/d_capability/humaneval_7b.json`. 14B staged (28G) for L2/L3.

**G3 interleaved (causal validity) — PASS 10/10.** `training/reformat.py` now emits a single interleaved sequence: diagnostic blocks `‹diag›…‹/diag›` spliced inline at `query_pos + latency_in_student_tokens` (D) or offset 0 (C, replacing old C′+C). Sync-diagnostic masking preserved; same-position collisions stack in arrival order; adversarial-leak fixture catches naive reformats. `runs/g3_interleaved/`, `training/INTERLEAVED_LAYOUT.md`. (Minor follow-up: re-verify the HF-tokenizer-time conversion in .venv-streams — the agent's interpreter lacked transformers and used the mock fallback.)

**G4 inline (payload equivalence) — PASS 10/10.** `lsp/` delivery layers retargeted to inline insertion: descriptor `{condition, insertion_offset_tokens, model_initiated}`; B=offset-0 model-initiated, C=offset-0 at edit boundary, D=`edit_pos+round(latency/ms_per_token)`. `delivery_cprime.py` deleted. Byte-identical normalized payloads across B/C/D. `runs/g4_inline/`, `lsp/INLINE_DELIVERY.md`. **Flag resolved-in-progress:** the old 200 ms/token (retired stream model) made D's ~21 ms latency round to 0 offset — `ms_per_token` is being re-measured on Qwen2.5-Coder now (splice prototype).

**In progress:** `scripts/d_splice_prototype.py` — measures real `ms_per_token` on Qwen2.5-Coder (resolves §0.9 Q4) and demonstrates the core mechanism (prefill a `‹diag›` block into the KV cache mid-generation, continue decoding) — the inference-time analog of the training-layout splice. Then the interleaved-async canary (#29).

**Status:** L0 of the v0.5 pivot is largely de-risked — substrate codes, both methodological gates green on the new layout, mechanism prototype underway. Clean batched standard inference (no throughput drama). Tasks #25/#26/#27 done; #28 running; #29 next.

---

## 2026-05-30 — D-pivot L0 COMPLETE (overnight). Canary passes; mechanism end-to-end validated.

**Overnight autonomous run summary (v0.5 Interaction-Model pivot). L0 of the new design is complete and green.**

| L0 gate | Result |
|---|---|
| **Substrate capability** (replaces G1) | Qwen2.5-Coder-7B HumanEval **0.829 measured / ≈0.89 true** (10 fails were extractor artifacts). The coder codes. 14B staged. |
| **G3 causal validity** (interleaved) | **PASS 10/10** — diag blocks spliced inline at query+latency; sync masked; adversarial-leak caught. |
| **G4 payload equivalence** (inline) | **PASS 10/10** — byte-identical across B/C/D; C′ removed. |
| **ms_per_token** (§0.9 Q4) | **92.6 ms/token** on 7B → pyrefly latency sub-token, 200 ms debounce ≈ 2 tokens; sync-vs-async is as much delivery-mode as raw latency → plan a latency sweep. |
| **Splice mechanism** | **VALIDATED** — prefill ‹diag› into live KV cache mid-generation + resume decodes coherently. |
| **Canary** (new G2) | **PASS** — A=7/8, D=8/8, D>A; diagnostic causally flips the one bug the model missed unaided (reversed_range). Effect understated by easy fixtures; magnitude needs harder tasks + SFT. |

**Net:** the pivot is empirically de-risked. The substrate genuinely codes (the thing the stream model could not), the interleaved-async mechanism is mechanically sound on a real coder, both methodological gates carry over green, and an inline diagnostic causally helps end-to-end. Standard batched inference → throughput is a non-issue.

**Decisions taken autonomously (all reversible, flagged for review):**
- Substrate = Qwen2.5-Coder dense family (7B dev / 14B mid / 32B headline) — one dense arch across the ladder.
- Conditions A/B/C/D; C′ dissolved (single-stream has no format axis).
- Diagnostic delimiter = `‹diag›…‹/diag›` guillemets (plain strings, tokenizer-agnostic; reconsider vs reusing chat special tokens at SFT time).
- Capability/canary run zero-shot on the base instruct model (pre-SFT) — appropriate for L0 mechanism validation.

**Open for Ian (morning):**
1. Confirm the substrate-ladder + delimiter defaults.
2. The latency-sweep design (0/2/8/32-token offsets) given pyrefly is sub-token-fast — is the delivery-mode contrast (continuous vs interrupt-and-wait) the right framing? This is the crux of what "async" buys and deserves a design pass.
3. Next phase = L1: the teacher-rollout → interleaved-SFT pipeline (reformat is built) on A/B/C/D layouts, then leakage probes (D-noise/adversarial/counter-factual) and the C-vs-D primary readout at L2.
4. Minor: re-verify HF-tokenizer-time conversion in .venv-streams (G3 agent used a mock fallback); the §1–§17 multi-stream sections of the plan can be archived.

**Artifacts:** `scripts/d_{capability_eval,splice_prototype,canary}.py`; `runs/d_capability/`, `runs/d_canary/`, `runs/g3_interleaved/`, `runs/g4_inline/`; `training/INTERLEAVED_LAYOUT.md`, `lsp/INLINE_DELIVERY.md`; plan v0.5 §0. Tasks #25–#29 complete.

---

## 2026-05-30 — L1 kickoff: first interleaved-SFT training run (bootstrap data + LoRA pipeline)

**Ian:** substrate + ladder confirmed; "live feedback" framing confirmed (the diagnostic arriving *while the model is still working* is the thing under test); "kick off the training run, keep churning, we can correct."

**Bootstrap decision (flagged for redirect):** the full teacher-rollout-on-SWE-bench pipeline (agent scaffold + teacher) is a multi-hour build *before* any training. To maximize churn, the first SFT bootstraps from **programmatic mutation data**: real Python functions (HumanEval/MBPP solutions) → inject one type/logic bug → **real pyrefly diagnostic** on the buggy version → teacher_trajectory (buggy → diag → fix) → interleaved reformat (A/C/D) → tokenized, loss-masked SFT data. This teaches the live-diagnostic format + reaction (the core skill) and validates the whole training pipeline; real teacher rollouts swap in at L2/L3.

**Key design details baked in:**
- **Loss masking = react-not-generate:** labels are -100 on the prompt + the entire ‹diag› block; loss is only on the agent's *fix* tokens. The model learns to CONDITION ON diagnostics, not emit them.
- **D = look-back via latency sweep:** D splices the diagnostic at `edit_pos + latency_tokens`, sampled from {0,2,8,32} per example (since pyrefly is sub-token-fast at 92.6 ms/tok, we train across offsets so the model generalizes to "live feedback arriving mid-work"). C splices at the edit boundary (offset 0). A has no diagnostic.

**Pipeline status:**
- Data-gen (`scripts/d_gen_sft_data.py`) — building (agent, non-GPU): mutation → pyrefly → reformat → tokenize+mask → `runs/d_sft_data/{A,C,D}/`, ~2–3k examples/condition.
- SFT (`scripts/d_sft.py`) — DONE: peft LoRA (rank 64, BF16, no QLoRA/trl needed), loss only on fix tokens, loads the per-condition datasets.
- First run plan: SFT **D layout first** (validate pipeline + format learning), then C and A for the matched comparison; eval = canary/held-out diagnostic-reaction, SFT'd vs base.

**Parallel (idle GPU):** 14B HumanEval baseline running (clean extractor) for the L2/L3 capability number.

**Tasks:** #30 (data-gen) in progress, #31 (SFT script) done, #32 (first SFT run) pending data. Will launch SFT inline on the GPU once data lands + validates, and keep churning toward a trained D model + eval.

---

## 2026-05-30 — L1 first SFT training chain LAUNCHED (D/C/A LoRA on interleaved data)

**Data ready (876/condition, validated).** `runs/d_sft_data/{A,C,D}/data.jsonl` — real pyrefly diagnostics, loss-masked (loss only on fix tokens; prompt+‹diag› masked; 0 diag-leak), D latency-swept {0,2,8,32}. Collator CPU-validated on real data (~71% masked, fix is the target, teacher-forcing correct).

**SFT smoke PASSED:** Qwen2.5-Coder-7B + LoRA (161M trainable, 2.08%), 20 D examples/1 epoch, saved adapter cleanly. Full peft+Trainer+device_map+grad-checkpointing path works.

**Training chain launched** (`scripts/d_sft.py`, background `bxuzmbn7b`): D → C → A, each 876 ex × 2 epochs, LoRA rank 64, bs 8 × accum 4, lr 1e-4 cosine. ~40 min each, ~2h total → `runs/d_sft/{D,C,A}/`. Then base-vs-SFT'd canary (`d_canary.py --adapter`).

**CAVEAT flagged for review (data signal):** smoke train_loss was low (0.083) because mutation-data "fix" is mostly a *copy* of the buggy code differing by a few tokens — the learning signal for *using* the diagnostic is concentrated in those few bug-fix tokens; most fix tokens are trivial copies. Consequence: the model may learn "output the obvious corrected code" without strongly depending on the diagnostic. The A-vs-D canary will reveal whether the diagnostic actually drives the fix (a D>A gap that widens post-SFT = success). If the signal is too weak, the upgrade is **real teacher rollouts** (multi-step edit-test trajectories where the diagnostic genuinely changes the agent's next action) — deferred to L2/L3 but may need to move earlier if this first run is inconclusive. This is the expected limitation of the bootstrap and the key thing to watch.

**Aside:** 14B HumanEval baseline timed out at 40 min (14B ~2× slower than 7B at batch 12) — non-blocking (L2/L3 number; 7B's 0.89 validates the substrate). Re-run later with longer budget / smaller subset.

**Status:** L1 training in flight. Tasks #30/#31 done, #32 in progress.

---

## 2026-05-30 — L1 first SFT readout: pipeline validated end-to-end; diagnostic-dependence inconclusive (eval ceiling + thin data). Need the SWE-bench regime.

**Trained:** D/C/A LoRA adapters (876 ex × 2 epochs, train_loss 0.052/0.033/0.043, ~1.9h each) → `runs/d_sft/{D,C,A}/`. Full data→SFT→adapter→eval pipeline works.

**3-way diagnostic-dependence probe** (`scripts/d_diaguse.py`, none/correct/wrong diagnostic, n=8 held-out fixtures):

| Model | none | correct | wrong |
|---|---|---|---|
| Base | 0.88 | 1.00 | 1.00 |
| D-SFT | 0.88 | 1.00 | **0.88** |
| C-SFT | 1.00 | 1.00 | 1.00 |
| A-SFT | 0.50 | 0.88 | 0.75 |

**Interpretation:**
- **Eval is ceiling-saturated.** Base coder fixes these toy bugs at 0.88–1.00 with OR without a diagnostic, and a *wrong* diagnostic doesn't fool it (wrong=1.00) — almost no headroom to detect dependence. n=8 → ±0.125 noise.
- **One faint directional signal:** D-SFT moved wrong 1.00→0.88 (the wrong diagnostic stopped spuriously helping) — consistent with learning to condition on diagnostic *content*, but within noise (one fixture).
- C-SFT saturated (no signal). A-SFT erratic (none=0.50 — likely narrow-distribution overfit on return-type-skewed mutation data, or noise).

**Verdict (honest):** the bootstrap-data caveat bit AND, more fundamentally, the toy eval cannot resolve this — a strong coder doesn't need a diagnostic for easy localized bugs; it finds them unaided. **Diagnostic value only manifests on hard bugs in complex code where the model fails without help — the SWE-bench regime, not toy fixtures.** What IS established: the full L1 machinery is functional (mutation→real pyrefly→interleaved SFT→3 condition adapters→3-way eval), and the mechanism (format consumption, splice, causal validity, payload equivalence) is solid.

**Next phase (needs Ian's direction — a big, consequential build, not a default):** the SWE-bench agent-rollout + eval pipeline — a coding agent that does multi-step edit-test-iterate on real repos with live LSP, producing (a) real teacher-rollout trajectories for richer SFT signal (replacing the copy-dominated mutation data), and (b) hard tasks where the base model fails unaided so the C-vs-D delivery-form effect has headroom to appear. This is the deferred Phase-0/L2 infrastructure; it's where the actual hypothesis gets tested. Flagged rather than started autonomously because it's a multi-day scaffold build with teacher-choice + scope decisions for Ian.

**Pausing autonomous churn here** — natural decision point. Toy-eval iteration won't resolve the question (ceiling); the principled next step is the SWE-bench pipeline, which warrants alignment. Everything durable; L1 pipeline + adapters retained for reuse.

**Aside:** 14B HumanEval baseline still pending (timed out earlier); re-run with longer budget when convenient.

---

## 2026-05-30 — L2 apparatus built + mechanism validated: continuous-stream agent (A/C/D) + TaskEnv

**Phase decisions (Ian, confirmed):** test *live in-flight* LSP feedback (D) vs turn-based-at-edit (C) vs none (A) for in-flight decision-making; SFT = self-distill + rejection sampling (teaching the interaction FORMAT, not capability); publishability via the **in-flight metrics** (rework-ratio primary, edit-cycles, time-to-correct), not just pass@1; **Goldilocks** task difficulty (base ~30-60%); 7B (14B reserve); zero-shot A/C/D first, then SFT; pilot → report → L2. (Full rationale in memory + plan §0.)

**Continuous-stream agent (`scaffold/stream_agent.py`) — built + mechanism validated.** One generation stream; parses completed `<edit>` search-replace blocks live; per condition: A never; C synchronous (pause-and-inject pyrefly diag at the edit boundary); D live (splice diag into the KV cache `latency_tokens` after the edit — the prototype-validated splice — no pause). Validated:
- Smoke (`d_agent_smoke.py`): gen + live edit-parse + apply + run_tests all work (solved a type-bug in 1 edit, all conditions resolved). Confirmed the canary lesson again: toy bugs are fixed first-try so the diagnostic never fires → need Goldilocks tasks.
- Plumbing (`d_agent_plumbing.py`, forced diag): **C delivers sync at the edit; D delivers async (text spliced into stream).** PASS both.
- **Design refinement:** in D, a `<done/>` while a diagnostic about the last edit is still in flight now *delivers it live and lets the model react before stopping* — the faithful "I'm done — oh wait, a squiggle appeared" dynamic, not a rigid turn boundary. Fixed pyrefly config (`project-includes` + `pyrefly init`; the basic preset reports nothing).

**TaskEnv (`harness/task_env.py`) — built + validated** (subagent). reset(clone@base + uv-venv@pinned-python + `pip install -e .` + apply/commit test_patch); read/list/apply_edit; pyrefly_diagnostics (per-task daemon, normalized, top-K, ranked at edited line); run_tests → **resolved = F2P pass AND P2P pass** (SWE-bench criterion); metrics → **rework_ratio**. Validated baseline-green on dask + 2 sympy; gold flips resolved=True on sympy-23950. **Pools frozen:** 15 train (SWE-Gym) + 10 eval (Verified: sympy/sphinx/pylint). Caveats: uv-venv python must not be `.resolve()`'d; pyrefly needs explicit site-package-path; aarch64 C-ext builds avoided.

**Now:** real-task de-risk running (`d_realtask.py` sympy-23950 / D) — does the agent produce valid edits + the live-diag path fire + resolve on an actual task. Then the **zero-shot A/C/D pilot** on Goldilocks eval tasks (oracle file-localization, fair across conditions) → first in-flight readout.

**Tasks:** #33 (TaskEnv) done, #34 (scaffold) in progress.

---

## 2026-05-30 — L2 apparatus COMPLETE; zero-shot A/C/D pilot launched

**Agent driving fixed → apparatus works end-to-end.** Real-task debugging surfaced two issues, both fixed:
1. **Per-token decode mangled the edit markers** → switched to full-sequence decode each step.
2. **My custom `<edit path="...">...</edit>` wrapper confused the model** (it emitted the right fix in the wrong shape, then sometimes dumped the whole file). → switched to the **standard aider SEARCH/REPLACE format** the model knows natively, single target file (oracle localization), + a **one-shot example** in the system prompt + "never reproduce the whole file" + **line-numbered file presentation** (a bare ```python fence primed whole-file mirroring).
Result (`d_format_check.py`): the 7B now emits clean minimal edits and **resolved a non-trivial LRU bug** zero-shot (`pop()`→`pop(0)`), and earlier produced the *correct* sympy-23950 fix (`Contains(BooleanFunction)`→`Contains(BooleanFunction, Set)`) — it was purely format-adherence, not capability.

**Zero-shot A/C/D pilot launched** (`scripts/d_pilot.py`, base 7B, no SFT, 6 Verified tasks, reset-once + cheap git-restore between conditions, incremental `runs/d_pilot/results.json`). Measures per condition: **resolve rate** (A reveals the Goldilocks band), **rework-ratio** (primary in-flight metric), edit-cycles, diagnostic-injections, tokens. This is the first meaningful readout: does *live* feedback (D) change the trajectory vs sync-at-edit (C) vs none (A) — on an untrained model that already uses diagnostics. Early-check watcher armed for task 1.

**Next:** integrate the pilot readout → if directional (D rework < C < A, or D resolves where A/C don't), that's a publishable-direction signal even zero-shot → self-distill SFT to amplify → re-eval → report at the pilot readout. If flat, the signal needs SFT and/or harder tasks.

**Tasks:** #34 (scaffold) done, #35 (pilot) in progress.

---

## 2026-05-30 — Zero-shot A/C/D pilot READOUT: apparatus works, but SWE-bench Verified too hard for the zero-shot single-file 7B → regime fork

**Result (6 Verified tasks, base 7B, no SFT):**
| Cond | resolve | mean_rework | mean_cycles | mean_diag |
|---|---|---|---|---|
| A | 0/6 | 0.000 | 0.00 | 0.00 |
| C | 0/6 | 0.091 | 0.17 | 0.33 |
| D | 0/6 | 0.000 | 0.00 | 0.17 |

**Per-task:** only `sympy-23950` produced edits (A 1/0diag, **C 2 edits/2 diag/reworked 0.54**, D 1 edit/1 diag/no rework). The other 5 (sympy-23824/23413/20916/16766, sphinx-10466) → **0 edits parsed in all conditions** (parsed==applied everywhere, so NOT a SEARCH-match bug — the model emitted no edit block at all; sympy-16766 rambled to the 1500-token cap).

**Diagnosis:**
1. **These Verified tasks are too hard for the zero-shot, single-file, single-stream 7B agent** — 0/6 resolved, and on 5/6 the model can't even formulate an edit. They need codebase *exploration* + test-iterate over many turns + a stronger model. My pilot agent is deliberately minimal (one file, one continuous stream, no exploration/test-loop) — too weak for real SWE-bench, and the 7B is likely too weak regardless.
2. **The C-vs-D effect lives in the multi-step edit→error→react loop**, which the zero-shot model barely does (it edits once + `<done/>`). The ONE engaging task (23950) showed the dynamic in miniature: **C's synchronous pause forced a reaction (2 edits, reworked); D's live diagnostic landed on-done and the untrained model didn't re-engage.** Consistent with the hypothesis that *untrained, forcing-sync (C) > ambient-live (D), and SFT teaching the model to use live feedback should flip it.*

**What's validated:** the full apparatus runs end-to-end (6 tasks × A/C/D, reset+restore, real pyrefly diagnostics fire + inject on real tasks, in-flight metrics captured). The mechanism + infra are sound. The blocker is the **eval regime**, not the apparatus.

**Direction fork (for Ian — this is the pilot-readout checkpoint):**
- (a) **SFT first** — self-distill the react-to-feedback behavior on the D-layout data we already built, then re-eval. Teaches the bottleneck behavior (engage + react to live feedback). But won't make a 7B solve real sympy bugs that need exploration.
- (b) **Controlled Goldilocks regime for the mechanism study** — bugs where the model engages + feedback is relevant (mutation-injected on real files, calibrated ~30-60% solvable), isolating the C-vs-D delivery-form effect cleanly; keep SWE-bench as the *downstream* harder headline. Cleaner science, less "SWE-bench headline" immediately.
- (c) **Stronger model (14B/32B) + a fuller agent** (explore + run-tests + iterate over turns) for real SWE-bench engagement. The "real" setup, much bigger build + compute.
- My lean: **(b)+(a)** — get the clean mechanism signal in a controlled regime + SFT the react behavior, then scale to (c) for the publishable SWE-bench headline. But this trades off the "SWE-bench result" immediacy Ian wanted, so it's his call.

**Status:** pilot complete (#35). Apparatus proven; regime decision pending Ian. `runs/d_pilot/{results,summary}.json`.

---

## 2026-05-31 — Major arc: pilot → agentic loop → agent tar-pit → multi-stream realization → PIVOT to validating Hooper single-stream interleaving via a ladder

This entry consolidates several decisions made across 2026-05-30/31 so the record is clean (per Ian).

**1. Zero-shot SWE-bench pilot readout (recap):** custom continuous-stream agent, base 7B, 6 Verified tasks → **0/6 resolved, 5/6 produced no edit** (model couldn't engage). Apparatus sound; blocker = the agent was ONE-SHOT (no try-and-correct) and the tasks too hard.

**2. Decision (Ian): agent must be an agentic try-and-correct LOOP.** Added a `<test/>` action (run suite → results back → iterate). Considered **mini-swe-agent** (Ian's suggestion): proven, ~100-line, turn-based, bash-only. **Decision: KEEP the custom continuous-stream agent (not mini-swe-agent)** so D stays the purest *mid-generation* live splice — mini-swe-agent is turn-based, so its 'live' is only turn-granular; the two are coupled, Ian chose novelty/purest-live.

**3. The custom-agent TAR PIT (the evidence that forced a rethink):** building the agentic loop on a 7B hit FIVE successive friction points — edit-format markers dropped → bare SEARCH/REPLACE → EOS/turn-ending (model ends its chat turn after each action) → **parroting** (the model echoes any content spliced into its own assistant stream) → multi-turn regex mis-parse (oscillation, no convergence). Each fixable, but classic onion-peeling: the custom continuous-stream paradigm fights the chat model's training at every level. mini-swe-agent (bash+turns) avoids all of it because that's what models are trained on.

**4. The conceptual realization (Ian):** the **parroting IS the multi-stream problem**. A single stream conflates *generation* and *consumption* — the model can't tell injected tokens from its own, so it echoes them. The Geiping multi-stream architecture dissolves this by construction: separate output/input channels, the model only READS the input channel → parroting impossible, true parallel generate-and-consume. We pivoted away from multi-stream (released model couldn't code) and thereby took on the burden of *teaching* a single-stream model the separation that multi-stream gives for free. Mid-generation live D is therefore **SFT-gated** (an untrained model parrots; SFT must teach ‹delim›=external-input).

**5. DECISION (Ian) — the new plan:** pursue **Hooper/TM single-stream interleaving validated via SFT** (NOT the heavy Geiping multi-stream-on-a-coder, ~weeks). Ian: "not convinced interleaving can't work if we train it — do the training to validate, and ladder it: build evals, set up self-distillation, monitor progress." **Get a clean recipe for trained interleaving FIRST (a contribution on its own), THEN layer on LSP.** This de-risks the load-bearing assumption in isolation and sidesteps the agent tar-pit (no agent needed yet — plain generation + a splice).

**The ladder (current workstream):**
- **R0 — consumption eval** (`scripts/i_eval.py`): the measuring stick. Model must USE a *random* value injected mid-generation as ‹info›…‹/info› (headroom-guaranteed — can't be guessed). Metrics: REACTION (generated code runs and returns the injected value) + PARROTING (echoes the ‹info› delimiters → should →0 trained). Arms: inject vs no-inject; base vs SFT'd. Running on base now to set the parrot/no-react baseline.
- **R1 — self-distill data** (`scripts/i_gen_data.py`, DONE): 2400 constructed sequences, ‹info›FACT‹/info› interleaved mid-function + **loss-masked** (the value is only knowable from the inject), loss on the function body. Mean seq-len 107, ~22 loss tokens. Masking verified.
- **R2 — train + monitor:** LoRA SFT (`scripts/d_sft.py` reused); eval reaction↑/parroting↓ at checkpoints to SEE if the separation is being learned.
- **R3 — layer on LSP:** once the recipe is clean, the diagnostic is a richer injected input; bring back the coding agent (`scaffold/stream_agent.py`, `harness/task_env.py` retained).

**Status:** plan v0.5 §0 (single-stream interleaved) still holds; this refines the *execution order* — validate the interleaving mechanism cleanly before the LSP+agent layer. Reformat/loss-masking/SFT pipelines carry over. Tasks: #36 (custom agent) superseded; #37 (R0 eval) in progress; R1 data done.

---

## 2026-05-31 — R0 baseline: clean forward-injection works ZERO-SHOT (16/16, 0 parrot) → reframes "SFT-gated"; build the discriminating backward-revision eval

**R0 forward-injection eval, base 7B (`scripts/i_eval.py`, prefill signature → splice ‹info›VALUE‹/info› → complete):**
- no_inject: reaction 0/16 (can't guess the random value — headroom confirmed).
- **inject: reaction 16/16, parrot 0/16.** Reaction lift +1.000.

**Key reframe:** a *single clean* mid-generation injection is consumed **perfectly zero-shot, no parroting**. So basic single-stream interleaving (use injected forward-looking info) does NOT need SFT. The agent's parroting was NOT inherent to mid-stream injection — it came from **repeated** injections (test-results + diagnostics) in a long multi-turn stream (and was already cured by delivering tool-results as user observations). This revises the earlier "mid-gen D is SFT-gated" claim: the basic mechanism is fine zero-shot; what's hard (and likely needs SFT) is the LSP-specific behaviors:
1. **Backward revision** — react to feedback about output ALREADY written by *revising* it, not just using new info going forward. THE core LSP behavior, and the discriminating test.
2. **Robustness to frequent/noisy interleaving** (the agent regime).

**Next (R0b):** a backward-revision eval — show the model a buggy attempt, inject ‹diag› about the bug mid-stream, measure whether its continuation produces a CORRECTED version (vs ignoring it). Base likely fails (models continue forward, don't revise) → that's where SFT's value should show. Then train on revision-style self-distill data + re-eval.

**Status:** R0 (forward) done — mechanism sound zero-shot. R0b (backward-revision, the real test) next. Plan/recipe unchanged; this sharpens WHAT the SFT must teach (revision + robustness, not basic consumption).

---

## 2026-05-31 — R0b: backward-revision ALSO works zero-shot (+0.70 lift) → interleaving is sound untrained; REFRAME to the delivery-form (C-vs-D) efficiency question

**R0b backward-revision eval, base 7B (`scripts/i_eval_revise.py`):** buggy attempt shown → ‹diag› injected mid-stream → does the continuation emit a corrected version?
- no_diag: **3/10** (spontaneously fixes a few obvious bugs).
- diag: **10/10**. **Revision lift +0.70** — the diagnostic causally drives the fix.

**Combined with R0 (forward 16/16):** the base 7B consumes mid-stream interleaved feedback ZERO-SHOT — both forward-looking info AND backward-looking revision — with no parroting. **The core single-stream interleaving mechanism does NOT need SFT.** The earlier "SFT-gated" conclusion was wrong; the agent's parroting/failures were ENGINEERING (repeated injections in long multi-turn streams, edit-format, turn-handling), not an inability to consume live feedback.

**REFRAME (the research question sharpens):** since *consuming* live feedback works zero-shot, the open question is no longer "can the model use interleaved feedback?" (yes) but **"does the DELIVERY FORM matter — does LIVE/mid-generation feedback (D) produce better in-flight decisions than SYNC/turn-boundary feedback (C)?"** For a single error, C and D both fix it — the delivery-form effect lives in the **efficiency of multi-step generation**: live feedback should reduce *rework* and *time-to-correct* (the model corrects as it goes vs going further down a wrong path before a turn-boundary batch). This is exactly the in-flight-metrics hypothesis (rework-ratio primary) — and it's now testable cleanly, zero-shot, WITHOUT the heavy SFT or the finicky agent.

**Next:** a controlled **C-vs-D efficiency eval** — a multi-error task where the model fixes errors as feedback arrives; compare LIVE (D, diagnostics mid-generation) vs SYNC (C, diagnostics batched at a turn boundary) vs NONE (A); measure rework, time-to-all-correct, final pass. If D < C on rework / faster convergence → the delivery-form effect is real even zero-shot (the headline). SFT/self-distill may still AMPLIFY it and is needed for the noisy/frequent agentic regime, but the core can be measured now.

**Status:** interleaving mechanism validated zero-shot (R0+R0b). Pivoting the next eval from "does interleaving work" (answered: yes) to "does delivery form matter" (the actual thesis). Big simplification — the SFT-heavy/agent-heavy path is not required to get the core signal. Tasks: #37 (R0) done, #38 (R0b) done.

---

## 2026-05-31 — R0c: C-vs-D efficiency, clean zero-shot result — D solves at MATCHED correctness with ~half the tokens of C

**C-vs-D efficiency eval, base 7B (`scripts/i_eval_cd.py`).** Task: write `config()` returning a dict mapping six keys to a configured random integer the model cannot guess. Same fact delivered three ways:
- **A (none):** **0/12** — the value is unguessable; floor confirmed.
- **C (sync/late):** model writes a full guessed attempt, gets the value at a turn boundary, rewrites. **12/12 correct, 139.0 tok.**
- **D (live/early):** value spliced in right after the signature, before the body. **12/12 correct, 70.5 tok.**

**At MATCHED correctness (both 12/12), D uses 68 fewer tokens than C — roughly half.** This is the delivery-form efficiency effect, zero-shot: sync feedback forces a full wrong attempt + rewrite (rework); live feedback lands before the wrong work is done, so it's written once.

**First cut was a bug** (all arms 0/12): the original "BASE + index" task made the model reference an undefined `BASE` variable / miscompute the index. Fixed to a literal-value, no-index task → arms now actually correct, so the efficiency comparison is valid.

**Honest caveat:** in THIS contrived task the gap is partly by-construction — C is *defined* to write a full attempt (n1) then revise (n2), so n1+n2 > D's single pass almost tautologically. That structural inevitability IS the mechanism (sync = wasted attempt on a wrong foundation), but a contrived single-fact task makes it look stronger/cleaner than it will be in the wild. **The real test is a realistic multi-step setting** where (a) the model doesn't know in advance which feedback matters, (b) feedback is noisy/frequent, (c) rework cascades across steps. That's the agentic-coding regime with live LSP — and it's where the headline rework-ratio metric earns its keep.

**Status:** the thesis is now *operationalized and positive zero-shot* on a clean toy (A floor, C and D both correct, D ~2× more efficient). Three legs stand: R0 forward 16/16, R0b revision +0.70, R0c efficiency D≈½C. Next fork (decision for Ian): (1) **strengthen the toy** — multi-error / unknown-relevance / cascading tasks to kill the by-construction caveat; (2) **SFT amplify** — train the LoRA on the 2400 interleaved examples and re-run R0/R0b/R0c to see if the delivery-form gap widens; (3) **go realistic** — fold live LSP into the coding agent (now de-risked: mechanism works zero-shot, so the agent needs robustness, not capability) and measure rework on real edits. Task #39 (R0c) done.

---

## 2026-05-31 — R1 (realistic): A/C/D agent apparatus WORKS end-to-end on real pyrefly — but toy tasks 1-shot (A=C=D, no headroom). Edit protocol forked SEARCH/REPLACE → whole-file REWRITE.

Carried the C-vs-D question into a real edit-test-fix agent loop (`scripts/agent_acd.py` driving `scaffold/stream_agent.py` over `scaffold/mock_env.py` with the REAL pyrefly CLI). Headline metric = mean `rework_ratio`.

**Re-hit the agent tar pit, then climbed out:**
- SEARCH/REPLACE with the 7B is **destructive and unrecoverable**: the model's first edit wiped the function body, leaving an empty shell, and SEARCH/REPLACE can't rebuild deleted code (nothing to match). It then looped emitting identical no-match edits. Three engineering fixes landed: (1) current-file view injected each turn (condition-neutral) so SEARCH can match; (2) tolerant parser (handles the `SEARCH…END REPLACE…END` drift the 7B emits) + no-op detection; (3) anti-degeneracy **bail** after 5 consecutive non-applying edits. These stopped the infinite loops but did NOT stop the file corruption — SEARCH/REPLACE is just too brittle for a weak agent.
- **Decision — edit protocol fork:** added a whole-file **REWRITE** protocol (`<rewrite>…</rewrite>` → `MockEnv.rewrite_file`, rework measured by difflib delete+replace chars on the old side). No match-failures, no irrecoverable corruption: a bad rewrite is simply rewritten. Preserves the continuous-stream + mid-generation-splice apparatus D needs. **`edit_mode="rewrite"` is now the default.**

**Result (rewrite mode, base 7B, 4 single-bug tasks, A/C/D):** every task resolved **4/4 in ALL conditions**, 1 edit, **rework 0**, identical token counts (A=C=D=33.75 tok). Apparatus validated; **but zero signal** — the base **one-shots** every toy bug (undefined name, int+str, list.max(), len-as-int), so the LSP diagnostic never influences anything and there's no rework to separate live (D) from sync (C). Classic **no-headroom** (same lesson as R0 no_inject): for delivery-form to matter the FIRST attempt must often be WRONG.

**Status / next:** the realistic apparatus is DONE and de-risked (real pyrefly, A/C/D, rewrite protocol, anti-degeneracy). The bottleneck is now **task difficulty** — need Goldilocks tasks (base resolves ~30–60% un-aided) where first attempts fail, errors cascade, and several feedback rounds occur, so timing can bite. Plan: build harder multi-step synthetic tasks first (fast, controlled — confirm the C-vs-D signal exists at all), THEN scale to real repos via `harness/task_env.py`. Task #40 (apparatus) done; #41 = Goldilocks task design.

---

## 2026-05-31 — MAJOR DIRECTION (Ian): real agent harness + real SWE-rebench tasks + EFFICIENCY headline (not rework)

Ian redirected the realistic phase after the toy A/C/D run. Four changes, and how they reshape the experiment:

1. **Efficiency is the headline, not rework.** The question is now: *does live, non-blocking in-stream feedback reduce total cost — wall-clock and tokens (INPUT and OUTPUT counted separately) — to reach a resolved state, vs sync turn-based feedback (C) vs none (A)?* Rework feeds into that but is downstream. Sharpens the project name: "Streams" = **non-blocking** feedback.
2. **The model owns edit granularity.** Drop the imposed rewrite/search "modes." Expose a standard editing tool (str_replace-style) + test runner; the model decides whether it touches one line or rewrites a function (like Claude Code / mini-swe-agent / aider). My job = make the tool robust (clean apply-or-reject, diffs shown, never silent corruption), NOT constrain the model. TaskEnv.apply_edit already does unique-match apply-or-reject — the right primitive.
3. **Live-vs-sync becomes about BLOCKING.** In a normal harness the model emits an edit and BLOCKS for the result, so "sync LSP" (C) just rides diagnostics back in that result. For **D to differ at all the model must NOT block** — it streams ahead (more reasoning/edits) while diagnostics arrive asynchronously mid-stream and catch up. That non-blocking stream IS the contribution (and my continuous-stream apparatus is the right substrate). The model still chooses when to yield; D lets it safely run further first. Concrete input-token angle: D splices into the existing KV cache (cheap) while C/turn re-feeds a fresh observation (re-tokenized) — D may win on input tokens too. (Flagged to Ian for correction; he can veto the non-blocking framing.)
4. **Natural cascades, real tasks** — no hand-built interdependent toys.

**Decisions (AskUserQuestion):**
- **Task source = SWE-rebench** (nebius/SWE-rebench), filtered to `created_at` year >= 2025 (after Qwen2.5-Coder-7B data cutoff → strongest "not in training" defense). Real repos, executable F2P/P2P tests via `harness/task_env.py`. We IGNORE the x86_64 docker_image and build envs natively on aarch64 (clone + uv-venv + `pip install -e .`).
- **Agent model = keep Qwen2.5-Coder-7B** (live KV-splice validated; weak agent so absolute resolve will be low — measure efficiency on the solvable/partial subset; risk = floor too low for A/C/D separation, accepted).

**Recon done:** SWE-rebench schema = standard SWE-bench + created_at + install_config/requirements (pinned) + x86 docker_image. `task_env.py` already targets this schema (apply_edit unique-match, PyreflyDaemon persistent LSP, F2P/P2P run_tests) — needs a rebench loader + maybe install_config wiring. **Selected 25 tractable post-cutoff tasks** (single src file, patch 386–906 chars, 1–3 F2P; circuitbreaker, dacite, func_adl, sentry-python, dvc, …) → `runs/rebench/candidates.jsonl`.

**Status:** de-risking substrate now (`scripts/rebench_smoke.py`: provision candidates natively, verify baseline F2P-fails+P2P-passes). Next: build the real non-blocking tool-based harness (#43) + the input/output-token + wall-clock efficiency metric, then A/C/D on the provisioned set. Tasks #42 (substrate), #43 (harness).

---

## 2026-05-31 — R2 substrate + harness BUILT and functional on real SWE-rebench; edit primitive pivoted str_replace → LINE-RANGE

**Substrate de-risked.** `scripts/rebench_select.py` filtered nebius/SWE-rebench to 25 post-2025, single-src-file, small-patch (386–906 char), 1–3 F2P tasks → `runs/rebench/candidates.jsonl`. `scripts/rebench_smoke.py` provisioned them natively on aarch64/GB10 (ignore x86 docker_image; `uv pip install -e .` — ALL 8 sampled installed, fast). **4 well-formed** (F2P fails at base = bug present, sampled P2P pass): `iris-hep__func_adl-185`, `gaogaotiantian__coredumpy-61`, `ASPP__pelita-875/863` → `runs/rebench/provisioned.jsonl`. (Ill-formed: dacite/sentry NO_OUTPUT, circuitbreaker P2P errors — test-id quirks, refine later.)

**Harness built** (`scaffold/stream_agent.py` evolved + `scripts/agent_swe.py`): non-blocking continuous stream, model-chosen edits against `TaskEnv` (multi-file via `<read path/>`, real PyreflyDaemon LSP), A/C/D delivery, input/output token + wall-clock + rework accounting. The model yields on `<test/>`/`<read>`; edits don't block (D splices diagnostics mid-stream as edits complete; C batches them to the next yield).

**Edit-primitive pivot (the key engineering finding).** `str_replace` (TaskEnv unique-match) is unusable for the weak 7B on real files: on func_adl (982 lines) it emitted ```python-fenced, tiny non-unique SEARCH snippets → **0/5 edits applied**, bailed, zero signal. Switched to **LINE-RANGE editing** (`apply_line_edit` on TaskEnv + `<edit path lines="A-B">…</edit>` off the numbered file view; fence/prefix stripping) — model picks the line span (still chooses scope). Immediately functional: on func_adl D the model **reasoned 619 tok, applied 2 real edits (lines 585-586), got LIVE diagnostics spliced mid-stream, iterated 5 tests, rework=0.011**. Not resolved (func_adl is hard for a 7B — expected; weak-agent floor accepted) but the apparatus now produces real edits + real LSP + real in-flight metrics.

**Token/speed control.** Full-file re-feed each turn blew input to 149k tok / 764s (982-line file × 10 turns, incl. test-only spinning). Added a `file_changed` gate: re-show the numbered file only when it changed since last shown (condition-neutral; also more realistic — real harnesses don't re-feed unchanged files). Final F2P+P2P run is authoritative; in-loop `<test/>` caps P2P=5 for speed.

**Status:** apparatus real and functional. Running the first A/C/D efficiency matrix (4 tasks × A/C/D, `runs/agent/swe_acd.json`). Open caveats: 7B may resolve ~0 (floor too low to separate A/C/D — if so, need easier tasks or accept partial-progress/rework as the signal); pyrefly diagnostic quality on real repos TBD. Task #43.

---

## 2026-05-31 — Real SWE-rebench: LSP fires but is NOISE on logic bugs → task-TYPE gates the LSP-delta. Pivot to synthetic type-signal tasks; subagent cross-review sharpens design.

**Region-scoped A/C/D (3 smallest real tasks): still 0/3 resolves** but engagement fixed (oracle localization → 7-12 edits vs 0). The 7B thrashes (rework 0.5-0.67), can't solve real bugs even when localized.

**KEY DIAGNOSTIC FINDING (Ian asked to check LSP quality).** Pyrefly diagnostics DO fire on edits, but on these real bugs they are NOISE not signal: `unused-import`, `unused-parameter`, `untyped-import: install types-networkx` (env noise repeated every turn), `parse-error` (from the model's OWN broken edits), `bad-override` (pre-existing). **Not one diagnostic points at the actual bug** — because SWE-rebench bugs are LOGIC bugs and pyrefly is a TYPE checker, structurally blind to them. → A≈C≈D is guaranteed regardless of model strength. **Task TYPE (does the bug yield a guiding type diagnostic?) gates the LSP-delta more than model SIZE.** This reframes 32B: it'd lift fix-rate but show no LSP-delta on logic bugs. (32B still worth a throughput/ability probe per Ian, but lower priority than task-type.)

**Premise confirmed:** on a natural unguarded-Optional bug, pyrefly fires a clean guiding error at the bug line (`missing-attribute: NoneType has no attribute qty`). So type-signal tasks are viable.

**Synthetic seed batch (`scripts/synth_tasks.py`, 6 tasks) + subagent cross-review (Ian-requested).** Three critics (confounder/defensibility, Python-realism, eval-difficulty) CONVERGED:
- **4/6 too easy** (reflexive 1-token/1-guard fixes; docstrings spell out intent) → near-ceiling base pass@1, no headroom.
- **"Loud-traceback" confounder:** bugs that crash with a message NAMING the fix (`'Point' has no attribute 'row'`) → A free-rides on the test traceback (all conds get it) → A≈D. LSP's value is strongest when the runtime signal does NOT name the fix.
- `group_parity_container` contrived; Optional over-weighted (1&6 dup); annotations a "tell"; cascade under-used.
- Methodology: greedy pass@1 deterministic → must SAMPLE (temp~0.7, k≥8) for fractional rates; calibrate SUITE MEAN to 30-60% empirically; verify A>0 per task AND drop A≈C≈D tasks; separate first-try vs within-N-iterations.

**SHARPENED DESIGN (the mechanism):** live LSP beats batched/none cleanest on **multi-site type-error cascades** — pytest reveals broken sites ONE AT A TIME (fix→retest→next site), so A grinds serially with rework; pyrefly reveals ALL sites at once → C/D fix together, D live. Crisp, measurable, naturally cascading (no contrivance). Rebuild the set around multi-site cascades + harder single-site (distractors, intent-in-tests-not-docstrings, partial typing); auto-flag loud-traceback tasks; add a base pass@k calibration harness. Task #44.

---

## 2026-05-31 — Synthetic multi-site cascade tasks: first A/C/D resolve+efficiency. Mechanism appears (1 clean D-win) but high-variance zero-shot; set mis-calibrated.

Rebuilt task set (`scripts/synth_tasks.py`, 8 multi-site type-signal tasks) run through the SAME agent harness (`scripts/synth_acd.py`, MockEnv + real pyrefly + line edits), greedy, 1 seed.

**Aggregate (noisy, hides the story):** A resolve 2/8, C 1/8, D 2/8. Below the 30-60% Goldilocks band.

**Per-task (the real signal):**
- **`lookup_optional_cascade` — D-WIN (the predicted mechanism):** D resolves in 5 edits / 5 tests / 0.77 rework; A and C THRASH (13 edits, 0.92 rework) and FAIL. Live feedback prevented the serial-thrash on the 2-site Optional cascade.
- **`return_container_ripple` — A-WIN (counter):** A resolves cleanly (3 edits); C and D thrash (12-13 edits) and fail.
- `grid_field_rename`: all three resolve, but D LESS efficient (3 tests/2 edits/0.56 rework vs A/C 1 test/1 edit).
- **`config_truthiness_distractor` (negative control, 0 type signal): A=C=D BYTE-IDENTICAL, 0 differentiation** — validates that the differences elsewhere are really the LSP channel, not noise. Good.
- Engagement failures: `records_arity_drift`, `mutable_default_none` → **0 edits** (model rambled, never emitted a parseable edit); `fmt_signature_drift`, `parse_branch_ripple` → thrash into the 1400-token budget.

**Read:** the mechanism is REAL but HIGH-VARIANCE zero-shot — live feedback helps when the model uses it well (lookup_optional) and distracts when it doesn't (return_container). Untrained, it's a double-edged sword. This MOTIVATES the SFT amplifier (train the model to use live feedback reliably → consistent D-win) — consistent with R0c (clean toy D-efficiency) being noisier in the agentic loop. The negative control proves the channel is doing something.

**Calibration problems to fix before the A/C/D comparison has power:** (1) too hard (resolve <30%) — ease tasks into 30-60%; (2) 0-edit engagement failures — inspect (save streams) whether format/prompt or capability, likely some are fixable; (3) raise token budget so thrash-tasks can finish; (4) SAMPLING + multiple seeds (greedy 1-seed has no power, D-wins and A-wins cancel); (5) expand the set (8 too few). Task #44.

---

## 2026-05-31 — 0-edit DIAGNOSIS: harness bug (single-line edit format dropped), not task difficulty. Fixed. Invalidates prior resolve numbers (undercount).

Diagnosed the three "0-edit" tasks (stream capture). **The model WAS emitting edits** — in the form `<edit path="sol.py" lines="13">` (a SINGLE line number). But `LINE_EDIT_RE` required `lines="N-M"` (a dash range), so every single-line edit was **silently dropped** → file unchanged → identical test failure → the model repeats the SAME edit forever to budget (and fail_streak never increments because the edit isn't even detected as failed → no anti-degeneracy bail). Hence "0 edits / 8 tests / hit budget."

**The dropped edits were often CORRECT:** `config_truthiness_distractor` → the model wrote the right `if v is None: ... else ...` (preserves the legit 0); `mutable_default_none` → `add_tags([])` which fixes the only no-arg call. Both would have RESOLVED. (`records_arity_drift` → the model mis-localized, blaming `summary`'s return instead of the `for name, qty` unpacking — a real localization miss that pyrefly's `bad-unpacking @L3,L9` would help with, IF edits applied.)

**Impact:** this bug suppressed ALL single-line `lines="N"` edits across every synth AND SWE run so far → resolve numbers are an UNDERCOUNT; the prior synth aggregate (A 2/8, C 1/8, D 2/8) is not trustworthy and must be re-run. **Fix:** `LINE_EDIT_RE` now accepts `lines="N"` and `lines="N-M"` (END optional; end defaults to start). The pattern the user asked us to find = a format-tolerance gap, exactly the class of weak-7B engineering friction we hit before (SEARCH/REPLACE markers, EOS turns, fences).

**Next (logged plan):** (1) validate the fix on the 3 diagnosed tasks; (2) full calibrated re-run — regex fix + higher token budget (1400→2500) + SAMPLING (temp 0.7) + multiple seeds for statistical power; (3) if a pattern holds, add fresh tasks. Decision (Ian): pursue a clean CALIBRATED ZERO-SHOT A/C/D number; SFT only if needed. Task #44.

---

## 2026-05-31 — POWERED zero-shot A/C/D result (synth, temp 0.7 × 4 seeds): live feedback (D) does NOT help and modestly HURTS resolve; trades resolve-rate for efficiency-when-successful. Motivates SFT.

`runs/agent/synth_acd_v2.json` — 8 synth multi-site tasks × A/C/D × 4 seeds (temp 0.7), regex fix + budget 2200. Calibration SUCCEEDED: suite resolve mean ~0.31-0.47 = Goldilocks.

**Resolve rate (n=32/condition):** A 14/32 (0.44), C 15/32 (0.47), **D 10/32 (0.31)**.
**Efficiency among RESOLVED (matched correctness):** D fewest out-tokens (362 vs A 424, C 450) and fewest test round-trips (2.8 vs 3.4/3.5).
**Per-task:** D never > A and never > C on resolve — loses or ties on ALL 8 tasks (D<A on config/lookup/fmt/parse, tied elsewhere; D<C on config/lookup/mutable, tied elsewhere). C ≈ A throughout (sync LSP doesn't beat no-feedback either). `records_arity_drift` 0/0/0 (too hard — unpacking mis-localization). Negative control (config, 0 type signal) shows no consistent D benefit.

**INTERPRETATION (the crux result so far):** zero-shot, in the agentic loop, live mid-stream LSP delivery (D) **trades resolve-rate for efficiency-when-successful** — it pushes the model to converge faster WHEN it works, but more often DISTRACTS it (mid-stream splices, including diagnostics about the model's OWN transient broken edits, interrupt coherent editing) → lower resolve. Even batched LSP (C) doesn't beat none (A). This cleanly separates two claims: (1) the model CAN consume a clean interleaved injection — R0 16/16, R0b +0.70, R0c toy D≈½C (yes); (2) live delivery HELPS an untrained agent in the loop (NO — neutral-to-harmful). The gap between (1) and (2) is exactly what the SFT amplifier exists to close. This is a publishable-shaped negative-for-D zero-shot result that motivates training.

**Caveats:** modest power (n=32/cond); the efficiency-when-resolved D-advantage is intriguing but on only 10 resolved D runs. Possible delivery confound: D may hurt partly because it splices diagnostics about the model's own transient errors live (a debounced/filtered D, or delivering at a natural pause, might recover benefit) — a cheap probe before/instead of SFT.

**Decision point (for Ian):** (a) accept the zero-shot negative + run the SFT amplifier (the laddered plan; shows the before/after that earns the headline), or (b) first a cheap D-delivery tweak probe (debounce / don't splice own-edit diagnostics / deliver at a pause) to see if zero-shot D can be made to help. Task #44.

---

## 2026-06-01 — D-delivery PROBE (debounce + pause-align + announce): doesn't recover resolve, but DEBOUNCE creates a genuine MATCHED ~2× efficiency gain. Rigorous speed↔reliability trade-off.

Tuned-D = debounce 24 tokens (re-query pyrefly only after the stream settles past the last edit; deliver CURRENT state so transient self-inflicted squiggles vanish) + pause-align (deliver at newline) + announce_lsp (prompt tells the model LSP is inline). `scaffold/stream_agent.py` flags `debounce/pause_align/announce_lsp`; `runs/agent/synth_dtune.json` (D only, temp 0.7 × 4 seeds; A/C reused from v2, identical per seed).

**Resolve (n=32):** A 0.438, C 0.469, **D-immediate 0.312, D-tuned 0.281**. Tuning did NOT recover resolve (still the lowest). Per-task D-tuned mostly ≤ others (fmt 0/4, but records_arity 1/4 = first non-zero — debounced bad-unpacking diag landed once).

**Efficiency — RESOLVED-ONLY (biased):** D-tuned out=254 tok / 1.9 round-trips, best by far (vs C 450/3.5).
**Efficiency — MATCHED PAIRS (selection-bias-free; same task+seed both resolved) — THE clean number:**
- **D-tuned vs C (6 pairs): 1.7 vs 3.2 round-trips, 232 vs 408 tokens** — D-tuned solves the SAME problems in ~half the iterations/tokens.
- **D-immediate vs C (8 pairs): 2.5 vs 2.5, 322 vs 321 — IDENTICAL.** Immediate-D's resolved-only edge was pure selection bias.
→ **The DEBOUNCE/pause-align is what turns live feedback from distraction into a real matched efficiency gain.** Delivery DESIGN is a first-class variable.

**CONSOLIDATED RESULT (robust across 2 D variants):** zero-shot, in the agentic loop, live in-stream LSP feedback is a **speed↔reliability trade-off**: it LOWERS resolve (reliability) but, when *debounced*, solves the problems it does solve ~2× faster (matched efficiency). The model can consume clean single injections (R0-R0c) but isn't trained to use live feedback without derailing in the messy multi-turn loop. Caveat: matched-pair n small (6) — needs more seeds for power; resolve drop ~5/32 also modest-n but consistent across both D variants.

**Story shape for write-up:** (1) interleaving consumable in isolation (R0/R0b/R0c); (2) naive live delivery HURTS an untrained agent's reliability in-loop; (3) delivery design (debounce/pause) recovers a clean matched efficiency gain but not reliability; (4) closing the reliability gap = the SFT amplifier (or richer constructive signal). This is a coherent, honest, novel arc.

**Decision point (Ian):** (a) POWER UP — more seeds/tasks to firm the resolve-drop + the 1.7-vs-3.2 matched-efficiency claim (cheap-ish, needed for any claim); (b) SFT amplifier — train to use live feedback without derailing (close the reliability gap = the headline fix); (c) RICHER SIGNAL — autocomplete-style / go-to-def constructive options instead of error diagnostics (Ian's idea; novel, may help reliability by offering the fix not just the error). Task #44.

---

## 2026-06-01 — Decision (Ian): POWER UP (more tasks + seeds), SHELVE SFT, add C-exploration. Plausible paper/blog forming.

Direction set after the speed↔reliability result: power up the finding (don't conclude on n=32 / 6 matched pairs), keep SFT for future, pilot whether a different C delivery ("something in C") helps. Ian sees a plausible paper + blog post.

**Task set expanded 8 → 14** (`scripts/synth_tasks.py`): +6 new multi-site cascades, all verified to fire guiding pyrefly diagnostics + fail behaviourally — `method_rename_cascade` (2 sites), `dict_key_type_drift`, `ctor_param_added` (2), `renamed_return_key` (2, via TypedDict so pyrefly sees the renamed key), `optional_two_helpers` (2), `tuple_return_widened`. Diverse: rename, key-type, ctor-arity, TypedDict-key, Optional-chain, tuple-widening.

**Powered run launched** (`runs/agent/synth_power.json`, tracked, checkpointed to `.partial` per-task): A / C / **D-tuned** (debounce 24 + pause-align + announce) × 14 tasks × **6 seeds** (temp 0.7), max_new 2200 — ~250 rollouts, ~8h overnight. This firms the resolve-drop and the matched ~2×-efficiency claim with real power and a broader task base.

**C-exploration ready** (`scaffold/stream_agent.py` flag `c_eager`, runner `--c-eager`): C-eager = post-edit hook (deliver the diagnostic IMMEDIATELY after each edit, the production-agent norm) vs current C-lazy (batched at the model's next yield). Pilot to run AFTER the powered run (GPU is serial): does eager sync beat lazy sync / none? Probes whether C≈A was a delivery artifact too.

**SFT: explicitly deferred** to a future exploration (per Ian). Richer-signal (autocomplete/go-to-def) also parked as a later novel direction.

**Status:** powered run in flight; C-eager pilot queued. Story arc holding (consume-in-isolation → naive-live-hurts-reliability → debounce-recovers-matched-efficiency → C/SFT/richer-signal as levers). Task #44.

---

## 2026-06-01 — POWERED RESULT (14 tasks × 6 seeds, n=84/cond): live feedback SIGNIFICANTLY hurts fix-rate; the matched-efficiency win did NOT survive. RETRACTION + clean negative.

`runs/agent/synth_power.json` — A / C / D-tuned (debounce+pause+announce), temp 0.7, 6 seeds, 14 tasks.

**Resolve (n=84):** A 0.476 (40/84), **C 0.548 (46/84), D-tuned 0.345 (29/84)**. Two-proportion tests: **D<C p≈0.008 (significant)**; D<A p≈0.08 (trend); C vs A p≈0.35 (n.s.). D ≤ C on 13/14 tasks.

**Matched-pair efficiency (selection-bias-free) — RETRACTION:** with 20 matched pairs (vs the earlier 6), D-tuned vs C = **2.5 vs 2.5 round-trips (identical)**, 275 vs 318 tokens (marginal); D-tuned vs A (15 pairs) = 2.3 vs 2.2, 250 vs 244 (identical). **The earlier "1.7 vs 3.2 ~2× efficiency win" was a small-sample artifact (6 pairs, dominated by lookup_optional) and did NOT survive powering up. Retracted.** The "speed↔reliability trade-off" framing is withdrawn.

**THE CLEAN, WELL-POWERED FINDING:** zero-shot, in the agentic loop, LIVE in-stream LSP feedback (D) **significantly reduces** an untrained 7B-coder's fix-rate vs sync (C), with **NO compensating efficiency**. SYNC feedback ≈ NONE (C not sig. > A). I.e. **delivery FORM matters — and it's the OPPOSITE of the human-LSP intuition: mid-stream live delivery derails the model; the benefit (if any) needs sync delivery AND/OR training.** This is a clean, honest, counter-intuitive NEGATIVE result (still publishable; negative/surprising results are valuable). Calibration good (suite mean ~0.35-0.55, upper Goldilocks; a few new tasks near-ceiling: mutable/method_rename/dict_key).

**Mechanism (consistent across all runs):** the model CAN consume a clean single interleaved injection in isolation (R0 16/16, R0b +0.70, R0c toy D≈½C) but in the messy multi-turn agent loop, mid-stream splices — including diagnostics about its own transient broken edits — interrupt coherent editing → more derailing → lower resolve. Debounce/pause/announce reduce the noise but don't flip the sign.

**Forward (the negative SHARPENS the motivation):** (1) **C-eager pilot** now MORE interesting — if eager sync (post-edit hook) beats lazy-C/none, then sync-LSP DOES help and only LIVE hurts (a crisp "delivery-timing flips the sign" result). (2) **SFT** is now clearly the lever to test whether training lets live feedback help (was deferred; the powered negative re-motivates it as the actual contribution if it works). (3) Richer constructive signal (autocomplete/go-to-def) still parked. Task #44.

---

## 2026-06-01 — C-EAGER pilot: delivery form orders fix-rate MONOTONICALLY — eager-sync > lazy-sync > none ≫ live. The positive sub-result.

`runs/agent/synth_ceager.json` — C-eager (post-edit hook: deliver diagnostic immediately after each edit, forcing a yield), 14 tasks × 6 seeds, temp 0.7, vs the powered A/C-lazy/D-tuned.

**Resolve (n=84):** **C-eager 0.595 (50/84) > C-lazy 0.548 > A 0.476 > D-tuned 0.345.** MONOTONIC in "how synchronous / at-a-boundary the delivery is." Significance: C-eager vs D z≈3.25 (**p≈0.001**); C-eager vs A z≈1.55 (p≈0.12, trend); C-eager vs C-lazy z≈0.62 (n.s.). Robust spine = the ORDERING + the live-is-worst extreme; "eager-sync actively helps over none" is suggestive not conclusive (needs more seeds). C-eager wins notably on ctor_param (5 vs 3 vs 2), renamed_return_key (6 vs 4 vs 3), tuple_return (2 vs 1 vs 0); rarely worse.

**REFINED HEADLINE (replaces the bare negative):** the FORM of feedback delivery decisively orders coding-agent reliability — synchronous post-edit feedback (the production hook) is best, live mid-generation delivery (the human-LSP analogue) is worst, by a wide significant margin. A clean dose-response: the more you interrupt mid-stream, the worse. This is the central, controlled finding of the project; "C≈A" earlier was partly because lazy-C < eager-C — the production-norm eager hook does help (trend).

**Paper/blog shape (current, honest):** "Delivery timing is the decisive variable for in-the-loop LSP feedback: eager-sync helps, live hurts, monotonically — contrary to the human-LSP intuition." Solid workshop/strong-blog on its own. Open lever for a top-tier POSITIVE live result = SFT (does training let the model exploit live feedback?) — re-motivated by the powered negative; Ian had deferred it, revisiting.

---

## 2026-06-02 — SFT-data validity check (Ian-gated): STALE — `runs/i_sft_data/data.jsonl` mismatches the current harness. Need to regenerate via self-distillation from current resolved trajectories.

Validated the 2400-example SFT set before any training. Tokenizer OK (Qwen2.5-Coder-7B-Instruct, same model). But mismatched on task/markers/format/behavior:
- It's the OLD forward-injection task: `def get_limit()->int:` + `‹info›value‹/info›` (masked) + body; trains the model to USE an injected value. Decoded example confirms.
- Markers/format: uses `‹info›`; the deployment uses `‹diag›`, `<edit path lines=...>`, `<test/>`, `<test_result>`, multi-turn chat. NONE present in the data.
- Behavior taught = forward-value-consumption, which ALREADY works zero-shot (R0 16/16). Teaches nothing about the FAILING behavior (react to live `‹diag›` mid-agent-loop without derailing). → Training on it optimizes a non-problem; won't transfer.

**Regeneration plan (the right SFT data = self-distillation matched to deployment):** harvest RESOLVED agent trajectories from the current harness (the line-edit + `<test/>` + `‹diag›` format) — especially D-tuned-resolved runs, which are demonstrations of "consumed live feedback and fixed it." Format as {input_ids, labels} with labels = train on model-generated action tokens, MASK (-100) all spliced observations (user turns, `‹diag›`, `<test_result>`, file views, turn scaffolding). This is on-policy rejection-sampled self-distillation (matches the project's original SFT intent). `d_sft.py` already consumes {input_ids,labels} with loss on labels≠-100 — compatible. Implementation: add a parallel label-mask to `stream_agent` (1 for argmax-generated tokens, -100 for splice()'d tokens) + a harvester that keeps resolved trajectories. Caveat: D resolved only 29/84 → ~29-58 demos at 12 seeds; small for SFT — may need more seeds/tasks or include "made-progress" trajectories.

**Status:** confirmation run (seeds 6-11, A/C-lazy/D-tuned) in flight. SFT gated on regenerating data — awaiting Ian's go on the self-distillation harvester approach. Task #44.

---

## 2026-06-02 — SFT pipeline built + verified (label-mask harvester); confirmation run in flight. Ready to harvest→train→eval once GPU frees.

Per Ian's go-ahead: built the self-distillation pipeline (regenerating data, since the old set is stale per the prior entry).
- **Label-mask in `stream_agent`:** returns `sft_input_ids` / `sft_labels` — model-generated action tokens trained, all spliced observations (prompt, ‹diag›, <test_result>, file views, turn scaffolding) masked (-100). `n_train_tokens` reported.
- **Harvester `scripts/harvest_sft.py`:** runs the DEPLOYMENT config (D-tuned: debounce 24 + pause + announce) over the 14 tasks × N seeds, keeps RESOLVED trajectories (rejection sampling on outcome), writes {input_ids, labels} → `runs/i_sft_data_v2/data.jsonl`, checkpointed per task.
- **`d_sft.py` verified compatible** (reads {input_ids,labels} jsonl, drops extra cols, LoRA r64). **CAVEAT: its `--max-len` default 1024 would DROP our 2000-4000-tok agentic trajectories — must pass `--max-len ~4000` + small bs/accum when training.**
- Open quality note: harvested resolved trajectories include the model's intermediate WRONG edits (we keep the whole successful trajectory). If SFT learns fumbling, add a cleanliness filter (resolved AND few edits/tests/low rework). First pass keeps all resolved (≥min_train_tokens).

**Run plan (after confirmation run frees GPU):** harvest D-tuned ×16 seeds (~70-80 demos expected at ~35% resolve) → `d_sft.py … --max-len 4000` → re-run A/C/D-tuned matrix WITH `--adapter` → does D climb? Task #44.

---

## 2026-06-02 — Restart killed the confirmation run at 69/252; resumed from checkpoint. Ultracode on → launching adversarial analysis of the 6-seed result.

Session restart killed the seeds-6-11 confirmation run mid-flight. Per-task checkpoint saved 3 COMPLETE tasks (grid_field_rename, fmt_signature_drift, records_arity_drift, full A/C/D × 6) to `synth_power_s6.json.partial`. Resumed the remaining 11 tasks (`synth_power_s6_resume.json`, seeds 6-11); will merge partial(3)+resume(11) → 14 tasks @ seeds 6-11, then combine with `synth_power.json` (seeds 0-5) for the 12-seed analysis. Ultracode enabled (`/effort`) → running a parallel adversarial/statistical analysis of the COMPLETE 6-seed data (`synth_power.json` + `synth_ceager.json`) while the GPU computes the extra seeds: proper paired (McNemar) tests, Wilson CIs, jackknife robustness, confound hunt (bail/budget/announce-prompt/self-diag artifacts), control validation. Goal: harden or break the delivery-form ordering BEFORE committing days of GPU to SFT.

---

## 2026-06-02 — ADVERSARIAL AUDIT (5-agent workflow) of the 6-seed result: corrections + a CONFOUND I introduced + a better experiment than SFT. Pivot.

Ran a parallel audit (paired-stats / robustness / confound-adversary / controls → synthesis) on the complete 6-seed data. Verdict:

**Corrections to my earlier (wrong) stats:** I used unpaired 2-proportion z; the right test is PAIRED McNemar (same task,seed across conds). Exact two-sided McNemar:
- D-tuned vs C-eager: **p=0.00019** (D worse) — significant, stronger than I'd claimed.
- D-tuned vs C-lazy: **p=0.0060** (D worse) — significant.
- **D-tuned vs A (none): p=0.108 — NOT significant.** ← key correction. Honest claim = "live is worse than BATCHED delivery," NOT "worse than nothing."
- C-eager vs A p=0.076 (trend); C-eager vs C-lazy p=0.557 (n.s.). So the "monotonic ordering" is point-estimates; only D<both-batched is statistically real.

**Robustness:** D deficit holds in 14/14 jackknife; broad-based (D tied-worst on 12/14); survives dropping the degenerate/control tasks (C-eager lead widens). Heaviest on LOUD and SINGLE-site tasks.

**Mechanism CONFIRMED (real, not a bug):** 78.3% (191/244) of D's delivered diagnostics are self-inflicted (about the model's OWN broken mid-edit state, e.g. parse errors); **92% (23/25) of the A-solves-but-D-fails regressions had D receiving a self-inflicted diagnostic.** Live delivery makes the agent chase its own transient mess.

**Confounds: bail RULED OUT** (D bails LESS: 0.107 vs C-lazy 0.167). **Budget RULED OUT for D** (only 7.3% of D failures near cap; ironically C-EAGER is the budget-bound one — 50% of its failures at cap → more budget would help the WINNER, not D). **Efficiency retraction CONFIRMED** (D vs C-lazy 20 matched pairs: 2.5 vs 2.5 round-trips, ratio 1.00, p=1.0).

**UNRESOLVED CONFOUND (I introduced it):** D-tuned ALONE carries an extra `announce_lsp` system sentence; A/C/C-eager don't, and there's no announce-control arm → can't separate prompt-effect from live-delivery-effect. MUST run announce-off D.

**PIVOT (audit's ranked follow-ups; SFT DEFERRED as premature):**
1. **D-plain** = D-tuned WITHOUT announce (kills the confound; the clean canonical live arm).
2. **D-gate** = syntax-gate variant: only deliver a live diagnostic when the file currently PARSES (`ast.parse`), suppressing the 78% self-inflicted syntax squiggles. Directly tests whether D's harm is FIXABLE (audit: "arguably more decisive than SFT"). Implemented `syntax_gate` flag in `stream_agent` + `--syntax-gate`.
3. Both at seeds 0-5 (pair with existing A/C-lazy/C-eager). SFT only after we know if live-harm is a fixable delivery defect vs intrinsic.

Killed the (confounded) announce-D seeds-6-11 resume — low marginal value vs the clean arms. Audit memo full text: `tasks/wss8eijea.output`. Task #44.

---

## 2026-06-03 — PIVOTAL CORRECTION: "live hurts" was DELIVERY HYGIENE, not intrinsic. Announce-confound + self-inflicted squiggles explain it; gated clean-live = PARITY. Eager-sync best.

Ran the audit's two clean arms (announce-OFF), seeds 0-5, paired with existing A/C-lazy/C-eager/D-tuned. `runs/agent/synth_dplain.json` (D, no announce), `runs/agent/synth_dgate.json` (D, no announce + syntax-gate).

**Resolve (n=84, Wilson CI):** C-eager 0.595 [.49,.69] > C-lazy 0.548 > **D-gate 0.500 [.40,.60]** > A 0.476 > **D-plain 0.452** > **D-tuned 0.345 [.25,.45]**.

**The deficit decomposes into two FIXABLE causes:**
1. **Announce-prompt confound (I introduced):** removing it, D-tuned 0.345 → D-plain 0.452 (+11pp). (D-tuned vs D-plain McNemar p=0.16 alone.)
2. **Self-inflicted mid-edit squiggles:** add the syntax-gate (deliver only when file parses), D-plain 0.452 → D-gate 0.500 (+5pp). The gate cut diagnostic deliveries 240→**71 (-70%)** — confirming most were self-inflicted — and resolve ROSE.

**Significance (paired McNemar exact):**
- **D-tuned (naive) vs D-gate (cleaned): p=0.041 SIG** → delivery hygiene significantly matters (the headline).
- **D-gate vs A: p=0.79** → cleaned live = PARITY with no-feedback. The dramatic "live hurts" is GONE once hygiene is fixed.
- D-gate vs C-lazy p=0.56 (parity); C-eager vs D-gate p=0.15 (eager numerically best, n.s. at this n); C-eager vs A p=0.076 (trend).

**CORRECTED HEADLINE (replaces "live hurts"):** *Delivery design dominates in-the-loop LSP feedback. A naive live implementation (mid-edit interruption with self-inflicted squiggles + a pushy prompt) significantly underperforms; ~70% of its diagnostics are noise about the model's own broken state. Gate them out and live delivery matches synchronous and no-feedback. Eager post-edit sync is the simplest and numerically best.* A clean, honest, USEFUL engineering recipe — and the audit's "harm is fixable, not intrinsic" hypothesis is CONFIRMED.

**SFT recalculus:** harm was fixable → clean-live reaches PARITY, not a win. SFT's job would now be to push live ABOVE sync (harder bar, uncertain payoff) — vs the simple practical takeaway "use eager post-edit sync." Decision deferred to after the n=168 power-up. Caveat: most pairwise diffs n.s. at n=84 (only the hygiene effect is sig); running seeds 6-11 (A/C-lazy/C-eager/D-plain/D-gate) for power. Task #45.

---

## 2026-06-03 — FINAL n=168 zero-shot result: the PARITY BAND locks. Write-up pushed to private GH (ianbarber/streams). Follow-ups launched (rich-signal; SFT queued).

**Final n=168 (14 tasks × 12 seeds):** C-lazy 0.530, C-eager 0.524, D-gate 0.482, A 0.482, D-plain 0.458 — **ALL pairwise McNemar p>0.26** (D-gate vs A exactly balanced 12/12, p=1.0; C-eager vs C-lazy p=1.0 — the eager "lead" at 6 seeds fully evaporated). D-naive (announce, ungated) remains the only condition outside the band: 0.345 (n=84), significantly below batched (p=.0002/.006), hygiene effect vs D-gate p=.041.

**LOCKED ZERO-SHOT CONCLUSION:** for an untrained 7B coding agent with a test loop, properly-delivered type-checker feedback of ANY timing ≈ no feedback (0.46–0.53 band); delivery can only subtract (naive-live −14pp), not add. "How you deliver" is the only lever that moved, and only downward.

**Ops:** WRITEUP.md drafted; private repo github.com/ianbarber/streams created + pushed (code, log, results, write-up; 3 commits). Context-explosion harness hole found mid-run (degenerate file bloat × per-turn file-view re-feed → 58k tokens > 32k ctx) → hard 24k context cap + 250-line file-view truncation; 4 overflow bails in final data, cleanly handled; fix pushed.

**Follow-ups (R3, the "beat the band" question):** (1) RICH-SIGNAL running: `rich_signal` appends go-to-def/hover-style context (signatures/fields of backticked symbols) to diagnostics; arms D-gate+rich and C-eager+rich at seeds 0-5. (2) SFT queued behind it: harvest train-split (even-index tasks) at seeds 100-111 (disjoint from eval) in D-gate deployment config → LoRA (--max-len 4000) → eval on HELD-OUT odd tasks (D-gate ± adapter; A ± adapter control). SFT data validity gate done earlier (old set stale; harvester emits deployment-format {input_ids,labels} with observation masking). Task #46.

---

## 2026-06-03 — RICH-SIGNAL result: mild nudge, band holds. SFT chain launched (last lever).

Rich-signal (go-to-def/hover context appended to diagnostics), seeds 0-5 vs plain counterparts, paired:
- **D-gate+rich 0.524 vs D-gate 0.500** — only 2 discordant pairs, BOTH favouring rich (b=2,c=0, p=0.5). Direction positive, magnitude tiny. Task texture: rename/key-type tasks hit ceiling with rich (return_container 6/6, dict_key 6/6) — context helps exactly where "what does this symbol look like now" is the question.
- **C-eager+rich 0.607 vs C-eager 0.595** — flat (b=3,c=2, p=1.0).
→ **Constructive content does NOT break the parity band untrained.** Neither timing nor content is the binding constraint; the test loop saturates the untrained model.

**SFT chain launched** (the last lever): harvest train-split (7 even-indexed tasks) × seeds 100-111 (disjoint from eval seeds) in the D-gate deployment config → expect ~40 resolved demos → `d_sft.py` LoRA (r64, --max-len 4000, bs1×accum8, 2 epochs) → eval next: D-gate ± adapter and A ± adapter on all 14 tasks @ seeds 0-5; the claim rides on the HELD-OUT (odd) tasks. Task #46.

---

## 2026-06-03 — SFT round 1: harvest thin (27 demos / ~7 optimizer steps) → topping up before the expensive eval.

First harvest: 27/84 resolved (32%) on the train split @ seeds 100-111; short clean solves (mean 364 train tok). LoRA trained (loss 0.16) but ~7 steps is too thin to judge — NOT spending the 6h eval on it. Chained instead: +12 seeds harvest (112-123, → data2.jsonl) → retrain on combined (~55-60 demos, runs/adapters/dgate_sft_v2) → eval D-gate+adapter (84) → eval A+adapter (84, control). Claim rides on the 7 HELD-OUT odd-index tasks; A±adapter separates "learned to use feedback" from "got better at bugs generally". Note: 2 harvested demos are from the no-signal control task (config) — generic-fix demos, acceptable. Task #46.

---

## 2026-06-04 — SFT RESULT: zero held-out transfer; train-task gains are memorization (A-control proves it). The circularity finding. Program COMPLETE.

Combined harvest 62 demos (27+35, 42% resolve on train split, seeds 100-123) → LoRA v2 (loss 0.17) → eval all 14 tasks @ seeds 0-5, D-gate ± adapter and A ± adapter.

**Held-out (odd) tasks — the claim:** D-gate+SFT 22/42 = D-gate base 22/42 EXACTLY (b=7,c=7,p=1.0). A+SFT 21/42 vs A base 26/42 (slightly worse, n.s.). **Zero feedback-use transfer.**
**Train (even) tasks — memorization check:** D-gate+SFT 0.667 vs 0.476 (p=.077); **A+SFT 0.595 vs 0.333 (p=.007)** — the NO-FEEDBACK arm gained MORE → the LoRA learned the training tasks' fixes, not feedback-use. Control did its job.

**THE CIRCULARITY FINDING (mechanism):** harvested demos avg ~364 trained tokens — short clean solves where the diag channel was barely exercised. Because feedback adds nothing zero-shot (the parity band), successful zero-shot trajectories contain almost no feedback-use to distill → **rejection-sampled self-distillation selects for easy solves, not channel exploitation.** Bootstrapping feedback-use needs demos where the diagnostic is load-bearing by construction (revision-style supervision à la R0b) or RL on feedback-dependent rewards. Future work.

**PROGRAM COMPLETE — final scorecard:** timing ✗ (parity band, n=168, all p>.26) · hygiene ✓ but downward-only (naive-live −14pp, p≈.04 vs gated; 78% self-inflicted mechanism) · content ~flat (rich-signal nudge, band holds) · training ✗ at this scale (memorization, circularity). WRITEUP.md finalized; repo current. Task #46 done.

---

## 2026-06-04 — Publication pass: citation audit (1 misattribution fixed) + repo cleanup (208→~70 files); stats reproduction script added.

**Citation audit (web-verified, every identifier fetched):** all 25 bibliography IDs resolve to the claimed works. One BLOCKING fix: `claudecodelsp2025` was misattributed to Anthropic — it is a third-party community-plugin blog post (Robert Allen / zircote); re-attributed and the §2 sentence reworded. One overstretch: the SWE-Bench-Illusion cite was wrapped in a "realism vs signal-isolation trade-off" claim the paper doesn't make (it's a contamination/memorization study) — reworded to cite it correctly. Author-field fixes (Su et al. full names, Reflexion +Berman, Illusion authors, SWE-rebench paper ID). Four verified missed-work citations added (RepoNavigator 2512.20957, AsyncVoice 2510.16156, context-length 2510.05381, ACON 2510.00615).

**Repo audit + cleanup (paper-reader framing):** deleted all era-1 multi-stream code (~70 files: g1/g3–g6 scripts, lsp delivery layers, training/, configs/, eval/, era-1 tests) and the D-pivot probe scripts; purged runs/ partials/smokes/pilots; CONSOLIDATED D-plain seeds 6-11 (checkpoint 4 tasks + resume 10 tasks → `synth_dplain_s6.json`, 84 rows verified) before purging — the audit caught that the partial was load-bearing. experiment_plan.md + WRITEUP.md → docs/history/ with superseded headers; runs/i_eval → runs/isolation (paper-section names); swe results → runs/rebench/. **`scripts/analysis/stats.py`** (was root audit.py) now reproduces EVERY paper statistic from the committed files — verified exact — closing the audit's finding that the n=168 headline table previously had no committed reproduction path. All READMEs rewritten for paper-readers incl. the condition→flag→file table and the debounce-defaults reproducibility trap. PAPER.md Appendix B updated accordingly. Deleted material remains in git history.

---

## 2026-06-06 — Ian's review: de-process the paper (done); SINGLE-FILE REDUNDANCY CONFOUND identified → multi-file suite built; typing-signal run launched.

**Ian's review points + decisions:** (1) the mistakes/process narrative in the paper detracts — cleaned: announce arm now presented as a designed ablation, efficiency-retraction section removed (one line in analysis-robustness), audit section → compact "analysis robustness," SFT shrunk to a circularity paragraph pending a proper redo; full history stays in log.md. (2) The SFT-as-run mostly proved its own data inadequate — agreed; redo deferred until a setting with headroom exists. (3) **KEY INSIGHT (Ian): all tasks were single-file, so every type definition the checker knows is already in the model's context — the channel was informationally REDUNDANT. This one confound plausibly explains the parity band, the flat rich-signal result, AND the SFT circularity.** The LSP's human value is cross-file knowledge. (4) The residual sync-vs-none signal (+4-5pp, one-sided p=.13/.19; D-gate vs A exactly null) would need ~740 paired units to power on single-file — instead, test where the effect should be larger.

**Built (verified ALL OK):** `scaffold/mock_env.py::MultiFileEnv` (workspace, cross-file pyrefly, fresh-subprocess tests) + `scripts/synth_tasks_mf.py` — 10 multi-file tasks where the misused type definitions live in UNSHOWN files; informativeness gradient pre-registered: **plain** (diag text itself carries the remote fact: missing-argument names the param, bad-unpacking shows the tuple shape — 5 tasks), **rich** (diag names the problem, only the remote definition names the fix: renamed field/method/key — 4 tasks), **control** (no type signal — 1). Each task gold-fix-proven solvable; behavioral-fail + cross-file-diag verified. `scripts/synth_mf.py` runner (prompt = target file only + names of other files + test-as-spec; `<read>` available at turn cost).

**Run 1 launched (typing signal first, per Ian):** A, C-eager, C-eager-rich × 10 × 6 seeds (~180 rollouts). Predictions: plain-group C-eager>A; rich-group C-eager-rich>C-eager; control flat; A's mean_reads ↑ (compensating by reading). Live arms (D-gate±rich) follow once channel value is established. Task #48.

---

## 2026-06-07 — MULTI-FILE RESULT: removing the redundancy confound makes the channel matter. C-eager > A by +18pp, p=0.017 (vs single-file parity).

(HF hub metadata stall hung the first launch 12h with 0 weights loaded; relaunched with HF_HUB_OFFLINE=1 — now standard for all launches.)

**Run 1 (A / C-eager / C-eager+rich × 10 tasks × 6 seeds, n=60 each):**
- A 20/60 = 0.333 | C-eager 31/60 = 0.517 | C-eager+rich 33/60 = 0.550.
- **C-eager > A: paired one-sided p=0.017 (b=17,c=6); C+rich > A p=0.004 (b=17,c=4).** SIGNIFICANT — vs single-file where best delivery vs none was +4.8pp p=0.13. Ian's redundancy hypothesis CONFIRMED: when the checker's knowledge is NOT already in context, the channel adds real value.
- Gradient (pre-registered): plain A 10/30→C 16/30 (p1=0.073, channel substitutes for reading the unseen file); rich C 12/24→+rich 14/24 (p1=0.25, right direction, underpowered — 2 discordant); control noisy at n=6.
- **Caveat to characterize:** A mean_reads ≈ 0.90 ≈ feedback arms — the no-feedback 7B mostly does NOT read the unseen files to compensate (first rollout: 0 reads/0 edits/budget-out). So the gap = "feedback hands you the fact" + "weak agent doesn't think to read." Both real; want the split.

**Run 2 launched:** D-gate, D-gate+rich (seeds 0-5) + A, C-eager power-up (seeds 6-11) → n=120 on core arms. Answers the original question — does LIVE (D-gate) beat SYNC (C-eager) — now that there is headroom. Task #48.

---

## 2026-06-07 — Reframe (Ian): paper centers on the DELIVERY MECHANISM; redundancy is a boundary condition, not the headline. Preread ablation built.

Ian: don't make the single-file-redundancy insight the big thing (obvious to the field). Instead — focus on the DELIVERY MECHANISM (timing/hygiene/content), and use single-file as a clean demonstration of WHEN feedback does not help, tied to the long-context-window question: "if I can stuff the whole codebase into context, do I need the channel?". Direct test: PREREAD ablation — same multi-file tasks but ALL files placed in the prompt up front, so the checker's knowledge is already in context. Prediction (2x2 context{partial,full} × feedback{none,sync}): feedback helps only under PARTIAL context; preread-A ≈ preread-C-eager (channel redundant) and both > partial-A.

**Built + verified:** `scripts/synth_mf.py --preread` (shows every workspace file's full contents in the prompt). Prompt construction confirmed (remote signature now in context). To run after the live arms (GPU serial): preread-A + preread-C-eager × 10 × 6.

**New paper arc (to restructure once live + preread land):** (1) how to deliver in-loop checker feedback, and when it matters at all; (2) BOUNDARY — single-file + preread show all delivery ≈ none when info is already in context (the long-context-window angle); (3) MECHANISM — in the multi-file/partial-context regime that HAS headroom, does timing/hygiene/content matter: the hygiene result (naive live hurts, fixable — 78% self-inflicted) + live-vs-sync (running) + rich-vs-plain. Hygiene/self-inflicted-squiggle mechanism stays central; single-file parity becomes the boundary condition rather than the headline negative. Task #48.

---

## 2026-06-07 — CORRECTION: the multi-file n=60 channel effect was SEED LUCK. At n=120 it is NOT significant (+5.8pp, p1=0.17) — indistinguishable from the single-file effect. Redundancy hypothesis NOT supported at power.

I reported the n=60 multi-file result ("C-eager > A +18pp, p=0.017; redundancy hypothesis confirmed") too eagerly, on data I had myself flagged as needing power. The power-up reverses it:
- A: seeds 0-5 = 20/60, seeds 6-11 = **28/60** -> n=120 = 48/120 = 0.400.
- C-eager: seeds 0-5 = 31/60, seeds 6-11 = **24/60** -> n=120 = 55/120 = 0.458.
- **Full n=120 paired C-eager vs A: b=24 c=17, one-sided p=0.174, two-sided p=0.349 — NOT significant.** The seeds-6-11 block went the OTHER way. Textbook Bjarnason seed-variance (and a live demonstration for the paper's own methodology point).
- By group n=120: plain A 22/60 vs C-eager 28/60 (p1=0.132, directional, the mechanistically-expected locus); rich 23/48 vs 22/48 (flat); control noisy.

**Implication:** the multi-file channel effect (+5.8pp, p1=0.17) ≈ the single-file effect (+4.8pp, p1=0.13). Removing the redundancy confound did NOT measurably increase the channel's value. The redundancy hypothesis (Ian's + mine) is NOT supported at power. This is a STRONGER negative: even with engineered non-redundant cross-file information, the type-checker channel does not significantly help a 7B with a test loop. (Live-vs-sync still pending D-gate n=120; D-gate/C-eager n=60 numbers are likely seed-inflated too — do not trust until powered.)

**In flight:** D-gate seeds 6-11 (-> n=120) + preread 2x2 (preread-A, preread-C-eager). Will give: live-vs-sync at power, and the context-saturation ablation. HOLD all conclusions until these land; report only n=120+ paired tests henceforth. Task #48.

---

## 2026-06-08 — Multi-file POWERED (n=120): same parity band as single-file. Live=none exactly. Redundancy hypothesis dead; the negative is now REPLICATED across two task regimes.

Partial-context multi-file, n=120: A 0.400, D-gate(live) 0.408, C-eager(sync) 0.458 — overlapping CIs. Paired: **D-gate vs A p1=0.500 (EXACTLY null — live does NOT beat none at power; the n=60 p=0.020 was seed luck)**; C-eager vs D-gate p1=0.189; C-eager vs A p1=0.174. All n.s. Same picture as single-file: channel doesn't significantly help; live <= sync ~ none.

**Preread 2x2 NOT yet trustworthy:** preread arms were only seeds 0-5 (the seeds we now KNOW were lucky for C-eager); n=60 preread "channel helps p1=0.032" is almost certainly the same artifact (and points backwards from the redundancy prediction: full-context+sync 0.533 > partial+sync 0.458). Launched preread seeds 6-11 to complete it honestly before reading anything in.

**Meta:** the type-checker channel's non-significant ~+5pp benefit now REPLICATES across single-file (n=168, +4.8pp p=0.13) and multi-file (n=120, +5.8pp p=0.17). Engineering non-redundant cross-file info did not help -> closes the "info was redundant / tasks too easy" escape hatch -> a STRONGER, twice-replicated negative. Only significant delivery effect anywhere remains the naive-live HARM (single-file hygiene/self-inflicted-squiggle, n=84). Decision pending preread: paper framing = "no significant benefit bounded ~5pp across two regimes; bad delivery hurts; delivery timing among proper deliveries within noise" rather than chasing a ~5pp effect (would need ~700+ paired units). Task #48.

---

## 2026-06-08 — POSITIVE, SEED-ROBUST result: feedback + in-context code are COMPLEMENTS. Full-context+sync significantly beats all (p=0.001). The long-context question INVERTS.

The preread 2x2 (n=120/cell) — and unlike the partial-context mirage, this one PASSES the seed-block check (both halves agree):

              partial ctx    full ctx (preread)
  none        0.400          0.383
  +sync       0.458          0.575   <- highest cell

- **Channel effect significant ONLY under full context:** preread sync-vs-none p1=0.001 (b=37,c=14); partial p1=0.174 (n.s.).
- **Full context ALONE does nothing:** preread-A 0.383 ~ partial-A 0.400 (p1=0.667) — stuffing all files in the prompt does not help the no-feedback agent.
- It is the COMBINATION. By group: plain p1=0.017, rich p1=0.047 (both type-signal groups independently significant), control noisy.
- **Seed-block check (the decisive test the partial result failed): PASSES.** seeds 0-5 PA22/PC32, seeds 6-11 PA24/PC37 — both halves favor PC strongly (partial result had DISAGREEING halves; this one agrees).
- **Mechanism:** partial-A mean_reads=0.94, 32% zero-read; 30/72 partial-A failures NEVER read any file. The agent does not reliably gather context itself. Full context + diagnostic localizes the bug within the now-complete context.

**INTERPRETATION:** feedback and in-context code are COMPLEMENTS not substitutes. The diagnostic's value is as a LOCALIZATION/ATTENTION signal, only actionable when the relevant code is present (which the agent won't assemble alone). **This inverts Ian's long-context question: with everything in the window, the LSP channel becomes MORE valuable, not less** — role shifts from information-delivery to attention-direction (cf. context-length-hurts-with-perfect-retrieval, du2025contextlength). The project's first robust POSITIVE, and counterintuitive.

**Discipline note:** trusting this (vs the retracted n=60) because: n=120, BOTH seed-blocks agree, p=0.001 (not marginal), significant in 2 groups independently, null control, corroborating reads mechanism — every check the mirage failed. STILL launched confirmation (+6 seeds -> n=180 on preread-A/C) + preread-D-gate (live under full context = the delivery-mechanism question in the regime where feedback works) before building the paper on it. Task #48.

---

## 2026-06-08 — CONFIRMED at n=180 (p=0.0005, all 3 seed-blocks agree). + live-under-full-context: SYNC captures the benefit, LIVE does not. The delivery-mechanism finding lands.

Preread channel effect CONFIRMED, stronger with power: n=180 preread-A 74/180=0.411, preread-C-eager 104/180=0.578; PC>PA b=51 c=21 one-sided p=0.0003, two-sided p=0.0005. Seed-block trust test PASSES all 3 blocks (PA 22/24/28, PC 32/37/35 — every block favors PC). Robust positive.

**Live under full context (preread-D-gate, n=60 s0-5, confirming to n=120):** preread none 0.367 / live(D-gate) 0.400 / sync(C-eager) 0.533. **Live ~ none (p1=0.377); sync > live (p1=0.092).** Synchronous post-edit delivery captures the feedback+context benefit; LIVE mid-stream delivery does NOT — same hygiene/self-inflicted/distraction mechanism (mid-stream splice can't be cleanly connected to context). Consistent with sync>=live throughout the project. n=120 CONFIRMED: under full context sync(0.578) > live(0.400) p1=0.0032 (seed-robust, live 24/60 both blocks); live~none p1=0.42. Delivery-mechanism finding SIGNIFICANT: sync captures the feedback+context benefit, live cannot.

**THESIS CRYSTALLIZING (paper restructure target):** (1) WHEN does checker feedback help an agent? Only when the relevant code is in context — feedback+context are COMPLEMENTS; the diagnostic is an attention/localization signal not an info carrier (preread p=0.0005, robust; full-context-alone null; 42% of no-feedback failures never read the needed file). (2) HOW to deliver it? SYNCHRONOUSLY — live mid-stream fails to capture the benefit (live~none under full context) and naive live actively hurts (single-file hygiene, 78% self-inflicted, p<0.005). (3) The LONG-CONTEXT INVERSION: more context -> channel MORE valuable (attention-direction), not less. Counterintuitive, novel, well-powered, timely. Came from Ian's single-file observation. Task #48.

---

## 2026-06-08 — SKEPTICAL REVIEW (verified vs raw data): effect REAL, but 2 of 3 interpretations OVERREACH. Three confounds, one is project-wide.

Reviewer reproduced every headline number exactly. The 0.41->0.58 preread sync-vs-none effect is real (exact p=0.00054, survives Bonferroni x5 and jackknife<0.021). But:

**[ALIVE, most dangerous] Claim 2 (sync>live) is a delivery-FORMAT confound, not timing.** sync(c_eager) delivers diagnostics as a clean USER TURN + refreshed file view; live(D) raw-splices ‹diag› markers mid-assistant-stream with no turn/file-view. **22% of live rollouts that got a diagnostic had the ‹diag› markers LEAK into the model's edits (it wrote ‹diag› as code).** Decisive: at matched 1-diagnostic, sync +0.50 vs live +0.04 (kills the delivery-COUNT confound — not a count effect — but the residual is plausibly FORMAT not timing). **This is a marker-leakage BUG in the live delivery path that confounds EVERY live-vs-sync comparison in the project, incl. the single-file "naive-live hurts" finding in PAPER.md.** We have never cleanly tested live timing.

**[ALIVE] Claim 3 interpretation wrong: it's largely an EDIT-ERROR/ANTI-THRASHING loop, not type-localization.** ~80% (6400/7874) of delivered sync diagnostics are SELF-INFLICTED syntax/scope errors from the model's brittle line-edits (unknown-name, invalid-syntax, parse-error); only ~700 are the cross-file TYPE errors the tasks are about. **The CONTROL task (no type signal at all) STILL benefits from sync: 0.222->0.611, p=0.039.** preread-none failures don't "never attempt" (5/106 zero-edit) — they THRASH (mean 10.3 edits, budget-out). Type-localization IS real (rich>plain p=0.016, weak on plain p=0.16 — refutes recency) but is a MINORITY of payload, entangled with generic edit feedback.

**[WEAKENED] Claim 1 ("ONLY under full context"): the INTERACTION is not significant (z=1.32, p~0.19).** Classic compare-significance-levels error. Full +0.167 (p<0.001) vs partial +0.058 (n.s.) — partial is underpowered, not demonstrably zero. Drop "ONLY"; restate as "substantially larger under full context."

**KILLED:** delivery-count (B), recency (B — rich>plain refutes it), model-ignores-context (D, preread reads drop to 0.12), multiple-comparisons (survives Bonferroni), single-task-driven (G, jackknife robust; but disclose 2/10 tasks REVERSE — sync sometimes hurts).

**Robust defensible core (reviewer):** "a TURN-STRUCTURED type-checker channel raises a weak 7B's multi-file resolve 0.41->0.58 (p<0.001, robust), benefit larger with code in context; the channel acts substantially as an early edit-error/anti-thrashing signal, with a real but minority type-localization component." Timing (sync vs live) and the attention-vs-information mechanism need 2 more arms before claimable.

**Decisive follow-ups (reviewer):** (1) CLEAN-LIVE arm — deliver D's debounced diagnostic via a clean turn + file view (fixes marker leakage; isolates timing from format). (2) MECHANISM — type-errors-only vs syntax-errors-only delivery (isolates anti-thrashing from type-localization). + bounds on model-weakness via rewrite edit-mode / stronger model. Task #48.

---

## 2026-06-08 — Built the two decisive arms (clean-live; type/syntax filter); smoke-verified; launched n=120 preread run.

Per Ian (yes on both): implemented `clean_delivery` (D delivers debounced diagnostic as a clean USER turn + file view, NOT a raw ‹diag› splice — isolates delivery FORMAT from timing, fixes the 22% marker-leakage bug) and `diag_filter` type|syntax (deliver only cross-file TYPE errors vs only self-inflicted SYNTAX/scope — isolates type-localization from anti-thrashing). Both in stream_agent.py + synth_mf.py. Smoke-verified: clean-live fired 8 diags as clean turns, 0 marker leak; type-filter delivered only type codes.

**Decisive run launched (preread, n=120 each, ~18h):**
1. clean-live (D, debounce+pause+syntax-gate+clean-delivery) — vs sync 0.578 (timing test: if clean-live ~ sync, the sync>live gap was FORMAT/leakage; if clean-live ~ raw-live 0.400, it's TIMING).
2. type-only sync (C-eager, diag-filter=type) — mechanism.
3. syntax-only sync (C-eager, diag-filter=syntax) — mechanism (if syntax-only ~ full-sync 0.578 and type-only ~ none 0.411, anti-thrashing dominates; if reversed, type-localization dominates).

Then: reframe paper around what survives + these results; REPEAT the skeptic review. NOTE: marker-leakage bug also confounds single-file PAPER.md live claims — reframe must caveat or re-run those too. (Intermittent HF-hub hang recurred on one launch despite HF_HUB_OFFLINE=1; direct relaunch fixed — set env inline per command.) Task #48.

---

## 2026-06-09 — TASK-SET deep-dive (practitioner/HN lens): tasks clean+solvable but NOT yet defensible; the 2 reversers are explained TASK ARTIFACTS; rework list.

Verdict: every task fails behaviourally + is gold-solvable (good artifact), but as EVIDENCE the suite has structural problems:
- **Single-file: type def is in the visible file -> channel is 100% REDUNDANT** -> can only test localization/batching, never signal-value. Confirms the reframe; single-file cannot carry "type signal helps".
- **~Half the diagnostics are TRANSCRIPTION** (fmt_signature_drift, ctor_param_added, mf_signature_drift, mf_ctor_param: "Missing argument `unit`" -> add unit="") -> inflates apparent channel value. Label or harden.
- **The 2 reversers are TASK ARTIFACTS, fully explained (NOT a feedback finding):**
  - mf_optional_return: test UNDER-CONSTRAINS — line_price's None branch is never exercised, so you can pass GREEN with 2 pyrefly errors still live; feedback then pushes the model PAST a passing test into churning an untested path. + a misleading 2nd diag (bad-return on the annotation).
  - mf_typeddict_key: correct key `total` is nowhere visible and the PLAIN diag never names it -> model must guess; guessing `count` -> pyrefly goes SILENT (valid key) but test still fails -> FALSE ALL-CLEAR. A rich task graded with a plain channel.
  - Root cause both: the diagnostic and the test DISAGREE about "done", and feedback makes the model trust the diagnostic. Fix: every test must fully constrain every edited site.
- **Control not clean:** mf_control_truthiness changes 4 things at once (remote file is a bare constant not a violated type; single-site; spec in a test comment). config_truthiness_distractor gold_note FALSELY claims pyrefly flags it (fires 0) — a mislabeled 2nd control.
- **records_arity_drift resolves 0% everywhere** despite being fully-specified+easy -> indicts the AGENT/harness, contaminates difficulty calibration.
- **Contrived (strawman):** dict_key_type_drift/mf_key_type (`idx[str(k)]`), mutable_default_none (not the classic `tags=[]` bug), return_container_ripple (`max(scores, key=lambda kv: scores[kv])`).
- **Bug-class imbalance:** heavily weighted to add-None-guard + rename-symbol — exactly the classes where a type checker is MOST redundant with reading. **Only 2 genuinely channel-necessary "rich" tasks: mf_field_rename, mf_method_rename.**

**Rework list (no agent change):** cut/rewrite the 3 strawmen; harden/label the 4 transcription tasks; FIX the 2 reversers (constrain line_price w/ unknown sku + kill the bad-return diag; grade typeddict with rich channel or add a test that fails the `count` guess); harden the control (real remote type, multi-site, spec in assertions not comments); add DISCRIMINATING bug-classes where the checker shines and the fix isn't readable: protocol/duck-type mismatch, wrong generic param across files, enum/Literal misuse, None-vs-empty at an API boundary; move all spec intent out of test comments.

**Implication:** the multi-file result is the right vehicle but needs the cleaned task set before it is publishable; the genuine signal-value evidence currently rests on just 2 tasks. Current decisive arms (clean-live/type/syntax, running) still worth finishing — claim 2 (delivery format) is task-independent. Task #48.

---

## 2026-06-09 — Killed intermediate arms (Ian); interim claim-2 read; built task set v2 (15 tasks clearing hardened bar); sending for review.

**Killed the v1-task decisive arms** (clean-live partial only, type/syntax never ran) per Ian: interpret what we have, plan a fresh run on high-confidence tasks. Stopped the babysitting /loop (sentinel runs/agent/.monitor_stop).

**Interim claim-2 (clean-live partial, 6 tasks, n=72 matched):** none 0.375, raw-live 0.417, **clean-live 0.389**, sync 0.556. clean-live ~ raw-live (p=0.77) ~ none (p=1.0); **sync > clean-live p=0.043.** => the marker-leakage/FORMAT fix did NOT close the sync>live gap. The skeptic's "it's just format" hypothesis is REFUTED on this data — live fails to capture the benefit even delivered cleanly. CAVEAT: gated/debounced live still delivers FEWER diagnostics than sync (count/gating confound remains) -> fresh run needs an UNGATED-clean-live arm to isolate pure timing.

**Task set v2 (`scripts/synth_tasks_mf2.py`, 15 tasks, ALL pass the hardened verifier):** rebuilt after both reviews. Design bar enforced: (R1) multi-file non-redundant; (R2) test<=>diag<=>correct — gold is BOTH test-green AND pyrefly-clean, test fully constrains every site (kills v1 reversers); (R3) no transcription (verifier flags fix-token-in-diagnostic); (R4) no spec in comments; (R5) realistic (no strawmen). 8 RICH (field/method rename, generic-widened int->list, enum member rename, callback-sig 1->2 args, obj-vs-tuple return, field-type int->str, arity drift) where only the remote def names the fix; 4 PLAIN; 3 CONTROL (truthiness/off-by-one/wrong-op logic bugs, 0 type signal, matched multi-file structure). Verifier also checks false-all-clear.

**Plan:** (1) send v2 for a fresh practitioner/skeptic review + expansion toward ~20-24 for power; (2) revise; (3) ONE clean powered run on v2 across the full channel matrix — context {partial, preread} x delivery {none, sync-eager, sync-lazy, live-raw, live-clean, live-clean-UNGATED} x content {plain via group, rich via group} + mechanism {type-only, syntax-only} — the real final data; intermediate (v1) results superseded. Task #48.

---

## 2026-06-09 — v2 task-set review: R2 SOLID, but "rich" over-claimed (~3-4 of 8 genuine); relabel/cut/extend + ~15 new rich proposals. + pyrefly-daemon concurrency deadlock.

Review of `scripts/synth_tasks_mf2.py` (15 tasks, all pass verifier). Verdict: R2 (test<=>diag<=>correct) is the strongest part — re-attacked 8 tasks with type-valid-but-wrong edits, NONE produced false-all-clear or green-with-errors (v1 reverser class does not reproduce). BUT the verifier only checks R3 (no transcription) against the DIAGNOSTIC, never the TEST or the diagnostic TYPE -> "rich" is over-claimed:
- GENUINE rich (fix value absent from target+test+diagnostic, needs the hidden remote def): **rich_field_rename, rich_method_rename, rich_return_obj_vs_tuple, rich_callback_sig** (the ARG-ORDER is type-invisible+remote-necessary even though arity is in the diag).
- RELABEL -> plain (diagnostic already conveys the fact): **rich_field_type_change** (diag says `qty: str`), **rich_arity_drift** (diag prints full `tuple[str,int,float]` shape), **rich_generic_widened** (diag prints `list[int]`; sum-vs-max comes from the test).
- CUT/rewrite **rich_enum_member** — R1 VIOLATION: the fix names ACTIVE/PAUSED appear VERBATIM in the test assertions -> solvable from test alone, no remote read. (Verifier missed it because R3 only scans the diagnostic.) Fix: assert on return values for inputs built by integer value (`Status(1)`) so member names never appear.
- REPLACE **plain_bad_index_key** — violates the suite's OWN R5 (`t[str(k)]` strawman the header bans).
- Controls well-matched (multi-file, multi-site, remote type used correctly, 0 diagnostics). R4 clean.
- Too-easy / won't-discriminate: rich_enum_member, rich_arity_drift.

**Verifier fix:** extend the R3 token scan to the TEST source (would auto-catch enum_member).

**~15 NEW rich-weighted proposals (clear R1-R3): ** protocol/structural-typing mismatch (rename a Protocol method), wrong generic type-arg across files (`Box[Coord].value.x`->`.lat`), overload/Literal misuse, kw-only & positional-only signature changes, NewType swap (UserId vs OrgId), dict-value-changed-to-dataclass (`stock()[sku]+1`->`.count`), sync->async/coroutine (needs `await`), inherited-attr-moved-to-base, units/enum-confusion, overload-return-depends-on-arg, + a 4th control (accumulator-reset logic bug). Target ~28-30 tasks, ~13 genuine rich.

**OPERATIONAL (new): pyrefly `init` opens a Unix socket to a shared daemon and DEADLOCKS under concurrency -> run MultiFileEnv/task eval STRICTLY SEQUENTIALLY, never parallelize the harness.**

**Plan unchanged:** finalize v2 (relabel 4, cut enum_member, replace bad_index_key, extend verifier, implement ~10 strong new rich, re-verify -> ~28-30) -> ONE clean powered run across the full channel matrix (context{partial,preread} x delivery{none,sync-eager,sync-lazy,live-raw,live-clean,live-clean-UNGATED} x content{plain/rich} + mechanism{type-only,syntax-only}) -> re-skeptic -> reframe PAPER.md/README. Task #48.

---

## 2026-06-09 — FRAMING decision (Ian): tasks are an UPPER BOUND on type-feedback value, not representative of general coding. Scope the claim accordingly.

Ian asked: are the tasks intentionally things a type checker should help with (an upper bound), not representative of general work? YES, by construction — every non-control task is selected so pyrefly fires a guiding diagnostic. So rich/plain = the FAVORABLE/type-checkable subset (conditional/UPPER-BOUND value), NOT the population value of attaching a checker to general coding (real logic bugs are type-blind; our real-SWE-rebench pilot got 0 bug-relevant diagnostics). DECOMPOSITION (why the design still works): the ANTI-THRASHING/edit-error value (linter catching the model's own broken edits) is GENERAL — shows up even in the controls; the TYPE-LOCALIZATION value is the narrow upper-bound part. So controls = type-blind regime (≈ general edit-hygiene floor), rich/plain = type-checkable ceiling; type-only/syntax-only arms decompose them. Population value ≈ general-component + P(type-checkable)·type-component. PAPER must state this external-validity scope explicitly and NOT claim "type feedback helps coding agents in general" — claim is "value where the checker can speak + how to deliver it." Keep/strengthen controls (they represent the complement). Task #48.

---

<!-- Add new entries above this line. Format: ## YYYY-MM-DD — short title -->

## 2026-06-09 — v2 task fixes landed (ALL OK) + intermediate arms retired

**Task fixes applied (per v2 review):**
- Relabeled rich_generic_widened / rich_field_type_change / rich_arity_drift → group="plain"
  (the diagnostic TEXT conveys the remote fact; not genuinely channel-necessary).
- CUT rich_enum_member: its test named ACTIVE/PAUSED (fix leak); a clean rich enum needs a
  remote-factory design — deferred to expansion.
- Replaced plain_bad_index_key (`t[str(k)]` strawman, R5 violation) with plain_composite_key
  (remote dict re-keyed to tuple[str,int], indexed by bare str — realistic schema drift).
- Added 2 genuinely-rich tasks (diagnostic names the BREAK, only the remote DEF supplies the FIX,
  fix token absent from test): rich_const_rename (renamed remote constant via import),
  rich_value_object (remote dict values became Money objects; .cents lives only in money.py).
- Added 4th control: ctrl_accumulator_reset (accumulator reset inside loop; type-invisible).
- Extended verifier R3: leak check now scans the TEST source too (a fix token in the test means
  solvable without the remote = R1 violation). Caught plain_none_default_arg's generic "[]" token
  → tightened to "extend([])".

**Verifier: ALL OK.** N=17 {rich:6, plain:7, control:4}. Every task: buggy→FAIL+pyrefly-fires,
gold→PASS+pyrefly-clean, no leak, controls fire 0 type errors.

**Intermediate arms — interpret-and-retire (per Ian: "kill the intermediate arms once we have this,
interpret whatever data we have, plan a fresh run once we have good tasks"):**
- clean-live completed 72 rollouts (6 tasks × 12 seeds). type-only / syntax-only never started.
- CLAIM 2 (format vs timing), n=72 matched: none 0.375 / raw-live 0.417 / clean-live 0.389 / sync 0.556.
  clean-live≈raw-live (p=0.774) — clean format does NOT recover the gap (marker bug was not the cause).
  sync>clean-live (b=21 c=9, p=0.043). clean-live≈none (p=1.000). => value is synchronous TIMING,
  not delivery format. Corroborates single-file (sync 0.578 vs none 0.411, p=5e-4). Caveat n=72.
- CLAIM 3 (type-only vs syntax-only anti-thrashing decomposition): NO DATA — deferred to fresh run.
- Retiring intermediate arms; NOT relaunching on v1 tasks. Babysit loop purpose served.

NEXT: adversarial review of the strengthened v2 set, then ONE clean powered run across the full
channel matrix on v2 (the real final data, incl. the mechanism arms).

## 2026-06-09 — adversarial review #2 of v2 set: TWO BLOCKERS found + fixed; set finalized (ALL OK, N=20)

A second adversarial reviewer (HN/practitioner + channel-necessity lens) audited the strengthened v2 set.
Verdict: 5/6 rich tasks genuinely channel-necessary, but it surfaced **two blocking infra bugs** that
would have invalidated the final run, plus several quality holes.

**BLOCKERS (both fixed):**
- B1: synth_mf.py imported the BROKEN v1 suite (`from scripts.synth_tasks_mf import TASKS_MF`) — the
  final paper run would have used v1 (the reversers!). Fixed: import `synth_tasks_mf2 import TASKS_MF2 as
  TASKS_MF`. Verified the 20 mf2 tasks expose every key the runner reads (name/group/files/target/test).
- B2: `--rich-signal` enrichment (`_diag_text`) scanned only the TARGET file for the named symbol's def —
  but on rich tasks the def is REMOTE, so enrichment appended nothing (no-op exactly where it should
  matter). Fixed: `_enrich_diag` now scans the whole workspace (`env.list_files()`). NOTE: --rich-signal
  is NOT in the planned final matrix (content axis = task GROUP, not the flag), so this didn't block the
  run, but the enrichment arm is now usable.

**Quality fixes:**
- Demoted rich_callback_sig -> plain_callback_sig: the diagnostic prints the full new sig `(int,int)->int`,
  so the arity (the remote fact) is in the diagnostic => plain, not rich.
- Renamed mislabeled plain tasks (were `rich_*` with group=plain): generic_widened, field_type_change,
  arity_drift. Name prefix now matches group (avoids miscount).
- A1 ARTIFACT FIXED — plain_optional_get: the 0-fallback made a truthiness fix (`v or 0`) behaviourally
  identical to gold, so the distractor didn't bite. Redesigned: missing->-1 sentinel, configured 0 must
  survive as 0 => `v or -1`/`if v` both clobber the 0 and FAIL the test; only `is not None` passes. Added
  wrong_typevalid=`v or -1` so the verifier now PROVES the test gates it ("guarded"). Renamed files
  (settings.py/fill.py) to de-dup from ctrl_truthiness.
- VERIFIER BUG FIXED: the R2 false-all-clear guard was (a) dead code (no task defined wrong_typevalid) and
  (b) inverted/unwired (`fc="BAD" if wf_clean and wf_fail` — that's the GOOD case, and `ok` ignored it).
  Now: HOLE (type-valid wrong fix PASSES test) is a hard failure wired into `ok`; "guarded" = test rejects
  it; "fires" = the wrong variant itself trips pyrefly.
- Stripped the `+ 0` tell from generic_widened (buggy now `return scores()[name]`, fires bad-return).

**New rich tasks (reviewer proposed 6; I kept the 3 that survive the test<=>diag invariant):**
- KEPT rich_protocol_rename (Protocol method draw->render; runtime AttributeError + missing-attribute),
  rich_attr_moved (attr moved into inherited meta["tag"]; runtime AttributeError + missing-attribute),
  plain_overload (overload returns list[str]; diagnostic reveals the type => plain).
- DROPPED rich_newtype (NewType is runtime-identity: buggy PASSES the test => violates test<=>diag),
  rich_enum_rename (same trap as cut enum_member: test leaks the name, or Status(1) solves it without the
  remote), rich_alias_retyped (diagnostic surfaces underlying list[int] => dup of generic_widened/plain).

**FINAL v2 SET: N=20 {rich:7, plain:9, control:4}. Verifier ALL OK.** Genuine-rich item-n=7 (reviewer
target was 10-12; further rich requires novel bug-classes that don't collapse to transcription/runtime-
identity — noted as future work). Runner wired to v2. Ready for the powered run.

DEFERRED (documented, non-blocking): populate wrong_typevalid for the remaining non-control tasks (only
optional_get guarded so far); minor realism polish (none_default_arg `extend(None)` is slightly synthetic);
strip local-var hint in return_obj_vs_tuple. The B2 enrichment fix is untested end-to-end (no rich-signal
arm in the planned matrix).

NEXT DECISION: scope the final powered run (full channel matrix is ~9-10 arms × 240 rollouts ≈ multi-day
on one GPU) — present scope/seed tradeoff to Ian before committing GPU-days.

## 2026-06-11 — oracle positive-control arm BUILT + QUEUED (run after the 9-arm final)

WHY: the 6 partial arms are a null (sync≈none, live trends negative) and arm 7 (preread) is
tracking toward null too (early 12/20: none 0.625 / sync 0.677, p=0.49 — NOT the single-file
+17pp complementarity). A null is only credible with a positive control proving the channel CAN
move behavior. Hypothesis for the null: the type signal is SQUEEZED — redundant with the failing
test (and, in preread, with the visible code), and where it's novel (partial) it's un-actionable
because the 7B won't navigate to the remote file. Live injection also seems to derail the small model.

IMPLEMENTED: scaffold/stream_agent.py gained `oracle_hint` — when set, _diag_text() delivers the
oracle string IN PLACE OF the pyrefly diagnostic, gated the same way (only while the file still
errors). scripts/synth_mf.py gained `--oracle {loc,fix}`; build_oracle() derives the hint per task
from buggy-target vs gold_target via difflib (auto, no manual authoring):
  loc = "bug on line(s) 5, 9 — type/contract mismatch. Fix those." (perfect WHERE, no fix value)
  fix = the gold replacement lines verbatim (the answer; ceiling of helpful feedback)
Verified: compiles; hint previews correct for field_rename/value_object/const_rename.

QUEUED: scripts/run_oracle.sh runs the partial-context ladder paired on final_partial_AC's seed grid:
  none(A, existing) -> pyrefly(C, existing) -> oracle-loc -> oracle-fix
  -> runs/agent/final_oracle_loc.json, final_oracle_fix.json (C --c-eager --oracle loc|fix, 8 seeds)
DIAGNOSIS: climb at oracle-loc => bottleneck is localization; only at oracle-fix => execution;
flat even at oracle-fix => model/harness can't act on feedback at all (paradigm needs rethink).
BABYSIT must, after the 9-arm analysis, launch `nohup bash scripts/run_oracle.sh` detached and
keep watching final_oracle_loc/fix, then McNemar each vs final_partial_AC cond A (none).

## 2026-06-11 — steered-prompt arm (gentle) added to the follow-up queue

Reason (Ian's question): the run does almost NO steering. SYS_LINE only OFFERS the read tool
("To READ another file for context: <read .../>") and has one generic, all-conditions line
("A static analyzer may also surface diagnostics; use them to catch mistakes early"); announce_lsp
(the explicit "use the checker") is OFF in the final matrix. Nothing links a diagnostic to reading
the named file. So the null may be partly a PROMPTING gap (latent-but-unelicited channel value).

IMPLEMENTED: stream_agent gained `steer_hint` (appended to the system prompt). synth_mf gained
`--steer {gentle,directive}`. GENTLE text (chosen): "Note: the static type-checker's diagnostics are
reliable — when one appears, treat it as a real problem and address it. If a diagnostic refers to a
name (a field, function, or import) that comes from another file, reading that file is usually the
quickest way to find the correct fix." (directive variant also implemented, not run.)

DESIGN: 2x2 partial-context, paired on final_partial_AC's seed grid:
  plain{A=none, C=pyrefly} [DONE] vs steered{A,C} [final_steer_AC.json, --conds A,C --c-eager --steer gentle]
  KEY = the INTERACTION: does steering help C (pyrefly) MORE than A (none)? -> channel value is
  latent-but-unelicited. steered-A vs plain-A alone = "does telling it to read help regardless."

QUEUE: consolidated into scripts/run_followups.sh (oracle_loc, oracle_fix, steer_AC). Compiles.
*** BABYSIT TODO: when all 7 final_* arms are done, run the 9-arm analysis, THEN launch
    `nohup bash scripts/run_followups.sh > /tmp/run_followups.out 2>&1 &` and keep babysitting
    final_oracle_loc/fix + final_steer_AC; McNemar oracle rungs vs final_partial_AC.A,
    and the steered 2x2 interaction. Update the /loop prompt to this at the next heartbeat. ***

## 2026-06-11 — FINAL 9-ARM RESULT (n=160/arm, 20 tasks x 8 seeds, Qwen2.5-Coder-7B, real pyrefly)

VERDICT: a clean NULL — NOTHING survives Holm across the primary family (all Holm p >= 0.67).
The single-file complementarity (+17pp, p=5e-4) does NOT replicate on multi-file at this power.

Resolve rates (overall | rich | plain | control):
  partial none(A)   0.456 | .46 .47 .41      PREREAD none(A)  0.537 | .59 .51 .50
  partial sync(C)   0.481 | .50 .47 .47       PREREAD sync(C)  0.594 | .68 .60 .44
  raw-live 0.425 | clean-live 0.425 | clean-ungated 0.412 | type-only 0.431 | syntax-only 0.481

CLAIM 1 COMPLEMENTARITY: preread sync vs none = +5.7pp, McNemar b28 c19 p=0.243 (excl-duds p=0.184),
  seed-block AGREE (both halves favor sync). Interaction: preread gap +5.6pp vs partial gap +2.5pp
  -> directionally on-hypothesis but tiny + n.s. Underpowered, not a replication.
CLAIM 2 TIMING-vs-FORMAT: sync~none (p=0.67); ALL live arms slightly NEGATIVE vs none (raw .425,
  clean .425, ungated .412); raw-live seed-block DISAGREE (noise). No delivery mode helps; live mildly hurts.
CLAIM 3 MECHANISM: type-only~none (p=0.56), syntax-only~none (p=0.64), full-sync~syntax-only (p=1.0). Nothing.

THE TWO REAL SIGNALS (both honest, both underpowered):
  (a) CONTEXT is the lever, not the diagnostic: prereading the remote files lifts resolution
      ~+8-11pp (none .456->.537, sync .481->.594); the type channel adds little on top.
  (b) Under PREREAD-SYNC the gain is GROUP-DIFFERENTIAL in the predicted direction: rich +9pp
      (.59->.68), plain +9pp (.51->.60), but control -6pp (.50->.44) — feedback helps
      type-checkable tasks and is NOISE on type-blind ones. On-message but n.s. (n/group 56/72/32).

INTERPRETATION: the type signal is SQUEEZED — redundant with the failing test (and, in preread,
the visible code), and where novel (partial) it's un-actionable (the 7B won't navigate; reads=0).
The oracle ladder + steered 2x2 (now launched) disambiguate localization vs execution vs prompting.
Paper pivots from "complementarity helps" to "when, and why, type feedback fails to help a 7B" +
the context-is-the-lever finding, pending the follow-up controls.

## Interpretation note (pre-oracle) — "are LSPs useless for agents?" (2026-06-11)
Provisional read while PHASE-B runs. The data does NOT yet support "LSPs are useless for agents";
it supports a narrower, mechanism-flavored claim. Pinning this BEFORE the oracle arm decides it,
so the framing isn't retrofitted to the result.
- CLEAN NEGATIVE (own it): the *Streams thesis* — that live/in-stream delivery beats batch-sync —
  is negative. D arms 0.41-0.42 sit BELOW do-nothing (A 0.456). Streaming diagnostics hurt.
- NOT ESTABLISHED: "LSP signal is worthless." Confounds: (1) one 7B model; (2) single-shot,
  test-bounded, cross-file type-bug repair = ONE use of an LSP (diagnostics-as-correction), not
  nav/autocomplete; (3) n=18x8, underpowered vs a true ~+5pp; (4) mechanism points at AGENCY not
  signal — preread (file in context) shows the rich/plain +9pp sync lift, partial (must <read>)
  does not; rich_attr_moved under oracle-LOC fails with reads=0 (knows WHERE, won't retrieve).
- DECIDER = final_oracle_fix. If handing the GOLD FIX still doesn't lift pass@1 -> capability floor
  ("this 7B can't act on any feedback"), which says ~nothing about LSPs. If it DOES lift -> the
  realistic-arm null is a retrieval/elicitation problem, and "useless" is the wrong word.
- PREFERRED HEADLINE: not "LSPs useless" (one model, easily rebutted) but "retrieval/context is the
  lever, not feedback delivery": +8-11pp from putting the file in context, ~0 (and can hurt) from
  streaming a diagnostic at it. The negative becomes mechanistic (agent's failure = not-reading),
  not dismissive.

## FOLLOWUP RESULTS — oracle ladder + steered (2026-06-12, complete, duds-excl n=144/arm)
ORACLE LADDER (partial context, paired on final_partial_AC):
  none .451 / sync .472 / oracle-LOC .250 / oracle-FIX .389
  loc vs none b40/c11 p<0.001 (rich+plain b32/c10 p=0.001)  -> perfect LOCALIZATION significantly HARMS
  fix vs none b33/c24 p=0.289 (rich+plain b26/c19 p=0.371)  -> the GOLD FIX does NOT beat doing nothing
  fix vs loc  b6/c26  p=0.001                               -> fix >> loc (monotonic recovery, but only to baseline)
  seed-block fix vs none deltas [-0.056,-0.069] (both halves agree fix <= none)
  => EXECUTION/CAPABILITY FLOOR: the 7B can't convert even the handed answer into a passing edit above its own
     baseline; and prescriptive mid-stream injection DERAILS (loc worst). Localization is NOT the bottleneck.
  CAVEAT: oracle gated on pyrefly firing -> controls never receive it (their numbers are noise; judged on rich+plain).

STEERED 2x2 (gentle system-prompt nudge to trust diagnostics + read the named remote file):
  plain A .451 / C .472 (C-A +.021)   steer A .549 / C .576 (C-A +.028)
  INTERACTION (steer C-A)-(plain C-A) = +.007  -> steering does NOT unlock the diagnostic channel
  steerA vs plainA b19/c33 p=0.070 (seed-block +.111/+.083 BOTH halves agree)
  steerC vs plainC b21/c36 p=0.063 (seed-block +.194/+.014)
  => the steer nudge LIFTS BOTH arms ~+10pp (robust for A), but EQUALLY -> it boosts general AGENCY/retrieval,
     not diagnostic value. CORRECTION to the earlier interim read: the +10pp is NOT task-order bias; it survived
     the full 8-seed run with seed-block agreement (esp. cond A).

BOTTLENECK VERDICT: not localization (harms), not feedback availability (gold fix doesn't help). The binding
constraints are (1) under-retrieval / agency — a generic "go read the file" nudge buys ~+10pp — and (2) weak
execution — even handed the fix it doesn't clear baseline. The realistic diagnostic channel stays null; prompting
moves the AGENT, not the CHANNEL. Motivates: PHASE-C MoE smoke (does scale lift the execution ceiling AND make
the channel matter?) and the auto-inject/auto-read feature directions that bypass retrieval.

## PHASE-C SMOKE VERDICT — Qwen3.6-35B-A3B (2026-06-12, 4 seeds, duds-excl, descriptive)
MoE rich+plain: none .929 / sync 1.000 / oracle-fix .982   (overall: none .903 / sync .972 / fix .958)
by group none: ctrl .81 plain .89 rich .96 ; sync: ctrl .88 plain 1.0 rich 1.0 ; fix: ctrl .88 plain .96 rich 1.0
McNemar rich+plain: sync vs none b0/c4 p=0.125 (seedblock +.107/+.036) ; oracle-fix vs none b1/c4 p=0.375 (+.071/+.036) ; fix vs sync b1/c0 p=1.0
DECISION (rule: rerun iff (a) oracle-fix lift AND (b) sync>none, ELSE pause+harden):
  (a) and (b) BOTH hold in DIRECTION, but HEADROOM ceiling fires (none rich+plain .929>=.9): the 35B SATURATES the
  suite, so the lifts are thin residual-headroom wisps (n=4 discordant, p>0.1, NOT significant). Cannot be a powered
  green-light. => INCONCLUSIVE-by-ceiling.
ENCOURAGING NUANCE (the real signal): the DIRECTION FLIPPED FAVORABLY vs the 7B —
  7B: sync +.021 (flat), oracle-fix -.062 (HURT).  MoE: sync +.071 (b0/c4 STRICTLY dominant), oracle-fix +.053 (HELPS).
  i.e. where the 7B couldn't use feedback (oracle-fix hurt), the 35B's residual shows feedback HELPING — consistent
  with the capability-floor interpretation (feedback becomes usable with scale).
VERDICT: NOT a clean (a)&(b) green-light (ceiling). PAUSE the as-is rerun. The binding constraint is now the TASK
  SUITE: too easy for a 35B (and too hard for the 7B — both models at extremes). NEXT = harder/more discriminating
  tasks (brainstorm §3: multi-edit cascading refactors, exploration-heavy, hallucinated-API), then re-probe at MoE
  scale where the favorable direction could become a real powered effect. Constructive path, NOT a negative paper.

## v3 CALIBRATION VERDICT (2026-06-13) — NOT CALIBRATED; the band is empty for single-shot repair
v3 HARD suite (synth_tasks_mf3, 7 tasks, compositional D2-weighted, verifier ALL OK + 4 Python-expert
reviewers no-holes). Calibration (4 seeds):
  35B (Qwen3.6-35B-A3B) none(A) 1.000 (28/28, ALL 7 tasks 4/4) / sync(C) 1.000  -> G1 FAIL (ceiling), G2 moot.
  7B (Qwen2.5-Coder-7B) oracle-fix 0.321 (rich 5/12, plain 2/8, control 2/8)    -> G4 FAIL (<0.5).
=> BOTH ENDS FAIL. The two models BRACKET OUT the entire single-shot difficulty axis: easy enough for the
7B to ACT -> the 35B ceilings; hard enough to challenge the 35B -> the 7B can't execute even handed the gold
fix. The discriminating band for single-shot cross-file diagnostic-as-correction is EMPTY for this 7B<->35B pair.
MECHANISM (the real result): the 35B self-retrieves the remote files (reads=2), so by R1's design the fix-info
it reads makes the diagnostic REDUNDANT; the 7B under-reads AND can't execute. There is no capability regime
where single-shot repair both NEEDS the diagnostic and can ACT on it -- a genuine boundary on
diagnostics-as-correction, not a build failure. Compositional bug complexity does NOT escape it (both facts are
in the remote def the strong model reads).
NEXT: abandon "harder single-shot bugs". PIVOT to a STRUCTURALLY different task class where reading-one-file is
INSUFFICIENT, so the LSP carries non-redundant info even for a strong self-retriever:
  - COMPLETENESS / find-refs / blast-radius (Phase 2 cross-file editing, brainstorm §3.1): rename across N files;
    the model can read a file but still miss site k-of-n; pyrefly enumerates ALL sites = info reading can't give.
  - PREVENTION / NAVIGATION (autocomplete-style hallucination suppression; large unfamiliar workspaces).

## mf4 BLAST-RADIUS CALIBRATION VERDICT (2026-06-14) — near-ceiling; completeness whisper only
v4 multi-file blast-radius suite (3 tasks; multi-file editing harness built+verified; 2 Python-expert
reviewers: valid, R1/R2 hold, 1 minor alias-hole in blast_const_rename). Calibration (4 seeds):
  35B (Qwen3.6-35B-A3B) none(A) 0.917 (rich 7/8, control 4/4) / sync(C) 1.000  -> G1 FAIL (near-ceiling, 0.917>0.80).
  C-A=+0.083; McNemar sync-vs-none b0/c1 p=1.0 (sync STRICTLY dominant — the 1 site the 35B missed unaided, sync
  caught — but n=1 discordant, NOT powered). cond A: mean 2.8 tests / 3.2 edits / 0.7 reads (the 35B BRUTE-FORCES
  completeness via the test loop, rarely missing a visible site).
  7B oracle-fix 0.083 (1/12) -> G4 FAIL: the 7B can't apply the multi-file gold (partly an oracle-format artifact +
  a genuine multi-file-edit floor); note it DID solve blast_field_rename under plain sync (iterative pyrefly guidance).
THIRD CEILING (v2 easy single-shot, v3 hard compositional single-shot, v4 small multi-file). THROUGH-LINE: the 35B
+ the test loop solves anything where it can SEE/REACH the sites; the diagnostic adds value ONLY when sites are
HIDDEN from the model. The completeness MECHANISM is real (sync caught the missed site; cond A iterates to find
them) but under-stressed at small, full-visibility scale.
NEXT LEVER (both reviewers): PARTIAL VISIBILITY (sites must be DISCOVERED via <read>, NOT shown — so a missed read
is a missed site only the cross-file diagnostic reveals) + DISTRACTOR sites (old token that must NOT change) +/or
SCALE (10+ files); close the blast_const_rename alias-import hole. STRATEGIC ALTERNATIVE: consolidate the boundary
result — diagnostics-as-correction is redundant for a capable self-retrieving model across a wide difficulty range.

## mf5 PARTIAL-VISIBILITY SMOKE (2026-06-15) — *** THE CHANNEL DISCRIMINATES ***
v5 partial-visibility blast-radius (1 task partial_field_rename: col->x in 5 HIDDEN site-modules among 6
distractors, only the primary shown; --max-reads 8 vs 11 modules). 35B, 4 seeds, cond A,C c-eager:
  cond A(none): 0.25 (1/4)  reads 7.8 (HIT cap) edits 2.5 (incomplete) tests 2.2
  cond C(sync): 1.00 (4/4)  reads 7.8 (the RIGHT modules) edits 4.0 (complete) tests 1.0
  GAP C-A = +0.75. McNemar direction b0/c3 (sync strictly dominant). Smoke (n=4, 1 task) -> strong
  DIRECTIONAL signal, not yet powered.
MECHANISM CONFIRMED: cond A exhausts its read budget hunting the HIDDEN sites blindly and edits incompletely
-> fails; cond C gets pyrefly's cross-file site list -> reads exactly the broken modules, edits completely,
solves with 1 test. The diagnostic carries NON-REDUNDANT info (WHICH cross-file modules are broken) that the
model cannot get by reading (can't read all) or the test loop (green/red, no location).
=> Resolves the whole arc: diagnostics-as-CORRECTION are redundant for a capable self-retriever (v2/v3/v4 all
ceilinged), but the cross-file COMPLETENESS/find-refs channel is non-redundant when SITES ARE HIDDEN + reads
capped. LSP value = DISCOVERY, not correction.
NEXT: cond A 0.25 is a touch below the [0.40,0.80] band (read cap 8 vs 11 modules is tight) — tune (max-reads
up / fewer distractors) to land cond A in-band; then EXPAND to a multi-task mf5 suite (rich + plain + control
analogues) + POWERED run (8 seeds, Holm) to confirm. This is the paper's positive result.

## mf5 PARTIAL-VISIBILITY CONFIRMATION (2026-06-15) — *** CONFIRMED: type-specific, non-redundant ***
35B, 4 seeds, --max-reads 8. RICH (partial_field_rename + partial_const_rename): none(A) 0.000 / sync(C) 1.000,
C-A=+1.00, McNemar b0/c8 p=0.008. CONTROL (partial_logic_control, pyrefly-blind): A 0.000 / C 0.000, C-A=+0.00.
Per-task: both rich 0->1 (capA_reads 7.8/8 every time); control 0->0 (capA_reads 7.0).
=> CLEAN DISSOCIATION. The cross-file type-diagnostic completeness channel is (a) NON-REDUNDANT — the 35B
literally cannot solve unaided (0%) because it can't discover the hidden sites within the read budget, and
(b) TYPE-SPECIFIC — the same iteration loop with no type signal (control) yields nothing (0->0), so the value
is pyrefly's cross-file ENUMERATION of broken sites, not generic feedback/turns.
THE COMPLETE STORY: diagnostics-as-CORRECTION are redundant for a capable self-retriever (v2 easy, v3 hard
compositional, v4 small multi-file all CEILINGED — the model reads files + the test loop finds visible sites);
the cross-file COMPLETENESS/find-refs channel is NON-REDUNDANT + type-specific when sites are HIDDEN and reads
are capped (mf5). LSP value for a capable coding agent = DISCOVERY, not correction.
NOTE: cond A=0 is a near-floor (max-reads 8 vs 11 modules); the gap is the strongest form of non-redundancy
(necessity). For the paper, a READ-BUDGET DOSE-RESPONSE (vary --max-reads; cond A rises toward C as reads->#modules)
would show the channel's value as a function of how expensive manual discovery is.
NEXT: POWERED run — 8 seeds, more rich tasks (method/import/attr renames) + 2 controls + a read-budget sweep,
McNemar+Holm. Then write up the positive (mf5) paired with the v2/v3/v4 negatives.

## mf5 FREE-READ DIAGNOSTIC (2026-06-16) — *** the mf5 "positive" was a READ-CAP ARTIFACT ***
Re-ran mf5 with a GENEROUS budget (--max-reads 20 --max-turns 30 --max-new 4000) to test the skepticism that the
+1.00 gap was just the arbitrary read cap. 35B, 4 seeds:
  RICH: cond A success 0.88 (was 0.00 at cap 8!) / cond C 1.00.  reads-to-solve A 10.4 / C 9.0; in_tokens A 2391 / C 2206.
  CONTROL: A 1.00 / C 1.00.
TWO NEGATIVES: (1) the cond-A=0 success gap was a CAP ARTIFACT — with free reads the 35B recovers to 0.88 (it CAN
find the hidden sites by reading; we'd forbidden it). (2) the efficiency benefit is MARGINAL — cond C reads 9.0 vs
cond A 10.4 (~13% fewer reads, ~8% fewer tokens): the model reads ~9-10 of 11 modules REGARDLESS of the diagnostic;
it does NOT exploit pyrefly's site-list to skip reading (reads defensively).
=> The mf5 partial-visibility "positive result" does NOT survive honest measurement. It collapses into the project
through-line: a capable self-retriever reads what it needs (and more), so the LSP channel is largely redundant —
now confirmed for the COMPLETENESS channel, not just correction. Behavioral nugget: even handed a perfect find-refs
list, the agent still reads almost everything (doesn't trust/exploit it for efficiency).
OPEN: the only non-artificial way the completeness channel could matter is at SCALE — when #modules far exceeds
what any agent will read (e.g. 60-100), reading is GENUINELY expensive/impractical (not a rigged cap), so either C
reads ~5 vs A reads ~50 (real efficiency) or A can't cover the space and fails (real navigation). Finding that
CROSSOVER (#modules where the diagnostic starts to matter) is the honest version of the efficiency claim.

---

## 2026-06-18 — mf5 SCALE SWEEP (efficiency, measured): NO CROSSOVER → consolidate the negative

**Setup (the honest version of the token-efficiency claim).** After conceding the mf5 +1.00 partial-visibility gap was a READ-CAP ARTIFACT (cond A recovers 0.00→0.88 with free reads), tested whether find-refs (cond C = pyrefly's cross-file site list) earns its keep NON-artificially at SCALE. Same task (`partial_field_rename`, 5 hidden sites), FIXED GENEROUS budget (`--max-reads 30 --max-turns 40`), Qwen3.6-35B-A3B `--gpu-only`, 3 seeds, conds A,C. Swept repo size via `MF5_NDISTRACT`: N=15/40/80 distractor modules = **21/46/86 files**. Hypothesis: as the repo grows past the budget, reading-to-FIND the sites gets too expensive → cond A degrades while cond C (handed the sites) stays flat.

**Curve (runs/agent/mf5_sweep_{15,40,80}.json):**

| N | files | A succ | C succ | A reads | C reads | A in_tok | C in_tok |
|---|-------|--------|--------|---------|---------|----------|----------|
| 15 | 21 | 1.00 | 1.00 | 7.67 | 7.33 | 2280 | 2166 |
| 40 | 46 | 1.00 | 1.00 | 7.00 | 7.00 | 2138 | 2138 |
| 80 | 86 | 1.00 | 1.00 | 6.33 | 6.33 | 2463 | 2515 |

A−C reads: +0.33 / 0.00 / 0.00. **NO CROSSOVER.** Cond A holds **1.00** at a *flat ~6–8 reads* as the repo grows 4×; reads do NOT scale with repo size (they slightly fall). The 35B navigates straight to the broken sites regardless of how many distractors surround them — it does not brute-force-scan, so it never approaches the 30-read budget, and find-refs is redundant on BOTH success and efficiency. Neither premise behind the realistic case bit: reading is not expensive (≤8 files at 86-file scale) and navigation is not hard (no read inflation with N).

**VERDICT — consolidate the negative.** Across the full matrix the LSP channel is redundant for a capable self-retriever:
- CORRECTION (v2 easy / v3 hard-compositional / v4 small-multifile): all CEILING — the 35B reads remote files unprompted and one-shots the fix; sync/oracle add nothing (oracle-loc even HARMS, p<0.001; oracle-fix doesn't beat none = capability floor).
- COMPLETENESS / find-refs (mf4 / mf5 / this sweep): redundant at tractable scale — the agent finds all sites cheaply (≤8 reads @ 86 files); the +1.00 mf5 gap was an artifact of an arbitrary read cap.
**Honest caveats for the write-up:** (1) 7B shows a capability FLOOR (can't act even on the gold fix) — the channel can't help a model that can't execute; a footnote, not a result. (2) The partial-visibility positive was real only under an artificial read cap — we found and retracted it ourselves. (3) Untested: genuinely AMBIGUOUS navigation (sites with no import/name trail to follow) and repos beyond single-GPU agent budgets — the honest boundary of the claim.

**Recommendation:** do NOT power this — there is no effect to power. Write up the negative ("for a capable agent that already self-retrieves, type-checker diagnostics — sync or in-stream — are largely redundant for both correction and find-completeness at tractable repo scale"), with the oracle ladder as the clean capability-vs-channel separator and the artifact retraction as a methods-honesty note.

---

## 2026-06-19 — NAV + PREVENTION oracle smokes: both channels REDUNDANT → the negative is now COMPLETE

Probed the two LSP channels untested after correction/completeness/scale all came up redundant. Both attack a
DIFFERENT failure mode (brainstorm §1.2 "agency-vs-signal": features that REMOVE a decision — auto-inject find-refs /
completion / signature-help). Oracle-first (perfect channel computed from the gold task; no daemon). Qwen3.6-35B-A3B
`--gpu-only`, 4 seeds, cond A (no LSP) vs cond C (oracle channel injected up front). New harness: `<grep>` workspace
search tool (the agent's realistic find-refs alternative) + `n_attractor_edits` metric (hallucinated symbols emitted).

**Navigation** (`np_nav.json`; renamed re-export defeats grep; A=grep+read vs C=find-refs site list). Design
neutralizes the test-as-nav-oracle (top-level value mismatch, no traceback; arg-swap fix so the only hard part is
FINDING the aliased sites).

| cond | group | n | succ | reads | greps |
|---|---|---|---|---|---|
| A | nav (aliased) | 4 | **1.00** | 12.0 | 1.75 |
| C | nav (aliased) | 4 | **1.00** | 5.25 | 0.25 |
| A | control (greppable) | 4 | 0.75 | 4.75 | 0 |
| C | control (greppable) | 4 | 0.25 | 4.75 | 0 |

Smoking gun: cond A greps the canonical `combine` → misses the aliased sites by design → FALLS BACK to reading
(s0 read 26 files) and solves anyway. **Equal success (both 1.00); find-refs only saves reads (12→5).** Navigation is
self-served by call-graph reading; find-refs is efficiency-only, not capability. (control thin/noisy at n=4 — the
arg-swap reasoning itself is flaky — but the channel lifts nothing.)

**Prevention** (`np_pcomp.json` completion / `np_psig.json` signature; A=read lib vs C=true member-list/signature
injected). Metric = `n_attractor_edits` (did the model EMIT a hallucinated `.get`/wrong-call?).

| suite | cond | rich succ | **rich attr_edits** | rich reads | read-lib-first |
|---|---|---|---|---|---|
| pcomp | A | 0.92 | **0.00** | 1.17 | 12/12 |
| pcomp | C | 1.00 | **0.00** | 0.75 | — |
| psig  | A | 0.92 | **0.00** | 1.58 | 12/12 |
| psig  | C | 1.00 | **0.00** | 1.00 | — |

**G-P1 FAILS for BOTH**: rich cond A `attr_edits = 0.0` across all 12+12 rollouts — the 35B **never emitted a
hallucinated member/call** because it **reads `lib.py` first in 12/12 (both)**, learns the real API/signature, and
writes correct code first try. Even psig's type-clean value-swap (`rich_optional_swap`) — read-the-signature suffices.
The prevention premise ("the model confidently guesses without checking") does NOT hold: it reads defensively. Cond C
shaves reads (1.17→0.75, 1.58→1.0) and nudges success (0.92→1.0 = one rollout) — a TINY efficiency win, never a
capability one; the hallucination it would prevent does not occur. Controls (familiar API) ≈ no-op as designed.

**COMPLETE ARC — the unifying mechanism is SELF-RETRIEVAL.** Across every channel type, the LSP is redundant for a
capable agent in a test-driven loop:
- correction (v2/v3/v4): reads remote defs, one-shots the fix; oracle-loc HARMS, oracle-fix = capability floor.
- completeness/find-refs (mf4/mf5/scale-sweep): reads the sites; the +1.00 mf5 gap was a read-cap artifact (retracted).
- navigation (nav): reads the call graph when grep fails; find-refs efficiency-only.
- prevention completion+signature (pcomp/psig): reads the lib; zero hallucination → nothing to prevent.
The LSP's value to HUMANS (live squigglies, jump-to-ref, autocomplete) is about saving a lookup; an agent that reads
rather than guesses doesn't need it. **RECOMMENDATION: consolidate the clean negative/boundary paper.** Oracle ladder
= the capability-vs-channel separator; self-retrieval = the unifying mechanism. Honest caveats: 7B capability floor
(can't act even on gold — a footnote, the channel can't help a model that can't execute); the mf5 read-cap artifact
(found+retracted ourselves); the consistent marginal efficiency wins (C shaves a read — real but not a capability
effect, and small because workspaces are small; would grow at repo scale, same story as the scale sweep). Open
frontiers NOT pursued (the honest edge): a model that does NOT self-retrieve (mid-capability, reads-insufficiently —
risks the capability-floor confound), or genuinely non-readable facts (runtime/dynamic). All infra reusable
(suites nav/pcomp/psig, `<grep>` tool, `--channel` oracle injection, hallucination metric, `--dry-run`).

---

## 2026-06-19 — EFFICIENCY-as-policy steer-smoke: prompting does NOT elicit LSP preference → training is the gap (Ian's hypothesis confirmed)

The one axis untested by everything prior: not whether the LSP INFORMS (redundant — the model reads & learns the
same) but whether USING the LSP is a more token-efficient retrieval ACTION than reading, and whether the model can be
made to PREFER it. Built PULL LSP actions <defn sym/> (go-to-def/hover) + <findrefs sym/> (oracle-backed, ~6-line
responses) the model can ELECT instead of <read>; an effic suite where the needed symbol lives in a ~14k-char
biglib.py (a <read> returns the whole file ~3500 tok; <defn> returns ~6 lines ~50 tok — 50-100x cheaper). 35B,
4 seeds, 2 tasks, cond A, 3 policy arms differing ONLY in prompt/tools. (Read-truncation confound caught+fixed:
read cap 4000->16000 so the big lib returns fully and the read arm SUCCEEDS rather than failing on truncation.)

| arm   | tools/prompt                         | success | mean in_tokens | reads | n_lsp |
|-------|--------------------------------------|---------|----------------|-------|-------|
| read  | <read> only                          | 1.00    | 4217           | 1.0   | 0     |
| lsp   | + <defn>/<findrefs> available        | 1.00    | 4258           | 1.0   | **0** |
| steer | + hard prompt "PREFER the LSP"       | 1.00    | 4340           | 1.0   | **0** |

**The model NEVER used the LSP tool — n_lsp=0 in ALL 24 rollouts, including the steered arm.** Verified it is not a
parsing bug: the model never emitted `<defn>`/`<findrefs>` text at all (0/8 in steer); the steer config was correctly
applied; the stream shows it read the full lib. in_tokens is FLAT (~4217/4258/4340) — it reads the big file every
time, and the steer instruction even adds a few tokens it ignores. The ~3500-token saving a <defn> would give is real
and available, but the model will not take it. **VERDICT (c) NEEDS TRAINING: a prompt does NOT shift the strong
read-the-file policy.** Mechanism: the model does not EXPERIENCE token cost, so it has no reason to weigh the cheap
<defn> against its read habit; a "prefer it" instruction gives no counterweight. Only a training signal that prices
tokens (RL reward) or demonstrates the behavior (SFT) would instill the preference.

**CRITICAL corollary for the self-distillation plan.** Ian's cheaper path was "hard-prompt a teacher to prefer the LSP,
then self-distill the solved trajectories." But the steer arm IS that hard-prompted teacher, and it produced ZERO
LSP-using trajectories — so there is **nothing to clone**. The cheap prompted-teacher distillation route is BLOCKED.
To unblock, the elicitation must FORCE the behavior, not request it:
  - **forced-tool arm** (next cheap step): DISABLE <read> for the lib (or penalize it) so the model MUST use <defn>/
    <findrefs> to solve. This (a) confirms the model CAN solve via the LSP at ~50 tok vs ~3500 (the efficacy +
    economics, the upper bound), and (b) HARVESTS the gold LSP-using trajectories for SFT. Then SFT the student on
    them and re-run the steer-smoke: does the preference now stick UNPROMPTED?
  - if SFT-on-forced-trajectories doesn't generalize -> **RL** with reward = solve at minimum tokens (LSP actions
    available, <read> allowed but token-priced). This is the proper-but-heavy path.

NET for the paper: the project's full result is now (1) LSP feedback as INFORMATION is redundant across every channel
(correction/completeness/navigation/prevention) because a capable agent self-retrieves; (2) the only residual value
is token-EFFICIENCY (LSP-query << file-read), which is a POLICY the model does not adopt by default OR under
instruction — establishing the training problem as the real open contribution. Infra all reusable: <defn>/<findrefs>
pull tools, --lsp-tools, --steer preferlsp, effic suite, n_lsp metric, 16k read cap.

---

## 2026-06-19 — 7B OPSD-harvest feasibility: GO. Forced-tool arm manufactures clean LSP-using teacher trajectories

Ian's plan: OPSD with a FORCED-tool arm as teacher — block <read> so the model MUST use <defn>, harvest the solved
LSP-using rollouts, LoRA-SFT the 7B, re-test if the preference sticks unprompted. (This fixes the blocker that
free-sampling can't: in the wild n_lsp=0, so there are no LSP trajectories to keep.) Built force_lsp (deny reads of
non-editable files under the NORMAL prompt -> trajectories stay in the deployment distribution). 7B feasibility smoke
(Qwen2.5-Coder-7B, effic suite, 6 seeds x 2 tasks):

| arm    | task     | success | reads | n_lsp | note |
|--------|----------|---------|-------|-------|------|
| read   | account  | 0.33    | 0.5   | 0     | failures had reads=0 -> 7B GUESSED .deposit/.balance -> thrashed (the 7B often doesn't retrieve at all) |
| read   | transfer | 1.00    | 0.0   | 0     | GUESSABLE -> no retrieval needed -> useless as an efficiency probe |
| forced | account  | 0.50    | 0.0   | 2.0   | blocked -> queries <defn sym="Account"> -> real def -> correct .credit/.worth edit -> pass |
| forced | transfer | 1.00    | 0.0   | 0     | still guesses |

HARVEST = 3/6 account rollouts (s1,s2,s5) resolved AND used <defn sym="Account"> (s2,s5 clean: 1 edit/1 test). The
forced arm DOES manufacture clean teacher trajectories the wild model never produces. VERDICT: **GO — 7B self-distill
is viable.** (Fixed a redirect bug en route: the first run's block message used a literal placeholder <defn sym="NAME">
which the weak 7B copied verbatim -> queried "NAME" -> useless; the block now names available symbols + a real example.
Minor residual: the 7B still wastes one call on the "<the symbol name>" example literal before self-correcting to
"Account" — tighten the redirect to drop the copyable literal.)

RECOMMENDED NEXT (the OPSD pipeline; one new piece = the SFT loop):
1. EXPAND the effic suite to ~10-15 NON-GUESSABLE retrieval tasks (drop transfer-style guessable APIs — retrieval must
   be REQUIRED; un-guessable member/signature names like Account.credit/.worth).
2. Bigger forced harvest (more seeds/tasks) -> the SFT training set (sft_input_ids/sft_labels are already captured per
   rollout; filter to resolved AND used-a-real-<defn>).
3. BUILD the LoRA-SFT loop (the only piece not yet in the harness) and SFT the 7B on the harvested trajectories.
4. RE-RUN the steer-smoke (read available, NO block, NO steer) = the payoff: does n_lsp rise UNPROMPTED and in_tokens
   drop toward ~700 (vs the ~4200 read baseline)? That answers Ian's whole hypothesis: can training instill the
   token-efficient LSP-preference that prompting could not.
Infra reusable: force_lsp read-block, <defn>/<findrefs> pull tools, --lsp-tools/--force-lsp/--steer preferlsp, effic
suite, n_lsp + sft-label capture.

---

## 2026-06-19 — OPSD payoff: NULL on the literal <defn>-preference, but training DOES move the retrieval policy (distribution gap = the key lesson)

Ian's hypothesis: prompting can't make the model prefer the cheap <defn> over reading (35B n_lsp=0 even steered) —
can OPSD TRAINING? Forced-tool harvest (block <read> -> 7B must use <defn>) -> 23 clean teacher trajectories (9 of 12
tasks) -> LoRA-SFT 7B (3 epochs, 0.53% params, loss 0.17->0.15) -> re-test with LSP tools available, NO block/steer.
Adapter confirmed loaded (config.adapter=runs/sft/effic_lora).

PRE (wild 7B) vs POST (SFT'd):
| arm | succ | n_lsp | %use-<defn> | reads | %retrieve | in_tok |
|-----|------|-------|-------------|-------|-----------|--------|
| PRE  overall | 0.65 | 0.02 | 2% | 0.50 | 42% | 3086 |
| POST overall | **0.83** | **0.00** | **0%** | 0.77 | **67%** | **3484** |
| PRE  seen-9  | 0.56 | 0.03 | 3% | 0.67 | 56% | 3865 |
| POST seen-9  | **0.81** | 0.00 | 0% | 1.00 | **86%** | 4350 |
| PRE  unseen-3| 0.92 | 0.00 | 0% | 0.00 | 0%  | 751 |
| POST unseen-3| 0.92 | 0.00 | 0% | 0.08 | 8%  | 887 |

**NARROW verdict — NULL (slightly counter):** the SFT'd model uses <defn> LESS, not more (0% vs 2%), and in_tokens
went UP (3484 vs 3086; seen 4350 vs 3865). The token-efficiency goal was NOT achieved — training on forced-<defn>
trajectories did not instill a <defn> preference.

**BROAD verdict — the retrieval policy IS trainable:** the SFT clearly shifted behavior — %retrieve 42->67 (seen
56->86), reads 0.5->0.77, and success 0.65->0.83 (seen 0.56->0.81). The model went from guess-and-thrash to
retrieve-then-solve. But it retrieves via its native <read>, not <defn>.

**MECHANISM — the distribution gap (the headline methodological lesson):** the harvest BLOCKED reads, so the
trajectories demonstrate "<defn> is HOW to retrieve when you can't read," NOT "<defn> is BETTER THAN an available
read." The model generalized the right META-behavior (retrieve before acting) but executed it through its
stronger-prior, now-available <read> at inference. It learned the right behavior on the wrong axis. To teach the
SPECIFIC cheap-tool preference, the harvest must show <defn> CHOSEN OVER an available read.

**CAVEATS (honest):** (1) the success gain is concentrated on SEEN tasks (in the SFT set) -> partly possible
solution-memorization, not purely a learned-retrieval skill. (2) the UNSEEN-3 (transfer/point/matrix) are GUESSABLE
(already 0.92, reads=0) so they do NOT cleanly test whether retrieval generalizes — a real generalization test needs
HELD-OUT tasks that REQUIRE retrieval. (3) only 23 trajectories, 3 epochs.

**RECOMMENDATION (next experiments, in order):**
1. PREFERENCE harvest: keep <read> available during harvest but DISADVANTAGE it (truncate/charge-tokens/penalize) so
   the model's solved trajectories show <defn> chosen OVER read -> re-SFT -> does POST n_lsp finally rise? (the direct
   fix for the distribution gap).
2. CLEAN generalization: hold out several NON-GUESSABLE (retrieval-required) tasks from the SFT set; measure
   retrieve-behavior + success on them to separate "learned to retrieve" from "memorized solutions".
3. RL escalation (now well-motivated): training demonstrably moves the retrieval policy (+25pp), so an explicit
   token-cost reward could push it to <defn> specifically — the policy is trainable, it just needs the right objective.

**PROJECT NARRATIVE (this strengthens the thesis, doesn't weaken it):** the LSP is info-redundant across every channel
(correction/completeness/navigation/prevention) because the agent self-retrieves; its only residual value is
token-EFFICIENCY, which is a POLICY the model adopts neither by default, nor under prompting (35B), nor under naive
forced-distillation (this run) — yet the retrieval policy IS trainable (SFT shifted it +25pp). So "make the agent
prefer the cheap LSP retrieval" is a real, open, training-shaped problem, and the distribution-gap finding is the
concrete next step. Infra all reusable: force_lsp, <defn>/<findrefs>, --save-sft, sft_lora.py, effic suite, OPSD pipeline.

---

## 2026-06-19 — Path-A DAgger: POSITIVE. On-policy cost-aware imitation makes the cheap-LSP preference a LEARNABLE POLICY (the gap offline RFT couldn't close)

After the offline RFT null (trained on the model's own <read> action -> reinforced reading), the lit review
(Ross-Bagnell compounding error; STaR's blind spot; DAgger/AggreVaTe arXiv:1406.5979; Revisiting-DAgger arXiv:2605.12913)
diagnosed it as off-policy exposure bias and prescribed on-policy cost-aware imitation with the rule oracle
(read X -> defn X is cost-dominant: same info, ~70x cheaper = AggreVaTe dominance). Implemented as DAgger round-0:
lead <defn sym> as the TRAINED first action under the DEPLOYMENT prompt (reads available), model continues on-policy ->
CLEAN defn-first, READ-FREE trajectories (NO <read> for SFT to clone). Harvest 9 train tasks, 8 seeds -> 71/72 solved,
all 72 read-free (vs the prior 23 read-contaminated). LoRA-SFT 7B (3ep). Re-test all 12 (3 held out), reads available,
NO lead/force/steer. Adapter confirmed loaded (config.adapter=runs/sft/effic_lora_dagger).

PRE (wild 7B) vs POST (DAgger-SFT):
| set | succ | %use-<defn> | reads | in_tok |
|-----|------|-------------|-------|--------|
| PRE  overall  | 0.65 | 2%   | 0.5 | 3086 |
| POST overall  | **0.98** | **100%** | 0.0 | **743** |
| PRE  seen-9   | 0.72 | 3%   | 0.5 | 2857 |
| POST seen-9   | 0.97 | 100% | 0.0 | 741 |
| PRE  unseen-3 | 0.42 | 0%   | 0.5 | 3775 |
| POST unseen-3 | **1.00** | **100%** | 0.0 | **751** |

VERDICT — **POSITIVE and generalizing.** The SFT'd model ELECTS <defn> in 100% of rollouts UNPROMPTED (vs 2% wild),
reads nothing, SOLVES MORE (0.98 vs 0.65), at ~4x fewer tokens (743 vs 3086). It holds on the HELD-OUT tasks
(0%->100% use-defn, 0.42->1.00 success) -> a learned POLICY, not memorization. On-policy cost-aware imitation CLOSED
the exact gap that offline RFT could not — and the off-policy-null vs on-policy-success contrast is itself a clean
methods point (Ross-Bagnell / STaR predicted it). Note the unseen-3 success jump (0.42->1.0): the wild model FAILED
those by guessing+thrashing; forced retrieval via <defn> both saves tokens AND fixes correctness.

PROJECT THESIS NOW COMPLETE & COHERENT:
1. LSP feedback as INFORMATION is redundant across every channel (correction v2/v3/v4; completeness mf4/mf5/scale;
   navigation; prevention completion+signature) — a capable agent self-retrieves.
2. The only residual value is the LSP-as-EFFICIENT-TOOL (cheap <defn> retrieval vs expensive <read>).
3. That efficiency is a POLICY the model adopts NEITHER by default, NOR under prompting (35B steered n_lsp=0),
   NOR under naive offline distillation/RFT (n_lsp->0, even counter) — BUT IS LEARNABLE via ON-POLICY cost-aware
   imitation (DAgger/AggreVaTe): 2%->100% use, +33pp success, 4x fewer tokens, generalizing.
=> The contribution: type-checker/LSP feedback is informationally redundant for a self-retrieving coding agent; its
   real value is token-efficient retrieval, which is a trainable policy that prompting and offline cloning cannot
   instill but on-policy imitation can. The method matters as much as the signal.

RECOMMENDED NEXT:
1. POWER it: more seeds/tasks; DAgger rounds 2..K (roll out the SFT'd student, relabel any residual reads); the
   tokens-to-solve curve; McNemar on success + a paired token test; a larger held-out generalization suite.
2. HONEST CAVEAT to check FIRST (the one real risk): all effic tasks are defn-SUFFICIENT by construction, so we can't
   see whether the model now OVER-prefers <defn> and fails tasks that genuinely NEED a full <read>. Add a few
   read-required tasks and confirm it learned "prefer defn WHEN sufficient," not "always defn." This is the gate
   before any powered claim.
3. Path B cost-RL (GRPO token-cost reward, OTC-PO/IKEA templates) as the principled scale-up + the cleanest
   "prefer the cheaper action" framing — now strongly motivated (the policy is demonstrably trainable).
Infra reusable: lead_defn, force_lsp, <defn>/<findrefs>, --save-sft, sft_lora.py, effic suite, run_dagger.sh,
docs/plan_opsd_efficiency_2026-06-19.md (lit review + both paths).

---

## 2026-06-20 — POWERED DAgger run: POSITIVE + BOUNDARY HELD (non-degenerate, generalizing policy)

Powered the Path-A positive on the effmix suite (12 defn-sufficient + 6 read-required boundary tasks), with the
validity gate. Mixed harvest (lead-<defn> on defn tasks, lead-<read> on read tasks) -> 139 clean trajectories
(107 defn-first + 32 read-first; CAUGHT+FIXED an sft_lora filter bug that had dropped all read trajectories, which
would have manufactured the always-defn collapse the gate exists to detect) -> LoRA-SFT 7B 3ep -> PRE(wild)+POST(sft)
retest all 18, reads available, no lead/steer. Adapter confirmed loaded. Train 9 defn+4 read; held out 3 defn + 2 read.

PRE (wild) vs POST (DAgger-SFT):
| set | succ | %use-<defn> | %read | in_tok |
|-----|------|-------------|-------|--------|
| DEFN-SUFFICIENT overall  | 0.65 -> **1.00** | 0% -> **100%** | 41% -> 0% | 3086 -> **687** |
|   seen-9                 | 0.72 -> 1.00 | 0->100 | 38->0 | 2857 -> 675 |
|   held-out-3             | 0.42 -> **1.00** | 0->100 | 50->0 | 3775 -> **722** |
| READ-REQUIRED overall    | 0.58 -> **0.79** | 0% -> 50% | 45% -> **100%** | 3508 -> 4918 |
|   name-hidden            | -> succ 0.69, %read 100 |
|   many-symbol            | -> succ 1.00, %read 100, %defn 0 (reads once instead of 4 defns) |

STATS: McNemar on SUCCESS (n=72 pairs): POST-only-solved 25, PRE-only-solved 3, **exact p=2.7e-5** (highly sig).
Token test (solved-in-both): DEFN-SUFFICIENT PRE 2304 -> POST 677 (**3.4x** by mean; paired sign p=0.15 — underpowered
at n=31 because PRE's solved subset is the easy cases it guessed cheaply). READ-REQUIRED PRE 2054 -> POST 4431 (POST
reads MORE = correct, tokens up as they should be).

VERDICT — **SUCCESS, boundary HELD.** (1) DEFN-SUFFICIENT: the trained model ELECTS <defn> in 100% of rollouts (from
0%), solves ALL (incl held-out 0.42->1.00), at ~3-4.5x fewer tokens. (2) READ-REQUIRED: it did NOT collapse to
always-defn — %read STAYED 100%, success ROSE 0.58->0.79; on many-symbol tasks it reads ONCE instead of 4 defns
(0% defn there, by economic choice). So the model learned the actual BOUNDARY: "defn when sufficient, read when
needed." Success gain is highly significant (p=2.7e-5) and GENERALIZES to held-out tasks. HONEST CAVEAT: the
token-savings MAGNITUDE is large (3-4x mean) but its strict paired significance is modest at this sample size — the
efficiency-token p-value wants more seeds; the success/policy result is rock-solid.

=> THE PROJECT'S POSITIVE RESULT IS POWERED AND CLEAN: type-checker/LSP feedback is informationally redundant for a
self-retrieving agent (correction/completeness/navigation/prevention all redundant); its REAL value is token-efficient
retrieval (cheap <defn> vs expensive <read>); that efficiency is a POLICY the model adopts NEITHER by default, NOR
under prompting (35B steered n_lsp=0), NOR under offline RFT (null/counter) — but IS LEARNABLE, GENERALIZING, and
NON-DEGENERATE via on-policy cost-aware imitation (DAgger/AggreVaTe). The off-policy-null vs on-policy-success
contrast + the boundary-preservation are the clean methods story.

RECOMMENDED NEXT: (1) one more seed-batch on defn-sufficient to tighten the token p-value (cheap); (2) WRITE-UP now —
review the Nous research-paper-writing guide; core note = you CAN benefit from an LSP (its EFFICIENCY), it requires
TRAINING the model, with the why (self-retrieval) + what-doesn't-work (prompting, offline cloning) rundown; (3) optional
Path B cost-RL (GRPO token reward) for the cleanest "prefer the cheaper action" framing + to push the token magnitude.
Infra: lead_defn/lead_read, force_lsp, effmix suite, sft_lora.py (filter fixed), analyze_dagger.py, run_dagger_powered.sh.

**POOLED (2026-06-20, 12 seeds, defn-sufficient):** token magnitude now significant. Paired token test
(solved-in-both, n=84): PRE 2108 -> POST 675 tok, POST cheaper 59/84, exact two-sided sign p=2.7e-4 (was p~0.15 at
n=31). All-rollout mean 3406->720 (4.7x), success 0.60->0.98. Success McNemar (n=144 pairs): c=57 b=3 p=6.2e-14.
%use-defn 0->100. => the efficiency claim is FULLY POWERED on both success and token magnitude. adapter confirmed loaded.

---

## 2026-06-20 — Efficiency ISOLATED (read-retrieval vs defn-retrieval, matched success) + relabel bug fixed

(A) EFFICIENCY ISOLATION (the clean control separating "LSP saves tokens" from "retrieval helps success"): a
read-trained model (retrieves via <read>, success 83%, 3484 tok) vs the defn-trained model (retrieves via <defn>,
success 100%, 687 tok). On tasks BOTH solve (n=40): read-retrieval 3191 tok -> defn-retrieval 684 tok = **4.7x cheaper,
defn cheaper on 31/40, exact sign p=0.00068**. Both models retrieve, so the saving is the ACTION CHOICE (cheap defn vs
expensive read), not retrieval-vs-guess. Efficiency claim isolated and significant.

REPRODUCIBILITY NOTE (not a result): an earlier --relabel implementation masked the read-attempt turn in place, leaving
the read+redirect in the model's context so <defn> was never trained as a first action from the clean prompt; it did
not transfer. Fixed to DROP that prefix from the SFT trace (model's own <defn> becomes the first trained action from the
clean prompt). The proper relabel test = relabel2 (running). The buggy variant is not a contrast/ablation — it was a bug.

---

## 2026-06-20 — RELABEL2: the PROPER on-policy relabel WORKS (the method, DAgger earned)

The genuine relabel (roll out the wild agent; when it reaches for `<read>` of a non-editable file the rule oracle
redirects; the agent picks `<defn>` ITSELF — its own symbol; DROP the read-attempt+redirect prefix and keep the agent's
own definition-first continuation, so the trained first action from the clean deployment prompt is the AGENT'S OWN
go-to-definition). NO gold action injected — only the retrieval CHANNEL of the agent's own behaviour is relabelled.

Pipeline: harvest 9 defn tasks (force-lsp + relabel, 8 seeds) -> 72 rows scanned -> 18 clean teacher trajectories kept
-> LoRA SFT effic_lora_relabel2 (loss 0.38->0.18) -> retest 12 defn-sufficient tasks x 4 seeds (n=48), adapter verified
== runs/sft/effic_lora_relabel2.

RESULT (POST relabel2 vs PRE powered_retest_base, defn-suff seeds0-3, n=48):
  %use-defn   0%   -> 100%   (48/48)
  %use-read   42%  -> 0%
  resolved    65%  -> 98%    (31/48 -> 47/48)
  mean in_tok 3086 -> 724    (4.3x fewer; median 2929 -> 674)
  paired: POST cheaper 37/48, worse 11, exact two-sided sign p=2.2e-4.

=> The on-policy relabel of the agent's OWN retrieval action instills the cheap-retrieval preference (0->100% defn,
65->98% success, 4.3x fewer tokens) — reproducing the powered lead-<defn> result WITHOUT any injected/teacher-forced
action. This EARNS the DAgger/AggreVaTe framing and is the method the report presents: "on-policy cost-aware imitation
— relabel the agent's own retrieval to go-to-definition." The teacher-forced lead-<defn> is a corroborating pilot /
stepping stone, NOT a separate featured method. (Prereq bug fixed this session: the force-lsp redirect's literal
self-closing <defn .../>/<findrefs .../> EXAMPLE tags were being matched as the model's action because the action-search
cursors weren't advanced past the spliced redirect obs; fixed stream_agent.py ~672 by advancing read_from/grep_from/
lsp_from to len(emitted). Sanity-gated: trained <defn> with the model's OWN real symbol confirmed before SFT. Bug fix,
reproducibility note only — not a contrast.)

Draft updated: §4 method now features the genuine relabel (lead-defn demoted to pilot); §5.1 leads with the relabel2
numbers. NEXT: real-LSP headline reproduction (effmix, real resolver), then tech-report review.

---

## 2026-06-21 — REAL-LSP HEADLINE: the value-add reproduces with a REAL go-to-definition resolver

Re-ran the full headline (effmix 18 tasks x 4 seeds = 72/arm) with <defn> backed by the REAL resolver
(mock_env.goto_definition: AST over the live workspace, NO oracle dict). PRE = untrained 7B; POST = effic_lora_powered
(adapter verified == runs/sft/effic_lora_powered).

DEFN-SUFFICIENT (n=48, real resolver):
  PRE  %defn 0   %read 42  succ 65%  mean_in_tok 3086
  POST %defn 100 %read 0   succ 100% mean_in_tok 688
  paired tokens 3086->688 (4.5x), POST cheaper 37/48, exact sign p=2.2e-4
  success McNemar: POST-only-solve b=17, PRE-only c=0, exact p=1.5e-5
READ-REQUIRED boundary (n=24):
  PRE  %defn 0   %read 42  succ 54%  mean_in_tok 2632
  POST %defn 50  %read 100 succ 83%  mean_in_tok 4844

=> The headline holds with a REAL tool, not an oracle: on defn-sufficient tasks the trained agent ELECTS the real
go-to-definition 100% (from 0), solves ALL (from 65%), at 4.5x fewer tokens (p=2.2e-4). On read-required boundary tasks
it did NOT collapse to always-defn — %read STAYS 100% and success RISES 54->83%; tokens there go UP (2632->4844) because
the trained agent correctly pays the read cost to actually solve genuinely read-required tasks (the efficiency win is on
defn-sufficient work; on read-required work it spends to succeed — an honest, non-degenerate tradeoff). The real
resolver returns content identical to the earlier oracle (validated 12/12), so this converts "we trained a preference
for a magic cheap action" into "we made a real LSP value-add." ALL <defn> results in the report now use the real
resolver. NEXT: tech-report critical+writing review -> incorporate.

## 2026-06-21 — Tech-report reviewer pass v2 (incorporated)
v2 critical+writing review: all four results HOLD (real resolver / efficiency isolated / genuine on-policy relabel /
real-LSP headline reproduces). Applied 8 FIX-NOW writing+scoping edits to docs/PAPER_draft.md (single primary headline
number 3086->688 4.5x with the other 3 token numbers labelled; 2%-vs-0% default-use disambiguated; oracle/coverage role
stated honestly in recipe+§4+limitations; "no oracle" softened to match the dead-fallback in code; boundary %defn=50/
%read=100 clarified; citation "all verified" softened to flag the 2 unverified 2026 IDs; cut v1 process-residue ->
docs/review_v1.md). Appended "## Reviewer pass v2 — open items for Ian": (1) mechanism-distinct held-out eval where
defn-sufficiency is NOT surface-predictable [highest value], (2) a 2nd model scale, (3) real pyrefly-LSP-client
replication, (4) report held-out use/tokens separately + non-clone held-out family, (5) optional Path-B cost-RL.
Shippable core today = C1+C2+C3 at 7B on synthetic suites, C4 scoped to motivation. Draft v0.4.

---

## 2026-06-21 — ITEM 1 (surface-decoupled "judge coverage"): POSITIVE on surface-keyed, CONFOUNDED on must-read

New suite effic_dc = 6 pairs x {A defn-sufficient _a, B defn-stub _b}, prompts BYTE-IDENTICAL within a pair (R7), real
<defn>. Tests whether the trained model JUDGES coverage from the <defn> return vs pattern-matches the task surface.
Ran PRE (untrained) + POST (effic_lora_powered), 12 tasks x4 seeds/arm. (Op note: a hang was a STALE pyrefly-init
daemon from an old killed session deadlocking the shared socket — purged all pyrefly procs, re-ran clean. An initial
run with a CHASEABLE stub `sym=_impl_k` showed the same coverage-judging; B was then HARDENED to an opaque subscript
`sym=_TBL[pos]` (names no _impl_, shuffled table) to block the one-hop defn-chase — that hardened run is the result.)

HARDENED RESULT (PRE -> POST, per surface-identical variant):
  A defn-sufficient:  PRE succ79 %defn0 %read54 tok2909  ->  POST succ100 %defn83 %read17 meanDefn0.8 tok1205
  B stub-MUST-READ:   PRE succ75 %defn0 %read54 tok4837  ->  POST succ88  %defn88 %read17 meanDefn1.2 tok1706
  POST B escalation breakdown (n=24): solved-via-READ=3, 2+defn-chase=3, 1-defn=15, edit-on-stub-FAIL=2.

VERDICT:
 (+) NOT SURFACE-KEYED (the reviewer's open concern, answered): despite identical surface, POST adapts to the <defn>
   RETURN — more defn calls on B (1.2 vs 0.8), higher cost (1706 vs 1205 tok), and almost never a blind edit-on-stub
   (2/24). It judges coverage from what retrieval returns, not from the task shape. On A it uses one cheap defn and
   solves 100% at 1205 tok.
 (-) CONFOUND on the must-read sub-claim: of the 15 one-defn B-solves, 10 were ONE-SHOT (n_tests=1) guesses with ZERO
   retrieval -> B is partly GUESSABLE (the gold behaviour is inferable from name/docstring despite the value-kind
   attractor guard, which only checks ONE idiomatic wrong-guess). So when cheap retrieval is blocked, the *efficient*
   trained model often GUESSES rather than escalating to a READ (reads stay flat at 0.2 on B vs A). The behavioural
   TEST LOOP is a second escape hatch (any test-driven agent can brute-force from test feedback). => the suite cleanly
   shows "judges coverage / not surface-keyed" but CANNOT cleanly establish "reads when defn insufficient": with a
   behavioural test loop + guessable targets, "must read" is fundamentally unachievable (consistent with the project's
   self-retrieval thesis — a capable test-driven agent rarely *has* to read).

NET for the report: item-1 upgrades C2/generalization on the SURFACE-KEYED question (the model judges coverage from the
return, not the suite's task shape). The stronger "learns to read on insufficiency" remains open and is arguably
ill-posed given the test loop. A fully clean must-read test would need non-inferable (arbitrary-constant) target
behaviour AND a test that hides expected outputs — flagged for Ian; not pursued autonomously (diminishing returns, core
question answered).

---

## 2026-06-21 — ITEM 2 (scale): the relabel method TRANSFERS to Qwen3.6-27B (kills the 7B-only objection)

Re-ran the genuine on-policy relabel pipeline on Qwen3.6-27B (qwen3_5 hybrid-reasoning arch, default thinking-on
config). Transformers 5.9 loads it; it emits parseable <read>/<defn>/<edit>/<test> despite <think> (no harness change
needed); sft_lora accepts qwen3_5 (no LoRA-target fix); LoRA trains on the 128GB unified box with no OOM (loss
0.30->0.09). Lighter seeds than the 7B headline (PRE/POST 2 seeds, harvest 4 seeds; 32/36 clean trajectories kept).

RESULT (12 defn-sufficient tasks, n=24/arm; adapter verified == runs/sft/effic_lora_relabel2_27b):
  PRE  (wild 27B):  %defn=0   %read=96  succ=96%  mean_in_tok=4058 (median 4154)
  POST (relabel):   %defn=100 %read=0   succ=100% mean_in_tok=726  (median 710)
  matched-success (solved-in-both, n=23): 4019 -> 726 tok = 5.5x cheaper.

VERDICT: the bigger, MORE capable model is the SAME story as the 7B — wild 27B is capable (96% success) but solves by
READING the whole file (96% read, 0% defn, 4058 tok); the genuine relabel flips it to 100% go-to-definition use, 0%
read, at maintained-or-better success (96->100%) and 5.5x fewer input tokens. The token win is actually LARGER than the
7B's (~4.5x) because the wild 27B reads even more aggressively. => the cheap-retrieval preference is NOT a small-model
artifact; the on-policy relabel method TRANSFERS across a 4x scale jump AND across a different model generation/family
(Qwen2.5-Coder-7B dense -> Qwen3.6-27B reasoning). This directly answers the dominant reviewer scope objection
("cost-preference shown only at 7B"). CAVEATS (honest): lighter seeds than the 7B headline (scale CHECK, not powered);
default thinking-on config (the <think> tokens are OUTPUT, do not inflate the input-token efficiency metric); same
effic defn-sufficient suite. Infra: run_relabel2_27b.sh, effic_lora_relabel2_27b.

## 2026-06-21 — ITEM 3 (pyrefly-LSP) SCOPED, needs Ian: pyrefly exposes only `lsp` server + `check` (no one-shot
definition CLI). go-to-def needs a JSON-RPC LSP client (~2h + the known daemon-deadlock care). The AST resolver is
already a real oracle-free go-to-def (claim holds), so item 3 is VALIDATION not capability. Plan + 3 options in
docs/plan_item3_pyrefly_lsp.md (rec: (A) standalone validation script, ~2h, low-risk; or (C) leave as-is). Not
auto-implemented (over the don't-guess bar + deadlock risk). Draft open-item-3 updated.

## 2026-06-21 — ITEM 5 (cost-RL GRPO) SCOPED -> DEFERRED. Feasible by gluing synth_mf rollouts (resolved+in_tokens) +
sft_lora's manual LoRA backward loop (swap CE -> advantage-weighted PG); ~150 lines. Deferred per Ian's "if straight-
forward": NOT low-risk (PG tuning) + GPU-expensive (multi-round harvests) + not needed (SFT relabel already the powered
headline). Plan + cheapest-informative spec in docs/plan_costrl.md. Draft open-item-5 updated.

## 2026-06-21 — FOLLOW-UPS COMPLETE. Items 1-5: (1) surface-decoupled judge-coverage = POSITIVE (not surface-keyed),
must-read confounded (flagged); (2) 27B scale = method TRANSFERS (0->100% defn, 5.5x cheaper, kills 7B-only); (3)
pyrefly-LSP = scoped, needs Ian (plan); (4) held-out reported separately (5.2x); (5) cost-RL = scoped, deferred (plan).
All written to log + memory + docs/PAPER_draft.md. Open for Ian: item-1 guessability fix, item-3 option A/C, item-5 go/no-go.

---

## 2026-06-22 — ITEM 1 (surface-decoupled) — RETRACTED coverage-judging claim: the suite's fix is DELEGATION, not retrieval

Guessability fix landed cleanly (arbitrary non-inferable gold + sha256 hash-only tests + _TBL[pos] stub; verifier ALL OK;
guess-rate 42%->0%, test-loop blocked). Clean eval (non-guessable), 12 tasks x4, PRE vs POST(effic_lora_powered):
  PRE  A succ54 %defn0 %read46 ; B succ50 %defn0 %read50   (untrained reads/reimplements/thrashes)
  POST A succ100 %defn100 %read0 meanDefn1.0 ; B succ88 %defn100 %read0 meanDefn1.0
INSPECTION (the catch): EVERY POST solve — 24/24 A and 21/21 B — fixes the bug by DELEGATING: `return combine(a,b)`
(call the helper), NEVER by reimplementing the rule. Because the gold fix is delegation, it works IDENTICALLY whether
`<defn helper>` returns the full body (A) or the opaque stub `helper = _TBL[pos]` (B) — `helper` resolves to the real
impl at runtime either way. So the agent NEVER needs the arbitrary body, and the A/B "coverage" distinction does NOT
gate solving. The earlier "judges coverage / not surface-keyed" reading (and the guessable run's apparent A-vs-B defn
delta) was an ARTIFACT: the model just learned "confirm the helper with one <defn>, then delegate", which is robust to
the stub for the trivial reason that delegation doesn't read the body.

=> RETRACT the §5.1c "judges coverage from the return" claim. The surface-decoupled suite, as built (both the guessable
and the non-guessable versions), does NOT isolate coverage-judging — its gold fix is delegation. What it DOES show
(honest, smaller): the trained model delegates efficiently (1 <defn> to confirm the helper + a call, ~100%/88% A/B)
where the untrained reads/reimplements/thrashes (~50%) — consistent with the efficiency story (cheap defn-confirm +
delegate beats read-and-reimplement) but NOT a coverage-judging result.

SCOPE: this affects ONLY item 1 (the decouple suite). The MAIN results are unaffected — the effic suite's fixes
genuinely require the RETRIEVED API (e.g. "use .credit not .deposit", knowable only from the Account defn), not
delegation; efficiency-isolation / relabel / real-LSP headline / 27B-scale all stand. A clean coverage-judging test
would need a NO-DELEGATION task where the fix REQUIRES the retrieved body (inline reimplementation, no helper to call) —
flagged for Ian, not auto-built (item 1 has now hit two distinct confounds; the main story doesn't need it).

---

## 2026-06-22 — ITEM 3 (pyrefly-LSP) DONE: <defn> == a live language server, and the headline reproduces with it

(1) VALIDATION: scripts/validate_pyrefly_lsp.py drives a real `pyrefly lsp` daemon (stdio JSON-RPC) and queries
textDocument/definition for the 12 effic symbols -> 12/12 AGREE with mock_env.goto_definition on file+defining-line
(0 disagree, 0 error). So the cheap <defn> action resolves to the same definition a production language server does.
(2) CONTENT FIX: the LSP returns the definition LOCATION (a line), not the body, so a drop-in --lsp-defn run handed the
model `class Account:` without methods and it thrashed. Patched mock_env.lsp_definition to EXPAND the LSP-resolved
location to the enclosing top-level node's full source span (the LSP drives RESOLUTION; the tool returns the body at
that location -- exactly what a go-to-definition tool does). Re-smoke then solved via live-LSP <defn> (1 defn, 0 reads,
728 tok).
(3) PROPER RUN (--lsp-defn, <defn> backed by the live daemon, 12 defn tasks x2 seeds; daemon spawned+killed per defn,
strictly sequential, no deadlock): PRE %defn0/%read25/succ58/2894tok -> POST(effic_lora_powered) %defn100/%read0/
succ100/689tok = ~4.2x cheaper. This MATCHES the AST-resolver headline (POST ~700tok, 100% defn, 100% succ).
=> the cheap-<defn> value-add HOLDS when <defn> is a REAL pyrefly language-server call, not just our static resolver.
Item 3 fully resolved (validation + working live-LSP run). Opt-in --lsp-defn (mock_env.lsp_definition / stream_agent
use_lsp_defn / synth_mf --lsp-defn) is sequential-only (daemon-per-defn cost + deadlock risk) -> default AST path for
bulk runs (validated equal); --lsp-defn confirms the live-server equivalence. Infra: validate_pyrefly_lsp.py,
run_lsp_headline.sh. Op note: daemon-spawning commands must launch via a script-file + standalone nohup in this harness.

---

## 2026-06-23 — ITEM 5 (cost-RL GRPO) DONE as scoped: mechanism trains cleanly; 1 round does NOT corroborate

The optional RL baseline (Path-B). Built scripts/grpo_cost.py + run_grpo.sh: group-sample rollouts per task ->
reward r = resolved ? 1 - lambda*min(in_tok/4000,1) : 0 -> group-normalized advantage (r-mean)/(std+1e-6) ->
advantage-weighted policy-gradient on the model's OWN action tokens (reuse the SFT label mask). lambda is the
token-cost knob, so a *resolved-cheaply* rollout out-rewards a *resolved-expensively* one — the same prefer-cheap
signal the SFT relabel instills, but via reward instead of imitation.

(1) MECHANISM — RUNS + TRAINS CLEANLY. Round-0 wild harvest: 48/72 solved, ~2048 mean in_tok, 18/48 use-defn (~38%).
Round-1 PG: loss 0.0496->0.0482->0.0456->0.0405 (monotone down), NO OOM on the 128GB box (after the F.cross_entropy
reduction='none' loss rewrite — see runs/agent/.grpo_note), adapter saved (161MB) and loads/runs in inference. So the
cost-RL machinery is real and demonstrated end-to-end.

(2) RESULT — 1 ROUND DOES NOT MOVE THE OPERATING POINT (honest negative). Retest of the round-1 adapter
(runs/agent/grpo_retest.json, config.adapter==runs/sft/effic_lora_grpo, 9 defn tasks x2 seeds, --lsp-tools, n=18):
  use-defn   6% (1/18)   -- DOWN from the 38% wild baseline (NOT toward the SFT ~100%)
  mean in_tok 3041 (all) / 1645 (solved) -- UP from 2048 baseline; incl. a 13.7k-tok thrash outlier on store_defn
  resolved   61% (11/18) -- ~= baseline (no success gain)
A single PG round at lr1e-5, K=4 is a weak/under-converged nudge; it did not bend the policy toward cheap <defn>, and
actually regressed defn-usage with an added thrash tail. (Full multi-round harvest was early-stopped at the budget
guardrail after round-1 trained clean — the point was to demonstrate the mechanism, not to win a tuning bake-off.)

VERDICT (no spin): cost-RL GRPO is mechanism-demonstrated (trains cleanly, no OOM, adapter produced) but a single round
does NOT corroborate the SFT relabel — it needs more rounds / tuning to converge. The on-policy SFT relabel remains the
efficient HEADLINE path (~100% defn, ~700 tok, ~100% succ); GRPO is scoped, demonstrated, and left as future-work, not
a competing result. The main story is unaffected (GRPO was always the *optional* corroborating baseline). Infra:
grpo_cost.py, run_grpo.sh, _grpo_retest.sh; recipe note runs/agent/.grpo_note.

---

## 2026-06-24 — ITEM 5 UPDATE (full multi-round GRPO): CONVERGES, corroborates the SFT relabel

Ran the full 4-round run_grpo.sh (K=4 PG steps/round, G=8 rollouts/task, N=4 rounds, lambda=0.5, lr=1e-5). The batch-2
single-round read ("does not corroborate") was UNDER-TRAINING, not a real negative — across rounds the policy converges.

HARVEST TRAJECTORY (train distribution, --force-lsp --relabel, n=72/round):
  round0/1 (wild):     resolved 48/72  mean_in_tok 2048  use-defn 18/48 (37%)
  round2 (r1 adapter): resolved 60/72  mean_in_tok 1740  use-defn 29/60 (48%)
  round3 (r1+2):       resolved 69/72  mean_in_tok  790  use-defn 59/69 (86%)   <- peak
  round4 (r1+2+3):     resolved 65/72  mean_in_tok 1011  use-defn 52/65 (80%)   <- mild PG wobble off the peak
Even store_defn, the round-0 thrasher (~4200 tok, unresolved), resolves in ~63 tok by round 4.

DEFINITIVE CLEAN RETEST (round-4 adapter runs/sft/effic_lora_grpo, --conds A --lsp-tools, NO --force-lsp, n=36):
  resolved 36/36 (100%)   mean_in_tokens 663   use-defn 31/36 (86%)
COMPARE: round0 baseline 67%res/2048tok/38%defn ; round-1 retest (under-trained) 61%/3041/6% (grpo_retest_round1.json) ;
SFT relabel operating point ~100%defn/~700tok/~100%res.

VERDICT (POSITIVE, replaces the batch-2 1-round negative): multi-round cost-RL GRPO CONVERGES to essentially the SFT
operating point (86% defn, 663 tok = 3.1x cheaper than baseline, 100% solved) on a clean held-out eval. Two different
training objectives — cost-aware on-policy imitation (the headline relabel) and a token-cost RL reward (GRPO) — instill
the SAME cheap-retrieval preference. This is an independent corroboration of the headline. Caveats: needs ~3-4 rounds (a
single PG round under-trains and even regresses to 6% defn); mild round-to-round oscillation (r3 86% -> r4 80% on the
harvest); small retest n=36; GRPO is NOT cheaper to RUN (multi-round harvests vs one relabel pass) so SFT stays the
headline RECIPE and GRPO is the corroboration, not a replacement. Round-1 retest archived at grpo_retest_round1.json;
final at grpo_retest.json. Infra unchanged: grpo_cost.py, run_grpo.sh.

---

## 2026-06-24 — ITEM 1 (no-delegation coverage suite) — HONEST NEGATIVE: floored, third distinct issue

Built scripts/synth_tasks_effic_nodel.py (12 tasks = 6 A/B pairs; verifier ALL OK 12) to fix the batch-2 DELEGATION
confound: gold fix must INLINE-transcribe an arbitrary multi-entry SPEC table (values-only scrambled tuple, key->value
only in source comments), so there is no callable to forward to. Escape audit was clean (delegation/introspection/
guessing/stub-pattern all blocked). Ran PRE (base 7B) vs POST (effic_lora_powered), --suite effic_nodel --lsp-tools
--conds A --seeds 3, n=36/arm.

RESULT (runs/agent/item1_nodel_{pre,post}.json):
  PRE  : 0/36 resolved (A 0/18, B 0/18)
  POST : 3/36 resolved (A 3/18, B 0/18); on B, POST reads 3/3 + uses defn 2-3/3 (correct retrieval BEHAVIOR) but solves 0
  clean coverage pairs (POST solves both A&B, PRE solves A-only): 0/6

VERDICT (honest negative, no spin): the suite is FLOORED — neither model solves even variant A (where the full defn is
handed over), so there is no A-vs-B success contrast to read. In fixing delegation we made the fix require correctly
transcribing an arbitrary table inline, which the 7B cannot do (floors at 0-3/18 on A) -> the suite ends up testing
TRANSCRIPTION ABILITY, not coverage-judging. POST does show the right retrieval behaviour on B (escalates to <read> on
the stub) but can't produce the correct edit. Item 1 has now hit THREE distinct issues: (1) guessability, (2)
delegation, (3) difficulty floor. The main efficiency story does not depend on it; §5.1c stays RETRACTED and item 1 is
left as an honest open. Did NOT auto-build a 4th iteration (per "focus on results, not the journey"). If wanted, a clean
coverage test would need either a stronger model that can transcribe (e.g. 27B) or a fix whose content is short enough
that the 7B's edit ability isn't the binding constraint. Infra: synth_tasks_effic_nodel.py, _item1_eval.sh.

---

## 2026-06-24 — COVERAGE-DISCOVERY: the cost-trained 27B reads exactly when needed, generalizing (closes the §1 hedge)

The §1 hedge ("coverage supplied not discovered") came from the 7B / labelled setup. Tested it properly on the 27B with
a clean instrument. METRIC: J = P(read | defn-insufficient) - P(read | defn-sufficient). Suites: 3 BYTE-IDENTICAL-surface
variants per topic (suf / f1ins / f2ins) — the agent can only tell coverage by calling <defn> and inspecting the span.

(1) cover (synth_tasks_cover.py, DEFN-CHAINABLE insufficiency: alias names _impl / body names _CUT). Floor cleared (27B
solves all). But insufficiency was resolvable by a 2nd cheap <defn>, so: base reads-everything (J_read=0) + escalates via
defn-chain on insuff (J_escalate=0.94, generalizing to held-out F2); cost-trained 27B (effic_lora_relabel2_27b, zero-shot)
solves ALL variants 100% via cheap defn-chains, 2.4-8.2x cheaper, barely reading -> no clean read signal (the defn-chain
escape). Informative but not a clean isolation.

(2) cover2 (synth_tasks_cover2.py, NON-DEFN-CHAINABLE: value lives in a module-level _reg("k", 53) CALL [F1'] or a
_CFG.attr = 53 ATTRIBUTE assignment [F2', held-out] — neither is a named top-level node, so goto_definition cannot return
it for ANY name; verified EXHAUSTIVELY (enumerated every top-level name across all 18 tasks, 0 defn reaches the value;
value read-only). Caught + fixed a filler leak (the value coincidentally appeared in _Aux53/_Aux50 filler -> offset all
filler numerics to >=90000). RESULT (27B, seeds 3, n=18/variant): cost-trained reads suf 0.17 / f1ins 1.00 / f2ins 1.00,
solves 1.00 across, ~1.5k tok on suf -> J_read = +0.83 on F1' AND held-out F2'; base reads everything (J_read=0). Because
surface is byte-identical, the read-decision is content-driven not shape; because it transfers across two indirection
mechanisms it is not a return-form heuristic. So the cost-trained model DISCOVERS coverage per-instance and reads only
when the cheap defn fell short.

VERDICT: coverage discovery is NOT a fundamental gap for a capable model — training the cost-preference instils the
efficient read-only-when-needed policy on top of a coverage-perception the base already has but spends indiscriminately.
PAPER.md §5.4 rewritten (positive), §7 boundary-limit + recipe caveat softened to "discovered per-instance." Caveats:
one model, synthetic, modest n, suf-read 0.17 not 0, legible insufficiency signal; a dedicated shape-keyed baseline would
make not-shape-keying airtight (next). Infra: synth_tasks_cover.py, synth_tasks_cover2.py, analysis/coverage_j.py,
run_cover.sh; data cover{,2}_{base,sft}.json.
