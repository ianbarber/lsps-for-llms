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

<!-- Add new entries above this line. Format: ## YYYY-MM-DD — short title -->
