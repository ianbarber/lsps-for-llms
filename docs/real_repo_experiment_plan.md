# Real-repository experiment plan: does cheap `<defn>` retrieval generalize?

## Scope and link to the current work

The headline result in `PAPER.md` is trained and evaluated on synthetic multi-file tasks: a large `biglib.py` contains one needed symbol, and `<defn sym>` returns the same information as `<read path="biglib.py"/>` for roughly 1/70 of the tokens. The open limitation is whether the learned cheap-retrieval preference survives in real repositories, where symbol resolution is ambiguous, methods live inside classes, imports are re-exported, and the "needed symbol" is not pre-labelled.

This plan describes a small, feasible real-repo arm that reuses the existing scaffold (`scaffold/stream_agent.py`, `scaffold/mock_env.py`, `scripts/sft_lora.py`) and the established on-policy relabel recipe (`scripts/run_relabel2.sh`).

---

## 1. Benchmark choice

**Recommendation:** start with a **curated subset of SWE-bench Verified** (500 instances, 12 Python repos), filtered down to **15–20 tasks** that look like the synthetic efficiency setting, plus a small held-out boundary set.

**Why SWE-bench Verified?**
- Real GitHub issues with executable environments and hidden fail-to-pass tests.
- The pass@1 metric is already standardized, so the result is comparable to the literature.
- Issue descriptions give the agent a natural-language goal, unlike hand-authored stubs.
- It is much smaller and cheaper than full SWE-bench.

**Why not run all 500?** Cost and signal. We are not trying to beat the SWE-bench leaderboard; we are testing whether a *specific retrieval preference* transfers. A focused subset keeps GPU time inside a few days and makes manual inspection tractable.

**Alternative / complement:** If environment setup for SWE-bench Verified proves too heavy, fall back to **SWE-Gym Lite** (~230 instances with pre-built environments) or a **hand-curated mini-benchmark of 10–20 issues from 2–3 small/medium Python repos** (e.g. `pallets/click`, `python-attrs/attrs`, `pydantic`). The curated route gives direct control over file size and symbol complexity.

---

## 2. Task selection criteria

A task is suitable for measuring `<defn>` vs `<read>` savings when:

1. **The gold patch is small and localized.**
   - ≤3 files edited, ≤30 lines changed.
   - Prefer single-file edits for the first experiment; multi-file blast-radius is a separate variable.

2. **The fix depends on understanding one external symbol.**
   - The buggy file imports or references a class/function from a *large* non-editable module.
   - The issue or failing test names the symbol, or the failing traceback points to it.

3. **There is a material cost gap.**
   - The defining file is at least ~200 lines, preferably 500+.
   - `<defn sym>` returns ≤10% of the file's tokens.

4. **The symbol is unambiguously resolvable by a Python LSP.**
   - Avoid tasks where the needed behavior is hidden behind dynamic dispatch, `getattr`, star imports, or C extensions.

5. **Baseline solvability.**
   - The untrained 7B agent should solve at least some seeds with `<read>` available, so we can compare tokens at matched outcome.

6. **No environment/build hacks.**
   - Skip tasks that require compiling C extensions, patching `setup.py`, or non-Python changes.

**Selection pipeline:** load the SWE-bench Verified JSON → filter by patch size → compute the largest file referenced by the failing test → keep tasks where the referenced file is large and the patch touches a call/attribute of an imported symbol → manual review the final 20.

---

## 3. Action space adaptation

The current `<defn sym="NAME"/>` only resolves top-level module names via a static AST resolver. Real repos need richer symbol syntax and a real language server.

### Symbol specification

`<defn>` should accept qualified symbols:

- `ClassName`
- `module.ClassName`
- `ClassName.method_name`
- `module.ClassName.method_name`
- top-level `function_name`

If a method name is given without a class, return `(no definition found)` to force disambiguation by the model.

### Use-site resolution

For ambiguous cases, optionally support:

```xml
<defn sym="method_name" file="src/pkg/foo.py" line="42" col="8"/>
```

The `file/line/col` attributes point to a use-site; the backend calls `textDocument/definition` at that position. This is especially useful for overloaded methods and imports that shadow names.

### Backend resolver

- **Primary:** drive a live `pyrefly lsp` daemon per task, reusing `scripts/validate_pyrefly_lsp.py::LspClient`.
- **Fallback:** static AST resolver over the checked-out repo.
- **Return format:** the full source span of the resolved definition (same shape as `goto_definition` today).
- **Sequential execution:** because pyrefly daemons deadlock under concurrency, one daemon per rollout, killed before the next task.

### `<read>`

Return a numbered, editable view of the requested file, truncated to ~16k tokens / 250 lines to avoid context overflow. The file view must be line-numbered so the existing `<edit path="..." lines="START-END">` action continues to work.

### `<findrefs>`

Keep the current `textDocument/references` wrapper, returning a list of `path:line` sites.

### Edit action

Start with line-range edits (`edit_mode="line"`). If real-repo patches turn out to need whole-function replacements, also allow `SEARCH/REPLACE/END`.

---

## 4. Coverage and boundary: how to decide "definition-sufficient"

We no longer have task labels. Use a **two-phase empirical boundary**:

1. **Harvest phase:** run every candidate task with `--force-lsp` (reads of non-editable files denied). Tasks that the untrained agent solves mostly via `<defn>` are strong candidates for definition-sufficient training demos.

2. **Post-SFT evaluation phase:** the trained model's own read decisions become the boundary signal.
   - If the trained model uses `<defn>` first and solves → task is definition-sufficient for that seed.
   - If it reads first and solves → task is a boundary success.
   - If it reads and fails, or defns and fails → unresolved.

For reporting, split results into:
- **Def-sufficient subset:** tasks where the trained policy used `<defn>` and solved.
- **Boundary subset:** tasks where the trained policy read and solved.
- **Unresolved:** everything else.

This is the same "model judges coverage per-instance" idea tested in §5.6 of `PAPER.md`, but now applied to real repos.

**Optional helper:** train a lightweight coverage classifier from the harvest trajectories. Features: issue text, failing-test names, first `<defn>` result length, whether the defn span references further unresolved names. Use it only to stratify analysis, not to gate training.

---

## 5. Metrics and controls

### Conditions

1. **Default 7B** – no adapter, no extra prompt.
2. **Explicit-prompt 7B** – append the `preferlsp` steer hint used in `scripts/synth_mf.py`.
3. **Trained LoRA 7B** – on-policy relabel SFT.
4. *(Optional)* **Read-only trained 7B** – SFT on read-first trajectories to isolate the cost preference from retrieval itself.

### Metrics per rollout

- `pass@1` from the repo's test command.
- `<defn>` use rate (% of rollouts with ≥1 real `<defn>` that found a definition).
- `<read>` count and rate.
- Input tokens, output tokens, total tokens.
- Turns and edit count.
- Rework ratio and n_edits from the env.

### Aggregated comparisons

- **Matched-outcome token comparison:** restrict to tasks both the base and trained policy solve; compare mean input tokens.
- **McNemar exact test** on pass@1.
- **Paired sign test** on input tokens.
- Report by subgroup (definition-sufficient vs boundary).

---

## 6. Data collection and training

### Harvesting on real-repo tasks without labels

Use the same on-policy relabel recipe as `scripts/run_relabel2.sh`, but split the harvest into two modes:

**Mode A – force_lsp (cheap-action demos):**
- Run with `--force-lsp --relabel --save-sft`.
- Reads of non-editable files are denied; the model is redirected to emit `<defn>` itself.
- Keep only resolved trajectories where a real `<defn>` returned a found definition.
- This produces the "definition-sufficient" half of the training mix.

**Mode B – reads allowed (boundary demos):**
- Run the same tasks without `--force-lsp`, with `--save-sft`.
- Keep resolved trajectories where the model used `<read>` first (or at all) on a large file.
- This produces the "read-when-needed" half of the training mix.

The combined set is fed to `scripts/sft_lora.py`. The existing `is_clean_teacher()` filter already does the right thing: it keeps resolved trajectories with a real `<defn>`/`<findrefs>` hit or a lead action.

### Identifying the "needed symbol"

We do **not** need a gold needed-symbol oracle. The relabel mechanism lets the model pick the symbol it was looking for. For analysis only, extract the first `<defn>` call from the resolved trajectory; that symbol is the model's guess at the needed API.

### Training hyperparameters

Reuse the headline recipe:
- Model: `Qwen/Qwen2.5-Coder-7B-Instruct`.
- LoRA rank 16, alpha 32, dropout 0.05.
- LR `1e-4`, 3 epochs, micro-batch 1, grad accumulation 8.
- `max_len=4096`.

---

## 7. Expected sample size and cost

| Stage | Tasks | Seeds/condition | Rollouts | Notes |
|-------|-------|-----------------|----------|-------|
| Task selection / dry-run | 50–100 filtered | 1 | 50–100 | No model load; just resolver smoke tests |
| Mode A harvest | 15–20 | 4–8 | 60–160 | `--force-lsp --relabel --save-sft` |
| Mode B harvest | 15–20 | 4–8 | 60–160 | reads allowed |
| LoRA SFT | — | — | — | ~30–60 min on DGX Spark |
| Retest base / prompt / trained | 20 | 4 | 240 total | 3 conditions |
| Scale check (optional) | subset 8–10 | 2 | ~60 | 27B |

**GPU time:** roughly **1–2 GPU-days** on the reported DGX Spark (GB10, 128 GB unified memory) for the full 7B arm. The optional 27B check adds another ~1 GPU-day.

**Token cost:** If running locally, the cost is electricity/GPU time. If using an API, estimate ~$100–$300 for ~20–30M tokens total, depending on rollout length.

---

## 8. Risks and fallback

| Risk | Why it matters | Fallback / interpretation |
|------|----------------|---------------------------|
| **Ambiguous imports / re-exports** | `<defn>` may land on the wrong file or return a re-export stub. | Use use-site `file/line/col`; if LSP still misses, agent falls back to `<read>`. |
| **Methods and dynamic dispatch** | A bare method name is ambiguous; real behavior may be in subclasses. | Require qualified `Class.method`; report resolution rate separately. |
| **Needing multiple symbols** | One `<read>` may be cheaper than several `<defn>` calls. | Track token spend; the boundary analysis will show when reading is rational. |
| **LSP / pyrefly failures on real repos** | Real repos may not pyrefly-parse cleanly. | Use AST fallback; skip tasks where both fail consistently. |
| **Multi-file / non-local patches** | The agent's line-edit action may be too weak. | Allow `SEARCH/REPLACE/END` and multi-file `SYS_LINE_MULTI`. |
| **Small cost gap** | Many real files are not large enough. | Report per-task savings; do not pool tasks with no gap. |
| **Trained policy under-reads and fails** | The cheap preference could hurt success. | Compare pass@1; if it drops, the boundary is not being learned. |

**How to interpret a null:** If `<defn>` use does not rise, or token savings disappear, the result is still valuable: it shows that a clean synthetic cost gap does not automatically transfer to messy real repositories. We would conclude that the preference must be trained *in* a real-repo distribution, not just on synthetic analogues, and that real-world indirection is the limiting factor identified in `PAPER.md` §7.

---

## 9. Implementation steps

1. **`scaffold/real_env.py`** – create `RealRepoEnv`.
   - Clone/checkout a task's git base commit.
   - Implement `read_file`, `apply_line_edit`, `apply_edit`, `list_files`, `run_tests`, `pyrefly_diagnostics`, `goto_definition`, `lsp_definition`, `find_references`, `metrics`, `current_patch`.
   - `run_tests` runs the repo-specific test command (pytest/unittest) and parses PASS/FAIL.
   - Reuse `LspClient` from `scripts/validate_pyrefly_lsp.py`.

2. **`scaffold/real_env.py::SymbolResolver`** – support qualified symbols and use-site resolution.
   - Parse `module.Class.method`.
   - Fall back to `textDocument/definition` when `file/line/col` are supplied.
   - Expand the LSP location to the enclosing top-level node's full span (as `lsp_definition` already does in `mock_env.py`).

3. **`scripts/real_repo_loader.py`** – task loader and filter.
   - Read SWE-bench Verified / SWE-Gym Lite JSON.
   - Filter by patch size, file count, and referenced-file size.
   - Build task dicts matching the schema expected by `synth_mf.py` (name, files dict, target, test command, gold patch).

4. **`scripts/real_mf.py`** – runner mirroring `scripts/synth_mf.py`.
   - Use `RealRepoEnv` instead of `MultiFileEnv`.
   - Keep `--force-lsp`, `--relabel`, `--save-sft`, `--steer`, `--adapter`, `--lsp-tools`, `--lsp-defn`.
   - Build prompts with the real issue description and editable-file list.

5. **`scaffold/stream_agent.py`** – extend `<defn>` parsing.
   - Update `DEFN_RE` to capture optional `file`, `line`, `col`.
   - Pass use-site to `_resolve_defn`.
   - Update system prompts to advertise qualified symbols and line-numbered views.

6. **`scripts/run_real_repo.sh`** – shell driver.
   - Stage 1: dry-run / resolver validation.
   - Stage 2: Mode A harvest.
   - Stage 3: Mode B harvest.
   - Stage 4: LoRA SFT.
   - Stage 5: retest base / prompt / trained.
   - Stage 6: run analysis.

7. **`scripts/analysis/real_repo_stats.py`** – metrics and tests.
   - Compute pass@1, `<defn>` use, input tokens, matched-outcome token reduction.
   - Run McNemar and paired sign tests.
   - Subgroup breakdown by post-hoc definition-sufficiency.

8. **(Optional) `scripts/real_coverage_classifier.py`** – lightweight boundary predictor.
   - Train on harvest trajectories to predict whether a task will need a full read.
   - Use only for stratification and diagnosis.

9. **Validation**
   - Run the resolver on 5 sample tasks and check agreement between LSP and AST.
   - End-to-end smoke test on 2 tasks with the default 7B model.
   - Confirm `sft_lora.py` filters and trains without errors on a small harvested JSON.
