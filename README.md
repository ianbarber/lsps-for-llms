# Streams

**Does live, in-stream type-checker feedback help an LLM coding agent?**
Mostly no — and delivered naively, it actively hurts. This repo is the complete
evidence base for the preprint:

> **[PAPER.md](./PAPER.md)** — *When Squigglies Don't Help: Delivery Hygiene and the
> Limits of Live Type-Checker Feedback for Coding Agents*

**The result in one breath (n=168 paired units/condition, 14 tasks x 12 seeds,
Qwen2.5-Coder-7B + Pyrefly):** every properly-delivered feedback configuration —
eager sync, lazy sync, hygiene-gated live — and the no-feedback baseline land at
fix-rates 0.46–0.53 with no detectable difference (min pairwise p = 0.12). A naive
live channel falls to 0.345 (p = 0.0002 vs eager sync): 78% of its diagnostics
describe the model's own half-finished edits, and gating those out recovers it.
Richer diagnostics and self-distillation SFT both fail to add value (the SFT gains
are task memorization, exposed by a no-feedback control).

## Start here

| | |
|---|---|
| `PAPER.md` | the preprint (results, stats, mechanism, retractions) |
| `log.md` | the complete chronological lab log — every run, decision, audit, and retraction, unedited |
| `scripts/analysis/stats.py` | **reproduce every headline number**: `python scripts/analysis/stats.py` |

## The experiment in one table

| paper condition | invocation (scripts/synth_acd.py) | result files (seeds 0–5 / 6–11) |
|---|---|---|
| A (none) | `--conds A` | `synth_power.json[A]` / `synth_ac_s6.json[A]` |
| C-lazy (batched at pause) | `--conds C` | `synth_power.json[C]` / `synth_ac_s6.json[C]` |
| C-eager (post-edit hook) | `--conds C --c-eager` | `synth_ceager.json` / `synth_ceager_s6.json` |
| D-naive (live + announce) | `--conds D --debounce 24 --pause-align --announce-lsp` | `synth_power.json[D]` (n=84; named `D-tuned` in log.md) |
| D-plain (live) | `--conds D --debounce 24 --pause-align` | `synth_dplain.json` / `synth_dplain_s6.json` |
| D-gate (live, parse-gated) | `--conds D --debounce 24 --pause-align --syntax-gate` | `synth_dgate.json` / `synth_dgate_s6.json` |

Note: the paper's D arms all use `--debounce 24 --pause-align`; the script's bare
defaults are debounce 0 / no pause-align (the un-debounced immediate splice).
Rich-signal arms add `--rich-signal`; SFT eval adds `--adapter runs/adapters/dgate_sft_v2`.

## Layout

- `scaffold/` — the non-blocking continuous-stream agent (every delivery condition
  as a flag) and the single-file task environment with real Pyrefly.
- `scripts/` — task suite + verifier (`synth_tasks.py`), condition runner
  (`synth_acd.py`), isolation probes (`i_eval*.py`, paper §4), SFT pipeline
  (`harvest_sft.py`, `d_sft.py`), analysis (`analysis/stats.py`).
- `runs/` — committed result data: `agent/` (all condition rollouts, per-rollout
  event traces incl. delivered diagnostics), `isolation/` (§4), `rebench/`
  (Appendix A real-repo groundwork), `adapters/` (the §6.2 LoRA). See `runs/README.md`.
- `harness/` + `lsp/` — the real-SWE-rebench pipeline (paper Appendix A only).
- `docs/history/` — superseded design docs from the project's earlier eras, kept
  for provenance (`experiment_plan.md`, `WRITEUP.md`).
- `bibliography.md` — BibTeX for the paper.

Hardware: single NVIDIA DGX Spark (GB10, 128GB unified), everything local.
