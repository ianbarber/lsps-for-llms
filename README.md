# LSPs for LLMs

Code, result data, and reproduction scripts for the tech report [REPORT.md](./REPORT.md),
*Making a Language Server Pay Off for a Coding Agent: Redundant Information, Cheaper Retrieval*.

Do language servers help coding agents by supplying **information**, and do they make retrieval
**cheaper**? The win is the second: a cheap go-to-definition action saves tokens at equal success
under three conditions, and getting the agent to use it is free for capable models and trainable for
weak ones. The first is no wherever the agent can read the source and derive the fact, which held on
every channel and task we tested. That covers a lot of real code; a very large or highly complex
codebase, where the fact is not readable in the steps available, is where a language server's
information might still help, and we did not test that.

## The result

1. **Efficiency is the win, under three conditions.** A `<defn>` action cuts input tokens 3.5 to 4.7
   times at equal success, when retrieval is required, the counterfactual is a whole-file read, and
   the agent chooses the cheap action. The tool-value ablation toggles the action on the same model:

   | model | tokens with `<defn>` | tokens read-only | factor |
   |---|---|---|---|
   | 27B (real obscure suite) | 1302 | 4563 | 3.5× |
   | claude-sonnet-4.5 (tool-calling) | 6018 | 21985 | 3.7× |
   | deepseek-chat-v3.1 (tool-calling) | 7705 | 36192 | 4.7× |

2. **Election is capability-gated.** A 7B uses `<defn>` 2% by default and ignores a prompt telling
   it to prefer the action; one on-policy training round takes it to 100% use and 4.5 times fewer
   tokens. A capable model needs no training: framing `<defn>` in the system prompt as cheaper than
   a read moves an untrained 27B to 88 to 93% use and a frontier model to 100%.

3. **Information is redundant when it is readable.** When the fact a language server would supply is
   derivable from the source by reading, handing it to a self-retrieving agent does not raise pass@1,
   on every channel we tested (correction, completeness, navigation, prevention, scale, type
   inference). Both frontier models solve the held-out-scored inference test 18/18 with zero latent
   bugs, with a `check_types()` tool and without it. The same holds for execution feedback, a fact not
   in the text: on 14 small bug-fixes two frontier models score 100% held-out whether they can run the
   code, elect to run it, or are handed the result free, and only efficiency (turns) moves. We did not
   test very large or highly complex codebases, where the fact may not be readable and a language
   server may then help.

## The recipe

1. Expose go-to-definition as an action, not diagnostics as context. The information in
   diagnostics, references, and completion is redundant for a self-retrieving agent.
2. Frame it in the system prompt as cheaper than a read. Capable models then use it without training.
3. If the model is weak enough to ignore the framing, train the preference on-policy: roll out the
   agent, rewrite its `<read>` of a resolvable symbol to `<defn sym>`, and fine-tune on the relabeled
   trajectories (one DAgger round).
4. Mix in tasks that genuinely need a full read, so the agent learns when the cheap action suffices.

## Reproduce

```bash
pip install -e .                       # add '.[api]' for the frontier tool-calling path
python scripts/analysis/stats.py       # recompute the 7B training numbers from committed JSONs
```

`stats.py` recomputes every weak-model training number from the committed `runs/agent/*.json` and
checks each against `REPORT.md` (`[MATCH]` lines). `effic_real_stats.py` does the paired stats for the
real-code and tool-ablation runs. The `scripts/run_*.sh` drivers regenerate the JSONs:

| Driver | Reproduces |
|---|---|
| `run_relabel2.sh` | 7B on-policy training (harvest, SFT, retest), report §5 |
| `run_toolablation.sh` | tool-value ablation, with `<defn>` vs read-only, report §4 |
| `run_frontier.sh` | frontier election and efficiency via OpenRouter, report §4 to §5 |
| `run_gapd_frontier.sh` | the type-inference information channel, report §3 |
| `run_gapd2_frontier.sh` | the held-out-scored fair inference test, report §3 |
| `run_runtime_frontier.sh` | the execution-feedback boundary test (no-run / run / handed-over), report §3 |
| `run_relabel2_27b.sh` | 27B cross-scale transfer, Appendix B |
| `run_grpo.sh` | cost-reward RL corroboration, Appendix A |

## Test a different model or approach

`scripts/api_agent.py` is a model-agnostic tool-calling harness. Set `OPENROUTER_API_KEY` (or place
the key in a `.orkey` file at the repo root, gitignored) and point it at any OpenRouter model to
measure election and the tool-value ablation, with a hard spend cap:

```bash
# election and efficiency on the obscure real-code suite:
python scripts/api_agent.py out.json --model anthropic/claude-sonnet-4.5 \
    --suite effic_real2 --seeds 2 --budget-usd 5
# the read-only counterfactual (same, with <defn> removed):
python scripts/api_agent.py out_ro.json --model anthropic/claude-sonnet-4.5 \
    --suite effic_real2 --no-defn --seeds 2 --budget-usd 12
# the information channel (a check_types() tool available vs not):
python scripts/api_agent.py out_gd.json --model anthropic/claude-sonnet-4.5 \
    --suite gapd --with-check --seeds 2 --budget-usd 6
# the execution-feedback boundary (no-run / elect-to-run / handed-over arms):
python scripts/api_agent.py out_rt.json --model anthropic/claude-sonnet-4.5 \
    --suite runtime --no-hint --no-test --seeds 3 --budget-usd 6   # R0: drop --no-test for R1, add --auto-feedback for R2
```

The local harness (`scripts/synth_mf.py`) does the same for open-weight models, with `--no-defn`
(tool ablation) and `--adapter` (a trained policy).

## Code layout

| Path | What it is |
|---|---|
| `scaffold/stream_agent.py` | local token-stream coding agent (`<read>`/`<defn>`/`<findrefs>`/`<test>`/`<edit>`, `--no-defn` ablation) |
| `scaffold/mock_env.py` | in-memory workspace with real Pyrefly and the `<defn>` resolver |
| `scaffold/real_env.py` | `RealRepoEnv` over a checked-out git repo, with qualified `module.Class.method` resolution |
| `scripts/api_agent.py` | OpenRouter tool-calling harness (test any frontier model) |
| `scripts/synth_tasks_effic.py` | synthetic definition-sufficient efficiency suite |
| `scripts/synth_tasks_efficread.py` | read-required boundary suite |
| `scripts/synth_tasks_effic_real{,2}.py` | real vendored-library suites (`effic_real_vendor/`) |
| `scripts/synth_tasks_gapd.py`, `scripts/synth_tasks_gapd2.py` | type-inference suite (gapd2 adds held-out scoring) |
| `scripts/synth_tasks_runtime.py` | execution-feedback boundary suite (structural, easy, and semantic-trap tiers) |
| `scripts/synth_mf.py`, `scripts/real_mf.py` | local condition runners |
| `scripts/sft_lora.py` | on-policy LoRA-SFT trainer |
| `scripts/validate_pyrefly_lsp.py` | validates `<defn>` against a live `pyrefly lsp` daemon |
| `scripts/grpo_cost.py` | cost-reward GRPO corroboration (optional) |

## Layout and environment

`runs/agent/` holds the committed result JSONs; `runs/sft/` holds trained LoRA adapters (gitignored
binaries); `log.md` is the full chronological lab log; `docs/` has figures and the bibliography.
Pyrefly is the static analyzer, default path `.venv-streams/bin/pyrefly`, overridable with
`STREAMS_PYREFLY`. Hardware for the local runs: a single NVIDIA DGX Spark (GB10, 128 GB unified
memory).
