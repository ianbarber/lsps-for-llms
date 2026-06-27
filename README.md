# Streams — training a coding agent to use cheap LSP retrieval

This repo contains the code, result data, and reproduction scripts for the tech report
[PAPER.md](./PAPER.md):

> *Making a Language Server Pay Off for a Coding Agent: Train It to Retrieve Cheaply*

The report shows that a coding agent which can read files on its own does not need a
language server for *information*, but can benefit from its *retrieval efficiency* — if
the agent is trained to prefer a cheap go-to-definition over a whole-file read.

## Headline result

On a synthetic mixed task suite resolved with a real Pyrefly go-to-definition resolver, a
Qwen2.5-Coder-7B agent starts with 0% `<defn>` use and 65% success. After on-policy
supervised fine-tuning it uses `<defn>` on 100% of definition-sufficient tasks, success
rises to 100%, and mean input tokens fall from 3086 to 688 (4.5× cheaper, paired sign
test p=2.2e-4, n=48). Prompting and offline imitation do not produce this preference;
on-policy training does. The cheap action is a real LSP query, validated against a live
`pyrefly lsp` daemon.

**Actions:** `<read path>` returns the full file. `<defn sym>` returns the definition span
of `sym` via an AST resolver over the live workspace. We validated the static resolver
against `pyrefly lsp` on 12 evaluation symbols (12/12 agreement) and reproduced the
headline with the live daemon (2894→689 tokens, 58→100% success).

## The recipe

1. **Use the LSP for efficiency, not information.** Give the agent a real
   go-to-definition *action* (`<defn sym>`), not diagnostics-as-context. On our suites the
   information in diagnostics, find-references, and completions is redundant for a
   self-retrieving agent; the residual value is cheaper retrieval.

2. **Train the preference; do not prompt it.** Asking the agent to prefer `<defn>` leaves
   use near 0% on the headline suite.

3. **Train it on-policy.** Roll out the untrained agent with both `<read>` and `<defn>`
   available. Where it emits `<read>` for a non-editable file and the needed symbol is
   resolvable, rewrite that step to `<defn sym>` and keep the rest of the trajectory.
   Fine-tune on these relabeled trajectories (one DAgger round). This teaches the
   *preference* for the cheap action in states where the expensive one is also available.

4. **Preserve the boundary.** Mix in tasks that genuinely need a full read, so the agent
   learns *when* `<defn>` suffices and when it does not. On our read-required boundary
   suite the trained agent still reads 100% of the time and success rises (0.54→0.83
   with the real resolver).

The training mix uses task labels to know which tasks are definition-sufficient, but at
*test* time the trained model judges coverage itself: §5.6 of `PAPER.md` shows a trained
27B reads only when the retrieved definition is genuinely insufficient, generalizing to
an indirection mechanism it never saw during training. A practitioner does not need a
perfect coverage oracle at inference.

### Install

```bash
pip install -e .
# or, if you prefer a requirements-style workflow:
# pip install -r requirements.txt
```

The scripts use Pyrefly for static analysis. The default path is
`.venv-streams/bin/pyrefly` under the repo root. If your Pyrefly binary lives
elsewhere, set:

```bash
export STREAMS_PYREFLY=/path/to/pyrefly
```

`HF_HOME` defaults to `~/.cache/huggingface`; set the environment variable
before running to override it.

## Start here

| File | What it is |
|---|---|
| `PAPER.md` | The tech report: method, results, limitations, and what does not work. |
| `log.md` | Complete chronological lab log — every run, decision, audit, and retraction. |
| `scripts/run_relabel2.sh` | Reproduce the headline on-policy relabel experiment (harvest → SFT → retest). |
| `scripts/analysis/stats.py` | Reproduce every headline number: `python scripts/analysis/stats.py`. |

`stats.py` recomputes the full result table from the committed `runs/agent/*.json` files
and checks each figure against `PAPER.md`. The `scripts/run_*.sh` drivers regenerate those
JSONs from scratch.

## Core code

- `scaffold/` — the non-blocking continuous-stream coding agent (`stream_agent.py`) and
  the task environment with real Pyrefly (`mock_env.py`, including the `<defn>` resolver).
- `scripts/synth_tasks_effic.py` — the definition-sufficient efficiency suite.
- `scripts/synth_tasks_efficread.py` — the read-required boundary tasks.
- `scripts/synth_mf.py` — the condition runner (rollouts, harvest, retest).
- `scripts/sft_lora.py` — the on-policy LoRA-SFT trainer.
- `scripts/validate_pyrefly_lsp.py` — validates `<defn>` against a live `pyrefly lsp` daemon.
- `scripts/grpo_cost.py` + `run_grpo.sh` — cost-reward GRPO corroboration (optional; SFT
  relabel is the headline).

## Layout

- `runs/agent/` — committed result JSONs backing `PAPER.md`.
- `runs/sft/` — trained LoRA adapters (git-ignored binaries).
- `docs/` — paper figures and the efficiency bibliography (`docs/bibliography_efficiency.bib`).
- `bibliography.md` — human-readable bibliography.

Hardware for the reported runs: single NVIDIA DGX Spark (GB10, 128 GB unified memory).
