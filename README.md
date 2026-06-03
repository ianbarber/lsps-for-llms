# Streams

Does **live, in-stream LSP feedback** (squigglies spliced into the decode stream as a
coding agent works) beat **synchronous post-edit delivery** of the same diagnostics —
or no feedback at all?

**Headline finding so far:** *delivery hygiene dominates.* A naive live channel
significantly tanks an untrained 7B agent (fix-rate 0.345 vs 0.548 batched,
McNemar p=0.006) — but the harm decomposes into fixable delivery mistakes (an
over-eager prompt + diagnostics about the model's own half-finished edits, 78% of
traffic). Gate those out and live delivery returns to parity (0.500) with sync and
none. *How* you deliver in-loop feedback matters more than *whether* you do.

**Read [`WRITEUP.md`](./WRITEUP.md)** for the full results, statistics, mechanism,
retractions, and the practical recipe.

- `log.md` — chronological decision log (every run, result, audit, and correction).
- `experiment_plan.md` — original plan (historical; superseded in parts by the log).
- `scaffold/`, `scripts/`, `harness/` — agent, tasks, runners, real-SWE pipeline.
- `runs/agent/*.json` — per-rollout results for all conditions.

Ongoing: n=168 power-up; self-distillation SFT; richer constructive signals
(hover/go-to-def-style context). Single-box setup: Qwen2.5-Coder-7B on a GB10,
real [pyrefly](https://pyrefly.org) as the checker.
