# G3 — Causal-Validity Gate (interleaved single-stream layout, v0.5 §0.4)

**Gate:** PASS (10/10 trajectories pass all assertions; collision check PASS)

- Layout: single interleaved token stream; diagnostic block delimited by `‹diag›` … `‹/diag›`, spliced inline at `query_pos + latency`.
- Condition C replaces the old C′ + C (single-stream removes the format axis); C = offset 0 (edit boundary), D = latency-shifted.
- Fixtures: `tests/fixtures/teacher_trajectories`
- Tokenizer rate: mock (teacher=student=4.0 chars/token, scale=1.0) (scale = student tok / teacher tok = 1.000)

## Assertions
- **(a)** no teacher sync-diagnostic content remains in D's main (agent) token stream
- **(b)** the diagnostic block is spliced inline (properly delimited), latency-shifted
- **(c)** spliced position is >= the query (latency >= 0) and == query + latency
- **(d)** C: block at the edit boundary (offset 0), sync original absent from main stream
- **(adv)** a deliberately naive (leaky) reformat IS flagged by (a) — proves non-vacuity
- **(col)** two diagnostics latency-shifted onto the same position stack in order

## Per-trajectory

| traj | n_diag | (a) | (b) | (c) | (d) | (adv) | all |
|---|---|---|---|---|---|---|---|
| t00 | 1 | PASS | PASS | PASS | PASS | PASS | PASS |
| t01 | 2 | PASS | PASS | PASS | PASS | PASS | PASS |
| t02 | 0 | PASS | PASS | PASS | PASS | n/a | PASS |
| t03 | 3 | PASS | PASS | PASS | PASS | PASS | PASS |
| t04 | 1 | PASS | PASS | PASS | PASS | PASS | PASS |
| t05 | 1 | PASS | PASS | PASS | PASS | PASS | PASS |
| t06 | 2 | PASS | PASS | PASS | PASS | PASS | PASS |
| t07 | 3 | PASS | PASS | PASS | PASS | PASS | PASS |
| t08 | 1 | PASS | PASS | PASS | PASS | PASS | PASS |
| t09 | 1 | PASS | PASS | PASS | PASS | PASS | PASS |

## Collision check

- collided on same position: True
- earlier-querying block spliced first: True
- blocks do not overlap: True
- both diagnostics present (none dropped): True
- **collision check: PASS**

## Interpretation

Assertion (a) is the load-bearing causal-validity check: it proves D's interleaved
training stream does not contain the teacher's synchronous inline diagnostic, so the
D student never sees both the sync and the latency-replayed async copy. The (adv)
column confirms the check is non-vacuous — a naive reformat that leaves the sync
block inline in the main stream is detected. n/a in (adv) marks clean-snapshot
trajectories with no diagnostics to leak. The collision check guards the original-G3
bug, re-cast for interleaving: two diagnostics latency-shifted onto the same position
are stacked in arrival order (later-querying block AFTER the earlier), not overwritten.
