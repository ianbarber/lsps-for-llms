# G3 — Causal-Validity Gate

**Gate:** PASS (10/10 trajectories pass all assertions)

- Fixtures: `tests/fixtures/teacher_trajectories`
- Tokenizer rate: mock (teacher=student=4.0 chars/token, scale=1.0) (scale = student tok / teacher tok = 1.000)

## Assertions
- **(a)** no teacher sync-diagnostic content remains in D's Output/prefix stream
- **(b)** the diagnostic appears on the Analytical side stream, latency-shifted
- **(c)** side-stream position is >= the query (latency >= 0) and matches the sampled offset
- **(d)** C′: diagnostic on side stream at the snapshot timestep (no shift), absent from Output
- **(adv)** a deliberately naive (leaky) reformat IS flagged by (a) — proves the test is not vacuous

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

## Interpretation

Assertion (a) is the load-bearing causal-validity check: it proves D's training
prefix does not contain the teacher's synchronous inline diagnostic, so the D
student never sees both the sync and the latency-replayed async copy. The (adv)
column confirms the check is non-vacuous — a naive reformat that leaks the sync
diagnostic is detected. n/a in (adv) marks clean-snapshot trajectories with no
diagnostics to leak.
