# Interleaved single-stream layout (v0.5 Interaction-Model Pivot, §0.4)

This documents the layout produced by `training/reformat.py` after the v0.5 pivot from
the multi-stream grid (`MultiStreamSequence`, now retired/provenance-only) to a single
**interleaved token stream** (`InterleavedSequence`).

## Why
v0.5 operationalizes "in-stream feedback" as single-stream interleaved-async tokens on a
real coder (Qwen2.5-Coder), following Hooper (2026) and Thinking Machines "interaction
models" (2026). Single-stream interleaving removes the multi-stream **format** axis, so
the old C′ dissolves: **C (sync inline) vs D (async inline)** isolates synchrony directly.
The latency-replay protocol and its causal-validity gate (G3) carry over unchanged in
*method*; only the output *layout* changes.

## Representation
`InterleavedSequence` is one ordered list of `InterleavedToken`:

- `pos` — contiguous position in the final interleaved stream (student-token time).
- `text` — the token text.
- `source` — `"agent"` or `"diagnostic"`.
- `meta` — for diagnostic tokens: `query_id`, `query_student_idx`, `latency_student`,
  `requested_pos`, `block_line`.

It also carries `diag_spans: list[DiagBlockSpan]` recording, for each spliced block, its
`[start, end)` token range plus the provenance (`query_student_idx`, `latency_student`,
`requested_pos`) used to place it.

## Diagnostic block + delimiter
A diagnostic snapshot becomes one **delimited inline block** spliced between agent tokens:

```
‹diag›
[error] L1 bad-return-type: expected int
[warning] L1 empty-class: C is empty
‹/diag›
```

- Open sentinel: `‹diag›`  (`DIAG_OPEN`)
- Close sentinel: `‹/diag›` (`DIAG_CLOSE`)
- Inner lines: one `Diagnostic.render()` per diagnostic — the exact `[sev] L{line}
  {code}: {msg}` content the old grid layout placed on the side channel, so payloads stay
  byte-comparable with G4 and the old runs (`render_diag_block`).

The sentinels are the guillemet form from §0.4. They are kept as plain strings so the
layout is tokenizer-agnostic; real SFT may map them to dedicated Qwen2.5-Coder special
tokens. The block is stored as multiple tokens (open sentinel / each diag line / close
sentinel) so it occupies a real span of stream positions.

## Splice position (the latency-replay core)
For each snapshot with a linked `(lsp_query, lsp_response)`:

1. `query_student_idx = TokenizerRate.teacher_to_student(query.t_emit)` — the edit/query
   position in **student-coder-tokenizer** time (step-4 tokenizer-rate conversion).
2. `latency_student = teacher_to_student(latency_sampler(rng))` — the sampled pyrefly
   latency, in teacher-token units (`EmpiricalLatencySampler`, pluggable; real values are
   daemon round-trips ÷ `ms_per_token`), converted to student time.
3. `requested_pos = query_student_idx + latency_student`.
4. The block is spliced **immediately before the first agent token whose `pos >=
   requested_pos`** — i.e. mid-stream, some tokens *after* the triggering edit.

- **D (async):** `latency_student >= 0` — the block lands after the edit the agent has
  already moved past.
- **C (sync):** `latency_student == 0` — the block lands at the **edit boundary** (offset
  0), immediately after the edit. C replaces both the old C′ and the old C.

Blocks whose `requested_pos` exceeds the last agent position append at the tail.

## Causal-validity masking (the methodological crux — G3)
The teacher's **synchronous** inline diagnostic (`lsp_response.text`, carrying
`SYNC_DIAG_MARKER` in mocks) is **never** placed on the main stream. The only diagnostic
the student sees is the async/sync block spliced by the reformat. If the sync copy
survived, the D student would train on both the inline sync signal and the latency-shifted
copy, invalidating the D-vs-C comparison. G3 asserts the sync content is absent from the
agent stream and that the spliced block is present at the latency-shifted position.

The adversarial path (`_adversarial_leak=True`) deliberately re-injects the sync block as
agent-sourced tokens at the edit boundary; G3 must catch it (non-vacuity).

## Collision handling
Two snapshots whose `requested_pos` land on the **same** agent position are **not**
overwritten. Blocks are planned, then spliced in `(requested_pos, query_teacher_idx)`
order, so the earlier-querying snapshot's block is emitted first and the later one
immediately after it — at the same anchor, stacked in arrival order. No diagnostic is
dropped; ordering is preserved. (This is the interleaved re-cast of the original-G3
same-position collision bug, which in the grid layout was solved by `next_free`.)
