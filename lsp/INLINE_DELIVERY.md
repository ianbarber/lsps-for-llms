# Inline delivery model (v0.5 Interaction-Model Pivot)

The v0.5 pivot (`experiment_plan.md` §0) replaces the superseded multi-stream
side-channel delivery model with a **single-stream inline-insertion** model. A
trajectory is one token stream; diagnostics are inserted **inline** as a
delimited block (e.g. `‹diag›(severity,line,code,msg)…‹/diag›`) at a token
position determined by the condition. There is no side stream and **no "format"
axis**, so the old C′ condition dissolves (§0.4/§0.6).

## Descriptor

`lsp/delivery_base.py:DeliveryDescriptor` (was `{condition, channel, timestep,
latency_offset_ms, model_initiated}`) is now:

| Field | Meaning |
|---|---|
| `condition` | "B", "C", or "D" |
| `insertion_offset_tokens` | token offset, **relative to the edit boundary**, at which the diagnostic block is spliced into the single stream |
| `model_initiated` | `True` only for B (the model asks); `False` for the push conditions C/D |

A real `_emit` splices `event.payload` into the token stream at
`edit_pos + insertion_offset_tokens`. At skeleton level `_emit` only records the
event (G4 needs payload + descriptor; G2 exercises the real splice).

## Conditions — position only

The payload bytes come from the shared `lsp.payload.normalize_payload` and are
**byte-identical** across B/C/D for the same `(prefix, edit)` trigger. Only the
insertion position differs.

| ID | Insertion | `insertion_offset_tokens` | `model_initiated` |
|---|---|---|---|
| **B** — instructed tool-call | inline at the model's request position | 0 (request-relative) | True |
| **C** — forced sync post-edit | inline at the **edit boundary** | 0 | False |
| **D** — async interleaved | inline at `edit_pos + round(latency_ms / ms_per_token)` (mid-generation) | `latency_steps` (> 0) | False |

- **C** now subsumes the old C′ role (synchronous delivery); there is no
  separate C′ and `delivery_cprime.py` is deleted.
- **D** keeps the real debounce machinery (200 ms quiet / hunk boundary, see
  `delivery_d.py:DebounceState`) and the measured-latency → token-offset replay.
  `ms_per_token` is to be re-measured on Qwen2.5-Coder before L1 (§0.9 open-q #4).

## What is unchanged

- `lsp/payload.py:normalize_payload` — the canonical normalizer (the SHA-256
  gate's invariant). Untouched.
- `lsp/pyrefly_client.py` — the daemon client. Untouched.
- The G4 fixtures (`lsp/g4_fixtures.py`) — the 10 fixed (prefix, edit) cases.

## Central comparison

**D vs C** isolates synchrony directly (format constant), a cleaner isolation
than the old C′ required. B is the use-vs-presence audit; A (no-LSP) is the floor.
