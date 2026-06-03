# lsp/

Pyrefly integration. Snapshot loop (daemon-mode, incremental), payload normalization
to `(severity, line, code, message)` tuples with top-K-by-recency selection,
debouncing (200 ms default for D), and the v0.5 single-stream inline delivery
layers (see `INLINE_DELIVERY.md`):

- **B** — instructed tool-call (inline, model-initiated; offset 0).
- **C** — forced sync post-edit (inline at the edit boundary; offset 0).
- **D** — async interleaved (inline at a latency-replayed offset > 0).

C′ is removed under v0.5 single-stream (no format axis — §0.4/§0.6). Conditions
differ only in inline insertion *position*; payload-equivalence (SHA-256) is
asserted across B/C/D for identical triggers.
