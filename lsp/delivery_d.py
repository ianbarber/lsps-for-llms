#!/usr/bin/env python3
"""Condition D — async interleaved (single-stream, inline at latency-replayed pos).

§0.4 / §0.5: the hypothesis. Snapshots are *debounced* — a snapshot fires after
~200 ms of no token emission OR on a stable hunk boundary — and the resulting
diagnostic block is inserted **inline** into the single token stream at
`edit_pos + round(measured_latency / ms_per_token)`, i.e. mid-generation of
whatever the agent did next (real pyrefly latency replayed onto the token axis).

This is the only condition with non-trivial timing logic, so it carries the real
debounce + latency-offset machinery:

- Debounce: `note_token(step, t)` records token activity; `should_snapshot(t)`
  returns True after `debounce_ms` of no-token quiet OR when a hunk boundary is
  flagged. The scaffold drives these.
- Latency replay: the block is inserted at `edit_pos + latency_steps`, where
  `latency_steps = round(measured_latency_ms / ms_per_token)`. At skeleton level
  the latency is an *injected* value (default = G5's measured p95); the real D
  measures it against the live daemon per snapshot (§7.1: "Latency is measured
  against the daemon, not the CLI").

Insertion: inline at `insertion_offset_tokens = latency_steps` (> 0 → async).
Model-initiated: False.

Crucially, D still calls the SAME `normalize_payload`. Only the *insertion
offset* differs from C (which is 0); the payload bytes are identical (G4).
"""

from __future__ import annotations

from dataclasses import dataclass

from lsp.delivery_base import (
    DEFAULT_MEASURED_LATENCY_MS,
    DeliveryDescriptor,
    DeliveryEvent,
    DeliveryLayer,
    EditEvent,
)

# D's debounce window (§0.4 / §7.1: ~200 ms of no token emission).
DEFAULT_DEBOUNCE_MS = 200.0
# Nominal emit rate used to convert a wall-clock latency to a token-step offset.
# Per-trajectory step time is what matters for placing the inline block on this
# trajectory's token axis (batched aggregate throughput is irrelevant here).
# Re-measure ms_per_token on Qwen2.5-Coder before L1 (§0.9 open-q #4).
DEFAULT_MS_PER_TOKEN = 200.0


@dataclass
class DebounceState:
    """Tracks token activity for the no-token-quiet debounce trigger."""

    last_token_t: float = 0.0
    pending_since_t: float | None = None
    hunk_boundary: bool = False

    def note_token(self, t: float) -> None:
        self.last_token_t = t
        if self.pending_since_t is None:
            self.pending_since_t = t

    def flag_hunk_boundary(self) -> None:
        self.hunk_boundary = True

    def reset(self) -> None:
        self.pending_since_t = None
        self.hunk_boundary = False


class DeliveryD(DeliveryLayer):
    CONDITION = "D"
    MODEL_INITIATED = False

    def __init__(self, top_k: int = 10,
                 debounce_ms: float = DEFAULT_DEBOUNCE_MS,
                 ms_per_token: float = DEFAULT_MS_PER_TOKEN,
                 measured_latency_ms: float = DEFAULT_MEASURED_LATENCY_MS) -> None:
        super().__init__(top_k=top_k)
        self.debounce_ms = debounce_ms
        self.ms_per_token = ms_per_token
        # Injected at skeleton level; real D measures per-snapshot vs the daemon.
        self.measured_latency_ms = measured_latency_ms
        self.debounce = DebounceState()

    # -- debounce driving (called by the scaffold) --
    def note_token(self, t: float) -> None:
        self.debounce.note_token(t)

    def flag_hunk_boundary(self) -> None:
        self.debounce.flag_hunk_boundary()

    def should_snapshot(self, now_t: float) -> bool:
        """Fire after `debounce_ms` of no-token quiet OR on a stable hunk
        boundary."""
        if self.debounce.hunk_boundary:
            return True
        if self.debounce.pending_since_t is None:
            return False
        quiet_ms = (now_t - self.debounce.last_token_t) * 1000.0
        return quiet_ms >= self.debounce_ms

    # -- latency-offset placement (inline token offset relative to edit pos) --
    def _latency_steps(self) -> int:
        if self.ms_per_token <= 0:
            return 0
        return round(self.measured_latency_ms / self.ms_per_token)

    def _descriptor_for(self, edit: EditEvent) -> DeliveryDescriptor:
        return DeliveryDescriptor(
            condition=self.CONDITION,
            # inline insertion at edit_pos + latency_steps (mid-generation).
            insertion_offset_tokens=self._latency_steps(),
            model_initiated=False,
        )

    def on_snapshot(self, edit: EditEvent,
                    measured_latency_ms: float | None = None) -> DeliveryEvent:
        """Fire a debounced snapshot. Optionally override the measured latency
        (the real D passes the per-snapshot daemon-measured value). Resets the
        debounce state."""
        if measured_latency_ms is not None:
            self.measured_latency_ms = measured_latency_ms
        payload = self._normalize(edit)
        descriptor = self._descriptor_for(edit)
        self.debounce.reset()
        return self._emit(DeliveryEvent(payload=payload, descriptor=descriptor))

    # on_edit reuses the base inline insertion but with D's async (non-zero)
    # offset, so tests can drive D through the same on_edit interface as C. The
    # real runtime drives D via the debounce loop (note_token/should_snapshot/
    # on_snapshot); the payload bytes are identical either way.
