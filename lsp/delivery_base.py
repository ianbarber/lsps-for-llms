#!/usr/bin/env python3
"""Delivery-layer base — the shared interface for conditions B/C/D (v0.5 inline).

Per experiment_plan §0 (v0.5 Interaction-Model Pivot): the trajectory is a SINGLE
token stream. Diagnostics are inserted **inline** as a delimited block at a token
position determined by the condition (§0.4). There is no side stream, no "format"
axis, and **no C′** (it dissolved — §0.4/§0.6). All conditions share ONE pyrefly
daemon and ONE payload normalizer (`lsp.payload.normalize_payload`); they differ
ONLY in *where in the single stream* the (byte-identical) payload is inserted.

A delivery layer turns an editing event into a `DeliveryEvent`:
  payload (bytes, from the shared normalizer) + a `DeliveryDescriptor`
  {insertion_offset_tokens, model_initiated} describing the inline insertion.

Descriptor semantics (the v0.5 inline model):
- `insertion_offset_tokens`: token offset, RELATIVE TO THE EDIT BOUNDARY, at
  which the diagnostic block is spliced into the single stream.
    * C (forced sync post-edit): 0 — inserted at the edit boundary.
    * B (instructed tool-call):  0 relative to the model's *request* position —
      the block is returned at the next position when the model asks.
    * D (async interleaved):     round(measured_latency_ms / ms_per_token) —
      mid-generation of whatever the agent did next (latency replay, §0.4).
- `model_initiated`: True only for B (the model asks); False for the push
  conditions C/D.

What is REAL here (skeleton level):
- The payload path: every layer calls the *same* `normalize_payload`, so the
  byte content is identical across conditions for the same trigger (G4).
- The per-condition inline insertion offset: 0 for C, request-relative 0 for B,
  latency-replayed for D.
- D's debounce / measured-latency-offset logic.

What is STUBBED:
- The model-facing emit (`_emit`): a real implementation splices `event.payload`
  into the single token stream at `edit_pos + insertion_offset_tokens`. Here it
  just records the DeliveryEvent and returns it. G2 (the canary) exercises the
  real splice later; G4 only needs the payload + descriptor, both of which are
  real.
- Pyrefly latency for D is an injected/stub value at skeleton level (default from
  G5's measured p95); the real D measures it against the live daemon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lsp.payload import EditedRegion, normalize_payload

# G5 measured daemon round-trip p95 (1674-line file). Used as D's default
# injected latency offset at skeleton level; real D measures per-snapshot.
DEFAULT_MEASURED_LATENCY_MS = 21.3


@dataclass(frozen=True)
class DeliveryDescriptor:
    """How a payload is inserted into the single token stream (v0.5 inline).

    - insertion_offset_tokens: token offset relative to the edit boundary at
      which the diagnostic block is spliced in. 0 for the synchronous conditions
      (C, and B relative to its request position); the latency-replayed offset
      for D (asynchronous, mid-generation).
    - model_initiated: True only for B (the model asks); False for C/D (push).
    """

    condition: str
    insertion_offset_tokens: int
    model_initiated: bool


@dataclass(frozen=True)
class DeliveryEvent:
    """A normalized payload plus its delivery descriptor. The unit the audit and
    (later) the substrate consume."""

    payload: bytes
    descriptor: DeliveryDescriptor

    @property
    def sha256(self) -> str:
        import hashlib
        return hashlib.sha256(self.payload).hexdigest()


@dataclass
class EditEvent:
    """An edit produced by the scaffold: the resulting raw pyrefly diagnostics,
    the edited line region (for top-K recency ranking), and the token step at
    which the edit completed (the edit boundary, the inline anchor)."""

    raw_diagnostics: list[dict]
    edited_region: EditedRegion
    edit_step: int


class DeliveryLayer:
    """Base delivery layer. Subclasses set CONDITION / MODEL_INITIATED and
    override `_descriptor_for` (and, for D, the debounce/latency logic).

    The payload path is shared and final: `_normalize` is the single call into
    the canonical normalizer, so subclasses cannot diverge the byte content.
    """

    CONDITION: str = "base"
    MODEL_INITIATED: bool = False

    def __init__(self, top_k: int = 10) -> None:
        self.top_k = top_k
        self.emitted: list[DeliveryEvent] = []

    # -- shared payload path (identical across all conditions) --
    def _normalize(self, edit: EditEvent) -> bytes:
        return normalize_payload(
            edit.raw_diagnostics, edit.edited_region, top_k=self.top_k
        )

    # -- per-condition inline insertion offset (overridden) --
    def _descriptor_for(self, edit: EditEvent) -> DeliveryDescriptor:
        return DeliveryDescriptor(
            condition=self.CONDITION,
            insertion_offset_tokens=0,
            model_initiated=self.MODEL_INITIATED,
        )

    # -- model-facing emit (STUBBED) --
    def _emit(self, event: DeliveryEvent) -> DeliveryEvent:
        """Stub: a real implementation splices `event.payload` into the single
        token stream at `edit_pos + event.descriptor.insertion_offset_tokens`.
        Here we just record and return it. G2 exercises the real splice; G4
        needs only payload + descriptor."""
        self.emitted.append(event)
        return event

    # -- public entry points --
    def on_edit(self, edit: EditEvent) -> DeliveryEvent | None:
        """Called after every scaffold Edit. Default: synchronous inline
        insertion at the edit boundary (C). D overrides the offset; B does NOT
        push on edit — see DeliveryB."""
        payload = self._normalize(edit)
        descriptor = self._descriptor_for(edit)
        return self._emit(DeliveryEvent(payload=payload, descriptor=descriptor))

    def normalized_payload(self, edit: EditEvent) -> bytes:
        """Direct access to the canonical payload for this trigger (the bytes
        G4 hashes). Always the same call regardless of condition."""
        return self._normalize(edit)
