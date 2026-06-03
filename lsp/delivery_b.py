#!/usr/bin/env python3
"""Condition B — instructed tool-call (single-stream, model-initiated).

§0.4 / §0.5: "Do agents use LSP when given the choice?" Diagnostics are returned
on-demand via an LSP `textDocument/diagnostic` request that the *model* issues.
Single-stream (v0.5): the payload is inserted **inline** into the token stream at
the next position after the model's request.

Insertion: inline, at offset 0 relative to the model's request position (the
block is returned at the next position when the model asks). No latency replay.
Model-initiated: True (the distinguishing property of B).

`on_edit` is intentionally a no-op for B: nothing is pushed after an edit. The
model must explicitly call `request_diagnostics` for a payload to be delivered.
"""

from __future__ import annotations

from lsp.delivery_base import (
    DeliveryDescriptor,
    DeliveryEvent,
    DeliveryLayer,
    EditEvent,
)


class DeliveryB(DeliveryLayer):
    CONDITION = "B"
    MODEL_INITIATED = True

    def on_edit(self, edit: EditEvent) -> DeliveryEvent | None:
        # B does not auto-push after an edit; diagnostics arrive only when the
        # model asks. Return None so the scaffold emits nothing here.
        return None

    def request_diagnostics(self, edit: EditEvent) -> DeliveryEvent:
        """Model-initiated diagnostic request (the `textDocument/diagnostic`
        path). Inserts the canonical payload inline at offset 0 relative to the
        request position (the next token position when the model asks)."""
        payload = self._normalize(edit)
        descriptor = DeliveryDescriptor(
            condition=self.CONDITION,
            insertion_offset_tokens=0,
            model_initiated=True,
        )
        return self._emit(DeliveryEvent(payload=payload, descriptor=descriptor))
