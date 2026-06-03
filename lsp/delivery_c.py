#!/usr/bin/env python3
"""Condition C — forced sync post-edit (single-stream, inline at edit boundary).

§0.4 / §0.5: production sync baseline (Claude Code-style). After *every* Edit, the
diagnostic block is inserted inline at the **edit boundary** (the next token
position, offset 0). No choice for the model. In v0.5 single-stream this also
subsumes the old C′ role (synchronous delivery) — there is no separate C′.

Insertion: inline at offset 0 (the edit boundary). Synchronous.
Model-initiated: False (forced).
"""

from __future__ import annotations

from lsp.delivery_base import (
    DeliveryDescriptor,
    DeliveryLayer,
    EditEvent,
)


class DeliveryC(DeliveryLayer):
    CONDITION = "C"
    MODEL_INITIATED = False

    def _descriptor_for(self, edit: EditEvent) -> DeliveryDescriptor:
        return DeliveryDescriptor(
            condition=self.CONDITION,
            insertion_offset_tokens=0,  # at the edit boundary, synchronous
            model_initiated=False,
        )
