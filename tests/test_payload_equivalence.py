#!/usr/bin/env python3
"""G4 regression — payload-equivalence (SHA-256) across B/C/D (v0.5 inline).

Pytest form of the G4 gate (experiment_plan §0 / §11.1), re-runnable at L1/L3 per
§7.1's "CI check asserts equality across conditions for the same trigger". v0.5
single-stream: C′ removed; conditions differ only in inline insertion *position*,
never payload bytes. Two tiers:

1. `test_payload_equivalence_real_pyrefly` — the full G4 gate against a live
   pyrefly daemon. Skipped automatically if the pyrefly binary is absent.
2. Fast unit checks on `normalize_payload` (no daemon) — order-invariance,
   committing-transaction stripping, top-K, recency ranking, determinism — so
   the canonicalization contract is guarded even where pyrefly is unavailable.

Run:
    PYTHONPATH=/home/ianbarber/Projects/Streams \
        .venv-streams/bin/python -m pytest tests/test_payload_equivalence.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lsp.delivery_b import DeliveryB  # noqa: E402
from lsp.delivery_c import DeliveryC  # noqa: E402
from lsp.delivery_d import DeliveryD  # noqa: E402
from lsp.delivery_base import EditEvent  # noqa: E402
from lsp.payload import (  # noqa: E402
    EditedRegion,
    normalize_diagnostics,
    normalize_payload,
)
from lsp.pyrefly_client import DEFAULT_PYREFLY  # noqa: E402


def _raw(line: int, code: str, msg: str, severity: int = 1) -> dict:
    """Build a raw pyrefly-shaped diagnostic, including the non-spec `data`
    field, so stripping is exercised."""
    return {
        "code": code,
        "data": "committing-transaction",
        "codeDescription": {"href": f"https://pyrefly.org/{code}"},
        "source": "Pyrefly",
        "message": msg,
        "severity": severity,
        "range": {"start": {"line": line, "character": 0},
                  "end": {"line": line, "character": 5}},
    }


# --------------------------- fast unit checks (no daemon) ---------------------


def test_committing_transaction_stripped():
    payload = normalize_payload([_raw(0, "x", "m")], EditedRegion(1, 1))
    assert b"committing-transaction" not in payload
    assert b"codeDescription" not in payload
    assert b"Pyrefly" not in payload


def test_order_invariance():
    a = _raw(2, "code-a", "msg a")
    b = _raw(5, "code-b", "msg b")
    region = EditedRegion(3, 3)
    p1 = normalize_payload([a, b], region)
    p2 = normalize_payload([b, a], region)
    assert p1 == p2, "payload must not depend on pyrefly emission order"


def test_recency_ranking():
    # Diagnostic on the edited line (3) must rank before a far one (line 20).
    near = _raw(2, "near", "near")   # line 3 (1-indexed)
    far = _raw(19, "far", "far")     # line 20
    recs = normalize_diagnostics([far, near], EditedRegion(3, 3))
    assert recs[0]["code"] == "near"
    assert recs[1]["code"] == "far"


def test_topk_truncation():
    raws = [_raw(i, f"c{i}", f"m{i}") for i in range(25)]
    recs = normalize_diagnostics(raws, EditedRegion(1, 1), top_k=10)
    assert len(recs) == 10


def test_severity_and_line_canonicalization():
    recs = normalize_diagnostics([_raw(0, "x", "m", severity=2)],
                                 EditedRegion(1, 1))
    assert recs[0]["severity"] == "warning"
    assert recs[0]["line"] == 1  # 0-indexed LSP -> 1-indexed canonical


def test_empty_diagnostics_empty_payload():
    assert normalize_payload([], EditedRegion(1, 1)) == b"[]"


def test_all_layers_same_payload_offline():
    """The three v0.5 delivery layers must produce identical payload bytes for
    the same EditEvent (the core G4 invariant, exercised without a daemon)."""
    raws = [_raw(4, "bad-assignment", "not assignable"),
            _raw(2, "bad-argument-type", "wrong arg")]
    edit = EditEvent(raw_diagnostics=raws,
                     edited_region=EditedRegion(3, 3), edit_step=100)
    pb = DeliveryB().request_diagnostics(edit).payload
    pc = DeliveryC().on_edit(edit).payload
    pd = DeliveryD().on_snapshot(edit).payload
    assert pb == pc == pd


def test_descriptors_differ_but_payload_matches():
    """Inline insertion *position* differs across conditions; payload bytes do
    not (v0.5 single-stream — there is no channel/format axis)."""
    edit = EditEvent(raw_diagnostics=[_raw(1, "x", "m")],
                     edited_region=EditedRegion(2, 2), edit_step=50)
    evb = DeliveryB().request_diagnostics(edit)
    evc = DeliveryC().on_edit(edit)
    # Drive D with a latency that spans several tokens so the async offset is
    # observably non-zero. (With the G5 default p95 of ~21 ms at ~200 ms/token
    # the offset rounds to 0 — sub-token latency; the real D re-measures
    # ms_per_token on Qwen2.5-Coder, §0.9 open-q #4.)
    evd = DeliveryD(ms_per_token=50.0, measured_latency_ms=300.0).on_snapshot(edit)
    # C inserts synchronously at the edit boundary (offset 0); D inserts at a
    # latency-replayed offset (> 0). This is the synchrony isolation.
    assert evc.descriptor.insertion_offset_tokens == 0
    assert evd.descriptor.insertion_offset_tokens > 0
    # B is the only model-initiated condition (and is request-relative offset 0).
    assert evb.descriptor.model_initiated is True
    assert evc.descriptor.model_initiated is False
    assert evd.descriptor.model_initiated is False
    # Payload identical regardless.
    assert evb.payload == evc.payload == evd.payload


def test_b_does_not_push_on_edit():
    edit = EditEvent(raw_diagnostics=[_raw(1, "x", "m")],
                     edited_region=EditedRegion(2, 2), edit_step=1)
    assert DeliveryB().on_edit(edit) is None  # model-initiated only


# --------------------------- the full G4 gate (real pyrefly) ------------------


_pyrefly_missing = not os.path.exists(DEFAULT_PYREFLY)


@pytest.mark.skipif(_pyrefly_missing,
                    reason=f"pyrefly binary not found at {DEFAULT_PYREFLY}")
def test_payload_equivalence_real_pyrefly():
    """G4 gate: 10 fixed (prefix, edit) cases through a real pyrefly daemon;
    SHA-256(payload) identical across B/C/D for every trigger (v0.5 inline)."""
    from lsp.g4_audit import run_audit

    result = run_audit()
    assert len(result.cases) == 10, "G4 requires exactly 10 cases"
    failures = [c.name for c in result.cases if not c.match]
    assert not failures, f"payload SHA-256 drift in cases: {failures}"
    assert result.all_match
    # Guard against a vacuous pass (daemon silently returning no diagnostics for
    # everything, making payloads trivially equal): the fixtures are designed to
    # produce real type errors, so most cases must carry diagnostics.
    with_diags = sum(1 for c in result.cases if c.diag_count > 0)
    assert with_diags >= 8, (
        f"only {with_diags}/10 cases produced diagnostics — daemon may not be "
        "type-checking the fixtures; G4 pass would be vacuous"
    )
    # And the payloads must not be uniformly empty.
    nonempty_preview = [c for c in result.cases if c.payload_preview != "[]"]
    assert nonempty_preview, "all payloads empty — vacuous equivalence"
