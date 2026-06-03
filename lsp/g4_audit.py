#!/usr/bin/env python3
"""G4 audit driver — payload-equivalence (SHA-256) across B/C/D (v0.5 inline).

Per experiment_plan §0 (v0.5 pivot) / §11.1 G4: for each of the 10 fixed
(prefix, edit) cases, drive a real pyrefly daemon prefix → edit, capture the raw
diagnostics, route them through all three v0.5 delivery layers' canonical payload
path, and assert SHA-256(payload) is identical across B/C/D. C′ is removed (it
dissolved under single-stream — §0.4/§0.6); only the *inline insertion position*
differs across B/C/D, never the payload bytes.

This is the hard L0 gate: information content must not drift between conditions.

The driver is shared by `scripts/g4_payload_audit.py` (artifact-producing run)
and `tests/test_payload_equivalence.py` (regression). It owns a per-run temp
pyrefly workspace so it needs no external repo or venv setup.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lsp.delivery_b import DeliveryB
from lsp.delivery_c import DeliveryC
from lsp.delivery_d import DeliveryD
from lsp.delivery_base import EditEvent
from lsp.g4_fixtures import CASES, G4Case
from lsp.pyrefly_client import DEFAULT_PYREFLY, PyreflyDaemon

CONDITIONS = ["B", "C", "D"]


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _layer_payload(condition: str, edit_event: EditEvent) -> bytes:
    """Run one EditEvent through the named condition's delivery layer and return
    the normalized payload bytes (the audited artifact)."""
    if condition == "B":
        layer: Any = DeliveryB()
        ev = layer.request_diagnostics(edit_event)
        return ev.payload
    if condition == "C":
        layer = DeliveryC()
        return layer.on_edit(edit_event).payload
    if condition == "D":
        layer = DeliveryD()
        return layer.on_snapshot(edit_event).payload
    raise ValueError(f"unknown condition {condition!r}")


@dataclass
class CaseResult:
    name: str
    note: str
    diag_count: int
    shas: dict[str, str] = field(default_factory=dict)
    match: bool = False
    payload_preview: str = ""


@dataclass
class AuditResult:
    pyrefly_version: str
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def all_match(self) -> bool:
        return bool(self.cases) and all(c.match for c in self.cases)


def _pyrefly_version(pyrefly: str) -> str:
    try:
        out = subprocess.run([pyrefly, "--version"], capture_output=True,
                             text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def run_audit(pyrefly: str = DEFAULT_PYREFLY,
              workdir: str | None = None) -> AuditResult:
    """Drive all 10 cases through a real pyrefly daemon and all four delivery
    layers; return per-case SHA-256s and match flags."""
    result = AuditResult(pyrefly_version=_pyrefly_version(pyrefly))

    own_workdir = workdir is None
    root = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="g4_"))
    root.mkdir(parents=True, exist_ok=True)
    # A pyrefly config so the daemon type-checks our fixtures (the `basic` preset
    # without a config produces no errors — confirmed during probing).
    subprocess.run([pyrefly, "init"], cwd=str(root),
                   capture_output=True, text=True)

    daemon: PyreflyDaemon | None = None
    try:
        daemon = PyreflyDaemon(str(root), pyrefly=pyrefly)
        for case in CASES:
            cr = _run_case(daemon, root, case)
            result.cases.append(cr)
    finally:
        if daemon is not None:
            daemon.close()
        if own_workdir:
            shutil.rmtree(root, ignore_errors=True)
    return result


def _run_case(daemon: PyreflyDaemon, root: Path, case: G4Case) -> CaseResult:
    # One file per case so the daemon's per-document state is isolated.
    target = root / f"{case.name}.py"
    target.write_text(case.prefix)
    daemon.open(str(target), text=case.prefix)
    raw = daemon.change(str(target), case.edit)

    edit_event = EditEvent(
        raw_diagnostics=raw,
        edited_region=case.edited_region,
        edit_step=100,  # arbitrary fixed step; same across conditions
    )

    shas: dict[str, str] = {}
    payloads: dict[str, bytes] = {}
    for cond in CONDITIONS:
        # B/D insert at a different inline *offset*, but the payload bytes (what
        # G4 audits) come from the shared normalizer and must match.
        payloads[cond] = _layer_payload(cond, edit_event)
        shas[cond] = _sha(payloads[cond])

    match = len(set(shas.values())) == 1
    preview = payloads["C"].decode("utf-8", errors="replace")
    if len(preview) > 240:
        preview = preview[:240] + "…"

    return CaseResult(
        name=case.name,
        note=case.note,
        diag_count=len(raw),
        shas=shas,
        match=match,
        payload_preview=preview,
    )
