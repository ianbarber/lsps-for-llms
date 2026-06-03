#!/usr/bin/env python3
"""Canonical diagnostic-payload normalizer — the load-bearing shared component.

Per experiment_plan §7.1: the diagnostic payload delivered to the model in
conditions B/C/C'/D must be **byte-identical** for the same (prefix, edit)
trigger. Information content must not drift between conditions, or the entire
"matched information content" claim (RQ1, §13 threat "Information-content drift")
collapses. G4 (§11.1) audits exactly this with a SHA-256 over the bytes this
module emits.

`normalize_payload(raw_diagnostics, edited_region) -> bytes` is the single
chokepoint every delivery layer calls. It:

1. Projects each raw pyrefly diagnostic to the canonical 4-field tuple
   `(severity, line, code, message)`. All other fields — notably pyrefly's
   non-spec `data: "committing-transaction"` (G5), `codeDescription`, `source`,
   `range.end`, `relatedInformation` — are dropped.
2. Selects **top-K=10 by recency-of-edited-region**: diagnostics whose line is
   closest to the edited region rank first (distance 0 = inside the region).
   Ties broken deterministically by (line, severity, code, message) so the
   result is a total order independent of pyrefly's emission order.
3. Serializes deterministically (canonical JSON: sorted keys, no whitespace
   ambiguity, UTF-8) so equal logical payloads produce equal bytes.

Determinism requirements that make G4 meaningful:
- Output must not depend on the *order* pyrefly emits diagnostics in.
- Output must not depend on *which condition* requested it (B/C/C'/D pass the
  same raw diagnostics + edited_region and get the same bytes).
- 1-indexed line numbers in the canonical tuple (LSP ranges are 0-indexed;
  we add 1 once, here, so downstream display is human-facing and consistent).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

DEFAULT_TOP_K = 10

# LSP DiagnosticSeverity -> stable lowercase name for the canonical tuple.
# (1=Error, 2=Warning, 3=Information, 4=Hint per the LSP spec.)
_SEVERITY_NAME = {1: "error", 2: "warning", 3: "information", 4: "hint"}


@dataclass(frozen=True)
class EditedRegion:
    """The line span (1-indexed, inclusive) most recently edited, used to rank
    diagnostics by recency-of-edited-region. `start_line == end_line` for a
    single-line edit. If unknown, pass `EditedRegion(0, 0)` (or None to the
    normalizer) and ranking falls back to top-of-file order."""

    start_line: int
    end_line: int

    def distance_to(self, line: int) -> int:
        """0 if `line` is inside the region; otherwise lines to the nearer edge.
        Both `line` and the region bounds are 1-indexed."""
        if self.start_line <= line <= self.end_line:
            return 0
        if line < self.start_line:
            return self.start_line - line
        return line - self.end_line


def _canonical_tuple(diag: dict) -> dict[str, Any]:
    """Project one raw pyrefly LSP diagnostic to the canonical 4-field record.

    Drops everything that is not (severity, line, code, message) — including the
    non-spec `data: "committing-transaction"` field. `line` is converted from
    the LSP 0-indexed range start to a 1-indexed line number.
    """
    sev_raw = diag.get("severity", 1)
    severity = _SEVERITY_NAME.get(sev_raw, "error")
    line0 = (diag.get("range", {}) or {}).get("start", {}).get("line", 0)
    line = int(line0) + 1  # 0-indexed LSP -> 1-indexed canonical
    code = diag.get("code", "")
    if code is None:
        code = ""
    message = diag.get("message", "")
    return {
        "severity": str(severity),
        "line": int(line),
        "code": str(code),
        "message": str(message),
    }


def _sort_key(rec: dict[str, Any], region: EditedRegion | None):
    """Total-ordering key: nearest-to-edited-region first, then a deterministic
    tiebreak that does NOT depend on pyrefly's emission order."""
    dist = region.distance_to(rec["line"]) if region is not None else rec["line"]
    return (dist, rec["line"], rec["severity"], rec["code"], rec["message"])


def normalize_diagnostics(
    raw_diagnostics: Iterable[dict],
    edited_region: EditedRegion | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """Canonical list-of-records form (pre-serialization). Exposed for tests and
    for delivery layers that want to inspect the payload before emit."""
    records = [_canonical_tuple(d) for d in raw_diagnostics]
    records.sort(key=lambda r: _sort_key(r, edited_region))
    return records[:top_k]


def normalize_payload(
    raw_diagnostics: Iterable[dict],
    edited_region: EditedRegion | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> bytes:
    """Canonical, byte-identical serialization of the diagnostic payload.

    This is the function G4 audits: for one (prefix, edit) trigger, the bytes
    returned here must be identical across conditions B/C/C'/D. The serialization
    is canonical JSON (sorted keys, compact separators, UTF-8, ensure_ascii=False
    so message text round-trips stably) over the ranked top-K records.
    """
    records = normalize_diagnostics(raw_diagnostics, edited_region, top_k)
    return json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
