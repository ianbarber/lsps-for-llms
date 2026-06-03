#!/usr/bin/env python3
"""G4 — payload-equivalence (SHA-256) audit runner (v0.5 inline model).

Drives the 10 fixed (prefix, edit) cases (lsp/g4_fixtures.py) through a real
pyrefly daemon and the three v0.5 delivery layers (B/C/D), then asserts the
normalized diagnostic payload is byte-identical across conditions for each
trigger (experiment_plan §0 / §11.1 G4). C′ is removed under single-stream; only
the inline insertion *position* differs across B/C/D, never the payload bytes.

Writes:
    runs/g4_inline/audit.json   per-case SHA per condition + match + preview
    runs/g4_inline/summary.md   top-line PASS/FAIL + per-case table

Usage:
    PYTHONPATH=/home/ianbarber/Projects/Streams \
        .venv-streams/bin/python scripts/g4_payload_audit.py \
        [--pyrefly /path/to/pyrefly] [--output runs/g4_inline]

Exit code 0 on PASS (all cases match), 1 on FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a plain script (python scripts/g4_payload_audit.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lsp.g4_audit import CONDITIONS, run_audit  # noqa: E402
from lsp.pyrefly_client import DEFAULT_PYREFLY  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pyrefly", default=DEFAULT_PYREFLY)
    p.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent.parent / "runs" / "g4_inline"),
    )
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_audit(pyrefly=args.pyrefly)

    audit = {
        "gate": "G4",
        "model": "v0.5 single-stream inline insertion",
        "description": "payload-equivalence SHA-256 audit across B/C/D",
        "conditions_removed": ["C'"],
        "pyrefly_version": result.pyrefly_version,
        "conditions": CONDITIONS,
        "all_match": result.all_match,
        "n_cases": len(result.cases),
        "cases": [
            {
                "name": c.name,
                "note": c.note,
                "diag_count": c.diag_count,
                "shas": c.shas,
                "match": c.match,
                "payload_preview": c.payload_preview,
            }
            for c in result.cases
        ],
    }
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    md: list[str] = []
    verdict = "PASS" if result.all_match else "FAIL"
    md.append("# G4 — payload-equivalence (SHA-256) audit (v0.5 inline)\n")
    md.append(f"**Verdict: {verdict}** "
              f"({sum(c.match for c in result.cases)}/{len(result.cases)} "
              f"cases byte-identical across B/C/D)\n")
    md.append("- model: **v0.5 single-stream inline insertion** "
              "(C′ removed — §0.4/§0.6); conditions differ only in inline "
              "insertion *position*, never payload bytes")
    md.append(f"- pyrefly: `{result.pyrefly_version}`")
    md.append(f"- conditions audited: {', '.join(CONDITIONS)}")
    md.append("- payload: canonical `(severity, line, code, message)` tuples, "
              "top-K=10 by recency-of-edited-region, deterministic JSON")
    md.append("")
    md.append("Each case drives a real pyrefly daemon `prefix → edit` and routes "
              "the raw diagnostics through every condition's shared "
              "`normalize_payload` path. The SHA-256 is over those payload bytes; "
              "equality is the matched-information-content guarantee (RQ1, §0.6).")
    md.append("")
    md.append("| Case | Diags | Match | SHA-256 (shared) | Note |")
    md.append("|---|---:|:---:|---|---|")
    for c in result.cases:
        shared_sha = next(iter(c.shas.values())) if c.match else "MISMATCH"
        sha_disp = shared_sha[:16] if c.match else "**MISMATCH**"
        md.append(
            f"| `{c.name}` | {c.diag_count} | "
            f"{'yes' if c.match else 'NO'} | `{sha_disp}` | {c.note} |"
        )
    md.append("")
    if not result.all_match:
        md.append("## Mismatched cases (per-condition SHA-256)\n")
        for c in result.cases:
            if not c.match:
                md.append(f"### `{c.name}`")
                for cond, sha in c.shas.items():
                    md.append(f"- {cond}: `{sha}`")
                md.append("")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")

    print(f"[G4] {verdict}: "
          f"{sum(c.match for c in result.cases)}/{len(result.cases)} cases match")
    print(f"[G4] artifacts in {out_dir}")
    return 0 if result.all_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
