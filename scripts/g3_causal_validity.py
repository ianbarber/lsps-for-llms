#!/usr/bin/env python3
"""G3 — causal-validity gate runner (§11.1 G3; v0.5 interleaved layout, §0.4).

Reformats the mock teacher trajectories to conditions D and C in the single-stream
**interleaved** layout, runs the four causal-validity assertions per trajectory (plus the
adversarial-leak and collision checks), and writes:

  runs/g3_interleaved/causal_validity.json   — per-trajectory pass/fail on each assertion
  runs/g3_interleaved/summary.md             — human-readable gate report

This is a HARD L0 gate: it must pass. The same checks live in
tests/test_causal_validity.py as the pytest regression that re-runs at L1/L3.

The assertions (interleaved layout):
  (a) no teacher sync-diagnostic content remains in D's main (agent) token stream
  (b) the diagnostic block IS present, spliced inline at the latency-shifted position
  (c) the spliced position is >= the query (latency >= 0) and consistent with the offset
  (d) C: block at the edit boundary (offset 0), sync original absent from the main stream
  (adv) a deliberately naive (leaky) reformat IS flagged by (a) — proves non-vacuity
  (col) two diagnostics shifted onto the same position stack in order (no overwrite)

Usage:
  python scripts/g3_causal_validity.py [--use-hf-tokenizer]

  --use-hf-tokenizer  derive the student chars/token from the real Qwen2.5-Coder tokenizer
                      (CPU-only, loaded from HF_HOME / NAS cache; no model weights). Falls
                      back to the mock rate if transformers is unavailable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.reformat import (  # noqa: E402
    DIAG_CLOSE,
    DIAG_OPEN,
    TokenizerRate,
    reformat_to_C_interleaved,
    reformat_to_D_interleaved,
)
from training.teacher_trajectory import (  # noqa: E402
    SYNC_DIAG_MARKER,
    TeacherTrajectory,
    build_mock_trajectories,
)

OUT_DIR = ROOT / "runs" / "g3_interleaved"
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "teacher_trajectories"


def expected_diag_texts(traj: TeacherTrajectory) -> list[str]:
    out = []
    for resp in traj.responses():
        out.extend(d.render() for d in resp.diagnostics())
    return out


def check_trajectory(traj: TeacherTrajectory, rate: TokenizerRate, seed: int = 7) -> dict:
    """Run all assertions for one trajectory; return a per-assertion pass/fail record."""
    expected = expected_diag_texts(traj)
    has_diags = len(expected) > 0
    n_diag_snapshots = sum(1 for r in traj.responses() if r.diagnostics())

    d_seq = reformat_to_D_interleaved(traj, student_tokenizer=rate, seed=seed)
    c_seq = reformat_to_C_interleaved(traj, student_tokenizer=rate, seed=seed)
    d_agent = d_seq.agent_text()
    d_diag = d_seq.diag_text()

    detail: dict = {}

    # (a) no sync diagnostic content in D's main (agent) stream
    a_marker = SYNC_DIAG_MARKER not in d_agent
    a_content = all(txt not in d_agent for txt in expected)
    a_noleakmeta = all(
        not tk.meta.get("LEAKED_SYNC_DIAG") for tk in d_seq.agent_tokens()
    )
    a_pass = a_marker and a_content and a_noleakmeta
    detail["a_no_sync_in_main_stream"] = a_pass

    # (b) diagnostic block present, spliced inline (delimited)
    b_present = all(txt in d_diag for txt in expected)
    b_count = len(d_seq.diag_spans) == n_diag_snapshots
    b_delim = all(
        s.text.startswith(DIAG_OPEN) and s.text.endswith(DIAG_CLOSE) for s in d_seq.diag_spans
    )
    b_pass = b_present and b_count and b_delim
    detail["b_block_spliced_inline"] = b_pass

    # (c) latency shift consistent: requested_pos == query + latency, >= query
    c_pass = True
    offsets = d_seq.meta["latency_offsets"]
    for span in d_seq.diag_spans:
        rec = offsets[span.query_id]
        if not (
            rec["latency_student"] >= 0
            and span.requested_pos == rec["query_student_idx"] + rec["latency_student"]
            and span.requested_pos >= rec["query_student_idx"]
        ):
            c_pass = False
    detail["c_latency_shift_consistent"] = c_pass

    # (d) C: block at edit boundary (offset 0), sync original absent from main stream
    c_agent = c_seq.agent_text()
    d_marker_absent = SYNC_DIAG_MARKER not in c_agent
    d_zero_latency = all(
        rec["latency_student"] == 0 for rec in c_seq.meta["latency_offsets"].values()
    )
    d_at_boundary = all(
        span.requested_pos == c_seq.meta["latency_offsets"][span.query_id]["query_student_idx"]
        for span in c_seq.diag_spans
    )
    d_pass = d_marker_absent and d_zero_latency and d_at_boundary
    detail["d_C_edit_boundary"] = d_pass

    # (adv) adversarial leak is detected by (a) — only meaningful when diags exist
    if has_diags:
        leaky = reformat_to_D_interleaved(
            traj, student_tokenizer=rate, seed=seed, _adversarial_leak=True
        )
        leaky_agent = leaky.agent_text()
        leaked = SYNC_DIAG_MARKER in leaky_agent or any(
            tk.meta.get("LEAKED_SYNC_DIAG") for tk in leaky.agent_tokens()
        )
        detail["adv_leak_detected"] = bool(leaked)
    else:
        detail["adv_leak_detected"] = None  # n/a: no diagnostics to leak

    relevant = [v for v in detail.values() if v is not None]
    detail["_all_pass"] = all(relevant)
    detail["_n_diagnostics"] = len(expected)
    return detail


def check_collision(rate: TokenizerRate) -> dict:
    """Global collision check: two diagnostics latency-shifted onto the same position must
    stack in arrival order (earlier-querying block first), not overwrite. Uses t06's
    back-to-back queries with latencies chosen to collide on one position."""

    class _DecreasingSampler:
        def __init__(self):
            self.calls = 0
            self.vals = [4.0, 3.0]  # q0 -> 3+4=7, q1 -> 4+3=7

        def __call__(self, rng):
            v = self.vals[self.calls % len(self.vals)]
            self.calls += 1
            return v

    traj = next(t for t in build_mock_trajectories() if t.traj_id == "t06")
    seq = reformat_to_D_interleaved(
        traj, latency_sampler=_DecreasingSampler(), student_tokenizer=rate, seed=1
    )
    spans = {s.query_id: s for s in seq.diag_spans}
    collided = spans["t06.q0"].requested_pos == spans["t06.q1"].requested_pos
    ordered = spans["t06.q0"].start < spans["t06.q1"].start
    no_overlap = spans["t06.q0"].end <= spans["t06.q1"].start
    both_present = ("p shadows builtin" in seq.diag_text()
                    and "p is str not int" in seq.diag_text())
    return {
        "collided_on_same_position": collided,
        "earlier_query_spliced_first": ordered,
        "blocks_do_not_overlap": no_overlap,
        "both_diagnostics_present": both_present,
        "pass": collided and ordered and no_overlap and both_present,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-hf-tokenizer", action="store_true",
                    help="derive student chars/token from the real Qwen2.5-Coder tokenizer (CPU)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Prefer on-disk fixtures (the canonical artifacts) and fall back to the generator.
    if FIXTURE_DIR.exists() and any(FIXTURE_DIR.glob("*.json")):
        trajs = [TeacherTrajectory.load(p) for p in sorted(FIXTURE_DIR.glob("*.json"))]
        fixture_src = str(FIXTURE_DIR.relative_to(ROOT))
    else:
        trajs = build_mock_trajectories()
        fixture_src = "build_mock_trajectories() (in-memory)"

    if args.use_hf_tokenizer:
        try:
            rate = TokenizerRate.from_hf()
            rate_src = (
                f"Qwen2.5-Coder tokenizer (student_chars_per_token="
                f"{rate.student_chars_per_token:.3f})"
            )
        except Exception as exc:  # transformers missing / cache miss → documented fallback
            rate = TokenizerRate()
            rate_src = f"mock (HF tokenizer unavailable: {type(exc).__name__}; scale=1.0)"
    else:
        rate = TokenizerRate()
        rate_src = "mock (teacher=student=4.0 chars/token, scale=1.0)"

    results = {}
    for traj in trajs:
        results[traj.traj_id] = check_trajectory(traj, rate)

    collision = check_collision(rate)

    n = len(results)
    n_pass = sum(1 for r in results.values() if r["_all_pass"])
    gate_pass = (n_pass == n) and collision["pass"]

    payload = {
        "gate": "G3_causal_validity",
        "layout": "interleaved-single-stream (v0.5 §0.4)",
        "delimiter": {"open": DIAG_OPEN, "close": DIAG_CLOSE},
        "fixture_source": fixture_src,
        "tokenizer_rate": rate_src,
        "tokenizer_scale": rate.scale,
        "n_trajectories": n,
        "n_pass": n_pass,
        "gate_pass": gate_pass,
        "assertions": {
            "a": "no teacher sync-diagnostic content remains in D's main (agent) stream",
            "b": "diagnostic block is spliced inline (delimited), latency-shifted",
            "c": "spliced position >= query (latency >= 0), requested_pos == query + latency",
            "d": "C: block at the edit boundary (offset 0), sync original absent",
            "adv": "a naive leaky reformat IS flagged by (a) — proves the gate is non-vacuous",
            "col": "two diagnostics shifted onto the same position stack in order (no overwrite)",
        },
        "collision_check": collision,
        "per_trajectory": results,
    }
    (OUT_DIR / "causal_validity.json").write_text(json.dumps(payload, indent=2))

    # summary.md
    cols = ["a_no_sync_in_main_stream", "b_block_spliced_inline", "c_latency_shift_consistent",
            "d_C_edit_boundary", "adv_leak_detected"]
    sym = lambda v: "n/a" if v is None else ("PASS" if v else "FAIL")
    lines = [
        "# G3 — Causal-Validity Gate (interleaved single-stream layout, v0.5 §0.4)",
        "",
        f"**Gate:** {'PASS' if gate_pass else 'FAIL'} "
        f"({n_pass}/{n} trajectories pass all assertions; "
        f"collision check {'PASS' if collision['pass'] else 'FAIL'})",
        "",
        f"- Layout: single interleaved token stream; diagnostic block delimited by "
        f"`{DIAG_OPEN}` … `{DIAG_CLOSE}`, spliced inline at `query_pos + latency`.",
        f"- Condition C replaces the old C′ + C (single-stream removes the format axis); "
        f"C = offset 0 (edit boundary), D = latency-shifted.",
        f"- Fixtures: `{fixture_src}`",
        f"- Tokenizer rate: {rate_src} (scale = student tok / teacher tok = {rate.scale:.3f})",
        "",
        "## Assertions",
        "- **(a)** no teacher sync-diagnostic content remains in D's main (agent) token stream",
        "- **(b)** the diagnostic block is spliced inline (properly delimited), latency-shifted",
        "- **(c)** spliced position is >= the query (latency >= 0) and == query + latency",
        "- **(d)** C: block at the edit boundary (offset 0), sync original absent from main stream",
        "- **(adv)** a deliberately naive (leaky) reformat IS flagged by (a) — proves non-vacuity",
        "- **(col)** two diagnostics latency-shifted onto the same position stack in order",
        "",
        "## Per-trajectory",
        "",
        "| traj | n_diag | (a) | (b) | (c) | (d) | (adv) | all |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for tid, r in results.items():
        lines.append(
            f"| {tid} | {r['_n_diagnostics']} | "
            + " | ".join(sym(r[c]) for c in cols)
            + f" | {'PASS' if r['_all_pass'] else 'FAIL'} |"
        )
    lines += [
        "",
        "## Collision check",
        "",
        f"- collided on same position: {collision['collided_on_same_position']}",
        f"- earlier-querying block spliced first: {collision['earlier_query_spliced_first']}",
        f"- blocks do not overlap: {collision['blocks_do_not_overlap']}",
        f"- both diagnostics present (none dropped): {collision['both_diagnostics_present']}",
        f"- **collision check: {'PASS' if collision['pass'] else 'FAIL'}**",
        "",
        "## Interpretation",
        "",
        "Assertion (a) is the load-bearing causal-validity check: it proves D's interleaved",
        "training stream does not contain the teacher's synchronous inline diagnostic, so the",
        "D student never sees both the sync and the latency-replayed async copy. The (adv)",
        "column confirms the check is non-vacuous — a naive reformat that leaves the sync",
        "block inline in the main stream is detected. n/a in (adv) marks clean-snapshot",
        "trajectories with no diagnostics to leak. The collision check guards the original-G3",
        "bug, re-cast for interleaving: two diagnostics latency-shifted onto the same position",
        "are stacked in arrival order (later-querying block AFTER the earlier), not overwritten.",
        "",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines))

    print(f"G3 gate (interleaved): {'PASS' if gate_pass else 'FAIL'}  ({n_pass}/{n}; "
          f"collision {'PASS' if collision['pass'] else 'FAIL'})")
    print(f"  -> {OUT_DIR / 'causal_validity.json'}")
    print(f"  -> {OUT_DIR / 'summary.md'}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
