"""G3 — causal-validity unit test (§11.1 G3, §13; v0.5 interleaved layout, §0.4).

Proves that the latency-replay reformat (training/reformat.py) does NOT leak the
teacher's synchronous inline diagnostic into condition D's main interleaved token stream.
If it did, the D student would be trained on both the inline sync signal and the delayed
async copy → the D-vs-C comparison (the project's central readout) would be invalid.

v0.5 pivot: the layout is now a single **interleaved** token stream (the diagnostic block
is spliced inline at the latency-shifted token position), not a multi-stream grid with a
side channel. C replaces the old C′ AND the old C (single-stream removes the format axis).

Four assertions per reformatted D trajectory:
  (a) no teacher sync-diagnostic content survives in D's main (agent) token stream
  (b) the diagnostic block IS present, spliced inline at the latency-shifted position
  (c) the spliced position is strictly after the query position and consistent with the
      sampled offset (in student-coder-tokenizer time)
  (d) for C: the block is at the edit boundary (offset 0) and the sync original is absent.

Plus an ADVERSARIAL-LEAK assertion: a deliberately naive reformat (sync block left inline
in the main stream) MUST be flagged by assertion (a). This proves the test detects leakage
rather than passing trivially.

Plus a COLLISION assertion: two diagnostics latency-shifted onto the same position must be
spliced in order (the later-querying one after, not overwriting the earlier).

This module is the pytest regression form — it re-runs at L1/L3. The CLI gate
(scripts/g3_causal_validity.py) reuses these same checks and writes the run artifacts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.reformat import (  # noqa: E402
    DIAG_CLOSE,
    DIAG_OPEN,
    EmpiricalLatencySampler,
    TokenizerRate,
    reformat_to_C_interleaved,
    reformat_to_D_interleaved,
)
from training.teacher_trajectory import (  # noqa: E402
    SYNC_DIAG_MARKER,
    build_mock_trajectories,
)

TRAJECTORIES = build_mock_trajectories()
TRAJ_IDS = [t.traj_id for t in TRAJECTORIES]

_WITH_DIAGS = [
    t for t in TRAJECTORIES
    if t.responses() and any(r.diagnostics() for r in t.responses())
]
_WITH_DIAGS_IDS = [t.traj_id for t in _WITH_DIAGS]


def _expected_diag_texts(traj) -> list[str]:
    """Rendered diagnostic strings the reformat should splice into the stream."""
    out = []
    for resp in traj.responses():
        out.extend(d.render() for d in resp.diagnostics())
    return out


# --------------------------------------------------------------------------- #
#  Assertion (a): no sync diagnostic content in D's main (agent) token stream
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("traj", TRAJECTORIES, ids=TRAJ_IDS)
def test_a_no_sync_diag_in_main_stream(traj):
    seq = reformat_to_D_interleaved(traj, seed=7)
    agent = seq.agent_text()
    # marker-based check (mock-specific): the sync marker must never reach the agent stream.
    assert SYNC_DIAG_MARKER not in agent, f"{traj.traj_id}: sync marker leaked into main stream"
    # content-based check (generalizes to real data): no rendered diagnostic text as agent
    # tokens. (The async block lives in diagnostic-source tokens, excluded from agent_text.)
    for diag_text in _expected_diag_texts(traj):
        assert diag_text not in agent, f"{traj.traj_id}: diag text leaked into main stream"
    # no agent token may be sourced from a leaked sync diagnostic.
    assert all(
        not tk.meta.get("LEAKED_SYNC_DIAG") for tk in seq.agent_tokens()
    ), f"{traj.traj_id}: a LEAKED_SYNC_DIAG token is present in the main stream"


# --------------------------------------------------------------------------- #
#  Assertion (b): diagnostic block spliced inline, latency-shifted
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("traj", TRAJECTORIES, ids=TRAJ_IDS)
def test_b_diag_block_spliced_inline(traj):
    seq = reformat_to_D_interleaved(traj, seed=7)
    diag_stream = seq.diag_text()
    expected = _expected_diag_texts(traj)
    # every expected diagnostic must appear in a spliced diagnostic block.
    for diag_text in expected:
        assert diag_text in diag_stream, f"{traj.traj_id}: '{diag_text}' missing from stream"
    # number of spliced blocks == number of snapshots that produced diagnostics.
    n_diag_snapshots = sum(1 for r in traj.responses() if r.diagnostics())
    assert len(seq.diag_spans) == n_diag_snapshots, (
        f"{traj.traj_id}: {len(seq.diag_spans)} blocks, expected {n_diag_snapshots}"
    )
    # each block is properly delimited.
    for span in seq.diag_spans:
        assert span.text.startswith(DIAG_OPEN) and span.text.endswith(DIAG_CLOSE), (
            f"{traj.traj_id}: block {span.query_id} missing delimiters"
        )
        assert span.end > span.start, f"{traj.traj_id}: empty block span"


# --------------------------------------------------------------------------- #
#  Assertion (c): spliced position later than query, consistent with offset
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("traj", _WITH_DIAGS, ids=_WITH_DIAGS_IDS)
def test_c_latency_shift_consistent(traj):
    seq = reformat_to_D_interleaved(traj, seed=7)
    offsets = seq.meta["latency_offsets"]
    # build a position->student-index map of agent tokens by their order in the stream.
    # an agent token at stream index i has the student position recorded in its meta.
    stream = seq.tokens
    for span in seq.diag_spans:
        rec = offsets[span.query_id]
        # latency >= 0 (async copy is never before the query)
        assert rec["latency_student"] >= 0, f"{traj.traj_id}: negative latency"
        # requested splice position == query_student_idx + latency_student
        assert span.requested_pos == rec["query_student_idx"] + rec["latency_student"], (
            f"{traj.traj_id}: requested_pos inconsistent with recorded offset"
        )
        # the block never precedes the query (edit) position in student time.
        assert span.requested_pos >= rec["query_student_idx"], (
            f"{traj.traj_id}: block requested before query position"
        )
        # placement invariant: the block is spliced immediately before the first agent
        # token whose student position >= requested_pos. So the last agent token BEFORE
        # the block (if any) has student pos < requested_pos, and the first agent token
        # AFTER the block (if any) has student pos >= requested_pos.
        before = [stream[i] for i in range(span.start) if stream[i].source == "agent"]
        after = [stream[i] for i in range(span.end, len(stream)) if stream[i].source == "agent"]
        if before:
            # last preceding agent token's student pos is strictly before requested_pos
            prev_student = _student_pos_of(seq, before[-1])
            assert prev_student < span.requested_pos, (
                f"{traj.traj_id}: block spliced too late (agent pos {prev_student} "
                f">= requested {span.requested_pos})"
            )
        if after:
            nxt_student = _student_pos_of(seq, after[0])
            assert nxt_student >= span.requested_pos, (
                f"{traj.traj_id}: block spliced too early (next agent pos {nxt_student} "
                f"< requested {span.requested_pos})"
            )


def _student_pos_of(seq, agent_tok) -> int:
    """The student-token position recorded for an agent token (rate-converted teacher idx)."""
    scale = seq.meta["tokenizer_scale"]
    return round(agent_tok.meta["teacher_idx"] * scale)


# --------------------------------------------------------------------------- #
#  Assertion (d): C — block at the edit boundary (offset 0), sync original absent
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("traj", TRAJECTORIES, ids=TRAJ_IDS)
def test_d_C_edit_boundary_no_shift(traj):
    seq = reformat_to_C_interleaved(traj, seed=7)
    agent = seq.agent_text()
    assert SYNC_DIAG_MARKER not in agent, f"{traj.traj_id}: C leaked sync marker into main stream"
    offsets = seq.meta["latency_offsets"]
    for rec in offsets.values():
        assert rec["latency_student"] == 0, f"{traj.traj_id}: C has nonzero latency"
    for span in seq.diag_spans:
        rec = offsets[span.query_id]
        # edit boundary: requested position equals the query (edit) position exactly.
        assert span.requested_pos == rec["query_student_idx"], (
            f"{traj.traj_id}: C block not at edit boundary (offset != 0)"
        )


# --------------------------------------------------------------------------- #
#  Adversarial-leak guard: the test MUST catch a naive (leaky) reformat
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("traj", _WITH_DIAGS, ids=_WITH_DIAGS_IDS)
def test_adversarial_leak_is_detected(traj):
    """A reformat that forgets the causal gate leaves the sync block inline in the main
    stream; the (a) check must flag it. If this fails to detect, the gate is vacuous.
    """
    leaky = reformat_to_D_interleaved(traj, seed=7, _adversarial_leak=True)
    agent = leaky.agent_text()
    leaked = SYNC_DIAG_MARKER in agent or any(
        tk.meta.get("LEAKED_SYNC_DIAG") for tk in leaky.agent_tokens()
    )
    assert leaked, (
        f"{traj.traj_id}: adversarial reformat did NOT leak — the G3 test would be "
        "vacuous (cannot prove it catches leakage)"
    )


# --------------------------------------------------------------------------- #
#  Collision guard: two diagnostics on the same shifted position keep their order
# --------------------------------------------------------------------------- #
def test_collision_preserves_order():
    """Two snapshots latency-shifted onto the SAME position must stack in arrival order,
    not overwrite. t06 has back-to-back queries (q0@3, q1@4); choose latencies so both
    land on the same requested position and assert the earlier-querying block comes first.
    """
    traj = next(t for t in TRAJECTORIES if t.traj_id == "t06")

    class _DecreasingSampler:
        def __init__(self):
            self.calls = 0
            self.vals = [4.0, 3.0]  # q0 -> 3+4=7, q1 -> 4+3=7  (collision at pos 7)

        def __call__(self, rng):
            v = self.vals[self.calls % len(self.vals)]
            self.calls += 1
            return v

    seq = reformat_to_D_interleaved(traj, latency_sampler=_DecreasingSampler(), seed=1)
    spans = {s.query_id: s for s in seq.diag_spans}
    assert spans["t06.q0"].requested_pos == spans["t06.q1"].requested_pos, (
        "test setup: the two blocks should collide on the same requested position"
    )
    # earlier-querying snapshot (q0) is spliced strictly before the later one (q1).
    assert spans["t06.q0"].start < spans["t06.q1"].start, (
        "collision: later-arriving diagnostic overwrote/preceded the earlier one"
    )
    # both blocks survive (no drop): two distinct, non-overlapping spans.
    assert spans["t06.q0"].end <= spans["t06.q1"].start, "collision: blocks overlap"
    # both diagnostics' content is present.
    diag = seq.diag_text()
    assert "p shadows builtin" in diag and "p is str not int" in diag


def test_tokenizer_rate_changes_offsets():
    """Tokenizer-rate adjustment (step 4) actually rescales student-time offsets."""
    traj = next(t for t in TRAJECTORIES if t.traj_id == "t08")  # long stream, late snapshot
    fast_student = TokenizerRate(teacher_chars_per_token=4.0, student_chars_per_token=2.0)  # scale 2
    sampler = EmpiricalLatencySampler(samples=[5.0])  # fixed latency
    base = reformat_to_D_interleaved(
        traj, latency_sampler=sampler, student_tokenizer=TokenizerRate(), seed=0
    )
    scaled = reformat_to_D_interleaved(
        traj, latency_sampler=sampler, student_tokenizer=fast_student, seed=0
    )
    b = next(iter(base.meta["latency_offsets"].values()))
    s = next(iter(scaled.meta["latency_offsets"].values()))
    assert s["query_student_idx"] == b["query_student_idx"] * 2
    assert s["latency_student"] == b["latency_student"] * 2
