"""Synchronous-teacher trajectory schema (§7.4 step 1-2).

A *synchronous-teacher trajectory* is the raw material the latency-replay reformat
(``training/reformat.py``) consumes. It is the log of a single condition-C scaffold
rollout produced by a strong teacher: agent tokens with token-level timestamps,
the LSP queries the scaffold fired, and the LSP responses that came back *inline*
(synchronously — the teacher blocked on them, so the diagnostic text physically sits
in the teacher's token stream right after the query).

That inline-sync placement is exactly the thing the causal-validity gate (G3, §11.1)
must strip from condition D's training prefix: in D the diagnostic must arrive only on
the side stream, latency-shifted. If the reformat left the sync copy in the Output
prefix, the D student would be trained on both the sync inline copy and the delayed
async copy → guaranteed leakage and an invalid D-vs-C′ comparison (§13).

This module is *pure data*: schema + (de)serialization + a small mock generator. No
model, no GPU, no real teacher rollouts (those are Phase 0). The mocks are sufficient
to build and unit-test the reformat machinery at L0.

Event-time convention
---------------------
``t_emit`` is an integer **teacher-tokenizer token index** — the position, in the
teacher's own decode stream, at which the event occurs. agent_token events are dense
(one per teacher token). An lsp_query is emitted at the token index where the scaffold
fired the snapshot. An lsp_response, in the *synchronous* teacher, is logged at the
token index where the inline diagnostic text begins (i.e. immediately after the query,
because the teacher blocked). The reformat converts these teacher-token indices to
student-tokenizer time and re-places the diagnostic on the side stream.

Diagnostic payload
------------------
The normalized diagnostic payload (§7.1) is a list of ``(severity, line, code,
message)`` tuples. For mocks we embed a unique, easily-detectable **marker token**
in the message text of every sync diagnostic so the G3 absence assertion has something
unambiguous to search for in D's Output stream (see ``SYNC_DIAG_MARKER``).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


# A marker substring stamped into every mock sync-diagnostic message. The G3
# causal-validity test asserts this string never survives into D's Output stream.
# In real Phase-0 data there is no marker; G3 instead matches the diagnostic's
# rendered text against the source lsp_response payloads (see reformat/G3 docs).
SYNC_DIAG_MARKER = "<<SYNC_DIAG>>"

EventType = Literal["agent_token", "lsp_query", "lsp_response"]


@dataclass
class Diagnostic:
    """A single normalized diagnostic tuple (§7.1 payload format)."""

    severity: str  # "error" | "warning" | "info"
    line: int
    code: str  # e.g. "bad-return-type"
    message: str

    def render(self) -> str:
        """Canonical text form used when this diagnostic is emitted into a stream."""
        return f"[{self.severity}] L{self.line} {self.code}: {self.message}"


@dataclass
class TrajectoryEvent:
    """One ordered event in a synchronous-teacher trajectory.

    Fields by type:
      - agent_token:  ``text`` = the decoded token text; ``t_emit`` = teacher token idx.
      - lsp_query:    ``payload['query_id']`` identifies the snapshot; ``t_emit`` = idx
                      at which the scaffold fired the snapshot.
      - lsp_response: ``payload['query_id']`` links back to its query;
                      ``payload['diagnostics']`` = list[Diagnostic-as-dict];
                      ``t_emit`` = idx where the *inline sync* diagnostic text begins
                      (immediately after the query in the synchronous teacher).
                      ``text`` = the rendered inline diagnostic block (this is the
                      content the causal-validity gate removes from D's prefix).
    """

    type: EventType
    t_emit: int
    text: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def diagnostics(self) -> list[Diagnostic]:
        return [Diagnostic(**d) for d in self.payload.get("diagnostics", [])]


@dataclass
class TeacherTrajectory:
    """An ordered synchronous-teacher rollout for one task."""

    traj_id: str
    events: list[TrajectoryEvent]
    teacher_tokenizer: str = "mock-teacher-tokenizer"
    meta: dict[str, Any] = field(default_factory=dict)

    # --- ordering / validation ------------------------------------------------
    def sorted_events(self) -> list[TrajectoryEvent]:
        # Stable sort by t_emit; lsp_query must precede its lsp_response at equal idx.
        order = {"lsp_query": 0, "lsp_response": 1, "agent_token": 2}
        return sorted(self.events, key=lambda e: (e.t_emit, order[e.type]))

    def queries(self) -> list[TrajectoryEvent]:
        return [e for e in self.sorted_events() if e.type == "lsp_query"]

    def responses(self) -> list[TrajectoryEvent]:
        return [e for e in self.sorted_events() if e.type == "lsp_response"]

    def response_for(self, query_id: str) -> TrajectoryEvent | None:
        for e in self.responses():
            if e.payload.get("query_id") == query_id:
                return e
        return None

    # --- (de)serialization ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "traj_id": self.traj_id,
            "teacher_tokenizer": self.teacher_tokenizer,
            "meta": self.meta,
            "events": [asdict(e) for e in self.sorted_events()],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TeacherTrajectory":
        return cls(
            traj_id=d["traj_id"],
            teacher_tokenizer=d.get("teacher_tokenizer", "mock-teacher-tokenizer"),
            meta=d.get("meta", {}),
            events=[TrajectoryEvent(**e) for e in d["events"]],
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "TeacherTrajectory":
        return cls.from_dict(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# Mock trajectory construction helpers
# ---------------------------------------------------------------------------
def _agent_tokens(text_tokens: list[str], start_idx: int) -> list[TrajectoryEvent]:
    return [
        TrajectoryEvent(type="agent_token", t_emit=start_idx + i, text=t)
        for i, t in enumerate(text_tokens)
    ]


def make_sync_response(
    query_id: str, t_emit: int, diagnostics: list[Diagnostic]
) -> tuple[TrajectoryEvent, TrajectoryEvent]:
    """Build a linked (lsp_query, lsp_response) pair.

    The response's ``text`` is the rendered inline diagnostic block, each line carrying
    SYNC_DIAG_MARKER — this is the synchronous diagnostic content the teacher emitted
    *inline in its own token stream*, and exactly what the causal-validity gate strips
    from D's Output prefix.
    """
    query = TrajectoryEvent(
        type="lsp_query", t_emit=t_emit, text="", payload={"query_id": query_id}
    )
    rendered = "\n".join(f"{SYNC_DIAG_MARKER} {d.render()}" for d in diagnostics)
    response = TrajectoryEvent(
        type="lsp_response",
        t_emit=t_emit,  # synchronous: response sits at the query index (teacher blocked)
        text=rendered,
        payload={
            "query_id": query_id,
            "diagnostics": [asdict(d) for d in diagnostics],
        },
    )
    return query, response


def build_mock_trajectories() -> list[TeacherTrajectory]:
    """~10 hand-built synchronous-teacher trajectories exercising the reformat + G3.

    Variety covered:
      - single vs multiple snapshots per trajectory
      - 0, 1, and many diagnostics per snapshot
      - snapshots early / mid / late in the stream
      - back-to-back snapshots (tests latency ordering between adjacent queries)
    """
    trajs: list[TeacherTrajectory] = []

    def diag(sev, line, code, msg):
        return Diagnostic(sev, line, code, msg)

    # --- t00: single snapshot, single error -----------------------------------
    ev = _agent_tokens(["def", " f", "(", "x", "):", " return", " x", "+", "1"], 0)
    q, r = make_sync_response("t00.q0", 9, [diag("error", 1, "bad-return-type", "expected int")])
    ev += [q, r] + _agent_tokens(["\n", " #", " fixed"], 10)
    trajs.append(TeacherTrajectory("t00", ev))

    # --- t01: two snapshots ----------------------------------------------------
    ev = _agent_tokens(["import", " os", "\n", "x", "=", "1"], 0)
    q, r = make_sync_response("t01.q0", 6, [diag("warning", 1, "unused-import", "os unused")])
    ev += [q, r] + _agent_tokens(["\n", "y", "=", "x", ".", "foo"], 7)
    q2, r2 = make_sync_response("t01.q1", 13, [diag("error", 2, "missing-attr", "int has no foo")])
    ev += [q2, r2] + _agent_tokens(["\n", "z", "=", "y"], 14)
    trajs.append(TeacherTrajectory("t01", ev))

    # --- t02: snapshot with zero diagnostics (clean) ---------------------------
    ev = _agent_tokens(["a", "=", "[", "]", "\n", "a", ".", "append", "(", "1", ")"], 0)
    q, r = make_sync_response("t02.q0", 11, [])  # clean snapshot, no diagnostics
    ev += [q, r] + _agent_tokens(["\n", "print", "(", "a", ")"], 12)
    trajs.append(TeacherTrajectory("t02", ev))

    # --- t03: many diagnostics in one snapshot ---------------------------------
    ev = _agent_tokens(["class", " C", ":", " pass"], 0)
    q, r = make_sync_response(
        "t03.q0",
        4,
        [
            diag("error", 1, "no-init", "C has no __init__"),
            diag("warning", 1, "empty-class", "C is empty"),
            diag("info", 1, "style", "missing docstring"),
        ],
    )
    ev += [q, r] + _agent_tokens(["\n", "c", "=", "C", "(", ")"], 5)
    trajs.append(TeacherTrajectory("t03", ev))

    # --- t04: snapshot at very start -------------------------------------------
    q, r = make_sync_response("t04.q0", 0, [diag("error", 1, "syntax", "unexpected EOF")])
    ev = [q, r] + _agent_tokens(["x", "=", "1", "\n", "y", "=", "2"], 1)
    trajs.append(TeacherTrajectory("t04", ev))

    # --- t05: snapshot at very end ---------------------------------------------
    ev = _agent_tokens(["def", " g", "(", ")", ":", " return"], 0)
    q, r = make_sync_response("t05.q0", 6, [diag("error", 1, "bad-return", "return needs value")])
    ev += [q, r]
    trajs.append(TeacherTrajectory("t05", ev))

    # --- t06: back-to-back snapshots (adjacent query indices) ------------------
    ev = _agent_tokens(["p", "=", "1"], 0)
    q, r = make_sync_response("t06.q0", 3, [diag("warning", 1, "shadow", "p shadows builtin")])
    q2, r2 = make_sync_response("t06.q1", 4, [diag("error", 2, "type", "p is str not int")])
    ev += [q, r] + _agent_tokens(["q"], 4) + [q2, r2] + _agent_tokens(["=", "p"], 5)
    trajs.append(TeacherTrajectory("t06", ev))

    # --- t07: three snapshots, mixed counts ------------------------------------
    ev = _agent_tokens(["a", "b", "c"], 0)
    q0, r0 = make_sync_response("t07.q0", 3, [diag("info", 1, "s0", "note zero")])
    q1, r1 = make_sync_response("t07.q1", 4, [])
    q2, r2 = make_sync_response(
        "t07.q2", 5, [diag("error", 3, "e2", "err two"), diag("warning", 3, "w2", "warn two")]
    )
    ev += [q0, r0] + _agent_tokens(["d"], 4) + [q1, r1] + _agent_tokens(["e"], 5) + [q2, r2]
    trajs.append(TeacherTrajectory("t07", ev))

    # --- t08: long stream, single late snapshot --------------------------------
    ev = _agent_tokens([f"tok{i}" for i in range(40)], 0)
    q, r = make_sync_response("t08.q0", 40, [diag("error", 12, "undef", "name X undefined")])
    ev += [q, r] + _agent_tokens(["end"], 41)
    trajs.append(TeacherTrajectory("t08", ev))

    # --- t09: snapshot whose message contains chars resembling code text -------
    # (guards against naive substring scans that confuse diag text with agent code)
    ev = _agent_tokens(["return", " x"], 0)
    q, r = make_sync_response("t09.q0", 2, [diag("error", 1, "ret", "cannot return x here")])
    ev += [q, r] + _agent_tokens(["\n", "return", " 0"], 3)
    trajs.append(TeacherTrajectory("t09", ev))

    return trajs
