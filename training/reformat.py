"""Latency-replay reformat pipeline (§0.4 / §7.4 — the load-bearing methodological piece).

**v0.5 Interaction-Model Pivot.** Turns a *synchronous-teacher* trajectory (condition C
scaffold; agent tokens + inline sync LSP responses + measured latencies) into a
**single interleaved token sequence** for condition D (async, latency-replayed) or
condition C (forced sync post-edit, no latency). This replaces the old multi-stream
grid layout (``MultiStreamSequence``, kept below for provenance) with the single-stream
interleaved layout of §0.4: there is no longer a separate "Analytical" side channel —
the diagnostic block is spliced **inline** into the agent's token stream at the right
token position, delimited by sentinels.

The four moving parts (§7.4 step 3, re-cast for interleaving):

  1. Agent tokens  -> the main interleaved stream, in student-token order.
  2. Sync diagnostic content **removed from the main stream** — the causal-validity
     gate. This is the whole point of G3: if we leave the teacher's inline sync
     diagnostic in the stream AND also splice the latency-shifted async copy, the D
     student trains on both and the D-vs-C comparison is invalid (§13).
  3. The diagnostic is re-emitted as a delimited inline block, spliced at
     ``query_pos + round(latency_sample / ms_per_token)`` for D (= the edit/query
     boundary, offset 0, for C).
  4. **Tokenizer-rate adjustment**: latency offsets are computed in *student-coder-
     tokenizer* time (the student is Qwen2.5-Coder), not teacher-tokenizer time.
     Teacher token indices are converted via the ratio of student-tokens-per-char to
     teacher-tokens-per-char (see ``TokenizerRate``).

The output is an in-memory ``InterleavedSequence``: an ordered list of tokens (agent +
diagnostic, interleaved) plus recorded diagnostic-block spans and provenance. This is the
structure SFT later consumes; a JSON serializer is provided.

Diagnostic-block delimiter (§0.9 q1): ``DIAG_OPEN`` / ``DIAG_CLOSE`` sentinels wrap the
block, with one rendered diagnostic per line inside. See ``training/INTERLEAVED_LAYOUT.md``.

No model / GPU here. A real Qwen2.5-Coder student tokenizer can be loaded CPU-side from
the NAS cache for the step-4 conversion (see ``TokenizerRate.from_hf``); a mock char-rate
is the default so the unit tests run with zero heavy deps.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from training.teacher_trajectory import (
    Diagnostic,
    TeacherTrajectory,
    TrajectoryEvent,
)

# --- interleaved-layout sentinels (§0.4 / §0.9 q1) -------------------------------------
# Tokenizer-friendly delimiters wrapping an inline diagnostic block. The guillemet form
# ‹…› matches the §0.4 example; the literal sentinel strings are what G3 scans for and
# what the SFT serializer emits. Real training may map these to dedicated special tokens
# on the Qwen2.5-Coder tokenizer; the string form keeps the layout tokenizer-agnostic.
DIAG_OPEN = "‹diag›"
DIAG_CLOSE = "‹/diag›"

# Legacy multi-stream channel names — kept importable for provenance / the retired grid
# layout below. The interleaved path does NOT use channels.
OUTPUT_CHANNEL = "Output"
DIAG_CHANNEL = "Analytical"
SILENCE = "<silence>"


# ---------------------------------------------------------------------------
# Tokenizer-rate adjustment (§7.4 step 4)
# ---------------------------------------------------------------------------
@dataclass
class TokenizerRate:
    """Converts teacher-token indices/offsets to student-token time.

    A latency measured as "k teacher tokens elapsed" is not k student tokens — the two
    tokenizers segment text at different rates. We convert through characters: a token
    index in the teacher stream maps to a character offset (via measured chars/token),
    then to a student token index (via the student's chars/token). The scale factor is
    ``teacher_chars_per_token / student_chars_per_token`` = student tokens per teacher
    token.

    For mocks we supply both rates directly. ``from_hf`` derives the student rate by
    actually tokenizing a corpus sample with the real Qwen2.5-Coder tokenizer (CPU, no
    weights).
    """

    teacher_chars_per_token: float = 4.0
    student_chars_per_token: float = 4.0

    @property
    def scale(self) -> float:
        """Student tokens per one teacher token."""
        return self.teacher_chars_per_token / self.student_chars_per_token

    def teacher_to_student(self, teacher_idx: float) -> int:
        """Map a teacher-token index/offset to a student-token index (rounded)."""
        return round(teacher_idx * self.scale)

    @classmethod
    def from_hf(
        cls,
        student_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        teacher_chars_per_token: float = 4.0,
        sample_text: str | None = None,
    ) -> "TokenizerRate":
        """Derive student chars/token from the real Qwen2.5-Coder tokenizer (CPU-only).

        Loads only the tokenizer (no model weights) from HF_HOME / the NAS cache
        (``HF_HOME=/mnt/nas/hf-cache``). The teacher rate stays a supplied constant until
        real Phase-0 teacher rollouts give a measured teacher chars/token. If transformers
        is unavailable, callers should fall back to the mock ``TokenizerRate()`` — the
        ratio interface is identical, so the reformat machinery is unaffected.
        """
        from transformers import AutoTokenizer  # local import: heavy, optional

        tok = AutoTokenizer.from_pretrained(student_model, trust_remote_code=True)
        sample = sample_text or _DEFAULT_RATE_SAMPLE
        n_tokens = len(tok(sample)["input_ids"])
        student_cpt = len(sample) / max(n_tokens, 1)
        return cls(
            teacher_chars_per_token=teacher_chars_per_token,
            student_chars_per_token=student_cpt,
        )


_DEFAULT_RATE_SAMPLE = (
    "def solve(items):\n    total = 0\n    for x in items:\n"
    "        total += x.value\n    return total\n"
) * 8


# ---------------------------------------------------------------------------
# Latency sampler (§7.4 step 3) — pluggable
# ---------------------------------------------------------------------------
LatencySampler = Callable[[random.Random], float]


@dataclass
class EmpiricalLatencySampler:
    """Samples a pyrefly latency, expressed in **teacher-token units**.

    The constructor takes an empirical latency distribution. For now this is a stub list
    of teacher-token-equivalent offsets; in Phase 0 it is filled with measured pyrefly
    daemon round-trips (G5: p95 6–21 ms) converted to teacher-token units via
    ``ms_per_token``. The interface (``__call__(rng) -> float``) is what
    ``reformat_to_D_interleaved`` depends on, so real latencies slot in by replacing the
    ``samples`` list.
    """

    samples: list[float] = field(
        # stub: spread of teacher-token-equivalent latency offsets (>0). Real values
        # come from Phase-0 daemon measurements; these merely exercise the machinery.
        default_factory=lambda: [1.0, 2.0, 3.0, 5.0, 8.0, 4.0, 6.0]
    )

    def __call__(self, rng: random.Random) -> float:
        return rng.choice(self.samples)


def zero_latency_sampler(rng: random.Random) -> float:
    """C's sampler: no latency offset (diagnostic at the edit boundary, offset 0)."""
    return 0.0


# ---------------------------------------------------------------------------
# Diagnostic-block rendering (§0.4) — shared payload form for C and D
# ---------------------------------------------------------------------------
def render_diag_block(diags: list[Diagnostic]) -> str:
    """Render a list of diagnostics as one delimited inline block.

    Layout:  ``‹diag›\n[sev] L{line} {code}: {msg}\n…\n‹/diag›``
    The inner lines reuse ``Diagnostic.render()`` — the exact content the old grid layout
    placed on the side channel — so payloads stay byte-comparable with G4 / the old runs.
    """
    body = "\n".join(d.render() for d in diags)
    return f"{DIAG_OPEN}\n{body}\n{DIAG_CLOSE}"


# ---------------------------------------------------------------------------
# Interleaved single-stream sequence representation (§0.4 — the new layout)
# ---------------------------------------------------------------------------
@dataclass
class InterleavedToken:
    """One token in the single interleaved stream.

    ``pos`` is the final ordered position in the interleaved stream (student-token time).
    ``source`` is "agent" or "diagnostic". Diagnostic tokens carry block provenance in
    ``meta`` (query_id, sampled latency, the requested splice position).
    """

    pos: int
    text: str
    source: str  # "agent" | "diagnostic"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagBlockSpan:
    """A spliced diagnostic block: its token range + the provenance used to place it."""

    query_id: str
    start: int  # first interleaved position of the block (the DIAG_OPEN sentinel)
    end: int  # one past the last interleaved position (exclusive)
    query_student_idx: int  # the edit/query position in student-token time
    latency_student: int  # the latency offset applied (0 for C)
    requested_pos: int  # query_student_idx + latency_student (pre-collision splice point)
    text: str  # the rendered block text


@dataclass
class InterleavedSequence:
    """A single ordered token stream with the agent's tokens and inline diagnostic blocks.

    The whole sequence is one list of ``InterleavedToken`` in emission order. Diagnostic
    blocks are spliced *between* agent tokens at the latency-shifted (D) or edit-boundary
    (C) position; ``diag_spans`` records where each block landed plus its provenance.

    Collision handling (§ the original-G3 bug, re-cast for interleaving): two diagnostics
    whose latency-shifted positions land on the *same* agent position are NOT overwritten.
    They are spliced in arrival order at that position — the earlier-querying snapshot's
    block first, the later one immediately after it — so no diagnostic is dropped and
    ordering is preserved. See ``_splice``.
    """

    traj_id: str
    condition: str  # "D" | "C"
    tokens: list[InterleavedToken] = field(default_factory=list)
    diag_spans: list[DiagBlockSpan] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    # --- text views (used by G3 to scan for leakage) --------------------------
    def full_text(self) -> str:
        """Concatenated text of the entire interleaved stream (agent + diagnostics)."""
        return "".join(tk.text for tk in self.tokens)

    def agent_text(self) -> str:
        """Concatenated text of the agent tokens only — the 'main stream' minus diag blocks.

        This is what assertion (a) scans: the teacher's sync diagnostic must NOT appear
        here. (The async diagnostic lives in diagnostic-source tokens, excluded here.)
        """
        return "".join(tk.text for tk in self.tokens if tk.source == "agent")

    def diag_text(self) -> str:
        """Concatenated text of diagnostic-source tokens only."""
        return "".join(tk.text for tk in self.tokens if tk.source == "diagnostic")

    def diagnostic_tokens(self) -> list[InterleavedToken]:
        return [tk for tk in self.tokens if tk.source == "diagnostic"]

    def agent_tokens(self) -> list[InterleavedToken]:
        return [tk for tk in self.tokens if tk.source == "agent"]

    def _renumber(self) -> None:
        """Reassign contiguous ``pos`` after splices/sorting."""
        for i, tk in enumerate(self.tokens):
            tk.pos = i

    # --- serialization --------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "traj_id": self.traj_id,
            "condition": self.condition,
            "layout": "interleaved-single-stream",
            "delimiter": {"open": DIAG_OPEN, "close": DIAG_CLOSE},
            "meta": self.meta,
            "tokens": [asdict(tk) for tk in self.tokens],
            "diag_spans": [asdict(s) for s in self.diag_spans],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InterleavedSequence":
        seq = cls(traj_id=d["traj_id"], condition=d["condition"])
        seq.meta = d.get("meta", {})
        seq.tokens = [InterleavedToken(**t) for t in d.get("tokens", [])]
        seq.diag_spans = [DiagBlockSpan(**s) for s in d.get("diag_spans", [])]
        return seq

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


# ---------------------------------------------------------------------------
# Core interleaved reformat
# ---------------------------------------------------------------------------
def _reformat_interleaved(
    trajectory: TeacherTrajectory,
    latency_sampler: LatencySampler,
    rate: TokenizerRate,
    condition: str,
    rng: random.Random,
    *,
    leak_sync_into_stream: bool = False,
) -> InterleavedSequence:
    """Shared interleaved reformat body for D and C.

    ``leak_sync_into_stream=True`` is the **adversarial / naive** path: it deliberately
    leaves the teacher's inline sync diagnostic in the main agent stream (spliced at the
    edit boundary, *not* removed). It exists ONLY so G3 can prove it *catches* leakage (a
    test that only ever sees clean data proves nothing). The production paths never set it.
    """
    events = trajectory.sorted_events()
    seq = InterleavedSequence(traj_id=trajectory.traj_id, condition=condition)
    seq.meta = {
        "teacher_tokenizer": trajectory.teacher_tokenizer,
        "tokenizer_scale": rate.scale,
        "delimiter": {"open": DIAG_OPEN, "close": DIAG_CLOSE},
        "latency_offsets": {},  # query_id -> {teacher/student idx, latency, requested_pos}
    }

    # 1. Agent tokens -> the main stream (student-token order). We build a list of agent
    #    tokens keyed by their student-token position; diagnostic blocks splice in between.
    #    Each agent token records its student position so we can find a splice anchor.
    agent_tokens: list[InterleavedToken] = []
    for e in events:
        if e.type != "agent_token":
            continue
        ts = rate.teacher_to_student(e.t_emit)
        agent_tokens.append(
            InterleavedToken(pos=ts, text=e.text, source="agent", meta={"teacher_idx": e.t_emit})
        )
    agent_tokens.sort(key=lambda tk: tk.pos)
    # max agent student-position, used to clamp a too-late splice onto the stream tail.
    max_agent_pos = agent_tokens[-1].pos if agent_tokens else 0

    # 2 + 3. Plan the diagnostic blocks. The sync inline response text is NOT placed on
    #        the main stream (causal gate); each block is spliced inline, latency-shifted.
    #        We first *plan* every block (compute its requested student position), then
    #        splice them in a deterministic order so colliding blocks stack rather than
    #        overwrite.
    planned: list[tuple[int, str, list[Diagnostic], TrajectoryEvent]] = []
    for resp in trajectory.responses():
        qid = resp.payload.get("query_id")
        query = next(
            (q for q in trajectory.queries() if q.payload.get("query_id") == qid), None
        )
        query_idx = query.t_emit if query else resp.t_emit
        query_idx_student = rate.teacher_to_student(query_idx)

        latency_teacher = latency_sampler(rng)
        latency_student = rate.teacher_to_student(latency_teacher)
        requested_pos = query_idx_student + latency_student

        seq.meta["latency_offsets"][qid] = {
            "query_teacher_idx": query_idx,
            "query_student_idx": query_idx_student,
            "latency_teacher": latency_teacher,
            "latency_student": latency_student,
            "requested_pos": requested_pos,
        }

        diags = resp.diagnostics()
        if not diags:
            continue  # clean snapshot: no inline block emitted
        planned.append((requested_pos, qid, diags, resp))

    # Deterministic splice order: by requested position, tie-broken by query order in the
    # trajectory (query_teacher_idx) so a later-querying snapshot that lands on the same
    # position is spliced AFTER the earlier one (ordering preserved, no overwrite).
    def _query_order(qid: str) -> int:
        rec = seq.meta["latency_offsets"][qid]
        return rec["query_teacher_idx"]

    planned.sort(key=lambda p: (p[0], _query_order(p[1])))

    # 4. Splice. We build the final token list by walking agent positions and inserting
    #    each planned block immediately before the first agent token whose student
    #    position is >= the block's requested position (i.e. "mid-stream, after the
    #    edit"). Blocks past the last agent token append at the tail. Equal requested
    #    positions retain the sorted (query-order) sequence, so they stack in order.
    out: list[InterleavedToken] = []
    bi = 0  # index into planned

    def _emit_block(requested_pos: int, qid: str, diags: list[Diagnostic]) -> None:
        rec = seq.meta["latency_offsets"][qid]
        block_text = render_diag_block(diags)
        start = len(out)
        # one InterleavedToken per rendered line of the block, so the block is a real span
        # of tokens (open sentinel / each diag line / close sentinel).
        lines = [DIAG_OPEN] + [d.render() for d in diags] + [DIAG_CLOSE]
        for j, line in enumerate(lines):
            out.append(
                InterleavedToken(
                    pos=-1,  # renumbered after the full splice
                    text=line if j == 0 else "\n" + line,
                    source="diagnostic",
                    meta={
                        "query_id": qid,
                        "query_student_idx": rec["query_student_idx"],
                        "latency_student": rec["latency_student"],
                        "requested_pos": requested_pos,
                        "block_line": j,
                    },
                )
            )
        end = len(out)
        seq.diag_spans.append(
            DiagBlockSpan(
                query_id=qid,
                start=start,
                end=end,
                query_student_idx=rec["query_student_idx"],
                latency_student=rec["latency_student"],
                requested_pos=requested_pos,
                text=block_text,
            )
        )

    for tk in agent_tokens:
        # before emitting this agent token, splice any planned block whose requested
        # position is at-or-before this agent token's position.
        while bi < len(planned) and planned[bi][0] <= tk.pos:
            req, qid, diags, _resp = planned[bi]
            _emit_block(req, qid, diags)
            bi += 1
        out.append(tk)

    # any remaining blocks requested past the last agent token: append at the tail.
    while bi < len(planned):
        req, qid, diags, _resp = planned[bi]
        _emit_block(req, qid, diags)
        bi += 1

    # ADVERSARIAL ONLY: a naive reformat that forgot the causal gate would leave the
    # teacher's inline sync block in the main agent stream at the edit boundary. We splice
    # it as *agent*-sourced tokens (so assertion (a), which scans agent text, catches it).
    if leak_sync_into_stream:
        out = _inject_sync_leak(out, trajectory, rate)

    seq.tokens = out
    seq._renumber()
    # fix span ranges if a leak shifted positions (leak appends/rebuilds out)
    if leak_sync_into_stream:
        _reindex_spans(seq)
    return seq


def _inject_sync_leak(
    tokens: list[InterleavedToken],
    trajectory: TeacherTrajectory,
    rate: TokenizerRate,
) -> list[InterleavedToken]:
    """Adversarial: re-insert the teacher's sync inline block as agent tokens at the edit
    boundary. Rebuilds the token list with the leaked blocks interleaved by student pos."""
    leaks: list[tuple[int, str]] = []
    for resp in trajectory.responses():
        if not resp.diagnostics():
            continue
        sync_pos = rate.teacher_to_student(resp.t_emit)
        leaks.append((sync_pos, resp.text))
    # rebuild: walk current tokens (which already carry pos), splice each leak before the
    # first token whose pos >= the leak's sync position.
    out: list[InterleavedToken] = []
    li = 0
    leaks.sort(key=lambda x: x[0])
    for tk in tokens:
        while li < len(leaks) and leaks[li][0] <= tk.pos:
            out.append(
                InterleavedToken(
                    pos=-1, text=leaks[li][1], source="agent",
                    meta={"LEAKED_SYNC_DIAG": True},
                )
            )
            li += 1
        out.append(tk)
    while li < len(leaks):
        out.append(
            InterleavedToken(
                pos=-1, text=leaks[li][1], source="agent",
                meta={"LEAKED_SYNC_DIAG": True},
            )
        )
        li += 1
    return out


def _reindex_spans(seq: InterleavedSequence) -> None:
    """Recompute diag-span start/end after a rebuild changed token indices."""
    # spans are identified by their first diagnostic token's query_id + block_line==0.
    by_qid_start: dict[str, int] = {}
    by_qid_end: dict[str, int] = {}
    for i, tk in enumerate(seq.tokens):
        if tk.source != "diagnostic":
            continue
        qid = tk.meta.get("query_id")
        if qid is None:
            continue
        by_qid_start.setdefault(qid, i)
        by_qid_end[qid] = i + 1
    for span in seq.diag_spans:
        if span.query_id in by_qid_start:
            span.start = by_qid_start[span.query_id]
            span.end = by_qid_end[span.query_id]


# ---------------------------------------------------------------------------
# Public interleaved reformat entry points (§0.5 conditions)
# ---------------------------------------------------------------------------
def reformat_to_D_interleaved(
    trajectory: TeacherTrajectory,
    latency_sampler: LatencySampler | None = None,
    student_tokenizer: TokenizerRate | None = None,
    *,
    seed: int = 0,
    _adversarial_leak: bool = False,
) -> InterleavedSequence:
    """Condition D (async interleaved): diagnostic block spliced mid-stream.

    The sync diagnostic is removed from the main stream; the async block is spliced inline
    at ``query_student_idx + latency_student`` (some tokens AFTER the triggering edit).
    Offsets are in student-coder-tokenizer time.
    """
    latency_sampler = latency_sampler or EmpiricalLatencySampler()
    rate = student_tokenizer or TokenizerRate()
    rng = random.Random(seed)
    return _reformat_interleaved(
        trajectory, latency_sampler, rate, "D", rng,
        leak_sync_into_stream=_adversarial_leak,
    )


def reformat_to_C_interleaved(
    trajectory: TeacherTrajectory,
    student_tokenizer: TokenizerRate | None = None,
    *,
    seed: int = 0,
) -> InterleavedSequence:
    """Condition C (forced sync post-edit): diagnostic block at the edit boundary (offset 0).

    Replaces both the old C′ (multi-stream sync) and the old C — single-stream interleaving
    removes the format axis, so C is simply D with zero latency. The sync original is still
    masked from the main stream; the canonical inline block is spliced at the edit boundary.
    """
    rate = student_tokenizer or TokenizerRate()
    rng = random.Random(seed)
    return _reformat_interleaved(trajectory, zero_latency_sampler, rate, "C", rng)


# ===========================================================================
# RETIRED multi-stream grid layout (provenance only — superseded by §0.4).
# Kept importable so old artifacts / provenance code still load. The interleaved
# path above is the v0.5 production layout; do not use the grid for new data.
# ===========================================================================
@dataclass
class StreamToken:
    """One token slot on one channel at one timestep (legacy grid layout)."""

    channel: str
    text: str  # token text, or SILENCE
    timestep: int
    source: str  # "agent" | "diagnostic" | "silence"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiStreamSequence:
    """Parallel channels, one token per (channel, timestep) (legacy grid layout).

    Superseded by ``InterleavedSequence`` in v0.5; retained for provenance.
    """

    traj_id: str
    condition: str  # "D" | "Cprime"
    channels: list[str]
    length: int
    grid: dict[str, list[StreamToken]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(cls, traj_id: str, condition: str, channels: list[str], length: int):
        grid = {
            ch: [
                StreamToken(channel=ch, text=SILENCE, timestep=t, source="silence")
                for t in range(length)
            ]
            for ch in channels
        }
        return cls(traj_id=traj_id, condition=condition, channels=channels, length=length, grid=grid)

    def place(self, channel: str, timestep: int, text: str, source: str, meta=None) -> None:
        if timestep >= self.length:
            self._grow(timestep + 1)
        self.grid[channel][timestep] = StreamToken(
            channel=channel, text=text, timestep=timestep, source=source, meta=meta or {}
        )

    def _grow(self, new_len: int) -> None:
        for ch in self.channels:
            for t in range(self.length, new_len):
                self.grid[ch].append(
                    StreamToken(channel=ch, text=SILENCE, timestep=t, source="silence")
                )
        self.length = new_len

    def next_free(self, channel: str, timestep: int) -> int:
        t = timestep
        while t < self.length and self.grid[channel][t].source != "silence":
            t += 1
        return t

    def channel_text(self, channel: str) -> str:
        return "".join(tk.text for tk in self.grid[channel] if tk.source != "silence")

    def non_silence(self, channel: str) -> list[StreamToken]:
        return [tk for tk in self.grid[channel] if tk.source != "silence"]

    def diagnostic_tokens(self) -> list[StreamToken]:
        return [tk for tk in self.grid[DIAG_CHANNEL] if tk.source == "diagnostic"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "traj_id": self.traj_id,
            "condition": self.condition,
            "channels": self.channels,
            "length": self.length,
            "meta": self.meta,
            "tokens": [
                asdict(tk)
                for ch in self.channels
                for tk in self.grid[ch]
                if tk.source != "silence"
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MultiStreamSequence":
        seq = cls.empty(d["traj_id"], d["condition"], d["channels"], d["length"])
        seq.meta = d.get("meta", {})
        for t in d["tokens"]:
            seq.place(t["channel"], t["timestep"], t["text"], t["source"], t.get("meta", {}))
        return seq

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
