#!/usr/bin/env python3
"""Deterministic scripted stand-in for the local model (CPU-only apparatus dry runs).

The stub drives the REAL `scaffold.stream_agent.StreamAgent` loop — the same parser,
cursor-advancement, completion-origin, stale-test-invalidation, and acceptance-gate code
paths the model runs use — but the "model" simply replays a fixed action script
character by character. No weights are loaded and no API is called, so the v6 scaffold
mechanics can be verified mechanically before any GPU time is spent.

Design notes:
- `CharTokenizer` is character-level and renders the same `<|im_start|>`/`<|im_end|>`
  chat-turn shape the scaffold splices, so observation boundaries are exercised
  exactly as in a real run.
- `ScriptedStubModel` emits the next scripted action one character at a time and yields
  a single EOS for a `None` script item (an explicit end-of-turn). It distinguishes its
  own single-token feedback from multi-token prompt/observation splices, so literal
  action text inside a user observation can never advance the script — the stream the
  agent parses, however, still contains that literal text, which is what the dry run
  is verifying the v6 cursors against.
- Token counts recorded by the agent are therefore CHARACTER counts; they are stub
  bookkeeping, not model-token measurements.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch


class CharTokenizer:
    """Character-level tokenizer with the chat-template shape the scaffold splices."""

    def __init__(self) -> None:
        self.eos_token_id = 0
        self._id2ch: dict[int, str] = {0: ""}
        self._ch2id: dict[str, int] = {}

    def _id(self, ch: str) -> int:
        got = self._ch2id.get(ch)
        if got is None:
            got = len(self._id2ch)
            self._id2ch[got] = ch
            self._ch2id[ch] = got
        return got

    def encode_text(self, text: str) -> list[int]:
        return [self._id(ch) for ch in text]

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        ids = self.encode_text(text)
        return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))

    def decode(self, ids) -> str:
        return "".join(self._id2ch[int(i)] for i in ids)

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **_):
        parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs]
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)


class ScriptedStubModel:
    """Replays a fixed action script through the live StreamAgent decoding loop.

    Script items are strings emitted verbatim (actions such as `<test/>`, `<done/>`,
    or a full multiline `<edit .../>` block); a `None` item emits exactly one EOS,
    i.e. an explicit model end-of-turn with no action.
    """

    def __init__(self, tokenizer: CharTokenizer, script: list) -> None:
        self.tok = tokenizer
        self.script = list(script)
        self.pending: list[int] = []
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(_commit_hash="scripted-stub")

    def eval(self):
        return self

    def _logits(self, target: int):
        size = max(len(self.tok._id2ch), target + 1)
        logits = torch.zeros((1, 1, size))
        logits[0, 0, target] = 1.0
        return SimpleNamespace(logits=logits, past_key_values=None)

    def __call__(self, input_ids=None, past_key_values=None, use_cache=True):
        ids = input_ids[0].tolist() if input_ids is not None else []
        if len(ids) == 1 and self.pending:
            # Single-token input is the agent feeding back the character we just
            # emitted. Multi-token input is always a prompt or observation splice.
            if ids[0] != self.pending[0]:
                raise RuntimeError(
                    "stub desync: fed-back token does not match the scripted emission"
                )
            self.pending.pop(0)
        if not self.pending and self.script and self.script[0] is None:
            self.script.pop(0)
            return self._logits(self.tok.eos_token_id)
        if not self.pending and self.script:
            self.pending = self.tok.encode_text(self.script.pop(0))
        target = self.pending[0] if self.pending else self.tok.eos_token_id
        return self._logits(target)


def gold_repair_line_edit(draft_source: str, gold_source: str, path: str) -> str:
    """Render the exact gold-restoring `<edit path lines="a-b">` action for a seeded defect.

    The seeded mutations are single contiguous rewrites of the validated gold target, so
    a top/bottom line scan isolates one divergent region. When the gold side of the
    region is empty (the defect only INSERTED lines), the region is widened by one
    shared context line so the replacement body is non-empty and the applied edit
    restores the gold file byte-identically.
    """
    a = draft_source.splitlines()
    b = gold_source.splitlines()
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    j = 0
    while j < min(len(a), len(b)) - i and a[len(a) - 1 - j] == b[len(b) - 1 - j]:
        j += 1
    if len(b) - j < i:
        raise ValueError("draft and gold do not share a single divergent region")
    if len(b) - j == i:
        if i == 0:
            raise ValueError("cannot anchor an insertion-only repair at file start")
        i -= 1
    start, end = i + 1, len(a) - j
    if end < start:
        raise ValueError("draft has no lines to replace")
    body = "\n".join(b[i:len(b) - j])
    return f'<edit path="{path}" lines="{start}-{end}">\n{body}\n</edit>'
