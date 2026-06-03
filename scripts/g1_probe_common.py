#!/usr/bin/env python3
"""Shared helpers for the G1 go/no-go probe of stream-qwen3-8b code capability.

Loads the stream model + the PATCHED generate (loosened all-silent early-stop),
drives it canonically (prompt on User channel, read Output channel idx 1,
warm_start primer, temperature=0 -> argmax), and provides a best-effort,
TRANSPARENT extractor that turns the Output text into runnable code.
"""
from __future__ import annotations

import os
import re
import sys

os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

STREAM_REPO = "JonasGeiping/stream-qwen3-8b"
VANILLA_REPO = "Qwen/Qwen3-8B"

# Import the PATCHED generate (loosened early-stop), NOT the HF-cache copy.
_PATCHED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "runs", "g1_probe", "patched",
)
sys.path.insert(0, _PATCHED_DIR)
from stream_inference_probe import generate, detect_silence_token  # noqa: E402


# ---------------------------------------------------------------------------
# Model loading


def load_stream():
    snapshot_download(STREAM_REPO)  # ensure cached; no-op offline
    model = AutoModelForCausalLM.from_pretrained(
        STREAM_REPO, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(STREAM_REPO, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    sil = detect_silence_token(tok)
    return model, tok, sil


def load_vanilla():
    model = AutoModelForCausalLM.from_pretrained(
        VANILLA_REPO, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(VANILLA_REPO)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


# ---------------------------------------------------------------------------
# Prompting

# Instruction-style prompt used for BOTH models (matched). The stream model's
# Output channel is chat-tuned; the vanilla Qwen3-8B is an instruct model.
HUMANEVAL_INSTR = (
    "Complete the following Python function. Return the COMPLETE function "
    "(signature and body) in a single ```python code block. Do not include "
    "tests or explanations.\n\n```python\n{prompt}```"
)


def build_instr(prompt: str) -> str:
    return HUMANEVAL_INSTR.format(prompt=prompt)


# ---------------------------------------------------------------------------
# Stream driving


@torch.no_grad()
def stream_output(model, tok, sil, user_text: str, *, silence_penalty: float,
                  max_rows: int = 256, temperature: float = 0.0) -> str:
    """Drive the stream model canonically and return the decoded Output channel."""
    out_ids = []
    g = generate(
        model, tok, user_text, sil,
        max_rows=max_rows, temperature=temperature,
        silence_penalty=silence_penalty, warm_start=True,
    )
    for _row_idx, row, is_prefill in g:
        if not is_prefill:
            out_ids.append(row[1])  # Output channel id=1
    nonsil = [t for t in out_ids if t != sil]
    return tok.decode(nonsil), len(nonsil)


# ---------------------------------------------------------------------------
# Vanilla driving


@torch.no_grad()
def vanilla_output(model, tok, instr: str, max_new_tokens: int = 512) -> str:
    msgs = [{"role": "user", "content": instr}]
    chat = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tok(chat, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Extraction (best-effort, transparent)


def _strip_think(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text


def _normalize_stream_spacing(text: str) -> str:
    """The stream Output decode shows BPE tokens joined with stray spaces
    (`def add (a , b)`) and sometimes literal backslash-n. Best-effort cleanup.

    We do NOT aggressively reformat — we only fix the artifacts that reliably
    break exec: literal `\\n` -> newline, and spacing around punctuation that
    Python tokenization is whitespace-insensitive to anyway is left, since
    Python ignores spaces inside expressions. The main hazard is INDENTATION,
    which the spaced decode usually preserves via newlines. Keep it light.
    """
    # Literal backslash-n / backslash-t that leaked as text.
    text = text.replace("\\n", "\n").replace("\\t", "\t")
    return text


def extract_code(raw_output: str, prompt: str, entry_point: str) -> tuple[str, str]:
    """Turn raw Output text into runnable code for the HumanEval test.

    Returns (code, note) where note records what extraction did (for triage).
    Python is whitespace-insensitive *within* lines, so the spaced-token decode
    (`def f (x , y)`) still execs fine as long as line structure / indentation
    survives. The real failure mode is missing the def or losing indentation.
    """
    text = _strip_think(raw_output)
    text = _normalize_stream_spacing(text)
    notes = []

    code = None
    if "```" in text:
        parts = text.split("```")
        blocks = []
        for idx in range(1, len(parts), 2):
            b = parts[idx]
            nl = b.find("\n")
            if nl != -1 and b[:nl].strip().lower() in ("python", "py", "python3"):
                b = b[nl + 1:]
            blocks.append(b)
        for b in blocks:
            if entry_point and f"def {entry_point}" in b.replace(" ", ""):
                code = b
                notes.append("fenced-block-with-entrypoint")
                break
        if code is None:
            for b in blocks:
                if "def " in b:
                    code = b
                    notes.append("fenced-block-with-def")
                    break
        if code is None and blocks:
            code = blocks[0]
            notes.append("fenced-block-first")
    if code is None:
        code = text
        notes.append("no-fence-raw-text")

    # If the def header got spacing artifacts (`def has_close_elements (numbers`),
    # the entry-point name may still be intact; check on a despaced view.
    despaced = code.replace(" ", "")
    has_ep = entry_point and f"def{entry_point}" in despaced

    glued = False
    if entry_point and not has_ep:
        # Model may have emitted only a body. Glue under the original open
        # signature prompt so the function exists.
        candidate = prompt + code
        if f"def {entry_point}" in candidate.replace(" ", ""):
            code = candidate
            glued = True
            notes.append("glued-onto-prompt")

    return code, ";".join(notes) if notes else "empty"
