"""Single-stream eval harness for HumanEval + MBPP.

Used by L0 G1 to compare vanilla Qwen3-8B against stream-qwen3-8b in single-stream
mode (only the Output channel is read from the stream model).

Inference modes:
    - "vanilla": uses HF AutoModelForCausalLM.generate() — for Qwen/Qwen3-8B.
    - "stream":  uses model.stream_generate() and reads the Output channel only —
                 for JonasGeiping/stream-qwen3-8b. model.generate() is disabled on
                 stream checkpoints, so we must use the streaming API.

Output: per-problem pass/fail and the model's generation as JSONL.

Usage:
    python harness/single_stream_eval.py --model Qwen/Qwen3-8B --bench humaneval --out runs/g1/vanilla_humaneval
    python harness/single_stream_eval.py --model JonasGeiping/stream-qwen3-8b \\
        --mode stream --bench mbpp --out runs/g1/stream_mbpp --limit 3

NOTE: scoring uses an unsafe `exec` of model output against test cases — run in a
trusted environment only. (HumanEval's reference implementation does the same.)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import multiprocessing as mp
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Inference backends


def load_model(model_id: str, mode: str, dtype: torch.dtype = torch.bfloat16):
    """Load model + tokenizer.

    mode = "vanilla" -> standard HF; mode = "stream" -> trust_remote_code stream model.
    """
    common = dict(torch_dtype=dtype, device_map="auto")
    if mode == "stream":
        common["trust_remote_code"] = True
    model = AutoModelForCausalLM.from_pretrained(model_id, **common)
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=(mode == "stream"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


@torch.no_grad()
def generate_vanilla(model, tok, prompt: str, max_new_tokens: int = 512) -> str:
    """Standard greedy generation (raw prompt continuation)."""
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,  # ignored when do_sample=False; T=0 effectively via greedy
        pad_token_id=tok.pad_token_id,
    )
    full = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return full


@torch.no_grad()
def generate_vanilla_chat(model, tok, chat_prompt: str, max_new_tokens: int = 512) -> str:
    """Greedy generation from a *pre-rendered ChatML string* (G1 fairness path).

    `chat_prompt` must already be the apply_chat_template output (built by
    build_chat_prompt_*). We tokenize with add_special_tokens=False because the
    template already contains the special tokens.
    """
    inputs = tok(chat_prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tok.pad_token_id,
    )
    full = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return full


@torch.no_grad()
def generate_stream(model, tok, prompt: str, max_new_tokens: int = 512) -> str:
    """Stream-model generation — read only the Output channel.

    max_new_tokens is interpreted as max_rows (one row = one Output-channel token after
    silence/skip-silence filtering when used as designed).
    """
    # T=0 not directly supported by the stream sampler; use very low temperature.
    # silence_penalty + skip_silence keep the Output channel writing rather than emitting silence.
    result = model.stream_generate(
        tok,
        prompt,
        max_rows=max_new_tokens,
        temperature=1e-3,  # ~greedy
        top_k=1,
        top_p=1.0,
        silence_penalty=10.0,
        skip_silence=True,
        warm_start=False,
    )
    return result.output


# ---------------------------------------------------------------------------
# Code execution / scoring


def _exec_target(code: str, test: str, entry_point: str | None, queue):
    """Run inside a subprocess: exec code+test, signal success/failure on queue."""
    try:
        # Suppress stdout/stderr from the candidate code.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            ns: dict = {}
            exec(code, ns)
            exec(test, ns)
            if entry_point is not None:
                # HumanEval style: test defines `def check(candidate)`, then calls check(<entry_point>).
                if "check" in ns and entry_point in ns:
                    ns["check"](ns[entry_point])
        queue.put(("ok", None))
    except BaseException as e:  # includes assertion errors
        queue.put(("fail", f"{type(e).__name__}: {e}"))


def run_with_timeout(code: str, test: str, entry_point: str | None, timeout: float = 10.0) -> tuple[bool, str | None]:
    """Run candidate+test in a sandbox subprocess with a hard timeout."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_exec_target, args=(code, test, entry_point, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join(1.0)
        if p.is_alive():
            p.kill()
        return False, "TIMEOUT"
    try:
        status, msg = q.get_nowait()
    except Exception:
        return False, "NO_RESULT"
    return status == "ok", msg


# ---------------------------------------------------------------------------
# Benchmark loaders


def load_humaneval(limit: int | None = None):
    ds = load_dataset("openai/openai_humaneval", split="test")
    items = []
    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            break
        items.append({
            "task_id": ex["task_id"],
            "prompt": ex["prompt"],
            "test": ex["test"],
            "entry_point": ex["entry_point"],
            "canonical_solution": ex.get("canonical_solution", ""),
        })
    return items


def load_mbpp_sanitized(limit: int | None = None):
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    items = []
    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            break
        # MBPP sanitized: keys are task_id, source_file, prompt, code, test_imports, test_list, ...
        # Build a HumanEval-style prompt: prompt + an assert from the first test as a hint.
        prompt = ex["prompt"].strip()
        first_test = ex["test_list"][0] if ex["test_list"] else ""
        # Construct a Python function prompt
        text = f'"""\n{prompt}\n{first_test}\n"""\n'
        test_block = "\n".join(ex.get("test_imports", []) + ex["test_list"])
        items.append({
            "task_id": f"mbpp/{ex['task_id']}",
            "prompt": text,
            "test": test_block,
            "entry_point": None,  # MBPP tests are bare asserts that call the function by name
            "canonical_solution": ex.get("code", ""),
        })
    return items


# ---------------------------------------------------------------------------
# Post-processing of generation


_STOP_MARKERS = ("\nclass ", "\nif __name__", "\nprint(", "\n# Test", "\n# test", "\n```")


def extract_humaneval_completion(generation: str) -> str:
    """For HumanEval-style prompts (open function body), cut at first dedent/marker."""
    # Take everything until a top-level def/class or markdown fence.
    out = generation
    # Truncate at common stop markers.
    cut_idxs = [out.find(m) for m in _STOP_MARKERS]
    cut_idxs = [i for i in cut_idxs if i > 0]
    if cut_idxs:
        out = out[: min(cut_idxs)]
    # Stop at the next unindented `def` after the function body.
    lines = out.splitlines()
    keep = []
    started = False
    for line in lines:
        if line.strip().startswith("def ") and started:
            break
        keep.append(line)
        if line.strip():
            started = True
    return "\n".join(keep)


def extract_mbpp_code(generation: str) -> str:
    """For MBPP, the prompt is a docstring; model should emit a `def ...:` body."""
    out = generation
    # If the generation begins with a markdown fence, strip it.
    if "```" in out:
        # take content between first pair of fences if present
        parts = out.split("```")
        for p in parts:
            if "def " in p:
                # drop a "python" language tag line
                if p.lstrip().startswith("python"):
                    p = p.split("\n", 1)[1] if "\n" in p else ""
                return p
    # Otherwise, return until a stop marker.
    cut_idxs = [out.find(m) for m in _STOP_MARKERS if m not in ("\nclass ",)]
    cut_idxs = [i for i in cut_idxs if i > 0]
    if cut_idxs:
        out = out[: min(cut_idxs)]
    return out


# ---------------------------------------------------------------------------
# G1 fairness: chat-style prompting + fence-aware code extraction
#
# Rationale (see runs/g1/fairness_notes.md): the stream model's Output channel
# is chat-tuned and emits markdown-fenced code blocks, not raw prompt
# continuation. Posing each benchmark problem as a natural-language *chat
# instruction* — and extracting the fenced code block — is the apples-to-apples
# comparison. Vanilla Qwen3-8B consumes the rendered ChatML string via its own
# (identical) chat template; the stream model consumes the same instruction as
# plain user text (its trust_remote_code path builds the assistant turn
# internally and would mangle pre-rendered special tokens through _tokenize_user).
# Both therefore receive the SAME instruction in each model's native input form.


_HUMANEVAL_INSTR = (
    "Complete the following Python function. Return the COMPLETE function "
    "(signature + body) in a single ```python code block. Do not include tests "
    "or explanations.\n\n```python\n{prompt}```"
)

_MBPP_INSTR = (
    "Write a Python function for the following task. Return ONLY the function "
    "in a single ```python code block, no tests or explanation. The function "
    "must be named exactly as required by this test: `{test}`\n\nTask: {desc}"
)


def build_chat_prompt_humaneval(prompt: str, entry_point: str, for_stream: bool = False) -> str:
    """Build the HumanEval instruction. If for_stream, return the plain
    instruction text (the stream model wraps it); else render ChatML for vanilla."""
    instr = _HUMANEVAL_INSTR.format(prompt=prompt)
    if for_stream:
        return instr
    return _CHAT_TOK.apply_chat_template(
        [{"role": "user", "content": instr}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def build_chat_prompt_mbpp(desc: str, test_hint: str, for_stream: bool = False) -> str:
    instr = _MBPP_INSTR.format(test=test_hint.strip(), desc=desc.strip())
    if for_stream:
        return instr
    return _CHAT_TOK.apply_chat_template(
        [{"role": "user", "content": instr}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


# Tokenizer used purely to render the ChatML template for the vanilla model.
# Both models ship the identical Qwen3 ChatML template, so either works; we use
# the vanilla one. Set lazily by set_chat_tokenizer().
_CHAT_TOK = None


def set_chat_tokenizer(tok) -> None:
    global _CHAT_TOK
    _CHAT_TOK = tok


def _strip_think(text: str) -> str:
    """Drop any <think>...</think> block (vanilla may emit one even with the flag)."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text


def extract_code_completion(generation: str, original_prompt: str, entry_point: str) -> str:
    """Fence-aware extractor for chat-style output (used for BOTH models).

    Returns a self-contained, runnable code string (imports + function def).
    Strategy:
      1. strip any think block;
      2. prefer the first ```...``` fenced block that defines `entry_point`,
         else the first fenced block containing any `def `;
      3. if no fence, fall back to raw text;
      4. if the extracted code never defines `entry_point` but the original
         HumanEval prompt does (open signature), prepend original_prompt so the
         function exists (covers the case where the model emitted only a body).
    """
    text = _strip_think(generation)

    code = None
    if "```" in text:
        parts = text.split("```")
        # fenced blocks are at odd indices: [pre, block, mid, block, ...]
        blocks = []
        for idx in range(1, len(parts), 2):
            b = parts[idx]
            # drop a leading language tag line (python / py)
            first_nl = b.find("\n")
            if first_nl != -1 and b[:first_nl].strip().lower() in ("python", "py", "python3"):
                b = b[first_nl + 1:]
            blocks.append(b)
        # prefer a block defining entry_point
        for b in blocks:
            if entry_point and (f"def {entry_point}" in b):
                code = b
                break
        if code is None:
            for b in blocks:
                if "def " in b:
                    code = b
                    break
        if code is None and blocks:
            code = blocks[0]
    if code is None:
        code = text

    # Ensure the entry point is defined; if the model returned only a body or a
    # helper, fall back to gluing onto the original open-signature prompt.
    if entry_point and (f"def {entry_point}" not in code):
        # Did it emit an indented body only (no def line at all)? Glue under prompt.
        glued = original_prompt + code
        if f"def {entry_point}" in glued:
            return glued
    return code


# ---------------------------------------------------------------------------
# Main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model ID")
    parser.add_argument("--mode", choices=["vanilla", "stream"], default=None,
                        help="Inference mode; auto-detected from model ID if omitted")
    parser.add_argument("--bench", choices=["humaneval", "mbpp"], required=True)
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit problems (for dry-run)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--exec-timeout", type=float, default=10.0)
    parser.add_argument("--dry-run-no-model", action="store_true",
                        help="Skip model loading; emit empty generations (plumbing test)")
    args = parser.parse_args()

    mode = args.mode
    if mode is None:
        mode = "stream" if "stream-qwen" in args.model.lower() else "vanilla"
    print(f"[harness] model={args.model} mode={mode} bench={args.bench} limit={args.limit}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.bench == "humaneval":
        items = load_humaneval(args.limit)
        extract = extract_humaneval_completion
        # For HumanEval, the model continues from the prompt, then we combine prompt+completion for exec.
        wrap_for_exec = lambda prompt, completion: prompt + completion
    else:
        items = load_mbpp_sanitized(args.limit)
        extract = extract_mbpp_code
        wrap_for_exec = lambda prompt, completion: completion  # MBPP: just the code

    if args.dry_run_no_model:
        model = tok = None
        def gen_fn(prompt: str, n: int) -> str:
            return ""
    else:
        model, tok = load_model(args.model, mode)
        _backend = generate_stream if mode == "stream" else generate_vanilla
        def gen_fn(prompt: str, n: int) -> str:
            return _backend(model, tok, prompt, max_new_tokens=n)

    results = []
    n_pass = 0
    t0 = time.time()
    for i, item in enumerate(items):
        t_gen = time.time()
        gen = gen_fn(item["prompt"], args.max_new_tokens) if not args.dry_run_no_model else ""
        gen_time = time.time() - t_gen
        completion = extract(gen)
        code_to_run = wrap_for_exec(item["prompt"], completion)
        ok, err = run_with_timeout(code_to_run, item["test"], item.get("entry_point"), timeout=args.exec_timeout)
        n_pass += int(ok)
        rec = {
            "task_id": item["task_id"],
            "pass": ok,
            "error": err,
            "gen_time_s": round(gen_time, 2),
            "completion": completion,
            "raw_generation": gen,
        }
        results.append(rec)
        print(f"  [{i+1}/{len(items)}] {item['task_id']:20s} pass={ok} t={gen_time:.1f}s err={err}")

    total = time.time() - t0
    pass_at_1 = n_pass / max(len(items), 1)
    summary = {
        "model": args.model,
        "mode": mode,
        "bench": args.bench,
        "n_problems": len(items),
        "n_pass": n_pass,
        "pass_at_1": pass_at_1,
        "total_time_s": round(total, 1),
    }
    (out_dir / "results.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[harness] pass@1 = {n_pass}/{len(items)} = {pass_at_1:.3f}  (wall={total:.1f}s)")


if __name__ == "__main__":
    main()
