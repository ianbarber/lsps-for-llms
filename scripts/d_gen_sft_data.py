#!/usr/bin/env python3
"""Interleaved-async SFT data-generation pipeline (v0.5 option D, §0.4–§0.6).

Produces tokenized, loss-masked SFT datasets for conditions **A / C / D** that
teach a Qwen2.5-Coder student to *react to live LSP diagnostics interleaved into
its generation*. The construction is the **latency-replay protocol** (§7.4) on a
*bootstrapped* corpus: real Python functions are programmatically bug-mutated, a
**real pyrefly** type diagnostic is captured on the buggy version, and a
synchronous-teacher trajectory (buggy -> [diag] -> fix) is reformatted into the
three conditions via the EXISTING interleaved reformat (`training/reformat.py`).

Pipeline (this file owns it; see the task brief):

  1. Source functions — HumanEval (prompt+canonical_solution) + MBPP (code),
     keep ones pyrefly type-checks clean (<=1 diagnostic) so the injected bug is
     the only signal.
  2. Inject ONE deterministic, type-checker-visible mutation per (function, mut).
  3. Real pyrefly diagnostic on the BUGGY version; keep only mutations that
     actually produce a diagnostic (drop silent ones).
  4. Build a TeacherTrajectory: agent emits buggy fn -> lsp_response carrying the
     normalized diagnostic at the edit position -> agent emits the corrected fn.
  5. Reformat to A / C / D via reformat_to_{C,D}_interleaved (A = D with the diag
     block stripped). D uses a latency sweep over {0,2,8,32} student tokens.
  6. Tokenize + loss-mask with the Qwen2.5-Coder-7B tokenizer: labels = input_ids
     EXCEPT -100 on (a) the prompt/instruction tokens and (b) the entire diag
     block span (from InterleavedSequence.diag_spans). Only the fix is loss-bearing.
  7. Save jsonl per condition + summary.md to runs/d_sft_data/.

Non-GPU. Pyrefly via the existing daemon client (temp workspace + `pyrefly init`).
Deterministic via `random.Random(seed)`.

Usage:
  PYTHONPATH=/home/ianbarber/Projects/Streams HF_HOME=/mnt/nas/hf-cache \
    .venv-streams/bin/python scripts/d_gen_sft_data.py \
      [--max-functions N] [--max-examples-per-condition N] [--seed 0] \
      [--no-tokenizer]  # skip HF tokenizer (mock rate, no input_ids) for a dry run
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lsp.payload import EditedRegion, normalize_diagnostics  # noqa: E402
from lsp.pyrefly_client import DEFAULT_PYREFLY, PyreflyDaemon  # noqa: E402
from training.reformat import (  # noqa: E402
    DIAG_CLOSE,
    DIAG_OPEN,
    EmpiricalLatencySampler,
    TokenizerRate,
    reformat_to_C_interleaved,
    reformat_to_D_interleaved,
)
from training.teacher_trajectory import (  # noqa: E402
    Diagnostic,
    TeacherTrajectory,
    TrajectoryEvent,
)

STUDENT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
OUT_DIR = ROOT / "runs" / "d_sft_data"
# Latency sweep (§0.9 Q4): student-token offsets D's diagnostic block lands after the edit.
LATENCY_SWEEP = [0, 2, 8, 32]


# ===========================================================================
# 1. Source functions
# ===========================================================================
@dataclass
class SourceFn:
    fn_id: str
    source: str  # "humaneval" | "mbpp"
    code: str    # full self-contained function text (clean / correct)


def load_source_functions(max_functions: int | None) -> list[SourceFn]:
    """HumanEval (prompt+canonical_solution) + MBPP (sanitized, code field).

    Both are small, self-contained typed-ish Python functions. HumanEval prompts
    carry type annotations (good pyrefly signal); MBPP is mixed but its `code` is
    short and self-contained. Cleanliness (pyrefly <=1 diagnostic) is enforced
    later in `keep_clean`.
    """
    from datasets import load_dataset  # heavy, local import

    fns: list[SourceFn] = []
    he = load_dataset("openai/openai_humaneval", split="test")
    for ex in he:
        code = ex["prompt"] + ex["canonical_solution"]
        fns.append(SourceFn(fn_id=f"he/{ex['task_id']}", source="humaneval", code=code))

    for split in ("train", "test", "validation", "prompt"):
        try:
            mb = load_dataset("google-research-datasets/mbpp", "sanitized", split=split)
        except Exception:
            continue
        for ex in mb:
            code = ex["code"]
            if not code or "def " not in code:
                continue
            fns.append(
                SourceFn(fn_id=f"mbpp/{split}/{ex['task_id']}", source="mbpp", code=code)
            )

    if max_functions is not None:
        fns = fns[:max_functions]
    return fns


# ===========================================================================
# 2. Mutations — ONE deterministic, type-checker-visible mutation per call.
#    Each returns (mutated_code, edited_line_1indexed) or None if inapplicable.
#    The "produces-a-diagnostic" filter (step 3) drops silent ones, so we can
#    afford mutations that only *sometimes* trip the type checker.
# ===========================================================================
Mutation = Callable[[str], tuple[str, int] | None]


def _line_of_offset(code: str, off: int) -> int:
    return code.count("\n", 0, off) + 1  # 1-indexed


def _first_replace(code: str, pattern: str, repl: str) -> tuple[str, int] | None:
    idx = code.find(pattern)
    if idx < 0:
        return None
    new = code[:idx] + repl + code[idx + len(pattern):]
    return new, _line_of_offset(code, idx)


def mut_return_wrong_type(code: str) -> tuple[str, int] | None:
    """Replace the first `return <expr>` body with a string literal — a bad-return
    type error wherever the function is annotated non-str (high pyrefly hit-rate)."""
    m = re.search(r"\n(\s*)return [^\n]+", code)
    if not m:
        return None
    line = code.count("\n", 0, m.start()) + 1 + 1  # the return is on the next line
    indent = m.group(1)
    new = code[: m.start()] + f"\n{indent}return \"BUG_INJECTED\"" + code[m.end():]
    return new, line


def mut_return_none(code: str) -> tuple[str, int] | None:
    """`return <expr>` -> bare `return` (returns None; bad-return where annotated)."""
    m = re.search(r"\n(\s*)return ([^\n]+)", code)
    if not m:
        return None
    indent = m.group(1)
    line = code.count("\n", 0, m.start()) + 1 + 1
    new = code[: m.start()] + f"\n{indent}return" + code[m.end():]
    return new, line


def mut_div(code: str) -> tuple[str, int] | None:
    """`//` -> `/` (int -> float; bad-return where the fn is annotated `-> int`)."""
    return _first_replace(code, "//", "/")


def mut_int_to_str_init(code: str) -> tuple[str, int] | None:
    """A typed `x: int = 0` accumulator -> `x: int = "0"` (annotation violation)."""
    m = re.search(r"(\w+)\s*:\s*int\s*=\s*0\b", code)
    if not m:
        return None
    line = code.count("\n", 0, m.start()) + 1
    new = code[: m.start()] + f"{m.group(1)}: int = \"0\"" + code[m.end():]
    return new, line


def mut_list_append_str(code: str) -> tuple[str, int] | None:
    """`.append(x)` -> `.append(str(x))` (str into a List[int]/numeric list)."""
    m = re.search(r"\.append\(([^()]+)\)", code)
    if not m:
        return None
    line = code.count("\n", 0, m.start()) + 1
    inner = m.group(1)
    new = code[: m.start()] + f".append(str({inner}))" + code[m.end():]
    return new, line


def mut_str_to_int_arg(code: str) -> tuple[str, int] | None:
    """Wrap a `len(...)` in `str(...)` where its result feeds arithmetic — turns an
    int into a str at a use site (assignable error)."""
    m = re.search(r"\breturn (len\([^()]*\))", code)
    if not m:
        return None
    line = code.count("\n", 0, m.start()) + 1
    new = code[: m.start()] + f"return str({m.group(1)}) + 1" + code[m.end():]
    return new, line


def mut_index_str(code: str) -> tuple[str, int] | None:
    """Index a numeric expression: `return total` -> `return total[0]` (not-iterable
    / bad-index where total is a number)."""
    m = re.search(r"\n(\s*)return (\w+)\n", code)
    if not m:
        return None
    indent, var = m.group(1), m.group(2)
    line = code.count("\n", 0, m.start()) + 1 + 1
    new = code[: m.start()] + f"\n{indent}return {var}[0]\n" + code[m.end():]
    return new, line


def mut_lte(code: str) -> tuple[str, int] | None:
    for a, b in ((" <= ", " < "), (" >= ", " > "), (" < ", " <= "), (" > ", " >= ")):
        r = _first_replace(code, a, b)
        if r:
            return r
    return None


def mut_anyall(code: str) -> tuple[str, int] | None:
    for a, b in (("all(", "any("), ("any(", "all(")):
        r = _first_replace(code, a, b)
        if r:
            return r
    return None


def mut_plus_to_minus(code: str) -> tuple[str, int] | None:
    """Concatenation/accumulation operator flip `+ ` -> `- ` (str - str is a type
    error; numeric is silent and gets dropped by the filter)."""
    return _first_replace(code, " + ", " - ")


def _return_occurrences(code: str) -> list[re.Match]:
    return list(re.finditer(r"\n(\s*)return ([^\n]+)", code))


def _mk_return_wrong_at(n: int) -> Mutation:
    """Mutate the n-th `return <expr>` to a string literal (distinct example per
    return site — diversifies the corpus on multi-return functions)."""
    def _f(code: str) -> tuple[str, int] | None:
        ms = _return_occurrences(code)
        if n >= len(ms):
            return None
        m = ms[n]
        indent = m.group(1)
        line = code.count("\n", 0, m.start()) + 1 + 1
        new = code[: m.start()] + f"\n{indent}return \"BUG_INJECTED\"" + code[m.end():]
        return new, line
    return _f


def _mk_return_none_at(n: int) -> Mutation:
    """Mutate the n-th `return <expr>` to a bare `return` (returns None)."""
    def _f(code: str) -> tuple[str, int] | None:
        ms = _return_occurrences(code)
        if n >= len(ms):
            return None
        m = ms[n]
        indent = m.group(1)
        line = code.count("\n", 0, m.start()) + 1 + 1
        new = code[: m.start()] + f"\n{indent}return" + code[m.end():]
        return new, line
    return _f


# Order matters only for naming; selection is randomized per function (seeded).
MUTATIONS: dict[str, Mutation] = {
    "return_wrong_type": mut_return_wrong_type,
    "return_none": mut_return_none,
    "div_to_truediv": mut_div,
    "int_init_to_str": mut_int_to_str_init,
    "append_str": mut_list_append_str,
    "str_arg": mut_str_to_int_arg,
    "index_number": mut_index_str,
    "lte_swap": mut_lte,
    "any_all_swap": mut_anyall,
    "plus_to_minus": mut_plus_to_minus,
    # Nth-return variants diversify multi-return functions into distinct examples.
    "return_wrong_type@1": _mk_return_wrong_at(1),
    "return_wrong_type@2": _mk_return_wrong_at(2),
    "return_none@1": _mk_return_none_at(1),
    "return_none@2": _mk_return_none_at(2),
}


# ===========================================================================
# Pyrefly workspace (one warm daemon, batched files, per the timebox note)
# ===========================================================================
class PyreflyWorkspace:
    """A temp pyrefly workspace + one warm daemon. Reuses the existing client."""

    def __init__(self, pyrefly: str = DEFAULT_PYREFLY) -> None:
        self.pyrefly = pyrefly
        self.root = Path(tempfile.mkdtemp(prefix="d_sft_"))
        subprocess.run([pyrefly, "init"], cwd=str(self.root),
                       capture_output=True, text=True)
        self.daemon = PyreflyDaemon(str(self.root), pyrefly=pyrefly)
        self._n = 0

    def diagnose(self, code: str) -> list[dict]:
        """Type-check `code`; return raw pyrefly diagnostics. Fresh file per call
        so per-document state never carries across functions."""
        self._n += 1
        tgt = self.root / f"f{self._n}.py"
        tgt.write_text(code)
        self.daemon.open(str(tgt), text=code)
        raw = self.daemon.change(str(tgt), code)
        return list(raw)

    def version(self) -> str:
        try:
            out = subprocess.run([self.pyrefly, "--version"], capture_output=True,
                                 text=True, check=True)
            return out.stdout.strip()
        except Exception:
            return "unknown"

    def close(self) -> None:
        try:
            self.daemon.close()
        finally:
            import shutil
            shutil.rmtree(self.root, ignore_errors=True)


# ===========================================================================
# 4. Teacher trajectory: buggy fn -> lsp_response(diag) -> corrected fn.
#    Token granularity: we treat each whitespace-split chunk as an "agent token"
#    in teacher time (the reformat re-times to student tokens via TokenizerRate).
# ===========================================================================
_TOK_RE = re.compile(r"\S+\s*|\s+")


def _tokenize_teacher(text: str) -> list[str]:
    """Coarse teacher tokenization preserving exact text on concat (the reformat
    only needs ordering + char counts, which TokenizerRate rescales)."""
    toks = _TOK_RE.findall(text)
    return toks if toks else [text]


def build_teacher_trajectory(
    traj_id: str,
    buggy_code: str,
    fixed_code: str,
    diags: list[Diagnostic],
    edited_line: int,
) -> TeacherTrajectory:
    """Synchronous-teacher trajectory in the schema the reformat consumes.

    Layout in teacher-token time:
      [buggy agent tokens] (lsp_query, lsp_response @ end-of-buggy) [fix agent tokens]

    The lsp_response.text is the rendered sync diagnostic (what the causal gate
    strips); payload carries the normalized Diagnostic dicts that the reformat
    re-emits as the inline block.
    """
    events: list[TrajectoryEvent] = []
    idx = 0
    for t in _tokenize_teacher(buggy_code):
        events.append(TrajectoryEvent(type="agent_token", t_emit=idx, text=t))
        idx += 1
    query_idx = idx  # snapshot fires right after the buggy edit
    qid = f"{traj_id}.q0"
    events.append(TrajectoryEvent(type="lsp_query", t_emit=query_idx, text="",
                                  payload={"query_id": qid}))
    rendered = "\n".join(d.render() for d in diags)
    events.append(TrajectoryEvent(
        type="lsp_response", t_emit=query_idx, text=rendered,
        payload={"query_id": qid, "diagnostics": [d.__dict__ for d in diags],
                 "edited_line": edited_line},
    ))
    idx = query_idx + 1
    for t in _tokenize_teacher("\n" + fixed_code):
        events.append(TrajectoryEvent(type="agent_token", t_emit=idx, text=t))
        idx += 1
    return TeacherTrajectory(traj_id=traj_id, events=events,
                             teacher_tokenizer="whitespace-mock",
                             meta={"edited_line": edited_line})


# ===========================================================================
# 6. Tokenize + loss-mask with the real Qwen2.5-Coder tokenizer.
# ===========================================================================
USER_INSTRUCTION = (
    "Here is a Python function with a bug. Emit the corrected function.\n\n"
    "```python\n{buggy}\n```"
)


@dataclass
class TokenizedExample:
    input_ids: list[int]
    labels: list[int]
    condition: str
    latency_tokens: int | None
    meta: dict[str, Any]


class StudentTokenizer:
    """Wraps the Qwen2.5-Coder chat tokenizer + the loss-mask construction.

    The assistant turn is built from the InterleavedSequence: we render the agent
    stream with the diagnostic block spliced inline (exactly as `full_text()`),
    then loss-mask:
      - all prompt/instruction (user turn + chat scaffolding) tokens -> -100
      - every diagnostic-block span -> -100   (condition on, don't generate)
      - the fix agent tokens -> the loss target (kept)
    """

    def __init__(self, model: str = STUDENT_MODEL) -> None:
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    def chars_per_token(self) -> float:
        sample = (
            "def solve(items):\n    total = 0\n    for x in items:\n"
            "        total += x.value\n    return total\n"
        ) * 8
        n = len(self.tok(sample)["input_ids"])
        return len(sample) / max(n, 1)

    def encode_example(
        self, seq, buggy_code: str, condition: str, latency_tokens: int | None,
        base_meta: dict[str, Any],
    ) -> TokenizedExample:
        """Build input_ids + labels for one reformatted InterleavedSequence.

        We assemble the assistant text as: [agent tokens before fix] + inline diag
        blocks + [fix tokens], i.e. the full interleaved stream. The split point
        between "buggy preamble" and "fix" is the first agent token whose teacher
        index is past the lsp_response (we tag fix tokens at build time via meta).
        Loss is borne ONLY by the fix agent tokens.
        """
        # Build assistant string piecewise so we can map char spans -> token spans.
        # Each interleaved token contributes its text; we track which char ranges
        # are diagnostic-block (mask) vs fix-agent (loss) vs buggy-agent (mask).
        pieces: list[tuple[str, str]] = []  # (text, role) role in {"buggy","fix","diag"}
        for tk in seq.tokens:
            if tk.source == "diagnostic":
                role = "diag"
            else:
                role = "fix" if tk.meta.get("is_fix") else "buggy"
            pieces.append((tk.text, role))

        assistant_text = "".join(p[0] for p in pieces)

        # Chat-template the user turn; we tokenize the user prompt and assistant
        # separately so the entire user side is masked and only fix tokens count.
        user_content = USER_INSTRUCTION.format(buggy=buggy_code)
        prompt_ids = self.tok.apply_chat_template(
            [{"role": "user", "content": user_content}],
            add_generation_prompt=True, tokenize=True, return_dict=False,
        )

        # Tokenize the assistant text with offset mapping so we can locate the
        # diagnostic-block and fix char spans precisely in token space.
        enc = self.tok(assistant_text, return_offsets_mapping=True,
                       add_special_tokens=False)
        asst_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]

        # Char-level role map over assistant_text.
        role_at = bytearray(len(assistant_text))  # 0 buggy,1 fix,2 diag
        ROLE = {"buggy": 0, "fix": 1, "diag": 2}
        c = 0
        for text, role in pieces:
            for _ in range(len(text)):
                role_at[c] = ROLE[role]
                c += 1

        labels: list[int] = [-100] * len(prompt_ids)
        ids = list(prompt_ids)
        for tid, (a, b) in zip(asst_ids, offsets):
            ids.append(tid)
            # Determine the dominant role of this token's char span.
            if b <= a:
                labels.append(-100)
                continue
            span_roles = role_at[a:b]
            # A token is a loss target only if it is (majority) fix and contains
            # no diagnostic chars (never train to generate the diag block).
            has_diag = 2 in span_roles
            n_fix = span_roles.count(1)
            if not has_diag and n_fix * 2 >= (b - a):
                labels.append(tid)   # fix token -> loss-bearing
            else:
                labels.append(-100)  # buggy preamble or diag block -> masked

        # append EOS as a loss target so the model learns to stop after the fix
        eos = self.tok.eos_token_id
        if eos is not None:
            ids.append(eos)
            labels.append(eos)

        meta = dict(base_meta)
        meta.update({
            "n_prompt_tokens": len(prompt_ids),
            "n_assistant_tokens": len(asst_ids),
            "n_diag_spans": len(seq.diag_spans),
            "seq_len": len(ids),
        })
        return TokenizedExample(
            input_ids=ids, labels=labels, condition=condition,
            latency_tokens=latency_tokens, meta=meta,
        )


# ===========================================================================
# Tagging fix tokens on the reformatted sequence
# ===========================================================================
def _tag_fix_tokens(seq, n_buggy_teacher_tokens: int) -> None:
    """Mark agent tokens that belong to the *fix* (teacher index > the query) so
    the loss-mask can keep only them. The reformat preserves each agent token's
    teacher index in meta['teacher_idx']."""
    for tk in seq.tokens:
        if tk.source != "agent":
            continue
        tidx = tk.meta.get("teacher_idx")
        tk.meta["is_fix"] = (tidx is not None and tidx >= n_buggy_teacher_tokens)


# ===========================================================================
# Causal sanity (reuse G3 logic): the sync diag text must not leak into fix tokens
# ===========================================================================
def fix_text(seq) -> str:
    return "".join(tk.text for tk in seq.tokens
                   if tk.source == "agent" and tk.meta.get("is_fix"))


def diag_leaks_into_fix(seq, diag_render_lines: list[str]) -> bool:
    ft = fix_text(seq)
    # The rendered diagnostic message text must not appear in the fix tokens.
    return any(line and line in ft for line in diag_render_lines)


# ===========================================================================
# Main generation
# ===========================================================================
@dataclass
class GenStats:
    n_source: int = 0
    n_clean: int = 0
    n_mutation_attempts: int = 0
    n_produced_diag: int = 0
    n_examples: dict[str, int] = field(default_factory=dict)
    mutation_dist: dict[str, int] = field(default_factory=dict)
    latency_dist: dict[int, int] = field(default_factory=dict)
    seq_lens: dict[str, list[int]] = field(default_factory=dict)
    leak_failures: int = 0
    pyrefly_version: str = ""


def keep_clean(ws: PyreflyWorkspace, fns: list[SourceFn]) -> list[SourceFn]:
    kept: list[SourceFn] = []
    for fn in fns:
        try:
            raw = ws.diagnose(fn.code)
        except Exception:
            continue
        if len(raw) <= 1:
            kept.append(fn)
    return kept


def generate(args) -> GenStats:
    rng = random.Random(args.seed)
    stats = GenStats(n_examples={"A": 0, "C": 0, "D": 0},
                     seq_lens={"A": [], "C": [], "D": []})

    print("[1] loading source functions ...", flush=True)
    fns = load_source_functions(args.max_functions)
    stats.n_source = len(fns)
    print(f"    {len(fns)} source functions", flush=True)

    ws = PyreflyWorkspace(pyrefly=args.pyrefly)
    stats.pyrefly_version = ws.version()

    # Tokenizer (optional for dry runs).
    student_tok: StudentTokenizer | None = None
    rate = TokenizerRate()  # mock 4.0/4.0 fallback
    if not args.no_tokenizer:
        print("[*] loading Qwen2.5-Coder tokenizer (CPU) ...", flush=True)
        student_tok = StudentTokenizer()
        cpt = student_tok.chars_per_token()
        rate = TokenizerRate(teacher_chars_per_token=cpt, student_chars_per_token=cpt)
        print(f"    student chars/token = {cpt:.3f}", flush=True)

    print("[2] filtering to pyrefly-clean functions ...", flush=True)
    t0 = time.time()
    fns = keep_clean(ws, fns)
    stats.n_clean = len(fns)
    print(f"    {len(fns)} clean functions ({time.time()-t0:.1f}s)", flush=True)

    out_paths = {c: (OUT_DIR / c / "data.jsonl") for c in ("A", "C", "D")}
    for c in out_paths:
        out_paths[c].parent.mkdir(parents=True, exist_ok=True)
    handles = {c: open(out_paths[c], "w") for c in out_paths}
    decoded_samples: dict[str, list[str]] = {"A": [], "C": [], "D": []}

    cap = args.max_examples_per_condition
    mut_names = list(MUTATIONS.keys())

    try:
        for fi, fn in enumerate(fns):
            if cap is not None and min(stats.n_examples.values()) >= cap:
                break
            # Sample mutations for this function (seeded, deterministic order).
            order = mut_names[:]
            rng.shuffle(order)
            produced_for_fn = 0
            for mname in order:
                if produced_for_fn >= args.mutations_per_function:
                    break
                mut = MUTATIONS[mname]
                res = mut(fn.code)
                if res is None:
                    continue
                buggy, edited_line = res
                if buggy == fn.code:
                    continue
                stats.n_mutation_attempts += 1
                # 3. Real pyrefly diagnostic on the buggy version.
                try:
                    raw = ws.diagnose(buggy)
                except Exception:
                    continue
                if not raw:
                    continue  # silent mutation: drop
                recs = normalize_diagnostics(raw, EditedRegion(edited_line, edited_line))
                if not recs:
                    continue
                stats.n_produced_diag += 1
                stats.mutation_dist[mname] = stats.mutation_dist.get(mname, 0) + 1
                produced_for_fn += 1

                diags = [Diagnostic(**r) for r in recs]
                diag_lines = [d.render() for d in diags]
                traj_id = f"{fn.fn_id}|{mname}".replace("/", "_")
                n_buggy_teacher = len(_tokenize_teacher(buggy))
                traj = build_teacher_trajectory(
                    traj_id, buggy, fn.code, diags, edited_line)

                base_meta = {
                    "fn_id": fn.fn_id, "source": fn.source, "mutation": mname,
                    "edited_line": edited_line, "n_diags": len(diags),
                    "diag_codes": [d.code for d in diags],
                }

                # --- D: latency sweep ---
                lat = rng.choice(LATENCY_SWEEP)
                d_seq = reformat_to_D_interleaved(
                    traj,
                    latency_sampler=lambda r, _l=lat: float(_l),
                    student_tokenizer=rate, seed=args.seed + fi,
                )
                _tag_fix_tokens(d_seq, n_buggy_teacher)

                # --- C: sync at edit boundary ---
                c_seq = reformat_to_C_interleaved(
                    traj, student_tokenizer=rate, seed=args.seed + fi)
                _tag_fix_tokens(c_seq, n_buggy_teacher)

                # --- A: no diag block (D with diag spans removed) ---
                a_seq = reformat_to_D_interleaved(
                    traj, latency_sampler=lambda r: 0.0,
                    student_tokenizer=rate, seed=args.seed + fi)
                _tag_fix_tokens(a_seq, n_buggy_teacher)
                # strip diagnostic tokens entirely for A (matched-volume floor:
                # buggy->fix trajectory with the diagnostic removed).
                a_seq.tokens = [tk for tk in a_seq.tokens if tk.source == "agent"]
                a_seq.diag_spans = []
                a_seq._renumber()

                # Causal sanity: sync diag must not leak into the fix tokens.
                for s in (d_seq, c_seq):
                    if diag_leaks_into_fix(s, diag_lines):
                        stats.leak_failures += 1

                # 6. Tokenize + loss-mask, write per condition.
                conds = [("A", a_seq, None), ("C", c_seq, 0), ("D", d_seq, lat)]
                for cond, seq, latency in conds:
                    if cap is not None and stats.n_examples[cond] >= cap:
                        continue
                    if student_tok is not None:
                        ex = student_tok.encode_example(
                            seq, buggy, cond, latency, base_meta)
                        rec = {
                            "input_ids": ex.input_ids, "labels": ex.labels,
                            "condition": cond, "latency_tokens": latency,
                            "meta": ex.meta,
                        }
                        stats.seq_lens[cond].append(ex.meta["seq_len"])
                    else:
                        rec = {
                            "input_ids": None, "labels": None, "condition": cond,
                            "latency_tokens": latency, "meta": base_meta,
                            "assistant_text": seq.full_text(),
                        }
                    handles[cond].write(json.dumps(rec) + "\n")
                    stats.n_examples[cond] += 1
                    if cond == "D" and latency is not None:
                        stats.latency_dist[latency] = stats.latency_dist.get(latency, 0) + 1
                    # keep a few decoded samples for the summary
                    if student_tok is not None and len(decoded_samples[cond]) < 5:
                        decoded_samples[cond].append(
                            _render_masked_example(student_tok, rec))

            if fi % 50 == 0:
                print(f"    [{fi}/{len(fns)}] examples "
                      f"A={stats.n_examples['A']} C={stats.n_examples['C']} "
                      f"D={stats.n_examples['D']}", flush=True)
    finally:
        for h in handles.values():
            h.close()
        ws.close()

    _write_summary(stats, decoded_samples, out_paths)
    return stats


def _render_masked_example(student_tok: StudentTokenizer, rec: dict) -> str:
    """Decode input_ids with loss-masked spans bracketed as «MASKED:...» and
    loss-bearing (fix) tokens shown plain. Truncated for readability."""
    tok = student_tok.tok
    ids, labels = rec["input_ids"], rec["labels"]
    out: list[str] = []
    run_masked: list[int] = []
    run_loss: list[int] = []

    def flush():
        if run_masked:
            out.append("«MASK:" + tok.decode(run_masked) + "»")
            run_masked.clear()
        if run_loss:
            out.append("⟦FIX:" + tok.decode(run_loss) + "⟧")
            run_loss.clear()

    for tid, lab in zip(ids, labels):
        if lab == -100:
            if run_loss:
                flush()
            run_masked.append(tid)
        else:
            if run_masked:
                flush()
            run_loss.append(tid)
    flush()
    text = "".join(out)
    if len(text) > 2200:
        text = text[:1100] + "\n   ... [truncated] ...\n" + text[-1100:]
    return text


def _write_summary(stats: GenStats, decoded: dict[str, list[str]],
                   out_paths: dict[str, Path]) -> None:
    import statistics as st
    md: list[str] = []
    md.append("# D-SFT data-generation summary (v0.5 interleaved-async, §0.4–§0.6)\n")
    md.append(f"- pyrefly: `{stats.pyrefly_version}`")
    md.append(f"- source functions loaded: **{stats.n_source}**")
    md.append(f"- pyrefly-clean (≤1 diag) functions kept: **{stats.n_clean}**")
    md.append(f"- mutation attempts: **{stats.n_mutation_attempts}**")
    md.append(f"- mutations that produced a real diagnostic (kept): "
              f"**{stats.n_produced_diag}** "
              f"({100*stats.n_produced_diag/max(stats.n_mutation_attempts,1):.0f}% survival)")
    md.append(f"- causal-sanity leak failures (sync diag in fix tokens): "
              f"**{stats.leak_failures}**")
    md.append("")
    md.append("## Counts per condition")
    md.append("| Condition | Examples | Output |")
    md.append("|---|---:|---|")
    for c in ("A", "C", "D"):
        md.append(f"| {c} | {stats.n_examples[c]} | `{out_paths[c]}` |")
    md.append("")
    md.append("## Mutation-type distribution (kept examples)")
    md.append("| Mutation | Count |")
    md.append("|---|---:|")
    for k, v in sorted(stats.mutation_dist.items(), key=lambda x: -x[1]):
        md.append(f"| `{k}` | {v} |")
    md.append("")
    md.append("## D latency distribution (student-token offset)")
    md.append("| Latency (tokens) | Count |")
    md.append("|---:|---:|")
    for k in sorted(stats.latency_dist):
        md.append(f"| {k} | {stats.latency_dist[k]} |")
    md.append("")
    md.append("## Sequence-length stats (tokens)")
    md.append("| Condition | n | mean | median | p95 | max |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for c in ("A", "C", "D"):
        sl = stats.seq_lens[c]
        if sl:
            sl_sorted = sorted(sl)
            p95 = sl_sorted[min(len(sl) - 1, int(0.95 * len(sl)))]
            md.append(f"| {c} | {len(sl)} | {st.mean(sl):.0f} | "
                      f"{st.median(sl):.0f} | {p95} | {max(sl)} |")
        else:
            md.append(f"| {c} | 0 | – | – | – | – |")
    md.append("")
    md.append("## Loss-masking & layout validation (decode-and-eyeball)")
    md.append("Legend: `«MASK:…»` = label -100 (prompt + buggy preamble + ‹diag› "
              "block; conditioned-on, not generated). `⟦FIX:…⟧` = loss-bearing "
              "fix tokens (the only training target).")
    md.append("Confirm per condition: A has **no** ‹diag› block; C's block sits at "
              "the edit boundary; D's block sits **after** the edit (look-back); "
              "in all cases the diag block is inside a `«MASK…»` run and the fix is "
              "`⟦FIX…⟧`.")
    md.append("")
    for c in ("A", "C", "D"):
        md.append(f"### Condition {c} — {len(decoded[c])} decoded examples")
        for i, s in enumerate(decoded[c]):
            md.append(f"**{c} example {i}:**")
            md.append("```")
            md.append(s)
            md.append("```")
            md.append("")
    (OUT_DIR / "summary.md").write_text("\n".join(md) + "\n")
    print(f"[done] summary -> {OUT_DIR/'summary.md'}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pyrefly", default=DEFAULT_PYREFLY)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-functions", type=int, default=None)
    p.add_argument("--max-examples-per-condition", type=int, default=3000)
    p.add_argument("--mutations-per-function", type=int, default=6)
    p.add_argument("--no-tokenizer", action="store_true",
                   help="skip the HF tokenizer (dry run; emits assistant_text only)")
    args = p.parse_args()

    os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = generate(args)
    print(f"\n[summary] A={stats.n_examples['A']} C={stats.n_examples['C']} "
          f"D={stats.n_examples['D']} | survival "
          f"{stats.n_produced_diag}/{stats.n_mutation_attempts} | "
          f"leaks={stats.leak_failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
