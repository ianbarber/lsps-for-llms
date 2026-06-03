#!/usr/bin/env python3
"""Phase C Route 2 — F4 output identity gate.

Greedy (T=0) decode, fixed seed, 5 prompts x 30 rows. The FlexAttention decoder
must produce bit-identical token sequences to the Phase B (SDPA) decoder.

Loads ONE model, runs the Phase B generator (SDPA dense mask), then installs the
flex patch and runs the Flex generator, and compares per-row tokens.

Writes runs/g6_phase_c_flex/identity.json. Idempotent.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "JonasGeiping/stream-qwen3-8b"
OUT = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_c_flex/identity.json")
PATCH_B = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_b/patched/stream_inference_phase_b.py")
FLEX_DIR = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_c_flex/patched")

PROMPTS = [
    "Write a Python function that reverses a linked list in place.",
    "Explain how a B-tree differs from a binary search tree.",
    "Refactor this code to use a context manager: open('f.txt'); read(); close().",
    "What is the time complexity of merge sort, and why?",
    "Sketch a unit test for a function that adds two integers.",
]
N_ROWS = 30


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_rows(gen_fn, model, tok, silence_token, prompt, n_rows):
    rows = []
    g = gen_fn(model, tok, prompt, silence_token, max_rows=n_rows + 5,
               warm_start=False, temperature=0.0)
    for row_idx, row, is_prefill in g:
        if is_prefill:
            continue
        rows.append(list(row))
        if len(rows) >= n_rows:
            break
    return rows


def main():
    snap = snapshot_download(MODEL_ID)
    if snap not in sys.path:
        sys.path.insert(0, snap)

    b = load_mod(PATCH_B, "si_phase_b")
    flex = load_mod(FLEX_DIR / "stream_inference_flex.py", "si_flex")

    print("[load] loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    silence_token = b.detect_silence_token(tok)

    # 1) Reference: Phase B (SDPA) — patch NOT installed.
    print("[ref] running Phase B (SDPA) reference...", flush=True)
    ref_rows = {}
    for i, p in enumerate(PROMPTS):
        torch.manual_seed(0)
        t0 = time.perf_counter()
        ref_rows[i] = run_rows(b.generate, model, tok, silence_token, p, N_ROWS)
        print(f"[ref] prompt {i}: {len(ref_rows[i])} rows in {time.perf_counter()-t0:.1f}s", flush=True)

    # 2) Candidate: FlexAttention — install patch.
    print("[flex] installing flex attention patch...", flush=True)
    AttnClass = flex.install_flex_attention(model)
    cand_rows = {}
    for i, p in enumerate(PROMPTS):
        torch.manual_seed(0)
        t0 = time.perf_counter()
        cand_rows[i] = run_rows(flex.generate, model, tok, silence_token, p, N_ROWS)
        print(f"[flex] prompt {i}: {len(cand_rows[i])} rows in {time.perf_counter()-t0:.1f}s", flush=True)

    # 3) Compare
    per_prompt = []
    all_identical = True
    total_tokens = 0
    total_mismatches = 0
    for i in range(len(PROMPTS)):
        r = ref_rows[i]
        c = cand_rows[i]
        n = min(len(r), len(c))
        mism = []
        for ri in range(n):
            for ci in range(len(r[ri])):
                total_tokens += 1
                if ci < len(c[ri]) and r[ri][ci] != c[ri][ci]:
                    total_mismatches += 1
                    mism.append({"row": ri, "channel": ci,
                                 "ref": r[ri][ci], "flex": c[ri][ci]})
        identical = (len(r) == len(c)) and len(mism) == 0
        all_identical = all_identical and identical
        per_prompt.append({
            "prompt_idx": i,
            "ref_rows": len(r),
            "flex_rows": len(c),
            "mismatches": len(mism),
            "first_mismatches": mism[:10],
            "verdict": "identical" if identical else "DIVERGED",
        })
        print(f"[cmp] prompt {i}: {'IDENTICAL' if identical else 'DIVERGED'} "
              f"({len(mism)} mism)", flush=True)

    out = {
        "harness": f"{len(PROMPTS)} prompts x {N_ROWS} rows, greedy T=0, seed=0, "
                   "Phase B (SDPA) vs FlexAttention BlockMask, same model instance.",
        "n_rows": N_ROWS,
        "total_tokens_compared": total_tokens,
        "total_mismatches": total_mismatches,
        "argmax_flip_rate": total_mismatches / max(total_tokens, 1),
        "per_prompt": per_prompt,
        "all_identical": all_identical,
        "verdict": "PASS" if all_identical else "FAIL",
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\n[done] all_identical={all_identical} "
          f"mismatch_rate={out['argmax_flip_rate']:.2e}", flush=True)
    print(f"[done] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
