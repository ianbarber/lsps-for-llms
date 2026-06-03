#!/usr/bin/env python3
"""Phase F smoke — load model, install flex+GQA, run a few in-place decode rows,
print outputs. Validates the in-place KV cache appends correctly and attention
reads the sliced valid region without error."""
from __future__ import annotations
import importlib.util, os, sys, time
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REVISION = "54c7451bfcccecc233fad91affa68563d1de9d66"
SNAP = os.path.expanduser(
    f"~/.cache/huggingface/hub/models--JonasGeiping--stream-qwen3-8b/snapshots/{REVISION}")
FDIR = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_f/patched")


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    if SNAP not in sys.path:
        sys.path.insert(0, SNAP)
    si = load_mod(FDIR / "stream_inference_inplace.py", "si_inplace")
    print("[load]", SNAP, flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        SNAP, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    tok = AutoTokenizer.from_pretrained(SNAP, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    silence_token = si.detect_silence_token(tok)
    si.install_flex_attention(model)
    print("[smoke] running 40 decode rows (smaller buffer)...", flush=True)
    rows = []
    t0 = time.perf_counter()
    g = si.generate(model, tok, "Write a function to add two numbers.", silence_token,
                    max_rows=40, warm_start=False, temperature=0.0, max_context_rows=512)
    for ri, row, isp in g:
        if isp:
            continue
        rows.append(row)
        if len(rows) >= 40:
            break
    torch.cuda.synchronize()
    res = si.collect_result(tok, silence_token, [(0, r, False) for r in rows])
    print(f"[smoke] {len(rows)} rows in {time.perf_counter()-t0:.1f}s", flush=True)
    print("[smoke] Output channel:", repr(res.output[:200]), flush=True)
    print("[smoke] Analytical channel:", repr(res.stream('Analytical')[:120]), flush=True)
    assert res.output.strip(), "empty output channel — in-place cache likely broken"
    print("[smoke] PASS", flush=True)


if __name__ == "__main__":
    main()
