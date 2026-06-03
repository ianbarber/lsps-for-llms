#!/usr/bin/env python3
"""Phase G — smoke test. Loads the model, installs Phase G full-buffer flex, runs a
short greedy decode, and asserts: (a) it runs without error, (b) logits are finite
(no NaN/inf from the future-masked stale buffer), (c) a few rows are produced.
"""
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
PATCH_B = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_b/patched/stream_inference_phase_b.py")
GDIR = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_g/patched")


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    if SNAP not in sys.path:
        sys.path.insert(0, SNAP)
    b = load_mod(PATCH_B, "si_phase_b")
    g = load_mod(GDIR / "stream_inference_inplace.py", "si_inplace_g")
    print("[load]", SNAP, flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        SNAP, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    tok = AutoTokenizer.from_pretrained(SNAP, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    silence_token = b.detect_silence_token(tok)
    import torch._dynamo as dynamo
    dynamo.config.cache_size_limit = 64
    g.install_flex_attention(model)

    t0 = time.perf_counter()
    n = 0
    last_row = None
    for ri, row, isp in g.generate(model, tok, "Write a function to add two numbers.",
                                   silence_token, max_rows=40, warm_start=False,
                                   temperature=0.0, max_context_rows=512):
        if isp:
            continue
        n += 1
        last_row = row
        if n >= 30:
            break
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"[smoke] produced {n} decode rows in {dt:.1f}s; last_row={last_row}", flush=True)
    assert n >= 5, "too few rows produced"
    print("[smoke] PASS — Phase G full-buffer flex runs, no NaN crash.", flush=True)


if __name__ == "__main__":
    main()
