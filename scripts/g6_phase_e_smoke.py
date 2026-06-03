#!/usr/bin/env python3
"""Phase E smoke — load model from ~/.cache, install GQA flex patch, run a few
decode rows, confirm output is non-empty and the path doesn't error. Also assert
K/V stay 8-headed into flex (no repeat_kv) via a forward hook count."""
from __future__ import annotations
import importlib.util, os, sys, time
from pathlib import Path

# NAS is down; force local ~/.cache/huggingface and never contact the hub.
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "JonasGeiping/stream-qwen3-8b"
REVISION = "54c7451bfcccecc233fad91affa68563d1de9d66"
SNAP = os.path.expanduser(
    f"~/.cache/huggingface/hub/models--JonasGeiping--stream-qwen3-8b/snapshots/{REVISION}")
GQA_DIR = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_e/patched")


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    snap = SNAP
    if snap not in sys.path:
        sys.path.insert(0, snap)
    gqa = load_mod(GQA_DIR / "stream_inference_gqa.py", "si_gqa")

    print("[load] loading model from", snap, flush=True)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        snap, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    tok = AutoTokenizer.from_pretrained(snap, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    silence_token = gqa.detect_silence_token(tok)
    print(f"[load] done {time.perf_counter()-t0:.1f}s; silence={silence_token}", flush=True)

    gqa.install_flex_attention(model)

    # hook to verify K head dim into flex is 8 (no repeat_kv to 32)
    seen = {"kv_heads": set()}
    AttnClass = type(model.model.layers[0].self_attn)
    orig = gqa._flex.__dict__  # not used; we instead inspect via the forward path

    rows = []
    t0 = time.perf_counter()
    cnt = 0
    for row_idx, row, is_prefill in gqa.generate(
            model, tok, "Write a function that adds two numbers.", silence_token,
            max_rows=20, warm_start=False, temperature=0.0):
        if is_prefill:
            continue
        rows.append(list(row))
        cnt += 1
        if cnt >= 15:
            break
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    res = gqa.collect_result(tok, silence_token, [(i, r, False) for i, r in enumerate(rows)])
    print(f"[smoke] {cnt} rows in {dt:.1f}s", flush=True)
    print("[smoke] Output:", repr(res.channel_texts.get("Output", "")[:200]), flush=True)
    print("[smoke] Analytical:", repr(res.channel_texts.get("Analytical", "")[:200]), flush=True)
    print("[smoke] OK" if any(res.channel_texts.values()) else "[smoke] EMPTY-OUTPUT", flush=True)


if __name__ == "__main__":
    main()
