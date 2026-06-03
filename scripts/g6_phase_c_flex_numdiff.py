#!/usr/bin/env python3
"""Phase C Route 2 — numerical-diff probe to localize identity divergence.

Runs ONE prefill + a few decode steps, comparing the FlexAttention BlockMask path
vs the Phase B dense additive mask path at the LOGITS level (same weights, same
inputs). Tells us whether divergence is (a) a mask-semantics bug (large, structured
diff) or (b) benign bf16 fp noise (tiny diff, rare argmax flips near ties).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "JonasGeiping/stream-qwen3-8b"
OUT = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_c_flex/numdiff.json")
PATCH_B = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_b/patched/stream_inference_phase_b.py")
FLEX_DIR = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_c_flex/patched")


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    snap = snapshot_download(MODEL_ID)
    if snap not in sys.path:
        sys.path.insert(0, snap)
    b = load_mod(PATCH_B, "si_phase_b")
    flex = load_mod(FLEX_DIR / "stream_inference_flex.py", "si_flex")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    silence_token = b.detect_silence_token(tok)
    device = model.get_input_embeddings().weight.device
    C = 10

    # Build a small flat context: 8 rows of random-ish real tokens (use the warm
    # start prefill so they're plausible) then one decode row.
    prefill_rows = b.build_system_prompt_prefill(tok, silence_token, num_channels=C)[:8]
    n_prefill = len(prefill_rows)
    flat = [t for row in prefill_rows for t in row]
    N = n_prefill * C
    input_ids = torch.tensor([flat], device=device, dtype=torch.long)
    position_ids = torch.tensor([[r for r in range(n_prefill) for _ in range(C)]],
                                device=device, dtype=torch.long)
    channel_ids = torch.tensor([[c for _ in range(n_prefill) for c in range(C)]],
                               device=device, dtype=torch.long)

    # --- Dense SDPA path (Phase B mask) ---
    rows_idx = torch.arange(N, device=device) // C
    allowed = (rows_idx.unsqueeze(0) < rows_idx.unsqueeze(1)) | torch.eye(N, dtype=torch.bool, device=device)
    dense_mask = torch.where(allowed, torch.tensor(0.0, device=device),
                             torch.tensor(-1e4, device=device)).to(torch.bfloat16).view(1, 1, N, N)
    with torch.no_grad():
        out_dense = model(input_ids=input_ids,
                          attention_mask={"full_attention": dense_mask, "sliding_attention": dense_mask},
                          position_ids=position_ids, use_cache=False, channel_ids=channel_ids)
    logits_dense = out_dense.logits[0].float()  # [N, V]

    # --- Flex path (BlockMask) ---
    flex.install_flex_attention(model)
    block_mask = flex.build_block_mask(C, q_len=N, kv_len=N, q_offset=0, device=device)
    with torch.no_grad():
        out_flex = model(input_ids=input_ids,
                         attention_mask={"full_attention": block_mask, "sliding_attention": block_mask},
                         position_ids=position_ids, use_cache=False, channel_ids=channel_ids)
    logits_flex = out_flex.logits[0].float()

    diff = (logits_dense - logits_flex).abs()
    argmax_dense = logits_dense.argmax(dim=-1)
    argmax_flex = logits_flex.argmax(dim=-1)
    flips = (argmax_dense != argmax_flex)

    # per-row diff breakdown
    per_row = []
    for r in range(n_prefill):
        sl = slice(r * C, (r + 1) * C)
        rd = diff[sl]
        per_row.append({
            "row": r,
            "max_abs_diff": rd.max().item(),
            "mean_abs_diff": rd.mean().item(),
            "argmax_flips": int(flips[sl].sum().item()),
        })

    result = {
        "context": f"{n_prefill} prefill rows x {C} channels = {N} tokens, full prefill (no cache)",
        "logits_max_abs_diff": diff.max().item(),
        "logits_mean_abs_diff": diff.mean().item(),
        "argmax_total_flips": int(flips.sum().item()),
        "argmax_total_positions": int(flips.numel()),
        "argmax_flip_rate": flips.float().mean().item(),
        "per_row": per_row,
    }
    # If flips exist, report the logit gap at flipped positions (tie-breaking noise check)
    if flips.any():
        fidx = flips.nonzero(as_tuple=True)[0]
        gaps = []
        for pos in fidx.tolist()[:20]:
            top2 = logits_dense[pos].topk(2).values
            gaps.append((top2[0] - top2[1]).item())
        result["flipped_top1_top2_gaps_dense"] = gaps
        result["flipped_max_gap"] = max(gaps) if gaps else None

    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
