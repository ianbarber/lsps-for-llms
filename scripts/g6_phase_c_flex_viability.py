#!/usr/bin/env python3
"""Phase C Route 2 (FlexAttention) — F1 viability gate on GB10 sm_12.1.

Standalone smoke test: does torch.nn.attention.flex_attention compile + run
+ produce correct output vs an SDPA reference on this hardware?

Tensor shape [B=1, H=32, S=512, D=128], bf16, simple causal mask_mod.
If FlexAttention's Triton codegen does NOT support sm_12.1, this fails fast.

Writes runs/g6_phase_c_flex/viability.json. Idempotent.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")

import torch
import torch.nn.functional as F

OUT = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_c_flex/viability.json")


def main():
    result = {
        "device_capability": list(torch.cuda.get_device_capability()),
        "device_name": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "shape": [1, 32, 512, 128],
        "dtype": "bfloat16",
    }
    try:
        import triton
        result["triton_version"] = triton.__version__
    except Exception as e:
        result["triton_version"] = f"import_error: {e!r}"

    try:
        from torch.nn.attention.flex_attention import (
            flex_attention,
            create_block_mask,
        )
        result["flex_import"] = "ok"
    except Exception as e:
        result["flex_import"] = f"FAILED: {e!r}"
        result["viable"] = False
        result["blocker"] = "import"
        OUT.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    device = "cuda"
    B, H, S, D = 1, 32, 512, 128
    torch.manual_seed(0)
    q = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)

    def causal_mask_mod(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    # --- (a) compile ---
    t0 = time.perf_counter()
    try:
        block_mask = create_block_mask(causal_mask_mod, B=None, H=None, Q_LEN=S, KV_LEN=S, device=device)
        result["create_block_mask"] = "ok"
        result["block_mask_repr"] = str(block_mask)[:500]
    except Exception as e:
        result["create_block_mask"] = f"FAILED: {e!r}"
        result["create_block_mask_traceback"] = traceback.format_exc()
        result["viable"] = False
        result["blocker"] = "create_block_mask"
        OUT.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    flex_compiled = torch.compile(flex_attention)

    # --- (b) run (compiled) ---
    try:
        torch.cuda.synchronize()
        tc0 = time.perf_counter()
        out_flex = flex_compiled(q, k, v, block_mask=block_mask)
        torch.cuda.synchronize()
        result["compile_plus_first_run_seconds"] = time.perf_counter() - tc0
        result["run_compiled"] = "ok"
        result["out_shape"] = list(out_flex.shape)
    except Exception as e:
        result["run_compiled"] = f"FAILED: {e!r}"
        result["run_compiled_traceback"] = traceback.format_exc()
        result["viable"] = False
        result["blocker"] = "compiled_run"
        OUT.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    # eager flex too (no compile) — informative
    try:
        out_flex_eager = flex_attention(q, k, v, block_mask=block_mask)
        result["run_eager"] = "ok"
    except Exception as e:
        result["run_eager"] = f"FAILED: {e!r}"

    # --- (c) correctness vs SDPA causal reference ---
    with torch.no_grad():
        out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    diff = (out_flex.float() - out_ref.float()).abs()
    result["max_abs_diff_vs_sdpa"] = diff.max().item()
    result["mean_abs_diff_vs_sdpa"] = diff.mean().item()
    # bf16 attention: tolerance ~2e-2 is generous; typically <1e-2
    result["correct"] = bool(diff.max().item() < 5e-2)

    # --- score_mod variant smoke (additive bias path) ---
    try:
        def neg_inf_offdiag_score(score, b, h, q_idx, kv_idx):
            return score  # identity score_mod; just confirm the codegen path
        out_sm = flex_compiled(q, k, v, score_mod=neg_inf_offdiag_score, block_mask=block_mask)
        result["score_mod_path"] = "ok"
    except Exception as e:
        result["score_mod_path"] = f"FAILED: {e!r}"

    result["total_seconds"] = time.perf_counter() - t0
    result["viable"] = bool(
        result.get("run_compiled") == "ok" and result.get("correct", False)
    )
    result["blocker"] = None if result["viable"] else "correctness_or_run"

    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
