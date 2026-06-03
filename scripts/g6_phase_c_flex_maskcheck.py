#!/usr/bin/env python3
"""Phase C Route 2 — CPU-only mask-equivalence check (no model, no GPU lock).

Materializes the FlexAttention mask_mod as a boolean allow-matrix and compares it
element-wise to the Phase B dense additive mask's allow pattern, for both the
full-prefill case and several decode steps. If these differ, the mask_mod is
wrong (a real bug). If they match, identity divergence must be numerical.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

FLEX_DIR = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_c_flex/patched")
sys.path.insert(0, str(FLEX_DIR))
import importlib.util
spec = importlib.util.spec_from_file_location("fap", FLEX_DIR / "flex_attention_patch.py")
fap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fap)

C = 10
OUT = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_c_flex/maskcheck.json")


def flex_allow_matrix(q_len, kv_len, q_offset):
    """Evaluate the mask_mod elementwise -> bool [q_len, kv_len] (True=attend)."""
    mm = fap.make_stream_mask_mod(C, q_offset)
    q = torch.arange(q_len)
    kv = torch.arange(kv_len)
    qi = q.view(-1, 1).expand(q_len, kv_len)
    ki = kv.view(1, -1).expand(q_len, kv_len)
    return mm(0, 0, qi, ki)


def dense_allow_prefill(N):
    rows_idx = torch.arange(N) // C
    return (rows_idx.unsqueeze(0) < rows_idx.unsqueeze(1)) | torch.eye(N, dtype=torch.bool)


def main():
    results = {}

    # --- Full prefill: N = 8 rows ---
    n_prefill = 8
    N = n_prefill * C
    flex_pf = flex_allow_matrix(N, N, 0)
    dense_pf = dense_allow_prefill(N)
    mism = (flex_pf != dense_pf)
    results["prefill"] = {
        "N": N, "mismatches": int(mism.sum().item()),
        "match": bool(mism.sum().item() == 0),
    }

    # --- Decode steps: cached_len = r*C, q_len=C, kv_len=cached_len+C ---
    decode = []
    for r in range(1, 6):
        cached = r * C
        kv_len = cached + C
        flex_d = flex_allow_matrix(C, kv_len, cached)  # [C, kv_len]
        # Phase B decode mask: cache block all-visible (0), peer_block diagonal-only.
        # allow[i, j] = True if j < cached (cache, prior rows) OR j == cached+i (self)
        dense_d = torch.zeros(C, kv_len, dtype=torch.bool)
        dense_d[:, :cached] = True
        for i in range(C):
            dense_d[i, cached + i] = True
        mism_d = (flex_d != dense_d)
        decode.append({
            "decode_row": r, "cached_len": cached, "kv_len": kv_len,
            "mismatches": int(mism_d.sum().item()),
            "match": bool(mism_d.sum().item() == 0),
        })
    results["decode"] = decode
    results["all_match"] = (results["prefill"]["match"]
                            and all(d["match"] for d in decode))

    OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
