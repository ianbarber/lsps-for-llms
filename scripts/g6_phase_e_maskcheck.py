#!/usr/bin/env python3
"""Phase E — CPU-only mask-equivalence check (no model, no GPU lock).

Materializes the Phase E GQA flex mask_mod as a boolean allow-matrix and compares
it element-wise to the Phase B dense additive mask's allow pattern, for the full
prefill case and several decode steps. The mask_mod is byte-identical to Route 2
(Phase E only changes the K/V grouping, not the mask), so this re-confirms the
cross-stream semantics under the Phase E patch module. all_match must be True.
"""
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import torch

GQA_DIR = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_e/patched")
sys.path.insert(0, str(GQA_DIR))
spec = importlib.util.spec_from_file_location("fag", GQA_DIR / "flex_attention_gqa.py")
fag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fag)

C = 10
OUT = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_e/maskcheck.json")


def flex_allow_matrix(q_len, kv_len, q_offset):
    mm = fag.make_stream_mask_mod(C, q_offset)
    q = torch.arange(q_len); kv = torch.arange(kv_len)
    qi = q.view(-1, 1).expand(q_len, kv_len)
    ki = kv.view(1, -1).expand(q_len, kv_len)
    return mm(0, 0, qi, ki)


def dense_allow_prefill(N):
    rows_idx = torch.arange(N) // C
    return (rows_idx.unsqueeze(0) < rows_idx.unsqueeze(1)) | torch.eye(N, dtype=torch.bool)


def main():
    results = {"note": "Phase E GQA mask_mod vs Phase B dense allow pattern. "
                       "mask_mod is identical to Route 2; enable_gqa changes only "
                       "K/V grouping, not the mask. Head-agnostic (B=None,H=None)."}
    n_prefill = 8
    N = n_prefill * C
    flex_pf = flex_allow_matrix(N, N, 0)
    dense_pf = dense_allow_prefill(N)
    mism = (flex_pf != dense_pf)
    results["prefill"] = {"N": N, "mismatches": int(mism.sum().item()),
                          "match": bool(mism.sum().item() == 0)}
    decode = []
    for r in range(1, 6):
        cached = r * C
        kv_len = cached + C
        flex_d = flex_allow_matrix(C, kv_len, cached)
        dense_d = torch.zeros(C, kv_len, dtype=torch.bool)
        dense_d[:, :cached] = True
        for i in range(C):
            dense_d[i, cached + i] = True
        mism_d = (flex_d != dense_d)
        decode.append({"decode_row": r, "cached_len": cached, "kv_len": kv_len,
                       "mismatches": int(mism_d.sum().item()),
                       "match": bool(mism_d.sum().item() == 0)})
    results["decode"] = decode
    results["all_match"] = (results["prefill"]["match"] and all(d["match"] for d in decode))
    OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
