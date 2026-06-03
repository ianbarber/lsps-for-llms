#!/usr/bin/env python3
"""Phase F — CPU-only mask-equivalence check (no model, no GPU lock).

The Phase F mask_mod is the SAME flex_attention_gqa.py as Phase E. The in-place
cache changes only HOW K/V is stored/returned (slice of a pre-grown buffer), not
the mask geometry: for a decode row the cache returns K[:, :, :cur+C, :] (kv_len
= cur+C) and the mask is built with q_offset=cur, kv_len=cur+C — identical to the
Phase E dynamic case. So re-confirm the cross-stream allow pattern matches the
Phase B dense additive mask for prefill + 5 decode rows. all_match must be True.
"""
from __future__ import annotations
import importlib.util, json
from pathlib import Path
import torch

FDIR = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_f/patched")
spec = importlib.util.spec_from_file_location("fag_f", FDIR / "flex_attention_gqa.py")
fag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fag)

C = 10
OUT = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_f/maskcheck.json")


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
    results = {"note": "Phase F in-place-cache mask vs Phase B dense allow pattern. "
                       "mask_mod identical to Phase E; in-place cache returns a "
                       "valid-region slice K[:, :, :cur+C, :] so kv_len=cur+C matches "
                       "the BlockMask exactly (q_offset=cur). Prefill + 5 decode rows."}
    n_prefill = 8
    N = n_prefill * C
    flex_pf = flex_allow_matrix(N, N, 0)
    dense_pf = dense_allow_prefill(N)
    mism = (flex_pf != dense_pf)
    results["prefill"] = {"N": N, "mismatches": int(mism.sum().item()),
                          "match": bool(mism.sum().item() == 0)}
    decode = []
    for r in range(1, 6):
        cached = r * C  # cursor before this decode row's append
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
