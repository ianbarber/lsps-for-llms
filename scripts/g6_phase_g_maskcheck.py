#!/usr/bin/env python3
"""Phase G — CPU-only mask-equivalence check (no model, no GPU lock).

Phase G builds the BlockMask over the FULL pre-grown buffer (KV_LEN = max_cols)
and adds a future-mask term (kv_idx < kv_valid) to exclude the unwritten region.
This check materializes the Phase G mask_mod over the FULL kv index range and
verifies, for prefill + 5 decode rows:
  1. columns [0, kv_valid)  : allow pattern == Phase B dense (block-causal + diag),
  2. columns [kv_valid, MAX): ALL masked (no leak from the unwritten buffer),
  3. no query row is fully masked (each keeps its own diagonal -> no NaN risk).
all_match must be True.
"""
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import torch

GQA_DIR = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_g/patched")
sys.path.insert(0, str(GQA_DIR))
spec = importlib.util.spec_from_file_location("fag_g", GQA_DIR / "flex_attention_gqa.py")
fag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fag)

C = 10
MAXCOLS = 512  # small stand-in for the full pre-grown buffer (must be >= kv_valid)
OUT = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_g/maskcheck.json")


def flex_allow_matrix(q_len, kv_len, q_offset, kv_valid):
    mm = fag.make_stream_mask_mod(C, q_offset, kv_valid)
    q = torch.arange(q_len); kv = torch.arange(kv_len)
    qi = q.view(-1, 1).expand(q_len, kv_len)
    ki = kv.view(1, -1).expand(q_len, kv_len)
    return mm(0, 0, qi, ki)


def dense_allow_prefill(N):
    rows_idx = torch.arange(N) // C
    return (rows_idx.unsqueeze(0) < rows_idx.unsqueeze(1)) | torch.eye(N, dtype=torch.bool)


def check_full_buffer(flex_full, dense_valid, q_len, kv_valid, maxcols):
    """flex_full: [q_len, maxcols] allow matrix over the FULL buffer.
       dense_valid: [q_len, kv_valid] reference over the written region."""
    written = flex_full[:, :kv_valid]
    future = flex_full[:, kv_valid:maxcols]
    written_mism = int((written != dense_valid).sum().item())
    future_leak = int(future.sum().item())  # any True in the future region = leak
    fully_masked_rows = int((~flex_full.any(dim=1)).sum().item())
    return {
        "written_region_mismatches": written_mism,
        "future_region_leaks": future_leak,
        "fully_masked_query_rows": fully_masked_rows,
        "match": written_mism == 0 and future_leak == 0 and fully_masked_rows == 0,
    }


def main():
    results = {"note": "Phase G full-buffer mask_mod (future-mask kv_idx<kv_valid). "
                       "Checks: written region == Phase B dense; future region fully "
                       "masked (no leak); no fully-masked query row (no NaN)."}
    # --- prefill: q_offset=0, kv_valid = N, full buffer = MAXCOLS ---
    n_prefill = 8
    N = n_prefill * C
    flex_pf = flex_allow_matrix(N, MAXCOLS, q_offset=0, kv_valid=N)
    dense_pf = dense_allow_prefill(N)
    results["prefill"] = {"N": N, "max_cols": MAXCOLS,
                          **check_full_buffer(flex_pf, dense_pf, N, N, MAXCOLS)}

    # --- decode rows: q_offset=cur, kv_valid = cur+C, full buffer = MAXCOLS ---
    decode = []
    for r in range(1, 6):
        cur = r * C
        kv_valid = cur + C
        flex_d = flex_allow_matrix(C, MAXCOLS, q_offset=cur, kv_valid=kv_valid)
        dense_d = torch.zeros(C, kv_valid, dtype=torch.bool)
        dense_d[:, :cur] = True
        for i in range(C):
            dense_d[i, cur + i] = True
        chk = check_full_buffer(flex_d, dense_d, C, kv_valid, MAXCOLS)
        decode.append({"decode_row": r, "cursor": cur, "kv_valid": kv_valid, **chk})
    results["decode"] = decode
    results["all_match"] = (results["prefill"]["match"] and all(d["match"] for d in decode))
    OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
