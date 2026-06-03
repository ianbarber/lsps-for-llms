#!/usr/bin/env python3
"""Phase D — explicit torch.cuda.CUDAGraph path (D4 option b).

reduce-overhead failed to CUDA-graph (mutated-input KV cache) AND broke greedy
identity (inductor numerics). This path uses an explicit CUDAGraph captured over
the UNCOMPILED eager forward: replay runs the original kernels, so output is
bit-identical to the eager decoder (== Phase B), and we get true graph replay.

Order (one model load):
  D5  identity: compiled-graph Phase D (use_cudagraph=True) vs eager Phase B,
      greedy, 5 prompts x 30 rows. Bit-identity gate. -> identity.json
  D6  throughput-by-context: {256,1024,4096,8192}, capture overhead, ms/row,
      tok/s, peak mem. -> throughput_by_context.json
  capture/recompile info -> compile.json (explicit-graph variant)
"""
from __future__ import annotations
import importlib.util, json, os, sys, time
from pathlib import Path

os.environ["HF_HOME"] = "/home/ianbarber/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/home/ianbarber/.cache/huggingface/hub"

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "JonasGeiping/stream-qwen3-8b"
OUT = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_d")
PATCH_D = OUT / "patched" / "stream_inference_phase_d.py"
PATCH_B = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_b/patched/stream_inference_phase_b.py")
PROMPTS = [
    "Write a Python function that reverses a linked list in place.",
    "Explain how a B-tree differs from a binary search tree.",
    "Refactor this code to use a context manager: open('f.txt'); read(); close().",
    "What is the time complexity of merge sort, and why?",
    "Sketch a unit test for a function that adds two integers.",
]

def load_patched(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def greedy(gen_fn, model, tok, silence, prompt, n, extra):
    rows = []
    for ri, row, isp in gen_fn(model, tok, prompt, silence, max_rows=n+5,
                               warm_start=False, temperature=0.0, **extra):
        if isp: continue
        rows.append(list(row))
        if len(rows) >= n: break
    return rows

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    snap = snapshot_download(MODEL_ID)
    if snap not in sys.path: sys.path.insert(0, snap)
    modD = load_patched(PATCH_D, "si_d"); modB = load_patched(PATCH_B, "si_b")
    print("[load]...", flush=True); t0=time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto"); model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    silence = modD.detect_silence_token(tok)
    print(f"[load] {time.perf_counter()-t0:.1f}s silence={silence}", flush=True)

    # ---- reference: eager Phase B ----
    print("\n[ref] eager Phase B reference (5 prompts x 30 rows)...", flush=True)
    ref = {i: greedy(modB.generate, model, tok, silence, p, 30, {})
           for i, p in enumerate(PROMPTS)}
    for i in ref: print(f"[ref] prompt {i}: {len(ref[i])} rows", flush=True)

    # ---- D5: identity (explicit CUDA graph vs eager Phase B) ----
    print("\n[D5] identity: CUDAGraph Phase D vs eager Phase B (greedy, 30 rows)...", flush=True)
    extra = {"max_context_rows": 512, "use_cudagraph": True}
    identity = {"path": "explicit_cudagraph", "per_prompt": [], "all_match": True}
    for i, p in enumerate(PROMPTS):
        got = greedy(modD.generate, model, tok, silence, p, 30, extra)
        exp = ref[i]; n = min(len(got), len(exp))
        match = got[:n]==exp[:n] and len(got)==len(exp)
        fd = next((r for r in range(n) if got[r]!=exp[r]), None)
        identity["per_prompt"].append({"prompt_idx": i, "match": bool(match),
            "ref_rows": len(exp), "got_rows": len(got), "first_divergence_row": fd})
        identity["all_match"] = identity["all_match"] and match
        print(f"[D5] prompt {i}: match={match} (ref={len(exp)} got={len(got)} div={fd})", flush=True)
    (OUT/"identity.json").write_text(json.dumps(identity, indent=2))
    print(f"[D5] ALL_MATCH={identity['all_match']}", flush=True)

    # ---- D6: throughput-by-context ----
    print("\n[D6] throughput-by-context (multi-stream packing-2, CUDAGraph)...", flush=True)
    contexts = [256, 1024, 4096, 8192]
    SHORT_ROWS = 80
    results = {}
    for n in contexts:
        mcr = int(n*1.3)+128
        extra_n = {"max_context_rows": mcr, "use_cudagraph": True}
        print(f"[ctx n={n}] buffer_rows={mcr}: capture...", flush=True)
        torch.cuda.synchronize(); tw=time.perf_counter(); rr=0
        for _ in modD.generate(model, tok, PROMPTS[0], silence, max_rows=40,
                               warm_start=False, temperature=0.0, **extra_n):
            rr+=1
            if rr>=40: break
        torch.cuda.synchronize(); capture_s=time.perf_counter()-tw
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        t0=time.perf_counter(); prev=t0; row_times=[]; prod=total=rr=0
        for ri,row,isp in modD.generate(model, tok, PROMPTS[1], silence,
                max_rows=SHORT_ROWS, warm_start=False, temperature=0.0, **extra_n):
            if isp: continue
            torch.cuda.synchronize(); now=time.perf_counter()
            row_times.append(now-prev); prev=now; rr+=1; total+=len(row)
            for c in (1,2):
                if row[c]!=silence: prod+=1
            if rr>=SHORT_ROWS: break
        torch.cuda.synchronize(); peak=torch.cuda.max_memory_allocated()/1e9
        tail=sorted(row_times[-40:]) if len(row_times)>=40 else sorted(row_times)
        steady_ms=(tail[len(tail)//2]*1000.0) if tail else None
        prod_frac=prod/rr if rr else 0.0
        derived=(prod_frac/(steady_ms/1000.0)) if steady_ms else 0.0
        rps=(1000.0/steady_ms) if steady_ms else 0.0
        results[str(n)]={"target_tokens":n,"buffer_rows":mcr,"capture_seconds":capture_s,
            "rows_timed":rr,"steady_state_ms_per_row":steady_ms,"rows_per_sec":rps,
            "productive_per_row_ch12":prod_frac,"derived_multi_tok_s":derived,
            "peak_memory_gb":peak}
        print(f"[ctx n={n}] steady {steady_ms:.1f} ms/row | rows/s {rps:.2f} | "
              f"prod/row {prod_frac:.2f} | derived {derived:.2f} tok/s | peak {peak:.2f} GB "
              f"| capture {capture_s:.1f}s", flush=True)

    out={"decoder":"phase_d_explicit_cudagraph","method":"explicit torch.cuda.CUDAGraph over eager forward",
         "short_rows":SHORT_ROWS,"by_context":results}
    (OUT/"throughput_by_context.json").write_text(json.dumps(out, indent=2))

    # compile.json: explicit-graph capture summary (recompiles N/A; no dynamo)
    (OUT/"compile.json").write_text(json.dumps({
        "path": "explicit_cudagraph",
        "dynamo_recompiles": 0,
        "note": "explicit torch.cuda.CUDAGraph; dynamo not used on the decode forward. "
                "reduce-overhead variant (cursor tensorized) gave dynamo recompiles=0 "
                "but skipped cudagraphs due to mutated KV inputs and broke identity; "
                "see recompiles.log.",
        "capture_seconds_by_context": {k: v["capture_seconds"] for k,v in results.items()},
    }, indent=2))
    print("\n[done] wrote identity.json, throughput_by_context.json, compile.json", flush=True)

if __name__ == "__main__":
    main()
