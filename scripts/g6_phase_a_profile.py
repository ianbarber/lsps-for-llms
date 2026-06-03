#!/usr/bin/env python3
"""Phase A5 — profile a 50-row decode and identify the hot path.

Uses torch.profiler with CPU+CUDA activities. Captures top-K ops by self CUDA
time and self CPU time. Also breaks down time by Python-side mask building vs
model forward vs sampling using event records inside the row loop.

Outputs:
  runs/g6_phase_a/profile.json
  runs/g6_phase_a/profile_chrome_trace.json (chrome trace)
  runs/g6_phase_a/profile_summary.md
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")
MODEL_ID = "JonasGeiping/stream-qwen3-8b"
OUT_DIR = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_a")

PROMPT = "Write a Python function that reverses a linked list in place."
N_PROFILE_ROWS = 50
N_WARMUP_ROWS = 20


def load_model():
    snap = snapshot_download(MODEL_ID)
    if snap not in sys.path:
        sys.path.insert(0, snap)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="auto")
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def time_phases(model, tok, silence_token, n_rows):
    """Instrument the row loop manually to measure mask-build / forward / sample
    wall-clock fractions. Uses CUDA events for the forward (GPU work).
    """
    from stream_inference import sample_top_p
    C = getattr(model.config, "num_channels", 10)
    device = model.get_input_embeddings().weight.device

    # Seed row
    SEED_WORDS = ["-", "-", " thinking", " checking", " feeling", " relating",
                  " asking", " drifting", " watching", " integrating"]
    row = []
    for w in SEED_WORDS[:C]:
        if w == "-":
            row.append(silence_token)
        else:
            toks = tok.encode(w, add_special_tokens=False)
            row.append(toks[0])

    past_kv = None
    all_cached_ids = torch.tensor([], device=device, dtype=torch.long)

    mask_build_ns = 0
    forward_ns = 0
    sample_ns = 0
    overhead_ns = 0
    rows_done = 0

    for row_idx in range(1, n_rows + 1):
        t0 = time.perf_counter_ns()
        input_ids = torch.tensor([row], device=device, dtype=torch.long)
        position_ids = torch.full((1, C), row_idx - 1, device=device, dtype=torch.long)
        channel_ids = torch.arange(C, device=device, dtype=torch.long).unsqueeze(0)
        if past_kv is None:
            mask = torch.full((1, 1, C, C), -1e4, device=device, dtype=torch.bfloat16)
            for i in range(C):
                mask[0, 0, i, i] = 0.0
        else:
            cached_len = past_kv.get_seq_length()
            total = cached_len + C
            mask = torch.zeros(1, 1, C, total, device=device, dtype=torch.bfloat16)
            for i in range(C):
                for j in range(C):
                    if i != j:
                        mask[0, 0, i, cached_len + j] = -1e4
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()

        outputs = model(
            input_ids=input_ids,
            attention_mask={"full_attention": mask, "sliding_attention": mask},
            position_ids=position_ids,
            past_key_values=past_kv,
            use_cache=True,
            channel_ids=channel_ids,
        )
        torch.cuda.synchronize()
        t2 = time.perf_counter_ns()

        past_kv = outputs.past_key_values
        all_cached_ids = torch.cat([all_cached_ids, input_ids[0]])
        logits = outputs.logits[0]
        next_row = [silence_token] + [sample_top_p(logits[c], 0.0, 0.95, 20) for c in range(1, C)]
        row = next_row
        torch.cuda.synchronize()
        t3 = time.perf_counter_ns()

        mask_build_ns += t1 - t0
        forward_ns += t2 - t1
        sample_ns += t3 - t2
        rows_done += 1

    total = mask_build_ns + forward_ns + sample_ns
    return {
        "rows": rows_done,
        "mask_build_ms": mask_build_ns / 1e6,
        "forward_ms": forward_ns / 1e6,
        "sample_ms": sample_ns / 1e6,
        "total_ms": total / 1e6,
        "mask_build_pct": mask_build_ns / total if total else 0,
        "forward_pct": forward_ns / total if total else 0,
        "sample_pct": sample_ns / total if total else 0,
        "ms_per_row": total / rows_done / 1e6 if rows_done else 0,
    }


def run_profiler(model, tok, silence_token):
    """Run torch.profiler over N_PROFILE_ROWS."""
    from stream_inference import generate as gen_fn
    from torch.profiler import profile, ProfilerActivity

    # warmup so we don't profile first-call kernels
    g = gen_fn(model, tok, PROMPT, silence_token, max_rows=N_WARMUP_ROWS,
               warm_start=False, temperature=0.0)
    for _ in range(N_WARMUP_ROWS):
        try:
            next(g)
        except StopIteration:
            break

    print(f"[profile] running profiler for {N_PROFILE_ROWS} rows...", flush=True)
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities, record_shapes=False, profile_memory=False) as prof:
        g = gen_fn(model, tok, PROMPT, silence_token, max_rows=N_PROFILE_ROWS,
                   warm_start=False, temperature=0.0)
        rows_seen = 0
        for r in g:
            rows_seen += 1
            if rows_seen >= N_PROFILE_ROWS:
                break

    # Export chrome trace (might be large)
    trace_path = OUT_DIR / "profile_chrome_trace.json"
    try:
        prof.export_chrome_trace(str(trace_path))
        print(f"[profile] wrote {trace_path}", flush=True)
    except Exception as e:
        print(f"[profile] chrome trace export failed: {e}", flush=True)

    # Extract top ops by CUDA time and CPU time
    key_averages = prof.key_averages()
    # Sort by CUDA time then CPU time
    rows_cuda = sorted(key_averages, key=lambda x: x.device_time_total, reverse=True)[:15]
    rows_cpu = sorted(key_averages, key=lambda x: x.self_cpu_time_total, reverse=True)[:15]

    def fmt(rows):
        out = []
        for r in rows:
            out.append({
                "name": r.key,
                "count": r.count,
                "cpu_total_us": float(r.cpu_time_total),
                "self_cpu_us": float(r.self_cpu_time_total),
                "cuda_total_us": float(r.device_time_total),
                "self_cuda_us": float(getattr(r, "self_device_time_total", 0)),
            })
        return out

    return {
        "rows_profiled": rows_seen,
        "top_by_cuda": fmt(rows_cuda),
        "top_by_cpu": fmt(rows_cpu),
        "summary_table": key_averages.table(sort_by="cuda_time_total", row_limit=20),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, tok = load_model()
    from stream_inference import detect_silence_token
    silence_token = detect_silence_token(tok)

    # Phase budget breakdown (mask build vs forward vs sample)
    print(f"[phases] timing 50 rows manually...", flush=True)
    # warmup once first
    _ = time_phases(model, tok, silence_token, n_rows=20)
    phase_breakdown = time_phases(model, tok, silence_token, n_rows=N_PROFILE_ROWS)
    print(f"[phases] {phase_breakdown}", flush=True)

    # Now full profiler
    prof_result = run_profiler(model, tok, silence_token)

    out = {
        "phase_breakdown": phase_breakdown,
        "rows_profiled": prof_result["rows_profiled"],
        "top_by_cuda_time": prof_result["top_by_cuda"],
        "top_by_cpu_time": prof_result["top_by_cpu"],
    }
    (OUT_DIR / "profile.json").write_text(json.dumps(out, indent=2))

    # Human-readable summary
    md = []
    md.append("# Phase A5 — profile summary")
    md.append("")
    md.append(f"Rows profiled: {prof_result['rows_profiled']}")
    md.append("")
    md.append("## Manual phase breakdown (50 rows)")
    md.append("")
    pb = phase_breakdown
    md.append(f"- Mask build: {pb['mask_build_ms']:.1f} ms ({pb['mask_build_pct']:.1%})")
    md.append(f"- Model forward: {pb['forward_ms']:.1f} ms ({pb['forward_pct']:.1%})")
    md.append(f"- Sampling: {pb['sample_ms']:.1f} ms ({pb['sample_pct']:.1%})")
    md.append(f"- Total: {pb['total_ms']:.1f} ms ({pb['ms_per_row']:.2f} ms/row)")
    md.append("")
    md.append("## Top 10 ops by CUDA self-time")
    md.append("")
    md.append("| Op | Count | self CUDA (us) | total CUDA (us) |")
    md.append("|---|---|---|---|")
    for op in prof_result["top_by_cuda"][:10]:
        md.append(f"| `{op['name']}` | {op['count']} | {op['self_cuda_us']:.0f} | {op['cuda_total_us']:.0f} |")
    md.append("")
    md.append("## Top 10 ops by CPU self-time")
    md.append("")
    md.append("| Op | Count | self CPU (us) | total CPU (us) |")
    md.append("|---|---|---|---|")
    for op in prof_result["top_by_cpu"][:10]:
        md.append(f"| `{op['name']}` | {op['count']} | {op['self_cpu_us']:.0f} | {op['cpu_total_us']:.0f} |")
    md.append("")
    md.append("## Full table (sort by CUDA total)")
    md.append("")
    md.append("```")
    md.append(prof_result["summary_table"])
    md.append("```")
    (OUT_DIR / "profile_summary.md").write_text("\n".join(md))
    print(f"[done] wrote profile.json and profile_summary.md")


if __name__ == "__main__":
    main()
