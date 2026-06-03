"""G1 full run: evaluate ONE model on HumanEval + MBPP, save per-problem JSON.

Loads the model once. Vanilla -> batched HF generate. Stream -> batched stream
decode (harness/batched_stream_decode). Chat-instruction prompting + fence-aware
extraction for both (the G1 fairness path).

Usage:
  g1_run.py vanilla  runs/g1   [--limit N] [--batch 16]
  g1_run.py stream   runs/g1   [--limit N] [--batch 8]
  g1_run.py vanilla  runs/g1 --bench humaneval   # restrict to one bench
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/ianbarber/Projects/Streams")

import torch  # noqa: E402

from harness.single_stream_eval import (  # noqa: E402
    load_humaneval, load_mbpp_sanitized, load_model,
    build_chat_prompt_humaneval, build_chat_prompt_mbpp,
    extract_code_completion, run_with_timeout, set_chat_tokenizer,
    generate_vanilla_chat,
)
from harness.batched_stream_decode import batched_stream_generate  # noqa: E402

MODELS = {
    "vanilla": "Qwen/Qwen3-8B",
    "stream": "JonasGeiping/stream-qwen3-8b",
}


def mbpp_entry_name(test_list) -> str:
    """Extract the function name MBPP tests call (best-effort from first assert)."""
    import re
    for t in test_list:
        m = re.search(r"assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", t)
        if m:
            return m.group(1)
    return ""


def build_items(bench: str):
    if bench == "humaneval":
        raw = load_humaneval()
        items = []
        for it in raw:
            items.append({
                "task_id": it["task_id"], "test": it["test"],
                "entry_point": it["entry_point"], "orig_prompt": it["prompt"],
                "instr_v": build_chat_prompt_humaneval(it["prompt"], it["entry_point"]),
                "instr_s": build_chat_prompt_humaneval(it["prompt"], it["entry_point"], for_stream=True),
            })
        return items
    else:
        raw = load_mbpp_sanitized()
        items = []
        # reload original to get test_list for entry name + raw description
        from datasets import load_dataset
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        for it, ex in zip(raw, ds):
            desc = ex["prompt"].strip()
            tl = ex["test_list"]
            entry = mbpp_entry_name(tl)
            test_hint = tl[0] if tl else ""
            items.append({
                "task_id": it["task_id"], "test": it["test"],
                "entry_point": entry, "orig_prompt": "",
                "instr_v": build_chat_prompt_mbpp(desc, test_hint),
                "instr_s": build_chat_prompt_mbpp(desc, test_hint, for_stream=True),
            })
        return items


def run_bench(mode, model, tok, bench, items, batch, max_new, limit):
    if limit:
        items = items[:limit]
    rows = []
    n_pass = 0
    t0 = time.time()
    eos = tok.eos_token_id
    # process in batches
    for start in range(0, len(items), batch):
        chunk = items[start:start + batch]
        tg = time.time()
        if mode == "vanilla":
            gens = [generate_vanilla_chat(model, tok, c["instr_v"], max_new_tokens=max_new)
                    for c in chunk]
        else:
            gens = batched_stream_generate(
                model, tok, [c["instr_s"] for c in chunk],
                max_rows=max_new, eos_token_id=eos)
        dt = (time.time() - tg) / max(len(chunk), 1)
        for c, g in zip(chunk, gens):
            comp = extract_code_completion(g, c["orig_prompt"], c["entry_point"])
            ok, err = run_with_timeout(comp, c["test"], c["entry_point"])
            n_pass += int(ok)
            rows.append({"task_id": c["task_id"], "pass": ok, "error": err,
                         "gen_time_s": round(dt, 1), "completion": comp,
                         "raw_generation": g})
        done = start + len(chunk)
        print(f"  [{bench}/{mode}] {done}/{len(items)} pass_so_far={n_pass} "
              f"({dt:.0f}s/prob)", flush=True)
    total = time.time() - t0
    return rows, n_pass, len(items), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["vanilla", "stream"])
    ap.add_argument("outdir")
    ap.add_argument("--bench", choices=["humaneval", "mbpp", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    batch = args.batch or (16 if args.mode == "vanilla" else 8)
    max_new = args.max_new or (512 if args.mode == "vanilla" else 320)

    print(f"[g1_run] mode={args.mode} bench={args.bench} batch={batch} max_new={max_new}")
    t0 = time.time()
    model, tok = load_model(MODELS[args.mode], args.mode)
    set_chat_tokenizer(tok)  # tok carries the (shared) Qwen3 ChatML template
    print(f"[g1_run] model loaded in {time.time()-t0:.0f}s", flush=True)

    benches = ["humaneval", "mbpp"] if args.bench == "both" else [args.bench]
    for bench in benches:
        items = build_items(bench)
        rows, n_pass, n, total = run_bench(
            args.mode, model, tok, bench, items, batch, max_new, args.limit)
        pass_at_1 = n_pass / max(n, 1)
        summary = {"model": MODELS[args.mode], "mode": args.mode, "bench": bench,
                   "n_problems": n, "n_pass": n_pass, "pass_at_1": pass_at_1,
                   "total_time_s": round(total, 1), "batch": batch, "max_new": max_new}
        (out / f"{bench}_{args.mode}.json").write_text(json.dumps(
            {"summary": summary, "results": rows}, indent=2))
        print(f"[g1_run] {bench}/{args.mode}: pass@1 = {n_pass}/{n} = {pass_at_1:.4f} "
              f"(wall={total:.0f}s)  -> {bench}_{args.mode}.json", flush=True)


if __name__ == "__main__":
    main()
