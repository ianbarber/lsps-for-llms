"""Validate batched_stream_generate against the reference stream_generate.

For 3 HumanEval prompts (chat-instruction form), compare:
  (a) reference model.stream_generate(...).output  (single sequence)
  (b) batched_stream_generate([prompt], B=1)         (our batched path, B=1)
  (c) batched_stream_generate([p0,p1,p2], B=3)       (real batch)

We do NOT require byte-identity (any batched mask/reduction reorder + our
seed-row handling can differ); we require that the extracted CODE passes/fails
the SAME on the gate's exec scorer for each problem. That is the property G1
actually depends on. Reports raw + extracted + pass for each path.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/ianbarber/Projects/Streams")

import torch  # noqa: E402

from harness.single_stream_eval import (  # noqa: E402
    load_humaneval, load_model, build_chat_prompt_humaneval,
    extract_code_completion, generate_stream, run_with_timeout,
)
from harness.batched_stream_decode import batched_stream_generate  # noqa: E402

OUT = Path("/home/ianbarber/Projects/Streams/runs/g1")
OUT.mkdir(parents=True, exist_ok=True)
MAX_ROWS = 320


def main() -> None:
    items = load_humaneval(limit=3)
    print("loading stream model ...")
    t0 = time.time()
    model, tok = load_model("JonasGeiping/stream-qwen3-8b", "stream")
    print(f"  loaded {time.time()-t0:.0f}s")
    eos = tok.eos_token_id

    instrs = [build_chat_prompt_humaneval(it["prompt"], it["entry_point"], for_stream=True)
              for it in items]

    # (a) reference single-sequence
    ref = []
    for it, instr in zip(items, instrs):
        t = time.time()
        g = generate_stream(model, tok, instr, max_new_tokens=MAX_ROWS)
        dt = time.time() - t
        comp = extract_code_completion(g, it["prompt"], it["entry_point"])
        ok, err = run_with_timeout(comp, it["test"], it["entry_point"])
        ref.append({"task": it["task_id"], "pass": ok, "err": err, "t": round(dt, 1),
                    "raw_head": g[:200]})
        print(f"  REF  {it['task_id']} pass={ok} t={dt:.0f}s")

    # (b) batched B=1
    b1 = []
    for it, instr in zip(items, instrs):
        t = time.time()
        g = batched_stream_generate(model, tok, [instr], max_rows=MAX_ROWS, eos_token_id=eos)[0]
        dt = time.time() - t
        comp = extract_code_completion(g, it["prompt"], it["entry_point"])
        ok, err = run_with_timeout(comp, it["test"], it["entry_point"])
        b1.append({"task": it["task_id"], "pass": ok, "err": err, "t": round(dt, 1),
                   "raw_head": g[:200]})
        print(f"  B=1  {it['task_id']} pass={ok} t={dt:.0f}s")

    # (c) batched B=3
    t = time.time()
    gs = batched_stream_generate(model, tok, instrs, max_rows=MAX_ROWS, eos_token_id=eos)
    dt = time.time() - t
    b3 = []
    for it, g in zip(items, gs):
        comp = extract_code_completion(g, it["prompt"], it["entry_point"])
        ok, err = run_with_timeout(comp, it["test"], it["entry_point"])
        b3.append({"task": it["task_id"], "pass": ok, "err": err, "raw_head": g[:200]})
        print(f"  B=3  {it['task_id']} pass={ok}")
    print(f"  B=3 batch wall={dt:.0f}s for {len(items)} problems "
          f"({dt/len(items):.0f}s/problem)")

    report = {"ref": ref, "batched_b1": b1, "batched_b3": b3,
              "ref_passes": [r["pass"] for r in ref],
              "b1_passes": [r["pass"] for r in b1],
              "b3_passes": [r["pass"] for r in b3],
              "b3_wall_s": round(dt, 1)}
    (OUT / "batched_validate.json").write_text(json.dumps(report, indent=2))
    match_b1 = report["ref_passes"] == report["b1_passes"]
    match_b3 = report["ref_passes"] == report["b3_passes"]
    print(f"\nref_passes={report['ref_passes']}  b1={report['b1_passes']} "
          f"(match={match_b1})  b3={report['b3_passes']} (match={match_b3})")
    print("wrote", OUT / "batched_validate.json")


if __name__ == "__main__":
    main()
