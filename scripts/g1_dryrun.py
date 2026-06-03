"""G1 dry-run: load each model once, run HumanEval(3) + MBPP(3), write outputs.

Validates the harness plumbing end-to-end without running the full G1.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Allow `from harness ...` imports.
sys.path.insert(0, "/home/ianbarber/Projects/Streams")

from harness.single_stream_eval import (  # noqa: E402
    extract_humaneval_completion,
    extract_mbpp_code,
    generate_stream,
    generate_vanilla,
    load_humaneval,
    load_mbpp_sanitized,
    load_model,
    run_with_timeout,
)

OUT_DIR = Path("/home/ianbarber/Projects/Streams/runs/g1_prep")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_bench(model, tok, gen_fn, bench: str, *, max_new_tokens: int) -> dict:
    if bench == "humaneval":
        items = load_humaneval(limit=3)
        extract = extract_humaneval_completion

        def wrap(prompt: str, completion: str) -> str:
            return prompt + completion
    else:
        items = load_mbpp_sanitized(limit=3)
        extract = extract_mbpp_code

        def wrap(prompt: str, completion: str) -> str:
            return completion

    rows = []
    n_pass = 0
    for i, item in enumerate(items):
        t0 = time.time()
        gen = gen_fn(item["prompt"], max_new_tokens)
        t_gen = time.time() - t0
        completion = extract(gen)
        code = wrap(item["prompt"], completion)
        ok, err = run_with_timeout(code, item["test"], item.get("entry_point"), timeout=10.0)
        n_pass += int(ok)
        rows.append({
            "task_id": item["task_id"],
            "pass": ok,
            "error": err,
            "gen_time_s": round(t_gen, 2),
            "completion_preview": completion[:400],
            "raw_generation_preview": gen[:400],
        })
        print(f"    [{i+1}/{len(items)}] {item['task_id']:20s} pass={ok} t={t_gen:.1f}s err={err}")
    return {"n_pass": n_pass, "n_total": len(items), "rows": rows}


def write_bench_report(path: Path, header: str, vanilla: dict, stream: dict) -> None:
    lines = [header, ""]
    for label, res in (("vanilla Qwen3-8B", vanilla), ("stream-qwen3-8b", stream)):
        lines.append(f"## {label}")
        lines.append(f"  n_pass = {res['n_pass']} / {res['n_total']}")
        for r in res["rows"]:
            lines.append(f"  - {r['task_id']:20s} pass={r['pass']} t={r['gen_time_s']}s err={r['error']}")
            lines.append(f"      raw_gen[:200]={r['raw_generation_preview'][:200]!r}")
            lines.append(f"      completion[:200]={r['completion_preview'][:200]!r}")
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    results = {}

    # ---- vanilla ----
    print("=" * 70)
    print("Loading Qwen/Qwen3-8B (vanilla)")
    t0 = time.time()
    model_v, tok_v = load_model("Qwen/Qwen3-8B", "vanilla")
    print(f"  loaded in {time.time() - t0:.1f}s")

    def gen_v(prompt: str, n: int) -> str:
        return generate_vanilla(model_v, tok_v, prompt, max_new_tokens=n)

    print("  HumanEval dry-run (3 problems):")
    results["vanilla_humaneval"] = run_bench(model_v, tok_v, gen_v, "humaneval", max_new_tokens=512)
    print("  MBPP dry-run (3 problems):")
    results["vanilla_mbpp"] = run_bench(model_v, tok_v, gen_v, "mbpp", max_new_tokens=512)

    # Free GPU mem before loading the stream model
    import torch
    del model_v, tok_v
    torch.cuda.empty_cache()

    # ---- stream ----
    print("=" * 70)
    print("Loading JonasGeiping/stream-qwen3-8b (stream)")
    t0 = time.time()
    model_s, tok_s = load_model("JonasGeiping/stream-qwen3-8b", "stream")
    print(f"  loaded in {time.time() - t0:.1f}s")

    def gen_s(prompt: str, n: int) -> str:
        return generate_stream(model_s, tok_s, prompt, max_new_tokens=n)

    print("  HumanEval dry-run (3 problems):")
    results["stream_humaneval"] = run_bench(model_s, tok_s, gen_s, "humaneval", max_new_tokens=256)
    print("  MBPP dry-run (3 problems):")
    results["stream_mbpp"] = run_bench(model_s, tok_s, gen_s, "mbpp", max_new_tokens=256)

    # ---- write reports ----
    write_bench_report(
        OUT_DIR / "humaneval_dryrun.txt",
        "# G1 prep: HumanEval 3-problem dry-run\n"
        "# Both models, T=0 (greedy), max_new_tokens=512 vanilla / max_rows=256 stream\n",
        results["vanilla_humaneval"],
        results["stream_humaneval"],
    )
    write_bench_report(
        OUT_DIR / "mbpp_dryrun.txt",
        "# G1 prep: MBPP-sanitized 3-problem dry-run\n"
        "# Both models, T=0 (greedy), max_new_tokens=512 vanilla / max_rows=256 stream\n",
        results["vanilla_mbpp"],
        results["stream_mbpp"],
    )
    (OUT_DIR / "dryrun_raw.json").write_text(json.dumps(results, indent=2))
    print("\nReports written:")
    print(f"  {OUT_DIR / 'humaneval_dryrun.txt'}")
    print(f"  {OUT_DIR / 'mbpp_dryrun.txt'}")
    print(f"  {OUT_DIR / 'dryrun_raw.json'}")


if __name__ == "__main__":
    main()
