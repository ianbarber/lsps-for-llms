"""G1 fairness probe: validate chat-format prompting, fence extraction, and T=0
determinism on BOTH models on a tiny sample (2 HumanEval problems), before the
full G1 run. Loads each model once.

Run synchronously (no detach). Writes runs/g1/fairness_probe.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/ianbarber/Projects/Streams")

import torch  # noqa: E402

from harness.single_stream_eval import (  # noqa: E402
    load_humaneval,
    load_model,
    build_chat_prompt_humaneval,
    extract_code_completion,
    generate_vanilla_chat,
    generate_stream,
    run_with_timeout,
    set_chat_tokenizer,
)

OUT = Path("/home/ianbarber/Projects/Streams/runs/g1")
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    items = load_humaneval(limit=2)
    report = {"vanilla": {}, "stream": {}}

    # ---------- vanilla ----------
    print("Loading vanilla Qwen/Qwen3-8B ...")
    t0 = time.time()
    mv, tv = load_model("Qwen/Qwen3-8B", "vanilla")
    set_chat_tokenizer(tv)  # render ChatML with the (identical) Qwen3 template
    print(f"  loaded {time.time()-t0:.0f}s")
    vrows = []
    for it in items:
        prompt = build_chat_prompt_humaneval(it["prompt"], it["entry_point"])
        g1 = generate_vanilla_chat(mv, tv, prompt, max_new_tokens=512)
        comp = extract_code_completion(g1, it["prompt"], it["entry_point"])
        ok, err = run_with_timeout(comp, it["test"], it["entry_point"])
        vrows.append({"task_id": it["task_id"], "pass": ok, "err": err,
                      "raw_head": g1[:300], "comp_head": comp[:300]})
        print(f"  vanilla {it['task_id']} pass={ok} err={err}")
    report["vanilla"]["rows"] = vrows
    del mv, tv
    torch.cuda.empty_cache()

    # ---------- stream ----------
    print("Loading stream JonasGeiping/stream-qwen3-8b ...")
    t0 = time.time()
    ms, ts = load_model("JonasGeiping/stream-qwen3-8b", "stream")
    print(f"  loaded {time.time()-t0:.0f}s")
    srows = []
    det = []
    for k, it in enumerate(items):
        # stream model takes plain user text (its chat tuning builds the assistant turn)
        instr = build_chat_prompt_humaneval(it["prompt"], it["entry_point"], for_stream=True)
        g1 = generate_stream(ms, ts, instr, max_new_tokens=320)
        comp = extract_code_completion(g1, it["prompt"], it["entry_point"])
        ok, err = run_with_timeout(comp, it["test"], it["entry_point"])
        srows.append({"task_id": it["task_id"], "pass": ok, "err": err,
                      "raw_head": g1[:400], "comp_head": comp[:300]})
        print(f"  stream {it['task_id']} pass={ok} err={err}")
        # T=0 determinism: run the FIRST problem twice, assert identical raw output
        if k == 0:
            g2 = generate_stream(ms, ts, instr, max_new_tokens=320)
            det = {"task_id": it["task_id"], "identical": g1 == g2,
                   "len1": len(g1), "len2": len(g2),
                   "first_diff": next((i for i in range(min(len(g1), len(g2))) if g1[i] != g2[i]), None)}
            print(f"  T=0 determinism: identical={det['identical']}")
    report["stream"]["rows"] = srows
    report["stream"]["determinism"] = det

    (OUT / "fairness_probe.json").write_text(json.dumps(report, indent=2))
    print("wrote", OUT / "fairness_probe.json")


if __name__ == "__main__":
    main()
