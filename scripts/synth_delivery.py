#!/usr/bin/env python3
"""Delivery-timing runner (ledger C37): WHEN should a type diagnostic reach a coding agent?

Six arms over the 14-task synthetic multi-site type-cascade suite
(`scripts/synth_tasks_delivery.py`), through the streaming agent with MockEnv and real
Pyrefly. The mechanism under test is a multi-site type-error cascade: `pytest` reveals
broken sites one at a time (arm A must grind serially: fix -> retest -> next crash) while
`pyrefly` reveals all sites at once. What differs between arms is only WHEN the checker's
output is handed to the model.

  arm      invocation
  A        --conds A
  C-lazy   --conds C
  C-eager  --conds C --c-eager
  D-naive  --conds D --debounce 24 --pause-align --announce-lsp
  D-plain  --conds D --debounce 24 --pause-align
  D-gate   --conds D --debounce 24 --pause-align --syntax-gate

The full argparse config is written into the FINAL json (`"config"`), not only into the
`.partial` checkpoint. The original runner recorded it only in the checkpoint, which is why
arm provenance for the June artifacts had to be recovered out of git
(runs/agent/synth_delivery_provenance.json).

The system prompt is pinned to SYS_LINE_DELIVERY below, a verbatim copy of the agent's June
`SYS_LINE`. The agent's live `SYS_LINE` has since gained a `<defn>` advertisement and lost
the static-analyzer sentence; pinning keeps every arm on the prompt the committed results
were produced under, and insulates this experiment from later prompt edits.

Usage: synth_delivery.py [out.json] [--conds A,C,D] [--names n1,n2] [--seeds K]
                         [--seed-start S] [--temp T] [--model ID] [--max-new T]
"""
import os, sys, json, time, argparse
os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Verbatim `SYS_LINE` from scaffold/stream_agent.py @ 779aa5c^ — the prompt every committed
# C37 rollout was produced under. Passed through `sys_override`, so the agent's evolving
# default prompt cannot silently redefine these arms.
SYS_LINE_DELIVERY = """You are a coding agent fixing a bug in a Python repository. The bug is in file
`{file}`, shown below with line numbers (`NNN| code`). Work iteratively.

To EDIT, replace a range of lines by emitting EXACTLY:
<edit path="{file}" lines="START-END">
<the new code for those lines, WITHOUT line-number prefixes>
</edit>
START-END are inclusive 1-based line numbers from the numbered view. Include proper
indentation. You choose the range — one line or many. Do not wrap the code in ``` fences.
To READ another file for context: <read path="pkg/other.py"/>
After editing, emit <test/> to RUN THE TESTS; you'll get the results, then a fresh numbered
view of the file (line numbers may have shifted — always use the latest view). Keep iterating:
edit -> <test/> -> fix -> <test/> until tests pass, then emit <done/>. Reason briefly between
actions. A static analyzer may also surface diagnostics; use them to catch mistakes early."""

# The six arms, as agent kwargs. `condition` selects the channel; the rest select the timing.
ARMS = {
    "A":       dict(condition="A"),
    "C-lazy":  dict(condition="C"),
    "C-eager": dict(condition="C", c_eager=True),
    "D-naive": dict(condition="D", debounce=24, pause_align=True, announce_lsp=True),
    "D-plain": dict(condition="D", debounce=24, pause_align=True),
    "D-gate":  dict(condition="D", debounce=24, pause_align=True, syntax_gate=True),
}
# The delivery event each arm must emit; None means "no diagnostic events at all".
ARM_WITNESS = {"A": None, "C-lazy": "diag_sync_queued", "C-eager": "diag_eager",
               "D-naive": "diag_debounced", "D-plain": "diag_debounced", "D-gate": "diag_debounced"}


def build_prompt(task):
    code = task["code"]
    numbered = "\n".join(f"{i+1:>3}| {ln}" for i, ln in enumerate(code.splitlines()))
    return (f"Fix the bug(s) in this Python module so the test below passes.\n\n"
            f"Module `sol.py`:\n{numbered}\n\n"
            f"The test that must pass (do NOT edit the test; it is the spec):\n"
            f"```python\n{task['test']}\n```\n"
            f"Make line-range edits, then run <test/>.")


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="runs/agent/synth_delivery.json")
    ap.add_argument("--conds", default="A,C,D")
    ap.add_argument("--names", default=None, help="comma subset of task names")
    ap.add_argument("--seeds", type=int, default=1, help="sampled rollouts per (task,cond) when temp>0")
    ap.add_argument("--seed-start", type=int, default=0, help="first seed index (offset for fresh seeds)")
    ap.add_argument("--temp", type=float, default=0.0, help="0 = greedy (deterministic, seeds ignored)")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--max-new", type=int, default=1400)
    ap.add_argument("--latency", type=int, default=8)
    ap.add_argument("--debounce", type=int, default=0, help="D: settle tokens before re-querying (0=immediate)")
    ap.add_argument("--pause-align", action="store_true", help="D: deliver at a newline/pause")
    ap.add_argument("--announce-lsp", action="store_true", help="D: tell the model LSP feedback is inline")
    ap.add_argument("--c-eager", action="store_true", help="C: post-edit hook (deliver diag immediately) vs batched at yield")
    ap.add_argument("--syntax-gate", action="store_true", help="D: only deliver live diag when the file parses")
    ap.add_argument("--rich-signal", action="store_true", help="append go-to-def/hover context to each diagnostic")
    return ap.parse_args(argv)


def main(argv=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scaffold.stream_agent import StreamAgent
    from scaffold.mock_env import MockEnv
    from scripts.synth_tasks_delivery import TASKS

    A = parse_args(argv)
    tasks = TASKS if not A.names else [t for t in TASKS if t["name"] in set(A.names.split(","))]
    conds = A.conds.split(",")
    n_seeds = 1 if A.temp == 0 else A.seeds

    print(f"[load] {A.model}{' + '+A.adapter if A.adapter else ''}  temp={A.temp} seeds={n_seeds}", flush=True)
    tok = AutoTokenizer.from_pretrained(A.model)
    model = AutoModelForCausalLM.from_pretrained(A.model, dtype=torch.bfloat16, device_map="auto")
    if A.adapter:
        from peft import PeftModel; model = PeftModel.from_pretrained(model, A.adapter)
    model = model.eval()

    # Recorded in every artifact this script writes: the exact flags, so an arm can never
    # again be identified by its filename alone.
    config = dict(vars(A), runner="scripts/synth_delivery.py",
                  harness="scaffold/stream_agent.py",
                  tasks="scripts/synth_tasks_delivery.py",
                  sys_prompt="SYS_LINE_DELIVERY (== stream_agent.SYS_LINE @ 779aa5c^)",
                  n_tasks=len(tasks))
    agg = {c: [] for c in conds}
    os.makedirs(os.path.dirname(A.out) or ".", exist_ok=True)

    def dump(path, summary=None):
        json.dump({"model": A.model, "adapter": A.adapter, "temp": A.temp, "config": config,
                   "summary": summary, "rows": agg}, open(path, "w"), indent=(2 if summary else None))

    for task in tasks:
        for c in conds:
            for seed in range(A.seed_start, A.seed_start + n_seeds):
                env = MockEnv(task["code"], task["test"], task["entry"])
                agent = StreamAgent(model, tok, env, condition=c, latency_tokens=A.latency,
                                    max_new_tokens=A.max_new, edit_mode="line",
                                    temperature=A.temp, seed=seed,
                                    sys_override=SYS_LINE_DELIVERY,
                                    debounce=A.debounce, pause_align=A.pause_align,
                                    announce_lsp=A.announce_lsp, c_eager=A.c_eager,
                                    syntax_gate=A.syntax_gate, rich_signal=A.rich_signal)
                t0 = time.time()
                r = agent.run(build_prompt(task), "sol.py")
                dt = time.time() - t0
                m = r["metrics"]
                row = {"task": task["name"], "cond": c, "seed": seed, "resolved": bool(r["resolved"]),
                       "bailed": r.get("bailed"), "in_tokens": r["in_tokens"],
                       "out_tokens": r["out_tokens"], "sec": round(dt, 1),
                       "rework_ratio": m.get("rework_ratio"), "n_edits": m.get("n_edits"),
                       "n_tests": r["n_tests"], "turns": r["turns"],
                       "termination_reason": r.get("termination_reason"),
                       "stream_tail": r["stream"][-3000:], "events": r["events"]}
                agg[c].append(row)
                env.close()
                print(f"  [{task['name']:26}] {c} s{seed}: resolved={row['resolved']} "
                      f"in={row['in_tokens']} out={row['out_tokens']} tests={row['n_tests']} "
                      f"edits={row['n_edits']} rework={row['rework_ratio']} ({row['sec']}s)", flush=True)
        dump(A.out + ".partial")   # incremental: partials survive an interrupted long run

    def mean(xs): return round(sum(xs)/len(xs), 1) if xs else 0.0

    print("\n=== aggregate (resolve + efficiency) ===", flush=True)
    summary = {}
    for c in conds:
        rs = agg[c]; res = [r for r in rs if r["resolved"]]
        kinds = sorted({e["type"] for r in rs for e in r["events"] if e["type"].startswith("diag")})
        ndeliv = sum(1 for r in rs for e in r["events"]
                     if e["type"] in ("diag_debounced", "diag_async", "diag_eager", "diag_sync_queued"))
        summary[c] = {"resolve_rate": round(len(res)/len(rs), 3) if rs else 0, "n": len(rs),
                      "mean_in": mean([r["in_tokens"] for r in rs]),
                      "mean_out": mean([r["out_tokens"] for r in rs]),
                      "mean_tests": mean([r["n_tests"] for r in rs]),
                      "mean_sec": mean([r["sec"] for r in rs]),
                      "diag_event_types": kinds, "n_deliveries": ndeliv,
                      # efficiency among RESOLVED only (matched correctness):
                      "resolved_mean_out": mean([r["out_tokens"] for r in res]),
                      "resolved_mean_tests": mean([r["n_tests"] for r in res])}
        print(f"  {c}: resolve={summary[c]['resolve_rate']} ({len(res)}/{len(rs)})  "
              f"out={summary[c]['mean_out']} tests={summary[c]['mean_tests']} "
              f"deliveries={ndeliv} {kinds}  | resolved-only out={summary[c]['resolved_mean_out']} "
              f"tests={summary[c]['resolved_mean_tests']}", flush=True)

    dump(A.out, summary)
    print(f"-> {A.out}", flush=True)


if __name__ == "__main__":
    main()
