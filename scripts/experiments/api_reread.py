#!/usr/bin/env python3
"""Strong-model (OpenRouter) validation of the reread-after-span probe.

Runs arms (a) auto_neutral and (b) auto_sufficient of the reread experiment
(scripts/experiments/run_navigation_reread.py) against a frontier model through the
repo's existing API-agent path: scripts/api_agent.py supplies the key loading,
pricing/budget guard, and the tool-executing Rollout; navigation_tasks supplies the
tasks/envs (RealRepoEnv is MultiFileEnv-shaped, which is what Rollout drives). The
frozen navigation protocol files are imported read-only and unmodified.

Tools offered: read_file / edit_lines / run_tests / done (no defn tools -- the span
is delivered automatically in the prompt, as in the local automatic arms).
Key outcome per row: whether the target file is read after the span was supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.api_agent import (  # noqa: E402
    OR_BASE,
    Rollout,
    build_tools,
    load_or_key,
    resolve_pricing,
)
from scripts.experiments.navigation_tasks import (  # noqa: E402
    PROTOCOL_VERSION,
    build_prompt,
    build_tasks,
    make_env,
    run_heldout,
)
from scripts.experiments.run_navigation import _method_from_lsp  # noqa: E402


TOOLS_TEXT = (
    "You are fixing one one-line Python bug in an implementation module under pkg/units/. "
    "Work iteratively using the tools provided (call tools; do not paste code in prose):\n"
    "- read_file(path): read a workspace file (numbered view, truncated).\n"
    "- edit_lines(path, start, end, new_text): replace the inclusive 1-based line range "
    "START..END of `path` with new_text.\n"
    "- run_tests(): run the behavioral test; returns 'ALL TESTS PASS' or the failure.\n"
    "- done(): finish after the test passes.\n\n"
)
AUTO_TEXT = (
    "A language-server result from the visible call site is supplied in the user message. It is the "
    "current source definition, not a proposed fix. Use it as compact context; infer and apply a fix "
    "only if the task evidence supports one.\n"
)
SUFFICIENT_TEXT = (
    "A language-server result from the visible call site is supplied in the user message. It is the "
    "current source definition, not a proposed fix. The supplied span is the COMPLETE current source of "
    "the method that binds at the call site, with its file path and real line numbers. It is sufficient "
    "context for the edit: do NOT call read_file on the defining file unless the span is insufficient. "
    "Edit directly at the span's stated path and line numbers, then run the tests.\n"
)
ARM_SYS = {
    "auto_neutral": TOOLS_TEXT + AUTO_TEXT,
    "auto_sufficient": TOOLS_TEXT + SUFFICIENT_TEXT,
}


def run_rollout(client, model, task, arm, args, price, spent_so_far):
    env = make_env(task, "typed")
    try:
        supplied, supplied_path, lsp_latency = _method_from_lsp(task, "typed", env)
        if supplied_path != task["target_path"]:
            raise RuntimeError(
                f"typed automatic result did not resolve the gold override: {supplied_path}"
            )
        if env.lsp_errors:
            raise RuntimeError(f"automatic semantic query failed: {env.lsp_errors}")
        if task["gold"]["new_text"] in supplied:
            raise RuntimeError("semantic context contains the gold replacement")
        prompt = build_prompt(task, "typed") + (
            "\n\nThe following current source span was supplied from a language-server "
            "definition result at the visible call site. It is source context, not a "
            "proposed correction.\n<semantic_result kind=\"current_source\">\n"
            + supplied + "\n</semantic_result>"
        )
        ro = Rollout(env, "pkg/app.py", args.max_reads)
        tools = build_tools(no_defn=True, with_check=False)
        messages = [
            {"role": "system", "content": ARM_SYS[arm]},
            {"role": "user", "content": prompt},
        ]
        pp, cp = price
        est_cost = 0.0
        in_tok = out_tok = turns = 0
        stop_reason = "max_turns"

        for _turn in range(args.max_turns):
            if spent_so_far + est_cost >= args.budget_usd:
                stop_reason = "budget"
                break
            resp = None
            last_err = None
            for attempt in range(4):
                try:
                    resp = client.chat.completions.create(
                        model=model, messages=messages, tools=tools,
                        tool_choice="auto", temperature=args.temperature, seed=args.seed,
                    )
                    break
                except Exception as e:  # noqa: BLE001 - transient-API resilience
                    last_err = e
                    msg = str(e).lower()
                    transient = any(s in msg for s in ("429", "rate", "timeout", "timed out",
                                                       "502", "503", "overload", "temporarily"))
                    if attempt < 3 and transient:
                        time.sleep(2 * (attempt + 1))
                        continue
                    break
            if resp is None:
                stop_reason = f"api_error: {type(last_err).__name__}: {str(last_err)[:200]}"
                ro.trace.append({"t": "api_error", "err": str(last_err)[:200]})
                break
            turns += 1
            u = getattr(resp, "usage", None)
            if u is not None:
                in_tok += getattr(u, "prompt_tokens", 0) or 0
                out_tok += getattr(u, "completion_tokens", 0) or 0
                est_cost = in_tok * pp + out_tok * cp
            if not getattr(resp, "choices", None):
                stop_reason = "no_choices"
                break
            msg = resp.choices[0].message
            tcs = getattr(msg, "tool_calls", None) or []
            amsg = {"role": "assistant", "content": msg.content or ""}
            if tcs:
                amsg["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name,
                                  "arguments": tc.function.arguments or "{}"}}
                    for tc in tcs]
            messages.append(amsg)
            if not tcs:
                messages.append({"role": "user",
                                 "content": "Use the tools to make progress: edit_lines, "
                                            "run_tests, then done. Do not answer in prose."})
                continue
            for tc in tcs:
                try:
                    a = json.loads(tc.function.arguments or "{}")
                except Exception:  # noqa: BLE001
                    a = {}
                result = ro.execute(tc.function.name, a)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": str(result)})
            if ro.done:
                stop_reason = "done"
                break
            if ro.last_test and ro.last_test.get("resolved"):
                stop_reason = "tests_pass"
                break

        visible = env.run_tests()
        held_out_pass = run_heldout(task, "typed")
        target_reads = [t for t in ro.trace
                        if t.get("t") == "read" and t.get("path") == task["target_path"]
                        and not t.get("capped")]
        row = {
            "task": task["name"], "family": task["seed"], "split": args.split,
            "variant": "typed", "arm": arm, "seed": args.seed, "model": model,
            "resolved": held_out_pass, "visible_pass": bool(visible.get("resolved")),
            "held_out_pass": held_out_pass, "stop_reason": stop_reason,
            "turns": turns, "prompt_tokens": in_tok, "completion_tokens": out_tok,
            "n_read": ro.counts["read"], "n_edit": ro.counts["edit"],
            "n_test": ro.counts["test"],
            "n_reads_after_span": len(target_reads),
            "read_after_span": bool(target_reads),
            "auto_span_lsp_latency_ms": round(lsp_latency * 1000, 1),
            "semantic_supplied_path": supplied_path,
            "semantic_payload_sha256": hashlib.sha256(supplied.encode()).hexdigest(),
            "est_cost_usd": round(est_cost, 6), "trace": ro.trace,
        }
        return row, est_cost
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--model", default="anthropic/claude-sonnet-4.5")
    # the reserved confirmation split is not offered: apparatus/pilot only (C15).
    parser.add_argument("--split", choices=("pilot", "apparatus"), default="apparatus")
    parser.add_argument("--arms", default="auto_neutral,auto_sufficient")
    parser.add_argument("--names", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-reads", type=int, default=12)
    parser.add_argument("--budget-usd", type=float, default=5.0)
    parser.add_argument("--tmp-root", default=None)
    args = parser.parse_args()
    out_path = Path(args.out)
    if out_path.exists():
        print(f"refusing to overwrite existing result: {out_path}", file=sys.stderr)
        return 73
    arms = args.arms.split(",")
    unknown = [a for a in arms if a not in ARM_SYS]
    if unknown:
        raise ValueError(f"unknown arm(s): {unknown}")

    root = args.tmp_root or str(Path(tempfile.gettempdir()) / "streams_navigation_v2_api_reread")
    tasks = build_tasks(Path(root) / args.split, args.split)
    if args.names:
        wanted = set(args.names.split(","))
        tasks = [task for task in tasks if task["name"] in wanted]
    if not tasks:
        raise ValueError("no navigation tasks selected")

    key = load_or_key()
    from openai import OpenAI
    client = OpenAI(base_url=OR_BASE, api_key=key,
                    default_headers={"HTTP-Referer": "https://localhost/streams",
                                     "X-Title": "streams-api-reread"})
    pp, cp, matched = resolve_pricing(args.model)

    rows = []
    spent = 0.0
    budget_hit = False

    def checkpoint():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "protocol": PROTOCOL_VERSION, "experiment": "reread-after-span-api",
            "model": args.model, "matched_pricing_id": matched, "config": vars(args),
            "arms": arms, "total_cost_usd": round(spent, 6), "budget_hit": budget_hit,
            "rows": rows,
        }, indent=2) + "\n", encoding="utf-8")

    for task in tasks:
        if budget_hit:
            break
        for arm in arms:
            if spent >= args.budget_usd:
                budget_hit = True
                break
            t0 = time.time()
            row, cost = run_rollout(client, args.model, task, arm, args, (pp, cp), spent)
            spent += cost
            row["sec"] = round(time.time() - t0, 1)
            rows.append(row)
            print(f"  [{task['name']:22}] {arm:16s}: pass={row['resolved']} "
                  f"stop={row['stop_reason']} reads_after_span={row['n_reads_after_span']} "
                  f"tests={row['n_test']} edits={row['n_edit']} in={row['prompt_tokens']} "
                  f"cost=${row['est_cost_usd']:.4f} (cum ${spent:.4f})", flush=True)
            checkpoint()
            if row["stop_reason"] == "budget":
                budget_hit = True
                break
    checkpoint()

    print("\n=== per-arm summary ===", flush=True)
    for arm in arms:
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        n = len(sub)
        print(f"  {arm:16s} pass={sum(1 for r in sub if r['resolved'])}/{n}  "
              f"read_after_span={sum(1 for r in sub if r['read_after_span'])}/{n}  "
              f"mean_in_toks={round(sum(r['prompt_tokens'] for r in sub) / n, 1)}", flush=True)
    print(f"total spend: ${spent:.4f} ({'BUDGET HIT' if budget_hit else 'within budget'})",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
