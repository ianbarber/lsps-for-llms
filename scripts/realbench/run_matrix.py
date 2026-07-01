#!/usr/bin/env python3
"""Real-repo efficiency matrix: run a frontier agent on an audited SWE-bench task under three tool
conditions and record the final patch + token cost. Scoring (does the patch resolve the task) is done
separately by scripts/realbench/score.py via the swebench Docker oracle.

Conditions (the variable is the RETRIEVAL/info tool, capability held fixed):
  R  read-only        : read_file, edit, done
  D  go-to-definition : + defn, find_references
  I  info             : D + check_types (pyrefly diagnostics)

The agent runs on a HOST checkout (RealRepoEnv): read/defn/find_references/check_types need no repo
dependencies, only source + pyrefly. It does NOT run the test suite in the loop (that needs the task
env); it is given the problem statement and the failing test's source as the spec, fixes, and calls
done(). The final `git diff` is the prediction, scored later.

  python scripts/realbench/run_matrix.py --tasks a,b,c --models M1,M2 --conds R,D,I --seeds 2 \
      --budget-usd 10 --out runs/realbench/matrix
"""
import os
import re
import sys
import json
import time
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from scaffold.real_env import RealRepoEnv
from scripts.realbench import swe_loader as L
from scripts import api_agent as A   # reuse OpenAI client, pricing, budget machinery

READ_TRUNC = 16000


def _numbered(src):
    return "\n".join(f"{i+1:>3}| {ln}" for i, ln in enumerate(src.splitlines()))


def system_prompt(cond):
    lines = [
        "You are a coding agent fixing a bug in a real Python repository. Make the smallest change that "
        "fixes the issue described below. Work by calling tools; do not paste code in prose.",
        "",
        "Tools:",
        "- read_file(path): read a repo file (numbered, truncated). Reading a whole large file is "
        "expensive.",
    ]
    if cond in ("D", "I"):
        lines += [
            "- defn(symbol): return just the definition/signature of a symbol (bare name, Class.method, "
            "or module.Class.method). CHEAPER than reading the whole file it lives in; prefer it when "
            "you only need a symbol's shape.",
            "- find_references(symbol): list files that use a symbol.",
        ]
    lines += [
        "- edit(path, search, replace): replace the FIRST exact occurrence of `search` in `path` with "
        "`replace`. `search` must match the file exactly (indentation included) and be unique enough to "
        "locate the spot.",
    ]
    if cond == "I":
        lines += [
            "- check_types(path): run the static type checker on a file and return its diagnostics.",
        ]
    lines += [
        "- done(): call this when the fix is complete.",
        "",
        "You cannot run the test suite. Reason from the issue, the shown failing test, and the code; "
        "make the fix; then call done().",
    ]
    return "\n".join(lines)


def user_prompt(task, env, editable):
    parts = [f"Repository: {task['repo']} at commit {task['base_commit'][:10]}.", "",
             "Issue:", task["problem_statement"].strip()[:6000], ""]
    # show the failing test (from the test_patch) so the agent knows the target behaviour
    tp = task.get("test_patch", "")
    added = L.added_lines_by_file(tp)
    if added:
        parts.append("The failing test(s) that must pass after your fix (do NOT edit tests):")
        for f, lines in list(added.items())[:2]:
            body = "\n".join(lines)[:2500]
            parts.append(f"`{f}`:\n```python\n{body}\n```")
        parts.append("")
    # localization hint: the enclosing function/class of each gold hunk, so the task isolates the
    # RETRIEVAL of the cross-file symbol rather than the separate problem of finding where to edit.
    hints = target_hints(task["patch"])
    if hints:
        parts.append("The change belongs here (isolating retrieval from localization):")
        for f, ctxs in hints.items():
            parts.append(f"  {f}: {', '.join(ctxs[:3])}")
        parts.append("")
    parts.append(f"You may edit only: {', '.join(editable)}. Here is the primary file, numbered:")
    a = editable[0]
    parts.append(f"`{a}`:\n{_numbered(env.read_file(a))[:READ_TRUNC]}")
    parts.append("\nRead any other file with read_file(path); the workspace is the full repo. "
                 "Make edits with edit(path, search, replace), then call done().")
    return "\n".join(parts)


def target_hints(patch):
    """{file: [enclosing def/class contexts]} from the gold patch hunk headers, for localization."""
    out, cur = {}, None
    for ln in patch.splitlines():
        m = re.match(r"^\+\+\+ b/(.+)$", ln)
        if m:
            cur = m.group(1); out.setdefault(cur, []); continue
        h = re.match(r"^@@ .*@@\s*(.+)$", ln)
        if cur and h and h.group(1).strip():
            ctx = h.group(1).strip()
            if ctx not in out[cur]:
                out[cur].append(ctx)
    return {f: c for f, c in out.items() if c}


def build_tools(cond):
    def fn(name, desc, props, req):
        return {"type": "function", "function": {"name": name, "description": desc,
                "parameters": {"type": "object", "properties": props, "required": req}}}
    tools = [fn("read_file", "Read a repo file (numbered, truncated).",
               {"path": {"type": "string"}}, ["path"])]
    if cond in ("D", "I"):
        tools += [fn("defn", "Definition/signature of a symbol (cheaper than reading its whole file).",
                     {"symbol": {"type": "string"}}, ["symbol"]),
                  fn("find_references", "Files that use a symbol.", {"symbol": {"type": "string"}}, ["symbol"])]
    tools += [fn("edit", "Replace the first exact occurrence of `search` with `replace` in `path`.",
                 {"path": {"type": "string"}, "search": {"type": "string"}, "replace": {"type": "string"}},
                 ["path", "search", "replace"])]
    if cond == "I":
        tools += [fn("check_types", "Static type-checker diagnostics for a file.",
                     {"path": {"type": "string"}}, ["path"])]
    tools += [fn("done", "Finish (fix complete).", {}, [])]
    return tools


class Roll:
    def __init__(self, env, editable, max_reads=8):
        self.env = env; self.editable = set(editable); self.max_reads = max_reads
        self.counts = {"read": 0, "defn": 0, "findrefs": 0, "edit": 0, "check": 0}
        self.trace = []; self.done = False

    def execute(self, name, a):
        if name == "read_file":
            p = a.get("path", "")
            if self.counts["read"] >= self.max_reads:
                self.trace.append({"t": "read", "path": p, "capped": True})
                return (f"[read cap {self.max_reads} reached] Stop reading. Use defn(symbol) for a "
                        f"specific symbol, or make your edit and call done().")
            self.counts["read"] += 1; self.trace.append({"t": "read", "path": p})
            try:
                return f"`{p}` (numbered, truncated):\n{_numbered(self.env.read_file(p))[:READ_TRUNC]}"
            except Exception as e:
                return f"(cannot read {p}: {e})"
        if name == "defn":
            s = a.get("symbol", ""); self.counts["defn"] += 1
            span, dp = self.env.goto_definition(s)
            self.trace.append({"t": "defn", "sym": s, "found": bool(span)})
            return f"definition of `{s}` (in {dp}):\n{span[:READ_TRUNC]}" if span else "(no definition found)"
        if name == "find_references":
            s = a.get("symbol", ""); self.counts["findrefs"] += 1
            refs = self.env.find_references(s)
            self.trace.append({"t": "findrefs", "sym": s, "n": len(refs)})
            return ("references to `%s`: %s" % (s, ", ".join(refs[:40]))) if refs else "(none found)"
        if name == "edit":
            p = a.get("path", "")
            if p not in self.editable:
                return f"edit refused: you may only edit {sorted(self.editable)}."
            ok, info = self.env.apply_edit(p, a.get("search", ""), a.get("replace", ""))
            self.counts["edit"] += 1; self.trace.append({"t": "edit", "path": p, "ok": bool(ok)})
            return "edit applied." if ok else f"edit failed: {info}"
        if name == "check_types":
            p = a.get("path", ""); self.counts["check"] += 1
            diag = self.env.pyrefly_diagnostics(p)
            self.trace.append({"t": "check", "n": len((diag or '').splitlines())})
            return diag if (diag or "").strip() else "(no type errors)"
        if name == "done":
            self.done = True; self.trace.append({"t": "done"}); return "(done)"
        return f"(unknown tool {name})"


def run_rollout(client, model, task, cond, seed, args, price, spent):
    repo_dir = L.ensure_clone(task["repo"]); L.checkout(repo_dir, task["base_commit"])
    editable = [f for f in L.patched_files(task["patch"]) if f.endswith(".py")][:3]
    pkg = editable[0].split("/")[0] if "/" in editable[0] else ""
    env = RealRepoEnv(repo_dir, editable=editable, test_spec="true", base_commit=task["base_commit"],
                      file_glob=f"{pkg}/**/*.py" if pkg else None, write_pyrefly_config=True)
    try:
        ro = Roll(env, editable, max_reads=args.max_reads)
        messages = [{"role": "system", "content": system_prompt(cond)},
                    {"role": "user", "content": user_prompt(task, env, editable)}]
        tools = build_tools(cond)
        pp, cp = price; in_tok = out_tok = turns = 0; est = 0.0; stop = "max_turns"
        for _ in range(args.max_turns):
            if spent + est >= args.budget_usd:
                stop = "budget"; break
            resp = None
            for att in range(4):
                try:
                    resp = client.chat.completions.create(model=model, messages=messages, tools=tools,
                            tool_choice="auto", temperature=args.temperature, seed=seed); break
                except Exception as e:
                    msg = str(e).lower()
                    if att < 3 and any(s in msg for s in ("429", "rate", "timeout", "502", "503", "overload")):
                        time.sleep(2 * (att + 1)); continue
                    stop = f"api_error:{type(e).__name__}"; break
            if resp is None:
                break
            turns += 1
            u = getattr(resp, "usage", None)
            if u:
                in_tok += getattr(u, "prompt_tokens", 0) or 0; out_tok += getattr(u, "completion_tokens", 0) or 0
                est = in_tok * pp + out_tok * cp
            msg = resp.choices[0].message if resp.choices else None
            tcs = getattr(msg, "tool_calls", None) or [] if msg else []
            am = {"role": "assistant", "content": (msg.content if msg else "") or ""}
            if tcs:
                am["tool_calls"] = [{"id": tc.id, "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}} for tc in tcs]
            messages.append(am)
            if not tcs:
                messages.append({"role": "user", "content": "Use the tools (edit, then done)."}); continue
            for tc in tcs:
                try:
                    aa = json.loads(tc.function.arguments or "{}")
                except Exception:
                    aa = {}
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": str(ro.execute(tc.function.name, aa))})
            if ro.done:
                stop = "done"; break
        patch = env.current_patch()
        return {"instance_id": task["instance_id"], "repo": task["repo"], "cond": cond,
                "model": model, "seed": seed, "stop": stop, "turns": turns,
                "prompt_tokens": in_tok, "completion_tokens": out_tok, "est_cost_usd": round(est, 6),
                "n_read": ro.counts["read"], "n_defn": ro.counts["defn"], "n_findrefs": ro.counts["findrefs"],
                "n_edit": ro.counts["edit"], "n_check": ro.counts["check"],
                "empty_patch": not patch.strip(), "model_patch": patch, "trace": ro.trace}, est
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="comma-separated instance_ids")
    ap.add_argument("--models", default="anthropic/claude-sonnet-4.5,deepseek/deepseek-chat-v3.1")
    ap.add_argument("--conds", default="R,D,I")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--budget-usd", type=float, default=10.0)
    ap.add_argument("--max-turns", type=int, default=14)
    ap.add_argument("--max-reads", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", default="runs/realbench/matrix")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "0"); os.environ.setdefault("HF_DATASETS_OFFLINE", "0")
    want = set(args.tasks.split(","))
    tasks = {t["instance_id"]: t for t in L.load_tasks() if t["instance_id"] in want}
    missing = want - set(tasks)
    if missing:
        print(f"[warn] instance_ids not found: {missing}", file=sys.stderr)
    models = args.models.split(","); conds = args.conds.split(",")
    key = A.load_or_key()
    from openai import OpenAI
    client = OpenAI(base_url=A.OR_BASE, api_key=key,
                    default_headers={"HTTP-Referer": "https://localhost/streams", "X-Title": "streams-realbench"})
    price_by_model = {m: A.resolve_pricing(m)[:2] for m in models}
    outdir = os.path.join(ROOT, args.out); os.makedirs(outdir, exist_ok=True)

    rows = []; spent = 0.0
    for iid in [i for i in args.tasks.split(",") if i in tasks]:
        for model in models:
            for cond in conds:
                for seed in range(args.seed_start, args.seed_start + args.seeds):
                    if spent >= args.budget_usd:
                        print("[BUDGET HIT]", flush=True); break
                    t0 = time.time()
                    row, est = run_rollout(client, model, tasks[iid], cond, seed, args, price_by_model[model], spent)
                    spent += est; row["sec"] = round(time.time() - t0, 1); rows.append(row)
                    print(f"  {iid:30} {model.split('/')[-1]:20} {cond} s{seed}: stop={row['stop']} "
                          f"edits={row['n_edit']} defn={row['n_defn']} read={row['n_read']} "
                          f"in={row['prompt_tokens']} empty={row['empty_patch']} ${spent:.3f}", flush=True)
                    json.dump(rows, open(os.path.join(outdir, "rollouts.json"), "w"), indent=2)
                    # also emit per-(model,cond) predictions for the scorer
                    for m in models:
                        for c in conds:
                            preds = [{"instance_id": r["instance_id"], "model_name_or_path": f"{m.split('/')[-1]}_{c}",
                                      "model_patch": r["model_patch"]}
                                     for r in rows if r["model"] == m and r["cond"] == c]
                            if preds:
                                tag = f"{m.split('/')[-1]}_{c}".replace(".", "")
                                json.dump(preds, open(os.path.join(outdir, f"pred_{tag}.json"), "w"), indent=2)
    print(f"\n=== {len(rows)} rollouts, ${spent:.3f}. predictions in {args.out}/pred_*.json -> score with score.py ===")


if __name__ == "__main__":
    main()
