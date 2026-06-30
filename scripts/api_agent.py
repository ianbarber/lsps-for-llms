#!/usr/bin/env python3
"""Turn-based tool-calling agent harness driving a FRONTIER model via OpenRouter
against the multi-file coding tasks (scaffold.mock_env.MultiFileEnv).

This validates the research recipe in the real deployment modality (a tool-calling
agent) instead of the local stream-decoding scaffold. Network + CPU only; no GPU.

The model is given OpenAI-style function tools and a standard loop:
  send system+user+tools -> model returns tool_calls -> execute each against the
  MultiFileEnv -> append tool results -> repeat until <done>, tests pass, or max_turns.

Tools (gated):
  read_file(path)                       always
  defn(symbol)                          OMITTED under --no-defn
  find_references(symbol)               OMITTED under --no-defn
  edit_lines(path, start, end, new_text) always
  run_tests()                           always
  check_types()                         ONLY under --with-check (Gap D info condition)
  done()                                always

The --no-defn condition is the tool-value ablation (read-only): the cheap <defn>
action is genuinely absent and is never mentioned in the prompt. --with-check adds
the type-checker tool (the Gap D info condition, default OFF).

Cost control is non-negotiable: pricing is fetched once from the OpenRouter /models
endpoint and a running est_cost_usd is maintained; before every API call we stop
cleanly if cumulative spend >= --budget-usd.

  python scripts/api_agent.py OUT --model MODEL --suite {effic_real2,gapd} \
      [--names a,b] [--seeds K] [--no-defn] [--with-check] [--budget-usd N] \
      [--max-turns N] [--max-reads N] [--temperature T]
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scaffold.mock_env import MultiFileEnv

OR_BASE = "https://openrouter.ai/api/v1"
MODELS_CACHE = os.path.join(os.environ.get("TMPDIR", "/tmp"), "or_models.json")


def load_or_key():
    """OpenRouter key from OPENROUTER_API_KEY, else a .orkey file (cwd or repo root)."""
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k.strip()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in (".orkey", os.path.join(here, ".orkey")):
        if os.path.exists(p):
            return open(p).read().strip()
    raise SystemExit("No OpenRouter key: set OPENROUTER_API_KEY or create a .orkey file in the repo root.")
READ_TRUNC = 16000   # like synth_mf: <read> truncates a file to 16000 chars


# ---------------------------------------------------------------------------
# task suites
# ---------------------------------------------------------------------------
def load_suite(suite):
    if suite == "effic_real2":
        from scripts.synth_tasks_effic_real2 import TASKS_EFFIC_REAL2
        return TASKS_EFFIC_REAL2
    if suite == "runtime":
        from scripts.synth_tasks_runtime import TASKS_RUNTIME
        return TASKS_RUNTIME
    if suite in ("gapd", "gapd2"):
        # Loaded lazily so this harness imports even if the suite file does not exist yet.
        mod = "synth_tasks_gapd" if suite == "gapd" else "synth_tasks_gapd2"
        var = "TASKS_GAPD" if suite == "gapd" else "TASKS_GAPD2"
        try:
            m = __import__(f"scripts.{mod}", fromlist=[var])
            return getattr(m, var)
        except Exception as e:
            print(f"[fatal] --suite {suite} requested but scripts/{mod}.py "
                  f"is unavailable ({e})", file=sys.stderr)
            sys.exit(2)
    raise ValueError(suite)


# ---------------------------------------------------------------------------
# pricing
# ---------------------------------------------------------------------------
def _models_json():
    """Fetch the OpenRouter model catalogue once; fall back to the cached copy."""
    import requests
    try:
        r = requests.get(OR_BASE + "/models", timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[pricing] live fetch failed ({e}); using cached {MODELS_CACHE}", flush=True)
        with open(MODELS_CACHE) as f:
            return json.load(f)


def resolve_pricing(model):
    """Return (prompt_$per_tok, completion_$per_tok, matched_id).

    Exact id match first, then a slug/suffix fuzzy match, else a conservative
    over-estimating fallback so the budget guard never *under*-counts spend."""
    data = _models_json()
    data = data.get("data", data)
    by_id = {m["id"]: m for m in data}
    m = by_id.get(model)
    matched = model
    if m is None:
        # fuzzy: same provider prefix, longest shared id
        cands = [mm for mm in data if mm["id"].split("/")[0] == model.split("/")[0]]
        tail = model.split("/")[-1]
        best = None
        for mm in cands:
            if tail in mm["id"] or mm["id"].split("/")[-1] in model:
                if best is None or len(os.path.commonprefix([mm["id"], model])) > \
                        len(os.path.commonprefix([best["id"], model])):
                    best = mm
        m = best
        matched = m["id"] if m else None
    if m is None:
        # nothing close: conservative over-estimate ($1/$3 per Mtok)
        print(f"[pricing] WARN no catalogue entry for {model}; using conservative "
              f"fallback $1.00/$3.00 per Mtok (over-estimate)", flush=True)
        return 1e-6, 3e-6, None
    p = m.get("pricing", {})
    pp = float(p.get("prompt", 0) or 0)
    cp = float(p.get("completion", 0) or 0)
    if matched != model:
        print(f"[pricing] '{model}' not in catalogue; priced via nearest '{matched}' "
              f"(${pp*1e6:.4f}/${cp*1e6:.4f} per Mtok)", flush=True)
    else:
        print(f"[pricing] {model}: ${pp*1e6:.4f}/${cp*1e6:.4f} per Mtok "
              f"(prompt/completion)", flush=True)
    return pp, cp, matched


# ---------------------------------------------------------------------------
# prompt construction (mirrors scripts/synth_mf.py::build_prompt single-target)
# ---------------------------------------------------------------------------
def _numbered(src):
    return "\n".join(f"{i+1:>3}| {ln}" for i, ln in enumerate(src.splitlines()))


def system_prompt(no_defn, with_check, no_test=False):
    """Mirror SYS_LINE's framing for the tool-calling modality. When --no-defn,
    defn / find_references are not mentioned anywhere. When --no-test, run_tests is
    absent: the agent cannot execute the code and must reason about correctness from
    the source and the shown spec (the R0 "no-execution" arm of the runtime test)."""
    lines = [
        "You are a coding agent fixing a bug in a Python repository. The bug is in "
        "the file `target.py`. Fix it so the test passes. Work iteratively using the "
        "tools provided (call tools; do not paste code in prose).",
        "",
        "Tools:",
        "- edit_lines(path, start, end, new_text): replace the inclusive 1-based line "
        "range START..END of `path` with new_text. `target.py` is shown with line "
        "numbers in the next message; after each edit you get a fresh numbered view "
        "(line numbers shift — always use the latest view).",
        "- read_file(path): read another workspace file for context (returns a "
        "numbered view, truncated).",
    ]
    if not no_defn:
        lines += [
            "- defn(symbol): look up just a symbol's definition/signature — this is "
            "CHEAPER than reading a whole file, so prefer it when you only need a "
            "symbol's signature or shape.",
            "- find_references(symbol): list the files where a symbol is used.",
        ]
    if not no_test:
        lines += [
            "- run_tests(): run the test suite; returns \"ALL TESTS PASS\" or the failure.",
        ]
    if with_check:
        lines += [
            "- check_types(): run the static type checker and return its diagnostics "
            "(file, line, code, message) — use it to catch type mismatches before "
            "running the tests.",
        ]
    if no_test:
        lines += [
            "- done(): call this when you believe the fix is correct.",
            "",
            "You CANNOT run the tests in this task: there is no run_tests tool. Reason "
            "carefully about what the code does on the inputs in the shown spec, make your "
            "edit, and call done() when you are confident the fix is correct.",
        ]
    else:
        lines += [
            "- done(): call this once the tests pass to finish.",
            "",
            "Loop: edit_lines -> run_tests; if it still fails, read the failure, fix, and "
            "run_tests again, until tests pass; then call done. Reason briefly between "
            "tool calls.",
        ]
    return "\n".join(lines)


def user_prompt(task, no_hint=False, no_test=False):
    """Mirror build_prompt: show ONLY the target (numbered) + names of the other
    workspace files + the test (the spec). The test is shown as static text in every
    arm, including --no-test, so the R0 agent knows the spec but cannot execute it."""
    target = task["target"]
    body = f"`{target}`:\n{_numbered(task['files'][target])}"
    head = f"Fix the bug(s) in `{target}` so the test below passes.\n\n{body}\n\n"
    others = [f for f in sorted(task["files"]) if f != target]
    if others:
        whereabouts = (f"The workspace also contains these files: {', '.join(others)} "
                       f"— you have NOT seen their contents; read any with "
                       f"read_file(path).\n\n")
    else:
        whereabouts = ""
    tail = (f"Make line-range edits to `{target}` with edit_lines, then call done()."
            if no_test else
            f"Make line-range edits to `{target}` with edit_lines, then run run_tests.")
    note = ""
    if task.get("held_out") and not no_hint:
        note = ("\n\nImportant: the shown test is a partial spec — it does NOT exercise every input "
                "(missing keys, empty or None values, other branches). Make your fix correct for ALL "
                "inputs, not only the shown cases, before calling done().")
    return (head + whereabouts +
            "The test that must pass (do NOT edit it; it is the spec):\n"
            f"```python\n{task['test']}\n```\n" + tail + note)


# ---------------------------------------------------------------------------
# tool schemas
# ---------------------------------------------------------------------------
def build_tools(no_defn, with_check, no_test=False):
    def fn(name, desc, props, required):
        return {"type": "function", "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}}
    tools = [
        fn("read_file", "Read a workspace file (numbered view, truncated). Cheaper to "
           "target your edits with the latest view.",
           {"path": {"type": "string", "description": "workspace-relative path"}}, ["path"]),
    ]
    if not no_defn:
        tools += [
            fn("defn", "Look up just a symbol's definition/signature (cheaper than "
               "reading the whole file it lives in).",
               {"symbol": {"type": "string", "description": "symbol name, e.g. reduceby"}},
               ["symbol"]),
            fn("find_references", "List the workspace files that use a symbol.",
               {"symbol": {"type": "string"}}, ["symbol"]),
        ]
    tools += [
        fn("edit_lines", "Replace the inclusive 1-based line range START..END of `path` "
           "with new_text (raw code, no line-number prefixes, no ``` fences).",
           {"path": {"type": "string"},
            "start": {"type": "integer", "description": "1-based first line, inclusive"},
            "end": {"type": "integer", "description": "1-based last line, inclusive"},
            "new_text": {"type": "string", "description": "replacement code"}},
           ["path", "start", "end", "new_text"]),
    ]
    if not no_test:
        tools += [
            fn("run_tests", "Run the test suite. Returns 'ALL TESTS PASS' or the failure.",
               {}, []),
        ]
    if with_check:
        tools += [
            fn("check_types", "Run the static type checker; returns diagnostics.", {}, []),
        ]
    tools += [
        fn("done", "Finish the rollout (call once tests pass).", {}, []),
    ]
    return tools


# ---------------------------------------------------------------------------
# tool execution
# ---------------------------------------------------------------------------
class Rollout:
    def __init__(self, env, target, max_reads):
        self.env = env
        self.target = target
        self.max_reads = max_reads
        self.counts = {"read": 0, "defn": 0, "findrefs": 0, "test": 0, "edit": 0,
                       "check": 0, "auto_test": 0}
        self.trace = []
        self.last_test = None   # last model-invoked run_tests dict
        self.done = False

    def _numbered_view(self, path):
        try:
            src = self.env.read_file(path)
        except Exception as e:
            return None, f"(cannot read {path}: {e})"
        return src, _numbered(src)[:READ_TRUNC]

    def execute(self, name, args):
        """Run one tool call, return (result_text). Also updates counters/trace."""
        if name == "read_file":
            path = args.get("path") or self.target
            if self.counts["read"] >= self.max_reads:
                self.trace.append({"t": "read", "path": path, "capped": True})
                return (f"[read cap reached: {self.max_reads} reads used] Stop reading "
                        f"and make your edit / run_tests instead.")
            self.counts["read"] += 1
            _, view = self._numbered_view(path)
            self.trace.append({"t": "read", "path": path})
            return f"`{path}` (numbered, truncated to {READ_TRUNC} chars):\n{view}"

        if name == "defn":
            sym = args.get("symbol", "")
            self.counts["defn"] += 1
            span, dpath = self.env.goto_definition(sym)
            self.trace.append({"t": "defn", "sym": sym, "found": bool(span)})
            if span:
                return f"definition of `{sym}` (in {dpath}):\n{span[:READ_TRUNC]}"
            return "(no definition found)"

        if name == "find_references":
            sym = args.get("symbol", "")
            self.counts["findrefs"] += 1
            refs = self.env.find_references(sym)
            self.trace.append({"t": "findrefs", "sym": sym, "n": len(refs)})
            return ("references to `%s`: %s" % (sym, ", ".join(refs))) if refs \
                else f"(no references found for `{sym}`)"

        if name == "edit_lines":
            path = args.get("path") or self.target
            try:
                start = int(args["start"]); end = int(args["end"])
            except (KeyError, ValueError, TypeError):
                self.trace.append({"t": "edit", "path": path, "ok": False, "err": "bad args"})
                return "edit failed: start/end must be integers."
            new_text = args.get("new_text", "")
            ok, info = self.env.apply_line_edit(path, start, end, new_text)
            self.counts["edit"] += 1
            self.trace.append({"t": "edit", "path": path, "lines": f"{start}-{end}", "ok": bool(ok)})
            if not ok:
                return f"edit failed: {info}"
            _, view = self._numbered_view(path)
            return f"edit applied to `{path}`. Fresh numbered view:\n{view}"

        if name == "run_tests":
            self.counts["test"] += 1
            tr = self.env.run_tests()
            self.last_test = tr
            self.trace.append({"t": "test", "resolved": bool(tr.get("resolved"))})
            if tr.get("resolved"):
                return "ALL TESTS PASS."
            return f"FAIL: {tr.get('failure', 'test failed')}"

        if name == "check_types":
            self.counts["check"] += 1
            diag = self.env.pyrefly_diagnostics()
            self.trace.append({"t": "check", "n_lines": len(diag.splitlines()) if diag else 0})
            return diag if diag.strip() else "(no type errors)"

        if name == "done":
            self.done = True
            self.trace.append({"t": "done"})
            return "(rollout ended)"

        self.trace.append({"t": "unknown", "name": name})
        return f"(unknown tool: {name})"


# ---------------------------------------------------------------------------
# one rollout
# ---------------------------------------------------------------------------
def run_rollout(client, model, task, seed, args, price, spent_so_far):
    """Drive the tool-calling loop for one (task, seed). Returns (row, est_cost)."""
    target = task["target"]
    env = MultiFileEnv(task["files"], target, task["test"], skip_pyrefly=not args.with_check,
                       held_out_src=task.get("held_out"))
    try:
        ro = Rollout(env, target, args.max_reads)
        tools = build_tools(args.no_defn, args.with_check, args.no_test)
        messages = [
            {"role": "system", "content": system_prompt(args.no_defn, args.with_check, args.no_test)},
            {"role": "user", "content": user_prompt(task, args.no_hint, args.no_test)},
        ]
        pp, cp = price
        est_cost = 0.0
        in_tok = out_tok = turns = 0
        stop_reason = "max_turns"

        for turn in range(args.max_turns):
            # BUDGET GUARD — before every API call.
            if spent_so_far + est_cost >= args.budget_usd:
                stop_reason = "budget"
                break
            resp = None
            last_err = None
            for attempt in range(4):   # transient-error resilience (rate limits under concurrency)
                try:
                    resp = client.chat.completions.create(
                        model=model, messages=messages, tools=tools,
                        tool_choice="auto", temperature=args.temperature, seed=seed,
                    )
                    break
                except Exception as e:
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

            # append the assistant turn verbatim (content + tool_calls)
            amsg = {"role": "assistant", "content": msg.content or ""}
            if tcs:
                amsg["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name,
                                  "arguments": tc.function.arguments or "{}"}}
                    for tc in tcs]
            messages.append(amsg)

            if not tcs:
                # model emitted prose only — nudge it back to tools
                messages.append({"role": "user",
                                 "content": "Use the tools to make progress: edit_lines, "
                                            "run_tests, then done. Do not answer in prose."})
                continue

            for tc in tcs:
                try:
                    a = json.loads(tc.function.arguments or "{}")
                except Exception:
                    a = {}
                result = ro.execute(tc.function.name, a)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": str(result)})

            # R2 (--auto-feedback): the environment VOLUNTEERS the visible-test result after
            # any edit, so the agent gets the execution signal for free without electing to
            # run_tests. Skip if it already ran the test itself this turn (no double-run).
            names_this_turn = [tc.function.name for tc in tcs]
            if (args.auto_feedback and "edit_lines" in names_this_turn
                    and "run_tests" not in names_this_turn):
                tr = env.run_tests()
                ro.last_test = tr
                ro.counts["auto_test"] += 1
                ro.trace.append({"t": "auto_test", "resolved": bool(tr.get("resolved"))})
                fb = ("ALL TESTS PASS." if tr.get("resolved")
                      else f"FAIL: {tr.get('failure', 'test failed')}")
                messages.append({"role": "user",
                                 "content": f"[automatic test run after your edit] {fb}"})

            if ro.done:
                stop_reason = "done"
                break
            # Auto-stop when the visible test passes — EXCEPT for held-out tasks, where the visible
            # test is only a partial spec; there the agent must call done() itself (so it has the
            # chance to verify, e.g. with check_types, before shipping a latent bug).
            if ro.last_test and ro.last_test.get("resolved") and (not task.get("held_out") or args.no_hint):
                stop_reason = "tests_pass"
                break

        # final ground-truth resolution. `resolved` is the HELD-OUT score (the real correctness);
        # `visible_pass` is the test the agent could run. They differ when the agent ships a latent
        # bug the visible test misses but a checker could have caught (the Gap D fair-test design).
        visible = env.run_tests()
        final = env.score()
        resolved = bool(final.get("resolved"))
        visible_pass = bool(visible.get("resolved"))

        row = {
            "model": model, "task": task["name"], "group": task.get("group"),
            "seed": seed, "no_defn": args.no_defn, "with_check": args.with_check,
            "resolved": resolved, "visible_pass": visible_pass, "stop_reason": stop_reason,
            "no_test": args.no_test, "auto_feedback": args.auto_feedback,
            "n_read": ro.counts["read"], "n_defn": ro.counts["defn"],
            "n_findrefs": ro.counts["findrefs"], "n_test": ro.counts["test"],
            "n_auto_test": ro.counts["auto_test"],
            "n_edit": ro.counts["edit"], "n_check": ro.counts["check"],
            "turns": turns, "prompt_tokens": in_tok, "completion_tokens": out_tok,
            "est_cost_usd": round(est_cost, 6), "trace": ro.trace,
            "final_failure": "" if resolved else final.get("failure", "")[:300],
        }
        return row, est_cost
    finally:
        env.close()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--suite", required=True,
                    choices=["effic_real2", "gapd", "gapd2", "runtime"])
    ap.add_argument("--no-hint", action="store_true",
                    help="REALISTIC deployment: omit the 'visible test is partial' note and stop when "
                         "the visible test passes (measures the natural latent-bug rate on held-out tasks)")
    ap.add_argument("--names", default=None, help="comma-separated task names subset")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--no-defn", action="store_true",
                    help="tool-value ablation: defn/find_references genuinely absent (read-only)")
    ap.add_argument("--no-test", action="store_true",
                    help="runtime-feedback ablation R0: NO run_tests tool — the agent cannot execute "
                         "the code and must reason from the source + the shown spec, then call done()")
    ap.add_argument("--auto-feedback", action="store_true",
                    help="runtime-feedback arm R2: the env volunteers the visible-test result after "
                         "every edit (execution handed over for free, no election needed)")
    ap.add_argument("--with-check", action="store_true",
                    help="offer the check_types tool (Gap D info condition; default OFF)")
    ap.add_argument("--budget-usd", type=float, default=5.0)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--max-reads", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    key = load_or_key()

    from openai import OpenAI
    client = OpenAI(base_url=OR_BASE, api_key=key,
                    default_headers={"HTTP-Referer": "https://localhost/streams",
                                     "X-Title": "streams-api-agent"})

    pp, cp, matched = resolve_pricing(args.model)
    price = (pp, cp)

    TASKS = load_suite(args.suite)
    if args.names:
        want = set(args.names.split(","))
        tasks = [t for t in TASKS if t["name"] in want]
    else:
        tasks = list(TASKS)
    if not tasks:
        print(f"[fatal] no tasks matched --names {args.names}", file=sys.stderr)
        sys.exit(2)

    print(f"[run] model={args.model} suite={args.suite} tasks={[t['name'] for t in tasks]} "
          f"seeds={args.seed_start}..{args.seed_start+args.seeds-1} no_defn={args.no_defn} "
          f"with_check={args.with_check} budget=${args.budget_usd}", flush=True)

    rows = []
    spent = 0.0
    budget_hit = False

    def checkpoint():
        meta = {"model": args.model, "matched_pricing_id": matched, "suite": args.suite,
                "config": vars(args), "total_cost_usd": round(spent, 6),
                "budget_hit": budget_hit, "rows": rows}
        with open(args.out + ".partial", "w") as f:
            json.dump(meta, f, indent=2)
        return meta

    for task in tasks:
        if budget_hit:
            break
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            if spent >= args.budget_usd:
                print(f"[BUDGET HIT] cumulative ${spent:.4f} >= ${args.budget_usd}; stopping.",
                      flush=True)
                budget_hit = True
                break
            t0 = time.time()
            row, cost = run_rollout(client, args.model, task, seed, args, price, spent)
            spent += cost
            row["sec"] = round(time.time() - t0, 1)
            rows.append(row)
            print(f"  [{task['name']:26}] s{seed}: resolved={row['resolved']} "
                  f"stop={row['stop_reason']} reads={row['n_read']} defn={row['n_defn']} "
                  f"tests={row['n_test']} edits={row['n_edit']} turns={row['turns']} "
                  f"in={row['prompt_tokens']} out={row['completion_tokens']} "
                  f"cost=${row['est_cost_usd']:.5f} (cum ${spent:.4f})", flush=True)
            checkpoint()
            if row["stop_reason"] == "budget":
                print(f"[BUDGET HIT] cumulative ${spent:.4f} >= ${args.budget_usd}; stopping.",
                      flush=True)
                budget_hit = True
                break

    meta = checkpoint()
    with open(args.out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n=== done: {len(rows)} rollouts, total ${spent:.4f} "
          f"({'BUDGET HIT' if budget_hit else 'within budget'}) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
