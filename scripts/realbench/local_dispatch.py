#!/usr/bin/env python3
"""Local-model DISPATCH-AMBIGUITY runner (efficiency + election arc, real language server).

Mirrors scripts/synth_mf.py but over the on-disk dispatch repos of dispatch_tasks.py and a real
pyrefly LSP. For each task x condition x seed it drives the StreamAgent and records INPUT tokens
(the efficiency metric) and whether the model ELECTS the cheap <defn> action.

Conditions (retrieval-tool advertisement is the only variable; capability held fixed):
  grep_base   : grep + ranged read only. <defn> genuinely unavailable (lsp_disabled).
  defn_avail  : same + <defn> advertised NEUTRALLY, one tool among others.
  defn_prompt : same tools, but the advertisement FRAMES <defn> as the cheaper/precise way to
                find which override binds for the receiver.

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/mnt/nas/hf-cache \
    .venv-streams.system/bin/python scripts/realbench/local_dispatch.py \
      --model Qwen/Qwen2.5-Coder-7B-Instruct --conds grep_base,defn_avail,defn_prompt --gpu-only
"""
import os
import re
import sys
import json
import time
import argparse

os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scaffold.stream_agent import StreamAgent
from scripts.realbench.dispatch_tasks import build_tasks, make_env


# --------------------------------------------------------------------------- condition prompts
_HEADER = (
    "You are a coding agent fixing a ONE-LINE bug in a Python package. A method is overridden on many "
    "classes; exactly ONE override -- the one that runs for the receiver in `{file}` -- has a wrong line. "
    "You may edit only these files: {files}.\n\nTools (emit the tag; you get the result back, then continue):\n"
)

_T_GREP = '  <grep pat="regex"/>            search all files; returns `file:line: text` hits (add path="F" to scope to one file).\n'
_T_DEFN = ('  <defn sym="NAME" file="F" line="N" col="C"/>  show the definition of NAME; given a use-site F:line:col '
           'it returns the definition that binds there.\n')
_T_READ = ('  <read path="F" lines="A-B"/>   read lines A-B of F, numbered. Bare <read path="F"/> reads the whole file.\n')
_T_EDIT = ('  <edit path="F" lines="A-B">    replace lines A-B of F with the code on the following lines then </edit>\n'
           '     (write the new code without ``` fences or NNN| prefixes; use F\'s latest numbered view).\n')
_T_TEST = "  <test/>                          run the tests; you get the pass/fail result.\n"
_T_DONE = "  <done/>                          emit once the tests pass.\n"

_FOOT_NEUTRAL = (
    "\nWork iteratively: find WHICH class's method runs for the receiver in `{file}`, read that one "
    "override, fix the single wrong line, then <test/>. Reason briefly between actions."
)
_FOOT_DEFN = (
    "\nThe cheapest way to find WHICH override binds: rather than grepping the many `def NAME` candidates "
    "and reading each class, emit ONE <defn sym=\"NAME\" file=\"{file}\" line=\"L\" col=\"C\"/> at the "
    "call site in `{file}` -- it returns the single definition that runs for that receiver's static type. "
    "Prefer it, then fix the single wrong line and <test/>. Reason briefly between actions."
)

SYS = {
    "grep_base":   _HEADER + _T_GREP + _T_READ + _T_EDIT + _T_TEST + _T_DONE + _FOOT_NEUTRAL,
    "defn_avail":  _HEADER + _T_GREP + _T_DEFN + _T_READ + _T_EDIT + _T_TEST + _T_DONE + _FOOT_NEUTRAL,
    "defn_prompt": _HEADER + _T_GREP + _T_DEFN + _T_READ + _T_EDIT + _T_TEST + _T_DONE + _FOOT_DEFN,
}


def _numbered(src):
    return "\n".join("%3d| %s" % (i + 1, ln) for i, ln in enumerate(src.splitlines()))


def build_prompt(task, env):
    """Same across conditions: task description + the failing test (the spec/symptom) + the call site."""
    us = task["use_site"]
    app_src = env.read_file(task["target_file"])
    test_src = env.read_file("test_dispatch.py")
    sym = task["symbol"]
    return (
        "A method `%s` is overridden on %d classes across the files you may edit. Exactly one override -- "
        "the one that RUNS for the statically-typed receiver in `%s` -- returns the wrong result, so the "
        "test below fails.\n\n"
        "`%s` (the call site, numbered):\n%s\n\n"
        "In `%s` the receiver `x` is type-annotated and the call `x.%s(...)` is on line %d; the method "
        "name `%s` starts at column %d (use these as the use-site if you look the definition up).\n\n"
        "The failing test (do NOT edit it; it is the spec):\n```python\n%s```\n\n"
        "The fix is a ONE-LINE change in the correct override. Find WHICH class's `%s` runs for the "
        "receiver in `%s`, fix the single wrong line in that override, then run <test/> until it passes."
        % (sym, task["n_overrides"], task["target_file"],
           task["target_file"], _numbered(app_src),
           task["target_file"], sym, us["line"], sym, us["col"],
           test_src, sym, task["target_file"])
    )


# --------------------------------------------------------------------------- event accounting
def count_events(events):
    n_grep = sum(1 for e in events if e.get("type") == "grep")
    n_defn = sum(1 for e in events if e.get("type") == "defn")
    n_read_whole = sum(1 for e in events if e.get("type") == "read" and not e.get("ranged"))
    n_read_ranged = sum(1 for e in events if e.get("type") == "read" and e.get("ranged"))
    return n_grep, n_defn, n_read_whole, n_read_ranged


def modeltag(model, adapter):
    tag = os.path.basename(model.rstrip("/")).replace("/", "_")
    if adapter:
        tag += "__" + os.path.basename(adapter.rstrip("/"))
    return re.sub(r"[^A-Za-z0-9_.-]", "_", tag)


def _mean(xs):
    return round(sum(xs) / len(xs), 1) if xs else None


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--conds", default="grep_base,defn_avail,defn_prompt")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--gpu-only", action="store_true")
    ap.add_argument("--max-new", type=int, default=2600)
    ap.add_argument("--max-reads", type=int, default=12)
    ap.add_argument("--max-turns", type=int, default=16)
    ap.add_argument("--tmp-root", default="/tmp/streams_dispatch")
    ap.add_argument("--typing", default="annotated",
                    choices=["annotated", "stripped", "indirection"],
                    help="receiver-type ablation level (default annotated = current behavior)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--names", default=None, help="comma-separated task names subset")
    args = ap.parse_args()

    conds = args.conds.split(",")
    n_seeds = 1 if args.temp == 0 else args.seeds

    print("[build] materializing dispatch repos (typing=%s) under %s"
          % (args.typing, args.tmp_root), flush=True)
    tasks = build_tasks(args.tmp_root, typing=args.typing)
    if args.names:
        want = set(args.names.split(","))
        tasks = [t for t in tasks if t["name"] in want]
    K = len(tasks)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("[load] %s%s  temp=%s seeds=%d gpu_only=%s"
          % (args.model, (" + " + args.adapter) if args.adapter else "", args.temp, n_seeds, args.gpu_only),
          flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    _dm = {"": 0} if args.gpu_only else "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=_dm)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model = model.eval()

    out = args.out or os.path.join(ROOT, "runs/realbench/dispatch",
                                   "local_%s.json" % modeltag(args.model, args.adapter))
    os.makedirs(os.path.dirname(out), exist_ok=True)

    rows = []
    for task in tasks:
        for c in conds:
            for seed in range(args.seed_start, args.seed_start + n_seeds):
                env = make_env(task)
                try:
                    agent = StreamAgent(
                        model, tok, env, edit_mode="line",
                        max_new_tokens=args.max_new, max_reads=args.max_reads,
                        max_turns=args.max_turns, temperature=args.temp, seed=seed,
                        use_lsp_defn=(c != "grep_base"), lsp_disabled=(c == "grep_base"),
                        sys_override=SYS[c])
                    t0 = time.time()
                    r = agent.run(build_prompt(task, env), task["target_file"], editable=task["editable"])
                    dt = time.time() - t0
                    n_grep, n_defn, n_read_whole, n_read_ranged = count_events(r["events"])
                    row = {"task": task["name"], "cond": c, "seed": seed,
                           "resolved": bool(r["resolved"]), "bailed": r.get("bailed"),
                           "in_toks": r["in_tokens"], "out_toks": r["out_tokens"],
                           "n_grep": n_grep, "n_defn": n_defn,
                           "n_read_whole": n_read_whole, "n_read_ranged": n_read_ranged,
                           "n_edits": r["n_edits"], "n_tests": r["n_tests"], "turns": r["turns"],
                           "sec": round(dt, 1), "stream_tail": r["stream"][-2500:],
                           "events": r["events"]}
                    rows.append(row)
                    print("  [%-15s] %-11s s%d: resolved=%s in_toks=%d grep=%d defn=%d "
                          "read=%d/%d edits=%d tests=%d (%.0fs)"
                          % (task["name"], c, seed, row["resolved"], row["in_toks"], n_grep, n_defn,
                             n_read_whole, n_read_ranged, row["n_edits"], row["n_tests"], dt), flush=True)
                finally:
                    env.close()
        json.dump({"model": args.model, "adapter": args.adapter, "config": vars(args), "rows": rows},
                  open(out, "w"), indent=2)

    # ----------------------------------------------------------------- summary
    print("\n=== per-condition summary (K=%d tasks x %d seeds) ===" % (K, n_seeds), flush=True)
    by = {c: [r for r in rows if r["cond"] == c] for c in conds}
    summary = {}
    for c in conds:
        rs = by[c]
        res = [r for r in rs if r["resolved"]]
        summary[c] = {
            "resolved": "%d/%d" % (len(res), len(rs)),
            "mean_in_toks": _mean([r["in_toks"] for r in rs]),
            "mean_in_toks_resolved": _mean([r["in_toks"] for r in res]),
            "mean_n_defn": _mean([r["n_defn"] for r in rs]),
            "mean_n_grep": _mean([r["n_grep"] for r in rs]),
            "mean_read_whole": _mean([r["n_read_whole"] for r in rs]),
            "mean_read_ranged": _mean([r["n_read_ranged"] for r in rs]),
        }
        s = summary[c]
        print("  %-11s resolved=%-5s  mean_in_toks=%-7s (resolved=%-7s)  "
              "n_defn=%-4s n_grep=%-4s read_whole=%-4s read_ranged=%-4s"
              % (c, s["resolved"], s["mean_in_toks"], s["mean_in_toks_resolved"],
                 s["mean_n_defn"], s["mean_n_grep"], s["mean_read_whole"], s["mean_read_ranged"]),
              flush=True)

    # matched-success paired EFFICIENCY DELTA: grep_base vs each defn condition, on tasks resolved in
    # BOTH. Reports the ratio, the mean signed token delta (>0 = defn cheaper), and the per-task pairs so
    # the distribution is visible, not just the mean.
    def _paired(base_c, defn_c):
        b = {(r["task"], r["seed"]): r for r in by.get(base_c, []) if r["resolved"]}
        d = {(r["task"], r["seed"]): r for r in by.get(defn_c, []) if r["resolved"]}
        return [{"task": k[0], "seed": k[1], "base_in_toks": b[k]["in_toks"],
                 "defn_in_toks": d[k]["in_toks"], "delta": b[k]["in_toks"] - d[k]["in_toks"]}
                for k in sorted(set(b) & set(d))]

    deltas = {}
    for defn_c in [c for c in conds if c != "grep_base"]:
        if "grep_base" not in by:
            break
        pairs = _paired("grep_base", defn_c)
        if not pairs:
            print("\n  [%s vs grep_base] no task resolved in BOTH conditions" % defn_c, flush=True)
            deltas[defn_c] = {"n": 0}
            continue
        gb_m = _mean([p["base_in_toks"] for p in pairs])
        dn_m = _mean([p["defn_in_toks"] for p in pairs])
        d_m = _mean([p["delta"] for p in pairs])
        ratio = round(gb_m / dn_m, 3) if dn_m else None
        deltas[defn_c] = {"n": len(pairs), "grep_base_mean": gb_m, "defn_mean": dn_m,
                          "mean_delta_toks": d_m, "ratio_grep_over_defn": ratio, "pairs": pairs}
        print("\n  [%s vs grep_base] matched n=%d: grep_base=%s  %s=%s  ratio=%s  "
              "mean_delta=%s toks (>0 = defn cheaper)"
              % (defn_c, len(pairs), gb_m, defn_c, dn_m, ratio, d_m), flush=True)
        for p in pairs:
            print("      %-18s grep_base=%-6d %s=%-6d delta=%+d"
                  % (p["task"], p["base_in_toks"], defn_c, p["defn_in_toks"], p["delta"]), flush=True)

    summary["efficiency_deltas"] = deltas
    json.dump({"model": args.model, "adapter": args.adapter, "config": vars(args),
               "summary": summary, "rows": rows}, open(out, "w"), indent=2)
    print("\n-> %s" % out, flush=True)


if __name__ == "__main__":
    main()
