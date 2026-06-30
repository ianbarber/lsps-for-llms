#!/usr/bin/env python3
"""Condition runner for the MULTI-FILE efficiency suite.

The prompt shows ONLY the target file (numbered) + the names of the other
workspace files (readable via <read path=.../>) + the behavioural test (the spec).
Type definitions the target misuses live in the unshown files, so the checker's
diagnostics carry information the model does not have in context.

Final-recipe invocation (condition A only):
  python scripts/synth_mf.py out.json --suite effic|efficread|effmix \
      --model MODEL --conds A --lsp-tools [--force-lsp --relabel --save-sft] \
      [--lsp-defn] [--adapter PATH] [--names ...] [--seeds K] [--seed-start S]
"""
import os, sys, json, time, argparse
os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scaffold.stream_agent import StreamAgent
from scaffold.mock_env import MultiFileEnv
from scripts.synth_tasks_effic import TASKS_EFFIC  # EFFICIENCY-as-policy: prefer cheap <defn> over reading a big lib
from scripts.synth_tasks_effic_real import TASKS_EFFIC_REAL  # REAL-CODE effic: same shape, real vendored library source
from scripts.synth_tasks_effic_real2 import TASKS_EFFIC_REAL2  # REAL-CODE effic, UN-MEMORIZED obscure-tail symbols
from scripts.synth_tasks_gapd import TASKS_GAPD  # GAP D: inference-hard tasks — is type-checker INFERENCE non-redundant?
from scripts.synth_tasks_efficread import TASKS_EFFICREAD  # READ-REQUIRED boundary: <defn> insufficient, must <read>

ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="effic",
                choices=["effic", "effic_real", "effic_real2", "gapd", "efficread", "effmix"],
                help="task suite: effic (prefer cheap <defn>), efficread (read-required boundary), "
                     "effmix (effic + efficread), gapd (type-inference info channel)")
ap.add_argument("--lsp-tools", action="store_true",
                help="advertise PULL LSP actions <defn sym=.../> and <findrefs sym=.../> alongside <read>")
ap.add_argument("--dry-run", action="store_true",
                help="render prompts for each task and exit (no model load; prompt QA)")
ap.add_argument("--save-sft", action="store_true",
                help="store sft_input_ids/sft_labels in each row -> training-set harvest")
ap.add_argument("--force-lsp", action="store_true",
                help="deny <read> of non-editable files (forces the model onto <defn>/<findrefs>)")
ap.add_argument("--relabel", action="store_true",
                help="genuine on-policy relabel: mask the read-attempt tokens and keep the model's own <defn>")
ap.add_argument("--lsp-defn", action="store_true",
                help="back <defn> with a LIVE pyrefly LSP daemon (env.lsp_definition) instead of the AST resolver")
ap.add_argument("--no-defn", action="store_true",
                help="TOOL-VALUE ABLATION: make <defn>/<findrefs> genuinely unavailable (read-only condition)")
ap.add_argument("out", nargs="?", default="runs/agent/mf_run.json")
ap.add_argument("--conds", default="A", choices=["A"],
                help="condition(s) to run; the final recipe uses A only")
ap.add_argument("--names", default=None)
ap.add_argument("--seeds", type=int, default=1)
ap.add_argument("--seed-start", type=int, default=0)
ap.add_argument("--temp", type=float, default=0.7)
ap.add_argument("--adapter", default=None)
ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
ap.add_argument("--gpu-only", action="store_true",
                help="force ALL weights onto cuda:0 (device_map={'':0}) instead of auto")
ap.add_argument("--max-new", type=int, default=2200)
ap.add_argument("--max-reads", type=int, default=6,
                help="cap on <read> calls")
ap.add_argument("--max-turns", type=int, default=12,
                help="cap on agent turns (reads/tests/edits)")
A = ap.parse_args()

TASKS_MF = {"effic": TASKS_EFFIC, "effic_real": TASKS_EFFIC_REAL, "effic_real2": TASKS_EFFIC_REAL2,
            "gapd": TASKS_GAPD, "efficread": TASKS_EFFICREAD,
            "effmix": (TASKS_EFFIC + TASKS_EFFICREAD)}[A.suite]
tasks = TASKS_MF if not A.names else [t for t in TASKS_MF if t["name"] in set(A.names.split(","))]
conds = A.conds.split(",")
n_seeds = 1 if A.temp == 0 else A.seeds

if not A.dry_run:
    print(f"[load] {A.model}{' + '+A.adapter if A.adapter else ''}  temp={A.temp} "
          f"seeds={A.seed_start}..{A.seed_start+n_seeds-1}", flush=True)
    tok = AutoTokenizer.from_pretrained(A.model)
    _dm = {"": 0} if A.gpu_only else "auto"
    model = AutoModelForCausalLM.from_pretrained(A.model, dtype=torch.bfloat16, device_map=_dm)
    if A.adapter:
        from peft import PeftModel; model = PeftModel.from_pretrained(model, A.adapter)
    model = model.eval()

def task_meta(task):
    """(target, editable, gold_map, shown) for all schemas:
      mf2/mf3 single-target (target/gold_target); mf4 multi-target blast-radius (targets/golds);
      partial-visibility (targets/golds + `shown` = subset of targets shown numbered up front)."""
    if "targets" in task:
        targets = list(task["targets"])
        shown = list(task.get("shown", targets))   # default: all targets shown (full visibility)
        return targets[0], targets, task["golds"], shown
    return task["target"], [task["target"]], {task["target"]: task["gold_target"]}, [task["target"]]

def _numbered(src):
    return "\n".join(f"{i+1:>3}| {ln}" for i, ln in enumerate(src.splitlines()))

def count_attractor_edits(events, attractors):
    """PREVENTION metric: # of applied edits whose new code emits a hallucinated symbol."""
    import re as _re
    if not attractors:
        return 0
    pats = []
    for a in attractors:
        if _re.fullmatch(r"\w+", a):
            pats.append(_re.compile(r"\." + _re.escape(a) + r"\b"))
        else:
            try: pats.append(_re.compile(a))
            except _re.error: pats.append(_re.compile(_re.escape(a)))
    n = 0
    for e in events:
        if e.get("type") in ("edit", "line_edit") and e.get("ok"):
            txt = e.get("replace", "")
            if any(p.search(txt) for p in pats):
                n += 1
    return n

def build_prompt(task):
    lsp_line = ("\nLanguage-server actions (cheap — return only the relevant lines): "
                "<defn sym=\"NAME\"/> shows a symbol's definition/signature; "
                "<findrefs sym=\"NAME\"/> lists where it is used."
                if A.lsp_tools else "")
    target, editable, _, shown = task_meta(task)
    others = [f for f in sorted(task["files"]) if f not in shown]
    multi = len(editable) > 1
    partial = multi and set(shown) != set(editable)   # some editable files are HIDDEN -> discovery
    body = "\n\n".join(f"`{f}`:\n{_numbered(task['files'][f])}" for f in shown)
    if not multi:
        head = f"Fix the bug(s) in `{target}` so the test below passes.\n\n{body}\n\n"
        tail = f"Make line-range edits to `{target}`, then run <test/>."
    elif partial:
        head = (f"Fix the bug so the test passes. The buggy symbol appears in `{shown[0]}` (shown below) AND "
                f"likely in OTHER modules of the package — you must find and fix EVERY module that uses it, "
                f"or the tests will still fail.\n\n{body}\n\n")
        tail = ("Inspect other modules with <read path=\"...\"/> (you get a numbered, editable view), then make "
                "line-range <edit path=\"FILE\" lines=\"START-END\"> edits to every affected module, then <test/>.")
    else:
        head = ("Fix the bug(s) so the test below passes. The bug SPANS these editable files: "
                f"{', '.join('`'+f+'`' for f in editable)} — fix EVERY affected site.\n\n{body}\n\n")
        tail = ("Make line-range edits with <edit path=\"FILE\" lines=\"START-END\"> to each "
                "affected file, then run <test/>.")
    if others:
        noun = "modules" if multi else "files"
        whereabouts = (f"The workspace also contains these {noun}: {', '.join(others)} — you have NOT seen "
                       f"their contents; inspect any with <read path=\"...\"/>.\n\n")
    else:
        whereabouts = ""
    return (head + whereabouts +
            "The test that must pass (do NOT edit it; it is the spec):\n"
            f"```python\n{task['test']}\n```\n" + tail + lsp_line)

agg = {c: {"rows": []} for c in conds}
os.makedirs(os.path.dirname(A.out), exist_ok=True)
def checkpoint():
    json.dump({"model": A.model, "config": vars(A),
               "rows": {c: agg[c]["rows"] for c in conds}}, open(A.out + ".partial", "w"))

if A.dry_run:
    for task in tasks:
        print(f"\n{'='*30} {task['name']} ({task['group']}) {'='*30}")
        print("--- cond A prompt ---\n" + build_prompt(task))
        print("   attractors:", task.get("attractors", []))
    sys.exit(0)

def advertised_symbols(task):
    """Symbols the agent may query when a read is force-LSP-blocked. These are the
    symbol NAMES from the task (already visible in the prompt via imports), not gold definitions."""
    if "symbols" in task:
        return list(task["symbols"])
    sym = task.get("symbol")
    return [sym] if sym else []

for task in tasks:
    target, editable, gold_map, _shown = task_meta(task)
    for c in conds:
        for seed in range(A.seed_start, A.seed_start + n_seeds):
            # cond A never reads pyrefly_diagnostics; skip `pyrefly init` (esp. under --lsp-defn, where the
            # persistent `pyrefly lsp` daemon backs <defn> and a second init-daemon would contend on the socket).
            env = MultiFileEnv(task["files"], target, task["test"], skip_pyrefly=True)
            agent = StreamAgent(model, tok, env,
                                max_new_tokens=A.max_new, max_reads=A.max_reads,
                                max_turns=A.max_turns, edit_mode="line",
                                temperature=A.temp, seed=seed,
                                force_lsp=A.force_lsp, relabel=A.relabel,
                                advertised_symbols=(advertised_symbols(task) if A.lsp_tools else []),
                                use_lsp_defn=A.lsp_defn, lsp_disabled=A.no_defn)
            t0 = time.time()
            r = agent.run(build_prompt(task), target, editable=editable)
            dt = time.time() - t0
            m = r["metrics"]
            row = {"task": task["name"], "group": task["group"], "cond": c, "seed": seed,
                   "resolved": bool(r["resolved"]), "bailed": r.get("bailed"),
                   "in_tokens": r["in_tokens"], "out_tokens": r["out_tokens"],
                   "sec": round(dt, 1), "rework_ratio": m.get("rework_ratio"),
                   "n_edits": m.get("n_edits"), "n_tests": r["n_tests"],
                   "n_reads": r["n_reads"], "n_greps": r.get("n_greps", 0), "n_lsp": r.get("n_lsp", 0),
                   "turns": r["turns"],
                   "n_attractor_edits": count_attractor_edits(r["events"], task.get("attractors", [])),
                   "stream_tail": r["stream"][-3000:], "events": r["events"]}
            if A.save_sft:
                row["sft_input_ids"] = r["sft_input_ids"]; row["sft_labels"] = r["sft_labels"]
                row["n_train_tokens"] = r.get("n_train_tokens")
            agg[c]["rows"].append(row)
            env.close()
            print(f"  [{task['name']:22}] {c} s{seed}: resolved={row['resolved']} "
                  f"reads={row['n_reads']} tests={row['n_tests']} edits={row['n_edits']} "
                  f"out={row['out_tokens']} ({row['sec']}s)", flush=True)
    checkpoint()

print("\n=== aggregate ===", flush=True)
summary = {}
for c in conds:
    rs = agg[c]["rows"]; res = [r for r in rs if r["resolved"]]
    bygrp = {}
    for g in ("plain", "rich", "control"):
        sub = [r for r in rs if r["group"] == g]
        bygrp[g] = f"{sum(r['resolved'] for r in sub)}/{len(sub)}"
    summary[c] = {"resolve_rate": round(len(res)/len(rs), 3) if rs else 0, "n": len(rs),
                  "by_group": bygrp,
                  "mean_reads": round(sum(r['n_reads'] for r in rs)/max(len(rs),1), 2)}
    print(f"  {c}: resolve={summary[c]['resolve_rate']} ({len(res)}/{len(rs)})  "
          f"by_group={bygrp}  mean_reads={summary[c]['mean_reads']}", flush=True)

json.dump({"model": A.model, "adapter": A.adapter, "config": vars(A), "summary": summary,
           "rows": {c: agg[c]["rows"] for c in conds}}, open(A.out, "w"), indent=2)
print(f"-> {A.out}", flush=True)
