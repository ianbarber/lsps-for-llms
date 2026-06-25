#!/usr/bin/env python3
"""Condition runner for the MULTI-FILE suite (non-redundant-channel setting).

The prompt shows ONLY the target file (numbered) + the names of the other
workspace files (readable via <read path=.../>) + the behavioural test (the spec).
Type definitions the target misuses live in the unshown files, so the checker's
diagnostics carry information the model does not have in context.

PAPER CONDITION -> INVOCATION (same flags as synth_acd.py):
  A        : --conds A
  C-eager  : --conds C --c-eager
  D-gate   : --conds D --debounce 24 --pause-align --syntax-gate
  +rich    : add --rich-signal   (appends remote definitions — the key arm here)

Usage: synth_mf.py [out.json] [--conds ...] [--seeds K] [--seed-start S] [...]
"""
import os, sys, json, time, argparse
os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scaffold.stream_agent import StreamAgent
from scaffold.mock_env import MultiFileEnv
from scripts.synth_tasks_effic import TASKS_EFFIC  # EFFICIENCY-as-policy: prefer cheap <defn> over reading a big lib
from scripts.synth_tasks_efficread import TASKS_EFFICREAD  # READ-REQUIRED boundary: <defn> insufficient, must <read>
from scripts.synth_tasks_effic_nodel import TASKS_EFFIC_ND  # NODEL no-delegation coverage probe (see log 2026-06-24)
try:
    from scripts.synth_tasks_cover import TASKS_COVER  # COVERAGE-JUDGING: read-on-insufficient-defn (suf/f1ins/f2ins)
except Exception:
    TASKS_COVER = []
try:
    from scripts.synth_tasks_cover2 import TASKS_COVER2  # COVERAGE v2: insufficiency is NON-DEFN-CHAINABLE (read forced)
except Exception:
    TASKS_COVER2 = []

ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="effic",
                choices=["effic", "efficread", "effmix", "effic_nodel", "cover", "cover2"],
                help="task suite: effic (efficiency-as-policy: prefer cheap <defn> over reading a big lib), "
                     "efficread (read-required boundary: <defn> insufficient, must <read>), "
                     "effmix (effic + efficread), "
                     "effic_nodel (no-delegation coverage probe; see log 2026-06-24), "
                     "cover (coverage-judging: read-on-insufficient-defn, suf/f1ins/f2ins variants), "
                     "cover2 (coverage v2: insufficiency is NON-DEFN-CHAINABLE, a <read> is forced)")
ap.add_argument("--lsp-tools", action="store_true",
                help="advertise PULL LSP actions <defn sym=.../> and <findrefs sym=.../> (oracle-backed, cheap) "
                     "alongside <read> — the efficiency-as-policy lever (does the model prefer the LSP over reading?)")
ap.add_argument("--channel", default=None, choices=[None, "navrefs", "complete", "sighelp"],
                help="ORACLE-injected LSP channel for the nav/prevention experiments (delivered to cond C up front): "
                     "navrefs = find-refs site list; complete = true member list; sighelp = true signature. "
                     "Perfect upper-bound (no daemon); if it doesn't move pass@1/hallucination, the channel is redundant.")
ap.add_argument("--dry-run", action="store_true",
                help="render prompts + channel hints for each task/cond and exit (no model load; prompt QA)")
ap.add_argument("--save-sft", action="store_true",
                help="store sft_input_ids/sft_labels (prompt+observations masked, model actions trained) in each row "
                     "-> the OPSD harvest training set (filter to resolved + real-<defn> downstream)")
ap.add_argument("--force-lsp", action="store_true",
                help="OPSD HARVEST: deny <read> of non-editable files (under the normal prompt) so the model is "
                     "forced onto <defn>/<findrefs> -> manufactures the LSP-using solved trajectories to distill")
ap.add_argument("--lead-defn", action="store_true",
                help="DAgger round-0 harvest: inject <defn sym=task.symbol/> as the TRAINED first action (deployment "
                     "prompt, reads available), then let the model continue -> clean defn-first trajectories with NO "
                     "<read> for SFT to clone (the fix for the off-policy SFT that reinforced reading). Needs --lsp-tools.")
ap.add_argument("--relabel", action="store_true",
                help="GENUINE on-policy relabel (earns the DAgger framing, no injected gold action): roll out with "
                     "--force-lsp; when the model emits <read> of a non-editable file, MASK that turn's tokens from the "
                     "SFT labels and redirect -> the model CHOOSES <defn sym=...> itself. Use with --force-lsp --save-sft.")
ap.add_argument("--grep", action="store_true",
                help="advertise the <grep q=.../> workspace-search tool in the prompt (the agent's realistic "
                     "find-refs alternative; navigation experiment — aliasing makes grep of the canonical name miss sites)")
ap.add_argument("--lsp-defn", action="store_true",
                help="OPT-IN: back the <defn> action with a LIVE pyrefly LSP daemon (env.lsp_definition) instead of "
                     "the AST resolver (env.goto_definition). Validated to agree 12/12 on effic "
                     "(scripts/validate_pyrefly_lsp.py). SEQUENTIAL ONLY — pyrefly daemons deadlock under concurrency; "
                     "falls back to the AST resolver on any LSP error. Default path is unchanged.")
ap.add_argument("out", nargs="?", default="runs/agent/mf_run.json")
ap.add_argument("--conds", default="A,C,D")
ap.add_argument("--names", default=None)
ap.add_argument("--seeds", type=int, default=1)
ap.add_argument("--seed-start", type=int, default=0)
ap.add_argument("--temp", type=float, default=0.7)
ap.add_argument("--adapter", default=None)
ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
ap.add_argument("--gpu-only", action="store_true",
                help="force ALL weights onto cuda:0 (device_map={'':0}) instead of auto — avoids "
                     "CPU offload that cripples decode on big models on a unified-memory box")
ap.add_argument("--max-new", type=int, default=2200)
ap.add_argument("--max-reads", type=int, default=6,
                help="cap on <read> calls — at scale (many distractor modules) this forces the model "
                     "to rely on the diagnostic to find which modules have sites (partial-visibility lever)")
ap.add_argument("--max-turns", type=int, default=12,
                help="cap on agent turns (reads/tests/edits) — raise it so reads aren't turn-limited "
                     "when measuring cost-to-solve under a generous read budget")
ap.add_argument("--latency", type=int, default=8)
ap.add_argument("--debounce", type=int, default=0)
ap.add_argument("--pause-align", action="store_true")
ap.add_argument("--announce-lsp", action="store_true")
ap.add_argument("--c-eager", action="store_true")
ap.add_argument("--syntax-gate", action="store_true")
ap.add_argument("--rich-signal", action="store_true")
ap.add_argument("--preread", action="store_true",
                help="put ALL workspace files in the prompt (context-saturation ablation: "
                     "the diagnostic channel becomes informationally redundant)")
ap.add_argument("--clean-delivery", action="store_true",
                help="D: deliver live diagnostic as a clean user turn + file view (isolates format from timing)")
ap.add_argument("--diag-filter", default=None, choices=[None, "type", "syntax"],
                help="deliver only cross-file TYPE errors, or only self-inflicted SYNTAX/scope errors")
ap.add_argument("--oracle", default=None, choices=[None, "loc", "fix"],
                help="POSITIVE CONTROL: replace the pyrefly diagnostic with an oracle hint — "
                     "'loc' = perfect localization (which buggy lines, no fix); "
                     "'fix' = the gold replacement lines (the answer). Ceiling for helpful feedback.")
ap.add_argument("--steer", default=None, choices=[None, "gentle", "directive", "preferlsp"],
                help="append a system-prompt instruction steering the model to USE diagnostics / READ "
                     "the named file (tests latent-but-unelicited channel value vs a real null); "
                     "preferlsp = prefer the cheap <defn>/<findrefs> LSP actions over reading whole files")
A = ap.parse_args()

STEER = {
    "gentle": ("Note: the static type-checker's diagnostics are reliable — when one appears, treat "
               "it as a real problem and address it. If a diagnostic refers to a name (a field, "
               "function, or import) that comes from another file, reading that file is usually the "
               "quickest way to find the correct fix."),
    "directive": ("IMPORTANT: the static type-checker's diagnostics are authoritative — act on every "
                  "one. Whenever a diagnostic references a symbol (field, function, or import) defined "
                  "in another file, FIRST read that file with <read> to learn its correct name/type, "
                  "THEN edit. Do not guess at cross-file names."),
    "preferlsp": ("IMPORTANT: to learn a symbol's definition, signature, or members, PREFER the language-"
                  "server actions <defn sym=\"NAME\"/> (go-to-definition) and <findrefs sym=\"NAME\"/> over "
                  "reading whole files with <read> — they return exactly the relevant lines and cost far "
                  "fewer tokens. Only <read> a file when you genuinely need its full contents (e.g. to edit it)."),
}
steer_hint = STEER.get(A.steer)

import difflib
def build_oracle(buggy, gold, mode):
    """Oracle hint from the buggy target vs its gold fix (diff-based, auto, no manual authoring)."""
    a, b = buggy.splitlines(), gold.splitlines()
    ops = difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()
    if mode == "loc":
        lines = [n for tag, i1, i2, j1, j2 in ops if tag in ("replace", "delete")
                 for n in range(i1 + 1, i2 + 1)]
        if not lines:
            return "No changes needed."
        return ("Static analysis (oracle): the bug is on line(s) " + ", ".join(map(str, lines)) +
                " of the target file — a type/contract mismatch with the imported definitions. "
                "Fix those line(s).")
    hunks = []
    for tag, i1, i2, j1, j2 in ops:
        if tag in ("replace", "insert"):
            where = f"line {i1 + 1}" if i2 == i1 + 1 else f"lines {i1 + 1}-{i2}"
            hunks.append(f"# at {where}:\n" + "\n".join(b[j1:j2]))
        elif tag == "delete":
            hunks.append(f"# delete line(s) {i1 + 1}-{i2}")
    return "Suggested fix (oracle):\n" + "\n".join(hunks)

TASKS_MF = {"effic": TASKS_EFFIC, "efficread": TASKS_EFFICREAD,
            "effmix": (TASKS_EFFIC + TASKS_EFFICREAD), "effic_nodel": TASKS_EFFIC_ND,
            "cover": TASKS_COVER, "cover2": TASKS_COVER2}[A.suite]
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

def build_channel_hint(task, channel):
    """The ORACLE LSP channel injected into cond C up front (perfect upper-bound; no daemon).
      navrefs  : find-references — the resolved call sites of the symbol (defeats the alias).
      complete : completion/hover — the true member list of the in-scope class.
      sighelp  : signature-help — the true signature of the function to call."""
    if channel == "navrefs":
        refs = task.get("oracle_refs", task.get("targets", []))
        return ("Language server — find references: `" + task.get("sym", "") + "()` is referenced "
                "(resolved through the re-exports) in: " + ", ".join(refs) +
                ". These are every call site you must fix.")
    if channel == "complete":
        # task supplies the true public members of the symbol the target must use.
        api = task.get("api", {})
        body = "\n".join(f"  {name}{sig}" for name, sig in api.get("members", []))
        return (f"Language server — completion for `{api.get('symbol','')}`. Its ONLY public members are:\n"
                f"{body}\n(there are no others; do not call members not in this list.)")
    if channel == "sighelp":
        api = task.get("api", {})
        sigs = "\n".join(f"  {s}" for s in api.get("signatures", []))
        return (f"Language server — signature help:\n{sigs}\n(call with exactly these parameters / order.)")
    return None

import re as _re
def count_attractor_edits(events, attractors):
    """PREVENTION metric: # of applied edits whose new code emits a hallucinated symbol.
    A bare-identifier attractor (e.g. 'get') matches `.get` as a member access (word-boundary);
    anything with non-word chars (e.g. 'transfer(cents', a regex) is searched as a regex."""
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

def build_prompt(task, channel_hint=None):
    grep_line = ("\nTo SEARCH the whole workspace for a name: <grep q=\"name\"/> (returns path:line matches)."
                 if A.grep else "")
    lsp_line = ("\nLanguage-server actions (cheap — return only the relevant lines): "
                "<defn sym=\"NAME\"/> shows a symbol's definition/signature; "
                "<findrefs sym=\"NAME\"/> lists where it is used."
                if A.lsp_tools else "")
    chan = (f"\n\nLanguage-server context (authoritative):\n{channel_hint}" if channel_hint else "")
    if A.suite == "nav":
        # NAVIGATION: the call sites are NOT shown — the agent must FIND them. Show the definition,
        # the re-export hub, and the test entry as read-only context; never enumerate the sites.
        anchors = task.get("shown", [])
        body = "\n\n".join(f"`{f}`:\n{_numbered(task['files'][f])}" for f in anchors)
        others = [f for f in sorted(task["files"]) if f not in anchors]
        head = (f"The test below fails: helper functions spread across the package call `{task['sym']}()` "
                f"with its two arguments in the WRONG ORDER. Find EVERY call site and correct the argument "
                f"order so the test passes. The function and how it is re-exported are shown for reference:\n\n{body}\n\n")
        whereabouts = (f"Other workspace files (inspect any with <read path=\"...\"/> to get a numbered, editable "
                       f"view): {', '.join(others)}.\n\n")
        tail = (f"{grep_line}\nEdit a call site with <edit path=\"FILE\" lines=\"START-END\">, then run <test/>. "
                f"Keep going until every site is fixed and the test passes, then <done/>.")
        return (head + whereabouts +
                "The test that must pass (do NOT edit it; it is the spec):\n"
                f"```python\n{task['test']}\n```\n" + tail + grep_line + lsp_line + chan)
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
    if A.preread and others:
        ctx = "\n".join(f"`{f}`:\n```python\n{task['files'][f]}```" for f in others)
        whereabouts = f"The rest of the workspace (read-only, shown in full):\n{ctx}\n\n"
    elif others:
        noun = "modules" if multi else "files"
        whereabouts = (f"The workspace also contains these {noun}: {', '.join(others)} — you have NOT seen "
                       f"their contents; inspect any with <read path=\"...\"/>.\n\n")
    else:
        whereabouts = ""
    return (head + whereabouts +
            "The test that must pass (do NOT edit it; it is the spec):\n"
            f"```python\n{task['test']}\n```\n" + tail + grep_line + lsp_line + chan)

agg = {c: {"rows": []} for c in conds}
os.makedirs(os.path.dirname(A.out), exist_ok=True)
def checkpoint():
    json.dump({"model": A.model, "config": vars(A),
               "rows": {c: agg[c]["rows"] for c in conds}}, open(A.out + ".partial", "w"))

if A.dry_run:
    for task in tasks:
        ch = build_channel_hint(task, A.channel)
        print(f"\n{'='*30} {task['name']} ({task['group']}) {'='*30}")
        print("--- cond A prompt ---\n" + build_prompt(task))
        print("\n--- cond C channel_hint ---\n" + (ch or "(none)"))
        print("   attractors:", task.get("attractors", []))
    sys.exit(0)

for task in tasks:
    target, editable, gold_map, _shown = task_meta(task)
    oracle_hint = None
    if A.oracle:
        parts = []
        for f in editable:
            p = build_oracle(task["files"][f], gold_map[f], A.oracle)
            if p and "No changes needed" not in p:
                parts.append(f"In `{f}`: {p}" if len(editable) > 1 else p)
        oracle_hint = "\n".join(parts) if parts else None
    for c in conds:
        for seed in range(A.seed_start, A.seed_start + n_seeds):
            env = MultiFileEnv(task["files"], target, task["test"])
            agent = StreamAgent(model, tok, env, condition=c, latency_tokens=A.latency,
                                max_new_tokens=A.max_new, max_reads=A.max_reads,
                                max_turns=A.max_turns, edit_mode="line",
                                temperature=A.temp, seed=seed,
                                debounce=A.debounce, pause_align=A.pause_align,
                                announce_lsp=A.announce_lsp, c_eager=A.c_eager,
                                syntax_gate=A.syntax_gate, rich_signal=A.rich_signal,
                                clean_delivery=A.clean_delivery, diag_filter=A.diag_filter,
                                oracle_hint=oracle_hint, steer_hint=steer_hint,
                                lsp_oracle=(task.get("lsp_oracle") if A.lsp_tools else None),
                                force_lsp=A.force_lsp,
                                lead_defn=(task.get("symbol") if (A.lead_defn and task.get("defn_sufficient", True)) else None),
                                lead_read=("biglib.py" if (A.lead_defn and not task.get("defn_sufficient", True)) else None),
                                relabel=A.relabel, use_lsp_defn=A.lsp_defn)
            # ORACLE channel (nav/prevention): inject the LSP info into cond C only; A is the no-LSP baseline.
            channel_hint = build_channel_hint(task, A.channel) if (A.channel and c == "C") else None
            t0 = time.time()
            r = agent.run(build_prompt(task, channel_hint=channel_hint), target, editable=editable)
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
