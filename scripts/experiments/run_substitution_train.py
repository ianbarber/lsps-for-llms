#!/usr/bin/env python3
"""Substitution training: DAgger-style relabel harvest for "span replaces read".

Follow-up to the reread-after-span null (C31): an automatically delivered definition span is
reread rather than substituted, and an explicit sufficiency instruction does NOT remove the
reread (0/12 for the 27B, 2/36 across three models). Election was a TRAINABLE policy on a 7B
(C2, `scripts/run_relabel2.sh`). This driver asks the parallel question for SUBSTITUTION.

Design mirrors `run_relabel2.sh` exactly:
  1. HARVEST on training instances with the span auto-delivered (the `auto_neutral` arm of
     `run_navigation_reread.py`). When the model attempts to <read> the file the span came
     from, an oracle DENIES the read and tells it the span is complete; the model's own next
     action (normally the edit) becomes the first TRAINED action from a CLEAN prompt, because
     the read-attempt + redirect prefix is DROPPED from the SFT trace. Dropping (rather than
     masking) the prefix is the detail that made relabel2 work.
  2. SFT a LoRA adapter (`scripts/sft_lora.py --filter sft_keep`).
  3. RETEST on HELD-OUT instances via `run_navigation_reread.py --adapter`.

Frozen-protocol constraint: `scaffold/stream_agent.py` is hash-gated by
`scripts/run_navigation_confirmation.sh`, so it is NOT modified. The read-denial oracle is
installed by patching the ENV INSTANCE, and the relabel trace is mirrored bit-for-bit by
wrapping the agent instance's `_ids`/`_prefill` (the mirror is asserted equal to the agent's
own decoded stream before any row is kept). Semantics are identical to
`StreamAgent(relabel=True)`: `keep = len(out_ids)` immediately after the LAST redirect splice.

Splits: the training instances are FRESH navigation-v2 instances on a `substrain` split whose
seeds and templates are disjoint from every existing split. The reserved confirmation split
(41xxx) is never built or touched.

Usage:
  python scripts/experiments/run_substitution_train.py validate --out runs/protocol/....json
  python scripts/experiments/run_substitution_train.py harvest runs/agent/....json \
      --model Qwen/Qwen3.6-27B --revision 6a9e13bd... --temp 0.7 --seeds 3 --gpu-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scaffold.stream_agent import StreamAgent  # noqa: E402
from scaffold.tooling import find_pyrefly  # noqa: E402
from scripts.experiments import navigation_tasks as NT  # noqa: E402
from scripts.experiments.run_navigation import (  # noqa: E402
    AUTO_SYS,
    _method_from_lsp,
    _metrics,
)
from scripts.experiments.run_navigation_reread import _reread_metrics  # noqa: E402


# ---------------------------------------------------------------------------
# fresh TRAINING split: new seeds, new templates, disjoint from pilot/apparatus/confirmation.
# Injected at runtime so navigation_tasks.py stays byte-identical (it is hash-frozen by
# scripts/run_navigation_confirmation.sh).
# ---------------------------------------------------------------------------
TRAIN_SPLIT = "substrain"
TRAIN_SEEDS = (
    53003, 53017, 53023, 53029, 53047, 53051,
    53063, 53069, 53077, 53087, 53093, 53101,
)
TRAIN_TEMPLATES = ("add", "multiply")   # apparatus = subtract/affine/xor; confirmation = modulo/...


def install_train_split() -> None:
    reserved = set(NT.SPLIT_SEEDS["confirmation"])
    existing = {s for seeds in NT.SPLIT_SEEDS.values() for s in seeds}
    if set(TRAIN_SEEDS) & existing:
        raise SystemExit("training seeds collide with an existing navigation-v2 split")
    if set(TRAIN_SEEDS) & reserved:
        raise SystemExit("training seeds collide with the RESERVED confirmation split")
    if set(TRAIN_TEMPLATES) & set(NT.SPLIT_TEMPLATES["apparatus"]):
        raise SystemExit("training templates collide with the held-out apparatus templates")
    if set(TRAIN_TEMPLATES) & set(NT.SPLIT_TEMPLATES["confirmation"]):
        raise SystemExit("training templates collide with the reserved confirmation templates")
    NT.SPLIT_SEEDS[TRAIN_SPLIT] = TRAIN_SEEDS
    NT.SPLIT_TEMPLATES[TRAIN_SPLIT] = TRAIN_TEMPLATES


# ---------------------------------------------------------------------------
# the substitution oracle
# ---------------------------------------------------------------------------
class ReadNotNeeded(Exception):
    """Raised by the patched env when the model tries to open the file the span came from."""


REDIRECT_MARKER = "ReadNotNeeded"
REDIRECT_TEXT = (
    "the complete current source of the method that binds at the call site is ALREADY supplied "
    "in the <semantic_result> block, with its real file path and line numbers. Opening this file "
    "is unnecessary and is disabled here. Edit directly at the supplied line numbers with "
    "<edit path=\"...\" lines=\"START-END\"> ... </edit>, then run <test/>."
)

# stream_agent reaches env.read_file from several places. Only a MODEL <read>/<read lines> action
# is denied; internal post-edit file views and grep must keep working.
READ_ACTION_CALLERS = ("_file_view", "_read_range")


def install_read_oracle(env, target_path: str) -> dict:
    """Deny <read> of `target_path` until the model has applied its first edit."""
    import sys as _sys

    state = {"blocking": True, "n_denied": 0, "denied_before_first_edit": 0, "n_edits_ok": 0}
    orig_read = env.read_file
    orig_apply = env.apply_line_edit

    def read_file(path, *a, **kw):
        if state["blocking"] and path == target_path:
            caller = _sys._getframe(1).f_code.co_name
            if caller in READ_ACTION_CALLERS:
                state["n_denied"] += 1
                state["denied_before_first_edit"] += 1
                raise ReadNotNeeded(REDIRECT_TEXT)
        return orig_read(path, *a, **kw)

    def apply_line_edit(path, *a, **kw):
        res = orig_apply(path, *a, **kw)
        ok = res.ok if hasattr(res, "ok") else (res[0] if isinstance(res, tuple) else bool(res))
        if ok:
            state["n_edits_ok"] += 1
            state["blocking"] = False   # post-edit numbered views must still work
        return res

    env.read_file = read_file
    env.apply_line_edit = apply_line_edit
    return state


# ---------------------------------------------------------------------------
# relabel trace mirror (bit-for-bit equivalent of StreamAgent(relabel=True))
# ---------------------------------------------------------------------------
def install_trace_mirror(agent) -> dict:
    """Mirror stream_agent's out_ids/out_labels and record the DROP point.

    stream_agent builds:  out_ids += [nxt] per generated token (labels = the token)
                          out_ids += ids  per spliced observation (labels = -100)
    and, under --relabel, sets keep = len(out_ids) right after each redirect splice (the LAST
    one wins). `_prefill` is called exactly once per generated token with shape (1,1) and once
    per splice with shape (1,N>1), plus once for the prompt (first call). `_ids` carries the
    splice TEXT, so the redirect splice is identifiable.
    """
    m = {"prompt_ids": None, "ids": [], "labels": [], "keep": None, "n_splices": 0,
         "last_text": None}
    orig_ids = agent._ids
    orig_prefill = agent._prefill

    def _ids(text, special=False):
        m["last_text"] = text
        return orig_ids(text, special=special)

    def _prefill(input_ids, cache):
        n = int(input_ids.shape[1])
        if m["prompt_ids"] is None:
            m["prompt_ids"] = input_ids[0].tolist()
        elif n == 1:
            tokid = int(input_ids[0, 0])
            m["ids"].append(tokid)
            m["labels"].append(tokid)
        else:
            m["ids"].extend(input_ids[0].tolist())
            m["labels"].extend([-100] * n)
            m["n_splices"] += 1
            if REDIRECT_MARKER in (m["last_text"] or ""):
                m["keep"] = len(m["ids"])   # LAST redirect wins (relabel2 semantics)
        return orig_prefill(input_ids, cache)

    agent._ids = _ids
    agent._prefill = _prefill
    return m


ACTION_RE = re.compile(r"<(edit|read|grep|test|done|defn|findrefs)\b")


def kept_assistant_text(tok, ids: list[int], labels: list[int]) -> str:
    """Decode only the MODEL-generated (trained) runs of the kept segment."""
    chunks, run = [], []
    for i, lab in zip(ids, labels):
        if lab == -100:
            if run:
                chunks.append(tok.decode(run))
                run = []
        else:
            run.append(i)
    if run:
        chunks.append(tok.decode(run))
    return "\n".join(chunks)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------
def cmd_validate(args: argparse.Namespace) -> int:
    install_train_split()
    root = args.tmp_root or str(Path(tempfile.gettempdir()) / "streams_substitution_train")
    tasks = NT.build_tasks(Path(root) / TRAIN_SPLIT, TRAIN_SPLIT)
    rows = []
    for task in tasks:
        row = NT._validate_task(task)
        rows.append(row)
        print(f"{task['name']}: {'PASS' if row['passed'] else 'FAIL'} "
              f"template={task['template']} typed={row['lsp']['typed']['path']} "
              f"erased={row['lsp']['erased']['path']}", flush=True)
    pyrefly = find_pyrefly()
    version = subprocess.run([pyrefly, "--version"], capture_output=True, text=True).stdout.strip()
    payload = {
        "protocol": NT.PROTOCOL_VERSION,
        "experiment": "substitution-training",
        "split": TRAIN_SPLIT,
        "generator": "scripts/experiments/run_substitution_train.py",
        "seeds": list(TRAIN_SEEDS),
        "templates": list(TRAIN_TEMPLATES),
        # The reserved confirmation split is deliberately NOT enumerated here: its seeds must not
        # appear in any artifact of this experiment. It was never built (build_tasks is called for
        # the substrain split only) and install_train_split() asserts non-collision with it.
        "disjoint_from": {
            "pilot": list(NT.SPLIT_SEEDS["pilot"]),
            "apparatus_heldout_retest": list(NT.SPLIT_SEEDS["apparatus"]),
            "confirmation_reserved": "not built, not enumerated, asserted disjoint",
        },
        "protocol_source_sha256": NT._protocol_hashes(),
        "pyrefly": {"path": pyrefly, "version": version},
        "rows": rows,
        "passed": all(r["passed"] for r in rows),
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"-> {out}", flush=True)
    print(f"passed={payload['passed']} ({sum(r['passed'] for r in rows)}/{len(rows)})")
    return 0 if payload["passed"] else 1


# ---------------------------------------------------------------------------
# harvest
# ---------------------------------------------------------------------------
def cmd_harvest(args: argparse.Namespace) -> int:
    out_path = Path(args.out)
    if out_path.exists():
        print(f"refusing to overwrite existing result: {out_path}", file=sys.stderr)
        return 73
    install_train_split()

    root = args.tmp_root or str(Path(tempfile.gettempdir()) / "streams_substitution_train")
    tasks = NT.build_tasks(Path(root) / TRAIN_SPLIT, TRAIN_SPLIT)
    if args.names:
        wanted = set(args.names.split(","))
        tasks = [t for t in tasks if t["name"] in wanted]
    if not tasks:
        raise ValueError("no training tasks selected")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    device_map = {"": 0} if args.gpu_only else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16, device_map=device_map
    ).eval()
    model_meta = {
        "revision": getattr(model.config, "_commit_hash", None) or args.revision,
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
        "dtype": str(model.dtype),
    }
    pyrefly = find_pyrefly()
    pyrefly_version = subprocess.run(
        [pyrefly, "--version"], capture_output=True, text=True
    ).stdout.strip()

    rows = []
    n_seeds = 1 if args.temp == 0 else args.seeds

    def flush():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "protocol": NT.PROTOCOL_VERSION,
            "experiment": "substitution-training-harvest",
            "model": args.model, "model_meta": model_meta, "adapter": None,
            "config": {k: v for k, v in vars(args).items() if k != "fn"}, "split": TRAIN_SPLIT,
            "train_seeds": list(TRAIN_SEEDS), "train_templates": list(TRAIN_TEMPLATES),
            "oracle": {"redirect_text": REDIRECT_TEXT,
                       "relabel": "drop-prefix (last redirect), mirrors StreamAgent(relabel=True)"},
            "protocol_source_sha256": NT._protocol_hashes(),
            "pyrefly": {"path": pyrefly, "version": pyrefly_version},
            "rows": rows,
        }, indent=2) + "\n", encoding="utf-8")

    for task in tasks:
        for seed in range(args.seed_start, args.seed_start + n_seeds):
            env = NT.make_env(task, "typed")
            try:
                prompt = NT.build_prompt(task, "typed")
                supplied, supplied_path, lsp_latency = _method_from_lsp(task, "typed", env)
                if supplied_path != task["target_path"]:
                    raise RuntimeError(
                        f"typed automatic result did not resolve the gold override: {supplied_path}")
                if env.lsp_errors:
                    raise RuntimeError(f"automatic semantic query failed: {env.lsp_errors}")
                if task["gold"]["new_text"] in supplied:
                    raise RuntimeError("semantic context contains the gold replacement")
                prompt += (
                    "\n\nThe following current source span was supplied from a language-server "
                    "definition result at the visible call site. It is source context, not a "
                    "proposed correction.\n<semantic_result kind=\"current_source\">\n"
                    + supplied + "\n</semantic_result>"
                )
                oracle = install_read_oracle(env, task["target_path"])
                agent = StreamAgent(
                    model, tokenizer, env, edit_mode="line", sys_override=AUTO_SYS,
                    max_new_tokens=args.max_new, max_turns=args.max_turns,
                    max_reads=args.max_reads, temperature=args.temp, seed=seed,
                    use_lsp_defn=False, lsp_disabled=True, lsp_fallback=False,
                )
                mirror = install_trace_mirror(agent)
                started = time.perf_counter()
                result = agent.run(prompt, "pkg/app.py", editable=task["editable"])
                elapsed = time.perf_counter() - started
                held_out_pass = NT.run_heldout(task, "typed")

                # --- verify the mirror reproduces the agent's own trace, then relabel ---
                mirror_ok = tokenizer.decode(mirror["ids"]) == result["stream"]
                keep = mirror["keep"]
                if keep is None:
                    kept_ids, kept_labels = [], []
                else:
                    kept_ids = mirror["ids"][keep:]
                    kept_labels = mirror["labels"][keep:]
                sft_input_ids = (mirror["prompt_ids"] or []) + kept_ids
                sft_labels = [-100] * len(mirror["prompt_ids"] or []) + kept_labels
                n_train_tokens = sum(1 for x in kept_labels if x != -100)
                kept_text = kept_assistant_text(tokenizer, kept_ids, kept_labels)
                kept_actions = ACTION_RE.findall(kept_text)

                # a CLEAN substitution demo: after the drop point the model goes straight to a
                # successful edit of the gold file, never reopening it, and the task passes.
                reasons = []
                if not held_out_pass:
                    reasons.append("held_out_fail")
                if keep is None:
                    reasons.append("no_redirect_fired")
                if not mirror_ok:
                    reasons.append("trace_mirror_mismatch")
                if "edit" not in kept_actions:
                    reasons.append("no_edit_in_kept_segment")
                if kept_actions and kept_actions[0] != "edit":
                    reasons.append(f"kept_segment_starts_with_{kept_actions[0]}")
                if "read" in kept_actions or "grep" in kept_actions:
                    reasons.append("retrieval_in_kept_segment")
                if n_train_tokens < 5:
                    reasons.append("too_few_train_tokens")
                sft_keep = not reasons

                reread = _reread_metrics(result["events"], task["target_path"], True)
                row = {
                    "task": task["name"], "family": task["seed"], "split": TRAIN_SPLIT,
                    "template": task["template"], "variant": "typed", "arm": "harvest_substitution",
                    "seed": seed, "temp": args.temp,
                    "resolved": held_out_pass, "visible_pass": bool(result["resolved"]),
                    "held_out_pass": held_out_pass, "bailed": result.get("bailed"),
                    "in_tokens": result["in_tokens"], "out_tokens": result["out_tokens"],
                    "turns": result["turns"], "n_reads": result["n_reads"],
                    "n_lsp": result["n_lsp"], "n_tests": result["n_tests"],
                    "n_edits": result["n_edits"], "wall_sec": round(elapsed, 3),
                    "n_reads_denied": oracle["n_denied"],
                    "denied_before_first_edit": oracle["denied_before_first_edit"],
                    "semantic_supplied_path": supplied_path,
                    "auto_span_lsp_latency_ms": round(lsp_latency * 1000, 1),
                    "semantic_payload_sha256": hashlib.sha256(supplied.encode()).hexdigest(),
                    "server_errors": list(env.lsp_errors),
                    "trace_mirror_ok": mirror_ok,
                    "relabel_keep_index": keep,
                    "n_out_ids": len(mirror["ids"]),
                    "n_train_tokens": n_train_tokens,
                    "kept_actions": kept_actions,
                    "kept_assistant_text": kept_text[:2000],
                    "sft_keep": sft_keep, "sft_reject_reasons": reasons,
                    **reread,
                    **_metrics(task, result["events"], supplied_path),
                    "events": result["events"], "stream_tail": result["stream"][-2500:],
                    "sft_input_ids": sft_input_ids, "sft_labels": sft_labels,
                }
                rows.append(row)
                print(f"{task['name']} s{seed}: pass={held_out_pass} denied={oracle['n_denied']} "
                      f"keep={keep} train_toks={n_train_tokens} actions={kept_actions} "
                      f"KEEP={sft_keep} {'' if sft_keep else reasons}", flush=True)
            finally:
                env.close()
            flush()

    n = len(rows)
    kept = sum(1 for r in rows if r["sft_keep"])
    fired = sum(1 for r in rows if r["relabel_keep_index"] is not None)
    print("\n=== harvest summary ===")
    print(f"  rollouts               {n}")
    print(f"  held-out pass          {sum(1 for r in rows if r['held_out_pass'])}/{n}")
    print(f"  redirect fired         {fired}/{n}  (= model tried to reread the span's file)")
    print(f"  clean substitution SFT {kept}/{n}")
    print(f"  trace mirror ok        {sum(1 for r in rows if r['trace_mirror_ok'])}/{n}")
    print(f"-> {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="build + mechanically validate the fresh training split")
    v.add_argument("--out", default=None)
    v.add_argument("--tmp-root", default=None)
    v.set_defaults(fn=cmd_validate)

    h = sub.add_parser("harvest", help="DAgger relabel harvest with the substitution oracle")
    h.add_argument("out")
    h.add_argument("--model", default="Qwen/Qwen3.6-27B")
    h.add_argument("--revision", default=None)
    h.add_argument("--names", default=None)
    h.add_argument("--temp", type=float, default=0.7)
    h.add_argument("--seeds", type=int, default=3)
    h.add_argument("--seed-start", type=int, default=0)
    h.add_argument("--max-new", type=int, default=1000)
    h.add_argument("--max-turns", type=int, default=12)
    h.add_argument("--max-reads", type=int, default=12)
    h.add_argument("--gpu-only", action="store_true")
    h.add_argument("--tmp-root", default=None)
    h.set_defaults(fn=cmd_harvest)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
