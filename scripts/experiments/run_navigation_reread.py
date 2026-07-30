#!/usr/bin/env python3
"""Reread-after-span (substitution) probe over navigation-v2 instances.

Context: in the navigation-v2 pilot (claim C24) every automatically delivered definition
span was followed by a read of the target file -- the span did not substitute for the
read -- while in the retrieval suite (C27) an ELECTED definition was never re-read.
This driver tests, at larger n, whether the reread persists under an explicit
sufficiency instruction (=> substitution needs training) or is promptable-away
(=> prompting suffices).

This is a SEPARATE driver so every frozen navigation protocol source
(navigation_tasks.py, run_navigation.py, scaffold/*) stays byte-identical; they are
imported read-only. The reserved confirmation split is refused by construction.

Arms (typed variant only, one deterministic rollout per task at temperature 0):
  auto_neutral    : automatic span delivery, neutral framing -- replicates the pilot's
                    typed/semantic_auto cell.
  auto_sufficient : automatic span delivery + explicit sufficiency instruction ("the
                    span is the complete definition; do not open the defining file
                    unless it is insufficient").
  framed_elective : elective <defn> under the existing strong framing (the model must
                    call the tool itself); measures reread after an elected result.

Key outcome per row: whether the target/defining file is read AFTER the span is
received (count, whole/ranged split), plus success and tokens.
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

from scaffold.stream_agent import StreamAgent  # noqa: E402
from scaffold.tooling import find_pyrefly  # noqa: E402
from scripts.experiments.navigation_tasks import (  # noqa: E402
    PROTOCOL_VERSION,
    _protocol_hashes,
    build_prompt,
    build_tasks,
    make_env,
    run_heldout,
)
from scripts.experiments.run_navigation import (  # noqa: E402
    AUTO_SYS,
    FRAMED_SYS,
    TOOLS,
    _method_from_lsp,
    _method_from_metadata,
    _metrics,
)


SUFFICIENT_SYS = TOOLS + (
    "A language-server result from the visible call site is supplied in the user message. It is the "
    "current source definition, not a proposed fix. The supplied span is the COMPLETE current source of "
    "the method that binds at the call site, with its file path and real line numbers. It is sufficient "
    "context for the edit: do NOT open or read the defining file unless the span is insufficient. Edit "
    "directly at the span's stated path and line numbers, then run the test.\n"
)

ARM_SPECS = {
    "auto_neutral": {"auto": True, "system": AUTO_SYS},
    "auto_sufficient": {"auto": True, "system": SUFFICIENT_SYS},
    "framed_elective": {"auto": False, "system": FRAMED_SYS},
    # ---- push-vs-elect within-instance contrast (C31 follow-up) ----
    # Both arms deliver the IDENTICAL, provably-usable pristine target-method span (from frozen
    # task metadata, verified per instance to equal the live-LSP composed span). They differ ONLY
    # in delivery mode: push_span injects the span unsolicited (as C31 auto_neutral); elect_span
    # withholds it and returns the same bytes verbatim when the model calls <defn> (oracle-backed,
    # so election can never be vacuous the way C31's framed_elective was).
    "push_span": {"auto": True, "system": AUTO_SYS, "shared_span": True},
    "elect_span": {"auto": False, "system": FRAMED_SYS, "shared_span": True, "oracle": True},
}


def _shared_span_and_verify(task: dict) -> tuple[str, str, dict]:
    """The single span both push_span and elect_span deliver, plus a mechanical usability gate.

    Sourced deterministically from frozen task metadata (`target_method_span`, the same payload the
    frozen `semantic_span_control` uses), then cross-checked byte-for-byte against the live-LSP
    composed span so the delivered bytes are provably the genuine push payload. Feeding the SAME
    string to both arms guarantees identical span content within each instance."""
    meta_span, meta_path = _method_from_metadata(task)
    env = make_env(task, "typed")
    lsp_span = lsp_path = None
    lsp_errors: list = []
    try:
        lsp_span, lsp_path, _lat = _method_from_lsp(task, "typed", env)
        lsp_errors = list(env.lsp_errors)
    finally:
        env.close()
    checks = {
        "meta_path_is_target": meta_path == task["target_path"],
        "lsp_path_is_target": lsp_path == task["target_path"],
        "lsp_span_equals_metadata": lsp_span == meta_span,
        "no_lsp_errors": not lsp_errors,
        "gold_not_in_span": task["gold"]["new_text"] not in meta_span,
        "span_has_numbered_lines": meta_span.count("|") >= 2,
        "span_names_target_path": meta_path in meta_span,
    }
    return meta_span, meta_path, checks


def _reread_metrics(events: list[dict], target_path: str, auto: bool) -> dict:
    """Count reads of the span's file AFTER the span was received. For automatic arms the
    span is present from the first token, so every target read counts; for the elective
    arm only reads after the first FOUND <defn> result count (against the file that
    result named)."""
    target_reads = [e for e in events if e.get("type") == "read" and e.get("path") == target_path]
    defining_path = target_path
    if auto:
        after = target_reads
        first_defn_idx = None
        n_defn_found = None
    else:
        first_defn_idx = next(
            (i for i, e in enumerate(events) if e.get("type") == "defn" and e.get("found")),
            None,
        )
        n_defn_found = sum(1 for e in events if e.get("type") == "defn" and e.get("found"))
        if first_defn_idx is None:
            after = []
        else:
            defining_path = events[first_defn_idx].get("path") or target_path
            after = [e for e in events[first_defn_idx + 1:]
                     if e.get("type") == "read" and e.get("path") == defining_path]
    return {
        "reread_defining_path": defining_path,
        "n_target_reads_total": len(target_reads),
        "n_reads_after_span": len(after),
        "n_reads_after_span_whole": sum(1 for e in after if not e.get("ranged")),
        "n_reads_after_span_ranged": sum(1 for e in after if e.get("ranged")),
        "read_after_span": bool(after),
        "first_found_defn_event_idx": first_defn_idx,
        "n_defn_found": n_defn_found,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--revision", default=None)
    # substitution-training retest: load a LoRA adapter on top of the base model. Absent (the
    # default) the run is byte-identical to the untrained baseline arms.
    parser.add_argument("--adapter", default=None)
    # "confirmation" SPENDS the reserved 41xxx split, once, as the pre-registered confirmation of
    # the substitution-training result (C33). It was reserved for the navigation-v2 typed/erased
    # confirmation, which is permanently blocked by its own ceiling gate in
    # scripts/run_navigation_confirmation.sh (that script refuses to run when every core pilot row
    # passes held-out, and all 8 do), so the seeds could never be spent on their original purpose.
    # Validated by: runs/protocol/navigation_v2_confirmation_substitution_validation.json (12/12).
    parser.add_argument("--split", choices=("pilot", "apparatus", "confirmation"), default="apparatus")
    parser.add_argument("--arms", default="auto_neutral,auto_sufficient,framed_elective")
    parser.add_argument("--names", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new", type=int, default=1000)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-reads", type=int, default=12)
    parser.add_argument("--gpu-only", action="store_true")
    parser.add_argument("--tmp-root", default=None)
    args = parser.parse_args()
    out_path = Path(args.out)
    if out_path.exists():
        print(f"refusing to overwrite existing result: {out_path}", file=sys.stderr)
        return 73

    arms = args.arms.split(",")
    unknown = [a for a in arms if a not in ARM_SPECS]
    if unknown:
        raise ValueError(f"unknown arm(s): {unknown}")

    root = args.tmp_root or str(Path(tempfile.gettempdir()) / "streams_navigation_v2_reread")
    tasks = build_tasks(Path(root) / args.split, args.split)
    if args.names:
        wanted = set(args.names.split(","))
        tasks = [task for task in tasks if task["name"] in wanted]
    if not tasks:
        raise ValueError("no navigation tasks selected")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    device_map = {"": 0} if args.gpu_only else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16, device_map=device_map
    )
    adapter_meta = None
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        meta_path = Path(args.adapter) / "streams_train_meta.json"
        adapter_meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    model = model.eval()
    model_meta = {
        "adapter": args.adapter,
        "adapter_meta": adapter_meta,
        "revision": getattr(model.config, "_commit_hash", None) or args.revision,
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
        "dtype": str(model.dtype),
    }

    pyrefly = find_pyrefly()
    pyrefly_version = __import__("subprocess").run(
        [pyrefly, "--version"], capture_output=True, text=True
    ).stdout.strip()

    use_shared = any(ARM_SPECS[a].get("shared_span") for a in arms)

    rows = []
    for task in tasks:
        shared_span = shared_path = shared_checks = shared_sha = None
        if use_shared:
            shared_span, shared_path, shared_checks = _shared_span_and_verify(task)
            shared_sha = hashlib.sha256(shared_span.encode()).hexdigest()
            print(f"{task['name']} shared-span usability: "
                  f"{'OK' if all(shared_checks.values()) else 'FAIL ' + str(shared_checks)} "
                  f"sha={shared_sha[:12]}", flush=True)
        for arm in arms:
            spec = ARM_SPECS[arm]
            env = make_env(task, "typed")
            try:
                supplied = None
                supplied_path = None
                lsp_latency = 0.0
                defn_oracle = None
                prompt = build_prompt(task, "typed")
                if spec.get("shared_span"):
                    if not all(shared_checks.values()):
                        raise RuntimeError(
                            f"shared span failed usability gate for {task['name']}: {shared_checks}"
                        )
                    supplied, supplied_path = shared_span, shared_path
                    if spec["auto"]:
                        # push arm: span delivered unsolicited in the prompt (identical text to C31 auto)
                        prompt += (
                            "\n\nThe following current source span was supplied from a language-server "
                            "definition result at the visible call site. It is source context, not a "
                            "proposed correction.\n<semantic_result kind=\"current_source\">\n"
                            + supplied + "\n</semantic_result>"
                        )
                    if spec.get("oracle"):
                        # elect arm: span NOT in the prompt; returned verbatim when the model calls <defn>
                        defn_oracle = {"span": supplied, "path": supplied_path}
                elif spec["auto"]:
                    supplied, supplied_path, lsp_latency = _method_from_lsp(task, "typed", env)
                    if supplied_path != task["target_path"]:
                        raise RuntimeError(
                            f"typed automatic result did not resolve the gold override: {supplied_path}"
                        )
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
                tool_enabled = not spec["auto"]
                agent = StreamAgent(
                    model, tokenizer, env, edit_mode="line", sys_override=spec["system"],
                    max_new_tokens=args.max_new, max_turns=args.max_turns,
                    max_reads=args.max_reads, temperature=args.temperature, seed=args.seed,
                    use_lsp_defn=tool_enabled, lsp_disabled=not tool_enabled,
                    lsp_fallback=False, defn_oracle=defn_oracle,
                )
                started = time.perf_counter()
                result = agent.run(prompt, "pkg/app.py", editable=task["editable"])
                elapsed = time.perf_counter() - started
                held_out_pass = run_heldout(task, "typed")
                reread = _reread_metrics(result["events"], task["target_path"], spec["auto"])
                row = {
                    "task": task["name"], "family": task["seed"], "split": args.split,
                    "variant": "typed", "arm": arm, "seed": args.seed,
                    "resolved": held_out_pass, "visible_pass": bool(result["resolved"]),
                    "held_out_pass": held_out_pass, "bailed": result.get("bailed"),
                    "in_tokens": result["in_tokens"], "out_tokens": result["out_tokens"],
                    "turns": result["turns"], "n_reads": result["n_reads"],
                    "n_lsp": result["n_lsp"], "n_tests": result["n_tests"],
                    "n_edits": result["n_edits"], "wall_sec": round(elapsed, 3),
                    "server_latency_ms": round(sum(env.lsp_latencies) * 1000, 1),
                    "server_errors": list(env.lsp_errors),
                    "auto_span_lsp_latency_ms": round(lsp_latency * 1000, 1),
                    "semantic_supplied_path": supplied_path,
                    "semantic_payload_sha256": (
                        hashlib.sha256(supplied.encode()).hexdigest() if supplied else None
                    ),
                    **reread,
                    **_metrics(task, result["events"], supplied_path),
                    "events": result["events"], "stream_tail": result["stream"][-2500:],
                }
                if spec.get("shared_span"):
                    # push-vs-elect audit fields — added ONLY on the new shared-span arms so the
                    # legacy arms' row schema (and byte-identity) is untouched.
                    row["shared_span_sha256"] = shared_sha
                    row["span_usability_check"] = shared_checks
                    row["elected_defn"] = (
                        bool(reread["n_defn_found"]) if not spec["auto"] else None
                    )
                rows.append(row)
                print(f"{task['name']} typed/{arm} s{args.seed}: pass={row['resolved']} "
                      f"in={row['in_tokens']} reads_after_span={row['n_reads_after_span']} "
                      f"lsp={row['n_lsp']} edits={row['n_edits']}", flush=True)
            finally:
                env.close()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({
                "protocol": PROTOCOL_VERSION, "experiment": "reread-after-span",
                "model": args.model, "model_meta": model_meta, "config": vars(args),
                "arms": arms, "protocol_source_sha256": _protocol_hashes(),
                "pyrefly": {"path": pyrefly, "version": pyrefly_version}, "rows": rows,
            }, indent=2) + "\n", encoding="utf-8")

    print("\n=== per-arm summary ===", flush=True)
    for arm in arms:
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        n = len(sub)
        n_pass = sum(1 for r in sub if r["resolved"])
        n_reread = sum(1 for r in sub if r["read_after_span"])
        mean_in = round(sum(r["in_tokens"] for r in sub) / n, 1)
        mean_after = round(sum(r["n_reads_after_span"] for r in sub) / n, 2)
        elect = ""
        if not ARM_SPECS[arm]["auto"]:
            n_elect = sum(1 for r in sub if r.get("n_defn_found"))
            n_reread_elect = sum(1 for r in sub if r.get("n_defn_found") and r["read_after_span"])
            elect = f"  elected={n_elect}/{n}  reread|elected={n_reread_elect}/{max(n_elect,1)}"
        print(f"  {arm:16s} pass={n_pass}/{n}  read_after_span={n_reread}/{n} "
              f"(mean {mean_after}/task)  mean_in_toks={mean_in}{elect}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
