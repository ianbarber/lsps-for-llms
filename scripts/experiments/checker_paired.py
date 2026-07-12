#!/usr/bin/env python3
"""Freeze natural drafts, then fork identical checker-revision trajectories."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scaffold.mock_env import MultiFileEnv  # noqa: E402
from scaffold.stream_agent import StreamAgent  # noqa: E402
from scaffold.tooling import find_pyrefly  # noqa: E402
from scripts.experiments.diagnostics import (  # noqa: E402
    DeltaDiagnosticEnv,
    collect_diagnostics,
    delta,
    fingerprint,
    format_diagnostics,
    is_coherent,
)
from scripts.synth_tasks_authoring import TASKS_AUTHORING  # noqa: E402


PROTOCOL_VERSION = "checker-paired-v2"
ARMS = ("control", "diagnostics", "gate", "noisy")
REVISION_SYS = """You are revising an existing coherent Python draft.
Tools:
  <read path="F" lines="A-B"/>     inspect a numbered source range
  <edit path="F" lines="A-B">...code...</edit>  replace a numbered range
  <test/>                            run the visible behavioral test
  <done/>                            submit the revised draft
Review the current draft, make only justified edits, and use the test within the revision budget.
"""
DRAFT_SYS = """You are producing one natural first draft of a typed Python module.
Tools:
  <read path="F" lines="A-B"/>     inspect a typed API before drafting
  <edit path="F" lines="A-B">...code...</edit>  write the implementation
  <submit_draft/>                    freeze the first complete draft
Do not run tests and do not wait for checker feedback. Inspect APIs as needed, write one complete
implementation, then emit <submit_draft/> exactly once.
"""


def _pyrefly_meta() -> dict:
    import subprocess
    path = find_pyrefly()
    version = subprocess.run([path, "--version"], capture_output=True, text=True).stdout.strip()
    return {"path": path, "version": version}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _workspace_hashes(files: dict[str, str]) -> tuple[dict[str, str], str]:
    hashes = {path: _sha(source) for path, source in sorted(files.items())}
    combined = _sha(json.dumps(hashes, sort_keys=True, separators=(",", ":")))
    return hashes, combined


def _protocol_hashes() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        ROOT / "scripts" / "experiments" / "diagnostics.py",
        ROOT / "scripts" / "analysis" / "analyze_checker_paired.py",
        ROOT / "scripts" / "synth_tasks_authoring.py",
        ROOT / "scaffold" / "stream_agent.py",
        ROOT / "scaffold" / "mock_env.py",
        ROOT / "scaffold" / "tooling.py",
        ROOT / "scripts" / "run_checker_paired.sh",
    ]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _task_map() -> dict[str, dict]:
    return {task["name"]: task for task in TASKS_AUTHORING}


def _replace_lines(source: str, start: int, end: int, replacement: str) -> str:
    lines = source.splitlines(keepends=True)
    if not (1 <= start <= end <= len(lines)):
        raise ValueError(f"invalid replay range {start}-{end} for {len(lines)} lines")
    text = replacement if replacement.endswith("\n") else replacement + "\n"
    return "".join(lines[:start - 1]) + text + "".join(lines[end:])


def _baseline(task: dict) -> list[dict]:
    env = MultiFileEnv(task["files"], task["target"], task["test"], skip_pyrefly=False,
                       held_out_src=task["held_out"])
    try:
        rows, _ = collect_diagnostics(env.ws, {task["target"]})
        return rows
    finally:
        env.close()


def _score(files: dict, task: dict) -> tuple[bool, bool]:
    env = MultiFileEnv(files, task["target"], task["test"], skip_pyrefly=True,
                       held_out_src=task["held_out"])
    try:
        return bool(env.run_tests()["resolved"]), bool(env.score()["resolved"])
    finally:
        env.close()


def import_legacy(source_run: Path, out: Path) -> int:
    """Recover exact drafts only when no logged edit body hit the historical 300-char cap."""
    payload = json.loads(source_run.read_text())
    source_hash = hashlib.sha256(source_run.read_bytes()).hexdigest()
    tasks = _task_map()
    drafts = []
    rejected = []
    rows = payload["rows"]["A"] if isinstance(payload["rows"], dict) else payload["rows"]
    for row in rows:
        task = tasks[row["task"]]
        applied = [event for event in row["events"]
                   if event.get("type") == "line_edit" and event.get("ok")]
        if any(len(event.get("replace", "")) >= 300 for event in applied):
            rejected.append({"task": task["name"], "reason": "historical edit body reached log cap"})
            continue
        source = task["files"][task["target"]]
        try:
            for event in applied:
                start, end = map(int, event["lines"].split("-"))
                source = _replace_lines(source, start, end, event["replace"])
        except (ValueError, KeyError) as exc:
            rejected.append({"task": task["name"], "reason": f"replay failed: {exc}"})
            continue
        files = {**task["files"], task["target"]: source}
        baseline = _baseline(task)
        env = MultiFileEnv(files, task["target"], task["test"], skip_pyrefly=False,
                           held_out_src=task["held_out"])
        try:
            current, latency = collect_diagnostics(env.ws, {task["target"]})
        finally:
            env.close()
        draft_delta = delta(current, baseline)
        visible, held = _score(files, task)
        file_hashes, workspace_hash = _workspace_hashes(files)
        drafts.append({
            "draft_id": f"legacy-{Path(source_run).stem}-{task['name']}-s{row['seed']}",
            "task": task["name"], "group": task["group"], "seed": row["seed"],
            "model": payload.get("model"), "source_run": str(source_run),
            "source_run_sha256": source_hash, "source_row_arm": row.get("arm", "none"),
            "files": files, "target": task["target"], "test": task["test"],
            "held_out": task["held_out"], "initial_target_sha256": _sha(task["files"][task["target"]]),
            "draft_target_sha256": _sha(source), "coherent": is_coherent(source),
            "file_sha256": file_hashes, "workspace_sha256": workspace_hash,
            "visible_pass": visible, "held_pass": held,
            "baseline_diagnostics": baseline, "draft_diagnostics": draft_delta,
            "checker_latency_ms": round(latency * 1000, 1),
            "historical_residual_count": row.get("n_resid_diag"),
            "historical_events": row["events"],
        })
    result = {
        "protocol": PROTOCOL_VERSION, "kind": "natural_drafts_recovered_from_committed_run",
        "source_run": str(source_run), "source_run_sha256": source_hash,
        "pyrefly": _pyrefly_meta(),
        "protocol_source_sha256": _protocol_hashes(),
        "drafts": drafts, "rejected": rejected,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"recovered {len(drafts)} exact drafts; rejected {len(rejected)} capped trajectories")
    for draft in drafts:
        print(f"  {draft['task']}: coherent={draft['coherent']} "
              f"new_semantic={sum(d['classification'] == 'semantic' for d in draft['draft_diagnostics'])}")
    return 0


def _authoring_prompt(task: dict) -> str:
    source = task["files"][task["target"]]
    numbered = "\n".join(f"{i:>3}| {line}" for i, line in enumerate(source.splitlines(), 1))
    others = ", ".join(path for path in sorted(task["files"]) if path != task["target"])
    return (
        f"Implement every unimplemented body in `{task['target']}` according to its docstrings.\n\n"
        f"{numbered}\n\nTyped APIs are available in: {others}. Read them as needed.\n\n"
        f"Behavioral specification example:\n```python\n{task['test']}\n```\n"
        "This example is partial. Write the full documented behavior, then submit the first draft "
        "without running it."
    )


def generate_drafts(args) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16,
        device_map={"": 0} if args.gpu_only else "auto"
    ).eval()
    model_meta = {
        "revision": getattr(model.config, "_commit_hash", None),
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__, "dtype": str(model.dtype),
    }
    wanted = set(args.names.split(",")) if args.names else None
    tasks = [task for task in TASKS_AUTHORING if wanted is None or task["name"] in wanted]
    drafts = []
    for task in tasks:
        baseline = _baseline(task)
        env = MultiFileEnv(task["files"], task["target"], task["test"], skip_pyrefly=False,
                           held_out_src=task["held_out"])
        try:
            agent = StreamAgent(
                model, tokenizer, env, edit_mode="line", sys_override=DRAFT_SYS,
                max_new_tokens=args.max_new, max_turns=args.max_turns,
                max_reads=args.max_reads, temperature=args.temperature, seed=args.seed,
                max_tests=0, lsp_disabled=True, draft_submission=True,
            )
            result = agent.run(_authoring_prompt(task), task["target"], editable=[task["target"]])
            source = env.read_file(task["target"])
            current, latency = collect_diagnostics(env.ws, {task["target"]})
            draft_delta = delta(current, baseline)
            # Score only after the explicit first-draft boundary; neither signal is available
            # to the model during generation.
            visible = bool(env.run_tests()["resolved"])
            held = bool(env.score()["resolved"])
        finally:
            env.close()
        files = {**task["files"], task["target"]: source}
        file_hashes, workspace_hash = _workspace_hashes(files)
        drafts.append({
            "draft_id": f"natural-{task['name']}-s{args.seed}", "task": task["name"],
            "group": task["group"], "seed": args.seed, "model": args.model,
            "files": files, "target": task["target"], "test": task["test"],
            "held_out": task["held_out"], "draft_target_sha256": _sha(source),
            "initial_target_sha256": _sha(task["files"][task["target"]]),
            "file_sha256": file_hashes, "workspace_sha256": workspace_hash,
            "draft_submitted": bool(result.get("draft_submitted")),
            "coherent": bool(result.get("draft_submitted") and is_coherent(source)),
            "visible_pass": visible, "held_pass": held,
            "baseline_diagnostics": baseline, "draft_diagnostics": draft_delta,
            "checker_latency_ms": round(latency * 1000, 1), "events": result["events"],
            "draft_stream": result["stream"],
            "in_tokens": result["in_tokens"], "out_tokens": result["out_tokens"],
        })
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "protocol": PROTOCOL_VERSION, "kind": "natural_drafts", "model": args.model,
            "model_meta": model_meta, "config": vars(args),
            "pyrefly": _pyrefly_meta(), "protocol_source_sha256": _protocol_hashes(),
            "drafts": drafts,
        }, indent=2) + "\n")
        print(f"{task['name']}: coherent={drafts[-1]['coherent']} "
              f"new={len(draft_delta)} held={held}", flush=True)
    return 0


def recheck_drafts(source: Path, out: Path) -> int:
    """Recompute diagnostics, behavior, coherence, and hashes without another model call."""
    payload = json.loads(source.read_text())
    tasks = _task_map()
    for draft in payload["drafts"]:
        task = tasks[draft["task"]]
        baseline = _baseline(task)
        env = MultiFileEnv(draft["files"], draft["target"], draft["test"], skip_pyrefly=False,
                           held_out_src=draft["held_out"])
        try:
            current, latency = collect_diagnostics(env.ws, {draft["target"]})
        finally:
            env.close()
        visible, held = _score(draft["files"], task)
        hashes, workspace_hash = _workspace_hashes(draft["files"])
        draft.update({
            "baseline_diagnostics": baseline,
            "draft_diagnostics": delta(current, baseline),
            "checker_latency_ms": round(latency * 1000, 1),
            "visible_pass": visible, "held_pass": held,
            "file_sha256": hashes, "workspace_sha256": workspace_hash,
            "coherent": bool(draft.get("draft_submitted", payload.get("kind", "").startswith(
                "natural_drafts_recovered")) and is_coherent(draft["files"][draft["target"]])),
        })
    payload["pyrefly"] = _pyrefly_meta()
    payload["protocol_source_sha256"] = _protocol_hashes()
    payload["rechecked_by"] = "checker-paired-v1 structured diagnostic collector"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"rechecked {len(payload['drafts'])} drafts -> {out}")
    return 0


def calibrate_drafts(path: Path, minimum: float, maximum: float, min_coherent: int) -> int:
    payload = json.loads(path.read_text())
    coherent = [draft for draft in payload["drafts"]
                if draft.get("draft_submitted") and draft.get("coherent")]
    opportunities = [draft for draft in coherent if any(
        item["classification"] == "semantic" for item in draft["draft_diagnostics"]
    )]
    viable = [draft for draft in coherent if any(
        event.get("type") == "line_edit" and event.get("ok") for event in draft.get("events", [])
    )]
    rate = len(opportunities) / len(coherent) if coherent else 0.0
    print(f"coherent submitted={len(coherent)} viable edits={len(viable)} "
          f"semantic opportunities={len(opportunities)} rate={rate:.3f}")
    if len(coherent) < min_coherent or len(viable) != len(coherent) or not (minimum <= rate <= maximum):
        print("calibration gate failed; do not run paired revisions", file=sys.stderr)
        return 2
    print("calibration gate passed")
    return 0


def select_legacy_case_series(source: Path, out: Path, names: str) -> int:
    """Freeze exact recovered checker-positive drafts without changing the source artifact."""
    if out.exists():
        raise FileExistsError(f"refusing to overwrite case-series selection: {out}")
    payload = json.loads(source.read_text())
    if payload.get("kind") != "natural_drafts_recovered_from_committed_run":
        raise ValueError("case-series selection requires the recovered legacy artifact")
    wanted = set(names.split(","))
    tasks = _task_map()
    selected = []
    for draft in payload["drafts"]:
        if draft["task"] not in wanted:
            continue
        hashes, workspace_hash = _workspace_hashes(draft["files"])
        semantic = [item for item in draft["draft_diagnostics"]
                    if item["classification"] == "semantic"]
        if (not draft["coherent"] or not semantic or hashes != draft["file_sha256"]
                or workspace_hash != draft["workspace_sha256"]):
            raise ValueError(f"legacy draft is not an exact coherent opportunity: {draft['task']}")
        task = tasks[draft["task"]]
        gold_files = {**draft["files"], draft["target"]: task["gold_target"]}
        gold_visible, gold_held = _score(gold_files, task)
        env = MultiFileEnv(
            gold_files, draft["target"], draft["test"], skip_pyrefly=False,
            held_out_src=draft["held_out"],
        )
        try:
            gold_current, _ = collect_diagnostics(env.ws, {draft["target"]})
        finally:
            env.close()
        gold_delta = delta(gold_current, draft["baseline_diagnostics"])
        if not (gold_visible and gold_held and not gold_delta):
            raise ValueError(f"gold repair does not clear behavior and diagnostics: {draft['task']}")
        selected.append({
            **draft,
            "revision_case_series_eligible": True,
            "selection_basis": (
                "exact recovered coherent workspace with at least one target-delta semantic diagnostic"
            ),
            "natural_submission_marker_available": False,
            "gold_repair_validation": {
                "visible_pass": gold_visible, "held_pass": gold_held,
                "diagnostic_delta": gold_delta,
                "gold_target_sha256": _sha(task["gold_target"]),
            },
        })
    if {draft["task"] for draft in selected} != wanted:
        raise ValueError("not every requested legacy task was selected")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "protocol": PROTOCOL_VERSION,
        "kind": "opportunity_conditioned_legacy_revision_case_series",
        "source_artifact": str(source),
        "source_artifact_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "selection_is_not_a_natural_opportunity_sample": True,
        "draft_generation_token_cost_available": False,
        "pyrefly": payload.get("pyrefly"),
        "protocol_source_sha256": _protocol_hashes(),
        "drafts": selected,
    }, indent=2) + "\n")
    print(f"selected {len(selected)} checker-positive legacy drafts -> {out}")
    return 0


def _revision_prompt(draft: dict, arm: str) -> str:
    source = draft["files"][draft["target"]]
    numbered = "\n".join(f"{i:>3}| {line}" for i, line in enumerate(source.splitlines(), 1))
    prompt = (
        "Review the exact frozen draft below for behavioral or API mistakes. You have a small revision "
        "budget shared across treatments. Run the visible test, make justified edits, and submit.\n\n"
        f"`{draft['target']}`:\n{numbered}\n\nVisible test:\n```python\n{draft['test']}\n```"
    )
    if arm == "diagnostics":
        prompt += "\n\nOne coherent-patch checker pass reported this delta:\n<diagnostic_delta>\n"
        prompt += format_diagnostics(draft["draft_diagnostics"]) or "(no new diagnostics)"
        prompt += "\n</diagnostic_delta>"
    return prompt


def _diag_effect(initial: list[dict], final: list[dict]) -> dict:
    before = Counter(fingerprint(item) for item in initial)
    after = Counter(fingerprint(item) for item in final)
    return {
        "diagnostics_eliminated": sum((before - after).values()),
        "diagnostics_retained": sum((before & after).values()),
        "diagnostics_introduced": sum((after - before).values()),
    }


def _edited_diagnosed_location(events: list[dict], diagnostics: list[dict]) -> bool:
    for event in events:
        if event.get("type") != "line_edit" or not event.get("ok"):
            continue
        try:
            start, end = map(int, event["lines"].split("-"))
        except (KeyError, ValueError):
            continue
        if any(item["path"] == event.get("path") and start <= item["line"] <= end
               for item in diagnostics):
            return True
    return False


def _post_rejection_metrics(events: list[dict]) -> tuple[int, bool]:
    rejected_at = next(
        (index for index, event in enumerate(events) if event.get("type") == "gate_reject"), None
    )
    if rejected_at is None:
        return 0, False
    rejected_locations = {
        (item.get("path"), item.get("line"))
        for event in events if event.get("type") == "gate_reject"
        for item in event.get("diagnostics", [])
    }
    edits = [event for event in events[rejected_at + 1:]
             if event.get("type") == "line_edit" and event.get("ok")]
    overlaps = False
    for event in edits:
        try:
            start, end = map(int, event["lines"].split("-"))
        except (KeyError, ValueError):
            continue
        overlaps |= any(
            path == event.get("path") and line is not None and start <= line <= end
            for path, line in rejected_locations
        )
    return len(edits), overlaps


def _validate_case_series_rows(drafts: list[dict], rows: list[dict], arms: list[str], seeds: int) -> None:
    expected = len(drafts) * len(arms) * seeds
    if len(rows) != expected:
        raise ValueError(f"incomplete case-series grid: expected {expected}, found {len(rows)}")
    if any(row.get("serialization_failures") for row in rows):
        raise ValueError("ambiguous inline edit serialization occurred")
    for row in rows:
        if row["arm"] == "gate":
            if row["gate_invocations"] != row["gate_acceptances"] + row["gate_rejections"]:
                raise ValueError("gate event counts are inconsistent")
            if row["n_checks"] != row["gate_invocations"]:
                raise ValueError("gate checker count differs from gate invocations")
            if row["accepted_dirty"]:
                raise ValueError("gate accepted a dirty workspace")
    by_cell = {(row["draft_id"], row["seed"], row["arm"]): row for row in rows}
    for draft in drafts:
        for seed in range(min(row["seed"] for row in rows),
                          min(row["seed"] for row in rows) + seeds):
            control = by_cell[(draft["draft_id"], seed, "control")]
            gate = by_cell[(draft["draft_id"], seed, "gate")]
            if control["first_done_prefix_sha256"] != gate["first_done_prefix_sha256"]:
                raise ValueError("control/gate trajectories diverge before the first completion attempt")


def revise(args) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    payload = json.loads(Path(args.drafts).read_text())
    if Path(args.out).exists():
        raise FileExistsError(f"refusing to overwrite paired revisions: {args.out}")
    drafts = [draft for draft in payload["drafts"]
              if (draft.get("draft_submitted") or draft.get("revision_case_series_eligible"))
              and draft.get("coherent")]
    if args.names:
        wanted = set(args.names.split(","))
        drafts = [draft for draft in drafts if draft["task"] in wanted]
    if args.checker_positive_only:
        drafts = [draft for draft in drafts if any(
            item["classification"] == "semantic" for item in draft["draft_diagnostics"]
        )]
    if not drafts:
        raise ValueError("no coherent submitted drafts selected for paired revision")
    arms = args.arms.split(",")
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise ValueError(f"unknown arms: {sorted(unknown)}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16,
        device_map={"": 0} if args.gpu_only else "auto"
    ).eval()
    model_meta = {
        "revision": getattr(model.config, "_commit_hash", None),
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__, "dtype": str(model.dtype),
    }
    rows = []
    for draft, seed, arm in itertools.product(
        drafts, range(args.seed, args.seed + args.seeds), arms
    ):
        env = DeltaDiagnosticEnv(
            draft["files"], draft["target"], draft["test"],
            baseline_diagnostics=draft["baseline_diagnostics"],
            diagnostic_scope={draft["target"]}, held_out_src=draft["held_out"],
            skip_pyrefly=False,
        )
        try:
            agent = StreamAgent(
                model, tokenizer, env, edit_mode="line", sys_override=REVISION_SYS,
                max_new_tokens=args.max_new, max_turns=args.max_turns,
                max_reads=args.max_reads, temperature=args.temperature, seed=seed,
                auto_check=(arm == "noisy"), acceptance_gate=(arm == "gate"),
                lsp_disabled=True,
            )
            started = time.perf_counter()
            result = agent.run(_revision_prompt(draft, arm), draft["target"],
                               editable=[draft["target"]])
            wall = time.perf_counter() - started
            held = bool(env.score()["resolved"])
            final_diags = env.raw_diagnostic_delta()
            one_shot_latency = draft["checker_latency_ms"] if arm == "diagnostics" else 0.0
            events = result["events"]
            done_attempts = sum(event.get("type") == "done_attempt" for event in events)
            gate_invocations = sum(event.get("type") == "gate_check" for event in events)
            gate_acceptances = sum(event.get("type") == "gate_accept" for event in events)
            gate_rejections = sum(event.get("type") == "gate_reject" for event in events)
            accepted = bool(gate_acceptances) if arm == "gate" else bool(result.get("done_seen"))
            post_rejection_edits, edited_rejected_location = _post_rejection_metrics(events)
            final_files = {path: env.read_file(path) for path in env.list_files()}
            final_file_hashes, final_workspace_hash = _workspace_hashes(final_files)
            serialization_failures = [
                event for event in events
                if event.get("type") == "line_edit"
                and event.get("serialization_mode") == "ambiguous_inline_indentation"
            ]
            row = {
                "draft_id": draft["draft_id"], "draft_hash": draft["draft_target_sha256"],
                "task": draft["task"], "group": draft["group"], "arm": arm, "seed": seed,
                "opportunity": any(item["classification"] == "semantic"
                                   for item in draft["draft_diagnostics"]),
                "initial_diagnostics": draft["draft_diagnostics"],
                "final_diagnostics": final_diags,
                **_diag_effect(draft["draft_diagnostics"], final_diags),
                "edited_diagnosed_location": _edited_diagnosed_location(
                    result["events"], draft["draft_diagnostics"]),
                "visible_pass": bool(result["resolved"]), "held_pass": held,
                "type_clean": not final_diags, "accepted": accepted,
                "semantic_clean": not any(item["classification"] == "semantic"
                                           for item in final_diags),
                "syntax_clean": not any(item["classification"] == "syntax_or_partial"
                                         for item in final_diags),
                "accepted_type_clean": bool(accepted and not final_diags),
                "accepted_type_clean_correct": bool(
                    accepted and not final_diags and held
                ),
                "unsubmitted": not bool(done_attempts),
                "done_attempts": done_attempts,
                "gate_invocations": gate_invocations,
                "gate_acceptances": gate_acceptances,
                "gate_rejections": gate_rejections,
                "gate_never_invoked": bool(arm == "gate" and not gate_invocations),
                "gate_rejected_then_accepted": bool(gate_rejections and gate_acceptances),
                "gate_rejected_unresolved": bool(gate_rejections and not gate_acceptances),
                "post_rejection_edits": post_rejection_edits,
                "edited_rejected_location": edited_rejected_location,
                "accepted_dirty": bool(accepted and final_diags),
                "first_done_prefix_sha256": next(
                    (event.get("prefix_sha256") for event in events
                     if event.get("type") == "done_attempt"), None
                ),
                "serialization_failures": serialization_failures,
                "final_target_sha256": _sha(final_files[draft["target"]]),
                "final_file_sha256": final_file_hashes,
                "final_workspace_sha256": final_workspace_hash,
                "draft_in_tokens": draft.get("in_tokens"),
                "draft_out_tokens": draft.get("out_tokens"),
                "in_tokens": result["in_tokens"], "out_tokens": result["out_tokens"],
                "turns": result["turns"], "n_edits": result["n_edits"],
                "n_tests": result["n_tests"], "n_checks": result["n_checks"],
                "checker_latency_ms": round(
                    one_shot_latency + env.last_checker_latency * 1000, 1
                ),
                "wall_sec": round(wall, 3), "termination_reason": result["termination_reason"],
                "events": events,
                "stream_tail": result["stream"][-2500:],
            }
            rows.append(row)
        finally:
            env.close()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "protocol": PROTOCOL_VERSION, "kind": (
                "opportunity_conditioned_legacy_paired_revision_case_series"
                if payload.get("selection_is_not_a_natural_opportunity_sample")
                else "paired_revisions"
            ),
            "model": args.model, "source_drafts": args.drafts,
            "model_meta": model_meta, "config": vars(args),
            "pyrefly": _pyrefly_meta(), "protocol_source_sha256": _protocol_hashes(),
            "rows": rows,
        }, indent=2) + "\n")
        print(f"{draft['task']} {arm}: held={row['held_pass']} clean={row['type_clean']} "
              f"accepted={row['accepted']} edits={row['n_edits']}", flush=True)
    if payload.get("selection_is_not_a_natural_opportunity_sample"):
        _validate_case_series_rows(drafts, rows, arms, args.seeds)
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    legacy = commands.add_parser("import-legacy")
    legacy.add_argument("source_run", type=Path)
    legacy.add_argument("out", type=Path)

    recheck = commands.add_parser("recheck")
    recheck.add_argument("source", type=Path)
    recheck.add_argument("out", type=Path)

    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("drafts", type=Path)
    calibrate.add_argument("--minimum", type=float, default=0.2)
    calibrate.add_argument("--maximum", type=float, default=0.7)
    calibrate.add_argument("--min-coherent", type=int, default=2)

    select = commands.add_parser("select-legacy-case-series")
    select.add_argument("source", type=Path)
    select.add_argument("out", type=Path)
    select.add_argument("--names", required=True)

    generate = commands.add_parser("generate")
    generate.add_argument("out")
    generate.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    generate.add_argument("--revision", default=None)
    generate.add_argument("--names", default=None)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--temperature", type=float, default=0.7)
    generate.add_argument("--max-new", type=int, default=2200)
    generate.add_argument("--max-turns", type=int, default=12)
    generate.add_argument("--max-reads", type=int, default=6)
    generate.add_argument("--gpu-only", action="store_true")

    revision = commands.add_parser("revise")
    revision.add_argument("drafts")
    revision.add_argument("out")
    revision.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    revision.add_argument("--revision", default=None)
    revision.add_argument("--names", default=None)
    revision.add_argument("--arms", default=",".join(ARMS))
    revision.add_argument("--checker-positive-only", action="store_true")
    revision.add_argument("--seed", type=int, default=0)
    revision.add_argument("--seeds", type=int, default=1)
    revision.add_argument("--temperature", type=float, default=0.0)
    revision.add_argument("--max-new", type=int, default=1200)
    revision.add_argument("--max-turns", type=int, default=6)
    revision.add_argument("--max-reads", type=int, default=4)
    revision.add_argument("--gpu-only", action="store_true")
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "import-legacy":
        return import_legacy(args.source_run, args.out)
    if args.command == "recheck":
        return recheck_drafts(args.source, args.out)
    if args.command == "calibrate":
        return calibrate_drafts(args.drafts, args.minimum, args.maximum, args.min_coherent)
    if args.command == "select-legacy-case-series":
        return select_legacy_case_series(args.source, args.out, args.names)
    if args.command == "generate":
        return generate_drafts(args)
    return revise(args)


if __name__ == "__main__":
    raise SystemExit(main())
