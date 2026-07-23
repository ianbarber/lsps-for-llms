#!/usr/bin/env python3
"""Freeze natural drafts, then fork identical checker-revision trajectories."""

from __future__ import annotations

import argparse
from collections import Counter
import difflib
import hashlib
import itertools
import json
import os
import sys
import tempfile
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


PROTOCOL_VERSION = "checker-paired-v6"
ARMS = ("control", "diagnostics", "gate", "noisy")
REVISION_SYS = """You are revising an existing coherent Python draft.
Tools:
  <read path="F" lines="A-B"/>     inspect a numbered source range
  <test/>                            run the visible behavioral test
  <done/>                            submit the revised draft
To edit, replace a numbered range using EXACTLY this multiline form:
<edit path="F" lines="A-B">
new code with exact indentation
</edit>
There must be a newline immediately after the opening `>`; never put replacement code on the
same line as the opening tag. Ranges are inclusive and use the latest numbered source view.

Review the current draft, make only justified edits, and use the test within the revision budget.
If an acceptance gate rejects a completion, repair the reported error, run the visible test, and
emit a fresh <done/> to resubmit the repaired workspace.
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
        ROOT / "scripts" / "experiments" / "checker_hidden.py",
        ROOT / "scripts" / "experiments" / "stub_policy.py",
        ROOT / "scripts" / "analysis" / "analyze_checker_paired.py",
        ROOT / "scripts" / "synth_tasks_authoring.py",
        ROOT / "scaffold" / "stream_agent.py",
        ROOT / "scaffold" / "mock_env.py",
        ROOT / "scaffold" / "tooling.py",
        ROOT / "scripts" / "run_checker_paired.sh",
        ROOT / "scripts" / "run_checker_case_series.sh",
        ROOT / "scripts" / "run_checker_hidden.sh",
        ROOT / "scripts" / "run_checker_gate_v2.sh",
        ROOT / "scripts" / "run_checker_gate_v3.sh",
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


def _edited_diagnosed_location(
    events: list[dict], diagnostics: list[dict], initial_files: dict | None = None,
    final_files: dict | None = None,
) -> bool:
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
    if initial_files is None or final_files is None:
        return False
    for item in diagnostics:
        path, line = item["path"], item["line"] - 1
        if path not in initial_files or path not in final_files:
            continue
        before = initial_files[path].splitlines()
        after = final_files[path].splitlines()
        for tag, start, end, _, _ in difflib.SequenceMatcher(a=before, b=after).get_opcodes():
            if tag != "equal" and start <= line < max(end, start + 1):
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
             if event.get("type") in {"line_edit", "edit"} and event.get("ok")]
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
        if row.get("done_attempts") and not row.get("first_done_model_generated"):
            raise ValueError("completion attempt was not verified as model-generated")
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
            control_done = control["first_done_prefix_sha256"]
            gate_done = gate["first_done_prefix_sha256"]
            if bool(control_done) != bool(gate_done):
                raise ValueError("control/gate completion boundaries differ before gate intervention")
            if control_done and control_done != gate_done:
                raise ValueError("control/gate trajectories diverge before the first completion attempt")
            if not control_done:
                control_trajectory = control.get("trajectory_sha256")
                gate_trajectory = gate.get("trajectory_sha256")
                if not control_trajectory or not gate_trajectory:
                    raise ValueError("full trajectory hashes are required when neither arm completes")
                if control_trajectory != gate_trajectory:
                    raise ValueError("control/gate trajectories diverge without a gate intervention")


def _publish_revision_payload(
    out: Path, result_payload: dict, drafts: list[dict], rows: list[dict],
    arms: list[str], seeds: int, validate_case_series: bool,
) -> None:
    """Stage a complete result beside its destination, validate it, then publish atomically."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{out.name}.", suffix=".tmp", dir=out.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(result_payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if validate_case_series:
            _validate_case_series_rows(drafts, rows, arms, seeds)
        os.replace(temporary, out)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


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
            done_events = [event for event in events if event.get("type") == "done_attempt"]
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
                "cohort": draft.get("cohort"),
                "initial_visible_pass": draft.get("visible_pass"),
                "initial_held_pass": draft.get("held_pass"),
                "opportunity": any(item["classification"] == "semantic"
                                   for item in draft["draft_diagnostics"]),
                "initial_diagnostics": draft["draft_diagnostics"],
                "final_diagnostics": final_diags,
                **_diag_effect(draft["draft_diagnostics"], final_diags),
                "edited_diagnosed_location": _edited_diagnosed_location(
                    result["events"], draft["draft_diagnostics"], draft["files"], final_files),
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
                "first_done_model_generated": bool(
                    done_events and done_events[0].get("source") == "model"
                ),
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
                "trajectory_sha256": _sha(result["stream"]),
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
        print(f"{draft['task']} {arm}: held={row['held_pass']} clean={row['type_clean']} "
              f"accepted={row['accepted']} edits={row['n_edits']}", flush=True)
    result_payload = {
        "protocol": PROTOCOL_VERSION, "kind": (
            f"{payload.get('kind', 'opportunity_conditioned')}_paired_revisions"
            if payload.get("selection_is_not_a_natural_opportunity_sample")
            else "paired_revisions"
        ),
        "model": args.model, "source_drafts": args.drafts,
        "model_meta": model_meta, "config": vars(args),
        "pyrefly": _pyrefly_meta(), "protocol_source_sha256": _protocol_hashes(),
        "rows": rows,
    }
    _publish_revision_payload(
        Path(args.out), result_payload, drafts, rows, arms, args.seeds,
        validate_case_series=bool(payload.get("selection_is_not_a_natural_opportunity_sample")),
    )
    return 0


def _stub_script(draft: dict, arm: str, tasks: dict) -> list:
    """Deterministic action script for one dry-run cell.

    Clean controls (and any diagnostic-free draft) simply test and submit in every arm.
    Defect drafts: control submits the bad completion; diagnostics repairs after reading
    the one-shot delta, retests, and submits; gate submits, is rejected, repairs,
    retests, and resubmits; noisy repairs first (triggering the volunteered
    after-every-edit check), then tests and submits.
    """
    semantic = any(item["classification"] == "semantic" for item in draft["draft_diagnostics"])
    if draft.get("cohort") == "clean_negative_control" or not semantic:
        return ["<test/>", "<done/>"]
    from scripts.experiments.stub_policy import gold_repair_line_edit
    repair = gold_repair_line_edit(
        draft["files"][draft["target"]], tasks[draft["task"]]["gold_target"], draft["target"]
    )
    if arm == "control":
        return ["<test/>", "<done/>"]
    if arm == "diagnostics":
        return ["<test/>", repair, "<test/>", "<done/>"]
    if arm == "gate":
        return ["<test/>", "<done/>", repair, "<test/>", "<done/>"]
    return [repair, "<test/>", "<done/>"]


def _run_stub_cell(draft: dict, arm: str, script: list, max_new: int, max_turns: int,
                   max_reads: int) -> dict:
    """Execute one scripted trajectory through the real StreamAgent scaffold."""
    from scripts.experiments.stub_policy import CharTokenizer, ScriptedStubModel

    tokenizer = CharTokenizer()
    model = ScriptedStubModel(tokenizer, script)
    env = DeltaDiagnosticEnv(
        draft["files"], draft["target"], draft["test"],
        baseline_diagnostics=draft["baseline_diagnostics"],
        diagnostic_scope={draft["target"]}, held_out_src=draft["held_out"],
        skip_pyrefly=False,
    )
    try:
        agent = StreamAgent(
            model, tokenizer, env, edit_mode="line", sys_override=REVISION_SYS,
            max_new_tokens=max_new, max_turns=max_turns, max_reads=max_reads,
            temperature=0.0, seed=0,
            auto_check=(arm == "noisy"), acceptance_gate=(arm == "gate"),
            lsp_disabled=True,
        )
        started = time.perf_counter()
        result = agent.run(_revision_prompt(draft, arm), draft["target"],
                           editable=[draft["target"]])
        wall = time.perf_counter() - started
        held = bool(env.score()["resolved"])
        final_diags = env.raw_diagnostic_delta()
        final_files = {path: env.read_file(path) for path in env.list_files()}
        checker_latency = env.last_checker_latency
    finally:
        env.close()
    return {
        "result": result, "wall": wall, "held": held, "final_diags": final_diags,
        "final_files": final_files, "checker_latency": checker_latency,
    }


def _stub_row(draft: dict, arm: str, cell: dict, script: list) -> dict:
    """Mirror the revise() row schema so the dry-run grid passes the same validators."""
    result = cell["result"]
    held, final_diags, final_files = cell["held"], cell["final_diags"], cell["final_files"]
    one_shot_latency = draft["checker_latency_ms"] if arm == "diagnostics" else 0.0
    events = result["events"]
    done_events = [event for event in events if event.get("type") == "done_attempt"]
    done_attempts = len(done_events)
    gate_invocations = sum(event.get("type") == "gate_check" for event in events)
    gate_acceptances = sum(event.get("type") == "gate_accept" for event in events)
    gate_rejections = sum(event.get("type") == "gate_reject" for event in events)
    accepted = bool(gate_acceptances) if arm == "gate" else bool(result.get("done_seen"))
    post_rejection_edits, edited_rejected_location = _post_rejection_metrics(events)
    final_file_hashes, final_workspace_hash = _workspace_hashes(final_files)
    serialization_failures = [
        event for event in events
        if event.get("type") == "line_edit"
        and event.get("serialization_mode") == "ambiguous_inline_indentation"
    ]
    return {
        "draft_id": draft["draft_id"], "draft_hash": draft["draft_target_sha256"],
        "task": draft["task"], "group": draft["group"], "arm": arm, "seed": 0,
        "cohort": draft.get("cohort"),
        "initial_visible_pass": draft.get("visible_pass"),
        "initial_held_pass": draft.get("held_pass"),
        "opportunity": any(item["classification"] == "semantic"
                           for item in draft["draft_diagnostics"]),
        "initial_diagnostics": draft["draft_diagnostics"],
        "final_diagnostics": final_diags,
        **_diag_effect(draft["draft_diagnostics"], final_diags),
        "edited_diagnosed_location": _edited_diagnosed_location(
            events, draft["draft_diagnostics"], draft["files"], final_files),
        "visible_pass": bool(result["resolved"]), "held_pass": held,
        "type_clean": not final_diags, "accepted": accepted,
        "semantic_clean": not any(item["classification"] == "semantic"
                                  for item in final_diags),
        "syntax_clean": not any(item["classification"] == "syntax_or_partial"
                                for item in final_diags),
        "accepted_type_clean": bool(accepted and not final_diags),
        "accepted_type_clean_correct": bool(accepted and not final_diags and held),
        "unsubmitted": not bool(done_attempts),
        "done_attempts": done_attempts,
        "first_done_model_generated": bool(
            done_events and done_events[0].get("source") == "model"
        ),
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
        "trajectory_sha256": _sha(result["stream"]),
        "serialization_failures": serialization_failures,
        "final_target_sha256": _sha(final_files[draft["target"]]),
        "final_file_sha256": final_file_hashes,
        "final_workspace_sha256": final_workspace_hash,
        "draft_in_tokens": draft.get("in_tokens"),
        "draft_out_tokens": draft.get("out_tokens"),
        "in_tokens": result["in_tokens"], "out_tokens": result["out_tokens"],
        "turns": result["turns"], "n_edits": result["n_edits"],
        "n_tests": result["n_tests"], "n_checks": result["n_checks"],
        "checker_latency_ms": round(one_shot_latency + cell["checker_latency"] * 1000, 1),
        "wall_sec": round(cell["wall"], 3),
        "termination_reason": result["termination_reason"],
        "events": events,
        "stream_tail": result["stream"][-2500:],
        "scripted_actions": [item if item is not None else "<EOS>" for item in script],
    }


def _event_type_subsequence(events: list[dict], wanted: list[str]) -> bool:
    """True if `wanted` appears as an ordered (not necessarily contiguous) subsequence."""
    index = 0
    for event in events:
        if index < len(wanted) and event.get("type") == wanted[index]:
            index += 1
    return index == len(wanted)


def _dry_run_mechanics(defect: dict, tasks: dict, max_new: int, max_turns: int,
                       max_reads: int) -> list[dict]:
    """Targeted single-cell scenarios for the v6 mechanics, outside the paired grid."""
    from scripts.experiments.stub_policy import gold_repair_line_edit

    repair = gold_repair_line_edit(
        defect["files"][defect["target"]], tasks[defect["task"]]["gold_target"],
        defect["target"]
    )
    gold_sha = defect["gold_repair_validation"]["gold_target_sha256"]
    scenarios = []

    # 1. Observation boundary: the passing-test observation contains literal `<done/>`
    #    text. With v6 cursor advancement no completion may fire from it.
    cell = _run_stub_cell(defect, "control", ["<test/>"], max_new, max_turns, max_reads)
    events = cell["result"]["events"]
    done_attempts = sum(event.get("type") == "done_attempt" for event in events)
    scenarios.append({
        "scenario": "observation_boundary_blocks_literal_done",
        "arm": "control", "script": ["<test/>"],
        "observed": {
            "done_attempts": done_attempts,
            "termination_reason": cell["result"]["termination_reason"],
        },
        "passed": done_attempts == 0
        and cell["result"]["termination_reason"] == "tests_passed_without_done",
    })

    # 2. Stale-test invalidation: a successful edit after a passing test must clear the
    #    resolved state, so an end-of-turn EOS may not terminate the trajectory.
    script = ["<test/>", repair, None, "<test/>", "<done/>"]
    cell = _run_stub_cell(defect, "control", script, max_new, max_turns, max_reads)
    result = cell["result"]
    turn_after_edit = _event_type_subsequence(result["events"], ["line_edit", "turn", "test"])
    scenarios.append({
        "scenario": "stale_test_state_invalidated_by_edit",
        "arm": "control", "script": [s if s is not None else "<EOS>" for s in script],
        "observed": {
            "termination_reason": result["termination_reason"],
            "n_tests": result["n_tests"],
            "turn_delivered_after_edit_eos": turn_after_edit,
            "final_target_is_gold": _sha(cell["final_files"][defect["target"]]) == gold_sha,
        },
        "passed": result["termination_reason"] == "done" and result["n_tests"] == 2
        and turn_after_edit
        and _sha(cell["final_files"][defect["target"]]) == gold_sha,
    })

    # 3. Full rejection -> repair -> retest -> resubmit -> accept cycle with
    #    model-generated completions on both submissions.
    script = ["<test/>", "<done/>", repair, "<test/>", "<done/>"]
    cell = _run_stub_cell(defect, "gate", script, max_new, max_turns, max_reads)
    result = cell["result"]
    events = result["events"]
    done_events = [event for event in events if event.get("type") == "done_attempt"]
    ordered = _event_type_subsequence(events, [
        "test", "done_attempt", "gate_check", "gate_reject",
        "line_edit", "test", "done_attempt", "gate_check", "gate_accept",
    ])
    prefixes = [event.get("prefix_sha256") for event in done_events]
    scenarios.append({
        "scenario": "gate_reject_repair_retest_resubmit_accept",
        "arm": "gate", "script": script,
        "observed": {
            "event_order_ok": ordered,
            "done_sources": [event.get("source") for event in done_events],
            "distinct_completion_prefixes": len(set(prefixes)) == len(prefixes),
            "held_pass": cell["held"], "type_clean": not cell["final_diags"],
            "final_target_is_gold": _sha(cell["final_files"][defect["target"]]) == gold_sha,
        },
        "passed": ordered and len(done_events) == 2
        and all(event.get("source") == "model" for event in done_events)
        and len(set(prefixes)) == 2
        and cell["held"] and not cell["final_diags"]
        and _sha(cell["final_files"][defect["target"]]) == gold_sha,
    })

    # 4. Documented property (descriptive, always "passes" as documentation): after a
    #    rejection the gate mechanically requires a fresh MODEL-GENERATED <done/> and has
    #    invalidated stale test state, but it re-checks diagnostics only — a repaired
    #    resubmission WITHOUT a fresh <test/> is still accepted when type-clean. The
    #    fresh-test step in live runs is elicited by the rejection instruction text.
    script = ["<test/>", "<done/>", repair, "<done/>"]
    cell = _run_stub_cell(defect, "gate", script, max_new, max_turns, max_reads)
    result = cell["result"]
    accepted = sum(event.get("type") == "gate_accept" for event in result["events"])
    scenarios.append({
        "scenario": "resubmission_without_retest_documented",
        "arm": "gate", "script": script,
        "observed": {
            "accepted": bool(accepted), "n_tests": result["n_tests"],
            "done_attempts": sum(
                event.get("type") == "done_attempt" for event in result["events"]
            ),
        },
        "documented_property": (
            "acceptance requires a fresh model-generated <done/>; a fresh <test/> is "
            "instructed, not mechanically enforced"
        ),
        "passed": bool(accepted) and result["n_tests"] == 1,
    })
    return scenarios


def _dry_run_grid_checks(rows: list[dict], drafts: list[dict]) -> dict:
    """Row-level assertions over the scripted grid, by cohort and arm."""
    by_cell = {(row["draft_id"], row["arm"]): row for row in rows}
    defects = [draft for draft in drafts if draft.get("cohort") == "hidden_defect"]
    clean = [draft for draft in drafts if draft.get("cohort") == "clean_negative_control"]
    checks = {
        "all_completions_model_generated": all(
            row["first_done_model_generated"] for row in rows if row["done_attempts"]
        ),
        "no_serialization_failures": not any(row["serialization_failures"] for row in rows),
        "control_accepts_every_bad_completion": all(
            by_cell[(d["draft_id"], "control")]["accepted"]
            and not by_cell[(d["draft_id"], "control")]["type_clean"]
            and not by_cell[(d["draft_id"], "control")]["held_pass"]
            for d in defects
        ),
        "gate_rejects_then_recovers_every_defect": all(
            by_cell[(d["draft_id"], "gate")]["gate_rejections"] == 1
            and by_cell[(d["draft_id"], "gate")]["gate_acceptances"] == 1
            and by_cell[(d["draft_id"], "gate")]["done_attempts"] == 2
            and by_cell[(d["draft_id"], "gate")]["post_rejection_edits"] == 1
            and by_cell[(d["draft_id"], "gate")]["edited_rejected_location"]
            and by_cell[(d["draft_id"], "gate")]["accepted_type_clean_correct"]
            for d in defects
        ),
        "gate_accepts_every_clean_first_check": all(
            by_cell[(d["draft_id"], "gate")]["gate_invocations"] == 1
            and by_cell[(d["draft_id"], "gate")]["gate_rejections"] == 0
            and by_cell[(d["draft_id"], "gate")]["accepted"]
            for d in clean
        ),
        "noisy_volunteers_check_after_edit_on_defects": all(
            by_cell[(d["draft_id"], "noisy")]["n_checks"] >= 1
            and any(event.get("type") == "auto_check"
                    for event in by_cell[(d["draft_id"], "noisy")]["events"])
            for d in defects
        ) if any((d["draft_id"], "noisy") in by_cell for d in defects) else None,
        "diagnostics_prompt_contains_one_shot_delta": all(
            "<diagnostic_delta>" in _revision_prompt(d, "diagnostics") for d in defects
        ),
        "repaired_arms_restore_exact_gold_target": all(
            by_cell[(d["draft_id"], arm)]["final_target_sha256"]
            == d["gold_repair_validation"]["gold_target_sha256"]
            for d in defects for arm in ("diagnostics", "gate", "noisy")
            if (d["draft_id"], arm) in by_cell
        ),
        "control_gate_first_completion_prefixes_identical": all(
            by_cell[(d["draft_id"], "control")]["first_done_prefix_sha256"]
            == by_cell[(d["draft_id"], "gate")]["first_done_prefix_sha256"]
            for d in drafts
        ),
    }
    return checks


def dry_run(args) -> int:
    """Scripted-policy dry run of the phase-gradient arms (no model, CPU only)."""
    payload = json.loads(Path(args.drafts).read_text())
    if Path(args.out).exists():
        raise FileExistsError(f"refusing to overwrite dry-run artifact: {args.out}")
    drafts = [draft for draft in payload["drafts"]
              if (draft.get("draft_submitted") or draft.get("revision_case_series_eligible"))
              and draft.get("coherent")]
    if args.names:
        wanted = set(args.names.split(","))
        drafts = [draft for draft in drafts if draft["task"] in wanted]
    if not drafts:
        raise ValueError("no eligible drafts selected for the dry run")
    arms = args.arms.split(",")
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise ValueError(f"unknown arms: {sorted(unknown)}")
    tasks = _task_map()
    rows = []
    for draft, arm in itertools.product(drafts, arms):
        script = _stub_script(draft, arm, tasks)
        cell = _run_stub_cell(draft, arm, script, args.max_new, args.max_turns,
                              args.max_reads)
        row = _stub_row(draft, arm, cell, script)
        rows.append(row)
        print(f"{draft['draft_id']} {arm}: held={row['held_pass']} "
              f"clean={row['type_clean']} accepted={row['accepted']} "
              f"rejects={row['gate_rejections']} done={row['done_attempts']}", flush=True)
    grid_checks = _dry_run_grid_checks(rows, drafts)
    defects = [draft for draft in drafts if draft.get("cohort") == "hidden_defect"]
    mechanics = _dry_run_mechanics(defects[0], tasks, args.max_new, args.max_turns,
                                   args.max_reads) if defects else []
    passed = all(value for value in grid_checks.values() if value is not None) and all(
        scenario["passed"] for scenario in mechanics
    )
    result_payload = {
        "protocol": PROTOCOL_VERSION,
        "benchmark": payload.get("benchmark"),
        "kind": "controlled_gate_scripted_dry_run",
        "model": "scripted-stub-policy",
        "model_meta": {
            "policy": "deterministic scripted action replay through the live StreamAgent",
            "weights_loaded": False,
            "token_unit": "characters (stub char-level tokenizer), not model tokens",
        },
        "source_drafts": args.drafts,
        "source_drafts_sha256": hashlib.sha256(Path(args.drafts).read_bytes()).hexdigest(),
        "config": vars(args),
        "pyrefly": _pyrefly_meta(),
        "protocol_source_sha256": _protocol_hashes(),
        "arm_divergence_notes": {
            "control_vs_gate": (
                "identical system and user prompts; identical scripted prefixes through "
                "the first completion attempt, enforced by the v6 case-series validator"
            ),
            "diagnostics": (
                "the one-shot delta is appended to the revision prompt, so trajectories "
                "may diverge from the first token (C26 delivery design; accepted)"
            ),
            "noisy": (
                "the after-every-edit advertisement changes the system prompt and each "
                "applied edit volunteers a checker observation, so trajectories diverge "
                "before any completion (authoring feedback-arm design; accepted)"
            ),
        },
        "rows": rows,
        "grid_checks": grid_checks,
        "mechanics_scenarios": mechanics,
        "passed": passed,
    }
    _publish_revision_payload(
        Path(args.out), result_payload, drafts, rows, arms, 1,
        validate_case_series=bool(payload.get("selection_is_not_a_natural_opportunity_sample")),
    )
    print(f"grid checks: { {k: v for k, v in grid_checks.items()} }")
    for scenario in mechanics:
        print(f"mechanics {scenario['scenario']}: {'PASS' if scenario['passed'] else 'FAIL'}")
    print(f"dry run {'PASSED' if passed else 'FAILED'} -> {args.out}")
    return 0 if passed else 2


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

    dry = commands.add_parser("dry-run")
    dry.add_argument("drafts")
    dry.add_argument("out")
    dry.add_argument("--names", default=None)
    dry.add_argument("--arms", default="control,diagnostics,gate,noisy")
    dry.add_argument("--max-new", type=int, default=8000)
    dry.add_argument("--max-turns", type=int, default=12)
    dry.add_argument("--max-reads", type=int, default=4)

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
    if args.command == "dry-run":
        return dry_run(args)
    if args.command == "generate":
        return generate_drafts(args)
    return revise(args)


if __name__ == "__main__":
    raise SystemExit(main())
