import math
import json
from pathlib import Path
import re
import subprocess
import sys

from scaffold.mock_env import MultiFileEnv
from scaffold.stream_agent import (
    LINE_EDIT_RE,
    StreamAgent,
    _normalize_inline_edit,
    _strip_fences,
)
from scripts.synth_delivery import ARMS as DELIVERY_ARMS, ARM_WITNESS, SYS_LINE_DELIVERY, parse_args
from scripts.analysis.effic_real_stats import binom_two_sided
from scripts.analysis.analyze_checker_paired import (
    controlled_gate_cohort_audit,
    end_to_end_summary,
    end_to_end_contrast,
    expected_cost_per_accepted_correct,
    gate_pairing_audit,
    paired_contrast,
    summarize_rows,
)
from scripts.analysis.analyze_checker_gate_v2 import analyze as analyze_checker_gate_v2
from scripts.analysis.analyze_navigation import interaction, paired_ratio
from scripts.analysis.analyze_retrieval_paired import _paired as retrieval_paired
from scripts.analysis.analyze_retrieval_suite import load_suite
from scripts.experiments.diagnostics import (
    DeltaDiagnosticEnv,
    collect_diagnostics,
    delta,
    format_diagnostics,
    is_coherent,
)
from scripts.experiments.navigation_tasks import build_prompt, build_tasks
from scripts.experiments.checker_paired import (
    REVISION_SYS,
    _edited_diagnosed_location,
    _publish_revision_payload,
    _validate_case_series_rows,
)
from scripts.experiments.run_navigation import (
    _control_floor_failure,
    _method_from_metadata,
    _positive_result,
)


def test_navigation_generator_pairs_runtime_and_hides_target(tmp_path):
    task = build_tasks(tmp_path, "pilot")[0]
    typed = task["variants"]["typed"]["files"]
    erased = task["variants"]["erased"]["files"]
    assert typed["pkg/factory.py"] == erased["pkg/factory.py"]
    assert typed["pkg/factory.pyi"] != erased["pkg/factory.pyi"]
    assert all(typed[path] == erased[path] for path in typed if path.endswith(".py"))
    assert "cast(" not in typed["pkg/factory.py"]
    for token, class_name in task["registry_contract"].items():
        assert f'Literal["{token}"]' in typed["pkg/factory.pyi"]
        assert f"-> {class_name}" in typed["pkg/factory.pyi"]
    assert 8 <= task["n_overrides"] <= 15
    prompt = build_prompt(task, "typed")
    assert task["target_class"] not in prompt
    assert task["target_path"] not in prompt
    positive = _positive_result(task)
    assert task["gold"]["new_text"] in positive
    assert task["target_method_span"]["source"] not in positive
    current_span, current_path = _method_from_metadata(task)
    assert current_path == task["target_path"]
    assert task["target_method_span"]["source"].splitlines()[-1] in current_span
    assert task["gold"]["new_text"] not in current_span
    repo = task["variants"]["typed"]["repo_dir"]
    probe = subprocess.run(
        [sys.executable, "-c", (
            "from pkg.factory import make\n"
            + "actual = {k: type(make(k)).__name__ for k in " + repr(task["tokens"]) + "}\n"
            + "assert actual == " + repr(task["registry_contract"]) + "\n"
        )],
        cwd=repo, capture_output=True, text=True,
    )
    assert probe.returncode == 0, probe.stderr


def test_diagnostic_delta_and_presentation():
    base = [{
        "path": "target.py", "line": 2, "column": 1, "stop_line": 2, "stop_column": 2,
        "code": "bad-return", "message": "wrong", "classification": "semantic", "frame": "2| x",
    }]
    moved_same_error = [{**base[0], "line": 4, "frame": "4| x"}]
    novel = [{**base[0], "code": "missing-attribute", "message": "no member"}]
    assert delta(moved_same_error, base) == []
    assert delta(novel, base) == novel
    assert "missing-attribute" in format_diagnostics(novel)


def test_coherence_requires_parseable_implemented_source():
    assert is_coherent("def f() -> int:\n    return 1\n")
    assert not is_coherent("def f():\n    raise NotImplementedError()\n")
    assert not is_coherent("def f(:\n    pass\n")


def test_behavioral_test_runner_does_not_pollute_workspace():
    env = MultiFileEnv({"target.py": "x = 1\n"}, "target.py", "from target import x\nassert x == 1",
                       skip_pyrefly=True)
    try:
        assert env.run_tests()["resolved"]
        assert not (Path(env.ws) / "_run_tests.py").exists()
    finally:
        env.close()


def test_checker_environment_applies_unique_exact_search_edit():
    env = DeltaDiagnosticEnv(
        {"target.py": "def f() -> int:\n    return 0\n"}, "target.py", "",
        baseline_diagnostics=[], diagnostic_scope={"target.py"}, skip_pyrefly=True,
    )
    try:
        assert env.apply_edit("target.py", "    return 0", "    return 1") == (True, "ok")
        assert "return 1" in env.read_file("target.py")
        assert env.apply_edit("target.py", "missing", "replacement")[0] is False
    finally:
        env.close()


def test_collect_diagnostics_resolves_pyrefly_relative_paths():
    env = MultiFileEnv({"target.py": 'def f() -> int:\n    return "wrong"\n'}, "target.py", "",
                       skip_pyrefly=False)
    try:
        diagnostics, _ = collect_diagnostics(env.ws, {"target.py"})
        assert diagnostics
        assert diagnostics[0]["path"] == "target.py"
        assert diagnostics[0]["code"] == "bad-return"
    finally:
        env.close()


def test_exact_sign_test_handles_all_ties_as_no_evidence():
    assert binom_two_sided(0, 0) == 1.0


def test_retrieval_suite_combines_disjoint_frozen_shards():
    root = Path(__file__).resolve().parents[1]
    rows, payload = load_suite([
        root / "runs/pilot/retrieval_paired_qwen35_27b_pilot3.json",
        root / "runs/pilot/retrieval_paired_qwen35_27b_remaining8.json",
    ])
    assert payload["protocol"] == "retrieval-paired-v1"
    assert len(rows) == 25
    result = retrieval_paired(rows, "text", "definition", draws=1_000)
    assert result["n_matched_success"] == 11
    assert result["tasks_treatment_cheaper"] == 10
    assert result["ratio_baseline_over_treatment"] > 1


def test_hidden_defect_gate_result_exercises_valid_rejection_and_repair():
    root = Path(__file__).resolve().parents[1]
    rows = json.loads((
        root / "runs/pilot/checker_hidden_qwen35_27b_multiline_pilot3.json"
    ).read_text())["rows"]
    assert gate_pairing_audit(rows)["valid"]
    control = [row for row in rows if row["arm"] == "control"]
    diagnostics = [row for row in rows if row["arm"] == "diagnostics"]
    gate = [row for row in rows if row["arm"] == "gate"]
    assert len(control) == len(diagnostics) == len(gate) == 3
    assert all(row["accepted"] and not row["held_pass"] for row in control + diagnostics)
    assert all(
        row["gate_rejections"] == 1
        and row["post_rejection_edits"] == 1
        and row["held_pass"]
        and row["type_clean"]
        and not row["accepted"]
        and not row["serialization_failures"]
        for row in gate
    )
    # V5 predates the observation/action cursor boundary. Its first completion
    # cannot be certified as model-generated and is superseded by the v6 cohort.
    assert all(not row.get("first_done_model_generated") for row in rows)


def test_checker_gate_v2_recovers_bad_completions_and_accepts_clean_controls():
    root = Path(__file__).resolve().parents[1]
    result = analyze_checker_gate_v2(
        json.loads((root / "runs/protocol/checker_gate_v2_pilot3.json").read_text()),
        json.loads((root / "runs/pilot/checker_gate_qwen35_27b_pilot3.json").read_text()),
    )
    assert result["gate_pairing"]["valid"]
    assert result["all_completions_model_generated"]
    assert result["defect_cohort"]["bad_completion_opportunities_reached"] == 2
    assert result["defect_cohort"]["conditional_accepted_recovery_rate"] == 1
    assert result["defect_cohort"]["gate_accepted_type_clean_correct"] == 3
    assert result["clean_cohort"]["false_rejection_rate"] == 0
    assert result["clean_cohort"]["first_submission_accepted"] == 3


def test_controlled_gate_cohort_audit_separates_recovery_from_false_rejection():
    rows = [
        {
            "arm": "gate", "opportunity": True, "gate_rejections": 1,
            "gate_acceptances": 1, "gate_rejected_then_accepted": True,
            "accepted": True, "accepted_type_clean_correct": True,
            "held_pass": True, "accepted_dirty": False,
        },
        {
            "arm": "gate", "opportunity": False, "gate_rejections": 0,
            "gate_acceptances": 1, "gate_rejected_then_accepted": False,
            "accepted": True, "accepted_type_clean_correct": True,
            "held_pass": True, "accepted_dirty": False,
        },
    ]
    audit = controlled_gate_cohort_audit(rows)
    assert audit["estimable"]
    assert audit["defect_rejected_then_accepted_rate"] == 1
    assert audit["clean_false_rejection_rate"] == 0
    assert audit["clean_first_submission_accepted_rate"] == 1


def test_line_edit_parser_accepts_same_line_body_without_losing_indentation():
    match = LINE_EDIT_RE.search(
        '<edit path="pkg/target.py" lines="5-6">    def f(self):\n        return 1</edit>'
    )
    assert match
    assert _strip_fences(match["body"]) == "    def f(self):\n        return 1"


def test_inline_edit_serialization_is_anchored_to_current_indentation():
    cases = (
        (" from lib import fold", "from lib import fold\n", 1,
         "from lib import fold", "inline_separator_removed"),
        ("from lib import fold", "from lib import fold\n", 1,
         "from lib import fold", "inline_exact_indentation"),
        ("    def f(self):", "    def old(self):\n", 1,
         "    def f(self):", "inline_exact_indentation"),
        ("     def f(self):", "    def old(self):\n", 1,
         "    def f(self):", "inline_separator_removed"),
    )
    for body, source, start, expected, mode in cases:
        assert _normalize_inline_edit(body, source, start) == (expected, mode)


def test_inline_edit_serialization_rejects_ambiguous_indentation():
    assert _normalize_inline_edit("  from lib import fold", "from lib import fold\n", 1) == (
        None, "ambiguous_inline_indentation"
    )


def test_checker_revision_requires_multiline_line_edit_and_diff_localizes_repair():
    assert '<edit path="F" lines="A-B">\nnew code' in REVISION_SYS
    assert "same line as the opening tag" in REVISION_SYS
    diagnostics = [{"path": "target.py", "line": 2}]
    assert _edited_diagnosed_location(
        [{"type": "edit", "path": "target.py", "ok": True}], diagnostics,
        {"target.py": "def f() -> int:\n    return 'bad'\n"},
        {"target.py": "def f() -> int:\n    return 0\n"},
    )
    assert _normalize_inline_edit("\treturn 1", "\treturn 0\n", 1) == (
        None, "ambiguous_inline_indentation"
    )


def test_newline_edit_body_preserves_leading_space():
    match = LINE_EDIT_RE.search('<edit path="x.py" lines="1">\n value = 1</edit>')
    assert match and match["newline"] == "\n"
    assert _strip_fences(match["body"]) == " value = 1"


def test_report_claim_links_have_row_level_ledger_anchors():
    root = Path(__file__).resolve().parents[1]
    report = (root / "REPORT.md").read_text()
    ledger = (root / "evidence" / "claim_ledger.md").read_text()
    references = set(re.findall(r"claim_ledger\.md#(c\d+)", report))
    anchors = set(re.findall(r'<a id="(c\d+)"></a>C\d+', ledger))
    assert references <= anchors


def test_navigation_token_equivalence_uses_ratio_of_task_weighted_means():
    rows = []
    for task, base, treatment in (("small", 10, 20), ("large", 100, 100)):
        rows.extend([
            {"task": task, "variant": "typed", "arm": "baseline",
             "held_out_pass": True, "in_tokens": base, "out_tokens": 0},
            {"task": task, "variant": "typed", "arm": "semantic_auto",
             "held_out_pass": True, "in_tokens": treatment, "out_tokens": 0},
        ])
    result = paired_ratio(
        rows, ("typed", "baseline"), ("typed", "semantic_auto"),
        "total_tokens", bootstrap=100, seed=1,
    )
    assert result["ratio_of_task_weighted_means"] == 120 / 110
    assert result["mean_task_ratio_descriptive"] == 1.5
    assert not result["exploratory_margin_compatible"]


def test_checker_expected_cost_includes_draft_and_failed_attempts():
    drafts = [
        {"draft_id": "d1", "task": "t1", "draft_submitted": True, "coherent": True,
         "draft_diagnostics": [], "in_tokens": 100, "out_tokens": 20},
        {"draft_id": "d2", "task": "t2", "draft_submitted": True, "coherent": True,
         "draft_diagnostics": [], "in_tokens": 200, "out_tokens": 20},
    ]
    rows = [
        {"draft_id": "d1", "task": "t1", "arm": "gate", "accepted": True,
         "held_pass": True, "type_clean": True, "in_tokens": 50, "out_tokens": 10},
        {"draft_id": "d2", "task": "t2", "arm": "gate", "accepted": False,
         "held_pass": False, "type_clean": True, "in_tokens": 30, "out_tokens": 10},
    ]
    result = expected_cost_per_accepted_correct(
        drafts, rows, "gate", False, bootstrap=100, seed=1,
    )
    assert result["accepted_correct_rate"] == 0.5
    assert result["mean_draft_plus_revision_tokens"] == 220
    assert result["expected_tokens_per_accepted_correct_patch"] == 440
    assert math.isinf(result["ci95"][1])


def test_checker_end_to_end_counts_incoherent_draft_as_failure():
    drafts = [
        {"draft_id": "d1", "task": "t1", "draft_submitted": True, "coherent": True,
         "draft_diagnostics": [], "in_tokens": 100, "out_tokens": 20},
        {"draft_id": "d2", "task": "t2", "draft_submitted": True, "coherent": False,
         "draft_diagnostics": [], "in_tokens": 200, "out_tokens": 20},
    ]
    rows = [{
        "draft_id": "d1", "task": "t1", "arm": "control", "accepted": True,
        "held_pass": True, "type_clean": True, "semantic_clean": True,
        "accepted_type_clean_correct": True,
        "gate_invocations": 0, "gate_rejections": 0, "gate_acceptances": 0,
        "unsubmitted": False,
        "in_tokens": 50, "out_tokens": 10,
    }]
    result = end_to_end_summary(drafts, rows, "control")
    assert result["final_held_pass_yield"] == 0.5
    assert result["accepted_correct_yield"] == 0.5
    assert result["pre_revision_failure_rate"] == 0.5


def test_checker_summary_keeps_missing_draft_cost_non_estimable():
    row = {
        "task": "t", "held_pass": True, "accepted": True, "type_clean": True,
        "semantic_clean": True, "accepted_type_clean_correct": True,
        "unsubmitted": False, "gate_invocations": 0, "gate_rejections": 0,
        "gate_acceptances": 0, "edited_diagnosed_location": True,
        "diagnostics_eliminated": 1, "diagnostics_retained": 0,
        "diagnostics_introduced": 0, "draft_in_tokens": None,
        "draft_out_tokens": None, "in_tokens": 10, "out_tokens": 5,
        "turns": 1, "checker_latency_ms": 2, "wall_sec": 3,
    }
    result = summarize_rows([row])
    assert not result["draft_plus_revision_tokens_estimable"]
    assert result["draft_plus_revision_tokens_mean"] is None
    assert result["revision_tokens_mean"] == 15


def test_unsubmitted_gate_is_not_a_rejection():
    drafts = [{"draft_id": "d", "task": "t", "draft_submitted": True,
               "coherent": True}]
    common = {"draft_id": "d", "task": "t", "seed": 0, "accepted": False,
              "held_pass": False, "semantic_clean": False,
              "accepted_type_clean_correct": False, "unsubmitted": True,
              "gate_invocations": 0, "gate_rejections": 0, "gate_acceptances": 0,
              "first_done_prefix_sha256": None, "trajectory_sha256": "same"}
    rows = [{**common, "arm": "control"}, {**common, "arm": "gate"}]
    rejected = end_to_end_contrast(drafts, rows, "gate", "gate_rejection", 100, 1)
    unsubmitted = end_to_end_contrast(drafts, rows, "gate", "unsubmitted", 100, 1)
    assert rejected["mean_delta"] == 0
    assert unsubmitted["mean_delta"] == 0


def test_gate_pairing_rejects_divergent_trajectories_without_completion():
    common = {
        "draft_id": "d", "task": "t", "seed": 0,
        "first_done_prefix_sha256": None, "serialization_failures": [],
        "gate_invocations": 0, "gate_acceptances": 0, "gate_rejections": 0,
        "n_checks": 0, "accepted_dirty": False,
    }
    rows = [
        {**common, "arm": "control", "trajectory_sha256": "control"},
        {**common, "arm": "gate", "trajectory_sha256": "gate"},
    ]
    audit = gate_pairing_audit(rows)
    assert not audit["valid"]
    assert audit["invalid_pairs"][0]["reason"] == (
        "full trajectories differ without gate intervention"
    )
    try:
        _validate_case_series_rows([{"draft_id": "d"}], rows, ["control", "gate"], 1)
    except ValueError as error:
        assert "diverge without a gate intervention" in str(error)
    else:
        raise AssertionError("divergent no-completion trajectories passed validation")


def test_failed_case_series_validation_never_publishes_target(tmp_path):
    common = {
        "draft_id": "d", "task": "t", "seed": 0,
        "first_done_prefix_sha256": None, "serialization_failures": [],
        "gate_invocations": 0, "gate_acceptances": 0, "gate_rejections": 0,
        "n_checks": 0, "accepted_dirty": False,
    }
    rows = [
        {**common, "arm": "control", "trajectory_sha256": "control"},
        {**common, "arm": "gate", "trajectory_sha256": "gate"},
    ]
    out = tmp_path / "result.json"
    try:
        _publish_revision_payload(
            out, {"rows": rows}, [{"draft_id": "d"}], rows,
            ["control", "gate"], 1, validate_case_series=True,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid case series was published")
    assert not out.exists()
    assert list(tmp_path.iterdir()) == []


def test_selected_case_series_analyzer_suppresses_natural_cohort_labels():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([
        sys.executable, "scripts/analysis/analyze_checker_paired.py",
        "--drafts", "runs/protocol/checker_opportunity_case_series_v4.json",
        "--revisions", "runs/pilot/checker_case_series_qwen36_27b_6a9e13bd_v2_s1.json",
        "--bootstrap", "10",
    ], cwd=root, capture_output=True, text=True, check=True)
    assert "unconditional opportunity rate" not in result.stdout
    assert "coherent_submitted_revision" not in result.stdout
    assert "ALL NATURAL DRAFTS" not in result.stdout
    assert "selected_checker_positive_revision" in result.stdout
    pairing_line = next(
        line for line in result.stdout.splitlines() if line.startswith("GATE PAIRING AUDIT ")
    )
    pairing = json.loads(pairing_line.removeprefix("GATE PAIRING AUDIT "))
    assert not pairing["valid"]
    assert "recorded trajectories differ" in pairing["invalid_pairs"][0]["reason"]


def test_checker_end_to_end_contrast_uses_joint_correct_clean_acceptance():
    drafts = [
        {"draft_id": "d1", "task": "t1", "draft_submitted": True, "coherent": True},
        {"draft_id": "d2", "task": "t2", "draft_submitted": True, "coherent": False},
    ]
    common = {
        "task": "t1", "draft_id": "d1", "seed": 0, "accepted": True,
        "semantic_clean": True, "first_done_prefix_sha256": "same",
    }
    rows = [
        {**common, "arm": "control", "held_pass": False,
         "accepted_type_clean_correct": False},
        {**common, "arm": "gate", "held_pass": True,
         "accepted_type_clean_correct": True},
    ]
    result = end_to_end_contrast(
        drafts, rows, "gate", "accepted_type_clean_correct", bootstrap=100, seed=1
    )
    assert result["estimable"]
    assert result["mean_delta"] == 0.5


def test_navigation_span_control_enforces_floor():
    passed = [{"arm": "semantic_span_control", "held_out_pass": True}]
    failed = [{"arm": "semantic_span_control", "held_out_pass": False}]
    assert _control_floor_failure("span-control", passed) is None
    assert _control_floor_failure("span-control", failed)


def test_paired_analyzers_reject_incomplete_nested_grids():
    nav_rows = [
        {"task": "t", "seed": 0, "variant": variant, "arm": arm,
         "held_out_pass": True, "in_tokens": 1, "out_tokens": 0}
        for variant, arm in (
            ("typed", "baseline"), ("typed", "semantic_auto"),
            ("erased", "baseline"), ("erased", "semantic_auto"),
        )
    ]
    nav_rows[-1]["seed"] = 1
    assert not interaction(nav_rows, "pass", 10, 1)["estimable"]

    checker_rows = [
        {"task": "t", "draft_id": "d", "seed": 0, "arm": "control", "held_pass": True},
        {"task": "t", "draft_id": "d", "seed": 1, "arm": "gate", "held_pass": True},
    ]
    assert not paired_contrast(checker_rows, "gate", "held_pass", False, 10, 1)["estimable"]


def test_delivery_channel_is_off_unless_an_arm_asks_for_it():
    """Every other experiment drives this agent with no delivery kwargs; the C37 channel
    must stay dead for them, or a diagnostic leaks into an unrelated condition."""
    agent = StreamAgent(None, None, None, edit_mode="line", device="cpu")
    assert agent.cond is None
    assert (agent.debounce, agent.latency) == (0, 8)
    assert not any((agent.pause_align, agent.announce_lsp, agent.c_eager,
                    agent.syntax_gate, agent.rich_signal))


def test_delivery_arms_are_distinct_and_each_names_its_own_witness_event():
    assert set(DELIVERY_ARMS) == set(ARM_WITNESS)
    configs = {name: tuple(sorted(kw.items())) for name, kw in DELIVERY_ARMS.items()}
    assert len(set(configs.values())) == len(configs)
    # The witness the analyzer keys arm identity on, independent of any filename.
    assert ARM_WITNESS["A"] is None
    assert ARM_WITNESS["C-lazy"] == "diag_sync_queued"
    assert ARM_WITNESS["C-eager"] == "diag_eager"
    assert all(ARM_WITNESS[a] == "diag_debounced" for a in ("D-naive", "D-plain", "D-gate"))
    # Only the naive arm carries the announce sentence; only the gate arm parses before delivering.
    assert [a for a, kw in DELIVERY_ARMS.items() if kw.get("announce_lsp")] == ["D-naive"]
    assert [a for a, kw in DELIVERY_ARMS.items() if kw.get("syntax_gate")] == ["D-gate"]


def test_delivery_witness_events_are_the_ones_the_agent_emits():
    """ARM_WITNESS and stats_delivery.py identify an arm by the event name in its rows.
    Those names are string literals in stream_agent.py, so a rename there would leave the
    arm table and the analyzer silently matching nothing while every test stayed green.
    Tie the two together."""
    root = Path(__file__).resolve().parents[1]
    agent_src = (root / "scaffold" / "stream_agent.py").read_text()
    emitted = set(re.findall(r'"type":\s*"(diag_[a-z_]+)"', agent_src))
    assert emitted, "no diagnostic events found in the agent; the delivery channel is gone"
    for arm, witness in ARM_WITNESS.items():
        if witness is None:
            continue
        assert witness in emitted, (
            f"arm {arm} keys on {witness!r}, which stream_agent.py no longer emits; "
            f"it emits {sorted(emitted)}"
        )


def test_delivery_sampling_defaults_match_the_published_arms():
    """The pinned prompt and the ARMS table only fix the channel and the timing. Sampling
    limits sit outside both, so the runner can reproduce an arm's flags exactly and still
    differ from the published run. That happened: the first v2 launch inherited a 1400-token
    cap against June's 2200 and truncated rollouts that routinely hit the cap. Assert the
    defaults against the configs recovered from the June checkpoints."""
    root = Path(__file__).resolve().parents[1]
    prov = json.loads((root / "runs" / "agent" / "synth_delivery_provenance.json").read_text())
    configs = [c for f in prov["files"].values() if (c := f.get("config"))]
    assert configs, "no recovered June configs; provenance file is empty or restructured"
    defaults = vars(parse_args([]))
    for key in ("max_new", "latency"):
        june = {c[key] for c in configs if key in c}
        assert len(june) == 1, f"June arms disagree on {key}: {june}"
        assert defaults[key] == june.pop(), (
            f"runner defaults {key}={defaults[key]}, but every published C37 arm ran "
            f"{key}={june}; a re-run at this default is not comparable to the June numbers"
        )


def test_delivery_prompt_is_pinned_to_the_june_harness_prompt():
    """The agent's live SYS_LINE has drifted (gained <defn>, lost the analyzer sentence).
    The delivery arms must keep the prompt the committed C37 rollouts were produced under."""
    root = Path(__file__).resolve().parents[1]
    june = subprocess.run(
        ["git", "show", "779aa5c^:scaffold/stream_agent.py"],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout
    namespace = {}
    exec(june[june.index('SYS_LINE = """'):june.index("\nSYS_REWRITE")], namespace)
    assert SYS_LINE_DELIVERY == namespace["SYS_LINE"]
