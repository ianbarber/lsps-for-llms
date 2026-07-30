#!/usr/bin/env python3
"""C36: does the substitution adapter suppress reads CONDITIONALLY or BLANKET?

The substitution adapter (`runs/sft/substitution_lora_27b`) drove reread of the defining file from
11/12 -> 0/12 (C33) and 12/12 -> 1/12 (C35). Every training instance had a SUFFICIENT span. This
driver runs the same protocol on the `readreq` split, whose span is genuine but INSUFFICIENT, so
that reading is the CORRECT action:

  trained, seeks the missing information and repairs  -> CONDITIONAL   (the policy is fine)
  trained, does not seek and fails                    -> BLANKET       (the policy is harmful)

ARMS (typed variant only, temperature 0, one rollout per cell -- the C33/C35 protocol). Every arm
is run twice: untrained, and with `--adapter runs/sft/substitution_lora_27b`.

  push_insufficient  hoisted repo; the genuine live method span pushed unsolicited under AUTO_SYS
                     in the byte-identical <semantic_result kind="current_source"> wrapper. THE CELL.
  push_sufficient    inline-twin repo (exactly ONE line differs), identical delivery. Sufficiency
                     control and adapter-live gate.
  push_chained       hoisted repo (byte-identical to arm 1); payload = the method span PLUS a
                     second genuine live goto at the helper token inside the override. Information
                     control with structure held exactly fixed.

Arms 2 and 3 bracket the delegation/out-of-distribution confound from both sides: arm 2 holds
delivery fixed and varies the repo by one line; arm 3 holds the repo byte-identical and varies only
how much genuine language-server output is delivered.

CO-PRIMARY OUTCOMES
  read_after_span                 -- `_reread_metrics`, imported verbatim from
                                     run_navigation_reread, so the number stays numerically
                                     comparable to C33/C35.
  retrieved_missing_information   -- a read (whole or ranged) or a grep that reaches the defining
                                     file, at any point in the rollout, with paths normalised.
                                     This is what the verdict keys on: it is the behaviour the
                                     adapter was trained to suppress.
  repair_correct                  -- visible_pass AND held_out_pass. A workspace that fails the
                                     visible test is not a repair, and scoring the conjunction
                                     closes a measured hazard in the xor family (see the row
                                     comment on `repair_correct`). Raw `held_out_pass` is
                                     published unchanged for comparability with C33/C35.

SECONDARY (reported, not decisive)
  sought_missing_information[_before_first_edit] -- the broader disjunction that also counts a
    denied <defn> and a runtime probe edit, with its four components broken out
    (`sought_component_*`) so a cell that only ever emits a refused <defn> and then guesses is
    visible rather than absorbed into "the adapter seeks the missing information".

Frozen-protocol constraint: navigation_tasks.py, run_navigation.py, run_navigation_reread.py and
scaffold/* stay byte-identical and are imported read-only. The hoist and the two new splits are
injected at runtime by navigation_readreq_tasks. This driver is the ONLY entry point for the
readreq splits: they are deliberately absent from run_navigation_reread.py's `--split` choices,
because that driver installs neither the held-out guard nor the pycache purge and would produce
numbers that look comparable to C33/C35 while being scored under different hygiene.

The base revision is DEFAULTED to the one the adapter was trained on and asserted equal to the
adapter's own recorded revision, so `main` can never silently move the base weights out from
under the LoRA.

Usage:
  python scripts/experiments/run_navigation_readreq.py runs/readreq/readreq_pilot_untrained.json \
      --split readreq_pilot --arms push_insufficient,push_sufficient \
      --validation runs/protocol/navigation_v2_readreq_pilot_validation.json --gpu-only
  python scripts/experiments/run_navigation_readreq.py runs/readreq/readreq_trained.json \
      --split readreq --adapter runs/sft/substitution_lora_27b --gpu-only
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
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
from scripts.experiments import navigation_readreq_tasks as RR  # noqa: E402
from scripts.experiments import navigation_tasks as NT  # noqa: E402
from scripts.experiments.run_navigation import (  # noqa: E402
    AUTO_SYS,
    _format_method_span,
    _method_from_lsp,
    _metrics,
)
from scripts.experiments.run_navigation_reread import (  # noqa: E402
    _reread_metrics,
    _shared_span_and_verify,
)


PUSH_PREAMBLE = (
    "\n\nThe following current source span was supplied from a language-server definition result "
    "at the visible call site. It is source context, not a proposed correction.\n"
    "<semantic_result kind=\"current_source\">\n{payload}\n</semantic_result>"
)
# arm 3 is composed from TWO real server answers and says so; it is never presented as one result.
CHAINED_PREAMBLE = (
    "\n\nThe following current source spans were supplied from two language-server definition "
    "results: one at the visible call site, and one at the call inside the definition that "
    "returned. They are source context, not a proposed correction.\n"
    "<semantic_result kind=\"current_source\" chained=\"true\">\n{payload}\n</semantic_result>"
)

ARM_SPECS = {
    "push_insufficient": {"flavour": "hoisted", "auto": True, "system": AUTO_SYS,
                          "chained": False, "preamble": PUSH_PREAMBLE},
    "push_sufficient": {"flavour": "inline", "auto": True, "system": AUTO_SYS,
                        "chained": False, "preamble": PUSH_PREAMBLE},
    "push_chained": {"flavour": "hoisted", "auto": True, "system": AUTO_SYS,
                     "chained": True, "preamble": CHAINED_PREAMBLE},
}

PROBE_MARKERS = ("raise ", "print(", "assert ", "1/0", "sys.exit")
DELIVERING_CHANNELS = ("model_read_whole", "model_read_ranged", "grep_scan", "post_edit_view")
RANGED_READ_RE = re.compile(r'<read\s+path="(?P<path>[^"]+)"\s+lines="(?P<a>\d+)\s*-\s*(?P<b>\d+)"')

# The base revision the adapter was trained against (runs/sft/substitution_lora_27b/
# streams_train_meta.json) and the one C33 and C35 pinned. Defaulted, not optional: resolving
# `main` to whatever it points at today would put the LoRA on different base weights and void
# every comparison with those two claims.
BASE_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"

# The pre-registered validity gates (V1-V6) and the CONDITIONAL/BLANKET/MIXED decision rule live
# in scripts/analysis/analyze_navigation_readreq.py, which is the only place that reports a verdict.
PILOT_FLOOR = 5   # of 6, on arm 1 untrained, for BOTH repair_correct and retrieved_missing_information


# ---------------------------------------------------------------------------
# per-rollout audit
# ---------------------------------------------------------------------------
def _chained_payload(task: dict, env, span: str) -> tuple[str, dict]:
    """span + a SECOND genuine live goto at the helper token on the delegation line."""
    helper = task["helper"]
    source = env.read_file(task["target_path"])
    call_line = helper["call_line"]
    col = source.splitlines()[call_line - 1].index(helper["name"]) + 1
    chained_span, chained_path = env.lsp_definition(
        helper["name"], file=task["target_path"], line=call_line, col=col
    )
    if chained_path != task["target_path"] or not chained_span:
        raise RuntimeError(f"chained goto did not resolve the helper: {chained_path}")
    if chained_span != helper["source"]:
        raise RuntimeError("chained goto did not return the pristine helper source")
    formatted = _format_method_span(chained_path, helper["def_line"], helper["return_line"],
                                    chained_span)
    audit = {
        "chained_span_path": chained_path,
        "chained_span_sha256": hashlib.sha256(chained_span.encode()).hexdigest(),
        "chained_span_is_pristine_helper": True,
    }
    return span + "\n" + formatted, audit


def _seeking_metrics(events: list[dict], task: dict, ledger: dict, stream: str) -> dict:
    """The seeking co-primary and its components, each reported separately.

    Every counter exists in a `_before_first_edit` and an at-any-point form, because the two
    answer different questions and conflating them biases the verdict:
      * `_before_first_edit` is the strict reading of "sought the information it was missing".
      * at-any-point is what the CATEGORY assignment uses, since a model that blind-edits, sees
        the test fail and THEN goes and looks has not blanket-suppressed.
    Paths are normalised: `RealRepoEnv.read_file` resolves through `_abspath`, so
    `<read path="./pkg/units/u.py"/>` succeeds while exact string matching records nothing --
    which would be a silent FALSE blanket signature, the failure direction that matters here.
    """
    target = task["target_path"]
    helper = task["helper"]
    norm = RR._norm

    def is_target_read(e):
        return e.get("type") == "read" and norm(e.get("path")) == norm(target)

    def is_target_grep(e):
        return e.get("type") == "grep" and any(
            norm(p) == norm(target) for p in (e.get("paths") or [])
        )

    def is_probe(e):
        return (e.get("type") in ("line_edit", "edit") and e.get("ok")
                and any(marker in (e.get("replace") or "") for marker in PROBE_MARKERS))

    first_edit = next((i for i, e in enumerate(events)
                       if e.get("type") in ("line_edit", "edit") and e.get("ok")), len(events))
    before = events[:first_edit]
    reads_before = [e for e in before if is_target_read(e)]
    greps_before = [e for e in before if is_target_grep(e)]
    reads_total = [e for e in events if is_target_read(e)]
    greps_total = [e for e in events if is_target_grep(e)]
    defn_blocked = [e for e in events if e.get("type") == "lsp_disabled"]
    defn_blocked_before = [e for e in before if e.get("type") == "lsp_disabled"]
    probe_edits = [e for e in events if is_probe(e)]
    # a probe edit is only evidence of seeking BEFORE the first edit if it happened there. The
    # first successful edit is excluded from `before` by construction, so a first edit that is
    # itself a probe is added back explicitly.
    probe_edits_before = [e for e in before if is_probe(e)]
    if first_edit < len(events) and is_probe(events[first_edit]):
        probe_edits_before = probe_edits_before + [events[first_edit]]
    ranged_covering = [
        m for m in RANGED_READ_RE.finditer(stream)
        if norm(m.group("path")) == norm(target)
        and int(m.group("a")) <= helper["return_line"] <= int(m.group("b"))
    ]
    records = ledger.get("records", [])
    delivering = [r for r in records if r["channel"] in DELIVERING_CHANNELS]
    hidden_line = helper["return_text"].strip()
    sought_before = bool(reads_before or greps_before or defn_blocked_before or probe_edits_before)
    return {
        "n_target_reads_before_first_edit": len(reads_before),
        "n_target_reads_total_normalised": len(reads_total),
        "n_target_greps_before_first_edit": len(greps_before),
        "n_target_greps_total": len(greps_total),
        "n_defn_blocked": len(defn_blocked),
        "n_defn_blocked_before_first_edit": len(defn_blocked_before),
        "n_probe_edits": len(probe_edits),
        "n_probe_edits_before_first_edit": len(probe_edits_before),
        "probe_edit": bool(probe_edits),
        "n_ranged_reads_covering_helper": len(ranged_covering),
        # The PRIMARY behavioural outcome: the model actually went and got the information, by a
        # read or a grep of the defining file. `sought_missing_information*` is broader (it also
        # counts a blocked <defn> and a runtime probe) and is reported as a secondary, because a
        # cell that only ever emits a refused <defn> and then guesses must not be read as
        # "the adapter seeks the missing information".
        "retrieved_missing_information": bool(reads_total or greps_total),
        "retrieved_missing_information_before_first_edit": bool(reads_before or greps_before),
        "sought_missing_information": bool(
            reads_total or greps_total or defn_blocked or probe_edits
        ),
        "sought_missing_information_before_first_edit": sought_before,
        # components, so an all-probe or all-blocked-defn cell is visible rather than absorbed
        "sought_component_read": bool(reads_before),
        "sought_component_grep": bool(greps_before),
        "sought_component_defn_blocked": bool(defn_blocked_before),
        "sought_component_probe": bool(probe_edits_before),
        "read_ledger": records,
        "read_ledger_channels": [r["channel"] for r in records],
        "first_target_read_channel": delivering[0]["channel"] if delivering else None,
        "hidden_line_in_context": hidden_line in stream,
        "gold_expression_in_context": helper["gold_expression"] in stream,
        "hidden_line_channel": delivering[0]["channel"] if delivering else None,
        "hidden_line_before_first_edit": any(
            r["n_edits_before"] == 0 and r["channel"] in ("model_read_whole", "model_read_ranged",
                                                          "grep_scan")
            for r in records
        ),
        "multiplier_literal_in_stream": str(helper["multiplier"]) in stream,
        "buggy_multiplier_literal_in_stream": str(helper["buggy_multiplier"]) in stream,
    }


def _generalization_pass(task: dict, variant: str = "typed") -> bool | None:
    """Post-hoc read-only probe: does the repaired repo match the GOLD function away from the two
    scored points? Catches an xor twin that satisfies the single held-out point by collision."""
    helper = task["helper"]
    values = [0, 1, task["input"] + 3, task["input"] + 11, task["input"] + 23, task["input"] + 50]
    expected = [RR.evaluate(task["template"], v, helper["multiplier"], helper["offset"])
                for v in values]
    repo = task["variants"][variant]["repo_dir"]
    RR.purge_pycache(repo)
    code = ("import json\nfrom pkg.app import execute\n"
            f"print(json.dumps([execute({task['token']!r}, v) for v in {values!r}]))\n")
    run = subprocess.run([sys.executable, "-c", code], cwd=repo, capture_output=True, text=True)
    if run.returncode:
        return False
    try:
        return json.loads(run.stdout) == expected
    except json.JSONDecodeError:
        return None


def _categorise(row: dict) -> str:
    """Pre-registered outcome categories (A-G), most specific first.

    A and D both use retrieval at ANY point, not only before the first edit, and for the same
    reason: a model that blind-edits, sees the test fail and THEN opens the file (or greps) has
    not blanket-suppressed. Scoping A to any-point but D to before-the-first-edit would classify
    the same trajectory as A with a read and as E with a grep.
    """
    passed = bool(row["repair_correct"])
    if row["n_target_reads_total"] or row["n_target_reads_total_normalised"]:
        return "A_read_then_repair" if passed else "A_read_then_fail"
    if row["n_target_greps_total"]:
        return "D_grep_then_repair" if passed else "D_grep_then_fail"
    if row["n_probe_edits"]:
        return "F_probe_then_repair" if passed else "F_probe_then_fail"
    if row["hidden_line_channel"] == "post_edit_view" and row["hidden_line_in_context"]:
        # E is a RESIDUAL-LEAK detector, not the designed channel: the redactor elides the one
        # informative line from the post-edit view, so a row must not be binned here merely for
        # having received a (redacted) view of the file it edited. It lands here only if the line
        # actually reached the stream through that channel.
        return "E_post_edit_view_then_repair" if passed else "E_post_edit_view_then_fail"
    if row["n_defn_blocked"]:
        return "G_defn_blocked_then_no_read"
    return "C_no_seeking_pass" if passed else "B_no_seeking_fail"


# ---------------------------------------------------------------------------
# floor / validity
# ---------------------------------------------------------------------------
def _floor_failure(rows: list[dict], split: str, adapter: str | None) -> str | None:
    if split != RR.PILOT_SPLIT or adapter is not None:
        return None
    cell = [r for r in rows if r["arm"] == "push_insufficient"]
    if not cell:
        return None
    n_pass = sum(1 for r in cell if r["repair_correct"])
    n_read = sum(1 for r in cell if r["retrieved_missing_information"])
    if len(cell) < RR.N_PILOT or n_pass < PILOT_FLOOR or n_read < PILOT_FLOOR:
        return (f"pilot floor failed on push_insufficient: repair_correct={n_pass}/{len(cell)}, "
                f"retrieved_missing_information={n_read}/{len(cell)} (need >= {PILOT_FLOOR}/"
                f"{RR.N_PILOT} on both). The INSTANCES are the problem, not the policy: revise "
                "before spending the 12-seed split.")
    return None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--revision", default=BASE_REVISION)
    parser.add_argument("--adapter", default=None,
                        help="LoRA adapter, e.g. runs/sft/substitution_lora_27b")
    parser.add_argument("--split", choices=RR.SPLITS, default=RR.PILOT_SPLIT)
    parser.add_argument("--arms", default="push_insufficient,push_sufficient,push_chained")
    parser.add_argument("--names", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new", type=int, default=1000)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-reads", type=int, default=12)
    parser.add_argument("--gpu-only", action="store_true")
    parser.add_argument("--tmp-root", default=None)
    parser.add_argument("--validation", default="runs/protocol/navigation_v2_readreq_validation.json")
    args = parser.parse_args()

    out_path = Path(args.out)
    if out_path.exists():
        print(f"refusing to overwrite existing result: {out_path}", file=sys.stderr)
        return 73

    arms = args.arms.split(",")
    unknown = [arm for arm in arms if arm not in ARM_SPECS]
    if unknown:
        raise ValueError(f"unknown arm(s): {unknown}")

    # the split must have been mechanically validated first, at this exact protocol hash
    validation_path = ROOT / args.validation
    if not validation_path.exists():
        raise SystemExit(f"missing readreq validation artifact: {validation_path}")
    validation = json.loads(validation_path.read_text())
    if not validation.get("passed") or validation.get("split") != args.split:
        raise SystemExit(f"validation artifact did not pass for split {args.split}")
    for rel, expected in validation.get("protocol_source_sha256", {}).items():
        actual = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"frozen protocol hash mismatch: {rel}")
    # the readreq sources (this driver included) must be the ones the validation was produced
    # against -- otherwise the gates that certified the instances no longer describe this code.
    for rel, expected in validation.get("readreq_source_sha256", {}).items():
        actual = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(
                f"readreq source changed since validation: {rel}\n"
                f"  validated {expected}\n  on disk   {actual}\n"
                "Re-run: python scripts/experiments/navigation_readreq_tasks.py validate"
            )
    RR.assert_frozen_restored()

    root = Path(args.tmp_root or Path(tempfile.gettempdir()) / "streams_navigation_readreq_runs")
    flavours = {ARM_SPECS[arm]["flavour"] for arm in arms}
    built = {flavour: RR.build_readreq_tasks(root, args.split, flavour) for flavour in flavours}
    if args.names:
        wanted = set(args.names.split(","))
        built = {f: [t for t in tasks if t["name"] in wanted] for f, tasks in built.items()}
    if not any(built.values()):
        raise ValueError("no readreq tasks selected")
    by_name = {flavour: {task["name"]: task for task in tasks} for flavour, tasks in built.items()}
    order = [task["name"] for task in built[sorted(flavours)[0]]]

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
        meta_path = Path(args.adapter) / "streams_train_meta.json"
        adapter_meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
        trained_on = (adapter_meta or {}).get("config", {}).get("revision")
        if trained_on and trained_on != args.revision:
            raise SystemExit(
                f"adapter was trained on base revision {trained_on} but this run resolves "
                f"{args.revision}. Putting the LoRA on different base weights voids the C33/C35 "
                "comparison; pass --revision explicitly if this is intended."
            )
        model = PeftModel.from_pretrained(model, args.adapter)
    model = model.eval()
    model_meta = {
        "adapter": args.adapter,
        "adapter_meta": adapter_meta,
        "condition": "trained" if args.adapter else "untrained",
        "revision": getattr(model.config, "_commit_hash", None) or args.revision,
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
        "dtype": str(model.dtype),
    }

    pyrefly = find_pyrefly()
    pyrefly_version = subprocess.run(
        [pyrefly, "--version"], capture_output=True, text=True
    ).stdout.strip()
    deviation_sha = {
        name: hashlib.sha256(inspect.getsource(getattr(RR, name)).encode()).hexdigest()
        for name in ("install_heldout_guard", "install_post_edit_redactor", "install_test_purge",
                     "install_read_ledger")
    }

    rows: list[dict] = []

    def flush():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "protocol": NT.PROTOCOL_VERSION,
            "experiment": RR.EXPERIMENT,
            "claim": "C36",
            "model": args.model,
            "model_meta": model_meta,
            "config": vars(args),
            "arms": arms,
            "split": args.split,
            "seeds": list(RR.SPLIT_SEEDS[args.split]),
            "templates": list(RR.TEMPLATES),
            "apparatus_deviation_sha256": deviation_sha,
            "apparatus_deviation_note": (
                "Installed identically in every cell and both model conditions: model-issued "
                "reads of held-out paths are denied (n_heldout_read_attempts per row); the "
                "INVOLUNTARY post-edit file view has the one informative line elided, with "
                "deliberate <read>/<read lines>/<grep> untouched (n_post_edit_redactions per "
                "row); __pycache__ is purged before every test the model runs "
                "(n_model_test_purges per row) as well as before scoring. Full rationale and "
                "per-instance verification are in the validation artifact's "
                "`apparatus_deviations`."
            ),
            "validation_artifact": str(validation_path.relative_to(ROOT)),
            "protocol_source_sha256": NT._protocol_hashes(),
            "pyrefly": {"path": pyrefly, "version": pyrefly_version},
            "rows": rows,
        }, indent=2) + "\n", encoding="utf-8")

    for name in order:
        gates: dict[str, dict] = {}
        spans: dict[str, tuple[str, str]] = {}
        for flavour in sorted(flavours):
            task = by_name[flavour][name]
            span, path, checks = _shared_span_and_verify(task)
            gates[flavour] = checks
            spans[flavour] = (span, path)
            print(f"{name}/{flavour} span usability: "
                  f"{'OK' if all(checks.values()) else 'FAIL ' + str(checks)} "
                  f"sha={hashlib.sha256(span.encode()).hexdigest()[:12]}", flush=True)

        for arm in arms:
            spec = ARM_SPECS[arm]
            flavour = spec["flavour"]
            task = by_name[flavour][name]
            if not all(gates[flavour].values()):
                raise RuntimeError(f"span usability gate failed for {name}/{flavour}: "
                                   f"{gates[flavour]}")
            repo = task["variants"]["typed"]["repo_dir"]
            RR.purge_pycache(repo)
            env = NT.make_env(task, "typed")
            try:
                supplied, supplied_path = spans[flavour]
                chained_audit: dict = {}
                if spec["chained"]:
                    supplied, chained_audit = _chained_payload(task, env, supplied)
                if env.lsp_errors:
                    raise RuntimeError(f"automatic semantic query failed: {env.lsp_errors}")
                if task["gold"]["new_text"] in supplied:
                    raise RuntimeError("semantic context contains the gold replacement")
                prompt = NT.build_prompt(task, "typed") + spec["preamble"].format(payload=supplied)

                # Three apparatus deviations, installed in this order in EVERY arm and BOTH model
                # conditions, and disclosed in the validation artifact:
                #   guard    -- denies model-issued reads of the held-out scoring oracle
                #   ledger   -- records reads of the defining file with their calling frames
                #   redactor -- elides the one informative line from the INVOLUNTARY post-edit
                #               file view only; deliberate reads/greps are untouched
                #   purge    -- purges __pycache__ before every test the model itself runs
                guard = RR.install_heldout_guard(env)
                ledger = RR.install_read_ledger(env, task["target_path"])
                redactor = RR.install_post_edit_redactor(
                    env, task["target_path"], task["helper"]["return_line"],
                    task["helper"]["return_text"],
                )
                purge = RR.install_test_purge(env, repo)
                agent = StreamAgent(
                    model, tokenizer, env, edit_mode="line", sys_override=spec["system"],
                    max_new_tokens=args.max_new, max_turns=args.max_turns,
                    max_reads=args.max_reads, temperature=args.temperature, seed=args.seed,
                    use_lsp_defn=False, lsp_disabled=True, lsp_fallback=False,
                )
                RR.purge_pycache(repo)
                started = time.perf_counter()
                result = agent.run(prompt, "pkg/app.py", editable=task["editable"])
                elapsed = time.perf_counter() - started

                held_unpurged = bool(NT.run_heldout(task, "typed"))
                RR.purge_pycache(repo)
                held_out_pass = bool(NT.run_heldout(task, "typed"))
                amb = RR.ambiguity_set(task, task["readreq"]["multipliers"])
                reread = _reread_metrics(result["events"], task["target_path"], True)
                seeking = _seeking_metrics(result["events"], task, ledger, result["stream"])
                row = {
                    "task": task["name"], "family": task["seed"], "split": args.split,
                    "variant": "typed", "arm": arm, "seed": args.seed,
                    "flavour": flavour, "template": task["template"],
                    "condition": model_meta["condition"], "adapter": args.adapter,
                    "resolved": held_out_pass, "visible_pass": bool(result["resolved"]),
                    "held_out_pass": held_out_pass,
                    # A workspace that fails the visible test is not a repair. Scoring the
                    # conjunction closes a measured hazard: in the xor family a blind
                    # `helper(value) + c` exists that hits the held-out point while failing the
                    # visible one (2/12 instances of the main split, found by live sweep), so raw
                    # held_out_pass can record a spurious pass if a rollout ends on it. Raw
                    # held_out_pass is still published, unchanged, for comparability with C33/C35.
                    "repair_correct": bool(held_out_pass and result["resolved"]),
                    "held_out_pass_unpurged": held_unpurged,
                    "stale_bytecode_suspected": held_out_pass != held_unpurged,
                    "bailed": result.get("bailed"),
                    "in_tokens": result["in_tokens"], "out_tokens": result["out_tokens"],
                    "turns": result["turns"], "n_reads": result["n_reads"],
                    "n_lsp": result["n_lsp"], "n_tests": result["n_tests"],
                    "n_edits": result["n_edits"], "wall_sec": round(elapsed, 3),
                    "server_latency_ms": round(sum(env.lsp_latencies) * 1000, 1),
                    "server_errors": list(env.lsp_errors),
                    "semantic_supplied_path": supplied_path,
                    "semantic_payload_sha256": hashlib.sha256(supplied.encode()).hexdigest(),
                    "span_usability_check": gates[flavour],
                    "helper": task["helper"],
                    "ambiguity_n": amb["n_members"],
                    "ambiguity_n_distinct_heldout": amb["n_distinct_heldout"],
                    "blind_guess_ceiling": amb["blind_guess_ceiling"],
                    "n_heldout_read_attempts": guard["n_attempts"],
                    "heldout_read_attempts": guard["attempts"],
                    "n_post_edit_redactions": redactor["n_redactions"],
                    "n_model_test_purges": purge["n_purges"],
                    **chained_audit,
                    **reread,
                    **seeking,
                    **_metrics(task, result["events"], supplied_path),
                    "generalization_pass": _generalization_pass(task),
                    "events": result["events"], "stream_tail": result["stream"][-2500:],
                }
                row["outcome_category"] = _categorise(row)
                rows.append(row)
                print(f"{task['name']} {arm} s{args.seed}: repair={row['repair_correct']} "
                      f"held={held_out_pass} visible={row['visible_pass']} "
                      f"read_after_span={row['read_after_span']} "
                      f"retrieved={row['retrieved_missing_information']} "
                      f"sought={row['sought_missing_information_before_first_edit']} "
                      f"cat={row['outcome_category']} "
                      f"heldout_attempts={row['n_heldout_read_attempts']} "
                      f"redactions={row['n_post_edit_redactions']} "
                      f"edits={row['n_edits']}", flush=True)
            finally:
                env.close()
            flush()

    print("\n=== per-arm summary ===", flush=True)
    for arm in arms:
        cell = [r for r in rows if r["arm"] == arm]
        if not cell:
            continue
        n = len(cell)
        print(f"  {arm:18s} n={n} "
              f"repair_correct={sum(1 for r in cell if r['repair_correct'])}/{n} "
              f"held_out_pass={sum(1 for r in cell if r['held_out_pass'])}/{n} "
              f"read_after_span={sum(1 for r in cell if r['read_after_span'])}/{n} "
              f"retrieved={sum(1 for r in cell if r['retrieved_missing_information'])}/{n} "
              f"sought={sum(1 for r in cell if r['sought_missing_information_before_first_edit'])}/{n} "
              f"probe={sum(1 for r in cell if r['probe_edit'])}/{n} "
              f"redactions={sum(r['n_post_edit_redactions'] for r in cell)} "
              f"stale_bytecode={sum(1 for r in cell if r['stale_bytecode_suspected'])}/{n}",
              flush=True)

    failure = _floor_failure(rows, args.split, args.adapter)
    if failure:
        print(failure, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
