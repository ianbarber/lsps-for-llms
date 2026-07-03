#!/usr/bin/env python3
"""Score real-repo predictions with the SWE-bench Docker oracle (the resolved/not verdict).

Wraps `swebench.harness.run_evaluation`. On ARM64 hosts (e.g. the DGX Spark GB10) the prebuilt
x86_64 images run under qemu emulation; register it once with
    docker run --privileged --rm tonistiigi/binfmt --install amd64
and this script sets DOCKER_DEFAULT_PLATFORM=linux/amd64 so the harness runs emulated.

  # validate the oracle on a gold patch (should resolve):
  python scripts/realbench/score.py --preds runs/realbench/eval/gold_pred.json --run-id goldval
  # score a matrix condition's predictions:
  python scripts/realbench/score.py --preds runs/realbench/matrix/pred_claude-sonnet-45_D.json --run-id mD
"""
import os
import sys
import json
import glob
import argparse
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET = "princeton-nlp/SWE-bench_Verified"


def score(preds_path, run_id, workers=4, dataset=DATASET):
    """Run the swebench oracle on a predictions file; return {instance_id: resolved_bool}."""
    preds_raw = json.load(open(preds_path))
    # accept both the swebench dict form {iid: {...}} (mini_ablate) and a list (run_matrix)
    preds = list(preds_raw.values()) if isinstance(preds_raw, dict) else preds_raw
    ids = sorted({p["instance_id"] for p in preds})
    model = preds[0]["model_name_or_path"] if preds else "model"
    env = dict(os.environ, DOCKER_DEFAULT_PLATFORM="linux/amd64",
               HF_HUB_OFFLINE="0", HF_DATASETS_OFFLINE="0")
    cmd = [sys.executable, "-m", "swebench.harness.run_evaluation",
           "--dataset_name", dataset, "--predictions_path", os.path.abspath(preds_path),
           "--run_id", run_id, "--instance_ids", *ids,
           "--max_workers", str(workers), "--namespace", "swebench", "--cache_level", "instance"]
    print(f"[score] {len(ids)} instances, model={model}, run_id={run_id} (emulated amd64)", flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, timeout=60 * 60)
    # swebench writes <model>.<run_id>.json in CWD
    report = None
    for cand in (f"{model}.{run_id}.json", *glob.glob(os.path.join(ROOT, f"*.{run_id}.json"))):
        p = cand if os.path.isabs(cand) else os.path.join(ROOT, cand)
        if os.path.exists(p):
            report = json.load(open(p)); break
    if report is None:
        print("[score] WARNING: no report file found", flush=True); return {}
    resolved = set(report.get("resolved_ids", []))
    return {iid: (iid in resolved) for iid in ids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="swebench predictions json (from run_matrix pred_*.json)")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--merge-into", default=None,
                    help="a run_matrix rollouts.json to annotate with `resolved` per matching rollout")
    args = ap.parse_args()
    res = score(args.preds, args.run_id, args.workers)
    n = sum(res.values())
    print(f"\n=== resolved {n}/{len(res)} ===")
    for iid, ok in sorted(res.items()):
        print(f"  {'RESOLVED' if ok else 'unresolved':11} {iid}")
    if args.merge_into and os.path.exists(args.merge_into):
        rows = json.load(open(args.merge_into))
        tag = json.load(open(args.preds))[0]["model_name_or_path"]
        for r in rows:
            if f"{r['model'].split('/')[-1]}_{r['cond']}" == tag and r["instance_id"] in res:
                r["resolved"] = res[r["instance_id"]]
        json.dump(rows, open(args.merge_into, "w"), indent=2)
        print(f"merged resolved into {args.merge_into}")


if __name__ == "__main__":
    main()
