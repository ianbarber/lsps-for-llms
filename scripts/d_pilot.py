#!/usr/bin/env python3
"""Zero-shot A/C/D pilot: run the continuous-stream agent on Goldilocks SWE-bench
tasks under each feedback condition (A none / C sync-at-edit / D live-async) with
the BASE 7B (no SFT yet), measuring the in-flight metrics. First meaningful readout
of whether live feedback changes the trajectory.

reset() once per task (slow clone+install); cheap git-restore between conditions.
Usage: d_pilot.py [n_tasks] [max_new] [source]
"""
import os, sys, re, json, subprocess, time, traceback
os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")
sys.path.insert(0, "/home/ianbarber/Projects/Streams")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from harness.task_env import TaskEnv, load_instance
from scaffold.stream_agent import StreamAgent
from scaffold.mock_env import MockEnv  # noqa

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
MAXNEW = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
SRC = sys.argv[3] if len(sys.argv) > 3 else "verified"
CONDS = ("A", "C", "D")
OUTDIR = "runs/d_pilot"; os.makedirs(OUTDIR, exist_ok=True)

cand = json.load(open("runs/task_env/eval_candidates.json"))
tasks = (cand.get("tasks") or cand.get("candidates") or cand) if isinstance(cand, dict) else cand
tasks = tasks[:N]

print("[load model]", flush=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct",
        torch_dtype=torch.bfloat16, device_map="auto").eval()

def restore(env):
    subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=env.repo_dir, capture_output=True)
    subprocess.run(["git", "clean", "-fdq"], cwd=env.repo_dir, capture_output=True)
    env.metrics_state = type(env.metrics_state)()

def build_task(inst, env):
    files = re.findall(r'^\+\+\+ b/(.+)$', inst.get("patch", ""), re.M)
    files = [f for f in files if not f.split("/")[-1].startswith("test") and f.endswith(".py")] or files
    tgt = files[0] if files else None
    blob = ""
    if tgt:
        try:
            content = env.read_file(tgt)
            numbered = "\n".join(f"{i+1:4d}| {ln}" for i, ln in enumerate(content[:6500].splitlines()))
            blob = (f"\n\nThe bug is in `{tgt}`. Current content (line-numbered for reference; "
                    f"SEARCH text must NOT include the line numbers):\n{numbered}")
        except Exception:
            pass
    prompt = (f"Resolve this issue by editing the repository.\n\n## Problem\n"
              f"{inst.get('problem_statement','')[:2500]}{blob}")
    return prompt, tgt

results = []
for ti, t in enumerate(tasks):
    iid = t["instance_id"]
    print(f"\n########## [{ti+1}/{len(tasks)}] {iid} ##########", flush=True)
    try:
        inst = load_instance(iid, source=SRC)
        env = TaskEnv(inst)
        env.reset()
    except Exception as e:
        print(f"  [skip] reset failed: {e}", flush=True)
        results.append({"instance_id": iid, "error": str(e)[:200]})
        continue
    prompt, tgt = build_task(inst, env)
    if not tgt:
        print("  [skip] no target file", flush=True); env.close(); continue
    row = {"instance_id": iid, "target": tgt, "conds": {}}
    for cond in CONDS:
        restore(env)
        try:
            t0 = time.time()
            r = StreamAgent(model, tok, env, condition=cond, latency_tokens=12,
                            max_new_tokens=MAXNEW).run(prompt, tgt)
            dt = time.time() - t0
            ev = r["events"]
            row["conds"][cond] = {
                "resolved": r["resolved"],
                "rework_ratio": r["metrics"].get("rework_ratio"),
                "edit_cycles": r["metrics"].get("region_edit_cycles", r["metrics"].get("edit_error_cycles", 0)),
                "n_edits": sum(1 for e in ev if e["type"] == "edit"),
                "n_edits_applied": sum(1 for e in ev if e["type"] == "edit" and e.get("ok")),
                "n_diag": sum(1 for e in ev if e["type"].startswith("diag")),
                "tokens": r["n_tokens"], "secs": round(dt, 1)}
            print(f"  {cond}: resolved={r['resolved']} rework={row['conds'][cond]['rework_ratio']} "
                  f"edits={row['conds'][cond]['n_edits_applied']} diag={row['conds'][cond]['n_diag']} "
                  f"tok={r['n_tokens']} ({dt:.0f}s)", flush=True)
        except Exception as e:
            row["conds"][cond] = {"error": str(e)[:200]}
            print(f"  {cond}: ERROR {e}", flush=True); traceback.print_exc()
    results.append(row)
    env.close()
    json.dump(results, open(f"{OUTDIR}/results.json", "w"), indent=2)

# aggregate
print("\n========== PILOT SUMMARY ==========", flush=True)
agg = {c: {"resolved": 0, "rework": [], "cycles": [], "diag": [], "n": 0} for c in CONDS}
for row in results:
    for c in CONDS:
        cd = row.get("conds", {}).get(c)
        if not cd or "error" in cd: continue
        agg[c]["n"] += 1; agg[c]["resolved"] += int(bool(cd["resolved"]))
        if cd["rework_ratio"] is not None: agg[c]["rework"].append(cd["rework_ratio"])
        agg[c]["cycles"].append(cd["edit_cycles"]); agg[c]["diag"].append(cd["n_diag"])
for c in CONDS:
    a = agg[c]; n = max(a["n"], 1)
    mr = sum(a["rework"])/max(len(a["rework"]),1); mc = sum(a["cycles"])/max(len(a["cycles"]),1)
    md = sum(a["diag"])/max(len(a["diag"]),1)
    print(f"  {c}: resolve={a['resolved']}/{a['n']}  mean_rework={mr:.3f}  mean_cycles={mc:.2f}  mean_diag_injections={md:.2f}", flush=True)
json.dump({"results": results, "agg": {c: {k:(v if not isinstance(v,list) else round(sum(v)/max(len(v),1),3)) for k,v in agg[c].items()} for c in CONDS}},
          open(f"{OUTDIR}/summary.json", "w"), indent=2)
print(f"\n[done] -> {OUTDIR}/", flush=True)
