#!/usr/bin/env python3
"""Provision SWE-rebench candidates natively on this box and verify the baseline:
a well-formed task installs, its PASS_TO_PASS pass, and its FAIL_TO_PASS FAIL
(the bug is present) before any agent edit. Collects the ones that provision
cleanly into runs/rebench/provisioned.jsonl as the eval set.

Usage: rebench_smoke.py [--want N] [--ids id1,id2,...] [--max-p2p K]
"""
import os, sys, json, argparse, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness.task_env import TaskEnv

ap = argparse.ArgumentParser()
ap.add_argument("--want", type=int, default=4)
ap.add_argument("--ids", default=None, help="comma list to try in order; else a pure-python-first default")
ap.add_argument("--max-p2p", type=int, default=8)
A = ap.parse_args()

cands = {}
for line in open("runs/rebench/candidates.jsonl"):
    r = json.loads(line); cands[r["instance"]["instance_id"]] = r

# pure-python / lightweight first (most likely to build fast on aarch64)
DEFAULT_ORDER = [
    # lightweight pure-python libs first (most likely small-file + fast install)
    "fabfuel__circuitbreaker-67", "konradhalas__dacite-274", "gaogaotiantian__coredumpy-61",
    "ASPP__pelita-875", "ASPP__pelita-863", "Davidyz__VectorCode-27", "CrossGL__crosstl-257",
    "iris-hep__func_adl-185", "koxudaxi__datamodel-code-generator-2389",
    "getsentry__sentry-python-3942", "PennLINC__CuBIDS-438", "iterative__dvc-10711",
    "brightway-lca__brightway2-data-235", "marrink-lab__vermouth-martinize-653",
    "Akkudoktor-EOS__EOS-459", "AzureAD__microsoft-authentication-library-for-python-795",
    "konradhalas__dacite-274", "lincc-frameworks__nested-pandas-236",
]
order = (A.ids.split(",") if A.ids else DEFAULT_ORDER)
order = [i for i in order if i in cands] + [i for i in cands if i not in order]

clean = []
results = []
for iid in order:
    if len(clean) >= A.want:
        break
    r = cands[iid]; inst = r["instance"]
    rec = {"instance_id": iid, "repo": inst["repo"], "src_file": r["src_file"]}
    print(f"\n=== {iid} ({inst['repo']}) ===", flush=True)
    try:
        env = TaskEnv(inst)
        state = env.reset(install=True)
        rec["install_ok"] = state.get("install_ok")
        print(f"  install_ok={state.get('install_ok')} f2p={state['fail_to_pass']} "
              f"p2p={state['pass_to_pass']}", flush=True)
        if not state.get("install_ok"):
            rec["status"] = "install_failed"; rec["log"] = env._install_log[-1200:]
            print("  INSTALL FAILED:\n" + env._install_log[-800:], flush=True)
            results.append(rec); env.close(); continue
        try:
            rec["src_lines"] = len(env.read_file(r["src_file"]).splitlines())
        except Exception:
            rec["src_lines"] = None
        tr = env.run_tests(max_p2p=A.max_p2p)
        rec.update({"baseline_resolved": tr.get("resolved"),
                    "f2p_pass": tr.get("f2p_pass"), "f2p_total": tr.get("f2p_total"),
                    "p2p_pass": tr.get("p2p_pass"), "p2p_total": tr.get("p2p_total")})
        well_formed = (tr.get("f2p_pass", 0) == 0 and tr.get("f2p_total", 0) > 0
                       and tr.get("p2p_total", 0) > 0 and tr.get("p2p_pass", 0) == tr.get("p2p_total", 0))
        rec["well_formed"] = well_formed
        print(f"  baseline: resolved={tr.get('resolved')} "
              f"F2P {tr.get('f2p_pass')}/{tr.get('f2p_total')} "
              f"P2P {tr.get('p2p_pass')}/{tr.get('p2p_total')}  well_formed={well_formed}", flush=True)
        if well_formed:
            rec["status"] = "clean"; r["src_lines"] = rec["src_lines"]; clean.append(r)
        else:
            rec["status"] = "ill_formed"
            for k in ("f2p_summary", "p2p_summary", "failure"):
                if tr.get(k): print(f"  {k}: {str(tr[k])[:500]}", flush=True)
        env.close()
    except Exception as e:
        rec["status"] = "exception"; rec["error"] = f"{type(e).__name__}: {e}"
        print(f"  EXCEPTION: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
    results.append(rec)

clean.sort(key=lambda r: (r.get("src_lines") or 10**9))   # smallest files first = easier tier
with open("runs/rebench/provisioned.jsonl", "w") as f:
    for r in clean:
        f.write(json.dumps(r) + "\n")
json.dump(results, open("runs/rebench/provision_report.json", "w"), indent=2)
print(f"\n=== CLEAN: {len(clean)}/{A.want} wanted ===", flush=True)
for r in clean:
    print(f"  {r['instance']['instance_id']}", flush=True)
print(f"-> runs/rebench/provisioned.jsonl ({len(clean)}), report -> provision_report.json", flush=True)
