#!/usr/bin/env python3
"""Select contamination-controlled, 7B-tractable SWE-rebench tasks.

Filters the nebius/SWE-rebench test split to:
  - created_at year >= MIN_YEAR (after Qwen2.5-Coder-7B's data cutoff -> not in training)
  - gold patch touches exactly ONE non-test source file, < MAX_PATCH chars
  - 1..MAX_F2P fail-to-pass tests, >=1 pass-to-pass (so we can verify provisioning)
  - has a problem_statement
Writes the full instance dicts (all SWE-bench fields, for TaskEnv) of the smallest
candidates to runs/rebench/candidates.jsonl, sorted by patch size ascending.

Streaming + os._exit to dodge the HF streaming finalizer segfault on aarch64.
"""
import os, sys, json, itertools
os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")
from datasets import load_dataset

MIN_YEAR = int(os.environ.get("MIN_YEAR", "2025"))
MAX_PATCH = int(os.environ.get("MAX_PATCH", "1200"))
MAX_F2P = int(os.environ.get("MAX_F2P", "3"))
WANT = int(os.environ.get("WANT", "25"))
SCAN = int(os.environ.get("SCAN", "12000"))

def src_files(patch):
    files = [l[len("diff --git a/"):].split(" b/")[0] for l in patch.splitlines()
             if l.startswith("diff --git ")]
    return [f for f in files if not ("/test" in f or f.startswith("test") or "test_" in os.path.basename(f))]

cands = []
ds = load_dataset("nebius/SWE-rebench", split="test", streaming=True)
for ex in itertools.islice(ds, SCAN):
    try:
        ca = str(ex.get("created_at", ""))
        yr = int(ca[:4]) if ca[:4].isdigit() else 0
        if yr < MIN_YEAR:
            continue
        patch = ex.get("patch", "") or ""
        sf = src_files(patch)
        f2p = ex.get("FAIL_TO_PASS", []) or []
        p2p = ex.get("PASS_TO_PASS", []) or []
        if (len(sf) == 1 and 0 < len(patch) < MAX_PATCH
                and 1 <= len(f2p) <= MAX_F2P and len(p2p) >= 1
                and (ex.get("problem_statement") or "").strip()):
            cands.append((len(patch), dict(ex), sf[0]))
    except Exception:
        continue

cands.sort(key=lambda x: x[0])
os.makedirs("runs/rebench", exist_ok=True)
with open("runs/rebench/candidates.jsonl", "w") as f:
    for plen, ex, sf in cands[:WANT]:
        f.write(json.dumps({"instance": ex, "src_file": sf, "patch_len": plen}) + "\n")

print(f"scanned<= {SCAN}  matched={len(cands)}  wrote={min(len(cands),WANT)} "
      f"(year>={MIN_YEAR}, 1 src file, patch<{MAX_PATCH}, f2p<={MAX_F2P})")
for plen, ex, sf in cands[:WANT]:
    print(f"  {ex['instance_id']:50} {str(ex.get('created_at'))[:10]} "
          f"plen={plen:5} f2p={len(ex.get('FAIL_TO_PASS',[]))} src={sf}")
sys.stdout.flush()
os._exit(0)
