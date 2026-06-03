"""Select 10 SWE-Gym tasks for L0 G1 validation.

Filters:
- Typed-friendly subprojects: django, sympy, scikit-learn, requests, flask, pandas, sphinx, pytest.
- Gold patch touches <= 3 files.
- Gold patch <= 200 LOC (added + removed lines combined).
- Mix difficulty by gold patch size (easy <= 30 LOC, medium <= 100, hard <= 200).

Outputs runs/g1_prep/swegym_tasks.json with metadata.

Repo-size filter (<= 500 MB) deferred: requires cloning. SWE-Gym is curated SWE-bench-style
data, so all included repos are reasonable; we check size at clone time on the validation task.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from datasets import load_dataset

TYPED_REPOS = {
    # SWE-Gym repos that are reasonably typed Python with manageable test surfaces.
    # The plan listed django/sympy/scikit-learn/requests/flask as illustrative; SWE-Gym
    # does not include them. Substitutes (typed, small-to-medium, install-friendly):
    "pandas-dev/pandas",
    "pydantic/pydantic",
    "python/mypy",
    "dask/dask",
    "modin-project/modin",
    "facebookresearch/hydra",
    "bokeh/bokeh",
    "conan-io/conan",
}

OUT = Path("/home/ianbarber/Projects/Streams/runs/g1_prep/swegym_tasks.json")


def patch_stats(patch: str) -> tuple[int, int]:
    """Return (files_touched, gold_loc_added_plus_removed)."""
    if not patch:
        return 0, 0
    files = set(re.findall(r"^diff --git a/(\S+) b/", patch, re.MULTILINE))
    loc = sum(
        1 for line in patch.splitlines()
        if (line.startswith("+") or line.startswith("-"))
        and not line.startswith(("+++", "---"))
    )
    return len(files), loc


def difficulty_bucket(loc: int) -> str:
    if loc <= 30:
        return "easy"
    if loc <= 100:
        return "medium"
    return "hard"


def main() -> None:
    ds = load_dataset("SWE-Gym/SWE-Gym", split="train")
    print(f"Loaded {len(ds)} SWE-Gym rows")

    # Bucket candidates by difficulty
    buckets: dict[str, list[dict]] = {"easy": [], "medium": [], "hard": []}
    for ex in ds:
        if ex["repo"] not in TYPED_REPOS:
            continue
        files, loc = patch_stats(ex["patch"])
        if files == 0 or files > 3:
            continue
        if loc > 200:
            continue
        diff = difficulty_bucket(loc)
        buckets[diff].append({
            "instance_id": ex["instance_id"],
            "repo": ex["repo"],
            "base_commit": ex["base_commit"],
            "files_touched": files,
            "gold_loc": loc,
            "expected_difficulty": diff,
            "version": ex.get("version", ""),
            "fail_to_pass_count": len(ex.get("FAIL_TO_PASS") or []),
            "pass_to_pass_count": len(ex.get("PASS_TO_PASS") or []),
        })

    for k, v in buckets.items():
        print(f"  {k}: {len(v)} candidates")

    # Pick a mix: 3 easy, 4 medium, 3 hard.
    # Diversify across repos: interleave so any single repo doesn't dominate the 10.
    targets = {"easy": 3, "medium": 4, "hard": 3}
    selected: list[dict] = []
    global_repo_count: dict[str, int] = {}
    for diff, n in targets.items():
        pool = sorted(buckets[diff], key=lambda r: (r["repo"], r["instance_id"]))
        # Round-robin across repos within this bucket.
        by_repo: dict[str, list[dict]] = {}
        for c in pool:
            by_repo.setdefault(c["repo"], []).append(c)
        repos_in_pool = sorted(by_repo.keys(), key=lambda r: global_repo_count.get(r, 0))
        bucket_picked = 0
        while bucket_picked < n:
            progressed = False
            for r in list(repos_in_pool):
                # Global cap: no more than 3 from a single repo across the whole set.
                if global_repo_count.get(r, 0) >= 3:
                    continue
                if not by_repo[r]:
                    continue
                cand = by_repo[r].pop(0)
                selected.append(cand)
                global_repo_count[r] = global_repo_count.get(r, 0) + 1
                bucket_picked += 1
                progressed = True
                if bucket_picked >= n:
                    break
            if not progressed:
                break  # no more candidates available

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "selection_filters": {
            "typed_repos": sorted(TYPED_REPOS),
            "files_touched_max": 3,
            "gold_loc_max": 200,
            "difficulty_buckets": {
                "easy": "<= 30 LOC",
                "medium": "31-100 LOC",
                "hard": "101-200 LOC",
            },
            "repo_size_max_mb": 500,
            "repo_size_filter_status": "deferred to clone-time check",
        },
        "target_mix": targets,
        "n_selected": len(selected),
        "tasks": selected,
    }, indent=2))
    print(f"Wrote {len(selected)} tasks to {OUT}")
    for s in selected:
        print(f"  {s['expected_difficulty']:6s} {s['instance_id']:40s} files={s['files_touched']} loc={s['gold_loc']}")


if __name__ == "__main__":
    main()
