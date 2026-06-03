#!/usr/bin/env python3
"""Phase E L4 extrapolation from measured GQA-flex throughput-by-context."""
import json
from pathlib import Path

OUT = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_e")
tp = json.loads((OUT / "throughput_by_context.json").read_text())
byc = tp["by_context"]

TOK_PER_RUN = 8000
FULL = {"tasks": 200, "seeds": 9, "conditions": 5}
DESC = {"tasks": 50, "seeds": 6, "conditions": 5}


def tps(k):
    return byc[k]["mean_productive_tokens_per_sec"]


# Longest measured context = realistic regime. Use it if present, else max key.
ctx_keys = sorted(byc.keys(), key=lambda x: int(x))
long_key = ctx_keys[-1]
short_key = ctx_keys[0]
tps_long = tps(long_key)
tps_short = tps(short_key)
tps_mean = sum(tps(k) for k in byc) / len(byc)


def weeks(scope, t):
    runs = scope["tasks"] * scope["seeds"] * scope["conditions"]
    total_tok = runs * TOK_PER_RUN
    sec = total_tok / t
    return sec / 3600 / 24 / 7, runs


res = {"tok_per_run": TOK_PER_RUN,
       "ctx_keys": ctx_keys,
       "tps_used": {f"at_{long_key}": tps_long, f"at_{short_key}": tps_short,
                    "curve_mean": tps_mean},
       "scenarios": {}}
for name, scope in (("full_200x9", FULL), ("descoped_50x6", DESC)):
    res["scenarios"][name] = {}
    for tlabel, t in ((f"at_{long_key}", tps_long), (f"at_{short_key}", tps_short),
                      ("curve_mean", tps_mean)):
        w, runs = weeks(scope, t)
        res["scenarios"][name][tlabel] = {"weeks": round(w, 1), "runs": runs}
(OUT / "extrapolation.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
