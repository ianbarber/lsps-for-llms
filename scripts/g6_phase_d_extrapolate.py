#!/usr/bin/env python3
"""Phase D L4 extrapolation from measured throughput-by-context."""
import json
from pathlib import Path

OUT = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_d")
tp = json.loads((OUT/"throughput_by_context.json").read_text())
byc = tp["by_context"]

# L4 token accounting (v0.3): tokens per (task,seed,condition) ~ 8000 productive.
TOK_PER_RUN = 8000
FULL = {"tasks":200,"seeds":9,"conditions":5}
DESC = {"tasks":50,"seeds":6,"conditions":5}

# Use the 8192-context tok/s as the realistic regime (8000-token trajectories
# spend most of their length at high context). Also report the optimistic 256.
tps_8192 = byc["8192"]["derived_multi_tok_s"]
tps_256  = byc["256"]["derived_multi_tok_s"]
# Average across the curve weighted toward the long-context end (trajectory spends
# more rows at high context): simple mean of the 4 points as a midpoint estimate.
tps_mean = sum(byc[k]["derived_multi_tok_s"] for k in byc)/len(byc)

def weeks(scope, tps):
    runs = scope["tasks"]*scope["seeds"]*scope["conditions"]
    total_tok = runs*TOK_PER_RUN
    sec = total_tok/tps
    return sec/3600/24/7, runs

res={"tok_per_run":TOK_PER_RUN,
     "tps_used":{"at_8192":tps_8192,"at_256":tps_256,"curve_mean":tps_mean},
     "scenarios":{}}
for name,scope in (("full_200x9",FULL),("descoped_50x6",DESC)):
    res["scenarios"][name]={}
    for tlabel,tps in (("at_8192",tps_8192),("at_256",tps_256),("curve_mean",tps_mean)):
        w,runs=weeks(scope,tps)
        res["scenarios"][name][tlabel]={"weeks":round(w,1),"runs":runs}
(OUT/"extrapolation.json").write_text(json.dumps(res,indent=2))
print(json.dumps(res,indent=2))
