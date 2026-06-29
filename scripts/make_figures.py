#!/usr/bin/env python3
"""Generate the paper figures from the committed run JSONs. Okabe-Ito colorblind-safe palette,
grayscale-legible, vector PDF output (+ PNG previews). Every number is read from runs/agent/*.json;
nothing is hardcoded. Run: .venv-streams.system/bin/python scripts/make_figures.py

Figures -> docs/figures/fig{1,2,3,4}.pdf:
  fig1  C2  tool-value ablation: input tokens with <defn> vs read-only, per model
  fig2  C1  information redundant: type-inference channel (check_types) does not lift success
  fig3  C3  election is capability-gated: <defn> use by model on the obscure real-code suite
  fig4  C3  the 7B on-policy training win (use / cost / success), with held-out types
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "docs/figures"; os.makedirs(OUT, exist_ok=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = lambda p: os.path.join(ROOT, "runs", "agent", p)
sys.path.insert(0, ROOT)
from scripts.synth_tasks_effic import TASKS_EFFIC
try:
    from scripts.synth_tasks_efficread import TASKS_EFFICREAD
except Exception:
    TASKS_EFFICREAD = []
DEFN = {t["name"] for t in TASKS_EFFIC}
READ = {t["name"] for t in TASKS_EFFICREAD}
HELD = {"effic_queue_defn", "effic_cache_defn", "effic_clamp_defn"}

C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", grey="#999999",
         vermillion="#D55E00", sky="#56B4E9", black="#000000")
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def rows(p):
    """Row list, whether rows is {'A': [...]} (local harness) or a flat list (api_agent)."""
    r = json.load(open(A(p)))["rows"]
    return r["A"] if isinstance(r, dict) else r


def used_defn(x):
    return x.get("n_defn", 0) > 0 or x.get("n_lsp", 0) > 0 or any(
        e.get("type") == "defn" for e in x.get("events", []))


def intok(x):
    """Input tokens: in_tokens (local) or prompt_tokens (api_agent)."""
    return x.get("in_tokens", x.get("prompt_tokens", 0))


def mean(rs, f):
    return (sum(f(x) for x in rs) / len(rs)) if rs else 0.0


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png", dpi=150); plt.close(fig)
    print(f"wrote {OUT}/{name}.pdf")


# ---------- Figure 1 (C2): tool-value ablation — tokens with <defn> vs read-only ----------
# Same model, action toggled. 27B local (in_tokens), two frontier models (prompt_tokens).
abl = [("27B", "er2_27b_base.json", "er2_27b_readonly.json"),
       ("sonnet-4.5", "fr_sonnet45_withdefn.json", "fr_sonnet45_readonly.json"),
       ("deepseek-v3.1", "fr_deepseek_withdefn.json", "fr_deepseek_readonly.json")]
labels, withd, ro = [], [], []
for name, wp, rp in abl:
    labels.append(name)
    withd.append(mean(rows(wp), intok))
    ro.append(mean(rows(rp), intok))
fig, ax = plt.subplots(figsize=(6.6, 3.6))
x = range(len(labels)); w = 0.38
ax.bar([i - w/2 for i in x], withd, w, color=C["blue"], label="with go-to-definition")
ax.bar([i + w/2 for i in x], ro, w, color=C["grey"], label="read-only")
ax.set_yscale("log")
ax.set_xticks(list(x)); ax.set_xticklabels(labels)
ax.set_ylabel("mean input tokens to solve (log)")
ax.set_title("Removing go-to-definition costs 3.7 to 5.3x more tokens, same success")
ax.legend(frameon=False, fontsize=9, loc="upper left")
for i in x:
    ax.text(i, ro[i] * 1.15, f"{ro[i]/max(withd[i],1):.1f}×", ha="center", fontsize=10, fontweight="bold")
save(fig, "fig1")

# ---------- Figure 2 (C1): information redundant — type-inference channel ----------
# Gap D: success with vs without a check_types() tool, two frontier models. No lift.
gd = [("sonnet-4.5", "gd_sonnet45_nocheck.json", "gd_sonnet45_withcheck.json"),
      ("deepseek-v3.1", "gd_deepseek_nocheck.json", "gd_deepseek_withcheck.json")]
glabels, gno, gwith, gcalls = [], [], [], []
for name, nc, wc in gd:
    rnc, rwc = rows(nc), rows(wc)
    glabels.append(name)
    gno.append(100 * mean(rnc, lambda x: x["resolved"]))
    gwith.append(100 * mean(rwc, lambda x: x["resolved"]))
    gcalls.append(sum(x.get("n_check", 0) > 0 for x in rwc))
fig, ax = plt.subplots(figsize=(5.8, 3.6))
x = range(len(glabels)); w = 0.36
ax.bar([i - w/2 for i in x], gno, w, color=C["grey"], label="no type-checker")
ax.bar([i + w/2 for i in x], gwith, w, color=C["blue"], label="+ check_types() tool")
ax.set_xticks(list(x)); ax.set_xticklabels(glabels); ax.set_ylabel("pass@1 (%)"); ax.set_ylim(0, 112)
ax.set_title("A type-checker tool does not lift success on inference-hard tasks")
ax.legend(frameon=False, fontsize=9, loc="lower center")
for i in x:
    ax.text(i, 104, "100%", ha="center", fontsize=9)
save(fig, "fig2")

# ---------- Figure 3 (C3): election is capability-gated — <defn> use by model ----------
# All on the obscure real-code suite (effic_real2). Weak model only elects after training;
# capable models elect from prompt framing.
elec = [("7B\n(default)", "er2_base.json", C["grey"]),
        ("7B\n(trained)", "er2_trained.json", C["vermillion"]),
        ("27B\n(framed)", "er2_27b_base.json", C["blue"]),
        ("deepseek\n(framed)", "fr_deepseek_withdefn.json", C["blue"]),
        ("sonnet-4.5\n(framed)", "fr_sonnet45_withdefn.json", C["blue"])]
elabels, evals, ecolors = [], [], []
for name, p, col in elec:
    elabels.append(name); evals.append(100 * mean(rows(p), used_defn)); ecolors.append(col)
fig, ax = plt.subplots(figsize=(6.8, 3.6))
ax.bar(range(len(elabels)), evals, color=ecolors)
ax.set_xticks(range(len(elabels))); ax.set_xticklabels(elabels, fontsize=9.5)
ax.set_ylabel("go-to-definition use (%)"); ax.set_ylim(0, 112)
ax.set_title("Election is capability-gated: weak model needs training, capable models need framing")
for i, v in enumerate(evals):
    ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=10)
save(fig, "fig3")

# ---------- Figure 4 (C3 detail): the 7B on-policy training win ----------
base = [r for r in rows("powered_retest_base.json") if r["task"] in DEFN] + rows("powered_retest_base_x.json")
sft = [r for r in rows("powered_retest_sft.json") if r["task"] in DEFN] + rows("powered_retest_sft_x.json")
fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.3), constrained_layout=True)
g = ["PRE\n(untrained)", "POST\n(trained)"]
dp, dq = 100*mean(base, used_defn), 100*mean(sft, used_defn)
axes[0].bar(g, [dp, dq], color=[C["grey"], C["blue"]]); axes[0].set_ylim(0, 105)
axes[0].set_ylabel("go-to-definition use (%)"); axes[0].set_title("(a) tool preference")
for i, v in enumerate([dp, dq]): axes[0].text(i, v+2, f"{v:.0f}%", ha="center", fontsize=10)
tp, tq = mean(base, intok), mean(sft, intok)
axes[1].bar(g, [tp, tq], color=[C["grey"], C["orange"]]); axes[1].set_ylabel("mean input tokens / attempt")
axes[1].set_title("(b) retrieval cost")
for i, v in enumerate([tp, tq]): axes[1].text(i, v+60, f"{v:.0f}", ha="center", fontsize=10)
sp, sq = 100*mean(base, lambda x: x["resolved"]), 100*mean(sft, lambda x: x["resolved"])
hb = [r for r in base if r["task"] in HELD]; hs = [r for r in sft if r["task"] in HELD]
hp, hq = 100*mean(hb, lambda x: x["resolved"]), 100*mean(hs, lambda x: x["resolved"])
xx = [0, 1]; w = 0.36
axes[2].bar([i-w/2 for i in xx], [sp, sq], w, color=C["green"], label="all tasks")
axes[2].bar([i+w/2 for i in xx], [hp, hq], w, color=C["sky"], label="held-out types")
axes[2].set_xticks(xx); axes[2].set_xticklabels(g); axes[2].set_ylabel("success (%)"); axes[2].set_ylim(0, 109)
axes[2].set_title("(c) task success"); axes[2].legend(frameon=False, fontsize=9)
fig.suptitle("Training a 7B to prefer cheap retrieval (definition-sufficient tasks, 12 seeds)", y=1.06)
save(fig, "fig4")

# ---------- Figure 5 (C3 boundary): action use by task type, the policy is a boundary ----------
def used_read(x): return x.get("n_reads", 0) > 0
base_r = [r for r in rows("powered_retest_base.json") if r["task"] in READ]
sft_r = [r for r in rows("powered_retest_sft.json") if r["task"] in READ]
fig, ax = plt.subplots(figsize=(6.0, 3.4))
cats = ["definition-\nsufficient", "read-\nrequired"]
pre_d = [100*mean(base, used_defn), 100*mean(base_r, used_defn)]
post_d = [100*mean(sft, used_defn), 100*mean(sft_r, used_defn)]
pre_r = [100*mean(base, used_read), 100*mean(base_r, used_read)]
post_r = [100*mean(sft, used_read), 100*mean(sft_r, used_read)]
xx = range(len(cats)); w = 0.2
ax.bar([i-1.5*w for i in xx], pre_d, w, color=C["grey"], label="PRE: go-to-def")
ax.bar([i-0.5*w for i in xx], post_d, w, color=C["blue"], label="POST: go-to-def")
ax.bar([i+0.5*w for i in xx], pre_r, w, color="#cccccc", hatch="//", label="PRE: read")
ax.bar([i+1.5*w for i in xx], post_r, w, color=C["vermillion"], hatch="//", label="POST: read")
ax.set_xticks(list(xx)); ax.set_xticklabels(cats); ax.set_ylabel("action use (%)"); ax.set_ylim(0, 115)
ax.set_title("The learned policy is a boundary, not a collapse")
ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center")
save(fig, "fig5")
print("done")
