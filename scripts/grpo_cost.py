#!/usr/bin/env python3
"""Cost-reward GRPO on harvested OPSD trajectories — instill the cheap-retrieval preference via RL.

Minimal group-relative policy gradient (GRPO-style) reusing the EXISTING rollout + LoRA infra:
  - Rollouts come from synth_mf.py --save-sft (each row carries sft_input_ids/sft_labels with the
    prompt+observations masked to -100, only the model's OWN action tokens un-masked — the same mask
    sft_lora.py trains on), plus `resolved` and `in_tokens`.
  - REWARD per trajectory:  r = resolved ? (1 - lambda * in_tokens_norm) : 0
    in_tokens_norm = clip(in_tokens / token_cap, 0, 1)  (solve-at-min-tokens: solving is necessary,
    fewer input tokens is better).
  - GROUP = the G rollouts that share the same `task`. ADVANTAGE: A = (r - mean_group)/(std_group + 1e-6).
  - POLICY-GRADIENT LOSS: recompute per-token logprobs of the model's OWN action tokens under the
    CURRENT LoRA (a forward pass on sft_input_ids, gather logprob at positions where sft_labels != -100),
    loss = - A * mean(logprob_of_action_tokens), summed/averaged over the batch -> backward -> step.
    (Optional KL/entropy kept OFF by default to stay minimal.)

Model / LoRA / optimizer setup MIRRORS scripts/sft_lora.py so it is known-good on the 7B.

Usage:
  python scripts/grpo_cost.py --harvest runs/agent/grpo_harvest_1.json \
      --model Qwen/Qwen2.5-Coder-7B-Instruct --out runs/sft/effic_lora_grpo \
      [--init-adapter runs/sft/effic_lora_grpo] --lambda 0.5 --steps 4 --lr 1e-5

CPU sanity (NO model load):
  python scripts/grpo_cost.py --dry-run-rewards runs/agent/relabel2_harvest.json --lambda 0.5
"""
import os, sys, json, argparse, math
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ap = argparse.ArgumentParser()
ap.add_argument("--harvest", nargs="+", help="harvest json(s) with --save-sft rows (one round of G rollouts/task)")
ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
ap.add_argument("--out", default="runs/sft/effic_lora_grpo")
ap.add_argument("--init-adapter", default=None, help="continue from a previous LoRA adapter (round r>1)")
ap.add_argument("--lambda", dest="lam", type=float, default=0.5, help="token-cost weight in the reward")
ap.add_argument("--token-cap", type=float, default=4000.0, help="in_tokens normalizer (clip in_tokens/cap to [0,1])")
ap.add_argument("--steps", type=int, default=4, help="K advantage-weighted PG steps over the batch")
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--bs", type=int, default=1)            # micro-batch (match sft_lora)
ap.add_argument("--accum", type=int, default=8)         # grad accumulation -> effective batch
ap.add_argument("--max-len", type=int, default=4096)
ap.add_argument("--rank", type=int, default=16)         # match sft_lora defaults
ap.add_argument("--alpha", type=int, default=32)        # match sft_lora defaults
ap.add_argument("--min-train-tokens", type=int, default=5, help="skip trajectories with too few action tokens")
ap.add_argument("--entropy", type=float, default=0.0, help="OPTIONAL entropy bonus coeff (default OFF, minimal)")
ap.add_argument("--dry-run-rewards", default=None,
                help="CPU-ONLY: load this harvest, run the REWARD+ADVANTAGE math (no torch model), print "
                     "per-task sanity, and exit. Validates the math path with NO GPU.")
A = ap.parse_args()


# ----------------------------------------------------------------------------
# REWARD + GROUP-ADVANTAGE math (pure python — no torch, shared by dry-run + train).
# ----------------------------------------------------------------------------
def load_rows(paths):
    """Flatten harvest json(s) to a list of rows (handles list rows OR {cond: [rows]} dict rows)."""
    out = []
    for p in paths:
        d = json.load(open(p))
        rows = d["rows"]
        rowlist = rows if isinstance(rows, list) else [r for c in rows for r in rows[c]]
        out.extend(rowlist)
    return out


def trajectory_reward(row, lam, token_cap):
    """r = resolved ? (1 - lambda * in_tokens_norm) : 0 ; in_tokens_norm = clip(in_tokens/cap, 0, 1)."""
    if not row.get("resolved"):
        return 0.0
    in_tok = float(row.get("in_tokens", 0) or 0)
    in_tokens_norm = min(max(in_tok / token_cap, 0.0), 1.0)        # clip to [0,1]
    return 1.0 - lam * in_tokens_norm


def compute_advantages(rows, lam, token_cap):
    """Attach 'reward' to each row, then group by `task` and set group-normalized 'advantage'
    A = (r - mean_group)/(std_group + 1e-6). Returns (rows, per_task_stats)."""
    groups = {}
    for r in rows:
        r["reward"] = trajectory_reward(r, lam, token_cap)
        groups.setdefault(r.get("task", "?"), []).append(r)
    stats = {}
    for task, grp in groups.items():
        rewards = [r["reward"] for r in grp]
        n = len(rewards)
        mean = sum(rewards) / n
        var = sum((x - mean) ** 2 for x in rewards) / n          # population std (matches GRPO group norm)
        std = math.sqrt(var)
        for r in grp:
            r["advantage"] = (r["reward"] - mean) / (std + 1e-6)
        advs = [r["advantage"] for r in grp]
        stats[task] = {
            "n": n,
            "n_resolved": sum(1 for r in grp if r.get("resolved")),
            "mean_reward": mean, "std_reward": std,
            "mean_in_tokens": sum(float(r.get("in_tokens", 0) or 0) for r in grp) / n,
            "adv_min": min(advs), "adv_max": max(advs), "adv_mean": sum(advs) / n,
        }
    return rows, stats


# ----------------------------------------------------------------------------
# DRY-RUN: reward/advantage sanity with NO torch model (CPU only).
# ----------------------------------------------------------------------------
def dry_run_rewards(path, lam, token_cap):
    rows = load_rows([path])
    rows, stats = compute_advantages(rows, lam, token_cap)
    print(f"[dry-run-rewards] {path}")
    print(f"[dry-run-rewards] lambda={lam} token_cap={token_cap}  n_rows={len(rows)}\n")
    all_adv = [r["advantage"] for r in rows]
    finite = all(math.isfinite(a) for a in all_adv)
    print(f"{'task':24} {'n':>3} {'res':>4} {'meanR':>7} {'stdR':>6} {'mInTok':>8} "
          f"{'advMin':>7} {'advMax':>7} {'advMean':>8}")
    zero_mean_ok = True
    monotonic_ok = True
    for task in sorted(stats):
        s = stats[task]
        print(f"{task:24} {s['n']:>3} {s['n_resolved']:>4} {s['mean_reward']:>7.3f} "
              f"{s['std_reward']:>6.3f} {s['mean_in_tokens']:>8.0f} "
              f"{s['adv_min']:>7.3f} {s['adv_max']:>7.3f} {s['adv_mean']:>8.4f}")
        if abs(s["adv_mean"]) > 1e-6:
            zero_mean_ok = False
        # within this group: among the SOLVED trajectories, fewer in_tokens => higher advantage.
        grp = [r for r in rows if r.get("task") == task and r.get("resolved")]
        for i in range(len(grp)):
            for j in range(len(grp)):
                ti, tj = grp[i].get("in_tokens", 0), grp[j].get("in_tokens", 0)
                if ti < tj and grp[i]["advantage"] < grp[j]["advantage"] - 1e-9:
                    monotonic_ok = False
    print()
    print(f"[check] all advantages finite:                 {finite}")
    print(f"[check] per-group advantage mean ~0 (zero-mean): {zero_mean_ok}")
    print(f"[check] within group, solved & fewer-in_tokens => higher advantage: {monotonic_ok}")
    # global reward sanity
    res = [r for r in rows if r.get("resolved")]
    unres = [r for r in rows if not r.get("resolved")]
    print(f"[check] unresolved trajectories all reward==0:  {all(r['reward'] == 0.0 for r in unres)}")
    if res:
        print(f"[stat ] resolved reward range: [{min(r['reward'] for r in res):.3f}, "
              f"{max(r['reward'] for r in res):.3f}]  (1 - lambda*in_tokens_norm)")
    ok = finite and zero_mean_ok and monotonic_ok
    print(f"\n[dry-run-rewards] MATH {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if A.dry_run_rewards:
    sys.exit(dry_run_rewards(A.dry_run_rewards, A.lam, A.token_cap))


# ----------------------------------------------------------------------------
# GPU PATH: train a LoRA with the advantage-weighted policy gradient.
# (Below imports torch only when actually training, so the dry-run stays CPU/torch-free.)
# ----------------------------------------------------------------------------
if not A.harvest:
    print("[grpo] --harvest required for training (or use --dry-run-rewards for the CPU math sanity)")
    sys.exit(2)

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

# ---- collect trajectories + compute reward/advantage (group by task) ----
rows = load_rows(A.harvest)
rows, stats = compute_advantages(rows, A.lam, A.token_cap)

examples = []   # list of (input_ids, labels, advantage)
seen, kept, by_task = 0, 0, {}
for r in rows:
    seen += 1
    if "sft_input_ids" not in r:
        continue
    ids, labs = r["sft_input_ids"], r["sft_labels"]
    if len(ids) > A.max_len:
        ids, labs = ids[:A.max_len], labs[:A.max_len]
    if sum(1 for x in labs if x != -100) < A.min_train_tokens:
        continue
    # Skip zero-advantage trajectories: they contribute no gradient (e.g. all-solved/all-failed groups).
    if abs(r["advantage"]) < 1e-9:
        continue
    examples.append((ids, labs, float(r["advantage"])))
    kept += 1
    by_task[r.get("task", "?")] = by_task.get(r.get("task", "?"), 0) + 1

print(f"[data] {seen} rows scanned -> {kept} trajectories with nonzero advantage kept")
print(f"[data] by task: {by_task}")
print("[reward] per-task mean reward / mean in_tokens / %resolved:")
for task in sorted(stats):
    s = stats[task]
    print(f"  {task:24} meanR={s['mean_reward']:.3f}  mInTok={s['mean_in_tokens']:.0f}  "
          f"resolved={s['n_resolved']}/{s['n']}")
if kept == 0:
    print("[data] NOTHING to train on — every group had uniform reward (zero advantage). "
          "Need within-task variance (mix of solved/unsolved or differing in_tokens).")
    sys.exit(1)

# ---- model + LoRA (MIRRORS sft_lora.py) ----
print(f"[load] {A.model}{' + '+A.init_adapter if A.init_adapter else ''}", flush=True)
tok = AutoTokenizer.from_pretrained(A.model)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(A.model, dtype=torch.bfloat16, device_map={"": 0})
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
if A.init_adapter:
    # Round r>1: continue training the SAME adapter (trainable).
    model = PeftModel.from_pretrained(model, A.init_adapter, is_trainable=True)
else:
    lcfg = LoraConfig(r=A.rank, lora_alpha=A.alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
model.print_trainable_parameters()
model.train()

dev = model.device
PAD = tok.pad_token_id


def batches(exs, bs):
    """Pad to the longest in the micro-batch (same scheme as sft_lora.batches), carrying advantages."""
    for i in range(0, len(exs), bs):
        chunk = exs[i:i + bs]
        m = max(len(x[0]) for x in chunk)
        input_ids = torch.full((len(chunk), m), PAD, dtype=torch.long)
        labels = torch.full((len(chunk), m), -100, dtype=torch.long)
        attn = torch.zeros((len(chunk), m), dtype=torch.long)
        advs = torch.zeros((len(chunk),), dtype=torch.float32)
        for j, (ids, labs, adv) in enumerate(chunk):
            n = len(ids)
            input_ids[j, :n] = torch.tensor(ids); labels[j, :n] = torch.tensor(labs); attn[j, :n] = 1
            advs[j] = adv
        yield input_ids.to(dev), attn.to(dev), labels.to(dev), advs.to(dev)


def pg_loss(logits, labels, advs, entropy_coef=0.0):
    """Advantage-weighted policy-gradient loss on the model's OWN action tokens.

    Causal-LM shift: token at position t predicts token t+1, so logits[:, :-1] line up with labels[:, 1:].
    The mask is labels != -100 — EXACTLY the SFT action-token mask (prompt/observations are -100), so we
    only score the tokens the model itself generated. For each kept token we gather its log p(token) under
    the current policy, average those per-trajectory, and weight by the trajectory advantage A.
    loss = - mean_b( A_b * mean_t logπ(action_token_t) ).  (Optional entropy bonus, default off.)
    """
    shift_logits = logits[:, :-1, :]                                   # (B, T-1, V)
    shift_labels = labels[:, 1:]                                       # (B, T-1)
    mask = (shift_labels != -100)                                      # (B, T-1) — the SFT action-token mask
    logp = F.log_softmax(shift_logits.float(), dim=-1)                 # (B, T-1, V)
    safe = shift_labels.clamp_min(0).unsqueeze(-1)                     # -100 -> 0 so gather is in-range
    tok_logp = logp.gather(-1, safe).squeeze(-1)                       # (B, T-1) log p(actual action token)
    tok_logp = tok_logp * mask                                        # zero out non-action positions
    counts = mask.sum(dim=1).clamp_min(1)                             # (B,) #action tokens per trajectory
    mean_logp = tok_logp.sum(dim=1) / counts                          # (B,) mean action-token logprob
    loss = -(advs * mean_logp).mean()                                # POLICY GRADIENT: -A * mean logπ(action)
    if entropy_coef > 0.0:
        ent = -(logp.exp() * logp).sum(-1)                           # (B, T-1) per-token entropy
        ent = (ent * mask).sum(dim=1) / counts                       # (B,) mean action-token entropy
        loss = loss - entropy_coef * ent.mean()                      # bonus -> maximize entropy
    return loss


opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=A.lr)
print(f"[train] {len(examples)} ex, bs={A.bs} accum={A.accum} steps(K)={A.steps} lr={A.lr} "
      f"lambda={A.lam} cap={A.token_cap}", flush=True)

import random
gstep = 0; running = 0.0; nrun = 0
for k in range(A.steps):
    order = list(range(len(examples)))
    random.Random(4242 + k).shuffle(order)
    shuffled = [examples[i] for i in order]
    opt.zero_grad()
    for input_ids, attn, labels, advs in batches(shuffled, A.bs):
        out = model(input_ids=input_ids, attention_mask=attn)        # NO labels -> we compute PG loss ourselves
        loss = pg_loss(out.logits, labels, advs, entropy_coef=A.entropy) / A.accum
        loss.backward()
        running += loss.item() * A.accum; nrun += 1
        gstep += 1
        if gstep % A.accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
    # flush a trailing partial-accumulation batch so the K-th step's gradient is applied
    if gstep % A.accum != 0:
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); opt.zero_grad()
    print(f"  [pg] step {k+1}/{A.steps} mean_loss {running/max(nrun,1):.4f}", flush=True)
    running = 0.0; nrun = 0

os.makedirs(A.out, exist_ok=True)
model.save_pretrained(A.out)
tok.save_pretrained(A.out)
print(f"[done] adapter saved -> {A.out}", flush=True)
