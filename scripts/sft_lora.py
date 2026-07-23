#!/usr/bin/env python3
"""LoRA-SFT on harvested OPSD trajectories — teach the model to PREFER <defn> over reading whole files.

Input: one or more harvest JSONs (run synth_mf.py with --save-sft) whose rows carry sft_input_ids/sft_labels
(prompt + observations masked = -100, only the model's own generated tokens are trained). We FILTER to the clean
teacher demos (resolved AND the model actually used a real <defn>/<findrefs> that returned content), then SFT a
LoRA adapter. The distilled adapter is loaded back via synth_mf.py --adapter for the payoff re-test.

Usage:
  python scripts/sft_lora.py --harvest runs/agent/h7b_forced.json [more.json ...] \
      --model Qwen/Qwen2.5-Coder-7B-Instruct --out runs/sft/effic_lora --epochs 3 --lr 1e-4
"""
import os, sys, json, argparse, math
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

ap = argparse.ArgumentParser()
ap.add_argument("--harvest", nargs="+", required=True, help="harvest json(s) with --save-sft rows")
ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
ap.add_argument("--revision", default=None, help="pin the model revision (recorded in the run log)")
ap.add_argument("--filter", default="teacher", choices=["teacher", "sft_keep"],
                help="teacher = the historical clean-retrieval filter (default, unchanged); "
                     "sft_keep = trust the harvest driver's own row['sft_keep'] flag "
                     "(substitution harvest: the driver already applied its acceptance criteria)")
ap.add_argument("--out", default="runs/sft/effic_lora")
ap.add_argument("--epochs", type=float, default=3)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--bs", type=int, default=1)            # micro-batch
ap.add_argument("--accum", type=int, default=8)         # grad accumulation -> effective batch
ap.add_argument("--max-len", type=int, default=4096)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--alpha", type=int, default=32)
ap.add_argument("--min-train-tokens", type=int, default=5, help="skip trajectories with too few trained tokens")
A = ap.parse_args()

def is_clean_teacher(row):
    """resolved AND a clean retrieval trajectory: a DAgger lead action (defn-first OR read-first), OR a real
    <defn>/<findrefs> that returned content. Keeping lead-read trajectories is essential for the BOUNDARY set
    (read-required tasks) so the student learns 'defn when sufficient, read when needed' (not always-defn)."""
    if not row.get("resolved"):
        return False
    for e in row.get("events", []):
        if e.get("lead"):                                       # DAgger lead action (defn-first or read-first)
            return True
        if e.get("type") == "defn" and e.get("found"):
            return True
        if e.get("type") == "findrefs" and (e.get("hits") or 0) > 0:
            return True
    return False

# ---- collect + filter trajectories ----
examples = []           # list of (input_ids, labels)
seen, kept, by_task = 0, 0, {}
for path in A.harvest:
    d = json.load(open(path))
    rows = d["rows"]
    rowlist = rows if isinstance(rows, list) else [r for c in rows for r in rows[c]]
    for r in rowlist:
        seen += 1
        if "sft_input_ids" not in r:
            continue
        if A.filter == "sft_keep":
            if not r.get("sft_keep"):
                continue
        elif not is_clean_teacher(r):
            continue
        ids, labs = r["sft_input_ids"], r["sft_labels"]
        if len(ids) > A.max_len:
            ids, labs = ids[:A.max_len], labs[:A.max_len]
        if sum(1 for x in labs if x != -100) < A.min_train_tokens:
            continue
        examples.append((ids, labs))
        kept += 1
        by_task[r.get("task", "?")] = by_task.get(r.get("task", "?"), 0) + 1

print(f"[data] {seen} rows scanned -> {kept} clean teacher trajectories kept")
print(f"[data] by task: {by_task}")
if kept == 0:
    print("[data] NOTHING to train on — check the harvest filter (resolved + real <defn>)."); sys.exit(1)

# ---- model + LoRA ----
print(f"[load] {A.model}{'@'+A.revision if A.revision else ''}  filter={A.filter}", flush=True)
tok = AutoTokenizer.from_pretrained(A.model, revision=A.revision)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(A.model, revision=A.revision,
                                             dtype=torch.bfloat16, device_map={"": 0})
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
lcfg = LoraConfig(r=A.rank, lora_alpha=A.alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lcfg)
model.print_trainable_parameters()
model.train()

dev = model.device
PAD = tok.pad_token_id

def batches(exs, bs):
    for i in range(0, len(exs), bs):
        chunk = exs[i:i + bs]
        m = max(len(x[0]) for x in chunk)
        input_ids = torch.full((len(chunk), m), PAD, dtype=torch.long)
        labels = torch.full((len(chunk), m), -100, dtype=torch.long)
        attn = torch.zeros((len(chunk), m), dtype=torch.long)
        for j, (ids, labs) in enumerate(chunk):
            n = len(ids)
            input_ids[j, :n] = torch.tensor(ids); labels[j, :n] = torch.tensor(labs); attn[j, :n] = 1
        yield input_ids.to(dev), attn.to(dev), labels.to(dev)

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=A.lr)
steps_per_epoch = math.ceil(len(examples) / A.bs)
total_steps = int(steps_per_epoch * A.epochs)
print(f"[train] {len(examples)} ex, bs={A.bs} accum={A.accum} epochs={A.epochs} -> ~{total_steps} micro-steps", flush=True)

import random
step = 0; running = 0.0
n_epochs = int(math.ceil(A.epochs))
for ep in range(n_epochs):
    order = list(range(len(examples)))
    # deterministic-ish shuffle by epoch (no Math.random equivalent needed; this is python random, fine here)
    random.Random(1234 + ep).shuffle(order)
    shuffled = [examples[i] for i in order]
    for input_ids, attn, labels in batches(shuffled, A.bs):
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        loss = out.loss / A.accum
        loss.backward()
        running += out.loss.item()
        step += 1
        if step % A.accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
        if step % 10 == 0:
            print(f"  ep{ep} step {step}/{total_steps} loss {running/10:.4f}", flush=True); running = 0.0
        if A.epochs < n_epochs and step >= total_steps:
            break

os.makedirs(A.out, exist_ok=True)
model.save_pretrained(A.out)
tok.save_pretrained(A.out)
json.dump({"config": vars(A), "n_examples": len(examples), "n_rows_scanned": seen,
           "by_task": by_task, "torch": torch.__version__},
          open(os.path.join(A.out, "streams_train_meta.json"), "w"), indent=2)
print(f"[done] adapter saved -> {A.out}", flush=True)
