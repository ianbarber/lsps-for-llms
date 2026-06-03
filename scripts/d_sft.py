#!/usr/bin/env python3
"""LoRA SFT of Qwen2.5-Coder on an interleaved-condition dataset (v0.5 L1).

Trains the coder to REACT to live LSP diagnostics interleaved into its stream.
Consumes a per-condition dataset of {input_ids, labels} where labels == input_ids
except -100 on prompt + ‹diag› spans (so loss is only on the agent's fix tokens —
the model conditions on the diagnostic but does not learn to generate it).

Usage: d_sft.py <data_dir_or_jsonl> <out_adapter_dir> [--model ID] [--epochs N]
       [--bs N] [--accum N] [--lr F] [--max-len N] [--limit N]
"""
import os, sys, json, argparse, glob
os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk, Dataset

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("out")
ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
ap.add_argument("--epochs", type=float, default=2.0)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--max-len", type=int, default=1024)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

# ---- load dataset (HF saved dataset dir, or jsonl) ----
def load_any(path):
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "dataset_info.json")):
        return load_from_disk(path)
    files = [path] if path.endswith(".jsonl") else sorted(glob.glob(os.path.join(path, "*.jsonl")))
    rows = []
    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if line: rows.append(json.loads(line))
    return Dataset.from_list(rows)

ds = load_any(args.data)
if args.limit: ds = ds.select(range(min(args.limit, len(ds))))
keep = [c for c in ds.column_names if c not in ("input_ids", "labels")]
ds = ds.remove_columns(keep)
ds = ds.filter(lambda e: len(e["input_ids"]) <= args.max_len)
print(f"[data] {len(ds)} examples (<= {args.max_len} tok)", flush=True)

tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token

def collate(batch):
    maxlen = max(len(b["input_ids"]) for b in batch)
    ids, lbl, att = [], [], []
    for b in batch:
        n = len(b["input_ids"]); pad = maxlen - n
        ids.append(b["input_ids"] + [tok.pad_token_id]*pad)
        lbl.append(b["labels"] + [-100]*pad)
        att.append([1]*n + [0]*pad)
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lbl),
            "attention_mask": torch.tensor(att)}

print(f"[load] {args.model}", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.bfloat16, device_map="auto")
model.config.use_cache = False
model.enable_input_require_grads()
lora = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, bias="none",
                  task_type="CAUSAL_LM",
                  target_modules=["q_proj","k_proj","v_proj","o_proj",
                                  "gate_proj","up_proj","down_proj"])
model = get_peft_model(model, lora)
model.print_trainable_parameters()

targs = TrainingArguments(
    output_dir=args.out, num_train_epochs=args.epochs,
    per_device_train_batch_size=args.bs, gradient_accumulation_steps=args.accum,
    learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
    bf16=True, logging_steps=10, save_strategy="epoch", report_to=[],
    gradient_checkpointing=True, optim="adamw_torch")
trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collate)
trainer.train()
model.save_pretrained(args.out); tok.save_pretrained(args.out)
print(f"[DONE] adapter -> {args.out}", flush=True)
