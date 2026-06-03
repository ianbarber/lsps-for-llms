#!/usr/bin/env python3
"""Self-distill data for the interleaving recipe (Rung 1). Constructs training
sequences where ‹info›FACT‹/info› is interleaved mid-generation and LOSS-MASKED, so
the model learns to USE injected info (the random value is only knowable from the
inject) without generating it. Output: jsonl {input_ids, labels} for d_sft.py.

Disjoint from the eval (different value seed / names / phrasings).
Usage: i_gen_data.py [out_dir] [n]
"""
import os, sys, json, random
os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")
from transformers import AutoTokenizer

OUT = sys.argv[1] if len(sys.argv) > 1 else "runs/i_sft_data"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2400
os.makedirs(OUT, exist_ok=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
INFO_OPEN, INFO_CLOSE = "\n‹info›\n", "\n‹/info›\n"
rng = random.Random(12345)  # disjoint from eval (seed 0)

NAMES = ["limit", "threshold", "max_size", "timeout", "port", "retries", "buffer_len",
         "seed", "offset", "capacity", "window", "batch", "depth", "stride", "quota",
         "ttl", "page_size", "max_conn", "chunk", "level"]
# preamble variants control WHERE the inject lands (model must generalize over position)
PREAMBLES = [
    "def {ep}() -> int:\n    ",
    "def {ep}() -> int:\n    \"\"\"Return the configured {nm}.\"\"\"\n    ",
    "def {ep}() -> int:\n    # returns the configured {nm}\n    ",
]

def build(i):
    nm = NAMES[i % len(NAMES)]
    val = rng.randint(1000, 99999)
    ep = f"get_{nm}"
    prompt = (f"Write a Python function `{ep}()` that takes no arguments and returns the "
              f"configured {nm} as an integer. Return only the function.")
    pre = rng.choice(PREAMBLES).format(ep=ep, nm=nm)
    fact = f"The configured {nm} is {val}. {ep}() must return exactly {val}."
    body = f"return {val}\n"
    # chat-formatted prompt (masked) + assistant: pre [‹info› masked] body
    head = tok.apply_chat_template([{"role": "system", "content": "You are a coding assistant. Write the requested function."},
                                    {"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    head_ids = tok(head, add_special_tokens=False).input_ids
    pre_ids  = tok(pre, add_special_tokens=False).input_ids
    info_ids = tok(INFO_OPEN + fact + INFO_CLOSE, add_special_tokens=False).input_ids
    body_ids = tok(body, add_special_tokens=False).input_ids + [tok.eos_token_id]
    input_ids = head_ids + pre_ids + info_ids + body_ids
    # loss: -100 on head (prompt) and on info span; train on pre + body (the model's output)
    labels = ([-100]*len(head_ids) + pre_ids + [-100]*len(info_ids) + body_ids)
    return {"input_ids": input_ids, "labels": labels, "ep": ep, "value": val,
            "n_loss": sum(1 for x in labels if x != -100)}

rows = [build(i) for i in range(N)]
with open(os.path.join(OUT, "data.jsonl"), "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
import statistics as st
print(f"[done] {len(rows)} examples -> {OUT}/data.jsonl")
print(f"  mean seq_len={st.mean(len(r['input_ids']) for r in rows):.0f}  "
      f"mean loss_tokens={st.mean(r['n_loss'] for r in rows):.0f}")
# sanity: decode one with mask markers
r = rows[0]
dec = []
for tid, lab in zip(r["input_ids"], r["labels"]):
    s = tok.decode([tid])
    dec.append(s if lab != -100 else f"·{s}·")
print("  sample (·masked·):", "".join(dec)[:400].replace("\n", "\\n"))
