#!/usr/bin/env python3
"""Option-D interleaved-async splicing prototype (v0.5 §0.9 Q2/Q4).

Two jobs:
  (1) Measure ms_per_token on Qwen2.5-Coder (for the latency->token-offset map;
      the old 200ms/token was the retired stream model — G4 flagged this).
  (2) Demonstrate the core mechanism: generate agent tokens, then SPLICE a
      diagnostic block into the live context stream and continue generating, so
      the model conditions on a diagnostic that arrived mid-stream (the D case).
      Confirms KV-cache handling: the spliced tokens are prefilled into the
      cache, then decoding resumes after them.

Run inline after the GPU frees. HF_HOME=/mnt/nas/hf-cache.
"""
import os, sys, time, json
os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-Coder-7B-Instruct"
OUT = sys.argv[2] if len(sys.argv) > 2 else "runs/d_capability/splice_proto.json"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()
dev = model.device
res = {"model": MODEL}

# ---- (1) ms_per_token: time steady-state single-sequence decode ----
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "Write a Python function to merge two sorted lists."}],
    tokenize=False, add_generation_prompt=True)
ids = tok(prompt, return_tensors="pt").to(dev)
with torch.no_grad():
    model.generate(**ids, max_new_tokens=16, do_sample=False, pad_token_id=tok.eos_token_id)  # warm
    torch.cuda.synchronize(); t0 = time.time()
    out = model.generate(**ids, max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize(); dt = time.time() - t0
n_new = out.shape[1] - ids["input_ids"].shape[1]
res["ms_per_token_single"] = 1000.0 * dt / max(n_new, 1)
res["tok_per_s_single"] = n_new / dt
print(f"[ms/token] {res['ms_per_token_single']:.1f} ms/token  ({res['tok_per_s_single']:.1f} tok/s single-seq)", flush=True)

# ---- (2) splice demo: KV-cache prefill of a mid-stream diagnostic block ----
from transformers import DynamicCache
DIAG = "\n‹diag›\n[error] L3 bad-return-type: expected int, got str\n‹/diag›\n"

def greedy_step(input_ids, past):
    with torch.no_grad():
        o = model(input_ids=input_ids, past_key_values=past, use_cache=True)
    nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
    return nxt, o.past_key_values

# prefill the prompt
cache = DynamicCache()
with torch.no_grad():
    o = model(**ids, use_cache=True); cache = o.past_key_values
cur = o.logits[:, -1, :].argmax(-1, keepdim=True)
emitted = []
# emit 12 agent tokens
for _ in range(12):
    emitted.append(cur.item()); cur, cache = greedy_step(cur, cache)
pre = tok.decode(emitted)
# SPLICE: prefill the diagnostic block into the cache (no generation), then continue
diag_ids = tok(DIAG, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
with torch.no_grad():
    o = model(input_ids=diag_ids, past_key_values=cache, use_cache=True); cache = o.past_key_values
cur = o.logits[:, -1, :].argmax(-1, keepdim=True)
post = []
for _ in range(40):
    post.append(cur.item()); cur, cache = greedy_step(cur, cache)
post_txt = tok.decode(post)
res["splice_demo"] = {"pre_splice": pre, "spliced_block": DIAG.strip(),
                      "post_splice": post_txt,
                      "ok": len(post_txt.strip()) > 0}
print(f"\n[splice] pre: {pre!r}\n[splice] post(diag): {post_txt!r}", flush=True)
print(f"[splice] continued after mid-stream diagnostic: {res['splice_demo']['ok']}", flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=2)
print(f"\n[DONE] -> {OUT}", flush=True)
