#!/usr/bin/env python3
"""Numeric-health + correct-recipe sanity check for stream-qwen3-8b.

Addresses two questions:
  (1) Is the model numerically healthy / correctly loaded? (finite logits)
  (2) Driven the README's BLESSED way (conversational prompt, warm_start,
      temperature=0.6, silence_penalty=5.0, skip_silence=True), does it produce
      COHERENT output? If yes -> healthy cognition model, our code struggles are
      domain/capability. If garbage -> real numeric/loading bug.
"""
import os, sys
os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from huggingface_hub import snapshot_download
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "JonasGeiping/stream-qwen3-8b"
snap = snapshot_download(REPO)
sys.path.insert(0, snap)
from stream_inference import detect_silence_token  # noqa: E402

print("[load] ...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    REPO, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
tok = AutoTokenizer.from_pretrained(REPO)
sil = detect_silence_token(tok)
print(f"[load] done; silence_token={sil}", flush=True)

# (1) numeric health: one forward over a tiny input, check logits finite + stats.
with torch.no_grad():
    ids = torch.tensor([[tok.encode(" hello", add_special_tokens=False)[0]]*10], device=model.device)
    pos = torch.zeros_like(ids); ch = torch.arange(10, device=model.device).view(1,10)
    try:
        out = model(input_ids=ids, position_ids=pos, channel_ids=ch)
        lg = out.logits
        print(f"[numeric] logits shape={tuple(lg.shape)} dtype={lg.dtype} "
              f"finite={torch.isfinite(lg).all().item()} "
              f"min={lg.float().min().item():.2f} max={lg.float().max().item():.2f} "
              f"mean={lg.float().mean().item():.3f}", flush=True)
    except Exception as e:
        print(f"[numeric] forward raised: {type(e).__name__}: {e}", flush=True)

# (2) README blessed recipe — conversational prompt, stochastic + greedy.
def chat(label, prompt, **kw):
    res = model.stream_generate(tok, prompt, max_rows=80, warm_start=True,
                                skip_silence=True, **kw)
    print(f"\n===== {label} =====", flush=True)
    print(f"Output:     {res.output[:400]!r}", flush=True)
    print(f"Analytical: {res.channel_texts.get('Analytical','')[:200]!r}", flush=True)
    print(f"Synthesis:  {res.channel_texts.get('Synthesis','')[:200]!r}", flush=True)

chat("README recipe (temp=0.6, sp=5)", "What's something you've been thinking about?",
     temperature=0.6, silence_penalty=5.0)
chat("greedy conversational (temp=0, sp=5)", "What's something you've been thinking about?",
     temperature=0.0, silence_penalty=5.0)
chat("greedy hello (temp=0, sp=5)", "Hello, what's up?",
     temperature=0.0, silence_penalty=5.0)
