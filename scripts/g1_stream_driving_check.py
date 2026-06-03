#!/usr/bin/env python3
"""Decisive check: can we get COHERENT code out of stream-qwen3-8b using the
substrate's OWN canonical generate() API (prompt on User channel, Output channel
response), vs the broken raw-completion path the G1 harness used?

If Output is coherent -> the G1 garbage was a driving bug; re-run G1 properly.
If still garbage      -> substrate code-capability problem (deeper).
"""
import os, sys
os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from huggingface_hub import snapshot_download
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "JonasGeiping/stream-qwen3-8b"
snap = snapshot_download(REPO)  # already cached on NAS
sys.path.insert(0, snap)
from stream_inference import generate, detect_silence_token  # noqa: E402

print("[load] model ...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    REPO, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
tok = AutoTokenizer.from_pretrained(REPO)
sil = detect_silence_token(tok)
print(f"[load] done; silence_token={sil}", flush=True)

# HumanEval/0 prompt, framed as an instruction (both models are instruct/chat).
he0 = '''from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
'''
prompt = ("Complete this Python function. Return only the full function "
          "implementation in a code block:\n\n" + he0)

def run(label, user_prompt, **kw):
    rows_out = []
    g = generate(model, tok, user_prompt, sil, max_rows=260, warm_start=True, **kw)
    for row_idx, row, is_prefill in g:
        if not is_prefill:
            rows_out.append(row[1])  # Output channel id=1
    nonsil = [t for t in rows_out if t != sil]
    text = tok.decode(nonsil).strip()
    print(f"\n===== {label} =====\n{text[:700]}\n----- ({len(nonsil)} non-silence Output tokens) -----", flush=True)
    return text

# TRUE greedy: temperature=0 -> argmax path (line 142). Determinism x3.
a = run("temp=0 run 1", prompt, temperature=0.0, silence_penalty=10.0)
b = run("temp=0 run 2", prompt, temperature=0.0, silence_penalty=10.0)
c = run("temp=0 run 3", prompt, temperature=0.0, silence_penalty=10.0)
print(f"\n[determinism temp=0] r1==r2=={a==b}, r2==r3={b==c}, all_nonempty={min(len(a),len(b),len(c))>0}", flush=True)

# A second, simpler problem to gauge breadth.
he_simple = ('Complete this Python function. Return only the function in a code block:\n\n'
             'def add(a: int, b: int) -> int:\n    """Return the sum of a and b."""\n')
d = run("temp=0 add()", he_simple, temperature=0.0, silence_penalty=10.0)
