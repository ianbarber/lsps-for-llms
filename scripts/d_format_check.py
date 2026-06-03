#!/usr/bin/env python3
"""Fast prompt/format iteration (no slow TaskEnv): does the 7B emit applicable
SEARCH/REPLACE blocks on a representative buggy file? Uses MockEnv (real pyrefly).
Iterate the SYS prompt here, then confirm once on the real task."""
import os, sys
os.environ.setdefault("HF_HOME", "/mnt/nas/hf-cache")
sys.path.insert(0, "/home/ianbarber/Projects/Streams")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scaffold.stream_agent import StreamAgent
from scaffold.mock_env import MockEnv

BUGGY = '''from typing import Optional

class Cache:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.store: dict = {}
        self.order: list = []

    def get(self, key: str) -> Optional[int]:
        if key not in self.store:
            return -1
        self.order.remove(key)
        self.order.append(key)
        return self.store[key]

    def put(self, key: str, value: int) -> None:
        if key in self.store:
            self.order.remove(key)
        elif len(self.store) >= self.capacity:
            oldest = self.order.pop()          # BUG: pop() removes most-recent, not LRU
            del self.store[oldest]
        self.store[key] = value
        self.order.append(key)
'''
TEST = ("c=Cache(2)\nc.put('a',1)\nc.put('b',2)\nc.get('a')\nc.put('c',3)\n"
        "assert c.get('b')==-1   # b is LRU, should be evicted\nassert c.get('a')==1\nassert c.get('c')==3")
TASK = ("File sol.py implements an LRU cache but eviction removes the WRONG entry "
        "(it evicts the most-recently-used instead of the least-recently-used), so "
        "the LRU test fails. Fix the eviction. The file content:\n\n" +
        "\n".join(f"{i+1:3d}| {l}" for i,l in enumerate(BUGGY.splitlines())))

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct",
        torch_dtype=torch.bfloat16, device_map="auto").eval()

for cond in ("A", "D"):
    env = MockEnv(BUGGY, TEST, "Cache")
    r = StreamAgent(model, tok, env, condition=cond, latency_tokens=8, max_new_tokens=900).run(TASK, "sol.py")
    ev = r["events"]
    napplied = sum(1 for e in ev if e["type"]=="edit" and e["ok"])
    ntest = sum(1 for e in ev if e["type"]=="test")
    seq = " -> ".join(e["type"]+("✓" if e.get("resolved") else "") for e in ev)
    print(f"\n=== {cond} === resolved={r['resolved']} edits={napplied} tests={ntest} "
          f"diag={sum(1 for e in ev if e['type'].startswith('diag'))} tok={r['n_tokens']}")
    print(f"  event seq: {seq}")
    if cond == "A":
        import os; os.makedirs("runs/d_agentcheck", exist_ok=True)
        open("runs/d_agentcheck/A_stream.txt","w").write(r["stream"])
        print(f"  --- stream repr (first 1400) ---\n{r['stream'][:1400]!r}")
    env.close()
