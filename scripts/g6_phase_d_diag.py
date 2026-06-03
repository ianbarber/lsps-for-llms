#!/usr/bin/env python3
"""Phase D diagnostics:
  (1) EAGER Phase D vs eager Phase B, 5 prompts x 30 rows -> identity_eager.json
      (proves the cursor refactor is correct; isolates capture bug from refactor)
  (2) profiler on a single eager decode row at buffer=460: where does the
      ~400 ms/row go? attention vs mlp vs proj. -> profile.json
  (3) a no-attention-context control: time one row at buffer=460 vs buffer=10240
      to confirm the context-scaling is attention-over-KV.
"""
from __future__ import annotations
import importlib.util, json, os, sys, time
from pathlib import Path

os.environ["HF_HOME"] = "/home/ianbarber/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/home/ianbarber/.cache/huggingface/hub"

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "JonasGeiping/stream-qwen3-8b"
OUT = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_d")
PATCH_D = OUT/"patched"/"stream_inference_phase_d.py"
PATCH_B = Path("/home/ianbarber/Projects/Streams/runs/g6_phase_b/patched/stream_inference_phase_b.py")
PROMPTS = [
    "Write a Python function that reverses a linked list in place.",
    "Explain how a B-tree differs from a binary search tree.",
    "Refactor this code to use a context manager: open('f.txt'); read(); close().",
    "What is the time complexity of merge sort, and why?",
    "Sketch a unit test for a function that adds two integers.",
]

def lp(path, name):
    s=importlib.util.spec_from_file_location(name,str(path)); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def greedy(gen_fn, model, tok, silence, prompt, n, extra):
    rows=[]
    for ri,row,isp in gen_fn(model,tok,prompt,silence,max_rows=n+5,warm_start=False,temperature=0.0,**extra):
        if isp: continue
        rows.append(list(row))
        if len(rows)>=n: break
    return rows

def main():
    snap=snapshot_download(MODEL_ID)
    if snap not in sys.path: sys.path.insert(0,snap)
    modD=lp(PATCH_D,"si_d"); modB=lp(PATCH_B,"si_b")
    print("[load]...",flush=True); t0=time.perf_counter()
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,trust_remote_code=True,torch_dtype=torch.bfloat16,device_map="auto"); model.eval()
    tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    silence=modD.detect_silence_token(tok)
    print(f"[load] {time.perf_counter()-t0:.1f}s",flush=True)

    # (1) eager identity 30 rows
    print("\n[eager-identity] Phase D eager vs Phase B eager, 30 rows x 5 prompts",flush=True)
    res={"per_prompt":[],"all_match":True}
    for i,p in enumerate(PROMPTS):
        b=greedy(modB.generate,model,tok,silence,p,30,{})
        d=greedy(modD.generate,model,tok,silence,p,30,{"max_context_rows":512})
        n=min(len(b),len(d)); match=b[:n]==d[:n] and len(b)==len(d)
        fd=next((r for r in range(n) if b[r]!=d[r]),None)
        res["per_prompt"].append({"prompt":i,"match":bool(match),"first_div":fd,"b_rows":len(b),"d_rows":len(d)})
        res["all_match"]=res["all_match"] and match
        print(f"[eager-identity] prompt {i}: match={match} div={fd}",flush=True)
    (OUT/"identity_eager.json").write_text(json.dumps(res,indent=2))
    print(f"[eager-identity] ALL_MATCH={res['all_match']}",flush=True)

    # (3) context-scaling control: time one steady row at small vs large buffer
    def time_rows(mcr, nrows=40):
        torch.cuda.synchronize(); times=[]; prev=None
        for ri,row,isp in modD.generate(model,tok,PROMPTS[1],silence,max_rows=nrows,
                warm_start=False,temperature=0.0,max_context_rows=mcr):
            if isp: continue
            torch.cuda.synchronize(); now=time.perf_counter()
            if prev is not None: times.append(now-prev)
            prev=now
            if len(times)>=nrows-2: break
        times.sort(); return times[len(times)//2]*1000.0
    print("\n[ctx-control] timing one steady row at varying buffer sizes (eager)",flush=True)
    ctl={}
    for mcr in (64, 460, 2048, 10240):
        ms=time_rows(mcr); ctl[str(mcr)]=ms
        print(f"[ctx-control] buffer={mcr} cols={mcr*10}: {ms:.1f} ms/row",flush=True)
    (OUT/"ctx_control.json").write_text(json.dumps(ctl,indent=2))

    # (2) profiler on a single eager row at buffer=460
    print("\n[profile] profiling steady rows at buffer=460 (eager)",flush=True)
    from torch.profiler import profile, ProfilerActivity
    g=modD.generate(model,tok,PROMPTS[1],silence,max_rows=20,warm_start=False,
                    temperature=0.0,max_context_rows=460)
    # advance past prefill + a few warm rows
    for k,(ri,row,isp) in enumerate(g):
        if isp: continue
        if k>5: break
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=False) as prof:
        for k,(ri,row,isp) in enumerate(g):
            if isp: continue
            torch.cuda.synchronize()
            if k>=8: break
    tbl=prof.key_averages().table(sort_by="cuda_time_total",row_limit=25)
    (OUT/"profile.txt").write_text(tbl)
    print(tbl[:2000],flush=True)
    print("\n[done] wrote identity_eager.json, ctx_control.json, profile.txt",flush=True)

if __name__=="__main__":
    main()
