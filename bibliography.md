# Bibliography

References for the Streams project (asynchronous in-stream LSP feedback for local coding LLMs). Each entry includes a BibTeX block for direct export.

---

## Core substrate

### Su, Yang, Li, Geiping, *Multi-Stream LLMs* (2026)
Architectural substrate. Stream-aware RoPE, learnable stream embeddings, cross-stream causal mask, interleaved packing for FlashAttention compatibility. Releases two checkpoints: `stream-qwen3-8b` (dense Qwen3-8B backbone, ~16.4 GB BF16) and `stream-qwen3.5-27b` (DeltaNet-hybrid backbone, ~53.8 GB BF16). Smaller variants discussed in `sec5_efficiency/` are training pipelines, not released weights. 10 fixed channels named for cognitive functions (User, Output, Analytical, Skeptical, Intuitive, Between, Curious, Void, Instinct, Synthesis). Inference API: `stream_generate_iter` returns a generator supporting `gen.send(tok)` for environment-driven input; `model.generate()` is intentionally disabled. Lists tool feedback as motivating but does not implement asynchronous tool streams. **[Foundational — D's architecture builds on this directly. v1 substrate is `stream-qwen3-8b`; 27B reserved for v2 follow-up due to dense vs DeltaNet-hybrid architecture difference.]**

```bibtex
@article{su2026multistream,
  title   = {Multi-Stream {LLM}s: Unblocking Language Models with Parallel Streams of Thoughts, Inputs and Outputs},
  author  = {Su and Yang and Li and Geiping, Jonas},
  journal = {arXiv preprint arXiv:2605.12460},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.12460}
}
```

---

## Adjacent prior work (asynchronous / streaming tool use)

### Ginart et al., *Asynchronous Tool Usage for Real-Time Agents* (Salesforce, 2024)
Event-driven FSM architecture wrapping a single-stream LLM; agent loop decomposed into states whose transitions are driven by external events (speech, tool returns, timeouts) rather than end-of-turn token. Integrated with ASR/TTS for concurrent voice interaction. **System demonstration, not a learned policy.** Distinguishing dimensions vs ours: (a) runtime FSM, not architectural multi-stream; (b) voice dialogue domain; (c) demonstrative rather than controlled comparison; (d) open-ended tool surface.

```bibtex
@misc{ginart2024async,
  title        = {Asynchronous Tool Usage for Real-Time Agents},
  author       = {Ginart, Antonio A. and Kodali, Naveen and Lee, Jason and
                  Xiong, Caiming and Savarese, Silvio and Emmons, John},
  year         = {2024},
  eprint       = {2410.21620},
  archivePrefix= {arXiv},
  primaryClass = {cs.AI},
  howpublished = {arXiv:2410.21620}
}
```

### Hooper et al., *Speculative Interaction Agents* (UC Berkeley, May 2026)
**Closest prior work in spirit.** Asynchronous I/O + speculative tool calling for real-time agents. Introduces *clock-based training*: timestamps interleaved into the token stream so the model learns time-aware behaviour, with a synthetic SFT corpus pairing streaming inputs with async tool returns. Reports **1.3–2.2× wall-clock speedup** on HotpotQA / conversational tasks with minor accuracy degradation. Distinguishing dimensions vs ours: (a) single-stream transformer with inline clock tokens, not separate streams; (b) latency target, not capability; (c) generic tools vs LSP; (d) open-domain Q&A, not SWE. Their synthetic-trace construction is a direct reference for our §7.3 corpus.

```bibtex
@misc{hooper2026speculative,
  title        = {Speculative Interaction Agents: Building Real-Time Agents
                  with Asynchronous I/O and Speculative Tool Calling},
  author       = {Hooper, Coleman and Kang, Minwoo and Moon, Suhong and
                  Lee, Nicholas and Wen, Eric and Wawrzynek, John and
                  Mahoney, Michael W. and Shao, Yakun Sophia and
                  Gholami, Amir and Keutzer, Kurt},
  year         = {2026},
  eprint       = {2605.13360},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  howpublished = {arXiv:2605.13360}
}
```

### Gong et al., *GhostShell* (Aug 2025)
Streaming LLM function calls for concurrent embodied programming. Three components: streaming XML function-token parser; dynamic function-interface mapper; **multi-channel scheduler** coordinating *intra-channel synchronous* and *inter-channel asynchronous* calls. Evaluated on robot prototype "COCO," 34 real-world interaction tasks, behavioural-correctness 0.85 (Claude-4-Sonnet), up to **66× faster** than native LLM function-calling. Distinguishing dimensions vs ours: (a) external scheduler, no architectural change to LLM; (b) embodied robotics; (c) system demonstration; (d) **outgoing** robot commands on parallel channels — direction of asynchrony is reversed from ours (we stream incoming diagnostics).

```bibtex
@misc{gong2025ghostshell,
  title        = {GhostShell: Streaming LLM Function Calls for
                  Concurrent Embodied Programming},
  author       = {Gong, Jian and Huang, Youwei and Yuan, Bo and Zhu, Ming and
                  Liao, Zhou and Liang, Jianhang and Zhan, Juncheng and
                  Wang, Jinke and Shu, Hang and Xiong, Mingyue and
                  Ye, Yanjun and Zu, Yufan and Zhou, Yang and Ding, Yihan and
                  Chen, Xuannian and Lu, Xingyu and Ban, Runjie and
                  Huang, Bingchao and Liu, Fusen},
  year         = {2025},
  eprint       = {2508.05298},
  archivePrefix= {arXiv},
  primaryClass = {cs.RO},
  howpublished = {arXiv:2508.05298}
}
```

---

## Methodology / evaluation

### Bjarnason, Silva & Monperrus, *On Randomness in Agentic Evals* (Feb 2026)
**Methodological backbone for our power analysis.** ~60 000 agent trajectories across three models × two scaffolds on SWE-bench Verified. Headline findings: single-run pass@1 estimates **vary by 2.2–6.0 pp**; standard deviations **exceed 1.5 pp even at T=0**; trajectories diverge within the first few percent of tokens. Many claimed 2–3 pp improvements in the SWE-bench literature are statistically indistinguishable from noise. Recommends (1) multiple runs per task, (2) explicit power analysis, (3) reporting pass@k *and* pass^k (probability every of k attempts passes) as a consistency metric.

```bibtex
@misc{bjarnason2026randomness,
  title        = {On Randomness in Agentic Evals},
  author       = {Bjarnason, Bjarni Haukur and Silva, Andr\'{e} and
                  Monperrus, Martin},
  year         = {2026},
  eprint       = {2602.07150},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  howpublished = {arXiv:2602.07150}
}
```

---

## LSP as RL signal (orthogonal but adjacent)

### Zhang et al., *RL from Compiler and Language Server Feedback* (2025)
Uses LSP and compiler diagnostics as RL reward. Orthogonal to our work: their contribution is policy shaping; ours is in-context delivery form.

```bibtex
@article{zhang2025rlcls,
  title   = {Reinforcement Learning from Compiler and Language Server Feedback},
  author  = {Zhang, Yifan and others},
  journal = {arXiv preprint arXiv:2510.22907},
  year    = {2025},
  url     = {https://arxiv.org/abs/2510.22907}
}
```

### Gehring et al., *RLEF* (2024)
Execution feedback as RL signal. Related prior art to Zhang et al.

```bibtex
@article{gehring2024rlef,
  title   = {{RLEF}: Grounding Code {LLM}s in Execution Feedback with Reinforcement Learning},
  author  = {Gehring, Jonas and others},
  journal = {arXiv preprint arXiv:2410.02089},
  year    = {2024},
  url     = {https://arxiv.org/abs/2410.02089}
}
```

---

## Coding-agent substrate

### DeepSWE (Together AI / Agentica, 2025)
Qwen3-32B RL-post-trained on R2E-Gym; 59% SWE-bench Verified w/ TTS, 42.2% pass@1. External reference baseline. Open weights, dataset, training code.

```bibtex
@misc{deepswe2025,
  title  = {{DeepSWE}: Training a Fully Open-sourced, State-of-the-Art Coding Agent by Scaling {RL}},
  author = {{Together AI and Agentica}},
  year   = {2025},
  month  = jul,
  url    = {https://www.together.ai/blog/deepswe}
}
```

---

## Benchmarks

### SWE-bench Verified (Jimenez et al., OpenAI)
Primary benchmark. 500 human-verified tasks from 12 Python repos.

```bibtex
@misc{swebenchverified,
  title  = {{SWE}-bench Verified},
  author = {Jimenez, Carlos E. and others},
  year   = {2024},
  url    = {https://www.swebench.com/verified.html}
}
```

### *The SWE-Bench Illusion* (2025)
Contamination audit. Cited as a contamination-defense motivation; informs the held-out-subset design.

```bibtex
@article{swebenchillusion2025,
  title   = {The {SWE}-Bench Illusion: When State-of-the-Art {LLM}s Remember Instead of Reason},
  journal = {arXiv preprint arXiv:2506.12286},
  year    = {2025},
  url     = {https://arxiv.org/abs/2506.12286}
}
```

---

## Training methodology

### Thinking Machines Lab, *On-Policy Distillation* (Oct 2025)
Student matches teacher distribution under student rollouts; dense per-token supervision. Mechanism for E (distilled stream model).

```bibtex
@misc{tml2025distillation,
  title  = {On-Policy Distillation},
  author = {{Thinking Machines Lab}},
  year   = {2025},
  month  = oct,
  url    = {https://thinkingmachines.ai/blog/on-policy-distillation/}
}
```

### Thinking Machines Lab, *Modular Manifolds* (Sept 2025)
Manifold-constrained weight optimization. Not load-bearing for v1 but adjacent.

```bibtex
@misc{tml2025manifolds,
  title  = {Modular Manifolds},
  author = {Bernstein, Jeremy},
  year   = {2025},
  month  = sep,
  url    = {https://thinkingmachines.ai/blog/modular-manifolds/}
}
```

---

## Hardware / tooling

### NVIDIA DGX Spark / GB10 Grace Blackwell
Local hardware target: 128 GB unified LPDDR5X, ~1 PFLOP FP4.

```bibtex
@misc{nvidiadgxspark,
  title  = {{NVIDIA DGX Spark}},
  author = {{NVIDIA}},
  year   = {2025},
  url    = {https://www.nvidia.com/en-us/products/workstations/dgx-spark/}
}
```

### Pyrefly (Meta)
Rust-based Python type checker. v1.0 released May 2026. Daemon mode, incremental analysis.

```bibtex
@misc{pyrefly,
  title  = {{Pyrefly}: A Fast Type Checker and {IDE} Experience for {P}ython},
  author = {{Meta}},
  year   = {2026},
  url    = {https://pyrefly.org/}
}
```

### seal-rg/streaming
Code release accompanying Su, Yang, Li, Geiping. Source of `stream-qwen3-8b` and `stream-qwen3.5-27b` checkpoints. No LICENSE file as of 2026-05-26 (HF weights themselves are Apache-2.0); to be clarified with authors before publication.

```bibtex
@misc{sealrgstreaming,
  title  = {seal-rg/streaming: Multi-Stream {LLM} Training Code},
  author = {Su and Yang and Li and Geiping, Jonas},
  year   = {2026},
  url    = {https://github.com/seal-rg/streaming}
}
```

---

## Tooling references (non-academic)

### Claude Code LSP integration (Dec 2025)
Native LSP support with edit-hook diagnostics. Production precedent for Condition C.

```bibtex
@misc{claudecodelsp2025,
  title  = {{LSP} Tools: Bringing {IDE} Intelligence to {C}laude {C}ode},
  author = {{Anthropic}},
  year   = {2025},
  month  = dec,
  url    = {https://zircote.com/blog/2025/12/lsp-tools-plugin-for-claude-code/}
}
```
