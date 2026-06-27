# Bibliography

References for *Making a Language Server Pay Off for a Coding Agent: Train It to Retrieve
Cheaply*. The canonical BibTeX source is `docs/bibliography_efficiency.bib`.

---

## Substrate and tooling

### Su, Yang, Li & Geiping, *Multi-Stream LLMs* (2026)
Architectural substrate for the stream harness. Releases `stream-qwen3-8b` and
`stream-qwen3.5-27b` checkpoints. The harness in `scaffold/stream_agent.py` builds on this,
but the paper's contribution is retrieval efficiency, not the stream architecture.

```bibtex
@article{su2026multistream,
  title   = {Multi-Stream {LLM}s: Unblocking Language Models with Parallel Streams of Thoughts, Inputs and Outputs},
  author  = {Su, Guinan and Yang, Yanwu and Li, Xueyan and Geiping, Jonas},
  journal = {arXiv preprint arXiv:2605.12460},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.12460}
}
```

### seal-rg/streaming (2026)
Training code and checkpoints for Su et al.

```bibtex
@misc{sealrgstreaming,
  title  = {seal-rg/streaming: Multi-Stream {LLM} Training Code},
  author = {Su, Guinan and Yang, Yanwu and Li, Xueyan and Geiping, Jonas},
  year   = {2026},
  url    = {https://github.com/seal-rg/streaming}
}
```

### Pyrefly (Meta, 2026)
Rust-based Python type checker and LSP. Used as the real go-to-definition resolver.

```bibtex
@misc{pyrefly,
  title  = {{Pyrefly}: A Fast Type Checker and {IDE} Experience for {P}ython},
  author = {{Meta}},
  year   = {2026},
  url    = {https://pyrefly.org/}
}
```

---

## Method: on-policy imitation and cost-aware tool use

### Agarwal et al., *On-Policy Distillation of Language Models* (ICLR 2024)
GKD: distillation under the student's own rollout distribution.

```bibtex
@inproceedings{agarwal2024gkd,
  title     = {On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes},
  author    = {Agarwal, Rishabh and Vieillard, Nino and Zhou, Yongchao and Stanczyk, Piotr and Ramos, Sabela and Geist, Matthieu and Bachem, Olivier},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024},
  note      = {arXiv:2306.13649}
}
```

### Ross, Gordon & Bagnell, *DAgger* (AISTATS 2011)
Dataset Aggregation: roll out the current policy and relabel with an expert.

```bibtex
@inproceedings{ross2011dagger,
  title     = {A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning},
  author    = {Ross, St{\'e}phane and Gordon, Geoffrey J. and Bagnell, J. Andrew},
  booktitle = {Proceedings of the 14th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year      = {2011},
  note      = {PMLR v15:627--635}
}
```

### Ross & Bagnell, *AggreVaTe* (arXiv 2014)
Cost-aware interactive imitation learning.

```bibtex
@article{ross2014aggrevate,
  title   = {Reinforcement and Imitation Learning via Interactive No-Regret Learning},
  author  = {Ross, St{\'e}phane and Bagnell, J. Andrew},
  journal = {arXiv preprint arXiv:1406.5979},
  year    = {2014}
}
```

### Li et al., *Revisiting DAgger in the Era of LLM-Agents* (2026)
Applies DAgger-style on-policy imitation to tool-using LLM agents.

```bibtex
@article{li2026revisitingdagger,
  title   = {Revisiting {DAgger} in the Era of {LLM}-Agents},
  author  = {Li, Changhao and Qiang, Rushi and Huang, Jiawei and Gao, Chenxiao and Zhang, Chao and He, Niao and Dai, Bo},
  journal = {arXiv preprint arXiv:2605.12913},
  year    = {2026}
}
```

### Zelikman et al., *STaR* (NeurIPS 2022)
Bootstraps reasoning from the model's own generated rationales; analogous in spirit to
bootstrapping a cost preference from the model's own trajectories.

```bibtex
@inproceedings{zelikman2022star,
  title     = {{STaR}: Bootstrapping Reasoning With Reasoning},
  author    = {Zelikman, Eric and Wu, Yuhuai and Mu, Jesse and Goodman, Noah D.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2022},
  note      = {arXiv:2203.14465}
}
```

### Wang et al., *OTC-PO* (2025)
Teaches models to act efficiently by rewarding fewer tool calls.

```bibtex
@article{wang2025otcpo,
  title   = {Acting Less is Reasoning More! Teaching Model to Act Efficiently},
  author  = {Wang, Hongru and Qian, Cheng and Zhong, Wanjun and Chen, Xiusi and Qiu, Jiahao and Huang, Shijue and Jin, Bowen and Wang, Mengdi and Wong, Kam-Fai and Ji, Heng},
  journal = {arXiv preprint arXiv:2504.14870},
  year    = {2025},
  note    = {OTC-PO}
}
```

### Huang et al., *IKEA* (2025)
Reinforced internal-external knowledge reasoning for efficient adaptive search.

```bibtex
@article{huang2025ikea,
  title   = {Reinforced Internal-External Knowledge Synergistic Reasoning for Efficient Adaptive Search Agent},
  author  = {Huang, Ziyang and Yuan, Xiaowei and Ju, Yiming and Zhao, Jun and Liu, Kang},
  journal = {arXiv preprint arXiv:2505.07596},
  year    = {2025},
  note    = {IKEA}
}
```

---

## Closest prior work: language servers as RL signal

### Zhang et al., *RLCSF* (2025)
Reinforcement Learning from Compiler and Language Server Feedback. Rewards LSP/compiler
diagnostics during RL. The closest prior work: RLCSF treats LSP feedback as a useful signal;
we find its information redundant on our suites and focus on its retrieval efficiency.

```bibtex
@article{zhang2025rlcsf,
  title   = {Reinforcement Learning from Compiler and Language Server Feedback},
  author  = {Zhang, Yifan and others},
  journal = {arXiv preprint arXiv:2510.22907},
  year    = {2025},
  note    = {RLCSF}
}
```

---

## Hardware note

### NVIDIA DGX Spark / GB10 Grace Blackwell
Local hardware target for the reported runs: 128 GB unified LPDDR5X.

```bibtex
@misc{nvidiadgxspark,
  title  = {{NVIDIA DGX Spark}},
  author = {{NVIDIA}},
  year   = {2025},
  url    = {https://www.nvidia.com/en-us/products/workstations/dgx-spark/}
}
```
