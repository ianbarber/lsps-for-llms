# scripts/

Everything that produced a number in PAPER.md, indexed by section:

- `synth_tasks.py` — §3: the 14-task suite + self-verifier (`python scripts/synth_tasks.py`
  checks every task fails behaviourally and fires bug-relevant pyrefly diagnostics).
- `synth_acd.py` — §5/§6: the condition runner. Paper-condition -> flag recipes are in
  the root README and in `analysis/stats.py`'s docstring.
- `i_eval.py`, `i_eval_revise.py`, `i_eval_cd.py` — §4 isolation probes
  (forward injection, backward revision, toy efficiency).
- `harvest_sft.py` -> `d_sft.py` — §6.2 self-distillation pipeline (harvest resolved
  deployment-format trajectories with observation-masked labels; LoRA-train).
- `rebench_select.py`, `rebench_smoke.py`, `agent_swe.py` — Appendix A real-repo
  pipeline (SWE-rebench selection, native provisioning, agent runs).
- `analysis/stats.py` — reproduces every headline table/statistic from `runs/`.
