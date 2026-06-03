# training/

SFT pipelines. Houses the shared-ancestor SFT pass, per-condition LoRA-SFT passes
(matched volume across A/B/C/C′/D), the latency-replay reformat that turns
synchronous-teacher rollouts into causally-valid multi-stream training data for D
(§7.4 of the plan), and the on-policy distillation pipeline for stretch condition E.
The causal-validity gate that masks the teacher's sync diagnostic from the D prefix
lives here.
