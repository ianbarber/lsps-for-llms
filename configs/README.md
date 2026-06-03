# configs/

Per-condition YAML configs for the five comparison arms (A, B, C, C′, D) and stretch
condition E. Holds training hyperparameters (quantization, LoRA rank, sequence length,
LR schedule), evaluation-harness configs (token/turn/wall-clock caps, seeds), and
LSP-delivery-layer settings (snapshot cadence, debounce, payload K). One file per
condition; shared defaults composed via includes.
