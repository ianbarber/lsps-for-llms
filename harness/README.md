# harness/

Evaluation harness. Loads filtered SWE-bench Verified tasks (and held-out SWE-rebench
subset), runs the frozen scaffold under a chosen condition config, enforces matched
caps, and captures full trajectories: agent tokens with timestamps, every LSP
snapshot, every diagnostic event, and the final patch. Scoring delegates to
SWE-bench's own test runner.
