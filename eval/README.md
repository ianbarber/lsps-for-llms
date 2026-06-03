# eval/

Benchmark adapters. Wraps SWE-bench Verified (primary), SWE-Gym (training-trajectory
source, disjoint from Verified), SWE-rebench (post-2025 contamination-free held-out
subset for the L4 generalisation check), and HumanEval / MBPP (used by the L0
single-stream-degradation gate). Each adapter exposes a uniform task interface to the
harness.
