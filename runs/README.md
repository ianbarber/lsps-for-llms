# runs/ — committed result data

Every file here backs a specific paper section. Finalized JSONs have shape
`{"rows": {"<cond>": [row, ...]}}`; each row carries task, seed, resolved,
token/test counts, and the full event trace (including every delivered
diagnostic's text). NOTE: condition keys inside files are the harness letters
(A/C/D) — the paper-name mapping and exact flag recipes are in the root README
and `scripts/analysis/stats.py`.

| file | paper | contents |
|---|---|---|
| agent/synth_power.json | 5.1, 5.2 | A, C-lazy, **D-naive** (log name: D-tuned), seeds 0-5 |
| agent/synth_ac_s6.json | 5.1 | A, C-lazy, seeds 6-11 |
| agent/synth_ceager.json / _s6 | 5.1 | C-eager, seeds 0-5 / 6-11 |
| agent/synth_dplain.json / _s6 | 5.1, 5.2 | D-plain, seeds 0-5 / 6-11 (_s6 merged from checkpoint+resume; see log 2026-06-02/03) |
| agent/synth_dgate.json / _s6 | 5.1, 5.2 | D-gate, seeds 0-5 / 6-11 |
| agent/synth_dgate_rich.json, synth_ceager_rich.json | 6.1 | rich-signal arms, seeds 0-5 |
| agent/synth_dgate_sft.json, synth_a_sft.json | 6.2 | adapter evals (D-gate+SFT, A+SFT), seeds 0-5 |
| isolation/forward_r0.json | 4 | forward injection 16/16 vs 0/16 |
| isolation/revise_r0b.json | 4 | backward revision 10/10 vs 3/10 |
| isolation/toy_efficiency_r0c.json | 4 | toy live-vs-sync efficiency (70 vs 139 tok) |
| rebench/candidates.jsonl, provisioned.jsonl, provision_report.json | App. A | SWE-rebench selection: 25 -> 8 provisioned -> 4 well-formed |
| rebench/swe_acd.json, swe_region.json | App. A | real-repo agent runs (incl. the 0/3 oracle-localized result) |
| adapters/dgate_sft_v2/ | 6.2 | the LoRA adapter (62 demos, observation-masked) |

Reproduce all headline numbers: `python scripts/analysis/stats.py` from the repo root.
