# G4 — payload-equivalence (SHA-256) audit

**Verdict: PASS** (10/10 cases byte-identical across B/C/C'/D)

- pyrefly: `pyrefly 1.0.0` (pinned 1.0.0, sha `2362c071`)
- conditions audited: B, C, C', D
- payload: canonical `(severity, line, code, message)` tuples, top-K=10 by recency-of-edited-region, deterministic JSON

Each case drives a real pyrefly daemon `prefix → edit` and routes the raw diagnostics through every condition's shared `normalize_payload` path. The SHA-256 is over those payload bytes; equality is the matched-information-content guarantee (RQ1, §13).

| Case | Diags | Match | SHA-256 (shared) | Note |
|---|---:|:---:|---|---|
| `01_bad_argument_type` | 1 | yes | `d85a369d8fb60e14` | pass a str where an int param is expected |
| `02_bad_assignment` | 1 | yes | `679f9611ec8072a4` | assign int to a str-annotated name |
| `03_bad_return_type` | 1 | yes | `83887a25534cf353` | return a str from an int-annotated function |
| `04_two_errors_topk_order` | 2 | yes | `55c292819d9d0c22` | two bad-assignments; recency ranking puts line 2 first |
| `05_attribute_error` | 1 | yes | `59e3c2a1f2b3b8d3` | access a missing attribute on an int |
| `06_undefined_name` | 1 | yes | `613f811c3caf116d` | reference a name that is not defined |
| `07_clean_edit_empty_payload` | 0 | yes | `4f53cda18c2baa0c` | well-typed edit; expected zero diagnostics → empty payload |
| `08_list_element_type` | 1 | yes | `630c77f9052cc613` | str element in a list[int] |
| `09_call_missing_arg` | 1 | yes | `6ba7c946bbe1faee` | call with a missing required argument |
| `10_dict_value_type` | 1 | yes | `e7585b2f2635318e` | str value in a dict[str, int] |

