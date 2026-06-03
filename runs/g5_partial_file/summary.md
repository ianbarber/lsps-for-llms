# G5 — pyrefly partial-file probe (daemon mode)

- pyrefly: `pyrefly 1.0.0` (sha `2362c071`)
- repo: `django__django-17087`
- target: `django/utils/version.py` (3628 chars, 121 lines)
- initialize wall-clock: 2 ms
- initial diag count (clean file): 2

## Round-trip latency (`didChange` -> `publishDiagnostics`)
- N edits: 20 (responded: 20)
- mean:   4.8 ms
- median: 4.6 ms
- p95:    6.2 ms
- p99:    6.5 ms
- max:    6.5 ms
- 200 ms p95 target: HIT (margin: +193.8 ms)

## Partial-file probe results

| Case | Description | AST parses | Got diags | Diag count | RT (ms) | Recovered clean | Recover RT (ms) |
|---|---|:---:|:---:|---:|---:|:---:|---:|
| `a_trailing_def_open_paren` | trailing `def foo(` with no body | no | yes | 3 | 3 | yes | 2 |
| `b_unclosed_string` | unclosed string literal | no | yes | 3 | 3 | yes | 3 |
| `c_if_colon_no_body` | trailing `if x:` with no body | no | yes | 3 | 3 | yes | 5 |
| `d_mid_statement_attr` | mid-statement truncation inside a function body (no closing) | yes | yes | 2 | 3 | yes | 5 |
| `e_unclosed_call_paren` | trailing unclosed parenthesis in a call | no | yes | 4 | 3 | yes | 3 |

- pyrefly returns diagnostics on at least one broken state: yes
- diagnostic volume bounded across all cases: yes
- pyrefly recovers cleanly after all broken states: yes

## Parse-validity gate recommendation

**No parse-validity gate required.** Pyrefly is tolerant: it returns bounded diagnostics on every broken state we probed and recovers cleanly when valid syntax is restored. Snapshot any state.

