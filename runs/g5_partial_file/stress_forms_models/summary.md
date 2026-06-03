# G5 — pyrefly partial-file probe (daemon mode)

- pyrefly: `pyrefly 1.0.0` (sha `2362c071`)
- repo: `django__django-17087`
- target: `django/forms/models.py` (60677 chars, 1674 lines)
- initialize wall-clock: 2 ms
- initial diag count (clean file): 97

## Round-trip latency (`didChange` -> `publishDiagnostics`)
- N edits: 20 (responded: 20)
- mean:   21.0 ms
- median: 20.1 ms
- p95:    21.3 ms
- p99:    36.3 ms
- max:    36.3 ms
- 200 ms p95 target: HIT (margin: +178.7 ms)

## Partial-file probe results

| Case | Description | AST parses | Got diags | Diag count | RT (ms) | Recovered clean | Recover RT (ms) |
|---|---|:---:|:---:|---:|---:|:---:|---:|
| `a_trailing_def_open_paren` | trailing `def foo(` with no body | no | yes | 98 | 20 | yes | 20 |
| `b_unclosed_string` | unclosed string literal | no | yes | 98 | 35 | yes | 20 |
| `c_if_colon_no_body` | trailing `if x:` with no body | no | yes | 98 | 20 | yes | 35 |
| `d_mid_statement_attr` | mid-statement truncation inside a function body (no closing) | yes | yes | 97 | 20 | yes | 20 |
| `e_unclosed_call_paren` | trailing unclosed parenthesis in a call | no | yes | 99 | 20 | yes | 20 |

- pyrefly returns diagnostics on at least one broken state: yes
- diagnostic volume bounded across all cases: yes
- pyrefly recovers cleanly after all broken states: yes

## Parse-validity gate recommendation

**No parse-validity gate required.** Pyrefly is tolerant: it returns bounded diagnostics on every broken state we probed and recovers cleanly when valid syntax is restored. Snapshot any state.

