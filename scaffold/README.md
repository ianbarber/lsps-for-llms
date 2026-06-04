# scaffold/ — the agent

- `stream_agent.py` — the non-blocking continuous-stream coding agent. One decode
  stream carrying reasoning + line-range edits + `<test/>`/`<read/>` actions; edits
  apply without yielding, which is what makes a *live* diagnostic channel distinct
  from a synchronous one. Every delivery condition in the paper is a constructor
  flag: condition A/C/D, `c_eager`, `debounce`, `pause_align`, `announce_lsp`,
  `syntax_gate`, `rich_signal`; plus SFT label-mask capture (`sft_input_ids`/
  `sft_labels`, observation tokens masked).
- `mock_env.py` — single-file task environment with the real Pyrefly checker
  (apply-or-reject line edits, behavioural tests, rework accounting). NOTE: the
  pyrefly binary path is currently hardcoded near the top — point it at your own
  pyrefly install to run.
