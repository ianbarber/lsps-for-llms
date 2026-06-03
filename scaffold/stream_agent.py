#!/usr/bin/env python3
"""Continuous-stream coding agent with live (in-flight) LSP feedback — the v0.5
option-D apparatus. The agent generates ONE continuous token stream of reasoning
+ edits; depending on the condition, pyrefly diagnostics about completed edits are
delivered:
  A : never (no feedback)
  C : synchronously — generation PAUSES at the edit boundary, the diagnostic is
      injected, then generation resumes (production post-edit-hook / turn feedback)
  D : live/async — generation does NOT pause; the diagnostic is spliced into the
      stream `latency_tokens` AFTER the edit completes (the model sees it
      mid-work and must look back), exactly the prototype-validated KV splice.

Edit format the model emits (parsed live from the stream):
    <edit path="FILE">
    <<<<<<< SEARCH
    ...old...
    =======
    ...new...
    >>>>>>> REPLACE
    </edit>
and a final  <done/>  to stop.

The agent is environment-agnostic: it calls an env with
  read_file(path) -> str
  apply_edit(path, search, replace) -> (ok: bool, info: str)   # records rework
  pyrefly_diagnostics(path) -> str   # normalized "(sev,line,code,msg)" text, or ""
  run_tests() -> dict  (resolved: bool, ...)
  metrics() -> dict
"""
from __future__ import annotations
import re, ast, time, torch

# Robust SEARCH/REPLACE parser: the 7B inconsistently drops the <<<<<<</=======/>>>>>>>
# markers, so make them optional and terminate the replace at END / blank line / next
# action / EOS. Matches both the canonical aider block and the bare SEARCH/REPLACE form.
EDIT_RE = re.compile(
    r'(?:<<<<<<<\s*)?SEARCH\s*\n(?P<search>.*?)\n(?:END\s*\n)?(?:=======|REPLACE)\s*\n(?P<replace>.*?)'
    r'(?=\n\s*(?:END\b|>>>>>>>|<test\s*/>|<done\s*/>|SEARCH\b)|\Z)', re.S)
DONE_RE = re.compile(r'<done\s*/>')
TEST_RE = re.compile(r'<test\s*/>')
READ_RE = re.compile(r'<read\s+path="(?P<path>[^"]+)"\s*/>')   # gather repo context
# whole-file rewrite protocol — robust for single-function tasks (no SEARCH-match
# brittleness, no irrecoverable corruption: a bad rewrite is just rewritten).
REWRITE_RE = re.compile(r'<rewrite>\s*\n(?P<src>.*?)\n\s*</rewrite>', re.S)
NUMPREFIX_RE = re.compile(r'(?m)^\s*\d+\|\s?')   # strip accidental "  3| " file-view prefixes
# line-range edit: robust for weak models on big files (no string matching).
LINE_EDIT_RE = re.compile(
    r'<edit\s+path="(?P<path>[^"]+)"\s+lines="(?P<start>\d+)(?:\s*-\s*(?P<end>\d+))?"\s*>\s*\n'
    r'(?P<body>.*?)\n?\s*</edit>', re.S)   # END optional: accept lines="N" and lines="N-M"

def _strip_fences(s: str) -> str:
    """Drop a leading ```lang and trailing ``` the model often wraps code in, and
    any leftover NNN| line-number prefixes it copied from the numbered file view."""
    s = s.strip("\n")
    lines = s.split("\n")
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return NUMPREFIX_RE.sub("", "\n".join(lines))
DIAG_OPEN, DIAG_CLOSE = "\n‹diag›\n", "\n‹/diag›\n"
TEST_OPEN, TEST_CLOSE = "\n<test_result>\n", "\n</test_result>\n"

SYS = """You are a coding agent fixing a bug in a Python repository. The bug is in file
`{file}`, whose current contents are shown below. Work iteratively.

To EDIT, emit a block in EXACTLY this form (the SEARCH text must match the file verbatim and
be UNIQUE in it; change as few lines as you need — you choose the scope):

SEARCH
def foo(x):
    return x + 0
REPLACE
def foo(x):
    return x + 1
END

To READ another file for context, emit:  <read path="pkg/other.py"/>
After editing, emit <test/> to RUN THE TESTS; you will receive the results. If tests fail,
read the failure, make another SEARCH/REPLACE/END edit, and <test/> again — keep iterating
until they pass. A static analyzer may also surface diagnostics; use them to catch mistakes
early. Emit <done/> only once the tests pass. Reason briefly between actions."""

SYS_LINE = """You are a coding agent fixing a bug in a Python repository. The bug is in file
`{file}`, shown below with line numbers (`NNN| code`). Work iteratively.

To EDIT, replace a range of lines by emitting EXACTLY:
<edit path="{file}" lines="START-END">
<the new code for those lines, WITHOUT line-number prefixes>
</edit>
START-END are inclusive 1-based line numbers from the numbered view. Include proper
indentation. You choose the range — one line or many. Do not wrap the code in ``` fences.
To READ another file for context: <read path="pkg/other.py"/>
After editing, emit <test/> to RUN THE TESTS; you'll get the results, then a fresh numbered
view of the file (line numbers may have shifted — always use the latest view). Keep iterating:
edit -> <test/> -> fix -> <test/> until tests pass, then emit <done/>. Reason briefly between
actions. A static analyzer may also surface diagnostics; use them to catch mistakes early."""

SYS_REWRITE = """You are a coding agent fixing a bug in file `{file}`. Work iteratively.
To EDIT, output the ENTIRE corrected file inside a rewrite block:

<rewrite>
def foo(x):
    return x + 1
</rewrite>

Then emit <test/> to RUN THE TESTS; you will receive the results. If they fail, read the
failure, output another COMPLETE <rewrite>...</rewrite> (the whole file), and <test/> again —
keep iterating until they pass. A static analyzer may also surface diagnostics mid-work; use
them to catch mistakes early. Output ONLY code inside <rewrite> — no line numbers, no markdown
fences. Emit <done/> only once the tests pass. Reason briefly between actions."""


class StreamAgent:
    def __init__(self, model, tok, env, condition="D", latency_tokens=8,
                 max_new_tokens=3000, max_tests=8, max_turns=12, max_bad=5,
                 max_reads=6, edit_mode="search", temperature=0.0, seed=0,
                 debounce=0, pause_align=False, announce_lsp=False, c_eager=False,
                 syntax_gate=False, device=None):
        assert condition in ("A", "C", "D")
        assert edit_mode in ("search", "rewrite", "line")
        self.temperature, self.seed = temperature, seed
        # D-delivery tuning: debounce>0 re-queries pyrefly only after the stream has
        # SETTLED `debounce` tokens since the last edit and (if pause_align) at a
        # newline — so transient self-inflicted squiggles the model already fixed are
        # never delivered, and delivery lands at a natural pause not mid-token.
        self.debounce, self.pause_align, self.announce_lsp = debounce, pause_align, announce_lsp
        self.c_eager = c_eager   # C: deliver the diagnostic immediately after each edit (post-edit hook) vs batched at the next yield
        self.syntax_gate = syntax_gate   # D: only deliver a live diagnostic when the file currently PARSES (suppress self-inflicted mid-edit syntax-error squiggles)
        self.model, self.tok, self.env = model, tok, env
        self.cond, self.latency = condition, latency_tokens
        self.max_new, self.max_tests, self.max_turns = max_new_tokens, max_tests, max_turns
        self.max_bad = max_bad   # bail after this many consecutive non-applying / no-op edits
        self.max_reads = max_reads
        self.test_p2p_cap = 5    # in-loop <test/> caps P2P for speed; final resolved runs full
        self.edit_mode = edit_mode
        self.dev = device or model.device

    def _run_tests(self, cap=None):
        """Env-agnostic: TaskEnv accepts max_p2p; MockEnv doesn't."""
        try:
            return self.env.run_tests(max_p2p=cap) if cap is not None else self.env.run_tests()
        except TypeError:
            return self.env.run_tests()

    @staticmethod
    def _turn_obs(last_test, n_edits):
        if last_test is None:
            if n_edits == 0:
                return ("You have not edited the file yet. Make a SEARCH/REPLACE edit to fix "
                        "the bug, then emit <test/> to run the tests.")
            return "Run <test/> to check whether your edit fixed the tests."
        return ("Tests are still failing (see the result above). Make another SEARCH/REPLACE "
                "edit to fix the remaining problem, then <test/> again. If you cannot find the "
                "fix, re-read the failing assertion carefully.")

    @staticmethod
    def _fmt_test(tr):
        if tr.get("resolved"):
            return "ALL TESTS PASS."
        parts = [f"FAIL: F2P {tr.get('f2p_pass','?')}/{tr.get('f2p_total','?')}, "
                 f"P2P {tr.get('p2p_pass','?')}/{tr.get('p2p_total','?')}."]
        for k in ("f2p_summary", "p2p_summary", "failure", "f2p_output", "output"):
            v = tr.get(k)
            if v and isinstance(v, str):
                parts.append(v[:600])
        return "\n".join(parts)

    def _ids(self, text, special=False):
        return self.tok(text, return_tensors="pt", add_special_tokens=special).input_ids.to(self.dev)

    def _file_view(self, path):
        """Current file state, line-numbered — delivered each turn so the model can
        write SEARCH blocks that match (after its own edits the file has changed).
        Condition-NEUTRAL: identical across A/C/D; only the LSP channel differs."""
        try:
            src = self.env.read_file(path)
        except Exception:
            return ""
        lines = src.splitlines() or [""]
        body = "\n".join(f"{i+1:>3}| {ln}" for i, ln in enumerate(lines))
        return f"Current `{path}` (edit against THIS exact text):\n{body}"

    @staticmethod
    def _fmt_diag(diag):
        """Normalize a diagnostics result to text. TaskEnv returns list[dict]
        {severity,line,code,message}; MockEnv returns a pre-formatted string."""
        if isinstance(diag, str):
            return diag
        if not diag:
            return ""
        lines = []
        for d in diag:
            if isinstance(d, dict):
                lines.append(f"[{d.get('severity','error')}] L{d.get('line','?')} "
                             f"{d.get('code','')}: {d.get('message','')}")
            elif isinstance(d, (tuple, list)) and len(d) >= 4:
                lines.append(f"[{d[0]}] L{d[1]} {d[2]}: {d[3]}")
            else:
                lines.append(str(d))
        return "\n".join(lines)

    def _next_token(self, logits):
        if self.temperature and self.temperature > 0:
            probs = torch.softmax(logits.float() / self.temperature, dim=-1)
            return int(torch.multinomial(probs, 1))
        return int(logits.argmax(-1))

    def _prefill(self, input_ids, cache):
        with torch.no_grad():
            o = self.model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        return o.logits[:, -1, :], o.past_key_values

    def run(self, task_prompt, target_file):
        tok = self.tok
        # build the chat prompt; assistant stream is what we generate + parse
        sys_tmpl = {"rewrite": SYS_REWRITE, "line": SYS_LINE}.get(self.edit_mode, SYS)
        sys_text = sys_tmpl.format(file=target_file)
        if self.announce_lsp and self.cond == "D":
            sys_text += ("\nA static type-checker runs continuously as you edit; its diagnostics "
                         "appear INLINE in your stream between ‹diag› and ‹/diag› markers a moment "
                         "after each edit. Treat them as live editor squigglies — when one appears, "
                         "fix the issue it names before moving on.")
        msgs = [{"role": "system", "content": sys_text},
                {"role": "user", "content": task_prompt}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        cache = None
        prompt_ids = self._ids(prompt, special=True)
        logits, cache = self._prefill(prompt_ids, cache)
        in_toks = int(prompt_ids.shape[1])   # input tokens FED to the model (prompt + every observation/splice)

        emitted = ""          # decoded assistant stream so far
        out_ids = []
        out_labels = []       # parallel to out_ids: token id if MODEL-generated (train), -100 if spliced (observation)
        applied_upto = 0      # char index up to which edits are applied
        done_from = 0         # char index from which to look for <done/>
        test_from = 0         # char index from which to look for <test/>
        read_from = 0         # char index from which to look for <read/>
        n_tests = n_edits = n_reads = turns = 0
        fail_streak = 0       # consecutive non-applying / no-op edits (anti-degeneracy)
        file_changed = False  # re-show the numbered file only when it changed (the prompt already has it)
        done_seen = resolved = bailed = False
        last_test = None
        pending = []          # D (immediate): in-flight diag spliced mid-stream (the live channel)
        d_dirty_at = None     # D (debounced): token index of the last edit; re-query after settle
        d_last_delivered = "" # D (debounced): avoid re-splicing an unchanged diagnostic
        c_diag_queue = []     # C: diags delivered as the next USER observation (sync, turn-boundary)
        events = []           # trajectory log
        eos = tok.eos_token_id

        def deliver_turn(obs_text):
            """End the assistant turn, give an observation as a USER message, start a
            new assistant turn. Tool results / sync feedback go here (not raw-spliced
            into the assistant stream — chat models parrot in-stream content). The
            current file view is appended so SEARCH blocks match the live file."""
            nudge = ""
            if fail_streak > 0:
                if self.edit_mode == "line":
                    nudge = ("\n[Your last edit did NOT apply (bad line range or format). Use line "
                             "numbers from the CURRENT numbered view below: "
                             "<edit path=\"...\" lines=\"START-END\"> then the new code (no ``` "
                             "fences, no NNN| prefixes) then </edit>.]")
                else:
                    nudge = ("\n[Your last edit did NOT apply. The SEARCH text must be copied "
                             "VERBATIM from the current file below — exact whitespace, no `END` "
                             "line inside SEARCH. Use one block: SEARCH / lines / REPLACE / lines / END.]")
            nonlocal file_changed
            fv = self._file_view(target_file) if file_changed else ""
            file_changed = False
            body = obs_text + nudge + ("\n\n" + fv if fv else "")
            splice("<|im_end|>\n<|im_start|>user\n" + body + "<|im_end|>\n<|im_start|>assistant\n")

        def splice(text):
            nonlocal logits, cache, emitted, out_ids, in_toks
            ids = self._ids(text, special=False)
            in_toks += int(ids.shape[1])    # spliced observations/diagnostics/turn-scaffold = INPUT tokens
            logits, cache = self._prefill(ids, cache)
            out_ids.extend(ids[0].tolist())
            out_labels.extend([-100] * ids.shape[1])   # spliced = observation = masked for SFT
            emitted = tok.decode(out_ids)   # full faithful decode (per-token concat mangles markers)

        if self.temperature and self.temperature > 0:
            torch.manual_seed(self.seed)
        t = 0
        while t < self.max_new:
            nxt = self._next_token(logits)
            if nxt == eos:
                # EOS = end of THIS turn, not necessarily the task. Continue into a
                # new turn with an observation, unless done/resolved/turn-capped.
                if done_seen or resolved or turns >= self.max_turns:
                    break
                turns += 1
                obs = self._turn_obs(last_test, n_edits)
                if self.cond == "C" and c_diag_queue:
                    obs = "Static analysis after your edits:\n" + "\n".join(c_diag_queue) + "\n" + obs
                    c_diag_queue = []
                deliver_turn(obs)
                events.append({"tok": t, "type": "turn", "n": turns, "via": "eos"})
                if self.cond == "D" and pending:   # live diag spliced into the new assistant turn
                    for _, diag in pending:
                        if diag:
                            splice(DIAG_OPEN + diag + DIAG_CLOSE)
                            events.append({"tok": t, "type": "diag_async", "ondone": True, "text": diag})
                    pending = []
                continue
            out_ids.append(nxt)
            out_labels.append(nxt)   # model-generated action token = trained for SFT
            t += 1
            # advance one token
            logits, cache = self._prefill(torch.tensor([[nxt]], device=self.dev), cache)
            emitted = tok.decode(out_ids)   # full faithful decode each step (markers intact)

            # D (immediate): deliver any pending diagnostics whose latency has elapsed
            if self.cond == "D" and self.debounce == 0 and pending:
                due = [d for d in pending if t >= d[0]]
                for d in due:
                    pending.remove(d)
                    if d[1]:
                        splice(DIAG_OPEN + d[1] + DIAG_CLOSE)
                        events.append({"tok": t, "type": "diag_async", "text": d[1]})

            # D (debounced): after the stream settles `debounce` tokens past the last
            # edit, re-query CURRENT diagnostics (transient self-inflicted ones the model
            # already fixed are gone) and deliver at a pause; only if changed.
            if self.cond == "D" and self.debounce > 0 and d_dirty_at is not None:
                settled = t - d_dirty_at >= self.debounce
                at_pause = (not self.pause_align) or emitted.endswith(("\n", " ", ".", ":")) \
                           or (t - d_dirty_at >= self.debounce * 3)
                # syntax-gate: if the file is mid-edit/unparseable, DEFER (don't deliver a
                # self-inflicted syntax squiggle); retry once the model makes it parse again.
                gated = False
                if settled and at_pause and self.syntax_gate:
                    try:
                        ast.parse(self.env.read_file(target_file))
                    except Exception:
                        gated = True
                        d_dirty_at = t   # reset settle timer; retry when parseable
                if settled and at_pause and not gated:
                    diag = self._fmt_diag(self.env.pyrefly_diagnostics(target_file))
                    if diag and diag != d_last_delivered:
                        splice(DIAG_OPEN + diag + DIAG_CLOSE)
                        d_last_delivered = diag
                        events.append({"tok": t, "type": "diag_debounced", "text": diag})
                    d_dirty_at = None

            # detect a newly-completed whole-file rewrite (rewrite mode)
            if self.edit_mode == "rewrite":
                rm = REWRITE_RE.search(emitted, applied_upto)
                if rm:
                    applied_upto = rm.end()
                    src = NUMPREFIX_RE.sub("", rm["src"])
                    res = self.env.rewrite_file(src)
                    ok = res[0] if isinstance(res, tuple) else res.ok
                    if ok: n_edits += 1; fail_streak = 0
                    else: fail_streak += 1
                    events.append({"tok": t, "type": "rewrite", "ok": ok, "fail_streak": fail_streak})
                    if fail_streak >= self.max_bad:
                        bailed = True; events.append({"tok": t, "type": "bail", "reason": "rewrite_fail_streak"}); break
                    if self.cond in ("C", "D") and ok:
                        diag = self._fmt_diag(self.env.pyrefly_diagnostics(target_file))
                        if self.cond == "C" and diag:
                            c_diag_queue.append(diag)
                            events.append({"tok": t, "type": "diag_sync_queued", "text": diag})
                        elif self.cond == "D" and diag:
                            pending.append((t + self.latency, diag))
                            events.append({"tok": t, "type": "diag_pending", "text": diag})

            # detect a newly-completed line-range edit (line mode)
            if self.edit_mode == "line":
                lm = LINE_EDIT_RE.search(emitted, applied_upto)
                if lm:
                    applied_upto = lm.end()
                    body = _strip_fences(lm["body"])
                    epath, s = lm["path"], int(lm["start"])
                    e = int(lm["end"]) if lm["end"] else s
                    if hasattr(self.env, "apply_line_edit"):
                        res = self.env.apply_line_edit(epath, s, e, body)
                        ok = res.ok if hasattr(res, "ok") else res[0]
                        info = res.reason if hasattr(res, "reason") else (res[1] if isinstance(res, tuple) else "")
                    else:
                        ok, info = False, "line edits unsupported by env"
                    if ok: n_edits += 1; fail_streak = 0
                    else: fail_streak += 1
                    file_changed = True
                    events.append({"tok": t, "type": "line_edit", "path": epath, "lines": f"{s}-{e}",
                                   "ok": ok, "info": str(info)[:80], "fail_streak": fail_streak})
                    if fail_streak >= self.max_bad:
                        bailed = True; events.append({"tok": t, "type": "bail", "reason": "edit_fail_streak"}); break
                    if self.cond == "D" and self.debounce > 0 and ok:
                        d_dirty_at = t   # settle timer; re-query at delivery (debounced)
                    elif self.cond in ("C", "D") and ok:
                        diag = self._fmt_diag(self.env.pyrefly_diagnostics(epath))
                        if self.cond == "C" and diag:
                            if self.c_eager:
                                turns += 1
                                deliver_turn("Static analysis after your edit:\n" + diag)
                                events.append({"tok": t, "type": "diag_eager", "text": diag})
                                continue   # eager post-edit hook: yield right after the edit
                            c_diag_queue.append(diag)
                            events.append({"tok": t, "type": "diag_sync_queued", "text": diag})
                        elif self.cond == "D" and diag:
                            pending.append((t + self.latency, diag))
                            events.append({"tok": t, "type": "diag_pending", "text": diag})

            # detect a newly-completed edit (search/replace mode)
            m = EDIT_RE.search(emitted, applied_upto) if self.edit_mode == "search" else None
            if m:
                applied_upto = m.end()
                noop = m["search"].strip() == m["replace"].strip()
                if noop:
                    ok, info = False, "no-op (search == replace)"
                else:
                    res = self.env.apply_edit(target_file, m["search"], m["replace"])
                    ok = res[0] if isinstance(res, tuple) else res.ok
                    info = (res[1] if isinstance(res, tuple) else res.reason)
                if ok: n_edits += 1; fail_streak = 0
                else: fail_streak += 1
                file_changed = True
                events.append({"tok": t, "type": "edit", "path": target_file, "ok": ok,
                               "info": str(info)[:80], "fail_streak": fail_streak})
                if fail_streak >= self.max_bad:
                    bailed = True
                    events.append({"tok": t, "type": "bail", "reason": "edit_fail_streak"})
                    break
                if self.cond in ("C", "D") and ok:
                    diag = self._fmt_diag(self.env.pyrefly_diagnostics(target_file))
                    if self.cond == "C" and diag:
                        # sync: queue for the next USER observation (turn boundary)
                        c_diag_queue.append(diag)
                        events.append({"tok": t, "type": "diag_sync_queued", "text": diag})
                    elif self.cond == "D" and diag:
                        # live: splice into the assistant stream latency_tokens later
                        pending.append((t + self.latency, diag))

            # detect <test/> -> run tests, splice results back (the try-and-correct
            # loop; all conditions get test feedback — only LSP delivery differs)
            tm = TEST_RE.search(emitted, test_from)
            if tm and n_tests < self.max_tests:
                test_from = tm.end()
                n_tests += 1
                tr = self._run_tests(cap=self.test_p2p_cap)
                last_test = tr; resolved = bool(tr.get("resolved"))
                events.append({"tok": t, "type": "test", "n": n_tests, "resolved": resolved})
                obs = "<test_result>\n" + self._fmt_test(tr) + "\n</test_result>"
                if resolved:
                    obs += "\nTests pass — emit <done/> if the fix is complete."
                elif self.cond == "C" and c_diag_queue:
                    obs += "\nStatic analysis after your edits:\n" + "\n".join(c_diag_queue)
                    c_diag_queue = []
                turns += 1
                deliver_turn(obs)
                events.append({"tok": t, "type": "turn", "n": turns, "via": "test"})
                continue

            # detect <read path="..."/> -> deliver that file's contents (gather repo
            # context). Blocking in all conditions (you can't continue without it).
            rdm = READ_RE.search(emitted, read_from)
            if rdm and n_reads < self.max_reads:
                read_from = rdm.end(); n_reads += 1
                rpath = rdm["path"]
                try:
                    content = self.env.read_file(rpath)
                    view = content if len(content) <= 4000 else content[:4000] + "\n... (truncated)"
                    obs = f"<file path=\"{rpath}\">\n{view}\n</file>"
                except Exception as e:
                    obs = f"Could not read {rpath}: {type(e).__name__}: {e}"
                turns += 1
                deliver_turn(obs)
                events.append({"tok": t, "type": "read", "n": n_reads, "path": rpath})
                continue

            dm = DONE_RE.search(emitted, done_from)
            if dm:
                done_from = dm.end()
                done_seen = True
                # In D, a 'done' while a diagnostic about the last edit is still in
                # flight: deliver it live and let the model react before stopping
                # (the "I'm done — oh wait, a squiggle appeared" case). Only truly
                # stop once done AND nothing pending.
                if self.cond == "D" and pending:
                    for dl, diag in pending:
                        if diag:
                            splice(DIAG_OPEN + diag + DIAG_CLOSE)
                            events.append({"tok": t, "type": "diag_async", "ondone": True, "text": diag})
                    pending = []
                    continue
                break

        # flush any straggler D diagnostics (info content matched vs C), recorded
        for _, diag in pending:
            if diag:
                splice(DIAG_OPEN + diag + DIAG_CLOSE)
                events.append({"tok": t, "type": "diag_async", "flushed": True, "text": diag})

        result = self._run_tests(cap=None)   # authoritative: full F2P + full P2P
        prompt_list = prompt_ids[0].tolist()
        sft_input_ids = prompt_list + out_ids
        sft_labels = [-100] * len(prompt_list) + out_labels   # prompt masked; train only on model actions
        return {"condition": self.cond, "resolved": result.get("resolved"),
                "bailed": bailed, "tests": result, "metrics": self.env.metrics(),
                "events": events, "stream": emitted,
                "out_tokens": t, "in_tokens": in_toks, "n_tokens": t,
                "n_edits": n_edits, "n_tests": n_tests, "n_reads": n_reads, "turns": turns,
                "sft_input_ids": sft_input_ids, "sft_labels": sft_labels,
                "n_train_tokens": sum(1 for x in out_labels if x != -100)}
