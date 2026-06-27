#!/usr/bin/env python3
"""Continuous-stream coding agent (efficiency-recipe edition).

The agent generates one token stream of reasoning and edits under condition A
(no inline diagnostic feedback). The final recipe trains it to prefer the cheap
`<defn>` / `<findrefs>` go-to-definition actions over expensive whole-file
`<read>`.

Supported edit formats (parsed live):
  SEARCH/REPLACE/END block
  <edit path="FILE" lines="START-END"> block

Supported retrieval/test actions:
  <read path="FILE"/>
  <defn sym="NAME"/>
  <findrefs sym="NAME"/>
  <test/>
  <done/>

The agent is environment-agnostic and expects the env to provide:
  read_file(path) -> str
  apply_edit(path, search, replace) -> (ok: bool, info: str)
  apply_line_edit(path, start, end, new_text) -> (ok: bool, info: str)
  goto_definition(sym) -> (src: str|None, path: str|None)
  find_references(sym) -> list[str]|None
  lsp_definition(sym) -> (src: str|None, path: str|None)   [optional, for --lsp-defn]
  run_tests() -> dict
  metrics() -> dict
"""
from __future__ import annotations
import re, torch

# Robust SEARCH/REPLACE parser: the 7B inconsistently drops the <<<<<<</=======/>>>>>>>
# markers, so make them optional and terminate the replace at END / blank line / next
# action / EOS. Matches both the canonical aider block and the bare SEARCH/REPLACE form.
EDIT_RE = re.compile(
    r'(?:<<<<<<<\s*)?SEARCH\s*\n(?P<search>.*?)\n(?:END\s*\n)?(?:=======|REPLACE)\s*\n(?P<replace>.*?)'
    r'(?=\n\s*(?:END\b|>>>>>>>|<test\s*/>|<done\s*/>|SEARCH\b)|\Z)', re.S)
DONE_RE = re.compile(r'<done\s*/>')
TEST_RE = re.compile(r'<test\s*/>')
READ_RE = re.compile(r'<read\s+path="(?P<path>[^"]+)"\s*/>')   # gather repo context
# PULL LSP actions (efficiency-as-policy experiment): a CHEAP retrieval action the model can ELECT
# instead of reading whole files. Backed by the env's live resolver, with no gold fallback.
FINDREFS_RE = re.compile(r'<findrefs\s+sym="(?P<sym>[^"]+)"\s*/>')   # go-to-references: where is SYM used
DEFN_RE     = re.compile(r'<defn\s+sym="(?P<sym>[^"]+)"\s*/>')       # go-to-definition / hover: SYM's def+signature
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
until they pass. Emit <done/> only once the tests pass. Reason briefly between actions."""

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
edit -> <test/> -> fix -> <test/> until tests pass, then emit <done/>. Reason briefly between actions."""

SYS_LINE_MULTI = """You are a coding agent fixing a bug that SPANS MULTIPLE FILES in a Python
repository. The EDITABLE files are: {files}. Each is shown below with line numbers (`NNN| code`).
The same fix is likely needed in SEVERAL of them — you must fix EVERY affected site across all
the files, or the tests will still fail. Work iteratively.

To EDIT a file, emit EXACTLY:
<edit path="pkg/file.py" lines="START-END">
<the new code for those lines, WITHOUT line-number prefixes>
</edit>
Use the path of the file you are editing and ITS OWN 1-based line numbers from the numbered view.
After each edit you receive a fresh numbered view of every editable file (line numbers shift —
always use the latest view for the file you are editing). To READ a non-editable file for
context: <read path="pkg/other.py"/>. After your edits, emit <test/> to RUN THE TESTS; you'll get
the results. Keep iterating: edit the affected sites -> <test/> -> fix any remaining site ->
<test/> until tests pass, then emit <done/>. Reason briefly between actions."""


class StreamAgent:
    def __init__(self, model, tok, env,
                 max_new_tokens=3000, max_tests=8, max_turns=12, max_bad=5,
                 max_reads=6, edit_mode="search", temperature=0.0, seed=0,
                 force_lsp=False, relabel=False, device=None,
                 use_lsp_defn=False, advertised_symbols=None):
        assert edit_mode in ("search", "line")
        self.temperature, self.seed = temperature, seed
        self.model, self.tok, self.env = model, tok, env
        self.max_new, self.max_tests, self.max_turns = max_new_tokens, max_tests, max_turns
        self.max_bad = max_bad   # bail after this many consecutive non-applying / no-op edits
        self.max_reads = max_reads
        self.force_lsp = force_lsp   # deny <read> of non-editable files -> force the model onto <defn>/<findrefs>
        self.relabel = relabel   # on-policy relabel: drop the read-attempt prefix so the model's own <defn> is trained first
        self.use_lsp_defn = use_lsp_defn   # back <defn> with a LIVE pyrefly LSP daemon (env.lsp_definition) instead of AST
        self.advertised_symbols = advertised_symbols or []   # symbols the model may query when reads are blocked
        self.test_p2p_cap = 5    # in-loop <test/> caps P2P for speed; final resolved runs full
        self.edit_mode = edit_mode
        self.dev = device or model.device

    def _resolve_defn(self, sym):
        """REAL go-to-definition: ask the env's resolver over the LIVE workspace.
        With use_lsp_defn, try a LIVE pyrefly LSP daemon first; otherwise use the AST resolver.
        No gold/oracle fallback."""
        if self.use_lsp_defn and hasattr(self.env, "lsp_definition"):
            src, _ = self.env.lsp_definition(sym)
            if src:
                return src
            # LSP miss/timeout: fall through to the AST resolver (never hang on the daemon)
        if hasattr(self.env, "goto_definition"):
            src, _ = self.env.goto_definition(sym)
            if src:
                return src
        return None

    def _resolve_refs(self, sym):
        """REAL find-references via the env's resolver. No oracle fallback."""
        if hasattr(self.env, "find_references"):
            refs = self.env.find_references(sym)
            if refs:
                return refs
        return None

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
        write SEARCH blocks that match (after its own edits the file has changed)."""
        try:
            src = self.env.read_file(path)
        except Exception:
            return ""
        lines = src.splitlines() or [""]
        if len(lines) > 250:   # degenerate bloat guard: never re-feed a runaway file in full
            shown = lines[:250]
            body = "\n".join(f"{i+1:>3}| {ln}" for i, ln in enumerate(shown))
            body += f"\n... (file has {len(lines)} lines; truncated — it has grown far beyond the original)"
        else:
            body = "\n".join(f"{i+1:>3}| {ln}" for i, ln in enumerate(lines))
        return f"Current `{path}` (edit against THIS exact text):\n{body}"

    def _next_token(self, logits):
        if self.temperature and self.temperature > 0:
            probs = torch.softmax(logits.float() / self.temperature, dim=-1)
            return int(torch.multinomial(probs, 1))
        return int(logits.argmax(-1))

    def _prefill(self, input_ids, cache):
        with torch.no_grad():
            o = self.model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        return o.logits[:, -1, :], o.past_key_values

    def run(self, task_prompt, target_file, editable=None):
        tok = self.tok
        editable = editable or [target_file]
        # build the chat prompt; assistant stream is what we generate + parse
        if self.edit_mode == "line" and len(editable) > 1:
            sys_text = SYS_LINE_MULTI.format(files=", ".join(f"`{f}`" for f in editable))
        else:
            sys_tmpl = {"line": SYS_LINE}.get(self.edit_mode, SYS)
            sys_text = sys_tmpl.format(file=target_file)
        msgs = [{"role": "system", "content": sys_text},
                {"role": "user", "content": task_prompt}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        cache = None
        prompt_ids = self._ids(prompt, special=True)
        logits, cache = self._prefill(prompt_ids, cache)
        in_toks = int(prompt_ids.shape[1])   # input tokens FED to the model (prompt + every observation)

        emitted = ""          # decoded assistant stream so far
        out_ids = []
        out_labels = []       # parallel to out_ids: token id if MODEL-generated (train), -100 if spliced (observation)
        turn_start_tok = [0]  # out_ids index where the CURRENT model turn began
        relabel_keep_from = [None]  # --relabel: drop everything before this out_ids index from the SFT trace
        applied_upto = 0      # char index up to which edits are applied
        done_from = 0         # char index from which to look for <done/>
        test_from = 0         # char index from which to look for <test/>
        read_from = 0         # char index from which to look for <read/>
        lsp_from = 0          # char index from which to look for <findrefs/>/<defn/>
        n_tests = n_edits = n_reads = n_lsp = turns = 0
        fail_streak = 0       # consecutive non-applying / no-op edits (anti-degeneracy)
        changed_files = set()  # files edited since the last turn -> re-show their numbered view
        done_seen = resolved = bailed = False
        last_test = None
        events = []           # trajectory log
        eos = tok.eos_token_id

        def deliver_turn(obs_text):
            """End the assistant turn, give an observation as a USER message, start a
            new assistant turn. Tool results go here (not raw-spliced into the assistant
            stream — chat models parrot in-stream content). The current file view is
            appended so SEARCH blocks match the live file."""
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
            fv = "\n\n".join(self._file_view(f) for f in sorted(changed_files)) if changed_files else ""
            changed_files.clear()
            body = obs_text + nudge + ("\n\n" + fv if fv else "")
            splice("<|im_end|>\n<|im_start|>user\n" + body + "<|im_end|>\n<|im_start|>assistant\n")

        def splice(text):
            nonlocal logits, cache, emitted, out_ids, in_toks
            ids = self._ids(text, special=False)
            in_toks += int(ids.shape[1])    # spliced observations = INPUT tokens
            logits, cache = self._prefill(ids, cache)
            out_ids.extend(ids[0].tolist())
            out_labels.extend([-100] * ids.shape[1])   # spliced = observation = masked for SFT
            turn_start_tok[0] = len(out_ids)            # a new model turn begins after this observation
            emitted = tok.decode(out_ids)   # full faithful decode (per-token concat mangles markers)

        if self.temperature and self.temperature > 0:
            torch.manual_seed(self.seed)
        t = 0
        CTX_CAP = 24000   # hard context cap (model max 32768): bail before indexing errors
        while t < self.max_new:
            if len(out_ids) + int(prompt_ids.shape[1]) > CTX_CAP:
                bailed = True
                events.append({"tok": t, "type": "bail", "reason": "context_overflow"})
                break
            nxt = self._next_token(logits)
            if nxt == eos:
                # EOS = end of THIS turn, not necessarily the task. Continue into a
                # new turn with an observation, unless done/resolved/turn-capped.
                if done_seen or resolved or turns >= self.max_turns:
                    break
                turns += 1
                obs = self._turn_obs(last_test, n_edits)
                deliver_turn(obs)
                events.append({"tok": t, "type": "turn", "n": turns, "via": "eos"})
                continue
            out_ids.append(nxt)
            out_labels.append(nxt)   # model-generated action token = trained for SFT
            t += 1
            # advance one token
            logits, cache = self._prefill(torch.tensor([[nxt]], device=self.dev), cache)
            emitted = tok.decode(out_ids)   # full faithful decode each step (markers intact)

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
                    if ok:
                        n_edits += 1; fail_streak = 0; changed_files.add(epath)
                    else:
                        fail_streak += 1
                    events.append({"tok": t, "type": "line_edit", "path": epath, "lines": f"{s}-{e}",
                                   "ok": ok, "info": str(info)[:80], "fail_streak": fail_streak,
                                   "replace": body[:300]})
                    if fail_streak >= self.max_bad:
                        bailed = True; events.append({"tok": t, "type": "bail", "reason": "edit_fail_streak"}); break

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
                if ok:
                    n_edits += 1; fail_streak = 0; changed_files.add(target_file)
                else:
                    fail_streak += 1
                events.append({"tok": t, "type": "edit", "path": target_file, "ok": ok,
                               "info": str(info)[:80], "fail_streak": fail_streak,
                               "replace": m["replace"][:300]})
                if fail_streak >= self.max_bad:
                    bailed = True
                    events.append({"tok": t, "type": "bail", "reason": "edit_fail_streak"})
                    break

            # detect <test/> -> run tests, splice results back
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
                turns += 1
                deliver_turn(obs)
                events.append({"tok": t, "type": "turn", "n": turns, "via": "test"})
                continue

            # detect <read path="..."/> -> deliver that file's contents
            rdm = READ_RE.search(emitted, read_from)
            if rdm and n_reads < self.max_reads:
                rpath = rdm["path"]
                # OPSD harvest: deny reads of non-editable files -> force the model onto <defn>/<findrefs>.
                if self.force_lsp and rpath not in editable:
                    read_from = rdm.end()
                    avail = [s for s in self.advertised_symbols if s]
                    eg = avail[0] if avail else "Account"
                    listing = (" Symbols you can query: " + ", ".join(avail) + ".") if avail else ""
                    obs = (f"Reading `{rpath}` is disabled (large file — costly to load in full). To see a "
                           f"symbol's definition/signature emit <defn sym=\"<the symbol name>\"/> "
                           f"(for example <defn sym=\"{eg}\"/>), or <findrefs sym=\"<the symbol name>\"/> "
                           f"to find where it is used.{listing}")
                    turns += 1
                    deliver_turn(obs)
                    # The redirect obs CONTAINS literal self-closing <defn .../>/<findrefs .../> examples.
                    # We MUST advance the action-search cursors past the spliced obs — otherwise the next
                    # search fires on the redirect's EXAMPLE tags instead of the model's OWN action.
                    read_from = lsp_from = test_from = done_from = applied_upto = len(emitted)
                    if self.relabel:
                        relabel_keep_from[0] = len(out_ids)   # start of the model's defn turn
                    events.append({"tok": t, "type": "read_blocked", "path": rpath})
                    continue
                read_from = rdm.end(); n_reads += 1
                try:
                    if len(editable) > 1:
                        # multi-file: hand back a NUMBERED view so a DISCOVERED file is immediately editable
                        obs = self._file_view(rpath)
                    else:
                        content = self.env.read_file(rpath)
                        view = content if len(content) <= 16000 else content[:16000] + "\n... (truncated)"
                        obs = f'<file path="{rpath}">\n{view}\n</file>'
                except Exception as e:
                    obs = f"Could not read {rpath}: {type(e).__name__}: {e}"
                turns += 1
                deliver_turn(obs)
                events.append({"tok": t, "type": "read", "n": n_reads, "path": rpath})
                continue

            # PULL LSP actions: <findrefs sym=.../> and <defn sym=.../>.
            frm = FINDREFS_RE.search(emitted, lsp_from)
            dfm = DEFN_RE.search(emitted, lsp_from)
            lm2 = min([x for x in (frm.start() if frm else None, dfm.start() if dfm else None) if x is not None], default=None)
            if lm2 is not None:
                if frm and frm.start() == lm2:
                    lsp_from = frm.end(); n_lsp += 1; sym = frm["sym"]
                    refs = self._resolve_refs(sym)
                    obs = (f"<findrefs sym=\"{sym}\">\n" +
                           ("\n".join(refs) if refs else "(no references found)") + "\n</findrefs>")
                    events.append({"tok": t, "type": "findrefs", "n": n_lsp, "sym": sym, "hits": len(refs or [])})
                else:
                    lsp_from = dfm.end(); n_lsp += 1; sym = dfm["sym"]
                    defn = self._resolve_defn(sym)
                    obs = (f"<defn sym=\"{sym}\">\n" + (defn if defn else "(no definition found)") + "\n</defn>")
                    events.append({"tok": t, "type": "defn", "n": n_lsp, "sym": sym, "found": bool(defn)})
                turns += 1
                deliver_turn(obs)
                continue

            dm = DONE_RE.search(emitted, done_from)
            if dm:
                done_from = dm.end()
                done_seen = True
                break

        result = self._run_tests(cap=None)   # authoritative: full F2P + full P2P
        prompt_list = prompt_ids[0].tolist()
        keep = relabel_keep_from[0] if (self.relabel and relabel_keep_from[0] is not None) else 0
        kept_ids, kept_labels = out_ids[keep:], out_labels[keep:]
        sft_input_ids = prompt_list + kept_ids
        sft_labels = [-100] * len(prompt_list) + kept_labels   # prompt masked; train only on model actions
        return {"resolved": result.get("resolved"),
                "bailed": bailed, "tests": result, "metrics": self.env.metrics(),
                "events": events, "stream": emitted,
                "out_tokens": t, "in_tokens": in_toks, "n_tokens": t,
                "n_edits": n_edits, "n_tests": n_tests, "n_reads": n_reads, "n_greps": 0, "n_lsp": n_lsp, "turns": turns,
                "sft_input_ids": sft_input_ids, "sft_labels": sft_labels,
                "n_train_tokens": sum(1 for x in kept_labels if x != -100)}
