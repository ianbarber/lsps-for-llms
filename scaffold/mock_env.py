#!/usr/bin/env python3
"""MockEnv: a single-file controlled environment for validating the StreamAgent
mechanism (edit-detect -> real pyrefly -> splice) before wiring the real TaskEnv.
Runs REAL pyrefly (one-shot CLI) so diagnostics are authentic; run_tests execs
the file against an assertion."""
import os, re, json, subprocess, tempfile, multiprocessing as mp

PYREFLY = os.path.expanduser("/home/ianbarber/Projects/Streams/.venv-streams/bin/pyrefly")  # NOTE: point at your own pyrefly binary (pip install pyrefly)
SEV = {0:"error",1:"error",2:"warning",3:"info"}

class MockEnv:
    def __init__(self, buggy_code, test_src, entry_point, force_diag=None):
        self.ws = tempfile.mkdtemp(prefix="mockenv_")
        self.path = "sol.py"
        self.fp = os.path.join(self.ws, self.path)
        self._write(buggy_code)
        with open(os.path.join(self.ws, "pyrefly.toml"), "w") as f:
            f.write("[tool.pyrefly]\nproject-includes = [\"*.py\"]\n")
        subprocess.run([PYREFLY, "init"], cwd=self.ws, capture_output=True, text=True)
        self.force_diag = force_diag   # if set, pyrefly_diagnostics returns this (plumbing test)
        self.test_src, self.ep = test_src, entry_point
        self.chars_written = 0
        self.chars_deleted_after_first = 0
        self.first_write_done = False
        self.n_edits = 0
        self.edit_regions = {}  # line -> count, for edit-cycle accounting

    def _write(self, s):
        with open(self.fp, "w") as f: f.write(s)
    def read_file(self, path=None):
        with open(self.fp) as f: return f.read()
    def list_files(self): return [self.path]

    def apply_edit(self, path, search, replace):
        cur = self.read_file()
        if search not in cur:
            return False, "search not found"
        # rework accounting: chars added now; chars of replaced region count as
        # deletions if we'd already written once (revising prior work)
        self.chars_written += len(replace)
        if self.first_write_done:
            self.chars_deleted_after_first += len(search)
        self.first_write_done = True
        self.n_edits += 1
        self._write(cur.replace(search, replace, 1))
        return True, "ok"

    def rewrite_file(self, new_src):
        """Whole-file rewrite protocol (robust alternative to SEARCH/REPLACE for the
        single-function tasks). Rework = chars of the PRIOR version discarded (diff
        delete+replace on the old side), so revising work you already wrote counts."""
        import difflib
        old = self.read_file()
        new_src = new_src if new_src.endswith("\n") else new_src + "\n"
        self.chars_written += len(new_src)
        if self.first_write_done:
            sm = difflib.SequenceMatcher(None, old, new_src, autojunk=False)
            self.chars_deleted_after_first += sum(
                i2 - i1 for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag in ("replace", "delete"))
        self.first_write_done = True
        self.n_edits += 1
        self._write(new_src)
        return True, "ok"

    def apply_line_edit(self, path, start, end, new_text):
        """Line-range edit (matches TaskEnv.apply_line_edit) so synthetic tasks run
        through the same line-mode agent. Rework = chars of replaced lines removed."""
        cur = self.read_file()
        lines = cur.splitlines(keepends=True)
        n = len(lines)
        if not (1 <= start <= end <= n):
            return False, f"line range {start}-{end} out of bounds (file has {n} lines)"
        removed = "".join(lines[start - 1:end])
        nt = new_text if new_text.endswith("\n") else new_text + "\n"
        new = "".join(lines[:start - 1]) + nt + "".join(lines[end:])
        self.chars_written += len(nt)
        if self.first_write_done:
            self.chars_deleted_after_first += len(removed)
        self.first_write_done = True
        self.n_edits += 1
        self._write(new)
        return True, "ok"

    def pyrefly_diagnostics(self, path=None):
        if self.force_diag is not None:
            return self.force_diag
        try:
            r = subprocess.run([PYREFLY, "check", "--output-format", "json", self.fp],
                               cwd=self.ws, capture_output=True, text=True, timeout=30)
            data = json.loads(r.stdout or "{}")
        except Exception:
            return ""
        diags = data.get("errors", data.get("diagnostics", [])) if isinstance(data, dict) else []
        out = []
        for d in diags[:10]:
            line = d.get("line", d.get("range", {}).get("start", {}).get("line", "?"))
            code = d.get("name", d.get("code", "diag"))
            msg = (d.get("description", d.get("message", "")) or "")[:120]
            out.append(f"[error] L{line} {code}: {msg}")
            if isinstance(line, int): self.edit_regions[line] = self.edit_regions.get(line,0)+1
        return "\n".join(out)

    def run_tests(self):
        code = self.read_file()
        q = mp.Queue()
        def w(q):
            G = {}
            try:
                exec("from typing import *\n"+code, G); exec(self.test_src, G)
                q.put((True, ""))
            except Exception as e:
                import traceback
                q.put((False, f"{type(e).__name__}: {e}"))
        p = mp.Process(target=w, args=(q,)); p.start(); p.join(8)
        if p.is_alive():
            p.terminate(); p.join(); resolved, fail = False, "timeout"
        else:
            try: resolved, fail = q.get_nowait()
            except Exception: resolved, fail = False, "no result"
        return {"resolved": resolved, "f2p_pass": int(resolved), "f2p_total": 1,
                "p2p_pass": 0, "p2p_total": 0, "failure": fail}

    def metrics(self):
        rr = self.chars_deleted_after_first / max(self.chars_written, 1)
        cycles = sum(v-1 for v in self.edit_regions.values() if v>1)
        return {"rework_ratio": round(rr,3), "n_edits": self.n_edits,
                "edit_error_cycles": cycles,
                "chars_written": self.chars_written,
                "chars_deleted_after_first": self.chars_deleted_after_first}
    def current_patch(self): return self.read_file()
    def close(self):
        import shutil; shutil.rmtree(self.ws, ignore_errors=True)
