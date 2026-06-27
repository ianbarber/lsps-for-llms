#!/usr/bin/env python3
"""Single-file environment for smoke-testing the StreamAgent.

Runs real pyrefly via its CLI so diagnostics are authentic. Tests are executed against
an assertion file."""
import os, re, json, subprocess, tempfile, multiprocessing as mp

PYREFLY = os.path.abspath(
    os.path.expanduser(
        os.environ.get(
            "STREAMS_PYREFLY",
            "/home/ianbarber/Projects/Streams/.venv-streams/bin/pyrefly",
        )
    )
)
SEV = {0: "error", 1: "error", 2: "warning", 3: "info"}

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


class MultiFileEnv:
    """Multi-file workspace environment: the bug lives in a TARGET file whose type
    definitions live in OTHER files the model has not been shown. This is the
    setting where the checker's knowledge is NON-REDUNDANT with the model's context
    (the single-file suite's information-redundancy confound, PAPER §9).

    - real pyrefly over the whole workspace (cross-file diagnostics)
    - read_file/apply_line_edit dispatch on any workspace file
    - behavioural tests run in a FRESH subprocess with the workspace on sys.path
    - rework accounting identical to MockEnv (target-file edits)
    """

    def __init__(self, files: dict, target: str, test_src: str, force_diag=None):
        import sys
        self.ws = tempfile.mkdtemp(prefix="mfenv_")
        self.files = dict(files)
        self.target = target
        self.test_src = test_src
        self.force_diag = force_diag
        for rel, content in files.items():
            p = os.path.join(self.ws, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True) if os.path.dirname(rel) else None
            with open(p, "w") as f: f.write(content)
        with open(os.path.join(self.ws, "pyrefly.toml"), "w") as f:
            f.write("[tool.pyrefly]\nproject-includes = [\"*.py\"]\n")
        subprocess.run([PYREFLY, "init"], cwd=self.ws, capture_output=True, text=True)
        self._py = sys.executable
        self.chars_written = 0
        self.chars_deleted_after_first = 0
        self.first_write_done = False
        self.n_edits = 0
        self.edit_regions = {}

    def _abspath(self, rel):
        p = os.path.normpath(os.path.join(self.ws, rel))
        if not p.startswith(self.ws):
            raise ValueError(f"path escapes workspace: {rel}")
        return p

    def read_file(self, path=None):
        with open(self._abspath(path or self.target)) as f:
            return f.read()

    def list_files(self):
        return sorted(self.files.keys())

    def goto_definition(self, symbol):
        """REAL go-to-definition (not an oracle): AST-resolve `symbol` against the LIVE workspace and return the
        source span of its top-level definition (class/def/assignment), exactly as an LSP go-to-def would — derived
        from the codebase, with no privileged knowledge of which symbol or what the answer is. Returns (src, path)
        or (None, None) if unresolved (e.g. the name is not defined anywhere -> a real LSP miss)."""
        import ast as _ast
        for path in self.list_files():
            try:
                src = self.read_file(path)
                tree = _ast.parse(src)
            except Exception:
                continue
            lines = src.splitlines()
            for node in tree.body:   # top-level only (module-scope defs), like go-to-def on an imported name
                name = None
                if isinstance(node, (_ast.ClassDef, _ast.FunctionDef, _ast.AsyncFunctionDef)):
                    name = node.name
                elif isinstance(node, _ast.Assign):
                    tgts = [t.id for t in node.targets if isinstance(t, _ast.Name)]
                    name = symbol if symbol in tgts else None
                if name == symbol:
                    end = getattr(node, "end_lineno", node.lineno)
                    span = "\n".join(lines[node.lineno - 1:end])
                    return span, path
        return None, None

    def lsp_definition(self, symbol):
        """OPT-IN live LSP go-to-definition: drive a real `pyrefly lsp` daemon over THIS workspace and return the
        symbol's definition as (span_text, path) — the SAME shape as goto_definition — so the cheap <defn> action is
        backed by a production language server instead of our AST resolver. Validated to agree with goto_definition
        12/12 on the effic suite (scripts/validate_pyrefly_lsp.py).

        DEADLOCK GOTCHA: pyrefly daemons deadlock under concurrency — this spawns ONE daemon, queries, and kills it,
        and must be called SEQUENTIALLY (the eval loop already is). On any error (no daemon, null result, timeout) it
        returns (None, None) — callers should fall back to the AST resolver, never hang. Reuses the validated client
        in scripts/validate_pyrefly_lsp.py."""
        try:
            import importlib.util, sys as _sys
            mod = _sys.modules.get("validate_pyrefly_lsp")
            if mod is None:
                _here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                _path = os.path.join(_here, "scripts", "validate_pyrefly_lsp.py")
                spec = importlib.util.spec_from_file_location("validate_pyrefly_lsp", _path)
                mod = importlib.util.module_from_spec(spec)
                _sys.modules["validate_pyrefly_lsp"] = mod
                spec.loader.exec_module(mod)
            # ground-truth defining file (to prefer a cross-file use-site); harmless if it misses
            _, defpath = self.goto_definition(symbol)
            files = {p: self.read_file(p) for p in self.list_files()}
            rel, _ls0, _ls1, span = mod.lsp_definition_for_task(files, symbol, defpath, self.ws)
            if rel is None:
                return None, None
            # The LSP's textDocument/definition returns the definition LOCATION (a line/range), not the
            # full body — the raw span is just `class Account:` / `def f(...):`. A go-to-definition *tool*
            # returns the definition itself: expand the LSP-resolved location to the enclosing top-level
            # node's full source span (the same body goto_definition returns, the form the model consumes).
            # The LSP drives the RESOLUTION; we fetch the body at the location it found.
            import ast as _ast
            try:
                src = files.get(rel) or self.read_file(rel)
                line1 = int(_ls0 or 0) + 1   # 0-based LSP line -> 1-based
                tree = _ast.parse(src); lines = src.splitlines()
                for node in tree.body:
                    if isinstance(node, (_ast.ClassDef, _ast.FunctionDef, _ast.AsyncFunctionDef, _ast.Assign)):
                        end = getattr(node, "end_lineno", node.lineno)
                        if node.lineno <= line1 <= end:
                            return "\n".join(lines[node.lineno - 1:end]), rel
            except Exception:
                pass
            return (span or None), rel
        except Exception:
            return None, None

    def find_references(self, symbol):
        """REAL find-references: scan the live workspace for files that USE `symbol` (whole-word), excluding its
        definition site. Returns a sorted list of paths (an LSP references result), derived from the codebase."""
        import re as _re
        pat = _re.compile(r"\b" + _re.escape(symbol) + r"\b")
        _, defpath = self.goto_definition(symbol)
        hits = []
        for path in self.list_files():
            if path == defpath:
                continue
            try:
                if pat.search(self.read_file(path)):
                    hits.append(path)
            except Exception:
                continue
        return sorted(hits)

    def apply_line_edit(self, path, start, end, new_text):
        path = path or self.target
        try:
            cur = self.read_file(path)
        except FileNotFoundError:
            return False, f"no such file: {path}"
        lines = cur.splitlines(keepends=True)
        n = len(lines)
        if not (1 <= start <= end <= n):
            return False, f"line range {start}-{end} out of bounds ({path} has {n} lines)"
        removed = "".join(lines[start - 1:end])
        nt = new_text if new_text.endswith("\n") else new_text + "\n"
        new = "".join(lines[:start - 1]) + nt + "".join(lines[end:])
        self.chars_written += len(nt)
        if self.first_write_done:
            self.chars_deleted_after_first += len(removed)
        self.first_write_done = True
        self.n_edits += 1
        with open(self._abspath(path), "w") as f:
            f.write(new)
        return True, "ok"

    def pyrefly_diagnostics(self, path=None):
        """Whole-workspace check; cross-file errors surface at the use sites."""
        if self.force_diag is not None:
            return self.force_diag
        try:
            r = subprocess.run([PYREFLY, "check", "--output-format", "json"],
                               cwd=self.ws, capture_output=True, text=True, timeout=30)
            data = json.loads(r.stdout or "{}")
        except Exception:
            return ""
        diags = data.get("errors", []) if isinstance(data, dict) else []
        out = []
        for d in diags[:10]:
            line = d.get("line", "?")
            fp = os.path.basename(d.get("path", "") or "")
            code = d.get("name", "diag")
            msg = (d.get("concise_description") or d.get("description") or "")[:160]
            out.append(f"[error] {fp} L{line} {code}: {msg}")
            if isinstance(line, int):
                self.edit_regions[line] = self.edit_regions.get(line, 0) + 1
        return "\n".join(out)

    def run_tests(self):
        runner = os.path.join(self.ws, "_run_tests.py")
        with open(runner, "w") as f:
            f.write("import sys, os\nsys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n"
                    + self.test_src + "\nprint('PASS')\n")
        try:
            r = subprocess.run([self._py, runner], cwd=self.ws, capture_output=True,
                               text=True, timeout=20)
            ok = r.returncode == 0 and "PASS" in r.stdout
            fail = "" if ok else (r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "test failed")
        except subprocess.TimeoutExpired:
            ok, fail = False, "timeout"
        return {"resolved": ok, "f2p_pass": int(ok), "f2p_total": 1,
                "p2p_pass": 0, "p2p_total": 0, "failure": fail[:300]}

    def metrics(self):
        rr = self.chars_deleted_after_first / max(self.chars_written, 1)
        cycles = sum(v - 1 for v in self.edit_regions.values() if v > 1)
        return {"rework_ratio": round(rr, 3), "n_edits": self.n_edits,
                "edit_error_cycles": cycles, "chars_written": self.chars_written,
                "chars_deleted_after_first": self.chars_deleted_after_first}

    def current_patch(self):
        return self.read_file(self.target)

    def close(self):
        import shutil; shutil.rmtree(self.ws, ignore_errors=True)
