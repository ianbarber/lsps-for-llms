#!/usr/bin/env python3
"""pyrefly_nav: type-aware, receiver-correct navigation via the pyrefly LSP.

The treatment tool for the dispatch-ambiguity experiment. Runs INSIDE the task container against
/testbed (or $PYREFLY_NAV_ROOT). Unlike grep / codenav (which match `def NAME` textually and cannot
say which override binds), this asks pyrefly to resolve the symbol AT ITS USAGE POSITION, so it uses
the receiver's static type:

  pyrefly_nav goto  FILE LINE SYMBOL   go-to-definition of SYMBOL as used at FILE:LINE -> the single
                                       receiver-correct definition (the right override)
  pyrefly_nav impls FILE LINE SYMBOL   find-implementations: the set of overrides over the receiver's
                                       class hierarchy (the honest tool for dynamic-dispatch cases,
                                       where a single goto cannot be resolved statically)

Self-contained (pure stdlib) so it can be injected into a container like codenav; the LSP client is
the one validated in scripts/validate_pyrefly_lsp.py. Requires a `pyrefly` binary on PATH (or
$PYREFLY_BIN). Writes an untracked pyrefly.toml at the root if absent so pyrefly indexes the project;
untracked, so it never appears in the agent's `git diff` submission.
"""
import io
import os
import re
import sys
import time
import subprocess
from urllib.parse import quote, unquote, urlparse

for _n in ("stdout", "stderr"):
    try:
        setattr(sys, _n, io.TextIOWrapper(getattr(sys, _n).buffer, encoding="utf-8",
                                          errors="replace", line_buffering=True))
    except Exception:
        pass

ROOT = os.environ.get("PYREFLY_NAV_ROOT", "/testbed")
PYREFLY = os.environ.get("PYREFLY_BIN", "pyrefly")
READ_TIMEOUT = float(os.environ.get("PYREFLY_RPC_TIMEOUT", "60"))
INDEX_WAIT = float(os.environ.get("PYREFLY_INDEX_WAIT", "6"))
MAX_SPAN = 80


def path_to_uri(p):
    return "file://" + quote(os.path.abspath(p))


def uri_to_path(u):
    return unquote(urlparse(u).path)


class LspClient:
    """Minimal stdio JSON-RPC LSP client with Content-Length framing and per-read deadlines
    (validated in scripts/validate_pyrefly_lsp.py)."""

    def __init__(self, cwd):
        self.proc = subprocess.Popen([PYREFLY, "lsp"], cwd=cwd,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, bufsize=0)
        self._id = 0

    def _send(self, obj):
        import json
        body = json.dumps(obj).encode("utf-8")
        self.proc.stdin.write(("Content-Length: %d\r\n\r\n" % len(body)).encode("ascii") + body)
        self.proc.stdin.flush()

    def _read_message(self, deadline):
        import json
        import select
        f = self.proc.stdout

        def read_line():
            buf = b""
            while not buf.endswith(b"\n"):
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError("timed out reading LSP header")
                r, _, _ = select.select([f], [], [], max(0.05, remaining))
                if not r:
                    raise TimeoutError("timed out waiting for LSP header byte")
                c = f.read(1)
                if not c:
                    raise RuntimeError("LSP daemon closed stdout (eof)")
                buf += c
            return buf

        headers = {}
        while True:
            line = read_line()
            if line in (b"\r\n", b"\n"):
                break
            k, _, v = line.partition(b":")
            headers[k.strip().lower()] = v.strip()
        n = int(headers.get(b"content-length", b"0"))
        body = b""
        while len(body) < n:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError("timed out reading LSP body")
            r, _, _ = select.select([f], [], [], max(0.05, remaining))
            if not r:
                raise TimeoutError("timed out waiting for LSP body bytes")
            chunk = f.read(n - len(body))
            if not chunk:
                raise RuntimeError("LSP daemon closed stdout mid-body (eof)")
            body += chunk
        return json.loads(body.decode("utf-8"))

    def request(self, method, params, timeout=READ_TIMEOUT):
        self._id += 1
        mid = self._id
        self._send({"jsonrpc": "2.0", "id": mid, "method": method, "params": params})
        deadline = time.time() + timeout
        while True:
            msg = self._read_message(deadline)
            if msg.get("id") == mid and ("result" in msg or "error" in msg):
                if "error" in msg:
                    raise RuntimeError("LSP error on %s: %s" % (method, msg["error"]))
                return msg.get("result")

    def notify(self, method, params):
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def close(self):
        try:
            self.notify("exit", None)
        except Exception:
            pass
        try:
            self.proc.kill()
            self.proc.wait(timeout=5)
        except Exception:
            pass


def _ensure_config():
    cfg = os.path.join(ROOT, "pyrefly.toml")
    if not os.path.exists(cfg):
        try:
            open(cfg, "w").write('project-includes = ["**/*.py"]\n')
        except Exception:
            pass


def _col_of(abspath, line1, symbol):
    try:
        lines = open(abspath, encoding="utf-8", errors="replace").read().splitlines()
    except Exception:
        return None
    if 1 <= line1 <= len(lines):
        m = re.search(r"\b" + re.escape(symbol) + r"\b", lines[line1 - 1])
        if m:
            return m.start()
    return None


def _query(use_rel, line0, char0, method):
    _ensure_config()
    use_abs = os.path.join(ROOT, use_rel)
    c = LspClient(ROOT)
    try:
        c.request("initialize", {
            "processId": os.getpid(), "rootUri": path_to_uri(ROOT),
            "capabilities": {"textDocument": {
                "definition": {"linkSupport": True},
                "implementation": {"linkSupport": True},
            }},
            "workspaceFolders": [{"uri": path_to_uri(ROOT), "name": "ws"}],
        })
        c.notify("initialized", {})
        c.notify("textDocument/didOpen", {"textDocument": {
            "uri": path_to_uri(use_abs), "languageId": "python", "version": 1,
            "text": open(use_abs, encoding="utf-8", errors="replace").read()}})
        time.sleep(INDEX_WAIT)
        return c.request(method, {"textDocument": {"uri": path_to_uri(use_abs)},
                                  "position": {"line": line0, "character": char0}})
    finally:
        c.close()


def _loc_parts(loc):
    uri = loc.get("targetUri") or loc["uri"]
    rng = loc.get("targetSelectionRange") or loc.get("targetRange") or loc["range"]
    return uri_to_path(uri), rng["start"]["line"]


def _span(abspath, start0):
    try:
        lines = open(abspath, encoding="utf-8", errors="replace").readlines()
    except Exception:
        return ""
    if start0 < 0 or start0 >= len(lines):
        return ""
    base = len(lines[start0]) - len(lines[start0].lstrip())
    out = [lines[start0]]
    j = start0 + 1
    while j < len(lines) and len(out) < MAX_SPAN:
        s = lines[j]
        if s.strip():
            ind = len(s) - len(s.lstrip())
            if ind < base or (ind == base and s.lstrip().startswith(("def ", "class ", "@", "async def "))):
                break
        out.append(s)
        j += 1
    return "".join(out).rstrip("\n")


def goto(file, line1, symbol):
    col = _col_of(os.path.join(ROOT, file), line1, symbol)
    if col is None:
        sys.stderr.write("pyrefly_nav: '%s' not found on %s:%d\n" % (symbol, file, line1))
        return 64
    res = _query(file, line1 - 1, col, "textDocument/definition")
    if not res:
        sys.stderr.write("pyrefly_nav: no definition resolved for '%s' at %s:%d\n" % (symbol, file, line1))
        return 2
    loc = res[0] if isinstance(res, list) else res
    tgt, start0 = _loc_parts(loc)
    rel = os.path.relpath(tgt, ROOT)
    sys.stdout.write("# %s:%d  (definition of '%s' resolved by type)  [pyrefly goto]\n%s\n"
                     % (rel, start0 + 1, symbol, _span(tgt, start0)))
    return 0


def impls(file, line1, symbol):
    col = _col_of(os.path.join(ROOT, file), line1, symbol)
    if col is None:
        sys.stderr.write("pyrefly_nav: '%s' not found on %s:%d\n" % (symbol, file, line1))
        return 64
    res = _query(file, line1 - 1, col, "textDocument/implementation")
    if not res:
        sys.stderr.write("pyrefly_nav: no implementations resolved for '%s' at %s:%d\n" % (symbol, file, line1))
        return 2
    locs = res if isinstance(res, list) else [res]
    sys.stdout.write("# implementations of '%s' (%d) over the receiver's hierarchy  [pyrefly impls]\n"
                     % (symbol, len(locs)))
    for loc in locs:
        tgt, start0 = _loc_parts(loc)
        sys.stdout.write("%s:%d\n" % (os.path.relpath(tgt, ROOT), start0 + 1))
    return 0


def main(argv):
    if argv[:1] == ["--selfcheck"]:
        v = subprocess.run([PYREFLY, "--version"], stdout=subprocess.PIPE, universal_newlines=True).stdout.strip()
        sys.stdout.write("pyrefly_nav ok (root=%s, %s)\n" % (ROOT, v))
        return 0
    if len(argv) < 4 or argv[0] not in ("goto", "impls"):
        sys.stderr.write("usage: pyrefly_nav {goto|impls} FILE LINE SYMBOL\n")
        return 64
    cmd, file, line, symbol = argv[0], argv[1], argv[2], argv[3]
    try:
        line1 = int(line)
    except ValueError:
        sys.stderr.write("pyrefly_nav: LINE must be an integer\n")
        return 64
    return goto(file, line1, symbol) if cmd == "goto" else impls(file, line1, symbol)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
