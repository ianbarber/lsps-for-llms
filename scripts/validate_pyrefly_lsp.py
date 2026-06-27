#!/usr/bin/env python3
"""Validate that the Streams `<defn>` go-to-definition (an AST resolver,
`scaffold/mock_env.py::MultiFileEnv.goto_definition`) AGREES with a LIVE
`pyrefly lsp` daemon's `textDocument/definition`.

For each of the 12 `effic` task symbols we:
  1. write the task's files + a `pyrefly.toml` into a fresh temp workspace,
  2. spawn `pyrefly lsp` (stdio JSON-RPC), initialize + didOpen every file,
  3. send `textDocument/definition` at a USE-site of `symbol` (first reference
     in target.py, computed 0-based line/char),
  4. map the returned Location (uri+range) back to file + source span,
  5. ALSO call `MultiFileEnv.goto_definition(symbol)` for ground truth,
  6. compare: SAME file AND the LSP range's defining line falls inside the AST
     span (overlap). Record agree / disagree / lsp-error.

DEADLOCK GOTCHA: stale pyrefly daemons deadlock new ones. We pkill -9 before
the run and terminate each daemon we spawn. STRICTLY SEQUENTIAL — one daemon at
a time. Every JSON-RPC read has a generous timeout so a hang fails loudly.

Run:
  pkill -9 -f pyrefly
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    .venv-streams.system/bin/python scripts/validate_pyrefly_lsp.py
"""
import os, sys, json, time, shutil, tempfile, subprocess
from urllib.parse import urlparse, unquote, quote

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PYREFLY = os.path.abspath(
    os.path.expanduser(
        os.environ.get(
            "STREAMS_PYREFLY",
            "/home/ianbarber/Projects/Streams/.venv-streams/bin/pyrefly",
        )
    )
)
READ_TIMEOUT = 20.0   # seconds per JSON-RPC read; a hang fails loudly not forever


def path_to_uri(p):
    return "file://" + quote(os.path.abspath(p))


def uri_to_path(uri):
    return unquote(urlparse(uri).path)


class LspClient:
    """Minimal stdio JSON-RPC LSP client with Content-Length framing.

    Reads are done synchronously on the main thread using `select` with a
    deadline so a daemon hang surfaces as a TimeoutError after `timeout`
    seconds instead of deadlocking forever. (A background-thread reader with
    `os.setsid`/`preexec_fn` was tried first and silently wedged the parent
    process here — the simple select loop is what actually drives the daemon.)
    Server->client requests/notifications (id-less, or with a method) are
    skipped; we only wait for the matching response id."""

    def __init__(self, cwd):
        self.cwd = cwd
        self.proc = subprocess.Popen(
            [PYREFLY, "lsp"], cwd=cwd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self._id = 0

    def _send(self, obj):
        body = json.dumps(obj).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        self.proc.stdin.write(header + body)
        self.proc.stdin.flush()

    def _read_message(self, deadline):
        """Read one framed JSON-RPC message, honoring `deadline` (epoch secs)."""
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
                    raise RuntimeError(f"LSP error on {method}: {msg['error']}")
                return msg.get("result")
            # otherwise a server->client request/notification — ignore and keep reading

    def notify(self, method, params):
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def close(self):
        try:
            self.notify("exit", None)
        except Exception:
            pass
        try:
            self.proc.kill()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            pass


def find_use_site(files, symbol, defpath):
    """Find a 0-based (path, line, char) USE-site of `symbol` in the workspace.

    Prefer a reference OUTSIDE the defining file (a genuine go-to-def jump);
    fall back to the first occurrence anywhere. Skip the definition line itself
    when possible so the daemon must resolve, not echo."""
    import re
    pat = re.compile(r"\b" + re.escape(symbol) + r"\b")
    # pass 1: a use in a non-def file (e.g. target.py importing from biglib.py)
    for path in sorted(files):
        if path == defpath:
            continue
        for li, line in enumerate(files[path].splitlines()):
            m = pat.search(line)
            if m:
                return path, li, m.start()
    # pass 2: anywhere
    for path in sorted(files):
        for li, line in enumerate(files[path].splitlines()):
            m = pat.search(line)
            if m:
                return path, li, m.start()
    return None


def lsp_definition_for_task(files, symbol, defpath, ws):
    """Drive a live pyrefly LSP daemon to resolve `symbol`. Returns
    (path_rel, start_line0, end_line0, span_text) or raises."""
    use = find_use_site(files, symbol, defpath)
    if use is None:
        raise RuntimeError(f"no use-site for {symbol!r} found")
    use_path, use_line, use_char = use

    client = LspClient(ws)
    try:
        client.request("initialize", {
            "processId": os.getpid(),
            "rootUri": path_to_uri(ws),
            "capabilities": {
                "textDocument": {
                    "definition": {"linkSupport": True},
                    "synchronization": {"didOpen": True},
                },
            },
            "workspaceFolders": [{"uri": path_to_uri(ws), "name": "ws"}],
        })
        client.notify("initialized", {})
        # didOpen every file
        for rel, content in files.items():
            client.notify("textDocument/didOpen", {
                "textDocument": {
                    "uri": path_to_uri(os.path.join(ws, rel)),
                    "languageId": "python",
                    "version": 1,
                    "text": content,
                }
            })
        # give the daemon a moment to index the opened docs
        time.sleep(1.0)
        result = client.request("textDocument/definition", {
            "textDocument": {"uri": path_to_uri(os.path.join(ws, use_path))},
            "position": {"line": use_line, "character": use_char},
        })
    finally:
        client.close()

    if not result:
        raise RuntimeError("definition returned null/empty")
    # result is Location | Location[] | LocationLink[]
    loc = result[0] if isinstance(result, list) else result
    if "targetUri" in loc:  # LocationLink
        uri = loc["targetUri"]
        rng = loc.get("targetSelectionRange") or loc["targetRange"]
    else:  # Location
        uri = loc["uri"]
        rng = loc["range"]
    tgt_path = uri_to_path(uri)
    rel = os.path.relpath(tgt_path, ws)
    start_line0 = rng["start"]["line"]
    end_line0 = rng["end"]["line"]
    # load the span text from the file on disk
    try:
        with open(tgt_path) as f:
            src_lines = f.read().splitlines()
        span = "\n".join(src_lines[start_line0:end_line0 + 1])
    except Exception:
        span = ""
    return rel, start_line0, end_line0, span


def ast_def_line(env, symbol):
    """0-based line of the AST resolver's definition (top-level node lineno)."""
    import ast as _ast
    for path in env.list_files():
        try:
            tree = _ast.parse(env.read_file(path))
        except Exception:
            continue
        for node in tree.body:
            name = None
            if isinstance(node, (_ast.ClassDef, _ast.FunctionDef, _ast.AsyncFunctionDef)):
                name = node.name
            elif isinstance(node, _ast.Assign):
                tgts = [t.id for t in node.targets if isinstance(t, _ast.Name)]
                name = symbol if symbol in tgts else None
            if name == symbol:
                end = getattr(node, "end_lineno", node.lineno)
                return path, node.lineno - 1, end - 1
    return None, None, None


def main():
    from scaffold.mock_env import MultiFileEnv
    from scripts.synth_tasks_effic import TASKS_EFFIC

    print(f"# pyrefly LSP <defn> validation — {len(TASKS_EFFIC)} effic tasks")
    print(f"# pyrefly: {PYREFLY}")
    ver = subprocess.run([PYREFLY, "--version"], capture_output=True, text=True)
    print(f"# {ver.stdout.strip() or ver.stderr.strip()}")
    print()

    agree = disagree = lsperr = 0
    rows = []
    for i, task in enumerate(TASKS_EFFIC):
        sym = task["symbol"]
        files = task["files"]
        name = task["name"]

        # ground truth from the AST resolver via a real MultiFileEnv
        env = MultiFileEnv(files, task["target"], task["test"])
        gt_span, gt_path = env.goto_definition(sym)
        gt_dpath, gt_l0, gt_l1 = ast_def_line(env, sym)

        # fresh workspace for the live daemon (separate from env.ws; one daemon
        # at a time, killed before the next task starts)
        ws = tempfile.mkdtemp(prefix="lspval_")
        for rel, content in files.items():
            p = os.path.join(ws, rel)
            d = os.path.dirname(p)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(p, "w") as f:
                f.write(content)
        with open(os.path.join(ws, "pyrefly.toml"), "w") as f:
            f.write("[tool.pyrefly]\nproject-includes = [\"*.py\"]\n")

        status = "ERR"
        detail = ""
        try:
            rel, ls0, ls1, lspan = lsp_definition_for_task(files, sym, gt_path, ws)
            same_file = (rel == gt_path)
            # overlap: the LSP defining line falls within the AST span (or v.v.)
            line_overlap = (
                gt_l0 is not None
                and not (ls1 < gt_l0 or ls0 > gt_l1)
            )
            if same_file and line_overlap:
                status = "AGREE"
                agree += 1
            else:
                status = "DISAGREE"
                disagree += 1
                detail = (f"lsp={rel}:L{ls0+1}-{ls1+1} "
                          f"ast={gt_path}:L{(gt_l0 or 0)+1}-{(gt_l1 or 0)+1}")
        except Exception as e:
            status = "LSP-ERR"
            lsperr += 1
            detail = f"{type(e).__name__}: {e}"
        finally:
            env.close()
            shutil.rmtree(ws, ignore_errors=True)
            # belt-and-braces: purge any daemon that escaped
            subprocess.run(["pkill", "-9", "-f", f"pyrefly lsp"],
                           capture_output=True)

        line = f"[{i+1:2d}/12] {name:24s} sym={sym:14s} {status}"
        if detail:
            line += f"  {detail}"
        print(line, flush=True)
        rows.append((name, sym, status, detail))

    print()
    print(f"SUMMARY: {agree}/{len(TASKS_EFFIC)} agree, "
          f"{disagree} disagree, {lsperr} lsp-error")
    return 0 if lsperr == 0 and disagree == 0 else (1 if agree == 0 else 0)


if __name__ == "__main__":
    sys.exit(main())
