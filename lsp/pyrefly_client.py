#!/usr/bin/env python3
"""Reusable pyrefly daemon-mode LSP client.

Promoted from the G5 probe (`scripts/g5_pyrefly_partial_file.py`, log 2026-05-27)
into a clean module the four delivery layers (B/C/C'/D) build on. Provides:

- `LSPClient` — minimal JSON-RPC-over-stdio client (threaded reader, request /
  notify, diagnostics callback). No `pygls` dependency.
- `PyreflyDaemon` — daemon lifecycle + document state: spawn `pyrefly lsp`,
  initialize against a workspace root, `did_open` / `did_change` a document, and
  block for the resulting `publishDiagnostics`. One persistent process per task
  (per experiment_plan §7.1 transport).

The daemon emits a non-LSP-spec `data: "committing-transaction"` field per
diagnostic (confirmed via G5 and re-confirmed here); that is *not* stripped at
this layer — normalization in `lsp/payload.py` is the single chokepoint that
projects raw diagnostics to the canonical payload, so the SHA-256 audit (G4) has
exactly one code path to reason about.

Round-trip `did_change -> publishDiagnostics` is p95 6-21 ms (G5), ~30x under D's
200 ms debounce budget.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Default pyrefly binary in this project's venv. Pinned version 1.0.0,
# sha 2362c071caa576f9112781b5571f9e283cd52920 (experiment_plan §6).
DEFAULT_PYREFLY = "/home/ianbarber/Projects/Streams/.venv-streams/bin/pyrefly"


def file_to_uri(path: str) -> str:
    return "file://" + os.path.abspath(path)


# ----------------------------- minimal LSP client -----------------------------


class LSPClient:
    """Minimal LSP client over JSON-RPC stdio.

    A daemon reader thread parses framed messages and routes them: responses to
    per-id events (so `request` can block), `publishDiagnostics` notifications to
    a registered callback, everything else onto a notification queue.
    """

    def __init__(self, cmd: list[str], cwd: str | None = None) -> None:
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            bufsize=0,
        )
        self._responses: dict[int, dict] = {}
        self._response_events: dict[int, threading.Event] = {}
        self._lock = threading.Lock()
        self._notifications: "queue.Queue[dict]" = queue.Queue()
        self._on_diagnostics: Callable[[dict], None] | None = None
        self._next_id = 1
        self._closed = False
        self._stderr_buf: list[str] = []
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(
            target=self._read_stderr, daemon=True
        )
        self._stderr_reader.start()

    def set_diagnostics_callback(self, cb: Callable[[dict], None]) -> None:
        self._on_diagnostics = cb

    def _read_stderr(self) -> None:
        try:
            while True:
                line = self.proc.stderr.readline()
                if not line:
                    break
                self._stderr_buf.append(line.decode("utf-8", errors="replace"))
        except Exception:
            pass

    def _read_loop(self) -> None:
        try:
            while not self._closed:
                headers: dict[str, str] = {}
                while True:
                    line = self.proc.stdout.readline()
                    if not line:
                        return
                    line = line.decode("ascii", errors="replace")
                    if line in ("\r\n", "\n"):
                        break
                    if ":" in line:
                        k, _, v = line.partition(":")
                        headers[k.strip().lower()] = v.strip()
                length = int(headers.get("content-length", "0"))
                if length <= 0:
                    continue
                body = b""
                while len(body) < length:
                    chunk = self.proc.stdout.read(length - len(body))
                    if not chunk:
                        return
                    body += chunk
                try:
                    msg = json.loads(body.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                self._dispatch(msg)
        except Exception:
            pass

    def _dispatch(self, msg: dict) -> None:
        if "id" in msg and ("result" in msg or "error" in msg):
            rid = msg["id"]
            with self._lock:
                self._responses[rid] = msg
                ev = self._response_events.get(rid)
            if ev is not None:
                ev.set()
            return
        if "method" in msg:
            method = msg["method"]
            if (method == "textDocument/publishDiagnostics"
                    and self._on_diagnostics):
                try:
                    self._on_diagnostics(msg.get("params", {}))
                except Exception:
                    pass
            self._notifications.put(msg)

    def _write(self, msg: dict) -> None:
        body = json.dumps(msg).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        self.proc.stdin.write(header + body)
        self.proc.stdin.flush()

    def request(self, method: str, params: dict | None = None,
                timeout: float = 30.0) -> dict:
        with self._lock:
            rid = self._next_id
            self._next_id += 1
            ev = threading.Event()
            self._response_events[rid] = ev
        self._write({"jsonrpc": "2.0", "id": rid, "method": method,
                     "params": params or {}})
        if not ev.wait(timeout=timeout):
            raise TimeoutError(f"request {method} (id={rid}) timed out")
        with self._lock:
            resp = self._responses.pop(rid)
            self._response_events.pop(rid, None)
        if "error" in resp:
            raise RuntimeError(f"LSP error on {method}: {resp['error']}")
        return resp.get("result", {})

    def notify(self, method: str, params: dict | None = None) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def close(self) -> None:
        self._closed = True
        try:
            self.proc.terminate()
            self.proc.wait(timeout=3)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass

    def stderr_text(self) -> str:
        return "".join(self._stderr_buf)


# ------------------------------ document state --------------------------------


@dataclass
class DiagState:
    """Latest diagnostics for one document URI, plus an event signalling that a
    *new* publishDiagnostics arrived for it since the last reset."""

    uri: str
    diags: list[dict] = field(default_factory=list)
    version_seen: int = -1
    arrived: threading.Event = field(default_factory=threading.Event)
    last_arrival_t: float = 0.0


def _make_diag_handler(state: DiagState) -> Callable[[dict], None]:
    def handler(params: dict) -> None:
        if params.get("uri") != state.uri:
            return
        state.diags = list(params.get("diagnostics", []))
        v = params.get("version")
        if isinstance(v, int):
            state.version_seen = v
        state.last_arrival_t = time.monotonic()
        state.arrived.set()
    return handler


def _wait_for_diag_after(state: DiagState, min_version: int | None,
                         timeout: float) -> bool:
    """Block until a publishDiagnostics arrives. If `min_version` is given, wait
    for one whose version field is >= it. Returns True on success."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if state.arrived.wait(timeout=remaining):
            state.arrived.clear()
            if min_version is None or state.version_seen >= min_version:
                return True
            continue
        return False
    return False


# ------------------------------ pyrefly daemon --------------------------------


class PyreflyDaemon:
    """One persistent `pyrefly lsp` daemon scoped to a workspace root.

    Usage:
        with PyreflyDaemon(repo_root) as daemon:
            diags = daemon.open(target_py)          # raw publishDiagnostics list
            diags = daemon.change(target_py, new_src)

    `open`/`change` return the raw pyrefly diagnostics (list of dicts). Callers
    feed them to `lsp.payload.normalize_payload` — this client deliberately does
    no normalization so there is a single canonicalization code path.
    """

    def __init__(self, repo_root: str,
                 pyrefly: str = DEFAULT_PYREFLY,
                 indexing_mode: str = "lazy-blocking",
                 diag_timeout: float = 10.0) -> None:
        self.repo_root = str(Path(repo_root).resolve())
        self.pyrefly = pyrefly
        self.diag_timeout = diag_timeout
        self._states: dict[str, DiagState] = {}
        self._version: dict[str, int] = {}
        self._active_uri: str | None = None

        self.client = LSPClient(
            [pyrefly, "lsp", "--indexing-mode", indexing_mode],
            cwd=self.repo_root,
        )
        self.client.set_diagnostics_callback(self._route_diag)
        self._initialize()

    # -- callback routing across multiple open documents --
    def _route_diag(self, params: dict) -> None:
        uri = params.get("uri")
        st = self._states.get(uri) if uri else None
        if st is not None:
            st.diags = list(params.get("diagnostics", []))
            v = params.get("version")
            if isinstance(v, int):
                st.version_seen = v
            st.last_arrival_t = time.monotonic()
            st.arrived.set()

    def _initialize(self) -> None:
        root_uri = file_to_uri(self.repo_root)
        self.client.request("initialize", {
            "processId": os.getpid(),
            "rootUri": root_uri,
            "workspaceFolders": [
                {"uri": root_uri, "name": Path(self.repo_root).name},
            ],
            "capabilities": {
                "textDocument": {
                    "synchronization": {"didSave": True},
                    "publishDiagnostics": {
                        "relatedInformation": True,
                        "versionSupport": True,
                    },
                },
                "workspace": {"configuration": True, "workspaceFolders": True},
            },
        }, timeout=15.0)
        self.client.notify("initialized", {})

    def _state_for(self, target_path: str) -> tuple[str, DiagState]:
        uri = file_to_uri(target_path)
        st = self._states.get(uri)
        if st is None:
            st = DiagState(uri=uri)
            self._states[uri] = st
            self._version[uri] = 0
        return uri, st

    def open(self, target_path: str, text: str | None = None) -> list[dict]:
        """didOpen a document; return raw diagnostics for it."""
        uri, st = self._state_for(target_path)
        if text is None:
            text = Path(target_path).read_text()
        self._version[uri] = 1
        st.arrived.clear()
        self.client.notify("textDocument/didOpen", {
            "textDocument": {"uri": uri, "languageId": "python",
                             "version": 1, "text": text},
        })
        _wait_for_diag_after(st, None, self.diag_timeout)
        return list(st.diags)

    def change(self, target_path: str, new_text: str) -> list[dict]:
        """didChange (full-document sync); block for the resulting diagnostics
        and return them raw. Round-trip is the daemon's IPC latency (G5: p95
        6-21 ms)."""
        uri, st = self._state_for(target_path)
        version = self._version.get(uri, 1) + 1
        self._version[uri] = version
        st.arrived.clear()
        t0 = time.monotonic()
        self.client.notify("textDocument/didChange", {
            "textDocument": {"uri": uri, "version": version},
            "contentChanges": [{"text": new_text}],  # full-document sync
        })
        ok = _wait_for_diag_after(st, version, self.diag_timeout)
        if not ok and st.last_arrival_t >= t0:
            # Server published without an echoed version; accept latest arrival.
            ok = True
        return list(st.diags)

    def close(self) -> None:
        try:
            self.client.request("shutdown", {}, timeout=5.0)
            self.client.notify("exit", {})
        except Exception:
            pass
        self.client.close()

    def __enter__(self) -> "PyreflyDaemon":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
