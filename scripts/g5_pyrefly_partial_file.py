#!/usr/bin/env python3
"""G5 — Pyrefly partial-file probe (daemon-mode LSP).

Per experiment_plan.md §11.1 G5 and §7.1: bring up `pyrefly lsp` in daemon mode,
measure round-trip `didChange -> publishDiagnostics` latency under small
incremental edits, and probe its behaviour on deliberately broken (mid-edit)
file states. The output calibrates the parse-validity gate parameters for the
C'/D snapshot loop.

Usage:
    python g5_pyrefly_partial_file.py \
        --pyrefly /tmp/pyrefly_venv/bin/pyrefly \
        --repo /tmp/swe_repos/django__django-17087 \
        --target django/utils/version.py \
        --output /home/ianbarber/Projects/Streams/runs/g5_partial_file \
        [--warm-edits 3] [--latency-edits 20] [--diag-timeout 5.0]

Outputs (in --output):
    env.txt                 pyrefly version, repo, target, python interpreter
    latency.json            per-edit didChange -> publishDiagnostics RT times
    partial_file_probe.json per-broken-state diagnostics result
    summary.md              top-line numbers and gate recommendation

The script is idempotent and reusable. It does NOT modify the target file on
disk; all edits are sent as LSP didChange operations against an in-memory copy.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ----------------------------- minimal LSP client -----------------------------


class LSPClient:
    """Minimal LSP client over JSON-RPC stdio.

    Threaded reader pushes parsed messages onto a queue; the main thread can
    block waiting for specific responses or notifications. Diagnostics arrive
    as `textDocument/publishDiagnostics` notifications and are routed to a
    callback so per-edit round-trip latency can be measured.
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
        self._notifications: queue.Queue[dict] = queue.Queue()
        self._on_diagnostics = None  # type: ignore[assignment]
        self._next_id = 1
        self._closed = False
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._stderr_buf: list[str] = []
        self._stderr_reader = threading.Thread(
            target=self._read_stderr, daemon=True
        )
        self._stderr_reader.start()

    def set_diagnostics_callback(self, cb) -> None:
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
                # Read headers
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
            if method == "textDocument/publishDiagnostics" and self._on_diagnostics:
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
        msg = {"jsonrpc": "2.0", "id": rid, "method": method,
               "params": params or {}}
        self._write(msg)
        if not ev.wait(timeout=timeout):
            raise TimeoutError(f"request {method} (id={rid}) timed out")
        with self._lock:
            resp = self._responses.pop(rid)
            self._response_events.pop(rid, None)
        if "error" in resp:
            raise RuntimeError(f"LSP error on {method}: {resp['error']}")
        return resp.get("result", {})

    def notify(self, method: str, params: dict | None = None) -> None:
        msg = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        self._write(msg)

    def close(self) -> None:
        self._closed = True
        try:
            self.proc.terminate()
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()

    def stderr_text(self) -> str:
        return "".join(self._stderr_buf)


# ------------------------------ helpers ---------------------------------------


def file_to_uri(path: str) -> str:
    p = os.path.abspath(path)
    return "file://" + p


@dataclass
class DiagState:
    """Latest diagnostics state for the target document, plus an event signalling
    arrival of a *new* publishDiagnostics for that URI since the last reset."""
    uri: str
    diags: list[dict] = field(default_factory=list)
    version_seen: int = -1
    arrived: threading.Event = field(default_factory=threading.Event)
    last_arrival_t: float = 0.0
    arrivals_since_reset: int = 0


def make_diag_handler(state: DiagState):
    def handler(params: dict) -> None:
        if params.get("uri") != state.uri:
            return
        state.diags = list(params.get("diagnostics", []))
        v = params.get("version")
        if isinstance(v, int):
            state.version_seen = v
        state.last_arrival_t = time.monotonic()
        state.arrivals_since_reset += 1
        state.arrived.set()
    return handler


def wait_for_diag_after(state: DiagState, min_version: int | None,
                       timeout: float) -> bool:
    """Block until a publishDiagnostics arrives. If min_version is given, wait
    for one whose version field is >= min_version. Returns True on success."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if state.arrived.wait(timeout=remaining):
            state.arrived.clear()
            if min_version is None:
                return True
            if state.version_seen >= min_version:
                return True
            # Older version, keep waiting.
            continue
        return False
    return False


def parses(src: str) -> tuple[bool, str | None]:
    try:
        ast.parse(src)
        return True, None
    except SyntaxError as e:
        return False, f"{e.msg} at line {e.lineno}"


def insert_at_offset(text: str, offset: int, what: str) -> str:
    return text[:offset] + what + text[offset:]


def delete_at_offset(text: str, offset: int, n: int) -> str:
    return text[:offset] + text[offset + n:]


def offset_to_position(text: str, offset: int) -> tuple[int, int]:
    """Convert byte offset within `text` to (line, character) — line and char
    are 0-indexed; character is utf-16 code units, but for our ASCII edits the
    UTF-16 count equals the Python str count."""
    if offset > len(text):
        offset = len(text)
    pre = text[:offset]
    line = pre.count("\n")
    last_nl = pre.rfind("\n")
    char = offset - (last_nl + 1) if last_nl >= 0 else offset
    return line, char


# ----------------------- partial-file probe definitions -----------------------


def make_partial_states(original: str) -> list[dict]:
    """Construct deliberately broken file states by mid-edit truncation.

    Returns list of {name, description, content, parse_ok, parse_err}.
    Each is constructed by editing `original` in a small, localised way.
    """
    states: list[dict] = []

    # Anchor: insert each broken snippet at end of file (after a newline).
    end = original.rstrip("\n") + "\n"

    cases = [
        ("a_trailing_def_open_paren",
         "trailing `def foo(` with no body",
         "\n\ndef foo(\n"),
        ("b_unclosed_string",
         "unclosed string literal",
         "\n\nbroken = \"hello\n"),
        ("c_if_colon_no_body",
         "trailing `if x:` with no body",
         "\n\nif True:\n"),
        ("d_mid_statement_attr",
         "mid-statement truncation inside a function body",
         "\n\ndef bar(x):\n    return x.some.attribute\n    return x.partial\n"),
        ("e_unclosed_call_paren",
         "trailing unclosed parenthesis in a call",
         "\n\nresult = open(\n"),
    ]

    # case (d) more accurately: truncate mid-statement — drop the closing.
    # Build by inserting then snipping the last few chars.
    cases[3] = (
        "d_mid_statement_attr",
        "mid-statement truncation inside a function body (no closing)",
        "\n\ndef bar(x):\n    return x.some.attribute.par",
    )

    for name, desc, suffix in cases:
        content = end + suffix
        ok, err = parses(content)
        states.append({
            "name": name,
            "description": desc,
            "content": content,
            "parse_ok": ok,
            "parse_err": err,
        })
    return states


# ----------------------- latency edit construction ----------------------------


def make_latency_edits(original: str, target_offset: int,
                       n: int) -> list[dict]:
    """Construct n small edits (~5 chars insert/delete) inside a typed function
    body. The edits alternate insert/delete so the final content equals the
    original — we measure round-trip per edit, not cumulative semantics.

    Each edit returns: {kind: 'insert'|'delete', offset, text, ...}.
    """
    edits = []
    snippet = "_aux_=1"  # 7 chars, contains assignment so pyrefly inspects it
    cur = original
    for i in range(n):
        if i % 2 == 0:
            # insert snippet at target_offset
            edits.append({"kind": "insert", "offset": target_offset,
                          "text": snippet})
            cur = insert_at_offset(cur, target_offset, snippet)
        else:
            # delete the just-inserted snippet
            edits.append({"kind": "delete", "offset": target_offset,
                          "length": len(snippet)})
            cur = delete_at_offset(cur, target_offset, len(snippet))
    return edits


# --------------------------------- main ---------------------------------------


def run_probe(args: argparse.Namespace) -> int:
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    repo = Path(args.repo).resolve()
    target_rel = args.target
    target_abs = (repo / target_rel).resolve()
    if not target_abs.exists():
        print(f"[err] target not found: {target_abs}", file=sys.stderr)
        return 2

    original = target_abs.read_text()
    target_uri = file_to_uri(str(target_abs))

    # Find a safe insertion offset: inside the body of `get_version`, after the
    # first opening triple-quoted docstring closes. We just stick it after the
    # first occurrence of "main = get_main_version(version)" if present, else
    # at end of file.
    anchor = 'main = get_main_version(version)\n'
    idx = original.find(anchor)
    if idx >= 0:
        target_offset = idx + len(anchor)
    else:
        target_offset = len(original)

    # ----- spawn pyrefly lsp -----
    pyrefly = args.pyrefly
    version_proc = subprocess.run(
        [pyrefly, "--version"], capture_output=True, text=True, check=True,
    )
    pyrefly_version = version_proc.stdout.strip()

    cmd = [pyrefly, "lsp", "--indexing-mode", "lazy-blocking"]
    client = LSPClient(cmd, cwd=str(repo))

    state = DiagState(uri=target_uri)
    client.set_diagnostics_callback(make_diag_handler(state))

    # ----- initialize -----
    init_t0 = time.monotonic()
    init_params = {
        "processId": os.getpid(),
        "rootUri": file_to_uri(str(repo)),
        "workspaceFolders": [
            {"uri": file_to_uri(str(repo)), "name": repo.name},
        ],
        "capabilities": {
            "textDocument": {
                "synchronization": {"didSave": True},
                "publishDiagnostics": {"relatedInformation": True,
                                       "versionSupport": True},
            },
            "workspace": {
                "configuration": True,
                "workspaceFolders": True,
            },
        },
    }
    try:
        client.request("initialize", init_params, timeout=15.0)
    except Exception as e:
        print(f"[err] initialize failed: {e}", file=sys.stderr)
        print("[stderr]", client.stderr_text()[:2000], file=sys.stderr)
        client.close()
        return 3
    client.notify("initialized", {})
    init_dt = time.monotonic() - init_t0

    # ----- didOpen -----
    version = 1
    client.notify("textDocument/didOpen", {
        "textDocument": {
            "uri": target_uri,
            "languageId": "python",
            "version": version,
            "text": original,
        },
    })

    # Wait for initial diagnostics (warm-up)
    initial_arrived = wait_for_diag_after(
        state, min_version=None, timeout=args.diag_timeout,
    )
    initial_diags = list(state.diags)
    initial_count = len(initial_diags)

    # Warm the daemon with a few no-op edits (insert+delete) so the first real
    # latency sample is not the cold compile.
    current_text = original
    for _ in range(args.warm_edits):
        version += 1
        client.notify("textDocument/didChange", {
            "textDocument": {"uri": target_uri, "version": version},
            "contentChanges": [{"text": current_text}],  # full-document sync
        })
        wait_for_diag_after(state, min_version=version,
                            timeout=args.diag_timeout)

    # ----- latency measurement -----
    edits = make_latency_edits(original, target_offset, args.latency_edits)
    latencies: list[dict] = []
    cur_text = original

    for i, e in enumerate(edits):
        # Apply edit locally
        if e["kind"] == "insert":
            new_text = insert_at_offset(cur_text, e["offset"], e["text"])
        else:
            new_text = delete_at_offset(cur_text, e["offset"], e["length"])
        version += 1
        state.arrived.clear()
        state.arrivals_since_reset = 0
        t0 = time.monotonic()
        client.notify("textDocument/didChange", {
            "textDocument": {"uri": target_uri, "version": version},
            "contentChanges": [{"text": new_text}],  # full document
        })
        ok = wait_for_diag_after(
            state, min_version=version, timeout=args.diag_timeout,
        )
        # If the server publishes diagnostics without a version field, accept
        # the next arrival as the response.
        if not ok and state.last_arrival_t >= t0:
            ok = True
        dt_ms = (time.monotonic() - t0) * 1000.0
        latencies.append({
            "i": i,
            "kind": e["kind"],
            "version": version,
            "rt_ms": dt_ms,
            "got_diag": ok,
            "diag_count": len(state.diags),
            "version_seen": state.version_seen,
        })
        cur_text = new_text

    # ----- partial-file probe -----
    partial_states = make_partial_states(original)
    probe_results: list[dict] = []

    # Restore to original first so each probe starts from the same baseline.
    version += 1
    state.arrived.clear()
    client.notify("textDocument/didChange", {
        "textDocument": {"uri": target_uri, "version": version},
        "contentChanges": [{"text": original}],
    })
    wait_for_diag_after(state, min_version=version, timeout=args.diag_timeout)

    for ps in partial_states:
        # Apply broken content
        version += 1
        state.arrived.clear()
        state.arrivals_since_reset = 0
        t0 = time.monotonic()
        client.notify("textDocument/didChange", {
            "textDocument": {"uri": target_uri, "version": version},
            "contentChanges": [{"text": ps["content"]}],
        })
        got_broken = wait_for_diag_after(
            state, min_version=version, timeout=args.diag_timeout,
        )
        broken_dt_ms = (time.monotonic() - t0) * 1000.0
        broken_diags = list(state.diags)
        broken_count = len(broken_diags)

        # Restore to original and verify recovery
        version += 1
        state.arrived.clear()
        state.arrivals_since_reset = 0
        t1 = time.monotonic()
        client.notify("textDocument/didChange", {
            "textDocument": {"uri": target_uri, "version": version},
            "contentChanges": [{"text": original}],
        })
        got_recovery = wait_for_diag_after(
            state, min_version=version, timeout=args.diag_timeout,
        )
        recover_dt_ms = (time.monotonic() - t1) * 1000.0
        recover_diags = list(state.diags)
        recover_count = len(recover_diags)
        recovered_clean = (recover_count == initial_count)

        probe_results.append({
            "name": ps["name"],
            "description": ps["description"],
            "ast_parses": ps["parse_ok"],
            "ast_error": ps["parse_err"],
            "rt_to_diag_ms": broken_dt_ms,
            "got_broken_diag": got_broken,
            "broken_diag_count": broken_count,
            "broken_diag_sample": broken_diags[:5],
            "rt_to_recovery_ms": recover_dt_ms,
            "got_recovery_diag": got_recovery,
            "recovery_diag_count": recover_count,
            "recovered_clean": recovered_clean,
        })

    # ----- close -----
    try:
        client.request("shutdown", {}, timeout=5.0)
        client.notify("exit", {})
    except Exception:
        pass
    stderr_text = client.stderr_text()
    client.close()

    # ----- write artifacts -----
    (out_dir / "env.txt").write_text(
        f"pyrefly_version: {pyrefly_version}\n"
        f"pyrefly_sha: 2362c071caa576f9112781b5571f9e283cd52920\n"
        f"pyrefly_binary: {pyrefly}\n"
        f"repo: {repo}\n"
        f"target: {target_rel}\n"
        f"target_abs: {target_abs}\n"
        f"python_interpreter: /tmp/swe_venvs/django__django-17087/bin/python\n"
        f"lsp_client: minimal in-script JSON-RPC stdio client (no pygls)\n"
        f"warm_edits: {args.warm_edits}\n"
        f"latency_edits: {args.latency_edits}\n"
        f"diag_timeout_s: {args.diag_timeout}\n"
        f"initialize_dt_s: {init_dt:.3f}\n"
        f"initial_diag_count: {initial_count}\n"
        f"initial_diag_arrived: {initial_arrived}\n"
    )

    # Latency stats
    rts = [x["rt_ms"] for x in latencies if x["got_diag"]]
    rts_sorted = sorted(rts)

    def pct(p: float) -> float:
        if not rts_sorted:
            return float("nan")
        k = int(round((p / 100.0) * (len(rts_sorted) - 1)))
        return rts_sorted[k]

    summary_stats = {
        "n_edits": len(latencies),
        "n_with_response": len(rts),
        "mean_ms": (sum(rts) / len(rts)) if rts else float("nan"),
        "median_ms": pct(50),
        "p95_ms": pct(95),
        "p99_ms": pct(99),
        "max_ms": max(rts) if rts else float("nan"),
        "min_ms": min(rts) if rts else float("nan"),
    }

    (out_dir / "latency.json").write_text(json.dumps({
        "pyrefly_version": pyrefly_version,
        "target": str(target_abs),
        "stats_ms": summary_stats,
        "edits": latencies,
    }, indent=2))

    (out_dir / "partial_file_probe.json").write_text(json.dumps({
        "pyrefly_version": pyrefly_version,
        "target": str(target_abs),
        "initial_diag_count_clean_file": initial_count,
        "cases": probe_results,
    }, indent=2))

    (out_dir / "stderr.log").write_text(stderr_text)

    # ----- summary.md -----
    target_hit_200ms = summary_stats["p95_ms"] <= 200.0 if rts else False
    bounded_diags = all(
        r["broken_diag_count"] <= max(50, initial_count * 2)
        for r in probe_results
    )
    all_recovered = all(r["recovered_clean"] for r in probe_results)
    pyrefly_returns_diags_on_broken = any(
        r["got_broken_diag"] for r in probe_results
    )

    # Gate recommendation
    if bounded_diags and all_recovered:
        gate_reco = (
            "**No parse-validity gate required.** Pyrefly is tolerant: it "
            "returns bounded diagnostics on every broken state we probed and "
            "recovers cleanly when valid syntax is restored. Snapshot any "
            "state."
        )
    elif all_recovered and not bounded_diags:
        gate_reco = (
            "**Cheap parse-validity gate recommended.** Diagnostics explode "
            "on at least one broken state, but recovery is clean. Use "
            "`ast.parse(text)` (microseconds) as a precheck before forwarding "
            "snapshots to the diagnostic stream — drop ones that fail."
        )
    else:
        gate_reco = (
            "**Hard parse-validity gate required.** Pyrefly does not recover "
            "cleanly from at least one broken state. Snapshot only when "
            "`ast.parse(text)` succeeds; otherwise skip the snapshot entirely."
        )

    md = []
    md.append("# G5 — pyrefly partial-file probe (daemon mode)\n")
    md.append(f"- pyrefly: `{pyrefly_version}` (sha `2362c071`)")
    md.append(f"- repo: `{repo.name}`")
    md.append(f"- target: `{target_rel}` ({len(original)} chars, "
              f"{original.count(chr(10))} lines)")
    md.append(f"- initialize wall-clock: {init_dt*1000:.0f} ms")
    md.append(f"- initial diag count (clean file): {initial_count}")
    md.append("")
    md.append("## Round-trip latency (`didChange` -> `publishDiagnostics`)")
    md.append(f"- N edits: {summary_stats['n_edits']} "
              f"(responded: {summary_stats['n_with_response']})")
    md.append(f"- mean:   {summary_stats['mean_ms']:.1f} ms")
    md.append(f"- median: {summary_stats['median_ms']:.1f} ms")
    md.append(f"- p95:    {summary_stats['p95_ms']:.1f} ms")
    md.append(f"- p99:    {summary_stats['p99_ms']:.1f} ms")
    md.append(f"- max:    {summary_stats['max_ms']:.1f} ms")
    md.append(f"- 200 ms p95 target: "
              f"{'HIT' if target_hit_200ms else 'MISSED'} "
              f"(margin: {200.0 - summary_stats['p95_ms']:+.1f} ms)")
    md.append("")
    md.append("## Partial-file probe results")
    md.append("")
    md.append("| Case | Description | AST parses | Got diags | "
              "Diag count | RT (ms) | Recovered clean | Recover RT (ms) |")
    md.append("|---|---|:---:|:---:|---:|---:|:---:|---:|")
    for r in probe_results:
        md.append(
            f"| `{r['name']}` | {r['description']} | "
            f"{'yes' if r['ast_parses'] else 'no'} | "
            f"{'yes' if r['got_broken_diag'] else 'no'} | "
            f"{r['broken_diag_count']} | {r['rt_to_diag_ms']:.0f} | "
            f"{'yes' if r['recovered_clean'] else 'NO'} | "
            f"{r['rt_to_recovery_ms']:.0f} |"
        )
    md.append("")
    md.append(f"- pyrefly returns diagnostics on at least one broken state: "
              f"{'yes' if pyrefly_returns_diags_on_broken else 'no'}")
    md.append(f"- diagnostic volume bounded across all cases: "
              f"{'yes' if bounded_diags else 'no'}")
    md.append(f"- pyrefly recovers cleanly after all broken states: "
              f"{'yes' if all_recovered else 'no'}")
    md.append("")
    md.append("## Parse-validity gate recommendation")
    md.append("")
    md.append(gate_reco)
    md.append("")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")

    print(f"[done] artifacts in {out_dir}")
    print(f"[latency] median={summary_stats['median_ms']:.1f} ms "
          f"p95={summary_stats['p95_ms']:.1f} ms "
          f"p99={summary_stats['p99_ms']:.1f} ms "
          f"(target 200 ms p95: "
          f"{'HIT' if target_hit_200ms else 'MISSED'})")
    print(f"[probe] bounded={bounded_diags} all_recovered={all_recovered}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pyrefly", default="/tmp/pyrefly_venv/bin/pyrefly",
                   help="Path to pyrefly binary")
    p.add_argument("--repo", default="/tmp/swe_repos/django__django-17087",
                   help="Workspace root (must have pyrefly config)")
    p.add_argument("--target", default="django/utils/version.py",
                   help="File (relative to --repo) to probe")
    p.add_argument("--output",
                   default="/home/ianbarber/Projects/Streams/runs/g5_partial_file",
                   help="Output directory for artifacts")
    p.add_argument("--warm-edits", type=int, default=3,
                   help="Number of warm-up edits before latency measurement")
    p.add_argument("--latency-edits", type=int, default=20,
                   help="Number of edits to time")
    p.add_argument("--diag-timeout", type=float, default=10.0,
                   help="Per-edit timeout waiting for publishDiagnostics (s)")
    args = p.parse_args()
    return run_probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
