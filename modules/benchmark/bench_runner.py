"""Sequential benchmark runner.

Runs a :class:`BenchPlan` — one or more benchmark commands — on a background
worker thread, streaming each process's *stderr* (progress / model-load logs)
into an event queue while *capturing its stdout* in full for result parsing.
This split is the key difference from ``BuildRunner`` (which merges the two):
``llama-bench -o json`` writes machine-readable results to stdout that must be
captured intact, not interleaved with progress noise.

Threading model mirrors ``modules/build/build_runner.py``: the worker thread
never touches Tk; the UI drains ``self.events`` on its mainloop tick. Cancel
terminates the current process group so a wedged benchmark actually stops.
"""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

EVENT_LINE = "line"  # ("line", text)
EVENT_STEP_START = "step_start"  # ("step_start", StepInfo)
EVENT_STEP_RESULT = "step_result"  # ("step_result", StepResult)
EVENT_DONE = "done"  # ("done", failed_count)
EVENT_CANCELLED = "cancelled"  # ("cancelled", None)
EVENT_ERROR = "error"  # ("error", message)

EVENT_QUEUE_MAXSIZE = 4000
PROC_TERMINATE_WAIT_SECONDS = 5.0
PROC_KILL_WAIT_SECONDS = 2.0
PROC_POLL_SECONDS = 0.05
MAX_CAPTURE_BYTES = 8 * 1024 * 1024  # cap stdout capture per step


@dataclass
class BenchStep:
    """One benchmark process to run."""

    cmd: list[str]
    combo: dict[str, str] = field(default_factory=dict)
    label: str = ""


@dataclass
class BenchPlan:
    """Everything needed to run a sweep."""

    tool: str
    steps: list[BenchStep]
    cwd: str = ""
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class StepInfo:
    index: int
    total: int
    combo: dict[str, str]
    cmd: list[str]
    label: str


@dataclass
class StepResult:
    index: int
    combo: dict[str, str]
    rc: int
    stdout: str
    tool: str


class BenchRunner:
    """Single-job sequential runner. Reuse one instance for the tab lifetime."""

    def __init__(self) -> None:
        self.events: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=EVENT_QUEUE_MAXSIZE)
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None
        self._cancel = threading.Event()
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- state
    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def cancel(self) -> None:
        self._cancel.set()
        with self._lock:
            proc = self._proc
        if proc and proc.poll() is None:
            self._signal_terminate(proc)

    # ---------------------------------------------------------------- signals
    @staticmethod
    def _signal_terminate(proc: subprocess.Popen) -> None:
        try:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            else:
                try:
                    pgid = os.getpgid(proc.pid)
                except (OSError, ProcessLookupError):
                    pgid = proc.pid
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except (OSError, ProcessLookupError):
                    proc.terminate()
        except Exception:
            pass

    @staticmethod
    def _signal_kill(proc: subprocess.Popen) -> None:
        try:
            if os.name == "nt":
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        capture_output=True,
                        timeout=PROC_KILL_WAIT_SECONDS,
                        check=False,
                    )
                except Exception:
                    proc.kill()
            else:
                try:
                    pgid = os.getpgid(proc.pid)
                except (OSError, ProcessLookupError):
                    pgid = proc.pid
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    proc.kill()
        except Exception:
            pass

    def _wait_for_proc_shutdown(self, proc: subprocess.Popen) -> int:
        try:
            try:
                return int(proc.wait(timeout=PROC_TERMINATE_WAIT_SECONDS) or 0)
            except subprocess.TimeoutExpired:
                self._emit_line("Process did not exit after termination; forcing shutdown.")
                self._signal_kill(proc)
                try:
                    return int(proc.wait(timeout=PROC_KILL_WAIT_SECONDS) or 0)
                except subprocess.TimeoutExpired:
                    rc = proc.poll()
                    return int(rc) if rc is not None else 1
        finally:
            with self._lock:
                if self._proc is proc:
                    self._proc = None

    # ---------------------------------------------------------------- events
    def _emit_line(self, text: str) -> None:
        try:
            self.events.put_nowait((EVENT_LINE, text))
        except queue.Full:
            pass

    def _emit_event(self, kind: str, payload: object) -> None:
        self.events.put((kind, payload))

    # ---------------------------------------------------------------- driver
    def start(self, plan: BenchPlan) -> bool:
        if self.is_running:
            return False
        try:
            while True:
                self.events.get_nowait()
        except queue.Empty:
            pass
        self._cancel.clear()
        self._thread = threading.Thread(target=self._run, args=(plan,), name="BenchRunner", daemon=True)
        self._thread.start()
        return True

    def _run(self, plan: BenchPlan) -> None:
        try:
            if not plan.steps:
                self._emit_event(EVENT_ERROR, "Nothing to run (empty matrix).")
                return
            env = os.environ.copy()
            for k, v in (plan.env or {}).items():
                try:
                    env[str(k)] = "" if v is None else str(v)
                except Exception:
                    continue
            cwd = plan.cwd or None
            if cwd and not Path(cwd).is_dir():
                cwd = None
            total = len(plan.steps)
            failed = 0
            for i, step in enumerate(plan.steps):
                if self._cancel.is_set():
                    self._emit_event(EVENT_CANCELLED, None)
                    return
                self._emit_event(
                    EVENT_STEP_START,
                    StepInfo(index=i, total=total, combo=step.combo, cmd=list(step.cmd), label=step.label),
                )
                rc, out = self._run_one(step.cmd, cwd, env)
                if self._cancel.is_set():
                    self._emit_event(EVENT_CANCELLED, None)
                    return
                if rc != 0:
                    failed += 1
                self._emit_event(
                    EVENT_STEP_RESULT,
                    StepResult(index=i, combo=step.combo, rc=rc, stdout=out, tool=plan.tool),
                )
            self._emit_event(EVENT_DONE, failed)
        except Exception as exc:
            self._emit_event(EVENT_ERROR, f"{type(exc).__name__}: {exc}")

    # ---------------------------------------------------------------- exec
    def _run_one(self, cmd: list[str], cwd: str | None, env: dict[str, str]) -> tuple[int, str]:
        """Run one process; stream stderr to the console, capture stdout.

        Returns ``(returncode, captured_stdout_text)``.
        """
        try:
            popen_kwargs: dict = {
                "cwd": cwd,
                "env": env,
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "bufsize": 0,
            }
            if os.name == "nt":
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            else:
                popen_kwargs["start_new_session"] = True
            proc = subprocess.Popen(cmd, **popen_kwargs)
        except FileNotFoundError as exc:
            self._emit_line(f"ERROR: Command not found: {cmd[0]} ({exc})")
            return 127, ""
        except Exception as exc:
            self._emit_line(f"ERROR: Failed to spawn {cmd[0]}: {exc}")
            return 1, ""

        with self._lock:
            self._proc = proc

        assert proc.stdout is not None and proc.stderr is not None
        captured = bytearray()
        capture_lock = threading.Lock()

        def _capture_stdout() -> None:
            try:
                while True:
                    chunk = proc.stdout.read(4096)  # type: ignore[union-attr]
                    if not chunk:
                        break
                    with capture_lock:
                        if len(captured) < MAX_CAPTURE_BYTES:
                            captured.extend(chunk)
            except Exception:
                pass

        stderr_reader = threading.Thread(target=self._stream_stderr, args=(proc.stderr,), daemon=True)
        stdout_reader = threading.Thread(target=_capture_stdout, daemon=True)
        stderr_reader.start()
        stdout_reader.start()
        try:
            while proc.poll() is None:
                if self._cancel.is_set():
                    self._signal_terminate(proc)
                    break
                time.sleep(PROC_POLL_SECONDS)
            rc = self._wait_for_proc_shutdown(proc)
        finally:
            for reader in (stderr_reader, stdout_reader):
                reader.join(timeout=PROC_KILL_WAIT_SECONDS)
            for stream in (proc.stdout, proc.stderr):
                try:
                    stream.close()  # type: ignore[union-attr]
                except Exception:
                    pass
        with capture_lock:
            text = bytes(captured).decode("utf-8", errors="replace")
        return rc, text

    def _stream_stderr(self, stderr) -> None:
        """Emit each stderr line as a console event."""
        buf = bytearray()
        max_line_len = 4096
        try:
            while True:
                chunk = stderr.read(4096)
                if not chunk:
                    break
                buf.extend(chunk)
                while True:
                    nl = -1
                    for i, b in enumerate(buf):
                        if b in (10, 13):
                            nl = i
                            break
                    if nl < 0:
                        if len(buf) >= max_line_len:
                            self._emit_line(buf.decode("utf-8", errors="replace"))
                            buf.clear()
                        break
                    sep = buf[nl]
                    self._emit_line(bytes(buf[:nl]).decode("utf-8", errors="replace"))
                    del buf[: nl + 1]
                    if sep == 13 and buf[:1] == b"\n":
                        del buf[:1]
            if buf:
                self._emit_line(buf.decode("utf-8", errors="replace"))
        except Exception as exc:
            if not self._cancel.is_set():
                self._emit_line(f"ERROR: stderr read error: {exc}")
