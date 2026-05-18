"""Build orchestration for the Build tab.

Threading model
---------------

All long-running work (git clone/pull/fetch, cmake configure, cmake --build,
git rev-list comparisons) runs on a background worker thread spawned by
``BuildRunner.start``. Output lines from the subprocess go onto a
``queue.Queue`` so the Tk UI can drain them on its mainloop tick without
calling tkinter from a non-main thread.

Each run is wrapped in a ``BuildJob`` with a single owning subprocess at a
time. ``cancel()`` terminates that subprocess (and the worker thread bails
out at the next pipeline boundary).

The runner never imports tkinter — it stays UI-framework agnostic so it can
be unit-tested headless.
"""

from __future__ import annotations

import os
import queue
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Event types pushed onto the runner's output queue
# ─────────────────────────────────────────────────────────────────────────────

EVENT_LINE = "line"           # ("line", text)
EVENT_STAGE = "stage"         # ("stage", name)
EVENT_DONE = "done"           # ("done", exit_code)
EVENT_CANCELLED = "cancelled" # ("cancelled", None)
EVENT_ERROR = "error"         # ("error", message)


# ─────────────────────────────────────────────────────────────────────────────
# Repos managed by the build tab
# ─────────────────────────────────────────────────────────────────────────────

UPSTREAMS = {
    "llama.cpp": "https://github.com/ggerganov/llama.cpp.git",
    "ik_llama": "https://github.com/ikawrakow/ik_llama.cpp.git",
}


# ─────────────────────────────────────────────────────────────────────────────
# Plan / job description
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BuildPlan:
    """All inputs needed to drive one configure-and-build run."""
    backend: str                          # "llama.cpp" | "ik_llama"
    source_dir: str                       # absolute path; clone target if missing
    build_dir: str                        # absolute (resolved by caller)
    cmake_args: List[str]                 # e.g. ["-DGGML_CUDA=ON", ...]
    cmake_env: Dict[str, str] = field(default_factory=dict)  # CC, CXX, CUDACXX, CUDA_TOOLKIT_ROOT_DIR
    jobs: int = 0                         # 0 => omit -j (cmake picks default)
    git_clone_if_missing: bool = True
    git_ref: str = ""                     # checkout this after clone/pull, if set
    git_pull_before_build: bool = False
    clean_build: bool = True
    generator: str = ""                   # "" = cmake default, else "Ninja", "Unix Makefiles", ...

    @property
    def upstream_url(self) -> str:
        return UPSTREAMS.get(self.backend, "")


# ─────────────────────────────────────────────────────────────────────────────
# Upstream-state probe (used by the update banner)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UpstreamStatus:
    is_git_repo: bool = False
    head_sha: str = ""
    head_subject: str = ""
    upstream_ref: str = ""        # "origin/main"
    behind: int = 0
    ahead: int = 0
    last_fetch_at: float = 0.0
    error: str = ""


def _run_capture(cmd: List[str], cwd: Optional[str] = None, timeout: float = 30.0) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True,
            timeout=timeout, check=False,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except Exception as exc:
        return 1, "", str(exc)


def probe_upstream(source_dir: str, *, do_fetch: bool = True) -> UpstreamStatus:
    """Determine how far behind ``source_dir`` is from its tracked upstream.

    When ``do_fetch`` is True we run ``git fetch --quiet`` first so the
    answer reflects current remote state. With False, we only consult what
    git already knows locally — useful for cheap on-tab-open status checks.
    Never raises.
    """
    status = UpstreamStatus()
    src = Path(source_dir or "")
    # .git can be a directory (regular repo) or a regular file (worktree /
    # submodule). exists() covers both.
    if not src.is_dir() or not (src / ".git").exists():
        return status
    status.is_git_repo = True

    if do_fetch:
        # 20s caps how long the UI's upstream-check banner is stale on a
        # slow / unreachable origin. The fetch runs on a background thread,
        # so this only blocks that worker, not the Tk mainloop.
        rc, _, err = _run_capture(["git", "fetch", "--quiet"], cwd=str(src), timeout=20.0)
        if rc != 0:
            status.error = (err or "git fetch failed").strip()
            # Don't bail — we can still report local-only state.

    rc, out, _ = _run_capture(
        ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
        cwd=str(src),
    )
    if rc != 0:
        if not status.error:
            status.error = "no upstream tracking branch configured"
        return status
    status.upstream_ref = out.strip()

    rc, out, _ = _run_capture(["git", "rev-parse", "HEAD"], cwd=str(src))
    if rc == 0:
        status.head_sha = out.strip()[:12]

    rc, out, _ = _run_capture(
        ["git", "log", "-1", "--pretty=%s", "HEAD"], cwd=str(src),
    )
    if rc == 0:
        status.head_subject = out.strip()

    rc, out, _ = _run_capture(
        ["git", "rev-list", "--left-right", "--count", f"HEAD...{status.upstream_ref}"],
        cwd=str(src),
    )
    if rc == 0 and out.strip():
        parts = out.strip().split()
        if len(parts) == 2:
            try:
                status.ahead = int(parts[0])
                status.behind = int(parts[1])
            except ValueError:
                pass

    status.last_fetch_at = time.time()
    return status


# ─────────────────────────────────────────────────────────────────────────────
# BuildRunner — owns the worker thread
# ─────────────────────────────────────────────────────────────────────────────

class BuildRunner:
    """Single-job pipeline runner. Reuse one instance for the tab lifetime."""

    def __init__(self) -> None:
        self.events: queue.Queue[Tuple[str, object]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._proc: Optional[subprocess.Popen] = None
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
            try:
                if os.name == "nt":
                    proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                else:
                    proc.terminate()
            except Exception:
                pass

    # ---------------------------------------------------------------- driver
    def start(self, plan: BuildPlan) -> bool:
        """Kick off a build in the background. Returns False if a job is
        already running; otherwise events stream onto ``self.events`` until
        the terminal ``done`` / ``cancelled`` / ``error`` event."""
        if self.is_running:
            return False
        self._cancel.clear()
        self._thread = threading.Thread(
            target=self._run, args=(plan,), name="BuildRunner", daemon=True,
        )
        self._thread.start()
        return True

    # ---------------------------------------------------------------- events
    def _emit_line(self, text: str) -> None:
        self.events.put((EVENT_LINE, text))

    def _emit_stage(self, name: str) -> None:
        self.events.put((EVENT_STAGE, name))
        self._emit_line(f"\n══ {name} ══")

    # ---------------------------------------------------------------- pipeline
    def _run(self, plan: BuildPlan) -> None:
        try:
            src = Path(plan.source_dir).expanduser()
            build = Path(plan.build_dir).expanduser()
            if not build.is_absolute():
                build = src / build

            # Stage: clone if needed
            if not src.exists():
                if not plan.git_clone_if_missing:
                    self.events.put((EVENT_ERROR, f"Source dir does not exist: {src}"))
                    return
                if not plan.upstream_url:
                    self.events.put((EVENT_ERROR, f"No upstream URL known for backend {plan.backend!r}"))
                    return
                src.parent.mkdir(parents=True, exist_ok=True)
                self._emit_stage(f"git clone {plan.upstream_url} → {src}")
                rc = self._stream(
                    ["git", "clone", "--recursive", plan.upstream_url, str(src)],
                    cwd=str(src.parent),
                )
                if self._cancel.is_set():
                    self.events.put((EVENT_CANCELLED, None))
                    return
                if rc != 0:
                    self.events.put((EVENT_DONE, rc))
                    return
            elif plan.git_pull_before_build:
                self._emit_stage("git pull --ff-only")
                rc = self._stream(["git", "pull", "--ff-only"], cwd=str(src))
                if self._cancel.is_set():
                    self.events.put((EVENT_CANCELLED, None))
                    return
                if rc != 0:
                    self._emit_line(f"git pull exited {rc}; continuing with current checkout")

            # Stage: optional checkout
            if plan.git_ref:
                self._emit_stage(f"git checkout {plan.git_ref}")
                rc = self._stream(["git", "checkout", plan.git_ref], cwd=str(src))
                if self._cancel.is_set():
                    self.events.put((EVENT_CANCELLED, None))
                    return
                if rc != 0:
                    self.events.put((EVENT_DONE, rc))
                    return
                self._stream(["git", "submodule", "update", "--init", "--recursive"], cwd=str(src))
                if self._cancel.is_set():
                    self.events.put((EVENT_CANCELLED, None))
                    return

            # Stage: clean
            if plan.clean_build and build.exists():
                self._emit_stage(f"rm -rf {build}")
                try:
                    shutil.rmtree(build)
                except Exception as exc:
                    self.events.put((EVENT_ERROR, f"Failed to clean build dir: {exc}"))
                    return

            build.mkdir(parents=True, exist_ok=True)

            # Stage: configure
            self._emit_stage("cmake configure")
            cfg_cmd = ["cmake", "-S", str(src), "-B", str(build)]
            if plan.generator:
                cfg_cmd += ["-G", plan.generator]
            cfg_cmd += list(plan.cmake_args)
            env = os.environ.copy()
            env.update(plan.cmake_env or {})
            self._emit_line("$ " + " ".join(shlex.quote(x) for x in cfg_cmd))
            rc = self._stream(cfg_cmd, cwd=str(src), env=env)
            if self._cancel.is_set():
                self.events.put((EVENT_CANCELLED, None))
                return
            if rc != 0:
                self.events.put((EVENT_DONE, rc))
                return

            # Stage: build
            jobs = plan.jobs if plan.jobs and plan.jobs > 0 else None
            build_cmd = ["cmake", "--build", str(build), "--config", "Release"]
            if jobs:
                build_cmd += ["-j", str(jobs)]
            self._emit_stage(f"cmake --build (-j{jobs or 'auto'})")
            self._emit_line("$ " + " ".join(shlex.quote(x) for x in build_cmd))
            rc = self._stream(build_cmd, cwd=str(src), env=env)
            if self._cancel.is_set():
                self.events.put((EVENT_CANCELLED, None))
                return
            self.events.put((EVENT_DONE, rc))
        except Exception as exc:
            self.events.put((EVENT_ERROR, f"{type(exc).__name__}: {exc}"))

    # ---------------------------------------------------------------- exec
    def _stream(
        self,
        cmd: List[str],
        cwd: str,
        env: Optional[Dict[str, str]] = None,
    ) -> int:
        """Run ``cmd``, stream its merged stdout/stderr to the events queue,
        and return its exit code. Sets ``self._proc`` so ``cancel()`` can
        signal it.

        We read in fixed-size byte chunks (not line-iteration) so a child
        that writes a huge unterminated chunk (carriage-return progress
        bars, binary blobs, partial flushes) doesn't block the worker
        accumulating an unbounded buffer. Each chunk is split on \\r and
        \\n boundaries and emitted as separate lines.
        """
        try:
            popen_kwargs: Dict[str, object] = {
                "cwd": cwd,
                "env": env,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "bufsize": 0,         # raw mode; we do our own buffering below
            }
            if os.name == "nt":
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            else:
                popen_kwargs["start_new_session"] = True

            proc = subprocess.Popen(cmd, **popen_kwargs)
        except FileNotFoundError as exc:
            self.events.put((EVENT_ERROR, f"Command not found: {cmd[0]} ({exc})"))
            return 127
        except Exception as exc:
            self.events.put((EVENT_ERROR, f"Failed to spawn {cmd[0]}: {exc}"))
            return 1

        with self._lock:
            self._proc = proc

        assert proc.stdout is not None
        chunk_size = 4096
        max_line_len = 4096
        buf = bytearray()
        try:
            while True:
                chunk = proc.stdout.read(chunk_size)
                if not chunk:
                    break
                buf.extend(chunk)
                # Split on \n; also break on \r so carriage-return progress
                # bars (compiler/linker) don't pile up into one giant line.
                while True:
                    nl = -1
                    for i, b in enumerate(buf):
                        if b == 0x0A or b == 0x0D:
                            nl = i
                            break
                    if nl < 0:
                        if len(buf) >= max_line_len:
                            # Flush oversized partial line so memory can't grow.
                            self._emit_line(buf.decode("utf-8", errors="replace"))
                            buf.clear()
                        break
                    line = bytes(buf[:nl]).decode("utf-8", errors="replace")
                    self._emit_line(line)
                    del buf[: nl + 1]
                if self._cancel.is_set() and proc.poll() is None:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
            if buf:
                self._emit_line(buf.decode("utf-8", errors="replace"))
        except Exception as exc:
            self.events.put((EVENT_ERROR, f"Stream read error: {exc}"))

        proc.wait()
        with self._lock:
            self._proc = None
        return int(proc.returncode or 0)


# ─────────────────────────────────────────────────────────────────────────────
# Shell-script emission (export a plan as a portable .sh file)
# ─────────────────────────────────────────────────────────────────────────────

def plan_to_shell_script(plan: BuildPlan, *, header: str = "") -> str:
    """Render a ``BuildPlan`` as a self-contained bash script equivalent to
    what BuildRunner would execute. Stable enough to commit/share.
    """
    lines: List[str] = []
    lines.append("#!/bin/bash")
    if header:
        for hl in header.splitlines():
            lines.append(f"# {hl}")
    else:
        lines.append(f"# Build script generated by llama-server-launcher")
        lines.append(f"# Backend: {plan.backend}")
    lines.append("set -e")
    lines.append("")

    src_q = shlex.quote(plan.source_dir)
    build_q = shlex.quote(plan.build_dir)

    lines.append(f"SRC_DIR={src_q}")
    lines.append(f"BUILD_DIR={build_q}")
    lines.append("")

    if plan.git_clone_if_missing and plan.upstream_url:
        lines.append('if [ ! -d "$SRC_DIR" ]; then')
        lines.append(f"  git clone --recursive {shlex.quote(plan.upstream_url)} \"$SRC_DIR\"")
        lines.append("fi")

    if plan.git_pull_before_build:
        lines.append('git -C "$SRC_DIR" pull --ff-only')

    if plan.git_ref:
        lines.append(f'git -C "$SRC_DIR" checkout {shlex.quote(plan.git_ref)}')
        lines.append('git -C "$SRC_DIR" submodule update --init --recursive')

    if plan.clean_build:
        lines.append('rm -rf "$BUILD_DIR"')

    lines.append('mkdir -p "$BUILD_DIR"')
    lines.append("")

    env_prefix = ""
    if plan.cmake_env:
        env_prefix = " ".join(
            f"{k}={shlex.quote(v)}" for k, v in plan.cmake_env.items()
        ) + " "

    cfg_head = f'{env_prefix}cmake -S "$SRC_DIR" -B "$BUILD_DIR"'
    if plan.generator:
        cfg_head += f" -G {shlex.quote(plan.generator)}"
    if plan.cmake_args:
        cfg_args = " \\\n  ".join(shlex.quote(a) for a in plan.cmake_args)
        lines.append(f"{cfg_head} \\")
        lines.append(f"  {cfg_args}")
    else:
        lines.append(cfg_head)
    lines.append("")

    jobs = plan.jobs if plan.jobs and plan.jobs > 0 else None
    jobs_arg = f" -j {jobs}" if jobs else " -j$(nproc)"
    lines.append(f'cmake --build "$BUILD_DIR" --config Release{jobs_arg}')
    lines.append("")
    return "\n".join(lines)
