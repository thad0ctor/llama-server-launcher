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


# ─────────────────────────────────────────────────────────────────────────────
# Event types pushed onto the runner's output queue
# ─────────────────────────────────────────────────────────────────────────────

EVENT_LINE = "line"           # ("line", text)
EVENT_STAGE = "stage"         # ("stage", name)
EVENT_DONE = "done"           # ("done", exit_code)
EVENT_CANCELLED = "cancelled" # ("cancelled", None)
EVENT_ERROR = "error"         # ("error", message)

EVENT_QUEUE_MAXSIZE = 4000
PROC_TERMINATE_WAIT_SECONDS = 5.0
PROC_KILL_WAIT_SECONDS = 2.0
PROC_POLL_SECONDS = 0.05


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
    cmake_args: list[str]                 # e.g. ["-DGGML_CUDA=ON", ...]
    cmake_env: dict[str, str] = field(default_factory=dict)  # CC, CXX, CUDACXX, CUDA_TOOLKIT_ROOT_DIR
    jobs: int = 0                         # 0 => omit -j (cmake picks default)
    git_clone_if_missing: bool = True
    git_ref: str = ""                     # checkout this after clone/pull, if set
    git_pull_before_build: bool = False
    clean_build: bool = True
    generator: str = ""                   # "" = cmake default, else "Ninja", "Unix Makefiles", ...

    @property
    def upstream_url(self) -> str:
        return UPSTREAMS.get(self.backend, "")


def _resolve_safe_build_paths(source_dir: str, build_dir: str) -> tuple[Path, Path]:
    """Resolve source/build paths and reject directories unsafe to delete."""
    # Reject empty inputs before resolving; ``Path("").expanduser()`` becomes
    # ``Path(".")``, which would have the runner / saved script silently
    # operating on the process CWD.
    if not str(source_dir).strip():
        raise ValueError("source_dir is empty; refusing to operate.")
    if not str(build_dir).strip():
        raise ValueError("build_dir is empty; refusing to operate.")

    src = Path(source_dir).expanduser()
    build = Path(build_dir).expanduser()
    # A user-supplied ``source_dir`` pointing at an existing FILE (not a
    # directory) would cause every downstream operation — ``git fetch``,
    # ``cmake -S``, etc. — to fail with a confusing tool-specific error
    # message. Catch it here with a clear cause.
    if src.exists() and src.is_file():
        raise ValueError(
            f"source_dir {source_dir!r} is a file, not a directory; "
            f"refusing to operate."
        )
    build_was_relative = not build.is_absolute()
    if build_was_relative:
        build = src / build
    # Same file-vs-directory guard for build_dir. Without this, a user
    # who typed a file path here would have ``rm -rf <build_dir>`` /
    # ``cmake -B <build_dir>`` fail with a tool-specific error instead
    # of a clear up-front message.
    if build.exists() and build.is_file():
        raise ValueError(
            f"build_dir {build_dir!r} is a file, not a directory; "
            f"refusing to operate."
        )

    try:
        build_resolved = build.resolve(strict=False)
        src_resolved = src.resolve(strict=False)
    except Exception as exc:
        raise ValueError(f"Could not resolve paths: {exc}") from exc

    if build_resolved == src_resolved or build_resolved in src_resolved.parents:
        raise ValueError(
            f"Refusing unsafe build dir {build_resolved!s} "
            f"(would target the source dir or an ancestor)."
        )

    # If the user supplied a RELATIVE build_dir (so we anchored it
    # under ``src``), make sure that after symlink-resolution it
    # actually stays inside ``src``. A devious value like
    # ``../../etc`` or a path component that's a symlink pointing
    # outside the source tree would otherwise let ``rm -rf
    # "$BUILD_DIR"`` reach unrelated directories. Absolute build_dirs
    # are user-explicit and can legitimately live anywhere.
    if build_was_relative:
        try:
            build_resolved.relative_to(src_resolved)
        except ValueError as exc:
            raise ValueError(
                f"Refusing relative build_dir {build_dir!r}: it resolves "
                f"to {build_resolved!s}, which is outside the source tree "
                f"{src_resolved!s} (likely via a parent traversal or a "
                f"symlinked path component)."
            ) from exc

    return src_resolved, build_resolved


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


def _run_capture(
    cmd: list[str],
    cwd: str | None = None,
    timeout: float = 30.0,
    *,
    env: dict | None = None,
    stdin=None,
) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True,
            timeout=timeout, check=False,
            env=env if env is not None else None,
            stdin=stdin if stdin is not None else subprocess.DEVNULL,
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
    # Refuse to fall through to ``Path(".")`` when no source has been
    # configured. Without this guard, running the launcher from inside a
    # git checkout would silently probe (and even ``git fetch``) the
    # current working directory repo and surface its upstream state in
    # the Build tab's update banner.
    if not str(source_dir or "").strip():
        return status
    # ``~/repos/foo`` is a valid value in the Build tab's source-dir entry;
    # without ``expanduser`` ``Path("~/...")`` would be treated as a
    # literal child directory of cwd, the ``.git`` check would fail, and
    # the probe would silently return "not a git repo".
    src = Path(source_dir).expanduser()
    # .git can be a directory (regular repo) or a regular file (worktree /
    # submodule). exists() covers both.
    if not src.is_dir() or not (src / ".git").exists():
        return status
    status.is_git_repo = True

    fetched = False
    if do_fetch:
        # 20s caps how long the UI's upstream-check banner is stale on a
        # slow / unreachable origin. The fetch runs on a background thread,
        # so this only blocks that worker, not the Tk mainloop.
        # Non-interactive ``git fetch``. Without ``GIT_TERMINAL_PROMPT=0``
        # and ``stdin=DEVNULL``, an HTTPS remote requiring auth (private
        # repo, expired credential helper) would block the worker thread
        # forever waiting on a credential prompt nobody can answer.
        fetch_env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        rc, _, err = _run_capture(
            ["git", "fetch", "--quiet"],
            cwd=str(src),
            timeout=20.0,
            env=fetch_env,
        )
        if rc != 0:
            status.error = (err or "git fetch failed").strip()
            # Don't bail — we can still report local-only state.
        else:
            fetched = True

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

    # Only stamp ``last_fetch_at`` when ``git fetch`` actually succeeded.
    # Callers use this to distinguish "remote state was refreshed" from
    # "this was a local-only / stale probe"; updating it on every probe
    # made the UI's "last refreshed N seconds ago" indicator lie when
    # fetch was skipped (do_fetch=False) or failed.
    if fetched:
        status.last_fetch_at = time.time()
    return status


# ─────────────────────────────────────────────────────────────────────────────
# BuildRunner — owns the worker thread
# ─────────────────────────────────────────────────────────────────────────────

class BuildRunner:
    """Single-job pipeline runner. Reuse one instance for the tab lifetime."""

    def __init__(self) -> None:
        # Bounded so a very noisy compiler/linker cannot grow Python memory
        # without limit if the Tk console falls behind. Line events may be
        # dropped under sustained pressure; terminal events block until the UI
        # drains space so completion/cancel state is still delivered.
        self.events: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=EVENT_QUEUE_MAXSIZE)
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._dropped_output_lines = 0

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

    @staticmethod
    def _signal_terminate(proc: subprocess.Popen) -> None:
        """Best-effort terminate of ``proc`` and any children it spawned.

        On POSIX, we start subprocesses with ``start_new_session=True``, which
        puts the child in its own process group. ``proc.terminate()`` would
        only signal the session leader and leave grandchildren (ninja → cc1,
        make → cc1) running. ``os.killpg`` sends SIGTERM to the whole group
        so a cancel actually stops the build.

        On Windows, sending CTRL_BREAK_EVENT to the new process group has the
        same effect.
        """
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
                    # Fall back to direct terminate if the group call fails
                    # (e.g. process already exited).
                    proc.terminate()
        except Exception:
            pass

    @staticmethod
    def _signal_kill(proc: subprocess.Popen) -> None:
        """Best-effort forced kill of ``proc`` and any children it spawned."""
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
        """Wait for process exit, escalating if graceful termination stalls."""
        try:
            try:
                return int(proc.wait(timeout=PROC_TERMINATE_WAIT_SECONDS) or 0)
            except subprocess.TimeoutExpired:
                self._emit_line(
                    "Process did not exit after termination; forcing shutdown."
                )
                self._signal_kill(proc)
                try:
                    return int(proc.wait(timeout=PROC_KILL_WAIT_SECONDS) or 0)
                except subprocess.TimeoutExpired:
                    self._emit_line("Process did not exit after forced shutdown.")
                    rc = proc.poll()
                    return int(rc) if rc is not None else 1
        finally:
            with self._lock:
                if self._proc is proc:
                    self._proc = None

    # ---------------------------------------------------------------- driver
    def start(self, plan: BuildPlan) -> bool:
        """Kick off a build in the background. Returns False if a job is
        already running; otherwise events stream onto ``self.events`` until
        the terminal ``done`` / ``cancelled`` / ``error`` event."""
        if self.is_running:
            return False
        # Drain any leftover events from the previous run before the new
        # one starts emitting. Without this, the UI consumer can poll
        # ``self.events`` between ``start()`` and the first new event
        # and see the old build's final ``done`` / ``cancelled`` event
        # again — racing the UI back into its "build finished" state.
        try:
            while True:
                self.events.get_nowait()
        except queue.Empty:
            pass
        # Same reasoning for the dropped-line counter — without this
        # reset, a status line like ``[3 lines dropped]`` from the
        # previous build would leak into the start of the next one.
        with self._lock:
            self._dropped_output_lines = 0
        self._cancel.clear()
        self._thread = threading.Thread(
            target=self._run, args=(plan,), name="BuildRunner", daemon=True,
        )
        self._thread.start()
        return True

    # ---------------------------------------------------------------- events
    def _take_dropped_count(self) -> int:
        """Atomically read-and-clear ``_dropped_output_lines``.

        Without the lock, ``_emit_line`` (reader thread) and ``_emit_event``
        (worker thread) could read-modify-write the counter concurrently and
        either double-emit the "[launcher skipped N lines]" notice or lose
        a few counts. Guarding read+clear behind ``_lock`` makes the
        counter sequentially consistent.
        """
        with self._lock:
            n = self._dropped_output_lines
            self._dropped_output_lines = 0
            return n

    def _bump_dropped_count(self) -> None:
        with self._lock:
            self._dropped_output_lines += 1

    def _emit_line(self, text: str) -> None:
        dropped = self._take_dropped_count()
        if dropped:
            notice = (
                f"[launcher skipped {dropped} build output line(s) "
                "while the UI caught up]"
            )
            try:
                self.events.put_nowait((EVENT_LINE, notice))
            except queue.Full:
                # Couldn't even fit the notice — re-credit the count plus
                # one for the line we're about to drop.
                with self._lock:
                    self._dropped_output_lines += dropped + 1
                return
        try:
            self.events.put_nowait((EVENT_LINE, text))
        except queue.Full:
            self._bump_dropped_count()

    def _emit_event(self, kind: str, payload: object) -> None:
        dropped = self._take_dropped_count()
        if dropped:
            self.events.put((
                EVENT_LINE,
                f"[launcher skipped {dropped} build output line(s) "
                "while the UI caught up]",
            ))
        self.events.put((kind, payload))

    def _emit_stage(self, name: str) -> None:
        self._emit_event(EVENT_STAGE, name)
        self._emit_line(f"\n══ {name} ══")

    # ---------------------------------------------------------------- pipeline
    def _run(self, plan: BuildPlan) -> None:
        try:
            try:
                src, build = _resolve_safe_build_paths(plan.source_dir, plan.build_dir)
            except ValueError as exc:
                self._emit_event(EVENT_ERROR, str(exc))
                return

            # Non-interactive git env shared by every live ``git`` invocation
            # below. Without ``GIT_TERMINAL_PROMPT=0`` a private repo, an
            # expired credential helper, or an SSH host-key prompt can wedge
            # the build forever waiting on stdin (which we already redirect to
            # DEVNULL, so the prompt would just hang). Mirrors the upstream
            # ``probe_upstream`` fetch.
            git_env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}

            # Helper: bail out at a stage boundary when the user has
            # cancelled. Without these checks, a cancel pressed BETWEEN
            # ``_stream`` calls would still let the next subprocess spin
            # up and only react when the current ``_stream`` returns.
            def _bail_if_cancelled() -> bool:
                if self._cancel.is_set():
                    self._emit_event(EVENT_CANCELLED, None)
                    return True
                return False

            # Stage: clone if needed
            if not src.exists():
                if not plan.git_clone_if_missing:
                    self._emit_event(EVENT_ERROR, f"Source dir does not exist: {src}")
                    return
                if not plan.upstream_url:
                    self._emit_event(EVENT_ERROR, f"No upstream URL known for backend {plan.backend!r}")
                    return
                if _bail_if_cancelled():
                    return
                src.parent.mkdir(parents=True, exist_ok=True)
                self._emit_stage(f"git clone {plan.upstream_url} → {src}")
                rc = self._stream(
                    ["git", "clone", "--recursive", plan.upstream_url, str(src)],
                    cwd=str(src.parent),
                    env=git_env,
                )
                if self._cancel.is_set():
                    self._emit_event(EVENT_CANCELLED, None)
                    return
                if rc != 0:
                    self._emit_event(EVENT_DONE, rc)
                    return
            elif plan.git_pull_before_build:
                if _bail_if_cancelled():
                    return
                self._emit_stage("git pull --ff-only")
                rc = self._stream(
                    ["git", "pull", "--ff-only"],
                    cwd=str(src),
                    env=git_env,
                )
                if self._cancel.is_set():
                    self._emit_event(EVENT_CANCELLED, None)
                    return
                if rc != 0:
                    self._emit_line(f"git pull exited {rc}; continuing with current checkout")

            # Stage: optional checkout
            if plan.git_ref:
                if _bail_if_cancelled():
                    return
                self._emit_stage(f"git checkout {plan.git_ref}")
                rc = self._stream(
                    ["git", "checkout", plan.git_ref],
                    cwd=str(src),
                    env=git_env,
                )
                if self._cancel.is_set():
                    self._emit_event(EVENT_CANCELLED, None)
                    return
                if rc != 0:
                    self._emit_event(EVENT_DONE, rc)
                    return
                if _bail_if_cancelled():
                    return
                rc = self._stream(
                    ["git", "submodule", "update", "--init", "--recursive"],
                    cwd=str(src),
                    env=git_env,
                )
                if self._cancel.is_set():
                    self._emit_event(EVENT_CANCELLED, None)
                    return
                if rc != 0:
                    # Submodules not fetched cleanly — surfacing this is much
                    # nicer than letting cmake fail later on a missing header.
                    self._emit_event(EVENT_DONE, rc)
                    return

            # Stage: clean
            if plan.clean_build and build.exists():
                if _bail_if_cancelled():
                    return
                self._emit_stage(f"rm -rf {build}")
                try:
                    shutil.rmtree(build)
                except Exception as exc:
                    self._emit_event(EVENT_ERROR, f"Failed to clean build dir: {exc}")
                    return

            if _bail_if_cancelled():
                return
            build.mkdir(parents=True, exist_ok=True)

            # Stage: configure
            if _bail_if_cancelled():
                return
            self._emit_stage("cmake configure")
            cfg_cmd = ["cmake", "-S", str(src), "-B", str(build)]
            if plan.generator:
                cfg_cmd += ["-G", plan.generator]
            cfg_cmd += list(plan.cmake_args)
            env = os.environ.copy()
            # Same env-name validation the exported script does (see
            # ``_ENV_NAME_RE`` in build_persistence.py). A programmatic
            # ``BuildPlan`` could ship a key like ``"CC FLAGS"`` /
            # ``"CC;echo pwned"`` that ``subprocess.Popen`` would
            # reject with a confusing ``ValueError`` mid-run instead
            # of being caught up front.
            import re as _re
            _env_name_re = _re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
            for k, v in (plan.cmake_env or {}).items():
                if not (isinstance(k, str) and _env_name_re.fullmatch(k)):
                    self._emit_line(
                        f"WARNING: dropping cmake_env entry with invalid name "
                        f"{k!r} (must match {_env_name_re.pattern!r})"
                    )
                    continue
                # Coerce values to str AND reject embedded NULs.
                # ``subprocess.Popen(env=...)`` rejects non-string
                # values with a confusing TypeError, and NUL in a
                # value breaks the C-level env emitter on POSIX
                # / corrupts the bash export line in the exported
                # script.
                try:
                    sv = str(v) if v is not None else ""
                except Exception:
                    self._emit_line(
                        f"WARNING: dropping cmake_env value for {k!r}: "
                        f"could not coerce to str"
                    )
                    continue
                if "\0" in sv:
                    self._emit_line(
                        f"WARNING: dropping cmake_env entry {k!r}: "
                        f"value contains NUL"
                    )
                    continue
                env[k] = sv
            self._emit_line("$ " + " ".join(shlex.quote(x) for x in cfg_cmd))
            rc = self._stream(cfg_cmd, cwd=str(src), env=env)
            if self._cancel.is_set():
                self._emit_event(EVENT_CANCELLED, None)
                return
            if rc != 0:
                self._emit_event(EVENT_DONE, rc)
                return

            # Stage: build
            if _bail_if_cancelled():
                return
            jobs = plan.jobs if plan.jobs and plan.jobs > 0 else None
            build_cmd = ["cmake", "--build", str(build), "--config", "Release"]
            if jobs:
                build_cmd += ["-j", str(jobs)]
            self._emit_stage(f"cmake --build (-j{jobs or 'auto'})")
            self._emit_line("$ " + " ".join(shlex.quote(x) for x in build_cmd))
            rc = self._stream(build_cmd, cwd=str(src), env=env)
            if self._cancel.is_set():
                self._emit_event(EVENT_CANCELLED, None)
                return
            self._emit_event(EVENT_DONE, rc)
        except Exception as exc:
            self._emit_event(EVENT_ERROR, f"{type(exc).__name__}: {exc}")

    # ---------------------------------------------------------------- exec
    def _stream(
        self,
        cmd: list[str],
        cwd: str,
        env: dict[str, str] | None = None,
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
            # Build the Popen call with explicit, typed kwargs (rather
            # than ``**dict[str, object]``) so mypy can match the overload.
            # ``stdin=DEVNULL`` prevents git/credential prompts from blocking
            # the build forever; ``bufsize=0`` keeps reads raw so we can do
            # our own line splitting below.
            if os.name == "nt":
                proc = subprocess.Popen(
                    cmd,
                    cwd=cwd,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=0,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,  # type: ignore[attr-defined]
                )
            else:
                proc = subprocess.Popen(
                    cmd,
                    cwd=cwd,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=0,
                    start_new_session=True,
                )
        except FileNotFoundError as exc:
            # _run is the sole emitter of terminal events; here we only emit
            # a diagnostic line and return a non-zero rc so _run can decide.
            self._emit_line(f"ERROR: Command not found: {cmd[0]} ({exc})")
            return 127
        except Exception as exc:
            self._emit_line(f"ERROR: Failed to spawn {cmd[0]}: {exc}")
            return 1

        with self._lock:
            self._proc = proc

        assert proc.stdout is not None
        reader = threading.Thread(
            target=self._stream_reader,
            args=(proc.stdout,),
            name="BuildRunnerOutput",
            daemon=True,
        )
        reader.start()
        try:
            while proc.poll() is None:
                if self._cancel.is_set():
                    self._signal_terminate(proc)
                    break
                time.sleep(PROC_POLL_SECONDS)
            return self._wait_for_proc_shutdown(proc)
        finally:
            reader.join(timeout=PROC_KILL_WAIT_SECONDS)
            if reader.is_alive():
                try:
                    proc.stdout.close()
                except Exception:
                    pass
                reader.join(timeout=PROC_POLL_SECONDS)

    def _stream_reader(self, stdout) -> None:
        """Drain subprocess output without owning process lifetime control."""
        chunk_size = 4096
        max_line_len = 4096
        buf = bytearray()
        try:
            while True:
                chunk = stdout.read(chunk_size)
                if not chunk:
                    break
                buf.extend(chunk)
                # Split on \n; also break on \r so carriage-return progress
                # bars (compiler/linker) don't pile up into one giant line.
                # CRLF (Windows builds) is consumed as a single break to
                # avoid spamming the console with empty lines.
                while True:
                    nl = -1
                    for i, b in enumerate(buf):
                        if b in (10, 13):
                            nl = i
                            break
                    if nl < 0:
                        if len(buf) >= max_line_len:
                            # Flush oversized partial line so memory can't grow.
                            self._emit_line(buf.decode("utf-8", errors="replace"))
                            buf.clear()
                        break
                    sep = buf[nl]
                    line = bytes(buf[:nl]).decode("utf-8", errors="replace")
                    self._emit_line(line)
                    del buf[: nl + 1]
                    # If the break was CR and the next byte is LF, consume
                    # the LF too so CRLF doesn't emit a phantom empty line.
                    if sep == 13 and buf[:1] == b"\n":
                        del buf[:1]
            if buf:
                self._emit_line(buf.decode("utf-8", errors="replace"))
        except Exception as exc:
            # Emit a diagnostic line; _run is the sole emitter of terminal
            # events so don't put EVENT_ERROR on the queue here.
            if not self._cancel.is_set():
                self._emit_line(f"ERROR: Stream read error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Shell-script emission (export a plan as a portable .sh file)
# ─────────────────────────────────────────────────────────────────────────────

def plan_to_shell_script(plan: BuildPlan, *, header: str = "") -> str:
    """Render a ``BuildPlan`` as a self-contained bash script equivalent to
    what BuildRunner would execute. Stable enough to commit/share.
    """
    src, build = _resolve_safe_build_paths(plan.source_dir, plan.build_dir)

    lines: list[str] = []
    # Env-based shebang matches the saved launch script in
    # ``modules/launch.py`` so a Homebrew, NixOS, or user-installed
    # bash on PATH wins. Hosts where bash isn't at ``/bin``
    # otherwise produce a script that fails immediately on execute.
    lines.append("#!/usr/bin/env bash")
    if header:
        for hl in header.splitlines():
            lines.append(f"# {hl}")
    else:
        lines.append(f"# Build script generated by llama-server-launcher")
        lines.append(f"# Backend: {plan.backend}")
    lines.append("set -e")
    # Close stdin for the WHOLE script so no downstream tool can block
    # waiting on input the user can't provide. ``BuildRunner._run`` /
    # ``_stream`` already pass ``stdin=subprocess.DEVNULL`` to every
    # Popen; the exported script needs the same treatment via a single
    # ``exec </dev/null`` at the top.
    lines.append("exec </dev/null")
    lines.append("")
    # Forbid interactive credential prompts from git. Without this an
    # exported script that hits a private repo or a stale auth cache
    # would still try (the closed stdin would make the prompt fail
    # rather than block, but the explicit flag makes the intent clear).
    # ``_run()`` enforces the same via ``stdin=DEVNULL`` + this env var
    # on the in-app Popen.
    lines.append("export GIT_TERMINAL_PROMPT=0")
    lines.append("")

    src_q = shlex.quote(str(src))
    build_q = shlex.quote(str(build))

    lines.append(f"SRC_DIR={src_q}")
    lines.append(f"BUILD_DIR={build_q}")
    lines.append("")

    # Refuse to proceed if either path EXISTS but is a regular file
    # (or another non-directory shape — symlink to file, socket, etc).
    # ``_resolve_safe_build_paths`` only runs at script-GENERATION time;
    # this guard handles the case where the same SRC/BUILD path got
    # repurposed between script export and script execution. Without
    # the guard, the script would happily continue into ``git`` or
    # ``rm -rf "$BUILD_DIR"`` against the wrong thing.
    lines.append('if [ -e "$SRC_DIR" ] && [ ! -d "$SRC_DIR" ]; then')
    lines.append('  echo "ERROR: source path is not a directory: $SRC_DIR" >&2')
    lines.append("  exit 1")
    lines.append("fi")
    lines.append('if [ -e "$BUILD_DIR" ] && [ ! -d "$BUILD_DIR" ]; then')
    lines.append('  echo "ERROR: build path is not a directory: $BUILD_DIR" >&2')
    lines.append("  exit 1")
    lines.append("fi")
    lines.append("")

    # Mirror ``_run()``'s precondition checks. Without these the exported
    # script silently proceeds into ``cmake -S`` against a non-existent
    # SRC_DIR (which then fails much further down with a confusing cmake
    # error message), and a misconfigured clone (clone-if-missing requested
    # but no upstream URL) would silently skip the clone and produce the
    # same downstream failure.
    if plan.git_clone_if_missing and plan.upstream_url:
        lines.append('if [ ! -d "$SRC_DIR" ]; then')
        # _run() implicitly relies on Popen's cwd being writable; the exported
        # script has no such caller, so create the parent explicitly.
        lines.append('  mkdir -p "$(dirname "$SRC_DIR")"')
        lines.append(f"  git clone --recursive {shlex.quote(plan.upstream_url)} \"$SRC_DIR\"")
        lines.append("fi")
    elif plan.git_clone_if_missing and not plan.upstream_url:
        # Misconfiguration — clone requested without a URL. Fail with a
        # clear message rather than letting the next ``git -C "$SRC_DIR"``
        # invocation surface as a generic "not a git repository" error.
        lines.append('if [ ! -d "$SRC_DIR" ]; then')
        lines.append('  echo "ERROR: git_clone_if_missing is set but no upstream URL was configured." >&2')
        lines.append("  exit 1")
        lines.append("fi")
    else:
        # No clone requested — SRC_DIR must already exist.
        lines.append('if [ ! -d "$SRC_DIR" ]; then')
        lines.append('  echo "ERROR: source directory does not exist: $SRC_DIR" >&2')
        lines.append("  exit 1")
        lines.append("fi")

    if plan.git_pull_before_build:
        # Mirror _run()'s tolerance: a failed `git pull --ff-only` (offline,
        # diverged history, …) must not abort the rest of the build under
        # `set -e`. _run() just logs and continues.
        lines.append('if ! git -C "$SRC_DIR" pull --ff-only; then')
        lines.append('  echo "git pull failed; continuing with current checkout" >&2')
        lines.append("fi")

    if plan.git_ref:
        lines.append(f'git -C "$SRC_DIR" checkout {shlex.quote(plan.git_ref)}')
        lines.append('git -C "$SRC_DIR" submodule update --init --recursive')

    if plan.clean_build:
        lines.append('rm -rf "$BUILD_DIR"')

    lines.append('mkdir -p "$BUILD_DIR"')
    lines.append("")

    env_prefix = ""
    if plan.cmake_env:
        # Defense in depth: ``BuildConfig.from_json`` already filters
        # ``cmake_env`` keys through ``_ENV_NAME_RE`` at load time, but
        # a ``BuildPlan`` can also be constructed programmatically (the
        # tests do this) or via a future loader that skips the
        # persistence layer. Re-validate KEY AND VALUE here so the
        # exported script never emits a shell-unsafe ``CC FLAGS="x"`` /
        # ``CC;echo pwned="x"`` line AND can't trip ``shlex.quote(v)``
        # on a non-string ``v`` or embed a NUL in a bash export.
        import re as _re
        _env_name_re = _re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
        safe_env_items: list[tuple[str, str]] = []
        for k, v in plan.cmake_env.items():
            if not (isinstance(k, str) and _env_name_re.fullmatch(k)):
                continue
            try:
                sv = str(v) if v is not None else ""
            except Exception:
                continue
            if "\0" in sv:
                continue
            safe_env_items.append((k, sv))
        env_prefix = " ".join(
            f"{k}={shlex.quote(v)}" for k, v in safe_env_items
        )
        if env_prefix:
            env_prefix += " "

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

    # Match BuildRunner: omit -j entirely when jobs is unset so the script
    # is portable (macOS has no `nproc` by default) and lets cmake pick the
    # default parallelism. The runner has identical semantics.
    jobs = plan.jobs if plan.jobs and plan.jobs > 0 else None
    jobs_arg = f" -j {jobs}" if jobs else ""
    # Apply ``env_prefix`` to ``cmake --build`` too — ``BuildRunner._run``
    # passes ``plan.cmake_env`` to BOTH the configure and build stages, so
    # the exported shell script must do the same or an in-app build that
    # depends on e.g. ``CUDACXX=/path/to/nvcc`` for the build step will
    # silently fail when run from the exported script.
    lines.append(f'{env_prefix}cmake --build "$BUILD_DIR" --config Release{jobs_arg}')
    lines.append("")
    return "\n".join(lines)
