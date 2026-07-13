import subprocess
import sys
import threading
import time
import types

import pytest

from modules.build import build_runner
from modules.build import detection
from modules.build.build_runner import (
    BuildPlan,
    BuildRunner,
    EVENT_LINE,
    EVENT_QUEUE_MAXSIZE,
    plan_to_shell_script,
)


def test_build_runner_line_queue_is_bounded():
    runner = BuildRunner()

    for i in range(EVENT_QUEUE_MAXSIZE):
        runner._emit_line(f"line {i}")

    assert runner.events.qsize() == EVENT_QUEUE_MAXSIZE

    runner._emit_line("overflow")

    assert runner.events.qsize() == EVENT_QUEUE_MAXSIZE
    assert runner._dropped_output_lines == 1


def test_build_runner_reports_dropped_output_after_queue_catches_up():
    runner = BuildRunner()

    for i in range(EVENT_QUEUE_MAXSIZE):
        runner._emit_line(f"line {i}")
    runner._emit_line("overflow")
    runner.events.get_nowait()

    runner._emit_line("after overflow")

    queued = []
    while not runner.events.empty():
        queued.append(runner.events.get_nowait())

    assert any(kind == EVENT_LINE and "skipped 1 build output line" in str(payload) for kind, payload in queued)


def test_build_runner_shutdown_escalates_and_clears_proc(monkeypatch):
    runner = BuildRunner()

    class HangingProc:
        pid = 12345
        returncode = None

        def __init__(self):
            self.wait_timeouts = []

        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)
            if len(self.wait_timeouts) == 1:
                raise subprocess.TimeoutExpired(["fake-build"], timeout)
            self.returncode = -9
            return self.returncode

        def poll(self):
            return self.returncode

    proc = HangingProc()
    killed = []

    def record_kill(proc_to_kill):
        killed.append(proc_to_kill)

    monkeypatch.setattr(
        BuildRunner,
        "_signal_kill",
        staticmethod(record_kill),
    )

    with runner._lock:
        runner._proc = proc

    rc = runner._wait_for_proc_shutdown(proc)

    assert rc == -9
    assert killed == [proc]
    assert proc.wait_timeouts == [
        build_runner.PROC_TERMINATE_WAIT_SECONDS,
        build_runner.PROC_KILL_WAIT_SECONDS,
    ]
    with runner._lock:
        assert runner._proc is None


def test_build_runner_cancel_not_blocked_by_stdout_read(monkeypatch):
    runner = BuildRunner()
    runner._cancel.set()

    class BlockingStdout:
        def __init__(self):
            self.closed = threading.Event()

        def read(self, _chunk_size):
            self.closed.wait(timeout=10)
            return b""

        def close(self):
            self.closed.set()

    class HangingProc:
        pid = 12345

        def __init__(self):
            self.stdout = BlockingStdout()
            self.returncode = None
            self.wait_timeouts = []

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)
            if self.returncode is None:
                raise subprocess.TimeoutExpired(["fake-build"], timeout)
            return self.returncode

    proc = HangingProc()
    terminated = []
    killed = []

    monkeypatch.setattr(
        build_runner.subprocess,
        "Popen",
        lambda *args, **kwargs: proc,
    )
    monkeypatch.setattr(
        build_runner,
        "PROC_TERMINATE_WAIT_SECONDS",
        0.01,
    )
    monkeypatch.setattr(
        build_runner,
        "PROC_KILL_WAIT_SECONDS",
        0.01,
    )
    monkeypatch.setattr(
        build_runner,
        "PROC_POLL_SECONDS",
        0.001,
    )

    def record_terminate(proc_to_signal):
        terminated.append(proc_to_signal)

    def record_kill(proc_to_kill):
        killed.append(proc_to_kill)
        proc_to_kill.returncode = -9

    monkeypatch.setattr(
        BuildRunner,
        "_signal_terminate",
        staticmethod(record_terminate),
    )
    monkeypatch.setattr(
        BuildRunner,
        "_signal_kill",
        staticmethod(record_kill),
    )

    start = time.perf_counter()
    rc = runner._stream(["fake-build"], cwd=".")
    elapsed = time.perf_counter() - start

    assert rc == -9
    # Larger budget reduces sporadic CI flakiness on noisy Windows runners
    # without weakening the deadlock guard — anything ≥ several seconds
    # would still indicate the cancel cascade got stuck.
    assert elapsed < 3.0
    assert terminated == [proc]
    assert killed == [proc]
    assert proc.stdout.closed.is_set()
    with runner._lock:
        assert runner._proc is None


def test_recommend_jobs_does_not_default_to_all_logical_cpus(monkeypatch):
    fake_psutil = types.SimpleNamespace(
        cpu_count=lambda logical=False: 24 if logical is False else 48,
        virtual_memory=lambda: types.SimpleNamespace(total=377 * (1024**3)),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(detection.os, "cpu_count", lambda: 48)

    reco = detection.recommend_jobs()

    assert reco.suggested == 16
    assert reco.cpu_count == 48
    assert reco.physical_cores == 24


def test_recommend_jobs_caps_low_ram_hosts(monkeypatch):
    fake_psutil = types.SimpleNamespace(
        cpu_count=lambda logical=False: 4 if logical is False else 8,
        virtual_memory=lambda: types.SimpleNamespace(total=12 * (1024**3)),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(detection.os, "cpu_count", lambda: 8)

    reco = detection.recommend_jobs()

    assert reco.suggested == 3
    assert "RAM cap=3" in reco.reason


def test_saved_build_script_refuses_source_dir_as_build_dir(tmp_path):
    src = tmp_path / "checkout"
    plan = BuildPlan(
        backend="llama.cpp",
        source_dir=str(src),
        build_dir=str(src),
        cmake_args=[],
    )

    with pytest.raises(ValueError, match="Refusing unsafe build dir"):
        plan_to_shell_script(plan)


def test_saved_build_script_refuses_source_ancestor_as_build_dir(tmp_path):
    src = tmp_path / "checkout"
    plan = BuildPlan(
        backend="llama.cpp",
        source_dir=str(src),
        build_dir=str(tmp_path),
        cmake_args=[],
    )

    with pytest.raises(ValueError, match="Refusing unsafe build dir"):
        plan_to_shell_script(plan)


# ─────────────────────────────────────────────────────────────────────────────
# CR-4467921108 (Critical): ``clean_build=True`` recurse-delete must
# refuse non-empty directories that don't look like a build output.
# Guards against ``build_dir = "$HOME"`` typos becoming data loss.
# ─────────────────────────────────────────────────────────────────────────────


def test_assert_safe_to_purge_allows_missing_directory(tmp_path):
    target = tmp_path / "build-not-yet"
    # No exception — the launcher will create it.
    build_runner._assert_safe_to_purge(target)


def test_assert_safe_to_purge_allows_empty_directory(tmp_path):
    target = tmp_path / "build-empty"
    target.mkdir()
    build_runner._assert_safe_to_purge(target)


def test_assert_safe_to_purge_allows_cmake_cache(tmp_path):
    target = tmp_path / "build-with-cache"
    target.mkdir()
    (target / "CMakeCache.txt").write_text("# cmake stamp")
    (target / "random-output.o").write_bytes(b"object")
    build_runner._assert_safe_to_purge(target)


def test_assert_safe_to_purge_allows_launcher_marker(tmp_path):
    target = tmp_path / "build-with-marker"
    target.mkdir()
    (target / build_runner._BUILD_DIR_MARKER).touch()
    (target / "random-output.o").write_bytes(b"object")
    build_runner._assert_safe_to_purge(target)


def test_assert_safe_to_purge_rejects_non_empty_no_marker(tmp_path):
    target = tmp_path / "user-home-like"
    target.mkdir()
    (target / "important-thing.txt").write_text("hello")
    with pytest.raises(ValueError, match="does not look like a CMake/Ninja/Make build output"):
        build_runner._assert_safe_to_purge(target)


def test_plan_to_shell_script_embeds_purge_safety_gate(tmp_path):
    """``rm -rf $BUILD_DIR`` must be wrapped in the marker-check guard."""
    src = tmp_path / "checkout"
    src.mkdir()
    build_dir = tmp_path / "build"
    plan = BuildPlan(
        backend="llama.cpp",
        source_dir=str(src),
        build_dir=str(build_dir),
        cmake_args=[],
        clean_build=True,
    )
    script = plan_to_shell_script(plan)
    # The literal recurse-delete must NOT appear without the
    # surrounding guard. Pin both the guard and the conditional
    # ``rm -rf`` invocation.
    assert "CMakeCache.txt" in script
    assert build_runner._BUILD_DIR_MARKER in script
    assert "refusing to rm -rf" in script
    # The unconditional pre-CR line should be gone.
    lines = [line.strip() for line in script.splitlines()]
    assert 'rm -rf "$BUILD_DIR"' in lines  # appears inside the guard
    # The guard wraps the rm -rf inside an ``if [ -d "$BUILD_DIR" ]; then``
    # block, so the rm line must appear AFTER the existence check.
    rm_idx = lines.index('rm -rf "$BUILD_DIR"')
    guard_idx = lines.index('if [ -d "$BUILD_DIR" ]; then')
    assert guard_idx < rm_idx, "rm -rf must live inside the existence/marker guard"
