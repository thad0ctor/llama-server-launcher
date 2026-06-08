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

    assert any(
        kind == EVENT_LINE and "skipped 1 build output line" in str(payload)
        for kind, payload in queued
    )


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
        virtual_memory=lambda: types.SimpleNamespace(total=377 * (1024 ** 3)),
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
        virtual_memory=lambda: types.SimpleNamespace(total=12 * (1024 ** 3)),
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
