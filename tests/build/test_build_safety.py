import sys
import types

import pytest

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
