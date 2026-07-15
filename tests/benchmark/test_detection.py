"""Tests for benchmark-binary discovery across configured builds."""

from __future__ import annotations

import stat
import sys
from types import SimpleNamespace

import pytest

from modules.benchmark.detection import (
    TOOL_LLAMA_BENCH,
    TOOL_SWEEP_BENCH,
    discover_builds,
    find_bench_executable,
)


def _make_exe(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _exe_name(tool):
    return f"{tool}.exe" if sys.platform == "win32" else tool


def test_find_bench_executable_in_build_bin(tmp_path):
    exe = tmp_path / "build" / "bin" / _exe_name(TOOL_LLAMA_BENCH)
    _make_exe(exe)
    found = find_bench_executable(tmp_path, TOOL_LLAMA_BENCH)
    assert found == exe.resolve()


def test_find_bench_executable_in_server_dir(tmp_path):
    # `server/` must be searched for parity with LaunchManager (FIX 5).
    exe = tmp_path / "server" / _exe_name(TOOL_LLAMA_BENCH)
    _make_exe(exe)
    found = find_bench_executable(tmp_path, TOOL_LLAMA_BENCH)
    assert found == exe.resolve()


def test_find_bench_executable_missing_returns_none(tmp_path):
    assert find_bench_executable(tmp_path, TOOL_SWEEP_BENCH) is None


def test_find_bench_executable_empty_dir():
    assert find_bench_executable("", TOOL_LLAMA_BENCH) is None


def _launcher(tmp_path, llama_dir="", ik_dir=""):
    return SimpleNamespace(
        llama_cpp_dir=SimpleNamespace(get=lambda: llama_dir),
        ik_llama_dir=SimpleNamespace(get=lambda: ik_dir),
        config_path=str(tmp_path / "cfg.json"),
    )


def test_discover_builds_backend_roots(tmp_path):
    cpp = tmp_path / "cpp"
    ik = tmp_path / "ik"
    _make_exe(cpp / "build" / "bin" / _exe_name(TOOL_LLAMA_BENCH))
    _make_exe(ik / "build" / "bin" / _exe_name(TOOL_LLAMA_BENCH))
    _make_exe(ik / "build" / "bin" / _exe_name(TOOL_SWEEP_BENCH))

    launcher = _launcher(tmp_path, str(cpp), str(ik))
    builds = discover_builds(launcher)
    by_backend = {b.backend: b for b in builds}

    assert set(by_backend) == {"llama.cpp", "ik_llama"}
    # llama.cpp build only has llama-bench
    assert by_backend["llama.cpp"].available_tools() == [TOOL_LLAMA_BENCH]
    # ik build has both
    assert by_backend["ik_llama"].available_tools() == [TOOL_LLAMA_BENCH, TOOL_SWEEP_BENCH]


def test_discover_builds_skips_roots_without_tools(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    launcher = _launcher(tmp_path, str(empty), "")
    assert discover_builds(launcher) == []


def test_discover_builds_dedupes_same_root(tmp_path):
    cpp = tmp_path / "cpp"
    _make_exe(cpp / "build" / "bin" / _exe_name(TOOL_LLAMA_BENCH))
    # Same path configured for both backend slots.
    launcher = _launcher(tmp_path, str(cpp), str(cpp))
    builds = discover_builds(launcher)
    assert len(builds) == 1
