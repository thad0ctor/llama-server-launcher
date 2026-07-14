"""Tests for benchmark .sh / .ps1 script export."""

from __future__ import annotations

import pytest

from modules.benchmark import bench_script


def test_sh_single_command():
    sh = bench_script.to_sh([["/b/llama-bench", "-m", "/m.gguf", "-ngl", "0,10"]])
    assert sh.startswith("#!/usr/bin/env bash")
    assert "set -e" not in sh  # must not fail-fast across a sweep
    assert "exec </dev/null" in sh
    assert "/b/llama-bench -m /m.gguf -ngl 0,10" in sh
    # A single command's own exit status is the script's; no spurious exit line
    # or aggregate rc bookkeeping.
    assert "exit" not in sh
    assert "_bench_rc" not in sh


def test_sh_quotes_spaces():
    sh = bench_script.to_sh([["/b/llama bench", "-m", "/path with space/m.gguf"]])
    assert "'/b/llama bench'" in sh
    assert "'/path with space/m.gguf'" in sh


def test_sh_multi_command_numbered():
    cmds = [["/b/x", "-c", "4096"], ["/b/x", "-c", "8192"]]
    sh = bench_script.to_sh(cmds)
    assert "== benchmark 1/2 ==" in sh
    assert "== benchmark 2/2 ==" in sh


def test_sh_continues_but_exits_nonzero_on_bad_combo():
    # No `set -e`: a failed combo must not abort the rest of the matrix
    # (mirrors BenchRunner). Failures are surfaced AND recorded so the script's
    # final exit status is nonzero when any combo failed.
    cmds = [["/b/x", "-c", "4096"], ["/b/x", "-c", "8192"]]
    sh = bench_script.to_sh(cmds)
    assert "set -e" not in sh
    assert "FAILED" in sh
    # A failing command records the aggregate rc but does not abort.
    assert "_bench_rc=1" in sh
    assert "_bench_rc=0" in sh
    assert 'exit "$_bench_rc"' in sh


def test_ps1_continues_but_exits_nonzero_on_failure():
    cmds = [["/b/x.exe", "-c", "4096"], ["/b/x.exe", "-c", "8192"]]
    ps = bench_script.to_ps1(cmds)
    assert "$ErrorActionPreference = 'Continue'" in ps
    assert "'Stop'" not in ps
    assert "$LASTEXITCODE" in ps
    # Failures are recorded and the script exits nonzero at the end, but each
    # combo still runs (continue, not fail-fast).
    assert "$_benchRc = 1" in ps
    assert "$_benchRc = 0" in ps
    assert "exit $_benchRc" in ps


def test_ps1_single_command_no_exit():
    ps = bench_script.to_ps1([["/b/x.exe", "-c", "4096"]])
    assert "exit" not in ps
    assert "$_benchRc" not in ps


def test_ps1_single_quoted_literals():
    ps = bench_script.to_ps1([["C:/b/llama-bench.exe", "-m", "C:/models/m.gguf"]])
    assert "$ErrorActionPreference = 'Continue'" in ps
    assert "& 'C:/b/llama-bench.exe' '-m' 'C:/models/m.gguf'" in ps


def test_ps1_escapes_single_quote():
    ps = bench_script.to_ps1([["/b/it's", "-m", "/m.gguf"]])
    assert "'/b/it''s'" in ps


def test_render_dispatch():
    cmds = [["/b/x", "-c", "1"]]
    assert bench_script.render(cmds, "sh").startswith("#!/usr/bin/env bash")
    assert "$ErrorActionPreference" in bench_script.render(cmds, "ps1")
    with pytest.raises(ValueError):
        bench_script.render(cmds, "bogus")
