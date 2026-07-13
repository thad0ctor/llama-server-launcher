"""Tests for benchmark .sh / .ps1 script export."""

from __future__ import annotations

import pytest

from modules.benchmark import bench_script


def test_sh_single_command():
    sh = bench_script.to_sh([["/b/llama-bench", "-m", "/m.gguf", "-ngl", "0,10"]])
    assert sh.startswith("#!/usr/bin/env bash")
    assert "set -e" in sh
    assert "exec </dev/null" in sh
    assert "/b/llama-bench -m /m.gguf -ngl 0,10" in sh


def test_sh_quotes_spaces():
    sh = bench_script.to_sh([["/b/llama bench", "-m", "/path with space/m.gguf"]])
    assert "'/b/llama bench'" in sh
    assert "'/path with space/m.gguf'" in sh


def test_sh_multi_command_numbered():
    cmds = [["/b/x", "-c", "4096"], ["/b/x", "-c", "8192"]]
    sh = bench_script.to_sh(cmds)
    assert "== benchmark 1/2 ==" in sh
    assert "== benchmark 2/2 ==" in sh


def test_ps1_single_quoted_literals():
    ps = bench_script.to_ps1([["C:/b/llama-bench.exe", "-m", "C:/models/m.gguf"]])
    assert "$ErrorActionPreference = 'Stop'" in ps
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
