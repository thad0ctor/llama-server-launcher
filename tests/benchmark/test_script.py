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
    # Each command's success is captured via $? (a native exe that fails to
    # launch leaves $LASTEXITCODE stale), and launch failure is a distinct
    # branch from a nonzero exit so a stale $LASTEXITCODE isn't reported.
    assert "$_benchSucceeded = $?" in ps
    assert "if (-not $_benchSucceeded) { $_benchRc = 1;" in ps
    assert "elseif ($LASTEXITCODE -ne 0)" in ps
    # Failures are recorded and the script exits nonzero at the end, but each
    # combo still runs (continue, not fail-fast).
    assert "$_benchRc = 1" in ps
    assert "$_benchRc = 0" in ps
    assert "exit $_benchRc" in ps


def test_ps1_single_command_propagates_failure():
    # A single-command -File script must reflect the native command's failure in
    # its own exit status (otherwise a failing exe reports success). It captures
    # $? and exits 1 when the command never ran, else propagates $LASTEXITCODE.
    ps = bench_script.to_ps1([["/b/x.exe", "-c", "4096"]])
    assert "$_benchSucceeded = $?" in ps
    assert "if (-not $_benchSucceeded) { exit 1 }" in ps
    assert "exit $LASTEXITCODE" in ps
    # No aggregate rc bookkeeping for a single command.
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


def test_sh_env_prelude_precedes_command():
    sh = bench_script.to_sh([["/b/llama-bench", "-m", "/m.gguf"]], env={"CUDA_VISIBLE_DEVICES": "0,1"})
    assert "export CUDA_VISIBLE_DEVICES='0,1'" in sh
    # The prelude must come before the command so the tool sees the GPUs.
    assert sh.index("export CUDA_VISIBLE_DEVICES") < sh.index("/b/llama-bench")


def test_ps1_env_prelude_precedes_command():
    ps = bench_script.to_ps1([["/b/x.exe", "-m", "/m.gguf"]], env={"CUDA_VISIBLE_DEVICES": "0,1"})
    assert "$env:CUDA_VISIBLE_DEVICES = '0,1'" in ps
    assert ps.index("$env:CUDA_VISIBLE_DEVICES") < ps.index("/b/x.exe")


def test_env_prelude_sorted_and_quote_escaped():
    sh = bench_script.to_sh([["/b/x"]], env={"B": "2", "A": "o'ne"})
    # Sorted keys: A before B.
    assert sh.index("export A=") < sh.index("export B=")
    # Embedded single quote is escaped for bash.
    assert "export A='o'\\''ne'" in sh


def test_no_env_prelude_when_none_or_empty():
    assert "export " not in bench_script.to_sh([["/b/x", "-c", "1"]], env=None)
    assert "export " not in bench_script.to_sh([["/b/x", "-c", "1"]], env={})
    assert "$env:" not in bench_script.to_ps1([["/b/x.exe"]], env=None)
    assert "$env:" not in bench_script.to_ps1([["/b/x.exe"]], env={})


def test_env_prelude_skips_shell_active_keys():
    # A KEY with shell-active characters would inject script syntax into the
    # prelude; it must be dropped (valid keys still emitted) for both formats.
    env = {"CUDA_VISIBLE_DEVICES": "0", "BAD;touch": "x"}
    sh = bench_script.to_sh([["/b/x"]], env=env)
    assert "export CUDA_VISIBLE_DEVICES='0'" in sh
    assert "BAD" not in sh
    ps = bench_script.to_ps1([["/b/x.exe"]], env=env)
    assert "$env:CUDA_VISIBLE_DEVICES = '0'" in ps
    assert "BAD" not in ps


def test_render_forwards_env():
    sh = bench_script.render([["/b/x", "-c", "1"]], "sh", env={"CUDA_VISIBLE_DEVICES": "0"})
    assert "export CUDA_VISIBLE_DEVICES='0'" in sh
    ps = bench_script.render([["/b/x.exe"]], "ps1", env={"CUDA_VISIBLE_DEVICES": "0"})
    assert "$env:CUDA_VISIBLE_DEVICES = '0'" in ps


def test_env_unset_emits_removal():
    # The resolver's "unset" action must actively REMOVE an inherited stale var.
    sh = bench_script.to_sh([["/b/x"]], env_unset=["CUDA_VISIBLE_DEVICES"])
    assert "unset CUDA_VISIBLE_DEVICES" in sh
    ps = bench_script.to_ps1([["/b/x.exe"]], env_unset=["CUDA_VISIBLE_DEVICES"])
    assert "Remove-Item -Path Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue" in ps
    # Illegal names are dropped from the unset prelude too.
    assert "BAD;x" not in bench_script.to_sh([["/b/x"]], env_unset=["BAD;x"])


def test_ps1_multi_command_splits_launch_from_exit_failure():
    ps = bench_script.to_ps1([["/b/x.exe", "-c", "1"], ["/b/x.exe", "-c", "2"]])
    # Launch failure ($? false) is reported WITHOUT the stale $LASTEXITCODE.
    assert "if (-not $_benchSucceeded) { $_benchRc = 1;" in ps
    assert "FAILED (launch error)" in ps
    # A nonzero exit is a distinct, elseif branch that does print $LASTEXITCODE.
    assert "elseif ($LASTEXITCODE -ne 0)" in ps
    assert "FAILED (exit $LASTEXITCODE)" in ps


def test_cwd_prelude_sh_and_ps1():
    # Saved scripts must cd into the runner's working directory so relative
    # paths (e.g. --lora adapter.gguf) resolve identically to the in-app run.
    sh = bench_script.to_sh([["/b/x", "-m", "m"]], cwd="/opt/ik/build/bin")
    assert "cd -- '/opt/ik/build/bin' || exit 1" in sh
    ps = bench_script.to_ps1([["x.exe"]], cwd="C:/ik/bin")
    assert "Set-Location -LiteralPath 'C:/ik/bin'" in ps
    # No cd/Set-Location when cwd is empty.
    assert "cd --" not in bench_script.to_sh([["/b/x"]])
    assert "Set-Location" not in bench_script.to_ps1([["x.exe"]])
