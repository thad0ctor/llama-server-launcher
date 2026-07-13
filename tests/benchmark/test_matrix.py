"""Tests for the sweep-matrix engine (value expansion + command building)."""

from __future__ import annotations

import pytest

from modules.benchmark.detection import TOOL_LLAMA_BENCH, TOOL_SWEEP_BENCH
from modules.benchmark.matrix import (
    Axis,
    SweepError,
    build_commands,
    expand_range,
    llama_bench_command,
    matrix_size,
    parse_list,
    split_axes_for_tool,
    sweep_bench_commands,
)


def test_parse_list_int_normalises_and_dedupes():
    assert parse_list("0, 08, 20, 20", "int") == ["0", "8", "20"]


def test_parse_list_str_preserves():
    assert parse_list("f16, q8_0", "str") == ["f16", "q8_0"]


def test_parse_list_fa_normalises():
    assert parse_list("on, 0, true, off", "fa") == ["on", "off"]


def test_parse_list_bad_int_raises():
    with pytest.raises(SweepError):
        parse_list("abc", "int")


def test_parse_list_int_rejects_float():
    # A non-integer token must not be coerced to float (FIX 2).
    with pytest.raises(SweepError):
        parse_list("0.5,2", "int")


def test_parse_list_int_allows_negative():
    assert parse_list("-1, 0, 8", "int") == ["-1", "0", "8"]


def test_expand_range_inclusive():
    assert expand_range(0, 33, 11, "int") == ["0", "11", "22", "33"]


def test_expand_range_zero_step_raises():
    with pytest.raises(SweepError):
        expand_range(0, 10, 0, "int")


def test_expand_range_runaway_guard():
    with pytest.raises(SweepError):
        expand_range(0, 100000, 1, "int")


def test_expand_range_contradictory_step_raises():
    # Step sign disagrees with min→max direction (FIX 3).
    with pytest.raises(SweepError):
        expand_range(0, 10, -2, "int")


def test_expand_range_positive_step_still_works():
    assert expand_range(0, 10, 2, "int") == ["0", "2", "4", "6", "8", "10"]


def test_llama_bench_single_command_with_lists():
    axes = [Axis("n_gpu_layers", ["0", "11", "22"]), Axis("threads", ["8", "16"])]
    cmd = llama_bench_command("/b/llama-bench", "/m.gguf", axes)
    assert cmd == [
        "/b/llama-bench",
        "-m",
        "/m.gguf",
        "-ngl",
        "0,11,22",
        "-t",
        "8,16",
        "-o",
        "json",
    ]


def test_llama_bench_repetitions_and_extra():
    axes = [Axis("threads", ["8"])]
    cmd = llama_bench_command("/b/llama-bench", "/m.gguf", axes, repetitions=3, extra_args="--numa distribute")
    assert "-r" in cmd and "3" in cmd
    assert cmd[-2:] == ["--numa", "distribute"]


def test_llama_bench_ignores_sweep_only_lever():
    # ctx_size (-c) is sweep-bench-only; llama-bench must skip it.
    axes = [Axis("ctx_size", ["4096"]), Axis("threads", ["8"])]
    applicable, ignored = split_axes_for_tool(axes, TOOL_LLAMA_BENCH)
    assert [a.key for a in applicable] == ["threads"]
    assert [a.key for a in ignored] == ["ctx_size"]


def test_sweep_bench_cartesian_product():
    axes = [Axis("n_gpu_layers", ["0", "10"]), Axis("threads", ["8", "16"])]
    pairs = sweep_bench_commands("/b/llama-sweep-bench", "/m.gguf", axes)
    assert len(pairs) == 4
    combos = [combo for _, combo in pairs]
    assert {"n_gpu_layers": "0", "threads": "8"} in combos
    assert {"n_gpu_layers": "10", "threads": "16"} in combos


def test_sweep_bench_flash_attn_bare_flag():
    axes = [Axis("flash_attn", ["on", "off"])]
    pairs = sweep_bench_commands("/b/llama-sweep-bench", "/m.gguf", axes)
    assert len(pairs) == 2
    on_cmd = next(cmd for cmd, combo in pairs if combo["flash_attn"] == "on")
    off_cmd = next(cmd for cmd, combo in pairs if combo["flash_attn"] == "off")
    assert "-fa" in on_cmd
    assert "-fa" not in off_cmd


def _fa_arg(cmd):
    """Return the value rendered after the ``-fa`` flag in a llama-bench cmd."""
    idx = cmd.index("-fa")
    return cmd[idx + 1]


def test_llama_bench_fa_default_backend_literal():
    axes = [Axis("flash_attn", ["on", "off"])]
    cmd = llama_bench_command("/b/llama-bench", "/m.gguf", axes)
    assert _fa_arg(cmd) == "on,off"


def test_llama_bench_fa_llama_cpp_backend_literal():
    axes = [Axis("flash_attn", ["on", "off"])]
    cmd = llama_bench_command("/b/llama-bench", "/m.gguf", axes, backend="llama.cpp")
    assert _fa_arg(cmd) == "on,off"


def test_llama_bench_fa_ik_llama_backend_numeric():
    axes = [Axis("flash_attn", ["on", "off"])]
    cmd = llama_bench_command("/b/llama-bench", "/m.gguf", axes, backend="ik_llama")
    assert _fa_arg(cmd) == "1,0"


def test_build_commands_forwards_backend_to_fa_rendering():
    axes = [Axis("flash_attn", ["on", "off"])]
    pairs = build_commands(TOOL_LLAMA_BENCH, "/b/llama-bench", "/m.gguf", axes, backend="ik_llama")
    assert len(pairs) == 1
    assert _fa_arg(pairs[0][0]) == "1,0"


def test_sweep_bench_fa_still_bare_flag_regardless_of_backend():
    # sweep-bench renders FA as a bare flag; no backend parameter involved.
    axes = [Axis("flash_attn", ["on", "off"])]
    pairs = sweep_bench_commands("/b/llama-sweep-bench", "/m.gguf", axes)
    on_cmd = next(cmd for cmd, combo in pairs if combo["flash_attn"] == "on")
    off_cmd = next(cmd for cmd, combo in pairs if combo["flash_attn"] == "off")
    assert "-fa" in on_cmd and "1" not in on_cmd
    assert "-fa" not in off_cmd


def test_matrix_size():
    axes = [Axis("n_gpu_layers", ["0", "11", "22"]), Axis("threads", ["8", "16"]), Axis("ctx_size", ["4096"])]
    # ctx_size ignored for llama-bench -> 3 * 2 = 6
    assert matrix_size(axes, TOOL_LLAMA_BENCH) == 6
    # all three apply to sweep-bench -> 3 * 2 * 1 = 6
    assert matrix_size(axes, TOOL_SWEEP_BENCH) == 6


def test_build_commands_llama_bench_returns_single_pair():
    axes = [Axis("threads", ["8", "16"])]
    pairs = build_commands(TOOL_LLAMA_BENCH, "/b/llama-bench", "/m.gguf", axes)
    assert len(pairs) == 1
    assert pairs[0][1] == {}


def test_build_commands_unknown_tool():
    with pytest.raises(SweepError):
        build_commands("nope", "/b/x", "/m.gguf", [Axis("threads", ["8"])])


def test_missing_model_raises():
    with pytest.raises(SweepError):
        llama_bench_command("/b/llama-bench", "", [Axis("threads", ["8"])])
