"""Tests for the sweep-matrix engine (value expansion + command building)."""

from __future__ import annotations

import pytest

from modules.benchmark.detection import TOOL_LLAMA_BENCH, TOOL_SWEEP_BENCH
from modules.benchmark.matrix import (
    KIND_VEC,
    MAX_SWEEP_COMBOS,
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


def test_tensor_split_is_one_value_not_comma_split():
    # A split vector like "0.6,0.4" is a single --tensor-split argument; commas
    # are internal, so it must NOT expand into separate sweep points.
    assert parse_list("0.6,0.4", KIND_VEC) == ["0.6,0.4"]
    assert parse_list("1,1", KIND_VEC) == ["1,1"]  # not deduped to "1"
    # Multiple splits sweep on ';'.
    assert parse_list("0.6,0.4;0.7,0.3", KIND_VEC) == ["0.6,0.4", "0.7,0.3"]


def test_tensor_split_sweep_bench_command_keeps_vector():
    axes = [Axis("tensor_split", ["0.6,0.4"])]
    pairs = sweep_bench_commands("/b/llama-sweep-bench", "/m.gguf", axes)
    assert len(pairs) == 1
    # sweep-bench uses server-style params: devices stay comma-separated (raw
    # vector), unchanged.
    assert pairs[0][0] == ["/b/llama-sweep-bench", "-m", "/m.gguf", "-ts", "0.6,0.4"]


def test_tensor_split_llama_bench_uses_device_separator():
    # llama-bench splits -ts on ',' into separate benchmark CASES and on '/'
    # (or ';') into devices WITHIN a vector, so a single 2-GPU split must render
    # as ONE device-separated token, not two 1-GPU cases.
    axes = [Axis("tensor_split", ["0.6,0.4"])]
    cmd = llama_bench_command("/b/llama-bench", "/m.gguf", axes)
    idx = cmd.index("-ts")
    assert cmd[idx + 1] == "0.6/0.4"


def test_tensor_split_llama_bench_multiple_vectors():
    # Two swept splits: devices joined by '/' within each vector, vectors joined
    # by ',' as separate benchmark cases.
    axes = [Axis("tensor_split", ["0.6,0.4", "0.7,0.3"])]
    cmd = llama_bench_command("/b/llama-bench", "/m.gguf", axes)
    idx = cmd.index("-ts")
    assert cmd[idx + 1] == "0.6/0.4,0.7/0.3"


def test_sweep_bench_matrix_cap_rejects_explosion():
    # Two big axes whose product exceeds the cap must raise before materialising.
    big = [str(i) for i in range(200)]
    axes = [Axis("n_gpu_layers", big), Axis("threads", big)]  # 200*200 = 40000
    assert matrix_size(axes, TOOL_SWEEP_BENCH) > MAX_SWEEP_COMBOS
    with pytest.raises(SweepError):
        sweep_bench_commands("/b/llama-sweep-bench", "/m.gguf", axes)


def test_malformed_extra_args_raises_sweep_error():
    # Unmatched quote -> shlex ValueError -> surfaced as SweepError, not a raw
    # exception escaping the Tk callback.
    with pytest.raises(SweepError):
        llama_bench_command("/b/llama-bench", "/m.gguf", [Axis("threads", ["8"])], extra_args='--numa "distribute')


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


def test_sweep_bench_flash_attn_explicit_value():
    # sweep-bench uses ik_llama's server-style --flash-attn, which requires an
    # explicit on/off token (a bare flag errors, and "off" must be explicit to
    # override the binary default), mirroring modules/launch.py.
    axes = [Axis("flash_attn", ["on", "off"])]
    pairs = sweep_bench_commands("/b/llama-sweep-bench", "/m.gguf", axes)
    assert len(pairs) == 2
    on_cmd = next(cmd for cmd, combo in pairs if combo["flash_attn"] == "on")
    off_cmd = next(cmd for cmd, combo in pairs if combo["flash_attn"] == "off")
    assert on_cmd[on_cmd.index("-fa") + 1] == "on"
    assert off_cmd[off_cmd.index("-fa") + 1] == "off"


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


def test_sweep_bench_fa_explicit_value_no_backend_param():
    # sweep-bench renders FA with an explicit on/off value and takes no backend
    # parameter (unlike llama-bench, it never maps to numeric 1/0).
    axes = [Axis("flash_attn", ["on", "off"])]
    pairs = sweep_bench_commands("/b/llama-sweep-bench", "/m.gguf", axes)
    on_cmd = next(cmd for cmd, combo in pairs if combo["flash_attn"] == "on")
    off_cmd = next(cmd for cmd, combo in pairs if combo["flash_attn"] == "off")
    assert on_cmd[on_cmd.index("-fa") + 1] == "on"
    assert off_cmd[off_cmd.index("-fa") + 1] == "off"
    assert "1" not in on_cmd and "0" not in off_cmd


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


# ─────────────────────────────────────────────────────────────────────────────
# Custom-flag axes
# ─────────────────────────────────────────────────────────────────────────────


def _custom_axis(flag, values, kind=None):
    from modules.benchmark.matrix import KIND_STR

    return Axis(key=flag, values=values, custom_flag=flag, custom_kind=kind or KIND_STR)


def test_builtin_axis_still_exposes_lever_and_uniform_accessors():
    # Regression: the built-in construction and the ``lever`` property must keep
    # working, and the new uniform accessors must delegate to the lever.
    axis = Axis("threads", ["8", "16"])
    assert axis.is_custom is False
    assert axis.lever.key == "threads"
    assert axis.flag == "-t"
    assert axis.label == "Threads (-t)"
    assert axis.applies_to(TOOL_LLAMA_BENCH)
    assert axis.applies_to(TOOL_SWEEP_BENCH)


def test_custom_axis_sweep_bench_renders_flag_value_per_combo():
    axes = [_custom_axis("-ot", ["exps=CPU", "attn=CPU"])]
    pairs = sweep_bench_commands("/b/llama-sweep-bench", "/m.gguf", axes)
    assert len(pairs) == 2
    # Each combo renders ``<flag> <value>`` and is tagged with the flag as key.
    for cmd, combo in pairs:
        val = combo["-ot"]
        idx = cmd.index("-ot")
        assert cmd[idx + 1] == val
    assert {combo["-ot"] for _, combo in pairs} == {"exps=CPU", "attn=CPU"}


def test_custom_axis_llama_bench_renders_comma_list():
    axes = [_custom_axis("--cache-reuse", ["0", "256"])]
    cmd = llama_bench_command("/b/llama-bench", "/m.gguf", axes)
    idx = cmd.index("--cache-reuse")
    assert cmd[idx + 1] == "0,256"


def test_custom_axis_applies_to_both_tools():
    axis = _custom_axis("-ot", ["exps=CPU"])
    assert axis.is_custom is True
    assert axis.applies_to(TOOL_LLAMA_BENCH)
    assert axis.applies_to(TOOL_SWEEP_BENCH)


def test_custom_flag_without_dash_raises():
    with pytest.raises(SweepError):
        Axis(key="ot", values=["exps=CPU"], custom_flag="ot")


def test_custom_axis_counts_toward_sweep_cap():
    # A custom axis applies to sweep-bench, so the MAX_SWEEP_COMBOS guard must
    # include it in the product.
    big = [str(i) for i in range(200)]
    axes = [Axis("n_gpu_layers", big), _custom_axis("-ot", big)]  # 200*200 = 40000
    assert matrix_size(axes, TOOL_SWEEP_BENCH) > MAX_SWEEP_COMBOS
    with pytest.raises(SweepError):
        sweep_bench_commands("/b/llama-sweep-bench", "/m.gguf", axes)


def test_custom_axis_matrix_size_both_tools():
    axes = [_custom_axis("-ot", ["a", "b", "c"]), Axis("threads", ["8", "16"])]
    assert matrix_size(axes, TOOL_LLAMA_BENCH) == 6
    assert matrix_size(axes, TOOL_SWEEP_BENCH) == 6


# ─────────────────────────────────────────────────────────────────────────────
# ik_llama-only levers (backend-gated)
# ─────────────────────────────────────────────────────────────────────────────


def test_ik_lever_applies_only_to_ik_llama_bench():
    from modules.benchmark.matrix import LEVERS_BY_KEY

    rtr = LEVERS_BY_KEY["rtr"]
    # Applies to llama-bench on the ik_llama backend only.
    assert rtr.applies_to(TOOL_LLAMA_BENCH, "ik_llama")
    # NOT on the llama.cpp backend (the flag doesn't exist upstream).
    assert not rtr.applies_to(TOOL_LLAMA_BENCH, "llama.cpp")
    # NOT on llama-sweep-bench even under ik_llama (scoped to llama-bench).
    assert not rtr.applies_to(TOOL_SWEEP_BENCH, "ik_llama")


def test_ik_lever_renders_native_comma_sweep_on_llama_bench():
    axes = [Axis("rtr", ["0", "1"])]
    pairs = build_commands(TOOL_LLAMA_BENCH, "/b/llama-bench", "/m.gguf", axes, backend="ik_llama")
    assert len(pairs) == 1
    cmd = pairs[0][0]
    idx = cmd.index("-rtr")
    assert cmd[idx + 1] == "0,1"


def test_mqkv_is_value_lever_not_bare_flag():
    # -mqkv takes a <0|1> value (NOT a bare flag): it renders as a native
    # comma-sweepable value on ik_llama's llama-bench and is dropped on llama.cpp.
    from modules.benchmark.matrix import LEVERS_BY_KEY

    mqkv = LEVERS_BY_KEY["mqkv"]
    assert mqkv.applies_to(TOOL_LLAMA_BENCH, "ik_llama")
    assert not mqkv.applies_to(TOOL_LLAMA_BENCH, "llama.cpp")
    assert not mqkv.applies_to(TOOL_SWEEP_BENCH, "ik_llama")

    axes = [Axis("mqkv", ["0", "1"])]
    pairs = build_commands(TOOL_LLAMA_BENCH, "/b/llama-bench", "/m.gguf", axes, backend="ik_llama")
    cmd = pairs[0][0]
    idx = cmd.index("-mqkv")
    assert cmd[idx + 1] == "0,1"
    # Dropped entirely on the llama.cpp backend.
    applicable, ignored = split_axes_for_tool(axes, TOOL_LLAMA_BENCH, "llama.cpp")
    assert applicable == []
    assert [a.key for a in ignored] == ["mqkv"]


def test_ik_lever_dropped_for_llama_cpp_backend():
    # The same axes on the llama.cpp backend must drop the ik-only axis.
    axes = [Axis("rtr", ["0", "1"]), Axis("threads", ["8"])]
    applicable, ignored = split_axes_for_tool(axes, TOOL_LLAMA_BENCH, "llama.cpp")
    assert [a.key for a in applicable] == ["threads"]
    assert [a.key for a in ignored] == ["rtr"]
    # And it applies under ik_llama.
    applicable_ik, ignored_ik = split_axes_for_tool(axes, TOOL_LLAMA_BENCH, "ik_llama")
    assert {a.key for a in applicable_ik} == {"rtr", "threads"}
    assert ignored_ik == []


def test_ser_vec_not_device_separated_on_llama_bench():
    # -ser's value is literally "i,f" (e.g. 7,1) — a single token that must NOT
    # have its comma rewritten to the tensor-split device separator.
    axes = [Axis("ser", ["7,1"])]
    cmd = llama_bench_command("/b/llama-bench", "/m.gguf", axes, backend="ik_llama")
    idx = cmd.index("-ser")
    assert cmd[idx + 1] == "7,1"


def test_ot_vec_pattern_passes_through_on_llama_bench():
    # -ot patterns use '=' and ',' literally; no device-separator conversion.
    axes = [Axis("ot", ["exps=CPU"])]
    cmd = llama_bench_command("/b/llama-bench", "/m.gguf", axes, backend="ik_llama")
    idx = cmd.index("-ot")
    assert cmd[idx + 1] == "exps=CPU"


def test_tensor_split_still_device_separated_after_gating():
    # Regression: gating the vec conversion to tensor_split must leave -ts's
    # 0.6,0.4 -> 0.6/0.4 rewrite intact.
    axes = [Axis("tensor_split", ["0.6,0.4"])]
    cmd = llama_bench_command("/b/llama-bench", "/m.gguf", axes)
    idx = cmd.index("-ts")
    assert cmd[idx + 1] == "0.6/0.4"
