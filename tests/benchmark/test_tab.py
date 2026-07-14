"""Headless tests for BenchmarkTab poll-loop robustness.

Uses the shared ``tk_root`` fixture (auto-skips when no display is available).
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace

import pytest

from modules.benchmark import BenchmarkTab


def _fake_launcher(tk_root, tmp_path):
    ns = SimpleNamespace(
        root=tk_root,
        config_path=str(tmp_path / "cfg.json"),
        model_path=tk.StringVar(master=tk_root, value=""),
        model_dirs=[],
        backend_selection=tk.StringVar(master=tk_root, value="llama.cpp"),
        llama_cpp_dir=tk.StringVar(master=tk_root, value=""),
        ik_llama_dir=tk.StringVar(master=tk_root, value=""),
    )
    for attr in (
        "n_gpu_layers",
        "threads",
        "batch_size",
        "ubatch_size",
        "cache_type_k",
        "cache_type_v",
        "tensor_split",
        "main_gpu",
    ):
        setattr(ns, attr, tk.StringVar(master=tk_root, value=""))
    ns.flash_attn = tk.BooleanVar(master=tk_root, value=True)
    ns.ctx_size = tk.IntVar(master=tk_root, value=4096)
    return ns


@pytest.fixture
def bench_tab(tk_root, tmp_path):
    launcher = _fake_launcher(tk_root, tmp_path)
    tab = BenchmarkTab(launcher)
    tab.register_with_notebook(None, "Benchmark")
    frame = ttk.Frame(tk_root)
    tab.setup_tab(frame)
    tk_root.update_idletasks()
    return tab


def test_poll_runner_liveness_fallback_recovers_ui(bench_tab):
    # Simulate a run whose terminal event was dropped (queue saturation +
    # cancel): the UI shows "running", but the worker is dead and the queue is
    # empty. The poll loop must leave the running state instead of spinning
    # forever with Start disabled.
    bench_tab.status_var.set("Running 1/2…")
    bench_tab._set_running(True)
    assert str(bench_tab._start_btn["state"]) == "disabled"

    # runner was never started -> is_running is False and events is empty.
    bench_tab._poll_runner()

    assert str(bench_tab._start_btn["state"]) == "normal"
    assert str(bench_tab._cancel_btn["state"]) == "disabled"
    assert bench_tab._poll_after_id is None
    assert bench_tab.status_var.get() == "Stopped."


def test_poll_runner_reschedules_while_worker_alive(bench_tab, monkeypatch):
    # When the worker is still alive, the loop must reschedule (not finalise).
    monkeypatch.setattr(type(bench_tab.runner), "is_running", property(lambda self: True))
    scheduled = {"n": 0}
    monkeypatch.setattr(bench_tab.root, "after", lambda *a, **k: scheduled.__setitem__("n", scheduled["n"] + 1) or "id")
    bench_tab._set_running(True)
    bench_tab._poll_runner()
    assert scheduled["n"] == 1
    assert str(bench_tab._start_btn["state"]) == "disabled"


def test_set_tool_keeps_canonical_id_and_display_label(bench_tab):
    from modules.benchmark.detection import TOOL_SWEEP_BENCH

    bench_tab._set_tool(TOOL_SWEEP_BENCH)
    # Canonical id in tool_var; human label in the combobox display var.
    assert bench_tab.tool_var.get() == TOOL_SWEEP_BENCH
    assert "(ik_llama)" in bench_tab._tool_display_var.get()


def test_load_sweep_bench_config_keeps_tool(bench_tab):
    # Regression (Codex P2): loading a sweep-bench config on an ik_llama build
    # that offers both tools must NOT revert the run to llama-bench.
    from modules.benchmark.bench_persistence import BenchConfig
    from modules.benchmark.detection import TOOL_LLAMA_BENCH, TOOL_SWEEP_BENCH, BuildEntry

    build = BuildEntry(
        label="ik (backend dir)",
        backend="ik_llama",
        root_dir="/ik",
        source="backend",
        tools={TOOL_LLAMA_BENCH: "/ik/llama-bench", TOOL_SWEEP_BENCH: "/ik/llama-sweep-bench"},
    )
    bench_tab._builds = [build]
    bench_tab._build_labels = [build.label]
    bench_tab._build_combo.configure(values=bench_tab._build_labels)
    bench_tab.build_var.set(build.label)

    bench_tab.store.save(
        BenchConfig(
            name="sweep",
            tool=TOOL_SWEEP_BENCH,
            backend="ik_llama",
            build_root="/ik",
            model_path="/m.gguf",
            axes={"ctx_size": {"enabled": True, "mode": "list", "raw": "4096", "min": 0, "max": 0, "step": 1}},
        )
    )
    bench_tab.config_name_var.set("sweep")
    bench_tab._load_config()

    assert bench_tab.tool_var.get() == TOOL_SWEEP_BENCH


def test_multiple_extra_args_rows_combine_in_command(bench_tab):
    from modules.benchmark.detection import TOOL_LLAMA_BENCH

    bench_tab._set_tool(TOOL_LLAMA_BENCH)
    bench_tab.model_var.set("/m.gguf")
    # Tick one built-in lever so a command is buildable.
    bench_tab.lever_vars["threads"]["include"].set(True)
    bench_tab.lever_vars["threads"]["mode"].set("list")
    bench_tab.lever_vars["threads"]["values"].set("8")
    # Two extra-args rows.
    bench_tab._set_extra_args_rows(["--numa distribute", "--no-mmap"])
    commands, _ = bench_tab._build_command_list()
    assert len(commands) == 1
    cmd = commands[0]
    # Combined, shlex-split tokens appear (in order) at the tail.
    assert cmd[-3:] == ["--numa", "distribute", "--no-mmap"]


def test_custom_sweep_flag_yields_two_combos(bench_tab):
    from modules.benchmark.detection import TOOL_SWEEP_BENCH

    bench_tab._set_tool(TOOL_SWEEP_BENCH)
    bench_tab.model_var.set("/m.gguf")
    bench_tab._set_custom_axis_rows(
        [{"flag": "-ot", "enabled": True, "mode": "list", "raw": "exps=CPU,attn=CPU", "min": 0, "max": 0, "step": 1}]
    )
    commands, combos = bench_tab._build_command_list()
    assert len(commands) == 2
    assert {c["-ot"] for c in combos} == {"exps=CPU", "attn=CPU"}
    for cmd, combo in zip(commands, combos):
        idx = cmd.index("-ot")
        assert cmd[idx + 1] == combo["-ot"]


def test_invalid_custom_flag_raises_sweep_error(bench_tab):
    from modules.benchmark.matrix import SweepError

    bench_tab._set_custom_axis_rows(
        [{"flag": "ot", "enabled": True, "mode": "list", "raw": "a,b", "min": 0, "max": 0, "step": 1}]
    )
    with pytest.raises(SweepError):
        bench_tab._collect_axes()


def test_save_load_roundtrips_extra_args_and_custom_axes(bench_tab):
    from modules.benchmark.detection import TOOL_SWEEP_BENCH

    bench_tab._set_tool(TOOL_SWEEP_BENCH)
    bench_tab._set_extra_args_rows(["--numa distribute", "--no-mmap"])
    bench_tab._set_custom_axis_rows(
        [{"flag": "-ot", "enabled": True, "mode": "list", "raw": "exps=CPU,attn=CPU", "min": 0, "max": 0, "step": 1}]
    )
    bench_tab.config_name_var.set("rt")
    bench_tab._save_config()

    # Wipe live state, then load it back.
    bench_tab._set_extra_args_rows([])
    bench_tab._set_custom_axis_rows([])
    bench_tab._load_config()

    assert bench_tab._current_extra_args() == ["--numa distribute", "--no-mmap"]
    assert len(bench_tab._custom_axis_rows) == 1
    row = bench_tab._custom_axis_rows[0]
    assert row["flag"].get() == "-ot"
    assert row["values"].get() == "exps=CPU,attn=CPU"
    assert bool(row["include"].get()) is True


def test_stop_polling_on_teardown_cancels_running_worker(bench_tab, monkeypatch):
    calls = {"cancel": 0}
    monkeypatch.setattr(type(bench_tab.runner), "is_running", property(lambda self: True))
    monkeypatch.setattr(bench_tab.runner, "cancel", lambda: calls.__setitem__("cancel", calls["cancel"] + 1))
    bench_tab._poll_after_id = "some-id"
    bench_tab._stop_polling_on_teardown()
    assert calls["cancel"] == 1
    assert bench_tab._poll_after_id is None
