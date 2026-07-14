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


def test_gpu_env_overrides_carries_cuda_visible_devices(bench_tab):
    # Finding 1: a plan built with a known GPU selection must carry the same
    # CUDA_VISIBLE_DEVICES the server launch would export, so the benchmark
    # child sees the user's logical GPU subset/order (not all physical GPUs).
    bench_tab.launcher.launch_manager = SimpleNamespace(
        _resolve_cuda_visible_devices_action=lambda: ("export", "2,0,1")
    )
    env = bench_tab._gpu_env_overrides()
    assert env == {"CUDA_VISIBLE_DEVICES": "2,0,1"}


def test_gpu_env_overrides_unset_leaves_var_unset(bench_tab):
    # 'unset'/'skip' actions intentionally add no override (child sees all GPUs,
    # matching server-launch behaviour in manual / all-deselected modes).
    bench_tab.launcher.launch_manager = SimpleNamespace(_resolve_cuda_visible_devices_action=lambda: ("unset", None))
    assert bench_tab._gpu_env_overrides() == {}


def test_gpu_env_overrides_defensive_without_launch_manager(bench_tab):
    # A launcher without a usable launch_manager must not raise and yields no
    # override. (The fixture's fake launcher has no launch_manager attribute.)
    assert bench_tab._gpu_env_overrides() == {}


def test_start_blocks_when_no_axis_applies_to_tool(bench_tab, monkeypatch, tmp_path):
    # Finding 2: ticking a llama-bench-only lever then selecting sweep-bench must
    # block with a clear message instead of building one unswept default command.
    from modules.benchmark import bench_tab as bench_tab_mod
    from modules.benchmark.detection import TOOL_SWEEP_BENCH

    bench_tab._set_tool(TOOL_SWEEP_BENCH)
    model = tmp_path / "m.gguf"
    model.write_text("x")
    bench_tab.model_var.set(str(model))
    # -p (Prompt tokens) applies ONLY to llama-bench.
    bench_tab.lever_vars["n_prompt"]["include"].set(True)
    bench_tab.lever_vars["n_prompt"]["mode"].set("list")
    bench_tab.lever_vars["n_prompt"]["values"].set("128")

    monkeypatch.setattr(bench_tab, "_current_exe", lambda: str(tmp_path / "llama-sweep-bench"))
    errors = []
    monkeypatch.setattr(bench_tab_mod.messagebox, "showerror", lambda title, msg: errors.append(msg))
    started = {"n": 0}
    monkeypatch.setattr(bench_tab.runner, "start", lambda plan: started.__setitem__("n", started["n"] + 1) or True)

    bench_tab.start()

    assert started["n"] == 0
    assert errors and "apply to" in errors[0]


def test_backend_radio_reveals_ik_levers_and_syncs_launcher(bench_tab):
    # Switching the Backend selection to ik_llama must reveal the ik-only lever
    # rows (grid_remove'd on llama.cpp) and push the choice to the launcher var.
    from modules.benchmark.detection import TOOL_LLAMA_BENCH

    bench_tab._set_tool(TOOL_LLAMA_BENCH)
    # Baseline: on llama.cpp the ik lever row is grid_removed (no grid info).
    assert bench_tab.backend_var.get() == "llama.cpp"
    assert bench_tab._lever_rows["rtr"]["chk"].grid_info() == {}

    bench_tab.backend_var.set("ik_llama")
    bench_tab._on_backend_radio_changed()

    assert bench_tab.launcher.backend_selection.get() == "ik_llama"
    # ik lever row is now gridded (position preserved).
    assert bench_tab._lever_rows["rtr"]["chk"].grid_info() != {}


def test_launcher_backend_change_mirrors_into_tab(bench_tab):
    # The reverse sync: writing the launcher backend var updates the tab.
    bench_tab.launcher.backend_selection.set("ik_llama")
    assert bench_tab.backend_var.get() == "ik_llama"


def test_ik_value_lever_renders_in_command(bench_tab):
    # -mqkv is a VALUE lever (0/1), not a bare flag: on ik_llama's llama-bench
    # it renders as a native comma-swept value.
    from modules.benchmark.detection import TOOL_LLAMA_BENCH

    bench_tab.backend_var.set("ik_llama")
    bench_tab._on_backend_radio_changed()
    bench_tab._set_tool(TOOL_LLAMA_BENCH)
    bench_tab.model_var.set("/m.gguf")
    bench_tab.lever_vars["mqkv"]["include"].set(True)
    bench_tab.lever_vars["mqkv"]["mode"].set("list")
    bench_tab.lever_vars["mqkv"]["values"].set("0,1")

    commands, _ = bench_tab._build_command_list()
    assert len(commands) == 1
    cmd = commands[0]
    idx = cmd.index("-mqkv")
    assert cmd[idx + 1] == "0,1"


def test_ik_sweep_lever_roundtrips_through_save_load(bench_tab):
    # ik value levers persist via the axes mechanism by their lever key.
    from modules.benchmark.detection import TOOL_LLAMA_BENCH

    bench_tab.backend_var.set("ik_llama")
    bench_tab._on_backend_radio_changed()
    bench_tab._set_tool(TOOL_LLAMA_BENCH)
    bench_tab.model_var.set("/m.gguf")
    bench_tab.lever_vars["rtr"]["include"].set(True)
    bench_tab.lever_vars["rtr"]["mode"].set("list")
    bench_tab.lever_vars["rtr"]["values"].set("0,1")

    bench_tab.config_name_var.set("ikcfg")
    bench_tab._save_config()

    # Wipe live state, then load it back.
    bench_tab.backend_var.set("llama.cpp")
    bench_tab.lever_vars["rtr"]["include"].set(False)
    bench_tab.lever_vars["rtr"]["values"].set("")
    bench_tab._load_config()

    assert bench_tab.backend_var.get() == "ik_llama"
    assert bool(bench_tab.lever_vars["rtr"]["include"].get()) is True
    assert bench_tab.lever_vars["rtr"]["values"].get() == "0,1"


def test_stop_polling_on_teardown_cancels_running_worker(bench_tab, monkeypatch):
    calls = {"cancel": 0}
    monkeypatch.setattr(type(bench_tab.runner), "is_running", property(lambda self: True))
    monkeypatch.setattr(bench_tab.runner, "cancel", lambda: calls.__setitem__("cancel", calls["cancel"] + 1))
    bench_tab._poll_after_id = "some-id"
    bench_tab._stop_polling_on_teardown()
    assert calls["cancel"] == 1
    assert bench_tab._poll_after_id is None
