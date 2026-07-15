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
    env = bench_tab._plan_env()
    # CUDA_DEVICE_ORDER is pinned alongside the visibility override (see the
    # dedicated Finding B test); the visibility value itself is the key thing.
    assert env["CUDA_VISIBLE_DEVICES"] == "2,0,1"


def test_gpu_env_overrides_unset_leaves_var_unset(bench_tab):
    # 'unset'/'skip' actions intentionally add no override (child sees all GPUs,
    # matching server-launch behaviour in manual / all-deselected modes).
    bench_tab.launcher.launch_manager = SimpleNamespace(_resolve_cuda_visible_devices_action=lambda: ("unset", None))
    assert bench_tab._plan_env() == {}


def test_gpu_env_overrides_defensive_without_launch_manager(bench_tab):
    # A launcher without a usable launch_manager must not raise and yields no
    # override. (The fixture's fake launcher has no launch_manager attribute.)
    assert bench_tab._plan_env() == {}


def test_plan_env_merges_enabled_env_vars_with_gpu_override(bench_tab):
    # Finding B: the benchmark plan env must carry BOTH the launcher's enabled
    # environment variables (same backend knobs the server launch applies) and
    # the GPU-visibility override, with GPU visibility winning on conflict.
    bench_tab.launcher.env_vars_manager = SimpleNamespace(
        get_enabled_env_vars=lambda: {"GGML_CUDA_FORCE_MMQ": "1", "CUDA_VISIBLE_DEVICES": "9"}
    )
    bench_tab.launcher.launch_manager = SimpleNamespace(
        _resolve_cuda_visible_devices_action=lambda: ("export", "2,0,1")
    )
    env = bench_tab._plan_env()
    assert env["GGML_CUDA_FORCE_MMQ"] == "1"
    # GPU visibility override wins on key conflict.
    assert env["CUDA_VISIBLE_DEVICES"] == "2,0,1"


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


def test_selecting_ik_build_flips_backend_and_syncs_launcher(bench_tab):
    # Finding A: manually picking a build whose backend differs from the radio
    # must flip backend_var (source of truth for ik-only levers / flash-attn)
    # AND push the choice to the launcher, before tool/lever state is refreshed.
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
    assert bench_tab.backend_var.get() == "llama.cpp"

    bench_tab.build_var.set(build.label)
    bench_tab._on_build_changed()

    assert bench_tab.backend_var.get() == "ik_llama"
    assert bench_tab.launcher.backend_selection.get() == "ik_llama"


def test_duplicate_custom_flag_raises_sweep_error(bench_tab):
    # Finding C: two included custom rows sharing a flag would collide (each
    # command reads the last value) — reject them from _collect_axes().
    from modules.benchmark.matrix import SweepError

    bench_tab._set_custom_axis_rows(
        [
            {"flag": "-ot", "enabled": True, "mode": "list", "raw": "exps=CPU", "min": 0, "max": 0, "step": 1},
            {"flag": "-ot", "enabled": True, "mode": "list", "raw": "attn=CPU", "min": 0, "max": 0, "step": 1},
        ]
    )
    with pytest.raises(SweepError):
        bench_tab._collect_axes()


def test_save_script_blocks_when_no_axis_applies_to_tool(bench_tab, monkeypatch):
    # Finding D: ticking a llama-bench-only lever then saving a sweep-bench
    # script must error (no file written) instead of writing a default command.
    from modules.benchmark import bench_tab as bench_tab_mod
    from modules.benchmark.detection import TOOL_SWEEP_BENCH

    bench_tab._set_tool(TOOL_SWEEP_BENCH)
    bench_tab.model_var.set("/m.gguf")
    # -p (Prompt tokens) applies ONLY to llama-bench.
    bench_tab.lever_vars["n_prompt"]["include"].set(True)
    bench_tab.lever_vars["n_prompt"]["mode"].set("list")
    bench_tab.lever_vars["n_prompt"]["values"].set("128")

    errors = []
    monkeypatch.setattr(bench_tab_mod.messagebox, "showerror", lambda title, msg: errors.append(msg))
    save_calls = {"n": 0}
    monkeypatch.setattr(
        bench_tab_mod.filedialog,
        "asksaveasfilename",
        lambda *a, **k: save_calls.__setitem__("n", save_calls["n"] + 1) or "/should-not-write.sh",
    )

    bench_tab._save_script("sh")

    assert save_calls["n"] == 0
    assert errors and "apply to" in errors[0]


def test_save_script_exports_gpu_env_line(bench_tab, monkeypatch, tmp_path):
    # Finding E: the saved script must export the same CUDA_VISIBLE_DEVICES the
    # in-app run uses. Depends on bench_script.render(..., env=...) landing (the
    # other agent adds it in parallel) — may transiently fail until then.
    from modules.benchmark import bench_tab as bench_tab_mod
    from modules.benchmark.detection import TOOL_LLAMA_BENCH

    bench_tab._set_tool(TOOL_LLAMA_BENCH)
    bench_tab.model_var.set("/m.gguf")
    bench_tab.lever_vars["threads"]["include"].set(True)
    bench_tab.lever_vars["threads"]["mode"].set("list")
    bench_tab.lever_vars["threads"]["values"].set("8")
    bench_tab.launcher.launch_manager = SimpleNamespace(
        _resolve_cuda_visible_devices_action=lambda: ("export", "2,0,1")
    )

    out = tmp_path / "bench.sh"
    monkeypatch.setattr(bench_tab_mod.filedialog, "asksaveasfilename", lambda *a, **k: str(out))
    # No build selected -> confirm the placeholder-exe prompt.
    monkeypatch.setattr(bench_tab_mod.messagebox, "askyesno", lambda *a, **k: True)
    errors = []
    monkeypatch.setattr(bench_tab_mod.messagebox, "showerror", lambda title, msg: errors.append(msg))

    bench_tab._save_script("sh")

    assert out.exists(), f"script not written; errors={errors}"
    content = out.read_text()
    assert "CUDA_VISIBLE_DEVICES" in content
    assert "2,0,1" in content


def test_mtp_rows_visible_only_on_sweep_bench(bench_tab):
    # The MTP levers apply only to ik_llama's llama-sweep-bench: their rows are
    # shown there and hidden (grid_removed) on llama-bench, where the
    # llama-bench-only ik levers (e.g. -fmoe) are the ones visible instead.
    from modules.benchmark.detection import TOOL_LLAMA_BENCH, TOOL_SWEEP_BENCH

    bench_tab.backend_var.set("ik_llama")
    bench_tab._on_backend_radio_changed()

    bench_tab._set_tool(TOOL_SWEEP_BENCH)
    bench_tab._on_tool_changed()
    for key in ("mtp", "draft_max", "draft_min", "draft_p_min", "mtprot"):
        assert bench_tab._lever_rows[key]["chk"].grid_info() != {}, key
    # -fmoe is llama-bench-only, so it is hidden on sweep-bench.
    assert bench_tab._lever_rows["fmoe"]["chk"].grid_info() == {}

    bench_tab._set_tool(TOOL_LLAMA_BENCH)
    bench_tab._on_tool_changed()
    for key in ("mtp", "draft_max", "draft_min", "draft_p_min", "mtprot"):
        assert bench_tab._lever_rows[key]["chk"].grid_info() == {}, key
    # -fmoe now applies (ik_llama + llama-bench) and is visible.
    assert bench_tab._lever_rows["fmoe"]["chk"].grid_info() != {}


def test_mtp_command_renders_bare_flag_and_draft_max(bench_tab):
    from modules.benchmark.detection import TOOL_SWEEP_BENCH

    bench_tab.backend_var.set("ik_llama")
    bench_tab._on_backend_radio_changed()
    bench_tab._set_tool(TOOL_SWEEP_BENCH)
    bench_tab.model_var.set("/m.gguf")
    bench_tab.lever_vars["mtp"]["include"].set(True)
    bench_tab.lever_vars["mtp"]["mode"].set("list")
    bench_tab.lever_vars["mtp"]["values"].set("1")
    bench_tab.lever_vars["draft_max"]["include"].set(True)
    bench_tab.lever_vars["draft_max"]["mode"].set("list")
    bench_tab.lever_vars["draft_max"]["values"].set("4")

    commands, _ = bench_tab._build_command_list()
    assert len(commands) == 1
    cmd = commands[0]
    # -mtp is a BARE flag: the token immediately after it must be the next flag,
    # not a value (guards against a regression that renders `-mtp 1`).
    assert "-mtp" in cmd
    assert cmd[cmd.index("-mtp") + 1] == "--draft-max"
    assert cmd[cmd.index("--draft-max") + 1] == "4"


def test_mtp_sweep_roundtrips_through_save_load(bench_tab):
    # MTP levers persist via the axes mechanism by their lever key (no new field).
    from modules.benchmark.detection import TOOL_SWEEP_BENCH

    bench_tab.backend_var.set("ik_llama")
    bench_tab._on_backend_radio_changed()
    bench_tab._set_tool(TOOL_SWEEP_BENCH)
    bench_tab.model_var.set("/m.gguf")
    bench_tab.lever_vars["mtp"]["include"].set(True)
    bench_tab.lever_vars["mtp"]["mode"].set("list")
    bench_tab.lever_vars["mtp"]["values"].set("0,1")
    bench_tab.lever_vars["draft_max"]["include"].set(True)
    bench_tab.lever_vars["draft_max"]["mode"].set("list")
    bench_tab.lever_vars["draft_max"]["values"].set("4")

    bench_tab.config_name_var.set("mtpcfg")
    bench_tab._save_config()

    # Wipe live state, then load it back.
    bench_tab.lever_vars["mtp"]["include"].set(False)
    bench_tab.lever_vars["mtp"]["values"].set("")
    bench_tab.lever_vars["draft_max"]["include"].set(False)
    bench_tab.lever_vars["draft_max"]["values"].set("")
    bench_tab._load_config()

    assert bool(bench_tab.lever_vars["mtp"]["include"].get()) is True
    assert bench_tab.lever_vars["mtp"]["values"].get() == "0,1"
    assert bool(bench_tab.lever_vars["draft_max"]["include"].get()) is True
    assert bench_tab.lever_vars["draft_max"]["values"].get() == "4"


def test_plan_env_filters_invalid_env_var_names(bench_tab):
    # Finding A: an enabled env var with an illegal name (injection-y
    # "BAD;touch x") must be dropped from the plan env — matching the server
    # launch path (LaunchManager._is_valid_env_var_name) — while valid names
    # pass through unchanged.
    bench_tab.launcher.env_vars_manager = SimpleNamespace(
        get_enabled_env_vars=lambda: {"GGML_CUDA_FORCE_MMQ": "1", "BAD;touch /tmp/x": "1"}
    )
    env = bench_tab._plan_env()
    assert env == {"GGML_CUDA_FORCE_MMQ": "1"}


def test_plan_env_pins_cuda_device_order_with_visibility(bench_tab):
    # Finding B: a concrete CUDA_VISIBLE_DEVICES override must also pin
    # CUDA_DEVICE_ORDER=PCI_BUS_ID (as the server launch + scripts do) so a saved
    # script run in a fresh shell maps the indices to the same physical cards.
    bench_tab.launcher.launch_manager = SimpleNamespace(
        _resolve_cuda_visible_devices_action=lambda: ("export", "2,0,1")
    )
    env = bench_tab._plan_env()
    assert env["CUDA_VISIBLE_DEVICES"] == "2,0,1"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"

    # 'unset'/'skip' leaves visibility unset, so device order must not be pinned.
    bench_tab.launcher.launch_manager = SimpleNamespace(_resolve_cuda_visible_devices_action=lambda: ("unset", None))
    assert "CUDA_DEVICE_ORDER" not in bench_tab._plan_env()


def test_plan_env_unset_honors_resolver_unset_action(bench_tab):
    # The resolver's explicit "unset" must remove an inherited CUDA_VISIBLE_DEVICES
    # (distinct from "skip"), carried through to the runner/script.
    bench_tab.launcher.launch_manager = SimpleNamespace(_resolve_cuda_visible_devices_action=lambda: ("unset", None))
    assert bench_tab._plan_env_unset() == ["CUDA_VISIBLE_DEVICES"]
    bench_tab.launcher.launch_manager = SimpleNamespace(_resolve_cuda_visible_devices_action=lambda: ("export", "0,1"))
    assert bench_tab._plan_env_unset() == []
    bench_tab.launcher.launch_manager = SimpleNamespace(_resolve_cuda_visible_devices_action=lambda: ("skip", None))
    assert bench_tab._plan_env_unset() == []


def test_backend_switch_with_no_matching_build_keeps_choice(bench_tab, monkeypatch):
    # Codex: clicking ik_llama when ONLY a llama.cpp build is detected must keep
    # the requested backend, not revert the launcher via a stale build's mirror.
    from modules.benchmark import bench_tab as bench_tab_mod
    from modules.benchmark.detection import TOOL_LLAMA_BENCH, BuildEntry

    llama_build = BuildEntry(
        label="llama.cpp (main)",
        backend="llama.cpp",
        root_dir="/lcpp",
        source="backend",
        tools={TOOL_LLAMA_BENCH: "/lcpp/llama-bench"},
    )
    monkeypatch.setattr(bench_tab_mod, "discover_builds", lambda launcher: [llama_build])
    bench_tab.rescan_builds()  # selects the only (llama.cpp) build
    assert bench_tab._current_backend() == "llama.cpp"

    # Now the user clicks the ik_llama radio; no ik build exists.
    bench_tab.backend_var.set("ik_llama")
    bench_tab._on_backend_radio_changed()

    assert bench_tab.backend_var.get() == "ik_llama"
    assert bench_tab._current_backend() == "ik_llama"
    assert bench_tab.launcher.backend_selection.get() == "ik_llama"


def test_stale_repetitions_do_not_block_sweep_bench(bench_tab):
    from modules.benchmark.detection import TOOL_SWEEP_BENCH

    # A malformed value left over from llama-bench must not raise for sweep-bench,
    # whose field is disabled and which renders no -r.
    bench_tab.repetitions_var.set("five")
    bench_tab._set_tool(TOOL_SWEEP_BENCH)
    assert bench_tab._repetitions() is None  # ignored, no SweepError


def test_collect_axes_skips_ticked_empty_lever_not_applicable_to_tool(bench_tab):
    # Codex: tick a llama-bench-only lever (-p) with NO values, then switch to
    # sweep-bench where its row is hidden. _collect_axes must skip it rather than
    # raise "... is ticked but has no values" for an uneditable hidden row.
    from modules.benchmark.detection import TOOL_LLAMA_BENCH, TOOL_SWEEP_BENCH

    bench_tab._set_tool(TOOL_LLAMA_BENCH)
    bench_tab.lever_vars["n_prompt"]["include"].set(True)
    bench_tab.lever_vars["n_prompt"]["values"].set("")  # ticked but empty
    bench_tab._set_tool(TOOL_SWEEP_BENCH)  # -p is n/a here and its row is hidden
    # Also tick an applicable lever so the result is a valid, non-empty sweep.
    bench_tab.lever_vars["ctx_size"]["include"].set(True)
    bench_tab.lever_vars["ctx_size"]["values"].set("4096")
    axes = bench_tab._collect_axes()  # must NOT raise
    assert [a.key for a in axes] == ["ctx_size"]


def test_save_config_reports_malformed_repetitions(bench_tab, monkeypatch):
    # Codex: Save must surface a SweepError (malformed -r) as a dialog, not let
    # it escape the Tk callback.
    from modules.benchmark import bench_tab as bench_tab_mod
    from modules.benchmark.detection import TOOL_LLAMA_BENCH

    bench_tab._set_tool(TOOL_LLAMA_BENCH)
    bench_tab.repetitions_var.set("five")
    bench_tab.config_name_var.set("cfg")
    errors = []
    monkeypatch.setattr(bench_tab_mod.messagebox, "showerror", lambda title, msg: errors.append((title, msg)))
    saved = []
    monkeypatch.setattr(bench_tab.store, "save", lambda cfg: saved.append(cfg) or True)

    bench_tab._save_config()  # must not raise

    assert errors and "Repetitions" in errors[0][1]
    assert saved == []  # never reached store.save


def test_start_blocks_on_malformed_repetitions(bench_tab, monkeypatch, tmp_path):
    # Finding C: a non-integer Repetitions value must surface as a validation
    # error and never start the runner (rather than silently omitting -r while
    # the UI still shows the typed value).
    from modules.benchmark import bench_tab as bench_tab_mod
    from modules.benchmark.detection import TOOL_LLAMA_BENCH

    bench_tab._set_tool(TOOL_LLAMA_BENCH)
    model = tmp_path / "m.gguf"
    model.write_text("x")
    bench_tab.model_var.set(str(model))
    bench_tab.lever_vars["threads"]["include"].set(True)
    bench_tab.lever_vars["threads"]["mode"].set("list")
    bench_tab.lever_vars["threads"]["values"].set("8")
    bench_tab.repetitions_var.set("five")

    monkeypatch.setattr(bench_tab, "_current_exe", lambda: str(tmp_path / "llama-bench"))
    errors = []
    monkeypatch.setattr(bench_tab_mod.messagebox, "showerror", lambda title, msg: errors.append(msg))
    started = {"n": 0}
    monkeypatch.setattr(bench_tab.runner, "start", lambda plan: started.__setitem__("n", started["n"] + 1) or True)

    bench_tab.start()

    assert started["n"] == 0
    assert errors and "Repetitions" in errors[0]


def test_empty_repetitions_still_runs(bench_tab, monkeypatch, tmp_path):
    # Finding C: an EMPTY Repetitions field means "tool default" and must not be
    # treated as an error — the run proceeds.
    from modules.benchmark.detection import TOOL_LLAMA_BENCH

    bench_tab._set_tool(TOOL_LLAMA_BENCH)
    model = tmp_path / "m.gguf"
    model.write_text("x")
    bench_tab.model_var.set(str(model))
    bench_tab.lever_vars["threads"]["include"].set(True)
    bench_tab.lever_vars["threads"]["mode"].set("list")
    bench_tab.lever_vars["threads"]["values"].set("8")
    bench_tab.repetitions_var.set("")

    monkeypatch.setattr(bench_tab, "_current_exe", lambda: str(tmp_path / "llama-bench"))
    monkeypatch.setattr(bench_tab, "_poll_runner", lambda: None)
    started = {"n": 0}
    monkeypatch.setattr(bench_tab.runner, "start", lambda plan: started.__setitem__("n", started["n"] + 1) or True)

    bench_tab.start()

    assert started["n"] == 1


def test_rescan_prefers_active_backend_build(bench_tab, monkeypatch):
    # Finding D: with builds for both backends discovered and the launcher on
    # ik_llama, opening/rescanning must select the ik_llama build and must NOT
    # flip the launcher back to llama.cpp (discovery probes llama.cpp first, and
    # _on_build_changed mirrors the chosen build's backend into the launcher).
    from modules.benchmark import bench_tab as bench_tab_mod
    from modules.benchmark.detection import TOOL_LLAMA_BENCH, TOOL_SWEEP_BENCH, BuildEntry

    llama_build = BuildEntry(
        label="llama.cpp (main)",
        backend="llama.cpp",
        root_dir="/lcpp",
        source="backend",
        tools={TOOL_LLAMA_BENCH: "/lcpp/llama-bench"},
    )
    ik_build = BuildEntry(
        label="ik (backend dir)",
        backend="ik_llama",
        root_dir="/ik",
        source="backend",
        tools={TOOL_LLAMA_BENCH: "/ik/llama-bench", TOOL_SWEEP_BENCH: "/ik/llama-sweep-bench"},
    )
    # Discovery returns the llama.cpp build first (probes llama_cpp_dir first).
    monkeypatch.setattr(bench_tab_mod, "discover_builds", lambda launcher: [llama_build, ik_build])

    bench_tab.launcher.backend_selection.set("ik_llama")
    assert bench_tab.backend_var.get() == "ik_llama"
    # Stale/empty selection so the default-selection path runs.
    bench_tab.build_var.set("")
    bench_tab.rescan_builds()

    assert bench_tab.build_var.get() == ik_build.label
    assert bench_tab._current_backend() == "ik_llama"
    assert bench_tab.launcher.backend_selection.get() == "ik_llama"


def test_load_config_stale_root_prefers_saved_backend_build(bench_tab):
    # Finding E: loading an ik_llama config whose build_root is stale must select
    # an ik_llama build that offers the saved tool (not the llama.cpp build that
    # also offers llama-bench), so _current_backend stays ik_llama and the ik
    # axes the config restores are applied.
    from modules.benchmark.bench_persistence import BenchConfig
    from modules.benchmark.detection import TOOL_LLAMA_BENCH, TOOL_SWEEP_BENCH, BuildEntry

    llama_build = BuildEntry(
        label="llama.cpp (main)",
        backend="llama.cpp",
        root_dir="/lcpp",
        source="backend",
        tools={TOOL_LLAMA_BENCH: "/lcpp/llama-bench"},
    )
    ik_build = BuildEntry(
        label="ik (backend dir)",
        backend="ik_llama",
        root_dir="/ik",
        source="backend",
        tools={TOOL_LLAMA_BENCH: "/ik/llama-bench", TOOL_SWEEP_BENCH: "/ik/llama-sweep-bench"},
    )
    bench_tab._builds = [llama_build, ik_build]
    bench_tab._build_labels = [llama_build.label, ik_build.label]
    bench_tab._build_combo.configure(values=bench_tab._build_labels)

    bench_tab.store.save(
        BenchConfig(
            name="ik-stale",
            tool=TOOL_LLAMA_BENCH,
            backend="ik_llama",
            build_root="/gone",  # stale: matches no discovered build
            model_path="/m.gguf",
            axes={"rtr": {"enabled": True, "mode": "list", "raw": "0,1", "min": 0, "max": 0, "step": 1}},
        )
    )
    bench_tab.config_name_var.set("ik-stale")
    bench_tab._load_config()

    assert bench_tab._selected_build() is ik_build
    assert bench_tab._current_backend() == "ik_llama"
    assert bool(bench_tab.lever_vars["rtr"]["include"].get()) is True


def test_repetitions_field_disabled_for_sweep_bench(bench_tab):
    # Finding F: llama-sweep-bench has no -r flag; the Repetitions entry must be
    # greyed for it and re-enabled for llama-bench.
    from modules.benchmark.detection import TOOL_LLAMA_BENCH, TOOL_SWEEP_BENCH

    bench_tab._set_tool(TOOL_SWEEP_BENCH)
    bench_tab._on_tool_changed()
    assert str(bench_tab._repetitions_entry["state"]) == "disabled"

    bench_tab._set_tool(TOOL_LLAMA_BENCH)
    bench_tab._on_tool_changed()
    assert str(bench_tab._repetitions_entry["state"]) == "normal"
