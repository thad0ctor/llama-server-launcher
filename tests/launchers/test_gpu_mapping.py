"""Tests for GPU selection → launch command mapping consistency.

These tests cover the full pipeline from checkbox selection through drag-reorder,
``get_ordered_selected_gpus()``, and on to ``CUDA_VISIBLE_DEVICES`` /
``--main-gpu`` / ``--tensor-split`` emission.

Focus areas:
  * Physical GPU IDs selected via checkboxes get written into
    ``CUDA_VISIBLE_DEVICES`` in the user-specified order.
  * ``get_ordered_selected_gpus()`` must not emit duplicates even when a buggy
    drag-reorder left ``gpu_order`` with repeats.
  * ``get_ordered_selected_gpus()`` appends newly-selected GPUs missing from
    ``gpu_order`` in ascending physical-ID order.
  * ``--main-gpu`` / ``--tensor-split`` propagate verbatim through ``build_cmd``.
  * Live-launch bash path and save-as-script bash path generate the same
    CUDA_VISIBLE_DEVICES handling (both set-when-selected, both
    unset-when-none).
"""

from __future__ import annotations

import importlib.util
import sys
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ENTRY_PATH = REPO_ROOT / "llamacpp-server-launcher.py"


# ---------------------------------------------------------------------------
# Load the hyphenated entry module so we can exercise
# LlamaCppLauncher.get_ordered_selected_gpus directly.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def entry_module():
    spec = importlib.util.spec_from_file_location("entry_module_gpu_map", ENTRY_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["entry_module_gpu_map"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def gos_callable(entry_module):
    """Return a ``get_ordered_selected_gpus`` bound to a fresh SimpleNamespace.

    The real method only touches ``self.app_settings``, so a stand-in with
    that one attribute is sufficient.
    """
    method = entry_module.LlamaCppLauncher.get_ordered_selected_gpus

    def _call(app_settings):
        stub = SimpleNamespace(app_settings=app_settings)
        return method(stub)

    return _call


# ---------------------------------------------------------------------------
# get_ordered_selected_gpus — unit behaviour
# ---------------------------------------------------------------------------


class TestGetOrderedSelectedGpus:
    """Every launch path goes through this method — it is the single source of
    truth for the GPU index list that ends up in ``CUDA_VISIBLE_DEVICES``.
    Any mapping bug here propagates everywhere, so these tests are the most
    important."""

    def test_empty_returns_empty(self, gos_callable):
        assert gos_callable({"selected_gpus": [], "gpu_order": []}) == []

    def test_natural_order_preserved(self, gos_callable):
        # All GPUs selected, no reorder — should match natural indices.
        out = gos_callable({"selected_gpus": [0, 1, 2, 3], "gpu_order": [0, 1, 2, 3]})
        assert out == [0, 1, 2, 3]

    def test_user_reorder_honored(self, gos_callable):
        # User dragged GPU 2 to the top.
        out = gos_callable({"selected_gpus": [0, 1, 2], "gpu_order": [2, 0, 1]})
        assert out == [2, 0, 1], (
            "The user's drag-order is the source of truth for CUDA_VISIBLE_DEVICES; "
            "dropping or resorting it would remap to the wrong physical GPUs."
        )

    def test_subset_only_emits_selected(self, gos_callable):
        # 4 GPUs detected, only 1 and 3 selected.
        out = gos_callable({"selected_gpus": [1, 3], "gpu_order": [0, 1, 2, 3]})
        assert out == [1, 3], (
            "Deselected GPUs must not leak into CUDA_VISIBLE_DEVICES — "
            "otherwise llama.cpp would use GPUs the user turned off."
        )

    def test_subset_respects_user_reorder(self, gos_callable):
        # User selected GPUs 1 and 3, reordered so 3 comes first.
        out = gos_callable({"selected_gpus": [1, 3], "gpu_order": [3, 1]})
        assert out == [3, 1]

    def test_gpu_in_selected_but_missing_from_order_gets_appended(self, gos_callable):
        # Defensive path: if gpu_order lost an entry that's still selected,
        # it must be appended (not dropped) so the user doesn't silently lose
        # a GPU from the launch command.
        out = gos_callable({"selected_gpus": [0, 1, 2], "gpu_order": [2]})
        # Ordered entries from gpu_order first, then missing selections in
        # ascending physical ID order.
        assert out == [2, 0, 1]

    def test_order_entries_not_in_selected_are_skipped(self, gos_callable):
        # gpu_order has stale entries for GPUs that were deselected.
        out = gos_callable({"selected_gpus": [1], "gpu_order": [0, 1, 2]})
        assert out == [1], (
            "Stale gpu_order entries must not resurrect deselected GPUs."
        )

    def test_duplicate_in_gpu_order_does_not_duplicate_device(self, gos_callable):
        """Regression: a corrupted drag-reorder could leave a duplicate in
        ``gpu_order`` (e.g. ``[1, 1, 2]``).

        ``CUDA_VISIBLE_DEVICES=1,1,2`` is rejected by the CUDA runtime with
        ``invalid device ordinal``, and ``--tensor-split`` with three entries
        for two selected GPUs silently offloads to the wrong device. The
        method must emit each selected physical GPU exactly once.
        """
        out = gos_callable({"selected_gpus": [1, 2], "gpu_order": [1, 1, 2]})
        assert out == [1, 2]
        assert len(out) == len(set(out)), (
            f"Duplicates in output {out} would produce an invalid "
            f"CUDA_VISIBLE_DEVICES string and misaligned --tensor-split."
        )

    def test_duplicate_preserves_first_occurrence_position(self, gos_callable):
        # When the user intentionally moved GPU 2 to the top and the
        # duplicate is at the original position, we keep the earlier one.
        out = gos_callable({"selected_gpus": [1, 2], "gpu_order": [2, 1, 2]})
        # First occurrence wins — GPU 2 stays first.
        assert out == [2, 1]


# ---------------------------------------------------------------------------
# LaunchManager end-to-end propagation
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tk_root():
    """Return a headless Tcl interpreter for tk.Var instances.

    Uses ``tk.Tcl()`` rather than ``tk.Tk()`` so these tests run on CI
    runners without a display (otherwise the tk.Tk() call raises TclError
    and every downstream test silently skips — a regression we only saw
    when CodeRabbit pointed it out). The launch.py code paths under test
    (build_cmd, save_sh_script, save_ps1_script, launch_server) only use
    the tk.*Var ``get()`` / ``set()`` APIs which work fine against a Tcl
    interpreter — no widgets are ever created.
    """
    try:
        root = tk.Tcl()
    except tk.TclError as exc:
        pytest.skip(f"tkinter unavailable: {exc}")
    yield root


def _mk(master, cls, value):
    v = cls(master=master)
    v.set(value)
    return v


@pytest.fixture()
def built_tree(tmp_path):
    base = tmp_path / "llama_cpp_root"
    bin_dir = base / "build" / "bin"
    bin_dir.mkdir(parents=True)
    exe_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    exe = bin_dir / exe_name
    exe.write_text("#!/bin/sh\necho fake\n")
    try:
        import os
        os.chmod(exe, 0o755)
    except OSError:
        pass
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF" + b"\x00" * 16)
    return {"base": base, "exe": exe, "model": model}


@pytest.fixture()
def launcher_mock(tk_root, built_tree):
    """Minimal launcher mock sufficient for build_cmd / save_sh_script."""
    m = MagicMock()
    m.backend_selection = _mk(tk_root, tk.StringVar, "llama.cpp")
    m.llama_cpp_dir = _mk(tk_root, tk.StringVar, str(built_tree["base"]))
    m.ik_llama_dir = _mk(tk_root, tk.StringVar, str(built_tree["base"]))
    m.venv_dir = _mk(tk_root, tk.StringVar, "")
    m.model_path = _mk(tk_root, tk.StringVar, str(built_tree["model"]))
    m.mmproj_enabled = _mk(tk_root, tk.BooleanVar, False)
    m.selected_mmproj_path = _mk(tk_root, tk.StringVar, "")
    m.cache_type_k = _mk(tk_root, tk.StringVar, "f16")
    m.cache_type_v = _mk(tk_root, tk.StringVar, "f16")
    m.threads = _mk(tk_root, tk.StringVar, "")
    m.logical_cores = 8
    m.threads_batch = _mk(tk_root, tk.StringVar, "")
    m.batch_size = _mk(tk_root, tk.StringVar, "")
    m.ubatch_size = _mk(tk_root, tk.StringVar, "")
    m.ctx_size = _mk(tk_root, tk.IntVar, 2048)
    m.seed = _mk(tk_root, tk.StringVar, "-1")
    m.temperature = _mk(tk_root, tk.StringVar, "0.8")
    m.min_p = _mk(tk_root, tk.StringVar, "0.05")
    m.tensor_split = _mk(tk_root, tk.StringVar, "")
    m.n_gpu_layers = _mk(tk_root, tk.StringVar, "0")
    m.main_gpu = _mk(tk_root, tk.StringVar, "0")
    m.flash_attn = _mk(tk_root, tk.BooleanVar, False)
    m.fit_enabled = _mk(tk_root, tk.BooleanVar, False)
    m.fit_ctx = _mk(tk_root, tk.StringVar, "")
    m.fit_target = _mk(tk_root, tk.StringVar, "")
    m.no_mmap = _mk(tk_root, tk.BooleanVar, False)
    m.mlock = _mk(tk_root, tk.BooleanVar, False)
    m.no_kv_offload = _mk(tk_root, tk.BooleanVar, False)
    m.prio = _mk(tk_root, tk.StringVar, "0")
    m.parallel = _mk(tk_root, tk.StringVar, "1")
    m.cpu_moe = _mk(tk_root, tk.BooleanVar, False)
    m.n_cpu_moe = _mk(tk_root, tk.StringVar, "")
    m.ignore_eos = _mk(tk_root, tk.BooleanVar, False)
    m.n_predict = _mk(tk_root, tk.StringVar, "-1")
    m.host = _mk(tk_root, tk.StringVar, "127.0.0.1")
    m.port = _mk(tk_root, tk.StringVar, "8080")
    m.template_source = _mk(tk_root, tk.StringVar, "default")
    m.current_template_display = _mk(tk_root, tk.StringVar, "")
    m.jinja_enabled = _mk(tk_root, tk.BooleanVar, False)
    m.custom_parameters_list = []
    m.get_ordered_selected_gpus = MagicMock(return_value=[])
    m.gpu_info = {"device_count": 0}
    m.model_listbox.curselection = MagicMock(return_value=())
    m.found_models = {}
    m.ik_llama_tab.get_ik_llama_flags = MagicMock(return_value=[])
    m.env_vars_manager.get_enabled_env_vars = MagicMock(return_value={})
    return m


@pytest.fixture()
def manager(launcher_mock):
    from modules.launch import LaunchManager
    return LaunchManager(launcher_mock)


class TestMainGpuPropagation:
    """``--main-gpu`` is passed verbatim. The UI label documents that the
    index is relative to the SELECTED GPUs (post-CUDA_VISIBLE_DEVICES remap),
    so the launcher itself must NOT reinterpret or remap the user's value —
    that would double-translate the index."""

    def test_main_gpu_default_omitted(self, manager, launcher_mock):
        launcher_mock.main_gpu.set("0")
        cmd = manager.build_cmd()
        assert "--main-gpu" not in cmd

    def test_main_gpu_nonzero_emitted_verbatim(self, manager, launcher_mock):
        launcher_mock.main_gpu.set("2")
        cmd = manager.build_cmd()
        assert "--main-gpu" in cmd
        # Verbatim — NOT translated through gpu_order.
        assert cmd[cmd.index("--main-gpu") + 1] == "2"

    def test_main_gpu_not_remapped_by_gpu_order(self, manager, launcher_mock):
        """Even when the user reordered GPUs so physical GPU 2 is first,
        ``--main-gpu 1`` must stay "1" — llama.cpp interprets this index
        against the post-CUDA_VISIBLE_DEVICES view, which the user is
        expected to reason about themselves (the UI label says so)."""
        launcher_mock.get_ordered_selected_gpus.return_value = [2, 0, 1]
        launcher_mock.main_gpu.set("1")
        cmd = manager.build_cmd()
        assert cmd[cmd.index("--main-gpu") + 1] == "1"


class TestTensorSplitPropagation:
    def test_tensor_split_passes_through_unmodified(self, manager, launcher_mock):
        # Asymmetric split to catch any accidental resort/normalize.
        launcher_mock.tensor_split.set("24,8,16")
        cmd = manager.build_cmd()
        assert cmd[cmd.index("--tensor-split") + 1] == "24,8,16"

    def test_tensor_split_not_reordered_with_gpu_order(self, manager, launcher_mock):
        # User reordered GPUs but tensor_split is a literal string.
        launcher_mock.get_ordered_selected_gpus.return_value = [2, 0]
        launcher_mock.tensor_split.set("3,1")
        cmd = manager.build_cmd()
        assert cmd[cmd.index("--tensor-split") + 1] == "3,1"


class TestCudaVisibleDevicesInShScript:
    """The bash save-as-script path is the ground truth the test suite
    already exercised. These tests extend it to the reorder edge cases that
    actual users hit."""

    def _write_and_read(self, manager, launcher_mock, out_path):
        with patch("modules.launch.filedialog") as fd, patch(
            "modules.launch.messagebox"
        ):
            fd.asksaveasfilename.return_value = str(out_path)
            manager.save_sh_script()
        return out_path.read_text()

    def test_cuda_visible_devices_reflects_user_reorder(
        self, manager, launcher_mock, tmp_path
    ):
        # User dragged physical GPU 2 to the top of the order list.
        launcher_mock.get_ordered_selected_gpus.return_value = [2, 0, 1]
        launcher_mock.gpu_info = {"device_count": 3}
        text = self._write_and_read(manager, launcher_mock, tmp_path / "launch.sh")
        assert 'export CUDA_VISIBLE_DEVICES="2,0,1"' in text, (
            "User reorder must be preserved verbatim in CUDA_VISIBLE_DEVICES — "
            "llama.cpp then sees logical GPU 0 = physical 2, logical 1 = physical 0, "
            "logical 2 = physical 1."
        )

    def test_cuda_visible_devices_subset(self, manager, launcher_mock, tmp_path):
        launcher_mock.get_ordered_selected_gpus.return_value = [1, 3]
        launcher_mock.gpu_info = {"device_count": 4}
        text = self._write_and_read(manager, launcher_mock, tmp_path / "launch.sh")
        assert 'export CUDA_VISIBLE_DEVICES="1,3"' in text

    def test_single_gpu_selected(self, manager, launcher_mock, tmp_path):
        launcher_mock.get_ordered_selected_gpus.return_value = [1]
        launcher_mock.gpu_info = {"device_count": 4}
        text = self._write_and_read(manager, launcher_mock, tmp_path / "launch.sh")
        assert 'export CUDA_VISIBLE_DEVICES="1"' in text


def _capture_live_launch_script(manager):
    """Run manager.launch_server() on Linux with Popen/messagebox patched and
    return the bash script text that would have been executed.

    Used to cross-check that the live-launch and save-as-script paths agree
    on how they handle CUDA_VISIBLE_DEVICES.
    """
    captured = {}

    def _fake_popen(*args, **kwargs):
        captured["cmd"] = args[0] if args else kwargs.get("args")
        rv = MagicMock()
        rv.wait = MagicMock(return_value=0)
        return rv

    with patch("modules.launch.shutil.which", return_value="/bin/bash"), \
         patch("modules.launch.subprocess.Popen", side_effect=_fake_popen), \
         patch("modules.launch.messagebox"):
        manager.launch_server()

    cmd = captured.get("cmd")
    assert cmd is not None, "launch_server must invoke subprocess.Popen"
    return " ".join(str(c) for c in cmd)


class TestCudaVisibleDevicesLiveVsScript:
    """The live-launch bash path and the save-as-script bash path must agree on
    how they handle CUDA_VISIBLE_DEVICES.

    Historical mismatch: the live path emitted ``export X=1,0`` unquoted
    while the save path used ``export X="1,0"`` — the difference is cosmetic
    for numeric values but becomes a real mismatch in edge cases (e.g. an
    empty selected list wasn't explicitly unset in one path). These tests
    pin down the contract."""

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="launch_server's bash path only runs on Linux/macOS",
    )
    def test_both_paths_set_devices_when_selected(
        self, manager, launcher_mock, tmp_path
    ):
        """When GPUs are selected, both the live-launch script and the saved
        .sh must export CUDA_VISIBLE_DEVICES with the same device list."""
        launcher_mock.get_ordered_selected_gpus.return_value = [1, 0]
        launcher_mock.gpu_info = {"device_count": 2}

        # Saved script:
        with patch("modules.launch.filedialog") as fd, patch(
            "modules.launch.messagebox"
        ):
            fd.asksaveasfilename.return_value = str(tmp_path / "launch.sh")
            manager.save_sh_script()
        saved_text = (tmp_path / "launch.sh").read_text()
        assert 'export CUDA_VISIBLE_DEVICES="1,0"' in saved_text

        # Live-launch script: accept either quoted or unquoted form. Both
        # are semantically equivalent in bash, and the style is an
        # implementation detail that tests shouldn't pin down.
        live_text = _capture_live_launch_script(manager)
        assert (
            "CUDA_VISIBLE_DEVICES=1,0" in live_text
            or 'CUDA_VISIBLE_DEVICES="1,0"' in live_text
        )

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="launch_server's bash path only runs on Linux/macOS",
    )
    def test_both_paths_clear_devices_when_gpus_detected_none_selected(
        self, manager, launcher_mock, tmp_path
    ):
        """When GPUs are detected but the user deselected all of them, both
        paths must explicitly unset CUDA_VISIBLE_DEVICES.

        Without this, an inherited shell CUDA_VISIBLE_DEVICES would silently
        steer llama.cpp to GPUs the user thought they had disabled — the exact
        kind of mapping inconsistency this test suite is guarding against."""
        launcher_mock.get_ordered_selected_gpus.return_value = []
        launcher_mock.gpu_info = {"device_count": 2}

        # Saved script:
        with patch("modules.launch.filedialog") as fd, patch(
            "modules.launch.messagebox"
        ):
            fd.asksaveasfilename.return_value = str(tmp_path / "launch.sh")
            manager.save_sh_script()
        saved_text = (tmp_path / "launch.sh").read_text()
        assert "unset CUDA_VISIBLE_DEVICES" in saved_text

        # Live-launch script:
        live_text = _capture_live_launch_script(manager)
        assert "unset CUDA_VISIBLE_DEVICES" in live_text


class TestCudaVisibleDevicesLiveLaunchBash:
    """Exercise ``launch_server`` directly on Linux to check that the live-launch
    bash command clears CUDA_VISIBLE_DEVICES when GPUs are detected but none
    selected — parity with save_sh_script."""

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="launch_server's bash path only runs on Linux/macOS",
    )
    def test_live_launch_no_cuda_lines_when_no_gpus_detected(
        self, manager, launcher_mock, tmp_path
    ):
        """Parity with save_sh_script's ``test_no_cuda_lines_when_no_gpus_detected``:
        when there are zero GPUs detected, the live-launch bash must not
        emit an unset either (no GPUs means no filtering semantics apply)."""
        launcher_mock.get_ordered_selected_gpus.return_value = []
        launcher_mock.gpu_info = {"device_count": 0}

        script_text = _capture_live_launch_script(manager)
        # CUDA_DEVICE_ORDER is always set; CUDA_VISIBLE_DEVICES must not be
        # touched when no GPUs are detected.
        assert "CUDA_VISIBLE_DEVICES" not in script_text

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="launch_server's bash path only runs on Linux/macOS",
    )
    def test_live_launch_exports_cuda_visible_devices_with_user_order(
        self, manager, launcher_mock, tmp_path
    ):
        launcher_mock.get_ordered_selected_gpus.return_value = [2, 0]
        launcher_mock.gpu_info = {"device_count": 3}

        script_text = _capture_live_launch_script(manager)
        # The user reorder "[2, 0]" must flow through verbatim.
        assert "CUDA_VISIBLE_DEVICES=2,0" in script_text or \
               'CUDA_VISIBLE_DEVICES="2,0"' in script_text


# ---------------------------------------------------------------------------
# Manual GPU mode: synthetic indices must NOT become CUDA_VISIBLE_DEVICES
# ---------------------------------------------------------------------------


class TestManualGpuModeDoesNotExportSyntheticIndices:
    """In manual GPU mode, the UI synthesises fake GPU entries (id 0..N-1)
    for capacity-planning / config preview. Those indices have no relation
    to physical PCIe bus IDs on the host. If we passed them through to
    ``CUDA_VISIBLE_DEVICES`` the CUDA runtime would filter the wrong real
    devices — e.g. the user planning for "2 x RTX 5090" on a 4-real-GPU
    host would launch llama-server against real GPUs 0 and 1, which is
    NOT what they selected.

    The correct behaviour: emit ``unset CUDA_VISIBLE_DEVICES`` in all
    script paths so any inherited shell value is cleared and the CUDA
    runtime sees its default device list. This also prevents a shell-set
    CUDA_VISIBLE_DEVICES from silently re-introducing the inconsistency
    this test class guards against."""

    def _save_sh(self, manager, launcher_mock, out_path):
        with patch("modules.launch.filedialog") as fd, patch(
            "modules.launch.messagebox"
        ):
            fd.asksaveasfilename.return_value = str(out_path)
            manager.save_sh_script()
        return out_path.read_text()

    def _save_ps1(self, manager, launcher_mock, out_path):
        with patch("modules.launch.filedialog") as fd, patch(
            "modules.launch.messagebox"
        ):
            fd.asksaveasfilename.return_value = str(out_path)
            manager.save_ps1_script()
        return out_path.read_text()

    def test_save_sh_script_unsets_when_manual_mode(
        self, manager, launcher_mock, tmp_path
    ):
        # Simulate manual mode: 2 synthetic GPUs, both selected.
        launcher_mock.get_ordered_selected_gpus.return_value = [0, 1]
        launcher_mock.gpu_info = {"device_count": 2, "manual_mode": True}

        text = self._save_sh(manager, launcher_mock, tmp_path / "launch.sh")
        # The synthetic "0,1" must NOT be exported.
        assert 'export CUDA_VISIBLE_DEVICES="0,1"' not in text, (
            "Manual-mode indices must not be exported as CUDA_VISIBLE_DEVICES — "
            "they are synthetic and would mis-filter real hardware."
        )
        # But the env var IS explicitly unset so inherited shell values are
        # cleared.
        assert "unset CUDA_VISIBLE_DEVICES" in text

    def test_save_ps1_script_unsets_when_manual_mode(
        self, manager, launcher_mock, tmp_path
    ):
        launcher_mock.get_ordered_selected_gpus.return_value = [0, 1]
        launcher_mock.gpu_info = {"device_count": 2, "manual_mode": True}

        text = self._save_ps1(manager, launcher_mock, tmp_path / "launch.ps1")
        assert '$env:CUDA_VISIBLE_DEVICES="0,1"' not in text, (
            "Manual-mode indices must not be exported as CUDA_VISIBLE_DEVICES "
            "in the PowerShell save-as-script either."
        )
        assert "Remove-Item Env:CUDA_VISIBLE_DEVICES" in text

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="launch_server's bash path only runs on Linux/macOS",
    )
    def test_live_launch_bash_unsets_when_manual_mode(
        self, manager, launcher_mock
    ):
        launcher_mock.get_ordered_selected_gpus.return_value = [0, 1]
        launcher_mock.gpu_info = {"device_count": 2, "manual_mode": True}

        script_text = _capture_live_launch_script(manager)
        assert "CUDA_VISIBLE_DEVICES=0,1" not in script_text
        assert 'CUDA_VISIBLE_DEVICES="0,1"' not in script_text
        assert "unset CUDA_VISIBLE_DEVICES" in script_text

    def test_manual_mode_with_no_selection_still_unsets(
        self, manager, launcher_mock, tmp_path
    ):
        # Manual mode, but no manual GPU selected. Still unset (no real
        # hardware should be filtered on the user's behalf).
        launcher_mock.get_ordered_selected_gpus.return_value = []
        launcher_mock.gpu_info = {"device_count": 2, "manual_mode": True}

        text = self._save_sh(manager, launcher_mock, tmp_path / "launch.sh")
        assert "unset CUDA_VISIBLE_DEVICES" in text
        # And the synthetic-index export is absent either way.
        assert "export CUDA_VISIBLE_DEVICES=" not in text


# ---------------------------------------------------------------------------
# _resolve_cuda_visible_devices_action — unit behaviour
# ---------------------------------------------------------------------------


class TestResolveCudaVisibleDevicesAction:
    """Central dispatch — every emit site goes through this. Any bug here
    fans out to all four script paths, so pin down every branch."""

    def test_export_when_gpus_selected(self, manager, launcher_mock):
        launcher_mock.get_ordered_selected_gpus.return_value = [1, 0]
        launcher_mock.gpu_info = {"device_count": 2}
        assert manager._resolve_cuda_visible_devices_action() == ("export", "1,0")

    def test_unset_when_gpus_detected_none_selected(self, manager, launcher_mock):
        launcher_mock.get_ordered_selected_gpus.return_value = []
        launcher_mock.gpu_info = {"device_count": 2}
        assert manager._resolve_cuda_visible_devices_action() == ("unset", None)

    def test_skip_when_no_gpus_detected(self, manager, launcher_mock):
        launcher_mock.get_ordered_selected_gpus.return_value = []
        launcher_mock.gpu_info = {"device_count": 0}
        assert manager._resolve_cuda_visible_devices_action() == ("skip", None)

    def test_unset_when_manual_mode_regardless_of_selection(
        self, manager, launcher_mock
    ):
        # Manual mode beats everything — even a full synthetic selection
        # maps to "unset" because those indices don't refer to physical
        # hardware.
        launcher_mock.get_ordered_selected_gpus.return_value = [0, 1, 2]
        launcher_mock.gpu_info = {
            "device_count": 3, "manual_mode": True,
        }
        assert manager._resolve_cuda_visible_devices_action() == ("unset", None)

    def test_unset_when_manual_mode_and_zero_device_count(
        self, manager, launcher_mock
    ):
        # Edge case: manual_mode flagged True with device_count 0 (no
        # manual GPUs in list yet). Still unset — the flag is the
        # controlling signal, not device_count.
        launcher_mock.get_ordered_selected_gpus.return_value = []
        launcher_mock.gpu_info = {"device_count": 0, "manual_mode": True}
        assert manager._resolve_cuda_visible_devices_action() == ("unset", None)


# ---------------------------------------------------------------------------
# Inherited CUDA_VISIBLE_DEVICES is cleared before PyTorch detection
# ---------------------------------------------------------------------------


class TestInheritedCudaVisibleDevicesCleared:
    """If the user launches the GUI from a shell that already had
    ``CUDA_VISIBLE_DEVICES`` set, PyTorch would otherwise enumerate the
    filtered subset and assign them logical indices 0..M-1 that do NOT
    match physical PCIe bus IDs. The UI would label them "GPU 0", "GPU 1"
    etc., and when the user selected one and launched, the generated
    script's ``export CUDA_VISIBLE_DEVICES=0`` would filter a DIFFERENT
    physical device than the one shown in the UI.

    modules/system.py must pop ``CUDA_VISIBLE_DEVICES`` at module-load
    time (before ``import torch``) so detection always enumerates all
    physical devices under PCI_BUS_ID ordering. User can still restrict
    selection via the checkbox list — those choices flow through to the
    launch command."""

    def test_system_module_clears_inherited_value(self, tmp_path):
        """Import modules.system in a subprocess with
        ``CUDA_VISIBLE_DEVICES`` pre-set, and assert the module clears it.

        The subprocess isolates the import side-effects from this test
        runner's own os.environ.
        """
        import json as _json
        import subprocess as _sp

        probe_script = (
            "import os\n"
            "import sys\n"
            f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
            # Capture the pre-import state.
            "pre = os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')\n"
            "import modules.system  # noqa: F401\n"
            "post = os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')\n"
            "import json\n"
            "print(json.dumps({'pre': pre, 'post': post}))\n"
        )

        env = {
            **__import__("os").environ,
            "CUDA_VISIBLE_DEVICES": "1,3",
        }
        result = _sp.run(
            [sys.executable, "-c", probe_script],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
        # Grab the LAST line of stdout — the module emits debug banners
        # on stderr and our probe emits a single JSON line on stdout.
        assert result.returncode == 0, (
            f"probe subprocess failed: stderr={result.stderr}"
        )
        stdout_lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        assert stdout_lines, "probe produced no stdout"
        state = _json.loads(stdout_lines[-1])

        assert state["pre"] == "1,3", (
            "Test harness failed to pass CUDA_VISIBLE_DEVICES to subprocess"
        )
        assert state["post"] == "<unset>", (
            "modules/system.py must pop CUDA_VISIBLE_DEVICES at import time "
            "so PyTorch's detection enumerates all physical GPUs, keeping "
            "UI indices consistent with what the generated launch command "
            "will use."
        )

    def test_system_module_is_noop_when_var_not_set(self, tmp_path):
        """If CUDA_VISIBLE_DEVICES wasn't set to begin with, importing
        modules.system must not invent a value or raise."""
        import json as _json
        import os as _os
        import subprocess as _sp

        probe_script = (
            "import os\n"
            "import sys\n"
            f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
            "import modules.system  # noqa: F401\n"
            "import json\n"
            "print(json.dumps({'post': os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}))\n"
        )

        env = {k: v for k, v in _os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
        result = _sp.run(
            [sys.executable, "-c", probe_script],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        stdout_lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        state = _json.loads(stdout_lines[-1])
        assert state["post"] == "<unset>"


def _probe_system_import(preset_env):
    """Start a fresh Python subprocess with ``preset_env``, import
    ``modules.system``, and return the post-import values of
    ``CUDA_VISIBLE_DEVICES`` and ``CUDA_DEVICE_ORDER``.

    This simulates how the launcher boots under any entry method — terminal,
    file-manager double-click, .desktop, Nemo/Nautilus context menu script,
    systemd --user service, cron — since all of them boil down to: "Python
    starts with some env, then imports modules.system." The exercised
    contract is: whatever was inherited, post-import the module has
    (a) popped CUDA_VISIBLE_DEVICES and (b) forced CUDA_DEVICE_ORDER to
    PCI_BUS_ID.
    """
    import json as _json
    import os as _os
    import subprocess as _sp

    probe_script = (
        "import os\n"
        "import sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "import modules.system  # noqa: F401\n"
        "import json\n"
        "print(json.dumps({\n"
        "    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>'),\n"
        "    'cuda_device_order':    os.environ.get('CUDA_DEVICE_ORDER', '<unset>'),\n"
        "    'inherited_cvd': getattr(modules.system, '_INHERITED_CUDA_VISIBLE_DEVICES', None),\n"
        "    'inherited_cdo': getattr(modules.system, '_INHERITED_CUDA_DEVICE_ORDER', None),\n"
        "}))\n"
    )

    # Start from the current env so the child has PATH/HOME/etc., then layer
    # the test's pre-set values on top (and pop any key the caller marked
    # with None to simulate "not set").
    env = dict(_os.environ)
    for k, v in preset_env.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v

    result = _sp.run(
        [sys.executable, "-c", probe_script],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"probe subprocess failed\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    stdout_lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert stdout_lines, f"probe produced no stdout\nstderr: {result.stderr}"
    return _json.loads(stdout_lines[-1]), result.stderr


# ---------------------------------------------------------------------------
# Entry-method scenarios — shell, file manager, .desktop, wrapper, systemd.
# The module-load-time env hygiene must hold regardless of HOW Python was
# started, because every path ultimately becomes "Python imports
# modules.system with some os.environ". The tests below simulate several
# realistic env shapes to pin that contract down.
# ---------------------------------------------------------------------------


class TestEntryMethodScenarios:
    """The fix for inherited CUDA_* env vars is at module load time, so it
    applies uniformly to every way a user might start the launcher:

      - Terminal prompt (``~/.bashrc`` may export the var).
      - Nemo/Nautilus "Open in Terminal" (interactive shell sources
        ``~/.bashrc`` → same as terminal).
      - Ubuntu "Run"/"Run in Terminal" on double-click (session env from
        ``/etc/environment``, ``~/.pam_environment``, or DM startup).
      - ``.desktop`` launcher with or without ``Terminal=true`` (session env
        plus any ``Exec=env VAR=val ...`` prefixes).
      - Wrapper scripts like ``CUDA_VISIBLE_DEVICES=1 python launcher.py``
        (explicitly injected).
      - ``systemd --user`` service (``Environment=`` directives).
      - cron/at job (minimal env, rarely carries CUDA_*).

    The only thing all these paths have in common is that they end with
    Python inheriting some ``os.environ`` and then importing our module.
    These tests prove the fix doesn't care about the caller — just the env
    shape."""

    def test_double_click_style_inherits_session_vars(self):
        """Simulates a desktop-session launch (e.g. Ubuntu's double-click
        "Run" or a .desktop file's Exec=) where CUDA_VISIBLE_DEVICES came
        from /etc/environment or PAM-level profile. Fix must clear it."""
        state, _ = _probe_system_import({"CUDA_VISIBLE_DEVICES": "1,3"})
        assert state["cuda_visible_devices"] == "<unset>"
        assert state["inherited_cvd"] == "1,3"

    def test_nemo_run_in_terminal_inherits_bashrc(self):
        """Simulates Nemo's "Open in Terminal" context menu: the spawned
        shell is interactive, so ``~/.bashrc`` runs and may export
        CUDA_VISIBLE_DEVICES. Same clearing contract."""
        state, _ = _probe_system_import({"CUDA_VISIBLE_DEVICES": "0"})
        assert state["cuda_visible_devices"] == "<unset>"
        assert state["inherited_cvd"] == "0"

    def test_wrapper_script_with_explicit_var_still_cleared(self):
        """Simulates ``CUDA_VISIBLE_DEVICES=1 python launcher.py``. The
        user set this explicitly, but the clearing still happens —
        they'll see the INFO message on stderr telling them to use the
        UI checkboxes instead. This prevents the UI-vs-launch mismatch
        that motivated the fix, at the cost of surprising wrapper users
        (who can see the INFO line and adjust)."""
        state, stderr = _probe_system_import({"CUDA_VISIBLE_DEVICES": "1"})
        assert state["cuda_visible_devices"] == "<unset>"
        # User gets a visible hint about what changed.
        assert "Cleared inherited CUDA_VISIBLE_DEVICES" in stderr

    def test_mig_uuid_value_is_cleared_verbatim(self):
        """MIG (Multi-Instance GPU) slices are referenced by UUID rather
        than integer index. The fix pops whatever string is there —
        UUID or comma-list — same contract."""
        uuid_value = "MIG-GPU-a1b2c3d4-0000-0000-0000-000000000000/0/0"
        state, _ = _probe_system_import({"CUDA_VISIBLE_DEVICES": uuid_value})
        assert state["cuda_visible_devices"] == "<unset>"
        assert state["inherited_cvd"] == uuid_value

    def test_empty_string_is_treated_as_set_and_cleared(self):
        """``CUDA_VISIBLE_DEVICES=""`` explicitly set to empty is a
        legitimate state (tells CUDA "no devices visible"). The fix
        must treat it as set and clear it — otherwise PyTorch would
        see zero devices at detection time while launch would override
        with the user's selection."""
        state, _ = _probe_system_import({"CUDA_VISIBLE_DEVICES": ""})
        assert state["cuda_visible_devices"] == "<unset>"
        # We preserve whatever was inherited (even empty) so the INFO
        # line can be informative.
        assert state["inherited_cvd"] == ""

    def test_cron_minimal_env_no_cuda_var(self):
        """Simulates a cron job with a minimal env — no CUDA_* vars at
        all. The fix must be a no-op: don't invent values, don't crash."""
        state, _ = _probe_system_import({"CUDA_VISIBLE_DEVICES": None})
        assert state["cuda_visible_devices"] == "<unset>"
        assert state["inherited_cvd"] is None

    def test_cuda_device_order_forced_to_pci_bus_id(self):
        """Companion to the CUDA_VISIBLE_DEVICES fix. The launch scripts
        unconditionally emit ``export CUDA_DEVICE_ORDER=PCI_BUS_ID``, so
        detection must do the same — otherwise a shell-inherited
        ``CUDA_DEVICE_ORDER=FASTEST_FIRST`` would make the UI enumerate
        devices by speed rank while llama-server enumerates them by PCIe
        bus, and "GPU 0" in the UI would target a different physical
        device than the launch command."""
        state, stderr = _probe_system_import(
            {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}
        )
        assert state["cuda_device_order"] == "PCI_BUS_ID"
        assert state["inherited_cdo"] == "FASTEST_FIRST"
        # User gets a visible hint about what changed.
        assert "Overriding inherited CUDA_DEVICE_ORDER" in stderr

    def test_cuda_device_order_already_pci_bus_id_no_info_emitted(self):
        """If the inherited value already matches, don't spam the user
        with an "Overriding" message about a non-change."""
        state, stderr = _probe_system_import(
            {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        )
        assert state["cuda_device_order"] == "PCI_BUS_ID"
        assert "Overriding inherited CUDA_DEVICE_ORDER" not in stderr

    def test_cuda_device_order_unset_gets_set_silently(self):
        """If no value was inherited, set PCI_BUS_ID silently (no
        override happened, there's nothing to explain)."""
        state, stderr = _probe_system_import({"CUDA_DEVICE_ORDER": None})
        assert state["cuda_device_order"] == "PCI_BUS_ID"
        assert "Overriding inherited CUDA_DEVICE_ORDER" not in stderr

    def test_both_vars_set_both_normalised(self):
        """Realistic worst case: user has both inherited. Both must
        get normalised by the module-load hygiene."""
        state, _ = _probe_system_import({
            "CUDA_VISIBLE_DEVICES": "1,3",
            "CUDA_DEVICE_ORDER": "FASTEST_FIRST",
        })
        assert state["cuda_visible_devices"] == "<unset>"
        assert state["cuda_device_order"] == "PCI_BUS_ID"


# ---------------------------------------------------------------------------
# Authoritative GPU-mapping dump
# ---------------------------------------------------------------------------


@pytest.fixture()
def system_module():
    """Yield ``modules.system`` with ``CUDA_VISIBLE_DEVICES`` /
    ``CUDA_DEVICE_ORDER`` snapshot/restored around use.

    ``modules.system`` mutates those env vars at import time by design
    (pops the former, forces the latter to ``PCI_BUS_ID``). In the normal
    test session ``tests/system/test_system.py``'s module-level import
    triggers that before any test here runs, but this file may be the
    first to import the module when someone runs
    ``pytest tests/launchers/test_gpu_mapping.py`` in isolation with the
    vars set in their shell. The snapshot/restore keeps the test runner's
    ``os.environ`` honest across tests in that case, so every in-process
    ``modules.system`` usage in this file goes through this fixture.
    """
    import importlib
    import os

    prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    prev_cdo = os.environ.get("CUDA_DEVICE_ORDER")
    # Enter the try BEFORE import_module — modules.system mutates os.environ
    # at module-load time before it imports torch, so an exception from the
    # torch import would otherwise leave the env mutated with no teardown.
    try:
        module = importlib.import_module("modules.system")
        yield module
    finally:
        if prev_cvd is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev_cvd
        if prev_cdo is None:
            os.environ.pop("CUDA_DEVICE_ORDER", None)
        else:
            os.environ["CUDA_DEVICE_ORDER"] = prev_cdo


class TestGpuMappingFormatter:
    """The startup dump is the single source of truth for a user debugging
    a mapping surprise. If the formatter is wrong, the user's mental model
    is wrong, and every other fix in this file becomes hard to trust."""

    def _mk_info(self, devices, manual=False, message=None):
        info = {
            "available": bool(devices),
            "device_count": len(devices),
            "devices": devices,
        }
        if manual:
            info["manual_mode"] = True
        if message:
            info["message"] = message
        return info

    def test_header_announces_mode_auto(self, system_module):
        text = system_module.format_gpu_mapping_table(self._mk_info([
            {"id": 0, "name": "RTX 3090", "total_memory_gb": 24.0,
             "compute_capability": "8.6"}
        ]))
        assert "auto-detected (PCI_BUS_ID order)" in text
        assert "manual GPU mode" not in text

    def test_header_announces_mode_manual(self, system_module):
        text = system_module.format_gpu_mapping_table(self._mk_info([
            {"id": 0, "name": "Planned GPU", "total_memory_gb": 16.0,
             "compute_capability": "Unknown"}
        ], manual=True))
        assert "manual GPU mode" in text
        assert "auto-detected" not in text

    def test_rows_preserve_detection_order(self, system_module):
        """The physical PCIe bus ID ordering must be preserved verbatim in
        the table. Re-sorting (e.g. by name or VRAM) would break the
        promise that "row N = physical GPU N = what launch scripts will
        emit for CUDA_VISIBLE_DEVICES"."""
        devices = [
            {"id": 0, "name": "Zeta GPU", "total_memory_gb": 8.0,
             "compute_capability": "7.5"},
            {"id": 1, "name": "Alpha GPU", "total_memory_gb": 24.0,
             "compute_capability": "8.6"},
            {"id": 2, "name": "Middle GPU", "total_memory_gb": 16.0,
             "compute_capability": "8.0"},
        ]
        text = system_module.format_gpu_mapping_table(self._mk_info(devices))
        zeta = text.index("Zeta GPU")
        alpha = text.index("Alpha GPU")
        middle = text.index("Middle GPU")
        assert zeta < alpha < middle, (
            "Device rows must appear in physical-id order, not alphabetical "
            "or VRAM order — the UI, tensor-split, and CUDA_VISIBLE_DEVICES "
            "all index by this order."
        )

    def test_no_devices_explains_why(self, system_module):
        text = system_module.format_gpu_mapping_table(self._mk_info(
            [], message="PyTorch not found"
        ))
        assert "(no devices" in text
        assert "PyTorch not found" in text

    def test_each_column_present_for_each_gpu(self, system_module):
        devices = [
            {"id": 0, "name": "First", "total_memory_gb": 12.0,
             "compute_capability": "8.0"},
            {"id": 1, "name": "Second", "total_memory_gb": 24.0,
             "compute_capability": "8.9"},
        ]
        text = system_module.format_gpu_mapping_table(self._mk_info(devices))
        # Every value must appear; VRAM rounded to 1 decimal place.
        for expected in ["First", "Second", "12.0 GB", "24.0 GB", "8.0", "8.9"]:
            assert expected in text, f"missing {expected!r} in formatted table"

    def test_footer_explains_the_index_semantics_auto_mode(self, system_module):
        """Users who hit a mapping surprise usually don't know that
        --main-gpu is interpreted AFTER CUDA_VISIBLE_DEVICES remap. The
        dump's footer is the documented place for that hint."""
        text = system_module.format_gpu_mapping_table(self._mk_info([
            {"id": 0, "name": "X", "total_memory_gb": 8.0,
             "compute_capability": "7.5"}
        ]))
        assert "CUDA_VISIBLE_DEVICES" in text
        assert "--main-gpu" in text
        assert "--tensor-split" in text
        # In auto mode, the text should not CLAIM the launcher will unset
        # CUDA_VISIBLE_DEVICES — that's the manual-mode footer.
        assert "unset CUDA_VISIBLE_DEVICES" not in text

    def test_footer_explains_manual_mode_does_not_export(self, system_module):
        """Manual-mode footer must warn that synthetic indices won't be
        emitted as CUDA_VISIBLE_DEVICES — otherwise users would read the
        auto-mode footer and think their selection filters real hardware."""
        text = system_module.format_gpu_mapping_table(self._mk_info([
            {"id": 0, "name": "Planned A", "total_memory_gb": 16.0,
             "compute_capability": "Unknown"}
        ], manual=True))
        # Explicitly states the synthetic nature and the "unset" behaviour.
        assert "synthetic" in text
        assert "unset CUDA_VISIBLE_DEVICES" in text
        # And the auto-mode "single source of truth" wording isn't present.
        assert "single source of truth" not in text

    def test_missing_fields_do_not_crash(self, system_module):
        """Real-world gpu_info from fallback paths can be sparse — ensure
        the formatter degrades gracefully instead of raising KeyError."""
        text = system_module.format_gpu_mapping_table(self._mk_info([{"id": 0}]))
        # Should produce *something* for the id=0 row without raising.
        assert "0" in text
        assert "?" in text  # placeholder for missing fields

    def test_log_helper_writes_to_stream(self, system_module):
        """``log_gpu_mapping`` is the print-to-stream wrapper; separate
        from the formatter so tests (and future log-capture users) can
        depend on either independently."""
        import io

        buf = io.StringIO()
        system_module.log_gpu_mapping({
            "available": True,
            "device_count": 1,
            "devices": [{"id": 0, "name": "GPU", "total_memory_gb": 8.0,
                         "compute_capability": "7.5"}],
        }, stream=buf)
        out = buf.getvalue()
        assert "GPU mapping" in out
        assert "GPU" in out


class TestMappingConsistencyInvariant:
    """All consumers across launcher + launch scripts key off the same
    ``id`` column shown in the dump. Pin down the invariants so a future
    refactor can't silently drift one consumer away from the others."""

    def test_detection_builds_ids_in_range_order(
        self, system_module, monkeypatch
    ):
        """Auto-detection assigns ``id=i`` for ``i in range(device_count)``.
        This is the foundation of the mapping — UI indices, CUDA ordinals,
        gpu_order storage all assume it.

        Stubs ``torch.cuda`` with a deterministic 4-device payload rather
        than calling real detection. CI runners are CPU-only, so the real
        path returns an empty list — the invariant would pass vacuously.
        With the stub, every ``id`` field must equal its list position or
        the assertion fires.
        """
        import types

        def _props(name):
            return types.SimpleNamespace(
                name=name,
                total_memory=16 * 1024**3,
                major=8,
                minor=6,
                multi_processor_count=84,
            )

        device_names = ["GPU-alpha", "GPU-beta", "GPU-gamma", "GPU-delta"]
        fake_cuda = types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: len(device_names),
            get_device_properties=lambda i: _props(device_names[i]),
        )
        monkeypatch.setattr(
            system_module, "torch",
            types.SimpleNamespace(cuda=fake_cuda),
        )
        monkeypatch.setattr(system_module, "TORCH_AVAILABLE", True)

        info = system_module.get_gpu_info_static()

        assert info["device_count"] == len(device_names), (
            "stubbed detection produced the wrong device count — helper drift?"
        )
        for i, dev in enumerate(info["devices"]):
            assert dev["id"] == i, (
                f"device[{i}]['id'] == {dev['id']}, expected {i} — "
                "breaks the invariant that list position == physical id."
            )

    def test_format_table_id_matches_list_position(self, system_module):
        """The formatter must show the ``id`` field verbatim — not the
        list position — so that if detection ever returns an out-of-order
        list the dump would make the discrepancy visible."""
        # Deliberately craft an out-of-order list to verify the formatter
        # prints each row's own ``id`` rather than ``enumerate`` position.
        devices = [
            {"id": 2, "name": "third", "total_memory_gb": 8.0,
             "compute_capability": "7.5"},
            {"id": 0, "name": "first", "total_memory_gb": 12.0,
             "compute_capability": "8.0"},
        ]
        text = system_module.format_gpu_mapping_table({
            "available": True,
            "device_count": 2,
            "devices": devices,
        })
        # The row with name "third" must display id 2, not id 0.
        third_line = [ln for ln in text.splitlines() if "third" in ln][0]
        first_line = [ln for ln in text.splitlines() if "first" in ln][0]
        # Each ID column is the first token in its row after the leading
        # indent.
        assert third_line.strip().startswith("2"), (
            f"formatter used list position instead of id field: {third_line!r}"
        )
        assert first_line.strip().startswith("0"), (
            f"formatter used list position instead of id field: {first_line!r}"
        )


class TestMappingDumpIntegration:
    """End-to-end check: when detection runs, the mapping dump appears on
    stderr with the same indices the launcher will use at launch time."""

    def test_dump_emitted_during_fetch_system_info(self, system_module):
        """SystemInfoManager.fetch_system_info must emit the mapping dump
        after it populates the launcher's detected_gpu_devices."""
        import io
        from contextlib import redirect_stderr

        from unittest.mock import MagicMock

        launcher = MagicMock()
        launcher.venv_dir.get.return_value = ""
        launcher.threads = MagicMock()
        launcher.threads_batch = MagicMock()
        launcher.recommended_threads_var = MagicMock()
        launcher.recommended_threads_batch_var = MagicMock()
        launcher.gpu_detected_status_var = MagicMock()

        fake_info = {
            "available": True,
            "device_count": 2,
            "devices": [
                {"id": 0, "name": "Card A", "total_memory_gb": 24.0,
                 "compute_capability": "8.6"},
                {"id": 1, "name": "Card B", "total_memory_gb": 12.0,
                 "compute_capability": "8.0"},
            ],
        }

        buf = io.StringIO()
        with redirect_stderr(buf), \
             patch.object(system_module, "get_gpu_info_with_venv",
                          return_value=fake_info), \
             patch.object(system_module, "get_ram_info_static",
                          return_value={"total_ram_gb": 32.0}), \
             patch.object(system_module, "get_cpu_info_static",
                          return_value={"logical_cores": 16,
                                        "physical_cores": 8}):
            mgr = system_module.SystemInfoManager(launcher)
            mgr.fetch_system_info()

        captured = buf.getvalue()
        # The dump's header and both device names must appear.
        assert "GPU mapping" in captured
        assert "Card A" in captured
        assert "Card B" in captured
        # And the indices shown match what downstream consumers will see.
        assert launcher.detected_gpu_devices[0]["id"] == 0
        assert launcher.detected_gpu_devices[1]["id"] == 1
