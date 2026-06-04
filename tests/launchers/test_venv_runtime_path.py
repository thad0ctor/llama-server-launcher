"""Runtime venv path normalization across launch and GPU detection."""

from __future__ import annotations

import importlib.util
import queue
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ENTRY_PATH = REPO_ROOT / "llamacpp-server-launcher.py"


def _make_platform_venv(base: Path) -> Path:
    """Create a venv-shaped directory under ``base`` for the current platform.

    Windows expects ``Scripts/python.exe``; POSIX expects ``bin/python``.
    Returns the venv directory path.
    """
    venv = base / "venv"
    if sys.platform.startswith("win"):
        bindir = venv / "Scripts"
        exe = bindir / "python.exe"
    else:
        bindir = venv / "bin"
        exe = bindir / "python"
    bindir.mkdir(parents=True)
    exe.write_text("", encoding="utf-8")
    return venv


class _Var:
    def __init__(self, value=""):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


@pytest.fixture(scope="module")
def entry_module():
    spec = importlib.util.spec_from_file_location("entry_module_venv_runtime", ENTRY_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["entry_module_venv_runtime"] = module
    spec.loader.exec_module(module)
    return module


def test_apply_cached_gpu_info_uses_normalized_active_venv_path(
    entry_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _make_platform_venv(tmp_path)
    monkeypatch.setattr(entry_module.venv_manager, "launcher_repo_dir", lambda: tmp_path)
    captured = {}
    cached = {
        "available": True,
        "device_count": 1,
        "devices": [{"id": 0, "name": "GPU0"}],
    }

    def fake_load_cached_gpu_info(config_dir, venv_path):
        captured["config_dir"] = config_dir
        captured["venv_path"] = venv_path
        return cached

    monkeypatch.setattr(entry_module, "load_cached_gpu_info", fake_load_cached_gpu_info)
    stub = SimpleNamespace(
        config_path=tmp_path / "settings.json",
        venv_dir=_Var(""),
        gpu_info={},
        detected_gpu_devices=[],
        gpu_availability_var=_Var(""),
        gpu_detected_status_var=_Var(""),
        _effective_venv_path=lambda: entry_module.LlamaCppLauncher._effective_venv_path(
            SimpleNamespace(venv_dir=_Var(""))
        ),
    )

    hit = entry_module.LlamaCppLauncher._apply_cached_gpu_info_if_any(stub)

    assert hit is True
    assert captured["config_dir"] == tmp_path
    assert captured["venv_path"] == str((tmp_path / "venv").resolve())
    assert stub.detected_gpu_devices == [{"id": 0, "name": "GPU0"}]


def test_launch_manager_effective_venv_path_resolves_relative_input(
    manager,
    launcher_mock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from modules import launch as launch_module

    launcher_mock.venv_dir.set("envs/dev")
    monkeypatch.setattr(launch_module.venv_manager, "launcher_repo_dir", lambda: tmp_path)

    assert manager._effective_venv_path() == str((tmp_path / "envs" / "dev").resolve())


def test_start_system_info_detection_passes_normalized_active_venv_path(
    entry_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _make_platform_venv(tmp_path)
    monkeypatch.setattr(entry_module.venv_manager, "launcher_repo_dir", lambda: tmp_path)
    captured = {}

    class ImmediateThread:
        def __init__(self, target=None, daemon=None):
            self._target = target
            self.daemon = daemon

        def start(self):
            self._target()

    class FakeSystemInfoManager:
        def __init__(self, launcher):
            self.launcher = launcher

        def fetch_system_info(self, venv_path=None, defer_tk_writes=False):
            captured["venv_path"] = venv_path
            captured["defer_tk_writes"] = defer_tk_writes
            self.launcher.gpu_info = {"available": True, "device_count": 0, "devices": []}
            self.launcher.ram_info = {}
            self.launcher.cpu_info = {}
            self.launcher.detected_gpu_devices = []
            self.launcher.logical_cores = 8
            self.launcher.physical_cores = 4

    monkeypatch.setattr(entry_module, "Thread", ImmediateThread)
    monkeypatch.setattr(entry_module, "SystemInfoManager", FakeSystemInfoManager)
    stub = SimpleNamespace(
        venv_dir=_Var(""),
        gpu_detected_status_var=_Var(""),
        _system_info_queue=queue.Queue(),
        _schedule_system_info_drain=lambda: None,
        _effective_venv_path=lambda: entry_module.LlamaCppLauncher._effective_venv_path(
            SimpleNamespace(venv_dir=_Var(""))
        ),
    )

    entry_module.LlamaCppLauncher._start_system_info_detection(stub)

    assert captured["venv_path"] == str((tmp_path / "venv").resolve())
    assert captured["defer_tk_writes"] is True


def test_initial_venv_bootstrap_prompt_opens_terminal_when_managed_deps_are_missing(
    entry_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    statuses = tuple(
        entry_module.venv_manager.DependencyStatus(
            dependency=dep,
            available=False,
            error="missing",
        )
        for dep in entry_module.venv_manager.MANAGED_DEPENDENCIES
    )
    launch_mock = MagicMock()
    info_mock = MagicMock()
    monkeypatch.setattr(entry_module.venv_manager, "probe_current_python_dependencies", lambda *_args: statuses)
    monkeypatch.setattr(entry_module.venv_manager, "launcher_repo_dir", lambda: tmp_path)
    monkeypatch.setattr(
        entry_module.venv_manager,
        "build_bootstrap_venv_command",
        lambda target: f"bootstrap {target}",
    )
    monkeypatch.setattr(entry_module.terminal_launcher, "open_command_in_terminal", launch_mock)
    monkeypatch.setattr(entry_module.messagebox, "showinfo", info_mock)
    monkeypatch.setattr(entry_module.messagebox, "showerror", MagicMock())
    stub = SimpleNamespace(
        app_settings={"venv_bootstrap_prompt_mode": "ask", "last_venv_dir": ""},
        repo_dir=tmp_path,
        venv_dir=_Var(""),
        _bootstrap_config_dirty=False,
        _effective_venv_path=lambda: "",
        # _maybe_prompt_for_initial_venv_setup looks the bootstrap-action
        # callable up via ``self._ask_initial_venv_bootstrap_action``; with a
        # SimpleNamespace stub, attribute lookup never reaches the patched
        # class, so attach it directly to the instance.
        _ask_initial_venv_bootstrap_action=lambda **kwargs: "create",
    )

    entry_module.LlamaCppLauncher._maybe_prompt_for_initial_venv_setup(stub)

    expected = str((tmp_path / "venv").resolve())
    launch_mock.assert_called_once_with(f"bootstrap {Path(expected)}", cwd=tmp_path)
    assert stub.venv_dir.get() == expected
    assert stub.app_settings["last_venv_dir"] == expected
    assert stub.app_settings["venv_bootstrap_prompt_mode"] == "ask"
    assert stub._bootstrap_config_dirty is True
    assert info_mock.called


def test_initial_venv_bootstrap_prompt_uses_configured_path_when_not_a_real_venv(
    entry_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    dep = entry_module.venv_manager.MANAGED_DEPENDENCIES[0]
    statuses = (
        entry_module.venv_manager.DependencyStatus(
            dependency=dep,
            available=False,
            error="missing",
        ),
    )
    target = tmp_path / "custom-env"
    launch_mock = MagicMock()
    monkeypatch.setattr(entry_module.venv_manager, "probe_current_python_dependencies", lambda *_args: statuses)
    monkeypatch.setattr(entry_module.venv_manager, "launcher_repo_dir", lambda: tmp_path)
    monkeypatch.setattr(
        entry_module.venv_manager,
        "build_bootstrap_venv_command",
        lambda chosen: f"bootstrap {chosen}",
    )
    monkeypatch.setattr(entry_module.terminal_launcher, "open_command_in_terminal", launch_mock)
    monkeypatch.setattr(entry_module.messagebox, "showinfo", MagicMock())
    monkeypatch.setattr(entry_module.messagebox, "showerror", MagicMock())
    stub = SimpleNamespace(
        app_settings={"venv_bootstrap_prompt_mode": "ask", "last_venv_dir": str(target)},
        repo_dir=tmp_path,
        venv_dir=_Var(str(target)),
        _bootstrap_config_dirty=False,
        _effective_venv_path=lambda: str(target),
        _ask_initial_venv_bootstrap_action=lambda **kwargs: "create",
    )

    entry_module.LlamaCppLauncher._maybe_prompt_for_initial_venv_setup(stub)

    launch_mock.assert_called_once_with(f"bootstrap {target.resolve()}", cwd=tmp_path)
    assert stub.venv_dir.get() == str(target.resolve())


def test_initial_venv_bootstrap_prompt_skips_when_all_deps_exist(
    entry_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    statuses = tuple(
        entry_module.venv_manager.DependencyStatus(
            dependency=dep,
            available=True,
            version="1.0.0",
        )
        for dep in entry_module.venv_manager.MANAGED_DEPENDENCIES
    )
    prompt_mock = MagicMock(return_value="create")
    monkeypatch.setattr(entry_module.venv_manager, "probe_current_python_dependencies", lambda *_args: statuses)
    monkeypatch.setattr(entry_module.LlamaCppLauncher, "_ask_initial_venv_bootstrap_action", prompt_mock)
    stub = SimpleNamespace(
        app_settings={"venv_bootstrap_prompt_mode": "ask"},
        repo_dir=tmp_path,
        venv_dir=_Var(""),
        _bootstrap_config_dirty=False,
        _effective_venv_path=lambda: "",
    )

    entry_module.LlamaCppLauncher._maybe_prompt_for_initial_venv_setup(stub)

    prompt_mock.assert_not_called()
    assert stub.app_settings["venv_bootstrap_prompt_mode"] == "ask"


def test_initial_venv_bootstrap_prompt_does_not_suppress_retry_on_terminal_error(
    entry_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    statuses = tuple(
        entry_module.venv_manager.DependencyStatus(
            dependency=dep,
            available=False,
            error="missing",
        )
        for dep in entry_module.venv_manager.MANAGED_DEPENDENCIES
    )
    monkeypatch.setattr(entry_module.venv_manager, "probe_current_python_dependencies", lambda *_args: statuses)
    monkeypatch.setattr(entry_module.venv_manager, "launcher_repo_dir", lambda: tmp_path)
    monkeypatch.setattr(entry_module.venv_manager, "build_bootstrap_venv_command", lambda target: f"bootstrap {target}")
    monkeypatch.setattr(entry_module.terminal_launcher, "open_command_in_terminal", MagicMock(side_effect=OSError("no terminal")))
    monkeypatch.setattr(entry_module.messagebox, "showinfo", MagicMock())
    error_mock = MagicMock()
    monkeypatch.setattr(entry_module.messagebox, "showerror", error_mock)
    stub = SimpleNamespace(
        app_settings={"venv_bootstrap_prompt_mode": "ask", "last_venv_dir": ""},
        repo_dir=tmp_path,
        venv_dir=_Var(""),
        _bootstrap_config_dirty=False,
        _effective_venv_path=lambda: "",
        _ask_initial_venv_bootstrap_action=lambda **kwargs: "create",
    )

    entry_module.LlamaCppLauncher._maybe_prompt_for_initial_venv_setup(stub)

    assert stub.app_settings["venv_bootstrap_prompt_mode"] == "ask"
    assert stub._bootstrap_config_dirty is False
    error_mock.assert_called_once()


def test_initial_venv_bootstrap_prompt_no_keeps_prompt_enabled(
    entry_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    dep = entry_module.venv_manager.MANAGED_DEPENDENCIES[0]
    statuses = (
        entry_module.venv_manager.DependencyStatus(
            dependency=dep,
            available=False,
            error="missing",
        ),
    )
    prompt_mock = MagicMock(return_value="skip")
    monkeypatch.setattr(entry_module.venv_manager, "probe_current_python_dependencies", lambda *_args: statuses)
    stub = SimpleNamespace(
        app_settings={"venv_bootstrap_prompt_mode": "ask"},
        repo_dir=tmp_path,
        venv_dir=_Var(""),
        _bootstrap_config_dirty=False,
        _effective_venv_path=lambda: "",
        _ask_initial_venv_bootstrap_action=prompt_mock,
    )

    entry_module.LlamaCppLauncher._maybe_prompt_for_initial_venv_setup(stub)

    prompt_mock.assert_called_once()
    assert stub.app_settings["venv_bootstrap_prompt_mode"] == "ask"
    assert stub._bootstrap_config_dirty is False


def test_initial_venv_bootstrap_prompt_never_disables_future_prompts(
    entry_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    dep = entry_module.venv_manager.MANAGED_DEPENDENCIES[0]
    statuses = (
        entry_module.venv_manager.DependencyStatus(
            dependency=dep,
            available=False,
            error="missing",
        ),
    )
    monkeypatch.setattr(entry_module.venv_manager, "probe_current_python_dependencies", lambda *_args: statuses)
    stub = SimpleNamespace(
        app_settings={"venv_bootstrap_prompt_mode": "ask"},
        repo_dir=tmp_path,
        venv_dir=_Var(""),
        _bootstrap_config_dirty=False,
        _effective_venv_path=lambda: "",
        _ask_initial_venv_bootstrap_action=lambda **kwargs: "never",
    )

    entry_module.LlamaCppLauncher._maybe_prompt_for_initial_venv_setup(stub)

    assert stub.app_settings["venv_bootstrap_prompt_mode"] == "never"
    assert stub._bootstrap_config_dirty is True
