"""Runtime venv path normalization across launch and GPU detection."""

from __future__ import annotations

import importlib.util
import queue
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ENTRY_PATH = REPO_ROOT / "llamacpp-server-launcher.py"


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
    bindir = tmp_path / "venv" / "bin"
    bindir.mkdir(parents=True)
    (bindir / "python").write_text("", encoding="utf-8")
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
    bindir = tmp_path / "venv" / "bin"
    bindir.mkdir(parents=True)
    (bindir / "python").write_text("", encoding="utf-8")
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
