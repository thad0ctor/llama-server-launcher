"""Regression tests for hardware/system detection in modules.system.

Covers:
    * get_gpu_info_static
    * get_gpu_info_with_venv / get_gpu_info_from_venv
    * get_ram_info_static
    * get_cpu_info_static
    * _create_fallback_gpu_info
    * SystemInfoManager.fetch_system_info

All external dependencies (torch.cuda, psutil, subprocess.run, ctypes) are
patched out — no real GPU or admin privileges required.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

# Ensure the project root is importable regardless of how pytest is invoked.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from modules import system as sysmod  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_device_props(
    name: str = "NVIDIA GeForce RTX 9999",
    total_memory: int = 24 * 1024**3,
    major: int = 8,
    minor: int = 9,
    mp_count: int = 128,
):
    return types.SimpleNamespace(
        name=name,
        total_memory=total_memory,
        major=major,
        minor=minor,
        multi_processor_count=mp_count,
    )


# ---------------------------------------------------------------------------
# get_gpu_info_static
# ---------------------------------------------------------------------------


def test_gpu_info_static_when_torch_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sysmod, "torch", None)
    monkeypatch.setattr(sysmod, "TORCH_AVAILABLE", False)

    info = sysmod.get_gpu_info_static()

    assert info["available"] is False
    assert info["device_count"] == 0
    assert info["devices"] == []
    assert "PyTorch" in info["message"]


def test_gpu_info_static_when_torch_present_but_cuda_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(sysmod, "torch", fake_torch)
    monkeypatch.setattr(sysmod, "TORCH_AVAILABLE", False)

    info = sysmod.get_gpu_info_static()

    assert info["available"] is False
    assert info["device_count"] == 0
    assert "CUDA" in info["message"] or "PyTorch" in info["message"]


def test_gpu_info_static_single_device(monkeypatch: pytest.MonkeyPatch) -> None:
    props = _fake_device_props()
    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_properties=lambda i: props,
    )
    fake_torch = types.SimpleNamespace(cuda=fake_cuda)
    monkeypatch.setattr(sysmod, "torch", fake_torch)
    monkeypatch.setattr(sysmod, "TORCH_AVAILABLE", True)

    info = sysmod.get_gpu_info_static()

    assert info["available"] is True
    assert info["device_count"] == 1
    assert len(info["devices"]) == 1
    dev = info["devices"][0]
    assert dev["id"] == 0
    assert dev["name"] == "NVIDIA GeForce RTX 9999"
    assert dev["total_memory_bytes"] == 24 * 1024**3
    assert dev["total_memory_gb"] == 24.0
    assert dev["compute_capability"] == "8.9"
    assert dev["multi_processor_count"] == 128


def test_gpu_info_static_multiple_devices_preserve_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    devices_props = [
        _fake_device_props(name="GPU-A", total_memory=8 * 1024**3, major=7, minor=5),
        _fake_device_props(name="GPU-B", total_memory=16 * 1024**3, major=8, minor=0),
        _fake_device_props(name="GPU-C", total_memory=24 * 1024**3, major=8, minor=9),
    ]
    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 3,
        get_device_properties=lambda i: devices_props[i],
    )
    monkeypatch.setattr(sysmod, "torch", types.SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setattr(sysmod, "TORCH_AVAILABLE", True)

    info = sysmod.get_gpu_info_static()

    assert info["available"] is True
    assert info["device_count"] == 3
    assert [d["id"] for d in info["devices"]] == [0, 1, 2]
    assert [d["name"] for d in info["devices"]] == ["GPU-A", "GPU-B", "GPU-C"]
    assert [d["compute_capability"] for d in info["devices"]] == ["7.5", "8.0", "8.9"]


def test_gpu_info_static_sets_cuda_device_order_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 0,
        get_device_properties=lambda i: _fake_device_props(),
    )
    monkeypatch.setattr(sysmod, "torch", types.SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setattr(sysmod, "TORCH_AVAILABLE", True)

    sysmod.get_gpu_info_static()

    import os
    assert os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"


def test_gpu_info_static_handles_device_query_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(i):
        raise RuntimeError("CUDA driver exploded")

    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_properties=boom,
    )
    monkeypatch.setattr(sysmod, "torch", types.SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setattr(sysmod, "TORCH_AVAILABLE", True)

    info = sysmod.get_gpu_info_static()

    assert info["available"] is False
    assert info["device_count"] == 0
    assert "CUDA driver exploded" in info["message"]


# ---------------------------------------------------------------------------
# get_gpu_info_with_venv / get_gpu_info_from_venv
# ---------------------------------------------------------------------------


def test_gpu_info_with_venv_none_falls_back_to_static(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = {"available": True, "device_count": 0, "devices": [], "message": "x"}
    monkeypatch.setattr(sysmod, "get_gpu_info_static", lambda: sentinel)

    assert sysmod.get_gpu_info_with_venv(None) is sentinel
    assert sysmod.get_gpu_info_with_venv("") is sentinel


def test_gpu_info_with_venv_missing_path_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = {"available": False, "device_count": 0, "devices": [], "message": "fallback"}
    monkeypatch.setattr(sysmod, "get_gpu_info_static", lambda: sentinel)
    # A path that doesn't exist
    result = sysmod.get_gpu_info_with_venv(str(tmp_path / "does_not_exist"))
    assert result is sentinel


def test_gpu_info_from_venv_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Build a fake venv layout with a python binary that exists.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_exe = bin_dir / "python"
    python_exe.write_text("#!/bin/sh\nexit 0\n")
    python_exe.chmod(0o755)

    expected = {
        "available": True,
        "device_count": 2,
        "devices": [
            {"id": 0, "name": "GPU0", "total_memory_gb": 8},
            {"id": 1, "name": "GPU1", "total_memory_gb": 16},
        ],
    }
    fake_result = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=json.dumps(expected), stderr=""
    )
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: fake_result)
    monkeypatch.setattr(sys, "platform", "linux")

    info = sysmod.get_gpu_info_from_venv(str(tmp_path))

    assert info == expected


def test_gpu_info_from_venv_empty_stdout_triggers_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").write_text("")
    (bin_dir / "python").chmod(0o755)

    fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: fake_result)

    sentinel = {"available": False, "device_count": 0, "devices": [], "message": "s"}
    monkeypatch.setattr(sysmod, "get_gpu_info_static", lambda: sentinel)

    info = sysmod.get_gpu_info_from_venv(str(tmp_path))
    # Fallback path should be hit and we get a dict back with message about
    # empty-output / current process.
    assert info["available"] is False
    assert "message" in info


def test_gpu_info_from_venv_bad_json_triggers_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "bin").mkdir()
    python_exe = tmp_path / "bin" / "python"
    python_exe.write_text("")
    python_exe.chmod(0o755)

    fake_result = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="not-json!!", stderr=""
    )
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: fake_result)
    monkeypatch.setattr(
        sysmod,
        "get_gpu_info_static",
        lambda: {"available": False, "device_count": 0, "devices": []},
    )

    info = sysmod.get_gpu_info_from_venv(str(tmp_path))
    assert info["available"] is False


def test_gpu_info_from_venv_nonzero_return_code_triggers_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "bin").mkdir()
    python_exe = tmp_path / "bin" / "python"
    python_exe.write_text("")
    python_exe.chmod(0o755)

    fake_result = subprocess.CompletedProcess(
        args=[], returncode=1, stdout="", stderr="ModuleNotFoundError: torch"
    )
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: fake_result)
    monkeypatch.setattr(
        sysmod,
        "get_gpu_info_static",
        lambda: {"available": False, "device_count": 0, "devices": []},
    )

    info = sysmod.get_gpu_info_from_venv(str(tmp_path))
    assert info["available"] is False


def test_gpu_info_from_venv_subprocess_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "bin").mkdir()
    python_exe = tmp_path / "bin" / "python"
    python_exe.write_text("")
    python_exe.chmod(0o755)

    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="python", timeout=30)

    monkeypatch.setattr(subprocess, "run", raise_timeout)
    monkeypatch.setattr(
        sysmod,
        "get_gpu_info_static",
        lambda: {"available": False, "device_count": 0, "devices": []},
    )

    info = sysmod.get_gpu_info_from_venv(str(tmp_path))
    assert info["available"] is False


def test_gpu_info_from_venv_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "bin").mkdir()
    python_exe = tmp_path / "bin" / "python"
    python_exe.write_text("")
    python_exe.chmod(0o755)

    def raise_perm(*args, **kwargs):
        raise PermissionError("no access")

    monkeypatch.setattr(subprocess, "run", raise_perm)
    monkeypatch.setattr(
        sysmod,
        "get_gpu_info_static",
        lambda: {"available": False, "device_count": 0, "devices": []},
    )

    info = sysmod.get_gpu_info_from_venv(str(tmp_path))
    assert info["available"] is False


def test_gpu_info_from_venv_missing_python_exe_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # tmp_path exists but has no bin/python or python at top-level.
    sentinel = {"available": False, "device_count": 0, "devices": [], "message": "s"}
    monkeypatch.setattr(sysmod, "get_gpu_info_static", lambda: sentinel)

    info = sysmod.get_gpu_info_from_venv(str(tmp_path))
    assert info is sentinel


# ---------------------------------------------------------------------------
# _create_fallback_gpu_info
# ---------------------------------------------------------------------------


def test_create_fallback_uses_static_when_it_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sysmod,
        "get_gpu_info_static",
        lambda: {"available": True, "device_count": 1, "devices": [{"id": 0}]},
    )

    info = sysmod._create_fallback_gpu_info("test-reason")

    assert info["available"] is True
    assert "test-reason" in info["message"]


def test_create_fallback_reports_both_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sysmod,
        "get_gpu_info_static",
        lambda: {"available": False, "device_count": 0, "devices": []},
    )

    info = sysmod._create_fallback_gpu_info("venv-broken")

    assert info["available"] is False
    assert "venv-broken" in info["message"]
    assert "current process also failed" in info["message"]


# ---------------------------------------------------------------------------
# get_ram_info_static
# ---------------------------------------------------------------------------


def test_ram_info_with_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    total = 32 * 1024**3
    avail = 16 * 1024**3
    fake_mem = types.SimpleNamespace(total=total, available=avail)
    fake_psutil = types.SimpleNamespace(virtual_memory=lambda: fake_mem)

    monkeypatch.setattr(sysmod, "psutil", fake_psutil)
    monkeypatch.setattr(sysmod, "PSUTIL_AVAILABLE", True)
    monkeypatch.setattr(sys, "platform", "linux")

    info = sysmod.get_ram_info_static()

    assert info["total_ram_bytes"] == total
    assert info["total_ram_gb"] == 32.0
    assert info["available_ram_bytes"] == avail
    assert info["available_ram_gb"] == 16.0
    assert "error" not in info


def test_ram_info_psutil_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom():
        raise RuntimeError("psutil exploded")

    fake_psutil = types.SimpleNamespace(virtual_memory=boom)
    monkeypatch.setattr(sysmod, "psutil", fake_psutil)
    monkeypatch.setattr(sysmod, "PSUTIL_AVAILABLE", True)
    monkeypatch.setattr(sys, "platform", "linux")

    info = sysmod.get_ram_info_static()

    assert "error" in info
    assert "psutil" in info["error"]


def test_ram_info_without_psutil_on_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sysmod, "psutil", None)
    monkeypatch.setattr(sysmod, "PSUTIL_AVAILABLE", False)
    monkeypatch.setattr(sys, "platform", "linux")

    info = sysmod.get_ram_info_static()

    assert "error" in info
    assert "psutil" in info["error"]


# ---------------------------------------------------------------------------
# get_cpu_info_static
# ---------------------------------------------------------------------------


def test_cpu_info_with_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    def cpu_count(logical=True):
        return 16 if logical else 8

    fake_psutil = types.SimpleNamespace(cpu_count=cpu_count)
    monkeypatch.setattr(sysmod, "psutil", fake_psutil)
    monkeypatch.setattr(sysmod, "PSUTIL_AVAILABLE", True)

    info = sysmod.get_cpu_info_static()

    assert info["logical_cores"] == 16
    assert info["physical_cores"] == 8
    assert info["model_name"] == "N/A"
    assert "error" not in info


def test_cpu_info_with_psutil_none_logical(monkeypatch: pytest.MonkeyPatch) -> None:
    """psutil can return None for logical cores on some platforms."""
    def cpu_count(logical=True):
        return None

    fake_psutil = types.SimpleNamespace(cpu_count=cpu_count)
    monkeypatch.setattr(sysmod, "psutil", fake_psutil)
    monkeypatch.setattr(sysmod, "PSUTIL_AVAILABLE", True)

    info = sysmod.get_cpu_info_static()

    assert info["logical_cores"] == 4  # fallback default
    assert info["physical_cores"] == 2  # fallback default


def test_cpu_info_with_psutil_none_physical_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Logical known, physical returns None -> estimate as logical // 2."""
    def cpu_count(logical=True):
        return 12 if logical else None

    fake_psutil = types.SimpleNamespace(cpu_count=cpu_count)
    monkeypatch.setattr(sysmod, "psutil", fake_psutil)
    monkeypatch.setattr(sysmod, "PSUTIL_AVAILABLE", True)

    info = sysmod.get_cpu_info_static()

    assert info["logical_cores"] == 12
    assert info["physical_cores"] == 6


def test_cpu_info_without_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sysmod, "psutil", None)
    monkeypatch.setattr(sysmod, "PSUTIL_AVAILABLE", False)

    info = sysmod.get_cpu_info_static()

    assert "error" in info
    assert info["logical_cores"] == 4
    assert info["physical_cores"] == 2


def test_cpu_info_psutil_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(logical=True):
        raise RuntimeError("cpu_count failed")

    fake_psutil = types.SimpleNamespace(cpu_count=boom)
    monkeypatch.setattr(sysmod, "psutil", fake_psutil)
    monkeypatch.setattr(sysmod, "PSUTIL_AVAILABLE", True)

    info = sysmod.get_cpu_info_static()

    assert "error" in info
    assert info["logical_cores"] == 4
    assert info["physical_cores"] == 2


# ---------------------------------------------------------------------------
# SystemInfoManager.fetch_system_info
# ---------------------------------------------------------------------------


class _Var:
    """Minimal stand-in for tkinter StringVar used by the launcher."""
    def __init__(self, value=""):
        self._v = value

    def get(self):
        return self._v

    def set(self, value):
        self._v = value


class _FakeLauncher:
    def __init__(self, venv_value=""):
        self.venv_dir = _Var(venv_value)
        self.threads = _Var()
        self.threads_batch = _Var()
        self.recommended_threads_var = _Var()
        self.recommended_threads_batch_var = _Var()
        self.gpu_detected_status_var = _Var()


def test_fetch_system_info_populates_launcher_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_gpu = {
        "available": True,
        "device_count": 1,
        "devices": [{"id": 0, "name": "GPU0"}],
        "message": "ok",
    }
    fake_ram = {"total_ram_bytes": 1, "total_ram_gb": 1,
                "available_ram_bytes": 1, "available_ram_gb": 1}
    fake_cpu = {"logical_cores": 12, "physical_cores": 6, "model_name": "N/A"}

    monkeypatch.setattr(sysmod, "get_gpu_info_with_venv", lambda v: fake_gpu)
    monkeypatch.setattr(sysmod, "get_ram_info_static", lambda: fake_ram)
    monkeypatch.setattr(sysmod, "get_cpu_info_static", lambda: fake_cpu)

    launcher = _FakeLauncher()
    mgr = sysmod.SystemInfoManager(launcher)
    mgr.fetch_system_info()

    assert launcher.gpu_info == fake_gpu
    assert launcher.ram_info == fake_ram
    assert launcher.cpu_info == fake_cpu
    assert launcher.detected_gpu_devices == [{"id": 0, "name": "GPU0"}]
    assert launcher.logical_cores == 12
    assert launcher.physical_cores == 6
    assert launcher.threads.get() == "6"
    assert launcher.threads_batch.get() == "12"
    assert "Recommended: 6" in launcher.recommended_threads_var.get()
    assert "Recommended: 12" in launcher.recommended_threads_batch_var.get()
    # Available GPU -> status var is blank
    assert launcher.gpu_detected_status_var.get() == ""


def test_fetch_system_info_sets_status_when_gpu_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_gpu = {
        "available": False,
        "device_count": 0,
        "devices": [],
        "message": "No CUDA",
    }
    monkeypatch.setattr(sysmod, "get_gpu_info_with_venv", lambda v: fake_gpu)
    monkeypatch.setattr(
        sysmod, "get_ram_info_static", lambda: {"total_ram_gb": 0, "available_ram_gb": 0}
    )
    monkeypatch.setattr(
        sysmod,
        "get_cpu_info_static",
        lambda: {"logical_cores": 4, "physical_cores": 2, "model_name": "N/A"},
    )

    launcher = _FakeLauncher()
    sysmod.SystemInfoManager(launcher).fetch_system_info()

    assert launcher.detected_gpu_devices == []
    assert launcher.gpu_detected_status_var.get() == "No CUDA"


def test_fetch_system_info_uses_configured_venv_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_gpu_with_venv(venv_path):
        captured["venv"] = venv_path
        return {"available": True, "device_count": 0, "devices": [], "message": ""}

    monkeypatch.setattr(sysmod, "get_gpu_info_with_venv", fake_gpu_with_venv)
    monkeypatch.setattr(sysmod, "get_ram_info_static", lambda: {})
    monkeypatch.setattr(
        sysmod,
        "get_cpu_info_static",
        lambda: {"logical_cores": 8, "physical_cores": 4, "model_name": "N/A"},
    )

    launcher = _FakeLauncher(venv_value="  /opt/myvenv  ")
    sysmod.SystemInfoManager(launcher).fetch_system_info()

    # Whitespace gets stripped before being forwarded.
    assert captured["venv"] == "/opt/myvenv"


def test_fetch_system_info_without_venv_passes_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_gpu_with_venv(venv_path):
        captured["venv"] = venv_path
        return {"available": True, "device_count": 0, "devices": [], "message": ""}

    monkeypatch.setattr(sysmod, "get_gpu_info_with_venv", fake_gpu_with_venv)
    monkeypatch.setattr(sysmod, "get_ram_info_static", lambda: {})
    monkeypatch.setattr(
        sysmod,
        "get_cpu_info_static",
        lambda: {"logical_cores": 8, "physical_cores": 4, "model_name": "N/A"},
    )

    launcher = _FakeLauncher(venv_value="")
    sysmod.SystemInfoManager(launcher).fetch_system_info()

    assert captured["venv"] is None


def test_fetch_system_info_handles_cpu_error_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sysmod,
        "get_gpu_info_with_venv",
        lambda v: {"available": False, "device_count": 0, "devices": [], "message": "x"},
    )
    monkeypatch.setattr(sysmod, "get_ram_info_static", lambda: {"error": "no ram"})
    # CPU info dict also carrying the fallback ints
    monkeypatch.setattr(
        sysmod,
        "get_cpu_info_static",
        lambda: {"error": "no cpu", "logical_cores": 4, "physical_cores": 2},
    )

    launcher = _FakeLauncher()
    sysmod.SystemInfoManager(launcher).fetch_system_info()

    assert launcher.logical_cores == 4
    assert launcher.physical_cores == 2
    assert launcher.threads.get() == "2"
    assert launcher.threads_batch.get() == "4"
