"""Edge-case regression tests for hardware/system detection paths.

Covers:
    * Windows venv Python discovery: Scripts\\python.exe present vs
      python.exe fallback.
    * Windows RAM detection via ``ctypes.windll.kernel32.GlobalMemoryStatusEx``
      (fully mocked — no real Windows required).
    * Zero-GPU case with torch available (device_count == 0).
    * ROCm fake-torch path: HIP devices masquerade as CUDA through PyTorch.
    * ARM / Apple Silicon platform detection.
    * Non-UTF-8 metadata strings (Latin-1 bytes) don't crash the parser.
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import types
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from modules import system as sysmod  # noqa: E402


# ---------------------------------------------------------------------------
# Windows venv Python discovery
# ---------------------------------------------------------------------------


def test_venv_windows_scripts_python_preferred(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On Windows, Scripts/python.exe is the canonical venv layout; the
    function must prefer it when present."""
    monkeypatch.setattr(sys, "platform", "win32")

    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    canonical = scripts / "python.exe"
    canonical.write_text("")
    # Also put a top-level python.exe to verify canonical is preferred.
    (tmp_path / "python.exe").write_text("")

    captured = {}

    def fake_run(args, **kwargs):
        captured["exe"] = args[0]
        return subprocess.CompletedProcess(
            args=args, returncode=0,
            stdout=json.dumps({"available": True, "device_count": 0, "devices": []}),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    sysmod.get_gpu_info_from_venv(str(tmp_path))

    assert captured["exe"] == str(canonical)


def test_venv_windows_top_level_python_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If Scripts/python.exe is missing, the top-level python.exe fallback
    must be picked up (the 'Some venv structures' branch)."""
    monkeypatch.setattr(sys, "platform", "win32")

    # No Scripts directory — only top-level python.exe.
    fallback = tmp_path / "python.exe"
    fallback.write_text("")

    captured = {}

    def fake_run(args, **kwargs):
        captured["exe"] = args[0]
        return subprocess.CompletedProcess(
            args=args, returncode=0,
            stdout=json.dumps({"available": False, "device_count": 0, "devices": []}),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    sysmod.get_gpu_info_from_venv(str(tmp_path))

    assert captured["exe"] == str(fallback)


def test_venv_linux_top_level_python_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On Linux, if bin/python is missing, top-level ./python is tried."""
    monkeypatch.setattr(sys, "platform", "linux")

    fallback = tmp_path / "python"
    fallback.write_text("")
    fallback.chmod(0o755)

    captured = {}

    def fake_run(args, **kwargs):
        captured["exe"] = args[0]
        return subprocess.CompletedProcess(
            args=args, returncode=0,
            stdout=json.dumps({"available": False, "device_count": 0, "devices": []}),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    sysmod.get_gpu_info_from_venv(str(tmp_path))

    assert captured["exe"] == str(fallback)


# ---------------------------------------------------------------------------
# Windows RAM detection via ctypes (fully mocked)
# ---------------------------------------------------------------------------


class _FakeMemStatus:
    """Stand-in for MEMORYSTATUSEX whose fields get written by the
    GlobalMemoryStatusEx mock."""

    def __init__(self) -> None:
        self.dwLength = 0
        self.dwMemoryLoad = 0
        self.ullTotalPhys = 0
        self.ullAvailPhys = 0
        self.ullTotalPageFile = 0
        self.ullAvailPageFile = 0
        self.ullTotalVirtual = 0
        self.ullAvailVirtual = 0
        self.ullAvailExtendedVirtual = 0


def test_windows_ram_via_ctypes_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock ctypes.windll.kernel32 so we can exercise the Windows RAM branch
    from Linux. The Global* call should fill the struct and the function
    should convert the byte values correctly."""
    monkeypatch.setattr(sys, "platform", "win32")

    import ctypes

    target = {"total": 64 * 1024**3, "avail": 32 * 1024**3}

    # The module references `ctypes.windll.kernel32` at call time inside the
    # try block; attach a fake windll that exposes kernel32 to our patched
    # ctypes module.
    def fake_global_memory_status_ex(pointer):
        # pointer is a ctypes.byref(memInfo); ctypes.byref doesn't expose
        # attribute access, so instead of writing through the pointer we
        # rely on the pre-populated instance returned by MEMORYSTATUSEX().
        # We capture the struct through closure via monkeypatching
        # MEMORYSTATUSEX below. This function just reports success.
        return 1

    fake_kernel32 = types.SimpleNamespace(
        GlobalMemoryStatusEx=fake_global_memory_status_ex
    )
    fake_windll = types.SimpleNamespace(kernel32=fake_kernel32)

    # ctypes.windll doesn't exist on Linux; patch it on the module's ctypes.
    monkeypatch.setattr(ctypes, "windll", fake_windll, raising=False)

    # MEMORYSTATUSEX is a local class declared inside get_ram_info_static,
    # so we can't monkeypatch it directly. Instead, patch ctypes.Structure's
    # __init__ so instantiating the struct returns something with the fields
    # we want. Simpler: patch ctypes.sizeof and override Structure via a
    # wrapper that returns our fake after construction.

    # Easiest path: replace ctypes.Structure in the module scope so the
    # local class inherits our fake, and override the constructor to seed
    # the memory values.
    class _FakeStructure:
        def __init__(self) -> None:
            self.dwLength = 0
            self.dwMemoryLoad = 0
            self.ullTotalPhys = target["total"]
            self.ullAvailPhys = target["avail"]
            self.ullTotalPageFile = 0
            self.ullAvailPageFile = 0
            self.ullTotalVirtual = 0
            self.ullAvailVirtual = 0
            self.ullAvailExtendedVirtual = 0

        def __init_subclass__(cls, **kwargs):
            pass

    monkeypatch.setattr(ctypes, "Structure", _FakeStructure, raising=False)
    monkeypatch.setattr(ctypes, "sizeof", lambda x: 64, raising=False)
    monkeypatch.setattr(ctypes, "byref", lambda x: x, raising=False)

    info = sysmod.get_ram_info_static()

    assert "error" not in info or info.get("total_ram_bytes") == target["total"]
    # Core correctness: Byte → GB conversion matches.
    assert info["total_ram_bytes"] == target["total"]
    assert info["total_ram_gb"] == round(target["total"] / (1024**3), 2)
    assert info["available_ram_bytes"] == target["avail"]


def test_windows_ram_ctypes_fails_psutil_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows path with ctypes throwing; if psutil is available the RAM
    check should still return a valid result via fallback."""
    monkeypatch.setattr(sys, "platform", "win32")

    import ctypes

    # Make the MEMORYSTATUSEX creation itself raise so we hit the broad
    # `except Exception as e_win:` psutil fallback branch.
    def _boom(*a, **kw):
        raise RuntimeError("ctypes exploded")

    monkeypatch.setattr(ctypes, "windll", _boom, raising=False)

    total = 16 * 1024**3
    avail = 8 * 1024**3
    fake_mem = types.SimpleNamespace(total=total, available=avail)
    fake_psutil = types.SimpleNamespace(virtual_memory=lambda: fake_mem)
    monkeypatch.setattr(sysmod, "psutil", fake_psutil)
    monkeypatch.setattr(sysmod, "PSUTIL_AVAILABLE", True)

    info = sysmod.get_ram_info_static()

    # psutil fallback kicks in → total_ram_bytes is set.
    assert info.get("total_ram_bytes") == total
    assert info.get("available_ram_bytes") == avail


def test_windows_ram_ctypes_fails_no_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows path with ctypes failing AND psutil unavailable — must
    produce an error dict rather than crashing."""
    monkeypatch.setattr(sys, "platform", "win32")

    import ctypes

    def _boom(*a, **kw):
        raise RuntimeError("ctypes exploded")

    monkeypatch.setattr(ctypes, "windll", _boom, raising=False)
    monkeypatch.setattr(sysmod, "psutil", None)
    monkeypatch.setattr(sysmod, "PSUTIL_AVAILABLE", False)

    info = sysmod.get_ram_info_static()
    assert "error" in info


# ---------------------------------------------------------------------------
# Zero-GPU case
# ---------------------------------------------------------------------------


def test_gpu_info_static_zero_devices_with_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Torch is installed, cuda.is_available() returns True, but the system
    reports 0 devices. The function should report available=True but an
    empty devices list (this is what torch actually does on some cloud VMs)."""
    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 0,
        get_device_properties=lambda i: (_ for _ in ()).throw(
            AssertionError("should not be called for zero devices")
        ),
    )
    monkeypatch.setattr(sysmod, "torch", types.SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setattr(sysmod, "TORCH_AVAILABLE", True)

    info = sysmod.get_gpu_info_static()

    assert info["available"] is True
    assert info["device_count"] == 0
    assert info["devices"] == []


# ---------------------------------------------------------------------------
# ROCm / HIP path: PyTorch-ROCm still exposes torch.cuda, devices appear as
# "AMD Radeon ...". Mock it to confirm detection doesn't choke on non-NVIDIA.
# ---------------------------------------------------------------------------


def test_gpu_info_static_rocm_like_device(monkeypatch: pytest.MonkeyPatch) -> None:
    amd_props = types.SimpleNamespace(
        name="AMD Radeon Pro W7900",
        total_memory=48 * 1024**3,
        major=11,
        minor=0,
        multi_processor_count=96,
    )
    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_properties=lambda i: amd_props,
    )
    monkeypatch.setattr(sysmod, "torch", types.SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setattr(sysmod, "TORCH_AVAILABLE", True)

    info = sysmod.get_gpu_info_static()

    assert info["available"] is True
    assert info["device_count"] == 1
    assert "AMD" in info["devices"][0]["name"]
    assert info["devices"][0]["total_memory_gb"] == 48.0


# ---------------------------------------------------------------------------
# ARM / Apple Silicon platform detection (CPU info path)
# ---------------------------------------------------------------------------


def test_cpu_info_on_apple_silicon(monkeypatch: pytest.MonkeyPatch) -> None:
    """psutil works fine on Apple Silicon; the function should return
    the real core counts regardless of platform string."""
    monkeypatch.setattr(sys, "platform", "darwin")

    def cpu_count(logical=True):
        return 12 if logical else 12  # Apple Silicon M2 Pro-ish: no SMT

    fake_psutil = types.SimpleNamespace(cpu_count=cpu_count)
    monkeypatch.setattr(sysmod, "psutil", fake_psutil)
    monkeypatch.setattr(sysmod, "PSUTIL_AVAILABLE", True)

    info = sysmod.get_cpu_info_static()

    assert info["logical_cores"] == 12
    assert info["physical_cores"] == 12


def test_ram_info_on_apple_silicon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")

    total = 64 * 1024**3
    avail = 40 * 1024**3
    fake_mem = types.SimpleNamespace(total=total, available=avail)
    fake_psutil = types.SimpleNamespace(virtual_memory=lambda: fake_mem)
    monkeypatch.setattr(sysmod, "psutil", fake_psutil)
    monkeypatch.setattr(sysmod, "PSUTIL_AVAILABLE", True)

    info = sysmod.get_ram_info_static()

    assert info["total_ram_bytes"] == total
    assert info["available_ram_bytes"] == avail


# ---------------------------------------------------------------------------
# Non-UTF-8 metadata strings (parser uses errors='replace')
# ---------------------------------------------------------------------------


def test_parse_non_utf8_metadata_does_not_crash(tmp_path: Path) -> None:
    """Latin-1 bytes smuggled into a STRING value should decode with
    replacement characters, not raise."""
    from modules.system import parse_gguf_header_simple

    # Hand-build a metadata entry whose value bytes aren't valid UTF-8.
    bad_bytes = b"\xfffo\xe9o"  # contains 0xFF + 0xE9 (é in Latin-1)

    blob = bytearray()
    blob += b"GGUF"
    blob += struct.pack("<I", 3)
    blob += struct.pack("<Q", 0)
    blob += struct.pack("<Q", 2)

    # Key (ASCII, legal UTF-8), value bytes are not legal UTF-8.
    key_bytes = b"general.name"
    blob += struct.pack("<Q", len(key_bytes)) + key_bytes
    blob += struct.pack("<I", 8)  # STRING
    blob += struct.pack("<Q", len(bad_bytes)) + bad_bytes

    # Legit architecture afterward so we can confirm parsing continued.
    arch_key = b"general.architecture"
    blob += struct.pack("<Q", len(arch_key)) + arch_key
    blob += struct.pack("<I", 8)
    arch_val = b"llama"
    blob += struct.pack("<Q", len(arch_val)) + arch_val

    path = tmp_path / "latin1.gguf"
    path.write_bytes(bytes(blob))

    result = parse_gguf_header_simple(str(path))

    assert result["error"] is None
    # errors='replace' turns the bad bytes into U+FFFD (replacement char).
    assert "general.name" in result["metadata"]
    assert "\ufffd" in result["metadata"]["general.name"]
    # And the parser kept going to find the architecture.
    assert result["architecture"] == "llama"


def test_parse_non_utf8_key_does_not_crash(tmp_path: Path) -> None:
    """Even key bytes that aren't valid UTF-8 must not crash the parser."""
    from modules.system import parse_gguf_header_simple

    bad_key = b"\xfe\xff.bad_key"

    blob = bytearray()
    blob += b"GGUF"
    blob += struct.pack("<I", 3)
    blob += struct.pack("<Q", 0)
    blob += struct.pack("<Q", 2)

    blob += struct.pack("<Q", len(bad_key)) + bad_key
    blob += struct.pack("<I", 8)
    val = b"whatever"
    blob += struct.pack("<Q", len(val)) + val

    # Real arch after
    arch_key = b"general.architecture"
    blob += struct.pack("<Q", len(arch_key)) + arch_key
    blob += struct.pack("<I", 8)
    arch_val = b"llama"
    blob += struct.pack("<Q", len(arch_val)) + arch_val

    path = tmp_path / "badkey.gguf"
    path.write_bytes(bytes(blob))

    result = parse_gguf_header_simple(str(path))

    assert result["error"] is None
    assert result["architecture"] == "llama"
