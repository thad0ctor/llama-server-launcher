#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
import traceback
import ctypes
import struct
import csv
import importlib.util
import io
from pathlib import Path

# Force CUDA device ordering to PCI_BUS_ID before importing torch or touching
# CUDA state.
#
# This must be a hard override (not ``setdefault``) — the generated launch
# scripts unconditionally ``export CUDA_DEVICE_ORDER=PCI_BUS_ID``. If we
# honoured a user-inherited value like ``FASTEST_FIRST`` here, PyTorch would
# enumerate devices by speed rank while llama-server enumerates them by
# PCIe bus, so the UI's "GPU 0" would point at a different physical card
# than the one the launch command eventually targets.
_INHERITED_CUDA_DEVICE_ORDER = os.environ.get("CUDA_DEVICE_ORDER")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if _INHERITED_CUDA_DEVICE_ORDER and _INHERITED_CUDA_DEVICE_ORDER != "PCI_BUS_ID":
    print(
        f"INFO: Overriding inherited CUDA_DEVICE_ORDER='{_INHERITED_CUDA_DEVICE_ORDER}' "
        "→ 'PCI_BUS_ID' so GPU detection enumerates devices the same way the "
        "generated launch command will.",
        file=sys.stderr,
    )

# Clear any inherited CUDA_VISIBLE_DEVICES before torch touches CUDA state.
#
# If the user ran the launcher from a shell that already had
# CUDA_VISIBLE_DEVICES set (e.g. they pre-filtered devices), PyTorch would
# detect the filtered view and label those devices 0..N-1 in the UI
# checkboxes. When the user then selected "GPU 0" and launched, the script
# would emit ``export CUDA_VISIBLE_DEVICES=0`` (physical 0 under PCI_BUS_ID
# ordering) — which is NOT the physical GPU the UI showed them. The UI index
# and the launch-time index would point at different physical cards.
#
# Popping the variable here — before ``import torch`` — ensures detection
# enumerates *all* physical GPUs, so UI indices == PCIe indices == what the
# generated launch script will use. Users can still hide GPUs by unchecking
# them in the UI; the emitted CUDA_VISIBLE_DEVICES will reflect that choice.
_INHERITED_CUDA_VISIBLE_DEVICES = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
if _INHERITED_CUDA_VISIBLE_DEVICES is not None:
    print(
        f"INFO: Cleared inherited CUDA_VISIBLE_DEVICES='{_INHERITED_CUDA_VISIBLE_DEVICES}' "
        "so GPU detection can enumerate all physical devices. Use the GPU "
        "checkbox list in the UI to select which GPUs to use — the generated "
        "launch command will restrict the server accordingly.",
        file=sys.stderr,
    )

# Optional debug prints for Python environment. Gated behind an explicit
# env flag because the unconditional dump (a) leaks usernames / home paths
# into stderr on every import and (b) clutters non-debug output. Set
# ``LLAMA_LAUNCHER_DEBUG_ENV=1`` to re-enable.
if os.environ.get("LLAMA_LAUNCHER_DEBUG_ENV") == "1":
    print("\n=== Python Environment Debug Info ===", file=sys.stderr)
    print(f"Python executable: {sys.executable}", file=sys.stderr)
    print(f"Python version: {sys.version}", file=sys.stderr)
    print(f"sys.path: {sys.path}", file=sys.stderr)
    print("===================================\n", file=sys.stderr)

# Torch is intentionally imported lazily. Importing the module and especially
# calling torch.cuda.is_available() can initialize CUDA on the Tk main thread
# during app startup. GPU detection now prefers nvidia-smi and only touches
# torch inside the background detection worker.
torch = None
_TORCH_IMPORT_ERROR = None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


# Check for requests module (required for version checking)
try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests library not found. Version checking and updates will not work.", file=sys.stderr)
except Exception as e:
    REQUESTS_AVAILABLE = False
    print(f"Warning: requests import failed: {e}", file=sys.stderr)

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None
    print("Warning: psutil library not found. RAM and CPU information may be limited.", file=sys.stderr)
except Exception as e:
    # A binary-incompatible or partially-installed ``psutil`` (e.g.
    # broken native extension after a Python upgrade, missing
    # platform-specific .so file) can raise non-ImportError exceptions
    # at import time. The dependency is optional — surface a warning
    # and keep the launcher startable instead of crashing the whole
    # process. Mirrors the ``requests`` handling above.
    PSUTIL_AVAILABLE = False
    psutil = None
    print(f"Warning: psutil import failed: {e}", file=sys.stderr)


# --- Dependency Check (Printed to console/stderr) ---
MISSING_DEPS = []

# Required dependencies
if not REQUESTS_AVAILABLE:
    MISSING_DEPS.append("requests (required for version checking and updates)")

# PyTorch is optional when nvidia-smi is available, but still useful as a
# fallback path for CUDA/Metal inspection in some environments.
if not TORCH_AVAILABLE:
    MISSING_DEPS.append("PyTorch (optional - fallback GPU detection and CUDA features)")

# Optional dependencies
if not PSUTIL_AVAILABLE:
    MISSING_DEPS.append("psutil (optional - provides enhanced system information)")

# Only print missing deps warning if there are actually missing deps
if MISSING_DEPS:
    print("\n--- Missing Dependencies Warning ---")
    print("The following Python libraries are recommended for full functionality but were not found:")
    for dep in MISSING_DEPS:
        print(f" - {dep}")
    print("Please install them using 'pip install -r requirements.txt' or individually with pip.")
    print("-------------------------------------\n")


# ═════════════════════════════════════════════════════════════════════
#  Helper Functions (These remain outside the class as they don't need 'self')
# ═════════════════════════════════════════════════════════════════════


def _load_torch_module():
    """Import torch lazily for the torch fallback path."""
    global torch, TORCH_AVAILABLE, _TORCH_IMPORT_ERROR
    if torch is not None:
        return torch
    if not TORCH_AVAILABLE:
        return None
    try:
        import torch as torch_module  # noqa: WPS433
    except ImportError as exc:
        TORCH_AVAILABLE = False
        _TORCH_IMPORT_ERROR = exc
        return None
    except Exception as exc:
        TORCH_AVAILABLE = False
        _TORCH_IMPORT_ERROR = exc
        print(f"Warning: PyTorch import failed: {exc}", file=sys.stderr)
        return None
    torch = torch_module
    return torch


def _normalize_compute_capability(value):
    text = str(value or "").strip()
    if not text or text.upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}:
        return "Unknown"
    match = re.search(r"(\d+)(?:\.(\d+))?", text)
    if not match:
        return "Unknown"
    major = match.group(1)
    minor = match.group(2) if match.group(2) is not None else "0"
    return f"{major}.{minor}"


def _unavailable_gpu_info(message, source):
    return {
        "available": False,
        "message": message,
        "device_count": 0,
        "devices": [],
        "detection_source": source,
    }


# nvidia-smi emits the bus id as zero-padded hex
# ``domain:bus:device.function`` (e.g. ``00000000:01:00.0``). Match
# exactly that shape so blank / truncated / non-bus-id values don't
# pass the validation in ``get_gpu_info_from_nvidia_smi`` and silently
# remap launcher GPU ids during the PCI_BUS_ID sort.
_PCI_BUS_ID_RE = re.compile(r"^[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9A-Fa-f]$")


def get_gpu_info_from_nvidia_smi(timeout=5):
    """Get GPU information via nvidia-smi without initializing CUDA."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,pci.bus_id,name,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return _unavailable_gpu_info("nvidia-smi not found", "nvidia-smi")
    except subprocess.TimeoutExpired:
        return _unavailable_gpu_info("nvidia-smi timed out", "nvidia-smi")
    except Exception as exc:
        return _unavailable_gpu_info(f"nvidia-smi failed: {exc}", "nvidia-smi")

    if result.returncode != 0:
        error = (result.stderr or result.stdout or "").strip() or "unknown error"
        return _unavailable_gpu_info(f"nvidia-smi failed: {error}", "nvidia-smi")

    rows = []
    try:
        reader = csv.reader(io.StringIO(result.stdout.strip()))
        for row in reader:
            fields = [part.strip() for part in row]
            if len(fields) < 5:
                # Fail closed on a short row rather than silently
                # skipping it — a malformed nvidia-smi line means
                # one or more fields couldn't be parsed, and
                # quietly returning the rest of the rows would
                # mask a real driver / format mismatch and let
                # ``fetch_system_info`` short-circuit on an
                # incomplete GPU list. Raise so the caller falls
                # through to the next detector.
                raise ValueError(f"nvidia-smi row has fewer than 5 fields (got {len(fields)}): {row!r}")
            _nvidia_idx, pci_bus_id, name, memory_total_mib, compute_cap = fields[:5]
            # Validate ``pci_bus_id`` shape (``DDDDDDDD:BB:DD.F``,
            # zero-padded hex domain:bus:device.function) BEFORE
            # the rows.sort below — the launcher remaps nvidia-smi's
            # natural order onto PCI_BUS_ID-sorted ids, and a blank
            # or malformed bus id would sort to the front and
            # silently shift every other device's launcher index,
            # breaking GPU-selection correctness in a way the user
            # would have to debug from CUDA_VISIBLE_DEVICES output.
            # Fail closed: bubble out to the next detector cascade.
            if not _PCI_BUS_ID_RE.fullmatch(pci_bus_id):
                raise ValueError(f"nvidia-smi row has malformed pci.bus_id: {pci_bus_id!r}")
            try:
                total_mib = float(memory_total_mib)
            except (TypeError, ValueError):
                total_mib = 0.0
            total_bytes = int(total_mib * 1024 * 1024)
            rows.append(
                {
                    "pci_bus_id": pci_bus_id,
                    "name": name or "Unknown GPU",
                    "total_memory_bytes": total_bytes,
                    "total_memory_gb": round(total_bytes / (1024**3), 2),
                    "compute_capability": _normalize_compute_capability(compute_cap),
                    # nvidia-smi does not expose SM count through the query API.
                    "multi_processor_count": None,
                }
            )
    except Exception as exc:
        return _unavailable_gpu_info(f"Failed to parse nvidia-smi output: {exc}", "nvidia-smi")

    if not rows:
        return _unavailable_gpu_info("nvidia-smi reported no CUDA devices", "nvidia-smi")

    # Match CUDA_DEVICE_ORDER=PCI_BUS_ID: assign launcher ids after sorting by
    # PCI bus, not by nvidia-smi's displayed index. nvidia-smi emits the bus
    # id as fixed-width zero-padded hex like ``00000000:01:00.0``, so plain
    # lexicographic sort produces the same order as a structural domain /
    # bus / device / function comparison would.
    rows.sort(key=lambda item: item.get("pci_bus_id", ""))
    devices = []
    for idx, row in enumerate(rows):
        row = dict(row)
        row["id"] = idx
        devices.append(row)

    return {
        "available": True,
        "device_count": len(devices),
        "devices": devices,
        "detection_source": "nvidia-smi",
        "message": "Detected via nvidia-smi",
    }


def get_gpu_info_with_venv(venv_path=None):
    """Get GPU information, cascading through every available detector.

    Order:
      1. nvidia-smi (cheapest, no CUDA init).
      2. PyTorch inside the configured venv (skipped when no venv is set).
      3. PyTorch inside the current launcher process.

    Each step is tried even after a previous step said the backend was
    "unavailable" — the user may have nvidia-smi off PATH while still
    having torch in their venv, etc. If every step fails we return a
    combined message so the UI doesn't misleadingly show only the *last*
    failure (e.g. "PyTorch not found in venv" while nvidia-smi was also
    silently unavailable).
    """
    attempts: list[str] = []
    # Per-backend failure DEBUG prints carry raw nvidia-smi /
    # torch / venv exception text — gate behind the env so they
    # don't leak into journalctl / stderr on every probe. The
    # ``attempts`` list still feeds the user-facing combined
    # message (which is now generic, see final fallback below).
    _debug = os.environ.get("LLAMA_LAUNCHER_DEBUG_ENV") == "1"

    smi_info = get_gpu_info_from_nvidia_smi()
    if smi_info.get("available"):
        if _debug:
            print(
                f"DEBUG: nvidia-smi GPU detection successful: {smi_info.get('device_count', 0)} devices",
                file=sys.stderr,
            )
        return smi_info
    attempts.append(f"nvidia-smi: {smi_info.get('message', 'unknown error')}")
    if _debug:
        print(
            f"DEBUG: nvidia-smi GPU detection unavailable: {smi_info.get('message', 'unknown error')}", file=sys.stderr
        )

    # Call ``get_gpu_info_from_venv`` whenever a venv was configured, even
    # when the directory doesn't exist yet — the helper returns a
    # standardized "torch-venv / Python executable not found in venv"
    # marker that's worth surfacing in ``attempts``. Skipping the call
    # here used to silently hide a configured-but-missing venv from the
    # diagnostic message at the bottom of this function.
    if venv_path:
        venv_info = get_gpu_info_from_venv(venv_path)
        if venv_info.get("available"):
            if _debug:
                print(
                    f"DEBUG: venv PyTorch GPU detection successful: " f"{venv_info.get('device_count', 0)} devices",
                    file=sys.stderr,
                )
            return venv_info
        attempts.append(f"venv PyTorch: {venv_info.get('message', 'unknown error')}")
        if _debug:
            print(
                f"DEBUG: venv PyTorch GPU detection unavailable: {venv_info.get('message', 'unknown error')}",
                file=sys.stderr,
            )

    # Final fallback: current process. Useful when the configured venv has
    # no torch but the launcher's own interpreter does.
    static_info = get_gpu_info_static()
    if static_info.get("available"):
        if _debug:
            print(
                f"DEBUG: in-process PyTorch GPU detection successful: " f"{static_info.get('device_count', 0)} devices",
                file=sys.stderr,
            )
        return static_info
    attempts.append(f"in-process PyTorch: {static_info.get('message', 'unknown error')}")
    if _debug:
        print(
            f"DEBUG: in-process PyTorch GPU detection unavailable: {static_info.get('message', 'unknown error')}",
            file=sys.stderr,
        )

    # Every detector failed. The full per-backend stderr / exception
    # text stays in DEBUG-only stderr (already printed above for each
    # attempt + the combined line below); the UI-facing ``message``
    # is generic to avoid re-surfacing raw subprocess output / venv
    # paths in ``gpu_detected_status_var`` after the per-site
    # sanitization pass.
    if attempts and _debug:
        print(
            "DEBUG: GPU detection backends failed: " + "; ".join(attempts),
            file=sys.stderr,
        )
    static_info["message"] = "No GPU detection backend succeeded."
    return static_info


def get_gpu_info_from_venv(venv_path):
    """Get GPU information by running PyTorch detection in a virtual environment."""
    import subprocess
    import json
    from pathlib import Path

    venv_path = Path(venv_path)

    # Determine the Python executable in the venv
    if sys.platform == "win32":
        python_exe = venv_path / "Scripts" / "python.exe"
        if not python_exe.exists():
            python_exe = venv_path / "python.exe"  # Some venv structures
    else:
        python_exe = venv_path / "bin" / "python"
        if not python_exe.exists():
            python_exe = venv_path / "python"  # Some venv structures

    if not python_exe.exists():
        # Full venv path is path-bearing — gate behind
        # ``LLAMA_LAUNCHER_DEBUG_ENV=1`` so the home dir / username
        # doesn't end up in stderr / journalctl on every startup
        # against a misconfigured venv. The UI ``message`` already
        # uses generic copy.
        if os.environ.get("LLAMA_LAUNCHER_DEBUG_ENV") == "1":
            print(f"DEBUG: Python executable not found in venv: {venv_path}", file=sys.stderr)
        # Return an "unavailable" marker — ``get_gpu_info_with_venv`` is
        # the single owner of the in-process torch fallback. Returning a
        # second ``get_gpu_info_static()`` here would re-run the same slow
        # CUDA init the orchestrator is about to run anyway.
        return _unavailable_gpu_info(
            "Python executable not found in virtual environment.",
            "torch-venv",
        )

    # Create a small Python script to check for PyTorch/CUDA in the venv
    detection_script = """
import sys
import os
import json

# Ensure consistent GPU ordering by PCIe bus ID (matches nvidia-smi and llama.cpp)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

try:
    import torch
    torch_available = torch.cuda.is_available()
    
    if torch_available:
        device_count = torch.cuda.device_count()
        # Mirror ``get_gpu_info_static``: cuda.is_available() can be
        # True on a CUDA-built torch running against a host with no
        # visible devices (cloud VM with no devices attached, MIG
        # mode, CUDA_VISIBLE_DEVICES=""). The cache contract
        # ``load_cached_gpu_info`` enforces requires available=False
        # in that case; otherwise the parent's cascade short-circuits
        # on the empty result and never falls through to the next
        # backend.
        if device_count <= 0:
            print(json.dumps({"available": False, "message": "CUDA reports available but no devices were enumerated in venv", "device_count": 0, "devices": [], "detection_source": "torch-venv"}))
        else:
            gpu_info = {
                "available": True,
                "device_count": device_count,
                "devices": [],
                "detection_source": "torch-venv",
                "message": "Detected via torch in configured venv"
            }

            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                gpu_info["devices"].append({
                    "id": i,
                    "name": props.name,
                    "total_memory_bytes": props.total_memory,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": props.multi_processor_count
                })

            print(json.dumps(gpu_info))
    else:
        print(json.dumps({"available": False, "message": "CUDA not available via PyTorch in venv", "device_count": 0, "devices": [], "detection_source": "torch-venv"}))

except ImportError:
    print(json.dumps({"available": False, "message": "PyTorch not found in venv", "device_count": 0, "devices": [], "detection_source": "torch-venv"}))
except Exception as e:
    # Mirror the static-path sanitization: raw exception text can leak
    # venv paths, module internals, and home/username strings into the
    # UI status label. Gate the full repr behind LLAMA_LAUNCHER_DEBUG_ENV
    # and emit a generic message in the JSON payload that the parent
    # process consumes for the UI.
    if os.environ.get("LLAMA_LAUNCHER_DEBUG_ENV") == "1":
        print(
            f"DEBUG: venv GPU detection exception: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
    print(json.dumps({"available": False, "message": "GPU detection failed in configured virtual environment.", "device_count": 0, "devices": [], "detection_source": "torch-venv"}))
"""

    try:
        # Same reasoning as above — gate the path-bearing line behind
        # the debug env so normal-path stderr stays free of user
        # home/username text.
        if os.environ.get("LLAMA_LAUNCHER_DEBUG_ENV") == "1":
            print(f"DEBUG: Running GPU detection in venv: {venv_path}", file=sys.stderr)
        # Run the detection script in the virtual environment
        result = subprocess.run([str(python_exe), "-c", detection_script], capture_output=True, text=True, timeout=30)

        # Every error path here returns an ``_unavailable_gpu_info`` marker
        # for ``source="torch-venv"`` so the orchestrator
        # (``get_gpu_info_with_venv``) owns the single in-process torch
        # fallback. Calling ``get_gpu_info_static`` (via
        # ``_create_fallback_gpu_info``) here would cause the same slow
        # CUDA init to run twice on every venv-detection failure.
        if result.returncode == 0:
            try:
                output = result.stdout.strip()
                if not output:
                    if os.environ.get("LLAMA_LAUNCHER_DEBUG_ENV") == "1":
                        print("DEBUG: Venv GPU detection returned empty output", file=sys.stderr)
                    return _unavailable_gpu_info("Empty output from venv detection", "torch-venv")

                gpu_info = json.loads(output)
                # The detection script SHOULD emit a JSON object, but
                # a stale wrapper script / partial flush could leave
                # us with a list or scalar. ``gpu_info.get(...)``
                # would crash downstream — fall back to the next
                # detector instead by returning a clean unavailable
                # marker.
                if not isinstance(gpu_info, dict):
                    if os.environ.get("LLAMA_LAUNCHER_DEBUG_ENV") == "1":
                        print(
                            f"DEBUG: Venv GPU detection returned non-dict "
                            f"({type(gpu_info).__name__}); treating as failure",
                            file=sys.stderr,
                        )
                    return _unavailable_gpu_info(
                        "GPU detection returned malformed output.",
                        "torch-venv",
                    )
                if os.environ.get("LLAMA_LAUNCHER_DEBUG_ENV") == "1":
                    print(
                        f"DEBUG: Venv GPU detection successful: {gpu_info.get('device_count', 0)} devices",
                        file=sys.stderr,
                    )
                return gpu_info
            except json.JSONDecodeError as e:
                # Raw subprocess stdout / exception text gated behind
                # the debug env so it doesn't leak into the
                # tab status bar via ``attempts``. UI ``message`` is
                # generic.
                if os.environ.get("LLAMA_LAUNCHER_DEBUG_ENV") == "1":
                    print(f"DEBUG: Failed to parse venv GPU detection output: {e}", file=sys.stderr)
                    print(f"DEBUG: Raw output: '{result.stdout}'", file=sys.stderr)
                return _unavailable_gpu_info(
                    "Failed to parse GPU detection output from virtual environment.",
                    "torch-venv",
                )
        else:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            if os.environ.get("LLAMA_LAUNCHER_DEBUG_ENV") == "1":
                print(f"DEBUG: Venv GPU detection failed with return code {result.returncode}", file=sys.stderr)
                print(f"DEBUG: Error output: {error_msg}", file=sys.stderr)

            # Check for specific error types
            if "ModuleNotFoundError" in error_msg or "ImportError" in error_msg:
                return _unavailable_gpu_info("Required modules not found in venv", "torch-venv")
            elif "CUDA" in error_msg:
                return _unavailable_gpu_info("CUDA error in venv", "torch-venv")
            else:
                # UI message no longer contains raw subprocess
                # stderr — full text stays in the DEBUG print
                # above (gated by ``LLAMA_LAUNCHER_DEBUG_ENV``).
                return _unavailable_gpu_info(
                    "GPU detection failed in configured virtual environment.",
                    "torch-venv",
                )

    except subprocess.TimeoutExpired:
        if os.environ.get("LLAMA_LAUNCHER_DEBUG_ENV") == "1":
            print("DEBUG: Venv GPU detection timed out after 30 seconds", file=sys.stderr)
        return _unavailable_gpu_info("Detection timeout", "torch-venv")
    except FileNotFoundError:
        # Full python_exe path stays in DEBUG-only output (gated).
        if os.environ.get("LLAMA_LAUNCHER_DEBUG_ENV") == "1":
            print(f"DEBUG: Python executable not found: {python_exe}", file=sys.stderr)
        return _unavailable_gpu_info("Python executable not found", "torch-venv")
    except PermissionError:
        # Full venv_path stays in DEBUG-only output (gated).
        if os.environ.get("LLAMA_LAUNCHER_DEBUG_ENV") == "1":
            print(f"DEBUG: Permission denied accessing venv: {venv_path}", file=sys.stderr)
        return _unavailable_gpu_info("Permission denied", "torch-venv")
    except Exception as e:
        # Raw exception text can include venv paths / module names /
        # subprocess fragments. Gate behind the debug env to match
        # the other exception sites in this function.
        if os.environ.get("LLAMA_LAUNCHER_DEBUG_ENV") == "1":
            print(
                f"DEBUG: Unexpected exception during venv GPU detection: " f"{type(e).__name__}: {e}",
                file=sys.stderr,
            )
        return _unavailable_gpu_info(f"Unexpected error: {type(e).__name__}", "torch-venv")


def _create_fallback_gpu_info(reason):
    """Create fallback GPU info with specific reason, then try current process detection."""
    # Gate the diagnostic prints behind LLAMA_LAUNCHER_DEBUG_ENV.
    # ``reason`` may include venv paths or exception text from the
    # caller's failure path, and the function is called on every
    # detection cascade — emitting these unconditionally produces
    # noise on every startup against a broken/missing venv and
    # leaks user paths into stderr / journalctl.
    if os.environ.get("LLAMA_LAUNCHER_DEBUG_ENV") == "1":
        print(f"DEBUG: Creating fallback GPU info due to: {reason}", file=sys.stderr)
        print("DEBUG: Attempting current process GPU detection as fallback", file=sys.stderr)

    # Try current process detection as fallback
    fallback_info = get_gpu_info_static()

    # If current process detection also fails, return a clear error message
    if not fallback_info.get("available", False):
        fallback_info["message"] = f"Venv detection failed ({reason}), current process also failed"
    else:
        fallback_info["message"] = f"Using current process (venv failed: {reason})"

    return fallback_info


def get_gpu_info_static():
    """Get GPU information using PyTorch (static method)."""
    torch_module = _load_torch_module()
    if not torch_module:
        msg = "PyTorch not found."
        if _TORCH_IMPORT_ERROR is not None:
            msg = f"PyTorch import failed: {_TORCH_IMPORT_ERROR}"
        return _unavailable_gpu_info(msg, "torch")

    try:
        if not torch_module.cuda.is_available():
            return _unavailable_gpu_info("CUDA not available via PyTorch.", "torch")
    except Exception as exc:
        return _unavailable_gpu_info(f"CUDA availability check failed: {exc}", "torch")

    # ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` is already pinned at module import
    # (see lines 24-25) BEFORE any torch CUDA call, so re-assigning it
    # here after ``torch.cuda.is_available()`` is a no-op. Dropped to
    # avoid implying a runtime guarantee that this point only had.

    try:
        device_count = torch_module.cuda.device_count()
        # ``cuda.is_available()`` can return True on a torch build that
        # was compiled with CUDA but is running on a host with no
        # CUDA-capable devices visible (driver missing, MIG mode,
        # ``CUDA_VISIBLE_DEVICES=""``). In that case ``device_count == 0``
        # and the empty ``devices: []`` would otherwise be reported as
        # ``available: True``, which downstream UI treats as a usable GPU.
        if device_count <= 0:
            return _unavailable_gpu_info(
                "CUDA reports available but no devices were enumerated.",
                "torch",
            )
        gpu_info = {
            "available": True,
            "device_count": device_count,
            "devices": [],
            "detection_source": "torch",
            "message": "Detected via torch",
        }

        for i in range(device_count):
            props = torch_module.cuda.get_device_properties(i)
            # Getting free memory can be slow/problematic in some envs, skip for basic info
            # free_mem, total_mem = torch.cuda.mem_get_info(i)
            gpu_info["devices"].append(
                {
                    "id": i,
                    "name": props.name,
                    "total_memory_bytes": props.total_memory,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    # "free_memory_bytes": free_mem,
                    # "free_memory_gb": round(free_mem / (1024**3), 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": props.multi_processor_count,
                }
            )
        return gpu_info
    except Exception as e:
        # Raw exception text + traceback gated behind the debug env
        # so they don't leak into normal-path stderr / journalctl on
        # every startup against a broken torch install. The
        # user-facing ``message`` is intentionally generic — match
        # the rest of the sanitization pass.
        if os.environ.get("LLAMA_LAUNCHER_DEBUG_ENV") == "1":
            print(f"DEBUG: Error querying CUDA devices: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        return _unavailable_gpu_info("Error querying CUDA devices", "torch")


def format_gpu_mapping_table(gpu_info):
    """Format the authoritative GPU mapping as a human-readable table.

    The returned string is what the whole launcher treats as ground truth:
      * ``id`` column is the physical PCIe-bus-id index (since we force
        ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` at module load).
      * This same ``id`` is what the UI checkbox label shows, what
        ``app_settings['selected_gpus']`` / ``gpu_order`` store, and what
        the generated launch script emits as ``CUDA_VISIBLE_DEVICES``.

    Kept as a pure formatter (no printing) so tests can assert on the exact
    text and callers can decide whether to print, log, or display.
    """
    is_manual = bool(gpu_info.get("manual_mode"))
    devices = list(gpu_info.get("devices") or [])
    mode_label = "manual GPU mode" if is_manual else "auto-detected (PCI_BUS_ID order)"

    header = f"===== GPU mapping — {mode_label} ====="
    if is_manual:
        # Manual indices are synthetic — they're planning placeholders that
        # do NOT map to physical PCIe devices, so the launcher deliberately
        # does NOT emit them as CUDA_VISIBLE_DEVICES (it unsets instead).
        # The dump's footer must say so, or users will read the auto-mode
        # footer and think manual-mode selections filter real hardware.
        footer_msg = (
            "These indices are synthetic (for capacity planning / preview "
            "only). They DO NOT map to physical CUDA devices, so in manual "
            "mode the generated launch script emits 'unset CUDA_VISIBLE_DEVICES' "
            "rather than exporting these indices — otherwise the CUDA runtime "
            "would filter the wrong real GPUs."
        )
    else:
        footer_msg = (
            "These indices are the single source of truth across the UI "
            "(checkbox labels, drag-reorder list), the recommended tensor-split, "
            "and the generated CUDA_VISIBLE_DEVICES in launch scripts. "
            "--main-gpu and --tensor-split are applied AFTER the CUDA runtime "
            "remaps to this set, so they count from 0 against the selected "
            "(not physical) subset."
        )
    footer = "=" * len(header)

    if not devices:
        reason = gpu_info.get("message") or "no CUDA devices detected"
        return f"{header}\n  (no devices — {reason})\n{footer}\n{footer_msg}"

    # Build table rows. Keep the column widths tight so the output stays
    # readable even on narrow terminals.
    rows = [("IDX", "NAME", "VRAM", "COMPUTE")]
    for dev in devices:
        idx = dev.get("id")
        name = str(dev.get("name") or "?")
        vram_gb = dev.get("total_memory_gb")
        vram_str = f"{vram_gb:.1f} GB" if isinstance(vram_gb, (int, float)) else "?"
        cc = str(dev.get("compute_capability") or "?")
        rows.append((str(idx), name, vram_str, cc))

    col_widths = [max(len(row[i]) for row in rows) for i in range(4)]

    def _fmt(row):
        return "  " + " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))

    header_row = _fmt(rows[0])
    sep = "  " + "-+-".join("-" * w for w in col_widths)
    body = "\n".join(_fmt(r) for r in rows[1:])

    return "\n".join([header, header_row, sep, body, footer, footer_msg])


def log_gpu_mapping(gpu_info, stream=None):
    """Print the mapping table to ``stream`` (defaults to stderr).

    Separate from :func:`format_gpu_mapping_table` so tests can check the
    formatter output without parsing stderr."""
    text = format_gpu_mapping_table(gpu_info)
    if stream is None:
        stream = sys.stderr
    print(text, file=stream)


def get_ram_info_static():
    """Get system RAM information (static method)."""
    try:
        if sys.platform == "win32":
            try:
                # Correct way to use MEMORYSTATUSEX with ctypes on Windows
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                kernel32 = ctypes.windll.kernel32
                memoryInfo = MEMORYSTATUSEX()
                memoryInfo.dwLength = ctypes.sizeof(memoryInfo)

                if kernel32.GlobalMemoryStatusEx(ctypes.byref(memoryInfo)):
                    return {
                        "total_ram_bytes": memoryInfo.ullTotalPhys,
                        "total_ram_gb": round(memoryInfo.ullTotalPhys / (1024**3), 2),
                        "available_ram_bytes": memoryInfo.ullAvailPhys,
                        "available_ram_gb": round(memoryInfo.ullAvailPhys / (1024**3), 2),
                    }
                else:
                    # Fallback to psutil if ctypes call fails on Windows
                    if PSUTIL_AVAILABLE and psutil:
                        try:
                            mem = psutil.virtual_memory()
                            return {
                                "total_ram_bytes": mem.total,
                                "total_ram_gb": round(mem.total / (1024**3), 2),
                                "available_ram_bytes": mem.available,
                                "available_ram_gb": round(mem.available / (1024**3), 2),
                            }
                        except Exception as e_psutil_win:
                            print(f"Windows psutil RAM check failed: {e_psutil_win}", file=sys.stderr)
                            return {
                                "error": f"Windows RAM checks failed (ctypes: GlobalMemoryStatusEx failed, psutil: {e_psutil_win})"
                            }
                    else:
                        print("Windows ctypes GlobalMemoryStatusEx failed, psutil not available.", file=sys.stderr)
                        return {
                            "error": "Windows RAM check failed (ctypes: GlobalMemoryStatusEx failed, psutil not available)"
                        }

            except Exception as e_win:
                # Fallback if ctypes fails unexpectedly or psutil is available
                if PSUTIL_AVAILABLE and psutil:
                    try:
                        mem = psutil.virtual_memory()
                        return {
                            "total_ram_bytes": mem.total,
                            "total_ram_gb": round(mem.total / (1024**3), 2),
                            "available_ram_bytes": mem.available,
                            "available_ram_gb": round(mem.available / (1024**3), 2),
                        }
                    except Exception as e_psutil:
                        print(f"Windows psutil RAM check failed: {e_psutil}", file=sys.stderr)
                        return {"error": f"Windows RAM checks failed (ctypes: {e_win}, psutil: {e_psutil})"}

                else:
                    print(f"Windows RAM check failed (ctypes: {e_win}, psutil not available)", file=sys.stderr)
                    return {"error": f"Windows RAM check failed (ctypes: {e_win}, psutil not available)"}

        elif PSUTIL_AVAILABLE and psutil:  # Linux, macOS, etc. with psutil
            try:
                mem = psutil.virtual_memory()
                return {
                    "total_ram_bytes": mem.total,
                    "total_ram_gb": round(mem.total / (1024**3), 2),
                    "available_ram_bytes": mem.available,
                    "available_ram_gb": round(mem.available / (1024**3), 2),
                }
            except Exception as e_psutil:
                print(f"psutil RAM check failed: {e_psutil}", file=sys.stderr)
                return {"error": f"psutil RAM check failed: {e_psutil}"}

        else:
            return {"error": "psutil not installed, cannot get RAM info on this platform."}
    except Exception as e:
        print(f"Failed to get RAM info: {str(e)}", file=sys.stderr)
        return {"error": f"Failed to get RAM info: {str(e)}"}


def get_cpu_info_static():
    """Get system CPU information (static method)."""
    try:
        if PSUTIL_AVAILABLE and psutil:
            logical_cores = psutil.cpu_count(logical=True)
            physical_cores = psutil.cpu_count(logical=False)
            # Clamp both counts to at least 1. ``psutil`` can legitimately
            # return ``None`` (handled by the existing fallback) but in
            # rare containers it has been observed to return ``0`` for
            # ``physical_cores``, and ``logical_cores // 2`` on
            # ``logical_cores == 1`` produces ``0`` too. A zero count
            # would propagate into ``--threads 0`` on the launch
            # command line and the server would refuse to start.
            resolved_logical = logical_cores if logical_cores is not None else 4
            resolved_physical = (
                physical_cores if physical_cores is not None else (resolved_logical // 2 if resolved_logical > 0 else 2)
            )
            return {
                "logical_cores": max(1, resolved_logical),
                "physical_cores": max(1, resolved_physical),
                "model_name": "N/A",  # psutil doesn't easily give model name cross-platform
            }
        else:
            return {
                "error": "psutil not installed, cannot get CPU info.",
                "logical_cores": 4,
                "physical_cores": 2,
            }  # Default to sensible minimums
    except Exception as e:
        print(f"Failed to get CPU info: {str(e)}", file=sys.stderr)
        return {"error": f"Failed to get CPU info: {str(e)}", "logical_cores": 4, "physical_cores": 2}


# ── GPU detection cache (skip ~3-10s torch+CUDA init on every startup) ──
#
# On systems with many GPUs and a venv-based torch install, the first
# ``torch.cuda.is_available()`` + ``get_device_properties`` call in a
# fresh subprocess takes several seconds (CUDA context init). The result
# is deterministic across launches as long as the hardware and venv
# don't change, so we persist the JSON result to ``config/`` keyed by
# ``last_venv_dir`` + path-to-this-checkout and reuse it on the next
# startup. A background re-detection still runs to refresh the cache,
# so swapped hardware will eventually self-heal.

_GPU_CACHE_FILENAME = "gpu_detection_cache.json"


def _gpu_cache_path(config_dir):
    """Return the gpu-detection cache path. Accepts a Path or a str; if
    ``config_dir`` is falsy, returns None so callers degrade gracefully.
    """
    if not config_dir:
        return None
    return Path(config_dir) / _GPU_CACHE_FILENAME


def load_cached_gpu_info(config_dir, venv_path):
    """Try to read the persisted GPU-detection result.

    Returns a dict shaped like :func:`get_gpu_info_static` on hit, or
    ``None`` if there's no cache, the venv key doesn't match, or the
    file is unreadable/malformed. The cache key is the venv path (or
    the literal string ``"__none__"`` when the user has no venv set)
    because the same hardware reports different ``torch.cuda``
    properties depending on which torch build runs the detection
    (different driver versions / build flags / etc.).
    """
    cache_path = _gpu_cache_path(config_dir)
    if cache_path is None:
        return None
    try:
        raw = cache_path.read_text(encoding="utf-8")
    except (OSError, FileNotFoundError):
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    cached_venv = data.get("venv_path") or "__none__"
    # ``venv_path`` callers occasionally pass a ``Path`` (callers that
    # forwarded the resolved active venv directly). ``Path.strip()`` does
    # not exist, so coerce first.
    expected_venv = (str(venv_path) if venv_path is not None else "").strip() or "__none__"
    if cached_venv != expected_venv:
        return None
    gpu_info = data.get("gpu_info")
    if not isinstance(gpu_info, dict):
        return None
    # Minimal shape check so a corrupt-but-valid-JSON file doesn't crash
    # downstream UI code that reads ``available`` unguarded and assumes
    # ``device_count`` is int + ``devices`` is a list.
    if not isinstance(gpu_info.get("available"), bool):
        return None
    # ``bool`` is an ``int`` subclass in Python, so
    # ``isinstance(True, int)`` is True and a hand-edited
    # ``{"device_count": true}`` would otherwise count as a 1-GPU
    # cache hit. Exclude booleans the same way the per-device ``id``
    # check below does.
    device_count_value = gpu_info.get("device_count")
    if not isinstance(device_count_value, int) or isinstance(device_count_value, bool):
        return None
    devices = gpu_info.get("devices")
    if not isinstance(devices, list):
        return None
    # Reject contradictory cached payloads where ``available`` is False
    # but ``device_count`` / ``devices`` say otherwise. Without this,
    # ``fetch_system_info`` would happily copy the (non-empty) devices
    # list into ``detected_gpu_devices`` and resurrect phantom GPU rows
    # even though the cache itself reports the system as unavailable.
    if gpu_info.get("available") is False and (gpu_info.get("device_count", 0) != 0 or len(devices) != 0):
        return None
    # Symmetric guard: reject ``available=True`` with NO devices and
    # ``device_count=0``. The launcher treats this as a successful
    # detection and skips re-probing, so a corrupted cache like
    # ``{"available": true, "device_count": 0, "devices": []}`` would
    # otherwise mask a real GPU on the next launch until the user
    # manually flushed the cache.
    if gpu_info.get("available") is True and device_count_value == 0 and len(devices) == 0:
        return None
    # Reject caches where ``device_count`` and ``len(devices)`` disagree.
    # A truncated/edited cache where these don't match would resurrect
    # phantom GPU slots in the UI (``device_count`` drives row count;
    # ``devices`` is consulted per-row) instead of being ignored so a
    # fresh detection runs.
    if gpu_info.get("device_count") != len(devices):
        return None
    # Downstream callers like ``format_gpu_mapping_table`` do ``dev.get(...)``
    # on each entry, so a hand-edited cache with ``devices=["oops"]`` would
    # crash startup despite passing the top-level shape checks above.
    if any(not isinstance(device, dict) for device in devices):
        return None
    # Downstream code keys off ``gpu["id"]`` (launch.py, spec_launch.py,
    # gpu mapping UI) and assumes it's a real launcher index AND that
    # the index matches the device's position in the list. A
    # hand-edited cache with ``[{"id": 0}, {"id": 7}]`` would otherwise
    # load fine but silently drift the mapping (UI row 1 = launcher
    # GPU 7, etc). Reject anything that isn't the canonical 0..N-1
    # sequence so the next detection pass rebuilds it. ``bool`` is
    # intentionally excluded — ``True`` would otherwise pass as
    # ``id == 1``.
    for expected_id, device in enumerate(devices):
        device_id = device.get("id")
        if not isinstance(device_id, int) or isinstance(device_id, bool) or device_id != expected_id:
            return None
    return gpu_info


def save_cached_gpu_info(config_dir, venv_path, gpu_info):
    """Persist the latest GPU-detection result. Best-effort: I/O errors
    are logged to stderr but never raised — a cache write failure must
    not block the UI cascade that runs after detection completes.
    """
    cache_path = _gpu_cache_path(config_dir)
    if cache_path is None:
        return
    payload = {
        # Mirror the normalization in ``load_cached_gpu_info``: coerce
        # to str so a ``Path`` caller doesn't trip ``str.strip``.
        "venv_path": (str(venv_path) if venv_path is not None else "").strip() or "__none__",
        "gpu_info": gpu_info,
    }
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        # ``WARNING:`` not ``DEBUG:`` — this is an operational
        # cache-write failure (disk full, permission denied, …)
        # that runs unconditionally rather than gated behind
        # ``LLAMA_LAUNCHER_DEBUG_ENV``, so the prefix should
        # match its actual severity.
        print(f"WARNING: GPU detection cache write failed at {cache_path}: {exc}", file=sys.stderr)


def calculate_total_gguf_size(model_path_str):
    """Calculate total size across all GGUF shards if this is a multi-part file."""
    import re
    from pathlib import Path

    model_path = Path(model_path_str)

    # Check if this looks like a multi-part GGUF file (e.g., "00001-of-00003.gguf").
    # Capture the separator and extension literally so we can reproduce the
    # original filename case when looking up sibling shards on case-sensitive
    # filesystems (e.g., "-OF-" / ".GGUF" stays uppercase).
    shard_pattern = re.search(r"-(\d+)(-of-)(\d+)(\.gguf)$", model_path.name, re.IGNORECASE)
    if not shard_pattern:
        # Not a multi-part file, return single file size
        return model_path.stat().st_size, 1, [model_path]

    current_shard_text = shard_pattern.group(1)
    current_shard = int(current_shard_text)
    separator = shard_pattern.group(2)
    total_shards_text = shard_pattern.group(3)
    total_shards = int(total_shards_text)
    extension = shard_pattern.group(4)
    # Preserve the ORIGINAL zero-pad widths of the shard / total
    # tokens when reconstructing sibling filenames. ``\d+`` accepts
    # any width (e.g. ``-1-of-3.gguf``, ``-001-of-003.gguf``), but
    # the previous ``:05d`` lookup hard-coded 5 digits. A repo
    # using the 1-digit or 3-digit form would silently miss every
    # sibling shard and ``calculate_total_gguf_size`` would report
    # the partial single-file size + 1 shard, breaking the layer-
    # count / total-size heuristics downstream.
    current_width = len(current_shard_text)
    total_width = len(total_shards_text)

    print(f"DEBUG: Detected multi-part GGUF: shard {current_shard} of {total_shards}", file=sys.stderr)

    # Find all related shard files
    base_name = model_path.name[: shard_pattern.start()]  # Everything before "-00001-of-00003.gguf"
    parent_dir = model_path.parent

    total_size = 0
    found_shards = []
    missing_shards = []

    for shard_num in range(1, total_shards + 1):
        shard_name = (
            f"{base_name}-"
            f"{shard_num:0{current_width}d}"
            f"{separator}"
            f"{total_shards:0{total_width}d}"
            f"{extension}"
        )
        shard_path = parent_dir / shard_name

        if shard_path.exists():
            shard_size = shard_path.stat().st_size
            total_size += shard_size
            found_shards.append(shard_path)
            print(
                f"DEBUG: Found shard {shard_num}: {shard_path.name} ({shard_size / (1024**3):.2f} GB)", file=sys.stderr
            )
        else:
            missing_shards.append(shard_name)
            print(f"DEBUG: Missing shard {shard_num}: {shard_name}", file=sys.stderr)

    if missing_shards:
        print(f"WARNING: Missing {len(missing_shards)} shards: {missing_shards}", file=sys.stderr)
        # Return what we found, but note it's incomplete
        return total_size, len(found_shards), found_shards

    print(f"DEBUG: Total size across {total_shards} shards: {total_size / (1024**3):.2f} GB", file=sys.stderr)
    return total_size, total_shards, found_shards


def parse_gguf_header_simple(model_path_str):
    """Simple GGUF header parser to extract basic metadata without dependencies.

    Uses correct GGUF spec type mappings:
    - Type 0:  UINT8   (1 byte)
    - Type 1:  INT8    (1 byte)
    - Type 2:  UINT16  (2 bytes)
    - Type 3:  INT16   (2 bytes)
    - Type 4:  UINT32  (4 bytes)
    - Type 5:  INT32   (4 bytes)
    - Type 6:  FLOAT32 (4 bytes)
    - Type 7:  BOOL    (1 byte, stored as int8)
    - Type 8:  STRING  (variable length)
    - Type 9:  ARRAY   (variable length)
    - Type 10: UINT64  (8 bytes)
    - Type 11: INT64   (8 bytes)
    - Type 12: FLOAT64 (8 bytes)
    """
    from pathlib import Path

    model_path = Path(model_path_str)

    analysis_result = {
        "path": str(model_path),
        "file_size_bytes": 0,
        "file_size_gb": 0,
        "architecture": "unknown",
        "n_layers": None,
        "metadata": {},
        "error": None,
        "message": "Analyzed using simple GGUF parser",
        "shard_count": 0,
        "all_shards": [],
    }

    # Calculate total size across all shards if this is a multi-part file.
    # The shard discovery touches the filesystem and used to live OUTSIDE
    # the outer ``try`` below, so a stale/deleted model path raised
    # ``FileNotFoundError`` / ``OSError`` straight out of the function
    # instead of populating ``analysis_result["error"]``. Callers expect
    # the error-return contract, so catch OS-level failures here and
    # short-circuit with the same shape every other failure mode uses.
    try:
        total_size_bytes, shard_count, all_shards = calculate_total_gguf_size(model_path_str)
    except OSError as exc:
        analysis_result["error"] = f"Failed to inspect GGUF file: {exc}"
        return analysis_result

    analysis_result["file_size_bytes"] = total_size_bytes
    analysis_result["file_size_gb"] = round(total_size_bytes / (1024**3), 2)
    analysis_result["message"] = (
        f"Analyzed using simple GGUF parser ({shard_count} shard{'s' if shard_count != 1 else ''})"
    )
    analysis_result["shard_count"] = shard_count
    analysis_result["all_shards"] = [str(p) for p in all_shards]

    # GGUF type definitions per spec
    # Format: type_id -> (struct_format, size_in_bytes) or None for variable-length types
    GGUF_TYPES = {
        0: ("<B", 1),  # UINT8
        1: ("<b", 1),  # INT8
        2: ("<H", 2),  # UINT16
        3: ("<h", 2),  # INT16
        4: ("<I", 4),  # UINT32
        5: ("<i", 4),  # INT32
        6: ("<f", 4),  # FLOAT32
        7: ("<b", 1),  # BOOL (stored as int8)
        8: None,  # STRING (variable length)
        9: None,  # ARRAY (variable length)
        10: ("<Q", 8),  # UINT64
        11: ("<q", 8),  # INT64
        12: ("<d", 8),  # FLOAT64
    }

    # Soft cap for a single string value; values larger than this are still
    # consumed from the stream (so parsing continues) but not retained in the
    # result dict. A warning is emitted so the caller can see it happened.
    _STRING_SOFT_CAP = 1_000_000

    def read_value(f, value_type):
        """Read a value from file based on GGUF type."""
        if value_type == 8:  # STRING
            value_len_bytes = f.read(8)
            if len(value_len_bytes) < 8:
                return None
            value_len = struct.unpack("<Q", value_len_bytes)[0]
            # Check the soft cap BEFORE allocating, so a corrupted or
            # adversarial value_len claim (e.g. 5 GB) can't OOM the parser
            # on a legitimately large GGUF that actually has those bytes.
            # Skip past the bytes with seek() to keep the stream aligned.
            if value_len > _STRING_SOFT_CAP:
                # Validate that the file is actually long enough to hold
                # the claimed value; otherwise we have a truncated/malformed
                # file and the parse should bail.
                current_pos = f.tell()
                f.seek(0, os.SEEK_END)
                file_end = f.tell()
                if current_pos + value_len > file_end:
                    return None
                f.seek(current_pos + value_len)
                print(
                    f"WARNING: GGUF string value is {value_len} bytes "
                    f"(> {_STRING_SOFT_CAP} soft cap); value dropped from "
                    f"metadata but parse continues.",
                    file=sys.stderr,
                )
                return None
            value_bytes = f.read(value_len)
            if len(value_bytes) < value_len:
                return None
            return value_bytes.decode("utf-8", errors="replace")
        elif value_type == 7:  # BOOL
            bool_byte = f.read(1)
            if len(bool_byte) < 1:
                return None
            return struct.unpack("<b", bool_byte)[0] != 0
        elif value_type in GGUF_TYPES and GGUF_TYPES[value_type] is not None:
            fmt, size = GGUF_TYPES[value_type]
            value_bytes = f.read(size)
            if len(value_bytes) < size:
                return None
            return struct.unpack(fmt, value_bytes)[0]
        else:
            # Caller (see ~line 1341) is responsible for rejecting
            # ``value_type not in GGUF_TYPES`` BEFORE invoking
            # ``read_value`` and bailing out of the metadata loop —
            # this branch is unreachable in practice. Returning
            # ``None`` here used to silently desynchronise the parse
            # stream because the unknown value's bytes were never
            # consumed. The pre-call guard makes that impossible.
            return None

    # Hard cap on recursion for nested array-of-arrays metadata. GGUF files in
    # the wild never approach this; a malicious file claiming deep nesting is
    # rejected before it can trigger RecursionError on the Python stack.
    _MAX_NESTING_DEPTH = 16

    def skip_array(f, array_type, array_len, depth=0):
        """Skip over array data in the file.

        Returns True on success, False if EOF was hit, a malformed length
        was encountered, or nesting exceeded ``_MAX_NESTING_DEPTH`` (caller
        should bail). Nested arrays (type 9) are uncommon but legal per
        spec; we recursively walk them so the parse loop doesn't silently
        halt on files that include them.

        Every seek that uses an untrusted length (``str_len`` read from the
        file, or ``array_len * element_size``) is bounds-checked against
        the on-disk file size, so a malformed file claiming absurd lengths
        can't leave the stream position far past EOF and corrupt
        downstream reads. We use ``os.fstat`` for the size — one syscall,
        no seek, no save/restore dance.
        """
        if depth > _MAX_NESTING_DEPTH:
            print(
                f"WARNING: GGUF nested array exceeds max depth " f"{_MAX_NESTING_DEPTH}; aborting parse.",
                file=sys.stderr,
            )
            return False
        file_end = os.fstat(f.fileno()).st_size
        if array_type == 8:  # String array - must read each string length
            for _ in range(array_len):
                str_len_bytes = f.read(8)
                if len(str_len_bytes) < 8:
                    return False
                str_len = struct.unpack("<Q", str_len_bytes)[0]
                # Don't blindly seek past EOF — a corrupted/adversarial
                # str_len could send the stream so far past the end that a
                # subsequent read appears truncated but the loop mistakes
                # it for an earlier truncation. Validate first.
                current_pos = f.tell()
                if str_len < 0 or current_pos + str_len > file_end:
                    return False
                f.seek(current_pos + str_len)
            return True
        elif array_type == 9:  # Nested array - recurse so we don't desync
            for _ in range(array_len):
                inner_type_bytes = f.read(4)
                inner_len_bytes = f.read(8)
                if len(inner_type_bytes) < 4 or len(inner_len_bytes) < 8:
                    return False
                inner_type = struct.unpack("<I", inner_type_bytes)[0]
                inner_len = struct.unpack("<Q", inner_len_bytes)[0]
                if not skip_array(f, inner_type, inner_len, depth + 1):
                    return False
            return True
        elif array_type in GGUF_TYPES and GGUF_TYPES[array_type] is not None:
            _, element_size = GGUF_TYPES[array_type]
            # Same bounds concern as above: ``array_len`` is an untrusted
            # 64-bit value, so ``array_len * element_size`` could be huge.
            current_pos = f.tell()
            total_bytes = array_len * element_size
            if array_len < 0 or current_pos + total_bytes > file_end:
                return False
            f.seek(current_pos + total_bytes)
            return True
        else:
            print(
                f"WARNING: GGUF array has unknown element type {array_type}; " f"cannot skip safely, aborting parse.",
                file=sys.stderr,
            )
            return False

    try:
        with open(model_path, "rb") as f:
            # Read GGUF magic number
            magic = f.read(4)
            if magic != b"GGUF":
                # The spec allowed a short-lived big-endian variant ('FUGG'
                # when read little-endian). Detect and reject it with a
                # clear error rather than silently misparsing.
                if magic == b"FUGG":
                    analysis_result["error"] = "Big-endian GGUF files are not supported by this parser"
                else:
                    analysis_result["error"] = "Not a valid GGUF file"
                return analysis_result

            # Read version
            version_bytes = f.read(4)
            if len(version_bytes) < 4:
                analysis_result["error"] = "Failed to parse GGUF header: truncated version field"
                return analysis_result
            version = struct.unpack("<I", version_bytes)[0]

            # Only GGUF v2 and v3 are widely used; v1 had a different layout
            # and anything beyond v3 is unknown to this parser. Flag it
            # clearly instead of silently producing garbage output.
            if version not in (2, 3):
                analysis_result["error"] = f"Unsupported GGUF version {version}; expected 2 or 3"
                return analysis_result

            # Read tensor count and metadata count
            tc_bytes = f.read(8)
            mc_bytes = f.read(8)
            if len(tc_bytes) < 8 or len(mc_bytes) < 8:
                analysis_result["error"] = "Failed to parse GGUF header: truncated counts"
                return analysis_result
            tensor_count = struct.unpack("<Q", tc_bytes)[0]
            metadata_count = struct.unpack("<Q", mc_bytes)[0]

            # Track what we've found for early exit
            found_architecture = False
            found_block_count = False
            architecture_value = None

            # Read metadata key-value pairs
            for _ in range(metadata_count):
                try:
                    # Read key length and key
                    key_len_bytes = f.read(8)
                    if len(key_len_bytes) < 8:
                        break
                    key_len = struct.unpack("<Q", key_len_bytes)[0]
                    if key_len > 1000:  # Key names should be reasonable length
                        break
                    key_bytes = f.read(key_len)
                    if len(key_bytes) < key_len:
                        break
                    key = key_bytes.decode("utf-8", errors="replace")

                    # Read value type
                    type_bytes = f.read(4)
                    if len(type_bytes) < 4:
                        break
                    value_type = struct.unpack("<I", type_bytes)[0]

                    # Handle arrays specially (type 9)
                    if value_type == 9:  # ARRAY
                        array_type_bytes = f.read(4)
                        array_len_bytes = f.read(8)
                        if len(array_type_bytes) < 4 or len(array_len_bytes) < 8:
                            break
                        array_type = struct.unpack("<I", array_type_bytes)[0]
                        array_len = struct.unpack("<Q", array_len_bytes)[0]

                        # Skip the array data (no limit on array size - tokenizers can have 150k+ elements)
                        if not skip_array(f, array_type, array_len):
                            print(
                                f"DEBUG: Failed to skip array for key '{key}': type {array_type}, length {array_len}",
                                file=sys.stderr,
                            )
                            break
                        continue

                    # Fail closed on unknown scalar metadata types
                    # BEFORE invoking ``read_value`` — the function
                    # consumes zero bytes for unrecognised types, so
                    # a silent ``continue`` here would let the next
                    # iteration read the unknown payload as the next
                    # key length and the rest of the metadata parse
                    # would desynchronise into garbage keys. Surface
                    # an error and break out of the loop so the
                    # caller can decide whether to fall back to
                    # heuristic layer-count detection or report the
                    # parse failure. Array type 9 is handled above.
                    if value_type not in GGUF_TYPES:
                        analysis_result["error"] = (
                            f"Unsupported GGUF metadata value_type {value_type!r} for key '{key}'"
                        )
                        break

                    # Read the value
                    value = read_value(f, value_type)

                    if value is not None:
                        analysis_result["metadata"][key] = value

                        # Extract key information
                        if key == "general.architecture":
                            analysis_result["architecture"] = str(value)
                            architecture_value = str(value)
                            found_architecture = True
                            print(f"DEBUG: Found architecture: {value}", file=sys.stderr)

                        # Check for block_count with architecture prefix
                        if architecture_value and key == f"{architecture_value}.block_count":
                            if isinstance(value, (int, float)) and value > 0:
                                analysis_result["n_layers"] = int(value)
                                found_block_count = True
                                print(
                                    f"DEBUG: Found layer count in key '{key}': {analysis_result['n_layers']}",
                                    file=sys.stderr,
                                )

                        # Also check for generic block_count patterns
                        elif any(
                            pattern in key.lower()
                            for pattern in [".block_count", ".n_layers", ".layer_count", ".num_layer"]
                        ):
                            if isinstance(value, (int, float)) and value > 0:
                                analysis_result["n_layers"] = int(value)
                                found_block_count = True
                                print(
                                    f"DEBUG: Found layer count in key '{key}': {analysis_result['n_layers']}",
                                    file=sys.stderr,
                                )

                    # Early exit if we have both architecture and block_count
                    if found_architecture and found_block_count:
                        print(f"DEBUG: Found both architecture and block_count, exiting early", file=sys.stderr)
                        break

                except Exception as parse_error:
                    # Skip this metadata entry if parsing fails
                    print(f"DEBUG: Failed to parse metadata entry: {parse_error}", file=sys.stderr)
                    continue

            # If we still don't have layer count, make a reasonable guess based on file size
            if analysis_result["n_layers"] is None:
                file_size_gb = analysis_result["file_size_gb"]
                if file_size_gb > 100:  # Very large model (100+ GB)
                    estimated_layers = 120  # Conservative estimate for huge models
                    print(
                        f"DEBUG: No layer count found, estimating {estimated_layers} layers based on very large file size ({file_size_gb:.1f} GB)",
                        file=sys.stderr,
                    )
                elif file_size_gb > 50:  # Large model (50-100 GB)
                    estimated_layers = 80
                    print(
                        f"DEBUG: No layer count found, estimating {estimated_layers} layers based on large file size ({file_size_gb:.1f} GB)",
                        file=sys.stderr,
                    )
                elif file_size_gb > 20:  # Medium-large model (20-50 GB)
                    estimated_layers = 60
                    print(
                        f"DEBUG: No layer count found, estimating {estimated_layers} layers based on medium-large file size ({file_size_gb:.1f} GB)",
                        file=sys.stderr,
                    )
                elif file_size_gb > 10:  # Medium model (10-20 GB)
                    estimated_layers = 40
                    print(
                        f"DEBUG: No layer count found, estimating {estimated_layers} layers based on medium file size ({file_size_gb:.1f} GB)",
                        file=sys.stderr,
                    )
                elif file_size_gb > 3:  # Small-medium model (3-10 GB)
                    estimated_layers = 32
                    print(
                        f"DEBUG: No layer count found, estimating {estimated_layers} layers based on small-medium file size ({file_size_gb:.1f} GB)",
                        file=sys.stderr,
                    )
                else:  # Small model (< 3 GB)
                    estimated_layers = 24
                    print(
                        f"DEBUG: No layer count found, estimating {estimated_layers} layers based on small file size ({file_size_gb:.1f} GB)",
                        file=sys.stderr,
                    )

                analysis_result["n_layers"] = estimated_layers
                analysis_result["message"] += f" (estimated {estimated_layers} layers from file size)"

            return analysis_result

    except Exception as e:
        analysis_result["error"] = f"Failed to parse GGUF header: {e}"
        return analysis_result


class SystemInfoManager:
    """Manages system information fetching and processing for the launcher."""

    def __init__(self, launcher_instance):
        """Initialize with reference to the main launcher instance."""
        self.launcher = launcher_instance

    def fetch_system_info(self, venv_path=None, defer_tk_writes=False):
        """Fetches GPU, RAM, and CPU info and populates class attributes.

        ``venv_path`` MUST be passed in (or left None) by the caller —
        it must NEVER be re-read from ``self.launcher.venv_dir`` here.
        The worker thread that calls this method has no safe way to
        touch a ``tk.StringVar``: Tcl is single-threaded, so a
        cross-thread ``.get()`` serializes through the Tcl interpreter
        lock and blocks until the Tk main loop is idle. With a busy
        startup (model-list population, GGUF analysis dispatch, etc.)
        that block can easily stretch to tens of seconds — observed
        in practice as a 33-second gap between worker start and the
        actual GPU probe firing. The launcher's
        ``_start_system_info_detection`` already captures the venv path
        on the main thread before forking, so the value is in hand by
        the time we get here.

        When ``defer_tk_writes=True`` the method skips the Tk ``.set()``
        calls entirely — the worker thread populates only plain-Python
        attributes (``gpu_info``, ``logical_cores``, etc.) and the main
        thread's completion callback (``_apply_system_info_to_tk_vars``)
        is responsible for writing the Tk vars after a destroyed-root
        guard. Bypassing the Tcl interpreter from the worker entirely
        is the only reliable way to avoid the Python 3.13 deadlock —
        even a guarded ``.set()`` acquires the Tcl mutex and competes
        with the next root's UI calls.
        """
        print("Fetching system info...", file=sys.stderr)

        self.launcher.gpu_info = get_gpu_info_with_venv(venv_path)
        self.launcher.ram_info = get_ram_info_static()
        self.launcher.cpu_info = get_cpu_info_static()  # Fetch CPU info here

        print(f"GPU Info: {self.launcher.gpu_info}", file=sys.stderr)
        if not self.launcher.gpu_info["available"] and "message" in self.launcher.gpu_info:
            print(f"GPU Detection Info: {self.launcher.gpu_info['message']}", file=sys.stderr)
        if "error" in self.launcher.ram_info:
            print(f"RAM Detection Error: {self.launcher.ram_info['error']}", file=sys.stderr)
        if "error" in self.launcher.cpu_info:
            print(f"CPU Detection Error: {self.launcher.cpu_info['error']}", file=sys.stderr)

        # Store detected devices separately for easier access
        self.launcher.detected_gpu_devices = self.launcher.gpu_info.get("devices", [])

        # One-time authoritative mapping dump. After this line, every consumer
        # (UI checkbox labels, drag-reorder list, tensor-split recommendation,
        # CUDA_VISIBLE_DEVICES emission) uses the indices shown here — so when
        # a user reports a mapping surprise, this is the table to cross-check.
        log_gpu_mapping(self.launcher.gpu_info)
        # Store logical/physical cores for initial thread defaults and recommendations
        self.launcher.logical_cores = self.launcher.cpu_info.get("logical_cores", 4)
        self.launcher.physical_cores = self.launcher.cpu_info.get(
            "physical_cores", 2
        )  # Use fallback 2 if psutil failed or physical count is 0

        if defer_tk_writes:
            # Worker path: do NOT touch any Tk var here. The main-thread
            # completion callback will read the populated attributes and
            # apply them via ``_apply_system_info_to_tk_vars``.
            return

        # Main-thread fallback (back-compat for any synchronous caller).
        # Update initial default values for threads and threads_batch.
        physical = self.launcher.physical_cores
        logical = self.launcher.logical_cores
        gpu_msg = ""
        if not self.launcher.gpu_info["available"] and self.launcher.gpu_info.get("message"):
            gpu_msg = self.launcher.gpu_info["message"]
        for var_name, value in (
            ("threads", str(physical)),
            ("threads_batch", str(logical)),
            ("recommended_threads_var", f"Recommended: {physical} (Your CPU physical cores)"),
            ("recommended_threads_batch_var", f"Recommended: {logical} (Your CPU logical cores)"),
            ("gpu_detected_status_var", gpu_msg),
        ):
            try:
                getattr(self.launcher, var_name).set(value)
            except Exception:  # noqa: BLE001 - includes tk.TclError, RuntimeError
                pass
