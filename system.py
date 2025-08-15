#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
import traceback
import ctypes
import struct
from pathlib import Path
from gguf import GGUFReader, GGUFValueType

# Add debug prints for Python environment
print("\n=== Python Environment Debug Info ===", file=sys.stderr)
print(f"Python executable: {sys.executable}", file=sys.stderr)
print(f"Python version: {sys.version}", file=sys.stderr)
print(f"sys.path: {sys.path}", file=sys.stderr)
print("===================================\n", file=sys.stderr)

# --- New Imports ---
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    torch = None # Define torch as None if import fails
except Exception as e:
    # Catch other potential issues during torch import (e.g., missing CUDA drivers)
    TORCH_AVAILABLE = False
    torch = None
    print(f"Warning: PyTorch import failed: {e}", file=sys.stderr)

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


# --- Dependency Check (Printed to console/stderr) ---
MISSING_DEPS = []

# Required dependencies
if not REQUESTS_AVAILABLE:
    MISSING_DEPS.append("requests (required for version checking and updates)")

# PyTorch is the primary requirement for GPU detection and features
if not TORCH_AVAILABLE:
    MISSING_DEPS.append("PyTorch (required for GPU detection and CUDA features)")

if not PSUTIL_AVAILABLE:
    MISSING_DEPS.append("psutil (optional - provides enhanced system information)")

# Only print missing deps warning if there are actually missing deps
if MISSING_DEPS:
    print("\n--- Missing Dependencies Warning ---")
    print("The following Python libraries are recommended for full functionality but were not found:")
    for dep in MISSING_DEPS:
        print(f" - {dep}")
    print("Please install them using 'pip install -r requirements.txt' or individually with pip.")
    print("Note: GGUF analysis works without llama-cpp-python using built-in tools or simple header parsing.")
    print("-------------------------------------\n")


# ═════════════════════════════════════════════════════════════════════
#  Helper Functions (These remain outside the class as they don't need 'self')
# ═════════════════════════════════════════════════════════════════════
def get_file_host_endian(reader: GGUFReader) -> tuple[str, str]:
    file_endian = reader.endianess.name
    if reader.byte_order == 'S':
        host_endian = 'BIG' if file_endian == 'LITTLE' else 'LITTLE'
    else:
        host_endian = file_endian
    return (host_endian, file_endian)

def get_gpu_info_with_venv(venv_path=None):
    """Get GPU information using PyTorch, optionally from a virtual environment."""
    if venv_path and Path(venv_path).exists():
        # Use the virtual environment to check for PyTorch/CUDA
        return get_gpu_info_from_venv(venv_path)
    else:
        # Use the current process (original behavior)
        return get_gpu_info_static()

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
        print(f"DEBUG: Python executable not found in venv: {venv_path}", file=sys.stderr)
        # Fall back to current process detection
        return get_gpu_info_static()

    # Create a small Python script to check for PyTorch/CUDA in the venv
    detection_script = '''
import sys
import json

try:
    import torch
    torch_available = torch.cuda.is_available()

    if torch_available:
        device_count = torch.cuda.device_count()
        gpu_info = {
            "available": True,
            "device_count": device_count,
            "devices": []
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
        print(json.dumps({"available": False, "message": "CUDA not available via PyTorch in venv", "device_count": 0, "devices": []}))

except ImportError:
    print(json.dumps({"available": False, "message": "PyTorch not found in venv", "device_count": 0, "devices": []}))
except Exception as e:
    print(json.dumps({"available": False, "message": f"Error in venv GPU detection: {e}", "device_count": 0, "devices": []}))
'''

    try:
        print(f"DEBUG: Running GPU detection in venv: {venv_path}", file=sys.stderr)
        # Run the detection script in the virtual environment
        result = subprocess.run(
            [str(python_exe), "-c", detection_script],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            try:
                output = result.stdout.strip()
                if not output:
                    print("DEBUG: Venv GPU detection returned empty output", file=sys.stderr)
                    return _create_fallback_gpu_info("Empty output from venv detection")

                gpu_info = json.loads(output)
                print(f"DEBUG: Venv GPU detection successful: {gpu_info.get('device_count', 0)} devices", file=sys.stderr)
                return gpu_info
            except json.JSONDecodeError as e:
                print(f"DEBUG: Failed to parse venv GPU detection output: {e}", file=sys.stderr)
                print(f"DEBUG: Raw output: '{result.stdout}'", file=sys.stderr)
                return _create_fallback_gpu_info(f"JSON parse error: {e}")
        else:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            print(f"DEBUG: Venv GPU detection failed with return code {result.returncode}", file=sys.stderr)
            print(f"DEBUG: Error output: {error_msg}", file=sys.stderr)

            # Check for specific error types
            if "ModuleNotFoundError" in error_msg or "ImportError" in error_msg:
                return _create_fallback_gpu_info("Required modules not found in venv")
            elif "CUDA" in error_msg:
                return _create_fallback_gpu_info("CUDA error in venv")
            else:
                return _create_fallback_gpu_info(f"Venv detection failed: {error_msg}")

    except subprocess.TimeoutExpired:
        print("DEBUG: Venv GPU detection timed out after 30 seconds", file=sys.stderr)
        return _create_fallback_gpu_info("Detection timeout")
    except FileNotFoundError:
        print(f"DEBUG: Python executable not found: {python_exe}", file=sys.stderr)
        return _create_fallback_gpu_info("Python executable not found")
    except PermissionError:
        print(f"DEBUG: Permission denied accessing venv: {venv_path}", file=sys.stderr)
        return _create_fallback_gpu_info("Permission denied")
    except Exception as e:
        print(f"DEBUG: Unexpected exception during venv GPU detection: {type(e).__name__}: {e}", file=sys.stderr)
        return _create_fallback_gpu_info(f"Unexpected error: {type(e).__name__}")

def _create_fallback_gpu_info(reason):
    """Create fallback GPU info with specific reason, then try current process detection."""
    print(f"DEBUG: Creating fallback GPU info due to: {reason}", file=sys.stderr)
    print("DEBUG: Attempting current process GPU detection as fallback", file=sys.stderr)

    # Try current process detection as fallback
    fallback_info = get_gpu_info_static()

    # If current process detection also fails, return a clear error message
    if not fallback_info.get('available', False):
        fallback_info['message'] = f"Venv detection failed ({reason}), current process also failed"
    else:
        fallback_info['message'] = f"Using current process (venv failed: {reason})"

    return fallback_info

def get_gpu_info_static():
    """Get GPU information using PyTorch (static method)."""
    if not torch or not TORCH_AVAILABLE:
        msg = "PyTorch not found." if not torch else "CUDA not available via PyTorch."
        return {"available": False, "message": msg, "device_count": 0, "devices": []}

    try:
        device_count = torch.cuda.device_count()
        gpu_info = {
            "available": True,
            "device_count": device_count,
            "devices": []
        }

        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            # Getting free memory can be slow/problematic in some envs, skip for basic info
            # free_mem, total_mem = torch.cuda.mem_get_info(i)
            gpu_info["devices"].append({
                "id": i,
                "name": props.name,
                "total_memory_bytes": props.total_memory,
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
                # "free_memory_bytes": free_mem,
                # "free_memory_gb": round(free_mem / (1024**3), 2),
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count
            })
        return gpu_info
    except Exception as e:
        # Catch potential torch errors during device query
        print(f"Error querying CUDA devices: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return {"available": False, "message": f"Error querying CUDA devices: {e}", "device_count": 0, "devices": []}


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
                        "available_ram_gb": round(memoryInfo.ullAvailPhys / (1024**3), 2)
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
                                "available_ram_gb": round(mem.available / (1024**3), 2)
                              }
                         except Exception as e_psutil_win:
                              print(f"Windows psutil RAM check failed: {e_psutil_win}", file=sys.stderr)
                              return {"error": f"Windows RAM checks failed (ctypes: GlobalMemoryStatusEx failed, psutil: {e_psutil_win})"}
                    else:
                         print("Windows ctypes GlobalMemoryStatusEx failed, psutil not available.", file=sys.stderr)
                         return {"error": "Windows RAM check failed (ctypes: GlobalMemoryStatusEx failed, psutil not available)"}


            except Exception as e_win:
                 # Fallback if ctypes fails unexpectedly or psutil is available
                 if PSUTIL_AVAILABLE and psutil:
                      try:
                           mem = psutil.virtual_memory()
                           return {
                             "total_ram_bytes": mem.total,
                             "total_ram_gb": round(mem.total / (1024**3), 2),
                             "available_ram_bytes": mem.available,
                             "available_ram_gb": round(mem.available / (1024**3), 2)
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
                    "available_ram_gb": round(mem.available / (1024**3), 2)
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
            return {
                "logical_cores": logical_cores if logical_cores is not None else 4, # Default to 4 if psutil somehow returns None
                "physical_cores": physical_cores if physical_cores is not None else (logical_cores // 2 if logical_cores is not None and logical_cores > 0 else 2), # Estimate physical if needed
                "model_name": "N/A" # psutil doesn't easily give model name cross-platform
            }
        else:
             return {"error": "psutil not installed, cannot get CPU info.", "logical_cores": 4, "physical_cores": 2} # Default to sensible minimums
    except Exception as e:
        print(f"Failed to get CPU info: {str(e)}", file=sys.stderr)
        return {"error": f"Failed to get CPU info: {str(e)}", "logical_cores": 4, "physical_cores": 2}

def calculate_total_gguf_size(model_path_str):
    """Calculate total size across all GGUF shards if this is a multi-part file."""
    import re
    from pathlib import Path

    model_path = Path(model_path_str)

    # Check if this looks like a multi-part GGUF file (e.g., "00001-of-00003.gguf")
    shard_pattern = re.search(r'-(\d+)-of-(\d+)\.gguf$', model_path.name, re.IGNORECASE)
    if not shard_pattern:
        # Not a multi-part file, return single file size
        return model_path.stat().st_size, 1, [model_path]

    current_shard = int(shard_pattern.group(1))
    total_shards = int(shard_pattern.group(2))

    print(f"DEBUG: Detected multi-part GGUF: shard {current_shard} of {total_shards}", file=sys.stderr)

    # Find all related shard files
    base_name = model_path.name[:shard_pattern.start()]  # Everything before "-00001-of-00003.gguf"
    parent_dir = model_path.parent

    total_size = 0
    found_shards = []
    missing_shards = []

    for shard_num in range(1, total_shards + 1):
        shard_name = f"{base_name}-{shard_num:05d}-of-{total_shards:05d}.gguf"
        shard_path = parent_dir / shard_name

        if shard_path.exists():
            shard_size = shard_path.stat().st_size
            total_size += shard_size
            found_shards.append(shard_path)
            print(f"DEBUG: Found shard {shard_num}: {shard_path.name} ({shard_size / (1024**3):.2f} GB)", file=sys.stderr)
        else:
            missing_shards.append(shard_name)
            print(f"DEBUG: Missing shard {shard_num}: {shard_name}", file=sys.stderr)

    if missing_shards:
        print(f"WARNING: Missing {len(missing_shards)} shards: {missing_shards}", file=sys.stderr)
        # Return what we found, but note it's incomplete
        return total_size, len(found_shards), found_shards

    print(f"DEBUG: Total size across {total_shards} shards: {total_size / (1024**3):.2f} GB", file=sys.stderr)
    return total_size, total_shards, found_shards

def dump_gguf_metadata(model_path_str, json_array = False, no_tensors = False):
    """
    refer https://github.com/ggml-org/llama.cpp/blob/main/gguf-py/gguf/scripts/gguf_dump.py
    json_array: Include full array values in JSON output (long)
    no-tensors: Don't dump tensor metadata
    """

    reader = GGUFReader(model_path_str, 'r')
    host_endian, file_endian = get_file_host_endian(reader)
    metadata: dict[str, Any] = {}
    tensors: dict[str, Any] = {}
    result = {
        "filename": Path(model_path_str).stem,
        "endian": file_endian,
        "metadata": metadata,
        "tensors": tensors,
    }
    for idx, field in enumerate(reader.fields.values()):
        curr: dict[str, Any] = {
            "index": idx,
            "type": field.types[0].name if field.types else 'UNKNOWN',
            "offset": field.offset,
        }
        metadata[field.name] = curr
        if field.types[:1] == [GGUFValueType.ARRAY]:
            curr["array_types"] = [t.name for t in field.types][1:]
            if not json_array:
                continue
            curr["value"] = field.contents()
        else:
            curr["value"] = field.contents()
    if not no_tensors:
        for idx, tensor in enumerate(reader.tensors):
            tensors[tensor.name] = {
                "index": idx,
                "shape": tensor.shape.tolist(),
                "type": tensor.tensor_type.name,
                "offset": tensor.field.offset,
            }
    return result

def analyze_gguf_model_static(model_path_str):
    """Analyze a GGUF model file and extract metadata (static method)."""
    model_path = Path(model_path_str)
    if not model_path.is_file():
         return {"error": f"Model file not found: {model_path}", "n_layers": None, "architecture": "N/A", "file_size_bytes": 0}

    # Calculate total size across all shards if this is a multi-part file
    total_size_bytes, shard_count, all_shards = calculate_total_gguf_size(model_path_str)

    analysis_result = {
        "path": str(model_path),
        "file_size_bytes": total_size_bytes,
        "file_size_gb": round(total_size_bytes / (1024**3), 2),
        "architecture": "unknown",
        "n_layers": None,
        "expert_count": 0,
        "expert_used_count": 0,
        "context_length": 0,
        "metadata": {},
        "error": None,
        "message": None,
        "shard_count": shard_count,
        "all_shards": [str(p) for p in all_shards]
    }

    llm_meta = {} # Initialize llm_meta outside the try block

    try:
        llm_meta = dump_gguf_metadata(model_path_str, no_tensors = True)["metadata"]
    except Exception as load_exc:
        analysis_result["error"] = f"Failed to load model for metadata analysis: {load_exc}"
        print(f"ERROR: Failed to load model '{model_path.name}' for analysis: {load_exc}", file=sys.stderr)
        # No traceback here, the caller should handle it or the log message is enough
        return analysis_result # Exit early if basic load fails

    # --- Extract Metadata ---
    # Attempt 1: Check common metadata keys and attributes
    for k,v in llm_meta.items():
        if ".block_count" in k: analysis_result["n_layers"] = llm_meta.get(k)["value"]
        if ".architecture" in k: analysis_result["architecture"] = llm_meta.get(k)["value"]
        if ".context_length" in k: analysis_result["context_length"] = llm_meta.get(k)["value"]
        if ".expert_count" in k: analysis_result["expert_count"] = llm_meta.get(k)["value"]
        if ".expert_used_count" in k: analysis_result["expert_used_count"] = llm_meta.get(k)["value"]

    return analysis_result

class SystemInfoManager:
    """Manages system information fetching and processing for the launcher."""

    def __init__(self, launcher_instance):
        """Initialize with reference to the main launcher instance."""
        self.launcher = launcher_instance

    def fetch_system_info(self):
        """Fetches GPU, RAM, and CPU info and populates class attributes."""
        print("Fetching system info...", file=sys.stderr)

        # Get the configured virtual environment path from the launcher
        venv_path = None
        if hasattr(self.launcher, 'venv_dir'):
            venv_path_str = self.launcher.venv_dir.get().strip()
            if venv_path_str:
                venv_path = venv_path_str
                print(f"DEBUG: Using configured venv for GPU detection: {venv_path}", file=sys.stderr)
            else:
                print("DEBUG: No venv configured, using current process for GPU detection", file=sys.stderr)

        self.launcher.gpu_info = get_gpu_info_with_venv(venv_path)
        self.launcher.ram_info = get_ram_info_static()
        self.launcher.cpu_info = get_cpu_info_static() # Fetch CPU info here

        print(f"GPU Info: {self.launcher.gpu_info}", file=sys.stderr)
        if not self.launcher.gpu_info["available"] and "message" in self.launcher.gpu_info:
             print(f"GPU Detection Info: {self.launcher.gpu_info['message']}", file=sys.stderr)
        if "error" in self.launcher.ram_info:
             print(f"RAM Detection Error: {self.launcher.ram_info['error']}", file=sys.stderr)
        if "error" in self.launcher.cpu_info:
             print(f"CPU Detection Error: {self.launcher.cpu_info['error']}", file=sys.stderr)

        # Store detected devices separately for easier access
        self.launcher.detected_gpu_devices = self.launcher.gpu_info.get("devices", [])
        # Store logical/physical cores for initial thread defaults and recommendations
        self.launcher.logical_cores = self.launcher.cpu_info.get("logical_cores", 4)
        self.launcher.physical_cores = self.launcher.cpu_info.get("physical_cores", 2) # Use fallback 2 if psutil failed or physical count is 0

        # Update initial default values for threads and threads_batch if they are still the initial fallback values
        # This handles the case where system info is fetched successfully *after* the variables were initialized
        # Also update the recommended variables immediately after fetching
        self.launcher.threads.set(str(self.launcher.physical_cores))
        self.launcher.threads_batch.set(str(self.launcher.logical_cores))
        self.launcher.recommended_threads_var.set(f"Recommended: {self.launcher.physical_cores} (Your CPU physical cores)")
        self.launcher.recommended_threads_batch_var.set(f"Recommended: {self.launcher.logical_cores} (Your CPU logical cores)")

        # Display initial GPU detection status message
        self.launcher.gpu_detected_status_var.set(self.launcher.gpu_info['message'] if not self.launcher.gpu_info['available'] and self.launcher.gpu_info.get('message') else "")
