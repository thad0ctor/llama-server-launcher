import json
import os
import re
import subprocess
import sys
import traceback
import ctypes
import struct
from pathlib import Path

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


try:
    # Import Llama only if llama_cpp_python is importable
    from llama_cpp import Llama
    LLAMA_CPP_PYTHON_AVAILABLE = True
except ImportError:
    LLAMA_CPP_PYTHON_AVAILABLE = False
    Llama = None # Define Llama as None if import fails
except Exception as e:
    # Catch other potential issues during llama_cpp import (e.g., libllama.so not found)
    LLAMA_CPP_PYTHON_AVAILABLE = False
    Llama = None
    print(f"Warning: llama-cpp-python import failed: {e}", file=sys.stderr)

# Add debug prints for Python environment



try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None
    print("Warning: psutil library not found. RAM and CPU information may be limited.", file=sys.stderr)


# --- Dependency Check (Printed to console/stderr) ---
MISSING_DEPS = []

# PyTorch is the primary requirement for GPU detection and features
if not TORCH_AVAILABLE:
    MISSING_DEPS.append("PyTorch (required for GPU detection and CUDA features)")

# Optional dependencies
if not LLAMA_CPP_PYTHON_AVAILABLE:
    MISSING_DEPS.append("llama-cpp-python (optional - provides fallback GGUF analysis if llama.cpp tools unavailable)")

# psutil check already prints a warning if not found.

# Only print missing deps warning if there are actually missing deps
if MISSING_DEPS:
    print("\n--- Missing Dependencies Warning ---")
    print("The following Python libraries are recommended for full functionality but were not found:")
    for dep in MISSING_DEPS:
        print(f" - {dep}")
    print("Please install them within your venv (e.g., 'pip install torch llama-cpp-python') for GPU features and enhanced functionality.")
    print("Note: GGUF analysis works without llama-cpp-python using built-in tools or simple header parsing.")
    print("-------------------------------------\n")


# ═════════════════════════════════════════════════════════════════════
#  Helper Functions (These remain outside the class as they don't need 'self')
# ═════════════════════════════════════════════════════════════════════

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


def analyze_gguf_with_llamacpp_tools(model_path_str, llama_cpp_dir=None):
    """Analyze GGUF model using llama.cpp tools instead of llama-cpp-python."""
    import subprocess
    import json
    from pathlib import Path
    
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
        "metadata": {},
        "error": None,
        "message": None,
        "shard_count": shard_count,
        "all_shards": [str(p) for p in all_shards]
    }

    # Try to find llama-inspect or similar tools
    tools_to_try = []
    
    if llama_cpp_dir:
        llama_base_dir = Path(llama_cpp_dir)
        # Common locations for llama.cpp tools
        search_paths = [
            llama_base_dir,
            llama_base_dir / "build" / "bin",
            llama_base_dir / "build",
            llama_base_dir / "bin",
        ]
        
        tool_names = ["llama-inspect", "gguf-dump", "llama-server"]
        if sys.platform == "win32":
            tool_names = [name + ".exe" for name in tool_names]
        
        for search_path in search_paths:
            for tool_name in tool_names:
                tool_path = search_path / tool_name
                if tool_path.is_file():
                    tools_to_try.append((str(tool_path), tool_name.replace(".exe", "")))

    # Try each tool
    for tool_path, tool_name in tools_to_try:
        try:
            if tool_name == "llama-inspect":
                # llama-inspect typically outputs JSON or structured data
                result = subprocess.run([tool_path, str(model_path)], 
                                      capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    # Parse the output (this would need to be adapted based on actual llama-inspect output format)
                    output = result.stdout.strip()
                    # Try to extract key information from the output
                    if "layers" in output.lower() or "block_count" in output.lower():
                        # Parse layer count from output
                        for line in output.split('\n'):
                            if 'block_count' in line.lower() or 'n_layers' in line.lower():
                                try:
                                    # Extract number from line (this is a simple approach, may need refinement)
                                    import re
                                    numbers = re.findall(r'\d+', line)
                                    if numbers:
                                        analysis_result["n_layers"] = int(numbers[-1])
                                        break
                                except:
                                    pass
                    
                    # Extract architecture if possible
                    for line in output.split('\n'):
                        if 'architecture' in line.lower():
                            # Simple extraction - would need refinement based on actual output format
                            parts = line.split()
                            if len(parts) > 1:
                                analysis_result["architecture"] = parts[-1]
                            break
                    
                    analysis_result["message"] = f"Analyzed using {tool_name}"
                    return analysis_result
                    
            elif tool_name == "llama-server":
                # Try to get model info from llama-server without starting it
                # Some versions support --model-info or similar flags
                for flag in ["--model-info", "--print-model-info", "--help"]:
                    try:
                        result = subprocess.run([tool_path, flag, "-m", str(model_path)], 
                                              capture_output=True, text=True, timeout=10)
                        if result.returncode == 0 and ("layer" in result.stdout.lower() or "block" in result.stdout.lower()):
                            # Similar parsing as above
                            output = result.stdout.strip()
                            for line in output.split('\n'):
                                if 'block' in line.lower() or 'layer' in line.lower():
                                    try:
                                        import re
                                        numbers = re.findall(r'\d+', line)
                                        if numbers:
                                            analysis_result["n_layers"] = int(numbers[-1])
                                            analysis_result["message"] = f"Analyzed using {tool_name}"
                                            return analysis_result
                                    except:
                                        pass
                            break
                    except:
                        continue
                        
        except subprocess.TimeoutExpired:
            continue
        except Exception as e:
            print(f"DEBUG: Tool {tool_name} failed: {e}", file=sys.stderr)
            continue

    # If no tools worked, try simple GGUF header parsing
    try:
        return parse_gguf_header_simple(model_path_str)
    except Exception as e:
        analysis_result["error"] = f"All analysis methods failed. Last error: {e}"
        analysis_result["message"] = "Could not analyze model with available tools"
        return analysis_result


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


def parse_gguf_header_simple(model_path_str):
    """Simple GGUF header parser to extract basic metadata without dependencies."""
    import struct
    from pathlib import Path
    
    model_path = Path(model_path_str)
    
    # Calculate total size across all shards if this is a multi-part file
    total_size_bytes, shard_count, all_shards = calculate_total_gguf_size(model_path_str)
    
    analysis_result = {
        "path": str(model_path),
        "file_size_bytes": total_size_bytes,
        "file_size_gb": round(total_size_bytes / (1024**3), 2),
        "architecture": "unknown",
        "n_layers": None,
        "metadata": {},
        "error": None,
        "message": f"Analyzed using simple GGUF parser ({shard_count} shard{'s' if shard_count != 1 else ''})",
        "shard_count": shard_count,
        "all_shards": [str(p) for p in all_shards]
    }
    
    try:
        with open(model_path, 'rb') as f:
            # Read GGUF magic number
            magic = f.read(4)
            if magic != b'GGUF':
                analysis_result["error"] = "Not a valid GGUF file"
                return analysis_result
            
            # Read version
            version = struct.unpack('<I', f.read(4))[0]
            
            # Read tensor count and metadata count
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            metadata_count = struct.unpack('<Q', f.read(8))[0]
            
            # Read metadata key-value pairs
            for _ in range(metadata_count):
                try:
                    # Read key length and key
                    key_len = struct.unpack('<Q', f.read(8))[0]
                    key = f.read(key_len).decode('utf-8')
                    
                    # Read value type
                    value_type = struct.unpack('<I', f.read(4))[0]
                    
                    # Parse value based on type (improved error handling)
                    value = None
                    if value_type == 8:  # String
                        try:
                            value_len_bytes = f.read(8)
                            if len(value_len_bytes) < 8:
                                break  # End of file
                            value_len = struct.unpack('<Q', value_len_bytes)[0]
                            if value_len > 100000:  # Sanity check for string length
                                print(f"DEBUG: Skipping large string for key '{key}': {value_len} bytes", file=sys.stderr)
                                f.seek(f.tell() + value_len)  # Skip the string
                                continue
                            value_bytes = f.read(value_len)
                            if len(value_bytes) < value_len:
                                break  # End of file
                            value = value_bytes.decode('utf-8', errors='replace')
                        except Exception as e:
                            print(f"DEBUG: Failed to read string for key '{key}': {e}", file=sys.stderr)
                            continue
                    elif value_type == 4:  # Uint32
                        try:
                            value_bytes = f.read(4)
                            if len(value_bytes) < 4:
                                break
                            value = struct.unpack('<I', value_bytes)[0]
                        except Exception as e:
                            print(f"DEBUG: Failed to read uint32 for key '{key}': {e}", file=sys.stderr)
                            continue
                    elif value_type == 5:  # Int32
                        try:
                            value_bytes = f.read(4)
                            if len(value_bytes) < 4:
                                break
                            value = struct.unpack('<i', value_bytes)[0]
                        except Exception as e:
                            print(f"DEBUG: Failed to read int32 for key '{key}': {e}", file=sys.stderr)
                            continue
                    elif value_type == 6:  # Uint64
                        try:
                            value_bytes = f.read(8)
                            if len(value_bytes) < 8:
                                break
                            value = struct.unpack('<Q', value_bytes)[0]
                        except Exception as e:
                            print(f"DEBUG: Failed to read uint64 for key '{key}': {e}", file=sys.stderr)
                            continue
                    elif value_type == 7:  # Float32
                        try:
                            value_bytes = f.read(4)
                            if len(value_bytes) < 4:
                                break
                            value = struct.unpack('<f', value_bytes)[0]
                        except Exception as e:
                            print(f"DEBUG: Failed to read float32 for key '{key}': {e}", file=sys.stderr)
                            continue
                    elif value_type == 9:  # Bool
                        try:
                            bool_byte = f.read(1)
                            if len(bool_byte) < 1:
                                break
                            value = bool_byte[0] != 0
                        except Exception as e:
                            print(f"DEBUG: Failed to read bool for key '{key}': {e}", file=sys.stderr)
                            continue
                    elif value_type == 10:  # Array
                        try:
                            array_type_bytes = f.read(4)
                            array_len_bytes = f.read(8)
                            if len(array_type_bytes) < 4 or len(array_len_bytes) < 8:
                                break
                            array_type = struct.unpack('<I', array_type_bytes)[0]
                            array_len = struct.unpack('<Q', array_len_bytes)[0]
                            
                            print(f"DEBUG: Skipping array for key '{key}': type {array_type}, length {array_len}", file=sys.stderr)
                            
                            # Skip array data based on type
                            if array_type == 8:  # String array
                                for i in range(min(array_len, 10000)):  # Limit to prevent infinite loops
                                    try:
                                        str_len_bytes = f.read(8)
                                        if len(str_len_bytes) < 8:
                                            break
                                        str_len = struct.unpack('<Q', str_len_bytes)[0]
                                        if str_len > 100000:  # Sanity check
                                            break
                                        f.seek(f.tell() + str_len)
                                    except:
                                        break
                            elif array_type == 4:  # Uint32 array
                                f.seek(f.tell() + array_len * 4)
                            elif array_type == 6:  # Uint64 array
                                f.seek(f.tell() + array_len * 8)
                            elif array_type == 7:  # Float32 array
                                f.seek(f.tell() + array_len * 4)
                            else:
                                # Unknown array type, try to skip conservatively
                                print(f"DEBUG: Unknown array type {array_type}, attempting to skip", file=sys.stderr)
                                break
                        except Exception as e:
                            print(f"DEBUG: Failed to skip array for key '{key}': {e}", file=sys.stderr)
                            break
                        continue
                    else:
                        print(f"DEBUG: Unknown value type {value_type} for key '{key}', skipping", file=sys.stderr)
                        continue
                    
                    if value is not None:
                        analysis_result["metadata"][key] = value
                        
                        # Extract key information
                        if key == "general.architecture":
                            analysis_result["architecture"] = str(value)
                        elif any(pattern in key.lower() for pattern in ['.block_count', '.n_layers', '.layer_count', '.num_layer']):
                            if isinstance(value, (int, float)) and value > 0:
                                analysis_result["n_layers"] = int(value)
                                print(f"DEBUG: Found layer count in key '{key}': {analysis_result['n_layers']}", file=sys.stderr)
                        
                        # Also check for model name patterns that might indicate layer count
                        if 'name' in key.lower() and isinstance(value, str):
                            # Try to extract parameter count from model name (e.g., "235B" -> estimate layers)
                            import re
                            param_match = re.search(r'(\d+)B', value.upper())
                            if param_match and analysis_result["n_layers"] is None:
                                param_count = int(param_match.group(1))
                                # Rough estimate: large models typically have ~80-120 layers per billion parameters
                                # For very large models like 235B, use a more conservative estimate
                                if param_count >= 100:
                                    estimated_layers = max(80, min(200, param_count // 2))  # Very conservative for huge models
                                else:
                                    estimated_layers = max(32, min(120, param_count * 2))  # More typical estimate
                                print(f"DEBUG: Estimated {estimated_layers} layers from model name '{value}' ({param_count}B parameters)", file=sys.stderr)
                                analysis_result["n_layers"] = estimated_layers
                        
                except Exception as parse_error:
                    # Skip this metadata entry if parsing fails
                    print(f"DEBUG: Failed to parse metadata entry: {parse_error}", file=sys.stderr)
                    continue
            
            # If we still don't have layer count, make a reasonable guess based on file size
            if analysis_result["n_layers"] is None:
                file_size_gb = analysis_result["file_size_gb"]
                if file_size_gb > 100:  # Very large model (100+ GB)
                    estimated_layers = 120  # Conservative estimate for huge models
                    print(f"DEBUG: No layer count found, estimating {estimated_layers} layers based on very large file size ({file_size_gb:.1f} GB)", file=sys.stderr)
                elif file_size_gb > 50:  # Large model (50-100 GB)
                    estimated_layers = 80
                    print(f"DEBUG: No layer count found, estimating {estimated_layers} layers based on large file size ({file_size_gb:.1f} GB)", file=sys.stderr)
                elif file_size_gb > 20:  # Medium-large model (20-50 GB)
                    estimated_layers = 60
                    print(f"DEBUG: No layer count found, estimating {estimated_layers} layers based on medium-large file size ({file_size_gb:.1f} GB)", file=sys.stderr)
                elif file_size_gb > 10:  # Medium model (10-20 GB)
                    estimated_layers = 40
                    print(f"DEBUG: No layer count found, estimating {estimated_layers} layers based on medium file size ({file_size_gb:.1f} GB)", file=sys.stderr)
                elif file_size_gb > 3:   # Small-medium model (3-10 GB)
                    estimated_layers = 32
                    print(f"DEBUG: No layer count found, estimating {estimated_layers} layers based on small-medium file size ({file_size_gb:.1f} GB)", file=sys.stderr)
                else:  # Small model (< 3 GB)
                    estimated_layers = 24
                    print(f"DEBUG: No layer count found, estimating {estimated_layers} layers based on small file size ({file_size_gb:.1f} GB)", file=sys.stderr)
                
                analysis_result["n_layers"] = estimated_layers
                analysis_result["message"] += f" (estimated {estimated_layers} layers from file size)"
            
            return analysis_result
            
    except Exception as e:
        analysis_result["error"] = f"Failed to parse GGUF header: {e}"
        return analysis_result


def analyze_gguf_model_static(model_path_str):
    """Analyze a GGUF model file and extract metadata (static method)."""
    if not LLAMA_CPP_PYTHON_AVAILABLE or not Llama:
        return {"error": "llama-cpp-python library not found.", "n_layers": None, "architecture": "N/A", "file_size_bytes": 0}

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
        "metadata": {},
        "error": None,
        "message": None,
        "shard_count": shard_count,
        "all_shards": [str(p) for p in all_shards]
    }

    llm_meta = None # Initialize llm_meta outside the try block

    try:
        # File size already calculated from shards above, no need to recalculate

        try:
             # Use minimal parameters for quick metadata load
             # Setting n_gpu_layers=0 is important to avoid trying to load layers onto potentially unavailable GPUs
             # Using minimal n_ctx, n_batch, n_threads for minimal resource usage
             llm_meta = Llama(model_path=str(model_path), n_ctx=32, n_threads=1, n_batch=32,
                              verbose=False, n_gpu_layers=0, logits_all=False) # logits_all=False to save memory/time
        except Exception as load_exc:
             analysis_result["error"] = f"Failed to load model for metadata analysis: {load_exc}"
             print(f"ERROR: Failed to load model '{model_path.name}' for analysis: {load_exc}", file=sys.stderr)
             # No traceback here, the caller should handle it or the log message is enough
             return analysis_result # Exit early if basic load fails


        # --- Extract Metadata ---
        # Attempt 1: Check common metadata keys and attributes
        if hasattr(llm_meta, 'metadata') and isinstance(llm_meta.metadata, dict) and llm_meta.metadata:
            analysis_result["metadata"] = llm_meta.metadata
            # Check various common keys for layer count
            analysis_result["n_layers"] = llm_meta.metadata.get('llama.block_count')
            if analysis_result["n_layers"] is None:
                 analysis_result["n_layers"] = llm_meta.metadata.get('general.architecture.block_count')
            # Add more architecture-specific keys as needed
            if analysis_result["n_layers"] is None: analysis_result["n_layers"] = llm_meta.metadata.get('qwen2.block_count')
            if analysis_result["n_layers"] is None: analysis_result["n_layers"] = llm_meta.metadata.get('gemma.block_count')
            if analysis_result["n_layers"] is None: analysis_result["n_layers"] = llm_meta.metadata.get('bert.block_count')
            if analysis_result["n_layers"] is None: analysis_result["n_layers"] = llm_meta.metadata.get('model.block_count') # Fallback


            # Check various common keys for architecture
            analysis_result["architecture"] = llm_meta.metadata.get('general.architecture', 'unknown')
            if analysis_result["architecture"] == 'unknown': analysis_result["architecture"] = llm_meta.metadata.get('qwen2.architecture', 'unknown')
            if analysis_result["architecture"] == 'unknown': analysis_result["architecture"] = llm_meta.metadata.get('gemma.architecture', 'unknown')
            if analysis_result["architecture"] == 'unknown': analysis_result["architecture"] = llm_meta.metadata.get('bert.architecture', 'unknown')


        # Attempt 2: Check direct attributes if metadata didn't yield layers
        if analysis_result["n_layers"] is None:
             if hasattr(llm_meta, 'n_layer'): # Common recent name
                  analysis_result["n_layers"] = getattr(llm_meta, 'n_layer', None)
             if analysis_result["n_layers"] is None and hasattr(llm_meta, 'n_layers'): # Older name
                  analysis_result["n_layers"] = getattr(llm_meta, 'n_layers', None)

        # Fallback architecture if still unknown (less reliable)
        if analysis_result["architecture"] == 'unknown' and hasattr(llm_meta, 'model_type'):
             analysis_result["architecture"] = getattr(llm_meta, 'model_type', 'unknown')


        # Validate n_layers
        if analysis_result["n_layers"] is not None:
            try:
                analysis_result["n_layers"] = int(analysis_result["n_layers"])
                if analysis_result["n_layers"] <= 0:
                    analysis_result["n_layers"] = None # Treat non-positive as unknown
                    analysis_result["message"] = "Layer count found was not positive."
            except (ValueError, TypeError):
                analysis_result["n_layers"] = None # Treat non-integer as unknown
                analysis_result["message"] = "Layer count metadata found was not an integer."

        if analysis_result["n_layers"] is None:
             if not analysis_result["message"]: # If no specific message set above
                  analysis_result["message"] = "Layer count metadata not found or recognized."


        return analysis_result

    except Exception as e:
        # Catch any other unexpected errors during metadata processing or attribute access
        print(f"ERROR: Failed during GGUF metadata processing for '{model_path.name}': {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        analysis_result["error"] = f"Unexpected error during analysis: {e}"
        analysis_result["n_layers"] = None # Ensure layers is None on error
        return analysis_result
    finally:
        # Ensure the temporary Llama object is deleted if it was created
        if llm_meta:
             try:
                  del llm_meta
             except Exception as clean_exc:
                  print(f"Warning: Failed to delete llama_cpp.Llama instance in analyze_gguf_model_static finally block: {clean_exc}", file=sys.stderr) 

class SystemInfoManager:
    """Manages system information fetching and processing for the launcher."""
    
    def __init__(self, launcher_instance):
        """Initialize with reference to the main launcher instance."""
        self.launcher = launcher_instance
    
    def fetch_system_info(self):
        """Fetches GPU, RAM, and CPU info and populates class attributes."""
        print("Fetching system info...", file=sys.stderr)
        self.launcher.gpu_info = get_gpu_info_static()
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