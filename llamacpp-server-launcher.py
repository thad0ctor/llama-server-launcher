import json
import os
import re
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox, scrolledtext
from threading import Thread
import traceback
import ctypes
import shlex
import math

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

# Check for Flash Attention library in the *GUI's* environment (fallback/initial status)
FLASH_ATTN_GUI_AVAILABLE = False
# Flash Attention typically depends on a CUDA build of PyTorch and a compatible llama.cpp build
# We check for the Python package as a proxy for availability in the environment.
if TORCH_AVAILABLE and sys.platform != "darwin": # Flash Attention is CUDA/Linux/Windows specific currently, not macOS
    try:
        # We specifically check for the underlying C++ extension or key components, not just the package import
        # as the 'flash_attn' package might import but not have the CUDA kernels built.
        # A simple check is just importing the top level, hoping for the best, or looking for specific symbols.
        # A more robust check would involve torch.ops.flash_attn, but that requires torch to be ready.
        # Let's stick to the simple import check as a proxy for the Python environment setup.
        import flash_attn
        FLASH_ATTN_GUI_AVAILABLE = True
    except ImportError:
        FLASH_ATTN_GUI_AVAILABLE = False
    except Exception as e:
        FLASH_ATTN_GUI_AVAILABLE = False
        print(f"Warning: Flash Attention library import failed in GUI environment: {e}", file=sys.stderr)
else:
     if sys.platform == "darwin":
          print("Note: Flash Attention is generally not supported on macOS.", file=sys.stderr)


try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None
    print("Warning: psutil library not found. RAM and CPU information may be limited.", file=sys.stderr)


# --- Dependency Check (Printed to console/stderr) ---
MISSING_DEPS = []
if not LLAMA_CPP_PYTHON_AVAILABLE:
    MISSING_DEPS.append("llama-cpp-python (for GGUF analysis)")
if LLAMA_CPP_PYTHON_AVAILABLE: # Only warn about these if llama_cpp is available, suggesting GPU interest
    if not TORCH_AVAILABLE:
         MISSING_DEPS.append("PyTorch (with CUDA support likely needed for GPU detection/offload)")
    # Only mention flash_attn if CUDA is available via torch, and not macOS
    if TORCH_AVAILABLE and sys.platform != "darwin" and not FLASH_ATTN_GUI_AVAILABLE:
         MISSING_DEPS.append("Flash Attention (Python package 'flash_attn' - optional for --flash-attn)")

# psutil check already prints a warning if not found.

if MISSING_DEPS:
    print("\n--- Missing Dependencies Warning ---", file=sys.stderr)
    print("The following Python libraries are recommended for full functionality but were not found:", file=sys.stderr)
    for dep in MISSING_DEPS:
        print(f" - {dep}")
    print("Please install them (e.g., 'pip install llama-cpp-python torch psutil flash_attn') if you need GGUF analysis, GPU features, or Flash Attention status.")
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
                    return {"error": "Windows GlobalMemoryStatusEx failed"}
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


def analyze_gguf_model_static(model_path_str):
    """Analyze a GGUF model file and extract metadata (static method)."""
    if not LLAMA_CPP_PYTHON_AVAILABLE or not Llama:
        return {"error": "llama-cpp-python library not found.", "n_layers": None, "architecture": "N/A", "file_size_bytes": 0}

    model_path = Path(model_path_str)
    if not model_path.is_file():
         return {"error": f"Model file not found: {model_path}", "n_layers": None, "architecture": "N/A", "file_size_bytes": 0}

    analysis_result = {
        "path": str(model_path),
        "file_size_bytes": 0,
        "file_size_gb": 0,
        "architecture": "unknown",
        "n_layers": None,
        "metadata": {},
        "error": None,
        "message": None
    }

    try:
        analysis_result["file_size_bytes"] = model_path.stat().st_size
        analysis_result["file_size_gb"] = round(analysis_result["file_size_bytes"] / (1024**3), 2)

        llm_meta = None
        try:
             # Use minimal parameters for quick metadata load
             # Setting n_gpu_layers=0 is important to avoid trying to load layers onto potentially unavailable GPUs
             # Using minimal n_ctx, n_batch, n_threads for minimal resource usage
             llm_meta = Llama(model_path=str(model_path), n_ctx=32, n_threads=1, n_batch=32,
                              verbose=False, n_gpu_layers=0, logits_all=False) # logits_all=False to save memory/time
        except Exception as load_exc:
             analysis_result["error"] = f"Failed to load model for metadata analysis: {load_exc}"
             print(f"ERROR: Failed to load model '{model_path.name}' for analysis: {load_exc}", file=sys.stderr)
             traceback.print_exc(file=sys.stderr)
             # Clean up the temporary Llama object on load failure too
             if llm_meta:
                  try: del llm_meta
                  except Exception as clean_exc: print(f"Warning: Failed to delete llama_cpp.Llama instance after load error: {clean_exc}", file=sys.stderr)

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


        # Clean up the temporary Llama object
        if llm_meta:
             try:
                  del llm_meta
             except Exception as clean_exc:
                  print(f"Warning: Failed to delete llama_cpp.Llama instance: {clean_exc}", file=sys.stderr)

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


# ═════════════════════════════════════════════════════════════════════
#  Main class
# ═════════════════════════════════════════════════════════════════════
class LlamaCppLauncher:
    """Tk‑based launcher for llama.cpp HTTP server."""

    # ──────────────────────────────────────────────────────────────────
    #  Predefined Chat Templates
    # ──────────────────────────────────────────────────────────────────
    # Define predefined chat templates here
    _predefined_templates = {
        "None (Use Model Default)": "", # Empty string means no --chat-template argument
        "Alpaca": "### Instruction:\\n{{instruction}}\\n### Response:\\n{{response}}",
        "ChatML": "<|im_start|>system\\n{{system_message}}<|im_end|>\\n<|im_start|>user\\n{{prompt}}<|im_end|>\\n<|im_start|>assistant\\n",
        "Llama 2 Chat": "[INST] <<SYS>>\\n{{system_message}}\\n<</SYS>>\\n\\n{{prompt}}[/INST]",
        "Vicuna": "A chat between a curious user and an AI assistant.\nThe assistant gives helpful, harmless, honest answers.\nUSER: {{prompt}}\nASSISTANT: ",
        # Add more templates as needed based on common formats
        # "Gemma": "<start_of_turn>user\\n{{prompt}}<end_of_turn>\\n<start_of_turn>model\\n",
        # "Mistral": "[INST] {{prompt}} [/INST]",
    }


    # ──────────────────────────────────────────────────────────────────
    #  construction / persistence
    # ──────────────────────────────────────────────────────────────────
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("LLaMa.cpp Server Launcher")
        self.root.geometry("900x850")
        self.root.minsize(800, 750)

        # --- System Info Attributes ---
        # These will be populated by _fetch_system_info
        self.gpu_info = {"available": False, "device_count": 0, "devices": []}
        self.ram_info = {}
        self.cpu_info = {"logical_cores": 4, "physical_cores": 2} # Default fallback


        # ------------------------------------------------ persistence --
        self.config_path = self._get_config_path()
        self.saved_configs = {}
        self.app_settings = {
            "last_llama_cpp_dir": "",
            "last_venv_dir":      "",
            "last_model_path":    "",
            "model_dirs":         [],
            "model_list_height":  8,
            "selected_gpus":      [], # Save/load selected GPU indices (indices of detected GPUs)
        }

        # ------------------------------------------------ Tk variables --
        self.llama_cpp_dir   = tk.StringVar()
        self.venv_dir        = tk.StringVar()
        self.model_path      = tk.StringVar()
        self.model_dirs_listvar = tk.StringVar()
        self.scan_status_var = tk.StringVar(value="Scan models to populate list.")

        # Basic settings
        self.cache_type_k    = tk.StringVar(value="f16") # KV cache type (applies to k & v)
        # Set initial thread defaults based on detected CPU cores *after* fetching info
        # These will be set in _fetch_system_info or immediately after if fetch works
        self.threads         = tk.StringVar(value="4") # Initial default before info fetch
        self.threads_batch   = tk.StringVar(value="4") # Initial default before info fetch
        self.batch_size      = tk.StringVar(value="512") # --batch-size (prompt)
        self.ubatch_size     = tk.StringVar(value="512") # --ubatch-size (unconditional batch)
        self.temperature     = tk.StringVar(value="0.8")
        self.min_p           = tk.StringVar(value="0.05")
        self.ctx_size        = tk.IntVar(value=2048)
        self.seed            = tk.StringVar(value="-1")

        # GPU settings
        self.n_gpu_layers    = tk.StringVar(value="0") # String var for the entry (-1, 0, N)
        self.n_gpu_layers_int = tk.IntVar(value=0) # Int var for the slider (0 to MaxLayers)
        self.max_gpu_layers  = tk.IntVar(value=0) # Max layers detected in model for slider max
        self.gpu_layers_status_var = tk.StringVar(value="Select model to see layer info") # Status/info next to layers control

        self.flash_attn      = tk.BooleanVar(value=False)
        self.tensor_split    = tk.StringVar(value="")
        self.main_gpu        = tk.StringVar(value="0") # String for main-gpu entry

        self.gpu_vars = [] # List of BooleanVars for GPU checkboxes (populated dynamically)

        # Memory & other advanced
        self.no_mmap         = tk.BooleanVar(value=False)
        self.no_cnv          = tk.BooleanVar(value=False) # Disable convolutional batching? (Might be related to -b / --batch-size internal implementation details)
        self.prio            = tk.StringVar(value="0")
        self.mlock           = tk.BooleanVar(value=False)
        self.no_kv_offload   = tk.BooleanVar(value=False)
        self.host            = tk.StringVar(value="127.0.0.1")
        self.port            = tk.StringVar(value="8080")
        self.config_name     = tk.StringVar(value="default_config") # Initial default name


        # --- NEW Parameters ---
        self.ignore_eos      = tk.BooleanVar(value=False) # --ignore-eos
        self.n_predict       = tk.StringVar(value="-1")   # --n-predict

        # --- Chat Template Parameters (NEW) ---
        # BooleanVar: True for custom entry, False for predefined combobox
        self.use_custom_template = tk.BooleanVar(value=False)
        # StringVar: Holds the key name of the selected predefined template
        self.predefined_template_name = tk.StringVar(value="None (Use Model Default)")
        # StringVar: Holds the user's custom template string
        self.custom_template_string = tk.StringVar(value="")
        # StringVar: Holds the *actual* template string being used (either from predefined or custom) for display and command line
        self.current_template_display = tk.StringVar(value="")


        # --- Model Info Variables ---
        self.model_architecture_var = tk.StringVar(value="N/A")
        self.model_filesize_var = tk.StringVar(value="N/A")
        self.model_total_layers_var = tk.StringVar(value="N/A") # Total layers from GGUF analysis
        self.model_kv_cache_type_var = tk.StringVar(value="N/A") # Current selected KV cache type

        # --- Recommendation Variables ---
        self.recommended_tensor_split_var = tk.StringVar(value="N/A - Need >1 selected GPU & model layers")
        # Initial placeholder, will be updated after system info fetch
        self.recommended_threads_var = tk.StringVar(value=f"Recommended: Detecting...")
        self.recommended_threads_batch_var = tk.StringVar(value=f"Recommended: Detecting...")


        # --- Status Variables ---
        # Initial status reflects GUI environment, will update based on venv check
        self.flash_attn_status_var = tk.StringVar(value=f"Status (GUI env): {'Installed (requires llama.cpp build support)' if FLASH_ATTN_GUI_AVAILABLE else 'Not Installed (Python package missing in GUI env)'}")
        self.gpu_detected_status_var = tk.StringVar(value="") # Updates with GPU detection message


        # Internal state
        self.model_dirs = []
        self.found_models = {} # {display_name: full_path_obj}
        self.current_model_analysis = {} # Holds the result of the last GGUF analysis
        self.analysis_thread = None
        # detected_gpu_devices is populated by _fetch_system_info
        self.detected_gpu_devices = [] # List of detected GPU info dicts
        # logical_cores and physical_cores are populated by _fetch_system_info
        self.logical_cores = 4 # Fallback
        self.physical_cores = 2 # Fallback


        # --- Fetch System Info ---
        self._fetch_system_info()
        # Update Tk vars based on detected info
        self.threads.set(str(self.physical_cores))
        self.threads_batch.set(str(self.logical_cores))
        # Update recommendation vars based on detected info
        self.recommended_threads_var.set(f"Recommended: {self.physical_cores} (Your CPU physical cores)")
        self.recommended_threads_batch_var.set(f"Recommended: {self.logical_cores} (Your CPU logical cores)")
        # Display initial GPU detection status message
        self.gpu_detected_status_var.set(self.gpu_info['message'] if not self.gpu_info['available'] and self.gpu_info.get('message') else "")


        # load previous settings
        self._load_saved_configs()
        self.llama_cpp_dir.set(self.app_settings.get("last_llama_cpp_dir", ""))
        self.venv_dir.set(self.app_settings.get("last_venv_dir", ""))
        # Ensure model_dirs is loaded as list of Paths
        self.model_dirs = [Path(d) for d in self.app_settings.get("model_dirs", []) if d]

        # build GUI
        self._create_widgets()

        # --- Setup validation for GPU Layers Entry AFTER widgets are created ---
        vcmd = (self.root.register(self._validate_gpu_layers_entry), '%P')
        if hasattr(self, 'n_gpu_layers_entry') and self.n_gpu_layers_entry.winfo_exists():
             self.n_gpu_layers_entry.config(validate='key', validatecommand=vcmd)
             self.n_gpu_layers_entry.bind("<FocusOut>", self._sync_gpu_layers_from_entry)
             self.n_gpu_layers_entry.bind("<Return>", self._sync_gpu_layers_from_entry)

        # --- Bind callbacks for recommendations/info update ---
        # Bind trace to cache_type_k variable to update the Model Info section
        self.cache_type_k.trace_add("write", lambda *args: self._update_recommendations())
        # Context size updates handled in _override_ctx_size and _update_ctx_label_from_slider
        # n_gpu_layers updates handled in _set_gpu_layers (called by entry/slider sync)
        # GPU selection updates handled in _on_gpu_selection_changed
        # Bind trace to venv_dir to trigger the flash attn check
        self.venv_dir.trace_add("write", lambda *args: self._check_venv_flash_attn())
        # Bind trace to n_predict to update default config name if it's currently the generated one
        self.n_predict.trace_add("write", lambda *args: self._update_default_config_name_if_needed())
        # Bind trace to ignore_eos to update default config name if needed
        self.ignore_eos.trace_add("write", lambda *args: self._update_default_config_name_if_needed())
        # Bind trace to other variables that affect the default config name
        self.cache_type_k.trace_add("write", lambda *args: self._update_default_config_name_if_needed())
        self.threads.trace_add("write", lambda *args: self._update_default_config_name_if_needed())
        self.threads_batch.trace_add("write", lambda *args: self._update_default_config_name_if_needed())
        self.batch_size.trace_add("write", lambda *args: self._update_default_config_name_if_needed())
        self.ubatch_size.trace_add("write", lambda *args: self._update_default_config_name_if_needed())
        self.ctx_size.trace_add("write", lambda *args: self._update_default_config_name_if_needed())
        self.seed.trace_add("write", lambda *args: self._update_default_config_name_if_needed())
        self.temperature.trace_add("write", lambda *args: self._update_default_config_name_if_needed())
        self.min_p.trace_add("write", lambda *args: self._update_default_config_name_if_needed())
        # GPU related updates are handled by _on_gpu_selection_changed (which calls _update_recommendations and then _generate_default_config_name)
        self.flash_attn.trace_add("write", lambda *args: self._update_default_config_name_if_needed())
        self.tensor_split.trace_add("write", lambda *args: self._update_default_config_name_if_needed())
        self.main_gpu.trace_add("write", lambda *args: self._update_default_config_name_if_needed())
        self.no_mmap.trace_add("write", lambda *args: self._update_default_config_name_if_needed())
        self.no_cnv.trace_add("write", lambda *args: self._update_default_config_name_if_needed())
        self.prio.trace_add("write", lambda *args: self._update_default_config_name_if_needed())
        self.mlock.trace_add("write", lambda *args: self._update_default_config_name_if_needed())
        self.no_kv_offload.trace_add("write", lambda *args: self._update_default_config_name_if_needed())

        # --- Chat Template Trace Bindings (NEW) ---
        self.use_custom_template.trace_add("write", lambda *args: self._update_template_controls_state())
        # Trace on combobox variable to update the displayed template string
        self.predefined_template_name.trace_add("write", lambda *args: self._update_effective_template_display())
        # Trace on custom entry variable to update the displayed template string (conditionally)
        self.custom_template_string.trace_add("write", lambda *args: self._update_effective_template_display())


        # Populate model directories listbox
        self._update_model_dirs_listbox()

        # Update initial GPU checkbox states and recommendations based on loaded config and detected GPUs
        self._update_gpu_checkboxes()
        self._update_recommendations() # Call initially to set all initial recommendations

        # Perform initial scan (in background) if dirs exist
        if self.model_dirs:
            self.scan_status_var.set("Scanning on startup...")
            scan_thread = Thread(target=self._scan_model_dirs, daemon=True)
            scan_thread.start()
        else:
             self.scan_status_var.set("Add directories and scan for models.")

        # Run initial Flash Attention check (might trigger venv check if venv is pre-filled)
        self._check_venv_flash_attn()

        # Update chat template display and controls initially
        self._update_template_controls_state() # Sets initial state based on self.use_custom_template
        self._update_effective_template_display() # Sets initial displayed template based on selected/custom value


    # ═════════════════════════════════════════════════════════════════
    #  System Info Fetching (Now a class method)
    # ═════════════════════════════════════════════════════════════════
    def _fetch_system_info(self):
        """Fetches GPU, RAM, and CPU info and populates class attributes."""
        print("Fetching system info...", file=sys.stderr)
        self.gpu_info = get_gpu_info_static()
        self.ram_info = get_ram_info_static()
        self.cpu_info = get_cpu_info_static() # Fetch CPU info here

        print(f"GPU Info: {self.gpu_info}", file=sys.stderr)
        if not self.gpu_info["available"] and "message" in self.gpu_info:
             print(f"GPU Detection Info: {self.gpu_info['message']}", file=sys.stderr)
        if "error" in self.ram_info:
             print(f"RAM Detection Error: {self.ram_info['error']}", file=sys.stderr)
        if "error" in self.cpu_info:
             print(f"CPU Detection Error: {self.cpu_info['error']}", file=sys.stderr)

        # Store detected devices separately for easier access
        self.detected_gpu_devices = self.gpu_info.get("devices", [])
        # Store logical/physical cores for initial thread defaults and recommendations
        self.logical_cores = self.cpu_info.get("logical_cores", 4)
        self.physical_cores = self.cpu_info.get("physical_cores", 2) # Use fallback 2 if psutil failed or physical count is 0

        # Update initial default values for threads and threads_batch if they are still the initial fallback values
        # This handles the case where system info is fetched successfully *after* the variables were initialized
        if self.threads.get() == "4" and self.physical_cores != 2: # Check against initial fallback 2
             self.threads.set(str(self.physical_cores))
        if self.threads_batch.get() == "4" and self.logical_cores != 4: # Check against initial fallback 4
             self.threads_batch.set(str(self.logical_cores))


    # ═════════════════════════════════════════════════════════════════
    #  persistence helpers
    # ═════════════════════════════════════════════════════════════════
    def _get_config_path(self):
        local_path = Path("llama_cpp_launcher_configs.json") # Renamed slightly to avoid potential clashes
        try:
            # Check if we can write to the current directory
            if os.access(".", os.W_OK):
                # Check if a config file exists and is empty (possibly from a failed previous run)
                # If empty, we can safely delete it and use the local path.
                if local_path.exists() and local_path.stat().st_size == 0:
                     try: local_path.unlink()
                     except OSError: pass # Ignore if delete fails
                return local_path
            else:
                 raise PermissionError("No write access in current directory.") # Force fallback

        except (OSError, PermissionError, IOError):
            # Fallback to user config directory
            if sys.platform == "win32":
                appdata = os.getenv("APPDATA")
                fallback_dir = Path(appdata) / "LlamaCppLauncher" if appdata else Path.home() / ".llama_cpp_launcher"
            else: # Linux, macOS, etc.
                 fallback_dir = Path.home() / ".config" / "llama_cpp_launcher"

            try:
                fallback_dir.mkdir(parents=True, exist_ok=True)
                fallback_path = fallback_dir / "configs.json"
                # Check if fallback path exists and is empty, clean it up if so
                if fallback_path.exists() and fallback_path.stat().st_size == 0:
                     try: fallback_path.unlink()
                     except OSError: pass
                print(f"Note: Using fallback config path due to permissions/IO issue: {fallback_path}", file=sys.stderr)
                return fallback_path
            except Exception as e:
                 print(f"CRITICAL ERROR: Could not use local config path or fallback config path {fallback_dir}. Configuration saving/loading is disabled. Error: {e}", file=sys.stderr)
                 messagebox.showerror("Config Error", f"Failed to set up configuration directory.\nSaving/loading configurations is disabled.\nError: {e}")
                 # Return a dummy non-existent path to prevent errors later
                 return Path("/dev/null") if sys.platform != "win32" else Path("NUL") # Use platform-appropriate null device


    def _load_saved_configs(self):
        if not self.config_path.exists() or not self.config_path.is_file() or self.config_path.name in ("null", "NUL"):
             print("No config file found or config saving is disabled. Using default settings.", file=sys.stderr)
             return

        try:
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
            self.saved_configs = data.get("configs", {})
            loaded_app_settings = data.get("app_settings", {})
            self.app_settings.update(loaded_app_settings)
            # Ensure model_list_height is a valid int
            if not isinstance(self.app_settings.get("model_list_height"), int):
                self.app_settings["model_list_height"] = 8
            # Ensure selected_gpus is a list
            if not isinstance(self.app_settings.get("selected_gpus"), list):
                 self.app_settings["selected_gpus"] = []

            # Filter selected_gpus to only include indices of currently detected GPUs
            valid_gpu_indices = {gpu['id'] for gpu in self.detected_gpu_devices}
            self.app_settings["selected_gpus"] = [idx for idx in self.app_settings["selected_gpus"] if idx in valid_gpu_indices]


        except json.JSONDecodeError as e:
             print(f"Config Load Error: Failed to parse JSON from {self.config_path}\nError: {e}", file=sys.stderr)
             messagebox.showerror("Config Load Error", f"Failed to parse config file:\n{self.config_path}\n\nError: {e}\n\nUsing default settings.")
             # Reset to defaults on parse error
             self.app_settings = {
                 "last_llama_cpp_dir": "", "last_venv_dir": "", "last_model_path": "",
                 "model_dirs": [], "model_list_height": 8, "selected_gpus": []
             }
             self.saved_configs = {}
        except Exception as exc:
            print(f"Config Load Error: Could not load config from {self.config_path}\nError: {exc}", file=sys.stderr)
            messagebox.showerror("Config Load Error", f"Could not load config from:\n{self.config_path}\n\nError: {exc}\n\nUsing default settings.")
            # Reset to defaults on other load errors
            self.app_settings = {
                "last_llama_cpp_dir": "", "last_venv_dir": "", "last_model_path": "",
                "model_dirs": [], "model_list_height": 8, "selected_gpus": []
            }
            self.saved_configs = {}


    def _save_configs(self):
        if self.config_path.name in ("null", "NUL"):
             print("Config saving is disabled.", file=sys.stderr)
             return

        self.app_settings["model_dirs"] = [str(p) for p in self.model_dirs]
        self.app_settings["last_model_path"] = self.model_path.get()
        # Save selected GPU indices from the current state of the checkboxes
        self.app_settings["selected_gpus"] = [i for i, v in enumerate(self.gpu_vars) if v.get()]

        payload = {
            "configs":      self.saved_configs,
            "app_settings": self.app_settings,
        }
        try:
            self.config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"Config Save Error: Failed to save settings to {self.config_path}\nError: {exc}", file=sys.stderr)
            # Attempt fallback only if the initial path wasn't already a fallback
            if not any(s in str(self.config_path).lower() for s in ["appdata", ".config"]):
                 original_path = self.config_path
                 self.config_path = self._get_config_path() # This might return a fallback
                 if self.config_path != original_path and self.config_path.name not in ("null", "NUL"):
                      try:
                         self.config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                         messagebox.showwarning("Config Save Info", f"Could not write to original location.\nSettings stored in:\n{self.config_path}")
                      except Exception as final_exc:
                          print(f"Config Save Error: Failed to save settings to fallback {self.config_path}\nError: {final_exc}", file=sys.stderr)
                          messagebox.showerror("Config Save Error", f"Failed to save settings to fallback location:\n{self.config_path}\n\nError: {final_exc}")
                 else:
                      messagebox.showerror("Config Save Error", f"Failed to save settings to:\n{self.config_path}\n\nError: {exc}")
            else:
                 messagebox.showerror("Config Save Error", f"Failed to save settings to:\n{self.config_path}\n\nError: {exc}")


    # ═════════════════════════════════════════════════════════════════
    #  UI builders
    # ═════════════════════════════════════════════════════════════════
    def _create_widgets(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=10, pady=10)

        main_frame = ttk.Frame(nb); adv_frame = ttk.Frame(nb); cfg_frame = ttk.Frame(nb); chat_frame = ttk.Frame(nb)
        nb.add(main_frame, text="Main Settings")
        nb.add(adv_frame,  text="Advanced Settings")
        nb.add(chat_frame, text="Chat Template") # Add the new tab
        nb.add(cfg_frame,  text="Configurations")


        self._setup_main_tab(main_frame)
        self._setup_advanced_tab(adv_frame)
        self._setup_chat_template_tab(chat_frame) # Setup the new tab
        self._setup_config_tab(cfg_frame)

        bar = ttk.Frame(self.root); bar.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(bar, text="Launch Server",   command=self.launch_server).pack(side="left",  padx=5)
        ttk.Button(bar, text="Save PS1 Script", command=self.save_ps1_script).pack(side="left", padx=5)
        ttk.Button(bar, text="Exit",            command=self.on_exit).pack(side="right", padx=5)

    # ░░░░░ MAIN TAB ░░░░░
    def _setup_main_tab(self, parent):
        canvas = tk.Canvas(parent, highlightthickness=0); vs = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner  = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(yscrollcommand=vs.set,
                                                             scrollregion=canvas.bbox("all")))
        canvas_window = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))
        canvas.pack(side="left", fill="both", expand=True); vs.pack(side="right", fill="y")

        inner.columnconfigure(1, weight=1) # Make model path entry expand

        r = 0
        ttk.Label(inner, text="Directories & Model", font=("TkDefaultFont", 12, "bold"))\
            .grid(column=0, row=r, sticky="w", padx=10, pady=(10,5)); r += 1
        ttk.Separator(inner, orient="horizontal")\
            .grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=5); r += 1

        ttk.Label(inner, text="LLaMa.cpp Root Directory:")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Entry(inner, textvariable=self.llama_cpp_dir, width=50)\
            .grid(column=1, row=r, columnspan=2, sticky="ew", padx=5, pady=3)
        ttk.Button(inner, text="Browse…", command=lambda: self._browse_dir(self.llama_cpp_dir))\
            .grid(column=3, row=r, padx=5, pady=3)
        ttk.Label(inner, text="Select the main 'llama.cpp' folder. The app will search for the server executable.", font=("TkSmallCaptionFont"))\
            .grid(column=1, row=r+1, columnspan=3, sticky="w", padx=5); r += 2

        ttk.Label(inner, text="Virtual Environment (optional):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Entry(inner, textvariable=self.venv_dir, width=50)\
            .grid(column=1, row=r, columnspan=2, sticky="ew", padx=5, pady=3)
        vf = ttk.Frame(inner); vf.grid(column=3, row=r, sticky="w", padx=5, pady=3)
        ttk.Button(vf, text="Browse…", width=8, command=lambda: self._browse_dir(self.venv_dir)).pack(side="left", padx=2)
        ttk.Button(vf, text="Clear",   width=8, command=self._clear_venv).pack(side="left", padx=2)
        ttk.Label(inner, text="If set, activates this Python venv before launching.", font=("TkSmallCaptionFont"))\
            .grid(column=1, row=r+1, columnspan=3, sticky="w", padx=5); r += 2

        ttk.Label(inner, text="Model Search Directories:")\
            .grid(column=0, row=r, sticky="nw", padx=10, pady=3)
        dir_list_frame = ttk.Frame(inner)
        dir_list_frame.grid(column=1, row=r, sticky="nsew", padx=5, pady=3, rowspan=2)
        dir_sb = ttk.Scrollbar(dir_list_frame, orient=tk.VERTICAL)
        self.model_dirs_listbox = tk.Listbox(dir_list_frame, listvariable=self.model_dirs_listvar,
                                             height=4, width=48,
                                             yscrollcommand=dir_sb.set, exportselection=False)
        dir_sb.config(command=self.model_dirs_listbox.yview)
        self.model_dirs_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        r += 1 # Advance row count after adding the listbox frame

        inner.rowconfigure(r-1, weight=0) # Don't expand the directory listbox row with height

        dir_btn_frame = ttk.Frame(inner)
        dir_btn_frame.grid(column=2, row=r-1, sticky="ew", padx=5, pady=3, rowspan=2)
        ttk.Button(dir_btn_frame, text="Add Dir…", width=10, command=self._add_model_dir)\
           .pack(side=tk.TOP, pady=2, fill=tk.X)
        ttk.Button(dir_btn_frame, text="Remove Dir", width=10, command=self._remove_model_dir)\
           .pack(side=tk.TOP, pady=2, fill=tk.X)


        ttk.Label(inner, text="Select Model:")\
            .grid(column=0, row=r, sticky="nw", padx=10, pady=(10, 3))
        model_select_frame = ttk.Frame(inner)
        model_select_frame.grid(column=1, row=r, columnspan=2, sticky="nsew", padx=5, pady=(10,0))
        inner.rowconfigure(r, weight=1) # This row contains the model listbox, allow it to expand
        model_list_sb = ttk.Scrollbar(model_select_frame, orient=tk.VERTICAL)
        self.model_listbox = tk.Listbox(model_select_frame,
                                        height=self.app_settings.get("model_list_height", 8),
                                        width=48,
                                        yscrollcommand=model_list_sb.set,
                                        exportselection=False,
                                        state=tk.DISABLED)
        model_list_sb.config(command=self.model_listbox.yview)
        self.model_listbox.bind("<<ListboxSelect>>", self._on_model_selected) # Bind select AFTER creation
        model_list_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.model_listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.resize_grip = ttk.Separator(model_select_frame, orient=tk.HORIZONTAL, cursor="sb_v_double_arrow")
        self.resize_grip.pack(side=tk.BOTTOM, fill=tk.X, pady=(2,0))
        self.resize_grip.bind("<ButtonPress-1>", self._start_resize)
        self.resize_grip.bind("<B1-Motion>", self._do_resize)
        self.resize_grip.bind("<ButtonRelease-1>", self._end_resize)

        scan_btn = ttk.Button(inner, text="Scan Models", command=self._trigger_scan)
        scan_btn.grid(column=3, row=r, sticky="n", padx=5, pady=(10,3))
        r += 1
        self.scan_status_label = ttk.Label(inner, textvariable=self.scan_status_var, foreground="grey", font=("TkSmallCaptionFont"))
        self.scan_status_label.grid(column=1, row=r, columnspan=3, sticky="nw", padx=5, pady=(2,5));
        r += 1

        ttk.Label(inner, text="Basic Settings", font=("TkDefaultFont", 12, "bold"))\
            .grid(column=0, row=r, sticky="w", padx=10, pady=(20,5)); r += 1
        ttk.Separator(inner, orient="horizontal")\
            .grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=5); r += 1

        # --- Threads ---
        ttk.Label(inner, text="Threads (--threads):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Entry(inner, textvariable=self.threads, width=10)\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3)
        # Recommendation label for threads
        ttk.Label(inner, textvariable=self.recommended_threads_var, font=("TkSmallCaptionFont")).grid(column=2, row=r, columnspan=2, sticky="w", padx=5, pady=3); r += 1
        ttk.Label(inner, text="Number of main threads. (llama.cpp default is often logical cores)", font=("TkSmallCaptionFont"))\
            .grid(column=1, row=r+1, columnspan=3, sticky="w", padx=5, pady=(0,3)); r += 1

        # --- Context Size ---
        ttk.Label(inner, text="Context Size:")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ctx_f = ttk.Frame(inner); ctx_f.grid(column=1, row=r, columnspan=3, sticky="ew", padx=5, pady=3)
        self.ctx_label = ttk.Label(ctx_f, text=f"{self.ctx_size.get():,}", width=9, anchor='e')
        self.ctx_entry = ttk.Entry(ctx_f, width=9)
        self.ctx_entry.insert(0, str(self.ctx_size.get())) # Initial value
        self.ctx_entry.bind("<FocusOut>", self._override_ctx_size)
        self.ctx_entry.bind("<Return>", self._override_ctx_size)

        ctx_slider = ttk.Scale(ctx_f, from_=1024, to=131072, orient="horizontal",
                               variable=self.ctx_size, command=self._update_ctx_label_from_slider)
        ctx_slider.grid(column=0, row=0, sticky="ew", padx=(0, 5))
        self.ctx_label.grid(column=1, row=0, padx=5)
        self.ctx_entry.grid(column=2, row=0, padx=5)
        ttk.Button(ctx_f, text="Set", command=self._override_ctx_size, width=4).grid(column=3, row=0, padx=(0, 5))
        # Setting the variable triggers the linked slider
        self.ctx_size.set(self.ctx_size.get()) # Sync display initially
        ctx_f.columnconfigure(0, weight=3) # Slider expands
        ttk.Label(inner, text="Prompt context size in tokens (-c)", font=("TkSmallCaptionFont"))\
            .grid(column=1, row=r+1, columnspan=3, sticky="w", padx=5); r += 2

        # --- Temperature ---
        ttk.Label(inner, text="Temperature:")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Entry(inner, textvariable=self.temperature, width=10)\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, text="Controls randomness (llama.cpp default: 0.8)", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r, columnspan=2, sticky="w", padx=5, pady=3); r += 1

        # --- Min P ---
        ttk.Label(inner, text="Min P:")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Entry(inner, textvariable=self.min_p, width=10)\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, text="Minimum probability sampling (llama.cpp default: 0.05)", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r, columnspan=2, sticky="w", padx=5, pady=3); r += 1

        # --- Seed ---
        ttk.Label(inner, text="Seed:")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Entry(inner, textvariable=self.seed, width=10)\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, text="RNG seed (-1 for random)", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r, columnspan=2, sticky="w", padx=5, pady=3); r += 1

        # --- Network Settings ---
        ttk.Label(inner, text="Network Settings", font=("TkDefaultFont", 12, "bold"))\
            .grid(column=0, row=r, sticky="w", padx=10, pady=(20,5)); r += 1
        ttk.Separator(inner, orient="horizontal")\
            .grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=5); r += 1
        ttk.Label(inner, text="Host IP:")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Entry(inner, textvariable=self.host, width=20)\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, text="IP address to listen on (llama.cpp default: 127.0.0.1)", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r, columnspan=2, sticky="w", padx=5, pady=3); r += 1
        ttk.Label(inner, text="Port:")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Entry(inner, textvariable=self.port, width=10)\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, text="Port to listen on (llama.cpp default: 8080)", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r, columnspan=2, sticky="w", padx=5, pady=3)

        # Make column 2 (info/small font text) expand slightly if needed, but column 1 for entry/listbox is primary
        inner.columnconfigure(2, weight=1)

    # ═════════════════════════════════════════════════════════════════
    #  UI action handlers (e.g., browse, clear)
    # ═════════════════════════════════════════════════════════════════
    def _browse_dir(self, var):
        """Opens a directory chooser and sets the given StringVar."""
        # Get the current value to use as initial directory
        current_dir = var.get().strip()
        initial_dir = current_dir if current_dir and Path(current_dir).is_dir() else str(Path.home())

        directory = filedialog.askdirectory(title="Select Directory", initialdir=initial_dir)
        if directory:
            try:
                 # Resolve the path before setting to handle symlinks etc.
                 resolved_path = Path(directory).resolve()
                 var.set(str(resolved_path))
            except Exception as e:
                 print(f"Error resolving path '{directory}': {e}", file=sys.stderr)
                 # Fallback to setting the raw path if resolving fails
                 var.set(directory)

    def _clear_venv(self):
        """Clears the virtual environment directory entry."""
        self.venv_dir.set("")
        # Setting the variable automatically triggers the trace, which calls _check_venv_flash_attn()

    def _override_ctx_size(self, event=None):
        """Manually set context size from entry."""
        try:
            value = int(self.ctx_entry.get())
            # Clamp value to a reasonable range (e.g., 1024 to 131072, matching slider)
            clamped_value = max(1024, min(131072, value))
            self.ctx_size.set(clamped_value)
            self._sync_ctx_display(clamped_value) # Update entry/label to clamped value
        except ValueError:
            # Revert entry and label to current valid value if input is invalid
            current_value = self.ctx_size.get()
            self._sync_ctx_display(current_value)
            messagebox.showwarning("Invalid Input", "Context size must be an integer.")

    def _update_ctx_label_from_slider(self, value_str):
        """Callback for slider to update label and entry."""
        try:
            # Slider value comes as a float string, convert to int
            value = int(float(value_str))
            self._sync_ctx_display(value)
        except ValueError:
            pass # Should not happen with slider


    def _sync_ctx_display(self, value):
        """Syncs the context size label and entry to a given integer value."""
        if hasattr(self, 'ctx_label') and self.ctx_label.winfo_exists():
            self.ctx_label.config(text=f"{value:,}")
        if hasattr(self, 'ctx_entry') and self.ctx_entry.winfo_exists():
            # Update entry text without triggering its FocusOut/Return binding
            # Temporarily unbind, update, rebind
            self.ctx_entry.unbind("<FocusOut>")
            self.ctx_entry.unbind("<Return>")
            self.ctx_entry.delete(0, tk.END)
            self.ctx_entry.insert(0, str(value))
            self.ctx_entry.bind("<FocusOut>", self._override_ctx_size)
            self.ctx_entry.bind("<Return>", self._override_ctx_size)

    # ░░░░░ ADVANCED TAB ░░░░░
    def _setup_advanced_tab(self, parent):
        canvas = tk.Canvas(parent, highlightthickness=0); vs = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner  = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(yscrollcommand=vs.set,
                                                             scrollregion=canvas.bbox("all")))
        canvas_window = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))
        canvas.pack(side="left", fill="both", expand=True); vs.pack(side="right", fill="y")

        inner.columnconfigure(1, weight=1) # Make relevant columns expandable

        r = 0
        # --- Model Information Display ---
        ttk.Label(inner, text="Model Information", font=("TkDefaultFont", 12, "bold"))\
            .grid(column=0, row=r, sticky="w", padx=10, pady=(10,5)); r += 1
        ttk.Separator(inner, orient="horizontal")\
            .grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=5); r += 1

        ttk.Label(inner, text="Architecture:")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Label(inner, textvariable=self.model_architecture_var)\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3, columnspan=3); r += 1

        ttk.Label(inner, text="File Size:")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Label(inner, textvariable=self.model_filesize_var)\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3, columnspan=3); r += 1

        ttk.Label(inner, text="Total Layers:")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Label(inner, textvariable=self.model_total_layers_var)\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3, columnspan=3); r += 1

        ttk.Label(inner, text="Current KV Cache Type:")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Label(inner, textvariable=self.model_kv_cache_type_var)\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3, columnspan=3); r += 1


        # --- System Info Display (RAM & CPU) ---
        r += 1 # Add a small gap
        ttk.Label(inner, text="System Information", font=("TkDefaultFont", 12, "bold"))\
            .grid(column=0, row=r, sticky="w", padx=10, pady=(10,5)); r += 1
        ttk.Separator(inner, orient="horizontal")\
            .grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=5); r += 1

        if self.ram_info and "error" not in self.ram_info:
            ttk.Label(inner, text="System RAM (Total):", font=("TkSmallCaptionFont", 8, ("bold",)))\
                .grid(column=0, row=r, sticky="w", padx=10, pady=(5,0))
            ttk.Label(inner, text=f"{self.ram_info.get('total_ram_gb', 'N/A')} GB", font="TkSmallCaptionFont")\
                .grid(column=1, row=r, sticky="w", padx=5, pady=(5,0), columnspan=3)
            r += 1
            ttk.Label(inner, text="System RAM (Available):", font=("TkSmallCaptionFont", 8, ("bold",)))\
                .grid(column=0, row=r, sticky="w", padx=10, pady=(0,5))
            ttk.Label(inner, text=f"{self.ram_info.get('available_ram_gb', 'N/A')} GB", font="TkSmallCaptionFont")\
                .grid(column=1, row=r, sticky="w", padx=5, pady=(0,5), columnspan=3)
            r += 1
        elif self.ram_info.get("error"):
             ttk.Label(inner, text=f"System RAM Info: {self.ram_info['error']}", font="TkSmallCaptionFont", foreground="red")\
                .grid(column=0, row=r, sticky="w", padx=10, pady=3, columnspan=4); r += 1
             r += 1 # Add spacing after error


        if self.cpu_info and "error" not in self.cpu_info:
            ttk.Label(inner, text="System CPU (Logical Cores):", font=("TkSmallCaptionFont", 8, ("bold",)))\
                .grid(column=0, row=r, sticky="w", padx=10, pady=(5,0))
            ttk.Label(inner, text=f"{self.cpu_info.get('logical_cores', 'N/A')}", font="TkSmallCaptionFont")\
                .grid(column=1, row=r, sticky="w", padx=5, pady=(5,0), columnspan=3)
            r += 1
            ttk.Label(inner, text="System CPU (Physical Cores):", font=("TkSmallCaptionFont", 8, ("bold",)))\
                .grid(column=0, row=r, sticky="w", padx=10, pady=(0,10))
            ttk.Label(inner, text=f"{self.cpu_info.get('physical_cores', 'N/A')}", font="TkSmallCaptionFont")\
                .grid(column=1, row=r, sticky="w", padx=5, pady=(0,10), columnspan=3)
            r += 1
        elif self.cpu_info.get("error"):
            ttk.Label(inner, text=f"System CPU Info: {self.cpu_info['error']}", font="TkSmallCaptionFont", foreground="red")\
                .grid(column=0, row=r, sticky="w", padx=10, pady=3, columnspan=4); r += 1


        # --- GPU Settings ---
        ttk.Label(inner, text="GPU Settings", font=("TkDefaultFont", 12, "bold"))\
            .grid(column=0, row=r, sticky="w", padx=10, pady=(20,5)); r += 1
        ttk.Separator(inner, orient="horizontal")\
            .grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=5); r += 1

        # CUDA Devices Info & Checkboxes
        # Use the status variable for the main message
        gpu_avail_text = "Available" if self.gpu_info['available'] else "Not available"
        ttk.Label(inner, text=f"CUDA Devices ({gpu_avail_text}):")\
            .grid(column=0, row=r, sticky="nw", padx=10, pady=3)
        # Display detection message below if available
        ttk.Label(inner, textvariable=self.gpu_detected_status_var, font=("TkSmallCaptionFont"), foreground="orange")\
             .grid(column=1, row=r, sticky="nw", padx=5, pady=3, columnspan=3)

        r += 1
        # Checkbox frame row
        self.gpu_checkbox_frame = ttk.Frame(inner)
        self.gpu_checkbox_frame.grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=(0, 5))
        # Checkboxes are populated dynamically in _update_gpu_checkboxes()
        r += 1

        # Display VRAM for each GPU in a separate row below checkboxes
        self._display_gpu_vram_info(inner, r)
        r += 1 # Advance row count past VRAM info

        # --- GPU Layers (Slider + Entry + Status Label) ---
        ttk.Label(inner, text="GPU Layers (--n-gpu-layers):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)

        gpu_layers_frame = ttk.Frame(inner)
        gpu_layers_frame.grid(column=1, row=r, columnspan=3, sticky="ew", padx=5, pady=3)

        self.n_gpu_layers_entry = ttk.Entry(gpu_layers_frame, textvariable=self.n_gpu_layers, width=6)
        self.n_gpu_layers_entry.grid(column=0, row=0, sticky="w", padx=(0, 10))
        # Validation command and bindings set in __init__

        self.gpu_layers_slider = ttk.Scale(gpu_layers_frame, from_=0, to=self.max_gpu_layers.get(),
                                           orient="horizontal", variable=self.n_gpu_layers_int,
                                           command=self._sync_gpu_layers_from_slider, state=tk.DISABLED)
        self.gpu_layers_slider.grid(column=1, row=0, sticky="ew", padx=5)

        self.gpu_layers_status_label = ttk.Label(gpu_layers_frame, textvariable=self.gpu_layers_status_var, width=35, anchor="w")
        self.gpu_layers_status_label.grid(column=2, row=0, sticky="w", padx=(10, 0))

        gpu_layers_frame.columnconfigure(1, weight=1) # Slider expands

        ttk.Label(inner, text="Layers to offload (0=CPU only, -1=All). Slider range updates with model analysis.", font=("TkSmallCaptionFont"))\
            .grid(column=1, row=r+1, columnspan=3, sticky="w", padx=5); r += 2

        # --- Tensor Split ---
        ttk.Label(inner, text="Tensor Split (--tensor-split):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Entry(inner, textvariable=self.tensor_split, width=25)\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, text="e.g., '3,1' splits layers 75%/25% across GPUs 0 and 1.", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r, columnspan=2, sticky="w", padx=5, pady=3); r += 1
        # Recommendation display for Tensor Split
        ttk.Label(inner, text="Recommended Split (VRAM-based):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Label(inner, textvariable=self.recommended_tensor_split_var)\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3, columnspan=3); r += 1


        # --- Main GPU ---
        # Re-position Main GPU setting here, as it's related to multi-GPU
        ttk.Label(inner, text="Main GPU (--main-gpu):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Entry(inner, textvariable=self.main_gpu, width=10)\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, text="Primary GPU index (usually 0). Used with --n-gpu-layers when tensor-split is not set.", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r, columnspan=2, sticky="w", padx=5, pady=3); r += 1


        # --- Flash Attention ---
        ttk.Label(inner, text="Flash Attention (--flash-attn):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Checkbutton(inner, variable=self.flash_attn)\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, text="Use Flash Attention kernel (CUDA only, requires specific build)", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r, sticky="w", padx=5, pady=3);
        # This label shows the status based on the venv check
        self.flash_attn_status_label = ttk.Label(inner, textvariable=self.flash_attn_status_var, font=("TkSmallCaptionFont", 8, ("bold",)))
        self.flash_attn_status_label.grid(column=3, row=r, sticky="w", padx=5, pady=3); r += 1


        # --- Memory & Cache settings ---
        ttk.Label(inner, text="Memory & Cache", font=("TkDefaultFont", 12, "bold"))\
            .grid(column=0, row=r, sticky="w", padx=10, pady=(20,5)); r += 1
        ttk.Separator(inner, orient="horizontal")\
            .grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=5); r += 1
        ttk.Label(inner, text="KV Cache Type (--cache-type-k):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        # Link this combobox variable change to update the Model Info section
        ttk.Combobox(inner, textvariable=self.cache_type_k, width=10, state="readonly",
                     values=("f16","f32","q8_0","q4_0","q4_1","q5_0","q5_1"))\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, text="Quantization for KV cache (f16 is default, lower Q=more memory saved)", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r, columnspan=2, sticky="w", padx=5, pady=3); r += 1

        ttk.Label(inner, text="Disable mmap (--no-mmap):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Checkbutton(inner, variable=self.no_mmap)\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, text="Load entire model into RAM instead of mapping file", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r, columnspan=2, sticky="w", padx=5, pady=3); r += 1
        ttk.Label(inner, text="Lock in RAM (--mlock):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Checkbutton(inner, variable=self.mlock)\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, text="Prevent swapping model/KV cache (may require permissions)", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r, columnspan=2, sticky="w", padx=5, pady=3); r += 1
        ttk.Label(inner, text="Disable KV Offload (--no-kv-offload):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Checkbutton(inner, variable=self.no_kv_offload)\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, text="Keep KV cache in CPU RAM even with GPU layers", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r, columnspan=2, sticky="w", padx=5, pady=3); r += 1


        # --- Performance (Batching & Threading) ---
        ttk.Label(inner, text="Performance Settings", font=("TkDefaultFont", 12, "bold"))\
            .grid(column=0, row=r, sticky="w", padx=10, pady=(20,5)); r += 1
        ttk.Separator(inner, orient="horizontal")\
            .grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=5); r += 1

        # Threads (--threads) moved to Basic Settings now

        ttk.Label(inner, text="Prompt Batch Size (--batch-size):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Entry(inner, textvariable=self.batch_size, width=10)\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, text="Number of tokens to process in a single batch during prompt processing (default: 512)", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r, columnspan=2, sticky="w", padx=5, pady=3); r += 1

        ttk.Label(inner, text="Unconditional Batch Size (--ubatch-size):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Entry(inner, textvariable=self.ubatch_size, width=10)\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, text="Overrides --batch-size for prompt processing; allows larger batch regardless of GPU mem.", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r, columnspan=2, sticky="w", padx=5, pady=3); r += 1

        # Threads Batch (--threads-batch)
        ttk.Label(inner, text="Batch Threads (--threads-batch):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Entry(inner, textvariable=self.threads_batch, width=10)\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, textvariable=self.recommended_threads_batch_var, font=("TkSmallCaptionFont")).grid(column=2, row=r, columnspan=2, sticky="w", padx=5, pady=3); r += 1
        ttk.Label(inner, text="Number of threads to use for batch processing (llama.cpp default: 4)", font=("TkSmallCaptionFont"))\
            .grid(column=1, row=r+1, columnspan=3, sticky="w", padx=5, pady=(0,3)); r += 1 # Add descriptive text below reco label


        # Disable Conv Batching (--no-cnv) - Moved here as it's related to batching
        ttk.Label(inner, text="Disable Conv Batching (--no-cnv):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Checkbutton(inner, variable=self.no_cnv)\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, text="Disable convolutional batching for prompt processing", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r, columnspan=2, sticky="w", padx=5, pady=3); r += 1


        # --- Scheduling Priority --- (Already existed, just moved)
        ttk.Label(inner, text="Scheduling Priority (--prio):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Combobox(inner, textvariable=self.prio, width=10, values=("0","1","2","3"), state="readonly")\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, text="0=Normal, 1=Medium, 2=High, 3=Realtime (OS dependent)", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r, columnspan=2, sticky="w", padx=5, pady=3); r += 1


        # --- NEW: Generation Settings ---
        ttk.Label(inner, text="Generation Settings", font=("TkDefaultFont", 12, "bold"))\
            .grid(column=0, row=r, sticky="w", padx=10, pady=(20,5)); r += 1
        ttk.Separator(inner, orient="horizontal")\
            .grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=5); r += 1

        # --ignore-eos
        ttk.Label(inner, text="Ignore EOS Token (--ignore-eos):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Checkbutton(inner, variable=self.ignore_eos)\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, text="Don't stop generation at EOS token", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r, columnspan=2, sticky="w", padx=5, pady=3); r += 1

        # --n-predict
        ttk.Label(inner, text="Max Tokens to Predict (--n-predict):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Entry(inner, textvariable=self.n_predict, width=10)\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, text="Tokens to predict (e.g., 128, 512, -1 for unlimited/context size)", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r, columnspan=2, sticky="w", padx=5, pady=3); r += 1


        inner.columnconfigure(2, weight=1) # Allow the description columns to expand


    # ░░░░░ CHAT TEMPLATE TAB ░░░░░ (NEW)
    def _setup_chat_template_tab(self, parent):
        frame = ttk.Frame(parent, padding=10); frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1) # Allow column 1 to expand for entries/combobox

        r = 0
        ttk.Label(frame, text="Chat Template Settings", font=("TkDefaultFont", 12, "bold"))\
            .grid(column=0, row=r, columnspan=3, sticky="w", padx=5, pady=(0,5)); r += 1

        ttk.Separator(frame, orient='horizontal').grid(column=0, row=r, columnspan=3, sticky='ew', padx=5, pady=10); r += 1

        # --- Template Source Selection (Radio Buttons) ---
        ttk.Label(frame, text="Select Template Source:").grid(column=0, row=r, sticky="w", padx=5, pady=3); r += 1

        radio_frame = ttk.Frame(frame)
        radio_frame.grid(column=0, row=r, columnspan=3, sticky="w", padx=5, pady=3); r += 1

        # Use Predefined Radio Button
        ttk.Radiobutton(radio_frame, text="Use Predefined Template:",
                        variable=self.use_custom_template, value=False)\
            .pack(side="left", padx=(0, 10))

        # Use Custom Radio Button
        ttk.Radiobutton(radio_frame, text="Use Custom Template:",
                        variable=self.use_custom_template, value=True)\
            .pack(side="left")


        # --- Predefined Template Dropdown ---
        ttk.Label(frame, text="Predefined Template:").grid(column=0, row=r, sticky="w", padx=5, pady=3)
        self.predefined_template_combobox = ttk.Combobox(frame, textvariable=self.predefined_template_name,
                                                        values=list(self._predefined_templates.keys()),
                                                        state="readonly") # Readonly combobox
        self.predefined_template_combobox.grid(column=1, row=r, sticky="ew", padx=5, pady=3, columnspan=2)
        # Binding for combobox selection is handled by trace on self.predefined_template_name

        r += 1 # Next row


        # --- Custom Template Entry ---
        ttk.Label(frame, text="Custom Template String (--chat-template):")\
            .grid(column=0, row=r, sticky="nw", padx=5, pady=3)
        # Use a ScrolledText for potentially long custom templates
        self.custom_template_entry = scrolledtext.ScrolledText(frame,
                                                               wrap=tk.WORD, # Wrap words
                                                               height=8,     # Height in lines
                                                               width=60,     # Width in characters (approx)
                                                               relief=tk.SUNKEN,
                                                               bd=1)
        # Set initial value from StringVar
        self.custom_template_entry.insert(tk.END, self.custom_template_string.get())
        # Bind events to sync ScrolledText content with StringVar
        self.custom_template_entry.bind('<<Modified>>', self._on_custom_template_modified)
        self.custom_template_string.trace_add("write", lambda *args: self._update_custom_template_text()) # Update text widget if string var changes

        self.custom_template_entry.grid(column=1, row=r, sticky="nsew", padx=5, pady=3, columnspan=2)
        frame.rowconfigure(r, weight=1) # Allow text area row to expand
        r += 1


        # --- Effective Template Display ---
        ttk.Label(frame, text="Effective Template:").grid(column=0, row=r, sticky="w", padx=5, pady=3)
        # Display the currently active template string (read-only)
        # Use a disabled Entry or read-only ScrolledText
        self.effective_template_display = ttk.Entry(frame, textvariable=self.current_template_display,
                                                    state="readonly")
        self.effective_template_display.grid(column=1, row=r, sticky="ew", padx=5, pady=3)
        ttk.Button(frame, text="Copy", command=self._copy_template_display)\
            .grid(column=2, row=r, sticky="w", padx=5, pady=3)

        r += 1


        ttk.Label(frame, text="Enter a Go-template string. e.g., \"### Instruction:\\n{{instruction}}\\n### Response:\\n{{response}}\"", font=("TkSmallCaptionFont"))\
            .grid(column=1, row=r, columnspan=2, sticky="w", padx=5, pady=(0,3)); r += 1
        ttk.Label(frame, text="Use double backslashes (\\\\) for newline characters within the template string.", font=("TkSmallCaptionFont"), foreground="orange")\
             .grid(column=1, row=r, columnspan=2, sticky="w", padx=5, pady=(0,3)); r += 1


        # Initial state update based on self.use_custom_template
        self._update_template_controls_state()


    # ░░░░░ CONFIG TAB ░░░░░
    def _setup_config_tab(self, parent):
        frame = ttk.Frame(parent, padding=10); frame.pack(fill="both", expand=True)
        r = 0
        ttk.Label(frame, text="Save/Load Launch Configurations", font=("TkDefaultFont", 12, "bold"))\
            .grid(column=0, row=r, columnspan=3, sticky="w", padx=5, pady=(0,5)); r += 1

        ttk.Label(frame, text="Configuration Name:")\
            .grid(column=0, row=r, sticky="w", padx=5, pady=3)
        ttk.Entry(frame, textvariable=self.config_name, width=30)\
            .grid(column=1, row=r, sticky="ew", padx=5, pady=3)
        ttk.Button(frame, text="Save Current Settings", command=self._save_configuration).grid(column=2, row=r, padx=5, pady=3); r += 1

        ttk.Separator(frame, orient='horizontal').grid(column=0, row=r, columnspan=3, sticky='ew', padx=5, pady=10); r += 1

        ttk.Label(frame, text="Saved Configurations:").grid(column=0, row=r, sticky="w", padx=5, pady=(10,3)); r += 1

        lb_frame = ttk.Frame(frame); lb_frame.grid(column=0, row=r, columnspan=3,
                                                   sticky="nsew", padx=5, pady=3)
        sb = ttk.Scrollbar(lb_frame); sb.pack(side="right", fill="y")
        self.config_listbox = tk.Listbox(lb_frame, yscrollcommand=sb.set, height=10, exportselection=False)
        self.config_listbox.pack(side="left", fill="both", expand=True)
        sb.config(command=self.config_listbox.yview); r += 1

        btns = ttk.Frame(frame); btns.grid(column=0, row=r, columnspan=3, sticky="ew", padx=5, pady=5)
        ttk.Button(btns, text="Load Selected",   command=self._load_configuration).pack(side="left", padx=5)
        ttk.Button(btns, text="Delete Selected", command=self._delete_configuration).pack(side="left", padx=5)

        frame.columnconfigure(1, weight=1); frame.rowconfigure(4, weight=1)
        self._update_config_listbox()


    # ═════════════════════════════════════════════════════════════════
    #  Listbox Resizing Logic
    # ═════════════════════════════════════════════════════════════════
    def _start_resize(self, event):
        self._resize_start_y = event.y_root
        # Ensure we get an integer height
        try:
            self._resize_start_height = int(self.model_listbox.cget("height"))
        except tk.TclError:
             self._resize_start_height = 8 # Fallback default
        self._resize_start_height = max(3, self._resize_start_height) # Minimum height

    def _do_resize(self, event):
        delta_y = event.y_root - self._resize_start_y
        # Estimate pixels per line. Use font metrics if possible, otherwise a sensible default.
        try:
             font_obj = tk.font.Font(font=self.model_listbox['font'])
             pixels_per_line = font_obj.metrics('linespace')
             if pixels_per_line <= 0: pixels_per_line = 15 # Fallback
        except Exception:
             pixels_per_line = 15 # Fallback default

        delta_lines = round(delta_y / pixels_per_line)
        new_height = max(3, min(30, self._resize_start_height + delta_lines)) # Clamp height
        if new_height != self.model_listbox.cget("height"):
            self.model_listbox.config(height=new_height)

    def _end_resize(self, event):
        final_height = int(self.model_listbox.cget("height")) # Ensure int
        if final_height != self.app_settings.get("model_list_height"):
            self.app_settings["model_list_height"] = final_height
            self._save_configs()

    # ═════════════════════════════════════════════════════════════════
    #  Model Directory and Scanning Logic
    # ═════════════════════════════════════════════════════════════════
    def _add_model_dir(self):
        # Use the last added directory or home as initial
        initial_dir = self.model_dirs[-1] if self.model_dirs and self.model_dirs[-1].is_dir() else str(Path.home())
        directory = filedialog.askdirectory(title="Select Model Directory", initialdir=initial_dir)
        if directory:
            p = Path(directory).resolve() # Resolve path to handle symlinks etc.
            if p not in self.model_dirs:
                self.model_dirs.append(p)
                self._update_model_dirs_listbox()
                self._save_configs()
                self._trigger_scan()
            else:
                messagebox.showinfo("Info", "Directory already in list.")

    def _remove_model_dir(self):
        selection = self.model_dirs_listbox.curselection()
        if not selection:
            messagebox.showerror("Error", "Select a directory to remove.")
            return
        index = selection[0]
        try:
             dir_to_remove_str = self.model_dirs_listbox.get(index)
             dir_to_remove_path = Path(dir_to_remove_str).resolve() # Resolve before comparing
             # Find the actual path object in our list
             actual_index = -1
             for i, p in enumerate(self.model_dirs):
                 if p.resolve() == dir_to_remove_path:
                     actual_index = i
                     break

             if actual_index != -1:
                if messagebox.askyesno("Confirm Remove", f"Remove directory:\n{self.model_dirs[actual_index]}?"):
                    del self.model_dirs[actual_index]
                    self._update_model_dirs_listbox()
                    self._save_configs()
                    self._trigger_scan() # Re-scan after removing a directory
             else:
                  messagebox.showerror("Error", "Selected directory path mismatch or not found internally.")

        except IndexError:
            messagebox.showerror("Error", "Invalid selection.")
        except Exception as e:
             messagebox.showerror("Error", f"An error occurred removing directory:\n{e}")


    def _update_model_dirs_listbox(self):
        current_selection = self.model_dirs_listbox.curselection()
        selected_text = self.model_dirs_listbox.get(current_selection[0]) if current_selection else None

        self.model_dirs_listbox.delete(0, tk.END)
        resolved_model_dirs_strs = []
        for p in self.model_dirs:
            try:
                 resolved_str = str(p.resolve()) # Resolve path for display and comparison
                 resolved_model_dirs_strs.append(resolved_str)
                 self.model_dirs_listbox.insert(tk.END, resolved_str)
            except Exception as e:
                 print(f"Warning: Could not resolve path '{p}': {e}", file=sys.stderr)
                 self.model_dirs_listbox.insert(tk.END, f"[Invalid Path] {p}")

        # Attempt to re-select the previously selected item based on resolved path
        if selected_text:
            try:
                new_index = resolved_model_dirs_strs.index(selected_text)
                self.model_dirs_listbox.selection_set(new_index)
                self.model_dirs_listbox.activate(new_index)
                self.model_dirs_listbox.see(new_index)
            except ValueError:
                pass # Previous selection not found in the new list

        self.model_dirs = [Path(d) for d in resolved_model_dirs_strs] # Update internal list with resolved paths


    def _trigger_scan(self):
        """Initiates model scanning in a background thread."""
        self.scan_status_var.set("Scanning...")
        self.model_listbox.config(state=tk.NORMAL)
        self.model_listbox.delete(0, tk.END)
        self.model_path.set("")
        self._reset_gpu_layer_controls()
        self._reset_model_info_display()
        self.current_model_analysis = {} # Clear analysis result
        self._update_recommendations() # Update recommendations display
        # Disable Add/Remove buttons during scan to prevent modifying the list while scanning
        # Need to find the buttons - assuming they are in the dir_btn_frame
        if hasattr(self, 'dir_btn_frame') and self.dir_btn_frame.winfo_exists():
             for child in self.dir_btn_frame.winfo_children():
                 if isinstance(child, ttk.Button):
                      child.config(state=tk.DISABLED)

        scan_thread = Thread(target=self._scan_model_dirs, daemon=True)
        scan_thread.start()

    def _scan_model_dirs(self):
        """Scans configured directories for GGUF models (runs in background thread)."""
        print("DEBUG: _scan_model_dirs thread started", file=sys.stderr)
        found = {} # {display_name: full_path_obj}
        # Pattern to match multi-part files like model-00001-of-00005.gguf or model-F1.gguf
        multipart_pattern = re.compile(r"^(.*?)(?:-\d{5}-of-\d{5}|-F\d+)\.gguf$", re.IGNORECASE)
        # Pattern to match the FIRST part of a multi-part file (e.g., model-00001-of-00005.gguf or model-F1.gguf)
        first_part_pattern = re.compile(r"^(.*?)-(?:00001-of-\d{5}|F1)\.gguf$", re.IGNORECASE)
        processed_multipart_bases = set()

        for model_dir in self.model_dirs:
            # Skip invalid or non-existent directories silently during scan
            if not model_dir.is_dir(): continue
            print(f"DEBUG: Scanning directory: {model_dir}", file=sys.stderr)
            try:
                # Use rglob for recursive search
                for gguf_path in model_dir.rglob('*.gguf'):
                    if not gguf_path.is_file(): continue
                    filename = gguf_path.name
                    # Skip non-model GGUF files often found with models
                    if "mmproj" in filename.lower() or filename.lower().endswith(".bin.gguf"):
                         continue

                    # Handle multi-part files: only list the base name, pointing to the first part
                    first_part_match = first_part_pattern.match(filename)
                    if first_part_match:
                        base_name = first_part_match.group(1)
                        if base_name not in processed_multipart_bases:
                            # Store the path to the first part
                            # Resolve the path before storing to avoid issues with relative paths later
                            found[base_name] = gguf_path.resolve()
                            processed_multipart_bases.add(base_name)
                        continue

                    # Handle subsequent parts of multi-part files: mark base as processed but don't add
                    multi_match = multipart_pattern.match(filename)
                    if multi_match:
                        base_name = multi_match.group(1)
                        processed_multipart_bases.add(base_name)
                        continue

                    # Handle single-part files (not matching the multi-part patterns)
                    if filename.lower().endswith(".gguf") and gguf_path.stem not in processed_multipart_bases:
                         display_name = gguf_path.stem
                         # Only add if we haven't already added a multi-part version with the same base name
                         if display_name not in found:
                            # Resolve the path before storing
                            found[display_name] = gguf_path.resolve()

            except Exception as e:
                print(f"ERROR: Error scanning directory {model_dir}: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

        self.root.after(0, self._update_model_listbox_after_scan, found)

    # ═════════════════════════════════════════════════════════════════
    #  Model Selection & Analysis
    # ═════════════════════════════════════════════════════════════════

    def _update_model_listbox_after_scan(self, found_models_dict):
        """Populates the model listbox AFTER scan and handles selection restoration."""
        self.found_models = found_models_dict
        model_names = sorted(list(self.found_models.keys()))

        self.model_listbox.config(state=tk.NORMAL)
        self.model_listbox.delete(0, tk.END)
        for name in model_names:
            self.model_listbox.insert(tk.END, name)
        self.root.update_idletasks()

        # Re-enable Add/Remove buttons after scan
        if hasattr(self, 'dir_btn_frame') and self.dir_btn_frame.winfo_exists():
             for child in self.dir_btn_frame.winfo_children():
                 if isinstance(child, ttk.Button):
                      child.config(state=tk.NORMAL)


        if not model_names:
            self.scan_status_var.set("No GGUF models found in specified directories.")
            self.model_path.set("")
            self._reset_gpu_layer_controls()
            self._reset_model_info_display()
            self.current_model_analysis = {}
            self._update_recommendations()
            self._generate_default_config_name() # Generate default name for no model state
        else:
            self.scan_status_var.set(f"Scan complete. Found {len(model_names)} models.")

            last_path_str = self.app_settings.get("last_model_path")
            selected_idx = -1
            if last_path_str:
                 try:
                     last_path_obj = Path(last_path_str).resolve() # Resolve the saved path
                     found_display_name = None
                     # Find the display name associated with the saved path in the *resolved* found_models
                     for display_name, full_path in self.found_models.items():
                         if full_path == last_path_obj:
                             found_display_name = display_name
                             break
                     # If found, get its index in the current listbox (which is sorted model_names)
                     if found_display_name:
                         selected_idx = model_names.index(found_display_name)
                 except (ValueError, OSError, IndexError): # Add IndexError just in case listbox is empty
                     pass # Handle potential errors with old paths or empty listbox

            # If no previous path or path invalid/not found, select the first model
            if selected_idx == -1 and model_names:
                selected_idx = 0

            if selected_idx != -1:
                # Use after() to allow UI update before selection/analysis
                self.root.after(50, self._select_model_in_listbox, selected_idx)
            else:
                 # No models at all, or no valid selection possible
                 self.model_path.set("") # Ensure model path is clear
                 self.app_settings["last_model_path"] = ""
                 self._save_configs() # Save empty path
                 self._reset_gpu_layer_controls(keep_entry_enabled=True) # Keep entry usable if no model info
                 self._reset_model_info_display()
                 self.current_model_analysis = {}
                 self._update_recommendations()
                 self._generate_default_config_name() # Generate default name for no model state


    def _select_model_in_listbox(self, index):
         """Selects a specific index in the model listbox."""
         try:
            if 0 <= index < self.model_listbox.size():
                 self.model_listbox.selection_clear(0, tk.END)
                 self.model_listbox.selection_set(index)
                 self.model_listbox.activate(index)
                 self.model_listbox.see(index)
                 self._on_model_selected() # Trigger the selection handler
            else:
                print(f"WARN: Attempted to select invalid index {index} in model listbox.", file=sys.stderr)
                self.model_path.set("")
                self.app_settings["last_model_path"] = ""
                self._save_configs()
                self._reset_gpu_layer_controls(keep_entry_enabled=True) # Keep entry usable if no model info
                self._reset_model_info_display()
                self.current_model_analysis = {}
                self._update_recommendations()
                self._generate_default_config_name() # Generate default name for no model state
         except Exception as e:
             print(f"ERROR during _select_model_in_listbox: {e}", file=sys.stderr)
             traceback.print_exc(file=sys.stderr)
             self.model_path.set("")
             self.app_settings["last_model_path"] = ""
             self._save_configs()
             self._reset_gpu_layer_controls(keep_entry_enabled=True)
             self._reset_model_info_display()
             self.current_model_analysis = {}
             self._update_recommendations()
             self._generate_default_config_name() # Generate default name for no model state


    def _on_model_selected(self, event=None):
        """Callback when a model is selected. Triggers GGUF analysis."""
        selection = self.model_listbox.curselection()
        if not selection:
            self.model_path.set("")
            self.app_settings["last_model_path"] = ""
            self._save_configs()
            self._reset_gpu_layer_controls()
            self._reset_model_info_display()
            self.current_model_analysis = {}
            self._update_recommendations() # Update based on no model
            self._generate_default_config_name() # Generate default name for no model state
            return

        index = selection[0]
        selected_name = self.model_listbox.get(index)

        if selected_name in self.found_models:
            full_path = self.found_models[selected_name] # Use the resolved path from scan results
            full_path_str = str(full_path)
            self.model_path.set(full_path_str)
            # Save last model path immediately on selection
            self.app_settings["last_model_path"] = full_path_str
            self._save_configs()


            # Update current KV cache type display immediately
            # This is called by _update_recommendations, which is triggered below

            if LLAMA_CPP_PYTHON_AVAILABLE:
                 self.gpu_layers_status_var.set("Analyzing model...")
                 self.gpu_layers_slider.config(state=tk.DISABLED)
                 # Entry remains enabled, validated state will apply
                 self._reset_model_info_display() # Reset info fields before analysis starts
                 self.current_model_analysis = {} # Clear old analysis
                 self._update_recommendations() # Update recommendations display based on no analysis yet

                 # Start analysis thread
                 if self.analysis_thread and self.analysis_thread.is_alive():
                      print("DEBUG: Previous analysis thread is still running.", file=sys.stderr)
                 self.analysis_thread = Thread(target=self._run_gguf_analysis, args=(full_path_str,), daemon=True)
                 self.analysis_thread.start()
            else:
                 self.gpu_layers_status_var.set("Analysis requires llama-cpp-python")
                 self._reset_gpu_layer_controls(keep_entry_enabled=True) # Keep entry enabled if lib missing
                 self._reset_model_info_display()
                 self.model_architecture_var.set("Analysis Unavailable")
                 self.model_filesize_var.set("Analysis Unavailable")
                 self.model_total_layers_var.set("Analysis Unavailable")
                 self.current_model_analysis = {}
                 self._update_recommendations() # Update recommendations based on no analysis
                 self._generate_default_config_name() # Generate default name even without analysis


        else:
            print(f"WARNING: Selected model '{selected_name}' not found in self.found_models dictionary.", file=sys.stderr)
            self.model_path.set("")
            self.app_settings["last_model_path"] = ""
            self._save_configs()
            self._reset_gpu_layer_controls()
            self._reset_model_info_display()
            self.current_model_analysis = {}
            self._update_recommendations() # Update recommendations based on no model
            self._generate_default_config_name() # Generate default name for no model state


    def _run_gguf_analysis(self, model_path_str):
        """Worker function for background GGUF analysis."""
        print(f"Analyzing GGUF in background: {model_path_str}", file=sys.stderr)
        analysis_result = analyze_gguf_model_static(model_path_str)
        self.root.after(0, self._update_ui_after_analysis, analysis_result)

    # ═════════════════════════════════════════════════════════════════
    #  Model Selection & Analysis - Updated Handler
    # ═════════════════════════════════════════════════════════════════
    def _update_ui_after_analysis(self, analysis_result):
        """Updates controls based on GGUF analysis results (runs in main thread)."""
        print("DEBUG: _update_ui_after_analysis running", file=sys.stderr)
        self.current_model_analysis = analysis_result
        error = analysis_result.get("error")
        n_layers = analysis_result.get("n_layers")
        message = analysis_result.get("message")

        # --- Update Model Info Display ---
        arch = analysis_result.get("architecture", "unknown")
        file_size_gb = analysis_result.get("file_size_gb")
        self.model_architecture_var.set(arch.capitalize() if arch and arch != "unknown" else "Unknown")
        self.model_filesize_var.set(f"{file_size_gb:.2f} GB" if file_size_gb is not None and file_size_gb > 0 else "Unknown Size")
        self.model_total_layers_var.set(str(n_layers) if n_layers is not None else "Unknown")

        # --- Update GPU Layer Controls ---
        # Always ensure the entry is NORMAL here, regardless of error state.
        if hasattr(self, 'n_gpu_layers_entry') and self.n_gpu_layers_entry.winfo_exists():
             self.n_gpu_layers_entry.config(state=tk.NORMAL)

        if error or n_layers is None or n_layers <= 0:
             # Analysis failed or layer count invalid/zero
             status_msg = error if error else (message if message else "Layer count not found or invalid")
             self.gpu_layers_status_var.set(f"Analysis Error: {status_msg}" if error else status_msg)
             # Resetting controls sets max_layers to 0 and disables the slider
             self._reset_gpu_layer_controls()
             # Now, sync the entry/slider based on the existing value in the entry and max_layers=0
             self._sync_gpu_layers_from_entry()

        else: # Analysis succeeded, layers found (n_layers > 0)
            self.max_gpu_layers.set(n_layers)
            self.gpu_layers_status_var.set(f"Max Layers: {n_layers}")
            if hasattr(self, 'gpu_layers_slider') and self.gpu_layers_slider.winfo_exists():
                self.gpu_layers_slider.config(to=n_layers, state=tk.NORMAL) # Enable slider

            # Sync the controls based on the *current* value in the n_gpu_layers StringVar
            # This will set the slider and potentially update the entry format (-1 vs number)
            self._sync_gpu_layers_from_entry()

        # --- Update Recommendations based on new analysis ---
        self._update_recommendations()

        # --- Generate Default Config Name ---
        # Call this after analysis completes, as n_layers is needed for the name
        self._generate_default_config_name()

    def _reset_gpu_layer_controls(self):
         """Resets GPU layer slider state and max layers (but *not* entry StringVar)."""
         print("DEBUG: _reset_gpu_layer_controls called", file=sys.stderr)
         # Reset max_gpu_layers first
         self.max_gpu_layers.set(0)

         # Update status label
         self.gpu_layers_status_var.set("Select model to see layer info")

         # Slider range and state
         if hasattr(self, 'gpu_layers_slider') and self.gpu_layers_slider.winfo_exists():
              self.gpu_layers_slider.config(to=0, state=tk.DISABLED) # Slider disabled if max_layers is 0

         # IMPORTANT: Do NOT reset self.n_gpu_layers.get() or call _set_gpu_layers(0) here.
         # This preserves the user's last input in the entry.
         # The caller (_update_ui_after_analysis) is responsible for calling
         # _sync_gpu_layers_from_entry after this, which will then process the
         # entry's value against the *new* max_layers (which is 0).

    def _reset_model_info_display(self):
         """Resets model info labels."""
         self.model_architecture_var.set("N/A")
         self.model_filesize_var.set("N/A")
         self.model_total_layers_var.set("N/A")
         # KV Cache Type display is linked to the variable, not reset here


# ═════════════════════════════════════════════════════════════════
    #  GPU Layer Slider/Entry Synchronization & Validation
    # ═════════════════════════════════════════════════════════════════

    def _set_gpu_layers(self, input_value):
        """
        Helper to set the internal GPU layers state (int) based on user input.
        Handles clamping based on max_layers. Does NOT directly set the StringVar
        but updates the IntVar and triggers recommendations.
        input_value can be -1 or a non-negative integer.
        """
        max_layers = self.max_gpu_layers.get() # Get current max layer count

        int_val = 0 # Default clamped value

        if input_value == -1:
            # If input is -1, internal value is max_layers if max > 0, else 0
            if max_layers > 0:
                int_val = max_layers
            else:
                int_val = 0 # Cannot offload all if max is unknown/zero
        elif input_value >= 0:
            # If input is non-negative
            if max_layers > 0:
                # Clamp positive input value to max_layers if max > 0
                int_val = min(input_value, max_layers)
            else:
                int_val = 0 # If max_layers <= 0, any positive input still means 0 layers offloaded internally
        # else: input_value < -1 (should be prevented by validation) -> default to 0

        # Update the Tk integer variable (linked to slider)
        # This will also indirectly trigger _sync_gpu_layers_from_slider if trace is set,
        # but we manage string sync in the entry/slider callbacks now.
        if self.n_gpu_layers_int.get() != int_val:
            self.n_gpu_layers_int.set(int_val)

        # Trigger recommendations update (e.g., KV cache type display)
        self._update_recommendations()


    def _sync_gpu_layers_from_slider(self, value_str):
        """Callback when slider changes. Updates entry StringVar and internal state."""
        if not hasattr(self, 'n_gpu_layers_entry') or not self.n_gpu_layers_entry.winfo_exists():
            return

        try:
            # Slider value is always an integer (converted from float string) between 0 and max
            value = int(float(value_str))
            max_layers = self.max_gpu_layers.get()

            # Update internal state using the slider's integer value
            self._set_gpu_layers(value) # This sets n_gpu_layers_int and updates recommendations

            # Determine the canonical string representation for the entry based on the clamped value
            canonical_str = str(value) # Default to the integer value
            if max_layers > 0 and value == max_layers:
                canonical_str = "-1" # If slider is at max and max > 0, entry should show -1

            # Update the entry's StringVar only if it's different
            if self.n_gpu_layers.get() != canonical_str:
                 self.n_gpu_layers.set(canonical_str)

        except ValueError:
            pass # Should not happen with a slider


    def _sync_gpu_layers_from_entry(self, event=None):
        """Callback when entry loses focus or Enter is pressed. Validates and syncs state."""
        if not hasattr(self, 'n_gpu_layers_entry') or not self.n_gpu_layers_entry.winfo_exists():
            return

        current_str = self.n_gpu_layers.get().strip() # Use stripped value from entry

        # If entry is empty, treat as 0
        if current_str == "":
            current_str = "0"

        try:
            value = int(current_str)
            max_layers = self.max_gpu_layers.get()

            # Determine the internal clamped integer value based on the input value
            # This logic is now encapsulated in _set_gpu_layers
            self._set_gpu_layers(value) # Use the helper which will calculate int_val based on max_layers


            # Now, determine what the entry StringVar *should* display for consistency
            # If max_layers > 0, the entry should show -1 if the clamped value is max,
            # or the clamped integer value otherwise.
            # If max_layers <= 0, the entry should retain the user's valid input string (current_str).

            canonical_str_for_entry = current_str # Assume user input is fine initially

            if max_layers > 0:
                 # If max_layers > 0, the entry should represent the *clamped* value
                 clamped_int_from_set = self.n_gpu_layers_int.get() # Get the result of _set_gpu_layers
                 if clamped_int_from_set == max_layers:
                      canonical_str_for_entry = "-1"
                 else:
                      canonical_str_for_entry = str(clamped_int_from_set)
            # else: max_layers <= 0, keep current_str (user input)

            # Update the entry's StringVar only if it's different from the calculated canonical string
            # This prevents loops and ensures the entry displays the correct format (-1 vs number)
            # when max_layers > 0, while preserving arbitrary valid input when max_layers <= 0.
            if self.n_gpu_layers.get() != canonical_str_for_entry:
                 self.n_gpu_layers.set(canonical_str_for_entry)


        except ValueError:
            # Invalid input (should be caught by validation, but this is a fallback)
            # Revert entry to the current valid representation based on the IntVar
            print(f"DEBUG: Invalid value '{current_str}' in GPU layers entry. Reverting.", file=sys.stderr)
            max_val = self.max_gpu_layers.get()
            current_int_value = self.n_gpu_layers_int.get()
            # Determine the correct string representation from the current IntVar/max_layers
            if max_val > 0 and current_int_value == max_val:
                 # If IntVar is at max and max > 0, the string should be "-1"
                 self.n_gpu_layers.set("-1")
            else:
                # Otherwise, the string should be the integer value
                 self.n_gpu_layers.set(str(current_int_value))
            # No need to call _update_recommendations here, it was last called by _set_gpu_layers


    def _validate_gpu_layers_entry(self, proposed_value):
        """Validation function for the n_gpu_layers entry."""
        if not hasattr(self, 'max_gpu_layers'): return True # Allow during init

        pv = proposed_value.strip()

        if pv == "": return True # Allow empty string (user hasn't typed yet, or cleared)
        if pv == "-": return True # Allow just a dash (user is typing -1)

        try:
            value = int(pv)

            # Always allow -1
            if value == -1: return True

            # Do not allow negative numbers other than -1
            if value < -1: return False

            # Allow any non-negative integer if max layers is unknown or 0
            # Note: Even if max_layers is 0, allowing positive input in the entry is fine;
            # the _sync_gpu_layers_from_entry will clamp it to 0 for the internal value/slider.
            if value >= 0: return True

            # This part should not be reached if value < -1 is handled
            return False

        except ValueError:
            # Reject non-numeric input (other than "", "-")
            return False

    # ═════════════════════════════════════════════════════════════════
    #  dynamic GPU checkboxes and info display
    # ═════════════════════════════════════════════════════════════════
    def _update_gpu_checkboxes(self):
        """Updates GPU checkboxes based on detected devices and loaded config."""
        # Clear previous checkboxes
        for w in self.gpu_checkbox_frame.winfo_children(): w.destroy()
        self.gpu_vars.clear() # Clear list of BooleanVars

        count = self.gpu_info["device_count"]
        # Get selected GPUs from app settings, default to empty list if not found
        loaded_selected_gpus = set(self.app_settings.get("selected_gpus", []))
        print(f"DEBUG: Detected GPUs: {self.detected_gpu_devices}", file=sys.stderr)
        print(f"DEBUG: Loaded selected GPUs indices: {loaded_selected_gpus}", file=sys.stderr)


        if count > 0:
            for i in range(count):
                # Get info for the detected GPU device
                gpu_details = self.detected_gpu_devices[i] if i < len(self.detected_gpu_devices) else {}

                v = tk.BooleanVar(value=(i in loaded_selected_gpus)) # Set initial state from loaded config
                gpu_name_display = f"GPU {i}"
                if gpu_details and gpu_details.get("name"):
                     gpu_name_display += f": {gpu_details['name']}"

                cb = ttk.Checkbutton(self.gpu_checkbox_frame, text=gpu_name_display, variable=v)
                cb.pack(side="left", padx=3, pady=2)
                # Bind callback to checkbox changes
                # Use lambda with default argument index=i to capture the current value of i
                v.trace_add("write", lambda *args, index=i: self._on_gpu_selection_changed(index))
                self.gpu_vars.append(v)

        else:
             ttk.Label(self.gpu_checkbox_frame, text="(No CUDA devices detected)").pack(side="left", padx=5, pady=3)

        # Trigger update recommendations after setting initial checkbox states
        # This ensures the recommendation is based on the *loaded* selection
        self.root.after(10, self._update_recommendations)


    def _display_gpu_vram_info(self, parent, row):
        """Displays VRAM information for detected GPUs."""
        # Clear any previous VRAM info labels in this grid location
        # Use grid_slaves to find widgets at specific row/column
        # We need to find the frame first if it exists to clear its contents, or create it
        if not hasattr(self, '_vram_info_frame') or not self._vram_info_frame.winfo_exists():
            self._vram_info_frame = ttk.Frame(parent)
            self._vram_info_frame.grid(column=0, row=row, columnspan=4, sticky="ew", padx=10, pady=(0, 10))
            # Allow VRAM frame contents to align left but not be squeezed
            self._vram_info_frame.columnconfigure(0, weight=0)
        else:
             # Clear existing widgets within the frame
             for child in self._vram_info_frame.winfo_children():
                  child.destroy()
             # Ensure it's still on the correct row (might not be necessary if grid setup is static)
             self._vram_info_frame.grid(column=0, row=row, columnspan=4, sticky="ew", padx=10, pady=(0, 10))


        if self.gpu_info['available'] and self.gpu_info['device_count'] > 0:
            ttk.Label(self._vram_info_frame, text="VRAM (Total GB):", font=("TkSmallCaptionFont", 8, ("bold",)))\
                .pack(side="left", padx=(0, 5))

            # Display VRAM for each detected GPU
            for gpu in self.detected_gpu_devices:
                 ttk.Label(self._vram_info_frame, text=f"GPU {gpu['id']}: {gpu['total_memory_gb']:.2f} GB", font="TkSmallCaptionFont")\
                    .pack(side="left", padx=5)

            # Add a label that expands to push the GPU info left
            ttk.Label(self._vram_info_frame, text="").pack(side="left", expand=True, fill="x")

        elif self.gpu_detected_status_var.get():
             # If there was a specific error message about GPU detection, display it here instead
             ttk.Label(self._vram_info_frame, textvariable=self.gpu_detected_status_var, font="TkSmallCaptionFont", foreground="orange")\
                 .pack(side="left", padx=5, expand=True, fill="x")

        # No action needed if no GPUs, no error, and no specific message


    # ═════════════════════════════════════════════════════════════════
    #  Recommendation Logic
    # ═════════════════════════════════════════════════════════════════
    def _update_recommendations(self):
        """Updates recommended values displayed in the UI."""
        print("DEBUG: Updating recommendations...", file=sys.stderr)
        # Update KV Cache Type display (always reflects current selection)
        self.model_kv_cache_type_var.set(self.cache_type_k.get())

        # --- Threads & Threads Batch Recommendation ---
        # Reco is always the detected CPU cores, based on the user's request pattern
        self.recommended_threads_var.set(f"Recommended: {self.physical_cores} (Your CPU physical cores)")
        self.recommended_threads_batch_var.set(f"Recommended: {self.logical_cores} (Your CPU logical cores)")


        # --- Tensor Split Recommendation (VRAM-based) ---
        n_layers = self.current_model_analysis.get("n_layers")
        selected_gpu_indices = [i for i, v in enumerate(self.gpu_vars) if v.get()]
        num_selected_gpus = len(selected_gpu_indices)

        if n_layers is None or n_layers <= 0:
            self.recommended_tensor_split_var.set("N/A - Model layers unknown")
        elif num_selected_gpus <= 1:
            self.recommended_tensor_split_var.set("N/A - Need >1 GPU selected")
        elif not self.gpu_info['available'] or not self.detected_gpu_devices:
             self.recommended_tensor_split_var.set("N/A - CUDA not available or GPUs not detected")
        else:
            # Gather VRAM for *selected* GPUs, maintaining their order
            selected_gpu_vram = []
            vram_info_found = True
            for gpu_id in selected_gpu_indices:
                 # Find the VRAM for this GPU ID in the detected list
                 gpu_details = next((gpu for gpu in self.detected_gpu_devices if gpu['id'] == gpu_id), None)
                 if gpu_details and gpu_details.get('total_memory_gb') is not None:
                      selected_gpu_vram.append((gpu_id, gpu_details['total_memory_gb']))
                 else:
                      print(f"WARNING: VRAM info missing for selected GPU ID {gpu_id}. Cannot calculate VRAM-based split.", file=sys.stderr)
                      self.recommended_tensor_split_var.set("N/A - VRAM info missing for selected GPU(s)")
                      vram_info_found = False
                      break # Cannot proceed with VRAM-based split if info is missing

            if not vram_info_found or not selected_gpu_vram: # Should be covered by above, but defensive
                 return # Exit if VRAM info is incomplete

            total_selected_vram = sum(vram for gpu_id, vram in selected_gpu_vram)
            if total_selected_vram == 0:
                 self.recommended_tensor_split_var.set("N/A - Selected GPUs have 0 total VRAM")
                 return

            # Calculate approximate layer distribution (float values) based on VRAM proportion
            float_layers_per_gpu = {gpu_id: n_layers * (vram / total_selected_vram) for gpu_id, vram in selected_gpu_vram}

            # Calculate rounded layer distribution (integer values)
            rounded_layers = {gpu_id: math.floor(layers) for gpu_id, layers in float_layers_per_gpu.items()}
            current_total_layers = sum(rounded_layers.values())
            remainder = n_layers - current_total_layers

            # Distribute the remainder layers to GPUs with the largest fractional parts
            # Sort GPUs by fractional part in descending order, based on selected_gpu_indices order
            # This ensures distribution bias follows the order the user selected, which might influence llama.cpp's internal assignment order
            fractional_parts_in_order = [(gpu_id, float_layers_per_gpu[gpu_id] - rounded_layers[gpu_id]) for gpu_id in selected_gpu_indices]
            fractional_parts_sorted = sorted(fractional_parts_in_order, key=lambda item: item[1], reverse=True)

            # Use a list to track which GPU IDs have received an extra layer to handle remainders efficiently
            gpu_ids_receiving_extra = [item[0] for item in fractional_parts_sorted[:remainder]]


            for gpu_id_to_add_layer in gpu_ids_receiving_extra:
                 rounded_layers[gpu_id_to_add_layer] += 1


            # Ensure the final split list is in the order of *selected* GPU indices
            recommended_split_list = [rounded_layers[gpu_id] for gpu_id in selected_gpu_indices]

            # Sanity check: does the sum of recommended layers equal total layers?
            if sum(recommended_split_list) != n_layers:
                 print(f"WARNING: Tensor split calculation error. Sum ({sum(recommended_split_list)}) != Total Layers ({n_layers})", file=sys.stderr)
                 # Fallback to an equal split if calculation failed
                 layers_per_gpu_equal = n_layers // num_selected_gpus
                 remainder_equal = n_layers % num_selected_gpus
                 equal_split_list = [layers_per_gpu_equal + (1 if i < remainder_equal else 0) for i in range(num_selected_gpus)]
                 recommended_split_str = "Calculation Error, try: " + ",".join(map(str, equal_split_list))
                 self.recommended_tensor_split_var.set(recommended_split_str)
            else:
                recommended_split_str = ",".join(map(str, recommended_split_list))
                self.recommended_tensor_split_var.set(recommended_split_str)

        # --- GPU Layers Recommendation ---
        # Displaying "Max Layers" from analysis serves as the recommendation for the upper bound.
        # A VRAM-based recommendation for the *number* of layers to offload is too complex to be accurate
        # without specific model layer sizes and precise VRAM usage estimations, which are not easily
        # available via llama-cpp-python metadata or simple system queries.
        # The existing status label next to the slider is sufficient to show the maximum.


    # ═════════════════════════════════════════════════════════════════
    #  Venv Flash Attention Check
    # ═════════════════════════════════════════════════════════════════

    # This small script code will be executed in a subprocess
    _FLASH_ATTN_CHECK_SCRIPT = """
import sys
try:
    import flash_attn
    # Optional: Check if flash_attn has a CUDA implementation (though typically it does if import succeeds)
    # import torch
    # if not hasattr(flash_attn, 'flash_attn_func') or not torch.cuda.is_available():
    #     print("FLASH_ATTN_NOT_INSTALLED") # Consider it 'not installed' if CUDA part isn't there
    # else:
    print("FLASH_ATTN_INSTALLED")
except ImportError:
    print("FLASH_ATTN_NOT_INSTALLED")
except Exception as e:
    print(f"FLASH_ATTN_CHECK_ERROR: {e}")
    sys.exit(1) # Indicate failure
sys.exit(0) # Indicate success
"""

    def _check_venv_flash_attn(self):
        """Checks if flash_attn is installed in the specified virtual environment."""
        venv_path_str = self.venv_dir.get().strip()

        if not venv_path_str:
            # If no venv is set, report status based on the GUI's environment
            gui_status = "Installed (GUI env, requires llama.cpp build support)" if FLASH_ATTN_GUI_AVAILABLE else "Not Installed (Python package missing in GUI env)"
            # Add note about venv if CUDA is available but no venv is set
            if self.gpu_info.get('available') and not FLASH_ATTN_GUI_AVAILABLE:
                 gui_status += " - Consider setting a Venv with Flash Attention installed."
            self.flash_attn_status_var.set(f"Status: {gui_status}")
            return

        venv_path = Path(venv_path_str)
        if not venv_path.is_dir():
             self.flash_attn_status_var.set("Status: Venv path invalid or not a directory.")
             return

        # Find the Python executable inside the venv
        if sys.platform == "win32":
            python_exe = venv_path / "Scripts" / "python.exe"
        else: # Linux/macOS
            python_exe = venv_path / "bin" / "python"

        if not python_exe.is_file():
             self.flash_attn_status_var.set("Status: Python executable not found in Venv 'Scripts' or 'bin'.")
             return

        # Update status display while checking
        self.flash_attn_status_var.set("Status: Checking Venv...")
        self.root.update_idletasks() # Update the GUI immediately

        def run_check():
            """Runs the subprocess check (in a separate thread)."""
            try:
                # Run the python executable in the venv with the check script code
                # Set a timeout to prevent hanging if something goes wrong
                # Use the full path to the python executable
                process = subprocess.run([str(python_exe), "-c", self._FLASH_ATTN_CHECK_SCRIPT],
                                         capture_output=True, text=True, timeout=10) # 10 sec timeout

                stdout = process.stdout.strip()
                stderr = process.stderr.strip()

                status_message = "Status: Unknown error during check."
                if process.returncode == 0:
                    if "FLASH_ATTN_INSTALLED" in stdout:
                        status_message = "Status (Venv): Installed (requires llama.cpp build support)"
                    elif "FLASH_ATTN_NOT_INSTALLED" in stdout:
                        status_message = "Status (Venv): Not Installed (Python package missing)"
                    else:
                        status_message = f"Status (Venv): Unexpected output. Return code 0. Output: {stdout} | Error: {stderr}"
                elif process.returncode != 0:
                    # Check stdout even on non-zero return code, as the script might print status then exit with 1 on non-zero
                    if "FLASH_ATTN_NOT_INSTALLED" in stdout:
                         status_message = "Status (Venv): Not Installed (Python package missing)"
                    elif "FLASH_ATTN_CHECK_ERROR" in stdout:
                         # Extract just the error message part
                         error_part = stdout.split("FLASH_ATTN_CHECK_ERROR: ", 1)[-1]
                         status_message = f"Status (Venv): Check Error: {error_part}"
                    else:
                         status_message = f"Status (Venv): Process failed. Return code {process.returncode}. Error: {stderr if stderr else 'No stderr output.'}"


                # Update GUI on the main thread
                self.root.after(0, self.flash_attn_status_var.set, status_message)

            except FileNotFoundError:
                 self.root.after(0, self.flash_attn_status_var.set, "Status: Python executable not found in Venv.")
            except subprocess.TimeoutExpired:
                 self.root.after(0, self.flash_attn_status_var.set, "Status: Venv check timed out.")
            except Exception as e:
                 print(f"Error running Venv check subprocess: {e}", file=sys.stderr)
                 traceback.print_exc(file=sys.stderr)
                 self.root.after(0, self.flash_attn_status_var.set, f"Status: Error running Venv check: {e}")

        # Run the check in a separate thread to keep the GUI responsive
        check_thread = Thread(target=run_check, daemon=True)
        check_thread.start()


    # ═════════════════════════════════════════════════════════════════
    #  Chat Template Logic (NEW)
    # ═════════════════════════════════════════════════════════════════
    def _update_template_controls_state(self, *args):
        """Enables/disables template controls based on radio button selection."""
        if self.use_custom_template.get():
            self.predefined_template_combobox.config(state=tk.DISABLED)
            if hasattr(self, 'custom_template_entry') and self.custom_template_entry.winfo_exists():
                 self.custom_template_entry.config(state=tk.NORMAL)
        else:
            self.predefined_template_combobox.config(state="readonly") # Re-enable readonly state
            if hasattr(self, 'custom_template_entry') and self.custom_template_entry.winfo_exists():
                 self.custom_template_entry.config(state=tk.DISABLED)

        # Also update the effective template display
        self._update_effective_template_display()


    def _update_effective_template_display(self, *args):
        """Updates the displayed effective template string."""
        if self.use_custom_template.get():
            effective_template = self.custom_template_string.get()
        else:
            selected_name = self.predefined_template_name.get()
            effective_template = self._predefined_templates.get(selected_name, "") # Default to empty if key not found

        # Ensure the displayed entry is writable before setting, then set back to readonly
        if hasattr(self, 'effective_template_display') and self.effective_template_display.winfo_exists():
             self.effective_template_display.config(state=tk.NORMAL)
             self.current_template_display.set(effective_template)
             self.effective_template_display.config(state="readonly")

        # Note: The trace on self.custom_template_string updates the display
        # only when the mode is custom. The combobox selection updates it
        # directly via its trace binding.


    def _on_custom_template_modified(self, event=None):
        """Callback for ScrolledText modifications to sync with StringVar."""
        # This is a bit tricky with ScrolledText. A standard Entry's textvariable
        # handles this sync automatically. With ScrolledText, we need to manually
        # get the content and update the StringVar.
        # We only want to update the StringVar (and trigger its trace) when the
        # text widget is actually enabled (i.e., custom mode is active).
        if self.use_custom_template.get() and hasattr(self, 'custom_template_entry') and self.custom_template_entry.winfo_exists():
            try:
                # Get the content from the text widget (from 1.0 to end-1c)
                content = self.custom_template_entry.get("1.0", tk.END).strip()
                # Update the StringVar if it's different
                if self.custom_template_string.get() != content:
                    self.custom_template_string.set(content)
                # Clear the modified flag so the event fires again on next change
                self.custom_template_entry.edit_modified(False)
            except tk.TclError:
                 # Handle case where widget might be destroyed during update
                 pass


    def _update_custom_template_text(self, *args):
        """Updates the custom template ScrolledText widget from the StringVar."""
        # This trace is needed if the StringVar is set externally (e.g., by loading a config)
        # It should only update the text widget if the content is different, to avoid loops.
        # We also need to ensure the widget is enabled before updating.
        if hasattr(self, 'custom_template_entry') and self.custom_template_entry.winfo_exists():
            content = self.custom_template_string.get()
            widget_content = self.custom_template_entry.get("1.0", tk.END).strip()

            if content != widget_content:
                 # Temporarily enable if it's disabled, update, then restore state
                 original_state = self.custom_template_entry.cget('state')
                 if original_state == tk.DISABLED:
                      self.custom_template_entry.config(state=tk.NORMAL)

                 self.custom_template_entry.delete("1.0", tk.END)
                 self.custom_template_entry.insert(tk.END, content)
                 self.custom_template_entry.edit_modified(False) # Clear modified flag

                 if original_state == tk.DISABLED:
                      self.custom_template_entry.config(state=tk.DISABLED)


    def _copy_template_display(self):
        """Copies the effective template string to the clipboard."""
        template_string = self.current_template_display.get()
        if template_string:
            try:
                self.root.clipboard_clear()
                self.root.clipboard_append(template_string)
                print("DEBUG: Copied effective template to clipboard.", file=sys.stderr)
            except Exception as e:
                print(f"ERROR: Failed to copy to clipboard: {e}", file=sys.stderr)
                messagebox.showerror("Copy Error", f"Failed to copy template to clipboard:\n{e}")
        else:
            messagebox.showinfo("Copy Info", "No template string to copy.")


    # ═════════════════════════════════════════════════════════════════
    #  Default Configuration Name Generation
    # ═════════════════════════════════════════════════════════════════
    def _generate_default_config_name(self):
        """Generates a default configuration name based on current settings."""
        print("DEBUG: Generating default config name...", file=sys.stderr)
        parts = []

        # 1. Model Name
        model_path_str = self.model_path.get().strip()
        if model_path_str:
            try:
                model_name = Path(model_path_str).stem # Get filename without extension
                # Sanitize the model name for filename use
                safe_model_name = re.sub(r'[\\/*?:"<>| ]', '_', model_name)
                safe_model_name = safe_model_name[:40].strip('_') # Truncate and clean
                if safe_model_name:
                    parts.append(safe_model_name)
                else:
                     parts.append("model") # Fallback if sanitization results in empty string
            except Exception:
                 parts.append("model") # Fallback on path error
        else:
            parts.append("default") # No model selected


        # 2. Key Parameters (add if NOT default)
        default_params = {
            "cache_type_k":  "f16",
            "threads":       str(self.physical_cores), # Default based on detection
            "threads_batch": "4", # llama.cpp default
            "batch_size":    "512", # llama.cpp default
            "ubatch_size":   "512", # llama.cpp default
            "ctx_size":      2048,
            "seed":          "-1",
            "temperature":   "0.8",
            "min_p":         "0.05",
            "n_gpu_layers":  "0", # String value from entry (default is 0)
            "tensor_split":  "", # Empty string default
            "main_gpu":      "0", # llama.cpp default
            "prio":          "0", # llama.cpp default
            "ignore_eos":    False, # Default for --ignore-eos flag
            "n_predict":     "-1", # Default for --n-predict
            # Booleans that are flags (present if True, absent if False)
            "flash_attn":    False, # Default for --flash-attn flag
            "no_mmap":       False, # Default for --no-mmap flag
            "mlock":         False, # Default for --mlock flag
            "no_kv_offload": False, # Default for --no-kv-offload flag
            "no_cnv":        False, # Default for --no-cnv flag
            # Chat template is not included in default name for brevity
        }

        current_params = {
            "cache_type_k":  self.cache_type_k.get().strip(),
            "threads":       self.threads.get().strip(),
            "threads_batch": self.threads_batch.get().strip(),
            "batch_size":    self.batch_size.get().strip(),
            "ubatch_size":   self.ubatch_size.get().strip(),
            "ctx_size":      self.ctx_size.get(), # int
            "seed":          self.seed.get().strip(),
            "temperature":   self.temperature.get().strip(),
            "min_p":         self.min_p.get().strip(),
            "n_gpu_layers":  self.n_gpu_layers.get().strip(), # String value from entry
            "tensor_split":  self.tensor_split.get().strip(),
            "main_gpu":      self.main_gpu.get().strip(),
            "prio":          self.prio.get().strip(),
            "ignore_eos":    self.ignore_eos.get(), # bool
            "n_predict":     self.n_predict.get().strip(),
            "flash_attn":    self.flash_attn.get(), # bool
            "no_mmap":       self.no_mmap.get(),   # bool
            "mlock":         self.mlock.get(),    # bool
            "no_kv_offload": self.no_kv_offload.get(), # bool
            "no_cnv":        self.no_cnv.get(),     # bool
        }

        # Add non-default parameters to name parts
        for key, current_val in current_params.items():
            default_val = default_params.get(key) # Get the default value

            # Special handling for GPU Layers: use the internal integer value
            if key == "n_gpu_layers":
                 # Use the integer value after clamping, not the raw entry string
                 gpu_layers_int = self.n_gpu_layers_int.get()
                 max_layers = self.max_gpu_layers.get()
                 # Compare the *effect* of the setting to the default (0 layers offloaded)
                 if gpu_layers_int > 0:
                      if max_layers > 0 and gpu_layers_int == max_layers:
                           parts.append("gpu-all")
                      else:
                           parts.append(f"gpu={gpu_layers_int}")
            # Special handling for Context Size: compare the integer value
            elif key == "ctx_size":
                 if current_val != default_val:
                      parts.append(f"ctx={current_val}")
            # Special handling for Boolean flags (add if True and default is False)
            elif isinstance(current_val, bool):
                 if current_val is True and default_val is False:
                      # Use a short name for the flag
                      flag_name_map = {
                         "flash_attn": "fa",
                         "no_mmap": "no-mmap",
                         "mlock": "mlock",
                         "no_kv_offload": "no-kv-offload",
                         "no_cnv": "no-cnv",
                         "ignore_eos": "no-eos",
                      }
                      parts.append(flag_name_map.get(key, key.replace('_', '-'))) # Use mapped name or just key
            # Handle other string parameters
            elif isinstance(current_val, str):
                 # Compare stripped strings. Handle empty string vs None default.
                 if current_val != default_val and current_val != "": # Also exclude empty strings
                      # Use abbreviations for common parameters
                      abbr_map = {
                          "cache_type_k": "kv",
                          "threads": "th",
                          "threads_batch": "tb",
                          "batch_size": "b",
                          "ubatch_size": "ub",
                          "seed": "s",
                          "temperature": "temp",
                          "min_p": "minp",
                          "tensor_split": "split",
                          "main_gpu": "main-gpu",
                          "prio": "prio",
                          "n_predict": "pred",
                      }
                      abbr = abbr_map.get(key, key.replace('_', '-')) # Use mapped name or just key
                      parts.append(f"{abbr}={current_val}")

        # 3. Assemble the name
        # Join parts with underscores, ensure total length isn't excessive
        generated_name = "_".join(parts)

        # Avoid leading/trailing underscores or multiple consecutive underscores
        generated_name = re.sub(r'_{2,}', '_', generated_name)
        generated_name = generated_name.strip('_')

        # Ensure it's not empty
        if not generated_name:
            generated_name = "default_config"

        # Limit total length (e.g., 80 characters)
        if len(generated_name) > 80:
             generated_name = generated_name[:80].rstrip('_') # Truncate and remove trailing underscore if any

        print(f"DEBUG: Generated config name: {generated_name}", file=sys.stderr)

        # Only update the variable if it's currently the default name or a previously generated name pattern
        # This prevents overwriting a name the user has manually typed.
        current_config_name = self.config_name.get().strip()
        # Check if the current name looks like a generated name (starts with model name or "default")
        # This check is a bit heuristic, a perfect check would require storing the *last* generated name.
        # A simpler approach: if the name is "default_config" or if it starts with the current model name part, update it.
        # Even simpler: just update it always. The user can change it back or type over it. Let's stick to the simple approach for now.
        # No, let's refine. Store the last auto-generated name. Only overwrite if the current name matches it.
        # This requires adding an attribute to store the last generated name.

        # Let's add a simple heuristic: only update if the current name is "default_config"
        # or if it matches a previously generated name (this part is hard without storing the previous name).
        # A slightly better approach: if the current name is "default_config" OR if the current name is empty, OR if it starts with the model name part (and the model name part exists), then replace it with the generated name, but *only* if the generated name is different.

        model_name_prefix = ""
        if parts and parts[0] != "default": # If model name was successfully added as the first part
             model_name_prefix = parts[0]

        # Decide whether to set the config_name variable
        # Rule: If the current name is 'default_config' OR if the current name is empty, OR if it starts with the model name part (and the model name part exists), then replace it with the generated name, but *only* if the generated name is different.
        update_variable = False
        if current_config_name == "default_config":
             update_variable = True
        elif not current_config_name: # If currently empty
             update_variable = True
        elif model_name_prefix and current_config_name.startswith(model_name_prefix): # If it starts with the model name part (could be "model_name" or "model_name_param=val")
            update_variable = True


        if update_variable and generated_name != current_config_name:
            self.config_name.set(generated_name)
            print("DEBUG: Updated config_name variable.", file=sys.stderr)
        elif not update_variable:
             print("DEBUG: Did not update config_name variable as it seems manually set.", file=sys.stderr)


    def _update_default_config_name_if_needed(self, *args):
        """Traced callback for variables that influence the default config name."""
        # This trace function is bound to variables like self.n_predict and self.ignore_eos.
        # It's called whenever those variables change.
        # We only want to regenerate and update the config name if the user hasn't
        # already manually set a custom name.
        # The _generate_default_config_name function already contains the logic
        # to decide whether to overwrite the current self.config_name value.
        # So we just call it here.
        self._generate_default_config_name()


    # ═════════════════════════════════════════════════════════════════
    #  configuration (save / load / delete)
    # ═════════════════════════════════════════════════════════════════
    def _current_cfg(self):
        gpu_layers_to_save = self.n_gpu_layers.get().strip()
        # Ensure selected_gpus in app_settings is up-to-date before saving
        self.app_settings["selected_gpus"] = [i for i, v in enumerate(self.gpu_vars) if v.get()]

        # Construct the configuration dictionary
        cfg = {
            "llama_cpp_dir": self.llama_cpp_dir.get(),
            "venv_dir":      self.venv_dir.get(),
            "model_path":    self.model_path.get(),
            "cache_type_k":  self.cache_type_k.get(),
            "threads":       self.threads.get(), # Save the user-set value
            "threads_batch": self.threads_batch.get(), # Save the user-set value
            "batch_size":    self.batch_size.get(), # Save the user-set value
            "ubatch_size":   self.ubatch_size.get(), # Save the user-set value
            "n_gpu_layers":  gpu_layers_to_save, # Save the string value (can be -1)
            "no_mmap":       self.no_mmap.get(),
            "no_cnv":        self.no_cnv.get(),
            "prio":          self.prio.get(),
            "temperature":   self.temperature.get(),
            "min_p":         self.min_p.get(),
            "ctx_size":      self.ctx_size.get(),
            "seed":          self.seed.get(),
            "flash_attn":    self.flash_attn.get(),
            "tensor_split":  self.tensor_split.get().strip(),
            "main_gpu":      self.main_gpu.get(),
            "mlock":         self.mlock.get(),
            "no_kv_offload": self.no_kv_offload.get(),
            "host":          self.host.get(),
            "port":          self.port.get(),
            # --- NEW: Add new parameters to config ---
            "ignore_eos":    self.ignore_eos.get(),
            "n_predict":     self.n_predict.get(),
            # --- NEW: Add chat template parameters ---
            "use_custom_template": self.use_custom_template.get(),
            "predefined_template_name": self.predefined_template_name.get(),
            "custom_template_string": self.custom_template_string.get(),
        }

        # Include selected_gpus directly in the config dictionary for easier loading from config tab
        cfg["gpu_indices"] = self.app_settings.get("selected_gpus", [])

        return cfg

    def _save_configuration(self):
        name = self.config_name.get().strip()
        if not name:
            # If the user clears the name and tries to save, regenerate a default name
            self._generate_default_config_name()
            name = self.config_name.get().strip() # Get the generated name
            if not name or name == "default_config": # If still empty or generic
                 return messagebox.showerror("Error","Enter a configuration name or select a model to generate one.")

        if name in self.saved_configs:
             if not messagebox.askyesno("Overwrite", f"Configuration '{name}' already exists. Overwrite?"):
                 return
        self.saved_configs[name] = self._current_cfg()
        self._save_configs() # This now includes saving selected_gpus from app_settings
        self._update_config_listbox()
        messagebox.showinfo("Saved", f"Configuration '{name}' saved.")

    def _load_configuration(self):
        if not self.config_listbox.curselection():
            return messagebox.showerror("Error","Select a configuration from the list to load.")
        name = self.config_listbox.get(self.config_listbox.curselection())
        cfg  = self.saved_configs.get(name)
        if not cfg:
             messagebox.showerror("Error", f"Configuration '{name}' data not found.")
             return

        # Load simple variables first
        self.llama_cpp_dir.set(cfg.get("llama_cpp_dir",""))
        self.venv_dir.set(cfg.get("venv_dir","")) # Setting this triggers the venv trace -> flash attn check
        self.cache_type_k.set(cfg.get("cache_type_k","f16"))
        # Load parameters, providing defaults for backward compatibility with older configs
        # Default to the *current* detected cores if not in the config for threads
        self.threads.set(cfg.get("threads", str(self.physical_cores))) # Default to detected physical cores
        self.threads_batch.set(cfg.get("threads_batch", str(self.logical_cores))) # Default to detected logical cores
        self.batch_size.set(cfg.get("batch_size", "512")) # Default to llama.cpp 512
        self.ubatch_size.set(cfg.get("ubatch_size", "512")) # Default to llama.cpp 512

        self.no_mmap.set(cfg.get("no_mmap",False))
        self.no_cnv.set(cfg.get("no_cnv",False))
        self.prio.set(cfg.get("prio","0"))
        self.temperature.set(cfg.get("temperature","0.8"))
        self.min_p.set(cfg.get("min_p","0.05"))
        ctx = cfg.get("ctx_size", 2048)
        self.ctx_size.set(ctx)
        self._sync_ctx_display(ctx) # Manually sync display
        self.seed.set(cfg.get("seed","-1"))
        self.flash_attn.set(cfg.get("flash_attn",False))
        self.tensor_split.set(cfg.get("tensor_split","").strip()) # Ensure strip on load too
        self.main_gpu.set(cfg.get("main_gpu","0"))
        self.mlock.set(cfg.get("mlock",False))
        self.no_kv_offload.set(cfg.get("no_kv_offload",False))
        self.host.set(cfg.get("host","127.0.0.1"))
        self.port.set(cfg.get("port","8080"))
        self.config_name.set(name)

        # --- NEW: Load new parameters ---
        self.ignore_eos.set(cfg.get("ignore_eos", False))
        self.n_predict.set(cfg.get("n_predict", "-1")) # Default -1 for backward compatibility

        # --- NEW: Load chat template parameters ---
        # Load mode first
        self.use_custom_template.set(cfg.get("use_custom_template", False))
        # Load predefined name and custom string
        self.predefined_template_name.set(cfg.get("predefined_template_name", "None (Use Model Default)"))
        self.custom_template_string.set(cfg.get("custom_template_string", ""))
        # Update UI state and display based on loaded values
        self._update_template_controls_state() # This re-enables/disables controls
        # The trace on custom_template_string or predefined_template_name should trigger display update,
        # but let's call it explicitly for robustness after loading
        self._update_effective_template_display()


        # Load GPU selections - This needs to update the checkboxes
        # Check for the 'gpu_indices' key directly in the config dictionary first
        loaded_gpu_indices = cfg.get("gpu_indices", self.app_settings.get("selected_gpus", [])) # Fallback to app_settings key if old config format
        # Store loaded indices in app_settings *before* updating checkboxes
        self.app_settings["selected_gpus"] = loaded_gpu_indices
        self._update_gpu_checkboxes() # This will set the checkboxes according to self.app_settings["selected_gpus"]
        # _update_gpu_checkboxes also triggers _update_recommendations


        # Load n_gpu_layers - This interacts with model analysis results
        loaded_gpu_layers_str = cfg.get("n_gpu_layers","0")
        self.n_gpu_layers.set(loaded_gpu_layers_str)
        try:
             val = int(loaded_gpu_layers_str)
             # Use _set_gpu_layers to set the int var and sync entry, clamping based on current max
             # (max_layers will be 0 initially if model not loaded yet)
             self._set_gpu_layers(val)
        except ValueError:
             self.n_gpu_layers.set("0")
             self._set_gpu_layers(0)

        # Load Model Path - This will trigger model selection logic and analysis
        loaded_model_path_str = cfg.get("model_path", "")
        # Set the variable first, then attempt to select in the listbox
        self.model_path.set(loaded_model_path_str)
        self.model_listbox.selection_clear(0, tk.END) # Clear previous visual selection

        selected_idx = -1
        if loaded_model_path_str:
            try:
                loaded_model_path_obj = Path(loaded_model_path_str).resolve() # Resolve the saved path before lookup
                found_display_name = None
                # Find the display name associated with the saved path in the *currently found* models (which are resolved)
                for display_name, full_path in self.found_models.items():
                    if full_path == loaded_model_path_obj:
                        found_display_name = display_name
                        break
                # If found, get its index in the current listbox (which is sorted by display name)
                if found_display_name:
                    listbox_items = self.model_listbox.get(0, tk.END)
                    selected_idx = listbox_items.index(found_display_name)
            except (ValueError, OSError, IndexError):
                 pass # Handle potential errors with old paths or listbox state

        if selected_idx != -1:
             # Select it visually and trigger the selection handler (_on_model_selected)
             # Using after(10, ...) gives the UI a moment to update before selection
             self.root.after(10, self._select_model_in_listbox, selected_idx)
        else:
             # If the model was not found in the current scan results
             self.model_path.set("") # Clear model path variable
             self._reset_model_info_display()
             self._reset_gpu_layer_controls(keep_entry_enabled=True) # Keep entry enabled if model not found
             self.current_model_analysis = {} # Clear analysis data
             self._update_recommendations() # Update recommendations based on no model
             self._generate_default_config_name() # Generate default name for no model state
             if loaded_model_path_str:
                  messagebox.showwarning("Model Not Found", f"The model from the config ('{Path(loaded_model_path_str).name if loaded_model_path_str else 'N/A'}') was not found in the current list.\nPlease ensure its directory is added and scanned, then select a model manually.")


        messagebox.showinfo("Loaded", f"Configuration '{name}' applied.")

    def _delete_configuration(self):
        if not self.config_listbox.curselection():
            return messagebox.showerror("Error","Select a configuration to delete.")
        name = self.config_listbox.get(self.config_listbox.curselection())
        if messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete the configuration '{name}'?"):
            self.saved_configs.pop(name, None)
            self._save_configs()
            self._update_config_listbox()
            messagebox.showinfo("Deleted", f"Configuration '{name}' deleted.")

    def _update_config_listbox(self):
        current_selection = self.config_listbox.curselection()
        selected_name = None
        if current_selection:
            selected_name = self.config_listbox.get(current_selection[0])
        self.config_listbox.delete(0, tk.END)
        sorted_names = sorted(self.saved_configs.keys())
        for cfg_name in sorted_names:
            self.config_listbox.insert(tk.END, cfg_name)
        if selected_name in sorted_names:
            try:
                 new_index = sorted_names.index(selected_name)
                 self.config_listbox.selection_set(new_index)
                 self.config_listbox.activate(new_index)
                 self.config_listbox.see(new_index)
            except ValueError: pass


    # ═════════════════════════════════════════════════════════════════
    #  command‑line builder
    # ═════════════════════════════════════════════════════════════════
    def _build_cmd(self):
        """Builds the command list for llama-server."""
        llama_dir_str = self.llama_cpp_dir.get().strip()
        if not llama_dir_str:
            messagebox.showerror("Error", "LLaMa.cpp root directory is not set.")
            return None
        try:
            llama_base_dir = Path(llama_dir_str).resolve() # Resolve the base dir
            if not llama_base_dir.is_dir(): raise NotADirectoryError()
        except Exception:
             messagebox.showerror("Error", f"Invalid LLaMa.cpp directory:\n{llama_dir_str}")
             return None

        exe_path = self._find_server_executable(llama_base_dir)
        if not exe_path:
            search_locs_str = "\n - ".join([str(p) for p in [
                 Path("."), Path("build/bin/Release"), Path("build/bin"), Path("build"), Path("bin"), Path("server")
            ]])
            exe_base_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
            simple_exe_name = "server.exe" if sys.platform == "win32" else "server"
            messagebox.showerror("Executable Not Found",
                                 f"Could not find '{exe_base_name}' or '{simple_exe_name}' within:\n{llama_base_dir}\n\n"
                                 f"Searched in common relative locations like:\n - {search_locs_str}\n\n"
                                 "Please ensure llama.cpp is built and the directory is correct.")
            return None

        cmd = [str(exe_path)]

        # --- Model Path ---
        model_full_path_str = self.model_path.get().strip()
        if not model_full_path_str:
            messagebox.showerror("Error", "No model selected. Please scan and select a model from the list.")
            return None
        try:
            model_path_obj = Path(model_full_path_str).resolve() # Resolve the path from the variable
            # Cross-check if the resolved path matches the one from our scan results
            # This handles cases where the user might manually type a path or the saved path is slightly different
            selected_name = ""
            sel = self.model_listbox.curselection()
            if sel: selected_name = self.model_listbox.get(sel[0])

            scan_matched_path = self.found_models.get(selected_name)

            if scan_matched_path and scan_matched_path == model_path_obj:
                 # The path from the variable matches the resolved path of the selected item from the scan. Use it.
                 final_model_path = str(model_path_obj)
            elif model_path_obj.is_file():
                 # The path from the variable is a valid file, but doesn't match the scan result for the selected item (or no item selected).
                 # This could happen if the user manually pasted a path. Use it, but maybe warn?
                 # For now, just use it if it's a valid file.
                 final_model_path = str(model_path_obj)
                 if selected_name and (not scan_matched_path or scan_matched_path != model_path_obj):
                     print(f"Warning: Model path from entry '{model_full_path_str}' doesn't exactly match the selected item '{selected_name}' from scan ('{scan_matched_path if scan_matched_path else 'Not in scan list'}'). Using path from entry.", file=sys.stderr)
            else:
                 # The path from the variable is not a valid file and doesn't match the scan result for the selected item.
                 error_msg = f"Invalid or missing model file:\n{model_full_path_str}"
                 if selected_name: error_msg += f"\n(Selected in GUI: {selected_name})"
                 error_msg += "\n\nPlease re-scan models or select a valid model file."
                 messagebox.showerror("Error", error_msg)
                 return None

        except Exception as e:
             # Catch other exceptions during path validation
             selected_name = ""
             sel = self.model_listbox.curselection()
             if sel: selected_name = self.model_listbox.get(sel[0])
             error_msg = f"Error validating model path:\n{model_full_path_str}\nError: {e}"
             if selected_name: error_msg += f"\n(Selected in GUI: {selected_name})"
             error_msg += "\n\nPlease re-scan models or select a valid model."
             messagebox.showerror("Error", error_msg)
             return None

        cmd.extend(["-m", final_model_path])

        # --- Other Arguments ---
        # --- KV Cache Type ---
        # llama.cpp default is f16. Add args only if different.
        kv_cache_type_val = self.cache_type_k.get().strip()
        # Add --cache-type-k and --cache-type-v if kv_cache_type_val is set and not 'f16'
        if kv_cache_type_val and kv_cache_type_val != "f16":
             cmd.extend(["--cache-type-k", kv_cache_type_val, "--cache-type-v", kv_cache_type_val])
             print(f"DEBUG: Adding --cache-type-k/v {kv_cache_type_val} (non-default)", file=sys.stderr)
        # If default, explicitly omit based on _add_arg logic when default_value is provided
        # This is redundant if we handle the pair explicitly, but _add_arg handles the "non-default" logic well.
        # Let's remove the manual pair logic and rely solely on _add_arg for --cache-type-k, assuming --cache-type-v
        # is handled internally by llama.cpp or should be set separately if needed (it's not in the GUI).
        # Ok, looking at llama.cpp, --cache-type-v defaults to k's value. So just setting k is sufficient.
        # Let's just use _add_arg for --cache-type-k comparing against f16.
        self._add_arg(cmd, "--cache-type-k", kv_cache_type_val, "f16")


        # --- Threads & Batching ---
        # Llama.cpp internal defaults: --threads=hardware_concurrency() (logical), --threads-batch=4
        # Note: The GUI default for --threads is physical cores, while llama.cpp default is logical.
        # The _add_arg helper needs to compare against the *llama.cpp* default for omission.
        # But the default *value* shown in the GUI should still be physical cores.
        # Let's compare against the llama.cpp default (logical cores) when deciding whether to *add* the arg.
        # This matches the _generate_default_config_name logic.
        self._add_arg(cmd, "--threads", self.threads.get(), str(self.logical_cores)) # Omit if matches llama.cpp default (logical)
        self._add_arg(cmd, "--threads-batch", self.threads_batch.get(), "4")         # Omit if matches llama.cpp default (4)

        # Llama.cpp internal defaults: --batch-size=512, --ubatch-size=512
        self._add_arg(cmd, "--batch-size", self.batch_size.get(), "512")
        self._add_arg(cmd, "--ubatch-size", self.ubatch_size.get(), "512")


        # Llama.cpp internal defaults: --ctx-size=2048, --seed=-1, --temp=0.8, --min-p=0.05
        self._add_arg(cmd, "--ctx-size", str(self.ctx_size.get()), "2048") # Use str() for int var
        self._add_arg(cmd, "--seed", self.seed.get(), "-1")
        self._add_arg(cmd, "--temp", self.temperature.get(), "0.8")
        self._add_arg(cmd, "--min-p", self.min_p.get(), "0.05")


        # Handle GPU arguments: --tensor-split takes precedence
        tensor_split_val = self.tensor_split.get().strip()
        n_gpu_layers_val = self.n_gpu_layers.get().strip() # Get the user's desired n_gpu_layers string value

        if tensor_split_val:
             cmd.extend(["--tensor-split", tensor_split_val])
             print(f"INFO: --tensor-split is set ('{tensor_split_val}'), this takes precedence over --n-gpu-layers for layer distribution.", file=sys.stderr)
             # If using --tensor-split, still pass --n-gpu-layers if it's *not* the default 0.
             # This allows '-1' with tensor-split if that's a supported mode, or a specific count if needed.
             if n_gpu_layers_val != "0":
                  cmd.extend(["--n-gpu-layers", n_gpu_layers_val])
        else:
             # If --tensor-split is NOT provided, use --n-gpu-layers
             # Add --n-gpu-layers only if it's not the default 0
             self._add_arg(cmd, "--n-gpu-layers", n_gpu_layers_val, "0")


        # --main-gpu is usually needed when offloading layers (either via --n-gpu-layers or --tensor-split)
        # It specifies which GPU is considered the "primary" one, often GPU 0.
        # Llama.cpp default is 0. Include --main-gpu if the user set a non-default value.
        main_gpu_val = self.main_gpu.get().strip()
        self._add_arg(cmd, "--main-gpu", main_gpu_val, "0")


        # Add --flash-attn flag if checked
        self._add_arg(cmd, "--flash-attn", self.flash_attn.get())

        # Memory options
        self._add_arg(cmd, "--no-mmap", self.no_mmap.get()) # Omit if False (default)
        self._add_arg(cmd, "--mlock", self.mlock.get()) # Omit if False (default)
        self._add_arg(cmd, "--no-kv-offload", self.no_kv_offload.get()) # Omit if False (default)

        # Performance options
        self._add_arg(cmd, "--no-cnv", self.no_cnv.get()) # Omit if False (default)
        self._add_arg(cmd, "--prio", self.prio.get(), "0") # Omit if 0 (default)

        # --- NEW: Generation options ---
        self._add_arg(cmd, "--ignore-eos", self.ignore_eos.get()) # Omit if False (default)
        self._add_arg(cmd, "--n-predict", self.n_predict.get(), "-1") # Omit if -1 (default)

        # --- NEW: Chat template option ---
        effective_template = self.current_template_display.get().strip()
        if effective_template: # Only add the argument if the effective template string is non-empty
             cmd.extend(["--chat-template", effective_template])
             print(f"DEBUG: Adding --chat-template: {effective_template[:50]}...", file=sys.stderr)


        # Add a note about using CUDA_VISIBLE_DEVICES if they selected specific GPUs via checkboxes
        # but are NOT using --tensor-split (which explicitly lists devices/split).
        # This warning is helpful because llama.cpp might use all GPUs by default unless restricted by env var or tensor-split.
        selected_gpu_indices = [i for i, v in enumerate(self.gpu_vars) if v.get()]
        if len(selected_gpu_indices) > 0 and len(selected_gpu_indices) < self.gpu_info["device_count"] and not tensor_split_val:
             # Only warn if the user explicitly selected a *subset* of GPUs using the checkboxes AND didn't use tensor-split
             print(f"\nINFO: Specific GPUs ({selected_gpu_indices}) were selected via checkboxes, but --tensor-split was not used.", file=sys.stderr)
             print("      llama-server might default to using all available GPUs unless CUDA_VISIBLE_DEVICES is set.", file=sys.stderr)
             print("      To *restrict* which GPUs are used without --tensor-split, set the CUDA_VISIBLE_DEVICES environment variable before launching (e.g., 'set CUDA_VISIBLE_DEVICES=0,2' on Windows).", file=sys.stderr)
             print("      Alternatively, use --tensor-split to explicitly assign layers.", file=sys.stderr)
        elif len(selected_gpu_indices) > 0 and self.gpu_info["device_count"] > 0:
             # If GPUs were selected (and there are GPUs), maybe a general reminder about env vars?
             # Or just assume the user knows if they selected. Keep the message only for subset selection without tensor-split.
             pass


        print("\n--- Generated Command ---", file=sys.stderr)
        # Use shlex.quote to make command printable and copy-pasteable in shells
        quoted_cmd = [shlex.quote(arg) for arg in cmd]
        print(" ".join(quoted_cmd), file=sys.stderr)
        print("-------------------------\n", file=sys.stderr)


        return cmd

    def _add_arg(self, cmd_list, arg_name, value, default_value=None):
        """
        Adds argument to cmd list if its value is non-empty or, if a default is given,
        if the value is different from the default. Handles bools.
        """
        # Handle boolean flags (always added if True)
        is_bool_var = isinstance(value, tk.BooleanVar)
        is_bool_py = isinstance(value, bool)

        if is_bool_var:
            if value.get():
                 cmd_list.append(arg_name)
            return

        if is_bool_py:
             if value:
                  cmd_list.append(arg_name)
             return

        # Handle non-boolean arguments (string, int, float)
        # Get the string representation of the value
        actual_value_str = str(value).strip()

        # Determine the string representation of the default value for comparison
        # If default_value is None, there is no default specified for comparison purposes,
        # so we add the argument if the actual value is non-empty.
        # If default_value is not None, we compare against its string representation.
        default_value_str = str(default_value).strip() if default_value is not None else None

        # Logic:
        # 1. If actual_value_str is empty:
        #    - If default_value_str is None or "", do not add (omitting means default).
        #    - If default_value_str is non-empty, do not add (omitting means default).
        #    => If actual_value_str is empty, never add the arg. The llama.cpp default will be used.
        # 2. If actual_value_str is non-empty:
        #    - If default_value_str is None, add the arg (no default to compare against).
        #    - If default_value_str is not None and actual_value_str != default_value_str, add the arg.
        #    - If default_value_str is not None and actual_value_str == default_value_str, do not add (redundant).
        #    => Add the arg if actual_value_str is non-empty AND (default_value_str is None OR actual_value_str != default_value_str).

        if actual_value_str: # Check if the user entered *any* value
             if default_value_str is None or actual_value_str != default_value_str:
                 # Add the argument if there's no default to compare against,
                 # OR if the user's value is different from the default.
                 cmd_list.extend([arg_name, actual_value_str])
             else:
                 # User entered the exact default value. Omit the argument.
                 print(f"DEBUG: Omitting '{arg_name} {actual_value_str}' as it matches default '{default_value_str}'", file=sys.stderr)
        else:
             # User entered an empty string. Omit the argument.
             print(f"DEBUG: Omitting '{arg_name}' due to empty value. Default '{default_value_str}' will be used.", file=sys.stderr)


    # ═════════════════════════════════════════════════════════════════
    #  launch & script helpers
    # ═════════════════════════════════════════════════════════════════
    def launch_server(self):
        cmd_list = self._build_cmd()
        if not cmd_list: return
        venv_path_str = self.venv_dir.get().strip()

        try:
            if sys.platform == "win32":
                 # On Windows, shell=True is often required for activate.bat and start
                 final_cmd_str = subprocess.list2cmdline(cmd_list)
                 if venv_path_str:
                    venv_path = Path(venv_path_str).resolve() # Resolve venv path
                    act_script = venv_path / "Scripts" / "activate.bat"
                    if not act_script.is_file():
                        messagebox.showerror("Error", f"Venv activation script (activate.bat) not found:\n{act_script}")
                        return
                    # Use 'start' to open a new window with a title
                    command = f'start "LLaMa.cpp Server" cmd /k ""{str(act_script)}" && {final_cmd_str}"'
                 else:
                    command = f'start "LLaMa.cpp Server" {final_cmd_str}'
                 subprocess.Popen(command, shell=True)

            else: # Linux/macOS
                # Build the base server command string
                quoted_cmd_parts = [shlex.quote(arg) for arg in cmd_list]
                server_command_str = " ".join(quoted_cmd_parts)

                launch_command = server_command_str
                if venv_path_str:
                    venv_path = Path(venv_path_str).resolve() # Resolve venv path
                    act_script = venv_path / "bin" / "activate"
                    if not act_script.is_file():
                        messagebox.showerror("Error", f"Venv activation script not found:\n{act_script}")
                        return
                    # Build command to activate venv and then run the server command
                    # Use 'source' to activate venv in the current shell of the terminal
                    # Add a pause at the end for xterm if needed
                    if sys.platform == "darwin": # macOS terminal usually keeps window open
                         pause_cmd = ""
                    else: # Linux terminals might close immediately
                         # Use a more robust check for interactive shell
                         pause_cmd = ' ; [ -t 0 ] && read -p "Press Enter to close..."' # Check if stdin is a terminal

                    # bash -c needs the entire command string to be quoted safely.
                    # shlex.quote is used for individual parts *within* the bash command, then the whole string is quoted.
                    # Ensure the activate script path is also quoted for 'source'
                    launch_command = f'echo "Activating venv: {venv_path}" && source {shlex.quote(str(act_script))} && echo "Launching server..." && {server_command_str}{pause_cmd}'


                # Attempt to launch in a new terminal window
                # Prioritize common desktop environment terminals
                terminals = ['gnome-terminal', 'konsole', 'xfce4-terminal', 'xterm']
                # Command patterns for various terminals. Use bash -c to run multiple commands.
                # Note: bash -c expects a single string argument containing the command(s) to execute.
                # This string needs its internal quotes/special chars handled.
                term_cmd_pattern = {
                    'gnome-terminal': " -- bash -c {}",
                    'konsole':        " -e bash -c {}",
                    'xfce4-terminal': " -e bash -c {}",
                    'xterm':          " -e bash -c {}", # xterm might need single quotes for the whole string depending on shell/version
                }
                launched = False
                import shutil # Use shutil.which to find terminal executable
                for term in terminals:
                    if shutil.which(term):
                         # Quote the *entire* launch_command string that bash will execute
                         quoted_full_command_for_bash = shlex.quote(launch_command)

                         # Construct the command for the terminal emulator itself
                         # Pass the quoted command string as a single argument to bash -c
                         term_cmd_parts = [term]
                         # Add terminal-specific args
                         if term in ['gnome-terminal', 'konsole', 'xfce4-terminal']:
                              term_cmd_parts.extend(['-e', 'bash', '-c', quoted_full_command_for_bash])
                         elif term == 'xterm':
                             # xterm often works better with the command string single-quoted
                             term_cmd_parts.extend(['-e', 'bash', '-c', "'" + quoted_full_command_for_bash.replace("'", "'\\''") + "'"]) # Escape single quotes within the command
                         else:
                              # Fallback for other terminals, might not work
                             term_cmd_parts.extend(['-e', 'bash', '-c', quoted_full_command_for_bash])


                         print(f"DEBUG: Attempting launch with terminal: {term_cmd_parts}", file=sys.stderr)
                         try:
                            # Use shell=False if passing command and args as a list.
                            # If the terminal command itself requires complex shell expansion (e.g., xterm single quotes),
                            # then shell=True might be needed for *that* command, but it's generally safer to avoid if possible.
                            # Let's try shell=False first with the parts list.
                            subprocess.Popen(term_cmd_parts, shell=False)
                            launched = True
                            break
                         except FileNotFoundError:
                              print(f"DEBUG: Terminal '{term}' not found or not executable using shell=False.", file=sys.stderr)
                         except Exception as term_err:
                             print(f"DEBUG: Failed to launch with {term} using shell=False: {term_err}", file=sys.stderr)
                             # If shell=False failed, try shell=True for commands that might expect it
                             try:
                                 # Reconstruct the command string if shell=True is needed
                                 term_cmd_str = subprocess.list2cmdline(term_cmd_parts)
                                 print(f"DEBUG: Retrying launch with {term} using shell=True: {term_cmd_str}", file=sys.stderr)
                                 subprocess.Popen(term_cmd_str, shell=True)
                                 launched = True
                                 break # Successfully launched with this terminal
                             except Exception as term_err_shell:
                                  print(f"DEBUG: Failed to launch with {term} using shell=True: {term_err_shell}", file=sys.stderr)


                if not launched:
                     messagebox.showerror("Launch Error", "Could not find a supported terminal (gnome-terminal, konsole, xfce4-terminal, xterm) or launch the server script.")

        except Exception as exc:
            messagebox.showerror("Launch Error", f"Failed to launch server process:\n{exc}")
            print(f"Failed to launch server process: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    def save_ps1_script(self):
        cmd_list = self._build_cmd()
        if not cmd_list: return

        selected_model_name = ""
        selection = self.model_listbox.curselection()
        if selection:
             selected_model_name = self.model_listbox.get(selection[0])

        default_name = "launch_llama_server.ps1"
        if selected_model_name:
            # Sanitize model name for filename
            model_name_part = re.sub(r'[\\/*?:"<>| ]', '_', selected_name)
            model_name_part = model_name_part[:50].strip('_') # Ensure no trailing underscore
            if model_name_part:
                default_name = f"launch_{model_name_part}.ps1"
            else:
                 default_name = "launch_selected_model.ps1" # Fallback if name is empty after sanitizing

            if not default_name.lower().endswith(".ps1"): default_name += ".ps1"


        path = filedialog.asksaveasfilename(defaultextension=".ps1",
                                            initialfile=default_name,
                                            filetypes=[("PowerShell Script", "*.ps1"), ("All Files", "*.*")])
        if not path: return

        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("<#\n")
                fh.write(" .SYNOPSIS\n")
                fh.write("    Launches the LLaMa.cpp server with saved settings.\n\n")
                fh.write(" .DESCRIPTION\n")
                fh.write("    Autogenerated PowerShell script from LLaMa.cpp Launcher GUI.\n")
                fh.write("    Activates virtual environment (if configured) and starts llama-server.\n")
                fh.write("#>\n\n")
                fh.write("$ErrorActionPreference = 'Continue'\n\n") # Use 'Continue' for better error reporting in script output


                venv = self.venv_dir.get().strip()
                if venv:
                    try:
                         venv_path = Path(venv).resolve() # Resolve venv path for script
                         act_script = venv_path / "Scripts" / "Activate.ps1"
                         if act_script.exists():
                             # Use literal path syntax for PowerShell activation script source
                             ps_act_path = f'"{str(act_script)}"' # Just quote the resolved path

                             fh.write(f'Write-Host "Activating virtual environment: {venv}" -ForegroundColor Cyan\n')
                             # Use 'try/catch' to report activation errors but continue if not critical
                             fh.write(f'try {{ . {ps_act_path} }} catch {{ Write-Warning "Failed to activate venv: $($_.Exception.Message)" }}\n\n')
                         else:
                              fh.write(f'Write-Warning "Virtual environment activation script (Activate.ps1) not found at: {act_script}"\n\n')
                    except Exception as path_ex:
                         fh.write(f'Write-Warning "Could not process venv path \'{venv}\': {path_ex}"\n\n')

                fh.write(f'Write-Host "Launching llama-server..." -ForegroundColor Green\n')

                # Build the command string for PowerShell
                # Need to handle spaces, quotes, and special PS characters in arguments
                # Use the call operator '&' for the executable path if it contains spaces or special characters.
                ps_cmd_parts = []
                for arg in cmd_list:
                    # Escape internal double quotes and backticks for PowerShell string literals
                    escaped_arg = arg.replace('"', '`"').replace('`', '``')

                    # Decide whether to enclose in double quotes or use literal string
                    # Simple heuristic: if it contains spaces or characters problematic without quotes, use quotes.
                    # More robust check: if it contains any character outside [a-zA-Z0-9-_./\\]
                    if re.search(r'[\s"\';&()|<>*?$]', arg): # Add '$' as it's variable marker in PS
                        ps_cmd_parts.append(f'"{escaped_arg}"') # Enclose in double quotes
                    else:
                        ps_cmd_parts.append(escaped_arg) # Use literal string

                # For the executable path specifically, use & operator if it's not just a simple name (like "llama-server")
                # Check if the first argument (executable path) contains path separators or spaces (after resolving)
                exe_path_obj_resolved = Path(cmd_list[0]).resolve() # Resolve the executable path
                exe_requires_call_operator = '/' in str(exe_path_obj_resolved) or '\\' in str(exe_path_obj_resolved) or ' ' in str(exe_path_obj_resolved)

                if exe_requires_call_operator:
                    # If the path needs the call operator, format it as '& "quoted_path"'
                    quoted_exe_path = str(exe_path_obj_resolved).replace('"', '`"').replace('`', '``') # PowerShell escape the resolved path
                    ps_cmd_parts[0] = f'& "{quoted_exe_path}"'
                # Else: ps_cmd_parts[0] is already the potentially quoted simple executable name, which is fine.


                fh.write(" ".join(ps_cmd_parts) + "\n\n")
                fh.write('Write-Host "Server process likely finished or detached." -ForegroundColor Yellow\n')
                fh.write('# Pause if script is run directly by double-clicking or outside an interactive shell\n')
                # Check if the host is ConsoleHost (typical when double-clicking or run from explorer)
                fh.write('if ($Host.Name -eq "ConsoleHost") {\n')
                fh.write('    Read-Host -Prompt "Press Enter to close..."\n') # Added ... for consistency
                fh.write('}\n')


            messagebox.showinfo("Saved", f"PowerShell script written to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Script Save Error", f"Could not save script:\n{exc}")
            print(f"Script Save Error: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


    def on_exit(self):
        self._save_configs()
        self.root.destroy()


# ═════════════════════════════════════════════════════════════════════
#  main
# ═════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    try:
        style = ttk.Style(root)
        themes = style.theme_names()
        # Prioritize themes that are likely dark or modern looking
        preferred_themes = ['forest-dark', 'forest-light', 'win11dark','win11light','vista', 'xpnative', 'winnative', 'clam', 'alt', 'default']
        used_theme = False
        for theme in preferred_themes:
            if theme in themes:
                try:
                     print(f"Applying theme: {theme}", file=sys.stderr)
                     style.theme_use(theme)
                     used_theme = True

                     # Basic heuristic to attempt dark theme configuration if name suggests it
                     # This might be redundant or conflict if the theme itself handles it well
                     # Only apply these if the theme name suggests dark and we successfully applied it
                     if 'dark' in theme.lower() or 'black' in theme.lower() or 'highcontrast' in theme.lower():
                          try:
                               # Get background/foreground from the theme's defaults
                               # Use lookup with default to avoid errors if element doesn't exist in theme
                               bg_color = style.lookup('TFrame', 'background', default='#2b2b2b')
                               fg_color = style.lookup('TLabel', 'foreground', default='#ffffff')
                               entry_bg = style.lookup('TEntry', 'fieldbackground', default='#3c3c3c')
                               entry_fg = style.lookup('TEntry', 'foreground', default='#ffffff')
                               listbox_bg = style.lookup('Tlistbox', 'background', default=entry_bg) # Note: Listbox is tk, not ttk, needs option_add
                               listbox_fg = style.lookup('Tlistbox', 'foreground', default=entry_fg) # Note: Listbox is tk, not ttk, needs option_add
                               listbox_select_bg = style.lookup('Tlistbox', 'selectbackground', default='#505050') # Note: Listbox is tk, not ttk, needs option_add
                               listbox_select_fg = style.lookup('Tlistbox', 'selectforeground', default='#ffffff') # Note: Listbox is tk, not ttk, needs option_add


                               # Apply to root and all widgets implicitly
                               root.configure(bg=bg_color)
                               # ttk widgets inherit from the style root, but apply explicit colors for safety
                               # Note: Style configuration is complex and theme-dependent.
                               # Applying these might override theme specifics.
                               # It's often better to just let the theme handle it if possible.
                               # Let's try just configuring the root and Listbox explicitly as they aren't ttk widgets.
                               # The 'forest' themes handle this automatically. Only needed for fallback themes.
                               if not theme.startswith('forest'):
                                    style.configure('.', background=bg_color, foreground=fg_color)
                                    style.configure('TFrame', background=bg_color)
                                    style.configure('TLabel', background=bg_color, foreground=fg_color)
                                    style.configure('TCheckbutton', background=bg_color, foreground=fg_color)
                                    style.configure('TRadiobutton', background=bg_color, foreground=fg_color) # If used
                                    style.configure('TButton', background=bg_color, foreground=fg_color) # Might not style button faces well

                                    # Listbox is tk, not ttk, needs option_add
                                    # Use try/except as option_add might not be available in all contexts or for all styles
                                    try:
                                       root.option_add('*Listbox.background', listbox_bg)
                                       root.option_add('*Listbox.foreground', listbox_fg)
                                       root.option_add('*Listbox.selectBackground', listbox_select_bg)
                                       root.option_add('*Listbox.selectForeground', listbox_select_fg)
                                    except tk.TclError: pass # Option might not exist for Listbox

                                    # Combobox dropdown list might need configuring
                                    try:
                                         style.map('TCombobox', fieldbackground=[('readonly', entry_bg), ('!readonly', entry_bg)],
                                                                foreground=[('readonly', entry_fg), ('!readonly', entry_fg)],
                                                                selectbackground=[('readonly', listbox_select_bg), ('!readonly', listbox_select_bg)],
                                                                selectforeground=[('readonly', listbox_select_fg), ('!readonly', listbox_select_fg)],
                                                                background=[('readonly', bg_color), ('!readonly', bg_color)])
                                         style.map('TEntry', fieldbackground=[('!disabled', entry_bg)], foreground=[('!disabled', entry_fg)])
                                    except tk.TclError: pass
                                    # ScrolledText is tk, needs option_add
                                    try:
                                        root.option_add('*ScrolledText.background', entry_bg)
                                        root.option_add('*ScrolledText.foreground', entry_fg)
                                        # Need to set insertbackground for cursor color in dark themes
                                        root.option_add('*ScrolledText.insertBackground', fg_color)
                                        # Selection colors
                                        root.option_add('*ScrolledText.selectBackground', listbox_select_bg)
                                        root.option_add('*ScrolledText.selectForeground', listbox_select_fg)
                                    except tk.TclError: pass


                          except tk.TclError as style_err:
                               print(f"Note: Could not fully configure basic dark theme styles after applying {theme}: {style_err}", file=sys.stderr)
                     break # Stop after applying the first preferred theme found
                except tk.TclError as e:
                     print(f"Failed to apply theme {theme}: {e}", file=sys.stderr)
                     continue # Try the next theme
        if not used_theme:
             print("Could not apply preferred ttk themes. Using default theme.", file=sys.stderr)
    except Exception as e:
         print(f"ttk themes not available or failed to apply: {e}", file=sys.stderr)


    app  = LlamaCppLauncher(root)
    root.protocol("WM_DELETE_WINDOW", app.on_exit)
    root.mainloop()