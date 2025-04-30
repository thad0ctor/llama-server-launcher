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
import shlex # <-- Import shlex for parameter splitting
import math
import time

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


# Check for Flash Attention library in the *GUI's* environment (fallback/initial status)
FLASH_ATTN_GUI_AVAILABLE = False
# Flash Attention typically depends on a CUDA build of PyTorch and a compatible llama.cpp build
# We check for the Python package as a proxy for availability in the environment.
if TORCH_AVAILABLE and sys.platform != "darwin": # Flash Attention is CUDA/Linux/Windows specific currently, not macOS
    try:
        # A simple check is just importing the top level, hoping for the best.
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
        print(f" - {dep}", file=sys.stderr) # Ensure print to stderr
    print("Please install them (e.g., 'pip install llama-cpp-python torch psutil flash_attn') if you need GGUF analysis, GPU features, or Flash Attention status.", file=sys.stderr)
    print("-------------------------------------\n", file=sys.stderr)


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

    llm_meta = None # Initialize llm_meta outside the try block

    try:
        analysis_result["file_size_bytes"] = model_path.stat().st_size
        analysis_result["file_size_gb"] = round(analysis_result["file_size_bytes"] / (1024**3), 2)

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


# Helper function cleanup needs to be defined outside or before launch_server's try/except
# Defined here as a static method, needs to be called using the class name or self
class LlamaCppLauncher:
    """Tk‑based launcher for llama.cpp HTTP server."""

    # Define the cleanup static method first
    @staticmethod
    def cleanup(path, delay=5):
        """Cleans up a temporary file after a delay."""
        import time # Ensure time is imported here if needed
        time.sleep(delay)
        try:
            p = Path(path)
            if p.exists():
                os.unlink(p)
                print(f"DEBUG: Cleaned up temporary file: {path}", file=sys.stderr)
        except OSError as e:
            print(f"Warning: Could not delete temporary file {path} after delay: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Unexpected error during cleanup of {path}: {e}", file=sys.stderr)


    # --- New Imports ---
    # Define hardcoded templates, now primarily just the "default" option
    _default_templates = {
        "Let llama.cpp Decide (Use Model Default)": "", # Key for the explicit default option
        # Add other core templates here if you want them available even without the JSON file
        "Alpaca": "### Instruction:\\n{{instruction}}\\n### Response:\\n{{response}}",
        "ChatML": "<|im_start|>system\\n{{system_message}}<|im_end|>\\n<|im_start|>user\\n{{prompt}}<|im_end|>\\n<|im_start|>assistant\\n",
        "Llama 2 Chat": "[INST] <<SYS>>\\n{{system_message}}\\n<</SYS>>\\n\\n{{prompt}}[/INST]",
        "Vicuna": "A chat between a curious user and an AI assistant.\nThe assistant gives helpful, harmless, honest answers.\nUSER: {{prompt}}\nASSISTANT: ",
        # Qwen3 templates are removed as requested
    }

    # ────────────────═════════════════════════════════════════════════
    #  construction / persistence
    # ────────────────═════════════════════════════════════════════════
    def __init__(self, root: tk.Tk):
        """
        Initializes the LLaMa.cpp Server Launcher application GUI and state.

        Args:
            root: The root Tkinter window.
        """
        self.root = root
        self.root.title("LLaMa.cpp Server Launcher")
        self.root.geometry("900x850")
        self.root.minsize(800, 750)

        # ------------------------------------------------ Internal Data Attributes --
        # Attributes that hold data not directly tied to Tk variables, often
        # populated during setup or used for internal logic.

        # --- System Info Attributes ---
        # These will be populated by _fetch_system_info later.
        self.gpu_info = {"available": False, "device_count": 0, "devices": []}
        self.ram_info = {}
        # Default fallback for CPU cores before fetching system info.
        # These will be updated by _fetch_system_info.
        self.cpu_info = {"logical_cores": 4, "physical_cores": 2}


        # --- Persistence & Configuration ---
        # Data structures for saving and loading application state and configurations.
        self.saved_configs = {}
        self.app_settings = {
            "last_llama_cpp_dir": "",
            "last_venv_dir":      "",
            "last_model_path":    "",
            "model_dirs":         [],
            "model_list_height":  8,
            # Save/load selected GPU indices (indices of detected GPUs)
            "selected_gpus":      [],
        }
        # List to store custom parameters entered by the user (strings)
        self.custom_parameters_list = [] # <-- New attribute for custom parameters

        # --- Hardcoded Chat Templates ---
        # These templates are always available.
        # The "Let llama.cpp Decide" key maps to an empty string, signaling the server
        # to use the model's built-in template or a simple default if none.
        # Qwen3 templates REMOVED here as requested.
        self._default_templates = {
            "Let llama.cpp Decide (Use Model Default)": "",
            "Alpaca": "### Instruction:\\n{{instruction}}\\n### Response:\\n{{response}}",
            "ChatML": "<|im_start|>system\\n{{system_message}}<|im_end|>\\n<|im_start|>user\\n{{prompt}}<|im_end|>\\n<|im_start|>assistant\\n",
            "Llama 2 Chat": "[INST] <<SYS>>\\n{{system_message}}\\n<</SYS>>\\n\\n{{prompt}}[/INST]",
            "Vicuna": "A chat between a curious user and an AI assistant.\nThe assistant gives helpful, harmless, honest answers.\nUSER: {{prompt}}\nASSISTANT: ",
        }


        # --- Template Loading & Processing ---
        # Load external chat templates and combine them with hardcoded ones.
        self.config_path = self._get_config_path() # Need config path first
        loaded_templates = self._find_and_load_chat_templates()

        # Combine hardcoded default templates with loaded ones
        # Start with hardcoded ones, then overwrite/add with loaded ones.
        self._all_templates = self._default_templates.copy()
        self._all_templates.update(loaded_templates)

        # Ensure the primary default key ("Let llama.cpp Decide...") is always present
        # and maps to an empty string, regardless of external JSON.
        default_key = "Let llama.cpp Decide (Use Model Default)" # Redefine locally for clarity
        self._all_templates[default_key] = "" # Force the empty string value

        # Order the final list of templates for the combobox: primary default first, then sorted others.
        all_merged_keys = list(self._all_templates.keys())
        if default_key in all_merged_keys:
            all_merged_keys.remove(default_key)
        sorted_other_keys = sorted(all_merged_keys)
        ordered_template_keys = [default_key] + sorted_other_keys

        # Reconstruct _all_templates dictionary using the ordered keys
        self._all_templates = {k: self._all_templates[k] for k in ordered_template_keys if k in self._all_templates}
        # At this point, self._all_templates contains all available templates in the desired order.


        # ------------------------------------------------ Tkinter Variables --
        # Variables linked directly to GUI widgets (StringVar, IntVar, BooleanVar).

        # --- File and Directory Paths ---
        self.llama_cpp_dir   = tk.StringVar() # Path to the llama.cpp directory
        self.venv_dir        = tk.StringVar() # Path to the Python virtual environment
        self.model_path      = tk.StringVar() # Path to the selected model file
        # List variable for model directories (used by Listbox)
        self.model_dirs_listvar = tk.StringVar()
        # Status variable for model scanning
        self.scan_status_var = tk.StringVar(value="Scan models to populate list.")

        # --- Basic Server Settings ---
        self.cache_type_k    = tk.StringVar(value="f16") # KV cache type (applies to k & v)
        # Initial default thread values before fetching system info.
        # These will be updated by _fetch_system_info or load_config.
        self.threads         = tk.StringVar(value="4")
        self.threads_batch   = tk.StringVar(value="4")
        self.batch_size      = tk.StringVar(value="512") # --batch-size (prompt)
        self.ubatch_size     = tk.StringVar(value="512") # --ubatch-size (unconditional batch)
        self.temperature     = tk.StringVar(value="0.8")
        self.min_p           = tk.StringVar(value="0.05")
        self.ctx_size        = tk.IntVar(value=2048)
        self.seed            = tk.StringVar(value="-1")

        # --- GPU Settings ---
        # String var for the entry (-1, 0, N)
        self.n_gpu_layers    = tk.StringVar(value="0")
        # Int var for the slider (0 to MaxLayers), updated when a model is loaded
        self.n_gpu_layers_int = tk.IntVar(value=0)
        # Max layers detected in the currently loaded model for slider max, updated with model info
        self.max_gpu_layers  = tk.IntVar(value=0)
        # Status/info next to layers control
        self.gpu_layers_status_var = tk.StringVar(value="Select model to see layer info")

        self.flash_attn      = tk.BooleanVar(value=False)
        # String for --tensor-split argument (e.g., "100,0,0" or device indices)
        self.tensor_split    = tk.StringVar(value="")
        self.main_gpu        = tk.StringVar(value="0") # String for --main-gpu entry (device index)

        # List of BooleanVars for dynamic GPU selection checkboxes (populated later)
        self.gpu_vars = []

        # --- Memory & Other Advanced Settings ---
        self.no_mmap         = tk.BooleanVar(value=False) # --no-mmap
        # Disable convolutional batching? (Related to -b / --batch-size implementation details)
        self.no_cnv          = tk.BooleanVar(value=False) # --no-cnv
        self.prio            = tk.StringVar(value="0") # --prio
        self.mlock           = tk.BooleanVar(value=False) # --mlock
        self.no_kv_offload   = tk.BooleanVar(value=False) # --no-kv-offload
        self.host            = tk.StringVar(value="127.0.0.1") # --host
        self.port            = tk.StringVar(value="8080") # --port

        # --- Configuration Management ---
        self.config_name     = tk.StringVar(value="default_config") # Name for saving/loading configs

        # --- New Parameters ---
        self.ignore_eos      = tk.BooleanVar(value=False) # --ignore-eos
        self.n_predict       = tk.StringVar(value="-1")   # --n-predict

        # --- Chat Template Selection Variables ---
        # Controls which template source is used: 'default', 'predefined', or 'custom'.
        # Initial state is 'default' (using the model's template if available).
        self.template_source = tk.StringVar(value="default")
        # Holds the key name of the selected predefined template from _all_templates.
        # Set initial value to the primary default key if it exists, otherwise the first available.
        default_key = "Let llama.cpp Decide (Use Model Default)" # Redefine locally for clarity
        initial_predefined_key = default_key if default_key in self._all_templates else (list(self._all_templates.keys())[0] if self._all_templates else "")
        self.predefined_template_name = tk.StringVar(value=initial_predefined_key)
        # Holds the user's custom template string entered in the text box.
        self.custom_template_string = tk.StringVar(value="")
        # Holds the *actual* template string currently selected for use (either from
        # predefined or custom) - primarily for display or internal logic.
        self.current_template_display = tk.StringVar(value="")

        # --- Custom Parameters Variables ---
        self.custom_param_entry_var = tk.StringVar() # For the entry field
        self.custom_parameters_listbox_var = tk.StringVar() # For the listbox display

        # Widgets are created after __init__ finishes. Initial variable values
        # will populate the widgets when they are bound.
        # Further setup (like loading saved settings, fetching system info)
        # typically happens *after* __init__, often in a dedicated setup method.

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
        # Load custom parameters list
        self.custom_parameters_list = self.app_settings.get("custom_parameters", [])


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

        # --- CHANGES FOR JSON TEMPLATES / DEFAULT OPTION ---
        # Bind trace to the new template source variable
        self.template_source.trace_add("write", lambda *args: self._update_template_controls_state())
        self.template_source.trace_add("write", lambda *args: self._update_effective_template_display())

        # Trace on combobox variable to update the displayed template string (only when in predefined mode)
        self.predefined_template_name.trace_add("write", lambda *args: self._update_effective_template_display())
        # Trace on custom entry variable to update the displayed template string (only when in custom mode)
        self.custom_template_string.trace_add("write", lambda *args: self._update_effective_template_display())


        # Populate model directories listbox
        self._update_model_dirs_listbox()
        # Populate custom parameters listbox
        self._update_custom_parameters_listbox() # <-- Update custom parameters listbox


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

        # Update chat template display and controls initially based on initial state
        self._update_template_controls_state() # Sets initial state based on self.template_source
        self._update_effective_template_display() # Sets initial displayed template based on source


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
        # Also update the recommended variables immediately after fetching
        self.threads.set(str(self.physical_cores))
        self.threads_batch.set(str(self.logical_cores))
        self.recommended_threads_var.set(f"Recommended: {self.physical_cores} (Your CPU physical cores)")
        self.recommended_threads_batch_var.set(f"Recommended: {self.logical_cores} (Your CPU logical cores)")

        # Display initial GPU detection status message
        self.gpu_detected_status_var.set(self.gpu_info['message'] if not self.gpu_info['available'] and self.gpu_info.get('message') else "")


    # ═════════════════════════════════════════════════════════════════
    #  persistence helpers
    # ═════════════════════════════════════════════════════════════════
    def _get_config_path(self):
        local_path = Path("llama_cpp_launcher_configs.json") # Renamed slightly to avoid potential clashes
        try:
            # Check if we can write to the current directory
            # Check if a config file exists and is empty (possibly from a failed previous run)
            # If empty, we can safely delete it and use the local path.
            if local_path.exists() and local_path.stat().st_size == 0:
                 try: local_path.unlink()
                 except OSError: pass # Ignore if delete fails

            # Check write permissions AFTER cleanup attempt
            if os.access(".", os.W_OK):
                 return local_path
            else:
                 raise PermissionError("No write access in current directory.") # Force fallback


        except (OSError, PermissionError, IOError) as e:
            print(f"Warning: Could not use local config path due to permissions/IO issue: {e}", file=sys.stderr)
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
                print(f"Note: Using fallback config path: {fallback_path}", file=sys.stderr)
                return fallback_path
            except Exception as e_fallback:
                 print(f"CRITICAL ERROR: Could not use local config path or fallback config path {fallback_dir}. Configuration saving/loading is disabled. Error: {e_fallback}", file=sys.stderr)
                 messagebox.showerror("Config Error", f"Failed to set up configuration directory.\nSaving/loading configurations is disabled.\nError: {e_fallback}")
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
            # Ensure custom_parameters is a list
            if not isinstance(self.app_settings.get("custom_parameters"), list):
                 self.app_settings["custom_parameters"] = []


            # Filter selected_gpus to only include indices of currently detected GPUs
            valid_gpu_indices = {gpu['id'] for gpu in self.detected_gpu_devices}
            self.app_settings["selected_gpus"] = [idx for idx in self.app_settings["selected_gpus"] if idx in valid_gpu_indices]

            # Load custom parameters into the internal list
            self.custom_parameters_list = self.app_settings.get("custom_parameters", [])


        except json.JSONDecodeError as e:
             print(f"Config Load Error: Failed to parse JSON from {self.config_path}\nError: {e}", file=sys.stderr)
             messagebox.showerror("Config Load Error", f"Failed to parse config file:\n{self.config_path}\n\nError: {e}\n\nUsing default settings.")
             # Reset to defaults on parse error
             self.app_settings = {
                 "last_llama_cpp_dir": "", "last_venv_dir": "", "last_model_path": "",
                 "model_dirs": [], "model_list_height": 8, "selected_gpus": [], "custom_parameters": []
             }
             self.saved_configs = {}
             self.custom_parameters_list = [] # Reset internal list
        except Exception as exc:
            print(f"Config Load Error: Could not load config from {self.config_path}\nError: {exc}", file=sys.stderr)
            messagebox.showerror("Config Load Error", f"Could not load config from:\n{self.config_path}\n\nError: {exc}\n\nUsing default settings.")
            # Reset to defaults on other load errors
            self.app_settings = {
                "last_llama_cpp_dir": "", "last_venv_dir": "", "last_model_path": "",
                "model_dirs": [], "model_list_height": 8, "selected_gpus": [], "custom_parameters": []
            }
            self.saved_configs = {}
            self.custom_parameters_list = [] # Reset internal list

    def _save_configs(self):
        if self.config_path.name in ("null", "NUL"):
             print("Config saving is disabled.", file=sys.stderr)
             return

        self.app_settings["model_dirs"] = [str(p) for p in self.model_dirs]
        self.app_settings["last_model_path"] = self.model_path.get()
        # Save selected GPU indices from the current state of the checkboxes
        self.app_settings["selected_gpus"] = [i for i, v in enumerate(self.gpu_vars) if v.get()]
        # Save custom parameters list
        self.app_settings["custom_parameters"] = self.custom_parameters_list

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
                 # Re-call _get_config_path to get the fallback path
                 self.config_path = self._get_config_path()
                 # Check if _get_config_path actually provided a different, writable path
                 if self.config_path != original_path and self.config_path.name not in ("null", "NUL"):
                      try:
                         self.config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                         messagebox.showwarning("Config Save Info", f"Could not write to original location.\nSettings stored in:\n{self.config_path}")
                      except Exception as final_exc:
                          print(f"Config Save Error: Failed to save settings to fallback {self.config_path}\nError: {final_exc}", file=sys.stderr)
                          messagebox.showerror("Config Save Error", f"Failed to save settings to fallback location:\n{self.config_path}\n\nError: {final_exc}")
                 else:
                      # If fallback path was the same or invalid, show error for original path
                      messagebox.showerror("Config Save Error", f"Failed to save settings to:\n{original_path}\n\nError: {exc}")
            else:
                 # If the original path was already a fallback, just report the error
                 messagebox.showerror("Config Save Error", f"Failed to save settings to:\n{self.config_path}\n\nError: {exc}")


    # --- FIX: Define the _save_configuration method ---
    def _save_configuration(self):
        """Saves the current UI settings as a named configuration."""
        name = self.config_name.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a name for the configuration.")
            return

        current_cfg = self._current_cfg()
        self.saved_configs[name] = current_cfg
        self._save_configs()
        self._update_config_listbox()
        messagebox.showinfo("Saved", f"Current settings saved as '{name}'.")

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
        # The button command now correctly references the method within the class
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


    # ═════════════════════════════════════════════════════════════════
    #  JSON Chat Template Loading
    # ═════════════════════════════════════════════════════════════════
    def _load_chat_templates_from_json(self, file_path):
        """Loads chat templates from a JSON file."""
        try:
            if file_path.is_file():
                content = file_path.read_text(encoding='utf-8')
                loaded_templates = json.loads(content)
                if not isinstance(loaded_templates, dict):
                    print(f"Warning: Chat templates file '{file_path}' is not a valid JSON object (dictionary). Ignoring.", file=sys.stderr)
                    return {}
                # Filter out any keys that are not non-empty strings or values that are not strings
                valid_templates = {k: v for k, v in loaded_templates.items() if isinstance(k, str) and k and isinstance(v, str)}
                print(f"Successfully loaded {len(valid_templates)} chat templates from {file_path}", file=sys.stderr)
                return valid_templates
            else:
                # print(f"DEBUG: Chat templates file not found at {file_path}", file=sys.stderr) # Too noisy for expected paths
                return {}
        except FileNotFoundError:
             # print(f"DEBUG: Chat templates file not found at {file_path}", file=sys.stderr) # Too noisy
            return {}
        except json.JSONDecodeError as e:
            print(f"Error decoding chat templates JSON from {file_path}: {e}", file=sys.stderr)
            messagebox.showwarning("Chat Template Load Error", f"Failed to decode chat templates JSON from:\n{file_path}\nError: {e}\n\nUsing default templates only.")
            return {}
        except Exception as e:
            print(f"Unexpected error loading chat templates from {file_path}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            messagebox.showwarning("Chat Template Load Error", f"Unexpected error loading chat templates from:\n{file_path}\nError: {e}\n\nUsing default templates only.")
            return {}

    def _find_and_load_chat_templates(self):
         """Finds and loads chat templates from the default JSON file locations."""
         template_file_name = "chat_templates.json"

         # 1. Check script directory
         script_dir = Path(__file__).parent
         local_path = script_dir / template_file_name
         loaded = self._load_chat_templates_from_json(local_path)
         if loaded: return loaded

         # 2. Check user config directory (same as config path's parent)
         # Ensure config_path is set before trying to get its parent
         if hasattr(self, 'config_path') and self.config_path and self.config_path.parent.exists():
            config_dir = self.config_path.parent
            fallback_path = config_dir / template_file_name
            loaded = self._load_chat_templates_from_json(fallback_path)
            if loaded: return loaded
         else:
             print("DEBUG: Config path not set or config directory does not exist. Skipping check for templates in config dir.", file=sys.stderr)


         # If not found in either place
         print(f"Note: Chat templates file '{template_file_name}' not found. Using hardcoded templates.", file=sys.stderr)
         return {} # Return empty dict if not found

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
        # --- Model Information Display --- (Labels only, not editable)
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


        # --- System Info Display (RAM & CPU) --- (Labels only, not editable)
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

        # CUDA Devices Info & Checkboxes (Populated dynamically in _update_gpu_checkboxes)
        gpu_avail_text = "Available" if self.gpu_info['available'] else "Not available"
        ttk.Label(inner, text=f"CUDA Devices ({gpu_avail_text}):")\
            .grid(column=0, row=r, sticky="nw", padx=10, pady=3)
        ttk.Label(inner, textvariable=self.gpu_detected_status_var, font=("TkSmallCaptionFont"), foreground="orange")\
             .grid(column=1, row=r, sticky="nw", padx=5, pady=3, columnspan=3)
        r += 1
        self.gpu_checkbox_frame = ttk.Frame(inner)
        self.gpu_checkbox_frame.grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=(0, 5))
        # The checkboxes themselves are added in _update_gpu_checkboxes and get their state there.
        r += 1

        # Display VRAM for each GPU in a separate row below checkboxes
        self._display_gpu_vram_info(inner, r)
        r += 1

        # --- GPU Layers (Slider + Entry + Status Label) ---
        ttk.Label(inner, text="GPU Layers (--n-gpu-layers):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        gpu_layers_frame = ttk.Frame(inner)
        gpu_layers_frame.grid(column=1, row=r, columnspan=3, sticky="ew", padx=5, pady=3)

        # Entry should be enabled by default
        self.n_gpu_layers_entry = ttk.Entry(gpu_layers_frame, textvariable=self.n_gpu_layers, width=6, state=tk.NORMAL)
        self.n_gpu_layers_entry.grid(column=0, row=0, sticky="w", padx=(0, 10))

        # Slider is intentionally disabled initially
        self.gpu_layers_slider = ttk.Scale(gpu_layers_frame, from_=0, to=self.max_gpu_layers.get(),
                                           orient="horizontal", variable=self.n_gpu_layers_int,
                                           command=self._sync_gpu_layers_from_slider, state=tk.DISABLED)
        self.gpu_layers_slider.grid(column=1, row=0, sticky="ew", padx=5)

        self.gpu_layers_status_label = ttk.Label(gpu_layers_frame, textvariable=self.gpu_layers_status_var, width=35, anchor="w")
        self.gpu_layers_status_label.grid(column=2, row=0, sticky="w", padx=(10, 0))

        gpu_layers_frame.columnconfigure(1, weight=1)

        ttk.Label(inner, text="Layers to offload (0=CPU only, -1=All). Slider range updates with model analysis.", font=("TkSmallCaptionFont"))\
            .grid(column=1, row=r+1, columnspan=3, sticky="w", padx=5); r += 2

        # --- Tensor Split ---
        ttk.Label(inner, text="Tensor Split (--tensor-split):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        # Ensure entry is explicitly NORMAL
        self.tensor_split_entry = ttk.Entry(inner, textvariable=self.tensor_split, width=25, state=tk.NORMAL)
        self.tensor_split_entry.grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, text="e.g., '3,1' splits layers 75%/25% across GPUs 0 and 1.", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r, columnspan=2, sticky="w", padx=5, pady=3); r += 1
        ttk.Label(inner, text="Recommended Split (VRAM-based):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Label(inner, textvariable=self.recommended_tensor_split_var)\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3, columnspan=3); r += 1


        # --- Main GPU ---
        ttk.Label(inner, text="Main GPU (--main-gpu):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        # Ensure entry is explicitly NORMAL
        self.main_gpu_entry = ttk.Entry(inner, textvariable=self.main_gpu, width=10, state=tk.NORMAL)
        self.main_gpu_entry.grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, text="Primary GPU index (usually 0). Used with --n-gpu-layers when tensor-split is not set.", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r, columnspan=2, sticky="w", padx=5, pady=3); r += 1


        # --- Flash Attention ---
        ttk.Label(inner, text="Flash Attention (--flash-attn):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        # Ensure checkbox is explicitly NORMAL
        self.flash_attn_check = ttk.Checkbutton(inner, variable=self.flash_attn, state=tk.NORMAL)
        self.flash_attn_check.grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, text="Use Flash Attention kernel (CUDA only, requires specific build)", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r, sticky="w", padx=5, pady=3);
        self.flash_attn_status_label = ttk.Label(inner, textvariable=self.flash_attn_status_var, font=("TkSmallCaptionFont", 8, ("bold",)))
        self.flash_attn_status_label.grid(column=3, row=r, sticky="w", padx=5, pady=3); r += 1


        # --- Memory & Cache settings ---
        ttk.Label(inner, text="Memory & Cache", font=("TkDefaultFont", 12, "bold"))\
            .grid(column=0, row=r, sticky="w", padx=10, pady=(20,5)); r += 1
        ttk.Separator(inner, orient="horizontal")\
            .grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=5); r += 1

        ttk.Label(inner, text="KV Cache Type (--cache-type-k):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        # Combobox state is "readonly" for selection, but not disabled
        self.cache_type_k_combo = ttk.Combobox(inner, textvariable=self.cache_type_k, width=10, values=("f16","f32","q8_0","q4_0","q4_1","q5_0","q5_1"), state="readonly")
        self.cache_type_k_combo.grid(column=1, row=r, sticky="w", padx=5, pady=3); r += 1
        ttk.Label(inner, text="Quantization for KV cache (f16 is default, lower Q=more memory saved)", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r-1, columnspan=2, sticky="w", padx=5, pady=3);


        ttk.Label(inner, text="Disable mmap (--no-mmap):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        # Ensure checkbox is explicitly NORMAL
        self.no_mmap_check = ttk.Checkbutton(inner, variable=self.no_mmap, state=tk.NORMAL)
        self.no_mmap_check.grid(column=1, row=r, sticky="w", padx=5, pady=3); r += 1
        ttk.Label(inner, text="Load entire model into RAM instead of mapping file", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r-1, columnspan=2, sticky="w", padx=5, pady=3); # Re-grid label

        ttk.Label(inner, text="Lock in RAM (--mlock):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        # Ensure checkbox is explicitly NORMAL
        self.mlock_check = ttk.Checkbutton(inner, variable=self.mlock, state=tk.NORMAL)
        self.mlock_check.grid(column=1, row=r, sticky="w", padx=5, pady=3); r += 1
        ttk.Label(inner, text="Prevent swapping model/KV cache (may require permissions)", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r-1, columnspan=2, sticky="w", padx=5, pady=3); # Re-grid label

        ttk.Label(inner, text="Disable KV Offload (--no-kv-offload):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        # Ensure checkbox is explicitly NORMAL
        self.no_kv_offload_check = ttk.Checkbutton(inner, variable=self.no_kv_offload, state=tk.NORMAL)
        self.no_kv_offload_check.grid(column=1, row=r, sticky="w", padx=5, pady=3); r += 1
        ttk.Label(inner, text="Keep KV cache in CPU RAM even with GPU layers", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r-1, columnspan=2, sticky="w", padx=5, pady=3); # Re-grid label


        # --- Performance (Batching & Threading) ---
        ttk.Label(inner, text="Performance Settings", font=("TkDefaultFont", 12, "bold"))\
            .grid(column=0, row=r, sticky="w", padx=10, pady=(20,5)); r += 1
        ttk.Separator(inner, orient="horizontal")\
            .grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=5); r += 1

        # Threads (--threads) moved to Basic Settings now (already checked in main tab setup)

        ttk.Label(inner, text="Prompt Batch Size (--batch-size):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        # Ensure entry is explicitly NORMAL
        self.batch_size_entry = ttk.Entry(inner, textvariable=self.batch_size, width=10, state=tk.NORMAL)
        self.batch_size_entry.grid(column=1, row=r, sticky="w", padx=5, pady=3); r += 1
        ttk.Label(inner, text="Number of tokens to process in a single batch during prompt processing (default: 512)", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r-1, columnspan=2, sticky="w", padx=5, pady=3); # Re-grid label


        ttk.Label(inner, text="Unconditional Batch Size (--ubatch-size):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        # Ensure entry is explicitly NORMAL
        self.ubatch_size_entry = ttk.Entry(inner, textvariable=self.ubatch_size, width=10, state=tk.NORMAL)
        self.ubatch_size_entry.grid(column=1, row=r, sticky="w", padx=5, pady=3); r += 1
        ttk.Label(inner, text="Overrides --batch-size for prompt processing; allows larger batch regardless of GPU mem.", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r-1, columnspan=2, sticky="w", padx=5, pady=3); # Re-grid label


        # Threads Batch (--threads-batch)
        ttk.Label(inner, text="Batch Threads (--threads-batch):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        # Ensure entry is explicitly NORMAL
        self.threads_batch_entry = ttk.Entry(inner, textvariable=self.threads_batch, width=10, state=tk.NORMAL)
        self.threads_batch_entry.grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, textvariable=self.recommended_threads_batch_var, font=("TkSmallCaptionFont")).grid(column=2, row=r, columnspan=2, sticky="w", padx=5, pady=3); r += 1
        ttk.Label(inner, text="Number of threads to use for batch processing (llama.cpp default: 4)", font=("TkSmallCaptionFont"))\
            .grid(column=1, row=r+1, columnspan=3, sticky="w", padx=5, pady=(0,3)); r += 1


        # Disable Conv Batching (--no-cnv) - Moved here as it's related to batching
        ttk.Label(inner, text="Disable Conv Batching (--no-cnv):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        # Ensure checkbox is explicitly NORMAL
        self.no_cnv_check = ttk.Checkbutton(inner, variable=self.no_cnv, state=tk.NORMAL)
        self.no_cnv_check.grid(column=1, row=r, sticky="w", padx=5, pady=3); r += 1
        ttk.Label(inner, text="Disable convolutional batching for prompt processing", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r-1, columnspan=2, sticky="w", padx=5, pady=3); # Re-grid label


        # --- Scheduling Priority --- (Already existed, just moved)
        ttk.Label(inner, text="Scheduling Priority (--prio):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        # Combobox state is "readonly"
        self.prio_combo = ttk.Combobox(inner, textvariable=self.prio, width=10, values=("0","1","2","3"), state="readonly")
        self.prio_combo.grid(column=1, row=r, sticky="w", padx=5, pady=3); r += 1
        ttk.Label(inner, text="0=Normal, 1=Medium, 2=High, 3=Realtime (OS dependent)", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r-1, columnspan=2, sticky="w", padx=5, pady=3); # Re-grid label


        # --- NEW: Generation Settings ---
        ttk.Label(inner, text="Generation Settings", font=("TkDefaultFont", 12, "bold"))\
            .grid(column=0, row=r, sticky="w", padx=10, pady=(20,5)); r += 1
        ttk.Separator(inner, orient="horizontal")\
            .grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=5); r += 1

        # --ignore-eos
        ttk.Label(inner, text="Ignore EOS Token (--ignore-eos):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        # Ensure checkbox is explicitly NORMAL
        self.ignore_eos_check = ttk.Checkbutton(inner, variable=self.ignore_eos, state=tk.NORMAL)
        self.ignore_eos_check.grid(column=1, row=r, sticky="w", padx=5, pady=3); r += 1
        ttk.Label(inner, text="Don't stop generation at EOS token", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r-1, columnspan=2, sticky="w", padx=5, pady=3); # Re-grid label


        # --n-predict
        ttk.Label(inner, text="Max Tokens to Predict (--n-predict):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        # Ensure entry is explicitly NORMAL
        self.n_predict_entry = ttk.Entry(inner, textvariable=self.n_predict, width=10, state=tk.NORMAL)
        self.n_predict_entry.grid(column=1, row=r, sticky="w", padx=5, pady=3); r += 1
        ttk.Label(inner, text="Tokens to predict (e.g., 128, 512, -1 for unlimited/context size)", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r-1, columnspan=2, sticky="w", padx=5, pady=3); # Re-grid label

        # --- NEW: Custom Parameters Section ---
        ttk.Label(inner, text="Custom Parameters", font=("TkDefaultFont", 12, "bold"))\
            .grid(column=0, row=r, sticky="w", padx=10, pady=(20,5)); r += 1
        ttk.Separator(inner, orient="horizontal")\
            .grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=5); r += 1

        ttk.Label(inner, text="Add Parameter:").grid(column=0, row=r, sticky="w", padx=10, pady=3)
        # Ensure entry is explicitly NORMAL
        self.custom_param_entry = ttk.Entry(inner, textvariable=self.custom_param_entry_var, state=tk.NORMAL)
        self.custom_param_entry.grid(column=1, row=r, sticky="ew", padx=5, pady=3, columnspan=2)
        # Ensure button is explicitly NORMAL
        self.add_custom_param_button = ttk.Button(inner, text="Add", command=self._add_custom_parameter, state=tk.NORMAL)
        self.add_custom_param_button.grid(column=3, row=r, sticky="w", padx=5, pady=3); r += 1
        ttk.Label(inner, text="Enter full parameter string, e.g., '--my-flag true' or '--config-path /path/to/config'.", font=("TkSmallCaptionFont"))\
             .grid(column=1, row=r, columnspan=3, sticky="w", padx=5, pady=(0,5)); r += 1


        ttk.Label(inner, text="Added Parameters:").grid(column=0, row=r, sticky="nw", padx=10, pady=3)
        custom_params_list_frame = ttk.Frame(inner)
        custom_params_list_frame.grid(column=1, row=r, columnspan=2, sticky="nsew", padx=5, pady=3, rowspan=2)
        custom_params_sb = ttk.Scrollbar(custom_params_list_frame, orient=tk.VERTICAL)
        # Ensure listbox is explicitly NORMAL
        self.custom_parameters_listbox = tk.Listbox(custom_params_list_frame,
                                                    listvariable=self.custom_parameters_listbox_var,
                                                    height=4, width=60, # Adjusted width
                                                    yscrollcommand=custom_params_sb.set,
                                                    exportselection=False, state=tk.NORMAL)
        custom_params_sb.config(command=self.custom_parameters_listbox.yview)
        self.custom_parameters_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        r += 1 # Advance row after adding the listbox frame

        btn_frame = ttk.Frame(inner); btn_frame.grid(column=3, row=r-1, sticky="ew", padx=5, pady=3, rowspan=2)
        # Ensure button is explicitly NORMAL
        self.remove_custom_param_button = ttk.Button(btn_frame, text="Remove Selected", command=self._remove_custom_parameter, state=tk.NORMAL)
        self.remove_custom_param_button.pack(side=tk.TOP, pady=2, fill=tk.X)
        r += 1

        # Ensure column 1 and 2 (entry, listbox frame) can expand
        inner.columnconfigure(1, weight=1)
        inner.columnconfigure(2, weight=1)

        
    # ░░░░░ CHAT TEMPLATE TAB ░░░░░ (NEW)
    def _setup_chat_template_tab(self, parent):
        frame = ttk.Frame(parent, padding=10); frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1) # Allow column 1 to expand for entries/combobox

        r = 0
        ttk.Label(frame, text="Chat Template Settings", font=("TkDefaultFont", 12, "bold"))\
            .grid(column=0, row=r, columnspan=3, sticky="w", padx=5, pady=(0,5)); r += 1

        ttk.Separator(frame, orient='horizontal').grid(column=0, row=r, columnspan=3, sticky='ew', padx=5, pady=10); r += 1

        # --- CHANGES FOR JSON TEMPLATES / DEFAULT OPTION ---
        # --- Template Source Selection (Radio Buttons) ---
        ttk.Label(frame, text="Select Template Source:").grid(column=0, row=r, sticky="w", padx=5, pady=3); r += 1

        radio_frame = ttk.Frame(frame)
        radio_frame.grid(column=0, row=r, columnspan=3, sticky="w", padx=5, pady=3); r += 1

        # Radio Button for "Let llama.cpp Decide"
        ttk.Radiobutton(radio_frame, text="Let llama.cpp Decide",
                        variable=self.template_source, value="default")\
            .pack(side="left", padx=(0, 10))

        # Radio Button for "Use Predefined Template"
        ttk.Radiobutton(radio_frame, text="Use Predefined Template:",
                        variable=self.template_source, value="predefined")\
            .pack(side="left", padx=(0, 10))

        # Radio Button for "Use Custom Template:"
        ttk.Radiobutton(radio_frame, text="Use Custom Template:",
                        variable=self.template_source, value="custom")\
            .pack(side="left")

        r += 1 # Advance row for the combobox/entry below


        # --- Predefined Template Dropdown ---
        # This is now only active when "Use Predefined Template" is selected via radio button
        ttk.Label(frame, text="Predefined Template:").grid(column=0, row=r, sticky="w", padx=5, pady=3)
        self.predefined_template_combobox = ttk.Combobox(frame, textvariable=self.predefined_template_name,
                                                        # Use keys from the combined dictionary
                                                        values=list(self._all_templates.keys()),
                                                        state="readonly") # State managed by _update_template_controls_state
        self.predefined_template_combobox.grid(column=1, row=r, sticky="ew", padx=5, pady=3, columnspan=2)
        # Binding remains for combobox selection, but it's traced to update effective display
        # self.predefined_template_name.trace_add("write", lambda *args: self._update_effective_template_display()) # Already exists


        r += 1 # Next row


        # --- Custom Template Entry ---
        # This is now only active when "Use Custom Template" is selected via radio button
        ttk.Label(frame, text="Custom Template String (--chat-template):")\
            .grid(column=0, row=r, sticky="nw", padx=5, pady=3)
        self.custom_template_entry = scrolledtext.ScrolledText(frame,
                                                               wrap=tk.WORD, # Wrap words
                                                               height=8,     # Height in lines
                                                               width=60,     # Width in characters (approx)
                                                               relief=tk.SUNKEN,
                                                               bd=1)
        # Set initial value from StringVar (already does this in __init__ via trace)
        # self.custom_template_entry.insert(tk.END, self.custom_template_string.get()) # Remove explicit insert here
        # Bind events to sync ScrolledText content with StringVar (already exists)
        self.custom_template_entry.bind('<<Modified>>', self._on_custom_template_modified)
        self.custom_template_string.trace_add("write", lambda *args: self._update_custom_template_text()) # Already exists and calls _update_effective_template_display


        self.custom_template_entry.grid(column=1, row=r, sticky="nsew", padx=5, pady=3, columnspan=2)
        frame.rowconfigure(r, weight=1) # Allow text area row to expand
        r += 1


        # --- Effective Template Display --- (Logic remains the same, only the source changes)
        ttk.Label(frame, text="Effective Template:").grid(column=0, row=r, sticky="w", padx=5, pady=3)
        self.effective_template_display = ttk.Entry(frame, textvariable=self.current_template_display,
                                                    state="readonly")
        self.effective_template_display.grid(column=1, row=r, sticky="ew", padx=5, pady=3)
        ttk.Button(frame, text="Copy", command=self._copy_template_display)\
            .grid(column=2, row=r, sticky="w", padx=5, pady=3)

        r += 1

        # Keep help labels, adjust wording if necessary
        ttk.Label(frame, text="Enter a Go-template string. e.g., \"### Instruction:\\n{{instruction}}\\n### Response:\\n{{response}}\"", font=("TkSmallCaptionFont"))\
            .grid(column=1, row=r, columnspan=2, sticky="w", padx=5, pady=(0,3)); r += 1
        ttk.Label(frame, text="Use double backslashes (\\\\) for newline characters within the template string for Python literals. The server uses Go-template syntax.", font=("TkSmallCaptionFont"), foreground="orange")\
             .grid(column=1, row=r, columnspan=2, sticky="w", padx=5, pady=(0,3)); r += 1


        # Initial state update based on self.template_source (called in __init__)


    # --- CHANGES FOR JSON TEMPLATES / DEFAULT OPTION ---
    # Modify _update_template_controls_state
    def _update_template_controls_state(self, *args):
        """Enables/disables template controls based on radio button selection."""
        source = self.template_source.get()
        if hasattr(self, 'predefined_template_combobox') and self.predefined_template_combobox.winfo_exists():
            if source == "predefined":
                self.predefined_template_combobox.config(state="readonly")
            else:
                self.predefined_template_combobox.config(state=tk.DISABLED)

        if hasattr(self, 'custom_template_entry') and self.custom_template_entry.winfo_exists():
            if source == "custom":
                self.custom_template_entry.config(state=tk.NORMAL)
            else:
                self.custom_template_entry.config(state=tk.DISABLED)

        # Trigger update of the effective template display (trace handles it)
        # self._update_effective_template_display()


    # --- CHANGES FOR JSON TEMPLATES / DEFAULT OPTION ---
    # Modify _update_effective_template_display
    def _update_effective_template_display(self, *args):
        """Updates the displayed effective template string based on current source."""
        source = self.template_source.get()
        effective_template = "" # Default to empty

        if source == "default":
            effective_template = "" # Explicitly empty when llama.cpp decides
        elif source == "predefined":
            selected_name = self.predefined_template_name.get()
            # Get template string from the combined dictionary
            effective_template = self._all_templates.get(selected_name, "") # Default to empty if key not found
        elif source == "custom":
            effective_template = self.custom_template_string.get()

        # Ensure the displayed entry is writable before setting, then set back to readonly
        if hasattr(self, 'effective_template_display') and self.effective_template_display.winfo_exists():
             self.effective_template_display.config(state=tk.NORMAL)
             self.current_template_display.set(effective_template)
             self.effective_template_display.config(state="readonly")


    def _on_custom_template_modified(self, event=None):
        """Callback for ScrolledText modifications to sync with StringVar."""
        # This is a bit tricky with ScrolledText. A standard Entry's textvariable
        # handles this sync automatically. With ScrolledText, we need to manually
        # get the content and update the StringVar.
        # We only want to update the StringVar (and trigger its trace) when the
        # text widget is actually enabled (i.e., custom mode is active).
        if self.template_source.get() == "custom" and hasattr(self, 'custom_template_entry') and self.custom_template_entry.winfo_exists():
            try:
                # Get the content from the text widget (from 1.0 to end-1c), stripping leading/trailing whitespace
                content = self.custom_template_entry.get("1.0", tk.END).strip()
                # Update the StringVar only if it's different
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
                 # Check current state, don't just rely on self.template_source == "custom"
                 original_state = self.custom_template_entry.cget('state')
                 if original_state == tk.DISABLED:
                      self.custom_template_entry.config(state=tk.NORMAL)

                 # Delete current content and insert new content
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
        # --- FIX: Correct the command here to _save_configuration ---
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
            # Add the resolved path string to avoid issues with comparison later
            p_str = str(p)
            if p_str not in [str(x) for x in self.model_dirs]: # Compare resolved paths
                self.model_dirs.append(p) # Store as Path object
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
             # Find the actual path object in our list based on the resolved string
             actual_index = -1
             for i, p in enumerate(self.model_dirs):
                 try:
                      if str(p.resolve()) == dir_to_remove_str: # Compare against resolved path string
                          actual_index = i
                          break
                 except Exception:
                     pass # Ignore errors resolving path in the list


             if actual_index != -1:
                # Confirm with the user using the resolved path string
                if messagebox.askyesno("Confirm Remove", f"Remove directory:\n{str(self.model_dirs[actual_index].resolve())}?"):
                    del self.model_dirs[actual_index]
                    self._update_model_dirs_listbox()
                    self._save_configs()
                    self._trigger_scan() # Re-scan after removing a directory
             else:
                  # This case should ideally not happen if _update_model_dirs_listbox works correctly,
                  # but it's a safe fallback.
                  messagebox.showerror("Error", "Selected directory path mismatch or not found internally.")

        except IndexError:
            messagebox.showerror("Error", "Invalid selection.")
        except Exception as e:
             messagebox.showerror("Error", f"An error occurred removing directory:\n{e}")


    def _update_model_dirs_listbox(self):
        current_selection = self.model_dirs_listbox.curselection()
        selected_text = self.model_dirs_listbox.get(current_selection[0]) if current_selection else None

        self.model_dirs_listbox.delete(0, tk.END)
        # Always display resolved paths in the listbox
        displayed_paths_strs = []
        for p in self.model_dirs:
            try:
                 resolved_str = str(p.resolve()) # Resolve path for display and comparison
                 displayed_paths_strs.append(resolved_str)
                 self.model_dirs_listbox.insert(tk.END, resolved_str)
            except Exception as e:
                 print(f"Warning: Could not resolve path for display '{p}': {e}", file=sys.stderr)
                 # If resolving fails, display the original path string with a warning
                 displayed_paths_strs.append(f"[UNRESOLVABLE] {p}") # Add marker for display
                 self.model_dirs_listbox.insert(tk.END, f"[UNRESOLVABLE] {p}") # Add marker for display


        # Filter out unresolvable paths from the internal list for scanning
        self.model_dirs = [Path(p_str.replace("[UNRESOLVABLE] ", "")) for p_str in displayed_paths_strs if not p_str.startswith("[UNRESOLVABLE]")]

        # Attempt to re-select the previously selected item based on resolved path string
        if selected_text and selected_text in displayed_paths_strs:
            try:
                new_index = displayed_paths_strs.index(selected_text)
                self.model_dirs_listbox.selection_set(new_index)
                self.model_dirs_listbox.activate(new_index)
                self.model_dirs_listbox.see(new_index)
            except ValueError:
                pass # Previous selection not found in the new list

        # Update the height based on saved settings
        if hasattr(self, 'model_listbox') and self.model_listbox.winfo_exists():
             self.model_listbox.config(height=self.app_settings.get("model_list_height", 8))


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

        # Ensure self.model_dirs contains Path objects before scanning
        # It should already contain Path objects from _update_model_dirs_listbox, but double check
        self.model_dirs = [Path(d) for d in [str(p) for p in self.model_dirs] if d] # Re-create list of Paths


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
            if not isinstance(model_dir, Path) or not model_dir.is_dir(): continue
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
                        # Ensure the base name corresponds to the *first* part before adding
                        # Check if the resolved path points to this specific file
                        try: # <-- This try block starts a new indentation level
                             resolved_gguf_path = gguf_path.resolve()
                             if base_name not in processed_multipart_bases:
                                 # Store the path to the first part
                                 found[base_name] = resolved_gguf_path
                                 processed_multipart_bases.add(base_name)
                        # The 'except' needs to match the 'try' it belongs to
                        except Exception as resolve_exc: # <-- Corrected indentation
                             print(f"Warning: Could not resolve path '{gguf_path}' during scan: {resolve_exc}", file=sys.stderr) # Log resolve errors
                        # The 'continue' needs to align with the 'if first_part_match:' block
                        continue # <-- Corrected indentation

                    # Handle subsequent parts of multi-part files: mark base as processed but don't add
                    multi_match = multipart_pattern.match(filename)
                    if multi_match:
                        base_name = multi_match.group(1)
                        processed_multipart_bases.add(base_name)
                        # The 'continue' needs to align with the 'if multi_match:' block
                        continue # <-- Corrected indentation

                    # Handle single-part files (not matching the multi-part patterns)
                    if filename.lower().endswith(".gguf") and gguf_path.stem not in processed_multipart_bases:
                         display_name = gguf_path.stem
                         try: # <-- This try block starts a new indentation level
                             resolved_gguf_path = gguf_path.resolve()
                             # Only add if we haven't already added a multi-part version with the same base name
                             if display_name not in found:
                                found[display_name] = resolved_gguf_path
                         # The 'except' needs to match the 'try' it belongs to
                         except Exception as resolve_exc: # <-- Corrected indentation
                             print(f"Warning: Could not resolve path '{gguf_path}' during scan: {resolve_exc}", file=sys.stderr) # Log resolve errors


            except Exception as e:
                print(f"ERROR: Error scanning directory {model_dir}: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

        self.root.after(0, self._update_model_listbox_after_scan, found)
        
    # ═════════════════════════════════════════════════════════════════
    #  Model Selection & Analysis
    # ═════════════════════════════════════════════════════════════════

    def _update_model_listbox_after_scan(self, found_models_dict):
        """Populates the model listbox AFTER scan and handles selection restoration."""
        # Store found models with their resolved paths
        self.found_models = {name: path.resolve() for name, path in found_models_dict.items() if path.is_file()}
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
            self._update_recommendations() # Update based on no model
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
                      print("DEBUG: Previous analysis thread is still running, cancelling old analysis.", file=sys.stderr)
                      # Ideally, you'd have a way to signal the thread to stop.
                      # For simplicity here, we just let the old thread finish and ignore its result
                      # if a new analysis starts, by checking self.model_path in _update_ui_after_analysis.
                      pass # No explicit cancel mechanism here

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
            self._update_recommendations() # Update based on no model
            self._generate_default_config_name() # Generate default name for no model state


    def _run_gguf_analysis(self, model_path_str):
        """Worker function for background GGUF analysis."""
        print(f"Analyzing GGUF in background: {model_path_str}", file=sys.stderr)
        # Check if the currently selected model in the GUI still matches the one being analyzed
        # This prevents updating the UI with stale results if the user quickly selects another model
        if self.model_path.get() == model_path_str:
             analysis_result = analyze_gguf_model_static(model_path_str)
             # Only update UI if the model path hasn't changed while analyzing
             if self.model_path.get() == model_path_str:
                self.root.after(0, self._update_ui_after_analysis, analysis_result)
             else:
                print(f"DEBUG: Analysis for {model_path_str} finished, but model selection changed. Discarding result.", file=sys.stderr)
        else:
            print(f"DEBUG: Analysis started for {model_path_str}, but model selection changed before analysis began. Skipping.", file=sys.stderr)


    # ═════════════════════════════════════════════════════════════════
    #  Model Selection & Analysis - Updated Handler
    # ═════════════════════════════════════════════════════════════════
    def _update_ui_after_analysis(self, analysis_result):
        """Updates controls based on GGUF analysis results (runs in main thread)."""
        print("DEBUG: _update_ui_after_analysis running", file=sys.stderr)
        # Crucially, check if the analysis result is for the *currently selected* model
        if self.model_path.get() != analysis_result.get("path"):
             print("DEBUG: _update_ui_after_analysis received result for a different model. Ignoring.", file=sys.stderr)
             return # Ignore stale results

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
        # Using after ensures the checkboxes are visually updated first
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


    def _on_gpu_selection_changed(self, index):
         """Callback when a GPU checkbox is changed."""
         # This callback doesn't need to do much itself, as the state is held by the BooleanVar.
         # Its primary purpose is to trigger recalculation/update of things that depend on selected GPUs.
         print(f"DEBUG: GPU {index} selection changed. Recalculating recommendations...", file=sys.stderr)
         # Update the selected_gpus list in app_settings immediately
         self.app_settings["selected_gpus"] = [i for i, v in enumerate(self.gpu_vars) if v.get()]
         self._update_recommendations()
         self._generate_default_config_name() # Update default config name as it depends on selected GPUs
         self._save_configs() # Save selection change


    # ═════════════════════════════════════════════════════════════════
    #  Recommendation Logic
    # ═════════════════════════════════════════════════════════════════
    def _update_recommendations(self):
        """Updates recommended values displayed in the UI."""
        print("DEBUG: Updating recommendations...", file=sys.stderr)
        # Update current KV Cache Type display (always reflects current selection)
        self.model_kv_cache_type_var.set(self.cache_type_k.get())

        # --- Threads & Threads Batch Recommendation ---
        # Reco is always the detected CPU cores, based on the user's request pattern
        # Ensure these use the *actual* detected values
        self.recommended_threads_var.set(f"Recommended: {self.physical_cores} (Your CPU physical cores)")
        self.recommended_threads_batch_var.set(f"Recommended: {self.logical_cores} (Your CPU logical cores)")


        # --- Tensor Split Recommendation (VRAM-based) ---
        n_layers = self.current_model_analysis.get("n_layers")
        # Get the list of indices of the GPUs that are *currently checked* in the UI
        selected_gpu_indices = [i for i, v in enumerate(self.gpu_vars) if v.get()]
        num_selected_gpus = len(selected_gpu_indices)

        if n_layers is None or n_layers <= 0:
            self.recommended_tensor_split_var.set("N/A - Model layers unknown")
        elif num_selected_gpus <= 1:
            self.recommended_tensor_split_var.set("N/A - Need >1 GPU selected")
        elif not self.gpu_info['available'] or not self.detected_gpu_devices:
             self.recommended_tensor_split_var.set("N/A - CUDA not available or GPUs not detected")
        else:
            # Gather VRAM for *selected* GPUs, maintaining their order based on selected_gpu_indices
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
            float_layers_per_gpu = {gpu_id: (vram / total_selected_vram) for gpu_id, vram in selected_gpu_vram}

            # Calculate rounded layer distribution (integer values)
            rounded_layers = {gpu_id: math.floor(n_layers * proportion) for gpu_id, proportion in float_layers_per_gpu.items()}
            current_total_layers = sum(rounded_layers.values())
            remainder = n_layers - current_total_layers

            # Distribute the remainder layers to GPUs with the largest fractional parts
            # Sort GPUs by fractional part in descending order, based on selected_gpu_indices order
            # This ensures distribution bias follows the order the user selected, which might influence llama.cpp's internal assignment order
            fractional_parts_in_order = [(gpu_id, (n_layers * float_layers_per_gpu[gpu_id]) - rounded_layers[gpu_id]) for gpu_id in selected_gpu_indices]
            fractional_parts_sorted = sorted(fractional_parts_in_order, key=lambda item: item[1], reverse=True)

            # Use a list to track which GPU IDs have received an extra layer to handle remainders efficiently
            gpu_ids_receiving_extra = [item[0] for item in fractional_parts_sorted[:remainder]]


            for gpu_id_to_add_layer in gpu_ids_receiving_extra:
                 rounded_layers[gpu_id_to_add_layer] += 1


            # Ensure the final split list is in the order of *selected* GPU indices
            # Create a dictionary from the rounded_layers result, then build the list
            # using the order from selected_gpu_indices
            final_rounded_layers = {gpu_id: count for gpu_id, count in rounded_layers.items()}
            recommended_split_list = [final_rounded_layers.get(gpu_id, 0) for gpu_id in selected_gpu_indices] # Use .get(gpu_id, 0) defensively


            # Sanity check: does the sum of recommended layers equal total layers?
            if sum(recommended_split_list) != n_layers:
                 print(f"WARNING: Tensor split calculation error. Sum ({sum(recommended_split_list)}) != Total Layers ({n_layers})", file=sys.stderr)
                 # Fallback to an equal split if calculation failed
                 layers_per_gpu_equal = n_layers // num_selected_gpus
                 remainder_equal = n_layers % num_selected_gpus
                 equal_split_list = [layers_per_gpu_equal + (1 if i < remainder_equal else 0) for i in range(num_selected_gpus)]
                 recommended_split_str = "Calc Error/Approx: " + ",".join(map(str, equal_split_list)) # Add label indicating it's an approximation/error fallback
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
            # Add note about venv if CUDA is available but no venv is set and flash_attn is not available in GUI env
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
                        # Log unexpected output for debugging
                        print(f"DEBUG: Venv check unexpected stdout: {stdout}", file=sys.stderr)
                        print(f"DEBUG: Venv check stderr: {stderr}", file=sys.stderr)
                        status_message = "Status (Venv): Unexpected check result."
                elif process.returncode != 0:
                    # Check stdout even on non-zero return code, as the script might print status then exit with 1 on non-zero
                    if "FLASH_ATTN_NOT_INSTALLED" in stdout:
                         status_message = "Status (Venv): Not Installed (Python package missing)"
                    elif "FLASH_ATTN_CHECK_ERROR" in stdout:
                         # Extract just the error message part
                         error_part = stdout.split("FLASH_ATTN_CHECK_ERROR: ", 1)[-1]
                         status_message = f"Status (Venv): Check Error: {error_part}"
                    else:
                         print(f"DEBUG: Venv check failed stdout: {stdout}", file=sys.stderr)
                         print(f"DEBUG: Venv check failed stderr: {stderr}", file=sys.stderr)
                         status_message = f"Status (Venv): Process failed (Code {process.returncode}). See console."

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
    # Handled by _update_template_controls_state, _update_effective_template_display, etc. above


    # ═════════════════════════════════════════════════════════════════
    #  Custom Parameter Logic (NEW)
    # ═════════════════════════════════════════════════════════════════
    def _add_custom_parameter(self):
        """Adds the parameter string from the entry to the list."""
        param_string = self.custom_param_entry_var.get().strip()
        if not param_string:
            messagebox.showwarning("Warning", "Please enter a parameter string.")
            return

        self.custom_parameters_list.append(param_string)
        self._update_custom_parameters_listbox()
        self.custom_param_entry_var.set("") # Clear the entry field
        self._save_configs()
        print(f"DEBUG: Added custom parameter: '{param_string}'", file=sys.stderr)


    def _remove_custom_parameter(self):
        """Removes the selected parameter from the list."""
        selection = self.custom_parameters_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Select a parameter to remove.")
            return

        # Get the index of the selected item
        index = selection[0]
        # Get the string representation displayed in the listbox
        selected_param_display = self.custom_parameters_listbox.get(index)

        # Find the actual string in our internal list (handle potential duplicates)
        # It's safer to rebuild the list excluding the selected item
        new_list = []
        removed = False
        for item in self.custom_parameters_list:
             # Compare against the displayed string for removal
             if not removed and item == selected_param_display:
                 removed = True
                 print(f"DEBUG: Removed custom parameter: '{item}'", file=sys.stderr)
             else:
                 new_list.append(item)

        self.custom_parameters_list = new_list
        self._update_custom_parameters_listbox()
        self._save_configs()
        # If multiple identical items exist and user removes one, only the first one selected is removed


    def _update_custom_parameters_listbox(self):
        """Updates the listbox to display the current custom parameters list."""
        # Preserve selection if possible (less critical for simple list)
        current_selection = self.custom_parameters_listbox.curselection()
        selected_text = self.custom_parameters_listbox.get(current_selection[0]) if current_selection else None


        self.custom_parameters_listbox.delete(0, tk.END)
        for param_string in self.custom_parameters_list:
            self.custom_parameters_listbox.insert(tk.END, param_string)

        # Attempt to re-select the previously selected item
        if selected_text and selected_text in self.custom_parameters_list:
             try:
                  new_index = self.custom_parameters_list.index(selected_text)
                  self.custom_parameters_listbox.selection_set(new_index)
                  self.custom_parameters_listbox.activate(new_index)
                  self.custom_parameters_listbox.see(new_index)
             except ValueError:
                  pass # Item no longer exists


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
                # Use the selected model name from the listbox if available,
                # otherwise fall back to the path stem.
                selected_name = ""
                sel = self.model_listbox.curselection()
                if sel:
                    selected_name = self.model_listbox.get(sel[0])

                if selected_name:
                    raw_name = selected_name
                else:
                     raw_name = Path(model_path_str).stem # Get filename without extension


                # Sanitize the raw name for filename use
                safe_name = re.sub(r'[\\/*?:"<>| ]', '_', raw_name)
                safe_name = safe_name[:40].strip('_') # Truncate and clean
                if safe_name:
                    parts.append(safe_name)
                else:
                     parts.append("model") # Fallback if sanitization results in empty string
            except Exception:
                 parts.append("model") # Fallback on path error
        else:
            parts.append("default") # No model selected


        # 2. Key Parameters (add if NOT default)
        # Define defaults, ensuring threads match detected, and excluding chat template params
        # Note: These defaults are for *comparison* for generating the name,
        # not necessarily the llama.cpp default values.
        # We compare against values that represent the *absence* of the arg in the command line.
        default_params_for_name = {
            # Values that are omitted by _add_arg if they match these
            "cache_type_k":  "f16",
            "threads":       str(self.logical_cores), # Llama.cpp default for threads is logical cores
            "threads_batch": "4", # Llama.cpp default for threads-batch is 4
            "batch_size":    "512", # Llama.cpp default
            "ubatch_size":   "512", # Llama.cpp default
            "ctx_size":      2048,  # Llama.cpp default
            "seed":          "-1",  # Llama.cpp default
            "temperature":   "0.8", # Llama.cpp default
            "min_p":         "0.05",# Llama.cpp default
            "n_gpu_layers":  "0", # Llama.cpp default for n-gpu-layers
            "tensor_split":  "", # Omitted if empty string
            "main_gpu":      "0", # Llama.cpp default
            "prio":          "0", # Llama.cpp default
            "n_predict":     "-1", # Llama.cpp default
            # Booleans that are flags (present if True, absent if False)
            "ignore_eos":    False, # Default for --ignore-eos flag
            "flash_attn":    False, # Default for --flash-attn flag
            "no_mmap":       False, # Default for --no-mmap flag
            "mlock":         False, # Default for --mlock flag
            "no_kv_offload": False, # Default for --no-kv-offload flag
            "no_cnv":        False, # Default for --no-cnv flag
            # Chat template parameters and custom parameters are deliberately excluded from default name generation
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
            default_val = default_params_for_name.get(key) # Get the default value used for name generation

            # Special handling for GPU Layers: use the internal integer value for comparison effect
            if key == "n_gpu_layers":
                 # Use the integer value after clamping, not the raw entry string
                 gpu_layers_int = self.n_gpu_layers_int.get()
                 max_layers = self.max_gpu_layers.get()
                 # Compare the *effect* of the setting to the default (0 layers offloaded)
                 # If the internal clamped value is > 0, consider it non-default
                 if gpu_layers_int > 0:
                      if max_layers > 0 and gpu_layers_int == max_layers:
                           parts.append("gpu-all")
                      else:
                           parts.append(f"gpu={gpu_layers_int}")
                 # Note: If input was -1 and max_layers is 0, gpu_layers_int is 0, which is correctly treated as default
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
                 # Only add if the current value is non-empty AND it's different from the default
                 if current_val and (default_val is None or current_val != default_val):
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

        # Decide whether to set the config_name variable
        # Rule: If the current name is 'default_config' OR if the current name is empty,
        # OR if it starts with the model name part (and the model name part exists),
        # then replace it with the generated name, but *only* if the generated name is different.
        current_config_name = self.config_name.get().strip()
        # Use the *actual* model name part used in the generated name
        model_name_prefix_in_gen = parts[0] if parts and parts[0] != "default" else ""


        update_variable = False
        if current_config_name == "default_config":
             update_variable = True
        elif not current_config_name: # If currently empty
             update_variable = True
        elif model_name_prefix_in_gen and current_config_name.startswith(model_name_prefix_in_gen):
            # Check if the current name is *only* the model name prefix, or starts with it followed by '_'
            # This heuristic avoids overwriting names like "modelname_manualedit" if they start with the model name
            if current_config_name == model_name_prefix_in_gen or current_config_name.startswith(model_name_prefix_in_gen + "_"):
                 update_variable = True


        if update_variable and generated_name != current_config_name:
            self.config_name.set(generated_name)
            print("DEBUG: Updated config_name variable.", file=sys.stderr)
        elif not update_variable:
             print("DEBUG: Did not update config_name variable as it seems manually set.", file=sys.stderr)


    def _update_default_config_name_if_needed(self, *args):
        """Traced callback for variables that influence the default config name."""
        # This trace function is bound to variables that influence the generated config name.
        # It's called whenever those variables change.
        # We only want to regenerate and update the config name if the user hasn't
        # already manually set a custom name.
        # The _generate_default_config_name function already contains the logic
        # to decide whether to overwrite the current self.config_name value.
        # So we just call it here.
        # Use after(1) to prevent recursive trace calls on config_name update
        self.root.after(1, self._generate_default_config_name)


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
            # --- CHANGES FOR JSON TEMPLATES / DEFAULT OPTION ---
            # Save the new template source variable
            "template_source": self.template_source.get(),
            "predefined_template_name": self.predefined_template_name.get(),
            "custom_template_string": self.custom_template_string.get(),
            # --- NEW: Save Custom Parameters ---
            "custom_parameters": self.custom_parameters_list, # Save the list of strings
        }

        # Include selected_gpus directly in the config dictionary for easier loading from config tab
        # This is redundant with app_settings, but keeps config self-contained for this tab.
        cfg["gpu_indices"] = self.app_settings.get("selected_gpus", [])

        return cfg

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
        self.config_name.set(name) # Set the config name entry


        # --- NEW: Load new parameters ---
        self.ignore_eos.set(cfg.get("ignore_eos", False))
        self.n_predict.set(cfg.get("n_predict", "-1")) # Default -1 for backward compatibility
        # --- NEW: Load Custom Parameters ---
        # Default to empty list [] for backward compatibility with older configs
        self.custom_parameters_list = cfg.get("custom_parameters", [])
        self._update_custom_parameters_listbox() # Update the GUI listbox


        # --- CHANGES FOR JSON TEMPLATES / DEFAULT OPTION ---
        # Load template parameters
        # Default the source to 'default' for backward compatibility
        loaded_source = cfg.get("template_source", "default")
        self.template_source.set(loaded_source)

        # Default the predefined name to the *first* key in _all_templates if not found
        # This handles cases where the saved name might no longer exist in _all_templates
        default_predefined_key = list(self._all_templates.keys())[0] if self._all_templates else ""
        self.predefined_template_name.set(cfg.get("predefined_template_name", default_predefined_key))

        self.custom_template_string.set(cfg.get("custom_template_string", ""))

        # Update UI state and display based on loaded values
        # The trace on self.template_source should trigger _update_template_controls_state and _update_effective_template_display
        # No need to call explicitly here if trace is reliable.

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
                listbox_items = self.model_listbox.get(0, tk.END)
                if found_display_name and found_display_name in listbox_items: # Check if the display name exists in the current listbox items
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
    #  Executable Finding (Added missing method)
    # ═════════════════════════════════════════════════════════════════
    def _find_server_executable(self, llama_base_dir):
        """Finds the llama-server executable within the llama.cpp directory."""
        exe_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
        simple_exe_name = "server.exe" if sys.platform == "win32" else "server" # Sometimes built as just 'server'

        # Define common potential locations relative to the base directory
        # Use Path objects directly for platform-independent path joining
        search_paths_rel = [
            Path("."), # Current directory (might be where user launched from, useful for local builds)
            Path("build/bin/Release"),
            Path("build/bin"),
            Path("build"),
            Path("bin"),
            Path("server"), # Some build scripts might put it directly in 'server'
        ]

        # Search in common relative paths first
        for rel_path in search_paths_rel:
            # Check for the primary name
            full_path = llama_base_dir / rel_path / exe_name
            if full_path.is_file():
                print(f"DEBUG: Found server executable at: {full_path}", file=sys.stderr)
                return full_path.resolve() # Return the resolved path

            # Check for the simple name if the primary wasn't found and names differ
            if exe_name != simple_exe_name:
                 full_path_simple = llama_base_dir / rel_path / simple_exe_name
                 if full_path_simple.is_file():
                     print(f"DEBUG: Found simple server executable at: {full_path_simple}", file=sys.stderr)
                     return full_path_simple.resolve() # Return the resolved path


        # As a last resort, check if the base directory *itself* is the bin directory
        # and contains the executable directly. This handles cases where build puts it
        # directly in the root, although less common.
        direct_path = llama_base_dir / exe_name
        if direct_path.is_file():
             print(f"DEBUG: Found server executable directly in base dir: {direct_path}", file=sys.stderr)
             return direct_path.resolve()

        if exe_name != simple_exe_name:
             direct_path_simple = llama_base_dir / simple_exe_name
             if direct_path_simple.is_file():
                  print(f"DEBUG: Found simple server executable directly in base dir: {direct_path_simple}", file=sys.stderr)
                  return direct_path_simple.resolve()


        print(f"DEBUG: Server executable '{exe_name}' or '{simple_exe_name}' not found in {llama_base_dir} or common subdirectories.", file=sys.stderr)
        return None # Executable not found anywhere


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

        # This call now succeeds because _find_server_executable is added
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
        # Note: llama.cpp's --cache-type-v defaults to the value of --cache-type-k if not specified.
        # So just setting --cache-type-k is usually sufficient.
        if kv_cache_type_val and kv_cache_type_val != "f16":
             cmd.extend(["--cache-type-k", kv_cache_type_val])
             print(f"DEBUG: Adding --cache-type-k {kv_cache_type_val} (non-default)", file=sys.stderr)


        # --- Threads & Batching ---
        # Llama.cpp internal defaults: --threads=hardware_concurrency() (logical), --threads-batch=4
        # Note: The GUI default for --threads is physical cores, while llama.cpp default is logical.
        # The _add_arg helper needs to compare against the *llama.cpp* default for omission.
        # But the default *value* shown in the GUI should still be physical cores.
        # Let's compare against the llama.cpp default (logical cores) when deciding whether to *add* the arg.
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


        # --- Handle GPU arguments: Now ADDING BOTH if set ---
        tensor_split_val = self.tensor_split.get().strip()
        n_gpu_layers_val = self.n_gpu_layers.get().strip()

        # Add --tensor-split if the value is non-empty
        # Use _add_arg which handles the non-empty check
        self._add_arg(cmd, "--tensor-split", tensor_split_val, "") # Add if non-empty string is provided by user

        # Add --n-gpu-layers if the value is non-empty AND not the default "0" string
        # This argument will now be added regardless of the --tensor-split value
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

        # --- CHANGES FOR JSON TEMPLATES / DEFAULT OPTION ---
        # Add --chat-template option ONLY if the source is not "default" (llama.cpp decides)
        source = self.template_source.get()
        if source in ["predefined", "custom"]:
             effective_template = self.current_template_display.get().strip()
             if effective_template: # Only add the argument if the effective template string is non-empty
                  # No default_value check needed here because if it's empty, the arg isn't added anyway by the outer if
                  cmd.extend(["--chat-template", effective_template])
                  print(f"DEBUG: Adding --chat-template: {effective_template[:50]}...", file=sys.stderr)
             else:
                  print("DEBUG: Chat template source is predefined/custom, but effective template string is empty. Omitting --chat-template.", file=sys.stderr)
        else: # source == "default"
             print("DEBUG: Chat template source is 'Let llama.cpp Decide'. Omitting --chat-template.", file=sys.stderr)

        # --- NEW: Add Custom Parameters ---
        print(f"DEBUG: Adding {len(self.custom_parameters_list)} custom parameters...", file=sys.stderr)
        for param_string in self.custom_parameters_list:
            try:
                # Use shlex.split to correctly parse potentially quoted arguments
                # shlex.split will split "--param value with spaces" into ["--param", "value with spaces"]
                split_params = shlex.split(param_string)
                cmd.extend(split_params)
                print(f"DEBUG: Added custom param: {param_string} -> {split_params}", file=sys.stderr)
            except Exception as e:
                print(f"WARNING: Could not parse custom parameter '{param_string}': {e}. Skipping.", file=sys.stderr)
                messagebox.showwarning("Custom Parameter Warning", f"Could not parse custom parameter '{param_string}': {e}\nIt will be ignored.")


        # Add a note about using CUDA_VISIBLE_DEVICES if they selected specific GPUs via checkboxes
        # but are NOT using --tensor-split (which explicitly lists devices/split).
        # This warning is helpful because llama.cpp might use all GPUs by default unless restricted by env var or tensor-split.
        selected_gpu_indices = [i for i, v in enumerate(self.gpu_vars) if v.get()]
        detected_gpu_count = self.gpu_info.get("device_count", 0)
        selected_indices_str = ",".join(map(str, sorted(selected_gpu_indices)))


        # Only warn if GPUs were detected, the user selected a *subset*, and --tensor-split is not used.
        if detected_gpu_count > 0 and len(selected_gpu_indices) > 0 and len(selected_gpu_indices) < detected_gpu_count and not tensor_split_val:
             # Only warn if the user explicitly selected a *subset* of GPUs using the checkboxes AND didn't use tensor-split
             print(f"\nINFO: Specific GPUs ({selected_indices_str}) were selected via checkboxes, but --tensor-split was not used.", file=sys.stderr)
             # The PowerShell script will set CUDA_VISIBLE_DEVICES, so the warning applies more generally now.
             print("      llama-server might default to using all available GPUs unless restricted by CUDA_VISIBLE_DEVICES environment variable.", file=sys.stderr)
             if sys.platform != "win32":
                  # Only print the bash/export example on Linux/macOS if needed
                  print(f"      To restrict server to GPUs {selected_indices_str}, set CUDA_VISIBLE_DEVICES={selected_indices_str} environment variable *before* launching (e.g., 'export CUDA_VISIBLE_DEVICES={selected_indices_str}' on Linux/macOS bash).", file=sys.stderr)
             else:
                 # On Windows, the script *will* set it if GPUs are selected, but reinforce
                  print(f"      The generated PowerShell script will set CUDA_VISIBLE_DEVICES={selected_indices_str}.", file=sys.stderr)

             print("      Alternatively, use --tensor-split to explicitly assign layers.", file=sys.stderr)
        elif len(selected_gpu_indices) > 0 and detected_gpu_count > 0:
             # If GPUs were selected (and there are GPUs), maybe a general reminder about env vars?
             # Or just assume the user knows if they selected. Keep the message only for subset selection without tensor-split.
             pass

        # Keep the info message about precedence if tensor-split is present, as the server will likely still follow it.
        if tensor_split_val:
             print(f"INFO: --tensor-split is set ('{tensor_split_val}'), this usually takes precedence over --n-gpu-layers for layer distribution.", file=sys.stderr)


        print("\n--- Generated Command ---", file=sys.stderr)
        # Use shlex.quote to make command printable and copy-pasteable in shells
        # Note: This quoting is primarily for display. The actual launch script
        # might need different quoting depending on the shell (batch, ps1, bash).
        quoted_cmd = [shlex.quote(arg) for arg in cmd]
        print(" ".join(quoted_cmd), file=sys.stderr)
        print("-------------------------\n", file=sys.stderr)

        return cmd

    def _add_arg(self, cmd_list, arg_name, value, default_value=None):
        """
        Adds argument to cmd list if its value is non-empty or, if a default is given,
        if the value is different from the default. Handles bools.
        """
        # Handle boolean flags (always added as just the flag name if True)
        is_bool_var = isinstance(value, tk.BooleanVar)
        is_bool_py = isinstance(value, bool)

        if is_bool_var:
            if value.get():
                 cmd_list.append(arg_name)
                 print(f"DEBUG: Adding flag '{arg_name}' (True)", file=sys.stderr)
            else:
                 print(f"DEBUG: Omitting flag '{arg_name}' (False)", file=sys.stderr)
            return

        if is_bool_py:
             if value:
                  cmd_list.append(arg_name)
                  print(f"DEBUG: Adding flag '{arg_name}' (True)", file=sys.stderr)
             else:
                  print(f"DEBUG: Omitting flag '{arg_name}' (False)", file=sys.stderr)
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
        # Add the arg and value if actual_value_str is non-empty AND (default_value_str is None OR actual_value_str != default_value_str).

        if actual_value_str: # Check if the user entered *any* value
             if default_value_str is None or actual_value_str != default_value_str:
                 # Add the argument if there's no default to compare against,
                 # OR if the user's value is different from the default.
                 cmd_list.extend([arg_name, actual_value_str])
                 print(f"DEBUG: Adding '{arg_name} {actual_value_str}' (non-default)", file=sys.stderr)
             else:
                 # User entered the exact default value. Omit the argument.
                 print(f"DEBUG: Omitting '{arg_name} {actual_value_str}' as it matches default '{default_value_str}'", file=sys.stderr)
        else:
             # User entered an empty string. Omit the argument.
             print(f"DEBUG: Omitting '{arg_name}' due to empty value. Default '{default_value_str}' will be used.", file=sys.stderr)


    # ═════════════════════════════════════════════════════════════════
    #  launch & script helpers
    # ═════════════════════════════════════════════════════════════════
    # The static cleanup method is defined at the top of the class now

    # This is the function that contained the SyntaxError due to incorrect try/except/finally nesting
    def launch_server(self):
        cmd_list = self._build_cmd()
        if not cmd_list:
            # _build_cmd already showed an error message
            return

        venv_path_str = self.venv_dir.get().strip()
        use_venv = bool(venv_path_str)

        tmp_path = None # Initialize tmp_path outside try/except/finally

        # Get selected GPU indices for CUDA_VISIBLE_DEVICES
        selected_gpu_indices = [i for i, v in enumerate(self.gpu_vars) if v.get()]
        cuda_devices_value = ",".join(map(str, sorted(selected_gpu_indices)))


        try: # Main try block for creating script and launching process
            if sys.platform == "win32":
                # --- MODIFIED: Use temporary PowerShell script instead of batch file ---
                import tempfile
                # Use mkstemp to create a secure temporary file with .ps1 suffix
                # Use text=True and encoding='utf-8' for cross-platform safety, although file handle needs closing
                fd, tmp_path = tempfile.mkstemp(suffix=".ps1", prefix="llamacpp_launch_", text=False)
                os.close(fd) # Close the file descriptor immediately

                # Use utf-8 encoding explicitly as templates can contain wide chars
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write("# Automatically generated PowerShell script from LLaMa.cpp Server Launcher GUI\n\n")
                    f.write("$ErrorActionPreference = 'Continue'\n\n")
                    # Set console output encoding to UTF-8
                    f.write('[Console]::OutputEncoding = [System.Text.Encoding]::UTF8 # Set console output encoding to UTF-8\n\n')

                    # --- Add CUDA_VISIBLE_DEVICES setting if GPUs are selected ---
                    if cuda_devices_value:
                        f.write(f'Write-Host "Setting CUDA_VISIBLE_DEVICES={cuda_devices_value}" -ForegroundColor DarkCyan\n')
                        f.write(f'$env:CUDA_VISIBLE_DEVICES="{cuda_devices_value}"\n\n')
                    elif self.gpu_info.get("device_count", 0) > 0:
                         # If GPUs are detected but none are selected, explicitly unset the variable
                         # to rely on default llama.cpp behavior or let the OS handle it.
                         # This avoids accidentally inheriting a variable from the environment.
                         f.write('Write-Host "Clearing CUDA_VISIBLE_DEVICES environment variable." -ForegroundColor DarkCyan\n')
                         f.write('Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue\n\n')


                    if use_venv:
                        venv_path = Path(venv_path_str).resolve()
                        # Check for Activate.ps1
                        act_script = venv_path / "Scripts" / "Activate.ps1"
                        if not act_script.is_file():
                            messagebox.showerror("Error", f"Venv activation script (Activate.ps1) not found:\n{act_script}")
                            # Note: Cleanup will be attempted in finally if tmp_path was created
                            # Clean up temporary file before returning on error
                            if tmp_path is not None:
                                 try: Path(tmp_path).unlink()
                                 except OSError as e: print(f"Warning: Failed to delete temporary script {tmp_path} after venv error: {e}", file=sys.stderr)
                            return # Exit the launch process

                        # Use dot-sourcing (. .\path\to\Activate.ps1) to activate in the current shell
                        # Format path with forward slashes for PowerShell compatibility and quote it
                        ps_act_path = str(act_script.as_posix())
                        # Use single quotes for the path itself in the script if it contains spaces/special chars
                        # For dot sourcing, path must be quoted if it contains spaces or special chars
                        # A simpler approach might be using double quotes with escaping if needed
                        # Let's stick to robust double quoting for the path
                        quoted_ps_act_path = f'"{ps_act_path.replace('"', '`"').replace('`', '``')}"' # Escape double quotes and backticks


                        f.write(f'Write-Host "Activating virtual environment: {venv_path_str}" -ForegroundColor Cyan\n')
                        # Use try/catch to report activation errors but continue
                        f.write(f'try {{ . {quoted_ps_act_path} }} catch {{ Write-Warning "Failed to activate venv: $($_.Exception.Message)"; $global:LASTEXITCODE=1; Start-Sleep -Seconds 2 }}\n\n') # Use global:LASTEXITCODE and pause on error


                    f.write(f'Write-Host "Launching llama-server..." -ForegroundColor Green\n')

                    # --- Build the command string for PowerShell using appropriate quoting ---
                    # Join the cmd_list parts with spaces, and then quote each part individually for PowerShell
                    # This handles parameters like --arg "value with spaces" correctly after shlex.split
                    ps_cmd_parts = []
                    # First argument is the executable path, handle specially with '&'.
                    exe_path_obj = Path(cmd_list[0]).resolve() # Resolve path for robustness
                    # Format path with forward slashes and quote it for PowerShell
                    # Escape internal quotes and backticks for safety within the string literal passed to PowerShell
                    quoted_exe_path_ps = str(exe_path_obj.as_posix()).replace('"', '`"').replace('`', '``')
                    ps_cmd_parts.append(f'& "{quoted_exe_path_ps}"') # Use double quotes for the path

                    # Process remaining arguments (from cmd_list, already includes custom params)
                    # For each argument in cmd_list (after the executable), quote it for PowerShell
                    # Use double quotes by default, escaping internal quotes and backticks
                    # Special case --chat-template needs value in single quotes.
                    for arg in cmd_list[1:]:
                         if arg == "--chat-template" and len(cmd_list) > cmd_list.index("--chat-template") + 1:
                             # This check is a bit redundant as build_cmd should ensure a value exists,
                             # but defensive coding is good. Need to find the *next* argument in cmd_list.
                             # Find index of the flag and get the next element
                             try:
                                 flag_index = cmd_list.index("--chat-template")
                                 if flag_index + 1 < len(cmd_list):
                                     template_string = cmd_list[flag_index + 1]
                                     # Escape internal single quotes by doubling them (' becomes '')
                                     escaped_template_string = template_string.replace("'", "''")
                                     # Enclose the *entire* escaped template string in single quotes (') for the final PS command string
                                     quoted_template_arg = f"'{escaped_template_string}'"

                                     ps_cmd_parts.append("--chat-template")
                                     ps_cmd_parts.append(quoted_template_arg)
                                     # Need to skip the *next* element in the loop's iteration
                                     # A simple loop like this is tricky. Let's reconstruct differently.

                                     # Reworking the loop structure to handle pairs like --chat-template
                                     # Re-initialize ps_cmd_parts and loop index
                                     ps_cmd_parts = []
                                     ps_cmd_parts.append(f'& "{quoted_exe_path_ps}"') # Add executable again

                                     i = 1 # Start index after the executable
                                     while i < len(cmd_list):
                                         current_arg = cmd_list[i]
                                         if current_arg == "--chat-template" and i + 1 < len(cmd_list):
                                              template_string = cmd_list[i+1]
                                              escaped_template_string = template_string.replace("'", "''")
                                              quoted_template_arg = f"'{escaped_template_string}'"
                                              ps_cmd_parts.append("--chat-template")
                                              ps_cmd_parts.append(quoted_template_arg)
                                              i += 2 # Skip both flag and value
                                         else:
                                              # Standard quoting for other args
                                              quoted_arg = f'"{current_arg.replace('"', '`"').replace('`', '``')}"'
                                              ps_cmd_parts.append(quoted_arg)
                                              i += 1
                                     break # Exit this inner while loop once reconstructed

                             except ValueError: # --chat-template not found, should not happen here
                                 pass # Continue with the original loop if this somehow occurs


                         else:
                             # Standard quoting for other args
                             quoted_arg = f'"{arg.replace('"', '`"').replace('`', '``')}"'
                             ps_cmd_parts.append(quoted_arg)


                    # After processing all args, write the final command string
                    f.write(" ".join(ps_cmd_parts) + "\n\n") # Join all parts and write


                    # Add error check after the command in case llama-server returns a non-zero exit code
                    # Check $LASTEXITCODE, not global:LASTEXITCODE for the server process itself
                    f.write('if ($LASTEXITCODE -ne 0) {\n')
                    f.write('    Write-Error "Llama-server exited with error code: $LASTEXITCODE."\n')
                    f.write('    $global:LASTEXITCODE = $LASTEXITCODE # Propagate error code\n') # Propagate for the pause logic
                    f.write('}\n')
                    f.write('Write-Host "Server process likely finished or detached." -ForegroundColor Yellow\n')
                    # Pause if script is run directly by double-clicking or outside an interactive shell
                    # Also pause if an error occurred ($global:LASTEXITCODE -ne 0)
                    f.write('if ($Host.Name -eq "ConsoleHost" -or $global:LASTEXITCODE -ne 0) {\n')
                    f.write('    Read-Host -Prompt "Press Enter to close..."\n')
                    f.write('}\n')


                # Launch the temporary PowerShell file in a new console window
                # Use shell=False and explicitly call powershell.exe
                # Use CREATE_NEW_CONSOLE flag to ensure a new window is opened
                print(f"DEBUG: Launching Windows PowerShell script: {tmp_path}", file=sys.stderr)
                # Use the full path to powershell.exe if needed, but it's usually in PATH
                # Using -File is crucial to execute the script correctly
                # Use resolved path for reliability
                subprocess.Popen(['powershell.exe', '-ExecutionPolicy', 'Bypass', '-File', str(Path(tmp_path).resolve())],
                                shell=False, creationflags=subprocess.CREATE_NEW_CONSOLE)


            else: # Linux/macOS (Existing logic using bash -c)
                # The quoting logic for bash -c using shlex.quote is generally robust.
                # We don't need to special-case --chat-template here using single quotes
                # because shlex.quote handles embedding complex strings correctly for bash.

                # Ensure command parts are quoted correctly for bash -c
                # cmd_list already includes custom parameters and other standard args.
                # We just need to quote *each element* of the final cmd_list.
                quoted_cmd_parts = [shlex.quote(arg) for arg in cmd_list]
                server_command_str = " ".join(quoted_cmd_parts)

                # Check if stdout is connected to a terminal (-t 1)
                # Use 'read -r' to prevent backslash interpretation
                # Use || true to make the read command succeed even if cancelled (like Ctrl+C)
                # Combine commands with '&&' so pause only happens if server command succeeds (or detaches)
                # Let's add a check for the last command's exit status ($?) and only pause on non-zero exit or if run interactively
                # bash pause: read -p "Press Enter to close..." -r || true
                # Combined bash script with error check and pause:
                # `command ; exit_status=$? ; if [ -t 1 ] || [ $exit_status -ne 0 ]; then read -rp "Press Enter to close..." ; fi ; exit $exit_status`
                # Or simpler: `command ; command_status=$? ; if [[ -t 1 || $command_status -ne 0 ]]; then read -rp "Press Enter to close..." </dev/tty ; fi ; exit $command_status`
                # Using </dev/tty ensures read prompts even if stdout is redirected.

                if use_venv:
                    venv_path = Path(venv_path_str).resolve()
                    act_script = venv_path / "bin" / "activate"
                    if not act_script.is_file():
                        messagebox.showerror("Error", f"Venv activation script not found:\n{act_script}")
                        return # Exit the launch process

                    # Build the core command that will be sourced
                    # sourced_command = f'source {shlex.quote(str(act_script))} && echo "Virtual environment activated." && echo "Launching server..." && {server_command_str}'
                    # It's better to source first, then execute the command in the activated shell
                    # The full command string passed to bash -c needs careful quoting.
                    # We pass a script string like: 'source ... && command ; exit_status=$? ; if ... read ; fi ; exit $exit_status'
                    # Need to shlex.quote the *entire* argument string passed to bash -c.
                    full_script_content = f'source {shlex.quote(str(act_script))} && echo "Virtual environment activated." && echo "Launching server..." && {server_command_str} ; command_status=$? ; if [[ -t 1 || $command_status -ne 0 ]]; then read -rp "Press Enter to close..." </dev/tty ; fi ; exit $command_status'
                    quoted_full_command_for_bash = shlex.quote(full_script_content)

                else: # No venv
                    full_script_content = f'echo "Launching server..." && {server_command_str} ; command_status=$? ; if [[ -t 1 || $command_status -ne 0 ]]; then read -rp "Press Enter to close..." </dev/tty ; fi ; exit $command_status'
                    quoted_full_command_for_bash = shlex.quote(full_script_content)

                # --- CUDA_VISIBLE_DEVICES for Linux/macOS (Bash) ---
                # Add export CUDA_VISIBLE_DEVICES="..." *before* the main command in the bash script
                # This should happen *after* venv activation if used.
                cuda_env_line = ""
                if cuda_devices_value:
                    # Quote the value in case it contains spaces or special characters (unlikely for indices but safe)
                    quoted_cuda_value = shlex.quote(cuda_devices_value)
                    cuda_env_line = f'export CUDA_VISIBLE_DEVICES={quoted_cuda_value} && echo "Setting CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"'
                elif self.gpu_info.get("device_count", 0) > 0:
                    # Unset if GPUs detected but none selected
                     cuda_env_line = f'unset CUDA_VISIBLE_DEVICES && echo "Clearing CUDA_VISIBLE_DEVICES environment variable."'

                # Integrate the CUDA env line into the full script content
                if cuda_env_line:
                    # Place it after source but before the server command
                    if use_venv:
                        # Split the script after source and first echo
                        parts = full_script_content.split('&& echo "Virtual environment activated." &&', 1)
                        if len(parts) == 2:
                             # Reconstruct with the cuda line inserted
                             full_script_content = f"{parts[0]}&& echo \"Virtual environment activated.\" && {cuda_env_line} && {parts[1]}"
                             quoted_full_command_for_bash = shlex.quote(full_script_content) # Re-quote the whole thing
                        else:
                             # Fallback if split failed
                             print("WARNING: Could not insert CUDA_VISIBLE_DEVICES after venv activation. Adding before everything.", file=sys.stderr)
                             full_script_content = f'{cuda_env_line} && {full_script_content}'
                             quoted_full_command_for_bash = shlex.quote(full_script_content) # Re-quote

                    else: # No venv, just prepend
                        full_script_content = f'{cuda_env_line} && {full_script_content}'
                        quoted_full_command_for_bash = shlex.quote(full_script_content) # Re-quote


                # Attempt to launch in a new terminal window
                # Use 'bash -c' to execute the command string.
                # Find common terminal emulators.
                terminals = ['gnome-terminal', 'konsole', 'xfce4-terminal', 'xterm', 'iterm'] # Add iTerm for macOS
                launched = False
                import shutil # For shutil.which
                for term in terminals:
                    term_path = shutil.which(term)
                    if term_path:
                        # Use the full path to the terminal executable
                        term_cmd_base = [str(Path(term_path).resolve())] # Resolve terminal path too
                        # Pass the command string to bash -c as a single argument
                        # Note: This doesn't need shell=True
                        # Different terminals use different flags to execute a command and stay open.
                        # -e for gnome-terminal/xfce4-terminal
                        # -e for xterm (though often just the command as args works)
                        # --noclose -e for konsole
                        # -e followed by command for iterm
                        # Let's try common patterns, starting with -e
                        term_cmds = []
                        if term in ['gnome-terminal', 'xfce4-terminal', 'iterm']:
                             # gnome-terminal/xfce4-terminal/iterm expect -e followed by command string or list
                             # Pass 'bash -c command_string' as two arguments to -e
                             term_cmds.append(term_cmd_base + ['-e', 'bash', '-c', quoted_full_command_for_bash])
                        elif term == 'konsole':
                             # Konsole needs --noclose and -e followed by the command list or string
                             term_cmds.append(term_cmd_base + ['--noclose', '-e', 'bash -c ' + quoted_full_command_for_bash]) # Requires bash -c inside quotes
                        elif term == 'xterm':
                             # xterm can often take the command directly after its own flags
                             term_cmds.append(term_cmd_base + ['-e', 'bash -c ' + quoted_full_command_for_bash]) # Requires bash -c inside quotes
                             # Another xterm pattern
                             term_cmds.append(term_cmd_base + ['-e', 'bash', '-c', quoted_full_command_for_bash])


                        for term_cmd_parts in term_cmds:
                            print(f"DEBUG: Attempting launch with terminal: {term_cmd_parts}", file=sys.stderr)
                            try:
                                subprocess.Popen(term_cmd_parts, shell=False)
                                launched = True
                                break # Stop after the first successful launch
                            except FileNotFoundError:
                                print(f"DEBUG: Terminal '{term}' not found or not executable using shell=False with these flags.", file=sys.stderr)
                                continue # Try next set of flags or next terminal
                            except Exception as term_err:
                                print(f"DEBUG: Failed to launch with {term} using shell=False: {term_err}", file=sys.stderr)
                                continue # Try next set of flags or next terminal

                        if launched: break # Stop checking terminals after successful launch


                if not launched:
                    # Fallback: Try launching directly without a specific terminal wrapper.
                    print("WARNING: Could not find a supported terminal emulator. Attempting direct launch.", file=sys.stderr)
                    messagebox.showwarning("Terminal Not Found", "Could not find a supported terminal emulator (gnome-terminal, konsole, xfce4-terminal, xterm, iterm).\nAttempting to launch directly.\n\nThe server output might appear in the GUI's console or launch in the background.")

                    try:
                        # For direct launch with 'source' and other shell features, shell=True is needed.
                        # Use the fully quoted command string.
                        subprocess.Popen(quoted_full_command_for_bash, shell=True)
                        launched = True # Mark as launched even if it's a fallback method

                    except Exception as direct_launch_err:
                        messagebox.showerror("Launch Error", f"Failed to launch server directly:\n{direct_launch_err}")
                        print(f"ERROR: Failed to launch server directly: {direct_launch_err}", file=sys.stderr)
                        traceback.print_exc(file=sys.stderr)
                        launched = False # Mark as failed if fallback also fails


                if not launched:
                    # Final fallback/error message if direct launch also failed
                    messagebox.showerror("Launch Error", "Could not find a supported terminal or launch the server script directly.")


        except Exception as exc: # Catch any errors during script writing or initial subprocess launch setup
            messagebox.showerror("Launch Error", f"An unexpected error occurred during launch preparation:\n{exc}")
            print(f"Unexpected error during launch preparation: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

        finally:
            # Clean up the temporary file AFTER the Popen call returns, or if an error occurred before Popen.
            # The cleanup itself runs in a separate thread after a delay.
            # Only attempt cleanup if a temporary path was successfully created.
            if sys.platform == "win32" and tmp_path is not None:
                # Start cleanup thread. It's daemon so it won't prevent app exit.
                # Use the class name to call the static method
                cleanup_thread = Thread(target=LlamaCppLauncher.cleanup, args=(tmp_path,), daemon=True)
                cleanup_thread.start()


    # ═════════════════════════════════════════════════════════════════
    #  save ps1 script
    # ═════════════════════════════════════════════════════════════════
    # Fixed structure: method definition is outside other methods
    def save_ps1_script(self):
        cmd_list = self._build_cmd()
        if not cmd_list: return

        # --- FIX: Use the actual selected model name from the listbox ---
        selected_model_name = ""
        selection = self.model_listbox.curselection()
        if selection:
             selected_model_name = self.model_listbox.get(selection[0])


        default_name = "launch_llama_server.ps1"
        if selected_model_name: # Check the correct variable here
            # Sanitize model name for filename
            model_name_part = re.sub(r'[\\/*?:"<>| ]', '_', selected_model_name)
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
                fh.write('[Console]::OutputEncoding = [System.Text.Encoding]::UTF8 # Set console output encoding to UTF-8\n\n')

                # --- Add CUDA_VISIBLE_DEVICES setting if GPUs are selected ---
                selected_gpu_indices = [i for i, v in enumerate(self.gpu_vars) if v.get()]
                cuda_devices_value = ",".join(map(str, sorted(selected_gpu_indices)))

                if cuda_devices_value:
                    fh.write(f'Write-Host "Setting CUDA_VISIBLE_DEVICES={cuda_devices_value}" -ForegroundColor DarkCyan\n')
                    fh.write(f'$env:CUDA_VISIBLE_DEVICES="{cuda_devices_value}"\n\n')
                elif self.gpu_info.get("device_count", 0) > 0:
                     # If GPUs are detected but none are selected, explicitly unset the variable
                     fh.write('Write-Host "Clearing CUDA_VISIBLE_DEVICES environment variable." -ForegroundColor DarkCyan\n')
                     fh.write('Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue\n\n')


                venv = self.venv_dir.get().strip()
                if venv:
                    try:
                        # --- FIX: Correct the typo 'vvenv_path' to 'venv_path' ---
                        venv_path = Path(venv).resolve() # Resolve venv path for script
                        act_script = venv_path / "Scripts" / "Activate.ps1"
                        if act_script.exists():
                            # Use literal path syntax for PowerShell activation script source
                            # Ensure path is correctly formatted for PowerShell ('/' separators often work better)
                            ps_act_path = str(act_script.as_posix())
                            # Quote path using double quotes and escape internal quotes/backticks
                            quoted_ps_act_path = f'"{ps_act_path.replace('"', '`"').replace('`', '``')}"'


                            fh.write(f'Write-Host "Activating virtual environment: {venv}" -ForegroundColor Cyan\n')
                            # Use 'try/catch' to report activation errors but continue if not critical
                            # Use a quoted string for the path in the command
                            fh.write(f'try {{ . {quoted_ps_act_path} }} catch {{ Write-Warning "Failed to activate venv: $($_.Exception.Message)"; $global:LASTEXITCODE=1; Start-Sleep -Seconds 2 }}\n\n') # Add exit code on failure and pause

                        else:
                            # Format path for warning message
                            warn_act_script_path = str(act_script).replace("'", "''") # Escape single quotes for PowerShell string
                            fh.write(f'Write-Warning "Virtual environment activation script (Activate.ps1) not found at: \'{warn_act_script_path}\'"\n\n')
                    except Exception as path_ex:
                         # Format path for warning message
                         warn_venv_path = venv.replace("'", "''")
                         fh.write(f'Write-Warning "Could not process venv path \'{warn_venv_path}\': {path_ex}"\n\n')


                fh.write(f'Write-Host "Launching llama-server..." -ForegroundColor Green\n')

                # --- Build the command string for PowerShell using appropriate quoting ---
                # Similar logic as launch_server for Windows
                ps_cmd_parts = []
                # First argument is the executable path, handle specially with '&'.
                exe_path_obj = Path(cmd_list[0]).resolve() # Resolve path for reliability.
                # Format path with forward slashes for PowerShell compatibility
                # Also escape internal quotes and backticks for safety within the string literal
                quoted_exe_path_ps = str(exe_path_obj.as_posix()).replace('"', '`"').replace('`', '``')
                ps_cmd_parts.append(f'& "{quoted_exe_path_ps}"') # Still use double quotes here as it's a path

                # Process remaining arguments (from cmd_list, already includes custom params)
                # Reworking the loop structure to handle pairs like --chat-template
                i = 1 # Start index after the executable
                while i < len(cmd_list):
                    current_arg = cmd_list[i]
                    if current_arg == "--chat-template" and i + 1 < len(cmd_list):
                         template_string = cmd_list[i+1]
                         escaped_template_string = template_string.replace("'", "''")
                         quoted_template_arg = f"'{escaped_template_string}'"
                         ps_cmd_parts.append("--chat-template")
                         ps_cmd_parts.append(quoted_template_arg)
                         i += 2 # Skip both flag and value
                    else:
                         # Standard quoting for other args
                         quoted_arg = f'"{current_arg.replace('"', '`"').replace('`', '``')}"'
                         ps_cmd_parts.append(quoted_arg)
                         i += 1


                fh.write(" ".join(ps_cmd_parts) + "\n\n") # Join all parts and write

                # Check for non-zero exit code after the command
                # Check $LASTEXITCODE, not global:LASTEXITCODE for the server process itself
                fh.write('if ($LASTEXITCODE -ne 0) {\n')
                fh.write('    Write-Error "Llama-server exited with error code: $LASTEXITCODE."\n')
                fh.write('    $global:LASTEXITCODE = $LASTEXITCODE # Propagate error code\n') # Propagate for the pause logic
                fh.write('}\n')
                fh.write('Write-Host "Server process likely finished or detached." -ForegroundColor Yellow\n')
                fh.write('# Pause if script is run directly by double-clicking or outside an interactive shell\n')
                # Check if the host is ConsoleHost (typical when double-clicking or run from explorer)
                # Also pause if an error occurred ($global:LASTEXITCODE -ne 0)
                fh.write('if ($Host.Name -eq "ConsoleHost" -or $global:LASTEXITCODE -ne 0) {\n')
                fh.write('    Read-Host -Prompt "Press Enter to close..."\n') # Added ... for consistency
                fh.write('}\n')


            messagebox.showinfo("Saved", f"PowerShell script written to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Script Save Error", f"Could not save script:\n{exc}")
            print(f"Script Save Error: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


    # ═════════════════════════════════════════════════════════════════
    #  on_exit
    # ═════════════════════════════════════════════════════════════════
    # Fixed structure: method definition is outside other methods
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
                     if not theme.startswith('forest') and ('dark' in theme.lower() or 'black' in theme.lower() or 'highcontrast' in theme.lower()):
                          try:
                               # Get background/foreground from the theme's defaults
                               # Use lookup with default to avoid errors if element doesn't exist in theme
                               # Note: These lookups might only work well *after* theme_use.
                               bg_color = style.lookup('.', 'background', default='#2b2b2b') # Use root style lookup
                               fg_color = style.lookup('TLabel', 'foreground', default='#ffffff')
                               entry_bg = style.lookup('TEntry', 'fieldbackground', default='#3c3c3c')
                               entry_fg = style.lookup('TEntry', 'foreground', default='#ffffff')
                               # Fallback specific colors if theme lookup fails (more robust)
                               if not bg_color: bg_color = '#2b2b2b'
                               if not fg_color: fg_color = '#ffffff'
                               if not entry_bg: entry_bg = '#3c3c3c'
                               if not entry_fg: entry_fg = '#ffffff'


                               # Listbox is tk, not ttk, needs option_add or direct config if available
                               listbox_bg = style.lookup('TListbox', 'background', default=entry_bg) or entry_bg
                               listbox_fg = style.lookup('TListbox', 'foreground', default=entry_fg) or entry_fg
                               listbox_select_bg = style.lookup('TListbox', 'selectbackground', default='#505050') or '#505050'
                               listbox_select_fg = style.lookup('TListbox', 'selectforeground', default='#ffffff') or '#ffffff'

                               # Apply to root and all widgets implicitly
                               root.configure(bg=bg_color)
                               # ttk widgets inherit from the style root, but apply explicit colors for safety
                               # Note: Style configuration is complex and theme-dependent.
                               # Applying these might override theme specifics.
                               # It's often better to just let the theme handle it if possible.
                               # Let's try just configuring the root and Listbox/ScrolledText explicitly as they aren't ttk widgets.
                               # The 'forest' themes handle this automatically. Only needed for fallback themes.

                               # Configure general style for implicit inheritance
                               style.configure('.', background=bg_color, foreground=fg_color)
                               # Explicitly configure some widget types
                               style.configure('TFrame', background=bg_color)
                               style.configure('TLabel', background=bg_color, foreground=fg_color)
                               style.configure('TCheckbutton', background=bg_color, foreground=fg_color)
                               style.configure('TRadiobutton', background=bg_color, foreground=fg_color) # If used
                               # style.configure('TButton', background=bg_color, foreground=fg_color) # Buttons are tricky

                               # Listbox is tk, not ttk, needs option_add
                               # Use try/except as option_add might not be available in all contexts or for all styles
                               try:
                                  root.option_add('*Listbox.background', listbox_bg)
                                  root.option_add('*Listbox.foreground', listbox_fg)
                                  root.option_add('*Listbox.selectBackground', listbox_select_bg)
                                  root.option_add('*Listbox.selectForeground', listbox_select_fg)
                               except tk.TclError: print("Note: Failed to configure Tk Listbox colors via option_add.", file=sys.stderr)

                               # Combobox dropdown list might need configuring (may inherit from Listbox options)
                               # Explicit mapping might be needed if listbox options don't propagate
                               try:
                                    style.map('TCombobox', fieldbackground=[('readonly', entry_bg), ('!readonly', entry_bg)],
                                                             foreground=[('readonly', entry_fg), ('!readonly', entry_fg)])
                                    style.map('TEntry', fieldbackground=[('!disabled', entry_bg)], foreground=[('!disabled', entry_fg)])
                               except tk.TclError: print("Note: Failed to configure TCombobox/TEntry colors via map.", file=sys.stderr)
                               # ScrolledText is tk, needs option_add
                               try:
                                   root.option_add('*ScrolledText.background', entry_bg)
                                   root.option_add('*ScrolledText.foreground', entry_fg)
                                   # Need to set insertbackground for cursor color in dark themes
                                   root.option_add('*ScrolledText.insertBackground', fg_color)
                                   # Selection colors
                                   root.option_add('*ScrolledText.selectBackground', listbox_select_bg)
                                   root.option_add('*ScrolledText.selectForeground', listbox_select_fg)
                               except tk.TclError: print("Note: Failed to configure Tk ScrolledText colors via option_add.", file=sys.stderr)


                          except tk.TclError as style_err:
                               print(f"Note: Could not fully configure basic dark theme styles after applying {theme}: {style_err}", file=sys.stderr)
                          except Exception as general_style_err:
                               print(f"Note: Unexpected error during dark theme styling after applying {theme}: {general_style_err}", file=sys.stderr)
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