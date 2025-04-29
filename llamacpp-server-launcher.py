#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLaMa.cpp HTTP‑Server Launcher (GUI)

• Keeps all explanatory labels / section headers
• Adds --n‑gpu‑layers, --tensor‑split, --flash‑attn controls
• Windows‑friendly persistence (falls back to %APPDATA%)
• Multi-directory model scanning and selection
• Finds server executable in common subdirs (select root llama.cpp dir)
• Model listbox instead of dropdown, with persistent height.
• Detects CUDA devices and displays names and VRAM.
• Parses selected GGUF for layer count and other info (Architecture, File Size).
• GPU Layer slider is informed by detected layer count, or disabled if not found.
• GPU Layer entry is usable even if layer count is not found.
"""

import json
import os
import re
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox
from threading import Thread # For background scanning AND analysis
import traceback # For detailed error printing
import ctypes # For Windows RAM info (fallback)
import shlex # For more robust command quoting

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
    print(f"Warning: PyTorch import failed: {e}")


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
    print(f"Warning: llama-cpp-python import failed: {e}")


try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

# --- Dependency Check ---
MISSING_DEPS = []
# Only note torch if llama_cpp is potentially installed (user likely wants GPU features)
if LLAMA_CPP_PYTHON_AVAILABLE and not TORCH_AVAILABLE:
     MISSING_DEPS.append("PyTorch (with CUDA support likely needed for GPU detection/offload)")
elif not LLAMA_CPP_PYTHON_AVAILABLE:
    MISSING_DEPS.append("llama-cpp-python (for GGUF analysis)")
# psutil is only needed for non-Windows RAM info if ctypes fails
if sys.platform != "win32" and not PSUTIL_AVAILABLE:
    MISSING_DEPS.append("psutil (for RAM detection on non-Windows)")


# ═════════════════════════════════════════════════════════════════════
#  Helper Functions (Adapted from provided code)
# ═════════════════════════════════════════════════════════════════════

def get_gpu_info_static():
    """Get GPU information using PyTorch (static method)."""
    if not torch or not TORCH_AVAILABLE:
        # Distinguish between no torch and no CUDA in torch
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
                           return {"error": f"Windows RAM checks failed (ctypes: {e_win}, psutil: {e_psutil})"}

                 else:
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
                 return {"error": f"psutil RAM check failed: {e_psutil}"}

        else:
             return {"error": "psutil not installed, cannot get RAM info on this platform."}
    except Exception as e:
        return {"error": f"Failed to get RAM info: {str(e)}"}


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
             llm_meta = Llama(model_path=str(model_path), n_ctx=32, n_threads=1, n_batch=32,
                              verbose=False, n_gpu_layers=0, logits_all=True, main_gpu=0)
        except Exception as load_exc:
             analysis_result["error"] = f"Failed to load model for metadata analysis: {load_exc}"
             print(f"ERROR: Failed to load model '{model_path.name}' for analysis: {load_exc}")
             traceback.print_exc()
             return analysis_result # Exit early if basic load fails

        # --- Extract Metadata ---
        # Attempt 1: Check common metadata keys and attributes
        if hasattr(llm_meta, 'metadata') and isinstance(llm_meta.metadata, dict) and llm_meta.metadata:
            analysis_result["metadata"] = llm_meta.metadata
            analysis_result["n_layers"] = llm_meta.metadata.get('llama.block_count')
            if analysis_result["n_layers"] is None:
                 analysis_result["n_layers"] = llm_meta.metadata.get('general.architecture.block_count')
            # Try specific architecture keys if generic ones fail
            if analysis_result["n_layers"] is None:
                 analysis_result["n_layers"] = llm_meta.metadata.get('qwen2.block_count') # Example for Qwen2

            analysis_result["architecture"] = llm_meta.metadata.get('general.architecture', 'unknown')
            if analysis_result["architecture"] == 'unknown':
                 analysis_result["architecture"] = llm_meta.metadata.get('qwen2.architecture', 'unknown') # Example

        # Attempt 2: Check direct attributes if metadata didn't yield layers
        if analysis_result["n_layers"] is None:
             if hasattr(llm_meta, 'n_layer'): # Common recent name
                  analysis_result["n_layers"] = getattr(llm_meta, 'n_layer', None)
             if analysis_result["n_layers"] is None and hasattr(llm_meta, 'n_layers'): # Older name
                  analysis_result["n_layers"] = getattr(llm_meta, 'n_layers', None)

        # Fallback architecture if still unknown
        if analysis_result["architecture"] == 'unknown' and hasattr(llm_meta, 'model_type'):
             analysis_result["architecture"] = getattr(llm_meta, 'model_type', 'unknown')


        # Clean up the temporary Llama object
        if llm_meta:
             try:
                  del llm_meta
             except Exception as clean_exc:
                  print(f"Warning: Failed to delete llama_cpp.Llama instance: {clean_exc}")

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
        print(f"ERROR: Failed during GGUF metadata processing for '{model_path.name}': {e}")
        traceback.print_exc()
        analysis_result["error"] = f"Unexpected error during analysis: {e}"
        analysis_result["n_layers"] = None # Ensure layers is None on error
        return analysis_result


# ═════════════════════════════════════════════════════════════════════
#  Main class
# ═════════════════════════════════════════════════════════════════════
class LlamaCppLauncher:
    """Tk‑based launcher for llama.cpp HTTP server."""

    # ──────────────────────────────────────────────────────────────────
    #  construction / persistence
    # ──────────────────────────────────────────────────────────────────
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("LLaMa.cpp Server Launcher")
        self.root.geometry("900x850")
        self.root.minsize(800, 750)

        # --- Check Dependencies ---
        if MISSING_DEPS:
            missing_str = "\n - ".join(MISSING_DEPS)
            messagebox.showwarning("Missing Dependencies",
                                   f"The following Python libraries are recommended for full functionality but were not found:\n\n - {missing_str}\n\nPlease install them (e.g., 'pip install llama-cpp-python torch psutil') if you need GGUF analysis or GPU detection/offload.")


        # Internal state for resizing
        self._resize_start_y = 0
        self._resize_start_height = 0

        # --- System Info ---
        self.gpu_info = {"available": False, "device_count": 0, "devices": []}
        self.ram_info = {}
        self._fetch_system_info()

        # ------------------------------------------------ persistence --
        self.config_path = self._get_config_path()
        self.saved_configs = {}
        self.app_settings = {
            "last_llama_cpp_dir": "",
            "last_venv_dir":      "",
            "last_model_path":    "",
            "model_dirs":         [],
            "model_list_height":  8,
        }

        # ------------------------------------------------ Tk variables --
        self.llama_cpp_dir   = tk.StringVar()
        self.venv_dir        = tk.StringVar()
        self.model_path      = tk.StringVar()
        self.model_dirs_listvar = tk.StringVar()
        self.scan_status_var = tk.StringVar(value="Scan models to populate list.")

        self.cache_type_k    = tk.StringVar(value="f16")
        self.threads         = tk.StringVar(value="4")

        self.n_gpu_layers    = tk.StringVar(value="0")
        self.n_gpu_layers_int = tk.IntVar(value=0)
        self.max_gpu_layers  = tk.IntVar(value=0)
        self.gpu_layers_status_var = tk.StringVar(value="Select model to see layer info")

        self.no_mmap         = tk.BooleanVar(value=False)
        self.no_cnv          = tk.BooleanVar(value=False)
        self.prio            = tk.StringVar(value="0")
        self.temperature     = tk.StringVar(value="0.8")
        self.min_p           = tk.StringVar(value="0.05")
        self.ctx_size        = tk.IntVar(value=2048)
        self.seed            = tk.StringVar(value="-1")
        self.flash_attn      = tk.BooleanVar(value=False)
        self.tensor_split    = tk.StringVar(value="")
        self.main_gpu        = tk.StringVar(value="0")
        self.gpu_count_var   = tk.IntVar(value=self.gpu_info["device_count"])
        self.gpu_vars = []

        self.mlock           = tk.BooleanVar(value=False)
        self.no_kv_offload   = tk.BooleanVar(value=False)
        self.host            = tk.StringVar(value="127.0.0.1")
        self.port            = tk.StringVar(value="8080")
        self.config_name     = tk.StringVar(value="default_config")

        self.model_architecture_var = tk.StringVar(value="N/A")
        self.model_filesize_var = tk.StringVar(value="N/A")

        # Internal state
        self.model_dirs = []
        self.found_models = {}
        self.current_model_analysis = {}
        self.analysis_thread = None

        # load previous settings
        self._load_saved_configs()
        self.llama_cpp_dir.set(self.app_settings.get("last_llama_cpp_dir", ""))
        self.venv_dir.set(self.app_settings.get("last_venv_dir", ""))
        self.model_dirs = [Path(d) for d in self.app_settings.get("model_dirs", []) if d]

        # build GUI
        self._create_widgets()

        # --- Setup validation for GPU Layers Entry AFTER widgets are created ---
        vcmd = (self.root.register(self._validate_gpu_layers_entry), '%P')
        if hasattr(self, 'n_gpu_layers_entry') and self.n_gpu_layers_entry.winfo_exists():
             self.n_gpu_layers_entry.config(validate='key', validatecommand=vcmd)
             self.n_gpu_layers_entry.bind("<FocusOut>", self._sync_gpu_layers_from_entry)
             self.n_gpu_layers_entry.bind("<Return>", self._sync_gpu_layers_from_entry)

        # Populate model directories listbox
        self._update_model_dirs_listbox()

        # Perform initial scan (in background) if dirs exist
        if self.model_dirs:
            self.scan_status_var.set("Scanning on startup...")
            scan_thread = Thread(target=self._scan_model_dirs, daemon=True)
            scan_thread.start()
        else:
             self.scan_status_var.set("Add directories and scan for models.")


    # ═════════════════════════════════════════════════════════════════
    #  System Info Fetching
    # ═════════════════════════════════════════════════════════════════
    def _fetch_system_info(self):
        """Fetches GPU and RAM info."""
        print("Fetching system info...")
        self.gpu_info = get_gpu_info_static()
        self.ram_info = get_ram_info_static()
        print(f"GPU Info: {self.gpu_info}")
        if hasattr(self, 'gpu_count_var'):
             self.gpu_count_var.set(self.gpu_info["device_count"])
        if not self.gpu_info["available"] and "message" in self.gpu_info:
             print(f"GPU Detection Info: {self.gpu_info['message']}")
        if "error" in self.ram_info:
             print(f"RAM Detection Error: {self.ram_info['error']}")

    # ═════════════════════════════════════════════════════════════════
    #  persistence helpers
    # ═════════════════════════════════════════════════════════════════
    def _get_config_path(self):
        local_path = Path("llama_cpp_configs.json")
        try:
            with local_path.open("a") as f: pass
            if local_path.exists() and local_path.stat().st_size == 0:
                 try: local_path.unlink()
                 except OSError: pass
            return local_path
        except (OSError, PermissionError, IOError):
            if sys.platform == "win32":
                appdata = os.getenv("APPDATA")
                fallback_dir = Path(appdata) / "LlamaCppLauncher" if appdata else Path.home() / ".llama_cpp_launcher"
            else:
                 fallback_dir = Path.home() / ".config" / "llama_cpp_launcher"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            return fallback_dir / "llama_cpp_configs.json"

    def _load_saved_configs(self):
        if self.config_path.exists():
            try:
                data = json.loads(self.config_path.read_text(encoding="utf-8"))
                self.saved_configs = data.get("configs", {})
                loaded_app_settings = data.get("app_settings", {})
                self.app_settings.update(loaded_app_settings)
                if not isinstance(self.app_settings.get("model_list_height"), int):
                    self.app_settings["model_list_height"] = 8
            except Exception as exc:
                messagebox.showerror("Config Load Error", f"Could not load config from:\n{self.config_path}\n\nError: {exc}\n\nUsing default settings.")
                self.app_settings = {
                    "last_llama_cpp_dir": "", "last_venv_dir": "", "last_model_path": "",
                    "model_dirs": [], "model_list_height": 8
                }
                self.saved_configs = {}

    def _save_configs(self):
        self.app_settings["model_dirs"] = [str(p) for p in self.model_dirs]
        self.app_settings["last_model_path"] = self.model_path.get()

        payload = {
            "configs":      self.saved_configs,
            "app_settings": self.app_settings,
        }
        try:
            self.config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            original_path = self.config_path
            self.config_path = self._get_config_path()
            if self.config_path != original_path:
                 try:
                    self.config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    messagebox.showwarning("Config Save Info", f"Could not write to original location.\nSettings stored in:\n{self.config_path}")
                 except Exception as final_exc:
                     messagebox.showerror("Config Save Error", f"Failed to save settings to:\n{self.config_path}\n\nError: {final_exc}")
            else:
                 messagebox.showerror("Config Save Error", f"Failed to save settings to:\n{self.config_path}\n\nError: {exc}")

    # ═════════════════════════════════════════════════════════════════
    #  UI builders
    # ═════════════════════════════════════════════════════════════════
    def _create_widgets(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=10, pady=10)

        main_frame = ttk.Frame(nb); adv_frame = ttk.Frame(nb); cfg_frame = ttk.Frame(nb)
        nb.add(main_frame, text="Main Settings")
        nb.add(adv_frame,  text="Advanced Settings")
        nb.add(cfg_frame,  text="Configurations")

        self._setup_main_tab(main_frame)
        self._setup_advanced_tab(adv_frame)
        self._setup_config_tab(cfg_frame)

        bar = ttk.Frame(self.root); bar.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(bar, text="Launch Server",   command=self.launch_server).pack(side="left",  padx=5)
        ttk.Button(bar, text="Save PS1 Script", command=self.save_ps1_script).pack(side="left", padx=5)
        ttk.Button(bar, text="Exit",            command=self.on_exit).pack(side="right", padx=5)

    # ░░░░░ MAIN TAB ░░░░░
    def _setup_main_tab(self, parent):
        canvas = tk.Canvas(parent); vs = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner  = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(yscrollcommand=vs.set,
                                                             scrollregion=canvas.bbox("all")))
        canvas_window = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))
        canvas.pack(side="left", fill="both", expand=True); vs.pack(side="right", fill="y")

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
        ttk.Button(vf, text="Browse…", command=lambda: self._browse_dir(self.venv_dir)).pack(side="left", padx=2)
        ttk.Button(vf, text="Clear",   command=self._clear_venv).pack(side="left", padx=2)
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
        dir_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.model_dirs_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        inner.rowconfigure(r, weight=0)
        dir_btn_frame = ttk.Frame(inner)
        dir_btn_frame.grid(column=2, row=r, sticky="ew", padx=5, pady=3, rowspan=2)
        ttk.Button(dir_btn_frame, text="Add Dir…", width=10, command=self._add_model_dir)\
           .pack(side=tk.TOP, pady=2, fill=tk.X)
        ttk.Button(dir_btn_frame, text="Remove Dir", width=10, command=self._remove_model_dir)\
           .pack(side=tk.TOP, pady=2, fill=tk.X)
        r += 2

        ttk.Label(inner, text="Select Model:")\
            .grid(column=0, row=r, sticky="nw", padx=10, pady=(10, 3))
        model_select_frame = ttk.Frame(inner)
        model_select_frame.grid(column=1, row=r, columnspan=2, sticky="nsew", padx=5, pady=(10,0))
        inner.rowconfigure(r, weight=1)
        inner.columnconfigure(1, weight=1)
        model_list_sb = ttk.Scrollbar(model_select_frame, orient=tk.VERTICAL)
        self.model_listbox = tk.Listbox(model_select_frame,
                                        height=self.app_settings.get("model_list_height", 8),
                                        width=48,
                                        yscrollcommand=model_list_sb.set,
                                        exportselection=False,
                                        state=tk.DISABLED)
        model_list_sb.config(command=self.model_listbox.yview)
        model_list_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.model_listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.resize_grip = ttk.Separator(model_select_frame, orient=tk.HORIZONTAL, cursor="sb_v_double_arrow")
        self.resize_grip.pack(side=tk.BOTTOM, fill=tk.X, pady=(2,0))
        self.resize_grip.bind("<ButtonPress-1>", self._start_resize)
        self.resize_grip.bind("<B1-Motion>", self._do_resize)
        self.resize_grip.bind("<ButtonRelease-1>", self._end_resize)
        self.model_listbox.bind("<<ListboxSelect>>", self._on_model_selected)
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
        ttk.Label(inner, text="Threads:")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Entry(inner, textvariable=self.threads, width=10)\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, text="Number of CPU threads to use (-t)", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r, columnspan=2, sticky="w", padx=5, pady=3); r += 1
        ttk.Label(inner, text="Context Size:")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ctx_f = ttk.Frame(inner); ctx_f.grid(column=1, row=r, columnspan=3, sticky="ew", padx=5, pady=3)
        self.ctx_label = ttk.Label(ctx_f, text=f"{self.ctx_size.get():,}", width=9, anchor='e')
        self.ctx_entry = ttk.Entry(ctx_f, width=9)
        self.ctx_entry.insert(0, str(self.ctx_size.get()))
        ctx_slider = ttk.Scale(ctx_f, from_=1024, to=1000000, orient="horizontal",
                               variable=self.ctx_size, command=self._update_ctx_label_from_slider)
        ctx_slider.grid(column=0, row=0, sticky="ew", padx=(0, 5))
        self.ctx_label.grid(column=1, row=0, padx=5)
        self.ctx_entry.grid(column=2, row=0, padx=5)
        ttk.Button(ctx_f, text="Set", command=self._override_ctx_size, width=4).grid(column=3, row=0, padx=(0, 5))
        ctx_slider.set(self.ctx_size.get())
        ctx_f.columnconfigure(0, weight=3)
        ttk.Label(inner, text="Prompt context size in tokens (-c)", font=("TkSmallCaptionFont"))\
            .grid(column=1, row=r+1, columnspan=3, sticky="w", padx=5); r += 2
        ttk.Label(inner, text="Temperature:")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Entry(inner, textvariable=self.temperature, width=10)\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, text="Controls randomness (llama.cpp default: 0.8)", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r, columnspan=2, sticky="w", padx=5, pady=3); r += 1
        ttk.Label(inner, text="Min P:")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Entry(inner, textvariable=self.min_p, width=10)\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, text="Minimum probability sampling (llama.cpp default: 0.05)", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r, columnspan=2, sticky="w", padx=5, pady=3); r += 1
        ttk.Label(inner, text="Seed:")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Entry(inner, textvariable=self.seed, width=10)\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, text="RNG seed (-1 for random)", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r, columnspan=2, sticky="w", padx=5, pady=3); r += 1

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

        inner.columnconfigure(1, weight=1)

    # ░░░░░ ADVANCED TAB ░░░░░
    def _setup_advanced_tab(self, parent):
        canvas = tk.Canvas(parent); vs = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner  = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(yscrollcommand=vs.set,
                                                             scrollregion=canvas.bbox("all")))
        canvas_window = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))
        canvas.pack(side="left", fill="both", expand=True); vs.pack(side="right", fill="y")

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

        # --- GPU Settings ---
        ttk.Label(inner, text="GPU Settings", font=("TkDefaultFont", 12, "bold"))\
            .grid(column=0, row=r, sticky="w", padx=10, pady=(20,5)); r += 1
        ttk.Separator(inner, orient="horizontal")\
            .grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=5); r += 1

        # CUDA Devices Info & Checkboxes
        gpu_avail_text = "Available" if self.gpu_info['available'] else "Not available"
        gpu_msg = f" ({self.gpu_info['message']})" if not self.gpu_info['available'] and self.gpu_info.get('message') else ""
        gpu_detected_text = f"CUDA Devices ({gpu_avail_text}{gpu_msg}):"

        ttk.Label(inner, text=gpu_detected_text)\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        self.gpu_checkbox_frame = ttk.Frame(inner)
        self.gpu_checkbox_frame.grid(column=1, row=r, columnspan=3, sticky="w", padx=5, pady=3)
        self._update_gpu_checkboxes()
        r += 1

        # Display VRAM for each GPU
        if self.gpu_info['available'] and self.gpu_info['device_count'] > 0:
            vram_info_frame = ttk.Frame(inner)
            vram_info_frame.grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=(0, 10))
            # FIX: Use tuple/list for font style
            ttk.Label(vram_info_frame, text="VRAM (Total GB):", font=("TkSmallCaptionFont", 8, ("bold",)))\
                .pack(side="left", padx=(0, 5))
            for gpu in self.gpu_info['devices']:
                 ttk.Label(vram_info_frame, text=f"GPU {gpu['id']}: {gpu['total_memory_gb']:.2f} GB", font="TkSmallCaptionFont")\
                    .pack(side="left", padx=5)
            r += 1

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

        gpu_layers_frame.columnconfigure(1, weight=1)

        ttk.Label(inner, text="Layers to offload (0=CPU, -1=All). Slider range updates with model analysis.", font=("TkSmallCaptionFont"))\
            .grid(column=1, row=r+1, columnspan=3, sticky="w", padx=5); r += 2


        # --- Tensor Split ---
        ttk.Label(inner, text="Tensor Split (--tensor-split):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Entry(inner, textvariable=self.tensor_split, width=25)\
            .grid(column=1, row=r, columnspan=2, sticky="w", padx=5, pady=3)
        ttk.Label(inner, text="e.g., '3,1' splits layers 75%/25% across 2 GPUs", font=("TkSmallCaptionFont"))\
            .grid(column=1, row=r+1, columnspan=3, sticky="w", padx=5); r += 2

        # --- Flash Attention ---
        ttk.Label(inner, text="Flash Attention (--flash-attn):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Checkbutton(inner, variable=self.flash_attn)\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, text="Use Flash Attention kernel (CUDA only, requires specific build)", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r, columnspan=2, sticky="w", padx=5, pady=3); r += 1

        # --- Memory settings ---
        ttk.Label(inner, text="Memory & Cache", font=("TkDefaultFont", 12, "bold"))\
            .grid(column=0, row=r, sticky="w", padx=10, pady=(20,5)); r += 1
        ttk.Separator(inner, orient="horizontal")\
            .grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=5); r += 1
        ttk.Label(inner, text="KV Cache Type (--cache-type-k):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Combobox(inner, textvariable=self.cache_type_k, width=10, state="readonly",
                     values=("f16","f32","q8_0","q4_0","q4_1","q5_0","q5_1"))\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, text="Quantization for KV cache (f16 is default)", font=("TkSmallCaptionFont"))\
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
        ttk.Label(inner, text="Disable Conv Batching (--no-cnv):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Checkbutton(inner, variable=self.no_cnv)\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, text="Disable convolutional batching for prompt processing", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r, columnspan=2, sticky="w", padx=5, pady=3); r += 1

        # --- Performance ---
        ttk.Label(inner, text="Performance", font=("TkDefaultFont", 12, "bold"))\
            .grid(column=0, row=r, sticky="w", padx=10, pady=(20,5)); r += 1
        ttk.Separator(inner, orient="horizontal")\
            .grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=5); r += 1
        ttk.Label(inner, text="Scheduling Priority (--prio):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Combobox(inner, textvariable=self.prio, width=10, values=("0","1","2","3"), state="readonly")\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, text="0=Normal, 1=Medium, 2=High, 3=Realtime (OS dependent)", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r, columnspan=2, sticky="w", padx=5, pady=3)

        inner.columnconfigure(2, weight=1)


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
        self._resize_start_height = self.model_listbox.cget("height")
    def _do_resize(self, event):
        delta_y = event.y_root - self._resize_start_y
        pixels_per_line = 15
        delta_lines = round(delta_y / pixels_per_line)
        new_height = max(3, min(30, self._resize_start_height + delta_lines))
        if new_height != self.model_listbox.cget("height"):
            self.model_listbox.config(height=new_height)
    def _end_resize(self, event):
        final_height = self.model_listbox.cget("height")
        if final_height != self.app_settings.get("model_list_height"):
            self.app_settings["model_list_height"] = final_height
            self._save_configs()

    # ═════════════════════════════════════════════════════════════════
    #  Model Directory and Scanning Logic
    # ═════════════════════════════════════════════════════════════════
    def _add_model_dir(self):
        initial_dir = self.model_dirs[-1] if self.model_dirs else str(Path.home())
        directory = filedialog.askdirectory(title="Select Model Directory", initialdir=initial_dir)
        if directory:
            p = Path(directory)
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
        del self.model_dirs[index]
        self._update_model_dirs_listbox()
        self._save_configs()
        self._trigger_scan()

    def _update_model_dirs_listbox(self):
        current_selection = self.model_dirs_listbox.curselection()
        self.model_dirs_listbox.delete(0, tk.END)
        for p in self.model_dirs:
            self.model_dirs_listbox.insert(tk.END, str(p))
        if current_selection:
            new_index = min(current_selection[0], self.model_dirs_listbox.size() - 1)
            if new_index >= 0:
                self.model_dirs_listbox.selection_set(new_index)
                self.model_dirs_listbox.activate(new_index)
                self.model_dirs_listbox.see(new_index)


    def _trigger_scan(self):
        """Initiates model scanning in a background thread."""
        self.scan_status_var.set("Scanning...")
        self.model_listbox.config(state=tk.NORMAL)
        self.model_listbox.delete(0, tk.END)
        self.model_path.set("")
        self._reset_gpu_layer_controls()
        self._reset_model_info_display()
        scan_thread = Thread(target=self._scan_model_dirs, daemon=True)
        scan_thread.start()

    def _scan_model_dirs(self):
        """Scans configured directories for GGUF models (runs in background thread)."""
        print("DEBUG: _scan_model_dirs thread started")
        found = {} # {display_name: full_path_obj}
        multipart_pattern = re.compile(r"^(.*?)(?:-\d{5}-of-\d{5}|-F\d+)\.gguf$", re.IGNORECASE)
        first_part_pattern = re.compile(r"^(.*?)-(?:00001-of-\d{5}|F1)\.gguf$", re.IGNORECASE)
        processed_multipart_bases = set()

        for model_dir in self.model_dirs:
            if not model_dir.is_dir(): continue
            print(f"DEBUG: Scanning directory: {model_dir}")
            try:
                for gguf_path in model_dir.rglob('*.gguf'):
                    if not gguf_path.is_file(): continue
                    filename = gguf_path.name
                    first_part_match = first_part_pattern.match(filename)
                    if first_part_match:
                        base_name = first_part_match.group(1)
                        if base_name not in processed_multipart_bases:
                            found[base_name] = gguf_path
                            processed_multipart_bases.add(base_name)
                        continue

                    multi_match = multipart_pattern.match(filename)
                    if multi_match:
                        base_name = multi_match.group(1)
                        processed_multipart_bases.add(base_name)
                        continue

                    if filename.lower().endswith(".gguf") and \
                       "mmproj" not in filename.lower() and \
                       gguf_path.stem not in processed_multipart_bases:
                         display_name = gguf_path.stem
                         if display_name not in found:
                            found[display_name] = gguf_path

            except Exception as e:
                print(f"ERROR: Error scanning directory {model_dir}: {e}")
                traceback.print_exc()

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

        if not model_names:
            self.scan_status_var.set("No GGUF models found in specified directories.")
            self.model_path.set("")
            self._reset_gpu_layer_controls()
            self._reset_model_info_display()
        else:
            self.scan_status_var.set(f"Scan complete. Found {len(model_names)} models.")

            last_path_str = self.app_settings.get("last_model_path")
            selected_idx = -1
            if last_path_str:
                 try:
                     last_path_obj = Path(last_path_str)
                     found_display_name = None
                     for display_name, full_path in self.found_models.items():
                         if full_path == last_path_obj:
                             found_display_name = display_name
                             break
                     if found_display_name:
                         listbox_items = self.model_listbox.get(0, tk.END)
                         selected_idx = listbox_items.index(found_display_name)
                 except (ValueError, OSError): pass

            if selected_idx == -1 and model_names:
                selected_idx = 0

            if selected_idx != -1:
                self.root.after(50, self._select_model_in_listbox, selected_idx)
            else:
                 self._reset_gpu_layer_controls()
                 self._reset_model_info_display()


    def _select_model_in_listbox(self, index):
         """Selects a specific index in the model listbox."""
         try:
            if 0 <= index < self.model_listbox.size():
                 self.model_listbox.selection_clear(0, tk.END)
                 self.model_listbox.selection_set(index)
                 self.model_listbox.activate(index)
                 self.model_listbox.see(index)
                 self._on_model_selected()
            else:
                print(f"WARN: Attempted to select invalid index {index} in model listbox.")
                self._reset_gpu_layer_controls()
                self._reset_model_info_display()
         except Exception as e:
             print(f"ERROR during _select_model_in_listbox: {e}")
             traceback.print_exc()
             self._reset_gpu_layer_controls()
             self._reset_model_info_display()


    def _on_model_selected(self, event=None):
        """Callback when a model is selected. Triggers GGUF analysis."""
        selection = self.model_listbox.curselection()
        if not selection:
            self.model_path.set("")
            self.app_settings["last_model_path"] = ""
            self._reset_gpu_layer_controls()
            self._reset_model_info_display()
            return

        index = selection[0]
        selected_name = self.model_listbox.get(index)

        if selected_name in self.found_models:
            full_path = self.found_models[selected_name]
            full_path_str = str(full_path)
            self.model_path.set(full_path_str)
            self.app_settings["last_model_path"] = full_path_str

            if LLAMA_CPP_PYTHON_AVAILABLE:
                 self.gpu_layers_status_var.set("Analyzing model...")
                 self.gpu_layers_slider.config(state=tk.DISABLED)
                 # Keep entry state as is for now, it's validated anyway
                 # self.n_gpu_layers_entry.config(state=tk.DISABLED)
                 self._reset_model_info_display()
                 # Start analysis thread
                 self.analysis_thread = Thread(target=self._run_gguf_analysis, args=(full_path_str,), daemon=True)
                 self.analysis_thread.start()
            else:
                 self.gpu_layers_status_var.set("Analysis requires llama-cpp-python")
                 self._reset_gpu_layer_controls(keep_entry_enabled=True) # Keep entry enabled if lib missing
                 self._reset_model_info_display()
                 self.model_architecture_var.set("Analysis Unavailable")
                 self.model_filesize_var.set("Analysis Unavailable")

        else:
            print(f"WARNING: Selected model '{selected_name}' not found in self.found_models.")
            self.model_path.set("")
            self.app_settings["last_model_path"] = ""
            self._reset_gpu_layer_controls()
            self._reset_model_info_display()


    def _run_gguf_analysis(self, model_path_str):
        """Worker function for background GGUF analysis."""
        print(f"Analyzing GGUF in background: {model_path_str}")
        analysis_result = analyze_gguf_model_static(model_path_str)
        self.root.after(0, self._update_ui_after_analysis, analysis_result)

    def _update_ui_after_analysis(self, analysis_result):
        """Updates controls based on GGUF analysis results (runs in main thread)."""
        self.current_model_analysis = analysis_result
        error = analysis_result.get("error")
        n_layers = analysis_result.get("n_layers")
        message = analysis_result.get("message")

        # --- Update Model Info Display ---
        arch = analysis_result.get("architecture", "unknown")
        file_size_gb = analysis_result.get("file_size_gb")
        self.model_architecture_var.set(arch.capitalize() if arch and arch != "unknown" else "Unknown")
        self.model_filesize_var.set(f"{file_size_gb:.2f} GB" if file_size_gb is not None and file_size_gb > 0 else "Unknown Size")


        # --- Update GPU Layer Controls ---
        # Always enable the entry after analysis finishes, unless there was a critical load error
        if not error or "Failed to load model" not in error:
             self.n_gpu_layers_entry.config(state=tk.NORMAL)


        if error:
            self.gpu_layers_status_var.set(f"Analysis Error: {error}")
            self._reset_gpu_layer_controls(keep_entry_enabled=True) # Reset slider, keep entry enabled
            self.max_gpu_layers.set(0) # Ensure max is 0 on analysis error


        elif n_layers is not None and n_layers > 0:
            # Analysis succeeded, layers found
            self.max_gpu_layers.set(n_layers)
            self.gpu_layers_status_var.set(f"Max Layers: {n_layers}")
            self.gpu_layers_slider.config(to=n_layers, state=tk.NORMAL)

            # Sync the controls based on the *current* value in the n_gpu_layers StringVar.
            current_value_str = self.n_gpu_layers.get()
            try:
                 value = int(current_value_str)
                 self._set_gpu_layers(value) # Use helper to clamp & sync based on new max
            except ValueError:
                 # If current value is invalid (shouldn't happen with validation), default to 0
                 self._set_gpu_layers(0)

        else: # Analysis succeeded, but layer count was 0, negative, not int, or None
            status_msg = message if message else "Layer count not found or invalid"
            self.gpu_layers_status_var.set(status_msg)
            self._reset_gpu_layer_controls(keep_entry_enabled=True)
             # max_gpu_layers is already 0 from _reset_gpu_layer_controls


    def _reset_gpu_layer_controls(self, keep_entry_enabled=False):
         """Resets GPU layer slider/entry state."""
         # Reset max_gpu_layers first
         self.max_gpu_layers.set(0)
         # Reset n_gpu_layers and n_gpu_layers_int to 0
         self._set_gpu_layers(0) # This sets both to 0 and syncs slider/entry visual

         self.gpu_layers_status_var.set("Select model to see layer info")
         # Slider range and state
         if hasattr(self, 'gpu_layers_slider') and self.gpu_layers_slider.winfo_exists():
              self.gpu_layers_slider.config(to=0, state=tk.DISABLED)

         # Entry state
         if hasattr(self, 'n_gpu_layers_entry') and self.n_gpu_layers_entry.winfo_exists():
             if not keep_entry_enabled:
                 self.n_gpu_layers_entry.config(state=tk.DISABLED)
             else:
                 # Ensure entry is usable if explicitly requested (e.g., lib missing, layer count unknown)
                 self.n_gpu_layers_entry.config(state=tk.NORMAL)


    def _reset_model_info_display(self):
         """Resets model info labels."""
         self.model_architecture_var.set("N/A")
         self.model_filesize_var.set("N/A")


    # ═════════════════════════════════════════════════════════════════
    #  GPU Layer Slider/Entry Synchronization & Validation
    # ═════════════════════════════════════════════════════════════════

    def _set_gpu_layers(self, value):
        """Helper to set GPU layers and sync slider/entry."""
        max_layers = self.max_gpu_layers.get() # Get current max layer count

        # Clamp/Adjust the *intended* value based on max_layers
        if max_layers <= 0: # Max is 0 if analysis failed or not run
             # If max is 0, only 0 and -1 are meaningful inputs
             if value == -1:
                  int_val = 0 # Slider represents 0
                  str_val = "-1" # Entry represents -1
             elif value == 0:
                  int_val = 0
                  str_val = "0"
             else: # Any other number input is invalid when max is 0, default to 0
                  int_val = 0
                  str_val = "0"
        else: # Max layers is known and positive
            if value == -1: # -1 means offload all
                 int_val = max_layers # Slider goes to max
                 str_val = "-1" # Entry shows -1
            else: # Clamp value between 0 and max_layers
                clamped_value = max(0, min(value, max_layers))
                int_val = clamped_value
                str_val = str(clamped_value)

        # Update the Tk variables
        self.n_gpu_layers_int.set(int_val)
        # Only update the StringVar if the *current* string value is different,
        # to avoid validation loops if validation just set it to the same value.
        if self.n_gpu_layers.get() != str_val:
             self.n_gpu_layers.set(str_val)

        # The slider is automatically updated because it's linked to n_gpu_layers_int


    def _sync_gpu_layers_from_slider(self, value_str):
        """Callback when slider changes."""
        try:
            # Slider value is always an integer (converted from float string) between 0 and max
            value = int(float(value_str))
            max_layers = self.max_gpu_layers.get()

            # If slider is at max AND max > 0, represent as -1 in entry
            if max_layers > 0 and value == max_layers:
                self.n_gpu_layers.set("-1")
                # n_gpu_layers_int is already max because it's linked
            else:
                # For any other slider position, use the integer value
                self.n_gpu_layers.set(str(value))
                # n_gpu_layers_int is already correct because it's linked
        except ValueError:
            pass # Should not happen with a slider

    def _sync_gpu_layers_from_entry(self, event=None):
        """Callback when entry loses focus or Enter is pressed."""
        if not hasattr(self, 'n_gpu_layers_entry') or not self.n_gpu_layers_entry.winfo_exists():
            return
        current_str = self.n_gpu_layers.get()
        try:
            value = int(current_str)
            # Use helper to handle clamping/syncing based on max_layers
            # This will also update the StringVar if needed (e.g., if -1 maps to max)
            self._set_gpu_layers(value)
        except ValueError:
            # Invalid input (should be caught by validation, but this is a fallback)
            # Revert entry to the current valid representation based on the IntVar
            print(f"DEBUG: Invalid value '{current_str}' in GPU layers entry. Reverting.")
            # Determine the correct string representation from the current IntVar/max_layers
            current_int_value = self.n_gpu_layers_int.get()
            max_val = self.max_gpu_layers.get()
            if max_val > 0 and current_int_value == max_val:
                 # If IntVar is at max and max > 0, the string should be "-1"
                 self.n_gpu_layers.set("-1")
            else:
                # Otherwise, the string should be the integer value
                 self.n_gpu_layers.set(str(current_int_value))


    def _validate_gpu_layers_entry(self, proposed_value):
        """Validation function for the n_gpu_layers entry."""
        if not hasattr(self, 'max_gpu_layers'): return True # Allow during init

        pv = proposed_value.strip()

        if pv == "": return True # Allow empty while typing
        if pv == "-": return True # Allow starting negative (for -1)
        if pv == "-1": return True # Allow exactly -1

        try:
            value = int(pv)
            max_val = self.max_gpu_layers.get()

            if max_val <= 0: # Layer count is unknown/invalid (max_layers is 0)
                 # Only allow 0 if max is 0 (since -1 is handled explicitly above)
                 return value == 0
            else: # Max layers is known and positive
                 # Allow values between 0 and max_val (inclusive)
                 return 0 <= value <= max_val
        except ValueError:
            # Reject non-numeric input (other than "" and "-")
            return False


    # ═════════════════════════════════════════════════════════════════
    #  dynamic GPU checkboxes
    # ═════════════════════════════════════════════════════════════════
    def _update_gpu_checkboxes(self):
        """Updates GPU checkboxes based on detected devices."""
        for w in self.gpu_checkbox_frame.winfo_children(): w.destroy()
        self.gpu_vars.clear()

        count = self.gpu_info["device_count"]

        if count > 0:
            for i in range(count):
                v = tk.BooleanVar(value=False)
                gpu_name = "Unknown GPU"
                if i < len(self.gpu_info["devices"]):
                     gpu_name = self.gpu_info["devices"][i].get("name", gpu_name)

                label_text = f"GPU {i}"
                if gpu_name and gpu_name != "Unknown GPU":
                     label_text += f": {gpu_name}"

                cb = ttk.Checkbutton(self.gpu_checkbox_frame, text=label_text, variable=v)
                cb.pack(side="left", padx=3, pady=2)
                self.gpu_vars.append(v)
        else:
             ttk.Label(self.gpu_checkbox_frame, text="(No CUDA devices detected)").pack(side="left", padx=5, pady=3)


    # ═════════════════════════════════════════════════════════════════
    #  misc helpers
    # ═════════════════════════════════════════════════════════════════
    def _clear_venv(self):
        self.venv_dir.set("")
        self.app_settings["last_venv_dir"] = ""

    def _update_ctx_label_from_slider(self, value_str):
        try:
            value = int(float(value_str))
            rounded = max(1024, round(value / 1024) * 1024)
            self.ctx_size.set(rounded)
            self._sync_ctx_display(rounded)
        except ValueError:
            pass

    def _override_ctx_size(self):
        try:
            raw = int(self.ctx_entry.get())
            rounded = max(1024, round(raw/1024)*1024)
            self.ctx_size.set(rounded)
            self._sync_ctx_display(rounded)
        except ValueError:
            messagebox.showerror("Input Error", "Context size must be a valid number.")
            self._sync_ctx_display(self.ctx_size.get())

    def _sync_ctx_display(self, value):
        formatted_value = f"{value:,}"
        str_value = str(value)
        if hasattr(self, 'ctx_label') and self.ctx_label.winfo_exists():
            self.ctx_label.config(text=formatted_value)
        if hasattr(self, 'ctx_entry') and self.ctx_entry.winfo_exists():
            if self.ctx_entry.get() != str_value:
                self.ctx_entry.delete(0, tk.END)
                self.ctx_entry.insert(0, str_value)

    def _browse_dir(self, var_to_set):
        if var_to_set == self.llama_cpp_dir:
            setting_key = "last_llama_cpp_dir"
            title = "Select LLaMa.cpp Root Directory"
        elif var_to_set == self.venv_dir:
             setting_key = "last_venv_dir"
             title = "Select Virtual Environment Directory"
        else:
             setting_key = None
             title = "Select Directory"

        initial = self.app_settings.get(setting_key) if setting_key else str(Path.home())
        if initial and not Path(initial).is_dir(): initial = str(Path.home())

        d = filedialog.askdirectory(initialdir=initial, title=title)
        if d:
            var_to_set.set(d)
            if setting_key:
                self.app_settings[setting_key] = d

    def _find_server_executable(self, base_dir: Path) -> Path | None:
        exe_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
        simple_exe_name = exe_name.replace("llama-","")
        search_paths = [
            Path("."), Path("server"),
            Path("build"), Path("bin"), Path("build/bin"),
            Path("Release"), Path("Debug"),
            Path("build/Release"), Path("build/Debug"),
            Path("build/bin/Release"), Path("build/bin/Debug"),
            Path("bin/Release"), Path("bin/Debug"),
        ]
        nested_specific_paths = [
             Path("build/bin") / exe_name, Path("build/bin") / simple_exe_name,
             Path("build/bin/Release") / exe_name, Path("build/bin/Release") / simple_exe_name,
             Path("build/bin/Debug") / exe_name, Path("build/bin/Debug") / simple_exe_name,
             Path("Release") / exe_name, Path("Release") / simple_exe_name,
             Path("Debug") / exe_name, Path("Debug") / simple_exe_name,
        ]
        checked_paths = set()

        def check_path(p: Path):
             resolved_p = None
             try:
                resolved_p = p.resolve()
                if resolved_p in checked_paths: return None
                if resolved_p.is_file():
                     print(f"Found server executable at: {resolved_p}")
                     return resolved_p
                checked_paths.add(resolved_p)
             except Exception:
                 # Handle cases where path.resolve() might fail (e.g., invalid chars, non-existent drive)
                 if resolved_p: checked_paths.add(resolved_p)
                 pass # Ignore errors resolving potential paths
             return None

        for rel_dir in search_paths:
             potential_path = base_dir / rel_dir / exe_name
             found = check_path(potential_path)
             if found: return found

             potential_path_simple = base_dir / rel_dir / simple_exe_name
             found = check_path(potential_path_simple)
             if found: return found

        for nested_path in nested_specific_paths:
             potential_path = base_dir / nested_path
             found = check_path(potential_path)
             if found: return found

        found = check_path(base_dir / exe_name)
        if found: return found
        found = check_path(base_dir / simple_exe_name)
        if found: return found

        print(f"Server executable ('{exe_name}' or '{simple_exe_name}') not found in common locations under {base_dir}")
        return None


    # ═════════════════════════════════════════════════════════════════
    #  configuration (save / load / delete)
    # ═════════════════════════════════════════════════════════════════
    def _current_cfg(self):
        gpu_layers_to_save = self.n_gpu_layers.get().strip()
        return {
            "llama_cpp_dir": self.llama_cpp_dir.get(),
            "venv_dir":      self.venv_dir.get(),
            "model_path":    self.model_path.get(),
            "cache_type_k":  self.cache_type_k.get(),
            "threads":       self.threads.get(),
            "n_gpu_layers":  gpu_layers_to_save,
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
            "gpu_indices":   [i for i, v in enumerate(self.gpu_vars) if v.get()],
            "mlock":         self.mlock.get(),
            "no_kv_offload": self.no_kv_offload.get(),
            "host":          self.host.get(),
            "port":          self.port.get(),
        }

    def _save_configuration(self):
        name = self.config_name.get().strip()
        if not name:
            return messagebox.showerror("Error","Enter a configuration name.")
        if name in self.saved_configs:
             if not messagebox.askyesno("Overwrite", f"Configuration '{name}' already exists. Overwrite?"):
                 return
        self.saved_configs[name] = self._current_cfg()
        self._save_configs()
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

        self.llama_cpp_dir.set(cfg.get("llama_cpp_dir",""))
        self.venv_dir.set(cfg.get("venv_dir",""))

        loaded_model_path_str = cfg.get("model_path", "")
        self.model_path.set(loaded_model_path_str)
        self.model_listbox.selection_clear(0, tk.END)

        selected_idx = -1
        if loaded_model_path_str:
            try:
                loaded_model_path_obj = Path(loaded_model_path_str)
                found_display_name = None
                for display_name, full_path in self.found_models.items():
                    if full_path == loaded_model_path_obj:
                        found_display_name = display_name
                        break
                if found_display_name:
                    listbox_items = self.model_listbox.get(0, tk.END)
                    selected_idx = listbox_items.index(found_display_name)
            except (ValueError, OSError, IndexError):
                 pass

        if selected_idx != -1:
             self.model_listbox.selection_set(selected_idx)
             self.model_listbox.see(selected_idx)
             self.model_listbox.activate(selected_idx)
             # Schedule selection handler to run after UI updates
             self.root.after(10, self._on_model_selected)
        else:
             self.model_path.set("")
             self._reset_gpu_layer_controls(keep_entry_enabled=True) # Keep entry enabled
             self._reset_model_info_display()
             if loaded_model_path_str:
                  messagebox.showwarning("Model Not Found", f"The model from the config ('{Path(loaded_model_path_str).name if loaded_model_path_str else 'N/A'}') was not found in the current list.\nPlease select a model manually.")

        self.cache_type_k.set(cfg.get("cache_type_k","f16"))
        self.threads.set(cfg.get("threads","4"))

        loaded_gpu_layers_str = cfg.get("n_gpu_layers","0")
        self.n_gpu_layers.set(loaded_gpu_layers_str)
        try:
             val = int(loaded_gpu_layers_str)
             # Use _set_gpu_layers to set the int var and sync entry, clamping based on current max
             self._set_gpu_layers(val)
        except ValueError:
             self.n_gpu_layers.set("0")
             self._set_gpu_layers(0)


        self.no_mmap.set(cfg.get("no_mmap",False))
        self.no_cnv.set(cfg.get("no_cnv",False))
        self.prio.set(cfg.get("prio","0"))
        self.temperature.set(cfg.get("temperature","0.8"))
        self.min_p.set(cfg.get("min_p","0.05"))
        ctx = cfg.get("ctx_size", 2048)
        self.ctx_size.set(ctx)
        self._sync_ctx_display(ctx)
        self.seed.set(cfg.get("seed","-1"))
        self.flash_attn.set(cfg.get("flash_attn",False))
        self.tensor_split.set(cfg.get("tensor_split","").strip()) # Ensure strip on load too
        self.main_gpu.set(cfg.get("main_gpu","0"))

        gpu_indices = cfg.get("gpu_indices", [])
        self._update_gpu_checkboxes()
        for i, v in enumerate(self.gpu_vars):
             v.set(i in gpu_indices)

        self.mlock.set(cfg.get("mlock",False))
        self.no_kv_offload.set(cfg.get("no_kv_offload",False))
        self.host.set(cfg.get("host","127.0.0.1"))
        self.port.set(cfg.get("port","8080"))
        self.config_name.set(name)
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
        llama_dir_str = self.llama_cpp_dir.get()
        if not llama_dir_str:
            messagebox.showerror("Error", "LLaMa.cpp root directory is not set.")
            return None
        try:
            llama_base_dir = Path(llama_dir_str)
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
            simple_exe_name = exe_base_name.replace("llama-","")
            messagebox.showerror("Executable Not Found",
                                 f"Could not find '{exe_base_name}' or '{simple_exe_name}' within:\n{llama_base_dir}\n\n"
                                 f"Searched in common relative locations like:\n - {search_locs_str}\n\n"
                                 "Please ensure llama.cpp is built and the directory is correct.")
            return None

        cmd = [str(exe_path)]

        # --- Model Path ---
        model_full_path_str = self.model_path.get()
        if not model_full_path_str:
            messagebox.showerror("Error", "No model selected. Please scan and select a model from the list.")
            return None
        try:
            # Check if the file represented by the stored path exists
            if not Path(model_full_path_str).is_file():
                 # It's not a direct file. Is it a known multipart base name from the scan?
                 # We stored the path to the *first part* in self.found_models for multipart bases.
                 # Check if the currently selected display name corresponds to a known model
                 # and if the stored path for *that* name exists.
                 selected_name = ""
                 sel = self.model_listbox.curselection()
                 if sel: selected_name = self.model_listbox.get(sel[0])

                 if selected_name and selected_name in self.found_models:
                     # Check if the path associated with the selected name exists
                     actual_model_path_from_scan = self.found_models[selected_name]
                     if not actual_model_path_from_scan.is_file():
                          # Selected name is in list, but the file it points to from the scan is missing
                          error_msg = f"Selected model file not found based on scan:\n{actual_model_path_from_scan}"
                          error_msg += f"\n(Selected in GUI: {selected_name})"
                          error_msg += "\n\nPlease re-scan models or check file existence."
                          messagebox.showerror("Error", error_msg)
                          return None
                     else:
                         # Update model_full_path_str to the actual path from the scan
                         model_full_path_str = str(actual_model_path_from_scan)
                 else:
                     # The path in self.model_path is not a file and doesn't match a name in found_models
                     error_msg = f"Invalid or missing model path:\n{model_full_path_str}"
                     if selected_name: error_msg += f"\n(Selected in GUI: {selected_name})"
                     error_msg += "\n\nPlease re-scan models or select a valid model."
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


        cmd.extend(["-m", model_full_path_str])

        # --- Other Arguments ---
        self._add_arg(cmd, "--cache-type-k", self.cache_type_k.get(), "f16")
        self._add_arg(cmd, "--threads", self.threads.get(), "4")

        n_gpu_layers_val = self.n_gpu_layers.get().strip()
        self._add_arg(cmd, "--n-gpu-layers", n_gpu_layers_val, "0")

        self._add_arg(cmd, "--no-mmap", self.no_mmap.get())
        self._add_arg(cmd, "--mlock", self.mlock.get())
        self._add_arg(cmd, "--no-kv-offload", self.no_kv_offload.get())
        self._add_arg(cmd, "--no-cnv", self.no_cnv.get())
        self._add_arg(cmd, "--prio", self.prio.get(), "0")
        self._add_arg(cmd, "--temp", self.temperature.get(), "0.8")
        self._add_arg(cmd, "--min-p", self.min_p.get(), "0.05")
        cmd.extend(["--ctx-size", str(self.ctx_size.get())])
        self._add_arg(cmd, "--seed", self.seed.get(), "-1")
        self._add_arg(cmd, "--flash-attn", self.flash_attn.get())
        self._add_arg(cmd, "--tensor-split", self.tensor_split.get().strip(), "")
        self._add_arg(cmd, "--main-gpu", self.main_gpu.get(), "0")

        self._add_arg(cmd, "--host", self.host.get(), "127.0.0.1")
        self._add_arg(cmd, "--port", self.port.get(), "8080")

        print("\n--- Generated Command ---")
        print(subprocess.list2cmdline(cmd))
        print("-------------------------\n")

        return cmd

    def _add_arg(self, cmd_list, arg_name, value, default_value=None):
        """Adds argument to cmd list if not default (handles bools/strings)."""
        is_bool_var = isinstance(value, tk.BooleanVar)
        is_bool_py = isinstance(value, bool)

        if is_bool_var:
            if value.get(): cmd_list.append(arg_name)
            return

        if is_bool_py:
             if value: cmd_list.append(arg_name)
             return

        actual_value = str(value).strip()

        if not actual_value:
             return

        if default_value is None:
            cmd_list.extend([arg_name, actual_value])
        elif str(actual_value) != str(default_value):
             cmd_list.extend([arg_name, actual_value])


    # ═════════════════════════════════════════════════════════════════
    #  launch & script helpers
    # ═════════════════════════════════════════════════════════════════
    def launch_server(self):
        cmd_list = self._build_cmd()
        if not cmd_list: return
        venv_path_str = self.venv_dir.get()

        try:
            if sys.platform == "win32":
                 final_cmd_str = subprocess.list2cmdline(cmd_list)
                 if venv_path_str:
                    venv_path = Path(venv_path_str)
                    act_script = venv_path / "Scripts" / "activate.bat"
                    if not act_script.is_file():
                        messagebox.showerror("Error", f"Venv activation script (activate.bat) not found:\n{act_script}")
                        return
                    command = f'start "LLaMa.cpp Server" cmd /k ""{str(act_script)}" && {final_cmd_str}"'
                 else:
                    command = f'start "LLaMa.cpp Server" {final_cmd_str}'
                 subprocess.Popen(command, shell=True)

            else: # Linux/macOS
                quoted_cmd_parts = [shlex.quote(arg) for arg in cmd_list]
                server_command_str = " ".join(quoted_cmd_parts)

                launch_command = server_command_str
                if venv_path_str:
                    venv_path = Path(venv_path_str)
                    act_script = venv_path / "bin" / "activate"
                    if not act_script.is_file():
                        messagebox.showerror("Error", f"Venv activation script not found:\n{act_script}")
                        return
                    launch_command = f'echo "Activating venv: {venv_path}" && source "{str(act_script)}" && echo "Launching server..." && {server_command_str}'

                terminals = ['gnome-terminal', 'konsole', 'xfce4-terminal', 'xterm']
                term_cmd_pattern = {
                    'gnome-terminal': " -- bash -c {command}",
                    'konsole':        " -e bash -c {command}",
                    'xfce4-terminal': " -e bash -c {command}",
                    'xterm':          " -e bash -c '{command} ; read -p \"Press Enter to close...\"'"
                }
                launched = False
                import shutil
                for term in terminals:
                    if shutil.which(term):
                         quoted_full_command = shlex.quote(launch_command)
                         term_cmd_str = f"{term}{term_cmd_pattern[term].format(command=quoted_full_command)}"
                         print(f"DEBUG: Using terminal command: {term_cmd_str}")
                         try:
                            subprocess.Popen(term_cmd_str, shell=True)
                            launched = True
                            break
                         except Exception as term_err:
                             print(f"DEBUG: Failed to launch with {term}: {term_err}")
                if not launched:
                     messagebox.showerror("Launch Error", "Could not find a supported terminal (gnome-terminal, konsole, xfce4-terminal, xterm) to launch the server script.")

        except Exception as exc:
            messagebox.showerror("Launch Error", f"Failed to launch server process:\n{exc}")
            traceback.print_exc()

    def save_ps1_script(self):
        cmd_list = self._build_cmd()
        if not cmd_list: return

        selected_model_name = ""
        selection = self.model_listbox.curselection()
        if selection:
             selected_model_name = self.model_listbox.get(selection[0])

        default_name = "launch_llama_server.ps1"
        if selected_model_name:
            model_name_part = re.sub(r'[\\/*?:"<>| ]', '_', selected_model_name)
            model_name_part = model_name_part[:50]
            default_name = f"launch_{model_name_part}.ps1"

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
                fh.write("$ErrorActionPreference = 'Stop'\n\n")

                venv = self.venv_dir.get()
                if venv:
                    try:
                         venv_path = Path(venv)
                         act_script = venv_path / "Scripts" / "Activate.ps1"
                         if act_script.exists():
                             ps_act_path = f'$(Join-Path "{venv_path.resolve()}" "Scripts" "Activate.ps1")'
                             fh.write(f'Write-Host "Activating virtual environment: {venv}" -ForegroundColor Cyan\n')
                             fh.write(f'try {{ . {ps_act_path} }} catch {{ Write-Error "Failed to activate venv: $($_.Exception.Message)"; exit 1 }}\n\n')
                         else:
                              fh.write(f'Write-Warning "Virtual environment activation script (Activate.ps1) not found at: {act_script}"\n\n')
                    except Exception as path_ex:
                         fh.write(f'Write-Warning "Could not process venv path \'{venv}\': {path_ex}"\n\n')

                fh.write(f'Write-Host "Launching llama-server..." -ForegroundColor Green\n')
                ps_cmd_parts = []
                # Escape backticks and double quotes within the command string for PowerShell
                # Then quote individual arguments for PowerShell
                quoted_cmd_list = []
                for arg in cmd_list:
                     escaped_arg = arg.replace('`', '``').replace('"', '`"')
                     # Quote arguments containing spaces or special characters for PS
                     if re.search(r'[ ";&()|<>*?]', escaped_arg):
                          quoted_cmd_list.append(f'"{escaped_arg}"')
                     else:
                          quoted_cmd_list.append(escaped_arg)

                # For the executable path specifically, use & operator if needed
                exe_path_ps = quoted_cmd_list[0]
                # Check if quoting/& is needed for the path itself (common if spaces)
                if ' ' in cmd_list[0] or '&' in cmd_list[0]: # Check original arg for simplicity
                    ps_cmd_parts.append(f'& {exe_path_ps}')
                else:
                    ps_cmd_parts.append(exe_path_ps) # No & needed if path is simple

                ps_cmd_parts.extend(quoted_cmd_list[1:]) # Add the rest of the arguments

                fh.write(" ".join(ps_cmd_parts) + "\n\n")
                fh.write('Write-Host "Server process likely finished or detached." -ForegroundColor Yellow\n')
                fh.write('# Pause if script is run directly by double-clicking\n')
                fh.write('if ($Host.Name -eq "ConsoleHost") {\n')
                fh.write('    Read-Host -Prompt "Press Enter to continue"\n')
                fh.write('}\n')


            messagebox.showinfo("Saved", f"PowerShell script written to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Script Save Error", f"Could not save script:\n{exc}")
            traceback.print_exc()


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
        preferred_themes = ['win11dark','win11light','vista', 'xpnative', 'winnative', 'clam', 'alt', 'default']
        used_theme = False
        for theme in preferred_themes:
            if theme in themes:
                try:
                     print(f"Applying theme: {theme}")
                     style.theme_use(theme)
                     used_theme = True
                     # Basic heuristic to attempt dark theme configuration if name suggests it
                     if any(dark_name in theme.lower() for dark_name in ['dark', 'black', 'highcontrast']) or theme == 'clam':
                          try:
                               root.configure(bg=style.lookup('TFrame', 'background'))
                               style.configure('TFrame', background=style.lookup('TFrame', 'background'))
                               style.configure('TLabel', background=style.lookup('TFrame', 'background'), foreground=style.lookup('TLabel', 'foreground'))
                               style.configure('TButton', background=style.lookup('TButton', 'background'), foreground=style.lookup('TButton', 'foreground'))
                               style.configure('TCheckbutton', background=style.lookup('TFrame', 'background'), foreground=style.lookup('TCheckbutton', 'foreground'))
                               style.configure('TCombobox', fieldbackground=style.lookup('TCombobox', 'fieldbackground'), foreground=style.lookup('TCombobox', 'foreground'), selectbackground=style.lookup('TCombobox', 'selectbackground'), selectforeground=style.lookup('TCombobox', 'selectforeground'))
                          except tk.TclError as style_err:
                               print(f"Note: Could not fully configure dark theme styles: {style_err}")
                     break
                except tk.TclError as e:
                     print(f"Failed to apply theme {theme}: {e}")
                     continue
        if not used_theme:
             print("Could not apply preferred ttk themes.")
    except Exception as e:
         print(f"ttk themes not available or failed to apply: {e}")


    app  = LlamaCppLauncher(root)
    root.protocol("WM_DELETE_WINDOW", app.on_exit)
    root.mainloop()