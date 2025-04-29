#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLaMa.cpp HTTP‑Server Launcher (GUI)

• Keeps all explanatory labels / section headers
• Adds --n‑gpu‑layers, --tensor‑split, --flash‑attn controls
• Windows‑friendly persistence (falls back to %APPDATA%)
• Multi-directory model scanning and selection
• Finds server executable in common subdirs (select root llama.cpp dir)
"""

import json
import os
import re
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox
from threading import Thread # For background scanning

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
        self.root.geometry("900x750") # Increased height a bit
        self.root.minsize(800, 650)

        # ------------------------------------------------ persistence --
        self.config_path = self._get_config_path() # Use helper
        self.saved_configs = {}
        # App settings now include model directories
        self.app_settings = {
            "last_llama_cpp_dir": "",
            "last_venv_dir":      "",
            "last_model_path":    "", # Stores the *full path* of the last selected model
            "model_dirs":         [], # List of directories to scan
        }

        # ------------------------------------------------ Tk variables --
        self.llama_cpp_dir   = tk.StringVar()
        self.venv_dir        = tk.StringVar()
        # self.model_path is still used internally for the selected model's full path
        self.model_path      = tk.StringVar()
        # New variables for model selection
        self.model_dirs_listvar = tk.StringVar() # For the listbox display
        self.selected_model_display_name = tk.StringVar() # Bound to combobox
        self.scan_status_var = tk.StringVar(value="Scan models to populate list.")

        self.cache_type_k    = tk.StringVar(value="f16")
        self.threads         = tk.StringVar(value="4")
        self.n_gpu_layers    = tk.StringVar(value="0")
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
        self.gpu_count_var   = tk.IntVar(value=0); self.gpu_vars = []
        self.mlock           = tk.BooleanVar(value=False)
        self.no_kv_offload   = tk.BooleanVar(value=False)
        self.host            = tk.StringVar(value="127.0.0.1")
        self.port            = tk.StringVar(value="8080")
        self.config_name     = tk.StringVar(value="default_config")

        # Internal state
        self.model_dirs = [] # Python list storing Path objects
        self.found_models = {} # {display_name: full_path_obj}

        # load previous settings
        self._load_saved_configs() # Loads app_settings and saved_configs
        self.llama_cpp_dir.set(self.app_settings.get("last_llama_cpp_dir", ""))
        self.venv_dir.set(self.app_settings.get("last_venv_dir", ""))
        # Load model dirs list (convert strings back to Path if needed)
        self.model_dirs = [Path(d) for d in self.app_settings.get("model_dirs", []) if d] # Filter empty strings
        # model_path is set after scan based on last_model_path

        # build GUI
        self._create_widgets()

        # Populate model directories listbox
        self._update_model_dirs_listbox()

        # Perform initial scan (in background) if dirs exist
        if self.model_dirs:
            self.scan_status_var.set("Scanning on startup...")
            # Run initial scan in background to avoid blocking UI
            scan_thread = Thread(target=self._scan_model_dirs, daemon=True)
            scan_thread.start()


    # ═════════════════════════════════════════════════════════════════
    #  persistence helpers
    # ═════════════════════════════════════════════════════════════════
    def _get_config_path(self):
        """Determines the config path, preferring local, falling back to AppData."""
        local_path = Path("llama_cpp_configs.json")
        try:
            # Test if we can write locally by trying to open/close a file
            with local_path.open("a") as f:
                pass
            # If successful, delete the potentially created empty file only if it was just created
            if local_path.exists() and local_path.stat().st_size == 0:
                 try:
                    local_path.unlink()
                 except OSError: # File might be locked briefly
                    pass
            return local_path
        except (OSError, PermissionError, IOError):
            # Fallback to AppData or home/.config
            if sys.platform == "win32":
                appdata = os.getenv("APPDATA")
                if appdata:
                    fallback_dir = Path(appdata) / "LlamaCppLauncher"
                else: # Should not happen on modern Windows, but fallback further
                    fallback_dir = Path.home() / ".llama_cpp_launcher"
            else: # Linux/macOS
                 fallback_dir = Path.home() / ".config" / "llama_cpp_launcher"

            fallback_dir.mkdir(parents=True, exist_ok=True)
            return fallback_dir / "llama_cpp_configs.json"


    def _load_saved_configs(self):
        if self.config_path.exists():
            try:
                data = json.loads(self.config_path.read_text(encoding="utf-8"))
                self.saved_configs = data.get("configs", {})
                # Merge loaded settings with defaults, preferring loaded ones
                loaded_app_settings = data.get("app_settings", {})
                self.app_settings.update(loaded_app_settings)
            except Exception as exc:
                messagebox.showerror("Config Load Error", f"Could not load config from:\n{self.config_path}\n\nError: {exc}\n\nUsing default settings.")
                # Reset to defaults if load fails badly
                self.app_settings = {
                    "last_llama_cpp_dir": "", "last_venv_dir": "", "last_model_path": "", "model_dirs": []
                }
                self.saved_configs = {}


    def _save_configs(self):
        # Ensure model_dirs are strings for JSON serialization
        self.app_settings["model_dirs"] = [str(p) for p in self.model_dirs]
        # Make sure last_model_path is updated with the currently selected one
        self.app_settings["last_model_path"] = self.model_path.get()

        payload = {
            "configs":      self.saved_configs,
            "app_settings": self.app_settings,
        }
        try:
            self.config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
             # Attempt to get a fallback path if the original failed (e.g., permissions changed)
            original_path = self.config_path
            self.config_path = self._get_config_path() # Re-evaluate path
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

        # bottom buttons
        bar = ttk.Frame(self.root); bar.pack(fill="x", padx=10, pady=(0, 10)) # Reduced bottom padding
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
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width)) # Resize inner frame
        canvas.pack(side="left", fill="both", expand=True); vs.pack(side="right", fill="y")


        r = 0
        # --- Directories ---
        ttk.Label(inner, text="Directories & Model", font=("TkDefaultFont", 12, "bold"))\
            .grid(column=0, row=r, sticky="w", padx=10, pady=(10,5)); r += 1
        ttk.Separator(inner, orient="horizontal")\
            .grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=5); r += 1 # Span 4 now

        # Llama.cpp Directory
        ttk.Label(inner, text="LLaMa.cpp Root Directory:")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Entry(inner, textvariable=self.llama_cpp_dir, width=50)\
            .grid(column=1, row=r, columnspan=2, sticky="ew", padx=5, pady=3) # Span 2
        ttk.Button(inner, text="Browse…", command=lambda: self._browse_dir(self.llama_cpp_dir))\
            .grid(column=3, row=r, padx=5, pady=3) # Column 3
        # --- UPDATED LABEL TEXT ---
        ttk.Label(inner, text="Select the main 'llama.cpp' folder. The app will search for the server executable.", font=("TkSmallCaptionFont"))\
            .grid(column=1, row=r+1, columnspan=3, sticky="w", padx=5); r += 2

        # Virtual Env Directory
        ttk.Label(inner, text="Virtual Environment (optional):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Entry(inner, textvariable=self.venv_dir, width=50)\
            .grid(column=1, row=r, columnspan=2, sticky="ew", padx=5, pady=3) # Span 2
        vf = ttk.Frame(inner); vf.grid(column=3, row=r, sticky="w", padx=5, pady=3) # Column 3
        ttk.Button(vf, text="Browse…", command=lambda: self._browse_dir(self.venv_dir)).pack(side="left", padx=2)
        ttk.Button(vf, text="Clear",   command=self._clear_venv).pack(side="left", padx=2)
        ttk.Label(inner, text="If set, activates this Python venv before launching.", font=("TkSmallCaptionFont"))\
            .grid(column=1, row=r+1, columnspan=3, sticky="w", padx=5); r += 2

        # --- Model Directories ---
        ttk.Label(inner, text="Model Search Directories:")\
            .grid(column=0, row=r, sticky="nw", padx=10, pady=3) # Use 'nw' for alignment

        # Frame for listbox and scrollbar
        dir_list_frame = ttk.Frame(inner)
        dir_list_frame.grid(column=1, row=r, sticky="nsew", padx=5, pady=3, rowspan=2) # Span 2 rows
        dir_sb = ttk.Scrollbar(dir_list_frame, orient=tk.VERTICAL)
        self.model_dirs_listbox = tk.Listbox(dir_list_frame, listvariable=self.model_dirs_listvar,
                                             height=4, width=48, # Adjusted width
                                             yscrollcommand=dir_sb.set, exportselection=False)
        dir_sb.config(command=self.model_dirs_listbox.yview)
        dir_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.model_dirs_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Frame for Add/Remove buttons
        dir_btn_frame = ttk.Frame(inner)
        dir_btn_frame.grid(column=2, row=r, sticky="ew", padx=5, pady=3, rowspan=2) # Span 2 rows
        ttk.Button(dir_btn_frame, text="Add Dir…", width=10, command=self._add_model_dir)\
           .pack(side=tk.TOP, pady=2, fill=tk.X)
        ttk.Button(dir_btn_frame, text="Remove Dir", width=10, command=self._remove_model_dir)\
           .pack(side=tk.TOP, pady=2, fill=tk.X)
        r += 2 # Increment by 2 because listbox area spans 2 rows visually

        # --- Model Selection ---
        ttk.Label(inner, text="Select Model:")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        self.model_combobox = ttk.Combobox(inner, textvariable=self.selected_model_display_name,
                                           width=48, state="readonly") # Start readonly
        self.model_combobox.grid(column=1, row=r, sticky="ew", padx=5, pady=3)
        self.model_combobox.bind("<<ComboboxSelected>>", self._on_model_selected)

        scan_btn = ttk.Button(inner, text="Scan Models", command=self._trigger_scan)
        scan_btn.grid(column=2, row=r, sticky="ew", padx=5, pady=3)

        # Status label for scanning
        self.scan_status_label = ttk.Label(inner, textvariable=self.scan_status_var, foreground="grey", font=("TkSmallCaptionFont"))
        self.scan_status_label.grid(column=1, row=r+1, columnspan=2, sticky="w", padx=5); r += 2

        # --- Basic Settings ---
        ttk.Label(inner, text="Basic Settings", font=("TkDefaultFont", 12, "bold"))\
            .grid(column=0, row=r, sticky="w", padx=10, pady=(20,5)); r += 1
        ttk.Separator(inner, orient="horizontal")\
            .grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=5); r += 1 # Span 4

        ttk.Label(inner, text="Threads:")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Entry(inner, textvariable=self.threads, width=10)\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, text="Number of CPU threads to use (-t)", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r, columnspan=2, sticky="w", padx=5, pady=3); r += 1

        ttk.Label(inner, text="Context Size:")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ctx_f = ttk.Frame(inner); ctx_f.grid(column=1, row=r, columnspan=3, sticky="ew", padx=5, pady=3) # Span 3
        ctx_slider = ttk.Scale(ctx_f, from_=1024, to=1000000, orient="horizontal",
                               variable=self.ctx_size, command=self._update_ctx_label_from_slider) # Renamed handler
        ctx_slider.grid(column=0, row=0, sticky="ew", padx=(0, 5))
        ctx_slider.set(self.ctx_size.get())
        self.ctx_label = ttk.Label(ctx_f, text=f"{self.ctx_size.get():,}", width=9, anchor='e')
        self.ctx_label.grid(column=1, row=0, padx=5)
        self.ctx_entry = ttk.Entry(ctx_f, width=9)
        self.ctx_entry.insert(0, str(self.ctx_size.get())); self.ctx_entry.grid(column=2, row=0, padx=5)
        ttk.Button(ctx_f, text="Set", command=self._override_ctx_size, width=4).grid(column=3, row=0, padx=(0, 5))
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

        # --- Network ---
        ttk.Label(inner, text="Network Settings", font=("TkDefaultFont", 12, "bold"))\
            .grid(column=0, row=r, sticky="w", padx=10, pady=(20,5)); r += 1
        ttk.Separator(inner, orient="horizontal")\
            .grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=5); r += 1 # Span 4

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

        inner.columnconfigure(1, weight=1) # Make column 1 expandable

    # ░░░░░ ADVANCED TAB ░░░░░
    def _setup_advanced_tab(self, parent):
        canvas = tk.Canvas(parent); vs = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner  = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(yscrollcommand=vs.set,
                                                             scrollregion=canvas.bbox("all")))
        canvas_window = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width)) # Resize inner frame
        canvas.pack(side="left", fill="both", expand=True); vs.pack(side="right", fill="y")

        r = 0
        ttk.Label(inner, text="GPU Settings", font=("TkDefaultFont", 12, "bold"))\
            .grid(column=0, row=r, sticky="w", padx=10, pady=(10,5)); r += 1
        ttk.Separator(inner, orient="horizontal")\
            .grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=5); r += 1

        ttk.Label(inner, text="Number of CUDA Devices:")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Spinbox(inner, from_=0, to=8, textvariable=self.gpu_count_var, width=5,
                    command=self._update_gpu_checkboxes)\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, text="Enable checkboxes for GPUs to use (requires CUDA build)", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r, columnspan=2, sticky="w", padx=5); r += 1


        self.gpu_checkbox_frame = ttk.Frame(inner)
        self.gpu_checkbox_frame.grid(column=0, row=r, columnspan=4, sticky="w", padx=10, pady=3)
        self._update_gpu_checkboxes(); r += 1

        ttk.Label(inner, text="GPU Layers (--n-gpu-layers):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Entry(inner, textvariable=self.n_gpu_layers, width=10)\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, text="Layers to offload to GPU(s) (0 = CPU only)", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r, columnspan=2, sticky="w", padx=5, pady=3); r += 1

        ttk.Label(inner, text="Tensor Split (--tensor-split):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Entry(inner, textvariable=self.tensor_split, width=25)\
            .grid(column=1, row=r, columnspan=2, sticky="w", padx=5, pady=3) # Span 2
        ttk.Label(inner, text="e.g., '3,1' splits layers 75%/25% across 2 GPUs", font=("TkSmallCaptionFont"))\
            .grid(column=1, row=r+1, columnspan=3, sticky="w", padx=5); r += 2 # Next row for explanation

        ttk.Label(inner, text="Flash Attention (--flash-attn):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Checkbutton(inner, variable=self.flash_attn)\
            .grid(column=1, row=r, sticky="w", padx=5, pady=3)
        ttk.Label(inner, text="Use Flash Attention kernel (CUDA only, requires specific build)", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r, columnspan=2, sticky="w", padx=5, pady=3); r += 1

        # Memory settings --------------------------------------------------
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

        # Performance ------------------------------------------------------
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
        ttk.Button(frame, text="Save Current Settings", command=self._save_configuration)\
            .grid(column=2, row=r, padx=5, pady=3); r += 1

        ttk.Separator(frame, orient='horizontal')\
           .grid(column=0, row=r, columnspan=3, sticky='ew', padx=5, pady=10); r += 1

        # --- CORRECTED INDENTATION HERE ---
        ttk.Label(frame, text="Saved Configurations:")\
            .grid(column=0, row=r, sticky="w", padx=5, pady=(10,3)); r += 1
        # --- END CORRECTION ---

        lb_frame = ttk.Frame(frame); lb_frame.grid(column=0, row=r, columnspan=3,
                                                   sticky="nsew", padx=5, pady=3)
        sb = ttk.Scrollbar(lb_frame); sb.pack(side="right", fill="y")
        self.config_listbox = tk.Listbox(lb_frame, yscrollcommand=sb.set, height=10, exportselection=False)
        self.config_listbox.pack(side="left", fill="both", expand=True)
        sb.config(command=self.config_listbox.yview); r += 1

        btns = ttk.Frame(frame); btns.grid(column=0, row=r, columnspan=3, sticky="ew", padx=5, pady=5)
        ttk.Button(btns, text="Load Selected",   command=self._load_configuration).pack(side="left", padx=5)
        ttk.Button(btns, text="Delete Selected", command=self._delete_configuration).pack(side="left", padx=5)

        frame.columnconfigure(1, weight=1); frame.rowconfigure(4, weight=1) # Row 4 is now listbox frame index
        self._update_config_listbox()
        
    # ═════════════════════════════════════════════════════════════════
    #  Model Directory and Scanning Logic
    # ═════════════════════════════════════════════════════════════════

    def _add_model_dir(self):
        """Adds a directory to the model search paths."""
        # Use the last successfully added/browsed directory as a starting point?
        initial_dir = self.model_dirs[-1] if self.model_dirs else None
        directory = filedialog.askdirectory(title="Select Model Directory", initialdir=initial_dir)
        if directory:
            p = Path(directory)
            if p not in self.model_dirs:
                self.model_dirs.append(p)
                self._update_model_dirs_listbox()
                self._save_configs() # Persist the change
                self._trigger_scan() # Automatically scan after adding
            else:
                messagebox.showinfo("Info", "Directory already in list.")

    def _remove_model_dir(self):
        """Removes the selected directory from the model search paths."""
        selection = self.model_dirs_listbox.curselection()
        if not selection:
            messagebox.showerror("Error", "Select a directory to remove.")
            return
        index = selection[0]
        del self.model_dirs[index]
        self._update_model_dirs_listbox()
        self._save_configs() # Persist the change
        self._trigger_scan() # Automatically re-scan after removing

    def _update_model_dirs_listbox(self):
        """Updates the listbox display from the self.model_dirs list."""
        # Keep track of selection
        current_selection = self.model_dirs_listbox.curselection()
        self.model_dirs_listbox.delete(0, tk.END)
        for p in self.model_dirs:
            self.model_dirs_listbox.insert(tk.END, str(p))
        # Try to restore selection
        if current_selection:
            new_index = min(current_selection[0], self.model_dirs_listbox.size() - 1)
            if new_index >= 0:
                self.model_dirs_listbox.selection_set(new_index)
                self.model_dirs_listbox.activate(new_index)
                self.model_dirs_listbox.see(new_index)


    def _trigger_scan(self):
        """Initiates model scanning in a background thread."""
        self.scan_status_var.set("Scanning...")
        self.model_combobox.set("") # Clear current selection display
        self.model_combobox.config(values=[]) # Clear dropdown list
        self.model_combobox.config(state="disabled") # Disable while scanning

        # Run scan in background to avoid freezing UI
        scan_thread = Thread(target=self._scan_model_dirs, daemon=True)
        scan_thread.start()

    def _scan_model_dirs(self):
        """Scans configured directories for GGUF models (runs in background thread)."""
        found = {} # {display_name: full_path_obj}
        # Regex to capture base name and part number for standard multipart GGUF
        # Handles variations like .gguf, -F00001-of-.gguf etc. loosely
        multipart_pattern = re.compile(r"^(.*?)(?:-\d{5}-of-\d{5}|-F\d+)\.gguf$", re.IGNORECASE)
        # Regex to specifically find the *first* part (00001 or F1)
        first_part_pattern = re.compile(r"^(.*?)-(?:00001-of-\d{5}|F1)\.gguf$", re.IGNORECASE)

        processed_multipart_bases = set() # Keep track of multipart models already added

        for model_dir in self.model_dirs:
            if not model_dir.is_dir():
                continue
            try:
                # Using iglob for potentially large directories - slightly more memory efficient maybe?
                # rglob scans recursively
                for gguf_path in model_dir.rglob('*.gguf'):
                    if gguf_path.is_file():
                        filename = gguf_path.name

                        # Check if it's the first part of a multipart model
                        first_part_match = first_part_pattern.match(filename)
                        if first_part_match:
                            base_name = first_part_match.group(1)
                            if base_name not in processed_multipart_bases:
                                display_name = base_name # Use the base name before numbering
                                found[display_name] = gguf_path
                                processed_multipart_bases.add(base_name)
                            continue # Don't process this file further if it's a first part

                        # Check if it's *any* part of a multipart model (and not the first)
                        multi_match = multipart_pattern.match(filename)
                        if multi_match:
                            base_name = multi_match.group(1)
                            # If we encounter any part, mark the base as processed so single files aren't added later
                            processed_multipart_bases.add(base_name)
                            continue # Skip non-first parts

                        # If it's not multipart and hasn't been processed as such, and not ignored
                        if filename.lower().endswith(".gguf") and \
                           "mmproj" not in filename.lower() and \
                           gguf_path.stem not in processed_multipart_bases:
                             display_name = gguf_path.stem # Name without extension
                             found[display_name] = gguf_path

            except Exception as e:
                # Consider logging this more formally or showing a non-blocking error
                print(f"Error scanning directory {model_dir}: {e}")

        # --- Update UI (must be done in main thread) ---
        self.root.after(0, self._update_model_combobox, found)

    def _update_model_combobox(self, found_models_dict):
        """Updates the model combobox - called via root.after from scan thread."""
        self.found_models = found_models_dict
        model_names = sorted(list(self.found_models.keys()))

        if not model_names:
            self.scan_status_var.set("No GGUF models found in specified directories.")
            self.model_combobox.config(values=[], state="disabled")
            self.selected_model_display_name.set("")
            self.model_path.set("") # Clear internal path
        else:
            self.model_combobox.config(values=model_names, state="readonly")
            self.scan_status_var.set(f"Scan complete. Found {len(model_names)} models.")

            # Try to restore previous selection
            last_full_path_str = self.app_settings.get("last_model_path", "")
            restored = False
            if last_full_path_str:
                last_full_path = Path(last_full_path_str)
                # Find the display name corresponding to the saved path
                for display_name, full_path in self.found_models.items():
                    if full_path == last_full_path:
                        self.selected_model_display_name.set(display_name)
                        self.model_path.set(str(full_path)) # Update internal path too
                        restored = True
                        break

            # If no previous selection or previous is no longer found, select first item
            if not restored and model_names:
                 first_model_name = model_names[0]
                 self.selected_model_display_name.set(first_model_name)
                 self.model_path.set(str(self.found_models[first_model_name]))
                 # Also update app_settings so this becomes the default if closed now
                 self.app_settings["last_model_path"] = self.model_path.get()

    def _on_model_selected(self, event=None):
        """Callback when a model is selected from the combobox."""
        selected_name = self.selected_model_display_name.get()
        if selected_name in self.found_models:
            full_path = self.found_models[selected_name]
            self.model_path.set(str(full_path))
            # Save this selection for next launch
            self.app_settings["last_model_path"] = str(full_path)
            # Optionally save immediately:
            # self._save_configs()
        else:
            # This shouldn't happen with readonly combobox, but good practice
            self.model_path.set("")
            self.app_settings["last_model_path"] = ""


    # ═════════════════════════════════════════════════════════════════
    #  dynamic GPU checkboxes
    # ═════════════════════════════════════════════════════════════════
    def _update_gpu_checkboxes(self):
        for w in self.gpu_checkbox_frame.winfo_children(): w.destroy()
        self.gpu_vars.clear()
        count = self.gpu_count_var.get()
        if count > 0:
            ttk.Label(self.gpu_checkbox_frame, text="Use GPUs:").pack(side="left", padx=(0,5))
        for i in range(count):
            v = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.gpu_checkbox_frame, text=f"CUDA {i}", variable=v)\
                .pack(side="left", padx=2)
            self.gpu_vars.append(v)

    # ═════════════════════════════════════════════════════════════════
    #  misc helpers
    # ═════════════════════════════════════════════════════════════════
    def _clear_venv(self):
        self.venv_dir.set("")
        self.app_settings["last_venv_dir"] = ""
        # No need to save config here unless explicitly desired

    def _update_ctx_label_from_slider(self, value_str):
        # Slider callback provides string, convert to float then int
        try:
            value = int(float(value_str))
             # Round to nearest 1024, minimum 1024
            rounded = max(1024, round(value / 1024) * 1024)
            self.ctx_size.set(rounded) # Update the IntVar
            self._sync_ctx_display(rounded) # Update label and entry
        except ValueError:
            pass # Ignore intermediate invalid values during sliding

    def _override_ctx_size(self):
        try:
            raw = int(self.ctx_entry.get())
            # Round to nearest 1024, minimum 1024
            rounded = max(1024, round(raw/1024)*1024)
            self.ctx_size.set(rounded) # Update the IntVar
            self._sync_ctx_display(rounded) # Update label and entry
        except ValueError:
            messagebox.showerror("Input Error", "Context size must be a valid number.")
            # Restore entry from the current IntVar value
            self._sync_ctx_display(self.ctx_size.get())


    def _sync_ctx_display(self, value):
        """Updates context label, entry, and slider from a given value."""
        formatted_value = f"{value:,}"
        str_value = str(value)
        # Update label
        self.ctx_label.config(text=formatted_value)
        # Update entry (only if different to avoid cursor jump)
        if self.ctx_entry.get() != str_value:
            self.ctx_entry.delete(0, tk.END)
            self.ctx_entry.insert(0, str_value)
        # Update slider via the variable binding
        # self.ctx_size.set(value) # Already done before calling usually


    def _browse_dir(self, var_to_set):
        # Determine key for app_settings based on the variable being set
        if var_to_set == self.llama_cpp_dir:
            setting_key = "last_llama_cpp_dir"
            title = "Select LLaMa.cpp Root Directory"
        elif var_to_set == self.venv_dir:
             setting_key = "last_venv_dir"
             title = "Select Virtual Environment Directory"
        else:
             setting_key = None # Should not happen for defined browse buttons
             title = "Select Directory"

        initial = self.app_settings.get(setting_key) if setting_key else None
        if initial and not Path(initial).is_dir(): initial = None # Check if still valid

        d = filedialog.askdirectory(initialdir=initial, title=title)
        if d:
            var_to_set.set(d)
            if setting_key:
                self.app_settings[setting_key] = d
            # No need to save config here unless explicitly desired


    def _find_server_executable(self, base_dir: Path) -> Path | None:
        """
        Searches for the llama-server executable in common locations
        relative to the provided base directory.

        Args:
            base_dir: The Path object for the selected root llama.cpp directory.

        Returns:
            The full Path to the executable if found, otherwise None.
        """
        exe_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"

        # Define relative paths to search within the base_dir. Order matters.
        search_paths = [
            # CMake typical structures (prioritize Release)
            Path("build/bin/Release"),
            Path("build/bin/Debug"),
            Path("build/bin"),
            Path("build/Release"), # Sometimes build type is top level
            Path("build/Debug"),
            Path("build"),
            # Makefile typical structures
            Path("."),
             # Alternative structures (less common)
            Path("bin/Release"),
            Path("bin/Debug"),
            Path("bin"),
        ]

        for rel_path in search_paths:
            # Construct full path and resolve potentially relative parts like '.'
            potential_path = (base_dir / rel_path / exe_name).resolve()
            if potential_path.is_file():
                # print(f"Found server executable at: {potential_path}") # Debug print
                return potential_path

        # print(f"Server executable '{exe_name}' not found in standard locations under: {base_dir}") # Debug print
        return None

    # ═════════════════════════════════════════════════════════════════
    #  configuration (save / load / delete)
    # ═════════════════════════════════════════════════════════════════
    def _current_cfg(self):
        # Note: Does NOT include model_dirs (app setting), but DOES include selected model path
        return {
            "llama_cpp_dir": self.llama_cpp_dir.get(),
            "venv_dir":      self.venv_dir.get(),
            "model_path":    self.model_path.get(), # Saves the full path of the selected model
            "cache_type_k":  self.cache_type_k.get(),
            "threads":       self.threads.get(),
            "n_gpu_layers":  self.n_gpu_layers.get(),
            "no_mmap":       self.no_mmap.get(),
            "no_cnv":        self.no_cnv.get(),
            "prio":          self.prio.get(),
            "temperature":   self.temperature.get(),
            "min_p":         self.min_p.get(),
            "ctx_size":      self.ctx_size.get(),
            "seed":          self.seed.get(),
            "flash_attn":    self.flash_attn.get(),
            "tensor_split":  self.tensor_split.get(),
            "main_gpu":      self.main_gpu.get(),
            # Store which GPU indices are checked
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
        self._save_configs() # Persist all configs and app settings
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

        # --- Apply settings from loaded config ---
        self.llama_cpp_dir.set(cfg.get("llama_cpp_dir",""))
        self.venv_dir.set(cfg.get("venv_dir",""))

        # Handle model path: Set internal path and try to select in combobox
        loaded_model_path_str = cfg.get("model_path", "")
        self.model_path.set(loaded_model_path_str)
        self.selected_model_display_name.set("") # Clear display name first
        if loaded_model_path_str:
            found_display_name = None
            try:
                loaded_model_path = Path(loaded_model_path_str)
                # Find display name matching the full path
                for display_name, full_path in self.found_models.items():
                    if full_path == loaded_model_path:
                        found_display_name = display_name
                        break
                if found_display_name:
                    self.selected_model_display_name.set(found_display_name)
                else:
                    # Model from config not found in current scan results
                    # Display the filename part as a hint
                    self.selected_model_display_name.set(f"❓ {loaded_model_path.name}")
            except Exception: # Handle potential invalid path string
                 self.selected_model_display_name.set("❓ Invalid Path")
                 self.model_path.set("") # Clear invalid path

        self.cache_type_k.set(cfg.get("cache_type_k","f16"))
        self.threads.set(cfg.get("threads","4"))
        self.n_gpu_layers.set(cfg.get("n_gpu_layers","0"))
        self.no_mmap.set(cfg.get("no_mmap",False))
        self.no_cnv.set(cfg.get("no_cnv",False))
        self.prio.set(cfg.get("prio","0"))
        self.temperature.set(cfg.get("temperature","0.8"))
        self.min_p.set(cfg.get("min_p","0.05"))
        ctx = cfg.get("ctx_size", 2048)
        self.ctx_size.set(ctx)
        self._sync_ctx_display(ctx) # Update UI elements for context size
        self.seed.set(cfg.get("seed","-1"))
        self.flash_attn.set(cfg.get("flash_attn",False))
        self.tensor_split.set(cfg.get("tensor_split",""))
        self.main_gpu.set(cfg.get("main_gpu","0"))

        # Handle GPU checkboxes based on saved indices
        gpu_indices = cfg.get("gpu_indices", [])
        max_needed_gpus = (max(gpu_indices) + 1) if gpu_indices else 0
        current_gpu_count = self.gpu_count_var.get()

        # Ensure enough checkboxes exist before trying to check them
        if max_needed_gpus > current_gpu_count:
             self.gpu_count_var.set(max_needed_gpus) # Adjust count
             self._update_gpu_checkboxes() # Recreate checkboxes

        # Now check the boxes based on loaded indices
        for i, v in enumerate(self.gpu_vars):
             v.set(i in gpu_indices)

        self.mlock.set(cfg.get("mlock",False))
        self.no_kv_offload.set(cfg.get("no_kv_offload",False))
        self.host.set(cfg.get("host","127.0.0.1"))
        self.port.set(cfg.get("port","8080"))

        self.config_name.set(name) # Set the name field to the loaded config name
        messagebox.showinfo("Loaded", f"Configuration '{name}' applied.")

    def _delete_configuration(self):
        if not self.config_listbox.curselection():
            return messagebox.showerror("Error","Select a configuration to delete.")
        name = self.config_listbox.get(self.config_listbox.curselection())
        if messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete the configuration '{name}'?"):
            self.saved_configs.pop(name, None)
            self._save_configs() # Persist deletion
            self._update_config_listbox()
            messagebox.showinfo("Deleted", f"Configuration '{name}' deleted.")

    def _update_config_listbox(self):
        # Keep track of selection
        current_selection = self.config_listbox.curselection()
        selected_name = None
        if current_selection:
            selected_name = self.config_listbox.get(current_selection[0])

        self.config_listbox.delete(0, tk.END)
        sorted_names = sorted(self.saved_configs.keys())
        for cfg_name in sorted_names:
            self.config_listbox.insert(tk.END, cfg_name)

        # Try to restore selection by name
        if selected_name in sorted_names:
            try:
                 new_index = sorted_names.index(selected_name)
                 self.config_listbox.selection_set(new_index)
                 self.config_listbox.activate(new_index)
                 self.config_listbox.see(new_index)
            except ValueError:
                 pass # Should not happen if name is in sorted_names


    # ═════════════════════════════════════════════════════════════════
    #  command‑line builder
    # ═════════════════════════════════════════════════════════════════
    def _build_cmd(self):
        """Builds the command list for llama-server."""
        # --- Executable Path ---
        llama_dir_str = self.llama_cpp_dir.get()
        if not llama_dir_str:
            messagebox.showerror("Error", "LLaMa.cpp root directory is not set.")
            return None
        try:
            llama_base_dir = Path(llama_dir_str)
            if not llama_base_dir.is_dir():
                 raise NotADirectoryError()
        except Exception: # Catch invalid paths early
             messagebox.showerror("Error", f"Invalid LLaMa.cpp directory:\n{llama_dir_str}")
             return None

        # --- Use the helper to find the executable ---
        exe_path = self._find_server_executable(llama_base_dir)
        if not exe_path:
            search_locs_str = "\n - ".join([str(p) for p in [ # List common search roots for clarity
                Path("build/bin/Release"), Path("build/bin"), Path("build"), Path(".")
            ]])
            exe_base_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
            messagebox.showerror("Executable Not Found",
                                 f"Could not find '{exe_base_name}' within:\n{llama_base_dir}\n\n"
                                 f"Searched in common relative locations like:\n - {search_locs_str}\n\n"
                                 "Please ensure llama.cpp is built and the directory is correct.")
            return None
        # --- End of modification for finding executable ---

        cmd = [str(exe_path)] # Start command with the found executable path

        # --- Model Path ---
        model_full_path_str = self.model_path.get()
        if not model_full_path_str:
            messagebox.showerror("Error", "No model selected. Please scan and select a model.")
            return None
        try:
            model_full_path = Path(model_full_path_str)
            if not model_full_path.is_file():
                messagebox.showerror("Error", f"Selected model file not found:\n{model_full_path_str}\n\nPlease re-scan models.")
                return None
        except Exception: # Catch invalid paths
             messagebox.showerror("Error", f"Invalid model path selected:\n{model_full_path_str}\n\nPlease re-scan models.")
             return None

        cmd.extend(["-m", model_full_path_str])

        # --- Other Arguments ---
        self._add_arg(cmd, "--cache-type-k", self.cache_type_k.get(), "f16")
        self._add_arg(cmd, "--threads", self.threads.get(), "4")
        self._add_arg(cmd, "--n-gpu-layers", self.n_gpu_layers.get(), "0")
        self._add_arg(cmd, "--no-mmap", self.no_mmap.get())
        self._add_arg(cmd, "--mlock", self.mlock.get())
        self._add_arg(cmd, "--no-kv-offload", self.no_kv_offload.get())
        self._add_arg(cmd, "--no-cnv", self.no_cnv.get())
        self._add_arg(cmd, "--prio", self.prio.get(), "0")
        self._add_arg(cmd, "--temp", self.temperature.get(), "0.8")
        self._add_arg(cmd, "--min-p", self.min_p.get(), "0.05")
        cmd.extend(["--ctx-size", str(self.ctx_size.get())]) # Always include ctx-size
        self._add_arg(cmd, "--seed", self.seed.get(), "-1")
        self._add_arg(cmd, "--flash-attn", self.flash_attn.get())
        self._add_arg(cmd, "--tensor-split", self.tensor_split.get().strip(), "") # Add if not empty/whitespace
        self._add_arg(cmd, "--main-gpu", self.main_gpu.get(), "0")

        checked_gpu_indices = [i for i, v in enumerate(self.gpu_vars) if v.get()]
        if checked_gpu_indices:
            cmd.extend(["--device", ",".join(map(str, checked_gpu_indices))])

        self._add_arg(cmd, "--host", self.host.get(), "127.0.0.1")
        self._add_arg(cmd, "--port", self.port.get(), "8080")

        return cmd

    def _add_arg(self, cmd_list, arg_name, value, default_value=None):
        """Helper to add command line arguments if they differ from default or are boolean True."""
        is_bool_var = isinstance(value, tk.BooleanVar)
        is_bool_py = isinstance(value, bool)
        is_string = isinstance(value, str)

        if is_bool_var:
            if value.get():
                cmd_list.append(arg_name)
        elif is_bool_py:
            if value:
                 cmd_list.append(arg_name)
        elif is_string:
            actual_value = value.strip() # Treat whitespace-only as empty
            if actual_value: # Only add if non-empty after stripping
                # Add if no default specified, or if it differs from default
                if default_value is None or actual_value != default_value:
                    cmd_list.extend([arg_name, actual_value])
        # Add other types here if necessary (e.g., int, float) but most are handled via strings/bools

    # ═════════════════════════════════════════════════════════════════
    #  launch & script helpers
    # ═════════════════════════════════════════════════════════════════
    def launch_server(self):
        cmd_list = self._build_cmd()
        if not cmd_list:
            return # Error message shown by _build_cmd

        venv_path_str = self.venv_dir.get()
        final_cmd_str = subprocess.list2cmdline(cmd_list) # Properly quote args for shell

        try:
            if venv_path_str and sys.platform == "win32":
                venv_path = Path(venv_path_str)
                act_script = venv_path / "Scripts" / "activate.bat"
                if not act_script.is_file():
                    messagebox.showerror("Error", f"Venv activation script not found:\n{act_script}")
                    return
                # Launch cmd, activate venv, then run the command
                subprocess.Popen(f'start "LLaMa.cpp Server" cmd /k ""{act_script}" && {final_cmd_str}"', shell=True)

            elif venv_path_str: # Linux/macOS venv
                venv_path = Path(venv_path_str)
                act_script = venv_path / "bin" / "activate"
                if not act_script.is_file():
                    messagebox.showerror("Error", f"Venv activation script not found:\n{act_script}")
                    return
                 # Use gnome-terminal (adapt for other terminals if needed)
                # Escape quotes carefully for shell within shell
                term_cmd = f'gnome-terminal -- bash -c \'echo "Activating venv: {venv_path}" && source "{act_script}" && echo "Launching server..." && {final_cmd_str} ; read -p "Server process finished. Press Enter to close terminal..."\''
                subprocess.Popen(term_cmd, shell=True)

            else: # No venv
                if sys.platform == "win32":
                    # Use start to open a new window, provide a title
                    subprocess.Popen(f'start "LLaMa.cpp Server" {final_cmd_str}', shell=True)
                else:
                    # Use gnome-terminal (adapt)
                    term_cmd = f'gnome-terminal -- bash -c \'{final_cmd_str} ; read -p "Server process finished. Press Enter to close terminal..."\''
                    subprocess.Popen(term_cmd, shell=True)

        except Exception as exc:
            messagebox.showerror("Launch Error", f"Failed to launch server process:\n{exc}")


    def save_ps1_script(self):
        cmd_list = self._build_cmd()
        if not cmd_list:
            return

        default_name = "launch_llama_server.ps1"
        if self.selected_model_display_name.get() and not self.selected_model_display_name.get().startswith("❓"):
            model_name_part = re.sub(r'[\\/*?:"<>| ]', '_', self.selected_model_display_name.get()) # Sanitize
            model_name_part = model_name_part[:50] # Limit length
            default_name = f"launch_{model_name_part}.ps1"


        path = filedialog.asksaveasfilename(defaultextension=".ps1",
                                            initialfile=default_name,
                                            filetypes=[("PowerShell Script", "*.ps1"), ("All Files", "*.*")])
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("<#\n")
                fh.write(" .SYNOPSIS\n")
                fh.write("    Launches the LLaMa.cpp server with saved settings.\n\n")
                fh.write(" .DESCRIPTION\n")
                fh.write("    Autogenerated PowerShell script from LLaMa.cpp Launcher GUI.\n")
                fh.write("    Activates virtual environment (if configured) and starts llama-server.\n")
                fh.write("#>\n\n")
                fh.write("# Ensure script stops on errors\n")
                fh.write("$ErrorActionPreference = 'Stop'\n\n")

                # Add venv activation only for Windows PowerShell standard path
                venv = self.venv_dir.get()
                if venv:
                    try:
                         venv_path = Path(venv)
                         act_script = venv_path / "Scripts" / "Activate.ps1"
                         if act_script.exists():
                             fh.write(f'Write-Host "Activating virtual environment: {venv}" -ForegroundColor Cyan\n')
                             # Use dot-sourcing, quote path carefully
                             # Use $() for subexpression to handle paths with spaces correctly within quotes
                             fh.write(f'try {{ . "$("{act_script.resolve()}")" }} catch {{ Write-Error "Failed to activate venv: $($_.Exception.Message)"; exit 1 }}\n\n')
                         else:
                              fh.write(f'Write-Warning "Virtual environment activation script not found at: {act_script}"\n\n')
                    except Exception as path_ex:
                         fh.write(f'Write-Warning "Could not process venv path \'{venv}\': {path_ex}"\n\n')


                fh.write(f'Write-Host "Launching llama-server..." -ForegroundColor Green\n')

                # Prepare command parts, quoting arguments carefully for PowerShell
                ps_cmd_parts = []
                # Executable: Use single quotes if no single quotes inside, else double quotes
                exe_path_str = cmd_list[0]
                if "'" not in exe_path_str:
                    ps_cmd_parts.append(f"'{exe_path_str}'")
                else:
                     ps_cmd_parts.append(f'"{exe_path_str}"') # Basic double quoting

                # Process arguments
                i = 1
                while i < len(cmd_list):
                    arg = cmd_list[i]
                    is_flag = (i + 1 == len(cmd_list)) or cmd_list[i+1].startswith('-')

                    if is_flag: # Boolean flag or arg without value
                         ps_cmd_parts.append(arg)
                         i += 1
                    else: # Argument with a value
                        val = cmd_list[i+1]
                        ps_cmd_parts.append(arg)
                        # Value Quoting: Use single quotes if no single quotes present, else double quotes
                        if "'" not in val:
                            ps_cmd_parts.append(f"'{val}'")
                        else:
                            # Basic double quoting (might need ` backticks for internal quotes)
                            ps_cmd_parts.append(f'"{val}"')
                        i += 2

                # Use the call operator '&'
                fh.write("& " + " ".join(ps_cmd_parts) + "\n\n")

            messagebox.showinfo("Saved", f"PowerShell script written to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Script Save Error", f"Could not save script:\n{exc}")

    def on_exit(self):
        self._save_configs()
        self.root.destroy()


# ═════════════════════════════════════════════════════════════════════
#  main
# ═════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    # Apply a theme for a slightly more modern look (optional)
    try:
        style = ttk.Style(root)
        # Available themes: 'winnative', 'clam', 'alt', 'default', 'classic', 'vista', 'xpnative'
        themes = style.theme_names()
        # Try common preferred themes first
        preferred_themes = ['vista', 'xpnative', 'winnative', 'clam', 'alt', 'default']
        used_theme = False
        for theme in preferred_themes:
            if theme in themes:
                try:
                     style.theme_use(theme)
                     used_theme = True
                     # print(f"Using theme: {theme}") # Debug print
                     break
                except tk.TclError:
                     continue # Theme might exist but fail to apply
        if not used_theme:
             print("Could not apply preferred ttk themes.")

    except Exception:
         print("ttk themes not available or failed to apply.") # Non-critical

    app  = LlamaCppLauncher(root)
    root.protocol("WM_DELETE_WINDOW", app.on_exit)
    root.mainloop()