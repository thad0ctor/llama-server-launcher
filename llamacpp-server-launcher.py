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
import traceback # For detailed error printing

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

        # Internal state for resizing
        self._resize_start_y = 0
        self._resize_start_height = 0

        # ------------------------------------------------ persistence --
        self.config_path = self._get_config_path() # Use helper
        self.saved_configs = {}
        # App settings now include model directories and list height
        self.app_settings = {
            "last_llama_cpp_dir": "",
            "last_venv_dir":      "",
            "last_model_path":    "", # Stores the *full path* of the last selected model
            "model_dirs":         [], # List of directories to scan
            "model_list_height":  8,  # Default height for the model listbox
        }

        # ------------------------------------------------ Tk variables --
        self.llama_cpp_dir   = tk.StringVar()
        self.venv_dir        = tk.StringVar()
        # self.model_path is still used internally for the selected model's full path
        self.model_path      = tk.StringVar()
        # New variable for model listbox height (used by listbox directly)
        # We don't bind this directly, we set it from app_settings initially
        # and update app_settings when resizing is done.

        # Model listbox stuff (no specific variable needed for selected display name now)
        self.model_dirs_listvar = tk.StringVar() # For the directory listbox display
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
                # Ensure model_list_height is an int, fallback if invalid type loaded
                if not isinstance(self.app_settings.get("model_list_height"), int):
                    self.app_settings["model_list_height"] = 8 # Reset to default
            except Exception as exc:
                messagebox.showerror("Config Load Error", f"Could not load config from:\n{self.config_path}\n\nError: {exc}\n\nUsing default settings.")
                # Reset to defaults if load fails badly
                self.app_settings = {
                    "last_llama_cpp_dir": "", "last_venv_dir": "", "last_model_path": "",
                    "model_dirs": [], "model_list_height": 8
                }
                self.saved_configs = {}


    def _save_configs(self):
        # Ensure model_dirs are strings for JSON serialization
        self.app_settings["model_dirs"] = [str(p) for p in self.model_dirs]
        # Make sure last_model_path is updated with the currently selected one
        self.app_settings["last_model_path"] = self.model_path.get()
        # Model list height is updated via _end_resize or kept from initial load

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
        inner  = ttk.Frame(canvas) # <<< Definition of inner frame
        inner.bind("<Configure>", lambda e: canvas.configure(yscrollcommand=vs.set,
                                                             scrollregion=canvas.bbox("all")))
        canvas_window = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width)) # Resize inner frame
        canvas.pack(side="left", fill="both", expand=True); vs.pack(side="right", fill="y")


        r = 0 # Row counter for the 'inner' frame grid
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
        inner.rowconfigure(r, weight=0) # Keep dir list fixed height for now

        # Frame for Add/Remove buttons
        dir_btn_frame = ttk.Frame(inner)
        dir_btn_frame.grid(column=2, row=r, sticky="ew", padx=5, pady=3, rowspan=2) # Span 2 rows
        ttk.Button(dir_btn_frame, text="Add Dir…", width=10, command=self._add_model_dir)\
           .pack(side=tk.TOP, pady=2, fill=tk.X)
        ttk.Button(dir_btn_frame, text="Remove Dir", width=10, command=self._remove_model_dir)\
           .pack(side=tk.TOP, pady=2, fill=tk.X)
        r += 2 # Increment by 2 because listbox area spans 2 rows visually

        # --- Model Selection (Listbox) ---
        ttk.Label(inner, text="Select Model:")\
            .grid(column=0, row=r, sticky="nw", padx=10, pady=(10, 3)) # Add padding top

        # Frame to hold the listbox, scrollbar, and resize grip
        # DEBUG: Use tk.Frame and set background to red
        model_select_frame = tk.Frame(inner, bg='red')
        model_select_frame.grid(column=1, row=r, columnspan=2, sticky="nsew", padx=5, pady=(10,0)) # span 2, remove bottom padding

        # Configure row/column weights for the inner frame to allow expansion
        inner.rowconfigure(r, weight=1) # Allow this row to expand vertically
        # Let column 1 (containing the listbox) expand horizontally
        # Column 0 (Labels), Column 1 (Entry/List), Column 2 (Buttons/Extra), Column 3 (Browse)
        inner.columnconfigure(1, weight=1)

        # DEBUG: Ensure scrollbar is using tk parent if frame is tk
        model_list_sb = ttk.Scrollbar(model_select_frame, orient=tk.VERTICAL)
        self.model_listbox = tk.Listbox(model_select_frame,
                                        height=self.app_settings.get("model_list_height", 8), # Use persisted height
                                        width=48, # Initial width, column weight handles expansion
                                        yscrollcommand=model_list_sb.set,
                                        exportselection=False,
                                        state=tk.DISABLED) # Start disabled until scan
        model_list_sb.config(command=self.model_listbox.yview)

        model_list_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.model_listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # DEBUG: Ensure resize grip is using tk parent if frame is tk
        # Using ttk.Separator on tk.Frame works fine visually
        self.resize_grip = ttk.Separator(model_select_frame, orient=tk.HORIZONTAL, cursor="sb_v_double_arrow")
        self.resize_grip.pack(side=tk.BOTTOM, fill=tk.X, pady=(2,0)) # Place below listbox
        self.resize_grip.bind("<ButtonPress-1>", self._start_resize)
        self.resize_grip.bind("<B1-Motion>", self._do_resize)
        self.resize_grip.bind("<ButtonRelease-1>", self._end_resize)

        self.model_listbox.bind("<<ListboxSelect>>", self._on_model_selected)

        scan_btn = ttk.Button(inner, text="Scan Models", command=self._trigger_scan)
        scan_btn.grid(column=3, row=r, sticky="n", padx=5, pady=(10,3)) # Move scan button to column 3, align top

        r += 1 # Move to next row for status label

        # Status label for scanning (placed below the listbox area)
        self.scan_status_label = ttk.Label(inner, textvariable=self.scan_status_var, foreground="grey", font=("TkSmallCaptionFont"))
        self.scan_status_label.grid(column=1, row=r, columnspan=3, sticky="nw", padx=5, pady=(2,5)); # span 3, add padding top/bottom
        r += 1

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

        # --- CTX LABEL FIX: Create label and entry FIRST ---
        self.ctx_label = ttk.Label(ctx_f, text=f"{self.ctx_size.get():,}", width=9, anchor='e')
        self.ctx_entry = ttk.Entry(ctx_f, width=9)
        self.ctx_entry.insert(0, str(self.ctx_size.get()))
        # --- END CTX LABEL FIX ---

        ctx_slider = ttk.Scale(ctx_f, from_=1024, to=1000000, orient="horizontal",
                               variable=self.ctx_size, command=self._update_ctx_label_from_slider)

        # Grid the context widgets
        ctx_slider.grid(column=0, row=0, sticky="ew", padx=(0, 5))
        self.ctx_label.grid(column=1, row=0, padx=5) # Grid the pre-created label
        self.ctx_entry.grid(column=2, row=0, padx=5) # Grid the pre-created entry
        ttk.Button(ctx_f, text="Set", command=self._override_ctx_size, width=4).grid(column=3, row=0, padx=(0, 5))

        # Set the slider value *after* all related widgets are created and gridded
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

        # Ensure column 1 weight is set (important for listbox expansion)
        inner.columnconfigure(1, weight=1)

    # ░░░░░ ADVANCED TAB ░░░░░
    def _setup_advanced_tab(self, parent):
        # This tab remains unchanged
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
        # This tab remains unchanged
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

        ttk.Label(frame, text="Saved Configurations:")\
            .grid(column=0, row=r, sticky="w", padx=5, pady=(10,3)); r += 1

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
    #  Listbox Resizing Logic
    # ═════════════════════════════════════════════════════════════════

    def _start_resize(self, event):
        """Record starting position for listbox resize."""
        self._resize_start_y = event.y_root
        self._resize_start_height = self.model_listbox.cget("height")

    def _do_resize(self, event):
        """Adjust listbox height based on mouse drag."""
        delta_y = event.y_root - self._resize_start_y
        # Estimate pixels per line (adjust if needed for your font/platform)
        pixels_per_line = 15
        delta_lines = round(delta_y / pixels_per_line)

        new_height = self._resize_start_height + delta_lines
        # Clamp height to reasonable limits
        new_height = max(3, min(30, new_height)) # Min 3 lines, Max 30 lines

        if new_height != self.model_listbox.cget("height"):
            self.model_listbox.config(height=new_height)

    def _end_resize(self, event):
        """Finalize resize and save the new height."""
        final_height = self.model_listbox.cget("height")
        if final_height != self.app_settings.get("model_list_height"):
            self.app_settings["model_list_height"] = final_height
            self._save_configs() # Persist the new height

    # ═════════════════════════════════════════════════════════════════
    #  Model Directory and Scanning Logic
    # ═════════════════════════════════════════════════════════════════

    def _add_model_dir(self):
        """Adds a directory to the model search paths."""
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
        print("DEBUG: _trigger_scan called")
        self.scan_status_var.set("Scanning...")
        self.model_listbox.delete(0, tk.END) # Clear listbox visually
        self.model_path.set("") # Clear internal selected path
        # self.model_listbox.config(state=tk.DISABLED) # ← REMOVE or COMMENT OUT this line

        # Prevent user interaction during scan by disabling selection actions if needed
        # (Though often just clearing + status message is enough)
        # self.model_listbox.unbind("<<ListboxSelect>>") # Example alternative

        # Run scan in background to avoid freezing UI
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
                    print(f"DEBUG: Found file: {filename}") # Very verbose, comment out if too much
                    first_part_match = first_part_pattern.match(filename)
                    if first_part_match:
                        base_name = first_part_match.group(1)
                        if base_name not in processed_multipart_bases:
                            print(f"DEBUG: Adding multipart base: {base_name} from {filename}")
                            found[base_name] = gguf_path
                            processed_multipart_bases.add(base_name)
                        continue

                    multi_match = multipart_pattern.match(filename)
                    if multi_match:
                        base_name = multi_match.group(1)
                        # Add base name to processed even if it's not the first part,
                        # to avoid adding the base single file later if it exists.
                        if base_name not in processed_multipart_bases:
                             print(f"DEBUG: Found other multipart, adding base to processed: {base_name}")
                             processed_multipart_bases.add(base_name)
                        continue # Skip non-first parts

                    # Exclude specific substrings and already processed bases
                    if filename.lower().endswith(".gguf") and \
                       "mmproj" not in filename.lower() and \
                       gguf_path.stem not in processed_multipart_bases:
                         display_name = gguf_path.stem
                         # Check if we accidentally found a base name already added via multipart logic
                         if display_name not in found:
                            print(f"DEBUG: Adding single file model: {display_name}")
                            found[display_name] = gguf_path
                         else:
                             print(f"DEBUG: Skipping single file {display_name}, already added as multipart base.")


            except Exception as e:
                print(f"ERROR: Error scanning directory {model_dir}: {e}")
                traceback.print_exc() # Print full traceback for scan errors

        # --- Update UI (must be done in main thread) ---
        print(f"DEBUG: Scan complete. Found models dictionary: {found}") # DEBUG
        self.root.after(0, self._update_model_listbox, found) # Changed target function

    def _update_model_listbox(self, found_models_dict):
        """Populates the model listbox and handles selection restoration."""
        print(f"DEBUG: _update_model_listbox called. Received: {len(found_models_dict)} items")
        self.found_models = found_models_dict
        model_names = sorted(list(self.found_models.keys()))
        print(f"DEBUG: Model names to insert count: {len(model_names)}")

        # --- CORE FIX: Enable BEFORE modifying ---
        print("DEBUG: Setting listbox state=NORMAL before modifying.")
        self.model_listbox.config(state=tk.NORMAL)
        # --- END FIX ---

        # Clear and repopulate (now that it's enabled)
        self.model_listbox.delete(0, tk.END)
        for name in model_names:
            self.model_listbox.insert(tk.END, name)

        # Force UI update to hopefully register inserts before size check
        print("DEBUG: Calling root.update_idletasks() after inserts")
        self.root.update_idletasks()
        immediate_size = self.model_listbox.size()
        print(f"DEBUG: Listbox size immediately after insert loop + update_idletasks: {immediate_size}")

        if not model_names:
            # This case handles when the scan itself found nothing
            print("DEBUG: No models found (model_names list empty), disabling listbox.")
            self.scan_status_var.set("No GGUF models found in specified directories.")
            self.model_listbox.config(state=tk.DISABLED) # Disable if empty
            self.model_path.set("")
        else:
            # This case handles when models were found and inserted
            if immediate_size == 0 and len(model_names) > 0:
                 # This warning should hopefully NOT appear now
                 print("CRITICAL WARNING: Listbox size is 0 even after inserts and update_idletasks!")

            # Listbox should already be NORMAL here, but doesn't hurt to ensure
            self.model_listbox.config(state=tk.NORMAL)
            print("DEBUG: Listbox state confirmed NORMAL")

            # Keep the delayed selection logic from before
            def set_initial_selection():
                print("DEBUG: Running set_initial_selection via root.after")
                try:
                    list_size = self.model_listbox.size() # Check size again inside 'after'
                    print(f"DEBUG: Listbox size inside 'after' callback: {list_size}")

                    if list_size > 0:
                        print("DEBUG: Attempting to focus and select index 0")
                        self.model_listbox.focus_set()
                        self.model_listbox.selection_clear(0, tk.END)
                        self.model_listbox.selection_set(0)
                        self.model_listbox.activate(0)
                        self.model_listbox.see(0)
                        self.model_listbox.update_idletasks()

                        current_sel = self.model_listbox.curselection()
                        print(f"DEBUG: curselection() after set/activate/update: {current_sel}")

                        if current_sel == (0,):
                            print("DEBUG: Selection confirmed for index 0")
                            print("DEBUG: Selection successful, calling _on_model_selected manually.")
                            self._on_model_selected()
                        else:
                            print("DEBUG: selection_set(0) still did NOT take effect.")
                    else:
                        print("DEBUG: Listbox size is still 0 inside 'after', cannot select.")

                    listbox_h = self.model_listbox.winfo_height()
                    frame_h = self.model_listbox.master.winfo_height()
                    print(f"DEBUG: After selection attempt - Listbox winfo_height: {listbox_h}, Frame winfo_height: {frame_h}")

                except Exception as e:
                    print(f"ERROR: Exception during delayed listbox selection/update: {e}")
                    traceback.print_exc()

            # Schedule the selection logic
            self.root.after(100, set_initial_selection) # Can try a shorter delay now

            self.scan_status_var.set(f"Scan complete. Found {len(model_names)} models.")

        # Re-bind event handler if it was unbound in _trigger_scan
        # self.model_listbox.bind("<<ListboxSelect>>", self._on_model_selected)

    def _on_model_selected(self, event=None):
        """Callback when a model is selected from the listbox."""
        print("DEBUG: _on_model_selected called")
        selection = self.model_listbox.curselection()
        if not selection:
            print("DEBUG: No selection in listbox.")
            # If nothing is selected (e.g., after clearing), clear the path
            self.model_path.set("")
            self.app_settings["last_model_path"] = ""
            return

        index = selection[0]
        selected_name = self.model_listbox.get(index)
        print(f"DEBUG: Listbox selection index: {index}, name: '{selected_name}'")

        if selected_name in self.found_models:
            full_path = self.found_models[selected_name]
            print(f"DEBUG: Setting model_path to: {full_path}")
            self.model_path.set(str(full_path))
            # Save this selection for next launch
            self.app_settings["last_model_path"] = str(full_path)
            # Optionally save immediately (can be noisy during selection changes)
            # self._save_configs()
        else:
            # Should not happen if listbox is populated correctly
            print(f"WARNING: Selected model '{selected_name}' not found in internal dictionary self.found_models.")
            self.model_path.set("")
            self.app_settings["last_model_path"] = ""


    # ═════════════════════════════════════════════════════════════════
    #  dynamic GPU checkboxes
    # ═════════════════════════════════════════════════════════════════
    def _update_gpu_checkboxes(self):
        # This function remains unchanged
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
        # This function remains unchanged
        self.venv_dir.set("")
        self.app_settings["last_venv_dir"] = ""

    def _update_ctx_label_from_slider(self, value_str):
        # This function remains unchanged
        try:
            value = int(float(value_str))
            rounded = max(1024, round(value / 1024) * 1024)
            self.ctx_size.set(rounded)
            self._sync_ctx_display(rounded)
        except ValueError:
            pass

    def _override_ctx_size(self):
        # This function remains unchanged
        try:
            raw = int(self.ctx_entry.get())
            rounded = max(1024, round(raw/1024)*1024)
            self.ctx_size.set(rounded)
            self._sync_ctx_display(rounded)
        except ValueError:
            messagebox.showerror("Input Error", "Context size must be a valid number.")
            self._sync_ctx_display(self.ctx_size.get())

    def _sync_ctx_display(self, value):
        # This function remains unchanged
        formatted_value = f"{value:,}"
        str_value = str(value)
        self.ctx_label.config(text=formatted_value)
        if self.ctx_entry.get() != str_value:
            self.ctx_entry.delete(0, tk.END)
            self.ctx_entry.insert(0, str_value)

    def _browse_dir(self, var_to_set):
        # This function remains unchanged
        if var_to_set == self.llama_cpp_dir:
            setting_key = "last_llama_cpp_dir"
            title = "Select LLaMa.cpp Root Directory"
        elif var_to_set == self.venv_dir:
             setting_key = "last_venv_dir"
             title = "Select Virtual Environment Directory"
        else:
             setting_key = None
             title = "Select Directory"

        initial = self.app_settings.get(setting_key) if setting_key else None
        if initial and not Path(initial).is_dir(): initial = None

        d = filedialog.askdirectory(initialdir=initial, title=title)
        if d:
            var_to_set.set(d)
            if setting_key:
                self.app_settings[setting_key] = d

    def _find_server_executable(self, base_dir: Path) -> Path | None:
        # This function remains unchanged
        exe_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
        search_paths = [
            Path("build/bin/Release"), Path("build/bin/Debug"), Path("build/bin"),
            Path("build/Release"), Path("build/Debug"), Path("build"), Path("."),
            Path("bin/Release"), Path("bin/Debug"), Path("bin"),
        ]
        for rel_path in search_paths:
            potential_path = (base_dir / rel_path / exe_name).resolve()
            if potential_path.is_file():
                return potential_path
        return None

    # ═════════════════════════════════════════════════════════════════
    #  configuration (save / load / delete)
    # ═════════════════════════════════════════════════════════════════
    def _current_cfg(self):
        # This function remains unchanged
        return {
            "llama_cpp_dir": self.llama_cpp_dir.get(),
            "venv_dir":      self.venv_dir.get(),
            "model_path":    self.model_path.get(),
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
            "gpu_indices":   [i for i, v in enumerate(self.gpu_vars) if v.get()],
            "mlock":         self.mlock.get(),
            "no_kv_offload": self.no_kv_offload.get(),
            "host":          self.host.get(),
            "port":          self.port.get(),
        }

    def _save_configuration(self):
        # This function needs no changes related to listbox vs combobox
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

        # Handle model path: Set internal path and try to select in listbox
        loaded_model_path_str = cfg.get("model_path", "")
        self.model_path.set(loaded_model_path_str) # Set internal path first
        self.model_listbox.selection_clear(0, tk.END) # Clear current selection first

        selected_idx = -1
        print(f"DEBUG: Loading config, trying to find model path: '{loaded_model_path_str}'")
        if loaded_model_path_str:
            try:
                loaded_model_path = Path(loaded_model_path_str)
                # Find display name matching the full path
                found_display_name = None
                for display_name, full_path in self.found_models.items():
                    if full_path == loaded_model_path:
                        found_display_name = display_name
                        print(f"DEBUG: Load Config - Found display name: {found_display_name}")
                        break

                # Find the index of the display name in the listbox
                if found_display_name:
                    listbox_items = self.model_listbox.get(0, tk.END) # Get current items
                    try:
                        selected_idx = listbox_items.index(found_display_name)
                        print(f"DEBUG: Load Config - Found listbox index: {selected_idx}")
                    except ValueError:
                        # Model name from config not in the current listbox items
                        print(f"WARNING: Loaded model '{found_display_name}' not found in current list.")
                        # Optionally display a message or placeholder?
                        self.model_path.set("") # Clear the internal path if not found in list
                        messagebox.showwarning("Model Not Found", f"The model specified in the saved configuration ('{found_display_name}') was not found during the last scan.\nPlease re-scan or select a different model.")


            except Exception as e: # Handle potential invalid path string
                 print(f"ERROR: Error processing loaded model path '{loaded_model_path_str}': {e}")
                 self.model_path.set("") # Clear invalid path

        # Apply selection if index found
        if selected_idx != -1:
             print(f"DEBUG: Load Config - Selecting index {selected_idx}")
             self.model_listbox.selection_set(selected_idx)
             self.model_listbox.see(selected_idx)
             self.model_listbox.activate(selected_idx)
             # Ensure internal path is synced with the selected item's path
             # This is redundant if self.model_path was set correctly earlier, but safe
             correct_path = self.found_models[self.model_listbox.get(selected_idx)]
             self.model_path.set(str(correct_path))
        else:
             # If model wasn't found or path was empty, ensure internal path is cleared
             print("DEBUG: Load Config - Model not found or path empty, clearing internal model path.")
             self.model_path.set("")
             # We could potentially display a status msg here if loaded_model_path_str was set but not found

        # Apply remaining settings (unchanged logic)
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
        self._sync_ctx_display(ctx)
        self.seed.set(cfg.get("seed","-1"))
        self.flash_attn.set(cfg.get("flash_attn",False))
        self.tensor_split.set(cfg.get("tensor_split",""))
        self.main_gpu.set(cfg.get("main_gpu","0"))
        gpu_indices = cfg.get("gpu_indices", [])
        max_needed_gpus = (max(gpu_indices) + 1) if gpu_indices else 0
        current_gpu_count = self.gpu_count_var.get()
        if max_needed_gpus > current_gpu_count:
             self.gpu_count_var.set(max_needed_gpus)
             self._update_gpu_checkboxes()
        for i, v in enumerate(self.gpu_vars): v.set(i in gpu_indices)
        self.mlock.set(cfg.get("mlock",False))
        self.no_kv_offload.set(cfg.get("no_kv_offload",False))
        self.host.set(cfg.get("host","127.0.0.1"))
        self.port.set(cfg.get("port","8080"))
        self.config_name.set(name)
        messagebox.showinfo("Loaded", f"Configuration '{name}' applied.")

    def _delete_configuration(self):
        # This function needs no changes related to listbox vs combobox
        if not self.config_listbox.curselection():
            return messagebox.showerror("Error","Select a configuration to delete.")
        name = self.config_listbox.get(self.config_listbox.curselection())
        if messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete the configuration '{name}'?"):
            self.saved_configs.pop(name, None)
            self._save_configs() # Persist deletion
            self._update_config_listbox()
            messagebox.showinfo("Deleted", f"Configuration '{name}' deleted.")

    def _update_config_listbox(self):
        # This function needs no changes related to listbox vs combobox
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
        # --- Executable Path ---
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
                Path("build/bin/Release"), Path("build/bin"), Path("build"), Path(".")
            ]])
            exe_base_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
            messagebox.showerror("Executable Not Found",
                                 f"Could not find '{exe_base_name}' within:\n{llama_base_dir}\n\n"
                                 f"Searched in common relative locations like:\n - {search_locs_str}\n\n"
                                 "Please ensure llama.cpp is built and the directory is correct.")
            return None

        cmd = [str(exe_path)]

        # --- Model Path (Reads from self.model_path, updated by listbox selection) ---
        model_full_path_str = self.model_path.get()
        if not model_full_path_str:
            messagebox.showerror("Error", "No model selected. Please scan and select a model from the list.")
            return None
        try:
            model_full_path = Path(model_full_path_str)
            if not model_full_path.is_file():
                # Try to find the display name for better error message
                selected_name = ""
                sel = self.model_listbox.curselection()
                if sel: selected_name = self.model_listbox.get(sel[0])
                error_msg = f"Selected model file not found:\n{model_full_path_str}"
                if selected_name: error_msg += f"\n(Selected in GUI: {selected_name})"
                error_msg += "\n\nPlease re-scan models or check file existence."
                messagebox.showerror("Error", error_msg)
                return None
        except Exception:
             messagebox.showerror("Error", f"Invalid model path selected:\n{model_full_path_str}\n\nPlease re-scan models.")
             return None

        cmd.extend(["-m", model_full_path_str])

        # --- Other Arguments (Unchanged logic) ---
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
        cmd.extend(["--ctx-size", str(self.ctx_size.get())])
        self._add_arg(cmd, "--seed", self.seed.get(), "-1")
        self._add_arg(cmd, "--flash-attn", self.flash_attn.get())
        self._add_arg(cmd, "--tensor-split", self.tensor_split.get().strip(), "")
        self._add_arg(cmd, "--main-gpu", self.main_gpu.get(), "0")
        checked_gpu_indices = [i for i, v in enumerate(self.gpu_vars) if v.get()]
        if checked_gpu_indices:
            # llama.cpp >= b2779 uses --gpu-layers, let's assume older for now
            # Format is -ngl X [-ts S,..] -mg G
            # Or new format -ngl X [-ts S,..] --device D,...
            # Let's stick to the older --main-gpu for now unless user specifically targets
            # a newer build needing --device. The build command is flexible.
            # TODO: Maybe add a checkbox for "--device" format? For now, keep main-gpu.
            pass # Main GPU handled by --main-gpu, tensor split handles distribution

        self._add_arg(cmd, "--host", self.host.get(), "127.0.0.1")
        self._add_arg(cmd, "--port", self.port.get(), "8080")

        return cmd

    def _add_arg(self, cmd_list, arg_name, value, default_value=None):
        # This helper function remains unchanged
        is_bool_var = isinstance(value, tk.BooleanVar)
        is_bool_py = isinstance(value, bool)
        is_string = isinstance(value, str)

        if is_bool_var:
            if value.get(): cmd_list.append(arg_name)
        elif is_bool_py:
            if value: cmd_list.append(arg_name)
        elif is_string:
            actual_value = value.strip()
            if actual_value:
                if default_value is None or actual_value != default_value:
                    cmd_list.extend([arg_name, actual_value])

    # ═════════════════════════════════════════════════════════════════
    #  launch & script helpers
    # ═════════════════════════════════════════════════════════════════
    def launch_server(self):
        # This function remains unchanged
        cmd_list = self._build_cmd()
        if not cmd_list: return
        venv_path_str = self.venv_dir.get()
        final_cmd_str = subprocess.list2cmdline(cmd_list)
        try:
            if venv_path_str and sys.platform == "win32":
                venv_path = Path(venv_path_str)
                act_script = venv_path / "Scripts" / "activate.bat"
                if not act_script.is_file():
                    messagebox.showerror("Error", f"Venv activation script not found:\n{act_script}")
                    return
                subprocess.Popen(f'start "LLaMa.cpp Server" cmd /k ""{act_script}" && {final_cmd_str}"', shell=True)
            elif venv_path_str: # Linux/macOS venv
                venv_path = Path(venv_path_str)
                act_script = venv_path / "bin" / "activate"
                if not act_script.is_file():
                    messagebox.showerror("Error", f"Venv activation script not found:\n{act_script}")
                    return
                # Try common terminals, fallback to basic xterm if needed
                terminals = ['gnome-terminal', 'konsole', 'xfce4-terminal', 'xterm']
                term_cmd_pattern = {
                    'gnome-terminal': " -- bash -c '{command} ; exec bash'", # Keeps window open
                    'konsole':        " -e bash -c '{command} ; exec bash'",
                    'xfce4-terminal': " -e 'bash -c \"{command} ; exec bash\"'",
                    'xterm':          " -e bash -c '{command} ; read -p \"Press Enter to close...\"'" # xterm needs explicit pause
                }
                launch_command = f'echo "Activating venv: {venv_path}" && source "{act_script}" && echo "Launching server..." && {final_cmd_str}'
                launched = False
                for term in terminals:
                    if subprocess.call(['which', term], stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0:
                         term_cmd = term + term_cmd_pattern[term].format(command=launch_command.replace("'", "'\\''")) # Basic escaping for single quotes in command
                         print(f"DEBUG: Using terminal command: {term_cmd}")
                         try:
                            subprocess.Popen(term_cmd, shell=True)
                            launched = True
                            break
                         except Exception as term_err:
                             print(f"DEBUG: Failed to launch with {term}: {term_err}")
                if not launched:
                    messagebox.showerror("Launch Error", "Could not find a supported terminal (gnome-terminal, konsole, xfce4-terminal, xterm) to launch the script.")

            else: # No venv
                if sys.platform == "win32":
                    subprocess.Popen(f'start "LLaMa.cpp Server" {final_cmd_str}', shell=True)
                else:
                    terminals = ['gnome-terminal', 'konsole', 'xfce4-terminal', 'xterm']
                    term_cmd_pattern = {
                        'gnome-terminal': " -- bash -c '{command} ; exec bash'",
                        'konsole':        " -e bash -c '{command} ; exec bash'",
                        'xfce4-terminal': " -e 'bash -c \"{command} ; exec bash\"'",
                        'xterm':          " -e bash -c '{command} ; read -p \"Press Enter to close...\"'"
                    }
                    launch_command = final_cmd_str
                    launched = False
                    for term in terminals:
                        if subprocess.call(['which', term], stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0:
                            term_cmd = term + term_cmd_pattern[term].format(command=launch_command.replace("'", "'\\''"))
                            print(f"DEBUG: Using terminal command: {term_cmd}")
                            try:
                                subprocess.Popen(term_cmd, shell=True)
                                launched = True
                                break
                            except Exception as term_err:
                                print(f"DEBUG: Failed to launch with {term}: {term_err}")
                    if not launched:
                         messagebox.showerror("Launch Error", "Could not find a supported terminal (gnome-terminal, konsole, xfce4-terminal, xterm) to launch the server.")

        except Exception as exc:
            messagebox.showerror("Launch Error", f"Failed to launch server process:\n{exc}")
            traceback.print_exc()

    def save_ps1_script(self):
        cmd_list = self._build_cmd()
        if not cmd_list: return

        # Get model name from listbox selection for default filename
        selected_model_name = ""
        selection = self.model_listbox.curselection()
        if selection:
             selected_model_name = self.model_listbox.get(selection[0])

        default_name = "launch_llama_server.ps1"
        if selected_model_name:
            model_name_part = re.sub(r'[\\/*?:"<>| ]', '_', selected_model_name) # Sanitize
            model_name_part = model_name_part[:50] # Limit length
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
                             # Use Join-Path for robustness with spaces
                             ps_act_path = f'$(Join-Path "{venv_path.resolve()}" "Scripts" "Activate.ps1")'
                             fh.write(f'Write-Host "Activating virtual environment: {venv}" -ForegroundColor Cyan\n')
                             fh.write(f'try {{ . {ps_act_path} }} catch {{ Write-Error "Failed to activate venv: $($_.Exception.Message)"; exit 1 }}\n\n')
                         else:
                              fh.write(f'Write-Warning "Virtual environment activation script not found at: {act_script}"\n\n')
                    except Exception as path_ex:
                         fh.write(f'Write-Warning "Could not process venv path \'{venv}\': {path_ex}"\n\n')

                fh.write(f'Write-Host "Launching llama-server..." -ForegroundColor Green\n')
                ps_cmd_parts = []
                # Quote the executable path properly for PowerShell
                exe_path_str = cmd_list[0]
                ps_cmd_parts.append(f'& "{exe_path_str}"') # Use call operator & and quotes

                i = 1
                while i < len(cmd_list):
                    arg = cmd_list[i]
                    # Check if the next element exists and *doesn't* start with '-' to determine if it's a value
                    is_value_next = (i + 1 < len(cmd_list)) and not cmd_list[i+1].startswith('-')

                    if is_value_next:
                        val = cmd_list[i+1]
                        # Quote arguments and values properly for PowerShell
                        ps_cmd_parts.append(f'"{arg}"') # Quote the flag/option
                        ps_cmd_parts.append(f'"{val}"') # Quote the value
                        i += 2 # Consumed flag and value
                    else:
                        # It's just a flag
                         ps_cmd_parts.append(f'"{arg}"') # Quote the flag
                         i += 1 # Consumed flag only

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
        self._save_configs() # Save includes the potentially updated listbox height
        self.root.destroy()


# ═════════════════════════════════════════════════════════════════════
#  main
# ═════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    # Apply a theme for a slightly more modern look (optional)
    try:
        style = ttk.Style(root)
        themes = style.theme_names()
        preferred_themes = ['vista', 'xpnative', 'winnative', 'clam', 'alt', 'default']
        used_theme = False
        for theme in preferred_themes:
            if theme in themes:
                try:
                     style.theme_use(theme)
                     used_theme = True
                     break
                except tk.TclError:
                     continue
        if not used_theme:
             print("Could not apply preferred ttk themes.")
    except Exception:
         print("ttk themes not available or failed to apply.")

    app  = LlamaCppLauncher(root)
    root.protocol("WM_DELETE_WINDOW", app.on_exit)
    root.mainloop()