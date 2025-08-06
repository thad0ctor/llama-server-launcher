#!/usr/bin/env python3
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

# Debug logging control
DEBUG_VERBOSE = os.getenv('LLAMA_LAUNCHER_DEBUG', '').lower() in ('1', 'true', 'yes')

def debug_print(message, force=False):
    """Print debug message only if verbose debug is enabled or force=True."""
    if force or DEBUG_VERBOSE:
        print(f"DEBUG: {message}", file=sys.stderr)

# Import the environmental variables module
from env_vars_module import EnvironmentalVariablesManager, EnvironmentalVariablesTab

# Import the about tab module
from about_tab import create_about_tab

# Import the ik_llama configuration tab module
from ik_llama import IkLlamaTab

# Import the launch functionality module
from launch import LaunchManager

# Import the configuration management module
from config import ConfigManager

# Import system helper functions
from system import (
    get_gpu_info_static, get_ram_info_static, get_cpu_info_static,
    analyze_gguf_with_llamacpp_tools, calculate_total_gguf_size,
    parse_gguf_header_simple, analyze_gguf_model_static, SystemInfoManager,
    LLAMA_CPP_PYTHON_AVAILABLE
)


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
        "Llama 2 Chat": "  <<SYS>>\\n{{system_message}}\\n<</SYS>>\\n\\n{{prompt}} ",
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
        self.root.geometry("900x1000")
        self.root.minsize(800, 750)

        # ------------------------------------------------ Internal Data Attributes --
        # Attributes that hold data not directly tied to Tk variables, often
        # populated during setup or used for internal logic.

        # --- System Info Attributes ---
        # These will be populated by SystemInfoManager later.
        self.gpu_info = {"available": False, "device_count": 0, "devices": []}
        self.ram_info = {}
        # Default fallback for CPU cores before fetching system info.
        # These will be updated by SystemInfoManager.
        self.cpu_info = {"logical_cores": 4, "physical_cores": 2}


        # --- Persistence & Configuration ---
        # Data structures for saving and loading application state and configurations.
        self.saved_configs = {}
        self.app_settings = {
            "last_llama_cpp_dir": "",
            "last_ik_llama_dir":  "",
            "last_venv_dir":      "",
            "last_model_path":    "",
            "model_dirs":         [],
            "model_list_height":  8,
            # Save/load selected GPU indices (indices of detected GPUs)
            "selected_gpus":      [],
            # Save/load network settings
            "host":              "127.0.0.1",
            "port":              "8080",
            # Backend selection (llama.cpp or ik_llama)
            "backend_selection":  "llama.cpp",
            # Manual GPU settings for when detection fails
            "manual_gpu_mode":    False,
            "manual_gpu_count":   "1",
            "manual_gpu_vram":    "8.0",
            # Manual model settings for when analysis fails
            "manual_model_mode":  False,
            "manual_model_layers": "32",
            "manual_model_size_gb": "7.0",
        }
        # List to store custom parameters entered by the user (strings)
        self.custom_parameters_list = [] # <-- New attribute for custom parameters

        # --- Environmental Variables Manager ---
        self.env_vars_manager = EnvironmentalVariablesManager()

        # --- ik_llama Tab Manager ---
        self.ik_llama_tab = IkLlamaTab(self)

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


        # --- Initialize configuration manager early (needed for config_path) ---
        self.config_manager = ConfigManager(self)

        # --- Initialize system info manager ---
        self.system_info_manager = SystemInfoManager(self)

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
        self.ik_llama_dir    = tk.StringVar() # Path to the ik_llama directory
        self.current_backend_dir = tk.StringVar() # Currently active directory based on backend selection
        self.venv_dir        = tk.StringVar() # Path to the Python virtual environment
        self.model_path      = tk.StringVar() # Path to the selected model file
        # List variable for model directories (used by Listbox)
        self.model_dirs_listvar = tk.StringVar()
        # Status variable for model scanning
        self.scan_status_var = tk.StringVar(value="Scan models to populate list.")

        # --- Backend Selection ---
        self.backend_selection = tk.StringVar(value=self.app_settings.get("backend_selection", "llama.cpp"))

        # --- Dynamic Labels for Backend-specific Text ---
        self.root_dir_label_text = tk.StringVar(value="LLaMa.cpp Root Directory:")
        self.root_dir_help_text = tk.StringVar(value="Select the main 'llama.cpp' folder. The app will search for the server executable.")

        # --- Basic Server Settings ---
        self.cache_type_k    = tk.StringVar(value="f16") # KV cache type (applies to k & v)
        self.cache_type_v    = tk.StringVar(value="f16") # V cache type
        # Initial default thread values before fetching system info.
        # These will be updated by SystemInfoManager or load_config.
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

        # --- Manual GPU Entry Variables ---
        # For when GPU detection fails but user wants to manually specify GPUs
        self.manual_gpu_mode = tk.BooleanVar(value=False)  # Whether we're in manual GPU mode
        # Changed: Now store individual GPUs as a list of dicts instead of count+vram
        self.manual_gpu_list = []  # List of {"name": str, "vram_gb": float}
        # UI variables for adding new manual GPUs
        self.manual_gpu_name_var = tk.StringVar(value="")
        self.manual_gpu_vram_var = tk.StringVar(value="8.0")

        # Legacy variables for config compatibility (will be migrated)
        self.manual_gpu_count = tk.StringVar(value="1")    # Deprecated but kept for config loading
        self.manual_gpu_vram = tk.StringVar(value="8.0")   # Deprecated but kept for config loading

        # --- Manual Model Entry Variables ---
        # For when model analysis fails but user wants to manually specify model info
        self.manual_model_mode = tk.BooleanVar(value=False)  # Whether we're in manual model mode
        self.manual_model_layers = tk.StringVar(value="32")  # User-specified number of layers
        self.manual_model_size_gb = tk.StringVar(value="7.0")  # User-specified model size (GB)

        # --- Memory & Other Advanced Settings ---
        self.no_mmap         = tk.BooleanVar(value=False) # --no-mmap

        self.prio            = tk.StringVar(value="0") # --prio
        self.mlock           = tk.BooleanVar(value=False) # --mlock
        self.no_kv_offload   = tk.BooleanVar(value=False) # --no-kv-offload
        self.host            = tk.StringVar(value=self.app_settings.get("host", "127.0.0.1")) # --host
        self.port            = tk.StringVar(value=self.app_settings.get("port", "8080")) # --port

        # --- Configuration Management ---
        self.config_name     = tk.StringVar(value="default_config") # Name for saving/loading configs

        # --- New Parameters ---
        self.ignore_eos      = tk.BooleanVar(value=False) # --ignore-eos
        self.n_predict       = tk.StringVar(value="-1")   # --n-predict

        # --- MoE CPU Parameters ---
        self.cpu_moe         = tk.BooleanVar(value=False) # --cpu-moe
        self.n_cpu_moe       = tk.StringVar(value="")     # --n-cpu-moe

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
        self.gpu_detected_status_var = tk.StringVar(value="") # Updates with GPU detection message
        self.gpu_availability_var = tk.StringVar(value="Detecting...") # For GPU availability status

        # --- CPU Info Display Variables ---
        self.cpu_logical_cores_display_var = tk.StringVar(value=str(self.cpu_info.get("logical_cores", "N/A")))
        self.cpu_physical_cores_display_var = tk.StringVar(value=str(self.cpu_info.get("physical_cores", "N/A")))


        # Internal state
        self.model_dirs = []
        self.found_models = {} # {display_name: full_path_obj}
        self.current_model_analysis = {} # Holds the result of the last GGUF analysis
        self.analysis_thread = None
        # detected_gpu_devices is populated by SystemInfoManager
        self.detected_gpu_devices = [] # List of detected GPU info dicts
        # logical_cores and physical_cores are populated by SystemInfoManager
        self.logical_cores = 4 # Fallback
        self.physical_cores = 2 # Fallback

        # --- Debounce Timer for Recommendations ---
        # Timer to prevent excessive calls to _update_recommendations when slider is moved
        self._recommendations_update_timer = None

        # --- Detection Progress Flag ---
        # Flag to prevent multiple simultaneous GPU detection threads
        self._detection_in_progress = False


        # --- System Info Initialization ---
        # Set initial fallback values so UI can initialize immediately
        self.threads.set(str(self.physical_cores))
        self.threads_batch.set(str(self.logical_cores))
        # Set initial recommendation vars with fallback info
        self.recommended_threads_var.set(f"Recommended: {self.physical_cores} (Detecting...)")
        self.recommended_threads_batch_var.set(f"Recommended: {self.logical_cores} (Detecting...)")
        # Set initial GPU availability status
        initial_gpu_count = self.gpu_info.get("device_count", 0)
        if self.gpu_info.get('available', False) and initial_gpu_count > 0:
            self.gpu_availability_var.set(f"CUDA Devices ({initial_gpu_count} available):")
        else:
            self.gpu_availability_var.set("CUDA Devices (Detecting...)")
        # Display initial GPU detection status message
        self.gpu_detected_status_var.set("Detecting GPUs...")


        # load previous settings
        self._load_saved_configs()
        self.llama_cpp_dir.set(self.app_settings.get("last_llama_cpp_dir", ""))
        self.ik_llama_dir.set(self.app_settings.get("last_ik_llama_dir", ""))
        self.venv_dir.set(self.app_settings.get("last_venv_dir", ""))
        # Ensure model_dirs is loaded as list of Paths
        raw_model_dirs = self.app_settings.get("model_dirs", [])
        self.model_dirs = []
        for d in raw_model_dirs:
            if d:
                try:
                    path_obj = Path(d).resolve()
                    if path_obj.exists() and path_obj.is_dir():
                        self.model_dirs.append(path_obj)
                        print(f"DEBUG: Loaded valid model directory: {path_obj}", file=sys.stderr)
                    else:
                        print(f"WARNING: Skipping invalid model directory from config: {d} (resolved to {path_obj})", file=sys.stderr)
                except Exception as e:
                    print(f"ERROR: Failed to process model directory from config '{d}': {e}", file=sys.stderr)
        # Load custom parameters list
        self.custom_parameters_list = self.app_settings.get("custom_parameters", [])
        # Load backend selection
        self.backend_selection.set(self.app_settings.get("backend_selection", "llama.cpp"))

        # Load environmental variables configuration
        self.env_vars_manager.load_from_config(self.app_settings)

        # Load ik_llama configuration
        self.ik_llama_tab.load_from_config(self.app_settings)

        # --- Initialize launch manager ---
        self.launch_manager = LaunchManager(self)

        # Update labels based on loaded backend selection and set current backend directory
        self._update_root_directory_labels()

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
        # Bind trace to n_predict to update default config name if it's currently the generated one
        self.n_predict.trace_add("write", lambda *args: self._update_default_config_name_if_needed())
        # Bind trace to ignore_eos to update default config name if needed
        self.ignore_eos.trace_add("write", lambda *args: self._update_default_config_name_if_needed())
        # Bind trace to MoE CPU parameters to update default config name if needed
        self.cpu_moe.trace_add("write", lambda *args: self._update_default_config_name_if_needed())
        self.n_cpu_moe.trace_add("write", lambda *args: self._update_default_config_name_if_needed())
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

        self.prio.trace_add("write", lambda *args: self._update_default_config_name_if_needed())
        self.mlock.trace_add("write", lambda *args: self._update_default_config_name_if_needed())
        self.no_kv_offload.trace_add("write", lambda *args: self._update_default_config_name_if_needed())
        # Add trace handler for port changes
        self.port.trace_add("write", lambda *args: self._save_configs())
        # Add trace handler for host changes
        self.host.trace_add("write", lambda *args: self._save_configs())
        # Add trace handler for backend selection changes
        self.backend_selection.trace_add("write", lambda *args: self._on_backend_selection_changed())
        # Add trace handler for current backend directory changes
        self.current_backend_dir.trace_add("write", lambda *args: self._on_backend_dir_changed())
        # Add trace handler for virtual environment changes to re-trigger GPU detection
        self.venv_dir.trace_add("write", lambda *args: self._on_venv_dir_changed())

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


        # Initialize GPU system based on saved settings
        if self.manual_gpu_mode.get():
            # Manual mode was loaded from config - setup immediately
            debug_print("Manual GPU mode loaded from config, setting up immediately")
            self.gpu_detected_status_var.set("Manual mode active")
            # Migrate legacy config format to new list format if needed
            self._migrate_legacy_manual_gpu_config()
            self._setup_manual_gpus()
        else:
            # Automatic mode - start background detection
            debug_print("Automatic GPU mode, starting background detection")
            self._start_system_info_detection()

        # Update initial GPU checkbox states and recommendations
        self._update_gpu_checkboxes()

        # If manual model mode was loaded from config, apply it
        if self.manual_model_mode.get():
            self._setup_manual_model()

        self._update_recommendations() # Call initially to set all initial recommendations

        # Perform initial scan (in background) if dirs exist
        if self.model_dirs:
            self.scan_status_var.set("Scanning on startup...")
            scan_thread = Thread(target=self._scan_model_dirs, daemon=True)
            scan_thread.start()
        else:
             self.scan_status_var.set("Add directories and scan for models.")



        # Update chat template display and controls initially based on initial state
        self._update_template_controls_state() # Sets initial state based on self.template_source
        self._update_effective_template_display() # Sets initial displayed template based on source


    # ═════════════════════════════════════════════════════════════════
    #  UI builders
    # ═════════════════════════════════════════════════════════════════
    def _create_widgets(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=10, pady=10)

        # Store notebook reference for tab visibility management
        self.notebook = nb

        main_frame = ttk.Frame(nb); adv_frame = ttk.Frame(nb); cfg_frame = ttk.Frame(nb); chat_frame = ttk.Frame(nb); env_frame = ttk.Frame(nb); ik_llama_frame = ttk.Frame(nb); about_frame = ttk.Frame(nb)
        nb.add(main_frame, text="Main Settings")
        nb.add(adv_frame,  text="Advanced Settings")
        nb.add(chat_frame, text="Chat Template") # Add the new tab
        nb.add(env_frame,  text="Environment Variables") # Add environmental variables tab
        # ik_llama tab will be added conditionally
        self.ik_llama_frame = ik_llama_frame
        nb.add(cfg_frame,  text="Configurations")
        nb.add(about_frame, text="About") # Add the about tab


        self._setup_main_tab(main_frame)
        self._setup_advanced_tab(adv_frame)
        self._setup_chat_template_tab(chat_frame) # Setup the new tab
        self._setup_env_vars_tab(env_frame) # Setup the environmental variables tab
        self._setup_ik_llama_tab(ik_llama_frame) # Setup the ik_llama tab
        self._setup_config_tab(cfg_frame)
        self._setup_about_tab(about_frame) # Setup the about tab

        # Update ik_llama tab visibility based on current backend selection
        self._update_ik_llama_tab_visibility()

        bar = ttk.Frame(self.root); bar.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(bar, text="Launch Server",   command=self.launch_manager.launch_server).pack(side="left",  padx=5)
        ttk.Button(bar, text="Save PS1 Script", command=self.launch_manager.save_ps1_script).pack(side="left", padx=5)
        ttk.Button(bar, text="Save SH Script",  command=self.launch_manager.save_sh_script).pack(side="left", padx=5)
        ttk.Button(bar, text="Exit",            command=self.on_exit).pack(side="right", padx=5)
        ttk.Button(bar, text="Save Config",     command=self._save_configuration).pack(side="right", padx=5)

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

        # --- Backend Selection ---
        ttk.Label(inner, text="Backend Selection", font=("TkDefaultFont", 12, "bold"))\
            .grid(column=0, row=r, sticky="w", padx=10, pady=(10,5)); r += 1

        backend_frame = ttk.Frame(inner)
        backend_frame.grid(column=0, row=r, columnspan=4, sticky="w", padx=10, pady=(0,10))

        ttk.Radiobutton(backend_frame, text="llama.cpp", variable=self.backend_selection, value="llama.cpp")\
            .pack(side="left", padx=(0, 15))
        ttk.Radiobutton(backend_frame, text="ik_llama", variable=self.backend_selection, value="ik_llama")\
            .pack(side="left")

        r += 1
        ttk.Separator(inner, orient="horizontal")\
            .grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=(0,15)); r += 1

        ttk.Label(inner, text="Directories & Model", font=("TkDefaultFont", 12, "bold"))\
            .grid(column=0, row=r, sticky="w", padx=10, pady=(10,5)); r += 1
        ttk.Separator(inner, orient="horizontal")\
            .grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=5); r += 1

        ttk.Label(inner, textvariable=self.root_dir_label_text)\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        ttk.Entry(inner, textvariable=self.current_backend_dir, width=50)\
            .grid(column=1, row=r, columnspan=2, sticky="ew", padx=5, pady=3)
        ttk.Button(inner, text="Browse…", command=lambda: self._browse_backend_dir())\
            .grid(column=3, row=r, padx=5, pady=3)
        ttk.Label(inner, textvariable=self.root_dir_help_text, font=("TkSmallCaptionFont"))\
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
        dir_sb.pack(side=tk.RIGHT, fill=tk.Y)
        r += 1 # Advance row count after adding the listbox frame

        inner.rowconfigure(r-1, weight=0) # Don't expand the directory listbox row with height

        # Move directory buttons to the right of the listbox
        dir_btn_frame = ttk.Frame(inner)
        dir_btn_frame.grid(column=2, row=r-1, sticky="n", padx=5, pady=3)
        ttk.Button(dir_btn_frame, text="Add Dir…", width=10, command=self._add_model_dir)\
           .pack(side=tk.TOP, pady=2, fill=tk.X)
        ttk.Button(dir_btn_frame, text="Remove Dir", width=10, command=self._remove_model_dir)\
           .pack(side=tk.TOP, pady=2, fill=tk.X)

        # Add scan button next to directory buttons
        scan_btn = ttk.Button(dir_btn_frame, text="Scan Models", command=self._trigger_scan)
        scan_btn.pack(side=tk.TOP, pady=2, fill=tk.X)

        # Add some vertical space between directory section and model selection
        r += 1
        ttk.Separator(inner, orient="horizontal")\
            .grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=10); r += 1

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

        # Remove the old scan button position since we moved it
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

    def _override_ctx_size(self, event=None):
        """Manually set context size from entry."""
        try:
            value = int(self.ctx_entry.get())
            # Allow any positive value, not just clamped to slider range
            if value < 1024:
                value = 1024  # Minimum reasonable context size
            self.ctx_size.set(value)
            self._sync_ctx_display(value) # Update entry/label to value
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
            .grid(column=1, row=r, sticky="w", padx=5, pady=3, columnspan=2)

        # Add Analyze Model button
        self.analyze_model_button = ttk.Button(inner, text="Analyze Model", command=self._force_analyze_model)
        self.analyze_model_button.grid(column=3, row=r, sticky="w", padx=5, pady=3)
        r += 1


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
            ttk.Label(inner, textvariable=self.cpu_logical_cores_display_var, font="TkSmallCaptionFont")\
                .grid(column=1, row=r, sticky="w", padx=5, pady=(5,0), columnspan=3)
            r += 1
            ttk.Label(inner, text="System CPU (Physical Cores):", font=("TkSmallCaptionFont", 8, ("bold",)))\
                .grid(column=0, row=r, sticky="w", padx=10, pady=(0,10))
            ttk.Label(inner, textvariable=self.cpu_physical_cores_display_var, font="TkSmallCaptionFont")\
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
        ttk.Label(inner, textvariable=self.gpu_availability_var)\
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

        # --- Manual Model Entry (when analysis fails) ---
        self.manual_model_frame = ttk.Frame(inner)
        self.manual_model_frame.grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=5)

        # Create manual model widgets (always create them, but control visibility)
        self.manual_model_label = ttk.Label(self.manual_model_frame, text="Model analysis unavailable. Manual entry:", foreground="orange")
        self.manual_model_label.pack(side="left", padx=5)

        # Layers entry
        ttk.Label(self.manual_model_frame, text="Layers:").pack(side="left", padx=(10, 2))
        self.manual_layers_entry = ttk.Entry(self.manual_model_frame, textvariable=self.manual_model_layers, width=5)
        self.manual_layers_entry.pack(side="left", padx=2)

        # Size entry
        ttk.Label(self.manual_model_frame, text="Size (GB):").pack(side="left", padx=(5, 2))
        self.manual_size_entry = ttk.Entry(self.manual_model_frame, textvariable=self.manual_model_size_gb, width=6)
        self.manual_size_entry.pack(side="left", padx=2)

        # Apply button
        self.manual_apply_btn = ttk.Button(self.manual_model_frame, text="Apply", command=self._apply_manual_model_settings)
        self.manual_apply_btn.pack(side="left", padx=(5, 2))

        # Toggle checkbox for manual mode
        self.manual_model_cb = ttk.Checkbutton(self.manual_model_frame, text="Use manual model info",
                                         variable=self.manual_model_mode,
                                         command=self._toggle_manual_model_mode)
        self.manual_model_cb.pack(side="left", padx=(10, 0))

        # Initially hide the manual model frame (will be shown/hidden dynamically)
        self._update_manual_model_visibility()

        r += 1

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
            .grid(column=2, row=r, columnspan=2, sticky="w", padx=5, pady=3); r += 1


        # --- Memory & Cache settings ---
        ttk.Label(inner, text="Memory & Cache", font=("TkDefaultFont", 12, "bold"))\
            .grid(column=0, row=r, sticky="w", padx=10, pady=(20,5)); r += 1
        ttk.Separator(inner, orient="horizontal")\
            .grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=5); r += 1

        ttk.Label(inner, text="KV Cache Type (--cache-type-k):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        # Combobox state is "readonly" for selection, but not disabled
        self.cache_type_k_combo = ttk.Combobox(inner, textvariable=self.cache_type_k, width=10, values=("f16","f32","q8_0","q4_0","q4_1","q5_0","q5_1","q6_k"), state="readonly")
        self.cache_type_k_combo.grid(column=1, row=r, sticky="w", padx=5, pady=3); r += 1
        ttk.Label(inner, text="Quantization for KV cache (f16 is default, lower Q=more memory saved)", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r-1, columnspan=2, sticky="w", padx=5, pady=3);

        ttk.Label(inner, text="V Cache Type (--cache-type-v):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        # Combobox state is "readonly" for selection, but not disabled
        self.cache_type_v_combo = ttk.Combobox(inner, textvariable=self.cache_type_v, width=10, values=("f16","f32","q8_0","q4_0","q4_1","q5_0","q5_1","q6_k"), state="readonly")
        self.cache_type_v_combo.grid(column=1, row=r, sticky="w", padx=5, pady=3); r += 1
        ttk.Label(inner, text="Quantization for V cache (f16 is default, lower Q=more memory saved)", font=("TkSmallCaptionFont"))\
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
            .grid(column=1, row=r, columnspan=3, sticky="w", padx=5, pady=(0,3)); r += 1



        # --- Scheduling Priority --- (Already existed, just moved)
        ttk.Label(inner, text="Scheduling Priority (--prio):")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        # Combobox state is "readonly"
        self.prio_combo = ttk.Combobox(inner, textvariable=self.prio, width=10, values=("0","1","2","3"), state="readonly")
        self.prio_combo.grid(column=1, row=r, sticky="w", padx=5, pady=3); r += 1
        ttk.Label(inner, text="0=Normal, 1=Medium, 2=High, 3=Realtime (OS dependent)", font=("TkSmallCaptionFont"))\
            .grid(column=2, row=r-1, columnspan=2, sticky="w", padx=5, pady=3); # Re-grid label

        # --- MoE CPU Settings --- (same row)
        ttk.Label(inner, text="MoE CPU Settings:")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=3)
        moe_frame = ttk.Frame(inner)
        moe_frame.grid(column=1, row=r, columnspan=3, sticky="w", padx=5, pady=3)

        self.cpu_moe_check = ttk.Checkbutton(moe_frame, text="Keep all MoE in CPU (--cpu-moe)", variable=self.cpu_moe, state=tk.NORMAL)
        self.cpu_moe_check.pack(side="left", padx=(0, 10))

        ttk.Label(moe_frame, text="First N layers in CPU (--n-cpu-moe):")\
            .pack(side="left", padx=(0, 5))
        self.n_cpu_moe_entry = ttk.Entry(moe_frame, textvariable=self.n_cpu_moe, width=8, state=tk.NORMAL)
        self.n_cpu_moe_entry.pack(side="left")
        r += 1


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
        canvas = tk.Canvas(parent, highlightthickness=0); vs = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner  = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(yscrollcommand=vs.set,
                                                             scrollregion=canvas.bbox("all")))
        canvas_window = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))
        canvas.pack(side="left", fill="both", expand=True); vs.pack(side="right", fill="y")

        # Configuration Management
        ttk.Label(inner, text="Configuration Management", font=("TkDefaultFont", 12, "bold")).pack(anchor="w", padx=10, pady=(10,5))
        ttk.Separator(inner, orient="horizontal").pack(fill="x", padx=10, pady=5)

        # Config name entry
        config_frame = ttk.Frame(inner); config_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(config_frame, text="Configuration Name:").pack(side="left", padx=(0,5))
        ttk.Entry(config_frame, textvariable=self.config_name).pack(side="left", fill="x", expand=True, padx=5)

        # Config buttons
        btn_frame = ttk.Frame(inner); btn_frame.pack(fill="x", padx=10, pady=5)
        ttk.Button(btn_frame, text="Save Configuration", command=self._save_configuration).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Load Configuration", command=self._load_configuration).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Delete Configuration", command=self._delete_configuration).pack(side="left", padx=5)

        # Import/Export buttons
        import_export_frame = ttk.Frame(inner); import_export_frame.pack(fill="x", padx=10, pady=5)
        ttk.Button(import_export_frame, text="Export Selected", command=self._export_configurations).pack(side="left", padx=5)
        ttk.Button(import_export_frame, text="Import from File", command=self._import_configurations).pack(side="left", padx=5)

        # Instructions for multi-select
        instructions_frame = ttk.Frame(inner); instructions_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(instructions_frame, text="Tip: Hold Ctrl to select multiple configurations for export",
                 font=("TkSmallCaptionFont", 8), foreground="gray").pack(side="left")

        # Config list
        ttk.Label(inner, text="Saved Configurations:", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", padx=10, pady=(15,5))
        list_frame = ttk.Frame(inner); list_frame.pack(fill="both", expand=True, padx=10, pady=5)
        list_sb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.config_listbox = tk.Listbox(list_frame, yscrollcommand=list_sb.set, height=12, exportselection=False, selectmode=tk.EXTENDED)
        list_sb.config(command=self.config_listbox.yview)
        self.config_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.config_listbox.bind("<<ListboxSelect>>", lambda e: self._on_config_selected())

        self._update_config_listbox()

    def _setup_env_vars_tab(self, parent):
        """Set up the Environmental Variables tab using the EnvironmentalVariablesTab class."""
        # Create the environmental variables tab using the dedicated class
        self.env_vars_tab = EnvironmentalVariablesTab(parent, self.env_vars_manager)

    def _setup_ik_llama_tab(self, parent):
        """Set up the ik_llama configuration tab using the IkLlamaTab class."""
        # Create the ik_llama tab using the dedicated class
        self.ik_llama_tab.create_tab(parent)

    def _setup_about_tab(self, parent):
        """Set up the About tab using the AboutTab class."""
        # Create the about tab using the dedicated class
        self.about_tab = create_about_tab()
        self.about_tab.setup_about_tab(parent)

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
            try:
                p = Path(directory).resolve() # Resolve path to handle symlinks etc.
                # Validate that the resolved path actually exists and is a directory
                if not p.exists():
                    messagebox.showerror("Error", f"Selected directory does not exist:\n{p}")
                    return
                if not p.is_dir():
                    messagebox.showerror("Error", f"Selected path is not a directory:\n{p}")
                    return

                # Add the resolved path string to avoid issues with comparison later
                p_str = str(p)
                print(f"DEBUG: Adding model directory - Original: '{directory}', Resolved: '{p_str}'", file=sys.stderr)

                if p_str not in [str(x) for x in self.model_dirs]: # Compare resolved paths
                    self.model_dirs.append(p) # Store as Path object
                    self._update_model_dirs_listbox()
                    self._save_configs()
                    self._trigger_scan()
                else:
                    messagebox.showinfo("Info", "Directory already in list.")
            except Exception as e:
                messagebox.showerror("Error", f"Error processing directory path:\n{directory}\n\nError: {e}")
                print(f"ERROR: Failed to process directory '{directory}': {e}", file=sys.stderr)

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
        # Pattern to match multi-part files
        # model_name-BF16-00001-of-00005.gguf from bartowski/unsloth
        # model_name.Q6_K.gguf.part01of12 from mradermacher
        re_multi1 = re.compile(r"^(.*?)-\d{5}-of-\d{5}\.gguf$", re.I)
        re_multi2 = re.compile(r"^(.*?)\.gguf\.part\d+of\d+$", re.I)
        # Pattern to match the FIRST part of a multi-part files
        re_first1 = re.compile(r"^(.*?)-00001-of-\d{5}\.gguf$", re.I)
        re_first2 = re.compile(r"^(.*?)\.gguf\.part0*1of\d+$", re.I)

        # Two-pass approach to handle multi-part files correctly
        all_gguf_files = []
        for model_dir in self.model_dirs:
            # Skip invalid or non-existent directories silently during scan
            if not isinstance(model_dir, Path) or not model_dir.is_dir(): continue
            print(f"DEBUG: Scanning directory: {model_dir}", file=sys.stderr)
            try:
                # Collect all GGUF files first
                for gguf_path in model_dir.rglob('*.gguf'):
                    if not gguf_path.is_file(): continue
                    filename = gguf_path.name
                    # Skip non-model GGUF files often found with models
                    if "mmproj" in filename.lower() or filename.lower().endswith(".bin.gguf"):
                         continue
                    all_gguf_files.append(gguf_path)
            except Exception as e:
                print(f"ERROR: Error scanning directory {model_dir}: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

        # First pass: find all first parts of multi-part files
        processed_multipart_bases = set()
        for gguf_path in all_gguf_files:
            filename = gguf_path.name
            first_part_match = re_first1.match(filename) or re_first2.match(filename)
            if first_part_match:
                base_name = first_part_match.group(1)
                if base_name not in processed_multipart_bases:
                    try:
                        resolved_gguf_path = gguf_path.resolve()
                        found[base_name] = resolved_gguf_path
                        processed_multipart_bases.add(base_name)
                        print(f"DEBUG: Found multi-part model: {base_name}", file=sys.stderr)
                    except Exception as resolve_exc:
                        print(f"Warning: Could not resolve path '{gguf_path}' during scan: {resolve_exc}", file=sys.stderr)

        # Second pass: find single-part files that aren't part of multi-part sets
        for gguf_path in all_gguf_files:
            filename = gguf_path.name

            # Skip if this is any part of a multi-part file
            if re_multi1.match(filename) or re_multi2.match(filename):
                continue

            # Handle single-part files
            if filename.lower().endswith(".gguf"):
                display_name = gguf_path.stem
                if display_name not in processed_multipart_bases and display_name not in found:
                    try:
                        resolved_gguf_path = gguf_path.resolve()
                        found[display_name] = resolved_gguf_path
                        print(f"DEBUG: Found single-part model: {display_name}", file=sys.stderr)
                    except Exception as resolve_exc:
                        print(f"Warning: Could not resolve path '{gguf_path}' during scan: {resolve_exc}", file=sys.stderr)

        print(f"DEBUG: Scan completed, found {len(found)} models", file=sys.stderr)
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
            self._update_manual_model_visibility() # Update manual model section visibility
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
                 # Analysis not available - check what options we have
                 backend = self.backend_selection.get()
                 if backend == "ik_llama":
                     backend_dir = self.ik_llama_dir.get().strip()
                     backend_name = "ik_llama"
                 else:
                     backend_dir = self.llama_cpp_dir.get().strip()
                     backend_name = "llama.cpp"

                 if backend_dir:
                     self.gpu_layers_status_var.set(f"Analyzing model using {backend_name} tools...")
                     # Try the analysis even without llama-cpp-python
                     self.analysis_thread = Thread(target=self._run_gguf_analysis, args=(full_path_str,), daemon=True)
                     self.analysis_thread.start()
                 else:
                     self.gpu_layers_status_var.set(f"Analysis available: Set {backend_name} directory or install llama-cpp-python")
                     self._reset_gpu_layer_controls(keep_entry_enabled=True) # Keep entry enabled if lib missing
                     self._reset_model_info_display()
                     self.model_architecture_var.set("Analysis Unavailable")
                     self.model_filesize_var.set("Analysis Unavailable")
                     self.model_total_layers_var.set("Analysis Unavailable")
                     self.current_model_analysis = {}
                     self._update_recommendations() # Update recommendations based on no analysis
                     self._generate_default_config_name() # Generate default name even without analysis
                     self._update_manual_model_visibility() # Update manual model section visibility


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
            self._update_manual_model_visibility() # Update manual model section visibility


    def _run_gguf_analysis(self, model_path_str):
        """Worker function for background GGUF analysis."""
        print(f"Analyzing GGUF in background: {model_path_str}", file=sys.stderr)
        # Check if the currently selected model in the GUI still matches the one being analyzed
        # This prevents updating the UI with stale results if the user quickly selects another model
        if self.model_path.get() == model_path_str:
             # Try backend-specific tools first, fall back to llama-cpp-python if available
             backend = self.backend_selection.get()
             if backend == "ik_llama":
                 backend_dir = self.ik_llama_dir.get().strip()
                 backend_name = "ik_llama"
             else:
                 backend_dir = self.llama_cpp_dir.get().strip()
                 backend_name = "llama.cpp"

             if backend_dir:
                 print(f"DEBUG: Trying {backend_name} tools from: {backend_dir}", file=sys.stderr)
                 analysis_result = analyze_gguf_with_llamacpp_tools(model_path_str, backend_dir)

                 # If backend tools failed and we have llama-cpp-python available, try that as fallback
                 if analysis_result.get("error") and LLAMA_CPP_PYTHON_AVAILABLE:
                     print(f"DEBUG: {backend_name} tools failed, falling back to llama-cpp-python", file=sys.stderr)
                     analysis_result = analyze_gguf_model_static(model_path_str)
             else:
                 # No backend directory set, try simple GGUF parser first
                 print(f"DEBUG: No {backend_name} directory set, trying simple GGUF parser", file=sys.stderr)
                 analysis_result = parse_gguf_header_simple(model_path_str)

                 # If simple parser failed and we have llama-cpp-python available, try that as fallback
                 if analysis_result.get("error") and LLAMA_CPP_PYTHON_AVAILABLE:
                     print("DEBUG: Simple GGUF parser failed, falling back to llama-cpp-python", file=sys.stderr)
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
    def _update_manual_model_visibility(self):
        """Show or hide the manual model entry section based on analysis availability."""
        # Check if model analysis failed or no model loaded
        analysis_failed = (not self.current_model_analysis or
                          self.current_model_analysis.get("error") or
                          self.current_model_analysis.get("n_layers") is None or
                          not self.model_path.get())

        # Also check if we're in manual mode (should always show in manual mode)
        show_manual = analysis_failed or self.manual_model_mode.get()

        if hasattr(self, 'manual_model_frame') and self.manual_model_frame.winfo_exists():
            if show_manual:
                self.manual_model_frame.grid()  # Show the frame
            else:
                self.manual_model_frame.grid_remove()  # Hide the frame but keep its grid position

    def _force_analyze_model(self):
        """Forces analysis of the currently selected model."""
        model_path_str = self.model_path.get().strip()
        if not model_path_str:
            print("DEBUG: No model selected for analysis", file=sys.stderr)
            return

        print(f"DEBUG: Force analyzing model: {model_path_str}", file=sys.stderr)

        # Update button to show analysis is in progress
        if hasattr(self, 'analyze_model_button') and self.analyze_model_button.winfo_exists():
            self.analyze_model_button.config(text="Analyzing...", state=tk.DISABLED)

        # Update status
        self.gpu_layers_status_var.set("Force analyzing model...")
        self._reset_model_info_display()
        self.current_model_analysis = {}
        self._update_recommendations()

        # Cancel any existing analysis
        if self.analysis_thread and self.analysis_thread.is_alive():
            print("DEBUG: Cancelling previous analysis thread for force analysis.", file=sys.stderr)

        # Start new analysis thread
        self.analysis_thread = Thread(target=self._run_gguf_analysis, args=(model_path_str,), daemon=True)
        self.analysis_thread.start()

    def _update_ui_after_analysis(self, analysis_result):
        """Updates controls based on GGUF analysis results (runs in main thread)."""
        print("DEBUG: _update_ui_after_analysis running", file=sys.stderr)

        # Re-enable the analyze button
        if hasattr(self, 'analyze_model_button') and self.analyze_model_button.winfo_exists():
            self.analyze_model_button.config(text="Analyze Model", state=tk.NORMAL)

        # Crucially, check if the analysis result is for the *currently selected* model
        if self.model_path.get() != analysis_result.get("path"):
             print("DEBUG: _update_ui_after_analysis received result for a different model. Ignoring.", file=sys.stderr)
             return # Ignore stale results

        # If we're in manual model mode, don't override with analysis results
        if self.manual_model_mode.get():
            print("DEBUG: Manual model mode is active, skipping automatic analysis update.", file=sys.stderr)
            return

        self.current_model_analysis = analysis_result
        error = analysis_result.get("error")
        n_layers = analysis_result.get("n_layers")
        message = analysis_result.get("message")

        # --- Update Model Info Display ---
        arch = analysis_result.get("architecture", "unknown")
        file_size_gb = analysis_result.get("file_size_gb")
        shard_count = analysis_result.get("shard_count", 1)

        self.model_architecture_var.set(arch.capitalize() if arch and arch != "unknown" else "Unknown")

        # Show file size with shard info if applicable
        if file_size_gb is not None and file_size_gb > 0:
            if shard_count > 1:
                self.model_filesize_var.set(f"{file_size_gb:.2f} GB ({shard_count} shards)")
            else:
                self.model_filesize_var.set(f"{file_size_gb:.2f} GB")
        else:
            self.model_filesize_var.set("Unknown Size")

        self.model_total_layers_var.set(str(n_layers) if n_layers is not None else "Unknown")

        # --- Update GPU Layer Controls ---
        # Always ensure the entry is NORMAL here, regardless of error state.
        if hasattr(self, 'n_gpu_layers_entry') and self.n_gpu_layers_entry.winfo_exists():
             self.n_gpu_layers_entry.config(state=tk.NORMAL)

        if error or n_layers is None or n_layers <= 0:
             # Analysis failed or layer count invalid/zero
             status_msg = error if error else (message if message else "Layer count not found or invalid")
             # Show that manual entry is still available
             full_status = f"Analysis Error: {status_msg}" if error else status_msg
             full_status += " (Manual entry available)"
             self.gpu_layers_status_var.set(full_status)
             # Resetting controls sets max_layers to 0 and disables the slider
             self._reset_gpu_layer_controls(keep_entry_enabled=True)
             # IMPORTANT: Don't sync from entry when analysis fails to preserve user's value
             # The entry remains enabled and the user can still manually set GPU layers

        else: # Analysis succeeded, layers found (n_layers > 0)
            self.max_gpu_layers.set(n_layers)
            self.gpu_layers_status_var.set(f"Max Layers: {n_layers}")
            if hasattr(self, 'gpu_layers_slider') and self.gpu_layers_slider.winfo_exists():
                self.gpu_layers_slider.config(to=n_layers, state=tk.NORMAL) # Enable slider

            # If tensor split is not blank, set max layers to the maximum quantity instead of defaulting to 0
            tensor_split_val = self.tensor_split.get().strip()
            if tensor_split_val and self.n_gpu_layers.get().strip() in ["0", ""]:
                # Set to max layers (show actual number instead of -1)
                self.n_gpu_layers.set(str(n_layers))

            # Sync the controls based on the *current* value in the n_gpu_layers StringVar
            # This will set the slider and potentially update the entry format (-1 vs number)
            self._sync_gpu_layers_from_entry()

        # --- Update Recommendations based on new analysis ---
        self._update_recommendations()

        # --- Generate Default Config Name ---
        # Call this after analysis completes, as n_layers is needed for the name
        self._generate_default_config_name()

        # --- Update Manual Model Visibility ---
        # Hide/show manual model section based on analysis success
        self._update_manual_model_visibility()

    def _reset_gpu_layer_controls(self, keep_entry_enabled=False):
         """Resets GPU layer slider state and max layers (but *not* entry StringVar).

         Args:
             keep_entry_enabled (bool): If True, keeps the entry widget enabled even when resetting controls.
         """
         print("DEBUG: _reset_gpu_layer_controls called", file=sys.stderr)
         # Reset max_gpu_layers first
         self.max_gpu_layers.set(0)

         # Update status label
         self.gpu_layers_status_var.set("Select model to see layer info")

         # Slider range and state
         if hasattr(self, 'gpu_layers_slider') and self.gpu_layers_slider.winfo_exists():
              self.gpu_layers_slider.config(to=0, state=tk.DISABLED) # Slider disabled if max_layers is 0

         # Set entry state based on keep_entry_enabled parameter
         if hasattr(self, 'n_gpu_layers_entry') and self.n_gpu_layers_entry.winfo_exists():
              self.n_gpu_layers_entry.config(state=tk.NORMAL if keep_entry_enabled else tk.DISABLED)

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

    def _set_gpu_layers(self, input_value, from_slider=False):
        """
        Helper to set the internal GPU layers state (int) based on user input.
        Handles clamping based on max_layers. Does NOT directly set the StringVar
        but updates the IntVar and triggers recommendations.
        input_value can be -1 or a non-negative integer.
        from_slider: if True, uses debounced recommendations update to prevent excessive calls
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
                # If max_layers <= 0 (analysis failed), preserve the user's input value
                # This allows users to manually set GPU layers when analysis fails
                int_val = input_value
        # else: input_value < -1 (should be prevented by validation) -> default to 0

        # Update the Tk integer variable (linked to slider)
        # This will also indirectly trigger _sync_gpu_layers_from_slider if trace is set,
        # but we manage string sync in the entry/slider callbacks now.
        if self.n_gpu_layers_int.get() != int_val:
            self.n_gpu_layers_int.set(int_val)

        # Trigger recommendations update (e.g., KV cache type display)
        # Use debounced update if called from slider to prevent excessive calls
        if from_slider:
            self._schedule_recommendations_update()
        else:
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
            # Use debounced update to prevent excessive calls when slider is moved rapidly
            self._set_gpu_layers(value, from_slider=True)

            # Determine the canonical string representation for the entry based on the clamped value
            canonical_str = str(value) # Always show the actual integer value

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
            # If max_layers > 0, the entry should show the clamped integer value.
            # If max_layers <= 0, the entry should retain the user's valid input string (current_str).

            canonical_str_for_entry = current_str # Assume user input is fine initially

            if max_layers > 0:
                 # If max_layers > 0, the entry should represent the *clamped* value
                 clamped_int_from_set = self.n_gpu_layers_int.get() # Get the result of _set_gpu_layers
                 canonical_str_for_entry = str(clamped_int_from_set) # Always show the actual integer value
            else:
                 # max_layers <= 0 (analysis failed), preserve user input
                 # The internal int value was set to the user's input by _set_gpu_layers
                 canonical_str_for_entry = current_str

            # Update the entry's StringVar only if it's different from the calculated canonical string
            # This prevents loops and ensures the entry displays the correct format (actual number)
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
            # Always show the actual integer value
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
    #  Manual Model Setup (when analysis fails)
    # ═════════════════════════════════════════════════════════════════
    def _setup_manual_model(self):
        """Set up manual model mode when analysis fails."""
        try:
            layers = int(self.manual_model_layers.get())
            size_gb = float(self.manual_model_size_gb.get())

            if layers <= 0:
                layers = 32
                self.manual_model_layers.set("32")
            if size_gb <= 0:
                size_gb = 7.0
                self.manual_model_size_gb.set("7.0")

        except ValueError:
            layers = 32
            size_gb = 7.0
            self.manual_model_layers.set("32")
            self.manual_model_size_gb.set("7.0")

        # Generate synthetic model analysis result
        self.current_model_analysis = {
            "path": self.model_path.get() or "Manual Model",
            "file_size_bytes": int(size_gb * (1024**3)),
            "file_size_gb": size_gb,
            "architecture": "Manual",
            "n_layers": layers,
            "metadata": {},
            "error": None,
            "message": None,
            "manual_mode": True
        }

        # Update model info display
        self.model_architecture_var.set("Manual Entry")
        self.model_filesize_var.set(f"{size_gb:.2f} GB (Manual)")
        self.model_total_layers_var.set(f"{layers} (Manual)")

        # Update GPU layer controls
        self.max_gpu_layers.set(layers)
        self.gpu_layers_status_var.set(f"Max Layers: {layers} (Manual)")

        if hasattr(self, 'gpu_layers_slider') and self.gpu_layers_slider.winfo_exists():
            self.gpu_layers_slider.config(to=layers, state=tk.NORMAL)

        if hasattr(self, 'n_gpu_layers_entry') and self.n_gpu_layers_entry.winfo_exists():
            self.n_gpu_layers_entry.config(state=tk.NORMAL)

        # Sync the controls based on current entry value
        self._sync_gpu_layers_from_entry()

        print(f"DEBUG: Set up manual model with {layers} layers and {size_gb} GB size", file=sys.stderr)

        # Update recommendations and save settings
        self._update_recommendations()
        self._generate_default_config_name()
        self._save_configs()

    def _apply_manual_model_settings(self):
        """Apply manual model settings without toggling the checkbox."""
        self.manual_model_mode.set(True)
        self._setup_manual_model()

    def _toggle_manual_model_mode(self):
        """Toggle between automatic and manual model analysis."""
        if self.manual_model_mode.get():
            # Switching to manual mode
            self._setup_manual_model()
        else:
            # Switching back to automatic analysis
            if self.model_path.get():
                # Re-trigger analysis if a model is selected
                self._on_model_selected()
            else:
                # No model selected, reset to default state
                self._reset_gpu_layer_controls()
                self._reset_model_info_display()
                self.current_model_analysis = {}
                self._update_recommendations()
                self._generate_default_config_name()
            self._save_configs()

        # Update manual model visibility after mode change
        self._update_manual_model_visibility()



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

        # Check if we're in manual mode
        is_manual_mode = self.manual_gpu_mode.get()

        if count > 0 and not is_manual_mode:
            # Show real detected GPU checkboxes
            for i in range(count):
                gpu_details = self.detected_gpu_devices[i] if i < len(self.detected_gpu_devices) else {}

                v = tk.BooleanVar(value=(i in loaded_selected_gpus))
                gpu_name_display = f"GPU {i}"
                if gpu_details and gpu_details.get("name"):
                     gpu_name_display += f": {gpu_details['name']}"

                cb = ttk.Checkbutton(self.gpu_checkbox_frame, text=gpu_name_display, variable=v)
                cb.pack(side="left", padx=3, pady=2)
                v.trace_add("write", lambda *args, index=i: self._on_gpu_selection_changed(index))
                self.gpu_vars.append(v)

        elif is_manual_mode and self.manual_gpu_list:
            # Show manual GPU checkboxes
            for i, manual_gpu in enumerate(self.manual_gpu_list):
                v = tk.BooleanVar(value=(i in loaded_selected_gpus))
                gpu_name_display = f"{manual_gpu['name']} ({manual_gpu['vram_gb']:.1f} GB)"

                cb = ttk.Checkbutton(self.gpu_checkbox_frame, text=gpu_name_display, variable=v)
                cb.pack(side="left", padx=3, pady=2)
                v.trace_add("write", lambda *args, index=i: self._on_gpu_selection_changed(index))
                self.gpu_vars.append(v)

        elif not is_manual_mode and count == 0:
            # No GPUs detected - show message
            no_gpu_label = ttk.Label(self.gpu_checkbox_frame, text="No CUDA devices detected.", foreground="orange")
            no_gpu_label.pack(side="left", padx=5, pady=3)

        # Add separator before manual controls
        if (count > 0 and not is_manual_mode) or (is_manual_mode and self.manual_gpu_list):
            separator = ttk.Separator(self.gpu_checkbox_frame, orient="vertical")
            separator.pack(side="left", fill="y", padx=(15, 15))

        # Manual mode controls
        manual_controls_frame = ttk.Frame(self.gpu_checkbox_frame)
        manual_controls_frame.pack(side="left", padx=5, pady=3)

        # Manual mode toggle
        manual_cb = ttk.Checkbutton(manual_controls_frame, text="Manual GPU Mode",
                                   variable=self.manual_gpu_mode,
                                   command=self._toggle_manual_gpu_mode)
        manual_cb.pack(side="top", anchor="w", pady=(0, 5))

        if is_manual_mode:
            # Manual GPU management interface
            mgmt_frame = ttk.LabelFrame(manual_controls_frame, text="Manage Manual GPUs", padding=5)
            mgmt_frame.pack(side="top", fill="both", expand=True)

            # Add new GPU section
            add_frame = ttk.Frame(mgmt_frame)
            add_frame.pack(side="top", fill="x", pady=(0, 5))

            ttk.Label(add_frame, text="Add GPU:").pack(side="left", padx=(0, 5))

            ttk.Label(add_frame, text="Name:").pack(side="left", padx=(5, 2))
            name_entry = ttk.Entry(add_frame, textvariable=self.manual_gpu_name_var, width=12)
            name_entry.pack(side="left", padx=2)

            ttk.Label(add_frame, text="VRAM (GB):").pack(side="left", padx=(5, 2))
            vram_entry = ttk.Entry(add_frame, textvariable=self.manual_gpu_vram_var, width=6)
            vram_entry.pack(side="left", padx=2)

            ttk.Button(add_frame, text="Add GPU", command=self._add_manual_gpu).pack(side="left", padx=(5, 0))

            # Current manual GPUs list
            if self.manual_gpu_list:
                list_frame = ttk.Frame(mgmt_frame)
                list_frame.pack(side="top", fill="both", expand=True, pady=(5, 0))

                ttk.Label(list_frame, text="Current Manual GPUs:").pack(side="top", anchor="w")

                # Create a scrollable listbox for manual GPUs
                list_container = ttk.Frame(list_frame)
                list_container.pack(side="top", fill="both", expand=True, pady=(2, 0))

                self.manual_gpu_listbox = tk.Listbox(list_container, height=3, width=40)
                scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=self.manual_gpu_listbox.yview)
                self.manual_gpu_listbox.config(yscrollcommand=scrollbar.set)

                self.manual_gpu_listbox.pack(side="left", fill="both", expand=True)
                scrollbar.pack(side="right", fill="y")

                # Populate the listbox
                for i, gpu in enumerate(self.manual_gpu_list):
                    display_text = f"GPU {i}: {gpu['name']} ({gpu['vram_gb']:.1f} GB)"
                    self.manual_gpu_listbox.insert(tk.END, display_text)

                # Remove button
                remove_frame = ttk.Frame(list_frame)
                remove_frame.pack(side="top", fill="x", pady=(5, 0))
                ttk.Button(remove_frame, text="Remove Selected GPU", command=self._remove_manual_gpu).pack(side="left")
                ttk.Button(remove_frame, text="Clear All", command=self._clear_manual_gpus).pack(side="left", padx=(5, 0))

        # Trigger immediate update of recommendations instead of delayed
        self._update_recommendations()

    def _refresh_vram_display(self):
        """Helper method to refresh VRAM display if it exists."""
        if hasattr(self, '_vram_info_frame'):
            try:
                # Get the parent and row from the existing frame's grid info
                grid_info = self._vram_info_frame.grid_info()
                if grid_info:
                    parent = self._vram_info_frame.master
                    row = grid_info.get('row', 0)
                    self._display_gpu_vram_info(parent, row)
            except Exception as e:
                print(f"DEBUG: Could not refresh VRAM display: {e}", file=sys.stderr)

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
            is_manual_mode = self.gpu_info.get("manual_mode", False)
            vram_label_text = "VRAM (Total GB):" if not is_manual_mode else "VRAM (Manual Setup):"

            ttk.Label(self._vram_info_frame, text=vram_label_text, font=("TkSmallCaptionFont", 8, ("bold",)))\
                .pack(side="left", padx=(0, 5))

            # Display VRAM for each detected GPU
            for gpu in self.detected_gpu_devices:
                 gpu_text = f"GPU {gpu['id']}: {gpu['total_memory_gb']:.2f} GB"
                 if is_manual_mode:
                     gpu_text += " (Manual)"
                 ttk.Label(self._vram_info_frame, text=gpu_text, font="TkSmallCaptionFont")\
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

        # Clear any pending debounced update to prevent duplicate calls
        if self._recommendations_update_timer is not None:
            self.root.after_cancel(self._recommendations_update_timer)
            self._recommendations_update_timer = None

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
    #  Debounced Recommendations Update
    # ═════════════════════════════════════════════════════════════════
    def _schedule_recommendations_update(self, delay_ms=300):
        """
        Schedules a debounced update to recommendations.
        Cancels any pending update and schedules a new one after delay_ms milliseconds.
        This prevents excessive calls when the user is rapidly changing slider values.
        """
        # Cancel any existing timer
        if self._recommendations_update_timer is not None:
            self.root.after_cancel(self._recommendations_update_timer)

        # Schedule new update
        self._recommendations_update_timer = self.root.after(delay_ms, self._update_recommendations)

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
    #  Configuration Management (Delegated to ConfigManager)
    # ═════════════════════════════════════════════════════════════════
    def _get_config_path(self):
        """Delegates to config manager."""
        return self.config_manager.get_config_path()

    def _load_saved_configs(self):
        """Delegates to config manager."""
        return self.config_manager.load_saved_configs()

    def _save_configs(self):
        """Delegates to config manager."""
        return self.config_manager.save_configs()

    def _save_configuration(self):
        """Delegates to config manager."""
        return self.config_manager.save_configuration()

    def _generate_default_config_name(self):
        """Delegates to config manager."""
        return self.config_manager.generate_default_config_name()

    def _update_default_config_name_if_needed(self, *args):
        """Delegates to config manager."""
        return self.config_manager.update_default_config_name_if_needed(*args)

    def _current_cfg(self):
        """Delegates to config manager."""
        return self.config_manager.current_cfg()

    def _load_configuration(self):
        """Delegates to config manager."""
        return self.config_manager.load_configuration()

    def _delete_configuration(self):
        """Delegates to config manager."""
        return self.config_manager.delete_configuration()

    def _on_config_selected(self):
        """Delegates to config manager."""
        return self.config_manager.on_config_selected()

    def _update_config_listbox(self):
        """Delegates to config manager."""
        return self.config_manager.update_config_listbox()

    def _export_configurations(self):
        """Delegates to config manager."""
        return self.config_manager.export_configurations()

    def _import_configurations(self):
        """Delegates to config manager."""
        return self.config_manager.import_configurations()

    # ═════════════════════════════════════════════════════════════════
    #  Backend Selection Handler
    # ═════════════════════════════════════════════════════════════════
    def _update_root_directory_labels(self):
        """Update the root directory labels based on the selected backend."""
        backend = self.backend_selection.get()
        if backend == "ik_llama":
            self.root_dir_label_text.set("ik_llama Root Directory:")
            self.root_dir_help_text.set("Select the main 'ik_llama' folder. The app will search for the server executable.")
            # Switch to ik_llama directory variable
            self.current_backend_dir.set(self.ik_llama_dir.get())
        else:  # Default to llama.cpp
            self.root_dir_label_text.set("LLaMa.cpp Root Directory:")
            self.root_dir_help_text.set("Select the main 'llama.cpp' folder. The app will search for the server executable.")
            # Switch to llama.cpp directory variable
            self.current_backend_dir.set(self.llama_cpp_dir.get())

    def _on_backend_selection_changed(self):
        """Handler for backend selection changes."""
        selected_backend = self.backend_selection.get()
        self.app_settings["backend_selection"] = selected_backend
        self._update_root_directory_labels()  # Update the labels
        self._update_ik_llama_tab_visibility()  # Update ik_llama tab visibility
        self._save_configs()

    def _update_ik_llama_tab_visibility(self):
        """Show or hide the ik_llama tab based on backend selection."""
        if not hasattr(self, 'notebook') or not hasattr(self, 'ik_llama_frame'):
            return  # Not initialized yet

        backend = self.backend_selection.get()

        if backend == "ik_llama":
            # Show the ik_llama tab if not already visible
            try:
                # Check if tab is already in notebook
                tab_ids = [self.notebook.tab(i, "text") for i in range(self.notebook.index("end"))]
                if "ik_llama Config" not in tab_ids:
                    # Insert ik_llama tab after Environment Variables tab
                    env_tab_index = None
                    for i, tab_text in enumerate(tab_ids):
                        if tab_text == "Environment Variables":
                            env_tab_index = i
                            break

                    if env_tab_index is not None:
                        self.notebook.insert(env_tab_index + 1, self.ik_llama_frame, text="ik_llama Config")
                    else:
                        # Fallback: add at the end before About tab
                        self.notebook.insert(self.notebook.index("end") - 1, self.ik_llama_frame, text="ik_llama Config")
            except Exception as e:
                print(f"DEBUG: Error adding ik_llama tab: {e}", file=sys.stderr)
        else:
            # Hide the ik_llama tab if visible
            try:
                tab_ids = [self.notebook.tab(i, "text") for i in range(self.notebook.index("end"))]
                for i, tab_text in enumerate(tab_ids):
                    if tab_text == "ik_llama Config":
                        self.notebook.forget(i)
                        break
            except Exception as e:
                print(f"DEBUG: Error removing ik_llama tab: {e}", file=sys.stderr)

    def _on_backend_dir_changed(self):
        """Handler for when the current backend directory is manually changed."""
        backend = self.backend_selection.get()
        new_dir = self.current_backend_dir.get()

        # Update the appropriate backend-specific variable
        if backend == "ik_llama":
            self.ik_llama_dir.set(new_dir)
            self.app_settings["last_ik_llama_dir"] = new_dir
        else:  # llama.cpp
            self.llama_cpp_dir.set(new_dir)
            self.app_settings["last_llama_cpp_dir"] = new_dir

        # Save the settings
        self._save_configs()

    # ═════════════════════════════════════════════════════════════════
    #  on_exit
    # ═════════════════════════════════════════════════════════════════
    # Fixed structure: method definition is outside other methods
    def on_exit(self):
        self._save_configs()
        self.root.destroy()

    def _browse_backend_dir(self):
        """Opens a directory chooser for the currently selected backend directory."""
        current_dir = self.current_backend_dir.get().strip()
        initial_dir = current_dir if current_dir and Path(current_dir).is_dir() else str(Path.home())

        directory = filedialog.askdirectory(title="Select Directory", initialdir=initial_dir)
        if directory:
            try:
                # Resolve the path before setting to handle symlinks etc.
                resolved_path = Path(directory).resolve()
                resolved_path_str = str(resolved_path)

                # Update the current backend directory
                self.current_backend_dir.set(resolved_path_str)

                # Save to the appropriate backend-specific variable
                backend = self.backend_selection.get()
                if backend == "ik_llama":
                    self.ik_llama_dir.set(resolved_path_str)
                    self.app_settings["last_ik_llama_dir"] = resolved_path_str
                    print(f"DEBUG: Updated ik_llama directory to: {resolved_path_str}", file=sys.stderr)
                else:  # llama.cpp
                    self.llama_cpp_dir.set(resolved_path_str)
                    self.app_settings["last_llama_cpp_dir"] = resolved_path_str
                    print(f"DEBUG: Updated llama.cpp directory to: {resolved_path_str}", file=sys.stderr)

                # Save the settings
                self._save_configs()

            except Exception as e:
                print(f"Error resolving path '{directory}': {e}", file=sys.stderr)
                # Fallback to setting the raw path if resolving fails
                self.current_backend_dir.set(directory)

    # ═════════════════════════════════════════════════════════════════
    #  Asynchronous System Info Detection
    # ═════════════════════════════════════════════════════════════════

    def _start_system_info_detection(self):
        """Start system info detection in a background thread."""
        # Prevent multiple simultaneous detection threads
        if hasattr(self, '_detection_in_progress') and self._detection_in_progress:
            debug_print("GPU detection already in progress, skipping new request")
            return

        self._detection_in_progress = True

        def detect_system_info():
            """Background thread function to detect system info."""
            try:
                print("DEBUG: Starting background system info detection...", file=sys.stderr)
                # Perform the potentially slow system info detection
                self.system_info_manager.fetch_system_info()
                print("DEBUG: Background system info detection completed.", file=sys.stderr)

                # Schedule UI update on main thread (flag will be cleared there)
                self.root.after(0, self._on_system_info_detection_complete)
            except Exception as e:
                print(f"ERROR: System info detection failed: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                # Even on error, schedule completion handler to update UI (flag will be cleared there)
                self.root.after(0, lambda: self._on_system_info_detection_complete(error=str(e)))

        # Start detection in background thread
        detection_thread = Thread(target=detect_system_info, daemon=True)
        detection_thread.start()

    def _on_system_info_detection_complete(self, error=None):
        """Handle completion of system info detection (runs on main thread)."""
        try:
            # Clear detection flag (single point of clearing)
            self._detection_in_progress = False

            if error:
                self._handle_detection_error(error)
            else:
                self._handle_detection_success()

            # Update UI components
            self._update_gpu_checkboxes()
            self._refresh_vram_display()
            self._update_recommendations()

            print("DEBUG: UI update after system info detection completed.", file=sys.stderr)

        except Exception as e:
            print(f"ERROR: Failed to update UI after system info detection: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    def _handle_detection_error(self, error):
        """Handle GPU detection error state."""
        print(f"DEBUG: System info detection completed with error: {error}", file=sys.stderr)
        self.gpu_detected_status_var.set(f"GPU detection failed: {error}")
        self.gpu_availability_var.set("CUDA Devices (Detection failed):")
        self.recommended_threads_var.set(f"Recommended: {self.physical_cores} (detection failed)")
        self.recommended_threads_batch_var.set(f"Recommended: {self.logical_cores} (detection failed)")
        self.cpu_logical_cores_display_var.set(f"{self.logical_cores} (detection failed)")
        self.cpu_physical_cores_display_var.set(f"{self.physical_cores} (detection failed)")

    def _handle_detection_success(self):
        """Handle successful GPU detection."""
        print("DEBUG: System info detection completed successfully, updating UI...", file=sys.stderr)

        # Update CPU information
        self._update_cpu_info()

        # Update GPU information
        self._update_gpu_info()

        # Log GPU detection results
        self._log_gpu_detection_results()

    def _update_cpu_info(self):
        """Update CPU-related UI components after detection."""
        self.threads.set(str(self.physical_cores))
        self.threads_batch.set(str(self.logical_cores))
        self.recommended_threads_var.set(f"Recommended: {self.physical_cores} (Your CPU physical cores)")
        self.recommended_threads_batch_var.set(f"Recommended: {self.logical_cores} (Your CPU logical cores)")

        if "error" not in self.cpu_info:
            self.cpu_info["logical_cores"] = self.logical_cores
            self.cpu_info["physical_cores"] = self.physical_cores
            self.cpu_logical_cores_display_var.set(str(self.logical_cores))
            self.cpu_physical_cores_display_var.set(str(self.physical_cores))

    def _update_gpu_info(self):
        """Update GPU-related UI components after detection."""
        # Update GPU detection status message
        self.gpu_detected_status_var.set(self.gpu_info['message'] if not self.gpu_info['available'] and self.gpu_info.get('message') else "")

        # Update GPU availability display
        gpu_count = len(self.detected_gpu_devices)
        if self.gpu_info['available'] and gpu_count > 0:
            self.gpu_availability_var.set(f"CUDA Devices ({gpu_count} available):")
        else:
            self.gpu_availability_var.set("CUDA Devices (Not available):")

    def _log_gpu_detection_results(self):
        """Log GPU detection results for debugging."""
        gpu_count = len(self.detected_gpu_devices)
        print(f"DEBUG: GPU detection result: {gpu_count} devices detected", file=sys.stderr)

        if self.gpu_info['available'] and gpu_count > 0:
            if self.manual_gpu_mode.get():
                self._log_manual_mode_results()
            else:
                self._log_automatic_mode_results()
        else:
            self._log_no_gpu_results()

    def _log_manual_mode_results(self):
        """Log results when in manual GPU mode."""
        detected_count = len(self.detected_gpu_devices)
        manual_count = len(self.manual_gpu_list)

        print(f"DEBUG: Manual mode active. Detected {detected_count} real GPUs, {manual_count} manual GPUs configured", file=sys.stderr)

        if detected_count > 0:
            avg_detected_vram = sum(gpu.get("total_memory_gb", 0) for gpu in self.detected_gpu_devices) / detected_count
            if manual_count > 0:
                avg_manual_vram = sum(gpu.get("vram_gb", 0) for gpu in self.manual_gpu_list) / manual_count
                debug_print(f"Hardware VRAM: {avg_detected_vram:.1f}GB avg, Manual VRAM: {avg_manual_vram:.1f}GB avg")

    def _log_automatic_mode_results(self):
        """Log results when in automatic GPU mode."""
        print("DEBUG: Real GPUs detected in automatic mode - updating UI", file=sys.stderr)
        total_vram = sum(gpu.get("total_memory_gb", 0) for gpu in self.detected_gpu_devices)
        debug_print(f"Total detected VRAM: {total_vram:.1f}GB across {len(self.detected_gpu_devices)} GPUs")

    def _log_no_gpu_results(self):
        """Log results when no GPUs are detected."""
        if self.manual_gpu_mode.get():
            print("DEBUG: No real GPUs detected, keeping current manual GPU configuration", file=sys.stderr)
            if self.manual_gpu_list:
                total_manual_vram = sum(gpu.get("vram_gb", 0) for gpu in self.manual_gpu_list)
                debug_print(f"Total manual VRAM: {total_manual_vram:.1f}GB across {len(self.manual_gpu_list)} manual GPUs")
        else:
            print("DEBUG: No GPUs detected in automatic mode", file=sys.stderr)

    # ═════════════════════════════════════════════════════════════════
    #  Manual GPU Management (Redesigned)
    # ═════════════════════════════════════════════════════════════════

    def _add_manual_gpu(self):
        """Add a new manual GPU with specified name and VRAM."""
        name = self.manual_gpu_name_var.get().strip()
        vram_str = self.manual_gpu_vram_var.get().strip()

        if not name:
            name = f"Manual GPU {len(self.manual_gpu_list)}"

        try:
            vram_gb = float(vram_str)
            if vram_gb <= 0:
                messagebox.showwarning("Invalid VRAM", "VRAM must be greater than 0.")
                return
        except ValueError:
            messagebox.showwarning("Invalid VRAM", "Please enter a valid number for VRAM.")
            return

        # Add the new GPU
        new_gpu = {"name": name, "vram_gb": vram_gb}
        self.manual_gpu_list.append(new_gpu)

        # Clear input fields
        self.manual_gpu_name_var.set("")
        self.manual_gpu_vram_var.set("8.0")

        # Add to listbox if it exists
        if hasattr(self, 'manual_gpu_listbox') and self.manual_gpu_listbox.winfo_exists():
            new_index = len(self.manual_gpu_list) - 1
            display_text = f"GPU {new_index}: {new_gpu['name']} ({new_gpu['vram_gb']:.1f} GB)"
            self.manual_gpu_listbox.insert(tk.END, display_text)
            # Select the newly added GPU
            self.manual_gpu_listbox.selection_clear(0, tk.END)
            self.manual_gpu_listbox.selection_set(new_index)
            self.manual_gpu_listbox.activate(new_index)
            self.manual_gpu_listbox.see(new_index)

        # Update the detected_gpu_devices to reflect the new list
        self.detected_gpu_devices = []
        for i, manual_gpu in enumerate(self.manual_gpu_list):
            fake_gpu = {
                "id": i,
                "name": manual_gpu["name"],
                "total_memory_bytes": int(manual_gpu["vram_gb"] * (1024**3)),
                "total_memory_gb": manual_gpu["vram_gb"],
                "compute_capability": "Unknown",
                "multi_processor_count": 0,
                "manual": True
            }
            self.detected_gpu_devices.append(fake_gpu)

        # Update gpu_info
        self.gpu_info = {
            "available": len(self.manual_gpu_list) > 0,
            "device_count": len(self.manual_gpu_list),
            "devices": self.detected_gpu_devices.copy(),
            "manual_mode": True
        }

        # Only update the main GPU checkboxes, not the entire manual interface
        self._update_gpu_checkboxes()
        self._save_configs()
        print(f"DEBUG: Added manual GPU: {name} with {vram_gb:.1f} GB VRAM", file=sys.stderr)

        # Update VRAM display to show the new manual GPU
        self._refresh_vram_display()

    def _remove_manual_gpu(self):
        """Remove the selected manual GPU."""
        if not hasattr(self, 'manual_gpu_listbox'):
            return

        selection = self.manual_gpu_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a GPU to remove.")
            return

        index = selection[0]
        if 0 <= index < len(self.manual_gpu_list):
            removed_gpu = self.manual_gpu_list.pop(index)
            print(f"DEBUG: Removed manual GPU: {removed_gpu['name']}", file=sys.stderr)

            # Update the listbox directly instead of rebuilding entire UI
            self.manual_gpu_listbox.delete(index)

            # If there are still GPUs, select the next one (or previous if we removed the last)
            if self.manual_gpu_list:
                new_selection = min(index, len(self.manual_gpu_list) - 1)
                self.manual_gpu_listbox.selection_set(new_selection)
                self.manual_gpu_listbox.activate(new_selection)
                self.manual_gpu_listbox.see(new_selection)

                # Update the detected_gpu_devices to reflect the new list
                self.detected_gpu_devices = []
                for i, manual_gpu in enumerate(self.manual_gpu_list):
                    fake_gpu = {
                        "id": i,
                        "name": manual_gpu["name"],
                        "total_memory_bytes": int(manual_gpu["vram_gb"] * (1024**3)),
                        "total_memory_gb": manual_gpu["vram_gb"],
                        "compute_capability": "Unknown",
                        "multi_processor_count": 0,
                        "manual": True
                    }
                    self.detected_gpu_devices.append(fake_gpu)

                # Update gpu_info
                self.gpu_info["device_count"] = len(self.manual_gpu_list)
                self.gpu_info["devices"] = self.detected_gpu_devices.copy()

                # Only update the main GPU checkboxes, not the entire manual interface
                self._update_gpu_checkboxes()
            else:
                # No manual GPUs left - switch back to automatic mode immediately
                print("DEBUG: No manual GPUs left, switching to automatic mode", file=sys.stderr)
                self.manual_gpu_mode.set(False)
                self.app_settings["manual_gpu_mode"] = False
                # Clear status message and start detection
                self.gpu_detected_status_var.set("Re-detecting GPUs...")
                # Don't wait for UI - start detection immediately
                self._start_system_info_detection()

            # Update VRAM display to reflect GPU removal
            self._refresh_vram_display()

            # Save the changes
            self._save_configs()

        else:
            messagebox.showwarning("Invalid Selection", "Selected GPU index is out of range.")

    def _clear_manual_gpus(self):
        """Clear all manual GPUs."""
        if self.manual_gpu_list:
            result = messagebox.askyesno("Clear All GPUs",
                                       f"Are you sure you want to remove all {len(self.manual_gpu_list)} manual GPUs?")
            if result:
                self.manual_gpu_list.clear()
                print("DEBUG: Cleared all manual GPUs", file=sys.stderr)

                # Switch back to automatic mode immediately
                self.manual_gpu_mode.set(False)
                self.app_settings["manual_gpu_mode"] = False

                # Update status and start detection without delay
                self.gpu_detected_status_var.set("Re-detecting GPUs...")
                self._start_system_info_detection()

                # Save the changes
                self._save_configs()

    def _setup_manual_gpus(self):
        """Setup manual GPUs based on the current manual_gpu_list."""
        if not self.manual_gpu_mode.get():
            return

        # Create fake GPU devices from the manual list
        self.detected_gpu_devices = []
        for i, manual_gpu in enumerate(self.manual_gpu_list):
            fake_gpu = {
                "id": i,
                "name": manual_gpu["name"],
                "total_memory_bytes": int(manual_gpu["vram_gb"] * (1024**3)),
                "total_memory_gb": manual_gpu["vram_gb"],
                "compute_capability": "Unknown",
                "multi_processor_count": 0,
                "manual": True
            }
            self.detected_gpu_devices.append(fake_gpu)

        # Update gpu_info to reflect manual GPUs
        self.gpu_info = {
            "available": len(self.manual_gpu_list) > 0,
            "device_count": len(self.manual_gpu_list),
            "devices": self.detected_gpu_devices.copy(),
            "manual_mode": True
        }

        print(f"DEBUG: Set up {len(self.manual_gpu_list)} manual GPUs", file=sys.stderr)
        for i, gpu in enumerate(self.manual_gpu_list):
            print(f"DEBUG:   GPU {i}: {gpu['name']} ({gpu['vram_gb']:.1f} GB)", file=sys.stderr)

        # Update the UI
        self._update_gpu_checkboxes()
        self._update_recommendations()

        # Update VRAM display to show manual GPU information
        self._refresh_vram_display()

    def _migrate_legacy_manual_gpu_config(self):
        """Migrate from old manual_gpu_count + manual_gpu_vram format to new list format."""
        if self.manual_gpu_list:
            return  # Already using new format

        try:
            count = int(self.manual_gpu_count.get())
            vram = float(self.manual_gpu_vram.get())

            if count > 0 and vram > 0:
                # Check if we have detected GPUs and if they differ from legacy config
                detected_count = len(self.detected_gpu_devices)
                if detected_count > 0:
                    # We have detected GPUs - check if legacy config matches
                    avg_detected_vram = sum(gpu.get("total_memory_gb", 0) for gpu in self.detected_gpu_devices) / detected_count

                    # If the legacy config differs significantly from detected hardware, prefer detected hardware
                    if (count != detected_count or
                        abs(vram - avg_detected_vram) > 1.0):  # More than 1GB difference
                        print(f"DEBUG: Legacy config ({count} GPUs @ {vram}GB) differs from detected hardware ({detected_count} GPUs @ {avg_detected_vram:.1f}GB avg). Using detected hardware.", file=sys.stderr)
                        # Don't migrate legacy config - let the caller create based on detected hardware
                        return

                print(f"DEBUG: Migrating legacy manual GPU config: {count} GPUs with {vram} GB each", file=sys.stderr)
                for i in range(count):
                    gpu = {"name": f"Manual GPU {i}", "vram_gb": vram}
                    self.manual_gpu_list.append(gpu)

                # Save migrated config
                self._save_configs()
                print("DEBUG: Legacy manual GPU config migrated successfully", file=sys.stderr)
        except (ValueError, TypeError):
            print("DEBUG: No valid legacy manual GPU config to migrate", file=sys.stderr)

    def _apply_manual_gpu_settings(self):
        """Legacy method for compatibility - now redirects to new system."""
        print("DEBUG: Legacy _apply_manual_gpu_settings called - migrating to new system", file=sys.stderr)
        self._migrate_legacy_manual_gpu_config()
        self._setup_manual_gpus()

    def _toggle_manual_gpu_mode(self):
        """Toggle between automatic and manual GPU detection."""
        if self.manual_gpu_mode.get():
            # Switching to manual mode - do this immediately without waiting for detection
            print("DEBUG: Switching to manual GPU mode", file=sys.stderr)

            # Clear any ongoing detection
            if hasattr(self, '_detection_in_progress'):
                self._detection_in_progress = False

            # Update status immediately
            self.gpu_detected_status_var.set("Manual mode active")

            # Migrate legacy config if needed
            self._migrate_legacy_manual_gpu_config()

            # If no manual GPUs exist after migration, create them based on detected hardware
            if not self.manual_gpu_list:
                # Use detected GPU information to create manual GPUs that match the hardware
                if self.detected_gpu_devices:
                    print(f"DEBUG: Creating manual GPUs based on {len(self.detected_gpu_devices)} detected GPUs", file=sys.stderr)
                    for i, detected_gpu in enumerate(self.detected_gpu_devices):
                        manual_gpu = {
                            "name": f"Manual GPU {i}",
                            "vram_gb": detected_gpu.get("total_memory_gb", 8.0)
                        }
                        self.manual_gpu_list.append(manual_gpu)
                        print(f"DEBUG: Added manual GPU {i}: {manual_gpu['name']} ({manual_gpu['vram_gb']:.2f} GB)", file=sys.stderr)
                else:
                    # Fallback if no GPUs detected
                    self.manual_gpu_list.append({"name": "Manual GPU 0", "vram_gb": 8.0})
                    print("DEBUG: Added default manual GPU (no hardware detected)", file=sys.stderr)

            # Setup manual GPUs immediately
            self._setup_manual_gpus()
        else:
            # Switching back to automatic detection - only now trigger background detection
            print("DEBUG: Switching back to automatic GPU detection...", file=sys.stderr)
            self.gpu_detected_status_var.set("Re-detecting GPUs...")

            # Reset manual GPU info
            self.manual_gpu_list.clear()

            # Start background detection (this will update the UI when complete)
            self._start_system_info_detection()

        self._save_configs()

    def _refresh_manual_gpus_from_detected(self):
        """Refresh manual GPU list to match detected hardware."""
        if not self.manual_gpu_mode.get() or not self.detected_gpu_devices:
            return

        print(f"DEBUG: Refreshing manual GPUs to match {len(self.detected_gpu_devices)} detected GPUs", file=sys.stderr)

        # Clear current manual GPU list
        self.manual_gpu_list.clear()

        # Create new manual GPUs based on detected hardware
        for i, detected_gpu in enumerate(self.detected_gpu_devices):
            if not detected_gpu.get("manual", False):  # Only use real detected GPUs
                manual_gpu = {
                    "name": f"Manual GPU {i}",
                    "vram_gb": detected_gpu.get("total_memory_gb", 8.0)
                }
                self.manual_gpu_list.append(manual_gpu)
                print(f"DEBUG: Refreshed manual GPU {i}: {manual_gpu['name']} ({manual_gpu['vram_gb']:.2f} GB)", file=sys.stderr)

        # Save the updated manual GPU list and re-setup
        self._save_configs()
        self._setup_manual_gpus()

    def _on_venv_dir_changed(self):
        """Handler for when the virtual environment directory is manually changed."""
        new_dir = self.venv_dir.get()
        if new_dir:
            self.app_settings["last_venv_dir"] = new_dir
            self._save_configs()
            print(f"DEBUG: Updated virtual environment directory to: {new_dir}", file=sys.stderr)

            # Re-trigger GPU detection if we're in automatic mode and not already detecting
            if not self.manual_gpu_mode.get():
                print("DEBUG: Re-triggering GPU detection due to venv change", file=sys.stderr)
                self.gpu_detected_status_var.set("Re-detecting GPUs with new venv...")
                self._start_system_info_detection()
        else:
            print("DEBUG: Virtual environment directory cleared", file=sys.stderr)
            self.app_settings["last_venv_dir"] = ""
            self._save_configs()

            # Re-trigger GPU detection if we're in automatic mode (fallback to no venv)
            if not self.manual_gpu_mode.get():
                print("DEBUG: Re-triggering GPU detection due to venv removal", file=sys.stderr)
                self.gpu_detected_status_var.set("Re-detecting GPUs without venv...")
                self._start_system_info_detection()

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
