#!/usr/bin/env python3
"""
Launch functionality extracted from LLaMa.cpp Server Launcher.
Contains command building and server launching methods.
"""

import os
import re
import sys
import subprocess
import tempfile
import traceback
import shlex
import shutil
import stat
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, filedialog
from threading import Thread


class LaunchManager:
    """Manages command building and server launching functionality."""
    
    def __init__(self, launcher_instance):
        """Initialize with a reference to the main launcher instance."""
        self.launcher = launcher_instance
    
    def build_cmd(self):
        """Builds the command list for the server based on selected backend."""
        # Get the backend selection and use appropriate directory
        backend = self.launcher.backend_selection.get()
        
        if backend == "ik_llama":
            backend_dir_str = self.launcher.ik_llama_dir.get().strip()
            backend_name = "ik_llama"
        else:  # Default to llama.cpp
            backend_dir_str = self.launcher.llama_cpp_dir.get().strip()
            backend_name = "llama.cpp"
            
        if not backend_dir_str:
            messagebox.showerror("Error", f"{backend_name} root directory is not set.")
            return None
        try:
            backend_base_dir = Path(backend_dir_str).resolve() # Resolve the base dir
            if not backend_base_dir.is_dir(): raise NotADirectoryError()
        except Exception:
             messagebox.showerror("Error", f"Invalid {backend_name} directory:\n{backend_dir_str}")
             return None

        # Find server executable for the selected backend
        exe_path = self._find_server_executable(backend_base_dir, backend)
        if not exe_path:
            search_locs_str = "\n - ".join([str(p) for p in [
                 Path("."), Path("build/bin/Release"), Path("build/bin"), Path("build"), Path("bin"), Path("server")
            ]])
            
            # Get backend-specific executable names for error message
            if backend == "ik_llama":
                exe_names = self._get_ik_llama_executable_names()
                backend_display = "ik_llama"
            else:
                exe_names = self._get_llama_cpp_executable_names()
                backend_display = "llama.cpp"
                
            exe_names_str = "', '".join(exe_names)
            messagebox.showerror("Executable Not Found",
                                 f"Could not find '{exe_names_str}' within:\n{backend_base_dir}\n\n"
                                 f"Searched in common relative locations like:\n - {search_locs_str}\n\n"
                                 f"Please ensure {backend_display} is built and the directory is correct.")
            return None

        cmd = [str(exe_path)]

        # --- Model Path ---
        model_full_path_str = self.launcher.model_path.get().strip()
        if not model_full_path_str:
            messagebox.showerror("Error", "No model selected. Please scan and select a model from the list.")
            return None
        try:
            model_path_obj = Path(model_full_path_str).resolve() # Resolve the path from the variable
            # Cross-check if the resolved path matches the one from our scan results
            # This handles cases where the user might manually type a path or the saved path is slightly different
            selected_name = ""
            sel = self.launcher.model_listbox.curselection()
            if sel: selected_name = self.launcher.model_listbox.get(sel[0])

            scan_matched_path = self.launcher.found_models.get(selected_name)

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
             sel = self.launcher.model_listbox.curselection()
             if sel: selected_name = self.launcher.model_listbox.get(sel[0])
             error_msg = f"Error validating model path:\n{model_full_path_str}\nError: {e}"
             if selected_name: error_msg += f"\n(Selected in GUI: {selected_name})"
             error_msg += "\n\nPlease re-scan models or select a valid model."
             messagebox.showerror("Error", error_msg)
             return None

        cmd.extend(["-m", final_model_path])

        # --- Other Arguments ---
        # --- KV Cache Type ---
        # Check if ik_llama backend is using -ctk or -ctv flags
        ik_llama_uses_cache_flags = False
        if backend == "ik_llama":
            ik_llama_flags = self.launcher.ik_llama_tab.get_ik_llama_flags()
            ik_llama_uses_cache_flags = "-ctk" in ik_llama_flags or "-ctv" in ik_llama_flags
        
        # Only add standard cache type flags if ik_llama is not using -ctk/-ctv
        if not ik_llama_uses_cache_flags:
            # llama.cpp default is f16. Add args only if different.
            kv_cache_type_val = self.launcher.cache_type_k.get().strip()
            # Note: llama.cpp's --cache-type-v defaults to the value of --cache-type-k if not specified.
            # So just setting --cache-type-k is usually sufficient.
            if kv_cache_type_val and kv_cache_type_val != "f16":
                cmd.extend(["--cache-type-k", kv_cache_type_val])
                print(f"DEBUG: Adding --cache-type-k {kv_cache_type_val} (non-default)", file=sys.stderr)
                # Always set V cache type to match K cache type to avoid unsupported combinations
                cmd.extend(["--cache-type-v", kv_cache_type_val])
                print(f"DEBUG: Adding --cache-type-v {kv_cache_type_val} (matching K cache type)", file=sys.stderr)
        else:
            print("DEBUG: Skipping standard --cache-type-k/v flags because ik_llama -ctk/-ctv is enabled", file=sys.stderr)

        # Remove the separate V cache type handling since we always want it to match K
        # v_cache_type_val = self.launcher.cache_type_v.get().strip()
        # if v_cache_type_val and v_cache_type_val != kv_cache_type_val:
        #     cmd.extend(["--cache-type-v", v_cache_type_val])
        #     print(f"DEBUG: Adding --cache-type-v {v_cache_type_val} (different from K cache type)", file=sys.stderr)

        # --- Threads & Batching ---
        # Llama.cpp internal defaults: --threads=hardware_concurrency() (logical), --threads-batch=4
        # Note: The GUI default for --threads is physical cores, while llama.cpp default is logical.
        # The _add_arg helper needs to compare against the *llama.cpp* default for omission.
        # But the default *value* shown in the GUI should still be physical cores.
        # Let's compare against the llama.cpp default (logical cores) when deciding whether to *add* the arg.
        self.add_arg(cmd, "--threads", self.launcher.threads.get(), str(self.launcher.logical_cores)) # Omit if matches llama.cpp default (logical)
        self.add_arg(cmd, "--threads-batch", self.launcher.threads_batch.get(), "4")         # Omit if matches llama.cpp default (4)

        # Llama.cpp internal defaults: --batch-size=512, --ubatch-size=512
        self.add_arg(cmd, "--batch-size", self.launcher.batch_size.get(), "512")
        self.add_arg(cmd, "--ubatch-size", self.launcher.ubatch_size.get(), "512")

        # Llama.cpp internal defaults: --ctx-size=2048, --seed=-1, --temp=0.8, --min-p=0.05
        self.add_arg(cmd, "--ctx-size", str(self.launcher.ctx_size.get()), "2048") # Use str() for int var
        self.add_arg(cmd, "--seed", self.launcher.seed.get(), "-1")
        self.add_arg(cmd, "--temp", self.launcher.temperature.get(), "0.8")
        self.add_arg(cmd, "--min-p", self.launcher.min_p.get(), "0.05")

        # --- Handle GPU arguments: Skip if tensor override is active ---
        # Check if tensor override is enabled and has parameters
        tensor_override_active = False
        if hasattr(self.launcher, 'tensor_override_tab'):
            try:
                tensor_override_enabled = self.launcher.tensor_override_tab.tensor_override_enabled.get()
                tensor_params = self.launcher.tensor_override_tab.get_tensor_override_parameters()
                tensor_override_active = tensor_override_enabled
            except Exception as e:
                print(f"WARNING: Error checking tensor override status: {e}", file=sys.stderr)
        
        # Always get these values for later use
        tensor_split_val = self.launcher.tensor_split.get().strip()
        n_gpu_layers_val = self.launcher.n_gpu_layers.get().strip()
        main_gpu_val = self.launcher.main_gpu.get().strip()
        
        if tensor_override_active:
            print("DEBUG: Skipping --tensor-split, --n-gpu-layers, and --main-gpu (tensor override is active)", file=sys.stderr)
        else:
            # Original GPU argument handling when tensor override is not active
            # Add --tensor-split if the value is non-empty
            # Use add_arg which handles the non-empty check
            self.add_arg(cmd, "--tensor-split", tensor_split_val, "") # Add if non-empty string is provided by user

            # Add --n-gpu-layers if the value is non-empty AND not the default "0" string
            # This argument will now be added regardless of the --tensor-split value
            self.add_arg(cmd, "--n-gpu-layers", n_gpu_layers_val, "0")

            # --main-gpu is usually needed when offloading layers (either via --n-gpu-layers or --tensor-split)
            # It specifies which GPU is considered the "primary" one, often GPU 0.
            # Llama.cpp default is 0. Include --main-gpu if the user set a non-default value.
            self.add_arg(cmd, "--main-gpu", main_gpu_val, "0")

        # Add --flash-attn flag if checked
        self.add_arg(cmd, "--flash-attn", self.launcher.flash_attn.get())

        # Memory options
        self.add_arg(cmd, "--no-mmap", self.launcher.no_mmap.get()) # Omit if False (default)
        self.add_arg(cmd, "--mlock", self.launcher.mlock.get()) # Omit if False (default)
        self.add_arg(cmd, "--no-kv-offload", self.launcher.no_kv_offload.get()) # Omit if False (default)

        # Performance options
        self.add_arg(cmd, "--prio", self.launcher.prio.get(), "0") # Omit if 0 (default)

        # --- NEW: Generation options ---
        self.add_arg(cmd, "--ignore-eos", self.launcher.ignore_eos.get()) # Omit if False (default)
        self.add_arg(cmd, "--n-predict", self.launcher.n_predict.get(), "-1") # Omit if -1 (default)

        # --- Network Settings ---
        self.add_arg(cmd, "--host", self.launcher.host.get(), "127.0.0.1") # Add host if not default
        self.add_arg(cmd, "--port", self.launcher.port.get(), "8080") # Add port if not default

        # --- CHANGES FOR JSON TEMPLATES / DEFAULT OPTION ---
        # Add --chat-template option ONLY if the source is not "default" (llama.cpp decides)
        source = self.launcher.template_source.get()
        if source in ["predefined", "custom"]:
             effective_template = self.launcher.current_template_display.get().strip()
             if effective_template: # Only add the argument if the effective template string is non-empty
                  # No default_value check needed here because if it's empty, the arg isn't added anyway by the outer if
                  cmd.extend(["--chat-template", effective_template])
                  print(f"DEBUG: Adding --chat-template: {effective_template[:50]}...", file=sys.stderr)
             else:
                  print("DEBUG: Chat template source is predefined/custom, but effective template string is empty. Omitting --chat-template.", file=sys.stderr)
        else: # source == "default"
             print("DEBUG: Chat template source is 'Let llama.cpp Decide'. Omitting --chat-template.", file=sys.stderr)

        # --- NEW: Add Custom Parameters ---
        print(f"DEBUG: Adding {len(self.launcher.custom_parameters_list)} custom parameters...", file=sys.stderr)
        for param_string in self.launcher.custom_parameters_list:
            try:
                # Use shlex.split to correctly parse potentially quoted arguments
                # shlex.split will split "--param value with spaces" into ["--param", "value with spaces"]
                split_params = shlex.split(param_string)
                cmd.extend(split_params)
                print(f"DEBUG: Added custom param: {param_string} -> {split_params}", file=sys.stderr)
            except Exception as e:
                print(f"WARNING: Could not parse custom parameter '{param_string}': {e}. Skipping.", file=sys.stderr)
                messagebox.showwarning("Custom Parameter Warning", f"Could not parse custom parameter '{param_string}': {e}\nIt will be ignored.")

        # --- NEW: Add Tensor Override Parameters ---
        if hasattr(self.launcher, 'tensor_override_tab') and hasattr(self.launcher, 'tensor_override_manager'):
            try:
                # Get tensor override parameters if enabled
                tensor_params = self.launcher.tensor_override_tab.get_tensor_override_parameters()
                if tensor_params:
                    # Validate parameters before adding
                    is_valid, issues = self.launcher.tensor_override_manager.validate_tensor_override_params(tensor_params)
                    if is_valid:
                        formatted_params = self.launcher.tensor_override_manager.format_params_for_command_line(tensor_params)
                        cmd.extend(formatted_params)
                        print(f"DEBUG: Added {len(tensor_params)} tensor override parameters", file=sys.stderr)
                        print(f"DEBUG: Tensor override devices: {[p.split()[2] for p in tensor_params if len(p.split()) >= 3]}", file=sys.stderr)
                    else:
                        print(f"WARNING: Invalid tensor override parameters: {issues}", file=sys.stderr)
                        messagebox.showwarning("Tensor Override Warning", f"Invalid tensor override parameters detected:\n{'; '.join(issues[:3])}\nThey will be ignored.")
                else:
                    print("DEBUG: No tensor override parameters enabled or available", file=sys.stderr)
            except Exception as e:
                print(f"WARNING: Error loading tensor override parameters: {e}", file=sys.stderr)

        # --- NEW: Add ik_llama Specific Flags ---
        if backend == "ik_llama":
            ik_llama_flags = self.launcher.ik_llama_tab.get_ik_llama_flags()
            if ik_llama_flags:
                cmd.extend(ik_llama_flags)
                print(f"DEBUG: Added ik_llama flags: {ik_llama_flags}", file=sys.stderr)
            else:
                print("DEBUG: No ik_llama flags enabled", file=sys.stderr)

        # Add a note about using CUDA_VISIBLE_DEVICES if they selected specific GPUs via checkboxes
        # but are NOT using --tensor-split (which explicitly lists devices/split).
        # This warning is helpful because llama.cpp might use all GPUs by default unless restricted by env var or tensor-split.
        selected_gpu_indices = [i for i, v in enumerate(self.launcher.gpu_vars) if v.get()]
        detected_gpu_count = self.launcher.gpu_info.get("device_count", 0)
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

    def _find_server_executable(self, llama_base_dir, backend):
        """Finds the llama-server executable within the llama.cpp directory."""
        # Get backend-specific executable names
        if backend == "ik_llama":
            exe_names = self._get_ik_llama_executable_names()
            backend_display = "ik_llama"
        else:
            exe_names = self._get_llama_cpp_executable_names()
            backend_display = "llama.cpp"

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
            for exe_name in exe_names:
                full_path = llama_base_dir / rel_path / exe_name
                if full_path.is_file():
                    print(f"DEBUG: Found server executable at: {full_path}", file=sys.stderr)
                    return full_path.resolve() # Return the resolved path

        # As a last resort, check if the base directory *itself* is the bin directory
        # and contains the executable directly. This handles cases where build puts it
        # directly in the root, although less common.
        for exe_name in exe_names:
            direct_path = llama_base_dir / exe_name
            if direct_path.is_file():
                 print(f"DEBUG: Found server executable directly in base dir: {direct_path}", file=sys.stderr)
                 return direct_path.resolve()

        print(f"DEBUG: Server executable '{', '.join(exe_names)}' not found in {llama_base_dir} or common subdirectories.", file=sys.stderr)
        return None # Executable not found anywhere

    def _get_llama_cpp_executable_names(self):
        """Get possible executable names for llama.cpp backend."""
        if sys.platform == "win32":
            # Prioritize llama-server over deprecated server binary
            return ["llama-server.exe", "server.exe", "llama-cpp-python-server.exe"]
        else:
            return ["llama-server", "server", "llama-cpp-python-server"]

    def _get_ik_llama_executable_names(self):
        """Get possible executable names for ik_llama backend."""
        # ik_llama.cpp should use llama-server (server is deprecated)
        # Only search for llama-server and ik_llama specific variants
        if sys.platform == "win32":
            return ["llama-server.exe", "ik-llama-server.exe", "ik_llama_server.exe"]
        else:
            return ["llama-server", "ik-llama-server", "ik_llama_server"]

    def add_arg(self, cmd_list, arg_name, value, default_value=None):
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

    def launch_server(self):
        """Launch the llama-server with the configured settings."""
        cmd_list = self.build_cmd()
        if not cmd_list:
            # build_cmd already showed an error message
            return

        venv_path_str = self.launcher.venv_dir.get().strip()
        use_venv = bool(venv_path_str)

        tmp_path = None # Initialize tmp_path outside try/except/finally

        # Get selected GPU indices for CUDA_VISIBLE_DEVICES
        selected_gpu_indices = [i for i, v in enumerate(self.launcher.gpu_vars) if v.get()]
        cuda_devices_value = ",".join(map(str, sorted(selected_gpu_indices)))

        try: # Main try block for creating script and launching process
            if sys.platform == "win32":
                # --- MODIFIED: Use temporary PowerShell script instead of batch file ---
                # Use mkstemp to create a secure temporary file with .ps1 suffix
                # Use text=True and encoding='utf-8' for cross-platform safety, although file handle needs closing
                fd, tmp_path = tempfile.mkstemp(suffix=".ps1",
                                                prefix="llamacpp_launch_",
                                                text=False)
                os.close(fd) # Close the file descriptor immediately

                # Use utf-8 encoding explicitly as templates can contain wide chars
                with open(tmp_path, "w", encoding="utf-8") as f:
                    # Get backend information for script header
                    backend = self.launcher.backend_selection.get()
                    backend_display = "ik_llama" if backend == "ik_llama" else "LLaMa.cpp"
                    
                    f.write("<#\n")
                    f.write(" .SYNOPSIS\n")
                    f.write(f"    Launches the {backend_display} server with saved settings.\n\n")
                    f.write(" .DESCRIPTION\n")
                    f.write(f"    Autogenerated PowerShell script from {backend_display} Launcher GUI.\n")
                    f.write(f"    Activates virtual environment (if configured) and starts {backend.replace('_', '-')}-server.\n")
                    f.write("#>\n\n")
                    f.write("$ErrorActionPreference = 'Continue'\n\n")
                    f.write('[Console]::OutputEncoding = [System.Text.Encoding]::UTF8 # Set console output encoding to UTF-8\n\n')

                    # --- Add CUDA_VISIBLE_DEVICES setting if GPUs are selected ---
                    if cuda_devices_value:
                        f.write(f'Write-Host "Setting CUDA_VISIBLE_DEVICES={cuda_devices_value}" -ForegroundColor DarkCyan\n')
                        f.write(f'$env:CUDA_VISIBLE_DEVICES="{cuda_devices_value}"\n\n')
                    elif self.launcher.gpu_info.get("device_count", 0) > 0:
                         # If GPUs are detected but none are selected, explicitly unset the variable
                         # to rely on default llama.cpp behavior or let the OS handle it.
                         # This avoids accidentally inheriting a variable from the environment.
                         f.write('Write-Host "Clearing CUDA_VISIBLE_DEVICES environment variable." -ForegroundColor DarkCyan\n')
                         f.write('Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue\n\n')

                    # --- Add Environmental Variables ---
                    env_vars = self.launcher.env_vars_manager.get_enabled_env_vars()
                    if env_vars:
                        f.write('Write-Host "Setting environmental variables..." -ForegroundColor DarkCyan\n')
                        for var_name, var_value in env_vars.items():
                            f.write(f'$env:{var_name}="{var_value}"\n')
                        f.write('\n')

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

                    f.write(f'Write-Host "Launching {backend.replace("_", "-")}-server..." -ForegroundColor Green\n')

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
                                              # Standard quoting for other args - fixed for KDE Konsole
                                              # Separate the string operations to avoid complex escaping
                                              escaped_arg = current_arg.replace('"', '""')
                                              escaped_arg = escaped_arg.replace('`', '``')
                                              quoted_arg = f'"{escaped_arg}"'
                                              ps_cmd_parts.append(quoted_arg)
                                              i += 1
                                     break # Exit this inner while loop once reconstructed

                             except ValueError: # --chat-template not found, should not happen here
                                 pass # Continue with the original loop if this somehow occurs

                         else:
                             # Standard quoting for other args - fixed for KDE Konsole
                             # Separate the string operations to avoid complex escaping
                             escaped_arg = arg.replace('"', '""')
                             escaped_arg = escaped_arg.replace('`', '``')
                             quoted_arg = f'"{escaped_arg}"'
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
                    # Remove the extra quoting since we're passing the command directly
                    full_script_content = f'source {str(act_script)} && echo "Virtual environment activated." && echo "Launching server..." && {server_command_str} ; command_status=$? ; if [[ -t 1 || $command_status -ne 0 ]]; then read -rp "Press Enter to close..." </dev/tty ; fi ; exit $command_status'

                else: # No venv
                    full_script_content = f'echo "Launching server..." && {server_command_str} ; command_status=$? ; if [[ -t 1 || $command_status -ne 0 ]]; then read -rp "Press Enter to close..." </dev/tty ; fi ; exit $command_status'

                # --- CUDA_VISIBLE_DEVICES and Environmental Variables for Linux/macOS (Bash) ---
                # Add export statements *before* the main command in the bash script
                # This should happen *after* venv activation if used.
                env_commands = []
                
                # Add CUDA_VISIBLE_DEVICES
                if cuda_devices_value:
                    env_commands.append(f'export CUDA_VISIBLE_DEVICES={cuda_devices_value}')
                    env_commands.append('echo "Setting CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"')
                
                # Add environmental variables
                env_vars = self.launcher.env_vars_manager.get_enabled_env_vars()
                if env_vars:
                    env_commands.append('echo "Setting environmental variables..."')
                    for var_name, var_value in env_vars.items():
                        env_commands.append(f'export {var_name}="{var_value}"')
                
                # Combine environment commands into a single string
                env_line = ""
                if env_commands:
                    env_line = " && ".join(env_commands) + " && "
                    
                # Insert the environment line after venv activation if used
                if use_venv and env_line:
                    full_script_content = full_script_content.replace('echo "Virtual environment activated." && ', 'echo "Virtual environment activated." && ' + env_line)
                elif env_line:
                    full_script_content = env_line + full_script_content

                # Attempt to launch in a new terminal window
                # Use 'bash -c' to execute the command string.
                # Find common terminal emulators.
                terminals = ['gnome-terminal', 'konsole', 'xfce4-terminal', 'xterm', 'iterm'] # Add iTerm for macOS
                launched = False
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
                             # gnome-terminal/xfce4-terminal/iterm expect -- followed by command string or list
                             # Pass 'bash -c command_string' as arguments after --
                             # Remove the extra single quotes around the command string
                             term_cmds.append(term_cmd_base + ['--', 'bash', '-c', full_script_content])
                        elif term == 'konsole':
                             # Konsole needs --noclose and -e followed by the command list or string
                             term_cmds.append(term_cmd_base + ['--noclose', '-e', 'bash', '-c', full_script_content])
                        elif term == 'xterm':
                             # xterm can often take the command directly after its own flags
                             term_cmds.append(term_cmd_base + ['-e', 'bash', '-c', full_script_content])
                             # Another xterm pattern
                             term_cmds.append(term_cmd_base + ['-e', 'bash', '-c', full_script_content])

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
                        subprocess.Popen(full_script_content, shell=True)
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
                cleanup_thread = Thread(target=self.launcher.cleanup, args=(tmp_path,), daemon=True)
                cleanup_thread.start()

    def save_ps1_script(self):
        """Save a PowerShell script with the current configuration."""
        cmd_list = self.build_cmd()
        if not cmd_list: return

        # --- FIX: Use the actual selected model name from the listbox ---
        selected_model_name = ""
        selection = self.launcher.model_listbox.curselection()
        if selection:
             selected_model_name = self.launcher.model_listbox.get(selection[0])

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
                # Get backend information for script header
                backend = self.launcher.backend_selection.get()
                backend_display = "ik_llama" if backend == "ik_llama" else "LLaMa.cpp"
                
                fh.write("<#\n")
                fh.write(" .SYNOPSIS\n")
                fh.write(f"    Launches the {backend_display} server with saved settings.\n\n")
                fh.write(" .DESCRIPTION\n")
                fh.write(f"    Autogenerated PowerShell script from {backend_display} Launcher GUI.\n")
                fh.write(f"    Activates virtual environment (if configured) and starts {backend.replace('_', '-')}-server.\n")
                fh.write("#>\n\n")
                fh.write("$ErrorActionPreference = 'Continue'\n\n")
                fh.write('[Console]::OutputEncoding = [System.Text.Encoding]::UTF8 # Set console output encoding to UTF-8\n\n')

                # --- Add CUDA_VISIBLE_DEVICES setting if GPUs are selected ---
                selected_gpu_indices = [i for i, v in enumerate(self.launcher.gpu_vars) if v.get()]
                cuda_devices_value = ",".join(map(str, sorted(selected_gpu_indices)))

                if cuda_devices_value:
                    fh.write(f'Write-Host "Setting CUDA_VISIBLE_DEVICES={cuda_devices_value}" -ForegroundColor DarkCyan\n')
                    fh.write(f'$env:CUDA_VISIBLE_DEVICES="{cuda_devices_value}"\n\n')
                elif self.launcher.gpu_info.get("device_count", 0) > 0:
                     # If GPUs are detected but none are selected, explicitly unset the variable
                     fh.write('Write-Host "Clearing CUDA_VISIBLE_DEVICES environment variable." -ForegroundColor DarkCyan\n')
                     fh.write('Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue\n\n')

                # --- Add Environmental Variables ---
                env_vars = self.launcher.env_vars_manager.get_enabled_env_vars()
                if env_vars:
                    fh.write('Write-Host "Setting environmental variables..." -ForegroundColor DarkCyan\n')
                    for var_name, var_value in env_vars.items():
                        fh.write(f'$env:{var_name}="{var_value}"\n')
                    fh.write('\n')

                venv = self.launcher.venv_dir.get().strip()
                if venv:
                    try:
                        venv_path = Path(venv).resolve() # Resolve venv path for script
                        
                        # Check for multiple activation script locations (cross-platform support)
                        possible_scripts = [
                            venv_path / "Scripts" / "Activate.ps1",     # Windows
                            venv_path / "bin" / "Activate.ps1",        # Linux/macOS with PowerShell Core
                            venv_path / "Scripts" / "activate.ps1",    # Alternative naming
                            venv_path / "bin" / "activate.ps1"         # Alternative naming
                        ]
                        
                        act_script = None
                        for script_path in possible_scripts:
                            if script_path.exists():
                                act_script = script_path
                                break
                        
                        if act_script:
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
                            # Format warning message with all checked paths
                            checked_paths = [str(p) for p in possible_scripts]
                            fh.write(f'Write-Warning "Virtual environment activation script not found in venv: {venv}"\n')
                            fh.write(f'Write-Warning "Checked locations: {", ".join(checked_paths)}"\n')
                            fh.write('Write-Warning "Note: PowerShell scripts work on Windows, Linux, and macOS with PowerShell Core installed."\n\n')
                    except Exception as path_ex:
                         # Format path for warning message
                         warn_venv_path = venv.replace("'", "''")
                         fh.write(f'Write-Warning "Could not process venv path \'{warn_venv_path}\': {path_ex}"\n\n')

                fh.write(f'Write-Host "Launching {backend.replace("_", "-")}-server..." -ForegroundColor Green\n')

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
                         quoted_arg = f'"{current_arg.replace('"', '""').replace("`", "``")}"'
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

    def save_sh_script(self):
        """Save a bash script with the current configuration."""
        cmd_list = self.build_cmd()
        if not cmd_list: return

        # --- Use the actual selected model name from the listbox ---
        selected_model_name = ""
        selection = self.launcher.model_listbox.curselection()
        if selection:
             selected_model_name = self.launcher.model_listbox.get(selection[0])

        default_name = "launch_llama_server.sh"
        if selected_model_name: 
            # Sanitize model name for filename
            model_name_part = re.sub(r'[\\/*?:"<>| ]', '_', selected_model_name)
            model_name_part = model_name_part[:50].strip('_') # Ensure no trailing underscore
            if model_name_part:
                default_name = f"launch_{model_name_part}.sh"
            else:
                default_name = "launch_selected_model.sh" # Fallback if name is empty after sanitizing

            if not default_name.lower().endswith(".sh"): default_name += ".sh"

        path = filedialog.asksaveasfilename(defaultextension=".sh",
                                            initialfile=default_name,
                                            filetypes=[("Bash Script", "*.sh"), ("All Files", "*.*")])
        if not path: return

        try:
            with open(path, "w", encoding="utf-8") as fh:
                # Get backend information for script header
                backend = self.launcher.backend_selection.get()
                backend_display = "ik_llama" if backend == "ik_llama" else "LLaMa.cpp"
                
                fh.write("#!/bin/bash\n")
                fh.write(f"# Autogenerated bash script from {backend_display} Launcher GUI\n")
                fh.write(f"# Activates virtual environment (if configured) and starts {backend.replace('_', '-')}-server\n\n")

                fh.write("set -e  # Exit on any error\n\n")

                # --- Add CUDA_VISIBLE_DEVICES setting if GPUs are selected ---
                selected_gpu_indices = [i for i, v in enumerate(self.launcher.gpu_vars) if v.get()]
                cuda_devices_value = ",".join(map(str, sorted(selected_gpu_indices)))

                if cuda_devices_value:
                    fh.write(f'echo "Setting CUDA_VISIBLE_DEVICES={cuda_devices_value}"\n')
                    fh.write(f'export CUDA_VISIBLE_DEVICES="{cuda_devices_value}"\n\n')
                elif self.launcher.gpu_info.get("device_count", 0) > 0:
                     # If GPUs are detected but none are selected, explicitly unset the variable
                     fh.write('echo "Clearing CUDA_VISIBLE_DEVICES environment variable."\n')
                     fh.write('unset CUDA_VISIBLE_DEVICES\n\n')

                # --- Add Environmental Variables ---
                env_vars = self.launcher.env_vars_manager.get_enabled_env_vars()
                if env_vars:
                    fh.write('echo "Setting environmental variables..."\n')
                    for var_name, var_value in env_vars.items():
                        # Escape shell special characters in the value
                        escaped_value = var_value.replace('"', '\\"').replace('$', '\\$').replace('`', '\\`')
                        fh.write(f'export {var_name}="{escaped_value}"\n')
                    fh.write('\n')

                venv = self.launcher.venv_dir.get().strip()
                if venv:
                    try:
                        venv_path = Path(venv).resolve() # Resolve venv path for script
                        # Check for both bin/activate (Linux/macOS) and Scripts/activate (Windows in WSL/Cygwin)
                        activate_script = venv_path / "bin" / "activate"
                        if not activate_script.exists():
                            activate_script = venv_path / "Scripts" / "activate"
                        
                        if activate_script.exists():
                            fh.write(f'echo "Activating virtual environment: {venv}"\n')
                            # Quote the path to handle spaces
                            quoted_activate_path = f'"{activate_script}"'
                            fh.write(f'source {quoted_activate_path} || {{ echo "Failed to activate venv"; exit 1; }}\n\n')
                        else:
                            fh.write(f'echo "Warning: Virtual environment activation script not found at: {activate_script}"\n')
                            fh.write(f'echo "Also checked: {venv_path / "bin" / "activate"} and {venv_path / "Scripts" / "activate"}"\n\n')
                    except Exception as path_ex:
                         fh.write(f'echo "Warning: Could not process venv path \'{venv}\': {path_ex}"\n\n')

                # Get backend information for script header
                backend = self.launcher.backend_selection.get()
                
                fh.write(f'echo "Launching {backend.replace("_", "-")}-server..."\n')

                # --- Build the command string for bash using appropriate quoting ---
                bash_cmd_parts = []
                
                # Process all arguments from cmd_list
                i = 0
                while i < len(cmd_list):
                    current_arg = cmd_list[i]
                    if current_arg == "--chat-template" and i + 1 < len(cmd_list):
                         template_string = cmd_list[i+1]
                         # Use single quotes for template to preserve special characters
                         escaped_template_string = template_string.replace("'", "'\"'\"'")
                         bash_cmd_parts.append("--chat-template")
                         bash_cmd_parts.append(f"'{escaped_template_string}'")
                         i += 2 # Skip both flag and value
                    else:
                         # Standard quoting for other args using shlex.quote which is bash-compatible
                         bash_cmd_parts.append(shlex.quote(current_arg))
                         i += 1

                fh.write(" ".join(bash_cmd_parts) + "\n\n")

                # Check exit code after the command
                fh.write('exit_code=$?\n')
                fh.write('if [ $exit_code -ne 0 ]; then\n')
                fh.write('    echo "Error: llama-server exited with error code: $exit_code" >&2\n')
                fh.write('    exit $exit_code\n')
                fh.write('fi\n')
                fh.write('echo "Server process finished."\n')

            # Make the script executable
            st = os.stat(path)
            os.chmod(path, st.st_mode | stat.S_IEXEC)

            messagebox.showinfo("Saved", f"Bash script written to:\n{path}\n\nThe script has been made executable.")
        except Exception as exc:
            messagebox.showerror("Script Save Error", f"Could not save script:\n{exc}")
            print(f"Script Save Error: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr) 