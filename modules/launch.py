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
        # Caches results of `<exe> --help` probes keyed by (exe_path, flag).
        # Used to gracefully skip flags older builds don't recognize, e.g.
        # --fit on ik_llama builds that predate fit support.
        self._feature_probe_cache: dict[tuple[str, str], bool] = {}

    def _backend_supports_flag(self, exe_path, flag):
        """Return True if `exe_path --help` advertises `flag`.

        Caches per (exe, flag) so repeated build_cmd() calls don't re-spawn
        the help process. On any probe failure (timeout, non-zero exit,
        OSError), assumes the flag is unsupported — safer than emitting an
        unknown argument that would crash the server at startup.
        """
        cache_key = (str(exe_path), flag)
        if cache_key in self._feature_probe_cache:
            return self._feature_probe_cache[cache_key]

        supported = False
        try:
            result = subprocess.run(
                [str(exe_path), "--help"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            help_text = (result.stdout or "") + (result.stderr or "")
            # Match the flag as a token: preceded by start-of-line/whitespace,
            # followed by whitespace, comma, end-of-line, or '='.
            pattern = rf"(?:^|\s){re.escape(flag)}(?:[\s,=]|$)"
            supported = bool(re.search(pattern, help_text, re.MULTILINE))
            print(
                f"DEBUG: Feature probe — {Path(exe_path).name} {flag}: "
                f"{'supported' if supported else 'not advertised'}",
                file=sys.stderr,
            )
        except Exception as e:
            print(
                f"DEBUG: Feature probe failed for {exe_path} {flag}: {e!r}; "
                f"assuming unsupported",
                file=sys.stderr,
            )
            supported = False

        self._feature_probe_cache[cache_key] = supported
        return supported

    def _build_llama_cpp_fit_args(self, cmd):
        """Append upstream llama.cpp fit args: --fit on|off, --fit-ctx,
        --fit-target."""
        if self.launcher.fit_enabled.get():
            cmd.extend(["--fit", "on"])
            print("DEBUG: Adding --fit on", file=sys.stderr)
            fit_ctx_val = self.launcher.fit_ctx.get().strip()
            if fit_ctx_val and fit_ctx_val != "4096":
                cmd.extend(["--fit-ctx", fit_ctx_val])
                print(f"DEBUG: Adding --fit-ctx {fit_ctx_val}", file=sys.stderr)
            fit_target_val = self.launcher.fit_target.get().strip()
            if fit_target_val and fit_target_val != "1024":
                cmd.extend(["--fit-target", fit_target_val])
                print(f"DEBUG: Adding --fit-target {fit_target_val}", file=sys.stderr)
        else:
            cmd.extend(["--fit", "off"])
            print("DEBUG: Adding --fit off", file=sys.stderr)

    def _build_ik_llama_fit_args(self, cmd, exe_path):
        """Append ik_llama fit args, translating from the shared UI fields:
          - shared `fit_enabled` -> bare `--fit` flag (no on/off arg)
          - shared `fit_target`  -> `--fit-margin N` (closest equivalent)
          - shared `fit_ctx`     -> dropped (no equivalent in ik_llama)
        Each flag is probed against `<exe> --help` so older builds that
        predate ik_llama's fit support don't crash on an unknown arg.
        """
        if not self.launcher.fit_enabled.get():
            # ik_llama has no `--fit off` — fit is opt-in via the bare flag,
            # so leaving it unchecked just means emitting nothing.
            print("DEBUG: ik_llama fit disabled — omitting --fit flag", file=sys.stderr)
            return

        if not self._backend_supports_flag(exe_path, "--fit"):
            print(
                f"DEBUG: ik_llama build at {exe_path} does not advertise --fit; "
                f"skipping fit flags (upgrade ik_llama to use this feature)",
                file=sys.stderr,
            )
            return

        cmd.append("--fit")
        print("DEBUG: Adding --fit (ik_llama bare flag)", file=sys.stderr)

        if self.launcher.fit_ctx.get().strip():
            print(
                "DEBUG: ik_llama has no --fit-ctx equivalent; "
                "ignoring fit_ctx value",
                file=sys.stderr,
            )

        fit_target_val = self.launcher.fit_target.get().strip()
        if fit_target_val and fit_target_val != "1024":
            if self._backend_supports_flag(exe_path, "--fit-margin"):
                cmd.extend(["--fit-margin", fit_target_val])
                print(
                    f"DEBUG: Adding --fit-margin {fit_target_val} "
                    f"(mapped from fit_target)",
                    file=sys.stderr,
                )
            else:
                print(
                    f"DEBUG: ik_llama build at {exe_path} does not advertise "
                    f"--fit-margin; skipping fit_target mapping",
                    file=sys.stderr,
                )

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

        # --- Append mmproj if enabled and exists ---
        if self.launcher.mmproj_enabled.get():
            try:
                mmproj_file = None
                selected_mmproj_str = self.launcher.selected_mmproj_path.get().strip() if hasattr(self.launcher, "selected_mmproj_path") else ""
                if selected_mmproj_str:
                    selected_mmproj = Path(selected_mmproj_str).resolve()
                    if selected_mmproj.is_file():
                        mmproj_file = selected_mmproj
                        print(f"DEBUG: Using selected mmproj from dropdown: {mmproj_file.name}", file=sys.stderr)
                    else:
                        print(f"WARNING: Selected mmproj path is not a file: {selected_mmproj_str}", file=sys.stderr)

                # Fallback to auto-detection if no explicit selection is available.
                # Prefer candidates whose name contains the model stem so multi-model dirs don't cross-match.
                if mmproj_file is None:
                    model_dir = Path(model_full_path_str).parent
                    if model_dir.exists() and model_dir.is_dir():
                        # Normalize multipart suffixes before stripping .gguf (Path.stem leaves .gguf
                        # embedded on foo.gguf.partXofY, which weakens the substring match below).
                        normalized_name = Path(model_full_path_str).name.lower()
                        for pattern in (r"\.gguf\.part\d+of\d+$", r"-\d+-of-\d+\.gguf$"):
                            normalized_name = re.sub(pattern, ".gguf", normalized_name, flags=re.I)
                        model_stem_l = re.sub(r"\.gguf$", "", normalized_name, flags=re.I)
                        candidates = [
                            c for c in model_dir.iterdir()
                            if c.is_file() and "mmproj" in c.name.lower()
                        ]
                        preferred = next((c for c in candidates if model_stem_l and model_stem_l in c.name.lower()), None)
                        mmproj_file = preferred or (candidates[0] if candidates else None)

                if mmproj_file:
                    cmd.extend(["--mmproj", str(mmproj_file.resolve())])
                    print(f"DEBUG: Adding --mmproj {mmproj_file.name}", file=sys.stderr)
            except Exception as e:
                print(f"WARNING: Error scanning for mmproj files: {e}", file=sys.stderr)

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
        self.add_arg(cmd, "--threads-batch", self.launcher.threads_batch.get())              # Always pass (no default comparison)

        # Llama.cpp internal defaults: --batch-size=512, --ubatch-size=512
        # Always pass batch settings regardless of default values
        self.add_arg(cmd, "--batch-size", self.launcher.batch_size.get())
        self.add_arg(cmd, "--ubatch-size", self.launcher.ubatch_size.get())

        # Llama.cpp internal defaults: --ctx-size=2048, --seed=-1, --temp=0.8, --min-p=0.05
        self.add_arg(cmd, "--ctx-size", str(self.launcher.ctx_size.get()), "2048") # Use str() for int var
        self.add_arg(cmd, "--seed", self.launcher.seed.get(), "-1")
        self.add_arg(cmd, "--temp", self.launcher.temperature.get(), "0.8")
        self.add_arg(cmd, "--min-p", self.launcher.min_p.get(), "0.05")

        # --- Handle GPU arguments: Now ADDING BOTH if set ---
        tensor_split_val = self.launcher.tensor_split.get().strip()
        n_gpu_layers_val = self.launcher.n_gpu_layers.get().strip()

        # Add --tensor-split if the value is non-empty
        # Use add_arg which handles the non-empty check
        self.add_arg(cmd, "--tensor-split", tensor_split_val, "") # Add if non-empty string is provided by user

        # Add --n-gpu-layers if the value is non-empty AND not the default "0" string
        # This argument will now be added regardless of the --tensor-split value
        self.add_arg(cmd, "--n-gpu-layers", n_gpu_layers_val, "0")

        # --main-gpu is usually needed when offloading layers (either via --n-gpu-layers or --tensor-split)
        # It specifies which GPU is considered the "primary" one, often GPU 0.
        # Llama.cpp default is 0. Include --main-gpu if the user set a non-default value.
        main_gpu_val = self.launcher.main_gpu.get().strip()
        self.add_arg(cmd, "--main-gpu", main_gpu_val, "0")

        # Add --flash-attn flag if checked
        if self.launcher.flash_attn.get():
            if backend == "ik_llama":
                # ik_llama doesn't accept "on" after --flash-attn
                cmd.append("--flash-attn")
            else:
                # llama.cpp --flash-attn requires a value (on|off|auto)
                cmd.extend(["--flash-attn", "on"])

        # --- Fit Parameters ---
        # llama.cpp and ik_llama both support --fit, but with different CLI
        # surfaces:
        #   llama.cpp: `--fit on|off`, `--fit-ctx N`, `--fit-target N`
        #   ik_llama:  `--fit` (bare flag, no on/off), `--fit-margin N`
        #              (no --fit-ctx equivalent)
        # Both surfaces are runtime-probed so older builds that lack any of
        # these flags degrade gracefully instead of crashing on an unknown arg.
        if backend == "ik_llama":
            self._build_ik_llama_fit_args(cmd, exe_path)
        else:
            self._build_llama_cpp_fit_args(cmd)

        # Memory options
        self.add_arg(cmd, "--no-mmap", self.launcher.no_mmap.get()) # Omit if False (default)
        self.add_arg(cmd, "--mlock", self.launcher.mlock.get()) # Omit if False (default)
        self.add_arg(cmd, "--no-kv-offload", self.launcher.no_kv_offload.get()) # Omit if False (default)

        # Performance options
        self.add_arg(cmd, "--prio", self.launcher.prio.get(), "0") # Omit if 0 (default)
        self.add_arg(cmd, "--parallel", self.launcher.parallel.get(), "1") # Omit if 1 (default)

        # --- MoE CPU options ---
        self.add_arg(cmd, "--cpu-moe", self.launcher.cpu_moe.get()) # Omit if False (default)
        self.add_arg(cmd, "--n-cpu-moe", self.launcher.n_cpu_moe.get(), "") # Omit if empty (default)

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

        # --- Jinja rendering (server-side chat template engine) ---
        # --jinja is a standalone boolean flag on both llama-server and ik_llama-server.
        if self.launcher.jinja_enabled.get():
            cmd.append("--jinja")
            print("DEBUG: Adding --jinja", file=sys.stderr)

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
        ordered_selected_gpus = self.launcher.get_ordered_selected_gpus()
        detected_gpu_count = self.launcher.gpu_info.get("device_count", 0)
        selected_indices_str = ",".join(map(str, ordered_selected_gpus))

        # Only warn if GPUs were detected, the user selected a *subset*, and --tensor-split is not used.
        if detected_gpu_count > 0 and len(ordered_selected_gpus) > 0 and len(ordered_selected_gpus) < detected_gpu_count and not tensor_split_val:
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
        elif len(ordered_selected_gpus) > 0 and detected_gpu_count > 0:
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

    @staticmethod
    def _ps_escape_double_quoted(s):
        """Escape a string so it is safe inside a PowerShell double-quoted literal.

        In PowerShell double-quoted strings the backtick is the escape character
        (``\\`` → `` ` ``) and a literal double quote is written as `` `" ``. So
        backticks must be escaped FIRST, then double quotes — otherwise the
        backtick we inject to escape `"` gets double-escaped by the second pass
        and emerges as `` `` `` (literal backtick) plus an unescaped `"`, which
        prematurely terminates the string.
        """
        return s.replace('`', '``').replace('"', '`"')

    @staticmethod
    def _ps_quote_arg(arg):
        """Quote a single command-line argument for a PowerShell command line.

        Uses double quotes and escapes internal double quotes by doubling them
        (the convention PS expects when an arg is passed to a native exe), and
        escapes backticks by doubling them first so they don't double-escape
        into literal backticks.
        """
        escaped = arg.replace('`', '``').replace('"', '""')
        return f'"{escaped}"'

    def _build_ps_cmd_parts(self, cmd_list):
        """Build a list of PowerShell-quoted command tokens from ``cmd_list``.

        The first element is treated as the executable path and rendered with
        ``& "..."``. ``--chat-template`` followed by its value is single-quoted
        (with embedded single quotes doubled) so that backslashes, ``$``, ``
        ``` ``, etc. in the Jinja template are preserved verbatim.

        Any malformed ``--chat-template`` (missing value) raises ``ValueError``
        to signal the caller; upstream ``build_cmd`` is expected to have
        validated this already.
        """
        if not cmd_list:
            raise ValueError("cmd_list is empty")

        parts = []
        exe_posix = str(Path(cmd_list[0]).resolve().as_posix())
        parts.append(f'& "{self._ps_escape_double_quoted(exe_posix)}"')

        i = 1
        while i < len(cmd_list):
            current = cmd_list[i]
            if current == "--chat-template":
                if i + 1 >= len(cmd_list):
                    raise ValueError(
                        "Malformed cmd_list: --chat-template flag has no value"
                    )
                template_string = cmd_list[i + 1]
                escaped_template = template_string.replace("'", "''")
                parts.append("--chat-template")
                parts.append(f"'{escaped_template}'")
                i += 2
            else:
                parts.append(self._ps_quote_arg(current))
                i += 1

        return parts

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

    def _resolve_cuda_visible_devices_action(self):
        """Decide how generated scripts should handle ``CUDA_VISIBLE_DEVICES``.

        Centralising this keeps the four emit sites (live bash, live
        PowerShell, save-sh, save-ps1) in lock-step so they can't silently
        drift apart again.

        Returns one of:
          ``('export', '<indices>')`` — emit ``CUDA_VISIBLE_DEVICES=<indices>``.
          ``('unset',  None)``        — explicitly unset the variable so an
                                        inherited shell value can't leak in.
          ``('skip',   None)``        — no CUDA context at all; leave alone.

        Manual GPU mode maps to ``unset``: manual indices are synthetic
        (``0..N-1`` against the user's planning list) and have no relation to
        physical PCIe bus IDs. Passing them through would filter the wrong
        real devices on a CUDA-enabled host.
        """
        is_manual_mode = self.launcher.gpu_info.get("manual_mode", False)
        device_count = self.launcher.gpu_info.get("device_count", 0)

        if is_manual_mode:
            # Synthetic indices — never export them. Unset so any inherited
            # shell value doesn't silently filter real hardware.
            return ("unset", None)

        ordered_gpus = self.launcher.get_ordered_selected_gpus()
        if ordered_gpus:
            return ("export", ",".join(map(str, ordered_gpus)))

        if device_count > 0:
            # Real GPUs detected but user deselected all of them.
            return ("unset", None)

        # No CUDA context to manage.
        return ("skip", None)

    def launch_server(self):
        """Launch the llama-server with the configured settings."""
        cmd_list = self.build_cmd()
        if not cmd_list:
            # build_cmd already showed an error message
            return

        venv_path_str = self.launcher.venv_dir.get().strip()
        use_venv = bool(venv_path_str)

        tmp_path = None # Initialize tmp_path outside try/except/finally

        # Resolve how to handle CUDA_VISIBLE_DEVICES for the child process.
        # The action is shared across the Windows (PowerShell) and POSIX (bash)
        # live-launch branches below.
        cuda_action, cuda_devices_value = self._resolve_cuda_visible_devices_action()
        # For backward-compat with code that just checks "is there a value to
        # emit": the export branch sets a non-empty string; unset/skip get "".
        if cuda_action != "export":
            cuda_devices_value = ""

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

                    # --- Set CUDA_DEVICE_ORDER for consistent PCIe bus ordering ---
                    f.write('# Ensure GPU ordering matches PCIe bus order (consistent with nvidia-smi)\n')
                    f.write('$env:CUDA_DEVICE_ORDER="PCI_BUS_ID"\n\n')

                    # --- Add CUDA_VISIBLE_DEVICES action ---
                    if cuda_action == "export":
                        f.write(f'Write-Host "Setting CUDA_VISIBLE_DEVICES={cuda_devices_value}" -ForegroundColor DarkCyan\n')
                        f.write(f'$env:CUDA_VISIBLE_DEVICES="{cuda_devices_value}"\n\n')
                    elif cuda_action == "unset":
                         # GPUs detected but none selected, or manual GPU mode —
                         # either way, clear CUDA_VISIBLE_DEVICES so an inherited
                         # shell value can't silently re-enable filtered hardware.
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
                        # Escape backticks FIRST, then double-quotes. Doing it in the
                        # other order double-escapes any backtick inserted during the
                        # quote replacement (turning `" into ``", which PS parses as
                        # a literal backtick followed by an unescaped quote).
                        quoted_ps_act_path = f'"{self._ps_escape_double_quoted(ps_act_path)}"'

                        f.write(f'Write-Host "Activating virtual environment: {venv_path_str}" -ForegroundColor Cyan\n')
                        # Use try/catch to report activation errors but continue
                        f.write(f'try {{ . {quoted_ps_act_path} }} catch {{ Write-Warning "Failed to activate venv: $($_.Exception.Message)"; $global:LASTEXITCODE=1; Start-Sleep -Seconds 2 }}\n\n') # Use global:LASTEXITCODE and pause on error

                    f.write(f'Write-Host "Launching {backend.replace("_", "-")}-server..." -ForegroundColor Green\n')

                    # --- Build the command string for PowerShell using appropriate quoting ---
                    try:
                        ps_cmd_parts = self._build_ps_cmd_parts(cmd_list)
                    except ValueError as ve:
                        messagebox.showerror(
                            "Launch Error",
                            f"Internal error building launch command: {ve}"
                        )
                        print(f"ERROR: {ve}", file=sys.stderr)
                        if tmp_path is not None:
                            try: Path(tmp_path).unlink()
                            except OSError: pass
                        return
                    f.write(" ".join(ps_cmd_parts) + "\n\n")

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

                if use_venv:
                    venv_path = Path(venv_path_str).resolve()
                    act_script = venv_path / "bin" / "activate"
                    if not act_script.is_file():
                        messagebox.showerror("Error", f"Venv activation script not found:\n{act_script}")
                        return # Exit the launch process

                # --- Build the bash script content as an ordered list of commands
                # chained with ' && ', then append the server launch + pause trailer
                # with ';' so the trailer runs even if the server exits non-zero.
                # Building as a list avoids brittle .replace() surgery on literal
                # markers that could collide with user-supplied paths.
                commands = []

                if use_venv:
                    # ``act_script`` was validated above.
                    commands.append(f'source {shlex.quote(str(act_script))}')
                    commands.append('echo "Virtual environment activated."')

                # CUDA_DEVICE_ORDER is always set for consistent PCIe ordering
                commands.append('export CUDA_DEVICE_ORDER=PCI_BUS_ID')

                if cuda_action == "export":
                    commands.append(f'export CUDA_VISIBLE_DEVICES={cuda_devices_value}')
                    commands.append('echo "Setting CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"')
                elif cuda_action == "unset":
                    # GPUs detected but none selected, or manual GPU mode —
                    # either way, clear CUDA_VISIBLE_DEVICES so an inherited
                    # shell value can't silently re-enable filtered hardware.
                    commands.append('unset CUDA_VISIBLE_DEVICES')
                    commands.append('echo "Clearing CUDA_VISIBLE_DEVICES environment variable."')

                env_vars = self.launcher.env_vars_manager.get_enabled_env_vars()
                if env_vars:
                    commands.append('echo "Setting environmental variables..."')
                    for var_name, var_value in env_vars.items():
                        # Escape the four metacharacters that bash processes
                        # inside a ``"..."`` string: ``\``, ``"``, ``$``, and
                        # backtick. Order matters — backslash first so the
                        # backslashes we add for the other three don't get
                        # doubled again on a second pass.
                        #
                        # Embedded newlines / carriage returns are deliberately
                        # NOT escaped: bash double-quoted strings allow literal
                        # newlines in the value, so ``export VAR="foo<NL>bar"``
                        # is syntactically valid and round-trips the user's
                        # multi-line value intact. Replacing ``\n`` with ``\\n``
                        # would corrupt the value because bash ``""`` does not
                        # process ``\n`` as a newline escape — the user would
                        # see a literal two-character ``\n`` in their env var.
                        # See tests/launchers/test_shell_safety.py
                        # ``TestEnvVarExportEscaping`` for the regression.
                        escaped_value = (
                            var_value.replace('\\', '\\\\')
                                     .replace('"', '\\"')
                                     .replace('$', '\\$')
                                     .replace('`', '\\`')
                        )
                        commands.append(f'export {var_name}="{escaped_value}"')

                commands.append('echo "Launching server..."')
                commands.append(server_command_str)

                # The launcher command is joined with ' && ' for fail-fast. After it,
                # attach the exit-status capture + pause trailer with ';'.
                # Using </dev/tty ensures read prompts even if stdout is redirected.
                pause_trailer = (
                    ' ; command_status=$? ; '
                    'if [[ -t 1 || $command_status -ne 0 ]]; then '
                    'read -rp "Press Enter to close..." </dev/tty ; fi ; '
                    'exit $command_status'
                )
                full_script_content = " && ".join(commands) + pause_trailer

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

    def _get_launchers_dir(self):
        """Return the launchers/ output dir, falling back to repo root if creation fails."""
        repo_root = Path(__file__).parent.parent
        launchers_dir = repo_root / "launchers"
        try:
            launchers_dir.mkdir(parents=True, exist_ok=True)
            return launchers_dir
        except OSError as exc:
            messagebox.showwarning(
                "Launcher Directory Warning",
                f"Could not create launcher directory:\n{launchers_dir}\n\n"
                f"Falling back to repository root.\nError: {exc}"
            )
            return repo_root

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

        launchers_dir = self._get_launchers_dir()
        path = filedialog.asksaveasfilename(defaultextension=".ps1",
                                            initialfile=default_name,
                                            initialdir=str(launchers_dir),
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

                # --- Set CUDA_DEVICE_ORDER for consistent PCIe bus ordering ---
                fh.write('# Ensure GPU ordering matches PCIe bus order (consistent with nvidia-smi)\n')
                fh.write('$env:CUDA_DEVICE_ORDER="PCI_BUS_ID"\n\n')

                # --- Add CUDA_VISIBLE_DEVICES action ---
                cuda_action, cuda_devices_value = self._resolve_cuda_visible_devices_action()
                if cuda_action == "export":
                    fh.write(f'Write-Host "Setting CUDA_VISIBLE_DEVICES={cuda_devices_value}" -ForegroundColor DarkCyan\n')
                    fh.write(f'$env:CUDA_VISIBLE_DEVICES="{cuda_devices_value}"\n\n')
                elif cuda_action == "unset":
                     # GPUs detected but none selected, or manual GPU mode —
                     # either way, clear CUDA_VISIBLE_DEVICES to avoid silently
                     # filtering real hardware via synthetic/stale indices.
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
                            quoted_ps_act_path = f'"{self._ps_escape_double_quoted(ps_act_path)}"'

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
                try:
                    ps_cmd_parts = self._build_ps_cmd_parts(cmd_list)
                except ValueError as ve:
                    messagebox.showerror(
                        "Script Save Error",
                        f"Internal error building command: {ve}"
                    )
                    print(f"ERROR: {ve}", file=sys.stderr)
                    return
                fh.write(" ".join(ps_cmd_parts) + "\n\n")

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

        launchers_dir = self._get_launchers_dir()
        path = filedialog.asksaveasfilename(defaultextension=".sh",
                                            initialfile=default_name,
                                            initialdir=str(launchers_dir),
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

                # --- Set CUDA_DEVICE_ORDER for consistent PCIe bus ordering ---
                fh.write('# Ensure GPU ordering matches PCIe bus order (consistent with nvidia-smi)\n')
                fh.write('export CUDA_DEVICE_ORDER=PCI_BUS_ID\n\n')

                # --- Add CUDA_VISIBLE_DEVICES action ---
                cuda_action, cuda_devices_value = self._resolve_cuda_visible_devices_action()
                if cuda_action == "export":
                    fh.write(f'echo "Setting CUDA_VISIBLE_DEVICES={cuda_devices_value}"\n')
                    fh.write(f'export CUDA_VISIBLE_DEVICES="{cuda_devices_value}"\n\n')
                elif cuda_action == "unset":
                     # GPUs detected but none selected, or manual GPU mode —
                     # either way, clear CUDA_VISIBLE_DEVICES to avoid silently
                     # filtering real hardware via synthetic/stale indices.
                     fh.write('echo "Clearing CUDA_VISIBLE_DEVICES environment variable."\n')
                     fh.write('unset CUDA_VISIBLE_DEVICES\n\n')

                # --- Add Environmental Variables ---
                env_vars = self.launcher.env_vars_manager.get_enabled_env_vars()
                if env_vars:
                    fh.write('echo "Setting environmental variables..."\n')
                    for var_name, var_value in env_vars.items():
                        # Escape the four metacharacters bash processes inside
                        # ``"..."``: backslash FIRST (so the backslashes we
                        # insert for the rest don't get doubled), then ``"``,
                        # ``$``, and backtick. Newlines deliberately pass
                        # through — see the matching block in launch_server()
                        # and tests/launchers/test_shell_safety.py
                        # ``TestEnvVarExportEscaping``.
                        escaped_value = (
                            var_value.replace('\\', '\\\\')
                                     .replace('"', '\\"')
                                     .replace('$', '\\$')
                                     .replace('`', '\\`')
                        )
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
