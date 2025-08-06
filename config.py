#!/usr/bin/env python3
import json
import os
import re
import sys
from pathlib import Path
from tkinter import messagebox, filedialog
from datetime import datetime


class ConfigManager:
    """Configuration management for LLaMa.cpp Server Launcher."""
    
    def __init__(self, launcher_instance):
        """Initialize with reference to the main launcher instance."""
        self.launcher = launcher_instance
    
    def generate_default_config_name(self):
        """Generates a default configuration name based on current settings."""
        print("DEBUG: Generating default config name...", file=sys.stderr)
        parts = []

        # 1. Model Name
        model_path_str = self.launcher.model_path.get().strip()
        if model_path_str:
            try:
                # Use the selected model name from the listbox if available,
                # otherwise fall back to the path stem.
                selected_name = ""
                try:
                    if hasattr(self.launcher, 'model_listbox'):
                        sel = self.launcher.model_listbox.curselection()
                        if sel:
                            selected_name = self.launcher.model_listbox.get(sel[0])
                except Exception:
                    pass  # UI might not be fully initialized

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
            "cache_type_v":  "f16", # Defaults to same as K cache type
            "threads":       str(self.launcher.logical_cores), # Llama.cpp default for threads is logical cores
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
            "cpu_moe":       False, # Default for --cpu-moe flag
            "n_cpu_moe":     "",    # Default for --n-cpu-moe (empty)
    
            # Chat template parameters and custom parameters are deliberately excluded from default name generation
        }

        current_params = {
            "cache_type_k":  self.launcher.cache_type_k.get().strip(),
            "cache_type_v":  self.launcher.cache_type_v.get().strip(),
            "threads":       self.launcher.threads.get().strip(),
            "threads_batch": self.launcher.threads_batch.get().strip(),
            "batch_size":    self.launcher.batch_size.get().strip(),
            "ubatch_size":   self.launcher.ubatch_size.get().strip(),
            "ctx_size":      self.launcher.ctx_size.get(), # int
            "seed":          self.launcher.seed.get().strip(),
            "temperature":   self.launcher.temperature.get().strip(),
            "min_p":         self.launcher.min_p.get().strip(),
            "n_gpu_layers":  self.launcher.n_gpu_layers.get().strip(), # String value from entry
            "tensor_split":  self.launcher.tensor_split.get().strip(),
            "main_gpu":      self.launcher.main_gpu.get().strip(),
            "prio":          self.launcher.prio.get().strip(),
            "ignore_eos":    self.launcher.ignore_eos.get(), # bool
            "n_predict":     self.launcher.n_predict.get().strip(),
            "flash_attn":    self.launcher.flash_attn.get(), # bool
            "no_mmap":       self.launcher.no_mmap.get(),   # bool
            "mlock":         self.launcher.mlock.get(),    # bool
            "no_kv_offload": self.launcher.no_kv_offload.get(), # bool
            "cpu_moe":       self.launcher.cpu_moe.get(),  # bool
            "n_cpu_moe":     self.launcher.n_cpu_moe.get().strip(),

        }

        # Add non-default parameters to name parts
        for key, current_val in current_params.items():
            default_val = default_params_for_name.get(key) # Get the default value used for name generation

            # Special handling for GPU Layers: use the internal integer value for comparison effect
            if key == "n_gpu_layers":
                 # Use the integer value after clamping, not the raw entry string
                 gpu_layers_int = self.launcher.n_gpu_layers_int.get()
                 max_layers = self.launcher.max_gpu_layers.get()
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
             
                         "ignore_eos": "no-eos",
                         "cpu_moe": "cpu-moe",
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
                          "cache_type_v": "vv",
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

        # Always update the config name field with the generated name in real-time
        # This provides immediate feedback about what the config represents
        current_config_name = self.launcher.config_name.get().strip()
        print(f"DEBUG: Current config name: '{current_config_name}'", file=sys.stderr)
        
        # Always update except for very specific custom names that don't look auto-generated
        should_update = True
        
        # Only preserve if it's a very obviously custom name:
        # - Not empty, not "default_config", not "default", not "model"
        # - Doesn't contain any auto-gen patterns like _gpu=, _ctx=, _th=, etc.
        # - Doesn't start with any model name patterns
        # - Must be at least 3 characters and contain letters (to avoid preserving junk)
        if (current_config_name and 
            len(current_config_name) >= 3 and
            any(c.isalpha() for c in current_config_name) and
            current_config_name not in ["default_config", "default", "model"] and
            not any(pattern in current_config_name for pattern in ["_gpu=", "_ctx=", "_temp=", "_batch=", "_threads=", "_th=", "_tb=", "_min_p=", "_seed=", "_n_predict="]) and
            not current_config_name.startswith(("default", "model", parts[0] if parts else ""))):
            should_update = False
            print(f"DEBUG: Detected truly custom name '{current_config_name}', will preserve", file=sys.stderr)

        if should_update and generated_name != current_config_name:
            self.launcher.config_name.set(generated_name)
            print("DEBUG: Updated config_name variable in real-time.", file=sys.stderr)
        elif not should_update:
             print("DEBUG: Preserved custom config name.", file=sys.stderr)
        else:
             print("DEBUG: Generated name same as current, no update needed.", file=sys.stderr)
        
        return generated_name


    def update_default_config_name_if_needed(self, *args):
        """Traced callback for variables that influence the default config name."""
        # This trace function is bound to variables that influence the generated config name.
        # It's called whenever those variables change.
        # We only want to regenerate and update the config name if the user hasn't
        # already manually set a custom name.
        # The generate_default_config_name function already contains the logic
        # to decide whether to overwrite the current self.launcher.config_name value.
        # So we just call it here.
        # Use after(1) to prevent recursive trace calls on config_name update
        self.launcher.root.after(1, self.generate_default_config_name)


    def current_cfg(self):
        """Get current configuration as dictionary."""
        gpu_layers_to_save = self.launcher.n_gpu_layers.get().strip()
        # Ensure selected_gpus in app_settings is up-to-date before saving
        self.launcher.app_settings["selected_gpus"] = [i for i, v in enumerate(self.launcher.gpu_vars) if v.get()]

        # Construct the configuration dictionary
        cfg = {
            "llama_cpp_dir": self.launcher.llama_cpp_dir.get(),
            "ik_llama_dir":  self.launcher.ik_llama_dir.get(),
            "venv_dir":      self.launcher.venv_dir.get(),
            "model_path":    self.launcher.model_path.get(),
            "cache_type_k":  self.launcher.cache_type_k.get(),
            "cache_type_v":  self.launcher.cache_type_v.get(),
            "threads":       self.launcher.threads.get(), # Save the user-set value
            "threads_batch": self.launcher.threads_batch.get(), # Save the user-set value
            "batch_size":    self.launcher.batch_size.get(), # Save the user-set value
            "ubatch_size":   self.launcher.ubatch_size.get(), # Save the user-set value
            "n_gpu_layers":  gpu_layers_to_save, # Save the string value (can be -1)
            "no_mmap":       self.launcher.no_mmap.get(),

            "prio":          self.launcher.prio.get(),
            "temperature":   self.launcher.temperature.get(),
            "min_p":         self.launcher.min_p.get(),
            "ctx_size":      self.launcher.ctx_size.get(),
            "seed":          self.launcher.seed.get(),
            "flash_attn":    self.launcher.flash_attn.get(),
            "tensor_split":  self.launcher.tensor_split.get().strip(),
            "main_gpu":      self.launcher.main_gpu.get(),
            "mlock":         self.launcher.mlock.get(),
            "no_kv_offload": self.launcher.no_kv_offload.get(),
            "host":          self.launcher.host.get(),
            "port":          self.launcher.port.get(),
            # --- Backend Selection ---
            "backend_selection": self.launcher.backend_selection.get(),
            # --- NEW: Add new parameters to config ---
            "ignore_eos":    self.launcher.ignore_eos.get(),
            "n_predict":     self.launcher.n_predict.get(),
            # --- MoE CPU parameters ---
            "cpu_moe":       self.launcher.cpu_moe.get(),
            "n_cpu_moe":     self.launcher.n_cpu_moe.get(),
            # --- CHANGES FOR JSON TEMPLATES / DEFAULT OPTION ---
            # Save the new template source variable
            "template_source": self.launcher.template_source.get(),
            "predefined_template_name": self.launcher.predefined_template_name.get(),
            "custom_template_string": self.launcher.custom_template_string.get(),
            # --- NEW: Save Custom Parameters ---
            "custom_parameters": self.launcher.custom_parameters_list, # Save the list of strings
        }

        # Include selected_gpus directly in the config dictionary for easier loading from config tab
        # This is redundant with app_settings, but keeps config self-contained for this tab.
        cfg["gpu_indices"] = self.launcher.app_settings.get("selected_gpus", [])

        # Add environmental variables configuration
        cfg.update(self.launcher.env_vars_manager.save_to_config())
        
        # Add ik_llama specific configuration
        cfg.update(self.launcher.ik_llama_tab.save_to_config())

        return cfg

    def load_configuration(self):
        """Load selected configuration from the listbox."""
        if not self.launcher.config_listbox.curselection():
            return messagebox.showerror("Error","Select a configuration from the list to load.")
        name = self.launcher.config_listbox.get(self.launcher.config_listbox.curselection())
        cfg  = self.launcher.saved_configs.get(name)
        if not cfg:
             messagebox.showerror("Error", f"Configuration '{name}' data not found.")
             return

        # Load simple variables first
        self.launcher.llama_cpp_dir.set(cfg.get("llama_cpp_dir",""))
        self.launcher.ik_llama_dir.set(cfg.get("ik_llama_dir",""))
        self.launcher.venv_dir.set(cfg.get("venv_dir","")) # Setting this triggers the venv trace -> flash attn check
        self.launcher.cache_type_k.set(cfg.get("cache_type_k","f16"))
        self.launcher.cache_type_v.set(cfg.get("cache_type_v","f16"))
        # Load parameters, providing defaults for backward compatibility with older configs
        # Default to the *current* detected cores if not in the config for threads
        self.launcher.threads.set(cfg.get("threads", str(self.launcher.physical_cores))) # Default to detected physical cores
        self.launcher.threads_batch.set(cfg.get("threads_batch", str(self.launcher.logical_cores))) # Default to detected logical cores
        self.launcher.batch_size.set(cfg.get("batch_size", "512")) # Default to llama.cpp 512
        self.launcher.ubatch_size.set(cfg.get("ubatch_size", "512")) # Default to llama.cpp 512

        self.launcher.no_mmap.set(cfg.get("no_mmap",False))

        self.launcher.prio.set(cfg.get("prio","0"))
        self.launcher.temperature.set(cfg.get("temperature","0.8"))
        self.launcher.min_p.set(cfg.get("min_p","0.05"))
        ctx = cfg.get("ctx_size", 2048)
        self.launcher.ctx_size.set(ctx)
        self.launcher._sync_ctx_display(ctx) # Manually sync display
        self.launcher.seed.set(cfg.get("seed","-1"))
        self.launcher.flash_attn.set(cfg.get("flash_attn",False))
        self.launcher.tensor_split.set(cfg.get("tensor_split","").strip()) # Ensure strip on load too
        self.launcher.main_gpu.set(cfg.get("main_gpu","0"))
        self.launcher.mlock.set(cfg.get("mlock",False))
        self.launcher.no_kv_offload.set(cfg.get("no_kv_offload",False))
        self.launcher.host.set(cfg.get("host", self.launcher.app_settings.get("host", "127.0.0.1"))) # --host
        self.launcher.port.set(cfg.get("port", self.launcher.app_settings.get("port", "8080"))) # --port
        self.launcher.config_name.set(name) # Set the config name entry

        # --- Backend Selection ---
        self.launcher.backend_selection.set(cfg.get("backend_selection", "llama.cpp"))

        # --- NEW: Load new parameters ---
        self.launcher.ignore_eos.set(cfg.get("ignore_eos", False))
        self.launcher.n_predict.set(cfg.get("n_predict", "-1")) # Default -1 for backward compatibility
        # --- MoE CPU parameters ---
        self.launcher.cpu_moe.set(cfg.get("cpu_moe", False))
        self.launcher.n_cpu_moe.set(cfg.get("n_cpu_moe", ""))
        # --- NEW: Load Custom Parameters ---
        # Default to empty list [] for backward compatibility with older configs
        self.launcher.custom_parameters_list = cfg.get("custom_parameters", [])
        self.launcher._update_custom_parameters_listbox() # Update the GUI listbox

        # Load environmental variables configuration
        self.launcher.env_vars_manager.load_from_config(cfg)
        
        # Load ik_llama specific configuration
        self.launcher.ik_llama_tab.load_from_config(cfg)

        # --- CHANGES FOR JSON TEMPLATES / DEFAULT OPTION ---
        # Load template parameters
        # Default the source to 'default' for backward compatibility
        loaded_source = cfg.get("template_source", "default")
        self.launcher.template_source.set(loaded_source)

        # Default the predefined name to the *first* key in _all_templates if not found
        # This handles cases where the saved name might no longer exist in _all_templates
        default_predefined_key = list(self.launcher._all_templates.keys())[0] if self.launcher._all_templates else ""
        self.launcher.predefined_template_name.set(cfg.get("predefined_template_name", default_predefined_key))

        self.launcher.custom_template_string.set(cfg.get("custom_template_string", ""))

        # Update UI state and display based on loaded values
        # The trace on self.launcher.template_source should trigger _update_template_controls_state and _update_effective_template_display
        # No need to call explicitly here if trace is reliable.

        # Load GPU selections - This needs to update the checkboxes
        # Check for the 'gpu_indices' key directly in the config dictionary first
        loaded_gpu_indices = cfg.get("gpu_indices", self.launcher.app_settings.get("selected_gpus", [])) # Fallback to app_settings key if old config format
        # Store loaded indices in app_settings *before* updating checkboxes
        self.launcher.app_settings["selected_gpus"] = loaded_gpu_indices
        self.launcher._update_gpu_checkboxes() # This will set the checkboxes according to self.launcher.app_settings["selected_gpus"]
        # _update_gpu_checkboxes also triggers _update_recommendations


        # Load n_gpu_layers - This interacts with model analysis results
        loaded_gpu_layers_str = cfg.get("n_gpu_layers","0")
        self.launcher.n_gpu_layers.set(loaded_gpu_layers_str)
        try:
             val = int(loaded_gpu_layers_str)
             # Use _set_gpu_layers to set the int var and sync entry, clamping based on current max
             # (max_layers will be 0 initially if model not loaded yet)
             self.launcher._set_gpu_layers(val)
        except ValueError:
             self.launcher.n_gpu_layers.set("0")
             self.launcher._set_gpu_layers(0)

        # Load Model Path - This will trigger model selection logic and analysis
        loaded_model_path_str = cfg.get("model_path", "")
        # Set the variable first, then attempt to select in the listbox
        self.launcher.model_path.set(loaded_model_path_str)
        self.launcher.model_listbox.selection_clear(0, "end") # Clear previous visual selection

        selected_idx = -1
        if loaded_model_path_str:
            try:
                loaded_model_path_obj = Path(loaded_model_path_str).resolve() # Resolve the saved path before lookup
                found_display_name = None
                # Find the display name associated with the saved path in the *currently found* models (which are resolved)
                for display_name, full_path in self.launcher.found_models.items():
                    if full_path == loaded_model_path_obj:
                        found_display_name = display_name
                        break
                # If found, get its index in the current listbox (which is sorted by display name)
                listbox_items = self.launcher.model_listbox.get(0, "end")
                if found_display_name and found_display_name in listbox_items: # Check if the display name exists in the current listbox items
                    selected_idx = listbox_items.index(found_display_name)
            except (ValueError, OSError, IndexError):
                 pass # Handle potential errors with old paths or listbox state

        if selected_idx != -1:
             # Select it visually and trigger the selection handler (_on_model_selected)
             # Using after(10, ...) gives the UI a moment to update before selection
             self.launcher.root.after(10, self.launcher._select_model_in_listbox, selected_idx)
        else:
             # If the model was not found in the current scan results
             self.launcher.model_path.set("") # Clear model path variable
             self.launcher._reset_model_info_display()
             self.launcher._reset_gpu_layer_controls(keep_entry_enabled=True) # Keep entry enabled if model not found
             self.launcher.current_model_analysis = {} # Clear analysis data
             self.launcher._update_recommendations() # Update recommendations based on no model
             self.generate_default_config_name() # Generate default name for no model state
             if loaded_model_path_str:
                  messagebox.showwarning("Model Not Found", f"The model from the config ('{Path(loaded_model_path_str).name if loaded_model_path_str else 'N/A'}') was not found in the current list.\nPlease ensure its directory is added and scanned, then select a model manually.")


        messagebox.showinfo("Loaded", f"Configuration '{name}' applied.")

    def delete_configuration(self):
        """Delete selected configuration(s) from the listbox."""
        selected_indices = self.launcher.config_listbox.curselection()
        if not selected_indices:
            return messagebox.showerror("Error","Select one or more configurations to delete.")
        
        # Get the names of selected configurations
        selected_names = []
        for index in selected_indices:
            name = self.launcher.config_listbox.get(index)
            selected_names.append(name)
        
        if not selected_names:
            return messagebox.showerror("Error", "No valid configurations selected for deletion.")
        
        # Create confirmation message
        if len(selected_names) == 1:
            confirm_msg = f"Are you sure you want to delete the configuration '{selected_names[0]}'?"
            result_msg = f"Configuration '{selected_names[0]}' deleted."
        else:
            config_list = "\n".join(f"  • {name}" for name in selected_names)
            confirm_msg = f"Are you sure you want to delete {len(selected_names)} configurations?\n\n{config_list}"
            result_msg = f"Successfully deleted {len(selected_names)} configurations."
        
        # Ask for confirmation
        if messagebox.askyesno("Confirm Deletion", confirm_msg):
            # Delete the configurations
            deleted_count = 0
            for name in selected_names:
                if name in self.launcher.saved_configs:
                    self.launcher.saved_configs.pop(name, None)
                    deleted_count += 1
            
            if deleted_count > 0:
                self.launcher._save_configs()
                self.launcher._update_config_listbox()
                messagebox.showinfo("Deleted", result_msg)
            else:
                messagebox.showerror("Error", "No configurations were found to delete.")

    def on_config_selected(self):
        """Callback when a configuration is selected in the config listbox."""
        selection = self.launcher.config_listbox.curselection()
        if not selection:
            return
        
        # For single selection, update the config name field
        # For multiple selections, show count in the config name field
        if len(selection) == 1:
            name = self.launcher.config_listbox.get(selection[0])
            self.launcher.config_name.set(name)
        else:
            # Multiple selections - show count
            self.launcher.config_name.set(f"({len(selection)} configs selected)")

    def update_config_listbox(self):
        """Update the configuration listbox with saved configurations."""
        current_selection = self.launcher.config_listbox.curselection()
        selected_name = None
        if current_selection:
            selected_name = self.launcher.config_listbox.get(current_selection[0])
        self.launcher.config_listbox.delete(0, "end")
        # Filter out None keys and convert to strings before sorting
        sorted_names = sorted([str(name) for name in self.launcher.saved_configs.keys() if name is not None])
        for cfg_name in sorted_names:
            self.launcher.config_listbox.insert("end", cfg_name)
        if selected_name in sorted_names:
            try:
                 new_index = sorted_names.index(selected_name)
                 self.launcher.config_listbox.selection_set(new_index)
                 self.launcher.config_listbox.activate(new_index)
                 self.launcher.config_listbox.see(new_index)
            except ValueError: pass

    def export_configurations(self):
        """Export selected configurations to a JSON file."""
        
        # Get selected configurations
        selected_indices = self.launcher.config_listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Error", "Please select at least one configuration to export.")
            return
        
        # Get the names of selected configurations
        selected_configs = {}
        for index in selected_indices:
            config_name = self.launcher.config_listbox.get(index)
            if config_name in self.launcher.saved_configs:
                selected_configs[config_name] = self.launcher.saved_configs[config_name]
        
        if not selected_configs:
            messagebox.showerror("Error", "No valid configurations selected for export.")
            return
        
        # Ask user for export file location
        default_filename = f"llama_configs_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        export_path = filedialog.asksaveasfilename(
            title="Export Configurations",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=default_filename
        )
        
        if not export_path:
            return  # User cancelled
        
        try:
            # Create export data structure
            export_data = {
                "export_info": {
                    "exported_at": datetime.now().isoformat(),
                    "source": "LLaMa.cpp Server Launcher",
                    "version": "1.0",
                    "config_count": len(selected_configs)
                },
                "configs": selected_configs
            }
            
            # Write to file
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            messagebox.showinfo("Export Successful", 
                              f"Successfully exported {len(selected_configs)} configuration(s) to:\n{export_path}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export configurations:\n{str(e)}")

    def import_configurations(self):
        """Import configurations from a JSON file."""
        
        # Ask user for import file
        import_path = filedialog.askopenfilename(
            title="Import Configurations",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not import_path:
            return  # User cancelled
        
        try:
            # Read and parse the JSON file
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Handle different import formats
            configs_to_import = {}
            
            if isinstance(import_data, dict):
                # Check if it's our export format
                if "configs" in import_data and isinstance(import_data["configs"], dict):
                    configs_to_import = import_data["configs"]
                    
                    # Show import info if available
                    if "export_info" in import_data:
                        info = import_data["export_info"]
                        config_count = info.get("config_count", len(configs_to_import))
                        exported_at = info.get("exported_at", "Unknown")
                        print(f"DEBUG: Importing {config_count} configs exported at {exported_at}", file=sys.stderr)
                
                # Check if it's a direct configs format (like our main config file)
                elif "configs" in import_data and isinstance(import_data["configs"], dict):
                    configs_to_import = import_data["configs"]
                    
                # Check if the entire file is just configs (legacy format)
                else:
                    # Assume each top-level key is a config name
                    # Validate that values look like config objects
                    valid_configs = {}
                    for key, value in import_data.items():
                        if isinstance(value, dict) and any(setting in value for setting in 
                                                         ["model_path", "llama_cpp_dir", "n_gpu_layers", "ctx_size"]):
                            valid_configs[key] = value
                    
                    if valid_configs:
                        configs_to_import = valid_configs
            
            if not configs_to_import:
                messagebox.showerror("Import Error", 
                                   "No valid configurations found in the selected file.\n\n"
                                   "Expected format: JSON file with configuration objects.")
                return
            
            # Check for conflicts and get user preferences
            conflicts = []
            new_configs = []
            
            for config_name in configs_to_import.keys():
                if config_name in self.launcher.saved_configs:
                    conflicts.append(config_name)
                else:
                    new_configs.append(config_name)
            
            # Show preview dialog
            import_summary = f"Found {len(configs_to_import)} configuration(s) to import:\n\n"
            
            if new_configs:
                import_summary += f"New configurations ({len(new_configs)}):\n"
                for name in new_configs[:5]:  # Show first 5
                    import_summary += f"  • {name}\n"
                if len(new_configs) > 5:
                    import_summary += f"  ... and {len(new_configs) - 5} more\n"
                import_summary += "\n"
            
            if conflicts:
                import_summary += f"Configurations that will be overwritten ({len(conflicts)}):\n"
                for name in conflicts[:5]:  # Show first 5
                    import_summary += f"  • {name}\n"
                if len(conflicts) > 5:
                    import_summary += f"  ... and {len(conflicts) - 5} more\n"
                import_summary += "\n"
            
            import_summary += "Do you want to proceed with the import?"
            
            if not messagebox.askyesno("Confirm Import", import_summary):
                return
            
            # Perform the import
            imported_count = 0
            for config_name, config_data in configs_to_import.items():
                try:
                    # Basic validation of config data
                    if not isinstance(config_data, dict):
                        print(f"WARNING: Skipping invalid config '{config_name}' - not a dictionary", file=sys.stderr)
                        continue
                    
                    # Import the configuration
                    self.launcher.saved_configs[config_name] = config_data
                    imported_count += 1
                    
                except Exception as e:
                    print(f"WARNING: Failed to import config '{config_name}': {e}", file=sys.stderr)
            
            if imported_count > 0:
                # Save the updated configurations
                self.launcher._save_configs()
                # Update the listbox
                self.update_config_listbox()
                
                messagebox.showinfo("Import Successful", 
                                  f"Successfully imported {imported_count} configuration(s).")
            else:
                messagebox.showerror("Import Error", "No configurations were successfully imported.")
                
        except json.JSONDecodeError as e:
            messagebox.showerror("Import Error", f"Invalid JSON file:\n{str(e)}")
        except FileNotFoundError:
            messagebox.showerror("Import Error", f"File not found:\n{import_path}")
        except Exception as e:
            messagebox.showerror("Import Error", f"Failed to import configurations:\n{str(e)}")

    def get_config_path(self):
        """Get the configuration file path, with fallback handling."""
        local_path = Path("llama_cpp_launcher_configs.json") # Renamed slightly to avoid potential clashes
        print(f"DEBUG: Checking local config path FULL PATH: {local_path.resolve()}", file=sys.stderr)
        print(f"DEBUG: Current working directory: {Path.cwd()}", file=sys.stderr)
        try:
            # Check if we can write to the current directory
            # Check if a config file exists and is empty (possibly from a failed previous run)
            # If empty, we can safely delete it and use the local path.
            if local_path.exists() and local_path.stat().st_size == 0:
                 try: local_path.unlink()
                 except OSError: pass # Ignore if delete fails

            # Check write permissions AFTER cleanup attempt
            if os.access(".", os.W_OK):
                 print(f"DEBUG: Using local config path FULL PATH: {local_path.resolve()}", file=sys.stderr)
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
                print(f"DEBUG: Using fallback config path FULL PATH: {fallback_path.resolve()}", file=sys.stderr)
                return fallback_path
            except Exception as e_fallback:
                 print(f"CRITICAL ERROR: Could not use local config path or fallback config path {fallback_dir}. Configuration saving/loading is disabled. Error: {e_fallback}", file=sys.stderr)
                 messagebox.showerror("Config Error", f"Failed to set up configuration directory.\nSaving/loading configurations is disabled.\nError: {e_fallback}")
                 # Return a dummy non-existent path to prevent errors later
                 return Path("/dev/null") if sys.platform != "win32" else Path("NUL") # Use platform-appropriate null device

    def load_saved_configs(self):
        """Load saved configurations from file."""
        if not self.launcher.config_path.exists() or not self.launcher.config_path.is_file() or self.launcher.config_path.name in ("null", "NUL"):
             print(f"DEBUG: No config file found at: {self.launcher.config_path.resolve()} or config saving is disabled. Using default settings.", file=sys.stderr)
             return

        print(f"DEBUG: Loading config from FULL PATH: {self.launcher.config_path.resolve()}", file=sys.stderr)
        try:
            data = json.loads(self.launcher.config_path.read_text(encoding="utf-8"))
            # Load configs and filter out any None keys
            raw_configs = data.get("configs", {})
            self.launcher.saved_configs = {k: v for k, v in raw_configs.items() if k is not None}
            loaded_app_settings = data.get("app_settings", {})
            print(f"DEBUG: Found {len(self.launcher.saved_configs)} saved configurations in config file", file=sys.stderr)
            print(f"DEBUG: Loaded saved config names: {list(self.launcher.saved_configs.keys())}", file=sys.stderr)
            print(f"DEBUG: Loading app settings from {self.launcher.config_path.resolve()}: {loaded_app_settings}", file=sys.stderr)
            self.launcher.app_settings.update(loaded_app_settings)
            # Ensure model_list_height is a valid int
            if not isinstance(self.launcher.app_settings.get("model_list_height"), int):
                self.launcher.app_settings["model_list_height"] = 8
            # Ensure selected_gpus is a list
            if not isinstance(self.launcher.app_settings.get("selected_gpus"), list):
                 self.launcher.app_settings["selected_gpus"] = []
            # Ensure custom_parameters is a list
            if not isinstance(self.launcher.app_settings.get("custom_parameters"), list):
                 self.launcher.app_settings["custom_parameters"] = []

            # Filter selected_gpus to only include indices of currently detected GPUs
            valid_gpu_indices = {gpu['id'] for gpu in self.launcher.detected_gpu_devices}
            self.launcher.app_settings["selected_gpus"] = [idx for idx in self.launcher.app_settings["selected_gpus"] if idx in valid_gpu_indices]

            # Load custom parameters into the internal list
            self.launcher.custom_parameters_list = self.launcher.app_settings.get("custom_parameters", [])

            # Load manual GPU settings
            self.launcher.manual_gpu_mode.set(self.launcher.app_settings.get("manual_gpu_mode", False))
            self.launcher.manual_gpu_count.set(self.launcher.app_settings.get("manual_gpu_count", "1"))
            self.launcher.manual_gpu_vram.set(self.launcher.app_settings.get("manual_gpu_vram", "8.0"))
            # Load new manual GPU list format with fallback to legacy
            self.launcher.manual_gpu_list = self.launcher.app_settings.get("manual_gpu_list", [])

            # Load manual model settings
            self.launcher.manual_model_mode.set(self.launcher.app_settings.get("manual_model_mode", False))
            self.launcher.manual_model_layers.set(self.launcher.app_settings.get("manual_model_layers", "32"))
            self.launcher.manual_model_size_gb.set(self.launcher.app_settings.get("manual_model_size_gb", "7.0"))

            # Ensure port and host are set from app_settings
            if "port" in self.launcher.app_settings:
                print(f"DEBUG: Setting port from app_settings: {self.launcher.app_settings['port']}") # Add debug print
                self.launcher.port.set(self.launcher.app_settings["port"])
            if "host" in self.launcher.app_settings:
                print(f"DEBUG: Setting host from app_settings: {self.launcher.app_settings['host']}") # Add debug print
                self.launcher.host.set(self.launcher.app_settings["host"])

        except json.JSONDecodeError as e:
             print(f"Config Load Error: Failed to parse JSON from {self.launcher.config_path}\nError: {e}", file=sys.stderr)
             messagebox.showerror("Config Load Error", f"Failed to parse config file:\n{self.launcher.config_path}\n\nError: {e}\n\nUsing default settings.")
             # Reset to defaults on parse error
             self.launcher.app_settings = {
                 "last_llama_cpp_dir": "", "last_venv_dir": "", "last_model_path": "",
                 "model_dirs": [], "model_list_height": 8, "selected_gpus": [], "custom_parameters": [],
                 "host": "127.0.0.1", "port": "8080"  # Add default network settings
             }
             self.launcher.saved_configs = {}
             self.launcher.custom_parameters_list = [] # Reset internal list
        except Exception as exc:
            print(f"Config Load Error: Could not load config from {self.launcher.config_path}\nError: {exc}", file=sys.stderr)
            messagebox.showerror("Config Load Error", f"Could not load config from:\n{self.launcher.config_path}\n\nError: {exc}\n\nUsing default settings.")
            # Reset to defaults on other load errors
            self.launcher.app_settings = {
                "last_llama_cpp_dir": "", "last_venv_dir": "", "last_model_path": "",
                "model_dirs": [], "model_list_height": 8, "selected_gpus": [], "custom_parameters": [],
                "host": "127.0.0.1", "port": "8080"  # Add default network settings
            }
            self.launcher.saved_configs = {}
            self.launcher.custom_parameters_list = [] # Reset internal list

    def save_configs(self):
        """Saves the app settings and configurations to file."""
        if self.launcher.config_path.name in ("null", "NUL"):
             print("Config saving is disabled.", file=sys.stderr)
             return

        # Validate and clean up model_dirs paths before saving
        valid_model_dirs = []
        for p in self.launcher.model_dirs:
            try:
                resolved_path = Path(p).resolve()
                if resolved_path.exists() and resolved_path.is_dir():
                    valid_model_dirs.append(resolved_path)
                    print(f"DEBUG: Saving valid model directory: {resolved_path}", file=sys.stderr)
                else:
                    print(f"WARNING: Skipping invalid model directory during save: {p} (resolved to {resolved_path})", file=sys.stderr)
            except Exception as e:
                print(f"ERROR: Failed to process model directory during save '{p}': {e}", file=sys.stderr)
        
        # Update the launcher's model_dirs with only valid paths
        self.launcher.model_dirs = valid_model_dirs

        # Save current values to app_settings
        self.launcher.app_settings["model_dirs"] = [str(p) for p in valid_model_dirs]
        self.launcher.app_settings["last_model_path"] = self.launcher.model_path.get()
        self.launcher.app_settings["last_llama_cpp_dir"] = self.launcher.llama_cpp_dir.get()
        self.launcher.app_settings["last_ik_llama_dir"] = self.launcher.ik_llama_dir.get()
        self.launcher.app_settings["last_venv_dir"] = self.launcher.venv_dir.get()
        # Save selected GPU indices from the current state of the checkboxes
        self.launcher.app_settings["selected_gpus"] = [i for i, v in enumerate(self.launcher.gpu_vars) if v.get()]
        # Save custom parameters list
        self.launcher.app_settings["custom_parameters"] = self.launcher.custom_parameters_list
        # Save network settings - ensure these are saved
        self.launcher.app_settings["host"] = self.launcher.host.get()
        self.launcher.app_settings["port"] = self.launcher.port.get()
        print(f"DEBUG: Saving port as {self.launcher.port.get()}") # Add debug print
        
        # Save backend selection
        self.launcher.app_settings["backend_selection"] = self.launcher.backend_selection.get()
        
        # Save manual GPU settings
        self.launcher.app_settings["manual_gpu_mode"] = self.launcher.manual_gpu_mode.get()
        self.launcher.app_settings["manual_gpu_count"] = self.launcher.manual_gpu_count.get()
        self.launcher.app_settings["manual_gpu_vram"] = self.launcher.manual_gpu_vram.get()
        # Save new manual GPU list format
        self.launcher.app_settings["manual_gpu_list"] = self.launcher.manual_gpu_list
        
        # Save manual model settings
        self.launcher.app_settings["manual_model_mode"] = self.launcher.manual_model_mode.get()
        self.launcher.app_settings["manual_model_layers"] = self.launcher.manual_model_layers.get()
        self.launcher.app_settings["manual_model_size_gb"] = self.launcher.manual_model_size_gb.get()
        
        # Save ik_llama settings to app_settings
        ik_llama_settings = self.launcher.ik_llama_tab.save_to_config()
        self.launcher.app_settings.update(ik_llama_settings)

        payload = {
            "configs":      self.launcher.saved_configs,
            "app_settings": self.launcher.app_settings,
        }
        try:
            self.launcher.config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"DEBUG: Successfully saved config to FULL PATH: {self.launcher.config_path.resolve()}") # Add debug print
        except Exception as exc:
            print(f"Config Save Error: Failed to save settings to {self.launcher.config_path}\nError: {exc}", file=sys.stderr)
            # Attempt fallback only if the initial path wasn't already a fallback
            if not any(s in str(self.launcher.config_path).lower() for s in ["appdata", ".config"]):
                 original_path = self.launcher.config_path
                 # Re-call get_config_path to get the fallback path
                 self.launcher.config_path = self.get_config_path()
                 # Check if get_config_path actually provided a different, writable path
                 if self.launcher.config_path != original_path and self.launcher.config_path.name not in ("null", "NUL"):
                      try:
                         self.launcher.config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                         messagebox.showwarning("Config Save Info", f"Could not write to original location.\nSettings stored in:\n{self.launcher.config_path}")
                      except Exception as final_exc:
                          print(f"Config Save Error: Failed to save settings to fallback {self.launcher.config_path}\nError: {final_exc}", file=sys.stderr)
                          messagebox.showerror("Config Save Error", f"Failed to save settings to fallback location:\n{self.launcher.config_path}\n\nError: {final_exc}")
                 else:
                      # If fallback path was the same or invalid, show error for original path
                      messagebox.showerror("Config Save Error", f"Failed to save settings to:\n{original_path}\n\nError: {exc}")
            else:
                 # If the original path was already a fallback, just report the error
                 messagebox.showerror("Config Save Error", f"Failed to save settings to:\n{self.launcher.config_path}\n\nError: {exc}")

    def save_configuration(self):
        """Saves the current UI settings as a named configuration."""
        name = self.launcher.config_name.get().strip()
        if not name:
            # Auto-generate a name based on current settings
            suggested_name = self.generate_default_config_name()
            self.launcher.config_name.set(suggested_name)
            name = suggested_name
        
        # Ensure name is not None
        if name is None:
            name = "default_config"
            self.launcher.config_name.set(name)

        current_cfg = self.current_cfg()
        self.launcher.saved_configs[name] = current_cfg
        self.save_configs()
        self.update_config_listbox()
        messagebox.showinfo("Saved", f"Current settings saved as '{name}'.") 