"""
Environmental Variables Module for LLaMa.cpp Server Launcher

This module provides functionality to manage environmental variables that will be
set when launching the llama-server process. It includes predefined CUDA variables
and allows for custom variables to be added.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import json
import subprocess
import sys
from typing import Dict, List, Tuple, Optional

class EnvironmentalVariablesManager:
    """Manages environmental variables for the launcher application."""
    
    # Predefined common CUDA environmental variables with descriptions
    PREDEFINED_ENV_VARS = {
        "GGML_CUDA_FORCE_MMQ": {
            "default": "1",
            "description": "Force matrix multiplication using CUDA compute"
        },
        "GGML_CUDA_F16": {
            "default": "1", 
            "description": "Enable FP16 CUDA operations"
        },
        "GGML_CUDA_GRAPH_FORCE": {
            "default": "1",
            "description": "Force CUDA graph usage for optimization"
        },
        "GGML_CUDA_FORCE_CUBLAS": {
            "default": "1",
            "description": "Force use of cuBLAS library"
        },
        "GGML_CUDA_DMMV_F16": {
            "default": "1",
            "description": "Enable FP16 for CUDA DMMV operations"
        },
        "GGML_CUDA_ALLOW_FP16_REDUCE": {
            "default": "1",
            "description": "Allow FP16 reduction operations in CUDA"
        }
    }
    
    def __init__(self):
        """Initialize the environmental variables manager."""
        # Dictionary to store enabled predefined variables and their values
        self.enabled_predefined_vars: Dict[str, str] = {}
        
        # List to store custom environmental variables (name, value) tuples
        self.custom_env_vars: List[Tuple[str, str]] = []
        
        # Flag to enable/disable all environmental variables
        self.env_vars_enabled = tk.BooleanVar(value=False)
    
    def get_enabled_env_vars(self) -> Dict[str, str]:
        """
        Get all enabled environmental variables as a dictionary.
        
        Returns:
            Dict mapping variable names to their values
        """
        if not self.env_vars_enabled.get():
            return {}
        
        result = {}
        
        # Add enabled predefined variables
        result.update(self.enabled_predefined_vars)
        
        # Add custom variables
        for name, value in self.custom_env_vars:
            if name.strip() and value.strip():  # Only include non-empty vars
                result[name.strip()] = value.strip()
        
        return result
    
    def set_predefined_var_enabled(self, var_name: str, enabled: bool, value: str = None):
        """
        Enable or disable a predefined environmental variable.
        
        Args:
            var_name: Name of the predefined variable
            enabled: Whether to enable the variable
            value: Value to set (uses default if None)
        """
        if enabled:
            if value is None:
                value = self.PREDEFINED_ENV_VARS[var_name]["default"]
            self.enabled_predefined_vars[var_name] = value
        else:
            self.enabled_predefined_vars.pop(var_name, None)
    
    def is_predefined_var_enabled(self, var_name: str) -> bool:
        """Check if a predefined variable is enabled."""
        return var_name in self.enabled_predefined_vars
    
    def get_predefined_var_value(self, var_name: str) -> str:
        """Get the value of a predefined variable."""
        return self.enabled_predefined_vars.get(var_name, 
                                               self.PREDEFINED_ENV_VARS[var_name]["default"])
    
    def add_custom_env_var(self, name: str, value: str):
        """Add a custom environmental variable."""
        if name.strip() and value.strip():
            self.custom_env_vars.append((name.strip(), value.strip()))
    
    def remove_custom_env_var(self, index: int):
        """Remove a custom environmental variable by index."""
        if 0 <= index < len(self.custom_env_vars):
            self.custom_env_vars.pop(index)
    
    def update_custom_env_var(self, index: int, name: str, value: str):
        """Update a custom environmental variable."""
        if 0 <= index < len(self.custom_env_vars):
            self.custom_env_vars[index] = (name.strip(), value.strip())
    
    def get_env_vars_for_launch(self) -> Dict[str, str]:
        """
        Get environmental variables formatted for process launch.
        This is the main method called during server launch.
        """
        return self.get_enabled_env_vars()
    
    def load_from_config(self, config_data: dict):
        """Load environmental variables configuration from saved data."""
        env_config = config_data.get("environmental_variables", {})
        
        # Load enabled state
        self.env_vars_enabled.set(env_config.get("enabled", False))
        
        # Load predefined variables
        predefined = env_config.get("predefined", {})
        self.enabled_predefined_vars = {}
        for var_name, var_data in predefined.items():
            if var_data.get("enabled", False):
                self.enabled_predefined_vars[var_name] = var_data.get("value", 
                    self.PREDEFINED_ENV_VARS.get(var_name, {}).get("default", ""))
        
        # Load custom variables
        custom = env_config.get("custom", [])
        self.custom_env_vars = [(item["name"], item["value"]) for item in custom 
                               if isinstance(item, dict) and "name" in item and "value" in item]
    
    def save_to_config(self) -> dict:
        """Save environmental variables configuration to a dictionary."""
        predefined_config = {}
        for var_name in self.PREDEFINED_ENV_VARS:
            predefined_config[var_name] = {
                "enabled": self.is_predefined_var_enabled(var_name),
                "value": self.get_predefined_var_value(var_name)
            }
        
        custom_config = [{"name": name, "value": value} for name, value in self.custom_env_vars]
        
        return {
            "environmental_variables": {
                "enabled": self.env_vars_enabled.get(),
                "predefined": predefined_config,
                "custom": custom_config
            }
        }
    
    def generate_env_string_preview(self) -> str:
        """
        Generate a preview string showing how the environmental variables
        would be set in the launch command.
        """
        if not self.env_vars_enabled.get():
            return "Environmental variables disabled"
        
        env_vars = self.get_enabled_env_vars()
        if not env_vars:
            return "No environmental variables configured"
        
        # Format for display
        parts = []
        for name, value in sorted(env_vars.items()):
            parts.append(f"{name}={value}")
        
        return " ".join(parts)

    def generate_clear_env_vars_command(self) -> str:
        """
        Generate a bash command that sets all CUDA environmental variables to 0
        to clear them from the system environment.
        
        Returns:
            String containing bash export commands to clear all CUDA variables
        """
        clear_commands = []
        
        # Clear all predefined CUDA variables
        for var_name in self.PREDEFINED_ENV_VARS:
            clear_commands.append(f"export {var_name}=0")
        
        # Also clear any custom variables that might be CUDA-related
        for name, _ in self.custom_env_vars:
            if name.strip():
                clear_commands.append(f"export {name.strip()}=0")
        
        # Add a command to show what was cleared
        clear_commands.append("echo 'CUDA environmental variables cleared (set to 0):'")
        
        # Show the cleared variables
        all_vars = list(self.PREDEFINED_ENV_VARS.keys()) + [name for name, _ in self.custom_env_vars if name.strip()]
        for var_name in all_vars:
            clear_commands.append(f"echo '{var_name}=0'")
        
        clear_commands.append("echo 'You can now run commands in this terminal with cleared CUDA variables.'")
        clear_commands.append("bash")  # Keep the terminal open
        
        return "; ".join(clear_commands)


class EnvironmentalVariablesTab:
    """GUI tab for managing environmental variables."""
    
    def __init__(self, parent_frame, env_manager: EnvironmentalVariablesManager):
        """
        Initialize the environmental variables tab.
        
        Args:
            parent_frame: Parent tkinter frame for this tab
            env_manager: EnvironmentalVariablesManager instance
        """
        self.parent = parent_frame
        self.env_manager = env_manager
        
        # Tkinter variables for predefined vars
        self.predefined_var_states = {}
        self.predefined_var_values = {}
        self.predefined_var_checkboxes = {}  # Store checkbox widget references
        self.predefined_var_entries = {}     # Store entry widget references
        
        # Tkinter variables for custom vars
        self.custom_var_name = tk.StringVar()
        self.custom_var_value = tk.StringVar()
        
        # Setup the tab UI automatically
        self._setup_tab()
    
    def _setup_tab(self):
        """Set up the environmental variables tab UI."""
        # Create scrollable frame
        canvas = tk.Canvas(self.parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind("<MouseWheel>", _on_mousewheel)  # Windows
        canvas.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))  # Linux
        canvas.bind("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))   # Linux
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Configure column weights
        scrollable_frame.columnconfigure(1, weight=1)
        
        row = 0
        
        # Main enable/disable checkbox
        ttk.Label(scrollable_frame, text="Environmental Variables", 
                 font=("TkDefaultFont", 12, "bold")).grid(
            column=0, row=row, sticky="w", padx=10, pady=(10, 5), columnspan=3)
        row += 1
        
        ttk.Separator(scrollable_frame, orient="horizontal").grid(
            column=0, row=row, columnspan=3, sticky="ew", padx=10, pady=5)
        row += 1
        
        # Enable/disable checkbox
        self.enable_checkbox = ttk.Checkbutton(
            scrollable_frame, 
            text="Enable Environmental Variables",
            variable=self.env_manager.env_vars_enabled,
            command=self._on_enable_toggle
        )
        self.enable_checkbox.grid(column=0, row=row, sticky="w", padx=10, pady=5, columnspan=3)
        row += 1
        
        # Description
        desc_text = ("Environmental variables will be set when launching the server. "
                    "These are useful for configuring CUDA behavior and other runtime settings.")
        ttk.Label(scrollable_frame, text=desc_text, font=("TkSmallCaptionFont",), 
                 wraplength=700).grid(
            column=0, row=row, sticky="w", padx=10, pady=5, columnspan=3)
        row += 1
        
        # Predefined variables section
        ttk.Label(scrollable_frame, text="Predefined CUDA Variables", 
                 font=("TkDefaultFont", 11, "bold")).grid(
            column=0, row=row, sticky="w", padx=10, pady=(20, 5), columnspan=3)
        row += 1
        
        ttk.Separator(scrollable_frame, orient="horizontal").grid(
            column=0, row=row, columnspan=3, sticky="ew", padx=10, pady=5)
        row += 1
        
        # Create predefined variables controls
        self._create_predefined_vars_section(scrollable_frame, row)
        row += len(self.env_manager.PREDEFINED_ENV_VARS) + 2
        
        # Custom variables section
        ttk.Label(scrollable_frame, text="Custom Environmental Variables", 
                 font=("TkDefaultFont", 11, "bold")).grid(
            column=0, row=row, sticky="w", padx=10, pady=(20, 5), columnspan=3)
        row += 1
        
        ttk.Separator(scrollable_frame, orient="horizontal").grid(
            column=0, row=row, columnspan=3, sticky="ew", padx=10, pady=5)
        row += 1
        
        # Create custom variables section
        self._create_custom_vars_section(scrollable_frame, row)
        row += 10  # Approximate space for custom vars section
        
        # Preview section
        ttk.Label(scrollable_frame, text="Preview", 
                 font=("TkDefaultFont", 11, "bold")).grid(
            column=0, row=row, sticky="w", padx=10, pady=(20, 5), columnspan=3)
        row += 1
        
        ttk.Separator(scrollable_frame, orient="horizontal").grid(
            column=0, row=row, columnspan=3, sticky="ew", padx=10, pady=5)
        row += 1
        
        self._create_preview_section(scrollable_frame, row)
        
        # Initial state update
        self._on_enable_toggle()
    
    def _create_predefined_vars_section(self, parent, start_row):
        """Create the predefined variables section."""
        row = start_row
        
        for var_name, var_info in self.env_manager.PREDEFINED_ENV_VARS.items():
            # Create BooleanVar for checkbox
            var_state = tk.BooleanVar(value=self.env_manager.is_predefined_var_enabled(var_name))
            self.predefined_var_states[var_name] = var_state
            
            # Create StringVar for value
            var_value = tk.StringVar(value=self.env_manager.get_predefined_var_value(var_name))
            self.predefined_var_values[var_name] = var_value
            
            # Checkbox
            checkbox = ttk.Checkbutton(
                parent, 
                text=var_name,
                variable=var_state,
                command=lambda vn=var_name: self._on_predefined_var_toggle(vn)
            )
            checkbox.grid(column=0, row=row, sticky="w", padx=10, pady=2)
            self.predefined_var_checkboxes[var_name] = checkbox  # Store reference
            
            # Value entry
            value_entry = ttk.Entry(parent, textvariable=var_value, width=10)
            value_entry.grid(column=1, row=row, sticky="w", padx=(10, 5), pady=2)
            self.predefined_var_entries[var_name] = value_entry  # Store reference
            
            # Description
            ttk.Label(parent, text=var_info["description"], 
                     font=("TkSmallCaptionFont",)).grid(
                column=2, row=row, sticky="w", padx=5, pady=2)
            
            # Bind value changes
            var_value.trace_add("write", lambda *args, vn=var_name: self._on_predefined_var_value_change(vn))
            
            row += 1
        
        # Quick preset buttons
        preset_frame = ttk.Frame(parent)
        preset_frame.grid(column=0, row=row, columnspan=3, sticky="w", padx=10, pady=10)
        
        ttk.Button(preset_frame, text="Enable All CUDA Optimizations", 
                  command=self._enable_all_cuda_optimizations).pack(side="left", padx=5)
        ttk.Button(preset_frame, text="Disable All", 
                  command=self._disable_all_predefined).pack(side="left", padx=5)
        ttk.Button(preset_frame, text="Clear All Variables (New Terminal)", 
                  command=self._launch_clear_env_terminal).pack(side="left", padx=5)
    
    def _create_custom_vars_section(self, parent, start_row):
        """Create the custom variables section."""
        row = start_row
        
        # Add custom variable controls
        ttk.Label(parent, text="Variable Name:").grid(
            column=0, row=row, sticky="w", padx=10, pady=5)
        
        custom_name_entry = ttk.Entry(parent, textvariable=self.custom_var_name, width=25)
        custom_name_entry.grid(column=1, row=row, sticky="w", padx=10, pady=5)
        row += 1
        
        ttk.Label(parent, text="Variable Value:").grid(
            column=0, row=row, sticky="w", padx=10, pady=5)
        
        custom_value_entry = ttk.Entry(parent, textvariable=self.custom_var_value, width=25)
        custom_value_entry.grid(column=1, row=row, sticky="w", padx=10, pady=5)
        row += 1
        
        # Add/Remove buttons
        button_frame = ttk.Frame(parent)
        button_frame.grid(column=0, row=row, columnspan=2, sticky="w", padx=10, pady=5)
        
        ttk.Button(button_frame, text="Add Variable", 
                  command=self._add_custom_variable).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Remove Selected", 
                  command=self._remove_custom_variable).pack(side="left", padx=5)
        row += 1
        
        # Custom variables list
        list_frame = ttk.Frame(parent)
        list_frame.grid(column=0, row=row, columnspan=3, sticky="nsew", padx=10, pady=5)
        
        # Listbox with scrollbar
        list_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.custom_vars_listbox = tk.Listbox(
            list_frame, 
            height=6, 
            yscrollcommand=list_scrollbar.set,
            exportselection=False
        )
        list_scrollbar.config(command=self.custom_vars_listbox.yview)
        
        self.custom_vars_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Update custom vars list
        self._update_custom_vars_listbox()
    
    def _create_preview_section(self, parent, start_row):
        """Create the preview section."""
        row = start_row
        
        ttk.Label(parent, text="Command Preview:", font=("TkSmallCaptionFont",)).grid(
            column=0, row=row, sticky="w", padx=10, pady=5)
        row += 1
        
        # Preview text widget
        preview_frame = ttk.Frame(parent)
        preview_frame.grid(column=0, row=row, columnspan=3, sticky="ew", padx=10, pady=5)
        preview_frame.columnconfigure(0, weight=1)
        
        self.preview_text = tk.Text(preview_frame, height=3, wrap=tk.WORD, 
                                   font=("TkFixedFont", 9))
        preview_scrollbar = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, 
                                         command=self.preview_text.yview)
        self.preview_text.configure(yscrollcommand=preview_scrollbar.set)
        
        self.preview_text.grid(column=0, row=0, sticky="ew")
        preview_scrollbar.grid(column=1, row=0, sticky="ns")
        
        # Update preview
        self._update_preview()
    
    def _on_enable_toggle(self):
        """Handle enable/disable toggle."""
        enabled = self.env_manager.env_vars_enabled.get()
        
        # Enable/disable individual variable controls based on main checkbox
        state = tk.NORMAL if enabled else tk.DISABLED
        
        # Update predefined variable checkboxes and entries
        for var_name in self.predefined_var_checkboxes:
            try:
                self.predefined_var_checkboxes[var_name].configure(state=state)
                self.predefined_var_entries[var_name].configure(state=state)
            except:
                pass  # Ignore errors
        
        self._update_preview()
    
    def _on_predefined_var_toggle(self, var_name):
        """Handle predefined variable checkbox toggle."""
        enabled = self.predefined_var_states[var_name].get()
        value = self.predefined_var_values[var_name].get()
        self.env_manager.set_predefined_var_enabled(var_name, enabled, value)
        self._update_preview()
    
    def _on_predefined_var_value_change(self, var_name):
        """Handle predefined variable value change."""
        if self.predefined_var_states[var_name].get():  # Only if enabled
            value = self.predefined_var_values[var_name].get()
            self.env_manager.set_predefined_var_enabled(var_name, True, value)
            self._update_preview()
    
    def _enable_all_cuda_optimizations(self):
        """Enable all CUDA optimization variables with default values."""
        for var_name in self.env_manager.PREDEFINED_ENV_VARS:
            self.predefined_var_states[var_name].set(True)
            default_value = self.env_manager.PREDEFINED_ENV_VARS[var_name]["default"]
            self.predefined_var_values[var_name].set(default_value)
            self.env_manager.set_predefined_var_enabled(var_name, True, default_value)
        self._update_preview()
    
    def _disable_all_predefined(self):
        """Disable all predefined variables."""
        for var_name in self.env_manager.PREDEFINED_ENV_VARS:
            self.predefined_var_states[var_name].set(False)
            self.env_manager.set_predefined_var_enabled(var_name, False)
        self._update_preview()
    
    def _add_custom_variable(self):
        """Add a custom environmental variable."""
        name = self.custom_var_name.get().strip()
        value = self.custom_var_value.get().strip()
        
        if not name:
            messagebox.showwarning("Invalid Input", "Please enter a variable name.")
            return
        
        if not value:
            messagebox.showwarning("Invalid Input", "Please enter a variable value.")
            return
        
        # Check for duplicate names
        for existing_name, _ in self.env_manager.custom_env_vars:
            if existing_name.upper() == name.upper():
                messagebox.showwarning("Duplicate Variable", 
                                     f"Variable '{name}' already exists.")
                return
        
        self.env_manager.add_custom_env_var(name, value)
        self.custom_var_name.set("")
        self.custom_var_value.set("")
        self._update_custom_vars_listbox()
        self._update_preview()
    
    def _remove_custom_variable(self):
        """Remove selected custom environmental variable."""
        selection = self.custom_vars_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a variable to remove.")
            return
        
        index = selection[0]
        self.env_manager.remove_custom_env_var(index)
        self._update_custom_vars_listbox()
        self._update_preview()
    
    def _update_custom_vars_listbox(self):
        """Update the custom variables listbox."""
        self.custom_vars_listbox.delete(0, tk.END)
        for name, value in self.env_manager.custom_env_vars:
            self.custom_vars_listbox.insert(tk.END, f"{name}={value}")
    
    def _update_preview(self):
        """Update the preview text."""
        if hasattr(self, 'preview_text'):
            preview = self.env_manager.generate_env_string_preview()
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(1.0, preview)

    def _launch_clear_env_terminal(self):
        """Launch a new terminal with all CUDA environmental variables set to 0."""
        try:
            clear_command = self.env_manager.generate_clear_env_vars_command()
            
            # Determine the appropriate terminal command based on the system
            if sys.platform.startswith('linux'):
                # Try common Linux terminals
                terminals = [
                    ['gnome-terminal', '--', 'bash', '-c', clear_command],
                    ['konsole', '-e', 'bash', '-c', clear_command],
                    ['xterm', '-e', 'bash', '-c', clear_command],
                    ['x-terminal-emulator', '-e', 'bash', '-c', clear_command]
                ]
                
                launched = False
                for terminal_cmd in terminals:
                    try:
                        subprocess.Popen(terminal_cmd)
                        launched = True
                        break
                    except FileNotFoundError:
                        continue
                
                if not launched:
                    messagebox.showerror("Error", 
                        "Could not find a suitable terminal emulator.\n"
                        "Please install gnome-terminal, konsole, or xterm.")
                    return
                        
            elif sys.platform == 'darwin':  # macOS
                # Use Terminal.app on macOS
                applescript = f'''
                tell application "Terminal"
                    do script "{clear_command}"
                    activate
                end tell
                '''
                subprocess.Popen(['osascript', '-e', applescript])
                
            elif sys.platform == 'win32':  # Windows
                # Use cmd on Windows with PowerShell-style variable setting
                win_clear_commands = []
                
                # Clear all predefined CUDA variables
                for var_name in self.env_manager.PREDEFINED_ENV_VARS:
                    win_clear_commands.append(f"set {var_name}=0")
                
                # Also clear any custom variables
                for name, _ in self.env_manager.custom_env_vars:
                    if name.strip():
                        win_clear_commands.append(f"set {name.strip()}=0")
                
                win_clear_commands.extend([
                    "echo CUDA environmental variables cleared (set to 0):",
                    "echo.",
                ])
                
                # Show the cleared variables
                all_vars = list(self.env_manager.PREDEFINED_ENV_VARS.keys()) + [name for name, _ in self.env_manager.custom_env_vars if name.strip()]
                for var_name in all_vars:
                    win_clear_commands.append(f"echo {var_name}=0")
                
                win_clear_commands.extend([
                    "echo.",
                    "echo You can now run commands in this terminal with cleared CUDA variables.",
                    "cmd /k"  # Keep command prompt open
                ])
                
                win_command = " & ".join(win_clear_commands)
                subprocess.Popen(['cmd', '/c', 'start', 'cmd', '/k', win_command])
            
            else:
                messagebox.showerror("Error", f"Unsupported platform: {sys.platform}")
                return
                
            messagebox.showinfo("Terminal Launched", 
                "New terminal opened with CUDA environmental variables cleared (set to 0).")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch terminal: {str(e)}") 