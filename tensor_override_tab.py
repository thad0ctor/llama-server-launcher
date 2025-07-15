#!/usr/bin/env python3
"""
Tensor Override Tab Module for LLaMa.cpp Server Launcher
Provides UI components for tensor override analysis and configuration.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import json
import sys
from pathlib import Path
import threading
import subprocess

class TensorOverrideTab:
    """Manages the tensor override tab UI and functionality."""
    
    def __init__(self, parent_launcher):
        """
        Initialize tensor override tab.
        
        Args:
            parent_launcher: Reference to the main LlamaCppLauncher instance
        """
        self.parent = parent_launcher
        self.root = parent_launcher.root
        
        # Tensor override variables
        self.tensor_override_enabled = tk.BooleanVar(value=False)
        self.tensor_analysis_status = tk.StringVar(value="Ready to analyze")
        self.tensor_params_count = tk.StringVar(value="0 parameters")
        self.analysis_progress = tk.StringVar(value="")
        
        # Current analysis state
        self.current_model_path = None
        self.current_config_name = None
        self.tensor_params_file = None
        self.analysis_thread = None
        
        # Per-GPU VRAM buffer settings (in MB)
        self.gpu_vram_buffers = {}  # Dictionary of GPU ID -> StringVar for buffer amount
        
        # Create tensor override subdirectory if it doesn't exist
        self.tensor_override_dir = Path("tensor_overrides")
        self.tensor_override_dir.mkdir(exist_ok=True)
    
    def setup_tab(self, frame):
        """Setup the tensor override tab UI."""
        
        # Main container with padding
        main_container = ttk.Frame(frame)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title and description
        title_frame = ttk.Frame(main_container)
        title_frame.pack(fill="x", pady=(0, 10))
        
        title_label = ttk.Label(title_frame, text="Tensor Override Configuration", 
                               font=("TkDefaultFont", 12, "bold"))
        title_label.pack(anchor="w")
        
        desc_label = ttk.Label(title_frame, 
                              text="Automatically analyze models and generate optimal tensor placement parameters for multi-GPU setups.\nWhen enabled, GPU Layers and Tensor Split controls will be disabled to prevent conflicts.",
                              foreground="gray")
        desc_label.pack(anchor="w", pady=(2, 0))
        
        # Enable/Disable section
        enable_frame = ttk.LabelFrame(main_container, text="Tensor Override Settings", padding=10)
        enable_frame.pack(fill="x", pady=(0, 10))
        
        self.enable_checkbox = ttk.Checkbutton(
            enable_frame,
            text="Enable automatic tensor override for this configuration",
            variable=self.tensor_override_enabled,
            command=self._on_enable_changed
        )
        self.enable_checkbox.pack(anchor="w")
        
        # Info label for when enabled
        self.enable_info_label = ttk.Label(
            enable_frame,
            text="When enabled, tensor override parameters will be automatically included in launch commands.\nGPU Layers and Tensor Split controls will be disabled to prevent conflicts.",
            foreground="blue"
        )
        self.enable_info_label.pack(anchor="w", pady=(5, 0))
        
        # GPU VRAM Buffer configuration section
        self.buffer_frame = ttk.LabelFrame(main_container, text="GPU VRAM Buffer Configuration", padding=10)
        self.buffer_frame.pack(fill="x", pady=(0, 10))
        
        # Buffer description
        buffer_desc_label = ttk.Label(
            self.buffer_frame,
            text="Configure per-GPU VRAM buffer (in MB) to prevent out-of-memory errors.\nBuffers are subtracted from available VRAM during tensor optimization.",
            foreground="gray"
        )
        buffer_desc_label.pack(anchor="w", pady=(0, 10))
        
        # GPU buffer controls container
        self.gpu_buffer_controls_frame = ttk.Frame(self.buffer_frame)
        self.gpu_buffer_controls_frame.pack(fill="x")
        
        # Initialize GPU buffer controls
        self._setup_gpu_buffer_controls()
        
        # Analysis section
        analysis_frame = ttk.LabelFrame(main_container, text="Model Analysis", padding=10)
        analysis_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        # Model info section
        model_info_frame = ttk.Frame(analysis_frame)
        model_info_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(model_info_frame, text="Current model:").pack(anchor="w")
        self.model_info_label = ttk.Label(model_info_frame, text="No model selected", 
                                         foreground="gray")
        self.model_info_label.pack(anchor="w", padx=(20, 0))
        
        ttk.Label(model_info_frame, text="Configuration:").pack(anchor="w", pady=(5, 0))
        self.config_info_label = ttk.Label(model_info_frame, text="No configuration", 
                                          foreground="gray")
        self.config_info_label.pack(anchor="w", padx=(20, 0))
        
        # Current launcher settings section
        settings_frame = ttk.Frame(analysis_frame)
        settings_frame.pack(fill="x", pady=(10, 0))
        
        ttk.Label(settings_frame, text="Analysis will use current launcher settings:", 
                 font=("TkDefaultFont", 9, "bold")).pack(anchor="w")
        
        self.settings_display_frame = ttk.Frame(settings_frame)
        self.settings_display_frame.pack(fill="x", padx=(20, 0), pady=(2, 0))
        
        # Analysis controls
        controls_frame = ttk.Frame(analysis_frame)
        controls_frame.pack(fill="x", pady=(10, 0))
        
        self.analyze_button = ttk.Button(
            controls_frame,
            text="Optimize VRAM Usage",
            command=self._analyze_model,
            state="disabled"
        )
        self.analyze_button.pack(side="left", padx=(0, 10))
        
        self.refresh_button = ttk.Button(
            controls_frame,
            text="Refresh Status",
            command=self._refresh_status
        )
        self.refresh_button.pack(side="left")
        
        # Status section
        status_frame = ttk.Frame(analysis_frame)
        status_frame.pack(fill="x", pady=(10, 0))
        
        ttk.Label(status_frame, text="Analysis status:").pack(anchor="w")
        self.status_label = ttk.Label(status_frame, textvariable=self.tensor_analysis_status)
        self.status_label.pack(anchor="w", padx=(20, 0))
        
        ttk.Label(status_frame, text="Generated parameters:").pack(anchor="w", pady=(5, 0))
        self.params_label = ttk.Label(status_frame, textvariable=self.tensor_params_count)
        self.params_label.pack(anchor="w", padx=(20, 0))
        
        # Progress section
        progress_frame = ttk.Frame(analysis_frame)
        progress_frame.pack(fill="x", pady=(10, 0))
        
        self.progress_label = ttk.Label(progress_frame, textvariable=self.analysis_progress, 
                                       foreground="blue")
        self.progress_label.pack(anchor="w")
        
        # Results section
        results_frame = ttk.LabelFrame(main_container, text="Analysis Results", padding=10)
        results_frame.pack(fill="both", expand=True)
        
        # Results text area with scrollbar
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            height=8,
            wrap=tk.WORD,
            state="disabled"
        )
        self.results_text.pack(fill="both", expand=True, pady=(0, 5))
        
        # Results controls
        results_controls_frame = ttk.Frame(results_frame)
        results_controls_frame.pack(fill="x")
        
        self.view_params_button = ttk.Button(
            results_controls_frame,
            text="View Parameters File",
            command=self._view_parameters_file,
            state="disabled"
        )
        self.view_params_button.pack(side="left", padx=(0, 10))
        
        self.clear_results_button = ttk.Button(
            results_controls_frame,
            text="Clear Results",
            command=self._clear_results
        )
        self.clear_results_button.pack(side="left")
        
        # Initial state update
        self._update_ui_state()
        self._update_gpu_controls_state()
    
    def _setup_gpu_buffer_controls(self):
        """Setup per-GPU VRAM buffer controls based on detected GPUs."""
        # Clear existing controls
        for widget in self.gpu_buffer_controls_frame.winfo_children():
            widget.destroy()
        
        # Clear existing buffer variables
        self.gpu_vram_buffers.clear()
        
        # Get GPU count and create controls
        gpu_count = self._detect_gpu_count()
        
        if gpu_count <= 0:
            # No GPUs detected, show message
            no_gpu_label = ttk.Label(
                self.gpu_buffer_controls_frame,
                text="No GPUs detected. Buffer configuration will be available once GPUs are detected.",
                foreground="orange"
            )
            no_gpu_label.pack(anchor="w")
            return
        
        # Create a grid of GPU buffer controls
        for gpu_id in range(gpu_count):
            # Create StringVar for this GPU's buffer
            buffer_var = tk.StringVar(value="512")  # Default 512MB buffer
            self.gpu_vram_buffers[gpu_id] = buffer_var
            
            # Create frame for this GPU's controls
            gpu_frame = ttk.Frame(self.gpu_buffer_controls_frame)
            gpu_frame.pack(fill="x", pady=2)
            
            # GPU label
            gpu_label = ttk.Label(gpu_frame, text=f"GPU {gpu_id}:")
            gpu_label.pack(side="left")
            
            # Buffer entry
            buffer_entry = ttk.Entry(gpu_frame, textvariable=buffer_var, width=8)
            buffer_entry.pack(side="left", padx=(5, 2))
            
            # MB label
            mb_label = ttk.Label(gpu_frame, text="MB")
            mb_label.pack(side="left")
            
            # GPU info if available
            if hasattr(self.parent, 'gpu_info') and self.parent.gpu_info:
                devices = self.parent.gpu_info.get("devices", [])
                if gpu_id < len(devices):
                    gpu_info = devices[gpu_id]
                    gpu_name = gpu_info.get("name", "Unknown")
                    total_vram_gb = gpu_info.get("total_memory_bytes", 0) / (1024**3)
                    
                    info_label = ttk.Label(
                        gpu_frame,
                        text=f"({gpu_name}, {total_vram_gb:.1f}GB)",
                        foreground="gray",
                        font=("TkDefaultFont", 8)
                    )
                    info_label.pack(side="left", padx=(10, 0))
        
        # Add preset buttons
        preset_frame = ttk.Frame(self.gpu_buffer_controls_frame)
        preset_frame.pack(fill="x", pady=(10, 0))
        
        ttk.Label(preset_frame, text="Presets:").pack(side="left")
        
        # Conservative preset (1GB per GPU)
        conservative_btn = ttk.Button(
            preset_frame,
            text="Conservative (1GB)",
            command=lambda: self._set_buffer_preset(1024),
            width=15
        )
        conservative_btn.pack(side="left", padx=(5, 2))
        
        # Moderate preset (512MB per GPU)
        moderate_btn = ttk.Button(
            preset_frame,
            text="Moderate (512MB)",
            command=lambda: self._set_buffer_preset(512),
            width=15
        )
        moderate_btn.pack(side="left", padx=(2, 2))
        
        # Aggressive preset (256MB per GPU)
        aggressive_btn = ttk.Button(
            preset_frame,
            text="Aggressive (256MB)",
            command=lambda: self._set_buffer_preset(256),
            width=15
        )
        aggressive_btn.pack(side="left", padx=(2, 2))
    
    def _set_buffer_preset(self, buffer_mb):
        """Set all GPU buffers to the specified preset value."""
        for buffer_var in self.gpu_vram_buffers.values():
            buffer_var.set(str(buffer_mb))
    
    def _on_enable_changed(self):
        """Handle enable checkbox change."""
        self._update_ui_state()
        self._update_gpu_controls_state()
        
        if self.tensor_override_enabled.get():
            self._check_current_model()
    
    def _update_ui_state(self):
        """Update UI state based on current settings."""
        enabled = self.tensor_override_enabled.get()
        
        # Update analyze button state
        if enabled and self.parent.model_path.get():
            self.analyze_button.config(state="normal")
        else:
            self.analyze_button.config(state="disabled")
        
        # Update info visibility
        if enabled:
            self.enable_info_label.config(foreground="blue")
        else:
            self.enable_info_label.config(foreground="gray")
        
        # Update settings display
        self._update_settings_display()
    
    def _update_gpu_controls_state(self):
        """Enable/disable GPU controls based on tensor override state."""
        enabled = self.tensor_override_enabled.get()
        
        # When tensor override is enabled, disable conflicting GPU controls
        # When disabled, re-enable the controls
        control_state = tk.DISABLED if enabled else tk.NORMAL
        
        try:
            # Disable/enable n-gpu-layers entry field
            if hasattr(self.parent, 'n_gpu_layers_entry') and self.parent.n_gpu_layers_entry.winfo_exists():
                self.parent.n_gpu_layers_entry.config(state=control_state)
            
            # Disable/enable GPU layers slider (only if it's not already disabled for other reasons)
            if hasattr(self.parent, 'gpu_layers_slider') and self.parent.gpu_layers_slider.winfo_exists():
                # Check if slider should be enabled (has max layers > 0)
                max_layers = getattr(self.parent, 'max_gpu_layers', None)
                if enabled:
                    # Tensor override enabled - always disable
                    self.parent.gpu_layers_slider.config(state=tk.DISABLED)
                elif max_layers and max_layers.get() > 0:
                    # Tensor override disabled and we have layers - enable
                    self.parent.gpu_layers_slider.config(state=tk.NORMAL)
                # If max_layers <= 0, leave disabled (normal launcher behavior)
            
            # Disable/enable tensor-split entry field
            if hasattr(self.parent, 'tensor_split_entry') and self.parent.tensor_split_entry.winfo_exists():
                self.parent.tensor_split_entry.config(state=control_state)
            
            # Show visual feedback about why controls are disabled
            if enabled:
                print("DEBUG: GPU controls disabled - tensor override is managing tensor placement", file=sys.stderr)
                self._add_gpu_control_info_labels()
            else:
                print("DEBUG: GPU controls re-enabled - tensor override is disabled", file=sys.stderr)
                self._remove_gpu_control_info_labels()
                
        except Exception as e:
            print(f"WARNING: Error updating GPU controls state: {e}", file=sys.stderr)
    
    def _add_gpu_control_info_labels(self):
        """Add informational labels near disabled GPU controls."""
        try:
            # Add info label near GPU layers control
            if hasattr(self.parent, 'n_gpu_layers_entry') and self.parent.n_gpu_layers_entry.winfo_exists():
                parent_frame = self.parent.n_gpu_layers_entry.master
                if not hasattr(self, 'gpu_layers_info_label') or not self.gpu_layers_info_label.winfo_exists():
                    self.gpu_layers_info_label = tk.Label(
                        parent_frame,
                        text="(Disabled: Tensor Override Active)",
                        foreground="orange",
                        font=("TkDefaultFont", 8)
                    )
                    # Try to place it next to the entry field
                    try:
                        info = self.parent.n_gpu_layers_entry.grid_info()
                        row = info.get('row', 0)
                        column = info.get('column', 0) + 2  # Place to the right
                        self.gpu_layers_info_label.grid(row=row, column=column, sticky="w", padx=(5, 0))
                    except:
                        # Fallback to pack if grid fails
                        self.gpu_layers_info_label.pack(side=tk.RIGHT, padx=(5, 0))
            
            # Add info label near tensor split control
            if hasattr(self.parent, 'tensor_split_entry') and self.parent.tensor_split_entry.winfo_exists():
                parent_frame = self.parent.tensor_split_entry.master
                if not hasattr(self, 'tensor_split_info_label') or not self.tensor_split_info_label.winfo_exists():
                    self.tensor_split_info_label = tk.Label(
                        parent_frame,
                        text="(Disabled: Tensor Override Active)",
                        foreground="orange",
                        font=("TkDefaultFont", 8)
                    )
                    # Try to place it next to the entry field
                    try:
                        info = self.parent.tensor_split_entry.grid_info()
                        row = info.get('row', 0)
                        column = info.get('column', 0) + 1  # Place to the right
                        self.tensor_split_info_label.grid(row=row, column=column, sticky="w", padx=(5, 0))
                    except:
                        # Fallback to pack if grid fails
                        self.tensor_split_info_label.pack(side=tk.RIGHT, padx=(5, 0))
                        
        except Exception as e:
            print(f"WARNING: Error adding GPU control info labels: {e}", file=sys.stderr)
    
    def _remove_gpu_control_info_labels(self):
        """Remove informational labels from disabled GPU controls."""
        try:
            # Remove GPU layers info label
            if hasattr(self, 'gpu_layers_info_label') and self.gpu_layers_info_label.winfo_exists():
                self.gpu_layers_info_label.destroy()
            
            # Remove tensor split info label
            if hasattr(self, 'tensor_split_info_label') and self.tensor_split_info_label.winfo_exists():
                self.tensor_split_info_label.destroy()
                
        except Exception as e:
            print(f"WARNING: Error removing GPU control info labels: {e}", file=sys.stderr)
    
    def _update_settings_display(self):
        """Update the display of current launcher settings."""
        # Clear existing widgets
        for widget in self.settings_display_frame.winfo_children():
            widget.destroy()
        
        try:
            # Get current launcher settings
            context_size = self.parent.ctx_size.get()
            cache_type_k = self.parent.cache_type_k.get()
            cache_type_v = self.parent.cache_type_v.get()
            n_gpu_layers = self.parent.n_gpu_layers.get()
            
            # Display key settings
            settings_text = f"Context: {context_size} | KV Cache: K={cache_type_k}, V={cache_type_v} | GPU Layers: {n_gpu_layers}"
            
            settings_label = ttk.Label(self.settings_display_frame, text=settings_text, 
                                     foreground="darkblue", font=("TkDefaultFont", 8))
            settings_label.pack(anchor="w")
            
        except Exception as e:
            # Show error if settings can't be read
            error_label = ttk.Label(self.settings_display_frame, 
                                  text="Error reading launcher settings", 
                                  foreground="red", font=("TkDefaultFont", 8))
            error_label.pack(anchor="w")
    
    def _check_current_model(self):
        """Check if current model/config has changed."""
        model_path = self.parent.model_path.get()
        config_name = self.parent.config_name.get()
        
        if model_path != self.current_model_path or config_name != self.current_config_name:
            self.current_model_path = model_path
            self.current_config_name = config_name
            self._update_model_info()
            self._check_existing_analysis()
    
    def _update_model_info(self):
        """Update model and config info display."""
        if self.current_model_path:
            model_name = Path(self.current_model_path).name
            self.model_info_label.config(text=model_name, foreground="black")
        else:
            self.model_info_label.config(text="No model selected", foreground="gray")
        
        if self.current_config_name:
            self.config_info_label.config(text=self.current_config_name, foreground="black")
        else:
            self.config_info_label.config(text="No configuration", foreground="gray")
    
    def _check_existing_analysis(self):
        """Check if analysis already exists for current model/config."""
        if not self.current_model_path or not self.current_config_name:
            self.tensor_analysis_status.set("Ready to analyze")
            self.tensor_params_count.set("0 parameters")
            self.view_params_button.config(state="disabled")
            return
        
        # Generate expected parameters file path
        model_name = Path(self.current_model_path).stem
        params_filename = f"{model_name}_{self.current_config_name}_tensor_params.txt"
        params_path = self.tensor_override_dir / params_filename
        
        if params_path.exists():
            try:
                # Count parameters in file
                with open(params_path, 'r') as f:
                    lines = f.readlines()
                
                param_lines = [line for line in lines if line.strip().startswith('-ot')]
                param_count = len(param_lines)
                
                self.tensor_analysis_status.set("Analysis complete")
                self.tensor_params_count.set(f"{param_count} parameters")
                self.tensor_params_file = str(params_path)
                self.view_params_button.config(state="normal")
                
                # Show summary in results
                self._show_results_summary(param_lines)
                
            except Exception as e:
                self.tensor_analysis_status.set(f"Error reading existing analysis: {e}")
                self.tensor_params_count.set("0 parameters")
                self.view_params_button.config(state="disabled")
        else:
            self.tensor_analysis_status.set("Ready to analyze")
            self.tensor_params_count.set("0 parameters")
            self.view_params_button.config(state="disabled")
    
    def _analyze_model(self):
        """Start model analysis in background thread using VRAM optimizer."""
        if not self.current_model_path:
            messagebox.showerror("Error", "No model selected")
            return
        
        if not self.current_config_name:
            messagebox.showerror("Error", "No configuration selected")
            return
        
        if self.analysis_thread and self.analysis_thread.is_alive():
            messagebox.showwarning("Warning", "Analysis is already running")
            return
        
        # Clear old analysis files before starting new one
        self._clear_old_analysis_files()
        
        # Use VRAM optimization instead of old regex analysis
        self.analysis_thread = threading.Thread(target=self._run_vram_optimization, daemon=True)
        self.analysis_thread.start()
    
    def _clear_old_analysis_files(self):
        """Clear old tensor analysis files for current model/config."""
        if not self.current_model_path or not self.current_config_name:
            return
        
        try:
            model_name = Path(self.current_model_path).stem
            
            # List of potential file patterns to clear
            patterns_to_clear = [
                f"{model_name}_{self.current_config_name}_tensor_params.txt",
                f"{model_name}_{self.current_config_name}_vram_optimized_tensor_params.txt",
                f"{model_name}*{self.current_config_name}*tensor_params.txt",
                f"{model_name}*tensor_params.txt"
            ]
            
            files_cleared = 0
            for pattern in patterns_to_clear:
                matching_files = list(self.tensor_override_dir.glob(pattern))
                for file_path in matching_files:
                    try:
                        file_path.unlink()
                        files_cleared += 1
                        print(f"DEBUG: Cleared old tensor file: {file_path.name}", file=sys.stderr)
                    except Exception as e:
                        print(f"DEBUG: Could not clear {file_path.name}: {e}", file=sys.stderr)
            
            if files_cleared > 0:
                print(f"DEBUG: Cleared {files_cleared} old tensor analysis files", file=sys.stderr)
                
        except Exception as e:
            print(f"DEBUG: Error clearing old analysis files: {e}", file=sys.stderr)
    
    def _run_analysis(self):
        """Run tensor analysis in background thread."""
        try:
            # Update UI state
            self.root.after(0, lambda: self.analyze_button.config(state="disabled"))
            self.root.after(0, lambda: self.tensor_analysis_status.set("Starting analysis..."))
            self.root.after(0, lambda: self.analysis_progress.set("Initializing tensor override analyzer..."))
            
            # Create config for analysis
            analysis_config = self._create_analysis_config()
            
            # Generate unique output filename
            model_name = Path(self.current_model_path).stem
            params_filename = f"{model_name}_{self.current_config_name}_tensor_params.txt"
            output_path = self.tensor_override_dir / params_filename
            
            # Run tensor override analysis
            config_path = Path("tensor-override/my_config.json").resolve()
            cmd = [
                "python3", 
                "tensor-override/main.py",
                self.current_model_path,
                "--config", str(config_path),
                "--output-dir", str(self.tensor_override_dir),
                "--output-filename", params_filename
            ]
            
            self.root.after(0, lambda: self.analysis_progress.set("Running tensor analysis..."))
            
            print(f"DEBUG: Starting tensor analysis with command: {' '.join(cmd)}", file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            print("TENSOR ANALYSIS OUTPUT:", file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            
            # Run the analysis with real-time output streaming
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr into stdout
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
                cwd=Path.cwd()
            )
            
            # Stream output to debug console in real-time
            output_lines = []
            step_count = 0
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    line = output.rstrip()
                    # Print to debug console
                    print(f"TENSOR: {line}", file=sys.stderr)
                    # Store for error handling
                    output_lines.append(output)
                    
                    # Update UI progress based on analysis steps
                    step_count += 1
                    if "Step 1:" in line:
                        self.root.after(0, lambda: self.analysis_progress.set("Step 1: Detecting CUDA devices..."))
                    elif "Step 2:" in line:
                        self.root.after(0, lambda: self.analysis_progress.set("Step 2: Analyzing GGUF model..."))
                    elif "Step 3:" in line:
                        self.root.after(0, lambda: self.analysis_progress.set("Step 3: Optimizing tensor placement..."))
                    elif "Step 4:" in line:
                        self.root.after(0, lambda: self.analysis_progress.set("Step 4: Generating parameters..."))
                    elif "Analysis complete" in line:
                        self.root.after(0, lambda: self.analysis_progress.set("Analysis complete!"))
                    elif "Error:" in line:
                        self.root.after(0, lambda: self.analysis_progress.set("Analysis failed!"))
                    elif "DETECTED:" in line:
                        # Extract the specific error detection for user feedback
                        if "unsupported quantization type" in line.lower():
                            self.root.after(0, lambda: self.analysis_progress.set("Error: Unsupported quantization type"))
                        elif "model loading failed" in line.lower():
                            self.root.after(0, lambda: self.analysis_progress.set("Error: Model loading failed"))
                    elif "Found" in line and "tensors of type" in line:
                        self.root.after(0, lambda: self.analysis_progress.set("Found tensor types..."))
                    elif "Categorized tensors:" in line:
                        self.root.after(0, lambda: self.analysis_progress.set("Categorizing tensors..."))
                    elif step_count % 10 == 0:  # Update every 10 lines
                        self.root.after(0, lambda: self.analysis_progress.set(f"Processing... ({step_count} lines)"))
            
            # Wait for process to complete
            return_code = process.poll()
            
            print("=" * 80, file=sys.stderr)
            print(f"TENSOR ANALYSIS COMPLETED - Return code: {return_code}", file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            
            if return_code == 0:
                # Check if the expected parameters file was generated
                if output_path.exists():
                    # Update UI with success
                    self.root.after(0, self._analysis_success, str(output_path))
                else:
                    raise Exception("Analysis completed but no parameters file was generated")
            else:
                # Join output lines for error message
                error_output = ''.join(output_lines)
                raise Exception(f"Analysis failed with return code {return_code}. Output:\n{error_output}")
                
        except Exception as e:
            # Update UI with error
            self.root.after(0, self._analysis_error, str(e))
        finally:
            # Re-enable analyze button
            self.root.after(0, lambda: self.analyze_button.config(state="normal"))
            self.root.after(0, lambda: self.analysis_progress.set(""))
    
    def _create_analysis_config(self):
        """Create analysis config based on current launcher settings."""
        # Get current launcher settings - these override any defaults in the tensor analysis script
        llama_cpp_path = self.parent.llama_cpp_dir.get().strip()
        venv_path = self.parent.venv_dir.get().strip()
        context_size = self.parent.ctx_size.get()
        cache_type_k = self.parent.cache_type_k.get()
        cache_type_v = self.parent.cache_type_v.get()
        
        # Get additional launcher settings for more accurate analysis
        try:
            # Get GPU settings from launcher
            n_gpu_layers_str = self.parent.n_gpu_layers.get().strip()
            n_gpu_layers = int(n_gpu_layers_str) if n_gpu_layers_str.isdigit() else -1
            
            # Get GPU info for better analysis
            gpu_count = self.parent.gpu_info.get("device_count", 0) if hasattr(self.parent, 'gpu_info') else 0
            
            # Get memory settings
            no_mmap = self.parent.no_mmap.get() if hasattr(self.parent, 'no_mmap') else False
            mlock = self.parent.mlock.get() if hasattr(self.parent, 'mlock') else False
            
        except (ValueError, AttributeError) as e:
            print(f"DEBUG: Could not get some launcher settings for tensor analysis: {e}", file=sys.stderr)
            n_gpu_layers = -1
            gpu_count = 0
            no_mmap = False
            mlock = False
        
        config = {
            "llama_cpp_path": llama_cpp_path if llama_cpp_path else "/home/rgilbreth/Desktop/AI-Software/llama.cpp",
            "virtual_env_path": venv_path if venv_path else None,
            "context_size": context_size,
            "kv_cache": {
                "type": cache_type_k,
                "type_k": cache_type_k,
                "type_v": cache_type_v
            },
            "tensor_analysis": {
                "prioritize_experts_on_cpu": True,
                "maximize_gpu_layers": True,
                "reserve_context_memory": True,
                "memory_safety_margin_mb": 512,
                "launcher_gpu_layers": n_gpu_layers,
                "launcher_gpu_count": gpu_count,
                "launcher_no_mmap": no_mmap,
                "launcher_mlock": mlock
            },
            "output": {
                "save_analysis_json": True,
                "save_override_params": True,
                "verbose": False
            }
        }
        
        print(f"DEBUG: Tensor analysis using launcher settings:", file=sys.stderr)
        print(f"  - Context size: {context_size}", file=sys.stderr)
        print(f"  - KV cache types: K={cache_type_k}, V={cache_type_v}", file=sys.stderr)
        print(f"  - GPU layers: {n_gpu_layers} (from launcher)", file=sys.stderr)
        print(f"  - GPU count: {gpu_count}", file=sys.stderr)
        print(f"  - Memory settings: no_mmap={no_mmap}, mlock={mlock}", file=sys.stderr)
        print(f"  - LLaMa.cpp path: {llama_cpp_path}", file=sys.stderr)
        print(f"  - Virtual env: {venv_path if venv_path else 'None'}", file=sys.stderr)
        
        # Save config for analysis
        config_path = Path("tensor-override/my_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config
    
    def _analysis_success(self, params_file_path):
        """Handle successful analysis completion."""
        self.tensor_params_file = params_file_path
        self.tensor_analysis_status.set("VRAM optimization complete")
        self.view_params_button.config(state="normal")
        
        # Update current model/config info to ensure get_tensor_override_parameters works
        self._check_current_model()
        
        # Count parameters and update display
        try:
            with open(params_file_path, 'r') as f:
                lines = f.readlines()
            
            param_lines = [line for line in lines if line.strip().startswith('-ot')]
            self.tensor_params_count.set(f"{len(param_lines)} optimized parameters")
            
            # Show VRAM optimization results
            self._show_vram_optimization_results(param_lines)
            
        except Exception as e:
            self.tensor_analysis_status.set(f"Optimization complete (error reading file: {e})")
    
    def _analysis_error(self, error_message):
        """Handle analysis error."""
        self.tensor_analysis_status.set(f"Analysis failed: {error_message}")
        self.tensor_params_count.set("0 parameters")
        self.view_params_button.config(state="disabled")
        
        # Show error in results
        self.results_text.config(state="normal")
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Analysis Error:\n{error_message}\n\n")
        self.results_text.insert(tk.END, "Please check your model path and configuration settings.")
        self.results_text.config(state="disabled")
    
    def _show_results_summary(self, param_lines):
        """Show analysis results summary."""
        self.results_text.config(state="normal")
        self.results_text.delete(1.0, tk.END)
        
        # Show summary
        self.results_text.insert(tk.END, f"Tensor Override Analysis Results\n")
        self.results_text.insert(tk.END, f"{'='*40}\n\n")
        self.results_text.insert(tk.END, f"Generated {len(param_lines)} tensor override parameters\n")
        self.results_text.insert(tk.END, f"Model: {Path(self.current_model_path).name}\n")
        self.results_text.insert(tk.END, f"Configuration: {self.current_config_name}\n\n")
        
        # Show first few parameters as preview
        if param_lines:
            self.results_text.insert(tk.END, "Parameter Preview (first 5):\n")
            self.results_text.insert(tk.END, "-" * 30 + "\n")
            for i, line in enumerate(param_lines[:5]):
                self.results_text.insert(tk.END, line.strip() + "\n")
            
            if len(param_lines) > 5:
                self.results_text.insert(tk.END, f"... and {len(param_lines) - 5} more parameters\n")
        
        self.results_text.config(state="disabled")
    
    def _view_parameters_file(self):
        """Open parameters file in default text editor."""
        if self.tensor_params_file and Path(self.tensor_params_file).exists():
            try:
                # Try to open with default system editor
                if os.name == 'nt':  # Windows
                    os.startfile(self.tensor_params_file)
                elif os.name == 'posix':  # Linux/Mac
                    subprocess.run(['xdg-open', self.tensor_params_file])
            except Exception as e:
                messagebox.showerror("Error", f"Could not open file: {e}")
        else:
            messagebox.showerror("Error", "No parameters file available")
    
    def _clear_results(self):
        """Clear results display."""
        self.results_text.config(state="normal")
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state="disabled")
    
    def _refresh_status(self):
        """Refresh analysis status."""
        self._check_current_model()
        self._update_ui_state()
    
    def get_tensor_override_parameters(self):
        """
        Get tensor override parameters for current configuration.
        
        Returns:
            list: List of tensor override parameter strings, or empty list if disabled/unavailable
        """
        if not self.tensor_override_enabled.get():
            print(f"DEBUG: Tensor override disabled, returning empty parameters", file=sys.stderr)
            return []
        
        # Try to get parameters from the expected file location
        # First check if we have a stored file path
        if self.tensor_params_file and Path(self.tensor_params_file).exists():
            params_file = self.tensor_params_file
        else:
            # Try to construct the expected file path
            if self.current_model_path and self.current_config_name:
                model_name = Path(self.current_model_path).stem
                params_filename = f"{model_name}_{self.current_config_name}_tensor_params.txt"
                params_file = str(self.tensor_override_dir / params_filename)
            else:
                print(f"DEBUG: No current model/config for tensor parameters", file=sys.stderr)
                return []
        
        if not Path(params_file).exists():
            print(f"DEBUG: Tensor parameters file not found: {params_file}", file=sys.stderr)
            return []
        
        try:
            print(f"DEBUG: Loading tensor parameters from: {params_file}", file=sys.stderr)
            with open(params_file, 'r') as f:
                lines = f.readlines()
            
            # Extract only the parameter lines (starting with -ot)
            param_lines = []
            for line in lines:
                line = line.strip()
                if line.startswith('-ot'):  # Include all -ot lines
                    param_lines.append(line)
            
            print(f"DEBUG: Found {len(param_lines)} tensor override parameters", file=sys.stderr)
            return param_lines
            
        except Exception as e:
            print(f"DEBUG: Error reading tensor parameters from {params_file}: {e}", file=sys.stderr)
            return []
    
    def update_from_model_change(self):
        """Called when model selection changes in main launcher."""
        self._check_current_model()
        self._update_ui_state()
    
    def update_from_config_change(self):
        """Called when configuration changes in main launcher."""
        self._check_current_model()
        self._update_ui_state()
        self._update_gpu_controls_state()
        # Settings display is updated by _update_ui_state() -> _update_settings_display()
    
    def update_gpu_buffer_controls(self):
        """Update GPU buffer controls when GPU info changes."""
        self._setup_gpu_buffer_controls()
    
    def get_gpu_buffer_config(self):
        """Get current GPU buffer configuration for saving to config file."""
        config = {}
        for gpu_id, buffer_var in self.gpu_vram_buffers.items():
            try:
                buffer_mb = int(buffer_var.get().strip())
                config[f"gpu_{gpu_id}_buffer_mb"] = buffer_mb
            except (ValueError, AttributeError):
                config[f"gpu_{gpu_id}_buffer_mb"] = 512  # Default
        return config
    
    def load_gpu_buffer_config(self, config):
        """Load GPU buffer configuration from config file."""
        # First setup the controls to create the buffer variables
        self._setup_gpu_buffer_controls()
        
        # Then load the values
        for gpu_id, buffer_var in self.gpu_vram_buffers.items():
            config_key = f"gpu_{gpu_id}_buffer_mb"
            if config_key in config:
                try:
                    buffer_mb = int(config[config_key])
                    buffer_var.set(str(buffer_mb))
                except (ValueError, TypeError):
                    buffer_var.set("512")  # Default if invalid value
    
    def get_tensor_override_config(self):
        """Get complete tensor override configuration for saving."""
        config = {
            "tensor_override_enabled": self.tensor_override_enabled.get(),
            "tensor_override_gpu_buffers": self.get_gpu_buffer_config()
        }
        return config
    
    def load_tensor_override_config(self, config):
        """Load complete tensor override configuration."""
        if "tensor_override_enabled" in config:
            self.tensor_override_enabled.set(config["tensor_override_enabled"])
        
        if "tensor_override_gpu_buffers" in config:
            self.load_gpu_buffer_config(config["tensor_override_gpu_buffers"])
        
        # Update UI state after loading config
        self._update_ui_state()
        self._update_gpu_controls_state()
    
    def _run_vram_optimization(self):
        """Run automated VRAM optimization using llama.cpp verbose analysis."""
        try:
            # Update UI state
            self.root.after(0, lambda: self.analyze_button.config(state="disabled"))
            self.root.after(0, lambda: self.tensor_analysis_status.set("Optimizing VRAM usage..."))
            self.root.after(0, lambda: self.analysis_progress.set("Initializing llama.cpp analyzer..."))
            
            # Import the new analyzer
            from llama_verbose_tensor_analyzer import LlamaVerboseTensorAnalyzer
            from tensor_override_logic import TensorOverrideManager
            
            manager = TensorOverrideManager()
            
            # Get llama.cpp directory from launcher settings
            llama_cpp_dir = self.parent.llama_cpp_dir.get().strip()
            if not llama_cpp_dir:
                raise Exception("LLaMa.cpp directory not configured in launcher")
            
            # Detect GPU configuration with validation
            gpu_count = self._detect_gpu_count()
            total_vram_gb = self._estimate_total_vram()
            
            # Validate GPU configuration to prevent division by zero
            if gpu_count <= 0:
                raise Exception(f"Invalid GPU count detected: {gpu_count}. Please ensure GPUs are properly detected.")
            
            if total_vram_gb <= 0:
                raise Exception(f"Invalid VRAM amount detected: {total_vram_gb}GB. Please ensure GPU memory is properly detected.")
            
            vram_per_gpu_gb = total_vram_gb / gpu_count
            
            # Validate resulting VRAM per GPU
            if vram_per_gpu_gb <= 0:
                raise Exception(f"Invalid VRAM per GPU calculated: {vram_per_gpu_gb}GB. Check GPU detection.")
            
            self.root.after(0, lambda: self.analysis_progress.set(f"Initializing analyzer for {gpu_count} GPUs..."))
            
            # Initialize analyzer with detailed error reporting
            try:
                analyzer = LlamaVerboseTensorAnalyzer(llama_cpp_dir, verbose=True)
                self.root.after(0, lambda: self.analysis_progress.set(f"Found llama-server at: {analyzer.llama_server_path.name}"))
            except Exception as e:
                error_msg = f"Failed to initialize analyzer: {str(e)}"
                print(f"DEBUG: {error_msg}", file=sys.stderr)
                raise Exception(error_msg)
            
            # Run analysis and generate parameters
            self.root.after(0, lambda: self.analysis_progress.set("Extracting tensor information from model..."))
            
            optimized_params, tensor_info = analyzer.analyze_and_generate_params(
                self.current_model_path,
                gpu_count,
                vram_per_gpu_gb,
                timeout=300
            )
            
            if optimized_params:
                # Save optimized parameters with clean config name
                model_name = Path(self.current_model_path).stem
                clean_config_name = self.current_config_name.replace('_vram_optimized', '')
                params_filename = f"{model_name}_{clean_config_name}_tensor_params.txt"
                output_path = self.tensor_override_dir / params_filename
                
                analysis_info = {
                    'timestamp': str(Path().stat().st_mtime),
                    'optimization_type': 'llama_cpp_verbose',
                    'gpu_count': gpu_count,
                    'total_vram_gb': total_vram_gb,
                    'vram_per_gpu_gb': vram_per_gpu_gb,
                    'tensor_count': len(tensor_info),
                    'total_model_size_mb': sum(t.size_mb for t in tensor_info.values())
                }
                
                success = manager.save_tensor_override_params(
                    self.current_model_path, 
                    clean_config_name,
                    optimized_params,
                    analysis_info
                )
                
                if success:
                    self.root.after(0, lambda: self.analysis_progress.set("VRAM optimization complete!"))
                    self.root.after(0, self._analysis_success, str(output_path))
                    
                    # Store tensor info for result display
                    self.tensor_analysis_results = {
                        'tensors': tensor_info,
                        'gpu_count': gpu_count,
                        'total_vram_gb': total_vram_gb
                    }
                else:
                    raise Exception("Failed to save optimized parameters")
            else:
                raise Exception("No optimization parameters generated")
                
        except Exception as e:
            self.root.after(0, self._analysis_error, f"VRAM optimization failed: {str(e)}")
        finally:
            self.root.after(0, lambda: self.analyze_button.config(state="normal"))
            self.root.after(0, lambda: self.analysis_progress.set(""))
    
    def _detect_gpu_count(self):
        """Detect number of available GPUs."""
        try:
            if hasattr(self.parent, 'gpu_info') and self.parent.gpu_info:
                device_count = self.parent.gpu_info.get("device_count", 0)
                print(f"DEBUG: GPU device count from parent: {device_count}", file=sys.stderr)
                if device_count > 0:
                    return device_count
            
            # Fallback detection
            import subprocess
            print(f"DEBUG: Attempting nvidia-smi fallback detection", file=sys.stderr)
            result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
            if result.returncode == 0:
                gpu_lines = [line for line in result.stdout.split('\n') if 'GPU' in line]
                gpu_count = len(gpu_lines)
                print(f"DEBUG: nvidia-smi detected {gpu_count} GPUs", file=sys.stderr)
                if gpu_count > 0:
                    return gpu_count
        except Exception as e:
            print(f"DEBUG: Error in GPU detection: {e}", file=sys.stderr)
        
        # Default assumption if detection fails
        print(f"DEBUG: Using default GPU count assumption: 4", file=sys.stderr)
        return 4  # Default assumption for this model
    
    def _estimate_total_vram(self):
        """Estimate total available VRAM across all GPUs, accounting for per-GPU buffers."""
        try:
            if hasattr(self.parent, 'gpu_info') and self.parent.gpu_info:
                total_vram = 0
                devices = self.parent.gpu_info.get("devices", [])
                
                # Debug GPU info
                print(f"DEBUG: GPU info available, found {len(devices)} devices", file=sys.stderr)
                
                for gpu_id, gpu in enumerate(devices):
                    memory_total = gpu.get("total_memory_bytes", 0)
                    
                    # Get buffer for this GPU
                    buffer_mb = self._get_gpu_buffer_mb(gpu_id)
                    buffer_bytes = buffer_mb * 1024 * 1024  # Convert MB to bytes
                    
                    # Subtract buffer from available VRAM
                    available_memory = max(0, memory_total - buffer_bytes)
                    total_vram += available_memory
                    
                    print(f"DEBUG: GPU {gpu_id} has {memory_total} bytes memory, buffer {buffer_mb}MB, available {available_memory} bytes", file=sys.stderr)
                
                if total_vram > 0:
                    total_vram_gb = total_vram / (1024**3)  # Convert to GB
                    print(f"DEBUG: Total VRAM calculated (after buffers): {total_vram_gb:.1f}GB", file=sys.stderr)
                    return total_vram_gb
                else:
                    print(f"DEBUG: Total VRAM is 0, using default", file=sys.stderr)
        except Exception as e:
            print(f"DEBUG: Error in _estimate_total_vram: {e}", file=sys.stderr)
        
        # Default assumption if detection fails
        print(f"DEBUG: Using default VRAM assumption: 96GB", file=sys.stderr)
        return 96  # Default assumption (4x24GB)
    
    def _get_gpu_buffer_mb(self, gpu_id):
        """Get the buffer amount in MB for a specific GPU."""
        if gpu_id in self.gpu_vram_buffers:
            try:
                buffer_str = self.gpu_vram_buffers[gpu_id].get().strip()
                if buffer_str and buffer_str.isdigit():
                    return int(buffer_str)
                else:
                    return 512
            except (ValueError, AttributeError):
                return 512
        return 512  # Default 512MB buffer
    
    
    
    
    def _show_vram_optimization_results(self, param_lines):
        """Show VRAM optimization results summary."""
        self.results_text.config(state="normal")
        self.results_text.delete(1.0, tk.END)
        
        gpu_count = self._detect_gpu_count()
        total_vram_gb = self._estimate_total_vram()
        
        # Show optimization summary
        self.results_text.insert(tk.END, f"LLaMa.cpp Verbose Analysis Results\n")
        self.results_text.insert(tk.END, f"{'='*45}\n\n")
        self.results_text.insert(tk.END, f"Generated {len(param_lines)} tensor override parameters\n")
        self.results_text.insert(tk.END, f"Target GPUs: {gpu_count} devices\n")
        self.results_text.insert(tk.END, f"Total VRAM: ~{total_vram_gb:.1f}GB (after buffers)\n")
        self.results_text.insert(tk.END, f"Model: {Path(self.current_model_path).name}\n")
        self.results_text.insert(tk.END, f"Configuration: {self.current_config_name}\n\n")
        
        # Show buffer information
        self.results_text.insert(tk.END, f"VRAM Buffer Settings:\n")
        for gpu_id in range(gpu_count):
            buffer_mb = self._get_gpu_buffer_mb(gpu_id)
            self.results_text.insert(tk.END, f"- GPU {gpu_id}: {buffer_mb}MB buffer\n")
        self.results_text.insert(tk.END, f"\n")
        
        # Show tensor analysis details if available
        if hasattr(self, 'tensor_analysis_results'):
            results = self.tensor_analysis_results
            tensors = results.get('tensors', {})
            
            if tensors:
                total_size_mb = sum(t.size_mb for t in tensors.values())
                self.results_text.insert(tk.END, f"Model Analysis:\n")
                self.results_text.insert(tk.END, f"- Total tensors analyzed: {len(tensors)}\n")
                self.results_text.insert(tk.END, f"- Total model size: {total_size_mb:.1f} MiB ({total_size_mb/1024:.1f} GiB)\n")
                
                # Categorize tensors for display
                from llama_verbose_tensor_analyzer import LlamaVerboseTensorAnalyzer
                analyzer = LlamaVerboseTensorAnalyzer(".", verbose=False)  # Dummy for categorization
                categories = analyzer.categorize_tensors(tensors)
                
                self.results_text.insert(tk.END, f"\nTensor Categories:\n")
                for category, tensor_list in categories.items():
                    if tensor_list:
                        category_size = sum(t.size_mb for t in tensor_list)
                        self.results_text.insert(tk.END, f"- {category.title()}: {len(tensor_list)} tensors, {category_size:.1f} MiB\n")
                
                self.results_text.insert(tk.END, f"\n")
        
        # Count parameters by device
        device_counts = {}
        device_size_mb = {}
        for line in param_lines:
            if '=' in line:
                device = line.split('=')[-1].strip()
                device_counts[device] = device_counts.get(device, 0) + 1
                device_size_mb[device] = device_size_mb.get(device, 0.0)
                
                # Try to get tensor size if available
                if hasattr(self, 'tensor_analysis_results'):
                    tensors = self.tensor_analysis_results.get('tensors', {})
                    tensor_name = line.split('=')[0].replace('-ot ', '').strip()
                    if tensor_name in tensors:
                        device_size_mb[device] += tensors[tensor_name].size_mb
        
        self.results_text.insert(tk.END, "Tensor Distribution:\n")
        self.results_text.insert(tk.END, "-" * 25 + "\n")
        for device, count in sorted(device_counts.items()):
            size_mb = device_size_mb.get(device, 0.0)
            size_text = f" ({size_mb:.1f} MiB)" if size_mb > 0 else ""
            self.results_text.insert(tk.END, f"{device}: {count} tensors{size_text}\n")
        
        self.results_text.insert(tk.END, "\nOptimization Features:\n")
        self.results_text.insert(tk.END, "- Uses actual tensor names and sizes from llama.cpp\n")
        self.results_text.insert(tk.END, "- Prioritizes critical tensors for GPU placement\n")
        self.results_text.insert(tk.END, "- Balances VRAM usage across available GPUs\n")
        self.results_text.insert(tk.END, "- Keeps small tensors on CPU to maximize VRAM\n")
        self.results_text.insert(tk.END, "- Works with all model architectures\n")
        
        self.results_text.config(state="disabled")