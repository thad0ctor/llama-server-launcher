import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import json
from pathlib import Path

class LlamaCppLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("LLaMa.cpp Server Launcher")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # Default values
        self.llama_cpp_dir = tk.StringVar(value="")
        self.venv_dir = tk.StringVar(value="")
        self.model_path = tk.StringVar(value="")
        self.cache_type_k = tk.StringVar(value="f16")
        self.threads = tk.StringVar(value="4")
        self.n_gpu_layers = tk.StringVar(value="0")
        self.no_mmap = tk.BooleanVar(value=False)
        self.no_cnv = tk.BooleanVar(value=False)
        self.prio = tk.StringVar(value="0")
        self.temperature = tk.StringVar(value="0.8")
        self.min_p = tk.StringVar(value="0.05")
        self.ctx_size = tk.IntVar(value=2048)
        self.seed = tk.StringVar(value="-1")
        self.flash_attn = tk.BooleanVar(value=False)
        self.tensor_split = tk.StringVar(value="")
        self.main_gpu = tk.StringVar(value="0")
        self.device_var = tk.StringVar(value="")
        self.mlock = tk.BooleanVar(value=False)
        self.no_kv_offload = tk.BooleanVar(value=False)
        self.host = tk.StringVar(value="127.0.0.1")
        self.port = tk.StringVar(value="8080")
        
        # Save configurations
        self.config_name = tk.StringVar(value="default_config")
        self.saved_configs = {}
        self.load_saved_configs()
        
        self.create_widgets()
    
    def load_saved_configs(self):
        config_path = Path("llama_cpp_configs.json")
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    self.saved_configs = json.load(f)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load saved configurations: {e}")
    
    def save_configs(self):
        config_path = Path("llama_cpp_configs.json")
        try:
            with open(config_path, 'w') as f:
                json.dump(self.saved_configs, f, indent=2)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configurations: {e}")
    
    def create_widgets(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Main tab
        main_frame = ttk.Frame(notebook)
        notebook.add(main_frame, text="Main Settings")
        
        # Advanced tab
        advanced_frame = ttk.Frame(notebook)
        notebook.add(advanced_frame, text="Advanced Settings")
        
        # Configurations tab
        config_frame = ttk.Frame(notebook)
        notebook.add(config_frame, text="Configurations")
        
        # Main tab content
        self.setup_main_tab(main_frame)
        
        # Advanced tab content
        self.setup_advanced_tab(advanced_frame)
        
        # Configurations tab content
        self.setup_config_tab(config_frame)
        
        # Buttons frame at the bottom
        buttons_frame = ttk.Frame(self.root)
        buttons_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(buttons_frame, text="Launch Server", command=self.launch_server).pack(side='left', padx=5)
        ttk.Button(buttons_frame, text="Save PS1 Script", command=self.save_ps1_script).pack(side='left', padx=5)
        ttk.Button(buttons_frame, text="Exit", command=self.root.destroy).pack(side='right', padx=5)
    
    def setup_main_tab(self, parent):
        # Create a canvas with scrollbar
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Directories section
        ttk.Label(scrollable_frame, text="Directories", font=("TkDefaultFont", 12, "bold")).grid(column=0, row=0, sticky="w", padx=10, pady=(10, 5))
        ttk.Separator(scrollable_frame, orient='horizontal').grid(column=0, row=1, columnspan=3, sticky="ew", padx=10, pady=5)
        
        ttk.Label(scrollable_frame, text="LLaMa.cpp Directory:").grid(column=0, row=2, sticky="w", padx=10, pady=5)
        ttk.Entry(scrollable_frame, textvariable=self.llama_cpp_dir, width=50).grid(column=1, row=2, sticky="ew", padx=5, pady=5)
        ttk.Button(scrollable_frame, text="Browse...", command=lambda: self.browse_directory(self.llama_cpp_dir)).grid(column=2, row=2, padx=5, pady=5)
        
        ttk.Label(scrollable_frame, text="Virtual Environment (optional):").grid(column=0, row=3, sticky="w", padx=10, pady=5)
        ttk.Entry(scrollable_frame, textvariable=self.venv_dir, width=50).grid(column=1, row=3, sticky="ew", padx=5, pady=5)
        ttk.Button(scrollable_frame, text="Browse...", command=lambda: self.browse_directory(self.venv_dir)).grid(column=2, row=3, padx=5, pady=5)
        
        ttk.Label(scrollable_frame, text="Model File:").grid(column=0, row=4, sticky="w", padx=10, pady=5)
        ttk.Entry(scrollable_frame, textvariable=self.model_path, width=50).grid(column=1, row=4, sticky="ew", padx=5, pady=5)
        ttk.Button(scrollable_frame, text="Browse...", command=self.browse_model).grid(column=2, row=4, padx=5, pady=5)
        
        # Basic settings section
        ttk.Label(scrollable_frame, text="Basic Settings", font=("TkDefaultFont", 12, "bold")).grid(column=0, row=5, sticky="w", padx=10, pady=(20, 5))
        ttk.Separator(scrollable_frame, orient='horizontal').grid(column=0, row=6, columnspan=3, sticky="ew", padx=10, pady=5)
        
        ttk.Label(scrollable_frame, text="Threads:").grid(column=0, row=7, sticky="w", padx=10, pady=5)
        ttk.Entry(scrollable_frame, textvariable=self.threads, width=10).grid(column=1, row=7, sticky="w", padx=5, pady=5)
        
        ttk.Label(scrollable_frame, text="Context Size:").grid(column=0, row=8, sticky="w", padx=10, pady=5)
        
        # Create a frame for the context size slider and its value display
        ctx_frame = ttk.Frame(scrollable_frame)
        ctx_frame.grid(column=1, row=8, sticky="ew", padx=5, pady=5, columnspan=2)
        
        # Add a slider for context size
        ctx_slider = ttk.Scale(ctx_frame, from_=1024, to=1000000, 
                              orient="horizontal", variable=self.ctx_size,
                              command=lambda v: self.update_ctx_label())
        ctx_slider.grid(column=0, row=0, sticky="ew", padx=5)
        
        # Set the increment value
        ctx_slider.configure(value=2048)
        
        # Add a label to display the current context size value
        self.ctx_label = ttk.Label(ctx_frame, text="2048", width=10)
        self.ctx_label.grid(column=1, row=0, padx=5)
        
        # Add an entry field for manual override
        self.ctx_entry = ttk.Entry(ctx_frame, width=10)
        self.ctx_entry.grid(column=2, row=0, padx=5)
        self.ctx_entry.insert(0, "2048")
        
        # Add a button to apply the manual override
        ttk.Button(ctx_frame, text="Set", command=self.override_ctx_size).grid(column=3, row=0, padx=5)
        
        # Configure grid weights to make the slider expand
        ctx_frame.columnconfigure(0, weight=3)
        ctx_frame.columnconfigure(1, weight=0)
        ctx_frame.columnconfigure(2, weight=0)
        ctx_frame.columnconfigure(3, weight=0)
        
        ttk.Label(scrollable_frame, text="Temperature:").grid(column=0, row=9, sticky="w", padx=10, pady=5)
        ttk.Entry(scrollable_frame, textvariable=self.temperature, width=10).grid(column=1, row=9, sticky="w", padx=5, pady=5)
        
        ttk.Label(scrollable_frame, text="Min P:").grid(column=0, row=10, sticky="w", padx=10, pady=5)
        ttk.Entry(scrollable_frame, textvariable=self.min_p, width=10).grid(column=1, row=10, sticky="w", padx=5, pady=5)
        
        ttk.Label(scrollable_frame, text="Seed:").grid(column=0, row=11, sticky="w", padx=10, pady=5)
        ttk.Entry(scrollable_frame, textvariable=self.seed, width=10).grid(column=1, row=11, sticky="w", padx=5, pady=5)
        
        # Network settings
        ttk.Label(scrollable_frame, text="Network Settings", font=("TkDefaultFont", 12, "bold")).grid(column=0, row=12, sticky="w", padx=10, pady=(20, 5))
        ttk.Separator(scrollable_frame, orient='horizontal').grid(column=0, row=13, columnspan=3, sticky="ew", padx=10, pady=5)
        
        ttk.Label(scrollable_frame, text="Host:").grid(column=0, row=14, sticky="w", padx=10, pady=5)
        ttk.Entry(scrollable_frame, textvariable=self.host, width=20).grid(column=1, row=14, sticky="w", padx=5, pady=5)
        
        ttk.Label(scrollable_frame, text="Port:").grid(column=0, row=15, sticky="w", padx=10, pady=5)
        ttk.Entry(scrollable_frame, textvariable=self.port, width=10).grid(column=1, row=15, sticky="w", padx=5, pady=5)
        
        # Make the frame expandable
        scrollable_frame.columnconfigure(1, weight=1)
    
    def setup_advanced_tab(self, parent):
        # Create a canvas with scrollbar
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # GPU Settings section
        ttk.Label(scrollable_frame, text="GPU Settings", font=("TkDefaultFont", 12, "bold")).grid(column=0, row=0, sticky="w", padx=10, pady=(10, 5))
        ttk.Separator(scrollable_frame, orient='horizontal').grid(column=0, row=1, columnspan=3, sticky="ew", padx=10, pady=5)
        
        ttk.Label(scrollable_frame, text="GPU Layers:").grid(column=0, row=2, sticky="w", padx=10, pady=5)
        ttk.Entry(scrollable_frame, textvariable=self.n_gpu_layers, width=10).grid(column=1, row=2, sticky="w", padx=5, pady=5)
        ttk.Label(scrollable_frame, text="(use 0 for CPU-only, or a number ≥ 1 for GPU)", font=("TkDefaultFont", 8)).grid(column=2, row=2, sticky="w", padx=5, pady=5)
        
        ttk.Label(scrollable_frame, text="Main GPU Index:").grid(column=0, row=3, sticky="w", padx=10, pady=5)
        ttk.Entry(scrollable_frame, textvariable=self.main_gpu, width=10).grid(column=1, row=3, sticky="w", padx=5, pady=5)
        ttk.Label(scrollable_frame, text="(default: 0, first GPU)", font=("TkDefaultFont", 8)).grid(column=2, row=3, sticky="w", padx=5, pady=5)
        
        ttk.Label(scrollable_frame, text="Tensor Split:").grid(column=0, row=4, sticky="w", padx=10, pady=5)
        ttk.Entry(scrollable_frame, textvariable=self.tensor_split, width=20).grid(column=1, row=4, sticky="w", padx=5, pady=5)
        ttk.Label(scrollable_frame, text="(e.g., '3,1' divides model between 2 GPUs)", font=("TkDefaultFont", 8)).grid(column=2, row=4, sticky="w", padx=5, pady=5)
        
        # Add a device selection
        ttk.Label(scrollable_frame, text="GPU Devices:").grid(column=0, row=5, sticky="w", padx=10, pady=5)
        self.device_var = tk.StringVar(value="")
        ttk.Entry(scrollable_frame, textvariable=self.device_var, width=20).grid(column=1, row=5, sticky="w", padx=5, pady=5)
        ttk.Label(scrollable_frame, text="(e.g., '0,1,2' to use GPUs 0,1,2)", font=("TkDefaultFont", 8)).grid(column=2, row=5, sticky="w", padx=5, pady=5)
        
        ttk.Label(scrollable_frame, text="Flash Attention:").grid(column=0, row=6, sticky="w", padx=10, pady=5)
        ttk.Checkbutton(scrollable_frame, variable=self.flash_attn).grid(column=1, row=6, sticky="w", padx=5, pady=5)
        ttk.Label(scrollable_frame, text="(faster GPU attention; requires compatible GPU)", font=("TkDefaultFont", 8)).grid(column=2, row=6, sticky="w", padx=5, pady=5)
        
        # Memory Settings
        ttk.Label(scrollable_frame, text="Memory Settings", font=("TkDefaultFont", 12, "bold")).grid(column=0, row=7, sticky="w", padx=10, pady=(20, 5))
        ttk.Separator(scrollable_frame, orient='horizontal').grid(column=0, row=8, columnspan=3, sticky="ew", padx=10, pady=5)
        
        ttk.Label(scrollable_frame, text="Cache Type K:").grid(column=0, row=9, sticky="w", padx=10, pady=5)
        cache_combobox = ttk.Combobox(scrollable_frame, textvariable=self.cache_type_k, width=10)
        cache_combobox['values'] = ('f16', 'f32', 'q8_0', 'q4_0', 'q4_1', 'q5_0', 'q5_1')
        cache_combobox.grid(column=1, row=9, sticky="w", padx=5, pady=5)
        ttk.Label(scrollable_frame, text="(quantization for KV cache; q8_0 uses less VRAM)", font=("TkDefaultFont", 8)).grid(column=2, row=9, sticky="w", padx=5, pady=5)
        
        ttk.Label(scrollable_frame, text="Memory-Map Model:").grid(column=0, row=10, sticky="w", padx=10, pady=5)
        ttk.Radiobutton(scrollable_frame, text="Yes (default)", variable=self.no_mmap, value=False).grid(column=1, row=10, sticky="w", padx=5, pady=5)
        ttk.Label(scrollable_frame, text="(faster loading, allows OS to manage memory)", font=("TkDefaultFont", 8)).grid(column=2, row=10, sticky="w", padx=5, pady=5)
        
        ttk.Radiobutton(scrollable_frame, text="No", variable=self.no_mmap, value=True).grid(column=1, row=11, sticky="w", padx=5, pady=5)
        ttk.Label(scrollable_frame, text="(loads entire model into RAM; slower load, no pageouts)", font=("TkDefaultFont", 8)).grid(column=2, row=11, sticky="w", padx=5, pady=5)
        
        # Add mlock option
        self.mlock = tk.BooleanVar(value=False)
        ttk.Label(scrollable_frame, text="Force Keep in RAM:").grid(column=0, row=12, sticky="w", padx=10, pady=5)
        ttk.Checkbutton(scrollable_frame, variable=self.mlock).grid(column=1, row=12, sticky="w", padx=5, pady=5)
        ttk.Label(scrollable_frame, text="(--mlock: prevent model from being swapped to disk)", font=("TkDefaultFont", 8)).grid(column=2, row=12, sticky="w", padx=5, pady=5)
        
        # Add KV offload option
        self.no_kv_offload = tk.BooleanVar(value=False)
        ttk.Label(scrollable_frame, text="Disable KV Offload:").grid(column=0, row=13, sticky="w", padx=10, pady=5)
        ttk.Checkbutton(scrollable_frame, variable=self.no_kv_offload).grid(column=1, row=13, sticky="w", padx=5, pady=5)
        ttk.Label(scrollable_frame, text="(--no-kv-offload: keep KV cache in GPU memory only)", font=("TkDefaultFont", 8)).grid(column=2, row=13, sticky="w", padx=5, pady=5)
        
        ttk.Label(scrollable_frame, text="No Convolution:").grid(column=0, row=14, sticky="w", padx=10, pady=5)
        ttk.Checkbutton(scrollable_frame, variable=self.no_cnv).grid(column=1, row=14, sticky="w", padx=5, pady=5)
        ttk.Label(scrollable_frame, text="(-no-cnv: may help with certain GPUs)", font=("TkDefaultFont", 8)).grid(column=2, row=14, sticky="w", padx=5, pady=5)
        
        # Performance Settings
        ttk.Label(scrollable_frame, text="Performance Settings", font=("TkDefaultFont", 12, "bold")).grid(column=0, row=15, sticky="w", padx=10, pady=(20, 5))
        ttk.Separator(scrollable_frame, orient='horizontal').grid(column=0, row=16, columnspan=3, sticky="ew", padx=10, pady=5)
        
        ttk.Label(scrollable_frame, text="Priority:").grid(column=0, row=17, sticky="w", padx=10, pady=5)
        prio_combobox = ttk.Combobox(scrollable_frame, textvariable=self.prio, width=10)
        prio_combobox['values'] = ('0', '1', '2', '3')
        prio_combobox.grid(column=1, row=17, sticky="w", padx=5, pady=5)
        ttk.Label(scrollable_frame, text="(0=normal, 1=medium, 2=high, 3=realtime)", font=("TkDefaultFont", 8)).grid(column=2, row=17, sticky="w", padx=5, pady=5)
        
        # Make the frame expandable
        scrollable_frame.columnconfigure(1, weight=1)
    
    def setup_config_tab(self, parent):
        frame = ttk.Frame(parent, padding=10)
        frame.pack(fill='both', expand=True)
        
        ttk.Label(frame, text="Save Current Configuration").grid(column=0, row=0, sticky="w", padx=5, pady=5, columnspan=2)
        
        ttk.Label(frame, text="Configuration Name:").grid(column=0, row=1, sticky="w", padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.config_name, width=30).grid(column=1, row=1, sticky="ew", padx=5, pady=5)
        ttk.Button(frame, text="Save Configuration", command=self.save_configuration).grid(column=2, row=1, padx=5, pady=5)
        
        ttk.Label(frame, text="Saved Configurations:").grid(column=0, row=2, sticky="w", padx=5, pady=(20, 5), columnspan=2)
        
        # Listbox with scrollbar for saved configurations
        listbox_frame = ttk.Frame(frame)
        listbox_frame.grid(column=0, row=3, columnspan=3, sticky="nsew", padx=5, pady=5)
        
        scrollbar = ttk.Scrollbar(listbox_frame)
        scrollbar.pack(side='right', fill='y')
        
        self.config_listbox = tk.Listbox(listbox_frame, yscrollcommand=scrollbar.set, height=15)
        self.config_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.config_listbox.yview)
        
        # Buttons for configuration management
        buttons_frame = ttk.Frame(frame)
        buttons_frame.grid(column=0, row=4, columnspan=3, sticky="ew", padx=5, pady=5)
        
        ttk.Button(buttons_frame, text="Load Configuration", command=self.load_configuration).pack(side='left', padx=5)
        ttk.Button(buttons_frame, text="Delete Configuration", command=self.delete_configuration).pack(side='left', padx=5)
        
        # Update the listbox with saved configurations
        self.update_config_listbox()
        
        # Make frames expandable
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(3, weight=1)
    
    def update_config_listbox(self):
        self.config_listbox.delete(0, tk.END)
        for config_name in self.saved_configs.keys():
            self.config_listbox.insert(tk.END, config_name)
            
    def update_ctx_label(self):
        # Round to nearest 1024
        value = self.ctx_size.get()
        rounded_value = round(value / 1024) * 1024
        if rounded_value < 1024:
            rounded_value = 1024
        
        # Update the scale and label
        self.ctx_size.set(rounded_value)
        self.ctx_label.config(text=f"{rounded_value:,}")
        self.ctx_entry.delete(0, tk.END)
        self.ctx_entry.insert(0, str(rounded_value))
        
    def override_ctx_size(self):
        try:
            # Get the value from the entry field
            value = int(self.ctx_entry.get())
            
            # Ensure the value is at least 1024 and a multiple of 1024
            if value < 1024:
                value = 1024
            else:
                value = round(value / 1024) * 1024
                
            # Update the context size
            self.ctx_size.set(value)
            self.ctx_label.config(text=f"{value:,}")
            self.ctx_entry.delete(0, tk.END)
            self.ctx_entry.insert(0, str(value))
        except ValueError:
            # If the entry is not a valid number, reset to the current context size
            messagebox.showerror("Error", "Please enter a valid number for context size")
            self.ctx_entry.delete(0, tk.END)
            self.ctx_entry.insert(0, str(self.ctx_size.get()))
    
    def save_configuration(self):
        config_name = self.config_name.get()
        if not config_name:
            messagebox.showerror("Error", "Please enter a configuration name")
            return
        
        config = {
            'llama_cpp_dir': self.llama_cpp_dir.get(),
            'venv_dir': self.venv_dir.get(),
            'model_path': self.model_path.get(),
            'cache_type_k': self.cache_type_k.get(),
            'threads': self.threads.get(),
            'n_gpu_layers': self.n_gpu_layers.get(),
            'no_mmap': self.no_mmap.get(),
            'no_cnv': self.no_cnv.get(),
            'prio': self.prio.get(),
            'temperature': self.temperature.get(),
            'min_p': self.min_p.get(),
            'ctx_size': self.ctx_size.get(),
            'seed': self.seed.get(),
            'flash_attn': self.flash_attn.get(),
            'tensor_split': self.tensor_split.get(),
            'main_gpu': self.main_gpu.get(),
            'device_var': self.device_var.get(),
            'mlock': self.mlock.get(),
            'no_kv_offload': self.no_kv_offload.get(),
            'host': self.host.get(),
            'port': self.port.get()
        }
        
        self.saved_configs[config_name] = config
        self.save_configs()
        self.update_config_listbox()
        messagebox.showinfo("Success", f"Configuration '{config_name}' saved successfully")
    
    def load_configuration(self):
        selected_idx = self.config_listbox.curselection()
        if not selected_idx:
            messagebox.showerror("Error", "Please select a configuration to load")
            return
        
        config_name = self.config_listbox.get(selected_idx)
        config = self.saved_configs.get(config_name)
        if not config:
            messagebox.showerror("Error", f"Configuration '{config_name}' not found")
            return
        
        # Load the configuration values
        self.llama_cpp_dir.set(config.get('llama_cpp_dir', ''))
        self.venv_dir.set(config.get('venv_dir', ''))
        self.model_path.set(config.get('model_path', ''))
        self.cache_type_k.set(config.get('cache_type_k', 'f16'))
        self.threads.set(config.get('threads', '4'))
        self.n_gpu_layers.set(config.get('n_gpu_layers', '0'))
        self.no_mmap.set(config.get('no_mmap', False))
        self.no_cnv.set(config.get('no_cnv', False))
        self.prio.set(config.get('prio', '0'))
        self.temperature.set(config.get('temperature', '0.8'))
        self.min_p.set(config.get('min_p', '0.05'))
        self.ctx_size.set(config.get('ctx_size', 2048))
        self.seed.set(config.get('seed', '-1'))
        self.flash_attn.set(config.get('flash_attn', False))
        self.tensor_split.set(config.get('tensor_split', ''))
        self.main_gpu.set(config.get('main_gpu', '0'))
        self.device_var.set(config.get('device_var', ''))
        self.mlock.set(config.get('mlock', False))
        self.no_kv_offload.set(config.get('no_kv_offload', False))
        self.host.set(config.get('host', '127.0.0.1'))
        self.port.set(config.get('port', '8080'))
        
        # Update context size label
        self.update_ctx_label()
        
        messagebox.showinfo("Success", f"Configuration '{config_name}' loaded successfully")
    
    def delete_configuration(self):
        selected_idx = self.config_listbox.curselection()
        if not selected_idx:
            messagebox.showerror("Error", "Please select a configuration to delete")
            return
        
        config_name = self.config_listbox.get(selected_idx)
        if config_name in self.saved_configs:
            confirm = messagebox.askyesno("Confirm", f"Are you sure you want to delete configuration '{config_name}'?")
            if confirm:
                del self.saved_configs[config_name]
                self.save_configs()
                self.update_config_listbox()
                messagebox.showinfo("Success", f"Configuration '{config_name}' deleted successfully")
    
    def browse_directory(self, string_var):
        directory = filedialog.askdirectory()
        if directory:
            string_var.set(directory)
    
    def browse_model(self):
        file_types = [("GGUF Files", "*.gguf"), ("All Files", "*.*")]
        file_path = filedialog.askopenfilename(filetypes=file_types)
        if file_path:
            self.model_path.set(file_path)
    
    def build_command_line(self):
        cmd_parts = []
        
        # Path to llama-server executable
        llama_cpp_dir = self.llama_cpp_dir.get()
        if not llama_cpp_dir:
            messagebox.showerror("Error", "LLaMa.cpp directory is required")
            return None
        
        if sys.platform == "win32":
            executable = os.path.join(llama_cpp_dir, "llama-server.exe")
        else:
            executable = os.path.join(llama_cpp_dir, "llama-server")
        
        cmd_parts.append(executable)
        
        # Model path
        model_path = self.model_path.get()
        if not model_path:
            messagebox.showerror("Error", "Model file is required")
            return None
        
        cmd_parts.extend(["-m", model_path])
        
        # Add other parameters
        if self.cache_type_k.get() != "f16":  # f16 is default
            cmd_parts.extend(["--cache-type-k", self.cache_type_k.get()])
        
        if self.threads.get() != "4":  # 4 is default
            cmd_parts.extend(["--threads", self.threads.get()])
        
        if self.n_gpu_layers.get() != "0":  # 0 is default
            cmd_parts.extend(["--n-gpu-layers", self.n_gpu_layers.get()])
        
        if self.no_mmap.get():
            cmd_parts.append("--no-mmap")
        
        if self.mlock.get():
            cmd_parts.append("--mlock")
        
        if self.no_kv_offload.get():
            cmd_parts.append("--no-kv-offload")
        
        if self.device_var.get():
            cmd_parts.extend(["--device", self.device_var.get()])
        
        if self.no_cnv.get():
            cmd_parts.append("-no-cnv")
        
        if self.prio.get() != "0":  # 0 is default
            cmd_parts.extend(["--prio", self.prio.get()])
        
        if self.temperature.get() != "0.8":  # 0.8 is default
            cmd_parts.extend(["--temp", self.temperature.get()])
        
        if self.min_p.get() != "0.05":  # 0.05 is default
            cmd_parts.extend(["--min-p", self.min_p.get()])
        
        # Context size (always include as it's a slider now)
        cmd_parts.extend(["--ctx-size", str(self.ctx_size.get())])
        
        if self.seed.get() != "-1":  # -1 is default
            cmd_parts.extend(["--seed", self.seed.get()])
        
        if self.flash_attn.get():
            cmd_parts.append("--flash-attn")
        
        if self.tensor_split.get():
            cmd_parts.extend(["--tensor-split", self.tensor_split.get()])
        
        if self.main_gpu.get() != "0":  # 0 is default
            cmd_parts.extend(["--main-gpu", self.main_gpu.get()])
        
        if self.host.get() != "127.0.0.1":  # 127.0.0.1 is default
            cmd_parts.extend(["--host", self.host.get()])
        
        if self.port.get() != "8080":  # 8080 is default
            cmd_parts.extend(["--port", self.port.get()])
        
        return cmd_parts
    
    def launch_server(self):
        cmd_parts = self.build_command_line()
        if not cmd_parts:
            return
        
        venv_dir = self.venv_dir.get()
        
        try:
            if venv_dir and sys.platform == "win32":
                # Activate the virtual environment on Windows and then run the command
                activate_script = os.path.join(venv_dir, "Scripts", "activate.bat")
                full_cmd = f'start cmd /k ""{activate_script}" && {" ".join(cmd_parts)}"'
                subprocess.Popen(full_cmd, shell=True)
            elif venv_dir:
                # On Unix-like systems (Linux/macOS)
                activate_script = os.path.join(venv_dir, "bin", "activate")
                full_cmd = f'gnome-terminal -- bash -c "source {activate_script} && {" ".join(cmd_parts)}; exec bash"'
                subprocess.Popen(full_cmd, shell=True)
            else:
                # Just run the server without virtual environment
                if sys.platform == "win32":
                    subprocess.Popen(cmd_parts)
                else:
                    subprocess.Popen(['gnome-terminal', '--', *cmd_parts])
            
            messagebox.showinfo("Success", "LLaMa.cpp server launched successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch server: {e}")
    
    def save_ps1_script(self):
        cmd_parts = self.build_command_line()
        if not cmd_parts:
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".ps1",
            filetypes=[("PowerShell Script", "*.ps1"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
        
        venv_dir = self.venv_dir.get()
        
        try:
            with open(file_path, 'w') as f:
                f.write("# LLaMa.cpp Server Launch Script\n")
                f.write("# Generated by LLaMa.cpp Server Launcher\n\n")
                
                if venv_dir:
                    if sys.platform == "win32":
                        # Activate the virtual environment on Windows
                        activate_script = os.path.join(venv_dir, "Scripts", "activate.ps1")
                        f.write(f'. "{activate_script}"\n\n')
                    else:
                        # For PowerShell on Unix-like systems
                        activate_script = os.path.join(venv_dir, "bin", "Activate.ps1")
                        f.write(f'. "{activate_script}"\n\n')
                
                # Write the command
                f.write("# Launch LLaMa.cpp Server\n")
                f.write(f'& {" ".join(cmd_parts)}\n')
            
            messagebox.showinfo("Success", f"PowerShell script saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save PowerShell script: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = LlamaCppLauncher(root)
    root.mainloop()