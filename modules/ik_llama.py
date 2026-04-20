#!/usr/bin/env python3
"""
ik_llama Configuration Tab Module

This module provides functionality to manage ik_llama-specific configuration options
that will be used when launching the ik_llama server. It includes the RTR (Real-Time
Reasoning) and FMoE (Fused Mixture of Experts) flags.
"""

import tkinter as tk
from tkinter import ttk


class IkLlamaTab:
    """Manages ik_llama-specific configuration options."""
    
    def __init__(self, launcher_instance):
        """
        Initialize the ik_llama tab.
        
        Args:
            launcher_instance: Reference to the main launcher instance
        """
        self.launcher = launcher_instance
        
        # ik_llama specific flags as BooleanVar
        self.rtr_enabled = tk.BooleanVar(value=False)
        self.fmoe_enabled = tk.BooleanVar(value=False)
        
        # New ik_llama configuration options
        self.ser_value = tk.StringVar(value="")  # Smart Expert Reduction
        self.amb_value = tk.StringVar(value="")  # Attention Max Batch
        self.ctk_value = tk.StringVar(value="f16")  # KV Cache Type K
        self.ctv_value = tk.StringVar(value="f16")  # KV Cache Type V
        
        # Set up trace bindings for config saving
        self.rtr_enabled.trace_add("write", lambda *args: self.launcher._save_configs())
        self.fmoe_enabled.trace_add("write", lambda *args: self.launcher._save_configs())
        self.ser_value.trace_add("write", lambda *args: self.launcher._save_configs())
        self.amb_value.trace_add("write", lambda *args: self.launcher._save_configs())
        self.ctk_value.trace_add("write", lambda *args: self.launcher._save_configs())
        self.ctv_value.trace_add("write", lambda *args: self.launcher._save_configs())
        
        # Also trigger default config name updates
        self.rtr_enabled.trace_add("write", lambda *args: self.launcher._update_default_config_name_if_needed())
        self.fmoe_enabled.trace_add("write", lambda *args: self.launcher._update_default_config_name_if_needed())
        self.ser_value.trace_add("write", lambda *args: self.launcher._update_default_config_name_if_needed())
        self.amb_value.trace_add("write", lambda *args: self.launcher._update_default_config_name_if_needed())
        self.ctk_value.trace_add("write", lambda *args: self.launcher._update_default_config_name_if_needed())
        self.ctv_value.trace_add("write", lambda *args: self.launcher._update_default_config_name_if_needed())
    
    def create_tab(self, parent):
        """
        Create the ik_llama configuration tab UI.
        
        Args:
            parent: Parent tkinter frame for this tab
        """
        # Create scrollable frame
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
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
        
        # Main header
        ttk.Label(scrollable_frame, text="ik_llama Specific Features", 
                 font=("TkDefaultFont", 12, "bold")).grid(
            column=0, row=row, sticky="w", padx=10, pady=(10, 5), columnspan=3)
        row += 1
        
        ttk.Separator(scrollable_frame, orient="horizontal").grid(
            column=0, row=row, columnspan=3, sticky="ew", padx=10, pady=5)
        row += 1
        
        # Description
        desc_text = ("ik_llama provides advanced features for enhanced model performance. "
                    "Enable the options below to use these specialized optimizations.")
        ttk.Label(scrollable_frame, text=desc_text, font=("TkSmallCaptionFont",), 
                 wraplength=700).grid(
            column=0, row=row, sticky="w", padx=10, pady=5, columnspan=3)
        row += 1
        
        # RTR (Real-Time Reasoning) option
        self.rtr_checkbox = ttk.Checkbutton(
            scrollable_frame, 
            text="Enable RTR (-rtr)", 
            variable=self.rtr_enabled
        )
        self.rtr_checkbox.grid(column=0, row=row, sticky="w", padx=10, pady=5)
        
        ttk.Label(scrollable_frame, text="Run Time Repack - repacks quants for improved performance", 
                 font=("TkSmallCaptionFont",)).grid(
            column=1, row=row, sticky="w", padx=5, pady=5, columnspan=2)
        row += 1
        
        # FMoE (FusedMixture of Experts) option
        self.fmoe_checkbox = ttk.Checkbutton(
            scrollable_frame, 
            text="Enable FMoE (-fmoe)", 
            variable=self.fmoe_enabled
        )
        self.fmoe_checkbox.grid(column=0, row=row, sticky="w", padx=10, pady=5)
        
        ttk.Label(scrollable_frame, text="Fused MoE - optimized for CUDA and some CPU configs", 
                 font=("TkSmallCaptionFont",)).grid(
            column=1, row=row, sticky="w", padx=5, pady=5, columnspan=2)
        row += 1
        
        # Smart Expert Reduction option
        ttk.Label(scrollable_frame, text="Smart Expert Reduction (-ser):", 
                 font=("TkDefaultFont", 10, "bold")).grid(
            column=0, row=row, sticky="w", padx=10, pady=(15, 5))
        row += 1
        
        ttk.Label(scrollable_frame, text="Format: i,f (e.g., 7,1 or 6,1 or 5,1)", 
                 font=("TkSmallCaptionFont",)).grid(
            column=0, row=row, sticky="w", padx=10, pady=0, columnspan=2)
        row += 1
        
        self.ser_entry = ttk.Entry(scrollable_frame, textvariable=self.ser_value, width=20)
        self.ser_entry.grid(column=0, row=row, sticky="w", padx=10, pady=5)
        
        ttk.Label(scrollable_frame, text="Trade off quality for speed (leave empty to disable)", 
                 font=("TkSmallCaptionFont",)).grid(
            column=1, row=row, sticky="w", padx=5, pady=5, columnspan=2)
        row += 1
        
        # Attention Max Batch option
        ttk.Label(scrollable_frame, text="K*Q Tensor Compute Buffer (-amb):", 
                 font=("TkDefaultFont", 10, "bold")).grid(
            column=0, row=row, sticky="w", padx=10, pady=(15, 5))
        row += 1
        
        ttk.Label(scrollable_frame, text="Size in MiB (e.g., 512)", 
                 font=("TkSmallCaptionFont",)).grid(
            column=0, row=row, sticky="w", padx=10, pady=0, columnspan=2)
        row += 1
        
        self.amb_entry = ttk.Entry(scrollable_frame, textvariable=self.amb_value, width=20)
        self.amb_entry.grid(column=0, row=row, sticky="w", padx=10, pady=5)
        
        ttk.Label(scrollable_frame, text="Good for models like DeepSeek-R1 671B (leave empty to disable)", 
                 font=("TkSmallCaptionFont",)).grid(
            column=1, row=row, sticky="w", padx=5, pady=5, columnspan=2)
        row += 1
        
        # KV Cache Quantization option
        ttk.Label(scrollable_frame, text="KV Cache Type K (-ctk):", 
                 font=("TkDefaultFont", 10, "bold")).grid(
            column=0, row=row, sticky="w", padx=10, pady=(15, 5))
        row += 1
        
        # Full precision and quantized cache type options
        kv_cache_options = ["f16", "f32", "bf16", "q4_0", "q4_1", "q5_0", "q5_1", "q6_0", "q8_0", "iq4_nl", "q8_KV"]
        self.ctk_combo = ttk.Combobox(scrollable_frame, textvariable=self.ctk_value, 
                                     values=kv_cache_options, state="readonly", width=18)
        self.ctk_combo.grid(column=0, row=row, sticky="w", padx=10, pady=5)
        
        ttk.Label(scrollable_frame, text="KV cache data type for K tensor (precision or quantized)", 
                 font=("TkSmallCaptionFont",)).grid(
            column=1, row=row, sticky="w", padx=5, pady=5, columnspan=2)
        row += 1
        
        # KV Cache Type V option
        ttk.Label(scrollable_frame, text="KV Cache Type V (-ctv):", 
                 font=("TkDefaultFont", 10, "bold")).grid(
            column=0, row=row, sticky="w", padx=10, pady=(15, 5))
        row += 1
        
        # Use the same cache type options for V cache
        self.ctv_combo = ttk.Combobox(scrollable_frame, textvariable=self.ctv_value, 
                                     values=kv_cache_options, state="readonly", width=18)
        self.ctv_combo.grid(column=0, row=row, sticky="w", padx=10, pady=5)
        
        ttk.Label(scrollable_frame, text="KV cache data type for V tensor (precision or quantized)", 
                 font=("TkSmallCaptionFont",)).grid(
            column=1, row=row, sticky="w", padx=5, pady=5, columnspan=2)
        row += 1
        
        # Help section
        row += 1
        ttk.Label(scrollable_frame, text="Help & Documentation", 
                 font=("TkDefaultFont", 11, "bold")).grid(
            column=0, row=row, sticky="w", padx=10, pady=(20, 5), columnspan=3)
        row += 1
        
        ttk.Separator(scrollable_frame, orient="horizontal").grid(
            column=0, row=row, columnspan=3, sticky="ew", padx=10, pady=5)
        row += 1
        
        help_text = ("• RTR (-rtr): Run Time Repack - repacks quants for improved performance on certain hardware configs\n"
                    "  NOTE: Disables mmap, requires enough RAM to malloc all repacked quants (good for hybrid GPU+CPU)\n"
                    "• FMoE (-fmoe): Fused MoE - optimized mixture of experts for CUDA and some CPU configurations\n"
                    "• SER (-ser): Smart expert reduction trades quality for speed (format: experts,factor)\n"
                    "• AMB (-amb): Sets K*Q tensor compute buffer size in MiB for memory optimization\n"
                    "• CTK (-ctk): Sets KV cache type for K tensor (f16/f32/bf16/q4_0/q4_1/q5_0/q5_1/q6_0/q8_0/iq4_nl/q8_KV)\n"
                    "• CTV (-ctv): Sets KV cache type for V tensor (f16/f32/bf16/q4_0/q4_1/q5_0/q5_1/q6_0/q8_0/iq4_nl/q8_KV)\n"
                    "• These flags are specific to ik_llama and will be ignored by standard llama.cpp")
        ttk.Label(scrollable_frame, text=help_text, font=("TkSmallCaptionFont",), 
                 wraplength=700, justify="left").grid(
            column=0, row=row, sticky="w", padx=10, pady=5, columnspan=3)
        row += 1
    
    def get_ik_llama_flags(self):
        """
        Get list of ik_llama flags for command building.
        
        Returns:
            List of flag strings to be added to the command
        """
        flags = []
        
        if self.rtr_enabled.get():
            flags.append("-rtr")
        
        if self.fmoe_enabled.get():
            flags.append("-fmoe")
        
        # Smart Expert Reduction
        ser_val = self.ser_value.get().strip()
        if ser_val:
            flags.extend(["-ser", ser_val])
        
        # Attention Max Batch
        amb_val = self.amb_value.get().strip()
        if amb_val:
            flags.extend(["-amb", amb_val])
        
        # KV Cache Type K
        ctk_val = self.ctk_value.get().strip()
        if ctk_val and ctk_val != "f16":  # f16 is default, no need to specify
            flags.extend(["-ctk", ctk_val])
        
        # KV Cache Type V
        ctv_val = self.ctv_value.get().strip()
        if ctv_val and ctv_val != "f16":  # f16 is default, no need to specify
            flags.extend(["-ctv", ctv_val])
        
        return flags
    
    def save_to_config(self):
        """
        Save ik_llama settings to configuration dictionary.
        
        Returns:
            Dictionary containing ik_llama configuration
        """
        return {
            "ik_llama_rtr_enabled": self.rtr_enabled.get(),
            "ik_llama_fmoe_enabled": self.fmoe_enabled.get(),
            "ik_llama_ser_value": self.ser_value.get(),
            "ik_llama_amb_value": self.amb_value.get(),
            "ik_llama_ctk_value": self.ctk_value.get(),
            "ik_llama_ctv_value": self.ctv_value.get()
        }
    
    def load_from_config(self, config_data):
        """
        Load ik_llama settings from configuration data.
        
        Args:
            config_data: Dictionary containing configuration data
        """
        self.rtr_enabled.set(config_data.get("ik_llama_rtr_enabled", False))
        self.fmoe_enabled.set(config_data.get("ik_llama_fmoe_enabled", False))
        self.ser_value.set(config_data.get("ik_llama_ser_value", ""))
        self.amb_value.set(config_data.get("ik_llama_amb_value", ""))
        self.ctk_value.set(config_data.get("ik_llama_ctk_value", "f16"))
        self.ctv_value.set(config_data.get("ik_llama_ctv_value", "f16")) 