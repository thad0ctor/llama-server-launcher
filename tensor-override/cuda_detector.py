#!/usr/bin/env python3
"""
CUDA Device Detection and VRAM Analysis
Detects all CUDA devices and their available VRAM for tensor allocation optimization.
"""

import subprocess
import json
import re
import os
import sys
from typing import List, Dict, Optional

class CUDADevice:
    def __init__(self, device_id: int, name: str, total_memory: int, free_memory: int, compute_capability: str):
        self.device_id = device_id
        self.name = name
        self.total_memory = total_memory  # in MB
        self.free_memory = free_memory    # in MB
        self.compute_capability = compute_capability
        self.allocated_memory = 0  # Track allocated memory for planning
    
    def __repr__(self):
        return f"CUDA{self.device_id}: {self.name} ({self.free_memory}/{self.total_memory} MB free)"

class CUDADetector:
    def __init__(self, config_path: str = "config.json"):
        self.devices: List[CUDADevice] = []
        self.total_vram = 0
        self.available_vram = 0
        self.config = self.load_config(config_path)
        self.setup_environment()
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_file = os.path.join(script_dir, config_path)
            
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config file {config_path}: {e}")
            return {
                "llama_cpp_path": "/home/rgilbreth/Desktop/AI-Software/llama.cpp",
                "virtual_env_path": None,
                "context_size": 8192,
                "tensor_analysis": {
                    "prioritize_experts_on_cpu": True,
                    "maximize_gpu_layers": True,
                    "reserve_context_memory": True,
                    "memory_safety_margin_mb": 512
                }
            }
    
    def setup_environment(self):
        """Setup virtual environment if configured"""
        venv_path = self.config.get("virtual_env_path")
        if venv_path and os.path.exists(venv_path):
            print(f"Using virtual environment: {venv_path}")
            
            # Add venv to Python path
            import glob
            # Find Python version in venv
            python_versions = glob.glob(os.path.join(venv_path, "lib", "python3.*"))
            if python_versions:
                for python_ver in python_versions:
                    site_packages = os.path.join(python_ver, "site-packages")
                    if os.path.exists(site_packages):
                        sys.path.insert(0, site_packages)
                        print(f"Added to Python path: {site_packages}")
            
            # Also add venv/bin to PATH for subprocess calls
            venv_bin = os.path.join(venv_path, "bin")
            if os.path.exists(venv_bin):
                current_path = os.environ.get("PATH", "")
                os.environ["PATH"] = f"{venv_bin}:{current_path}"
                print(f"Added to PATH: {venv_bin}")
                
            # Set VIRTUAL_ENV environment variable
            os.environ["VIRTUAL_ENV"] = venv_path
            
            # Update Python executable path if needed
            venv_python = os.path.join(venv_path, "bin", "python")
            if os.path.exists(venv_python):
                os.environ["PYTHON"] = venv_python
        elif venv_path:
            print(f"Warning: Virtual environment path does not exist: {venv_path}")
    
    def detect_devices(self) -> List[CUDADevice]:
        """Detect all CUDA devices and their VRAM information"""
        try:
            # Use nvidia-smi to get device information
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,compute_cap',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True)
            
            devices = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 5:
                        device_id = int(parts[0])
                        name = parts[1]
                        total_memory = int(parts[2])
                        free_memory = int(parts[3])
                        compute_capability = parts[4]
                        
                        device = CUDADevice(device_id, name, total_memory, free_memory, compute_capability)
                        devices.append(device)
            
            self.devices = devices
            self.total_vram = sum(d.total_memory for d in devices)
            self.available_vram = sum(d.free_memory for d in devices)
            
            return devices
            
        except subprocess.CalledProcessError as e:
            print(f"Error running nvidia-smi: {e}")
            return []
        except Exception as e:
            print(f"Error detecting CUDA devices: {e}")
            return []
    
    def get_device_info(self) -> Dict:
        """Get structured device information"""
        return {
            'devices': [
                {
                    'device_id': d.device_id,
                    'name': d.name,
                    'total_memory_mb': d.total_memory,
                    'free_memory_mb': d.free_memory,
                    'compute_capability': d.compute_capability
                } for d in self.devices
            ],
            'total_vram_mb': self.total_vram,
            'available_vram_mb': self.available_vram,
            'device_count': len(self.devices)
        }
    
    def reserve_memory_for_context(self, context_size: int = None) -> int:
        """Reserve memory for context processing with K/V cache quantization"""
        if context_size is None:
            context_size = self.config.get("context_size", 8192)
            
        # Get K/V cache configuration
        kv_cache_config = self.config.get("kv_cache", {})
        kv_cache_type = kv_cache_config.get("type", "f16")
        kv_cache_type_k = kv_cache_config.get("type_k", kv_cache_type)
        kv_cache_type_v = kv_cache_config.get("type_v", kv_cache_type)
        
        # Calculate K/V cache memory based on quantization type
        context_memory_mb = self._calculate_kv_cache_memory(
            context_size, kv_cache_type_k, kv_cache_type_v
        )
        
        # Add processing overhead (attention computation, etc.)
        processing_overhead = int(context_memory_mb * 0.1)  # Reduced from 20% to 10%
        context_memory_mb += processing_overhead
        
        # Add safety margin (reduced from 512MB to 256MB)
        safety_margin_config = self.config.get("tensor_analysis", {}).get("memory_safety_margin_mb")
        safety_margin = int(safety_margin_config) if safety_margin_config is not None else 256
        context_memory_mb += safety_margin
        
        # Don't pre-reserve context memory on one device - this is wrong!
        # Instead, we'll calculate per-GPU headroom during tensor optimization
        print(f"K/V cache memory (type: {kv_cache_type_k}/{kv_cache_type_v}): {context_memory_mb - safety_margin - processing_overhead:.0f} MB")
        print(f"Total context memory to distribute: {context_memory_mb} MB")
        
        return context_memory_mb
    
    def _calculate_kv_cache_memory(self, context_size: int, type_k: str, type_v: str) -> int:
        """Calculate K/V cache memory based on quantization types"""
        
        # Model architecture estimates (adjust based on actual model)
        # For Kimi/DeepSeek2 architecture
        model_params = {
            "n_embd": 7168,      # embedding dimension
            "n_layer": 61,       # number of layers
            "n_head": 64,        # number of heads
            "n_head_kv": 1,      # number of KV heads (for MQA/GQA)
            "head_dim": 112      # head dimension (n_embd / n_head)
        }
        
        # Bytes per element based on quantization type
        type_sizes = {
            "f32": 4,     # 32-bit float
            "f16": 2,     # 16-bit float
            "bf16": 2,    # bfloat16
            "q8_0": 1,    # 8-bit quantized
            "q4_0": 0.5,  # 4-bit quantized
            "q4_1": 0.5,  # 4-bit quantized
            "q5_0": 0.625, # 5-bit quantized
            "q5_1": 0.625, # 5-bit quantized
            "iq4_nl": 0.5, # 4-bit quantized
        }
        
        k_type_size = type_sizes.get(type_k, 2)  # Default to f16
        v_type_size = type_sizes.get(type_v, 2)  # Default to f16
        
        # Calculate memory for K and V caches
        # K cache: context_size * n_embd * n_layer * k_type_size
        # V cache: context_size * n_embd * n_layer * v_type_size
        # Note: For MQA/GQA, KV heads are fewer than Q heads
        
        k_cache_size = context_size * model_params["n_embd"] * model_params["n_layer"] * k_type_size
        v_cache_size = context_size * model_params["n_embd"] * model_params["n_layer"] * v_type_size
        
        # Convert to MB
        total_cache_mb = (k_cache_size + v_cache_size) / (1024 * 1024)
        
        return int(total_cache_mb)
    
    def print_device_summary(self):
        """Print a summary of detected devices"""
        print("CUDA Device Detection Summary:")
        print("=" * 50)
        for device in self.devices:
            print(f"Device {device.device_id}: {device.name}")
            print(f"  Compute Capability: {device.compute_capability}")
            print(f"  Total VRAM: {device.total_memory:,} MB")
            print(f"  Free VRAM: {device.free_memory:,} MB")
            print(f"  Used VRAM: {device.total_memory - device.free_memory:,} MB")
            print()
        
        print(f"Total VRAM across all devices: {self.total_vram:,} MB")
        print(f"Available VRAM: {self.available_vram:,} MB")

def main():
    detector = CUDADetector()
    devices = detector.detect_devices()
    
    if devices:
        detector.print_device_summary()
        
        # Export to JSON for use by other scripts
        info = detector.get_device_info()
        output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cuda_devices.json')
        with open(output_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\nDevice information saved to {output_file}")
    else:
        print("No CUDA devices detected or nvidia-smi not available")

if __name__ == "__main__":
    main()