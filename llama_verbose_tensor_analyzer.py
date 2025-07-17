#!/usr/bin/env python3
"""
LLaMa.cpp Verbose Tensor Analyzer
Uses llama.cpp's --verbose mode to extract actual tensor names and sizes for optimal allocation.
"""

import subprocess
import re
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tempfile
import time


class TensorInfo:
    """Represents information about a tensor extracted from llama.cpp verbose output."""
    
    def __init__(self, name: str, size_mb: float, quantization: str):
        self.name = name
        self.size_mb = size_mb
        self.quantization = quantization
        self.size_bytes = int(size_mb * 1024 * 1024)
    
    def __str__(self):
        return f"{self.name} ({self.size_mb:.1f} MiB {self.quantization})"
    
    def __repr__(self):
        return f"TensorInfo('{self.name}', {self.size_mb}, '{self.quantization}')"


class LlamaVerboseTensorAnalyzer:
    """Analyzes model tensors using llama.cpp --verbose output."""
    
    def __init__(self, llama_cpp_dir: str, verbose: bool = False):
        """
        Initialize the analyzer.
        
        Args:
            llama_cpp_dir: Path to llama.cpp directory containing llama-server binary
            verbose: Enable verbose debug output
        """
        self.llama_cpp_dir = Path(llama_cpp_dir)
        self.verbose = verbose
        self.llama_server_path = self._find_llama_server()
        
        if not self.llama_server_path:
            # Show available files in the directory for debugging
            available_files = []
            try:
                for item in self.llama_cpp_dir.iterdir():
                    if item.is_file():
                        available_files.append(item.name)
            except:
                pass
            
            error_msg = f"llama-server not found in {llama_cpp_dir}"
            if available_files:
                error_msg += f"\nAvailable files in directory: {', '.join(available_files[:10])}"
                if len(available_files) > 10:
                    error_msg += f" ... and {len(available_files) - 10} more"
            
            raise FileNotFoundError(error_msg)
    
    def _find_llama_server(self) -> Optional[Path]:
        """Find the llama-server executable."""
        # Check common locations for the server executable
        possible_locations = [
            # Build directory locations
            self.llama_cpp_dir / "build" / "bin" / "llama-server",
            self.llama_cpp_dir / "build" / "llama-server",
            self.llama_cpp_dir / "bin" / "llama-server",
            # Direct in llama.cpp directory
            self.llama_cpp_dir / "llama-server",
            self.llama_cpp_dir / "server",
            self.llama_cpp_dir / "llama_server",
            # Legacy names
            self.llama_cpp_dir / "build" / "bin" / "server",
            self.llama_cpp_dir / "build" / "server",
        ]
        
        for server_path in possible_locations:
            if server_path.exists() and os.access(server_path, os.X_OK):
                self._debug_print(f"Found llama-server at: {server_path}")
                return server_path
        
        self._debug_print(f"llama-server not found in any of these locations:")
        for path in possible_locations:
            self._debug_print(f"  - {path} (exists: {path.exists()})")
        
        return None
    
    def _debug_print(self, message: str):
        """Print debug message if verbose mode is enabled."""
        if self.verbose:
            print(f"DEBUG: {message}", file=sys.stderr)
    
    def analyze_model_tensors(self, model_path: str, timeout: int = 300, gpu_count: int = 4) -> Tuple[Dict[str, TensorInfo], Dict[str, any], Dict[str, any]]:
        """
        Analyze model tensors using llama.cpp --verbose mode.
        
        Args:
            model_path: Path to the GGUF model file
            timeout: Timeout in seconds for the analysis process
            gpu_count: Number of GPUs to use for tensor-split analysis
            
        Returns:
            Tuple of (tensors_dict, kv_cache_info_dict, compute_buffer_info_dict)
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If analysis fails
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self._debug_print(f"Starting tensor analysis for {model_path.name}")
        
        # Create temporary directory for analysis
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Run llama-server with minimal settings to get tensor info
            # Use a random high port to avoid conflicts, server will exit after loading
            cmd = [
                str(self.llama_server_path),
                "--model", str(model_path),
                "--ctx-size", "512",  # Minimal context to speed up loading
                "--port", "0",  # Use port 0 for automatic assignment
                "--verbose"
            ]
            
            # Add split-mode and tensor-split parameters for proper KV cache analysis
            # This ensures we get accurate KV cache allocation information
            cmd.extend(["--split-mode", "row"])
            tensor_split_ratio = ",".join(["1"] * gpu_count)
            cmd.extend(["--tensor-split", tensor_split_ratio])
            self._debug_print(f"Using tensor-split: {tensor_split_ratio} for {gpu_count} GPUs")
            
            # Add -ot parameters to force all tensors to CPU initially
            # This ensures we get all tensor information without GPU allocation
            cmd.extend(["-ot", ".*=CPU"])
            
            self._debug_print(f"Running command: {' '.join(cmd)}")
            
            try:
                # Run the command and capture output
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=temp_path
                )
                
                output_lines = []
                start_time = time.time()
                tensors_found = {}
                loading_complete = False
                
                # Read output line by line with timeout
                kv_cache_found = False
                while True:
                    if time.time() - start_time > timeout:
                        process.terminate()
                        raise RuntimeError(f"Analysis timed out after {timeout} seconds")
                    
                    line = process.stdout.readline()
                    if line == '' and process.poll() is not None:
                        break
                    
                    if line:
                        line_stripped = line.rstrip()
                        output_lines.append(line_stripped)
                        self._debug_print(f"llama.cpp: {line_stripped}")
                        
                        # Check for tensor information and loading completion markers
                        if "tensor" in line_stripped and "MiB" in line_stripped:
                            # Parse tensor info immediately
                            tensor_match = re.search(r'tensor\s+([^\s]+)\s+\(([0-9\.]+)\s+MiB\s+([^)]+)\)', line_stripped)
                            if tensor_match:
                                tensor_name = tensor_match.group(1)
                                size_mb = float(tensor_match.group(2))
                                quantization = tensor_match.group(3)
                                tensors_found[tensor_name] = TensorInfo(tensor_name, size_mb, quantization)
                        
                        # Check for KV cache information
                        if "llama_kv_cache_unified:" in line_stripped:
                            if "size =" in line_stripped:
                                kv_cache_found = True
                                self._debug_print("Found KV cache size information")
                        
                        # Check for completion markers - but only terminate after we have KV cache info
                        if any(marker in line_stripped.lower() for marker in [
                            "model loaded",
                            "server listening", 
                            "http server listening",
                            "all slots are idle",
                            "waiting for requests"
                        ]):
                            loading_complete = True
                            # Only break if we have KV cache info or we've been waiting too long
                            if kv_cache_found or (time.time() - start_time > 120):  # Wait up to 2 minutes for KV cache
                                self._debug_print("Detected model loading completion and KV cache info - terminating process")
                                break
                            else:
                                self._debug_print("Model loading complete, waiting for KV cache information...")
                        
                        # Alternative termination condition - if we see compute buffer allocation, we can stop
                        if "compute buffer size" in line_stripped:
                            self._debug_print("Detected compute buffer allocation - analysis complete")
                            break
                
                # Terminate the process if it's still running
                if process.poll() is None:
                    self._debug_print("Terminating llama-server process")
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self._debug_print("Process didn't terminate gracefully, killing it")
                        process.kill()
                        process.wait()
                
                return_code = process.returncode
                
                # Use tensors found during real-time parsing, fallback to full output parsing if needed
                if tensors_found:
                    tensors = tensors_found
                else:
                    # Fallback to parsing full output if real-time parsing didn't work
                    tensors = self._parse_tensor_output(output_lines)
                
                # Parse KV cache information from the output
                kv_cache_info = self._parse_kv_cache_info(output_lines)
                
                # Parse compute buffer information
                compute_buffer_info = self._parse_compute_buffer_info(output_lines)
                
                if not tensors:
                    # If no tensors found, this might be an error
                    output_text = '\n'.join(output_lines[-20:])  # Show last 20 lines for debugging
                    raise RuntimeError(f"No tensors found in output. Return code: {return_code}\nLast output:\n{output_text}")
                
                self._debug_print(f"Successfully extracted {len(tensors)} tensors (return code: {return_code})")
                if kv_cache_info:
                    self._debug_print(f"KV cache info: {kv_cache_info.get('total_size_mb', 'unknown')} MiB")
                if compute_buffer_info:
                    self._debug_print(f"Compute buffer info: {compute_buffer_info.get('total_size_mb', 'unknown')} MiB")
                    
                return tensors, kv_cache_info, compute_buffer_info
                
            except subprocess.TimeoutExpired:
                process.kill()
                raise RuntimeError(f"Analysis process timed out after {timeout} seconds")
            except Exception as e:
                raise RuntimeError(f"Failed to analyze tensors: {str(e)}")
    
    def _parse_tensor_output(self, output_lines: List[str]) -> Dict[str, TensorInfo]:
        """
        Parse tensor information from llama.cpp verbose output.
        
        Args:
            output_lines: Lines of output from llama.cpp
            
        Returns:
            Dictionary mapping tensor names to TensorInfo objects
        """
        tensors = {}
        
        # Regex pattern to match tensor lines
        # Example: "tensor token_embd.weight (630 MiB q4_K) buffer type overridden to CUDA0"
        tensor_pattern = re.compile(
            r'tensor\s+([^\s]+)\s+\(([0-9\.]+)\s+MiB\s+([^)]+)\)\s+buffer\s+type\s+overridden\s+to\s+(\w+)'
        )
        
        for line in output_lines:
            match = tensor_pattern.search(line)
            if match:
                tensor_name = match.group(1)
                size_mb = float(match.group(2))
                quantization = match.group(3)
                device = match.group(4)
                
                # Store tensor info
                tensors[tensor_name] = TensorInfo(tensor_name, size_mb, quantization)
                
                self._debug_print(f"Found tensor: {tensor_name} ({size_mb} MiB {quantization})")
        
        return tensors
    
    def _parse_kv_cache_info(self, output_lines: List[str]) -> Dict[str, any]:
        """
        Parse KV cache information from llama.cpp verbose output.
        
        Args:
            output_lines: Lines of output from llama.cpp
            
        Returns:
            Dictionary containing KV cache information
        """
        kv_cache_info = {}
        
        # Pattern to match KV cache size line
        # Example: "llama_kv_cache_unified: size = 4666.50 MiB (131072 cells, 61 layers, 1 seqs), K (q4_0): 2470.50 MiB, V (q4_0): 2196.00 MiB"
        kv_size_pattern = re.compile(
            r'llama_kv_cache_unified:\s+size\s+=\s+([0-9\.]+)\s+MiB\s+\(([0-9]+)\s+cells,\s+([0-9]+)\s+layers,\s+([0-9]+)\s+seqs\),\s+K\s+\(([^)]+)\):\s+([0-9\.]+)\s+MiB,\s+V\s+\(([^)]+)\):\s+([0-9\.]+)\s+MiB'
        )
        
        # Pattern to match individual KV cache buffer size
        # Example: "llama_kv_cache_unified:        CPU KV buffer size =  4666.50 MiB"
        kv_buffer_pattern = re.compile(
            r'llama_kv_cache_unified:\s+(\w+)\s+KV\s+buffer\s+size\s+=\s+([0-9\.]+)\s+MiB'
        )
        
        for line in output_lines:
            # Check for detailed KV cache size information
            match = kv_size_pattern.search(line)
            if match:
                kv_cache_info['total_size_mb'] = float(match.group(1))
                kv_cache_info['cells'] = int(match.group(2))
                kv_cache_info['layers'] = int(match.group(3))
                kv_cache_info['seqs'] = int(match.group(4))
                kv_cache_info['k_type'] = match.group(5)
                kv_cache_info['k_size_mb'] = float(match.group(6))
                kv_cache_info['v_type'] = match.group(7)
                kv_cache_info['v_size_mb'] = float(match.group(8))
                
                self._debug_print(f"Found KV cache info: {kv_cache_info['total_size_mb']} MiB total, {kv_cache_info['layers']} layers")
            
            # Check for KV buffer location information
            buffer_match = kv_buffer_pattern.search(line)
            if buffer_match:
                device = buffer_match.group(1)
                size_mb = float(buffer_match.group(2))
                
                if 'buffer_locations' not in kv_cache_info:
                    kv_cache_info['buffer_locations'] = {}
                kv_cache_info['buffer_locations'][device] = size_mb
                
                self._debug_print(f"Found KV buffer: {device} = {size_mb} MiB")
        
        return kv_cache_info

    def _parse_compute_buffer_info(self, output_lines: List[str]) -> Dict[str, any]:
        """Parses compute buffer allocation from llama.cpp verbose output."""
        compute_info = {'total_size_mb': 0, 'buffer_count': 0}
        
        # Pattern for compute buffer size
        # Example: "ggml_backend_cuda_buffer_type_alloc_buffer: compute buffer size =   384.00 MiB"
        compute_pattern = re.compile(
            r'compute buffer size\s+=\s+([0-9\.]+)\s+MiB'
        )
        
        for line in output_lines:
            match = compute_pattern.search(line)
            if match:
                size_mb = float(match.group(1))
                compute_info['total_size_mb'] += size_mb
                compute_info['buffer_count'] += 1
                self._debug_print(f"Found compute buffer: {size_mb:.2f} MiB")
        
        return compute_info
    
    def categorize_tensors(self, tensors: Dict[str, TensorInfo]) -> Dict[str, List[TensorInfo]]:
        """
        Categorize tensors by type for optimal allocation.
        
        Args:
            tensors: Dictionary of tensor name to TensorInfo
            
        Returns:
            Dictionary with tensor categories as keys and lists of TensorInfo as values
        """
        categories = {
            'embedding': [],
            'output': [],
            'expert_ffn': [],
            'attention': [],
            'norm': [],
            'other': []
        }
        
        for tensor_info in tensors.values():
            name = tensor_info.name
            
            if 'embd' in name or 'embed' in name:
                categories['embedding'].append(tensor_info)
            elif name == 'output.weight' or name.endswith('output.weight'):
                categories['output'].append(tensor_info)
            elif 'ffn_' in name and ('_exps.' in name or '_exp.' in name):
                categories['expert_ffn'].append(tensor_info)
            elif 'attn_' in name and '.weight' in name and 'norm' not in name:
                categories['attention'].append(tensor_info)
            elif 'norm.weight' in name or '.bias' in name:
                categories['norm'].append(tensor_info)
            else:
                categories['other'].append(tensor_info)
        
        # Sort each category by size (largest first) for better allocation
        for category in categories.values():
            category.sort(key=lambda t: t.size_mb, reverse=True)
        
        return categories
    
    def generate_optimal_allocation(self, tensors: Dict[str, TensorInfo], 
                                  gpu_count: int, 
                                  vram_per_gpu_gb: float,
                                  prioritize_performance: bool = True) -> List[str]:
        """
        Generate optimal tensor allocation parameters.
        
        Args:
            tensors: Dictionary of tensor name to TensorInfo
            gpu_count: Number of available GPUs
            vram_per_gpu_gb: VRAM available per GPU in GB
            prioritize_performance: If True, prioritize performance over VRAM efficiency
            
        Returns:
            List of tensor override parameter strings (-ot format)
        """
        categories = self.categorize_tensors(tensors)
        allocation_params = []
        
        # Track VRAM usage per GPU
        gpu_usage_bytes = [0] * gpu_count
        vram_per_gpu_bytes = vram_per_gpu_gb * 1024 * 1024 * 1024
        
        # Reserve some VRAM for context and other overhead
        safety_margin = 0.9  # Use 90% of available VRAM
        vram_limit_per_gpu = vram_per_gpu_bytes * safety_margin
        
        def find_best_gpu(tensor_size_bytes: int) -> int:
            """Find GPU with best balance of available space and current usage that can fit this tensor."""
            best_gpu = -1
            best_score = -1
            
            for i in range(gpu_count):
                available = vram_limit_per_gpu - gpu_usage_bytes[i]
                if available >= tensor_size_bytes:
                    # Calculate a score that balances available space and current usage
                    # Lower usage percentage gets higher score (better balance)
                    usage_ratio = gpu_usage_bytes[i] / vram_limit_per_gpu if vram_limit_per_gpu > 0 else 1.0
                    available_ratio = available / vram_limit_per_gpu if vram_limit_per_gpu > 0 else 0.0
                    
                    # Score favors GPUs with lower usage and sufficient available space
                    # Weight available space more heavily for very large tensors
                    tensor_size_ratio = tensor_size_bytes / vram_limit_per_gpu if vram_limit_per_gpu > 0 else 1.0
                    if tensor_size_ratio > 0.1:  # Large tensor (>10% of GPU memory)
                        score = available_ratio * 0.7 + (1.0 - usage_ratio) * 0.3
                    else:  # Small tensor
                        score = available_ratio * 0.4 + (1.0 - usage_ratio) * 0.6
                    
                    if score > best_score:
                        best_score = score
                        best_gpu = i
            
            return best_gpu
        
        def allocate_tensor(tensor_info: TensorInfo, preferred_device: str = None) -> str:
            """Allocate a single tensor to the best available device."""
            if preferred_device == "CPU":
                return f"-ot {tensor_info.name}=CPU"
            
            gpu_id = find_best_gpu(tensor_info.size_bytes)
            if gpu_id >= 0:
                gpu_usage_bytes[gpu_id] += tensor_info.size_bytes
                return f"-ot {tensor_info.name}=CUDA{gpu_id}"
            else:
                # No GPU has space, put on CPU
                return f"-ot {tensor_info.name}=CPU"
        
        # PRIORITY 1: Critical large tensors (embedding, output) - always try GPU
        # Sort by size (largest first) for better load balancing
        critical_tensors = sorted(categories['embedding'] + categories['output'], 
                                key=lambda t: t.size_mb, reverse=True)
        for tensor_info in critical_tensors:
            param = allocate_tensor(tensor_info)
            allocation_params.append(param)
            self._debug_print(f"Critical tensor: {param}")
        
        # PRIORITY 2: Attention tensors (performance critical)
        # Sort by size (largest first) for better load balancing
        attention_tensors = sorted(categories['attention'], key=lambda t: t.size_mb, reverse=True)
        for tensor_info in attention_tensors:
            param = allocate_tensor(tensor_info)
            allocation_params.append(param)
            self._debug_print(f"Attention tensor: {param}")
        
        # PRIORITY 3: Expert FFN tensors (fill remaining VRAM)
        # Sort by size (largest first) for better load balancing
        expert_tensors = sorted(categories['expert_ffn'], key=lambda t: t.size_mb, reverse=True)
        for tensor_info in expert_tensors:
            param = allocate_tensor(tensor_info)
            allocation_params.append(param)
            self._debug_print(f"Expert tensor: {param}")
        
        # PRIORITY 4: Small tensors (CPU to save VRAM)
        for tensor_info in categories['norm']:
            param = allocate_tensor(tensor_info, preferred_device="CPU")
            allocation_params.append(param)
            self._debug_print(f"Norm tensor: {param}")
        
        # PRIORITY 5: Other tensors
        # Sort by size (largest first) for better load balancing
        other_tensors = sorted(categories['other'], key=lambda t: t.size_mb, reverse=True)
        for tensor_info in other_tensors:
            param = allocate_tensor(tensor_info)
            allocation_params.append(param)
            self._debug_print(f"Other tensor: {param}")
        
        # Print allocation summary
        if self.verbose:
            print("Allocation Summary:", file=sys.stderr)
            for i in range(gpu_count):
                usage_gb = gpu_usage_bytes[i] / (1024**3)
                usage_pct = (gpu_usage_bytes[i] / vram_limit_per_gpu) * 100
                print(f"  CUDA{i}: {usage_gb:.1f}GB ({usage_pct:.1f}%)", file=sys.stderr)
        
        return allocation_params
    
    def generate_optimal_allocation_with_configs(self, tensors: Dict[str, TensorInfo], 
                                               gpu_configs: List[Dict],
                                               safety_margin: float = 0.90) -> List[str]:
        """
        Generate optimal tensor allocation with detailed per-GPU configurations.
        
        Args:
            tensors: Dictionary of tensor name to TensorInfo
            gpu_configs: List of GPU configuration dictionaries
            safety_margin: Percentage of available VRAM to use (default 0.90 = 90%)
            
        Returns:
            List of tensor override parameter strings (-ot format)
        """
        categories = self.categorize_tensors(tensors)
        allocation_params = []
        
        # Track VRAM usage per GPU using actual available memory
        gpu_usage_bytes = [0] * len(gpu_configs)
        gpu_limits_bytes = []
        
        # Calculate limits for each GPU based on available memory
        for i, gpu_config in enumerate(gpu_configs):
            available_gb = gpu_config['available_gb']
            # Use configurable safety margin (default 90%)
            # We've already accounted for buffers, KV cache, and compute buffers
            limit_bytes = int(available_gb * 1024 * 1024 * 1024 * safety_margin)
            gpu_limits_bytes.append(limit_bytes)
            
            total_gb = gpu_config.get('total_memory_gb', 0)
            base_gb = gpu_config.get('base_memory_gb', total_gb)
            self._debug_print(f"GPU {i}: Available {available_gb:.1f}GB, Limit {limit_bytes / (1024**3):.1f}GB (safety margin: {safety_margin:.0%})")
            self._debug_print(f"GPU {i}: Base memory {base_gb:.1f}GB (from {total_gb:.1f}GB total)")
            self._debug_print(f"GPU {i}: Buffer {gpu_config['buffer_mb']}MB, KV Cache {gpu_config['kv_cache_mb']:.1f}MB")
            self._debug_print(f"GPU {i}: Compute Buffer {gpu_config.get('compute_buffer_mb', 0):.0f}MB")
        
        def find_best_gpu_with_config(tensor_size_bytes: int) -> int:
            """Find GPU with best balance of available space and current usage that can fit this tensor."""
            best_gpu = -1
            best_score = -1
            
            # Debug logging for allocation decisions
            tensor_size_mb = tensor_size_bytes / (1024 * 1024)
            self._debug_print(f"Finding GPU for tensor: {tensor_size_mb:.1f} MB")
            
            for i in range(len(gpu_configs)):
                available = gpu_limits_bytes[i] - gpu_usage_bytes[i]
                if available >= tensor_size_bytes:
                    # Calculate a score that balances available space and current usage
                    # Lower usage percentage gets higher score (better balance)
                    usage_ratio = gpu_usage_bytes[i] / gpu_limits_bytes[i] if gpu_limits_bytes[i] > 0 else 1.0
                    available_ratio = available / gpu_limits_bytes[i] if gpu_limits_bytes[i] > 0 else 0.0
                    
                    # Score favors GPUs with lower usage and sufficient available space
                    # Weight available space more heavily for very large tensors
                    tensor_size_ratio = tensor_size_bytes / gpu_limits_bytes[i] if gpu_limits_bytes[i] > 0 else 1.0
                    if tensor_size_ratio > 0.1:  # Large tensor (>10% of GPU memory)
                        score = available_ratio * 0.7 + (1.0 - usage_ratio) * 0.3
                    else:  # Small tensor
                        score = available_ratio * 0.4 + (1.0 - usage_ratio) * 0.6
                    
                    # Debug logging
                    usage_gb = gpu_usage_bytes[i] / (1024**3)
                    available_gb = available / (1024**3)
                    limit_gb = gpu_limits_bytes[i] / (1024**3)
                    self._debug_print(f"  GPU {i}: usage={usage_gb:.1f}GB ({usage_ratio:.3f}), available={available_gb:.1f}GB ({available_ratio:.3f}), score={score:.3f}")
                    
                    if score > best_score:
                        best_score = score
                        best_gpu = i
                else:
                    # Debug logging for GPUs that can't fit the tensor
                    usage_gb = gpu_usage_bytes[i] / (1024**3)
                    available_gb = available / (1024**3)
                    self._debug_print(f"  GPU {i}: usage={usage_gb:.1f}GB, available={available_gb:.1f}GB - CANNOT FIT")
            
            if best_gpu >= 0:
                self._debug_print(f"  Selected GPU {best_gpu} with score {best_score:.3f}")
            else:
                self._debug_print(f"  No GPU can fit tensor, will use CPU")
            
            return best_gpu
        
        def allocate_tensor_with_config(tensor_info: TensorInfo, preferred_device: str = None) -> str:
            """Allocate a single tensor to the best available device."""
            if preferred_device == "CPU":
                return f"-ot {tensor_info.name}=CPU"
            
            gpu_id = find_best_gpu_with_config(tensor_info.size_bytes)
            if gpu_id >= 0:
                gpu_usage_bytes[gpu_id] += tensor_info.size_bytes
                return f"-ot {tensor_info.name}=CUDA{gpu_id}"
            else:
                # No GPU has space, put on CPU
                return f"-ot {tensor_info.name}=CPU"
        
        # PRIORITY 1: Critical large tensors (embedding, output) - always try GPU
        # Sort by size (largest first) for better load balancing
        critical_tensors = sorted(categories['embedding'] + categories['output'], 
                                key=lambda t: t.size_mb, reverse=True)
        for tensor_info in critical_tensors:
            param = allocate_tensor_with_config(tensor_info)
            allocation_params.append(param)
            self._debug_print(f"Critical tensor: {param}")
        
        # PRIORITY 2: Attention tensors (performance critical)
        # Sort by size (largest first) for better load balancing
        attention_tensors = sorted(categories['attention'], key=lambda t: t.size_mb, reverse=True)
        for tensor_info in attention_tensors:
            param = allocate_tensor_with_config(tensor_info)
            allocation_params.append(param)
            self._debug_print(f"Attention tensor: {param}")
        
        # PRIORITY 3: Expert FFN tensors (fill remaining VRAM)
        # Sort by size (largest first) for better load balancing
        expert_tensors = sorted(categories['expert_ffn'], key=lambda t: t.size_mb, reverse=True)
        for tensor_info in expert_tensors:
            param = allocate_tensor_with_config(tensor_info)
            allocation_params.append(param)
            self._debug_print(f"Expert tensor: {param}")
        
        # PRIORITY 4: Small tensors (CPU to save VRAM)
        for tensor_info in categories['norm']:
            param = allocate_tensor_with_config(tensor_info, preferred_device="CPU")
            allocation_params.append(param)
            self._debug_print(f"Norm tensor: {param}")
        
        # PRIORITY 5: Other tensors
        # Sort by size (largest first) for better load balancing
        other_tensors = sorted(categories['other'], key=lambda t: t.size_mb, reverse=True)
        for tensor_info in other_tensors:
            param = allocate_tensor_with_config(tensor_info)
            allocation_params.append(param)
            self._debug_print(f"Other tensor: {param}")
        
        # Print detailed allocation summary
        if self.verbose:
            print("Enhanced Allocation Summary:", file=sys.stderr)
            for i, gpu_config in enumerate(gpu_configs):
                usage_gb = gpu_usage_bytes[i] / (1024**3)
                limit_gb = gpu_limits_bytes[i] / (1024**3)
                usage_pct = (gpu_usage_bytes[i] / gpu_limits_bytes[i]) * 100 if gpu_limits_bytes[i] > 0 else 0
                total_gb = gpu_config['total_memory_gb']
                buffer_mb = gpu_config['buffer_mb']
                kv_cache_mb = gpu_config['kv_cache_mb']
                compute_buffer_mb = gpu_config.get('compute_buffer_mb', 0)
                
                print(f"  CUDA{i}: {usage_gb:.1f}GB used / {limit_gb:.1f}GB limit ({usage_pct:.1f}%)", file=sys.stderr)
                print(f"    Total: {total_gb:.1f}GB, Buffer: {buffer_mb}MB, KV Cache: {kv_cache_mb:.1f}MB, Compute: {compute_buffer_mb:.0f}MB", file=sys.stderr)
        
        return allocation_params
    
    def analyze_and_generate_params(self, model_path: str, 
                                  gpu_count: int,
                                  vram_per_gpu_gb: float,
                                  timeout: int = 300) -> Tuple[List[str], Dict[str, TensorInfo], Dict[str, any], Dict[str, any]]:
        """
        Complete analysis and parameter generation pipeline.
        
        Args:
            model_path: Path to the GGUF model file
            gpu_count: Number of available GPUs
            vram_per_gpu_gb: VRAM available per GPU in GB
            timeout: Timeout in seconds for the analysis process
            
        Returns:
            Tuple of (allocation_parameters, tensor_info_dict, kv_cache_info, compute_buffer_info)
        """
        # Extract tensor information, KV cache info, and compute buffer info
        tensors, kv_cache_info, compute_buffer_info = self.analyze_model_tensors(model_path, timeout, gpu_count)
        
        if not tensors:
            raise RuntimeError("No tensors found in model analysis")
        
        # Generate optimal allocation
        allocation_params = self.generate_optimal_allocation(
            tensors, gpu_count, vram_per_gpu_gb
        )
        
        return allocation_params, tensors, kv_cache_info, compute_buffer_info
    
    def analyze_and_generate_params_with_gpu_configs(self, model_path: str, 
                                                    gpu_configs: List[Dict],
                                                    timeout: int = 300,
                                                    safety_margin: float = 0.90) -> Tuple[List[str], Dict[str, TensorInfo], Dict[str, any], Dict[str, any]]:
        """
        Enhanced analysis and parameter generation with detailed GPU configurations.
        
        Args:
            model_path: Path to the GGUF model file
            gpu_configs: List of GPU configuration dictionaries with keys:
                        - gpu_id: GPU index
                        - total_memory_gb: Total GPU memory in GB
                        - buffer_mb: Reserved buffer in MB
                        - kv_cache_mb: KV cache allocation in MB
                        - available_gb: Available VRAM for tensors in GB
            timeout: Timeout in seconds for the analysis process
            safety_margin: Percentage of available VRAM to use (default 0.90 = 90%)
            
        Returns:
            Tuple of (allocation_parameters, tensor_info_dict, kv_cache_info, compute_buffer_info)
        """
        gpu_count = len(gpu_configs)
        
        # Extract tensor information, KV cache info, and compute buffer info
        tensors, kv_cache_info, compute_buffer_info = self.analyze_model_tensors(model_path, timeout, gpu_count)
        
        if not tensors:
            raise RuntimeError("No tensors found in model analysis")
        
        # Generate optimal allocation with detailed GPU configs
        allocation_params = self.generate_optimal_allocation_with_configs(
            tensors, gpu_configs, safety_margin
        )
        
        return allocation_params, tensors, kv_cache_info, compute_buffer_info


def main():
    """Command line interface for testing the analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze model tensors using llama.cpp verbose output")
    parser.add_argument("model_path", help="Path to GGUF model file")
    parser.add_argument("--llama-cpp-dir", required=True, help="Path to llama.cpp directory")
    parser.add_argument("--gpu-count", type=int, default=4, help="Number of GPUs")
    parser.add_argument("--vram-per-gpu", type=float, default=24.0, help="VRAM per GPU in GB")
    parser.add_argument("--timeout", type=int, default=300, help="Analysis timeout in seconds")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    try:
        analyzer = LlamaVerboseTensorAnalyzer(args.llama_cpp_dir, verbose=args.verbose)
        
        print(f"Analyzing {args.model_path}...")
        allocation_params, tensors, kv_cache_info, compute_buffer_info = analyzer.analyze_and_generate_params(
            args.model_path, args.gpu_count, args.vram_per_gpu, args.timeout
        )
        
        print(f"\nFound {len(tensors)} tensors")
        if kv_cache_info:
            print(f"KV Cache: {kv_cache_info.get('total_size_mb', 'unknown')} MiB")
        if compute_buffer_info:
            print(f"Compute Buffer: {compute_buffer_info.get('total_size_mb', 'unknown')} MiB")
        print(f"Generated {len(allocation_params)} allocation parameters")
        
        print("\nTensor Override Parameters:")
        print("=" * 50)
        for param in allocation_params:
            print(param)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()