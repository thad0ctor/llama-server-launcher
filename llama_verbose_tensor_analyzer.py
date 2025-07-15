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
    
    def analyze_model_tensors(self, model_path: str, timeout: int = 300) -> Dict[str, TensorInfo]:
        """
        Analyze model tensors using llama.cpp --verbose mode.
        
        Args:
            model_path: Path to the GGUF model file
            timeout: Timeout in seconds for the analysis process
            
        Returns:
            Dictionary mapping tensor names to TensorInfo objects
            
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
                        
                        # Check for completion markers
                        if any(marker in line_stripped.lower() for marker in [
                            "model loaded",
                            "server listening",
                            "http server listening",
                            "all slots are idle",
                            "waiting for requests"
                        ]):
                            loading_complete = True
                            self._debug_print("Detected model loading completion - terminating process")
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
                
                if not tensors:
                    # If no tensors found, this might be an error
                    output_text = '\n'.join(output_lines[-20:])  # Show last 20 lines for debugging
                    raise RuntimeError(f"No tensors found in output. Return code: {return_code}\nLast output:\n{output_text}")
                
                self._debug_print(f"Successfully extracted {len(tensors)} tensors (return code: {return_code})")
                return tensors
                
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
            """Find GPU with most available space that can fit this tensor."""
            best_gpu = -1
            max_available = 0
            
            for i in range(gpu_count):
                available = vram_limit_per_gpu - gpu_usage_bytes[i]
                if available >= tensor_size_bytes and available > max_available:
                    max_available = available
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
        for tensor_info in categories['embedding'] + categories['output']:
            param = allocate_tensor(tensor_info)
            allocation_params.append(param)
            self._debug_print(f"Critical tensor: {param}")
        
        # PRIORITY 2: Attention tensors (performance critical)
        for tensor_info in categories['attention']:
            param = allocate_tensor(tensor_info)
            allocation_params.append(param)
            self._debug_print(f"Attention tensor: {param}")
        
        # PRIORITY 3: Expert FFN tensors (fill remaining VRAM)
        for tensor_info in categories['expert_ffn']:
            param = allocate_tensor(tensor_info)
            allocation_params.append(param)
            self._debug_print(f"Expert tensor: {param}")
        
        # PRIORITY 4: Small tensors (CPU to save VRAM)
        for tensor_info in categories['norm']:
            param = allocate_tensor(tensor_info, preferred_device="CPU")
            allocation_params.append(param)
            self._debug_print(f"Norm tensor: {param}")
        
        # PRIORITY 5: Other tensors
        for tensor_info in categories['other']:
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
    
    def analyze_and_generate_params(self, model_path: str, 
                                  gpu_count: int,
                                  vram_per_gpu_gb: float,
                                  timeout: int = 300) -> Tuple[List[str], Dict[str, TensorInfo]]:
        """
        Complete analysis and parameter generation pipeline.
        
        Args:
            model_path: Path to the GGUF model file
            gpu_count: Number of available GPUs
            vram_per_gpu_gb: VRAM available per GPU in GB
            timeout: Timeout in seconds for the analysis process
            
        Returns:
            Tuple of (allocation_parameters, tensor_info_dict)
        """
        # Extract tensor information
        tensors = self.analyze_model_tensors(model_path, timeout)
        
        if not tensors:
            raise RuntimeError("No tensors found in model analysis")
        
        # Generate optimal allocation
        allocation_params = self.generate_optimal_allocation(
            tensors, gpu_count, vram_per_gpu_gb
        )
        
        return allocation_params, tensors


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
        allocation_params, tensors = analyzer.analyze_and_generate_params(
            args.model_path, args.gpu_count, args.vram_per_gpu, args.timeout
        )
        
        print(f"\nFound {len(tensors)} tensors")
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