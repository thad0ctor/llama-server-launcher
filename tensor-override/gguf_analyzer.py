#!/usr/bin/env python3
"""
GGUF Model Analyzer
Analyzes GGUF model files to extract tensor information for optimal GPU/CPU allocation.
"""

import subprocess
import json
import re
import os
import sys
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class TensorInfo:
    name: str
    size_mb: float
    tensor_type: str
    is_expert: bool = False
    is_layer: bool = False
    layer_id: Optional[int] = None
    is_embedding: bool = False
    
    def __post_init__(self):
        # Determine tensor characteristics from name
        self.is_expert = 'expert' in self.name or 'ffn_gate_exps' in self.name or 'ffn_down_exps' in self.name or 'ffn_up_exps' in self.name
        self.is_layer = 'blk.' in self.name
        self.is_embedding = 'token_embd' in self.name or 'output' in self.name
        
        # Extract layer ID if it's a layer tensor
        if self.is_layer:
            match = re.search(r'blk\.(\d+)\.', self.name)
            if match:
                self.layer_id = int(match.group(1))

class GGUFAnalyzer:
    def __init__(self, config_path: str = "config.json"):
        self.config = self.load_config(config_path)
        self.tensors: List[TensorInfo] = []
        self.total_size_mb = 0
        self.expert_tensors: List[TensorInfo] = []
        self.layer_tensors: List[TensorInfo] = []
        self.embedding_tensors: List[TensorInfo] = []
        self.other_tensors: List[TensorInfo] = []
        
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
                "context_size": 8192
            }
    
    def analyze_gguf_file(self, model_path: str) -> bool:
        """Analyze GGUF file using llama.cpp's llama-quantize or server with verbose output"""
        try:
            llama_cpp_path = self.config.get("llama_cpp_path", "/home/rgilbreth/Desktop/AI-Software/llama.cpp")
            server_path = os.path.join(llama_cpp_path, "build", "bin", "llama-server")
            
            if not os.path.exists(server_path):
                print(f"Error: llama-server not found at {server_path}")
                return False
            
            # Run llama-server with minimal context to get tensor information
            # We'll start the server and immediately stop it to capture tensor loading info
            # Use stdbuf to force unbuffered output for better real-time feedback
            cmd = [
                "stdbuf", "-oL", "-eL",  # Line buffered output
                server_path,
                "--model", model_path,
                "--ctx-size", "1",
                "--verbose"
            ]
            
            print(f"Running: {' '.join(cmd)}")
            print("This may take several minutes for large models - please be patient...")
            print("Starting analysis...")
            sys.stdout.flush()
            
            # Use Popen to capture output as it streams with unbuffered output
            import select
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                     text=True, bufsize=0, universal_newlines=True, env=env)
            
            stdout_lines = []
            stderr_lines = []
            
            # Read output until we get enough tensor information
            import time
            start_time = time.time()
            tensor_info_found = False
            type_summary_found = False
            last_progress_time = start_time
            lines_processed = 0
            
            # Use select for non-blocking I/O
            max_wait_time = 300  # 5 minutes timeout - need more time for tensor sizes
            tensor_types_collected = 0
            tensor_sizes_collected = 0
            expected_tensor_types = 12  # We expect around 12 different tensor types
            expert_tensors_found = 0
            last_data_time = start_time  # Track when we last received meaningful data
            no_progress_timeout = 10  # Timeout if no progress for 10 seconds
            
            while process.poll() is None:
                # Check if there's data available to read
                ready_streams, _, _ = select.select([process.stdout, process.stderr], [], [], 0.1)
                
                # Show progress even if no new data - indicates the process is still running
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Early termination conditions:
                # Since we now have comprehensive estimation from type summary,
                # we can terminate as soon as we have complete type information
                # Be more conservative with multi-part GGUF files
                min_types_needed = 8 if "00001-of-" in model_path else expected_tensor_types
                
                # Terminate if we have enough types AND either:
                # 1. 30 seconds have passed (not 60)
                # 2. No new data for 10 seconds
                if tensor_types_collected >= min_types_needed:
                    if elapsed > 30 or (current_time - last_data_time) > no_progress_timeout:
                        print(f"Collected complete tensor type summary: {tensor_types_collected} types")
                        print("Sufficient for comprehensive estimation - terminating early...")
                        process.terminate()
                        time.sleep(2)
                        if process.poll() is None:
                            process.kill()
                        break
                
                # Timeout if we're taking too long without getting useful tensor data
                if elapsed > max_wait_time:
                    print(f"Analysis timed out after {elapsed:.1f}s - killing process...")
                    process.terminate()
                    time.sleep(2)
                    if process.poll() is None:
                        process.kill()
                    break
                
                # Also timeout if no new data for extended period
                if (current_time - last_data_time) > 30:  # 30 seconds without data
                    print(f"No progress for 30 seconds (stuck at {lines_processed} lines) - terminating...")
                    process.terminate()
                    time.sleep(2)
                    if process.poll() is None:
                        process.kill()
                    break
                
                if current_time - last_progress_time > 3:  # Progress every 3 seconds
                    if lines_processed > 0:
                        print(f"Progress: {lines_processed:,} lines processed, {elapsed:.1f}s elapsed...")
                        print(f"  - Tensor types: {tensor_types_collected}, Tensor sizes: {tensor_sizes_collected}, Expert tensors: {expert_tensors_found}")
                    else:
                        print(f"Analysis running... {elapsed:.1f}s elapsed (waiting for output)")
                    sys.stdout.flush()
                    last_progress_time = current_time
                
                for stream in ready_streams:
                    try:
                        line = stream.readline()
                        if line:
                            if stream == process.stdout:
                                stdout_lines.append(line)
                            else:
                                stderr_lines.append(line)
                            
                            lines_processed += 1
                            last_data_time = current_time  # Reset timer on any new data
                            
                            # Track tensor information more precisely
                            if ("Starting to collect tensor information" in line or 
                                "Collecting tensor information" in line):
                                tensor_info_found = True
                                print("Model is loading tensors, collecting information...")
                                sys.stdout.flush()
                            
                            # Look for tensor type summaries
                            if line.startswith("llama_model_loader: - type"):
                                type_summary_found = True
                                # Extract tensor type and count
                                match = re.search(r'type\s+(\w+):\s+(\d+)\s+tensors', line)
                                if match:
                                    tensor_type = match.group(1)
                                    count = int(match.group(2))
                                    tensor_types_collected += 1
                                    last_data_time = current_time  # Reset no-progress timer
                                    print(f"Found tensor type: {line.strip()}")
                                    sys.stdout.flush()
                            
                            # Progress tracking
                            if lines_processed % 30 == 0:  # Every 30 lines
                                elapsed = time.time() - start_time
                                print(f"Progress: {lines_processed} lines processed, {elapsed:.1f}s elapsed...")
                                # Count what we've found so far
                                current_output = ''.join(stdout_lines)
                                type_count = len(re.findall(r'type\s+\w+:\s+\d+\s+tensors', current_output))
                                tensor_count = len(re.findall(r'tensor.*weight', current_output))
                                expert_count = len(re.findall(r'ffn.*exps', current_output))
                                print(f"  - Tensor types: {type_count}, Tensor sizes: {tensor_count}, Expert tensors: {expert_count}")
                                sys.stdout.flush()
                            
                            # Check for tensor loading start
                            if ("load_tensors: loading model tensors" in line):
                                print("Starting to collect tensor information...")
                                tensor_info_found = True
                                sys.stdout.flush()
                            
                            # Look for error conditions
                            if ("error" in line.lower() or "failed" in line.lower() or 
                                "abort" in line.lower()):
                                print(f"Error/Warning: {line.strip()}")
                                sys.stdout.flush()
                            
                            # Proper completion detection based on actual model loading stages
                            # Break when we have tensor type summary and see model loading completion
                            if type_summary_found and tensor_info_found:
                                # Check for signs that tensor loading is complete
                                if ("model params" in line or "print_info:" in line or 
                                    "llama_context: constructing llama_context" in line):
                                    print("Tensor information collected, model loading complete...")
                                    sys.stdout.flush()
                                    break
                            
                            # Emergency conditions - break immediately
                            if ("cudaMalloc failed: out of memory" in line or
                                "failed to allocate" in line):
                                print("Memory allocation failed - stopping analysis...")
                                sys.stdout.flush()
                                break
                            
                            # If we have collected tensor types and see the model is fully loaded
                            if type_summary_found and "model loaded" in line:
                                print("Model loaded successfully, analysis complete...")
                                sys.stdout.flush()
                                break
                    except:
                        pass
                
                # Small delay to prevent busy waiting
                time.sleep(0.1)
            
            # Wait for process to complete naturally
            if process.poll() is None:
                print("Waiting for llama-server to complete...")
                try:
                    process.wait(timeout=30)  # Wait up to 30 seconds for completion
                except subprocess.TimeoutExpired:
                    print("Process didn't complete, terminating...")
                    process.terminate()
                    time.sleep(2)
                    if process.poll() is None:
                        process.kill()
            
            # Read any remaining output after process completion
            try:
                remaining_stdout, remaining_stderr = process.communicate(timeout=5)
                if remaining_stdout:
                    stdout_lines.append(remaining_stdout)
                if remaining_stderr:
                    stderr_lines.append(remaining_stderr)
            except subprocess.TimeoutExpired:
                pass  # Process already finished
            except Exception as e:
                print(f"Warning: Could not read remaining output: {e}")
            
            output = ''.join(stdout_lines) + ''.join(stderr_lines)
            
            if not output:
                print("Error: No output from llama-server")
                return False
            
            # Check for specific error patterns in stderr before main processing
            stderr_output = ''.join(stderr_lines)
            if stderr_output:
                if "invalid ggml type" in stderr_output:
                    print("DETECTED: Model uses unsupported quantization type")
                    print("This model may require a newer version of llama.cpp")
                    print("or the quantization type is not supported by tensor override analysis")
                elif "failed to load model" in stderr_output:
                    print("DETECTED: Model loading failed")
                    print("The model file may be corrupted or incompatible")
            
            # Parse the output to extract tensor information
            success = self.parse_tensor_output(output)
            
            # If we have tensor type information but no specific tensor data, 
            # use the estimation approach
            if not success and type_summary_found:
                print("Using tensor type summary for estimation...")
                return self.parse_tensor_output(output)
            
            if not success:
                print("ERROR: Failed to extract any tensor information from llama-server output")
                print("This might be due to:")
                print("1. Multi-part GGUF file requiring all parts to be present")
                print("2. Corrupted model file")
                print("3. Incompatible llama.cpp version")
                print("4. Model file too large for available system memory")
                print("5. Unsupported quantization type (e.g., newer IQ formats)")
                # Save output for debugging
                debug_output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_output.txt")
                try:
                    with open(debug_output_file, 'w') as f:
                        f.write("LLAMA-SERVER OUTPUT DEBUG:\n")
                        f.write("=" * 50 + "\n")
                        f.write("STDOUT:\n")
                        f.write(''.join(stdout_lines))
                        f.write("\n" + "=" * 50 + "\n")
                        f.write("STDERR:\n")
                        f.write(''.join(stderr_lines))
                        f.write("\n" + "=" * 50 + "\n")
                        f.write("COMBINED OUTPUT:\n")
                        f.write(output)
                    print(f"Full llama-server output saved to: {debug_output_file}")
                    
                except Exception as e:
                    print(f"Could not save debug output: {e}")
            
            return success
            
        except subprocess.TimeoutExpired:
            print("Error: llama-server analysis timed out")
            return False
        except Exception as e:
            print(f"Error analyzing GGUF file: {e}")
            return False
    
    def parse_tensor_output(self, output: str) -> bool:
        """Parse llama-server output to extract tensor information"""
        try:
            tensors = []
            
            # Parse tensor type summary first (like from your example)
            type_summary = {}
            type_pattern = r'llama_model_loader:\s*-\s*type\s+([^:]+):\s*(\d+)\s*tensors'
            for line in output.split('\n'):
                match = re.search(type_pattern, line)
                if match:
                    tensor_type = match.group(1).strip()
                    count = int(match.group(2))
                    type_summary[tensor_type] = count
                    print(f"Found {count} tensors of type {tensor_type}")
            
            # First, try to extract actual tensor names from the output
            actual_tensor_names = self._extract_actual_tensor_names(output)
            
            # Look for tensor information from verbose output (like from your example)
            # Pattern: tensor blk.15.ffn_gate_exps.weight (1050 MiB iq1_s) buffer type overridden to CPU
            tensor_pattern = r'tensor\s+([^\s]+)\s+\((\d+(?:\.\d+)?)\s*(MiB|GiB)\s+([^)]+)\)'
            
            for line in output.split('\n'):
                match = re.search(tensor_pattern, line)
                if match:
                    name = match.group(1)
                    size_value = float(match.group(2))
                    size_unit = match.group(3)
                    tensor_type = match.group(4)
                    
                    # Convert to MB
                    size_mb = size_value if size_unit == 'MiB' else size_value * 1024
                    
                    tensor = TensorInfo(name, size_mb, tensor_type)
                    
                    # Check if this is being overridden to CPU (likely expert)
                    if "buffer type overridden to CPU" in line:
                        tensor.is_expert = True
                    
                    tensors.append(tensor)
            
            # Look for layer assignments (like from your example)
            layer_pattern = r'load_tensors:\s+layer\s+(\d+)\s+assigned to device\s+(CUDA\d+|CPU)'
            layers_info = {}
            for line in output.split('\n'):
                match = re.search(layer_pattern, line)
                if match:
                    layer_num = int(match.group(1))
                    device = match.group(2)
                    layers_info[layer_num] = device
            
            # If we have actual tensor names, use them to create more accurate estimations
            if actual_tensor_names and type_summary:
                # Create tensors based on actual names with estimated sizes
                tensors = self._create_tensors_from_actual_names(actual_tensor_names, type_summary)
            elif type_summary and not tensors:
                # Fallback to estimation based on type summary and layer info
                tensors = self._estimate_tensors_from_summary(type_summary, layers_info, output)
            
            # Look for general tensor information if available
            if not tensors:
                # Try to parse any tensor mentions
                general_tensor_pattern = r'tensor\s+([^\s]+)\s+\(([^)]+)\)'
                for line in output.split('\n'):
                    match = re.search(general_tensor_pattern, line)
                    if match:
                        name = match.group(1)
                        tensor_type = match.group(2)
                        
                        # Estimate size based on type (rough estimates)
                        size_mb = self._estimate_tensor_size(name, tensor_type)
                        
                        tensor = TensorInfo(name, size_mb, tensor_type)
                        tensors.append(tensor)
            
            self.tensors = tensors
            self.total_size_mb = sum(t.size_mb for t in tensors)
            
            # Categorize tensors
            self.categorize_tensors()
            
            return len(tensors) > 0
            
        except Exception as e:
            print(f"Error parsing tensor output: {e}")
            return False
    
    def _estimate_tensors_from_summary(self, type_summary: Dict[str, int], layers_info: Dict[int, str], output: str) -> List[TensorInfo]:
        """Estimate tensor information from type summary and layer assignments"""
        tensors = []
        
        print(f"Estimating tensors from summary: {type_summary}")
        total_tensors = sum(type_summary.values())
        print(f"Total tensors to distribute: {total_tensors}")
        
        # Base size estimates per quantization type (MB per typical tensor)
        base_sizes = {
            'f32': {'small': 5, 'medium': 50, 'large': 500, 'expert': 1000},
            'q8_0': {'small': 2, 'medium': 20, 'large': 200, 'expert': 400},
            'q4_K': {'small': 1, 'medium': 10, 'large': 100, 'expert': 200},
            'q5_K': {'small': 1.5, 'medium': 15, 'large': 150, 'expert': 300},
            'q6_K': {'small': 2, 'medium': 20, 'large': 200, 'expert': 400},
            'iq2_xxs': {'small': 0.5, 'medium': 5, 'large': 50, 'expert': 100},
            'iq3_xxs': {'small': 0.8, 'medium': 8, 'large': 80, 'expert': 160},
            'iq1_s': {'small': 0.3, 'medium': 3, 'large': 30, 'expert': 1050},  # Expert size from observed data
            'iq3_s': {'small': 0.8, 'medium': 8, 'large': 80, 'expert': 160},
            'iq2_s': {'small': 0.5, 'medium': 5, 'large': 50, 'expert': 100},
            'iq4_xs': {'small': 1, 'medium': 10, 'large': 100, 'expert': 200},
            'iq1_m': {'small': 0.4, 'medium': 4, 'large': 40, 'expert': 120},
        }
        
        # Distribute tensors based on known patterns
        for tensor_type, count in type_summary.items():
            if tensor_type not in base_sizes:
                print(f"Unknown tensor type: {tensor_type}, using default sizes")
                base_sizes[tensor_type] = {'small': 5, 'medium': 50, 'large': 500, 'expert': 1000}
            
            sizes = base_sizes[tensor_type]
            
            # Distribute tensor count across different categories
            # Based on typical transformer architecture:
            # - ~15% expert tensors (large)
            # - ~20% embedding/output tensors (large) 
            # - ~60% layer tensors (medium)
            # - ~5% norm tensors (small)
            
            expert_count = int(count * 0.15)
            embedding_count = int(count * 0.20)
            layer_count = int(count * 0.60)
            norm_count = count - expert_count - embedding_count - layer_count
            
            # Generate expert tensors
            for i in range(expert_count):
                layer_id = i % 61  # Distribute across 61 layers
                expert_types = ['ffn_gate_exps', 'ffn_down_exps', 'ffn_up_exps']
                expert_type = expert_types[i % len(expert_types)]
                name = f"blk.{layer_id}.{expert_type}.weight"
                tensor = TensorInfo(name, sizes['expert'], tensor_type)
                tensor.is_expert = True
                tensors.append(tensor)
            
            # Generate embedding tensors
            for i in range(embedding_count):
                if i == 0:
                    name = "token_embd.weight"
                elif i == 1:
                    name = "output.weight"
                elif i == 2:
                    name = "output_norm.weight"
                else:
                    name = f"embd_{i}.weight"
                tensor = TensorInfo(name, sizes['large'], tensor_type)
                tensors.append(tensor)
            
            # Generate layer tensors
            for i in range(layer_count):
                layer_id = i % 61
                tensor_types = ['attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate', 'ffn_up', 'ffn_down']
                tensor_name = tensor_types[i % len(tensor_types)]
                name = f"blk.{layer_id}.{tensor_name}.weight"
                tensor = TensorInfo(name, sizes['medium'], tensor_type)
                tensors.append(tensor)
            
            # Generate norm tensors
            for i in range(norm_count):
                layer_id = i % 61
                if i == 0:
                    name = "norm.weight"
                else:
                    name = f"blk.{layer_id}.attn_norm.weight" if i % 2 == 0 else f"blk.{layer_id}.ffn_norm.weight"
                tensor = TensorInfo(name, sizes['small'], tensor_type)
                tensors.append(tensor)
        
        print(f"Generated {len(tensors)} estimated tensors from {total_tensors} total")
        return tensors
    
    def _estimate_tensor_size(self, name: str, tensor_type: str) -> float:
        """Estimate tensor size based on name and type"""
        # Base size estimates in MB
        base_sizes = {
            'token_embd': 500.0,
            'output': 500.0,
            'attn_q': 50.0,
            'attn_k': 50.0,
            'attn_v': 50.0,
            'attn_output': 50.0,
            'ffn_gate': 100.0,
            'ffn_up': 100.0,
            'ffn_down': 100.0,
            'ffn_gate_exps': 1000.0,  # Expert layers are typically larger
            'ffn_up_exps': 1000.0,
            'ffn_down_exps': 1000.0,
        }
        
        # Type multipliers
        type_multipliers = {
            'f32': 1.0,
            'q8_0': 0.25,
            'q4_K': 0.125,
            'q5_K': 0.15625,
            'q6_K': 0.1875,
            'iq2_xxs': 0.075,
            'iq3_xxs': 0.1,
            'iq1_s': 0.05,
            'iq3_s': 0.1,
            'iq2_s': 0.075,
            'iq4_xs': 0.125,
            'iq1_m': 0.0625,
        }
        
        # Find matching pattern
        estimated_size = 50.0  # Default size
        for pattern, size in base_sizes.items():
            if pattern in name:
                estimated_size = size
                break
        
        # Apply type multiplier
        multiplier = type_multipliers.get(tensor_type, 1.0)
        return estimated_size * multiplier
    
    def categorize_tensors(self):
        """Categorize tensors into different types"""
        self.expert_tensors = [t for t in self.tensors if t.is_expert]
        self.layer_tensors = [t for t in self.tensors if t.is_layer and not t.is_expert]
        self.embedding_tensors = [t for t in self.tensors if t.is_embedding]
        self.other_tensors = [t for t in self.tensors if not (t.is_expert or t.is_layer or t.is_embedding)]
        
        print(f"Categorized tensors:")
        print(f"  Expert tensors: {len(self.expert_tensors)} ({sum(t.size_mb for t in self.expert_tensors):.1f} MB)")
        print(f"  Layer tensors: {len(self.layer_tensors)} ({sum(t.size_mb for t in self.layer_tensors):.1f} MB)")
        print(f"  Embedding tensors: {len(self.embedding_tensors)} ({sum(t.size_mb for t in self.embedding_tensors):.1f} MB)")
        print(f"  Other tensors: {len(self.other_tensors)} ({sum(t.size_mb for t in self.other_tensors):.1f} MB)")
    
    def get_tensor_analysis(self) -> Dict:
        """Get structured tensor analysis"""
        return {
            'total_tensors': len(self.tensors),
            'total_size_mb': self.total_size_mb,
            'expert_tensors': {
                'count': len(self.expert_tensors),
                'size_mb': sum(t.size_mb for t in self.expert_tensors),
                'tensors': [{'name': t.name, 'size_mb': t.size_mb, 'type': t.tensor_type} for t in self.expert_tensors]
            },
            'layer_tensors': {
                'count': len(self.layer_tensors),
                'size_mb': sum(t.size_mb for t in self.layer_tensors),
                'layers': self.get_layer_breakdown()
            },
            'embedding_tensors': {
                'count': len(self.embedding_tensors),
                'size_mb': sum(t.size_mb for t in self.embedding_tensors),
                'tensors': [{'name': t.name, 'size_mb': t.size_mb, 'type': t.tensor_type} for t in self.embedding_tensors]
            },
            'other_tensors': {
                'count': len(self.other_tensors),
                'size_mb': sum(t.size_mb for t in self.other_tensors),
                'tensors': [{'name': t.name, 'size_mb': t.size_mb, 'type': t.tensor_type} for t in self.other_tensors]
            }
        }
    
    def get_layer_breakdown(self) -> Dict:
        """Get breakdown of tensors by layer"""
        layer_breakdown = {}
        
        for tensor in self.layer_tensors:
            if tensor.layer_id is not None:
                if tensor.layer_id not in layer_breakdown:
                    layer_breakdown[tensor.layer_id] = {
                        'tensors': [],
                        'total_size_mb': 0
                    }
                
                layer_breakdown[tensor.layer_id]['tensors'].append({
                    'name': tensor.name,
                    'size_mb': tensor.size_mb,
                    'type': tensor.tensor_type
                })
                layer_breakdown[tensor.layer_id]['total_size_mb'] += tensor.size_mb
        
        return layer_breakdown
    
    def print_analysis_summary(self):
        """Print a summary of the tensor analysis"""
        print("\nGGUF Tensor Analysis Summary:")
        print("=" * 50)
        print(f"Total tensors: {len(self.tensors)}")
        print(f"Total model size: {self.total_size_mb:.1f} MB ({self.total_size_mb/1024:.1f} GB)")
        print()
        
        print(f"Expert tensors: {len(self.expert_tensors)} ({sum(t.size_mb for t in self.expert_tensors):.1f} MB)")
        print(f"Layer tensors: {len(self.layer_tensors)} ({sum(t.size_mb for t in self.layer_tensors):.1f} MB)")
        print(f"Embedding tensors: {len(self.embedding_tensors)} ({sum(t.size_mb for t in self.embedding_tensors):.1f} MB)")
        print(f"Other tensors: {len(self.other_tensors)} ({sum(t.size_mb for t in self.other_tensors):.1f} MB)")
        
        if self.expert_tensors:
            print(f"\nLargest expert tensors:")
            sorted_experts = sorted(self.expert_tensors, key=lambda t: t.size_mb, reverse=True)[:5]
            for tensor in sorted_experts:
                print(f"  {tensor.name}: {tensor.size_mb:.1f} MB ({tensor.tensor_type})")

    def _extract_actual_tensor_names(self, output: str) -> List[str]:
        """Extract actual tensor names from llama-server output for dynamic pattern generation"""
        tensor_names = []
        
        # Look for tensor loading patterns in the output
        # Pattern 1: "ggml_tensor_get: blk.0.ffn_gate_exps.weight"
        # Pattern 2: "load_tensor: blk.1.attn_q.weight"
        # Pattern 3: Direct tensor mentions during loading
        
        tensor_patterns = [
            r'ggml_tensor_get:\s+([^\s]+)',
            r'load_tensor:\s+([^\s]+)',
            r'tensor\s+([^\s]+)\s+\(',
            r'loading tensor:\s+([^\s]+)',
            r'tensor_name:\s+([^\s]+)',
        ]
        
        for line in output.split('\n'):
            for pattern in tensor_patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    if 'blk.' in match and '.weight' in match:
                        tensor_names.append(match)
        
        # Remove duplicates and sort
        unique_names = sorted(list(set(tensor_names)))
        
        if unique_names:
            print(f"Extracted {len(unique_names)} actual tensor names from model")
            return unique_names
        else:
            print("No actual tensor names found in output - falling back to estimation")
            return []

    def _create_tensors_from_actual_names(self, tensor_names: List[str], type_summary: Dict[str, int]) -> List[TensorInfo]:
        """Create tensors using actual tensor names from the model with estimated sizes"""
        tensors = []
        
        print(f"Creating tensors from {len(tensor_names)} actual tensor names")
        
        # Create a mapping of tensor types to their typical sizes
        type_size_map = {
            'f32': 4.0,    # bytes per parameter
            'q8_0': 1.125,
            'q4_K': 0.5,
            'q5_K': 0.625,
            'q6_K': 0.75,
            'iq2_xxs': 0.25,
            'iq3_xxs': 0.375,
            'iq1_s': 0.125,
            'iq3_s': 0.375,
            'iq2_s': 0.25,
            'iq4_xs': 0.5,
            'iq1_m': 0.1875,
        }
        
        # Estimate sizes for each actual tensor name
        for name in tensor_names:
            # Determine tensor type and size based on patterns
            estimated_size = self._estimate_tensor_size(name, 'iq1_s')  # Default type
            
            # Try to assign a more appropriate type based on name patterns
            if 'ffn_gate_exps' in name or 'ffn_down_exps' in name or 'ffn_up_exps' in name:
                # Expert tensors are typically larger
                estimated_size = 1050.0
                tensor_type = 'iq1_s'
            elif 'attn_' in name:
                # Attention tensors
                estimated_size = 150.0
                tensor_type = 'iq1_s'
            elif 'ffn_' in name:
                # FFN tensors
                estimated_size = 300.0
                tensor_type = 'iq1_s'
            elif 'embd' in name or 'output' in name:
                # Embedding tensors
                estimated_size = 1500.0
                tensor_type = 'q4_K'
            elif 'norm' in name:
                # Normalization tensors
                estimated_size = 14.0
                tensor_type = 'f32'
            else:
                # Default
                estimated_size = 50.0
                tensor_type = 'iq1_s'
            
            tensor = TensorInfo(name, estimated_size, tensor_type)
            tensors.append(tensor)
        
        print(f"Created {len(tensors)} tensors from actual names")
        return tensors

def main():
    if len(sys.argv) < 2:
        print("Usage: python gguf_analyzer.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    analyzer = GGUFAnalyzer()
    
    print(f"Analyzing GGUF model: {model_path}")
    if analyzer.analyze_gguf_file(model_path):
        analyzer.print_analysis_summary()
        
        # Save analysis to file
        analysis = analyzer.get_tensor_analysis()
        output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tensor_analysis.json")
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"\nTensor analysis saved to {output_file}")
    else:
        print("Failed to analyze GGUF file")
        sys.exit(1)

if __name__ == "__main__":
    main()