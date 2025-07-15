#!/usr/bin/env python3
"""
Simple Tensor Override Analyzer - Uses straightforward capacity-based optimization
"""

import argparse
import json
import os
import sys
from datetime import datetime

from cuda_detector import CUDADetector
from gguf_analyzer import GGUFAnalyzer
from simple_capacity_optimizer import SimpleCapacityOptimizer
from parameter_generator_simple import SimpleParameterGenerator

def main():
    parser = argparse.ArgumentParser(description="Simple capacity-based tensor override analyzer")
    parser.add_argument('model_path', help='Path to GGUF model file')
    parser.add_argument('--config', default='config.json', help='Configuration file path')
    parser.add_argument('--output-dir', default=os.path.dirname(os.path.abspath(__file__)), help='Output directory')
    parser.add_argument('--output-filename', default='simple_tensor_params.txt', help='Output filename')
    parser.add_argument('--dry-run', action='store_true', help='Analyze without generating parameters')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Load configuration
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(script_dir, args.config)
        with open(config_file, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load config file {args.config}: {e}")
        config = {
            "llama_cpp_path": "/home/rgilbreth/Desktop/AI-Software/llama.cpp",
            "virtual_env_path": "/home/rgilbreth/pytorch-venv",
            "context_size": 131072,
            "kv_cache": {"type_k": "q4_0", "type_v": "q4_0"}
        }
    
    print("=" * 70)
    print("SIMPLE Tensor Override Analyzer")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Approach: Straightforward capacity-based optimization")
    print()
    
    # Step 1: Detect CUDA devices
    print("Step 1: Detecting CUDA devices...")
    cuda_detector = CUDADetector(args.config)
    devices = cuda_detector.detect_devices()
    
    if not devices:
        print("Error: No CUDA devices detected")
        sys.exit(1)
    
    device_info = cuda_detector.get_device_info()
    print(f"Found {len(devices)} CUDA devices")
    print()
    
    # Step 2: Analyze GGUF model
    print("Step 2: Analyzing GGUF model...")
    gguf_analyzer = GGUFAnalyzer(args.config)
    
    if not gguf_analyzer.analyze_gguf_file(args.model_path):
        print("Error: Failed to analyze GGUF model")
        sys.exit(1)
    
    tensor_analysis = gguf_analyzer.get_tensor_analysis()
    print(f"Analyzed {tensor_analysis['total_tensors']} tensors")
    print()
    
    # Step 3: Simple capacity-based optimization
    print("Step 3: Simple capacity-based tensor placement...")
    optimizer = SimpleCapacityOptimizer(device_info['devices'], config)
    assignments = optimizer.optimize_simple(tensor_analysis)
    print()
    
    # Step 4: Generate parameters
    if not args.dry_run:
        print("Step 4: Generating tensor override parameters...")
        
        # Convert assignments to the format expected by parameter generator
        tensor_assignments = []
        for assignment in assignments:
            tensor_assignments.append({
                'tensor_name': assignment['name'],
                'device': assignment['device'],
                'size_mb': assignment['size_mb']
            })
        
        param_generator = SimpleParameterGenerator(tensor_assignments, config)
        
        # Save parameters
        param_file = os.path.join(args.output_dir, args.output_filename)
        config_name = args.output_filename.replace('_tensor_params.txt', '').replace('.txt', '')
        param_generator.save_parameters(param_file, args.model_path, config_name)
        
        # Print summary
        param_summary = param_generator.get_parameter_summary()
        print(f"Generated {param_summary['total_parameters']} override parameters")
        print(f"Parameters saved to: {param_file}")
        print()
        
        # Show example usage
        print("Example llama.cpp usage:")
        print("-" * 40)
        
        llama_cpp_path = config.get('llama_cpp_path', '/path/to/llama.cpp')
        server_path = os.path.join(llama_cpp_path, 'build', 'bin', 'llama-server')
        
        print(f"{server_path} \\")
        print(f"  --model {args.model_path} \\")
        print(f"  --ctx-size {config.get('context_size', 8192)} \\")
        
        # Show first few parameters
        for i, param in enumerate(param_summary['parameters'][:3]):
            print(f"  {param} \\")
        
        if len(param_summary['parameters']) > 3:
            print(f"  # ... and {len(param_summary['parameters']) - 3} more parameters")
        print()
    else:
        print("Dry run mode - parameters not generated")
    
    print("Simple analysis complete!")

if __name__ == "__main__":
    main()