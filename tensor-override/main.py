#!/usr/bin/env python3
"""
Main Tensor Override Analyzer
Orchestrates CUDA detection, GGUF analysis, and tensor optimization for llama.cpp.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List

# Import our modules
from cuda_detector import CUDADetector
from gguf_analyzer import GGUFAnalyzer
from tensor_optimizer import TensorOptimizer
from parameter_generator_simple import SimpleParameterGenerator

def main():
    parser = argparse.ArgumentParser(
        description="Analyze CUDA devices and GGUF models to optimize tensor placement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py model.gguf
  python main.py model.gguf --config custom_config.json
  python main.py model.gguf --output-dir ./analysis
  python main.py model.gguf --dry-run
        """
    )
    
    parser.add_argument('model_path', help='Path to GGUF model file')
    parser.add_argument('--config', default='config.json', help='Configuration file path')
    parser.add_argument('--output-dir', default=os.path.dirname(os.path.abspath(__file__)), help='Output directory for analysis files')
    parser.add_argument('--output-filename', default='tensor_override_params.txt', help='Custom filename for tensor parameters')
    parser.add_argument('--dry-run', action='store_true', help='Analyze without generating parameters')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Load configuration
    config = load_config(args.config)
    if args.verbose:
        config['output']['verbose'] = True
    
    print("=" * 70)
    print("Tensor Override Analyzer for llama.cpp")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Config: {args.config}")
    print(f"Output: {args.output_dir}")
    print()
    
    # Step 1: Detect CUDA devices
    print("Step 1: Detecting CUDA devices...")
    cuda_detector = CUDADetector(args.config)
    devices = cuda_detector.detect_devices()
    
    if not devices:
        print("Error: No CUDA devices detected. Cannot optimize tensor placement.")
        sys.exit(1)
    
    if config.get('output', {}).get('verbose', False):
        cuda_detector.print_device_summary()
    
    # Context memory will be handled per-GPU in tensor optimization
    context_memory = cuda_detector.reserve_memory_for_context()
    print(f"Context memory to distribute: {context_memory} MB")
    print()
    
    # Step 2: Analyze GGUF model
    print("Step 2: Analyzing GGUF model...")
    gguf_analyzer = GGUFAnalyzer(args.config)
    
    if not gguf_analyzer.analyze_gguf_file(args.model_path):
        print("Error: Failed to analyze GGUF model")
        sys.exit(1)
    
    if config.get('output', {}).get('verbose', False):
        gguf_analyzer.print_analysis_summary()
    print()
    
    # Step 3: Optimize tensor placement
    print("Step 3: Optimizing tensor placement...")
    device_info = cuda_detector.get_device_info()
    tensor_analysis = gguf_analyzer.get_tensor_analysis()
    
    optimizer = TensorOptimizer(device_info['devices'], tensor_analysis, config)
    assignments = optimizer.optimize_tensor_placement()
    print()
    
    # Step 4: Generate parameters
    if not args.dry_run:
        print("Step 4: Generating llama.cpp parameters...")
        param_generator = SimpleParameterGenerator(assignments, config)
        
        # Save parameters
        param_file = os.path.join(args.output_dir, args.output_filename)
        # Extract config name from filename if possible
        config_name = args.output_filename.replace('_tensor_params.txt', '').replace(os.path.basename(args.model_path).replace('.gguf', ''), '').strip('_')
        param_generator.save_parameters(param_file, args.model_path, config_name)
        
        # Print parameter summary
        param_summary = param_generator.get_parameter_summary()
        print(f"Generated {param_summary['total_parameters']} override parameters")
        print(f"  CPU patterns: {param_summary['cpu_patterns']}")
        print(f"  GPU patterns: {param_summary['gpu_patterns']}")
        print()
        
        # Print example usage
        print("Example llama.cpp usage:")
        print("-" * 40)
        
        # Create a command line example
        llama_cpp_path = config.get('llama_cpp_path', '/path/to/llama.cpp')
        server_path = os.path.join(llama_cpp_path, 'build', 'bin', 'llama-server')
        
        print(f"{server_path} \\")
        print(f"  --model {args.model_path} \\")
        print(f"  --ctx-size {config.get('context_size', 8192)} \\")
        
        # Add first few parameters as example
        example_params = param_summary['parameters'][:3]
        for param in example_params:
            print(f"  {param} \\")
        
        if len(param_summary['parameters']) > 3:
            print(f"  # ... and {len(param_summary['parameters']) - 3} more -ot parameters")
        
        print()
        print(f"Full parameter list available in: {param_file}")
    else:
        print("Dry run mode - parameters not generated")
    
    # Step 5: Save analysis results
    if config.get('output', {}).get('save_analysis_json', True):
        analysis_file = os.path.join(args.output_dir, 'tensor_analysis.json')
        
        analysis_data = {
            'timestamp': datetime.now().isoformat(),
            'model_path': args.model_path,
            'config': config,
            'cuda_devices': device_info,
            'tensor_analysis': tensor_analysis,
            'assignments': optimizer.get_assignment_data(),
        }
        
        if not args.dry_run:
            analysis_data['parameters'] = param_generator.get_parameter_summary()
        
        with open(analysis_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print(f"Analysis results saved to: {analysis_file}")
    
    print("\nAnalysis complete!")
    print_final_summary(optimizer.get_assignment_data())

def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(script_dir, config_path)
        
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load config file {config_path}: {e}")
        print("Using default configuration")
        return {
            "llama_cpp_path": "/home/rgilbreth/Desktop/AI-Software/llama.cpp",
            "virtual_env_path": None,
            "context_size": 8192,
            "tensor_analysis": {
                "prioritize_experts_on_cpu": True,
                "maximize_gpu_layers": True,
                "reserve_context_memory": True,
                "memory_safety_margin_mb": 512
            },
            "output": {
                "save_analysis_json": True,
                "save_override_params": True,
                "verbose": False
            }
        }

def print_final_summary(assignment_data: Dict):
    """Print final summary of the optimization"""
    summary = assignment_data['summary']
    
    print("\nFinal Summary:")
    print("=" * 50)
    print(f"Total tensors analyzed: {assignment_data['total_tensors']}")
    print(f"CPU assignment: {summary['cpu_tensor_count']} tensors ({summary['cpu_total_size_mb']:.1f} MB)")
    print(f"GPU assignment: {summary['gpu_tensor_count']} tensors ({summary['gpu_total_size_mb']:.1f} MB)")
    print()
    
    print("GPU utilization:")
    for device_id, utilization in summary['gpu_utilization'].items():
        print(f"  CUDA{device_id}: {utilization['used_mb']:.1f} MB / {utilization['capacity_mb']:.1f} MB ({utilization['utilization_percent']:.1f}%)")
    
    total_memory = summary['cpu_total_size_mb'] + summary['gpu_total_size_mb']
    gpu_percentage = (summary['gpu_total_size_mb'] / total_memory * 100) if total_memory > 0 else 0
    
    print(f"\nOverall GPU utilization: {gpu_percentage:.1f}% of model on GPU")
    
    # Performance tips
    print("\nPerformance Tips:")
    print("- Expert tensors on CPU reduce memory pressure while maintaining speed")
    print("- Layer tensors on GPU maximize inference speed")
    print("- High GPU utilization indicates optimal memory usage")
    
    if gpu_percentage > 90:
        print("- Excellent GPU utilization achieved!")
    elif gpu_percentage > 70:
        print("- Good GPU utilization. Consider more aggressive optimization if needed.")
    else:
        print("- Low GPU utilization. Check if model is too large for available VRAM.")

if __name__ == "__main__":
    main()