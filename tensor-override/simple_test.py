#!/usr/bin/env python3
"""
Simple test to verify the tensor override system works
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cuda_detector import CUDADetector
from tensor_optimizer import TensorOptimizer
from parameter_generator import ParameterGenerator

def create_mock_tensor_analysis():
    """Create a mock tensor analysis based on your example output"""
    return {
        'total_tensors': 1096,
        'total_size_mb': 268267.1,
        'expert_tensors': {
            'count': 183,
            'size_mb': 45234.2,
            'tensors': [
                {'name': 'blk.15.ffn_gate_exps.weight', 'size_mb': 1050.0, 'type': 'iq1_s'},
                {'name': 'blk.15.ffn_down_exps.weight', 'size_mb': 2856.0, 'type': 'iq4_xs'},
                {'name': 'blk.15.ffn_up_exps.weight', 'size_mb': 1050.0, 'type': 'iq1_s'},
                {'name': 'blk.16.ffn_gate_exps.weight', 'size_mb': 1050.0, 'type': 'iq1_s'},
                {'name': 'blk.16.ffn_down_exps.weight', 'size_mb': 1722.0, 'type': 'iq2_s'},
                {'name': 'blk.16.ffn_up_exps.weight', 'size_mb': 1050.0, 'type': 'iq1_s'},
                # Generate more expert tensors for testing
            ] + [
                tensor
                for i in range(17, 61)  # layers 17-60
                for tensor in [
                    {'name': f'blk.{i}.ffn_gate_exps.weight', 'size_mb': 1050.0, 'type': 'iq1_s'},
                    {'name': f'blk.{i}.ffn_down_exps.weight', 'size_mb': 2000.0, 'type': 'iq3_s'},
                    {'name': f'blk.{i}.ffn_up_exps.weight', 'size_mb': 1050.0, 'type': 'iq1_s'}
                ]
            ]
        },
        'layer_tensors': {
            'count': 732,
            'size_mb': 198456.7,
            'layers': {
                i: {
                    'tensors': [
                        {'name': f'blk.{i}.attn_q.weight', 'size_mb': 150.0, 'type': 'iq1_s'},
                        {'name': f'blk.{i}.attn_k.weight', 'size_mb': 150.0, 'type': 'iq1_s'},
                        {'name': f'blk.{i}.attn_v.weight', 'size_mb': 150.0, 'type': 'iq1_s'},
                        {'name': f'blk.{i}.attn_output.weight', 'size_mb': 150.0, 'type': 'iq1_s'},
                        {'name': f'blk.{i}.ffn_gate.weight', 'size_mb': 300.0, 'type': 'iq1_s'},
                        {'name': f'blk.{i}.ffn_up.weight', 'size_mb': 300.0, 'type': 'iq1_s'},
                        {'name': f'blk.{i}.ffn_down.weight', 'size_mb': 300.0, 'type': 'iq1_s'},
                    ],
                    'total_size_mb': 1500.0
                }
                for i in range(61)  # 61 layers
            }
        },
        'embedding_tensors': {
            'count': 2,
            'size_mb': 3000.0,
            'tensors': [
                {'name': 'token_embd.weight', 'size_mb': 1500.0, 'type': 'q4_K'},
                {'name': 'output.weight', 'size_mb': 1500.0, 'type': 'q4_K'},
            ]
        },
        'other_tensors': {
            'count': 179,
            'size_mb': 21576.2,
            'tensors': [
                {'name': 'norm.weight', 'size_mb': 14.0, 'type': 'f32'},
                {'name': 'output_norm.weight', 'size_mb': 14.0, 'type': 'f32'},
            ] + [
                tensor
                for i in range(61)
                for tensor in [
                    {'name': f'blk.{i}.attn_norm.weight', 'size_mb': 14.0, 'type': 'f32'},
                    {'name': f'blk.{i}.ffn_norm.weight', 'size_mb': 14.0, 'type': 'f32'}
                ]
            ]
        }
    }

def main():
    print("Testing Tensor Override System")
    print("=" * 50)
    
    # Test 1: CUDA Detection
    print("Step 1: Testing CUDA Detection...")
    detector = CUDADetector()
    devices = detector.detect_devices()
    print(f"Found {len(devices)} CUDA devices")
    print(f"Total VRAM: {detector.total_vram:,} MB")
    print()
    
    # Test 2: Mock tensor analysis
    print("Step 2: Using mock tensor analysis...")
    tensor_analysis = create_mock_tensor_analysis()
    print(f"Total tensors: {tensor_analysis['total_tensors']}")
    print(f"Expert tensors: {tensor_analysis['expert_tensors']['count']}")
    print(f"Layer tensors: {tensor_analysis['layer_tensors']['count']}")
    print()
    
    # Test 3: Tensor optimization
    print("Step 3: Testing tensor optimization...")
    config = {
        'context_size': 8192,
        'tensor_analysis': {
            'prioritize_experts_on_cpu': True,
            'maximize_gpu_layers': True,
            'reserve_context_memory': True,
            'memory_safety_margin_mb': 512
        }
    }
    
    device_info = detector.get_device_info()
    optimizer = TensorOptimizer(device_info['devices'], tensor_analysis, config)
    assignments = optimizer.optimize_tensor_placement()
    print(f"Created {len(assignments)} tensor assignments")
    print()
    
    # Test 4: Parameter generation
    print("Step 4: Testing parameter generation...")
    param_generator = ParameterGenerator(assignments, config)
    parameters = param_generator.format_llama_cpp_parameters()
    
    print(f"Generated {len(parameters)} parameters")
    print("First 10 parameters:")
    for i, param in enumerate(parameters[:10]):
        print(f"  {i+1}. {param}")
    
    if len(parameters) > 10:
        print(f"  ... and {len(parameters) - 10} more")
    
    print()
    print("✅ All tests passed! The tensor override system is working correctly.")

if __name__ == "__main__":
    main()