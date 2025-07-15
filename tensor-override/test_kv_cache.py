#!/usr/bin/env python3
"""
Test K/V cache memory calculations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cuda_detector import CUDADetector

def test_kv_cache_configs():
    """Test different K/V cache configurations"""
    
    configs = [
        {
            "name": "Default f16",
            "config": {
                "context_size": 132000,
                "kv_cache": {"type": "f16", "type_k": "f16", "type_v": "f16"},
                "tensor_analysis": {"memory_safety_margin_mb": 512}
            }
        },
        {
            "name": "Quantized q8_0",
            "config": {
                "context_size": 132000,
                "kv_cache": {"type": "q8_0", "type_k": "q8_0", "type_v": "q8_0"},
                "tensor_analysis": {"memory_safety_margin_mb": 512}
            }
        },
        {
            "name": "Quantized q4_0",
            "config": {
                "context_size": 132000,
                "kv_cache": {"type": "q4_0", "type_k": "q4_0", "type_v": "q4_0"},
                "tensor_analysis": {"memory_safety_margin_mb": 512}
            }
        },
        {
            "name": "Mixed K=q4_0, V=q8_0",
            "config": {
                "context_size": 132000,
                "kv_cache": {"type": "f16", "type_k": "q4_0", "type_v": "q8_0"},
                "tensor_analysis": {"memory_safety_margin_mb": 512}
            }
        }
    ]
    
    print("K/V Cache Memory Usage Comparison")
    print("=" * 60)
    print(f"Context Size: 132,000 tokens")
    print(f"Model: Kimi/DeepSeek2 (7168 embd, 61 layers)")
    print()
    
    for test_config in configs:
        print(f"Configuration: {test_config['name']}")
        print("-" * 40)
        
        # Create detector with this config
        detector = CUDADetector()
        detector.config = test_config['config']
        
        # Calculate memory usage
        context_memory = detector._calculate_kv_cache_memory(
            test_config['config']['context_size'],
            test_config['config']['kv_cache']['type_k'],
            test_config['config']['kv_cache']['type_v']
        )
        
        # Show breakdown
        kv_config = test_config['config']['kv_cache']
        print(f"  K cache type: {kv_config['type_k']}")
        print(f"  V cache type: {kv_config['type_v']}")
        print(f"  K/V cache memory: {context_memory:,} MB ({context_memory/1024:.1f} GB)")
        
        # Calculate with overhead
        total_with_overhead = context_memory + int(context_memory * 0.2) + 512
        print(f"  Total with overhead: {total_with_overhead:,} MB ({total_with_overhead/1024:.1f} GB)")
        
        # Show savings vs f16
        if test_config['name'] != "Default f16":
            f16_memory = 132000 * 7168 * 61 * 2 * 2 / (1024 * 1024)  # f16 calculation
            savings = f16_memory - context_memory
            savings_percent = (savings / f16_memory) * 100
            print(f"  Memory savings: {savings:,.0f} MB ({savings_percent:.1f}%)")
        
        print()

if __name__ == "__main__":
    test_kv_cache_configs()