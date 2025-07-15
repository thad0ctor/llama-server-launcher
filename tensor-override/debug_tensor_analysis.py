#!/usr/bin/env python3
"""
Debug Tensor Analysis
Analyzes the existing tensor analysis data to identify GPU utilization issues.
"""

import json
import os
from tensor_optimizer import TensorOptimizer, DeviceType

def load_existing_analysis():
    """Load the existing tensor analysis data"""
    try:
        with open('tensor_analysis.json', 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading tensor analysis: {e}")
        return None

def debug_tensor_analysis():
    """Debug the tensor analysis and optimization"""
    
    # Load existing analysis
    data = load_existing_analysis()
    if not data:
        return
    
    print("=== TENSOR ANALYSIS DEBUG ===")
    print()
    
    # Extract components
    cuda_devices = data['cuda_devices']['devices']
    tensor_analysis = data['tensor_analysis']
    config = data['config']
    
    print("CUDA Devices:")
    for device in cuda_devices:
        print(f"  CUDA{device['device_id']}: {device['free_memory_mb']:.0f} MB free")
    print()
    
    print("Tensor Summary:")
    print(f"  Total tensors: {tensor_analysis['total_tensors']}")
    print(f"  Total size: {tensor_analysis['total_size_mb']:.1f} MB")
    print(f"  Expert tensors: {tensor_analysis['expert_tensors']['count']} ({tensor_analysis['expert_tensors']['size_mb']:.1f} MB)")
    print(f"  Layer tensors: {tensor_analysis['layer_tensors']['count']} ({tensor_analysis['layer_tensors']['size_mb']:.1f} MB)")
    print(f"  Embedding tensors: {tensor_analysis['embedding_tensors']['count']} ({tensor_analysis['embedding_tensors']['size_mb']:.1f} MB)")
    print(f"  Other tensors: {tensor_analysis['other_tensors']['count']} ({tensor_analysis['other_tensors']['size_mb']:.1f} MB)")
    print()
    
    # Check tensor size estimates
    print("=== TENSOR SIZE ANALYSIS ===")
    expert_tensors = tensor_analysis['expert_tensors']['tensors']
    print(f"Expert tensor sizes (first 10):")
    for i, tensor in enumerate(expert_tensors[:10]):
        print(f"  {tensor['name']}: {tensor['size_mb']} MB")
    
    # Check if all experts are same size (suspicious)
    expert_sizes = [t['size_mb'] for t in expert_tensors]
    unique_sizes = set(expert_sizes)
    print(f"Unique expert tensor sizes: {unique_sizes}")
    if len(unique_sizes) == 1:
        print("WARNING: All expert tensors have the same size - this might be incorrect estimation!")
    print()
    
    # Test the optimizer
    print("=== OPTIMIZATION TEST ===")
    try:
        optimizer = TensorOptimizer(cuda_devices, tensor_analysis, config)
        assignments = optimizer.optimize_tensor_placement()
        
        # Analyze GPU utilization
        print("GPU Utilization Analysis:")
        total_gpu_capacity = sum(optimizer.gpu_capacities.values())
        total_gpu_used = sum(optimizer.gpu_used.values())
        
        print(f"Total GPU capacity (after context): {total_gpu_capacity:.1f} MB")
        print(f"Total GPU used: {total_gpu_used:.1f} MB")
        print(f"Overall utilization: {(total_gpu_used/total_gpu_capacity*100):.1f}%")
        print()
        
        for device_id in sorted(optimizer.gpu_capacities.keys()):
            capacity = optimizer.gpu_capacities[device_id]
            used = optimizer.gpu_used[device_id]
            utilization = (used / capacity * 100) if capacity > 0 else 0
            print(f"CUDA{device_id}: {used:.1f} MB / {capacity:.1f} MB ({utilization:.1f}%)")
        
        # Check what's assigned to each device
        print()
        print("=== DEVICE ASSIGNMENTS ===")
        cpu_assignments = [a for a in assignments if a.device_type == DeviceType.CPU]
        gpu_assignments = [a for a in assignments if a.device_type == DeviceType.GPU]
        
        print(f"CPU assignments: {len(cpu_assignments)} tensors")
        cpu_size = sum(a.size_mb for a in cpu_assignments)
        print(f"CPU total size: {cpu_size:.1f} MB")
        
        for device_id in sorted(set(a.device_id for a in gpu_assignments)):
            device_assignments = [a for a in gpu_assignments if a.device_id == device_id]
            device_size = sum(a.size_mb for a in device_assignments)
            print(f"CUDA{device_id}: {len(device_assignments)} tensors, {device_size:.1f} MB")
        
        # Check for unused devices
        used_devices = set(a.device_id for a in gpu_assignments)
        all_devices = set(d['device_id'] for d in cuda_devices)
        unused_devices = all_devices - used_devices
        if unused_devices:
            print(f"UNUSED DEVICES: {unused_devices}")
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_tensor_analysis()