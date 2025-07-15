#!/usr/bin/env python3
"""
Simple Capacity-Based Tensor Optimizer
Uses straightforward capacity math and clear priorities to maximize GPU utilization.
"""

import json
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class TensorInfo:
    name: str
    size_mb: float
    priority: int  # 1=highest, 5=lowest
    device: str = "CPU"  # Default to CPU

class SimpleCapacityOptimizer:
    def __init__(self, cuda_devices: List[Dict], config: Dict):
        self.cuda_devices = cuda_devices
        self.config = config
        self.gpu_capacities = {}
        self.gpu_used = {}
        
        # Calculate actual available capacity per GPU
        self._calculate_realistic_capacities()
        
    def _calculate_realistic_capacities(self):
        """Calculate realistic GPU capacities with minimal overhead"""
        context_size = self.config.get('context_size', 8192)
        
        # Simple KV cache calculation
        # 131K tokens * 7168 embd_dim * 2 bytes (q4_0) * 61 layers / 1024^2
        kv_cache_total_mb = (context_size * 7168 * 1.0 * 61) / (1024 * 1024)  # q4_0 = 1 byte per element
        
        # Distribute KV cache evenly across GPUs (simple approach)
        kv_cache_per_gpu = kv_cache_total_mb / len(self.cuda_devices)
        
        # Add minimal overhead: 512MB safety + 256MB compute buffers
        overhead_per_gpu = 512 + 256
        
        print(f"Realistic capacity calculation:")
        print(f"  KV cache total: {kv_cache_total_mb:.0f} MB")
        print(f"  KV cache per GPU: {kv_cache_per_gpu:.0f} MB")
        print(f"  Overhead per GPU: {overhead_per_gpu} MB")
        
        for device in self.cuda_devices:
            device_id = device['device_id']
            total_memory = device['free_memory_mb']
            
            # Available = Total - KV Cache - Overhead
            available = total_memory - kv_cache_per_gpu - overhead_per_gpu
            
            self.gpu_capacities[device_id] = max(0, available)
            self.gpu_used[device_id] = 0
            
            print(f"  CUDA{device_id}: {total_memory:.0f} MB total → {available:.0f} MB available")
        
        total_gpu_capacity = sum(self.gpu_capacities.values())
        print(f"Total GPU capacity for tensors: {total_gpu_capacity:.0f} MB ({total_gpu_capacity/1024:.1f} GB)")
        
    def optimize_simple(self, tensor_analysis: Dict) -> List[Dict]:
        """
        Simple optimization: Fill GPUs by priority until capacity is reached
        """
        # Extract and prioritize all tensors
        all_tensors = self._extract_and_prioritize_tensors(tensor_analysis)
        
        print(f"\nSimple capacity-based optimization:")
        print(f"Total tensors to place: {len(all_tensors)}")
        print(f"Total tensor size: {sum(t.size_mb for t in all_tensors):.0f} MB")
        
        # Sort by priority (1=highest priority first)
        all_tensors.sort(key=lambda x: (x.priority, -x.size_mb))
        
        # Fill GPUs in round-robin fashion by priority
        assignments = []
        gpu_ids = list(self.gpu_capacities.keys())
        current_gpu_idx = 0
        
        for tensor in all_tensors:
            placed = False
            attempts = 0
            
            # Try to place on GPUs starting from current GPU
            while attempts < len(gpu_ids) and not placed:
                gpu_id = gpu_ids[current_gpu_idx]
                
                # Check if tensor fits
                if self.gpu_used[gpu_id] + tensor.size_mb <= self.gpu_capacities[gpu_id]:
                    # Place on this GPU
                    tensor.device = f"CUDA{gpu_id}"
                    self.gpu_used[gpu_id] += tensor.size_mb
                    placed = True
                else:
                    # Try next GPU
                    current_gpu_idx = (current_gpu_idx + 1) % len(gpu_ids)
                    attempts += 1
            
            # If couldn't place on any GPU, keep on CPU
            if not placed:
                tensor.device = "CPU"
            
            assignments.append({
                'name': tensor.name,
                'size_mb': tensor.size_mb,
                'device': tensor.device,
                'priority': tensor.priority
            })
        
        self._print_assignment_summary(assignments)
        return assignments
        
    def _extract_and_prioritize_tensors(self, tensor_analysis: Dict) -> List[TensorInfo]:
        """Extract tensors and assign priorities"""
        tensors = []
        
        # Priority 1: Attention layers (highest impact on speed)
        for tensor_data in tensor_analysis.get('layer_tensors', {}).get('tensors', []):
            if 'attn' in tensor_data['name']:
                tensors.append(TensorInfo(
                    name=tensor_data['name'],
                    size_mb=tensor_data['size_mb'],
                    priority=1
                ))
        
        # Priority 2: FFN layers (good impact on speed, but not experts)
        for tensor_data in tensor_analysis.get('layer_tensors', {}).get('tensors', []):
            if 'ffn' in tensor_data['name'] and 'exps' not in tensor_data['name']:
                tensors.append(TensorInfo(
                    name=tensor_data['name'],
                    size_mb=tensor_data['size_mb'],
                    priority=2
                ))
        
        # Priority 3: Embeddings (moderate impact)
        for tensor_data in tensor_analysis.get('embedding_tensors', {}).get('tensors', []):
            tensors.append(TensorInfo(
                name=tensor_data['name'],
                size_mb=tensor_data['size_mb'],
                priority=3
            ))
        
        # Priority 4: Other tensors (norms, etc.)
        for tensor_data in tensor_analysis.get('other_tensors', {}).get('tensors', []):
            tensors.append(TensorInfo(
                name=tensor_data['name'],
                size_mb=tensor_data['size_mb'],
                priority=4
            ))
        
        # Priority 5: Expert tensors (lowest priority, usually keep on CPU)
        for tensor_data in tensor_analysis.get('expert_tensors', {}).get('tensors', []):
            tensors.append(TensorInfo(
                name=tensor_data['name'],
                size_mb=tensor_data['size_mb'],
                priority=5
            ))
        
        return tensors
    
    def _print_assignment_summary(self, assignments: List[Dict]):
        """Print clear summary of assignments"""
        print(f"\n" + "="*60)
        print("SIMPLE CAPACITY-BASED ASSIGNMENT SUMMARY")
        print("="*60)
        
        # Count by device and priority
        device_stats = {}
        priority_stats = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        for assignment in assignments:
            device = assignment['device']
            priority = assignment['priority']
            size = assignment['size_mb']
            
            if device not in device_stats:
                device_stats[device] = {'count': 0, 'size_mb': 0}
            
            device_stats[device]['count'] += 1
            device_stats[device]['size_mb'] += size
            priority_stats[priority] += size
        
        # Print device utilization
        total_gpu_mb = 0
        for device_id, capacity in self.gpu_capacities.items():
            device_name = f"CUDA{device_id}"
            if device_name in device_stats:
                used_mb = device_stats[device_name]['size_mb']
                count = device_stats[device_name]['count']
                utilization = (used_mb / capacity * 100) if capacity > 0 else 0
                print(f"{device_name}: {used_mb:.0f} MB / {capacity:.0f} MB ({utilization:.1f}%) - {count} tensors")
                total_gpu_mb += used_mb
            else:
                print(f"{device_name}: 0 MB / {capacity:.0f} MB (0.0%) - 0 tensors")
        
        if 'CPU' in device_stats:
            cpu_mb = device_stats['CPU']['size_mb']
            cpu_count = device_stats['CPU']['count']
            print(f"CPU: {cpu_mb:.0f} MB - {cpu_count} tensors")
        else:
            cpu_mb = 0
            
        total_mb = total_gpu_mb + cpu_mb
        gpu_percentage = (total_gpu_mb / total_mb * 100) if total_mb > 0 else 0
        
        print(f"\nOverall GPU Utilization: {gpu_percentage:.1f}% of model on GPU")
        print(f"GPU Memory Used: {total_gpu_mb:.0f} MB ({total_gpu_mb/1024:.1f} GB)")
        print(f"CPU Memory Used: {cpu_mb:.0f} MB ({cpu_mb/1024:.1f} GB)")
        
        # Print by priority
        print(f"\nTensor Placement by Priority:")
        priority_names = {1: "Attention", 2: "FFN", 3: "Embeddings", 4: "Other", 5: "Experts"}
        for priority in [1, 2, 3, 4, 5]:
            size_mb = priority_stats[priority]
            print(f"  Priority {priority} ({priority_names[priority]}): {size_mb:.0f} MB")
        
        print("="*60)

def main():
    """Test the simple capacity optimizer"""
    # Mock data for testing
    cuda_devices = [
        {'device_id': 0, 'free_memory_mb': 31000},
        {'device_id': 1, 'free_memory_mb': 31000},
        {'device_id': 2, 'free_memory_mb': 31000},
        {'device_id': 3, 'free_memory_mb': 31000},
    ]
    
    config = {
        'context_size': 131072,
        'kv_cache': {'type_k': 'q4_0', 'type_v': 'q4_0'}
    }
    
    # Mock tensor analysis
    tensor_analysis = {
        'layer_tensors': {
            'tensors': [
                {'name': 'blk.0.attn_q.weight', 'size_mb': 150},
                {'name': 'blk.0.ffn_gate.weight', 'size_mb': 300},
                # Add more tensors here for realistic testing
            ]
        },
        'expert_tensors': {
            'tensors': [
                {'name': 'blk.0.ffn_gate_exps.weight', 'size_mb': 1050},
            ]
        },
        'embedding_tensors': {
            'tensors': [
                {'name': 'token_embd.weight', 'size_mb': 1500},
            ]
        },
        'other_tensors': {
            'tensors': [
                {'name': 'norm.weight', 'size_mb': 14},
            ]
        }
    }
    
    optimizer = SimpleCapacityOptimizer(cuda_devices, config)
    assignments = optimizer.optimize_simple(tensor_analysis)
    
if __name__ == "__main__":
    main()