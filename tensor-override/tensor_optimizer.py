#!/usr/bin/env python3
"""
Tensor Assignment Optimizer
Optimizes tensor placement across GPU and CPU for maximum performance.
Priority: Experts on CPU, maximize GPU usage for other layers.
"""

import json
import os
import sys
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class DeviceType(Enum):
    CPU = "CPU"
    GPU = "GPU"

@dataclass
class TensorAssignment:
    tensor_name: str
    device_type: DeviceType
    device_id: Optional[int] = None  # GPU device ID, None for CPU
    size_mb: float = 0.0
    priority: int = 0  # Lower number = higher priority
    
    def __str__(self):
        if self.device_type == DeviceType.CPU:
            return f"{self.tensor_name} -> CPU"
        else:
            return f"{self.tensor_name} -> CUDA{self.device_id}"

class TensorOptimizer:
    def __init__(self, cuda_devices: List[Dict], tensor_analysis: Dict, config: Dict):
        # Validate inputs
        if not cuda_devices:
            raise ValueError("No CUDA devices provided")
        
        if not tensor_analysis:
            raise ValueError("No tensor analysis data provided")
        
        if not config:
            raise ValueError("No configuration provided")
        
        # Validate required tensor analysis structure
        required_keys = ['expert_tensors', 'layer_tensors', 'embedding_tensors', 'other_tensors']
        for key in required_keys:
            if key not in tensor_analysis:
                raise ValueError(f"Missing required tensor analysis key: {key}")
        
        self.cuda_devices = cuda_devices
        self.tensor_analysis = tensor_analysis
        self.config = config
        
        # Initialize device capacities with proper context memory distribution
        self.gpu_capacities = {}
        self.gpu_used = {}
        
        # Calculate per-GPU context headroom
        self._calculate_per_gpu_context_headroom()
        
        print(f"GPU capacities after context headroom:")
        for device_id, capacity in self.gpu_capacities.items():
            print(f"  CUDA{device_id}: {capacity:.0f} MB available for tensors")
        
        self.assignments: List[TensorAssignment] = []
        self.cpu_assignments: List[TensorAssignment] = []
        self.gpu_assignments: List[TensorAssignment] = []
    
    def _calculate_per_gpu_context_headroom(self):
        """Calculate GPU capacities with proper KV cache reservation"""
        if not self.cuda_devices:
            return
            
        # Get context parameters
        context_size = self.config.get('context_size', 8192)
        kv_cache_config = self.config.get("kv_cache", {})
        kv_cache_type = kv_cache_config.get("type", "f16")
        kv_cache_type_k = kv_cache_config.get("type_k", kv_cache_type)
        kv_cache_type_v = kv_cache_config.get("type_v", kv_cache_type)
        
        # Calculate bytes per element
        type_multipliers = {
            'f32': 4.0, 'f16': 2.0, 'bf16': 2.0,
            'q8_0': 1.0, 'q4_0': 0.5, 'q4_1': 0.5,
            'q5_0': 0.625, 'q5_1': 0.625, 'iq4_nl': 0.5
        }
        
        k_multiplier = type_multipliers.get(kv_cache_type_k, 2.0)
        v_multiplier = type_multipliers.get(kv_cache_type_v, 2.0)
        
        # Model parameters
        embd_dim = 7168
        n_layers = 61
        per_layer_kv_cache_mb = (context_size * embd_dim * (k_multiplier + v_multiplier)) / (1024 * 1024)
        
        # Estimate layers per GPU (more conservative)
        estimated_layers_per_gpu = max(1, n_layers // len(self.cuda_devices))
        
        safety_margin = self.config.get('tensor_analysis', {}).get('memory_safety_margin_mb', 512)
        
        print(f"GPU capacity calculation with KV cache:")
        print(f"  - Context size: {context_size:,} tokens")
        print(f"  - KV cache types: K={kv_cache_type_k}, V={kv_cache_type_v}")
        print(f"  - Per-layer KV cache: {per_layer_kv_cache_mb:.1f} MB")
        print(f"  - Estimated layers per GPU: {estimated_layers_per_gpu}")
        print(f"  - Safety margin: {safety_margin} MB")
        
        for device in self.cuda_devices:
            device_id = device['device_id']
            raw_capacity = device['free_memory_mb']
            
            # Reserve space for KV cache
            kv_cache_reservation = per_layer_kv_cache_mb * estimated_layers_per_gpu
            total_reservation = kv_cache_reservation + safety_margin
            available_mb = raw_capacity - total_reservation
            
            if available_mb < 0:
                available_mb = raw_capacity * 0.6  # Use 60% for tensors if insufficient
                total_reservation = raw_capacity - available_mb
            
            self.gpu_capacities[device_id] = available_mb
            self.gpu_used[device_id] = 0
            
            print(f"  CUDA{device_id}: {raw_capacity:.0f} MB total, {total_reservation:.0f} MB reserved (KV+safety), {available_mb:.0f} MB for tensors")
    
    def _calculate_final_kv_cache_allocation(self):
        """Calculate actual KV cache allocation based on final layer placement"""
        # Get context parameters from config
        context_size = self.config.get('context_size', 8192)
        kv_cache_config = self.config.get("kv_cache", {})
        kv_cache_type = kv_cache_config.get("type", "f16")
        kv_cache_type_k = kv_cache_config.get("type_k", kv_cache_type)
        kv_cache_type_v = kv_cache_config.get("type_v", kv_cache_type)
        
        # Calculate bytes per element based on quantization
        type_multipliers = {
            'f32': 4.0, 'f16': 2.0, 'bf16': 2.0,
            'q8_0': 1.0, 'q4_0': 0.5, 'q4_1': 0.5,
            'q5_0': 0.625, 'q5_1': 0.625, 'iq4_nl': 0.5
        }
        
        k_multiplier = type_multipliers.get(kv_cache_type_k, 2.0)
        v_multiplier = type_multipliers.get(kv_cache_type_v, 2.0)
        
        # Model parameters
        embd_dim = 7168
        per_layer_kv_cache_mb = (context_size * embd_dim * (k_multiplier + v_multiplier)) / (1024 * 1024)
        
        # Count actual layers assigned to each GPU
        gpu_layer_counts = {device['device_id']: set() for device in self.cuda_devices}
        
        # Count unique layers assigned to each GPU
        for assignment in self.gpu_assignments:
            if assignment.device_type == DeviceType.GPU and assignment.priority == 3:
                # Extract layer number from tensor name
                tensor_name = assignment.tensor_name
                if 'blk.' in tensor_name:
                    try:
                        layer_num = int(tensor_name.split('blk.')[1].split('.')[0])
                        gpu_layer_counts[assignment.device_id].add(layer_num)
                    except (ValueError, IndexError):
                        pass
        
        # Convert sets to counts
        gpu_layer_counts = {device_id: len(layers) for device_id, layers in gpu_layer_counts.items()}
        
        # Calculate KV cache memory per GPU
        total_kv_cache_mb = 0
        print(f"\nFinal KV cache allocation:")
        print(f"  - Context size: {context_size:,} tokens")
        print(f"  - KV cache types: K={kv_cache_type_k}, V={kv_cache_type_v}")
        print(f"  - Per-layer KV cache: {per_layer_kv_cache_mb:.1f} MB")
        
        for device_id, layer_count in gpu_layer_counts.items():
            if layer_count > 0:
                kv_cache_mb = per_layer_kv_cache_mb * layer_count
                total_kv_cache_mb += kv_cache_mb
                print(f"  - CUDA{device_id}: {layer_count} layers = {kv_cache_mb:.1f} MB KV cache")
        
        print(f"  - Total KV cache: {total_kv_cache_mb:.1f} MB")
        
        return gpu_layer_counts, total_kv_cache_mb
        
    def optimize_tensor_placement(self) -> List[TensorAssignment]:
        """
        Optimize tensor placement with the following priority:
        1. Experts -> CPU (preferred)
        2. Embeddings -> GPU (if space available)
        3. Layer tensors -> GPU (maximize usage)
        4. Other tensors -> GPU if space, otherwise CPU
        5. If GPU space remains, move some experts to GPU
        """
        
        print("Optimizing tensor placement...")
        print(f"Available GPU memory: {sum(self.gpu_capacities.values()):.1f} MB across {len(self.cuda_devices)} devices")
        
        # Step 1: Assign experts to CPU (preferred)
        self._assign_experts_to_cpu()
        
        # Step 2: Assign embeddings to GPU (high priority)
        self._assign_embeddings_to_gpu()
        
        # Step 3: Assign layer tensors to GPU (maximize usage)
        self._assign_layers_to_gpu()
        
        # Step 4: Assign remaining tensors
        self._assign_remaining_tensors()
        
        # Step 5: Optimize expert placement if GPU space available
        self._optimize_expert_placement()
        
        # Step 6: Calculate final KV cache allocation based on actual layer placement
        self._calculate_final_kv_cache_allocation()
        
        # Combine all assignments
        self.assignments = self.cpu_assignments + self.gpu_assignments
        
        # Print summary
        self._print_assignment_summary()
        
        return self.assignments
    
    def _assign_experts_to_cpu(self):
        """Assign expert tensors to CPU (preferred for performance)"""
        expert_tensors = self.tensor_analysis.get('expert_tensors', {}).get('tensors', [])
        
        if not expert_tensors:
            print("No expert tensors found to assign")
            return
        
        print(f"Assigning {len(expert_tensors)} expert tensors to CPU...")
        
        for tensor in expert_tensors:
            # Validate tensor structure
            if not isinstance(tensor, dict) or 'name' not in tensor or 'size_mb' not in tensor:
                print(f"Warning: Invalid tensor data: {tensor}")
                continue
                
            try:
                assignment = TensorAssignment(
                    tensor_name=tensor['name'],
                    device_type=DeviceType.CPU,
                    size_mb=float(tensor['size_mb']),
                    priority=1  # High priority for CPU
                )
                self.cpu_assignments.append(assignment)
            except (ValueError, TypeError) as e:
                print(f"Warning: Failed to create assignment for tensor {tensor.get('name', 'unknown')}: {e}")
                continue
        
        cpu_expert_size = sum(t['size_mb'] for t in expert_tensors if isinstance(t, dict) and 'size_mb' in t)
        print(f"  Assigned {cpu_expert_size:.1f} MB of expert tensors to CPU")
    
    def _assign_embeddings_to_gpu(self):
        """Assign embedding tensors to GPU (high priority)"""
        embedding_tensors = self.tensor_analysis.get('embedding_tensors', {}).get('tensors', [])
        
        print(f"Assigning {len(embedding_tensors)} embedding tensors to GPU...")
        
        for tensor in embedding_tensors:
            device_id = self._find_best_gpu_device(tensor['size_mb'])
            if device_id is not None:
                assignment = TensorAssignment(
                    tensor_name=tensor['name'],
                    device_type=DeviceType.GPU,
                    device_id=device_id,
                    size_mb=tensor['size_mb'],
                    priority=2
                )
                self.gpu_assignments.append(assignment)
                self.gpu_used[device_id] += tensor['size_mb']
            else:
                # Fallback to CPU if no GPU space
                assignment = TensorAssignment(
                    tensor_name=tensor['name'],
                    device_type=DeviceType.CPU,
                    size_mb=tensor['size_mb'],
                    priority=2
                )
                self.cpu_assignments.append(assignment)
        
        gpu_embedding_size = sum(a.size_mb for a in self.gpu_assignments if a.priority == 2)
        print(f"  Assigned {gpu_embedding_size:.1f} MB of embedding tensors to GPU")
    
    def _assign_layers_to_gpu(self):
        """Assign layer tensors to GPU to maximize usage"""
        layer_breakdown = self.tensor_analysis.get('layer_tensors', {}).get('layers', {})
        
        # Sort layers by speed impact priority (attention layers first, then by layer number)
        def layer_priority(layer_item):
            layer_id, layer_info = layer_item
            layer_num = int(layer_id)
            # Count attention vs FFN tensors to prioritize layers with attention
            attention_count = sum(1 for t in layer_info['tensors'] if 'attn' in t['name'])
            return (0 if attention_count > 0 else 1, layer_num)
        
        sorted_layers = sorted(layer_breakdown.items(), key=layer_priority)
        
        print(f"Assigning layer tensors to GPU ({len(sorted_layers)} layers) prioritizing speed impact...")
        
        assigned_layers = 0
        total_layer_size = 0
        
        # First pass: try to assign complete layers (starting with attention-heavy layers)
        for layer_id, layer_info in sorted_layers:
            layer_size = layer_info['total_size_mb']
            
            # Try to assign entire layer to a single GPU device
            device_id = self._find_best_gpu_device(layer_size)
            if device_id is not None:
                # Assign all tensors in this layer to the same GPU
                for tensor in layer_info['tensors']:
                    assignment = TensorAssignment(
                        tensor_name=tensor['name'],
                        device_type=DeviceType.GPU,
                        device_id=device_id,
                        size_mb=tensor['size_mb'],
                        priority=3
                    )
                    self.gpu_assignments.append(assignment)
                
                self.gpu_used[device_id] += layer_size
                assigned_layers += 1
                total_layer_size += layer_size
        
        # Second pass: assign individual tensors from unassigned layers, prioritizing attention tensors
        for layer_id, layer_info in sorted_layers:
            # Check if this layer was already assigned in first pass
            layer_tensors = [t['name'] for t in layer_info['tensors']]
            assigned_tensors = {a.tensor_name for a in self.gpu_assignments}
            layer_already_assigned = any(name in assigned_tensors for name in layer_tensors)
            
            if not layer_already_assigned:
                # Prioritize tensor types within the layer: attention > FFN (non-expert) > others
                attention_tensors = [t for t in layer_info['tensors'] if 'attn' in t['name']]
                ffn_tensors = [t for t in layer_info['tensors'] if 'ffn' in t['name'] and 'exps' not in t['name']]
                other_tensors = [t for t in layer_info['tensors'] if t not in attention_tensors and t not in ffn_tensors]
                
                # Try to assign in priority order
                for tensor_group in [attention_tensors, ffn_tensors, other_tensors]:
                    for tensor in tensor_group:
                        device_id = self._find_best_gpu_device(tensor['size_mb'])
                        if device_id is not None:
                            assignment = TensorAssignment(
                                tensor_name=tensor['name'],
                                device_type=DeviceType.GPU,
                                device_id=device_id,
                                size_mb=tensor['size_mb'],
                                priority=3
                            )
                            self.gpu_assignments.append(assignment)
                            self.gpu_used[device_id] += tensor['size_mb']
                            total_layer_size += tensor['size_mb']
                        else:
                            # Fallback to CPU
                            assignment = TensorAssignment(
                                tensor_name=tensor['name'],
                                device_type=DeviceType.CPU,
                                size_mb=tensor['size_mb'],
                                priority=3
                            )
                            self.cpu_assignments.append(assignment)
        
        print(f"  Assigned {assigned_layers} complete layers and {total_layer_size:.1f} MB to GPU")
    
    def _assign_remaining_tensors(self):
        """Assign remaining tensors to GPU if space available, otherwise CPU"""
        other_tensors = self.tensor_analysis.get('other_tensors', {}).get('tensors', [])
        
        print(f"Assigning {len(other_tensors)} other tensors...")
        
        gpu_assigned = 0
        gpu_size = 0
        
        for tensor in other_tensors:
            device_id = self._find_best_gpu_device(tensor['size_mb'])
            if device_id is not None:
                assignment = TensorAssignment(
                    tensor_name=tensor['name'],
                    device_type=DeviceType.GPU,
                    device_id=device_id,
                    size_mb=tensor['size_mb'],
                    priority=4
                )
                self.gpu_assignments.append(assignment)
                self.gpu_used[device_id] += tensor['size_mb']
                gpu_assigned += 1
                gpu_size += tensor['size_mb']
            else:
                assignment = TensorAssignment(
                    tensor_name=tensor['name'],
                    device_type=DeviceType.CPU,
                    size_mb=tensor['size_mb'],
                    priority=4
                )
                self.cpu_assignments.append(assignment)
        
        print(f"  Assigned {gpu_assigned} other tensors ({gpu_size:.1f} MB) to GPU")
    
    def _optimize_expert_placement(self):
        """Move some expert tensors to GPU if there's surplus capacity"""
        if not self.config.get('tensor_analysis', {}).get('maximize_gpu_layers', True):
            return
        
        # First, do comprehensive excess utilization to maximize GPU usage
        self._maximize_gpu_utilization()
    
    def _maximize_gpu_utilization(self):
        """Maximize GPU utilization by moving CPU tensors to GPU excess capacity"""
        print("Maximizing GPU utilization across all devices...")
        
        # Calculate remaining GPU capacity
        remaining_capacity = {
            device_id: capacity - self.gpu_used[device_id] 
            for device_id, capacity in self.gpu_capacities.items()
        }
        
        total_remaining = sum(remaining_capacity.values())
        print(f"Total remaining GPU capacity: {total_remaining:.1f} MB")
        
        if total_remaining < 100:  # If less than 100MB remaining, skip
            print("Insufficient remaining capacity for optimization")
            return
        
        # Check if we should use aggressive GPU utilization
        aggressive_mode = self.config.get('tensor_analysis', {}).get('aggressive_gpu_utilization', True)  # Default to True
        expert_gpu_threshold = self.config.get('tensor_analysis', {}).get('expert_gpu_threshold', 0.5)  # Reduced from 0.7
        
        if aggressive_mode:
            print("Using aggressive GPU utilization mode")
        
        # Priority order for moving to GPU (enhanced):
        # 1. Large expert tensors (highest performance impact)
        # 2. Medium expert tensors 
        # 3. Large layer tensors
        # 4. Other high-impact tensors
        # 5. Small tensors for space efficiency
        
        # Find all CPU tensors categorized by priority and size
        cpu_tensors = []
        
        # Priority 1: Large expert tensors (>500MB)
        for assignment in self.cpu_assignments:
            if assignment.priority == 1 and assignment.size_mb > 500:  # Large expert tensors
                cpu_tensors.append((assignment, 1, assignment.size_mb))
        
        # Priority 2: Medium expert tensors (100-500MB)
        for assignment in self.cpu_assignments:
            if assignment.priority == 1 and 100 <= assignment.size_mb <= 500:  # Medium expert tensors
                cpu_tensors.append((assignment, 2, assignment.size_mb))
        
        # Priority 3: Large layer tensors (>100MB)
        for assignment in self.cpu_assignments:
            if assignment.priority == 3 and assignment.size_mb > 100:  # Large layer tensors
                cpu_tensors.append((assignment, 3, assignment.size_mb))
        
        # Priority 4: Embedding and output tensors (high impact)
        for assignment in self.cpu_assignments:
            if assignment.priority == 2:  # Embedding tensors
                cpu_tensors.append((assignment, 4, assignment.size_mb))
        
        # Priority 5: Other tensors
        for assignment in self.cpu_assignments:
            if assignment.priority == 4:  # Other tensors
                cpu_tensors.append((assignment, 5, assignment.size_mb))
        
        # Priority 6: Small expert tensors (if aggressive mode)
        if aggressive_mode:
            for assignment in self.cpu_assignments:
                if assignment.priority == 1 and assignment.size_mb < 100:  # Small expert tensors
                    cpu_tensors.append((assignment, 6, assignment.size_mb))
        
        # Priority 7: Small layer tensors (fill remaining space)
        for assignment in self.cpu_assignments:
            if assignment.priority == 3 and assignment.size_mb <= 100:  # Small layer tensors
                cpu_tensors.append((assignment, 7, assignment.size_mb))
        
        # Sort by priority (ascending) then by size (descending for better packing)
        cpu_tensors.sort(key=lambda x: (x[1], -x[2]))
        
        moved_tensors = 0
        moved_size = 0
        tensors_to_move = []
        
        # Enhanced allocation strategy
        for assignment, priority_level, size in cpu_tensors:
            # For expert tensors, check if we have enough GPU utilization
            if priority_level <= 2:  # Expert tensors
                total_gpu_capacity = sum(self.gpu_capacities.values())
                total_gpu_used = sum(self.gpu_used.values())
                gpu_utilization = total_gpu_used / total_gpu_capacity if total_gpu_capacity > 0 else 0
                
                # Only move expert tensors if we have good GPU utilization or aggressive mode
                if gpu_utilization < expert_gpu_threshold and not aggressive_mode:
                    continue
            
            # Try to find the best device (prefer device with tight fit)
            device_id = self._find_best_gpu_device(assignment.size_mb)
            if device_id is not None:
                tensors_to_move.append((assignment, device_id))
                moved_tensors += 1
                moved_size += assignment.size_mb
                # Reserve the memory so subsequent tensors see reduced capacity
                self.gpu_used[device_id] += assignment.size_mb
            
            # If we're in aggressive mode, try to split large tensors across devices
            elif aggressive_mode and assignment.size_mb > 1000:
                # For very large tensors, try to see if we can use multiple devices
                # This is a simplification - in reality, tensors can't be split
                # But we can try smaller devices if available
                for device_id in sorted(self.gpu_capacities.keys()):
                    remaining = self.gpu_capacities[device_id] - self.gpu_used[device_id]
                    if remaining >= assignment.size_mb * 0.8:  # 80% fit threshold
                        tensors_to_move.append((assignment, device_id))
                        moved_tensors += 1
                        moved_size += assignment.size_mb
                        self.gpu_used[device_id] += assignment.size_mb
                        break
        
        # Now actually move the tensors (avoiding iteration modification)
        for assignment, device_id in tensors_to_move:
            # Move from CPU to GPU
            self.cpu_assignments.remove(assignment)
            
            gpu_assignment = TensorAssignment(
                tensor_name=assignment.tensor_name,
                device_type=DeviceType.GPU,
                device_id=device_id,
                size_mb=assignment.size_mb,
                priority=assignment.priority
            )
            self.gpu_assignments.append(gpu_assignment)
        
        if moved_tensors > 0:
            print(f"  Moved {moved_tensors} tensors ({moved_size:.1f} MB) from CPU to GPU")
            
            # Show final utilization
            print("Final GPU utilization (85% max to prevent OOM):")
            total_capacity = 0
            total_used = 0
            for device_id, capacity in self.gpu_capacities.items():
                used = self.gpu_used[device_id]
                utilization = (used / capacity) * 100 if capacity > 0 else 0
                print(f"    CUDA{device_id}: {used:.1f} MB / {capacity:.1f} MB ({utilization:.1f}%)")
                total_capacity += capacity
                total_used += used
            
            overall_utilization = (total_used / total_capacity) * 100 if total_capacity > 0 else 0
            print(f"    Overall: {overall_utilization:.1f}% GPU utilization")
        else:
            print("  No tensors could be moved to GPU")
        
        # If we still have significant remaining capacity, warn the user
        final_remaining = sum(self.gpu_capacities[device_id] - self.gpu_used[device_id] 
                             for device_id in self.gpu_capacities.keys())
        if final_remaining > 5000:  # More than 5GB remaining
            print(f"WARNING: {final_remaining:.1f} MB of GPU memory remains unused")
            print("Consider:")
            print("- Reducing context size further")
            print("- Enabling aggressive_gpu_utilization in config")
            print("- Using a smaller safety margin")
    
    def _find_best_gpu_device(self, size_mb: float) -> Optional[int]:
        """Find the best GPU device for a tensor of given size with better load balancing"""
        best_device = None
        best_score = float('inf')
        
        # Maximum fill percentage to prevent OOM (85% to be safe)
        max_fill_percent = 0.85
        
        # Calculate utilization for each device and prefer least utilized
        for device_id, capacity in self.gpu_capacities.items():
            used = self.gpu_used[device_id]
            remaining = capacity - used
            
            if remaining >= size_mb:
                # Calculate utilization percentage after adding this tensor
                new_used = used + size_mb
                new_utilization = new_used / capacity if capacity > 0 else 1.0
                
                # Skip if would exceed max fill percentage
                if new_utilization > max_fill_percent:
                    continue
                
                # Prefer devices with lower utilization for better load balancing
                # Add small penalty for remaining space to still prefer tighter fits
                score = new_utilization + (remaining / (capacity * 10))  # Small remaining space penalty
                
                if score < best_score:
                    best_device = device_id
                    best_score = score
        
        return best_device
    
    def _find_device_with_most_space(self) -> Optional[int]:
        """Find the GPU device with the most available space"""
        best_device = None
        best_remaining = 0
        
        for device_id, capacity in self.gpu_capacities.items():
            used = self.gpu_used[device_id]
            remaining = capacity - used
            
            if remaining > best_remaining:
                best_device = device_id
                best_remaining = remaining
        
        return best_device
    
    def _print_assignment_summary(self):
        """Print a summary of tensor assignments"""
        print(f"\nTensor Assignment Summary:")
        print("=" * 60)
        
        # CPU assignments
        cpu_count = len(self.cpu_assignments)
        cpu_size = sum(a.size_mb for a in self.cpu_assignments)
        print(f"CPU: {cpu_count} tensors, {cpu_size:.1f} MB")
        
        # GPU assignments by device
        for device_id in sorted(self.gpu_capacities.keys()):
            gpu_tensors = [a for a in self.gpu_assignments if a.device_id == device_id]
            gpu_count = len(gpu_tensors)
            gpu_size = sum(a.size_mb for a in gpu_tensors)
            capacity = self.gpu_capacities[device_id]
            utilization = (gpu_size / capacity * 100) if capacity > 0 else 0
            
            print(f"CUDA{device_id}: {gpu_count} tensors, {gpu_size:.1f} MB / {capacity:.1f} MB ({utilization:.1f}%)")
        
        # Overall statistics
        total_gpu_size = sum(a.size_mb for a in self.gpu_assignments)
        total_size = cpu_size + total_gpu_size
        gpu_percentage = (total_gpu_size / total_size * 100) if total_size > 0 else 0
        
        print(f"\nOverall: {total_size:.1f} MB total, {gpu_percentage:.1f}% on GPU")
        
        # Expert placement summary
        cpu_experts = [a for a in self.cpu_assignments if a.priority == 1]
        gpu_experts = [a for a in self.gpu_assignments if a.priority == 1]
        
        print(f"Expert tensors: {len(cpu_experts)} on CPU, {len(gpu_experts)} on GPU")
    
    def get_assignment_data(self) -> Dict:
        """Get structured assignment data for export"""
        return {
            'total_tensors': len(self.assignments),
            'cpu_assignments': [
                {
                    'tensor_name': a.tensor_name,
                    'device_type': a.device_type.value,
                    'size_mb': a.size_mb,
                    'priority': a.priority
                } for a in self.cpu_assignments
            ],
            'gpu_assignments': [
                {
                    'tensor_name': a.tensor_name,
                    'device_type': a.device_type.value,
                    'device_id': a.device_id,
                    'size_mb': a.size_mb,
                    'priority': a.priority
                } for a in self.gpu_assignments
            ],
            'summary': {
                'cpu_tensor_count': len(self.cpu_assignments),
                'cpu_total_size_mb': sum(a.size_mb for a in self.cpu_assignments),
                'gpu_tensor_count': len(self.gpu_assignments),
                'gpu_total_size_mb': sum(a.size_mb for a in self.gpu_assignments),
                'gpu_utilization': {
                    device_id: {
                        'used_mb': self.gpu_used[device_id],
                        'capacity_mb': self.gpu_capacities[device_id],
                        'utilization_percent': (self.gpu_used[device_id] / self.gpu_capacities[device_id] * 100) if self.gpu_capacities[device_id] > 0 else 0
                    } for device_id in self.gpu_capacities.keys()
                }
            }
        }

def main():
    # This is mainly for testing - normally called from the main script
    print("Tensor Optimizer - use main.py for full analysis")

if __name__ == "__main__":
    main()