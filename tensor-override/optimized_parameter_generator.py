#!/usr/bin/env python3
"""
Optimized Parameter Generator for llama.cpp -ot (Override Tensor) Parameters
Generates clean, efficient regex patterns for tensor overrides.
"""

import json
import re
import os
from typing import List, Dict, Optional, Set
from tensor_optimizer import TensorAssignment, DeviceType

class OptimizedParameterGenerator:
    def __init__(self, assignments: List[TensorAssignment], config: Dict):
        self.assignments = assignments
        self.config = config
        
    def generate_override_parameters(self) -> List[str]:
        """Generate optimized llama.cpp -ot regex parameters for tensor overrides"""
        
        # Group assignments by device type and ID
        cpu_assignments = [a for a in self.assignments if a.device_type == DeviceType.CPU]
        gpu_assignments = [a for a in self.assignments if a.device_type == DeviceType.GPU]
        
        parameters = []
        
        # Generate CPU patterns (more aggressive grouping)
        if cpu_assignments:
            parameters.extend(self._generate_optimized_cpu_patterns(cpu_assignments))
        
        # Generate GPU patterns by device (cleaner patterns)
        gpu_by_device = {}
        for assignment in gpu_assignments:
            device_id = assignment.device_id
            if device_id not in gpu_by_device:
                gpu_by_device[device_id] = []
            gpu_by_device[device_id].append(assignment)
        
        for device_id, device_assignments in gpu_by_device.items():
            parameters.extend(self._generate_optimized_gpu_patterns(device_assignments, device_id))
        
        return parameters
    
    def _generate_optimized_cpu_patterns(self, assignments: List[TensorAssignment]) -> List[str]:
        """Generate optimized CPU tensor override patterns"""
        patterns = []
        
        # Group by tensor type with more intelligent pattern generation
        expert_tensors = [a for a in assignments if 'ffn_' in a.tensor_name and '_exps' in a.tensor_name]
        embedding_tensors = [a for a in assignments if 'embd_' in a.tensor_name]
        norm_tensors = [a for a in assignments if 'norm' in a.tensor_name]
        other_tensors = [a for a in assignments if a not in expert_tensors + embedding_tensors + norm_tensors]
        
        # Expert tensors (use wildcard patterns)
        if expert_tensors:
            patterns.append('-ot "blk\\.(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45|46|47|48|49|50|51|52|53|54|55|56|57|58|59|60)\\.ffn.*_exps\\.weight=CPU"')
        
        # Embedding tensors  
        if embedding_tensors:
            embd_numbers = self._extract_numbers_from_embeddings(embedding_tensors)
            if embd_numbers:
                embd_pattern = "|".join(map(str, sorted(embd_numbers)))
                patterns.append(f'-ot "embd_({embd_pattern})\\.weight=CPU"')
        
        # Norm tensors
        if norm_tensors:
            patterns.append('-ot ".*norm\\.weight=CPU"')
        
        # Other tensors (be specific)
        if other_tensors:
            other_names = [self._escape_tensor_name(a.tensor_name) for a in other_tensors]
            if other_names:
                other_pattern = "|".join(other_names)
                patterns.append(f'-ot "({other_pattern})=CPU"')
        
        return patterns
    
    def _generate_optimized_gpu_patterns(self, assignments: List[TensorAssignment], device_id: int) -> List[str]:
        """Generate optimized GPU tensor override patterns"""
        patterns = []
        
        # Group by tensor type
        ffn_tensors = [a for a in assignments if 'ffn' in a.tensor_name and '_exps' not in a.tensor_name]
        attn_tensors = [a for a in assignments if 'attn' in a.tensor_name]
        embedding_tensors = [a for a in assignments if 'embd_' in a.tensor_name]
        output_tensors = [a for a in assignments if a.tensor_name in ['token_embd.weight', 'output.weight', 'output_norm.weight']]
        norm_tensors = [a for a in assignments if 'norm' in a.tensor_name and a not in output_tensors]
        other_tensors = [a for a in assignments if a not in ffn_tensors + attn_tensors + embedding_tensors + output_tensors + norm_tensors]
        
        # FFN tensors (non-expert)
        if ffn_tensors:
            layer_numbers = self._extract_layer_numbers(ffn_tensors)
            if layer_numbers:
                layer_pattern = "|".join(map(str, sorted(layer_numbers)))
                patterns.append(f'-ot "blk\\.({layer_pattern})\\.ffn.*=CUDA{device_id}"')
        
        # Attention tensors
        if attn_tensors:
            layer_numbers = self._extract_layer_numbers(attn_tensors)
            if layer_numbers:
                layer_pattern = "|".join(map(str, sorted(layer_numbers)))
                patterns.append(f'-ot "blk\\.({layer_pattern})\\.attn.*=CUDA{device_id}"')
        
        # Output/main tensors
        if output_tensors:
            output_names = [a.tensor_name for a in output_tensors]
            output_pattern = "|".join(output_names)
            patterns.append(f'-ot "({output_pattern})=CUDA{device_id}"')
        
        # Norm tensors
        if norm_tensors:
            patterns.append(f'-ot ".*norm\\.weight=CUDA{device_id}"')
        
        # Embedding tensors
        if embedding_tensors:
            embd_numbers = self._extract_numbers_from_embeddings(embedding_tensors)
            if embd_numbers:
                embd_pattern = "|".join(map(str, sorted(embd_numbers)))
                patterns.append(f'-ot "embd_({embd_pattern})\\.weight=CUDA{device_id}"')
        
        # Other tensors
        if other_tensors:
            other_names = [self._escape_tensor_name(a.tensor_name) for a in other_tensors]
            if other_names:
                other_pattern = "|".join(other_names)
                patterns.append(f'-ot "({other_pattern})=CUDA{device_id}"')
        
        return patterns
    
    def _extract_layer_numbers(self, assignments: List[TensorAssignment]) -> Set[int]:
        """Extract layer numbers from tensor assignments"""
        layer_numbers = set()
        for assignment in assignments:
            match = re.search(r'blk\.(\d+)\.', assignment.tensor_name)
            if match:
                layer_numbers.add(int(match.group(1)))
        return layer_numbers
    
    def _extract_numbers_from_embeddings(self, assignments: List[TensorAssignment]) -> Set[int]:
        """Extract numbers from embedding tensor names"""
        numbers = set()
        for assignment in assignments:
            match = re.search(r'embd_(\d+)\.weight', assignment.tensor_name)
            if match:
                numbers.add(int(match.group(1)))
        return numbers
    
    def _escape_tensor_name(self, name: str) -> str:
        """Escape special regex characters in tensor names"""
        # Escape dots and other regex special characters
        return name.replace('.', '\\.')
    
    def save_parameters(self, output_path: str, model_path: str = None):
        """Save the optimized parameters to a file"""
        parameters = self.generate_override_parameters()
        
        with open(output_path, 'w') as f:
            f.write("# llama.cpp tensor override parameters\n")
            f.write("# Generated by optimized tensor-override analyzer\n")
            if model_path:
                f.write(f"# Model: {os.path.basename(model_path)}\n")
            f.write("# Optimized for maximum GPU utilization with balanced load distribution\n")
            f.write("\n")
            
            for param in parameters:
                f.write(param + "\n")
        
        print(f"Optimized tensor override parameters saved to: {output_path}")
        print(f"Generated {len(parameters)} parameter lines")

def main():
    """Test the optimized parameter generator"""
    print("Optimized Parameter Generator - use with tensor assignments from optimizer")

if __name__ == "__main__":
    main()