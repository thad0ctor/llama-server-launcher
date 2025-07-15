#!/usr/bin/env python3
"""
Simple Parameter Generator for llama.cpp -ot (Override Tensor) Parameters
Generates simple regex patterns for tensor overrides based on optimized assignments.
"""

import json
import re
import os
from typing import List, Dict, Optional
from tensor_optimizer import TensorAssignment, DeviceType

class SimpleParameterGenerator:
    def __init__(self, assignments: List[TensorAssignment], config: Dict):
        self.assignments = assignments
        self.config = config
        
    def generate_override_parameters(self) -> List[str]:
        """Generate llama.cpp -ot regex parameters for tensor overrides"""
        
        # Group assignments by device type and ID
        cpu_assignments = [a for a in self.assignments if a.device_type == DeviceType.CPU]
        gpu_assignments = [a for a in self.assignments if a.device_type == DeviceType.GPU]
        
        parameters = []
        
        # Generate CPU patterns
        if cpu_assignments:
            parameters.extend(self._generate_cpu_patterns(cpu_assignments))
        
        # Generate GPU patterns by device
        gpu_by_device = {}
        for assignment in gpu_assignments:
            device_id = assignment.device_id
            if device_id not in gpu_by_device:
                gpu_by_device[device_id] = []
            gpu_by_device[device_id].append(assignment)
        
        for device_id, device_assignments in gpu_by_device.items():
            parameters.extend(self._generate_gpu_patterns(device_assignments, device_id))
        
        return parameters
    
    def _generate_cpu_patterns(self, assignments: List[TensorAssignment]) -> List[str]:
        """Generate CPU tensor override patterns"""
        patterns = []
        
        # Group by tensor type
        tensor_groups = self._group_by_tensor_type(assignments)
        
        for tensor_type, tensors in tensor_groups.items():
            pattern = self._create_pattern(tensors, tensor_type)
            if pattern:
                patterns.append(f"-ot \"{pattern}=CPU\"")
        
        return patterns
    
    def _generate_gpu_patterns(self, assignments: List[TensorAssignment], device_id: int) -> List[str]:
        """Generate GPU tensor override patterns"""
        patterns = []
        
        # Group by tensor type
        tensor_groups = self._group_by_tensor_type(assignments)
        
        for tensor_type, tensors in tensor_groups.items():
            pattern = self._create_pattern(tensors, tensor_type)
            if pattern:
                patterns.append(f"-ot \"{pattern}=CUDA{device_id}\"")
        
        return patterns
    
    def _group_by_tensor_type(self, assignments: List[TensorAssignment]) -> Dict[str, List[TensorAssignment]]:
        """Group assignments by tensor type"""
        groups = {
            'ffn': [],
            'attn': [],
            'norm': [],
            'embd': [],
            'other': []
        }
        
        for assignment in assignments:
            tensor_name = assignment.tensor_name
            
            if 'ffn' in tensor_name:
                groups['ffn'].append(assignment)
            elif 'attn' in tensor_name:
                groups['attn'].append(assignment)
            elif 'norm' in tensor_name:
                groups['norm'].append(assignment)
            elif 'embd' in tensor_name:
                groups['embd'].append(assignment)
            else:
                groups['other'].append(assignment)
        
        # Remove empty groups
        return {k: v for k, v in groups.items() if v}
    
    def _create_pattern(self, assignments: List[TensorAssignment], tensor_type: str) -> Optional[str]:
        """Create a regex pattern for a group of tensors"""
        if not assignments:
            return None
        
        tensor_names = [a.tensor_name for a in assignments]
        
        if tensor_type == 'ffn':
            return self._create_ffn_pattern(tensor_names)
        elif tensor_type == 'attn':
            return self._create_attn_pattern(tensor_names)
        elif tensor_type == 'norm':
            return self._create_norm_pattern(tensor_names)
        elif tensor_type == 'embd':
            return self._create_embd_pattern(tensor_names)
        else:
            return self._create_other_pattern(tensor_names)
    
    def _create_ffn_pattern(self, tensor_names: List[str]) -> str:
        """Create pattern for FFN tensors"""
        # Extract layer numbers from blk.X.ffn* patterns
        layer_numbers = set()
        for name in tensor_names:
            # Match both blk.X.ffn.* and blk.X.ffn_*_exps.weight patterns
            match = re.search(r'blk\.(\d+)\.ffn', name)
            if match:
                layer_numbers.add(int(match.group(1)))
        
        if not layer_numbers:
            return self._create_exact_pattern(tensor_names)
        
        # Create pattern that matches FFN expert tensors specifically
        # Pattern: blk.X.ffn_gate_exps.weight, blk.X.ffn_up_exps.weight, blk.X.ffn_down_exps.weight
        sorted_layers = sorted(layer_numbers)
        if len(sorted_layers) == 1:
            return f"blk.{sorted_layers[0]}.ffn_.*_exps.weight"
        else:
            layer_list = "|".join(str(n) for n in sorted_layers)
            return f"blk.({layer_list}).ffn_.*_exps.weight"
    
    def _create_attn_pattern(self, tensor_names: List[str]) -> str:
        """Create pattern for attention tensors"""
        # Extract layer numbers from blk.X.attn_* patterns
        layer_numbers = set()
        for name in tensor_names:
            match = re.search(r'blk\.(\d+)\.attn', name)
            if match:
                layer_numbers.add(int(match.group(1)))
        
        if not layer_numbers:
            return self._create_exact_pattern(tensor_names)
        
        # Create pattern that matches attention tensors specifically
        # Pattern: blk.X.attn_q.weight, blk.X.attn_k.weight, blk.X.attn_v.weight, blk.X.attn_output.weight, blk.X.attn_norm.weight
        sorted_layers = sorted(layer_numbers)
        if len(sorted_layers) == 1:
            return f"blk.{sorted_layers[0]}.attn_.*"
        else:
            layer_list = "|".join(str(n) for n in sorted_layers)
            return f"blk.({layer_list}).attn_.*"
    
    def _create_norm_pattern(self, tensor_names: List[str]) -> str:
        """Create pattern for norm tensors"""
        # For norm tensors, try to group by layer
        layer_numbers = set()
        for name in tensor_names:
            match = re.search(r'blk\.(\d+)\..*norm', name)
            if match:
                layer_numbers.add(int(match.group(1)))
        
        if layer_numbers:
            sorted_layers = sorted(layer_numbers)
            if len(sorted_layers) == 1:
                return f"blk.{sorted_layers[0]}.*norm"
            else:
                layer_list = "|".join(str(n) for n in sorted_layers)
                return f"blk.({layer_list}).*norm"
        
        return self._create_exact_pattern(tensor_names)
    
    def _create_embd_pattern(self, tensor_names: List[str]) -> str:
        """Create pattern for embedding tensors"""
        # For embedding tensors like embd_3.weight
        embd_numbers = set()
        for name in tensor_names:
            match = re.search(r'embd_(\d+)\.weight', name)
            if match:
                embd_numbers.add(int(match.group(1)))
        
        if embd_numbers:
            sorted_numbers = sorted(embd_numbers)
            if len(sorted_numbers) == 1:
                return f"embd_{sorted_numbers[0]}.weight"
            else:
                number_list = "|".join(str(n) for n in sorted_numbers)
                return f"embd_({number_list}).weight"
        
        return self._create_exact_pattern(tensor_names)
    
    def _create_other_pattern(self, tensor_names: List[str]) -> str:
        """Create pattern for other tensors"""
        # For other tensors, create an exact match pattern
        return self._create_exact_pattern(tensor_names)
    
    def _create_exact_pattern(self, tensor_names: List[str]) -> str:
        """Create exact match pattern for tensor names"""
        if len(tensor_names) == 1:
            return tensor_names[0]
        elif len(tensor_names) <= 10:
            return f"({'|'.join(tensor_names)})"
        else:
            # For large groups, create a general pattern
            return f"({'|'.join(tensor_names[:10])})"  # First 10 only
    
    def format_llama_cpp_parameters(self) -> List[str]:
        """Format parameters for llama.cpp command line"""
        return self.generate_override_parameters()
    
    def save_parameters(self, output_file: str, model_path: Optional[str] = None, config_name: Optional[str] = None):
        """Save parameters to a file"""
        parameters = self.format_llama_cpp_parameters()
        
        with open(output_file, 'w') as f:
            f.write("# llama.cpp tensor override parameters\n")
            f.write("# Generated by tensor-override optimizer\n")
            
            if model_path is not None:
                import os
                model_name = os.path.basename(model_path)
                f.write(f"# Model: {model_name}\n")
                
            if config_name is not None:
                f.write(f"# Configuration: {config_name}\n")
                
            f.write("\n")
            
            for param in parameters:
                f.write(f"{param}\n")
        
        print(f"Parameters saved to {output_file}")
    
    def get_parameter_summary(self) -> Dict:
        """Get a summary of generated parameters"""
        parameters = self.format_llama_cpp_parameters()
        
        cpu_count = len([p for p in parameters if '=CPU' in p])
        gpu_count = len([p for p in parameters if '=CUDA' in p])
        
        return {
            'total_parameters': len(parameters),
            'cpu_patterns': cpu_count,
            'gpu_patterns': gpu_count,
            'parameters': parameters
        }

def main():
    print("Simple Parameter Generator - use main.py for full analysis")

if __name__ == "__main__":
    main() 