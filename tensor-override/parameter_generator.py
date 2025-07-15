#!/usr/bin/env python3
"""
Parameter Generator for llama.cpp -ot (Override Tensor) Parameters
Generates regex patterns for tensor overrides based on optimized assignments.
"""

import json
import re
import os
from typing import List, Dict, Optional
from tensor_optimizer import TensorAssignment, DeviceType

class ParameterGenerator:
    def __init__(self, assignments: List[TensorAssignment], config: Dict):
        self.assignments = assignments
        self.config = config
        self.cpu_patterns = []
        self.gpu_patterns = {}  # device_id -> patterns
        
    def generate_override_parameters(self) -> Dict[str, List[str]]:
        """Generate llama.cpp -ot regex parameters for tensor overrides"""
        
        # Group assignments by device type and ID
        cpu_assignments = [a for a in self.assignments if a.device_type == DeviceType.CPU]
        gpu_assignments = [a for a in self.assignments if a.device_type == DeviceType.GPU]
        
        # Generate CPU patterns
        self.cpu_patterns = self._generate_cpu_patterns(cpu_assignments)
        
        # Generate GPU patterns by device
        for device_id in set(a.device_id for a in gpu_assignments):
            device_assignments = [a for a in gpu_assignments if a.device_id == device_id]
            self.gpu_patterns[device_id] = self._generate_gpu_patterns(device_assignments, device_id)
        
        return {
            'cpu_patterns': self.cpu_patterns,
            'gpu_patterns': self.gpu_patterns
        }
    
    def _generate_cpu_patterns(self, cpu_assignments: List[TensorAssignment]) -> List[str]:
        """Generate regex patterns for CPU tensor overrides"""
        patterns = []
        
        if not cpu_assignments:
            return patterns
        
        # Group by tensor name patterns to create efficient regex
        tensor_groups = self._group_tensors_by_pattern(cpu_assignments)
        
        for group_name, tensor_names in tensor_groups.items():
            if len(tensor_names) == 1:
                # Single tensor - exact match
                pattern = f"^{re.escape(tensor_names[0])}$"
            else:
                # Multiple tensors - try to create efficient regex
                pattern = self._create_efficient_regex(tensor_names)
            
            patterns.append(pattern)
        
        return patterns
    
    def _generate_gpu_patterns(self, gpu_assignments: List[TensorAssignment], device_id: int) -> List[str]:
        """Generate regex patterns for GPU tensor overrides"""
        patterns = []
        
        if not gpu_assignments:
            return patterns
        
        # Group by tensor name patterns
        tensor_groups = self._group_tensors_by_pattern(gpu_assignments)
        
        for group_name, tensor_names in tensor_groups.items():
            if len(tensor_names) == 1:
                # Single tensor - exact match
                pattern = f"^{re.escape(tensor_names[0])}$"
            else:
                # Multiple tensors - try to create efficient regex
                pattern = self._create_efficient_regex(tensor_names)
            
            patterns.append(pattern)
        
        return patterns
    
    def _group_tensors_by_pattern(self, assignments: List[TensorAssignment]) -> Dict[str, List[str]]:
        """Group tensors by common patterns to optimize regex generation"""
        groups = {}
        
        # Group by common prefixes and patterns
        expert_tensors = []
        layer_tensors = {}
        embedding_tensors = []
        other_tensors = []
        
        for assignment in assignments:
            tensor_name = assignment.tensor_name
            
            if 'expert' in tensor_name or 'ffn_gate_exps' in tensor_name or 'ffn_down_exps' in tensor_name or 'ffn_up_exps' in tensor_name:
                expert_tensors.append(tensor_name)
            elif 'blk.' in tensor_name:
                # Extract layer number
                match = re.search(r'blk\.(\d+)\.', tensor_name)
                if match:
                    layer_num = int(match.group(1))
                    if layer_num not in layer_tensors:
                        layer_tensors[layer_num] = []
                    layer_tensors[layer_num].append(tensor_name)
                else:
                    other_tensors.append(tensor_name)
            elif 'token_embd' in tensor_name or 'output' in tensor_name:
                embedding_tensors.append(tensor_name)
            else:
                other_tensors.append(tensor_name)
        
        # Create groups
        if expert_tensors:
            groups['experts'] = expert_tensors
        
        if embedding_tensors:
            groups['embeddings'] = embedding_tensors
        
        if other_tensors:
            groups['others'] = other_tensors
        
        # Group layer tensors by ranges for efficiency
        if layer_tensors:
            layer_groups = self._group_layers_by_range(layer_tensors)
            groups.update(layer_groups)
        
        return groups
    
    def _group_layers_by_range(self, layer_tensors: Dict[int, List[str]]) -> Dict[str, List[str]]:
        """Group layer tensors by consecutive ranges for efficient regex"""
        groups = {}
        
        # Find consecutive layer ranges
        layer_nums = sorted(layer_tensors.keys())
        ranges = []
        current_range = [layer_nums[0]]
        
        for i in range(1, len(layer_nums)):
            if layer_nums[i] == layer_nums[i-1] + 1:
                current_range.append(layer_nums[i])
            else:
                ranges.append(current_range)
                current_range = [layer_nums[i]]
        ranges.append(current_range)
        
        # Create groups for each range
        for range_layers in ranges:
            if len(range_layers) == 1:
                layer_num = range_layers[0]
                group_name = f"layer_{layer_num}"
                groups[group_name] = layer_tensors[layer_num]
            else:
                start_layer = range_layers[0]
                end_layer = range_layers[-1]
                group_name = f"layers_{start_layer}_{end_layer}"
                
                # Combine all tensors in this range
                all_tensors = []
                for layer_num in range_layers:
                    all_tensors.extend(layer_tensors[layer_num])
                
                groups[group_name] = all_tensors
        
        return groups
    
    def _create_efficient_regex(self, tensor_names: List[str]) -> str:
        """Create an efficient regex pattern for a list of tensor names"""
        if not tensor_names:
            return ""
        
        if len(tensor_names) == 1:
            return tensor_names[0]
        
        # Check for layer patterns first (most common)
        layer_pattern = self._detect_layer_pattern(tensor_names)
        if layer_pattern:
            return layer_pattern
        
        # Check for expert patterns
        expert_pattern = self._detect_expert_pattern(tensor_names)
        if expert_pattern:
            return expert_pattern
        
        # Check for embedding patterns
        embedding_pattern = self._detect_embedding_pattern(tensor_names)
        if embedding_pattern:
            return embedding_pattern
        
        # Fallback to simple alternation for exact matches
        if len(tensor_names) <= 20:  # Reasonable limit
            return f"({'|'.join(tensor_names)})"
        
        # For very large groups, use a more general pattern
        return self._create_general_pattern(tensor_names)
    
    def _detect_layer_pattern(self, tensor_names: List[str]) -> Optional[str]:
        """Detect if tensor names follow a layer pattern"""
        # Extract layer numbers and suffixes
        layer_info = []
        
        for name in tensor_names:
            match = re.search(r'blk\.(\d+)\.(.+)', name)
            if match:
                layer_num = int(match.group(1))
                layer_suffix = match.group(2)
                layer_info.append((layer_num, layer_suffix))
        
        if not layer_info:
            return None
        
        # Group by suffix to see if we have consistent patterns
        suffix_groups = {}
        for layer_num, suffix in layer_info:
            if suffix not in suffix_groups:
                suffix_groups[suffix] = []
            suffix_groups[suffix].append(layer_num)
        
        # If we have multiple different suffixes, create a more general pattern
        if len(suffix_groups) > 1:
            # Find common layer numbers across different suffixes
            all_layers = set()
            for layers in suffix_groups.values():
                all_layers.update(layers)
            
            # If all layers are the same, use a simple pattern
            if len(all_layers) > 1:
                layer_range = "|".join(str(layer) for layer in sorted(all_layers))
                return f"blk.({layer_range})."
        
        # Single suffix pattern
        suffix = list(suffix_groups.keys())[0]
        layers = suffix_groups[suffix]
        
        if len(layers) == 1:
            return f"blk.{layers[0]}.{suffix}"
        
        # Multiple layers with same suffix
        layer_range = "|".join(str(layer) for layer in sorted(layers))
        return f"blk.({layer_range}).{suffix}"
    
    def _detect_embedding_pattern(self, tensor_names: List[str]) -> Optional[str]:
        """Detect if tensor names follow an embedding pattern"""
        # Check for embedding patterns like embd_3.weight, embd_4.weight
        embedding_numbers = []
        for name in tensor_names:
            match = re.search(r'embd_(\d+)\.weight', name)
            if match:
                embedding_numbers.append(int(match.group(1)))
        
        if not embedding_numbers:
            return None
        
        if len(embedding_numbers) == 1:
            return f"embd_{embedding_numbers[0]}.weight"
        
        # Multiple embedding numbers
        sorted_numbers = sorted(embedding_numbers)
        number_range = "|".join(str(n) for n in sorted_numbers)
        return f"embd_({number_range}).weight"
    
    def _create_general_pattern(self, tensor_names: List[str]) -> str:
        """Create a general pattern for a large group of tensors"""
        # For very large groups, create a more general pattern
        # Find common prefixes and patterns
        prefixes = set()
        for name in tensor_names:
            # Extract the first part before the first dot
            parts = name.split('.')
            if len(parts) > 1:
                prefixes.add(parts[0])
        
        if len(prefixes) == 1:
            prefix = list(prefixes)[0]
            return f"{prefix}."
        
        # Multiple prefixes, use alternation
        prefix_list = "|".join(sorted(prefixes))
        return f"({prefix_list})."
                else:
                    # Non-consecutive, list first few
                    layer_list = "|".join(str(n) for n in sorted_layers[:10])
                    return f"^blk\\.({layer_list})\\..*$"
        
        # Single suffix type
        suffix = list(suffix_groups.keys())[0]
        layer_nums = suffix_groups[suffix]
        
        if len(layer_nums) == 1:
            # Single layer, single suffix
            return f"^blk\\.{layer_nums[0]}\\.{re.escape(suffix)}$"
        else:
            # Multiple layers, single suffix
            layer_nums.sort()
            if self._is_consecutive_range(layer_nums):
                start, end = layer_nums[0], layer_nums[-1]
                if end - start <= 10:  # Reasonable range
                    return f"^blk\\.[{start}-{end}]\\.{re.escape(suffix)}$"
                else:
                    # Large range, use alternation
                    layer_list = "|".join(str(n) for n in layer_nums[:10])
                    return f"^blk\\.({layer_list})\\.{re.escape(suffix)}$"
            else:
                # Non-consecutive
                layer_list = "|".join(str(n) for n in layer_nums[:10])
                return f"^blk\\.({layer_list})\\.{re.escape(suffix)}$"
    
    def _detect_expert_pattern(self, tensor_names: List[str]) -> Optional[str]:
        """Detect if tensor names follow an expert pattern"""
        # Check for common expert patterns
        expert_patterns = [
            (r'ffn_gate_exps', r'.*\.ffn_gate_exps\..*'),
            (r'ffn_down_exps', r'.*\.ffn_down_exps\..*'),
            (r'ffn_up_exps', r'.*\.ffn_up_exps\..*'),
            (r'expert', r'.*expert.*')
        ]
        
        for pattern_name, pattern_regex in expert_patterns:
            matching_names = [name for name in tensor_names if re.search(pattern_regex, name)]
            if len(matching_names) > len(tensor_names) * 0.8:  # At least 80% match
                return pattern_regex
        
        # If no single pattern matches well, try to create a combined pattern
        if any('ffn_gate_exps' in name for name in tensor_names):
            return r'.*\.ffn_.*_exps\..*'
        elif any('expert' in name for name in tensor_names):
            return r'.*expert.*'
        
        return None
    
    def _is_consecutive_range(self, numbers: List[int]) -> bool:
        """Check if a list of numbers forms a consecutive range"""
        if len(numbers) < 2:
            return True
        
        for i in range(1, len(numbers)):
            if numbers[i] != numbers[i-1] + 1:
                return False
        
        return True
    
    def format_llama_cpp_parameters(self) -> List[str]:
        """Format parameters for llama.cpp command line"""
        parameters = []
        patterns = self.generate_override_parameters()
        
        # Add CPU override patterns
        for pattern in patterns['cpu_patterns']:
            # Validate regex pattern before adding
            if self._validate_regex(pattern):
                parameters.append(f"-ot {pattern}=CPU")
            else:
                print(f"Warning: Invalid regex pattern skipped: {pattern}")
        
        # Add GPU override patterns
        for device_id, device_patterns in patterns['gpu_patterns'].items():
            for pattern in device_patterns:
                # Validate regex pattern before adding
                if self._validate_regex(pattern):
                    parameters.append(f"-ot {pattern}=CUDA{device_id}")
                else:
                    print(f"Warning: Invalid regex pattern skipped: {pattern}")
        
        return parameters
    
    def _validate_regex(self, pattern: str) -> bool:
        """Validate that a regex pattern compiles correctly"""
        try:
            re.compile(pattern)
            return True
        except re.error as e:
            print(f"Regex compilation error for pattern '{pattern}': {e}")
            return False
    
    def save_parameters(self, output_file: str, model_path: str = None, config_name: str = None):
        """Save parameters to a file"""
        parameters = self.format_llama_cpp_parameters()
        
        with open(output_file, 'w') as f:
            f.write("# llama.cpp tensor override parameters\n")
            f.write("# Generated by tensor-override optimizer\n")
            
            if model_path:
                import os
                model_name = os.path.basename(model_path)
                f.write(f"# Model: {model_name}\n")
                
            if config_name:
                f.write(f"# Configuration: {config_name}\n")
                
            f.write("\n")
            
            for param in parameters:
                f.write(f"{param}\n")
        
        print(f"Parameters saved to {output_file}")
    
    def get_parameter_summary(self) -> Dict:
        """Get a summary of generated parameters"""
        patterns = self.generate_override_parameters()
        parameters = self.format_llama_cpp_parameters()
        
        return {
            'total_parameters': len(parameters),
            'cpu_patterns': len(patterns['cpu_patterns']),
            'gpu_patterns': sum(len(patterns) for patterns in patterns['gpu_patterns'].values()),
            'parameters': parameters,
            'pattern_details': patterns
        }

def main():
    # This is mainly for testing - normally called from the main script
    print("Parameter Generator - use main.py for full analysis")

if __name__ == "__main__":
    main()