#!/usr/bin/env python3
"""
Debug script to verify tensor counting
"""

import subprocess
import sys
import os
import re

def analyze_tensor_distribution():
    """Analyze what tensors we can actually extract from llama-server output"""
    
    # Simulate the type summary we captured
    type_summary = {
        'f32': 365,
        'q8_0': 122,
        'q4_K': 56,
        'q5_K': 35,
        'q6_K': 18,
        'iq2_xxs': 24,
        'iq3_xxs': 49,
        'iq1_s': 82,
        'iq3_s': 158,
        'iq2_s': 34,
        'iq4_xs': 139,
        'iq1_m': 14
    }
    
    total_from_summary = sum(type_summary.values())
    print(f"Total tensors from type summary: {total_from_summary}")
    print(f"Expected total: 1,096")
    print(f"Match: {'✅' if total_from_summary == 1096 else '❌'}")
    
    print("\nTensor type distribution:")
    for tensor_type, count in type_summary.items():
        print(f"  {tensor_type}: {count} tensors")
    
    # The problem: we're only counting tensors that appear in verbose output
    # with actual size information, but that's only a subset
    
    print("\nKey insight:")
    print("- Type summary gives us ALL 1,096 tensors")
    print("- Verbose output with sizes only shows tensors during loading")
    print("- We need to use the type summary + architecture knowledge")
    print("- Not rely on individual tensor size messages")
    
    return type_summary

def estimate_completeness():
    """Estimate how complete our tensor size detection is"""
    
    # DeepSeek/Kimi model has 61 layers (0-60)
    # Each layer should have these tensors:
    layer_tensors_per_layer = [
        'attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'attn_output.weight',
        'ffn_gate.weight', 'ffn_up.weight', 'ffn_down.weight',
        'attn_norm.weight', 'ffn_norm.weight',
        'ffn_gate_exps.weight', 'ffn_down_exps.weight', 'ffn_up_exps.weight'  # Expert tensors
    ]
    
    expected_layer_tensors = 61 * len(layer_tensors_per_layer)
    
    # Plus embedding/output tensors
    embedding_tensors = ['token_embd.weight', 'output.weight', 'norm.weight', 'output_norm.weight']
    
    total_expected = expected_layer_tensors + len(embedding_tensors)
    
    print(f"\nArchitecture-based estimate:")
    print(f"Expected layer tensors: {expected_layer_tensors} (61 layers × {len(layer_tensors_per_layer)} tensors)")
    print(f"Expected embedding tensors: {len(embedding_tensors)}")
    print(f"Total expected: {total_expected}")
    print(f"Actual from GGUF: 1,096")
    print(f"Difference: {1096 - total_expected} (likely due to additional norm/bias tensors)")

if __name__ == "__main__":
    analyze_tensor_distribution()
    estimate_completeness()