#!/usr/bin/env python3
"""
Quick test of the tensor override system with estimation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gguf_analyzer import GGUFAnalyzer

def test_estimation():
    """Test the estimation approach directly"""
    analyzer = GGUFAnalyzer('my_config.json')
    
    # Simulate the tensor type summary we already captured
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
    
    print("Testing tensor estimation with captured type summary...")
    tensors = analyzer._estimate_tensors_from_summary(type_summary, {}, "")
    
    if tensors:
        analyzer.tensors = tensors
        analyzer.total_size_mb = sum(t.size_mb for t in tensors)
        analyzer.categorize_tensors()
        analyzer.print_analysis_summary()
        return True
    else:
        print("Failed to generate tensor estimates")
        return False

if __name__ == "__main__":
    success = test_estimation()
    if success:
        print("\nEstimation test passed!")
    else:
        print("\nEstimation test failed!")
        sys.exit(1)