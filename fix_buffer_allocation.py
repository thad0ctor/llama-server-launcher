#!/usr/bin/env python3
"""
Script to regenerate tensor allocation with reduced buffer sizes to fix compute buffer OOM issue.
"""

import sys
import os
import json
from pathlib import Path

def fix_buffer_allocation():
    """Regenerate tensor allocation with smaller GPU buffers."""
    
    # Read current config
    config_path = "llama_cpp_launcher_configs.json"
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update app_settings to reduce GPU buffers
        if 'app_settings' in config:
            print("Current buffer settings:")
            for i in range(4):  # 4 GPUs
                buffer_key = f'gpu_{i}_buffer'
                if buffer_key in config['app_settings']:
                    current_buffer = config['app_settings'][buffer_key]
                    print(f"  GPU {i}: {current_buffer}MB")
                    # Reduce from 512MB to 256MB
                    config['app_settings'][buffer_key] = '256'
                    print(f"  GPU {i}: Updated to 256MB")
                else:
                    # Set default reduced buffer
                    config['app_settings'][buffer_key] = '256'
                    print(f"  GPU {i}: Set to 256MB (new)")
        
        # Save updated config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("\nBuffer settings updated successfully!")
        print("Please run the tensor analysis again to regenerate parameters with the new buffer settings.")
        return True
        
    except Exception as e:
        print(f"Error updating config: {e}")
        return False

if __name__ == "__main__":
    fix_buffer_allocation()