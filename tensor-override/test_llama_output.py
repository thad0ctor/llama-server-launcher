#!/usr/bin/env python3
"""
Test llama-server output directly
"""

import subprocess
import sys
import os

# Load config
with open('my_config.json', 'r') as f:
    import json
    config = json.load(f)

llama_cpp_path = config.get('llama_cpp_path', '/home/rgilbreth/Desktop/AI-Software/llama.cpp')
server_path = os.path.join(llama_cpp_path, 'build', 'bin', 'llama-server')

if len(sys.argv) < 2:
    print("Usage: python test_llama_output.py <model_path>")
    sys.exit(1)

model_path = sys.argv[1]

# Run with minimal flags first
cmd = [
    server_path,
    "--model", model_path,
    "--ctx-size", "1",
    "--verbose",
    "--n-gpu-layers", "0"  # Force CPU to see all tensor info
]

print(f"Running: {' '.join(cmd)}")
print("=" * 60)

# Run directly without capture to see output
try:
    process = subprocess.Popen(cmd)
    # Let it run for 30 seconds then kill
    import time
    time.sleep(30)
    process.terminate()
    process.wait()
except KeyboardInterrupt:
    print("\nInterrupted by user")
    process.terminate()
    process.wait()