# Tensor Override Optimizer for llama.cpp

This tool analyzes CUDA devices and GGUF models to generate optimized tensor placement parameters for llama.cpp, maximizing GPU utilization while prioritizing expert tensors on CPU for optimal performance.

## Features

- **CUDA Device Detection**: Automatically detects all CUDA devices and available VRAM
- **GGUF Model Analysis**: Analyzes tensor sizes and types from GGUF files
- **Intelligent Tensor Assignment**: 
  - Prioritizes expert tensors on CPU (better for performance)
  - Maximizes GPU usage for layer tensors
  - Optimizes placement based on available VRAM
- **Parameter Generation**: Creates llama.cpp `-ot` regex parameters for tensor overrides
- **Virtual Environment Support**: Configurable virtual environment activation
- **Configurable Settings**: Customizable via JSON configuration

## Quick Start

```bash
# Basic usage
python main.py /path/to/model.gguf

# With custom configuration
python main.py /path/to/model.gguf --config my_config.json

# Dry run (analysis only, no parameter generation)
python main.py /path/to/model.gguf --dry-run

# Verbose output
python main.py /path/to/model.gguf --verbose
```

## Configuration

Edit `config.json` to customize behavior:

```json
{
  "llama_cpp_path": "/home/rgilbreth/Desktop/AI-Software/llama.cpp",
  "virtual_env_path": "/path/to/your/venv",
  "context_size": 8192,
  "tensor_analysis": {
    "prioritize_experts_on_cpu": true,
    "maximize_gpu_layers": true,
    "reserve_context_memory": true,
    "memory_safety_margin_mb": 512
  },
  "output": {
    "save_analysis_json": true,
    "save_override_params": true,
    "verbose": true
  }
}
```

## Algorithm

The optimizer uses the following priority system:

1. **Expert tensors** → CPU (preferred for performance)
2. **Embedding tensors** → GPU (if space available)
3. **Layer tensors** → GPU (maximize GPU utilization)
4. **Other tensors** → GPU if space available, otherwise CPU
5. **Optimization pass** → Move some experts to GPU if surplus capacity exists

## Output

The tool generates:

- **tensor_override_params.txt**: llama.cpp `-ot` parameters ready to use
- **tensor_analysis.json**: Detailed analysis results for review
- **Console output**: Summary of optimization results and usage examples

## Example Usage

After running the optimizer, use the generated parameters with llama.cpp:

```bash
/home/rgilbreth/Desktop/AI-Software/llama.cpp/build/bin/llama-server \
  --model /path/to/model.gguf \
  --ctx-size 8192 \
  -ot "^blk\.0\..*":CPU \
  -ot "^blk\.[1-5]\..*":CUDA0 \
  -ot "^blk\.[6-10]\..*":CUDA1 \
  # ... additional parameters from tensor_override_params.txt
```

## Files

- `main.py`: Main orchestration script
- `cuda_detector.py`: CUDA device detection and VRAM analysis
- `gguf_analyzer.py`: GGUF model tensor analysis
- `tensor_optimizer.py`: Tensor assignment optimization algorithm
- `parameter_generator.py`: llama.cpp parameter generation
- `config.json`: Configuration file
- `README.md`: This documentation

## Requirements

- Python 3.7+
- nvidia-smi (for CUDA device detection)
- llama.cpp built with CUDA support
- GGUF model files

## Virtual Environment Support

To use with a virtual environment, set `virtual_env_path` in config.json:

```json
{
  "virtual_env_path": "/path/to/your/venv"
}
```

The tool will automatically activate the virtual environment before running analysis.

## Performance Tips

- **Expert tensors on CPU**: Reduces memory pressure while maintaining inference speed
- **Layer tensors on GPU**: Maximizes inference performance
- **High GPU utilization**: Indicates optimal memory usage
- **Context size tuning**: Adjust based on your typical usage patterns

## Troubleshooting

1. **No CUDA devices detected**: Ensure nvidia-smi is available and CUDA drivers are installed
2. **GGUF analysis fails**: Check that llama.cpp is built and the path is correct in config.json
3. **Virtual environment issues**: Verify the path and that the venv contains required packages
4. **Memory allocation errors**: Increase safety margin in config or reduce context size

## Example Output

```
======================================================================
Tensor Override Analyzer for llama.cpp
======================================================================
Model: /path/to/Kimi-K2-Instruct-UD-IQ1_S-00001-of-00006.gguf
Config: config.json
Output: /tmp

Step 1: Detecting CUDA devices...
CUDA Device Detection Summary:
==================================================
Device 0: NVIDIA GeForce RTX 5090
  Total VRAM: 32,768 MB
  Free VRAM: 31,064 MB
...

Step 2: Analyzing GGUF model...
GGUF Tensor Analysis Summary:
==================================================
Total tensors: 1096
Total model size: 268,267.1 MB (262.0 GB)
Expert tensors: 183 (45,234.2 MB)
Layer tensors: 732 (198,456.7 MB)
...

Step 3: Optimizing tensor placement...
Assigning 183 expert tensors to CPU...
Assigning layer tensors to GPU (61 layers)...
Moved 23 expert tensors (5,432.1 MB) to GPU for optimization
...

Step 4: Generating llama.cpp parameters...
Generated 47 override parameters
  CPU patterns: 12
  GPU patterns: 35

Example llama.cpp usage:
----------------------------------------
/home/rgilbreth/Desktop/AI-Software/llama.cpp/build/bin/llama-server \
  --model /path/to/model.gguf \
  --ctx-size 8192 \
  -ot "^.*ffn.*exps.*":CPU \
  -ot "^blk\.[0-15]\..*":CUDA0 \
  -ot "^blk\.[16-31]\..*":CUDA1 \
  # ... and 44 more -ot parameters

Full parameter list available in: /tmp/tensor_override_params.txt

Final Summary:
==================================================
Total tensors analyzed: 1096
CPU assignment: 160 tensors (39,802.1 MB)
GPU assignment: 936 tensors (228,465.0 MB)

GPU utilization:
  CUDA0: 57,234.2 MB / 61,064.0 MB (93.7%)
  CUDA1: 56,123.1 MB / 60,805.0 MB (92.3%)
  ...

Overall GPU utilization: 85.2% of model on GPU
```