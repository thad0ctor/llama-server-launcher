# Llama.cpp Server Launcher

![Main Menu](images/main.png)

**A user-friendly GUI (Tkinter) to easily configure and launch the `llama.cpp` server, manage model configurations, set environment variables, and generate launch scripts.**

This python script provides a comprehensive graphical interface for `llama.cpp`'s server, simplifying the managing of command-line arguments and models.

## ✨ Key Features

*   **Intuitive GUI:** Easy-to-use Tkinter interface with tabbed sections for:
    *   Main Settings (paths, model selection, basic parameters)
    *   Advanced Settings (GPU, memory, cache, performance, generation)
    *   Chat Templates (select predefined, use model default, or provide custom)
    *   Environment Variables (manage CUDA and custom variables)
    *   Configurations (save/load/import/export launch setups)

<details>
<summary><h3>📸 View Advanced Settings Screenshot</h3></summary>

![Parameter Customization](images/advanced.png)

</details>

*   **Comprehensive Parameter Control:** Fine-tune your `llama.cpp` server:
    *   **Model Management:** Scan directories for GGUF models, automatic model analysis (layers, architecture, size) with fallbacks, manual model info entry.
    *   **Core Parameters:** Threads (main & batch), context size, batch sizes (prompt & ubatch), sampling (temperature, min_p, seed).
    *   **GPU Offloading:** GPU layers, tensor split (with VRAM-based recommendations), main GPU selection, Flash Attention toggle.
    *   **Memory & Cache:** KV cache types (K & V), mmap, mlock, no KV offload.
    *   **Network:** Host IP and port configuration.
    *   **Generation:** Ignore EOS, n_predict (max tokens).
    *   **Custom Arguments:** Pass any additional `llama.cpp` server parameters.
*   **System & GPU Insights:**
    *   Detects and displays CUDA GPU(s) (via PyTorch), system RAM, and CPU core information.
    *   Supports manual GPU configuration if automatic detection is unavailable.

<details>
<summary><h3>📸 View Chat Templates Screenshot</h3></summary>

![Chat Templates](images/chat-templates.png)

</details>


*   **Chat Template Flexibility:**
    *   Load predefined chat templates from `chat_templates.json`.
    *   Option to let `llama.cpp` decide the template based on model metadata.
    *   Provide your own custom Jinja2 template string.

<details>
<summary><h3>📸 View Environment Variables Screenshot</h3></summary>

![CUDA Flags](images/env.png)

</details>

*   **Environment Variable Management:**
    *   Easily enable/disable common CUDA environment variables (e.g., `GGML_CUDA_FORCE_MMQ`).
    *   Add and manage custom environment variables to fine tune CUDA performance.

<details>
<summary><h3>📸 View Configuration Management Screenshot</h3></summary>

![Configs](images/configs.png)

</details>

*   **Configuration Hub:**
    *   Save, load, and delete named launch configurations.
    *   Import and export configurations to JSON for sharing or backup.
    *   Application settings (last used paths, UI preferences) are remembered.
*   **Script Generation:**
    *   Generate ready-to-use PowerShell (`.ps1`) and Bash (`.sh`) scripts from your current settings (including environment variables).
*   **Cross-Platform Design:**
    *   Works on Windows (tested), Linux (tested), and macOS (untested).
    *   Includes platform-specific considerations for venv activation (for GPU recognition) and terminal launching.
*   **Dependency Awareness:**
    *   Checks for optional but recommended dependencies for GPU detection and model information

## 📋 Dependencies

### Required
*   **Python 3.7+** with tkinter support (typically included with Python)
*   **llama.cpp** built with server support (`llama-server` executable)

### Optional (Recommended)
*   **PyTorch** (`torch`) - **Required if you want automatic GPU detection and selection**
    *   Install in your virtual environment: `pip install torch`
    *   Without PyTorch, you can still manually configure GPU settings
    *   Enables automatic CUDA device detection and system resource information
*   **llama-cpp-python** - **Optional fallback for GGUF model analysis**
    *   Install in your virtual environment: `pip install llama-cpp-python`
    *   Provides enhanced model analysis when llama.cpp tools are unavailable
    *   The launcher works without it using built-in GGUF parsing and llama.cpp tools
*   **psutil** - **Optional for enhanced system information**
    *   Provides detailed CPU and RAM information across platforms
    *   Install with: `pip install psutil`

### Installation Example
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install recommended dependencies
pip install torch llama-cpp-python psutil
```

## 🛠️ Installation & Setup

### 1. Clone the Launcher
```bash
git clone https://github.com/thad0ctor/llama-server-launcher.git
cd llama-server-launcher
```

### 2. Setup Dependencies
Follow the [Dependencies](#-dependencies) section above to install Python dependencies in your virtual environment.

### 3. Build llama.cpp with CUDA Support

You'll need to build `llama.cpp` separately and point the launcher to the build directory. Here's an example build configuration:

> **⚠️ Example Environment Disclaimer:**  
> The following build example was tested on **Ubuntu 24.04** with **CUDA 12.9** and **GCC 13**. Your build flags may need adjustment based on your system configuration, CUDA version, GCC version, and GPU architecture.

```bash
# Navigate to your llama.cpp directory
cd /path/to/llama.cpp

# Clean previous builds
rm -rf build CMakeCache.txt CMakeFiles
mkdir build && cd build

# Configure with CUDA support and optimization flags
CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 cmake .. \
  -DGGML_CUDA=on \
  -DGGML_CUDA_FORCE_MMQ=on \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_FLAGS="--use_fast_math"

# Build with all available cores
make -j$(nproc)
```

> **📚 Need More Build Help?**  
> For additional building guidance, platform-specific instructions, and troubleshooting, refer to the official [llama.cpp documentation](https://github.com/ggerganov/llama.cpp#build).

**Key Build Flags Explained:**
- `-DGGML_CUDA=on` - Enables CUDA support
- `-DGGML_CUDA_FORCE_MMQ=on` - Forces use of multi-matrix quantization for better performance
- `-DCMAKE_CUDA_ARCHITECTURES=120` - Targets specific GPU architecture (adjust for your GPU)
- `-DCMAKE_CUDA_FLAGS="--use_fast_math"` - Enables fast math optimizations

### 4. Configure the Launcher
1. Run the launcher: `python llamacpp-server-launcher.py`
2. In the **Main** tab, set the **"llama.cpp Directory"** to your llama.cpp build folder
3. The launcher will automatically find the `llama-server` executable

&nbsp;

## 🚀 Core Components

This launcher aims to streamline your `llama.cpp` server workflow when working with and testing multiple models while making it more accessible and efficient for both new and experienced users.