"""CMake flag schema for the Build tab.

The two upstream projects (llama.cpp and ik_llama.cpp) share most options but
diverge in a few important ways:

  * llama.cpp uses ``GGML_CUDA_GRAPHS`` (default ON in source); ik_llama uses
    ``GGML_CUDA_USE_GRAPHS`` (same idea, different name) and additionally
    ``GGML_CUDA_FUSION`` (string, "1" default).
  * llama.cpp has ``GGML_CUDA_FA`` (toggleable); ik_llama unconditionally
    compiles FA kernels but has ``GGML_CUDA_FA_ALL_QUANTS``.
  * ik_llama exposes a family of kernel-tuning knobs absent in llama.cpp:
    ``GGML_CUDA_DMMV_X``, ``GGML_CUDA_MMV_Y``, ``GGML_CUDA_FORCE_DMMV``,
    ``GGML_CUDA_KQUANTS_ITER``, ``GGML_CUDA_MIN_BATCH_OFFLOAD``,
    ``GGML_CUDA_IQK_FORCE_BF16``, ``GGML_CUDA_F16``, plus the IQK CPU kernels
    ``GGML_IQK_MUL_MAT``, ``GGML_IQK_FLASH_ATTENTION``,
    ``GGML_IQK_FA_ALL_QUANTS``.
  * Recent llama.cpp adds ``GGML_CUDA_NCCL`` and the
    ``GGML_CUDA_COMPRESSION_MODE`` enum (cuda 12.8+), plus
    ``GGML_CPU_ALL_VARIANTS`` + ``GGML_BACKEND_DL`` for dynamic backend
    loading. ik_llama doesn't have those.

Every flag below records ``backends`` so the UI hides options that wouldn't
apply to the current backend. ``visible_when`` chains let the CUDA tuning
knobs collapse when ``GGML_CUDA`` is off.

The list is intentionally curated — not every cmake option lives here, only
the ones a user would meaningfully set in the build tab.
"""

from __future__ import annotations

import re
import shlex
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

_IS_MACOS = sys.platform == "darwin"


BOOL = "bool"
STRING = "string"
ENUM = "enum"

BACKEND_LLAMA = "llama.cpp"
BACKEND_IK = "ik_llama"
BOTH = (BACKEND_LLAMA, BACKEND_IK)


@dataclass
class CMakeFlag:
    key: str                                            # e.g. "GGML_CUDA"
    label: str                                          # display name
    group: str                                          # group title
    type: str                                           # BOOL | STRING | ENUM
    default: Any
    backends: tuple[str, ...] = BOTH
    choices: list[str] | None = None                 # for ENUM
    help: str = ""
    visible_when: Callable[[dict[str, Any]], bool] | None = None
    cuda_version_min: str | None = None              # informational
    # Some cmake options (e.g. CMAKE_CUDA_FLAGS) are not booleans/ints — they
    # ride along as raw strings. ``placeholder`` is the hint shown in the
    # entry widget.
    placeholder: str = ""
    # Optional validator for STRING flags. Given the raw entry value, returns
    # an error message if invalid, or ``None`` if acceptable. An empty value
    # is never passed here — empties are skipped (they fall back to the
    # upstream cmake default), so a validator only ever sees a user-typed
    # non-empty string.
    validate: Callable[[str], str | None] | None = None

    def applies_to(self, backend: str) -> bool:
        return backend in self.backends


# ─────────────────────────────────────────────────────────────────────────────
# Group registry — declared in display order
# ─────────────────────────────────────────────────────────────────────────────

GROUPS_ORDER = [
    "Build Targets",
    "CMake System",
    "Compiler & Linker",
    "CUDA",
    "CUDA Tuning",
    "CUDA (ik_llama-specific)",
    "ik_llama IQK CPU Kernels",
    "Other GPU Backends",
    "HIP / ROCm options",
    "Vulkan options",
    "SYCL options",
    "Metal options",
    "CPU Backend",
    "CPU SIMD (x86)",
    "BLAS / Accelerated math",
    "Optimization",
    "Runtime",
]


# ─────────────────────────────────────────────────────────────────────────────
# Visibility predicates
# ─────────────────────────────────────────────────────────────────────────────

def _truthy(values: dict[str, Any], key: str) -> bool:
    v = values.get(key)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "on", "true", "yes"}
    return bool(v)


def _cuda_on(values: dict[str, Any]) -> bool:
    return _truthy(values, "GGML_CUDA")


def _backend_dl_on(values: dict[str, Any]) -> bool:
    return _truthy(values, "GGML_BACKEND_DL")


def _hip_on(values: dict[str, Any]) -> bool:
    return _truthy(values, "GGML_HIP") or _truthy(values, "GGML_HIPBLAS")


def _vulkan_on(values: dict[str, Any]) -> bool:
    return _truthy(values, "GGML_VULKAN")


def _sycl_on(values: dict[str, Any]) -> bool:
    return _truthy(values, "GGML_SYCL")


def _metal_on(values: dict[str, Any]) -> bool:
    return _truthy(values, "GGML_METAL")


def _blas_on(values: dict[str, Any]) -> bool:
    return _truthy(values, "GGML_BLAS")


def _ui_on(values: dict[str, Any]) -> bool:
    return _truthy(values, "LLAMA_BUILD_UI")


def _cpu_on(values: dict[str, Any]) -> bool:
    return _truthy(values, "GGML_CPU")


# ─────────────────────────────────────────────────────────────────────────────
# Validators for STRING flags
# ─────────────────────────────────────────────────────────────────────────────

def _validate_power_of_two(v: str) -> str | None:
    """Accept a positive power of two (1, 2, 4, 8, 16, 32, …)."""
    s = v.strip()
    try:
        n = int(s)
    except ValueError:
        return f"must be a positive power of two, got {v!r}"
    if n <= 0 or (n & (n - 1)) != 0:
        return f"must be a positive power of two (got {n})"
    return None


# ``\d+`` + optional ``a``/``f`` (Hopper / Blackwell variants) +
# optional ``-real`` / ``-virtual`` suffix. nvcc 13.2 syntax. The
# Build tab's auto-detect produces tokens like ``86-real`` /
# ``120a-real``; manual entries need to follow the same shape or
# cmake will reject the whole list at configure time.
_CUDA_ARCH_TOKEN_RE = re.compile(
    r"^\d{2,3}[af]?(?:-real|-virtual)?$",
    re.IGNORECASE,
)


def _validate_cuda_archs(v: str) -> str | None:
    """Accept a ``;``-separated cmake-style CUDA arch list.

    Each token must look like ``<digits>[a|f][-real|-virtual]`` —
    e.g. ``86``, ``86-real``, ``120a-real``, ``90-virtual``. Empty
    tokens (from a trailing ``;`` or ``;;``) and unparseable
    tokens fail the validator so cmake doesn't reject the whole
    build at configure time. Mirrors the auto-detect output shape.
    """
    s = v.strip()
    if not s:
        return None
    tokens = s.split(";")
    bad: list[str] = []
    for token in tokens:
        candidate = token.strip()
        if not candidate:
            return "empty entry in semicolon-separated list"
        if not _CUDA_ARCH_TOKEN_RE.match(candidate):
            bad.append(candidate)
    if bad:
        return (
            f"invalid CUDA arch token(s): {bad!r}. "
            "Use ``<digits>[a|f][-real|-virtual]`` (e.g. ``86`` / "
            "``86-real`` / ``120a-real``)."
        )
    return None


def _validate_positive_int(v: str) -> str | None:
    s = v.strip()
    try:
        n = int(s)
    except ValueError:
        return f"must be a positive integer, got {v!r}"
    if n <= 0:
        return f"must be a positive integer (got {n})"
    return None


def _validate_one_or_two(v: str) -> str | None:
    """K-quants iters/thread: ik_llama only compiles kernels for 1 or 2."""
    s = v.strip()
    if s not in {"1", "2"}:
        return f"must be 1 or 2 (got {v!r})"
    return None


def _validate_zero_or_one(v: str) -> str | None:
    """Boolean-as-int knob (e.g. ``GGML_CUDA_FUSION``)."""
    s = v.strip()
    if s not in {"0", "1"}:
        return f"must be 0 or 1 (got {v!r})"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Flag catalogue
# ─────────────────────────────────────────────────────────────────────────────

FLAGS: list[CMakeFlag] = [
    # ── Build Targets ──
    CMakeFlag("LLAMA_BUILD_SERVER", "Build server", "Build Targets", BOOL, True,
              help="llama-server binary (always wanted for this launcher)."),
    CMakeFlag("LLAMA_BUILD_TESTS", "Build tests", "Build Targets", BOOL, False,
              help="Build the unit test suite. Off saves time."),
    CMakeFlag("LLAMA_BUILD_EXAMPLES", "Build examples", "Build Targets", BOOL, True,
              help="Build the bundled CLI examples (llama-cli, perplexity, etc.)."),
    CMakeFlag("LLAMA_BUILD_TOOLS", "Build tools", "Build Targets", BOOL, True,
              backends=(BACKEND_LLAMA,),
              help="llama.cpp: llama-quantize, llama-bench, etc. Separate from examples."),
    CMakeFlag("LLAMA_BUILD_COMMON", "Build common utils lib", "Build Targets", BOOL, True,
              backends=(BACKEND_LLAMA,),
              help="llama.cpp: the common/ helper lib used by examples + tools."),
    CMakeFlag("LLAMA_BUILD_UI", "Build embedded Web UI", "Build Targets", BOOL, True,
              backends=(BACKEND_LLAMA,),
              help="llama.cpp: build the server's bundled Web UI. Off saves time + Node toolchain."),
    CMakeFlag("LLAMA_USE_PREBUILT_UI", "Use prebuilt Web UI", "Build Targets", BOOL, True,
              backends=(BACKEND_LLAMA,), visible_when=_ui_on,
              help="llama.cpp: fetch the prebuilt UI bundle from HuggingFace instead of building it (no Node toolchain needed)."),
    CMakeFlag("LLAMA_OPENSSL", "Enable OpenSSL (HTTPS)", "Build Targets", BOOL, True,
              backends=(BACKEND_LLAMA,),
              help="llama.cpp: HTTPS support in the server. Off if you don't need TLS."),
    CMakeFlag("LLAMA_CURL", "Enable libcurl", "Build Targets", BOOL, False,
              help="Download models by URL with libcurl. Requires libcurl-dev installed."),
    CMakeFlag("LLAMA_LLGUIDANCE", "Enable LLGuidance", "Build Targets", BOOL, False,
              help="Structured-output / grammar support via LLGuidance."),
    CMakeFlag("LLAMA_USE_SYSTEM_GGML", "Use system libggml", "Build Targets", BOOL, False,
              backends=(BACKEND_LLAMA,),
              help="llama.cpp: link against a system-installed libggml instead of building the vendored one."),

    # ── CMake System ──
    CMakeFlag("CMAKE_INSTALL_PREFIX", "Install prefix", "CMake System", STRING, "",
              placeholder="/usr/local",
              help="Where `cmake --install` lands binaries/headers/libs."),
    CMakeFlag("CMAKE_EXPORT_COMPILE_COMMANDS", "Export compile_commands.json", "CMake System", BOOL, True,
              help="Emit compile_commands.json for clangd / IDE integration."),
    CMakeFlag("CMAKE_VERBOSE_MAKEFILE", "Verbose build output", "CMake System", BOOL, False,
              help="Print every compile/link command. Use for debugging build issues."),
    CMakeFlag("CMAKE_POSITION_INDEPENDENT_CODE", "Position-independent code (-fPIC)", "CMake System", BOOL, True,
              help="Required when building shared libs. Usually keep this on."),

    # ── Compiler & Linker ──
    CMakeFlag("CMAKE_BUILD_TYPE", "Build type", "Compiler & Linker", ENUM,
              "Release", choices=["Release", "RelWithDebInfo", "Debug", "MinSizeRel"],
              help="cmake build type. Release is the only one you usually want."),
    CMakeFlag("BUILD_SHARED_LIBS", "Build shared libs", "Compiler & Linker", BOOL, True,
              help="Build .so/.dll instead of .a. Required for GGML_BACKEND_DL."),
    CMakeFlag("CMAKE_C_FLAGS", "Extra C flags", "Compiler & Linker", STRING, "",
              placeholder="-march=native",
              help="Free-form C flag passthrough. The build script you referenced uses -march=native."),
    CMakeFlag("CMAKE_CXX_FLAGS", "Extra C++ flags", "Compiler & Linker", STRING, "",
              placeholder="-march=native",
              help="Free-form C++ flag passthrough."),
    CMakeFlag("CMAKE_CUDA_FLAGS", "Extra CUDA flags", "Compiler & Linker", STRING, "",
              placeholder="--use_fast_math -O3",
              visible_when=_cuda_on,
              help="Passed to nvcc. The reference scripts use --use_fast_math -O3."),

    # ── CUDA core ──
    CMakeFlag("GGML_CUDA", "Enable CUDA", "CUDA", BOOL, False,
              help="Master toggle for NVIDIA CUDA backend."),
    CMakeFlag("CMAKE_CUDA_ARCHITECTURES", "CUDA architectures", "CUDA", STRING, "",
              visible_when=_cuda_on,
              placeholder="86-real;120a-real;120-real",
              validate=_validate_cuda_archs,
              help=("Standard cmake variable. Semi-colon list of <code>[-real|-virtual]. "
                    "Use the auto-detect button to populate from your GPUs.")),

    # ── Flash Attention / kernels ──
    CMakeFlag("GGML_CUDA_FA", "Compile FlashAttention", "CUDA", BOOL, True,
              backends=(BACKEND_LLAMA,), visible_when=_cuda_on,
              help="llama.cpp only: compile FA CUDA kernels (default on)."),
    CMakeFlag("GGML_CUDA_FA_ALL_QUANTS", "FA: all quant types", "CUDA", BOOL, False,
              visible_when=_cuda_on,
              help="Compile FlashAttention kernels for every quant type. Slower build, faster inference for odd quants."),
    CMakeFlag("GGML_CUDA_GRAPHS", "CUDA graphs", "CUDA", BOOL, True,
              backends=(BACKEND_LLAMA,), visible_when=_cuda_on,
              help="llama.cpp: CUDA-graph kernel-launch batching. Reduces overhead."),
    CMakeFlag("GGML_CUDA_USE_GRAPHS", "CUDA graphs", "CUDA", BOOL, True,
              backends=(BACKEND_IK,), visible_when=_cuda_on,
              help="ik_llama: CUDA-graph kernel-launch batching. Reduces overhead."),
    CMakeFlag("GGML_CUDA_FORCE_MMQ", "Force MMQ kernels", "CUDA", BOOL, False,
              visible_when=_cuda_on,
              help="Use MMQ kernels instead of cuBLAS for batched matmuls."),
    CMakeFlag("GGML_CUDA_FORCE_CUBLAS", "Force cuBLAS", "CUDA", BOOL, False,
              visible_when=_cuda_on,
              help="Always use cuBLAS instead of MMQ. Faster for very large batches; uses more VRAM."),
    CMakeFlag("GGML_CUDA_PEER_MAX_BATCH_SIZE", "Peer max batch size", "CUDA", STRING, "128",
              visible_when=_cuda_on,
              validate=_validate_positive_int,
              help="Largest batch that uses peer-to-peer copy in multi-GPU setups."),
    CMakeFlag("GGML_CUDA_NO_PEER_COPY", "Disable P2P copies", "CUDA", BOOL, False,
              visible_when=_cuda_on,
              help="Disable GPU↔GPU peer copies (use only if NCCL/P2P misbehaves)."),
    CMakeFlag("GGML_CUDA_NO_VMM", "Disable CUDA VMM", "CUDA", BOOL, False,
              visible_when=_cuda_on,
              help="Skip CUDA Virtual Memory Management. Try this if you see VMM allocation errors."),
    CMakeFlag("GGML_CUDA_NCCL", "NCCL collectives", "CUDA", BOOL, False,
              backends=(BACKEND_LLAMA,), visible_when=_cuda_on,
              help="llama.cpp only: enable NVIDIA Collective Comm. Library for multi-GPU. "
                   "Requires libnccl-dev; default off because single-GPU builds don't need it."),
    CMakeFlag("GGML_CUDA_COMPRESSION_MODE", "PTX compression", "CUDA", ENUM, "size",
              choices=["none", "speed", "balance", "size"],
              backends=(BACKEND_LLAMA,), visible_when=_cuda_on,
              cuda_version_min="12.8",
              help="llama.cpp only, CUDA 12.8+: binary compression mode for the CUDA backend lib."),

    # ── CUDA Tuning (kernel/iter knobs — mostly ik_llama) ──
    CMakeFlag("GGML_CUDA_DMMV_X", "DMMV x-stride", "CUDA Tuning", STRING, "32",
              backends=(BACKEND_IK,), visible_when=_cuda_on,
              validate=_validate_power_of_two,
              help="ik_llama: x-stride for the dmmv kernels. Power-of-two only."),
    CMakeFlag("GGML_CUDA_MMV_Y", "MMV y-block", "CUDA Tuning", STRING, "1",
              backends=(BACKEND_IK,), visible_when=_cuda_on,
              validate=_validate_positive_int,
              help="ik_llama: y-block size for the mmv kernels."),
    CMakeFlag("GGML_CUDA_FORCE_DMMV", "Force DMMV", "CUDA Tuning", BOOL, False,
              backends=(BACKEND_IK,), visible_when=_cuda_on,
              help="ik_llama: prefer dmmv kernels over mmvq."),
    CMakeFlag("GGML_CUDA_KQUANTS_ITER", "K-quants iters/thread", "CUDA Tuning", STRING, "2",
              backends=(BACKEND_IK,), visible_when=_cuda_on,
              validate=_validate_one_or_two,
              help="ik_llama: iters-per-thread for Q2_K/Q6_K. Tune for SM occupancy."),
    CMakeFlag("GGML_CUDA_MIN_BATCH_OFFLOAD", "Min batch offload", "CUDA Tuning", STRING, "32",
              backends=(BACKEND_IK,), visible_when=_cuda_on,
              validate=_validate_positive_int,
              help="ik_llama: smallest batch size that triggers GPU offload."),
    CMakeFlag("GGML_CUDA_F16", "CUDA F16 intermediates", "CUDA Tuning", BOOL, False,
              backends=(BACKEND_IK,), visible_when=_cuda_on,
              help="ik_llama: use FP16 for some intermediate calculations. Slightly faster on Ampere+."),
    CMakeFlag("GGML_CUDA_IQK_FORCE_BF16", "IQK: force BF16 cuBLAS", "CUDA Tuning", BOOL, False,
              backends=(BACKEND_IK,), visible_when=_cuda_on,
              help="ik_llama: when no MMQ kernel is available, fall back to BF16 cuBLAS."),
    CMakeFlag("GGML_CUDA_FUSION", "CUDA fusion (0/1)", "CUDA Tuning", STRING, "1",
              backends=(BACKEND_IK,), visible_when=_cuda_on,
              validate=_validate_zero_or_one,
              help="ik_llama: enable kernel fusion. Set to 0 to disable."),

    # ── IK_LLAMA IQK CPU kernels ──
    CMakeFlag("GGML_IQK_MUL_MAT", "IQK matmul kernels", "ik_llama IQK CPU Kernels", BOOL, True,
              backends=(BACKEND_IK,),
              help="Optimized integer quantized kernel (IQK) matmuls — the headline ik_llama feature."),
    CMakeFlag("GGML_IQK_FLASH_ATTENTION", "IQK FlashAttention (CPU)", "ik_llama IQK CPU Kernels", BOOL, True,
              backends=(BACKEND_IK,),
              help="IQK FlashAttention CPU kernels."),
    CMakeFlag("GGML_IQK_FA_ALL_QUANTS", "IQK FA: all quants", "ik_llama IQK CPU Kernels", BOOL, True,
              backends=(BACKEND_IK,),
              help="Compile IQK FA kernels for every quant type. Off reduces build time."),

    # ── Other GPU backends ──
    CMakeFlag("GGML_VULKAN", "Vulkan", "Other GPU Backends", BOOL, False),
    CMakeFlag("GGML_HIP", "HIP / ROCm", "Other GPU Backends", BOOL, False,
              backends=(BACKEND_LLAMA,),
              help="llama.cpp: AMD HIP / ROCm backend."),
    CMakeFlag("GGML_HIPBLAS", "hipBLAS (ROCm)", "Other GPU Backends", BOOL, False,
              backends=(BACKEND_IK,),
              help="ik_llama: hipBLAS-based ROCm backend."),
    CMakeFlag("GGML_METAL", "Metal (macOS)", "Other GPU Backends", BOOL, False,
              help="Apple Metal backend. Defaults to ON on macOS automatically."),
    CMakeFlag("GGML_SYCL", "SYCL (Intel)", "Other GPU Backends", BOOL, False),
    CMakeFlag("GGML_OPENCL", "OpenCL", "Other GPU Backends", BOOL, False,
              backends=(BACKEND_LLAMA,)),
    CMakeFlag("GGML_KOMPUTE", "Kompute (Vulkan)", "Other GPU Backends", BOOL, False,
              backends=(BACKEND_IK,)),
    CMakeFlag("GGML_MUSA", "MUSA (Moore Threads)", "Other GPU Backends", BOOL, False,
              help="Moore Threads MUSA backend (Chinese GPUs)."),
    CMakeFlag("GGML_RPC", "RPC backend", "Other GPU Backends", BOOL, False,
              help="Network-RPC backend for distributed inference."),

    # ── HIP sub-options (visible only when HIP/HIPBLAS is on) ──
    CMakeFlag("GGML_HIP_GRAPHS", "HIP graphs", "HIP / ROCm options", BOOL, True,
              backends=(BACKEND_LLAMA,), visible_when=_hip_on,
              help="HIP-graph kernel launch batching (analogous to CUDA graphs)."),
    CMakeFlag("GGML_HIP_RCCL", "RCCL collectives", "HIP / ROCm options", BOOL, False,
              backends=(BACKEND_LLAMA,), visible_when=_hip_on,
              help="ROCm Collective Comm. Library for multi-GPU."),
    CMakeFlag("GGML_HIP_NO_VMM", "Disable HIP VMM", "HIP / ROCm options", BOOL, True,
              backends=(BACKEND_LLAMA,), visible_when=_hip_on,
              help="Skip HIP virtual memory mgmt. Default ON in upstream."),
    CMakeFlag("GGML_HIP_ROCWMMA_FATTN", "rocWMMA FlashAttention", "HIP / ROCm options", BOOL, False,
              backends=(BACKEND_LLAMA,), visible_when=_hip_on,
              help="Use rocWMMA for FlashAttention kernels (CDNA cards)."),
    CMakeFlag("GGML_HIP_MMQ_MFMA", "MFMA MMQ kernels", "HIP / ROCm options", BOOL, True,
              backends=(BACKEND_LLAMA,), visible_when=_hip_on,
              help="Use MFMA matrix instructions for MMQ on CDNA."),
    CMakeFlag("GGML_HIP_EXPORT_METRICS", "Kernel perf metrics", "HIP / ROCm options", BOOL, False,
              backends=(BACKEND_LLAMA,), visible_when=_hip_on,
              help="Emit per-kernel performance metrics. Adds runtime overhead."),
    CMakeFlag("GGML_HIP_UMA", "HIP unified memory", "HIP / ROCm options", BOOL, False,
              backends=(BACKEND_IK,), visible_when=_hip_on,
              help="ik_llama: HIP unified-memory architecture (APUs)."),

    # ── Vulkan sub-options ──
    CMakeFlag("GGML_VULKAN_CHECK_RESULTS", "Run Vulkan op checks", "Vulkan options", BOOL, False,
              visible_when=_vulkan_on,
              help="Validate Vulkan op outputs against CPU reference. Very slow; debug only."),
    CMakeFlag("GGML_VULKAN_DEBUG", "Vulkan debug output", "Vulkan options", BOOL, False,
              visible_when=_vulkan_on),
    CMakeFlag("GGML_VULKAN_MEMORY_DEBUG", "Vulkan memory debug", "Vulkan options", BOOL, False,
              visible_when=_vulkan_on),
    CMakeFlag("GGML_VULKAN_SHADER_DEBUG_INFO", "Shader debug info", "Vulkan options", BOOL, False,
              visible_when=_vulkan_on),
    CMakeFlag("GGML_VULKAN_VALIDATE", "Validation layer", "Vulkan options", BOOL, False,
              visible_when=_vulkan_on,
              help="Enable Vulkan validation layer. Useful for driver bugs; slow."),
    CMakeFlag("GGML_VULKAN_RUN_TESTS", "Run Vulkan tests", "Vulkan options", BOOL, False,
              visible_when=_vulkan_on),
    CMakeFlag("GGML_VULKAN_NO_COOPMAT", "Disable coopmat", "Vulkan options", BOOL, False,
              backends=(BACKEND_IK,), visible_when=_vulkan_on,
              help="ik_llama: don't use VK_KHR_cooperative_matrix even if supported."),
    CMakeFlag("GGML_VULKAN_NO_COOPMAT2", "Disable coopmat2", "Vulkan options", BOOL, False,
              backends=(BACKEND_IK,), visible_when=_vulkan_on),
    CMakeFlag("GGML_VULKAN_NO_BF16", "Disable Vulkan BF16", "Vulkan options", BOOL, False,
              backends=(BACKEND_IK,), visible_when=_vulkan_on),
    CMakeFlag("GGML_VULKAN_NO_INT_DOT", "Disable Vulkan integer dot", "Vulkan options", BOOL, False,
              backends=(BACKEND_IK,), visible_when=_vulkan_on),

    # ── SYCL sub-options ──
    CMakeFlag("GGML_SYCL_F16", "SYCL FP16", "SYCL options", BOOL, False,
              visible_when=_sycl_on,
              help="Use FP16 for some SYCL kernel computations."),
    CMakeFlag("GGML_SYCL_GRAPH", "SYCL graphs", "SYCL options", BOOL, True,
              backends=(BACKEND_LLAMA,), visible_when=_sycl_on),
    CMakeFlag("GGML_SYCL_HOST_MEM_FALLBACK", "Host memory fallback", "SYCL options", BOOL, True,
              backends=(BACKEND_LLAMA,), visible_when=_sycl_on,
              help="Allow host-memory fallback in SYCL reorder (kernel 6.8+)."),
    CMakeFlag("GGML_SYCL_SUPPORT_LEVEL_ZERO", "Level Zero API", "SYCL options", BOOL, True,
              backends=(BACKEND_LLAMA,), visible_when=_sycl_on,
              help="Use the Intel Level Zero API path."),
    CMakeFlag("GGML_SYCL_DNN", "oneDNN integration", "SYCL options", BOOL, True,
              backends=(BACKEND_LLAMA,), visible_when=_sycl_on,
              help="Use Intel oneDNN inside the SYCL backend."),
    CMakeFlag("GGML_SYCL_TARGET", "SYCL target device", "SYCL options", ENUM, "INTEL",
              choices=["INTEL", "NVIDIA", "AMD"], visible_when=_sycl_on,
              help="Which SYCL backend implementation to target."),
    CMakeFlag("GGML_SYCL_DEVICE_ARCH", "SYCL device arch", "SYCL options", STRING, "",
              placeholder="intel_gpu_pvc",
              visible_when=_sycl_on,
              help="Device architecture passed to the SYCL toolchain (optional)."),

    # ── Metal sub-options ──
    CMakeFlag("GGML_METAL_NDEBUG", "Disable Metal debugging", "Metal options", BOOL, False,
              visible_when=_metal_on),
    CMakeFlag("GGML_METAL_SHADER_DEBUG", "Shader debug (-fno-fast-math)", "Metal options", BOOL, False,
              visible_when=_metal_on),
    CMakeFlag("GGML_METAL_EMBED_LIBRARY", "Embed Metal library", "Metal options", BOOL, True,
              visible_when=_metal_on,
              help="Embed the Metal shader library in the binary (default ON on macOS)."),
    CMakeFlag("GGML_METAL_MACOSX_VERSION_MIN", "Minimum macOS version", "Metal options", STRING, "",
              placeholder="11.0", visible_when=_metal_on),
    CMakeFlag("GGML_METAL_STD", "Metal std (-std flag)", "Metal options", STRING, "",
              backends=(BACKEND_LLAMA,), visible_when=_metal_on),

    # ── CPU Backend (master toggle + tuning) ──
    CMakeFlag("GGML_CPU", "Enable CPU backend", "CPU Backend", BOOL, True,
              help="Master CPU-backend toggle. Off for GPU-only inference (rare)."),
    CMakeFlag("GGML_CPU_REPACK", "Runtime Q4_0→Q4_X_X repack", "CPU Backend", BOOL, True,
              backends=(BACKEND_LLAMA,), visible_when=_cpu_on,
              help="Runtime weight conversion for better CPU throughput on certain quants."),
    CMakeFlag("GGML_CPU_HBM", "CPU HBM (memkind)", "CPU Backend", BOOL, False,
              visible_when=_cpu_on,
              help="Use memkind for HBM allocation. Niche; for HBM-equipped Xeons."),
    CMakeFlag("GGML_CPU_KLEIDIAI", "KleidiAI kernels (ARM)", "CPU Backend", BOOL, False,
              backends=(BACKEND_LLAMA,), visible_when=_cpu_on,
              help="Optimized ARM matmul kernels via KleidiAI."),
    CMakeFlag("GGML_SCHED_MAX_COPIES", "Pipeline copies", "CPU Backend", STRING, "4",
              backends=(BACKEND_LLAMA,),
              validate=_validate_positive_int,
              help="Max input copies for pipeline parallelism. Higher = more memory, slightly higher throughput."),

    # ── CPU SIMD ──
    CMakeFlag("GGML_NATIVE", "Native (-march=native)", "CPU SIMD (x86)", BOOL, True,
              help="Let the compiler enable every ISA the host supports. Overrides individual AVX flags."),
    CMakeFlag("GGML_SSE42", "SSE 4.2", "CPU SIMD (x86)", BOOL, True,
              backends=(BACKEND_LLAMA,),
              help="x86 SSE 4.2 baseline. Almost universally available on modern CPUs."),
    CMakeFlag("GGML_AVX", "AVX", "CPU SIMD (x86)", BOOL, True),
    CMakeFlag("GGML_AVX2", "AVX2", "CPU SIMD (x86)", BOOL, True),
    CMakeFlag("GGML_AVX_VNNI", "AVX-VNNI", "CPU SIMD (x86)", BOOL, False,
              backends=(BACKEND_LLAMA,)),
    CMakeFlag("GGML_AVX512", "AVX512F", "CPU SIMD (x86)", BOOL, False),
    CMakeFlag("GGML_AVX512_VBMI", "AVX512-VBMI", "CPU SIMD (x86)", BOOL, False),
    CMakeFlag("GGML_AVX512_VNNI", "AVX512-VNNI", "CPU SIMD (x86)", BOOL, False),
    CMakeFlag("GGML_AVX512_BF16", "AVX512-BF16", "CPU SIMD (x86)", BOOL, False),
    CMakeFlag("GGML_FMA", "FMA", "CPU SIMD (x86)", BOOL, True),
    CMakeFlag("GGML_F16C", "F16C", "CPU SIMD (x86)", BOOL, True),
    CMakeFlag("GGML_BMI2", "BMI2", "CPU SIMD (x86)", BOOL, True,
              backends=(BACKEND_LLAMA,)),
    CMakeFlag("GGML_AMX_TILE", "AMX-TILE", "CPU SIMD (x86)", BOOL, False,
              backends=(BACKEND_LLAMA,)),
    CMakeFlag("GGML_AMX_INT8", "AMX-INT8", "CPU SIMD (x86)", BOOL, False,
              backends=(BACKEND_LLAMA,)),
    CMakeFlag("GGML_AMX_BF16", "AMX-BF16", "CPU SIMD (x86)", BOOL, False,
              backends=(BACKEND_LLAMA,)),

    # ── BLAS / Accelerated math ──
    CMakeFlag("GGML_BLAS", "Use BLAS", "BLAS / Accelerated math", BOOL, False,
              help="External BLAS via Accelerate / OpenBLAS / MKL / etc. Defaults ON on macOS automatically."),
    CMakeFlag("GGML_BLAS_VENDOR", "BLAS vendor", "BLAS / Accelerated math", ENUM, "Generic",
              choices=[
                  "Generic", "Apple", "OpenBLAS", "FLAME", "ATLAS",
                  "Intel10_64lp", "Intel10_64lp_seq", "Intel10_64ilp", "Intel10_64ilp_seq",
                  "FlexiBLAS", "NVHPC", "IBMESSL",
              ],
              visible_when=_blas_on,
              help="Which BLAS implementation to use. 'Apple' on macOS uses Accelerate."),
    CMakeFlag("GGML_ACCELERATE", "Apple Accelerate", "BLAS / Accelerated math", BOOL, _IS_MACOS,
              # Hide on Linux/Windows — Accelerate is a macOS-only framework
              # and the checkbox does nothing on other platforms.
              visible_when=lambda _v: _IS_MACOS,
              help="Apple Accelerate framework. Auto-enabled on macOS; OFF elsewhere."),

    # ── Optimization ──
    CMakeFlag("GGML_LTO", "Link-time optimization", "Optimization", BOOL, True,
              help="LTO — usually a small but free win."),
    CMakeFlag("GGML_CCACHE", "ccache", "Optimization", BOOL, True,
              help="Use ccache if available. Re-builds become near-instant."),
    CMakeFlag("GGML_LLAMAFILE", "tinyBLAS (llamafile)", "Optimization", BOOL, True,
              backends=(BACKEND_LLAMA,),
              help="llama.cpp only: bundle tinyBLAS kernels. Off if you only use cuBLAS."),
    CMakeFlag("GGML_STATIC", "Static link", "Optimization", BOOL, False,
              help="Statically link the binaries."),

    # ── Runtime ──
    CMakeFlag("GGML_OPENMP", "OpenMP", "Runtime", BOOL, True,
              help="OpenMP parallelization on CPU paths."),
    CMakeFlag("GGML_BACKEND_DL", "Dynamic backend loading", "Runtime", BOOL, False,
              backends=(BACKEND_LLAMA,),
              help="llama.cpp: build backends as runtime-loadable .so/.dll. Required by GGML_CPU_ALL_VARIANTS."),
    CMakeFlag("GGML_CPU_ALL_VARIANTS", "All CPU SIMD variants", "Runtime", BOOL, False,
              backends=(BACKEND_LLAMA,), visible_when=_backend_dl_on,
              help="llama.cpp: build every CPU variant for runtime dispatch. Requires GGML_BACKEND_DL=ON."),
]


# ─────────────────────────────────────────────────────────────────────────────
# Lookup + grouping helpers
# ─────────────────────────────────────────────────────────────────────────────

_FLAG_BY_KEY: dict[str, CMakeFlag] = {f.key: f for f in FLAGS}


def get_flag(key: str) -> CMakeFlag | None:
    return _FLAG_BY_KEY.get(key)


def flags_for_backend(backend: str) -> list[CMakeFlag]:
    return [f for f in FLAGS if f.applies_to(backend)]


def groups_for_backend(backend: str) -> list[tuple[str, list[CMakeFlag]]]:
    """Returns groups in declared order, only those non-empty for the backend."""
    by_group: dict[str, list[CMakeFlag]] = {}
    for f in flags_for_backend(backend):
        by_group.setdefault(f.group, []).append(f)
    out: list[tuple[str, list[CMakeFlag]]] = []
    for grp in GROUPS_ORDER:
        if grp in by_group:
            out.append((grp, by_group[grp]))
    for grp, lst in by_group.items():
        if grp not in GROUPS_ORDER:
            out.append((grp, lst))
    return out


def default_values_for_backend(backend: str) -> dict[str, Any]:
    return {f.key: f.default for f in flags_for_backend(backend)}


# ─────────────────────────────────────────────────────────────────────────────
# Auto-detected preset — "Optimized for this system"
# ─────────────────────────────────────────────────────────────────────────────

def _parse_dotted_version(v: str | None) -> tuple[int, ...] | None:
    """Coerce a "12.8" / "12.8.1" / "12" string into a comparable tuple. Returns
    None if the input is missing or contains a non-numeric segment."""
    if not v:
        return None
    parts: list[int] = []
    for chunk in str(v).strip().split("."):
        if not chunk:
            return None
        try:
            parts.append(int(chunk))
        except ValueError:
            return None
    return tuple(parts) if parts else None


def _cuda_version_satisfies(
    min_required: str | None,
    detected: str | None,
    *,
    assume_compatible_if_unknown: bool = False,
) -> bool:
    """Return True iff a flag declaring ``cuda_version_min=min_required`` is
    permitted for ``detected``.

    Defaults to **fail-closed** semantics: an unknown ``detected`` is
    treated as INCOMPATIBLE when ``min_required`` is set. The emit /
    validate paths (``values_to_cmake_args``, ``validate_values``) use
    this default so a failed CUDA-toolkit probe doesn't silently inject
    a version-fenced flag like ``GGML_CUDA_COMPRESSION_MODE`` and break
    configure on older toolkits.

    Call sites that genuinely need permissive behaviour (e.g. preserving
    a user's manual toggle when detection failed) pass
    ``assume_compatible_if_unknown=True`` explicitly.

    - If the flag has no ``cuda_version_min`` set, always True.
    - If the detected CUDA version is unknown (None), return
      ``assume_compatible_if_unknown`` (False by default).
    - Otherwise compare numerically: detected >= min_required.
    """
    if not min_required:
        return True
    detected_t = _parse_dotted_version(detected)
    if detected_t is None:
        return assume_compatible_if_unknown
    min_t = _parse_dotted_version(min_required)
    if min_t is None:
        # Unparseable ``min_required`` — schema-level bug, not a runtime
        # toolkit issue. Stay permissive so a flag with a malformed
        # ``cuda_version_min`` isn't silently dropped for everyone.
        return True
    return detected_t >= min_t


def build_autodetect_values(
    backend: str,
    *,
    cuda_available: bool,
    avx512_supported: bool,
    has_ccache: bool,
    cuda_version: str | None = None,
    cuda_device_count: int = 0,
) -> dict[str, Any]:
    """Return a flag-values dict tuned for the detected system, mirroring
    the reference scripts (fast-math CUDA, FA-all-quants, LTO, P2P 512,
    AVX512 if the CPU has it). Caller is expected to additionally set
    CMAKE_CUDA_ARCHITECTURES from CudaArchInfo via detection.archs_to_cmake_value.

    ``cuda_version`` (e.g. ``"12.8"``) gates the version-fenced flags such as
    ``GGML_CUDA_COMPRESSION_MODE``. When unknown, leave the flag at its
    schema default so the resulting preset doesn't silently inject a flag
    the user's toolkit can't accept.

    ``cuda_device_count`` is currently UNUSED. ``GGML_CUDA_NCCL`` used to
    auto-enable on ``>= 2`` devices, but multi-GPU alone doesn't prove
    ``libnccl-dev`` is installed — toggling the flag on without the
    library turned the autodetect path's "optimized" preset into a
    confusing configure failure for users without it. NCCL is now
    a manual opt-in from the Build tab. The parameter stays on the
    signature for back-compat with existing callers.
    """
    values = default_values_for_backend(backend)

    if cuda_available:
        values["GGML_CUDA"] = True
        values["GGML_CUDA_FA_ALL_QUANTS"] = True
        values["GGML_CUDA_PEER_MAX_BATCH_SIZE"] = "512"
        values["CMAKE_CUDA_FLAGS"] = "--use_fast_math -O3"
        if backend == BACKEND_LLAMA:
            values["GGML_CUDA_FA"] = True
            values["GGML_CUDA_GRAPHS"] = True
            # NCCL is intentionally NOT auto-enabled. Multi-GPU alone
            # doesn't guarantee ``libnccl-dev`` is installed, and
            # turning the flag on without the lib turns "optimized
            # preset" into a confusing configure failure. Users with
            # multi-GPU + NCCL installed toggle ``GGML_CUDA_NCCL`` on
            # manually from the Build tab. (The autodetect path used
            # to set this on ``cuda_device_count >= 2`` alone, which
            # tripped up everyone without libnccl-dev.)
            compression_flag = _FLAG_BY_KEY.get("GGML_CUDA_COMPRESSION_MODE")
            # Strict gate: only AUTO-enable a version-fenced flag when we
            # actually know the toolkit version. ``_cuda_version_satisfies``
            # is now ALSO fail-closed on unknown detection by default
            # (matches the emit / validate paths), but keeping the
            # explicit ``cuda_version is not None`` guard here documents
            # the autodetect-side intent — an undetected CUDA install
            # must not get version-fenced flags silently injected, even
            # if a future refactor changes the helper's defaults again.
            if (
                compression_flag is not None
                and cuda_version is not None
                and _cuda_version_satisfies(
                    compression_flag.cuda_version_min, cuda_version
                )
            ):
                values["GGML_CUDA_COMPRESSION_MODE"] = "speed"
        else:
            values["GGML_CUDA_USE_GRAPHS"] = True
            values["GGML_CUDA_FORCE_MMQ"] = True
            values["GGML_CUDA_IQK_FORCE_BF16"] = True

    values["GGML_NATIVE"] = True
    values["GGML_LTO"] = True
    values["GGML_CCACHE"] = has_ccache
    values["GGML_OPENMP"] = True
    values["GGML_LLAMAFILE"] = True

    if avx512_supported:
        # Only auto-enable the umbrella ``GGML_AVX512`` flag. The
        # sub-features (``_VBMI`` / ``_VNNI`` / ``_BF16``) are gated
        # on dedicated CPU capability bits that
        # ``avx512_supported`` (a single AVX-512 baseline probe)
        # doesn't measure — turning them on without those bits
        # produces "illegal instruction" at runtime on CPUs that
        # advertise AVX-512 but lack the specific extensions
        # (Skylake-X, Cannon Lake, etc.). The user can still flip
        # them on manually from the Build tab if their CPU
        # actually supports them.
        values["GGML_AVX512"] = True

    # Don't hard-code -march=native here: it's a GCC/Clang-only flag and
    # breaks CMake configure on MSVC / clang-cl out of the box. ``GGML_NATIVE``
    # (set True above) is the portable knob — upstream's cmake translates it
    # to the right compiler-specific flag (``-march=native`` on GCC/Clang,
    # ``/arch:AVX2`` etc. on MSVC). Users who want extra C/C++ flags can
    # type them into the "Extra C flags" / "Extra C++ flags" fields.

    values["LLAMA_BUILD_TESTS"] = False
    values["LLAMA_BUILD_SERVER"] = True
    values["LLAMA_BUILD_EXAMPLES"] = True

    if backend == BACKEND_IK:
        values["GGML_IQK_MUL_MAT"] = True
        values["GGML_IQK_FLASH_ATTENTION"] = True
        values["GGML_IQK_FA_ALL_QUANTS"] = True

    return values


# ─────────────────────────────────────────────────────────────────────────────
# Materialise to cmake -D args
# ─────────────────────────────────────────────────────────────────────────────

def _bool_str(v: Any) -> str:
    if isinstance(v, bool):
        return "ON" if v else "OFF"
    if isinstance(v, str):
        return "ON" if v.strip().lower() in {"1", "on", "true", "yes"} else "OFF"
    return "ON" if v else "OFF"


def values_to_cmake_args(
    backend: str,
    values: dict[str, Any],
    *,
    extra_cmake_args: str = "",
    cuda_version: str | None = None,
) -> list[str]:
    """Convert a flag-values dict to ``-DKEY=VAL`` strings, skipping flags
    that don't apply to ``backend`` and skipping empty STRING entries.
    ``extra_cmake_args`` is appended verbatim after shell-split.

    Flags declaring ``cuda_version_min`` are skipped when
    ``_cuda_version_satisfies`` says the toolkit can't accept them. The
    helper now defaults to FAIL-CLOSED: passing ``cuda_version=None``
    causes every version-fenced flag to be SKIPPED (so an undetected
    toolkit can't get a 12.8+-only flag silently injected). Pass an
    actual detected version string to enable the flags."""
    out: list[str] = []
    seen: set[str] = set()
    for flag in flags_for_backend(backend):
        if flag.key in seen:
            continue
        seen.add(flag.key)
        if flag.key not in values:
            continue
        if flag.visible_when and not flag.visible_when(values):
            # Hidden because a dependency is off; don't emit.
            continue
        if not _cuda_version_satisfies(flag.cuda_version_min, cuda_version):
            continue
        v = values[flag.key]
        if flag.type == BOOL:
            out.append(f"-D{flag.key}={_bool_str(v)}")
        elif flag.type == ENUM:
            if v:
                out.append(f"-D{flag.key}={v}")
        else:  # STRING
            sv = "" if v is None else str(v).strip()
            if sv:
                out.append(f"-D{flag.key}={sv}")
    if extra_cmake_args.strip():
        # ``posix=True`` (the default) treats ``\`` as an escape, which
        # mangles Windows paths like ``-DCMAKE_PREFIX_PATH=C:\path\to\lib``
        # into ``-DCMAKE_PREFIX_PATH=C:pathtoli``. Use platform-appropriate
        # quoting rules.
        import os as _os
        try:
            out.extend(shlex.split(extra_cmake_args, posix=(_os.name != "nt")))
        except ValueError:
            out.extend(extra_cmake_args.split())
    return out


def validate_flag_value(flag: CMakeFlag, value: Any) -> str | None:
    """Run ``flag.validate`` against ``value`` and return an error message,
    or ``None`` if the value is acceptable (or the flag has no validator).

    Empty/blank values are always accepted: they are not emitted to cmake
    (see :func:`values_to_cmake_args`) so the upstream default applies."""
    if flag.validate is None:
        return None
    sv = "" if value is None else str(value).strip()
    if not sv:
        return None
    return flag.validate(sv)


def validate_values(
    backend: str,
    values: dict[str, Any],
    *,
    cuda_version: str | None = None,
) -> list[tuple[str, str]]:
    """Validate every applicable, currently-visible flag in ``values`` —
    STRING flags via ``flag.validate`` AND ENUM flags against
    ``flag.choices``. A hand-edited preset with
    ``"GGML_CUDA_COMPRESSION_MODE": "garbage"`` is now rejected here
    instead of blowing up at cmake-configure time.

    Returns a list of ``(label, message)`` tuples for flags that fail
    validation. Flags hidden by ``visible_when`` are skipped — they
    aren't emitted to cmake, so their value can't break the build.
    Flags whose ``cuda_version_min`` exceeds the supplied
    ``cuda_version`` are likewise skipped (and the same
    fail-closed-on-unknown semantics apply as in
    :func:`values_to_cmake_args`)."""
    errors: list[tuple[str, str]] = []
    seen: set[str] = set()
    for flag in flags_for_backend(backend):
        if flag.key in seen:
            continue
        seen.add(flag.key)
        if flag.key not in values:
            continue
        if flag.visible_when and not flag.visible_when(values):
            continue
        if not _cuda_version_satisfies(flag.cuda_version_min, cuda_version):
            continue
        if flag.type == ENUM:
            # Validate persisted enum values against the declared choices —
            # this used to be skipped entirely when ``flag.validate is None``
            # (which it is for most enums), so a hand-edited preset with
            # ``"GGML_CUDA_COMPRESSION_MODE": "garbage"`` would slip through
            # and only blow up at cmake-configure time.
            raw_value = values[flag.key]
            sv = "" if raw_value is None else str(raw_value).strip()
            if sv and flag.choices and sv not in flag.choices:
                errors.append(
                    (
                        flag.label,
                        f"value {sv!r} is not one of the allowed choices "
                        f"({', '.join(flag.choices)}).",
                    )
                )
            continue
        if flag.validate is None:
            continue
        msg = validate_flag_value(flag, values[flag.key])
        if msg:
            errors.append((flag.label, msg))
    # Cross-flag dependency: ``GGML_BACKEND_DL=ON`` requires
    # ``BUILD_SHARED_LIBS=ON`` (documented in BUILD_SHARED_LIBS's help
    # text — building static libs makes the dynamic-backend loader
    # nonsensical). Surface this at start-build validation rather than
    # letting cmake fail with a less obvious linker error.
    if _truthy(values, "GGML_BACKEND_DL") and not _truthy(values, "BUILD_SHARED_LIBS"):
        errors.append(
            (
                "Dynamic backend loading",
                "GGML_BACKEND_DL requires BUILD_SHARED_LIBS=ON; either enable "
                "shared-libs or disable dynamic-backend loading.",
            )
        )
    return errors
