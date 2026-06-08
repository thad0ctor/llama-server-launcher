"""Detection helpers for the Build tab.

Everything here is best-effort: each probe returns a structured result and
never raises out to callers. The Build tab presents whatever was found and
falls back to manual entry when a probe fails.
"""

from __future__ import annotations

import os
import re
import csv
import io
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Known CUDA compute-capability catalogue
# ─────────────────────────────────────────────────────────────────────────────
#
# Comprehensive list of CUDA architectures users may want to build for, even
# if they aren't present on the build host. The ``has_a_variant`` flag marks
# architectures that have an ``-a`` (architecture-accelerated) form in nvcc;
# those unlock SM-specific features like Hopper wgmma/TMA, Blackwell FP4
# tensor cores, etc. The ``-a`` variant must be paired with the plain variant
# for portability across minor revisions in that family.
#
# CUDA toolkit version that first supports the architecture is shown for
# reference (informational; not enforced here).

@dataclass(frozen=True)
class KnownArch:
    cc: str               # "8.6"
    name: str             # "RTX 30xx (Ampere)"
    family: str           # "Ampere"
    has_a_variant: bool   # True if sm_XXa exists in nvcc (Hopper+, Blackwell+)
    has_f_variant: bool   # True if sm_XXf exists in nvcc (CUDA 13+, Blackwell only)
    min_cuda: str         # earliest CUDA toolkit that supports this
    deprecated: bool = False  # True if dropped from current CUDA toolchains


# Sourced from `nvcc --help` (CUDA 13.2) for the modern entries and from
# NVIDIA Programming Guide tables for the older deprecated ones. The
# has_a_variant / has_f_variant flags match what nvcc 13.2 actually accepts.
# Deprecated entries are gated in the UI behind "Show deprecated archs".
KNOWN_CUDA_ARCHS: list[KnownArch] = [
    # Kepler (deprecated in CUDA 11.x; removed in CUDA 12+)
    KnownArch("3.5", "Tesla K20/K40 / GTX Titan (Kepler)",   "Kepler",   False, False, "5.0", deprecated=True),
    KnownArch("3.7", "Tesla K80 (Kepler datacenter)",        "Kepler",   False, False, "7.0", deprecated=True),
    # Maxwell (deprecated in CUDA 13.x)
    KnownArch("5.0", "GTX 9xx / Quadro M (Maxwell 1)",       "Maxwell",  False, False, "6.0", deprecated=True),
    KnownArch("5.2", "GTX 9xx Ti / Titan X (Maxwell 2)",     "Maxwell",  False, False, "6.0", deprecated=True),
    KnownArch("5.3", "Tegra X1 (Maxwell mobile)",            "Maxwell",  False, False, "6.5", deprecated=True),
    # Pascal (deprecated in CUDA 13.x)
    KnownArch("6.0", "Tesla P100 (Pascal datacenter)",       "Pascal",   False, False, "8.0", deprecated=True),
    KnownArch("6.1", "GTX 10xx / Titan Xp (Pascal)",         "Pascal",   False, False, "8.0", deprecated=True),
    KnownArch("6.2", "Tegra X2 (Pascal mobile)",             "Pascal",   False, False, "8.0", deprecated=True),
    # Volta (deprecated in CUDA 13.x)
    KnownArch("7.0", "Tesla V100 / Titan V (Volta)",         "Volta",    False, False, "9.0", deprecated=True),
    KnownArch("7.2", "Tegra Xavier (Volta mobile)",          "Volta",    False, False, "9.2", deprecated=True),
    # Turing
    KnownArch("7.5", "RTX 20xx / GTX 16xx / T4 (Turing)",    "Turing",   False, False, "10.0"),
    # Ampere
    KnownArch("8.0", "A100 / A30 (Ampere datacenter)",       "Ampere",   False, False, "11.0"),
    KnownArch("8.6", "RTX 30xx / RTX A-series / A40/A10/A16/A2 (Ampere)",
                                                                    "Ampere",   False, False, "11.1"),
    KnownArch("8.7", "Jetson AGX Orin / Orin NX / Orin Nano","Ampere",   False, False, "11.4"),
    # Niche sm_88 — nvcc 13 accepts it; product confirmation pending.
    KnownArch("8.8", "sm_88 (Ampere/Hopper variant)",        "Ampere",   False, False, "12.x"),
    # Ada Lovelace
    KnownArch("8.9", "RTX 40xx / L4/L40/L40S / RTX Ada",     "Ada",      False, False, "11.8"),
    # Hopper
    KnownArch("9.0", "H100 / H200 / GH200 (Hopper)",         "Hopper",   True,  False, "11.8"),
    # Blackwell datacenter (sm_100/103/110) — -a and -f both valid.
    KnownArch("10.0", "B200 / GB200 (Blackwell datacenter)", "Blackwell", True, True, "12.8"),
    KnownArch("10.3", "B300 / GB300 (Blackwell datacenter)", "Blackwell", True, True, "12.9"),
    KnownArch("11.0", "Jetson T5000 / T4000 (Blackwell)",    "Blackwell", True, True, "13.0"),
    # Blackwell consumer (sm_120/121)
    KnownArch("12.0", "RTX 50xx / RTX PRO Blackwell",        "Blackwell", True, True, "12.8"),
    KnownArch("12.1", "NVIDIA GB10 / DGX Spark",             "Blackwell", True, True, "12.9"),
]


def known_arch_for(cc: str) -> KnownArch | None:
    cc = cc.strip()
    for k in KNOWN_CUDA_ARCHS:
        if k.cc == cc:
            return k
    return None


def cc_to_arch_token(cc: str, *, real: bool = True) -> str:
    """Convert "8.6" → "86-real" (or "86" if real=False)."""
    parts = cc.split(".")
    if len(parts) != 2:
        return cc
    major = parts[0]
    minor = parts[1]
    base = f"{major}{minor}"
    return f"{base}-real" if real else base


def arch_token_with_a(cc: str) -> str:
    """Return "120a-real" for "12.0" if that compute capability supports
    the ``-a`` variant, else the plain "-real" token."""
    parts = cc.split(".")
    if len(parts) != 2:
        return cc
    base = f"{parts[0]}{parts[1]}"
    k = known_arch_for(cc)
    if k and k.has_a_variant:
        return f"{base}a-real"
    return f"{base}-real"


def arch_token_with_f(cc: str) -> str:
    """Return "120f-real" if the compute capability has an ``-f`` family-
    forward variant (Blackwell only as of CUDA 13), else the plain
    "-real" token."""
    parts = cc.split(".")
    if len(parts) != 2:
        return cc
    base = f"{parts[0]}{parts[1]}"
    k = known_arch_for(cc)
    if k and k.has_f_variant:
        return f"{base}f-real"
    return f"{base}-real"


# ─────────────────────────────────────────────────────────────────────────────
# CUDA architecture detection (via torch)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CudaArchInfo:
    """Per-GPU CUDA capability info used to assemble CMAKE_CUDA_ARCHITECTURES."""
    index: int
    name: str
    compute_capability: str   # "8.6", "12.0"
    arch_token: str           # "86-real", "120-real"
    # ``-a`` token (e.g. "120a-real", "90a-real") when the compute capability
    # has an architecture-accelerated variant in nvcc. Both Hopper (sm_90a)
    # and Blackwell (sm_100a / sm_120a / sm_121a) qualify.
    a_variant_token: str | None = None
    family: str = ""          # "Ampere" / "Hopper" / "Blackwell" / ...


def _cuda_arch_info_from_parts(index: int, name: str, cc: str) -> CudaArchInfo | None:
    match = re.match(r"^\s*(\d+)(?:\.(\d+))?\s*$", str(cc or ""))
    if not match:
        return None
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    if major <= 0:
        return None
    normalized_cc = f"{major}.{minor}"
    base = f"{major}{minor}"
    known = known_arch_for(normalized_cc)
    return CudaArchInfo(
        index=index,
        name=name or f"GPU {index}",
        compute_capability=normalized_cc,
        arch_token=f"{base}-real",
        a_variant_token=(f"{base}a-real" if (known and known.has_a_variant) else None),
        family=known.family if known else "unknown",
    )


def cuda_archs_from_gpu_info(gpu_info: dict | None) -> list[CudaArchInfo]:
    """Derive Build-tab CUDA arch records from launcher GPU detection output."""
    if not isinstance(gpu_info, dict) or not gpu_info.get("available"):
        return []
    infos: list[CudaArchInfo] = []
    for fallback_idx, dev in enumerate(gpu_info.get("devices") or []):
        if not isinstance(dev, dict):
            continue
        try:
            idx = int(dev.get("id", fallback_idx))
        except (TypeError, ValueError):
            idx = fallback_idx
        info = _cuda_arch_info_from_parts(
            idx,
            str(dev.get("name") or f"GPU {idx}"),
            str(dev.get("compute_capability") or ""),
        )
        if info is not None:
            infos.append(info)
    return infos


def detect_cuda_archs_from_nvidia_smi(timeout: float = 5.0) -> list[CudaArchInfo]:
    """Return CUDA archs via nvidia-smi without importing torch."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=pci.bus_id,name,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []
    if result.returncode != 0:
        return []

    rows: list[tuple[str, str, str]] = []
    try:
        reader = csv.reader(io.StringIO(result.stdout.strip()))
        for row in reader:
            fields = [part.strip() for part in row]
            if len(fields) < 3:
                continue
            rows.append((fields[0], fields[1], fields[2]))
    except Exception:
        return []

    rows.sort(key=lambda item: item[0])
    infos: list[CudaArchInfo] = []
    for idx, (_bus_id, name, cc) in enumerate(rows):
        info = _cuda_arch_info_from_parts(idx, name, cc)
        if info is not None:
            infos.append(info)
    return infos


def _detect_cuda_archs_from_torch() -> list[CudaArchInfo]:
    """Torch fallback for CUDA arch detection. Call from worker threads."""
    try:
        import torch  # noqa: PLC0415, WPS433
    except Exception:
        return []

    try:
        if not torch.cuda.is_available():
            return []
        n = torch.cuda.device_count()
    except Exception:
        return []

    out: list[CudaArchInfo] = []
    for i in range(n):
        try:
            props = torch.cuda.get_device_properties(i)
        except Exception:
            continue
        info = _cuda_arch_info_from_parts(
            i,
            getattr(props, "name", f"GPU {i}"),
            f"{int(getattr(props, 'major', 0))}.{int(getattr(props, 'minor', 0))}",
        )
        if info is not None:
            out.append(info)
    return out


def detect_cuda_archs(*, allow_torch_fallback: bool = True) -> list[CudaArchInfo]:
    """Return one CudaArchInfo per visible CUDA device.

    nvidia-smi is tried first because it does not initialize CUDA. If that
    fails, torch is used as a fallback; callers on the Tk thread should pass
    ``allow_torch_fallback=False`` or run this function in a worker.

    For compute capabilities with an ``-a`` (architecture-accelerated)
    variant in nvcc — currently sm_90 (Hopper), sm_100/101/103 (Blackwell
    datacenter), sm_120/121/122 (Blackwell consumer) — we surface the
    ``XXa-real`` token too. Building with both tokens means the compiler
    can pick the optimized variant where supported and fall back to the
    plain code path otherwise.
    """
    infos = detect_cuda_archs_from_nvidia_smi()
    if infos or not allow_torch_fallback:
        return infos
    return _detect_cuda_archs_from_torch()


def archs_to_cmake_value(infos: list[CudaArchInfo], *, prefer_a_variant: bool = True) -> str:
    """Collapse a list of CudaArchInfo into a semicolon-separated cmake
    value for CMAKE_CUDA_ARCHITECTURES, deduped while preserving order.

    When ``prefer_a_variant`` is True (default), every detected arch that has
    a ``-a`` variant emits BOTH the ``XXa-real`` and plain ``XX-real`` tokens
    so the build supports the architecture-accelerated features AND keeps a
    portable fallback. Drop ``prefer_a_variant`` to emit plain tokens only.
    """
    seen: set[str] = set()
    out: list[str] = []
    for info in infos:
        tokens: list[str] = []
        if prefer_a_variant and info.a_variant_token:
            tokens.append(info.a_variant_token)
        tokens.append(info.arch_token)
        for t in tokens:
            if t not in seen:
                seen.add(t)
                out.append(t)
    return ";".join(out)


def merge_arch_tokens(*token_lists: str) -> str:
    """Combine multiple ';'-separated arch token strings, dedupe in order."""
    seen: set[str] = set()
    out: list[str] = []
    for s in token_lists:
        if not s:
            continue
        for raw in s.split(";"):
            t = raw.strip()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
    return ";".join(out)


def family_to_tokens(
    family: str,
    *,
    prefer_a_variant: bool = True,
    prefer_f_variant: bool = False,
    include_deprecated: bool = False,
) -> str:
    """Return a ';'-separated arch token list for every known arch in
    a generation (Ampere / Hopper / Blackwell / ...). The two ``prefer_*``
    flags control whether ``-a`` and ``-f`` sibling tokens are emitted in
    addition to the plain ``-real`` token. They are independent — pick one
    or both; defaults emit -a (Hopper+) but not -f.
    """
    out: list[str] = []
    seen: set[str] = set()
    for k in KNOWN_CUDA_ARCHS:
        if k.family.lower() != family.lower():
            continue
        if k.deprecated and not include_deprecated:
            continue
        if prefer_a_variant and k.has_a_variant:
            t = arch_token_with_a(k.cc)
            if t not in seen:
                seen.add(t)
                out.append(t)
        if prefer_f_variant and k.has_f_variant:
            t = arch_token_with_f(k.cc)
            if t not in seen:
                seen.add(t)
                out.append(t)
        t = cc_to_arch_token(k.cc, real=True)
        if t not in seen:
            seen.add(t)
            out.append(t)
    return ";".join(out)


def all_families(*, include_deprecated: bool = True) -> list[str]:
    """Distinct family names in catalogue order. With include_deprecated=False
    only emits families that have at least one non-deprecated arch.
    """
    out: list[str] = []
    for k in KNOWN_CUDA_ARCHS:
        if k.deprecated and not include_deprecated:
            continue
        if k.family not in out:
            out.append(k.family)
    return out


def family_has_only_deprecated(family: str) -> bool:
    for k in KNOWN_CUDA_ARCHS:
        if k.family.lower() == family.lower() and not k.deprecated:
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Toolchain probes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CudaInstall:
    """One CUDA toolkit installation discovered on disk.

    ``root_dir`` is what we'd pass as ``CUDA_TOOLKIT_ROOT_DIR`` (the
    parent of bin/, lib/, include/). cmake derives everything else
    from this when ``CMAKE_CUDA_COMPILER`` is also set.
    """
    version: str                  # "12.8" (parsed from nvcc --version)
    root_dir: str                 # /usr/local/cuda-12.8
    nvcc_path: str                # /usr/local/cuda-12.8/bin/nvcc
    on_path: bool = False         # True if this install is the one in $PATH

    def label(self) -> str:
        """Human-readable string for combobox display."""
        suffix = "  (on PATH)" if self.on_path else ""
        return f"CUDA {self.version}  ·  {self.root_dir}{suffix}"


@dataclass
class ToolchainProbe:
    """Whatever we could discover about the build toolchain on this host."""
    cuda_version: str | None = None      # "12.8"  (selected install)
    nvcc_path: str | None = None         # /usr/local/cuda/bin/nvcc (selected)
    cuda_installs: list[CudaInstall] = field(default_factory=list)  # all detected
    cc_candidates: list[str] = field(default_factory=list)   # ["/usr/bin/gcc-13", ...]
    cxx_candidates: list[str] = field(default_factory=list)
    cmake_path: str | None = None
    cmake_version: str | None = None
    ninja_path: str | None = None
    ninja_version: str | None = None
    ccache_path: str | None = None
    git_path: str | None = None
    git_version: str | None = None


@dataclass(frozen=True)
class ToolInstallPlan:
    """Install command for a missing tool on the current platform."""
    tool_key: str
    tool_label: str
    package_manager: str
    command: str


@dataclass(frozen=True)
class BuildToolStatus:
    """UI-friendly status row for a build prerequisite."""
    key: str
    label: str
    path: str | None = None
    version: str | None = None
    install_plan: ToolInstallPlan | None = None

    @property
    def installed(self) -> bool:
        return bool(self.path)


def _which(name: str) -> str | None:
    return shutil.which(name)


def _run(cmd: list[str], timeout: float = 1.5) -> str | None:
    """Best-effort subprocess capture. Short default timeout — these probes
    run during startup and a misconfigured tool must not block the UI."""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        if proc.returncode != 0:
            return None
        return (proc.stdout or "") + (proc.stderr or "")
    except Exception:
        return None


def _cuda_search_paths() -> list[str]:
    """Glob-style candidate locations for CUDA toolkit installs, by OS.

    Returned as a list of nvcc paths (not roots) so callers can run
    ``nvcc --version`` directly to confirm + extract version.
    """
    candidates: list[str] = []
    seen: set[str] = set()

    def add(p: str) -> None:
        if p and p not in seen:
            seen.add(p)
            candidates.append(p)

    # PATH first.
    on_path = _which("nvcc")
    if on_path:
        add(on_path)

    # Environment variables some toolchains set.
    for env_var in ("CUDA_PATH", "CUDA_HOME", "CUDA_TOOLKIT_ROOT_DIR"):
        v = os.environ.get(env_var)
        if v:
            nvcc = _nvcc_in_root(v)
            if nvcc:
                add(nvcc)

    if sys.platform.startswith("win"):
        # Windows: NVIDIA installs to Program Files by default. Env var
        # names on Windows are case-insensitive but ruff (SIM112) wants the
        # canonical uppercase form.
        prog_files = [
            os.environ.get("PROGRAMFILES", r"C:\Program Files"),
            os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
        ]
        for pf in prog_files:
            base = os.path.join(pf, "NVIDIA GPU Computing Toolkit", "CUDA")
            if os.path.isdir(base):
                try:
                    for entry in sorted(os.listdir(base), reverse=True):
                        nvcc = os.path.join(base, entry, "bin", "nvcc.exe")
                        if os.path.isfile(nvcc):
                            add(nvcc)
                except Exception:
                    pass
    else:
        # Linux/macOS: enumerate common install roots.
        roots: list[str] = []
        for base in (
            "/usr/local",
            "/opt",
            "/Developer/NVIDIA",
            os.path.expanduser("~/cuda"),
        ):
            if not os.path.isdir(base):
                continue
            # ``base`` itself may be a CUDA toolkit root (typical for
            # ``~/cuda`` where the user dropped the toolkit straight in).
            # Without this check we'd skip past ``~/cuda/bin/nvcc`` because
            # the child-enumeration below only matches ``cuda*`` names.
            direct_nvcc = _nvcc_in_root(base)
            if direct_nvcc:
                add(direct_nvcc)
            try:
                for entry in os.listdir(base):
                    full = os.path.join(base, entry)
                    if not os.path.isdir(full):
                        continue
                    # Match: cuda, cuda-*, CUDA-*
                    low = entry.lower()
                    if low == "cuda" or low.startswith("cuda-") or low.startswith("cuda_"):
                        roots.append(full)
            except Exception:
                pass
        # Sort newest-version-looking first (lexicographic descending works for cuda-XX.Y).
        roots.sort(reverse=True)
        for r in roots:
            nvcc = _nvcc_in_root(r)
            if nvcc:
                add(nvcc)
    return candidates


def _nvcc_in_root(root: str) -> str | None:
    """Given a CUDA root, return the nvcc inside it if present."""
    if not root:
        return None
    for name in ("nvcc", "nvcc.exe"):
        p = os.path.join(root, "bin", name)
        if os.path.isfile(p):
            return p
    # Some Mac installs put nvcc directly in root or nest differently — try root/.
    for name in ("nvcc", "nvcc.exe"):
        p = os.path.join(root, name)
        if os.path.isfile(p):
            return p
    return None


def _root_from_nvcc(nvcc_path: str) -> str:
    """Given .../<root>/bin/nvcc, return <root>. Falls back to the nvcc
    file's parent if the bin/ layout isn't present (rare)."""
    parent = os.path.dirname(nvcc_path)
    grand = os.path.dirname(parent)
    if os.path.basename(parent).lower() == "bin" and grand:
        return grand
    return parent


def detect_cuda_installs() -> list[CudaInstall]:
    """Scan the filesystem for every CUDA toolkit install. Each entry is
    de-duped by version + root_dir. Best-effort and bounded — every nvcc
    probe has a short timeout so a misconfigured install can't pin the UI.
    """
    on_path_nvcc = _which("nvcc")
    out: list[CudaInstall] = []
    seen: set[tuple[str, str]] = set()
    for nvcc in _cuda_search_paths():
        if not os.path.isfile(nvcc):
            continue
        text = _run([nvcc, "--version"], timeout=1.5)
        if not text:
            continue
        m = re.search(r"release\s+(\d+\.\d+)", text)
        version = m.group(1) if m else "?"
        root = _root_from_nvcc(nvcc)
        key = (version, root)
        if key in seen:
            continue
        seen.add(key)
        # samefile can raise OSError on broken symlinks or stat permission
        # errors (especially on Windows); fall back to absolute-path compare.
        is_on_path = False
        if on_path_nvcc and os.path.isfile(on_path_nvcc):
            try:
                is_on_path = os.path.samefile(nvcc, on_path_nvcc)
            except OSError:
                is_on_path = os.path.abspath(nvcc) == os.path.abspath(on_path_nvcc)
        out.append(CudaInstall(
            version=version,
            root_dir=root,
            nvcc_path=nvcc,
            on_path=is_on_path,
        ))
    # Sort newest-first by *parsed* version (lexicographic order on the path
    # text mis-ranks e.g. "13.10" vs "13.9"). Installs with an unparseable
    # version sort last so PATH discovery still surfaces them but they never
    # win the auto-pick.
    def _ver_key(inst: CudaInstall) -> tuple[int, int, int]:
        vm = re.match(r"^(\d+)\.(\d+)$", inst.version or "")
        if not vm:
            return (0, 0, 0)
        return (1, int(vm.group(1)), int(vm.group(2)))
    out.sort(key=_ver_key, reverse=True)
    return out


def _detect_cuda() -> tuple[str | None, str | None]:
    """Return (cuda_version, nvcc_path) for the *preferred* install:
    PATH first, then the newest detected install. Used to seed the UI's
    initial selection. See ``detect_cuda_installs`` for the full list.
    """
    installs = detect_cuda_installs()
    if not installs:
        return None, None
    # Prefer the one on PATH.
    for inst in installs:
        if inst.on_path:
            return (None if inst.version == "?" else inst.version), inst.nvcc_path
    # Otherwise the first (newest) found.
    inst = installs[0]
    return (None if inst.version == "?" else inst.version), inst.nvcc_path


def _detect_gcc_candidates() -> tuple[list[str], list[str]]:
    """Return (cc_candidates, cxx_candidates). Cross-platform compiler scan:

    * Linux: gcc-9..15 from PATH, plus stock /usr/bin/gcc.
    * macOS: Homebrew gcc-12/13/14 in /opt/homebrew/bin (Apple Silicon)
      and /usr/local/bin (Intel) — Apple's clang is also surfaced.
    * Windows: stock gcc + MSYS2 / MinGW prefixes if present; MSVC is the
      cmake default so we don't try to enumerate cl.exe versions here.

    Order matters: newer versions first so the UI's first option is the
    most-likely-best for current CUDA toolkits (e.g. gcc-13 for CUDA 12.x).
    """
    cc: list[str] = []
    cxx: list[str] = []

    def add_cc(p: str | None) -> None:
        if p and p not in cc:
            cc.append(p)

    def add_cxx(p: str | None) -> None:
        if p and p not in cxx:
            cxx.append(p)

    # Versioned gcc/g++ binaries — newer first.
    for v in ("15", "14", "13", "12", "11", "10", "9"):
        add_cc(_which(f"gcc-{v}"))
        add_cxx(_which(f"g++-{v}"))

    # macOS Homebrew explicit paths (in case `which` finds Apple's clang first).
    if sys.platform == "darwin":
        for prefix in ("/opt/homebrew/bin", "/usr/local/bin"):
            for v in ("14", "13", "12", "11"):
                p = os.path.join(prefix, f"gcc-{v}")
                if os.path.isfile(p):
                    add_cc(p)
                p = os.path.join(prefix, f"g++-{v}")
                if os.path.isfile(p):
                    add_cxx(p)
        # Apple clang is fine for CPU-only builds.
        for p in ("/usr/bin/clang",):
            if os.path.isfile(p):
                add_cc(p)
        for p in ("/usr/bin/clang++",):
            if os.path.isfile(p):
                add_cxx(p)

    # Windows MSYS2/MinGW prefixes — best-effort.
    if sys.platform.startswith("win"):
        for prefix in (r"C:\msys64\mingw64\bin", r"C:\msys64\ucrt64\bin",
                       r"C:\mingw64\bin"):
            for fname in ("gcc.exe",):
                p = os.path.join(prefix, fname)
                if os.path.isfile(p):
                    add_cc(p)
            for fname in ("g++.exe",):
                p = os.path.join(prefix, fname)
                if os.path.isfile(p):
                    add_cxx(p)

    # Stock unversioned tools last (so versioned ones float to the top).
    add_cc(_which("gcc"))
    add_cxx(_which("g++"))

    return cc, cxx


def probe_toolchain() -> ToolchainProbe:
    """Best-effort scan of the build toolchain. Never raises."""
    probe = ToolchainProbe()

    probe.cuda_installs = detect_cuda_installs()
    if probe.cuda_installs:
        # Pick PATH install if present, else newest detected.
        preferred = next((i for i in probe.cuda_installs if i.on_path),
                         probe.cuda_installs[0])
        probe.cuda_version = preferred.version if preferred.version != "?" else None
        probe.nvcc_path = preferred.nvcc_path
    probe.cc_candidates, probe.cxx_candidates = _detect_gcc_candidates()

    probe.cmake_path = _which("cmake")
    if probe.cmake_path:
        out = _run([probe.cmake_path, "--version"], timeout=1.5)
        if out:
            m = re.search(r"cmake version\s+(\S+)", out)
            if m:
                probe.cmake_version = m.group(1)

    probe.ninja_path = _which("ninja")
    if probe.ninja_path:
        out = _run([probe.ninja_path, "--version"], timeout=1.5)
        if out:
            probe.ninja_version = (out.strip().splitlines() or [""])[0].strip() or None
    probe.ccache_path = _which("ccache")
    probe.git_path = _which("git")
    if probe.git_path:
        out = _run([probe.git_path, "--version"], timeout=1.5)
        if out:
            m = re.search(r"git version\s+(\S+)", out)
            if m:
                probe.git_version = m.group(1)
    return probe


_TOOL_LABELS = {
    "cmake": "CMake",
    "ninja": "Ninja",
    "git": "Git",
}


def install_plan_for_tool(tool_key: str) -> ToolInstallPlan | None:
    """Return an OS-aware install command for ``tool_key`` if we know one."""
    label = _TOOL_LABELS.get(tool_key)
    if label is None:
        return None
    if sys.platform.startswith("linux"):
        return _linux_install_plan(tool_key, label)
    if sys.platform == "darwin":
        return _darwin_install_plan(tool_key, label)
    if sys.platform.startswith("win"):
        return _windows_install_plan(tool_key, label)
    return None


def _linux_install_plan(tool_key: str, label: str) -> ToolInstallPlan | None:
    packages = {
        "apt-get": {"cmake": "cmake", "ninja": "ninja-build", "git": "git"},
        "dnf": {"cmake": "cmake", "ninja": "ninja-build", "git": "git"},
        "yum": {"cmake": "cmake", "ninja": "ninja-build", "git": "git"},
        "pacman": {"cmake": "cmake", "ninja": "ninja", "git": "git"},
        "zypper": {"cmake": "cmake", "ninja": "ninja", "git": "git"},
        "apk": {"cmake": "cmake", "ninja": "ninja-build", "git": "git"},
        "brew": {"cmake": "cmake", "ninja": "ninja", "git": "git"},
    }
    commands = {
        "apt-get": lambda pkg: f"sudo apt-get update && sudo apt-get install -y {pkg}",
        "dnf": lambda pkg: f"sudo dnf install -y {pkg}",
        "yum": lambda pkg: f"sudo yum install -y {pkg}",
        "pacman": lambda pkg: f"sudo pacman -Sy --needed {pkg}",
        "zypper": lambda pkg: f"sudo zypper install -y {pkg}",
        "apk": lambda pkg: f"sudo apk add {pkg}",
        "brew": lambda pkg: f"brew install {pkg}",
    }
    for manager in ("apt-get", "dnf", "yum", "pacman", "zypper", "apk", "brew"):
        if _which(manager) is None:
            continue
        package_name = packages[manager].get(tool_key)
        if not package_name:
            continue
        return ToolInstallPlan(
            tool_key=tool_key,
            tool_label=label,
            package_manager=manager,
            command=commands[manager](package_name),
        )
    return None


def _darwin_install_plan(tool_key: str, label: str) -> ToolInstallPlan | None:
    if _which("brew") is None:
        return None
    package_name = {"cmake": "cmake", "ninja": "ninja", "git": "git"}.get(tool_key)
    if package_name is None:
        return None
    return ToolInstallPlan(
        tool_key=tool_key,
        tool_label=label,
        package_manager="brew",
        command=f"brew install {package_name}",
    )


def _windows_install_plan(tool_key: str, label: str) -> ToolInstallPlan | None:
    winget_ids = {
        "cmake": "Kitware.CMake",
        "ninja": "Ninja-build.Ninja",
        "git": "Git.Git",
    }
    if _which("winget") is not None and tool_key in winget_ids:
        return ToolInstallPlan(
            tool_key=tool_key,
            tool_label=label,
            package_manager="winget",
            command=(
                "winget install "
                f"--id {winget_ids[tool_key]} -e "
                "--accept-package-agreements --accept-source-agreements"
            ),
        )
    packages = {"cmake": "cmake", "ninja": "ninja", "git": "git"}
    package_name = packages.get(tool_key)
    if package_name is None:
        return None
    if _which("choco") is not None:
        return ToolInstallPlan(
            tool_key=tool_key,
            tool_label=label,
            package_manager="choco",
            command=f"choco install {package_name} -y",
        )
    if _which("scoop") is not None:
        return ToolInstallPlan(
            tool_key=tool_key,
            tool_label=label,
            package_manager="scoop",
            command=f"scoop install {package_name}",
        )
    return None


def build_tool_statuses(probe: ToolchainProbe) -> list[BuildToolStatus]:
    """Return installed/missing status rows for key build tools."""
    rows = [
        ("cmake", "CMake", probe.cmake_path, probe.cmake_version),
        ("ninja", "Ninja", probe.ninja_path, probe.ninja_version),
        ("git", "Git", probe.git_path, probe.git_version),
    ]
    out: list[BuildToolStatus] = []
    for key, label, path, version in rows:
        out.append(
            BuildToolStatus(
                key=key,
                label=label,
                path=path,
                version=version,
                install_plan=None if path else install_plan_for_tool(key),
            )
        )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# System resource recommendations
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class JobRecommendation:
    suggested: int       # what we recommend
    cpu_count: int       # raw nproc
    physical_cores: int | None = None
    total_ram_gb: float | None = None
    reason: str = ""     # human-readable rationale shown in the UI


def recommend_jobs() -> JobRecommendation:
    """Pick an interactive-safe build-jobs count.

    A CMake/Ninja build can make the whole desktop feel frozen if it uses
    every logical CPU, and NVCC peak memory per parallel job can be several
    GB for heavy CUDA translation units. Prefer physical cores, reserve a
    little CPU for the shell/desktop, and cap the default at 16 jobs. The
    user can still override this in the UI.
    """
    cpu_count = os.cpu_count() or 1
    physical: int | None = None
    ram_gb: float | None = None

    try:
        import psutil  # type: ignore  # noqa: PLC0415
        physical = psutil.cpu_count(logical=False) or None
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass

    cpu_cap = max(1, cpu_count - 2)
    if physical:
        cpu_cap = min(cpu_cap, physical)
    default_cap = 16
    suggested = min(cpu_cap, default_cap)
    caps = [f"interactive CPU cap={cpu_cap}", f"default max={default_cap}"]

    if ram_gb is not None:
        ram_cap = max(1, int(ram_gb // 4))
        if ram_cap < suggested:
            suggested = ram_cap
        caps.append(f"RAM cap={ram_cap} (≈4 GB/job)")
    reason = ", ".join(caps)

    return JobRecommendation(
        suggested=suggested,
        cpu_count=cpu_count,
        physical_cores=physical,
        total_ram_gb=ram_gb,
        reason=reason,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Default backend source dirs
# ─────────────────────────────────────────────────────────────────────────────

def default_source_dir(backend: str, existing_dir: str) -> str:
    """Return an existing backend dir if present, else suggest a sibling
    path of the launcher home so 'Clone' creates the repo somewhere sane.
    """
    existing_dir = (existing_dir or "").strip()
    if existing_dir and os.path.isdir(existing_dir):
        return existing_dir
    home = str(Path.home())
    folder = "ik_llama.cpp" if backend == "ik_llama" else "llama.cpp"
    return str(Path(home) / "Documents" / "GitHub" / folder)
