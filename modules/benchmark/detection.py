"""Discovery of benchmark binaries across configured builds.

The Benchmark tab does not assume a single "active" build. Instead it
probes every build the launcher knows about — the two backend root dirs
(``llama_cpp_dir`` / ``ik_llama_dir``) and every named build in
``config/build_configs.json`` — and reports which benchmark tools each
one ships. Picking the build then determines the available tools, which
sidesteps having to reason about the currently-selected backend.

Two tools are probed:

* ``llama-bench``       — present in both llama.cpp and ik_llama builds.
* ``llama-sweep-bench`` — ik_llama.cpp only.

The search mirrors ``LaunchManager._find_server_executable`` so a build
laid out for the launcher's server discovery is also found here.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

# Tool identifiers used throughout the benchmark package.
TOOL_LLAMA_BENCH = "llama-bench"
TOOL_SWEEP_BENCH = "llama-sweep-bench"
ALL_TOOLS = (TOOL_LLAMA_BENCH, TOOL_SWEEP_BENCH)

# Relative locations searched under a build root, in priority order. Kept
# in sync with ``LaunchManager._find_server_executable`` (modules/launch.py)
# so anything the server discovery finds, the bench discovery finds too.
_SEARCH_PATHS_REL = (
    Path("."),
    Path("build/bin/Release"),
    Path("build/bin"),
    Path("build"),
    Path("bin"),
    Path("server"),
)


def _exe_names(tool: str) -> list[str]:
    """Candidate filenames for ``tool`` on the current platform."""
    if sys.platform == "win32":
        return [f"{tool}.exe"]
    return [tool]


def find_bench_executable(base_dir: str | Path, tool: str) -> Path | None:
    """Locate ``tool`` under ``base_dir``.

    Returns the resolved path to the first match, or ``None`` if the tool
    isn't present. Never raises — a malformed ``base_dir`` simply yields
    ``None``.
    """
    if not str(base_dir or "").strip():
        return None
    try:
        root = Path(base_dir).expanduser()
    except Exception:
        return None
    names = _exe_names(tool)
    for rel in _SEARCH_PATHS_REL:
        for name in names:
            candidate = root / rel / name
            try:
                if candidate.is_file():
                    return candidate.resolve()
            except OSError:
                continue
    return None


@dataclass
class BuildEntry:
    """One benchmarkable build and the tools it offers."""

    label: str  # human-readable, unique within a discovery pass
    backend: str  # "llama.cpp" | "ik_llama"
    root_dir: str  # the build/backend root the tools were found under
    source: str  # "backend" | "build_config"
    tools: dict[str, str] = field(default_factory=dict)  # tool -> resolved exe path

    @property
    def has_any_tool(self) -> bool:
        return bool(self.tools)

    def tool_path(self, tool: str) -> str | None:
        return self.tools.get(tool)

    def available_tools(self) -> list[str]:
        """Tools present in this build, in canonical order."""
        return [t for t in ALL_TOOLS if t in self.tools]


def _probe_root(label: str, backend: str, root_dir: str, source: str) -> BuildEntry:
    """Build a :class:`BuildEntry` by probing ``root_dir`` for every tool.

    ``llama-sweep-bench`` is only meaningful on ik_llama, but we still probe
    for it unconditionally and let its physical presence decide — a llama.cpp
    tree simply won't contain it, and a custom fork might. The backend label
    is informational.
    """
    entry = BuildEntry(label=label, backend=backend, root_dir=root_dir, source=source)
    for tool in ALL_TOOLS:
        found = find_bench_executable(root_dir, tool)
        if found is not None:
            entry.tools[tool] = str(found)
    return entry


def discover_builds(launcher) -> list[BuildEntry]:
    """Enumerate every build the launcher knows about that ships >=1 bench tool.

    Sources, in order:

    1. ``launcher.llama_cpp_dir`` — the llama.cpp backend root.
    2. ``launcher.ik_llama_dir``  — the ik_llama backend root.
    3. Each named entry in ``config/build_configs.json`` (via the Build tab's
       ``BuildConfigStore``), resolving each config's ``build_dir`` relative
       to its ``source_dir`` when relative.

    De-duplicates by resolved root path so the same directory configured as
    both a backend root and a build config only appears once. Only builds
    with at least one discovered tool are returned. Never raises.
    """
    entries: list[BuildEntry] = []
    seen_roots: set[str] = set()

    def _add(label: str, backend: str, root_dir: str, source: str) -> None:
        if not str(root_dir or "").strip():
            return
        try:
            resolved = str(Path(root_dir).expanduser().resolve(strict=False))
        except Exception:
            return
        if resolved in seen_roots:
            return
        entry = _probe_root(label, backend, root_dir, source)
        if entry.has_any_tool:
            seen_roots.add(resolved)
            entries.append(entry)

    # 1 & 2: backend roots
    for attr, backend, label in (
        ("llama_cpp_dir", "llama.cpp", "llama.cpp (backend dir)"),
        ("ik_llama_dir", "ik_llama", "ik_llama (backend dir)"),
    ):
        var = getattr(launcher, attr, None)
        root = ""
        if var is not None:
            try:
                root = var.get()
            except Exception:
                root = ""
        _add(label, backend, root, "backend")

    # 3: named build configs from the Build tab's store
    for label, backend, root in _iter_build_config_roots(launcher):
        _add(f"{label} (build config)", backend, root, "build_config")

    return entries


def _iter_build_config_roots(launcher):
    """Yield ``(name, backend, resolved_build_dir)`` for each saved build config.

    Best-effort: any failure to import the store, read the config dir, or
    resolve a single entry is swallowed so discovery degrades gracefully to
    just the backend roots.
    """
    try:
        from modules.build.build_persistence import BuildConfigStore
    except Exception:
        return

    config_path = getattr(launcher, "config_path", None)
    if config_path is None:
        return
    try:
        config_dir = Path(config_path).parent
    except Exception:
        return

    try:
        store = BuildConfigStore(config_dir)
        names = store.list_names()
    except Exception:
        return

    for name in names:
        try:
            cfg = store.get(name)
        except Exception:
            cfg = None
        if cfg is None:
            continue
        source_dir = (cfg.source_dir or "").strip()
        build_dir = (cfg.build_dir or "build").strip() or "build"
        if not source_dir:
            continue
        try:
            src = Path(source_dir).expanduser()
            bd = Path(build_dir).expanduser()
            resolved = bd if bd.is_absolute() else (src / bd)
        except Exception:
            continue
        yield name, cfg.backend, str(resolved)
