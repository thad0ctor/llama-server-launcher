"""Build tab package.

Subdivides the build feature into:
  detection      - CUDA arch detection (via torch), nvcc/gcc/ninja/ccache probes,
                   system resource recommendations (jobs count, RAM-aware caps).
  cmake_flags    - Authoritative flag schema (groups, defaults, help, per-backend
                   visibility); preset model that serializes to/from JSON.
  build_runner   - Git clone / checkout / pull, fetch-upstream comparison,
                   cmake configure + cmake --build, line-streamed output via
                   a thread-safe queue. Also emits an equivalent .sh script.
  build_persistence
                 - Save/load/delete named build configs to
                   config/build_configs.json.
  build_tab      - Tk Notebook tab UI binding everything together, with
                   detach-to-Toplevel support and an update banner.
"""

from importlib import import_module

__all__ = ["BuildTab"]


def __getattr__(name: str):
    if name == "BuildTab":
        return import_module(".build_tab", __name__).BuildTab
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
