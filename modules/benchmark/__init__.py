"""Benchmark tab package.

Runs ``llama-bench`` and ``llama-sweep-bench`` (ik_llama-only) with a
parameter-sweep matrix, streams output into an in-app console, parses the
results into a table, and exports results / configs / runnable scripts.

Modules:
  detection        - Probe configured builds for the bench binaries.
  matrix           - Lever catalogue, value expansion, command building
                     (native comma-lists for llama-bench, cartesian product
                     for llama-sweep-bench).
  bench_runner     - Sequential subprocess runner; streams stderr to the
                     console while capturing stdout per step for parsing.
  results          - Parse tool output into rows; export CSV/JSON/Markdown.
  bench_persistence- Save/load named sweep configs to config/bench_configs.json.
  bench_script     - Export a run as a portable .sh / .ps1 script.
  bench_tab        - The Tk Notebook tab tying it together.
"""

from importlib import import_module

__all__ = ["BenchmarkTab"]


def __getattr__(name: str):
    if name == "BenchmarkTab":
        return import_module(".bench_tab", __name__).BenchmarkTab
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
