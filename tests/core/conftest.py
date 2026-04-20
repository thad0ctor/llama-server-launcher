"""Shared fixtures and helpers for the core config/env_vars test suite.

Adds a richer launcher-mock factory that additionally exposes the UI-ish
attributes needed to exercise :meth:`ConfigManager.load_configuration` and
:meth:`ConfigManager.delete_configuration` without a real Tk UI.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable
from unittest.mock import MagicMock

import pytest

# Ensure the repo root is importable.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class FakeVar:
    """Minimal stand-in for ``tk.StringVar`` / ``tk.BooleanVar``.

    Stored value is whatever the caller ``set()``s last. Tests that need to
    observe write-order can inspect ``history``.
    """

    def __init__(self, value=""):
        self._value = value
        self.history = [value]

    def get(self):
        return self._value

    def set(self, v):
        self._value = v
        self.history.append(v)


class FakeListbox:
    """Minimal Listbox stand-in supporting the subset used by ConfigManager."""

    def __init__(self, items=None):
        self._items = list(items) if items else []
        self._selection = ()

    # --- helpers used by tests ---
    def set_items(self, items):
        self._items = list(items)

    def set_selection(self, indices):
        self._selection = tuple(indices)

    # --- tkinter.Listbox surface ---
    def curselection(self):
        return tuple(self._selection)

    def get(self, first, last=None):
        # Tkinter's Listbox.get tolerates a tuple as ``first`` (returns
        # the item at the first element of the tuple) — mirror that so
        # callers that pass curselection() directly don't break.
        if isinstance(first, tuple):
            first = first[0] if first else None
            if first is None:
                return ""
        if last is None:
            if isinstance(first, int):
                return self._items[first]
            return ""
        if last == "end":
            return tuple(self._items[first:])
        return tuple(self._items[first:last + 1])

    def delete(self, first, last=None):
        if last == "end":
            self._items = self._items[:first] if isinstance(first, int) else []
        elif isinstance(first, int):
            del self._items[first]

    def insert(self, index, value):
        if index == "end":
            self._items.append(value)
        else:
            self._items.insert(index, value)

    def selection_clear(self, first, last=None):
        self._selection = ()

    def selection_set(self, index):
        self._selection = (index,)

    def activate(self, index):
        pass

    def see(self, index):
        pass


def build_rich_launcher_mock(
    config_path: Path,
    *,
    saved_configs=None,
    app_settings=None,
    model_dirs=None,
    gpu_devices=None,
    all_templates=None,
    found_models=None,
):
    """Build a launcher mock wired up for both save/load and UI-path tests."""
    launcher = MagicMock()

    # Core data containers
    launcher.config_path = config_path
    launcher.saved_configs = dict(saved_configs) if saved_configs else {}
    launcher.app_settings = dict(app_settings) if app_settings else {
        "selected_gpus": [],
        "gpu_order": [],
    }
    launcher.model_dirs = list(model_dirs) if model_dirs else []
    launcher.detected_gpu_devices = list(gpu_devices) if gpu_devices else []
    launcher.custom_parameters_list = []
    launcher.manual_gpu_list = []
    launcher.gpu_vars = []
    launcher.physical_cores = 4
    launcher.logical_cores = 8
    launcher.fit_ctx_synced = True
    launcher._all_templates = dict(all_templates) if all_templates else {"ChatML": "tpl"}
    launcher.found_models = dict(found_models) if found_models else {}
    launcher.current_model_analysis = {}

    # Simulated Tk variables used across ConfigManager
    for name, default in [
        ("llama_cpp_dir", ""),
        ("ik_llama_dir", ""),
        ("venv_dir", ""),
        ("model_path", ""),
        ("selected_mmproj_path", ""),
        ("cache_type_k", "f16"),
        ("cache_type_v", "f16"),
        ("threads", "8"),
        ("threads_batch", "8"),
        ("batch_size", "512"),
        ("ubatch_size", "512"),
        ("n_gpu_layers", "0"),
        ("tensor_split", ""),
        ("main_gpu", "0"),
        ("prio", "0"),
        ("temperature", "0.8"),
        ("min_p", "0.05"),
        ("seed", "-1"),
        ("n_predict", "-1"),
        ("n_cpu_moe", ""),
        ("fit_ctx", ""),
        ("fit_target", "1024"),
        ("template_source", "default"),
        ("predefined_template_name", ""),
        ("custom_template_string", ""),
        ("host", "127.0.0.1"),
        ("port", "8080"),
        ("backend_selection", "llama.cpp"),
        ("parallel", "1"),
        ("config_name", ""),
        ("manual_gpu_mode", False),
        ("manual_gpu_count", "1"),
        ("manual_gpu_vram", "8.0"),
        ("manual_model_mode", False),
        ("manual_model_layers", "32"),
        ("manual_model_size_gb", "7.0"),
    ]:
        setattr(launcher, name, FakeVar(default))

    for name in ("no_mmap", "flash_attn", "mlock", "no_kv_offload", "ignore_eos",
                 "cpu_moe", "mmproj_enabled", "fit_enabled", "jinja_enabled"):
        setattr(launcher, name, FakeVar(False))

    launcher.ctx_size = FakeVar(2048)
    launcher.n_gpu_layers_int = FakeVar(0)
    launcher.max_gpu_layers = FakeVar(0)

    # Listboxes used by load / delete / update_config_listbox flows
    launcher.config_listbox = FakeListbox()
    launcher.model_listbox = FakeListbox()

    # Root.after(delay, fn, *args) -> call fn synchronously so tests see effects.
    def _after(delay, fn=None, *args):
        if fn is not None:
            fn(*args)
    launcher.root = MagicMock()
    launcher.root.after = _after

    # Sub-manager stubs
    launcher.env_vars_manager = MagicMock()
    launcher.env_vars_manager.save_to_config.return_value = {
        "environmental_variables": {"enabled": False, "predefined": {}, "custom": []}
    }
    launcher.ik_llama_tab = MagicMock()
    launcher.ik_llama_tab.save_to_config.return_value = {}

    # Launcher methods exercised by load_configuration
    launcher._sync_ctx_display = MagicMock()
    launcher._update_fit_fields_state = MagicMock()
    launcher._update_custom_parameters_listbox = MagicMock()
    launcher._update_gpu_checkboxes = MagicMock()
    launcher._set_gpu_layers = MagicMock()
    launcher._reset_model_info_display = MagicMock()
    launcher._reset_gpu_layer_controls = MagicMock()
    launcher._update_recommendations = MagicMock()
    launcher._select_model_in_listbox = MagicMock()
    launcher._save_configs = MagicMock()
    launcher._update_config_listbox = MagicMock()

    return launcher


@pytest.fixture
def rich_launcher_factory():
    """Factory for a richer launcher mock than the one in ``test_config.py``."""
    def _factory(config_path: Path, **kwargs):
        return build_rich_launcher_mock(config_path, **kwargs)
    return _factory
