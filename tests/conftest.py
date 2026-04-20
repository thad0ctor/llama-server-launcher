"""Shared pytest fixtures for llama-server-launcher tests.

Provides:
- ``repo_root``:            path to the project root so tests can import ``modules.*``.
- ``project_config_dir``:   path to the live ``config/`` directory (read-only).
- ``fixtures_config_dir``:  path to ``tests/fixtures/config`` for per-test JSON samples.
- ``sample_config_dict``:   a minimal but realistic single launcher config dict.
- ``sample_configs_file``:  factory that writes a valid configs-file payload to a
                            tmp path and returns it.
- ``tk_root``:              a single hidden ``tkinter.Tk`` root reused across tests
                            that need to instantiate ``tk.BooleanVar``/``tk.StringVar``.
                            Skipped cleanly when no DISPLAY is available.

All fixtures are defined here (rather than in a module-local conftest) so sibling
test suites under ``tests/launchers`` and ``tests/system`` can reuse them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import pytest


# Ensure the project root is on sys.path so tests can import the ``modules`` package
# regardless of the cwd pytest is invoked from.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the project root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def project_config_dir(repo_root: Path) -> Path:
    """Absolute path to the live ``config/`` directory (JSON files shipped with repo)."""
    return repo_root / "config"


@pytest.fixture(scope="session")
def fixtures_config_dir() -> Path:
    """Absolute path to ``tests/fixtures/config`` for test-local JSON samples."""
    return Path(__file__).resolve().parent / "fixtures" / "config"


@pytest.fixture
def sample_config_dict() -> dict:
    """A representative single launcher configuration dict.

    Mirrors the shape produced by :meth:`ConfigManager.current_cfg` and consumed
    by :meth:`ConfigManager.load_configuration`.
    """
    return {
        "llama_cpp_dir": "/opt/llama.cpp",
        "ik_llama_dir": "",
        "venv_dir": "/opt/llama.cpp/venv",
        "model_path": "/models/test-7b.gguf",
        "selected_mmproj_path": "",
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "threads": "8",
        "threads_batch": "8",
        "batch_size": "512",
        "ubatch_size": "512",
        "n_gpu_layers": "32",
        "no_mmap": False,
        "prio": "0",
        "temperature": "0.8",
        "min_p": "0.05",
        "ctx_size": 4096,
        "seed": "-1",
        "flash_attn": True,
        "tensor_split": "",
        "main_gpu": "0",
        "mlock": False,
        "no_kv_offload": False,
        "host": "127.0.0.1",
        "port": "8080",
        "backend_selection": "llama.cpp",
        "ignore_eos": False,
        "n_predict": "-1",
        "cpu_moe": False,
        "n_cpu_moe": "",
        "parallel": "1",
        "mmproj_enabled": False,
        "fit_enabled": True,
        "fit_ctx": "4096",
        "fit_ctx_synced": True,
        "fit_target": "1024",
        "template_source": "default",
        "predefined_template_name": "ChatML",
        "custom_template_string": "",
        "jinja_enabled": False,
        "custom_parameters": ["--no-warmup"],
        "gpu_indices": [0],
        "gpu_order": [0],
        "environmental_variables": {
            "enabled": False,
            "predefined": {},
            "custom": [],
        },
    }


@pytest.fixture
def sample_configs_file(
    tmp_path: Path, sample_config_dict: dict
) -> Callable[..., Path]:
    """Factory that writes a configs-file payload to tmp and returns its path.

    Usage::

        cfg_path = sample_configs_file(name="my_config", cfg=custom_dict)

    Parameters
    ----------
    name:
        Key under which ``cfg`` is stored. Defaults to ``"test_config"``.
    cfg:
        Configuration dict (defaults to a copy of ``sample_config_dict``).
    app_settings:
        Optional ``app_settings`` block. Defaults to a minimal, valid one.
    filename:
        Optional file name (defaults to ``"configs.json"``).
    """

    def _factory(
        name: str = "test_config",
        cfg: dict | None = None,
        app_settings: dict | None = None,
        filename: str = "configs.json",
    ) -> Path:
        payload: dict[str, Any] = {
            "configs": {name: cfg if cfg is not None else dict(sample_config_dict)},
            "app_settings": app_settings
            if app_settings is not None
            else {
                "last_llama_cpp_dir": "",
                "last_venv_dir": "",
                "last_model_path": "",
                "selected_mmproj_path": "",
                "model_dirs": [],
                "model_list_height": 8,
                "selected_gpus": [],
                "gpu_order": [],
                "custom_parameters": [],
                "host": "127.0.0.1",
                "port": "8080",
                "ui_theme_mode": "auto",
                "ui_theme_name": "",
                "ui_font_family": "",
                "ui_font_size": 0,
            },
        }
        path = tmp_path / filename
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    return _factory


@pytest.fixture(scope="session")
def tk_root():
    """Hidden Tk root shared across tests that need ``tk.*Var`` instances.

    Skips the test when no display/tk runtime is available rather than failing
    with an obscure tkinter error.
    """
    try:
        import tkinter as tk
    except ImportError:  # pragma: no cover - tkinter is in stdlib
        pytest.skip("tkinter is not available")
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - headless CI
        pytest.skip(f"Tk root unavailable: {exc}")
    root.withdraw()
    yield root
    try:
        root.destroy()
    except Exception:
        pass
