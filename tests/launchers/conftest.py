"""Shared fixtures for ``tests/launchers``.

Intentionally mirrors the fixture scaffolding used by the existing
``test_launch.py`` but exposes it package-wide so the new test modules
(``test_launch_server.py``, ``test_shell_safety.py``, ``test_bugfixes.py``,
``test_robustness.py``) can reuse it without duplicating 90 lines of
``MagicMock`` wiring per file.

Nothing here modifies the root ``tests/conftest.py``.
"""

from __future__ import annotations

import os
import sys
import tkinter as tk
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# Ensure repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# NOTE: The root ``tests/conftest.py`` already defines a session-scoped
# ``tk_root`` fixture. We rely on that so a single hidden Tk root is shared
# across the entire test session. A module-scoped fixture with the same
# name in ``test_launch.py`` currently takes precedence for that module;
# here we pick up the session one for the sibling test files.


def _mk(cls, value):
    """Create a Tk variable of type ``cls`` with an initial value."""
    v = cls()
    v.set(value)
    return v


@pytest.fixture()
def built_tree(tmp_path):
    """Creates a fake llama.cpp build tree with an executable and a
    dummy model.gguf. Returns paths dict.
    """
    base = tmp_path / "llama_cpp_root"
    bin_dir = base / "build" / "bin"
    bin_dir.mkdir(parents=True)

    exe_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    exe = bin_dir / exe_name
    exe.write_text("#!/bin/sh\necho fake llama-server\n")
    try:
        os.chmod(exe, 0o755)
    except OSError:
        pass

    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF" + b"\x00" * 16)

    return {"base": base, "exe": exe, "model": model, "tmp_path": tmp_path}


@pytest.fixture()
def launcher_mock(tk_root, built_tree):
    """Fully-populated launcher stand-in.

    Pre-seeded to llama.cpp defaults so ``build_cmd()`` produces a minimal
    command — individual tests tweak only what they care about.
    """
    paths = built_tree
    m = MagicMock()

    m.backend_selection = _mk(tk.StringVar, "llama.cpp")
    m.llama_cpp_dir = _mk(tk.StringVar, str(paths["base"]))
    m.ik_llama_dir = _mk(tk.StringVar, str(paths["base"]))
    m.venv_dir = _mk(tk.StringVar, "")

    m.model_path = _mk(tk.StringVar, str(paths["model"]))
    m.mmproj_enabled = _mk(tk.BooleanVar, False)
    m.selected_mmproj_path = _mk(tk.StringVar, "")

    m.cache_type_k = _mk(tk.StringVar, "f16")
    m.cache_type_v = _mk(tk.StringVar, "f16")

    m.threads = _mk(tk.StringVar, "")
    m.logical_cores = 8
    m.threads_batch = _mk(tk.StringVar, "")
    m.batch_size = _mk(tk.StringVar, "")
    m.ubatch_size = _mk(tk.StringVar, "")

    m.ctx_size = _mk(tk.IntVar, 2048)
    m.seed = _mk(tk.StringVar, "-1")
    m.temperature = _mk(tk.StringVar, "0.8")
    m.min_p = _mk(tk.StringVar, "0.05")

    m.tensor_split = _mk(tk.StringVar, "")
    m.n_gpu_layers = _mk(tk.StringVar, "0")
    m.main_gpu = _mk(tk.StringVar, "0")
    m.flash_attn = _mk(tk.BooleanVar, False)

    m.fit_enabled = _mk(tk.BooleanVar, False)
    m.fit_ctx = _mk(tk.StringVar, "")
    m.fit_target = _mk(tk.StringVar, "")

    m.no_mmap = _mk(tk.BooleanVar, False)
    m.mlock = _mk(tk.BooleanVar, False)
    m.no_kv_offload = _mk(tk.BooleanVar, False)

    m.prio = _mk(tk.StringVar, "0")
    m.parallel = _mk(tk.StringVar, "1")

    m.cpu_moe = _mk(tk.BooleanVar, False)
    m.n_cpu_moe = _mk(tk.StringVar, "")

    m.ignore_eos = _mk(tk.BooleanVar, False)
    m.n_predict = _mk(tk.StringVar, "-1")

    m.host = _mk(tk.StringVar, "127.0.0.1")
    m.port = _mk(tk.StringVar, "8080")

    m.template_source = _mk(tk.StringVar, "default")
    m.current_template_display = _mk(tk.StringVar, "")
    m.jinja_enabled = _mk(tk.BooleanVar, False)

    m.custom_parameters_list = []
    m.get_ordered_selected_gpus = MagicMock(return_value=[])
    m.gpu_info = {"device_count": 0}

    m.model_listbox.curselection = MagicMock(return_value=())
    m.found_models = {}

    m.ik_llama_tab.get_ik_llama_flags = MagicMock(return_value=[])
    m.env_vars_manager.get_enabled_env_vars = MagicMock(return_value={})

    return m


@pytest.fixture()
def manager(launcher_mock):
    """``LaunchManager`` bound to a fully-populated ``launcher_mock``."""
    from modules.launch import LaunchManager
    return LaunchManager(launcher_mock)
