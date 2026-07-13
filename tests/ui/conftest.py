"""Shared fixtures for UI-cluster tests.

Reuses the session-scoped ``tk_root`` fixture from ``tests/conftest.py`` and
adds helpers tailored to the settings / about / theme tabs. Nothing here
instantiates real launcher widget trees — that's e2e territory and would
require a full ``LlamaCppLauncher`` plus a display.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# Defensive sys.path setup — ``tests/conftest.py`` does this already, but if a
# user runs ``pytest tests/ui/ -v`` with a virtualenv cwd the import order can
# surface this module before the parent conftest.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def launcher_stub(tk_root, tmp_path):
    """Minimal object-shape that ``SettingsTab`` expects from ``launcher``.

    ``SettingsTab.__init__`` reads ``launcher.root`` and
    ``launcher.app_settings``, and ``_persist_ui_settings`` calls
    ``launcher._save_configs``. We stub just those to keep tests focused.
    """
    stub = MagicMock()
    stub.root = tk_root
    # Seed ``repo_dir`` from the test's ``tmp_path`` so SettingsTab's
    # ``__init__`` doesn't fall back to ``venv_manager.launcher_repo_dir()``
    # (the real launcher checkout). On a local run that fallback
    # would let blank-path tests probe / mutate the actual
    # ``./venv`` next to the source code, which can flip
    # ``looks_like_venv`` and trigger unrelated confirmation dialogs.
    # Each test gets its own pristine ``tmp_path``, so isolation
    # is guaranteed.
    stub.repo_dir = tmp_path
    # A fresh dict per test so cross-test mutation can't leak.
    stub.app_settings = {}
    import tkinter as tk  # noqa: PLC0415

    stub.venv_dir = tk.StringVar(master=tk_root, value="")
    stub._save_configs = MagicMock()
    return stub
