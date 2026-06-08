"""Shared fixtures + helpers for the ``tests/build`` suite.

The two build-tab tests (``test_build_tab_sync.py`` and
``test_build_tool_install.py``) used to redefine the same
``_make_launcher`` / ``_make_build_tab`` pair. Each new spec-tab var
or backend pref required updating the helper in two places; one of
them inevitably went stale during refactors. They now live here and
are imported from both files.
"""

from __future__ import annotations

from types import SimpleNamespace

import tkinter as tk

from modules.build.build_tab import BuildTab


def make_launcher(tk_root, tmp_path):
    """Build a minimal SimpleNamespace launcher that ``BuildTab`` can
    consume in unit tests. ``backend_selection`` wires up the same
    ``current_backend_dir`` sync the real launcher does so trace-based
    UI behaviour can be exercised without booting the full app.
    """
    app_settings: dict = {}
    launcher = SimpleNamespace(
        root=tk_root,
        config_path=str(tmp_path / "launcher_config.json"),
        app_settings=app_settings,
        backend_selection=tk.StringVar(value="llama.cpp"),
        current_backend_dir=tk.StringVar(value="/repos/main-llama"),
        llama_cpp_dir=tk.StringVar(value="/repos/main-llama"),
        ik_llama_dir=tk.StringVar(value="/repos/main-ik"),
    )

    def sync_current_backend_dir(*_args):
        if launcher.backend_selection.get() == "ik_llama":
            launcher.current_backend_dir.set(launcher.ik_llama_dir.get())
        else:
            launcher.current_backend_dir.set(launcher.llama_cpp_dir.get())

    launcher.backend_selection.trace_add("write", sync_current_backend_dir)
    return launcher


def make_build_tab(tk_root, tmp_path, monkeypatch):
    """Construct a ``BuildTab`` with ``tk_root.after*`` stubbed so the
    tab can be instantiated without an event loop.
    """
    monkeypatch.setattr(tk_root, "after", lambda *_args: "after-id")
    monkeypatch.setattr(tk_root, "after_idle", lambda *_args: "idle-id")
    monkeypatch.setattr(tk_root, "after_cancel", lambda *_args: None)
    launcher = make_launcher(tk_root, tmp_path)
    return launcher, BuildTab(launcher)
