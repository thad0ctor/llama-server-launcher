from types import SimpleNamespace

import tkinter as tk

from modules.build.build_tab import BuildTab


def _make_launcher(tk_root, tmp_path):
    app_settings = {}
    return SimpleNamespace(
        root=tk_root,
        config_path=str(tmp_path / "launcher_config.json"),
        app_settings=app_settings,
        backend_selection=tk.StringVar(value="llama.cpp"),
        current_backend_dir=tk.StringVar(value="/repos/main-llama"),
        llama_cpp_dir=tk.StringVar(value="/repos/main-llama"),
        ik_llama_dir=tk.StringVar(value="/repos/main-ik"),
    )


def _make_build_tab(tk_root, tmp_path, monkeypatch):
    monkeypatch.setattr(tk_root, "after", lambda *_args: "after-id")
    monkeypatch.setattr(tk_root, "after_idle", lambda *_args: "idle-id")
    monkeypatch.setattr(tk_root, "after_cancel", lambda *_args: None)
    launcher = _make_launcher(tk_root, tmp_path)
    return launcher, BuildTab(launcher)


def test_main_active_root_dir_updates_build_source(tk_root, tmp_path, monkeypatch):
    launcher, tab = _make_build_tab(tk_root, tmp_path, monkeypatch)

    launcher.current_backend_dir.set("/repos/llama-new")

    assert tab.var_source_dir.get() == "/repos/llama-new"


def test_build_source_updates_active_main_root_dir(tk_root, tmp_path, monkeypatch):
    launcher, tab = _make_build_tab(tk_root, tmp_path, monkeypatch)

    tab.var_source_dir.set("/repos/build-llama")

    assert launcher.llama_cpp_dir.get() == "/repos/build-llama"
    assert launcher.current_backend_dir.get() == "/repos/build-llama"
    assert launcher.app_settings["last_llama_cpp_dir"] == "/repos/build-llama"


def test_build_source_updates_inactive_backend_dir_only(tk_root, tmp_path, monkeypatch):
    launcher, tab = _make_build_tab(tk_root, tmp_path, monkeypatch)

    tab.var_backend.set("ik_llama")
    tab.var_source_dir.set("/repos/build-ik")

    assert launcher.ik_llama_dir.get() == "/repos/build-ik"
    assert launcher.current_backend_dir.get() == "/repos/main-llama"
    assert launcher.app_settings["last_ik_llama_dir"] == "/repos/build-ik"


def test_main_backend_switch_uses_matching_backend_root(tk_root, tmp_path, monkeypatch):
    launcher, tab = _make_build_tab(tk_root, tmp_path, monkeypatch)

    launcher.ik_llama_dir.set("/repos/ik-selected")
    launcher.backend_selection.set("ik_llama")

    assert tab.var_backend.get() == "ik_llama"
    assert tab.var_source_dir.get() == "/repos/ik-selected"
