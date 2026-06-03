from __future__ import annotations

import sys
from tkinter import ttk
from unittest.mock import MagicMock

import pytest

from modules import terminal_launcher
from modules import venv_manager
from modules.settings_tab import SettingsTab


# These fixtures hard-code the POSIX venv layout (``bin/python``) and assert
# ``rm -rf`` in the generated remove command. The SettingsTab code itself is
# cross-platform, so on Windows we skip the layout-dependent cases rather
# than fail spuriously.
posix_only = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="POSIX venv layout (bin/python, rm -rf) — Windows uses Scripts/ and rmdir.",
)


@pytest.fixture
def settings_tab(launcher_stub):
    return SettingsTab(launcher_stub)


def test_init_reuses_launcher_venv_var(launcher_stub):
    launcher_stub.venv_dir.set("/tmp/existing-venv")

    tab = SettingsTab(launcher_stub)

    assert tab.venv_dir_var is launcher_stub.venv_dir
    assert tab.venv_dir_var.get() == "/tmp/existing-venv"


def test_current_venv_info_blank_uses_repo_default(settings_tab):
    settings_tab.venv_dir_var.set("")

    info = settings_tab._current_venv_info()

    assert info.uses_default is True
    assert info.effective_dir == venv_manager.default_venv_dir(repo_dir=settings_tab.repo_dir).resolve()


@posix_only
def test_current_active_venv_path_blank_uses_default_only_when_real_venv(settings_tab, tmp_path):
    settings_tab.repo_dir = tmp_path
    bindir = tmp_path / "venv" / "bin"
    bindir.mkdir(parents=True)
    (bindir / "python").write_text("", encoding="utf-8")
    settings_tab.venv_dir_var.set("")

    assert settings_tab._current_active_venv_path() == str((tmp_path / "venv").resolve())


@posix_only
def test_current_active_venv_path_relative_matches_launcher_resolution(settings_tab, tmp_path):
    settings_tab.repo_dir = tmp_path
    bindir = tmp_path / "envs" / "custom" / "bin"
    bindir.mkdir(parents=True)
    (bindir / "python").write_text("", encoding="utf-8")
    settings_tab.venv_dir_var.set("envs/custom")

    assert settings_tab._current_active_venv_path() == str((tmp_path / "envs" / "custom").resolve())


def test_create_venv_uses_default_repo_path_when_blank(settings_tab, monkeypatch):
    monkeypatch.setattr(settings_tab, "_schedule_venv_dependency_probe", lambda: None)
    launch_mock = MagicMock()
    info_mock = MagicMock()
    error_mock = MagicMock()
    monkeypatch.setattr(terminal_launcher, "open_command_in_terminal", launch_mock)
    monkeypatch.setattr("modules.settings_tab.messagebox.showinfo", info_mock)
    monkeypatch.setattr("modules.settings_tab.messagebox.showerror", error_mock)
    settings_tab.venv_dir_var.set("")

    settings_tab._on_create_venv()

    expected = venv_manager.default_venv_dir(repo_dir=settings_tab.repo_dir)
    assert settings_tab.venv_dir_var.get() == str(expected)
    launch_mock.assert_called_once()
    assert launch_mock.call_args.kwargs["cwd"] == settings_tab.repo_dir
    assert str(expected) in launch_mock.call_args.args[0]
    assert info_mock.called
    error_mock.assert_not_called()


def test_remove_venv_requires_existing_directory(settings_tab, monkeypatch):
    info_mock = MagicMock()
    launch_mock = MagicMock()
    monkeypatch.setattr("modules.settings_tab.messagebox.showinfo", info_mock)
    monkeypatch.setattr(terminal_launcher, "open_command_in_terminal", launch_mock)
    settings_tab.venv_dir_var.set("/tmp/does-not-exist")

    settings_tab._on_remove_venv()

    assert info_mock.called
    launch_mock.assert_not_called()


@posix_only
def test_remove_venv_opens_terminal_after_confirmation(settings_tab, monkeypatch, tmp_path):
    venv_dir = tmp_path / "my env"
    # _on_remove_venv now requires the target to look like a venv (so it can't
    # rm -rf an arbitrary directory the user typed). Lay out a minimal POSIX
    # venv so the success path still runs.
    bindir = venv_dir / "bin"
    bindir.mkdir(parents=True)
    (bindir / "python").write_text("", encoding="utf-8")
    launch_mock = MagicMock()
    monkeypatch.setattr(settings_tab, "_schedule_venv_dependency_probe", lambda: None)
    monkeypatch.setattr("modules.settings_tab.messagebox.askyesno", lambda *a, **kw: True)
    monkeypatch.setattr("modules.settings_tab.messagebox.showerror", MagicMock())
    monkeypatch.setattr(terminal_launcher, "open_command_in_terminal", launch_mock)
    settings_tab.venv_dir_var.set(str(venv_dir))

    settings_tab._on_remove_venv()

    launch_mock.assert_called_once()
    assert launch_mock.call_args.kwargs["cwd"] == venv_dir.parent
    assert "rm -rf" in launch_mock.call_args.args[0]


def test_install_dependency_requires_detected_venv(settings_tab, monkeypatch):
    dep = venv_manager.MANAGED_DEPENDENCIES[0]
    error_mock = MagicMock()
    launch_mock = MagicMock()
    monkeypatch.setattr("modules.settings_tab.messagebox.showerror", error_mock)
    monkeypatch.setattr(terminal_launcher, "open_command_in_terminal", launch_mock)
    settings_tab.venv_dir_var.set("/tmp/missing-venv")

    settings_tab._on_install_dependency(dep)

    error_mock.assert_called_once()
    launch_mock.assert_not_called()


@posix_only
def test_install_dependency_opens_terminal_for_existing_venv(settings_tab, monkeypatch, tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir(parents=True)
    python = bindir / "python"
    python.write_text("", encoding="utf-8")
    dep = venv_manager.MANAGED_DEPENDENCIES[3]
    launch_mock = MagicMock()
    monkeypatch.setattr(settings_tab, "_schedule_venv_dependency_probe", lambda: None)
    monkeypatch.setattr("modules.settings_tab.messagebox.showerror", MagicMock())
    monkeypatch.setattr(terminal_launcher, "open_command_in_terminal", launch_mock)
    settings_tab.venv_dir_var.set(str(tmp_path))

    settings_tab._on_install_dependency(dep)

    launch_mock.assert_called_once()
    assert "pip install huggingface_hub" in launch_mock.call_args.args[0]


@posix_only
def test_remove_dependency_opens_terminal_for_existing_venv(settings_tab, monkeypatch, tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir(parents=True)
    python = bindir / "python"
    python.write_text("", encoding="utf-8")
    dep = venv_manager.MANAGED_DEPENDENCIES[2]
    launch_mock = MagicMock()
    monkeypatch.setattr(settings_tab, "_schedule_venv_dependency_probe", lambda: None)
    monkeypatch.setattr("modules.settings_tab.messagebox.showerror", MagicMock())
    monkeypatch.setattr(terminal_launcher, "open_command_in_terminal", launch_mock)
    settings_tab.venv_dir_var.set(str(tmp_path))

    settings_tab._on_remove_dependency(dep)

    launch_mock.assert_called_once()
    assert "pip uninstall -y psutil" in launch_mock.call_args.args[0]


def test_format_dependency_status_includes_version_and_torch_note():
    dep = venv_manager.MANAGED_DEPENDENCIES[1]
    status = venv_manager.DependencyStatus(
        dependency=dep,
        available=True,
        version="2.7.0",
    )

    text = SettingsTab._format_dependency_status(dep, status)

    assert "✓ Installed 2.7.0." in text
    assert "Optional." in text
    assert "GPU-specific wheels" in text


def test_dependency_buttons_follow_installed_state(settings_tab):
    settings_tab._venv_dependencies_frame = ttk.Frame(settings_tab.root)
    settings_tab._current_venv_info = lambda: MagicMock(looks_like_venv=True)
    settings_tab._current_active_venv_path = lambda: "/tmp/test-venv"
    statuses = [
        venv_manager.DependencyStatus(
            dependency=venv_manager.MANAGED_DEPENDENCIES[0],
            available=True,
            version="2.32.0",
        ),
        venv_manager.DependencyStatus(
            dependency=venv_manager.MANAGED_DEPENDENCIES[1],
            available=False,
            error="not installed",
        ),
    ]

    settings_tab._rebuild_dependency_rows(statuses)

    buttons = [
        child for child in settings_tab._venv_dependencies_frame.winfo_children()
        if isinstance(child, ttk.Frame)
        for grandchild in child.winfo_children()
        if isinstance(grandchild, ttk.Button)
    ]
    button_state_by_text = {(button.cget("text"), index): str(button.cget("state")) for index, button in enumerate(buttons)}

    assert button_state_by_text[("Install", 0)] == "disabled"
    assert button_state_by_text[("Remove", 1)] == "normal"
    assert button_state_by_text[("Install", 2)] == "normal"
    assert button_state_by_text[("Remove", 3)] == "disabled"
