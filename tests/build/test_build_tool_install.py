from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import tkinter as tk

from modules import terminal_launcher
from modules.build import detection
from modules.build.build_persistence import BuildConfig
from modules.build.build_tab import BuildTab, DEFAULT_GENERATOR_LABEL


def _make_launcher(tk_root, tmp_path):
    app_settings = {}
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


def _make_build_tab(tk_root, tmp_path, monkeypatch):
    monkeypatch.setattr(tk_root, "after", lambda *_args: "after-id")
    monkeypatch.setattr(tk_root, "after_idle", lambda *_args: "idle-id")
    monkeypatch.setattr(tk_root, "after_cancel", lambda *_args: None)
    launcher = _make_launcher(tk_root, tmp_path)
    return launcher, BuildTab(launcher)


def test_linux_install_plan_uses_apt_for_missing_ninja(monkeypatch):
    monkeypatch.setattr(detection.sys, "platform", "linux")
    monkeypatch.setattr(
        detection,
        "_which",
        lambda name: f"/usr/bin/{name}" if name == "apt-get" else None,
    )

    plan = detection.install_plan_for_tool("ninja")

    assert plan is not None
    assert plan.package_manager == "apt-get"
    assert plan.command == "sudo apt-get update && sudo apt-get install -y ninja-build"


def test_windows_install_plan_uses_winget_for_cmake(monkeypatch):
    monkeypatch.setattr(detection.sys, "platform", "win32")
    monkeypatch.setattr(
        detection,
        "_which",
        lambda name: rf"C:\Tools\{name}.exe" if name == "winget" else None,
    )

    plan = detection.install_plan_for_tool("cmake")

    assert plan is not None
    assert plan.package_manager == "winget"
    assert "Kitware.CMake" in plan.command
    assert "--accept-package-agreements" in plan.command


def test_build_tool_statuses_include_install_plan_for_missing_tools(monkeypatch):
    monkeypatch.setattr(detection.sys, "platform", "linux")
    monkeypatch.setattr(
        detection,
        "_which",
        lambda name: f"/usr/bin/{name}" if name == "apt-get" else None,
    )
    probe = detection.ToolchainProbe(
        cmake_path="/usr/bin/cmake",
        cmake_version="3.30.1",
        git_path="/usr/bin/git",
        git_version="2.49.0",
    )

    rows = {row.key: row for row in detection.build_tool_statuses(probe)}

    assert rows["cmake"].installed is True
    assert rows["cmake"].install_plan is None
    assert rows["ninja"].installed is False
    assert rows["ninja"].install_plan is not None
    assert rows["ninja"].install_plan.command.endswith("ninja-build")
    assert rows["git"].installed is True


def test_terminal_launcher_uses_first_available_linux_terminal(monkeypatch):
    monkeypatch.setattr(terminal_launcher.sys, "platform", "linux")
    monkeypatch.setattr(
        terminal_launcher.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name == "gnome-terminal" else None,
    )
    popen = MagicMock()
    monkeypatch.setattr(terminal_launcher.subprocess, "Popen", popen)

    terminal_launcher.open_command_in_terminal("sudo apt-get install -y cmake")

    argv = popen.call_args.args[0]
    assert argv[0] == "/usr/bin/gnome-terminal"
    assert argv[1:4] == ["--", "bash", "-lc"]
    assert "sudo apt-get install -y cmake" in argv[4]


def test_terminal_launcher_uses_osascript_on_macos(monkeypatch):
    monkeypatch.setattr(terminal_launcher.sys, "platform", "darwin")
    popen = MagicMock()
    monkeypatch.setattr(terminal_launcher.subprocess, "Popen", popen)

    terminal_launcher.open_command_in_terminal("brew install ninja", cwd="/tmp/project")

    argv = popen.call_args.args[0]
    assert argv[0] == "osascript"
    assert argv[1] == "-e"
    assert 'do script "cd /tmp/project && brew install ninja"' in argv[2]


def test_terminal_launcher_uses_cmd_start_on_windows(monkeypatch):
    monkeypatch.setattr(terminal_launcher.sys, "platform", "win32")
    popen = MagicMock()
    monkeypatch.setattr(terminal_launcher.subprocess, "Popen", popen)

    terminal_launcher.open_command_in_terminal("winget install --id Kitware.CMake -e")

    assert popen.call_args.args[0] == [
        "cmd",
        "/c",
        "start",
        "",
        "cmd",
        "/k",
        "winget install --id Kitware.CMake -e",
    ]


def test_build_tab_generator_defaults_to_cmake_label(tk_root, tmp_path, monkeypatch):
    _launcher, tab = _make_build_tab(tk_root, tmp_path, monkeypatch)

    plan = tab._build_plan()

    assert tab.var_generator.get() == DEFAULT_GENERATOR_LABEL
    assert plan is not None
    assert plan.generator == ""


def test_loading_config_without_generator_resets_ui_to_cmake(tk_root, tmp_path, monkeypatch):
    _launcher, tab = _make_build_tab(tk_root, tmp_path, monkeypatch)
    tab.var_generator.set("Ninja")

    tab._apply_loaded_config(
        BuildConfig(
            name="default-generator",
            backend="llama.cpp",
            source_dir="/repos/llama",
            build_dir="build",
            ui_state={},
        )
    )

    assert tab.var_generator.get() == DEFAULT_GENERATOR_LABEL


def test_install_missing_tool_opens_terminal_from_build_tab(tk_root, tmp_path, monkeypatch):
    _launcher, tab = _make_build_tab(tk_root, tmp_path, monkeypatch)
    install_plan = detection.ToolInstallPlan(
        tool_key="cmake",
        tool_label="CMake",
        package_manager="apt-get",
        command="sudo apt-get update && sudo apt-get install -y cmake",
    )
    monkeypatch.setattr(
        detection,
        "build_tool_statuses",
        lambda _probe: [
            detection.BuildToolStatus(
                key="cmake",
                label="CMake",
                path=None,
                version=None,
                install_plan=install_plan,
            )
        ],
    )
    launch_mock = MagicMock()
    monkeypatch.setattr(terminal_launcher, "open_command_in_terminal", launch_mock)
    info_mock = MagicMock()
    error_mock = MagicMock()
    monkeypatch.setattr("modules.build.build_tab.messagebox.showinfo", info_mock)
    monkeypatch.setattr("modules.build.build_tab.messagebox.showerror", error_mock)

    tab._on_install_build_tool("cmake")

    launch_mock.assert_called_once_with(install_plan.command)
    assert info_mock.called
    error_mock.assert_not_called()
