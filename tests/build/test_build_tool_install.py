from __future__ import annotations

from unittest.mock import MagicMock

from modules import terminal_launcher
from modules.build import detection
from modules.build.build_persistence import BuildConfig
from modules.build.build_tab import DEFAULT_GENERATOR_LABEL

# Shared helpers live in ``tests/build/conftest.py`` so this file and
# ``test_build_tab_sync.py`` can't drift independently.
from tests.build.conftest import make_build_tab


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
    assert 'echo "Running command..."' in argv[4]
    assert "sudo apt-get install -y cmake" in argv[4]
    assert "Command completed successfully." in argv[4]


def test_terminal_launcher_uses_osascript_on_macos(monkeypatch, tmp_path):
    monkeypatch.setattr(terminal_launcher.sys, "platform", "darwin")
    popen = MagicMock()
    monkeypatch.setattr(terminal_launcher.subprocess, "Popen", popen)

    terminal_launcher.open_command_in_terminal("brew install ninja", cwd="/tmp/project")

    argv = popen.call_args.args[0]
    assert argv[0] == "osascript"
    assert argv[1] == "-e"
    # The macOS path now writes the bash payload to a temp ``.command``
    # script and tells Terminal.app to ``do script <script_path>`` —
    # safer than splicing the body through layered AppleScript+shell
    # quoting. Assert the new structure (not the prior inline shell).
    assert 'tell application "Terminal"' in argv[2]
    assert 'do script "' in argv[2]
    assert ".command" in argv[2]
    # Pull the script path out of the AppleScript and read it back to
    # confirm the bash payload actually got the user's command + cwd.
    import re as _re
    m = _re.search(r"do script \"([^\"]+\.command)\"", argv[2])
    assert m, f"Could not extract script path from {argv[2]!r}"
    script_path = m.group(1)
    try:
        with open(script_path, encoding="utf-8") as fh:
            contents = fh.read()
        assert "cd /tmp/project" in contents
        assert "brew install ninja" in contents
        assert 'echo "Running command..."' in contents
    finally:
        try:
            import os as _os
            _os.unlink(script_path)
        except OSError:
            pass


def test_terminal_launcher_uses_cmd_start_on_windows(monkeypatch):
    monkeypatch.setattr(terminal_launcher.sys, "platform", "win32")
    popen = MagicMock()
    monkeypatch.setattr(terminal_launcher.subprocess, "Popen", popen)

    terminal_launcher.open_command_in_terminal("winget install --id Kitware.CMake -e")

    argv = popen.call_args.args[0]
    # ``_cmd_keep_open`` now writes the user command to a temp .cmd
    # script and points ``cmd /k`` at the file (rather than passing
    # the wrapped command inline). This sidesteps both ``!literal!``
    # delayed-expansion AND ``%VAR%`` outer-parser expansion, so a
    # path with ``%PATH%`` in it survives intact.
    assert argv[:6] == ["cmd", "/c", "start", "", "cmd", "/k"]
    script_path = argv[6]
    assert script_path.endswith(".cmd"), f"expected .cmd path, got {script_path!r}"
    # Inspect the generated batch file to verify the wrapper contract
    # (header, user command, exit-code echo, self-delete).
    import os as _os
    try:
        with open(script_path, encoding="utf-8") as fh:
            body = fh.read()
    finally:
        try:
            _os.unlink(script_path)
        except OSError:
            pass
    assert "winget install --id Kitware.CMake -e" in body
    assert "Running command..." in body
    # User command must be wrapped in ``cmd /d /c`` so a
    # ``.bat`` / ``.cmd`` payload returns control to the wrapper
    # for the exit-code echo and the self-delete line. ``/d``
    # skips AutoRun registry hooks.
    assert "cmd /d /c winget install --id Kitware.CMake -e" in body
    # ``%ERRORLEVEL%`` (single ``%``) is the runtime value inside a
    # batch file — the previous ``%%`` form was needed only because
    # the string went through ``cmd /c``'s parser first.
    assert "Command finished with exit code %ERRORLEVEL%." in body
    assert 'del "%~f0"' in body


def test_build_tab_generator_defaults_to_cmake_label(tk_root, tmp_path, monkeypatch):
    _launcher, tab = make_build_tab(tk_root, tmp_path, monkeypatch)

    plan = tab._build_plan()

    assert tab.var_generator.get() == DEFAULT_GENERATOR_LABEL
    assert plan is not None
    assert plan.generator == ""


def test_loading_config_without_generator_resets_ui_to_cmake(tk_root, tmp_path, monkeypatch):
    _launcher, tab = make_build_tab(tk_root, tmp_path, monkeypatch)
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
    _launcher, tab = make_build_tab(tk_root, tmp_path, monkeypatch)
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
