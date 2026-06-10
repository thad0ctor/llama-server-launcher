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
    # ``open_command_in_terminal`` prefers ``$TERMINAL`` if set, so
    # a developer running ``TERMINAL=kitty pytest`` would short-
    # circuit the preference-list probe this test is asserting
    # against. Drop the variable so the fallback to
    # ``shutil.which`` is the path under test on every host.
    monkeypatch.delenv("TERMINAL", raising=False)
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
    # ``_cmd_keep_open`` writes the user command to a temp PAYLOAD
    # .cmd script and a separate WRAPPER .cmd that ``call``s into
    # it. ``cmd /k`` points at a ``call "%LLAMA_LAUNCHER_WRAPPER_PATH%"``
    # token — the wrapper path is propagated via an environment
    # variable (``Popen(env=...)``) instead of embedded in the
    # /k argument string. This avoids cmd's ``%FOO%`` expansion
    # in the /k argument from mangling temp paths that legitimately
    # contain ``%`` (e.g. under a ``%-prefixed username); the env
    # var carries the path as an opaque single-string lookup.
    assert argv == [
        "cmd",
        "/c",
        "start",
        "",
        "cmd",
        "/k",
        'call "%LLAMA_LAUNCHER_WRAPPER_PATH%"',
    ]
    # The actual wrapper path lives in the env kwarg passed to Popen.
    env_kwarg = popen.call_args.kwargs["env"]
    wrapper_path = env_kwarg["LLAMA_LAUNCHER_WRAPPER_PATH"]
    assert wrapper_path.endswith(".cmd"), f"expected .cmd wrapper path, got {wrapper_path!r}"
    # Inspect the generated wrapper to verify the contract: it
    # prints the running banner, ``call``s the payload, echoes the
    # exit code, and self-deletes.
    import os as _os

    try:
        with open(wrapper_path, encoding="utf-8") as fh:
            wrapper_body = fh.read()
    finally:
        try:
            _os.unlink(wrapper_path)
        except OSError:
            pass
    assert "Running command..." in wrapper_body
    # The wrapper should NOT embed the user command inline — the
    # whole point of the rewrite was to avoid the outer batch
    # parser ever seeing the user's string. It should only have a
    # ``call "<payload_path>"`` reference.
    assert "winget install --id Kitware.CMake -e" not in wrapper_body
    import re

    call_match = re.search(r'call "([^"]+\.cmd)"', wrapper_body)
    assert call_match, f"wrapper missing call line; body={wrapper_body!r}"
    payload_path = call_match.group(1)
    # ``%ERRORLEVEL%`` (single ``%``) is the runtime value inside
    # a batch file — bytes used to be ``%%`` only because the
    # string went through ``cmd /c``'s parser first.
    assert "Command finished with exit code %ERRORLEVEL%." in wrapper_body
    assert 'del "%~f0"' in wrapper_body
    # And the payload itself should contain the user command on
    # its own line + an exit-code passthrough + self-delete.
    try:
        with open(payload_path, encoding="utf-8") as fh:
            payload_body = fh.read()
    finally:
        try:
            _os.unlink(payload_path)
        except OSError:
            pass
    # The payload now invokes the user command via a nested
    # ``cmd /d /c ""<command>""`` so a nested ``.bat`` / ``.cmd``
    # without ``call`` can't transfer control away and skip the
    # self-delete / exit-code tail.
    assert 'cmd /d /c ""winget install --id Kitware.CMake -e""' in payload_body
    assert 'del "%~f0"' in payload_body
    assert "exit /b" in payload_body.lower()


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
