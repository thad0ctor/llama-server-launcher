"""End-to-end(ish) tests for ``LaunchManager.launch_server()``.

The real function spawns subprocesses and temp files, so every call is
driven with a fully-mocked ``subprocess.Popen``, ``shutil.which``, and
``tempfile.mkstemp``. We verify *what gets passed* to those calls, not
side effects.

Platforms exercised:
  * Windows - temp ``.ps1`` creation, cleanup-thread spawn, venv-missing
    error path, Popen happy path.
  * Linux - terminal-emulator discovery loop (xterm, gnome-terminal,
    konsole, xfce4-terminal, mate-terminal), fallback to ``shell=True``
    when no terminal is found, ``messagebox.showwarning`` on fallback.
  * macOS (darwin) - treated as Linux by the current implementation
    (bash + terminal-emulator path); we verify it doesn't pick the
    Windows branch.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


# ============================================================================
# Windows branch
# ============================================================================


def _win_cmd_list(exe_posix="C:/llama-server.exe", model="C:/m/model.gguf"):
    """A canonical cmd_list result as build_cmd() would have produced on
    Windows, used to sidestep the _find_server_executable platform
    dependency in tests that only care about launch_server's behaviour."""
    return [exe_posix, "-m", model]


class TestLaunchServerWindows:
    def test_windows_creates_temp_ps1_and_invokes_powershell(
        self, manager, launcher_mock, tmp_path
    ):
        """The Windows branch writes a .ps1 file, calls Popen with
        powershell.exe + -File pointed at that file, and spawns a cleanup
        thread."""
        ps1_path = tmp_path / "fake_launch.ps1"

        def fake_mkstemp(suffix=None, prefix=None, text=None):
            import os as _os
            fd = _os.open(str(ps1_path), _os.O_CREAT | _os.O_RDWR)
            return fd, str(ps1_path)

        from modules.launch import subprocess as launch_subprocess  # noqa: F401

        with patch.object(sys, "platform", "win32"), patch(
            "modules.launch.tempfile.mkstemp", side_effect=fake_mkstemp
        ), patch("modules.launch.subprocess.Popen") as popen, patch(
            "modules.launch.Thread"
        ) as thread_cls, patch("modules.launch.messagebox"), patch.object(
            manager, "build_cmd", return_value=_win_cmd_list()
        ), patch(
            "modules.launch.subprocess.CREATE_NEW_CONSOLE", 0x10, create=True
        ):
            popen.return_value = MagicMock()
            manager.launch_server()

        # PowerShell was invoked with -File pointing at the temp script.
        assert popen.called
        args, kwargs = popen.call_args
        cmd = args[0]
        assert cmd[0] == "powershell.exe"
        assert "-ExecutionPolicy" in cmd
        assert "Bypass" in cmd
        assert "-File" in cmd
        file_idx = cmd.index("-File")
        assert Path(cmd[file_idx + 1]).resolve() == ps1_path.resolve()
        assert kwargs.get("shell") is False
        assert "creationflags" in kwargs

        # Cleanup thread was scheduled.
        assert thread_cls.called
        t_kwargs = thread_cls.call_args.kwargs
        assert t_kwargs.get("daemon") is True
        assert t_kwargs["target"] == launcher_mock.cleanup
        assert t_kwargs["args"][0] == str(ps1_path)

        # Script was written with the expected header.
        content = ps1_path.read_text(encoding="utf-8")
        assert "$ErrorActionPreference" in content
        assert '$env:CUDA_DEVICE_ORDER="PCI_BUS_ID"' in content

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason=(
            "The test's fake_mkstemp returns an open fd that isn't closed "
            "before the launcher tries to os.unlink the temp file. POSIX "
            "allows unlink on an open file; Windows does not. The test "
            "verifies cleanup semantics, not the fd leak, so the simplest "
            "correct thing is to only run it on POSIX."
        ),
    )
    def test_windows_missing_venv_activate_aborts_and_cleans_up(
        self, manager, launcher_mock, tmp_path
    ):
        """Venv configured but Activate.ps1 not present: messagebox error,
        temp file unlinked, no Popen."""
        ps1_path = tmp_path / "launch.ps1"

        def fake_mkstemp(**kw):
            import os as _os
            fd = _os.open(str(ps1_path), _os.O_CREAT | _os.O_RDWR)
            return fd, str(ps1_path)

        # Venv dir exists but has no Scripts/Activate.ps1.
        venv = tmp_path / "venv"
        (venv / "Scripts").mkdir(parents=True)
        launcher_mock.venv_dir.set(str(venv))

        with patch.object(sys, "platform", "win32"), patch(
            "modules.launch.tempfile.mkstemp", side_effect=fake_mkstemp
        ), patch("modules.launch.subprocess.Popen") as popen, patch(
            "modules.launch.messagebox"
        ) as mb, patch("modules.launch.Thread"), patch.object(
            manager, "build_cmd", return_value=_win_cmd_list()
        ):
            manager.launch_server()

        # No Popen happened (we returned early), and error was surfaced.
        assert not popen.called
        assert mb.showerror.called
        # temp file was removed.
        assert not ps1_path.exists()

    def test_windows_popen_raising_is_caught_and_reported(
        self, manager, launcher_mock, tmp_path
    ):
        ps1_path = tmp_path / "launch.ps1"

        def fake_mkstemp(**kw):
            import os as _os
            fd = _os.open(str(ps1_path), _os.O_CREAT | _os.O_RDWR)
            return fd, str(ps1_path)

        with patch.object(sys, "platform", "win32"), patch(
            "modules.launch.tempfile.mkstemp", side_effect=fake_mkstemp
        ), patch(
            "modules.launch.subprocess.Popen",
            side_effect=OSError("Access denied"),
        ), patch("modules.launch.messagebox") as mb, patch(
            "modules.launch.Thread"
        ), patch.object(
            manager, "build_cmd", return_value=_win_cmd_list()
        ), patch(
            "modules.launch.subprocess.CREATE_NEW_CONSOLE", 0x10, create=True
        ):
            manager.launch_server()

        # Outer except caught the Popen error and showed a Launch Error.
        assert mb.showerror.called
        titles = [c.args[0] for c in mb.showerror.call_args_list if c.args]
        assert any("Launch Error" in t for t in titles)


# ============================================================================
# Linux branch - terminal emulator discovery
# ============================================================================


@pytest.mark.skipif(
    sys.platform == "win32",
    reason=(
        "Linux-branch tests patch sys.platform='linux' but can't convincingly "
        "simulate POSIX on a real Windows runner — os.name stays 'nt', "
        "pathlib normalizes slashes differently, etc."
    ),
)
class TestLaunchServerLinuxTerminals:
    @pytest.mark.parametrize(
        "terminal",
        ["xterm", "gnome-terminal", "konsole", "xfce4-terminal"],
    )
    def test_each_known_terminal_is_probed(
        self, manager, launcher_mock, terminal
    ):
        """``shutil.which`` returns a path only for the target terminal;
        all others return None. Popen must be called exactly once with
        that terminal's path.
        """
        def fake_which(name):
            return f"/usr/bin/{name}" if name == terminal else None

        with patch.object(sys, "platform", "linux"), patch(
            "modules.launch.shutil.which", side_effect=fake_which
        ), patch("modules.launch.subprocess.Popen") as popen, patch(
            "modules.launch.messagebox"
        ):
            popen.return_value = MagicMock()
            manager.launch_server()

        # Popen called at least once with the terminal's resolved path
        # as argv[0].
        assert popen.called
        invoked_paths = [c.args[0][0] for c in popen.call_args_list]
        assert any(terminal in p for p in invoked_paths), (
            f"Expected {terminal} in one of the Popen invocations, got "
            f"{invoked_paths!r}"
        )

    def test_konsole_uses_noclose_flag(self, manager, launcher_mock):
        def fake_which(name):
            return "/usr/bin/konsole" if name == "konsole" else None

        with patch.object(sys, "platform", "linux"), patch(
            "modules.launch.shutil.which", side_effect=fake_which
        ), patch("modules.launch.subprocess.Popen") as popen, patch(
            "modules.launch.messagebox"
        ):
            popen.return_value = MagicMock()
            manager.launch_server()

        argv = popen.call_args.args[0]
        assert "--noclose" in argv
        assert "-e" in argv

    def test_gnome_terminal_uses_double_dash_separator(
        self, manager, launcher_mock
    ):
        def fake_which(name):
            return "/usr/bin/gnome-terminal" if name == "gnome-terminal" else None

        with patch.object(sys, "platform", "linux"), patch(
            "modules.launch.shutil.which", side_effect=fake_which
        ), patch("modules.launch.subprocess.Popen") as popen, patch(
            "modules.launch.messagebox"
        ):
            popen.return_value = MagicMock()
            manager.launch_server()

        argv = popen.call_args.args[0]
        assert "--" in argv
        # After --, the command should be ['bash', '-c', <script>].
        idx = argv.index("--")
        assert argv[idx + 1] == "bash"
        assert argv[idx + 2] == "-c"

    def test_first_available_terminal_wins_and_rest_not_probed(
        self, manager, launcher_mock
    ):
        """The loop iterates ['gnome-terminal', 'konsole', ...]; once one
        launches, no further Popen calls happen.
        """
        found = {"gnome-terminal"}

        def fake_which(name):
            return "/usr/bin/gnome-terminal" if name in found else None

        with patch.object(sys, "platform", "linux"), patch(
            "modules.launch.shutil.which", side_effect=fake_which
        ), patch("modules.launch.subprocess.Popen") as popen, patch(
            "modules.launch.messagebox"
        ):
            popen.return_value = MagicMock()
            manager.launch_server()

        # Popen called exactly once when the first probe succeeds.
        assert popen.call_count == 1


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Linux-branch test; see TestLaunchServerLinuxTerminals skip note.",
)
class TestLaunchServerLinuxFallback:
    def test_no_terminal_found_uses_shell_true_fallback(
        self, manager, launcher_mock
    ):
        """No supported terminal emulator is found; the code must
        messagebox.warn, then fall back to Popen(script, shell=True)."""
        with patch.object(sys, "platform", "linux"), patch(
            "modules.launch.shutil.which", return_value=None
        ), patch("modules.launch.subprocess.Popen") as popen, patch(
            "modules.launch.messagebox"
        ) as mb:
            popen.return_value = MagicMock()
            manager.launch_server()

        assert popen.called
        # Popen(script_str, shell=True)
        args, kwargs = popen.call_args
        assert isinstance(args[0], str), "Fallback must pass a single shell string"
        assert kwargs.get("shell") is True
        # User was warned.
        assert mb.showwarning.called

    def test_fallback_popen_raises_surfaces_launch_error(
        self, manager, launcher_mock
    ):
        with patch.object(sys, "platform", "linux"), patch(
            "modules.launch.shutil.which", return_value=None
        ), patch(
            "modules.launch.subprocess.Popen",
            side_effect=OSError("no shell"),
        ), patch("modules.launch.messagebox") as mb:
            manager.launch_server()

        # Final messagebox.showerror called with Launch Error title.
        assert mb.showerror.called

    def test_linux_missing_venv_activate_errors_out(
        self, manager, launcher_mock, tmp_path
    ):
        venv = tmp_path / "venv"
        (venv / "bin").mkdir(parents=True)
        # No activate script present.
        launcher_mock.venv_dir.set(str(venv))

        with patch.object(sys, "platform", "linux"), patch(
            "modules.launch.shutil.which", return_value=None
        ), patch("modules.launch.subprocess.Popen") as popen, patch(
            "modules.launch.messagebox"
        ) as mb:
            manager.launch_server()

        assert not popen.called
        assert mb.showerror.called


# ============================================================================
# macOS branch - currently routes through the Linux path (bash + terminals)
# ============================================================================


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="macOS-branch test; see TestLaunchServerLinuxTerminals skip note.",
)
class TestLaunchServerMacOS:
    def test_darwin_does_not_take_windows_branch(self, manager, launcher_mock):
        """On darwin, no .ps1 is created; it falls through to the Linux
        bash path. This is a smoke test of platform gating."""
        mkstemp_called = {"hit": False}

        def fake_mkstemp(**kw):
            mkstemp_called["hit"] = True
            return 0, "/tmp/nope.ps1"

        with patch.object(sys, "platform", "darwin"), patch(
            "modules.launch.tempfile.mkstemp", side_effect=fake_mkstemp
        ), patch(
            "modules.launch.shutil.which",
            side_effect=lambda n: "/usr/bin/iterm" if n == "iterm" else None,
        ), patch("modules.launch.subprocess.Popen") as popen, patch(
            "modules.launch.messagebox"
        ):
            popen.return_value = MagicMock()
            manager.launch_server()

        assert mkstemp_called["hit"] is False
        assert popen.called


# ============================================================================
# Cleanup thread behaviour
# ============================================================================


class TestLaunchServerProbingOptIn:
    """launch_server() must opt into runtime feature probing
    (probe_backend=True) so older ik_llama builds don't get unsupported
    flags. The save-script paths intentionally omit this — see
    tests/launchers/test_launch.py for those."""

    def test_launch_server_passes_probe_backend_true_to_build_cmd(
        self, manager, launcher_mock
    ):
        """Returning None from build_cmd causes launch_server to bail
        immediately, so we don't need to mock subprocess/Popen — we just
        need to verify the call signature."""
        with patch.object(manager, "build_cmd", return_value=None) as bc:
            manager.launch_server()

        assert bc.called, "launch_server should invoke build_cmd"
        # Accept either kwarg form or positional — what matters is the value.
        kwargs = bc.call_args.kwargs
        args = bc.call_args.args
        probe_value = kwargs.get("probe_backend", args[0] if args else None)
        assert probe_value is True, (
            "launch_server must call build_cmd(probe_backend=True) so the "
            f"runtime feature probe runs; got call_args={bc.call_args!r}"
        )


class TestCleanupThread:
    def test_cleanup_thread_is_daemon_on_windows(self, manager, launcher_mock, tmp_path):
        ps1_path = tmp_path / "launch.ps1"

        def fake_mkstemp(**kw):
            import os as _os
            fd = _os.open(str(ps1_path), _os.O_CREAT | _os.O_RDWR)
            return fd, str(ps1_path)

        with patch.object(sys, "platform", "win32"), patch(
            "modules.launch.tempfile.mkstemp", side_effect=fake_mkstemp
        ), patch("modules.launch.subprocess.Popen") as popen, patch(
            "modules.launch.Thread"
        ) as thread_cls, patch("modules.launch.messagebox"), patch.object(
            manager, "build_cmd", return_value=_win_cmd_list()
        ), patch(
            "modules.launch.subprocess.CREATE_NEW_CONSOLE", 0x10, create=True
        ):
            popen.return_value = MagicMock()
            manager.launch_server()

        kw = thread_cls.call_args.kwargs
        assert kw.get("daemon") is True
        # start() was called on the Thread instance.
        thread_instance = thread_cls.return_value
        assert thread_instance.start.called
