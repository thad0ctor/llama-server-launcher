"""Helpers for opening commands in a new terminal window."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def _bash_hold_open(command: str) -> str:
    """Run ``command`` and keep the terminal open long enough to read output."""
    return (
        'echo "Running command..."; '
        f"{command}; "
        "status=$?; "
        'echo; '
        'if [[ $status -eq 0 ]]; then '
        'echo "Command completed successfully."; '
        'else '
        'echo "Command failed with exit code $status."; '
        'fi; '
        'if [[ -t 1 || $status -ne 0 ]]; then '
        'read -rp "Press Enter to close..." </dev/tty; '
        "fi; "
        "exit $status"
    )


def _cmd_keep_open(command: str) -> list[str]:
    """Return a Windows ``cmd`` invocation that reports success/failure."""
    wrapped = (
        'echo Running command... & '
        f'{command} & '
        'set "launcher_status=!errorlevel!" & '
        'echo. & '
        'if not "!launcher_status!"=="0" ('
        'echo Command failed with exit code !launcher_status!.'
        ') else ('
        'echo Command completed successfully.'
        ')'
    )
    return ["cmd", "/v:on", "/c", "start", "", "cmd", "/v:on", "/k", wrapped]


def open_command_in_terminal(command: str, *, cwd: str | Path | None = None) -> None:
    """Launch ``command`` in a new terminal window on the current platform."""
    cwd_text = str(cwd) if cwd is not None else None

    if sys.platform.startswith("linux"):
        term_command = _bash_hold_open(command)
        # Preference order:
        #   1. ``$TERMINAL`` if the user set it (explicit intent wins).
        #   2. Debian's ``x-terminal-emulator`` alternative, which the
        #      user/distro picks once and every well-behaved app respects.
        #   3. Specific emulators in priority order, biased toward what's
        #      typically installed on each desktop. Without this, a system
        #      that happens to have xterm installed alongside konsole on
        #      KDE used to silently get xterm.
        emulator_args: dict[str, list[str]] = {
            "gnome-terminal": ["--", "bash", "-lc", term_command],
            "konsole": ["--noclose", "-e", "bash", "-lc", term_command],
            "xfce4-terminal": ["--hold", "-e", f"bash -lc {shlex.quote(term_command)}"],
            "x-terminal-emulator": ["-e", "bash", "-lc", term_command],
            "xterm": ["-hold", "-e", "bash", "-lc", term_command],
        }
        env_terminal = os.environ.get("TERMINAL", "").strip()
        ordered_names: list[str] = []
        if env_terminal:
            ordered_names.append(env_terminal)
        ordered_names.append("x-terminal-emulator")
        ordered_names.extend(["gnome-terminal", "konsole", "xfce4-terminal", "xterm"])
        seen: set[str] = set()
        for terminal_name in ordered_names:
            if terminal_name in seen:
                continue
            seen.add(terminal_name)
            terminal_path = shutil.which(terminal_name)
            if terminal_path is None:
                continue
            # An emulator not in our argument map (e.g. user-set $TERMINAL
            # pointing at something exotic) gets a safe default of ``-e``.
            args = emulator_args.get(
                terminal_name, ["-e", "bash", "-lc", term_command]
            )
            subprocess.Popen([str(Path(terminal_path).resolve()), *args], cwd=cwd_text)
            return
        raise FileNotFoundError("No supported terminal emulator found")

    if sys.platform == "darwin":
        script_command = _bash_hold_open(command)
        if cwd_text:
            script_command = f"cd {shlex.quote(cwd_text)} && {script_command}"
        script_command = script_command.replace("\\", "\\\\").replace('"', '\\"')
        applescript = (
            'tell application "Terminal"\n'
            f'    do script "{script_command}"\n'
            "    activate\n"
            "end tell\n"
        )
        subprocess.Popen(["osascript", "-e", applescript], cwd=cwd_text)
        return

    if sys.platform.startswith("win"):
        subprocess.Popen(_cmd_keep_open(command), cwd=cwd_text)
        return

    raise RuntimeError(f"Unsupported platform: {sys.platform}")
