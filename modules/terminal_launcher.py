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
        env_terminal_raw = os.environ.get("TERMINAL", "").strip()
        # ``$TERMINAL`` is conventionally a name (``gnome-terminal``), but
        # users sometimes set it to an absolute path
        # (``/usr/local/bin/gnome-terminal``). Look up our argument template
        # by the basename so a path-form $TERMINAL still gets the right
        # ``-- bash -lc`` / ``--hold -e ...`` form instead of falling
        # through to the safe-default ``-e`` invocation.
        env_terminal = Path(env_terminal_raw).name if env_terminal_raw else ""
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
        # Write the bash payload to a temp ``.command`` script and have
        # Terminal.app run that, instead of splicing the entire bash
        # script through layered AppleScript+shell quoting (which is
        # fragile for paths with backticks, dollar signs, newlines, etc.).
        # The script self-deletes after running so the temp file doesn't
        # accumulate.
        import tempfile

        bash_payload = _bash_hold_open(command)
        if cwd_text:
            bash_payload = f"cd {shlex.quote(cwd_text)} && {bash_payload}"
        fd, script_path = tempfile.mkstemp(suffix=".command", prefix="llama-launcher-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write("#!/bin/bash\n")
                fh.write(f"trap 'rm -f {shlex.quote(script_path)}' EXIT\n")
                fh.write(bash_payload)
                fh.write("\n")
            os.chmod(script_path, 0o755)
        except Exception:
            try:
                os.unlink(script_path)
            except OSError:
                pass
            raise
        # Two-stage quoting is needed here:
        #   1. ``do script "..."`` hands the inner string to the user's
        #      shell, so the path itself must be SHELL-quoted (or a path
        #      with spaces fails to execute).
        #   2. The whole thing is embedded inside an AppleScript double-
        #      quoted string, so backslashes and double quotes inside the
        #      shell-quoted form need AppleScript escaping (``\\`` and
        #      ``\"``) — without this, a single-quoted shell payload
        #      survives, but a path containing ``"`` would still break the
        #      AppleScript.
        shell_cmd = shlex.quote(script_path)
        escaped_path = shell_cmd.replace("\\", "\\\\").replace('"', '\\"')
        applescript = (
            'tell application "Terminal"\n'
            f'    do script "{escaped_path}"\n'
            "    activate\n"
            "end tell\n"
        )
        subprocess.Popen(["osascript", "-e", applescript], cwd=cwd_text)
        return

    if sys.platform.startswith("win"):
        subprocess.Popen(_cmd_keep_open(command), cwd=cwd_text)
        return

    raise RuntimeError(f"Unsupported platform: {sys.platform}")
