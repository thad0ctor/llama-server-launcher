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
        # (``/opt/homebrew/bin/kitty``). Use TWO values per entry:
        #   * launch name  – passed to ``shutil.which`` / used directly
        #     when absolute. Without this, a user-set
        #     ``TERMINAL=/opt/homebrew/bin/kitty`` was stripped to
        #     ``"kitty"`` and ``shutil.which`` then looked it up on PATH,
        #     ignoring the user's explicit choice if PATH didn't have it.
        #   * arg-template key – basename used to look up
        #     ``emulator_args`` so a path-form $TERMINAL still gets the
        #     right ``-- bash -lc`` / ``--hold -e ...`` form instead of
        #     falling through to the safe-default ``-e``.
        env_terminal_key = Path(env_terminal_raw).name if env_terminal_raw else ""
        ordered_terms: list[tuple[str, str]] = []
        if env_terminal_raw:
            ordered_terms.append(
                (env_terminal_raw, env_terminal_key or env_terminal_raw)
            )
        ordered_terms.extend(
            [
                ("x-terminal-emulator", "x-terminal-emulator"),
                ("gnome-terminal", "gnome-terminal"),
                ("konsole", "konsole"),
                ("xfce4-terminal", "xfce4-terminal"),
                ("xterm", "xterm"),
            ]
        )
        seen: set[str] = set()
        for terminal_name, terminal_key in ordered_terms:
            if terminal_name in seen:
                continue
            seen.add(terminal_name)
            if os.path.isabs(terminal_name):
                # Absolute ``$TERMINAL`` must be a real, executable regular
                # file. Otherwise (broken symlink, directory entry, no +x)
                # treat it like "not installed" and fall through to the
                # next emulator candidate — better than ``Popen`` raising
                # ``PermissionError`` halfway down the loop with the
                # remaining candidates unexamined.
                candidate = Path(terminal_name)
                if (
                    candidate.is_file()
                    and os.access(str(candidate), os.X_OK)
                ):
                    terminal_path = str(candidate)
                else:
                    terminal_path = None
            else:
                terminal_path = shutil.which(terminal_name)
            if terminal_path is None:
                continue
            # An emulator not in our argument map (e.g. user-set $TERMINAL
            # pointing at something exotic) gets a safe default of ``-e``.
            args = emulator_args.get(
                terminal_key, ["-e", "bash", "-lc", term_command]
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
                # Embedding ``shlex.quote(script_path)`` inside an outer
                # single-quoted ``trap '...'`` string breaks the moment
                # the path contains a single quote — ``shlex.quote``
                # emits ``'foo'\''bar'`` which ends the outer trap quote
                # mid-sequence. ``tempfile.mkstemp`` doesn't normally
                # produce such paths today, but defending the boundary
                # is cheap: assign the path to a shell variable first
                # (single-quoted by Python — and ``shlex.quote`` already
                # handles arbitrary content there), then reference the
                # variable inside the trap body with double-quoted
                # expansion. The trap body is a single-quoted string so
                # ``$LLAMA_LAUNCHER_SCRIPT_PATH`` survives until trap fires.
                fh.write(
                    f"LLAMA_LAUNCHER_SCRIPT_PATH={shlex.quote(script_path)}\n"
                )
                fh.write('trap \'rm -f "$LLAMA_LAUNCHER_SCRIPT_PATH"\' EXIT\n')
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
        # The script self-deletes once Terminal.app runs it, but if
        # ``osascript`` itself fails to spawn (PATH issue, sandbox denial,
        # OS resource limit) that self-delete never runs and the temp
        # launcher file leaks. Every retry would then leave another stub
        # behind. Clean up explicitly on failure before bubbling the
        # exception to the caller.
        try:
            subprocess.Popen(["osascript", "-e", applescript], cwd=cwd_text)
        except Exception:
            try:
                os.unlink(script_path)
            except OSError:
                pass
            raise
        return

    if sys.platform.startswith("win"):
        subprocess.Popen(_cmd_keep_open(command), cwd=cwd_text)
        return

    raise RuntimeError(f"Unsupported platform: {sys.platform}")
