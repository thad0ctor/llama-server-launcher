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
        "echo; "
        "if [[ $status -eq 0 ]]; then "
        'echo "Command completed successfully."; '
        "else "
        'echo "Command failed with exit code $status."; '
        "fi; "
        "if [[ -t 1 || $status -ne 0 ]]; then "
        'read -rp "Press Enter to close..." </dev/tty; '
        "fi; "
        "exit $status"
    )


def _cmd_keep_open(command: str) -> tuple[list[str], list[str]]:
    """Return a Windows ``cmd`` invocation that reports success/failure.

    Writes the user command to a SEPARATE payload ``.cmd`` file and a
    wrapper ``.cmd`` that calls into it. Both files self-delete after
    running so they don't accumulate in ``%TEMP%``.

    The two-file split exists because the earlier inline form
    ``cmd /d /c ""{command}""`` embedded inside the wrapper script
    re-introduced the OUTER batch parser at the moment the wrapper
    line was read. cmd.exe expands ``%FOO%`` while parsing each
    line of a ``.cmd``, BEFORE the line is handed to the child
    ``cmd /d /c`` — so a user command of ``echo %PATH%`` was being
    substituted by the wrapper's own environment, not the payload's.
    Putting the user command on its own line in a dedicated .cmd
    that the wrapper merely ``call``s side-steps that: the parsing
    happens once, inside the payload, the same as if the user had
    typed it at a fresh cmd prompt.

    Other layered-cmd problems we previously worked around:

    * ``!literal!`` was expanded as a delayed-expansion variable
      (``/v:on``). Dropping ``/v:on`` solved that.
    * Wrapping in ``cmd /d /c ""…""`` was added so a payload with
      its own double-quotes (``msiexec /i "C:\\…\\foo.msi"``)
      survived parsing — that argument no longer applies because
      the payload lives in its own file now and is invoked by
      ``call "<path>"`` instead of being a quoted argument.
    """
    import tempfile

    # Payload file: holds the user's command verbatim, on its own
    # line, with NO outer wrapper around it. cmd will parse this
    # line ONCE when the wrapper calls into it.
    payload_fd, payload_path = tempfile.mkstemp(suffix=".cmd", prefix="llama-launcher-payload-", text=False)
    # cmd.exe requires CRLF line endings in .cmd files for reliable
    # parsing; ``newline=""`` + explicit ``\r\n`` ensures that even
    # on POSIX hosts running cross-platform tooling. Use
    # ``utf-8-sig`` so cmd.exe sees a UTF-8 BOM and parses non-ASCII
    # paths / commands correctly — without the BOM cmd would
    # interpret the file with the active code page (usually CP1252
    # on US-English Windows) and corrupt anything outside that
    # subset.
    try:
        with os.fdopen(payload_fd, "w", encoding="utf-8-sig", newline="") as fh:
            fh.write("@echo off\r\n")
            # Wrap the user command in ``cmd /d /c "<command>"`` here
            # too, even though the WRAPPER already wraps via
            # ``call <payload>``. Without this inner wrap, a user
            # command that invokes another ``.bat``/``.cmd`` without
            # ``call`` (``vcvars*.bat``, ``conda.bat activate``,
            # ``activate.bat && build.bat``) transfers control to
            # THAT script under batch-chain semantics — the rest of
            # this payload (the ``set _LLAMA_LAUNCHER_RC`` line, the
            # self-delete, the ``exit /b``) never runs, and the
            # wrapper's ``%ERRORLEVEL%`` reports the wrong status.
            # The nested ``cmd /d /c`` keeps that batch-chain inside
            # its own cmd instance so control returns here before
            # the tail runs. Documented quoting form:
            # https://ss64.com/nt/cmd.html — the ``""…""`` double-
            # double survives one strip-pass for payloads with their
            # own embedded quotes. ``/d`` skips AutoRun.
            escaped_command = command.replace('"', '""')
            fh.write(f'cmd /d /c ""{escaped_command}""\r\n')
            # Hand the user command's exit code back to the wrapper
            # via ``exit /b`` so ``%ERRORLEVEL%`` in the wrapper
            # reflects the payload's status rather than the
            # ``del`` below. The wrapper checks ERRORLEVEL after
            # the call to print the success/failure summary.
            fh.write("set _LLAMA_LAUNCHER_RC=%ERRORLEVEL%\r\n")
            # Self-delete the payload before returning — we don't
            # want it surviving past the call, and the wrapper has
            # already loaded our line into the running cmd.
            fh.write('del "%~f0"\r\n')
            fh.write("exit /b %_LLAMA_LAUNCHER_RC%\r\n")
    except Exception:
        try:
            os.unlink(payload_path)
        except OSError:
            pass
        raise

    # Wrapper file: prints the running banner, ``call``s the
    # payload (so control returns here even if the payload is a
    # batch chain or invokes another .cmd without ``call``), then
    # echoes the exit code and self-deletes.
    wrapper_fd, wrapper_path = tempfile.mkstemp(suffix=".cmd", prefix="llama-launcher-", text=False)
    try:
        with os.fdopen(wrapper_fd, "w", encoding="utf-8-sig", newline="") as fh:
            fh.write("@echo off\r\n")
            fh.write("echo Running command...\r\n")
            # ``call`` so control returns to this wrapper after
            # the payload finishes — without ``call``, batch-chain
            # semantics transfer control to the payload script and
            # the wrapper's exit-code echo + self-delete would
            # never run (https://ss64.com/nt/call.html). Quote the
            # path so spaces in ``%TEMP%`` (e.g. ``C:\Users\Bob
            # Smith\AppData\…``) don't truncate the argument.
            fh.write(f'call "{payload_path}"\r\n')
            fh.write("echo.\r\n")
            fh.write("echo Command finished with exit code %ERRORLEVEL%.\r\n")
            # Self-delete the wrapper. ``%~f0`` is the full path
            # of the running .cmd. The user's ``cmd /k`` shell
            # stays open after this so they can inspect output,
            # but the script file itself is gone.
            fh.write('del "%~f0"\r\n')
    except Exception:
        for path in (payload_path, wrapper_path):
            try:
                os.unlink(path)
            except OSError:
                pass
        raise

    # Return BOTH the argv AND BOTH script paths so the caller can
    # clean up the temp files on ``Popen`` failure. The scripts
    # normally self-delete after running, but if the parent
    # ``cmd`` never starts (PATH issue, AppLocker block, etc.)
    # either file would otherwise leak in ``%TEMP%``.
    # ``wrapper_path`` is composed into the ``cmd /k`` argument as a
    # ``call "<path>"`` token, NOT as a bare argv element. Without
    # the quotes, a temp path containing a shell metacharacter (the
    # canonical example is ``&`` in a username like ``AT&T``) reaches
    # cmd's parser unquoted, which then chains the rest of the line
    # as a separate command — the wrapper never runs and the user
    # sees a broken terminal. ``subprocess.list2cmdline`` (used by
    # ``Popen`` on Windows when argv is a list) quotes args containing
    # whitespace or quote chars but does NOT escape ``&`` / ``|`` /
    # ``^``, so we have to wrap the path ourselves.
    return (
        ["cmd", "/c", "start", "", "cmd", "/k", f'call "{wrapper_path}"'],
        [wrapper_path, payload_path],
    )


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
            ordered_terms.append((env_terminal_raw, env_terminal_key or env_terminal_raw))
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
                if candidate.is_file() and os.access(str(candidate), os.X_OK):
                    terminal_path = str(candidate)
                else:
                    terminal_path = None
            else:
                terminal_path = shutil.which(terminal_name)
            if terminal_path is None:
                continue
            # An emulator not in our argument map (e.g. user-set $TERMINAL
            # pointing at something exotic) gets a safe default of ``-e``.
            args = emulator_args.get(terminal_key, ["-e", "bash", "-lc", term_command])
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
            # ``_bash_hold_open`` returns a ``;``-separated chain.
            # Without the brace group, ``cd /foo && cmd1 ; cmd2 ;
            # ...`` only gates ``cmd1`` on the cd — the ``;``-chained
            # follow-ups run from whatever the previous cwd was even
            # if cd failed. Wrap in ``{ ...; }`` so the cd guards the
            # WHOLE payload.
            bash_payload = f"cd {shlex.quote(cwd_text)} && {{ {bash_payload}; }}"
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
                fh.write(f"LLAMA_LAUNCHER_SCRIPT_PATH={shlex.quote(script_path)}\n")
                fh.write("trap 'rm -f \"$LLAMA_LAUNCHER_SCRIPT_PATH\"' EXIT\n")
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
        applescript = 'tell application "Terminal"\n' f'    do script "{escaped_path}"\n' "    activate\n" "end tell\n"
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
        argv, script_paths = _cmd_keep_open(command)
        try:
            subprocess.Popen(argv, cwd=cwd_text)
        except Exception:
            # ``Popen`` blew up before the inner ``cmd /k`` had a
            # chance to execute the scripts' self-delete lines.
            # Clean BOTH the wrapper and the payload ourselves so
            # we don't litter ``%TEMP%`` on repeated failures.
            # Mirrors the macOS branch's cleanup.
            for path in script_paths:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            raise
        return

    raise RuntimeError(f"Unsupported platform: {sys.platform}")
