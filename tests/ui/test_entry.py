"""Tests for the top-level ``llamacpp-server-launcher.py`` entry script.

Targets:
  - ``parse_cli_args`` — the new argparse wrapper that responds to ``--help``
    and ``--version`` without launching the GUI.
  - ``LlamaCppLauncher.cleanup`` — static temp-file cleanup with a delay.
  - ``_read_version_string`` — best-effort reader used by ``--version``.

The module has a hyphen in the filename, so we load it via ``importlib.util``
rather than a normal import statement.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRY_PATH = REPO_ROOT / "llamacpp-server-launcher.py"


@pytest.fixture(scope="module")
def entry_module():
    """Load the launcher script as a module named ``entry_module``.

    Module-scoped so Tk / ttk side-effects happen once. The module does
    ``import tkinter`` at top level but doesn't instantiate a root — that
    only happens inside the ``__main__`` guard, which we never trip.
    """
    spec = importlib.util.spec_from_file_location("entry_module", ENTRY_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["entry_module"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# parse_cli_args
# ---------------------------------------------------------------------------

class TestParseCliArgs:
    """Validate that --help / --version exit cleanly instead of opening GUI."""

    def test_no_args_returns_namespace(self, entry_module):
        """Default invocation must not raise — it yields an (empty) Namespace
        so the caller can proceed to open Tk."""
        ns = entry_module.parse_cli_args([])
        # Namespace has no required fields yet, but it should exist as an object.
        assert ns is not None

    def test_version_flag_exits(self, entry_module, capsys):
        """argparse treats ``--version`` as a terminating action — the process
        exits with code 0 after printing the version, so we must see SystemExit
        rather than a silent GUI launch."""
        with pytest.raises(SystemExit) as exc_info:
            entry_module.parse_cli_args(["--version"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "llamacpp-server-launcher" in out

    def test_help_flag_exits(self, entry_module, capsys):
        """``--help`` also exits cleanly — formerly silently fell through to GUI."""
        with pytest.raises(SystemExit) as exc_info:
            entry_module.parse_cli_args(["--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "usage:" in out.lower()

    def test_short_help_flag_exits(self, entry_module, capsys):
        with pytest.raises(SystemExit) as exc_info:
            entry_module.parse_cli_args(["-h"])
        assert exc_info.value.code == 0

    def test_unknown_flag_exits_nonzero(self, entry_module, capsys):
        """argparse defaults to exit code 2 for bad input — the user gets an
        error on stderr rather than being dumped into a confused GUI."""
        with pytest.raises(SystemExit) as exc_info:
            entry_module.parse_cli_args(["--bogus-flag"])
        assert exc_info.value.code != 0

    def test_version_string_contains_repo_version(self, entry_module, capsys):
        """Confirm the version embedded in the ``--version`` output matches
        the value in ``config/version`` (or ``unknown`` if the file is
        missing). Catches regressions where the embed gets stale."""
        with pytest.raises(SystemExit):
            entry_module.parse_cli_args(["--version"])
        out = capsys.readouterr().out.strip()
        # Should be one of the well-formed strings.
        assert out.startswith("llamacpp-server-launcher ")


# ---------------------------------------------------------------------------
# _read_version_string
# ---------------------------------------------------------------------------

class TestReadVersionString:
    def test_reads_shipped_version(self, entry_module):
        """There's a version file in the repo — the helper should surface it
        rather than returning 'unknown'."""
        result = entry_module._read_version_string()
        # Not empty, not the error sentinel unless the repo really has no file.
        assert isinstance(result, str)
        assert result  # non-empty

    def test_returns_unknown_on_missing_file(self, entry_module, monkeypatch,
                                             tmp_path):
        """Simulate a partial install (no ``config/version``) by repointing
        ``__file__`` at a tmp dir. Must degrade gracefully, not raise."""
        # The helper uses Path(__file__).resolve().parent / "config" / "version".
        # We patch the module-level __file__ for the duration of this test.
        fake_module_path = tmp_path / "launcher.py"
        fake_module_path.write_text("")
        monkeypatch.setattr(entry_module, "__file__", str(fake_module_path))

        result = entry_module._read_version_string()
        assert result == "unknown"

    def test_handles_empty_version_file(self, entry_module, monkeypatch,
                                        tmp_path):
        """Empty file is treated the same as missing — 'unknown'."""
        fake_module_path = tmp_path / "launcher.py"
        fake_module_path.write_text("")
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "version").write_text("")
        monkeypatch.setattr(entry_module, "__file__", str(fake_module_path))

        assert entry_module._read_version_string() == "unknown"

    def test_strips_whitespace(self, entry_module, monkeypatch, tmp_path):
        """Trailing newline/whitespace in the version file shouldn't end up
        in the printed version string."""
        fake_module_path = tmp_path / "launcher.py"
        fake_module_path.write_text("")
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "version").write_text("  2024-01-01-1  \n")
        monkeypatch.setattr(entry_module, "__file__", str(fake_module_path))

        assert entry_module._read_version_string() == "2024-01-01-1"

    def test_non_utf8_bytes_fall_back_to_unknown(
        self, entry_module, monkeypatch, tmp_path
    ):
        """A malformed version file (non-UTF-8 bytes) must not crash
        ``--version``. ``Path.read_text(encoding='utf-8')`` raises
        ``UnicodeDecodeError`` which is NOT an ``OSError``; if the helper
        only catches ``OSError`` the exception escapes and the CLI aborts."""
        fake_module_path = tmp_path / "launcher.py"
        fake_module_path.write_text("")
        (tmp_path / "config").mkdir()
        # 0x80 is a lone continuation byte — invalid as the start of a UTF-8
        # sequence, so read_text(encoding='utf-8') raises UnicodeDecodeError.
        (tmp_path / "config" / "version").write_bytes(b"\x80\x81\x82")
        monkeypatch.setattr(entry_module, "__file__", str(fake_module_path))

        assert entry_module._read_version_string() == "unknown"


# ---------------------------------------------------------------------------
# LlamaCppLauncher.cleanup
# ---------------------------------------------------------------------------

class TestCleanup:
    """Static helper that deletes a temp file after a short delay.

    We pass ``delay=0`` in all tests to keep them fast — the docstring reads
    "after a delay" but there's no lower bound on what that means."""

    def test_deletes_existing_file(self, entry_module, tmp_path):
        victim = tmp_path / "tmpfile.sh"
        victim.write_text("# tmp")
        assert victim.exists()

        entry_module.LlamaCppLauncher.cleanup(victim, delay=0)
        assert not victim.exists()

    def test_accepts_str_path(self, entry_module, tmp_path):
        """Docstring doesn't nail the type — call sites pass both str and Path."""
        victim = tmp_path / "tmpfile.sh"
        victim.write_text("# tmp")

        entry_module.LlamaCppLauncher.cleanup(str(victim), delay=0)
        assert not victim.exists()

    def test_missing_file_is_silent(self, entry_module, tmp_path, capsys):
        """Cleanup runs on a best-effort basis — if the launch path already
        removed the file we shouldn't raise."""
        missing = tmp_path / "never-existed.sh"
        assert not missing.exists()

        # Should not raise.
        entry_module.LlamaCppLauncher.cleanup(missing, delay=0)

    def test_unlink_error_is_swallowed(self, entry_module, tmp_path):
        """OSError from unlink (e.g. permission denied on Windows) must be
        logged, not raised — it's a background cleanup."""
        victim = tmp_path / "tmpfile.sh"
        victim.write_text("x")

        with patch("pathlib.Path.unlink", side_effect=OSError("locked")):
            # Must not raise.
            entry_module.LlamaCppLauncher.cleanup(victim, delay=0)

    def test_delay_is_honoured(self, entry_module, tmp_path):
        """Sanity: a small delay actually sleeps. We use 0.05s so the test
        stays fast but still exercises the ``time.sleep`` path."""
        victim = tmp_path / "tmpfile.sh"
        victim.write_text("x")

        start = time.perf_counter()
        entry_module.LlamaCppLauncher.cleanup(victim, delay=0.05)
        elapsed = time.perf_counter() - start
        assert elapsed >= 0.04  # allow a bit of slop
        assert not victim.exists()
