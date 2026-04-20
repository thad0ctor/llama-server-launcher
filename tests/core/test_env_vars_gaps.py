"""Fills test gaps in ``modules/env_vars_module.py``.

Scope: additional behavioural edges of ``EnvironmentalVariablesManager``:
  * ``generate_clear_env_vars_command`` with adversarial custom-var payloads
    that could break AppleScript / cmd.exe quoting if they flowed through
    the Tab's ``_launch_clear_env_terminal`` path unchanged.
  * Case-insensitive duplicate rejection — baseline + boundary cases.
  * Unicode / BOM / special char round-trips.

The :class:`EnvironmentalVariablesTab` class itself is explicitly out of
scope (per audit): its behaviour is dominated by tkinter widget layout
that can't be meaningfully unit-tested.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytestmark = pytest.mark.usefixtures("tk_root")


@pytest.fixture
def env_manager(tk_root):
    from modules.env_vars_module import EnvironmentalVariablesManager
    return EnvironmentalVariablesManager()


# ---------------------------------------------------------------------------
# generate_clear_env_vars_command — adversarial inputs
# ---------------------------------------------------------------------------

class TestGenerateClearEnvVarsCommand:
    def test_output_is_semicolon_joined(self, env_manager):
        """Sanity check: the command is ``a; b; ...; bash``."""
        cmd = env_manager.generate_clear_env_vars_command()
        parts = cmd.split("; ")
        assert parts[-1] == "bash"
        # All predefined vars appear as export statements up front.
        for var in env_manager.PREDEFINED_ENV_VARS:
            assert f"export {var}=0" in parts

    def test_all_echo_lines_present(self, env_manager):
        cmd = env_manager.generate_clear_env_vars_command()
        for var in env_manager.PREDEFINED_ENV_VARS:
            assert f"echo '{var}=0'" in cmd

    def test_custom_var_with_double_quote_does_not_break_format(self, env_manager):
        """A variable name with a literal double quote should not corrupt
        the output structure — we still produce semicolon-joined exports.

        This isn't a security claim; it's a regression guard. If downstream
        (AppleScript / cmd.exe) quoting changes, this test forces us to
        revisit escape handling."""
        env_manager.add_custom_env_var('FOO"BAR', "1")
        cmd = env_manager.generate_clear_env_vars_command()
        # Structure intact: still semicolon-joined.
        assert cmd.endswith("bash")
        # The name appears verbatim in the output — documenting current
        # behaviour so a future safety patch has to update this test.
        assert 'export FOO"BAR=0' in cmd

    def test_custom_var_with_semicolon_is_not_sanitized(self, env_manager):
        """Current behaviour: a semicolon in a var name bleeds into the
        shell command. Document it so a fix is a conscious change, not
        accidental."""
        env_manager.add_custom_env_var("FOO;rm -rf /tmp/x", "1")
        cmd = env_manager.generate_clear_env_vars_command()
        # The raw payload flows through unescaped.
        assert "FOO;rm -rf /tmp/x" in cmd

    def test_custom_var_with_backtick(self, env_manager):
        env_manager.add_custom_env_var("FOO`id`", "1")
        cmd = env_manager.generate_clear_env_vars_command()
        assert "FOO`id`" in cmd

    def test_custom_var_unicode_name(self, env_manager):
        env_manager.add_custom_env_var("日本語_VAR", "1")
        cmd = env_manager.generate_clear_env_vars_command()
        assert "export 日本語_VAR=0" in cmd
        assert "echo '日本語_VAR=0'" in cmd

    def test_whitespace_only_name_does_not_emit_export(self, env_manager):
        # Bypass add_custom_env_var's filter to force the edge case.
        env_manager.custom_env_vars.append(("   ", "1"))
        cmd = env_manager.generate_clear_env_vars_command()
        # The whitespace-only name must be filtered out.
        assert "export    =0" not in cmd
        assert "export =0" not in cmd
        # But predefined vars still appear.
        for var in env_manager.PREDEFINED_ENV_VARS:
            assert f"export {var}=0" in cmd

    def test_custom_var_with_equals_in_name(self, env_manager):
        """A name containing '=' is accepted at the manager level but
        produces a malformed export statement (documenting current behaviour)."""
        env_manager.custom_env_vars.append(("FOO=BAR", "1"))
        cmd = env_manager.generate_clear_env_vars_command()
        # This is documented-weird: 'export FOO=BAR=0' is semantically weird
        # but bash accepts it (sets FOO to BAR=0). Just confirm no crash.
        assert "FOO=BAR" in cmd

    def test_with_empty_custom_list(self, env_manager):
        """No custom vars — the command only contains predefined exports."""
        cmd = env_manager.generate_clear_env_vars_command()
        # Exactly the class predefined set, in iteration order.
        predefined_count = len(env_manager.PREDEFINED_ENV_VARS)
        export_lines = [p for p in cmd.split("; ") if p.startswith("export ")]
        assert len(export_lines) == predefined_count


# ---------------------------------------------------------------------------
# Case-insensitive duplicate rejection (bug fix 2)
# ---------------------------------------------------------------------------

class TestCaseInsensitiveDuplicates:
    def test_exact_duplicate_rejected(self, env_manager):
        assert env_manager.add_custom_env_var("X", "1") is True
        assert env_manager.add_custom_env_var("X", "2") is False
        assert env_manager.custom_env_vars == [("X", "1")]

    def test_different_case_rejected(self, env_manager):
        assert env_manager.add_custom_env_var("HOME", "/root") is True
        assert env_manager.add_custom_env_var("home", "/anywhere") is False
        assert env_manager.add_custom_env_var("Home", "/elsewhere") is False
        assert env_manager.custom_env_vars == [("HOME", "/root")]

    def test_whitespace_padded_duplicate_rejected(self, env_manager):
        assert env_manager.add_custom_env_var("HOME", "/a") is True
        assert env_manager.add_custom_env_var("  home  ", "/b") is False
        assert env_manager.custom_env_vars == [("HOME", "/a")]

    def test_unicode_case_folding_matches_python_upper(self, env_manager):
        # Python's str.upper() folds 'ß' -> 'SS', so the manager's
        # case-insensitive compare treats them as duplicates. Document this.
        assert env_manager.add_custom_env_var("ß", "1") is True
        assert env_manager.add_custom_env_var("SS", "2") is False
        # But Turkish 'İ'.upper() is itself — these two are distinct names.
        assert env_manager.add_custom_env_var("İ", "3") is True
        assert env_manager.add_custom_env_var("i", "4") is True
        assert len(env_manager.custom_env_vars) == 3

    def test_empty_still_rejected(self, env_manager):
        assert env_manager.add_custom_env_var("", "1") is False
        assert env_manager.add_custom_env_var("   ", "1") is False
        assert env_manager.add_custom_env_var("X", "") is False
        assert env_manager.custom_env_vars == []

    def test_duplicate_does_not_override_existing_value(self, env_manager):
        env_manager.add_custom_env_var("A", "original")
        env_manager.add_custom_env_var("a", "replacement")
        # Original must be preserved (reject-new-on-conflict is the contract).
        assert env_manager.custom_env_vars == [("A", "original")]

    def test_update_custom_env_var_does_not_reject_self(self, env_manager):
        # update() should still allow setting a var to a different case of
        # its own name (because it's replacing in place, not adding anew).
        env_manager.add_custom_env_var("A", "1")
        env_manager.update_custom_env_var(0, "a", "2")
        assert env_manager.custom_env_vars == [("a", "2")]


# ---------------------------------------------------------------------------
# Round-trip with duplicates in existing saved state
# ---------------------------------------------------------------------------

def test_load_from_config_dedupes_case_insensitive_duplicates(env_manager):
    """The duplicate-name invariant is class-wide, not just for new adds.
    Legacy configs that happened to contain case-variant duplicates get
    deduped on load — first occurrence wins — so the post-load state
    matches what add_custom_env_var and update_custom_env_var will
    enforce. Otherwise the ``load -> use -> save`` round-trip would let
    forbidden states keep propagating."""
    payload = {
        "environmental_variables": {
            "enabled": True,
            "predefined": {},
            "custom": [
                {"name": "FOO", "value": "1"},
                {"name": "foo", "value": "2"},
                {"name": "BAR", "value": "x"},
            ],
        }
    }
    env_manager.load_from_config(payload)
    # First-occurrence-wins dedup: FOO kept, foo dropped, BAR kept.
    assert env_manager.custom_env_vars == [("FOO", "1"), ("BAR", "x")]
    # Subsequent adds with any casing of the retained names are still rejected.
    assert env_manager.add_custom_env_var("Foo", "3") is False
    assert env_manager.add_custom_env_var("bar", "3") is False


def test_load_from_config_normalises_whitespace_and_drops_empties(env_manager):
    """Whitespace-only names/values should be dropped on load the same way
    add_custom_env_var rejects them, otherwise a hand-edited config could
    seed entries the manager itself would refuse."""
    payload = {
        "environmental_variables": {
            "enabled": True,
            "predefined": {},
            "custom": [
                {"name": "  SPACED  ", "value": "  v  "},
                {"name": "   ", "value": "x"},       # empty name after strip
                {"name": "EMPTY_VAL", "value": "  "},  # empty value after strip
                {"name": 123, "value": "bad-type"},    # non-string name
            ],
        }
    }
    env_manager.load_from_config(payload)
    assert env_manager.custom_env_vars == [("SPACED", "v")]


def test_update_custom_env_var_rejects_case_insensitive_conflict(env_manager):
    """Renaming index 1 to collide (case-insensitively) with index 0 must
    be rejected — otherwise the UI + manager can end up in a state where
    add() forbids a name that update() just created."""
    env_manager.add_custom_env_var("FOO", "1")
    env_manager.add_custom_env_var("BAR", "2")
    # Attempt to rename BAR -> foo (collides with FOO at index 0).
    result = env_manager.update_custom_env_var(1, "foo", "new")
    assert result is False
    # State unchanged.
    assert env_manager.custom_env_vars == [("FOO", "1"), ("BAR", "2")]


def test_update_custom_env_var_same_index_recase_allowed(env_manager):
    """Editing the entry at index ``i`` to a different casing of its own
    name is still allowed — it's an in-place re-save, not a duplicate."""
    env_manager.add_custom_env_var("FOO", "1")
    env_manager.add_custom_env_var("BAR", "2")
    result = env_manager.update_custom_env_var(0, "foo", "1-lower")
    assert result is True
    assert env_manager.custom_env_vars == [("foo", "1-lower"), ("BAR", "2")]


def test_update_custom_env_var_returns_false_on_empty_or_oob(env_manager):
    """Empty name/value and out-of-range index both return False and leave
    state unchanged — mirrors add_custom_env_var."""
    env_manager.add_custom_env_var("X", "1")
    assert env_manager.update_custom_env_var(0, "", "v") is False
    assert env_manager.update_custom_env_var(0, "N", "") is False
    assert env_manager.update_custom_env_var(99, "N", "v") is False
    assert env_manager.custom_env_vars == [("X", "1")]
