"""Tests for ``modules.settings_tab.SettingsTab``.

Focused on the small pure helpers (``_resolve_font_size``, ``_validate_family``,
``_validate_custom_digit``) and the rollback behaviour of ``_persist_ui_settings``.
Widget construction (``setup_settings_tab``) is exercised indirectly via
manual wiring of the input StringVars — we never call ``setup_settings_tab``
since it needs a realized parent frame and a fully-wired launcher, which is
e2e territory.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from modules.settings_tab import SettingsTab


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

@pytest.fixture
def settings_tab(launcher_stub):
    """Instantiate a ``SettingsTab`` backed by the stub launcher.

    ``__init__`` only reads ``app_settings`` and constructs a handful of
    ``tk.StringVar`` instances — it doesn't build widgets, so we don't need
    a parent frame.
    """
    return SettingsTab(launcher_stub)


# ---------------------------------------------------------------------------
# _validate_custom_digit
# ---------------------------------------------------------------------------

class TestValidateCustomDigit:
    """Entry ``validatecommand`` filter — empty / digit-only / <=3 chars."""

    def test_accepts_empty(self, settings_tab):
        assert settings_tab._validate_custom_digit("") is True

    def test_accepts_single_digit(self, settings_tab):
        assert settings_tab._validate_custom_digit("9") is True

    def test_accepts_three_digits(self, settings_tab):
        assert settings_tab._validate_custom_digit("123") is True

    def test_rejects_four_digits(self, settings_tab):
        """Cap lets us keep the Entry width tiny — 32 is the true upper bound
        but 999 is a generous safety net while still fitting the UI."""
        assert settings_tab._validate_custom_digit("1234") is False

    def test_rejects_letters(self, settings_tab):
        assert settings_tab._validate_custom_digit("12a") is False

    def test_rejects_negative_sign(self, settings_tab):
        assert settings_tab._validate_custom_digit("-1") is False

    def test_rejects_decimal(self, settings_tab):
        assert settings_tab._validate_custom_digit("1.5") is False


# ---------------------------------------------------------------------------
# _resolve_font_size
# ---------------------------------------------------------------------------

class TestResolveFontSize:
    """Reads radio + custom entry and returns an int (raises on bad custom)."""

    def test_preset_default_returns_zero(self, settings_tab):
        settings_tab.font_size_choice_var.set("0")
        assert settings_tab._resolve_font_size() == 0

    def test_preset_12_returns_int(self, settings_tab):
        settings_tab.font_size_choice_var.set("12")
        assert settings_tab._resolve_font_size() == 12

    def test_custom_valid(self, settings_tab):
        settings_tab.font_size_choice_var.set("custom")
        settings_tab.font_size_custom_var.set("18")
        assert settings_tab._resolve_font_size() == 18

    def test_custom_strips_whitespace(self, settings_tab):
        settings_tab.font_size_choice_var.set("custom")
        settings_tab.font_size_custom_var.set("  14  ")
        assert settings_tab._resolve_font_size() == 14

    def test_custom_empty_raises(self, settings_tab):
        """Empty custom entry is caught before reaching Tk's font machinery."""
        settings_tab.font_size_choice_var.set("custom")
        settings_tab.font_size_custom_var.set("")
        with pytest.raises(ValueError, match="Enter a number"):
            settings_tab._resolve_font_size()

    def test_custom_non_digit_raises(self, settings_tab):
        """Validate filter blocks typing, but pasted/scripted bad values
        still hit _resolve_font_size — belt-and-braces."""
        settings_tab.font_size_choice_var.set("custom")
        settings_tab.font_size_custom_var.set("abc")
        with pytest.raises(ValueError, match="must be an integer"):
            settings_tab._resolve_font_size()

    def test_custom_zero_raises(self, settings_tab):
        settings_tab.font_size_choice_var.set("custom")
        settings_tab.font_size_custom_var.set("0")
        with pytest.raises(ValueError, match="greater than 0"):
            settings_tab._resolve_font_size()

    def test_custom_at_max_raises(self, settings_tab):
        """``FONT_SIZE_MAX`` is exclusive — 32 should reject."""
        settings_tab.font_size_choice_var.set("custom")
        settings_tab.font_size_custom_var.set(str(SettingsTab.FONT_SIZE_MAX))
        with pytest.raises(ValueError, match="less than"):
            settings_tab._resolve_font_size()

    def test_custom_above_max_raises(self, settings_tab):
        settings_tab.font_size_choice_var.set("custom")
        settings_tab.font_size_custom_var.set("99")
        with pytest.raises(ValueError, match="less than"):
            settings_tab._resolve_font_size()

    def test_custom_just_below_max_ok(self, settings_tab):
        settings_tab.font_size_choice_var.set("custom")
        settings_tab.font_size_custom_var.set(str(SettingsTab.FONT_SIZE_MAX - 1))
        assert settings_tab._resolve_font_size() == SettingsTab.FONT_SIZE_MAX - 1


# ---------------------------------------------------------------------------
# _validate_family
# ---------------------------------------------------------------------------

class TestValidateFamily:
    """Case-insensitive lookup against the cached system-font list."""

    def test_empty_returns_empty(self, settings_tab):
        settings_tab._available_font_families = ["Arial", "DejaVu Sans"]
        assert settings_tab._validate_family("") == ""

    def test_exact_match_preserved(self, settings_tab):
        settings_tab._available_font_families = ["Arial", "DejaVu Sans"]
        assert settings_tab._validate_family("Arial") == "Arial"

    def test_lowercase_canonicalised(self, settings_tab):
        """Typing ``arial`` should persist ``Arial``."""
        settings_tab._available_font_families = ["Arial", "DejaVu Sans"]
        assert settings_tab._validate_family("arial") == "Arial"

    def test_upper_canonicalised(self, settings_tab):
        settings_tab._available_font_families = ["Arial", "DejaVu Sans"]
        assert settings_tab._validate_family("DEJAVU SANS") == "DejaVu Sans"

    def test_unknown_raises(self, settings_tab):
        settings_tab._available_font_families = ["Arial"]
        with pytest.raises(ValueError, match="not available"):
            settings_tab._validate_family("Comic Sans")

    def test_missing_cache_raises_for_non_empty(self, settings_tab):
        """If the cache isn't populated (e.g. setup_settings_tab never ran),
        non-empty family still fails cleanly instead of silently accepting."""
        if hasattr(settings_tab, "_available_font_families"):
            delattr(settings_tab, "_available_font_families")
        with pytest.raises(ValueError):
            settings_tab._validate_family("Arial")


# ---------------------------------------------------------------------------
# __init__ state seeding
# ---------------------------------------------------------------------------

class TestInitSeedsFromAppSettings:
    """Constructor reads previously-persisted values into the tk Vars."""

    def test_blank_settings_gives_defaults(self, launcher_stub):
        launcher_stub.app_settings = {}
        st = SettingsTab(launcher_stub)
        assert st.theme_mode_var.get() == "auto"
        assert st.theme_name_var.get() == ""
        assert st.font_family_var.get() == ""
        assert st.font_size_choice_var.get() == "0"
        assert st.font_size_custom_var.get() == ""

    def test_preset_size_selects_preset(self, launcher_stub):
        launcher_stub.app_settings = {"ui_font_size": 14}
        st = SettingsTab(launcher_stub)
        assert st.font_size_choice_var.get() == "14"
        assert st.font_size_custom_var.get() == ""

    def test_non_preset_size_selects_custom(self, launcher_stub):
        """11 isn't in the preset list, so UI should reopen in Custom mode."""
        launcher_stub.app_settings = {"ui_font_size": 11}
        st = SettingsTab(launcher_stub)
        assert st.font_size_choice_var.get() == "custom"
        assert st.font_size_custom_var.get() == "11"

    def test_zero_size_gives_default(self, launcher_stub):
        launcher_stub.app_settings = {"ui_font_size": 0}
        st = SettingsTab(launcher_stub)
        assert st.font_size_choice_var.get() == "0"

    def test_negative_stored_size_treated_as_default(self, launcher_stub):
        """Tk uses negative sizes for pixel fonts; the settings tab only deals
        in positive point sizes, so anything <=0 must fall back to default."""
        launcher_stub.app_settings = {"ui_font_size": -12}
        st = SettingsTab(launcher_stub)
        assert st.font_size_choice_var.get() == "0"

    def test_theme_mode_preserved(self, launcher_stub):
        launcher_stub.app_settings = {
            "ui_theme_mode": "dark",
            "ui_theme_name": "forest-dark",
            "ui_font_family": "Arial",
        }
        st = SettingsTab(launcher_stub)
        assert st.theme_mode_var.get() == "dark"
        assert st.theme_name_var.get() == "forest-dark"
        assert st.font_family_var.get() == "Arial"


# ---------------------------------------------------------------------------
# _persist_ui_settings — rollback on save failure
# ---------------------------------------------------------------------------

class TestPersistUiSettings:
    """Rollback is the only non-trivial path here.

    The docstring explicitly states we must restore previous values if
    ``_save_configs`` raises — otherwise another save path could later
    persist the rejected state."""

    def test_success_updates_settings(self, launcher_stub, settings_tab, monkeypatch):
        monkeypatch.setattr(
            "modules.settings_tab.messagebox.showerror",
            lambda *a, **kw: None,
        )
        launcher_stub.app_settings = {"ui_theme_mode": "auto"}
        ok = settings_tab._persist_ui_settings(
            {"ui_theme_mode": "dark", "ui_font_size": 14},
            failure_title="x", failure_prefix="y", log_prefix="z",
        )
        assert ok is True
        assert launcher_stub.app_settings["ui_theme_mode"] == "dark"
        assert launcher_stub.app_settings["ui_font_size"] == 14
        launcher_stub._save_configs.assert_called_once()

    def test_save_failure_restores_previous(self, launcher_stub, settings_tab,
                                            monkeypatch):
        """When ``_save_configs`` raises, the settings dict must revert so a
        later Save / on-exit path doesn't persist the user-rejected state."""
        # Suppress the user-facing dialog so the test doesn't block on GUI.
        shown = {}
        def fake_showerror(title, message):
            shown["title"] = title
            shown["message"] = message
        monkeypatch.setattr(
            "modules.settings_tab.messagebox.showerror", fake_showerror,
        )

        launcher_stub.app_settings = {
            "ui_theme_mode": "auto",
            "ui_font_size": 12,
        }
        launcher_stub._save_configs.side_effect = OSError("disk full")

        ok = settings_tab._persist_ui_settings(
            {"ui_theme_mode": "dark", "ui_font_size": 20},
            failure_title="Save failed",
            failure_prefix="Could not persist",
            log_prefix="Settings save error",
        )

        assert ok is False
        assert launcher_stub.app_settings["ui_theme_mode"] == "auto"
        assert launcher_stub.app_settings["ui_font_size"] == 12
        assert shown["title"] == "Save failed"
        assert "disk full" in shown["message"]

    def test_save_failure_removes_previously_absent_keys(self, launcher_stub,
                                                         settings_tab,
                                                         monkeypatch):
        """If a key wasn't in app_settings before the update and the save
        later fails, the rollback must *remove* it rather than leaving the
        attempted value in place."""
        monkeypatch.setattr(
            "modules.settings_tab.messagebox.showerror",
            lambda *a, **kw: None,
        )
        launcher_stub.app_settings = {}
        launcher_stub._save_configs.side_effect = RuntimeError("fs readonly")

        ok = settings_tab._persist_ui_settings(
            {"ui_theme_mode": "dark"},
            failure_title="x", failure_prefix="y", log_prefix="z",
        )

        assert ok is False
        assert "ui_theme_mode" not in launcher_stub.app_settings
