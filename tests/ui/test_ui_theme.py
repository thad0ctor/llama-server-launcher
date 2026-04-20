"""Tests for ``modules.ui_theme``.

Targets the pure helpers (``_looks_dark``, ``_pick_theme``, ``_is_dark_palette``,
``_palette_for``) and the Tk-backed queries (``list_available_themes``,
``list_font_families``, ``apply_fonts``). We mock ``_detect_os_dark_mode``
rather than poking at real OS state — it shells out to ``defaults`` /
``gsettings`` which isn't deterministic in CI.

Widget re-colouring / ``apply_theme`` are partially exercised via ``apply_fonts``
and the theme-picker helpers, but the full colour-push pipeline
(``_apply_style_palette``) is treated as e2e because it only produces visible
effects — no easy return value to assert against.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from modules import ui_theme


# ---------------------------------------------------------------------------
# _looks_dark
# ---------------------------------------------------------------------------

class TestLooksDark:
    def test_empty_is_not_dark(self):
        assert ui_theme._looks_dark("") is False

    def test_none_is_not_dark(self):
        """None happens when Tk has no active theme — must not crash."""
        assert ui_theme._looks_dark(None) is False

    def test_dark_substring(self):
        assert ui_theme._looks_dark("forest-dark") is True

    def test_black_substring(self):
        assert ui_theme._looks_dark("black") is True

    def test_highcontrast_substring(self):
        assert ui_theme._looks_dark("HighContrast") is True

    def test_night_suffix(self):
        assert ui_theme._looks_dark("solarized-night") is True

    def test_mixed_case(self):
        assert ui_theme._looks_dark("FOREST-DARK") is True

    def test_light_not_dark(self):
        assert ui_theme._looks_dark("forest-light") is False

    def test_vista_not_dark(self):
        assert ui_theme._looks_dark("vista") is False


# ---------------------------------------------------------------------------
# _pick_theme
# ---------------------------------------------------------------------------

class TestPickTheme:
    def test_picks_first_available_from_preferred(self):
        available = ["alt", "clam", "default"]
        preferred = ["vista", "clam", "alt"]
        # Skips vista (not available), picks clam
        assert ui_theme._pick_theme(available, preferred) == "clam"

    def test_fallback_used_when_no_preferred_match(self):
        available = ["weirdtheme"]
        preferred = ["vista", "clam"]
        assert ui_theme._pick_theme(available, preferred, fallback="weirdtheme") == "weirdtheme"

    def test_returns_first_available_when_no_match_no_fallback(self):
        available = ["themeA", "themeB"]
        preferred = ["notfound"]
        assert ui_theme._pick_theme(available, preferred) == "themeA"

    def test_empty_available_returns_none(self):
        assert ui_theme._pick_theme([], ["vista"]) is None

    def test_fallback_ignored_if_not_available(self):
        """A bogus fallback shouldn't cause theme_use(None) later."""
        available = ["themeA"]
        preferred = ["notfound"]
        assert ui_theme._pick_theme(available, preferred, fallback="alsomissing") == "themeA"


# ---------------------------------------------------------------------------
# _is_dark_palette
# ---------------------------------------------------------------------------

class TestIsDarkPalette:
    def test_dark_palette_recognised(self):
        assert ui_theme._is_dark_palette(ui_theme.DARK_PALETTE) is True

    def test_light_palette_recognised(self):
        assert ui_theme._is_dark_palette(ui_theme.LIGHT_PALETTE) is False

    def test_none_returns_false(self):
        assert ui_theme._is_dark_palette(None) is False

    def test_empty_returns_false(self):
        assert ui_theme._is_dark_palette({}) is False

    def test_malformed_bg_returns_false(self):
        """Non-hex bg shouldn't crash the luminance math."""
        assert ui_theme._is_dark_palette({"bg": "not-a-color"}) is False

    def test_invalid_hex_returns_false(self):
        assert ui_theme._is_dark_palette({"bg": "#zzzzzz"}) is False

    def test_short_hex_returns_false(self):
        """Accepts only 6-digit hex — 3-digit shorthand isn't supported."""
        assert ui_theme._is_dark_palette({"bg": "#123"}) is False

    def test_pure_black(self):
        assert ui_theme._is_dark_palette({"bg": "#000000"}) is True

    def test_pure_white(self):
        assert ui_theme._is_dark_palette({"bg": "#ffffff"}) is False

    def test_boundary(self):
        """~0.35 luminance is the cutoff. #595959 has lum≈0.349, should be dark."""
        assert ui_theme._is_dark_palette({"bg": "#595959"}) is True
        # #606060 is lum≈0.376, should be light
        assert ui_theme._is_dark_palette({"bg": "#606060"}) is False


# ---------------------------------------------------------------------------
# _palette_for
# ---------------------------------------------------------------------------

class TestPaletteFor:
    def test_explicit_dark_mode_uses_dark_palette(self):
        """Mode=dark with a light theme must still get a dark palette — that's
        the whole point of the mode override."""
        p = ui_theme._palette_for("vista", "dark")
        assert ui_theme._is_dark_palette(p) is True

    def test_explicit_dark_mode_prefers_themed_dark(self):
        """If the theme has its own dark palette, we should get IT (so we keep
        the theme's accent colours) rather than the generic DARK_PALETTE."""
        p = ui_theme._palette_for("win11dark", "dark")
        assert p is ui_theme.THEME_PALETTES["win11dark"]

    def test_explicit_light_mode_uses_light_palette(self):
        p = ui_theme._palette_for("forest-dark", "light")
        assert ui_theme._is_dark_palette(p) is False

    def test_explicit_light_mode_prefers_themed_light(self):
        p = ui_theme._palette_for("win11light", "light")
        assert p is ui_theme.THEME_PALETTES["win11light"]

    def test_auto_known_theme_uses_themed(self):
        p = ui_theme._palette_for("forest-light", "auto")
        assert p is ui_theme.THEME_PALETTES["forest-light"]

    def test_auto_unknown_dark_sounding_theme_gets_dark(self):
        """Third-party 'azure-dark' isn't in THEME_PALETTES — we must infer
        dark from the name so text stays readable."""
        p = ui_theme._palette_for("azure-dark", "auto")
        assert ui_theme._is_dark_palette(p) is True

    def test_auto_unknown_light_theme_gets_light(self):
        p = ui_theme._palette_for("made-up-theme", "auto")
        assert ui_theme._is_dark_palette(p) is False

    def test_specific_honours_theme_name_dark(self):
        """``specific`` mode is treated the same as auto — the theme name drives
        the palette choice."""
        p = ui_theme._palette_for("sun-valley-dark", "specific")
        assert ui_theme._is_dark_palette(p) is True


# ---------------------------------------------------------------------------
# list_available_themes
# ---------------------------------------------------------------------------

class TestListAvailableThemes:
    def test_returns_a_list(self, tk_root):
        themes = ui_theme.list_available_themes(tk_root)
        assert isinstance(themes, list)

    def test_returns_nonempty_on_real_tk(self, tk_root):
        """Tk always ships at least ``default`` and ``clam`` — if neither is
        there something is very wrong with the env."""
        themes = ui_theme.list_available_themes(tk_root)
        assert len(themes) > 0

    def test_contains_default(self, tk_root):
        themes = ui_theme.list_available_themes(tk_root)
        assert "default" in themes

    def test_returns_empty_on_tclerror(self):
        """If someone passes in a non-Tk root the helper must not propagate
        the TclError — the settings tab calls this at construction time."""
        import tkinter as tk
        class FakeRoot:
            pass
        # ttk.Style(FakeRoot()) raises; just mock the whole call.
        with patch("modules.ui_theme.ttk.Style",
                   side_effect=tk.TclError("no root")):
            assert ui_theme.list_available_themes(FakeRoot()) == []


# ---------------------------------------------------------------------------
# list_font_families
# ---------------------------------------------------------------------------

class TestListFontFamilies:
    def test_returns_sorted_list(self, tk_root):
        families = ui_theme.list_font_families(tk_root)
        assert families == sorted(families)

    def test_excludes_at_prefixed(self, tk_root):
        """On Windows Tk surfaces ``@Family`` vertical fonts; we filter them."""
        families = ui_theme.list_font_families(tk_root)
        assert not any(f.startswith("@") for f in families)

    def test_excludes_empty_strings(self, tk_root):
        families = ui_theme.list_font_families(tk_root)
        assert "" not in families


# ---------------------------------------------------------------------------
# apply_fonts
# ---------------------------------------------------------------------------

class TestApplyFonts:
    def test_noop_when_both_empty(self, tk_root):
        """Empty family + zero size means 'leave fonts alone'."""
        from tkinter import font as tkfont
        before = tkfont.nametofont("TkDefaultFont", root=tk_root).actual()
        ui_theme.apply_fonts(tk_root, family="", size=0)
        after = tkfont.nametofont("TkDefaultFont", root=tk_root).actual()
        assert before == after

    def test_applies_size_to_default_font(self, tk_root):
        """Positive size with blank family should update TkDefaultFont size."""
        from tkinter import font as tkfont

        original_size = int(
            tkfont.nametofont("TkDefaultFont", root=tk_root).cget("size") or 0,
        )
        try:
            ui_theme.apply_fonts(tk_root, family="", size=15)
            new_size = int(
                tkfont.nametofont("TkDefaultFont", root=tk_root).cget("size"),
            )
            assert new_size == 15
        finally:
            # Restore whatever the fixture started with to avoid bleed between tests.
            if original_size:
                tkfont.nametofont("TkDefaultFont", root=tk_root).configure(size=original_size)

    def test_negative_size_ignored(self, tk_root):
        """A negative size means 'pixel units' in Tk vocabulary — this function
        only accepts positive point values, anything else should be a no-op."""
        from tkinter import font as tkfont
        original_size = int(
            tkfont.nametofont("TkDefaultFont", root=tk_root).cget("size") or 0,
        )
        ui_theme.apply_fonts(tk_root, family="", size=-5)
        current = int(
            tkfont.nametofont("TkDefaultFont", root=tk_root).cget("size") or 0,
        )
        assert current == original_size


# ---------------------------------------------------------------------------
# _detect_os_dark_mode — mocked per the audit plan
# ---------------------------------------------------------------------------

class TestDetectOsDarkMode:
    """Exercise the public behaviour by mocking the subprocess probe."""

    def test_returns_none_when_no_tool_available(self, monkeypatch):
        """If neither defaults/gsettings is on PATH the helper returns None,
        letting ``apply_theme`` fall back to the startup theme."""
        monkeypatch.setattr("modules.ui_theme._run_resolved", lambda *a, **kw: None)
        monkeypatch.setattr("modules.ui_theme.sys.platform", "linux")
        assert ui_theme._detect_os_dark_mode() is None

    def test_linux_gsettings_dark(self, monkeypatch):
        """gsettings emits something like `'prefer-dark'` (with quotes)."""
        class FakeResp:
            returncode = 0
            stdout = "'prefer-dark'\n"

        monkeypatch.setattr("modules.ui_theme._run_resolved",
                            lambda *a, **kw: FakeResp())
        monkeypatch.setattr("modules.ui_theme.sys.platform", "linux")
        assert ui_theme._detect_os_dark_mode() is True

    def test_linux_gsettings_light(self, monkeypatch):
        class FakeResp:
            returncode = 0
            stdout = "'default'\n"

        monkeypatch.setattr("modules.ui_theme._run_resolved",
                            lambda *a, **kw: FakeResp())
        monkeypatch.setattr("modules.ui_theme.sys.platform", "linux")
        assert ui_theme._detect_os_dark_mode() is False


# ---------------------------------------------------------------------------
# apply_ui_preferences — the top-level entry point
# ---------------------------------------------------------------------------

class TestApplyUiPreferences:
    """Make sure the convenience wrapper routes to both halves."""

    def test_calls_both_fonts_and_theme(self, tk_root, monkeypatch):
        calls = []
        monkeypatch.setattr("modules.ui_theme.apply_fonts",
                            lambda root, family="", size=0: calls.append(("fonts", family, size)))
        monkeypatch.setattr("modules.ui_theme.apply_theme",
                            lambda root, mode="auto", explicit_theme=None:
                                calls.append(("theme", mode, explicit_theme)) or ("clam", False))

        result = ui_theme.apply_ui_preferences(
            tk_root, theme_mode="dark",
            explicit_theme=None, font_family="Arial", font_size=14,
        )

        assert calls == [("fonts", "Arial", 14), ("theme", "dark", None)]
        assert result == ("clam", False)
