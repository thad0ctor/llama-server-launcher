"""Tests for ``modules.env_vars_module.EnvironmentalVariablesTab``.

The ``Manager`` class is already covered by ``tests/core/test_env_vars.py``.
This file focuses on:
  - The new ``autosetup=False`` / :meth:`build_ui` split so the Tab can be
    instantiated without a realized parent frame for partial testing.
  - The duplicate-name rejection in :meth:`_add_custom_variable`, which is a
    pure logic check that happens before any widget mutation.

Anything that requires the widget tree (scroll behaviour, predefined-var
checkbox grid, preview text) is deferred — those are e2e-only.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from modules.env_vars_module import (
    EnvironmentalVariablesManager,
    EnvironmentalVariablesTab,
)


# ---------------------------------------------------------------------------
# deferred construction — new API added for testability
# ---------------------------------------------------------------------------

class TestDeferredConstruction:
    """``autosetup=False`` lets callers defer the widget build.

    This matters for two reasons:
      1. It makes the class testable in a minimal Tk environment — we can
         instantiate the object without a parent frame that's realized.
      2. It documents the contract: the default path still builds eagerly.
    """

    def test_default_autosetup_builds_ui(self, tk_root):
        """Back-compat: the old no-kwarg signature still builds immediately."""
        import tkinter as tk
        parent = tk.Frame(tk_root)
        manager = EnvironmentalVariablesManager()

        tab = EnvironmentalVariablesTab(parent, manager)

        assert tab._ui_built is True
        # Widgets should exist on the tab instance.
        assert hasattr(tab, "enable_checkbox")
        assert hasattr(tab, "preview_text")
        parent.destroy()

    def test_autosetup_false_skips_widget_construction(self, tk_root):
        """Deferred mode must leave the widget attributes unset so a caller
        can distinguish 'ready' from 'built' state."""
        import tkinter as tk
        parent = tk.Frame(tk_root)
        manager = EnvironmentalVariablesManager()

        tab = EnvironmentalVariablesTab(parent, manager, autosetup=False)

        assert tab._ui_built is False
        # Non-widget attributes — tkinter StringVars etc. — are still available.
        assert tab.custom_var_name.get() == ""
        assert tab.custom_var_value.get() == ""
        assert tab.predefined_var_states == {}
        # Widget-y attributes must NOT exist yet.
        assert not hasattr(tab, "enable_checkbox")
        assert not hasattr(tab, "preview_text")
        parent.destroy()

    def test_build_ui_is_idempotent(self, tk_root):
        """Calling build_ui twice must be a no-op on the second call —
        otherwise we'd leak duplicate widgets into the parent frame."""
        import tkinter as tk
        parent = tk.Frame(tk_root)
        manager = EnvironmentalVariablesManager()

        tab = EnvironmentalVariablesTab(parent, manager, autosetup=False)
        tab.build_ui()
        first_checkbox = tab.enable_checkbox

        tab.build_ui()  # second call
        assert tab.enable_checkbox is first_checkbox
        parent.destroy()

    def test_build_ui_after_defer_completes(self, tk_root):
        """When the parent is realized, deferred build succeeds normally."""
        import tkinter as tk
        parent = tk.Frame(tk_root)
        manager = EnvironmentalVariablesManager()

        tab = EnvironmentalVariablesTab(parent, manager, autosetup=False)
        tab.build_ui()

        assert tab._ui_built is True
        assert hasattr(tab, "preview_text")
        parent.destroy()


# ---------------------------------------------------------------------------
# _add_custom_variable — duplicate / empty rejection BEFORE widget mutation
# ---------------------------------------------------------------------------

class TestAddCustomVariableValidation:
    """_add_custom_variable early-returns on bad input *before* touching any
    widgets. We patch ``messagebox.showwarning`` so the warning dialog doesn't
    pop, and assert the manager state is unchanged.
    """

    @pytest.fixture
    def wired_tab(self, tk_root):
        """Fully-built tab so widget paths are exercised too."""
        import tkinter as tk
        parent = tk.Frame(tk_root)
        manager = EnvironmentalVariablesManager()
        tab = EnvironmentalVariablesTab(parent, manager)
        yield tab
        parent.destroy()

    def test_empty_name_rejected(self, wired_tab, monkeypatch):
        warnings = []
        monkeypatch.setattr(
            "modules.env_vars_module.messagebox.showwarning",
            lambda title, msg: warnings.append((title, msg)),
        )
        wired_tab.custom_var_name.set("")
        wired_tab.custom_var_value.set("1")

        wired_tab._add_custom_variable()

        assert wired_tab.env_manager.custom_env_vars == []
        assert warnings[0][0] == "Invalid Input"

    def test_empty_value_rejected(self, wired_tab, monkeypatch):
        warnings = []
        monkeypatch.setattr(
            "modules.env_vars_module.messagebox.showwarning",
            lambda title, msg: warnings.append((title, msg)),
        )
        wired_tab.custom_var_name.set("MY_VAR")
        wired_tab.custom_var_value.set("")

        wired_tab._add_custom_variable()

        assert wired_tab.env_manager.custom_env_vars == []
        assert warnings[0][0] == "Invalid Input"

    def test_duplicate_rejected_case_insensitive(self, wired_tab, monkeypatch):
        """UI-layer treats FOO and foo as a conflict; confirm the tab doesn't
        bypass this check. Manager's own API also rejects duplicates (see the
        manager tests), so this is defence in depth."""
        warnings = []
        monkeypatch.setattr(
            "modules.env_vars_module.messagebox.showwarning",
            lambda title, msg: warnings.append((title, msg)),
        )
        wired_tab.env_manager.add_custom_env_var("FOO", "1")

        wired_tab.custom_var_name.set("foo")
        wired_tab.custom_var_value.set("2")
        wired_tab._add_custom_variable()

        # Still just the original entry
        assert wired_tab.env_manager.custom_env_vars == [("FOO", "1")]
        assert warnings[0][0] == "Duplicate Variable"

    def test_valid_add_clears_inputs_and_appends(self, wired_tab, monkeypatch):
        monkeypatch.setattr(
            "modules.env_vars_module.messagebox.showwarning",
            lambda title, msg: None,
        )
        wired_tab.custom_var_name.set("NEW_VAR")
        wired_tab.custom_var_value.set("value1")
        wired_tab._add_custom_variable()

        assert ("NEW_VAR", "value1") in wired_tab.env_manager.custom_env_vars
        # Inputs should be cleared so the user can type the next one.
        assert wired_tab.custom_var_name.get() == ""
        assert wired_tab.custom_var_value.get() == ""


# ---------------------------------------------------------------------------
# _remove_custom_variable — no-selection warning
# ---------------------------------------------------------------------------

class TestRemoveCustomVariable:
    @pytest.fixture
    def wired_tab(self, tk_root):
        import tkinter as tk
        parent = tk.Frame(tk_root)
        manager = EnvironmentalVariablesManager()
        tab = EnvironmentalVariablesTab(parent, manager)
        yield tab
        parent.destroy()

    def test_no_selection_warns(self, wired_tab, monkeypatch):
        warnings = []
        monkeypatch.setattr(
            "modules.env_vars_module.messagebox.showwarning",
            lambda title, msg: warnings.append((title, msg)),
        )
        wired_tab._remove_custom_variable()
        assert warnings[0][0] == "No Selection"
