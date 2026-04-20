"""Regression tests for modules.ik_llama.IkLlamaTab.

Covers:
  - get_ik_llama_flags() flag emission rules (boolean, string, defaults).
  - save_to_config() / load_from_config() round trip.
  - trace bindings wiring back to launcher._save_configs /
    launcher._update_default_config_name_if_needed.
  - default values on construction.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure repo root is on sys.path so "modules.*" is importable without
# needing an installed package.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --- tk root bootstrap -------------------------------------------------------
# BooleanVar/StringVar need a default root. Skip the suite cleanly if a Tk
# display cannot be opened (e.g. truly headless CI without Xvfb).
@pytest.fixture(scope="module")
def tk_root():
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - env dependent
        pytest.skip(f"tkinter unavailable: {exc}")
    root.withdraw()
    yield root
    try:
        root.destroy()
    except Exception:
        pass


@pytest.fixture()
def launcher_mock():
    """Stand-in for the real LlamaCppLauncher. Only the two trace callbacks
    are needed - both are invoked every time a tk var is written."""
    m = MagicMock()
    m._save_configs = MagicMock()
    m._update_default_config_name_if_needed = MagicMock()
    return m


@pytest.fixture()
def tab(tk_root, launcher_mock):
    from modules.ik_llama import IkLlamaTab
    return IkLlamaTab(launcher_mock)


# --- construction defaults --------------------------------------------------


class TestDefaults:
    def test_initial_flag_values(self, tab):
        assert tab.rtr_enabled.get() is False
        assert tab.fmoe_enabled.get() is False
        assert tab.ser_value.get() == ""
        assert tab.amb_value.get() == ""
        assert tab.ctk_value.get() == "f16"
        assert tab.ctv_value.get() == "f16"

    def test_no_flags_by_default(self, tab):
        assert tab.get_ik_llama_flags() == []


# --- get_ik_llama_flags -----------------------------------------------------


class TestGetFlags:
    def test_rtr_only(self, tab):
        tab.rtr_enabled.set(True)
        assert tab.get_ik_llama_flags() == ["-rtr"]

    def test_fmoe_only(self, tab):
        tab.fmoe_enabled.set(True)
        assert tab.get_ik_llama_flags() == ["-fmoe"]

    def test_both_booleans(self, tab):
        tab.rtr_enabled.set(True)
        tab.fmoe_enabled.set(True)
        flags = tab.get_ik_llama_flags()
        assert flags == ["-rtr", "-fmoe"]

    def test_ser_value(self, tab):
        tab.ser_value.set("7,1")
        assert tab.get_ik_llama_flags() == ["-ser", "7,1"]

    def test_ser_value_whitespace_is_stripped(self, tab):
        tab.ser_value.set("  5,1  ")
        assert tab.get_ik_llama_flags() == ["-ser", "5,1"]

    def test_ser_empty_string_is_omitted(self, tab):
        tab.ser_value.set("")
        assert "-ser" not in tab.get_ik_llama_flags()

    def test_ser_only_whitespace_is_omitted(self, tab):
        tab.ser_value.set("   ")
        assert "-ser" not in tab.get_ik_llama_flags()

    def test_amb_value(self, tab):
        tab.amb_value.set("512")
        assert tab.get_ik_llama_flags() == ["-amb", "512"]

    def test_amb_whitespace_only_is_omitted(self, tab):
        tab.amb_value.set("\t ")
        assert "-amb" not in tab.get_ik_llama_flags()

    def test_ctk_default_f16_is_omitted(self, tab):
        # f16 is the documented default and must NOT appear on the command line.
        assert tab.ctk_value.get() == "f16"
        assert "-ctk" not in tab.get_ik_llama_flags()

    def test_ctk_non_default_is_emitted(self, tab):
        tab.ctk_value.set("q8_0")
        flags = tab.get_ik_llama_flags()
        assert "-ctk" in flags
        assert flags[flags.index("-ctk") + 1] == "q8_0"

    def test_ctv_default_f16_is_omitted(self, tab):
        assert "-ctv" not in tab.get_ik_llama_flags()

    def test_ctv_non_default_is_emitted(self, tab):
        tab.ctv_value.set("q4_0")
        flags = tab.get_ik_llama_flags()
        assert "-ctv" in flags
        assert flags[flags.index("-ctv") + 1] == "q4_0"

    def test_ctk_empty_string_is_omitted(self, tab):
        # Empty value should not emit a flag even though it's != "f16".
        tab.ctk_value.set("")
        assert "-ctk" not in tab.get_ik_llama_flags()

    @pytest.mark.parametrize(
        "kv",
        ["f32", "bf16", "q4_0", "q4_1", "q5_0", "q5_1", "q6_0", "q8_0", "iq4_nl", "q8_KV"],
    )
    def test_ctk_all_documented_non_default_types(self, tab, kv):
        tab.ctk_value.set(kv)
        flags = tab.get_ik_llama_flags()
        assert ["-ctk", kv] == flags[:2]

    def test_full_combo_ordering(self, tab):
        # The module emits flags in a fixed order: rtr, fmoe, ser, amb, ctk, ctv.
        tab.rtr_enabled.set(True)
        tab.fmoe_enabled.set(True)
        tab.ser_value.set("6,1")
        tab.amb_value.set("512")
        tab.ctk_value.set("q8_0")
        tab.ctv_value.set("q4_0")

        assert tab.get_ik_llama_flags() == [
            "-rtr",
            "-fmoe",
            "-ser", "6,1",
            "-amb", "512",
            "-ctk", "q8_0",
            "-ctv", "q4_0",
        ]

    def test_flags_pair_values_extend_not_append(self, tab):
        # Regression: -ser/-amb/-ctk/-ctv must be followed by their value as a
        # separate argv element, not joined.
        tab.ser_value.set("6,1")
        tab.amb_value.set("512")
        tab.ctk_value.set("q8_0")
        tab.ctv_value.set("q4_0")
        flags = tab.get_ik_llama_flags()
        # Each pair should be exactly two list entries.
        for pair_flag in ("-ser", "-amb", "-ctk", "-ctv"):
            idx = flags.index(pair_flag)
            assert idx + 1 < len(flags), f"{pair_flag} has no value"
            assert not flags[idx + 1].startswith("-"), (
                f"{pair_flag} value looks like a flag: {flags[idx + 1]!r}"
            )


# --- save / load config -----------------------------------------------------


class TestSaveLoadConfig:
    def test_save_returns_all_six_keys(self, tab):
        saved = tab.save_to_config()
        assert set(saved) == {
            "ik_llama_rtr_enabled",
            "ik_llama_fmoe_enabled",
            "ik_llama_ser_value",
            "ik_llama_amb_value",
            "ik_llama_ctk_value",
            "ik_llama_ctv_value",
        }

    def test_save_defaults(self, tab):
        saved = tab.save_to_config()
        assert saved["ik_llama_rtr_enabled"] is False
        assert saved["ik_llama_fmoe_enabled"] is False
        assert saved["ik_llama_ser_value"] == ""
        assert saved["ik_llama_amb_value"] == ""
        assert saved["ik_llama_ctk_value"] == "f16"
        assert saved["ik_llama_ctv_value"] == "f16"

    def test_save_reflects_mutations(self, tab):
        tab.rtr_enabled.set(True)
        tab.fmoe_enabled.set(True)
        tab.ser_value.set("7,1")
        tab.amb_value.set("256")
        tab.ctk_value.set("q8_0")
        tab.ctv_value.set("q4_0")
        saved = tab.save_to_config()
        assert saved == {
            "ik_llama_rtr_enabled": True,
            "ik_llama_fmoe_enabled": True,
            "ik_llama_ser_value": "7,1",
            "ik_llama_amb_value": "256",
            "ik_llama_ctk_value": "q8_0",
            "ik_llama_ctv_value": "q4_0",
        }

    def test_load_round_trip(self, tab):
        payload = {
            "ik_llama_rtr_enabled": True,
            "ik_llama_fmoe_enabled": True,
            "ik_llama_ser_value": "6,1",
            "ik_llama_amb_value": "512",
            "ik_llama_ctk_value": "q8_KV",
            "ik_llama_ctv_value": "iq4_nl",
        }
        tab.load_from_config(payload)
        assert tab.save_to_config() == payload

    def test_load_falls_back_to_defaults_for_missing_keys(self, tab):
        tab.load_from_config({})
        assert tab.rtr_enabled.get() is False
        assert tab.fmoe_enabled.get() is False
        assert tab.ser_value.get() == ""
        assert tab.amb_value.get() == ""
        assert tab.ctk_value.get() == "f16"
        assert tab.ctv_value.get() == "f16"

    def test_load_partial_payload(self, tab):
        tab.load_from_config({"ik_llama_rtr_enabled": True, "ik_llama_ser_value": "5,1"})
        assert tab.rtr_enabled.get() is True
        assert tab.ser_value.get() == "5,1"
        # Unset keys should fall back to defaults
        assert tab.fmoe_enabled.get() is False
        assert tab.ctk_value.get() == "f16"
        assert tab.ctv_value.get() == "f16"


# --- trace wiring -----------------------------------------------------------


class TestTraceWiring:
    """Every tkinter variable writes should trigger the two launcher callbacks."""

    @pytest.mark.parametrize(
        "attr, value",
        [
            ("rtr_enabled", True),
            ("fmoe_enabled", True),
            ("ser_value", "7,1"),
            ("amb_value", "512"),
            ("ctk_value", "q8_0"),
            ("ctv_value", "q4_0"),
        ],
    )
    def test_var_write_invokes_save_and_update(self, tab, launcher_mock, attr, value):
        launcher_mock._save_configs.reset_mock()
        launcher_mock._update_default_config_name_if_needed.reset_mock()
        getattr(tab, attr).set(value)
        assert launcher_mock._save_configs.called
        assert launcher_mock._update_default_config_name_if_needed.called
