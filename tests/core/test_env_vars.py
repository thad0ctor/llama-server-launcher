"""Regression tests for ``modules/env_vars_module.py``.

Scope: public API of :class:`EnvironmentalVariablesManager` — the non-UI half
of the module. The :class:`EnvironmentalVariablesTab` GUI class is out of scope
because its behaviour is bound to tkinter widget state.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure imports work even if test is run directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Require a Tk root for the BooleanVar that EnvironmentalVariablesManager
# instantiates in __init__. The session-scoped ``tk_root`` fixture creates it.
pytestmark = pytest.mark.usefixtures("tk_root")


@pytest.fixture
def env_manager(tk_root):
    """Fresh EnvironmentalVariablesManager for each test."""
    from modules.env_vars_module import EnvironmentalVariablesManager
    return EnvironmentalVariablesManager()


# ---------------------------------------------------------------------------
# Construction & initial state
# ---------------------------------------------------------------------------

def test_initial_state_is_empty_and_disabled(env_manager):
    assert env_manager.enabled_predefined_vars == {}
    assert env_manager.custom_env_vars == []
    assert env_manager.env_vars_enabled.get() is False


def test_predefined_vars_class_constant_has_expected_keys(env_manager):
    # Acts as a canary: schema changes should require a conscious update here.
    expected = {
        "GGML_CUDA_FORCE_MMQ",
        "GGML_CUDA_F16",
        "GGML_CUDA_GRAPH_FORCE",
        "GGML_CUDA_FORCE_CUBLAS",
        "GGML_CUDA_DMMV_F16",
        "GGML_CUDA_ALLOW_FP16_REDUCE",
    }
    assert set(env_manager.PREDEFINED_ENV_VARS.keys()) == expected
    for key, meta in env_manager.PREDEFINED_ENV_VARS.items():
        assert "default" in meta, key
        assert "description" in meta, key
        assert isinstance(meta["default"], str)


# ---------------------------------------------------------------------------
# get_enabled_env_vars
# ---------------------------------------------------------------------------

def test_get_enabled_env_vars_returns_empty_when_disabled(env_manager):
    env_manager.set_predefined_var_enabled("GGML_CUDA_F16", True, "1")
    env_manager.add_custom_env_var("FOO", "bar")
    # Not enabled at the top level.
    assert env_manager.env_vars_enabled.get() is False
    assert env_manager.get_enabled_env_vars() == {}


def test_get_enabled_env_vars_includes_predefined_and_custom(env_manager):
    env_manager.env_vars_enabled.set(True)
    env_manager.set_predefined_var_enabled("GGML_CUDA_F16", True, "1")
    env_manager.add_custom_env_var("MY_VAR", "hello")
    result = env_manager.get_enabled_env_vars()
    assert result == {"GGML_CUDA_F16": "1", "MY_VAR": "hello"}


def test_get_enabled_env_vars_strips_whitespace_in_custom(env_manager):
    env_manager.env_vars_enabled.set(True)
    # Insert directly to bypass the add-time strip; checks the get-time strip.
    env_manager.custom_env_vars.append(("  NAME  ", "  value  "))
    result = env_manager.get_enabled_env_vars()
    assert result == {"NAME": "value"}


def test_get_enabled_env_vars_skips_empty_custom_names(env_manager):
    env_manager.env_vars_enabled.set(True)
    env_manager.custom_env_vars.append(("   ", "value"))
    env_manager.custom_env_vars.append(("NAME", "   "))
    assert env_manager.get_enabled_env_vars() == {}


def test_get_env_vars_for_launch_mirrors_get_enabled_env_vars(env_manager):
    env_manager.env_vars_enabled.set(True)
    env_manager.set_predefined_var_enabled("GGML_CUDA_F16", True, "2")
    assert env_manager.get_env_vars_for_launch() == env_manager.get_enabled_env_vars()


# ---------------------------------------------------------------------------
# Predefined variables management
# ---------------------------------------------------------------------------

def test_set_predefined_var_enabled_uses_default_when_value_is_none(env_manager):
    env_manager.set_predefined_var_enabled("GGML_CUDA_F16", True)
    assert env_manager.enabled_predefined_vars["GGML_CUDA_F16"] == "1"


def test_set_predefined_var_enabled_accepts_explicit_value(env_manager):
    env_manager.set_predefined_var_enabled("GGML_CUDA_F16", True, "custom-value")
    assert env_manager.enabled_predefined_vars["GGML_CUDA_F16"] == "custom-value"


def test_set_predefined_var_enabled_can_disable(env_manager):
    env_manager.set_predefined_var_enabled("GGML_CUDA_F16", True, "1")
    assert env_manager.is_predefined_var_enabled("GGML_CUDA_F16") is True
    env_manager.set_predefined_var_enabled("GGML_CUDA_F16", False)
    assert env_manager.is_predefined_var_enabled("GGML_CUDA_F16") is False


def test_set_predefined_var_enabled_disable_when_not_enabled_is_noop(env_manager):
    # Should not raise KeyError when disabling a var that was never enabled.
    env_manager.set_predefined_var_enabled("GGML_CUDA_F16", False)
    assert env_manager.enabled_predefined_vars == {}


@pytest.mark.parametrize("var_name", list({
    "GGML_CUDA_FORCE_MMQ",
    "GGML_CUDA_F16",
    "GGML_CUDA_GRAPH_FORCE",
    "GGML_CUDA_FORCE_CUBLAS",
    "GGML_CUDA_DMMV_F16",
    "GGML_CUDA_ALLOW_FP16_REDUCE",
}))
def test_get_predefined_var_value_returns_default_when_disabled(env_manager, var_name):
    # When not enabled, should return the class default (not "").
    assert env_manager.get_predefined_var_value(var_name) == \
        env_manager.PREDEFINED_ENV_VARS[var_name]["default"]


def test_get_predefined_var_value_returns_stored_value_when_enabled(env_manager):
    env_manager.set_predefined_var_enabled("GGML_CUDA_F16", True, "7")
    assert env_manager.get_predefined_var_value("GGML_CUDA_F16") == "7"


def test_set_predefined_var_enabled_unknown_name_raises_key_error(env_manager):
    # Setting an unknown name with value=None requires a lookup in the class
    # constant, which raises KeyError — preserve current behaviour.
    with pytest.raises(KeyError):
        env_manager.set_predefined_var_enabled("NOT_A_REAL_VAR", True)


def test_get_predefined_var_value_unknown_name_raises_key_error(env_manager):
    # Unknown var + not enabled => falls through to PREDEFINED_ENV_VARS lookup.
    with pytest.raises(KeyError):
        env_manager.get_predefined_var_value("NOT_A_REAL_VAR")


# ---------------------------------------------------------------------------
# Custom variables management
# ---------------------------------------------------------------------------

def test_add_custom_env_var_stores_stripped_values(env_manager):
    env_manager.add_custom_env_var("  NAME  ", "  value  ")
    assert env_manager.custom_env_vars == [("NAME", "value")]


@pytest.mark.parametrize("name, value", [
    ("", "value"),
    ("NAME", ""),
    ("   ", "   "),
    ("NAME", "   "),
    ("   ", "value"),
])
def test_add_custom_env_var_rejects_empty_parts(env_manager, name, value):
    env_manager.add_custom_env_var(name, value)
    assert env_manager.custom_env_vars == []


def test_add_custom_env_var_rejects_case_insensitive_duplicate(env_manager):
    # The manager now rejects case-insensitive duplicates to match the UI's
    # duplicate-check rule (fixes a consistency bug where the Tab layer
    # rejected FOO/foo but the manager's direct API silently accepted them).
    assert env_manager.add_custom_env_var("FOO", "1") is True
    assert env_manager.add_custom_env_var("FOO", "2") is False
    assert env_manager.add_custom_env_var("foo", "3") is False
    assert env_manager.add_custom_env_var("Foo", "4") is False
    assert env_manager.custom_env_vars == [("FOO", "1")]


def test_remove_custom_env_var_removes_by_index(env_manager):
    env_manager.add_custom_env_var("A", "1")
    env_manager.add_custom_env_var("B", "2")
    env_manager.add_custom_env_var("C", "3")
    env_manager.remove_custom_env_var(1)
    assert [n for n, _ in env_manager.custom_env_vars] == ["A", "C"]


@pytest.mark.parametrize("index", [-1, 5, 100])
def test_remove_custom_env_var_out_of_range_is_noop(env_manager, index):
    env_manager.add_custom_env_var("A", "1")
    env_manager.remove_custom_env_var(index)
    assert env_manager.custom_env_vars == [("A", "1")]


def test_update_custom_env_var_replaces_entry(env_manager):
    env_manager.add_custom_env_var("A", "1")
    env_manager.update_custom_env_var(0, "B", "2")
    assert env_manager.custom_env_vars == [("B", "2")]


def test_update_custom_env_var_strips(env_manager):
    env_manager.add_custom_env_var("A", "1")
    env_manager.update_custom_env_var(0, "  B  ", "  2  ")
    assert env_manager.custom_env_vars == [("B", "2")]


def test_update_custom_env_var_out_of_range_is_noop(env_manager):
    env_manager.add_custom_env_var("A", "1")
    env_manager.update_custom_env_var(42, "X", "9")
    assert env_manager.custom_env_vars == [("A", "1")]


# ---------------------------------------------------------------------------
# save_to_config / load_from_config round-trip
# ---------------------------------------------------------------------------

def test_save_to_config_returns_full_schema(env_manager):
    cfg = env_manager.save_to_config()
    assert "environmental_variables" in cfg
    block = cfg["environmental_variables"]
    assert set(block.keys()) == {"enabled", "predefined", "custom"}
    assert set(block["predefined"].keys()) == set(env_manager.PREDEFINED_ENV_VARS.keys())
    for var_name, var_data in block["predefined"].items():
        assert set(var_data.keys()) == {"enabled", "value"}


def test_save_to_config_reflects_current_state(env_manager):
    env_manager.env_vars_enabled.set(True)
    env_manager.set_predefined_var_enabled("GGML_CUDA_F16", True, "9")
    env_manager.add_custom_env_var("MY", "val")

    cfg = env_manager.save_to_config()["environmental_variables"]
    assert cfg["enabled"] is True
    assert cfg["predefined"]["GGML_CUDA_F16"] == {"enabled": True, "value": "9"}
    assert cfg["predefined"]["GGML_CUDA_FORCE_MMQ"]["enabled"] is False
    assert cfg["custom"] == [{"name": "MY", "value": "val"}]


def test_load_from_config_round_trip(env_manager):
    env_manager.env_vars_enabled.set(True)
    env_manager.set_predefined_var_enabled("GGML_CUDA_F16", True, "3")
    env_manager.set_predefined_var_enabled("GGML_CUDA_GRAPH_FORCE", True, "1")
    env_manager.add_custom_env_var("ALPHA", "a")
    env_manager.add_custom_env_var("BETA", "b")

    saved = env_manager.save_to_config()

    # Fresh manager receives the saved data:
    from modules.env_vars_module import EnvironmentalVariablesManager
    other = EnvironmentalVariablesManager()
    other.load_from_config(saved)

    assert other.env_vars_enabled.get() is True
    assert other.enabled_predefined_vars == {
        "GGML_CUDA_F16": "3",
        "GGML_CUDA_GRAPH_FORCE": "1",
    }
    assert other.custom_env_vars == [("ALPHA", "a"), ("BETA", "b")]
    # And the serialisation should be identical.
    assert other.save_to_config() == saved


def test_load_from_config_handles_missing_block(env_manager):
    env_manager.env_vars_enabled.set(True)
    env_manager.set_predefined_var_enabled("GGML_CUDA_F16", True, "1")
    env_manager.add_custom_env_var("A", "1")

    # Load from a config dict that has no environmental_variables key.
    env_manager.load_from_config({"other_key": 123})
    assert env_manager.env_vars_enabled.get() is False
    assert env_manager.enabled_predefined_vars == {}
    assert env_manager.custom_env_vars == []


def test_load_from_config_ignores_malformed_custom_entries(env_manager):
    payload = {
        "environmental_variables": {
            "enabled": True,
            "predefined": {},
            "custom": [
                {"name": "GOOD", "value": "1"},
                {"name": "missing_value"},       # missing 'value' key
                {"value": "missing_name"},       # missing 'name' key
                "not_a_dict",                    # wrong type
                42,                              # wrong type
            ],
        }
    }
    env_manager.load_from_config(payload)
    assert env_manager.custom_env_vars == [("GOOD", "1")]


def test_load_from_config_uses_class_default_for_missing_value(env_manager):
    payload = {
        "environmental_variables": {
            "enabled": True,
            "predefined": {
                "GGML_CUDA_F16": {"enabled": True},  # no 'value'
            },
            "custom": [],
        }
    }
    env_manager.load_from_config(payload)
    assert env_manager.enabled_predefined_vars["GGML_CUDA_F16"] == "1"


def test_load_from_config_skips_disabled_predefined(env_manager):
    payload = {
        "environmental_variables": {
            "enabled": False,
            "predefined": {
                "GGML_CUDA_F16": {"enabled": False, "value": "42"},
            },
            "custom": [],
        }
    }
    env_manager.load_from_config(payload)
    # A disabled entry should not be loaded into enabled_predefined_vars.
    assert "GGML_CUDA_F16" not in env_manager.enabled_predefined_vars


def test_load_from_config_supports_unknown_predefined_keys(env_manager):
    # Custom/unknown predefined key (e.g. from a future version) should still
    # be tolerated: we don't crash, and we do our best to persist it.
    payload = {
        "environmental_variables": {
            "enabled": True,
            "predefined": {
                "SOME_FUTURE_VAR": {"enabled": True, "value": "xyz"},
            },
            "custom": [],
        }
    }
    env_manager.load_from_config(payload)
    assert env_manager.enabled_predefined_vars.get("SOME_FUTURE_VAR") == "xyz"


def test_round_trip_preserves_unicode_and_special_chars(env_manager):
    env_manager.env_vars_enabled.set(True)
    env_manager.add_custom_env_var("PATH_ISH", "/tmp/日本語/α:β")
    env_manager.add_custom_env_var("LITERAL_DOLLAR", "$NOT_EXPANDED")
    env_manager.add_custom_env_var("EQUALS", "a=b=c")
    saved = env_manager.save_to_config()

    # Serialise via json to verify it round-trips through a file-ish path.
    import json as _json
    serialised = _json.dumps(saved, ensure_ascii=False)
    restored = _json.loads(serialised)

    from modules.env_vars_module import EnvironmentalVariablesManager
    other = EnvironmentalVariablesManager()
    other.load_from_config(restored)
    assert dict(other.custom_env_vars) == {
        "PATH_ISH": "/tmp/日本語/α:β",
        "LITERAL_DOLLAR": "$NOT_EXPANDED",
        "EQUALS": "a=b=c",
    }


# ---------------------------------------------------------------------------
# generate_env_string_preview
# ---------------------------------------------------------------------------

def test_preview_disabled_message(env_manager):
    assert env_manager.generate_env_string_preview() == "Environmental variables disabled"


def test_preview_no_vars_configured(env_manager):
    env_manager.env_vars_enabled.set(True)
    assert env_manager.generate_env_string_preview() == "No environmental variables configured"


def test_preview_sorts_and_formats(env_manager):
    env_manager.env_vars_enabled.set(True)
    env_manager.add_custom_env_var("ZETA", "z")
    env_manager.add_custom_env_var("ALPHA", "a")
    env_manager.set_predefined_var_enabled("GGML_CUDA_F16", True, "1")
    result = env_manager.generate_env_string_preview()
    # Sorted alphabetically by name, joined with single spaces, NAME=VALUE form.
    parts = result.split(" ")
    # All parts should be NAME=VALUE and names should be sorted.
    names = [p.split("=")[0] for p in parts]
    assert names == sorted(names)
    assert "ALPHA=a" in parts
    assert "ZETA=z" in parts
    assert "GGML_CUDA_F16=1" in parts


# ---------------------------------------------------------------------------
# generate_clear_env_vars_command
# ---------------------------------------------------------------------------

def test_generate_clear_env_vars_command_includes_all_predefined(env_manager):
    cmd = env_manager.generate_clear_env_vars_command()
    for var_name in env_manager.PREDEFINED_ENV_VARS:
        assert f"export {var_name}=0" in cmd


def test_generate_clear_env_vars_command_includes_custom(env_manager):
    env_manager.add_custom_env_var("MY_CUSTOM", "x")
    cmd = env_manager.generate_clear_env_vars_command()
    assert "export MY_CUSTOM=0" in cmd


def test_generate_clear_env_vars_command_keeps_terminal_open(env_manager):
    # Trailing ';bash' keeps the spawned shell alive after the exports run.
    cmd = env_manager.generate_clear_env_vars_command()
    assert cmd.rstrip().endswith("bash")


def test_generate_clear_env_vars_command_skips_blank_custom_names(env_manager):
    # Force an invalid entry in directly, bypassing add_custom_env_var's filter.
    env_manager.custom_env_vars.append(("   ", "x"))
    cmd = env_manager.generate_clear_env_vars_command()
    assert "export    =0" not in cmd
    assert "export =0" not in cmd


# ---------------------------------------------------------------------------
# Interaction between env_vars_enabled flag and getters
# ---------------------------------------------------------------------------

def test_disable_toggle_hides_all_values(env_manager):
    env_manager.env_vars_enabled.set(True)
    env_manager.set_predefined_var_enabled("GGML_CUDA_F16", True, "1")
    env_manager.add_custom_env_var("FOO", "bar")
    assert env_manager.get_enabled_env_vars()
    env_manager.env_vars_enabled.set(False)
    assert env_manager.get_enabled_env_vars() == {}
    # But underlying state is preserved.
    assert env_manager.enabled_predefined_vars == {"GGML_CUDA_F16": "1"}
    assert env_manager.custom_env_vars == [("FOO", "bar")]
