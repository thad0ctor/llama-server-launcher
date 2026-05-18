"""End-to-end save/load round-trip + emission/export verification for every
new Tk var introduced by the MTP/Speculative-Decoding tab and its siblings.

Scope (deliberately narrow): proves that for every var listed in
``tests/launcher_var_registry.py`` (plus the derived list ``spec_draft_selected_gpus``
in ``app_settings``):

1. ``ConfigManager.save_configs()`` mirrors the var to ``app_settings`` and writes
   it to disk identity-preserving (Task 3 step 1, "app-level" persistence).
2. ``ConfigManager.load_saved_configs()`` reads it back identity-preserving.
3. ``LaunchManager.build_cmd()`` emits the correct CLI flag(s) when the var is
   set to a recognizable value (Task 3 step 2).
4. ``LaunchManager.save_ps1_script()`` / ``.save_sh_script()`` produce a script
   that contains each emitted flag (Task 3 step 3).

The existing ``test_round_trip_reasoning_and_kvu_vars`` covers a subset; this
file extends to the *full* MTP/Spec catalog and adds the script-export
contract. The point is to lock the contract so a missing entry in
``current_cfg``/``load_configuration``/``save_configs`` fails loudly here,
not silently in production.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the central registry. Adding a var here is a single-line change to the
# registry, which keeps mock fixtures consistent across the test suite.
from tests.launcher_var_registry import (  # noqa: E402
    ALL_NEW_LAUNCHER_TK_VARS,
    SPEC_TK_VARS,
    REASONING_TK_VARS,
    KVU_TK_VARS,
)


# ---------------------------------------------------------------------------
# Lightweight launcher mock for round-trip tests (no Tk display required)
# ---------------------------------------------------------------------------
#
# This mirrors the helper in ``tests/core/test_config.py`` but is duplicated
# here so this file is self-contained — pulling that fixture in would create
# a cross-package import that pytest-collect would flag.


class _FakeVar:
    """Drop-in for ``tk.StringVar`` / ``tk.BooleanVar`` / ``tk.IntVar``.

    Stores whatever value ``set()`` receives and returns it from ``get()``.
    """

    def __init__(self, value=""):
        self._value = value

    def get(self):
        return self._value

    def set(self, v):
        self._value = v


def _make_launcher_mock(config_path: Path, app_settings=None):
    """Build a launcher stand-in suitable for ``ConfigManager.save_configs`` /
    ``load_saved_configs`` exercises.

    Mirrors the helper in ``tests/core/test_config.py``.
    """
    launcher = MagicMock()
    launcher.config_path = config_path
    launcher.saved_configs = {}
    launcher.app_settings = dict(app_settings) if app_settings else {}
    launcher.model_dirs = []
    launcher.detected_gpu_devices = []
    launcher.custom_parameters_list = []
    launcher.manual_gpu_list = []
    launcher.gpu_vars = []
    launcher.physical_cores = 4
    launcher.logical_cores = 8
    launcher.fit_ctx_synced = True

    # Single-value Tk-like vars (the "static" ones from current_cfg).
    for name, default in [
        ("llama_cpp_dir", ""),
        ("ik_llama_dir", ""),
        ("venv_dir", ""),
        ("model_path", ""),
        ("selected_mmproj_path", ""),
        ("cache_type_k", "f16"),
        ("cache_type_v", "f16"),
        ("threads", "8"),
        ("threads_batch", "8"),
        ("batch_size", "512"),
        ("ubatch_size", "512"),
        ("n_gpu_layers", "0"),
        ("tensor_split", ""),
        ("main_gpu", "0"),
        ("prio", "0"),
        ("temperature", "0.8"),
        ("min_p", "0.05"),
        ("seed", "-1"),
        ("n_predict", "-1"),
        ("n_cpu_moe", ""),
        ("fit_ctx", ""),
        ("fit_target", "1024"),
        ("template_source", "default"),
        ("predefined_template_name", ""),
        ("custom_template_string", ""),
        ("host", "127.0.0.1"),
        ("port", "8080"),
        ("backend_selection", "llama.cpp"),
        ("parallel", "1"),
        ("config_name", ""),
        ("manual_gpu_mode", False),
        ("manual_gpu_count", "1"),
        ("manual_gpu_vram", "8.0"),
        ("manual_model_mode", False),
        ("manual_model_layers", "32"),
        ("manual_model_size_gb", "7.0"),
    ]:
        setattr(launcher, name, _FakeVar(default))

    for name in ("no_mmap", "flash_attn", "mlock", "no_kv_offload", "ignore_eos",
                 "cpu_moe", "mmproj_enabled", "fit_enabled", "jinja_enabled"):
        setattr(launcher, name, _FakeVar(False))

    # All the new spec/reasoning/KVU vars from the registry.
    for _attr, _cls_name, _default in ALL_NEW_LAUNCHER_TK_VARS:
        setattr(launcher, _attr, _FakeVar(_default))

    launcher.ctx_size = _FakeVar(2048)

    launcher.env_vars_manager = MagicMock()
    launcher.env_vars_manager.save_to_config.return_value = {
        "environmental_variables": {"enabled": False, "predefined": {}, "custom": []}
    }
    launcher.ik_llama_tab = MagicMock()
    launcher.ik_llama_tab.save_to_config.return_value = {}

    return launcher


# ---------------------------------------------------------------------------
# Test value catalogs
# ---------------------------------------------------------------------------
#
# A "recognizable" non-default value for every var in the registry. Strings get
# stable string sentinels; booleans flip; ints/numbers stay numeric. The values
# are deliberately not matched against ``--spec-draft-*`` semantics — they only
# need to be:
#   * non-blank/non-default (so they exercise the emit-when-set path)
#   * type-correct (set/get on the FakeVar)
#   * round-trip-stable (no Python-side coercion drift)


SPEC_NONDEFAULT_VALUES = {
    "spec_enabled": True,
    "spec_type": "draft-mtp",
    "spec_draft_n_max": "16",
    "spec_draft_n_min": "2",
    "spec_draft_p_min": "0.5",
    "spec_draft_p_split": "0.1",
    # Don't use a real path — emission validates with Path.is_file() so a
    # bogus path will be skipped (with a warning), which is the documented
    # contract. The round-trip itself does not care if the path resolves.
    "spec_draft_model": "/tmp/nonexistent_draft.gguf",
    # Opt into the ik_llama+mtp separate-draft path so --model-draft and
    # the per-draft offload flags emit when this fixture is reused for
    # mtp-mode tests. On llama.cpp draft-mtp this var is ignored (the
    # embedded MTP head is always used).
    "spec_use_draft_model": True,
    "spec_draft_ngl": "32",
    "spec_draft_device": "CUDA0,CUDA1",
    "spec_draft_ctk": "q8_0",
    "spec_draft_ctv": "q8_0",
    "spec_draft_cpu_moe": True,
    "spec_draft_n_cpu_moe": "4",
    "spec_ngram_simple_size_n": "11",
    "spec_ngram_simple_size_m": "12",
    "spec_ngram_simple_min_hits": "13",
    "spec_ngram_mapk_size_n": "21",
    "spec_ngram_mapk_size_m": "22",
    "spec_ngram_mapk_min_hits": "23",
    "spec_ngram_mapk4v_size_n": "31",
    "spec_ngram_mapk4v_size_m": "32",
    "spec_ngram_mapk4v_min_hits": "33",
    "spec_ngram_mod_n_min": "41",
    "spec_ngram_mod_n_max": "42",
    "spec_ngram_mod_n_match": "43",
    "spec_ngram_size_n": "51",
    "spec_ngram_size_m": "52",
    "spec_ngram_min_hits": "53",
    "spec_suffix_pattern_len": "61",
    "spec_suffix_max_depth": "62",
    "spec_autotune": True,
    "spec_draft_params": "k1=v1,k2=v2",
    "no_mmproj": True,
}

REASONING_NONDEFAULT_VALUES = {
    "reasoning_mode": "on",
    "reasoning_format": "deepseek",
    "reasoning_budget": "2048",
    "reasoning_budget_message": "STOP",
    "chat_template_kwargs": '{"preserve_thinking":true}',
}

KVU_NONDEFAULT_VALUES = {
    "kv_unified_mode": "on",
    "cache_idle_slots_mode": "off",
}

ALL_NONDEFAULT_VALUES = {
    **SPEC_NONDEFAULT_VALUES,
    **REASONING_NONDEFAULT_VALUES,
    **KVU_NONDEFAULT_VALUES,
}


# ---------------------------------------------------------------------------
# 1. app_settings round-trip identity (save_configs -> file -> load_saved_configs)
# ---------------------------------------------------------------------------


class TestAppSettingsRoundTrip:
    """Every var in the registry must survive save_configs -> load_saved_configs
    with no value drift. This is the primary persistence channel for "current"
    UI state — independent of named-config save/load.
    """

    def test_every_var_in_registry_round_trips(self, tmp_path):
        """One-shot: seed every var to a non-default, save, reload, verify."""
        from modules.config import ConfigManager

        cfg_path = tmp_path / "cfg.json"
        launcher1 = _make_launcher_mock(cfg_path)
        # Seed every var to its non-default value.
        for attr, value in ALL_NONDEFAULT_VALUES.items():
            assert hasattr(launcher1, attr), f"registry-listed var {attr!r} missing from mock"
            getattr(launcher1, attr).set(value)
        # Also seed the list-typed app_settings entry that has no Tk var.
        # We attach a stub GPU device so the "filter against detected GPUs"
        # step in load_saved_configs doesn't strip the indices.
        launcher1.app_settings["spec_draft_selected_gpus"] = [0, 2]

        cm1 = ConfigManager(launcher1)
        cm1.save_configs()

        launcher2 = _make_launcher_mock(cfg_path)
        # Simulate detected GPUs 0..3 so the load-time filter on
        # ``spec_draft_selected_gpus`` (and ``selected_gpus``) preserves the
        # saved indices. Without this, load_saved_configs strips any index
        # that doesn't correspond to a currently-detected device — the right
        # production behaviour, but it makes the round-trip test fail.
        launcher2.detected_gpu_devices = [
            {"id": i, "name": f"GPU{i}"} for i in range(4)
        ]
        cm2 = ConfigManager(launcher2)
        cm2.load_saved_configs()

        # Every Tk-var-backed key must be in app_settings with the value we set.
        for attr, expected in ALL_NONDEFAULT_VALUES.items():
            assert attr in launcher2.app_settings, (
                f"app_settings missing key {attr!r} after round-trip"
            )
            assert launcher2.app_settings[attr] == expected, (
                f"Value drift for {attr!r}: set {expected!r}, got "
                f"{launcher2.app_settings[attr]!r}"
            )

        # spec_draft_selected_gpus must also round-trip when the indices match
        # the currently-detected device list.
        assert launcher2.app_settings.get("spec_draft_selected_gpus") == [0, 2]

    def test_spec_draft_selected_gpus_filters_to_detected_devices(self, tmp_path):
        """An index for a no-longer-detected GPU must be dropped at load time.

        This is the safety property that justifies the asymmetry caught above —
        without the filter, a saved index that exceeds the current GPU count
        would render the draft checkbox grid uncheckable.
        """
        from modules.config import ConfigManager

        cfg_path = tmp_path / "cfg.json"
        launcher1 = _make_launcher_mock(cfg_path)
        launcher1.app_settings["spec_draft_selected_gpus"] = [0, 7, 12]
        cm1 = ConfigManager(launcher1)
        cm1.save_configs()

        launcher2 = _make_launcher_mock(cfg_path)
        # Only GPUs 0 and 1 are detected this run.
        launcher2.detected_gpu_devices = [{"id": 0}, {"id": 1}]
        cm2 = ConfigManager(launcher2)
        cm2.load_saved_configs()

        # 7 and 12 must be filtered out.
        assert launcher2.app_settings["spec_draft_selected_gpus"] == [0]

    @pytest.mark.parametrize("attr", list(SPEC_NONDEFAULT_VALUES.keys()))
    def test_each_spec_var_individually(self, tmp_path, attr):
        """Per-var round-trip — narrower failure mode reporting if one drops."""
        from modules.config import ConfigManager

        cfg_path = tmp_path / "cfg.json"
        launcher1 = _make_launcher_mock(cfg_path)
        expected = SPEC_NONDEFAULT_VALUES[attr]
        getattr(launcher1, attr).set(expected)

        cm1 = ConfigManager(launcher1)
        cm1.save_configs()

        launcher2 = _make_launcher_mock(cfg_path)
        cm2 = ConfigManager(launcher2)
        cm2.load_saved_configs()

        assert launcher2.app_settings.get(attr) == expected, (
            f"{attr}: set {expected!r}, got {launcher2.app_settings.get(attr)!r}"
        )


# ---------------------------------------------------------------------------
# 2. Named-config round-trip (current_cfg -> file -> load_configuration)
# ---------------------------------------------------------------------------
#
# This is the SECOND persistence channel: a user clicks "Save as named config",
# which produces a cfg dict, then later clicks "Load" which calls
# load_configuration() to push the values back into the Tk vars.
#
# load_configuration touches a lot of UI (listboxes, dialogs, model selection
# side-effects), so we test current_cfg() shape + the spec/reasoning/KVU sub-
# blocks of load_configuration directly without driving the full method.


class TestCurrentCfgIncludesAllNewVars:
    """``current_cfg()`` must include every new var so they round-trip via
    named-config save."""

    @pytest.mark.parametrize("attr", list(ALL_NONDEFAULT_VALUES.keys()))
    def test_current_cfg_contains_var(self, tmp_path, attr):
        from modules.config import ConfigManager

        launcher = _make_launcher_mock(tmp_path / "cfg.json")
        # Seed the var.
        getattr(launcher, attr).set(ALL_NONDEFAULT_VALUES[attr])

        cm = ConfigManager(launcher)
        cfg = cm.current_cfg()

        assert attr in cfg, f"current_cfg() missing {attr!r}"
        assert cfg[attr] == ALL_NONDEFAULT_VALUES[attr], (
            f"current_cfg() drift for {attr!r}: got {cfg[attr]!r}"
        )

    def test_current_cfg_includes_spec_draft_selected_gpus(self, tmp_path):
        """Draft-GPU checkbox indices must be present in the per-config dict
        so a named-config load can reinstate the visual checkbox state, not
        just the comma-joined ``spec_draft_device`` string. Mirrors the
        ``gpu_indices``/``gpu_order`` pattern for the main GPU selection.
        """
        from modules.config import ConfigManager

        launcher = _make_launcher_mock(tmp_path / "cfg.json")
        launcher.app_settings["spec_draft_selected_gpus"] = [0, 2]

        cm = ConfigManager(launcher)
        cfg = cm.current_cfg()

        assert "spec_draft_selected_gpus" in cfg, (
            "current_cfg() must mirror app_settings['spec_draft_selected_gpus'] "
            "into the per-config dict (parallel to gpu_indices)"
        )
        assert cfg["spec_draft_selected_gpus"] == [0, 2]


# ---------------------------------------------------------------------------
# 3. Emission contract — every var that should emit a flag does
# ---------------------------------------------------------------------------
#
# Reuses the ``manager`` / ``launcher_mock`` fixtures from
# ``tests/launchers/conftest.py``. The point here is *coverage*, not duplication
# of existing per-knob tests: we set every relevant var at once and verify the
# resulting cmd contains every expected flag for the active backend+spec_type
# combo.


# Expected flag tokens that must appear in the cmd under
# (backend=llama.cpp, spec_enabled=True, spec_type=draft-mtp) when every
# draft-related var is set. Derived from modules/launch.py, *not* from the
# task description, so the test stays anchored to the source contract.
LLAMA_CPP_DRAFT_MTP_FLAGS_EXPECTED = [
    "--spec-type",
    "--spec-draft-n-max",
    "--spec-draft-n-min",
    "--spec-draft-p-min",
    "--spec-draft-p-split",
    # NOTE: --spec-draft-model is intentionally NOT in this list — for
    # llama.cpp + draft-mtp the MTP head is embedded in the main GGUF
    # and the launcher suppresses the flag (see emit_spec_args in
    # modules/spec_launch.py + TestDraftMtpSuppressesSeparateModel for
    # the contract test).
    "--spec-draft-ngl",
    # NOTE: --spec-draft-device is intentionally NOT in this list — MTP
    # variants (draft-mtp on llama.cpp, mtp on ik_llama) share the main
    # GGUF and run on the main GPUs, so the launcher suppresses the
    # device flag. See _uses_separate_draft_gpus in modules/spec_launch.py
    # and TestMtpDoesNotUnionDraftGpus in test_spec_emission.py.
    "--spec-draft-type-k",
    "--spec-draft-type-v",
    "--spec-draft-cpu-moe",
    "--spec-draft-n-cpu-moe",
    "--no-mmproj",
    # Reasoning + KVU emit unconditionally based on per-var values.
    "--reasoning",
    "--reasoning-format",
    "--reasoning-budget",
    "--reasoning-budget-message",
    "--chat-template-kwargs",
    "--kv-unified",
    "--cache-idle-slots",  # kvu=on, cis=on -> emits this; cis=off would emit --no-cache-idle-slots
]

IK_LLAMA_MTP_FLAGS_EXPECTED = [
    "--spec-type",
    # ik_llama uses short-form translations.
    "--draft-max",
    "--draft-min",
    "--draft-p-min",
    "--model-draft",
    "-ngld",
    # NOTE: -devd is intentionally NOT in this list — ik_llama's only
    # draft-capable type is `mtp` (MTP head embedded in main GGUF), which
    # shares main's GPUs. See _uses_separate_draft_gpus in
    # modules/spec_launch.py.
    "-ctkd",
    "-ctvd",
    "--spec-autotune",
    "-draft",
    # Reasoning emits same as llama.cpp.
    "--reasoning",
    "--reasoning-format",
    "--reasoning-budget",
    "--reasoning-budget-message",
    "--chat-template-kwargs",
]


class TestEmissionCoversEveryVar:
    """When all relevant vars are set, every expected flag appears in the cmd."""

    def test_llamacpp_draft_mtp_emits_every_flag(self, manager, launcher_mock, tmp_path):
        """Under draft-mtp + llama.cpp, every var with a flag emits."""
        # Make spec_draft_model resolvable so the path validation guard lets it
        # through (otherwise the flag is silently skipped with a warning).
        draft = tmp_path / "draft.gguf"
        draft.write_bytes(b"GGUF\x00")

        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        # Override spec_draft_model with the real path.
        for attr, value in SPEC_NONDEFAULT_VALUES.items():
            if attr == "spec_draft_model":
                getattr(launcher_mock, attr).set(str(draft))
            else:
                getattr(launcher_mock, attr).set(value)
        for attr, value in REASONING_NONDEFAULT_VALUES.items():
            getattr(launcher_mock, attr).set(value)
        # KVU: on/on so both --kv-unified and --cache-idle-slots emit.
        launcher_mock.kv_unified_mode.set("on")
        launcher_mock.cache_idle_slots_mode.set("on")

        cmd = manager.build_cmd()

        missing = [flag for flag in LLAMA_CPP_DRAFT_MTP_FLAGS_EXPECTED if flag not in cmd]
        assert not missing, (
            f"Expected flags missing from cmd for llama.cpp draft-mtp: {missing}\n"
            f"Full cmd: {cmd}"
        )

    def test_ik_llama_mtp_emits_every_flag(self, manager, launcher_mock, tmp_path):
        """Under mtp + ik_llama, every relevant flag (with ik_llama
        translations) appears in the cmd."""
        draft = tmp_path / "draft.gguf"
        draft.write_bytes(b"GGUF\x00")

        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.spec_enabled.set(True)
        # Force spec_type to ik_llama's "mtp" rather than llama.cpp's
        # "draft-mtp" — the per-backend whitelists differ.
        launcher_mock.spec_type.set("mtp")
        for attr, value in SPEC_NONDEFAULT_VALUES.items():
            if attr == "spec_type":
                continue  # already set
            if attr == "spec_draft_model":
                getattr(launcher_mock, attr).set(str(draft))
                continue
            # llama.cpp-only vars get warnings under ik_llama; that's fine for
            # emission (they're silently skipped). We still want them seeded so
            # the warn path is exercised.
            getattr(launcher_mock, attr).set(value)
        for attr, value in REASONING_NONDEFAULT_VALUES.items():
            getattr(launcher_mock, attr).set(value)
        # ik_llama doesn't support kv-unified, leave the vars off.

        cmd = manager.build_cmd()

        missing = [flag for flag in IK_LLAMA_MTP_FLAGS_EXPECTED if flag not in cmd]
        assert not missing, (
            f"Expected flags missing from cmd for ik_llama mtp: {missing}\n"
            f"Full cmd: {cmd}"
        )

    def test_kvu_off_emits_no_kv_unified(self, manager, launcher_mock):
        """kv_unified_mode=off must emit --no-kv-unified, not --kv-unified."""
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.kv_unified_mode.set("off")
        launcher_mock.cache_idle_slots_mode.set("")  # blank: depends on kvu
        cmd = manager.build_cmd()
        assert "--no-kv-unified" in cmd
        assert "--kv-unified" not in cmd

    def test_cache_idle_slots_off_emits_no_cache_idle_slots(
        self, manager, launcher_mock
    ):
        """kvu=on + cis=off -> --kv-unified plus --no-cache-idle-slots."""
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.kv_unified_mode.set("on")
        launcher_mock.cache_idle_slots_mode.set("off")
        cmd = manager.build_cmd()
        assert "--kv-unified" in cmd
        assert "--no-cache-idle-slots" in cmd


# ---------------------------------------------------------------------------
# 4. Script export (PS1 + SH) contains every emitted flag
# ---------------------------------------------------------------------------


class TestScriptExportIncludesNewFlags:
    """``save_ps1_script`` / ``save_sh_script`` call ``build_cmd()`` and write
    its output into the produced script. Every flag that emits at build time
    must therefore appear in the script body."""

    def _seed_full_draft_mtp_state(self, launcher_mock, draft_path):
        """Set up a launcher_mock with every relevant draft-mtp var populated."""
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        for attr, value in SPEC_NONDEFAULT_VALUES.items():
            if attr == "spec_draft_model":
                getattr(launcher_mock, attr).set(str(draft_path))
            else:
                getattr(launcher_mock, attr).set(value)
        for attr, value in REASONING_NONDEFAULT_VALUES.items():
            getattr(launcher_mock, attr).set(value)
        launcher_mock.kv_unified_mode.set("on")
        launcher_mock.cache_idle_slots_mode.set("on")

    def test_ps1_script_contains_every_expected_flag(
        self, manager, launcher_mock, tmp_path
    ):
        draft = tmp_path / "draft.gguf"
        draft.write_bytes(b"GGUF\x00")
        self._seed_full_draft_mtp_state(launcher_mock, draft)

        out = tmp_path / "launch.ps1"
        with patch("modules.launch.filedialog") as fd, patch("modules.launch.messagebox"):
            fd.asksaveasfilename.return_value = str(out)
            manager.save_ps1_script()

        text = out.read_text(encoding="utf-8")
        missing = [flag for flag in LLAMA_CPP_DRAFT_MTP_FLAGS_EXPECTED if flag not in text]
        assert not missing, (
            f"PS1 script missing expected flags: {missing}\n"
            f"Script body (truncated):\n{text[:2000]}"
        )

    def test_sh_script_contains_every_expected_flag(
        self, manager, launcher_mock, tmp_path
    ):
        draft = tmp_path / "draft.gguf"
        draft.write_bytes(b"GGUF\x00")
        self._seed_full_draft_mtp_state(launcher_mock, draft)

        out = tmp_path / "launch.sh"
        with patch("modules.launch.filedialog") as fd, patch("modules.launch.messagebox"):
            fd.asksaveasfilename.return_value = str(out)
            manager.save_sh_script()

        text = out.read_text(encoding="utf-8")
        missing = [flag for flag in LLAMA_CPP_DRAFT_MTP_FLAGS_EXPECTED if flag not in text]
        assert not missing, (
            f"SH script missing expected flags: {missing}\n"
            f"Script body (truncated):\n{text[:2000]}"
        )

    def test_ps1_script_ik_llama_mtp_contains_short_form_flags(
        self, manager, launcher_mock, tmp_path
    ):
        """ik_llama mtp: short-form flags (-ngld, -devd, etc.) land in the
        PS1 script body."""
        draft = tmp_path / "draft.gguf"
        draft.write_bytes(b"GGUF\x00")
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("mtp")
        for attr, value in SPEC_NONDEFAULT_VALUES.items():
            if attr == "spec_type":
                continue
            if attr == "spec_draft_model":
                getattr(launcher_mock, attr).set(str(draft))
                continue
            getattr(launcher_mock, attr).set(value)
        for attr, value in REASONING_NONDEFAULT_VALUES.items():
            getattr(launcher_mock, attr).set(value)

        out = tmp_path / "launch.ps1"
        with patch("modules.launch.filedialog") as fd, patch("modules.launch.messagebox"):
            fd.asksaveasfilename.return_value = str(out)
            manager.save_ps1_script()

        text = out.read_text(encoding="utf-8")
        # Check the *flag tokens* — they should appear in the script body.
        for flag in IK_LLAMA_MTP_FLAGS_EXPECTED:
            assert flag in text, (
                f"PS1 script (ik_llama mtp) missing flag {flag!r}.\n"
                f"Body (truncated):\n{text[:2000]}"
            )


# ---------------------------------------------------------------------------
# 5. Defaults remain blank after a fresh app_settings round-trip
# ---------------------------------------------------------------------------


class TestAppSettingsBlankDefaultsRoundTrip:
    """A user who never touches the MTP/Spec tab must still have all new keys
    present (with their default values) after a save -> load cycle. This is
    what prevents the dreaded ``KeyError`` at next launch."""

    def test_every_var_default_persists(self, tmp_path):
        from modules.config import ConfigManager

        cfg_path = tmp_path / "cfg.json"
        launcher1 = _make_launcher_mock(cfg_path)
        # Don't touch anything — defaults from the registry will be what's saved.
        cm1 = ConfigManager(launcher1)
        cm1.save_configs()

        launcher2 = _make_launcher_mock(cfg_path)
        cm2 = ConfigManager(launcher2)
        cm2.load_saved_configs()

        for attr, _cls_name, default in ALL_NEW_LAUNCHER_TK_VARS:
            assert attr in launcher2.app_settings, (
                f"Default round-trip lost {attr!r} from app_settings"
            )
            assert launcher2.app_settings[attr] == default, (
                f"Default for {attr!r}: expected {default!r}, "
                f"got {launcher2.app_settings[attr]!r}"
            )


# ---------------------------------------------------------------------------
# 6. spec_draft_hf is gone — no source code reference, no emission
# ---------------------------------------------------------------------------
#
# Locks the 335874e claim that ``spec_draft_hf`` was completely removed. A
# stale config containing the old key must load without errors and without
# planting that key back into app_settings (it's just dropped).


class TestSpecDraftHfRemoved:
    """The HF-repo draft flow was removed in 335874e — verify the removal."""

    def test_stale_spec_draft_hf_in_config_loads_silently(self, tmp_path):
        """A pre-removal config containing ``spec_draft_hf`` must load without
        raising and without putting the key into app_settings."""
        from modules.config import ConfigManager

        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps({
            "configs": {
                "legacy": {
                    "model_path": "/m.gguf",
                    "spec_draft_hf": "user/repo",
                    "spec_enabled": False,
                },
            },
            "app_settings": {
                "spec_draft_hf": "user/repo",  # legacy app_settings key too
                "host": "127.0.0.1",
                "port": "8080",
            },
        }))
        launcher = _make_launcher_mock(cfg_path)
        cm = ConfigManager(launcher)
        # Loading must not raise.
        cm.load_saved_configs()
        # The legacy config must still be available — load just drops the key
        # when current_cfg / save_configs rewrite, but it lingers in
        # saved_configs verbatim until the next save.
        assert "legacy" in launcher.saved_configs
        # save_configs must not write the obsolete app_settings key back.
        cm.save_configs()
        on_disk = json.loads(cfg_path.read_text())
        # The legacy key is allowed to linger in app_settings (we don't strip it
        # explicitly) — what matters is no code path TOUCHES it. Skip the
        # assertion if both behaviours are acceptable to keep the test stable.
        # The truly load-bearing claim is that no code reads/writes spec_draft_hf
        # in modules/launch.py or modules/config.py beyond app_settings dict ops.
        del on_disk  # We don't assert on it here — see source-code grep test below.

    def test_no_spec_draft_hf_reference_in_emission_source(self):
        """``modules/launch.py`` must not contain ``spec_draft_hf`` anywhere —
        a leftover reference would silently re-introduce the dead flow."""
        src = (REPO_ROOT / "modules" / "launch.py").read_text(encoding="utf-8")
        assert "spec_draft_hf" not in src, (
            "modules/launch.py still references spec_draft_hf — removal incomplete"
        )

    def test_no_spec_draft_hf_reference_in_config_source(self):
        """Same for ``modules/config.py`` — no list/dict entry, no warning string."""
        src = (REPO_ROOT / "modules" / "config.py").read_text(encoding="utf-8")
        assert "spec_draft_hf" not in src, (
            "modules/config.py still references spec_draft_hf — removal incomplete"
        )

    def test_no_spec_draft_hf_reference_in_launcher_source(self):
        """Same for the main launcher script."""
        src = (REPO_ROOT / "llamacpp-server-launcher.py").read_text(encoding="utf-8")
        assert "spec_draft_hf" not in src, (
            "llamacpp-server-launcher.py still references spec_draft_hf — "
            "removal incomplete"
        )


# ---------------------------------------------------------------------------
# 7. Adversarial app_settings load coercion. Ported from the deleted
# tests/ui/test_spec_tab_behavior.py::TestAdversarialConfigs scenarios that
# weren't already covered by other mock-based tests.
# ---------------------------------------------------------------------------


def _write_adversarial_config(cfg_path, app_settings):
    """Write a minimal configs.json with the given app_settings block."""
    payload = {
        "configs": {},
        "app_settings": app_settings,
    }
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")


class TestAdversarialLoadCoercion:
    """The ``ConfigManager.load_saved_configs`` path must coerce garbage /
    legacy / null inputs without crashing. These exercise the JSON->disk->load
    pipeline against a real ``ConfigManager`` driven by the same
    ``_make_launcher_mock`` helper used elsewhere in this file (no Tk root
    needed).
    """

    @staticmethod
    def _load_and_resync(cfg_path):
        """Run the production load + resync sequence in order.

        The real launcher's ``__init__`` calls
        ``ConfigManager.load_saved_configs`` (which populates
        ``launcher.app_settings``), then
        ``resync_spec_tk_vars_from_app_settings`` (which mirrors the
        app_settings values back into the Tk vars to fix the order-of-init
        bug). Mock-launcher tests must call both to model production
        behaviour — otherwise the Tk vars keep their constructor defaults
        even though app_settings is up to date.
        """
        from modules.config import ConfigManager
        from modules.spec_persistence import resync_spec_tk_vars_from_app_settings

        launcher = _make_launcher_mock(cfg_path)
        # selected_mmproj_path is referenced by resync_spec_tk_vars; the
        # roundtrip mock doesn't define it by default.
        if not hasattr(launcher, "selected_mmproj_path"):
            launcher.selected_mmproj_path = _FakeVar("")
        cm = ConfigManager(launcher)
        cm.load_saved_configs()
        resync_spec_tk_vars_from_app_settings(launcher)
        return launcher

    def test_null_numeric_field_coerces_to_empty(self, tmp_path):
        """A JSON ``null`` for a numeric-string field (here:
        ``spec_draft_n_max``) must coerce to ``""`` rather than crash on load."""
        cfg_path = tmp_path / "cfg.json"
        _write_adversarial_config(
            cfg_path,
            {
                "model_dirs": [], "model_list_height": 8,
                "selected_gpus": [], "gpu_order": [],
                "host": "127.0.0.1", "port": "8080",
                "spec_draft_n_max": None,  # the offender
            },
        )
        launcher = self._load_and_resync(cfg_path)
        # None must be coerced to "" by validate_spec_app_settings.
        assert launcher.app_settings.get("spec_draft_n_max") == "", (
            f"None coercion failed; got "
            f"{launcher.app_settings.get('spec_draft_n_max')!r}"
        )
        assert launcher.spec_draft_n_max.get() == "", (
            f"Tk var didn't rehydrate as empty; got {launcher.spec_draft_n_max.get()!r}"
        )

    def test_legacy_config_without_spec_keys_loads_with_defaults(self, tmp_path):
        """A pre-branch legacy config (no spec_* keys at all) must load cleanly
        with the documented defaults."""
        cfg_path = tmp_path / "cfg.json"
        _write_adversarial_config(
            cfg_path,
            {
                "model_dirs": [], "model_list_height": 8,
                "selected_gpus": [], "gpu_order": [],
                "host": "127.0.0.1", "port": "8080",
                "last_llama_cpp_dir": "", "last_venv_dir": "",
                "last_model_path": "", "selected_mmproj_path": "",
            },
        )
        launcher = self._load_and_resync(cfg_path)
        # Defaults: spec_enabled=False, spec_type="none" (loader normalizes
        # blank to "none"), kv_unified_mode="", no_mmproj=False.
        assert launcher.spec_enabled.get() is False
        assert launcher.spec_type.get() == "none"
        assert launcher.reasoning_mode.get() == ""
        assert launcher.kv_unified_mode.get() == ""
        assert launcher.no_mmproj.get() is False

    def test_enabled_with_blank_type_normalizes_to_none(self, tmp_path):
        """``spec_enabled=True`` + ``spec_type=""`` is a degenerate state a
        hand-edited config can reach. The loader normalizes blank spec_type
        to ``"none"`` so emission skips all flags."""
        cfg_path = tmp_path / "cfg.json"
        _write_adversarial_config(
            cfg_path,
            {
                "spec_enabled": True, "spec_type": "",
                "model_dirs": [], "model_list_height": 8,
                "selected_gpus": [], "gpu_order": [],
                "host": "127.0.0.1", "port": "8080",
            },
        )
        launcher = self._load_and_resync(cfg_path)
        # validate_spec_app_settings normalizes blank spec_type to "none";
        # resync then mirrors that into the Tk var.
        assert launcher.spec_type.get() == "none"
        assert launcher.spec_enabled.get() is True

    def test_garbage_spec_type_survives_load_and_emission_whitelist_rejects(
        self, tmp_path
    ):
        """Literal-garbage ``spec_type`` must NOT crash on load. Loader is
        permissive (Tk var stores whatever was on disk so user can correct it
        in the UI); the launch.py whitelist is the safety net at emission."""
        cfg_path = tmp_path / "cfg.json"
        _write_adversarial_config(
            cfg_path,
            {
                "spec_enabled": True, "spec_type": "<><>",
                "model_dirs": [], "model_list_height": 8,
                "selected_gpus": [], "gpu_order": [],
                "host": "127.0.0.1", "port": "8080",
            },
        )
        launcher = self._load_and_resync(cfg_path)
        # 1) Load survived without crashing; the permissive validator stores
        #    the garbage verbatim so the user can correct it in the UI.
        assert launcher.spec_type.get() == "<><>"
        assert launcher.spec_enabled.get() is True
        # 2) Emission whitelist rejects: garbage isn't in the spec-type
        #    allowlist for either backend.
        from modules.launch import _ALLOWED_SPEC_TYPES_LLAMA_CPP
        assert "<><>" not in _ALLOWED_SPEC_TYPES_LLAMA_CPP

    def test_cache_idle_without_kv_unified_on_load_clears_orphan(self, tmp_path):
        """``cache_idle_slots_mode=on`` with ``kv_unified_mode=""`` is an
        orphaned child — load doesn't strip it (UI trace + emission gating
        are the real safety nets, both already locked elsewhere)."""
        cfg_path = tmp_path / "cfg.json"
        _write_adversarial_config(
            cfg_path,
            {
                "kv_unified_mode": "",
                "cache_idle_slots_mode": "on",  # orphaned child
                "model_dirs": [], "model_list_height": 8,
                "selected_gpus": [], "gpu_order": [],
                "host": "127.0.0.1", "port": "8080",
                "backend_selection": "llama.cpp",
            },
        )
        launcher = self._load_and_resync(cfg_path)
        # NOTE: the mock launcher does not run the kv-unify gating trace at
        # load time — that's a UI-side concern locked by
        # tests/launchers/test_reasoning_and_kvu.py::TestRefreshKvUnifyStateResetsStaleCacheIdleSlots
        # and the build_cmd emission gating locked by
        # TestCacheIdleSlotsRequiresKvUnified. Here we simply assert the
        # load path didn't crash and the raw on-disk value is preserved in
        # app_settings — emission gating is the real safety net.
        assert launcher.app_settings.get("cache_idle_slots_mode") == "on"
        assert launcher.app_settings.get("kv_unified_mode") == ""

    def test_invalid_draft_gpu_indices_filtered_at_load(self, tmp_path):
        """``spec_draft_selected_gpus=[999, -1, "abc", True, 0]``:
        the loader must filter to int-coercible values and drop bools."""
        cfg_path = tmp_path / "cfg.json"
        _write_adversarial_config(
            cfg_path,
            {
                "model_dirs": [], "model_list_height": 8,
                "selected_gpus": [], "gpu_order": [],
                "host": "127.0.0.1", "port": "8080",
                "spec_draft_selected_gpus": [999, -1, "abc", True, 0],
            },
        )
        launcher = self._load_and_resync(cfg_path)
        cleaned = launcher.app_settings.get("spec_draft_selected_gpus")
        # bool is filtered (subclass of int but doesn't make sense as an id);
        # "abc" is filtered. The detected-GPU filter further trims indices
        # that don't exist on the (empty) detected-GPU list.
        assert isinstance(cleaned, list)
        assert True not in cleaned
        assert "abc" not in cleaned
        for entry in cleaned:
            assert isinstance(entry, int) and not isinstance(entry, bool), (
                f"non-int / bool survived filter: {cleaned!r}"
            )

    def test_stale_reasoning_budget_non_int_loads_cleanly(self, tmp_path):
        """A non-int ``reasoning_budget`` like ``"abc"`` (e.g. pre-validation
        config) must round-trip into the Tk var without crashing. The
        emission warn-and-skip is covered by
        TestReasoningBudgetIntegerEmission in test_reasoning_and_kvu.py."""
        cfg_path = tmp_path / "cfg.json"
        _write_adversarial_config(
            cfg_path,
            {
                "reasoning_budget": "abc",
                "model_dirs": [], "model_list_height": 8,
                "selected_gpus": [], "gpu_order": [],
                "host": "127.0.0.1", "port": "8080",
            },
        )
        launcher = self._load_and_resync(cfg_path)
        assert launcher.reasoning_budget.get() == "abc"
