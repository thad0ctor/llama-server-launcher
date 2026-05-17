"""Direct emission tests for the Speculative Decoding (MTP/spec) block and
the sibling ``--no-mmproj`` block in ``modules/launch.py`` (``LaunchManager.
build_cmd()``).

The reasoning + KVU blocks are already covered in
``test_reasoning_and_kvu.py``; the spec block — including the headline
``--spec-type draft-mtp`` flag — was previously untested. This file plugs
that gap by exercising every emission path the contract documents:

* Master toggle gates (spec_enabled / spec_type).
* llama.cpp happy paths for all 8 supported spec_types.
* llama.cpp per-variant ngram knob gating.
* llama.cpp cross-backend warn-and-skip for ik_llama-only knobs.
* ik_llama happy paths for all 7 supported spec_types, with special
  attention to the flag-name translations (``--draft-max`` vs
  ``--spec-draft-n-max``, ``--model-draft`` vs ``--spec-draft-model``,
  short-form ``-ngld``/``-devd``/``-ctkd``/``-ctvd``).
* ik_llama shared ngram set (``--spec-ngram-size-n`` etc.) plus the
  negative case where llama.cpp's per-variant knobs are ignored.
* ik_llama suffix tuning, autotune, and ``-draft <params>`` extras.
* ik_llama cross-backend warn-and-skip for llama.cpp-only knobs.
* The independent ``--no-mmproj`` block (llama.cpp emits / ik_llama warns).

Mirrors the fixture style of ``test_reasoning_and_kvu.py``: pulls
``launcher_mock`` / ``manager`` from ``tests/launchers/conftest.py`` and
uses ``capsys`` for stderr-warning assertions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# All 8 llama.cpp spec_types supported by the emission block.
LLAMACPP_SPEC_TYPES = [
    "draft-simple",
    "draft-eagle3",
    "draft-mtp",
    "ngram-simple",
    "ngram-map-k",
    "ngram-map-k4v",
    "ngram-mod",
    "ngram-cache",
]

# All 7 ik_llama spec_types supported by the emission block.
IK_LLAMA_SPEC_TYPES = [
    "mtp",
    "ngram-cache",
    "ngram-simple",
    "ngram-map-k",
    "ngram-map-k4v",
    "ngram-mod",
    "suffix",
]


# ============================================================================
# Master toggle
# ============================================================================


class TestSpecMasterToggle:
    """``spec_enabled`` is the single gate. When False, nothing the spec
    block manages should appear in the cmd, regardless of how loud the
    other vars are. Same for ``spec_type`` values of "" or "none"."""

    def test_spec_disabled_emits_nothing(self, manager, launcher_mock):
        launcher_mock.spec_enabled.set(False)
        launcher_mock.spec_type.set("draft-mtp")
        # Loud values that would absolutely emit if the gate let them.
        launcher_mock.spec_draft_n_max.set("3")
        launcher_mock.spec_draft_model.set("/tmp/draft.gguf")
        cmd = manager.build_cmd()
        assert "--spec-type" not in cmd
        assert "--spec-draft-n-max" not in cmd
        assert "--spec-draft-model" not in cmd

    def test_spec_enabled_with_empty_spec_type_emits_nothing(
        self, manager, launcher_mock
    ):
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("")
        launcher_mock.spec_draft_n_max.set("3")
        cmd = manager.build_cmd()
        assert "--spec-type" not in cmd
        assert "--spec-draft-n-max" not in cmd

    def test_spec_enabled_with_none_spec_type_emits_nothing(
        self, manager, launcher_mock
    ):
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("none")
        launcher_mock.spec_draft_n_max.set("3")
        cmd = manager.build_cmd()
        assert "--spec-type" not in cmd
        assert "--spec-draft-n-max" not in cmd

    def test_default_state_emits_no_spec_flags(self, manager, launcher_mock):
        """Zero-noise default: untouched fixture must not emit any spec or
        draft flag."""
        cmd = manager.build_cmd()
        for flag in (
            "--spec-type",
            "--spec-draft-n-max",
            "--spec-draft-n-min",
            "--spec-draft-p-min",
            "--spec-draft-p-split",
            "--spec-draft-model",
            "--spec-draft-hf",
            "--spec-draft-ngl",
            "--spec-draft-device",
            "--spec-draft-type-k",
            "--spec-draft-type-v",
            "--spec-draft-cpu-moe",
            "--spec-draft-n-cpu-moe",
            "--draft-max",
            "--draft-min",
            "--draft-p-min",
            "--model-draft",
            "-ngld",
            "-devd",
            "-ctkd",
            "-ctvd",
            "--spec-autotune",
            "-draft",
            "--suffix-pattern-len",
            "--suffix-max-depth",
        ):
            assert flag not in cmd, f"Unexpected spec flag in default cmd: {flag}"


# ============================================================================
# llama.cpp emission
# ============================================================================


class TestSpecEmissionLlamaCpp:
    """Happy-path emission under the mainline llama.cpp backend."""

    @pytest.mark.parametrize("spec_type", LLAMACPP_SPEC_TYPES)
    def test_spec_type_flag_emits_for_each_valid_value(
        self, manager, launcher_mock, spec_type
    ):
        """Every supported llama.cpp spec_type lands ``--spec-type <value>``
        verbatim."""
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set(spec_type)
        cmd = manager.build_cmd()
        assert "--spec-type" in cmd
        assert cmd[cmd.index("--spec-type") + 1] == spec_type

    def test_draft_mtp_basic_emission(self, manager, launcher_mock):
        """The flagship case the entire branch exists to support."""
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("draft-mtp")
        cmd = manager.build_cmd()
        assert "--spec-type" in cmd
        assert cmd[cmd.index("--spec-type") + 1] == "draft-mtp"

    def test_spec_draft_n_max_emits(self, manager, launcher_mock):
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("draft-mtp")
        launcher_mock.spec_draft_n_max.set("16")
        cmd = manager.build_cmd()
        assert "--spec-draft-n-max" in cmd
        assert cmd[cmd.index("--spec-draft-n-max") + 1] == "16"

    def test_spec_draft_n_min_emits(self, manager, launcher_mock):
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("draft-mtp")
        launcher_mock.spec_draft_n_min.set("2")
        cmd = manager.build_cmd()
        assert "--spec-draft-n-min" in cmd
        assert cmd[cmd.index("--spec-draft-n-min") + 1] == "2"

    def test_spec_draft_p_min_emits(self, manager, launcher_mock):
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("draft-mtp")
        launcher_mock.spec_draft_p_min.set("0.5")
        cmd = manager.build_cmd()
        assert "--spec-draft-p-min" in cmd
        assert cmd[cmd.index("--spec-draft-p-min") + 1] == "0.5"

    def test_spec_draft_p_split_emits(self, manager, launcher_mock):
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("draft-mtp")
        launcher_mock.spec_draft_p_split.set("0.1")
        cmd = manager.build_cmd()
        assert "--spec-draft-p-split" in cmd
        assert cmd[cmd.index("--spec-draft-p-split") + 1] == "0.1"

    def test_spec_draft_model_emits(self, manager, launcher_mock):
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("draft-mtp")
        launcher_mock.spec_draft_model.set("/models/draft.gguf")
        cmd = manager.build_cmd()
        assert "--spec-draft-model" in cmd
        assert cmd[cmd.index("--spec-draft-model") + 1] == "/models/draft.gguf"

    def test_spec_draft_hf_emits(self, manager, launcher_mock):
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("draft-mtp")
        launcher_mock.spec_draft_hf.set("org/repo:Q4_K_M")
        cmd = manager.build_cmd()
        assert "--spec-draft-hf" in cmd
        assert cmd[cmd.index("--spec-draft-hf") + 1] == "org/repo:Q4_K_M"

    def test_spec_draft_offload_flags_emit(self, manager, launcher_mock):
        """ngl/device/ctk/ctv all use the long llama.cpp flag names."""
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("draft-mtp")
        launcher_mock.spec_draft_ngl.set("32")
        launcher_mock.spec_draft_device.set("CUDA0")
        launcher_mock.spec_draft_ctk.set("q8_0")
        launcher_mock.spec_draft_ctv.set("q8_0")
        cmd = manager.build_cmd()
        assert cmd[cmd.index("--spec-draft-ngl") + 1] == "32"
        assert cmd[cmd.index("--spec-draft-device") + 1] == "CUDA0"
        assert cmd[cmd.index("--spec-draft-type-k") + 1] == "q8_0"
        assert cmd[cmd.index("--spec-draft-type-v") + 1] == "q8_0"
        # Sanity: the short-form ik_llama flags must NOT appear.
        for short in ("-ngld", "-devd", "-ctkd", "-ctvd"):
            assert short not in cmd

    def test_spec_draft_cpu_moe_bare_flag_emits(self, manager, launcher_mock):
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("draft-mtp")
        launcher_mock.spec_draft_cpu_moe.set(True)
        cmd = manager.build_cmd()
        assert "--spec-draft-cpu-moe" in cmd
        # Bare flag — next token (if any) must not be a value like "True".
        idx = cmd.index("--spec-draft-cpu-moe")
        if idx + 1 < len(cmd):
            assert cmd[idx + 1] != "True"
            assert cmd[idx + 1] != "true"

    def test_spec_draft_n_cpu_moe_emits(self, manager, launcher_mock):
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("draft-mtp")
        launcher_mock.spec_draft_n_cpu_moe.set("4")
        cmd = manager.build_cmd()
        assert "--spec-draft-n-cpu-moe" in cmd
        assert cmd[cmd.index("--spec-draft-n-cpu-moe") + 1] == "4"

    def test_blank_draft_tuning_vars_omit_flags(self, manager, launcher_mock):
        """Per-flag predicate: each draft tuning var must individually
        decide to emit. Blank ones are dropped."""
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("draft-mtp")
        # Set only n_max; leave the others blank.
        launcher_mock.spec_draft_n_max.set("5")
        cmd = manager.build_cmd()
        assert "--spec-draft-n-max" in cmd
        for absent in (
            "--spec-draft-n-min",
            "--spec-draft-p-min",
            "--spec-draft-p-split",
            "--spec-draft-model",
            "--spec-draft-hf",
            "--spec-draft-ngl",
            "--spec-draft-device",
            "--spec-draft-type-k",
            "--spec-draft-type-v",
            "--spec-draft-cpu-moe",
            "--spec-draft-n-cpu-moe",
        ):
            assert absent not in cmd

    def test_all_llamacpp_draft_knobs_combined(self, manager, launcher_mock):
        """Setting many knobs at once: each one independently emits and
        none drops out due to interaction. Mirrors the
        ``test_all_reasoning_flags_together`` style."""
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("draft-mtp")
        launcher_mock.spec_draft_n_max.set("16")
        launcher_mock.spec_draft_n_min.set("2")
        launcher_mock.spec_draft_p_min.set("0.5")
        launcher_mock.spec_draft_p_split.set("0.1")
        launcher_mock.spec_draft_model.set("/models/draft.gguf")
        launcher_mock.spec_draft_hf.set("org/repo:Q4_K_M")
        launcher_mock.spec_draft_ngl.set("32")
        launcher_mock.spec_draft_device.set("CUDA0")
        launcher_mock.spec_draft_ctk.set("q8_0")
        launcher_mock.spec_draft_ctv.set("q8_0")
        launcher_mock.spec_draft_cpu_moe.set(True)
        launcher_mock.spec_draft_n_cpu_moe.set("4")
        cmd = manager.build_cmd()
        # spec_type
        assert cmd[cmd.index("--spec-type") + 1] == "draft-mtp"
        # Long-form draft tuning flags
        assert cmd[cmd.index("--spec-draft-n-max") + 1] == "16"
        assert cmd[cmd.index("--spec-draft-n-min") + 1] == "2"
        assert cmd[cmd.index("--spec-draft-p-min") + 1] == "0.5"
        assert cmd[cmd.index("--spec-draft-p-split") + 1] == "0.1"
        # Model/HF
        assert cmd[cmd.index("--spec-draft-model") + 1] == "/models/draft.gguf"
        assert cmd[cmd.index("--spec-draft-hf") + 1] == "org/repo:Q4_K_M"
        # Offload long forms
        assert cmd[cmd.index("--spec-draft-ngl") + 1] == "32"
        assert cmd[cmd.index("--spec-draft-device") + 1] == "CUDA0"
        assert cmd[cmd.index("--spec-draft-type-k") + 1] == "q8_0"
        assert cmd[cmd.index("--spec-draft-type-v") + 1] == "q8_0"
        # Bare flag + n_cpu_moe value
        assert "--spec-draft-cpu-moe" in cmd
        assert cmd[cmd.index("--spec-draft-n-cpu-moe") + 1] == "4"


# ============================================================================
# llama.cpp ngram per-variant knob gating
# ============================================================================


class TestSpecNgramKnobsLlamaCpp:
    """Each llama.cpp ngram variant has its own knob trio; the emission
    block must only emit the trio matching the active spec_type."""

    def _set_all_llamacpp_ngram_knobs(self, launcher_mock):
        """Helper: populate every per-variant knob with a recognizable
        value so we can prove the gating routes the right ones."""
        launcher_mock.spec_ngram_simple_size_n.set("11")
        launcher_mock.spec_ngram_simple_size_m.set("12")
        launcher_mock.spec_ngram_simple_min_hits.set("13")
        launcher_mock.spec_ngram_mapk_size_n.set("21")
        launcher_mock.spec_ngram_mapk_size_m.set("22")
        launcher_mock.spec_ngram_mapk_min_hits.set("23")
        launcher_mock.spec_ngram_mapk4v_size_n.set("31")
        launcher_mock.spec_ngram_mapk4v_size_m.set("32")
        launcher_mock.spec_ngram_mapk4v_min_hits.set("33")
        launcher_mock.spec_ngram_mod_n_min.set("41")
        launcher_mock.spec_ngram_mod_n_max.set("42")
        launcher_mock.spec_ngram_mod_n_match.set("43")

    def test_ngram_simple_emits_only_simple_knobs(self, manager, launcher_mock):
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("ngram-simple")
        self._set_all_llamacpp_ngram_knobs(launcher_mock)
        cmd = manager.build_cmd()
        # Simple trio emits.
        assert cmd[cmd.index("--spec-ngram-simple-size-n") + 1] == "11"
        assert cmd[cmd.index("--spec-ngram-simple-size-m") + 1] == "12"
        assert cmd[cmd.index("--spec-ngram-simple-min-hits") + 1] == "13"
        # Other variants must NOT emit.
        for absent in (
            "--spec-ngram-map-k-size-n",
            "--spec-ngram-map-k-size-m",
            "--spec-ngram-map-k-min-hits",
            "--spec-ngram-map-k4v-size-n",
            "--spec-ngram-map-k4v-size-m",
            "--spec-ngram-map-k4v-min-hits",
            "--spec-ngram-mod-n-min",
            "--spec-ngram-mod-n-max",
            "--spec-ngram-mod-n-match",
        ):
            assert absent not in cmd

    def test_ngram_map_k_emits_only_mapk_knobs(self, manager, launcher_mock):
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("ngram-map-k")
        self._set_all_llamacpp_ngram_knobs(launcher_mock)
        cmd = manager.build_cmd()
        assert cmd[cmd.index("--spec-ngram-map-k-size-n") + 1] == "21"
        assert cmd[cmd.index("--spec-ngram-map-k-size-m") + 1] == "22"
        assert cmd[cmd.index("--spec-ngram-map-k-min-hits") + 1] == "23"
        for absent in (
            "--spec-ngram-simple-size-n",
            "--spec-ngram-simple-size-m",
            "--spec-ngram-simple-min-hits",
            "--spec-ngram-map-k4v-size-n",
            "--spec-ngram-map-k4v-size-m",
            "--spec-ngram-map-k4v-min-hits",
            "--spec-ngram-mod-n-min",
            "--spec-ngram-mod-n-max",
            "--spec-ngram-mod-n-match",
        ):
            assert absent not in cmd

    def test_ngram_map_k4v_emits_only_mapk4v_knobs(self, manager, launcher_mock):
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("ngram-map-k4v")
        self._set_all_llamacpp_ngram_knobs(launcher_mock)
        cmd = manager.build_cmd()
        assert cmd[cmd.index("--spec-ngram-map-k4v-size-n") + 1] == "31"
        assert cmd[cmd.index("--spec-ngram-map-k4v-size-m") + 1] == "32"
        assert cmd[cmd.index("--spec-ngram-map-k4v-min-hits") + 1] == "33"
        for absent in (
            "--spec-ngram-simple-size-n",
            "--spec-ngram-simple-size-m",
            "--spec-ngram-simple-min-hits",
            "--spec-ngram-map-k-size-n",
            "--spec-ngram-map-k-size-m",
            "--spec-ngram-map-k-min-hits",
            "--spec-ngram-mod-n-min",
            "--spec-ngram-mod-n-max",
            "--spec-ngram-mod-n-match",
        ):
            assert absent not in cmd

    def test_ngram_mod_emits_only_mod_knobs(self, manager, launcher_mock):
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("ngram-mod")
        self._set_all_llamacpp_ngram_knobs(launcher_mock)
        cmd = manager.build_cmd()
        assert cmd[cmd.index("--spec-ngram-mod-n-min") + 1] == "41"
        assert cmd[cmd.index("--spec-ngram-mod-n-max") + 1] == "42"
        assert cmd[cmd.index("--spec-ngram-mod-n-match") + 1] == "43"
        for absent in (
            "--spec-ngram-simple-size-n",
            "--spec-ngram-simple-size-m",
            "--spec-ngram-simple-min-hits",
            "--spec-ngram-map-k-size-n",
            "--spec-ngram-map-k-size-m",
            "--spec-ngram-map-k-min-hits",
            "--spec-ngram-map-k4v-size-n",
            "--spec-ngram-map-k4v-size-m",
            "--spec-ngram-map-k4v-min-hits",
        ):
            assert absent not in cmd

    def test_ngram_cache_emits_no_ngram_knobs(self, manager, launcher_mock):
        """ngram-cache has no per-variant knobs; even if every per-variant
        var is populated, none of the variant flags should emit."""
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("ngram-cache")
        self._set_all_llamacpp_ngram_knobs(launcher_mock)
        cmd = manager.build_cmd()
        # spec_type still emits.
        assert "--spec-type" in cmd
        assert cmd[cmd.index("--spec-type") + 1] == "ngram-cache"
        # But no variant ngram knob does.
        for absent in (
            "--spec-ngram-simple-size-n",
            "--spec-ngram-simple-size-m",
            "--spec-ngram-simple-min-hits",
            "--spec-ngram-map-k-size-n",
            "--spec-ngram-map-k-size-m",
            "--spec-ngram-map-k-min-hits",
            "--spec-ngram-map-k4v-size-n",
            "--spec-ngram-map-k4v-size-m",
            "--spec-ngram-map-k4v-min-hits",
            "--spec-ngram-mod-n-min",
            "--spec-ngram-mod-n-max",
            "--spec-ngram-mod-n-match",
        ):
            assert absent not in cmd


# ============================================================================
# llama.cpp cross-backend warnings (ik_llama-only vars set under llama.cpp)
# ============================================================================


class TestSpecCrossBackendWarningsLlamaCpp:
    """When the backend is llama.cpp and an ik_llama-only spec var is set,
    the block must warn to stderr AND NOT push the flag onto cmd."""

    def test_spec_autotune_warns_and_skips(self, manager, launcher_mock, capsys):
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("draft-mtp")
        launcher_mock.spec_autotune.set(True)
        cmd = manager.build_cmd()
        assert "--spec-autotune" not in cmd
        captured = capsys.readouterr()
        assert "autotune" in captured.err.lower()
        assert "ik_llama" in captured.err.lower()

    def test_spec_draft_params_warns_and_skips(self, manager, launcher_mock, capsys):
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("draft-mtp")
        launcher_mock.spec_draft_params.set("alpha=0.5")
        cmd = manager.build_cmd()
        # The -draft flag is ik_llama-only.
        assert "-draft" not in cmd
        # The value itself must not survive either.
        assert "alpha=0.5" not in cmd
        captured = capsys.readouterr()
        assert "draft" in captured.err.lower()
        assert "ik_llama" in captured.err.lower()

    def test_spec_suffix_pattern_len_warns_and_skips(
        self, manager, launcher_mock, capsys
    ):
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("draft-mtp")
        launcher_mock.spec_suffix_pattern_len.set("4")
        cmd = manager.build_cmd()
        assert "--suffix-pattern-len" not in cmd
        captured = capsys.readouterr()
        assert "suffix" in captured.err.lower()
        assert "ik_llama" in captured.err.lower()

    def test_spec_suffix_max_depth_warns_and_skips(
        self, manager, launcher_mock, capsys
    ):
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("draft-mtp")
        launcher_mock.spec_suffix_max_depth.set("8")
        cmd = manager.build_cmd()
        assert "--suffix-max-depth" not in cmd
        captured = capsys.readouterr()
        assert "suffix" in captured.err.lower()
        assert "ik_llama" in captured.err.lower()


# ============================================================================
# ik_llama emission
# ============================================================================


class TestSpecEmissionIkLlama:
    """Happy-path emission under ik_llama, with emphasis on the flag-name
    translations: ``--draft-max`` not ``--spec-draft-n-max``,
    ``--model-draft`` not ``--spec-draft-model``, short-form offload flags."""

    @pytest.mark.parametrize("spec_type", IK_LLAMA_SPEC_TYPES)
    def test_spec_type_flag_emits_for_each_valid_value(
        self, manager, launcher_mock, spec_type
    ):
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set(spec_type)
        cmd = manager.build_cmd()
        assert "--spec-type" in cmd
        assert cmd[cmd.index("--spec-type") + 1] == spec_type

    def test_draft_tuning_uses_short_form_names(self, manager, launcher_mock):
        """The headline ik_llama translation: ``--draft-max`` /
        ``--draft-min`` / ``--draft-p-min`` instead of
        ``--spec-draft-n-max`` / etc."""
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("mtp")
        launcher_mock.spec_draft_n_max.set("16")
        launcher_mock.spec_draft_n_min.set("2")
        launcher_mock.spec_draft_p_min.set("0.5")
        cmd = manager.build_cmd()
        # ik_llama short forms ARE present...
        assert "--draft-max" in cmd
        assert cmd[cmd.index("--draft-max") + 1] == "16"
        assert "--draft-min" in cmd
        assert cmd[cmd.index("--draft-min") + 1] == "2"
        assert "--draft-p-min" in cmd
        assert cmd[cmd.index("--draft-p-min") + 1] == "0.5"
        # ...and the llama.cpp long forms are NOT.
        assert "--spec-draft-n-max" not in cmd
        assert "--spec-draft-n-min" not in cmd
        assert "--spec-draft-p-min" not in cmd

    def test_draft_model_uses_model_draft_flag(self, manager, launcher_mock):
        """``spec_draft_model`` -> ``--model-draft`` under ik_llama, NOT
        ``--spec-draft-model``."""
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("mtp")
        launcher_mock.spec_draft_model.set("/models/draft.gguf")
        cmd = manager.build_cmd()
        assert "--model-draft" in cmd
        assert cmd[cmd.index("--model-draft") + 1] == "/models/draft.gguf"
        assert "--spec-draft-model" not in cmd

    def test_draft_offload_uses_short_form_flags(self, manager, launcher_mock):
        """``-ngld``, ``-devd``, ``-ctkd``, ``-ctvd`` instead of the long
        ``--spec-draft-*`` names."""
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("mtp")
        launcher_mock.spec_draft_ngl.set("24")
        launcher_mock.spec_draft_device.set("CUDA1")
        launcher_mock.spec_draft_ctk.set("q4_0")
        launcher_mock.spec_draft_ctv.set("q4_0")
        cmd = manager.build_cmd()
        # Short forms present.
        assert cmd[cmd.index("-ngld") + 1] == "24"
        assert cmd[cmd.index("-devd") + 1] == "CUDA1"
        assert cmd[cmd.index("-ctkd") + 1] == "q4_0"
        assert cmd[cmd.index("-ctvd") + 1] == "q4_0"
        # Long forms absent.
        for absent in (
            "--spec-draft-ngl",
            "--spec-draft-device",
            "--spec-draft-type-k",
            "--spec-draft-type-v",
        ):
            assert absent not in cmd

    def test_ik_llama_blank_draft_vars_omit_flags(self, manager, launcher_mock):
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("mtp")
        # Only n_max set.
        launcher_mock.spec_draft_n_max.set("7")
        cmd = manager.build_cmd()
        assert "--draft-max" in cmd
        for absent in (
            "--draft-min",
            "--draft-p-min",
            "--model-draft",
            "-ngld",
            "-devd",
            "-ctkd",
            "-ctvd",
        ):
            assert absent not in cmd

    def test_ik_llama_all_draft_knobs_combined(self, manager, launcher_mock):
        """Setting many ik_llama knobs at once: each translation lands
        without interaction."""
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("mtp")
        launcher_mock.spec_draft_n_max.set("16")
        launcher_mock.spec_draft_n_min.set("2")
        launcher_mock.spec_draft_p_min.set("0.5")
        launcher_mock.spec_draft_model.set("/models/draft.gguf")
        launcher_mock.spec_draft_ngl.set("24")
        launcher_mock.spec_draft_device.set("CUDA1")
        launcher_mock.spec_draft_ctk.set("q4_0")
        launcher_mock.spec_draft_ctv.set("q4_0")
        cmd = manager.build_cmd()
        assert cmd[cmd.index("--spec-type") + 1] == "mtp"
        assert cmd[cmd.index("--draft-max") + 1] == "16"
        assert cmd[cmd.index("--draft-min") + 1] == "2"
        assert cmd[cmd.index("--draft-p-min") + 1] == "0.5"
        assert cmd[cmd.index("--model-draft") + 1] == "/models/draft.gguf"
        assert cmd[cmd.index("-ngld") + 1] == "24"
        assert cmd[cmd.index("-devd") + 1] == "CUDA1"
        assert cmd[cmd.index("-ctkd") + 1] == "q4_0"
        assert cmd[cmd.index("-ctvd") + 1] == "q4_0"
        # Long forms still absent.
        for absent in (
            "--spec-draft-n-max",
            "--spec-draft-n-min",
            "--spec-draft-p-min",
            "--spec-draft-model",
            "--spec-draft-ngl",
            "--spec-draft-device",
            "--spec-draft-type-k",
            "--spec-draft-type-v",
        ):
            assert absent not in cmd


# ============================================================================
# ik_llama ngram knobs (single shared set)
# ============================================================================


class TestSpecNgramKnobsIkLlama:
    """ik_llama uses one shared trio of ngram knobs regardless of which
    ``ngram-*`` variant is active. The per-variant llama.cpp knobs are
    silently ignored under ik_llama."""

    @pytest.mark.parametrize(
        "spec_type",
        ["ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod", "ngram-cache"],
    )
    def test_shared_ngram_set_emits_for_every_ngram_variant(
        self, manager, launcher_mock, spec_type
    ):
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set(spec_type)
        launcher_mock.spec_ngram_size_n.set("3")
        launcher_mock.spec_ngram_size_m.set("5")
        launcher_mock.spec_ngram_min_hits.set("2")
        cmd = manager.build_cmd()
        assert cmd[cmd.index("--spec-ngram-size-n") + 1] == "3"
        assert cmd[cmd.index("--spec-ngram-size-m") + 1] == "5"
        assert cmd[cmd.index("--spec-ngram-min-hits") + 1] == "2"

    def test_ngram_shared_set_does_not_emit_for_non_ngram_spec_type(
        self, manager, launcher_mock
    ):
        """Outside an ``ngram-*`` spec_type, the shared set must stay
        silent even when populated."""
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("mtp")
        launcher_mock.spec_ngram_size_n.set("3")
        launcher_mock.spec_ngram_size_m.set("5")
        launcher_mock.spec_ngram_min_hits.set("2")
        cmd = manager.build_cmd()
        for absent in (
            "--spec-ngram-size-n",
            "--spec-ngram-size-m",
            "--spec-ngram-min-hits",
        ):
            assert absent not in cmd

    def test_llamacpp_per_variant_ngram_knobs_ignored_under_ik_llama(
        self, manager, launcher_mock
    ):
        """Populate every llama.cpp-specific per-variant knob; under
        ik_llama none of them should land on the command."""
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("ngram-simple")
        launcher_mock.spec_ngram_simple_size_n.set("11")
        launcher_mock.spec_ngram_simple_size_m.set("12")
        launcher_mock.spec_ngram_simple_min_hits.set("13")
        launcher_mock.spec_ngram_mapk_size_n.set("21")
        launcher_mock.spec_ngram_mapk4v_size_n.set("31")
        launcher_mock.spec_ngram_mod_n_min.set("41")
        cmd = manager.build_cmd()
        for absent in (
            "--spec-ngram-simple-size-n",
            "--spec-ngram-simple-size-m",
            "--spec-ngram-simple-min-hits",
            "--spec-ngram-map-k-size-n",
            "--spec-ngram-map-k4v-size-n",
            "--spec-ngram-mod-n-min",
        ):
            assert absent not in cmd


# ============================================================================
# ik_llama suffix tuning
# ============================================================================


class TestSpecSuffixIkLlama:
    """``--suffix-pattern-len`` / ``--suffix-max-depth`` emit ONLY when
    ``spec_type == "suffix"`` under ik_llama."""

    def test_suffix_flags_emit_when_spec_type_is_suffix(
        self, manager, launcher_mock
    ):
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("suffix")
        launcher_mock.spec_suffix_pattern_len.set("4")
        launcher_mock.spec_suffix_max_depth.set("12")
        cmd = manager.build_cmd()
        assert cmd[cmd.index("--suffix-pattern-len") + 1] == "4"
        assert cmd[cmd.index("--suffix-max-depth") + 1] == "12"

    def test_suffix_flags_omitted_for_non_suffix_spec_type(
        self, manager, launcher_mock
    ):
        """Populate suffix vars but pick spec_type=mtp; nothing emits."""
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("mtp")
        launcher_mock.spec_suffix_pattern_len.set("4")
        launcher_mock.spec_suffix_max_depth.set("12")
        cmd = manager.build_cmd()
        assert "--suffix-pattern-len" not in cmd
        assert "--suffix-max-depth" not in cmd


# ============================================================================
# ik_llama extras: --spec-autotune, -draft <params>
# ============================================================================


class TestSpecIkLlamaExtras:
    """``--spec-autotune`` (bare flag) and ``-draft <value>`` are
    ik_llama-only extras that the master toggle still gates."""

    def test_spec_autotune_emits_bare_flag(self, manager, launcher_mock):
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("mtp")
        launcher_mock.spec_autotune.set(True)
        cmd = manager.build_cmd()
        assert "--spec-autotune" in cmd
        idx = cmd.index("--spec-autotune")
        # Bare flag — next token (if any) must not be "True" / "true".
        if idx + 1 < len(cmd):
            assert cmd[idx + 1] != "True"
            assert cmd[idx + 1] != "true"

    def test_spec_draft_params_emits_dash_draft(self, manager, launcher_mock):
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("mtp")
        launcher_mock.spec_draft_params.set("alpha=0.5,beta=0.2")
        cmd = manager.build_cmd()
        assert "-draft" in cmd
        assert cmd[cmd.index("-draft") + 1] == "alpha=0.5,beta=0.2"


# ============================================================================
# ik_llama cross-backend warnings (llama.cpp-only vars set under ik_llama)
# ============================================================================


class TestSpecCrossBackendWarningsIkLlama:
    """Mirror image of TestSpecCrossBackendWarningsLlamaCpp: under
    ik_llama, llama.cpp-only knobs must warn-and-skip."""

    def test_spec_draft_p_split_warns_and_skips(
        self, manager, launcher_mock, capsys
    ):
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("mtp")
        launcher_mock.spec_draft_p_split.set("0.1")
        cmd = manager.build_cmd()
        assert "--spec-draft-p-split" not in cmd
        captured = capsys.readouterr()
        assert "p-split" in captured.err.lower()
        assert "llama.cpp" in captured.err.lower()

    def test_spec_draft_hf_warns_and_skips(
        self, manager, launcher_mock, capsys
    ):
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("mtp")
        launcher_mock.spec_draft_hf.set("org/repo:Q4_K_M")
        cmd = manager.build_cmd()
        assert "--spec-draft-hf" not in cmd
        # Value must not appear in cmd either.
        assert "org/repo:Q4_K_M" not in cmd
        captured = capsys.readouterr()
        assert "hf" in captured.err.lower()
        assert "llama.cpp" in captured.err.lower()

    def test_spec_draft_cpu_moe_warns_and_skips(
        self, manager, launcher_mock, capsys
    ):
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("mtp")
        launcher_mock.spec_draft_cpu_moe.set(True)
        cmd = manager.build_cmd()
        assert "--spec-draft-cpu-moe" not in cmd
        captured = capsys.readouterr()
        assert "cpu-moe" in captured.err.lower()
        assert "llama.cpp" in captured.err.lower()

    def test_spec_draft_n_cpu_moe_warns_and_skips(
        self, manager, launcher_mock, capsys
    ):
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("mtp")
        launcher_mock.spec_draft_n_cpu_moe.set("4")
        cmd = manager.build_cmd()
        assert "--spec-draft-n-cpu-moe" not in cmd
        captured = capsys.readouterr()
        assert "n-cpu-moe" in captured.err.lower()
        assert "llama.cpp" in captured.err.lower()


# ============================================================================
# --no-mmproj (independent of spec_enabled)
# ============================================================================


class TestNoMmprojEmission:
    """The ``--no-mmproj`` flag is emitted by the same block author but is
    completely independent of ``spec_enabled``: a simple llama.cpp-only
    toggle."""

    def test_no_mmproj_true_llamacpp_emits_flag(self, manager, launcher_mock):
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.no_mmproj.set(True)
        cmd = manager.build_cmd()
        assert "--no-mmproj" in cmd

    def test_no_mmproj_true_ik_llama_warns_and_skips(
        self, manager, launcher_mock, capsys
    ):
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.no_mmproj.set(True)
        cmd = manager.build_cmd()
        assert "--no-mmproj" not in cmd
        captured = capsys.readouterr()
        assert "no-mmproj" in captured.err.lower()
        assert "ik_llama" in captured.err.lower()

    def test_no_mmproj_false_llamacpp_omits_flag(self, manager, launcher_mock):
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.no_mmproj.set(False)
        cmd = manager.build_cmd()
        assert "--no-mmproj" not in cmd

    def test_no_mmproj_false_ik_llama_omits_flag(self, manager, launcher_mock):
        """When False under ik_llama, no flag AND no warning."""
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.no_mmproj.set(False)
        cmd = manager.build_cmd()
        assert "--no-mmproj" not in cmd

    def test_no_mmproj_independent_of_spec_enabled(
        self, manager, launcher_mock
    ):
        """``--no-mmproj`` is its own block; ``spec_enabled=False`` doesn't
        suppress it."""
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(False)
        launcher_mock.no_mmproj.set(True)
        cmd = manager.build_cmd()
        assert "--no-mmproj" in cmd


# ============================================================================
# Per-backend spec_type whitelist (CR review): a stale/imported config
# can hold a spec_type value that isn't valid for the active backend.
# build_cmd() must reject it with a stderr warning, not forward it verbatim
# to the server.
# ============================================================================


class TestSpecTypeWhitelist:
    """Reject unknown / cross-backend spec_type values before they reach
    the server. Mirrors the per-backend dropdown choices in the UI."""

    @pytest.mark.parametrize("bad_value", [
        "draft-mtp",  # mainline-only, invalid for ik_llama
        "draft-simple",  # mainline-only
        "draft-eagle3",  # mainline-only
        "garbage",  # nonsense
        "DRAFT-MTP",  # case-sensitive
    ])
    def test_invalid_spec_type_under_ik_llama_skips_and_warns(
        self, manager, launcher_mock, capsys, bad_value
    ):
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set(bad_value)
        cmd = manager.build_cmd()
        assert "--spec-type" not in cmd
        captured = capsys.readouterr()
        assert "spec_type" in captured.err.lower()
        assert bad_value in captured.err

    @pytest.mark.parametrize("bad_value", [
        "mtp",  # ik_llama-only (mainline uses 'draft-mtp'), invalid for llama.cpp
        "suffix",  # ik_llama-only
        "garbage",
        "MTP",  # case-sensitive
    ])
    def test_invalid_spec_type_under_llama_cpp_skips_and_warns(
        self, manager, launcher_mock, capsys, bad_value
    ):
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set(bad_value)
        cmd = manager.build_cmd()
        assert "--spec-type" not in cmd
        captured = capsys.readouterr()
        assert "spec_type" in captured.err.lower()
        assert bad_value in captured.err

    def test_invalid_spec_type_also_suppresses_dependent_flags(
        self, manager, launcher_mock, capsys
    ):
        """If spec_type is rejected, downstream draft tuning flags must
        also be suppressed — they only make sense in the context of a
        valid spec_type."""
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("draft-mtp")  # invalid for ik_llama
        launcher_mock.spec_draft_n_max.set("3")
        cmd = manager.build_cmd()
        assert "--spec-type" not in cmd
        assert "--draft-max" not in cmd
        assert "--spec-draft-n-max" not in cmd

    def test_valid_spec_type_still_emits(self, manager, launcher_mock):
        """Sanity: every UI-valid value still passes the whitelist."""
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("draft-mtp")
        cmd = manager.build_cmd()
        assert "--spec-type" in cmd
        assert cmd[cmd.index("--spec-type") + 1] == "draft-mtp"
