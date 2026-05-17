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

    def test_spec_draft_model_emits(self, manager, launcher_mock, tmp_path):
        # Path is validated before emission (CR Comment B), so it must be a
        # real file. Use tmp_path to materialize a stand-in draft GGUF.
        draft = tmp_path / "draft.gguf"
        draft.write_bytes(b"GGUF\x00")
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("draft-mtp")
        launcher_mock.spec_draft_model.set(str(draft))
        cmd = manager.build_cmd()
        assert "--spec-draft-model" in cmd
        assert cmd[cmd.index("--spec-draft-model") + 1] == str(draft.resolve())

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
            "--spec-draft-ngl",
            "--spec-draft-device",
            "--spec-draft-type-k",
            "--spec-draft-type-v",
            "--spec-draft-cpu-moe",
            "--spec-draft-n-cpu-moe",
        ):
            assert absent not in cmd

    def test_all_llamacpp_draft_knobs_combined(self, manager, launcher_mock, tmp_path):
        """Setting many knobs at once: each one independently emits and
        none drops out due to interaction. Mirrors the
        ``test_all_reasoning_flags_together`` style."""
        # Real file for the path-validation guard (CR Comment B).
        draft = tmp_path / "draft.gguf"
        draft.write_bytes(b"GGUF\x00")
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("draft-mtp")
        launcher_mock.spec_draft_n_max.set("16")
        launcher_mock.spec_draft_n_min.set("2")
        launcher_mock.spec_draft_p_min.set("0.5")
        launcher_mock.spec_draft_p_split.set("0.1")
        launcher_mock.spec_draft_model.set(str(draft))
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
        # Model (resolved absolute path)
        assert cmd[cmd.index("--spec-draft-model") + 1] == str(draft.resolve())
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

    def test_draft_model_uses_model_draft_flag(self, manager, launcher_mock, tmp_path):
        """``spec_draft_model`` -> ``--model-draft`` under ik_llama, NOT
        ``--spec-draft-model``."""
        # Path is validated before emission (CR Comment B), so it must be a
        # real file.
        draft = tmp_path / "draft.gguf"
        draft.write_bytes(b"GGUF\x00")
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("mtp")
        launcher_mock.spec_draft_model.set(str(draft))
        cmd = manager.build_cmd()
        assert "--model-draft" in cmd
        assert cmd[cmd.index("--model-draft") + 1] == str(draft.resolve())
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

    def test_ik_llama_all_draft_knobs_combined(self, manager, launcher_mock, tmp_path):
        """Setting many ik_llama knobs at once: each translation lands
        without interaction."""
        # Real file for the path-validation guard (CR Comment B).
        draft = tmp_path / "draft.gguf"
        draft.write_bytes(b"GGUF\x00")
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("mtp")
        launcher_mock.spec_draft_n_max.set("16")
        launcher_mock.spec_draft_n_min.set("2")
        launcher_mock.spec_draft_p_min.set("0.5")
        launcher_mock.spec_draft_model.set(str(draft))
        launcher_mock.spec_draft_ngl.set("24")
        launcher_mock.spec_draft_device.set("CUDA1")
        launcher_mock.spec_draft_ctk.set("q4_0")
        launcher_mock.spec_draft_ctv.set("q4_0")
        cmd = manager.build_cmd()
        assert cmd[cmd.index("--spec-type") + 1] == "mtp"
        assert cmd[cmd.index("--draft-max") + 1] == "16"
        assert cmd[cmd.index("--draft-min") + 1] == "2"
        assert cmd[cmd.index("--draft-p-min") + 1] == "0.5"
        assert cmd[cmd.index("--model-draft") + 1] == str(draft.resolve())
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


# ============================================================================
# Draft device emission contract (spec_draft_device -> --spec-draft-device / -devd)
# ============================================================================
#
# The new MTP/Spec tab builds the draft device-name string from a checkbox
# grid, but emission is still driven exclusively by the contents of
# self.spec_draft_device (a StringVar). These tests pin the emission
# behavior the checkbox UI ultimately funnels into, so the contract stays
# stable regardless of how the UI populates that var.


class TestSpecDraftDeviceEmission:
    """Direct emission contract for spec_draft_device under both backends."""

    def test_blank_spec_draft_device_omits_flag_llamacpp(self, manager, launcher_mock):
        """Empty string -> no --spec-draft-device on the cmd."""
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("draft-mtp")
        launcher_mock.spec_draft_device.set("")
        cmd = manager.build_cmd()
        assert "--spec-draft-device" not in cmd

    def test_single_cuda_device_llamacpp(self, manager, launcher_mock):
        """A single ``CUDA0`` string lands verbatim under llama.cpp's long flag."""
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("draft-mtp")
        launcher_mock.spec_draft_device.set("CUDA0")
        cmd = manager.build_cmd()
        assert "--spec-draft-device" in cmd
        assert cmd[cmd.index("--spec-draft-device") + 1] == "CUDA0"

    def test_multi_cuda_device_csv_llamacpp(self, manager, launcher_mock):
        """A comma-joined list (e.g. ``CUDA0,CUDA1``) is forwarded as one
        token — emission does NOT split it. Matches what the checkbox grid
        produces when the user selects multiple draft GPUs."""
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("draft-mtp")
        launcher_mock.spec_draft_device.set("CUDA0,CUDA1")
        cmd = manager.build_cmd()
        assert "--spec-draft-device" in cmd
        assert cmd[cmd.index("--spec-draft-device") + 1] == "CUDA0,CUDA1"

    def test_multi_cuda_device_csv_ik_llama_uses_short_form(
        self, manager, launcher_mock
    ):
        """Same value under ik_llama emits via the short flag ``-devd`` and
        does NOT emit the long ``--spec-draft-device`` form."""
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("mtp")
        launcher_mock.spec_draft_device.set("CUDA0,CUDA1")
        cmd = manager.build_cmd()
        assert "-devd" in cmd
        assert cmd[cmd.index("-devd") + 1] == "CUDA0,CUDA1"
        assert "--spec-draft-device" not in cmd


# ============================================================================
# Draft KV cache type combobox: allowed values
# ============================================================================


class TestSpecDraftCacheTypeComboboxValues:
    """The draft K/V cache type combos must offer a blank ("don't emit")
    option in addition to the same set the main combos offer. The "" entry
    is the one that makes the combos different from the main combos."""

    _EXPECTED_VALUES = ("", "f16", "f32", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "q6_k")

    def test_ctk_combo_values_match_expected(self, tk_root):
        """spec_draft_ctk_combo offers the full draft set including blank."""
        from tkinter import ttk
        var = tk.StringVar(master=tk_root, value="")
        combo = ttk.Combobox(
            tk_root,
            textvariable=var,
            values=self._EXPECTED_VALUES,
            state="readonly",
        )
        assert tuple(combo.cget("values")) == self._EXPECTED_VALUES

    def test_ctv_combo_values_match_expected(self, tk_root):
        """Same contract for spec_draft_ctv_combo (paste-twin of the K combo)."""
        from tkinter import ttk
        var = tk.StringVar(master=tk_root, value="")
        combo = ttk.Combobox(
            tk_root,
            textvariable=var,
            values=self._EXPECTED_VALUES,
            state="readonly",
        )
        assert tuple(combo.cget("values")) == self._EXPECTED_VALUES

    def test_blank_value_omits_flag_under_llamacpp(self, manager, launcher_mock):
        """When the combo is left blank, no --spec-draft-type-k flag emits."""
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("draft-mtp")
        launcher_mock.spec_draft_ctk.set("")
        launcher_mock.spec_draft_ctv.set("")
        cmd = manager.build_cmd()
        assert "--spec-draft-type-k" not in cmd
        assert "--spec-draft-type-v" not in cmd

    def test_blank_value_omits_flag_under_ik_llama(self, manager, launcher_mock):
        """Same for ik_llama's short forms."""
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("mtp")
        launcher_mock.spec_draft_ctk.set("")
        launcher_mock.spec_draft_ctv.set("")
        cmd = manager.build_cmd()
        assert "-ctkd" not in cmd
        assert "-ctvd" not in cmd


# ============================================================================
# CR Comment A: gate draft-only flags by active spec_type
# ============================================================================
#
# Forwarding ``spec_draft_*`` values regardless of the active spec_type was a
# UI/CLI contract violation: a saved config preserved from a prior draft-mtp
# session would emit ``--spec-draft-model`` / ``--spec-draft-n-max`` even
# after the user switched to ngram-simple (which doesn't read those flags).
# These tests pin the gate: ``spec_draft_*`` emission requires a
# draft-capable spec_type (llama.cpp: draft-simple/draft-eagle3/draft-mtp;
# ik_llama: mtp).


# llama.cpp non-draft-capable spec_types (the ones that must NOT forward
# spec_draft_* fields even when those fields hold stale values).
LLAMACPP_NON_DRAFT_SPEC_TYPES = [
    "ngram-simple",
    "ngram-map-k",
    "ngram-map-k4v",
    "ngram-mod",
    "ngram-cache",
]

# ik_llama non-draft-capable spec_types.
IK_LLAMA_NON_DRAFT_SPEC_TYPES = [
    "ngram-cache",
    "ngram-simple",
    "ngram-map-k",
    "ngram-map-k4v",
    "ngram-mod",
    "suffix",
]


class TestSpecDraftFlagsGatedByType:
    """``spec_draft_*`` flags only emit under draft-capable spec_types.

    Non-draft modes (ngram-*, suffix, ngram-cache) must drop those values
    silently — the UI hides their Draft Model section, so emission would
    silently violate the grayed-field contract for saved configs.
    """

    @pytest.mark.parametrize("spec_type", LLAMACPP_NON_DRAFT_SPEC_TYPES)
    def test_llamacpp_ngram_drops_spec_draft_n_max(
        self, manager, launcher_mock, tmp_path, spec_type
    ):
        """llama.cpp ngram-*/cache: stale draft tuning vars must be dropped."""
        draft = tmp_path / "draft.gguf"
        draft.write_bytes(b"GGUF\x00")
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set(spec_type)
        launcher_mock.spec_draft_n_max.set("3")
        launcher_mock.spec_draft_model.set(str(draft))
        cmd = manager.build_cmd()
        assert "--spec-type" in cmd
        assert cmd[cmd.index("--spec-type") + 1] == spec_type
        # Draft-only knobs must not appear.
        for absent in (
            "--spec-draft-n-max",
            "--spec-draft-n-min",
            "--spec-draft-p-min",
            "--spec-draft-p-split",
            "--spec-draft-model",
            "--spec-draft-ngl",
            "--spec-draft-device",
            "--spec-draft-type-k",
            "--spec-draft-type-v",
            "--spec-draft-cpu-moe",
            "--spec-draft-n-cpu-moe",
        ):
            assert absent not in cmd, (
                f"{absent} leaked into cmd for non-draft spec_type {spec_type!r}"
            )

    @pytest.mark.parametrize("spec_type", LLAMACPP_NON_DRAFT_SPEC_TYPES)
    def test_llamacpp_ngram_drops_all_draft_offload_and_moe(
        self, manager, launcher_mock, spec_type
    ):
        """Same gate applies to the ngl/device/ctk/ctv + cpu_moe family."""
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set(spec_type)
        launcher_mock.spec_draft_ngl.set("32")
        launcher_mock.spec_draft_device.set("CUDA0")
        launcher_mock.spec_draft_ctk.set("q8_0")
        launcher_mock.spec_draft_ctv.set("q8_0")
        launcher_mock.spec_draft_cpu_moe.set(True)
        launcher_mock.spec_draft_n_cpu_moe.set("4")
        cmd = manager.build_cmd()
        for absent in (
            "--spec-draft-ngl",
            "--spec-draft-device",
            "--spec-draft-type-k",
            "--spec-draft-type-v",
            "--spec-draft-cpu-moe",
            "--spec-draft-n-cpu-moe",
        ):
            assert absent not in cmd, (
                f"{absent} leaked into cmd for non-draft spec_type {spec_type!r}"
            )

    @pytest.mark.parametrize("spec_type", IK_LLAMA_NON_DRAFT_SPEC_TYPES)
    def test_ik_llama_non_draft_drops_draft_max(
        self, manager, launcher_mock, spec_type
    ):
        """ik_llama ngram-*/suffix: ``--draft-max`` etc. must NOT emit."""
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set(spec_type)
        launcher_mock.spec_draft_n_max.set("3")
        cmd = manager.build_cmd()
        assert "--spec-type" in cmd
        assert cmd[cmd.index("--spec-type") + 1] == spec_type
        assert "--draft-max" not in cmd
        assert "--draft-min" not in cmd
        assert "--draft-p-min" not in cmd

    def test_ik_llama_suffix_drops_model_draft(
        self, manager, launcher_mock, tmp_path
    ):
        """ik_llama suffix mode: ``--model-draft`` must NOT emit even when
        the path is valid; suffix doesn't use a separate draft model."""
        draft = tmp_path / "draft.gguf"
        draft.write_bytes(b"GGUF\x00")
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("suffix")
        launcher_mock.spec_draft_model.set(str(draft))
        launcher_mock.spec_suffix_pattern_len.set("8")
        cmd = manager.build_cmd()
        assert "--spec-type" in cmd
        # Suffix knobs ARE emitted...
        assert "--suffix-pattern-len" in cmd
        assert cmd[cmd.index("--suffix-pattern-len") + 1] == "8"
        # ...but the draft model is NOT (suffix has no separate draft model).
        assert "--model-draft" not in cmd

    @pytest.mark.parametrize("spec_type", IK_LLAMA_NON_DRAFT_SPEC_TYPES)
    def test_ik_llama_non_draft_drops_all_short_offload(
        self, manager, launcher_mock, spec_type
    ):
        """The short ``-ngld``/``-devd``/``-ctkd``/``-ctvd`` family is gated
        too."""
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set(spec_type)
        launcher_mock.spec_draft_ngl.set("24")
        launcher_mock.spec_draft_device.set("CUDA1")
        launcher_mock.spec_draft_ctk.set("q4_0")
        launcher_mock.spec_draft_ctv.set("q4_0")
        cmd = manager.build_cmd()
        for absent in ("-ngld", "-devd", "-ctkd", "-ctvd"):
            assert absent not in cmd, (
                f"{absent} leaked into cmd for non-draft spec_type {spec_type!r}"
            )

    def test_llamacpp_draft_mtp_still_emits_n_max(
        self, manager, launcher_mock
    ):
        """Regression: a draft-capable spec_type still forwards the draft
        tuning knobs (this test guards against an over-zealous gate)."""
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("draft-mtp")
        launcher_mock.spec_draft_n_max.set("3")
        cmd = manager.build_cmd()
        assert "--spec-draft-n-max" in cmd
        assert cmd[cmd.index("--spec-draft-n-max") + 1] == "3"

    def test_ik_llama_mtp_still_emits_draft_max(
        self, manager, launcher_mock
    ):
        """Regression: ik_llama's only draft-capable spec_type (``mtp``)
        still forwards ``--draft-max``."""
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("mtp")
        launcher_mock.spec_draft_n_max.set("7")
        cmd = manager.build_cmd()
        assert "--draft-max" in cmd
        assert cmd[cmd.index("--draft-max") + 1] == "7"

    def test_llamacpp_switch_from_draft_mtp_to_ngram_drops_all_draft(
        self, manager, launcher_mock, tmp_path
    ):
        """The exact scenario from the CR prompt: a user who set up
        draft-mtp with every draft knob filled in, then switched to
        ngram-simple, must NOT see any draft flag leak through. Only
        ``--spec-type`` and ngram knobs should appear."""
        draft = tmp_path / "draft.gguf"
        draft.write_bytes(b"GGUF\x00")
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        # User set ALL the draft knobs while spec_type was draft-mtp...
        launcher_mock.spec_draft_n_max.set("3")
        launcher_mock.spec_draft_n_min.set("1")
        launcher_mock.spec_draft_p_min.set("0.75")
        launcher_mock.spec_draft_p_split.set("0.10")
        launcher_mock.spec_draft_model.set(str(draft))
        launcher_mock.spec_draft_ngl.set("32")
        launcher_mock.spec_draft_device.set("CUDA0")
        launcher_mock.spec_draft_ctk.set("q8_0")
        launcher_mock.spec_draft_ctv.set("q8_0")
        launcher_mock.spec_draft_cpu_moe.set(True)
        launcher_mock.spec_draft_n_cpu_moe.set("4")
        # ...then switched to ngram-simple, and added ngram-specific tuning.
        launcher_mock.spec_type.set("ngram-simple")
        launcher_mock.spec_ngram_simple_size_n.set("4")
        cmd = manager.build_cmd()
        # spec_type + ngram knobs are emitted.
        assert "--spec-type" in cmd
        assert cmd[cmd.index("--spec-type") + 1] == "ngram-simple"
        assert "--spec-ngram-simple-size-n" in cmd
        # NO draft flag leaks.
        for absent in (
            "--spec-draft-n-max",
            "--spec-draft-n-min",
            "--spec-draft-p-min",
            "--spec-draft-p-split",
            "--spec-draft-model",
            "--spec-draft-ngl",
            "--spec-draft-device",
            "--spec-draft-type-k",
            "--spec-draft-type-v",
            "--spec-draft-cpu-moe",
            "--spec-draft-n-cpu-moe",
        ):
            assert absent not in cmd, (
                f"{absent} leaked after switching from draft-mtp to ngram-simple"
            )


# ============================================================================
# CR Comment B: validate the draft-model path before appending it
# ============================================================================
#
# ``spec_draft_model`` is a listbox-picked path. A saved config can hold a
# stale path pointing to a moved/deleted draft GGUF; emitting it verbatim
# leads to a confusing server-side failure later. Mirror the main ``-m``
# behaviour: ``Path(...).is_file()`` gate + stderr warning + skip on miss.


class TestSpecDraftModelPathValidation:
    """``--spec-draft-model`` / ``--model-draft`` are gated on
    ``Path(value).is_file()``. Misses log a stderr warning and skip the
    flag instead of forwarding garbage."""

    def test_llamacpp_existing_file_emits_resolved_absolute_path(
        self, manager, launcher_mock, tmp_path
    ):
        """Valid path: the flag emits, value is the resolved absolute path."""
        draft = tmp_path / "draft.gguf"
        draft.write_bytes(b"GGUF\x00")
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("draft-mtp")
        launcher_mock.spec_draft_model.set(str(draft))
        cmd = manager.build_cmd()
        assert "--spec-draft-model" in cmd
        emitted = cmd[cmd.index("--spec-draft-model") + 1]
        # Resolved absolute path is what gets emitted.
        assert emitted == str(draft.resolve())
        assert Path(emitted).is_absolute()

    def test_llamacpp_nonexistent_path_skips_flag_with_warning(
        self, manager, launcher_mock, tmp_path, capsys
    ):
        """Nonexistent path: flag is NOT emitted, stderr warning fires."""
        bogus = tmp_path / "does_not_exist.gguf"
        # Intentionally do NOT create the file.
        assert not bogus.exists()
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("draft-mtp")
        launcher_mock.spec_draft_model.set(str(bogus))
        cmd = manager.build_cmd()
        assert "--spec-draft-model" not in cmd
        captured = capsys.readouterr()
        assert str(bogus) in captured.err
        assert "--spec-draft-model" in captured.err
        assert "not a file" in captured.err

    def test_llamacpp_directory_path_skips_flag_with_warning(
        self, manager, launcher_mock, tmp_path, capsys
    ):
        """Directory path (not a file) is rejected even though it exists."""
        dir_path = tmp_path / "models_dir"
        dir_path.mkdir()
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("draft-mtp")
        launcher_mock.spec_draft_model.set(str(dir_path))
        cmd = manager.build_cmd()
        assert "--spec-draft-model" not in cmd
        captured = capsys.readouterr()
        assert "not a file" in captured.err
        assert str(dir_path) in captured.err

    def test_ik_llama_existing_file_emits_resolved_absolute_path(
        self, manager, launcher_mock, tmp_path
    ):
        """Same contract under ik_llama: resolved absolute path emitted."""
        draft = tmp_path / "draft.gguf"
        draft.write_bytes(b"GGUF\x00")
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("mtp")
        launcher_mock.spec_draft_model.set(str(draft))
        cmd = manager.build_cmd()
        assert "--model-draft" in cmd
        emitted = cmd[cmd.index("--model-draft") + 1]
        assert emitted == str(draft.resolve())
        assert Path(emitted).is_absolute()

    def test_ik_llama_nonexistent_path_skips_flag_with_warning(
        self, manager, launcher_mock, tmp_path, capsys
    ):
        """ik_llama miss: warning must reference ``--model-draft`` (NOT
        the llama.cpp flag name)."""
        bogus = tmp_path / "missing-draft.gguf"
        assert not bogus.exists()
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("mtp")
        launcher_mock.spec_draft_model.set(str(bogus))
        cmd = manager.build_cmd()
        assert "--model-draft" not in cmd
        captured = capsys.readouterr()
        assert str(bogus) in captured.err
        assert "--model-draft" in captured.err
        assert "--spec-draft-model" not in captured.err


# ============================================================================
# Pre-fill defaults for blank draft-tuning fields
# ============================================================================
#
# Exercises ``LlamaCppLauncher._apply_spec_defaults_if_blank`` via the
# ``entry_module`` import pattern (see test_reasoning_and_kvu.py's
# kvu_stub fixture for prior art). The method only touches three vars
# (spec_draft_n_max, spec_draft_p_min, spec_draft_p_split) and only when
# they're blank — user-typed values must persist.


import importlib.util  # noqa: E402
import tkinter as tk  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ENTRY_PATH = REPO_ROOT / "llamacpp-server-launcher.py"


@pytest.fixture(scope="module")
def entry_module():
    spec = importlib.util.spec_from_file_location("entry_module_spec_defaults", ENTRY_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["entry_module_spec_defaults"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def spec_defaults_stub(tk_root):
    """Minimal SimpleNamespace stub exposing the tk vars
    ``_apply_spec_defaults_if_blank`` reads/writes."""
    stub = SimpleNamespace()
    stub.spec_enabled = tk.BooleanVar(master=tk_root, value=True)
    stub.spec_type = tk.StringVar(master=tk_root, value="")
    stub.spec_draft_n_max = tk.StringVar(master=tk_root, value="")
    stub.spec_draft_n_min = tk.StringVar(master=tk_root, value="")
    stub.spec_draft_p_min = tk.StringVar(master=tk_root, value="")
    stub.spec_draft_p_split = tk.StringVar(master=tk_root, value="")
    return stub


class TestSpecDefaultsPrefill:
    """Verify the spec-defaults prefill helper only fills blanks and only
    for spec_types that benefit from defaults."""

    def test_spec_disabled_is_noop(self, spec_defaults_stub, entry_module):
        """When the master is off, nothing must be written even if the
        spec_type would otherwise trigger defaults."""
        spec_defaults_stub.spec_enabled.set(False)
        spec_defaults_stub.spec_type.set("draft-mtp")
        entry_module.LlamaCppLauncher._apply_spec_defaults_if_blank(spec_defaults_stub)
        assert spec_defaults_stub.spec_draft_n_max.get() == ""
        assert spec_defaults_stub.spec_draft_n_min.get() == ""
        assert spec_defaults_stub.spec_draft_p_min.get() == ""
        assert spec_defaults_stub.spec_draft_p_split.get() == ""

    def test_draft_mtp_prefills_n_max_3(self, spec_defaults_stub, entry_module):
        """draft-mtp uses n_max=3 (MTP sweet spot, not the binary default of 16).
        n_min=0 means 'always speculate'."""
        spec_defaults_stub.spec_enabled.set(True)
        spec_defaults_stub.spec_type.set("draft-mtp")
        entry_module.LlamaCppLauncher._apply_spec_defaults_if_blank(spec_defaults_stub)
        assert spec_defaults_stub.spec_draft_n_max.get() == "3"
        assert spec_defaults_stub.spec_draft_n_min.get() == "0"
        assert spec_defaults_stub.spec_draft_p_min.get() == "0.75"
        assert spec_defaults_stub.spec_draft_p_split.get() == "0.10"

    def test_mtp_ik_llama_prefills_same_as_draft_mtp(self, spec_defaults_stub, entry_module):
        """ik_llama's ``mtp`` spec_type shares the same defaults — p-split
        is still set even though ik_llama skips it at emission (warns)."""
        spec_defaults_stub.spec_enabled.set(True)
        spec_defaults_stub.spec_type.set("mtp")
        entry_module.LlamaCppLauncher._apply_spec_defaults_if_blank(spec_defaults_stub)
        assert spec_defaults_stub.spec_draft_n_max.get() == "3"
        assert spec_defaults_stub.spec_draft_n_min.get() == "0"
        assert spec_defaults_stub.spec_draft_p_min.get() == "0.75"
        assert spec_defaults_stub.spec_draft_p_split.get() == "0.10"

    def test_draft_simple_prefills_n_max_16(self, spec_defaults_stub, entry_module):
        """draft-simple / draft-eagle3 use the binary default of n_max=16."""
        spec_defaults_stub.spec_enabled.set(True)
        spec_defaults_stub.spec_type.set("draft-simple")
        entry_module.LlamaCppLauncher._apply_spec_defaults_if_blank(spec_defaults_stub)
        assert spec_defaults_stub.spec_draft_n_max.get() == "16"
        assert spec_defaults_stub.spec_draft_n_min.get() == "0"
        assert spec_defaults_stub.spec_draft_p_min.get() == "0.75"
        assert spec_defaults_stub.spec_draft_p_split.get() == "0.10"

    def test_ngram_simple_no_prefill(self, spec_defaults_stub, entry_module):
        """ngram-* types don't use the draft tuning knobs and must not be
        autofilled (would clutter the UI for users picking ngram)."""
        spec_defaults_stub.spec_enabled.set(True)
        spec_defaults_stub.spec_type.set("ngram-simple")
        entry_module.LlamaCppLauncher._apply_spec_defaults_if_blank(spec_defaults_stub)
        assert spec_defaults_stub.spec_draft_n_max.get() == ""
        assert spec_defaults_stub.spec_draft_n_min.get() == ""
        assert spec_defaults_stub.spec_draft_p_min.get() == ""
        assert spec_defaults_stub.spec_draft_p_split.get() == ""

    def test_user_typed_value_persists(self, spec_defaults_stub, entry_module):
        """If the user has already typed a value, prefill must NOT overwrite
        it — only blank fields are filled."""
        spec_defaults_stub.spec_enabled.set(True)
        spec_defaults_stub.spec_type.set("draft-mtp")
        spec_defaults_stub.spec_draft_n_max.set("5")
        spec_defaults_stub.spec_draft_n_min.set("2")
        entry_module.LlamaCppLauncher._apply_spec_defaults_if_blank(spec_defaults_stub)
        # User's typed values stay; the other two get default-filled because they're blank.
        assert spec_defaults_stub.spec_draft_n_max.get() == "5"
        assert spec_defaults_stub.spec_draft_n_min.get() == "2"
        assert spec_defaults_stub.spec_draft_p_min.get() == "0.75"
        assert spec_defaults_stub.spec_draft_p_split.get() == "0.10"


# ============================================================================
# Draft GPU-layers sync helpers (_set_spec_draft_gpu_layers et al.)
# ============================================================================
#
# Parity test against the main ``_set_gpu_layers`` clamping logic, ported to
# the draft equivalent. The contract:
# * input=N (entry, from_slider=False)  -> int = N (no clamp, even past max)
# * input=N (slider, from_slider=True)  -> int = min(N, max)
# * input=-1                            -> int = max (or 0 if max unknown)


@pytest.fixture()
def draft_layers_stub(tk_root):
    """Stub with the four Tk vars _set_spec_draft_gpu_layers reads/writes."""
    stub = SimpleNamespace()
    stub.spec_draft_ngl_int = tk.IntVar(master=tk_root, value=0)
    stub.max_spec_draft_gpu_layers = tk.IntVar(master=tk_root, value=0)
    # _set_spec_draft_gpu_layers doesn't read these, but the sibling sync
    # helpers do. Pre-create so a follow-up test can use the same stub.
    stub.spec_draft_ngl = tk.StringVar(master=tk_root, value="0")
    return stub


class TestSetSpecDraftGpuLayers:
    """Parametrized parity test for the clamp/promote/-1-as-max contract."""

    @pytest.mark.parametrize(
        "input_value,from_slider,max_layers,expected_int",
        [
            # Entry input below max: int matches verbatim.
            (5, False, 10, 5),
            # Slider input above max: clamped to max.
            (15, True, 10, 10),
            # -1 from entry: maps to max.
            (-1, False, 10, 10),
            # -1 with max=0 (no analysis yet): maps to 0.
            (-1, False, 0, 0),
            # Entry input above max: NOT clamped — user can manually exceed max.
            (15, False, 10, 15),
            # Slider input below max: stays as-is.
            (3, True, 10, 3),
        ],
    )
    def test_clamp_matrix(
        self,
        draft_layers_stub,
        entry_module,
        input_value,
        from_slider,
        max_layers,
        expected_int,
    ):
        draft_layers_stub.max_spec_draft_gpu_layers.set(max_layers)
        entry_module.LlamaCppLauncher._set_spec_draft_gpu_layers(
            draft_layers_stub, input_value, from_slider=from_slider
        )
        assert draft_layers_stub.spec_draft_ngl_int.get() == expected_int


class TestValidateSpecDraftGpuLayersEntry:
    """Validation: accept blank/dash/-1/non-negative; reject everything else."""

    @pytest.mark.parametrize("value", ["", "-", "0", "1", "100", "-1", "999"])
    def test_accepts_valid(self, draft_layers_stub, entry_module, value):
        assert entry_module.LlamaCppLauncher._validate_spec_draft_gpu_layers_entry(
            draft_layers_stub, value
        ) is True

    @pytest.mark.parametrize("value", ["abc", "1.5", "-2", "1e3", "0x10", "--1"])
    def test_rejects_invalid(self, draft_layers_stub, entry_module, value):
        assert entry_module.LlamaCppLauncher._validate_spec_draft_gpu_layers_entry(
            draft_layers_stub, value
        ) is False


# ============================================================================
# MTP enforces --parallel 1 (single-slot operation).
# Two layers of defense:
#   1) The UI auto-sets self.parallel = "1" when MTP is selected
#      (covered by tests/launchers/test_reasoning_and_kvu.py for the Tk side
#       and in this file by TestMtpParallelDefault below for the helper).
#   2) build_cmd() overrides any non-"1" value at emission and warns. This
#      is the authoritative last line of defense regardless of what other
#      UI surface set --parallel.
# ============================================================================


class TestMtpParallelDefault:
    """The MTP/Spec trace callback ``_apply_mtp_parallel_default`` forces
    ``self.parallel`` to "1" whenever MTP mode is active, overwriting any
    pre-existing value."""

    @pytest.fixture
    def mtp_parallel_stub(self, tk_root, entry_module):
        stub = SimpleNamespace()
        stub.spec_enabled = tk.BooleanVar(master=tk_root, value=True)
        stub.spec_type = tk.StringVar(master=tk_root, value="")
        stub.parallel = tk.StringVar(master=tk_root, value="1")
        return stub

    def test_draft_mtp_forces_parallel_to_1(self, mtp_parallel_stub, entry_module):
        mtp_parallel_stub.parallel.set("8")
        mtp_parallel_stub.spec_type.set("draft-mtp")
        entry_module.LlamaCppLauncher._apply_mtp_parallel_default(mtp_parallel_stub)
        assert mtp_parallel_stub.parallel.get() == "1"

    def test_ik_llama_mtp_forces_parallel_to_1(self, mtp_parallel_stub, entry_module):
        mtp_parallel_stub.parallel.set("4")
        mtp_parallel_stub.spec_type.set("mtp")
        entry_module.LlamaCppLauncher._apply_mtp_parallel_default(mtp_parallel_stub)
        assert mtp_parallel_stub.parallel.get() == "1"

    def test_overwrites_even_if_user_typed_value(self, mtp_parallel_stub, entry_module):
        """Unlike soft prefills, this is a hard constraint — user input is
        OVERWRITTEN, not preserved."""
        mtp_parallel_stub.parallel.set("16")
        mtp_parallel_stub.spec_type.set("draft-mtp")
        entry_module.LlamaCppLauncher._apply_mtp_parallel_default(mtp_parallel_stub)
        assert mtp_parallel_stub.parallel.get() == "1"

    def test_non_mtp_spec_type_leaves_parallel_alone(self, mtp_parallel_stub, entry_module):
        mtp_parallel_stub.parallel.set("8")
        mtp_parallel_stub.spec_type.set("ngram-simple")
        entry_module.LlamaCppLauncher._apply_mtp_parallel_default(mtp_parallel_stub)
        assert mtp_parallel_stub.parallel.get() == "8"

    def test_spec_disabled_leaves_parallel_alone(self, mtp_parallel_stub, entry_module):
        """When master is off, no enforcement should happen even if spec_type
        happens to be draft-mtp."""
        mtp_parallel_stub.spec_enabled.set(False)
        mtp_parallel_stub.parallel.set("8")
        mtp_parallel_stub.spec_type.set("draft-mtp")
        entry_module.LlamaCppLauncher._apply_mtp_parallel_default(mtp_parallel_stub)
        assert mtp_parallel_stub.parallel.get() == "8"

    def test_already_1_is_idempotent(self, mtp_parallel_stub, entry_module):
        mtp_parallel_stub.parallel.set("1")
        mtp_parallel_stub.spec_type.set("draft-mtp")
        entry_module.LlamaCppLauncher._apply_mtp_parallel_default(mtp_parallel_stub)
        assert mtp_parallel_stub.parallel.get() == "1"


class TestMtpParallelEmissionOverride:
    """build_cmd() forces --parallel 1 (omitted because it matches default)
    whenever MTP is active, even if launcher.parallel still says something
    else. This is the authoritative last line of defense — protects against
    any future UI surface setting --parallel without coordinating with the
    MTP/Spec tab."""

    def test_mtp_active_with_parallel_8_overrides_to_1_and_warns(
        self, manager, launcher_mock, capsys
    ):
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("draft-mtp")
        launcher_mock.parallel.set("8")
        cmd = manager.build_cmd()
        # --parallel 1 is the binary default so it's omitted from argv,
        # but the wrong value (8) must NOT appear.
        assert "--parallel" not in cmd
        # Specifically: no "8" right after a --parallel token.
        captured = capsys.readouterr()
        assert "MTP requires --parallel 1" in captured.err
        assert "'8'" in captured.err

    def test_mtp_active_with_parallel_1_emits_nothing_no_warning(
        self, manager, launcher_mock, capsys
    ):
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("draft-mtp")
        launcher_mock.parallel.set("1")
        cmd = manager.build_cmd()
        assert "--parallel" not in cmd  # matches default, omitted
        captured = capsys.readouterr()
        assert "MTP requires --parallel" not in captured.err

    def test_mtp_inactive_with_parallel_8_emits_normally(
        self, manager, launcher_mock, capsys
    ):
        """When MTP is NOT active, the multi-slot value passes through."""
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(False)
        launcher_mock.parallel.set("8")
        cmd = manager.build_cmd()
        assert "--parallel" in cmd
        assert cmd[cmd.index("--parallel") + 1] == "8"
        captured = capsys.readouterr()
        assert "MTP requires --parallel" not in captured.err

    def test_ngram_with_parallel_8_does_not_override(self, manager, launcher_mock, capsys):
        """spec_type=ngram-simple is not MTP — no override should fire."""
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("ngram-simple")
        launcher_mock.parallel.set("8")
        cmd = manager.build_cmd()
        assert "--parallel" in cmd
        assert cmd[cmd.index("--parallel") + 1] == "8"
        captured = capsys.readouterr()
        assert "MTP requires --parallel" not in captured.err

    def test_ik_llama_mtp_with_parallel_4_also_overrides(
        self, manager, launcher_mock, capsys
    ):
        """ik_llama's 'mtp' spec_type triggers the same override."""
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.spec_enabled.set(True)
        launcher_mock.spec_type.set("mtp")
        launcher_mock.parallel.set("4")
        cmd = manager.build_cmd()
        assert "--parallel" not in cmd
        captured = capsys.readouterr()
        assert "MTP requires --parallel 1" in captured.err
        assert "'4'" in captured.err


# ============================================================================
# "Reset to defaults" button — OVERWRITES the four common controls with the
# recommended values for the active spec_type. Unlike the soft prefill that
# only fills blanks, this is an explicit user action that ignores any
# existing values. For spec_types without recommended defaults (ngram-*,
# suffix, ngram-cache, none) the fields are cleared.
# ============================================================================


class TestResetSpecDefaults:
    """``_reset_spec_defaults`` is the click handler for the "Reset to
    defaults" button on the Common draft controls section."""

    @pytest.fixture
    def reset_stub(self, tk_root):
        stub = SimpleNamespace()
        stub.spec_type = tk.StringVar(master=tk_root, value="")
        stub.spec_draft_n_max = tk.StringVar(master=tk_root, value="")
        stub.spec_draft_n_min = tk.StringVar(master=tk_root, value="")
        stub.spec_draft_p_min = tk.StringVar(master=tk_root, value="")
        stub.spec_draft_p_split = tk.StringVar(master=tk_root, value="")
        return stub

    def test_draft_mtp_overwrites_to_mtp_defaults(self, reset_stub, entry_module):
        """Even if the user has typed values, draft-mtp reset wipes them
        to the MTP-recommended set (n_max=3, n_min=0, p_min=0.75, p_split=0.10)."""
        reset_stub.spec_draft_n_max.set("999")
        reset_stub.spec_draft_n_min.set("42")
        reset_stub.spec_draft_p_min.set("0.01")
        reset_stub.spec_draft_p_split.set("0.99")
        reset_stub.spec_type.set("draft-mtp")
        entry_module.LlamaCppLauncher._reset_spec_defaults(reset_stub)
        assert reset_stub.spec_draft_n_max.get() == "3"
        assert reset_stub.spec_draft_n_min.get() == "0"
        assert reset_stub.spec_draft_p_min.get() == "0.75"
        assert reset_stub.spec_draft_p_split.get() == "0.10"

    def test_mtp_ik_llama_overwrites_to_mtp_defaults(self, reset_stub, entry_module):
        reset_stub.spec_draft_n_max.set("999")
        reset_stub.spec_type.set("mtp")
        entry_module.LlamaCppLauncher._reset_spec_defaults(reset_stub)
        assert reset_stub.spec_draft_n_max.get() == "3"
        assert reset_stub.spec_draft_n_min.get() == "0"
        assert reset_stub.spec_draft_p_min.get() == "0.75"
        assert reset_stub.spec_draft_p_split.get() == "0.10"

    @pytest.mark.parametrize("draft_type", ["draft-simple", "draft-eagle3"])
    def test_classical_draft_overwrites_to_n_max_16(self, reset_stub, entry_module, draft_type):
        reset_stub.spec_draft_n_max.set("999")
        reset_stub.spec_type.set(draft_type)
        entry_module.LlamaCppLauncher._reset_spec_defaults(reset_stub)
        assert reset_stub.spec_draft_n_max.get() == "16"
        assert reset_stub.spec_draft_n_min.get() == "0"
        assert reset_stub.spec_draft_p_min.get() == "0.75"
        assert reset_stub.spec_draft_p_split.get() == "0.10"

    @pytest.mark.parametrize("spec_type", [
        "ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod", "ngram-cache",
        "suffix", "none", "",
    ])
    def test_no_recommended_defaults_clears_all_fields(
        self, reset_stub, entry_module, spec_type
    ):
        """For spec_types without recommended defaults, reset clears the
        fields to blank (= use binary defaults)."""
        reset_stub.spec_draft_n_max.set("999")
        reset_stub.spec_draft_n_min.set("42")
        reset_stub.spec_draft_p_min.set("0.01")
        reset_stub.spec_draft_p_split.set("0.99")
        reset_stub.spec_type.set(spec_type)
        entry_module.LlamaCppLauncher._reset_spec_defaults(reset_stub)
        assert reset_stub.spec_draft_n_max.get() == ""
        assert reset_stub.spec_draft_n_min.get() == ""
        assert reset_stub.spec_draft_p_min.get() == ""
        assert reset_stub.spec_draft_p_split.get() == ""

    def test_reset_is_idempotent(self, reset_stub, entry_module):
        """Calling reset twice produces the same result as calling it once."""
        reset_stub.spec_type.set("draft-mtp")
        entry_module.LlamaCppLauncher._reset_spec_defaults(reset_stub)
        first = (reset_stub.spec_draft_n_max.get(), reset_stub.spec_draft_n_min.get(),
                 reset_stub.spec_draft_p_min.get(), reset_stub.spec_draft_p_split.get())
        entry_module.LlamaCppLauncher._reset_spec_defaults(reset_stub)
        second = (reset_stub.spec_draft_n_max.get(), reset_stub.spec_draft_n_min.get(),
                  reset_stub.spec_draft_p_min.get(), reset_stub.spec_draft_p_split.get())
        assert first == second
