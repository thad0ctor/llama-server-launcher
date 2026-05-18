"""Regression tests for the reasoning/thinking and KV-unification emission
blocks added on top of the existing MTP/Spec implementation.

Mirrors the test style of ``tests/launchers/test_launch.py``: pulls in the
``launcher_mock`` / ``manager`` fixtures from ``tests/launchers/conftest.py``
(which exposes the new vars with default "" values) and exercises both
backends via ``LaunchManager.build_cmd()``.

The two emission blocks live in ``modules/launch.py`` right after the
existing speculative-decoding block; both are independent of spec_enabled
and emit only when their own vars are non-empty.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ============================================================================
# Reasoning / Chat-Template KWargs emission (both backends)
# ============================================================================


class TestReasoningEmission:
    """The reasoning block is independent of spec_enabled and emits only
    when the per-var values are non-empty."""

    def test_reasoning_mode_on_emits_flag(self, manager, launcher_mock):
        launcher_mock.reasoning_mode.set("on")
        cmd = manager.build_cmd()
        assert "--reasoning" in cmd
        assert cmd[cmd.index("--reasoning") + 1] == "on"

    def test_reasoning_mode_off_emits_flag(self, manager, launcher_mock):
        launcher_mock.reasoning_mode.set("off")
        cmd = manager.build_cmd()
        assert "--reasoning" in cmd
        assert cmd[cmd.index("--reasoning") + 1] == "off"

    def test_reasoning_mode_auto_emits_flag(self, manager, launcher_mock):
        launcher_mock.reasoning_mode.set("auto")
        cmd = manager.build_cmd()
        assert "--reasoning" in cmd
        assert cmd[cmd.index("--reasoning") + 1] == "auto"

    def test_reasoning_mode_blank_omits_flag(self, manager, launcher_mock):
        launcher_mock.reasoning_mode.set("")
        cmd = manager.build_cmd()
        assert "--reasoning" not in cmd

    def test_reasoning_mode_invalid_value_omits_flag(self, manager, launcher_mock):
        """The whitelist in launch.py filters out anything that isn't
        on/off/auto, so user-typed garbage is silently dropped."""
        launcher_mock.reasoning_mode.set("invalid")
        cmd = manager.build_cmd()
        assert "--reasoning" not in cmd

    def test_reasoning_mode_works_under_ik_llama_backend(self, manager, launcher_mock):
        """Both backends support --reasoning."""
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.reasoning_mode.set("on")
        cmd = manager.build_cmd()
        assert "--reasoning" in cmd
        assert cmd[cmd.index("--reasoning") + 1] == "on"

    def test_reasoning_format_deepseek_emits_flag(self, manager, launcher_mock):
        launcher_mock.reasoning_format.set("deepseek")
        cmd = manager.build_cmd()
        assert "--reasoning-format" in cmd
        assert cmd[cmd.index("--reasoning-format") + 1] == "deepseek"

    def test_reasoning_format_blank_omits_flag(self, manager, launcher_mock):
        launcher_mock.reasoning_format.set("")
        cmd = manager.build_cmd()
        assert "--reasoning-format" not in cmd

    def test_reasoning_format_works_under_ik_llama_backend(self, manager, launcher_mock):
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.reasoning_format.set("deepseek-legacy")
        cmd = manager.build_cmd()
        assert "--reasoning-format" in cmd
        assert cmd[cmd.index("--reasoning-format") + 1] == "deepseek-legacy"

    def test_reasoning_budget_integer_emits_flag(self, manager, launcher_mock):
        launcher_mock.reasoning_budget.set("2048")
        cmd = manager.build_cmd()
        assert "--reasoning-budget" in cmd
        assert cmd[cmd.index("--reasoning-budget") + 1] == "2048"

    def test_reasoning_budget_blank_omits_flag(self, manager, launcher_mock):
        launcher_mock.reasoning_budget.set("")
        cmd = manager.build_cmd()
        assert "--reasoning-budget" not in cmd

    def test_reasoning_budget_negative_value_still_emits(self, manager, launcher_mock):
        """-1 (unlimited) is a legitimate explicit override."""
        launcher_mock.reasoning_budget.set("-1")
        cmd = manager.build_cmd()
        assert "--reasoning-budget" in cmd
        assert cmd[cmd.index("--reasoning-budget") + 1] == "-1"

    def test_reasoning_budget_message_emits_flag(self, manager, launcher_mock):
        launcher_mock.reasoning_budget_message.set("STOP")
        cmd = manager.build_cmd()
        assert "--reasoning-budget-message" in cmd
        assert cmd[cmd.index("--reasoning-budget-message") + 1] == "STOP"

    def test_reasoning_budget_message_blank_omits_flag(self, manager, launcher_mock):
        launcher_mock.reasoning_budget_message.set("")
        cmd = manager.build_cmd()
        assert "--reasoning-budget-message" not in cmd

    def test_chat_template_kwargs_json_string_emits_flag(self, manager, launcher_mock):
        launcher_mock.chat_template_kwargs.set('{"a":1}')
        cmd = manager.build_cmd()
        assert "--chat-template-kwargs" in cmd
        assert cmd[cmd.index("--chat-template-kwargs") + 1] == '{"a":1}'

    def test_chat_template_kwargs_blank_omits_flag(self, manager, launcher_mock):
        launcher_mock.chat_template_kwargs.set("")
        cmd = manager.build_cmd()
        assert "--chat-template-kwargs" not in cmd

    def test_all_reasoning_flags_together(self, manager, launcher_mock):
        """All five reasoning-related flags emit independently and survive
        in the same command line."""
        launcher_mock.reasoning_mode.set("on")
        launcher_mock.reasoning_format.set("deepseek")
        launcher_mock.reasoning_budget.set("4096")
        launcher_mock.reasoning_budget_message.set("END")
        launcher_mock.chat_template_kwargs.set('{"preserve_thinking":true}')
        cmd = manager.build_cmd()
        assert [
            cmd[cmd.index("--reasoning")],
            cmd[cmd.index("--reasoning") + 1],
        ] == ["--reasoning", "on"]
        assert [
            cmd[cmd.index("--reasoning-format")],
            cmd[cmd.index("--reasoning-format") + 1],
        ] == ["--reasoning-format", "deepseek"]
        assert [
            cmd[cmd.index("--reasoning-budget")],
            cmd[cmd.index("--reasoning-budget") + 1],
        ] == ["--reasoning-budget", "4096"]
        assert [
            cmd[cmd.index("--reasoning-budget-message")],
            cmd[cmd.index("--reasoning-budget-message") + 1],
        ] == ["--reasoning-budget-message", "END"]
        assert [
            cmd[cmd.index("--chat-template-kwargs")],
            cmd[cmd.index("--chat-template-kwargs") + 1],
        ] == ["--chat-template-kwargs", '{"preserve_thinking":true}']

    def test_no_reasoning_flags_emitted_by_default(self, manager, launcher_mock):
        """Zero-noise default: none of the five reasoning flags appear when
        every var is blank."""
        cmd = manager.build_cmd()
        for flag in ("--reasoning", "--reasoning-format", "--reasoning-budget",
                     "--reasoning-budget-message", "--chat-template-kwargs"):
            assert flag not in cmd, f"Unexpected flag emitted by default: {flag}"


# ============================================================================
# KV-Unified / cache-idle-slots emission (llama.cpp only)
# ============================================================================


class TestKvUnifiedEmission:
    """``--kv-unified`` / ``--no-kv-unified`` and ``--cache-idle-slots`` /
    ``--no-cache-idle-slots`` are mainline-only — ik_llama doesn't accept them
    and the emission block warns + skips for ik_llama."""

    def test_kvu_on_emits_kv_unified(self, manager, launcher_mock):
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.kv_unified_mode.set("on")
        cmd = manager.build_cmd()
        assert "--kv-unified" in cmd
        assert "--no-kv-unified" not in cmd

    def test_kvu_off_emits_no_kv_unified(self, manager, launcher_mock):
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.kv_unified_mode.set("off")
        cmd = manager.build_cmd()
        assert "--no-kv-unified" in cmd
        assert "--kv-unified" not in cmd

    def test_kvu_blank_emits_neither(self, manager, launcher_mock):
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.kv_unified_mode.set("")
        cmd = manager.build_cmd()
        assert "--kv-unified" not in cmd
        assert "--no-kv-unified" not in cmd

    def test_cache_idle_slots_on_emits_flag(self, manager, launcher_mock):
        """Happy path: --cache-idle-slots emits only when --kv-unified=on."""
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.kv_unified_mode.set("on")
        launcher_mock.cache_idle_slots_mode.set("on")
        cmd = manager.build_cmd()
        assert "--cache-idle-slots" in cmd
        assert "--no-cache-idle-slots" not in cmd

    def test_cache_idle_slots_off_emits_no_flag(self, manager, launcher_mock):
        """--no-cache-idle-slots requires --kv-unified=on too — emission is
        gated as the last line of defense against stale configs."""
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.kv_unified_mode.set("on")
        launcher_mock.cache_idle_slots_mode.set("off")
        cmd = manager.build_cmd()
        assert "--no-cache-idle-slots" in cmd
        assert "--cache-idle-slots" not in cmd

    def test_cache_idle_slots_blank_emits_neither(self, manager, launcher_mock):
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.cache_idle_slots_mode.set("")
        cmd = manager.build_cmd()
        assert "--cache-idle-slots" not in cmd
        assert "--no-cache-idle-slots" not in cmd

    def test_combo_kvu_on_and_cache_idle_on(self, manager, launcher_mock):
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.kv_unified_mode.set("on")
        launcher_mock.cache_idle_slots_mode.set("on")
        cmd = manager.build_cmd()
        assert "--kv-unified" in cmd
        assert "--cache-idle-slots" in cmd

    def test_ik_llama_kvu_on_emits_nothing_and_warns(
        self, manager, launcher_mock, capsys
    ):
        """ik_llama doesn't support these flags. The block must warn to
        stderr and never push them onto the command."""
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.kv_unified_mode.set("on")
        cmd = manager.build_cmd()
        assert "--kv-unified" not in cmd
        assert "--no-kv-unified" not in cmd
        captured = capsys.readouterr()
        assert "kv-unified" in captured.err.lower()
        assert "ik_llama" in captured.err.lower()

    def test_ik_llama_cache_idle_on_emits_nothing_and_warns(
        self, manager, launcher_mock, capsys
    ):
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.cache_idle_slots_mode.set("on")
        cmd = manager.build_cmd()
        assert "--cache-idle-slots" not in cmd
        assert "--no-cache-idle-slots" not in cmd
        captured = capsys.readouterr()
        assert "cache-idle-slots" in captured.err.lower() or "kv-unified" in captured.err.lower()
        assert "ik_llama" in captured.err.lower()

    def test_ik_llama_kvu_off_and_cis_off_also_warns(
        self, manager, launcher_mock, capsys
    ):
        """Negative values (off) must also be suppressed under ik_llama —
        ik_llama doesn't know --no-kv-unified either."""
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.kv_unified_mode.set("off")
        launcher_mock.cache_idle_slots_mode.set("off")
        cmd = manager.build_cmd()
        for flag in ("--kv-unified", "--no-kv-unified",
                     "--cache-idle-slots", "--no-cache-idle-slots"):
            assert flag not in cmd
        # The test name promises a warning — assert it actually fires.
        captured = capsys.readouterr()
        assert "ik_llama" in captured.err.lower()
        assert ("kv-unified" in captured.err.lower()
                or "cache-idle-slots" in captured.err.lower())

    def test_ik_llama_blank_values_emit_no_warning(
        self, manager, launcher_mock, capsys
    ):
        """Sanity: when both kvu vars are blank under ik_llama, no warning
        about kv-unified should appear (those vars are inactive)."""
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.kv_unified_mode.set("")
        launcher_mock.cache_idle_slots_mode.set("")
        manager.build_cmd()
        captured = capsys.readouterr()
        assert "kv-unified" not in captured.err.lower()

    def test_no_kvu_flags_emitted_by_default(self, manager, launcher_mock):
        """Zero-noise default: when both vars are blank and backend is
        llama.cpp, none of the four kvu/cis flags appear."""
        launcher_mock.backend_selection.set("llama.cpp")
        cmd = manager.build_cmd()
        for flag in ("--kv-unified", "--no-kv-unified",
                     "--cache-idle-slots", "--no-cache-idle-slots"):
            assert flag not in cmd


# ============================================================================
# Hardening: validate integer-only --reasoning-budget at emission
# ============================================================================


class TestReasoningBudgetIntegerEmission:
    """A stale or hand-edited config could store a non-integer
    --reasoning-budget value (the Entry validator only guards live typing).
    The emission block must reject it with a stderr warning rather than
    emitting garbage that crashes llama-server."""

    def test_non_integer_value_is_skipped_and_warns(
        self, manager, launcher_mock, capsys
    ):
        launcher_mock.reasoning_budget.set("abc")
        cmd = manager.build_cmd()
        assert "--reasoning-budget" not in cmd
        captured = capsys.readouterr()
        assert "reasoning-budget" in captured.err.lower()
        assert "integer" in captured.err.lower()

    def test_float_value_is_skipped(self, manager, launcher_mock):
        """A float string is not an integer; reject it."""
        launcher_mock.reasoning_budget.set("3.14")
        cmd = manager.build_cmd()
        assert "--reasoning-budget" not in cmd

    def test_signed_integer_still_emits(self, manager, launcher_mock):
        """Sanity check the existing happy path survives the new validation."""
        launcher_mock.reasoning_budget.set("-42")
        cmd = manager.build_cmd()
        assert "--reasoning-budget" in cmd
        assert cmd[cmd.index("--reasoning-budget") + 1] == "-42"


# ============================================================================
# --cache-idle-slots dependency: emission requires --kv-unified=on.
# The UI clears cache_idle_slots_mode on transitions, but build_cmd() also
# enforces the dependency as a last line of defense against stale or
# hand-edited configs.
# ============================================================================


class TestCacheIdleSlotsRequiresKvUnified:
    """``--cache-idle-slots`` and ``--no-cache-idle-slots`` are only emitted
    when ``--kv-unified=on`` is also being emitted. The server requires
    unified KV for cache-idle-slots; emitting it alone would just produce
    a server-side warning and silent disable."""

    def test_cis_on_without_kvu_is_skipped_with_warning(
        self, manager, launcher_mock, capsys
    ):
        """kv_unified_mode='' + cache_idle_slots_mode='on' must NOT emit
        --cache-idle-slots, and must warn to stderr."""
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.kv_unified_mode.set("")
        launcher_mock.cache_idle_slots_mode.set("on")
        cmd = manager.build_cmd()
        assert "--cache-idle-slots" not in cmd
        assert "--no-cache-idle-slots" not in cmd
        assert "--kv-unified" not in cmd
        captured = capsys.readouterr()
        assert "cache-idle-slots" in captured.err.lower()
        assert "kv-unified" in captured.err.lower()

    def test_cis_on_with_kvu_off_emits_only_no_kvu_with_warning(
        self, manager, launcher_mock, capsys
    ):
        """kv_unified=off + cis=on: --no-kv-unified emits, --cache-idle-slots
        is skipped with a warning."""
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.kv_unified_mode.set("off")
        launcher_mock.cache_idle_slots_mode.set("on")
        cmd = manager.build_cmd()
        assert "--no-kv-unified" in cmd
        assert "--cache-idle-slots" not in cmd
        assert "--no-cache-idle-slots" not in cmd
        captured = capsys.readouterr()
        assert "cache-idle-slots" in captured.err.lower()

    def test_cis_off_without_kvu_is_skipped_with_warning(
        self, manager, launcher_mock, capsys
    ):
        """The negative ('off') value is gated the same way — without
        --kv-unified=on, neither variant emits."""
        launcher_mock.backend_selection.set("llama.cpp")
        launcher_mock.kv_unified_mode.set("")
        launcher_mock.cache_idle_slots_mode.set("off")
        cmd = manager.build_cmd()
        assert "--cache-idle-slots" not in cmd
        assert "--no-cache-idle-slots" not in cmd
        captured = capsys.readouterr()
        assert "cache-idle-slots" in captured.err.lower()


# ============================================================================
# UI-state contract: _refresh_kv_unify_state clears stale cache-idle-slots
# value so emission stays clean. Loads the hyphenated entry module to call
# the method directly with a SimpleNamespace stub.
# ============================================================================


import importlib.util  # noqa: E402
import tkinter as tk  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

ENTRY_PATH = REPO_ROOT / "llamacpp-server-launcher.py"


@pytest.fixture(scope="module")
def entry_module():
    spec = importlib.util.spec_from_file_location("entry_module_kvu_state", ENTRY_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["entry_module_kvu_state"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def kvu_stub(tk_root, entry_module):
    """Stub with the attributes _refresh_kv_unify_state touches.

    Tk vars are real (so set()/get() work and traces would fire if anyone
    wired them). Widgets are MagicMocks with ``winfo_exists() -> True``.
    """
    stub = SimpleNamespace()
    stub.backend_selection = tk.StringVar(master=tk_root, value="llama.cpp")
    stub.kv_unified_mode = tk.StringVar(master=tk_root, value="on")
    stub.cache_idle_slots_mode = tk.StringVar(master=tk_root, value="on")

    def _make_widget():
        w = MagicMock()
        w.winfo_exists.return_value = True
        return w

    stub.kv_unified_mode_combo = _make_widget()
    stub.cache_idle_slots_mode_combo = _make_widget()
    stub.kv_unified_backend_label = _make_widget()
    stub.cache_idle_slots_warn_label = _make_widget()
    return stub


class TestRefreshKvUnifyStateResetsStaleCacheIdleSlots:
    def test_kvu_on_does_not_clear_cache_idle(self, kvu_stub, entry_module):
        """When kv_unified is 'on', cache-idle-slots is meaningful and its
        value must survive the refresh."""
        kvu_stub.kv_unified_mode.set("on")
        kvu_stub.cache_idle_slots_mode.set("on")
        entry_module.LlamaCppLauncher._refresh_kv_unify_state(kvu_stub)
        assert kvu_stub.cache_idle_slots_mode.get() == "on"

    def test_kvu_off_clears_stale_cache_idle(self, kvu_stub, entry_module):
        """The fix: toggling kv_unified away from 'on' must reset
        cache_idle_slots_mode so the launcher doesn't emit
        --cache-idle-slots without --kv-unified."""
        kvu_stub.kv_unified_mode.set("on")
        kvu_stub.cache_idle_slots_mode.set("on")
        # User flips kv_unified -> "off"
        kvu_stub.kv_unified_mode.set("off")
        entry_module.LlamaCppLauncher._refresh_kv_unify_state(kvu_stub)
        assert kvu_stub.cache_idle_slots_mode.get() == ""

    def test_kvu_blank_clears_stale_cache_idle(self, kvu_stub, entry_module):
        """Same thing for the 'blank' (unset) case."""
        kvu_stub.kv_unified_mode.set("on")
        kvu_stub.cache_idle_slots_mode.set("on")
        kvu_stub.kv_unified_mode.set("")
        entry_module.LlamaCppLauncher._refresh_kv_unify_state(kvu_stub)
        assert kvu_stub.cache_idle_slots_mode.get() == ""

    def test_kvu_off_with_blank_cache_idle_leaves_it_blank(self, kvu_stub, entry_module):
        """Sanity: no spurious set() calls when there's nothing to clear."""
        kvu_stub.kv_unified_mode.set("off")
        kvu_stub.cache_idle_slots_mode.set("")
        entry_module.LlamaCppLauncher._refresh_kv_unify_state(kvu_stub)
        assert kvu_stub.cache_idle_slots_mode.get() == ""


class TestValidateIntOrBlank:
    """Tk validatecommand for --reasoning-budget. Allows blank, bare '-',
    and signed integers; rejects floats and non-numeric strings."""

    @pytest.mark.parametrize("value", ["", "-", "0", "1", "-1", "1234567890", "-42"])
    def test_accepts_valid(self, value, entry_module):
        assert entry_module.LlamaCppLauncher._validate_int_or_blank(value) is True

    @pytest.mark.parametrize("value", ["abc", "1.5", "1e3", "1,000", "--1", "0x10"])
    def test_rejects_invalid(self, value, entry_module):
        assert entry_module.LlamaCppLauncher._validate_int_or_blank(value) is False


# ============================================================================
# CR regression: _refresh_spec_tab_state must NOT mutate self.spec_type when
# the stored value isn't valid for the active backend. The stored setting is
# preserved across backend toggles so flipping ik_llama <-> llama.cpp doesn't
# silently destroy a user's previously-chosen draft-mtp / mtp value.
# ============================================================================


@pytest.fixture()
def spec_tab_stub(tk_root, entry_module):
    """Stub with the attributes `_refresh_spec_tab_state` reads. Real Tk vars
    for ``backend_selection``, ``spec_enabled``, ``spec_type``, and the three
    hint vars so set()/get() works. Widget/section dicts are empty — the
    method tolerates missing widgets via ``self._spec_widgets.get(...)`` and
    iterates an empty section set if ``_spec_sections`` is empty."""
    stub = SimpleNamespace()
    stub.backend_selection = tk.StringVar(master=tk_root, value="llama.cpp")
    stub.spec_enabled = tk.BooleanVar(master=tk_root, value=True)
    stub.spec_type = tk.StringVar(master=tk_root, value="none")
    stub.spec_pmin_hint_var = tk.StringVar(master=tk_root, value="")
    stub.spec_psplit_hint_var = tk.StringVar(master=tk_root, value="")
    stub.spec_parallel_hint_var = tk.StringVar(master=tk_root, value="")
    stub.spec_status_var = tk.StringVar(master=tk_root, value="")
    stub._spec_widgets = {}
    stub._spec_sections = {}
    # Mirror the class constants so the method's per-backend whitelist works.
    stub._SPEC_TYPES_LLAMA_CPP = entry_module.LlamaCppLauncher._SPEC_TYPES_LLAMA_CPP
    stub._SPEC_TYPES_IK_LLAMA = entry_module.LlamaCppLauncher._SPEC_TYPES_IK_LLAMA
    return stub


class TestRefreshSpecTabStatePreservesSpecType:
    """Backend toggles must NOT clobber a stored spec_type that's invalid for
    the new backend. The user may be inspecting the other backend briefly and
    intend to flip back; silently resetting their setting is a UX regression."""

    def test_draft_mtp_under_ik_llama_preserves_value(self, spec_tab_stub, entry_module):
        """User has draft-mtp (mainline) selected, then flips to ik_llama.
        Stored value must remain 'draft-mtp' — only the effective behavior
        changes (no flags emit while ik_llama is active)."""
        spec_tab_stub.spec_type.set("draft-mtp")
        spec_tab_stub.backend_selection.set("ik_llama")
        entry_module.SpecTab._refresh_spec_tab_state(spec_tab_stub)
        assert spec_tab_stub.spec_type.get() == "draft-mtp"

    def test_mtp_under_llama_cpp_preserves_value(self, spec_tab_stub, entry_module):
        """And the inverse: ik_llama 'mtp' survives a flip to llama.cpp."""
        spec_tab_stub.spec_type.set("mtp")
        spec_tab_stub.backend_selection.set("llama.cpp")
        entry_module.SpecTab._refresh_spec_tab_state(spec_tab_stub)
        assert spec_tab_stub.spec_type.get() == "mtp"

    def test_round_trip_backend_flip_keeps_value(self, spec_tab_stub, entry_module):
        """Full round-trip: llama.cpp -> ik_llama -> back to llama.cpp.
        The stored draft-mtp value should still be there at the end."""
        spec_tab_stub.spec_type.set("draft-mtp")
        # llama.cpp active — value is valid here
        entry_module.SpecTab._refresh_spec_tab_state(spec_tab_stub)
        assert spec_tab_stub.spec_type.get() == "draft-mtp"
        # Flip to ik_llama — value invalid for this backend
        spec_tab_stub.backend_selection.set("ik_llama")
        entry_module.SpecTab._refresh_spec_tab_state(spec_tab_stub)
        assert spec_tab_stub.spec_type.get() == "draft-mtp"
        # Flip back to llama.cpp — value still there
        spec_tab_stub.backend_selection.set("llama.cpp")
        entry_module.SpecTab._refresh_spec_tab_state(spec_tab_stub)
        assert spec_tab_stub.spec_type.get() == "draft-mtp"

    def test_status_label_explains_inactive_state(self, spec_tab_stub, entry_module):
        """User-facing surface for the preservation: status label tells the
        user the stored value isn't valid on this backend but is preserved."""
        spec_tab_stub.spec_type.set("draft-mtp")
        spec_tab_stub.backend_selection.set("ik_llama")
        entry_module.SpecTab._refresh_spec_tab_state(spec_tab_stub)
        status = spec_tab_stub.spec_status_var.get().lower()
        assert "draft-mtp" in status
        assert "ik_llama" in status or "not valid" in status or "inactive" in status

    def test_valid_spec_type_still_works(self, spec_tab_stub, entry_module):
        """Sanity: a valid spec_type for the current backend stays valid and
        the status label reports 'Active'."""
        spec_tab_stub.spec_type.set("draft-mtp")
        spec_tab_stub.backend_selection.set("llama.cpp")
        entry_module.SpecTab._refresh_spec_tab_state(spec_tab_stub)
        assert spec_tab_stub.spec_type.get() == "draft-mtp"
        assert "active" in spec_tab_stub.spec_status_var.get().lower()


# ============================================================================
# Backend-switch contract: combobox values track the active backend.
# Ported from the deleted tests/ui/test_spec_tab_behavior.py::TestBackendSwitch.
# ============================================================================


class TestRefreshSpecTabStateCombobox:
    """The ``type_combo`` widget values must mirror the per-backend whitelist
    after _refresh_spec_tab_state runs. Locks the contract that draft-mtp is
    visible only under llama.cpp and mtp/suffix only under ik_llama."""

    def test_combobox_values_track_llama_cpp_backend(
        self, spec_tab_stub, entry_module
    ):
        """llama.cpp: draft-mtp present, mtp absent."""
        # Plant a fake combo into _spec_widgets so the refresh method
        # writes the per-backend whitelist into it.
        fake_combo = MagicMock()
        # Mimic mapping-style ``combo["values"] = (...)`` assignment.
        fake_combo._values = None

        def _setitem(key, value):
            if key == "values":
                fake_combo._values = list(value)

        fake_combo.__setitem__ = MagicMock(side_effect=_setitem)
        # Also make it readable like a real ttk.Combobox (not strictly needed).
        spec_tab_stub._spec_widgets["type_combo"] = fake_combo

        spec_tab_stub.backend_selection.set("llama.cpp")
        entry_module.SpecTab._refresh_spec_tab_state(spec_tab_stub)
        values = fake_combo._values
        assert values is not None, "combobox values were not assigned"
        assert "draft-mtp" in values
        assert "mtp" not in values

    def test_combobox_values_track_ik_llama_backend(
        self, spec_tab_stub, entry_module
    ):
        """ik_llama: mtp + suffix present, draft-mtp absent."""
        fake_combo = MagicMock()
        fake_combo._values = None

        def _setitem(key, value):
            if key == "values":
                fake_combo._values = list(value)

        fake_combo.__setitem__ = MagicMock(side_effect=_setitem)
        spec_tab_stub._spec_widgets["type_combo"] = fake_combo

        spec_tab_stub.backend_selection.set("ik_llama")
        entry_module.SpecTab._refresh_spec_tab_state(spec_tab_stub)
        values = fake_combo._values
        assert values is not None
        assert "mtp" in values
        assert "suffix" in values
        assert "draft-mtp" not in values


# ============================================================================
# ik_llama disables both kv-unified combos. Ported from
# tests/ui/test_spec_tab_behavior.py::TestKvUnifiedGating.
# ============================================================================


class TestRefreshKvUnifyStateBackendGating:
    """``_refresh_kv_unify_state`` must disable BOTH the kv_unified and
    cache_idle_slots combos under ik_llama (ik_llama doesn't accept either
    flag) and leave them enabled under llama.cpp (kvu always; cis only when
    kvu=on)."""

    def test_ik_llama_disables_both_combos(self, kvu_stub, entry_module):
        kvu_stub.backend_selection.set("ik_llama")
        entry_module.LlamaCppLauncher._refresh_kv_unify_state(kvu_stub)
        # The method calls ``combo.config(state=tk.DISABLED)`` on both.
        # We just inspect the MagicMock call list — no real Tk needed.
        kvu_combo_calls = [
            c for c in kvu_stub.kv_unified_mode_combo.config.call_args_list
        ]
        cis_combo_calls = [
            c for c in kvu_stub.cache_idle_slots_mode_combo.config.call_args_list
        ]
        # Each combo should have been configured with state=DISABLED.
        assert any(
            ("state" in c.kwargs and c.kwargs["state"] == tk.DISABLED)
            for c in kvu_combo_calls
        ), f"kv_unified combo not disabled under ik_llama; calls={kvu_combo_calls}"
        assert any(
            ("state" in c.kwargs and c.kwargs["state"] == tk.DISABLED)
            for c in cis_combo_calls
        ), f"cache_idle_slots combo not disabled under ik_llama; calls={cis_combo_calls}"

    def test_llama_cpp_with_kvu_on_enables_cis_combo(self, kvu_stub, entry_module):
        """llama.cpp + kvu=on: cis combo enabled (readonly)."""
        kvu_stub.backend_selection.set("llama.cpp")
        kvu_stub.kv_unified_mode.set("on")
        entry_module.LlamaCppLauncher._refresh_kv_unify_state(kvu_stub)
        cis_combo_calls = [
            c for c in kvu_stub.cache_idle_slots_mode_combo.config.call_args_list
        ]
        assert any(
            ("state" in c.kwargs and c.kwargs["state"] == "readonly")
            for c in cis_combo_calls
        ), f"cis combo should be readonly when kvu=on; calls={cis_combo_calls}"

    def test_llama_cpp_with_kvu_off_disables_cis_combo(self, kvu_stub, entry_module):
        """llama.cpp + kvu=off: cis combo disabled (and stale value cleared
        — separately covered by TestRefreshKvUnifyStateResetsStaleCacheIdleSlots)."""
        kvu_stub.backend_selection.set("llama.cpp")
        kvu_stub.kv_unified_mode.set("off")
        entry_module.LlamaCppLauncher._refresh_kv_unify_state(kvu_stub)
        cis_combo_calls = [
            c for c in kvu_stub.cache_idle_slots_mode_combo.config.call_args_list
        ]
        assert any(
            ("state" in c.kwargs and c.kwargs["state"] == tk.DISABLED)
            for c in cis_combo_calls
        ), f"cis combo should be disabled when kvu=off; calls={cis_combo_calls}"


# ============================================================================
# Reasoning-budget FocusOut normalization. Ported from
# tests/ui/test_spec_tab_behavior.py::TestReasoningBudgetFocusOut.
# The original test bound a real <FocusOut> event to a real Entry; the
# normalization logic itself was refactored out of the inline lambda into
# ``LlamaCppLauncher._normalize_reasoning_budget_on_focus_out`` so we can
# call it directly with a SimpleNamespace stub here.
# ============================================================================


class TestNormalizeReasoningBudgetOnFocusOut:
    """A bare ``-`` accepted by the keystroke validator must be normalized
    back to ``""`` on focus-out — otherwise it persists into the config and
    is silently dropped at emission with a stderr warning."""

    @pytest.fixture()
    def reasoning_stub(self, tk_root):
        stub = SimpleNamespace()
        stub.reasoning_budget = tk.StringVar(master=tk_root, value="")
        return stub

    def test_bare_minus_normalizes_to_empty(self, reasoning_stub, entry_module):
        reasoning_stub.reasoning_budget.set("-")
        entry_module.LlamaCppLauncher._normalize_reasoning_budget_on_focus_out(
            reasoning_stub
        )
        assert reasoning_stub.reasoning_budget.get() == ""

    def test_minus_with_whitespace_normalizes(self, reasoning_stub, entry_module):
        """``  -  `` (whitespace around a lone dash) is also a degenerate case."""
        reasoning_stub.reasoning_budget.set("  -  ")
        entry_module.LlamaCppLauncher._normalize_reasoning_budget_on_focus_out(
            reasoning_stub
        )
        assert reasoning_stub.reasoning_budget.get() == ""

    def test_valid_negative_integer_preserved(self, reasoning_stub, entry_module):
        """``-1`` (unlimited) is the legitimate use case — must NOT be wiped."""
        reasoning_stub.reasoning_budget.set("-1")
        entry_module.LlamaCppLauncher._normalize_reasoning_budget_on_focus_out(
            reasoning_stub
        )
        assert reasoning_stub.reasoning_budget.get() == "-1"

    def test_positive_integer_preserved(self, reasoning_stub, entry_module):
        reasoning_stub.reasoning_budget.set("2048")
        entry_module.LlamaCppLauncher._normalize_reasoning_budget_on_focus_out(
            reasoning_stub
        )
        assert reasoning_stub.reasoning_budget.get() == "2048"

    def test_blank_value_left_blank(self, reasoning_stub, entry_module):
        """Sanity: ``""`` -> ``""`` (no spurious set call)."""
        reasoning_stub.reasoning_budget.set("")
        entry_module.LlamaCppLauncher._normalize_reasoning_budget_on_focus_out(
            reasoning_stub
        )
        assert reasoning_stub.reasoning_budget.get() == ""


# ============================================================================
# Draft GPU checkbox count==0 selection-wipe regression. Ported from
# tests/ui/test_spec_tab_behavior.py::TestDraftGpuSelectionWipeRegression.
#
# The bug: ``SpecTab._update_spec_draft_gpu_checkboxes`` used to unconditionally
# overwrite ``app_settings["spec_draft_selected_gpus"]`` with the sanitized
# list — but during early init the SystemInfoManager hasn't completed yet, so
# ``gpu_info["device_count"]==0`` and the sanitized list is always ``[]``,
# wiping the user's persisted selection.
#
# Fix: the mirror-back happens only when ``count > 0 and not manual_mode``,
# so the count==0 path preserves whatever was loaded.
# ============================================================================


class _DraftGpuStub(SimpleNamespace):
    """A SimpleNamespace shaped like ``SpecTab`` for the methods under test.

    Drives ``_update_spec_draft_gpu_checkboxes`` directly without
    instantiating the real Tk widget hierarchy.
    """


@pytest.fixture()
def draft_gpu_stub(tk_root):
    """Stub for ``SpecTab._update_spec_draft_gpu_checkboxes``.

    Provides a fake checkbox-frame whose ``winfo_exists`` returns ``True``
    (so the early-exit guard doesn't trigger) and ``winfo_children`` returns
    an empty list (so the destroy-children loop runs against nothing). The
    launcher reference carries the gpu_info / app_settings / detected_gpu_devices
    state needed by the method.
    """
    stub = _DraftGpuStub()

    # Fake the checkbox frame: winfo_exists True, no children to destroy.
    fake_frame = MagicMock()
    fake_frame.winfo_exists.return_value = True
    fake_frame.winfo_children.return_value = []
    stub.spec_draft_gpu_checkbox_frame = fake_frame

    # Tk vars the method touches directly.
    stub.spec_draft_device = tk.StringVar(master=tk_root, value="")
    stub.spec_draft_gpu_vars = []

    # Avoid the _refresh_spec_tab_state side-effect at the tail of the method.
    stub._refresh_spec_tab_state = lambda: None

    # The launcher sub-object the method reads.
    launcher = SimpleNamespace()
    launcher.gpu_info = {"available": False, "device_count": 0, "devices": []}
    launcher.detected_gpu_devices = []
    launcher.app_settings = {}
    launcher.manual_gpu_mode = tk.BooleanVar(master=tk_root, value=False)
    launcher._save_configs = lambda: None
    stub.launcher = launcher
    return stub


class TestUpdateSpecDraftGpuCheckboxesPreservesSelection:
    """Locks the count==0 initial-render bug: when the async SystemInfoManager
    hasn't completed yet, ``_update_spec_draft_gpu_checkboxes`` must NOT
    wipe ``app_settings["spec_draft_selected_gpus"]`` or ``spec_draft_device``."""

    def test_count_zero_preserves_loaded_selection(self, draft_gpu_stub, entry_module):
        """The exact regression: count==0, persisted selection survives."""
        draft_gpu_stub.launcher.app_settings["spec_draft_selected_gpus"] = [2, 5]
        entry_module.SpecTab._update_spec_draft_gpu_checkboxes(draft_gpu_stub)
        assert draft_gpu_stub.launcher.app_settings["spec_draft_selected_gpus"] == [2, 5], (
            f"selection wiped during count==0 init; got "
            f"{draft_gpu_stub.launcher.app_settings['spec_draft_selected_gpus']!r}"
        )

    def test_count_zero_does_not_clobber_spec_draft_device(self, draft_gpu_stub, entry_module):
        """A persisted free-text ``spec_draft_device`` value (e.g. ``Vulkan0``
        for a non-CUDA backend) must also survive the count==0 init — the
        old code wrote ``""`` back into the Tk var when count==0."""
        draft_gpu_stub.spec_draft_device.set("Vulkan0")
        entry_module.SpecTab._update_spec_draft_gpu_checkboxes(draft_gpu_stub)
        assert draft_gpu_stub.spec_draft_device.get() == "Vulkan0"

    def test_async_detection_sequence_preserves_then_validates(
        self, draft_gpu_stub, entry_module
    ):
        """Stage 1 (count==0) preserves; stage 2 (count==N) sanitizes against
        the now-known device list. Deferred sanitization, not eager wipe."""
        # Stage 1: count==0, persisted selection.
        draft_gpu_stub.launcher.app_settings["spec_draft_selected_gpus"] = [2, 5]
        entry_module.SpecTab._update_spec_draft_gpu_checkboxes(draft_gpu_stub)
        assert draft_gpu_stub.launcher.app_settings["spec_draft_selected_gpus"] == [2, 5]
        # Stage 2: SystemInfoManager finished — 8 GPUs, all indices valid.
        draft_gpu_stub.launcher.gpu_info = {
            "available": True, "device_count": 8, "devices": [],
        }
        draft_gpu_stub.launcher.detected_gpu_devices = [
            {"id": i, "name": f"GPU {i}"} for i in range(8)
        ]
        entry_module.SpecTab._update_spec_draft_gpu_checkboxes(draft_gpu_stub)
        # 2 and 5 are valid indices on an 8-GPU host — still there.
        assert draft_gpu_stub.launcher.app_settings["spec_draft_selected_gpus"] == [2, 5]
        # spec_draft_device rebuilt to match.
        assert draft_gpu_stub.spec_draft_device.get() == "CUDA2,CUDA5"

    def test_count_zero_with_manual_mode_also_preserves(
        self, draft_gpu_stub, entry_module
    ):
        """Manual GPU mode + count==0 is another path through the early
        branch. The mirror-back guard ``count > 0 and not manual_mode``
        must skip both."""
        draft_gpu_stub.launcher.gpu_info = {
            "available": True, "device_count": 0, "devices": [], "manual_mode": True,
        }
        draft_gpu_stub.launcher.manual_gpu_mode.set(True)
        draft_gpu_stub.launcher.app_settings["spec_draft_selected_gpus"] = [2, 5]
        entry_module.SpecTab._update_spec_draft_gpu_checkboxes(draft_gpu_stub)
        assert draft_gpu_stub.launcher.app_settings["spec_draft_selected_gpus"] == [2, 5]

    def test_post_detection_emits_comma_joined_device_string(
        self, draft_gpu_stub, entry_module
    ):
        """Locks the legacy ``TestDraftGpuCheckboxes`` contract: after the
        checkbox grid has been built against detected GPUs, toggling indices
        produces ``CUDA<i>,CUDA<j>,...`` in ``spec_draft_device``."""
        # Build the grid with 3 GPUs and a selection of [0, 2].
        draft_gpu_stub.launcher.gpu_info = {
            "available": True, "device_count": 3, "devices": [],
        }
        draft_gpu_stub.launcher.detected_gpu_devices = [
            {"id": i, "name": f"GPU {i}"} for i in range(3)
        ]
        draft_gpu_stub.launcher.app_settings["spec_draft_selected_gpus"] = [0, 2]
        entry_module.SpecTab._update_spec_draft_gpu_checkboxes(draft_gpu_stub)
        assert draft_gpu_stub.spec_draft_device.get() == "CUDA0,CUDA2"
        # And the persisted selection is mirrored back as well.
        assert draft_gpu_stub.launcher.app_settings["spec_draft_selected_gpus"] == [0, 2]
