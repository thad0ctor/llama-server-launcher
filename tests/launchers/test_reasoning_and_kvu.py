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
        entry_module.LlamaCppLauncher._refresh_spec_tab_state(spec_tab_stub)
        assert spec_tab_stub.spec_type.get() == "draft-mtp"

    def test_mtp_under_llama_cpp_preserves_value(self, spec_tab_stub, entry_module):
        """And the inverse: ik_llama 'mtp' survives a flip to llama.cpp."""
        spec_tab_stub.spec_type.set("mtp")
        spec_tab_stub.backend_selection.set("llama.cpp")
        entry_module.LlamaCppLauncher._refresh_spec_tab_state(spec_tab_stub)
        assert spec_tab_stub.spec_type.get() == "mtp"

    def test_round_trip_backend_flip_keeps_value(self, spec_tab_stub, entry_module):
        """Full round-trip: llama.cpp -> ik_llama -> back to llama.cpp.
        The stored draft-mtp value should still be there at the end."""
        spec_tab_stub.spec_type.set("draft-mtp")
        # llama.cpp active — value is valid here
        entry_module.LlamaCppLauncher._refresh_spec_tab_state(spec_tab_stub)
        assert spec_tab_stub.spec_type.get() == "draft-mtp"
        # Flip to ik_llama — value invalid for this backend
        spec_tab_stub.backend_selection.set("ik_llama")
        entry_module.LlamaCppLauncher._refresh_spec_tab_state(spec_tab_stub)
        assert spec_tab_stub.spec_type.get() == "draft-mtp"
        # Flip back to llama.cpp — value still there
        spec_tab_stub.backend_selection.set("llama.cpp")
        entry_module.LlamaCppLauncher._refresh_spec_tab_state(spec_tab_stub)
        assert spec_tab_stub.spec_type.get() == "draft-mtp"

    def test_status_label_explains_inactive_state(self, spec_tab_stub, entry_module):
        """User-facing surface for the preservation: status label tells the
        user the stored value isn't valid on this backend but is preserved."""
        spec_tab_stub.spec_type.set("draft-mtp")
        spec_tab_stub.backend_selection.set("ik_llama")
        entry_module.LlamaCppLauncher._refresh_spec_tab_state(spec_tab_stub)
        status = spec_tab_stub.spec_status_var.get().lower()
        assert "draft-mtp" in status
        assert "ik_llama" in status or "not valid" in status or "inactive" in status

    def test_valid_spec_type_still_works(self, spec_tab_stub, entry_module):
        """Sanity: a valid spec_type for the current backend stays valid and
        the status label reports 'Active'."""
        spec_tab_stub.spec_type.set("draft-mtp")
        spec_tab_stub.backend_selection.set("llama.cpp")
        entry_module.LlamaCppLauncher._refresh_spec_tab_state(spec_tab_stub)
        assert spec_tab_stub.spec_type.get() == "draft-mtp"
        assert "active" in spec_tab_stub.spec_status_var.get().lower()
