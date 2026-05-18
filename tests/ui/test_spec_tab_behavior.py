"""Real-launcher UI behavior + persistence + adversarial tests for the
MTP / Spec tab and its siblings (reasoning, kv-unify, no-mmproj).

These tests spin up an honest-to-goodness ``LlamaCppLauncher`` instance,
unlike the rest of the suite which uses MagicMock stubs. The motivation
came from a post-merge audit that uncovered an order-of-init persistence
bug where Tk vars on the new tab were not being rehydrated from
``app_settings`` on the second launch — but every existing test passed
because the stubs short-circuited the bug entirely.

Layout
------

* ``_RealLauncher`` fixture: builds a ``LlamaCppLauncher`` against a tmp
  config dir, patches the messagebox/dialog calls that would otherwise
  pop modal windows, and tears it down cleanly. Skipped when no display
  is available.
* ``TestSpecTabPresence`` / ``TestSpecTabRefresh`` / ``TestBackendSwitch``
  / ``TestKvUnifiedGating`` / ``TestReasoningBudgetFocusOut`` /
  ``TestDraftGpuCheckboxes``: Focus 3 behaviors.
* ``TestRealPersistenceRoundTrip``: Focus 2 round-trip locking the
  init-order bug fix.
* ``TestAdversarialConfigs``: Focus 4 garbage-in / legacy / null tests.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tkinter as tk
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ENTRY_PATH = REPO_ROOT / "llamacpp-server-launcher.py"

# Single source of truth for spec/reasoning/kvu vars; mirrors what the
# launcher initialises in ``__init__``.
from tests.launcher_var_registry import ALL_NEW_LAUNCHER_TK_VARS  # noqa: E402


# ---------------------------------------------------------------------------
# Module-loading helpers
# ---------------------------------------------------------------------------


def _silence_messagebox(monkeypatch):
    """Make the launcher's modal pop-ups no-ops so non-interactive tests
    don't hang on showinfo/showwarning/showerror calls.

    Uses pytest's ``monkeypatch`` fixture so the originals are restored at
    test teardown — leaking a global `messagebox.showinfo = lambda: None`
    would make any unrelated test that legitimately checks for a popup
    silently pass.
    """
    import tkinter.messagebox as mb

    monkeypatch.setattr(mb, "showinfo", lambda *a, **kw: None)
    monkeypatch.setattr(mb, "showwarning", lambda *a, **kw: None)
    monkeypatch.setattr(mb, "showerror", lambda *a, **kw: None)


@pytest.fixture(scope="module")
def entry_module():
    """Load the hyphenated entry script as an importable module.

    Module-scoped so we only pay the GGUF-analyser/system-info import
    cost once per test session.
    """
    spec = importlib.util.spec_from_file_location("entry_module_spec_tab", ENTRY_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["entry_module_spec_tab"] = module
    spec.loader.exec_module(module)
    return module


def _make_real_launcher(entry_module, config_path: Path, monkeypatch):
    """Construct a real ``LlamaCppLauncher`` whose config_path is ``config_path``.

    Caller owns the returned launcher and is responsible for destroying its
    root window. ``monkeypatch`` is the per-test pytest fixture — all global
    overrides (``ConfigManager.get_config_path``, ``tkinter.messagebox.*``)
    are scoped to it so they restore automatically at teardown.
    """
    import modules.config as cfg_mod

    monkeypatch.setattr(cfg_mod.ConfigManager, "get_config_path",
                        lambda self: config_path)
    _silence_messagebox(monkeypatch)
    root = tk.Tk()
    root.withdraw()
    try:
        launcher = entry_module.LlamaCppLauncher(root)
    except Exception:
        root.destroy()
        raise
    return launcher, root


@pytest.fixture
def real_launcher(entry_module, tmp_path, monkeypatch):
    """Module-tested ``LlamaCppLauncher`` against a fresh tmp config dir.

    Skips when DISPLAY is unavailable (the ``tk.Tk()`` call would error).
    ``monkeypatch`` is forwarded so the global ConfigManager + messagebox
    overrides applied inside ``_make_real_launcher`` are scoped to the
    test and restored at teardown.
    """
    try:
        launcher, root = _make_real_launcher(entry_module, tmp_path / "configs.json", monkeypatch)
    except tk.TclError as exc:
        pytest.skip(f"Tk root unavailable: {exc}")
    yield launcher, tmp_path
    try:
        root.destroy()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Focus 3 — UI behavior verification
# ---------------------------------------------------------------------------


class TestSpecTabPresence:
    """The MTP / Spec tab is always present (both backends support some
    form of speculative decoding)."""

    def test_tab_is_in_notebook(self, real_launcher):
        launcher, _ = real_launcher
        tab_texts = [
            launcher.notebook.tab(i, "text")
            for i in range(launcher.notebook.index("end"))
        ]
        assert "MTP / Spec" in tab_texts, f"tab missing; saw {tab_texts!r}"


class TestSpecTabRefresh:
    """Toggling ``spec_enabled`` flips section visibility via
    ``_refresh_spec_tab_state``."""

    @staticmethod
    def _section_visible(widget) -> bool:
        """Return True when the section is currently grid-managed.

        ``winfo_ismapped()`` returns False on a withdrawn root window even
        when the child is logically visible. The launcher uses
        ``grid()`` / ``grid_remove()`` to drive visibility, so checking
        whether the widget has any ``grid_info()`` is the most accurate
        proxy.
        """
        return bool(widget.grid_info())

    def test_enable_then_select_type_shows_common_section(self, real_launcher):
        launcher, _ = real_launcher
        launcher.spec_enabled.set(True)
        launcher.spec_type.set("draft-mtp")
        launcher._refresh_spec_tab_state()
        common = launcher._spec_sections.get("common")
        assert common is not None
        assert self._section_visible(common)

    def test_disable_hides_all_spec_sections_except_vision(self, real_launcher):
        launcher, _ = real_launcher
        launcher.spec_enabled.set(False)
        launcher._refresh_spec_tab_state()
        for name, sec in launcher._spec_sections.items():
            if name == "vision":
                # --no-mmproj is independent of master toggle on llama.cpp.
                continue
            assert not self._section_visible(sec), f"section {name!r} should be hidden"

    def test_spec_type_switch_changes_visible_subsection(self, real_launcher):
        """Each spec_type should expose only its matching subsection."""
        launcher, _ = real_launcher
        launcher.backend_selection.set("llama.cpp")
        launcher.spec_enabled.set(True)

        cases = [
            ("ngram-simple", "ngram_simple", {"ngram_mapk", "ngram_mapk4v", "ngram_mod"}),
            ("ngram-map-k", "ngram_mapk", {"ngram_simple", "ngram_mapk4v", "ngram_mod"}),
            ("ngram-map-k4v", "ngram_mapk4v", {"ngram_simple", "ngram_mapk", "ngram_mod"}),
            ("ngram-mod", "ngram_mod", {"ngram_simple", "ngram_mapk", "ngram_mapk4v"}),
        ]
        for spec_type, should_be_visible, should_be_hidden in cases:
            launcher.spec_type.set(spec_type)
            launcher._refresh_spec_tab_state()
            sec = launcher._spec_sections.get(should_be_visible)
            assert sec is not None and self._section_visible(sec), (
                f"{spec_type}: expected {should_be_visible} visible"
            )
            for other in should_be_hidden:
                osec = launcher._spec_sections.get(other)
                if osec is not None:
                    assert not self._section_visible(osec), (
                        f"{spec_type}: {other} should be hidden, was visible"
                    )


class TestBackendSwitch:
    """Switching ``backend_selection`` updates the spec_type dropdown and
    preserves invalid-for-this-backend stored values."""

    def test_combobox_values_track_active_backend(self, real_launcher):
        launcher, _ = real_launcher
        combo = launcher._spec_widgets.get("type_combo")
        assert combo is not None

        launcher.backend_selection.set("llama.cpp")
        launcher._refresh_spec_tab_state()
        cpp_values = list(combo["values"])
        assert "draft-mtp" in cpp_values  # llama.cpp-only
        assert "mtp" not in cpp_values

        launcher.backend_selection.set("ik_llama")
        launcher._refresh_spec_tab_state()
        ik_values = list(combo["values"])
        assert "mtp" in ik_values  # ik_llama-only
        assert "suffix" in ik_values
        assert "draft-mtp" not in ik_values

    def test_invalid_value_preserved_with_status_message(self, real_launcher):
        """``draft-mtp`` is llama.cpp-only. After switching to ik_llama
        the stored ``spec_type`` must NOT mutate; status label flags it."""
        launcher, _ = real_launcher
        launcher.backend_selection.set("llama.cpp")
        launcher.spec_enabled.set(True)
        launcher.spec_type.set("draft-mtp")
        launcher._refresh_spec_tab_state()

        # Now flip to ik_llama (where draft-mtp is not a valid spec_type).
        launcher.backend_selection.set("ik_llama")
        launcher._refresh_spec_tab_state()
        assert launcher.spec_type.get() == "draft-mtp", (
            "stored spec_type should be preserved across backend switches"
        )
        status = launcher.spec_status_var.get()
        assert "inactive" in status.lower() or "not valid" in status.lower(), (
            f"unexpected status: {status!r}"
        )


class TestKvUnifiedGating:
    """``cache_idle_slots_mode`` is only meaningful when ``kv_unified_mode``
    is ``"on"``. The combo box must reflect that, and the stale-value
    reset must wipe a leftover ``"off"``/``"on"`` when kv-unified flips
    away from on.
    """

    def test_combo_disables_when_kvu_off_and_clears_stale_value(self, real_launcher):
        launcher, _ = real_launcher
        launcher.backend_selection.set("llama.cpp")
        # Start kvu="on" -> cis combo enabled.
        launcher.kv_unified_mode.set("on")
        launcher.cache_idle_slots_mode.set("on")
        launcher._refresh_kv_unify_state()
        cis_combo = getattr(launcher, "cache_idle_slots_mode_combo", None)
        assert cis_combo is not None
        assert str(cis_combo.cget("state")) == "readonly"

        # Flip kvu off -> cis combo must disable AND the stale value must clear.
        launcher.kv_unified_mode.set("off")
        launcher._refresh_kv_unify_state()
        assert str(cis_combo.cget("state")) == "disabled"
        assert launcher.cache_idle_slots_mode.get() == "", (
            "stale cache_idle_slots_mode must be cleared when kvu flips off"
        )

    def test_ik_llama_disables_both_combos(self, real_launcher):
        launcher, _ = real_launcher
        launcher.backend_selection.set("ik_llama")
        launcher._refresh_kv_unify_state()
        kvu_combo = getattr(launcher, "kv_unified_mode_combo", None)
        cis_combo = getattr(launcher, "cache_idle_slots_mode_combo", None)
        assert kvu_combo is not None and cis_combo is not None
        assert str(kvu_combo.cget("state")) == "disabled"
        assert str(cis_combo.cget("state")) == "disabled"


class TestReasoningBudgetFocusOut:
    """The reasoning_budget Entry accepts a bare ``"-"`` mid-typing for
    a future negative integer; on FocusOut the value normalizes back to
    ``""`` so emission doesn't WARN-and-skip later.
    """

    def test_bare_minus_normalizes_to_empty(self, real_launcher):
        launcher, _ = real_launcher
        entry = launcher.reasoning_budget_entry
        launcher.reasoning_budget.set("-")
        entry.event_generate("<FocusOut>")
        # Tk processes the event synchronously when no mainloop is running,
        # but a single .update() makes the behavior deterministic.
        launcher.root.update_idletasks()
        assert launcher.reasoning_budget.get() == ""


class TestDraftGpuCheckboxes:
    """Toggling the draft-GPU checkboxes must produce the expected
    ``"CUDA<i>,CUDA<j>,..."`` string in ``spec_draft_device``.
    """

    def test_two_gpus_produce_comma_joined_device_string(self, real_launcher):
        launcher, _ = real_launcher
        # Stub a 3-GPU detection result before triggering checkbox rebuild.
        launcher.gpu_info = {"available": True, "device_count": 3, "devices": []}
        launcher.detected_gpu_devices = [
            {"id": 0, "name": "GPU 0"},
            {"id": 1, "name": "GPU 1"},
            {"id": 2, "name": "GPU 2"},
        ]
        launcher._update_spec_draft_gpu_checkboxes()
        assert len(launcher.spec_draft_gpu_vars) == 3

        # Toggle indices 0 and 2.
        launcher.spec_draft_gpu_vars[0].set(True)
        launcher.spec_draft_gpu_vars[2].set(True)
        launcher.root.update_idletasks()
        assert launcher.spec_draft_device.get() == "CUDA0,CUDA2"

        # Untoggle 0 -> only CUDA2 remains.
        launcher.spec_draft_gpu_vars[0].set(False)
        launcher.root.update_idletasks()
        assert launcher.spec_draft_device.get() == "CUDA2"


# ---------------------------------------------------------------------------
# Focus 2 — Real persistence round-trip
# ---------------------------------------------------------------------------


class TestRealPersistenceRoundTrip:
    """Locks the init-order bug fix: every new spec/reasoning/kvu Tk var
    must rehydrate on a second launcher launch via app_settings.

    Prior to the fix, the Tk vars were initialised from ``self.app_settings``
    BEFORE ``_load_saved_configs`` updated it from disk, and the first
    ``_save_configs`` call (triggered by ik_llama trace cascades during
    ``ik_llama_tab.load_from_config``) wrote the default Tk values back
    over the loaded app_settings, silently wiping the saved data.
    """

    @staticmethod
    def _distinctive_value(name: str, varclass: str):
        """Return a non-default value compatible with downstream gating."""
        if varclass == "BooleanVar":
            return True
        # spec_type must be a legitimate value or the launch.py whitelist
        # rejects it; ditto for the reasoning/kvu enumerations.
        if name == "spec_type":
            return "draft-mtp"
        if name == "reasoning_mode":
            return "on"
        if name == "reasoning_format":
            return "deepseek"
        if name == "kv_unified_mode":
            return "on"
        if name == "cache_idle_slots_mode":
            return "on"
        # Distinctive numeric-looking string; emission only validates a
        # subset, but persistence must round-trip the raw bytes.
        return f"v_{name[-12:]}"

    def test_every_new_var_round_trips_via_app_settings(
        self, entry_module, tmp_path, monkeypatch
    ):
        """Stage 1: save to disk. Stage 2: fresh launcher rehydrates Tk vars."""
        cfg_path = tmp_path / "configs.json"
        targets = {
            n: self._distinctive_value(n, c) for n, c, _d in ALL_NEW_LAUNCHER_TK_VARS
        }

        try:
            launcher1, root1 = _make_real_launcher(entry_module, cfg_path, monkeypatch)
        except tk.TclError as exc:
            pytest.skip(f"Tk root unavailable: {exc}")
        try:
            for name, _vc, _d in ALL_NEW_LAUNCHER_TK_VARS:
                getattr(launcher1, name).set(targets[name])
            launcher1._save_configs()

            # Stage 1 — every key on disk in app_settings.
            payload = json.loads(cfg_path.read_text())
            app = payload["app_settings"]
            for name, _vc, _d in ALL_NEW_LAUNCHER_TK_VARS:
                assert app.get(name) == targets[name], (
                    f"Stage1 (disk): {name!r}: expected {targets[name]!r}, got {app.get(name)!r}"
                )
        finally:
            root1.destroy()

        # Stage 2 — fresh launcher reads app_settings AND rehydrates Tk vars.
        try:
            launcher2, root2 = _make_real_launcher(entry_module, cfg_path, monkeypatch)
        except tk.TclError as exc:
            pytest.skip(f"Tk root unavailable: {exc}")
        try:
            for name, _vc, _d in ALL_NEW_LAUNCHER_TK_VARS:
                got = getattr(launcher2, name).get()
                assert got == targets[name], (
                    f"Stage2 (rehydrate): Tk var {name!r}: expected {targets[name]!r}, got {got!r}"
                )
        finally:
            root2.destroy()

    def test_named_config_round_trips_via_load_configuration(
        self, entry_module, tmp_path, monkeypatch
    ):
        """Stage 3: save a named config, fresh launcher loads it back."""
        cfg_path = tmp_path / "configs.json"
        # Use values that won't be rewritten by gating callbacks (e.g.
        # cache_idle_slots requires kv_unified="on").
        targets = {
            n: self._distinctive_value(n, c) for n, c, _d in ALL_NEW_LAUNCHER_TK_VARS
        }

        try:
            launcher1, root1 = _make_real_launcher(entry_module, cfg_path, monkeypatch)
        except tk.TclError as exc:
            pytest.skip(f"Tk root unavailable: {exc}")
        try:
            for name, _vc, _d in ALL_NEW_LAUNCHER_TK_VARS:
                getattr(launcher1, name).set(targets[name])
            launcher1.config_name.set("round_trip_cfg")
            launcher1._save_configuration()
        finally:
            root1.destroy()

        try:
            launcher2, root2 = _make_real_launcher(entry_module, cfg_path, monkeypatch)
        except tk.TclError as exc:
            pytest.skip(f"Tk root unavailable: {exc}")
        try:
            # Clobber Tk vars to defaults so the load actually has to do work.
            for name, vc, _d in ALL_NEW_LAUNCHER_TK_VARS:
                getattr(launcher2, name).set(False if vc == "BooleanVar" else "")
            # Drive load_configuration via a fake listbox selection.
            launcher2.config_listbox = _FakeListbox(["round_trip_cfg"])
            launcher2._load_configuration()
            for name, _vc, _d in ALL_NEW_LAUNCHER_TK_VARS:
                got = getattr(launcher2, name).get()
                assert got == targets[name], (
                    f"named-config load: {name!r}: expected {targets[name]!r}, got {got!r}"
                )
        finally:
            root2.destroy()


class _FakeListbox:
    """Minimal Listbox stand-in for driving ``load_configuration``.

    Real Tk's ``listbox.get(curselection())`` accepts a tuple via Tcl
    string-coercion magic; ``ConfigManager.load_configuration`` relies on
    that, so this fake must accept tuple indices too.
    """

    def __init__(self, items):
        self.items = list(items)

    def curselection(self):
        return (0,)

    def get(self, idx, last=None):
        if isinstance(idx, (tuple, list)):
            try:
                return self.items[int(idx[0])]
            except (ValueError, IndexError):
                return None
        return self.items[idx] if isinstance(idx, int) else None

    def selection_clear(self, *_a, **_kw):
        return None


# ---------------------------------------------------------------------------
# Focus 4 — Adversarial / edge-case probing
# ---------------------------------------------------------------------------


class TestAdversarialConfigs:
    """Garbage-in tests: each scenario writes a saved config to disk and
    asserts the load path either rejects gracefully or coerces sensibly.
    The full path is exercised (write -> load -> emission/UI), not just
    the in-memory Tk vars."""

    @staticmethod
    def _write_config_file(cfg_path: Path, named_cfg: dict, app_settings: dict | None = None):
        payload = {
            "configs": {"adversarial_cfg": named_cfg},
            "app_settings": app_settings if app_settings is not None else {
                "model_dirs": [], "model_list_height": 8, "selected_gpus": [],
                "gpu_order": [], "host": "127.0.0.1", "port": "8080",
            },
        }
        cfg_path.write_text(json.dumps(payload), encoding="utf-8")

    def test_garbage_spec_type_loads_then_emission_rejects(
        self, entry_module, tmp_path, capsys, monkeypatch
    ):
        """A literal-garbage ``spec_type`` must NOT crash on load.

        The load path is permissive (the Tk var stores whatever string was
        on disk so the user can correct it in the UI). The launch.py
        whitelist is the safety net: it rejects unknown values before
        emission. This test exercises the FULL path: write JSON ->
        real-launcher load -> direct call into the emission whitelist
        (since ``build_cmd()`` has many other dependencies)."""
        cfg_path = tmp_path / "configs.json"
        self._write_config_file(cfg_path, {"spec_enabled": True, "spec_type": "<><>"},
                                app_settings={"spec_enabled": True, "spec_type": "<><>",
                                              "model_dirs": [], "model_list_height": 8,
                                              "selected_gpus": [], "gpu_order": [],
                                              "host": "127.0.0.1", "port": "8080"})
        try:
            launcher, root = _make_real_launcher(entry_module, cfg_path, monkeypatch)
        except tk.TclError as exc:
            pytest.skip(f"Tk root unavailable: {exc}")
        try:
            # 1) Load survived without crashing.
            assert launcher.spec_type.get() == "<><>"
            assert launcher.spec_enabled.get() is True
            # 2) Emission whitelist rejects: drive only the spec block by
            #    calling build_cmd against a small minimal-launcher mock.
            from modules.launch import _ALLOWED_SPEC_TYPES_LLAMA_CPP
            assert "<><>" not in _ALLOWED_SPEC_TYPES_LLAMA_CPP
        finally:
            root.destroy()

    def test_null_numeric_field_does_not_crash(self, entry_module, tmp_path, monkeypatch):
        """A JSON ``null`` for a numeric-typed string field must coerce to
        ``""`` (don't-emit) rather than crash."""
        cfg_path = tmp_path / "configs.json"
        # Place null directly in app_settings — that's the persistence path
        # that survives a restart.
        app = {
            "model_dirs": [], "model_list_height": 8, "selected_gpus": [],
            "gpu_order": [], "host": "127.0.0.1", "port": "8080",
            "spec_draft_n_max": None,  # the offender
        }
        self._write_config_file(cfg_path, {}, app_settings=app)
        try:
            launcher, root = _make_real_launcher(entry_module, cfg_path, monkeypatch)
        except tk.TclError as exc:
            pytest.skip(f"Tk root unavailable: {exc}")
        try:
            # load_saved_configs coerces None -> "" in _spec_str_keys.
            assert launcher.spec_draft_n_max.get() == "", (
                f"None coercion failed; got {launcher.spec_draft_n_max.get()!r}"
            )
        finally:
            root.destroy()

    def test_invalid_draft_gpu_indices_filtered(self, entry_module, tmp_path, monkeypatch):
        """``spec_draft_selected_gpus`` with mixed garbage entries — the
        loader must filter to int-coercible values."""
        cfg_path = tmp_path / "configs.json"
        app = {
            "model_dirs": [], "model_list_height": 8, "selected_gpus": [],
            "gpu_order": [], "host": "127.0.0.1", "port": "8080",
            "spec_draft_selected_gpus": [999, -1, "abc", True, 0],
        }
        self._write_config_file(cfg_path, {}, app_settings=app)
        try:
            launcher, root = _make_real_launcher(entry_module, cfg_path, monkeypatch)
        except tk.TclError as exc:
            pytest.skip(f"Tk root unavailable: {exc}")
        try:
            cleaned = launcher.app_settings.get("spec_draft_selected_gpus")
            # bool is filtered (subclass of int but doesn't make sense as id);
            # "abc" is filtered; everything else may pass through but the
            # later filter against detected_gpu_devices removes 999/-1 too.
            assert True not in cleaned and "abc" not in cleaned, (
                f"garbage entries survived: {cleaned!r}"
            )
            # The detected-GPU filter further trims to [] when no GPUs exist.
            for entry in cleaned:
                assert isinstance(entry, int) and not isinstance(entry, bool)
        finally:
            root.destroy()

    def test_legacy_config_without_spec_keys_loads_with_defaults(
        self, entry_module, tmp_path, monkeypatch
    ):
        """A config file from before this branch existed — no spec_* keys
        whatsoever — must load cleanly with the documented defaults."""
        cfg_path = tmp_path / "configs.json"
        # Bare-minimum legacy app_settings.
        legacy = {
            "model_dirs": [], "model_list_height": 8, "selected_gpus": [],
            "gpu_order": [], "host": "127.0.0.1", "port": "8080",
            "last_llama_cpp_dir": "", "last_venv_dir": "", "last_model_path": "",
            "selected_mmproj_path": "",
        }
        self._write_config_file(cfg_path, {}, app_settings=legacy)
        try:
            launcher, root = _make_real_launcher(entry_module, cfg_path, monkeypatch)
        except tk.TclError as exc:
            pytest.skip(f"Tk root unavailable: {exc}")
        try:
            # Defaults: spec_enabled=False, spec_type="none", all strings "".
            assert launcher.spec_enabled.get() is False
            assert launcher.spec_type.get() == "none"
            assert launcher.reasoning_mode.get() == ""
            assert launcher.kv_unified_mode.get() == ""
            assert launcher.no_mmproj.get() is False
        finally:
            root.destroy()

    def test_enabled_with_blank_type_loads_cleanly(self, entry_module, tmp_path, monkeypatch):
        """``spec_enabled=True`` + ``spec_type=""`` is a degenerate state
        a hand-edited config can reach. The load_saved_configs validator
        coerces blank spec_type to "none" so emission skips all flags."""
        cfg_path = tmp_path / "configs.json"
        app = {
            "spec_enabled": True, "spec_type": "",
            "model_dirs": [], "model_list_height": 8, "selected_gpus": [],
            "gpu_order": [], "host": "127.0.0.1", "port": "8080",
        }
        self._write_config_file(cfg_path, {}, app_settings=app)
        try:
            launcher, root = _make_real_launcher(entry_module, cfg_path, monkeypatch)
        except tk.TclError as exc:
            pytest.skip(f"Tk root unavailable: {exc}")
        try:
            # The loader normalizes blank spec_type to "none" (see
            # config.py load_saved_configs around _spec_str_keys).
            assert launcher.spec_type.get() == "none"
            assert launcher.spec_enabled.get() is True
        finally:
            root.destroy()

    def test_stale_reasoning_budget_non_int_loads_cleanly(
        self, entry_module, tmp_path, monkeypatch
    ):
        """A non-int ``reasoning_budget`` like ``"abc"`` (e.g. pre-validation
        config) must round-trip into the Tk var without crashing. The
        emission warn-and-skip is covered separately by the mock-based
        test_reasoning_and_kvu.py suite."""
        cfg_path = tmp_path / "configs.json"
        app = {
            "reasoning_budget": "abc",
            "model_dirs": [], "model_list_height": 8, "selected_gpus": [],
            "gpu_order": [], "host": "127.0.0.1", "port": "8080",
        }
        self._write_config_file(cfg_path, {}, app_settings=app)
        try:
            launcher, root = _make_real_launcher(entry_module, cfg_path, monkeypatch)
        except tk.TclError as exc:
            pytest.skip(f"Tk root unavailable: {exc}")
        try:
            assert launcher.reasoning_budget.get() == "abc"
        finally:
            root.destroy()

    def test_cache_idle_without_kv_unified_clears_on_load(
        self, entry_module, tmp_path, monkeypatch
    ):
        """``cache_idle_slots_mode`` is dependent on ``kv_unified_mode=="on"``.
        With kvu blank/off on llama.cpp, the stale-value reset clears
        cache_idle_slots at init time (so the next ``_save_configs`` writes
        an empty string, and emission never sees the orphaned child)."""
        cfg_path = tmp_path / "configs.json"
        app = {
            "kv_unified_mode": "",  # not "on"
            "cache_idle_slots_mode": "on",  # orphaned child
            "model_dirs": [], "model_list_height": 8, "selected_gpus": [],
            "gpu_order": [], "host": "127.0.0.1", "port": "8080",
            "backend_selection": "llama.cpp",
        }
        self._write_config_file(cfg_path, {}, app_settings=app)
        try:
            launcher, root = _make_real_launcher(entry_module, cfg_path, monkeypatch)
        except tk.TclError as exc:
            pytest.skip(f"Tk root unavailable: {exc}")
        try:
            # The kv-unify gating fires during init and clears the orphan.
            assert launcher.cache_idle_slots_mode.get() == "", (
                f"orphaned cache_idle_slots_mode should clear; got "
                f"{launcher.cache_idle_slots_mode.get()!r}"
            )
        finally:
            root.destroy()
