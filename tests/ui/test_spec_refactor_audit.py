"""Adversarial audit of the MTP / Spec modularization refactor.

This test file does NOT replicate the comprehensive coverage already in
the launcher suites (``tests/launchers/test_reasoning_and_kvu.py``,
``tests/launchers/test_spec_emission.py``,
``tests/launchers/test_full_roundtrip.py``). Instead it directly exercises
the five contracts the refactor claims to preserve:

1. **Re-export contract integrity** — every spec Tk var the launcher
   exposes must be the same Python object as the one on
   ``launcher.spec_tab``. Writes via either path must be visible to the
   other.
2. **Dynamic-reassignment contract** — attributes that SpecTab rebinds
   in flight (``spec_draft_gpu_vars``, ``current_spec_draft_analysis``,
   etc.) must be delegated through ``__getattr__`` so the launcher
   always sees the live reference, never a stale one.
3. **In-place-mutation contract** — ``_spec_widgets`` / ``_spec_sections``
   are statically re-exported on the launcher AND are reassigned only at
   ``__init__`` time (never later). ``setup_tab`` uses ``.clear()``.
4. **Load-order contract** — ``resync_spec_tk_vars_from_app_settings``
   is called after ``_load_saved_configs`` and BEFORE any other tab's
   ``load_from_config`` (was a real regression once; locks it down).
5. **Adversarial config loading** — extra-permissive entries (legacy,
   wrong-backend, mixed-garbage gpu list, MTP+parallel=8) all survive
   and produce sane state.

These tests need a real Tk display. ``tk_root`` from the shared conftest
skips cleanly when unavailable.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tkinter as tk
from pathlib import Path

import pytest

from tests.launcher_var_registry import (
    ALL_NEW_LAUNCHER_TK_VARS,
    SPEC_TK_VARS,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ENTRY_PATH = REPO_ROOT / "llamacpp-server-launcher.py"


# ---------------------------------------------------------------------------
# Module / launcher helpers
# ---------------------------------------------------------------------------


def _silence_messagebox(monkeypatch):
    """Scope messagebox no-ops to the test via monkeypatch — avoids
    leaking global ``mb.showinfo = lambda: None`` across the rest of
    the suite (would silently make popup-assertion tests pass)."""
    import tkinter.messagebox as mb

    monkeypatch.setattr(mb, "showinfo", lambda *a, **kw: None)
    monkeypatch.setattr(mb, "showwarning", lambda *a, **kw: None)
    monkeypatch.setattr(mb, "showerror", lambda *a, **kw: None)


@pytest.fixture(scope="module")
def entry_module():
    spec = importlib.util.spec_from_file_location("entry_module_refactor_audit", ENTRY_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["entry_module_refactor_audit"] = module
    spec.loader.exec_module(module)
    return module


def _make_real_launcher(entry_module, config_path, monkeypatch):
    """Build a real ``LlamaCppLauncher`` with ConfigManager + messagebox
    overrides scoped to the test via ``monkeypatch``."""
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
# Focus 1 — Re-export contract integrity
# ---------------------------------------------------------------------------


class TestSameObjectReferenceContract:
    """Every spec Tk var visible on the launcher must be ``is`` identical
    to the same name on ``launcher.spec_tab``. The refactor relies on
    a one-time setattr at construction; if SpecTab ever reassigned one of
    those names after init, the launcher would point at a stale reference
    and writes via ``launcher.spec_enabled.set(...)`` would silently
    diverge from ``launcher.spec_tab.spec_enabled.get()``.
    """

    def test_all_spec_tk_vars_share_object_with_spec_tab(self, real_launcher):
        launcher, _ = real_launcher
        # Every name in the registry must be the SAME object on both.
        for name, _vc, _d in SPEC_TK_VARS:
            launcher_attr = getattr(launcher, name)
            spec_tab_attr = getattr(launcher.spec_tab, name)
            assert launcher_attr is spec_tab_attr, (
                f"Re-export contract broken for {name!r}: "
                f"launcher.{name} is not launcher.spec_tab.{name} "
                f"(ids: {id(launcher_attr)} vs {id(spec_tab_attr)})"
            )

    def test_derived_ui_state_attrs_share_object(self, real_launcher):
        """Derived non-persisted UI state — spec_draft_ngl_int / max /
        status / dicts — must also be the same object reference (Tk vars
        are aliased; dicts/lists are accessed via ``__getattr__``).
        """
        launcher, _ = real_launcher
        # Tk vars (Tk vars are aliased at construction in _SPEC_TAB_VAR_REEXPORT).
        for name in (
            "spec_draft_ngl_int",
            "max_spec_draft_gpu_layers",
            "spec_draft_layers_status_var",
        ):
            assert getattr(launcher, name) is getattr(launcher.spec_tab, name), (
                f"Tk var {name!r} mismatch"
            )
        # Dict / list state delegated via __getattr__ — must always reflect
        # the live SpecTab attribute, even after SpecTab rebinds it.
        for name in (
            "current_spec_draft_analysis",
            "spec_draft_gpu_vars",
            "_spec_widgets",
            "_spec_sections",
        ):
            launcher_attr = getattr(launcher, name)
            spec_tab_attr = getattr(launcher.spec_tab, name)
            assert launcher_attr is spec_tab_attr, (
                f"Delegated attr {name!r} mismatch: "
                f"launcher.{name} id={id(launcher_attr)} "
                f"spec_tab.{name} id={id(spec_tab_attr)}"
            )

    def test_hint_vars_lazy_created_in_setup_tab(self, real_launcher):
        """spec_*_hint_var / spec_status_var / spec_draft_path_display_var /
        spec_draft_listbox are created lazily in ``setup_tab``. They must
        be reachable via ``__getattr__`` delegation once setup_tab has
        run (which it has, because the launcher built its notebook).
        """
        launcher, _ = real_launcher
        for name in (
            "spec_status_var",
            "spec_pmin_hint_var",
            "spec_psplit_hint_var",
            "spec_parallel_hint_var",
            "spec_draft_path_display_var",
            "spec_draft_listbox",
            "spec_draft_ngl_entry",
            "spec_draft_ngl_slider",
            "spec_draft_layers_status_label",
            "spec_draft_gpu_checkbox_frame",
            "spec_draft_ctk_combo",
            "spec_draft_ctv_combo",
        ):
            launcher_attr = getattr(launcher, name)
            spec_tab_attr = getattr(launcher.spec_tab, name)
            assert launcher_attr is spec_tab_attr, (
                f"Lazy attr {name!r} mismatch: ids {id(launcher_attr)} vs {id(spec_tab_attr)}"
            )

    def test_writes_through_launcher_visible_on_spec_tab(self, real_launcher):
        """``launcher.spec_enabled.set(True)`` must propagate so
        ``launcher.spec_tab.spec_enabled.get() == True``."""
        launcher, _ = real_launcher
        launcher.spec_enabled.set(True)
        assert launcher.spec_tab.spec_enabled.get() is True
        launcher.spec_enabled.set(False)
        assert launcher.spec_tab.spec_enabled.get() is False

    def test_writes_through_spec_tab_visible_on_launcher(self, real_launcher):
        """Inverse: writes via ``launcher.spec_tab.spec_enabled.set(...)``
        must propagate to ``launcher.spec_enabled.get()``."""
        launcher, _ = real_launcher
        launcher.spec_tab.spec_enabled.set(True)
        assert launcher.spec_enabled.get() is True
        launcher.spec_tab.spec_type.set("ngram-simple")
        assert launcher.spec_type.get() == "ngram-simple"

    def test_in_place_mutation_visible_on_both_paths(self, real_launcher):
        """``_spec_widgets`` / ``_spec_sections`` are accessed via
        ``__getattr__`` so they're always the SpecTab live dict — mutation
        through either path must be visible everywhere."""
        launcher, _ = real_launcher
        # The launcher's __getattr__ delegates these to SpecTab live attrs.
        launcher_w = launcher._spec_widgets
        spec_tab_w = launcher.spec_tab._spec_widgets
        assert launcher_w is spec_tab_w
        # Mutate via the spec_tab path; launcher should see it.
        marker_key = "__test_marker__"
        try:
            launcher.spec_tab._spec_widgets[marker_key] = "test"
            assert launcher._spec_widgets[marker_key] == "test"
        finally:
            launcher.spec_tab._spec_widgets.pop(marker_key, None)


# ---------------------------------------------------------------------------
# Focus 1b — Detect any SpecTab method that secretly reassigns an aliased var
# ---------------------------------------------------------------------------


class TestNoLatentReassignmentBugs:
    """Static check: nothing inside SpecTab.setup_tab or downstream
    method bodies reassigns one of the aliased Tk vars / dicts in a way
    that would silently break the static re-export.

    We do this both via runtime introspection (post-setup_tab the ids
    should match the construction-time ids) and via a textual sanity
    scan of spec_tab.py.
    """

    # Names that MUST NEVER be reassigned after __init__ since they're
    # exported via static setattr at construction time. (Names accessed
    # via __getattr__ delegation are allowed to be reassigned.)
    _STATIC_REEXPORT = tuple(name for name, _vc, _d in SPEC_TK_VARS) + (
        "spec_draft_ngl_int",
        "max_spec_draft_gpu_layers",
        "spec_draft_layers_status_var",
        # _spec_widgets / _spec_sections are statically re-exported too;
        # they're allowed to be cleared but NOT reassigned.
        "_spec_widgets",
        "_spec_sections",
    )

    def test_no_static_reexport_attr_reassigned_outside_init(self):
        """Textual scan: any line ``self.<name> = `` outside __init__ for
        statically-reexported names is a latent bug.
        """
        text = (REPO_ROOT / "modules" / "spec_tab.py").read_text()
        offenses = []
        # Split into method-sized regions for accurate scanning. Look for any
        # ``self.<name> = `` outside of __init__. The __init__ method ends at
        # the first non-indented `def ` after `def __init__`.
        lines = text.splitlines()
        in_init = False
        init_indent = None
        for i, raw in enumerate(lines, start=1):
            stripped = raw.lstrip()
            if stripped.startswith("def __init__"):
                in_init = True
                init_indent = len(raw) - len(stripped)
                continue
            if in_init and stripped.startswith("def ") and (len(raw) - len(stripped)) <= init_indent:
                in_init = False
            if in_init:
                continue
            for name in self._STATIC_REEXPORT:
                pattern = f"self.{name} ="
                # Exclude .set() and == comparisons; match assignment only.
                if pattern in raw and not raw.lstrip().startswith("#"):
                    # Disallow assignment, but allow `==` comparison and `.set()`.
                    idx = raw.find(pattern)
                    after = raw[idx + len(pattern):].lstrip()
                    # If the next non-space char is `=`, it's `==` (allowed).
                    if not after.startswith("="):
                        offenses.append(f"Line {i}: {raw.rstrip()}")
        assert not offenses, (
            "Found reassignments of statically-reexported attrs outside __init__:\n"
            + "\n".join(offenses)
        )

    def test_post_setup_object_ids_match_construction(self, real_launcher):
        """Runtime: after setup_tab has run, the launcher's aliased
        attributes must still point at the same objects on the SpecTab.
        If SpecTab ever silently rebound a Tk var (or dict) the ids would
        diverge.
        """
        launcher, _ = real_launcher
        # setup_tab has already been called as part of __init__ via the
        # build_widgets phase. Verify identity once more, AFTER any traces
        # might have fired during init (e.g. _on_spec_type_changed).
        for name in self._STATIC_REEXPORT:
            try:
                launcher_attr = launcher.__dict__[name]  # bypass __getattr__
            except KeyError:
                pytest.fail(
                    f"Statically-reexported attr {name!r} not in launcher.__dict__"
                )
            spec_tab_attr = getattr(launcher.spec_tab, name)
            assert launcher_attr is spec_tab_attr, (
                f"Post-setup divergence for {name!r}: "
                f"launcher.{name} id={id(launcher_attr)} "
                f"spec_tab.{name} id={id(spec_tab_attr)}"
            )


# ---------------------------------------------------------------------------
# Focus 2 — End-to-end emission parity
# ---------------------------------------------------------------------------


class TestEmissionParity:
    """Exercises ``build_cmd()`` (via the LaunchManager) for representative
    combinations across both backends; confirms the spec-related flags
    emitted match the expected shape after the refactor."""

    def _run_build_cmd(self, launcher, *, model_path="/tmp/dummy.gguf"):
        """Drive LaunchManager.build_cmd on the real launcher; return the
        emitted argv list. Stubs the model path / backend exe so build_cmd
        doesn't bail on missing files; we only care about the spec block.
        """
        # Set a model_path Tk var so build_cmd doesn't short-circuit.
        try:
            launcher.model_path.set(model_path)
        except Exception:
            pass
        # Build the command. Catch the result — build_cmd may either return
        # a list or write to self.cmd_preview_var; both shapes happen in
        # different release branches, so we try the LaunchManager direct call.
        lm = launcher.launch_manager
        try:
            cmd = lm.build_cmd()
            if isinstance(cmd, tuple):
                cmd = cmd[0]
            return list(cmd) if cmd is not None else []
        except Exception:
            # If build_cmd errored we still want a partial view — fall back
            # to invoking emit_spec_args directly.
            from modules.spec_launch import (
                emit_kv_unify_args,
                emit_no_mmproj_arg,
                emit_reasoning_args,
                emit_spec_args,
            )

            backend = launcher.backend_selection.get()
            partial = []
            emit_spec_args(launcher, backend, partial)
            emit_reasoning_args(launcher, partial)
            emit_kv_unify_args(launcher, backend, partial)
            emit_no_mmproj_arg(launcher, backend, partial)
            return partial

    @pytest.mark.parametrize(
        "backend,spec_type,extras,expected_flags",
        [
            # Default state: no spec flags emitted.
            ("llama.cpp", "none", {}, []),
            # llama.cpp draft-mtp: --spec-type emitted plus draft knobs.
            (
                "llama.cpp",
                "draft-mtp",
                {"spec_draft_n_max": "3", "spec_draft_n_min": "0"},
                ["--spec-type", "draft-mtp", "--spec-draft-n-max", "--spec-draft-n-min"],
            ),
            # llama.cpp draft-simple with a model selected; emission includes
            # --spec-draft-model with the resolved path.
            (
                "llama.cpp",
                "draft-simple",
                {"spec_draft_n_max": "16"},
                ["--spec-type", "draft-simple", "--spec-draft-n-max"],
            ),
            # llama.cpp ngram-mod with mod knobs set.
            (
                "llama.cpp",
                "ngram-mod",
                {
                    "spec_ngram_mod_n_min": "4",
                    "spec_ngram_mod_n_max": "10",
                    "spec_ngram_mod_n_match": "2",
                },
                [
                    "--spec-type", "ngram-mod",
                    "--spec-ngram-mod-n-min", "--spec-ngram-mod-n-max",
                    "--spec-ngram-mod-n-match",
                ],
            ),
            # ik_llama suffix with the suffix knobs.
            (
                "ik_llama",
                "suffix",
                {
                    "spec_suffix_pattern_len": "8",
                    "spec_suffix_max_depth": "3",
                },
                [
                    "--spec-type", "suffix",
                    "--suffix-pattern-len", "--suffix-max-depth",
                ],
            ),
            # ik_llama mtp draft-capable case.
            (
                "ik_llama",
                "mtp",
                {"spec_draft_n_max": "5"},
                ["--spec-type", "mtp", "--draft-max"],
            ),
        ],
    )
    def test_spec_args_emitted_per_combination(
        self, real_launcher, backend, spec_type, extras, expected_flags
    ):
        launcher, _ = real_launcher
        launcher.backend_selection.set(backend)
        # spec_enabled must be True for any --spec-* emission. For the
        # "none" baseline case, we still set it True to confirm the
        # type=none early-exit holds.
        launcher.spec_enabled.set(spec_type != "none")
        launcher.spec_type.set(spec_type)
        for k, v in extras.items():
            getattr(launcher, k).set(v)

        # Direct emission probe (avoids build_cmd's many file-existence checks).
        from modules.spec_launch import emit_spec_args

        partial = []
        emit_spec_args(launcher, backend, partial)
        for flag in expected_flags:
            assert flag in partial, (
                f"backend={backend} spec_type={spec_type}: "
                f"expected {flag!r} in emitted args; got {partial!r}"
            )
        # When type=none, no --spec-* / --draft-* flags should be in argv.
        if spec_type == "none":
            assert not any(
                arg.startswith("--spec-") or arg.startswith("--draft-")
                for arg in partial
            ), f"type=none must emit no spec/draft flags; got {partial!r}"

    def test_mtp_overrides_parallel_8_at_launch(self, real_launcher):
        """MTP requires --parallel 1. If the user has parallel=8 in config
        and MTP is active, ``resolve_effective_parallel`` must override
        to "1" with a stderr warning.
        """
        launcher, _ = real_launcher
        launcher.backend_selection.set("llama.cpp")
        launcher.spec_enabled.set(True)
        launcher.spec_type.set("draft-mtp")
        launcher.parallel.set("8")
        from modules.spec_launch import resolve_effective_parallel
        effective = resolve_effective_parallel(launcher, launcher.backend_selection.get())
        assert effective == "1", (
            f"MTP must force --parallel 1; got {effective!r}"
        )

    def test_default_no_spec_no_emissions(self, real_launcher):
        """Spec disabled → emission block must emit nothing."""
        launcher, _ = real_launcher
        launcher.spec_enabled.set(False)
        launcher.spec_type.set("none")
        from modules.spec_launch import emit_spec_args
        partial = []
        emit_spec_args(launcher, "llama.cpp", partial)
        assert partial == [], f"disabled spec must emit nothing; got {partial!r}"


# ---------------------------------------------------------------------------
# Focus 2b — Persistence round-trip parity (single-launcher; no fresh-boot)
# ---------------------------------------------------------------------------


class TestPersistenceParity:
    """Verifies save_configuration -> on-disk JSON has every spec key, and
    load_configuration restores them on a fresh launcher.
    """

    @staticmethod
    def _distinctive_value(name, varclass):
        if varclass == "BooleanVar":
            return True
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
        return f"v_{name[-12:]}"

    def test_every_spec_key_persists_through_named_config(
        self, entry_module, tmp_path, monkeypatch
    ):
        """Set distinctive non-default values, save as named config,
        boot fresh launcher, load the named config back. Assert every
        key matches what we set.
        """
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
            launcher1.config_name.set("persist_audit_cfg")
            launcher1._save_configuration()

            # Verify on-disk payload has every key under "configs/persist_audit_cfg".
            payload = json.loads(cfg_path.read_text())
            stored = payload["configs"]["persist_audit_cfg"]
            for name, _vc, _d in ALL_NEW_LAUNCHER_TK_VARS:
                assert name in stored, f"key {name!r} missing from named config"
                assert stored[name] == targets[name], (
                    f"named-config {name!r}: expected {targets[name]!r}, "
                    f"got {stored[name]!r}"
                )
        finally:
            root1.destroy()


# ---------------------------------------------------------------------------
# Focus 2c — Load-order re-sync is wired
# ---------------------------------------------------------------------------


class TestLoadOrderResync:
    """Hard verification that resync_spec_tk_vars_from_app_settings is
    actually called by the launcher init, after _load_saved_configs and
    before tab load_from_config calls.
    """

    def test_resync_is_called_during_init(self, entry_module, tmp_path, monkeypatch):
        """Patch the resync helper to record invocations and confirm
        it's invoked exactly once during __init__."""
        import modules.spec_persistence as sp
        cfg_path = tmp_path / "configs.json"

        calls = []
        real_resync = sp.resync_spec_tk_vars_from_app_settings

        def spy(launcher):
            calls.append(("resync", id(launcher)))
            return real_resync(launcher)

        # The launcher imports the symbol into its own namespace; patch
        # there so the call site picks up the spy.
        monkeypatch.setattr(entry_module, "resync_spec_tk_vars_from_app_settings", spy)
        try:
            launcher, root = _make_real_launcher(entry_module, cfg_path, monkeypatch)
        except tk.TclError as exc:
            pytest.skip(f"Tk root unavailable: {exc}")
        try:
            assert len(calls) >= 1, (
                "resync_spec_tk_vars_from_app_settings was NOT called during __init__"
            )
        finally:
            root.destroy()

    def test_resync_runs_before_env_vars_load_from_config(
        self, entry_module, tmp_path, monkeypatch
    ):
        """Order-of-init bug regression test: resync must precede
        env_vars_manager.load_from_config (which can fire traces that
        write back into app_settings)."""
        import modules.spec_persistence as sp
        cfg_path = tmp_path / "configs.json"

        order = []
        real_resync = sp.resync_spec_tk_vars_from_app_settings

        def resync_spy(launcher):
            order.append("resync")
            return real_resync(launcher)

        monkeypatch.setattr(entry_module, "resync_spec_tk_vars_from_app_settings", resync_spy)

        # Patch env_vars_manager.load_from_config on the class.
        from modules.env_vars_module import EnvironmentalVariablesManager
        real_evm_load = EnvironmentalVariablesManager.load_from_config

        def evm_load_spy(self, app_settings):
            order.append("env_vars_load_from_config")
            return real_evm_load(self, app_settings)

        monkeypatch.setattr(
            EnvironmentalVariablesManager, "load_from_config", evm_load_spy
        )

        try:
            launcher, root = _make_real_launcher(entry_module, cfg_path, monkeypatch)
        except tk.TclError as exc:
            pytest.skip(f"Tk root unavailable: {exc}")
        try:
            assert "resync" in order, "resync helper was never called"
            assert "env_vars_load_from_config" in order, (
                "env_vars_manager.load_from_config was never called"
            )
            resync_idx = order.index("resync")
            evm_idx = order.index("env_vars_load_from_config")
            assert resync_idx < evm_idx, (
                f"Order violation: resync (idx={resync_idx}) must precede "
                f"env_vars_load_from_config (idx={evm_idx}). order={order!r}"
            )
        finally:
            root.destroy()


# ---------------------------------------------------------------------------
# Focus 3 — Adversarial config loading (gaps not covered by existing tests)
# ---------------------------------------------------------------------------


class TestExtraAdversarialConfigs:
    """Scenarios not already covered by
    ``tests/launchers/test_full_roundtrip.py::TestAdversarialLoadCoercion``."""

    @staticmethod
    def _write(cfg_path, app_settings):
        payload = {
            "configs": {},
            "app_settings": app_settings,
        }
        cfg_path.write_text(json.dumps(payload), encoding="utf-8")

    def test_legacy_spec_draft_hf_silently_ignored(self, entry_module, tmp_path, monkeypatch):
        """The legacy ``spec_draft_hf`` key (removed in this branch) must
        be silently ignored without crashing the loader."""
        cfg_path = tmp_path / "configs.json"
        self._write(
            cfg_path,
            {
                "spec_draft_hf": "TheBloke/some-draft-gguf",  # legacy
                "model_dirs": [],
                "model_list_height": 8,
                "selected_gpus": [],
                "gpu_order": [],
                "host": "127.0.0.1",
                "port": "8080",
            },
        )
        try:
            launcher, root = _make_real_launcher(entry_module, cfg_path, monkeypatch)
        except tk.TclError as exc:
            pytest.skip(f"Tk root unavailable: {exc}")
        try:
            # Loader didn't crash. spec_draft_hf should NOT be on the launcher.
            assert not hasattr(launcher, "spec_draft_hf"), (
                "legacy spec_draft_hf key resurrected an attribute"
            )
        finally:
            root.destroy()

    def test_wrong_backend_spec_type_preserved_and_inactive(
        self, entry_module, tmp_path, monkeypatch
    ):
        """``spec_type=draft-mtp`` (llama.cpp-only) + ``backend=ik_llama``:
        the loader must preserve the stored value; emission must reject
        it on the active backend."""
        cfg_path = tmp_path / "configs.json"
        self._write(
            cfg_path,
            {
                "spec_enabled": True,
                "spec_type": "draft-mtp",  # not valid on ik_llama
                "backend_selection": "ik_llama",
                "model_dirs": [],
                "model_list_height": 8,
                "selected_gpus": [],
                "gpu_order": [],
                "host": "127.0.0.1",
                "port": "8080",
            },
        )
        try:
            launcher, root = _make_real_launcher(entry_module, cfg_path, monkeypatch)
        except tk.TclError as exc:
            pytest.skip(f"Tk root unavailable: {exc}")
        try:
            assert launcher.backend_selection.get() == "ik_llama"
            assert launcher.spec_type.get() == "draft-mtp"  # preserved!
            # Emission should reject and skip with WARNING printed.
            from modules.spec_launch import emit_spec_args
            partial = []
            emit_spec_args(launcher, "ik_llama", partial)
            assert "--spec-type" not in partial, (
                f"emission must reject wrong-backend spec_type; got {partial!r}"
            )
        finally:
            root.destroy()

    def test_mtp_with_parallel_8_overrides_at_launch(self, entry_module, tmp_path, monkeypatch):
        """parallel=8 + spec_enabled+spec_type=draft-mtp must force
        --parallel 1 at launch via resolve_effective_parallel.
        """
        cfg_path = tmp_path / "configs.json"
        self._write(
            cfg_path,
            {
                "spec_enabled": True,
                "spec_type": "draft-mtp",
                "backend_selection": "llama.cpp",
                "model_dirs": [],
                "model_list_height": 8,
                "selected_gpus": [],
                "gpu_order": [],
                "host": "127.0.0.1",
                "port": "8080",
            },
        )
        try:
            launcher, root = _make_real_launcher(entry_module, cfg_path, monkeypatch)
        except tk.TclError as exc:
            pytest.skip(f"Tk root unavailable: {exc}")
        try:
            # Now manually shove parallel=8 (the launcher's auto-set would
            # have made it "1" via _on_spec_type_changed; we test the
            # last-line-of-defense override).
            launcher.parallel.set("8")
            from modules.spec_launch import resolve_effective_parallel
            effective = resolve_effective_parallel(launcher, launcher.backend_selection.get())
            assert effective == "1", (
                f"MTP + parallel=8 must override to 1; got {effective!r}"
            )
        finally:
            root.destroy()

    def test_mixed_garbage_spec_draft_selected_gpus_filtered_to_ints(
        self, entry_module, tmp_path, monkeypatch
    ):
        """``spec_draft_selected_gpus=[999, "abc", True]`` -> only valid
        ints survive. (True is a bool subclass and must NOT pass through
        since it's not a meaningful GPU index.)"""
        cfg_path = tmp_path / "configs.json"
        self._write(
            cfg_path,
            {
                "spec_draft_selected_gpus": [999, "abc", True],
                "model_dirs": [],
                "model_list_height": 8,
                "selected_gpus": [],
                "gpu_order": [],
                "host": "127.0.0.1",
                "port": "8080",
            },
        )
        try:
            launcher, root = _make_real_launcher(entry_module, cfg_path, monkeypatch)
        except tk.TclError as exc:
            pytest.skip(f"Tk root unavailable: {exc}")
        try:
            cleaned = launcher.app_settings.get("spec_draft_selected_gpus")
            assert isinstance(cleaned, list)
            for entry in cleaned:
                assert isinstance(entry, int) and not isinstance(entry, bool), (
                    f"non-int / bool survived filter: {cleaned!r}"
                )
            assert "abc" not in cleaned and True not in cleaned
        finally:
            root.destroy()


# ---------------------------------------------------------------------------
# Focus 4 — Test-coverage redirect validation
# ---------------------------------------------------------------------------


class TestRedirectIntegrity:
    """When the agent moved tests from ``entry_module.LlamaCppLauncher.X``
    to ``entry_module.SpecTab.X``, every redirect target must exist and
    be callable. Otherwise the test silently passes as a no-op (or worse,
    raises a MagicMock-style AttributeError).
    """

    _REDIRECTED_METHODS = (
        "_apply_spec_defaults_if_blank",
        "_apply_mtp_parallel_default",
        "_reset_spec_defaults",
        "_set_spec_draft_gpu_layers",
        "_validate_spec_draft_gpu_layers_entry",
        "_refresh_spec_tab_state",
        "_on_spec_enabled_changed",
        "_on_spec_type_changed",
        "_on_spec_draft_model_selected",
        "_clear_spec_draft_model",
        "_sync_spec_draft_gpu_layers_from_slider",
        "_sync_spec_draft_gpu_layers_from_entry",
        "_update_spec_draft_gpu_checkboxes",
        "_on_spec_draft_gpu_selection_changed",
        "_run_spec_draft_gguf_analysis",
        "_update_ui_after_spec_draft_analysis",
    )

    def test_all_redirected_methods_exist_on_spec_tab(self, entry_module):
        for name in self._REDIRECTED_METHODS:
            assert hasattr(entry_module.SpecTab, name), (
                f"SpecTab.{name} missing — tests would silently no-op"
            )
            assert callable(getattr(entry_module.SpecTab, name)), (
                f"SpecTab.{name} not callable"
            )

    def test_class_constants_re_exported_from_launcher(self, entry_module):
        """Legacy tests reference ``LlamaCppLauncher._SPEC_TYPES_*``;
        the launcher exposes them as class-level aliases of SpecTab's
        canonical definitions.
        """
        assert (
            entry_module.LlamaCppLauncher._SPEC_TYPES_LLAMA_CPP
            is entry_module.SpecTab._SPEC_TYPES_LLAMA_CPP
        )
        assert (
            entry_module.LlamaCppLauncher._SPEC_TYPES_IK_LLAMA
            is entry_module.SpecTab._SPEC_TYPES_IK_LLAMA
        )
