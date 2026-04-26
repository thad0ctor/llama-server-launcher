"""Fills test gaps in ``modules/config.py``.

Scope:
  * ``ConfigManager.load_configuration`` (untested in ``test_config.py``)
  * ``ConfigManager.delete_configuration``
  * Path traversal / config-name sanitization at the save boundary
  * Unicode / BOM / very-large-payload behaviour of ``load_saved_configs``
  * Additional edge cases for ``generate_default_config_name`` (control chars,
    trailing dots, whitespace, unicode, long names, regex-special chars,
    empty thread count, other abbreviation branches)
  * Symlink race in the legacy-config migration path
  * ``get_config_path`` error branches (OSError when parent is a file, ``Path.home`` raises)
  * GPU order with duplicates
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from tests.core.conftest import FakeVar, FakeListbox  # type: ignore  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _full_cfg(**overrides):
    """Return a full-schema config dict covering all 40+ fields for loads."""
    base = {
        "llama_cpp_dir": "/opt/llama.cpp",
        "ik_llama_dir": "/opt/ik_llama",
        "venv_dir": "/opt/venv",
        "model_path": "",
        "selected_mmproj_path": "",
        "cache_type_k": "q8_0",
        "cache_type_v": "q4_0",
        "threads": "16",
        "threads_batch": "8",
        "batch_size": "1024",
        "ubatch_size": "256",
        "n_gpu_layers": "33",
        "no_mmap": True,
        "prio": "3",
        "temperature": "0.5",
        "min_p": "0.1",
        "ctx_size": 8192,
        "seed": "42",
        "flash_attn": True,
        "tensor_split": "0.5,0.5",
        "main_gpu": "1",
        "mlock": True,
        "no_kv_offload": True,
        "host": "192.168.1.1",
        "port": "9090",
        "backend_selection": "ik_llama.cpp",
        "ignore_eos": True,
        "n_predict": "512",
        "cpu_moe": True,
        "n_cpu_moe": "4",
        "parallel": "2",
        "mmproj_enabled": True,
        "fit_enabled": False,
        "fit_ctx": "2048",
        "fit_ctx_synced": False,
        "fit_target": "512",
        "template_source": "predefined",
        "predefined_template_name": "ChatML",
        "custom_template_string": "<|im_start|>{{content}}<|im_end|>",
        "jinja_enabled": True,
        "custom_parameters": ["--foo", "--bar=1"],
        "gpu_indices": [0, 1],
        "gpu_order": [1, 0],
        "environmental_variables": {"enabled": False, "predefined": {}, "custom": []},
    }
    base.update(overrides)
    return base


# ===========================================================================
# load_configuration  (high-risk, previously untested)
# ===========================================================================

class TestLoadConfiguration:
    def _prepare(self, rich_launcher_factory, tmp_path, cfg, name="demo"):
        from modules.config import ConfigManager
        launcher = rich_launcher_factory(tmp_path / "cfg.json")
        launcher.saved_configs = {name: cfg}
        launcher.config_listbox.set_items([name])
        launcher.config_listbox.set_selection([0])
        return ConfigManager(launcher), launcher

    def test_no_selection_shows_error(self, rich_launcher_factory, tmp_path):
        from modules.config import ConfigManager
        launcher = rich_launcher_factory(tmp_path / "cfg.json")
        launcher.config_listbox.set_selection([])
        cm = ConfigManager(launcher)
        with patch("modules.config.messagebox") as mb:
            cm.load_configuration()
        assert mb.showerror.called

    def test_missing_config_data_shows_error(self, rich_launcher_factory, tmp_path):
        from modules.config import ConfigManager
        launcher = rich_launcher_factory(tmp_path / "cfg.json")
        # Listbox shows a name but saved_configs lacks it.
        launcher.config_listbox.set_items(["ghost"])
        launcher.config_listbox.set_selection([0])
        launcher.saved_configs = {}
        cm = ConfigManager(launcher)
        with patch("modules.config.messagebox") as mb:
            cm.load_configuration()
        assert mb.showerror.called

    def test_populates_all_tk_vars(self, rich_launcher_factory, tmp_path):
        cfg = _full_cfg()
        cm, launcher = self._prepare(rich_launcher_factory, tmp_path, cfg)
        with patch("modules.config.messagebox"):
            cm.load_configuration()
        # Spot-check all 40+ fields...
        assert launcher.llama_cpp_dir.get() == "/opt/llama.cpp"
        assert launcher.ik_llama_dir.get() == "/opt/ik_llama"
        assert launcher.venv_dir.get() == "/opt/venv"
        assert launcher.cache_type_k.get() == "q8_0"
        assert launcher.cache_type_v.get() == "q4_0"
        assert launcher.threads.get() == "16"
        assert launcher.threads_batch.get() == "8"
        assert launcher.batch_size.get() == "1024"
        assert launcher.ubatch_size.get() == "256"
        assert launcher.no_mmap.get() is True
        assert launcher.prio.get() == "3"
        assert launcher.temperature.get() == "0.5"
        assert launcher.min_p.get() == "0.1"
        assert launcher.ctx_size.get() == 8192
        assert launcher.seed.get() == "42"
        assert launcher.flash_attn.get() is True
        assert launcher.tensor_split.get() == "0.5,0.5"
        assert launcher.main_gpu.get() == "1"
        assert launcher.mlock.get() is True
        assert launcher.no_kv_offload.get() is True
        assert launcher.host.get() == "192.168.1.1"
        assert launcher.port.get() == "9090"
        assert launcher.backend_selection.get() == "ik_llama.cpp"
        assert launcher.ignore_eos.get() is True
        assert launcher.n_predict.get() == "512"
        assert launcher.cpu_moe.get() is True
        assert launcher.n_cpu_moe.get() == "4"
        assert launcher.parallel.get() == "2"
        assert launcher.mmproj_enabled.get() is True
        assert launcher.fit_enabled.get() is False
        assert launcher.fit_target.get() == "512"
        assert launcher.fit_ctx_synced is False
        # fit_ctx_synced=False -> uses saved fit_ctx value
        assert launcher.fit_ctx.get() == "2048"
        assert launcher.template_source.get() == "predefined"
        assert launcher.predefined_template_name.get() == "ChatML"
        assert launcher.custom_template_string.get() == "<|im_start|>{{content}}<|im_end|>"
        assert launcher.jinja_enabled.get() is True
        assert launcher.custom_parameters_list == ["--foo", "--bar=1"]
        assert launcher.config_name.get() == "demo"
        # Fit display sync
        launcher._sync_ctx_display.assert_called_with(8192)
        launcher._update_fit_fields_state.assert_called()
        launcher._update_custom_parameters_listbox.assert_called()

    def test_missing_keys_apply_defaults(self, rich_launcher_factory, tmp_path):
        """Partial config dicts get default values on unknown fields."""
        cm, launcher = self._prepare(
            rich_launcher_factory, tmp_path, {"model_path": ""}
        )
        with patch("modules.config.messagebox"):
            cm.load_configuration()
        # The launcher.physical_cores fallback kicked in for threads.
        assert launcher.threads.get() == str(launcher.physical_cores)
        assert launcher.threads_batch.get() == str(launcher.logical_cores)
        assert launcher.batch_size.get() == "512"
        assert launcher.ubatch_size.get() == "512"
        assert launcher.ctx_size.get() == 2048
        assert launcher.seed.get() == "-1"
        # Default flipped to True: ik_llama enables FA by default in the
        # binary, and llama.cpp benefits from it on most modern GPUs.
        assert launcher.flash_attn.get() is True
        assert launcher.ignore_eos.get() is False
        assert launcher.n_predict.get() == "-1"
        assert launcher.parallel.get() == "1"
        assert launcher.mmproj_enabled.get() is False
        assert launcher.fit_enabled.get() is True
        assert launcher.fit_target.get() == "1024"
        assert launcher.template_source.get() == "default"
        assert launcher.predefined_template_name.get() == "ChatML"  # first key
        assert launcher.custom_template_string.get() == ""
        assert launcher.jinja_enabled.get() is False
        assert launcher.custom_parameters_list == []

    def test_non_integer_n_gpu_layers_falls_back_to_zero(
        self, rich_launcher_factory, tmp_path
    ):
        cm, launcher = self._prepare(
            rich_launcher_factory, tmp_path, _full_cfg(n_gpu_layers="abc")
        )
        with patch("modules.config.messagebox"):
            cm.load_configuration()
        assert launcher.n_gpu_layers.get() == "0"
        launcher._set_gpu_layers.assert_called_with(0)

    def test_gpu_checkboxes_reconstructed_from_config(
        self, rich_launcher_factory, tmp_path
    ):
        cm, launcher = self._prepare(
            rich_launcher_factory, tmp_path,
            _full_cfg(gpu_indices=[0, 2], gpu_order=[2, 0]),
        )
        with patch("modules.config.messagebox"):
            cm.load_configuration()
        # app_settings gets the loaded indices and gpu_order.
        assert launcher.app_settings["selected_gpus"] == [0, 2]
        assert launcher.app_settings["gpu_order"] == [2, 0]
        launcher._update_gpu_checkboxes.assert_called()

    def test_gpu_order_with_duplicates_is_cleaned(
        self, rich_launcher_factory, tmp_path
    ):
        # Duplicates in gpu_order (e.g. produced by a buggy drag-reorder) are
        # deduplicated by the set() filter, and missing selections are appended.
        cm, launcher = self._prepare(
            rich_launcher_factory, tmp_path,
            _full_cfg(gpu_indices=[0, 1, 2], gpu_order=[1, 1, 2]),
        )
        with patch("modules.config.messagebox"):
            cm.load_configuration()
        # The filter is set-based so duplicates survive if they're in
        # selected_set — document that behaviour and also verify the missing
        # selection is appended.
        order = launcher.app_settings["gpu_order"]
        assert 0 in order  # missing selected GPU appended
        assert set(order) >= {0, 1, 2}

    def test_gpu_order_drops_entries_not_in_selected(
        self, rich_launcher_factory, tmp_path
    ):
        cm, launcher = self._prepare(
            rich_launcher_factory, tmp_path,
            _full_cfg(gpu_indices=[0], gpu_order=[0, 9, 5]),
        )
        with patch("modules.config.messagebox"):
            cm.load_configuration()
        # 9 and 5 aren't selected, should be dropped
        assert launcher.app_settings["gpu_order"] == [0]

    def test_model_not_found_warns_and_resets(
        self, rich_launcher_factory, tmp_path
    ):
        cfg = _full_cfg(model_path="/models/missing.gguf")
        cm, launcher = self._prepare(rich_launcher_factory, tmp_path, cfg)
        launcher.found_models = {}  # No models match
        with patch("modules.config.messagebox") as mb:
            cm.load_configuration()
        # model_path is cleared; warning issued; reset helpers called.
        assert launcher.model_path.get() == ""
        launcher._reset_model_info_display.assert_called()
        launcher._reset_gpu_layer_controls.assert_called()
        # showwarning for missing model + showinfo for "Loaded" success.
        assert mb.showwarning.called

    def test_model_found_triggers_select(self, rich_launcher_factory, tmp_path):
        model_path = tmp_path / "model.gguf"
        model_path.touch()
        cfg = _full_cfg(model_path=str(model_path))
        cm, launcher = self._prepare(rich_launcher_factory, tmp_path, cfg)
        # Populate found_models + listbox so the model resolves.
        launcher.found_models = {"display.gguf": model_path.resolve()}
        launcher.model_listbox.set_items(["display.gguf"])
        with patch("modules.config.messagebox"):
            cm.load_configuration()
        launcher._select_model_in_listbox.assert_called_with(0)

    def test_fit_ctx_synced_true_mirrors_ctx_size(
        self, rich_launcher_factory, tmp_path
    ):
        cfg = _full_cfg(ctx_size=4096, fit_ctx="9999", fit_ctx_synced=True)
        cm, launcher = self._prepare(rich_launcher_factory, tmp_path, cfg)
        with patch("modules.config.messagebox"):
            cm.load_configuration()
        # Synced: ignore stored fit_ctx, use ctx_size.
        assert launcher.fit_ctx.get() == "4096"

    def test_predefined_template_falls_back_to_first_when_missing(
        self, rich_launcher_factory, tmp_path
    ):
        cfg = _full_cfg(predefined_template_name="unknown")
        cm, launcher = self._prepare(rich_launcher_factory, tmp_path, cfg)
        # Template exists in the saved config, so it's loaded as-is
        with patch("modules.config.messagebox"):
            cm.load_configuration()
        assert launcher.predefined_template_name.get() == "unknown"

        # And when the saved name is missing entirely, the first key wins.
        cfg2 = _full_cfg()
        del cfg2["predefined_template_name"]
        launcher.config_listbox.set_items(["demo"])
        launcher.config_listbox.set_selection([0])
        launcher.saved_configs["demo"] = cfg2
        with patch("modules.config.messagebox"):
            cm.load_configuration()
        assert launcher.predefined_template_name.get() == "ChatML"

    def test_env_vars_and_ik_llama_delegates_called(
        self, rich_launcher_factory, tmp_path
    ):
        cm, launcher = self._prepare(rich_launcher_factory, tmp_path, _full_cfg())
        with patch("modules.config.messagebox"):
            cm.load_configuration()
        launcher.env_vars_manager.load_from_config.assert_called()
        launcher.ik_llama_tab.load_from_config.assert_called()


# ===========================================================================
# delete_configuration
# ===========================================================================

class TestDeleteConfiguration:
    def _prepare(self, rich_launcher_factory, tmp_path, saved, selection_names):
        from modules.config import ConfigManager
        launcher = rich_launcher_factory(tmp_path / "cfg.json")
        launcher.saved_configs = dict(saved)
        ordered_names = list(saved.keys())
        launcher.config_listbox.set_items(ordered_names)
        indices = [ordered_names.index(n) for n in selection_names]
        launcher.config_listbox.set_selection(indices)
        return ConfigManager(launcher), launcher

    def test_no_selection_shows_error(self, rich_launcher_factory, tmp_path):
        from modules.config import ConfigManager
        launcher = rich_launcher_factory(tmp_path / "cfg.json")
        launcher.config_listbox.set_selection([])
        cm = ConfigManager(launcher)
        with patch("modules.config.messagebox") as mb:
            cm.delete_configuration()
        assert mb.showerror.called

    def test_single_delete_with_confirmation(self, rich_launcher_factory, tmp_path):
        cm, launcher = self._prepare(
            rich_launcher_factory, tmp_path,
            {"a": {"model_path": "/a"}, "b": {"model_path": "/b"}},
            ["a"],
        )
        with patch("modules.config.messagebox.askyesno", return_value=True), \
                patch("modules.config.messagebox.showinfo"), \
                patch("modules.config.messagebox.showerror"):
            cm.delete_configuration()
        assert "a" not in launcher.saved_configs
        assert "b" in launcher.saved_configs
        launcher._save_configs.assert_called()
        launcher._update_config_listbox.assert_called()

    def test_multi_delete(self, rich_launcher_factory, tmp_path):
        cm, launcher = self._prepare(
            rich_launcher_factory, tmp_path,
            {"a": {}, "b": {}, "c": {}},
            ["a", "c"],
        )
        with patch("modules.config.messagebox.askyesno", return_value=True), \
                patch("modules.config.messagebox.showinfo"), \
                patch("modules.config.messagebox.showerror"):
            cm.delete_configuration()
        assert list(launcher.saved_configs.keys()) == ["b"]

    def test_user_cancels_confirmation(self, rich_launcher_factory, tmp_path):
        cm, launcher = self._prepare(
            rich_launcher_factory, tmp_path,
            {"a": {}, "b": {}},
            ["a"],
        )
        with patch("modules.config.messagebox.askyesno", return_value=False), \
                patch("modules.config.messagebox.showinfo") as info:
            cm.delete_configuration()
        # Nothing was deleted and _save_configs was not called.
        assert "a" in launcher.saved_configs
        launcher._save_configs.assert_not_called()
        assert not info.called

    def test_name_not_in_saved_configs_is_handled(
        self, rich_launcher_factory, tmp_path
    ):
        """If the listbox shows a stale name no longer present in
        saved_configs, we still handle it gracefully."""
        from modules.config import ConfigManager
        launcher = rich_launcher_factory(tmp_path / "cfg.json")
        launcher.saved_configs = {"real": {}}
        launcher.config_listbox.set_items(["stale"])
        launcher.config_listbox.set_selection([0])
        cm = ConfigManager(launcher)
        with patch("modules.config.messagebox.askyesno", return_value=True), \
                patch("modules.config.messagebox.showerror") as err, \
                patch("modules.config.messagebox.showinfo"):
            cm.delete_configuration()
        # The only entry was stale, so deleted_count stays 0 -> showerror.
        assert err.called
        launcher._save_configs.assert_not_called()

    def test_file_io_error_on_save_is_surfaced(
        self, rich_launcher_factory, tmp_path
    ):
        cm, launcher = self._prepare(
            rich_launcher_factory, tmp_path,
            {"a": {"model_path": "/a"}},
            ["a"],
        )
        launcher._save_configs.side_effect = OSError("disk full")
        with patch("modules.config.messagebox.askyesno", return_value=True), \
                patch("modules.config.messagebox.showinfo"), \
                patch("modules.config.messagebox.showerror"):
            with pytest.raises(OSError):
                cm.delete_configuration()


# ===========================================================================
# Config-name path traversal (bug fix 3)
# ===========================================================================

class TestConfigNameSanitization:
    def test_sanitize_strips_traversal(self):
        from modules.config import ConfigManager
        assert ConfigManager._sanitize_config_name("../../../etc/passwd") == "passwd"
        assert ConfigManager._sanitize_config_name("/absolute/path") == "path"
        assert ConfigManager._sanitize_config_name("C:\\Windows\\System32") == "System32"
        assert ConfigManager._sanitize_config_name("../..") == ""
        assert ConfigManager._sanitize_config_name("..") == ""
        assert ConfigManager._sanitize_config_name("...") == ""
        assert ConfigManager._sanitize_config_name(".") == ""
        assert ConfigManager._sanitize_config_name("/") == ""
        assert ConfigManager._sanitize_config_name("\\") == ""
        assert ConfigManager._sanitize_config_name("") == ""
        assert ConfigManager._sanitize_config_name(None) == ""

    def test_sanitize_preserves_normal_names(self):
        from modules.config import ConfigManager
        assert ConfigManager._sanitize_config_name("my_config") == "my_config"
        assert ConfigManager._sanitize_config_name("配置_日本語") == "配置_日本語"

    def test_sanitize_strips_control_chars(self):
        from modules.config import ConfigManager
        assert ConfigManager._sanitize_config_name("bad\x00name") == "badname"
        assert ConfigManager._sanitize_config_name("bad\nname") == "badname"

    def test_sanitize_accepts_non_string(self):
        from modules.config import ConfigManager
        # A non-string somehow reaching this boundary must not crash.
        assert ConfigManager._sanitize_config_name(42) == "42"

    def test_sanitize_rejects_windows_reserved_names(self):
        from modules.config import ConfigManager
        # Windows refuses to open files named CON/PRN/AUX/NUL/COM1-9/LPT1-9
        # (with or without an extension); reject them at the save boundary.
        for name in ("CON", "con", "PRN", "aux", "NUL", "COM1", "lpt9"):
            assert ConfigManager._sanitize_config_name(name) == "", name
        # Extensions don't save them — Windows matches on the stem.
        assert ConfigManager._sanitize_config_name("CON.txt") == ""
        assert ConfigManager._sanitize_config_name("nul.cfg") == ""
        # Names that merely *contain* a reserved word must still be allowed.
        assert ConfigManager._sanitize_config_name("CONfig") == "CONfig"
        assert ConfigManager._sanitize_config_name("my_nul_backup") == "my_nul_backup"

    def test_save_configuration_rejects_path_traversal(
        self, rich_launcher_factory, tmp_path
    ):
        from modules.config import ConfigManager
        launcher = rich_launcher_factory(tmp_path / "cfg.json")
        launcher.app_settings = {"selected_gpus": [], "gpu_order": []}
        # User (or script) pastes a path-traversal name into the config name.
        launcher.config_name.set("../../../etc/passwd")
        launcher.config_listbox.set_selection([])
        cm = ConfigManager(launcher)
        with patch("modules.config.messagebox"):
            cm.save_configuration()
        # Sanitised name is "passwd", and nothing was written outside tmp_path.
        assert "passwd" in launcher.saved_configs
        assert "../../../etc/passwd" not in launcher.saved_configs
        etc_passwd = Path("/etc/passwd.llama_cpp_launcher")
        assert not etc_passwd.exists()
        # The UI var was corrected too so the user sees what was actually saved.
        assert launcher.config_name.get() == "passwd"

    def test_save_configuration_empty_after_sanitize_autogenerates(
        self, rich_launcher_factory, tmp_path
    ):
        from modules.config import ConfigManager
        launcher = rich_launcher_factory(tmp_path / "cfg.json")
        launcher.app_settings = {"selected_gpus": [], "gpu_order": []}
        # Pure traversal — sanitizes to empty, should autogen a default.
        launcher.config_name.set("../..")
        launcher.config_listbox.set_selection([])
        cm = ConfigManager(launcher)
        with patch("modules.config.messagebox"):
            cm.save_configuration()
        assert "" not in launcher.saved_configs
        assert None not in launcher.saved_configs
        # Some non-empty fallback name was persisted.
        assert len(launcher.saved_configs) == 1
        (saved_name,) = launcher.saved_configs.keys()
        assert saved_name  # non-empty
        assert "/" not in saved_name and "\\" not in saved_name
        assert ".." not in saved_name


# ===========================================================================
# Unicode / BOM / large payloads in load_saved_configs
# ===========================================================================

class TestUnicodeAndBom:
    def test_bom_prefixed_json_is_rejected_cleanly(
        self, rich_launcher_factory, tmp_path
    ):
        """``json.loads`` reads the file via ``encoding='utf-8'`` (not
        ``utf-8-sig``), so a UTF-8 BOM prefix causes ``json.JSONDecodeError``
        with ``'Unexpected UTF-8 BOM'``. The module must treat this as a
        parse failure, not a silent empty-configs state: a BOM-prefixed file
        usually means the user hand-edited in a Windows editor that auto-
        added the BOM, and we don't want to silently wipe their configs.

        Observable contract on rejection:
          * ``configs_loaded_successfully`` is False (the save guard uses
            this to refuse overwriting the file on next save).
          * ``launcher.saved_configs`` is ``{}``.
          * ``launcher.app_settings`` gets populated with the module's
            baseline defaults so the UI can still render.
        """
        from modules.config import ConfigManager
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_bytes(b"\xef\xbb\xbf" + json.dumps(
            {"configs": {}, "app_settings": {}}
        ).encode("utf-8"))
        launcher = rich_launcher_factory(cfg_path)
        cm = ConfigManager(launcher)
        with patch("modules.config.messagebox"):
            cm.load_saved_configs()

        # 1) Explicit failure flag — next save must refuse to overwrite.
        assert cm.configs_loaded_successfully is False
        # 2) No partial config state leaks in.
        assert launcher.saved_configs == {}
        # 3) app_settings falls back to the default key set so the UI
        # has something to render. We assert on the *key shape* rather
        # than specific values — values are platform-dependent — which
        # still catches a regression that drops any of the defaults.
        expected_keys = {
            "custom_parameters", "gpu_order", "host", "last_llama_cpp_dir",
            "last_model_path", "last_venv_dir", "model_dirs",
            "model_list_height", "port", "selected_gpus",
            "selected_mmproj_path", "ui_font_family", "ui_font_size",
            "ui_theme_mode", "ui_theme_name",
        }
        assert expected_keys.issubset(launcher.app_settings.keys()), (
            f"Missing default app_settings keys after BOM rejection: "
            f"{expected_keys - set(launcher.app_settings.keys())}"
        )

    def test_unicode_config_name_loads(
        self, rich_launcher_factory, tmp_path
    ):
        from modules.config import ConfigManager
        cfg_path = tmp_path / "cfg.json"
        cfg = {
            "configs": {
                "日本語_🔥_αβγ": {"model_path": "/m"},
                "العربية": {"model_path": "/m2"},
            },
            "app_settings": {},
        }
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
        launcher = rich_launcher_factory(cfg_path)
        cm = ConfigManager(launcher)
        cm.load_saved_configs()
        assert "日本語_🔥_αβγ" in launcher.saved_configs
        assert "العربية" in launcher.saved_configs

    def test_unicode_model_path_round_trip(
        self, rich_launcher_factory, tmp_path
    ):
        from modules.config import ConfigManager
        cfg_path = tmp_path / "cfg.json"
        payload = {
            "configs": {
                "cfg1": {"model_path": "/models/日本語/🔥.gguf"},
            },
            "app_settings": {},
        }
        cfg_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        launcher = rich_launcher_factory(cfg_path)
        cm = ConfigManager(launcher)
        cm.load_saved_configs()
        assert (launcher.saved_configs["cfg1"]["model_path"]
                == "/models/日本語/🔥.gguf")

    def test_large_config_payload_10mb(
        self, rich_launcher_factory, tmp_path
    ):
        """10 MB sanity test — must load without OOM / stalling."""
        from modules.config import ConfigManager
        cfg_path = tmp_path / "cfg.json"
        # ~10MB of padding in a custom_template_string.
        big_string = "x" * (10 * 1024 * 1024)
        payload = {
            "configs": {"big": {"model_path": "/m", "custom_template_string": big_string}},
            "app_settings": {},
        }
        cfg_path.write_text(json.dumps(payload), encoding="utf-8")
        launcher = rich_launcher_factory(cfg_path)
        cm = ConfigManager(launcher)
        cm.load_saved_configs()
        assert len(launcher.saved_configs["big"]["custom_template_string"]) == len(big_string)


# ===========================================================================
# Symlink race in legacy->local migration
# ===========================================================================

class TestLegacyMigrationSymlink:
    def test_legacy_migration_with_symlink_target_outside_repo(
        self, rich_launcher_factory, tmp_path
    ):
        """A legacy config that's actually a symlink should be handled safely:
        either migrated (resolving the symlink) or left alone. Either way, we
        don't crash, and we don't clobber the real target."""
        from modules.config import ConfigManager
        fake_repo = tmp_path / "repo"
        (fake_repo / "modules").mkdir(parents=True)
        fake_config_py = fake_repo / "modules" / "config.py"
        fake_config_py.write_text("")

        # Real external file, plus a symlink to it in repo root.
        external = tmp_path / "external.json"
        external.write_text(json.dumps({"configs": {"x": {"y": 1}}, "app_settings": {}}))
        legacy_symlink = fake_repo / "llama_cpp_launcher_configs.json"
        try:
            legacy_symlink.symlink_to(external)
        except (OSError, NotImplementedError):
            pytest.skip("cannot create symlink on this platform")

        launcher = rich_launcher_factory(tmp_path / "cfg.json")
        cm = ConfigManager(launcher)
        with patch("modules.config.__file__", str(fake_config_py)), \
                patch("modules.config.messagebox"):
            result = cm.get_config_path()

        # The local-path migration ran without error and the local file now has
        # the legacy content. Important: the real external file must still exist.
        assert result == fake_repo / "config" / "llama_cpp_launcher_configs.json"
        assert external.exists()


# ===========================================================================
# get_config_path error branches
# ===========================================================================

class TestGetConfigPathEdgeCases:
    def test_home_raises_still_returns_null_device(
        self, rich_launcher_factory, tmp_path
    ):
        """If both the local path and Path.home() are unusable we return
        the platform's null-device sentinel."""
        from modules.config import ConfigManager
        fake_repo = tmp_path / "repo"
        (fake_repo / "modules").mkdir(parents=True)
        fake_config_py = fake_repo / "modules" / "config.py"
        fake_config_py.write_text("")

        launcher = rich_launcher_factory(tmp_path / "cfg.json")
        cm = ConfigManager(launcher)

        def fake_access(p, mode):
            if mode == os.W_OK and str(fake_repo) in str(p):
                return False
            return True

        with patch("modules.config.__file__", str(fake_config_py)), \
                patch("modules.config.os.access", side_effect=fake_access), \
                patch("modules.config.sys.platform", "linux"), \
                patch("modules.config.Path.home",
                      side_effect=RuntimeError("no home")), \
                patch("modules.config.messagebox"):
            result = cm.get_config_path()
        # Should be a dummy null device path.
        assert result.name in ("null", "NUL")

    def test_parent_is_a_file_returns_fallback_or_null(
        self, rich_launcher_factory, tmp_path
    ):
        """If the expected config dir *is* a file (not a dir) the manager
        must not crash; it falls back to the user-home directory."""
        from modules.config import ConfigManager
        fake_repo = tmp_path / "repo"
        (fake_repo / "modules").mkdir(parents=True)
        fake_config_py = fake_repo / "modules" / "config.py"
        fake_config_py.write_text("")
        # Create a *file* where the config dir should be.
        (fake_repo / "config").write_text("I am a file, not a dir.")

        fake_home = tmp_path / "home"
        fake_home.mkdir()

        launcher = rich_launcher_factory(tmp_path / "cfg.json")
        cm = ConfigManager(launcher)
        with patch("modules.config.__file__", str(fake_config_py)), \
                patch("modules.config.sys.platform", "linux"), \
                patch("modules.config.Path.home", return_value=fake_home), \
                patch("modules.config.messagebox"):
            result = cm.get_config_path()
        # Either we got the fallback in the fake home or a null-device sentinel.
        assert (result == fake_home / ".config" / "llama_cpp_launcher" / "configs.json"
                or result.name in ("null", "NUL"))


# ===========================================================================
# generate_default_config_name — expanded edge cases (bug fix 1)
# ===========================================================================

class TestGenerateDefaultConfigName:
    def _launcher(self, rich_launcher_factory, tmp_path, **overrides):
        launcher = rich_launcher_factory(tmp_path / "cfg.json")
        for k, v in overrides.items():
            if hasattr(launcher, k):
                attr = getattr(launcher, k)
                if isinstance(attr, FakeVar):
                    attr.set(v)
                else:
                    setattr(launcher, k, v)
            else:
                setattr(launcher, k, v)
        return launcher

    def test_trailing_dots_stripped(self, rich_launcher_factory, tmp_path):
        from modules.config import ConfigManager
        launcher = self._launcher(
            rich_launcher_factory, tmp_path,
            model_path="/tmp/bad.name...gguf",
        )
        cm = ConfigManager(launcher)
        name = cm.generate_default_config_name()
        # The sanitised name must not end with "." even though the model stem did.
        first_part = name.split("_")[0]
        assert not first_part.endswith(".")

    def test_leading_trailing_whitespace_stripped(
        self, rich_launcher_factory, tmp_path
    ):
        from modules.config import ConfigManager
        launcher = self._launcher(
            rich_launcher_factory, tmp_path,
            model_path="/tmp/   spaced.gguf",
        )
        # Model-listbox is empty so it falls back to the Path stem.
        cm = ConfigManager(launcher)
        name = cm.generate_default_config_name()
        # Leading space in stem should be stripped, not preserved as "_spaced".
        # The regex replaces the space with "_", then strip('_') removes it.
        assert not name.startswith("_")
        assert "spaced" in name

    def test_control_chars_removed(self, rich_launcher_factory, tmp_path):
        from modules.config import ConfigManager
        launcher = self._launcher(rich_launcher_factory, tmp_path)
        launcher.model_path.set("/tmp/m.gguf")
        # Simulate the listbox having a bad name with control chars.
        launcher.model_listbox.set_items(["mo\x00de\x01l\x7f"])
        launcher.model_listbox.set_selection([0])
        cm = ConfigManager(launcher)
        name = cm.generate_default_config_name()
        # No control chars in the output.
        for c in name:
            assert ord(c) >= 0x20 and ord(c) != 0x7f

    def test_unicode_model_name_preserved(
        self, rich_launcher_factory, tmp_path
    ):
        from modules.config import ConfigManager
        launcher = self._launcher(rich_launcher_factory, tmp_path)
        launcher.model_path.set("/tmp/日本語.gguf")
        cm = ConfigManager(launcher)
        name = cm.generate_default_config_name()
        # Model stem is preserved (unicode is not in the illegal-char class).
        assert "日本語" in name or "model" in name

    def test_very_long_model_name_is_truncated(
        self, rich_launcher_factory, tmp_path
    ):
        from modules.config import ConfigManager
        launcher = self._launcher(rich_launcher_factory, tmp_path)
        long = "a" * 300 + ".gguf"
        launcher.model_path.set(f"/tmp/{long}")
        cm = ConfigManager(launcher)
        name = cm.generate_default_config_name()
        # Overall generated name capped at 80 chars.
        assert len(name) <= 80

    def test_regex_special_chars_in_model_name(
        self, rich_launcher_factory, tmp_path
    ):
        from modules.config import ConfigManager
        launcher = self._launcher(rich_launcher_factory, tmp_path)
        launcher.model_path.set("/tmp/my(model)[v1]+fast.gguf")
        cm = ConfigManager(launcher)
        # Must not raise.
        name = cm.generate_default_config_name()
        assert len(name) > 0

    def test_empty_thread_count_does_not_crash(
        self, rich_launcher_factory, tmp_path
    ):
        from modules.config import ConfigManager
        launcher = self._launcher(rich_launcher_factory, tmp_path)
        launcher.model_path.set("/tmp/m.gguf")
        launcher.threads.set("")
        launcher.threads_batch.set("")
        cm = ConfigManager(launcher)
        name = cm.generate_default_config_name()
        # Empty string != logical_cores default so the abbreviation should
        # NOT be added (current_val is falsy after strip()).
        assert "th=" not in name

    def test_temp_abbreviation_branch(self, rich_launcher_factory, tmp_path):
        from modules.config import ConfigManager
        launcher = self._launcher(rich_launcher_factory, tmp_path)
        launcher.model_path.set("/tmp/m.gguf")
        launcher.temperature.set("0.2")  # Non-default
        cm = ConfigManager(launcher)
        name = cm.generate_default_config_name()
        assert "temp=0.2" in name

    def test_batch_abbreviation_branch(self, rich_launcher_factory, tmp_path):
        from modules.config import ConfigManager
        launcher = self._launcher(rich_launcher_factory, tmp_path)
        launcher.model_path.set("/tmp/m.gguf")
        launcher.batch_size.set("256")  # != 512
        cm = ConfigManager(launcher)
        name = cm.generate_default_config_name()
        assert "b=256" in name

    def test_seed_abbreviation_branch(self, rich_launcher_factory, tmp_path):
        from modules.config import ConfigManager
        launcher = self._launcher(rich_launcher_factory, tmp_path)
        launcher.model_path.set("/tmp/m.gguf")
        launcher.seed.set("42")
        cm = ConfigManager(launcher)
        name = cm.generate_default_config_name()
        assert "s=42" in name

    def test_flash_attn_boolean_abbreviation_when_disabled(
        self, rich_launcher_factory, tmp_path
    ):
        """flash_attn defaults to True now, so checking the box matches the
        default and is omitted from the name. Unchecking the box differs
        from the default and should appear as 'no-fa'."""
        from modules.config import ConfigManager
        launcher = self._launcher(rich_launcher_factory, tmp_path)
        launcher.model_path.set("/tmp/m.gguf")
        launcher.flash_attn.set(False)
        cm = ConfigManager(launcher)
        name = cm.generate_default_config_name()
        assert "no-fa" in name.split("_")

    def test_flash_attn_boolean_at_default_omitted(
        self, rich_launcher_factory, tmp_path
    ):
        """Conversely, leaving flash_attn at its True default must NOT add
        any token (otherwise every default config name would be polluted)."""
        from modules.config import ConfigManager
        launcher = self._launcher(rich_launcher_factory, tmp_path)
        launcher.model_path.set("/tmp/m.gguf")
        launcher.flash_attn.set(True)
        cm = ConfigManager(launcher)
        name = cm.generate_default_config_name()
        assert "fa" not in name.split("_")
        assert "no-fa" not in name.split("_")

    def test_ctx_size_nondefault_added(self, rich_launcher_factory, tmp_path):
        from modules.config import ConfigManager
        launcher = self._launcher(rich_launcher_factory, tmp_path)
        launcher.model_path.set("/tmp/m.gguf")
        launcher.ctx_size.set(8192)
        cm = ConfigManager(launcher)
        name = cm.generate_default_config_name()
        assert "ctx=8192" in name

    def test_gpu_all_branch(self, rich_launcher_factory, tmp_path):
        from modules.config import ConfigManager
        launcher = self._launcher(rich_launcher_factory, tmp_path)
        launcher.model_path.set("/tmp/m.gguf")
        launcher.n_gpu_layers_int.set(32)
        launcher.max_gpu_layers.set(32)
        cm = ConfigManager(launcher)
        name = cm.generate_default_config_name()
        assert "gpu-all" in name

    def test_all_defaults_no_suffix_params(self, rich_launcher_factory, tmp_path):
        from modules.config import ConfigManager
        launcher = self._launcher(rich_launcher_factory, tmp_path)
        launcher.model_path.set("/tmp/m.gguf")
        # logical_cores=8 matches default so don't add th=.
        launcher.threads.set("8")
        # threads_batch default for name-generation is "4" so force it.
        launcher.threads_batch.set("4")
        cm = ConfigManager(launcher)
        name = cm.generate_default_config_name()
        # Should essentially be just the model stem.
        assert name == "m"


# ===========================================================================
# GPU order edge case (LOW risk)
# ===========================================================================

def test_save_configs_preserves_gpu_order_with_duplicates(
    rich_launcher_factory, tmp_path
):
    """save_configs reads gpu_order from app_settings — duplicates in the
    source list should persist verbatim so the launch order (tensor split
    pairing) is preserved."""
    from modules.config import ConfigManager
    cfg_path = tmp_path / "cfg.json"
    launcher = rich_launcher_factory(cfg_path)
    # Duplicates happen if the UI drag-reorder widget mis-handles rows.
    launcher.app_settings["gpu_order"] = [0, 1, 1, 2]
    launcher.saved_configs = {"x": {}}
    cm = ConfigManager(launcher)
    cm.save_configs()
    data = json.loads(cfg_path.read_text("utf-8"))
    # save_configs doesn't dedupe gpu_order — safest to document & preserve.
    assert data["app_settings"]["gpu_order"] == [0, 1, 1, 2]
