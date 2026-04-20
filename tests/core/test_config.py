"""Regression tests for ``modules/config.py`` (``ConfigManager``).

``ConfigManager`` is tightly coupled to the Tk launcher instance (it reads from
dozens of ``tk.*Var`` widgets and calls methods on listboxes, UI callbacks, and
``messagebox``). These tests exercise the public API that can meaningfully be
tested without a real UI: ``load_saved_configs``, ``save_configs``,
``get_config_path``, ``update_config_listbox``, ``on_config_selected``,
``save_configuration``, ``generate_default_config_name``, ``current_cfg``, and
data round-trips against real config files under ``config/``.

UI-heavy methods (``load_configuration``, ``delete_configuration``,
``export_configurations``, ``import_configurations``) are exercised indirectly
through save/load round-trips; only a handful of lightweight behaviour checks
are attempted with the messagebox/filedialog layers mocked.
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


# ---------------------------------------------------------------------------
# Helpers / launcher-mock fixture
# ---------------------------------------------------------------------------

class _FakeVar:
    """Minimal stand-in for ``tk.StringVar`` / ``tk.BooleanVar``.

    Stores whatever value ``set()`` receives and returns it from ``get()``.
    Unlike the real Tk vars, this requires no display and no event loop.
    """

    def __init__(self, value=""):
        self._value = value

    def get(self):
        return self._value

    def set(self, v):
        self._value = v


def _make_launcher_mock(config_path: Path, *, saved_configs=None, app_settings=None,
                       model_dirs=None, gpu_devices=None):
    """Build a MagicMock that looks enough like the real launcher for
    ``ConfigManager`` to operate on it.

    Only the attributes exercised by the tested methods are populated.
    """
    launcher = MagicMock()

    # Paths and data containers
    launcher.config_path = config_path
    launcher.saved_configs = dict(saved_configs) if saved_configs else {}
    launcher.app_settings = dict(app_settings) if app_settings else {}
    launcher.model_dirs = list(model_dirs) if model_dirs else []
    launcher.detected_gpu_devices = list(gpu_devices) if gpu_devices else []
    launcher.custom_parameters_list = []
    launcher.manual_gpu_list = []
    launcher.gpu_vars = []
    launcher.physical_cores = 4
    launcher.logical_cores = 8
    launcher.fit_ctx_synced = True

    # Simulated Tk variables used during save_configs/current_cfg.
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

    launcher.ctx_size = _FakeVar(2048)

    # Sub-manager stubs — their save_to_config returns something mergeable.
    launcher.env_vars_manager = MagicMock()
    launcher.env_vars_manager.save_to_config.return_value = {
        "environmental_variables": {"enabled": False, "predefined": {}, "custom": []}
    }
    launcher.ik_llama_tab = MagicMock()
    launcher.ik_llama_tab.save_to_config.return_value = {}

    return launcher


@pytest.fixture
def launcher_factory():
    """Factory so tests can pick their own config_path / state."""
    def _factory(config_path: Path, **kwargs):
        return _make_launcher_mock(config_path, **kwargs)
    return _factory


@pytest.fixture
def config_manager(launcher_factory, tmp_path):
    from modules.config import ConfigManager
    launcher = launcher_factory(tmp_path / "cfg.json")
    return ConfigManager(launcher), launcher


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

def test_init_sets_launcher_and_loaded_flag(launcher_factory, tmp_path):
    from modules.config import ConfigManager
    launcher = launcher_factory(tmp_path / "cfg.json")
    cm = ConfigManager(launcher)
    assert cm.launcher is launcher
    assert cm.configs_loaded_successfully is True


# ---------------------------------------------------------------------------
# load_saved_configs
# ---------------------------------------------------------------------------

def test_load_saved_configs_missing_file_sets_flag_true(launcher_factory, tmp_path):
    from modules.config import ConfigManager
    cfg_path = tmp_path / "does_not_exist.json"
    launcher = launcher_factory(cfg_path)
    cm = ConfigManager(launcher)
    cm.configs_loaded_successfully = False  # force a non-default so we see the write
    cm.load_saved_configs()
    assert cm.configs_loaded_successfully is True


def test_load_saved_configs_null_device_is_treated_as_missing(launcher_factory, tmp_path):
    from modules.config import ConfigManager
    # Simulate the "disabled" sentinel paths returned by get_config_path fallbacks.
    for sentinel_name in ("null", "NUL"):
        p = tmp_path / sentinel_name
        p.touch()
        launcher = launcher_factory(p)
        cm = ConfigManager(launcher)
        cm.load_saved_configs()
        assert cm.configs_loaded_successfully is True
        assert launcher.saved_configs == {}


def test_load_saved_configs_reads_configs_and_app_settings(
    launcher_factory, tmp_path, sample_config_dict, sample_configs_file
):
    from modules.config import ConfigManager
    cfg_path = sample_configs_file(name="cfg_a")
    launcher = launcher_factory(cfg_path)
    cm = ConfigManager(launcher)
    cm.load_saved_configs()
    assert "cfg_a" in launcher.saved_configs
    assert launcher.saved_configs["cfg_a"] == sample_config_dict
    assert launcher.app_settings["host"] == "127.0.0.1"
    assert cm.configs_loaded_successfully is True


def test_load_saved_configs_filters_none_keys(launcher_factory, tmp_path):
    """A None key in the configs dict (which can happen after buggy serialisation
    or in-memory corruption) should be silently dropped on load."""
    from modules.config import ConfigManager
    cfg_path = tmp_path / "cfg.json"
    # Need any valid JSON so the read/parse chain runs; the actual value is
    # replaced by the ``json.loads`` patch below so we can slip a None key in.
    cfg_path.write_text(json.dumps({"configs": {}, "app_settings": {}}))
    launcher = launcher_factory(cfg_path)
    cm = ConfigManager(launcher)

    with patch(
        "modules.config.json.loads",
        return_value={"configs": {None: {"x": 1}, "ok": {"y": 2}}, "app_settings": {}},
    ):
        cm.load_saved_configs()
    assert list(launcher.saved_configs.keys()) == ["ok"]


def test_load_saved_configs_malformed_json_sets_failure_flag(
    launcher_factory, tmp_path
):
    from modules.config import ConfigManager
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text("{not valid json")
    launcher = launcher_factory(cfg_path)
    cm = ConfigManager(launcher)
    with patch("modules.config.messagebox"):
        cm.load_saved_configs()
    assert cm.configs_loaded_successfully is False
    assert launcher.saved_configs == {}
    # Defaults should be repopulated.
    assert launcher.app_settings["host"] == "127.0.0.1"
    assert launcher.app_settings["port"] == "8080"


def test_load_saved_configs_invalid_model_list_height_coerced(
    launcher_factory, tmp_path
):
    from modules.config import ConfigManager
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({
        "configs": {},
        "app_settings": {"model_list_height": "not-an-int"},
    }))
    launcher = launcher_factory(cfg_path)
    cm = ConfigManager(launcher)
    cm.load_saved_configs()
    assert launcher.app_settings["model_list_height"] == 8


def test_load_saved_configs_invalid_selected_gpus_coerced(launcher_factory, tmp_path):
    from modules.config import ConfigManager
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({
        "configs": {},
        "app_settings": {"selected_gpus": "not-a-list"},
    }))
    launcher = launcher_factory(cfg_path)
    cm = ConfigManager(launcher)
    cm.load_saved_configs()
    assert launcher.app_settings["selected_gpus"] == []


@pytest.mark.parametrize("bad_mode", ["neon", "BLACK", 42, None])
def test_load_saved_configs_invalid_ui_theme_mode_resets_to_auto(
    launcher_factory, tmp_path, bad_mode
):
    from modules.config import ConfigManager
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({
        "configs": {},
        "app_settings": {"ui_theme_mode": bad_mode},
    }))
    launcher = launcher_factory(cfg_path)
    cm = ConfigManager(launcher)
    cm.load_saved_configs()
    assert launcher.app_settings["ui_theme_mode"] == "auto"


@pytest.mark.parametrize("font_in, expected", [
    (12, 12),
    (0, 0),
    (31, 31),
    (32, 0),
    (-5, 0),
    ("eight", 0),
    (None, 0),
])
def test_load_saved_configs_ui_font_size_clamped(
    launcher_factory, tmp_path, font_in, expected
):
    from modules.config import ConfigManager
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({
        "configs": {},
        "app_settings": {"ui_font_size": font_in},
    }))
    launcher = launcher_factory(cfg_path)
    cm = ConfigManager(launcher)
    cm.load_saved_configs()
    assert launcher.app_settings["ui_font_size"] == expected


def test_load_saved_configs_strips_ui_scaling_legacy_key(launcher_factory, tmp_path):
    from modules.config import ConfigManager
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({
        "configs": {},
        "app_settings": {"ui_scaling": 1.5},
    }))
    launcher = launcher_factory(cfg_path)
    cm = ConfigManager(launcher)
    cm.load_saved_configs()
    assert "ui_scaling" not in launcher.app_settings


def test_load_saved_configs_filters_selected_gpus_to_detected(launcher_factory, tmp_path):
    from modules.config import ConfigManager
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({
        "configs": {},
        "app_settings": {"selected_gpus": [0, 1, 9], "gpu_order": [9, 0]},
    }))
    launcher = launcher_factory(
        cfg_path, gpu_devices=[{"id": 0, "name": "gpu0"}, {"id": 1, "name": "gpu1"}]
    )
    cm = ConfigManager(launcher)
    cm.load_saved_configs()
    assert launcher.app_settings["selected_gpus"] == [0, 1]
    # gpu_order keeps existing valid entries, then appends any missing selected GPUs.
    assert launcher.app_settings["gpu_order"] == [0, 1]


# ---------------------------------------------------------------------------
# save_configs
# ---------------------------------------------------------------------------

def test_save_configs_writes_json_with_expected_top_level_keys(
    launcher_factory, tmp_path
):
    from modules.config import ConfigManager
    cfg_path = tmp_path / "cfg.json"
    launcher = launcher_factory(cfg_path)
    launcher.saved_configs = {"one": {"model_path": "/x"}}
    launcher.app_settings = {"host": "0.0.0.0"}
    cm = ConfigManager(launcher)
    cm.save_configs()
    data = json.loads(cfg_path.read_text("utf-8"))
    assert "configs" in data and "app_settings" in data
    assert data["configs"] == {"one": {"model_path": "/x"}}
    assert data["app_settings"]["host"] == "127.0.0.1"  # from launcher.host var


def test_save_configs_disabled_for_null_path(launcher_factory, tmp_path):
    from modules.config import ConfigManager
    p = tmp_path / "NUL"
    launcher = launcher_factory(p)
    launcher.saved_configs = {"x": {"model_path": "/a"}}
    cm = ConfigManager(launcher)
    cm.save_configs()
    # Did not write the file:
    assert not p.exists() or p.read_text() == ""


def test_save_configs_cleans_invalid_model_dirs(launcher_factory, tmp_path):
    from modules.config import ConfigManager
    real_dir = tmp_path / "models"
    real_dir.mkdir()
    launcher = launcher_factory(tmp_path / "cfg.json")
    launcher.model_dirs = [str(real_dir), "/this/does/not/exist"]
    cm = ConfigManager(launcher)
    cm.save_configs()
    saved = json.loads((tmp_path / "cfg.json").read_text("utf-8"))
    # Only the real dir survived (resolved to its real absolute path).
    dirs = saved["app_settings"]["model_dirs"]
    assert len(dirs) == 1
    assert Path(dirs[0]) == real_dir.resolve()


def test_save_configs_data_loss_safeguard_blocks_empty_wipe(
    launcher_factory, tmp_path
):
    """If load failed, in-memory is empty, but on-disk is populated → refuse to save."""
    from modules.config import ConfigManager
    cfg_path = tmp_path / "cfg.json"
    # Pre-populate the on-disk file with a real config.
    cfg_path.write_text(json.dumps({
        "configs": {"important": {"model_path": "/x"}},
        "app_settings": {},
    }))
    launcher = launcher_factory(cfg_path)
    launcher.saved_configs = {}
    cm = ConfigManager(launcher)
    cm.configs_loaded_successfully = False  # Simulate load failure.
    with patch("modules.config.messagebox"):
        cm.save_configs()
    # File unchanged: the original configs are still present.
    after = json.loads(cfg_path.read_text("utf-8"))
    assert "important" in after["configs"]


def test_save_configs_allows_intentional_wipe_after_successful_load(
    launcher_factory, tmp_path
):
    from modules.config import ConfigManager
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({
        "configs": {"to_delete": {"model_path": "/x"}},
        "app_settings": {},
    }))
    launcher = launcher_factory(cfg_path)
    launcher.saved_configs = {}
    cm = ConfigManager(launcher)
    cm.configs_loaded_successfully = True   # User deliberately deleted all.
    cm.save_configs()
    after = json.loads(cfg_path.read_text("utf-8"))
    assert after["configs"] == {}


def test_save_configs_success_resets_loaded_flag(launcher_factory, tmp_path):
    from modules.config import ConfigManager
    cfg_path = tmp_path / "cfg.json"
    launcher = launcher_factory(cfg_path)
    launcher.saved_configs = {"ok": {"x": 1}}
    cm = ConfigManager(launcher)
    cm.configs_loaded_successfully = False
    cm.save_configs()
    assert cm.configs_loaded_successfully is True


# ---------------------------------------------------------------------------
# save_configs / load_saved_configs round-trip
# ---------------------------------------------------------------------------

def test_round_trip_save_then_load_preserves_data(launcher_factory, tmp_path):
    from modules.config import ConfigManager
    cfg_path = tmp_path / "cfg.json"

    launcher1 = launcher_factory(cfg_path)
    launcher1.saved_configs = {
        "simple": {"model_path": "/m.gguf", "ctx_size": 4096, "flash_attn": True},
        "nested": {
            "list_of_strings": ["--foo", "--bar"],
            "dict_in_dict": {"a": {"b": {"c": 1}}},
            "empty_list": [],
            "empty_dict": {},
            "null_value": None,
            "unicode": "日本語テスト αβγ",
        },
    }
    launcher1.app_settings = {"host": "0.0.0.0", "port": "9999",
                              "model_dirs": [], "selected_gpus": [],
                              "custom_parameters": []}
    cm1 = ConfigManager(launcher1)
    cm1.save_configs()

    launcher2 = launcher_factory(cfg_path)
    cm2 = ConfigManager(launcher2)
    cm2.load_saved_configs()
    assert launcher2.saved_configs == launcher1.saved_configs


def test_round_trip_handles_empty_configs(launcher_factory, tmp_path):
    from modules.config import ConfigManager
    cfg_path = tmp_path / "cfg.json"
    launcher1 = launcher_factory(cfg_path)
    launcher1.saved_configs = {}
    cm1 = ConfigManager(launcher1)
    cm1.save_configs()

    launcher2 = launcher_factory(cfg_path)
    cm2 = ConfigManager(launcher2)
    cm2.load_saved_configs()
    assert launcher2.saved_configs == {}
    assert cm2.configs_loaded_successfully is True


# ---------------------------------------------------------------------------
# get_config_path
# ---------------------------------------------------------------------------

def test_get_config_path_returns_local_path_when_writable(launcher_factory, tmp_path, monkeypatch):
    """``get_config_path`` prefers the local ``config/`` dir under repo root."""
    from modules.config import ConfigManager
    # Patch __file__ relative repo_root so we get an isolated path.
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    (fake_repo / "modules").mkdir()
    fake_config_py = fake_repo / "modules" / "config.py"
    fake_config_py.write_text("")  # Just needs to exist for __file__ resolution.

    launcher = launcher_factory(tmp_path / "cfg.json")
    cm = ConfigManager(launcher)

    with patch("modules.config.__file__", str(fake_config_py)):
        path = cm.get_config_path()
    assert path == (fake_repo / "config" / "llama_cpp_launcher_configs.json")
    assert path.parent.exists()


def test_get_config_path_falls_back_when_local_unwritable(launcher_factory, tmp_path, monkeypatch):
    from modules.config import ConfigManager
    fake_repo = tmp_path / "repo"
    (fake_repo / "modules").mkdir(parents=True)
    fake_config_py = fake_repo / "modules" / "config.py"
    fake_config_py.write_text("")

    launcher = launcher_factory(tmp_path / "cfg.json")
    cm = ConfigManager(launcher)

    # Force no write access to the local dir.
    def fake_access(p, mode):
        if mode == os.W_OK and str(fake_repo) in str(p):
            return False
        return True

    fake_home = tmp_path / "home"
    fake_home.mkdir()

    with patch("modules.config.__file__", str(fake_config_py)), \
         patch("modules.config.os.access", side_effect=fake_access), \
         patch("modules.config.sys.platform", "linux"), \
         patch("modules.config.Path.home", return_value=fake_home):
        path = cm.get_config_path()

    # The fallback path on Linux is ~/.config/llama_cpp_launcher/configs.json.
    assert path == fake_home / ".config" / "llama_cpp_launcher" / "configs.json"


def test_get_config_path_migrates_legacy_file(launcher_factory, tmp_path):
    """If a pre-reorg config exists in repo root it should be migrated into ``config/``."""
    from modules.config import ConfigManager
    fake_repo = tmp_path / "repo"
    (fake_repo / "modules").mkdir(parents=True)
    fake_config_py = fake_repo / "modules" / "config.py"
    fake_config_py.write_text("")

    legacy = fake_repo / "llama_cpp_launcher_configs.json"
    legacy.write_text(json.dumps({"configs": {"legacy": {"x": 1}}, "app_settings": {}}))

    launcher = launcher_factory(tmp_path / "cfg.json")
    cm = ConfigManager(launcher)

    with patch("modules.config.__file__", str(fake_config_py)):
        path = cm.get_config_path()

    assert path == fake_repo / "config" / "llama_cpp_launcher_configs.json"
    assert path.exists()
    assert not legacy.exists()   # It was moved.
    # And a .premigration_backup was left behind:
    backup = fake_repo / "llama_cpp_launcher_configs.json.premigration_backup"
    assert backup.exists()
    assert json.loads(backup.read_text("utf-8")) == json.loads(path.read_text("utf-8"))


def test_get_config_path_removes_empty_existing_file(launcher_factory, tmp_path):
    from modules.config import ConfigManager
    fake_repo = tmp_path / "repo"
    (fake_repo / "modules").mkdir(parents=True)
    fake_config_py = fake_repo / "modules" / "config.py"
    fake_config_py.write_text("")
    # Create an empty local config file.
    cfg_dir = fake_repo / "config"
    cfg_dir.mkdir()
    stale = cfg_dir / "llama_cpp_launcher_configs.json"
    stale.write_text("")

    launcher = launcher_factory(tmp_path / "cfg.json")
    cm = ConfigManager(launcher)

    with patch("modules.config.__file__", str(fake_config_py)):
        path = cm.get_config_path()

    # The empty file was deleted and the returned path is still the local path.
    assert path == stale
    assert not stale.exists() or stale.stat().st_size == 0


# ---------------------------------------------------------------------------
# update_config_listbox, on_config_selected, save_configuration
# ---------------------------------------------------------------------------

def test_update_config_listbox_sorts_and_filters_none_names(launcher_factory, tmp_path):
    from modules.config import ConfigManager
    launcher = launcher_factory(tmp_path / "cfg.json")
    launcher.saved_configs = {"zeta": {}, "alpha": {}, None: {}, "mu": {}}

    # Simulate a listbox by collecting inserted items.
    inserted = []
    launcher.config_listbox = MagicMock()
    launcher.config_listbox.curselection.return_value = ()
    launcher.config_listbox.insert.side_effect = lambda idx, val: inserted.append(val)

    cm = ConfigManager(launcher)
    cm.update_config_listbox()
    assert inserted == ["alpha", "mu", "zeta"]


def test_on_config_selected_single_sets_name(launcher_factory, tmp_path):
    from modules.config import ConfigManager
    launcher = launcher_factory(tmp_path / "cfg.json")
    launcher.config_listbox = MagicMock()
    launcher.config_listbox.curselection.return_value = (0,)
    launcher.config_listbox.get.return_value = "my_config"

    cm = ConfigManager(launcher)
    cm.on_config_selected()
    assert launcher.config_name.get() == "my_config"


def test_on_config_selected_multi_shows_count(launcher_factory, tmp_path):
    from modules.config import ConfigManager
    launcher = launcher_factory(tmp_path / "cfg.json")
    launcher.config_listbox = MagicMock()
    launcher.config_listbox.curselection.return_value = (0, 1, 2)
    cm = ConfigManager(launcher)
    cm.on_config_selected()
    assert "3" in launcher.config_name.get()
    assert "selected" in launcher.config_name.get()


def test_on_config_selected_empty_is_noop(launcher_factory, tmp_path):
    from modules.config import ConfigManager
    launcher = launcher_factory(tmp_path / "cfg.json")
    launcher.config_listbox = MagicMock()
    launcher.config_listbox.curselection.return_value = ()
    launcher.config_name.set("preserved")
    cm = ConfigManager(launcher)
    cm.on_config_selected()
    assert launcher.config_name.get() == "preserved"


# ---------------------------------------------------------------------------
# current_cfg
# ---------------------------------------------------------------------------

def test_current_cfg_produces_dict_with_expected_keys(launcher_factory, tmp_path):
    from modules.config import ConfigManager
    launcher = launcher_factory(tmp_path / "cfg.json")
    launcher.app_settings = {"selected_gpus": [], "gpu_order": []}
    cm = ConfigManager(launcher)
    cfg = cm.current_cfg()

    # Key properties required for the round-trip with load_saved_configs.
    required = {
        "llama_cpp_dir", "venv_dir", "model_path", "cache_type_k", "cache_type_v",
        "threads", "threads_batch", "batch_size", "ubatch_size", "n_gpu_layers",
        "no_mmap", "prio", "temperature", "min_p", "ctx_size", "seed", "flash_attn",
        "tensor_split", "main_gpu", "mlock", "no_kv_offload", "host", "port",
        "backend_selection", "ignore_eos", "n_predict", "cpu_moe", "n_cpu_moe",
        "parallel", "mmproj_enabled", "fit_enabled", "template_source",
        "custom_parameters", "gpu_indices", "gpu_order",
    }
    missing = required - set(cfg.keys())
    assert not missing, f"Missing keys in current_cfg(): {missing}"
    # Env var config merged in.
    assert "environmental_variables" in cfg


def test_current_cfg_preserves_custom_parameters_list(launcher_factory, tmp_path):
    from modules.config import ConfigManager
    launcher = launcher_factory(tmp_path / "cfg.json")
    launcher.custom_parameters_list = ["--a", "--b=1"]
    launcher.app_settings = {"selected_gpus": [], "gpu_order": []}
    cm = ConfigManager(launcher)
    cfg = cm.current_cfg()
    assert cfg["custom_parameters"] == ["--a", "--b=1"]


def test_current_cfg_syncs_selected_gpus_from_gpu_vars(launcher_factory, tmp_path):
    from modules.config import ConfigManager
    launcher = launcher_factory(tmp_path / "cfg.json")
    launcher.app_settings = {"selected_gpus": [], "gpu_order": [0, 2]}
    # Three GPUs, only 0 and 2 are checked.
    launcher.gpu_vars = [_FakeVar(True), _FakeVar(False), _FakeVar(True)]
    cm = ConfigManager(launcher)
    cfg = cm.current_cfg()
    assert cfg["gpu_indices"] == [0, 2]
    assert cfg["gpu_order"] == [0, 2]


# ---------------------------------------------------------------------------
# save_configuration (high-level: builds name, stashes current_cfg)
# ---------------------------------------------------------------------------

def test_save_configuration_uses_supplied_name(launcher_factory, tmp_path):
    from modules.config import ConfigManager
    launcher = launcher_factory(tmp_path / "cfg.json")
    launcher.app_settings = {"selected_gpus": [], "gpu_order": []}
    launcher.config_name.set("my_named_cfg")
    launcher.config_listbox = MagicMock()
    launcher.config_listbox.curselection.return_value = ()

    cm = ConfigManager(launcher)
    with patch("modules.config.messagebox"):
        cm.save_configuration()
    assert "my_named_cfg" in launcher.saved_configs
    # And it was actually persisted to disk.
    saved = json.loads((tmp_path / "cfg.json").read_text("utf-8"))
    assert "my_named_cfg" in saved["configs"]


# ---------------------------------------------------------------------------
# Real ``config/*.json`` files
# ---------------------------------------------------------------------------

def test_real_llama_cpp_launcher_configs_json_loads(
    launcher_factory, project_config_dir
):
    """The shipped config file should parse without errors."""
    from modules.config import ConfigManager
    cfg_file = project_config_dir / "llama_cpp_launcher_configs.json"
    if not cfg_file.exists():
        pytest.skip("live config file missing in this checkout")

    # Copy into a tmp path so load_saved_configs can mutate app_settings.
    launcher = launcher_factory(cfg_file)
    cm = ConfigManager(launcher)
    with patch("modules.config.messagebox"):
        cm.load_saved_configs()
    assert cm.configs_loaded_successfully is True
    assert isinstance(launcher.saved_configs, dict)
    assert launcher.saved_configs, "live config file should contain at least one config"


def test_real_chat_templates_json_is_valid_mapping(project_config_dir):
    tpl_file = project_config_dir / "chat_templates.json"
    if not tpl_file.exists():
        pytest.skip("chat_templates.json missing in this checkout")
    data = json.loads(tpl_file.read_text("utf-8"))
    assert isinstance(data, dict)
    # Every value is a non-empty template string.
    for name, template in data.items():
        assert isinstance(name, str) and name
        assert isinstance(template, str) and template


def test_real_config_every_saved_config_has_required_keys(project_config_dir):
    """Sanity check: every saved config carries the minimum keys the launcher
    expects. Protects against accidental schema regressions."""
    cfg_file = project_config_dir / "llama_cpp_launcher_configs.json"
    if not cfg_file.exists():
        pytest.skip("live config file missing")
    data = json.loads(cfg_file.read_text("utf-8"))
    required = {"model_path", "ctx_size", "host", "port"}
    for name, cfg in data.get("configs", {}).items():
        missing = required - set(cfg.keys())
        assert not missing, f"config {name!r} missing keys: {missing}"


# ---------------------------------------------------------------------------
# generate_default_config_name (light-touch: bypasses recursion on traces)
# ---------------------------------------------------------------------------

def test_generate_default_config_name_empty_returns_default(launcher_factory, tmp_path):
    from modules.config import ConfigManager
    launcher = launcher_factory(tmp_path / "cfg.json")
    # Integer-backed Tk vars used by generate_default_config_name.
    launcher.n_gpu_layers_int = _FakeVar(0)
    launcher.max_gpu_layers = _FakeVar(0)
    # Model path is empty -> parts starts with "default".
    cm = ConfigManager(launcher)
    name = cm.generate_default_config_name()
    assert name.startswith("default")


def test_generate_default_config_name_respects_custom_name(launcher_factory, tmp_path):
    from modules.config import ConfigManager
    launcher = launcher_factory(tmp_path / "cfg.json")
    launcher.n_gpu_layers_int = _FakeVar(0)
    launcher.max_gpu_layers = _FakeVar(0)
    launcher.config_name.set("MyHandPickedName")  # Custom-looking name.
    cm = ConfigManager(launcher)
    cm.generate_default_config_name()
    assert launcher.config_name.get() == "MyHandPickedName"


def test_generate_default_config_name_appends_non_default_params(launcher_factory, tmp_path):
    from modules.config import ConfigManager
    launcher = launcher_factory(tmp_path / "cfg.json")
    launcher.n_gpu_layers_int = _FakeVar(16)
    launcher.max_gpu_layers = _FakeVar(32)
    launcher.model_path.set("/tmp/some-model.gguf")
    launcher.threads.set("16")  # Different from logical_cores=8.
    cm = ConfigManager(launcher)
    name = cm.generate_default_config_name()
    assert "gpu=16" in name
    assert "th=16" in name


# ---------------------------------------------------------------------------
# export_configurations / import_configurations (UI layer mocked)
# ---------------------------------------------------------------------------

def test_export_configurations_writes_valid_json(launcher_factory, tmp_path):
    from modules.config import ConfigManager
    launcher = launcher_factory(tmp_path / "cfg.json")
    launcher.saved_configs = {"a": {"model_path": "/x"}, "b": {"model_path": "/y"}}
    launcher.config_listbox = MagicMock()
    launcher.config_listbox.curselection.return_value = (0, 1)
    launcher.config_listbox.get.side_effect = lambda idx: ["a", "b"][idx]

    export_path = tmp_path / "export.json"
    cm = ConfigManager(launcher)
    with patch("modules.config.filedialog.asksaveasfilename", return_value=str(export_path)), \
         patch("modules.config.messagebox"):
        cm.export_configurations()

    data = json.loads(export_path.read_text("utf-8"))
    assert data["configs"] == {"a": {"model_path": "/x"}, "b": {"model_path": "/y"}}
    assert data["export_info"]["config_count"] == 2


def test_import_configurations_accepts_export_format(launcher_factory, tmp_path):
    from modules.config import ConfigManager
    import_payload = {
        "export_info": {"exported_at": "now", "config_count": 1},
        "configs": {"imported": {"model_path": "/i"}},
    }
    import_file = tmp_path / "to_import.json"
    import_file.write_text(json.dumps(import_payload))

    launcher = launcher_factory(tmp_path / "cfg.json")
    launcher.saved_configs = {}
    launcher.config_listbox = MagicMock()
    launcher.config_listbox.curselection.return_value = ()
    launcher._update_config_listbox = MagicMock()

    cm = ConfigManager(launcher)
    with patch("modules.config.filedialog.askopenfilename", return_value=str(import_file)), \
         patch("modules.config.messagebox.askyesno", return_value=True), \
         patch("modules.config.messagebox.showinfo"), \
         patch("modules.config.messagebox.showerror"):
        cm.import_configurations()
    assert "imported" in launcher.saved_configs


def test_import_configurations_accepts_legacy_format(launcher_factory, tmp_path):
    from modules.config import ConfigManager
    # A "legacy" format has config entries at the top level.
    legacy_payload = {
        "my_cfg": {"model_path": "/m", "ctx_size": 2048},
        "not_a_config": "just a string",
    }
    import_file = tmp_path / "legacy.json"
    import_file.write_text(json.dumps(legacy_payload))

    launcher = launcher_factory(tmp_path / "cfg.json")
    launcher.saved_configs = {}
    launcher.config_listbox = MagicMock()
    launcher.config_listbox.curselection.return_value = ()
    launcher._update_config_listbox = MagicMock()

    cm = ConfigManager(launcher)
    with patch("modules.config.filedialog.askopenfilename", return_value=str(import_file)), \
         patch("modules.config.messagebox.askyesno", return_value=True), \
         patch("modules.config.messagebox.showinfo"), \
         patch("modules.config.messagebox.showerror"):
        cm.import_configurations()
    assert "my_cfg" in launcher.saved_configs
    assert "not_a_config" not in launcher.saved_configs


def test_import_configurations_rejects_malformed_json(launcher_factory, tmp_path):
    from modules.config import ConfigManager
    import_file = tmp_path / "bad.json"
    import_file.write_text("{not json")
    launcher = launcher_factory(tmp_path / "cfg.json")
    launcher.saved_configs = {}

    cm = ConfigManager(launcher)
    mock_error = MagicMock()
    with patch("modules.config.filedialog.askopenfilename", return_value=str(import_file)), \
         patch("modules.config.messagebox.showerror", mock_error):
        cm.import_configurations()
    # Error was reported AND the in-memory store was not polluted.
    assert mock_error.called
    assert launcher.saved_configs == {}


def test_import_configurations_no_valid_configs_reports_error(launcher_factory, tmp_path):
    from modules.config import ConfigManager
    bad_payload = {"foo": "bar", "baz": 42}
    import_file = tmp_path / "irrelevant.json"
    import_file.write_text(json.dumps(bad_payload))
    launcher = launcher_factory(tmp_path / "cfg.json")
    launcher.saved_configs = {}

    cm = ConfigManager(launcher)
    mock_error = MagicMock()
    with patch("modules.config.filedialog.askopenfilename", return_value=str(import_file)), \
         patch("modules.config.messagebox.showerror", mock_error):
        cm.import_configurations()
    assert mock_error.called
    assert launcher.saved_configs == {}


def test_import_configurations_user_cancel_is_noop(launcher_factory, tmp_path):
    from modules.config import ConfigManager
    launcher = launcher_factory(tmp_path / "cfg.json")
    launcher.saved_configs = {"existing": {"model_path": "/x"}}
    cm = ConfigManager(launcher)
    with patch("modules.config.filedialog.askopenfilename", return_value=""):
        cm.import_configurations()
    assert launcher.saved_configs == {"existing": {"model_path": "/x"}}


def test_export_configurations_user_cancel_is_noop(launcher_factory, tmp_path):
    from modules.config import ConfigManager
    launcher = launcher_factory(tmp_path / "cfg.json")
    launcher.saved_configs = {"one": {"model_path": "/x"}}
    launcher.config_listbox = MagicMock()
    launcher.config_listbox.curselection.return_value = (0,)
    launcher.config_listbox.get.return_value = "one"
    cm = ConfigManager(launcher)
    with patch("modules.config.filedialog.asksaveasfilename", return_value=""), \
         patch("modules.config.messagebox"):
        cm.export_configurations()
    # No file was written (tmp_path is empty).
    assert list(tmp_path.glob("*.json")) == []


# ---------------------------------------------------------------------------
# save_configs: fallback on write failure
# ---------------------------------------------------------------------------

def test_save_configs_reports_error_on_fallback_same_path(launcher_factory, tmp_path):
    """When writing fails and the fallback is the same path, the user is notified."""
    from modules.config import ConfigManager
    cfg_path = tmp_path / "cfg.json"
    launcher = launcher_factory(cfg_path)
    launcher.saved_configs = {"ok": {"x": 1}}
    cm = ConfigManager(launcher)

    # Force the initial write to blow up, and force get_config_path to return the same
    # path so the fallback path can't recover.
    with patch.object(Path, "write_text", side_effect=OSError("disk full")), \
         patch.object(ConfigManager, "get_config_path", return_value=cfg_path), \
         patch("modules.config.messagebox.showerror") as mock_err, \
         patch("modules.config.messagebox.showwarning"):
        cm.save_configs()
    assert mock_err.called
