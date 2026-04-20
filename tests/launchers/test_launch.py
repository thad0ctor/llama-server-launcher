"""Regression tests for modules.launch.LaunchManager.

Covers:
  - add_arg() behaviour across BooleanVar / bool / str / int / float and defaults.
  - _get_llama_cpp_executable_names() / _get_ik_llama_executable_names()
    Windows vs POSIX switching.
  - _find_server_executable() across several common subdirectory layouts.
  - _get_launchers_dir() directory creation.
  - build_cmd() under a comprehensive mock launcher - happy paths and the
    error paths that return None (missing backend dir / missing model file /
    invalid backend dir / executable not found).
  - Argument gating rules: n-gpu-layers, host/port/ctx-size/seed default
    omission, flash-attn backend differences, fit parameters only under
    llama.cpp, KV cache type suppression when ik_llama uses -ctk/-ctv,
    custom parameters, mmproj auto-detection, chat-template handling.
  - save_sh_script() / save_ps1_script() script content - path quoting
    (including spaces), exec bit, env vars, chat-template quoting,
    CUDA_VISIBLE_DEVICES emission, venv activation branches.
"""

from __future__ import annotations

import os
import stat
import sys
import tkinter as tk
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Ensure repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --- tk root bootstrap ------------------------------------------------------
@pytest.fixture(scope="module")
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - env dependent
        pytest.skip(f"tkinter unavailable: {exc}")
    root.withdraw()
    yield root
    try:
        root.destroy()
    except Exception:
        pass


def _mk(cls, value):
    v = cls()
    v.set(value)
    return v


@pytest.fixture()
def built_tree(tmp_path):
    """Creates a fake llama.cpp build tree with an executable at build/bin/
    and a dummy model.gguf. Returns paths dict."""
    base = tmp_path / "llama_cpp_root"
    bin_dir = base / "build" / "bin"
    bin_dir.mkdir(parents=True)

    exe_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    exe = bin_dir / exe_name
    exe.write_text("#!/bin/sh\necho fake llama-server\n")
    try:
        os.chmod(exe, 0o755)
    except OSError:
        pass

    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF" + b"\x00" * 16)

    return {"base": base, "exe": exe, "model": model, "tmp_path": tmp_path}


@pytest.fixture()
def launcher_mock(tk_root, built_tree):
    """Fully-populated launcher stand-in.

    All tk vars have been pre-set to values matching llama.cpp defaults so
    build_cmd() produces a minimal command - individual tests tweak the vars
    they care about.
    """
    paths = built_tree
    m = MagicMock()

    # Backend + dirs
    m.backend_selection = _mk(tk.StringVar, "llama.cpp")
    m.llama_cpp_dir = _mk(tk.StringVar, str(paths["base"]))
    m.ik_llama_dir = _mk(tk.StringVar, str(paths["base"]))
    m.venv_dir = _mk(tk.StringVar, "")

    # Model
    m.model_path = _mk(tk.StringVar, str(paths["model"]))
    m.mmproj_enabled = _mk(tk.BooleanVar, False)
    m.selected_mmproj_path = _mk(tk.StringVar, "")

    # KV cache
    m.cache_type_k = _mk(tk.StringVar, "f16")
    m.cache_type_v = _mk(tk.StringVar, "f16")

    # Threads / batching
    m.threads = _mk(tk.StringVar, "")  # empty => omit
    m.logical_cores = 8
    m.threads_batch = _mk(tk.StringVar, "")
    m.batch_size = _mk(tk.StringVar, "")
    m.ubatch_size = _mk(tk.StringVar, "")

    # Context / sampling
    m.ctx_size = _mk(tk.IntVar, 2048)
    m.seed = _mk(tk.StringVar, "-1")
    m.temperature = _mk(tk.StringVar, "0.8")
    m.min_p = _mk(tk.StringVar, "0.05")

    # GPU
    m.tensor_split = _mk(tk.StringVar, "")
    m.n_gpu_layers = _mk(tk.StringVar, "0")
    m.main_gpu = _mk(tk.StringVar, "0")
    m.flash_attn = _mk(tk.BooleanVar, False)

    # Fit
    m.fit_enabled = _mk(tk.BooleanVar, False)
    m.fit_ctx = _mk(tk.StringVar, "")
    m.fit_target = _mk(tk.StringVar, "")

    # Memory
    m.no_mmap = _mk(tk.BooleanVar, False)
    m.mlock = _mk(tk.BooleanVar, False)
    m.no_kv_offload = _mk(tk.BooleanVar, False)

    # Perf
    m.prio = _mk(tk.StringVar, "0")
    m.parallel = _mk(tk.StringVar, "1")

    # MoE
    m.cpu_moe = _mk(tk.BooleanVar, False)
    m.n_cpu_moe = _mk(tk.StringVar, "")

    # Gen
    m.ignore_eos = _mk(tk.BooleanVar, False)
    m.n_predict = _mk(tk.StringVar, "-1")

    # Net
    m.host = _mk(tk.StringVar, "127.0.0.1")
    m.port = _mk(tk.StringVar, "8080")

    # Template / jinja
    m.template_source = _mk(tk.StringVar, "default")
    m.current_template_display = _mk(tk.StringVar, "")
    m.jinja_enabled = _mk(tk.BooleanVar, False)

    # Custom params + GPU info
    m.custom_parameters_list = []
    m.get_ordered_selected_gpus = MagicMock(return_value=[])
    m.gpu_info = {"device_count": 0}

    # Listbox behaviour: no row selected
    m.model_listbox.curselection = MagicMock(return_value=())
    m.found_models = {}

    # ik_llama subtab - get_ik_llama_flags returns nothing by default.
    m.ik_llama_tab.get_ik_llama_flags = MagicMock(return_value=[])

    # Env vars manager - nothing enabled.
    m.env_vars_manager.get_enabled_env_vars = MagicMock(return_value={})

    return m


@pytest.fixture()
def manager(launcher_mock):
    from modules.launch import LaunchManager
    return LaunchManager(launcher_mock)


# ============================================================================
# add_arg
# ============================================================================


class TestAddArg:
    def test_bool_true_appends_flag_name(self, manager):
        cmd = []
        manager.add_arg(cmd, "--flag", True)
        assert cmd == ["--flag"]

    def test_bool_false_omits(self, manager):
        cmd = []
        manager.add_arg(cmd, "--flag", False)
        assert cmd == []

    def test_booleanvar_true(self, manager, tk_root):
        cmd = []
        manager.add_arg(cmd, "--bv", _mk(tk.BooleanVar, True))
        assert cmd == ["--bv"]

    def test_booleanvar_false(self, manager, tk_root):
        cmd = []
        manager.add_arg(cmd, "--bv", _mk(tk.BooleanVar, False))
        assert cmd == []

    def test_string_different_from_default(self, manager):
        cmd = []
        manager.add_arg(cmd, "--x", "5", "10")
        assert cmd == ["--x", "5"]

    def test_string_equals_default_omitted(self, manager):
        cmd = []
        manager.add_arg(cmd, "--x", "5", "5")
        assert cmd == []

    def test_empty_value_always_omitted(self, manager):
        cmd = []
        manager.add_arg(cmd, "--x", "", "0")
        assert cmd == []

    def test_whitespace_only_value_treated_as_empty(self, manager):
        cmd = []
        manager.add_arg(cmd, "--x", "   ", "0")
        assert cmd == []

    def test_no_default_value_means_add_when_non_empty(self, manager):
        cmd = []
        manager.add_arg(cmd, "--x", "anything")
        assert cmd == ["--x", "anything"]

    def test_int_value_coerced_to_string(self, manager):
        cmd = []
        manager.add_arg(cmd, "--x", 42, "0")
        assert cmd == ["--x", "42"]

    def test_float_value_coerced_to_string(self, manager):
        cmd = []
        manager.add_arg(cmd, "--x", 0.25, "0")
        assert cmd == ["--x", "0.25"]

    def test_value_stripped_before_comparison(self, manager):
        # "  5  " should compare equal to "5"
        cmd = []
        manager.add_arg(cmd, "--x", "  5  ", "5")
        assert cmd == []


# ============================================================================
# Executable name platform branching
# ============================================================================


class TestExecutableNames:
    def test_llama_cpp_win32(self, manager):
        with patch.object(sys, "platform", "win32"):
            names = manager._get_llama_cpp_executable_names()
        assert names == [
            "llama-server.exe",
            "server.exe",
            "llama-cpp-python-server.exe",
        ]

    def test_llama_cpp_linux(self, manager):
        with patch.object(sys, "platform", "linux"):
            names = manager._get_llama_cpp_executable_names()
        assert names == ["llama-server", "server", "llama-cpp-python-server"]

    def test_llama_cpp_darwin(self, manager):
        # Assert the exact expected names — the old assertion ``".exe" not
        # in "".join(names)`` passed on an empty list too, so a regression
        # that dropped the names would have slipped through.
        with patch.object(sys, "platform", "darwin"):
            names = manager._get_llama_cpp_executable_names()
        assert names == ["llama-server", "server", "llama-cpp-python-server"]

    def test_ik_llama_win32(self, manager):
        with patch.object(sys, "platform", "win32"):
            names = manager._get_ik_llama_executable_names()
        assert names == [
            "llama-server.exe",
            "ik-llama-server.exe",
            "ik_llama_server.exe",
        ]

    def test_ik_llama_linux(self, manager):
        with patch.object(sys, "platform", "linux"):
            names = manager._get_ik_llama_executable_names()
        assert names == ["llama-server", "ik-llama-server", "ik_llama_server"]


# ============================================================================
# _find_server_executable
# ============================================================================


class TestFindServerExecutable:
    def test_finds_in_build_bin(self, manager, tmp_path):
        base = tmp_path / "root"
        (base / "build" / "bin").mkdir(parents=True)
        exe_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
        exe = base / "build" / "bin" / exe_name
        exe.write_text("fake")
        os.chmod(exe, 0o755)
        result = manager._find_server_executable(base, "llama.cpp")
        assert result is not None
        assert result.resolve() == exe.resolve()

    def test_finds_in_build_bin_release(self, manager, tmp_path):
        base = tmp_path / "root"
        (base / "build" / "bin" / "Release").mkdir(parents=True)
        exe_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
        exe = base / "build" / "bin" / "Release" / exe_name
        exe.write_text("fake")
        os.chmod(exe, 0o755)
        result = manager._find_server_executable(base, "llama.cpp")
        assert result is not None
        assert result.resolve() == exe.resolve()

    def test_finds_in_root(self, manager, tmp_path):
        base = tmp_path / "root"
        base.mkdir()
        exe_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
        exe = base / exe_name
        exe.write_text("fake")
        os.chmod(exe, 0o755)
        result = manager._find_server_executable(base, "llama.cpp")
        assert result is not None
        assert result.resolve() == exe.resolve()

    def test_returns_none_if_missing(self, manager, tmp_path):
        base = tmp_path / "empty_root"
        base.mkdir()
        assert manager._find_server_executable(base, "llama.cpp") is None

    def test_ik_llama_backend_finds_ik_specific_name(self, manager, tmp_path):
        base = tmp_path / "root"
        bin_dir = base / "build" / "bin"
        bin_dir.mkdir(parents=True)
        exe_name = (
            "ik-llama-server.exe" if sys.platform == "win32" else "ik-llama-server"
        )
        exe = bin_dir / exe_name
        exe.write_text("fake")
        os.chmod(exe, 0o755)
        result = manager._find_server_executable(base, "ik_llama")
        assert result is not None
        assert result.name == exe_name

    def test_llama_cpp_backend_does_not_pick_up_ik_specific_name(
        self, manager, tmp_path
    ):
        # llama.cpp exe list should not include ik-llama-server, so a tree that
        # ONLY contains ik-llama-server must not be matched.
        base = tmp_path / "root"
        bin_dir = base / "build" / "bin"
        bin_dir.mkdir(parents=True)
        exe_name = (
            "ik-llama-server.exe" if sys.platform == "win32" else "ik-llama-server"
        )
        (bin_dir / exe_name).write_text("fake")
        os.chmod(bin_dir / exe_name, 0o755)
        assert manager._find_server_executable(base, "llama.cpp") is None


# ============================================================================
# _get_launchers_dir
# ============================================================================


class TestLaunchersDir:
    def test_returns_existing_launchers_dir(self, manager):
        # The repo ships with a launchers/ directory already. The method
        # should return it without prompting on success.
        d = manager._get_launchers_dir()
        assert d.name == "launchers"
        assert d.is_dir()


# ============================================================================
# build_cmd: happy path + gating
# ============================================================================


class TestBuildCmdHappyPath:
    def test_minimal_command_emitted(self, manager, launcher_mock, built_tree):
        """With everything at defaults, only exe + -m model + --fit off
        should be emitted (since all other args either match defaults or
        are empty)."""
        cmd = manager.build_cmd()
        assert cmd is not None
        assert cmd[0] == str(built_tree["exe"].resolve())
        assert "-m" in cmd
        # Model path comes right after -m
        m_idx = cmd.index("-m")
        assert Path(cmd[m_idx + 1]).resolve() == built_tree["model"].resolve()
        # Chat template NOT included under "default" source
        assert "--chat-template" not in cmd
        # Under llama.cpp, --fit off is always emitted when fit_enabled=False
        assert "--fit" in cmd
        assert cmd[cmd.index("--fit") + 1] == "off"

    def test_ik_llama_backend_skips_fit_flags(self, manager, launcher_mock, built_tree):
        launcher_mock.backend_selection.set("ik_llama")
        cmd = manager.build_cmd()
        assert cmd is not None
        assert "--fit" not in cmd, "ik_llama backend should not emit --fit"

    def test_ctx_size_default_omitted(self, manager, launcher_mock):
        launcher_mock.ctx_size.set(2048)
        cmd = manager.build_cmd()
        assert "--ctx-size" not in cmd

    def test_ctx_size_non_default_emitted(self, manager, launcher_mock):
        launcher_mock.ctx_size.set(4096)
        cmd = manager.build_cmd()
        assert "--ctx-size" in cmd
        assert cmd[cmd.index("--ctx-size") + 1] == "4096"

    def test_host_port_defaults_omitted(self, manager, launcher_mock):
        launcher_mock.host.set("127.0.0.1")
        launcher_mock.port.set("8080")
        cmd = manager.build_cmd()
        assert "--host" not in cmd
        assert "--port" not in cmd

    def test_host_port_non_defaults_emitted(self, manager, launcher_mock):
        launcher_mock.host.set("0.0.0.0")
        launcher_mock.port.set("9090")
        cmd = manager.build_cmd()
        assert "--host" in cmd and cmd[cmd.index("--host") + 1] == "0.0.0.0"
        assert "--port" in cmd and cmd[cmd.index("--port") + 1] == "9090"

    def test_n_gpu_layers_zero_omitted(self, manager, launcher_mock):
        launcher_mock.n_gpu_layers.set("0")
        cmd = manager.build_cmd()
        assert "--n-gpu-layers" not in cmd

    def test_n_gpu_layers_nonzero_emitted(self, manager, launcher_mock):
        launcher_mock.n_gpu_layers.set("32")
        cmd = manager.build_cmd()
        assert "--n-gpu-layers" in cmd
        assert cmd[cmd.index("--n-gpu-layers") + 1] == "32"

    def test_tensor_split_empty_omitted(self, manager, launcher_mock):
        launcher_mock.tensor_split.set("")
        cmd = manager.build_cmd()
        assert "--tensor-split" not in cmd

    def test_tensor_split_emitted(self, manager, launcher_mock):
        launcher_mock.tensor_split.set("0.5,0.5")
        cmd = manager.build_cmd()
        assert "--tensor-split" in cmd
        assert cmd[cmd.index("--tensor-split") + 1] == "0.5,0.5"

    def test_flash_attn_llama_cpp_uses_on(self, manager, launcher_mock):
        launcher_mock.flash_attn.set(True)
        launcher_mock.backend_selection.set("llama.cpp")
        cmd = manager.build_cmd()
        idx = cmd.index("--flash-attn")
        assert cmd[idx + 1] == "on"

    def test_flash_attn_ik_llama_bare_flag(self, manager, launcher_mock):
        launcher_mock.flash_attn.set(True)
        launcher_mock.backend_selection.set("ik_llama")
        cmd = manager.build_cmd()
        # ik_llama uses bare --flash-attn (no "on" / "off" follower).
        idx = cmd.index("--flash-attn")
        # Either we're at the end, or the next token is another flag.
        if idx + 1 < len(cmd):
            # Chained-comparison trap: `a in b is False` parses as
            # `(a in b) and (b is False)` — always False. Use an explicit
            # `not in` instead.
            assert cmd[idx + 1] not in ("on", "off")
            assert cmd[idx + 1].startswith("-")

    def test_cache_type_k_default_omitted(self, manager, launcher_mock):
        launcher_mock.cache_type_k.set("f16")
        cmd = manager.build_cmd()
        assert "--cache-type-k" not in cmd

    def test_cache_type_k_emits_both_k_and_v(self, manager, launcher_mock):
        launcher_mock.cache_type_k.set("q8_0")
        cmd = manager.build_cmd()
        assert "--cache-type-k" in cmd
        assert cmd[cmd.index("--cache-type-k") + 1] == "q8_0"
        # When K is set non-default, V is also forcibly set to match K.
        assert "--cache-type-v" in cmd
        assert cmd[cmd.index("--cache-type-v") + 1] == "q8_0"

    def test_ik_llama_ctk_suppresses_standard_cache_flags(
        self, manager, launcher_mock
    ):
        """When ik_llama's -ctk/-ctv flags are used, build_cmd should
        NOT also add the standard --cache-type-k even if the UI value is
        non-default."""
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.cache_type_k.set("q8_0")
        launcher_mock.ik_llama_tab.get_ik_llama_flags.return_value = [
            "-ctk", "q8_0",
            "-ctv", "q8_0",
        ]
        cmd = manager.build_cmd()
        assert "--cache-type-k" not in cmd
        assert "--cache-type-v" not in cmd
        # But the ik_llama flags themselves are appended.
        assert "-ctk" in cmd
        assert "-ctv" in cmd

    def test_jinja_flag(self, manager, launcher_mock):
        launcher_mock.jinja_enabled.set(True)
        cmd = manager.build_cmd()
        assert "--jinja" in cmd

    def test_jinja_not_set_when_disabled(self, manager, launcher_mock):
        launcher_mock.jinja_enabled.set(False)
        cmd = manager.build_cmd()
        assert "--jinja" not in cmd

    def test_custom_parameters_applied(self, manager, launcher_mock):
        launcher_mock.custom_parameters_list = ['--foo bar', '--baz "value with spaces"']
        cmd = manager.build_cmd()
        # shlex splits the single-string parameter into args.
        assert "--foo" in cmd
        assert "bar" in cmd
        assert "--baz" in cmd
        assert "value with spaces" in cmd

    def test_chat_template_predefined_adds_arg(self, manager, launcher_mock):
        launcher_mock.template_source.set("predefined")
        launcher_mock.current_template_display.set("chatml")
        cmd = manager.build_cmd()
        assert "--chat-template" in cmd
        assert cmd[cmd.index("--chat-template") + 1] == "chatml"

    def test_chat_template_custom_with_empty_string_omits(
        self, manager, launcher_mock
    ):
        launcher_mock.template_source.set("custom")
        launcher_mock.current_template_display.set("")
        cmd = manager.build_cmd()
        assert "--chat-template" not in cmd

    def test_mlock_flag(self, manager, launcher_mock):
        launcher_mock.mlock.set(True)
        cmd = manager.build_cmd()
        assert "--mlock" in cmd

    def test_no_mmap_flag(self, manager, launcher_mock):
        launcher_mock.no_mmap.set(True)
        cmd = manager.build_cmd()
        assert "--no-mmap" in cmd

    def test_parallel_default_omitted(self, manager, launcher_mock):
        launcher_mock.parallel.set("1")
        cmd = manager.build_cmd()
        assert "--parallel" not in cmd

    def test_parallel_nondefault_emitted(self, manager, launcher_mock):
        launcher_mock.parallel.set("4")
        cmd = manager.build_cmd()
        assert "--parallel" in cmd
        assert cmd[cmd.index("--parallel") + 1] == "4"

    def test_fit_enabled_emits_on_and_ctx_target(self, manager, launcher_mock):
        launcher_mock.fit_enabled.set(True)
        launcher_mock.fit_ctx.set("8192")
        launcher_mock.fit_target.set("2048")
        cmd = manager.build_cmd()
        assert cmd[cmd.index("--fit") + 1] == "on"
        assert "--fit-ctx" in cmd and cmd[cmd.index("--fit-ctx") + 1] == "8192"
        assert "--fit-target" in cmd and cmd[cmd.index("--fit-target") + 1] == "2048"

    def test_fit_enabled_with_default_ctx_and_target_omits_sub_flags(
        self, manager, launcher_mock
    ):
        launcher_mock.fit_enabled.set(True)
        launcher_mock.fit_ctx.set("4096")  # matches llama.cpp default
        launcher_mock.fit_target.set("1024")  # matches default
        cmd = manager.build_cmd()
        assert cmd[cmd.index("--fit") + 1] == "on"
        assert "--fit-ctx" not in cmd
        assert "--fit-target" not in cmd

    def test_ik_llama_flags_appended(self, manager, launcher_mock):
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.ik_llama_tab.get_ik_llama_flags.return_value = ["-rtr", "-fmoe"]
        cmd = manager.build_cmd()
        assert "-rtr" in cmd
        assert "-fmoe" in cmd

    def test_ignore_eos_and_n_predict(self, manager, launcher_mock):
        launcher_mock.ignore_eos.set(True)
        launcher_mock.n_predict.set("128")
        cmd = manager.build_cmd()
        assert "--ignore-eos" in cmd
        assert "--n-predict" in cmd
        assert cmd[cmd.index("--n-predict") + 1] == "128"


# ============================================================================
# build_cmd: error paths
# ============================================================================


class TestBuildCmdErrors:
    def test_missing_backend_dir_returns_none(self, manager, launcher_mock):
        launcher_mock.llama_cpp_dir.set("")
        with patch("modules.launch.messagebox") as mb:
            cmd = manager.build_cmd()
        assert cmd is None
        assert mb.showerror.called

    def test_invalid_backend_dir_returns_none(self, manager, launcher_mock, tmp_path):
        launcher_mock.llama_cpp_dir.set(str(tmp_path / "does_not_exist"))
        with patch("modules.launch.messagebox") as mb:
            cmd = manager.build_cmd()
        assert cmd is None
        assert mb.showerror.called

    def test_backend_dir_is_file_returns_none(self, manager, launcher_mock, tmp_path):
        f = tmp_path / "not_a_dir"
        f.write_text("")
        launcher_mock.llama_cpp_dir.set(str(f))
        with patch("modules.launch.messagebox") as mb:
            cmd = manager.build_cmd()
        assert cmd is None
        assert mb.showerror.called

    def test_executable_not_found_returns_none(self, manager, launcher_mock, tmp_path):
        # Valid dir but no executable anywhere in search paths.
        empty = tmp_path / "empty"
        empty.mkdir()
        launcher_mock.llama_cpp_dir.set(str(empty))
        with patch("modules.launch.messagebox") as mb:
            cmd = manager.build_cmd()
        assert cmd is None
        assert mb.showerror.called
        # Assert the structured showerror args directly — the prior
        # ``"llama" in str(call_args).lower()`` fallback was satisfied by
        # almost any error message in this project (every error references
        # "llama.cpp" or the llama-server exe name), so a regression
        # showing an unrelated error under the same title would have
        # slipped through.
        title, message = mb.showerror.call_args.args[:2]
        assert title == "Executable Not Found"
        # The message must point at the search location so the user knows
        # where to look.
        assert str(empty) in message

    def test_missing_model_returns_none(self, manager, launcher_mock):
        launcher_mock.model_path.set("")
        with patch("modules.launch.messagebox") as mb:
            cmd = manager.build_cmd()
        assert cmd is None
        assert mb.showerror.called

    def test_invalid_model_path_returns_none(self, manager, launcher_mock, tmp_path):
        launcher_mock.model_path.set(str(tmp_path / "no_such_model.gguf"))
        with patch("modules.launch.messagebox") as mb:
            cmd = manager.build_cmd()
        assert cmd is None
        assert mb.showerror.called

    def test_ik_llama_backend_with_empty_ik_dir_returns_none(
        self, manager, launcher_mock
    ):
        launcher_mock.backend_selection.set("ik_llama")
        launcher_mock.ik_llama_dir.set("")
        with patch("modules.launch.messagebox") as mb:
            cmd = manager.build_cmd()
        assert cmd is None
        assert mb.showerror.called


# ============================================================================
# build_cmd: mmproj behaviour
# ============================================================================


class TestBuildCmdMmproj:
    def test_mmproj_auto_detected_from_model_dir(
        self, manager, launcher_mock, built_tree
    ):
        model_dir = built_tree["model"].parent
        mmproj_file = model_dir / "mmproj-model-f16.gguf"
        mmproj_file.write_bytes(b"GGUF\x00")

        launcher_mock.mmproj_enabled.set(True)
        cmd = manager.build_cmd()
        assert "--mmproj" in cmd
        assert Path(cmd[cmd.index("--mmproj") + 1]).name == "mmproj-model-f16.gguf"

    def test_mmproj_explicit_selection_wins_over_autodetect(
        self, manager, launcher_mock, built_tree
    ):
        model_dir = built_tree["model"].parent
        # Both a default and an explicit file
        (model_dir / "mmproj-default.gguf").write_bytes(b"GGUF")
        explicit = model_dir / "picked-mmproj.gguf"
        explicit.write_bytes(b"GGUF")

        launcher_mock.mmproj_enabled.set(True)
        launcher_mock.selected_mmproj_path.set(str(explicit))

        cmd = manager.build_cmd()
        assert "--mmproj" in cmd
        assert Path(cmd[cmd.index("--mmproj") + 1]).name == "picked-mmproj.gguf"

    def test_mmproj_disabled_skips_flag(self, manager, launcher_mock, built_tree):
        model_dir = built_tree["model"].parent
        (model_dir / "mmproj-file.gguf").write_bytes(b"GGUF")
        launcher_mock.mmproj_enabled.set(False)
        cmd = manager.build_cmd()
        assert "--mmproj" not in cmd

    def test_mmproj_none_found_does_not_add_flag(
        self, manager, launcher_mock, built_tree
    ):
        launcher_mock.mmproj_enabled.set(True)
        cmd = manager.build_cmd()
        assert "--mmproj" not in cmd


# ============================================================================
# save_sh_script
# ============================================================================


class TestSaveShScript:
    def _write_and_read(self, manager, launcher_mock, out_path):
        with patch("modules.launch.filedialog") as fd, patch(
            "modules.launch.messagebox"
        ):
            fd.asksaveasfilename.return_value = str(out_path)
            manager.save_sh_script()
        return out_path.read_text()

    def test_script_has_shebang_and_set_e(self, manager, launcher_mock, tmp_path):
        out = tmp_path / "launch.sh"
        text = self._write_and_read(manager, launcher_mock, out)
        assert text.startswith("#!/bin/bash")
        assert "set -e" in text

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Windows filesystems don't honor POSIX exec bits",
    )
    def test_script_is_executable(self, manager, launcher_mock, tmp_path):
        out = tmp_path / "launch.sh"
        self._write_and_read(manager, launcher_mock, out)
        mode = os.stat(out).st_mode
        assert mode & stat.S_IEXEC, "Script should have exec bit set"

    def test_script_has_cuda_device_order(self, manager, launcher_mock, tmp_path):
        out = tmp_path / "launch.sh"
        text = self._write_and_read(manager, launcher_mock, out)
        assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in text

    def test_cuda_visible_devices_set_when_gpus_selected(
        self, manager, launcher_mock, tmp_path
    ):
        launcher_mock.get_ordered_selected_gpus.return_value = [1, 0]
        launcher_mock.gpu_info = {"device_count": 2}
        out = tmp_path / "launch.sh"
        text = self._write_and_read(manager, launcher_mock, out)
        assert 'export CUDA_VISIBLE_DEVICES="1,0"' in text

    def test_cuda_visible_devices_cleared_when_gpus_exist_none_selected(
        self, manager, launcher_mock, tmp_path
    ):
        launcher_mock.get_ordered_selected_gpus.return_value = []
        launcher_mock.gpu_info = {"device_count": 2}
        out = tmp_path / "launch.sh"
        text = self._write_and_read(manager, launcher_mock, out)
        assert "unset CUDA_VISIBLE_DEVICES" in text

    def test_no_cuda_lines_when_no_gpus_detected(
        self, manager, launcher_mock, tmp_path
    ):
        launcher_mock.get_ordered_selected_gpus.return_value = []
        launcher_mock.gpu_info = {"device_count": 0}
        out = tmp_path / "launch.sh"
        text = self._write_and_read(manager, launcher_mock, out)
        # CUDA_DEVICE_ORDER is always emitted, but CUDA_VISIBLE_DEVICES should not
        # be set or unset.
        assert "CUDA_VISIBLE_DEVICES" not in text

    def test_env_vars_exported(self, manager, launcher_mock, tmp_path):
        launcher_mock.env_vars_manager.get_enabled_env_vars.return_value = {
            "FOO": "bar",
            "BAZ": "val with spaces",
        }
        out = tmp_path / "launch.sh"
        text = self._write_and_read(manager, launcher_mock, out)
        assert 'export FOO="bar"' in text
        assert 'export BAZ="val with spaces"' in text

    def test_env_var_value_with_special_chars_escaped(
        self, manager, launcher_mock, tmp_path
    ):
        launcher_mock.env_vars_manager.get_enabled_env_vars.return_value = {
            "Q": 'has "quotes" $var `back`',
        }
        out = tmp_path / "launch.sh"
        text = self._write_and_read(manager, launcher_mock, out)
        # Double quotes, $, and backticks should be backslash-escaped.
        assert r'\"quotes\"' in text
        assert r"\$var" in text
        assert r"\`back\`" in text

    def test_venv_activation_line_when_venv_configured(
        self, manager, launcher_mock, tmp_path
    ):
        venv = tmp_path / "my venv"  # space in name
        (venv / "bin").mkdir(parents=True)
        activate = venv / "bin" / "activate"
        activate.write_text("# mock activate\n")
        launcher_mock.venv_dir.set(str(venv))

        out = tmp_path / "launch.sh"
        text = self._write_and_read(manager, launcher_mock, out)
        # Path quoted with double quotes to preserve the space.
        assert f'source "{activate}"' in text

    def test_no_venv_means_no_source_line(self, manager, launcher_mock, tmp_path):
        launcher_mock.venv_dir.set("")
        out = tmp_path / "launch.sh"
        text = self._write_and_read(manager, launcher_mock, out)
        assert "source " not in text

    def test_model_path_with_space_is_quoted(
        self, manager, launcher_mock, tmp_path
    ):
        import shlex

        # Build a tree with a space in the path.
        spaced = tmp_path / "dir with space"
        spaced.mkdir()
        bin_dir = spaced / "build" / "bin"
        bin_dir.mkdir(parents=True)
        exe_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
        exe = bin_dir / exe_name
        exe.write_text("fake")
        os.chmod(exe, 0o755)
        model = spaced / "model file.gguf"
        model.write_bytes(b"GGUF")

        launcher_mock.llama_cpp_dir.set(str(spaced))
        launcher_mock.model_path.set(str(model))

        out = tmp_path / "launch.sh"
        text = self._write_and_read(manager, launcher_mock, out)

        # Assert the exact shlex-quoted token is present — a bare substring
        # like "'" or "model file.gguf" would pass even if quoting broke
        # and left the path word-split across two bash tokens.
        expected_model_token = shlex.quote(str(model))
        expected_exe_token = shlex.quote(str(exe))
        assert expected_model_token in text, (
            f"Model path not shlex-quoted in script. Expected token: "
            f"{expected_model_token!r}"
        )
        assert expected_exe_token in text, (
            f"Executable path not shlex-quoted in script. Expected token: "
            f"{expected_exe_token!r}"
        )
        # And the ``-m`` flag must be immediately followed by the quoted
        # model path (catches a regression where quoting is right but the
        # flag/value pair gets reordered).
        assert f"-m {expected_model_token}" in text

    def test_chat_template_gets_single_quote_escape(
        self, manager, launcher_mock, tmp_path
    ):
        import shlex

        template = "it's a template"
        launcher_mock.template_source.set("custom")
        launcher_mock.current_template_display.set(template)
        out = tmp_path / "launch.sh"
        text = self._write_and_read(manager, launcher_mock, out)

        # shlex.quote of an apostrophe-containing value uses the canonical
        # bash trick: close the single-quoted string, emit a double-quoted
        # ``'``, reopen the single-quoted string.
        #     it's a template  ->  'it'"'"'s a template'
        expected = shlex.quote(template)
        # Sanity-check the encoded form so a shlex.quote behavior change
        # would surface here first:
        assert expected == "'it'\"'\"'s a template'"
        # The ``--chat-template`` flag must appear paired with this exact
        # token. A weaker substring check (e.g. ``"it" in text``) would
        # have passed even if the apostrophe escape collapsed.
        assert f"--chat-template {expected}" in text

    def test_script_returns_early_if_user_cancels_dialog(
        self, manager, launcher_mock, tmp_path
    ):
        with patch("modules.launch.filedialog") as fd, patch(
            "modules.launch.messagebox"
        ) as mb:
            fd.asksaveasfilename.return_value = ""  # user cancelled
            manager.save_sh_script()
        # Nothing written and no "Saved" info dialog.
        assert not mb.showinfo.called

    def test_script_returns_early_if_build_cmd_fails(
        self, manager, launcher_mock, tmp_path
    ):
        launcher_mock.model_path.set("")  # triggers build_cmd None
        with patch("modules.launch.filedialog") as fd, patch(
            "modules.launch.messagebox"
        ):
            fd.asksaveasfilename.return_value = str(tmp_path / "x.sh")
            manager.save_sh_script()
        # The save-dialog must not even have been invoked.
        assert not fd.asksaveasfilename.called


# ============================================================================
# save_ps1_script
# ============================================================================


class TestSavePs1Script:
    def _write_and_read(self, manager, launcher_mock, out_path):
        with patch("modules.launch.filedialog") as fd, patch(
            "modules.launch.messagebox"
        ):
            fd.asksaveasfilename.return_value = str(out_path)
            manager.save_ps1_script()
        return out_path.read_text(encoding="utf-8")

    def test_ps1_header_present(self, manager, launcher_mock, tmp_path):
        out = tmp_path / "launch.ps1"
        text = self._write_and_read(manager, launcher_mock, out)
        assert "$ErrorActionPreference" in text
        assert ".SYNOPSIS" in text

    def test_ps1_sets_cuda_device_order(self, manager, launcher_mock, tmp_path):
        out = tmp_path / "launch.ps1"
        text = self._write_and_read(manager, launcher_mock, out)
        assert '$env:CUDA_DEVICE_ORDER="PCI_BUS_ID"' in text

    def test_ps1_cuda_visible_devices_quoted(
        self, manager, launcher_mock, tmp_path
    ):
        launcher_mock.get_ordered_selected_gpus.return_value = [0, 2]
        launcher_mock.gpu_info = {"device_count": 3}
        out = tmp_path / "launch.ps1"
        text = self._write_and_read(manager, launcher_mock, out)
        assert '$env:CUDA_VISIBLE_DEVICES="0,2"' in text

    def test_ps1_cuda_cleared_when_gpus_detected_none_selected(
        self, manager, launcher_mock, tmp_path
    ):
        launcher_mock.get_ordered_selected_gpus.return_value = []
        launcher_mock.gpu_info = {"device_count": 3}
        out = tmp_path / "launch.ps1"
        text = self._write_and_read(manager, launcher_mock, out)
        assert "Remove-Item Env:CUDA_VISIBLE_DEVICES" in text

    def test_ps1_env_vars_emitted(self, manager, launcher_mock, tmp_path):
        launcher_mock.env_vars_manager.get_enabled_env_vars.return_value = {
            "FOO": "bar"
        }
        out = tmp_path / "launch.ps1"
        text = self._write_and_read(manager, launcher_mock, out)
        assert '$env:FOO="bar"' in text

    def test_ps1_chat_template_single_quoted(self, manager, launcher_mock, tmp_path):
        launcher_mock.template_source.set("custom")
        launcher_mock.current_template_display.set("Jinja's template")
        out = tmp_path / "launch.ps1"
        text = self._write_and_read(manager, launcher_mock, out)
        # Expect the PowerShell single-quoted form with doubled quote for apostrophe.
        assert "--chat-template" in text
        # Single quote should have been doubled.
        assert "Jinja''s template" in text

    def test_ps1_exe_path_rendered_with_forward_slashes(
        self, manager, launcher_mock, tmp_path, built_tree
    ):
        out = tmp_path / "launch.ps1"
        text = self._write_and_read(manager, launcher_mock, out)
        # PowerShell block uses forward slashes (from Path.as_posix()).
        exe_posix = built_tree["exe"].resolve().as_posix()
        assert f'& "{exe_posix}"' in text

    def test_ps1_user_cancel_returns_early(self, manager, launcher_mock, tmp_path):
        with patch("modules.launch.filedialog") as fd, patch(
            "modules.launch.messagebox"
        ) as mb:
            fd.asksaveasfilename.return_value = ""
            manager.save_ps1_script()
        assert not mb.showinfo.called


# ============================================================================
# Filename default generation (model name sanitisation)
# ============================================================================


class TestSaveScriptDefaultFilename:
    def test_sh_default_filename_uses_selected_model_name(
        self, manager, launcher_mock, tmp_path
    ):
        launcher_mock.model_listbox.curselection = MagicMock(return_value=(0,))
        launcher_mock.model_listbox.get = MagicMock(
            return_value="My Fancy Model v2.gguf"
        )

        captured_initialfile = {}

        def fake_save(**kw):
            captured_initialfile["name"] = kw.get("initialfile")
            return str(tmp_path / "out.sh")

        with patch("modules.launch.filedialog") as fd, patch(
            "modules.launch.messagebox"
        ):
            fd.asksaveasfilename.side_effect = fake_save
            manager.save_sh_script()

        # Unsafe chars replaced with underscores; must end with .sh.
        assert captured_initialfile["name"].endswith(".sh")
        assert " " not in captured_initialfile["name"]

    def test_ps1_default_filename_uses_selected_model_name(
        self, manager, launcher_mock, tmp_path
    ):
        launcher_mock.model_listbox.curselection = MagicMock(return_value=(0,))
        launcher_mock.model_listbox.get = MagicMock(
            return_value='bad<chars>"name'
        )

        captured = {}

        def fake_save(**kw):
            captured["name"] = kw.get("initialfile")
            return str(tmp_path / "out.ps1")

        with patch("modules.launch.filedialog") as fd, patch(
            "modules.launch.messagebox"
        ):
            fd.asksaveasfilename.side_effect = fake_save
            manager.save_ps1_script()

        assert captured["name"].endswith(".ps1")
        for ch in '<>:"/\\|?*':
            assert ch not in captured["name"]
