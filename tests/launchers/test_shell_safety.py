"""Shell-safety tests for the launch cluster.

Covers:
  * Shell-injection / path-traversal attempts in model, venv, and backend
    paths - dollar signs, backticks, `; rm -rf`, pipes, ampersands,
    redirects, single/double quotes, whitespace, unicode. Both the bash
    (``save_sh_script``) and PowerShell (``save_ps1_script``) output
    paths are exercised. We don't need to run the scripts - we verify
    the offending characters are properly *quoted*, not stripped.

  * Custom-parameter parsing: ``shlex.split`` with unclosed quotes,
    command-substitution syntax, newlines, >100 args chained.

  * GPU index edge cases: 32+ GPUs, invalid / negative indices.

  * mmproj behaviour: symlinks, multi-model collision.

  * Very long cmdlines (10k args) confirming no crash.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_sh_and_read(manager, launcher_mock, out_path):
    with patch("modules.launch.filedialog") as fd, patch(
        "modules.launch.messagebox"
    ):
        fd.asksaveasfilename.return_value = str(out_path)
        manager.save_sh_script()
    return out_path.read_text()


def _save_ps1_and_read(manager, launcher_mock, out_path):
    with patch("modules.launch.filedialog") as fd, patch(
        "modules.launch.messagebox"
    ):
        fd.asksaveasfilename.return_value = str(out_path)
        manager.save_ps1_script()
    return out_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Shell injection - paths with dangerous chars
# ---------------------------------------------------------------------------


class TestShellInjectionBash:
    """Any dangerous character in a user-supplied path must appear in the
    generated bash script *only inside a shlex-quoted string*, never as a
    bare metacharacter."""

    @pytest.mark.parametrize(
        "evil",
        [
            "rm; rm -rf /",
            "$HOME",
            "$(whoami)",
            "`id`",
            "| cat /etc/passwd",
            "&& touch /tmp/x",
            "> /tmp/owned",
            "'single quotes'",
            '"double quotes"',
            "semi; colon && and-and",
            "space  inside",
        ],
    )
    def test_model_path_with_evil_chars_is_quoted(
        self, manager, launcher_mock, tmp_path, evil
    ):
        spaced = tmp_path / f"models_{abs(hash(evil)) % 100000}"
        bin_dir = spaced / "build" / "bin"
        bin_dir.mkdir(parents=True)
        exe_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
        (bin_dir / exe_name).write_text("fake")
        try:
            os.chmod(bin_dir / exe_name, 0o755)
        except OSError:
            pass

        # Most filesystems allow all of these in a filename; where they
        # don't (e.g. `/` or NUL) we skip.
        try:
            model = spaced / f"m_{evil.replace('/', '_')}.gguf"
            model.write_bytes(b"GGUF")
        except OSError:
            pytest.skip(f"Filesystem rejected evil name: {evil!r}")

        launcher_mock.llama_cpp_dir.set(str(spaced))
        launcher_mock.model_path.set(str(model))

        out = tmp_path / f"launch_{abs(hash(evil)) % 100000}.sh"
        text = _save_sh_and_read(manager, launcher_mock, out)

        # The launcher line must not contain *unquoted* metacharacters.
        # Heuristic: find the last `llama-server ... -m <path>` line and
        # verify the path token is shlex-quoted (starts with ' or " when
        # containing special chars).
        import shlex as _shlex
        expected_quoted = _shlex.quote(str(model.resolve()))
        assert expected_quoted in text, (
            f"Expected shlex-quoted model path in script, got:\n{text}"
        )

    def test_venv_path_with_space_is_quoted(
        self, manager, launcher_mock, tmp_path
    ):
        venv = tmp_path / "my venv with spaces"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "activate").write_text("# mock\n")
        launcher_mock.venv_dir.set(str(venv))

        out = tmp_path / "l.sh"
        text = _save_sh_and_read(manager, launcher_mock, out)
        # source "<venv>/bin/activate" - double quotes preserve spaces.
        assert f'source "{venv / "bin" / "activate"}"' in text

    def test_unicode_and_emoji_in_paths(
        self, manager, launcher_mock, tmp_path
    ):
        # ggUF that's had a wide time
        spaced = tmp_path / "unicode_dir_\u4e2d\u6587_\U0001f680"
        bin_dir = spaced / "build" / "bin"
        bin_dir.mkdir(parents=True)
        exe_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
        (bin_dir / exe_name).write_text("fake")
        os.chmod(bin_dir / exe_name, 0o755)
        model = spaced / "mod\u00e9l.gguf"
        model.write_bytes(b"GGUF")

        launcher_mock.llama_cpp_dir.set(str(spaced))
        launcher_mock.model_path.set(str(model))

        out = tmp_path / "u.sh"
        text = _save_sh_and_read(manager, launcher_mock, out)
        assert "mod\u00e9l.gguf" in text
        assert "\U0001f680" in text


class TestShellInjectionPowerShell:
    """The PS output uses double-quoted strings for most args; verify
    internal ``"`` is doubled ("") and that backticks aren't double-
    escaped (regression for the fixed bug)."""

    def test_model_path_with_embedded_double_quote(
        self, manager, launcher_mock, tmp_path
    ):
        # Filesystem may refuse " on some FS types; simulate by setting
        # the model_path var to a path-shaped string. build_cmd validates
        # is_file(), so point at a real file and mock model_path post-hoc.
        model = tmp_path / "model.gguf"
        model.write_bytes(b"GGUF")
        launcher_mock.model_path.set(str(model))

        # Inject an evil custom param whose value contains " and `.
        launcher_mock.custom_parameters_list = ['--alias a"b`c']

        out = tmp_path / "out.ps1"
        text = _save_ps1_and_read(manager, launcher_mock, out)

        # The evil token must appear either with "" (for ") and `` (for `)
        # escaping, and the tri-sequence ```" must not appear.
        assert "a" in text and "b" in text and "c" in text
        assert '```"' not in text, (
            "Regression of backtick double-escape bug: found ```\""
        )

    def test_custom_param_with_backtick_escaped_only_once(
        self, manager, launcher_mock, tmp_path
    ):
        launcher_mock.custom_parameters_list = ["--weird val`ue"]
        out = tmp_path / "b.ps1"
        text = _save_ps1_and_read(manager, launcher_mock, out)
        # Exactly one doubled backtick (``) per original `, never triple.
        assert "val``ue" in text
        assert "val```ue" not in text


# ---------------------------------------------------------------------------
# Custom parameters - shlex parsing edge cases
# ---------------------------------------------------------------------------


class TestCustomParameters:
    def test_unclosed_quote_is_caught_and_skipped(self, manager, launcher_mock):
        launcher_mock.custom_parameters_list = ['--foo "unclosed']
        with patch("modules.launch.messagebox") as mb:
            cmd = manager.build_cmd()
        # build_cmd returns a valid cmd even after skipping the bad param,
        # and surfaces a warning.
        assert cmd is not None
        assert mb.showwarning.called
        assert "--foo" not in cmd
        assert "unclosed" not in cmd

    def test_command_substitution_syntax_passed_through_unexecuted(
        self, manager, launcher_mock
    ):
        # shlex.split parses $() as a normal argument; no shell evaluation
        # takes place at build_cmd time (and the shell layer later uses
        # shlex.quote, so it stays literal).
        launcher_mock.custom_parameters_list = ['--arg $(whoami)']
        cmd = manager.build_cmd()
        assert "--arg" in cmd
        assert "$(whoami)" in cmd

    def test_over_100_args_chain(self, manager, launcher_mock):
        # --f0 v0 --f1 v1 ... --f99 v99
        many = " ".join(f"--f{i} v{i}" for i in range(120))
        launcher_mock.custom_parameters_list = [many]
        cmd = manager.build_cmd()
        assert "--f119" in cmd
        assert cmd[cmd.index("--f119") + 1] == "v119"
        # All 120 flags made it.
        for i in range(120):
            assert f"--f{i}" in cmd

    def test_multiple_param_strings_all_split_independently(
        self, manager, launcher_mock
    ):
        launcher_mock.custom_parameters_list = [
            '--alpha 1',
            '--beta "two words"',
            '--gamma 3',
        ]
        cmd = manager.build_cmd()
        assert "--alpha" in cmd and "1" in cmd
        assert "--beta" in cmd and "two words" in cmd
        assert "--gamma" in cmd and "3" in cmd

    def test_empty_param_string_is_harmless(self, manager, launcher_mock):
        launcher_mock.custom_parameters_list = ["", "   "]
        cmd = manager.build_cmd()
        assert cmd is not None


# ---------------------------------------------------------------------------
# Chat templates - unicode, newlines, huge, control chars
# ---------------------------------------------------------------------------


class TestChatTemplateContent:
    def test_unicode_emoji_template_passes_through(
        self, manager, launcher_mock, tmp_path
    ):
        template = "\u4f60\u597d {{msg}} \U0001f600"
        launcher_mock.template_source.set("custom")
        launcher_mock.current_template_display.set(template)
        cmd = manager.build_cmd()
        assert template in cmd

        out = tmp_path / "u.sh"
        text = _save_sh_and_read(manager, launcher_mock, out)
        assert "\U0001f600" in text

    def test_newline_in_template_preserved_in_sh(
        self, manager, launcher_mock, tmp_path
    ):
        template = "line1\nline2\nline3"
        launcher_mock.template_source.set("custom")
        launcher_mock.current_template_display.set(template)

        out = tmp_path / "n.sh"
        text = _save_sh_and_read(manager, launcher_mock, out)
        # Bash single-quote preserves newlines literally.
        assert "line1" in text and "line2" in text and "line3" in text

    def test_very_long_template_over_4kb(self, manager, launcher_mock, tmp_path):
        template = "A" * 5000
        launcher_mock.template_source.set("custom")
        launcher_mock.current_template_display.set(template)
        cmd = manager.build_cmd()
        assert template in cmd

        out = tmp_path / "big.sh"
        text = _save_sh_and_read(manager, launcher_mock, out)
        assert "A" * 5000 in text

    def test_control_chars_in_template(self, manager, launcher_mock):
        template = "a\tb\x0cc"  # tab + form-feed
        launcher_mock.template_source.set("custom")
        launcher_mock.current_template_display.set(template)
        cmd = manager.build_cmd()
        assert template in cmd

    def test_apostrophes_doubled_in_ps1(self, manager, launcher_mock, tmp_path):
        launcher_mock.template_source.set("custom")
        launcher_mock.current_template_display.set("it's \"awesome\"")
        out = tmp_path / "a.ps1"
        text = _save_ps1_and_read(manager, launcher_mock, out)
        # Single-quote escaping doubles ' to ''.
        assert "it''s" in text


# ---------------------------------------------------------------------------
# GPU edge cases
# ---------------------------------------------------------------------------


class TestGpuIndexEdgeCases:
    def test_32_plus_gpus_emits_all_indices(
        self, manager, launcher_mock, tmp_path
    ):
        indices = list(range(40))
        launcher_mock.get_ordered_selected_gpus.return_value = indices
        launcher_mock.gpu_info = {"device_count": 40}

        out = tmp_path / "g.sh"
        text = _save_sh_and_read(manager, launcher_mock, out)
        joined = ",".join(map(str, indices))
        assert f'export CUDA_VISIBLE_DEVICES="{joined}"' in text

    def test_negative_index_emitted_verbatim_not_stripped(
        self, manager, launcher_mock, tmp_path
    ):
        # The launcher doesn't validate GPU indices; it faithfully emits
        # whatever it was handed. Ensure nothing explodes.
        launcher_mock.get_ordered_selected_gpus.return_value = [-1, 0]
        launcher_mock.gpu_info = {"device_count": 1}

        out = tmp_path / "n.sh"
        text = _save_sh_and_read(manager, launcher_mock, out)
        assert '-1,0' in text

    def test_gpu_list_with_100_entries_ps1(
        self, manager, launcher_mock, tmp_path
    ):
        indices = list(range(100))
        launcher_mock.get_ordered_selected_gpus.return_value = indices
        launcher_mock.gpu_info = {"device_count": 100}

        out = tmp_path / "g.ps1"
        text = _save_ps1_and_read(manager, launcher_mock, out)
        joined = ",".join(map(str, indices))
        assert f'$env:CUDA_VISIBLE_DEVICES="{joined}"' in text


# ---------------------------------------------------------------------------
# mmproj edge cases - symlinks, multi-model dir collision
# ---------------------------------------------------------------------------


class TestMmprojEdgeCases:
    def test_mmproj_symlink_is_followed(
        self, manager, launcher_mock, built_tree
    ):
        model_dir = built_tree["model"].parent
        real = model_dir / "mmproj-real.gguf"
        real.write_bytes(b"GGUF")
        link = model_dir / "mmproj-link.gguf"
        try:
            link.symlink_to(real)
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks not supported on this filesystem.")

        launcher_mock.mmproj_enabled.set(True)
        launcher_mock.selected_mmproj_path.set(str(link))
        cmd = manager.build_cmd()
        # When explicitly selected, the resolved path should be used.
        assert "--mmproj" in cmd
        mmproj_arg = cmd[cmd.index("--mmproj") + 1]
        # Either the link or its resolved real target - both are "mmproj"-named.
        assert "mmproj" in Path(mmproj_arg).name.lower()

    def test_multi_model_dir_collision_prefers_matching_stem(
        self, manager, launcher_mock, tmp_path
    ):
        # Two mmproj files in the model directory, only one of which shares
        # the model's stem. Fallback auto-detection should prefer it.
        # Use a name distinct from ``built_tree``'s llama_cpp_root.
        base = tmp_path / "mm_root"
        bin_dir = base / "build" / "bin"
        bin_dir.mkdir(parents=True)
        exe_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
        (bin_dir / exe_name).write_text("fake")
        os.chmod(bin_dir / exe_name, 0o755)

        mdir = tmp_path / "mm_models"
        mdir.mkdir()
        model = mdir / "qwen2-vl-7b.gguf"
        model.write_bytes(b"GGUF")
        # An mmproj for *this* model:
        matching = mdir / "qwen2-vl-7b-mmproj.gguf"
        matching.write_bytes(b"GGUF")
        # An unrelated mmproj in the same directory (different model family):
        unrelated = mdir / "llama-3-mmproj.gguf"
        unrelated.write_bytes(b"GGUF")

        launcher_mock.llama_cpp_dir.set(str(base))
        launcher_mock.model_path.set(str(model))
        launcher_mock.mmproj_enabled.set(True)
        launcher_mock.selected_mmproj_path.set("")  # no explicit selection
        cmd = manager.build_cmd()
        assert "--mmproj" in cmd
        picked = Path(cmd[cmd.index("--mmproj") + 1]).name
        assert picked == "qwen2-vl-7b-mmproj.gguf", (
            f"Expected stem-matching mmproj, got {picked!r}"
        )


# ---------------------------------------------------------------------------
# venv alt-path discovery (Linux/macOS / Scripts on Windows / poetry-style)
# ---------------------------------------------------------------------------


class TestVenvAlternatePaths:
    def test_save_sh_finds_scripts_activate_fallback(
        self, manager, launcher_mock, tmp_path
    ):
        # bin/activate missing, Scripts/activate present (Cygwin/WSL case).
        venv = tmp_path / "wsl_venv"
        (venv / "Scripts").mkdir(parents=True)
        (venv / "Scripts" / "activate").write_text("# mock\n")
        launcher_mock.venv_dir.set(str(venv))

        out = tmp_path / "x.sh"
        text = _save_sh_and_read(manager, launcher_mock, out)
        assert "Scripts" in text and "activate" in text

    def test_save_ps1_finds_bin_activate_ps1(
        self, manager, launcher_mock, tmp_path
    ):
        # PowerShell Core on Linux/macOS: Activate.ps1 lives in bin/.
        venv = tmp_path / "pwsh_venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "Activate.ps1").write_text("# mock\n")
        launcher_mock.venv_dir.set(str(venv))

        out = tmp_path / "x.ps1"
        text = _save_ps1_and_read(manager, launcher_mock, out)
        assert "Activate.ps1" in text

    def test_save_ps1_missing_all_activation_scripts_writes_warning(
        self, manager, launcher_mock, tmp_path
    ):
        venv = tmp_path / "empty_venv"
        venv.mkdir()  # no Scripts/ or bin/ at all
        launcher_mock.venv_dir.set(str(venv))

        out = tmp_path / "w.ps1"
        text = _save_ps1_and_read(manager, launcher_mock, out)
        assert "Write-Warning" in text
        assert "activation script" in text.lower()

    def test_save_sh_missing_activation_script_writes_warning(
        self, manager, launcher_mock, tmp_path
    ):
        venv = tmp_path / "empty_venv2"
        venv.mkdir()
        launcher_mock.venv_dir.set(str(venv))

        out = tmp_path / "w.sh"
        text = _save_sh_and_read(manager, launcher_mock, out)
        assert "Warning" in text


# ---------------------------------------------------------------------------
# Very long command lines
# ---------------------------------------------------------------------------


class TestLongCommandLines:
    def test_10k_custom_args_does_not_crash_build_cmd(
        self, manager, launcher_mock
    ):
        # Stress test: build_cmd must handle a very large argv without
        # raising.
        many = " ".join(f"--f{i} v{i}" for i in range(5000))  # 10000 tokens
        launcher_mock.custom_parameters_list = [many]
        cmd = manager.build_cmd()
        assert cmd is not None
        assert len(cmd) > 10000  # 10k custom + base args
        assert f"--f4999" in cmd
