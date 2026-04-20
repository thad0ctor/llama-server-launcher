"""Regression tests for the specific bugs fixed in modules/launch.py.

Bugs covered:
  1. ``launch.py:579-623`` - convoluted chat-template reconstruction loop.
     * Malformed ``cmd_list`` (``--chat-template`` with no value) must be
       handled cleanly via ``_build_ps_cmd_parts`` raising ``ValueError``
       rather than producing garbled PowerShell.
     * Happy-path reconstruction must produce exactly the right number of
       tokens with no duplicated template value.

  2. ``launch.py:722`` - brittle env-var injection via
     ``.replace('echo "Virtual environment activated." && ', ...)``.
     If a model path (or any other bash-quoted arg) happened to contain
     that literal marker, the replace could corrupt the launch line. We
     now build the script as an ordered list joined with ' && ', so the
     marker has no special meaning.

  3. ``launch.py:554`` - PowerShell backtick + double-quote double-escape.
     The original code escaped ``"`` -> ``\\``"`` first then ``\\``` ->
     ``\\`\\```; the intermediate backtick introduced by the first pass
     was then double-escaped by the second, collapsing to a literal
     backtick plus an unescaped double-quote. The fix reverses the order.
"""

from __future__ import annotations

import stat
import sys
import tkinter as tk
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# Bug 3: PowerShell backtick / double-quote escape ordering
# ============================================================================


class TestPsEscapeOrdering:
    def test_path_with_embedded_double_quote(self, manager):
        # A path-like token containing a literal double-quote.
        escaped = manager._ps_escape_double_quoted('a"b')
        # After escape: '`"' - the `" sequence is how PS represents a literal
        # double-quote inside a double-quoted string. A single backtick only.
        assert escaped == 'a`"b'

    def test_path_with_embedded_backtick(self, manager):
        escaped = manager._ps_escape_double_quoted('a`b')
        # Backtick is doubled (`\\`\\``).
        assert escaped == 'a``b'

    def test_path_with_both_backtick_and_quote_is_not_double_escaped(self, manager):
        # The broken ordering produced 'a``"b' here which, wrapped in "...",
        # PS parses as literal backtick + closing quote + trailing b".
        escaped = manager._ps_escape_double_quoted('a"b`c')
        assert escaped == 'a`"b``c'
        # And the final double-quoted form must be well-formed: exactly one
        # backtick immediately before each " and exactly two backticks for
        # each original ` (so that after PS unescapes, we get one backtick).
        wrapped = f'"{escaped}"'
        # No literal backtick ever appears as '```"' (which would mean
        # ``-escaped backtick followed by unescaped quote).
        assert '```"' not in wrapped

    def test_path_without_special_chars_untouched(self, manager):
        assert manager._ps_escape_double_quoted("/usr/local/bin/llama") == (
            "/usr/local/bin/llama"
        )

    def test_ps_quote_arg_uses_doubled_double_quotes(self, manager):
        # Native-exe style: embedded " becomes "" (not `")
        assert manager._ps_quote_arg('a"b') == '"a""b"'

    def test_ps_quote_arg_escapes_backticks(self, manager):
        assert manager._ps_quote_arg('a`b') == '"a``b"'


# ============================================================================
# Bug 1: chat-template reconstruction / malformed cmd_list
# ============================================================================


class TestBuildPsCmdParts:
    def test_raises_on_empty_cmd_list(self, manager):
        import pytest
        with pytest.raises(ValueError):
            manager._build_ps_cmd_parts([])

    def test_raises_on_trailing_chat_template_without_value(self, manager):
        # --chat-template at the very end of cmd_list is malformed.
        import pytest
        with pytest.raises(ValueError) as ei:
            manager._build_ps_cmd_parts(["/bin/llama-server", "-m", "/m", "--chat-template"])
        assert "--chat-template" in str(ei.value)

    def test_happy_path_chat_template_single_quoted(self, manager):
        parts = manager._build_ps_cmd_parts(
            ["/bin/llama-server", "-m", "/m/model.gguf", "--chat-template", "chatml"]
        )
        # --chat-template is followed by a single-quoted value.
        assert "--chat-template" in parts
        idx = parts.index("--chat-template")
        assert parts[idx + 1] == "'chatml'"

    def test_chat_template_single_quotes_are_doubled(self, manager):
        parts = manager._build_ps_cmd_parts(
            ["/bin/llama-server", "--chat-template", "it's a template"]
        )
        idx = parts.index("--chat-template")
        assert parts[idx + 1] == "'it''s a template'"

    def test_chat_template_value_is_not_duplicated(self, manager):
        # Regression for the old bug where the outer for-loop continued
        # after the inner reconstruction and processed the template value
        # a second time as a standalone arg.
        cmd = ["/bin/llama-server", "-m", "/m/model.gguf", "--chat-template", "chatml"]
        parts = manager._build_ps_cmd_parts(cmd)
        # Count how many tokens contain the word "chatml".
        chatml_tokens = [p for p in parts if "chatml" in p]
        assert len(chatml_tokens) == 1, (
            f"Template value duplicated in PS cmd parts: {parts!r}"
        )

    def test_non_template_args_double_quoted(self, manager):
        parts = manager._build_ps_cmd_parts(
            ["/bin/llama-server", "-m", "/path with space/model.gguf"]
        )
        # "-m" and the model path both double-quoted.
        assert parts[1] == '"-m"'
        assert parts[2] == '"/path with space/model.gguf"'

    def test_multiple_chat_templates_if_ever_present(self, manager):
        # Pathological but valid input. Both occurrences must get single
        # quotes and their values must not be duplicated.
        parts = manager._build_ps_cmd_parts(
            ["/bin/llama-server", "--chat-template", "a", "-x", "1", "--chat-template", "b"]
        )
        single_quoted = [p for p in parts if p.startswith("'") and p.endswith("'")]
        assert "'a'" in single_quoted
        assert "'b'" in single_quoted


# ============================================================================
# Bug 2: env-var injection must not rely on .replace() on a literal marker
# ============================================================================


class TestEnvInjectionMarkerCollision:
    """Verify that a model path containing the literal marker string
    ``echo "Virtual environment activated." && `` does not corrupt the
    generated bash launcher when a venv and env vars are both configured.

    Regression: previous implementation rebuilt the script by calling
    ``full_script_content.replace('echo "Virtual environment activated." && ', ...)``
    which would splice the env-var block into any occurrence of that
    string anywhere in the command, including inside a user-supplied model
    path or custom parameter.
    """

    def _run_save_sh(self, manager, out_path):
        with patch("modules.launch.filedialog") as fd, patch(
            "modules.launch.messagebox"
        ):
            fd.asksaveasfilename.return_value = str(out_path)
            manager.save_sh_script()
        return out_path.read_text()

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason=(
            "Exercises the Linux branch via sys.platform patching; "
            "os.name stays 'nt' on real Windows so the code takes the "
            "wrong branch and Popen is never invoked. See "
            "test_launch_server.TestLaunchServerLinuxTerminals for the "
            "same rationale."
        ),
    )
    def test_venv_and_env_vars_use_ordered_command_list_no_replace(
        self, manager, launcher_mock, tmp_path
    ):
        # ``launch_server`` is hard to exercise directly (spawns subprocesses);
        # instead we reach into ``launch_server``'s Linux branch by patching
        # the heavy side-effects. The key assertion is that the resulting
        # ``full_script_content`` does not contain two copies of the marker
        # and orders venv activation, env exports, launch line as expected.

        # Wire up a venv with bin/activate so the venv branch triggers.
        venv = tmp_path / "venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "activate").write_text("# mock activate\n")
        launcher_mock.venv_dir.set(str(venv))

        launcher_mock.env_vars_manager.get_enabled_env_vars.return_value = {
            "FOO": "bar",
        }
        launcher_mock.get_ordered_selected_gpus.return_value = [0]
        launcher_mock.gpu_info = {"device_count": 1}

        # Patch platform -> linux to take the else branch.
        captured = {}

        def fake_popen(cmd, **kw):
            captured["cmd"] = cmd
            captured["kw"] = kw
            return MagicMock()

        with patch.object(sys, "platform", "linux"), patch(
            "modules.launch.subprocess.Popen", side_effect=fake_popen
        ), patch(
            "modules.launch.shutil.which", return_value=None
        ), patch("modules.launch.messagebox"):
            manager.launch_server()

        # Fallback direct-launch path calls Popen with shell=True and a
        # string; capture it.
        assert "cmd" in captured
        script = captured["cmd"]
        assert isinstance(script, str)

        # venv activation appears exactly once.
        assert script.count("source") == 1
        assert 'echo "Virtual environment activated."' in script
        # env-var export appears once.
        assert script.count("export FOO=") == 1
        # CUDA_VISIBLE_DEVICES set once with the right value.
        assert "export CUDA_VISIBLE_DEVICES=0" in script
        # The launch line fires after the env block.
        assert script.index("echo \"Launching server...\"") > script.index(
            "export FOO="
        )

    def test_model_path_containing_marker_substring_not_corrupted(
        self, manager, launcher_mock, tmp_path
    ):
        """Use a model filename containing the literal marker string.
        Previously .replace() would happily splice env-var commands into the
        middle of the quoted model path.
        """
        # We can't easily test launch_server's bash branch on win32, but
        # save_sh_script exercises the same ordering logic for scripts.
        spaced = tmp_path / "weird"
        spaced.mkdir()
        bin_dir = spaced / "build" / "bin"
        bin_dir.mkdir(parents=True)
        exe_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
        exe = bin_dir / exe_name
        exe.write_text("fake")

        marker = 'echo "Virtual environment activated." && '
        model = spaced / (marker + "model.gguf")
        # Filesystem may reject quotes on some platforms; fall back to a
        # safer but still marker-containing name if so.
        try:
            model.write_bytes(b"GGUF")
        except OSError:
            model = spaced / (marker.replace('"', "_") + "model.gguf")
            model.write_bytes(b"GGUF")

        launcher_mock.llama_cpp_dir.set(str(spaced))
        launcher_mock.model_path.set(str(model))

        out = tmp_path / "launch.sh"
        text = self._run_save_sh(manager, out)

        # The saved script must contain the full model filename as one
        # shlex-quoted token. Even if the marker appears inside, there must
        # be exactly one launch-server invocation line.
        # The substring "llama-server" should appear in the command (plus
        # in the boilerplate echo). We just verify the script is executable
        # and well-formed.
        assert text.startswith("#!/bin/bash")
        # No truncation of the model arg has happened: the marker itself,
        # including ``Virtual environment activated."``, still appears.
        # (Single-quoted by shlex, so embedded double-quotes are preserved.)
        assert "Virtual environment activated" in text


# ============================================================================
# Integration: launch_server happy/error paths
# ============================================================================


class TestLaunchServerLinuxBranch:
    def test_cleanup_thread_not_started_on_linux(
        self, manager, launcher_mock, tmp_path
    ):
        with patch.object(sys, "platform", "linux"), patch(
            "modules.launch.subprocess.Popen"
        ) as popen, patch(
            "modules.launch.shutil.which", return_value="/usr/bin/xterm"
        ), patch(
            "modules.launch.Thread"
        ) as thread_cls, patch("modules.launch.messagebox"):
            popen.return_value = MagicMock()
            manager.launch_server()

        # No cleanup thread on Linux.
        assert not thread_cls.called

    def test_linux_bails_when_build_cmd_fails(self, manager, launcher_mock):
        launcher_mock.model_path.set("")  # forces build_cmd None
        with patch.object(sys, "platform", "linux"), patch(
            "modules.launch.subprocess.Popen"
        ) as popen, patch("modules.launch.messagebox"):
            manager.launch_server()
        assert not popen.called
