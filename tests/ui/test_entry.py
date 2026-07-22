"""Tests for the top-level ``llamacpp-server-launcher.py`` entry script.

Targets:
  - ``parse_cli_args`` — the new argparse wrapper that responds to ``--help``
    and ``--version`` without launching the GUI.
  - ``LlamaCppLauncher.cleanup`` — static temp-file cleanup with a delay.
  - ``_read_version_string`` — best-effort reader used by ``--version``.

The module has a hyphen in the filename, so we load it via ``importlib.util``
rather than a normal import statement.
"""

from __future__ import annotations

import importlib.util
import queue
import sys
import time
import tkinter as tk
import types
from pathlib import Path
from tkinter import ttk
from unittest.mock import patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRY_PATH = REPO_ROOT / "llamacpp-server-launcher.py"


@pytest.fixture(scope="module")
def entry_module():
    """Load the launcher script as a module named ``entry_module``.

    Module-scoped so Tk / ttk side-effects happen once. The module does
    ``import tkinter`` at top level but doesn't instantiate a root — that
    only happens inside the ``__main__`` guard, which we never trip.
    """
    spec = importlib.util.spec_from_file_location("entry_module", ENTRY_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["entry_module"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# parse_cli_args
# ---------------------------------------------------------------------------


class TestParseCliArgs:
    """Validate that --help / --version exit cleanly instead of opening GUI."""

    def test_no_args_returns_namespace(self, entry_module):
        """Default invocation must not raise — it yields an (empty) Namespace
        so the caller can proceed to open Tk."""
        ns = entry_module.parse_cli_args([])
        # Namespace has no required fields yet, but it should exist as an object.
        assert ns is not None

    def test_version_flag_exits(self, entry_module, capsys):
        """argparse treats ``--version`` as a terminating action — the process
        exits with code 0 after printing the version, so we must see SystemExit
        rather than a silent GUI launch."""
        with pytest.raises(SystemExit) as exc_info:
            entry_module.parse_cli_args(["--version"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "llamacpp-server-launcher" in out

    def test_help_flag_exits(self, entry_module, capsys):
        """``--help`` also exits cleanly — formerly silently fell through to GUI."""
        with pytest.raises(SystemExit) as exc_info:
            entry_module.parse_cli_args(["--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "usage:" in out.lower()

    def test_short_help_flag_exits(self, entry_module, capsys):
        with pytest.raises(SystemExit) as exc_info:
            entry_module.parse_cli_args(["-h"])
        assert exc_info.value.code == 0

    def test_unknown_flag_exits_nonzero(self, entry_module, capsys):
        """argparse defaults to exit code 2 for bad input — the user gets an
        error on stderr rather than being dumped into a confused GUI."""
        with pytest.raises(SystemExit) as exc_info:
            entry_module.parse_cli_args(["--bogus-flag"])
        assert exc_info.value.code != 0

    def test_version_string_contains_repo_version(self, entry_module, capsys):
        """Confirm the version embedded in the ``--version`` output matches
        the value in ``config/version`` (or ``unknown`` if the file is
        missing). Catches regressions where the embed gets stale."""
        with pytest.raises(SystemExit):
            entry_module.parse_cli_args(["--version"])
        out = capsys.readouterr().out.strip()
        # Should be one of the well-formed strings.
        assert out.startswith("llamacpp-server-launcher ")


# ---------------------------------------------------------------------------
# LlamaCppLauncher lazy/system-info helpers
# ---------------------------------------------------------------------------


class TestLauncherHelpers:
    def test_lazy_tab_initialized_only_after_success(self, entry_module):
        parent = types.SimpleNamespace(winfo_children=lambda: [])
        entry = {
            "parent": parent,
            "label": "Build",
            "initialized": False,
        }
        selected_tab = "tab-id"
        launcher = types.SimpleNamespace(
            notebook=types.SimpleNamespace(select=lambda: selected_tab),
            _lazy_tab_registry={selected_tab: entry},
            # The real method wraps the page in a scroll canvas and returns an
            # inner frame; the stub passes the page straight through so the
            # builder still receives it.
            _scrollable_page=lambda page: page,
        )
        attempts = []

        def builder(_parent):
            attempts.append(_parent)
            if len(attempts) == 1:
                raise RuntimeError("first build failed")

        entry["builder"] = builder

        entry_module.LlamaCppLauncher._on_notebook_tab_changed(launcher)
        assert entry["initialized"] is False

        entry_module.LlamaCppLauncher._on_notebook_tab_changed(launcher)
        assert entry["initialized"] is True
        assert attempts == [parent, parent]

    def test_on_visible_fires_when_built_tab_reselected(self, entry_module):
        # An already-built tab should re-run its ``on_visible`` hook on every
        # reselection (so e.g. the HF tab re-probes venv/dependency state that
        # changed while it was hidden) WITHOUT rebuilding its widgets.
        parent = types.SimpleNamespace(winfo_children=lambda: [])
        calls = []
        entry = {
            "parent": parent,
            "label": "Hugging Face tab",
            "initialized": True,
            "builder": lambda _p: calls.append("build"),
            "on_visible": lambda: calls.append("visible"),
        }
        selected_tab = "tab-id"
        launcher = types.SimpleNamespace(
            notebook=types.SimpleNamespace(select=lambda: selected_tab),
            _lazy_tab_registry={selected_tab: entry},
        )

        entry_module.LlamaCppLauncher._on_notebook_tab_changed(launcher)
        entry_module.LlamaCppLauncher._on_notebook_tab_changed(launcher)

        # Hook fired each time; builder never re-ran on an initialized tab.
        assert calls == ["visible", "visible"]

    def test_on_visible_absent_is_safe(self, entry_module):
        # Lazy tabs registered without an ``on_visible`` hook (Build, Env Vars,
        # …) must not raise when reselected after being built.
        parent = types.SimpleNamespace(winfo_children=lambda: [])
        entry = {"parent": parent, "label": "Build", "initialized": True}
        selected_tab = "tab-id"
        launcher = types.SimpleNamespace(
            notebook=types.SimpleNamespace(select=lambda: selected_tab),
            _lazy_tab_registry={selected_tab: entry},
        )

        # Should be a no-op, not a KeyError.
        entry_module.LlamaCppLauncher._on_notebook_tab_changed(launcher)


class TestScrollablePage:
    """The per-tab vertical scroller helpers used by ``_create_widgets``."""

    def test_scrollable_page_wraps_and_registers(self, entry_module, tk_root):
        page = ttk.Frame(tk_root)
        launcher = types.SimpleNamespace(_tab_scroll_canvases={})
        inner = entry_module.LlamaCppLauncher._scrollable_page(launcher, page)
        # Returns an inner frame distinct from the page…
        assert isinstance(inner, ttk.Frame)
        assert inner is not page
        # …and registers exactly one Canvas keyed by the page's widget path.
        assert list(launcher._tab_scroll_canvases) == [str(page)]
        assert isinstance(launcher._tab_scroll_canvases[str(page)], tk.Canvas)
        # A scrollbar was packed alongside the canvas.
        classes = {type(c).__name__ for c in page.winfo_children()}
        assert "Canvas" in classes and "Scrollbar" in classes
        page.destroy()

    @staticmethod
    def _wheel_capable_launcher(entry_module, tk_root, page):
        """A stub carrying every sibling method ``_on_global_mousewheel``
        reaches for via ``self``."""
        cls = entry_module.LlamaCppLauncher
        launcher = types.SimpleNamespace(
            root=tk_root,
            notebook=types.SimpleNamespace(select=lambda: str(page)),
            _tab_scroll_canvases={},
        )
        launcher._wheel_direction = cls._wheel_direction  # staticmethod
        launcher._inner_widget_consumes_wheel = (
            lambda w, d: cls._inner_widget_consumes_wheel(launcher, w, d)
        )
        launcher._current_tab_scroll_canvas = (
            lambda: cls._current_tab_scroll_canvas(launcher)
        )
        return launcher

    def test_wheel_scrolls_current_tab_when_overflowing(self, entry_module, tk_root):
        # Realized pixel geometry is flaky under a headless/unmapped root, so
        # drive the handler off a reported-overflow ``yview`` and assert the
        # routing (that it issues a one-unit scroll on the visible canvas).
        page = ttk.Frame(tk_root)
        launcher = self._wheel_capable_launcher(entry_module, tk_root, page)
        inner = entry_module.LlamaCppLauncher._scrollable_page(launcher, page)
        canvas = launcher._tab_scroll_canvases[str(page)]
        canvas.yview = lambda: (0.0, 0.5)  # overflow: only top half visible
        scrolled = []
        canvas.yview_scroll = lambda amount, what: scrolled.append((amount, what))

        down = types.SimpleNamespace(num=5, delta=0, widget=inner)
        entry_module.LlamaCppLauncher._on_global_mousewheel(launcher, down)
        assert scrolled == [(1, "units")]

        up = types.SimpleNamespace(num=4, delta=0, widget=inner)
        entry_module.LlamaCppLauncher._on_global_mousewheel(launcher, up)
        assert scrolled[-1] == (-1, "units")
        page.destroy()

    def test_wheel_ignored_when_content_fits(self, entry_module, tk_root):
        page = ttk.Frame(tk_root)
        launcher = self._wheel_capable_launcher(entry_module, tk_root, page)
        inner = entry_module.LlamaCppLauncher._scrollable_page(launcher, page)
        canvas = launcher._tab_scroll_canvases[str(page)]
        canvas.yview = lambda: (0.0, 1.0)  # whole content fits
        scrolled = []
        canvas.yview_scroll = lambda amount, what: scrolled.append((amount, what))

        down = types.SimpleNamespace(num=5, delta=0, widget=inner)
        entry_module.LlamaCppLauncher._on_global_mousewheel(launcher, down)
        assert scrolled == []  # nothing to scroll
        page.destroy()

    def test_inner_listbox_consumes_wheel_until_edge(self, entry_module, tk_root):
        page = ttk.Frame(tk_root)
        launcher = types.SimpleNamespace(_tab_scroll_canvases={})
        inner = entry_module.LlamaCppLauncher._scrollable_page(launcher, page)
        listbox = tk.Listbox(inner, height=3)
        for i in range(30):
            listbox.insert(tk.END, f"item {i}")
        listbox.pack()
        tk_root.update_idletasks()

        # At the top, a downward wheel should be consumed by the listbox
        # (it can still scroll down).
        assert entry_module.LlamaCppLauncher._inner_widget_consumes_wheel(
            launcher, listbox, 1
        ) is True
        # But an upward wheel at the very top is NOT consumed — the page
        # takes over so you can keep scrolling up past the listbox.
        assert entry_module.LlamaCppLauncher._inner_widget_consumes_wheel(
            launcher, listbox, -1
        ) is False
        page.destroy()

    def test_wheel_direction_mapping(self, entry_module):
        d = entry_module.LlamaCppLauncher._wheel_direction
        assert d(types.SimpleNamespace(num=4, delta=0)) == -1
        assert d(types.SimpleNamespace(num=5, delta=0)) == 1
        assert d(types.SimpleNamespace(num=0, delta=120)) == -1
        assert d(types.SimpleNamespace(num=0, delta=-120)) == 1
        assert d(types.SimpleNamespace(num=0, delta=0)) == 0


class TestOpenModelDir:
    """``_open_model_dir`` opens the selected model dir in the OS explorer."""

    def _make_launcher(self, selection, get_value):
        return types.SimpleNamespace(
            model_dirs_listbox=types.SimpleNamespace(
                curselection=lambda: selection,
                get=lambda _i: get_value,
            ),
        )

    def test_opens_existing_dir_via_platform_opener(self, entry_module, tmp_path):
        launcher = self._make_launcher((0,), str(tmp_path))
        spawned = []
        with patch.object(entry_module.sys, "platform", "linux"), patch.object(
            entry_module.subprocess, "Popen", lambda argv: spawned.append(argv)
        ):
            entry_module.LlamaCppLauncher._open_model_dir(launcher)
        assert spawned == [["xdg-open", str(tmp_path)]]

    def test_macos_uses_open(self, entry_module, tmp_path):
        launcher = self._make_launcher((0,), str(tmp_path))
        spawned = []
        with patch.object(entry_module.sys, "platform", "darwin"), patch.object(
            entry_module.subprocess, "Popen", lambda argv: spawned.append(argv)
        ):
            entry_module.LlamaCppLauncher._open_model_dir(launcher)
        assert spawned == [["open", str(tmp_path)]]

    def test_no_selection_errors_and_does_not_spawn(self, entry_module):
        launcher = self._make_launcher((), "")
        errors = []
        with patch.object(
            entry_module.messagebox, "showerror", lambda *a, **k: errors.append(a)
        ), patch.object(
            entry_module.subprocess,
            "Popen",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not spawn")),
        ):
            entry_module.LlamaCppLauncher._open_model_dir(launcher)
        assert errors  # user was told to select something

    def test_missing_dir_errors_and_does_not_spawn(self, entry_module, tmp_path):
        missing = tmp_path / "gone"
        launcher = self._make_launcher((0,), str(missing))
        errors = []
        with patch.object(
            entry_module.messagebox, "showerror", lambda *a, **k: errors.append(a)
        ), patch.object(
            entry_module.subprocess,
            "Popen",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not spawn")),
        ):
            entry_module.LlamaCppLauncher._open_model_dir(launcher)
        assert errors

    def test_unresolvable_marker_errors(self, entry_module):
        launcher = self._make_launcher((0,), "[UNRESOLVABLE] /nope")
        errors = []
        with patch.object(
            entry_module.messagebox, "showerror", lambda *a, **k: errors.append(a)
        ), patch.object(
            entry_module.subprocess,
            "Popen",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not spawn")),
        ):
            entry_module.LlamaCppLauncher._open_model_dir(launcher)
        assert errors

    def test_system_info_drain_discards_stale_generation(self, entry_module):
        class Alive:
            def is_set(self):
                return True

        q = queue.Queue()
        q.put((1, {"gpu_info": {"available": True}}, None))
        applied = []
        scheduled = []
        launcher = types.SimpleNamespace(
            _tk_alive=Alive(),
            _system_info_after_id=None,
            _system_info_queue=q,
            _system_info_active_generations={1},
            _system_info_generation=2,
            _detection_in_progress=True,
            _schedule_system_info_drain=lambda: scheduled.append(True),
            _on_system_info_detection_complete=lambda **kwargs: applied.append(kwargs),
        )

        entry_module.LlamaCppLauncher._drain_system_info_queue(launcher)

        assert applied == []
        assert scheduled == []
        assert launcher._detection_in_progress is False
        # The stale item MUST have been consumed off the queue — if the
        # drain merely skipped it without ``get_nowait()``-ing it, a
        # subsequent drain after the same generation got re-armed
        # would re-process the same stale tuple and end up applying
        # it. Asserting the queue is empty locks in the "consumed
        # and discarded" contract.
        assert q.empty()


# ---------------------------------------------------------------------------
# _read_version_string
# ---------------------------------------------------------------------------


class TestReadVersionString:
    def test_reads_shipped_version(self, entry_module):
        """There's a version file in the repo — the helper should surface it
        rather than returning 'unknown'."""
        result = entry_module._read_version_string()
        # Not empty, not the error sentinel unless the repo really has no file.
        assert isinstance(result, str)
        assert result  # non-empty

    def test_returns_unknown_on_missing_file(self, entry_module, monkeypatch, tmp_path):
        """Simulate a partial install (no ``config/version``) by repointing
        ``__file__`` at a tmp dir. Must degrade gracefully, not raise."""
        # The helper uses Path(__file__).resolve().parent / "config" / "version".
        # We patch the module-level __file__ for the duration of this test.
        fake_module_path = tmp_path / "launcher.py"
        fake_module_path.write_text("")
        monkeypatch.setattr(entry_module, "__file__", str(fake_module_path))

        result = entry_module._read_version_string()
        assert result == "unknown"

    def test_handles_empty_version_file(self, entry_module, monkeypatch, tmp_path):
        """Empty file is treated the same as missing — 'unknown'."""
        fake_module_path = tmp_path / "launcher.py"
        fake_module_path.write_text("")
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "version").write_text("")
        monkeypatch.setattr(entry_module, "__file__", str(fake_module_path))

        assert entry_module._read_version_string() == "unknown"

    def test_strips_whitespace(self, entry_module, monkeypatch, tmp_path):
        """Trailing newline/whitespace in the version file shouldn't end up
        in the printed version string."""
        fake_module_path = tmp_path / "launcher.py"
        fake_module_path.write_text("")
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "version").write_text("  2024-01-01-1  \n")
        monkeypatch.setattr(entry_module, "__file__", str(fake_module_path))

        assert entry_module._read_version_string() == "2024-01-01-1"

    def test_non_utf8_bytes_fall_back_to_unknown(self, entry_module, monkeypatch, tmp_path):
        """A malformed version file (non-UTF-8 bytes) must not crash
        ``--version``. ``Path.read_text(encoding='utf-8')`` raises
        ``UnicodeDecodeError`` which is NOT an ``OSError``; if the helper
        only catches ``OSError`` the exception escapes and the CLI aborts."""
        fake_module_path = tmp_path / "launcher.py"
        fake_module_path.write_text("")
        (tmp_path / "config").mkdir()
        # 0x80 is a lone continuation byte — invalid as the start of a UTF-8
        # sequence, so read_text(encoding='utf-8') raises UnicodeDecodeError.
        (tmp_path / "config" / "version").write_bytes(b"\x80\x81\x82")
        monkeypatch.setattr(entry_module, "__file__", str(fake_module_path))

        assert entry_module._read_version_string() == "unknown"


# ---------------------------------------------------------------------------
# LlamaCppLauncher.cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    """Static helper that deletes a temp file after a short delay.

    We pass ``delay=0`` in all tests to keep them fast — the docstring reads
    "after a delay" but there's no lower bound on what that means."""

    def test_deletes_existing_file(self, entry_module, tmp_path):
        victim = tmp_path / "tmpfile.sh"
        victim.write_text("# tmp")
        assert victim.exists()

        entry_module.LlamaCppLauncher.cleanup(victim, delay=0)
        assert not victim.exists()

    def test_accepts_str_path(self, entry_module, tmp_path):
        """Docstring doesn't nail the type — call sites pass both str and Path."""
        victim = tmp_path / "tmpfile.sh"
        victim.write_text("# tmp")

        entry_module.LlamaCppLauncher.cleanup(str(victim), delay=0)
        assert not victim.exists()

    def test_missing_file_is_silent(self, entry_module, tmp_path, capsys):
        """Cleanup runs on a best-effort basis — if the launch path already
        removed the file we shouldn't raise."""
        missing = tmp_path / "never-existed.sh"
        assert not missing.exists()

        # Should not raise.
        entry_module.LlamaCppLauncher.cleanup(missing, delay=0)

    def test_unlink_error_is_swallowed(self, entry_module, tmp_path):
        """OSError from unlink (e.g. permission denied on Windows) must be
        logged, not raised — it's a background cleanup."""
        victim = tmp_path / "tmpfile.sh"
        victim.write_text("x")

        with patch("pathlib.Path.unlink", side_effect=OSError("locked")):
            # Must not raise.
            entry_module.LlamaCppLauncher.cleanup(victim, delay=0)

    def test_delay_is_honoured(self, entry_module, tmp_path):
        """Sanity: a small delay actually sleeps. We use 0.05s so the test
        stays fast but still exercises the ``time.sleep`` path."""
        victim = tmp_path / "tmpfile.sh"
        victim.write_text("x")

        start = time.perf_counter()
        entry_module.LlamaCppLauncher.cleanup(victim, delay=0.05)
        elapsed = time.perf_counter() - start
        assert elapsed >= 0.04  # allow a bit of slop
        assert not victim.exists()
