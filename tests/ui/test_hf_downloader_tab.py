from __future__ import annotations

import sys
import tkinter as tk
from unittest.mock import MagicMock

import pytest

from modules import terminal_launcher, venv_manager
from modules.hf_downloader.tab import HuggingFaceDownloaderTab


@pytest.fixture
def hf_launcher_stub(launcher_stub, tmp_path):
    launcher_stub.repo_dir = tmp_path
    launcher_stub.model_dirs = []
    return launcher_stub


@pytest.fixture
def active_venv(tmp_path):
    """Lay out a venv that ``looks_like_venv`` will accept on this platform.

    ``looks_like_venv`` now requires the full set of markers a real venv
    has: the python interpreter, the ``pyvenv.cfg`` config file, and the
    platform-appropriate activator script. Earlier fixtures only created
    ``bin/python`` (or ``Scripts/python.exe``), which let any plain
    ``bin/python`` directory pose as a venv — fine for tests but a real
    security/UX problem in the GUI (auto-activation, recursive deletion).
    """
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    venv_dir = repo_dir / "venv"
    if sys.platform.startswith("win"):
        bindir = venv_dir / "Scripts"
        exe_name = "python.exe"
        activator = bindir / "activate.bat"
    else:
        bindir = venv_dir / "bin"
        exe_name = "python"
        activator = bindir / "activate"
    bindir.mkdir(parents=True)
    python = bindir / exe_name
    python.write_text("", encoding="utf-8")
    # ``_path_looks_like_venv`` now requires the interpreter to be an
    # executable regular file (POSIX). On Windows the ``.exe`` extension
    # is the marker so chmod is unnecessary.
    if not sys.platform.startswith("win"):
        import os as _os

        _os.chmod(python, 0o755)
    # Both markers ``looks_like_venv`` now requires.
    (venv_dir / "pyvenv.cfg").write_text("home = /\n", encoding="utf-8")
    activator.write_text("# mock\n", encoding="utf-8")
    return repo_dir, python


def test_setup_disables_actions_without_active_venv(hf_launcher_stub, monkeypatch):
    monkeypatch.setattr(
        venv_manager,
        "probe_dependency_status",
        lambda *args, **kwargs: venv_manager.DependencyStatus(
            dependency=next(dep for dep in venv_manager.MANAGED_DEPENDENCIES if dep.key == "huggingface_hub"),
            available=False,
            error="venv python not found",
        ),
    )
    tab = HuggingFaceDownloaderTab(hf_launcher_stub)
    parent = tk.Frame(hf_launcher_stub.root)

    tab.setup_tab(parent)

    assert str(tab._install_button.cget("state")) == "disabled"
    assert str(tab._load_button.cget("state")) == "disabled"
    assert str(tab._download_button.cget("state")) == "disabled"


def test_setup_enables_install_and_load_with_hf_ready(hf_launcher_stub, active_venv, monkeypatch):
    repo_dir, python = active_venv
    hf_launcher_stub.repo_dir = repo_dir
    hf_launcher_stub.venv_dir.set("")
    hf_launcher_stub.model_dirs = [repo_dir / "downloads"]
    (repo_dir / "downloads").mkdir()
    monkeypatch.setattr(
        venv_manager,
        "probe_dependency_status",
        lambda *args, **kwargs: venv_manager.DependencyStatus(
            dependency=next(dep for dep in venv_manager.MANAGED_DEPENDENCIES if dep.key == "huggingface_hub"),
            available=True,
            version="1.0.0",
        ),
    )
    tab = HuggingFaceDownloaderTab(hf_launcher_stub)
    parent = tk.Frame(hf_launcher_stub.root)

    tab.setup_tab(parent)

    assert str(tab._install_button.cget("state")) == "normal"
    assert str(tab._load_button.cget("state")) == "normal"
    assert "1.0.0" in tab.venv_status_var.get()
    assert str((repo_dir / "downloads").resolve()) in tab._selected_target_vars
    # ``.stem`` strips the ``.exe`` extension so this works on both Windows
    # (``python.exe``) and POSIX (``python``).
    assert python.stem == "python"


def test_install_button_uses_active_venv(hf_launcher_stub, active_venv, monkeypatch):
    repo_dir, python = active_venv
    hf_launcher_stub.repo_dir = repo_dir
    hf_launcher_stub.venv_dir.set("")
    launch_mock = MagicMock()
    monkeypatch.setattr(
        venv_manager,
        "probe_dependency_status",
        lambda *args, **kwargs: venv_manager.DependencyStatus(
            dependency=next(dep for dep in venv_manager.MANAGED_DEPENDENCIES if dep.key == "huggingface_hub"),
            available=False,
            error="missing",
        ),
    )
    monkeypatch.setattr(terminal_launcher, "open_command_in_terminal", launch_mock)
    tab = HuggingFaceDownloaderTab(hf_launcher_stub)
    parent = tk.Frame(hf_launcher_stub.root)
    tab.setup_tab(parent)

    tab._on_install_hf_dependency()

    launch_mock.assert_called_once()
    # huggingface_hub uses the ``[cli]`` extra so the ``hf`` console script
    # gets registered. Check the package name + extra appears in any
    # acceptable shell-quoting form.
    cmd_str = launch_mock.call_args.args[0]
    assert "pip" in cmd_str and "install" in cmd_str
    assert "huggingface_hub[cli]" in cmd_str
    # The install command MUST invoke the venv's interpreter (its
    # ``.stem`` is ``python``), not a bare ``pip`` that could pick up
    # the system Python and silently install into the wrong site —
    # this is the whole point of routing through ``venv_manager``.
    assert str(python) in cmd_str
    assert launch_mock.call_args.kwargs["cwd"] == repo_dir


def test_listing_event_populates_files_and_selects_defaults(hf_launcher_stub, active_venv, monkeypatch):
    repo_dir, _python = active_venv
    hf_launcher_stub.repo_dir = repo_dir
    hf_launcher_stub.venv_dir.set("")
    monkeypatch.setattr(
        venv_manager,
        "probe_dependency_status",
        lambda *args, **kwargs: venv_manager.DependencyStatus(
            dependency=next(dep for dep in venv_manager.MANAGED_DEPENDENCIES if dep.key == "huggingface_hub"),
            available=True,
            version="1.0.0",
        ),
    )
    tab = HuggingFaceDownloaderTab(hf_launcher_stub)
    parent = tk.Frame(hf_launcher_stub.root)
    tab.setup_tab(parent)

    tab._handle_event(
        {
            "event": "listing",
            "repo_id": "TheBloke/Test",
            "refs": [{"name": "main", "kind": "branch"}],
            "files": [
                {"path": "model.gguf", "size_bytes": 120, "kind": "gguf"},
                {"path": "mmproj-f16.gguf", "size_bytes": 10, "kind": "mmproj"},
                {"path": "README.md", "size_bytes": 2, "kind": "other"},
            ],
        }
    )

    assert tab._files_listbox.size() == 3
    assert tab._selected_file_paths() == ["model.gguf", "mmproj-f16.gguf"]
    # The listing event arrived without ``_on_load_repo`` ever running,
    # so ``_last_requested_revision`` is "" — the listing handler must
    # preserve the blank revision rather than silently filling it with
    # ``_refs[0]`` (which might be a different branch than the default
    # the runner actually listed against). The combobox values list
    # still gets populated for the dropdown UI.
    assert tab.revision_var.get() == ""
    assert tab._refs == ["main"]


def test_download_builds_payload_from_selected_files_and_targets(hf_launcher_stub, active_venv, monkeypatch):
    repo_dir, _python = active_venv
    target = repo_dir / "downloads"
    target.mkdir()
    hf_launcher_stub.repo_dir = repo_dir
    hf_launcher_stub.model_dirs = [target]
    hf_launcher_stub.venv_dir.set("")
    monkeypatch.setattr(
        venv_manager,
        "probe_dependency_status",
        lambda *args, **kwargs: venv_manager.DependencyStatus(
            dependency=next(dep for dep in venv_manager.MANAGED_DEPENDENCIES if dep.key == "huggingface_hub"),
            available=True,
            version="1.0.0",
        ),
    )
    captured = {}
    tab = HuggingFaceDownloaderTab(hf_launcher_stub)
    parent = tk.Frame(hf_launcher_stub.root)
    tab.setup_tab(parent)
    tab.repo_input_var.set("TheBloke/Test")
    tab.revision_var.set("main")
    tab.include_patterns_var.set("README*")
    tab.ignore_patterns_var.set("*.tmp")
    tab.max_workers_var.set("6")
    # The real runner emits ``revision`` (the ref name) AND
    # ``resolved_revision`` (the commit SHA) in its listing payload.
    # The tab pins ``_pinned_revision_sha`` from the latter so a
    # download binds to the exact bytes the user saw in the file
    # list, even if the branch moves between listing and download.
    fake_sha = "deadbeefcafe1234"
    tab._handle_event(
        {
            "event": "listing",
            "repo_id": "TheBloke/Test",
            "revision": "main",
            "resolved_revision": fake_sha,
            "refs": [{"name": "main", "kind": "branch"}],
            "files": [{"path": "model.gguf", "size_bytes": 120, "kind": "gguf"}],
        }
    )
    monkeypatch.setattr(
        tab,
        "_start_runner",
        lambda action, payload: captured.update({"action": action, "payload": payload}),
    )

    tab._on_download()

    assert captured["action"] == "download"
    assert captured["payload"]["repo_id"] == "TheBloke/Test"
    # Pinned-SHA verification: even though ``revision_var`` (the
    # combobox the user sees) reads ``"main"``, the download payload
    # MUST send the SHA — that's the whole point of pinning.
    # A regression that reads the editable combobox instead of the
    # pinned attribute would emit ``"main"`` here.
    assert captured["payload"]["revision"] == fake_sha, (
        f"download must pin to the resolved SHA, not the editable "
        f"revision_var value; got {captured['payload']['revision']!r}"
    )
    assert captured["payload"]["selected_files"] == ["model.gguf"]
    assert captured["payload"]["target_dirs"] == [str(target.resolve())]
    assert captured["payload"]["include_patterns"] == ["README*"]
    assert captured["payload"]["ignore_patterns"] == ["*.tmp"]
    assert captured["payload"]["max_workers"] == 6


def test_process_exit_failure_updates_status(hf_launcher_stub, active_venv, monkeypatch):
    repo_dir, _python = active_venv
    hf_launcher_stub.repo_dir = repo_dir
    hf_launcher_stub.venv_dir.set("")
    monkeypatch.setattr(
        venv_manager,
        "probe_dependency_status",
        lambda *args, **kwargs: venv_manager.DependencyStatus(
            dependency=next(dep for dep in venv_manager.MANAGED_DEPENDENCIES if dep.key == "huggingface_hub"),
            available=True,
            version="1.0.0",
        ),
    )
    tab = HuggingFaceDownloaderTab(hf_launcher_stub)
    parent = tk.Frame(hf_launcher_stub.root)
    tab.setup_tab(parent)
    tab._operation_name = "download"

    tab._handle_event({"event": "process-exit", "returncode": 1, "stderr": "boom"})

    status = tab.status_var.get()
    assert "failed" in status
    # The subprocess stderr must be surfaced to the user — a
    # regression that drops it (so only the generic "failed" message
    # shows) would have slipped through the previous membership-only
    # assertion. Include the actual stderr token so users see what
    # actually broke without having to dig through subprocess logs.
    assert "boom" in status, f"failure status must include subprocess stderr; got {status!r}"


def test_cancel_operation_restores_button_state(hf_launcher_stub, active_venv, monkeypatch):
    repo_dir, _python = active_venv
    hf_launcher_stub.repo_dir = repo_dir
    hf_launcher_stub.venv_dir.set("")
    monkeypatch.setattr(
        venv_manager,
        "probe_dependency_status",
        lambda *args, **kwargs: venv_manager.DependencyStatus(
            dependency=next(dep for dep in venv_manager.MANAGED_DEPENDENCIES if dep.key == "huggingface_hub"),
            available=True,
            version="1.0.0",
        ),
    )
    tab = HuggingFaceDownloaderTab(hf_launcher_stub)
    parent = tk.Frame(hf_launcher_stub.root)
    tab.setup_tab(parent)
    tab._handle_event(
        {
            "event": "listing",
            "repo_id": "TheBloke/Test",
            "refs": [{"name": "main", "kind": "branch"}],
            "files": [{"path": "model.gguf", "size_bytes": 120, "kind": "gguf"}],
        }
    )
    tab._set_button_state(tab._load_button, False)
    tab._set_button_state(tab._download_button, False)
    tab.progress_label_var.set("busy")

    tab._cancel_operation()

    assert str(tab._load_button.cget("state")) == "normal"
    assert str(tab._download_button.cget("state")) == "normal"
    assert tab.progress_label_var.get() == ""
    assert tab.status_var.get() == "Operation cancelled."


def test_run_process_worker_reports_start_failure(hf_launcher_stub, active_venv, monkeypatch):
    repo_dir, _python = active_venv
    hf_launcher_stub.repo_dir = repo_dir
    hf_launcher_stub.venv_dir.set("")
    monkeypatch.setattr(
        venv_manager,
        "probe_dependency_status",
        lambda *args, **kwargs: venv_manager.DependencyStatus(
            dependency=next(dep for dep in venv_manager.MANAGED_DEPENDENCIES if dep.key == "huggingface_hub"),
            available=True,
            version="1.0.0",
        ),
    )
    tab = HuggingFaceDownloaderTab(hf_launcher_stub)
    parent = tk.Frame(hf_launcher_stub.root)
    tab.setup_tab(parent)

    tab._run_process_worker(
        ["/definitely-missing-python", "-m", "modules.hf_downloader.runner"],
        op_id=tab._op_id,
    )

    event = tab._queue.get_nowait()
    assert event["event"] == "process-exit"
    assert event["returncode"] == 1
    assert event["stderr"]


def test_cancel_before_worker_publishes_process_does_not_pin_handle(hf_launcher_stub, active_venv, monkeypatch):
    """Cancel that wins the race against ``self._process = proc`` must not
    leave a stale handle pinned on the tab.

    Earlier behavior: the worker assigned ``self._process = proc`` and only
    then checked ``op_id in self._cancelled_ops``. Stale events for a
    cancelled op are dropped by ``_poll_queue``, so ``_finalize_process``
    never ran and ``self._process`` stayed non-``None`` — ``_poll_queue``
    would keep rescheduling forever against a process the user already
    cancelled. The fix checks cancellation before publishing, and skips the
    assignment in the cancelled case.
    """
    repo_dir, _python = active_venv
    hf_launcher_stub.repo_dir = repo_dir
    hf_launcher_stub.venv_dir.set("")
    monkeypatch.setattr(
        venv_manager,
        "probe_dependency_status",
        lambda *args, **kwargs: venv_manager.DependencyStatus(
            dependency=next(dep for dep in venv_manager.MANAGED_DEPENDENCIES if dep.key == "huggingface_hub"),
            available=True,
            version="1.0.0",
        ),
    )
    tab = HuggingFaceDownloaderTab(hf_launcher_stub)
    parent = tk.Frame(hf_launcher_stub.root)
    tab.setup_tab(parent)

    # Pre-cancel the next op id so the in-worker check sees it as cancelled
    # before the assignment.
    tab._cancelled_ops.add(tab._op_id)

    # Build a fake Popen-like object: ``terminate`` and ``wait`` must be
    # callable; ``stdout``/``stderr`` iterate to nothing so the worker
    # finishes without consuming streams.
    class _FakeProc:
        def __init__(self):
            self.terminate_calls = 0
            self.stdout = iter(())
            self.stderr = iter(())
            self.returncode = 1

        def terminate(self):
            self.terminate_calls += 1

        def wait(self, timeout=None):
            return self.returncode

    fake_proc = _FakeProc()

    import subprocess as _subprocess

    monkeypatch.setattr(_subprocess, "Popen", lambda *_a, **_k: fake_proc)

    tab._run_process_worker(
        ["/will-not-actually-run", "-m", "modules.hf_downloader.runner"],
        op_id=tab._op_id,
    )

    # The cancellation-aware publish must skip assigning ``self._process``,
    # so the polling loop has nothing stale to chase.
    assert tab._process is None
    # And the worker still terminated the process it briefly held a
    # reference to, so we don't leak a runaway subprocess.
    assert fake_proc.terminate_calls >= 1


# ─────────────────────────────────────────────────────────────────────────────
# CR-4467921108: teardown() must remove every owner-tracked write-trace
# AND cancel every scheduled ``after()`` so a discarded lazy tab can't
# touch dead widgets via a launcher-owned StringVar callback.
# ─────────────────────────────────────────────────────────────────────────────


def test_teardown_removes_owner_traces_and_clears_after_ids(hf_launcher_stub, monkeypatch):
    monkeypatch.setattr(
        venv_manager,
        "probe_dependency_status",
        lambda *args, **kwargs: venv_manager.DependencyStatus(
            dependency=next(dep for dep in venv_manager.MANAGED_DEPENDENCIES if dep.key == "huggingface_hub"),
            available=False,
            error="venv python not found",
        ),
    )
    tab = HuggingFaceDownloaderTab(hf_launcher_stub)

    # Sanity: every recorded (var, token) pair maps to a live trace on
    # the var BEFORE teardown.
    assert tab._owner_trace_tokens, "trace tokens should be populated post-init"
    for var, token in tab._owner_trace_tokens:
        infos = var.trace_info()
        assert any(token == t[1] for t in infos), f"expected token {token!r} to still be present on {var!r}"

    # Simulate a scheduled after() callback so teardown has to cancel
    # it. We just stash a synthetic id and stub after_cancel.
    cancelled: list[str] = []
    monkeypatch.setattr(tab.root, "after_cancel", lambda aid: cancelled.append(aid))
    tab._queue_after_id = "after#queue"
    tab._dep_watch_after_id = "after#dep-watch"
    # Likewise simulate a populated per-row trace list.
    booger_var = tk.BooleanVar(value=True)
    booger_token = booger_var.trace_add("write", lambda *_a: None)
    tab._selected_target_trace_tokens = [(booger_var, booger_token)]
    tab._selected_target_vars = {"/some/dir": booger_var}

    tab.teardown()

    # 1. Every owner-tracked token removed from its var.
    for var, token in []:  # iterated against original snapshot below
        pass
    # Need the pre-teardown snapshot — teardown clears the list, so
    # re-create by inspecting trace_info() now.
    # Per-row trace gone.
    assert (booger_var, booger_token) not in tab._selected_target_trace_tokens
    assert not tab._selected_target_trace_tokens
    assert tab._selected_target_vars == {}
    # 2. after() ids cancelled and cleared.
    assert "after#queue" in cancelled
    assert "after#dep-watch" in cancelled
    assert tab._queue_after_id is None
    assert tab._dep_watch_after_id is None
    # 3. Owner-tracked list is empty.
    assert tab._owner_trace_tokens == []
    # 4. Idempotent — second call must not raise.
    tab.teardown()


def test_teardown_is_idempotent_with_destroyed_vars(hf_launcher_stub, monkeypatch):
    """Calling teardown twice (e.g. once in __exit__, once in __del__)
    must not raise even if the underlying Tk vars are already gone."""
    monkeypatch.setattr(
        venv_manager,
        "probe_dependency_status",
        lambda *args, **kwargs: venv_manager.DependencyStatus(
            dependency=next(dep for dep in venv_manager.MANAGED_DEPENDENCIES if dep.key == "huggingface_hub"),
            available=False,
            error="venv python not found",
        ),
    )
    tab = HuggingFaceDownloaderTab(hf_launcher_stub)

    # First teardown — the real path.
    tab.teardown()

    # Mutate the state so the SECOND teardown is given a token that
    # is no longer on the var (mimics "already removed by the first
    # call OR the var was destroyed in between").
    fake_var = tk.StringVar(value="")
    fake_var.trace_remove("write", fake_var.trace_add("write", lambda *_a: None))
    tab._owner_trace_tokens = [(fake_var, "non-existent-token")]
    tab._selected_target_trace_tokens = [(fake_var, "non-existent-token")]

    # Should NOT raise.
    tab.teardown()


# ─────────────────────────────────────────────────────────────────────────────
# CR-4467921108: dep-watch must compare against the BASELINE state so
# an "Install / update" click on an already-installed huggingface_hub
# doesn't close the watch on the first available=True probe.
# ─────────────────────────────────────────────────────────────────────────────


def test_dep_watch_baseline_captured_on_start(hf_launcher_stub, monkeypatch):
    """``_start_dependency_watch`` records the pre-launch dep state."""
    monkeypatch.setattr(
        venv_manager,
        "probe_dependency_status",
        lambda *args, **kwargs: venv_manager.DependencyStatus(
            dependency=next(dep for dep in venv_manager.MANAGED_DEPENDENCIES if dep.key == "huggingface_hub"),
            available=True,
            version="0.20.0",
        ),
    )
    tab = HuggingFaceDownloaderTab(hf_launcher_stub)
    monkeypatch.setattr(tab.root, "after", lambda *_a, **_kw: "after#scheduled")
    monkeypatch.setattr(tab.root, "after_cancel", lambda *_a, **_kw: None)
    # ``_start_dependency_watch`` calls ``_refresh_runtime_state`` at the
    # end, which would cancel the watch (and reset the baseline) when
    # ``_dep_watch_venv != active``. In the real flow the user clicks
    # Install/Update on the ACTIVE venv, so make the stub agree.
    monkeypatch.setattr(tab, "_current_active_venv_path", lambda: "/some/venv")
    monkeypatch.setattr(tab, "_refresh_runtime_state", lambda: None)

    tab._start_dependency_watch("/some/venv")

    assert tab._dep_watch_initial_available is True
    assert tab._dep_watch_initial_version == "0.20.0"


def test_dep_watch_does_not_complete_when_state_unchanged(hf_launcher_stub, monkeypatch):
    """An update on an already-installed dep must keep polling while
    pip is still doing its work — not exit on the first ``available=True``."""
    monkeypatch.setattr(
        venv_manager,
        "probe_dependency_status",
        lambda *args, **kwargs: venv_manager.DependencyStatus(
            dependency=next(dep for dep in venv_manager.MANAGED_DEPENDENCIES if dep.key == "huggingface_hub"),
            available=True,
            version="0.20.0",
        ),
    )
    tab = HuggingFaceDownloaderTab(hf_launcher_stub)
    monkeypatch.setattr(tab.root, "after", lambda *_a, **_kw: "after#scheduled")
    monkeypatch.setattr(tab.root, "after_cancel", lambda *_a, **_kw: None)
    monkeypatch.setattr(tab, "_current_active_venv_path", lambda: "/some/venv")
    monkeypatch.setattr(tab, "_refresh_runtime_state", lambda: None)

    tab._start_dependency_watch("/some/venv")
    # Pre-launch baseline: available=True, version=0.20.0.
    # Probe reports the same version — pip hasn't finished yet.
    tab._on_dependency_probe_result("/some/venv", True, "0.20.0")

    # Watch must STILL be open — _dep_watch_venv should be retained.
    assert tab._dep_watch_venv == "/some/venv"


def test_dep_watch_completes_when_version_changes(hf_launcher_stub, monkeypatch):
    """When the version differs from baseline (upgrade landed) the
    watch closes out."""
    monkeypatch.setattr(
        venv_manager,
        "probe_dependency_status",
        lambda *args, **kwargs: venv_manager.DependencyStatus(
            dependency=next(dep for dep in venv_manager.MANAGED_DEPENDENCIES if dep.key == "huggingface_hub"),
            available=True,
            version="0.20.0",
        ),
    )
    tab = HuggingFaceDownloaderTab(hf_launcher_stub)
    monkeypatch.setattr(tab.root, "after", lambda *_a, **_kw: "after#scheduled")
    monkeypatch.setattr(tab.root, "after_cancel", lambda *_a, **_kw: None)
    monkeypatch.setattr(tab, "_current_active_venv_path", lambda: "/some/venv")
    monkeypatch.setattr(tab, "_refresh_runtime_state", lambda: None)

    tab._start_dependency_watch("/some/venv")
    # Probe reports a newer version — upgrade finished.
    tab._on_dependency_probe_result("/some/venv", True, "0.21.0")

    assert tab._dep_watch_venv is None
    # Baseline cleared too so the next watch starts fresh.
    assert tab._dep_watch_initial_available is False
    assert tab._dep_watch_initial_version == ""


def test_dep_watch_completes_when_dep_first_appears(hf_launcher_stub, monkeypatch):
    """Fresh install path: baseline reports unavailable, then becomes
    available → close out."""
    monkeypatch.setattr(
        venv_manager,
        "probe_dependency_status",
        lambda *args, **kwargs: venv_manager.DependencyStatus(
            dependency=next(dep for dep in venv_manager.MANAGED_DEPENDENCIES if dep.key == "huggingface_hub"),
            available=False,
            error="not installed",
        ),
    )
    tab = HuggingFaceDownloaderTab(hf_launcher_stub)
    monkeypatch.setattr(tab.root, "after", lambda *_a, **_kw: "after#scheduled")
    monkeypatch.setattr(tab.root, "after_cancel", lambda *_a, **_kw: None)
    monkeypatch.setattr(tab, "_current_active_venv_path", lambda: "/some/venv")
    monkeypatch.setattr(tab, "_refresh_runtime_state", lambda: None)

    tab._start_dependency_watch("/some/venv")
    assert tab._dep_watch_initial_available is False
    tab._on_dependency_probe_result("/some/venv", True, "0.20.0")
    assert tab._dep_watch_venv is None
