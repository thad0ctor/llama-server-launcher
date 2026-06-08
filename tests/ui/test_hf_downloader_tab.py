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

    Windows expects ``venv/Scripts/python.exe``; POSIX expects
    ``venv/bin/python``. The earlier hard-coded POSIX layout caused
    ``_current_active_venv_path()`` to return "" on Windows, which routed
    tests into a blocking ``messagebox.showerror`` and wedged the runner.
    """
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    if sys.platform.startswith("win"):
        bindir = repo_dir / "venv" / "Scripts"
        exe_name = "python.exe"
    else:
        bindir = repo_dir / "venv" / "bin"
        exe_name = "python"
    bindir.mkdir(parents=True)
    python = bindir / exe_name
    python.write_text("", encoding="utf-8")
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
    repo_dir, _python = active_venv
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
    assert tab.revision_var.get() == "main"


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
    tab._handle_event(
        {
            "event": "listing",
            "repo_id": "TheBloke/Test",
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

    assert "failed" in tab.status_var.get()


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


def test_cancel_before_worker_publishes_process_does_not_pin_handle(
    hf_launcher_stub, active_venv, monkeypatch
):
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
