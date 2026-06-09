"""Tk UI for browsing and downloading models from Hugging Face."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from modules import terminal_launcher, venv_manager
from modules.hf_downloader.helpers import (
    HfRepoFile,
    build_runner_command,
    collect_target_directory_options,
    normalize_repo_input,
    parse_bool,
    parse_pattern_lines,
    payload_bytes,
    summarize_repo_listing,
)


class HuggingFaceDownloaderTab:
    """Downloader UI backed by a selected virtual environment."""

    POLL_MS = 120
    DEP_WATCH_INTERVAL_MS = 3000
    DEP_WATCH_TIMEOUT_S = 600

    def __init__(self, launcher):
        self.launcher = launcher
        self.root = launcher.root
        # Mirror SettingsTab.__init__: normalize to an absolute,
        # expanded ``Path`` so the HF download flow (``cwd=`` on the
        # terminal launch, venv resolution) sees the SAME repo root
        # regardless of the launcher's incoming shape. A relative or
        # ``~``-prefixed value used to vary by launch context — the
        # downloader would build a venv command rooted at one cwd
        # while the settings tab probed against a different one.
        # ``strict=False`` keeps the path object even if the
        # directory hasn't been created yet.
        raw_repo_dir = getattr(launcher, "repo_dir", None)
        if not isinstance(raw_repo_dir, (str, Path)):
            raw_repo_dir = venv_manager.launcher_repo_dir()
        try:
            self.repo_dir = Path(raw_repo_dir).expanduser().resolve(strict=False)
        except Exception:
            self.repo_dir = Path(raw_repo_dir).expanduser()
        settings = launcher.app_settings
        self.repo_input_var = tk.StringVar(value=settings.get("hf_repo_input", ""))
        self.revision_var = tk.StringVar(value=settings.get("hf_repo_revision", ""))
        self.download_mode_var = tk.StringVar(value=settings.get("hf_download_mode", "selected"))
        self.include_patterns_var = tk.StringVar(value=settings.get("hf_include_patterns", ""))
        self.ignore_patterns_var = tk.StringVar(value=settings.get("hf_ignore_patterns", ""))
        # ``bool("false")`` and ``bool("0")`` are both True in Python; a
        # JSON-edited settings file with ``"hf_force_download": "false"``
        # would silently re-enable destructive behaviour. ``parse_bool``
        # mirrors the runner-side coercion, so a UI-rehydration and a
        # subprocess parse agree on what a stored string means.
        self.force_download_var = tk.BooleanVar(value=parse_bool(settings.get("hf_force_download", False)))
        self.local_files_only_var = tk.BooleanVar(value=parse_bool(settings.get("hf_local_files_only", False)))
        self.max_workers_var = tk.StringVar(value=str(settings.get("hf_max_workers", 4)))
        # Re-entry guard so the validating trace below doesn't recurse when
        # it normalizes its own value.
        self._max_workers_validating = False
        self.token_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Create/select a venv, install huggingface_hub, then load a repo.")
        # Mirror SettingsTab's normalization (see settings_tab.py). The
        # MagicMock-based test fixtures auto-vivify ``launcher.venv_dir``
        # as a Mock that quacks like an attribute but doesn't satisfy
        # ``isinstance(..., tk.StringVar)``; without this guard the
        # downstream ``trace_add`` / ``.get()`` / ``.set()`` calls
        # below would either silently no-op or crash.
        existing_venv_dir = getattr(launcher, "venv_dir", None)
        if not isinstance(existing_venv_dir, tk.StringVar):
            launcher.venv_dir = tk.StringVar(value=settings.get("last_venv_dir", ""))
        self.venv_var = launcher.venv_dir
        self.venv_status_var = tk.StringVar(value="")
        self.progress_label_var = tk.StringVar(value="")
        self._selected_target_vars: dict[str, tk.BooleanVar] = {}
        # (var, trace_token) pairs so ``_refresh_target_rows`` can
        # ``trace_remove`` on the old vars before discarding them — without
        # this, every refresh leaks the trace callback and the var it
        # references.
        self._selected_target_trace_tokens: list[tuple[tk.BooleanVar, str]] = []
        self._file_rows: list[HfRepoFile] = []
        self._file_path_by_index: list[str] = []
        self._refs: list[str] = []
        # Revision the user submitted via ``_on_load_repo``. The listing
        # handler reads this to decide whether to auto-populate the
        # ``revision_var`` field from ``_refs[0]`` — only safe when the
        # request itself was blank (otherwise the user explicitly typed a
        # ref and clearing the box mid-flight is a deliberate edit, not
        # something the listing handler should silently overwrite).
        self._last_requested_revision: str = ""
        # Repo / revision that the LISTING currently in the file rows
        # was actually loaded for. ``_on_download`` checks these so a
        # user who pasted a new repo URL but didn't reload can't
        # accidentally fetch from the wrong source (the file selection
        # is meaningful only for the loaded repo). ``_loaded_revision``
        # uses ``None`` as the "never loaded" sentinel; an explicit
        # ``""`` means "the listing was loaded with a BLANK revision"
        # (i.e. the runner resolved against the default branch) — that
        # must be distinguishable from "user hasn't loaded yet" so
        # ``_on_download`` doesn't fall back to the live revision_var
        # text the user may have edited mid-flight.
        self._loaded_repo_id: str = ""
        self._loaded_revision: str | None = None
        # ``_loaded_revision`` holds the human-facing ref the user
        # asked for (``"main"`` / ``"v1.0"`` / ``""``) and is what
        # the mismatch-detection in ``_on_download`` compares against.
        # ``_pinned_revision_sha`` holds the runner's
        # ``resolved_revision`` (a commit SHA) so the download payload
        # binds to the EXACT commit the listing was generated against
        # — branches and tags move, but a SHA doesn't. Empty string =
        # listing didn't report a SHA (older runner, or repo without
        # one); fall back to the ref name in that case.
        self._pinned_revision_sha: str = ""
        self._target_container = None
        self._files_listbox = None
        self._revision_combo = None
        self._progress = None
        self._install_button = None
        self._load_button = None
        self._download_button = None
        self._cancel_button = None
        self._refresh_targets_button = None
        self._options_toggle_btn = None
        self._options_body = None
        self._options_expanded = False
        self._queue: queue.Queue = queue.Queue()
        self._queue_after_id = None
        self._worker_thread = None
        self._process: subprocess.Popen[str] | None = None
        # ``True`` while ``_cancel_operation`` is mid-flight — the daemon
        # ``_async_terminate`` thread can take up to 7 s to actually
        # kill a stubborn child, and ``_cleanup_process_state`` clears
        # ``self._process`` before then. ``_refresh_runtime_state``
        # treats the runtime as busy while this is set so the Load /
        # Download / Install buttons stay disabled until the worker is
        # really gone.
        self._terminating: bool = False
        self._payload_path: Path | None = None
        self._operation_name = ""
        self._dep_watch_after_id: str | None = None
        self._dep_watch_deadline: float = 0.0
        self._dep_watch_venv: str | None = None
        # Generation token bumped on every _start_runner / _cancel_operation.
        # Worker-thread events carry the op_id captured at launch; events with
        # a stale op_id are dropped by _handle_event. Closes the race where
        # Cancel was clicked before the subprocess handle was published to the
        # main thread — see _run_process_worker which also self-terminates if
        # its op_id was cancelled during startup.
        # ``_process_lock`` guards the cross-thread publish/check of
        # ``_process`` and ``_cancelled_ops``: the worker assigns
        # ``_process = proc`` and the main thread reads it from
        # ``_cancel_operation``. Without a lock there is a window where the
        # main thread sees ``None`` and never terminates the child.
        self._process_lock = threading.Lock()
        self._op_id: int = 0
        self._cancelled_ops: set[int] = set()
        self._trace_tokens = [
            self.repo_input_var.trace_add("write", lambda *_a: self._persist_settings()),
            self.revision_var.trace_add("write", lambda *_a: self._persist_settings()),
            # ``download_mode_var`` and ``include_patterns_var`` also
            # gate the Download button: snapshot mode and any non-empty
            # include pattern allow downloading without a loaded
            # listing. Refresh runtime state so the button enables
            # immediately when the user switches mode / types a pattern.
            self.download_mode_var.trace_add(
                "write", lambda *_a: (self._persist_settings(), self._refresh_runtime_state())
            ),
            self.include_patterns_var.trace_add(
                "write", lambda *_a: (self._persist_settings(), self._refresh_runtime_state())
            ),
            self.ignore_patterns_var.trace_add("write", lambda *_a: self._persist_settings()),
            self.force_download_var.trace_add("write", lambda *_a: self._persist_settings()),
            self.local_files_only_var.trace_add("write", lambda *_a: self._persist_settings()),
            # Validate-and-clamp on every keystroke so a bad value can't be
            # persisted to settings (and so the next launch doesn't load
            # a malformed string). Then persist the (normalized) value.
            self.max_workers_var.trace_add("write", lambda *_a: self._validate_max_workers()),
            self.max_workers_var.trace_add("write", lambda *_a: self._persist_settings()),
            self.venv_var.trace_add("write", lambda *_a: self._refresh_runtime_state()),
        ]

    def setup_tab(self, parent):
        parent.columnconfigure(0, weight=1)

        main = ttk.Frame(parent)
        main.grid(column=0, row=0, sticky="nsew", padx=10, pady=10)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)

        source = ttk.LabelFrame(main, text="Repo")
        source.grid(column=0, row=0, columnspan=2, sticky="ew", pady=(0, 8))
        source.columnconfigure(1, weight=1)
        ttk.Label(source, text="Repo ID / URL:").grid(column=0, row=0, sticky="w", padx=6, pady=4)
        ttk.Entry(source, textvariable=self.repo_input_var, width=48).grid(column=1, row=0, sticky="ew", padx=6, pady=4)
        self._load_button = ttk.Button(source, text="Load repo", command=self._on_load_repo)
        self._load_button.grid(column=2, row=0, sticky="w", padx=6, pady=4)
        ttk.Label(source, text="Revision:").grid(column=0, row=1, sticky="w", padx=6, pady=4)
        self._revision_combo = ttk.Combobox(source, textvariable=self.revision_var, values=(), width=32)
        self._revision_combo.grid(column=1, row=1, sticky="w", padx=6, pady=4)
        ttk.Label(source, text="Token:").grid(column=0, row=2, sticky="w", padx=6, pady=4)
        ttk.Entry(source, textvariable=self.token_var, width=32, show="*").grid(
            column=1, row=2, sticky="w", padx=6, pady=4
        )
        ttk.Label(
            source,
            text="Accepts plain repo IDs or huggingface.co URLs. Tree/blob URLs seed the revision.",
            font=("TkSmallCaptionFont",),
        ).grid(column=0, row=3, columnspan=3, sticky="w", padx=6, pady=(0, 6))

        venv_frame = ttk.LabelFrame(main, text="Hub Runtime")
        venv_frame.grid(column=0, row=1, sticky="nsew", padx=(0, 6), pady=(0, 8))
        venv_frame.columnconfigure(1, weight=1)
        ttk.Label(venv_frame, text="Active venv:").grid(column=0, row=0, sticky="w", padx=6, pady=4)
        ttk.Label(venv_frame, textvariable=self.venv_status_var).grid(column=1, row=0, sticky="w", padx=6, pady=4)
        self._install_button = ttk.Button(
            venv_frame,
            text="Install / update huggingface_hub",
            command=self._on_install_hf_dependency,
        )
        self._install_button.grid(column=0, row=1, columnspan=2, sticky="w", padx=6, pady=4)

        targets = ttk.LabelFrame(main, text="Download Targets")
        targets.grid(column=1, row=1, sticky="nsew", padx=(6, 0), pady=(0, 8))
        targets.columnconfigure(0, weight=1)
        self._target_container = ttk.Frame(targets)
        self._target_container.grid(column=0, row=0, sticky="ew", padx=6, pady=4)
        self._refresh_targets_button = ttk.Button(
            targets,
            text="Refresh model directories",
            command=self._refresh_target_rows,
        )
        self._refresh_targets_button.grid(column=0, row=1, sticky="w", padx=6, pady=(0, 6))

        files = ttk.LabelFrame(main, text="Files")
        files.grid(column=0, row=2, columnspan=2, sticky="nsew", pady=(0, 8))
        files.columnconfigure(0, weight=1)
        files.rowconfigure(0, weight=1)
        self._files_listbox = tk.Listbox(files, selectmode=tk.EXTENDED, height=14, exportselection=False)
        self._files_listbox.grid(column=0, row=0, sticky="nsew", padx=6, pady=6)
        scroll = ttk.Scrollbar(files, orient="vertical", command=self._files_listbox.yview)
        scroll.grid(column=1, row=0, sticky="ns", pady=6)
        self._files_listbox.config(yscrollcommand=scroll.set)
        file_buttons = ttk.Frame(files)
        file_buttons.grid(column=0, row=1, sticky="w", padx=6, pady=(0, 6))
        ttk.Button(file_buttons, text="Select defaults", command=self._select_default_files).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(file_buttons, text="Select all", command=lambda: self._select_all_files(True)).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(file_buttons, text="Clear selection", command=lambda: self._select_all_files(False)).pack(
            side="left"
        )

        self._options_toggle_btn = ttk.Button(
            main,
            text="▶ Download Options",
            command=self._toggle_options_section,
            style="Toolbutton",
        )
        self._options_toggle_btn.grid(column=0, row=3, columnspan=2, sticky="w", pady=(8, 0))

        options = ttk.LabelFrame(main, text="")
        options.grid(column=0, row=4, columnspan=2, sticky="ew", pady=(4, 0))
        options.columnconfigure(1, weight=1)
        self._options_body = options
        ttk.Label(options, text="Mode:").grid(column=0, row=0, sticky="w", padx=6, pady=4)
        ttk.Radiobutton(options, text="Selected files", value="selected", variable=self.download_mode_var).grid(
            column=1, row=0, sticky="w", padx=6, pady=4
        )
        ttk.Radiobutton(options, text="Repo snapshot", value="snapshot", variable=self.download_mode_var).grid(
            column=1, row=1, sticky="w", padx=6, pady=4
        )
        ttk.Label(options, text="Include patterns:").grid(column=0, row=2, sticky="nw", padx=6, pady=4)
        ttk.Entry(options, textvariable=self.include_patterns_var).grid(column=1, row=2, sticky="ew", padx=6, pady=4)
        ttk.Label(options, text="Ignore patterns:").grid(column=0, row=3, sticky="nw", padx=6, pady=4)
        ttk.Entry(options, textvariable=self.ignore_patterns_var).grid(column=1, row=3, sticky="ew", padx=6, pady=4)
        ttk.Checkbutton(options, text="Force download", variable=self.force_download_var).grid(
            column=0, row=4, sticky="w", padx=6, pady=4
        )
        ttk.Checkbutton(options, text="Local files only", variable=self.local_files_only_var).grid(
            column=1, row=4, sticky="w", padx=6, pady=4
        )
        ttk.Label(options, text="Max workers:").grid(column=0, row=5, sticky="w", padx=6, pady=4)
        ttk.Entry(options, textvariable=self.max_workers_var, width=6).grid(column=1, row=5, sticky="w", padx=6, pady=4)
        options.grid_remove()

        status = ttk.LabelFrame(main, text="Status")
        status.grid(column=0, row=5, columnspan=2, sticky="ew", pady=(8, 0))
        status.columnconfigure(0, weight=1)
        ttk.Label(status, textvariable=self.status_var).grid(column=0, row=0, sticky="w", padx=6, pady=(6, 2))
        self._progress = ttk.Progressbar(status, mode="determinate", maximum=100, value=0)
        self._progress.grid(column=0, row=1, sticky="ew", padx=6, pady=4)
        ttk.Label(status, textvariable=self.progress_label_var, font=("TkSmallCaptionFont",)).grid(
            column=0, row=2, sticky="w", padx=6, pady=(0, 6)
        )
        buttons = ttk.Frame(status)
        buttons.grid(column=0, row=3, sticky="w", padx=6, pady=(0, 6))
        self._download_button = ttk.Button(buttons, text="Download", command=self._on_download)
        self._download_button.pack(side="left", padx=(0, 6))
        self._cancel_button = ttk.Button(buttons, text="Cancel", command=self._cancel_operation, state=tk.DISABLED)
        self._cancel_button.pack(side="left")

        self._refresh_target_rows()
        self._refresh_runtime_state()

    def _persist_settings(self):
        selected_targets = [path for path, var in self._selected_target_vars.items() if bool(var.get())]
        self.launcher.app_settings["hf_repo_input"] = self.repo_input_var.get()
        self.launcher.app_settings["hf_repo_revision"] = self.revision_var.get()
        self.launcher.app_settings["hf_download_mode"] = self.download_mode_var.get()
        self.launcher.app_settings["hf_include_patterns"] = self.include_patterns_var.get()
        self.launcher.app_settings["hf_ignore_patterns"] = self.ignore_patterns_var.get()
        self.launcher.app_settings["hf_force_download"] = bool(self.force_download_var.get())
        self.launcher.app_settings["hf_local_files_only"] = bool(self.local_files_only_var.get())
        self.launcher.app_settings["hf_max_workers"] = self.max_workers_var.get()
        self.launcher.app_settings["hf_target_dirs"] = selected_targets
        save = getattr(self.launcher, "_save_configs", None)
        if callable(save):
            save()

    def _current_active_venv_path(self) -> str:
        return venv_manager.resolve_active_venv_path(
            self.venv_var.get(),
            repo_dir=self.repo_dir,
            platform=sys.platform,
        )

    def _huggingface_dependency(self):
        for dependency in venv_manager.MANAGED_DEPENDENCIES:
            if dependency.key == "huggingface_hub":
                return dependency
        raise LookupError("huggingface_hub dependency not registered")

    def _current_venv_python(self) -> Path | None:
        active = self._current_active_venv_path()
        if not active:
            return None
        return venv_manager.locate_venv_python(active, platform=sys.platform)

    def _refresh_runtime_state(self):
        active = self._current_active_venv_path()
        if self._dep_watch_venv and self._dep_watch_venv != active:
            # The active venv switched out from under an in-flight
            # install-watch (user changed Venv dir in Settings). Cancel
            # the watch AND clear the "Waiting for it to appear in the
            # venv…" status string — otherwise the panel stays stuck on
            # a message about the previous environment.
            self._cancel_dependency_watch()
            self.status_var.set("")
        if not active:
            self.venv_status_var.set("No active venv. Create one in Settings first.")
            self._set_button_state(self._install_button, False)
            self._set_button_state(self._load_button, False)
            self._set_button_state(self._download_button, False)
            return
        python = self._current_venv_python()
        dep = self._huggingface_dependency()
        status = venv_manager.probe_dependency_status(active, dep, platform=sys.platform)
        # Treat runtime as busy while EITHER (a) the cancel-terminate
        # daemon thread is still reaping the subprocess, OR (b) an
        # ``Install / Update huggingface_hub`` dependency-watch is in
        # flight against THIS active venv. The latter polls the venv
        # for the package; clicking Load/Download/Install on top of
        # that watch can race the eventual ``_on_dependency_probe_result``
        # update or trigger a duplicate pip install.
        # ``_cleanup_process_state`` clears ``self._process`` before
        # the kill actually lands, so ``self._process is None`` alone
        # would briefly re-enable buttons before the worker is gone.
        watching = bool(self._dep_watch_venv) and self._dep_watch_venv == active
        # Also block on the worker thread itself. If Cancel wins the
        # startup race before ``_run_process_worker`` publishes
        # ``self._process``, BOTH ``self._process is None`` AND
        # ``_terminating == False`` hold, but the worker thread may
        # still be spawning or tearing down the subprocess. Without
        # this check, Load/Download briefly re-enable mid-shutdown.
        worker_alive = bool(self._worker_thread is not None and self._worker_thread.is_alive())
        idle = self._process is None and not self._terminating and not watching and not worker_alive
        if status.available:
            version = f" (v{status.version})" if status.version else ""
            self.venv_status_var.set(f"{active}{version}")
            self._set_button_state(self._load_button, idle)
            # Download is reachable WITHOUT a loaded file listing under
            # two flows ``_on_download`` now accepts:
            #   * snapshot mode (download everything in the repo);
            #   * pattern-only ``selected`` mode (download whatever
            #     matches ``include_patterns_var`` without picking
            #     individual files).
            # Listbox-selection mode still requires a loaded listing
            # so the user has something to pick from.
            mode = self.download_mode_var.get()
            include_patterns_raw = self.include_patterns_var.get().strip()
            download_ok_without_listing = mode == "snapshot" or bool(include_patterns_raw)
            self._set_button_state(
                self._download_button,
                idle and (bool(self._file_rows) or download_ok_without_listing),
            )
        else:
            detail = status.error or "not installed"
            self.venv_status_var.set(f"{active} [huggingface_hub missing: {detail}]")
            self._set_button_state(self._load_button, False)
            self._set_button_state(self._download_button, False)
        # Disable Install/Update while an HF runner is active — running
        # ``pip install`` into the same venv concurrently with a list or
        # download can corrupt the env (and pip itself complains loudly).
        self._set_button_state(self._install_button, python is not None and idle)
        # ``_refresh_target_rows`` is intentionally NOT called every
        # tick — it destroys and rebuilds the entire target-row
        # widget tree, which (a) churns widgets on every venv probe
        # and (b) re-fires the per-row trace_add callbacks. The
        # initial render happens in ``setup_tab``; subsequent
        # rebuilds are user-triggered (Refresh button) or driven by
        # model-dirs changes via dedicated handlers.

    def _refresh_target_rows(self):
        if self._target_container is None:
            return
        for child in self._target_container.winfo_children():
            child.destroy()
        # Remove the write traces on the old per-row BooleanVars before we
        # drop the references, otherwise each refresh leaks a trace
        # callback whose closure still holds ``self`` — the old vars never
        # get garbage-collected and Tk keeps them alive in the interpreter.
        for old_var, token in self._selected_target_trace_tokens:
            try:
                old_var.trace_remove("write", token)
            except Exception:
                pass
        self._selected_target_trace_tokens = []
        # Defensive normalization: a JSON-edited ``hf_target_dirs``
        # could ship a bare string (``"/models"`` instead of
        # ``["/models"]``) or a scalar (``null`` / a bool / a number).
        # ``tuple(...)`` on a string would iterate character-by-character
        # and ``collect_target_directory_options`` would then try to
        # ``Path("/")`` / ``Path("m")`` etc. Wrap strings as a
        # single-item list, accept list/tuple as-is, and discard
        # anything else.
        raw_target_dirs = self.launcher.app_settings.get("hf_target_dirs", [])
        if isinstance(raw_target_dirs, str):
            normalized_targets = [raw_target_dirs] if raw_target_dirs else []
        elif isinstance(raw_target_dirs, (list, tuple)):
            normalized_targets = [p for p in raw_target_dirs if isinstance(p, str) and p]
        else:
            normalized_targets = []
        selected_paths = tuple(normalized_targets)
        state = collect_target_directory_options(
            list(getattr(self.launcher, "model_dirs", [])),
            selected_paths=selected_paths,
        )
        self._selected_target_vars = {}
        if not state.options:
            ttk.Label(
                self._target_container,
                text="No model directories configured on the Main tab.",
            ).grid(column=0, row=0, sticky="w")
            # Do NOT call ``_persist_settings`` here — it would write
            # ``hf_target_dirs=[]`` to the config, erasing the user's
            # prior selections any time this UI is rendered with no
            # currently-configured model dirs (e.g. the first paint
            # before Main tab loads). The user's stored target list
            # should only change when they explicitly toggle a row.
            return
        for row, option in enumerate(state.options):
            var = tk.BooleanVar(value=option.selected)
            token = var.trace_add("write", lambda *_a: self._persist_settings())
            self._selected_target_trace_tokens.append((var, token))
            self._selected_target_vars[str(option.path)] = var
            ttk.Checkbutton(
                self._target_container,
                text=option.label,
                variable=var,
            ).grid(column=0, row=row, sticky="w", pady=2)
        self._persist_settings()

    @staticmethod
    def _set_button_state(button, enabled: bool):
        if button is None:
            return
        button.config(state=(tk.NORMAL if enabled else tk.DISABLED))

    @staticmethod
    def _format_bytes(n) -> str:
        try:
            size = float(int(n))
        except (TypeError, ValueError):
            return str(n)
        if size < 0:
            size = 0.0
        units = ("B", "KB", "MB", "GB", "TB", "PB")
        idx = 0
        while size >= 1024 and idx < len(units) - 1:
            size /= 1024
            idx += 1
        if idx == 0:
            return f"{int(size)} B"
        return f"{size:.2f} {units[idx]}"

    # Bound max workers to a sensible range; 32 is generous (HF's default
    # is 8 and few networks benefit from more parallelism than that).
    _MAX_WORKERS_FLOOR = 1
    _MAX_WORKERS_CEILING = 32

    def _validate_max_workers(self) -> None:
        """Normalize ``max_workers_var`` to an int in the allowed range.

        Empty input is left blank so the user can clear and retype.
        Non-numeric input is silently corrected to the floor value; the
        clamp also runs at the Download click as a final safety net.
        """
        if self._max_workers_validating:
            return
        raw = self.max_workers_var.get().strip()
        if not raw:
            return  # Allow empty while editing.
        try:
            value = int(raw)
        except ValueError:
            value = self._MAX_WORKERS_FLOOR
        clamped = max(self._MAX_WORKERS_FLOOR, min(self._MAX_WORKERS_CEILING, value))
        normalized = str(clamped)
        if normalized == raw:
            return
        # Set under the re-entry flag so this trace handler doesn't recurse.
        self._max_workers_validating = True
        try:
            self.max_workers_var.set(normalized)
        finally:
            self._max_workers_validating = False

    @staticmethod
    def _looks_like_tqdm_progress(line: str) -> bool:
        # Match a real tqdm progress bar, not arbitrary error messages
        # that happen to mention bytes-per-second. tqdm always emits a
        # ``NN%|`` percent prefix or a bar character (``█``) inside the
        # rendered bar — both are rare in genuine stderr exceptions.
        if "%|" in line:
            return True
        if "█" in line and "|" in line:
            return True
        # Final fallback: throughput suffix only if the line also carries
        # a tqdm-style ETA bracket like ``[00:12<00:34, ...]``.
        if ("B/s" in line or "it/s" in line) and "[" in line and "<" in line:
            return True
        return False

    def _toggle_options_section(self):
        if self._options_body is None or self._options_toggle_btn is None:
            return
        self._options_expanded = not self._options_expanded
        if self._options_expanded:
            self._options_body.grid()
            self._options_toggle_btn.config(text="▼ Download Options")
        else:
            self._options_body.grid_remove()
            self._options_toggle_btn.config(text="▶ Download Options")

    def _on_install_hf_dependency(self):
        active = self._current_active_venv_path()
        if not active:
            messagebox.showerror("No venv", "Create or select a usable venv in Settings first.")
            return
        # Build the command AND launch in the same try/except.
        # ``build_install_dependency_command`` performs venv path /
        # platform validation that can raise ``ValueError`` (and
        # ``_shell_join`` raises on Windows ``%``/``!`` injection),
        # which would otherwise unwind the Tk callback and leave the
        # tab in a broken state. Mirror the same recoverable-error
        # pattern Settings tab uses around venv command builders.
        try:
            command = venv_manager.build_install_dependency_command(
                active,
                self._huggingface_dependency(),
                platform=sys.platform,
            )
            terminal_launcher.open_command_in_terminal(command, cwd=self.repo_dir)
        except Exception as exc:
            messagebox.showerror(
                "Install / update huggingface_hub",
                f"Failed to start huggingface_hub install:\n{exc}",
            )
            self.status_var.set("Failed to start huggingface_hub install.")
            return
        self.status_var.set(
            "Opened terminal to install or update huggingface_hub. Waiting for it to appear in the venv…"
        )
        self._start_dependency_watch(active)

    def _start_dependency_watch(self, venv_path: str):
        self._cancel_dependency_watch()
        self._dep_watch_venv = venv_path
        self._dep_watch_deadline = time.monotonic() + self.DEP_WATCH_TIMEOUT_S
        try:
            self._dep_watch_after_id = self.root.after(self.DEP_WATCH_INTERVAL_MS, self._fire_dependency_probe)
        except tk.TclError:
            # Tk root already destroyed (e.g. tab teardown during install).
            self._dep_watch_after_id = None
            self._dep_watch_venv = None
        # Refresh button state now that ``_dep_watch_venv`` is set, so
        # Load / Download / Install grey out immediately for the duration
        # of the watch (otherwise they only flip on the next external
        # state change).
        try:
            self._refresh_runtime_state()
        except (tk.TclError, RuntimeError):
            pass

    def _cancel_dependency_watch(self):
        if self._dep_watch_after_id is not None:
            try:
                self.root.after_cancel(self._dep_watch_after_id)
            except Exception:
                pass
            self._dep_watch_after_id = None
        self._dep_watch_venv = None

    def _fire_dependency_probe(self):
        self._dep_watch_after_id = None
        venv_path = self._dep_watch_venv
        if not venv_path:
            return
        if venv_path != self._current_active_venv_path():
            self._dep_watch_venv = None
            return
        if time.monotonic() > self._dep_watch_deadline:
            # Deadline crossed between poll ticks — close the watch with a
            # visible status so the panel doesn't stay stuck on the earlier
            # "Waiting for it to appear…" message. Mirrors the timeout
            # branch in ``_on_dependency_probe_result``.
            self.status_var.set(
                "Timed out waiting for huggingface_hub to appear. " "Click 'Install / update huggingface_hub' to retry."
            )
            self._dep_watch_venv = None
            self._refresh_runtime_state()
            return
        dep = self._huggingface_dependency()

        def worker():
            try:
                status = venv_manager.probe_dependency_status(venv_path, dep, platform=sys.platform)
                available = bool(status.available)
            except Exception:
                available = False
            try:
                self.root.after(0, self._on_dependency_probe_result, venv_path, available)
            except (tk.TclError, RuntimeError):
                # Tk root destroyed while probe was in flight — drop the
                # result silently; the watch is over.
                pass

        threading.Thread(target=worker, daemon=True).start()

    def _on_dependency_probe_result(self, venv_path: str, available: bool):
        if venv_path != self._dep_watch_venv:
            return
        if venv_path != self._current_active_venv_path():
            self._dep_watch_venv = None
            return
        if available:
            # Close out the install-watch with a visible status. Otherwise
            # the status panel stays stuck on the earlier "Waiting for it
            # to appear in the venv…" message even though buttons have
            # already re-enabled — confusing.
            self.status_var.set("huggingface_hub is now available in the active venv.")
            self._dep_watch_venv = None
            self._refresh_runtime_state()
            return
        if time.monotonic() > self._dep_watch_deadline:
            # Same visibility issue on the timeout branch — the watch
            # used to exit silently, leaving the user with no signal that
            # the install never completed.
            self.status_var.set(
                "Timed out waiting for huggingface_hub to appear. " "Click 'Install / update huggingface_hub' to retry."
            )
            self._dep_watch_venv = None
            self._refresh_runtime_state()
            return
        try:
            self._dep_watch_after_id = self.root.after(self.DEP_WATCH_INTERVAL_MS, self._fire_dependency_probe)
        except tk.TclError:
            self._dep_watch_after_id = None
            self._dep_watch_venv = None

    def _on_load_repo(self):
        try:
            parsed = normalize_repo_input(self.repo_input_var.get())
        except ValueError as exc:
            messagebox.showerror("Invalid repo", str(exc))
            return
        # Clear any previously loaded listing so that a *failing* new load
        # can't leave Download enabled against stale filenames from a
        # different repo. _refresh_runtime_state() (fired after the runner
        # exits) will recompute button state from the empty rows.
        self._file_rows = []
        self._file_path_by_index = []
        self._refs = []
        # Drop the loaded-listing identity too — until the new ``list``
        # runner posts a ``listing`` event, ``_on_download`` must not
        # treat the previous repo's identity as still active. Reset to
        # ``None`` sentinel so the "never loaded" branches fire.
        self._loaded_repo_id = ""
        self._loaded_revision = None
        self._pinned_revision_sha = ""
        if self._revision_combo is not None:
            self._revision_combo.config(values=())
        if self._files_listbox is not None:
            self._files_listbox.delete(0, tk.END)
        self._set_button_state(self._download_button, False)
        # Apply URL-derived revision hints UNCONDITIONALLY. The old guard
        # only set the hint when revision_var was blank/whitespace, so a
        # stale persisted or manually-typed revision could shadow a
        # ``tree/<branch>`` hint pulled from the freshly pasted URL — the
        # user's most recent intent must win.
        if parsed.revision_hint:
            self.revision_var.set(parsed.revision_hint)
        # Token deliberately NOT in the payload — see ``_start_runner``
        # for the env-var handoff. The temp JSON file lives on disk for
        # the subprocess's lifetime; a credential there would be visible
        # to anyone with read access to ``/tmp`` (and to forensic disk
        # reads after the process exits).
        requested_revision = self.revision_var.get().strip()
        # Captured so the ``listing`` handler can tell whether the user
        # explicitly typed a revision (in which case it must NOT silently
        # rewrite the field to ``_refs[0]`` if the user clears the box
        # between submit and response) or really left it blank for the
        # runner to resolve.
        self._last_requested_revision = requested_revision
        payload = {
            "repo_id": parsed.repo_id,
            "revision": requested_revision,
        }
        self.status_var.set(f"Loading {parsed.repo_id}…")
        self._start_runner("list", payload)

    def _on_download(self):
        selected_targets = [path for path, var in self._selected_target_vars.items() if bool(var.get())]
        if not selected_targets:
            messagebox.showerror("No target directory", "Select at least one model directory target.")
            return
        selected_files = self._selected_file_paths()
        # Compute include_patterns early so the "selected" mode guard can
        # consider EITHER source: include_patterns alone (a power-user
        # ``*.gguf`` filter on a tree-only listing) used to be rejected
        # for having no listbox selection, which forced the user to either
        # tick rows that don't exist or switch to snapshot mode.
        include_patterns = list(parse_pattern_lines(self.include_patterns_var.get()))
        download_mode = self.download_mode_var.get()
        # Snapshot mode AND pattern-only "selected" mode can both
        # legitimately start without a loaded listing — the runner
        # downloads ``allow_patterns``/everything directly against the
        # current repo input. Only require a loaded listing when the
        # user picked specific files from the listbox (``selected``
        # mode + no include patterns).
        if download_mode == "selected" and not selected_files and not include_patterns:
            if not self._file_rows:
                messagebox.showinfo(
                    "No repo loaded",
                    "Load a repo first, or switch to snapshot mode / enter "
                    "an include pattern to download without a listing.",
                )
                return
            messagebox.showerror(
                "No files selected",
                "Select at least one file, enter an include pattern, or " "switch to repo snapshot mode.",
            )
            return
        try:
            # Clamp to BOTH ends of the configured range. The keystroke
            # validator (``_validate_max_workers``) already enforces this
            # for new input, but a value persisted before that validator
            # existed could be arbitrarily high — bypass that and we'd
            # spawn ``parsed_workers`` parallel downloads with no upper
            # bound.
            parsed_workers = int(self.max_workers_var.get().strip() or "4")
            max_workers = max(
                self._MAX_WORKERS_FLOOR,
                min(self._MAX_WORKERS_CEILING, parsed_workers),
            )
        except ValueError:
            messagebox.showerror("Invalid workers", "Max workers must be a positive integer.")
            return
        try:
            parsed = normalize_repo_input(self.repo_input_var.get())
        except ValueError as exc:
            messagebox.showerror("Invalid repo", str(exc))
            return
        current_revision = self.revision_var.get().strip()
        # Drift checks ONLY apply when the user is relying on the
        # loaded file listing — i.e. they ticked specific files in the
        # listbox. Snapshot mode and pattern-only ``selected`` mode
        # don't depend on the loaded file rows, so a different
        # repo/revision in the input box doesn't make the download
        # "wrong" — it just means the user wants to fetch from a
        # different source than the one they last browsed. Skip the
        # mismatch dialogs and the loaded-identity binding in those
        # flows; fall back to the current input.
        # Snapshot mode auto-selects default files via
        # ``_select_default_files``, so ``selected_files`` can be
        # truthy even when the user explicitly chose "snapshot" (=
        # download the whole repo). Without the explicit
        # ``download_mode == "selected"`` guard, a snapshot download
        # would still be treated as listing-bound and the drift
        # checks below could block it or bind it to a stale
        # ``_loaded_repo_id`` / ``_loaded_revision``.
        using_loaded_listing = download_mode == "selected" and bool(selected_files) and bool(self._file_rows)
        if using_loaded_listing:
            # Refuse the download if the input has drifted since the
            # file list was loaded. The selection in
            # ``self._file_rows`` is only meaningful for the
            # repo/revision that produced the listing; silently
            # fetching against the changed input would download the
            # wrong files (or fail with a confusing 404).
            if self._loaded_repo_id and parsed.repo_id != self._loaded_repo_id:
                messagebox.showerror(
                    "Repo changed since load",
                    f"The file list was loaded for {self._loaded_repo_id!r}, but "
                    f"the input now reads {parsed.repo_id!r}. Click 'Load repo' "
                    f"again to refresh the listing, or restore the original URL.",
                )
                return
            # ``is not None`` (not truthy): an explicit empty
            # ``_loaded_revision`` means "listing came back for the repo's
            # default branch" and the user later typing a specific ref
            # must trigger the mismatch dialog. The previous truthy
            # check let the new text silently win.
            if self._loaded_revision is not None and current_revision != self._loaded_revision:
                messagebox.showerror(
                    "Revision changed since load",
                    f"The file list was loaded for revision {self._loaded_revision!r}, "
                    f"but the field now reads {current_revision!r}. Click 'Load repo' "
                    f"again so the file selection matches the revision you'll download.",
                )
                return
        # Use the LOADED repo/revision only when the user is actually
        # relying on the listing's file selection. Snapshot and
        # pattern-only flows use the current input directly.
        if not using_loaded_listing:
            effective_repo_id = parsed.repo_id
        else:
            effective_repo_id = self._loaded_repo_id or parsed.repo_id
        # Prefer the pinned SHA so the download binds to the EXACT
        # commit that produced the file list. ``_loaded_revision``
        # (the ref name) is the user-facing fallback for older
        # listing payloads that didn't include a SHA. The
        # current-input fallback fires when:
        # * no listing has been accepted yet (patterns-only flow), OR
        # * the download isn't using the loaded listing (snapshot /
        #   pattern-only) — see ``using_loaded_listing`` above.
        if not using_loaded_listing:
            effective_revision = current_revision
        elif self._pinned_revision_sha:
            effective_revision = self._pinned_revision_sha
        elif self._loaded_revision is not None:
            effective_revision = self._loaded_revision
        else:
            effective_revision = current_revision
        payload = {
            "repo_id": effective_repo_id,
            "revision": effective_revision,
            # Token NOT in payload — handed to the subprocess via env
            # (see ``_start_runner``) so it never lands on disk.
            "download_mode": self.download_mode_var.get(),
            "selected_files": selected_files,
            "include_patterns": include_patterns,
            "ignore_patterns": list(parse_pattern_lines(self.ignore_patterns_var.get())),
            "force_download": bool(self.force_download_var.get()),
            "local_files_only": bool(self.local_files_only_var.get()),
            "max_workers": max_workers,
            "target_dirs": selected_targets,
        }
        self.status_var.set(f"Downloading {effective_repo_id}…")
        self._start_runner("download", payload)

    def _selected_file_paths(self) -> list[str]:
        if self._files_listbox is None:
            return []
        return [
            self._file_path_by_index[index]
            for index in self._files_listbox.curselection()
            if 0 <= index < len(self._file_path_by_index)
        ]

    def _select_all_files(self, selected: bool):
        if self._files_listbox is None:
            return
        self._files_listbox.selection_clear(0, tk.END)
        if selected and self._file_path_by_index:
            self._files_listbox.selection_set(0, tk.END)

    def _select_default_files(self):
        if self._files_listbox is None:
            return
        self._files_listbox.selection_clear(0, tk.END)
        for index, row in enumerate(self._file_rows):
            if row.selected_by_default:
                self._files_listbox.selection_set(index)

    def _start_runner(self, action: str, payload: dict):
        python = self._current_venv_python()
        if python is None:
            messagebox.showerror("No venv", "A usable venv is required for Hugging Face operations.")
            return
        # _cancel_operation bumps self._op_id; the value after the call is the
        # generation for *this* new run.
        self._cancel_operation(clean_only=True)
        op_id = self._op_id
        # Tempdir-out-of-space, EACCES on /tmp, or any other OS-level
        # failure used to bubble straight through the Tk callback, which
        # left the tab disabled (Cancel + Load + Download all flipped off
        # below) with no recovery. Catch + surface a recoverable error.
        payload_file: Path | None = None
        try:
            fd, payload_path = tempfile.mkstemp(prefix="hf-downloader-", suffix=".json")
            os.close(fd)
            payload_file = Path(payload_path)
            payload_file.write_bytes(payload_bytes(payload))
        except OSError as exc:
            if payload_file is not None:
                try:
                    payload_file.unlink()
                except OSError:
                    pass
            messagebox.showerror(
                "Hugging Face operation",
                f"Failed to prepare runner payload file:\n{exc}",
            )
            return
        self._payload_path = payload_file
        self._operation_name = action
        self._progress_stop()
        self._progress.config(mode="indeterminate", value=0)
        self._progress.start(12)
        self.progress_label_var.set("")
        self._set_button_state(self._cancel_button, True)
        self._set_button_state(self._load_button, False)
        self._set_button_state(self._download_button, False)
        # Disable Install/Update synchronously here too. The
        # ``_refresh_runtime_state`` re-gate only fires after the worker
        # publishes ``self._process``, leaving a window where the user
        # could click Install during ``list``/``download`` and run
        # ``pip install`` into the same venv mid-operation.
        self._set_button_state(self._install_button, False)
        command = build_runner_command(python, action, payload_file)
        # Capture the HF token here (Tk-thread only) and hand it to the
        # worker as an arg. The worker injects it into the child's env
        # so the credential never lands in the on-disk payload JSON.
        try:
            hf_token = (self.token_var.get() or "").strip()
        except Exception:
            hf_token = ""
        self._worker_thread = threading.Thread(
            target=self._run_process_worker,
            args=(command, op_id, hf_token),
            daemon=True,
        )
        self._worker_thread.start()
        if self._queue_after_id is None:
            self._queue_after_id = self.root.after(self.POLL_MS, self._poll_queue)

    def _run_process_worker(self, command: list[str], op_id: int, hf_token: str = ""):
        print(f"[hf-runner] launching: {' '.join(command)}", file=sys.stderr, flush=True)
        # ``CREATE_NO_WINDOW`` on Windows suppresses the brief flashing
        # console for the runner subprocess. ``encoding`` + ``errors`` guard
        # against a non-UTF-8 system locale crashing the line iterator
        # midway through a download. ``stdin=DEVNULL`` makes sure the child
        # cannot ever steal the parent terminal (git credential prompts,
        # interactive HF auth prompts, etc.).
        # Build a child env that inherits the parent's, then injects
        # ``HF_TOKEN`` if the user supplied one. Done here (not in the
        # payload JSON) so the token never lands on disk — see
        # ``_token_value`` in runner.py for the resolution order.
        child_env = os.environ.copy()
        if hf_token:
            child_env["HF_TOKEN"] = hf_token
            child_env.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)
        popen_kwargs: dict = {
            "cwd": self.repo_dir,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "bufsize": 1,
            "encoding": "utf-8",
            "errors": "replace",
            "env": child_env,
        }
        if sys.platform.startswith("win"):
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            proc = subprocess.Popen(command, **popen_kwargs)
        except Exception as exc:
            print(f"[hf-runner] failed to launch: {exc}", file=sys.stderr, flush=True)
            self._queue.put(
                {
                    "event": "process-exit",
                    "op_id": op_id,
                    "returncode": 1,
                    "stderr": str(exc),
                }
            )
            return
        # Publish the handle under the lock so _cancel_operation, which
        # reads it on the Tk thread, can never observe a torn state.
        # Cancellation that beat the ``self._process = proc`` assignment must
        # NOT leave a stale handle pinned: ``_finalize_process`` only fires on
        # events that ``_poll_queue`` accepts, and cancelled-op events are
        # dropped — so without this guard ``_poll_queue`` would keep
        # rescheduling forever against a process the user already cancelled.
        with self._process_lock:
            cancelled_at_start = op_id in self._cancelled_ops
            if not cancelled_at_start:
                self._process = proc
        # If Cancel was clicked while Popen was still launching, the main
        # thread had no handle to terminate. Detect that here and run the
        # same terminate → wait → kill escalation as ``_cancel_operation``;
        # a single ``terminate()`` would leave the subprocess alive if it
        # ignores SIGTERM, and ``_poll_queue`` would then reschedule
        # forever because ``_worker_thread.is_alive()`` never flips false.
        if cancelled_at_start:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=2)
                except Exception:
                    pass
            except Exception:
                pass

        stderr_chunks: list[str] = []

        def stderr_reader():
            assert proc.stderr is not None
            for raw in proc.stderr:
                stripped = raw.rstrip()
                if not stripped:
                    continue
                stderr_chunks.append(stripped)
                print(f"[hf-runner stderr] {stripped}", file=sys.stderr, flush=True)
                self._queue.put({"event": "stderr", "op_id": op_id, "message": stripped})

        stderr_thread = threading.Thread(target=stderr_reader, daemon=True)
        stderr_thread.start()

        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                print(f"[hf-runner stdout] {line}", file=sys.stderr, flush=True)
                self._queue.put({"event": "stderr", "op_id": op_id, "message": line})
                continue
            event["op_id"] = op_id
            self._queue.put(event)
        returncode = proc.wait()
        stderr_thread.join(timeout=2.0)
        stderr_text = "\n".join(stderr_chunks).strip()
        print(f"[hf-runner] exited with code {returncode}", file=sys.stderr, flush=True)
        self._queue.put(
            {
                "event": "process-exit",
                "op_id": op_id,
                "returncode": returncode,
                "stderr": stderr_text,
            }
        )

    def _poll_queue(self):
        self._queue_after_id = None
        drained = False
        while True:
            try:
                event = self._queue.get_nowait()
            except queue.Empty:
                break
            drained = True
            self._handle_event(event)
        # Keep polling while either the process is still tracked OR the
        # worker thread is still alive (it may still emit one final
        # ``process-exit`` after the main thread already cleared
        # ``self._process``) OR the queue isn't empty. Without the
        # worker-alive check, a late ``process-exit`` event after Cancel
        # was orphaned and the runtime state never got refreshed.
        worker_alive = bool(self._worker_thread is not None and self._worker_thread.is_alive())
        if self._process is not None or worker_alive or not self._queue.empty():
            try:
                self._queue_after_id = self.root.after(self.POLL_MS, self._poll_queue)
            except tk.TclError:
                # Tk root destroyed mid-poll (e.g. app teardown). Drop the
                # next tick rather than letting the exception escape.
                self._queue_after_id = None
        elif drained:
            self._refresh_runtime_state()

    def _handle_event(self, event: dict):
        op_id = event.get("op_id")
        if op_id is not None and op_id != self._op_id:
            # Stale: this event came from a worker we already cancelled or
            # replaced. Dropping it keeps the UI consistent with what the
            # user actually sees.
            return
        kind = event.get("event")
        if kind == "status":
            self.status_var.set(event.get("message", ""))
            return
        if kind == "listing":
            refs, files = summarize_repo_listing(event.get("refs", []), event.get("files", []))
            self._refs = [ref.name for ref in refs]
            if self._revision_combo is not None:
                self._revision_combo.config(values=self._refs)
            # Auto-restore ONLY when the user submitted with a non-empty
            # revision and then cleared the box mid-flight. Restore the
            # exact value they submitted (``_last_requested_revision``)
            # rather than ``_refs[0]`` — ``_refs`` is alphabetically
            # sorted, so its first entry can be a completely different
            # branch/tag than the one the runner actually listed, which
            # would silently retarget the next download at the wrong ref.
            requested_revision = getattr(self, "_last_requested_revision", "").strip()
            # Restore even when ``self._refs`` is empty: the listing
            # runner can return zero refs for a repo with no branches/
            # tags reported (or a payload that just didn't include
            # them) yet still resolve files against the submitted
            # revision. Tying restore to ``self._refs`` here would
            # silently leave the field blank in that case and the
            # next download would re-resolve against the default
            # branch instead of the user's submitted ref.
            if not self.revision_var.get().strip() and requested_revision:
                self.revision_var.set(requested_revision)
            self._file_rows = list(files)
            self._file_path_by_index = [row.path for row in self._file_rows]
            # Pin the repo / revision that produced THIS file list so
            # ``_on_download`` can refuse to launch against a mismatched
            # current input.
            self._loaded_repo_id = str(event.get("repo_id", "") or "")
            # ``event.get("revision")`` is the runner's echo of the
            # requested ref (a moving branch/tag name). Use it for the
            # mismatch-detection in ``_on_download`` — that's what the
            # user typed and what they expect to see. Critically: DO
            # NOT read the live ``revision_var`` here — the user can
            # edit it mid-flight and we'd silently bind to their
            # unrelated new typing. An explicit ``""`` is the correct
            # value when the request itself was blank; the
            # ``None``-vs-``""`` distinction at the read sites tells
            # "never loaded" apart from "loaded with blank".
            runner_resolved = str(event.get("revision") or "").strip()
            self._loaded_revision = runner_resolved or requested_revision
            # ``event.get("resolved_revision")`` is the runner's
            # ACTUAL commit SHA. Pinning the download against the SHA
            # rather than the ref name means a force-push or new
            # commit to the branch between the listing and the
            # download fetches the same bytes the user saw in the
            # file list. Empty string = older runner / repo without
            # SHA; the download falls back to the ref name.
            self._pinned_revision_sha = str(event.get("resolved_revision") or "").strip()
            if self._files_listbox is not None:
                self._files_listbox.delete(0, tk.END)
                for row in self._file_rows:
                    self._files_listbox.insert(tk.END, row.display_name)
                self._select_default_files()
            self.status_var.set(f"Loaded {len(files)} files from {event.get('repo_id', '')}.")
            # Don't enable Download here — the list runner subprocess is
            # still alive. Re-enabling now would let the user click
            # Download against an in-flight worker; ``_start_runner``
            # would then race with the still-cleaning-up list process.
            # The button comes back on via ``_refresh_runtime_state`` once
            # ``process-exit`` fires (which clears ``self._process``).
            return
        if kind == "target-start":
            self.status_var.set(event.get("message", "Downloading…"))
            return
        if kind == "progress":
            current = event.get("current")
            total = event.get("total")
            desc = (event.get("description") or "").strip()
            if total:
                cur = int(current or 0)
                tot = int(total)
                self._progress_stop()
                self._progress.config(mode="determinate", maximum=max(tot, 1))
                self._progress["value"] = cur
                pct = (cur / tot * 100) if tot else 0
                label = f"{self._format_bytes(cur)} / {self._format_bytes(tot)} ({pct:.1f}%)"
                if desc:
                    label = f"{label} · {desc}"
                self.progress_label_var.set(label)
            elif desc:
                self.progress_label_var.set(desc)
            return
        if kind == "target-complete":
            self.progress_label_var.set(
                f"Completed target {event.get('index')}/{event.get('total_targets')}: {event.get('target')}"
            )
            self._progress.config(mode="determinate", maximum=max(int(event.get("total_targets") or 1), 1))
            self._progress["value"] = int(event.get("index") or 0)
            return
        if kind == "complete":
            self.status_var.set(event.get("message", "Completed."))
            return
        if kind == "stderr":
            message = event.get("message", "")
            if self._looks_like_tqdm_progress(message):
                return
            self.progress_label_var.set(message)
            return
        if kind == "error":
            self.status_var.set(event.get("message", "Operation failed."))
            return
        if kind == "warn":
            # Non-fatal runner-side warnings (probe-cleanup failures,
            # leaked temp files, etc). Surface to the status bar so
            # log scrapers / users notice without aborting the
            # operation. The runner keeps going.
            self.status_var.set(event.get("message", "Warning from runner."))
            return
        if kind == "process-exit":
            self._finalize_process(event)

    def _finalize_process(self, event: dict):
        returncode = int(event.get("returncode") or 0)
        stderr = event.get("stderr", "")
        self._progress_stop()
        # Remember any final progress string so a successful run doesn't
        # blank out the "Completed target N/N" message in
        # _cleanup_process_state.
        final_progress_label = self.progress_label_var.get() if returncode == 0 else ""
        op_name = self._operation_name
        if returncode != 0:
            detail = stderr or self.status_var.get() or f"exit code {returncode}"
            self.status_var.set(f"{op_name} failed: {detail}")
        else:
            # Don't overwrite a more-specific terminal message that an
            # earlier ``complete`` event already wrote (e.g. "Finished
            # downloading TheBloke/...").
            current = self.status_var.get() or ""
            if not current or current.lower().startswith(("downloading", "loading", "operation cancelled")):
                if op_name == "download":
                    self.status_var.set("Download finished.")
                elif op_name == "list":
                    self.status_var.set("Repo loaded.")
                else:
                    self.status_var.set("Operation finished.")
        self._cleanup_process_state()
        if final_progress_label:
            self.progress_label_var.set(final_progress_label)
        self._refresh_runtime_state()

    # Cap to keep _cancelled_ops from growing unbounded across the app
    # lifetime. Anything older than this many cancelled ops can no longer
    # produce queue events because the worker would have exited long ago.
    _CANCELLED_OPS_CAP = 64

    def _cancel_operation(self, *, clean_only: bool = False):
        # Invalidate the current op so any in-flight events from this worker
        # are dropped by _handle_event before they can touch the UI.
        with self._process_lock:
            cancelled_op = self._op_id
            self._cancelled_ops.add(cancelled_op)
            # Bound the set: drop the lowest op_ids first.
            if len(self._cancelled_ops) > self._CANCELLED_OPS_CAP:
                excess = len(self._cancelled_ops) - self._CANCELLED_OPS_CAP
                for stale in sorted(self._cancelled_ops)[:excess]:
                    self._cancelled_ops.discard(stale)
            self._op_id += 1
            proc = self._process
        if proc is not None and proc.poll() is None:
            # Don't block the Tk main thread on the terminate-wait-kill
            # escalation: a stubborn child could freeze the UI for up to
            # 7 s (5 s terminate + 2 s kill wait). Run the escalation on
            # a daemon thread and let the Tk event loop keep ticking;
            # ``_run_process_worker`` will see ``proc.returncode`` and
            # post the ``process-exit`` event for ``_poll_queue`` to
            # consume.
            # Mark termination-in-progress so ``_refresh_runtime_state``
            # keeps the action buttons disabled until the daemon thread
            # actually finishes. Otherwise the user could click Load /
            # Download in the gap between ``_process = None`` (set in
            # ``_cleanup_process_state`` below) and the actual SIGKILL.
            self._terminating = True

            def _async_terminate(_proc=proc):
                try:
                    _proc.terminate()
                except Exception:
                    pass
                try:
                    _proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        _proc.kill()
                    except Exception:
                        pass
                    try:
                        _proc.wait(timeout=2)
                    except Exception:
                        pass
                except Exception:
                    pass
                # Re-arm the button-gating once the worker is actually
                # gone. ``root.after(0, ...)`` hops back to the Tk
                # thread so ``_refresh_runtime_state`` can safely poke
                # widgets.
                try:
                    self.root.after(0, self._on_terminate_done)
                except (tk.TclError, RuntimeError):
                    # Tk torn down before the daemon finished — drop.
                    self._terminating = False

            threading.Thread(target=_async_terminate, daemon=True).start()
        if not clean_only:
            self.status_var.set("Operation cancelled.")
        self._cleanup_process_state()
        if not clean_only:
            self._refresh_runtime_state()

    def _on_terminate_done(self):
        """Called on the Tk thread once ``_async_terminate`` has actually
        reaped the subprocess. Drops the ``_terminating`` flag and
        re-runs ``_refresh_runtime_state`` so the Load / Download /
        Install buttons come back at the right moment.
        """
        self._terminating = False
        try:
            self._refresh_runtime_state()
        except (tk.TclError, RuntimeError):
            pass

    def _cleanup_process_state(self):
        with self._process_lock:
            self._process = None
        self._operation_name = ""
        self._progress_stop()
        self._set_button_state(self._cancel_button, False)
        self.progress_label_var.set("")
        if self._payload_path is not None:
            try:
                self._payload_path.unlink()
            except OSError:
                pass
            self._payload_path = None

    def _progress_stop(self):
        if self._progress is None:
            return
        try:
            self._progress.stop()
        except Exception:
            pass


def create_hf_downloader_tab(launcher):
    return HuggingFaceDownloaderTab(launcher)
