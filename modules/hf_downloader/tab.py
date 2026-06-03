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
        self.repo_dir = getattr(launcher, "repo_dir", venv_manager.launcher_repo_dir())
        settings = launcher.app_settings
        self.repo_input_var = tk.StringVar(value=settings.get("hf_repo_input", ""))
        self.revision_var = tk.StringVar(value=settings.get("hf_repo_revision", ""))
        self.download_mode_var = tk.StringVar(value=settings.get("hf_download_mode", "selected"))
        self.include_patterns_var = tk.StringVar(value=settings.get("hf_include_patterns", ""))
        self.ignore_patterns_var = tk.StringVar(value=settings.get("hf_ignore_patterns", ""))
        self.force_download_var = tk.BooleanVar(value=bool(settings.get("hf_force_download", False)))
        self.local_files_only_var = tk.BooleanVar(value=bool(settings.get("hf_local_files_only", False)))
        self.max_workers_var = tk.StringVar(value=str(settings.get("hf_max_workers", 4)))
        self.token_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Create/select a venv, install huggingface_hub, then load a repo.")
        self.venv_var = launcher.venv_dir
        self.venv_status_var = tk.StringVar(value="")
        self.progress_label_var = tk.StringVar(value="")
        self._selected_target_vars: dict[str, tk.BooleanVar] = {}
        self._file_rows: list[HfRepoFile] = []
        self._file_path_by_index: list[str] = []
        self._refs: list[str] = []
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
        self._payload_path: Path | None = None
        self._operation_name = ""
        self._dep_watch_after_id: str | None = None
        self._dep_watch_deadline: float = 0.0
        self._dep_watch_venv: str | None = None
        self._trace_tokens = [
            self.repo_input_var.trace_add("write", lambda *_a: self._persist_settings()),
            self.revision_var.trace_add("write", lambda *_a: self._persist_settings()),
            self.download_mode_var.trace_add("write", lambda *_a: self._persist_settings()),
            self.include_patterns_var.trace_add("write", lambda *_a: self._persist_settings()),
            self.ignore_patterns_var.trace_add("write", lambda *_a: self._persist_settings()),
            self.force_download_var.trace_add("write", lambda *_a: self._persist_settings()),
            self.local_files_only_var.trace_add("write", lambda *_a: self._persist_settings()),
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
        ttk.Entry(source, textvariable=self.repo_input_var, width=48).grid(
            column=1, row=0, sticky="ew", padx=6, pady=4
        )
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
        ttk.Label(venv_frame, textvariable=self.venv_status_var).grid(
            column=1, row=0, sticky="w", padx=6, pady=4
        )
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
        ttk.Button(file_buttons, text="Select defaults", command=self._select_default_files).pack(side="left", padx=(0, 6))
        ttk.Button(file_buttons, text="Select all", command=lambda: self._select_all_files(True)).pack(side="left", padx=(0, 6))
        ttk.Button(file_buttons, text="Clear selection", command=lambda: self._select_all_files(False)).pack(side="left")

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
        selected_targets = [
            path for path, var in self._selected_target_vars.items()
            if bool(var.get())
        ]
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
            self._cancel_dependency_watch()
        if not active:
            self.venv_status_var.set("No active venv. Create one in Settings first.")
            self._set_button_state(self._install_button, False)
            self._set_button_state(self._load_button, False)
            self._set_button_state(self._download_button, False)
            return
        python = self._current_venv_python()
        dep = self._huggingface_dependency()
        status = venv_manager.probe_dependency_status(active, dep, platform=sys.platform)
        if status.available:
            version = f" (v{status.version})" if status.version else ""
            self.venv_status_var.set(f"{active}{version}")
            self._set_button_state(self._load_button, self._process is None)
            self._set_button_state(self._download_button, self._process is None and bool(self._file_rows))
        else:
            detail = status.error or "not installed"
            self.venv_status_var.set(f"{active} [huggingface_hub missing: {detail}]")
            self._set_button_state(self._load_button, False)
            self._set_button_state(self._download_button, False)
        self._set_button_state(self._install_button, python is not None)
        self._refresh_target_rows()

    def _refresh_target_rows(self):
        if self._target_container is None:
            return
        for child in self._target_container.winfo_children():
            child.destroy()
        selected_paths = tuple(self.launcher.app_settings.get("hf_target_dirs", []))
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
            self._persist_settings()
            return
        for row, option in enumerate(state.options):
            var = tk.BooleanVar(value=option.selected)
            var.trace_add("write", lambda *_a: self._persist_settings())
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

    @staticmethod
    def _looks_like_tqdm_progress(line: str) -> bool:
        return "%|" in line or "B/s" in line or "it/s" in line

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
        command = venv_manager.build_install_dependency_command(
            active,
            self._huggingface_dependency(),
            platform=sys.platform,
        )
        terminal_launcher.open_command_in_terminal(command, cwd=self.repo_dir)
        self.status_var.set(
            "Opened terminal to install or update huggingface_hub. Waiting for it to appear in the venv…"
        )
        self._start_dependency_watch(active)

    def _start_dependency_watch(self, venv_path: str):
        self._cancel_dependency_watch()
        self._dep_watch_venv = venv_path
        self._dep_watch_deadline = time.monotonic() + self.DEP_WATCH_TIMEOUT_S
        self._dep_watch_after_id = self.root.after(
            self.DEP_WATCH_INTERVAL_MS, self._fire_dependency_probe
        )

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
            self._dep_watch_venv = None
            return
        dep = self._huggingface_dependency()

        def worker():
            try:
                status = venv_manager.probe_dependency_status(
                    venv_path, dep, platform=sys.platform
                )
                available = bool(status.available)
            except Exception:
                available = False
            self.root.after(0, self._on_dependency_probe_result, venv_path, available)

        threading.Thread(target=worker, daemon=True).start()

    def _on_dependency_probe_result(self, venv_path: str, available: bool):
        if venv_path != self._dep_watch_venv:
            return
        if venv_path != self._current_active_venv_path():
            self._dep_watch_venv = None
            return
        if available:
            self._dep_watch_venv = None
            self._refresh_runtime_state()
            return
        if time.monotonic() > self._dep_watch_deadline:
            self._dep_watch_venv = None
            return
        self._dep_watch_after_id = self.root.after(
            self.DEP_WATCH_INTERVAL_MS, self._fire_dependency_probe
        )

    def _on_load_repo(self):
        try:
            parsed = normalize_repo_input(self.repo_input_var.get())
        except ValueError as exc:
            messagebox.showerror("Invalid repo", str(exc))
            return
        if parsed.revision_hint and not self.revision_var.get().strip():
            self.revision_var.set(parsed.revision_hint)
        payload = {
            "repo_id": parsed.repo_id,
            "revision": self.revision_var.get().strip(),
            "token": self.token_var.get().strip(),
        }
        self.status_var.set(f"Loading {parsed.repo_id}…")
        self._start_runner("list", payload)

    def _on_download(self):
        if not self._file_rows:
            messagebox.showinfo("No repo loaded", "Load a repo before downloading.")
            return
        selected_targets = [
            path for path, var in self._selected_target_vars.items()
            if bool(var.get())
        ]
        if not selected_targets:
            messagebox.showerror("No target directory", "Select at least one model directory target.")
            return
        selected_files = self._selected_file_paths()
        if self.download_mode_var.get() == "selected" and not selected_files:
            messagebox.showerror("No files selected", "Select at least one file or switch to repo snapshot mode.")
            return
        try:
            max_workers = max(1, int(self.max_workers_var.get().strip() or "4"))
        except ValueError:
            messagebox.showerror("Invalid workers", "Max workers must be a positive integer.")
            return
        try:
            parsed = normalize_repo_input(self.repo_input_var.get())
        except ValueError as exc:
            messagebox.showerror("Invalid repo", str(exc))
            return
        payload = {
            "repo_id": parsed.repo_id,
            "revision": self.revision_var.get().strip(),
            "token": self.token_var.get().strip(),
            "download_mode": self.download_mode_var.get(),
            "selected_files": selected_files,
            "include_patterns": list(parse_pattern_lines(self.include_patterns_var.get())),
            "ignore_patterns": list(parse_pattern_lines(self.ignore_patterns_var.get())),
            "force_download": bool(self.force_download_var.get()),
            "local_files_only": bool(self.local_files_only_var.get()),
            "max_workers": max_workers,
            "target_dirs": selected_targets,
        }
        self.status_var.set(f"Downloading {parsed.repo_id}…")
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
        self._cancel_operation(clean_only=True)
        fd, payload_path = tempfile.mkstemp(prefix="hf-downloader-", suffix=".json")
        os.close(fd)
        payload_file = Path(payload_path)
        payload_file.write_bytes(payload_bytes(payload))
        self._payload_path = payload_file
        self._operation_name = action
        self._progress_stop()
        self._progress.config(mode="indeterminate", value=0)
        self._progress.start(12)
        self.progress_label_var.set("")
        self._set_button_state(self._cancel_button, True)
        self._set_button_state(self._load_button, False)
        self._set_button_state(self._download_button, False)
        command = build_runner_command(python, action, payload_file)
        self._worker_thread = threading.Thread(
            target=self._run_process_worker,
            args=(command,),
            daemon=True,
        )
        self._worker_thread.start()
        if self._queue_after_id is None:
            self._queue_after_id = self.root.after(self.POLL_MS, self._poll_queue)

    def _run_process_worker(self, command: list[str]):
        print(f"[hf-runner] launching: {' '.join(command)}", file=sys.stderr, flush=True)
        try:
            proc = subprocess.Popen(
                command,
                cwd=self.repo_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            print(f"[hf-runner] failed to launch: {exc}", file=sys.stderr, flush=True)
            self._queue.put(
                {
                    "event": "process-exit",
                    "returncode": 1,
                    "stderr": str(exc),
                }
            )
            return
        self._process = proc

        stderr_chunks: list[str] = []

        def stderr_reader():
            assert proc.stderr is not None
            for raw in proc.stderr:
                stripped = raw.rstrip()
                if not stripped:
                    continue
                stderr_chunks.append(stripped)
                print(f"[hf-runner stderr] {stripped}", file=sys.stderr, flush=True)
                self._queue.put({"event": "stderr", "message": stripped})

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
                self._queue.put({"event": "stderr", "message": line})
                continue
            self._queue.put(event)
        returncode = proc.wait()
        stderr_thread.join(timeout=2.0)
        stderr_text = "\n".join(stderr_chunks).strip()
        print(f"[hf-runner] exited with code {returncode}", file=sys.stderr, flush=True)
        self._queue.put(
            {
                "event": "process-exit",
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
        if self._process is not None or not self._queue.empty():
            self._queue_after_id = self.root.after(self.POLL_MS, self._poll_queue)
        elif drained:
            self._refresh_runtime_state()

    def _handle_event(self, event: dict):
        kind = event.get("event")
        if kind == "status":
            self.status_var.set(event.get("message", ""))
            return
        if kind == "listing":
            refs, files = summarize_repo_listing(event.get("refs", []), event.get("files", []))
            self._refs = [ref.name for ref in refs]
            if self._revision_combo is not None:
                self._revision_combo.config(values=self._refs)
            if not self.revision_var.get().strip() and self._refs:
                self.revision_var.set(self._refs[0])
            self._file_rows = list(files)
            self._file_path_by_index = [row.path for row in self._file_rows]
            if self._files_listbox is not None:
                self._files_listbox.delete(0, tk.END)
                for row in self._file_rows:
                    self._files_listbox.insert(tk.END, row.display_name)
                self._select_default_files()
            self.status_var.set(f"Loaded {len(files)} files from {event.get('repo_id', '')}.")
            self._set_button_state(self._download_button, bool(files))
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
        if kind == "process-exit":
            self._finalize_process(event)

    def _finalize_process(self, event: dict):
        returncode = int(event.get("returncode") or 0)
        stderr = event.get("stderr", "")
        self._progress_stop()
        if returncode != 0:
            detail = stderr or self.status_var.get() or f"exit code {returncode}"
            self.status_var.set(f"{self._operation_name} failed: {detail}")
        self._cleanup_process_state()
        self._refresh_runtime_state()

    def _cancel_operation(self, *, clean_only: bool = False):
        proc = self._process
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        if not clean_only:
            self.status_var.set("Operation cancelled.")
        self._cleanup_process_state()
        if not clean_only:
            self._refresh_runtime_state()

    def _cleanup_process_state(self):
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
