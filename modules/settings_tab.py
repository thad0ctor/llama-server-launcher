"""Settings tab: UI theme/font controls plus venv management."""

import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox, font as tkfont

from modules import terminal_launcher
from modules import ui_theme
from modules import venv_manager


class SettingsTab:
    """Settings tab content. Reads/writes values on the launcher's app_settings dict."""

    THEME_MODE_LABELS = [
        ("Auto (follow OS/default)", "auto"),
        ("Light",                    "light"),
        ("Dark",                     "dark"),
        ("Specific theme…",          "specific"),
    ]

    # Mutually-exclusive font size presets. "0" = system default, "custom" = manual override.
    FONT_SIZE_PRESETS = [
        ("Default",  "0"),
        ("10",       "10"),
        ("12",       "12"),
        ("14",       "14"),
        ("16",       "16"),
        ("20",       "20"),
    ]
    FONT_SIZE_MAX = 32  # exclusive upper bound for custom override
    VENV_PROBE_DEBOUNCE_MS = 400

    def __init__(self, launcher):
        self.launcher = launcher
        self.root = launcher.root
        # Mirror ``LaunchManager._effective_venv_path`` resolution so
        # SettingsTab's venv probes/create/remove and the launch path
        # see the SAME repo root on a non-default checkout. Without
        # this, Settings could create one venv while launch reads
        # another. Type-guard against MagicMock auto-vivification.
        repo_dir = getattr(launcher, "repo_dir", None)
        if not isinstance(repo_dir, (str, Path)):
            repo_dir = venv_manager.launcher_repo_dir()
        # Normalize to an absolute, expanded ``Path`` so downstream
        # callers (``describe_venv_target(..., repo_dir=self.repo_dir)``,
        # ``open_command_in_terminal(..., cwd=self.repo_dir)``) always
        # receive the same shape. A relative or ``~``-prefixed value
        # used to vary by launch context — Settings would create one
        # venv layout while the launch path resolved against a
        # different cwd. ``strict=False`` keeps the path object even
        # if the directory hasn't been created yet (the venv-create
        # flow MAKES the parent later).
        try:
            self.repo_dir = Path(repo_dir).expanduser().resolve(strict=False)
        except Exception:
            self.repo_dir = Path(repo_dir).expanduser()

        s = launcher.app_settings
        self.theme_mode_var = tk.StringVar(value=s.get("ui_theme_mode", "auto"))
        self.theme_name_var = tk.StringVar(value=s.get("ui_theme_name", ""))
        self.font_family_var = tk.StringVar(value=s.get("ui_font_family", ""))
        # MagicMock / unrelated assignments could leave ``launcher.venv_dir``
        # set to a non-StringVar (a plain str, a Mock, or None). Anything
        # other than a real ``tk.StringVar`` would break the trace_add /
        # .get() / .set() calls downstream. ``isinstance`` is the only safe
        # check here — ``hasattr`` alone passes for an auto-vivified Mock.
        existing_venv_dir = getattr(self.launcher, "venv_dir", None)
        if not isinstance(existing_venv_dir, tk.StringVar):
            self.launcher.venv_dir = tk.StringVar(value=s.get("last_venv_dir", ""))
        self.venv_dir_var = self.launcher.venv_dir

        # Font size: convert the persisted integer to our two-part state
        # (radio choice + custom Entry). If the stored size matches a preset,
        # select that preset; otherwise select "custom" and seed the entry.
        stored_size = int(s.get("ui_font_size", 0) or 0)
        preset_values = {int(v) for _, v in self.FONT_SIZE_PRESETS}
        if stored_size in preset_values:
            self.font_size_choice_var = tk.StringVar(value=str(stored_size))
            self.font_size_custom_var = tk.StringVar(value="")
        elif stored_size > 0:
            self.font_size_choice_var = tk.StringVar(value="custom")
            self.font_size_custom_var = tk.StringVar(value=str(stored_size))
        else:
            self.font_size_choice_var = tk.StringVar(value="0")
            self.font_size_custom_var = tk.StringVar(value="")
        self._status_var = tk.StringVar(value="")
        self._info_theme_var = tk.StringVar(value="")
        self._info_font_var = tk.StringVar(value="")
        self._venv_effective_var = tk.StringVar(value="")
        self._venv_status_var = tk.StringVar(value="")
        self._venv_note_var = tk.StringVar(value="")
        self._venv_action_status_var = tk.StringVar(value="")
        self._venv_dependencies_frame = None
        self._venv_probe_after_id = None
        # Drain callback id is tracked separately so ``teardown`` can cancel
        # an in-flight drain. Without this, a probe that lands in the
        # queue after the tab is torn down used to call
        # ``_rebuild_dependency_rows`` against destroyed widgets.
        self._venv_probe_drain_after_id = None
        # Tracks the ``root.after`` id for the 2 s deferred probe scheduled
        # after the user clicks Remove venv. Without this, a tab torn down
        # in that 2 s window could still get its dependency table rebuilt
        # against destroyed widgets.
        self._venv_remove_refresh_after_id = None
        self._venv_probe_results = queue.Queue()
        self._venv_probe_generation = 0
        self._venv_trace_token = self.venv_dir_var.trace_add(
            "write",
            lambda *_a: self._on_venv_dir_changed(),
        )

    def teardown(self) -> None:
        """Release the launcher-owned ``venv_dir`` write trace.

        SettingsTab attaches a trace to the launcher's StringVar but the
        var outlives the tab, so without an explicit teardown the trace
        callback keeps a reference to this SettingsTab forever and
        writes to (possibly destroyed) Tk widgets on every later
        ``venv_dir.set(...)``. Safe to call multiple times.
        """
        token = getattr(self, "_venv_trace_token", None)
        if token is not None:
            try:
                self.venv_dir_var.trace_remove("write", token)
            except Exception:
                pass
            self._venv_trace_token = None
        # Cancel any in-flight ``after()`` callbacks so they don't fire
        # against destroyed widgets after teardown.
        for attr in (
            "_venv_probe_after_id",
            "_venv_probe_drain_after_id",
            "_venv_remove_refresh_after_id",
        ):
            after_id = getattr(self, attr, None)
            if after_id is not None:
                try:
                    self.root.after_cancel(after_id)
                except Exception:
                    pass
                setattr(self, attr, None)

    # ------------------------------------------------------------------ setup
    def setup_settings_tab(self, parent):
        parent.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(parent, text="UI Appearance", font=("TkDefaultFont", 12, "bold")) \
            .grid(column=0, row=row, columnspan=3, sticky="w", padx=10, pady=(10, 5))
        row += 1
        ttk.Separator(parent, orient="horizontal") \
            .grid(column=0, row=row, columnspan=3, sticky="ew", padx=10, pady=(0, 10))
        row += 1

        # --- Theme mode ---
        ttk.Label(parent, text="Theme mode:") \
            .grid(column=0, row=row, sticky="w", padx=10, pady=4)
        mode_frame = ttk.Frame(parent)
        mode_frame.grid(column=1, row=row, columnspan=2, sticky="w", padx=5, pady=4)
        for label, value in self.THEME_MODE_LABELS:
            ttk.Radiobutton(
                mode_frame, text=label, value=value,
                variable=self.theme_mode_var,
                command=self._on_theme_mode_changed,
            ).pack(side="left", padx=(0, 10))
        row += 1

        # --- Specific theme picker ---
        ttk.Label(parent, text="Specific theme:") \
            .grid(column=0, row=row, sticky="w", padx=10, pady=4)
        available = ui_theme.list_available_themes(self.root)
        self.theme_combo = ttk.Combobox(
            parent, textvariable=self.theme_name_var,
            values=available, state="readonly", width=30,
        )
        self.theme_combo.grid(column=1, row=row, sticky="w", padx=5, pady=4)
        ttk.Label(parent, text="(used when mode = Specific)", font=("TkSmallCaptionFont",)) \
            .grid(column=2, row=row, sticky="w", padx=5, pady=4)
        row += 1

        ttk.Separator(parent, orient="horizontal") \
            .grid(column=0, row=row, columnspan=3, sticky="ew", padx=10, pady=10)
        row += 1

        # --- Font family ---
        ttk.Label(parent, text="Font family:") \
            .grid(column=0, row=row, sticky="w", padx=10, pady=4)
        # Cache the system font list so _apply_and_save can validate against
        # it without re-querying Tk every time.
        self._available_font_families = ui_theme.list_font_families(self.root)
        self.font_family_combo = ttk.Combobox(
            parent, textvariable=self.font_family_var,
            values=[""] + self._available_font_families, width=30,
        )
        self.font_family_combo.grid(column=1, row=row, sticky="w", padx=5, pady=4)
        ttk.Label(parent, text="(blank = system default)", font=("TkSmallCaptionFont",)) \
            .grid(column=2, row=row, sticky="w", padx=5, pady=4)
        row += 1

        # --- Font size (preset radios + custom override) ---
        ttk.Label(parent, text="Font size:") \
            .grid(column=0, row=row, sticky="nw", padx=10, pady=4)
        size_frame = ttk.Frame(parent)
        size_frame.grid(column=1, row=row, columnspan=2, sticky="w", padx=5, pady=4)

        # Presets
        for label, value in self.FONT_SIZE_PRESETS:
            ttk.Radiobutton(
                size_frame, text=label, value=value,
                variable=self.font_size_choice_var,
                command=self._on_font_size_choice_changed,
            ).pack(side="left", padx=(0, 8))

        # Custom radio + entry
        ttk.Radiobutton(
            size_frame, text="Custom:", value="custom",
            variable=self.font_size_choice_var,
            command=self._on_font_size_choice_changed,
        ).pack(side="left", padx=(0, 4))

        # Validation: restrict typing to digits only; range check happens at apply time.
        vcmd = (self.root.register(self._validate_custom_digit), "%P")
        self.font_size_custom_entry = ttk.Entry(
            size_frame, textvariable=self.font_size_custom_var,
            width=4, validate="key", validatecommand=vcmd,
        )
        self.font_size_custom_entry.pack(side="left")
        # Typing in the entry auto-selects the Custom radio so the user doesn't
        # have to click it manually.
        self.font_size_custom_entry.bind("<KeyRelease>", self._on_custom_entry_keyrelease)
        ttk.Label(size_frame, text=f"  (must be < {self.FONT_SIZE_MAX})",
                  font=("TkSmallCaptionFont",)).pack(side="left")
        row += 1

        ttk.Label(
            parent,
            text="(Font changes may require restarting the launcher to fully take effect.)",
            font=("TkSmallCaptionFont",),
        ).grid(column=0, row=row, columnspan=3, sticky="w", padx=10, pady=(0, 10))
        row += 1

        # --- Action buttons ---
        btns = ttk.Frame(parent)
        btns.grid(column=0, row=row, columnspan=3, sticky="w", padx=10, pady=(10, 5))
        ttk.Button(btns, text="Apply & Save", command=self._apply_and_save) \
            .pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="Reset to Defaults", command=self._reset_defaults) \
            .pack(side="left", padx=(0, 8))
        row += 1

        ttk.Label(parent, textvariable=self._status_var,
                  foreground="#5a9", font=("TkSmallCaptionFont",)) \
            .grid(column=0, row=row, columnspan=3, sticky="w", padx=10, pady=(0, 10))
        row += 1

        # --- Active appearance readout ---
        info_frame = ttk.LabelFrame(parent, text="Active appearance")
        info_frame.grid(column=0, row=row, columnspan=3, sticky="ew", padx=10, pady=(10, 10))
        info_frame.columnconfigure(1, weight=1)
        ttk.Label(info_frame, text="Theme:").grid(column=0, row=0, sticky="w", padx=6, pady=2)
        ttk.Label(info_frame, textvariable=self._info_theme_var).grid(column=1, row=0, sticky="w", padx=6, pady=2)
        ttk.Label(info_frame, text="Font:").grid(column=0, row=1, sticky="w", padx=6, pady=2)
        ttk.Label(info_frame, textvariable=self._info_font_var).grid(column=1, row=1, sticky="w", padx=6, pady=2)
        row += 1

        ttk.Separator(parent, orient="horizontal") \
            .grid(column=0, row=row, columnspan=3, sticky="ew", padx=10, pady=(10, 10))
        row += 1

        self._build_venv_section(parent, row)
        row += 1

        self._on_theme_mode_changed()
        self._on_font_size_choice_changed()
        self._refresh_active_info()
        self._refresh_venv_summary()
        self._schedule_venv_dependency_probe()

    def _build_venv_section(self, parent, row):
        lf = ttk.LabelFrame(parent, text="Python Environment")
        lf.grid(column=0, row=row, columnspan=3, sticky="ew", padx=10, pady=(0, 10))
        lf.columnconfigure(1, weight=1)

        ttk.Label(lf, text="Virtual environment:") \
            .grid(column=0, row=0, sticky="w", padx=6, pady=4)
        ttk.Entry(lf, textvariable=self.venv_dir_var, width=48) \
            .grid(column=1, row=0, columnspan=2, sticky="ew", padx=4, pady=4)
        ttk.Label(
            lf,
            text="Leave blank to use the repo default venv folder.",
            font=("TkSmallCaptionFont",),
        ).grid(column=1, row=1, columnspan=2, sticky="w", padx=4)

        ttk.Label(lf, text="Effective path:") \
            .grid(column=0, row=2, sticky="w", padx=6, pady=4)
        ttk.Label(lf, textvariable=self._venv_effective_var) \
            .grid(column=1, row=2, columnspan=2, sticky="w", padx=4, pady=4)

        ttk.Label(lf, text="Status:") \
            .grid(column=0, row=3, sticky="w", padx=6, pady=4)
        ttk.Label(lf, textvariable=self._venv_status_var) \
            .grid(column=1, row=3, columnspan=2, sticky="w", padx=4, pady=4)

        btns = ttk.Frame(lf)
        btns.grid(column=1, row=4, columnspan=2, sticky="w", padx=4, pady=(4, 6))
        ttk.Button(btns, text="Create venv", command=self._on_create_venv) \
            .pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="Remove venv", command=self._on_remove_venv) \
            .pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="Refresh deps", command=self._schedule_venv_dependency_probe) \
            .pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="Clear path", command=lambda: self.venv_dir_var.set("")) \
            .pack(side="left")

        ttk.Label(
            lf,
            textvariable=self._venv_note_var,
            font=("TkSmallCaptionFont",),
            foreground="#666",
        ).grid(column=0, row=5, columnspan=3, sticky="w", padx=6, pady=(0, 6))

        deps = ttk.LabelFrame(lf, text="Managed packages")
        deps.grid(column=0, row=6, columnspan=3, sticky="ew", padx=6, pady=(0, 6))
        deps.columnconfigure(1, weight=1)
        self._venv_dependencies_frame = deps
        self._rebuild_dependency_rows([])

        ttk.Label(
            lf,
            textvariable=self._venv_action_status_var,
            foreground="#5a9",
            font=("TkSmallCaptionFont",),
        ).grid(column=0, row=7, columnspan=3, sticky="w", padx=6, pady=(0, 6))

    # ------------------------------------------------------------------ events
    def _on_theme_mode_changed(self):
        mode = self.theme_mode_var.get()
        state = "readonly" if mode == "specific" else "disabled"
        try:
            self.theme_combo.configure(state=state)
        except tk.TclError:
            pass

    def _on_font_size_choice_changed(self):
        """Enable the custom Entry only when 'Custom' is selected."""
        custom_selected = self.font_size_choice_var.get() == "custom"
        try:
            self.font_size_custom_entry.configure(
                state="normal" if custom_selected else "disabled",
            )
        except tk.TclError:
            pass

    def _on_custom_entry_keyrelease(self, _event):
        """User typed in the Custom entry — auto-select the Custom radio."""
        self.font_size_choice_var.set("custom")
        self._on_font_size_choice_changed()

    def _on_venv_dir_changed(self):
        self._refresh_venv_summary()
        self._schedule_venv_dependency_probe()

    def _refresh_venv_summary(self):
        info = self._current_venv_info()
        active_path = self._current_active_venv_path()
        self._venv_effective_var.set(str(info.effective_dir))
        if info.looks_like_venv:
            self._venv_status_var.set("Virtual environment detected.")
        elif info.exists:
            self._venv_status_var.set("Directory exists but does not contain a venv Python interpreter.")
        else:
            self._venv_status_var.set("Virtual environment not created yet.")
        if info.uses_default:
            if active_path:
                self._venv_note_var.set(
                    f"Blank entry activates the repo default venv: {active_path}"
                )
            else:
                self._venv_note_var.set(
                    f"Blank entry creates or probes the repo default path: {info.effective_dir}"
                )
        else:
            self._venv_note_var.set(
                "Relative paths resolve from the repo root, and launch/GPU detection use this same resolved path."
            )

    def _schedule_venv_dependency_probe(self):
        if self._venv_probe_after_id is not None:
            try:
                self.root.after_cancel(self._venv_probe_after_id)
            except Exception:
                pass
        self._venv_probe_after_id = self.root.after(
            self.VENV_PROBE_DEBOUNCE_MS,
            self._start_venv_dependency_probe,
        )

    def _start_venv_dependency_probe(self):
        self._venv_probe_after_id = None
        info = self._current_venv_info()
        active_path = self._current_active_venv_path()
        generation = self._venv_probe_generation + 1
        self._venv_probe_generation = generation
        if not info.looks_like_venv or not active_path:
            # Cancel any drain scheduled by a prior generation. Otherwise
            # ``_drain_venv_dependency_probe`` keeps rescheduling every
            # 75 ms forever because the now-current generation will never
            # produce a result.
            drain_id = self._venv_probe_drain_after_id
            if drain_id is not None:
                try:
                    self.root.after_cancel(drain_id)
                except Exception:
                    pass
                self._venv_probe_drain_after_id = None
            self._rebuild_dependency_rows([])
            self._venv_action_status_var.set(
                "Create a venv or point this field at an existing one to manage packages."
            )
            return
        self._venv_action_status_var.set("Checking managed packages…")
        threading.Thread(
            target=self._probe_venv_dependencies_worker,
            args=(generation, active_path),
            name="SettingsVenvProbe",
            daemon=True,
        ).start()
        self._schedule_venv_probe_drain()

    def _schedule_venv_probe_drain(self):
        """Track the drain ``after()`` id so teardown can cancel it."""
        prior = getattr(self, "_venv_probe_drain_after_id", None)
        if prior is not None:
            try:
                self.root.after_cancel(prior)
            except Exception:
                pass
        try:
            self._venv_probe_drain_after_id = self.root.after(
                75, self._drain_venv_dependency_probe
            )
        except tk.TclError:
            self._venv_probe_drain_after_id = None

    def _probe_venv_dependencies_worker(self, generation, venv_dir):
        try:
            statuses = venv_manager.probe_dependencies(venv_dir)
            self._venv_probe_results.put((generation, statuses, None))
        except Exception as exc:
            self._venv_probe_results.put((generation, [], exc))

    def _drain_venv_dependency_probe(self):
        # Clear the tracked id at the top so a fresh ``_schedule_venv_probe_drain``
        # call doesn't get cancelled by ``teardown`` after we already started.
        self._venv_probe_drain_after_id = None
        # Drain stale generations first; without this, a stale entry at the
        # head of the queue swallowed the scheduled drain and the *next*
        # (fresh) result sat in the queue forever, leaving the UI stuck on
        # "Checking managed packages…" until the user re-edited the venv
        # path.
        while True:
            try:
                generation, statuses, error = self._venv_probe_results.get_nowait()
            except queue.Empty:
                # Nothing matched yet — try again on the next tick.
                self._schedule_venv_probe_drain()
                return
            if generation == self._venv_probe_generation:
                break
            # Stale: drop and keep looking.
        if error is not None:
            self._venv_action_status_var.set(f"Dependency probe failed: {error}")
            self._rebuild_dependency_rows([])
            return
        self._rebuild_dependency_rows(statuses)
        self._venv_action_status_var.set("Managed package status refreshed.")

    def _rebuild_dependency_rows(self, statuses):
        frame = self._venv_dependencies_frame
        if frame is None:
            return
        for child in frame.winfo_children():
            child.destroy()
        status_by_key = {row.dependency.key: row for row in statuses}
        has_venv = self._current_venv_info().looks_like_venv and bool(
            self._current_active_venv_path()
        )
        for row_index, dep in enumerate(venv_manager.MANAGED_DEPENDENCIES):
            status = status_by_key.get(dep.key)
            can_install = bool(has_venv and status is not None and not status.available)
            can_remove = bool(
                has_venv and status is not None and status.available and not dep.required
            )
            ttk.Label(frame, text=f"{dep.label}:").grid(
                column=0, row=row_index, sticky="nw", padx=4, pady=4,
            )
            ttk.Label(
                frame,
                text=self._format_dependency_status(dep, status),
                font=("TkSmallCaptionFont",),
            ).grid(column=1, row=row_index, sticky="w", padx=4, pady=4)
            btns = ttk.Frame(frame)
            btns.grid(column=2, row=row_index, sticky="e", padx=4, pady=4)
            ttk.Button(
                btns,
                text="Install",
                command=lambda d=dep: self._on_install_dependency(d),
                state="normal" if can_install else "disabled",
            ).pack(side="left", padx=(0, 4))
            ttk.Button(
                btns,
                text="Remove",
                command=lambda d=dep: self._on_remove_dependency(d),
                state="normal" if can_remove else "disabled",
            ).pack(side="left")

    @staticmethod
    def _format_dependency_status(dependency, status):
        parts = ["Required." if dependency.required else "Optional.", dependency.description]
        if status is None:
            parts.append("Status unavailable until a venv is detected.")
        elif status.available:
            version = f" {status.version}" if status.version else ""
            parts.append(f"✓ Installed{version}.")
        elif status.error:
            parts.append(f"Not installed ({status.error}).")
        else:
            parts.append("Not installed.")
        if dependency.key == "torch":
            parts.append("Generic install uses `pip install torch`; GPU-specific wheels may need manual replacement.")
        return " ".join(parts)

    def _current_venv_info(self):
        return venv_manager.describe_venv_target(
            self.venv_dir_var.get(),
            repo_dir=self.repo_dir,
        )

    def _current_active_venv_path(self):
        return venv_manager.resolve_active_venv_path(
            self.venv_dir_var.get(),
            repo_dir=self.repo_dir,
        )

    def _on_create_venv(self):
        info = self._current_venv_info()
        # If a venv already exists at the target, re-running ``python -m venv``
        # over it can rewrite ``pyvenv.cfg`` against a different base
        # interpreter and leave the env in a half-rebuilt state. Confirm
        # explicitly rather than silently re-bootstrapping.
        if info.exists and info.looks_like_venv:
            if not messagebox.askyesno(
                "Create venv",
                "A virtual environment already exists at:\n\n"
                f"{info.effective_dir}\n\n"
                "Re-running `python -m venv` will rewrite its configuration "
                "(activators, pyvenv.cfg) and may leave it in an inconsistent "
                "state if the system python has changed. Proceed anyway?",
            ):
                return
        elif info.exists:
            # Target exists but isn't a venv. The previous behavior was to
            # prompt and proceed if the user confirmed; that left the user
            # one accidental click from mixing venv files into an
            # arbitrary project directory which Remove venv would then
            # refuse to clean up. Refuse outright when the directory is
            # non-empty — point the user at an empty/new path instead.
            try:
                has_contents = any(info.effective_dir.iterdir())
            except OSError:
                # If we can't list it (permission error, race), err on
                # the safe side and refuse rather than proceeding.
                has_contents = True
            if has_contents:
                messagebox.showerror(
                    "Create venv",
                    "The selected directory already exists and is not a "
                    "virtual environment:\n\n"
                    f"{info.effective_dir}\n\n"
                    "Refusing to bootstrap a venv inside a non-empty "
                    "non-venv directory — venv files would mix with the "
                    "existing contents and Remove venv would later refuse "
                    "to clean it up. Point ``Venv dir`` at an empty or "
                    "new path.",
                )
                return
        try:
            info.effective_dir.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            messagebox.showerror("Create venv", f"Could not prepare parent directory:\n{exc}")
            return
        try:
            # Build the command AND launch in the same try/except so a
            # ``_shell_join``/path-validation ``ValueError`` raised
            # inside ``build_create_venv_command`` reaches the user as
            # a clear messagebox instead of bubbling up through the Tk
            # callback (which on Python 3.13 can corrupt the next
            # Tcl interp).
            command = venv_manager.build_create_venv_command(info.effective_dir)
            terminal_launcher.open_command_in_terminal(command, cwd=self.repo_dir)
        except Exception as exc:
            messagebox.showerror("Create venv", f"Failed to create venv:\n{exc}")
            return
        # Do NOT overwrite the user's typed value with the resolved
        # absolute path. If they left the field blank or used a relative
        # repo-anchored value (``venv``, ``./venv``), preserve that
        # intent so the next launch/launcher continues to resolve it
        # against the active repo_dir. ``info.effective_dir`` is only
        # used internally for the actual ``python -m venv`` invocation.
        self._venv_action_status_var.set(
            f"Opened terminal to create venv at {info.effective_dir}."
        )
        messagebox.showinfo(
            "Create venv",
            "Opened a terminal to create the virtual environment. The terminal will report success or failure; refresh deps when it finishes.",
        )

    def _on_remove_venv(self):
        info = self._current_venv_info()
        if not info.exists:
            messagebox.showinfo("Remove venv", "No virtual environment directory exists at the selected path.")
            return
        if not info.looks_like_venv:
            messagebox.showerror(
                "Remove venv",
                "The selected directory does not look like a virtual environment "
                "(no python interpreter found under bin/ or Scripts/). Refusing to "
                f"recursively delete:\n\n{info.effective_dir}",
            )
            return
        if not messagebox.askyesno(
            "Remove venv",
            f"Remove the virtual environment directory?\n\n{info.effective_dir}",
        ):
            return
        try:
            # Wrap both build + launch so ``_shell_join`` rejections
            # (``%``/``!``) or path-resolution ValueErrors raised inside
            # ``build_remove_venv_command`` surface as a messagebox.
            command = venv_manager.build_remove_venv_command(info.effective_dir)
            terminal_launcher.open_command_in_terminal(command, cwd=info.effective_dir.parent)
        except Exception as exc:
            messagebox.showerror("Remove venv", f"Failed to remove venv:\n{exc}")
            return
        self._venv_action_status_var.set(
            f"Opened terminal to remove venv at {info.effective_dir}."
        )
        # Bump the probe generation BEFORE clearing the table so any
        # in-flight ``_schedule_venv_dependency_probe`` worker that
        # completes after this point silently drops its result via
        # the existing ``generation == self._venv_probe_generation``
        # check (line ~472). Without this, a probe that started just
        # before the user clicked Remove venv would land its
        # "installed" rows BACK into the table we just cleared,
        # making it look like the rm didn't actually run.
        self._venv_probe_generation += 1
        # Also cancel any QUEUED probe callbacks — the generation bump
        # only covers workers that have already started. A pending
        # ``after()`` timer would otherwise fire, schedule a fresh
        # probe (which bumps the generation again), and restore the
        # old "installed" rows while the delete terminal is still
        # running. Mirrors the ``_venv_remove_refresh_after_id``
        # cancel pattern below.
        for attr in ("_venv_probe_after_id", "_venv_probe_drain_after_id"):
            pending_id = getattr(self, attr, None)
            if pending_id is not None:
                try:
                    self.root.after_cancel(pending_id)
                except Exception:
                    pass
                setattr(self, attr, None)
        # Clear the dependency table immediately so the UI doesn't keep
        # showing "installed" rows for packages whose venv is being
        # deleted in another terminal. Schedule a refresh ~2 s later so
        # the table catches up once the rm completes; the user can also
        # click Refresh deps manually.
        self._rebuild_dependency_rows([])
        # If the user clicks Remove venv multiple times inside the 2 s
        # window, the prior ``after()`` id would be lost and the orphan
        # callback could re-enter ``_schedule_venv_dependency_probe``
        # against destroyed widgets after teardown.
        prior_refresh_id = self._venv_remove_refresh_after_id
        if prior_refresh_id is not None:
            try:
                self.root.after_cancel(prior_refresh_id)
            except Exception:
                pass
            self._venv_remove_refresh_after_id = None
        try:
            self._venv_remove_refresh_after_id = self.root.after(
                2000, self._run_remove_venv_refresh
            )
        except tk.TclError:
            self._venv_remove_refresh_after_id = None

    def _run_remove_venv_refresh(self):
        # Clear the tracked id at top so a manual cancel during this callback
        # window doesn't try to ``after_cancel`` a callback that's already
        # firing.
        self._venv_remove_refresh_after_id = None
        # Refresh the summary too — without this, the "Virtual environment
        # detected." label stays stuck even after the directory was deleted
        # in another terminal, until the user manually edits the path.
        self._refresh_venv_summary()
        self._schedule_venv_dependency_probe()

    def _on_install_dependency(self, dependency):
        info = self._current_venv_info()
        if not info.looks_like_venv:
            messagebox.showerror(
                "Install dependency",
                "Create a venv or point this field at an existing venv first.",
            )
            return
        try:
            command = venv_manager.build_install_dependency_command(info.effective_dir, dependency)
            terminal_launcher.open_command_in_terminal(command, cwd=self.repo_dir)
        except Exception as exc:
            messagebox.showerror("Install dependency", f"Failed to install dependency:\n{exc}")
            return
        self._venv_action_status_var.set(
            f"Opened terminal to install {dependency.package_name}."
        )

    def _on_remove_dependency(self, dependency):
        info = self._current_venv_info()
        if not info.looks_like_venv:
            messagebox.showerror(
                "Remove dependency",
                "Create a venv or point this field at an existing venv first.",
            )
            return
        try:
            command = venv_manager.build_remove_dependency_command(info.effective_dir, dependency)
            terminal_launcher.open_command_in_terminal(command, cwd=self.repo_dir)
        except Exception as exc:
            messagebox.showerror("Remove dependency", f"Failed to remove dependency:\n{exc}")
            return
        self._venv_action_status_var.set(
            f"Opened terminal to remove {dependency.package_name}."
        )

    def _validate_custom_digit(self, proposed):
        """Entry validatecommand: only allow empty or pure digit strings up to 3 chars."""
        if proposed == "":
            return True
        if not proposed.isdigit():
            return False
        if len(proposed) > 3:
            return False
        return True

    def _resolve_font_size(self):
        """Read the radio+entry combo and return an int, or raise ValueError on invalid."""
        choice = self.font_size_choice_var.get()
        if choice == "custom":
            raw = self.font_size_custom_var.get().strip()
            if not raw:
                raise ValueError("Enter a number for Custom font size.")
            if not raw.isdigit():
                raise ValueError(f"Font size must be an integer (got '{raw}').")
            size = int(raw)
            if size <= 0:
                raise ValueError("Font size must be greater than 0.")
            if size >= self.FONT_SIZE_MAX:
                raise ValueError(
                    f"Font size must be less than {self.FONT_SIZE_MAX} (got {size}).",
                )
            return size
        try:
            return int(choice)
        except (TypeError, ValueError):
            return 0

    def _persist_ui_settings(self, new_values, failure_title,
                             failure_prefix, log_prefix):
        """Apply a dict of new UI settings to app_settings and save.

        Snapshots the keys we're about to touch first, so that if
        ``launcher._save_configs()`` raises we can restore the previous
        values — otherwise another save path could later persist the
        rejected state the user was told wasn't saved.

        Returns True on success, False on failure (caller should short-circuit).
        """
        s = self.launcher.app_settings
        snapshot = {k: s.get(k) for k in new_values}
        s.update(new_values)
        try:
            self.launcher._save_configs()
        except Exception as e:
            for k, v in snapshot.items():
                if v is None:
                    s.pop(k, None)
                else:
                    s[k] = v
            print(f"{log_prefix}: {e}", file=sys.stderr)
            messagebox.showerror(failure_title, f"{failure_prefix}:\n{e}")
            return False
        return True

    def _validate_family(self, family):
        """Confirm ``family`` is in the system font list and return its
        canonical (case-matched) spelling. Returns "" for empty input.
        Raises ValueError if the typed family doesn't match any system font —
        keeps typos from being persisted to app_settings.
        """
        if not family:
            return ""
        available = getattr(self, "_available_font_families", None) or []
        # Match case-insensitively so "arial" still resolves to "Arial", but
        # persist the canonical spelling Tk actually advertises.
        lut = {f.lower(): f for f in available}
        canonical = lut.get(family.lower())
        if not canonical:
            raise ValueError(
                f"'{family}' is not available on this system.\n\n"
                "Choose a family from the dropdown, or leave the field blank "
                "to use the system default.",
            )
        return canonical

    # ------------------------------------------------------------------ actions
    def _apply_and_save(self):
        mode = self.theme_mode_var.get()
        theme_name = self.theme_name_var.get() if mode == "specific" else ""
        family = self.font_family_var.get().strip()

        try:
            family = self._validate_family(family)
        except ValueError as e:
            messagebox.showwarning("Unknown font family", str(e))
            return
        # Normalize the var so the user sees the canonical spelling
        if family != self.font_family_var.get():
            self.font_family_var.set(family)

        try:
            size = self._resolve_font_size()
        except ValueError as e:
            messagebox.showwarning("Invalid font size", str(e))
            return

        if mode == "specific":
            if not theme_name:
                messagebox.showwarning(
                    "Select a theme",
                    "Choose a theme name from the list when mode is 'Specific theme…'.",
                )
                return
            # The combobox is readonly, but theme_name_var can also be seeded
            # from a stale app_settings value. Confirm the theme is actually
            # installed — otherwise apply_theme() silently falls back to auto
            # and we'd re-save mode="specific" pointing at something that
            # never applied.
            available = set(ui_theme.list_available_themes(self.root))
            if theme_name not in available:
                messagebox.showwarning(
                    "Unknown theme",
                    f"'{theme_name}' is not available on this system.\n\n"
                    "Pick a theme from the dropdown, or change the mode to "
                    "Auto / Light / Dark.",
                )
                return

        # Always start from the shipped named-font values before layering the
        # user's overrides. apply_fonts only touches attributes that are set
        # (blank family or size<=0 are treated as "don't change"), so if the
        # user clears just one field after previously customising both, the old
        # value would otherwise stick — e.g. Arial/16 → blank family + 12
        # would render as Arial/12 instead of system-default/12.
        try:
            ui_theme.reset_fonts_to_system(self.root)
        except Exception as e:
            print(f"Font reset error: {e}", file=sys.stderr)

        try:
            ui_theme.apply_ui_preferences(
                self.root,
                theme_mode=mode,
                explicit_theme=theme_name or None,
                font_family=family,
                font_size=size,
            )
        except Exception as e:
            # Don't persist (or even mutate in-memory state) when apply fails —
            # the user would otherwise reopen the app thinking this theme/font
            # is active, and other save paths could pick up a state that never
            # actually rendered.
            print(f"Settings apply error: {e}", file=sys.stderr)
            messagebox.showerror("Apply failed", f"Could not apply settings:\n{e}")
            return

        # Commit to app_settings only after a successful apply — and roll
        # back on save failure so a later Save Config / on_exit path can't
        # persist values the user was told weren't saved.
        if not self._persist_ui_settings(
            {
                "ui_theme_mode":  mode,
                "ui_theme_name":  theme_name,
                "ui_font_family": family,
                "ui_font_size":   size,
            },
            failure_title="Save failed",
            failure_prefix="Could not persist settings",
            log_prefix="Settings save error",
        ):
            return

        self._refresh_active_info()
        self._status_var.set("Settings applied and saved. Some changes may need a restart.")

    def _reset_defaults(self):
        """Clear all UI preferences and apply + persist immediately."""
        if not messagebox.askyesno(
            "Reset appearance",
            "Reset theme and font to system defaults?\n\n"
            "Font changes applied at runtime will revert to their shipped values. "
            "Some restored settings may only fully take effect after restarting the launcher.",
        ):
            return
        self.theme_mode_var.set("auto")
        self.theme_name_var.set("")
        self.font_family_var.set("")
        self.font_size_choice_var.set("0")
        self.font_size_custom_var.set("")
        self._on_theme_mode_changed()
        self._on_font_size_choice_changed()

        try:
            ui_theme.reset_fonts_to_system(self.root)
        except Exception as e:
            print(f"Font reset error: {e}", file=sys.stderr)

        try:
            ui_theme.apply_ui_preferences(
                self.root, theme_mode="auto",
                explicit_theme=None, font_family="", font_size=0,
            )
        except Exception as e:
            # Same rationale as _apply_and_save: never mutate / persist a
            # state that didn't actually take effect.
            print(f"Reset apply error: {e}", file=sys.stderr)
            messagebox.showerror("Reset failed", f"Could not apply reset:\n{e}")
            return

        # Commit to app_settings only after a successful apply — and roll
        # back on save failure so a later Save Config / on_exit path can't
        # persist values the user was told weren't saved.
        if not self._persist_ui_settings(
            {
                "ui_theme_mode":  "auto",
                "ui_theme_name":  "",
                "ui_font_family": "",
                "ui_font_size":   0,
            },
            failure_title="Save failed",
            failure_prefix="Could not persist reset",
            log_prefix="Reset save error",
        ):
            return

        self._refresh_active_info()
        self._status_var.set("Reset to defaults and saved. Restart for full effect if needed.")

    def _refresh_active_info(self):
        try:
            active_theme = ttk.Style(self.root).theme_use()
        except tk.TclError:
            active_theme = "(unknown)"
        try:
            default_font = tkfont.nametofont("TkDefaultFont", root=self.root)
            fam = default_font.cget("family")
            sz = int(default_font.cget("size") or 0)
            # Tk stores pixel sizes as negatives; display the absolute value and
            # whether it's points (positive) or pixels (negative original).
            unit = "pt" if sz >= 0 else "px"
            self._info_font_var.set(f"{fam} @ {abs(sz)}{unit}")
        except Exception:
            self._info_font_var.set("(unknown)")
        self._info_theme_var.set(str(active_theme))


def create_settings_tab(launcher):
    return SettingsTab(launcher)
