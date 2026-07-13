"""Benchmark tab UI.

Ties the benchmark package together: a build/tool picker (probing every
configured build for the bench binaries), a per-lever sweep editor, a live
command preview, an in-app streaming run with a results grid, and
export/save actions (results → CSV/JSON/MD, named sweep configs, runnable
.sh/.ps1 scripts).

Heavy widget construction is deferred via the launcher's lazy-tab machinery;
only cheap Tk vars are created in ``__init__``.
"""

from __future__ import annotations

import shlex
import sys
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from . import bench_script, results
from .bench_persistence import BenchConfig, BenchConfigStore
from .bench_runner import (
    EVENT_CANCELLED,
    EVENT_DONE,
    EVENT_ERROR,
    EVENT_LINE,
    EVENT_STEP_RESULT,
    EVENT_STEP_START,
    BenchPlan,
    BenchRunner,
    BenchStep,
    StepInfo,
    StepResult,
)
from .detection import ALL_TOOLS, TOOL_LLAMA_BENCH, TOOL_SWEEP_BENCH, BuildEntry, discover_builds
from .matrix import (
    LEVERS,
    Axis,
    SweepError,
    build_commands,
    expand_range,
    matrix_size,
    parse_list,
    split_axes_for_tool,
)

CONSOLE_MAX_LINES = 5000
RUNNER_POLL_MS = 60
RUNNER_CATCHUP_POLL_MS = 5
RUNNER_MAX_EVENTS_PER_POLL = 200

_TOOL_LABELS = {
    TOOL_LLAMA_BENCH: "llama-bench",
    TOOL_SWEEP_BENCH: "llama-sweep-bench (ik_llama)",
}


class BenchmarkTab:
    def __init__(self, launcher) -> None:
        self.launcher = launcher
        self.root = launcher.root
        try:
            config_dir = Path(launcher.config_path).parent
        except Exception:
            config_dir = Path.cwd() / "config"
        self.store = BenchConfigStore(config_dir)
        self.runner = BenchRunner()

        # Discovery state
        self._builds: list[BuildEntry] = []
        self._build_labels: list[str] = []

        # Selection vars
        self.build_var = tk.StringVar()
        self.tool_var = tk.StringVar(value=TOOL_LLAMA_BENCH)
        self.model_var = tk.StringVar(value=self._launcher_model_path())
        self.output_format_var = tk.StringVar(value="json")
        self.repetitions_var = tk.StringVar(value="")
        self.extra_args_var = tk.StringVar(value="")
        self.config_name_var = tk.StringVar()
        self.autoscroll_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Idle.")

        # Per-lever row vars: key -> dict(include, mode, values, vmin, vmax, vstep)
        self.lever_vars: dict[str, dict[str, tk.Variable]] = {}
        for lever in LEVERS:
            self.lever_vars[lever.key] = {
                "include": tk.BooleanVar(value=False),
                "mode": tk.StringVar(value="list"),
                "values": tk.StringVar(value=""),
                "vmin": tk.StringVar(value="0"),
                "vmax": tk.StringVar(value="0"),
                "vstep": tk.StringVar(value="1"),
            }

        # Widgets populated in setup_tab
        self._console: tk.Text | None = None
        self._console_buffer: list[tuple[str, str | None]] = []
        self._preview: tk.Text | None = None
        self._results_tree: ttk.Treeview | None = None
        self._start_btn: ttk.Button | None = None
        self._cancel_btn: ttk.Button | None = None
        self._build_combo: ttk.Combobox | None = None
        self._tool_combo: ttk.Combobox | None = None
        self._config_combo: ttk.Combobox | None = None
        self._lever_rows: dict[str, dict] = {}

        self._poll_after_id: str | None = None
        self._result_rows: list[results.ResultRow] = []
        self._notebook = None
        self._tab_text = "Benchmark"

    # ------------------------------------------------------------------ helpers
    def _launcher_model_path(self) -> str:
        try:
            return self.launcher.model_path.get()
        except Exception:
            return ""

    def register_with_notebook(self, notebook, tab_text: str) -> None:
        self._notebook = notebook
        self._tab_text = tab_text

    # ------------------------------------------------------------------ UI build
    def setup_tab(self, parent) -> None:
        outer = ttk.Frame(parent)
        outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(outer, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        body = ttk.Frame(canvas)
        body_id = canvas.create_window((0, 0), window=body, anchor="nw")

        def _on_body_config(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_config(event):
            canvas.itemconfigure(body_id, width=event.width)

        body.bind("<Configure>", _on_body_config)
        canvas.bind("<Configure>", _on_canvas_config)

        self._build_source_section(body)
        self._build_sweep_section(body)
        self._build_options_section(body)
        self._build_preview_section(body)
        self._build_run_section(body)
        self._build_results_section(body)
        self._build_configs_section(body)

        self.rescan_builds()
        self._on_tool_changed()
        self.refresh_preview()

    def _section(self, parent, title: str) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text=title)
        frame.pack(fill="x", expand=False, padx=8, pady=6)
        return frame

    def _build_source_section(self, parent) -> None:
        sec = self._section(parent, "Build & Tool")

        row = ttk.Frame(sec)
        row.pack(fill="x", padx=6, pady=3)
        ttk.Label(row, text="Build:", width=12).pack(side="left")
        self._build_combo = ttk.Combobox(row, textvariable=self.build_var, state="readonly")
        self._build_combo.pack(side="left", fill="x", expand=True)
        self._build_combo.bind("<<ComboboxSelected>>", lambda e: self._on_build_changed())
        ttk.Button(row, text="Rescan", command=self.rescan_builds).pack(side="left", padx=4)

        row2 = ttk.Frame(sec)
        row2.pack(fill="x", padx=6, pady=3)
        ttk.Label(row2, text="Tool:", width=12).pack(side="left")
        self._tool_combo = ttk.Combobox(
            row2,
            textvariable=self.tool_var,
            state="readonly",
            values=[_TOOL_LABELS[t] for t in ALL_TOOLS],
        )
        self._tool_combo.pack(side="left", fill="x", expand=True)
        self._tool_combo.bind("<<ComboboxSelected>>", lambda e: self._on_tool_label_selected())
        self._tool_combo.set(_TOOL_LABELS[TOOL_LLAMA_BENCH])

        row3 = ttk.Frame(sec)
        row3.pack(fill="x", padx=6, pady=3)
        ttk.Label(row3, text="Model:", width=12).pack(side="left")
        ttk.Entry(row3, textvariable=self.model_var).pack(side="left", fill="x", expand=True)
        ttk.Button(row3, text="Browse…", command=self._browse_model).pack(side="left", padx=4)

        row4 = ttk.Frame(sec)
        row4.pack(fill="x", padx=6, pady=3)
        ttk.Button(row4, text="Seed from current config", command=self.seed_from_config).pack(side="left")
        ttk.Label(row4, textvariable=self._tool_note_var()).pack(side="left", padx=8)

    _tool_note = None

    def _tool_note_var(self):
        if self._tool_note is None:
            self._tool_note = tk.StringVar(value="")
        return self._tool_note

    def _build_sweep_section(self, parent) -> None:
        sec = self._section(parent, "Sweep parameters")
        ttk.Label(
            sec,
            text="Tick a parameter to include it. 'list' = comma-separated values (e.g. 0,10,20); "
            "'range' = min/max/step. Multiple values sweep the matrix.",
            foreground="#666",
            wraplength=680,
            justify="left",
        ).pack(fill="x", padx=6, pady=(2, 4))

        grid = ttk.Frame(sec)
        grid.pack(fill="x", padx=6, pady=2)
        headers = ["", "Parameter", "Mode", "List values", "Min", "Max", "Step"]
        for c, text in enumerate(headers):
            ttk.Label(grid, text=text, foreground="#444").grid(row=0, column=c, sticky="w", padx=3)

        for r, lever in enumerate(LEVERS, start=1):
            v = self.lever_vars[lever.key]
            widgets: dict = {}
            chk = ttk.Checkbutton(grid, variable=v["include"], command=self.refresh_preview)
            chk.grid(row=r, column=0, sticky="w", padx=3)
            lbl = ttk.Label(grid, text=lever.label)
            lbl.grid(row=r, column=1, sticky="w", padx=3)
            mode = ttk.Combobox(grid, textvariable=v["mode"], state="readonly", values=["list", "range"], width=6)
            mode.grid(row=r, column=2, sticky="w", padx=3)
            mode.bind("<<ComboboxSelected>>", lambda e, k=lever.key: self._on_mode_changed(k))
            vals = ttk.Entry(grid, textvariable=v["values"], width=22)
            vals.grid(row=r, column=3, sticky="we", padx=3)
            vals.bind("<FocusOut>", lambda e: self.refresh_preview())
            emin = ttk.Entry(grid, textvariable=v["vmin"], width=6)
            emin.grid(row=r, column=4, padx=2)
            emax = ttk.Entry(grid, textvariable=v["vmax"], width=6)
            emax.grid(row=r, column=5, padx=2)
            estep = ttk.Entry(grid, textvariable=v["vstep"], width=6)
            estep.grid(row=r, column=6, padx=2)
            for e in (emin, emax, estep):
                e.bind("<FocusOut>", lambda ev: self.refresh_preview())
            widgets.update(
                row=r, chk=chk, lbl=lbl, mode=mode, vals=vals, emin=emin, emax=emax, estep=estep, lever=lever
            )
            self._lever_rows[lever.key] = widgets
        grid.columnconfigure(3, weight=1)

    def _build_options_section(self, parent) -> None:
        sec = self._section(parent, "Options")
        row = ttk.Frame(sec)
        row.pack(fill="x", padx=6, pady=3)
        ttk.Label(row, text="Run output format:", width=18).pack(side="left")
        # The run always parses JSON into the results grid; only JSON is offered
        # here to avoid an empty grid. CSV/JSON/Markdown are still available via
        # the Results section's Export buttons.
        self.output_format_var.set("json")
        ttk.Combobox(row, textvariable=self.output_format_var, state="readonly", values=["json"], width=10).pack(
            side="left"
        )
        ttk.Label(row, text="  (results are exported as CSV/JSON/Markdown below)", foreground="#666").pack(side="left")

        row2 = ttk.Frame(sec)
        row2.pack(fill="x", padx=6, pady=3)
        ttk.Label(row2, text="Repetitions (-r):", width=18).pack(side="left")
        ttk.Entry(row2, textvariable=self.repetitions_var, width=8).pack(side="left")

        row3 = ttk.Frame(sec)
        row3.pack(fill="x", padx=6, pady=3)
        ttk.Label(row3, text="Extra args:", width=18).pack(side="left")
        e = ttk.Entry(row3, textvariable=self.extra_args_var)
        e.pack(side="left", fill="x", expand=True)
        e.bind("<FocusOut>", lambda ev: self.refresh_preview())

    def _build_preview_section(self, parent) -> None:
        sec = self._section(parent, "Command preview")
        self._preview = tk.Text(sec, height=6, wrap="none")
        self._preview.pack(fill="x", padx=6, pady=3)
        self._preview.configure(state="disabled")
        row = ttk.Frame(sec)
        row.pack(fill="x", padx=6, pady=3)
        ttk.Button(row, text="Refresh preview", command=self.refresh_preview).pack(side="left")
        ttk.Button(row, text="Copy", command=self._copy_preview).pack(side="left", padx=4)
        ttk.Button(row, text="Save .sh…", command=lambda: self._save_script("sh")).pack(side="left", padx=4)
        ttk.Button(row, text="Save .ps1…", command=lambda: self._save_script("ps1")).pack(side="left")

    def _build_run_section(self, parent) -> None:
        sec = self._section(parent, "Run")
        row = ttk.Frame(sec)
        row.pack(fill="x", padx=6, pady=3)
        self._start_btn = ttk.Button(row, text="Start benchmark", command=self.start)
        self._start_btn.pack(side="left")
        self._cancel_btn = ttk.Button(row, text="Cancel", command=self.cancel, state="disabled")
        self._cancel_btn.pack(side="left", padx=4)
        ttk.Checkbutton(row, text="Auto-scroll", variable=self.autoscroll_var).pack(side="left", padx=8)
        ttk.Label(row, textvariable=self.status_var, foreground="#333").pack(side="left", padx=8)

        self._console = tk.Text(sec, height=12, wrap="word")
        self._console.pack(fill="both", expand=True, padx=6, pady=3)
        self._console.tag_configure("stage", foreground="#1a7f37")
        self._console.tag_configure("error", foreground="#cf222e")
        self._console.tag_configure("result", foreground="#0969da")
        self._console.configure(state="disabled")

    def _build_results_section(self, parent) -> None:
        sec = self._section(parent, "Results")
        self._results_tree = ttk.Treeview(sec, show="headings", height=8)
        self._results_tree.pack(fill="both", expand=True, padx=6, pady=3)
        row = ttk.Frame(sec)
        row.pack(fill="x", padx=6, pady=3)
        ttk.Button(row, text="Export CSV…", command=lambda: self._export_results("csv")).pack(side="left")
        ttk.Button(row, text="Export JSON…", command=lambda: self._export_results("json")).pack(side="left", padx=4)
        ttk.Button(row, text="Export Markdown…", command=lambda: self._export_results("markdown")).pack(side="left")
        ttk.Button(row, text="Clear results", command=self._clear_results).pack(side="left", padx=12)

    def _build_configs_section(self, parent) -> None:
        sec = self._section(parent, "Saved sweep configs")
        row = ttk.Frame(sec)
        row.pack(fill="x", padx=6, pady=3)
        self._config_combo = ttk.Combobox(row, textvariable=self.config_name_var, width=26)
        self._config_combo.pack(side="left")
        ttk.Button(row, text="Load", command=self._load_config).pack(side="left", padx=4)
        ttk.Button(row, text="Save", command=self._save_config).pack(side="left")
        ttk.Button(row, text="Delete", command=self._delete_config).pack(side="left", padx=4)
        self._refresh_config_list()

    # ------------------------------------------------------------------ discovery
    def rescan_builds(self) -> None:
        try:
            self._builds = discover_builds(self.launcher)
        except Exception as exc:
            self._builds = []
            print(f"DEBUG: bench discover_builds failed: {exc}", file=sys.stderr)
        self._build_labels = [b.label for b in self._builds]
        if self._build_combo is not None:
            self._build_combo.configure(values=self._build_labels)
            if self._build_labels:
                if self.build_var.get() not in self._build_labels:
                    self.build_var.set(self._build_labels[0])
            else:
                self.build_var.set("")
        self._on_build_changed()

    def _selected_build(self):
        label = self.build_var.get()
        for b in self._builds:
            if b.label == label:
                return b
        return None

    def _on_build_changed(self) -> None:
        build = self._selected_build()
        # Constrain the tool list to what this build actually ships.
        available = build.available_tools() if build else list(ALL_TOOLS)
        if self._tool_combo is not None:
            self._tool_combo.configure(
                values=[_TOOL_LABELS[t] for t in available] or [_TOOL_LABELS[t] for t in ALL_TOOLS]
            )
        current = self.tool_var.get()
        if available and current not in available:
            self.tool_var.set(available[0])
            if self._tool_combo is not None:
                self._tool_combo.set(_TOOL_LABELS[available[0]])
        self._on_tool_changed()

    def _on_tool_label_selected(self) -> None:
        label = self._tool_combo.get() if self._tool_combo else ""
        for tool, lbl in _TOOL_LABELS.items():
            if lbl == label:
                self.tool_var.set(tool)
                break
        self._on_tool_changed()

    def _on_tool_changed(self) -> None:
        tool = self.tool_var.get()
        build = self._selected_build()
        # Show/hide lever rows by applicability to the current tool.
        for key, widgets in self._lever_rows.items():
            lever = widgets["lever"]
            applies = lever.applies_to(tool)
            state = "normal" if applies else "disabled"
            for w in ("chk", "mode", "vals", "emin", "emax", "estep"):
                try:
                    widgets[w].configure(state=state)
                except Exception:
                    pass
            fg = "#000" if applies else "#999"
            try:
                widgets["lbl"].configure(foreground=fg)
            except Exception:
                pass
            self._apply_mode_state(key)
        # Tool availability note
        note = ""
        if build is not None and tool not in build.available_tools():
            note = f"⚠ {_TOOL_LABELS.get(tool, tool)} not found in this build."
        elif tool == TOOL_SWEEP_BENCH:
            note = "llama-sweep-bench sweeps context depth internally (ik_llama)."
        self._tool_note_var().set(note)
        self.refresh_preview()

    def _on_mode_changed(self, key: str) -> None:
        self._apply_mode_state(key)
        self.refresh_preview()

    def _apply_mode_state(self, key: str) -> None:
        widgets = self._lever_rows.get(key)
        if not widgets:
            return
        lever = widgets["lever"]
        applies = lever.applies_to(self.tool_var.get())
        mode = self.lever_vars[key]["mode"].get()
        if not applies:
            for w in ("vals", "emin", "emax", "estep"):
                try:
                    widgets[w].configure(state="disabled")
                except Exception:
                    pass
            return
        list_state = "normal" if mode == "list" else "disabled"
        range_state = "normal" if mode == "range" else "disabled"
        try:
            widgets["vals"].configure(state=list_state)
            for w in ("emin", "emax", "estep"):
                widgets[w].configure(state=range_state)
        except Exception:
            pass

    # ------------------------------------------------------------------ axes
    def _collect_axes(self) -> list[Axis]:
        """Build sweep axes from the included lever rows. Raises SweepError."""
        axes: list[Axis] = []
        for lever in LEVERS:
            v = self.lever_vars[lever.key]
            if not v["include"].get():
                continue
            mode = v["mode"].get()
            if mode == "range":
                try:
                    vmin = float(v["vmin"].get())
                    vmax = float(v["vmax"].get())
                    vstep = float(v["vstep"].get())
                except ValueError as exc:
                    raise SweepError(f"{lever.label}: min/max/step must be numeric") from exc
                values = expand_range(vmin, vmax, vstep, lever.kind)
            else:
                values = parse_list(v["values"].get(), lever.kind)
            if not values:
                raise SweepError(f"{lever.label} is ticked but has no values.")
            axes.append(Axis(lever.key, values))
        return axes

    def _current_exe(self) -> str | None:
        build = self._selected_build()
        if build is None:
            return None
        return build.tool_path(self.tool_var.get())

    def _current_backend(self) -> str:
        """Backend of the selected build (drives flash-attn handling in the
        matrix), defaulting to ``llama.cpp`` when no build is selected."""
        build = self._selected_build()
        return getattr(build, "backend", "llama.cpp") if build is not None else "llama.cpp"

    def _repetitions(self) -> int | None:
        raw = self.repetitions_var.get().strip()
        if not raw:
            return None
        try:
            n = int(raw)
            return n if n > 0 else None
        except ValueError:
            return None

    def _build_command_list(self) -> tuple[list[list[str]], list[dict]]:
        """Return (commands, combos) for the current settings, exe-agnostic
        placeholder allowed for preview when no build is selected."""
        tool = self.tool_var.get()
        exe = self._current_exe() or tool  # placeholder for preview
        model = self.model_var.get().strip()
        axes = self._collect_axes()
        pairs = build_commands(
            tool,
            exe,
            model,
            axes,
            backend=self._current_backend(),
            output_format=self.output_format_var.get(),
            repetitions=self._repetitions(),
            extra_args=self.extra_args_var.get(),
        )
        return [c for c, _ in pairs], [combo for _, combo in pairs]

    # ------------------------------------------------------------------ preview
    def refresh_preview(self) -> None:
        if self._preview is None:
            return
        tool = self.tool_var.get()
        lines: list[str] = []
        try:
            axes = self._collect_axes()
        except SweepError as exc:
            self._set_preview(f"(incomplete) {exc}")
            return
        if not axes:
            self._set_preview("(no parameters selected — tick at least one to sweep)")
            return
        _applicable, ignored = split_axes_for_tool(axes, tool)
        size = matrix_size(axes, tool)
        model = self.model_var.get().strip() or "<model.gguf>"
        exe = self._current_exe() or f"<{tool}>"
        try:
            pairs = build_commands(
                tool,
                exe,
                model,
                axes,
                backend=self._current_backend(),
                output_format=self.output_format_var.get(),
                repetitions=self._repetitions(),
                extra_args=self.extra_args_var.get(),
            )
        except SweepError as exc:
            self._set_preview(f"(incomplete) {exc}")
            return
        if tool == TOOL_LLAMA_BENCH:
            lines.append(f"# 1 invocation, {size} matrix row(s)")
        else:
            lines.append(f"# {len(pairs)} invocation(s) (cartesian product)")
        if ignored:
            names = ", ".join(a.lever.label for a in ignored)
            lines.append(f"# ignored (n/a for {tool}): {names}")
        for cmd, _combo in pairs[:24]:
            lines.append(" ".join(shlex.quote(t) for t in cmd))
        if len(pairs) > 24:
            lines.append(f"# … {len(pairs) - 24} more")
        self._set_preview("\n".join(lines))

    def _set_preview(self, text: str) -> None:
        if self._preview is None:
            return
        self._preview.configure(state="normal")
        self._preview.delete("1.0", "end")
        self._preview.insert("1.0", text)
        self._preview.configure(state="disabled")

    def _copy_preview(self) -> None:
        if self._preview is None:
            return
        text = self._preview.get("1.0", "end").strip()
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
        except Exception:
            pass

    # ------------------------------------------------------------------ seed
    def seed_from_config(self) -> None:
        # Model
        model = self._launcher_model_path()
        if model:
            self.model_var.set(model)
        # Backend → tool + matching build if available
        try:
            backend = self.launcher.backend_selection.get()
        except Exception:
            backend = ""
        if backend == "ik_llama":
            for b in self._builds:
                if b.backend == "ik_llama" and TOOL_SWEEP_BENCH in b.tools:
                    self.build_var.set(b.label)
                    self._on_build_changed()
                    break
        # Seed each lever's list value from the launcher's live setting. Any
        # lever that receives a non-empty value is also *ticked* (include=True)
        # so the seeded baseline actually runs on Start; the user can untick.
        seeded = 0
        for lever in LEVERS:
            if not lever.seed_attr:
                continue
            var = getattr(self.launcher, lever.seed_attr, None)
            if var is None:
                continue
            try:
                raw = var.get()
            except Exception:
                continue
            v = self.lever_vars[lever.key]
            if lever.kind == "fa":
                v["values"].set("on" if bool(raw) else "off")
            else:
                text = str(raw).strip()
                if text == "":
                    continue
                v["values"].set(text)
            v["mode"].set("list")
            v["include"].set(True)
            seeded += 1
            self._apply_mode_state(lever.key)
        self.refresh_preview()
        if seeded:
            self.status_var.set(f"Seeded {seeded} parameter(s) from the current configuration (ticked to run).")
        else:
            self.status_var.set("Seeded parameters from the current configuration.")

    def _browse_model(self) -> None:
        initial = ""
        try:
            dirs = getattr(self.launcher, "model_dirs", [])
            if dirs:
                initial = str(dirs[-1])
        except Exception:
            initial = ""
        path = filedialog.askopenfilename(
            title="Select model",
            initialdir=initial or str(Path.home()),
            filetypes=[("GGUF models", "*.gguf"), ("All files", "*.*")],
        )
        if path:
            self.model_var.set(path)
            self.refresh_preview()

    # ------------------------------------------------------------------ run
    def start(self) -> None:
        if self.runner.is_running:
            return
        exe = self._current_exe()
        if not exe:
            messagebox.showerror("Benchmark", "No benchmark tool found in the selected build.")
            return
        model = self.model_var.get().strip()
        if not model:
            messagebox.showerror("Benchmark", "Select a model first.")
            return
        if not Path(model).is_file():
            if not messagebox.askyesno("Benchmark", f"Model path does not exist:\n{model}\n\nRun anyway?"):
                return
        try:
            axes = self._collect_axes()
        except SweepError as exc:
            messagebox.showerror("Benchmark", str(exc))
            return
        if not axes:
            messagebox.showerror("Benchmark", "Tick at least one parameter to sweep.")
            return
        tool = self.tool_var.get()
        try:
            pairs = build_commands(
                tool,
                exe,
                model,
                axes,
                backend=self._current_backend(),
                output_format=self.output_format_var.get(),
                repetitions=self._repetitions(),
                extra_args=self.extra_args_var.get(),
            )
        except SweepError as exc:
            messagebox.showerror("Benchmark", str(exc))
            return
        steps = [BenchStep(cmd=cmd, combo=combo, label=self._combo_label(combo)) for cmd, combo in pairs]
        cwd = str(Path(exe).parent)
        plan = BenchPlan(tool=tool, steps=steps, cwd=cwd)

        self._clear_console()
        self._append_console(f"Running {len(steps)} benchmark invocation(s) with {tool}…", tag="stage")
        if not self.runner.start(plan):
            self._append_console("A benchmark is already running.", tag="error")
            return
        self._set_running(True)
        self._poll_runner()

    def _combo_label(self, combo: dict[str, str]) -> str:
        if not combo:
            return ""
        return ", ".join(f"{k}={v}" for k, v in combo.items())

    def cancel(self) -> None:
        if self.runner.is_running:
            self.runner.cancel()
            self.status_var.set("Cancelling…")

    def _set_running(self, running: bool) -> None:
        if self._start_btn is not None:
            self._start_btn.configure(state="disabled" if running else "normal")
        if self._cancel_btn is not None:
            self._cancel_btn.configure(state="normal" if running else "disabled")

    def _poll_runner(self) -> None:
        import queue as _queue

        drained = 0
        had_events = False
        try:
            while drained < RUNNER_MAX_EVENTS_PER_POLL:
                kind, payload = self.runner.events.get_nowait()
                drained += 1
                had_events = True
                # One misbehaving handler (e.g. a Treeview/Tk error while
                # rebuilding the results grid) must never escape the ``after``
                # callback: if it did, polling would stop, the worker's blocking
                # queue would fill and park, and ``is_running`` would stay True
                # forever (Start stuck disabled, Cancel a no-op). Swallow and log.
                try:
                    self._handle_event(kind, payload)
                except tk.TclError:
                    # A widget was torn down under us; stop polling cleanly.
                    self._poll_after_id = None
                    return
                except Exception:
                    print(f"BenchmarkTab: error handling event {kind!r}:", file=sys.stderr)
                    traceback.print_exc()
                if kind in (EVENT_DONE, EVENT_CANCELLED, EVENT_ERROR):
                    self._set_running(False)
                    self._poll_after_id = None
                    return
        except _queue.Empty:
            pass
        except tk.TclError:
            # Root/widgets destroyed while draining; stop rescheduling.
            self._poll_after_id = None
            return
        delay = RUNNER_CATCHUP_POLL_MS if (had_events and drained >= RUNNER_MAX_EVENTS_PER_POLL) else RUNNER_POLL_MS
        try:
            self._poll_after_id = self.root.after(delay, self._poll_runner)
        except tk.TclError:
            # Root is gone; nothing left to poll.
            self._poll_after_id = None

    def _handle_event(self, kind: str, payload) -> None:
        if kind == EVENT_LINE:
            self._append_console(str(payload))
        elif kind == EVENT_STEP_START:
            info: StepInfo = payload
            label = f" [{self._combo_label(info.combo)}]" if info.combo else ""
            self._append_console(f"\n══ run {info.index + 1}/{info.total}{label} ══", tag="stage")
            self.status_var.set(f"Running {info.index + 1}/{info.total}…")
        elif kind == EVENT_STEP_RESULT:
            self._ingest_result(payload)
        elif kind == EVENT_DONE:
            failed = int(payload) if payload else 0
            msg = "Done." if not failed else f"Done with {failed} failed run(s)."
            self.status_var.set(msg)
            self._append_console(msg, tag="stage" if not failed else "error")
        elif kind == EVENT_CANCELLED:
            self.status_var.set("Cancelled.")
            self._append_console("Cancelled.", tag="error")
        elif kind == EVENT_ERROR:
            self.status_var.set("Error.")
            self._append_console(f"ERROR: {payload}", tag="error")

    def _ingest_result(self, result: StepResult) -> None:
        # Echo captured stdout to the console so the user sees the raw output.
        text = (result.stdout or "").strip()
        if text:
            for line in text.splitlines():
                self._append_console(line, tag="result")
        if result.rc != 0:
            self._append_console(f"(run exited with code {result.rc})", tag="error")
        # Parse into rows.
        if result.tool == TOOL_LLAMA_BENCH:
            rows = results.parse_llama_bench_json(result.stdout)
        else:
            rows = results.parse_sweep_bench_table(result.stdout, result.combo)
        if rows:
            self._result_rows.extend(rows)
            self._refresh_results_grid()

    # ------------------------------------------------------------------ console
    def _append_console(self, text: str, *, tag: str | None = None) -> None:
        if self._console is None:
            return
        self._console_buffer.append((text, tag))
        if len(self._console_buffer) > CONSOLE_MAX_LINES:
            self._console_buffer = self._console_buffer[-CONSOLE_MAX_LINES:]
        self._console.configure(state="normal")
        self._console.insert("end", text + "\n", (tag,) if tag else ())
        # Trim widget content to bound memory.
        line_count = int(self._console.index("end-1c").split(".")[0])
        if line_count > CONSOLE_MAX_LINES:
            self._console.delete("1.0", f"{line_count - CONSOLE_MAX_LINES}.0")
        self._console.configure(state="disabled")
        if self.autoscroll_var.get():
            self._console.see("end")

    def _clear_console(self) -> None:
        self._console_buffer.clear()
        if self._console is not None:
            self._console.configure(state="normal")
            self._console.delete("1.0", "end")
            self._console.configure(state="disabled")

    # ------------------------------------------------------------------ results grid
    def _refresh_results_grid(self) -> None:
        tree = self._results_tree
        if tree is None:
            return
        cols = results.collect_columns(self._result_rows)
        tree.configure(columns=cols)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=max(60, min(200, len(c) * 10)), anchor="w", stretch=False)
        tree.delete(*tree.get_children())
        for row in self._result_rows:
            tree.insert("", "end", values=[row.get(c) for c in cols])

    def _clear_results(self) -> None:
        self._result_rows.clear()
        self._refresh_results_grid()

    def _export_results(self, fmt: str) -> None:
        if not self._result_rows:
            messagebox.showinfo("Export", "No results to export yet.")
            return
        _exporter, ext = results.EXPORTERS[fmt]
        path = filedialog.asksaveasfilename(
            title=f"Export results as {fmt}",
            defaultextension=ext,
            filetypes=[(fmt.upper(), f"*{ext}"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            Path(path).write_text(results.export(self._result_rows, fmt), encoding="utf-8")
            self.status_var.set(f"Exported {len(self._result_rows)} row(s) to {path}")
        except Exception as exc:
            messagebox.showerror("Export", f"Failed to write file:\n{exc}")

    # ------------------------------------------------------------------ scripts
    def _save_script(self, fmt: str) -> None:
        try:
            commands, _combos = self._build_command_list()
        except SweepError as exc:
            messagebox.showerror("Save script", str(exc))
            return
        if not commands:
            messagebox.showinfo("Save script", "Nothing to save — select parameters first.")
            return
        if self._current_exe() is None:
            if not messagebox.askyesno(
                "Save script",
                "No build/tool is selected, so the script will contain a placeholder " "executable name. Save anyway?",
            ):
                return
        ext = ".sh" if fmt == "sh" else ".ps1"
        path = filedialog.asksaveasfilename(
            title=f"Save benchmark {fmt} script",
            defaultextension=ext,
            filetypes=[(fmt, f"*{ext}"), ("All files", "*.*")],
        )
        if not path:
            return
        header = f"Benchmark: {self.tool_var.get()} — {len(commands)} invocation(s)"
        try:
            script = bench_script.render(commands, fmt, header=header)
            Path(path).write_text(script, encoding="utf-8")
            if fmt == "sh":
                try:
                    Path(path).chmod(0o755)
                except OSError:
                    pass
            self.status_var.set(f"Saved {fmt} script to {path}")
        except Exception as exc:
            messagebox.showerror("Save script", f"Failed to write script:\n{exc}")

    # ------------------------------------------------------------------ configs
    def _refresh_config_list(self) -> None:
        if self._config_combo is None:
            return
        try:
            names = self.store.list_names()
        except Exception:
            names = []
        self._config_combo.configure(values=names)

    def _current_config(self, name: str) -> BenchConfig:
        build = self._selected_build()
        axes_spec: dict[str, dict] = {}
        for lever in LEVERS:
            v = self.lever_vars[lever.key]
            if not v["include"].get():
                continue
            axes_spec[lever.key] = {
                "enabled": True,
                "mode": v["mode"].get(),
                "raw": v["values"].get(),
                "min": self._safe_float(v["vmin"].get()),
                "max": self._safe_float(v["vmax"].get()),
                "step": self._safe_float(v["vstep"].get(), 1.0),
            }
        return BenchConfig(
            name=name,
            tool=self.tool_var.get(),
            backend=(build.backend if build else "llama.cpp"),
            build_root=(build.root_dir if build else ""),
            model_path=self.model_var.get().strip(),
            axes=axes_spec,
            output_format=self.output_format_var.get(),
            repetitions=self._repetitions() or 0,
            extra_args=self.extra_args_var.get(),
        )

    @staticmethod
    def _safe_float(raw: str, default: float = 0.0) -> float:
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default

    def _save_config(self) -> None:
        name = self.config_name_var.get().strip()
        if not name:
            messagebox.showerror("Save config", "Enter a name for the sweep config.")
            return
        if self.store.save(self._current_config(name)):
            self.status_var.set(f"Saved sweep config '{name}'.")
            self._refresh_config_list()
        else:
            messagebox.showerror("Save config", "Failed to save (see console/stderr).")

    def _delete_config(self) -> None:
        name = self.config_name_var.get().strip()
        if not name:
            return
        if not messagebox.askyesno("Delete config", f"Delete sweep config '{name}'?"):
            return
        if self.store.delete(name):
            self.status_var.set(f"Deleted '{name}'.")
            self.config_name_var.set("")
            self._refresh_config_list()

    def _load_config(self) -> None:
        name = self.config_name_var.get().strip()
        if not name:
            return
        cfg = self.store.get(name)
        if cfg is None:
            messagebox.showerror("Load config", f"No sweep config named '{name}'.")
            return
        saved_tool = cfg.tool

        def _root_matches(b) -> bool:
            if not cfg.build_root:
                return False
            try:
                return str(Path(b.root_dir)) == str(Path(cfg.build_root)) or b.root_dir == cfg.build_root
            except Exception:
                return b.root_dir == cfg.build_root

        def _offers_tool(b) -> bool:
            try:
                return saved_tool in b.available_tools()
            except Exception:
                return False

        # Pick a build that actually offers the saved tool so the trailing
        # _on_build_changed() doesn't silently revert the tool. Prefer the
        # config's own build_root; otherwise any build providing the tool.
        target = next((b for b in self._builds if _root_matches(b) and _offers_tool(b)), None)
        if target is None:
            target = next((b for b in self._builds if _offers_tool(b)), None)
        if target is not None:
            self.build_var.set(target.label)
        elif self._builds:
            # No detected build ships this tool — warn instead of silently
            # switching the tool out from under the user on _on_build_changed().
            messagebox.showwarning(
                "Load config",
                f"The tool '{_TOOL_LABELS.get(saved_tool, saved_tool)}' from config "
                f"'{name}' isn't available in any detected build. The tool selection "
                "may change to one this build provides.",
            )
        self.tool_var.set(saved_tool)
        if self._tool_combo is not None:
            self._tool_combo.set(_TOOL_LABELS.get(saved_tool, saved_tool))
        if cfg.model_path:
            self.model_var.set(cfg.model_path)
        # Run output is JSON-only (see Options); ignore any legacy csv/markdown.
        self.output_format_var.set("json")
        self.repetitions_var.set(str(cfg.repetitions) if cfg.repetitions else "")
        self.extra_args_var.set(cfg.extra_args)
        # Reset all levers, then apply saved axes.
        for lever in LEVERS:
            v = self.lever_vars[lever.key]
            v["include"].set(False)
        for key, spec in cfg.axes.items():
            if key not in self.lever_vars:
                continue
            v = self.lever_vars[key]
            v["include"].set(bool(spec.get("enabled", True)))
            v["mode"].set(spec.get("mode", "list"))
            v["values"].set(spec.get("raw", ""))
            v["vmin"].set(self._fmt(spec.get("min", 0)))
            v["vmax"].set(self._fmt(spec.get("max", 0)))
            v["vstep"].set(self._fmt(spec.get("step", 1)))
        self._on_build_changed()
        self.status_var.set(f"Loaded sweep config '{name}'.")
        self.refresh_preview()

    @staticmethod
    def _fmt(value) -> str:
        try:
            f = float(value)
            return str(int(f)) if f.is_integer() else str(f)
        except (TypeError, ValueError):
            return str(value)
