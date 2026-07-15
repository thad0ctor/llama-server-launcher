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
    KIND_INT,
    KIND_STR,
    LEVERS,
    LEVERS_BY_KEY,
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

# MTP / speculative-decoding levers get their own labelled subsection below the
# main sweep grid (they apply only to ik_llama's llama-sweep-bench). Their rows
# are created into ``_lever_rows`` like any other lever, so the shared
# applicability-driven visibility pass hides them off sweep-bench automatically.
MTP_LEVER_KEYS = ("mtp", "draft_max", "draft_min", "draft_p_min", "mtprot")

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
        # Backend is the source of truth for ik-only UI + matrix rendering. Seed
        # from the launcher's live backend selection and keep the two in sync.
        seed_backend = "llama.cpp"
        try:
            seed_backend = self.launcher.backend_selection.get() or "llama.cpp"
        except Exception:
            pass
        self.backend_var = tk.StringVar(value=seed_backend)
        # Re-entrancy guard for the two-way backend sync (mirrors build_tab.py's
        # ``_syncing_backend_selection``).
        self._syncing_backend = False
        # Mirror the launcher's backend selection into this tab (build_tab.py
        # does the same for its own backend var). Guarded so a launcher without a
        # usable backend var simply skips the two-way sync.
        try:
            self.launcher.backend_selection.trace_add("write", self._on_launcher_backend_changed)
        except Exception:
            pass

        self.build_var = tk.StringVar()
        # ``tool_var`` always holds the CANONICAL tool id; ``_tool_display_var``
        # is what the combobox shows (a human label). Keeping them separate
        # stops the combobox from writing its display label back into the
        # canonical var (which broke tool selection when loading a config).
        self.tool_var = tk.StringVar(value=TOOL_LLAMA_BENCH)
        self._tool_display_var = tk.StringVar(value=_TOOL_LABELS[TOOL_LLAMA_BENCH])
        self.model_var = tk.StringVar(value=self._launcher_model_path())
        self.output_format_var = tk.StringVar(value="json")
        self.repetitions_var = tk.StringVar(value="")
        self.config_name_var = tk.StringVar()
        self.autoscroll_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Idle.")

        # Dynamic multi-row editors (rebuilt into the UI on add/remove/load).
        # Each Extra-args row is a free-form string appended to EVERY command;
        # each custom-flag row is a user-defined sweep axis. Both are lists of
        # dict-of-Tk-vars so they round-trip through save/load like lever rows.
        self._extra_args_rows: list[dict[str, tk.Variable]] = []
        self._custom_axis_rows: list[dict[str, tk.Variable]] = []
        self._extra_args_container: ttk.Frame | None = None
        self._custom_axis_container: ttk.Frame | None = None

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
        self._repetitions_entry: ttk.Entry | None = None
        self._repetitions_note: ttk.Label | None = None
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
        self._build_mtp_section(body)
        self._build_custom_flags_section(body)
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

        brow = ttk.Frame(sec)
        brow.pack(fill="x", padx=6, pady=3)
        ttk.Label(brow, text="Backend:", width=12).pack(side="left")
        for label, value in (("llama.cpp", "llama.cpp"), ("ik_llama", "ik_llama")):
            ttk.Radiobutton(
                brow,
                text=label,
                value=value,
                variable=self.backend_var,
                command=self._on_backend_radio_changed,
            ).pack(side="left", padx=(0, 12))

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
            textvariable=self._tool_display_var,
            state="readonly",
            values=[_TOOL_LABELS[t] for t in ALL_TOOLS],
        )
        self._tool_combo.pack(side="left", fill="x", expand=True)
        self._tool_combo.bind("<<ComboboxSelected>>", lambda e: self._on_tool_label_selected())

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

        # The MTP / speculative levers live in their own subsection (built next),
        # so skip them here even though they share the LEVERS catalogue.
        r = 1
        for lever in LEVERS:
            if lever.key in MTP_LEVER_KEYS:
                continue
            self._build_lever_row(grid, r, lever)
            r += 1
        grid.columnconfigure(3, weight=1)

    def _build_mtp_section(self, parent) -> None:
        sec = self._section(parent, "MTP / speculative (llama-sweep-bench only)")
        ttk.Label(
            sec,
            text="MTP uses the model's embedded head (no draft model). " "Sweep MTP enable = 0,1 to measure speedup.",
            foreground="#666",
            wraplength=680,
            justify="left",
        ).pack(fill="x", padx=6, pady=(2, 4))

        grid = ttk.Frame(sec)
        grid.pack(fill="x", padx=6, pady=2)
        headers = ["", "Parameter", "Mode", "List values", "Min", "Max", "Step"]
        for c, text in enumerate(headers):
            ttk.Label(grid, text=text, foreground="#444").grid(row=0, column=c, sticky="w", padx=3)
        for r, key in enumerate(MTP_LEVER_KEYS, start=1):
            self._build_lever_row(grid, r, LEVERS_BY_KEY[key])
        grid.columnconfigure(3, weight=1)

    def _build_lever_row(self, grid, r: int, lever) -> None:
        """Create one lever row's widgets into ``grid`` at row ``r``.

        Shared by the main sweep grid and the MTP subsection so both render
        identical rows and register into ``self._lever_rows`` keyed by lever key
        (which is all the applicability-driven visibility pass needs).
        """
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
        widgets.update(row=r, chk=chk, lbl=lbl, mode=mode, vals=vals, emin=emin, emax=emax, estep=estep, lever=lever)
        self._lever_rows[lever.key] = widgets

    def _build_custom_flags_section(self, parent) -> None:
        sec = self._section(parent, "Custom sweep flags")
        ttk.Label(
            sec,
            text="Sweep an arbitrary flag not listed above (e.g. -ot, --override-tensor, --cache-reuse). "
            "Tick to include; use 'list' for comma-separated values or 'range' for numeric min/max/step. "
            "Each value becomes its own matrix combo on llama-sweep-bench; on llama-bench the values are "
            "passed as one comma-list, so a custom flag only sweeps there if it accepts comma-separated values.",
            foreground="#666",
            wraplength=680,
            justify="left",
        ).pack(fill="x", padx=6, pady=(2, 4))

        grid = ttk.Frame(sec)
        grid.pack(fill="x", padx=6, pady=2)
        headers = ["", "Flag", "Mode", "List values", "Min", "Max", "Step", ""]
        for c, text in enumerate(headers):
            ttk.Label(grid, text=text, foreground="#444").grid(row=0, column=c, sticky="w", padx=3)
        grid.columnconfigure(3, weight=1)
        # Rows are re-gridded into this container on add/remove/load.
        self._custom_axis_container = grid
        ttk.Button(sec, text="Add custom flag", command=self._add_custom_axis_row).pack(
            side="left", padx=6, pady=(0, 4)
        )
        self._render_custom_axis_rows()

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
        self._repetitions_entry = ttk.Entry(row2, textvariable=self.repetitions_var, width=8)
        self._repetitions_entry.pack(side="left")
        # llama-sweep-bench has no -r flag; the tool-change handler greys this
        # field (and shows the note) so a typed value isn't silently dropped.
        self._repetitions_note = ttk.Label(row2, text="", foreground="#666")
        self._repetitions_note.pack(side="left", padx=6)

        # Multi-row Extra-args editor. Every non-empty row is shlex-split and
        # appended (in order) to EVERY command in the matrix.
        ttk.Label(sec, text="Extra args (appended to every run):", foreground="#444").pack(
            anchor="w", padx=6, pady=(3, 0)
        )
        self._extra_args_container = ttk.Frame(sec)
        self._extra_args_container.pack(fill="x", padx=6, pady=1)
        ttk.Button(sec, text="Add extra-args row", command=self._add_extra_args_row).pack(
            side="left", padx=6, pady=(0, 4)
        )
        if not self._extra_args_rows:
            # Seed a single empty row so the editor is never blank on first open.
            self._extra_args_rows.append({"value": tk.StringVar(value="")})
        self._render_extra_args_rows()

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

    def _set_tool(self, tool: str) -> None:
        """Set the canonical tool id AND sync the combobox's display label.

        Routing every programmatic tool change through here keeps ``tool_var``
        (canonical id) and ``_tool_display_var`` (human label) consistent
        without the combobox ever writing its label back into ``tool_var``.
        """
        self.tool_var.set(tool)
        self._tool_display_var.set(_TOOL_LABELS.get(tool, tool))

    def _on_build_changed(self) -> None:
        build = self._selected_build()
        # Keep the backend selection in lockstep with the chosen build BEFORE
        # any tool/lever refresh: _current_backend() gates ik-only levers and
        # flash-attn handling, so a manually-picked build whose backend differs
        # from the radio (e.g. an ik_llama build while the radio says llama.cpp)
        # must flip the backend first, or the UI would render/gate flags for the
        # stale backend. Guarded by _syncing_backend so the launcher write-back
        # (which traces back into this tab) can't recurse.
        if build is not None and not self._syncing_backend:
            build_backend = getattr(build, "backend", "")
            if build_backend in ("llama.cpp", "ik_llama") and build_backend != self.backend_var.get():
                self._syncing_backend = True
                try:
                    self.backend_var.set(build_backend)
                    try:
                        self.launcher.backend_selection.set(build_backend)
                    except Exception:
                        pass
                finally:
                    self._syncing_backend = False
        # Constrain the tool list to what this build actually ships.
        available = build.available_tools() if build else list(ALL_TOOLS)
        if self._tool_combo is not None:
            self._tool_combo.configure(
                values=[_TOOL_LABELS[t] for t in available] or [_TOOL_LABELS[t] for t in ALL_TOOLS]
            )
        current = self.tool_var.get()
        if available and current not in available:
            self._set_tool(available[0])
        self._on_tool_changed()

    def _on_tool_label_selected(self) -> None:
        # The combobox already wrote the selected label into ``_tool_display_var``;
        # map it back to the canonical id in ``tool_var``.
        label = self._tool_display_var.get()
        for tool, lbl in _TOOL_LABELS.items():
            if lbl == label:
                self.tool_var.set(tool)
                break
        self._on_tool_changed()

    @staticmethod
    def _is_ik_lever(lever) -> bool:
        """True for levers that exist only in ik_llama's tool build."""
        return "llama.cpp" not in lever.backends

    @staticmethod
    def _set_lever_row_visible(widgets: dict, visible: bool) -> None:
        """Grid-show or grid-remove a whole lever row (position preserved)."""
        for w in ("chk", "lbl", "mode", "vals", "emin", "emax", "estep"):
            try:
                if visible:
                    widgets[w].grid()
                else:
                    widgets[w].grid_remove()
            except Exception:
                pass

    # ------------------------------------------------------------------ backend
    def _on_backend_radio_changed(self) -> None:
        """Fired by the Backend radio: push the choice to the launcher, then
        refresh build/tool/ik state and the preview."""
        self._sync_launcher_backend_from_tab()
        self._on_backend_changed()

    def _on_launcher_backend_changed(self, *_a) -> None:
        """Trace on ``launcher.backend_selection``: mirror it into this tab.

        Guarded by ``_syncing_backend`` so the write-back path can't recurse.
        """
        if self._syncing_backend:
            return
        try:
            new_backend = self.launcher.backend_selection.get()
        except Exception:
            return
        if new_backend and new_backend != self.backend_var.get():
            self._syncing_backend = True
            try:
                self.backend_var.set(new_backend)
            finally:
                self._syncing_backend = False
            self._on_backend_changed()

    def _sync_launcher_backend_from_tab(self) -> None:
        """Write this tab's backend selection back to the launcher var."""
        if self._syncing_backend:
            return
        backend = self.backend_var.get() or "llama.cpp"
        try:
            if self.launcher.backend_selection.get() == backend:
                return
        except Exception:
            return
        self._syncing_backend = True
        try:
            self.launcher.backend_selection.set(backend)
        except Exception:
            pass
        finally:
            self._syncing_backend = False

    def _on_backend_changed(self) -> None:
        """Backend switched: auto-pick a matching build, refresh the tool list
        (which re-evaluates ik-only lever-row visibility) and the preview."""
        backend = self.backend_var.get()
        # Prefer a discovered build of this backend (if any); _on_build_changed
        # then refreshes the tool list and lever-row applicability.
        self._auto_select_build_for_backend(backend)
        self.refresh_preview()

    def _auto_select_build_for_backend(self, backend: str) -> None:
        """Select the first discovered build matching ``backend`` (if any), then
        re-run the build-changed cascade so the tool list + lever rows update."""
        for b in self._builds:
            if getattr(b, "backend", "") == backend:
                if self.build_var.get() != b.label:
                    self.build_var.set(b.label)
                self._on_build_changed()
                return
        # No build of this backend — still refresh tool/lever state so ik levers
        # appear/disappear even without a concrete build selected.
        self._on_build_changed()

    def _on_tool_changed(self) -> None:
        tool = self.tool_var.get()
        backend = self._current_backend()
        build = self._selected_build()
        # Row visibility is purely applicability-driven: a lever row is shown iff
        # it applies to the CURRENT tool AND backend. This single rule hides the
        # ik_llama-only levers on a llama.cpp backend, hides llama-bench-only ik
        # levers (e.g. -fmoe) when the tool is llama-sweep-bench, and hides the
        # MTP levers on llama-bench — all from the levers' own tools/backends.
        for key, widgets in self._lever_rows.items():
            lever = widgets["lever"]
            applies = lever.applies_to(tool, backend)
            self._set_lever_row_visible(widgets, applies)
            if not applies:
                continue
            # A visible (applicable) row is fully enabled; the mode toggle then
            # governs which of the list vs min/max/step entries stay active.
            for w in ("chk", "mode", "vals", "emin", "emax", "estep"):
                try:
                    widgets[w].configure(state="normal")
                except Exception:
                    pass
            try:
                widgets["lbl"].configure(foreground="#000")
            except Exception:
                pass
            self._apply_mode_state(key)
        # Repetitions (-r) exists only on llama-bench: grey the field for
        # llama-sweep-bench so a typed value isn't silently ignored.
        if self._repetitions_entry is not None:
            if tool == TOOL_SWEEP_BENCH:
                try:
                    self._repetitions_entry.configure(state="disabled")
                except Exception:
                    pass
                if self._repetitions_note is not None:
                    self._repetitions_note.configure(text="(n/a for llama-sweep-bench)")
            else:
                try:
                    self._repetitions_entry.configure(state="normal")
                except Exception:
                    pass
                if self._repetitions_note is not None:
                    self._repetitions_note.configure(text="")
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
        applies = lever.applies_to(self.tool_var.get(), self._current_backend())
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

    # ------------------------------------------------------------------ extra-args rows
    def _render_extra_args_rows(self) -> None:
        """Re-grid the Extra-args editor from ``self._extra_args_rows``."""
        c = self._extra_args_container
        if c is None:
            return
        for w in c.winfo_children():
            w.destroy()
        for idx, row in enumerate(self._extra_args_rows):
            r = ttk.Frame(c)
            r.pack(fill="x", pady=1)
            ttk.Label(r, text=f"{idx + 1}.", width=3).pack(side="left")
            e = ttk.Entry(r, textvariable=row["value"])
            e.pack(side="left", fill="x", expand=True)
            e.bind("<FocusOut>", lambda ev: self.refresh_preview())
            ttk.Button(r, text="Remove", command=lambda i=idx: self._remove_extra_args_row(i)).pack(side="left", padx=4)

    def _add_extra_args_row(self) -> None:
        self._extra_args_rows.append({"value": tk.StringVar(value="")})
        self._render_extra_args_rows()
        self.refresh_preview()

    def _remove_extra_args_row(self, idx: int) -> None:
        if 0 <= idx < len(self._extra_args_rows):
            self._extra_args_rows.pop(idx)
        if not self._extra_args_rows:
            # Never leave the editor with zero rows — keep one empty row.
            self._extra_args_rows.append({"value": tk.StringVar(value="")})
        self._render_extra_args_rows()
        self.refresh_preview()

    def _set_extra_args_rows(self, values: list[str]) -> None:
        """Rebuild the Extra-args editor from persisted row strings."""
        self._extra_args_rows = [{"value": tk.StringVar(value=str(v))} for v in values]
        if not self._extra_args_rows:
            self._extra_args_rows.append({"value": tk.StringVar(value="")})
        self._render_extra_args_rows()

    def _current_extra_args(self) -> list[str]:
        """Free-form Extra-args rows to append to every command, empties dropped.

        Returned as a list so the matrix builder shlex-splits each row
        independently and concatenates them in order.
        """
        rows: list[str] = []
        for row in self._extra_args_rows:
            text = str(row["value"].get()).strip()
            if text:
                rows.append(text)
        return rows

    # ------------------------------------------------------------------ custom-flag rows
    def _make_custom_axis_row(
        self,
        *,
        flag: str = "",
        include: bool = True,
        mode: str = "list",
        values: str = "",
        vmin: str = "0",
        vmax: str = "0",
        vstep: str = "1",
    ) -> dict[str, tk.Variable]:
        return {
            "include": tk.BooleanVar(value=include),
            "flag": tk.StringVar(value=flag),
            "mode": tk.StringVar(value=mode if mode in ("list", "range") else "list"),
            "values": tk.StringVar(value=values),
            "vmin": tk.StringVar(value=vmin),
            "vmax": tk.StringVar(value=vmax),
            "vstep": tk.StringVar(value=vstep),
        }

    def _render_custom_axis_rows(self) -> None:
        """Re-grid the custom-flag editor from ``self._custom_axis_rows``."""
        grid = self._custom_axis_container
        if grid is None:
            return
        # Wipe every widget below the header row (row 0) and re-grid.
        for w in grid.grid_slaves():
            try:
                if int(w.grid_info().get("row", 0)) > 0:
                    w.destroy()
            except Exception:
                pass
        for i, row in enumerate(self._custom_axis_rows, start=1):
            ttk.Checkbutton(grid, variable=row["include"], command=self.refresh_preview).grid(
                row=i, column=0, sticky="w", padx=3
            )
            fe = ttk.Entry(grid, textvariable=row["flag"], width=18)
            fe.grid(row=i, column=1, sticky="we", padx=3)
            fe.bind("<FocusOut>", lambda ev: self.refresh_preview())
            me = ttk.Combobox(grid, textvariable=row["mode"], state="readonly", values=["list", "range"], width=6)
            me.grid(row=i, column=2, sticky="w", padx=3)
            me.bind("<<ComboboxSelected>>", lambda ev: self.refresh_preview())
            ve = ttk.Entry(grid, textvariable=row["values"], width=22)
            ve.grid(row=i, column=3, sticky="we", padx=3)
            ve.bind("<FocusOut>", lambda ev: self.refresh_preview())
            emin = ttk.Entry(grid, textvariable=row["vmin"], width=6)
            emin.grid(row=i, column=4, padx=2)
            emax = ttk.Entry(grid, textvariable=row["vmax"], width=6)
            emax.grid(row=i, column=5, padx=2)
            estep = ttk.Entry(grid, textvariable=row["vstep"], width=6)
            estep.grid(row=i, column=6, padx=2)
            for e in (emin, emax, estep):
                e.bind("<FocusOut>", lambda ev: self.refresh_preview())
            ttk.Button(grid, text="Remove", command=lambda r=row: self._remove_custom_axis_row(r)).grid(
                row=i, column=7, padx=3
            )

    def _add_custom_axis_row(self) -> None:
        self._custom_axis_rows.append(self._make_custom_axis_row())
        self._render_custom_axis_rows()
        self.refresh_preview()

    def _remove_custom_axis_row(self, row: dict[str, tk.Variable]) -> None:
        try:
            self._custom_axis_rows.remove(row)
        except ValueError:
            pass
        self._render_custom_axis_rows()
        self.refresh_preview()

    def _set_custom_axis_rows(self, specs: list[dict]) -> None:
        """Rebuild the custom-flag editor from persisted specs."""
        self._custom_axis_rows = [
            self._make_custom_axis_row(
                flag=str(spec.get("flag", "")),
                include=bool(spec.get("enabled", True)),
                mode=str(spec.get("mode", "list")),
                values=str(spec.get("raw", "")),
                vmin=self._fmt(spec.get("min", 0)),
                vmax=self._fmt(spec.get("max", 0)),
                vstep=self._fmt(spec.get("step", 1)),
            )
            for spec in specs
        ]
        self._render_custom_axis_rows()

    # ------------------------------------------------------------------ axes
    def _collect_axes(self) -> list[Axis]:
        """Build sweep axes from the included lever rows AND custom-flag rows.

        Raises :class:`SweepError` on malformed input (bad numbers, empty
        included rows, or a custom flag that doesn't start with ``-``).
        """
        axes: list[Axis] = []
        backend = self._current_backend()
        for lever in LEVERS:
            v = self.lever_vars[lever.key]
            if not v["include"].get():
                continue
            # ik_llama-only levers must not produce axes on a llama.cpp backend
            # (their rows are hidden there, but a stale ``include`` could linger).
            if self._is_ik_lever(lever) and backend != "ik_llama":
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
        axes.extend(self._collect_custom_axes())
        return axes

    def _collect_custom_axes(self) -> list[Axis]:
        """Turn the included, named custom-flag rows into real matrix axes.

        A custom flag uses KIND_STR for list mode and KIND_INT for numeric
        range mode, mirroring the built-in lever rows. The flag itself is the
        axis key, so results tag combos with a ``[<flag>]`` column.
        """
        axes: list[Axis] = []
        seen_flags: set[str] = set()
        for row in self._custom_axis_rows:
            if not row["include"].get():
                continue
            flag = str(row["flag"].get()).strip()
            if not flag:
                continue
            # Two included rows sharing a flag would collide: the per-flag combo
            # dict overwrites, so every command reads the second value (e.g.
            # ``-ot C -ot C``) while the preview still claims the full product.
            if flag in seen_flags:
                raise SweepError(f"duplicate custom flag {flag!r} — combine its values into one row")
            seen_flags.add(flag)
            mode = row["mode"].get()
            if mode == "range":
                try:
                    vmin = float(row["vmin"].get())
                    vmax = float(row["vmax"].get())
                    vstep = float(row["vstep"].get())
                except ValueError as exc:
                    raise SweepError(f"custom flag {flag!r}: min/max/step must be numeric") from exc
                values = expand_range(vmin, vmax, vstep, KIND_INT)
                kind = KIND_INT
            else:
                values = parse_list(row["values"].get(), KIND_STR)
                kind = KIND_STR
            if not values:
                raise SweepError(f"custom flag {flag!r} is included but has no values.")
            # Axis.__post_init__ rejects a flag that doesn't start with '-'.
            axes.append(Axis(key=flag, values=values, custom_flag=flag, custom_kind=kind))
        return axes

    def _current_exe(self) -> str | None:
        build = self._selected_build()
        if build is None:
            return None
        return build.tool_path(self.tool_var.get())

    def _current_backend(self) -> str:
        """Backend selection is the source of truth (drives flash-attn handling,
        ik-only levers and the ik fuse toggles), defaulting to ``llama.cpp``."""
        try:
            return self.backend_var.get() or "llama.cpp"
        except Exception:
            return "llama.cpp"

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
            extra_args=self._current_extra_args(),
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
        backend = self._current_backend()
        _applicable, ignored = split_axes_for_tool(axes, tool, backend)
        if not _applicable:
            names = ", ".join(a.label for a in ignored)
            self._set_preview(
                f"(none of the selected parameters apply to {_TOOL_LABELS.get(tool, tool)} — "
                f"the ignored ones are: {names})"
            )
            return
        size = matrix_size(axes, tool, backend)
        model = self.model_var.get().strip() or "<model.gguf>"
        exe = self._current_exe() or f"<{tool}>"
        try:
            pairs = build_commands(
                tool,
                exe,
                model,
                axes,
                backend=backend,
                output_format=self.output_format_var.get(),
                repetitions=self._repetitions(),
                extra_args=self._current_extra_args(),
            )
        except SweepError as exc:
            self._set_preview(f"(incomplete) {exc}")
            return
        if tool == TOOL_LLAMA_BENCH:
            lines.append(f"# 1 invocation, {size} matrix row(s)")
        else:
            lines.append(f"# {len(pairs)} invocation(s) (cartesian product)")
        if ignored:
            names = ", ".join(a.label for a in ignored)
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
    def _plan_env(self) -> dict[str, str]:
        """Environment overrides for the benchmark child: the launcher's enabled
        environment variables merged with the ``CUDA_VISIBLE_DEVICES`` choice.

        Two sources, GPU visibility winning on a key conflict:

        1. The launcher's *enabled* env vars (e.g. ``GGML_CUDA_FORCE_MMQ``,
           ``GGML_CUDA_FORCE_CUBLAS``) — the same set the server launch applies
           via ``env_vars_manager.get_enabled_env_vars()`` — so the benchmark
           child runs with the same backend knobs as the server being compared.
        2. The ``CUDA_VISIBLE_DEVICES`` override. ``system.py`` clears any
           inherited value at startup, so a benchmark child would otherwise see
           ALL physical GPUs — making a seeded ``main_gpu`` / ``tensor_split``
           index refer to a different logical device than the server would use.
           Reuse the launcher's single source of truth
           (``LaunchManager._resolve_cuda_visible_devices_action``) so benchmark
           and live server agree on the visible-device subset and order. Only the
           ``export`` action carries a concrete value; ``unset`` / ``skip`` leave
           the variable unset (the child sees all GPUs), matching server launch.

        Applied last so GPU visibility wins if an enabled env var also set
        ``CUDA_VISIBLE_DEVICES``.

        Defensive: a launcher missing either accessor simply contributes nothing.
        """
        env: dict[str, str] = {}
        # 1) Enabled environment variables (same accessor the server launch uses).
        try:
            manager = getattr(self.launcher, "env_vars_manager", None)
            getter = getattr(manager, "get_enabled_env_vars", None)
            if getter is not None:
                for key, value in (getter() or {}).items():
                    env[str(key)] = str(value)
        except Exception:
            pass
        # 2) GPU visibility override — wins on key conflict (applied last).
        try:
            launch_manager = getattr(self.launcher, "launch_manager", None)
            resolver = getattr(launch_manager, "_resolve_cuda_visible_devices_action", None)
            if resolver is not None:
                action, value = resolver()
                if action == "export" and value not in (None, ""):
                    env["CUDA_VISIBLE_DEVICES"] = str(value)
        except Exception:
            pass
        return env

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
        # Reject a matrix whose ticked rows all apply to the OTHER tool: the
        # empty-axes guard above counts axes BEFORE tool filtering, so without
        # this check build_commands() would silently drop every axis and run a
        # single unswept default command.
        backend = self._current_backend()
        applicable, ignored = split_axes_for_tool(axes, tool, backend)
        if not applicable:
            names = ", ".join(a.label for a in ignored)
            messagebox.showerror(
                "Benchmark",
                f"None of the selected parameters apply to {_TOOL_LABELS.get(tool, tool)} — "
                f"the ignored ones are: {names}.",
            )
            return
        try:
            pairs = build_commands(
                tool,
                exe,
                model,
                axes,
                backend=backend,
                output_format=self.output_format_var.get(),
                repetitions=self._repetitions(),
                extra_args=self._current_extra_args(),
            )
        except SweepError as exc:
            messagebox.showerror("Benchmark", str(exc))
            return
        steps = [BenchStep(cmd=cmd, combo=combo, label=self._combo_label(combo)) for cmd, combo in pairs]
        cwd = str(Path(exe).parent)
        plan = BenchPlan(tool=tool, steps=steps, cwd=cwd, env=self._plan_env())

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
                    self._stop_polling_on_teardown()
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
            self._stop_polling_on_teardown()
            return
        # Liveness fallback: the worker can drop its terminal event under queue
        # saturation + cancel (bench_runner._emit_event drops on a full queue
        # once cancelled). Without this, the worker would exit but the UI would
        # never leave the "running" state (Start stuck disabled, Cancel a
        # no-op). Once the thread is dead AND the queue is fully drained, no
        # terminal event is still coming, so finalise here.
        if not self.runner.is_running and self.runner.events.empty():
            self._set_running(False)
            if self.status_var.get().strip() in ("", "Running…", "Cancelling…") or self.status_var.get().startswith(
                "Running "
            ):
                self.status_var.set("Stopped.")
            self._poll_after_id = None
            return
        delay = RUNNER_CATCHUP_POLL_MS if (had_events and drained >= RUNNER_MAX_EVENTS_PER_POLL) else RUNNER_POLL_MS
        try:
            self._poll_after_id = self.root.after(delay, self._poll_runner)
        except tk.TclError:
            # Root is gone; nothing left to poll.
            self._stop_polling_on_teardown()

    def _stop_polling_on_teardown(self) -> None:
        """Stop the poll loop and make sure the worker can't park undrained.

        Called when Tk raises ``TclError`` (a widget/root was destroyed under
        us). If we simply stop polling while the runner is still alive, nothing
        will drain its event queue and the worker can eventually block on a full
        queue. Cancelling it (signal-only, non-blocking) prevents that. Safe if
        the run already finished — ``cancel`` is idempotent.
        """
        self._poll_after_id = None
        try:
            if self.runner.is_running:
                self.runner.cancel()
        except Exception:
            pass

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
        tool = self.tool_var.get()
        backend = self._current_backend()
        try:
            axes = self._collect_axes()
        except SweepError as exc:
            messagebox.showerror("Save script", str(exc))
            return
        # Same guard start()/refresh_preview() apply: a matrix whose ticked axes
        # are ALL ignored for this tool (e.g. a llama-bench-only lever with
        # tool=sweep-bench) still yields one unswept default command, which the
        # ``if not commands`` check below wouldn't catch. Reject it here.
        if axes:
            applicable, ignored = split_axes_for_tool(axes, tool, backend)
            if not applicable:
                names = ", ".join(a.label for a in ignored)
                messagebox.showerror(
                    "Save script",
                    f"None of the selected parameters apply to {_TOOL_LABELS.get(tool, tool)} — "
                    f"the ignored ones are: {names}.",
                )
                return
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
            # Same merged GPU + enabled-env-vars environment as the in-app run,
            # so the exported script exports CUDA_VISIBLE_DEVICES and the enabled
            # env vars the server launch would apply.
            script = bench_script.render(commands, fmt, header=header, env=self._plan_env())
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
        extra_args_rows = self._current_extra_args()
        custom_axes_spec: list[dict] = []
        for row in self._custom_axis_rows:
            flag = str(row["flag"].get()).strip()
            if not flag:
                continue
            custom_axes_spec.append(
                {
                    "flag": flag,
                    "enabled": bool(row["include"].get()),
                    "mode": row["mode"].get(),
                    "raw": row["values"].get(),
                    "min": self._safe_float(row["vmin"].get()),
                    "max": self._safe_float(row["vmax"].get()),
                    "step": self._safe_float(row["vstep"].get(), 1.0),
                }
            )
        return BenchConfig(
            name=name,
            tool=self.tool_var.get(),
            # Backend selection is the source of truth (a build may not be
            # selected); fall back to the selected build only if unavailable.
            backend=self._current_backend(),
            build_root=(build.root_dir if build else ""),
            model_path=self.model_var.get().strip(),
            axes=axes_spec,
            output_format=self.output_format_var.get(),
            repetitions=self._repetitions() or 0,
            # Keep the scalar joined for external consumers; rows are canonical.
            extra_args=" ".join(extra_args_rows),
            extra_args_rows=extra_args_rows,
            custom_axes=custom_axes_spec,
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

        # Restore the backend first (source of truth for ik-only UI + rendering)
        # and mirror it to the launcher, guarded against the sync trace recursing.
        if cfg.backend in ("llama.cpp", "ik_llama"):
            self._syncing_backend = True
            try:
                self.backend_var.set(cfg.backend)
                try:
                    self.launcher.backend_selection.set(cfg.backend)
                except Exception:
                    pass
            finally:
                self._syncing_backend = False

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
        self._set_tool(saved_tool)
        if cfg.model_path:
            self.model_var.set(cfg.model_path)
        # Run output is JSON-only (see Options); ignore any legacy csv/markdown.
        self.output_format_var.set("json")
        self.repetitions_var.set(str(cfg.repetitions) if cfg.repetitions else "")
        # Restore the multi-row Extra-args editor (persistence migrates a legacy
        # scalar extra_args into a single row for us).
        self._set_extra_args_rows(list(cfg.extra_args_rows))
        self._set_custom_axis_rows(list(cfg.custom_axes))
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
