"""Build tab: clone, configure, build llama.cpp / ik_llama.cpp.

Wiring
------
The launcher creates one ``BuildTab(launcher)`` at startup and calls
``tab.setup_tab(parent_frame)`` to render the UI into a Notebook tab.

State model
-----------
All persistent state lives on ``self`` as plain attributes and Tk variables.
Widgets are rebuilt on detach/reattach but the underlying state survives,
because:

  * Tk variables (StringVar, BooleanVar, IntVar) are not parented by widgets.
  * The build console history is mirrored into ``self._console_buffer``.
  * Flag values are mirrored into ``self._flag_vars`` (a dict of Tk vars,
    rebuilt only the first time the UI is constructed and reused thereafter).

Threads
-------
The build pipeline runs on the ``BuildRunner`` worker thread; the UI polls
its events queue from the Tk mainloop via ``root.after(...)``. Upstream
fetches (used by the update banner) are spawned via ``threading.Thread`` and
post their result back to the UI through ``self._pending_status``.
"""

from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any

from . import cmake_flags as cf
from . import detection
from .build_persistence import BuildConfig, BuildConfigStore
from .build_runner import (
    BuildPlan,
    BuildRunner,
    EVENT_CANCELLED,
    EVENT_DONE,
    EVENT_ERROR,
    EVENT_LINE,
    EVENT_STAGE,
    UPSTREAMS,
    UpstreamStatus,
    plan_to_shell_script,
    probe_upstream,
)
from modules import terminal_launcher


CONSOLE_MAX_LINES = 5000
PREVIEW_REFRESH_DEBOUNCE_MS = 120
RUNNER_POLL_MS = 60
RUNNER_CATCHUP_POLL_MS = 5
RUNNER_MAX_EVENTS_PER_POLL = 250
RUNNER_MAX_CHARS_PER_POLL = 128 * 1024
PULL_DRAIN_MS = 80
DEFAULT_GENERATOR_LABEL = "cmake"


def _truthy_flag_str(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "on", "true", "yes"}
    return bool(v)

# Parse a CUDA arch token like "86-real", "120a-real", "120f-real", "90a",
# "75" into its parts. Group "base" is the digit pair, "variant" is the
# optional 'a' (arch-accelerated) or 'f' (family-forward) suffix, "suffix"
# is the trailing "-real" / "-virtual" if present.
_ARCH_TOKEN_RE = re.compile(r"^(?P<base>\d{2,3})(?P<variant>[af]?)(?P<suffix>(?:-real|-virtual)?)$")


class BuildTab:
    """Owner of the Build tab UI and its build pipeline."""

    # ---------------------------------------------------------------- init
    def __init__(self, launcher: Any) -> None:
        self.launcher = launcher
        self.root = launcher.root

        # Backing store for named build configs (config/build_configs.json).
        try:
            config_dir = Path(launcher.config_path).parent
        except Exception:
            config_dir = Path("config")
        self.store = BuildConfigStore(config_dir)

        # Runner — single instance for the tab lifetime.
        self.runner = BuildRunner()

        # Streaming console state.
        self._console_buffer: list[str] = []
        self._poll_after_id: str | None = None
        self._preview_after_id: str | None = None
        self._drain_after_id: str | None = None
        self._pull_drain_after_id: str | None = None

        # Update-banner state.
        self._upstream_status = UpstreamStatus()
        self._upstream_check_in_flight = False
        self._pending_status: queue.Queue[tuple[UpstreamStatus, bool]] = queue.Queue()
        self._pending_pull_events: queue.Queue[tuple[str, object]] = queue.Queue()

        # Notebook integration. Detach-to-Toplevel was removed because the
        # rebuild involved in re-parenting the heavy widget tree caused
        # lock-ups in practice.
        self._notebook = None
        self._tab_frame = None
        self._tab_text = "Build (beta)"
        # Async-refresh guard + queue so we don't start two concurrent toolchain
        # probes. Worker thread puts the new ToolchainProbe (or an exception) on
        # the queue; the Tk main thread polls and applies it.
        self._toolchain_refresh_in_flight = False
        self._toolchain_refresh_queue: queue.Queue = queue.Queue()
        self._toolchain_refresh_after_id: str | None = None
        self._pull_only_in_flight = False
        self._cuda_arch_cache: list[detection.CudaArchInfo] | None = None
        self._cuda_arch_cache_avx512 = False

        # Re-entry / scheduling guards: _build_ui destroys its parent's
        # children, so we must NEVER call it synchronously from a widget's
        # command callback — that destroys the very widget mid-event and
        # Tk's event loop hangs. _rebuild_pending coalesces deferred rebuilds.
        # _rebuild_full_pending elevates a pending light rebuild to a full one.
        self._rebuild_pending: bool = False
        self._rebuild_full_pending: bool = False
        self._suspend_traces: bool = False
        self._syncing_backend_dirs: bool = False
        self._syncing_backend_selection: bool = False

        # ── Tk variables (state survives widget rebuilds) ──
        seed_backend = "llama.cpp"
        try:
            seed_backend = launcher.backend_selection.get() or "llama.cpp"
        except Exception:
            pass
        self.var_backend = tk.StringVar(value=seed_backend)
        self.var_source_dir = tk.StringVar(value=self._initial_source_dir(seed_backend))
        self.var_build_dir = tk.StringVar(value="build")
        self.var_git_ref = tk.StringVar(value="")
        self.var_git_pull = tk.BooleanVar(value=False)
        self.var_clean_build = tk.BooleanVar(value=True)
        self.var_jobs = tk.IntVar(value=detection.recommend_jobs().suggested)
        self.var_cuda_archs = tk.StringVar(value="")
        self.var_prefer_a_variant = tk.BooleanVar(value=True)
        self.var_prefer_f_variant = tk.BooleanVar(value=False)
        self.var_show_deprecated_archs = tk.BooleanVar(value=False)
        self.var_cc = tk.StringVar(value="")
        self.var_cxx = tk.StringVar(value="")
        self.var_cudacxx = tk.StringVar(value="")
        # Root dir of the active CUDA toolkit. Drives CMAKE_CUDA_TOOLKIT_ROOT_DIR
        # so cmake doesn't pick up a stray install via PATH. Empty = derive from nvcc.
        self.var_cuda_root = tk.StringVar(value="")
        # Picker for "which detected CUDA install"; updated when var_cudacxx changes.
        self.var_cuda_pick = tk.StringVar(value="")
        # cmake -G generator. Empty = whatever cmake's platform default is.
        self.var_generator = tk.StringVar(value=DEFAULT_GENERATOR_LABEL)
        self.var_extra_args = tk.StringVar(value="")
        self.var_config_name = tk.StringVar(value="")
        self.var_status = tk.StringVar(value="Idle")
        self.var_jobs_hint = tk.StringVar(value="")
        self.var_toolchain_hint = tk.StringVar(value="")
        # Status banner shown next to the source-dir entry. Updates live
        # as the user types (or pastes) a new path: tells them whether the
        # path is an existing git repo, an existing non-git dir, or doesn't
        # exist yet and will be auto-cloned on build.
        self.var_source_status = tk.StringVar(value="")
        self.var_source_status_color = tk.StringVar(value="#666")
        self.var_auto_check_updates = tk.BooleanVar(value=True)
        self.var_autoscroll = tk.BooleanVar(value=True)

        # Lazy-initialised on first setup_tab().
        self._flag_vars: dict[str, tk.Variable] = {}
        self._flag_widgets: dict[str, tk.Widget] = {}
        self._group_frames: dict[str, ttk.LabelFrame] = {}
        self._values_snapshot: dict[str, Any] = {}
        # The outer "CMake flags" LabelFrame. Kept stable across backend
        # switches so we only have to rebuild its contents, not the whole
        # tab (rebuilding the whole tab tears down ~349 widgets including
        # the canvas/scrollable which can deadlock on pending Configure
        # events).
        self._flags_label_frame: ttk.LabelFrame | None = None
        # The CUDA arch-picker grid (one LabelFrame inside Build environment).
        # Stashed so the show-deprecated toggle and load-config paths can
        # rebuild just its contents instead of the whole tab. None until
        # _build_section_environment has run.
        self._archs_grid_frame: ttk.LabelFrame | None = None
        self._tool_install_frame: ttk.Frame | None = None

        # CUDA arch picker state: one BooleanVar per known CC, plus a guard
        # to suppress recursive sync between checkboxes and the text entry.
        self._arch_check_vars: dict[str, tk.BooleanVar] = {}
        self._arch_sync_in_progress: bool = False
        # Update the picker checkboxes whenever the entry changes (e.g. via
        # auto-detect, load-config, manual typing).
        self.var_cuda_archs.trace_add("write", lambda *_a: self._sync_arch_pickers_from_value())

        # Toolchain probe is deferred to a worker thread so __init__ doesn't
        # block the Tk main thread. ``probe_toolchain`` shells out to
        # nvcc/cmake/git/which/ls and can spend 0.5–3 s on a system with
        # multiple CUDA installs; doing it inline made cold startup feel
        # frozen. Start with an empty probe so attribute access works,
        # render the UI immediately, then post the real probe back via
        # ``_toolchain_refresh_queue`` (the same plumbing as the manual
        # "Refresh toolchain" button).
        self._toolchain = detection.ToolchainProbe()
        self._refresh_toolchain_hint()
        self._refresh_jobs_hint()
        # No _seed_toolchain_defaults / _apply_autodetect_defaults here —
        # both depend on probed data (cc/cxx/nvcc candidates, cuda_installs,
        # torch.cuda.get_device_properties). They run inside
        # ``_refresh_toolchain_finish`` once the worker delivers a real
        # probe. _on_cudacxx_changed is a no-op without nvcc set, so we
        # also skip the explicit call here.

        # Kick off the deferred probe. ``after_idle`` lets the rest of
        # __init__ + _create_widgets finish before the worker is started,
        # so the user sees the tab render first.
        try:
            self.root.after_idle(self._start_initial_toolchain_probe)
        except Exception:
            # Fall back to synchronous probe if Tk's idle queue isn't
            # available (e.g. in a headless test harness).
            self._toolchain = detection.probe_toolchain()
            self._refresh_toolchain_hint()
            self._refresh_jobs_hint()
            self._seed_toolchain_defaults()
            self._on_cudacxx_changed()
            self._apply_autodetect_defaults()

        # Keep the build source directory and main-tab backend root directory
        # mirrored per backend. Users can still select either backend here; the
        # source path writes through to the matching launcher root.
        try:
            launcher.backend_selection.trace_add("write", self._on_launcher_backend_changed)
        except Exception:
            pass
        try:
            launcher.current_backend_dir.trace_add("write", self._on_launcher_current_dir_changed)
        except Exception:
            pass
        try:
            launcher.llama_cpp_dir.trace_add(
                "write",
                lambda *_a: self._on_launcher_backend_dir_var_changed("llama.cpp"),
            )
        except Exception:
            pass
        try:
            launcher.ik_llama_dir.trace_add(
                "write",
                lambda *_a: self._on_launcher_backend_dir_var_changed("ik_llama"),
            )
        except Exception:
            pass

        # Traces that drive the live cmake-preview refresh.
        for v in (self.var_backend, self.var_source_dir, self.var_build_dir,
                  self.var_git_ref, self.var_git_pull, self.var_clean_build,
                  self.var_jobs, self.var_cuda_archs, self.var_extra_args,
                  self.var_cc, self.var_cxx, self.var_cudacxx,
                  self.var_cuda_root, self.var_generator):
            v.trace_add("write", lambda *_a: self._schedule_preview_refresh())

        # Whenever CUDACXX changes (user-edited or programmatic), keep
        # CUDA root + picker label in sync.
        self.var_cudacxx.trace_add("write", lambda *_a: self._on_cudacxx_changed())

        # Live source-dir state indicator: updates as the user types so
        # they get immediate feedback about whether the path exists, is
        # a git repo, or will be auto-cloned.
        self.var_source_dir.trace_add("write", self._on_build_source_dir_changed)
        self.var_backend.trace_add("write", self._on_build_backend_var_changed)

    # ─────────────────────────────────────────────────────────────────────
    # Seeding helpers
    # ─────────────────────────────────────────────────────────────────────
    def _initial_source_dir(self, backend: str) -> str:
        existing = ""
        try:
            if backend == "ik_llama":
                existing = self.launcher.ik_llama_dir.get()
            else:
                existing = self.launcher.llama_cpp_dir.get()
        except Exception:
            pass
        return detection.default_source_dir(backend, existing)

    def _seed_toolchain_defaults(self) -> None:
        tp = self._toolchain
        if not self.var_cc.get() and tp.cc_candidates:
            self.var_cc.set(tp.cc_candidates[0])
        if not self.var_cxx.get() and tp.cxx_candidates:
            self.var_cxx.set(tp.cxx_candidates[0])
        if not self.var_cudacxx.get() and tp.nvcc_path:
            self.var_cudacxx.set(tp.nvcc_path)
        # _on_cudacxx_changed runs via trace when var_cudacxx was set above.

    def _on_cudacxx_changed(self) -> None:
        """Sync the CUDA root + picker label whenever CUDACXX changes."""
        nvcc = self.var_cudacxx.get().strip()
        if not nvcc:
            self.var_cuda_root.set("")
            self.var_cuda_pick.set("")
            return
        # Try to match against detected installs first.
        for inst in self._toolchain.cuda_installs:
            if not os.path.isfile(nvcc) or not os.path.isfile(inst.nvcc_path):
                continue
            try:
                if os.path.samefile(nvcc, inst.nvcc_path):
                    self.var_cuda_root.set(inst.root_dir)
                    self.var_cuda_pick.set(inst.label())
                    return
            except OSError:
                # samefile can raise OSError on broken symlinks or stat errors
                # (especially on Windows). Try the next install.
                continue
        # Fallback: derive root from the nvcc path. If resolution fails,
        # clear the previous CUDA root rather than leaving a stale one —
        # otherwise the build plan would pair the new CUDACXX with the
        # last detected install's toolkit root.
        try:
            self.var_cuda_root.set(detection._root_from_nvcc(nvcc))
        except Exception:
            self.var_cuda_root.set("")
        self.var_cuda_pick.set(f"Custom · {nvcc}")

    @staticmethod
    def _safe_int(var: tk.Variable, default: int = 0, minimum: int = 0) -> int:
        """Read an IntVar/StringVar as int, falling back to ``default`` on bad
        input and clamping to ``minimum``. Used for Spinbox-bound vars where
        a user could type non-numeric text and crash the build path."""
        try:
            v = int(var.get() or 0)
        except (TypeError, ValueError, tk.TclError):
            return default
        return max(minimum, v)

    def _refresh_jobs_hint(self) -> None:
        reco = detection.recommend_jobs()
        self.var_jobs_hint.set(f"recommended: {reco.suggested} ({reco.reason})")

    def _refresh_toolchain_hint(self) -> None:
        tp = self._toolchain
        bits: list[str] = []
        if tp.cuda_version:
            bits.append(f"CUDA {tp.cuda_version}")
        if tp.cmake_version:
            bits.append(f"cmake {tp.cmake_version}")
        if tp.git_version:
            bits.append(f"git {tp.git_version}")
        if tp.ccache_path:
            bits.append("ccache")
        if tp.ninja_path:
            bits.append(f"ninja {tp.ninja_version}" if tp.ninja_version else "ninja")
        self.var_toolchain_hint.set(
            " · ".join(bits) if bits else "no CUDA/cmake/git/ninja/ccache detected"
        )

    @staticmethod
    def _generator_display_value(value: str | None) -> str:
        text = (value or "").strip()
        return DEFAULT_GENERATOR_LABEL if not text else text

    def _selected_generator_value(self) -> str:
        text = self.var_generator.get().strip()
        if not text or text.casefold() == DEFAULT_GENERATOR_LABEL:
            return ""
        return text

    @staticmethod
    def _format_build_tool_status(tool: detection.BuildToolStatus) -> str:
        if tool.installed:
            version = f" {tool.version}" if tool.version else ""
            return f"detected{version} at {tool.path}"
        if tool.install_plan is not None:
            return f"not detected; install via {tool.install_plan.package_manager}"
        return "not detected; no supported installer found"

    def _rebuild_build_tool_rows(self) -> None:
        frame = self._tool_install_frame
        if frame is None or not frame.winfo_exists():
            return
        for child in frame.winfo_children():
            child.destroy()
        frame.columnconfigure(1, weight=1)
        tools = detection.build_tool_statuses(self._toolchain)
        for row, tool in enumerate(tools):
            ttk.Label(frame, text=f"{tool.label}:").grid(row=row, column=0, sticky="w")
            ttk.Label(
                frame,
                text=self._format_build_tool_status(tool),
                font=("TkSmallCaptionFont",),
            ).grid(row=row, column=1, sticky="w", padx=(6, 0))
            btn_text = "Installed" if tool.installed else (
                f"Install {tool.label}" if tool.install_plan else "No installer"
            )
            ttk.Button(
                frame,
                text=btn_text,
                command=lambda k=tool.key: self._on_install_build_tool(k),
                state="disabled" if tool.installed or tool.install_plan is None else "normal",
            ).grid(row=row, column=2, sticky="e", padx=(8, 0))
        ttk.Label(
            frame,
            text="Install commands open in a new terminal. Refresh toolchain after they finish.",
            font=("TkSmallCaptionFont",),
        ).grid(row=len(tools), column=0, columnspan=3, sticky="w", pady=(4, 0))

    def _apply_autodetect_defaults(self) -> None:
        """Seed the flag-values dict from the auto-detected preset.

        Merges over the current snapshot rather than replacing it: when the
        user switches backend we keep their overlapping per-flag overrides
        and only fill in flags that didn't exist (or were unset) before.
        Call _on_apply_autodetect for a hard reset instead.

        CUDA arch detection is cached for the tab's lifetime. Startup detection
        is fed by the worker probe or the launcher's cached/nvidia-smi GPU info;
        this method deliberately avoids torch so it is safe on the Tk thread.
        """
        if self._cuda_arch_cache is None:
            self._cuda_arch_cache = detection.cuda_archs_from_gpu_info(
                getattr(self.launcher, "gpu_info", None)
            )
            self._cuda_arch_cache_avx512 = self._cpu_has_avx512()
        cuda_infos = self._cuda_arch_cache
        avx512 = self._cuda_arch_cache_avx512
        cuda_available = bool(cuda_infos) and self._toolchain.nvcc_path is not None
        defaults = cf.build_autodetect_values(
            self.var_backend.get(),
            cuda_available=cuda_available,
            avx512_supported=avx512,
            has_ccache=bool(self._toolchain.ccache_path),
        )
        # Merge: defaults provide a baseline, user values (in current snapshot)
        # win for any flag that exists in both. For a fresh seed at startup
        # _values_snapshot is empty, so defaults take effect cleanly.
        self._values_snapshot = {**defaults, **self._values_snapshot}
        if cuda_available and not self.var_cuda_archs.get().strip():
            self.var_cuda_archs.set(
                detection.archs_to_cmake_value(
                    cuda_infos, prefer_a_variant=self.var_prefer_a_variant.get()
                )
            )

    def _reset_to_autodetect(self) -> None:
        """Hard reset — discards user overrides. Used by the 'Auto-detect
        preset' button explicitly, not by backend-switch flows."""
        self._values_snapshot = {}
        self.var_cuda_archs.set("")
        self._apply_autodetect_defaults()

    def _cpu_has_avx512(self) -> bool:
        try:
            with open("/proc/cpuinfo", "r") as fh:
                txt = fh.read()
            return " avx512f " in (" " + txt + " ") or "avx512f" in txt.split()
        except Exception:
            return False

    # ─────────────────────────────────────────────────────────────────────
    # Public API used by launcher
    # ─────────────────────────────────────────────────────────────────────
    def setup_tab(self, parent: tk.Widget) -> None:
        """Render the build tab UI into ``parent`` (a notebook tab frame)."""
        self._tab_frame = parent
        self._build_ui(parent)
        # Kick the preview + initial update-banner check.
        self._schedule_preview_refresh()
        if self.var_auto_check_updates.get():
            self.check_for_updates(do_fetch=False)

    def register_with_notebook(self, notebook: ttk.Notebook, tab_text: str) -> None:
        self._notebook = notebook
        self._tab_text = tab_text

    # ─────────────────────────────────────────────────────────────────────
    # UI construction (re-entrant on detach/reattach)
    # ─────────────────────────────────────────────────────────────────────
    def _build_ui(self, parent: tk.Widget) -> None:
        # Cancel any pending after() callbacks that reference about-to-be-
        # destroyed widgets so they don't fire against stale references.
        for attr in ("_preview_after_id", "_drain_after_id"):
            aid = getattr(self, attr, None)
            if aid is not None:
                try:
                    self.root.after_cancel(aid)
                except Exception:
                    pass
                setattr(self, attr, None)
        for child in parent.winfo_children():
            child.destroy()

        outer = ttk.Frame(parent)
        outer.pack(fill="both", expand=True)
        outer.rowconfigure(1, weight=2)
        outer.columnconfigure(0, weight=1)

        # Top: header bar with detach + status + update banner area.
        self._build_header(outer)

        # Scrollable middle pane with all editor sections.
        canvas = tk.Canvas(outer, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.grid(row=1, column=0, sticky="nsew", padx=(8, 0), pady=4)
        vsb.grid(row=1, column=1, sticky="ns", padx=(0, 4), pady=4)

        scrollable = ttk.Frame(canvas)
        canvas_window = canvas.create_window((0, 0), window=scrollable, anchor="nw")

        def _on_scrollable_config(_event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_config(event):
            canvas.itemconfigure(canvas_window, width=event.width)

        scrollable.bind("<Configure>", _on_scrollable_config)
        canvas.bind("<Configure>", _on_canvas_config)

        def _wheel(event):
            # Cross-platform scroll-wheel handling:
            #   * Windows: event.delta is ±120 per notch.
            #   * macOS:   event.delta is small (±1..±N) per notch.
            #   * Linux:   no event.delta; uses Button-4 (up) / Button-5 (down).
            # We dispatch by which attribute is meaningful for this event.
            if getattr(event, "delta", 0):
                # Use the sign of delta, not its magnitude — macOS reports tiny
                # values where dividing by 120 truncates to 0.
                if abs(event.delta) >= 120:
                    delta = -1 * (event.delta // 120)
                else:
                    delta = -1 if event.delta > 0 else 1
            elif getattr(event, "num", None) == 4:
                delta = -1
            else:
                delta = 1
            canvas.yview_scroll(delta, "units")

        canvas.bind("<MouseWheel>", _wheel)
        canvas.bind("<Button-4>", _wheel)
        canvas.bind("<Button-5>", _wheel)

        # Each section is laid out top-to-bottom in `scrollable`.
        scrollable.columnconfigure(0, weight=1)
        row = 0
        self._build_section_config_bar(scrollable, row)
        row += 1
        self._build_section_source(scrollable, row)
        row += 1
        self._build_section_environment(scrollable, row)
        row += 1
        self._build_section_flags(scrollable, row)
        row += 1
        self._build_section_preview(scrollable, row)
        row += 1

        # Bottom: action bar + console.
        self._build_action_bar(outer)
        self._build_console(outer)

        # Now that widgets exist, push current state into them and refresh
        # config-name dropdown.
        self._refresh_saved_configs_dropdown()
        # Suspend per-flag traces while bulk-writing — otherwise each
        # var.set() fires _on_flag_changed which itself does O(N) work
        # (iterates every visible_when flag), producing O(N²) cost and
        # a flood of redundant preview-refresh schedules.
        self._suspend_traces = True
        try:
            self._sync_flag_widgets_from_values()
        finally:
            self._suspend_traces = False
        self._update_status_banner_visibility()

    # ── Header (title + detach + status) ─────────────────────────────────
    def _build_header(self, parent: ttk.Frame) -> None:
        hdr = ttk.Frame(parent)
        hdr.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 0))
        hdr.columnconfigure(1, weight=1)

        ttk.Label(hdr, text="Build llama.cpp / ik_llama.cpp",
                  font=("TkDefaultFont", 13, "bold")) \
            .grid(row=0, column=0, sticky="w")

        right = ttk.Frame(hdr)
        right.grid(row=0, column=2, sticky="e")
        ttk.Label(right, textvariable=self.var_toolchain_hint,
                  font=("TkSmallCaptionFont",)).pack(side="left", padx=(0, 8))
        ttk.Button(right, text="Refresh toolchain",
                   command=self._on_refresh_toolchain).pack(side="left", padx=2)

        # Update banner — hidden when no updates available.
        self._banner = tk.Frame(parent, bg="#fff5cf", highlightbackground="#c5a800",
                                highlightthickness=1)
        self._banner.columnconfigure(0, weight=1)
        self._banner_label = tk.Label(self._banner, bg="#fff5cf", anchor="center",
                                      justify="center",
                                      text="Checking upstream…",
                                      font=("TkDefaultFont", 10))
        self._banner_label.grid(row=0, column=0, sticky="ew", padx=12, pady=(8, 2))
        self._banner_btns = tk.Frame(self._banner, bg="#fff5cf")
        self._banner_btns.grid(row=1, column=0, pady=(0, 8))
        ttk.Button(self._banner_btns, text="Check",
                   command=lambda: self.check_for_updates(do_fetch=True, show_output=True)) \
            .pack(side="left", padx=2)
        ttk.Button(self._banner_btns, text="Pull only",
                   command=self._on_pull_only).pack(side="left", padx=2)
        ttk.Button(self._banner_btns, text="Pull & Rebuild",
                   command=self._on_pull_and_rebuild).pack(side="left", padx=2)
        ttk.Button(self._banner_btns, text="Dismiss",
                   command=self._banner.grid_remove).pack(side="left", padx=2)
        # Initially hidden; grid is set in _update_status_banner_visibility().

    # ── Named config bar ────────────────────────────────────────────────
    def _build_section_config_bar(self, parent: ttk.Frame, row: int) -> None:
        lf = ttk.LabelFrame(parent, text="Saved build configurations")
        lf.grid(row=row, column=0, sticky="ew", padx=8, pady=6)
        lf.columnconfigure(1, weight=1)

        ttk.Label(lf, text="Name:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self._cfg_combo = ttk.Combobox(lf, textvariable=self.var_config_name)
        self._cfg_combo.grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        self._cfg_combo.bind("<<ComboboxSelected>>", lambda *_a: self._on_load_config())

        btns = ttk.Frame(lf)
        btns.grid(row=0, column=2, sticky="e", padx=4)
        ttk.Button(btns, text="Load", command=self._on_load_config).pack(side="left", padx=2)
        ttk.Button(btns, text="Save", command=self._on_save_config).pack(side="left", padx=2)
        ttk.Button(btns, text="Save as…", command=self._on_save_as_config).pack(side="left", padx=2)
        ttk.Button(btns, text="Delete", command=self._on_delete_config).pack(side="left", padx=2)
        ttk.Button(btns, text="Auto-detect preset",
                   command=self._on_apply_autodetect).pack(side="left", padx=(8, 2))

    # ── Source / backend ────────────────────────────────────────────────
    def _build_section_source(self, parent: ttk.Frame, row: int) -> None:
        lf = ttk.LabelFrame(parent, text="Source")
        lf.grid(row=row, column=0, sticky="ew", padx=8, pady=6)
        lf.columnconfigure(1, weight=1)

        ttk.Label(lf, text="Backend:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        backend_frame = ttk.Frame(lf)
        backend_frame.grid(row=0, column=1, columnspan=2, sticky="w", padx=4)
        for label, value in (("llama.cpp", "llama.cpp"), ("ik_llama", "ik_llama")):
            ttk.Radiobutton(backend_frame, text=label, value=value,
                            variable=self.var_backend,
                            command=self._on_backend_changed).pack(side="left", padx=(0, 12))

        ttk.Label(lf, text="Source dir:").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(lf, textvariable=self.var_source_dir).grid(row=1, column=1, sticky="ew", padx=4)
        src_btns = ttk.Frame(lf)
        src_btns.grid(row=1, column=2, sticky="e", padx=4)
        ttk.Button(src_btns, text="Browse…", command=self._on_browse_source).pack(side="left", padx=2)
        ttk.Button(src_btns, text="Use backend dir",
                   command=self._on_use_backend_dir).pack(side="left", padx=2)

        # Live status line for the source dir. Recolored based on state
        # (gray = neutral / will-be-cloned, red = problem). Updated by
        # _update_source_dir_status whenever var_source_dir or var_backend
        # changes.
        self._source_status_label = tk.Label(
            lf, textvariable=self.var_source_status,
            anchor="w", justify="left", font=("TkSmallCaptionFont",),
            fg=self.var_source_status_color.get(),
        )
        self._source_status_label.grid(row=2, column=1, columnspan=2, sticky="w", padx=4)

        ttk.Label(lf, text="Build dir:").grid(row=3, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(lf, textvariable=self.var_build_dir).grid(row=3, column=1, sticky="ew", padx=4)
        ttk.Label(lf, text="(relative to source dir or absolute)",
                  font=("TkSmallCaptionFont",)).grid(row=3, column=2, sticky="w", padx=4)

        ttk.Label(lf, text="Git ref:").grid(row=4, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(lf, textvariable=self.var_git_ref).grid(row=4, column=1, sticky="ew", padx=4)
        ttk.Label(lf, text="branch/tag/commit (blank = leave as-is)",
                  font=("TkSmallCaptionFont",)).grid(row=4, column=2, sticky="w", padx=4)

        opts = ttk.Frame(lf)
        opts.grid(row=5, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 4))
        ttk.Checkbutton(opts, text="git pull --ff-only before build",
                        variable=self.var_git_pull).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(opts, text="Clean build dir first",
                        variable=self.var_clean_build).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(opts, text="Auto-check for updates",
                        variable=self.var_auto_check_updates).pack(side="left", padx=(0, 12))
        ttk.Button(opts, text="Clear CMake cache",
                   command=self._on_clear_cache).pack(side="left", padx=(0, 12))
        ttk.Label(lf,
                  text=("Tip: if source dir doesn't exist it will be auto-cloned. "
                        "Set git ref to a branch/tag/commit, or leave blank to use the current checkout."),
                  font=("TkSmallCaptionFont",), foreground="#666") \
            .grid(row=6, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 4))
        # Seed the status line now that the widget exists.
        self._update_source_dir_status()

    # ── Environment (jobs / archs / compilers) ──────────────────────────
    def _build_section_environment(self, parent: ttk.Frame, row: int) -> None:
        lf = ttk.LabelFrame(parent, text="Build environment")
        lf.grid(row=row, column=0, sticky="ew", padx=8, pady=6)
        lf.columnconfigure(1, weight=1)

        # Jobs
        ttk.Label(lf, text="Parallel jobs:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        jobs_frame = ttk.Frame(lf)
        jobs_frame.grid(row=0, column=1, sticky="w", padx=4)
        # validate="key" + validatecommand restricts the Spinbox to digit-only
        # keystrokes so var_jobs (IntVar) cannot be coerced to non-numeric text.
        # The lambda accepts "" (intermediate empty state while editing) plus
        # any digit string; everything else is rejected at the keystroke level.
        vcmd = (self.root.register(lambda P: P == "" or P.isdigit()), "%P")
        ttk.Spinbox(jobs_frame, from_=1, to=512, textvariable=self.var_jobs, width=6,
                    validate="key", validatecommand=vcmd) \
            .pack(side="left")
        ttk.Label(jobs_frame, textvariable=self.var_jobs_hint,
                  font=("TkSmallCaptionFont",)).pack(side="left", padx=8)

        # CUDA archs — entry + auto-detect + per-arch multi-select grid
        ttk.Label(lf, text="CUDA archs:").grid(row=1, column=0, sticky="nw", padx=6, pady=4)
        arch_frame = ttk.Frame(lf)
        arch_frame.grid(row=1, column=1, columnspan=2, sticky="ew", padx=4)
        arch_frame.columnconfigure(0, weight=1)

        ttk.Entry(arch_frame, textvariable=self.var_cuda_archs).grid(row=0, column=0, sticky="ew")
        arch_btns = ttk.Frame(arch_frame)
        arch_btns.grid(row=0, column=1, padx=4)
        ttk.Button(arch_btns, text="Detect", width=8,
                   command=self._on_autodetect_archs).pack(side="left", padx=2)
        ttk.Button(arch_btns, text="Clear", width=6,
                   command=lambda: self.var_cuda_archs.set("")).pack(side="left", padx=2)

        toggles_row = ttk.Frame(arch_frame)
        toggles_row.grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 0))
        ttk.Checkbutton(toggles_row, text="Prefer -a (arch-accelerated; Hopper+/Blackwell)",
                        variable=self.var_prefer_a_variant,
                        command=self._on_variant_toggle).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(toggles_row, text="Prefer -f (family-forward; Blackwell, CUDA 13+)",
                        variable=self.var_prefer_f_variant,
                        command=self._on_variant_toggle).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(toggles_row, text="Show deprecated archs (Kepler/Maxwell/Pascal/Volta)",
                        variable=self.var_show_deprecated_archs,
                        command=self._on_show_deprecated_toggle).pack(side="left")

        # Per-arch multi-select grid, grouped by generation family. Each
        # checkbutton appends/removes its arch tokens from var_cuda_archs.
        # Stash the container so the deprecated-archs toggle / load-config
        # paths can rebuild just this grid (~50 widgets) instead of doing
        # a tab-wide rebuild (~350 widgets).
        archs_grid = ttk.LabelFrame(arch_frame, text="Add architectures by generation")
        archs_grid.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self._archs_grid_frame = archs_grid
        self._build_cuda_arch_picker(archs_grid)
        ttk.Label(arch_frame,
                  text="Ticks add the matching arch tokens to the field above; untick to remove.",
                  font=("TkSmallCaptionFont",)) \
            .grid(row=3, column=0, columnspan=2, sticky="w", padx=2, pady=(2, 0))

        tp = self._toolchain

        # ── CUDA install picker ──
        cuda_pick_row = tk.Frame(lf)
        cuda_pick_row.grid(row=2, column=0, columnspan=3, sticky="ew", padx=6, pady=(8, 4))
        cuda_pick_row.columnconfigure(1, weight=1)
        ttk.Label(cuda_pick_row, text="CUDA install:") \
            .grid(row=0, column=0, sticky="w")
        # Build label → install map for the combobox.
        self._cuda_install_by_label: dict[str, detection.CudaInstall] = {
            inst.label(): inst for inst in tp.cuda_installs
        }
        labels = list(self._cuda_install_by_label.keys())
        if labels:
            labels.append("Custom path…")
        self._cuda_pick_combo = ttk.Combobox(
            cuda_pick_row, textvariable=self.var_cuda_pick,
            values=labels, state="readonly" if labels else "normal",
        )
        self._cuda_pick_combo.grid(row=0, column=1, sticky="ew", padx=4)
        self._cuda_pick_combo.bind("<<ComboboxSelected>>", self._on_cuda_install_picked)
        ttk.Button(cuda_pick_row, text="Re-scan",
                   command=self._on_rescan_cuda_installs).grid(row=0, column=2, padx=4)
        ttk.Label(cuda_pick_row,
                  text=(f"{len(tp.cuda_installs)} CUDA install(s) found" if tp.cuda_installs
                        else "No CUDA installs found"),
                  font=("TkSmallCaptionFont",)) \
            .grid(row=1, column=1, sticky="w", padx=4)
        # Inline warning shown when GGML_CUDA is ON but no toolkit was found.
        self._cuda_warning_var = tk.StringVar(value="")
        ttk.Label(cuda_pick_row, textvariable=self._cuda_warning_var,
                  foreground="#b00", font=("TkSmallCaptionFont",), wraplength=600) \
            .grid(row=2, column=0, columnspan=3, sticky="w", padx=4, pady=(2, 0))

        ttk.Label(lf, text="CUDACXX (nvcc):").grid(row=3, column=0, sticky="w", padx=6, pady=4)
        nvcc_vals = [inst.nvcc_path for inst in tp.cuda_installs]
        ttk.Combobox(lf, textvariable=self.var_cudacxx,
                     values=nvcc_vals).grid(row=3, column=1, sticky="ew", padx=4, columnspan=2)
        ttk.Label(lf, text="CUDA root:").grid(row=4, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(lf, textvariable=self.var_cuda_root).grid(row=4, column=1, sticky="ew", padx=4)
        ttk.Label(lf, text="(passed as CUDA_TOOLKIT_ROOT_DIR)",
                  font=("TkSmallCaptionFont",)).grid(row=4, column=2, sticky="w", padx=4)

        # ── Host compilers ──
        ttk.Label(lf, text="CC:").grid(row=5, column=0, sticky="w", padx=6, pady=(8, 4))
        ttk.Combobox(lf, textvariable=self.var_cc,
                     values=tp.cc_candidates).grid(row=5, column=1, sticky="ew", padx=4, columnspan=2)
        ttk.Label(lf, text="CXX:").grid(row=6, column=0, sticky="w", padx=6, pady=4)
        ttk.Combobox(lf, textvariable=self.var_cxx,
                     values=tp.cxx_candidates).grid(row=6, column=1, sticky="ew", padx=4, columnspan=2)

        # ── build-tool discovery / installers ──
        ttk.Label(lf, text="Build tools:").grid(row=7, column=0, sticky="nw", padx=6, pady=(8, 4))
        tools_frame = ttk.Frame(lf)
        tools_frame.grid(row=7, column=1, columnspan=2, sticky="ew", padx=4, pady=(8, 4))
        self._tool_install_frame = tools_frame
        self._rebuild_build_tool_rows()

        # ── cmake generator ──
        ttk.Label(lf, text="Generator:").grid(row=8, column=0, sticky="w", padx=6, pady=(8, 4))
        gen_values = [DEFAULT_GENERATOR_LABEL, "Ninja", "Unix Makefiles"]
        if sys.platform.startswith("win"):
            gen_values += ["Visual Studio 17 2022", "Visual Studio 16 2019", "NMake Makefiles"]
        elif sys.platform == "darwin":
            gen_values += ["Xcode"]
        ttk.Combobox(lf, textvariable=self.var_generator,
                     values=gen_values, width=24).grid(row=8, column=1, sticky="w", padx=4)
        hint = "cmake uses the platform default generator; Ninja is usually fastest"
        if self._toolchain.ninja_path:
            hint += " (detected)"
        else:
            hint += " (not installed)"
        ttk.Label(lf, text=hint + ".",
                  font=("TkSmallCaptionFont",)).grid(row=8, column=2, sticky="w", padx=4)

        ttk.Label(lf, text="Extra cmake args:").grid(row=9, column=0, sticky="w", padx=6, pady=(8, 4))
        ttk.Entry(lf, textvariable=self.var_extra_args).grid(row=9, column=1, sticky="ew", padx=4)
        ttk.Label(lf, text="(passed through verbatim)",
                  font=("TkSmallCaptionFont",)).grid(row=9, column=2, sticky="w", padx=4)

    # ── Flags grouped into LabelFrames ──────────────────────────────────
    def _build_section_flags(self, parent: ttk.Frame, row: int) -> None:
        # The outer LabelFrame is stable across backend switches; only its
        # children change. _populate_flag_groups builds the per-backend
        # group frames and widgets inside it.
        lf = ttk.LabelFrame(parent, text="CMake flags")
        lf.grid(row=row, column=0, sticky="ew", padx=8, pady=6)
        lf.columnconfigure(0, weight=1)
        self._flags_label_frame = lf
        self._populate_flag_groups()

    def _populate_flag_groups(self) -> None:
        """Build widgets for EVERY flag across EVERY group, regardless of
        backend. Called once at construction (and again on full-rebuild).

        Backend switches no longer rebuild widgets — they just toggle
        visibility via :meth:`_apply_all_flag_visibilities`. Building once
        eliminates the per-swap widget destruction/creation that was the
        primary source of the multi-second freeze users were seeing
        (≈100 widgets × ~5-10ms each = 1-2s of pure Tk churn per swap).
        """
        lf = self._flags_label_frame
        if lf is None or not lf.winfo_exists():
            return
        # Clear out any existing per-group frames (full rebuild path).
        for child in lf.winfo_children():
            child.destroy()
        self._flag_widgets = {}
        self._flag_help_widgets: dict[str, tk.Widget] = {}
        self._flag_label_widgets: dict[str, tk.Widget] = {}
        self._group_frames = {}

        # Ensure every flag has a Tk var bound (created once, reused on rebuilds).
        for flag in cf.FLAGS:
            if flag.key in self._flag_vars:
                continue
            initial = self._values_snapshot.get(flag.key, flag.default)
            if flag.type == cf.BOOL:
                var: tk.Variable = tk.BooleanVar(value=bool(initial))
            elif flag.type == cf.ENUM:
                var = tk.StringVar(value=str(initial))
            else:
                var = tk.StringVar(value=str(initial) if initial is not None else "")
            var.trace_add("write", lambda *_a, k=flag.key, v=var:
                          self._on_flag_changed(k, v))
            self._flag_vars[flag.key] = var

        # Group flags by group name across ALL backends so each widget gets
        # built exactly once. We use cf.GROUPS_ORDER for stable display order.
        flags_by_group: dict[str, list[cf.CMakeFlag]] = {}
        for flag in cf.FLAGS:
            flags_by_group.setdefault(flag.group, []).append(flag)
        ordered_groups = [g for g in cf.GROUPS_ORDER if g in flags_by_group]
        # Any group that wasn't in GROUPS_ORDER, append in catalogue order.
        for g in flags_by_group:
            if g not in ordered_groups:
                ordered_groups.append(g)

        for group_name in ordered_groups:
            flags = flags_by_group[group_name]
            grp = ttk.LabelFrame(lf, text=group_name)
            grp.pack(fill="x", padx=4, pady=4)
            grp.columnconfigure(1, weight=1)
            self._group_frames[group_name] = grp
            for i, flag in enumerate(flags):
                self._build_flag_widget(grp, flag, i)
        # Apply backend filter + visibility predicates at the end.
        self._apply_all_flag_visibilities()

    def _apply_all_flag_visibilities(self) -> None:
        """Single-pass widget visibility refresh.

        Handles two filters in one O(N) sweep:
          * Backend applicability — hide widgets whose ``flag.backends`` set
            doesn't include the currently-selected backend.
          * ``visible_when`` predicates — disable widgets whose dependency
            isn't satisfied (e.g. CUDA tuning knobs when GGML_CUDA=OFF).

        Also hides group LabelFrames whose flags are entirely backend-
        incompatible so the user doesn't see empty "HIP / ROCm options"
        when targeting ik_llama on a non-HIP system.

        Uses ``grid_remove``/``grid``/``pack_forget``/``pack`` to toggle —
        no widget destruction. This is what makes backend swap feel
        instant: ~100 grid operations vs ~200 widget create+destroy.
        """
        try:
            values = self._current_flag_values_dict()
        except Exception as exc:
            print(f"WARN: could not build flag-values snapshot: {exc}",
                  file=sys.stderr)
            return
        backend = self.var_backend.get()
        # Track which groups still have at least one visible flag.
        group_has_visible: dict[str, bool] = dict.fromkeys(self._group_frames, False)

        for flag in cf.FLAGS:
            w = self._flag_widgets.get(flag.key)
            if w is None:
                continue
            applies = flag.applies_to(backend)
            predicate_ok = True
            if flag.visible_when:
                try:
                    predicate_ok = bool(flag.visible_when(values))
                except Exception as exc:
                    print(f"WARN: visible_when predicate for {flag.key!r} raised: {exc}",
                          file=sys.stderr)
                    predicate_ok = True
            help_w = self._flag_help_widgets.get(flag.key)
            label_w = self._flag_label_widgets.get(flag.key)
            try:
                if not w.winfo_exists():
                    continue
                if applies:
                    # Backend applies — show widget; let visible_when control
                    # enabled/disabled state.
                    w.grid()
                    if help_w is not None:
                        help_w.grid()
                    if label_w is not None:
                        label_w.grid()
                    state = "normal" if predicate_ok else "disabled"
                    w.configure(state=state)
                    group_has_visible[flag.group] = True
                else:
                    # Backend doesn't apply — hide widget entirely.
                    w.grid_remove()
                    if help_w is not None:
                        help_w.grid_remove()
                    if label_w is not None:
                        label_w.grid_remove()
            except Exception:
                pass

        # Hide groups that have no applicable flags for the current backend.
        for group_name, grp in self._group_frames.items():
            try:
                if not grp.winfo_exists():
                    continue
                if group_has_visible.get(group_name, False):
                    grp.pack(fill="x", padx=4, pady=4)
                else:
                    grp.pack_forget()
            except Exception:
                pass

    def _build_flag_widget(self, parent: ttk.LabelFrame, flag: cf.CMakeFlag, row: int) -> None:
        """Create the main widget + companion help label + (for non-BOOL flags)
        a name label. Stash references so :meth:`_apply_all_flag_visibilities`
        can hide/show them together on backend swap.
        """
        var = self._flag_vars[flag.key]
        name_lbl: tk.Widget | None = None
        if flag.type == cf.BOOL:
            w = ttk.Checkbutton(parent, text=flag.label, variable=var)
            w.grid(row=row, column=0, sticky="w", padx=6, pady=2)
            help_lbl = ttk.Label(parent, text=flag.help, font=("TkSmallCaptionFont",))
            help_lbl.grid(row=row, column=1, sticky="w", padx=8, pady=2)
        elif flag.type == cf.ENUM:
            name_lbl = ttk.Label(parent, text=flag.label + ":")
            name_lbl.grid(row=row, column=0, sticky="w", padx=6, pady=2)
            w = ttk.Combobox(parent, textvariable=var,
                             values=flag.choices or [], state="readonly", width=14)
            w.grid(row=row, column=1, sticky="w", padx=4, pady=2)
            help_lbl = ttk.Label(parent, text=flag.help, font=("TkSmallCaptionFont",))
            help_lbl.grid(row=row, column=2, sticky="w", padx=8, pady=2)
        else:
            name_lbl = ttk.Label(parent, text=flag.label + ":")
            name_lbl.grid(row=row, column=0, sticky="w", padx=6, pady=2)
            w = ttk.Entry(parent, textvariable=var, width=24)
            w.grid(row=row, column=1, sticky="w", padx=4, pady=2)
            hint = flag.help
            if flag.placeholder:
                hint = f"{flag.help} (e.g. {flag.placeholder})"
            help_lbl = ttk.Label(parent, text=hint, font=("TkSmallCaptionFont",))
            help_lbl.grid(row=row, column=2, sticky="w", padx=8, pady=2)
        self._flag_widgets[flag.key] = w
        if flag.type == cf.STRING and flag.validate is not None:
            self._validate_flag_widget(flag.key)
        self._flag_help_widgets[flag.key] = help_lbl
        if name_lbl is not None:
            self._flag_label_widgets[flag.key] = name_lbl
        # Visibility is applied in a single pass at the end of
        # _populate_flag_groups, not per-widget — see _apply_all_flag_visibilities.

    def _apply_flag_visibility(self, flag: cf.CMakeFlag) -> None:
        w = self._flag_widgets.get(flag.key)
        if w is None:
            return
        visible = True
        if flag.visible_when:
            try:
                visible = flag.visible_when(self._current_flag_values_dict())
            except Exception as exc:
                # Predicates should be pure functions of flag values; if one
                # raises, surface it once rather than swallowing silently —
                # otherwise the bug hides forever and the flag stays visible.
                print(f"WARN: visible_when predicate for {flag.key!r} raised: {exc}",
                      file=sys.stderr)
                visible = True
        state = "normal" if visible else "disabled"
        try:
            if w.winfo_exists():
                w.configure(state=state)
        except Exception:
            pass

    # ── Live preview ────────────────────────────────────────────────────
    def _build_section_preview(self, parent: ttk.Frame, row: int) -> None:
        lf = ttk.LabelFrame(parent, text="Resolved cmake invocation (preview)")
        lf.grid(row=row, column=0, sticky="ew", padx=8, pady=6)
        lf.columnconfigure(0, weight=1)

        self._preview_text = tk.Text(lf, height=8, wrap="word",
                                     font=("TkFixedFont",), state="disabled")
        self._preview_text.grid(row=0, column=0, sticky="ew", padx=4, pady=4)

        btns = ttk.Frame(lf)
        btns.grid(row=1, column=0, sticky="e", padx=4, pady=(0, 4))
        ttk.Button(btns, text="Copy command",
                   command=self._on_copy_preview).pack(side="left", padx=2)
        ttk.Button(btns, text="Save as .sh…",
                   command=self._on_save_script).pack(side="left", padx=2)

    # ── Action bar + console ───────────────────────────────────────────
    def _build_action_bar(self, parent: ttk.Frame) -> None:
        ttk.Separator(parent, orient="horizontal").grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=8, pady=(2, 6)
        )
        bar = ttk.Frame(parent)
        bar.grid(row=3, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 0))
        bar.columnconfigure(2, weight=1)
        self._start_btn = ttk.Button(bar, text="▶ Start build",
                                     command=self._on_start_build)
        self._start_btn.grid(row=0, column=0, padx=2)
        self._cancel_btn = ttk.Button(bar, text="■ Cancel",
                                      command=self._on_cancel, state="disabled")
        self._cancel_btn.grid(row=0, column=1, padx=2)
        ttk.Label(bar, textvariable=self.var_status,
                  font=("TkDefaultFont", 10, "bold")) \
            .grid(row=0, column=2, padx=12, sticky="w")
        ttk.Checkbutton(bar, text="Auto-scroll",
                        variable=self.var_autoscroll) \
            .grid(row=0, column=3, padx=8)
        ttk.Button(bar, text="Clear log",
                   command=self._on_clear_console).grid(row=0, column=4, padx=2)
        ttk.Button(bar, text="Save log…",
                   command=self._on_save_log).grid(row=0, column=5, padx=2)

    def _build_console(self, parent: ttk.Frame) -> None:
        cf_frame = ttk.LabelFrame(parent, text="Build output")
        cf_frame.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=8, pady=(4, 8))
        parent.rowconfigure(4, weight=1)
        cf_frame.rowconfigure(0, weight=1)
        cf_frame.columnconfigure(0, weight=1)

        self._console = tk.Text(cf_frame, wrap="none", height=8,
                                font=("TkFixedFont",), state="disabled")
        self._console.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        sb_y = ttk.Scrollbar(cf_frame, orient="vertical", command=self._console.yview)
        sb_x = ttk.Scrollbar(cf_frame, orient="horizontal", command=self._console.xview)
        self._console.configure(yscrollcommand=sb_y.set, xscrollcommand=sb_x.set)
        sb_y.grid(row=0, column=1, sticky="ns")
        sb_x.grid(row=1, column=0, sticky="ew")

        # Tags for stage headers.
        self._console.tag_configure("stage", foreground="#0a6", font=("TkFixedFont", 10, "bold"))
        self._console.tag_configure("error", foreground="#c00")
        self._console.tag_configure("ok",    foreground="#0a6")

        # Replay any history saved across rebuilds.
        if self._console_buffer:
            self._console.configure(state="normal")
            for line in self._console_buffer[-CONSOLE_MAX_LINES:]:
                self._console.insert("end", line + "\n")
            self._console.see("end")
            self._console.configure(state="disabled")

    # ─────────────────────────────────────────────────────────────────────
    # Event handlers
    # ─────────────────────────────────────────────────────────────────────
    def _on_backend_changed(self) -> None:
        # Fired by the radio button's command=. The radio button is a child of
        # the tab's content frame which _build_ui destroys; doing the rebuild
        # synchronously freezes Tk. Defer until the event has fully unwound.
        self._sync_launcher_backend_from_build()
        self._schedule_rebuild()

    def _on_launcher_backend_changed(self, *_a) -> None:
        if self._syncing_backend_selection:
            return
        try:
            new_backend = self.launcher.backend_selection.get()
        except Exception:
            return
        if new_backend and new_backend != self.var_backend.get():
            self._syncing_backend_selection = True
            try:
                self._set_var_if_changed(self.var_backend, new_backend)
            finally:
                self._syncing_backend_selection = False
            # Setting var_backend programmatically doesn't fire the radio
            # button's command callback, so we need to schedule the rebuild
            # ourselves — otherwise flag groups remain stuck on the old backend.
            self._schedule_rebuild()
        if new_backend:
            self._sync_source_dir_from_launcher(new_backend, prefer_current=False)
            self._refresh_saved_configs_dropdown()

    def _schedule_rebuild(self, *, full: bool = False) -> None:
        """Coalesce rapid rebuild requests onto a single after_idle callback.

        ``full=True`` upgrades a pending light rebuild to a full one. A
        previously-scheduled full rebuild stays full.
        """
        if full:
            self._rebuild_full_pending = True
        if self._rebuild_pending:
            return
        self._rebuild_pending = True
        self.root.after_idle(self._dispatch_rebuild)

    def _dispatch_rebuild(self) -> None:
        do_full = getattr(self, "_rebuild_full_pending", False)
        self._rebuild_full_pending = False
        if do_full:
            # _full_rebuild manages _rebuild_pending itself? No — set it here.
            self._rebuild_pending = False
            self._full_rebuild()
        else:
            self._do_rebuild()

    def _do_rebuild(self) -> None:
        """Lightweight backend swap. No widget destruction or creation —
        all 112 flag widgets exist permanently in the DOM after the first
        construction. We just:

          1. Update flag values for the new backend's defaults
             (suspended traces, so no per-write cascade).
          2. Toggle widget visibility via grid/grid_remove based on which
             flags apply to the new backend.

        This is the fix for the multi-second freeze users were seeing.
        Previous implementation destroyed and re-created ~100 widgets
        each swap (~500ms-2s of pure Tk churn). This path is ~10-50ms.

        See :meth:`_full_rebuild` for the slightly heavier path used
        when the CUDA arch picker also needs to be refreshed (the
        ``show_deprecated`` toggle and saved-config load).
        """
        self._rebuild_pending = False
        try:
            self._suspend_traces = True
            try:
                self._apply_autodetect_defaults()
                self._sync_flag_widgets_from_values()
            finally:
                self._suspend_traces = False
            # The cheap part: just re-evaluate per-flag visibility against
            # the new backend. No destroy/create.
            self._apply_all_flag_visibilities()
            self._schedule_preview_refresh()
            if self.var_auto_check_updates.get():
                self.check_for_updates(do_fetch=False)
        except Exception as exc:
            print(f"WARN: Build tab rebuild failed: {exc}", file=sys.stderr)

    def _full_rebuild(self) -> None:
        """Surgical "full" rebuild for the two paths that need more than a
        backend swap: the show-deprecated arch toggle and saved-config
        load. Despite the name, this no longer tears the tab down — it
        just runs the same lightweight path as :meth:`_do_rebuild` and
        then refreshes the CUDA arch picker (the only sub-section whose
        widget composition can actually change at runtime).

        The previous implementation called ``_build_ui`` which destroyed
        and recreated every widget in the tab (~350 widgets including
        ~112 flag widgets). On Linux/X11 that consistently spent
        500ms-2s in pure Tk widget churn; on a slow display server it
        could appear to hang. Since detach-to-Toplevel was removed (see
        constructor comment), ``_tab_frame`` is stable for the tab's
        lifetime — there is no legitimate need to recreate the heavy
        widget tree.
        """
        try:
            self._rebuild_pending = False
            self._suspend_traces = True
            try:
                self._apply_autodetect_defaults()
                self._sync_flag_widgets_from_values()
            finally:
                self._suspend_traces = False
            # Flag visibility tracks the (possibly newly-loaded) backend.
            self._apply_all_flag_visibilities()
            # Arch picker may need to add/remove deprecated rows. This
            # destroys ~50 widgets in the worst case (vs ~350 with the
            # old tab-wide rebuild). _arch_check_vars survive — the
            # picker reuses any matching BooleanVar by `cc` key.
            self._rebuild_arch_picker_only()
            self._update_source_dir_status()
            self._schedule_preview_refresh()
            if self.var_auto_check_updates.get():
                self.check_for_updates(do_fetch=False)
        except Exception as exc:
            print(f"WARN: Build tab full rebuild failed: {exc}", file=sys.stderr)

    def _rebuild_arch_picker_only(self) -> None:
        """Destroy and recreate the CUDA arch-picker grid in place.

        Only the per-arch checkbuttons + family labels + ``+all <family>``
        buttons live in this grid (~50 widgets total). The owning
        ``_archs_grid_frame`` LabelFrame is preserved so we don't have to
        touch the surrounding ``arch_frame`` layout.

        ``_arch_check_vars`` is NOT cleared — ``_build_cuda_arch_picker``
        reuses any existing BooleanVar keyed by ``cc``, so checkbox
        state survives the rebuild for archs that are still visible
        (which is what users expect when they toggle "Show deprecated").
        """
        grid = self._archs_grid_frame
        if grid is None or not grid.winfo_exists():
            return
        for child in grid.winfo_children():
            try:
                child.destroy()
            except Exception:
                pass
        self._build_cuda_arch_picker(grid)

    def _on_launcher_current_dir_changed(self, *_a) -> None:
        try:
            backend = self.launcher.backend_selection.get()
        except Exception:
            return
        self._sync_source_dir_from_launcher(backend, prefer_current=True)

    def _on_launcher_backend_dir_var_changed(self, backend: str) -> None:
        self._sync_source_dir_from_launcher(backend, prefer_current=False)

    def _on_build_source_dir_changed(self, *_a) -> None:
        self._update_source_dir_status()
        self._sync_launcher_dir_from_source()

    def _on_build_backend_var_changed(self, *_a) -> None:
        self._update_source_dir_status()
        self._sync_launcher_backend_from_build()
        self._sync_source_dir_from_launcher(self.var_backend.get(), prefer_current=False)
        self._refresh_saved_configs_dropdown()

    @staticmethod
    def _set_var_if_changed(var: tk.Variable, value: str) -> None:
        try:
            if var.get() != value:
                var.set(value)
        except Exception:
            pass

    def _launcher_backend_dir_var(self, backend: str) -> tk.Variable | None:
        try:
            if backend == "ik_llama":
                return self.launcher.ik_llama_dir
            return self.launcher.llama_cpp_dir
        except Exception:
            return None

    def _launcher_backend_settings_key(self, backend: str) -> str:
        return "last_ik_llama_dir" if backend == "ik_llama" else "last_llama_cpp_dir"

    def _sync_launcher_backend_from_build(self) -> None:
        if self._syncing_backend_selection:
            return
        backend = self.var_backend.get() or "llama.cpp"
        try:
            if self.launcher.backend_selection.get() == backend:
                return
        except Exception:
            return
        self._syncing_backend_selection = True
        try:
            self._set_var_if_changed(self.launcher.backend_selection, backend)
        finally:
            self._syncing_backend_selection = False

    def _sync_source_dir_from_launcher(self, backend: str, *, prefer_current: bool) -> None:
        if self._syncing_backend_dirs or backend != self.var_backend.get():
            return
        try:
            if prefer_current and self.launcher.backend_selection.get() == backend:
                source_dir = self.launcher.current_backend_dir.get()
            else:
                dir_var = self._launcher_backend_dir_var(backend)
                if dir_var is None:
                    return
                source_dir = dir_var.get()
        except Exception:
            return
        self._syncing_backend_dirs = True
        try:
            self._set_var_if_changed(self.var_source_dir, source_dir)
        finally:
            self._syncing_backend_dirs = False

    def _sync_launcher_dir_from_source(self) -> None:
        if self._syncing_backend_dirs:
            return
        backend = self.var_backend.get() or "llama.cpp"
        source_dir = self.var_source_dir.get()
        dir_var = self._launcher_backend_dir_var(backend)
        if dir_var is None:
            return
        self._syncing_backend_dirs = True
        try:
            self._set_var_if_changed(dir_var, source_dir)
            try:
                self.launcher.app_settings[self._launcher_backend_settings_key(backend)] = source_dir
            except Exception:
                pass
            try:
                if self.launcher.backend_selection.get() == backend:
                    self._set_var_if_changed(self.launcher.current_backend_dir, source_dir)
            except Exception:
                pass
        finally:
            self._syncing_backend_dirs = False

    def _on_browse_source(self) -> None:
        start = self.var_source_dir.get().strip() or str(Path.home())
        directory = filedialog.askdirectory(
            initialdir=start if os.path.isdir(start) else str(Path.home()),
            title="Select source directory",
        )
        if directory:
            self.var_source_dir.set(directory)
            self.check_for_updates(do_fetch=False)

    def _on_use_backend_dir(self) -> None:
        try:
            d = self.launcher.current_backend_dir.get()
        except Exception:
            d = ""
        if d:
            self.var_source_dir.set(d)
            self.check_for_updates(do_fetch=False)

    def _update_source_dir_status(self) -> None:
        """Update the source-dir status line (and its color) based on what
        the path currently points at:

          * empty                     → blank
          * doesn't exist             → "will be auto-cloned from <url>" (neutral)
          * exists, not a git repo    → "exists but not a git repository" (red)
          * exists, is a git repo     → "✓ Git repository detected" (neutral)

        Called by traces on var_source_dir and var_backend, and once at
        section-build time.
        """
        if not hasattr(self, "_source_status_label"):
            return  # label not built yet
        src = self.var_source_dir.get().strip()
        if not src:
            self.var_source_status.set("")
            self._source_status_label.configure(fg="#666")
            return
        if not os.path.isdir(src):
            upstream = UPSTREAMS.get(self.var_backend.get(), "")
            url_suffix = f" from {upstream}" if upstream else ""
            self.var_source_status.set(
                f"⚠ Directory does not exist — will be auto-cloned{url_suffix} on Start build."
            )
            self._source_status_label.configure(fg="#b08000")
            return
        git_path = os.path.join(src, ".git")
        is_git = os.path.exists(git_path)  # exists() covers worktree (.git is a file)
        if not is_git:
            self.var_source_status.set(
                "⚠ Directory exists but is not a git repository. "
                "Pick a different folder or delete it to allow auto-clone."
            )
            self._source_status_label.configure(fg="#b00000")
        else:
            self.var_source_status.set("✓ Git repository detected.")
            self._source_status_label.configure(fg="#0a6")

    def _start_initial_toolchain_probe(self) -> None:
        """Deferred startup probe: kicks off the same worker that the
        "Refresh toolchain" button uses, but without the console banner
        (we don't want to flood the console on every app start). Called
        from ``__init__`` via ``after_idle`` so the UI renders first.
        """
        print(
            f"DEBUG: _start_initial_toolchain_probe FIRED "
            f"(t={time.perf_counter():.3f}) [after_idle reached]",
            file=sys.stderr,
        )
        if self._toolchain_refresh_in_flight:
            return
        self._toolchain_refresh_in_flight = True
        threading.Thread(
            target=self._refresh_toolchain_worker,
            name="ToolchainProbe",
            daemon=True,
        ).start()
        if self._toolchain_refresh_after_id is None:
            try:
                self._toolchain_refresh_after_id = self.root.after(
                    200, self._drain_toolchain_refresh
                )
            except Exception:
                # If Tk is no longer accepting after() (teardown race), the
                # worker will just exit when it finishes; no UI update needed.
                pass

    def _on_refresh_toolchain(self) -> None:
        """Re-scan CUDA toolkits, compilers, etc. Runs the probe on a
        background thread because ``probe_toolchain`` makes multiple
        subprocess calls (nvcc/cmake --version, etc.) that can easily
        spend 5+ seconds — running it on the Tk main thread would freeze
        the entire app.

        Uses a Queue + Tk after() polling pattern (rather than calling
        root.after() directly from the worker) because Tkinter's
        createcommand is not thread-safe and the direct-call pattern
        breaks during shutdown / teardown.
        """
        if self._toolchain_refresh_in_flight:
            return
        self._toolchain_refresh_in_flight = True
        self._append_console("Re-probing toolchain…\n", tag="stage")
        threading.Thread(
            target=self._refresh_toolchain_worker,
            name="ToolchainProbe",
            daemon=True,
        ).start()
        # Start polling the queue from the main thread.
        if self._toolchain_refresh_after_id is None:
            self._toolchain_refresh_after_id = self.root.after(
                200, self._drain_toolchain_refresh
            )

    def _refresh_toolchain_worker(self) -> None:
        """Background-thread worker. Posts a single result onto the
        toolchain-refresh queue and exits. No direct Tk calls."""
        _w_start = time.perf_counter()
        print(f"DEBUG: toolchain-worker START (t={_w_start:.3f})", file=sys.stderr)
        try:
            _t = time.perf_counter()
            new_tc = detection.probe_toolchain()
            print(
                f"DEBUG: toolchain-worker probe_toolchain "
                f"{(time.perf_counter() - _t) * 1000.0:.1f} ms",
                file=sys.stderr,
            )
            _t = time.perf_counter()
            cuda_infos = detection.detect_cuda_archs()
            print(
                f"DEBUG: toolchain-worker detect_cuda_archs "
                f"{(time.perf_counter() - _t) * 1000.0:.1f} ms",
                file=sys.stderr,
            )
            avx512 = self._cpu_has_avx512()
            _w_total = (time.perf_counter() - _w_start) * 1000.0
            print(
                f"DEBUG: toolchain-worker END total {_w_total:.1f} ms; "
                f"queueing result",
                file=sys.stderr,
            )
            self._toolchain_refresh_queue.put((
                "ok",
                {
                    "toolchain": new_tc,
                    "cuda_infos": cuda_infos,
                    "avx512": avx512,
                },
            ))
        except Exception as exc:
            self._toolchain_refresh_queue.put(("error", exc))

    def _drain_toolchain_refresh(self) -> None:
        """Main-thread poll: pull a finished probe off the queue and apply
        it. Reschedules itself while the worker is still in-flight."""
        self._toolchain_refresh_after_id = None
        try:
            kind, payload = self._toolchain_refresh_queue.get_nowait()
        except queue.Empty:
            if self._toolchain_refresh_in_flight:
                self._toolchain_refresh_after_id = self.root.after(
                    200, self._drain_toolchain_refresh
                )
            return
        if kind == "ok":
            self._refresh_toolchain_finish(payload)
        else:
            self._toolchain_refresh_in_flight = False
            self._append_console(f"Toolchain probe failed: {payload}\n", tag="error")

    def _refresh_toolchain_finish(self, payload) -> None:
        """Main-thread callback: apply the freshly-probed toolchain in
        place, without rebuilding the whole tab. Only the CUDA-install
        combobox values, the CUDACXX combobox values, and the toolchain
        hint label need updating.

        Also runs on the deferred startup probe (see
        ``_start_initial_toolchain_probe``), so it owns the work
        ``__init__`` used to do synchronously: seeding compiler defaults
        from candidates, syncing CUDA root from CUDACXX, and applying
        the auto-detect flag preset. On the *first* completion we also
        push the freshly-detected snapshot into the flag widgets, since
        ``_populate_flag_groups`` rendered them before the probe data
        was available.
        """
        _f_start = time.perf_counter()
        print(
            f"DEBUG: _refresh_toolchain_finish ENTER (t={_f_start:.3f})",
            file=sys.stderr,
        )
        if isinstance(payload, dict):
            new_tc = payload.get("toolchain")
            worker_cuda_infos = payload.get("cuda_infos")
            worker_avx512 = bool(payload.get("avx512", False))
        else:
            new_tc = payload
            worker_cuda_infos = None
            worker_avx512 = self._cpu_has_avx512()
        if new_tc is None:
            self._toolchain_refresh_in_flight = False
            return
        is_first = not getattr(self, "_initial_probe_finished", False)
        self._toolchain = new_tc
        if worker_cuda_infos is not None:
            self._cuda_arch_cache = list(worker_cuda_infos)
            self._cuda_arch_cache_avx512 = worker_avx512
        else:
            self._cuda_arch_cache = detection.cuda_archs_from_gpu_info(
                getattr(self.launcher, "gpu_info", None)
            )
            self._cuda_arch_cache_avx512 = worker_avx512
        _t = time.perf_counter()
        self._refresh_toolchain_hint()
        self._refresh_jobs_hint()
        self._seed_toolchain_defaults()
        # Tk's trace on var_cudacxx will have fired during the seed
        # above, but call directly so the initial state stays
        # deterministic even if the trace timing is debounced or queued.
        self._on_cudacxx_changed()
        print(
            f"DEBUG: tc-finish: seed+cudacxx took "
            f"{(time.perf_counter() - _t) * 1000.0:.1f} ms",
            file=sys.stderr,
        )
        # Snapshot defaults follow probed data (cuda_available, ccache,
        # avx512 inferred from /proc/cpuinfo). Idempotent merge — user
        # overrides in _values_snapshot win, so refresh-button presses
        # don't clobber the user's customisations.
        _t = time.perf_counter()
        try:
            self._apply_autodetect_defaults()
        except Exception as exc:
            print(f"DEBUG: _apply_autodetect_defaults after probe failed: {exc}",
                  file=sys.stderr)
        print(
            f"DEBUG: tc-finish: _apply_autodetect_defaults took "
            f"{(time.perf_counter() - _t) * 1000.0:.1f} ms",
            file=sys.stderr,
        )
        if is_first:
            # First probe ever — widgets were rendered with empty defaults
            # because the probe wasn't ready yet. Push the autodetect
            # snapshot into them now, with traces suspended so we don't
            # fire ~100 per-flag visibility refreshes during the bulk
            # sync. A single full sweep at the end does the same work
            # in O(N) rather than O(N²).
            _t = time.perf_counter()
            self._suspend_traces = True
            try:
                self._sync_flag_widgets_from_values()
            finally:
                self._suspend_traces = False
            print(
                f"DEBUG: tc-finish: _sync_flag_widgets_from_values took "
                f"{(time.perf_counter() - _t) * 1000.0:.1f} ms",
                file=sys.stderr,
            )
            _t = time.perf_counter()
            try:
                self._apply_all_flag_visibilities()
            except Exception as exc:
                print(f"DEBUG: post-probe visibility sweep failed: {exc}",
                      file=sys.stderr)
            print(
                f"DEBUG: tc-finish: _apply_all_flag_visibilities took "
                f"{(time.perf_counter() - _t) * 1000.0:.1f} ms",
                file=sys.stderr,
            )
            self._initial_probe_finished = True
        print(
            f"DEBUG: _refresh_toolchain_finish EXIT total "
            f"{(time.perf_counter() - _f_start) * 1000.0:.1f} ms",
            file=sys.stderr,
        )
        # In-place refresh of the CUDA-install combobox + label map.
        if hasattr(self, "_cuda_pick_combo") and self._cuda_pick_combo.winfo_exists():
            self._cuda_install_by_label = {
                inst.label(): inst for inst in new_tc.cuda_installs
            }
            labels = list(self._cuda_install_by_label.keys())
            if labels:
                labels.append("Custom path…")
            try:
                self._cuda_pick_combo["values"] = labels
                self._cuda_pick_combo["state"] = "readonly" if labels else "normal"
            except Exception:
                pass
        self._rebuild_build_tool_rows()
        # Refresh compiler comboboxes' value lists in place if we can find them.
        # (They're tied to var_cc/var_cxx; we don't track the widget refs, but
        # the values shown only matter on next dropdown open — Tk reads them
        # fresh from the combobox values config.)
        self._toolchain_refresh_in_flight = False
        self._append_console(
            f"Toolchain re-probed: {len(new_tc.cuda_installs)} CUDA install(s), "
            f"{len(new_tc.cc_candidates)} compiler(s).\n",
            tag="ok",
        )
        # Update source-dir status if it now depends on a re-probed git binary.
        self._update_source_dir_status()

    def _on_cuda_install_picked(self, *_args) -> None:
        """Combobox callback: a label like 'CUDA 12.8 · /usr/local/cuda-12.8'
        was selected — look it up and set CUDACXX + root accordingly."""
        label = self.var_cuda_pick.get()
        if label == "Custom path…":
            path = filedialog.askopenfilename(
                title="Select nvcc",
                initialdir="/usr/local",
                filetypes=[("nvcc", "nvcc nvcc.exe"), ("All files", "*.*")],
            )
            if path:
                self.var_cudacxx.set(path)
            else:
                # User cancelled — restore previous label.
                self._on_cudacxx_changed()
            return
        inst = self._cuda_install_by_label.get(label)
        if inst is None:
            return
        self.var_cudacxx.set(inst.nvcc_path)
        self.var_cuda_root.set(inst.root_dir)

    def _on_rescan_cuda_installs(self) -> None:
        self._on_refresh_toolchain()

    def _on_autodetect_archs(self) -> None:
        # Surface the diagnostic in the build console too — much easier
        # to debug "nothing happened" reports than just a popup.
        self._append_console("\nDetect CUDA architectures…\n", tag="stage")
        infos = detection.cuda_archs_from_gpu_info(
            getattr(self.launcher, "gpu_info", None)
        )
        if infos:
            self._append_console("  using launcher GPU info cache\n")
        else:
            infos = detection.detect_cuda_archs(allow_torch_fallback=False)
            if infos:
                self._append_console("  using nvidia-smi\n")
        if not infos:
            self._append_console(
                "  nvidia-smi did not return architectures; running torch "
                "fallback in the background…\n",
                tag="stage",
            )
            self.var_cuda_archs.set("")
            self._cuda_arch_cache = None
            self._on_refresh_toolchain()
            return
        for info in infos:
            self._append_console(
                f"  GPU{info.index}: {info.name}  (sm_{info.compute_capability.replace('.', '')}, "
                f"family={info.family or 'unknown'})\n"
            )
        # Replace whatever's in the field with just the detected GPUs.
        value = detection.archs_to_cmake_value(
            infos, prefer_a_variant=self.var_prefer_a_variant.get()
        )
        self.var_cuda_archs.set(value)
        self._sync_arch_pickers_from_value()
        self._append_console(
            f"  CMAKE_CUDA_ARCHITECTURES = {value}\n", tag="ok",
        )

    def _build_cuda_arch_picker(self, parent: ttk.LabelFrame) -> None:
        """Render a grouped grid of CUDA-arch checkboxes (Turing → Blackwell
        by default; Kepler/Maxwell/Pascal/Volta appear when "Show deprecated
        archs" is on).

        Each ticked box adds its tokens to var_cuda_archs; unticking removes
        them. A '+all <family>' button per row flips the entire family.
        """
        # Group archs by family in catalogue order, optionally hiding the
        # deprecated families.
        show_deprecated = self.var_show_deprecated_archs.get()
        families: dict[str, list[detection.KnownArch]] = {}
        for k in detection.KNOWN_CUDA_ARCHS:
            if k.deprecated and not show_deprecated:
                continue
            families.setdefault(k.family, []).append(k)

        family_order = [f for f in detection.all_families() if f in families]

        for row_idx, family in enumerate(family_order):
            row_frame = ttk.Frame(parent)
            row_frame.grid(row=row_idx, column=0, sticky="ew", padx=4, pady=2)
            row_frame.columnconfigure(1, weight=1)
            is_deprecated_family = detection.family_has_only_deprecated(family)
            family_label = family + (" (deprecated)" if is_deprecated_family else "")
            # Dim the label color for deprecated families to make their
            # status visually obvious in addition to the text suffix.
            label_widget = ttk.Label(
                row_frame, text=f"{family_label}:",
                font=("TkDefaultFont", 9, "bold"),
                width=18, anchor="e",
            )
            if is_deprecated_family:
                try:
                    label_widget.configure(foreground="#888")
                except Exception:
                    pass
            label_widget.grid(row=0, column=0, sticky="w", padx=(0, 6))
            boxes = ttk.Frame(row_frame)
            boxes.grid(row=0, column=1, sticky="w")
            for col, k in enumerate(families[family]):
                var = self._arch_check_vars.get(k.cc)
                if var is None:
                    var = tk.BooleanVar(value=False)
                    self._arch_check_vars[k.cc] = var
                base = k.cc.split(".")
                label = f"sm_{base[0]}{base[1]}"
                # Tag -a / -f capability on the label so users can see at a glance.
                if k.has_a_variant or k.has_f_variant:
                    suffix_bits = []
                    if k.has_a_variant:
                        suffix_bits.append("-a")
                    if k.has_f_variant:
                        suffix_bits.append("-f")
                    label = f"{label} ({'/'.join(suffix_bits)})"
                cb = ttk.Checkbutton(
                    boxes, text=label, variable=var,
                    command=lambda cc=k.cc: self._on_arch_check_toggle(cc),
                )
                cb.grid(row=0, column=col, sticky="w", padx=2, pady=1)
            ttk.Button(row_frame, text=f"+all {family}",
                       width=14,
                       command=lambda fam=family: self._on_add_family(fam)) \
                .grid(row=0, column=2, sticky="e", padx=(8, 0))
        # Sync the initial check state from whatever's in the entry.
        self._sync_arch_pickers_from_value()

    def _on_show_deprecated_toggle(self) -> None:
        """Re-render the arch picker when the deprecated-archs toggle flips.

        Uses the full-rebuild path because the arch picker lives in the
        env section, not the flag section.
        """
        self._schedule_rebuild(full=True)

    def _on_arch_check_toggle(self, cc: str) -> None:
        if self._arch_sync_in_progress:
            return
        var = self._arch_check_vars.get(cc)
        if var is None:
            return
        tokens_to_add: list[str] = []
        tokens_to_remove: list[str] = []
        base = "".join(cc.split("."))
        plain = f"{base}-real"
        known = detection.known_arch_for(cc)
        a_token = f"{base}a-real" if (known and known.has_a_variant) else None
        f_token = f"{base}f-real" if (known and known.has_f_variant) else None
        if var.get():
            if self.var_prefer_a_variant.get() and a_token:
                tokens_to_add.append(a_token)
            if self.var_prefer_f_variant.get() and f_token:
                tokens_to_add.append(f_token)
            tokens_to_add.append(plain)
        else:
            tokens_to_remove.append(plain)
            if a_token:
                tokens_to_remove.append(a_token)
            if f_token:
                tokens_to_remove.append(f_token)
        current = [t.strip() for t in self.var_cuda_archs.get().split(";") if t.strip()]
        out: list[str] = []
        seen: set[str] = set()
        for t in current:
            if t in tokens_to_remove or t in seen:
                continue
            seen.add(t)
            out.append(t)
        for t in tokens_to_add:
            if t not in seen:
                seen.add(t)
                out.append(t)
        self._arch_sync_in_progress = True
        try:
            self.var_cuda_archs.set(";".join(out))
        finally:
            self._arch_sync_in_progress = False

    def _on_add_family(self, family: str) -> None:
        """Add every arch in a generation family to the field, honoring
        the -a/-f and deprecated-archs toggles."""
        addition = detection.family_to_tokens(
            family,
            prefer_a_variant=self.var_prefer_a_variant.get(),
            prefer_f_variant=self.var_prefer_f_variant.get(),
            include_deprecated=self.var_show_deprecated_archs.get(),
        )
        merged = detection.merge_arch_tokens(self.var_cuda_archs.get(), addition)
        self.var_cuda_archs.set(merged)

    def _sync_arch_pickers_from_value(self) -> None:
        """Update picker checkboxes to reflect whatever tokens are in the
        field. Tolerates user-typed tokens we don't know about."""
        if self._arch_sync_in_progress:
            return
        if not self._arch_check_vars:
            return
        present_ccs: set[str] = set()
        for raw in self.var_cuda_archs.get().split(";"):
            tok = raw.strip()
            if not tok:
                continue
            m = _ARCH_TOKEN_RE.match(tok)
            if not m:
                continue
            base = m.group("base")
            try:
                cc = f"{int(base) // 10}.{int(base) % 10}"
            except ValueError:
                continue
            present_ccs.add(cc)
        self._arch_sync_in_progress = True
        try:
            for cc, var in self._arch_check_vars.items():
                want = cc in present_ccs
                if var.get() != want:
                    var.set(want)
        finally:
            self._arch_sync_in_progress = False

    def _on_variant_toggle(self) -> None:
        """Toggle preference for the ``-a`` and/or ``-f`` arch variants.
        Rewrites the current arch field: each present compute-capability
        gets its plain ``-real`` token plus the ``-a`` and ``-f`` siblings
        the user's toggles request. Leaves user-typed tokens we don't
        recognize untouched."""
        prefer_a = self.var_prefer_a_variant.get()
        prefer_f = self.var_prefer_f_variant.get()
        current = self.var_cuda_archs.get().strip()
        if not current:
            return
        seen_ccs: set[str] = set()
        passthrough: list[str] = []
        for raw in current.split(";"):
            tok = raw.strip()
            if not tok:
                continue
            m = _ARCH_TOKEN_RE.match(tok)
            if not m:
                if tok not in passthrough:
                    passthrough.append(tok)
                continue
            base = m.group("base")
            try:
                # Handle 2-digit (sm_86) and 3-digit (sm_120) bases.
                if len(base) == 2:
                    major = int(base[0])
                    minor = int(base[1])
                else:
                    major = int(base[:2])
                    minor = int(base[2:])
                cc = f"{major}.{minor}"
            except ValueError:
                continue
            seen_ccs.add(cc)
        # Now reconstruct.
        out: list[str] = []
        emitted: set[str] = set()
        for cc in seen_ccs:
            base = "".join(cc.split("."))
            known = detection.known_arch_for(cc)
            if prefer_a and known and known.has_a_variant:
                t = f"{base}a-real"
                if t not in emitted:
                    emitted.add(t)
                    out.append(t)
            if prefer_f and known and known.has_f_variant:
                t = f"{base}f-real"
                if t not in emitted:
                    emitted.add(t)
                    out.append(t)
            t = f"{base}-real"
            if t not in emitted:
                emitted.add(t)
                out.append(t)
        for t in passthrough:
            if t not in emitted:
                emitted.add(t)
                out.append(t)
        self.var_cuda_archs.set(";".join(out))
        self._sync_arch_pickers_from_value()

    def _on_apply_autodetect(self) -> None:
        self._suspend_traces = True
        try:
            self._reset_to_autodetect()
            self._sync_flag_widgets_from_values()
        finally:
            self._suspend_traces = False
        self._append_console("Applied auto-detected preset (reset overrides).\n", tag="stage")
        self._schedule_preview_refresh()

    def _ensure_validation_style(self) -> None:
        """Define the 'Invalid.TEntry' style once. Tints an entry's text and
        field background red so a bad value (e.g. a non-power-of-two DMMV
        x-stride) is obvious as the user types."""
        if getattr(self, "_validation_style_ready", False):
            return
        try:
            style = ttk.Style()
            style.configure("Invalid.TEntry",
                            foreground="#b00020", fieldbackground="#fde8e8")
            self._validation_style_ready = True
        except Exception:
            # Theming can fail on exotic Tk builds — degrade to no live tint;
            # the start-build gate still blocks invalid values.
            self._validation_style_ready = False

    def _validate_flag_widget(self, key: str) -> None:
        """Re-check one STRING flag's value and tint its entry if invalid.

        Tracks the live invalid set in ``self._invalid_flags`` purely for the
        visual cue; the authoritative block lives in ``_on_start_build`` via
        ``cf.validate_values``."""
        flag = cf.get_flag(key)
        if flag is None or flag.validate is None or flag.type != cf.STRING:
            return
        w = self._flag_widgets.get(key)
        if w is None:
            return
        try:
            if not w.winfo_exists():
                return
        except Exception:
            return
        var = self._flag_vars.get(key)
        value = var.get() if var is not None else ""
        msg = cf.validate_flag_value(flag, value)
        invalid = getattr(self, "_invalid_flags", None)
        if invalid is None:
            invalid = self._invalid_flags = set()
        if msg:
            self._ensure_validation_style()
            invalid.add(key)
            try:
                w.configure(style="Invalid.TEntry")
            except Exception:
                pass
        else:
            invalid.discard(key)
            try:
                w.configure(style="TEntry")
            except Exception:
                pass

    def _on_flag_changed(self, key: str, var: tk.Variable) -> None:
        # Bulk-write paths suspend the trace handler so we don't fire O(N²)
        # visibility refreshes when seeding defaults.
        if self._suspend_traces:
            return
        # Live-validate STRING flags that carry a validator (e.g. DMMV x-stride
        # must be a power of two). Cheap — only runs for the one changed key.
        self._validate_flag_widget(key)
        # Update visibility chain (a flag may gate another flag). Only the
        # dependents of the changed key need a refresh; a full O(N) sweep
        # on every checkbox click was visibly laggy with 112 flags.
        dependents = self._visible_when_dependents_for(key)
        if dependents is None:
            # Couldn't build the map — fall back to full sweep.
            self._apply_all_flag_visibilities()
        else:
            self._refresh_flag_visibilities(dependents)
        self._schedule_preview_refresh()

    def _visible_when_dependents_for(self, source_key: str) -> set[str] | None:
        """Return the set of flag keys whose ``visible_when`` reads
        ``source_key``. Builds the dependents map lazily on first call.

        Returns ``None`` if the map can't be built (e.g. a predicate
        raises during introspection) — the caller should fall back to a
        full sweep.
        """
        cache = getattr(self, "_visible_when_dependents_cache", None)
        if cache is None:
            cache = self._build_visible_when_dependents_map()
            self._visible_when_dependents_cache = cache
        if cache is False:
            return None
        return cache.get(source_key, set())

    def _build_visible_when_dependents_map(self):
        """Run each flag's ``visible_when`` once against a tracking dict
        that records which keys it inspects. Invert that into a
        ``{source_key: {dependent_flag_keys}}`` map.

        Returns ``False`` if any predicate raised — the caller switches
        to a full-sweep fallback rather than risk silently dropping a
        chained-visibility refresh.
        """
        class _TrackingDict(dict):
            __slots__ = ("touched",)

            def __init__(self, base):
                super().__init__(base)
                self.touched = set()

            def __getitem__(self, k):
                self.touched.add(k)
                return super().__getitem__(k)

            def __contains__(self, k):
                self.touched.add(k)
                return super().__contains__(k)

            def get(self, k, default=None):
                self.touched.add(k)
                return super().get(k, default)

        try:
            base = self._current_flag_values_dict()
        except Exception:
            return False
        dependents: dict[str, set[str]] = {}
        for flag in cf.FLAGS:
            if not flag.visible_when:
                continue
            tracker = _TrackingDict(base)
            try:
                flag.visible_when(tracker)
            except Exception as exc:
                print(f"WARN: visible_when introspection failed for "
                      f"{flag.key!r}: {exc}", file=sys.stderr)
                return False
            for src in tracker.touched:
                dependents.setdefault(src, set()).add(flag.key)
        return dependents

    def _refresh_flag_visibilities(self, flag_keys) -> None:
        """Partial visibility refresh — only the flags in ``flag_keys``
        plus their group-empty bookkeeping. Used by the incremental
        ``_on_flag_changed`` path; backend swap still uses the full
        ``_apply_all_flag_visibilities`` sweep because hiding/showing
        whole groups needs an all-flags pass.
        """
        if not flag_keys:
            return
        try:
            values = self._current_flag_values_dict()
        except Exception as exc:
            print(f"WARN: could not build flag-values snapshot: {exc}",
                  file=sys.stderr)
            return
        backend = self.var_backend.get()
        flags_by_key = {f.key: f for f in cf.FLAGS}
        for key in flag_keys:
            flag = flags_by_key.get(key)
            if flag is None:
                continue
            w = self._flag_widgets.get(key)
            if w is None:
                continue
            applies = flag.applies_to(backend)
            predicate_ok = True
            if flag.visible_when:
                try:
                    predicate_ok = bool(flag.visible_when(values))
                except Exception as exc:
                    print(f"WARN: visible_when predicate for {key!r} raised: {exc}",
                          file=sys.stderr)
                    predicate_ok = True
            help_w = self._flag_help_widgets.get(key)
            label_w = self._flag_label_widgets.get(key)
            try:
                if not w.winfo_exists():
                    continue
                if applies:
                    w.grid()
                    if help_w is not None:
                        help_w.grid()
                    if label_w is not None:
                        label_w.grid()
                    state = "normal" if predicate_ok else "disabled"
                    w.configure(state=state)
                else:
                    w.grid_remove()
                    if help_w is not None:
                        help_w.grid_remove()
                    if label_w is not None:
                        label_w.grid_remove()
            except Exception:
                pass

    # ── Config save/load ────────────────────────────────────────────────
    def _refresh_saved_configs_dropdown(self) -> None:
        if hasattr(self, "_cfg_combo"):
            names = self._saved_config_names_for_backend(self.var_backend.get())
            self._cfg_combo["values"] = names
            if self.var_config_name.get().strip() not in names:
                self.var_config_name.set("")

    def _saved_config_names_for_backend(self, backend: str) -> list[str]:
        names = []
        for name in self.store.list_names():
            cfg = self.store.get(name)
            if cfg is not None and cfg.backend == backend:
                names.append(name)
        return names

    def _on_load_config(self) -> None:
        name = self.var_config_name.get().strip()
        if not name:
            return
        cfg = self.store.get(name)
        if cfg is None:
            messagebox.showerror("Load", f"No saved config named {name!r}.")
            return
        self._apply_loaded_config(cfg)
        self.store.touch_last_used(name)
        self._refresh_saved_configs_dropdown()
        self.var_config_name.set(name)
        self._append_console(f"Loaded config: {name}\n", tag="stage")

    def _apply_loaded_config(self, cfg: BuildConfig) -> None:
        self._suspend_traces = True
        try:
            self.var_backend.set(cfg.backend)
            self.var_source_dir.set(cfg.source_dir)
            self.var_build_dir.set(cfg.build_dir or "build")
            self.var_git_ref.set(cfg.git_ref)
            self.var_git_pull.set(cfg.git_pull_before_build)
            self.var_clean_build.set(cfg.clean_build)
            if cfg.jobs:
                self.var_jobs.set(cfg.jobs)
            self.var_cuda_archs.set(cfg.cuda_archs)
            self.var_extra_args.set(cfg.extra_cmake_args)
            self.var_generator.set(DEFAULT_GENERATOR_LABEL)
            env = cfg.env or {}
            if env.get("CC"):
                self.var_cc.set(env["CC"])
            if env.get("CXX"):
                self.var_cxx.set(env["CXX"])
            if env.get("CUDACXX"):
                self.var_cudacxx.set(env["CUDACXX"])
            if env.get("CUDA_TOOLKIT_ROOT_DIR"):
                self.var_cuda_root.set(env["CUDA_TOOLKIT_ROOT_DIR"])
            # UI-only state (kept off the cmake env).
            ui = cfg.ui_state or {}
            self.var_generator.set(
                self._generator_display_value(ui.get("generator", ""))
            )
            if "prefer_a" in ui:
                self.var_prefer_a_variant.set(ui["prefer_a"] == "1")
            if "prefer_f" in ui:
                self.var_prefer_f_variant.set(ui["prefer_f"] == "1")
            if "show_deprecated" in ui:
                self.var_show_deprecated_archs.set(ui["show_deprecated"] == "1")
            # Migration: older configs stashed UI state inside env with __KEY__
            # markers. Read those if present so we don't lose user prefs.
            legacy_map = {
                "__GENERATOR__": (
                    "generator",
                    lambda v: self.var_generator.set(self._generator_display_value(v)),
                ),
                "__PREFER_A__": ("prefer_a", lambda v: self.var_prefer_a_variant.set(v == "1")),
                "__PREFER_F__": ("prefer_f", lambda v: self.var_prefer_f_variant.set(v == "1")),
                "__SHOW_DEPRECATED__": ("show_deprecated",
                                        lambda v: self.var_show_deprecated_archs.set(v == "1")),
            }
            for legacy_key, (_ui_key, apply) in legacy_map.items():
                if legacy_key in env:
                    apply(env[legacy_key])
            # Apply flag values, defaulting unspecified flags from the schema.
            backend_defaults = cf.default_values_for_backend(cfg.backend)
            self._values_snapshot = {**backend_defaults, **(cfg.flag_values or {})}
            self._sync_flag_widgets_from_values()
        finally:
            self._suspend_traces = False
        # Defer the rebuild — this method is called from a button command.
        # Use full rebuild because a loaded config may have changed
        # ui_state preferences that affect non-flag sections.
        self._schedule_rebuild(full=True)

    def _on_save_config(self) -> None:
        name = self.var_config_name.get().strip()
        if not name:
            self._on_save_as_config()
            return
        self._persist_current_as(name)

    def _on_save_as_config(self) -> None:
        name = simpledialog.askstring("Save build config", "Name:",
                                      initialvalue=self.var_config_name.get())
        if not name:
            return
        self._persist_current_as(name.strip())

    def _persist_current_as(self, name: str) -> None:
        if not name:
            return
        # Build the UI-state dict separately so it never reaches cmake_env.
        ui_state: dict[str, str] = {
            "prefer_a": "1" if self.var_prefer_a_variant.get() else "0",
            "prefer_f": "1" if self.var_prefer_f_variant.get() else "0",
            "show_deprecated": "1" if self.var_show_deprecated_archs.get() else "0",
        }
        generator = self._selected_generator_value()
        if generator:
            ui_state["generator"] = generator
        cfg = BuildConfig(
            name=name,
            backend=self.var_backend.get(),
            source_dir=self.var_source_dir.get().strip(),
            build_dir=self.var_build_dir.get().strip() or "build",
            git_ref=self.var_git_ref.get().strip(),
            git_pull_before_build=self.var_git_pull.get(),
            clean_build=self.var_clean_build.get(),
            jobs=self._safe_int(self.var_jobs, default=0, minimum=0),
            cuda_archs=self.var_cuda_archs.get().strip(),
            env=self._current_env_dict(),
            flag_values=self._current_flag_values_dict(),
            extra_cmake_args=self.var_extra_args.get().strip(),
            ui_state=ui_state,
        )
        self.store.save(cfg)
        self.var_config_name.set(name)
        self._refresh_saved_configs_dropdown()
        self._append_console(f"Saved config: {name}\n", tag="stage")

    def _on_delete_config(self) -> None:
        name = self.var_config_name.get().strip()
        if not name:
            return
        if not messagebox.askyesno("Delete", f"Delete saved build config {name!r}?"):
            return
        if self.store.delete(name):
            self.var_config_name.set("")
            self._refresh_saved_configs_dropdown()
            self._append_console(f"Deleted config: {name}\n", tag="stage")

    # ── Preview ─────────────────────────────────────────────────────────
    def _schedule_preview_refresh(self) -> None:
        if self._preview_after_id is not None:
            try:
                self.root.after_cancel(self._preview_after_id)
            except Exception:
                pass
        self._preview_after_id = self.root.after(
            PREVIEW_REFRESH_DEBOUNCE_MS, self._refresh_preview
        )

    def _refresh_preview(self) -> None:
        self._preview_after_id = None
        plan = self._build_plan()
        if plan is None:
            return
        try:
            shell = plan_to_shell_script(plan, header="Preview (not yet executed)")
        except Exception as exc:
            shell = f"# preview failed: {exc}"
        if not hasattr(self, "_preview_text") or not self._preview_text.winfo_exists():
            return
        self._preview_text.configure(state="normal")
        self._preview_text.delete("1.0", "end")
        self._preview_text.insert("1.0", shell)
        self._preview_text.configure(state="disabled")
        # Warn if CUDA is requested but we couldn't detect a toolkit.
        self._update_cuda_warning()

    def _update_cuda_warning(self) -> None:
        """Show/hide the inline 'no CUDA toolkit found' warning next to the
        CUDA install picker. Triggered from the preview-refresh path so it
        stays in sync with GGML_CUDA toggles."""
        if not hasattr(self, "_cuda_warning_var"):
            return
        wants_cuda = _truthy_flag_str(self._values_snapshot.get("GGML_CUDA"))
        try:
            cur_val = self._flag_vars.get("GGML_CUDA")
            if cur_val is not None:
                wants_cuda = _truthy_flag_str(cur_val.get())
        except Exception:
            pass
        if wants_cuda and not self._toolchain.cuda_installs:
            self._cuda_warning_var.set(
                "⚠ GGML_CUDA is ON but no CUDA toolkit was detected. "
                "The build will likely fail at cmake configure."
            )
        else:
            self._cuda_warning_var.set("")

    def _on_copy_preview(self) -> None:
        if not hasattr(self, "_preview_text"):
            return
        txt = self._preview_text.get("1.0", "end-1c")
        self.root.clipboard_clear()
        self.root.clipboard_append(txt)
        self._append_console("cmake command copied to clipboard.\n", tag="stage")

    def _on_save_script(self) -> None:
        plan = self._build_plan()
        if plan is None:
            return
        default_name = f"build_{plan.backend.replace('.', '_')}.sh"
        path = filedialog.asksaveasfilename(
            title="Save build script",
            defaultextension=".sh",
            initialfile=default_name,
            filetypes=[("Shell script", "*.sh"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            Path(path).write_text(plan_to_shell_script(plan), encoding="utf-8")
            os.chmod(path, 0o755)
        except Exception as exc:
            messagebox.showerror("Save script", str(exc))
            return
        self._append_console(f"Wrote script: {path}\n", tag="stage")

    # ── Update banner / upstream check ──────────────────────────────────
    def check_for_updates(self, *, do_fetch: bool, show_output: bool = False) -> None:
        src = self.var_source_dir.get().strip()
        if not src:
            if show_output:
                self._append_console("Update check skipped: no source directory selected.\n", tag="error")
            return
        if self._upstream_check_in_flight:
            if show_output:
                self._append_console("Update check already running.\n", tag="stage")
            return
        self._upstream_check_in_flight = True
        if show_output:
            action = "Fetching upstream and checking for updates" if do_fetch else "Checking for updates"
            self._append_console(f"{action}...\n", tag="stage")

        def worker() -> None:
            # probe_upstream is best-effort, but if anything raises (e.g.
            # an OSError from a permission glitch on the .git dir) we must
            # still post *something* to the queue — otherwise the drain
            # loop reschedules forever and _upstream_check_in_flight never
            # clears, blocking all subsequent checks.
            try:
                status = probe_upstream(src, do_fetch=do_fetch)
            except Exception as exc:
                status = UpstreamStatus(error=f"probe failed: {exc}")
            self._pending_status.put((status, show_output))

        threading.Thread(target=worker, name="UpstreamProbe", daemon=True).start()
        # Cancel any prior drain before scheduling a new one so we don't
        # leak overlapping after() callbacks if the user clicks Check rapidly.
        if self._drain_after_id is not None:
            try:
                self.root.after_cancel(self._drain_after_id)
            except Exception:
                pass
        self._drain_after_id = self.root.after(150, self._drain_pending_status)

    def _drain_pending_status(self) -> None:
        self._drain_after_id = None
        try:
            while True:
                status, show_output = self._pending_status.get_nowait()
                self._upstream_status = status
                self._upstream_check_in_flight = False
                self._update_status_banner_visibility()
                if show_output:
                    self._append_update_check_result(status)
        except queue.Empty:
            if self._upstream_check_in_flight:
                self._drain_after_id = self.root.after(200, self._drain_pending_status)

    def _append_update_check_result(self, status: UpstreamStatus) -> None:
        if not status.is_git_repo:
            self._append_console("Update check skipped: source directory is not a git repository.\n", tag="error")
            return
        if status.error:
            self._append_console(f"Update check warning: {status.error}\n", tag="error")
        if not status.upstream_ref:
            return
        if status.behind > 0:
            self._append_console(
                f"{status.behind} new commit(s) available on {status.upstream_ref}.\n",
                tag="stage",
            )
            return
        msg = f"Up to date with {status.upstream_ref}"
        if status.ahead > 0:
            msg += f" (local branch is {status.ahead} commit(s) ahead)"
        self._append_console(f"{msg}.\n", tag="ok")

    def _update_status_banner_visibility(self) -> None:
        if not hasattr(self, "_banner"):
            return
        status = self._upstream_status
        if status.behind > 0:
            self._banner.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(4, 0))
            msg = (
                f"{status.behind} new commit(s) available on {status.upstream_ref}"
                + (f"  (HEAD {status.head_sha}: {status.head_subject})"
                   if status.head_subject else "")
            )
            self._banner_label.configure(text=msg, bg="#fff5cf")
            self._banner.configure(bg="#fff5cf", highlightbackground="#c5a800")
            self._banner_btns.configure(bg="#fff5cf")
        elif status.error and status.is_git_repo:
            self._banner.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(4, 0))
            self._banner_label.configure(text=f"Update check: {status.error}", bg="#fde0e0")
            self._banner.configure(bg="#fde0e0", highlightbackground="#c00000")
            self._banner_btns.configure(bg="#fde0e0")
        else:
            self._banner.grid_remove()

    def _on_pull_only(self) -> None:
        src = self.var_source_dir.get().strip()
        if not src:
            return
        if self._pull_only_in_flight:
            self._append_console(
                "git pull already running, ignoring duplicate request.\n",
                tag="stage",
            )
            return
        if self.runner.is_running:
            self._append_console(
                "build is running, cannot pull simultaneously.\n",
                tag="stage",
            )
            return
        self._pull_only_in_flight = True
        self._append_console("\n══ git pull --ff-only ══\n", tag="stage")
        # git pull can stall on network/disk for many seconds — run it off
        # the Tk main thread. Output is queued and drained by the Tk mainloop;
        # worker threads must not call Tk directly.
        threading.Thread(
            target=self._run_pull_only_worker, args=(src,),
            name="PullOnlyWorker", daemon=True,
        ).start()
        self._schedule_pull_drain()

    def _run_pull_only_worker(self, src: str) -> None:
        try:
            proc = subprocess.Popen(
                ["git", "pull", "--ff-only"], cwd=src,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                self._pending_pull_events.put(("line", line.rstrip("\n") + "\n"))
            rc = proc.wait()
            self._pending_pull_events.put(("done", rc))
        except Exception as exc:
            self._pending_pull_events.put(("error", str(exc)))

    def _schedule_pull_drain(self) -> None:
        if self._pull_drain_after_id is None:
            self._pull_drain_after_id = self.root.after(PULL_DRAIN_MS, self._drain_pull_only)

    def _drain_pull_only(self) -> None:
        self._pull_drain_after_id = None
        line_parts: list[str] = []
        line_chars = 0

        def flush_lines() -> None:
            nonlocal line_parts, line_chars
            if line_parts:
                self._append_console("".join(line_parts))
                line_parts = []
                line_chars = 0

        finished = False
        for _ in range(RUNNER_MAX_EVENTS_PER_POLL):
            try:
                kind, payload = self._pending_pull_events.get_nowait()
            except queue.Empty:
                break
            if kind == "line":
                text = str(payload)
                line_parts.append(text)
                line_chars += len(text)
                if line_chars >= RUNNER_MAX_CHARS_PER_POLL:
                    break
            elif kind == "done":
                flush_lines()
                rc = int(payload or 0)
                if rc != 0:
                    self._append_console(f"git pull exited {rc}\n", tag="error")
                finished = True
                break
            elif kind == "error":
                flush_lines()
                self._append_console(f"git pull failed: {payload}\n", tag="error")
                finished = True
                break

        flush_lines()
        if finished:
            self._pull_only_in_flight = False
            self.check_for_updates(do_fetch=True)
            return
        if self._pull_only_in_flight or not self._pending_pull_events.empty():
            self._schedule_pull_drain()

    def _on_pull_and_rebuild(self) -> None:
        if self._pull_only_in_flight:
            self._append_console(
                "git pull already running, ignoring pull-and-rebuild request.\n",
                tag="stage",
            )
            return
        self.var_git_pull.set(True)
        self._on_start_build()

    # ── Action bar handlers ────────────────────────────────────────────
    def _on_start_build(self) -> None:
        if self.runner.is_running:
            messagebox.showinfo("Build", "A build is already running.")
            return
        plan = self._build_plan()
        if plan is None:
            return
        if not plan.source_dir.strip():
            messagebox.showerror("Build", "Source directory is required.")
            return
        errors = cf.validate_values(
            self.var_backend.get(), self._current_flag_values_dict()
        )
        if errors:
            detail = "\n".join(f"  • {label}: {msg}" for label, msg in errors)
            messagebox.showerror(
                "Build",
                "Fix these invalid CUDA tuning values before building:\n\n"
                + detail,
            )
            return
        self._console_buffer.clear()
        if hasattr(self, "_console"):
            self._console.configure(state="normal")
            self._console.delete("1.0", "end")
            self._console.configure(state="disabled")

        self.var_status.set("Running…")
        self._start_btn.configure(state="disabled")
        self._cancel_btn.configure(state="normal")
        started = self.runner.start(plan)
        if not started:
            self.var_status.set("Idle")
            self._start_btn.configure(state="normal")
            self._cancel_btn.configure(state="disabled")
            return
        # Persist last-used name.
        name = self.var_config_name.get().strip()
        if name:
            self.store.touch_last_used(name)
        # Drain events on a Tk-after loop.
        self._poll_runner()

    def _on_cancel(self) -> None:
        self.runner.cancel()
        self.var_status.set("Cancelling…")

    def _poll_runner(self) -> None:
        drained_any = False
        line_parts: list[str] = []
        line_chars = 0

        def flush_lines() -> None:
            nonlocal line_parts, line_chars
            if line_parts:
                self._append_console("".join(line_parts))
                line_parts = []
                line_chars = 0

        for _ in range(RUNNER_MAX_EVENTS_PER_POLL):
            try:
                kind, payload = self.runner.events.get_nowait()
            except queue.Empty:
                break
            drained_any = True
            if kind == EVENT_LINE:
                text = str(payload) + "\n"
                line_parts.append(text)
                line_chars += len(text)
                if line_chars >= RUNNER_MAX_CHARS_PER_POLL:
                    break
            elif kind == EVENT_STAGE:
                flush_lines()
                self.var_status.set(str(payload))
            elif kind == EVENT_DONE:
                flush_lines()
                rc = int(payload or 0)
                if rc == 0:
                    self.var_status.set("Done ✓")
                    self._append_console("\nBuild succeeded.\n", tag="ok")
                else:
                    self.var_status.set(f"Failed (rc={rc})")
                    self._append_console(f"\nBuild failed with exit {rc}.\n", tag="error")
                self._on_build_finished()
                return
            elif kind == EVENT_CANCELLED:
                flush_lines()
                self.var_status.set("Cancelled")
                self._append_console("\nBuild cancelled.\n", tag="error")
                self._on_build_finished()
                return
            elif kind == EVENT_ERROR:
                flush_lines()
                self.var_status.set("Error")
                self._append_console(f"\nError: {payload}\n", tag="error")
                self._on_build_finished()
                return

        flush_lines()
        if self.runner.is_running or not self.runner.events.empty():
            delay = RUNNER_CATCHUP_POLL_MS if drained_any else RUNNER_POLL_MS
            self._poll_after_id = self.root.after(delay, self._poll_runner)
        else:
            self._poll_after_id = None

    def _on_build_finished(self) -> None:
        self._poll_after_id = None
        for attr in ("_start_btn", "_cancel_btn"):
            w = getattr(self, attr, None)
            if w is None:
                continue
            try:
                if w.winfo_exists():
                    w.configure(state="normal" if attr == "_start_btn" else "disabled")
            except Exception:
                pass
        # Re-check upstream after a successful pull-and-rebuild flow.
        if self.var_auto_check_updates.get():
            self.check_for_updates(do_fetch=False)

    def _on_clear_console(self) -> None:
        self._console_buffer.clear()
        if hasattr(self, "_console"):
            self._console.configure(state="normal")
            self._console.delete("1.0", "end")
            self._console.configure(state="disabled")

    def _on_save_log(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save build log",
            defaultextension=".log",
            filetypes=[("Log", "*.log"), ("Text", "*.txt"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            Path(path).write_text("\n".join(self._console_buffer), encoding="utf-8")
        except Exception as exc:
            messagebox.showerror("Save log", str(exc))

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────
    def _current_flag_values_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, var in self._flag_vars.items():
            try:
                v = var.get()
            except Exception:
                continue
            out[key] = v
        # Snapshot for next rebuild.
        self._values_snapshot = {**self._values_snapshot, **out}
        return out

    def _sync_flag_widgets_from_values(self) -> None:
        for key, val in self._values_snapshot.items():
            var = self._flag_vars.get(key)
            if var is None:
                continue
            try:
                if isinstance(var, tk.BooleanVar):
                    var.set(bool(val))
                else:
                    var.set("" if val is None else str(val))
            except Exception:
                pass

    def _current_env_dict(self) -> dict[str, str]:
        env: dict[str, str] = {}
        cc = self.var_cc.get().strip()
        cxx = self.var_cxx.get().strip()
        cudacxx = self.var_cudacxx.get().strip()
        cuda_root = self.var_cuda_root.get().strip()
        if cc:
            env["CC"] = cc
        if cxx:
            env["CXX"] = cxx
        if cudacxx:
            env["CUDACXX"] = cudacxx
        if cuda_root:
            # cmake reads this; pinning it avoids stray PATH installs hijacking the build.
            env["CUDA_TOOLKIT_ROOT_DIR"] = cuda_root
        return env

    def _on_install_build_tool(self, tool_key: str) -> None:
        tool = next(
            (row for row in detection.build_tool_statuses(self._toolchain) if row.key == tool_key),
            None,
        )
        if tool is None:
            messagebox.showerror("Install tool", f"Unknown build tool {tool_key!r}.")
            return
        if tool.installed:
            messagebox.showinfo("Install tool", f"{tool.label} is already detected.")
            return
        if tool.install_plan is None:
            messagebox.showerror(
                "Install tool",
                f"No supported install command was found for {tool.label} on this system.",
            )
            return
        try:
            terminal_launcher.open_command_in_terminal(tool.install_plan.command)
        except Exception as exc:
            messagebox.showerror("Install tool", f"Failed to open terminal: {exc}")
            return
        self._append_console(
            f"Opened {tool.install_plan.package_manager} install command for {tool.label}.\n",
            tag="stage",
        )
        messagebox.showinfo(
            "Install tool",
            f"Opened a terminal to install {tool.label}. Refresh toolchain when it finishes.",
        )

    def _resolved_build_dir(self) -> str:
        src = self.var_source_dir.get().strip()
        build = self.var_build_dir.get().strip() or "build"
        bp = Path(build)
        if bp.is_absolute():
            return str(bp)
        return str(Path(src) / bp) if src else build

    def _on_clear_cache(self) -> None:
        """Delete CMakeCache.txt + CMakeFiles/ in the build dir so the next
        configure re-reads every -D fresh. This is the fix for stale cache
        variables — e.g. a previously-built DMMV x-stride lingering in the
        cache. CMake won't overwrite an existing cache STRING on its own, and
        compile-definition changes like DMMV_X only take effect once the old
        object files (under CMakeFiles/) are also gone, so we remove both."""
        if self.runner.is_running:
            messagebox.showinfo("Clear cache", "A build is running; cancel it first.")
            return
        build_dir = self._resolved_build_dir().strip()
        if not build_dir:
            messagebox.showerror("Clear cache", "Build directory is not set.")
            return
        build = Path(build_dir).expanduser()
        cache_file = build / "CMakeCache.txt"
        cmake_files = build / "CMakeFiles"
        if not cache_file.exists() and not cmake_files.exists():
            messagebox.showinfo(
                "Clear cache",
                f"No CMake cache found in:\n{build}\n\nNothing to clear.",
            )
            return
        if not messagebox.askyesno(
            "Clear cache",
            "Delete the CMake cache in:\n"
            f"{build}\n\n"
            "Removes CMakeCache.txt and CMakeFiles/. The next build will "
            "re-configure from scratch and recompile. Source and any compiled "
            "binaries outside CMakeFiles/ are left untouched.\n\nProceed?",
        ):
            return
        import shutil
        removed = []
        try:
            if cache_file.exists():
                cache_file.unlink()
                removed.append("CMakeCache.txt")
            if cmake_files.exists():
                shutil.rmtree(cmake_files)
                removed.append("CMakeFiles/")
        except OSError as exc:
            messagebox.showerror("Clear cache", f"Failed to clear cache:\n{exc}")
            self._append_console(f"Clear cache failed: {exc}\n", tag="stage")
            return
        what = ", ".join(removed) if removed else "(nothing)"
        self._append_console(
            f"Cleared CMake cache in {build}: {what}\n", tag="stage"
        )

    def _build_plan(self) -> BuildPlan | None:
        values = self._current_flag_values_dict()
        backend = self.var_backend.get()
        # Inject the CMAKE_CUDA_ARCHITECTURES value as a regular cmake arg
        # (the schema lists it as a STRING flag — it materialises from the
        # values dict). Same goes for CMAKE_*_FLAGS.
        archs = self.var_cuda_archs.get().strip()
        if archs:
            values["CMAKE_CUDA_ARCHITECTURES"] = archs
        args = cf.values_to_cmake_args(
            backend, values, extra_cmake_args=self.var_extra_args.get().strip()
        )
        # Also pin CUDA_TOOLKIT_ROOT_DIR via -D so it's recorded in the cache;
        # cmake otherwise auto-derives from CMAKE_CUDA_COMPILER but pinning
        # makes the value explicit in the build artifacts.
        cuda_root = self.var_cuda_root.get().strip()
        if cuda_root and _truthy_flag_str(values.get("GGML_CUDA")):
            # CUDAToolkit_ROOT is the documented user-facing variable that
            # FindCUDAToolkit consumes (CMake 3.17+); the legacy CMAKE_CUDA_
            # COMPILER_TOOLKIT_ROOT spelling is an internal cache variable
            # and isn't reliable for steering CMake's CUDA discovery.
            args.append(f"-DCUDAToolkit_ROOT={cuda_root}")
        return BuildPlan(
            backend=backend,
            source_dir=self.var_source_dir.get().strip(),
            build_dir=self._resolved_build_dir(),
            cmake_args=args,
            cmake_env=self._current_env_dict(),
            jobs=self._safe_int(self.var_jobs, default=0, minimum=0),
            git_clone_if_missing=True,
            git_ref=self.var_git_ref.get().strip(),
            git_pull_before_build=self.var_git_pull.get(),
            clean_build=self.var_clean_build.get(),
            generator=self._selected_generator_value(),
        )

    def _append_console(self, text: str, *, tag: str | None = None) -> None:
        # Mirror to buffer so detach/reattach can replay history.
        for line in text.splitlines() or [""]:
            self._console_buffer.append(line)
        # Track how many lines we need to trim from the on-screen Text widget
        # *before* we clamp the buffer, so the widget shrinks in lockstep
        # with the buffer and can't grow unboundedly across long builds.
        overflow = len(self._console_buffer) - CONSOLE_MAX_LINES
        if len(self._console_buffer) > CONSOLE_MAX_LINES:
            self._console_buffer[:] = self._console_buffer[-CONSOLE_MAX_LINES:]
        if not hasattr(self, "_console") or not self._console.winfo_exists():
            return
        self._console.configure(state="normal")
        if overflow > 0:
            self._console.delete("1.0", f"{overflow + 1}.0")
        if tag:
            self._console.insert("end", text, tag)
        else:
            self._console.insert("end", text)
        if self.var_autoscroll.get():
            self._console.see("end")
        self._console.configure(state="disabled")
