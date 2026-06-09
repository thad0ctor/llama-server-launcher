#!/usr/bin/env python3
"""
MTP / Speculative Decoding tab.

This module owns the Tk vars, UI construction, and behavior handlers for
the MTP/Spec tab. Mirrors the ``IkLlamaTab`` / ``EnvironmentalVariablesTab``
pattern: the launcher instantiates ``SpecTab(self)`` and re-exposes the
SpecTab's Tk vars / handler methods on itself so existing call sites in
``modules/launch.py``, ``modules/config.py``, and the test suite keep
working without churn.

Backend-aware: most knobs differ between llama.cpp (mainline) and
ik_llama. The single source of truth for spec-type values is
``_SPEC_TYPES_LLAMA_CPP`` / ``_SPEC_TYPES_IK_LLAMA``, mirrored by the
launch.py emission block.
"""

import sys
import queue
import tkinter as tk
from threading import Event, Lock, Thread
from tkinter import ttk

from modules.system import parse_gguf_header_simple

SPEC_DRAFT_ANALYSIS_POLL_MS = 80


# Allowed values for the draft KV cache type comboboxes. Leading "" lets the
# user pick "don't emit the flag" (emission code already treats blank as
# omission). Module-level so tests can import and assert against the SAME
# tuple the UI actually uses — preventing silent drift when the set changes.
SPEC_DRAFT_CACHE_TYPE_VALUES = (
    "",
    "f16",
    "f32",
    "q8_0",
    "q4_0",
    "q4_1",
    "q5_0",
    "q5_1",
    "q6_k",
)


class SpecTab:
    """MTP / Speculative Decoding tab.

    Owns the spec_*/no_mmproj Tk vars and the methods that read/write them.
    Launcher-level attributes (``self.launcher.root``, ``self.launcher.found_models``,
    ``self.launcher.app_settings``, ``self.launcher.gpu_info``,
    ``self.launcher.detected_gpu_devices``, ``self.launcher._save_configs``,
    ``self.launcher.current_model_analysis``) are accessed through the launcher
    reference; the Tk vars / hint vars / widget dicts are direct attributes on
    SpecTab itself so test stubs can call methods like
    ``SpecTab._apply_spec_defaults_if_blank(stub)`` with the same shape they
    used for the launcher previously.
    """

    # Backend-aware spec_type choices. The single source of truth — mirrored
    # by the per-backend whitelists in modules/spec_launch.py.
    _SPEC_TYPES_LLAMA_CPP = (
        "none",
        "draft-simple",
        "draft-eagle3",
        "draft-mtp",
        "ngram-simple",
        "ngram-map-k",
        "ngram-map-k4v",
        "ngram-mod",
        "ngram-cache",
    )
    _SPEC_TYPES_IK_LLAMA = (
        "none",
        "mtp",
        "ngram-cache",
        "ngram-simple",
        "ngram-map-k",
        "ngram-map-k4v",
        "ngram-mod",
        "suffix",
    )

    @staticmethod
    def _init_bool(app_settings, key):
        v = app_settings.get(key, False)
        return bool(v) if isinstance(v, bool) else (str(v).lower() in ("1", "true", "yes"))

    @staticmethod
    def _init_str(app_settings, key, default=""):
        v = app_settings.get(key, default)
        return v if isinstance(v, str) else (str(v) if v is not None else default)

    def __init__(self, launcher):
        """Initialize Tk vars, reading defaults from ``launcher.app_settings``.

        Mirrors the legacy in-launcher block exactly: every Tk var starts
        from the persisted value via ``_init_bool`` / ``_init_str`` so the
        launcher's resync block (``modules/spec_persistence.resync_spec_tk_vars_from_app_settings``)
        is still authoritative for the post-load_saved_configs catch-up.
        """
        self.launcher = launcher
        app_settings = launcher.app_settings

        ib = self._init_bool
        is_ = self._init_str

        # --- MTP / Speculative Decoding ---
        # Master toggle: when False, no --spec-* / --draft-* flags are emitted.
        # Initial values are sourced from app_settings so they persist across
        # sessions (same pattern as mmproj/selected_mmproj_path).
        self.spec_enabled = tk.BooleanVar(value=ib(app_settings, "spec_enabled"))
        self.spec_type = tk.StringVar(value=is_(app_settings, "spec_type", "none") or "none")
        # Common draft controls (numeric entries; blank = use binary default).
        self.spec_draft_n_max = tk.StringVar(value=is_(app_settings, "spec_draft_n_max"))
        self.spec_draft_n_min = tk.StringVar(value=is_(app_settings, "spec_draft_n_min"))
        self.spec_draft_p_min = tk.StringVar(value=is_(app_settings, "spec_draft_p_min"))
        self.spec_draft_p_split = tk.StringVar(value=is_(app_settings, "spec_draft_p_split"))  # llama.cpp only
        # Draft model selection.
        self.spec_draft_model = tk.StringVar(value=is_(app_settings, "spec_draft_model"))  # -md path
        # Opt-in for ik_llama+mtp: when False, hide the draft picker UI AND
        # suppress --model-draft / draft offload emission so the embedded MTP
        # head in the base GGUF is used. Required-draft modes (draft-simple /
        # draft-eagle3 on llama.cpp) ignore this and always emit draft flags.
        self.spec_use_draft_model = tk.BooleanVar(value=ib(app_settings, "spec_use_draft_model"))
        self.spec_draft_ngl = tk.StringVar(value=is_(app_settings, "spec_draft_ngl"))
        self.spec_draft_device = tk.StringVar(value=is_(app_settings, "spec_draft_device"))
        self.spec_draft_ctk = tk.StringVar(value=is_(app_settings, "spec_draft_ctk"))
        self.spec_draft_ctv = tk.StringVar(value=is_(app_settings, "spec_draft_ctv"))
        self.spec_draft_cpu_moe = tk.BooleanVar(value=ib(app_settings, "spec_draft_cpu_moe"))  # llama.cpp only
        self.spec_draft_n_cpu_moe = tk.StringVar(value=is_(app_settings, "spec_draft_n_cpu_moe"))  # llama.cpp only
        # Derived/UI state for the draft model's GPU layer slider + status (mirrors
        # self.n_gpu_layers_int / self.max_gpu_layers / self.gpu_layers_status_var
        # for the main model). Not persisted directly — set after draft GGUF
        # analysis succeeds and consumed only by the slider widget + status label.
        self.spec_draft_ngl_int = tk.IntVar(value=0)
        self.max_spec_draft_gpu_layers = tk.IntVar(value=0)
        self.spec_draft_layers_status_var = tk.StringVar(value="Select draft model to see layer info")
        self.current_spec_draft_analysis = {}  # mirrors self.current_model_analysis
        self._spec_draft_analysis_generation = 0
        self._spec_draft_analysis_queue = queue.Queue()
        self._spec_draft_analysis_lock = Lock()
        self._spec_draft_analysis_after_id = None
        self._spec_draft_analysis_thread = None
        # Coalescing slots: ``_spec_draft_latest_path`` holds the most
        # recent ``_start_spec_draft_gguf_analysis`` request; the
        # single long-lived worker always re-reads this between
        # parses so rapid listbox navigation collapses to one parse
        # (the latest). ``_spec_draft_request_event`` wakes the
        # idle-blocked worker; ``_spec_draft_worker_active`` is the
        # atomic "do we already have a worker running?" flag, set /
        # cleared only under ``_spec_draft_analysis_lock``.
        self._spec_draft_latest_path: str | None = None
        self._spec_draft_request_event = Event()
        self._spec_draft_worker_active = False
        # Snapshot of the SELECTION the checkbox UI last rendered into
        # ``spec_draft_device``. Used by the conditional-clear logic so
        # ``prior_derived`` reflects what the UI actually wrote — not
        # the (possibly stale) value in ``app_settings`` /
        # ``loaded_selected``. Without this snapshot, a change to
        # ``spec_draft_selected_gpus`` between refreshes leaves the OLD
        # ``CUDA…`` string in ``spec_draft_device`` and the next
        # comparison treats it as a manual override.
        self._spec_draft_last_rendered_selected: list[int] = []
        # Suppresses ``_on_spec_draft_gpu_selection_changed`` for the
        # duration of a programmatic ``var.set(...)`` sweep in
        # ``_update_spec_draft_gpu_checkboxes``. See the handler for
        # the full rationale.
        self._suppress_spec_draft_gpu_events: bool = False
        # Ngram tuning (llama.cpp has per-variant size sets; ik_llama has a single shared set).
        self.spec_ngram_simple_size_n = tk.StringVar(value=is_(app_settings, "spec_ngram_simple_size_n"))
        self.spec_ngram_simple_size_m = tk.StringVar(value=is_(app_settings, "spec_ngram_simple_size_m"))
        self.spec_ngram_simple_min_hits = tk.StringVar(value=is_(app_settings, "spec_ngram_simple_min_hits"))
        self.spec_ngram_mapk_size_n = tk.StringVar(value=is_(app_settings, "spec_ngram_mapk_size_n"))
        self.spec_ngram_mapk_size_m = tk.StringVar(value=is_(app_settings, "spec_ngram_mapk_size_m"))
        self.spec_ngram_mapk_min_hits = tk.StringVar(value=is_(app_settings, "spec_ngram_mapk_min_hits"))
        self.spec_ngram_mapk4v_size_n = tk.StringVar(value=is_(app_settings, "spec_ngram_mapk4v_size_n"))
        self.spec_ngram_mapk4v_size_m = tk.StringVar(value=is_(app_settings, "spec_ngram_mapk4v_size_m"))
        self.spec_ngram_mapk4v_min_hits = tk.StringVar(value=is_(app_settings, "spec_ngram_mapk4v_min_hits"))
        self.spec_ngram_mod_n_min = tk.StringVar(value=is_(app_settings, "spec_ngram_mod_n_min"))
        self.spec_ngram_mod_n_max = tk.StringVar(value=is_(app_settings, "spec_ngram_mod_n_max"))
        self.spec_ngram_mod_n_match = tk.StringVar(value=is_(app_settings, "spec_ngram_mod_n_match"))
        # Shared single ngram set used by ik_llama (one --spec-ngram-* set).
        self.spec_ngram_size_n = tk.StringVar(value=is_(app_settings, "spec_ngram_size_n"))
        self.spec_ngram_size_m = tk.StringVar(value=is_(app_settings, "spec_ngram_size_m"))
        self.spec_ngram_min_hits = tk.StringVar(value=is_(app_settings, "spec_ngram_min_hits"))
        # Suffix tuning (ik_llama only).
        self.spec_suffix_pattern_len = tk.StringVar(value=is_(app_settings, "spec_suffix_pattern_len"))
        self.spec_suffix_max_depth = tk.StringVar(value=is_(app_settings, "spec_suffix_max_depth"))
        # ik_llama extras.
        self.spec_autotune = tk.BooleanVar(value=ib(app_settings, "spec_autotune"))
        self.spec_draft_params = tk.StringVar(value=is_(app_settings, "spec_draft_params"))  # -draft "k=v,k=v"
        # llama.cpp vision toggle.
        self.no_mmproj = tk.BooleanVar(value=ib(app_settings, "no_mmproj"))  # --no-mmproj

        # Hint/status vars populated by setup_tab. Pre-create here so test
        # stubs (which don't run setup_tab) can still call refresh helpers.
        # setup_tab will overwrite via attribute reassignment when needed.
        self.spec_draft_gpu_vars = []
        self._spec_widgets = {}
        self._spec_sections = {}

        # Aliases for launcher-level Tk vars the spec handlers read/write.
        # Aliased by-reference so writes via either path are visible to the
        # other. Tests pass a stub that already has these attributes flat;
        # the production launcher exposes them on `launcher.<name>`.
        # ``backend_selection`` is read by ``_refresh_spec_tab_state`` to
        # gate per-backend visibility; ``parallel`` is overwritten by
        # ``_apply_mtp_parallel_default`` when MTP is active.
        self.backend_selection = launcher.backend_selection
        self.parallel = launcher.parallel

    def setup_tab(self, parent):
        """Set up the MTP / Speculative Decoding tab.

        Always visible regardless of backend selection — both llama.cpp and
        ik_llama expose `--spec-type`, but their flag surfaces differ. UI
        widgets that only apply to one backend are disabled (not hidden) when
        the other backend is active so the user can see what's not available.
        """
        # Scrolling canvas pattern (matches _setup_advanced_tab).
        canvas = tk.Canvas(parent, highlightthickness=0)
        vs = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind(
            "<Configure>",
            lambda e: canvas.configure(yscrollcommand=vs.set, scrollregion=canvas.bbox("all")),
        )
        canvas_window = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))
        canvas.pack(side="left", fill="both", expand=True)
        vs.pack(side="right", fill="y")

        inner.columnconfigure(1, weight=1)

        # Tracked widgets are kept on self for _refresh_spec_tab_state() to
        # enable/disable and show/hide based on backend + spec_type + master
        # toggle. Mutate-in-place rather than reassigning so the launcher's
        # ``__getattr__`` delegation always sees the live mapping (callers
        # read these via ``launcher.<name>``).
        self._spec_widgets.clear()
        # Sections we hide/show wholesale.
        self._spec_sections.clear()

        r = 0

        # --- Header / master toggle ---
        ttk.Label(inner, text="MTP / Speculative Decoding", font=("TkDefaultFont", 12, "bold")).grid(
            column=0, row=r, sticky="w", padx=10, pady=(10, 5), columnspan=4
        )
        r += 1
        ttk.Separator(inner, orient="horizontal").grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=5)
        r += 1

        master_cb = ttk.Checkbutton(
            inner,
            text="Enable speculative decoding",
            variable=self.spec_enabled,
        )
        master_cb.grid(column=0, row=r, sticky="w", padx=10, pady=4, columnspan=4)
        r += 1
        self._spec_widgets["master_cb"] = master_cb

        self.spec_status_var = tk.StringVar(value="")
        ttk.Label(inner, textvariable=self.spec_status_var, foreground="gray").grid(
            column=0, row=r, sticky="w", padx=10, pady=(0, 6), columnspan=4
        )
        r += 1

        # --- Speculative type ---
        ttk.Label(inner, text="Speculative type:", font=("TkDefaultFont", 10, "bold")).grid(
            column=0, row=r, sticky="w", padx=10, pady=(8, 2), columnspan=4
        )
        r += 1
        type_combo = ttk.Combobox(
            inner,
            textvariable=self.spec_type,
            values=list(self._SPEC_TYPES_LLAMA_CPP),
            state="readonly",
            width=28,
        )
        type_combo.grid(column=0, row=r, sticky="w", padx=10, pady=2)
        self._spec_widgets["type_combo"] = type_combo
        ttk.Label(
            inner,
            text="(values depend on backend; 'none' = no spec flags)",
            foreground="gray",
        ).grid(column=1, row=r, sticky="w", padx=5, pady=2, columnspan=3)
        r += 1

        # --- Common draft controls section ---
        sec = ttk.LabelFrame(inner, text="Common draft controls")
        sec.grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=(10, 4))
        sec.columnconfigure(1, weight=1)
        sec.columnconfigure(3, weight=1)
        self._spec_sections["common"] = sec
        r += 1

        sr = 0
        ttk.Label(sec, text="n-max:").grid(column=0, row=sr, sticky="w", padx=6, pady=2)
        e_nmax = ttk.Entry(sec, textvariable=self.spec_draft_n_max, width=10)
        e_nmax.grid(column=1, row=sr, sticky="w", padx=4, pady=2)
        self._spec_widgets["n_max"] = e_nmax
        ttk.Label(sec, text="n-min:").grid(column=2, row=sr, sticky="w", padx=6, pady=2)
        e_nmin = ttk.Entry(sec, textvariable=self.spec_draft_n_min, width=10)
        e_nmin.grid(column=3, row=sr, sticky="w", padx=4, pady=2)
        self._spec_widgets["n_min"] = e_nmin
        sr += 1

        ttk.Label(sec, text="p-min:").grid(column=0, row=sr, sticky="w", padx=6, pady=2)
        e_pmin = ttk.Entry(sec, textvariable=self.spec_draft_p_min, width=10)
        e_pmin.grid(column=1, row=sr, sticky="w", padx=4, pady=2)
        self._spec_widgets["p_min"] = e_pmin
        self.spec_pmin_hint_var = tk.StringVar(value="")
        ttk.Label(sec, textvariable=self.spec_pmin_hint_var, foreground="gray").grid(
            column=2, row=sr, sticky="w", padx=4, pady=2, columnspan=2
        )

        sr += 1
        ttk.Label(sec, text="p-split:").grid(column=0, row=sr, sticky="w", padx=6, pady=2)
        e_psplit = ttk.Entry(sec, textvariable=self.spec_draft_p_split, width=10)
        e_psplit.grid(column=1, row=sr, sticky="w", padx=4, pady=2)
        self._spec_widgets["p_split"] = e_psplit
        self.spec_psplit_hint_var = tk.StringVar(value="(llama.cpp only)")
        ttk.Label(sec, textvariable=self.spec_psplit_hint_var, foreground="gray").grid(
            column=2, row=sr, sticky="w", padx=4, pady=2, columnspan=2
        )

        # MTP requires --parallel 1 (single-slot operation). The trace
        # callbacks force this when MTP is selected, but show the hint
        # so users understand what's happening and can verify.
        sr += 1
        self.spec_parallel_hint_var = tk.StringVar(value="")
        ttk.Label(
            sec, textvariable=self.spec_parallel_hint_var, foreground="#888888", font=("TkSmallCaptionFont")
        ).grid(column=0, row=sr, columnspan=4, sticky="w", padx=6, pady=(4, 2))

        # Reset-to-default button: overwrites all four common controls with
        # the recommended values for the current spec_type. For ngram/suffix
        # types (which have no recommended defaults), clears the fields.
        sr += 1
        reset_btn = ttk.Button(sec, text="Reset to defaults", command=self._reset_spec_defaults)
        reset_btn.grid(column=0, row=sr, sticky="w", padx=6, pady=(6, 4))
        self._spec_widgets["reset_defaults_btn"] = reset_btn
        ttk.Label(
            sec,
            text="Overwrites n-max / n-min / p-min / p-split with the recommended defaults for the active type.",
            foreground="#888888",
            font=("TkSmallCaptionFont"),
        ).grid(column=1, row=sr, columnspan=3, sticky="w", padx=4, pady=(6, 4))

        # --- Draft model section ---
        # Picks the draft GGUF from the same scanned-models pool as the
        # main model listbox. The HF-repo entry was removed: users either
        # have the draft GGUF locally (and it shows up in the listbox) or
        # they don't use it. A "Clear" button reverts to "use base GGUF".
        sec = ttk.LabelFrame(inner, text="Draft model")
        sec.grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=4)
        sec.columnconfigure(1, weight=1)
        self._spec_sections["draft_model"] = sec
        r += 1

        # Opt-in toggle. Visible only for ik_llama+mtp (gated in
        # _refresh_spec_tab_state); inner draft widgets below are hidden
        # when this is unchecked so the section reads as "MTP head only".
        self.spec_use_draft_cb = ttk.Checkbutton(
            sec,
            text="Use a separate draft model "
            "(optional for ik_llama MTP — leave unchecked to use the embedded head from the base GGUF)",
            variable=self.spec_use_draft_model,
        )
        self.spec_use_draft_cb.grid(column=0, row=0, sticky="w", padx=6, pady=(4, 2), columnspan=4)
        self._spec_widgets["use_draft_cb"] = self.spec_use_draft_cb

        sr = 1
        ttk.Label(sec, text="Select draft GGUF:").grid(column=0, row=sr, sticky="nw", padx=6, pady=2)
        draft_list_frame = ttk.Frame(sec)
        draft_list_frame.grid(column=1, row=sr, columnspan=2, sticky="nsew", padx=4, pady=2)
        sec.columnconfigure(1, weight=1)
        draft_list_sb = ttk.Scrollbar(draft_list_frame, orient=tk.VERTICAL)
        self.spec_draft_listbox = tk.Listbox(
            draft_list_frame,
            height=6,
            width=48,
            yscrollcommand=draft_list_sb.set,
            exportselection=False,
            state=tk.DISABLED,
        )
        draft_list_sb.config(command=self.spec_draft_listbox.yview)
        self.spec_draft_listbox.bind("<<ListboxSelect>>", self._on_spec_draft_model_selected)
        draft_list_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.spec_draft_listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._spec_widgets["draft_listbox"] = self.spec_draft_listbox
        b_clear = ttk.Button(sec, text="Clear", command=self._clear_spec_draft_model)
        b_clear.grid(column=3, row=sr, sticky="nw", padx=4, pady=2)
        self._spec_widgets["draft_clear_btn"] = b_clear
        sr += 1

        ttk.Label(sec, text="Selected path:").grid(column=0, row=sr, sticky="w", padx=6, pady=2)
        self.spec_draft_path_display_var = tk.StringVar(
            value=self.spec_draft_model.get() or "(none — uses base GGUF for MTP)"
        )
        path_lbl = ttk.Label(
            sec,
            textvariable=self.spec_draft_path_display_var,
            foreground="gray",
        )
        path_lbl.grid(column=1, row=sr, columnspan=3, sticky="ew", padx=4, pady=2)
        self._spec_widgets["draft_path_display"] = path_lbl
        sr += 1

        # --- Draft GPU layers (Entry + Slider + Status, mirrors main model) ---
        ttk.Label(sec, text="Draft GPU layers (-ngld):").grid(column=0, row=sr, sticky="w", padx=6, pady=2)
        draft_ngl_frame = ttk.Frame(sec)
        draft_ngl_frame.grid(column=1, row=sr, columnspan=3, sticky="ew", padx=4, pady=2)
        draft_ngl_frame.columnconfigure(1, weight=1)

        # Entry stays NORMAL so the user can type a value even before analysis.
        self.spec_draft_ngl_entry = ttk.Entry(
            draft_ngl_frame,
            textvariable=self.spec_draft_ngl,
            width=6,
            state=tk.NORMAL,
        )
        self.spec_draft_ngl_entry.grid(column=0, row=0, sticky="w", padx=(0, 10))

        # Slider is DISABLED until draft analysis succeeds and provides a max.
        self.spec_draft_ngl_slider = ttk.Scale(
            draft_ngl_frame,
            from_=0,
            to=self.max_spec_draft_gpu_layers.get(),
            orient="horizontal",
            variable=self.spec_draft_ngl_int,
            command=self._sync_spec_draft_gpu_layers_from_slider,
            state=tk.DISABLED,
        )
        self.spec_draft_ngl_slider.grid(column=1, row=0, sticky="ew", padx=5)

        self.spec_draft_layers_status_label = ttk.Label(
            draft_ngl_frame,
            textvariable=self.spec_draft_layers_status_var,
            width=35,
            anchor="w",
        )
        self.spec_draft_layers_status_label.grid(column=2, row=0, sticky="w", padx=(10, 0))

        # Validation + sync bindings, mirroring the main model entry.
        try:
            vcmd_draft = (self.launcher.root.register(self._validate_spec_draft_gpu_layers_entry), "%P")
            self.spec_draft_ngl_entry.config(validate="key", validatecommand=vcmd_draft)
        except tk.TclError:
            pass
        self.spec_draft_ngl_entry.bind("<FocusOut>", self._sync_spec_draft_gpu_layers_from_entry)
        self.spec_draft_ngl_entry.bind("<Return>", self._sync_spec_draft_gpu_layers_from_entry)

        # Track the entry as the "draft_ngl" widget so _refresh_spec_tab_state
        # can toggle just the entry's NORMAL/DISABLED state alongside the rest
        # of the section. Slider state is governed by analysis success, not by
        # the spec master toggle.
        self._spec_widgets["draft_ngl"] = self.spec_draft_ngl_entry
        self._spec_widgets["draft_ngl_slider"] = self.spec_draft_ngl_slider
        sr += 1

        # --- Draft devices: checkbox grid (mirrors main GPU checkboxes) ---
        ttk.Label(sec, text="Draft devices (-devd):").grid(column=0, row=sr, sticky="nw", padx=6, pady=2)
        self.spec_draft_gpu_checkbox_frame = ttk.Frame(sec)
        self.spec_draft_gpu_checkbox_frame.grid(
            column=1,
            row=sr,
            columnspan=3,
            sticky="ew",
            padx=4,
            pady=2,
        )
        self.spec_draft_gpu_vars = []
        # Register the parent frame so _refresh_spec_tab_state's enable/disable
        # rules can propagate to every checkbox child.
        self._spec_widgets["draft_gpu_frame"] = self.spec_draft_gpu_checkbox_frame
        sr += 1

        # --- Draft KV cache types: comboboxes (blank = use server default) ---
        # Blank value lets the user pick "don't emit the flag"; emission code
        # already treats "" as omission so behavior is unchanged.
        ttk.Label(sec, text="Draft K cache type (-ctkd):").grid(column=0, row=sr, sticky="w", padx=6, pady=2)
        self.spec_draft_ctk_combo = ttk.Combobox(
            sec,
            textvariable=self.spec_draft_ctk,
            width=10,
            values=SPEC_DRAFT_CACHE_TYPE_VALUES,
            state="readonly",
        )
        self.spec_draft_ctk_combo.grid(column=1, row=sr, sticky="w", padx=4, pady=2)
        self._spec_widgets["draft_ctk"] = self.spec_draft_ctk_combo
        ttk.Label(sec, text="Draft V cache type (-ctvd):").grid(column=2, row=sr, sticky="w", padx=6, pady=2)
        self.spec_draft_ctv_combo = ttk.Combobox(
            sec,
            textvariable=self.spec_draft_ctv,
            width=10,
            values=SPEC_DRAFT_CACHE_TYPE_VALUES,
            state="readonly",
        )
        self.spec_draft_ctv_combo.grid(column=3, row=sr, sticky="w", padx=4, pady=2)
        self._spec_widgets["draft_ctv"] = self.spec_draft_ctv_combo
        sr += 1

        cb_cmoed = ttk.Checkbutton(
            sec,
            text="Offload draft MoE to CPU (--spec-draft-cpu-moe)",
            variable=self.spec_draft_cpu_moe,
        )
        cb_cmoed.grid(column=0, row=sr, sticky="w", padx=6, pady=2, columnspan=2)
        self._spec_widgets["draft_cpu_moe"] = cb_cmoed
        ttk.Label(sec, text="n-cpu-moe:").grid(column=2, row=sr, sticky="w", padx=6, pady=2)
        e_ncm = ttk.Entry(sec, textvariable=self.spec_draft_n_cpu_moe, width=10)
        e_ncm.grid(column=3, row=sr, sticky="w", padx=4, pady=2)
        self._spec_widgets["draft_n_cpu_moe"] = e_ncm

        # Snapshot of every child of the draft_model section except the opt-in
        # checkbox. _refresh_spec_tab_state grid_remove()s these as a group when
        # ik_llama+mtp is active and spec_use_draft_model is False (so the
        # section collapses to just the checkbox); grid()s them back otherwise.
        self._spec_draft_inner_widgets = [w for w in sec.winfo_children() if w is not self.spec_use_draft_cb]

        # --- Ngram tuning (llama.cpp per-variant; ik_llama shared) ---
        # Per-variant simple/mapk/mapk4v/mod groups for llama.cpp:
        for key, label, vars_triplet in [
            (
                "ngram_simple",
                "Ngram simple (--spec-ngram-simple-*)",
                (self.spec_ngram_simple_size_n, self.spec_ngram_simple_size_m, self.spec_ngram_simple_min_hits),
            ),
            (
                "ngram_mapk",
                "Ngram map-k (--spec-ngram-map-k-*)",
                (self.spec_ngram_mapk_size_n, self.spec_ngram_mapk_size_m, self.spec_ngram_mapk_min_hits),
            ),
            (
                "ngram_mapk4v",
                "Ngram map-k4v (--spec-ngram-map-k4v-*)",
                (self.spec_ngram_mapk4v_size_n, self.spec_ngram_mapk4v_size_m, self.spec_ngram_mapk4v_min_hits),
            ),
        ]:
            sec = ttk.LabelFrame(inner, text=label)
            sec.grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=4)
            sec.columnconfigure(1, weight=1)
            sec.columnconfigure(3, weight=1)
            self._spec_sections[key] = sec
            r += 1
            v_sn, v_sm, v_mh = vars_triplet
            ttk.Label(sec, text="size-n:").grid(column=0, row=0, sticky="w", padx=6, pady=2)
            ttk.Entry(sec, textvariable=v_sn, width=10).grid(column=1, row=0, sticky="w", padx=4, pady=2)
            ttk.Label(sec, text="size-m:").grid(column=2, row=0, sticky="w", padx=6, pady=2)
            ttk.Entry(sec, textvariable=v_sm, width=10).grid(column=3, row=0, sticky="w", padx=4, pady=2)
            ttk.Label(sec, text="min-hits:").grid(column=0, row=1, sticky="w", padx=6, pady=2)
            ttk.Entry(sec, textvariable=v_mh, width=10).grid(column=1, row=1, sticky="w", padx=4, pady=2)

        # Ngram mod (n-min, n-max, n-match) for llama.cpp:
        sec = ttk.LabelFrame(inner, text="Ngram mod (--spec-ngram-mod-*)")
        sec.grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=4)
        sec.columnconfigure(1, weight=1)
        sec.columnconfigure(3, weight=1)
        self._spec_sections["ngram_mod"] = sec
        r += 1
        ttk.Label(sec, text="n-min:").grid(column=0, row=0, sticky="w", padx=6, pady=2)
        ttk.Entry(sec, textvariable=self.spec_ngram_mod_n_min, width=10).grid(
            column=1, row=0, sticky="w", padx=4, pady=2
        )
        ttk.Label(sec, text="n-max:").grid(column=2, row=0, sticky="w", padx=6, pady=2)
        ttk.Entry(sec, textvariable=self.spec_ngram_mod_n_max, width=10).grid(
            column=3, row=0, sticky="w", padx=4, pady=2
        )
        ttk.Label(sec, text="n-match:").grid(column=0, row=1, sticky="w", padx=6, pady=2)
        ttk.Entry(sec, textvariable=self.spec_ngram_mod_n_match, width=10).grid(
            column=1, row=1, sticky="w", padx=4, pady=2
        )

        # Shared ngram set (ik_llama uses a single set across all ngram types):
        sec = ttk.LabelFrame(inner, text="Ngram tuning (--spec-ngram-*)")
        sec.grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=4)
        sec.columnconfigure(1, weight=1)
        sec.columnconfigure(3, weight=1)
        self._spec_sections["ngram_shared"] = sec
        r += 1
        ttk.Label(sec, text="size-n:").grid(column=0, row=0, sticky="w", padx=6, pady=2)
        ttk.Entry(sec, textvariable=self.spec_ngram_size_n, width=10).grid(column=1, row=0, sticky="w", padx=4, pady=2)
        ttk.Label(sec, text="size-m:").grid(column=2, row=0, sticky="w", padx=6, pady=2)
        ttk.Entry(sec, textvariable=self.spec_ngram_size_m, width=10).grid(column=3, row=0, sticky="w", padx=4, pady=2)
        ttk.Label(sec, text="min-hits:").grid(column=0, row=1, sticky="w", padx=6, pady=2)
        ttk.Entry(sec, textvariable=self.spec_ngram_min_hits, width=10).grid(
            column=1, row=1, sticky="w", padx=4, pady=2
        )

        # --- Suffix (ik_llama only) ---
        sec = ttk.LabelFrame(inner, text="Suffix tuning (ik_llama, --suffix-*)")
        sec.grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=4)
        sec.columnconfigure(1, weight=1)
        self._spec_sections["suffix"] = sec
        r += 1
        ttk.Label(sec, text="pattern-len:").grid(column=0, row=0, sticky="w", padx=6, pady=2)
        ttk.Entry(sec, textvariable=self.spec_suffix_pattern_len, width=10).grid(
            column=1, row=0, sticky="w", padx=4, pady=2
        )
        ttk.Label(sec, text="max-depth:").grid(column=2, row=0, sticky="w", padx=6, pady=2)
        ttk.Entry(sec, textvariable=self.spec_suffix_max_depth, width=10).grid(
            column=3, row=0, sticky="w", padx=4, pady=2
        )

        # --- ik_llama extras ---
        sec = ttk.LabelFrame(inner, text="ik_llama extras")
        sec.grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=4)
        sec.columnconfigure(1, weight=1)
        self._spec_sections["ik_extras"] = sec
        r += 1
        ttk.Checkbutton(
            sec,
            text="Enable spec autotune (--spec-autotune)",
            variable=self.spec_autotune,
        ).grid(column=0, row=0, sticky="w", padx=6, pady=2, columnspan=2)
        ttk.Label(sec, text="Draft params (-draft):").grid(column=0, row=1, sticky="w", padx=6, pady=2)
        ttk.Entry(sec, textvariable=self.spec_draft_params).grid(
            column=1, row=1, sticky="ew", padx=4, pady=2, columnspan=2
        )
        ttk.Label(
            sec,
            text='Free-form comma list, e.g. "k=v,k=v"',
            foreground="gray",
        ).grid(column=0, row=2, sticky="w", padx=6, pady=(0, 4), columnspan=3)

        # --- Vision (llama.cpp only) ---
        sec = ttk.LabelFrame(inner, text="Vision (llama.cpp)")
        sec.grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=(4, 10))
        self._spec_sections["vision"] = sec
        r += 1
        ttk.Checkbutton(
            sec,
            text="Disable embedded mmproj at launch (--no-mmproj)",
            variable=self.no_mmproj,
        ).grid(column=0, row=0, sticky="w", padx=6, pady=2)
        ttk.Label(
            sec,
            text="Useful for MTP GGUFs that embed a vision projector you don't need.",
            foreground="gray",
        ).grid(column=0, row=1, sticky="w", padx=6, pady=(0, 4))

        # Apply per-backend / per-spec_type visibility + enable rules
        # now that every section + widget reference is registered.
        # This call used to be unnecessary because ``setup_tab`` ran
        # during launcher ``__init__`` and the subsequent config-load
        # traces (on backend_selection / spec_enabled / spec_type)
        # fired ``_refresh_spec_tab_state`` for us. With the tab now
        # built lazily on first selection (see launcher
        # ``_register_lazy_tab``), config load has long since finished
        # — no trace fires when we build — so widgets render in their
        # default (everything-shown) state unless we kick the refresh
        # explicitly here.
        try:
            self._refresh_spec_tab_state()
        except tk.TclError as exc:
            print(
                f"WARN: post-setup _refresh_spec_tab_state failed: {exc}",
                file=sys.stderr,
            )
        # Replay the draft GPU checkbox build. Any earlier GPU-detection
        # refresh fired before this lazy tab was constructed and short-
        # circuited on ``spec_draft_gpu_checkbox_frame`` being None; without
        # this explicit replay the "Draft devices" section opens blank
        # until some unrelated later refresh restores it.
        try:
            self._update_spec_draft_gpu_checkboxes()
        except tk.TclError as exc:
            print(
                f"WARN: post-setup _update_spec_draft_gpu_checkboxes failed: {exc}",
                file=sys.stderr,
            )

    def _on_spec_draft_model_selected(self, event=None):
        """Listbox <<ListboxSelect>> handler for the draft GGUF picker.

        Resolves the selected display name against ``self.launcher.found_models``
        (populated by the main model scan) and writes the absolute path into
        ``self.spec_draft_model``. Also refreshes the read-only path label
        so the user can see the full path of what they picked, and kicks off
        a background GGUF analysis to populate the draft GPU-layer slider.
        """
        try:
            lb = getattr(self, "spec_draft_listbox", None)
            if lb is None:
                return
            sel = lb.curselection()
            if not sel:
                return
            display_name = lb.get(sel[0])
            full_path = getattr(self.launcher, "found_models", {}).get(display_name)
            if full_path is None:
                return
            full_path_str = str(full_path)
            self.spec_draft_model.set(full_path_str)
            if hasattr(self, "spec_draft_path_display_var"):
                self.spec_draft_path_display_var.set(full_path_str)
            # Kick off a background analysis so the slider can be enabled with
            # a sensible max. Reuse main-model analysis when the user picked
            # the same GGUF for both — saves a redundant header parse.
            main_analysis = getattr(self.launcher, "current_model_analysis", None) or {}
            if main_analysis.get("path") == full_path_str and main_analysis.get("n_layers") is not None:
                self._update_ui_after_spec_draft_analysis(main_analysis)
            else:
                self.spec_draft_layers_status_var.set("Analyzing draft model...")
                if hasattr(self, "spec_draft_ngl_slider") and self.spec_draft_ngl_slider.winfo_exists():
                    self.spec_draft_ngl_slider.config(state=tk.DISABLED)
                self.current_spec_draft_analysis = {}
                self._start_spec_draft_gguf_analysis(full_path_str)
        except Exception as e:
            print(f"WARN: _on_spec_draft_model_selected failed: {e}", file=sys.stderr)

    def _clear_spec_draft_model(self):
        """Reset the draft model selection (no -md / --model-draft will be emitted)."""
        self.spec_draft_model.set("")
        if hasattr(self, "spec_draft_path_display_var"):
            self.spec_draft_path_display_var.set("(none — uses base GGUF for MTP)")
        try:
            lb = getattr(self, "spec_draft_listbox", None)
            if lb is not None and lb.winfo_exists():
                lb.selection_clear(0, tk.END)
        except (tk.TclError, AttributeError):
            pass
        # Wipe derived layer state so a stale "Max Layers" status from the
        # previous draft model doesn't linger after the user clears it.
        self.current_spec_draft_analysis = {}
        try:
            self.max_spec_draft_gpu_layers.set(0)
            self.spec_draft_layers_status_var.set("Select draft model to see layer info")
            if hasattr(self, "spec_draft_ngl_slider") and self.spec_draft_ngl_slider.winfo_exists():
                self.spec_draft_ngl_slider.config(to=0, state=tk.DISABLED)
        except (tk.TclError, AttributeError):
            pass

    # -- Draft GPU-layer sync helpers (mirror main _set_gpu_layers et al.) --

    def _set_spec_draft_gpu_layers(self, input_value, from_slider=False):
        """Helper that updates the draft model's IntVar based on user input.

        Mirrors ``_set_gpu_layers`` exactly:
        * ``input_value == -1`` -> map to ``max_spec_draft_gpu_layers`` if known.
        * Slider input is clamped to the max; entry input is allowed to exceed
          max (so a user with manual knowledge can set a higher value).
        """
        max_layers = self.max_spec_draft_gpu_layers.get()
        int_val = 0
        if input_value == -1:
            int_val = max_layers if max_layers > 0 else 0
        elif input_value >= 0:
            if from_slider and max_layers > 0:
                int_val = min(input_value, max_layers)
            else:
                int_val = input_value
        try:
            if self.spec_draft_ngl_int.get() != int_val:
                self.spec_draft_ngl_int.set(int_val)
        except tk.TclError:
            pass

    def _sync_spec_draft_gpu_layers_from_slider(self, value_str):
        """Slider callback for the draft layers control."""
        if not hasattr(self, "spec_draft_ngl_entry") or not self.spec_draft_ngl_entry.winfo_exists():
            return
        try:
            value = int(float(value_str))
            self._set_spec_draft_gpu_layers(value, from_slider=True)
            canonical_str = str(value)
            if self.spec_draft_ngl.get() != canonical_str:
                self.spec_draft_ngl.set(canonical_str)
        except ValueError:
            pass

    def _sync_spec_draft_gpu_layers_from_entry(self, event=None):
        """FocusOut/Return callback for the draft layers entry."""
        if not hasattr(self, "spec_draft_ngl_entry") or not self.spec_draft_ngl_entry.winfo_exists():
            return
        current_str = self.spec_draft_ngl.get().strip()
        if current_str == "":
            current_str = "0"
        try:
            value = int(current_str)
            self._set_spec_draft_gpu_layers(value)
            if self.spec_draft_ngl.get() != current_str:
                self.spec_draft_ngl.set(current_str)
        except ValueError:
            try:
                current_int_value = self.spec_draft_ngl_int.get()
            except tk.TclError:
                current_int_value = 0
            self.spec_draft_ngl.set(str(current_int_value))

    def _validate_spec_draft_gpu_layers_entry(self, proposed_value):
        """Validation for the draft-layers entry. Same rules as the main one:
        allow blank/just-dash mid-typing, allow -1, allow non-negative ints.
        """
        if not hasattr(self, "max_spec_draft_gpu_layers"):
            return True
        pv = proposed_value.strip()
        if pv in ("", "-"):
            return True
        try:
            value = int(pv)
            if value == -1:
                return True
            if value < -1:
                return False
            return value >= 0
        except ValueError:
            return False

    # -- Draft device checkbox grid (mirror _update_gpu_checkboxes simpler form) --

    def _update_spec_draft_gpu_checkboxes(self):
        """Build the draft device checkbox grid from detected CUDA devices.

        Simpler than ``_update_gpu_checkboxes`` because we only support the
        detected-GPU mode for the draft section. The launcher's GPU detection
        is CUDA-only, so device names are hardcoded to ``CUDA<i>``.
        Persistence lives in ``app_settings["spec_draft_selected_gpus"]`` and
        the resulting comma-joined string is written to ``self.spec_draft_device``
        so the existing emission block in ``modules/launch.py`` picks it up
        unchanged.

        Incremental: when the GPU shape (count + names + manual-mode flag)
        matches the prior render we just update the existing BooleanVars in
        place. The main GPU panel uses the same trick — see the comment on
        ``_gpu_checkboxes_fingerprint`` in the launcher for the motivation
        (post-detection refresh was destroying + recreating 8 checkboxes
        every time even when nothing changed).
        """
        if not hasattr(self, "spec_draft_gpu_checkbox_frame") or not self.spec_draft_gpu_checkbox_frame.winfo_exists():
            return

        gpu_info = getattr(self.launcher, "gpu_info", {})
        count = gpu_info.get("device_count", 0) if isinstance(gpu_info, dict) else 0
        # Strictly coerce persisted indices to ``int`` before membership
        # checks. Without this, a saved ``"1"`` (string) wouldn't match
        # GPU 1 (int), and ``True``/``1.0`` would silently match GPU 1 —
        # the checkbox state ended up corrupted, and the value written
        # back to ``app_settings`` at lines below propagated the corruption.
        # Mirror ``modules.spec_launch._coerce_strict_gpu_index``.
        from modules.spec_launch import _coerce_strict_gpu_index as _coerce_idx

        raw_persisted = self.launcher.app_settings.get("spec_draft_selected_gpus", []) or []
        # Mirror the launch path (``spec_launch._resolve_draft_device_value``
        # and ``get_effective_visible_gpu_indices``): only iterate
        # genuine sequences. A persisted ``"0,1"`` string would
        # otherwise iterate character-by-character into
        # ``["0", ",", "1"]`` and either crash ``_coerce_idx`` or
        # silently drop every entry. A bare scalar would raise on
        # ``for raw_idx in raw_persisted``.
        if not isinstance(raw_persisted, (list, tuple)):
            raw_persisted = []
        loaded_selected: set[int] = set()
        for raw_idx in raw_persisted:
            idx = _coerce_idx(raw_idx)
            if idx is not None:
                loaded_selected.add(idx)
        detected_devices = getattr(self.launcher, "detected_gpu_devices", [])
        # Manual GPU mode disables draft device emission entirely — the
        # manual GPU list isn't real CUDA hardware, so we can't tell the
        # binary "use CUDA<i>" reliably.
        manual_mode = bool(getattr(getattr(self.launcher, "manual_gpu_mode", None), "get", lambda: False)())

        new_fp = (
            manual_mode,
            count,
            tuple(
                (
                    detected_devices[i].get("name", "")
                    if i < len(detected_devices) and isinstance(detected_devices[i], dict)
                    else ""
                )
                for i in range(count)
            ),
        )
        existing_fp = getattr(self, "_spec_draft_checkboxes_last_fp", None)
        if (
            existing_fp is not None
            and existing_fp == new_fp
            and count > 0
            and not manual_mode
            and len(self.spec_draft_gpu_vars) == count
        ):
            valid_selected = []
            # Suppress the per-var trace handler during this
            # programmatic ``var.set(desired)`` sweep. The handler
            # (``_on_spec_draft_gpu_selection_changed``) would
            # otherwise fire mid-loop on every flipped checkbox and
            # write a PARTIAL ``CUDA…`` string into
            # ``spec_draft_device`` before the loop finishes —
            # producing transient intermediate values that the
            # conditional-clear logic below then sees and
            # mis-classifies as "manual override".
            self._suppress_spec_draft_gpu_events = True
            try:
                for i, var in enumerate(self.spec_draft_gpu_vars):
                    desired = i in loaded_selected
                    if desired:
                        valid_selected.append(i)
                    try:
                        if var.get() != desired:
                            var.set(desired)
                    except Exception:
                        pass
            finally:
                self._suppress_spec_draft_gpu_events = False
            self.launcher.app_settings["spec_draft_selected_gpus"] = valid_selected
            # Same conditional-clear logic the manual-mode branch
            # below uses: only overwrite ``spec_draft_device`` when
            # it's empty OR already equals what the checkboxes would
            # have produced (so an explicit override like
            # ``Vulkan0`` / ``CUDA2,SYCL1`` survives a checkbox
            # refresh). The previous unconditional ``set`` wiped
            # those overrides on every UI refresh that came through
            # this fast path.
            checkbox_derived = ",".join(f"CUDA{i}" for i in valid_selected)
            try:
                current = self.spec_draft_device.get()
                # Compare against what THIS code last actually wrote
                # into ``spec_draft_device`` — not ``loaded_selected``
                # (which is the NEW persisted set after a possible
                # external mutation). Using the live snapshot
                # prevents a stale ``CUDA…`` string from being
                # treated as a manual override.
                # ``getattr(..., [])`` so older test stubs (and any
                # subclass that bypasses ``__init__``) don't crash on
                # a missing attribute; an empty list correctly
                # represents "no prior render".
                prior_derived = ",".join(f"CUDA{i}" for i in getattr(self, "_spec_draft_last_rendered_selected", []))
                if current in ("", prior_derived):
                    self.spec_draft_device.set(checkbox_derived)
                    self._spec_draft_last_rendered_selected = list(valid_selected)
                else:
                    # Preserving an explicit manual override
                    # (``Vulkan0`` / ``CUDA2,SYCL1``). Clear the
                    # persisted checkbox indices so the launch path's
                    # ``_resolve_draft_device_value`` actually falls
                    # back to ``spec_draft_device``. Without this,
                    # ``app_settings["spec_draft_selected_gpus"]``
                    # still holds the checkbox indices and the launch
                    # command emits the checkbox-derived ``CUDA…``
                    # list while the UI shows the user's override —
                    # a silent contract mismatch.
                    self.launcher.app_settings["spec_draft_selected_gpus"] = []
                    # Also clear the live checkbox vars + the
                    # rendered-selection snapshot so the UI ticks
                    # match: with a manual override active there is
                    # no checkbox-derived contribution, and a stale
                    # snapshot here would let a future refresh
                    # mistake itself into thinking ``CUDA0,CUDA1``
                    # was the checkbox-derived string and clear the
                    # override on the next pass. Use the
                    # ``_suppress_spec_draft_gpu_events`` flag to
                    # avoid the per-var trace handler firing in the
                    # middle of the sweep (same pattern the
                    # rebuild-from-scratch path uses ~line 897).
                    self._suppress_spec_draft_gpu_events = True
                    try:
                        for var in self.spec_draft_gpu_vars:
                            try:
                                if var.get():
                                    var.set(False)
                            except Exception:
                                pass
                    finally:
                        self._suppress_spec_draft_gpu_events = False
                    self._spec_draft_last_rendered_selected = []
            except Exception:
                pass
            try:
                self._refresh_spec_tab_state()
            except Exception:
                pass
            return

        self._spec_draft_checkboxes_last_fp = new_fp
        for w in self.spec_draft_gpu_checkbox_frame.winfo_children():
            w.destroy()
        self.spec_draft_gpu_vars = []
        # Sanitize: rebuild the persisted-index list from what's currently
        # valid, so a stale saved selection (e.g. GPUs that no longer exist
        # or were filtered, or any selection while manual GPU mode is on)
        # never re-emits as a phantom CUDA<i>. Also derive a fresh device
        # string so spec_draft_device matches the visible checkbox state.
        valid_selected = []

        if count > 0 and not manual_mode:
            MAX_GPUS_PER_ROW = 3
            for i in range(count):
                gpu_details = detected_devices[i] if i < len(detected_devices) else {}
                is_selected = i in loaded_selected
                if is_selected:
                    valid_selected.append(i)
                v = tk.BooleanVar(value=is_selected)
                gpu_name_display = f"GPU {i}"
                if gpu_details and gpu_details.get("name"):
                    gpu_name_display += f": {gpu_details['name']}"
                row = i // MAX_GPUS_PER_ROW
                col = i % MAX_GPUS_PER_ROW
                cb = ttk.Checkbutton(
                    self.spec_draft_gpu_checkbox_frame,
                    text=gpu_name_display,
                    variable=v,
                )
                cb.grid(row=row, column=col, sticky="w", padx=3, pady=2)
                v.trace_add(
                    "write",
                    lambda *args, index=i: self._on_spec_draft_gpu_selection_changed(index),
                )
                self.spec_draft_gpu_vars.append(v)
        else:
            ttk.Label(
                self.spec_draft_gpu_checkbox_frame,
                text=(
                    "No CUDA devices detected."
                    if not manual_mode
                    else "Draft device selection disabled in manual GPU mode."
                ),
                foreground="orange",
            ).grid(row=0, column=0, sticky="w", padx=5, pady=3)

        # Mirror the sanitized selection back into app_settings + spec_draft_device
        # ONLY when we actually rendered real checkboxes (count > 0 + non-manual).
        # When count == 0 we don't have enough information to sanitize: GPU
        # detection may still be in flight (async SystemInfoManager run), and
        # wiping spec_draft_selected_gpus here would destroy the user's stored
        # selection BEFORE it has a chance to be applied. Same reasoning for
        # spec_draft_device on machines/tests without detected CUDA hardware.
        if count > 0 and not manual_mode:
            self.launcher.app_settings["spec_draft_selected_gpus"] = valid_selected
            # Same conditional-clear logic the fast path and the
            # manual-mode branch below use: only overwrite
            # ``spec_draft_device`` when the current value is empty
            # OR equals the checkbox-derived string from the prior
            # render. An explicit override like ``Vulkan0`` or
            # ``CUDA2,SYCL1`` must survive a full rebuild
            # (first lazy-tab render, GPU-shape change, manual/auto
            # mode flip) instead of being wiped on every refresh
            # that comes through this slow path.
            checkbox_derived = ",".join(f"CUDA{i}" for i in valid_selected)
            try:
                current = self.spec_draft_device.get()
                # Compare against what THIS code last actually wrote
                # — same rationale as the fast path above. The
                # snapshot is updated after a successful ``set``
                # below so future refreshes recognise our own value.
                # ``getattr(..., [])`` so older test stubs (and any
                # subclass that bypasses ``__init__``) don't crash on
                # a missing attribute; an empty list correctly
                # represents "no prior render".
                prior_derived = ",".join(f"CUDA{i}" for i in getattr(self, "_spec_draft_last_rendered_selected", []))
                if current in ("", prior_derived):
                    self.spec_draft_device.set(checkbox_derived)
                    self._spec_draft_last_rendered_selected = list(valid_selected)
                else:
                    # Same rationale as the fast path above: when we
                    # preserve a manual override here, the persisted
                    # ``spec_draft_selected_gpus`` must be cleared too
                    # so the launch-time fallback in
                    # ``_resolve_draft_device_value`` reads
                    # ``spec_draft_device`` instead of re-emitting the
                    # checkbox-derived ``CUDA…`` list.
                    self.launcher.app_settings["spec_draft_selected_gpus"] = []
                    # Mirror the fast-path sweep: also untick the live
                    # checkbox vars and clear the rendered-selection
                    # snapshot. The slow rebuild branch just created
                    # fresh BooleanVars at lines ~969-978 with
                    # ``value=is_selected`` (i.e. the persisted
                    # selection), so without this sweep the UI shows
                    # ticked checkboxes that the launch path ignores
                    # — same silent contract mismatch as the fast
                    # path before its sibling fix.
                    self._suppress_spec_draft_gpu_events = True
                    try:
                        for var in self.spec_draft_gpu_vars:
                            try:
                                if var.get():
                                    var.set(False)
                            except Exception:
                                pass
                    finally:
                        self._suppress_spec_draft_gpu_events = False
                    self._spec_draft_last_rendered_selected = []
            except Exception:
                pass
        elif manual_mode:
            # Manual GPU mode disables CUDA<i> draft device emission
            # for the checkbox-derived value, but must NOT wipe out a
            # user / imported override like ``Vulkan0`` /
            # ``CUDA2,SYCL1``. Only clear when the current
            # ``spec_draft_device`` value exactly matches the string
            # the checkbox UI would have produced from the persisted
            # selection — that's the leftover-from-prior-non-manual
            # case the original clear was meant to handle.
            #
            # ``loaded_selected`` includes EVERY persisted index,
            # even out-of-range ones (``CUDA99`` from a stale config)
            # that the non-manual branch above filters out via
            # ``valid_selected``. To detect the leftover-from-prior-
            # non-manual case correctly, reconstruct the same
            # filtered/valid list — otherwise a "stale config" with
            # ``CUDA99`` baked into ``spec_draft_device`` would never
            # match the recomputed string and we'd treat it as a
            # manual override, which it isn't.
            # Compare against the live snapshot of what THIS code
            # last wrote, so we recognise our own checkbox-derived
            # value AND the manual-mode flip cleanly clears it.
            # ``getattr`` defaults to ``[]`` so test stubs and any
            # subclass that bypasses ``__init__`` don't crash.
            indices = getattr(self, "_spec_draft_last_rendered_selected", [])
            if not indices:
                # First render in manual mode after a non-manual session
                # leaves the snapshot empty even though ``app_settings``
                # may still hold the persisted checkbox-derived
                # ``"CUDA0,CUDA1"`` string. Without a fallback, the
                # snapshot-derived ``checkbox_derived`` is ``""`` and
                # we'd treat the leftover as a manual override and
                # leave it alone — meaning CUDA<i> emission survives
                # into the launch command after the user toggled
                # manual mode on. Reconstruct from ``loaded_selected``
                # (the persisted indices) so the leftover-string
                # detection works on first manual-mode render too.
                #
                # IMPORTANT: do NOT unconditionally gate on
                # ``i < count``. In manual mode ``count`` is often
                # 0 (manual mode hides the auto-detected GPU list),
                # and on a cold start with ``count == 0`` and a
                # persisted ``"CUDA0"`` we still need to recognise
                # ``CUDA0`` as our own leftover and clear it. But
                # when ``count > 0`` the upper bound DOES apply —
                # without it, a stale config like
                # ``loaded_selected == {0, 99}`` would reconstruct
                # ``CUDA0,CUDA99`` and an old checkbox-derived
                # ``"CUDA0"`` would no longer match, leaking the
                # stale string into the launch command as a phantom
                # "manual override". Pick the tighter bound when we
                # have one and fall back to ``i >= 0`` only when
                # there are no detected GPUs yet.
                indices = sorted(i for i in loaded_selected if i >= 0 and (count <= 0 or i < count))
            checkbox_derived = ",".join(f"CUDA{i}" for i in indices)
            try:
                if self.spec_draft_device.get() == checkbox_derived:
                    self.spec_draft_device.set("")
                    # Clear the snapshot too — there's nothing
                    # checkbox-derived in the field now.
                    self._spec_draft_last_rendered_selected = []
            except Exception:
                pass

        # Re-apply enable/disable rules now that children exist. Safe to call
        # before _spec_sections is populated (the method short-circuits).
        try:
            self._refresh_spec_tab_state()
        except Exception:
            pass

    def _on_spec_draft_gpu_selection_changed(self, index):
        """Trace callback when a draft GPU checkbox flips.

        Recomputes the selected-index list, persists it, then builds the
        ``"CUDA0,CUDA2,..."`` device-name string the llama.cpp / ik_llama
        emission blocks consume. CUDA prefix is hardcoded because the
        launcher's GPU detection is CUDA-only.
        """
        # Programmatic ``var.set(...)`` sweeps in
        # ``_update_spec_draft_gpu_checkboxes`` set this flag for the
        # duration of the for-loop so we don't fire on each
        # intermediate flip and stamp a partial ``CUDA…`` string
        # into ``spec_draft_device`` before the sweep completes.
        if getattr(self, "_suppress_spec_draft_gpu_events", False):
            return
        try:
            selected_indices = [i for i, v in enumerate(self.spec_draft_gpu_vars) if v.get()]
            self.launcher.app_settings["spec_draft_selected_gpus"] = selected_indices
            device_str = ",".join(f"CUDA{i}" for i in selected_indices)
            if self.spec_draft_device.get() != device_str:
                self.spec_draft_device.set(device_str)
            # Snapshot what THIS code just wrote so a later refresh /
            # manual-mode flip recognises it as "checkbox-set" via the
            # conditional-clear logic in
            # ``_update_spec_draft_gpu_checkboxes``. Without the
            # update, the snapshot stays at the previous render and
            # the new ``CUDA…`` string looks like a manual override.
            self._spec_draft_last_rendered_selected = list(selected_indices)
            try:
                self.launcher._save_configs()
            except Exception:
                pass
        except Exception as e:
            print(
                f"WARN: _on_spec_draft_gpu_selection_changed failed: {e}",
                file=sys.stderr,
            )

    # -- Draft GGUF analysis (mirrors _on_model_selected/_run_gguf_analysis) --

    def _start_spec_draft_gguf_analysis(self, draft_path_str):
        """Submit ``draft_path_str`` to the single background analyser.

        Coalescing: each call overwrites the "latest pending path" slot
        and bumps the generation counter. A SINGLE long-lived worker
        thread processes only the latest pending path between parses,
        so rapid listbox navigation no longer fans out concurrent
        ``parse_gguf_header_simple`` invocations on superseded paths.
        Stale-result-after-parse guard is preserved via the generation
        check in ``_drain_spec_draft_gguf_analysis``.
        """
        with self._get_spec_draft_analysis_lock():
            self._spec_draft_analysis_generation += 1
            self._spec_draft_latest_path = draft_path_str
            self._spec_draft_request_event.set()
            need_spawn = not self._spec_draft_worker_active
            if need_spawn:
                self._spec_draft_worker_active = True
                t = Thread(
                    target=self._run_spec_draft_gguf_analysis_loop,
                    daemon=True,
                )
                self._spec_draft_analysis_thread = t
        if need_spawn:
            try:
                t.start()
            except Exception:
                # ``Thread.start()`` can raise ``RuntimeError`` (already
                # started — shouldn't happen here) or ``OSError`` on
                # systems that hit a thread-creation limit. The
                # worker-active latch was set under the lock above
                # before ``start()`` ran; without rolling it back the
                # next request would see ``_spec_draft_worker_active``
                # still True and skip the respawn, so the GGUF analysis
                # path silently goes dead until a process restart.
                with self._get_spec_draft_analysis_lock():
                    self._spec_draft_worker_active = False
                    self._spec_draft_analysis_thread = None
                raise
        if self._spec_draft_analysis_after_id is None:
            try:
                self._spec_draft_analysis_after_id = self.launcher.root.after(
                    SPEC_DRAFT_ANALYSIS_POLL_MS,
                    self._drain_spec_draft_gguf_analysis,
                )
            except tk.TclError:
                # Root was destroyed mid-flight (window closed while a
                # background draft-analysis thread was still running). Drop
                # the poll silently — the launcher is shutting down anyway.
                self._spec_draft_analysis_after_id = None

    def _get_spec_draft_analysis_lock(self):
        lock = getattr(self, "_spec_draft_analysis_lock", None)
        if lock is None:
            lock = Lock()
            self._spec_draft_analysis_lock = lock
        return lock

    def _run_spec_draft_gguf_analysis_loop(self):
        """Single long-lived worker that processes only the LATEST
        requested draft path. Exits when no request is pending after
        a brief idle window so the next selection re-spawns cheaply.

        Never calls Tk APIs directly — results are queued for the
        main thread's ``_drain_spec_draft_gguf_analysis`` poll.
        """
        # Idle timeout: after this many seconds without a new request,
        # the worker exits. A small value keeps the thread cheap when
        # the spec tab is dormant; the next selection re-spawns under
        # the lock atomically.
        IDLE_TIMEOUT_S = 2.0
        try:
            while True:
                signalled = self._spec_draft_request_event.wait(timeout=IDLE_TIMEOUT_S)
                with self._get_spec_draft_analysis_lock():
                    pending = self._spec_draft_latest_path
                    analysis_id = self._spec_draft_analysis_generation
                    self._spec_draft_latest_path = None
                    self._spec_draft_request_event.clear()
                    if pending is None:
                        if not signalled:
                            # Idle timeout AND no request pending →
                            # exit. Selection-side ``need_spawn`` check
                            # under the same lock guarantees the next
                            # selection re-spawns.
                            self._spec_draft_worker_active = False
                            return
                        # Race: event fired but the producer reset the
                        # slot before we read it. Loop back and wait again.
                        continue
                # Parse the latest pending path. May take a while; we
                # release the lock so additional selections can keep
                # updating the slot during this parse — they just won't
                # spawn a second worker.
                try:
                    analysis_result = parse_gguf_header_simple(pending)
                except Exception as exc:
                    analysis_result = {"path": pending, "error": str(exc)}
                with self._get_spec_draft_analysis_lock():
                    # Stale guard: a newer selection bumped the
                    # generation while we were parsing → drop our
                    # result. The next loop iteration handles the new
                    # latest path.
                    if analysis_id != self._spec_draft_analysis_generation:
                        continue
                    self._spec_draft_analysis_queue.put((analysis_id, analysis_result))
        except Exception:
            # Worker exception is fatal for this worker; let the next
            # selection re-spawn a fresh one rather than masquerade as
            # alive.
            with self._get_spec_draft_analysis_lock():
                self._spec_draft_worker_active = False
            raise

    def _run_spec_draft_gguf_analysis(self, draft_path_str, analysis_id=None):
        """Compatibility shim for tests that call the worker entry-point
        directly with an analysis_id (mocked path scenarios). The
        production code path now uses ``_run_spec_draft_gguf_analysis_loop``;
        this single-shot wrapper preserves the pre-coalescing test API
        without leaving an old per-thread design in production."""
        # Defensive init: ``_get_spec_draft_analysis_lock`` already
        # lazily creates its lock if ``__init__`` hasn't run (test
        # subclasses / stubs that bypass ``__init__``). Mirror that
        # pattern for ``_spec_draft_analysis_generation`` and
        # ``_spec_draft_analysis_queue`` so this shim doesn't
        # AttributeError on those callers — the production
        # ``_run_spec_draft_gguf_analysis_loop`` doesn't need this
        # because ``__init__`` always runs before it spawns a worker.
        if not hasattr(self, "_spec_draft_analysis_generation"):
            # Seed from the caller's ``analysis_id`` (if any) so the
            # stale-result gate below (``analysis_id !=
            # self._spec_draft_analysis_generation``) doesn't drop
            # the very result this shim is about to enqueue.
            # Defaulting to ``0`` regardless made a call like
            # ``_run_spec_draft_gguf_analysis(path, analysis_id=42)``
            # against a test-stubbed instance fail the gate and
            # silently discard the parse result.
            self._spec_draft_analysis_generation = analysis_id if analysis_id is not None else 0
        if not hasattr(self, "_spec_draft_analysis_queue"):
            self._spec_draft_analysis_queue = queue.Queue()
        try:
            if analysis_id is None:
                with self._get_spec_draft_analysis_lock():
                    analysis_id = self._spec_draft_analysis_generation
            analysis_result = parse_gguf_header_simple(draft_path_str)
        except Exception as e:
            analysis_result = {"path": draft_path_str, "error": str(e)}
        with self._get_spec_draft_analysis_lock():
            if analysis_id != self._spec_draft_analysis_generation:
                return
            self._spec_draft_analysis_queue.put((analysis_id, analysis_result))

    def _drain_spec_draft_gguf_analysis(self):
        self._spec_draft_analysis_after_id = None
        try:
            while True:
                analysis_id, analysis_result = self._spec_draft_analysis_queue.get_nowait()
                if analysis_id != self._spec_draft_analysis_generation:
                    continue
                if self.spec_draft_model.get() != analysis_result.get("path"):
                    continue
                self._update_ui_after_spec_draft_analysis(analysis_result)
                return
        except queue.Empty:
            pass
        if (
            self._spec_draft_analysis_thread and self._spec_draft_analysis_thread.is_alive()
        ) or not self._spec_draft_analysis_queue.empty():
            try:
                self._spec_draft_analysis_after_id = self.launcher.root.after(
                    SPEC_DRAFT_ANALYSIS_POLL_MS,
                    self._drain_spec_draft_gguf_analysis,
                )
            except tk.TclError:
                self._spec_draft_analysis_after_id = None

    def _update_ui_after_spec_draft_analysis(self, analysis_result):
        """Apply analysis result to the draft slider/status (Tk thread)."""
        # Stale result guard.
        if self.spec_draft_model.get() != analysis_result.get("path"):
            return
        self.current_spec_draft_analysis = analysis_result
        error = analysis_result.get("error")
        n_layers = analysis_result.get("n_layers")
        if error or n_layers is None or n_layers <= 0:
            msg = error if error else "Could not determine layers"
            self.spec_draft_layers_status_var.set(f"{msg} (manual entry available)")
            self.max_spec_draft_gpu_layers.set(0)
            if hasattr(self, "spec_draft_ngl_slider") and self.spec_draft_ngl_slider.winfo_exists():
                self.spec_draft_ngl_slider.config(to=0, state=tk.DISABLED)
            return
        # Success: enable slider and update status. Mirrors main +1 for output.
        max_offloadable = n_layers + 1
        self.max_spec_draft_gpu_layers.set(max_offloadable)
        self.spec_draft_layers_status_var.set(f"Max Layers: {max_offloadable} ({n_layers} blocks + output)")
        if hasattr(self, "spec_draft_ngl_slider") and self.spec_draft_ngl_slider.winfo_exists():
            self.spec_draft_ngl_slider.config(to=max_offloadable, state=tk.NORMAL)
        # Re-sync entry -> int so the slider reflects the entry's current value.
        try:
            self._sync_spec_draft_gpu_layers_from_entry()
        except Exception:
            pass

    def _apply_spec_defaults_if_blank(self):
        """Pre-fill blank draft-tuning fields with sensible defaults based on
        the current spec_type. Only fills *blank* fields — user input is never
        overwritten. Called on spec_enabled and spec_type changes."""
        # No-op when speculative decoding is disabled.
        try:
            if not self.spec_enabled.get():
                return
        except Exception:
            return
        spec_type = (self.spec_type.get() or "").strip()
        # n_min=0 means "always try speculation" — the most generally useful
        # baseline; users tuning further can raise it. n_max/p_min/p_split are
        # spec-type specific (MTP is essentially free so n_max=3 is the
        # benchmark-validated sweet spot; classical draft models gain from
        # the binary's larger default of 16).
        if spec_type in ("draft-mtp", "mtp"):
            defaults = {"n_max": "3", "n_min": "0", "p_min": "0.75", "p_split": "0.10"}
        elif spec_type in ("draft-simple", "draft-eagle3"):
            defaults = {"n_max": "16", "n_min": "0", "p_min": "0.75", "p_split": "0.10"}
        else:
            # ngram-*, suffix, cache, none, or unknown — no defaults to pre-fill.
            return
        var_map = {
            "n_max": self.spec_draft_n_max,
            "n_min": self.spec_draft_n_min,
            "p_min": self.spec_draft_p_min,
            "p_split": self.spec_draft_p_split,
        }
        for key, default_value in defaults.items():
            var = var_map.get(key)
            if var is None:
                continue
            try:
                if not var.get().strip():
                    var.set(default_value)
            except Exception:
                pass

    def _reset_spec_defaults(self):
        """Button handler: OVERWRITE all four common draft controls with
        the recommended values for the current spec_type. Unlike
        ``_apply_spec_defaults_if_blank`` (which only fills blanks), this
        ignores existing values — it's the explicit user action to revert
        to known-good settings. For spec_types without recommended
        defaults (ngram-*, suffix, ngram-cache, none), all four fields
        are cleared.
        """
        spec_type = (self.spec_type.get() or "").strip()
        if spec_type in ("draft-mtp", "mtp"):
            values = {"n_max": "3", "n_min": "0", "p_min": "0.75", "p_split": "0.10"}
        elif spec_type in ("draft-simple", "draft-eagle3"):
            values = {"n_max": "16", "n_min": "0", "p_min": "0.75", "p_split": "0.10"}
        else:
            # ngram-*, suffix, cache, blank — clear to "use binary default".
            values = {"n_max": "", "n_min": "", "p_min": "", "p_split": ""}
        var_map = {
            "n_max": self.spec_draft_n_max,
            "n_min": self.spec_draft_n_min,
            "p_min": self.spec_draft_p_min,
            "p_split": self.spec_draft_p_split,
        }
        for key, value in values.items():
            var = var_map.get(key)
            if var is None:
                continue
            try:
                var.set(value)
            except Exception:
                pass

    def _apply_mtp_parallel_default(self):
        """When MTP mode is active, force ``--parallel`` to 1.

        MTP currently requires single-slot operation (-np 1). Unlike the
        soft prefill in ``_apply_spec_defaults_if_blank``, this is a hard
        constraint of the MTP implementation upstream, so we OVERWRITE
        whatever value is in ``self.parallel`` rather than only filling
        blanks. Users can still type a different value afterwards; the
        launch block will emit a stderr warning in that case.
        """
        try:
            if not self.spec_enabled.get():
                return
        except Exception:
            return
        spec_type = (self.spec_type.get() or "").strip()
        if spec_type not in ("draft-mtp", "mtp"):
            return
        try:
            if self.parallel.get().strip() != "1":
                self.parallel.set("1")
        except Exception:
            pass

    def _on_spec_enabled_changed(self):
        """Trace callback chained after ``spec_enabled`` writes.

        Refreshes tab visibility/enabled state and pre-fills the draft
        tuning fields when the master toggle flips to True with a
        spec_type that has defaults.
        """
        self._refresh_spec_tab_state()
        self._apply_spec_defaults_if_blank()
        self._apply_mtp_parallel_default()

    def _on_spec_type_changed(self):
        """Trace callback chained after ``spec_type`` writes.

        Same shape as ``_on_spec_enabled_changed`` — refresh visibility,
        then top up blank draft tuning fields with the per-type defaults.
        """
        self._refresh_spec_tab_state()
        self._apply_spec_defaults_if_blank()
        self._apply_mtp_parallel_default()

    def _refresh_spec_tab_state(self):
        """Recompute visibility/enabled state for MTP/Spec tab widgets.

        Called from traces on backend_selection, spec_enabled, and spec_type,
        plus once on tab creation and once on _on_backend_selection_changed.
        Safe to call before widgets exist (no-ops gracefully).
        """
        # The tab may not be built yet during early init — bail quietly.
        if not hasattr(self, "_spec_widgets") or not hasattr(self, "_spec_sections"):
            return

        backend = self.backend_selection.get() if hasattr(self, "backend_selection") else "llama.cpp"
        is_ik = backend == "ik_llama"
        enabled = bool(self.spec_enabled.get())
        spec_type = (self.spec_type.get() or "none").strip()

        # 1) Refresh the spec_type combobox values for the active backend.
        # ``effective_spec_type`` is what drives this tab's visibility/state for
        # the *current* backend. The stored ``self.spec_type`` is left alone so
        # a user who flips backends to inspect the other side, then flips back,
        # doesn't silently lose their previously-selected value (e.g. a
        # ``draft-mtp`` setting under llama.cpp survives a brief ik_llama
        # excursion). build_cmd() independently re-validates against the per-
        # backend whitelist, so emission is safe regardless.
        combo = self._spec_widgets.get("type_combo")
        allowed = list(self._SPEC_TYPES_IK_LLAMA if is_ik else self._SPEC_TYPES_LLAMA_CPP)
        if combo is not None:
            try:
                combo["values"] = allowed
            except tk.TclError:
                pass
        spec_type_is_valid_for_backend = spec_type in allowed
        effective_spec_type = spec_type if spec_type_is_valid_for_backend else "none"

        # 2) Master enable state: when off, everything except the master checkbox
        # is disabled. When on, all *visible* widgets default to enabled and the
        # per-backend/per-type rules below trim further.
        def _set_state(widget, state):
            try:
                # Combobox needs explicit "readonly" rather than "normal".
                if isinstance(widget, ttk.Combobox) and state == "normal":
                    widget.configure(state="readonly")
                else:
                    widget.configure(state=state)
            except (tk.TclError, AttributeError):
                pass

        # Iterate all child widgets in each section and set state uniformly.
        # Recurses into nested frames so the draft device checkbox grid (which
        # lives inside its own ttk.Frame) and the draft GPU-layers frame
        # (Entry + Slider + status Label) also pick up the right state. Frames
        # themselves don't accept a "state" so we skip them and recurse.
        def _set_section_state(section_name, state):
            sec = self._spec_sections.get(section_name)
            if sec is None:
                return

            def _walk(parent):
                for child in parent.winfo_children():
                    if isinstance(child, (ttk.Frame, tk.Frame, ttk.LabelFrame)):
                        _walk(child)
                        continue
                    _set_state(child, state)

            _walk(sec)

        type_combo_target_state = "normal" if enabled else "disabled"
        _set_state(combo, type_combo_target_state)

        # 3) Section visibility based on backend + spec_type.
        all_sections = set(self._spec_sections.keys())
        # Determine which sections to show.
        visible = set()
        # Vision (--no-mmproj) is independent of spec_enabled: a user may want
        # to suppress an embedded mmproj projector regardless of speculative
        # decoding. Always visible on llama.cpp.
        if not is_ik:
            visible.add("vision")
        if enabled:
            # Below uses ``effective_spec_type`` so a stored value that's
            # invalid for the active backend (e.g. ``draft-mtp`` while ik_llama
            # is active) collapses to "none" for visibility/state purposes
            # without mutating the stored ``self.spec_type``.
            # "Common draft controls" is shown for any non-none type (draft/mtp/ngram/suffix
            # all benefit from n-max/n-min/p-min knobs; backend-specific gating handles
            # p-split disable on ik_llama).
            if effective_spec_type and effective_spec_type != "none":
                visible.add("common")
            # Draft model section: shown for the spec_types that actually use
            # a separate draft model. On llama.cpp this means draft-simple /
            # draft-eagle3 (draft-mtp shares the base GGUF). On ik_llama, the
            # legacy --model-draft FNAME flag is also supported for mtp mode.
            if effective_spec_type in ("draft-simple", "draft-eagle3") or (is_ik and effective_spec_type == "mtp"):
                visible.add("draft_model")
            # Ngram sections - mainline has per-variant; ik_llama has shared.
            if effective_spec_type.startswith("ngram-"):
                if is_ik:
                    visible.add("ngram_shared")
                else:
                    if effective_spec_type == "ngram-simple":
                        visible.add("ngram_simple")
                    elif effective_spec_type == "ngram-map-k":
                        visible.add("ngram_mapk")
                    elif effective_spec_type == "ngram-map-k4v":
                        visible.add("ngram_mapk4v")
                    elif effective_spec_type == "ngram-mod":
                        visible.add("ngram_mod")
                    # ngram-cache has no extra knobs - no section to show.
            # Suffix only on ik_llama.
            if is_ik and effective_spec_type == "suffix":
                visible.add("suffix")
            # ik_llama extras shown whenever ik_llama is active and master is on.
            if is_ik:
                visible.add("ik_extras")

        for name in all_sections:
            sec = self._spec_sections[name]
            try:
                if name in visible:
                    sec.grid()
                else:
                    sec.grid_remove()
            except tk.TclError:
                pass

        # 4) Per-widget enable/disable inside visible sections.
        if enabled:
            for name in visible:
                _set_section_state(name, "normal")
            # p-split is llama.cpp only - if ik_llama is active, disable it.
            psplit_w = self._spec_widgets.get("p_split")
            if psplit_w is not None and "common" in visible:
                if is_ik:
                    _set_state(psplit_w, "disabled")
                    self.spec_psplit_hint_var.set("(disabled: ik_llama does not support --spec-draft-p-split)")
                else:
                    _set_state(psplit_w, "normal")
                    self.spec_psplit_hint_var.set("(llama.cpp only)")
            # p-min on draft-mtp: leave editable but warn the user via hint label.
            if "common" in visible:
                if effective_spec_type == "draft-mtp":
                    self.spec_pmin_hint_var.set("Note: currently disabled for MTP in mainline (post-merge TODO).")
                else:
                    self.spec_pmin_hint_var.set("")
                # MTP constraint hint: surface the --parallel 1 requirement.
                if effective_spec_type in ("draft-mtp", "mtp"):
                    self.spec_parallel_hint_var.set(
                        "Note: MTP requires --parallel 1 (single-slot). The launcher "
                        "auto-sets and enforces this at launch — overrides from elsewhere "
                        "are ignored while MTP is active."
                    )
                else:
                    self.spec_parallel_hint_var.set("")
            # cpu-moe knobs are llama.cpp-only when the draft_model section is visible.
            if "draft_model" in visible:
                for k in ("draft_cpu_moe", "draft_n_cpu_moe"):
                    w = self._spec_widgets.get(k)
                    if w is not None:
                        _set_state(w, "disabled" if is_ik else "normal")
                # Draft GPU-layer slider must remain DISABLED until draft model
                # analysis succeeds (mirrors how the main slider only goes NORMAL
                # after analysis populates max_gpu_layers). The section walk
                # above would otherwise flip it to "normal" while the analysis
                # hasn't run.
                slider_w = self._spec_widgets.get("draft_ngl_slider")
                if slider_w is not None:
                    try:
                        max_draft = self.max_spec_draft_gpu_layers.get()
                    except (tk.TclError, AttributeError):
                        max_draft = 0
                    _set_state(slider_w, "normal" if max_draft > 0 else "disabled")
                # Opt-in toggle: only shown for ik_llama+mtp (the only mode where
                # the draft model is optional). For draft-simple/draft-eagle3 the
                # draft model is required, so hide the checkbox and force the
                # inner widgets visible regardless of the stored value.
                use_cb = self._spec_widgets.get("use_draft_cb")
                is_optional_draft = is_ik and effective_spec_type == "mtp"
                if use_cb is not None:
                    try:
                        if is_optional_draft:
                            use_cb.grid()
                        else:
                            use_cb.grid_remove()
                    except tk.TclError:
                        pass
                # Show or hide the inner draft widgets (listbox, path, GPU
                # layers, devices, cache types, cpu_moe) based on the checkbox
                # for ik_llama+mtp; always show them for required-draft modes.
                show_inner = (not is_optional_draft) or bool(self.spec_use_draft_model.get())
                inner_widgets = getattr(self, "_spec_draft_inner_widgets", []) or []
                for w in inner_widgets:
                    try:
                        if show_inner:
                            w.grid()
                        else:
                            w.grid_remove()
                    except tk.TclError:
                        pass
        else:
            # Master off: disable everything except the master checkbox AND
            # the vision section (--no-mmproj is independent of spec_enabled).
            for name in all_sections:
                _set_section_state(name, "disabled")
            if "vision" in visible:
                _set_section_state("vision", "normal")
            self.spec_pmin_hint_var.set("")
            try:
                self.spec_parallel_hint_var.set("")
            except (AttributeError, tk.TclError):
                pass

        # 5) Status label so users know what's emitted. Surface the
        # "stored but inactive on this backend" case explicitly so a user
        # who flipped backends knows their setting is preserved.
        backend_label = "ik_llama" if is_ik else "llama.cpp"
        if not enabled:
            self.spec_status_var.set("Disabled - no --spec-* / --draft-* flags will be emitted.")
        elif not spec_type_is_valid_for_backend and spec_type not in ("", "none"):
            self.spec_status_var.set(
                f"Stored type '{spec_type}' is not valid for {backend_label} - inactive on this "
                f"backend (value preserved). Pick a valid type or switch backend to use it."
            )
        elif effective_spec_type in ("", "none"):
            self.spec_status_var.set("Enabled, but type is 'none' - no spec flags will be emitted.")
        else:
            self.spec_status_var.set(f"Active: type={effective_spec_type} (backend: {backend_label}).")
