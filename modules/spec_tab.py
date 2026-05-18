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
import tkinter as tk
from pathlib import Path
from threading import Thread
from tkinter import ttk

from modules.system import parse_gguf_header_simple


# Allowed values for the draft KV cache type comboboxes. Leading "" lets the
# user pick "don't emit the flag" (emission code already treats blank as
# omission). Module-level so tests can import and assert against the SAME
# tuple the UI actually uses — preventing silent drift when the set changes.
SPEC_DRAFT_CACHE_TYPE_VALUES = (
    "", "f16", "f32", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "q6_k",
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
        self.spec_enabled        = tk.BooleanVar(value=ib(app_settings, "spec_enabled"))
        self.spec_type           = tk.StringVar(value=is_(app_settings, "spec_type", "none") or "none")
        # Common draft controls (numeric entries; blank = use binary default).
        self.spec_draft_n_max    = tk.StringVar(value=is_(app_settings, "spec_draft_n_max"))
        self.spec_draft_n_min    = tk.StringVar(value=is_(app_settings, "spec_draft_n_min"))
        self.spec_draft_p_min    = tk.StringVar(value=is_(app_settings, "spec_draft_p_min"))
        self.spec_draft_p_split  = tk.StringVar(value=is_(app_settings, "spec_draft_p_split"))   # llama.cpp only
        # Draft model selection.
        self.spec_draft_model    = tk.StringVar(value=is_(app_settings, "spec_draft_model"))    # -md path
        # Opt-in for ik_llama+mtp: when False, hide the draft picker UI AND
        # suppress --model-draft / draft offload emission so the embedded MTP
        # head in the base GGUF is used. Required-draft modes (draft-simple /
        # draft-eagle3 on llama.cpp) ignore this and always emit draft flags.
        self.spec_use_draft_model = tk.BooleanVar(value=ib(app_settings, "spec_use_draft_model"))
        self.spec_draft_ngl      = tk.StringVar(value=is_(app_settings, "spec_draft_ngl"))
        self.spec_draft_device   = tk.StringVar(value=is_(app_settings, "spec_draft_device"))
        self.spec_draft_ctk      = tk.StringVar(value=is_(app_settings, "spec_draft_ctk"))
        self.spec_draft_ctv      = tk.StringVar(value=is_(app_settings, "spec_draft_ctv"))
        self.spec_draft_cpu_moe  = tk.BooleanVar(value=ib(app_settings, "spec_draft_cpu_moe"))  # llama.cpp only
        self.spec_draft_n_cpu_moe= tk.StringVar(value=is_(app_settings, "spec_draft_n_cpu_moe"))  # llama.cpp only
        # Derived/UI state for the draft model's GPU layer slider + status (mirrors
        # self.n_gpu_layers_int / self.max_gpu_layers / self.gpu_layers_status_var
        # for the main model). Not persisted directly — set after draft GGUF
        # analysis succeeds and consumed only by the slider widget + status label.
        self.spec_draft_ngl_int           = tk.IntVar(value=0)
        self.max_spec_draft_gpu_layers    = tk.IntVar(value=0)
        self.spec_draft_layers_status_var = tk.StringVar(value="Select draft model to see layer info")
        self.current_spec_draft_analysis  = {}  # mirrors self.current_model_analysis
        # Ngram tuning (llama.cpp has per-variant size sets; ik_llama has a single shared set).
        self.spec_ngram_simple_size_n   = tk.StringVar(value=is_(app_settings, "spec_ngram_simple_size_n"))
        self.spec_ngram_simple_size_m   = tk.StringVar(value=is_(app_settings, "spec_ngram_simple_size_m"))
        self.spec_ngram_simple_min_hits = tk.StringVar(value=is_(app_settings, "spec_ngram_simple_min_hits"))
        self.spec_ngram_mapk_size_n     = tk.StringVar(value=is_(app_settings, "spec_ngram_mapk_size_n"))
        self.spec_ngram_mapk_size_m     = tk.StringVar(value=is_(app_settings, "spec_ngram_mapk_size_m"))
        self.spec_ngram_mapk_min_hits   = tk.StringVar(value=is_(app_settings, "spec_ngram_mapk_min_hits"))
        self.spec_ngram_mapk4v_size_n   = tk.StringVar(value=is_(app_settings, "spec_ngram_mapk4v_size_n"))
        self.spec_ngram_mapk4v_size_m   = tk.StringVar(value=is_(app_settings, "spec_ngram_mapk4v_size_m"))
        self.spec_ngram_mapk4v_min_hits = tk.StringVar(value=is_(app_settings, "spec_ngram_mapk4v_min_hits"))
        self.spec_ngram_mod_n_min       = tk.StringVar(value=is_(app_settings, "spec_ngram_mod_n_min"))
        self.spec_ngram_mod_n_max       = tk.StringVar(value=is_(app_settings, "spec_ngram_mod_n_max"))
        self.spec_ngram_mod_n_match     = tk.StringVar(value=is_(app_settings, "spec_ngram_mod_n_match"))
        # Shared single ngram set used by ik_llama (one --spec-ngram-* set).
        self.spec_ngram_size_n          = tk.StringVar(value=is_(app_settings, "spec_ngram_size_n"))
        self.spec_ngram_size_m          = tk.StringVar(value=is_(app_settings, "spec_ngram_size_m"))
        self.spec_ngram_min_hits        = tk.StringVar(value=is_(app_settings, "spec_ngram_min_hits"))
        # Suffix tuning (ik_llama only).
        self.spec_suffix_pattern_len    = tk.StringVar(value=is_(app_settings, "spec_suffix_pattern_len"))
        self.spec_suffix_max_depth      = tk.StringVar(value=is_(app_settings, "spec_suffix_max_depth"))
        # ik_llama extras.
        self.spec_autotune              = tk.BooleanVar(value=ib(app_settings, "spec_autotune"))
        self.spec_draft_params          = tk.StringVar(value=is_(app_settings, "spec_draft_params"))  # -draft "k=v,k=v"
        # llama.cpp vision toggle.
        self.no_mmproj                  = tk.BooleanVar(value=ib(app_settings, "no_mmproj"))  # --no-mmproj

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
        ttk.Label(inner, text="MTP / Speculative Decoding", font=("TkDefaultFont", 12, "bold"))\
            .grid(column=0, row=r, sticky="w", padx=10, pady=(10, 5), columnspan=4); r += 1
        ttk.Separator(inner, orient="horizontal")\
            .grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=5); r += 1

        master_cb = ttk.Checkbutton(
            inner,
            text="Enable speculative decoding",
            variable=self.spec_enabled,
        )
        master_cb.grid(column=0, row=r, sticky="w", padx=10, pady=4, columnspan=4); r += 1
        self._spec_widgets["master_cb"] = master_cb

        self.spec_status_var = tk.StringVar(value="")
        ttk.Label(inner, textvariable=self.spec_status_var, foreground="gray")\
            .grid(column=0, row=r, sticky="w", padx=10, pady=(0, 6), columnspan=4); r += 1

        # --- Speculative type ---
        ttk.Label(inner, text="Speculative type:", font=("TkDefaultFont", 10, "bold"))\
            .grid(column=0, row=r, sticky="w", padx=10, pady=(8, 2), columnspan=4); r += 1
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
        ttk.Label(sec, textvariable=self.spec_pmin_hint_var, foreground="gray")\
            .grid(column=2, row=sr, sticky="w", padx=4, pady=2, columnspan=2)

        sr += 1
        ttk.Label(sec, text="p-split:").grid(column=0, row=sr, sticky="w", padx=6, pady=2)
        e_psplit = ttk.Entry(sec, textvariable=self.spec_draft_p_split, width=10)
        e_psplit.grid(column=1, row=sr, sticky="w", padx=4, pady=2)
        self._spec_widgets["p_split"] = e_psplit
        self.spec_psplit_hint_var = tk.StringVar(value="(llama.cpp only)")
        ttk.Label(sec, textvariable=self.spec_psplit_hint_var, foreground="gray")\
            .grid(column=2, row=sr, sticky="w", padx=4, pady=2, columnspan=2)

        # MTP requires --parallel 1 (single-slot operation). The trace
        # callbacks force this when MTP is selected, but show the hint
        # so users understand what's happening and can verify.
        sr += 1
        self.spec_parallel_hint_var = tk.StringVar(value="")
        ttk.Label(sec, textvariable=self.spec_parallel_hint_var,
                  foreground="#888888", font=("TkSmallCaptionFont"))\
            .grid(column=0, row=sr, columnspan=4, sticky="w", padx=6, pady=(4, 2))

        # Reset-to-default button: overwrites all four common controls with
        # the recommended values for the current spec_type. For ngram/suffix
        # types (which have no recommended defaults), clears the fields.
        sr += 1
        reset_btn = ttk.Button(sec, text="Reset to defaults",
                               command=self._reset_spec_defaults)
        reset_btn.grid(column=0, row=sr, sticky="w", padx=6, pady=(6, 4))
        self._spec_widgets["reset_defaults_btn"] = reset_btn
        ttk.Label(sec, text="Overwrites n-max / n-min / p-min / p-split with the recommended defaults for the active type.",
                  foreground="#888888", font=("TkSmallCaptionFont"))\
            .grid(column=1, row=sr, columnspan=3, sticky="w", padx=4, pady=(6, 4))

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
            draft_ngl_frame, textvariable=self.spec_draft_ngl, width=6, state=tk.NORMAL,
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
            column=1, row=sr, columnspan=3, sticky="ew", padx=4, pady=2,
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
            sec, textvariable=self.spec_draft_ctk, width=10,
            values=SPEC_DRAFT_CACHE_TYPE_VALUES, state="readonly",
        )
        self.spec_draft_ctk_combo.grid(column=1, row=sr, sticky="w", padx=4, pady=2)
        self._spec_widgets["draft_ctk"] = self.spec_draft_ctk_combo
        ttk.Label(sec, text="Draft V cache type (-ctvd):").grid(column=2, row=sr, sticky="w", padx=6, pady=2)
        self.spec_draft_ctv_combo = ttk.Combobox(
            sec, textvariable=self.spec_draft_ctv, width=10,
            values=SPEC_DRAFT_CACHE_TYPE_VALUES, state="readonly",
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
        self._spec_draft_inner_widgets = [
            w for w in sec.winfo_children() if w is not self.spec_use_draft_cb
        ]

        # --- Ngram tuning (llama.cpp per-variant; ik_llama shared) ---
        # Per-variant simple/mapk/mapk4v/mod groups for llama.cpp:
        for key, label, vars_triplet in [
            ("ngram_simple", "Ngram simple (--spec-ngram-simple-*)",
             (self.spec_ngram_simple_size_n, self.spec_ngram_simple_size_m, self.spec_ngram_simple_min_hits)),
            ("ngram_mapk", "Ngram map-k (--spec-ngram-map-k-*)",
             (self.spec_ngram_mapk_size_n, self.spec_ngram_mapk_size_m, self.spec_ngram_mapk_min_hits)),
            ("ngram_mapk4v", "Ngram map-k4v (--spec-ngram-map-k4v-*)",
             (self.spec_ngram_mapk4v_size_n, self.spec_ngram_mapk4v_size_m, self.spec_ngram_mapk4v_min_hits)),
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
        ttk.Entry(sec, textvariable=self.spec_ngram_mod_n_min, width=10)\
            .grid(column=1, row=0, sticky="w", padx=4, pady=2)
        ttk.Label(sec, text="n-max:").grid(column=2, row=0, sticky="w", padx=6, pady=2)
        ttk.Entry(sec, textvariable=self.spec_ngram_mod_n_max, width=10)\
            .grid(column=3, row=0, sticky="w", padx=4, pady=2)
        ttk.Label(sec, text="n-match:").grid(column=0, row=1, sticky="w", padx=6, pady=2)
        ttk.Entry(sec, textvariable=self.spec_ngram_mod_n_match, width=10)\
            .grid(column=1, row=1, sticky="w", padx=4, pady=2)

        # Shared ngram set (ik_llama uses a single set across all ngram types):
        sec = ttk.LabelFrame(inner, text="Ngram tuning (--spec-ngram-*)")
        sec.grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=4)
        sec.columnconfigure(1, weight=1)
        sec.columnconfigure(3, weight=1)
        self._spec_sections["ngram_shared"] = sec
        r += 1
        ttk.Label(sec, text="size-n:").grid(column=0, row=0, sticky="w", padx=6, pady=2)
        ttk.Entry(sec, textvariable=self.spec_ngram_size_n, width=10)\
            .grid(column=1, row=0, sticky="w", padx=4, pady=2)
        ttk.Label(sec, text="size-m:").grid(column=2, row=0, sticky="w", padx=6, pady=2)
        ttk.Entry(sec, textvariable=self.spec_ngram_size_m, width=10)\
            .grid(column=3, row=0, sticky="w", padx=4, pady=2)
        ttk.Label(sec, text="min-hits:").grid(column=0, row=1, sticky="w", padx=6, pady=2)
        ttk.Entry(sec, textvariable=self.spec_ngram_min_hits, width=10)\
            .grid(column=1, row=1, sticky="w", padx=4, pady=2)

        # --- Suffix (ik_llama only) ---
        sec = ttk.LabelFrame(inner, text="Suffix tuning (ik_llama, --suffix-*)")
        sec.grid(column=0, row=r, columnspan=4, sticky="ew", padx=10, pady=4)
        sec.columnconfigure(1, weight=1)
        self._spec_sections["suffix"] = sec
        r += 1
        ttk.Label(sec, text="pattern-len:").grid(column=0, row=0, sticky="w", padx=6, pady=2)
        ttk.Entry(sec, textvariable=self.spec_suffix_pattern_len, width=10)\
            .grid(column=1, row=0, sticky="w", padx=4, pady=2)
        ttk.Label(sec, text="max-depth:").grid(column=2, row=0, sticky="w", padx=6, pady=2)
        ttk.Entry(sec, textvariable=self.spec_suffix_max_depth, width=10)\
            .grid(column=3, row=0, sticky="w", padx=4, pady=2)

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
        ttk.Entry(sec, textvariable=self.spec_draft_params)\
            .grid(column=1, row=1, sticky="ew", padx=4, pady=2, columnspan=2)
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
                if (
                    hasattr(self, "spec_draft_ngl_slider")
                    and self.spec_draft_ngl_slider.winfo_exists()
                ):
                    self.spec_draft_ngl_slider.config(state=tk.DISABLED)
                self.current_spec_draft_analysis = {}
                t = Thread(
                    target=self._run_spec_draft_gguf_analysis,
                    args=(full_path_str,),
                    daemon=True,
                )
                t.start()
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
        if (
            not hasattr(self, "spec_draft_ngl_entry")
            or not self.spec_draft_ngl_entry.winfo_exists()
        ):
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
        if (
            not hasattr(self, "spec_draft_ngl_entry")
            or not self.spec_draft_ngl_entry.winfo_exists()
        ):
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
        """
        if (
            not hasattr(self, "spec_draft_gpu_checkbox_frame")
            or not self.spec_draft_gpu_checkbox_frame.winfo_exists()
        ):
            return
        for w in self.spec_draft_gpu_checkbox_frame.winfo_children():
            w.destroy()
        self.spec_draft_gpu_vars = []

        gpu_info = getattr(self.launcher, "gpu_info", {})
        count = gpu_info.get("device_count", 0) if isinstance(gpu_info, dict) else 0
        loaded_selected = set(self.launcher.app_settings.get("spec_draft_selected_gpus", []) or [])
        detected_devices = getattr(self.launcher, "detected_gpu_devices", [])
        # Manual GPU mode disables draft device emission entirely — the
        # manual GPU list isn't real CUDA hardware, so we can't tell the
        # binary "use CUDA<i>" reliably.
        manual_mode = bool(
            getattr(getattr(self.launcher, "manual_gpu_mode", None), "get", lambda: False)()
        )
        # Sanitize: rebuild the persisted-index list from what's currently
        # valid, so a stale saved selection (e.g. GPUs that no longer exist
        # or were filtered, or any selection while manual GPU mode is on)
        # never re-emits as a phantom CUDA<i>. Also derive a fresh device
        # string so spec_draft_device matches the visible checkbox state.
        valid_selected = []

        if count > 0 and not manual_mode:
            MAX_GPUS_PER_ROW = 3
            for i in range(count):
                gpu_details = (
                    detected_devices[i]
                    if i < len(detected_devices)
                    else {}
                )
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
                text=("No CUDA devices detected." if not manual_mode
                      else "Draft device selection disabled in manual GPU mode."),
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
            try:
                self.spec_draft_device.set(
                    ",".join(f"CUDA{i}" for i in valid_selected)
                )
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
        try:
            selected_indices = [
                i for i, v in enumerate(self.spec_draft_gpu_vars) if v.get()
            ]
            self.launcher.app_settings["spec_draft_selected_gpus"] = selected_indices
            device_str = ",".join(f"CUDA{i}" for i in selected_indices)
            if self.spec_draft_device.get() != device_str:
                self.spec_draft_device.set(device_str)
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

    def _run_spec_draft_gguf_analysis(self, draft_path_str):
        """Background worker that parses the draft GGUF and dispatches the
        result back onto the Tk thread."""
        try:
            if self.spec_draft_model.get() != draft_path_str:
                return  # selection changed before we even started
            analysis_result = parse_gguf_header_simple(draft_path_str)
            if self.spec_draft_model.get() == draft_path_str:
                self.launcher.root.after(0, self._update_ui_after_spec_draft_analysis, analysis_result)
        except Exception as e:
            print(f"WARN: spec draft GGUF analysis failed: {e}", file=sys.stderr)

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
            if (
                hasattr(self, "spec_draft_ngl_slider")
                and self.spec_draft_ngl_slider.winfo_exists()
            ):
                self.spec_draft_ngl_slider.config(to=0, state=tk.DISABLED)
            return
        # Success: enable slider and update status. Mirrors main +1 for output.
        max_offloadable = n_layers + 1
        self.max_spec_draft_gpu_layers.set(max_offloadable)
        self.spec_draft_layers_status_var.set(
            f"Max Layers: {max_offloadable} ({n_layers} blocks + output)"
        )
        if (
            hasattr(self, "spec_draft_ngl_slider")
            and self.spec_draft_ngl_slider.winfo_exists()
        ):
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
        is_ik = (backend == "ik_llama")
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
