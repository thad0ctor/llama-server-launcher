"""Single source of truth for launcher Tk var defaults used by mock fixtures.

Five sibling test modules each need to populate a MockLauncher with the same
catalog of speculative-decoding / reasoning / KV-unification Tk vars. When the
launcher gains a new var, every mock builder used to need a parallel edit;
this registry collapses that to a single edit here.

Each tuple is ``(attribute_name, tk_var_class_name, default_value)``.

``tk_var_class_name`` is a string ("StringVar" | "BooleanVar" | "IntVar" |
"DoubleVar") so consumers can resolve it to either the real ``tkinter`` class
or a test stand-in (e.g. ``FakeVar``) at their discretion.

The grouping into ``SPEC_TK_VARS`` / ``REASONING_TK_VARS`` / ``KVU_TK_VARS``
mirrors the three discrete sections in ``llamacpp-server-launcher.py``'s
``__init__``. Callers that don't care about the grouping can use
``ALL_NEW_LAUNCHER_TK_VARS``.
"""

from __future__ import annotations


# --- MTP / Speculative Decoding ---------------------------------------------
# Mirrors the ``self.spec_*`` (+ ``self.no_mmproj``) declarations in
# llamacpp-server-launcher.py around the ``_spec_init_bool`` /
# ``_spec_init_str`` helpers. Order matches the source file for ease of
# audit.
SPEC_TK_VARS = [
    ("spec_enabled",                "BooleanVar", False),
    ("spec_type",                   "StringVar",  "none"),
    # Common draft controls
    ("spec_draft_n_max",            "StringVar",  ""),
    ("spec_draft_n_min",            "StringVar",  ""),
    ("spec_draft_p_min",            "StringVar",  ""),
    ("spec_draft_p_split",          "StringVar",  ""),
    # Draft model selection
    ("spec_draft_model",            "StringVar",  ""),
    ("spec_use_draft_model",        "BooleanVar", False),
    ("spec_draft_ngl",              "StringVar",  ""),
    ("spec_draft_device",           "StringVar",  ""),
    ("spec_draft_ctk",              "StringVar",  ""),
    ("spec_draft_ctv",              "StringVar",  ""),
    ("spec_draft_cpu_moe",          "BooleanVar", False),
    ("spec_draft_n_cpu_moe",        "StringVar",  ""),
    # Ngram tuning (llama.cpp per-variant)
    ("spec_ngram_simple_size_n",    "StringVar",  ""),
    ("spec_ngram_simple_size_m",    "StringVar",  ""),
    ("spec_ngram_simple_min_hits",  "StringVar",  ""),
    ("spec_ngram_mapk_size_n",      "StringVar",  ""),
    ("spec_ngram_mapk_size_m",      "StringVar",  ""),
    ("spec_ngram_mapk_min_hits",    "StringVar",  ""),
    ("spec_ngram_mapk4v_size_n",    "StringVar",  ""),
    ("spec_ngram_mapk4v_size_m",    "StringVar",  ""),
    ("spec_ngram_mapk4v_min_hits",  "StringVar",  ""),
    ("spec_ngram_mod_n_min",        "StringVar",  ""),
    ("spec_ngram_mod_n_max",        "StringVar",  ""),
    ("spec_ngram_mod_n_match",      "StringVar",  ""),
    # Shared single ngram set (ik_llama)
    ("spec_ngram_size_n",           "StringVar",  ""),
    ("spec_ngram_size_m",           "StringVar",  ""),
    ("spec_ngram_min_hits",         "StringVar",  ""),
    # Suffix tuning (ik_llama)
    ("spec_suffix_pattern_len",     "StringVar",  ""),
    ("spec_suffix_max_depth",       "StringVar",  ""),
    # ik_llama extras
    ("spec_autotune",               "BooleanVar", False),
    ("spec_draft_params",           "StringVar",  ""),
    # llama.cpp vision toggle (lives in the spec init block in the source)
    ("no_mmproj",                   "BooleanVar", False),
]


# --- Reasoning / Thinking (both backends) -----------------------------------
REASONING_TK_VARS = [
    ("reasoning_mode",              "StringVar",  ""),
    ("reasoning_format",            "StringVar",  ""),
    ("reasoning_budget",            "StringVar",  ""),
    ("reasoning_budget_message",    "StringVar",  ""),
    ("chat_template_kwargs",        "StringVar",  ""),
]


# --- KV Unification + cache-idle-slots (llama.cpp only) ---------------------
KVU_TK_VARS = [
    ("kv_unified_mode",             "StringVar",  ""),
    ("cache_idle_slots_mode",       "StringVar",  ""),
]


# Convenience aggregate for callers that don't care about the grouping.
ALL_NEW_LAUNCHER_TK_VARS = SPEC_TK_VARS + REASONING_TK_VARS + KVU_TK_VARS
