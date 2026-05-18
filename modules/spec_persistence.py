#!/usr/bin/env python3
"""
Spec / MTP / Reasoning / KV-Unification persistence helpers.

This module owns the cfg-dict and app_settings-dict mirroring for every
``spec_*`` / ``reasoning_*`` / ``kv_unified_*`` / ``cache_idle_slots_*`` /
``no_mmproj`` key persisted across launcher sessions. Extracted from
``modules/config.py`` so the named-config save/load and the app_settings
save/load all share a single source of truth for which keys belong to the
MTP/Spec feature.

The launcher's Tk vars are owned by ``modules.spec_tab.SpecTab`` (re-exposed
on the launcher itself for backward compat with existing call sites). These
helpers read/write those vars via ``launcher.<var_name>`` exactly the way
the inline blocks did before extraction.
"""

# Exhaustive list of boolean keys. Mirrors `_spec_bool_keys` in the
# original `load_saved_configs` validation block.
SPEC_BOOL_KEYS = (
    "spec_enabled", "spec_draft_cpu_moe", "spec_autotune", "no_mmproj",
    "spec_use_draft_model",
)

# Exhaustive list of string keys. Mirrors `_spec_str_keys` in the original
# `load_saved_configs` validation block, including reasoning/KV-unification
# keys which all default to "" so the corresponding flag is omitted.
SPEC_STR_KEYS = (
    "spec_type",
    "spec_draft_n_max", "spec_draft_n_min", "spec_draft_p_min", "spec_draft_p_split",
    "spec_draft_model", "spec_draft_ngl", "spec_draft_device",
    "spec_draft_ctk", "spec_draft_ctv", "spec_draft_n_cpu_moe",
    "spec_ngram_simple_size_n", "spec_ngram_simple_size_m", "spec_ngram_simple_min_hits",
    "spec_ngram_mapk_size_n", "spec_ngram_mapk_size_m", "spec_ngram_mapk_min_hits",
    "spec_ngram_mapk4v_size_n", "spec_ngram_mapk4v_size_m", "spec_ngram_mapk4v_min_hits",
    "spec_ngram_mod_n_min", "spec_ngram_mod_n_max", "spec_ngram_mod_n_match",
    "spec_ngram_size_n", "spec_ngram_size_m", "spec_ngram_min_hits",
    "spec_suffix_pattern_len", "spec_suffix_max_depth",
    "spec_draft_params",
    # Reasoning / Thinking + KV Unification keys. All start as "" so
    # the corresponding flags are omitted by default.
    "reasoning_mode", "reasoning_format", "reasoning_budget",
    "reasoning_budget_message", "chat_template_kwargs",
    "kv_unified_mode", "cache_idle_slots_mode",
)


def collect_spec_into_cfg(launcher, cfg):
    """Mirror every spec/reasoning/kvu Tk var into the named-config ``cfg``
    dict.

    Called from ``ConfigManager.current_cfg`` (formerly the inline block in
    ``_collect_config``); also bundles the defensive
    ``spec_draft_selected_gpus`` mirror so the checkbox state round-trips
    in the named-config payload.
    """
    cfg["spec_enabled"]               = launcher.spec_enabled.get()
    cfg["spec_type"]                  = launcher.spec_type.get()
    cfg["spec_draft_n_max"]           = launcher.spec_draft_n_max.get()
    cfg["spec_draft_n_min"]           = launcher.spec_draft_n_min.get()
    cfg["spec_draft_p_min"]           = launcher.spec_draft_p_min.get()
    cfg["spec_draft_p_split"]         = launcher.spec_draft_p_split.get()
    cfg["spec_draft_model"]           = launcher.spec_draft_model.get()
    cfg["spec_use_draft_model"]       = launcher.spec_use_draft_model.get()
    cfg["spec_draft_ngl"]             = launcher.spec_draft_ngl.get()
    cfg["spec_draft_device"]          = launcher.spec_draft_device.get()
    cfg["spec_draft_ctk"]             = launcher.spec_draft_ctk.get()
    cfg["spec_draft_ctv"]             = launcher.spec_draft_ctv.get()
    cfg["spec_draft_cpu_moe"]         = launcher.spec_draft_cpu_moe.get()
    cfg["spec_draft_n_cpu_moe"]       = launcher.spec_draft_n_cpu_moe.get()
    cfg["spec_ngram_simple_size_n"]   = launcher.spec_ngram_simple_size_n.get()
    cfg["spec_ngram_simple_size_m"]   = launcher.spec_ngram_simple_size_m.get()
    cfg["spec_ngram_simple_min_hits"] = launcher.spec_ngram_simple_min_hits.get()
    cfg["spec_ngram_mapk_size_n"]     = launcher.spec_ngram_mapk_size_n.get()
    cfg["spec_ngram_mapk_size_m"]     = launcher.spec_ngram_mapk_size_m.get()
    cfg["spec_ngram_mapk_min_hits"]   = launcher.spec_ngram_mapk_min_hits.get()
    cfg["spec_ngram_mapk4v_size_n"]   = launcher.spec_ngram_mapk4v_size_n.get()
    cfg["spec_ngram_mapk4v_size_m"]   = launcher.spec_ngram_mapk4v_size_m.get()
    cfg["spec_ngram_mapk4v_min_hits"] = launcher.spec_ngram_mapk4v_min_hits.get()
    cfg["spec_ngram_mod_n_min"]       = launcher.spec_ngram_mod_n_min.get()
    cfg["spec_ngram_mod_n_max"]       = launcher.spec_ngram_mod_n_max.get()
    cfg["spec_ngram_mod_n_match"]     = launcher.spec_ngram_mod_n_match.get()
    cfg["spec_ngram_size_n"]          = launcher.spec_ngram_size_n.get()
    cfg["spec_ngram_size_m"]          = launcher.spec_ngram_size_m.get()
    cfg["spec_ngram_min_hits"]        = launcher.spec_ngram_min_hits.get()
    cfg["spec_suffix_pattern_len"]    = launcher.spec_suffix_pattern_len.get()
    cfg["spec_suffix_max_depth"]      = launcher.spec_suffix_max_depth.get()
    cfg["spec_autotune"]              = launcher.spec_autotune.get()
    cfg["spec_draft_params"]          = launcher.spec_draft_params.get()
    cfg["no_mmproj"]                  = launcher.no_mmproj.get()
    # --- Reasoning / Thinking (both backends) ---
    cfg["reasoning_mode"]             = launcher.reasoning_mode.get()
    cfg["reasoning_format"]           = launcher.reasoning_format.get()
    cfg["reasoning_budget"]           = launcher.reasoning_budget.get()
    cfg["reasoning_budget_message"]   = launcher.reasoning_budget_message.get()
    cfg["chat_template_kwargs"]       = launcher.chat_template_kwargs.get()
    # --- KV Unification (llama.cpp only) ---
    cfg["kv_unified_mode"]            = launcher.kv_unified_mode.get()
    cfg["cache_idle_slots_mode"]      = launcher.cache_idle_slots_mode.get()


def load_spec_from_cfg(launcher, cfg):
    """Set every spec/reasoning/kvu Tk var from the named-config ``cfg`` dict.

    Called from ``ConfigManager.load_configuration``. Uses the same
    permissive ``_spec_bool`` / ``_spec_str`` coercion as the original
    inline block so legacy/missing entries don't crash later set() calls.

    Also handles defensive coercion of ``spec_draft_selected_gpus`` (list
    of ints) — anything else is replaced with []. The caller still needs
    to copy the cleaned list into ``launcher.app_settings`` so the
    checkbox grid can render it; this function performs the coercion and
    returns the cleaned list as a side effect on the cfg dict.
    """
    def _spec_bool(key):
        val = cfg.get(key, False)
        return bool(val) if isinstance(val, bool) else (str(val).lower() in ("1", "true", "yes"))

    def _spec_str(key, default=""):
        val = cfg.get(key, default)
        return val if isinstance(val, str) else (str(val) if val is not None else default)

    # --- MTP / Speculative Decoding ---
    # Boolean toggles default False; strings default "" (= "don't emit");
    # spec_type defaults to "none" so older configs keep emitting nothing.
    launcher.spec_enabled.set(_spec_bool("spec_enabled"))
    launcher.spec_type.set(_spec_str("spec_type", "none") or "none")
    launcher.spec_draft_n_max.set(_spec_str("spec_draft_n_max"))
    launcher.spec_draft_n_min.set(_spec_str("spec_draft_n_min"))
    launcher.spec_draft_p_min.set(_spec_str("spec_draft_p_min"))
    launcher.spec_draft_p_split.set(_spec_str("spec_draft_p_split"))
    launcher.spec_draft_model.set(_spec_str("spec_draft_model"))
    launcher.spec_use_draft_model.set(_spec_bool("spec_use_draft_model"))
    launcher.spec_draft_ngl.set(_spec_str("spec_draft_ngl"))
    launcher.spec_draft_device.set(_spec_str("spec_draft_device"))
    launcher.spec_draft_ctk.set(_spec_str("spec_draft_ctk"))
    launcher.spec_draft_ctv.set(_spec_str("spec_draft_ctv"))
    launcher.spec_draft_cpu_moe.set(_spec_bool("spec_draft_cpu_moe"))
    launcher.spec_draft_n_cpu_moe.set(_spec_str("spec_draft_n_cpu_moe"))
    launcher.spec_ngram_simple_size_n.set(_spec_str("spec_ngram_simple_size_n"))
    launcher.spec_ngram_simple_size_m.set(_spec_str("spec_ngram_simple_size_m"))
    launcher.spec_ngram_simple_min_hits.set(_spec_str("spec_ngram_simple_min_hits"))
    launcher.spec_ngram_mapk_size_n.set(_spec_str("spec_ngram_mapk_size_n"))
    launcher.spec_ngram_mapk_size_m.set(_spec_str("spec_ngram_mapk_size_m"))
    launcher.spec_ngram_mapk_min_hits.set(_spec_str("spec_ngram_mapk_min_hits"))
    launcher.spec_ngram_mapk4v_size_n.set(_spec_str("spec_ngram_mapk4v_size_n"))
    launcher.spec_ngram_mapk4v_size_m.set(_spec_str("spec_ngram_mapk4v_size_m"))
    launcher.spec_ngram_mapk4v_min_hits.set(_spec_str("spec_ngram_mapk4v_min_hits"))
    launcher.spec_ngram_mod_n_min.set(_spec_str("spec_ngram_mod_n_min"))
    launcher.spec_ngram_mod_n_max.set(_spec_str("spec_ngram_mod_n_max"))
    launcher.spec_ngram_mod_n_match.set(_spec_str("spec_ngram_mod_n_match"))
    launcher.spec_ngram_size_n.set(_spec_str("spec_ngram_size_n"))
    launcher.spec_ngram_size_m.set(_spec_str("spec_ngram_size_m"))
    launcher.spec_ngram_min_hits.set(_spec_str("spec_ngram_min_hits"))
    launcher.spec_suffix_pattern_len.set(_spec_str("spec_suffix_pattern_len"))
    launcher.spec_suffix_max_depth.set(_spec_str("spec_suffix_max_depth"))
    launcher.spec_autotune.set(_spec_bool("spec_autotune"))
    launcher.spec_draft_params.set(_spec_str("spec_draft_params"))
    launcher.no_mmproj.set(_spec_bool("no_mmproj"))
    # --- Reasoning / Thinking (both backends) ---
    launcher.reasoning_mode.set(_spec_str("reasoning_mode"))
    launcher.reasoning_format.set(_spec_str("reasoning_format"))
    launcher.reasoning_budget.set(_spec_str("reasoning_budget"))
    launcher.reasoning_budget_message.set(_spec_str("reasoning_budget_message"))
    launcher.chat_template_kwargs.set(_spec_str("chat_template_kwargs"))
    # --- KV Unification (llama.cpp only) ---
    launcher.kv_unified_mode.set(_spec_str("kv_unified_mode"))
    launcher.cache_idle_slots_mode.set(_spec_str("cache_idle_slots_mode"))


def coerce_spec_draft_selected_gpus(raw_value):
    """Coerce ``spec_draft_selected_gpus`` to a list of ints, dropping any
    bool / non-coercible entry. Returns a new list (caller assigns).
    """
    if not isinstance(raw_value, list):
        return []
    cleaned = []
    for entry in raw_value:
        if isinstance(entry, bool):
            # bool is a subclass of int but doesn't make sense as a GPU id
            continue
        if isinstance(entry, int):
            cleaned.append(entry)
        else:
            try:
                cleaned.append(int(entry))
            except (TypeError, ValueError):
                continue
    return cleaned


def validate_spec_app_settings(app_settings):
    """Type-coerce bool/str keys and apply defaults. Mutates ``app_settings``
    in place.

    Validates MTP / Speculative Decoding entries so legacy/missing entries
    don't break later set() calls (strings stay strings; booleans stay
    booleans; everything else falls back to a default).
    """
    for k in SPEC_BOOL_KEYS:
        v = app_settings.get(k, False)
        if not isinstance(v, bool):
            app_settings[k] = (str(v).lower() in ("1", "true", "yes"))
    for k in SPEC_STR_KEYS:
        v = app_settings.get(k, "")
        if not isinstance(v, str):
            app_settings[k] = "" if v is None else str(v)
    # spec_type defaults to "none" so we never emit a flag with an unknown empty value.
    if not app_settings.get("spec_type"):
        app_settings["spec_type"] = "none"


def sync_spec_to_app_settings(launcher):
    """Mirror every spec/reasoning/kvu Tk var into ``launcher.app_settings``.

    Called from ``ConfigManager.save_configs`` so the keys round-trip
    independently of named-config save/load (matches the
    selected_mmproj_path pattern).
    """
    spec_app_keys = [
        ("spec_enabled", launcher.spec_enabled),
        ("spec_type", launcher.spec_type),
        ("spec_draft_n_max", launcher.spec_draft_n_max),
        ("spec_draft_n_min", launcher.spec_draft_n_min),
        ("spec_draft_p_min", launcher.spec_draft_p_min),
        ("spec_draft_p_split", launcher.spec_draft_p_split),
        ("spec_draft_model", launcher.spec_draft_model),
        ("spec_use_draft_model", launcher.spec_use_draft_model),
        ("spec_draft_ngl", launcher.spec_draft_ngl),
        ("spec_draft_device", launcher.spec_draft_device),
        ("spec_draft_ctk", launcher.spec_draft_ctk),
        ("spec_draft_ctv", launcher.spec_draft_ctv),
        ("spec_draft_cpu_moe", launcher.spec_draft_cpu_moe),
        ("spec_draft_n_cpu_moe", launcher.spec_draft_n_cpu_moe),
        ("spec_ngram_simple_size_n", launcher.spec_ngram_simple_size_n),
        ("spec_ngram_simple_size_m", launcher.spec_ngram_simple_size_m),
        ("spec_ngram_simple_min_hits", launcher.spec_ngram_simple_min_hits),
        ("spec_ngram_mapk_size_n", launcher.spec_ngram_mapk_size_n),
        ("spec_ngram_mapk_size_m", launcher.spec_ngram_mapk_size_m),
        ("spec_ngram_mapk_min_hits", launcher.spec_ngram_mapk_min_hits),
        ("spec_ngram_mapk4v_size_n", launcher.spec_ngram_mapk4v_size_n),
        ("spec_ngram_mapk4v_size_m", launcher.spec_ngram_mapk4v_size_m),
        ("spec_ngram_mapk4v_min_hits", launcher.spec_ngram_mapk4v_min_hits),
        ("spec_ngram_mod_n_min", launcher.spec_ngram_mod_n_min),
        ("spec_ngram_mod_n_max", launcher.spec_ngram_mod_n_max),
        ("spec_ngram_mod_n_match", launcher.spec_ngram_mod_n_match),
        ("spec_ngram_size_n", launcher.spec_ngram_size_n),
        ("spec_ngram_size_m", launcher.spec_ngram_size_m),
        ("spec_ngram_min_hits", launcher.spec_ngram_min_hits),
        ("spec_suffix_pattern_len", launcher.spec_suffix_pattern_len),
        ("spec_suffix_max_depth", launcher.spec_suffix_max_depth),
        ("spec_autotune", launcher.spec_autotune),
        ("spec_draft_params", launcher.spec_draft_params),
        ("no_mmproj", launcher.no_mmproj),
        # Reasoning / Thinking (both backends).
        ("reasoning_mode", launcher.reasoning_mode),
        ("reasoning_format", launcher.reasoning_format),
        ("reasoning_budget", launcher.reasoning_budget),
        ("reasoning_budget_message", launcher.reasoning_budget_message),
        ("chat_template_kwargs", launcher.chat_template_kwargs),
        # KV Unification (llama.cpp only).
        ("kv_unified_mode", launcher.kv_unified_mode),
        ("cache_idle_slots_mode", launcher.cache_idle_slots_mode),
    ]
    for key, var in spec_app_keys:
        try:
            launcher.app_settings[key] = var.get()
        except Exception:
            # Defensive: never let a UI sync wedge config save.
            pass


def resync_spec_tk_vars_from_app_settings(launcher):
    """Re-sync spec/reasoning/kvu Tk vars from launcher.app_settings.

    Order-of-init bug: the Tk vars on the launcher are initialized from
    ``launcher.app_settings`` BEFORE ``_load_saved_configs`` updates it
    from disk. Without this re-sync the Tk vars keep their constructor
    defaults, AND the very next ``_save_configs()`` (triggered by traces
    fired during ``env_vars_manager`` / ``ik_llama_tab`` load_from_config)
    mirrors those defaults back into app_settings, silently wiping the
    disk values. Same issue affects ``selected_mmproj_path`` and
    ``mmproj_enabled``.
    """
    def _resync_bool(key, var):
        raw = launcher.app_settings.get(key, None)
        if raw is None:
            return
        try:
            if isinstance(raw, bool):
                var.set(raw)
            else:
                var.set(str(raw).lower() in ("1", "true", "yes"))
        except Exception:
            pass

    def _resync_str(key, var, default=""):
        raw = launcher.app_settings.get(key, None)
        if raw is None:
            return
        try:
            var.set(raw if isinstance(raw, str) else (str(raw) if raw is not None else default))
        except Exception:
            pass

    # mmproj-related (pre-existing, same ordering bug).
    _resync_str("selected_mmproj_path", launcher.selected_mmproj_path)
    # MTP / Speculative Decoding.
    _resync_bool("spec_enabled", launcher.spec_enabled)
    # spec_type defaults to "none" so an empty stored value doesn't blank
    # the var; the launch.py whitelist also re-validates before emission.
    spec_type_loaded = launcher.app_settings.get("spec_type", None)
    if spec_type_loaded not in (None, ""):
        try:
            launcher.spec_type.set(str(spec_type_loaded))
        except Exception:
            pass
    for _key, _var in (
        ("spec_draft_n_max", launcher.spec_draft_n_max),
        ("spec_draft_n_min", launcher.spec_draft_n_min),
        ("spec_draft_p_min", launcher.spec_draft_p_min),
        ("spec_draft_p_split", launcher.spec_draft_p_split),
        ("spec_draft_model", launcher.spec_draft_model),
        ("spec_draft_ngl", launcher.spec_draft_ngl),
        ("spec_draft_device", launcher.spec_draft_device),
        ("spec_draft_ctk", launcher.spec_draft_ctk),
        ("spec_draft_ctv", launcher.spec_draft_ctv),
        ("spec_draft_n_cpu_moe", launcher.spec_draft_n_cpu_moe),
        ("spec_ngram_simple_size_n", launcher.spec_ngram_simple_size_n),
        ("spec_ngram_simple_size_m", launcher.spec_ngram_simple_size_m),
        ("spec_ngram_simple_min_hits", launcher.spec_ngram_simple_min_hits),
        ("spec_ngram_mapk_size_n", launcher.spec_ngram_mapk_size_n),
        ("spec_ngram_mapk_size_m", launcher.spec_ngram_mapk_size_m),
        ("spec_ngram_mapk_min_hits", launcher.spec_ngram_mapk_min_hits),
        ("spec_ngram_mapk4v_size_n", launcher.spec_ngram_mapk4v_size_n),
        ("spec_ngram_mapk4v_size_m", launcher.spec_ngram_mapk4v_size_m),
        ("spec_ngram_mapk4v_min_hits", launcher.spec_ngram_mapk4v_min_hits),
        ("spec_ngram_mod_n_min", launcher.spec_ngram_mod_n_min),
        ("spec_ngram_mod_n_max", launcher.spec_ngram_mod_n_max),
        ("spec_ngram_mod_n_match", launcher.spec_ngram_mod_n_match),
        ("spec_ngram_size_n", launcher.spec_ngram_size_n),
        ("spec_ngram_size_m", launcher.spec_ngram_size_m),
        ("spec_ngram_min_hits", launcher.spec_ngram_min_hits),
        ("spec_suffix_pattern_len", launcher.spec_suffix_pattern_len),
        ("spec_suffix_max_depth", launcher.spec_suffix_max_depth),
        ("spec_draft_params", launcher.spec_draft_params),
        # Reasoning / Thinking.
        ("reasoning_mode", launcher.reasoning_mode),
        ("reasoning_format", launcher.reasoning_format),
        ("reasoning_budget", launcher.reasoning_budget),
        ("reasoning_budget_message", launcher.reasoning_budget_message),
        ("chat_template_kwargs", launcher.chat_template_kwargs),
        # KV Unification.
        ("kv_unified_mode", launcher.kv_unified_mode),
        ("cache_idle_slots_mode", launcher.cache_idle_slots_mode),
    ):
        _resync_str(_key, _var)
    for _key, _var in (
        ("spec_draft_cpu_moe", launcher.spec_draft_cpu_moe),
        ("spec_autotune", launcher.spec_autotune),
        ("no_mmproj", launcher.no_mmproj),
        ("spec_use_draft_model", launcher.spec_use_draft_model),
    ):
        _resync_bool(_key, _var)
