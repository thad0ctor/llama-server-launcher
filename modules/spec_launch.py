#!/usr/bin/env python3
"""
Spec / MTP / Reasoning / KV-Unification command emission helpers.

This module owns the per-backend emission logic for the MTP / Speculative
Decoding feature (``--spec-*`` and ``--draft-*`` flag families), plus the
sibling chat-template / reasoning / KV-unification emission blocks and the
``--no-mmproj`` toggle, all of which live alongside the spec feature in the
UI.

Extracted from ``modules/launch.py`` to keep ``LaunchManager.build_cmd``
focused on per-arg add-arg calls. The helpers read from the launcher's
existing Tk vars (``launcher.spec_enabled``, ``launcher.spec_type``, etc.)
and append to the in-progress ``cmd`` list; they do not return new lists.

The per-backend whitelists below are the single source of truth for valid
``--spec-type`` values; they mirror the UI's per-backend dropdown choices
in ``modules.spec_tab.SpecTab``.
"""

import sys
from pathlib import Path


# Per-backend allowed values for `--spec-type`. Used to validate spec_type
# coming from saved/imported configs before emission; an unknown string can
# crash the server at startup, so reject it with a stderr warning instead.
# Sets mirror the UI's per-backend dropdown choices.
_ALLOWED_SPEC_TYPES_LLAMA_CPP = frozenset({
    "none",
    "draft-simple", "draft-eagle3", "draft-mtp",
    "ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod", "ngram-cache",
})
_ALLOWED_SPEC_TYPES_IK_LLAMA = frozenset({
    "none",
    "mtp",
    "ngram-cache", "ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod",
    "suffix",
})

# Per-backend subsets of spec_types that use a separate draft model + the
# associated draft-tuning/offload knobs. Non-draft-capable spec_types
# (ngram-*, suffix, etc.) reuse the main model and don't read these flags;
# emitting them anyway produces nonsensical CLI combos from saved configs
# where the user toggled spec_type without clearing prior draft fields.
_DRAFT_CAPABLE_SPEC_TYPES_LLAMA_CPP = frozenset({"draft-simple", "draft-eagle3", "draft-mtp"})
_DRAFT_CAPABLE_SPEC_TYPES_IK_LLAMA = frozenset({"mtp"})


def emit_spec_args(launcher, backend, cmd):
    """Append all ``--spec-*`` / ``--draft-*`` family flags to ``cmd`` based on
    ``launcher.spec_*`` Tk vars and the active backend.

    Master toggle pattern: only emit any --spec-*/--draft-* flag when
    spec_enabled is True AND a non-"none" spec_type is selected. Backend
    surfaces diverge significantly between mainline llama.cpp and ik_llama;
    the two branches below mirror the UI's per-backend gating.
    """
    try:
        spec_enabled_var = getattr(launcher, "spec_enabled", None)
        if spec_enabled_var is not None and spec_enabled_var.get():
            spec_type_var = getattr(launcher, "spec_type", None)
            spec_type = (spec_type_var.get().strip() if spec_type_var is not None else "")
            # Reject unknown spec_type values before forwarding them — a
            # stale/hand-edited config can otherwise emit a garbage value
            # and crash the server at startup. Per-backend whitelists
            # match the UI dropdown choices.
            if spec_type and spec_type != "none":
                allowed = (_ALLOWED_SPEC_TYPES_IK_LLAMA if backend == "ik_llama"
                           else _ALLOWED_SPEC_TYPES_LLAMA_CPP)
                if spec_type not in allowed:
                    print(
                        f"WARNING: spec_type {spec_type!r} is not valid for backend "
                        f"{backend!r}; skipping --spec-type emission.",
                        file=sys.stderr,
                    )
                    spec_type = "none"
            if spec_type and spec_type != "none":
                if backend == "ik_llama":
                    cmd.extend(["--spec-type", spec_type])
                    # Only draft-capable spec_types (ik_llama: "mtp")
                    # use a separate draft model + the matching
                    # tuning/offload knobs. For ngram-*/suffix, the
                    # draft fields are stale state from a prior session
                    # and must NOT be forwarded — the UI hides them in
                    # those modes, so emission would silently violate
                    # the grayed-field contract.
                    is_draft_capable = spec_type in _DRAFT_CAPABLE_SPEC_TYPES_IK_LLAMA
                    if is_draft_capable:
                        # ik_llama draft tuning flags use --draft-max/--draft-min/--draft-p-min.
                        for var_name, flag in [
                            ("spec_draft_n_max", "--draft-max"),
                            ("spec_draft_n_min", "--draft-min"),
                            ("spec_draft_p_min", "--draft-p-min"),
                        ]:
                            var = getattr(launcher, var_name, None)
                            if var is not None:
                                v = var.get().strip()
                                if v:
                                    cmd.extend([flag, v])
                        # Draft model file (rare for ik_llama, but supported via --model-draft).
                        # Validate the path before emitting — a saved config can hold a
                        # stale path to a moved/deleted draft GGUF; mirror the main -m
                        # behaviour of resolving + skipping with a stderr warning.
                        mp_var = getattr(launcher, "spec_draft_model", None)
                        if mp_var is not None:
                            mp = mp_var.get().strip()
                            if mp:
                                if Path(mp).is_file():
                                    cmd.extend(["--model-draft", str(Path(mp).resolve())])
                                else:
                                    print(
                                        f"WARNING: draft model path '{mp}' is not a file; skipping --model-draft emission.",
                                        file=sys.stderr,
                                    )
                        # ik_llama uses the same short-form draft offload flags.
                        for var_name, flag in [
                            ("spec_draft_ngl", "-ngld"),
                            ("spec_draft_device", "-devd"),
                            ("spec_draft_ctk", "-ctkd"),
                            ("spec_draft_ctv", "-ctvd"),
                        ]:
                            var = getattr(launcher, var_name, None)
                            if var is not None:
                                v = var.get().strip()
                                if v:
                                    cmd.extend([flag, v])
                    # ngram: ik_llama has a single shared --spec-ngram-* set.
                    if spec_type.startswith("ngram-"):
                        for var_name, flag in [
                            ("spec_ngram_size_n", "--spec-ngram-size-n"),
                            ("spec_ngram_size_m", "--spec-ngram-size-m"),
                            ("spec_ngram_min_hits", "--spec-ngram-min-hits"),
                        ]:
                            var = getattr(launcher, var_name, None)
                            if var is not None:
                                v = var.get().strip()
                                if v:
                                    cmd.extend([flag, v])
                    # suffix
                    if spec_type == "suffix":
                        for var_name, flag in [
                            ("spec_suffix_pattern_len", "--suffix-pattern-len"),
                            ("spec_suffix_max_depth", "--suffix-max-depth"),
                        ]:
                            var = getattr(launcher, var_name, None)
                            if var is not None:
                                v = var.get().strip()
                                if v:
                                    cmd.extend([flag, v])
                    # ik_llama extras.
                    autotune_var = getattr(launcher, "spec_autotune", None)
                    if autotune_var is not None and autotune_var.get():
                        cmd.append("--spec-autotune")
                    dp_var = getattr(launcher, "spec_draft_params", None)
                    if dp_var is not None:
                        dp = dp_var.get().strip()
                        if dp:
                            cmd.extend(["-draft", dp])
                    # Warn (don't crash) if the user set llama.cpp-only knobs while ik_llama is active.
                    for var_name, label in [
                        ("spec_draft_p_split", "--spec-draft-p-split"),
                        ("spec_draft_cpu_moe", "--spec-draft-cpu-moe"),
                        ("spec_draft_n_cpu_moe", "--spec-draft-n-cpu-moe"),
                    ]:
                        var = getattr(launcher, var_name, None)
                        if var is None:
                            continue
                        try:
                            raw = var.get()
                        except Exception:
                            raw = None
                        if isinstance(raw, bool):
                            if raw:
                                print(f"WARNING: {label} is llama.cpp-only; ignoring for ik_llama backend.", file=sys.stderr)
                        elif isinstance(raw, str) and raw.strip():
                            print(f"WARNING: {label} is llama.cpp-only; ignoring for ik_llama backend.", file=sys.stderr)
                else:
                    # llama.cpp (mainline) branch.
                    cmd.extend(["--spec-type", spec_type])
                    # Only draft-capable spec_types (llama.cpp:
                    # draft-simple/draft-eagle3/draft-mtp) read a
                    # separate draft model + the matching tuning/offload
                    # knobs. ngram-*/cache types reuse the main model;
                    # forwarding stale draft fields from a prior
                    # session would silently break the UI's
                    # grayed-field contract.
                    is_draft_capable = spec_type in _DRAFT_CAPABLE_SPEC_TYPES_LLAMA_CPP
                    if is_draft_capable:
                        for var_name, flag in [
                            ("spec_draft_n_max", "--spec-draft-n-max"),
                            ("spec_draft_n_min", "--spec-draft-n-min"),
                            ("spec_draft_p_min", "--spec-draft-p-min"),
                            ("spec_draft_p_split", "--spec-draft-p-split"),
                        ]:
                            var = getattr(launcher, var_name, None)
                            if var is not None:
                                v = var.get().strip()
                                if v:
                                    cmd.extend([flag, v])
                        # Validate the draft model path before emitting —
                        # a saved config can hold a stale path to a
                        # moved/deleted draft GGUF; mirror the main -m
                        # behaviour of resolving + skipping with a
                        # stderr warning.
                        mp_var = getattr(launcher, "spec_draft_model", None)
                        if mp_var is not None:
                            mp = mp_var.get().strip()
                            if mp:
                                if Path(mp).is_file():
                                    cmd.extend(["--spec-draft-model", str(Path(mp).resolve())])
                                else:
                                    print(
                                        f"WARNING: draft model path '{mp}' is not a file; skipping --spec-draft-model emission.",
                                        file=sys.stderr,
                                    )
                        for var_name, flag in [
                            ("spec_draft_ngl", "--spec-draft-ngl"),
                            ("spec_draft_device", "--spec-draft-device"),
                            ("spec_draft_ctk", "--spec-draft-type-k"),
                            ("spec_draft_ctv", "--spec-draft-type-v"),
                        ]:
                            var = getattr(launcher, var_name, None)
                            if var is not None:
                                v = var.get().strip()
                                if v:
                                    cmd.extend([flag, v])
                        cpu_moe_var = getattr(launcher, "spec_draft_cpu_moe", None)
                        if cpu_moe_var is not None and cpu_moe_var.get():
                            cmd.append("--spec-draft-cpu-moe")
                        ncm_var = getattr(launcher, "spec_draft_n_cpu_moe", None)
                        if ncm_var is not None:
                            ncm = ncm_var.get().strip()
                            if ncm:
                                cmd.extend(["--spec-draft-n-cpu-moe", ncm])
                    # llama.cpp has per-ngram-variant size knobs.
                    if spec_type == "ngram-simple":
                        for var_name, flag in [
                            ("spec_ngram_simple_size_n", "--spec-ngram-simple-size-n"),
                            ("spec_ngram_simple_size_m", "--spec-ngram-simple-size-m"),
                            ("spec_ngram_simple_min_hits", "--spec-ngram-simple-min-hits"),
                        ]:
                            var = getattr(launcher, var_name, None)
                            if var is not None:
                                v = var.get().strip()
                                if v:
                                    cmd.extend([flag, v])
                    elif spec_type == "ngram-map-k":
                        for var_name, flag in [
                            ("spec_ngram_mapk_size_n", "--spec-ngram-map-k-size-n"),
                            ("spec_ngram_mapk_size_m", "--spec-ngram-map-k-size-m"),
                            ("spec_ngram_mapk_min_hits", "--spec-ngram-map-k-min-hits"),
                        ]:
                            var = getattr(launcher, var_name, None)
                            if var is not None:
                                v = var.get().strip()
                                if v:
                                    cmd.extend([flag, v])
                    elif spec_type == "ngram-map-k4v":
                        for var_name, flag in [
                            ("spec_ngram_mapk4v_size_n", "--spec-ngram-map-k4v-size-n"),
                            ("spec_ngram_mapk4v_size_m", "--spec-ngram-map-k4v-size-m"),
                            ("spec_ngram_mapk4v_min_hits", "--spec-ngram-map-k4v-min-hits"),
                        ]:
                            var = getattr(launcher, var_name, None)
                            if var is not None:
                                v = var.get().strip()
                                if v:
                                    cmd.extend([flag, v])
                    elif spec_type == "ngram-mod":
                        for var_name, flag in [
                            ("spec_ngram_mod_n_min", "--spec-ngram-mod-n-min"),
                            ("spec_ngram_mod_n_max", "--spec-ngram-mod-n-max"),
                            ("spec_ngram_mod_n_match", "--spec-ngram-mod-n-match"),
                        ]:
                            var = getattr(launcher, var_name, None)
                            if var is not None:
                                v = var.get().strip()
                                if v:
                                    cmd.extend([flag, v])
                    # ngram-cache has no extra knobs.
                    # Warn (don't crash) if ik_llama-only knobs are set while llama.cpp is active.
                    for var_name, label in [
                        ("spec_autotune", "--spec-autotune"),
                        ("spec_draft_params", "-draft"),
                        ("spec_suffix_pattern_len", "--suffix-pattern-len"),
                        ("spec_suffix_max_depth", "--suffix-max-depth"),
                    ]:
                        var = getattr(launcher, var_name, None)
                        if var is None:
                            continue
                        try:
                            raw = var.get()
                        except Exception:
                            raw = None
                        if isinstance(raw, bool):
                            if raw:
                                print(f"WARNING: {label} is ik_llama-only; ignoring for llama.cpp backend.", file=sys.stderr)
                        elif isinstance(raw, str) and raw.strip():
                            print(f"WARNING: {label} is ik_llama-only; ignoring for llama.cpp backend.", file=sys.stderr)
    except Exception as exc:
        # Never let a UI mis-state crash the launch flow - log and continue.
        print(f"WARNING: speculative-decoding block raised: {exc}", file=sys.stderr)


def emit_reasoning_args(launcher, cmd):
    """Append ``--reasoning`` / ``--reasoning-*`` / ``--chat-template-kwargs``
    flags to ``cmd``.

    Independent of spec_enabled — emit unconditionally based on per-var
    values. All five flags are accepted by mainline llama.cpp and ik_llama.
    """
    try:
        rm_var = getattr(launcher, "reasoning_mode", None)
        if rm_var is not None:
            rm = rm_var.get().strip()
            if rm and rm in ("on", "off", "auto"):
                cmd.extend(["--reasoning", rm])
        rf_var = getattr(launcher, "reasoning_format", None)
        if rf_var is not None:
            rf = rf_var.get().strip()
            if rf:
                cmd.extend(["--reasoning-format", rf])
        rb_var = getattr(launcher, "reasoning_budget", None)
        if rb_var is not None:
            rb = rb_var.get().strip()
            if rb:
                # Defend against a stale non-integer value persisted from a
                # pre-validation config. The Entry validator blocks new
                # bad input; this catches anything that slipped through.
                try:
                    int(rb)
                    cmd.extend(["--reasoning-budget", rb])
                except ValueError:
                    print(f"WARNING: --reasoning-budget value {rb!r} is not an integer; skipping.", file=sys.stderr)
        rbm_var = getattr(launcher, "reasoning_budget_message", None)
        if rbm_var is not None:
            rbm = rbm_var.get().strip()
            if rbm:
                cmd.extend(["--reasoning-budget-message", rbm])
        ctk_var = getattr(launcher, "chat_template_kwargs", None)
        if ctk_var is not None:
            ctk = ctk_var.get().strip()
            if ctk:
                cmd.extend(["--chat-template-kwargs", ctk])
    except Exception as exc:
        print(f"WARNING: reasoning/chat-template emission raised: {exc}", file=sys.stderr)


def emit_kv_unify_args(launcher, backend, cmd):
    """Append ``--kv-unified`` / ``--no-kv-unified`` / ``--cache-idle-slots`` /
    ``--no-cache-idle-slots`` to ``cmd``, gated by backend and dependency.

    ik_llama does NOT support these flags (verified absent from common.cpp).
    Warn and skip when ik_llama is active; emit explicit on/off otherwise.
    """
    try:
        kvu_var = getattr(launcher, "kv_unified_mode", None)
        cis_var = getattr(launcher, "cache_idle_slots_mode", None)
        kvu = kvu_var.get().strip() if kvu_var is not None else ""
        cis = cis_var.get().strip() if cis_var is not None else ""
        if backend == "ik_llama":
            if kvu in ("on", "off") or cis in ("on", "off"):
                print("WARNING: --kv-unified / --cache-idle-slots are llama.cpp-only; ignoring for ik_llama backend.", file=sys.stderr)
        else:
            # --cache-idle-slots requires --kv-unified to be on (the
            # server itself warns and disables otherwise). Enforce the
            # dependency at emission as a last line of defense against
            # stale/imported configs that survived the UI gating.
            if kvu == "on":
                cmd.append("--kv-unified")
                if cis == "on":
                    cmd.append("--cache-idle-slots")
                elif cis == "off":
                    cmd.append("--no-cache-idle-slots")
            elif kvu == "off":
                cmd.append("--no-kv-unified")
                if cis in ("on", "off"):
                    print(
                        "WARNING: --cache-idle-slots requires --kv-unified=on; "
                        "skipping (kv_unified_mode is 'off').",
                        file=sys.stderr,
                    )
            elif cis in ("on", "off"):
                print(
                    "WARNING: --cache-idle-slots requires --kv-unified=on; "
                    "skipping (kv_unified_mode is unset).",
                    file=sys.stderr,
                )
    except Exception as exc:
        print(f"WARNING: kv-unified/cache-idle-slots emission raised: {exc}", file=sys.stderr)


def emit_no_mmproj_arg(launcher, backend, cmd):
    """Append ``--no-mmproj`` if requested (llama.cpp only).

    Independent of spec_enabled: --no-mmproj is useful on its own when a
    base GGUF embeds a vision projector that the user wants suppressed.
    """
    try:
        no_mmproj_var = getattr(launcher, "no_mmproj", None)
        if no_mmproj_var is not None and no_mmproj_var.get():
            if backend == "ik_llama":
                print("WARNING: --no-mmproj is llama.cpp-only; ignoring for ik_llama backend.", file=sys.stderr)
            else:
                cmd.append("--no-mmproj")
    except Exception as exc:
        print(f"WARNING: --no-mmproj emission raised: {exc}", file=sys.stderr)


def resolve_effective_parallel(launcher, backend):
    """Return the effective ``--parallel`` value, applying the MTP 'force 1'
    override only when MTP is the *backend-valid* active spec_type. Prints
    stderr warning on override.

    MTP enforces single-slot operation (-np 1) — OVERRIDES whatever the
    user (or another UI surface like the Advanced tab) set, since MTP
    upstream simply does not work with multi-slot. Decide the effective
    parallel value here, BEFORE the add_arg call, so the wrong value can
    never make it into argv. The MTP/Spec tab also auto-sets parallel
    to "1" when MTP is selected, but this guard is the authoritative
    last line of defense regardless of where parallel was set from.

    Backend validation: the launcher preserves the stored spec_type even
    when the user flips backends, so a value like ``draft-mtp`` may sit
    in ``self.spec_type`` while ``ik_llama`` is active (and vice versa).
    In that case ``emit_spec_args`` correctly skips ``--spec-type`` for
    that backend, so MTP is NOT actually active and we must NOT force
    parallel to 1. Only the backend-correct MTP variant counts:
        - llama.cpp + spec_type=='draft-mtp' → MTP active
        - ik_llama  + spec_type=='mtp'       → MTP active
        - anything else                       → not MTP active
    """
    parallel_val = launcher.parallel.get()
    try:
        spec_enabled_var = getattr(launcher, "spec_enabled", None)
        spec_type_var = getattr(launcher, "spec_type", None)
        spec_type = (spec_type_var.get() or "").strip() if spec_type_var is not None else ""
        if backend == "ik_llama":
            mtp_type_for_backend = "mtp"
        else:
            mtp_type_for_backend = "draft-mtp"
        mtp_active = (
            spec_enabled_var is not None
            and spec_enabled_var.get()
            and spec_type == mtp_type_for_backend
        )
    except Exception:
        mtp_active = False
    if mtp_active and (parallel_val or "").strip() != "1":
        print(
            f"WARNING: MTP requires --parallel 1; overriding '{parallel_val}' -> '1'.",
            file=sys.stderr,
        )
        parallel_val = "1"
    return parallel_val
