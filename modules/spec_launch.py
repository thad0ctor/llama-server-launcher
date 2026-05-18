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

# Spec types that load a SEPARATE draft model on its own GPU subset.
# MTP (both backends) is excluded because the MTP head is embedded in
# the main GGUF and rides along with the main model's GPU distribution
# — it has no separate device requirement. A stale draft-GPU selection
# from a prior draft-simple session must NOT cause a CUDA_VISIBLE_DEVICES
# union or a --spec-draft-device emission when the user is now on
# draft-mtp. Only the explicit draft-model variants need separate GPU
# handling.
_SEPARATE_DRAFT_GPU_SPEC_TYPES_LLAMA_CPP = frozenset({"draft-simple", "draft-eagle3"})
_SEPARATE_DRAFT_GPU_SPEC_TYPES_IK_LLAMA = frozenset()  # mtp uses main GPUs


def _uses_separate_draft_gpus(spec_type, backend, use_draft_model_opt_in=False):
    """Return True iff ``spec_type`` is a variant that loads a SEPARATE
    draft model and therefore wants its own GPU subset / device flag.

    Strictly narrower than ``_is_draft_capable_for_backend``: normally
    excludes draft-mtp / mtp because their MTP head is embedded in the
    main GGUF and shares the main GPU distribution. The exception is
    ik_llama + mtp when the user has opted into the legacy separate
    ``--model-draft`` path (``spec_use_draft_model`` checkbox): in that
    case a real draft GGUF is loaded and needs its own GPU subset, so
    union the draft GPUs into ``CUDA_VISIBLE_DEVICES`` and emit ``-devd``.
    Used to gate ``get_effective_visible_gpu_indices`` and
    ``_resolve_draft_device_value``.
    """
    if not spec_type or spec_type == "none":
        return False
    if backend == "ik_llama":
        if spec_type == "mtp" and use_draft_model_opt_in:
            return True
        return spec_type in _SEPARATE_DRAFT_GPU_SPEC_TYPES_IK_LLAMA
    return spec_type in _SEPARATE_DRAFT_GPU_SPEC_TYPES_LLAMA_CPP


def _use_draft_model_opt_in(launcher):
    """Read the ``spec_use_draft_model`` Tk var safely. Returns False if
    the var is missing (defensive) or raises."""
    try:
        udm_var = getattr(launcher, "spec_use_draft_model", None)
        return bool(udm_var.get()) if udm_var is not None else False
    except Exception:
        return False


def _is_draft_capable_for_backend(spec_type, backend):
    """Return True iff ``spec_type`` is a draft-capable variant for ``backend``.

    Mirrors the per-branch ``is_draft_capable`` checks inside ``emit_spec_args``
    so the GPU-union helper below can apply the same gating without re-deriving
    the rule (and silently drifting from emission).
    """
    if not spec_type or spec_type == "none":
        return False
    if backend == "ik_llama":
        return spec_type in _DRAFT_CAPABLE_SPEC_TYPES_IK_LLAMA
    return spec_type in _DRAFT_CAPABLE_SPEC_TYPES_LLAMA_CPP


def get_effective_visible_gpu_indices(launcher):
    """Return the ordered list of physical GPU indices that should appear in
    ``CUDA_VISIBLE_DEVICES`` — the union of the user's main GPU selection
    and the draft GPU selection (only when spec is enabled and the active
    spec_type is draft-capable for the active backend).

    Ordering contract:
        - Main GPUs first, in the user-specified order from
          ``get_ordered_selected_gpus()`` (which honours drag-reorder).
        - Then draft-only GPUs (those NOT already in main) appended in
          numeric order — picked deterministically so the post-filter
          CUDA<i> remap is reproducible across launches.

    This is the single source of truth for both
    ``LaunchManager._resolve_cuda_visible_devices_action`` (which exports
    the env var) and ``_resolve_draft_device_value`` (which remaps draft
    indices to post-filter CUDA<i>). Keeping them in lock-step prevents the
    failure mode where the binary's reindexed device list and the
    launcher's --spec-draft-device value disagree.

    Gating rules:
        - Spec disabled, or spec_type non-draft-capable for the active
          backend → return main_ordered only (no union).
        - Empty draft_indices → main_ordered only.
        - All other cases → ordered union as described above.

    Returns a list of ints (possibly empty). Never raises — defensive
    ``except Exception`` falls back to main_ordered so launch flow can't
    crash on a UI mis-state.
    """
    try:
        main_ordered = list(launcher.get_ordered_selected_gpus())
    except Exception:
        main_ordered = []
    # Resolve gating up front so we can early-return main-only.
    spec_enabled = False
    try:
        spec_enabled_var = getattr(launcher, "spec_enabled", None)
        spec_enabled = bool(spec_enabled_var is not None and spec_enabled_var.get())
    except Exception:
        spec_enabled = False
    if not spec_enabled:
        return main_ordered
    try:
        spec_type_var = getattr(launcher, "spec_type", None)
        spec_type = (spec_type_var.get() or "").strip() if spec_type_var is not None else ""
    except Exception:
        spec_type = ""
    try:
        backend = launcher.backend_selection.get()
    except Exception:
        backend = ""
    # Only the SEPARATE-draft-model variants need a draft GPU subset
    # unioned into CUDA_VISIBLE_DEVICES. MTP variants share the main GGUF
    # and run on whatever GPUs the main model uses, so a stale draft-GPU
    # selection (e.g. from a prior draft-simple session) must NOT widen
    # the visible-device set when the user is now on draft-mtp / mtp —
    # UNLESS the user has explicitly opted into a separate --model-draft
    # for ik_llama mtp via the "Use a separate draft model" checkbox.
    if not _uses_separate_draft_gpus(spec_type, backend, _use_draft_model_opt_in(launcher)):
        return main_ordered
    try:
        draft_indices = list(
            launcher.app_settings.get("spec_draft_selected_gpus", []) or []
        )
    except Exception:
        draft_indices = []
    if not draft_indices:
        return main_ordered
    # Don't auto-create a CUDA_VISIBLE_DEVICES filter when the user hasn't
    # selected any main GPUs explicitly. Empty main means "no restriction"
    # (the binary's default device discovery is in charge) — adding only
    # draft GPUs would silently restrict the device set, which is the
    # opposite of what the user asked for. Pass through unchanged so the
    # caller's no-filter branch handles it.
    if not main_ordered:
        return main_ordered
    main_set = set(main_ordered)
    extras = sorted({d for d in draft_indices if d not in main_set})
    return main_ordered + extras


def _resolve_main_device_value(launcher):
    """Return the value for the MAIN model's ``--device`` flag when draft
    GPUs were unioned into ``CUDA_VISIBLE_DEVICES`` — otherwise ``""``.

    Background: when the union helper adds draft-only GPUs to
    ``CUDA_VISIBLE_DEVICES`` so the binary can see them, the binary's
    default device discovery will then spread the MAIN model's
    ``--n-gpu-layers`` across ALL visible GPUs (including the
    draft-target ones). That fills the draft-target GPUs with main
    layers and OOMs the draft model on those GPUs. The fix: when the
    union added extras, also restrict the main model to just the user's
    main-selected GPUs via ``--device CUDA0,CUDA1,...``, mapped through
    the same post-filter ordering ``--spec-draft-device`` already uses.

    Skip conditions (return ""):
        - Manual GPU mode: synthetic indices have no relation to real
          CUDA, and the launcher already unsets ``CUDA_VISIBLE_DEVICES``;
          emitting ``--device CUDA<i>`` would refer to wrong/nonexistent
          devices.
        - No main selection: nothing to constrain.
        - No extras unioned in (draft empty / draft is a subset of main /
          spec not draft-capable / spec disabled): the effective visible
          set equals the main set, the binary's view already matches the
          user intent, and ``--device`` is redundant.
        - All-GPUs selected (effective == detected count, no filter in
          effect): launcher passes indices through unchanged, no
          re-mapping needed.

    Emission case (non-empty return):
        - Main=[1,7], draft=[2,5] → effective=[1,7,2,5] →
          ``--device CUDA0,CUDA1`` (positions of 1 and 7 in the union).

    Returns a comma-joined ``CUDA<i>`` string, or ``""`` when emission
    should be skipped. Never raises.
    """
    try:
        if getattr(launcher, "gpu_info", {}).get("manual_mode", False):
            return ""
    except Exception:
        return ""
    try:
        main_ordered = list(launcher.get_ordered_selected_gpus())
    except Exception:
        return ""
    if not main_ordered:
        return ""
    effective_ordered = get_effective_visible_gpu_indices(launcher)
    # Only emit when the union actually grew the visible set. If
    # effective == main (same length, same members in main order), the
    # binary already only sees the user's main selection and --device
    # would be a redundant restriction.
    if len(effective_ordered) <= len(main_ordered):
        return ""
    # Defensive: ensure every main GPU is present in the effective list
    # (the union helper guarantees this — main comes first verbatim — but
    # guard against future helper changes).
    parts = []
    for g in main_ordered:
        try:
            pos = effective_ordered.index(g)
        except ValueError:
            # Should not happen; if it does, skip rather than emit a
            # nonsensical index.
            continue
        parts.append(f"CUDA{pos}")
    return ",".join(parts)


def emit_main_device_arg(launcher, backend, cmd):
    """Append ``--device CUDA<i>,...`` for the MAIN model when needed.

    Pair to ``_resolve_draft_device_value``: when draft GPUs are unioned
    into ``CUDA_VISIBLE_DEVICES`` (so the binary can SEE them), the main
    model must be told to use ONLY the user's main-selected GPUs —
    otherwise the binary spreads main layers across every visible GPU
    and OOMs the draft model on the draft targets.

    Skipped when ``--tensor-split`` is set (tensor-split takes precedence
    and already constrains the main model's distribution). Skipped in
    manual GPU mode (synthetic indices), when no main GPUs are selected,
    and when the union didn't add any extras (no remap needed).

    Both llama.cpp mainline (``-dev``/``--device``) and ik_llama
    (``-dev``/``--device``) accept the same long-form spelling, so a
    single emission path covers both backends.
    """
    try:
        # Tensor-split takes precedence over per-device restrictions —
        # don't double-constrain and don't risk a backend rejecting both
        # together. The advisory in build_cmd already tells the user
        # tensor-split is in charge.
        try:
            ts_val = launcher.tensor_split.get().strip()
        except Exception:
            ts_val = ""
        if ts_val:
            return
        dev_val = _resolve_main_device_value(launcher)
        if dev_val:
            cmd.extend(["--device", dev_val])
    except Exception as exc:
        print(f"WARNING: --device (main) emission raised: {exc}", file=sys.stderr)


def _resolve_draft_device_value(launcher):
    """Return the correctly-remapped value for ``--spec-draft-device``
    given the user's draft GPU selection and the effective visible-GPU set.

    The launch script applies
    ``CUDA_VISIBLE_DEVICES=<effective_visible_indices>`` to restrict the
    GPUs the server can see — and that filter REINDEXES the devices the
    binary then knows about. The "effective visible" set is the ordered
    UNION of the user's main GPU selection and the draft GPU selection
    (see ``get_effective_visible_gpu_indices``); this guarantees that
    every draft GPU is visible to the binary, while still honouring the
    user-specified main ordering for tensor-split.

    Mapping rules (all post-union):
        - No effective subset selected (CUDA_VISIBLE_DEVICES not set):
          launcher index N == binary CUDA<N>. Pass through.
        - Effective subset = ordered list (e.g. ``[1, 2, 5, 7]``):
          binary CUDA<i> corresponds to ``effective[i]``. Each draft
          launcher index ``d`` is remapped to ``CUDA<effective.index(d)>``.
        - Draft selection includes an index that — astonishingly —
          is NOT in the effective set (shouldn't happen since the
          helper unions draft into effective, but defended for safety):
          warn and skip the offending index.

    Returns the comma-joined CUDA string, or "" when the result is
    empty (no flag should be emitted). Falls back to the launcher's
    free-text ``spec_draft_device`` value if ``spec_draft_selected_gpus``
    is empty (allowing power users to type a raw override).
    """
    # MTP variants (draft-mtp on llama.cpp, mtp on ik_llama) normally DON'T
    # use a separate draft model — the MTP head is embedded in the main
    # GGUF and rides along with the main GPU distribution. So any stored
    # draft-GPU selection (likely persisted from a prior draft-simple
    # session) must be ignored. The ik_llama+mtp exception: when the user
    # has opted into a separate --model-draft via the "Use a separate
    # draft model" checkbox, the draft GPUs are real and -devd must be
    # emitted normally (with the same CUDA_VISIBLE_DEVICES remap).
    try:
        spec_type = (launcher.spec_type.get() or "").strip()
        backend = launcher.backend_selection.get()
    except Exception:
        spec_type, backend = "", ""
    if not _uses_separate_draft_gpus(spec_type, backend, _use_draft_model_opt_in(launcher)):
        return ""

    draft_indices = list(launcher.app_settings.get("spec_draft_selected_gpus", []) or [])
    if not draft_indices:
        # No checkbox selection → fall back to the free-text override.
        try:
            return launcher.spec_draft_device.get().strip()
        except Exception:
            return ""
    # Effective visible GPU list (main ∪ draft). Single source of truth
    # shared with LaunchManager._resolve_cuda_visible_devices_action so
    # the env var the script exports and the indices emitted here can't
    # disagree.
    effective_ordered = get_effective_visible_gpu_indices(launcher)
    detected_count = 0
    try:
        gpu_info = getattr(launcher, "gpu_info", {})
        if isinstance(gpu_info, dict):
            detected_count = int(gpu_info.get("device_count", 0) or 0)
    except Exception:
        detected_count = 0
    # If no filter is in effect (no selection at all, or the user selected
    # every detected GPU), launcher indices pass through unchanged.
    no_filter = (not effective_ordered) or (
        detected_count > 0 and len(effective_ordered) == detected_count
    )
    parts = []
    if no_filter:
        for d in draft_indices:
            parts.append(f"CUDA{d}")
    else:
        effective_set = set(effective_ordered)
        for d in draft_indices:
            if d not in effective_set:
                # Defensive: under Option C this should never happen because
                # get_effective_visible_gpu_indices unions draft in. But
                # if a future change opts out of the union (e.g. ngram
                # spec_type stops being draft-capable), the warning is
                # the user-facing signal that the index was dropped.
                print(
                    f"WARNING: draft GPU index {d} is not in the effective visible "
                    f"GPU set {sorted(effective_set)}; --spec-draft-device entry "
                    f"for that GPU is skipped (would be filtered out by "
                    f"CUDA_VISIBLE_DEVICES).",
                    file=sys.stderr,
                )
                continue
            try:
                pos = effective_ordered.index(d)
            except ValueError:
                continue
            parts.append(f"CUDA{pos}")
    return ",".join(parts)


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
                        # These tune draft generation for both the embedded MTP head
                        # and a separate draft model, so emit regardless of the
                        # ``spec_use_draft_model`` opt-in.
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
                        # ik_llama+mtp opt-in for a SEPARATE draft model: when the
                        # user hasn't checked "Use a separate draft model", suppress
                        # --model-draft AND the per-draft offload flags. Embedded
                        # MTP head rides with the main model's GPU distribution.
                        udm_var = getattr(launcher, "spec_use_draft_model", None)
                        use_separate_draft = bool(udm_var.get()) if udm_var is not None else False
                        if use_separate_draft:
                            # Validate the path before emitting — a saved config can
                            # hold a stale path to a moved/deleted draft GGUF; mirror
                            # the main -m behaviour of resolving + skipping with a
                            # stderr warning.
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
                            # ``spec_draft_device`` is resolved through the
                            # CUDA_VISIBLE_DEVICES remap helper rather than read
                            # verbatim so checkbox-driven indices stay valid
                            # post-filter (e.g. CUDA4 → CUDA1 when main selection
                            # reindexes the device list).
                            for var_name, flag in [
                                ("spec_draft_ngl", "-ngld"),
                                ("spec_draft_ctk", "-ctkd"),
                                ("spec_draft_ctv", "-ctvd"),
                            ]:
                                var = getattr(launcher, var_name, None)
                                if var is not None:
                                    v = var.get().strip()
                                    if v:
                                        cmd.extend([flag, v])
                            devd_val = _resolve_draft_device_value(launcher)
                            if devd_val:
                                cmd.extend(["-devd", devd_val])
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
                        # llama.cpp mainline's ``--spec-type draft-mtp`` uses
                        # an MTP head embedded INSIDE the main GGUF (the model
                        # file is the MTP-converted variant); there is NO
                        # separate draft model and emitting
                        # ``--spec-draft-model`` makes the server try to load
                        # an unrelated GGUF as the MTP head — architecture
                        # mismatch crashes the binary. A stale value from a
                        # prior ``draft-simple`` session can still sit in the
                        # Tk var (the UI hides but does not auto-clear it,
                        # preserving the "flip back" UX), so suppress the
                        # emission here with a one-line advisory.
                        if spec_type == "draft-mtp":
                            mp_var = getattr(launcher, "spec_draft_model", None)
                            if mp_var is not None:
                                mp = mp_var.get().strip()
                                if mp:
                                    print(
                                        f"INFO: spec_type=draft-mtp uses the MTP head embedded in the main GGUF;\n"
                                        f"     ignoring spec_draft_model='{mp}'. Switch to draft-simple/draft-eagle3\n"
                                        f"     to use a separate draft model.",
                                        file=sys.stderr,
                                    )
                        else:
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
                        # ``spec_draft_device`` is resolved through the
                        # CUDA_VISIBLE_DEVICES remap helper (see ik_llama branch
                        # comment) so the value emitted matches what the binary
                        # sees post-filter.
                        for var_name, flag in [
                            ("spec_draft_ngl", "--spec-draft-ngl"),
                            ("spec_draft_ctk", "--spec-draft-type-k"),
                            ("spec_draft_ctv", "--spec-draft-type-v"),
                        ]:
                            var = getattr(launcher, var_name, None)
                            if var is not None:
                                v = var.get().strip()
                                if v:
                                    cmd.extend([flag, v])
                        devd_val = _resolve_draft_device_value(launcher)
                        if devd_val:
                            cmd.extend(["--spec-draft-device", devd_val])
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
