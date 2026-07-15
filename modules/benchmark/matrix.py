"""Sweep-matrix model: levers, value expansion, and command building.

Two execution models are supported, reflecting how the underlying tools
work:

* **llama-bench** accepts comma-separated *lists* for most parameters and
  builds the matrix itself in a single process. :func:`llama_bench_command`
  therefore emits one command with each swept lever joined by commas.

* **llama-sweep-bench** (ik_llama) takes server-style flags and has no list
  syntax, so :func:`sweep_bench_commands` computes the cartesian product of
  the swept levers and returns one command per combination, each tagged with
  the combo it represents (for result labelling).

A *lever* is one tunable parameter. A *sweep axis* is a lever plus the list
of values to try for it. The UI collects axes; this module turns them into
commands. Nothing here touches Tk or the filesystem, so it is unit-testable
headless.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .detection import TOOL_LLAMA_BENCH, TOOL_SWEEP_BENCH

# Value kinds
KIND_INT = "int"
KIND_STR = "str"
KIND_FA = "fa"  # flash-attn: on/off, rendered per-tool (list value vs bare flag)
KIND_VEC = "vec"  # a whole vector value (e.g. tensor-split "0.6,0.4"); commas are
# INTERNAL to one value, so sweep points are separated by ';' instead of ','.

# Upper bound on the cartesian product a single llama-sweep-bench sweep may
# expand to. Guards the UI against a two-range sweep allocating millions of
# per-combo commands (each range can already reach 4096 values on its own).
MAX_SWEEP_COMBOS = 4096

# Backends a lever can apply to. Most levers exist in BOTH backends' builds of
# the benchmark tools; a handful of tuning knobs (run-time repack, fused MoE,
# MLA, …) exist ONLY in ik_llama's fork of llama-bench.
_BOTH_BACKENDS = frozenset({"llama.cpp", "ik_llama"})
_IK_BACKEND = frozenset({"ik_llama"})


@dataclass(frozen=True)
class Lever:
    """A single benchmarkable parameter and how it maps to each tool."""

    key: str
    label: str
    flag: str
    kind: str
    tools: frozenset[str]
    # Backends whose build of the tool actually accepts this flag. Defaults to
    # both; ik_llama-only levers narrow it to ``_IK_BACKEND``.
    backends: frozenset[str] = _BOTH_BACKENDS
    # On llama-sweep-bench a handful of ik_llama toggles are BARE flags (present
    # when on, absent when off) rather than the ``<flag> <0|1>`` value form they
    # keep on llama-bench. When True, :func:`sweep_bench_commands` renders this
    # lever like flash-attn's bare pattern; llama-bench rendering is unaffected.
    sweep_bare: bool = False
    # Attribute on the launcher to seed a baseline value from, if any.
    seed_attr: str = ""
    help: str = ""

    def applies_to(self, tool: str, backend: str = "llama.cpp") -> bool:
        return tool in self.tools and backend in self.backends


_BOTH = frozenset({TOOL_LLAMA_BENCH, TOOL_SWEEP_BENCH})
_BENCH_ONLY = frozenset({TOOL_LLAMA_BENCH})
_SWEEP_ONLY = frozenset({TOOL_SWEEP_BENCH})


# Ordered lever catalogue. Order drives the UI row order and the cartesian
# product's column order.
LEVERS: tuple[Lever, ...] = (
    Lever(
        "n_prompt",
        "Prompt tokens (-p)",
        "-p",
        KIND_INT,
        _BENCH_ONLY,
        help="Prompt length(s) for prompt-processing throughput.",
    ),
    Lever(
        "n_gen",
        "Generation tokens (-n)",
        "-n",
        KIND_INT,
        _BOTH,  # llama-bench --n-gen AND llama-sweep-bench -n (TG tokens); both take a value.
        help="Tokens generated for text-generation throughput.",
    ),
    Lever(
        "n_depth",
        "Context depth (-d)",
        "-d",
        KIND_INT,
        _BENCH_ONLY,
        help="Pre-fill the KV cache to this depth before timing.",
    ),
    Lever(
        "ctx_size",
        "Context size (-c)",
        "-c",
        KIND_INT,
        _SWEEP_ONLY,
        seed_attr="ctx_size",
        help="Context size swept internally by llama-sweep-bench.",
    ),
    Lever("n_gpu_layers", "GPU layers (-ngl)", "-ngl", KIND_INT, _BOTH, seed_attr="n_gpu_layers"),
    Lever("threads", "Threads (-t)", "-t", KIND_INT, _BOTH, seed_attr="threads"),
    Lever("batch_size", "Batch size (-b)", "-b", KIND_INT, _BOTH, seed_attr="batch_size"),
    Lever("ubatch_size", "U-batch size (-ub)", "-ub", KIND_INT, _BOTH, seed_attr="ubatch_size"),
    Lever("cache_type_k", "KV cache K (-ctk)", "-ctk", KIND_STR, _BOTH, seed_attr="cache_type_k"),
    Lever("cache_type_v", "KV cache V (-ctv)", "-ctv", KIND_STR, _BOTH, seed_attr="cache_type_v"),
    Lever("flash_attn", "Flash attention (-fa)", "-fa", KIND_FA, _BOTH, seed_attr="flash_attn"),
    Lever(
        "tensor_split",
        "Tensor split (-ts)",
        "-ts",
        KIND_VEC,
        _BOTH,
        seed_attr="tensor_split",
        help="A whole split vector like 0.6,0.4 is ONE value; separate multiple splits with ';'.",
    ),
    Lever("main_gpu", "Main GPU (-mg)", "-mg", KIND_INT, _BOTH, seed_attr="main_gpu"),
    # ── ik_llama-only sweep levers ──────────────────────────────────────────
    # These flags exist only in ik_llama's builds (not upstream llama.cpp), so
    # they are scoped to the ik_llama backend. Some apply to llama-bench only;
    # others (verified against the real binaries) also apply to
    # llama-sweep-bench. On llama-bench every value lever is natively
    # comma-sweepable just like the built-in list levers; on sweep-bench the
    # ``sweep_bare`` toggles below render as bare present/absent flags instead.
    Lever(
        "rtr",
        "Run-time repack (-rtr)",
        "-rtr",
        KIND_INT,
        _BOTH,
        backends=_IK_BACKEND,
        sweep_bare=True,
        help="ik_llama run-time tensor repack (0/1).",
    ),
    Lever(
        "fmoe",
        "Fused MoE (-fmoe)",
        "-fmoe",
        KIND_INT,
        _BENCH_ONLY,
        backends=_IK_BACKEND,
        help="ik_llama fused mixture-of-experts (0/1).",
    ),
    Lever(
        "ger",
        "Grouped expert routing (-ger)",
        "-ger",
        KIND_INT,
        _BENCH_ONLY,
        backends=_IK_BACKEND,
        help="ik_llama grouped expert routing (0/1).",
    ),
    Lever(
        "no_fug",
        "No fused up-gate (-no-fug)",
        "-no-fug",
        KIND_INT,
        _BENCH_ONLY,
        backends=_IK_BACKEND,
        help="ik_llama disable fused up-gate (0/1).",
    ),
    Lever(
        "mla",
        "MLA attention (-mla)",
        "-mla",
        KIND_INT,
        _BOTH,
        backends=_IK_BACKEND,
        help="ik_llama MLA attention mode (0/1/2).",
    ),
    Lever(
        "amb",
        "Attn max batch (-amb)",
        "-amb",
        KIND_INT,
        _BOTH,
        backends=_IK_BACKEND,
        help="ik_llama attention max batch size.",
    ),
    Lever(
        "ik_n_cpu_moe",
        "N CPU MoE (--n-cpu-moe)",
        "--n-cpu-moe",
        KIND_INT,
        _BOTH,
        backends=_IK_BACKEND,
        help="ik_llama number of MoE layers kept on CPU.",
    ),
    Lever(
        "mqkv",
        "Merge QKV (-mqkv)",
        "-mqkv",
        KIND_INT,
        _BOTH,
        backends=_IK_BACKEND,
        sweep_bare=True,
        help="ik_llama merge QKV (0/1).",
    ),
    Lever(
        "muge",
        "Merge up-gate experts (-muge)",
        "-muge",
        KIND_INT,
        _BENCH_ONLY,
        backends=_IK_BACKEND,
        help="ik_llama merge up-gate experts (0/1).",
    ),
    Lever(
        "rcache",
        "Rope cache (-rcache)",
        "-rcache",
        KIND_INT,
        _BENCH_ONLY,
        backends=_IK_BACKEND,
        help="ik_llama rope cache (0/1).",
    ),
    Lever(
        "ser",
        "Smart expert reduction (-ser)",
        "-ser",
        KIND_VEC,
        _BOTH,
        backends=_IK_BACKEND,
        help='Value is "i,f" (e.g. 7,1) — ONE value; sweep several with ";".',
    ),
    Lever(
        "ot",
        "Override tensor (-ot)",
        "-ot",
        KIND_VEC,
        _BOTH,
        backends=_IK_BACKEND,
        help='A tensor-override pattern is ONE value; sweep several with ";".',
    ),
    # ── MTP / speculative-decoding levers (llama-sweep-bench + ik_llama only) ─
    # ik_llama's llama-sweep-bench supports embedded-head multi-token prediction:
    # the bare ``-mtp`` legacy shortcut (the embedded head — NO draft model) plus
    # the ``--draft-*`` controls and the ``-mtprot`` requant-output knob. ik
    # llama-bench REJECTS all of these ("invalid parameter for argument"), so they
    # are scoped to sweep-bench only. ``-mtp`` is itself a BARE on/off flag on
    # sweep-bench (present when enabled, absent otherwise), so it renders via the
    # same ``sweep_bare`` path as -rtr/-mqkv and can sweep 0/1 to measure MTP's
    # speedup. ``-mtp`` cannot be combined with ``--spec-stage``, so only the
    # shortcut + ``--draft-*`` controls are exposed (no --spec-stage lever).
    Lever(
        "mtp",
        "MTP enable (-mtp)",
        "-mtp",
        KIND_INT,
        _SWEEP_ONLY,
        backends=_IK_BACKEND,
        sweep_bare=True,
        help="ik_llama embedded-head multi-token prediction (0/1); no draft model.",
    ),
    Lever(
        "draft_max",
        "MTP draft max (--draft-max)",
        "--draft-max",
        KIND_INT,
        _SWEEP_ONLY,
        backends=_IK_BACKEND,
        help="Max speculative draft tokens per step.",
    ),
    Lever(
        "draft_min",
        "MTP draft min (--draft-min)",
        "--draft-min",
        KIND_INT,
        _SWEEP_ONLY,
        backends=_IK_BACKEND,
        help="Min speculative draft tokens per step.",
    ),
    Lever(
        "draft_p_min",
        "MTP draft p-min (--draft-p-min)",
        "--draft-p-min",
        KIND_STR,
        _SWEEP_ONLY,
        backends=_IK_BACKEND,
        help="Min draft acceptance probability, a float like 0.5.",
    ),
    Lever(
        "mtprot",
        "MTP requant output (-mtprot)",
        "-mtprot",
        KIND_STR,
        _SWEEP_ONLY,
        backends=_IK_BACKEND,
        help="MTP requantize-output-tensor type, e.g. q8_0.",
    ),
)

LEVERS_BY_KEY: dict[str, Lever] = {lever.key: lever for lever in LEVERS}

# Accepted flash-attn values (normalised, lower-case).
FA_VALUES = ("on", "off")


class SweepError(ValueError):
    """Raised for malformed sweep specifications (bad ranges, empty axes)."""


# ─────────────────────────────────────────────────────────────────────────────
# Value expansion
# ─────────────────────────────────────────────────────────────────────────────


def _fmt_number(n: float) -> str:
    """Render a numeric value without a trailing ``.0`` for whole numbers."""
    if isinstance(n, int) or (isinstance(n, float) and n.is_integer()):
        return str(int(n))
    # Trim floating noise to a stable, compact form.
    return repr(round(n, 6))


def parse_list(raw: str, kind: str = KIND_STR) -> list[str]:
    """Split a comma/whitespace-separated list into de-duplicated values.

    Order is preserved; duplicates are dropped. Integer/float kinds are
    validated and re-normalised (``"08"`` -> ``"8"``). Flash-attn values are
    normalised to ``on``/``off``.
    """
    if raw is None:
        return []
    # Vector values (tensor-split) keep commas internally, so they sweep on
    # ';' rather than ','. Everything else sweeps on ','.
    sep = ";" if kind == KIND_VEC else ","
    tokens = [t.strip() for t in str(raw).replace("\n", sep).split(sep)]
    tokens = [t for t in tokens if t]
    out: list[str] = []
    for tok in tokens:
        value = _normalise_token(tok, kind)
        if value not in out:
            out.append(value)
    return out


def _normalise_token(tok: str, kind: str) -> str:
    if kind == KIND_INT:
        # int() accepts leading "-"/"+" so negatives like "-1" pass; anything
        # non-integer (e.g. "0.5") is rejected rather than coerced to float.
        try:
            return str(int(tok, 10))
        except ValueError as exc:
            raise SweepError(f"{tok!r} is not a valid integer") from exc
    if kind == KIND_FA:
        low = tok.strip().lower()
        if low in ("1", "true", "yes", "on"):
            return "on"
        if low in ("0", "false", "no", "off"):
            return "off"
        raise SweepError(f"{tok!r} is not a valid flash-attn value (use on/off)")
    return tok


def expand_range(vmin: float, vmax: float, vstep: float, kind: str = KIND_INT) -> list[str]:
    """Inclusive ``vmin..vmax`` in ``vstep`` increments, as normalised strings.

    ``vstep`` must be non-zero. If ``vmax < vmin`` the range is walked
    downward when ``vstep`` is negative, otherwise a single value is
    returned. Guards against runaway ranges (>4096 points).
    """
    if vstep == 0:
        raise SweepError("increment must not be zero")
    if (vmax - vmin) * vstep < 0:
        # Step sign disagrees with the min→max direction, so the range can
        # never reach vmax — this is a contradictory spec, not an empty walk.
        raise SweepError(
            f"increment {_fmt_number(vstep)} moves away from {_fmt_number(vmax)} " f"starting at {_fmt_number(vmin)}"
        )
    count = int(abs((vmax - vmin) / vstep)) + 1
    if count > 4096:
        raise SweepError(f"range expands to {count} values (max 4096)")
    out: list[str] = []
    val = vmin
    for _ in range(count):
        token = _normalise_token(_fmt_number(val), kind)
        if token not in out:
            out.append(token)
        val += vstep
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Sweep axes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Axis:
    """A sweep dimension: concrete values plus how they map to each tool.

    An axis wraps EITHER a built-in :class:`Lever` (identified by ``key``) or a
    user-defined *custom flag* (``custom_flag`` set to something like ``-ot``).
    Command builders never reach into the lever directly; they go through the
    uniform :pyattr:`flag` / :pyattr:`kind` / :pyattr:`label` accessors and
    :meth:`applies_to`, which dispatch to the lever or the custom fields as
    appropriate. This keeps a custom flag a first-class matrix dimension while
    leaving the built-in ``Axis(key, values)`` construction (and the
    :pyattr:`lever` property) working unchanged.

    Custom flags apply to BOTH tools and are never flash-attn. Their values use
    :data:`KIND_STR` (list mode) or :data:`KIND_INT` (numeric range mode),
    mirroring the built-in lever rows.
    """

    key: str
    values: list[str] = field(default_factory=list)
    custom_flag: str = ""
    custom_kind: str = KIND_STR

    def __post_init__(self) -> None:
        # A custom flag must look like a CLI flag; otherwise a stray word would
        # be rendered as ``word value`` and silently mis-invoke the tool.
        if self.custom_flag and not str(self.custom_flag).startswith("-"):
            raise SweepError(
                f"custom flag {self.custom_flag!r} must start with '-' " f"(e.g. -ot or --override-tensor)"
            )

    @property
    def is_custom(self) -> bool:
        return bool(self.custom_flag)

    @property
    def lever(self) -> Lever:
        try:
            return LEVERS_BY_KEY[self.key]
        except KeyError as exc:
            raise SweepError(f"unknown lever {self.key!r}") from exc

    @property
    def flag(self) -> str:
        """CLI flag this axis renders (e.g. ``-ngl`` or a custom ``-ot``)."""
        return self.custom_flag if self.is_custom else self.lever.flag

    @property
    def kind(self) -> str:
        """Value kind driving rendering (KIND_INT/KIND_STR/KIND_FA/KIND_VEC)."""
        return self.custom_kind if self.is_custom else self.lever.kind

    @property
    def label(self) -> str:
        """Human label for previews and ignored-axis notes."""
        return self.custom_flag if self.is_custom else self.lever.label

    @property
    def sweep_bare(self) -> bool:
        """Whether sweep-bench renders this as a bare present/absent flag.

        Custom flags always render as ``<flag> <value>`` pairs, so they are
        never bare; built-in levers delegate to their lever definition.
        """
        return False if self.is_custom else self.lever.sweep_bare

    def applies_to(self, tool: str, backend: str = "llama.cpp") -> bool:
        # A custom flag has no per-tool/per-backend restriction — it sweeps on
        # whichever tool the user runs (a comma-list on llama-bench, per-combo
        # otherwise) and on either backend.
        return True if self.is_custom else self.lever.applies_to(tool, backend)

    @property
    def is_swept(self) -> bool:
        return len(self.values) > 1


def split_axes_for_tool(axes: list[Axis], tool: str, backend: str = "llama.cpp") -> tuple[list[Axis], list[Axis]]:
    """Partition ``axes`` into (applicable, ignored) for ``tool``/``backend``.

    An axis is ignored when it doesn't apply to the tool (e.g. ``-p`` on
    sweep-bench) or backend (e.g. an ik_llama-only lever on a llama.cpp build)
    or when it has no values. Custom flags always apply.
    """
    applicable: list[Axis] = []
    ignored: list[Axis] = []
    for axis in axes:
        if not axis.values:
            continue
        if axis.applies_to(tool, backend):
            applicable.append(axis)
        else:
            ignored.append(axis)
    return applicable, ignored


def matrix_size(axes: list[Axis], tool: str, backend: str = "llama.cpp") -> int:
    """Number of distinct parameter combinations for ``tool``/``backend``.

    For llama-bench this is the number of rows the single invocation
    produces; for sweep-bench it is the number of separate processes run.
    """
    applicable, _ = split_axes_for_tool(axes, tool, backend)
    total = 1
    for axis in applicable:
        total *= max(1, len(axis.values))
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Command building
# ─────────────────────────────────────────────────────────────────────────────


def _extra_tokens(extra_args: str | list[str] | None) -> list[str]:
    """Flatten fixed extra-args into shell tokens appended to every command.

    Accepts a single free-form string OR a list of free-form strings (one per
    UI "Extra args" row). Each element is ``shlex.split`` independently and the
    results are concatenated in order, so the multi-row editor composes exactly
    like the old single field did.
    """
    if not extra_args:
        return []
    import shlex

    rows = [extra_args] if isinstance(extra_args, str) else extra_args
    tokens: list[str] = []
    for row in rows:
        text = str(row)
        if not text:
            continue
        try:
            if os.name == "nt":
                # posix=False keeps a backslash path (e.g. ``C:\models\a.gguf``)
                # intact instead of eating the backslashes — but it also RETAINS
                # the grouping quotes around a spaced path (``"C:\a b.gguf"`` ->
                # ``'"C:\\a b.gguf"'``), which would reach the tool as a literal
                # quoted filename. Strip a single matched surrounding quote pair
                # from each token to recover the intended value.
                for tok in shlex.split(text, posix=False):
                    if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in ('"', "'"):
                        tok = tok[1:-1]
                    tokens.append(tok)
            else:
                tokens.extend(shlex.split(text, posix=True))
        except ValueError as exc:
            # e.g. an unmatched quote in an Extra-args row. Surface it as a
            # SweepError so the UI's validation path reports it instead of
            # letting it escape the Tk callback as an uncaught exception.
            raise SweepError(f"could not parse extra args: {exc}") from exc
    return tokens


# Within-vector device separator for llama-bench. Upstream llama-bench parses a
# ``-ts`` argument by splitting on ',' into separate benchmark CASES and on ';'
# or '/' into the devices WITHIN one vector (unlike llama-server, which uses ','
# for devices). So a KIND_VEC value like "0.6,0.4" must have its internal commas
# rewritten to this separator BEFORE we comma-join the swept cases, otherwise
# llama-bench reads each device as its own 1-GPU benchmark case.
_LLAMA_BENCH_VEC_DEVICE_SEP = "/"


def _vec_for_llama_bench(value: str) -> str:
    """Rewrite a KIND_VEC value's internal device commas for llama-bench."""
    return value.replace(",", _LLAMA_BENCH_VEC_DEVICE_SEP)


def _is_truthy(value: str) -> bool:
    """Whether a sweep-bare toggle value means "on" (present) vs "off" (absent)."""
    return str(value).strip().lower() in ("1", "on", "true")


def _render_fa_values(values: list[str], backend: str) -> list[str]:
    """Map canonical ``on``/``off`` flash-attn values for a llama-bench backend.

    Upstream llama.cpp accepts the literal ``on``/``off`` tokens; ik_llama's
    fork (and older builds) expect numeric ``1``/``0``. Any non-canonical
    token is passed through untouched.
    """
    if backend != "ik_llama":
        return list(values)
    mapping = {"on": "1", "off": "0"}
    return [mapping.get(v, v) for v in values]


def llama_bench_command(
    exe: str,
    model: str,
    axes: list[Axis],
    *,
    backend: str = "llama.cpp",
    output_format: str = "json",
    repetitions: int | None = None,
    extra_args: str | list[str] | None = None,
) -> list[str]:
    """Build the single llama-bench invocation covering the whole matrix.

    Swept levers become comma-joined list arguments; llama-bench expands
    them internally. Levers that don't apply to llama-bench are silently
    skipped (use :func:`split_axes_for_tool` to surface them in the UI).

    ``backend`` selects flash-attn rendering: ``"ik_llama"`` emits numeric
    ``-fa 1,0``; any other value keeps the upstream ``-fa on,off`` tokens.
    """
    if not exe:
        raise SweepError("no llama-bench executable")
    if not model:
        raise SweepError("no model selected")
    cmd: list[str] = [exe, "-m", model]
    applicable, _ = split_axes_for_tool(axes, TOOL_LLAMA_BENCH, backend)
    for axis in applicable:
        if axis.kind == KIND_FA:
            values = _render_fa_values(axis.values, backend)
        elif axis.kind == KIND_VEC and axis.key == "tensor_split":
            # ONLY tensor-split's vector commas separate DEVICES: rewrite them to
            # llama-bench's within-vector separator before the comma-join across
            # swept vectors turns comma into the case delimiter. Other KIND_VEC
            # levers (ik_llama's -ser "i,f" and -ot patterns) use commas/'='
            # literally, so they must pass through UNCHANGED.
            values = [_vec_for_llama_bench(v) for v in axis.values]
        elif axis.kind == KIND_VEC:
            # Non-tensor-split vector levers (ik_llama's -ser "i,f", -ot patterns)
            # keep their commas LITERAL. Comma-joining several swept values here
            # would collide with llama-bench's own comma matrix-splitter and lose
            # the pair boundaries (e.g. -ser ["7,1","8,0"] -> "-ser 7,1,8,0",
            # read as four scalars). A single value is unambiguous and parses on
            # the real binary; more than one simply can't be expressed here.
            if len(axis.values) > 1:
                raise SweepError(
                    f"cannot sweep multiple {axis.flag!r} vector values on llama-bench "
                    f"(its commas collide with the matrix separator); use a single value, "
                    f"or llama-sweep-bench"
                )
            values = axis.values
        else:
            values = axis.values
        # Custom flags render as a native comma-list too; they only sweep on
        # llama-bench when the flag itself accepts comma-separated values.
        cmd += [axis.flag, ",".join(values)]
    if repetitions and repetitions > 0:
        cmd += ["-r", str(repetitions)]
    if output_format:
        cmd += ["-o", output_format]
    cmd += _extra_tokens(extra_args)
    return cmd


def _cartesian(applicable: list[Axis]) -> list[dict[str, str]]:
    """Cartesian product of axis values → list of ``{key: value}`` combos."""
    combos: list[dict[str, str]] = [{}]
    for axis in applicable:
        combos = [dict(combo, **{axis.key: value}) for combo in combos for value in axis.values]
    return combos


def sweep_bench_commands(
    exe: str,
    model: str,
    axes: list[Axis],
    *,
    backend: str = "llama.cpp",
    extra_args: str | list[str] | None = None,
) -> list[tuple[list[str], dict[str, str]]]:
    """Build one llama-sweep-bench command per parameter combination.

    Returns ``(command, combo)`` pairs, where ``combo`` maps lever key to the
    value used — the UI tags each result row with it. Flash-attn is rendered
    with an EXPLICIT value for both states (``-fa on`` / ``-fa off``), mirroring
    the server launch path (``modules/launch.py``): sweep-bench uses ik_llama's
    server-style flags, which require a value after ``--flash-attn`` (a bare flag
    errors), and an explicit ``off`` is needed to override ik_llama's ``on``
    default rather than silently benchmarking it.
    """
    if not exe:
        raise SweepError("no llama-sweep-bench executable")
    if not model:
        raise SweepError("no model selected")
    applicable, _ = split_axes_for_tool(axes, TOOL_SWEEP_BENCH, backend)
    # Bound the cartesian product BEFORE materialising it. Each axis range can
    # expand to 4096 values, so two ranges could otherwise allocate millions of
    # combo dicts and hang/exhaust memory in the UI (even just building the
    # preview). Compute the product from lengths and refuse early.
    total = 1
    for axis in applicable:
        total *= max(1, len(axis.values))
    if total > MAX_SWEEP_COMBOS:
        raise SweepError(
            f"sweep expands to {total} llama-sweep-bench runs (max {MAX_SWEEP_COMBOS}); "
            f"narrow the parameter ranges."
        )
    extra = _extra_tokens(extra_args)
    out: list[tuple[list[str], dict[str, str]]] = []
    for combo in _cartesian(applicable):
        cmd: list[str] = [exe, "-m", model]
        for axis in applicable:
            value = combo[axis.key]
            if axis.sweep_bare:
                # A handful of ik_llama toggles (-rtr, -mqkv) are BARE flags on
                # sweep-bench: present when on, absent when off. Emitting
                # "-rtr 1" errors here (unlike llama-bench, where they are
                # native "<flag> 0,1" value sweeps). Append the bare flag only
                # for a truthy value; the combo dict still records the value so
                # the result row stays labelled either way.
                if _is_truthy(value):
                    cmd.append(axis.flag)
                continue
            # Flash-attn renders like any other flag/value pair: ik_llama's
            # server-style --flash-attn requires an explicit on/off token (a
            # bare flag errors, and "off" must be explicit to override the
            # binary's default), matching modules/launch.py.
            cmd += [axis.flag, value]
        cmd += extra
        out.append((cmd, combo))
    return out


def build_commands(
    tool: str,
    exe: str,
    model: str,
    axes: list[Axis],
    *,
    backend: str = "llama.cpp",
    output_format: str = "json",
    repetitions: int | None = None,
    extra_args: str | list[str] | None = None,
) -> list[tuple[list[str], dict[str, str]]]:
    """Uniform entry point returning ``(command, combo)`` pairs for ``tool``.

    llama-bench yields a single pair with an empty combo (it emits the whole
    matrix itself); sweep-bench yields one pair per combination. ``backend``
    is forwarded to :func:`llama_bench_command` for flash-attn rendering.
    """
    if tool == TOOL_LLAMA_BENCH:
        cmd = llama_bench_command(
            exe,
            model,
            axes,
            backend=backend,
            output_format=output_format,
            repetitions=repetitions,
            extra_args=extra_args,
        )
        return [(cmd, {})]
    if tool == TOOL_SWEEP_BENCH:
        # ``repetitions`` is intentionally NOT forwarded: llama-sweep-bench has
        # no -r/--repetitions flag (it sweeps context internally), so there is
        # nothing to render for it.
        return sweep_bench_commands(exe, model, axes, backend=backend, extra_args=extra_args)
    raise SweepError(f"unknown tool {tool!r}")
