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

from dataclasses import dataclass, field

from .detection import TOOL_LLAMA_BENCH, TOOL_SWEEP_BENCH

# Value kinds
KIND_INT = "int"
KIND_STR = "str"
KIND_FA = "fa"  # flash-attn: on/off, rendered per-tool (list value vs bare flag)


@dataclass(frozen=True)
class Lever:
    """A single benchmarkable parameter and how it maps to each tool."""

    key: str
    label: str
    flag: str
    kind: str
    tools: frozenset[str]
    # Attribute on the launcher to seed a baseline value from, if any.
    seed_attr: str = ""
    help: str = ""

    def applies_to(self, tool: str) -> bool:
        return tool in self.tools


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
        _BENCH_ONLY,
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
    Lever("tensor_split", "Tensor split (-ts)", "-ts", KIND_STR, _BOTH, seed_attr="tensor_split"),
    Lever("main_gpu", "Main GPU (-mg)", "-mg", KIND_INT, _BOTH, seed_attr="main_gpu"),
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
    tokens = [t.strip() for t in str(raw).replace("\n", ",").split(",")]
    tokens = [t for t in tokens if t]
    out: list[str] = []
    for tok in tokens:
        value = _normalise_token(tok, kind)
        if value not in out:
            out.append(value)
    return out


def _normalise_token(tok: str, kind: str) -> str:
    if kind == KIND_INT:
        try:
            return str(int(tok, 10))
        except ValueError:
            # Allow "-1" style already handled by int(); anything else is bad.
            try:
                return _fmt_number(float(tok))
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
        # Direction of step disagrees with min→max direction; just the start.
        return [_fmt_number(vmin)]
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
    """A lever plus the concrete values to benchmark for it."""

    key: str
    values: list[str] = field(default_factory=list)

    @property
    def lever(self) -> Lever:
        try:
            return LEVERS_BY_KEY[self.key]
        except KeyError as exc:
            raise SweepError(f"unknown lever {self.key!r}") from exc

    @property
    def is_swept(self) -> bool:
        return len(self.values) > 1


def split_axes_for_tool(axes: list[Axis], tool: str) -> tuple[list[Axis], list[Axis]]:
    """Partition ``axes`` into (applicable, ignored) for ``tool``.

    An axis is ignored when its lever doesn't apply to the tool (e.g.
    ``-p`` on sweep-bench) or when it has no values.
    """
    applicable: list[Axis] = []
    ignored: list[Axis] = []
    for axis in axes:
        if not axis.values:
            continue
        if axis.lever.applies_to(tool):
            applicable.append(axis)
        else:
            ignored.append(axis)
    return applicable, ignored


def matrix_size(axes: list[Axis], tool: str) -> int:
    """Number of distinct parameter combinations for ``tool``.

    For llama-bench this is the number of rows the single invocation
    produces; for sweep-bench it is the number of separate processes run.
    """
    applicable, _ = split_axes_for_tool(axes, tool)
    total = 1
    for axis in applicable:
        total *= max(1, len(axis.values))
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Command building
# ─────────────────────────────────────────────────────────────────────────────


def _extra_tokens(extra_args: str | list[str] | None) -> list[str]:
    if not extra_args:
        return []
    if isinstance(extra_args, str):
        import shlex

        return shlex.split(extra_args)
    return [str(a) for a in extra_args]


def llama_bench_command(
    exe: str,
    model: str,
    axes: list[Axis],
    *,
    output_format: str = "json",
    repetitions: int | None = None,
    extra_args: str | list[str] | None = None,
) -> list[str]:
    """Build the single llama-bench invocation covering the whole matrix.

    Swept levers become comma-joined list arguments; llama-bench expands
    them internally. Levers that don't apply to llama-bench are silently
    skipped (use :func:`split_axes_for_tool` to surface them in the UI).
    """
    if not exe:
        raise SweepError("no llama-bench executable")
    if not model:
        raise SweepError("no model selected")
    cmd: list[str] = [exe, "-m", model]
    applicable, _ = split_axes_for_tool(axes, TOOL_LLAMA_BENCH)
    for axis in applicable:
        lever = axis.lever
        # flash-attn list values (on/off) are passed through as-is.
        cmd += [lever.flag, ",".join(axis.values)]
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
    extra_args: str | list[str] | None = None,
) -> list[tuple[list[str], dict[str, str]]]:
    """Build one llama-sweep-bench command per parameter combination.

    Returns ``(command, combo)`` pairs, where ``combo`` maps lever key to the
    value used — the UI tags each result row with it. Flash-attn is rendered
    as a bare flag (``-fa`` present for ``on``, absent for ``off``) because
    sweep-bench takes server-style boolean flags.
    """
    if not exe:
        raise SweepError("no llama-sweep-bench executable")
    if not model:
        raise SweepError("no model selected")
    applicable, _ = split_axes_for_tool(axes, TOOL_SWEEP_BENCH)
    extra = _extra_tokens(extra_args)
    out: list[tuple[list[str], dict[str, str]]] = []
    for combo in _cartesian(applicable):
        cmd: list[str] = [exe, "-m", model]
        for axis in applicable:
            lever = axis.lever
            value = combo[axis.key]
            if lever.kind == KIND_FA:
                if value == "on":
                    cmd.append(lever.flag)
                # "off" => omit the flag entirely (server-style default off)
            else:
                cmd += [lever.flag, value]
        cmd += extra
        out.append((cmd, combo))
    return out


def build_commands(
    tool: str,
    exe: str,
    model: str,
    axes: list[Axis],
    *,
    output_format: str = "json",
    repetitions: int | None = None,
    extra_args: str | list[str] | None = None,
) -> list[tuple[list[str], dict[str, str]]]:
    """Uniform entry point returning ``(command, combo)`` pairs for ``tool``.

    llama-bench yields a single pair with an empty combo (it emits the whole
    matrix itself); sweep-bench yields one pair per combination.
    """
    if tool == TOOL_LLAMA_BENCH:
        cmd = llama_bench_command(
            exe,
            model,
            axes,
            output_format=output_format,
            repetitions=repetitions,
            extra_args=extra_args,
        )
        return [(cmd, {})]
    if tool == TOOL_SWEEP_BENCH:
        return sweep_bench_commands(exe, model, axes, extra_args=extra_args)
    raise SweepError(f"unknown tool {tool!r}")
