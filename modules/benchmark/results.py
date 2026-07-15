"""Parse benchmark tool output into rows and export them.

``llama-bench -o json`` writes a JSON array of per-test objects to stdout;
:func:`parse_llama_bench_json` turns that into :class:`ResultRow` records.
``llama-sweep-bench`` prints a pipe-delimited table (no machine-readable
output mode), so :func:`parse_sweep_bench_table` scrapes that, prefixing the
swept-parameter combo that produced it.

Rows carry an ordered ``columns`` mapping so heterogeneous runs still export
cleanly — the exporters compute the union of columns in first-seen order.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field


@dataclass
class ResultRow:
    """One benchmark measurement as an ordered mapping of column -> value."""

    columns: dict[str, str] = field(default_factory=dict)

    def get(self, key: str, default: str = "") -> str:
        return self.columns.get(key, default)


# Preferred column order for llama-bench rows; unknown extras are appended.
_LLAMA_BENCH_PREFERRED = (
    "model",
    "test",
    "t/s",
    "t/s_stddev",
    "n_gpu_layers",
    "n_threads",
    "n_batch",
    "n_ubatch",
    "type_k",
    "type_v",
    "flash_attn",
    "main_gpu",
    "tensor_split",
    "n_prompt",
    "n_gen",
    "n_depth",
)

# JSON field -> display column for the curated subset.
_LLAMA_BENCH_FIELD_MAP = {
    "test": "test",
    "avg_ts": "t/s",
    "stddev_ts": "t/s_stddev",
    "n_gpu_layers": "n_gpu_layers",
    "n_threads": "n_threads",
    "n_batch": "n_batch",
    "n_ubatch": "n_ubatch",
    "type_k": "type_k",
    "type_v": "type_v",
    "flash_attn": "flash_attn",
    "main_gpu": "main_gpu",
    "tensor_split": "tensor_split",
    "n_prompt": "n_prompt",
    "n_gen": "n_gen",
    "n_depth": "n_depth",
}

# Noisy metadata fields that must NOT be surfaced as extra columns even though
# llama-bench emits them. Everything else scalar (custom-flag / option sweeps
# like ``use_mmap`` or ``split_mode``) is appended after the curated columns so
# those rows stay distinguishable in the grid/export. Fields ending in ``_ns``/
# ``_ts`` (timing internals / already-mapped throughput) and any non-scalar
# (sample arrays, nested objects) are skipped separately.
_LLAMA_BENCH_EXTRA_DENYLIST = frozenset(
    {
        "build_commit",
        "build_number",
        "cpu_info",
        "gpu_info",
        "backend",
        "model_filename",
        "model_type",
        "model_size",
        "model_n_params",
        "test_time",
    }
)


def _stringify(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        # Compact but readable; throughput values keep two decimals.
        return f"{value:.2f}" if abs(value) >= 0.01 or value == 0 else repr(value)
    return str(value)


def _model_name(obj: dict) -> str:
    for key in ("model_type", "model_filename", "model"):
        val = obj.get(key)
        if val:
            text = str(val)
            # Reduce a filesystem path to its basename for readability.
            return text.replace("\\", "/").rsplit("/", 1)[-1]
    return ""


def parse_llama_bench_json(stdout: str) -> list[ResultRow]:
    """Parse ``llama-bench -o json`` stdout into rows.

    Tolerates leading/trailing non-JSON noise by extracting the outermost
    JSON array. Returns ``[]`` if nothing parseable is found (the caller can
    then fall back to showing raw output).
    """
    text = (stdout or "").strip()
    if not text:
        return []
    data = _extract_json_array(text)
    if data is None:
        return []
    rows: list[ResultRow] = []
    for obj in data:
        if not isinstance(obj, dict):
            continue
        cols: dict[str, str] = {"model": _model_name(obj)}
        for field_name, display in _LLAMA_BENCH_FIELD_MAP.items():
            if field_name in obj:
                cols[display] = _stringify(obj[field_name])
        # Surface any remaining scalar fields the curated map dropped so custom /
        # option-sweep rows stay distinguishable, skipping noisy metadata, the
        # ``_ns``/``_ts`` timing internals, and non-scalars (sample arrays etc.).
        for key, value in obj.items():
            if key in _LLAMA_BENCH_FIELD_MAP or key in _LLAMA_BENCH_EXTRA_DENYLIST:
                continue
            if key.endswith("_ns") or key.endswith("_ts"):
                continue
            if not isinstance(value, (str, int, float, bool)):
                continue
            cols.setdefault(key, _stringify(value))
        rows.append(ResultRow(columns=_order_columns(cols, _LLAMA_BENCH_PREFERRED)))
    return rows


def _extract_json_array(text: str):
    """Return the first top-level JSON array in ``text``, or ``None``."""
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else None
    except json.JSONDecodeError:
        pass
    # A backend may print a bracketed banner (e.g. "[INFO] loading model")
    # before the results array, so a first-'[' .. last-']' span won't parse.
    # Try to decode an array starting at each '[' and return the first that
    # yields a list, ignoring '[' positions that don't begin valid JSON.
    decoder = json.JSONDecoder()
    idx = text.find("[")
    while idx >= 0:
        try:
            parsed, _ = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return parsed
        idx = text.find("[", idx + 1)
    return None


def _order_columns(cols: dict[str, str], preferred: tuple[str, ...]) -> dict[str, str]:
    ordered: dict[str, str] = {}
    for key in preferred:
        if key in cols:
            ordered[key] = cols[key]
    for key, val in cols.items():
        if key not in ordered:
            ordered[key] = val
    return ordered


def parse_sweep_bench_table(stdout: str, combo: dict[str, str] | None = None) -> list[ResultRow]:
    """Parse llama-sweep-bench's result table into rows.

    Two on-disk shapes exist: a markdown pipe-delimited table (``| PP | TG |
    ...``, most builds) and a plain WHITESPACE-delimited one (``PP  TG  N_KV
    S_PP t/s ...`` with no pipes, printed by some ik_llama builds). This
    dispatches on whether any line is pipe-delimited and parses accordingly.

    Each swept-parameter value in ``combo`` is prefixed as its own column so
    rows from different combinations remain distinguishable after merging.
    Lines that aren't table rows are ignored; the header row supplies column
    names and dashed separator rows are skipped.
    """
    combo = combo or {}
    prefix_cols = {f"[{k}]": v for k, v in combo.items()}
    lines = (stdout or "").splitlines()
    if any(line.strip().startswith("|") for line in lines):
        return _parse_pipe_sweep_table(lines, prefix_cols)
    return _parse_ws_sweep_table(lines, prefix_cols)


def _parse_pipe_sweep_table(lines: list[str], prefix_cols: dict[str, str]) -> list[ResultRow]:
    """Parse the markdown pipe-delimited sweep-bench table."""
    rows: list[ResultRow] = []
    header: list[str] | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if not cells:
            continue
        if set("".join(cells)) <= set("-: "):
            # Separator row like |---|---|
            continue
        if header is None:
            header = cells
            continue
        cols: dict[str, str] = dict(prefix_cols)
        for name, value in zip(header, cells, strict=False):
            if name:
                cols[name] = value
        rows.append(ResultRow(columns=cols))
    return rows


def _ws_split(text: str) -> list[str]:
    """Split a whitespace-delimited table row into cells.

    Column names like ``S_PP t/s`` contain a single internal space, so split on
    runs of 2+ spaces to keep them intact; fall back to a plain whitespace split
    when that yields a single column (a run of single-space-separated cells).
    """
    parts = re.split(r"\s{2,}", text.strip())
    if len(parts) <= 1:
        parts = text.split()
    return [p.strip() for p in parts if p.strip()]


def _parse_ws_sweep_table(lines: list[str], prefix_cols: dict[str, str]) -> list[ResultRow]:
    """Parse the whitespace-delimited sweep-bench table.

    The header is the row naming ``PP``/``TG``/``N_KV`` (case-insensitive);
    dashed separator rows and any trailing log lines (whose column count differs
    from the header) are ignored.
    """
    rows: list[ResultRow] = []
    header: list[str] | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if set(stripped) <= set("-: "):
            # Dashed separator row like "----  ----  ----".
            continue
        cells = _ws_split(stripped)
        if header is None:
            upper = {c.upper() for c in cells}
            if {"PP", "TG", "N_KV"} <= upper:
                header = cells
            continue
        if len(cells) != len(header):
            # A trailing log line, not a data row.
            continue
        cols: dict[str, str] = dict(prefix_cols)
        for name, value in zip(header, cells, strict=False):
            if name:
                cols[name] = value
        rows.append(ResultRow(columns=cols))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────


def collect_columns(rows: list[ResultRow]) -> list[str]:
    """Union of all column names across ``rows``, in first-seen order."""
    seen: list[str] = []
    for row in rows:
        for key in row.columns:
            if key not in seen:
                seen.append(key)
    return seen


# Excel/Sheets treat a cell beginning with one of these as a formula, so a
# value like ``=CMD()`` can execute on open. Prefix such cells with an
# apostrophe to neutralise formula injection (CWE-1236).
_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@")

# A cell that is nothing but a signed integer/float is never a formula, so it
# must NOT be apostrophe-prefixed — otherwise legitimate negative values like
# ``-1`` (``-ngl -1`` = "all layers") or ``main_gpu`` columns would be turned
# into the text string ``'-1`` for spreadsheet consumers.
_NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _csv_safe(value: str) -> str:
    text = "" if value is None else str(value)
    # Spreadsheets strip leading whitespace/control chars before evaluating a
    # cell, so a value like "\t=CMD()" or " -2+3" is still a formula candidate.
    # Test the first NON-whitespace character, and the numeric-passthrough check
    # against the stripped value (a bare signed number is never a formula).
    stripped = text.lstrip("\t\r\n\v\f ")
    if stripped[:1] in _CSV_FORMULA_PREFIXES and not _NUMERIC_RE.fullmatch(stripped):
        return "'" + text
    return text


def to_csv(rows: list[ResultRow]) -> str:
    cols = collect_columns(rows)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([_csv_safe(c) for c in cols])
    for row in rows:
        writer.writerow([_csv_safe(row.get(c)) for c in cols])
    return buf.getvalue()


def to_json(rows: list[ResultRow]) -> str:
    return json.dumps([row.columns for row in rows], indent=2, ensure_ascii=False)


def _md_safe(value: str) -> str:
    """Neutralise a value for a Markdown table cell.

    Escapes ``|`` (which would otherwise start a new column) and folds
    newlines to spaces (which would otherwise break the row).
    """
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\r\n", " ").replace("\n", " ").replace("\r", " ")


def to_markdown(rows: list[ResultRow]) -> str:
    cols = collect_columns(rows)
    if not cols:
        return "_(no results)_\n"
    header = [_md_safe(c) for c in cols]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_md_safe(row.get(c)) for c in cols) + " |")
    return "\n".join(lines) + "\n"


EXPORTERS = {
    "csv": (to_csv, ".csv"),
    "json": (to_json, ".json"),
    "markdown": (to_markdown, ".md"),
}


def export(rows: list[ResultRow], fmt: str) -> str:
    try:
        exporter, _ = EXPORTERS[fmt]
    except KeyError as exc:
        raise ValueError(f"unknown export format {fmt!r}") from exc
    return exporter(rows)
