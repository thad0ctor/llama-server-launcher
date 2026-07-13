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
    "n_prompt": "n_prompt",
    "n_gen": "n_gen",
    "n_depth": "n_depth",
}


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
        rows.append(ResultRow(columns=_order_columns(cols, _LLAMA_BENCH_PREFERRED)))
    return rows


def _extract_json_array(text: str):
    """Return the first top-level JSON array in ``text``, or ``None``."""
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else None
    except json.JSONDecodeError:
        pass
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, list) else None
    except json.JSONDecodeError:
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
    """Parse llama-sweep-bench's pipe-delimited table into rows.

    Each swept-parameter value in ``combo`` is prefixed as its own column so
    rows from different combinations remain distinguishable after merging.
    Lines that aren't table rows are ignored; the header row supplies column
    names and the ``|---|`` separator is skipped.
    """
    combo = combo or {}
    prefix_cols = {f"[{k}]": v for k, v in combo.items()}
    rows: list[ResultRow] = []
    header: list[str] | None = None
    for line in (stdout or "").splitlines():
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


def _csv_safe(value: str) -> str:
    text = "" if value is None else str(value)
    if text[:1] in _CSV_FORMULA_PREFIXES:
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
