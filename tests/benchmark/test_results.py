"""Tests for benchmark result parsing and export."""

from __future__ import annotations

import csv
import io
import json

from modules.benchmark import results


def test_main_gpu_column_present_when_emitted():
    # -mg is a sweepable lever; llama-bench emits main_gpu in JSON, so the grid
    # must surface it (otherwise -mg sweep rows are indistinguishable).
    stdout = json.dumps([{"model_type": "Q", "main_gpu": 1, "test": "pp512", "avg_ts": 10.0}])
    rows = results.parse_llama_bench_json(stdout)
    assert rows[0].get("main_gpu") == "1"


def test_csv_does_not_escape_negative_or_plain_numbers():
    # Regression: formula-injection escaping must not corrupt legitimate
    # signed numbers like -1 (``-ngl -1`` = "all layers") or main_gpu columns.
    rows = [results.ResultRow(columns={"[n_gpu_layers]": "-1", "t/s": "-5", "x": "3.14", "y": "+2"})]
    parsed = list(csv.reader(io.StringIO(results.to_csv(rows))))
    assert parsed[1] == ["-1", "-5", "3.14", "+2"]


def test_csv_still_escapes_formulas_and_nonnumeric_dash():
    rows = [results.ResultRow(columns={"a": "=CMD()", "b": "-cmd", "c": "@x", "d": "+SUM(1)"})]
    parsed = list(csv.reader(io.StringIO(results.to_csv(rows))))
    assert parsed[1] == ["'=CMD()", "'-cmd", "'@x", "'+SUM(1)"]


def test_parse_llama_bench_json_basic():
    stdout = json.dumps(
        [
            {"model_type": "Qwen", "n_gpu_layers": 0, "n_threads": 8, "test": "pp512", "avg_ts": 142.3},
            {"model_type": "Qwen", "n_gpu_layers": 0, "n_threads": 16, "test": "tg128", "avg_ts": 44.1},
        ]
    )
    rows = results.parse_llama_bench_json(stdout)
    assert len(rows) == 2
    assert rows[0].get("model") == "Qwen"
    assert rows[0].get("test") == "pp512"
    assert rows[0].get("t/s") == "142.30"
    assert rows[1].get("n_threads") == "16"


def test_parse_llama_bench_json_surfaces_unmapped_scalar_fields():
    # Custom-flag / option sweeps emit JSON keys the curated map drops (e.g.
    # use_mmap, split_mode); they must appear as columns so rows stay distinct.
    # Noisy metadata (build_commit, cpu_info, ...) and sample arrays must NOT.
    stdout = json.dumps(
        [
            {
                "model_type": "Q",
                "test": "pp512",
                "avg_ts": 10.0,
                "use_mmap": False,
                "split_mode": "layer",
                "no_kv_offload": True,
                "build_commit": "abc123",
                "cpu_info": "some cpu",
                "samples_ns": [1, 2, 3],
            }
        ]
    )
    rows = results.parse_llama_bench_json(stdout)
    assert rows[0].get("use_mmap") == "0"
    assert rows[0].get("split_mode") == "layer"
    assert rows[0].get("no_kv_offload") == "1"
    cols = rows[0].columns
    assert "build_commit" not in cols
    assert "cpu_info" not in cols
    assert "samples_ns" not in cols  # non-scalar array is skipped


def test_parse_llama_bench_json_basename_model():
    stdout = json.dumps([{"model_filename": "/models/sub/Qwen3-9B-Q6_K.gguf", "avg_ts": 1.0, "test": "pp"}])
    rows = results.parse_llama_bench_json(stdout)
    assert rows[0].get("model") == "Qwen3-9B-Q6_K.gguf"


def test_parse_llama_bench_json_tolerates_noise():
    stdout = 'loading...\n[{"test":"pp","avg_ts":10.0}]\ndone\n'
    rows = results.parse_llama_bench_json(stdout)
    assert len(rows) == 1


def test_parse_llama_bench_json_bracketed_banner_before_array():
    # A log line containing brackets (e.g. "[INFO] ...") before the JSON array
    # must not defeat extraction: first-'[' .. last-']' would span the banner
    # and fail to parse. The scanner tries each '[' and returns the first array.
    stdout = '[INFO] loading model\n[{"test":"pp","avg_ts":1.0}]\n'
    rows = results.parse_llama_bench_json(stdout)
    assert len(rows) == 1
    assert rows[0].get("test") == "pp"


def test_parse_llama_bench_json_empty():
    assert results.parse_llama_bench_json("") == []
    assert results.parse_llama_bench_json("not json") == []


def test_parse_sweep_bench_table_with_combo():
    table = (
        "some log line\n"
        "|   PP |   TG | N_KV | S_PP t/s | S_TG t/s |\n"
        "|------|------|------|----------|----------|\n"
        "|  512 |  128 |    0 |  1024.00 |    64.00 |\n"
        "|  512 |  128 |  512 |   980.00 |    60.00 |\n"
    )
    rows = results.parse_sweep_bench_table(table, {"n_gpu_layers": "10"})
    assert len(rows) == 2
    assert rows[0].get("[n_gpu_layers]") == "10"
    assert rows[0].get("N_KV") == "0"
    assert rows[1].get("S_TG t/s") == "60.00"


def test_parse_sweep_bench_whitespace_table():
    # Some ik_llama builds print a WHITESPACE-delimited table (no pipes). It must
    # parse into the same shape as the markdown table, splitting on 2+ spaces so
    # multi-word columns like "S_PP t/s" stay intact, and prefixing the combo.
    table = (
        "loading model ...\n"
        "   PP    TG   N_KV    S_PP t/s    S_TG t/s\n"
        "  ----  ----  ----   --------    --------\n"
        "  512   128      0     1024.00       64.00\n"
        "  512   128    512      980.00       60.00\n"
        "done\n"
    )
    rows = results.parse_sweep_bench_table(table, {"n_gpu_layers": "10"})
    assert len(rows) == 2
    assert rows[0].get("[n_gpu_layers]") == "10"
    assert rows[0].get("N_KV") == "0"
    assert rows[0].get("S_PP t/s") == "1024.00"
    assert rows[1].get("S_TG t/s") == "60.00"


def test_collect_columns_union_first_seen():
    rows = [
        results.ResultRow(columns={"a": "1", "b": "2"}),
        results.ResultRow(columns={"b": "3", "c": "4"}),
    ]
    assert results.collect_columns(rows) == ["a", "b", "c"]


def test_export_csv_roundtrip():
    rows = [results.ResultRow(columns={"model": "Q", "t/s": "10.0"})]
    text = results.export(rows, "csv")
    parsed = list(csv.reader(io.StringIO(text)))
    assert parsed[0] == ["model", "t/s"]
    assert parsed[1] == ["Q", "10.0"]


def test_export_json_and_markdown():
    rows = [results.ResultRow(columns={"model": "Q", "t/s": "10.0"})]
    assert json.loads(results.export(rows, "json")) == [{"model": "Q", "t/s": "10.0"}]
    md = results.export(rows, "markdown")
    assert "| model | t/s |" in md
    assert "| Q | 10.0 |" in md


def test_csv_neutralises_formula_injection():
    # Non-numeric formula-led cells must be apostrophe-prefixed. Bare signed
    # numbers (e.g. "-2", "+1") are covered separately and must NOT be escaped.
    rows = [results.ResultRow(columns={"model": "=CMD()", "note": "+SUM(A1)", "x": "@SUM", "y": "-2+3"})]
    text = results.export(rows, "csv")
    parsed = list(csv.reader(io.StringIO(text)))
    # csv.reader strips the field back to the stored text, so the apostrophe
    # prefix is preserved verbatim in the parsed cell.
    assert parsed[1][0] == "'=CMD()"
    assert parsed[1][1] == "'+SUM(A1)"
    assert parsed[1][2] == "'@SUM"
    assert parsed[1][3] == "'-2+3"


def test_csv_escapes_dangerous_header():
    rows = [results.ResultRow(columns={"=evil": "1"})]
    text = results.export(rows, "csv")
    parsed = list(csv.reader(io.StringIO(text)))
    assert parsed[0][0] == "'=evil"


def test_markdown_escapes_pipe_and_newline():
    rows = [results.ResultRow(columns={"model": "a|b", "note": "line1\nline2"})]
    md = results.export(rows, "markdown")
    # The stray pipe is escaped and the newline folded to a space, so the row
    # stays a single physical line with the right column count.
    assert "a\\|b" in md
    assert "line1 line2" in md
    body = [ln for ln in md.splitlines() if ln.startswith("| a")]
    assert len(body) == 1


def test_markdown_escapes_pipe_in_header():
    rows = [results.ResultRow(columns={"a|b": "1"})]
    md = results.export(rows, "markdown")
    assert "| a\\|b |" in md
