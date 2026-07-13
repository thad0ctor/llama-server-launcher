"""Tests for benchmark result parsing and export."""

from __future__ import annotations

import csv
import io
import json

from modules.benchmark import results


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


def test_parse_llama_bench_json_basename_model():
    stdout = json.dumps([{"model_filename": "/models/sub/Qwen3-9B-Q6_K.gguf", "avg_ts": 1.0, "test": "pp"}])
    rows = results.parse_llama_bench_json(stdout)
    assert rows[0].get("model") == "Qwen3-9B-Q6_K.gguf"


def test_parse_llama_bench_json_tolerates_noise():
    stdout = 'loading...\n[{"test":"pp","avg_ts":10.0}]\ndone\n'
    rows = results.parse_llama_bench_json(stdout)
    assert len(rows) == 1


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
