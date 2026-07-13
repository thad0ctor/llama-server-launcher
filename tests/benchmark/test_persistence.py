"""Tests for named benchmark-sweep persistence."""

from __future__ import annotations

import json

from modules.benchmark.bench_persistence import BenchConfig, BenchConfigStore


def _cfg(name="ngl sweep"):
    return BenchConfig(
        name=name,
        tool="llama-bench",
        backend="llama.cpp",
        build_root="/opt/llama.cpp",
        model_path="/m/model.gguf",
        axes={
            "n_gpu_layers": {"enabled": True, "mode": "range", "raw": "", "min": 0, "max": 33, "step": 11},
            "threads": {"enabled": True, "mode": "list", "raw": "8,16", "min": 0, "max": 0, "step": 1},
        },
        output_format="json",
        repetitions=5,
        extra_args="--numa distribute",
    )


def test_save_load_roundtrip(tmp_path):
    store = BenchConfigStore(tmp_path)
    assert store.save(_cfg())
    loaded = store.get("ngl sweep")
    assert loaded is not None
    assert loaded.tool == "llama-bench"
    assert loaded.model_path == "/m/model.gguf"
    assert loaded.axes["threads"]["raw"] == "8,16"
    assert loaded.repetitions == 5
    assert loaded.created_at  # stamped on save


def test_list_and_delete(tmp_path):
    store = BenchConfigStore(tmp_path)
    store.save(_cfg("a"))
    store.save(_cfg("b"))
    assert store.list_names() == ["a", "b"]
    assert store.delete("a")
    assert store.list_names() == ["b"]


def test_reject_blank_name(tmp_path):
    store = BenchConfigStore(tmp_path)
    assert not store.save(_cfg("   "))


def test_get_returns_independent_copy(tmp_path):
    store = BenchConfigStore(tmp_path)
    store.save(_cfg())
    a = store.get("ngl sweep")
    a.model_path = "/tampered"
    b = store.get("ngl sweep")
    assert b.model_path == "/m/model.gguf"


def test_malformed_tool_and_backend_coerced(tmp_path):
    cfg = BenchConfig(name="x", tool="bogus", backend="gpu")
    store = BenchConfigStore(tmp_path)
    store.save(cfg)
    loaded = store.get("x")
    assert loaded.tool == "llama-bench"
    assert loaded.backend == "llama.cpp"


def test_load_refuses_to_clobber_unreadable(tmp_path):
    # A non-object top level should refuse to load (and thus not be silently
    # overwritten on the next save).
    path = tmp_path / "bench_configs.json"
    path.write_text(json.dumps([1, 2, 3]))
    store = BenchConfigStore(tmp_path)
    assert store.list_names() == []
    assert not store.save(_cfg())  # refuses because load failed


def test_coerce_axes_drops_malformed(tmp_path):
    path = tmp_path / "bench_configs.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "configs": {
                    "c": {
                        "tool": "llama-bench",
                        "axes": {"threads": {"mode": "weird", "raw": "8"}, "bad": "not-a-dict"},
                    }
                },
            }
        )
    )
    store = BenchConfigStore(tmp_path)
    cfg = store.get("c")
    assert "bad" not in cfg.axes
    assert cfg.axes["threads"]["mode"] == "list"  # coerced from invalid
