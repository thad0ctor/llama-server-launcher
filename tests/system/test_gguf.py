"""Regression tests for the simple GGUF parser in modules.system.

Covers:
    * parse_gguf_header_simple
    * calculate_total_gguf_size

A local byte-blob builder is defined here (scoped to this file, not exported)
so tests can assemble minimal valid / invalid GGUF files without shipping
real model binaries.
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import pytest

# Ensure the project root is importable regardless of how pytest is invoked.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from modules.system import (  # noqa: E402
    calculate_total_gguf_size,
    parse_gguf_header_simple,
)


# ---------------------------------------------------------------------------
# Local helper: synthetic GGUF byte-blob builder
# ---------------------------------------------------------------------------

# GGUF type IDs per spec
_T_UINT8 = 0
_T_INT8 = 1
_T_UINT16 = 2
_T_INT16 = 3
_T_UINT32 = 4
_T_INT32 = 5
_T_FLOAT32 = 6
_T_BOOL = 7
_T_STRING = 8
_T_ARRAY = 9
_T_UINT64 = 10
_T_INT64 = 11
_T_FLOAT64 = 12


def _pack_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _pack_value(value_type: int, value) -> bytes:
    if value_type == _T_UINT8:
        return struct.pack("<B", value)
    if value_type == _T_INT8:
        return struct.pack("<b", value)
    if value_type == _T_UINT16:
        return struct.pack("<H", value)
    if value_type == _T_INT16:
        return struct.pack("<h", value)
    if value_type == _T_UINT32:
        return struct.pack("<I", value)
    if value_type == _T_INT32:
        return struct.pack("<i", value)
    if value_type == _T_FLOAT32:
        return struct.pack("<f", value)
    if value_type == _T_BOOL:
        return struct.pack("<b", 1 if value else 0)
    if value_type == _T_UINT64:
        return struct.pack("<Q", value)
    if value_type == _T_INT64:
        return struct.pack("<q", value)
    if value_type == _T_FLOAT64:
        return struct.pack("<d", value)
    if value_type == _T_STRING:
        return _pack_string(value)
    raise ValueError(f"Unsupported scalar type {value_type}")


def _pack_metadata_entry(key: str, value_type: int, value) -> bytes:
    out = _pack_string(key)
    out += struct.pack("<I", value_type)
    if value_type == _T_ARRAY:
        # value must be a tuple of (inner_type, [items])
        inner_type, items = value
        out += struct.pack("<I", inner_type)
        out += struct.pack("<Q", len(items))
        for item in items:
            out += _pack_value(inner_type, item)
    else:
        out += _pack_value(value_type, value)
    return out


def build_gguf_blob(
    metadata: list[tuple[str, int, object]],
    *,
    magic: bytes = b"GGUF",
    version: int = 3,
    tensor_count: int = 0,
) -> bytes:
    """Assemble a minimal GGUF byte blob.

    Parameters
    ----------
    metadata
        List of (key, type_id, value) tuples. For arrays use type 9 and pass
        value=(inner_type, [items]).
    magic, version, tensor_count
        Header overrides (useful for error-path tests).
    """
    blob = bytearray()
    blob += magic
    blob += struct.pack("<I", version)
    blob += struct.pack("<Q", tensor_count)
    blob += struct.pack("<Q", len(metadata))
    for key, value_type, value in metadata:
        blob += _pack_metadata_entry(key, value_type, value)
    return bytes(blob)


# ---------------------------------------------------------------------------
# parse_gguf_header_simple
# ---------------------------------------------------------------------------


def test_parse_valid_minimal_header(tmp_path: Path) -> None:
    blob = build_gguf_blob([
        ("general.architecture", _T_STRING, "llama"),
        ("llama.block_count", _T_UINT32, 32),
    ])
    path = tmp_path / "tiny.gguf"
    path.write_bytes(blob)

    result = parse_gguf_header_simple(str(path))

    assert result["error"] is None
    assert result["architecture"] == "llama"
    assert result["n_layers"] == 32
    assert result["shard_count"] == 1
    assert result["file_size_bytes"] == len(blob)
    assert result["metadata"]["general.architecture"] == "llama"
    assert result["metadata"]["llama.block_count"] == 32


def test_parse_wrong_magic_reports_error(tmp_path: Path) -> None:
    blob = build_gguf_blob([], magic=b"XXXX")
    path = tmp_path / "bad_magic.gguf"
    path.write_bytes(blob)

    result = parse_gguf_header_simple(str(path))

    assert result["error"] is not None
    assert "GGUF" in result["error"]
    assert result["architecture"] == "unknown"


def test_parse_truncated_file_does_not_raise(tmp_path: Path) -> None:
    # Only the magic + partial version bytes — unpack of tensor/metadata
    # counts will blow up, but the parser must convert it into an error field
    # rather than crashing.
    path = tmp_path / "truncated.gguf"
    path.write_bytes(b"GGUF\x03")

    result = parse_gguf_header_simple(str(path))

    assert result["error"] is not None
    assert "Failed to parse GGUF header" in result["error"]


def test_parse_nonexistent_file_reports_error(tmp_path: Path) -> None:
    # Multi-part regex requires "-00001-of-00003.gguf"; this plain name hits
    # the single-file path and .stat() raises. Expect a clear error, not crash.
    missing = tmp_path / "does_not_exist.gguf"
    with pytest.raises((FileNotFoundError, OSError)):
        parse_gguf_header_simple(str(missing))


def test_parse_various_metadata_types(tmp_path: Path) -> None:
    """Exercise every scalar type the parser advertises in GGUF_TYPES."""
    metadata = [
        ("general.architecture", _T_STRING, "falcon"),
        ("falcon.block_count", _T_UINT32, 40),
        ("test.uint8", _T_UINT8, 200),
        ("test.int8", _T_INT8, -12),
        ("test.uint16", _T_UINT16, 65000),
        ("test.int16", _T_INT16, -1234),
        ("test.int32", _T_INT32, -123456),
        ("test.float32", _T_FLOAT32, 3.5),
        ("test.bool_true", _T_BOOL, True),
        ("test.bool_false", _T_BOOL, False),
        ("test.uint64", _T_UINT64, 2**40),
        ("test.int64", _T_INT64, -(2**40)),
        ("test.float64", _T_FLOAT64, 2.71828),
    ]
    path = tmp_path / "types.gguf"
    path.write_bytes(build_gguf_blob(metadata))

    # Parser exits early once it has arch+block_count, so architecture must be
    # discoverable even when early-exit fires before later metadata is read.
    result = parse_gguf_header_simple(str(path))

    assert result["error"] is None
    assert result["architecture"] == "falcon"
    assert result["n_layers"] == 40


def test_parse_exercises_all_scalar_types(tmp_path: Path) -> None:
    """Put the exotic scalar types BEFORE the architecture keys so they
    actually get decoded (the parser bails after finding arch + block_count).
    """
    metadata = [
        ("test.uint8", _T_UINT8, 200),
        ("test.int8", _T_INT8, -12),
        ("test.uint16", _T_UINT16, 65000),
        ("test.int16", _T_INT16, -1234),
        ("test.uint32", _T_UINT32, 4000000000),
        ("test.int32", _T_INT32, -123456),
        ("test.float32", _T_FLOAT32, 1.5),
        ("test.bool_true", _T_BOOL, True),
        ("test.bool_false", _T_BOOL, False),
        ("test.uint64", _T_UINT64, 2**40),
        ("test.int64", _T_INT64, -(2**40)),
        ("test.float64", _T_FLOAT64, 1.25),
        ("test.str", _T_STRING, "hello"),
        ("general.architecture", _T_STRING, "gemma"),
        ("gemma.block_count", _T_UINT32, 28),
    ]
    path = tmp_path / "all_types.gguf"
    path.write_bytes(build_gguf_blob(metadata))

    result = parse_gguf_header_simple(str(path))

    md = result["metadata"]
    assert result["error"] is None
    assert md["test.uint8"] == 200
    assert md["test.int8"] == -12
    assert md["test.uint16"] == 65000
    assert md["test.int16"] == -1234
    assert md["test.uint32"] == 4000000000
    assert md["test.int32"] == -123456
    assert abs(md["test.float32"] - 1.5) < 1e-6
    assert md["test.bool_true"] is True
    assert md["test.bool_false"] is False
    assert md["test.uint64"] == 2**40
    assert md["test.int64"] == -(2**40)
    assert abs(md["test.float64"] - 1.25) < 1e-12
    assert md["test.str"] == "hello"
    assert result["architecture"] == "gemma"
    assert result["n_layers"] == 28


def test_parse_skips_scalar_array_and_continues(tmp_path: Path) -> None:
    """Parser should skip over a numeric array and still find layer count."""
    metadata = [
        ("general.architecture", _T_STRING, "mistral"),
        ("tokenizer.ggml.scores", _T_ARRAY, (_T_FLOAT32, [0.1, 0.2, 0.3, 0.4])),
        ("mistral.block_count", _T_UINT32, 48),
    ]
    path = tmp_path / "arr_scalar.gguf"
    path.write_bytes(build_gguf_blob(metadata))

    result = parse_gguf_header_simple(str(path))

    assert result["error"] is None
    assert result["architecture"] == "mistral"
    assert result["n_layers"] == 48


def test_parse_skips_string_array_and_continues(tmp_path: Path) -> None:
    metadata = [
        ("general.architecture", _T_STRING, "qwen"),
        (
            "tokenizer.ggml.tokens",
            _T_ARRAY,
            (_T_STRING, ["<s>", "</s>", "<unk>", "<pad>"]),
        ),
        ("qwen.block_count", _T_UINT32, 24),
    ]
    path = tmp_path / "arr_string.gguf"
    path.write_bytes(build_gguf_blob(metadata))

    result = parse_gguf_header_simple(str(path))

    assert result["error"] is None
    assert result["architecture"] == "qwen"
    assert result["n_layers"] == 24


def test_parse_generic_block_count_pattern(tmp_path: Path) -> None:
    """Hits the `.block_count` suffix branch without a matching architecture
    prefix. We only put the block_count key (no architecture) so the generic
    branch fires — parser uses .lower() substring match.
    """
    metadata = [
        ("some.model.n_layers", _T_UINT32, 16),
    ]
    path = tmp_path / "generic.gguf"
    path.write_bytes(build_gguf_blob(metadata))

    result = parse_gguf_header_simple(str(path))

    assert result["error"] is None
    # architecture stays "unknown" but layer count should still be extracted
    assert result["n_layers"] == 16


def test_parse_estimates_layers_from_file_size_when_missing(tmp_path: Path) -> None:
    """Padding a file past 3GB in a test would be silly — verify the smallest
    bucket (< 3GB => 24 layers) fires when no block_count metadata exists.
    """
    metadata = [
        ("general.architecture", _T_STRING, "tinyllama"),
    ]
    path = tmp_path / "no_layers.gguf"
    path.write_bytes(build_gguf_blob(metadata))

    result = parse_gguf_header_simple(str(path))

    assert result["error"] is None
    assert result["architecture"] == "tinyllama"
    assert result["n_layers"] == 24
    assert "estimated" in result["message"]


def test_parse_version_and_tensor_counts(tmp_path: Path) -> None:
    """The parser stores version/tensor_count internally but doesn't expose
    them; we just ensure non-default values don't break parsing.
    """
    metadata = [
        ("general.architecture", _T_STRING, "phi"),
        ("phi.block_count", _T_UINT32, 32),
    ]
    path = tmp_path / "v2_with_tensors.gguf"
    path.write_bytes(build_gguf_blob(metadata, version=2, tensor_count=7))

    result = parse_gguf_header_simple(str(path))

    assert result["error"] is None
    assert result["architecture"] == "phi"


# ---------------------------------------------------------------------------
# calculate_total_gguf_size
# ---------------------------------------------------------------------------


def test_calculate_size_single_file(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    path.write_bytes(b"x" * 1024)

    total, count, shards = calculate_total_gguf_size(str(path))

    assert total == 1024
    assert count == 1
    assert shards == [path]


def test_calculate_size_multi_part_all_present(tmp_path: Path) -> None:
    sizes = [100, 200, 300]
    for i, size in enumerate(sizes, start=1):
        p = tmp_path / f"big-{i:05d}-of-00003.gguf"
        p.write_bytes(b"x" * size)

    shard1 = tmp_path / "big-00001-of-00003.gguf"
    total, count, shards = calculate_total_gguf_size(str(shard1))

    assert total == sum(sizes)
    assert count == 3
    assert [p.name for p in shards] == [
        "big-00001-of-00003.gguf",
        "big-00002-of-00003.gguf",
        "big-00003-of-00003.gguf",
    ]


def test_calculate_size_multi_part_with_missing_shards(tmp_path: Path) -> None:
    # Only 1 and 3 present; 2 is missing.
    (tmp_path / "m-00001-of-00003.gguf").write_bytes(b"a" * 50)
    (tmp_path / "m-00003-of-00003.gguf").write_bytes(b"c" * 70)

    shard_path = tmp_path / "m-00001-of-00003.gguf"
    total, count, shards = calculate_total_gguf_size(str(shard_path))

    assert total == 120  # only found shards
    assert count == 2
    assert len(shards) == 2


def test_calculate_size_multi_part_detected_from_later_shard(tmp_path: Path) -> None:
    """Pointing at shard 2 should still discover shards 1 and 3."""
    for i in range(1, 4):
        (tmp_path / f"nm-{i:05d}-of-00003.gguf").write_bytes(b"z" * (i * 10))

    middle = tmp_path / "nm-00002-of-00003.gguf"
    total, count, shards = calculate_total_gguf_size(str(middle))

    assert total == 10 + 20 + 30
    assert count == 3
    assert {p.name for p in shards} == {
        "nm-00001-of-00003.gguf",
        "nm-00002-of-00003.gguf",
        "nm-00003-of-00003.gguf",
    }


def test_calculate_size_shard_pattern_case_insensitive(tmp_path: Path) -> None:
    """Shard discovery must preserve the original casing of the `-of-`
    separator and `.gguf` extension so lookups succeed on case-sensitive
    filesystems when upstream filenames use uppercase (e.g. `-OF-`, `.GGUF`).
    """
    (tmp_path / "C-00001-OF-00002.gguf").write_bytes(b"1" * 11)
    (tmp_path / "C-00002-OF-00002.gguf").write_bytes(b"2" * 22)

    target = tmp_path / "C-00001-OF-00002.gguf"
    total, count, shards = calculate_total_gguf_size(str(target))

    # Ideally all shards get discovered and summed.
    assert total == 33
    assert count == 2
    assert len(shards) == 2


def test_calculate_size_single_file_missing_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope.gguf"
    with pytest.raises((FileNotFoundError, OSError)):
        calculate_total_gguf_size(str(missing))


# ---------------------------------------------------------------------------
# Integration: parse_gguf_header_simple on multi-part GGUF
# ---------------------------------------------------------------------------


def test_parse_multi_part_aggregates_sizes(tmp_path: Path) -> None:
    """parse_gguf_header_simple is passed shard 1; size should be total of
    all shards, and the first shard must contain a valid header.
    """
    # Shard 1 is the only one with header content; others are opaque bytes.
    header_blob = build_gguf_blob([
        ("general.architecture", _T_STRING, "llama"),
        ("llama.block_count", _T_UINT32, 80),
    ])
    shard1 = tmp_path / "mpmodel-00001-of-00002.gguf"
    shard2 = tmp_path / "mpmodel-00002-of-00002.gguf"
    shard1.write_bytes(header_blob)
    shard2.write_bytes(b"\0" * 1024)

    result = parse_gguf_header_simple(str(shard1))

    assert result["error"] is None
    assert result["shard_count"] == 2
    assert result["file_size_bytes"] == len(header_blob) + 1024
    assert result["architecture"] == "llama"
    assert result["n_layers"] == 80
    assert len(result["all_shards"]) == 2
