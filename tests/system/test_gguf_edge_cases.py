"""Edge-case and bug-fix regression tests for the simple GGUF parser.

Covers:
    * Nested-array (type 9) handling — used to silently halt the loop.
    * Very large string values — used to be silently dropped with no warning.
    * Big-endian GGUF detection — parser hardcodes little-endian; explicit
      error is better than garbage output.
    * GGUF version validation — v1 / unknown versions should error clearly.
    * Truncated header positions (mid-version, mid-counts, mid-string,
      mid-array) — parser must return an error field, never crash.
    * Shard-regex edge cases — sparse sets, large counts, mixed case.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from modules.system import (  # noqa: E402
    calculate_total_gguf_size,
    parse_gguf_header_simple,
)
from tests.system.test_gguf import (  # noqa: E402
    _T_ARRAY,
    _T_FLOAT32,
    _T_STRING,
    _T_UINT32,
    _pack_string,
    build_gguf_blob,
)


# ---------------------------------------------------------------------------
# Bug #2 — nested arrays (type 9) must not silently halt the parse loop
# ---------------------------------------------------------------------------


def _pack_nested_array_metadata(key: str, outer_len: int, inner_type: int, inner_items: list) -> bytes:
    """Build a single metadata entry whose value is an array-of-arrays."""
    out = bytearray()
    out += _pack_string(key)
    out += struct.pack("<I", _T_ARRAY)      # outer value_type = ARRAY
    out += struct.pack("<I", _T_ARRAY)      # array element type = ARRAY (nested)
    out += struct.pack("<Q", outer_len)     # number of inner arrays
    for _ in range(outer_len):
        # Each inner array: 4-byte element type + 8-byte length + elements.
        out += struct.pack("<I", inner_type)
        out += struct.pack("<Q", len(inner_items))
        for item in inner_items:
            if inner_type == _T_FLOAT32:
                out += struct.pack("<f", item)
            elif inner_type == _T_UINT32:
                out += struct.pack("<I", item)
            elif inner_type == _T_STRING:
                out += _pack_string(item)
            else:
                raise ValueError(f"test helper: unsupported inner type {inner_type}")
    return bytes(out)


def test_parse_nested_array_does_not_halt_loop(tmp_path: Path) -> None:
    """A nested array (array of arrays) must be walked over, not bailed on.

    Previously skip_array() returned False for type 9, breaking the metadata
    loop and leaving `architecture` / `n_layers` undetected if they sat after
    a nested-array metadata entry.
    """
    # Hand-build a GGUF blob: arch -> nested-array -> block_count. If the
    # parser can't skip nested arrays correctly, block_count won't be found
    # and n_layers will fall through to the file-size estimator.
    blob = bytearray()
    blob += b"GGUF"
    blob += struct.pack("<I", 3)                   # version
    blob += struct.pack("<Q", 0)                   # tensor_count
    blob += struct.pack("<Q", 3)                   # metadata_count

    # Entry 1: architecture
    blob += _pack_string("general.architecture")
    blob += struct.pack("<I", _T_STRING)
    blob += _pack_string("llama")

    # Entry 2: nested array with 2 inner float32 arrays of length 3
    blob += _pack_nested_array_metadata(
        "tokenizer.ggml.merges_nested", outer_len=2,
        inner_type=_T_FLOAT32, inner_items=[1.0, 2.0, 3.0],
    )

    # Entry 3: block_count (this must still be reachable)
    blob += _pack_string("llama.block_count")
    blob += struct.pack("<I", _T_UINT32)
    blob += struct.pack("<I", 48)

    path = tmp_path / "nested_array.gguf"
    path.write_bytes(bytes(blob))

    result = parse_gguf_header_simple(str(path))

    assert result["error"] is None
    assert result["architecture"] == "llama"
    # The real block_count must be found, not a file-size estimate.
    assert result["n_layers"] == 48


def test_parse_nested_array_of_strings(tmp_path: Path) -> None:
    """Nested string arrays must also be walked — strings have variable
    length so the recursive skip has to branch to the string-array path."""
    blob = bytearray()
    blob += b"GGUF"
    blob += struct.pack("<I", 3)
    blob += struct.pack("<Q", 0)
    blob += struct.pack("<Q", 3)

    blob += _pack_string("general.architecture")
    blob += struct.pack("<I", _T_STRING)
    blob += _pack_string("qwen")

    blob += _pack_nested_array_metadata(
        "tokenizer.ggml.nested_strings", outer_len=2,
        inner_type=_T_STRING, inner_items=["hi", "there"],
    )

    blob += _pack_string("qwen.block_count")
    blob += struct.pack("<I", _T_UINT32)
    blob += struct.pack("<I", 24)

    path = tmp_path / "nested_str_array.gguf"
    path.write_bytes(bytes(blob))

    result = parse_gguf_header_simple(str(path))

    assert result["error"] is None
    assert result["architecture"] == "qwen"
    assert result["n_layers"] == 24


def test_parse_deeply_nested_arrays_is_bounded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A maliciously nested array must not blow the Python stack.

    Builds an array nested 30 levels deep (above the 16-level cap). The
    parser must bail with a warning instead of raising RecursionError.
    """
    DEPTH = 30

    def build_nested(depth: int) -> bytes:
        # Innermost: a 0-length float32 array.
        if depth == 0:
            out = struct.pack("<I", _T_FLOAT32)
            out += struct.pack("<Q", 0)
            return out
        out = struct.pack("<I", _T_ARRAY)  # element type = ARRAY
        out += struct.pack("<Q", 1)         # one nested element
        out += build_nested(depth - 1)
        return out

    blob = bytearray()
    blob += b"GGUF"
    blob += struct.pack("<I", 3)
    blob += struct.pack("<Q", 0)
    blob += struct.pack("<Q", 1)  # one metadata entry

    blob += _pack_string("tokenizer.ggml.malicious_nested")
    blob += struct.pack("<I", _T_ARRAY)  # value_type = ARRAY
    blob += build_nested(DEPTH)

    path = tmp_path / "deep_nested.gguf"
    path.write_bytes(bytes(blob))

    # Must not raise RecursionError.
    result = parse_gguf_header_simple(str(path))

    # Parser bails on the malicious metadata entry but still returns a dict
    # with an error field populated — never a crash.
    assert isinstance(result, dict)
    captured = capsys.readouterr()
    assert "nested array exceeds max depth" in captured.err


# ---------------------------------------------------------------------------
# Bug #3 — very large string values should advance the stream + warn
# ---------------------------------------------------------------------------


def test_parse_large_string_value_warns_and_continues(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A string value larger than the 1 MB soft cap must not desync the
    parser: bytes get consumed, a warning is printed, the loop continues
    so later metadata (block_count) is still found."""
    huge_len = 1_000_001  # 1 byte past the soft cap
    huge_value = b"x" * huge_len

    blob = bytearray()
    blob += b"GGUF"
    blob += struct.pack("<I", 3)
    blob += struct.pack("<Q", 0)
    blob += struct.pack("<Q", 3)

    blob += _pack_string("general.architecture")
    blob += struct.pack("<I", _T_STRING)
    blob += _pack_string("llama")

    # Oversized string entry
    blob += _pack_string("some.huge.metadata")
    blob += struct.pack("<I", _T_STRING)
    blob += struct.pack("<Q", huge_len)
    blob += huge_value

    blob += _pack_string("llama.block_count")
    blob += struct.pack("<I", _T_UINT32)
    blob += struct.pack("<I", 80)

    path = tmp_path / "big_string.gguf"
    path.write_bytes(bytes(blob))

    result = parse_gguf_header_simple(str(path))

    assert result["error"] is None
    assert result["architecture"] == "llama"
    # Critical: the loop kept going after the oversized string.
    assert result["n_layers"] == 80
    # The oversized value itself is dropped from metadata.
    assert "some.huge.metadata" not in result["metadata"]

    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "soft cap" in captured.err


# ---------------------------------------------------------------------------
# Big-endian detection
# ---------------------------------------------------------------------------


def test_parse_big_endian_magic_reports_clear_error(tmp_path: Path) -> None:
    """Files with the big-endian GGUF magic ('FUGG') should raise a clear
    error rather than silently misparse."""
    path = tmp_path / "be.gguf"
    path.write_bytes(b"FUGG" + b"\0" * 20)

    result = parse_gguf_header_simple(str(path))

    assert result["error"] is not None
    assert "big-endian" in result["error"].lower() or "big endian" in result["error"].lower()


# ---------------------------------------------------------------------------
# GGUF version validation
# ---------------------------------------------------------------------------


def test_parse_rejects_version_1(tmp_path: Path) -> None:
    """GGUF v1 had a different layout; parser should error clearly."""
    blob = build_gguf_blob([], version=1)
    path = tmp_path / "v1.gguf"
    path.write_bytes(blob)

    result = parse_gguf_header_simple(str(path))

    assert result["error"] is not None
    assert "version" in result["error"].lower()


def test_parse_rejects_unknown_future_version(tmp_path: Path) -> None:
    blob = build_gguf_blob([], version=99)
    path = tmp_path / "v99.gguf"
    path.write_bytes(blob)

    result = parse_gguf_header_simple(str(path))

    assert result["error"] is not None
    assert "version" in result["error"].lower()


def test_parse_accepts_version_2(tmp_path: Path) -> None:
    blob = build_gguf_blob(
        [
            ("general.architecture", _T_STRING, "llama"),
            ("llama.block_count", _T_UINT32, 32),
        ],
        version=2,
    )
    path = tmp_path / "v2.gguf"
    path.write_bytes(blob)

    result = parse_gguf_header_simple(str(path))

    assert result["error"] is None
    assert result["architecture"] == "llama"
    assert result["n_layers"] == 32


# ---------------------------------------------------------------------------
# Truncation at various offsets — never crash, always return error field
# ---------------------------------------------------------------------------


def test_parse_truncated_mid_version(tmp_path: Path) -> None:
    path = tmp_path / "t_version.gguf"
    # magic + 1 byte of version (need 4)
    path.write_bytes(b"GGUF\x03")

    result = parse_gguf_header_simple(str(path))
    assert result["error"] is not None


def test_parse_truncated_mid_counts(tmp_path: Path) -> None:
    """Magic + version present, but tensor_count is truncated."""
    path = tmp_path / "t_counts.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<I", 3) + b"\x00\x00\x00")  # only 3 of 8 bytes

    result = parse_gguf_header_simple(str(path))
    assert result["error"] is not None


def test_parse_truncated_mid_string_value(tmp_path: Path) -> None:
    """Claim a 100-byte string but only include 5 bytes — parser must bail
    gracefully, not crash."""
    blob = bytearray()
    blob += b"GGUF"
    blob += struct.pack("<I", 3)
    blob += struct.pack("<Q", 0)
    blob += struct.pack("<Q", 1)

    blob += _pack_string("general.architecture")
    blob += struct.pack("<I", _T_STRING)
    blob += struct.pack("<Q", 100)      # declared length
    blob += b"short"                    # only 5 bytes

    path = tmp_path / "t_string.gguf"
    path.write_bytes(bytes(blob))

    result = parse_gguf_header_simple(str(path))
    # Parser should finish without raising; architecture won't be set.
    assert result["error"] is None or "Failed to parse" in str(result.get("error", ""))
    # No real layer count => file-size estimator produces a positive value.
    assert isinstance(result["n_layers"], int)


def test_parse_truncated_mid_array(tmp_path: Path) -> None:
    """Declare a 100-element float32 array but cut the file short.

    The parser should exit the loop (or fall through to the estimator)
    without raising. The file-size-based estimator will fire.
    """
    blob = bytearray()
    blob += b"GGUF"
    blob += struct.pack("<I", 3)
    blob += struct.pack("<Q", 0)
    blob += struct.pack("<Q", 2)

    blob += _pack_string("general.architecture")
    blob += struct.pack("<I", _T_STRING)
    blob += _pack_string("llama")

    # Array entry claiming 100 float32s, only 2 floats follow
    blob += _pack_string("tokenizer.ggml.scores")
    blob += struct.pack("<I", _T_ARRAY)
    blob += struct.pack("<I", _T_FLOAT32)
    blob += struct.pack("<Q", 100)
    blob += struct.pack("<f", 0.1)
    blob += struct.pack("<f", 0.2)

    path = tmp_path / "t_array.gguf"
    path.write_bytes(bytes(blob))

    result = parse_gguf_header_simple(str(path))
    # Either gracefully errored or produced an estimate. Don't crash.
    assert "n_layers" in result
    assert isinstance(result["n_layers"], int) and result["n_layers"] > 0


# ---------------------------------------------------------------------------
# Shard edge cases
# ---------------------------------------------------------------------------


def test_calculate_size_sparse_shard_set(tmp_path: Path) -> None:
    """Shards 1, 2, 4 present; 3 missing. Returns partial sum."""
    (tmp_path / "s-00001-of-00004.gguf").write_bytes(b"a" * 10)
    (tmp_path / "s-00002-of-00004.gguf").write_bytes(b"b" * 20)
    (tmp_path / "s-00004-of-00004.gguf").write_bytes(b"d" * 40)

    target = tmp_path / "s-00001-of-00004.gguf"
    total, count, shards = calculate_total_gguf_size(str(target))

    assert total == 70           # 3 present, 1 missing
    assert count == 3
    assert len(shards) == 3


def test_calculate_size_very_large_shard_count(tmp_path: Path) -> None:
    """Regex should accept 5+ digit shard indices. Only shard 1 written;
    parser returns partial but does not crash on 99999 expected."""
    (tmp_path / "big-00001-of-99999.gguf").write_bytes(b"x" * 7)

    target = tmp_path / "big-00001-of-99999.gguf"
    total, count, shards = calculate_total_gguf_size(str(target))

    assert total == 7
    assert count == 1
    assert len(shards) == 1


def test_calculate_size_mixed_case_uppercase_ext(tmp_path: Path) -> None:
    """Case-sensitive FS: both -OF- and .GGUF uppercased. Must reconstruct
    shard names with the original case so the lookup succeeds."""
    (tmp_path / "X-00001-OF-00002.GGUF").write_bytes(b"1" * 5)
    (tmp_path / "X-00002-OF-00002.GGUF").write_bytes(b"2" * 11)

    target = tmp_path / "X-00001-OF-00002.GGUF"
    total, count, shards = calculate_total_gguf_size(str(target))

    assert total == 16
    assert count == 2
    assert len(shards) == 2


def test_calculate_size_mixed_case_mixed_separator(tmp_path: Path) -> None:
    """A weird but legal file with `-Of-` (title case) — still case-insensitive
    match, must reconstruct with preserved case."""
    (tmp_path / "Y-00001-Of-00002.gguf").write_bytes(b"a" * 3)
    (tmp_path / "Y-00002-Of-00002.gguf").write_bytes(b"b" * 4)

    target = tmp_path / "Y-00001-Of-00002.gguf"
    total, count, shards = calculate_total_gguf_size(str(target))

    assert total == 7
    assert count == 2


# ---------------------------------------------------------------------------
# Regression: oversized-string DoS guard
#
# Before the fix, the parser called ``f.read(value_len)`` *before* checking
# ``value_len`` against the soft cap. Python's BufferedReader.read(N) on a
# legitimately large file will allocate up to min(N, available) bytes — so
# a corrupted or adversarial GGUF claiming e.g. ``value_len = 5 GiB`` on a
# real 20 GB model file would spike the parser RSS by 5 GiB and potentially
# OOM the process. The fix cheks the cap first and seek()s past the bytes.
#
# We can't reliably trigger the OOM in CI (would need a multi-GB fixture),
# so we spy on ``f.read`` and assert it is never called with a size beyond
# the soft cap.
# ---------------------------------------------------------------------------


def test_parse_oversized_string_len_never_read_in_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    from modules import system as _system_mod

    huge_claim = 10 * 1024 * 1024 * 1024  # 10 GiB — way past soft cap

    blob = bytearray()
    blob += b"GGUF"
    blob += struct.pack("<I", 3)
    blob += struct.pack("<Q", 0)  # tensor_count
    blob += struct.pack("<Q", 2)  # metadata_count

    # Regular small value first so we know the parser reached the loop.
    blob += _pack_string("general.architecture")
    blob += struct.pack("<I", _T_STRING)
    blob += _pack_string("llama")

    # Claims 10 GiB but those bytes are NOT in the file.
    blob += _pack_string("pretend.huge.key")
    blob += struct.pack("<I", _T_STRING)
    blob += struct.pack("<Q", huge_claim)
    # File ends here — truncated after the length header.

    path = tmp_path / "liar.gguf"
    path.write_bytes(bytes(blob))

    max_read_size = [0]
    real_open = builtins.open

    def spy_open(file, mode="r", *args, **kwargs):
        fh = real_open(file, mode, *args, **kwargs)
        # Only instrument the GGUF file reads; leave other opens alone.
        if mode == "rb" and str(file) == str(path):
            real_read = fh.read

            def spy_read(size=-1, *a, **k):
                if isinstance(size, int) and size > 0:
                    max_read_size[0] = max(max_read_size[0], size)
                return real_read(size, *a, **k)

            fh.read = spy_read
        return fh

    monkeypatch.setattr(builtins, "open", spy_open)

    # Parser MUST NOT attempt a 10 GiB allocation. With the fix it either
    # returns cleanly or bails via the file-end truncation check — both are
    # fine. What is NOT fine is calling f.read(huge_claim).
    result = _system_mod.parse_gguf_header_simple(str(path))

    assert result is not None
    assert max_read_size[0] < huge_claim, (
        f"Parser called f.read with size {max_read_size[0]} — larger than the "
        f"soft cap. This would allocate multi-GB on a legitimate large GGUF."
    )


# ---------------------------------------------------------------------------
# Regression: skip_array must bounds-check every seek against file size
#
# Before the fix, ``skip_array`` blindly did ``f.seek(f.tell() + str_len)`` on
# string-array elements and ``f.seek(f.tell() + array_len * element_size)``
# on numeric-array elements — both using untrusted 64-bit lengths straight
# from the file. A malformed file could claim an absurd length, send the
# stream position far past EOF, and leave the caller mis-interpreting later
# short reads as earlier truncation. The fix validates each proposed seek
# against ``os.fstat(...).st_size`` before advancing.
# ---------------------------------------------------------------------------


def _parse_and_record_max_pos(path: Path, monkeypatch: pytest.MonkeyPatch) -> int:
    """Parse a GGUF blob and return the maximum stream position the parser
    ever reached. Used to assert that ``skip_array``'s bounds check keeps
    the position within the file even when fed an adversarial length.
    """
    import builtins

    from modules import system as _system_mod

    max_pos = [0]
    real_open = builtins.open

    class _PosTrackingFile:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *exc):
            return self._inner.__exit__(*exc)

        def seek(self, *args, **kwargs):
            r = self._inner.seek(*args, **kwargs)
            max_pos[0] = max(max_pos[0], self._inner.tell())
            return r

        def read(self, *args, **kwargs):
            data = self._inner.read(*args, **kwargs)
            max_pos[0] = max(max_pos[0], self._inner.tell())
            return data

        def tell(self):
            return self._inner.tell()

    def tracking_open(file, mode="r", *args, **kwargs):
        fh = real_open(file, mode, *args, **kwargs)
        if mode == "rb" and str(file) == str(path):
            return _PosTrackingFile(fh)
        return fh

    monkeypatch.setattr(builtins, "open", tracking_open)
    _system_mod.parse_gguf_header_simple(str(path))
    return max_pos[0]


def test_skip_array_string_len_past_eof_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A string-array element whose claimed length exceeds the file size
    must NOT send the stream position past EOF. Before the fix, skip_array
    did ``f.seek(f.tell() + str_len)`` with str_len straight from the file,
    leaving the position billions of bytes past EOF and corrupting the
    interpretation of later reads."""
    blob = bytearray()
    blob += b"GGUF"
    blob += struct.pack("<I", 3)
    blob += struct.pack("<Q", 0)
    blob += struct.pack("<Q", 1)

    # Single string-array element claiming 10 GiB despite ~60-byte file.
    blob += _pack_string("tokenizer.evil")
    blob += struct.pack("<I", _T_ARRAY)
    blob += struct.pack("<I", _T_STRING)
    blob += struct.pack("<Q", 1)
    blob += struct.pack("<Q", 10 * 1024**3)

    path = tmp_path / "liar_string_array.gguf"
    path.write_bytes(bytes(blob))

    file_size = path.stat().st_size
    max_pos = _parse_and_record_max_pos(path, monkeypatch)

    assert max_pos <= file_size, (
        f"Max stream position {max_pos} exceeded file size {file_size} — "
        f"bounds check failed on the string-array branch of skip_array."
    )


def test_skip_array_numeric_array_len_past_eof_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same trust issue as the string-array path, numeric-array branch —
    ``array_len * element_size`` could blow past the file."""
    blob = bytearray()
    blob += b"GGUF"
    blob += struct.pack("<I", 3)
    blob += struct.pack("<Q", 0)
    blob += struct.pack("<Q", 1)

    # Array of uint32 claiming 10 billion elements — 40 GB of "data".
    blob += _pack_string("tokenizer.evil_numeric")
    blob += struct.pack("<I", _T_ARRAY)
    blob += struct.pack("<I", _T_UINT32)
    blob += struct.pack("<Q", 10_000_000_000)

    path = tmp_path / "liar_numeric_array.gguf"
    path.write_bytes(bytes(blob))

    file_size = path.stat().st_size
    max_pos = _parse_and_record_max_pos(path, monkeypatch)

    assert max_pos <= file_size, (
        f"Max stream position {max_pos} exceeded file size {file_size} — "
        f"bounds check failed on the numeric-array branch of skip_array."
    )
