import dataclasses
import subprocess
import sys
from pathlib import Path

from modules.build.build_persistence import BuildConfig, BuildConfigStore


# Repo root = three levels up from this file (tests/build/test_*.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_build_package_loads_build_tab_lazily():
    code = """
import importlib
import sys

pkg = importlib.import_module("modules.build")
assert "modules.build.build_tab" not in sys.modules
from modules.build import BuildTab
assert BuildTab.__name__ == "BuildTab"
assert "modules.build.build_tab" in sys.modules
"""
    # Pin cwd to the repo root so the child can import ``modules`` regardless
    # of where pytest was launched from. Previously this depended on the
    # parent process CWD, which broke when running from outside the repo.
    # Capture output so an import traceback in the child surfaces in the
    # failure message instead of silently producing an opaque ``rc != 0``.
    result = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        cwd=_REPO_ROOT,
        capture_output=True,
    )

    assert result.returncode == 0, (
        f"child exited rc={result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


def test_build_config_bool_fields_parse_persisted_strings():
    cfg = BuildConfig.from_json(
        "strings",
        {
            "git_pull_before_build": "false",
            "clean_build": "0",
        },
    )

    assert cfg.git_pull_before_build is False
    assert cfg.clean_build is False


def test_build_config_bool_fields_accept_truthy_strings():
    cfg = BuildConfig.from_json(
        "strings",
        {
            "git_pull_before_build": "yes",
            "clean_build": "on",
        },
    )

    assert cfg.git_pull_before_build is True
    assert cfg.clean_build is True


def _populated_build_config(name: str = "round-trip") -> BuildConfig:
    """Return a BuildConfig with a non-default value in every field.

    Used to lock down the to_json/from_json round-trip so a future schema
    addition fails this test loudly rather than silently dropping the new
    field on save.
    """
    return BuildConfig(
        name=name,
        backend="ik_llama",
        source_dir="/home/u/ik_llama.cpp",
        build_dir="build-release",
        git_ref="v1.2.3",
        git_pull_before_build=True,
        clean_build=False,
        jobs=12,
        cuda_archs="86-real;120a-real;120-real",
        env={
            "CC": "/usr/bin/gcc-13",
            "CXX": "/usr/bin/g++-13",
            "CUDACXX": "/usr/local/cuda-12.8/bin/nvcc",
            "CUDA_TOOLKIT_ROOT_DIR": "/usr/local/cuda-12.8",
        },
        flag_values={
            "GGML_CUDA": True,
            "GGML_LTO": True,
            "GGML_CUDA_DMMV_X": "32",
            "GGML_CUDA_KQUANTS_ITER": "2",
            "CMAKE_CUDA_FLAGS": "--use_fast_math -O3",
            "CMAKE_CUDA_ARCHITECTURES": "86;120",
            "GGML_CCACHE": True,
        },
        extra_cmake_args="-DGGML_OPENMP=ON",
        ui_state={
            "generator": "Ninja",
            "prefer_a": "1",
            "prefer_f": "0",
            "show_deprecated": "1",
        },
        created_at="2026-05-18T12:34:56Z",
        last_used_at="2026-06-03T20:00:00Z",
    )


def test_build_config_to_json_covers_every_field():
    """``to_json`` must serialize every dataclass field except ``name``
    (which is the dict key in the persisted layout). Catches the case
    where someone adds a field to the dataclass but forgets to ensure it
    flows through ``asdict``."""
    cfg = _populated_build_config()

    payload = cfg.to_json()

    serialized_keys = set(payload.keys())
    declared_keys = {f.name for f in dataclasses.fields(BuildConfig)} - {"name"}
    assert serialized_keys == declared_keys, (
        f"to_json drops {declared_keys - serialized_keys} or adds "
        f"{serialized_keys - declared_keys}"
    )


def test_build_config_round_trip_through_disk_preserves_every_field(tmp_path):
    """A populated BuildConfig must survive a full save → load cycle with
    no field lost or mutated. This is the load-bearing test for "save
    and load works with all settings": if anyone ever forgets to read a
    new field back in ``from_json`` it surfaces here, not in user reports."""
    store = BuildConfigStore(tmp_path)
    cfg_in = _populated_build_config("round-trip-A")

    store.save(cfg_in)
    # New store with the same file forces a real on-disk reload.
    reloaded_store = BuildConfigStore(tmp_path)
    cfg_out = reloaded_store.get("round-trip-A")

    assert cfg_out is not None, "saved config did not survive reload"
    # ``BuildConfigStore.save`` only fills the timestamps when they are
    # blank, so a pre-populated ``cfg_in`` should round-trip them
    # verbatim. Compare every field (including ``created_at`` /
    # ``last_used_at``) exactly so any regression that mutates a populated
    # timestamp on save surfaces here.
    for field in dataclasses.fields(BuildConfig):
        assert getattr(cfg_out, field.name) == getattr(cfg_in, field.name), (
            f"field {field.name!r} did not round-trip: "
            f"in={getattr(cfg_in, field.name)!r} out={getattr(cfg_out, field.name)!r}"
        )


def test_build_config_round_trip_two_configs_in_same_file(tmp_path):
    """Two presets in the same file must both round-trip — guards against
    code that accidentally treats configs as a single record."""
    store = BuildConfigStore(tmp_path)
    cfg_a = _populated_build_config("A")
    cfg_b = dataclasses.replace(
        _populated_build_config("B"),
        backend="llama.cpp",
        cuda_archs="89",
        flag_values={"GGML_CUDA": False},
    )
    store.save(cfg_a)
    store.save(cfg_b)

    reloaded = BuildConfigStore(tmp_path)
    out_a = reloaded.get("A")
    out_b = reloaded.get("B")

    assert out_a is not None and out_b is not None
    assert out_a.backend == "ik_llama"
    assert out_b.backend == "llama.cpp"
    assert out_a.flag_values["GGML_CUDA"] is True
    assert out_b.flag_values["GGML_CUDA"] is False
    assert out_a.cuda_archs == "86-real;120a-real;120-real"
    assert out_b.cuda_archs == "89"
