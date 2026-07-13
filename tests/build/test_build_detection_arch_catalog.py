from modules.build import detection
from modules.build.cmake_flags import _validate_cuda_archs


def test_arch_catalog_names_current_embedded_and_dev_systems():
    expected = {
        "8.7": "Jetson AGX Orin",
        "11.0": "Jetson T5000",
        "12.1": "DGX Spark",
    }

    for cc, label in expected.items():
        arch = detection.known_arch_for(cc)
        assert arch is not None
        assert label in arch.name


# ─────────────────────────────────────────────────────────────────────────────
# CR-4467921108: ``_validate_cuda_archs`` must reject impossible
# ``-a`` / ``-f`` suffixes by consulting KNOWN_CUDA_ARCHS, not just
# the regex shape.
# ─────────────────────────────────────────────────────────────────────────────


def test_validate_cuda_archs_accepts_real_tokens():
    assert _validate_cuda_archs("86-real") is None
    assert _validate_cuda_archs("89") is None
    assert _validate_cuda_archs("75-virtual") is None
    assert _validate_cuda_archs("86-real;89-real;120-real") is None


def test_validate_cuda_archs_accepts_a_suffix_on_hopper_and_blackwell():
    # Hopper sm_90a exists.
    assert _validate_cuda_archs("90a-real") is None
    # Blackwell sm_120a exists.
    assert _validate_cuda_archs("120a-real") is None
    # Blackwell datacenter sm_100a, sm_103a, sm_110a all exist.
    assert _validate_cuda_archs("100a-real;103a-real;110a-real") is None


def test_validate_cuda_archs_accepts_f_suffix_only_on_blackwell():
    # Blackwell has -f in CUDA 13+.
    assert _validate_cuda_archs("120f-real") is None
    assert _validate_cuda_archs("100f-real") is None


def test_validate_cuda_archs_rejects_a_suffix_on_pre_hopper():
    # Ampere sm_86 has NO -a variant — CR-flagged example.
    msg = _validate_cuda_archs("86a-real")
    assert msg is not None
    assert "sm_86" in msg
    assert "no -a variant" in msg


def test_validate_cuda_archs_rejects_f_suffix_on_pre_blackwell():
    # Turing sm_75 has NO -f variant — CR-flagged example.
    msg = _validate_cuda_archs("75f-real")
    assert msg is not None
    assert "sm_75" in msg
    assert "no -f variant" in msg


def test_validate_cuda_archs_rejects_unknown_base_compute_capability():
    # sm_99 isn't in the catalog.
    msg = _validate_cuda_archs("99-real")
    assert msg is not None
    assert "sm_99" in msg
    assert "catalog" in msg


def test_validate_cuda_archs_rejects_empty_entry():
    # Trailing semicolon — preserves existing behaviour.
    assert _validate_cuda_archs("86-real;") == "empty entry in semicolon-separated list"
    assert _validate_cuda_archs("86-real;;89-real") == "empty entry in semicolon-separated list"


def test_validate_cuda_archs_rejects_garbage_token_shape():
    # Regex-level failure remains caught.
    msg = _validate_cuda_archs("abc-real")
    assert msg is not None
    assert "abc-real" in msg
