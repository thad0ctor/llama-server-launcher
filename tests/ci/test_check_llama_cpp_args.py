from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_llama_cpp_args.py"
SPEC = importlib.util.spec_from_file_location("check_llama_cpp_args", SCRIPT_PATH)
check_llama_cpp_args = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = check_llama_cpp_args
SPEC.loader.exec_module(check_llama_cpp_args)


def _synthetic_help_for(flag_inputs=None):
    flag_inputs = flag_inputs or check_llama_cpp_args.LLAMA_CPP_FLAG_INPUTS
    lines = []
    for flag, spec in sorted(flag_inputs.items()):
        suffix = " VALUE" if spec.takes_value else ""
        lines.append(f"  {flag}{suffix}    synthetic help for {spec.intended_input}")
    return "\n".join(lines)


def _source_paths_for(backend):
    root = Path(__file__).resolve().parents[2]
    paths = [root / "modules" / "launch.py", root / "modules" / "spec_launch.py"]
    if backend == "ik_llama":
        paths.append(root / "modules" / "ik_llama.py")
    return paths


def test_extract_flags_handles_long_and_short_tokens():
    text = """
      -m, --model FNAME
      --threads N
      --ctx-size N
      default value: -1
      path: ./not-a-flag
    """

    assert check_llama_cpp_args.extract_flags(text) == {"-m", "--model", "--threads", "--ctx-size"}


def test_every_tracked_llama_cpp_flag_has_intended_input_metadata():
    failures = check_llama_cpp_args.validate_flag_input_manifest()

    assert failures == []
    assert set(check_llama_cpp_args.LLAMA_CPP_FLAG_INPUTS) == set(check_llama_cpp_args.LLAMA_CPP_FLAGS)
    assert all(
        spec.kind and isinstance(spec.takes_value, bool) and spec.intended_input
        for spec in check_llama_cpp_args.LLAMA_CPP_FLAG_INPUTS.values()
    )


def test_every_tracked_ik_llama_flag_has_intended_input_metadata():
    failures = check_llama_cpp_args.validate_flag_input_manifest(
        check_llama_cpp_args.IK_LLAMA_FLAGS,
        check_llama_cpp_args.IK_LLAMA_FLAG_INPUTS,
        backend_name="ik_llama",
    )

    assert failures == []
    assert set(check_llama_cpp_args.IK_LLAMA_FLAG_INPUTS) == set(check_llama_cpp_args.IK_LLAMA_FLAGS)
    assert all(
        spec.kind and isinstance(spec.takes_value, bool) and spec.intended_input
        for spec in check_llama_cpp_args.IK_LLAMA_FLAG_INPUTS.values()
    )


def test_help_line_takes_value_detects_supported_shapes():
    assert check_llama_cpp_args.help_line_takes_value("  --threads N    thread count", "--threads")
    assert check_llama_cpp_args.help_line_takes_value("  --flash-attn on|off|auto", "--flash-attn")
    assert check_llama_cpp_args.help_line_takes_value("  --model=<path>", "--model")
    assert not check_llama_cpp_args.help_line_takes_value("  --jinja       enable jinja", "--jinja")
    assert not check_llama_cpp_args.help_line_takes_value("  --no-mmap,    disable mmap", "--no-mmap")


def test_alias_group_takes_value_when_value_marker_is_on_an_alias():
    line = "  --spec-draft-device, -devd, --device-draft <dev1,dev2,..>"

    assert not check_llama_cpp_args.help_line_takes_value(line, "--spec-draft-device")
    assert check_llama_cpp_args.help_alias_group_takes_value(line)


def test_audit_reports_missing_upstream_flags_without_source_scan():
    failures = check_llama_cpp_args.audit(
        help_text="-m MODEL, --threads N\n--ctx-size N\n",
        source_paths=[],
        expected_flags={"-m", "--threads", "--ctx-size", "--flash-attn"},
        known_non_llama_cpp_flags=set(),
        flag_inputs={
            "-m": check_llama_cpp_args.FlagInput("path", True, "model path"),
            "--threads": check_llama_cpp_args.FlagInput("integer", True, "thread count"),
            "--ctx-size": check_llama_cpp_args.FlagInput("integer", True, "context size"),
            "--flash-attn": check_llama_cpp_args.FlagInput("enum", True, "flash attention mode"),
        },
    )

    assert len(failures) == 1
    assert "--flash-attn" in failures[0]


def test_audit_reports_input_shape_mismatch_without_source_scan():
    failures = check_llama_cpp_args.audit(
        help_text="--threads       thread count\n--jinja VALUE    unexpected value\n",
        source_paths=[],
        expected_flags={"--threads", "--jinja"},
        known_non_llama_cpp_flags=set(),
        flag_inputs={
            "--threads": check_llama_cpp_args.FlagInput("integer", True, "CPU generation thread count"),
            "--jinja": check_llama_cpp_args.FlagInput("switch", False, "enable Jinja rendering"),
        },
    )

    assert len(failures) == 1
    assert "input shapes diverge" in failures[0]
    assert "--threads" in failures[0]
    assert "--jinja" in failures[0]


def test_audit_requires_new_source_flags_to_be_categorized(tmp_path):
    source = tmp_path / "launch_like.py"
    source.write_text(
        "def emit(cmd):\n"
        "    cmd.extend(['--threads', '4'])\n"
        "    cmd.extend(['--new-upstream-flag', 'on'])\n",
        encoding="utf-8",
    )

    failures = check_llama_cpp_args.audit(
        help_text="--threads N\n--new-upstream-flag on\n",
        source_paths=[source],
        expected_flags={"--threads"},
        known_non_llama_cpp_flags=set(),
        flag_inputs={"--threads": check_llama_cpp_args.FlagInput("integer", True, "thread count")},
    )

    assert len(failures) == 1
    assert "--new-upstream-flag" in failures[0]


def test_source_scan_ignores_labels_and_help_text(tmp_path):
    source = tmp_path / "launch_like.py"
    source.write_text(
        "def emit(cmd):\n"
        "    label = 'Removed flag --old-label-only'\n"
        "    cmd.extend(['--threads', '4'])\n",
        encoding="utf-8",
    )

    flags = check_llama_cpp_args.extract_source_flags([source])

    assert flags == {"--threads"}


def test_source_scan_follows_loop_flag_values_used_by_cmd_extend(tmp_path):
    source = tmp_path / "launch_like.py"
    source.write_text(
        "def emit(cmd):\n"
        "    for var_name, flag in [('threads', '--threads'), ('ctx', '--ctx-size')]:\n"
        "        value = getattr(config, var_name, '')\n"
        "        if value:\n"
        "            cmd.extend([flag, value])\n",
        encoding="utf-8",
    )

    flags = check_llama_cpp_args.extract_source_flags([source])

    assert flags == {"--threads", "--ctx-size"}


def test_audit_accepts_categorized_non_llama_cpp_source_flags(tmp_path):
    source = tmp_path / "launch_like.py"
    source.write_text(
        "def emit(cmd):\n"
        "    cmd.extend(['--threads', '4'])\n"
        "    cmd.extend(['--model-draft', 'draft.gguf'])\n",
        encoding="utf-8",
    )

    failures = check_llama_cpp_args.audit(
        help_text="--threads N\n",
        source_paths=[source],
        expected_flags={"--threads"},
        known_non_llama_cpp_flags={"--model-draft"},
        flag_inputs={"--threads": check_llama_cpp_args.FlagInput("integer", True, "thread count")},
    )

    assert failures == []


def test_binary_probe_does_not_accept_version_short_circuit(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        assert "--version" not in cmd
        sentinel = check_llama_cpp_args.BINARY_PROBE_UNKNOWN_FLAG
        if cmd[1:3] == ["--known", "1"] and sentinel in cmd:
            return check_llama_cpp_args.subprocess.CompletedProcess(
                cmd,
                2,
                stdout="",
                stderr=f"unknown option: {sentinel}",
            )
        return check_llama_cpp_args.subprocess.CompletedProcess(
            cmd,
            2,
            stdout="",
            stderr=f"unknown option: {cmd[1]}",
        )

    monkeypatch.setattr(check_llama_cpp_args.subprocess, "run", fake_run)

    assert check_llama_cpp_args.binary_accepts_flag(
        Path("fake-server"),
        "--known",
        check_llama_cpp_args.FlagInput("integer", True, "known flag"),
    )
    assert not check_llama_cpp_args.binary_accepts_flag(
        Path("fake-server"),
        "--missing",
        check_llama_cpp_args.FlagInput("integer", True, "missing flag"),
    )
    assert len(calls) == 2


def test_binary_probe_rejects_when_both_options_are_unknown(monkeypatch):
    sentinel = check_llama_cpp_args.BINARY_PROBE_UNKNOWN_FLAG

    def fake_run(cmd, **kwargs):
        return check_llama_cpp_args.subprocess.CompletedProcess(
            cmd,
            2,
            stdout="",
            stderr=f"unknown option: {cmd[1]}\nunknown option: {sentinel}\n",
        )

    monkeypatch.setattr(check_llama_cpp_args.subprocess, "run", fake_run)

    assert not check_llama_cpp_args.binary_accepts_flag(
        Path("fake-server"),
        "--removed",
        check_llama_cpp_args.FlagInput("integer", True, "removed flag"),
    )


def test_extract_text_flags_ignores_c_like_comments(tmp_path):
    upstream = tmp_path / "common.cpp"
    upstream.write_text(
        'if (arg == "--active-flag") return true;\n'
        "// if (arg == \"--line-commented\") return true;\n"
        "/* if (arg == \"--block-commented\") return true; */\n",
        encoding="utf-8",
    )

    assert check_llama_cpp_args.extract_text_flags([upstream]) == {"--active-flag"}


def test_audit_accepts_cpu_hidden_flag_when_present_in_upstream_source(tmp_path):
    upstream = tmp_path / "common.cpp"
    upstream.write_text('if (arg == "--tensor-split" || arg == "-ts") return true;\n', encoding="utf-8")

    failures = check_llama_cpp_args.audit(
        help_text="--threads N\n",
        source_paths=[],
        expected_flags={"--threads", "--tensor-split"},
        known_non_llama_cpp_flags=set(),
        flag_inputs={
            "--threads": check_llama_cpp_args.FlagInput("integer", True, "thread count"),
            "--tensor-split": check_llama_cpp_args.FlagInput("csv", True, "tensor split"),
        },
        help_hidden_flags={"--tensor-split"},
        upstream_source_paths=[upstream],
    )

    assert failures == []


def test_audit_reports_cpu_hidden_flag_when_absent_from_upstream_source(tmp_path):
    upstream = tmp_path / "common.cpp"
    upstream.write_text('if (arg == "--threads") return true;\n', encoding="utf-8")

    failures = check_llama_cpp_args.audit(
        help_text="--threads N\n",
        source_paths=[],
        expected_flags={"--threads", "--tensor-split"},
        known_non_llama_cpp_flags=set(),
        flag_inputs={
            "--threads": check_llama_cpp_args.FlagInput("integer", True, "thread count"),
            "--tensor-split": check_llama_cpp_args.FlagInput("csv", True, "tensor split"),
        },
        help_hidden_flags={"--tensor-split"},
        upstream_source_paths=[upstream],
    )

    assert len(failures) == 1
    assert "--tensor-split" in failures[0]


def test_llama_cpp_manifest_source_scan_passes_when_all_tracked_flags_are_advertised():
    failures = check_llama_cpp_args.audit(
        help_text=_synthetic_help_for(),
        source_paths=_source_paths_for("llama.cpp"),
    )

    assert failures == []


def test_ik_llama_manifest_source_scan_passes_when_all_tracked_flags_are_advertised():
    failures = check_llama_cpp_args.audit(
        help_text=_synthetic_help_for(check_llama_cpp_args.IK_LLAMA_FLAG_INPUTS),
        source_paths=_source_paths_for("ik_llama"),
        expected_flags=check_llama_cpp_args.IK_LLAMA_FLAGS,
        known_non_llama_cpp_flags=check_llama_cpp_args.SOURCE_KNOWN_NON_IK_LLAMA_FLAGS,
        flag_inputs=check_llama_cpp_args.IK_LLAMA_FLAG_INPUTS,
        backend_name="ik_llama",
    )

    assert failures == []
