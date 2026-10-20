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
        if spec.emitted_values:
            suffix = " {" + ",".join(spec.emitted_values) + "}"
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
      -no-kvu, --no-kv-unified
      default value: -1
      path: ./not-a-flag
    """

    assert check_llama_cpp_args.extract_flags(text) == {
        "-m",
        "--model",
        "--threads",
        "--ctx-size",
        "-no-kvu",
        "--no-kv-unified",
    }


def test_extract_flags_ignores_wildcard_flag_prose():
    text = "use --spec-ngram-*-size-n or --draft-* family settings"

    assert check_llama_cpp_args.extract_flags(text) == set()


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


def test_help_alias_flags_classifies_same_line_aliases():
    help_text = "  -m, --model FNAME\n  --threads, -t N\n  --unrelated VALUE\n"

    assert check_llama_cpp_args.help_alias_flags(help_text, {"-m", "--threads"}) == {"--model", "-t"}


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


def test_audit_result_reports_untracked_upstream_flags_without_failing():
    result = check_llama_cpp_args.audit_result(
        help_text="--threads, -t N\n--new-upstream-flag VALUE\n",
        source_paths=[],
        expected_flags={"--threads"},
        known_non_llama_cpp_flags=set(),
        flag_inputs={"--threads": check_llama_cpp_args.FlagInput("integer", True, "thread count")},
    )

    assert result.failures == ()
    assert result.missing_upstream_flags == ()
    assert result.tracked_alias_flags == frozenset({"-t"})
    assert result.untracked_upstream_flags == ("--new-upstream-flag",)
    assert check_llama_cpp_args._format_failures_from_result(result) == []


def test_audit_result_does_not_report_aliases_as_untracked_upstream_flags():
    result = check_llama_cpp_args.audit_result(
        help_text="-h, --help, --usage\n--version\n-m, --model FNAME\n--n-gpu-layers, --gpu-layers N\n--predict N\n",
        source_paths=[],
        expected_flags={"-m", "--n-gpu-layers"},
        known_non_llama_cpp_flags=set(),
        flag_inputs={
            "-m": check_llama_cpp_args.FlagInput("path", True, "model path"),
            "--n-gpu-layers": check_llama_cpp_args.FlagInput("integer", True, "GPU layers"),
        },
    )

    assert result.tracked_alias_flags == frozenset({"--model", "--gpu-layers"})
    assert result.untracked_upstream_flags == ("--predict",)


def test_report_file_contains_untracked_upstream_flags(tmp_path):
    help_file = tmp_path / "help.txt"
    report_file = tmp_path / "report.md"
    help_file.write_text(
        _synthetic_help_for() + "\n--new-upstream-flag VALUE\n",
        encoding="utf-8",
    )

    status = check_llama_cpp_args.main(
        [
            "--backend",
            "llama.cpp",
            "--help-file",
            str(help_file),
            "--skip-source-scan",
            "--report-file",
            str(report_file),
            "--upstream-ref",
            "upstream-test",
            "--launcher-ref",
            "launcher-test",
        ]
    )

    assert status == 0
    report = report_file.read_text(encoding="utf-8")
    assert "## llama.cpp upstream drift" in report
    assert "No compatibility failures; upstream classification backlog present" in report
    assert "### Upstream flags not currently classified" in report
    assert "This is feature inventory, not a launch compatibility failure." in report
    assert "`--new-upstream-flag`" in report
    assert "`upstream-test`" in report
    assert "`launcher-test`" in report


def test_fail_on_untracked_upstream_flags(tmp_path, capsys):
    help_file = tmp_path / "help.txt"
    report_file = tmp_path / "report.md"
    help_file.write_text(
        _synthetic_help_for() + "\n--new-upstream-flag VALUE\n",
        encoding="utf-8",
    )

    status = check_llama_cpp_args.main(
        [
            "--backend",
            "llama.cpp",
            "--help-file",
            str(help_file),
            "--skip-source-scan",
            "--report-file",
            str(report_file),
            "--fail-on-untracked-upstream",
        ]
    )

    assert status == 1
    assert report_file.exists()
    assert "--new-upstream-flag" in capsys.readouterr().err


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


def test_extract_text_flags_ignores_backslash_continued_line_comments(tmp_path):
    upstream = tmp_path / "common.cpp"
    upstream.write_text(
        "// removed parser branch \\\n"
        'if (arg == "--continued-commented") return true;\n'
        'if (arg == "--active-flag") return true;\n',
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


# ---------------------------------------------------------------------------
# Launcher-emitted literal values (-ngl all, --spec-type names, --load-mode)
# ---------------------------------------------------------------------------

# Excerpts from llama.cpp b11718 --help: values on the flag line and on
# indented continuation lines.
_B11718_HELP = (
    "-ngl,  --gpu-layers, --n-gpu-layers N   max. number of layers to store in VRAM, either an exact number,\n"
    "                                        'auto', or 'all' (default: auto)\n"
    "-lm,   --load-mode MODE                 model loading mode (default: auto)\n"
    "                                        - none: no special loading mode\n"
    "                                        - mlock: force system to keep model in RAM\n"
    "                                        - mmap+mlock: mmap + force system to keep model in RAM\n"
    "--spec-type none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,draft-dspark,ngram-simple,"
    "ngram-map-k,ngram-map-k4v,ngram-mod,ngram-cache\n"
    "                                        comma-separated list of types of speculative decoding\n"
)


def _value_specs():
    inputs = check_llama_cpp_args.LLAMA_CPP_FLAG_INPUTS
    return {flag: inputs[flag] for flag in ("--n-gpu-layers", "--load-mode", "--spec-type")}


def test_emitted_values_found_on_flag_and_continuation_lines():
    assert check_llama_cpp_args.detect_missing_emitted_values(_B11718_HELP, _value_specs()) == {}


def test_missing_emitted_values_are_reported():
    # A b10107-era build: no 'all', no mmap+mlock, no dflash/dspark.
    help_text = (
        "-ngl,  --gpu-layers, --n-gpu-layers N   max. number of layers to store in VRAM\n"
        "-lm,   --load-mode MODE                 model loading mode (default: mmap)\n"
        "                                        - none: no special loading mode\n"
        "                                        - mlock: mmap + force system to keep model in RAM\n"
        "--spec-type none,draft-simple,draft-eagle3,draft-mtp,ngram-simple,ngram-map-k,ngram-map-k4v,"
        "ngram-mod,ngram-cache\n"
    )
    missing = check_llama_cpp_args.detect_missing_emitted_values(help_text, _value_specs())
    # 'mlock' must not be satisfied by 'mmap+mlock' and vice versa.
    assert missing == {
        "--n-gpu-layers": ["all"],
        "--load-mode": ["mmap+mlock"],
        "--spec-type": ["draft-dflash", "draft-dspark"],
    }
    failures = check_llama_cpp_args.audit(
        help_text=help_text,
        source_paths=[],
        expected_flags=set(_value_specs()),
        known_non_llama_cpp_flags=set(),
        flag_inputs=_value_specs(),
    )
    assert len(failures) == 1
    assert "values missing from upstream help" in failures[0]
    assert "draft-dflash" in failures[0] and "all" in failures[0]


def test_prose_mentions_do_not_count_as_listed_values():
    """R2-2: only enumerated lists count. Here 'mlock' and 'all' appear only
    in prose/defaults, and the mlock bullet was removed upstream."""
    help_text = (
        "-ngl,  --gpu-layers, --n-gpu-layers N   layers in VRAM; use all of them (default: all)\n"
        "-lm,   --load-mode MODE                 model loading mode (default: auto)\n"
        "                                        - none: no special loading mode\n"
        "                                        - mmap: memory-map model (pageouts if not using mlock)\n"
        "                                        - mmap+mlock: mmap + keep model in RAM\n"
    )
    missing = check_llama_cpp_args.detect_missing_emitted_values(help_text, _value_specs())
    assert missing["--n-gpu-layers"] == ["all"]
    assert missing["--load-mode"] == ["mlock"]


def test_enumerated_values_shapes():
    ev = check_llama_cpp_args.enumerated_values
    assert {"none", "draft-mtp", "ngram-map-k4v"} <= ev("--spec-type none,draft-mtp,ngram-map-k4v\n   desc, with commas")
    assert ev("  --fit {on,off,auto}    fit mode, e.g. on") == {"on", "off", "auto"}
    ik = "  --spec-type SPEC[:k=v,...]   stage\n      types: none, draft, dflash, mtp\n      examples: --spec-type mtp:n_max=1\n"
    assert {"none", "draft", "dflash", "mtp"} <= ev(ik)
    assert "max" not in ev("  -ngl N   max. number, either an exact number, 'auto', or 'all'")


def test_binary_mode_probes_every_emitted_value(monkeypatch, tmp_path):
    """R2-2: with --binary, every emitted value is probed, even ones the help
    lists (help text can outlive the parser)."""
    seen = []

    def _accepts(binary, flag, spec):
        seen.append((flag, spec.sample_value))
        return not (flag == "--spec-type" and spec.sample_value == "draft-dspark")

    monkeypatch.setattr(check_llama_cpp_args, "binary_accepts_flag", _accepts)
    result = check_llama_cpp_args.audit_result(
        help_text=_B11718_HELP,
        source_paths=[],
        expected_flags=set(_value_specs()),
        known_non_llama_cpp_flags=set(),
        flag_inputs=_value_specs(),
        probe_binary=tmp_path / "llama-server",
    )
    for flag, spec in _value_specs().items():
        for value in spec.emitted_values:
            assert (flag, value) in seen
    assert len(result.missing_emitted_values) == 1
    assert "draft-dspark" in result.missing_emitted_values[0] and "binary rejects" in result.missing_emitted_values[0]


def test_spec_type_manifest_matches_launcher_whitelists():
    from modules.spec_launch import _ALLOWED_SPEC_TYPES_IK_LLAMA, _ALLOWED_SPEC_TYPES_LLAMA_CPP

    assert set(check_llama_cpp_args.LLAMA_CPP_SPEC_TYPES) == _ALLOWED_SPEC_TYPES_LLAMA_CPP
    assert set(check_llama_cpp_args.IK_LLAMA_SPEC_TYPES) == _ALLOWED_SPEC_TYPES_IK_LLAMA


def test_load_mode_manifest_matches_launcher_mapping():
    from modules.launch import LaunchManager

    spec = check_llama_cpp_args.LLAMA_CPP_FLAG_INPUTS["--load-mode"]
    assert set(spec.emitted_values) == set(LaunchManager._LLAMA_CPP_LOAD_MODES.values())


def test_ik_hidden_flag_fallback_stays_minimal():
    """R2-1: CI's RPC build advertises the GPU flags; a wider fallback list
    would let a really-removed flag pass whenever its spelling survives in
    the source scan."""
    hidden = check_llama_cpp_args.IK_LLAMA_HELP_HIDDEN_FLAGS
    assert hidden == {"--tensor-split"}


def test_hidden_flag_source_scan_is_limited_to_server_sources(tmp_path):
    """R2-1: a spelling that only survives in llama-bench must not confirm a
    help-hidden server flag."""
    (tmp_path / "common").mkdir()
    (tmp_path / "common" / "common.cpp").write_text('if (arg == "--threads") {}\n', encoding="utf-8")
    (tmp_path / "examples" / "llama-bench").mkdir(parents=True)
    (tmp_path / "examples" / "llama-bench" / "llama-bench.cpp").write_text(
        'if (arg == "--tensor-split") {}\n', encoding="utf-8"
    )
    files = check_llama_cpp_args.upstream_source_files([tmp_path])
    assert [f.name for f in files] == ["common.cpp"]
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
        upstream_source_paths=[tmp_path],
    )
    assert len(failures) == 1 and "--tensor-split" in failures[0]


def test_ik_ci_build_enables_rpc_so_gpu_flags_are_in_help():
    workflow = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "llama-cpp-compatibility.yml"
    text = workflow.read_text(encoding="utf-8")
    ik_build = text[text.index("cmake -S ik_llama.cpp") :]
    ik_build = ik_build[: ik_build.index("cmake --build")]
    assert "-DGGML_RPC=ON" in ik_build
