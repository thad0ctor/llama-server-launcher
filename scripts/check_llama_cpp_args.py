#!/usr/bin/env python3
"""Audit launcher server flags against a llama-server --help surface.

This script is intentionally boring: it tracks launcher-owned argument tokens
per backend, parses an upstream ``llama-server --help`` payload, and reports
any launcher flag that upstream no longer advertises or accepts.
It also scans the launch emission modules for option-like string literals so a
future launcher argument change has to update this manifest deliberately.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import re
import subprocess
import sys
from pathlib import Path
from textwrap import indent


HELP_TIMEOUT_SECONDS = 20
BINARY_PROBE_UNKNOWN_FLAG = "--llama-launcher-compat-probe-unknown-option"

FLAG_RE = re.compile(r"(?<![\w./])--[A-Za-z0-9][A-Za-z0-9-]*|(?<![\w./])-[A-Za-z][A-Za-z0-9]*(?![\w-])")
FLAG_LINE_RE = re.compile(r"(?<![\w./])(--[A-Za-z0-9][A-Za-z0-9-]*|-[A-Za-z][A-Za-z0-9]*)(?![\w-])")


@dataclass(frozen=True)
class FlagInput:
    """Launcher intent for a server CLI flag."""

    kind: str
    takes_value: bool
    intended_input: str
    sample_value: str = "1"


@dataclass(frozen=True)
class BackendSpec:
    name: str
    flag_inputs: dict[str, FlagInput]
    known_non_backend_flags: frozenset[str]
    help_hidden_flags: frozenset[str] = frozenset()


# Flags emitted by the launcher when the active backend is llama.cpp.
#
# Do not include user-provided custom parameters here; those are intentionally
# arbitrary pass-through flags and cannot be validated against upstream.
LLAMA_CPP_FLAGS = frozenset(
    {
        "-m",
        "--batch-size",
        "--cache-idle-slots",
        "--cache-type-k",
        "--cache-type-v",
        "--chat-template",
        "--chat-template-kwargs",
        "--cpu-moe",
        "--ctx-size",
        "--device",
        "--fit",
        "--fit-ctx",
        "--fit-target",
        "--flash-attn",
        "--host",
        "--ignore-eos",
        "--jinja",
        "--kv-unified",
        "--main-gpu",
        "--min-p",
        "--mlock",
        "--mmproj",
        "--n-cpu-moe",
        "--n-gpu-layers",
        "--n-predict",
        "--no-cache-idle-slots",
        "--no-kv-offload",
        "--no-kv-unified",
        "--no-mmap",
        "--no-mmproj",
        "--parallel",
        "--port",
        "--prio",
        "--reasoning",
        "--reasoning-budget",
        "--reasoning-budget-message",
        "--reasoning-format",
        "--seed",
        "--spec-draft-cpu-moe",
        "--spec-draft-device",
        "--spec-draft-model",
        "--spec-draft-n-cpu-moe",
        "--spec-draft-n-max",
        "--spec-draft-n-min",
        "--spec-draft-ngl",
        "--spec-draft-p-min",
        "--spec-draft-p-split",
        "--spec-draft-type-k",
        "--spec-draft-type-v",
        "--spec-ngram-map-k-size-m",
        "--spec-ngram-map-k-size-n",
        "--spec-ngram-map-k-min-hits",
        "--spec-ngram-map-k4v-size-m",
        "--spec-ngram-map-k4v-size-n",
        "--spec-ngram-map-k4v-min-hits",
        "--spec-ngram-mod-n-match",
        "--spec-ngram-mod-n-max",
        "--spec-ngram-mod-n-min",
        "--spec-ngram-simple-min-hits",
        "--spec-ngram-simple-size-m",
        "--spec-ngram-simple-size-n",
        "--spec-type",
        "--temp",
        "--tensor-split",
        "--threads",
        "--threads-batch",
        "--ubatch-size",
    }
)


LLAMA_CPP_FLAG_INPUTS = {
    "-m": FlagInput("path", True, "main GGUF model file path"),
    "--batch-size": FlagInput("integer", True, "logical prompt batch size"),
    "--cache-idle-slots": FlagInput("switch", False, "enable idle-slot KV cache behavior"),
    "--cache-type-k": FlagInput("enum", True, "K-cache quantization type such as f16/q8_0"),
    "--cache-type-v": FlagInput("enum", True, "V-cache quantization type matching K-cache"),
    "--chat-template": FlagInput("string", True, "built-in chat template name or custom template text"),
    "--chat-template-kwargs": FlagInput("json", True, "JSON object forwarded to chat template rendering"),
    "--cpu-moe": FlagInput("switch", False, "place MoE expert tensors on CPU"),
    "--ctx-size": FlagInput("integer", True, "context size in tokens"),
    "--device": FlagInput("device-list", True, "comma-separated main-model devices such as CUDA0,CUDA1"),
    "--fit": FlagInput("enum", True, "llama.cpp fit mode: on or off"),
    "--fit-ctx": FlagInput("integer", True, "context size used by llama.cpp fit planning"),
    "--fit-target": FlagInput("integer", True, "MiB reserve target used by llama.cpp fit planning"),
    "--flash-attn": FlagInput("enum", True, "flash-attention mode: on/off/auto"),
    "--host": FlagInput("host", True, "server bind host or address"),
    "--ignore-eos": FlagInput("switch", False, "ignore end-of-stream token during generation"),
    "--jinja": FlagInput("switch", False, "enable server-side Jinja chat template rendering"),
    "--kv-unified": FlagInput("switch", False, "enable unified KV cache"),
    "--main-gpu": FlagInput("integer", True, "main GPU index"),
    "--min-p": FlagInput("float", True, "min-p sampling value"),
    "--mlock": FlagInput("switch", False, "lock model memory"),
    "--mmproj": FlagInput("path", True, "multimodal projector file path"),
    "--n-cpu-moe": FlagInput("integer", True, "number of MoE layers to keep on CPU"),
    "--n-gpu-layers": FlagInput("integer", True, "number of model layers to offload to GPU"),
    "--n-predict": FlagInput("integer", True, "maximum generated tokens"),
    "--no-cache-idle-slots": FlagInput("switch", False, "disable idle-slot KV cache behavior"),
    "--no-kv-offload": FlagInput("switch", False, "disable KV cache GPU offload"),
    "--no-kv-unified": FlagInput("switch", False, "disable unified KV cache"),
    "--no-mmap": FlagInput("switch", False, "disable mmap for model loading"),
    "--no-mmproj": FlagInput("switch", False, "disable embedded or auto-detected mmproj"),
    "--parallel": FlagInput("integer", True, "number of parallel server slots"),
    "--port": FlagInput("integer", True, "server bind port"),
    "--prio": FlagInput("integer", True, "process/thread priority value"),
    "--reasoning": FlagInput("enum", True, "reasoning mode: on/off/auto"),
    "--reasoning-budget": FlagInput("integer", True, "reasoning token budget"),
    "--reasoning-budget-message": FlagInput("string", True, "message emitted when reasoning budget is exceeded"),
    "--reasoning-format": FlagInput("enum", True, "reasoning parser/format name"),
    "--seed": FlagInput("integer", True, "sampling RNG seed"),
    "--spec-draft-cpu-moe": FlagInput("switch", False, "place speculative draft MoE tensors on CPU"),
    "--spec-draft-device": FlagInput("device-list", True, "comma-separated draft-model devices such as CUDA0,CUDA1"),
    "--spec-draft-model": FlagInput("path", True, "speculative draft GGUF model file path"),
    "--spec-draft-n-cpu-moe": FlagInput("integer", True, "draft MoE layers to keep on CPU"),
    "--spec-draft-n-max": FlagInput("integer", True, "maximum speculative draft tokens"),
    "--spec-draft-n-min": FlagInput("integer", True, "minimum speculative draft tokens"),
    "--spec-draft-ngl": FlagInput("integer", True, "draft model layers to offload to GPU"),
    "--spec-draft-p-min": FlagInput("float", True, "minimum draft probability"),
    "--spec-draft-p-split": FlagInput("float", True, "draft split probability"),
    "--spec-draft-type-k": FlagInput("enum", True, "draft K-cache quantization type"),
    "--spec-draft-type-v": FlagInput("enum", True, "draft V-cache quantization type"),
    "--spec-ngram-map-k-min-hits": FlagInput("integer", True, "minimum hits for ngram map-k speculation"),
    "--spec-ngram-map-k-size-m": FlagInput("integer", True, "M size for ngram map-k speculation"),
    "--spec-ngram-map-k-size-n": FlagInput("integer", True, "N size for ngram map-k speculation"),
    "--spec-ngram-map-k4v-min-hits": FlagInput("integer", True, "minimum hits for ngram map-k4v speculation"),
    "--spec-ngram-map-k4v-size-m": FlagInput("integer", True, "M size for ngram map-k4v speculation"),
    "--spec-ngram-map-k4v-size-n": FlagInput("integer", True, "N size for ngram map-k4v speculation"),
    "--spec-ngram-mod-n-match": FlagInput("integer", True, "match count for ngram-mod speculation"),
    "--spec-ngram-mod-n-max": FlagInput("integer", True, "maximum N for ngram-mod speculation"),
    "--spec-ngram-mod-n-min": FlagInput("integer", True, "minimum N for ngram-mod speculation"),
    "--spec-ngram-simple-min-hits": FlagInput("integer", True, "minimum hits for simple ngram speculation"),
    "--spec-ngram-simple-size-m": FlagInput("integer", True, "M size for simple ngram speculation"),
    "--spec-ngram-simple-size-n": FlagInput("integer", True, "N size for simple ngram speculation"),
    "--spec-type": FlagInput("enum", True, "speculative decoding type selected by backend"),
    "--temp": FlagInput("float", True, "sampling temperature"),
    "--tensor-split": FlagInput("csv", True, "comma-separated tensor split values"),
    "--threads": FlagInput("integer", True, "CPU generation thread count"),
    "--threads-batch": FlagInput("integer", True, "CPU batch/prompt processing thread count"),
    "--ubatch-size": FlagInput("integer", True, "physical micro-batch size"),
}


IK_LLAMA_FLAG_INPUTS = {
    "-amb": FlagInput("integer", True, "ik_llama K*Q tensor compute buffer size in MiB"),
    "-ctk": FlagInput("enum", True, "ik_llama K-cache quantization type"),
    "-ctkd": FlagInput("enum", True, "ik_llama draft K-cache quantization type"),
    "-ctv": FlagInput("enum", True, "ik_llama V-cache quantization type"),
    "-ctvd": FlagInput("enum", True, "ik_llama draft V-cache quantization type"),
    "-devd": FlagInput("device-list", True, "comma-separated ik_llama draft-model devices"),
    "-draft": FlagInput("string", True, "ik_llama extra draft/spec parameters"),
    "-m": FlagInput("path", True, "main GGUF model file path"),
    "-ngld": FlagInput("integer", True, "ik_llama draft model layers to offload to GPU"),
    "-ser": FlagInput("csv", True, "ik_llama smart expert reduction pair such as 6,1", "6,1"),
    "--batch-size": FlagInput("integer", True, "logical prompt batch size"),
    "--cache-type-k": FlagInput("enum", True, "K-cache quantization type such as f16/q8_0"),
    "--cache-type-v": FlagInput("enum", True, "V-cache quantization type matching K-cache"),
    "--chat-template": FlagInput("string", True, "built-in chat template name or custom template text"),
    "--chat-template-kwargs": FlagInput("json", True, "JSON object forwarded to chat template rendering"),
    "--cpu-moe": FlagInput("switch", False, "place MoE expert tensors on CPU"),
    "--ctx-size": FlagInput("integer", True, "context size in tokens"),
    "--device": FlagInput("device-list", True, "comma-separated main-model devices such as CUDA0,CUDA1"),
    "--fit": FlagInput("switch", False, "enable ik_llama memory fitting"),
    "--fit-margin": FlagInput("integer", True, "MiB memory margin mapped from launcher fit target"),
    "--flash-attn": FlagInput("enum", True, "flash-attention mode: on/off/auto"),
    "--host": FlagInput("host", True, "server bind host or address"),
    "--ignore-eos": FlagInput("switch", False, "ignore end-of-stream token during generation"),
    "--jinja": FlagInput("switch", False, "enable server-side Jinja chat template rendering"),
    "--main-gpu": FlagInput("integer", True, "main GPU index"),
    "--min-p": FlagInput("float", True, "min-p sampling value"),
    "--mlock": FlagInput("switch", False, "lock model memory"),
    "--mmproj": FlagInput("path", True, "multimodal projector file path"),
    "--model-draft": FlagInput("path", True, "ik_llama speculative draft GGUF model file path"),
    "--n-cpu-moe": FlagInput("integer", True, "number of MoE layers to keep on CPU"),
    "--n-gpu-layers": FlagInput("integer", True, "number of model layers to offload to GPU"),
    "--n-predict": FlagInput("integer", True, "maximum generated tokens"),
    "--no-kv-offload": FlagInput("switch", False, "disable KV cache GPU offload"),
    "--no-mmap": FlagInput("switch", False, "disable mmap for model loading"),
    "--port": FlagInput("integer", True, "server bind port"),
    "--parallel": FlagInput("integer", True, "number of parallel server slots"),
    "--reasoning": FlagInput("enum", True, "reasoning mode: on/off/auto"),
    "--reasoning-budget": FlagInput("integer", True, "reasoning token budget"),
    "--reasoning-budget-message": FlagInput("string", True, "message emitted when reasoning budget is exceeded"),
    "--reasoning-format": FlagInput("enum", True, "reasoning parser/format name"),
    "--seed": FlagInput("integer", True, "sampling RNG seed"),
    "--run-time-repack": FlagInput("switch", False, "enable ik_llama run-time repack"),
    "--spec-autotune": FlagInput("switch", False, "enable ik_llama speculative autotune"),
    "--spec-type": FlagInput("enum", True, "ik_llama speculative decoding type", "mtp:n_max=1,p_min=0.0"),
    "--temp": FlagInput("float", True, "sampling temperature"),
    "--tensor-split": FlagInput("csv", True, "comma-separated tensor split values"),
    "--threads": FlagInput("integer", True, "CPU generation thread count"),
    "--threads-batch": FlagInput("integer", True, "CPU batch/prompt processing thread count"),
    "--ubatch-size": FlagInput("integer", True, "physical micro-batch size"),
}

IK_LLAMA_FLAGS = frozenset(IK_LLAMA_FLAG_INPUTS)


# ik_llama hides some GPU-only flags from --help when compiled without GPU
# offload, even though the parser still accepts them in GPU-enabled builds.
IK_LLAMA_HELP_HIDDEN_FLAGS = frozenset({"--tensor-split"})


# Option tokens present in the same emission modules but intentionally excluded
# from llama.cpp upstream comparison.
SOURCE_KNOWN_NON_LLAMA_CPP_FLAGS = frozenset(
    {
        "-c",
        "-ctk",
        "-ctkd",
        "-ctv",
        "-ctvd",
        "-dev",
        "-devd",
        "-draft",
        "-e",
        "-eq",
        "-ErrorAction",
        "-ExecutionPolicy",
        "-File",
        "-ForegroundColor",
        "-fmoe",
        "-lc",
        "-ne",
        "-ngld",
        "-NoNewline",
        "-np",
        "-or",
        "-Prompt",
        "-rf",
        "-rp",
        "-rtr",
        "-Seconds",
        "-server",
        "-t",
        "--draft-",
        "--draft-max",
        "--draft-min",
        "--draft-p-min",
        "--fit-margin",
        "--help",
        "--model-draft",
        "--noclose",
        "--reasoning-",
        "--spec-",
        "--spec-autotune",
        "--spec-draft-",
        "--spec-draft-*",
        "--spec-ngram-",
        "--spec-ngram-*",
        "--spec-ngram-min-hits",
        "--spec-ngram-size-m",
        "--spec-ngram-size-n",
        "--suffix-",
        "--suffix-max-depth",
        "--suffix-pattern-len",
    }
)

SOURCE_KNOWN_NON_IK_LLAMA_FLAGS = frozenset(
    {
        "-c",
        "-dev",
        "-e",
        "-eq",
        "-ExecutionPolicy",
        "-ErrorAction",
        "-File",
        "-ForegroundColor",
        "-lc",
        "-ne",
        "-NoNewline",
        "-np",
        "-or",
        "-Prompt",
        "-rf",
        "-rp",
        "-Seconds",
        "-server",
        "-t",
        "--cache-idle-slots",
        "--draft-",
        "--fit-ctx",
        "--fit-target",
        "--help",
        "--kv-unified",
        "--no-cache-idle-slots",
        "--no-kv-unified",
        "--no-mmproj",
        "--noclose",
        "--prio",
        "--reasoning-",
        "--spec-",
        "--spec-draft-",
        "--spec-draft-*",
        "--spec-draft-cpu-moe",
        "--spec-draft-device",
        "--spec-draft-model",
        "--spec-draft-n-cpu-moe",
        "--spec-draft-n-max",
        "--spec-draft-n-min",
        "--spec-draft-ngl",
        "--spec-draft-p-min",
        "--spec-draft-p-split",
        "--spec-draft-type-k",
        "--spec-draft-type-v",
        "--spec-ngram-*",
        "--spec-ngram-",
        "--spec-ngram-min-hits",
        "--spec-ngram-size-m",
        "--spec-ngram-size-n",
        "--spec-ngram-map-k-min-hits",
        "--spec-ngram-map-k-size-m",
        "--spec-ngram-map-k-size-n",
        "--spec-ngram-map-k4v-min-hits",
        "--spec-ngram-map-k4v-size-m",
        "--spec-ngram-map-k4v-size-n",
        "--spec-ngram-mod-n-match",
        "--spec-ngram-mod-n-max",
        "--spec-ngram-mod-n-min",
        "--spec-ngram-simple-min-hits",
        "--spec-ngram-simple-size-m",
        "--spec-ngram-simple-size-n",
        "--suffix-max-depth",
        "--suffix-pattern-len",
        "--suffix-",
    }
)


BACKENDS = {
    "llama.cpp": BackendSpec("llama.cpp", LLAMA_CPP_FLAG_INPUTS, SOURCE_KNOWN_NON_LLAMA_CPP_FLAGS),
    "ik_llama": BackendSpec(
        "ik_llama",
        IK_LLAMA_FLAG_INPUTS,
        SOURCE_KNOWN_NON_IK_LLAMA_FLAGS,
        IK_LLAMA_HELP_HIDDEN_FLAGS,
    ),
}


def extract_flags(text: str) -> set[str]:
    """Return option-looking tokens from help text or Python string literals."""
    return set(FLAG_RE.findall(text))


COMMAND_ARG_NAMES = frozenset({"cmd", "flags"})


def _is_command_arg_collection(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id in COMMAND_ARG_NAMES


def _call_attr_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return ""


def _flag_strings_in_node(node: ast.AST) -> set[str]:
    flags: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            flags.update(extract_flags(child.value))
    return flags


def _command_call_uses_name(call: ast.Call, name: str) -> bool:
    method = _call_attr_name(call)
    if method not in {"append", "extend"}:
        return False
    if not isinstance(call.func, ast.Attribute) or not _is_command_arg_collection(call.func.value):
        return False
    return any(isinstance(child, ast.Name) and child.id == name for arg in call.args for child in ast.walk(arg))


def _extract_for_loop_flag_values(node: ast.For) -> set[str]:
    if isinstance(node.target, ast.Name):
        target_names = [node.target.id]
    elif isinstance(node.target, (ast.Tuple, ast.List)):
        target_names = [elt.id for elt in node.target.elts if isinstance(elt, ast.Name)]
    else:
        target_names = []

    used_command_names = {
        name
        for name in target_names
        if any(isinstance(child, ast.Call) and _command_call_uses_name(child, name) for child in ast.walk(node))
    }
    if not used_command_names:
        return set()

    flags: set[str] = set()
    if isinstance(node.iter, (ast.List, ast.Tuple)):
        for item in node.iter.elts:
            if isinstance(item, (ast.List, ast.Tuple)):
                for index, elt in enumerate(item.elts):
                    if index < len(target_names) and target_names[index] in used_command_names:
                        flags.update(_flag_strings_in_node(elt))
            elif len(target_names) == 1 and target_names[0] in used_command_names:
                flags.update(_flag_strings_in_node(item))
    return flags


def extract_source_flags(paths: list[Path]) -> set[str]:
    """Extract option tokens from launcher argument-building call sites."""
    flags: set[str] = set()
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                method = _call_attr_name(node)
                if method in {"append", "extend"}:
                    if isinstance(node.func, ast.Attribute) and _is_command_arg_collection(node.func.value):
                        for arg in node.args:
                            flags.update(_flag_strings_in_node(arg))
                elif method == "add_arg" and node.args and _is_command_arg_collection(node.args[0]):
                    for arg in node.args[1:]:
                        flags.update(_flag_strings_in_node(arg))
            elif isinstance(node, ast.For):
                flags.update(_extract_for_loop_flag_values(node))
    return flags


def extract_text_flags(paths: list[Path]) -> set[str]:
    """Extract option-looking tokens from arbitrary upstream source text."""
    flags: set[str] = set()
    for path in paths:
        flags.update(extract_flags(path.read_text(encoding="utf-8", errors="ignore")))
    return flags


def upstream_source_files(paths: list[Path]) -> list[Path]:
    """Expand upstream source files or directories for hidden-flag checks."""
    files: list[Path] = []
    suffixes = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}
    for path in paths:
        if path.is_dir():
            files.extend(candidate for candidate in path.rglob("*") if candidate.suffix in suffixes)
        else:
            files.append(path)
    return files


def validate_flag_input_manifest(
    expected_flags: set[str] | frozenset[str] = LLAMA_CPP_FLAGS,
    flag_inputs: dict[str, FlagInput] = LLAMA_CPP_FLAG_INPUTS,
    backend_name: str = "llama.cpp",
) -> list[str]:
    """Return failures for missing, extra, or underspecified input metadata."""
    failures: list[str] = []
    missing = set(expected_flags) - set(flag_inputs)
    if missing:
        failures.append(f"tracked {backend_name} flags missing input metadata:\n" + indent(format_list(missing), "  "))

    extra = set(flag_inputs) - set(expected_flags)
    if extra:
        failures.append("input metadata exists for non-tracked flags:\n" + indent(format_list(extra), "  "))

    incomplete = {
        flag
        for flag, spec in flag_inputs.items()
        if not spec.kind.strip() or not spec.intended_input.strip()
    }
    if incomplete:
        failures.append("input metadata entries must describe kind and intended input:\n" + indent(format_list(incomplete), "  "))

    return failures


def help_flag_lines(help_text: str) -> dict[str, list[str]]:
    """Map each option token to the help lines that advertise it."""
    lines: dict[str, list[str]] = {}
    for raw_line in help_text.splitlines():
        line = raw_line.rstrip()
        for match in FLAG_LINE_RE.finditer(line):
            lines.setdefault(match.group(1), []).append(line)
    return lines


def _next_word_after_flag(line: str, flag: str) -> str:
    match = re.search(rf"(?<![\w./]){re.escape(flag)}(?![\w-])", line)
    if match is None:
        return ""
    rest = line[match.end() :]
    if rest.startswith("="):
        return rest.split(maxsplit=1)[0]
    stripped = rest.lstrip()
    if not stripped or stripped[0] == ",":
        return ""
    return stripped.split(maxsplit=1)[0].rstrip(",")


def help_line_takes_value(line: str, flag: str) -> bool:
    """Best-effort parser for llama.cpp's help layout.

    llama.cpp formats value-taking flags as ``--flag VALUE`` or
    ``--flag=<VALUE>``. Boolean switches are followed by a comma, end of line,
    or a padded prose description. This parser intentionally recognizes common
    metavariable shapes and stays conservative for description text.
    """
    word = _next_word_after_flag(line, flag)
    if not word:
        return False
    if word.startswith("="):
        return True
    if word.startswith(("<", "{", "[")):
        return True
    if "|" in word:
        return True
    if "," in word:
        return True
    if any(char.isdigit() for char in word):
        return True
    return any(char.isupper() for char in word)


def help_alias_group_takes_value(line: str) -> bool:
    """Return True if any flag alias on a help line carries a value marker."""
    return any(help_line_takes_value(line, match.group(1)) for match in FLAG_LINE_RE.finditer(line))


def detect_input_shape_mismatches(
    help_text: str,
    flag_inputs: dict[str, FlagInput] = LLAMA_CPP_FLAG_INPUTS,
) -> dict[str, tuple[bool, list[str]]]:
    """Return flags whose upstream help arity differs from launcher intent."""
    lines_by_flag = help_flag_lines(help_text)
    mismatches: dict[str, tuple[bool, list[str]]] = {}
    for flag, spec in flag_inputs.items():
        lines = lines_by_flag.get(flag, [])
        if not lines:
            continue
        advertised_takes_value = any(help_alias_group_takes_value(line) for line in lines)
        if advertised_takes_value != spec.takes_value:
            mismatches[flag] = (advertised_takes_value, lines)
    return mismatches


def sample_value_for(spec: FlagInput) -> str:
    samples = {
        "csv": "1,0",
        "device-list": "CUDA0",
        "enum": "on",
        "float": "0.1",
        "host": "127.0.0.1",
        "integer": "1",
        "json": '{"key":true}',
        "path": "/tmp/nonexistent.gguf",
        "string": "sample",
    }
    return spec.sample_value or samples.get(spec.kind, "1")


def binary_accepts_flag(binary: Path, flag: str, spec: FlagInput) -> bool:
    cmd = [str(binary), flag]
    if spec.takes_value:
        cmd.append(sample_value_for(spec))
    cmd.append(BINARY_PROBE_UNKNOWN_FLAG)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            check=False,
            text=True,
            timeout=HELP_TIMEOUT_SECONDS,
        )
    except Exception:
        return False
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode != 0 and BINARY_PROBE_UNKNOWN_FLAG in output


def read_help_text(binary: Path | None, help_file: Path | None) -> str:
    """Load help text from a file or by running ``binary --help``."""
    if help_file is not None:
        return help_file.read_text(encoding="utf-8")
    if binary is None:
        raise ValueError("either --binary or --help-file is required")
    result = subprocess.run(
        [str(binary), "--help"],
        capture_output=True,
        check=False,
        text=True,
        timeout=HELP_TIMEOUT_SECONDS,
    )
    help_text = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0 and "usage:" not in help_text:
        raise RuntimeError(f"{binary} --help exited with {result.returncode}\n{help_text}")
    if not help_text.strip():
        raise RuntimeError(f"{binary} --help returned empty output")
    return help_text


def format_list(values: set[str]) -> str:
    return "\n".join(f"- {value}" for value in sorted(values))


def audit(
    *,
    help_text: str,
    source_paths: list[Path],
    expected_flags: set[str] | frozenset[str] = LLAMA_CPP_FLAGS,
    known_non_llama_cpp_flags: set[str] | frozenset[str] = SOURCE_KNOWN_NON_LLAMA_CPP_FLAGS,
    flag_inputs: dict[str, FlagInput] = LLAMA_CPP_FLAG_INPUTS,
    backend_name: str = "llama.cpp",
    probe_binary: Path | None = None,
    help_hidden_flags: set[str] | frozenset[str] = frozenset(),
    upstream_source_paths: list[Path] | None = None,
) -> list[str]:
    """Return human-readable audit failures."""
    failures: list[str] = []
    failures.extend(validate_flag_input_manifest(expected_flags, flag_inputs, backend_name=backend_name))

    upstream_flags = extract_flags(help_text)
    missing_upstream = set(expected_flags) - upstream_flags
    if probe_binary is not None and missing_upstream:
        missing_upstream = {
            flag for flag in missing_upstream if not binary_accepts_flag(probe_binary, flag, flag_inputs[flag])
        }
    hidden_missing = missing_upstream & set(help_hidden_flags)
    if hidden_missing and upstream_source_paths:
        upstream_flags = extract_text_flags(upstream_source_files(upstream_source_paths))
        missing_upstream -= hidden_missing & upstream_flags
    if missing_upstream:
        failures.append(
            f"launcher {backend_name} flags missing from upstream llama-server --help:\n"
            + indent(format_list(missing_upstream), "  ")
        )

    shape_mismatches = detect_input_shape_mismatches(help_text, flag_inputs)
    if probe_binary is not None and shape_mismatches:
        shape_mismatches = {
            flag: mismatch
            for flag, mismatch in shape_mismatches.items()
            if not binary_accepts_flag(probe_binary, flag, flag_inputs[flag])
        }
    if shape_mismatches:
        lines = []
        for flag, (advertised_takes_value, help_lines) in sorted(shape_mismatches.items()):
            expected = "value" if flag_inputs[flag].takes_value else "switch"
            actual = "value" if advertised_takes_value else "switch"
            rendered_help = " | ".join(help_lines)
            lines.append(
                f"- {flag}: launcher expects {expected} ({flag_inputs[flag].intended_input}); "
                f"upstream help looks like {actual}: {rendered_help}"
            )
        failures.append(
            f"launcher {backend_name} flag input shapes diverge from upstream help:\n"
            + indent("\n".join(lines), "  ")
        )

    if source_paths:
        source_flags = extract_source_flags(source_paths)
        categorized = set(expected_flags) | set(known_non_llama_cpp_flags)
        uncategorized = source_flags - categorized
        if uncategorized:
            failures.append(
                "option-like source literals are not categorized in scripts/check_llama_cpp_args.py:\n"
                + indent(format_list(uncategorized), "  ")
            )

        unused_tracked = set(expected_flags) - source_flags
        if unused_tracked:
            failures.append(
                f"tracked {backend_name} flags were not found in the scanned source files:\n"
                + indent(format_list(unused_tracked), "  ")
            )

    return failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source_defaults = {
        "llama.cpp": [Path("modules/launch.py"), Path("modules/spec_launch.py")],
        "ik_llama": [Path("modules/launch.py"), Path("modules/spec_launch.py"), Path("modules/ik_llama.py")],
    }
    parser.add_argument(
        "--backend",
        choices=sorted(BACKENDS),
        default="llama.cpp",
        help="Launcher backend flag surface to validate",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--binary", type=Path, help="Path to llama-server executable")
    input_group.add_argument("--help-file", type=Path, help="Saved llama-server --help text")
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        type=Path,
        help="Python source file to scan for option-like literals; repeatable",
    )
    parser.add_argument(
        "--upstream-source-dir",
        action="append",
        default=[],
        type=Path,
        help="Upstream source directory/file to scan for tracked flags hidden from CPU-only --help",
    )
    parser.add_argument(
        "--skip-source-scan",
        action="store_true",
        help="Only compare tracked flags against upstream help text",
    )
    args = parser.parse_args(argv)
    if args.skip_source_scan:
        args.source = []
    elif args.source is None:
        args.source = source_defaults[args.backend]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    backend = BACKENDS[args.backend]
    try:
        help_text = read_help_text(args.binary, args.help_file)
        failures = audit(
            help_text=help_text,
            source_paths=args.source,
            expected_flags=set(backend.flag_inputs),
            known_non_llama_cpp_flags=backend.known_non_backend_flags,
            flag_inputs=backend.flag_inputs,
            backend_name=backend.name,
            probe_binary=args.binary,
            help_hidden_flags=backend.help_hidden_flags,
            upstream_source_paths=args.upstream_source_dir,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if failures:
        print("\n\n".join(failures), file=sys.stderr)
        return 1

    print(
        f"OK: {len(backend.flag_inputs)} tracked {backend.name} launcher flags are advertised "
        "by upstream help, accepted by the binary, or present as CPU-hidden upstream flags "
        "with expected input shapes."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
