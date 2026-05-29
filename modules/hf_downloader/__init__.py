"""Hugging Face downloader package."""

from importlib import import_module

from .helpers import (
    DownloadTargetsState,
    HfRepoFile,
    HfRepoRef,
    ParsedRepoInput,
    TargetDirectoryOption,
    build_runner_command,
    collect_target_directory_options,
    format_bytes,
    normalize_repo_input,
    parse_pattern_lines,
    summarize_repo_listing,
)

__all__ = [
    "DownloadTargetsState",
    "HfRepoFile",
    "HfRepoRef",
    "HuggingFaceDownloaderTab",
    "ParsedRepoInput",
    "TargetDirectoryOption",
    "build_runner_command",
    "collect_target_directory_options",
    "create_hf_downloader_tab",
    "format_bytes",
    "normalize_repo_input",
    "parse_pattern_lines",
    "summarize_repo_listing",
]


def __getattr__(name):
    if name in {"HuggingFaceDownloaderTab", "create_hf_downloader_tab"}:
        mod = import_module(".tab", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
