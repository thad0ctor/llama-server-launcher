"""Pure helpers for the Hugging Face downloader UI."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


_LIKELY_MODEL_SUFFIXES = (
    ".gguf",
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
)
_RELATED_SUFFIXES = (
    ".json",
    ".model",
    ".txt",
)
_DEFAULT_METADATA_BASENAMES = {
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
}
_PREFERRED_GGUF_TOKENS = (
    "q4_k_m",
    "q4_k_s",
    "q5_k_m",
    "q5_k_s",
    "q6_k",
    "q8_0",
    "f16",
    "bf16",
)


@dataclass(frozen=True)
class ParsedRepoInput:
    """Normalized repo identifier and optional revision hint."""

    repo_id: str
    revision_hint: str = ""


@dataclass(frozen=True)
class HfRepoRef:
    """One branch or tag exposed by the Hub."""

    name: str
    kind: str
    target_commit: str = ""

    @property
    def display_name(self) -> str:
        prefix = "branch" if self.kind == "branch" else self.kind
        return f"{self.name} ({prefix})"


@dataclass(frozen=True)
class HfRepoFile:
    """One file available in a Hub repo revision."""

    path: str
    size_bytes: int | None = None
    kind: str = "other"
    selected_by_default: bool = False

    @property
    def display_name(self) -> str:
        suffix = format_bytes(self.size_bytes) if self.size_bytes is not None else "size unknown"
        label = self.kind
        return f"{self.path} [{label}; {suffix}]"


@dataclass(frozen=True)
class TargetDirectoryOption:
    """One local destination directory the user can toggle."""

    path: Path
    free_bytes: int | None
    exists: bool
    selected: bool

    @property
    def label(self) -> str:
        free_text = format_bytes(self.free_bytes) if self.free_bytes is not None else "unknown"
        missing = "" if self.exists else " [missing]"
        return f"{self.path} ({free_text} free){missing}"


@dataclass(frozen=True)
class DownloadTargetsState:
    """Resolved target directories and persisted checkbox choices."""

    options: tuple[TargetDirectoryOption, ...]
    selected_paths: tuple[str, ...]


def format_bytes(size_bytes: int | None) -> str:
    """Format a byte count for compact UI display."""
    if size_bytes is None:
        return "unknown"
    if size_bytes < 0:
        return "unknown"
    units = ("B", "KB", "MB", "GB", "TB", "PB")
    value = float(size_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{size_bytes} B"


def parse_bool(value) -> bool:
    """Coerce an arbitrary persisted value to a strict ``bool``.

    ``bool("false")`` and ``bool("0")`` are both ``True`` in Python because
    they're non-empty strings, so a JSON-edited settings file with
    ``"hf_force_download": "false"`` would silently flip the flag on.
    This helper accepts the strings a human would write (``"true"``,
    ``"1"``, ``"yes"``, ``"on"`` …) and treats anything else as ``False``.
    Shared with ``hf_downloader.runner`` so the UI checkbox seeding and
    the subprocess parsing agree on what a stored value means.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "on"}
    return False


def parse_pattern_lines(raw: str) -> tuple[str, ...]:
    """Parse newline/comma-separated Hub patterns."""
    rows: list[str] = []
    for line in (raw or "").replace(",", "\n").splitlines():
        item = line.strip()
        if item:
            rows.append(item)
    return tuple(rows)


def _validate_repo_id(repo_id: str) -> None:
    """Reject repo ids that would either break the ``target_dir / repo_id``
    materialization on Windows or escape the target directory via path
    traversal.

    Called from every ``normalize_repo_input`` branch so all input shapes
    (``hf://``, bare, ``https://``) reject the same set consistently.
    """
    if any(ch in repo_id for ch in '<>:"|?*'):
        raise ValueError(
            f"Repo ID {repo_id!r} contains characters that are not legal on "
            "Windows filesystems; check the URL for stray text."
        )
    # ``snapshot_download`` materializes the repo under
    # ``<target_dir>/<repo_id>``. A repo id containing ``..`` segments,
    # backslashes (Windows path separator), or URL-encoded escapes
    # could redirect the materialization outside ``target_dir`` —
    # forensic example: ``owner/../../../etc/passwd/repo``.
    parts = repo_id.split("/")
    if any(seg in {"", ".", ".."} for seg in parts):
        raise ValueError(
            f"Repo ID {repo_id!r} contains a path-traversal segment "
            f"(``.``/``..``/empty); refusing."
        )
    if "\\" in repo_id or "\x00" in repo_id or "%" in repo_id:
        # Backslash is the Windows separator; NUL terminates C strings;
        # ``%`` could be the leading byte of a URL-encoded ``..``.
        raise ValueError(
            f"Repo ID {repo_id!r} contains backslash, NUL, or ``%`` — "
            f"refusing as a potential path-traversal vector."
        )


def normalize_repo_input(raw: str) -> ParsedRepoInput:
    """Normalize a model repo input field into repo id + revision hint."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("Enter a Hugging Face repo ID or URL.")

    if text.startswith("hf://"):
        path = text[5:]
        if path.startswith("model/"):
            path = path[6:]
        parts = [part for part in path.split("/") if part]
        # Mirror the same dataset/space rejection the https:// branch does
        # below. Without this, ``hf://datasets/<owner>/<repo>`` is silently
        # rewritten to the bogus model id ``datasets/<owner>`` and fails
        # confusingly later when the runner uses ``repo_type="model"``.
        if parts and parts[0] in {"datasets", "spaces"}:
            kind = parts[0]
            raise ValueError(
                f"This is a HuggingFace {kind} URL; this tab only downloads "
                "model repos. Use a model owner/repo (or ``hf://model/<owner>/<repo>``)."
            )
        if len(parts) < 2:
            raise ValueError("Repo input must include both owner and repo name.")
        repo_id = "/".join(parts[:2])
        _validate_repo_id(repo_id)
        return ParsedRepoInput(repo_id=repo_id)

    if "://" not in text:
        # A schemeless ``huggingface.co/<owner>/<repo>`` / ``huggingface.co/tree/...``
        # is conceptually a URL, not a bare repo id. Without this guard the
        # leading ``huggingface.co`` is silently treated as ``<owner>``,
        # producing the bogus repo id ``huggingface.co/<owner>``. Promote
        # to the URL-handling branch so the same ``tree``/``blob``
        # revision-hint extraction, dataset/space rejection, and netloc
        # validation apply.
        leading = text.split("/", 1)[0].lower()
        if leading in {"huggingface.co", "www.huggingface.co"}:
            text = "https://" + text
        else:
            parts = [part for part in text.split("/") if part]
            # Same dataset/space rejection the ``hf://`` and ``https://``
            # branches do — otherwise bare ``datasets/<owner>/<repo>`` would
            # silently truncate to the bogus model id ``datasets/<owner>``
            # and fail much later in the runner.
            if parts and parts[0] in {"datasets", "spaces"}:
                kind = parts[0]
                raise ValueError(
                    f"This is a HuggingFace {kind} URL; this tab only downloads "
                    "model repos."
                )
            if len(parts) < 2:
                raise ValueError("Repo input must include both owner and repo name.")
            repo_id = "/".join(parts[:2])
            _validate_repo_id(repo_id)
            return ParsedRepoInput(repo_id=repo_id)

    parsed = urlparse(text)
    # ``urlparse`` preserves the case of the netloc, so a URL like
    # ``https://HuggingFace.co/owner/repo`` (perfectly legal — hostnames
    # are case-insensitive) used to fall through the set check and be
    # rejected. ``parsed.hostname`` is RFC-3986-lowercased, with port
    # stripped; fall back to the original netloc lowered if hostname is
    # ``None`` (e.g. malformed input).
    host = (parsed.hostname or parsed.netloc or "").lower()
    if host not in {"huggingface.co", "www.huggingface.co"}:
        raise ValueError("Only huggingface.co repo URLs are supported.")
    parts = [part for part in parsed.path.split("/") if part]
    if not parts:
        raise ValueError("Repo URL did not contain a repository path.")
    # ``huggingface.co/datasets/<owner>/<repo>`` and
    # ``huggingface.co/spaces/<owner>/<repo>`` look superficially valid
    # but the downloader runner uses ``repo_type="model"``, so silently
    # treating them as a model called ``datasets/<owner>`` (or
    # ``spaces/<owner>``) would produce baffling errors halfway through.
    # Reject them up front with an actionable message.
    if parts[0] in {"datasets", "spaces"}:
        kind = parts[0]
        raise ValueError(
            f"This is a HuggingFace {kind} URL; this tab only downloads "
            "model repos. Paste a ``huggingface.co/<owner>/<repo>`` URL."
        )
    if parts[0] in {"models", "model"}:
        parts = parts[1:]
    if len(parts) < 2:
        raise ValueError("Repo URL must include both owner and repo name.")
    repo_id = "/".join(parts[:2])
    _validate_repo_id(repo_id)
    revision_hint = ""
    # ``resolve`` is the third valid URL form Hugging Face uses for
    # revision-scoped file/tree URLs (``/owner/repo/resolve/<branch>/path``),
    # alongside ``tree`` (branch browser) and ``blob`` (file view). Without
    # it, a ``…/resolve/dev/…`` URL would silently drop the revision hint.
    if len(parts) >= 4 and parts[2] in {"tree", "blob", "resolve"}:
        remainder = parts[3:]
        if remainder:
            if remainder[0] == "refs" and len(remainder) >= 3:
                revision_hint = "/".join(remainder[:3])
            else:
                revision_hint = remainder[0]
    return ParsedRepoInput(repo_id=repo_id, revision_hint=revision_hint)


def classify_repo_file(path: str) -> str:
    """Return a lightweight category used for filtering/default selection."""
    path_l = path.lower()
    name = Path(path_l).name
    if "mmproj" in name:
        return "mmproj"
    if path_l.endswith(".gguf"):
        return "gguf"
    if path_l.endswith(_LIKELY_MODEL_SUFFIXES):
        return "weights"
    if name in _DEFAULT_METADATA_BASENAMES or path_l.endswith(_RELATED_SUFFIXES):
        return "metadata"
    return "other"


def default_selected_repo_paths(paths: list[str]) -> tuple[str, ...]:
    """Choose safe default file selections from a repo listing."""
    ggufs = [path for path in paths if classify_repo_file(path) == "gguf"]
    mmproj = [path for path in paths if classify_repo_file(path) == "mmproj"]
    if ggufs:
        primary = min(ggufs, key=_gguf_sort_key)
        return tuple(dict.fromkeys([primary, *mmproj]))
    weights = [path for path in paths if classify_repo_file(path) == "weights"]
    if weights:
        return tuple(weights[:1])
    return tuple(paths[:1])


def _gguf_sort_key(path: str) -> tuple[int, str, int, str]:
    """Rank GGUF files so the default is one likely-usable quant, not all of them."""
    name = Path(path).name.lower()
    token_rank = len(_PREFERRED_GGUF_TOKENS)
    for index, token in enumerate(_PREFERRED_GGUF_TOKENS):
        if token in name:
            token_rank = index
            break
    return (token_rank, name, len(path), path.lower())


def summarize_repo_listing(
    refs_payload: list[dict],
    files_payload: list[dict],
) -> tuple[tuple[HfRepoRef, ...], tuple[HfRepoFile, ...]]:
    """Convert raw runner payloads into typed rows for the UI."""
    refs = tuple(
        HfRepoRef(
            name=str(item.get("name", "")),
            kind=str(item.get("kind", "branch")),
            target_commit=str(item.get("target_commit", "")),
        )
        for item in refs_payload
        if item.get("name")
    )
    default_paths = set(default_selected_repo_paths([str(item.get("path", "")) for item in files_payload if item.get("path")]))
    files = tuple(
        HfRepoFile(
            path=str(item.get("path", "")),
            size_bytes=item.get("size_bytes"),
            kind=str(item.get("kind") or classify_repo_file(str(item.get("path", "")))),
            selected_by_default=str(item.get("path", "")) in default_paths,
        )
        for item in files_payload
        if item.get("path")
    )
    return refs, files


def collect_target_directory_options(
    model_dirs: list[Path],
    *,
    selected_paths: tuple[str, ...] = (),
) -> DownloadTargetsState:
    """Build checkbox rows for active model directories with free-space info."""
    # ``Path(path).expanduser().resolve()`` can raise on malformed entries
    # (a Windows path containing NUL, an empty-after-strip surrogate, etc).
    # The previous set-comprehension would surface that as a hard error and
    # block the entire Download tab from rendering. Skip individual bad
    # entries so the "select the first live option" fallback still works.
    normalized_selected: set[str] = set()
    for path in selected_paths:
        if not path:
            continue
        try:
            normalized_selected.add(str(Path(path).expanduser().resolve()))
        except (OSError, ValueError, TypeError, RuntimeError):
            continue
    options: list[TargetDirectoryOption] = []
    for index, raw_path in enumerate(model_dirs):
        path = Path(raw_path).expanduser().resolve()
        exists = path.exists() and path.is_dir()
        free_bytes: int | None = None
        try:
            usage = shutil.disk_usage(path if exists else path.parent)
            free_bytes = int(usage.free)
        except Exception:
            free_bytes = None
        if normalized_selected:
            is_selected = str(path) in normalized_selected
        else:
            is_selected = index == 0
        options.append(
            TargetDirectoryOption(
                path=path,
                free_bytes=free_bytes,
                exists=exists,
                selected=is_selected,
            )
        )
    # If a persisted selection was supplied but didn't match any current
    # model_dirs (user removed the directory from Settings, or moved
    # disks), fall back to checking the first option so the download UI
    # always has a destination. Without this the Download button would
    # be blocked with "no target dirs selected" until the user clicked.
    if normalized_selected and options and not any(o.selected for o in options):
        head = options[0]
        options[0] = TargetDirectoryOption(
            path=head.path,
            free_bytes=head.free_bytes,
            exists=head.exists,
            selected=True,
        )
    selected_paths_tuple = tuple(str(option.path) for option in options if option.selected)
    return DownloadTargetsState(options=tuple(options), selected_paths=selected_paths_tuple)


def build_runner_command(
    python_path: str | Path,
    action: str,
    payload_path: str | Path,
) -> list[str]:
    """Return the subprocess argv for the venv-backed runner."""
    return [
        str(Path(python_path)),
        "-m",
        "modules.hf_downloader.runner",
        action,
        str(Path(payload_path)),
    ]


def payload_bytes(payload: dict) -> bytes:
    """Stable JSON serialization for subprocess payload handoff."""
    return json.dumps(payload, sort_keys=True).encode("utf-8")
