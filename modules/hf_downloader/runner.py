"""Subprocess entrypoint for venv-backed Hugging Face operations."""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path


def _emit(event: str, **payload) -> None:
    print(json.dumps({"event": event, **payload}), flush=True)


def _load_payload(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _token_value(payload: dict):
    """Resolve the HF auth token for ``snapshot_download`` / list calls.

    Priority order:
      1. ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` in the environment
         (set by the parent process; never written to the temp payload
         file). This is the preferred channel — the payload JSON lives
         on disk for the lifetime of the subprocess and could be
         exposed by ``ps``/``lsof`` / forensic disk reads.
      2. ``payload["token"]`` for backwards compatibility with old
         callers that haven't been updated.
    Empty / unset → ``None`` (anonymous Hub access).
    """
    env_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or ""
    ).strip()
    if env_token:
        return env_token
    token = (payload.get("token") or "").strip()
    return token or None


def _extract_file_size(info) -> int | None:
    for attr in ("size", "blob_size", "lfs_size", "file_size"):
        value = getattr(info, attr, None)
        if isinstance(value, int):
            return value
    lfs = getattr(info, "lfs", None)
    if isinstance(lfs, dict):
        value = lfs.get("size")
        if isinstance(value, int):
            return value
    return None


def _classify(path: str) -> str:
    path_l = path.lower()
    name = Path(path_l).name
    # Check ``mmproj`` BEFORE generic ``.gguf`` — multimodal projector files
    # like ``mmproj-model-f16.gguf`` end in ``.gguf`` too, and downstream
    # logic that trusts the emitted ``kind`` needs to see them as
    # ``mmproj`` rather than ordinary weight shards.
    if "mmproj" in name:
        return "mmproj"
    if path_l.endswith(".gguf"):
        return "gguf"
    if path_l.endswith((".safetensors", ".bin", ".pt", ".pth")):
        return "weights"
    if name.endswith((".json", ".txt", ".model")):
        return "metadata"
    return "other"


def run_list(payload: dict) -> int:
    from huggingface_hub import HfApi

    api = HfApi()
    repo_id = payload["repo_id"]
    revision = (payload.get("revision") or "").strip() or None
    token = _token_value(payload)

    _emit("status", message=f"Loading {repo_id}…")
    refs_payload: list[dict] = []
    try:
        refs = api.list_repo_refs(repo_id, repo_type="model", token=token)
    except Exception:
        refs = None
    if refs is not None:
        for branch in getattr(refs, "branches", []) or []:
            refs_payload.append(
                {
                    "name": getattr(branch, "name", ""),
                    "kind": "branch",
                    "target_commit": getattr(branch, "target_commit", "") or "",
                }
            )
        for tag in getattr(refs, "tags", []) or []:
            refs_payload.append(
                {
                    "name": getattr(tag, "name", ""),
                    "kind": "tag",
                    "target_commit": getattr(tag, "target_commit", "") or "",
                }
            )

    try:
        info = api.model_info(
            repo_id,
            revision=revision,
            files_metadata=True,
            token=token,
        )
    except TypeError:
        info = api.model_info(
            repo_id,
            revision=revision,
            token=token,
        )

    files_payload: list[dict] = []
    for sibling in getattr(info, "siblings", []) or []:
        path = getattr(sibling, "rfilename", None) or getattr(sibling, "path", None)
        if not path:
            continue
        files_payload.append(
            {
                "path": path,
                "size_bytes": _extract_file_size(sibling),
                "kind": _classify(path),
            }
        )
    files_payload.sort(key=lambda item: item["path"])
    refs_payload.sort(key=lambda item: (item["kind"], item["name"]))
    _emit(
        "listing",
        repo_id=repo_id,
        revision=revision or "",
        resolved_revision=getattr(info, "sha", "") or "",
        refs=refs_payload,
        files=files_payload,
    )
    _emit("complete", message=f"Loaded {len(files_payload)} files from {repo_id}.")
    return 0


def _build_progress_tqdm_class():  # pragma: no cover - exercised indirectly
    """Construct a tqdm subclass that also emits JSON progress events.

    Must subclass tqdm (not wrap) so class-level hooks like ``get_lock`` /
    ``set_lock`` that huggingface_hub's thread_map uses still work.
    """
    from tqdm.auto import tqdm as _BaseTqdm

    class _ProgressTqdm(_BaseTqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            _emit(
                "progress",
                current=self.n,
                total=self.total,
                description=str(self.desc or ""),
            )

        def update(self, n=1):
            result = super().update(n)
            _emit(
                "progress",
                current=self.n,
                total=self.total,
                description=str(self.desc or ""),
            )
            return result

        def close(self):
            super().close()
            _emit(
                "progress",
                current=self.n,
                total=self.total,
                description=str(self.desc or ""),
            )

    return _ProgressTqdm


def _parse_bool(value) -> bool:
    """Coerce a payload value to a strict bool.

    ``bool("false")`` and ``bool("0")`` are both ``True`` in Python because
    they're non-empty strings, so a CLI-edited payload with
    ``"force_download": "false"`` used to silently enable force-download.
    This helper accepts the strings the average human would write
    (``"true"``/``"1"``/``"yes"``/``"on"`` etc.) and treats anything else
    as ``False``.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "on"}
    # Unknown payload types (list/dict/object) → False. The previous
    # ``bool(value)`` fallback flipped a non-empty list/dict to True,
    # which would silently enable a destructive flag like
    # ``force_download`` if a hand-edited JSON payload shipped an
    # accidentally-list-valued entry.
    return False


def _normalize_path_list(value) -> list[Path]:
    """Coerce a payload field to ``list[Path]`` at the runner boundary.

    Same rationale as ``_normalize_pattern_list``: a CLI/manual payload
    with ``"target_dirs": "/models"`` would otherwise iterate
    character-by-character and create junk single-letter directories.
    """
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        value = [value]
    return [Path(item) for item in value if isinstance(item, (str, Path)) and str(item)]


def _normalize_pattern_list(value) -> list[str]:
    """Coerce a payload field to ``list[str]`` at the runner boundary.

    The UI always sends list/tuple values, but this module is also reachable
    via ``python -m modules.hf_downloader.runner ... <payload.json>`` from
    the CLI, where a hand-written payload can easily carry a bare string
    (e.g. ``"include_patterns": "*.gguf"``). That string would silently
    iterate as characters and hand ``snapshot_download`` a nonsense filter.
    Wrap bare strings in a single-item list and drop non-string entries.
    """
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    return [item for item in value if isinstance(item, str) and item]


_ALLOWED_DOWNLOAD_MODES = frozenset({"selected", "snapshot"})


def _combined_allow_patterns(payload: dict) -> list[str] | None:
    mode = payload.get("download_mode", "selected")
    if mode not in _ALLOWED_DOWNLOAD_MODES:
        # The old code treated any non-``"selected"`` value as snapshot,
        # so a typo like ``"snapshop"`` silently widened scope to a full
        # repo download. Reject the typo at the runner boundary instead.
        raise ValueError(
            f"Unknown download_mode {mode!r}; expected one of "
            f"{sorted(_ALLOWED_DOWNLOAD_MODES)}."
        )
    patterns: list[str] = []
    if mode == "selected":
        patterns.extend(_normalize_pattern_list(payload.get("selected_files")))
    patterns.extend(_normalize_pattern_list(payload.get("include_patterns")))
    return patterns or None


def run_download(payload: dict) -> int:
    from huggingface_hub import snapshot_download

    repo_id = payload["repo_id"]
    revision = (payload.get("revision") or "").strip() or None
    token = _token_value(payload)
    allow_patterns = _combined_allow_patterns(payload)
    # ``selected`` mode with no selected files AND no include_patterns used
    # to silently turn into a full-repo download (allow_patterns=None lets
    # snapshot_download fetch everything). Fail fast instead so a CLI/manual
    # payload that forgot to populate ``selected_files`` can't accidentally
    # pull dozens of GB.
    if payload.get("download_mode", "selected") == "selected" and not allow_patterns:
        raise ValueError(
            "'selected' download mode requires at least one selected_file or "
            "include_pattern; got an empty selection. To download the whole "
            "repo set ``download_mode`` to ``snapshot``."
        )
    ignore_patterns = _normalize_pattern_list(payload.get("ignore_patterns")) or None
    target_dirs = _normalize_path_list(payload.get("target_dirs"))
    # Validate + clamp at the runner boundary. The UI clamps too, but this
    # module is also reachable via ``python -m modules.hf_downloader.runner
    # download <payload.json>`` from the CLI, where a hand-written payload
    # could carry a non-int, negative, or wildly-large value straight to
    # ``snapshot_download``'s threadpool.
    try:
        parsed_workers = int(payload.get("max_workers") or 4)
    except (TypeError, ValueError):
        parsed_workers = 4
    max_workers = max(1, min(32, parsed_workers))
    try:
        tqdm_class = _build_progress_tqdm_class()
    except ImportError:
        # tqdm is a transitive dep of huggingface_hub in real use; if it's
        # absent (e.g. tests that mock the hub) fall back to upstream's
        # default progress class and skip our JSON-emitting wrapper.
        tqdm_class = None

    if not target_dirs:
        raise ValueError("No target directories were selected.")

    # Pre-flight: confirm every target directory is writable before any
    # network I/O. A typo in the repo_id used to silently ``mkdir`` empty
    # directories under each target and then fail with a confusing HF
    # error; verifying write access here keeps the failure local and
    # actionable, and avoids leaving stub directories behind.
    import tempfile

    for target_dir in target_dirs:
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OSError(f"Cannot create target directory {target_dir}: {exc}") from exc
        # Use a uniquely-named temp file inside ``target_dir`` rather than
        # a fixed sentinel like ``.hf-download-write-test``. The fixed
        # name could (a) collide with a legitimate user file at that
        # path and clobber it on success or unlink it on failure, and
        # (b) race with a parallel HF runner against the same dir.
        # ``mkstemp`` guarantees an exclusive new inode.
        probe_fd = None
        probe_path = None
        try:
            probe_fd, probe_path = tempfile.mkstemp(
                prefix=".hf-write-probe-",
                dir=str(target_dir),
            )
        except OSError as exc:
            raise OSError(
                f"Target directory {target_dir} is not writable: {exc}"
            ) from exc
        finally:
            if probe_fd is not None:
                try:
                    os.close(probe_fd)
                except OSError:
                    pass
            if probe_path:
                try:
                    os.unlink(probe_path)
                except OSError:
                    # Probe file leaked — surface as a writability failure
                    # rather than silently leaving the stub.
                    pass

    for index, target_dir in enumerate(target_dirs, start=1):
        _emit(
            "target-start",
            target=str(target_dir),
            index=index,
            total_targets=len(target_dirs),
            message=f"Downloading into {target_dir}…",
        )
        kwargs = {
            "repo_id": repo_id,
            "repo_type": "model",
            "revision": revision,
            "local_dir": target_dir,
            "allow_patterns": allow_patterns,
            "ignore_patterns": ignore_patterns,
            "force_download": _parse_bool(payload.get("force_download")),
            "local_files_only": _parse_bool(payload.get("local_files_only")),
            "token": token,
            "max_workers": max_workers,
        }
        if tqdm_class is not None:
            kwargs["tqdm_class"] = tqdm_class
        snapshot_download(**kwargs)
        _emit(
            "target-complete",
            target=str(target_dir),
            index=index,
            total_targets=len(target_dirs),
        )
    _emit("complete", message=f"Finished downloading {repo_id}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[0] not in {"list", "download"}:
        print("usage: python -m modules.hf_downloader.runner <list|download> <payload.json>", file=sys.stderr)
        return 2
    action, payload_path = args
    # Pull payload parsing INSIDE the error envelope. Otherwise a
    # ``FileNotFoundError`` (payload path missing) or ``JSONDecodeError``
    # (corrupt payload) would skip the structured ``error`` event the UI
    # relies on and leave the caller with only raw stderr + exit code 1
    # to interpret.
    try:
        payload = _load_payload(payload_path)
        if action == "list":
            return run_list(payload)
        return run_download(payload)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        _emit("error", message=str(exc), exc_type=type(exc).__name__)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
