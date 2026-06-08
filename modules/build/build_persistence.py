"""Named build-config persistence.

Each entry on disk:

    {
      "schema_version": 1,
      "configs": {
        "RTX 5090 prod": {
          "backend": "llama.cpp" | "ik_llama",
          "source_dir": "/abs/path",
          "build_dir": "build",                  # relative to source_dir or abs
          "git_ref": "" | "main" | "v0.0.6" | "<sha>",
          "git_pull_before_build": true,
          "clean_build": true,
          "jobs": 8,
          "cuda_archs": "86-real;120a-real;120-real",
          "env": {                                # tool overrides
            "CC":   "/usr/bin/gcc-13",
            "CXX":  "/usr/bin/g++-13",
            "CUDACXX": "/usr/local/cuda/bin/nvcc"
          },
          "flag_values": { "GGML_CUDA": true, "GGML_LTO": true, ... },
          "extra_cmake_args": "",                # free-form passthrough
          "created_at": "2026-05-18T12:34:56Z",
          "last_used_at": "2026-05-18T12:34:56Z"
        },
        ...
      }
    }
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
DEFAULT_FILENAME = "build_configs.json"


def _safe_int(value: Any, *, default: int = 0, min_value: int = 0) -> int:
    """Parse ``value`` as int, falling back to ``default`` on bad input and
    clamping to ``min_value``. Used for fields where one malformed entry
    shouldn't poison a whole config (e.g. ``jobs``)."""
    try:
        n = int(value or 0)
    except (TypeError, ValueError):
        return default
    return max(min_value, n)


def _safe_bool(value: Any, *, default: bool = False) -> bool:
    """Parse persisted bool-ish values without treating all strings as true."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off"}:
            return False
        if not normalized:
            return default
        return default
    if isinstance(value, int | float):
        return bool(value)
    return default


@dataclass
class BuildConfig:
    name: str
    backend: str = "llama.cpp"          # "llama.cpp" | "ik_llama"
    source_dir: str = ""
    build_dir: str = "build"
    git_ref: str = ""                   # empty = leave as-is
    git_pull_before_build: bool = False
    clean_build: bool = True
    jobs: int = 0                       # 0 = auto (use nproc)
    cuda_archs: str = ""                # CMAKE_CUDA_ARCHITECTURES value
    env: dict[str, str] = field(default_factory=dict)
    flag_values: dict[str, Any] = field(default_factory=dict)
    extra_cmake_args: str = ""
    # UI-only state (generator selection, -a/-f preferences, etc.). Kept
    # separate from ``env`` so it never leaks to the cmake subprocess.
    ui_state: dict[str, str] = field(default_factory=dict)
    created_at: str = ""
    last_used_at: str = ""

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("name", None)
        return d

    @classmethod
    def from_json(cls, name: str, data: dict[str, Any]) -> BuildConfig:
        def _as_str(value: Any, default: str = "") -> str:
            """Coerce a persisted value to ``str``.

            JSON-edited configs can leave bare ints / null where strings are
            expected (e.g. ``"source_dir": 1``); passing those straight into
            ``Path(...)`` or ``subprocess.Popen`` later would raise at use
            time. Coerce here so the runner sees consistent string-ish
            fields. ``None`` collapses to the default rather than the
            string ``"None"``.
            """
            if value is None:
                return default
            try:
                return str(value)
            except Exception:
                return default

        def _coerce_env(raw: Any) -> dict[str, str]:
            """Coerce both keys and values of the env mapping to str."""
            if not isinstance(raw, Mapping):
                return {}
            out: dict[str, str] = {}
            for k, v in raw.items():
                try:
                    key = str(k)
                except Exception:
                    continue
                if v is None:
                    continue
                try:
                    out[key] = str(v)
                except Exception:
                    continue
            return out

        return cls(
            name=name,
            backend=_as_str(data.get("backend"), "llama.cpp"),
            source_dir=_as_str(data.get("source_dir"), ""),
            build_dir=_as_str(data.get("build_dir"), "build"),
            git_ref=_as_str(data.get("git_ref"), ""),
            git_pull_before_build=_safe_bool(
                data.get("git_pull_before_build", False), default=False
            ),
            clean_build=_safe_bool(data.get("clean_build", True), default=True),
            jobs=_safe_int(data.get("jobs", 0)),
            cuda_archs=_as_str(data.get("cuda_archs"), ""),
            # Defensive: a malformed persisted value ("env": "") would raise
            # in dict(...) and _load() would then drop the entire preset.
            # Default non-mapping values to {} so one bad field doesn't make
            # the whole config disappear from the UI.
            env=_coerce_env(data.get("env")),
            flag_values=(
                dict(data["flag_values"])
                if isinstance(data.get("flag_values"), Mapping)
                else {}
            ),
            extra_cmake_args=_as_str(data.get("extra_cmake_args"), ""),
            ui_state=(
                dict(data["ui_state"])
                if isinstance(data.get("ui_state"), Mapping)
                else {}
            ),
            created_at=_as_str(data.get("created_at"), ""),
            last_used_at=_as_str(data.get("last_used_at"), ""),
        )


def _utcnow_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clone_cfg(cfg: BuildConfig) -> BuildConfig:
    """Round-trip a BuildConfig through its JSON form to make an independent
    copy. ``to_json`` already takes care of which fields are persistable, so
    the resulting object has no shared mutable state with the input — safe to
    hand to callers (so they can't mutate the cache) and to stash as a
    rollback snapshot (so save-failure restoration doesn't restore an alias
    to the same mutated object)."""
    return BuildConfig.from_json(cfg.name, cfg.to_json())


class BuildConfigStore:
    """File-backed CRUD for named build configs."""

    def __init__(self, config_dir: str | os.PathLike):
        self.config_dir = Path(config_dir)
        self.path = self.config_dir / DEFAULT_FILENAME
        self._cache: dict[str, BuildConfig] = {}
        self._loaded = False

    # ---------------------------------------------------------------- io
    def _load(self) -> bool:
        # Don't mark loaded until we've actually succeeded — otherwise a
        # transient unreadable file (mid-edit, permissions glitch) latches us
        # into an empty cache for the rest of the session.
        # Returns True on success so mutators can refuse to write a fresh
        # state that would clobber an unreadable on-disk file with an empty
        # in-memory cache.
        if self._loaded:
            return True
        self._cache = {}
        if not self.path.is_file():
            self._loaded = True
            return True
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"WARN: build_configs.json unreadable: {exc}", file=sys.stderr)
            return False  # leave _loaded=False so we retry next time
        if isinstance(raw, dict):
            version = raw.get("schema_version")
            if version is not None and version != SCHEMA_VERSION:
                print(
                    f"WARN: build_configs.json schema_version={version} "
                    f"(expected {SCHEMA_VERSION}); loading best-effort.",
                    file=sys.stderr,
                )
            configs = raw.get("configs")
        else:
            configs = None
        if isinstance(configs, dict):
            for name, data in configs.items():
                if not isinstance(name, str) or not isinstance(data, dict):
                    continue
                try:
                    self._cache[name] = BuildConfig.from_json(name, data)
                except Exception as exc:
                    print(f"WARN: skipping build config {name!r}: {exc}", file=sys.stderr)
        self._loaded = True
        return True

    def _save(self) -> bool:
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            print(f"ERROR: cannot create {self.config_dir}: {exc}", file=sys.stderr)
            return False
        payload = {
            "schema_version": SCHEMA_VERSION,
            "configs": {name: cfg.to_json() for name, cfg in self._cache.items()},
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.path)
        except Exception as exc:
            print(f"ERROR: failed to write {self.path}: {exc}", file=sys.stderr)
            return False
        return True

    # ---------------------------------------------------------------- crud
    def list_names(self) -> list[str]:
        self._load()
        return sorted(self._cache.keys(), key=str.lower)

    def get(self, name: str) -> BuildConfig | None:
        self._load()
        cfg = self._cache.get(name)
        # Return an independent copy so callers can't reach back into the
        # cache and mutate the stored BuildConfig in place. Without this,
        # a caller that edits the returned config before handing it to
        # ``save()`` would observe a rollback that restored a reference to
        # the same mutated object — i.e. no rollback at all.
        return _clone_cfg(cfg) if cfg is not None else None

    def save(self, cfg: BuildConfig) -> None:
        if not self._load():
            # If the on-disk file is unreadable, refuse to overwrite it with
            # an empty cache — that would silently destroy the user's
            # presets.
            return
        # Normalize + reject blank names so we don't create unusable entries
        # (e.g. {"": {...}} which would be invisible in the picker).
        cfg.name = (cfg.name or "").strip()
        if not cfg.name:
            print("WARN: refusing to save build config with empty name",
                  file=sys.stderr)
            return
        if not cfg.created_at:
            cfg.created_at = _utcnow_iso()
        if not cfg.last_used_at:
            cfg.last_used_at = cfg.created_at
        # Snapshot prior state AND store an independent copy so a failed
        # ``_save`` can roll back to the original. Aliasing would defeat
        # the rollback if the caller continued to mutate ``cfg`` after we
        # returned.
        prior = self._cache.get(cfg.name)
        prior_snapshot = _clone_cfg(prior) if prior is not None else None
        self._cache[cfg.name] = _clone_cfg(cfg)
        if not self._save():
            if prior_snapshot is not None:
                self._cache[cfg.name] = prior_snapshot
            else:
                self._cache.pop(cfg.name, None)

    def touch_last_used(self, name: str) -> None:
        if not self._load():
            return
        cfg = self._cache.get(name)
        if cfg is None:
            return
        prior_last_used = cfg.last_used_at
        cfg.last_used_at = _utcnow_iso()
        if not self._save():
            # Roll back the timestamp bump so a write that didn't reach
            # disk doesn't leave memory believing it did.
            cfg.last_used_at = prior_last_used

    def delete(self, name: str) -> bool:
        if not self._load():
            return False
        if name not in self._cache:
            return False
        cfg = self._cache.pop(name)
        # Restore the cache entry and report failure if the disk write
        # didn't actually complete — otherwise callers see ``True`` but the
        # entry comes back on the next session.
        if not self._save():
            self._cache[name] = cfg
            return False
        return True

    def rename(self, old: str, new: str) -> bool:
        if not self._load():
            return False
        # Mirror the empty-name guard from save(): strip + reject blanks so
        # whitespace-only names can't sneak in through rename().
        new = (new or "").strip()
        if old not in self._cache or not new or new in self._cache:
            return False
        cfg = self._cache.pop(old)
        cfg.name = new
        self._cache[new] = cfg
        if not self._save():
            # Roll back so an unreported write failure doesn't leave the
            # in-memory state divergent from disk.
            self._cache.pop(new, None)
            cfg.name = old
            self._cache[old] = cfg
            return False
        return True
