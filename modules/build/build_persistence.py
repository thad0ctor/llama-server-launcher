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
          "notes": "",
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
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


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
    notes: str = ""
    created_at: str = ""
    last_used_at: str = ""

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("name", None)
        return d

    @classmethod
    def from_json(cls, name: str, data: dict[str, Any]) -> BuildConfig:
        return cls(
            name=name,
            backend=data.get("backend", "llama.cpp"),
            source_dir=data.get("source_dir", ""),
            build_dir=data.get("build_dir", "build"),
            git_ref=data.get("git_ref", ""),
            git_pull_before_build=bool(data.get("git_pull_before_build", False)),
            clean_build=bool(data.get("clean_build", True)),
            jobs=_safe_int(data.get("jobs", 0)),
            cuda_archs=data.get("cuda_archs", ""),
            env=dict(data.get("env", {}) or {}),
            flag_values=dict(data.get("flag_values", {}) or {}),
            extra_cmake_args=data.get("extra_cmake_args", ""),
            ui_state=dict(data.get("ui_state", {}) or {}),
            notes=data.get("notes", ""),
            created_at=data.get("created_at", ""),
            last_used_at=data.get("last_used_at", ""),
        )


def _utcnow_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class BuildConfigStore:
    """File-backed CRUD for named build configs."""

    def __init__(self, config_dir: str | os.PathLike):
        self.config_dir = Path(config_dir)
        self.path = self.config_dir / DEFAULT_FILENAME
        self._cache: dict[str, BuildConfig] = {}
        self._loaded = False

    # ---------------------------------------------------------------- io
    def _load(self) -> None:
        # Don't mark loaded until we've actually succeeded — otherwise a
        # transient unreadable file (mid-edit, permissions glitch) latches us
        # into an empty cache for the rest of the session.
        if self._loaded:
            return
        self._cache = {}
        if not self.path.is_file():
            self._loaded = True
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"WARN: build_configs.json unreadable: {exc}", file=sys.stderr)
            return  # leave _loaded=False so we retry next time
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

    def _save(self) -> None:
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            print(f"ERROR: cannot create {self.config_dir}: {exc}", file=sys.stderr)
            return
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

    # ---------------------------------------------------------------- crud
    def list_names(self) -> list[str]:
        self._load()
        return sorted(self._cache.keys(), key=str.lower)

    def get(self, name: str) -> BuildConfig | None:
        self._load()
        return self._cache.get(name)

    def save(self, cfg: BuildConfig) -> None:
        self._load()
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
        self._cache[cfg.name] = cfg
        self._save()

    def touch_last_used(self, name: str) -> None:
        self._load()
        cfg = self._cache.get(name)
        if cfg is None:
            return
        cfg.last_used_at = _utcnow_iso()
        self._save()

    def delete(self, name: str) -> bool:
        self._load()
        if name not in self._cache:
            return False
        del self._cache[name]
        self._save()
        return True

    def rename(self, old: str, new: str) -> bool:
        self._load()
        if old not in self._cache or not new or new in self._cache:
            return False
        cfg = self._cache.pop(old)
        cfg.name = new
        self._cache[new] = cfg
        self._save()
        return True
