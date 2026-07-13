"""Named benchmark-sweep persistence.

Each entry on disk:

    {
      "schema_version": 1,
      "configs": {
        "ngl sweep": {
          "tool": "llama-bench" | "llama-sweep-bench",
          "backend": "llama.cpp" | "ik_llama",
          "build_root": "/abs/path/to/build/root",
          "model_path": "/abs/path/model.gguf",
          "axes": {
            "n_gpu_layers": {"enabled": true, "mode": "range",
                             "raw": "", "min": 0, "max": 33, "step": 11},
            "threads":      {"enabled": true, "mode": "list",
                             "raw": "8,16,24", "min": 0, "max": 0, "step": 1}
          },
          "output_format": "json",
          "repetitions": 5,
          "extra_args": "",
          "created_at": "2026-07-13T00:00:00Z",
          "last_used_at": "2026-07-13T00:00:00Z"
        }
      }
    }

Mirrors ``modules/build/build_persistence.py`` (atomic write, schema-version
guard, defensive coercion, load-failure refuses to clobber the file).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DEFAULT_FILENAME = "bench_configs.json"

_VALID_TOOLS = {"llama-bench", "llama-sweep-bench"}
_VALID_BACKENDS = {"llama.cpp", "ik_llama"}
_VALID_MODES = {"list", "range"}


def _safe_int(value: Any, *, default: int = 0, min_value: int | None = None) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    if min_value is not None:
        return max(min_value, n)
    return n


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        low = value.strip().casefold()
        if low in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if low in {"0", "false", "f", "no", "n", "off"}:
            return False
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        return str(value)
    except Exception:
        return default


def _coerce_axes(raw: Any) -> dict[str, dict[str, Any]]:
    """Coerce persisted axes to ``{key: {enabled, mode, raw, min, max, step}}``.

    Malformed individual axes are dropped rather than poisoning the config.
    """
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, spec in raw.items():
        if not isinstance(key, str) or not isinstance(spec, Mapping):
            continue
        mode = _as_str(spec.get("mode"), "list")
        if mode not in _VALID_MODES:
            mode = "list"
        out[key] = {
            "enabled": _safe_bool(spec.get("enabled", True), default=True),
            "mode": mode,
            "raw": _as_str(spec.get("raw"), ""),
            "min": _safe_float(spec.get("min", 0)),
            "max": _safe_float(spec.get("max", 0)),
            "step": _safe_float(spec.get("step", 1)) or 1.0,
        }
    return out


@dataclass
class BenchConfig:
    name: str
    tool: str = "llama-bench"
    backend: str = "llama.cpp"
    build_root: str = ""
    model_path: str = ""
    axes: dict[str, dict[str, Any]] = field(default_factory=dict)
    output_format: str = "json"
    repetitions: int = 0
    extra_args: str = ""
    created_at: str = ""
    last_used_at: str = ""

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("name", None)
        return d

    @classmethod
    def from_json(cls, name: str, data: dict[str, Any]) -> BenchConfig:
        tool = _as_str(data.get("tool"), "llama-bench")
        if tool not in _VALID_TOOLS:
            tool = "llama-bench"
        backend = _as_str(data.get("backend"), "llama.cpp")
        if backend not in _VALID_BACKENDS:
            backend = "llama.cpp"
        return cls(
            name=name,
            tool=tool,
            backend=backend,
            build_root=_as_str(data.get("build_root"), ""),
            model_path=_as_str(data.get("model_path"), ""),
            axes=_coerce_axes(data.get("axes")),
            output_format=_as_str(data.get("output_format"), "json") or "json",
            repetitions=_safe_int(data.get("repetitions", 0), min_value=0),
            extra_args=_as_str(data.get("extra_args"), ""),
            created_at=_as_str(data.get("created_at"), ""),
            last_used_at=_as_str(data.get("last_used_at"), ""),
        )


def _utcnow_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clone(cfg: BenchConfig) -> BenchConfig:
    return BenchConfig.from_json(cfg.name, cfg.to_json())


class BenchConfigStore:
    """File-backed CRUD for named benchmark sweep configs."""

    def __init__(self, config_dir: str | os.PathLike):
        self.config_dir = Path(config_dir)
        self.path = self.config_dir / DEFAULT_FILENAME
        self._cache: dict[str, BenchConfig] = {}
        self._loaded = False

    # ------------------------------------------------------------------ io
    def _load(self) -> bool:
        if self._loaded:
            return True
        self._cache = {}
        if not self.path.is_file():
            self._loaded = True
            return True
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"WARN: {DEFAULT_FILENAME} unreadable: {exc}", file=sys.stderr)
            return False
        if not isinstance(raw, dict):
            print(
                f"WARN: {DEFAULT_FILENAME} top-level is not a JSON object "
                f"(got {type(raw).__name__}); refusing to load.",
                file=sys.stderr,
            )
            return False
        version = raw.get("schema_version")
        if version is not None and version != SCHEMA_VERSION:
            print(
                f"WARN: {DEFAULT_FILENAME} schema_version={version} "
                f"(expected {SCHEMA_VERSION}); loading best-effort.",
                file=sys.stderr,
            )
        if "configs" not in raw or raw.get("configs") is None:
            print(
                f"WARN: {DEFAULT_FILENAME} is missing the top-level "
                f"``configs`` object; refusing to load to avoid clobbering it.",
                file=sys.stderr,
            )
            return False
        configs = raw.get("configs")
        if not isinstance(configs, dict):
            print(
                f"WARN: {DEFAULT_FILENAME} ``configs`` field is not a JSON "
                f"object (got {type(configs).__name__}); refusing to load.",
                file=sys.stderr,
            )
            return False
        for raw_name, data in configs.items():
            if not isinstance(raw_name, str) or not isinstance(data, dict):
                continue
            name = raw_name.strip()
            if not name:
                continue
            try:
                self._cache[name] = BenchConfig.from_json(name, data)
            except Exception as exc:
                print(f"WARN: skipping bench config {name!r}: {exc}", file=sys.stderr)
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
        if not self._load():
            print(
                f"WARN: list_names() returning empty because {DEFAULT_FILENAME} " f"could not be loaded.",
                file=sys.stderr,
            )
        return sorted(self._cache.keys(), key=str.lower)

    def get(self, name: str) -> BenchConfig | None:
        if not self._load():
            return None
        cfg = self._cache.get(name)
        return _clone(cfg) if cfg is not None else None

    def save(self, cfg: BenchConfig) -> bool:
        if not self._load():
            return False
        normalized_name = (cfg.name or "").strip()
        if not normalized_name:
            print("WARN: refusing to save bench config with empty name", file=sys.stderr)
            return False
        stored = _clone(cfg)
        stored.name = normalized_name
        if not stored.created_at:
            stored.created_at = _utcnow_iso()
        stored.last_used_at = _utcnow_iso()
        prior = self._cache.get(normalized_name)
        prior_snapshot = _clone(prior) if prior is not None else None
        self._cache[normalized_name] = stored
        if not self._save():
            if prior_snapshot is not None:
                self._cache[normalized_name] = prior_snapshot
            else:
                self._cache.pop(normalized_name, None)
            return False
        return True

    def delete(self, name: str) -> bool:
        if not self._load():
            return False
        if name not in self._cache:
            return False
        cfg = self._cache.pop(name)
        prior_snapshot = _clone(cfg)
        if not self._save():
            self._cache[name] = prior_snapshot
            return False
        return True
