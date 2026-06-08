"""Helpers for launcher-managed Python virtual environments."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManagedDependency:
    """One dependency the settings tab can inspect/install in a venv."""
    key: str
    label: str
    package_name: str
    import_name: str
    description: str
    required: bool = False
    install_name: str | None = None


@dataclass(frozen=True)
class VenvTargetInfo:
    """Resolved target path + best-effort interpreter discovery."""
    raw_input: str
    effective_dir: Path
    uses_default: bool
    python_path: Path | None

    @property
    def exists(self) -> bool:
        return self.effective_dir.exists()

    @property
    def looks_like_venv(self) -> bool:
        return self.python_path is not None


@dataclass(frozen=True)
class DependencyStatus:
    """Installed/version state for one dependency inside a venv."""
    dependency: ManagedDependency
    available: bool
    version: str | None = None
    error: str | None = None


MANAGED_DEPENDENCIES: tuple[ManagedDependency, ...] = (
    ManagedDependency(
        key="requests",
        label="requests",
        package_name="requests",
        import_name="requests",
        description="Required for version checks and update downloads.",
        required=True,
    ),
    ManagedDependency(
        key="torch",
        label="torch",
        package_name="torch",
        import_name="torch",
        description="Recommended for GPU detection and CUDA/Metal device discovery.",
    ),
    ManagedDependency(
        key="psutil",
        label="psutil",
        package_name="psutil",
        import_name="psutil",
        description="Recommended for richer CPU and RAM detection.",
    ),
    ManagedDependency(
        key="huggingface_hub",
        label="huggingface_hub / hf",
        package_name="huggingface_hub",
        import_name="huggingface_hub",
        # The bare ``huggingface_hub`` distribution does not register the
        # ``hf`` console script unless the ``[cli]`` extra is requested.
        # Pin the extra here so the description (which advertises the CLI)
        # actually matches what gets installed.
        install_name="huggingface_hub[cli]",
        description="Model downloads; installs the `hf` CLI.",
    ),
)


def required_managed_dependencies() -> tuple[ManagedDependency, ...]:
    """Return launcher-managed dependencies that are required by default."""
    return tuple(dep for dep in MANAGED_DEPENDENCIES if dep.required)


def launcher_repo_dir() -> Path:
    """Return the repository root for this launcher checkout."""
    return Path(__file__).resolve().parent.parent


def default_venv_dir(*, repo_dir: str | Path | None = None) -> Path:
    """Default venv target: ``<repo>/venv``."""
    base = Path(repo_dir) if repo_dir is not None else launcher_repo_dir()
    return base / "venv"


def resolve_venv_dir(raw_path: str, *, repo_dir: str | Path | None = None) -> Path:
    """Resolve ``raw_path`` to an absolute venv target.

    Blank input maps to the default ``<repo>/venv`` directory. Relative input
    is resolved relative to the repo root so a plain ``venv`` means the same
    thing on every platform.
    """
    repo = Path(repo_dir) if repo_dir is not None else launcher_repo_dir()
    raw = (raw_path or "").strip()
    if not raw:
        return default_venv_dir(repo_dir=repo).resolve()
    target = Path(raw).expanduser()
    if not target.is_absolute():
        target = repo / target
    return target.resolve()


def venv_python_candidates(
    venv_dir: str | Path, *, platform: str | None = None
) -> tuple[Path, ...]:
    """Return plausible Python interpreter locations inside ``venv_dir``."""
    plat = platform or sys.platform
    root = Path(venv_dir)
    if plat.startswith("win"):
        return (
            root / "Scripts" / "python.exe",
            root / "python.exe",
            root / "Scripts" / "python",
            root / "python",
        )
    return (
        root / "bin" / "python",
        root / "python",
    )


def locate_venv_python(
    venv_dir: str | Path, *, platform: str | None = None
) -> Path | None:
    """Return the first existing Python interpreter inside ``venv_dir``."""
    for candidate in venv_python_candidates(venv_dir, platform=platform):
        if candidate.is_file():
            return candidate
    return None


def describe_venv_target(
    raw_path: str, *, repo_dir: str | Path | None = None, platform: str | None = None
) -> VenvTargetInfo:
    """Return UI-friendly path/interpreter info for ``raw_path``."""
    effective = resolve_venv_dir(raw_path, repo_dir=repo_dir)
    return VenvTargetInfo(
        raw_input=(raw_path or "").strip(),
        effective_dir=effective,
        uses_default=not (raw_path or "").strip(),
        python_path=locate_venv_python(effective, platform=platform),
    )


def resolve_active_venv_path(
    raw_path: str,
    *,
    repo_dir: str | Path | None = None,
    platform: str | None = None,
) -> str:
    """Return the venv path the launcher should actually use at runtime.

    Rules:
    - Explicit non-blank input always resolves to an absolute path.
    - Blank input activates the default ``<repo>/venv`` only when that target
      already looks like a real virtual environment.
    - Otherwise return ``""`` to mean "no active venv".
    """
    raw = (raw_path or "").strip()
    if raw:
        return str(resolve_venv_dir(raw, repo_dir=repo_dir))
    info = describe_venv_target("", repo_dir=repo_dir, platform=platform)
    return str(info.effective_dir) if info.looks_like_venv else ""


def _win_cmd_quote(arg: str) -> str:
    """Wrap ``arg`` in double quotes for a cmd.exe command line.

    ``subprocess.list2cmdline`` only quotes args that contain whitespace or
    double quotes, so a path like ``C:\\projects&work\\venv`` passes through
    unquoted — and cmd then parses the embedded ``&`` as a command
    separator. That turns a routine ``rmdir /s /q <path>`` into a deletion
    of just ``C:\\projects`` followed by an attempt to execute
    ``work\\venv`` as a fresh command. Force-quoting every argument makes
    every cmd metacharacter inert.
    """
    return '"' + arg.replace('"', '""') + '"'


def _shell_join(args: list[str], *, platform: str | None = None) -> str:
    plat = platform or sys.platform
    if plat.startswith("win"):
        # Always double-quote each arg so embedded cmd metacharacters
        # (& | ^ < > and friends) can't break out of the intended
        # command. See _win_cmd_quote for the rationale.
        return " ".join(_win_cmd_quote(arg) for arg in args)
    return " ".join(shlex.quote(arg) for arg in args)


def default_venv_base_python_args(
    *, platform: str | None = None
) -> tuple[str, ...]:
    """Return the preferred Python launcher for creating a new venv."""
    plat = platform or sys.platform
    exe_name = Path(sys.executable).name.lower()

    if plat.startswith("win"):
        if shutil.which("py"):
            return ("py", "-3")
        if shutil.which("python"):
            return ("python",)
        if exe_name.startswith("python"):
            return (sys.executable,)
        return ("python",)

    for candidate in ("python3", "python"):
        if shutil.which(candidate):
            return (candidate,)
    if exe_name.startswith("python"):
        return (sys.executable,)
    return ("python3",)


def build_create_venv_command(
    venv_dir: str | Path,
    *,
    base_python: str | tuple[str, ...] | list[str] | None = None,
    platform: str | None = None,
) -> str:
    """Return a shell command that creates ``venv_dir``."""
    if base_python is None:
        python_args = list(default_venv_base_python_args(platform=platform))
    elif isinstance(base_python, (tuple, list)):
        python_args = list(base_python)
    else:
        python_args = [base_python]
    args = [*python_args, "-m", "venv", str(Path(venv_dir))]
    return _shell_join(args, platform=platform)


def build_bootstrap_venv_command(
    venv_dir: str | Path,
    *,
    dependencies: tuple[ManagedDependency, ...] | None = None,
    base_python: str | tuple[str, ...] | list[str] | None = None,
    platform: str | None = None,
) -> str:
    """Return a shell command that creates a venv and installs managed packages."""
    target = Path(venv_dir)
    create_cmd = build_create_venv_command(
        target,
        base_python=base_python,
        platform=platform,
    )
    python_path = venv_python_candidates(target, platform=platform)[0]
    # An explicit ``dependencies=()`` (or ``[]``) must be honored as "no
    # managed installs" — using ``or`` collapsed empty tuples to the
    # required-deps default and silently overrode the caller's intent.
    install_list = (
        required_managed_dependencies()
        if dependencies is None
        else dependencies
    )
    packages = [dep.install_name or dep.package_name for dep in install_list]
    upgrade_pip_cmd = _shell_join(
        [str(python_path), "-m", "pip", "install", "--upgrade", "pip"],
        platform=platform,
    )
    # Skip the trailing ``pip install`` step when there are no packages to
    # install — otherwise we emit a bare ``pip install`` that exits non-zero
    # and breaks the chained shell pipeline.
    commands = [create_cmd, upgrade_pip_cmd]
    if packages:
        commands.append(
            _shell_join(
                [str(python_path), "-m", "pip", "install", *packages],
                platform=platform,
            )
        )
    return " && ".join(commands)


def build_install_dependency_command(
    venv_dir: str | Path,
    dependency: ManagedDependency,
    *,
    platform: str | None = None,
) -> str:
    """Return a shell command that installs ``dependency`` into ``venv_dir``."""
    python = locate_venv_python(venv_dir, platform=platform)
    if python is None:
        python = venv_python_candidates(venv_dir, platform=platform)[0]
    pkg = dependency.install_name or dependency.package_name
    args = [str(python), "-m", "pip", "install", pkg]
    return _shell_join(args, platform=platform)


def build_remove_dependency_command(
    venv_dir: str | Path,
    dependency: ManagedDependency,
    *,
    platform: str | None = None,
) -> str:
    """Return a shell command that uninstalls ``dependency`` from ``venv_dir``."""
    python = locate_venv_python(venv_dir, platform=platform)
    if python is None:
        python = venv_python_candidates(venv_dir, platform=platform)[0]
    args = [str(python), "-m", "pip", "uninstall", "-y", dependency.package_name]
    return _shell_join(args, platform=platform)


def build_remove_venv_command(
    venv_dir: str | Path,
    *,
    platform: str | None = None,
) -> str:
    """Return a shell command that removes ``venv_dir``.

    Rejects unsafe targets up front: blank/``"."``/filesystem-root paths
    would otherwise compose into ``rm -rf .`` (current dir) or ``rm -rf /``
    (whole disk), which is catastrophic on POSIX and arbitrary-folder
    deletion on Windows (``rmdir /s /q .``).
    """
    raw = str(venv_dir or "")
    if not raw.strip():
        raise ValueError("Refusing to remove an empty venv path.")
    target_path = Path(raw).expanduser()
    # Resolve the target before comparing — the literal-only check used to
    # let dangerous aliases through:
    #   - ``"."`` and ``".."`` against a non-resolved ``Path(".")`` only
    #     matched the exact literal, missing ``"./"``, ``"foo/.."``, etc.
    #   - ``~`` would expand to a real path that bypassed the literal
    #     guard but the resolved path equals ``Path.home()`` — exactly the
    #     thing we want to refuse.
    try:
        resolved_target = target_path.resolve(strict=False)
    except OSError:
        resolved_target = target_path
    try:
        cwd_resolved = Path.cwd().resolve(strict=False)
    except OSError:
        cwd_resolved = Path.cwd()
    try:
        home_resolved = Path.home().resolve(strict=False)
    except (OSError, RuntimeError):
        home_resolved = Path.home()
    if resolved_target in {cwd_resolved, home_resolved}:
        raise ValueError(f"Refusing to remove unsafe venv path: {raw!r}")
    if resolved_target == Path(resolved_target.anchor):
        # Anchor is the filesystem root (``/`` on POSIX, ``C:\`` on Windows).
        raise ValueError(f"Refusing to remove filesystem root: {raw!r}")
    target = str(resolved_target)
    plat = platform or sys.platform
    if plat.startswith("win"):
        # Force-quote so a venv path containing cmd metacharacters can't
        # break out of the rmdir invocation.
        return " ".join(_win_cmd_quote(s) for s in ["rmdir", "/s", "/q", target])
    return _shell_join(["rm", "-rf", target], platform=plat)


def probe_dependency_status(
    venv_dir: str | Path,
    dependency: ManagedDependency,
    *,
    platform: str | None = None,
    timeout: float = 2.5,
) -> DependencyStatus:
    """Inspect one dependency inside a venv without importing it into the GUI."""
    python = locate_venv_python(venv_dir, platform=platform)
    if python is None:
        return DependencyStatus(
            dependency=dependency,
            available=False,
            error="venv python not found",
        )

    script = """
import importlib.metadata
import importlib.util
import json
import sys

dist, mod = sys.argv[1], sys.argv[2]
available = importlib.util.find_spec(mod) is not None
version = None
error = None
if available:
    try:
        version = importlib.metadata.version(dist)
    except Exception as exc:
        error = str(exc)
print(json.dumps({"available": available, "version": version, "error": error}))
"""
    try:
        proc = subprocess.run(
            [str(python), "-c", script, dependency.package_name, dependency.import_name],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return DependencyStatus(
            dependency=dependency,
            available=False,
            error="probe timed out",
        )
    except Exception as exc:
        return DependencyStatus(
            dependency=dependency,
            available=False,
            error=str(exc),
        )

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        return DependencyStatus(
            dependency=dependency,
            available=False,
            error=detail,
        )

    try:
        payload = json.loads(proc.stdout.strip() or "{}")
    except json.JSONDecodeError as exc:
        return DependencyStatus(
            dependency=dependency,
            available=False,
            error=f"bad probe output: {exc}",
        )

    return DependencyStatus(
        dependency=dependency,
        available=bool(payload.get("available")),
        version=payload.get("version"),
        error=payload.get("error"),
    )


def probe_dependencies(
    venv_dir: str | Path,
    *,
    platform: str | None = None,
    timeout: float = 2.5,
) -> list[DependencyStatus]:
    """Inspect all managed dependencies inside ``venv_dir`` in parallel.

    Each probe spawns its own ``python -c`` subprocess; running them
    sequentially used to block the calling worker thread for up to
    ``len(MANAGED_DEPENDENCIES) * timeout`` seconds on a slow venv (~10 s
    with the default 2.5 s timeout). The dependencies are independent so
    we parallelize with a small thread pool and preserve the input order.
    """
    deps = list(MANAGED_DEPENDENCIES)
    if not deps:
        return []
    # ``ThreadPoolExecutor.map`` preserves the input order and surfaces
    # per-call exceptions when the result is iterated. Cap workers so
    # we don't fork-bomb a venv when MANAGED_DEPENDENCIES grows.
    from concurrent.futures import ThreadPoolExecutor

    def _probe(dep: ManagedDependency) -> DependencyStatus:
        return probe_dependency_status(
            venv_dir,
            dep,
            platform=platform,
            timeout=timeout,
        )

    with ThreadPoolExecutor(max_workers=min(len(deps), 4)) as pool:
        return list(pool.map(_probe, deps))


def probe_current_python_dependencies(
    dependencies: tuple[ManagedDependency, ...] | None = None,
) -> tuple[DependencyStatus, ...]:
    """Inspect launcher-managed dependencies in the current Python process."""
    rows: list[DependencyStatus] = []
    # Same convention as ``build_bootstrap_venv_command``: an explicit
    # empty tuple/list is a legitimate "probe nothing" no-op; only ``None``
    # falls back to the managed default set.
    selected = MANAGED_DEPENDENCIES if dependencies is None else dependencies
    for dependency in selected:
        try:
            available = importlib.util.find_spec(dependency.import_name) is not None
        except Exception as exc:
            rows.append(
                DependencyStatus(
                    dependency=dependency,
                    available=False,
                    error=str(exc),
                )
            )
            continue

        version = None
        error = None
        if available:
            try:
                version = importlib.metadata.version(dependency.package_name)
            except importlib.metadata.PackageNotFoundError:
                version = None
            except Exception as exc:
                error = str(exc)
        rows.append(
            DependencyStatus(
                dependency=dependency,
                available=available,
                version=version,
                error=error,
            )
        )
    return tuple(rows)
