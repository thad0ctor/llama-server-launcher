"""Helpers for launcher-managed Python virtual environments."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Sequence
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
        """True only when the target really looks like a venv.

        Any plain file at ``bin/python`` / ``Scripts/python.exe`` used to
        satisfy this check, which let a non-venv project directory be
        auto-activated by ``resolve_active_venv_path()`` and even targeted
        for recursive deletion by ``SettingsTab._on_remove_venv``. Now we
        also require:

          * ``pyvenv.cfg`` to exist (the file ``python -m venv`` writes
            into every freshly created environment), and
          * the platform-appropriate activator script
            (``bin/activate`` on POSIX, ``Scripts/activate.bat`` /
            ``Scripts/Activate.ps1`` on Windows) to exist.

        Together these are reliable enough to gate the activate/delete
        paths without misfiring on a folder that happens to contain a
        ``python`` symlink.
        """
        if self.python_path is None:
            return False
        # Mirror the stricter interpreter check used by
        # ``_path_looks_like_venv``: an actual regular file, and
        # executable on POSIX. A symlink-to-directory or non-executable
        # marker file at ``bin/python`` used to pass this gate, which
        # let activation/auto-selection target the wrong directory.
        if not self.python_path.is_file():
            return False
        name = self.python_path.name.lower()
        if not name.endswith(".exe"):
            # POSIX: require +x. Windows: ``.exe`` is the executable
            # marker (``os.access(X_OK)`` is unreliable there).
            if not os.access(str(self.python_path), os.X_OK):
                return False
        if not (self.effective_dir / "pyvenv.cfg").is_file():
            return False
        # Different layouts ship different activators (bash vs cmd vs PS,
        # bin/ vs Scripts/). Accept any one of them so this works on
        # POSIX, Windows, and the occasional Cygwin / MSYS layout.
        activator_candidates = (
            self.effective_dir / "bin" / "activate",
            self.effective_dir / "Scripts" / "activate.bat",
            self.effective_dir / "Scripts" / "Activate.ps1",
            self.effective_dir / "Scripts" / "activate",
        )
        return any(p.is_file() for p in activator_candidates)


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


def venv_python_candidates(venv_dir: str | Path, *, platform: str | None = None) -> tuple[Path, ...]:
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


def locate_venv_python(venv_dir: str | Path, *, platform: str | None = None) -> Path | None:
    """Return the first existing Python interpreter inside ``venv_dir``.

    Mirrors the stricter check used by ``VenvTargetInfo.looks_like_venv``
    and ``_path_looks_like_venv``: a non-executable file at ``bin/python``
    used to be returned as the interpreter, and downstream callers
    (``build_install_dependency_command``, ``build_remove_dependency_command``,
    ``probe_dependency_status``) would then run subprocess against a
    file that isn't actually executable and fail with a confusing
    ``PermissionError``. Require an executable interpreter on POSIX;
    on Windows the ``.exe`` extension is the executable marker.
    """
    for candidate in venv_python_candidates(venv_dir, platform=platform):
        if not candidate.is_file():
            continue
        # ``.exe`` is the Windows executable marker; ``os.access(X_OK)``
        # is unreliable there. Everywhere else require the bit.
        if candidate.name.lower().endswith(".exe"):
            return candidate
        if os.access(str(candidate), os.X_OK):
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

    NOTE: ``cmd.exe`` STILL performs ``%VAR%`` expansion inside double-quoted
    strings (and ``!VAR!`` when delayed expansion is enabled). Quoting alone
    can't disarm those — a path like ``%TEMP%\\evil`` gets rewritten before
    the target command sees it. The caller in :func:`_shell_join` therefore
    refuses Windows args containing ``%`` or ``!`` outright; this helper
    handles the no-expansion metacharacter set.
    """
    return '"' + arg.replace('"', '""') + '"'


_WIN_CMD_EXPANSION_CHARS = ("%", "!")


def _shell_join(
    args: list[str],
    *,
    platform: str | None = None,
    quote_cmd_word: bool = True,
) -> str:
    """Compose a shell command string from an argv list with the right
    quoting rules for the target platform.

    ``quote_cmd_word=False`` leaves the FIRST argument unquoted so
    cmd.exe builtins like ``rmdir`` / ``del`` are still recognised as
    builtins (a quoted ``"rmdir"`` makes cmd look for an executable
    file with that name first). The ``%`` / ``!`` rejection and the
    quoting rules for the remaining arguments are unchanged.
    """
    plat = platform or sys.platform
    if plat.startswith("win"):
        # cmd.exe expands ``%VAR%`` inside double-quoted strings and ``!VAR!``
        # when delayed expansion is enabled — neither is disarmed by
        # quoting. Reject early instead of emitting a single ``cmd.exe``
        # string that could silently rewrite the target path
        # (``rmdir /s /q "%TEMP%\foo"`` becomes ``rmdir /s /q "C:\Users\u\...\foo"``).
        # Callers that genuinely need a path containing these chars must
        # switch to an args-list / ``shell=False`` Popen invocation instead
        # of using this single-string composer.
        for arg in args:
            if any(ch in arg for ch in _WIN_CMD_EXPANSION_CHARS):
                raise ValueError(
                    "Refusing to emit a cmd.exe command containing "
                    f"variable-expansion characters ('%' or '!'): {arg!r}. "
                    "cmd.exe expands these even inside quoted strings, which "
                    "can retarget the command after the path has been "
                    "safety-checked."
                )
        # Always double-quote each arg so embedded cmd metacharacters
        # (& | ^ < > and friends) can't break out of the intended
        # command. See _win_cmd_quote for the rationale. The optional
        # ``quote_cmd_word=False`` leaves the first arg (the command
        # name) unquoted for cmd.exe builtin compatibility.
        if quote_cmd_word or not args:
            return " ".join(_win_cmd_quote(arg) for arg in args)
        return " ".join([args[0], *(_win_cmd_quote(a) for a in args[1:])])
    if quote_cmd_word or not args:
        return " ".join(shlex.quote(arg) for arg in args)
    return " ".join([args[0], *(shlex.quote(a) for a in args[1:])])


def default_venv_base_python_args(*, platform: str | None = None) -> tuple[str, ...]:
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
    """Return a shell command that creates ``venv_dir``.

    Normalizes ``venv_dir`` through :func:`resolve_venv_dir` so this helper
    honors the module's documented "blank means ``<repo>/venv``" and
    "relative means repo-relative" contract instead of building shell
    commands against the literal UI value (which would silently target the
    current working directory for ``""`` and ``"."``).
    """
    target = resolve_venv_dir(str(venv_dir), repo_dir=launcher_repo_dir())
    if base_python is None:
        python_args = list(default_venv_base_python_args(platform=platform))
    elif isinstance(base_python, (tuple, list)):
        python_args = list(base_python)
    else:
        python_args = [base_python]
    args = [*python_args, "-m", "venv", str(target)]
    return _shell_join(args, platform=platform)


def build_bootstrap_venv_command(
    venv_dir: str | Path,
    *,
    dependencies: Sequence[ManagedDependency] | None = None,
    base_python: str | tuple[str, ...] | list[str] | None = None,
    platform: str | None = None,
) -> str:
    """Return a shell command that creates a venv and installs managed packages."""
    target = resolve_venv_dir(str(venv_dir), repo_dir=launcher_repo_dir())
    create_cmd = build_create_venv_command(
        target,
        base_python=base_python,
        platform=platform,
    )
    python_path = venv_python_candidates(target, platform=platform)[0]
    # An explicit ``dependencies=()`` (or ``[]``) must be honored as "no
    # managed installs" — using ``or`` collapsed empty tuples to the
    # required-deps default and silently overrode the caller's intent.
    install_list = required_managed_dependencies() if dependencies is None else dependencies
    packages = [dep.install_name or dep.package_name for dep in install_list]
    # ``commands`` starts as just the venv create. The pip upgrade
    # and the install line only get appended when there are packages
    # to install. Without this gate, a caller passing
    # ``dependencies=()`` (genuinely "create venv only, no managed
    # installs") would still emit ``python -m pip install --upgrade
    # pip`` and require network access — exactly the offline /
    # "create only" flow that empty-deps was meant to support.
    commands = [create_cmd]
    if packages:
        commands.append(
            _shell_join(
                [str(python_path), "-m", "pip", "install", "--upgrade", "pip"],
                platform=platform,
            )
        )
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
    target = resolve_venv_dir(str(venv_dir), repo_dir=launcher_repo_dir())
    python = locate_venv_python(target, platform=platform)
    if python is None:
        python = venv_python_candidates(target, platform=platform)[0]
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
    target = resolve_venv_dir(str(venv_dir), repo_dir=launcher_repo_dir())
    python = locate_venv_python(target, platform=platform)
    if python is None:
        python = venv_python_candidates(target, platform=platform)[0]
    args = [str(python), "-m", "pip", "uninstall", "-y", dependency.package_name]
    return _shell_join(args, platform=platform)


def _path_looks_like_venv(path: Path) -> bool:
    """Standalone version of ``VenvTargetInfo.looks_like_venv``.

    The dataclass property needs a constructed ``VenvTargetInfo`` (with
    a python_path that's already been located); this helper checks an
    arbitrary Path directly for the same three markers (python + pyvenv.cfg
    + activator) so command builders that only have a resolved path can
    apply the same gate.

    Reuses ``locate_venv_python`` for the interpreter check so this
    helper, ``VenvTargetInfo.looks_like_venv``, and every command
    builder agree on which paths count as a real venv — otherwise a
    directory accepted by the Settings tab could be rejected by
    ``build_remove_venv_command`` (or vice versa) and the user would
    see an opaque ``ValueError`` bubble through a Tk callback.
    """
    if not path.is_dir():
        return False
    if not (path / "pyvenv.cfg").is_file():
        return False
    if locate_venv_python(path) is None:
        return False
    activator_candidates = (
        path / "bin" / "activate",
        path / "Scripts" / "activate.bat",
        path / "Scripts" / "Activate.ps1",
        path / "Scripts" / "activate",
    )
    return any(p.is_file() for p in activator_candidates)


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
    # ``resolve_venv_dir`` applies the module's repo-relative + expanduser
    # normalization, so a relative entry like ``"venv"`` or ``"~/.venvs/x"``
    # resolves the same way the rest of the module sees it. The safety
    # check below then compares the resolved Path against cwd / home /
    # repo / filesystem root — all the targets we refuse to ``rm -rf``.
    try:
        resolved_target = resolve_venv_dir(raw, repo_dir=launcher_repo_dir())
    except OSError:
        resolved_target = Path(raw).expanduser()
    try:
        cwd_resolved = Path.cwd().resolve(strict=False)
    except OSError:
        cwd_resolved = Path.cwd()
    try:
        home_resolved = Path.home().resolve(strict=False)
    except (OSError, RuntimeError):
        home_resolved = Path.home()
    try:
        repo_resolved = launcher_repo_dir().resolve(strict=False)
    except OSError:
        repo_resolved = launcher_repo_dir()
    # Reject the repo root too: ``resolve_venv_dir("")`` and
    # ``resolve_venv_dir(".")`` both normalize to the repo root, so without
    # this guard a Remove venv triggered before the user picked a venv path
    # would emit ``rm -rf <repo>`` / ``rmdir /s /q <repo>``.
    if resolved_target in {cwd_resolved, home_resolved, repo_resolved}:
        raise ValueError(f"Refusing to remove unsafe venv path: {raw!r}")
    if resolved_target == Path(resolved_target.anchor):
        # Anchor is the filesystem root (``/`` on POSIX, ``C:\`` on Windows).
        raise ValueError(f"Refusing to remove filesystem root: {raw!r}")
    # If the path EXISTS, additionally require it to look like a real venv
    # (pyvenv.cfg + a python interpreter + an activator) before emitting a
    # recursive delete. Without this guard, a user pointing ``venv_dir``
    # at an arbitrary directory the launcher previously auto-detected
    # could have that directory recursively removed. ``resolved_target``
    # not existing is fine — ``rm -rf`` on a missing path is a no-op and
    # the GUI already rejects that case earlier.
    if resolved_target.exists() and not _path_looks_like_venv(resolved_target):
        raise ValueError(
            f"Refusing to remove {raw!r}: target exists but does not look "
            "like a venv (missing pyvenv.cfg, activator, or python "
            "interpreter)."
        )
    target = str(resolved_target)
    plat = platform or sys.platform
    if plat.startswith("win"):
        # Route through ``_shell_join`` so the args get the same ``%``/``!``
        # rejection every other Windows command builder gets. Without this
        # a path that passed the safety-resolution above could still be
        # rewritten by ``cmd.exe`` variable expansion before ``rmdir``
        # executes — i.e. the deletion would target a different directory
        # than the one we validated. ``quote_cmd_word=False`` keeps
        # ``rmdir`` unquoted so cmd.exe resolves it as a builtin
        # (a quoted ``"rmdir"`` makes cmd search for a binary first).
        return _shell_join(
            ["rmdir", "/s", "/q", target],
            platform=plat,
            quote_cmd_word=False,
        )
    return _shell_join(["rm", "-rf", target], platform=plat)


def probe_dependency_status(
    venv_dir: str | Path,
    dependency: ManagedDependency,
    *,
    platform: str | None = None,
    timeout: float = 2.5,
) -> DependencyStatus:
    """Inspect one dependency inside a venv without importing it into the GUI."""
    # Normalize the venv path the same way the command builders do, so a
    # blank/relative UI value isn't probed against the launcher's cwd.
    target = resolve_venv_dir(str(venv_dir), repo_dir=launcher_repo_dir())
    python = locate_venv_python(target, platform=platform)
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
    dependencies: Sequence[ManagedDependency] | None = None,
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
