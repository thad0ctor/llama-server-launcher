from __future__ import annotations

from types import SimpleNamespace

from modules import venv_manager


def test_default_venv_dir_uses_repo_folder(tmp_path):
    assert venv_manager.default_venv_dir(repo_dir=tmp_path) == tmp_path / "venv"


def test_resolve_venv_dir_blank_uses_repo_default(tmp_path):
    assert venv_manager.resolve_venv_dir("", repo_dir=tmp_path) == (tmp_path / "venv").resolve()


def test_resolve_venv_dir_relative_uses_repo_dir(tmp_path):
    assert venv_manager.resolve_venv_dir("envs/dev", repo_dir=tmp_path) == (
        tmp_path / "envs" / "dev"
    ).resolve()


def test_describe_venv_target_marks_blank_as_default(tmp_path):
    info = venv_manager.describe_venv_target("", repo_dir=tmp_path, platform="linux")

    assert info.uses_default is True
    assert info.effective_dir == (tmp_path / "venv").resolve()
    assert info.looks_like_venv is False


def test_resolve_active_venv_path_blank_stays_empty_without_default_venv(tmp_path):
    assert venv_manager.resolve_active_venv_path("", repo_dir=tmp_path, platform="linux") == ""


def test_resolve_active_venv_path_blank_uses_default_when_venv_exists(tmp_path):
    bindir = tmp_path / "venv" / "bin"
    bindir.mkdir(parents=True)
    (bindir / "python").write_text("", encoding="utf-8")

    resolved = venv_manager.resolve_active_venv_path("", repo_dir=tmp_path, platform="linux")

    assert resolved == str((tmp_path / "venv").resolve())


def test_resolve_active_venv_path_relative_is_repo_relative(tmp_path):
    resolved = venv_manager.resolve_active_venv_path(
        "envs/dev",
        repo_dir=tmp_path,
        platform="linux",
    )

    assert resolved == str((tmp_path / "envs" / "dev").resolve())


def test_locate_venv_python_prefers_windows_scripts_path(tmp_path):
    scripts = tmp_path / "Scripts"
    scripts.mkdir(parents=True)
    exe = scripts / "python.exe"
    exe.write_text("", encoding="utf-8")

    found = venv_manager.locate_venv_python(tmp_path, platform="win32")

    assert found == exe


def test_locate_venv_python_prefers_posix_bin_path(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir(parents=True)
    exe = bindir / "python"
    exe.write_text("", encoding="utf-8")

    found = venv_manager.locate_venv_python(tmp_path, platform="linux")

    assert found == exe


def test_build_create_venv_command_quotes_windows_paths(tmp_path):
    target = tmp_path / "my env"
    command = venv_manager.build_create_venv_command(
        target,
        base_python=r"C:\Python 3.12\python.exe",
        platform="win32",
    )

    assert '"C:\\Python 3.12\\python.exe"' in command
    assert f'"{target}"' in command


def test_default_venv_base_python_args_prefers_python3_on_posix(monkeypatch):
    monkeypatch.setattr(
        venv_manager.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name == "python3" else None,
    )

    assert venv_manager.default_venv_base_python_args(platform="linux") == ("python3",)


def test_default_venv_base_python_args_prefers_py_launcher_on_windows(monkeypatch):
    monkeypatch.setattr(
        venv_manager.shutil,
        "which",
        lambda name: rf"C:\Windows\{name}.exe" if name == "py" else None,
    )

    assert venv_manager.default_venv_base_python_args(platform="win32") == ("py", "-3")


def test_build_bootstrap_venv_command_installs_managed_packages(tmp_path):
    command = venv_manager.build_bootstrap_venv_command(tmp_path / "venv", platform="linux")

    assert "python3 -m venv" in command or "python -m venv" in command
    assert "pip install --upgrade pip" in command
    assert "pip install requests" in command
    assert "torch" not in command


def test_probe_current_python_dependencies_reports_availability(monkeypatch):
    dep = venv_manager.MANAGED_DEPENDENCIES[0]
    monkeypatch.setattr(
        venv_manager.importlib.util,
        "find_spec",
        lambda name: object() if name == dep.import_name else None,
    )
    monkeypatch.setattr(
        venv_manager.importlib.metadata,
        "version",
        lambda name: "2.32.0" if name == dep.package_name else None,
    )

    statuses = venv_manager.probe_current_python_dependencies((dep,))

    assert len(statuses) == 1
    assert statuses[0].available is True
    assert statuses[0].version == "2.32.0"


def test_build_install_dependency_command_uses_venv_python(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir(parents=True)
    exe = bindir / "python"
    exe.write_text("", encoding="utf-8")

    command = venv_manager.build_install_dependency_command(
        tmp_path,
        venv_manager.MANAGED_DEPENDENCIES[0],
        platform="linux",
    )

    assert str(exe) in command
    assert "pip install requests" in command


def test_build_install_dependency_command_quotes_windows_python(tmp_path):
    # Every arg is force-quoted on Windows now (defence against cmd
    # metacharacter injection through paths containing ``&``/``|``/etc.).
    base = tmp_path / "venv with space"
    scripts = base / "Scripts"
    scripts.mkdir(parents=True)
    exe = scripts / "python.exe"
    exe.write_text("", encoding="utf-8")

    command = venv_manager.build_install_dependency_command(
        base,
        venv_manager.MANAGED_DEPENDENCIES[3],
        platform="win32",
    )

    assert f'"{exe}"' in command
    # huggingface_hub[cli] is the actual install_name; bare "huggingface_hub"
    # would skip the entry-point. Both forms must round-trip quoted.
    assert '"pip" "install"' in command
    assert "huggingface_hub" in command


def test_build_remove_venv_command_windows_uses_rmdir(tmp_path):
    command = venv_manager.build_remove_venv_command(tmp_path / "my env", platform="win32")

    # Every cmd arg is quoted to neutralize metacharacters in path strings;
    # see _win_cmd_quote in modules/venv_manager.py.
    assert command.startswith('"rmdir" "/s" "/q" ')
    assert f'"{tmp_path / "my env"}"' in command


def test_build_remove_venv_command_neutralizes_cmd_metacharacters(tmp_path):
    """A path with ``&`` must not be able to break out of rmdir on Windows."""
    evil = tmp_path / "proj&work" / "venv"

    command = venv_manager.build_remove_venv_command(evil, platform="win32")

    # The full path must appear inside a single quoted token, with no
    # bare ``&`` outside quotes that cmd would parse as a separator.
    assert f'"{evil}"' in command
    # No unquoted ``&`` between args.
    assert "& " not in command.replace(f'"{evil}"', "")


def test_build_remove_venv_command_posix_uses_rm_rf(tmp_path):
    command = venv_manager.build_remove_venv_command(tmp_path / "my env", platform="linux")

    assert command.startswith("rm -rf ")
    assert str(tmp_path / "my env") in command


def test_probe_dependency_status_reports_missing_python(tmp_path):
    dep = venv_manager.MANAGED_DEPENDENCIES[0]

    status = venv_manager.probe_dependency_status(tmp_path, dep, platform="linux")

    assert status.available is False
    assert status.error == "venv python not found"


def test_probe_dependency_status_parses_success(monkeypatch, tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir(parents=True)
    exe = bindir / "python"
    exe.write_text("", encoding="utf-8")
    dep = venv_manager.MANAGED_DEPENDENCIES[1]

    def fake_run(args, **kwargs):
        assert args[0] == str(exe)
        return SimpleNamespace(
            returncode=0,
            stdout='{"available": true, "version": "2.7.0", "error": null}\n',
            stderr="",
        )

    monkeypatch.setattr(venv_manager.subprocess, "run", fake_run)

    status = venv_manager.probe_dependency_status(tmp_path, dep, platform="linux")

    assert status.available is True
    assert status.version == "2.7.0"
    assert status.error is None


def test_probe_dependency_status_nonzero_return_surfaces_error(monkeypatch, tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir(parents=True)
    exe = bindir / "python"
    exe.write_text("", encoding="utf-8")
    dep = venv_manager.MANAGED_DEPENDENCIES[0]

    monkeypatch.setattr(
        venv_manager.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="ModuleNotFoundError: requests",
        ),
    )

    status = venv_manager.probe_dependency_status(tmp_path, dep, platform="linux")

    assert status.available is False
    assert "ModuleNotFoundError" in status.error
