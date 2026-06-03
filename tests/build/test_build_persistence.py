import subprocess
import sys
from pathlib import Path

from modules.build.build_persistence import BuildConfig


# Repo root = three levels up from this file (tests/build/test_*.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_build_package_loads_build_tab_lazily():
    code = """
import importlib
import sys

pkg = importlib.import_module("modules.build")
assert "modules.build.build_tab" not in sys.modules
from modules.build import BuildTab
assert BuildTab.__name__ == "BuildTab"
assert "modules.build.build_tab" in sys.modules
"""
    # Pin cwd to the repo root so the child can import ``modules`` regardless
    # of where pytest was launched from. Previously this depended on the
    # parent process CWD, which broke when running from outside the repo.
    result = subprocess.run([sys.executable, "-c", code], text=True, cwd=_REPO_ROOT)

    assert result.returncode == 0


def test_build_config_bool_fields_parse_persisted_strings():
    cfg = BuildConfig.from_json(
        "strings",
        {
            "git_pull_before_build": "false",
            "clean_build": "0",
        },
    )

    assert cfg.git_pull_before_build is False
    assert cfg.clean_build is False


def test_build_config_bool_fields_accept_truthy_strings():
    cfg = BuildConfig.from_json(
        "strings",
        {
            "git_pull_before_build": "yes",
            "clean_build": "on",
        },
    )

    assert cfg.git_pull_before_build is True
    assert cfg.clean_build is True
