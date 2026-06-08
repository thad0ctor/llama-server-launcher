from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from modules.hf_downloader import helpers, runner


def test_normalize_repo_input_accepts_plain_repo_id():
    parsed = helpers.normalize_repo_input("TheBloke/Mistral-7B-GGUF")

    assert parsed.repo_id == "TheBloke/Mistral-7B-GGUF"
    assert parsed.revision_hint == ""


def test_normalize_repo_input_extracts_revision_from_url():
    parsed = helpers.normalize_repo_input(
        "https://huggingface.co/TheBloke/Mistral-7B-GGUF/tree/main"
    )

    assert parsed.repo_id == "TheBloke/Mistral-7B-GGUF"
    assert parsed.revision_hint == "main"


def test_normalize_repo_input_ignores_tree_subfolder_after_revision():
    parsed = helpers.normalize_repo_input(
        "https://huggingface.co/TheBloke/Mistral-7B-GGUF/tree/main/gguf"
    )

    assert parsed.repo_id == "TheBloke/Mistral-7B-GGUF"
    assert parsed.revision_hint == "main"


def test_normalize_repo_input_keeps_refs_pr_revision():
    parsed = helpers.normalize_repo_input(
        "https://huggingface.co/TheBloke/Mistral-7B-GGUF/tree/refs/pr/7/files"
    )

    assert parsed.repo_id == "TheBloke/Mistral-7B-GGUF"
    assert parsed.revision_hint == "refs/pr/7"


def test_parse_pattern_lines_accepts_commas_and_newlines():
    assert helpers.parse_pattern_lines("*.gguf,*.json\nREADME*") == (
        "*.gguf",
        "*.json",
        "README*",
    )


def test_collect_target_directory_options_defaults_to_first(tmp_path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()

    state = helpers.collect_target_directory_options([first, second])

    assert len(state.options) == 2
    assert state.options[0].selected is True
    assert state.options[1].selected is False
    assert state.selected_paths == (str(first.resolve()),)


def test_collect_target_directory_options_honors_saved_selection(tmp_path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()

    state = helpers.collect_target_directory_options(
        [first, second],
        selected_paths=(str(second.resolve()),),
    )

    assert state.options[0].selected is False
    assert state.options[1].selected is True
    assert state.selected_paths == (str(second.resolve()),)


def test_summarize_repo_listing_defaults_to_one_preferred_gguf_and_mmproj():
    refs, files = helpers.summarize_repo_listing(
        [{"name": "main", "kind": "branch"}],
        [
            {"path": "model.Q2_K.gguf", "size_bytes": 90, "kind": "gguf"},
            {"path": "model.Q4_K_M.gguf", "size_bytes": 100, "kind": "gguf"},
            {"path": "mmproj-model-f16.gguf", "size_bytes": 10, "kind": "mmproj"},
            {"path": "README.md", "size_bytes": 1, "kind": "other"},
        ],
    )

    assert refs[0].name == "main"
    assert [row.path for row in files if row.selected_by_default] == [
        "model.Q4_K_M.gguf",
        "mmproj-model-f16.gguf",
    ]


def test_build_runner_command_uses_module_entrypoint(tmp_path):
    python_path = tmp_path / "venv" / "bin" / "python"
    argv = helpers.build_runner_command(python_path, "list", tmp_path / "payload.json")

    # ``build_runner_command`` round-trips paths through ``Path(...)``, which
    # on Windows substitutes backslashes. Compare against the same
    # platform-normalized representation rather than the raw POSIX literal.
    assert argv == [
        str(python_path),
        "-m",
        "modules.hf_downloader.runner",
        "list",
        str(tmp_path / "payload.json"),
    ]


def test_payload_bytes_are_json():
    payload = {"repo_id": "TheBloke/Test", "revision": "main"}

    data = helpers.payload_bytes(payload)

    assert json.loads(data.decode("utf-8")) == payload


def test_runner_list_emits_refs_and_files(monkeypatch, capsys):
    class FakeRef:
        def __init__(self, name, target_commit="abc123"):
            self.name = name
            self.target_commit = target_commit

    class FakeSibling:
        def __init__(self, path, size):
            self.rfilename = path
            self.size = size

    class FakeInfo:
        sha = "deadbeef"
        siblings = [
            FakeSibling("model.gguf", 123),
            FakeSibling("README.md", 4),
            FakeSibling("mmproj-myname.gguf", 64),
        ]

    class FakeApi:
        def list_repo_refs(self, repo_id, repo_type=None, token=None):
            assert repo_id == "TheBloke/Test"
            return SimpleNamespace(branches=[FakeRef("main")], tags=[FakeRef("v1", "ff00")])

        def model_info(self, repo_id, revision=None, files_metadata=True, token=None):
            assert repo_id == "TheBloke/Test"
            assert revision == "main"
            return FakeInfo()

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi))

    rc = runner.run_list({"repo_id": "TheBloke/Test", "revision": "main", "token": ""})

    assert rc == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines[-2]["event"] == "listing"
    assert lines[-2]["resolved_revision"] == "deadbeef"
    assert lines[-2]["refs"][0]["name"] == "main"
    files_by_path = {row["path"]: row for row in lines[-2]["files"]}
    assert set(files_by_path) == {"README.md", "mmproj-myname.gguf", "model.gguf"}
    # mmproj must classify as ``mmproj`` (not generic ``gguf``) so the UI
    # picks them up as multimodal projectors instead of as another weight
    # quant.
    assert files_by_path["mmproj-myname.gguf"]["kind"] == "mmproj"
    assert files_by_path["model.gguf"]["kind"] == "gguf"


def test_runner_download_passes_patterns_and_targets(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=fake_snapshot_download))

    target = tmp_path / "models"
    rc = runner.run_download(
        {
            "repo_id": "TheBloke/Test",
            "revision": "main",
            "download_mode": "selected",
            "selected_files": ["model.gguf"],
            "include_patterns": ["README*"],
            "ignore_patterns": ["*.tmp"],
            "force_download": True,
            "local_files_only": False,
            "max_workers": 3,
            "target_dirs": [str(target)],
            "token": "",
        }
    )

    assert rc == 0
    assert len(calls) == 1
    assert calls[0]["allow_patterns"] == ["model.gguf", "README*"]
    assert calls[0]["ignore_patterns"] == ["*.tmp"]
    assert calls[0]["repo_type"] == "model"
    assert calls[0]["local_dir"] == target
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines[0]["event"] == "target-start"
    assert lines[-1]["event"] == "complete"
