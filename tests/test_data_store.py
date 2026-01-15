import json
import subprocess
from pathlib import Path

import pytest


@pytest.fixture()
def shellock_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "shellock-home"
    monkeypatch.setenv("SHELLOCK_HOME", str(home))
    return home


@pytest.fixture()
def shellock_module(shellock_home: Path):
    # Import after SHELLOCK_HOME is set.
    import shellock  # type: ignore

    return shellock


def test_explain_creates_command_file(
    shellock_module, shellock_home: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = {"count": 0}

    def fake_run(cmd, capture_output, text, timeout, env=None):
        calls["count"] += 1
        cmd_str = " ".join(cmd)
        if "tree" in cmd_str and "--help" in cmd_str:
            return subprocess.CompletedProcess(
                cmd, 0, stdout="  -a  All files\n  -L level  Depth\n", stderr=""
            )
        # No man pages
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

    monkeypatch.setattr(shellock_module.subprocess, "run", fake_run)

    out = shellock_module.format_explanations(
        explanations=shellock_module.explain_command(
            protocol=shellock_module._default_protocol(), cmdline="tree -a -L 2"
        ),
        use_color=False,
    )
    assert "-a" in out

    data_path = shellock_home / "data" / "tree.json"
    assert data_path.exists()

    payload = json.loads(data_path.read_text())
    assert payload["command"] == "tree"
    assert "flags" in payload

    # Subsequent run should not need subprocess again.
    calls["count"] = 0
    shellock_module.explain_command(
        protocol=shellock_module._default_protocol(), cmdline="tree -a"
    )
    assert calls["count"] == 0


def test_refresh_command(
    shellock_module, shellock_home: Path, monkeypatch: pytest.MonkeyPatch
):
    help_text_v1 = "  -a  All files\n"
    help_text_v2 = "  -a  All files\n  -d  Dirs only\n"

    state = {"version": 1}

    def fake_run(cmd, capture_output, text, timeout, env=None):
        cmd_str = " ".join(cmd)
        if "tree" in cmd_str and "--help" in cmd_str:
            txt = help_text_v1 if state["version"] == 1 else help_text_v2
            return subprocess.CompletedProcess(cmd, 0, stdout=txt, stderr="")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

    monkeypatch.setattr(shellock_module.subprocess, "run", fake_run)

    protocol = shellock_module._default_protocol()
    shellock_module.ensure_command_scanned(
        protocol=protocol, command="tree", refresh=False
    )

    data_path = shellock_home / "data" / "tree.json"
    payload = json.loads(data_path.read_text())
    assert "-d" not in payload["flags"]

    state["version"] = 2
    shellock_module.refresh_command(protocol=protocol, command="tree")

    payload = json.loads(data_path.read_text())
    assert "-d" in payload["flags"]
