from __future__ import annotations

from pathlib import Path

from autoexperiments.agent import _tool_bash, _tool_edit_file
from autoexperiments.task_config import MetricConfig, TaskConfig


def _config() -> TaskConfig:
    return TaskConfig(
        name="test",
        run_command="echo metric: 1.0",
        time_budget=10,
        mutable_files=["mutable.py"],
        readonly_files=["readonly.py"],
        metric=MetricConfig(
            name="metric",
            direction="minimize",
            extract_pattern=r"^metric:\s+(\S+)",
        ),
    )


def test_edit_file_rejects_readonly_files(tmp_path: Path) -> None:
    (tmp_path / "readonly.py").write_text("value = 1\n")
    config = _config()

    out = _tool_edit_file(
        tmp_path,
        config,
        {"path": "readonly.py", "old_string": "value = 1", "new_string": "value = 2"},
    )

    assert "error" in out
    assert "read-only" in out["error"]
    assert (tmp_path / "readonly.py").read_text() == "value = 1\n"


def test_bash_blocks_write_redirection(tmp_path: Path) -> None:
    out = _tool_bash(tmp_path, {"command": "echo hacked > readonly.py"})

    assert "error" in out
    assert not (tmp_path / "readonly.py").exists()


def test_bash_allows_read_only_command(tmp_path: Path) -> None:
    out = _tool_bash(tmp_path, {"command": "pwd"})

    assert out["returncode"] == 0
