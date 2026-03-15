from __future__ import annotations

from pathlib import Path

from autoexperiments.runner import run_and_record, run_experiment
from autoexperiments.task_config import MetricConfig, TaskConfig
from autoexperiments.tracker import ExperimentTracker


def _config(command: str, time_budget: int = 10) -> TaskConfig:
    return TaskConfig(
        name="test",
        run_command=command,
        time_budget=time_budget,
        mutable_files=[],
        metric=MetricConfig(
            name="metric",
            direction="minimize",
            extract_pattern=r"^metric:\s+(\S+)",
        ),
    )


def test_timeout_enforced_for_silent_process(tmp_path: Path) -> None:
    config = _config("python3 -c \"import time; time.sleep(3)\"", time_budget=1)

    result = run_experiment(config, tmp_path)

    assert result.status == "timeout"
    assert result.wall_seconds < 3.0


def test_run_and_record_keeps_null_metric_as_null(tmp_path: Path) -> None:
    config = _config("echo no_metric_here")
    tracker = ExperimentTracker(tmp_path / ".autoexp" / "experiments.db")
    try:
        result, status, improved = run_and_record(config, tmp_path, tracker, description="missing metric")
        row = tracker.history(1)[0]
    finally:
        tracker.close()

    assert result.metric is None
    assert status == "crash"
    assert improved is False
    assert row.metric_value is None
