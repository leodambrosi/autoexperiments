from __future__ import annotations

from pathlib import Path

from autoexperiments.tracker import ExperimentTracker


def test_config_snapshot_is_decoded_to_dict(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path / "experiments.db")
    try:
        tracker.log(
            commit="abc1234",
            metric_name="metric",
            metric_value=1.23,
            status="keep",
            config_snapshot={"train.py": "print('ok')\n"},
        )
        rec = tracker.history(1)[0]
    finally:
        tracker.close()

    assert isinstance(rec.config_snapshot, dict)
    assert rec.config_snapshot["train.py"] == "print('ok')\n"
