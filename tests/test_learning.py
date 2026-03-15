from __future__ import annotations

from autoexperiments.learning import (
    classify_family,
    format_learning_summary,
    learning_payload,
    summarize_learning,
)
from autoexperiments.tracker import ExperimentRecord


def _record(
    i: int,
    status: str,
    metric: float | None,
    description: str,
) -> ExperimentRecord:
    return ExperimentRecord(
        id=i,
        timestamp=0.0 + i,
        commit=f"c{i}",
        parent_commit=None,
        metric_name="eval_loss",
        metric_value=metric,
        constraints={},
        status=status,
        description=description,
        wall_seconds=10.0,
        config_snapshot={},
    )


def test_classify_family_keywords() -> None:
    assert classify_family("Increase LoRA rank and alpha") == "adapter_peft"
    assert classify_family("Reduce gradient accumulation steps") == "batching_steps"
    assert classify_family("Change AdamW optimizer and weight decay") == "optimizer"


def test_summarize_learning_prioritize_and_avoid() -> None:
    records = [
        _record(1, "keep", 0.700, "Reduce gradient accumulation steps"),
        _record(2, "discard", 0.710, "Increase batch size"),
        _record(3, "discard", 0.730, "Increase dropout to 0.1"),
        _record(4, "discard", 0.735, "Increase dropout to 0.2"),
        _record(5, "discard", 0.740, "Apply stronger regularization"),
    ]
    summary = summarize_learning(records, direction="minimize", last_n=20)

    assert summary.best_commit == "c1"
    assert summary.best_metric == 0.700
    assert "batching_steps" in summary.prioritize
    assert "regularization" in summary.avoid


def test_learning_payload_and_format() -> None:
    records = [
        _record(1, "keep", 0.680, "Use LoRA on later layers"),
        _record(2, "discard", 0.690, "Tune warmup ratio"),
    ]
    payload = learning_payload(
        records=records,
        direction="minimize",
        metric_name="eval_loss",
        metric_format=".6f",
    )

    assert payload["best_commit"] == "c1"
    assert payload["best_metric"] == 0.680
    assert isinstance(payload["families"], list)
    text = format_learning_summary(
        summarize_learning(records, direction="minimize"),
        metric_name="eval_loss",
        metric_format=".6f",
    )
    assert "best so far:" in text
