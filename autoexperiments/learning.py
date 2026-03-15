"""
Systematic learning from experiment history.

This module turns raw experiment logs into:
- Family-level win/loss statistics
- Stable "what to try next" guidance
- Compact text summaries for agents and humans
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .tracker import ExperimentRecord


FAMILY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "optimizer": ("adam", "adamw", "lion", "adafactor", "sgd", "optimizer", "weight decay"),
    "lr_schedule": ("learning rate", "lr", "warmup", "schedule", "cosine", "one-cycle", "decay"),
    "batching_steps": ("batch", "gradient accumulation", "grad_accum", "micro-step", "steps"),
    "adapter_peft": ("lora", "ia3", "adalora", "peft", "rank", "alpha", "target modules"),
    "regularization": ("dropout", "label smoothing", "focal", "ema", "neftune", "clip_grad", "regulariz"),
    "architecture_layers": ("layer", "freeze", "unfreeze", "projection", "q_proj", "k_proj", "v_proj"),
    "training_loop": ("epoch", "curriculum", "loss function", "training loop", "checkpointing"),
    "data_pipeline": ("dataset", "token", "mask", "prompt format", "data"),
}

FAMILY_PRIORITY = [
    "adapter_peft",
    "batching_steps",
    "lr_schedule",
    "optimizer",
    "regularization",
    "architecture_layers",
    "training_loop",
    "data_pipeline",
    "other",
]


@dataclass
class FamilyStats:
    family: str
    trials: int = 0
    keeps: int = 0
    discards: int = 0
    failures: int = 0
    best_metric: float | None = None
    last_description: str = ""


@dataclass
class LearningSummary:
    total_trials: int
    kept_trials: int
    best_commit: str | None
    best_metric: float | None
    family_stats: list[FamilyStats]
    prioritize: list[str]
    avoid: list[str]
    underexplored: list[str]


def classify_family(description: str) -> str:
    text = description.lower()
    scores: dict[str, int] = {}
    for family, keywords in FAMILY_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in text:
                score += 2 if " " in keyword else 1
        if score > 0:
            scores[family] = score
    if not scores:
        return "other"
    ordered = sorted(
        scores.items(),
        key=lambda kv: (
            -kv[1],
            FAMILY_PRIORITY.index(kv[0]) if kv[0] in FAMILY_PRIORITY else len(FAMILY_PRIORITY),
        ),
    )
    return ordered[0][0]


def _is_failure_status(status: str) -> bool:
    return status in {"crash", "timeout", "constraint_violated"}


def summarize_learning(
    records: list[ExperimentRecord],
    direction: str,
    last_n: int = 40,
) -> LearningSummary:
    """Build deterministic guidance from recent experiment history."""
    if last_n <= 0:
        last_n = 1
    window = records[-last_n:]

    best_keep: ExperimentRecord | None = None
    family_map: dict[str, FamilyStats] = {}
    kept_trials = 0
    total_trials = 0

    for r in window:
        if r.status not in {"keep", "discard", "crash", "timeout", "constraint_violated"}:
            continue
        total_trials += 1

        family = classify_family(r.description or "")
        stats = family_map.setdefault(family, FamilyStats(family=family))
        stats.trials += 1
        stats.last_description = r.description or stats.last_description

        if r.status == "keep":
            kept_trials += 1
            stats.keeps += 1
            if r.metric_value is not None:
                if stats.best_metric is None:
                    stats.best_metric = r.metric_value
                elif direction == "minimize" and r.metric_value < stats.best_metric:
                    stats.best_metric = r.metric_value
                elif direction == "maximize" and r.metric_value > stats.best_metric:
                    stats.best_metric = r.metric_value

            if best_keep is None:
                best_keep = r
            elif (
                r.metric_value is not None
                and best_keep.metric_value is not None
                and (
                    (direction == "minimize" and r.metric_value < best_keep.metric_value)
                    or (direction == "maximize" and r.metric_value > best_keep.metric_value)
                )
            ):
                best_keep = r
        elif r.status == "discard":
            stats.discards += 1
        elif _is_failure_status(r.status):
            stats.failures += 1

    family_stats = sorted(family_map.values(), key=lambda s: (-s.keeps, -s.trials, s.family))

    prioritize = [
        s.family
        for s in family_stats
        if s.keeps > 0 and s.trials <= max(6, s.keeps * 4)
    ]
    avoid = [
        s.family
        for s in family_stats
        if s.keeps == 0 and s.trials >= 3
    ]
    underexplored = [
        s.family
        for s in family_stats
        if s.trials <= 1 and s.family not in avoid
    ]

    return LearningSummary(
        total_trials=total_trials,
        kept_trials=kept_trials,
        best_commit=best_keep.commit if best_keep else None,
        best_metric=best_keep.metric_value if best_keep else None,
        family_stats=family_stats,
        prioritize=prioritize[:3],
        avoid=avoid[:4],
        underexplored=underexplored[:4],
    )


def format_learning_summary(
    summary: LearningSummary,
    metric_name: str,
    metric_format: str = ".6f",
) -> str:
    """Render compact guidance for humans/agents."""
    lines = [
        "Learning summary:",
        f"- trials analyzed: {summary.total_trials}",
        f"- kept: {summary.kept_trials}",
    ]

    if summary.best_metric is not None and summary.best_commit:
        lines.append(f"- best so far: {summary.best_commit} ({metric_name}={summary.best_metric:{metric_format}})")

    if summary.prioritize:
        lines.append(f"- prioritize: {', '.join(summary.prioritize)}")
    if summary.avoid:
        lines.append(f"- avoid for now: {', '.join(summary.avoid)}")
    if summary.underexplored:
        lines.append(f"- underexplored: {', '.join(summary.underexplored)}")

    if summary.family_stats:
        lines.append("- family performance:")
        for s in summary.family_stats[:8]:
            best = f"{s.best_metric:{metric_format}}" if s.best_metric is not None else "---"
            lines.append(
                f"  - {s.family}: trials={s.trials}, keep={s.keeps}, discard={s.discards}, "
                f"fail={s.failures}, best={best}"
            )

    return "\n".join(lines)


def learning_payload(
    records: list[ExperimentRecord],
    direction: str,
    metric_name: str,
    metric_format: str,
    last_n: int = 40,
) -> dict:
    """
    Structured payload returned by tooling and used in prompts.
    """
    summary = summarize_learning(records, direction=direction, last_n=last_n)
    return {
        "total_trials": summary.total_trials,
        "kept_trials": summary.kept_trials,
        "best_commit": summary.best_commit,
        "best_metric": summary.best_metric,
        "prioritize": summary.prioritize,
        "avoid": summary.avoid,
        "underexplored": summary.underexplored,
        "families": [
            {
                "family": s.family,
                "trials": s.trials,
                "keeps": s.keeps,
                "discards": s.discards,
                "failures": s.failures,
                "best_metric": s.best_metric,
            }
            for s in summary.family_stats
        ],
        "summary_text": format_learning_summary(summary, metric_name=metric_name, metric_format=metric_format),
    }
