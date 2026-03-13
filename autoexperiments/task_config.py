"""
Task configuration: loads task.toml and provides typed access to all settings.

A task.toml defines everything the framework needs to run experiments:
- What command to run
- What files the agent can modify
- What metric to optimize
- Time and resource constraints
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MetricConfig:
    name: str
    direction: str  # "minimize" or "maximize"
    extract_pattern: str  # regex with one capture group for the metric value
    format: str = ".6f"  # display format

    def is_better(self, new: float, old: float) -> bool:
        if self.direction == "minimize":
            return new < old
        return new > old


@dataclass
class ConstraintConfig:
    name: str
    extract_pattern: str  # regex to extract the value from run output
    warn: float | None = None
    hard: float | None = None

    def check(self, value: float) -> str:
        """Returns 'ok', 'warn', or 'fail'."""
        if self.hard is not None and value > self.hard:
            return "fail"
        if self.warn is not None and value > self.warn:
            return "warn"
        return "ok"


@dataclass
class TaskConfig:
    name: str
    run_command: str
    time_budget: int  # seconds
    mutable_files: list[str]
    readonly_files: list[str] = field(default_factory=list)
    setup_command: str | None = None
    metric: MetricConfig = field(default_factory=lambda: MetricConfig(
        name="metric", direction="minimize", extract_pattern=r"^metric:\s+(\S+)"
    ))
    constraints: list[ConstraintConfig] = field(default_factory=list)
    tips: str = ""

    @classmethod
    def from_file(cls, path: str | Path) -> TaskConfig:
        path = Path(path)
        with open(path, "rb") as f:
            raw = tomllib.load(f)

        task = raw["task"]
        metric_raw = raw.get("metric", {})
        metric = MetricConfig(
            name=metric_raw.get("name", "metric"),
            direction=metric_raw.get("direction", "minimize"),
            extract_pattern=metric_raw.get("extract_pattern", r"^metric:\s+(\S+)"),
            format=metric_raw.get("format", ".6f"),
        )

        constraints = []
        for name, craw in raw.get("constraints", {}).items():
            constraints.append(ConstraintConfig(
                name=name,
                extract_pattern=craw["extract_pattern"],
                warn=craw.get("warn"),
                hard=craw.get("hard"),
            ))

        return cls(
            name=task["name"],
            run_command=task["run_command"],
            time_budget=task.get("time_budget", 300),
            mutable_files=task.get("mutable_files", []),
            readonly_files=task.get("readonly_files", []),
            setup_command=task.get("setup_command"),
            metric=metric,
            constraints=constraints,
            tips=task.get("tips", ""),
        )
