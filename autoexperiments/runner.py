"""
Experiment runner: executes a task's run command, enforces time budget,
and extracts structured results from stdout.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .git_ops import current_commit, snapshot_files
from .task_config import TaskConfig


@dataclass
class ExperimentResult:
    metric: float | None = None
    constraints: dict[str, float] = field(default_factory=dict)
    status: str = "success"  # "success", "crash", "timeout"
    wall_seconds: float = 0.0
    stdout: str = ""
    stderr: str = ""
    tail: str = ""  # last N lines for crash diagnosis

    @property
    def crashed(self) -> bool:
        return self.status in ("crash", "timeout")


def _extract_value(pattern: str, text: str) -> float | None:
    """Extract a float from text using a regex with one capture group."""
    match = re.search(pattern, text, re.MULTILINE)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, IndexError):
            return None
    return None


def run_experiment(config: TaskConfig, task_dir: str | Path, log_path: str | Path | None = None) -> ExperimentResult:
    """
    Run the task's command, enforce timeout, extract metric and constraints.

    Args:
        config: Task configuration.
        task_dir: Working directory for the command.
        log_path: If provided, write raw stdout+stderr to this file.

    Returns:
        ExperimentResult with extracted metric and status.
    """
    task_dir = Path(task_dir)
    timeout = config.time_budget * 2  # hard kill at 2x budget

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            config.run_command,
            shell=True,
            cwd=task_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        wall = time.monotonic() - t0
        stdout = proc.stdout
        stderr = proc.stderr
        returncode = proc.returncode
    except subprocess.TimeoutExpired as e:
        wall = time.monotonic() - t0
        stdout = e.stdout or ""
        stderr = e.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        result = ExperimentResult(
            status="timeout",
            wall_seconds=wall,
            stdout=stdout,
            stderr=stderr,
            tail=_tail(stdout + "\n" + stderr, 50),
        )
        if log_path:
            Path(log_path).write_text(stdout + "\n" + stderr)
        return result

    combined = stdout + "\n" + stderr

    if log_path:
        Path(log_path).write_text(combined)

    if returncode != 0:
        return ExperimentResult(
            status="crash",
            wall_seconds=wall,
            stdout=stdout,
            stderr=stderr,
            tail=_tail(combined, 50),
        )

    # Extract metric
    metric = _extract_value(config.metric.extract_pattern, combined)

    # Extract constraints
    constraint_values = {}
    for c in config.constraints:
        val = _extract_value(c.extract_pattern, combined)
        if val is not None:
            constraint_values[c.name] = val

    # Check hard constraints
    status = "success"
    for c in config.constraints:
        if c.name in constraint_values and c.check(constraint_values[c.name]) == "fail":
            status = "constraint_violated"
            break

    return ExperimentResult(
        metric=metric,
        constraints=constraint_values,
        status=status,
        wall_seconds=wall,
        stdout=stdout,
        stderr=stderr,
        tail=_tail(combined, 50),
    )


def run_and_record(
    config: TaskConfig,
    task_dir: str | Path,
    tracker,
    description: str = "",
    log_path: str | Path | None = None,
) -> tuple[ExperimentResult, str, bool]:
    """
    Run an experiment, classify as keep/discard, and log to tracker.

    Returns (result, status, improved).
    """
    from .tracker import ExperimentTracker

    task_dir = Path(task_dir)
    result = run_experiment(config, task_dir, log_path=log_path)

    # Classify
    status = result.status
    improved = False
    if status == "success" and result.metric is not None:
        best = tracker.best(direction=config.metric.direction)
        if best is None:
            status = "keep"
            improved = True
        elif config.metric.is_better(result.metric, best.metric_value):
            status = "keep"
            improved = True
        else:
            status = "discard"

    # Log
    commit_hash = current_commit(task_dir)
    snapshot = snapshot_files(task_dir, config.mutable_files)
    tracker.log(
        commit=commit_hash,
        metric_name=config.metric.name,
        metric_value=result.metric if result.metric is not None else 0.0,
        status=status,
        description=description,
        wall_seconds=result.wall_seconds,
        constraints={k: v for k, v in result.constraints.items()},
        config_snapshot=snapshot,
    )

    return result, status, improved


def _tail(text: str, n: int) -> str:
    lines = text.strip().splitlines()
    return "\n".join(lines[-n:])
