"""
Experiment runner: executes a task's run command, enforces time budget,
and extracts structured results from stdout.
"""

from __future__ import annotations

import re
import subprocess
import threading
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
    Streams output to log_path in real time if provided.

    Args:
        config: Task configuration.
        task_dir: Working directory for the command.
        log_path: If provided, stream stdout+stderr to this file in real time.

    Returns:
        ExperimentResult with extracted metric and status.
    """
    task_dir = Path(task_dir)
    timeout = config.time_budget * 2  # hard kill at 2x budget

    log_file = open(log_path, "w", encoding="utf-8") if log_path else None
    collected: list[str] = []

    t0 = time.monotonic()
    try:
        proc = subprocess.Popen(
            config.run_command,
            shell=True,
            cwd=task_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        def _drain_stdout() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                collected.append(line)
                if log_file:
                    log_file.write(line)
                    log_file.flush()

        reader = threading.Thread(target=_drain_stdout, daemon=True)
        reader.start()

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            reader.join(timeout=1.0)
            wall = time.monotonic() - t0
            output = "".join(collected)
            return ExperimentResult(
                status="timeout",
                wall_seconds=wall,
                stdout=output,
                stderr="",
                tail=_tail(output, 50),
            )

        reader.join(timeout=1.0)

        wall = time.monotonic() - t0
        returncode = proc.returncode
        output = "".join(collected)

    except Exception as e:
        wall = time.monotonic() - t0
        output = "".join(collected)
        return ExperimentResult(
            status="crash",
            wall_seconds=wall,
            stdout=output,
            stderr=str(e),
            tail=_tail(output + "\n" + str(e), 50),
        )
    finally:
        if log_file:
            log_file.close()

    if returncode != 0:
        return ExperimentResult(
            status="crash",
            wall_seconds=wall,
            stdout=output,
            stderr="",
            tail=_tail(output, 50),
        )

    # Extract metric
    metric = _extract_value(config.metric.extract_pattern, output)
    if metric is None:
        return ExperimentResult(
            status="crash",
            wall_seconds=wall,
            stdout=output,
            stderr=f"Failed to extract metric '{config.metric.name}' using pattern: {config.metric.extract_pattern}",
            tail=_tail(output, 50),
        )

    # Extract constraints
    constraint_values = {}
    for c in config.constraints:
        val = _extract_value(c.extract_pattern, output)
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
        stdout=output,
        stderr="",
        tail=_tail(output, 50),
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
        metric_value=result.metric,
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
