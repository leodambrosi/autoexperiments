"""
autoexperiments — autonomous AI-driven experimentation framework.

Python API for use in notebooks (Colab, Jupyter) and scripts:

    from autoexperiments import setup_task, Experiment

    setup_task("llm-finetune")           # extracts task files to ./llm-finetune/
    exp = Experiment("llm-finetune")
    exp.run_agent(max_iterations=20)
    exp.history()
"""

from __future__ import annotations

import importlib.resources
import subprocess
from pathlib import Path

from .task_config import TaskConfig
from .tracker import ExperimentTracker, ExperimentRecord
from .runner import ExperimentResult, run_experiment


def setup_task(task_name: str, dest: str | Path | None = None) -> Path:
    """
    Extract a bundled task to a local directory.

    Args:
        task_name: Name of the bundled task (e.g. "llm-finetune").
        dest: Destination directory. Defaults to ./<task_name>/ in the current directory.

    Returns:
        Path to the extracted task directory.
    """
    dest = Path(dest) if dest else Path(task_name)

    # Find the bundled task inside the package
    tasks_pkg = importlib.resources.files("autoexperiments.tasks") / task_name
    if not tasks_pkg.is_dir():
        available = [
            p.name for p in importlib.resources.files("autoexperiments.tasks").iterdir()
            if p.is_dir() and p.name != "__pycache__"
        ]
        raise ValueError(f"Unknown task {task_name!r}. Available: {available}")

    dest.mkdir(parents=True, exist_ok=True)

    for item in tasks_pkg.iterdir():
        if item.name.startswith(("__", ".")):
            continue
        target = dest / item.name
        if not target.exists():
            target.write_text(item.read_text())
            print(f"  Created {target}")
        else:
            print(f"  Skipped {target} (already exists)")

    # Init git repo so the agent can commit/revert experiments
    from .git_ops import has_git
    if not has_git(dest):
        print("\nInitializing git repo...")
        subprocess.run(["git", "init"], cwd=dest, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=dest, capture_output=True)
        subprocess.run(["git", "commit", "-m", "initial task setup"], cwd=dest, capture_output=True)

    # Run setup command (e.g. data preparation) if defined in task.toml
    config = TaskConfig.from_file(dest / "task.toml")
    if config.setup_command:
        print(f"\nRunning setup: {config.setup_command}")
        result = subprocess.run(
            config.setup_command, shell=True, cwd=dest,
        )
        if result.returncode != 0:
            print(f"Warning: setup command exited with code {result.returncode}")

    print(f"\nTask ready at {dest.resolve()}")
    print(f"Next: Experiment({str(dest)!r}).run_agent()")
    return dest


class Experiment:
    """Main entry point for running experiments programmatically."""

    def __init__(self, task_dir: str | Path):
        self.task_dir = Path(task_dir).resolve()
        self.config = TaskConfig.from_file(self.task_dir / "task.toml")
        db_path = self.task_dir / ".autoexp" / "experiments.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.tracker = ExperimentTracker(db_path)

    def run(self, description: str = "") -> ExperimentResult:
        """Run a single experiment, log to tracker, return the result."""
        from .git_ops import current_commit, snapshot_files

        result = run_experiment(
            self.config, self.task_dir,
            log_path=self.task_dir / "run.log",
        )

        # Classify keep/discard
        status = result.status
        if status == "success" and result.metric is not None:
            best = self.tracker.best(direction=self.config.metric.direction)
            if best is None or self.config.metric.is_better(result.metric, best.metric_value):
                status = "keep"
            else:
                status = "discard"

        # Log
        commit_hash = current_commit(self.task_dir)
        snapshot = snapshot_files(self.task_dir, self.config.mutable_files)
        self.tracker.log(
            commit=commit_hash,
            metric_name=self.config.metric.name,
            metric_value=result.metric,
            status=status,
            description=description,
            wall_seconds=result.wall_seconds,
            constraints=dict(result.constraints),
            config_snapshot=snapshot,
        )

        fmt = self.config.metric.format
        if result.metric is not None:
            marker = " (new best!)" if status == "keep" else ""
            print(f"{self.config.metric.name}: {result.metric:{fmt}}{marker}")
        print(f"Status: {status} | Wall time: {result.wall_seconds:.1f}s")

        return result

    def run_agent(
        self,
        model: str = "gemini-3.1-pro-preview",
        max_iterations: int = 50,
        api_key: str | None = None,
    ) -> None:
        """Run the autonomous Gemini agent loop."""
        from .agent import run_agent
        run_agent(
            task_dir=self.task_dir,
            config=self.config,
            model=model,
            max_iterations=max_iterations,
            api_key=api_key,
        )

    def history(self, last_n: int = 20) -> list[ExperimentRecord]:
        """Print and return experiment history."""
        records = self.tracker.history(last_n=last_n)
        if not records:
            print("No experiments recorded yet.")
            return []

        fmt = self.config.metric.format
        print(f"{'#':>4}  {'commit':7}  {'metric':>12}  {'status':>10}  {'time':>7}  description")
        print("-" * 70)
        for r in records:
            metric_str = f"{r.metric_value:{fmt}}" if r.metric_value is not None else "---"
            print(f"{r.id:4}  {r.commit:7}  {metric_str:>12}  {r.status:>10}  {r.wall_seconds:6.1f}s  {r.description}")

        best = self.tracker.best(direction=self.config.metric.direction)
        if best:
            print(f"\nBest: {best.commit} ({self.config.metric.name}={best.metric_value:{fmt}})")

        total = self.tracker.count()
        kept = self.tracker.count("keep")
        print(f"Total: {total} experiments, {kept} kept")
        return records

    def best(self) -> ExperimentRecord | None:
        """Return the best kept experiment."""
        return self.tracker.best(direction=self.config.metric.direction)

    def init(self) -> Path:
        """Generate program.md from task config."""
        from .program_gen import write_program
        return write_program(self.config, self.task_dir / "program.md")
