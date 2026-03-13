"""
CLI entry point for autoexperiments.

Commands:
  autoexp init <task_dir>       — validate task.toml and generate program.md
  autoexp run <task_dir>        — run a single experiment and print results
  autoexp history <task_dir>    — show experiment history
  autoexp export <task_dir>     — export results to TSV
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .task_config import TaskConfig
from .runner import run_experiment
from .tracker import ExperimentTracker
from .program_gen import write_program
from .git_ops import current_commit, snapshot_files


def cmd_init(args: argparse.Namespace) -> None:
    """Validate task config and generate program.md."""
    task_dir = Path(args.task_dir).resolve()
    config_path = task_dir / "task.toml"

    if not config_path.exists():
        print(f"Error: {config_path} not found", file=sys.stderr)
        sys.exit(1)

    config = TaskConfig.from_file(config_path)
    print(f"Task: {config.name}")
    print(f"Run command: {config.run_command}")
    print(f"Time budget: {config.time_budget}s")
    print(f"Metric: {config.metric.name} ({config.metric.direction})")
    print(f"Mutable files: {config.mutable_files}")
    print(f"Readonly files: {config.readonly_files}")

    if config.constraints:
        print("Constraints:")
        for c in config.constraints:
            print(f"  {c.name}: warn={c.warn}, hard={c.hard}")

    # Generate program.md
    program_path = write_program(config, task_dir / "program.md")
    print(f"\nGenerated: {program_path}")


def cmd_run(args: argparse.Namespace) -> None:
    """Run a single experiment."""
    task_dir = Path(args.task_dir).resolve()
    config = TaskConfig.from_file(task_dir / "task.toml")

    log_path = task_dir / "run.log"
    print(f"Running: {config.run_command}")
    print(f"Time budget: {config.time_budget}s (hard timeout: {config.time_budget * 2}s)")
    print()

    result = run_experiment(config, task_dir, log_path=log_path)

    print(f"Status: {result.status}")
    print(f"Wall time: {result.wall_seconds:.1f}s")

    if result.metric is not None:
        fmt = config.metric.format
        print(f"{config.metric.name}: {result.metric:{fmt}}")
    else:
        print(f"{config.metric.name}: (not found in output)")

    for name, value in result.constraints.items():
        constraint = next((c for c in config.constraints if c.name == name), None)
        if constraint:
            level = constraint.check(value)
            marker = " ⚠" if level == "warn" else " ✗" if level == "fail" else ""
            print(f"{name}: {value:.1f}{marker}")

    if result.crashed:
        print(f"\n--- Last 20 lines ---\n{_tail(result.tail, 20)}")

    # Log to tracker if in a git repo
    try:
        commit_hash = current_commit(task_dir)
        snapshot = snapshot_files(task_dir, config.mutable_files)
        tracker = ExperimentTracker(task_dir / ".autoexp" / "experiments.db")
        tracker.log(
            commit=commit_hash,
            metric_name=config.metric.name,
            metric_value=result.metric if result.metric is not None else 0.0,
            status=result.status,
            description=args.description or "",
            wall_seconds=result.wall_seconds,
            constraints={k: v for k, v in result.constraints.items()},
            config_snapshot=snapshot,
        )
        tracker.close()
        print(f"\nLogged to .autoexp/experiments.db")
    except Exception:
        pass  # not in a git repo or other issue, skip tracking


def cmd_history(args: argparse.Namespace) -> None:
    """Show experiment history."""
    task_dir = Path(args.task_dir).resolve()
    config = TaskConfig.from_file(task_dir / "task.toml")
    db_path = task_dir / ".autoexp" / "experiments.db"

    if not db_path.exists():
        print("No experiments recorded yet.", file=sys.stderr)
        sys.exit(1)

    tracker = ExperimentTracker(db_path)
    records = tracker.history(last_n=args.last)

    if not records:
        print("No experiments found.")
        tracker.close()
        return

    # Header
    fmt = config.metric.format
    print(f"{'#':>4}  {'commit':7}  {'metric':>12}  {'status':>10}  {'time':>7}  description")
    print("-" * 70)
    for r in records:
        metric_str = f"{r.metric_value:{fmt}}" if r.status != "crash" else "---"
        print(f"{r.id:4}  {r.commit:7}  {metric_str:>12}  {r.status:>10}  {r.wall_seconds:6.1f}s  {r.description}")

    best = tracker.best(direction=config.metric.direction)
    if best:
        print(f"\nBest: {best.commit} ({config.metric.name}={best.metric_value:{fmt}})")

    total = tracker.count()
    kept = tracker.count("keep")
    print(f"Total: {total} experiments, {kept} kept")
    tracker.close()


def cmd_export(args: argparse.Namespace) -> None:
    """Export results to TSV."""
    task_dir = Path(args.task_dir).resolve()
    db_path = task_dir / ".autoexp" / "experiments.db"

    if not db_path.exists():
        print("No experiments recorded yet.", file=sys.stderr)
        sys.exit(1)

    output = Path(args.output) if args.output else task_dir / "results.tsv"
    tracker = ExperimentTracker(db_path)
    tracker.export_tsv(output)
    tracker.close()
    print(f"Exported to {output}")


def _tail(text: str, n: int) -> str:
    lines = text.strip().splitlines()
    return "\n".join(lines[-n:])


def main():
    parser = argparse.ArgumentParser(
        prog="autoexp",
        description="Autonomous experimentation framework",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = sub.add_parser("init", help="Validate task config and generate program.md")
    p_init.add_argument("task_dir", help="Path to task directory containing task.toml")

    # run
    p_run = sub.add_parser("run", help="Run a single experiment")
    p_run.add_argument("task_dir", help="Path to task directory")
    p_run.add_argument("-d", "--description", default="", help="Description of this experiment")

    # history
    p_hist = sub.add_parser("history", help="Show experiment history")
    p_hist.add_argument("task_dir", help="Path to task directory")
    p_hist.add_argument("-n", "--last", type=int, default=20, help="Number of recent experiments to show")

    # export
    p_export = sub.add_parser("export", help="Export results to TSV")
    p_export.add_argument("task_dir", help="Path to task directory")
    p_export.add_argument("-o", "--output", help="Output file path (default: <task_dir>/results.tsv)")

    args = parser.parse_args()

    commands = {
        "init": cmd_init,
        "run": cmd_run,
        "history": cmd_history,
        "export": cmd_export,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
