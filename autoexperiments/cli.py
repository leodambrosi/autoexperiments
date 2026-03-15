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
from .runner import run_and_record
from .tracker import ExperimentTracker
from .program_gen import write_program
from .agent import run_agent


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

    print(f"Running: {config.run_command}")
    print(f"Time budget: {config.time_budget}s (hard timeout: {config.time_budget * 2}s)")
    print()

    tracker = ExperimentTracker(task_dir / ".autoexp" / "experiments.db")
    result, status, improved = run_and_record(
        config, task_dir, tracker,
        description=args.description or "",
        log_path=task_dir / "run.log",
    )
    tracker.close()

    print(f"Status: {status}")
    print(f"Wall time: {result.wall_seconds:.1f}s")

    if result.metric is not None:
        fmt = config.metric.format
        print(f"{config.metric.name}: {result.metric:{fmt}}")
        if improved:
            print("  ✓ New best!")
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


def cmd_agent(args: argparse.Namespace) -> None:
    """Run the autonomous experiment agent."""
    task_dir = Path(args.task_dir).resolve()
    config = TaskConfig.from_file(task_dir / "task.toml")

    print(f"Starting autonomous agent for: {config.name}")
    print(f"Model: {args.model}")
    print(f"Max iterations: {args.max_iterations}")

    run_agent(
        task_dir=task_dir,
        config=config,
        model=args.model,
        max_iterations=args.max_iterations,
        api_key=args.api_key,
    )


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

    # agent
    p_agent = sub.add_parser("agent", help="Run autonomous experiment agent")
    p_agent.add_argument("task_dir", help="Path to task directory")
    p_agent.add_argument("-m", "--model", default="gemini-3.1-pro-preview", help="Gemini model (default: gemini-3.1-pro-preview). Cheaper: gemini-3-flash-preview")
    p_agent.add_argument("-n", "--max-iterations", type=int, default=50, help="Max agent iterations (default: 50)")
    p_agent.add_argument("--api-key", help="Google API key (or set GEMINI_API_KEY env var)")

    # export
    p_export = sub.add_parser("export", help="Export results to TSV")
    p_export.add_argument("task_dir", help="Path to task directory")
    p_export.add_argument("-o", "--output", help="Output file path (default: <task_dir>/results.tsv)")

    args = parser.parse_args()

    commands = {
        "init": cmd_init,
        "run": cmd_run,
        "agent": cmd_agent,
        "history": cmd_history,
        "export": cmd_export,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
