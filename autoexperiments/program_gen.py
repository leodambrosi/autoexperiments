"""
Generate agent program.md files from a task config + base template.

The base template contains the universal experiment protocol.
Task-specific details (metric name, files, commands, tips) are interpolated.
"""

from __future__ import annotations

from pathlib import Path

from .task_config import TaskConfig

BASE_TEMPLATE = """\
# autoexperiments

This is an autonomous experimentation framework. You are an AI agent running
experiments in a loop: modify code, run, measure, keep or discard, repeat.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoexp/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoexp/<tag>` from current main branch.
3. **Read the in-scope files**: Read all files for full context:
{file_list}
4. **Run setup** (if needed): {setup_instruction}
5. **Confirm and go**: Confirm setup looks good, then begin experimentation.

## Task

**Name**: {task_name}
**Goal**: {goal_description}
**Metric**: `{metric_name}` ({metric_direction} is better)

## Rules

**What you CAN do:**
{mutable_section}

**What you CANNOT do:**
{readonly_section}
- Install new packages or add dependencies.
- Modify the evaluation harness or metric extraction.

## Running an experiment

```bash
{run_command} > run.log 2>&1
```

Time budget: **{time_budget} seconds**. If a run exceeds {timeout_seconds} seconds, kill it and treat as failure.

Extract results:
```bash
grep -E '{grep_patterns}' run.log
```
{constraints_section}

## The experiment loop

LOOP FOREVER:

1. Look at the git state: current branch/commit.
2. Modify the mutable file(s) with an experimental idea.
3. `git commit` the changes.
4. Run the experiment: `{run_command} > run.log 2>&1`
5. Extract the metric: check `run.log` for `{metric_name}`.
6. If the output is empty or the run crashed, run `tail -n 50 run.log` to diagnose. Attempt a fix if trivial; otherwise give up and move on.
7. If {metric_name} improved ({improvement_word}), keep the commit and advance.
8. If {metric_name} is equal or worse, `git reset --hard` to the previous good commit.
9. Log the result and continue.

**Crashes**: If a run crashes from a trivial bug (typo, missing import), fix and re-run. If the idea is fundamentally broken, log as crash and move on.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. You are autonomous. If you run out of ideas, re-read the code, try combining previous near-misses, try more radical changes. The loop runs until manually interrupted.
{tips_section}
"""


def generate_program(config: TaskConfig) -> str:
    """Generate a complete program.md from a task config."""

    # File listing
    all_files = config.readonly_files + config.mutable_files
    file_list = "\n".join(f"   - `{f}`" for f in all_files)

    # Setup instruction
    setup_instruction = f"`{config.setup_command}`" if config.setup_command else "No setup command needed."

    # Goal description
    goal_description = f"Get the {'lowest' if config.metric.direction == 'minimize' else 'highest'} `{config.metric.name}`."

    # Mutable/readonly sections
    mutable_section = "\n".join(
        f"- Modify `{f}` — everything in this file is fair game."
        for f in config.mutable_files
    )
    readonly_section = "\n".join(
        f"- Modify `{f}`. It is read-only."
        for f in config.readonly_files
    )

    # Grep patterns for metric + constraints
    patterns = [config.metric.extract_pattern.lstrip("^").split("\\s")[0]]
    for c in config.constraints:
        patterns.append(c.extract_pattern.lstrip("^").split("\\s")[0])
    grep_patterns = "|".join(f"^{p}" for p in patterns)

    # Constraints section
    constraints_section = ""
    if config.constraints:
        lines = ["\n## Constraints\n"]
        for c in config.constraints:
            parts = []
            if c.warn is not None:
                parts.append(f"warn above {c.warn}")
            if c.hard is not None:
                parts.append(f"hard limit {c.hard}")
            lines.append(f"- **{c.name}**: {', '.join(parts)}")
        constraints_section = "\n".join(lines)

    # Tips
    tips_section = ""
    if config.tips:
        tips_section = f"\n## Tips\n\n{config.tips}\n"

    # Improvement direction
    improvement_word = "lower" if config.metric.direction == "minimize" else "higher"

    return BASE_TEMPLATE.format(
        task_name=config.name,
        file_list=file_list,
        setup_instruction=setup_instruction,
        goal_description=goal_description,
        metric_name=config.metric.name,
        metric_direction=config.metric.direction,
        mutable_section=mutable_section,
        readonly_section=readonly_section,
        run_command=config.run_command,
        time_budget=config.time_budget,
        timeout_seconds=config.time_budget * 2,
        grep_patterns=grep_patterns,
        constraints_section=constraints_section,
        tips_section=tips_section,
        improvement_word=improvement_word,
    )


def write_program(config: TaskConfig, output_path: str | Path) -> Path:
    """Generate and write program.md to disk."""
    path = Path(output_path)
    path.write_text(generate_program(config))
    return path
