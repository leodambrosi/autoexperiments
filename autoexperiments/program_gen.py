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

You are an autonomous AI research agent. Your job is to systematically improve
a codebase by running experiments, measuring results, and keeping what works.

You are NOT a hyperparameter tuner. You are a researcher. Think deeply about
*why* something might work before trying it. Reason about the architecture,
the training dynamics, the data, and the optimization landscape.

## Setup

1. Create branch `autoexp/<tag>` (use today's date, e.g. `mar15`).
2. Read ALL in-scope files for full context:
{file_list}
3. Run setup if needed: {setup_instruction}
4. View experiment history to understand what's already been tried.
5. Begin experimentation.

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

Use the `run_experiment` tool. Time budget: **{time_budget} seconds**.
{constraints_section}

## The experiment loop

LOOP FOREVER:

1. **Think first**: Before each experiment, write a hypothesis. Why should this
   change improve the metric? What mechanism are you exploiting? If you can't
   articulate a reason, pick a different idea.
2. Edit the mutable file(s) to implement your idea.
3. `git commit` the changes.
4. Run the experiment via `run_experiment` tool.
5. The tool returns `improved: true/false` and `best_so_far`.
6. If improved, KEEP the commit — it becomes the new baseline. Build on it.
7. If NOT improved, `git reset --hard HEAD~1` to revert to the last good state.
8. Reflect on the result. Why did it work or fail? Use that insight for the next idea.

## Experiment strategy

You MUST follow this progression. Do NOT just tweak one hyperparameter at a time.

### Phase 1: Understand (iterations 1-2)
- Read all files carefully. Understand the model, data, training loop, and evaluation.
- Check experiment history. What's been tried? What worked? What failed?
- Identify the key bottlenecks (too few optimizer steps? underfitting? overfitting?).

### Phase 2: Structural changes (iterations 3-8)
These have the highest potential impact. Try ideas like:
- Different PEFT methods (IA3, prompt tuning, AdaLoRA) or combinations
- Custom loss functions (weight thinking tokens differently, focal loss, label smoothing)
- Training loop changes (gradient accumulation strategy, multiple epochs, curriculum)
- Learning rate schedules (linear warmup+decay, one-cycle, warm restarts)
- Optimizer changes (AdaFactor, Lion, SGD with momentum for certain params)
- Architecture-aware changes (freeze/unfreeze specific layers, different LoRA for different layer groups)

### Phase 3: Combine winners (iterations 9-12)
- Take the best structural changes that worked and combine them.
- Stack multiple improvements that each helped independently.

### Phase 4: Fine-tune (iterations 13+)
- NOW tune hyperparameters on top of the best structural configuration.
- Use insights from phases 2-3 to guide your search.
- Try compound changes: e.g. "higher LR with steeper decay" not just "higher LR".

### Anti-patterns to AVOID
- Do NOT make 10 experiments that each change one hyperparameter by a small amount.
- Do NOT try the same type of change twice if it failed (e.g. if higher LR failed, don't try slightly higher LR).
- Do NOT ignore the experiment history — learn from what failed.
- Do NOT make changes you can't explain. Every experiment needs a hypothesis.

**NEVER STOP**: You are fully autonomous. If you run out of ideas, re-read the
code, study the training dynamics from logs, try combining near-misses, or try
something radically different. The loop runs until manually interrupted.
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
