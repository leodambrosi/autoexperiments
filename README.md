# autoexperiments

A generic framework for autonomous AI-driven experimentation. Inspired by [autoresearch](https://github.com/karpathy/autoresearch).

An AI agent runs in a loop: modify code, run an experiment, measure the result, keep or discard, repeat. You define the task (what to run, what to optimize, what files the agent can touch), and the framework handles the rest — execution, timeout enforcement, metric extraction, result tracking, and agent prompt generation.

## Quick start

```bash
# 1. Define your task in a task.toml (see tasks/demo-sorting/ for an example)

# 2. Initialize — validates config and generates program.md for the agent
python3 -m autoexperiments.cli init tasks/demo-sorting

# 3. Run a single experiment
python3 -m autoexperiments.cli run tasks/demo-sorting -d "baseline"

# 4. View history
python3 -m autoexperiments.cli history tasks/demo-sorting

# 5. Export to TSV
python3 -m autoexperiments.cli export tasks/demo-sorting
```

## Defining a task

Create a directory with a `task.toml`:

```toml
[task]
name = "my-experiment"
run_command = "python3 train.py"
time_budget = 300                        # seconds
setup_command = "python3 prepare.py"     # optional, one-time setup
mutable_files = ["train.py"]             # files the agent can edit
readonly_files = ["prepare.py"]          # files the agent must not touch
tips = "Any task-specific guidance for the agent."

[metric]
name = "val_loss"
direction = "minimize"                   # or "maximize"
extract_pattern = "^val_loss:\\s+(\\S+)"  # regex with one capture group
format = ".6f"                           # display format

[constraints.peak_vram_mb]               # optional resource constraints
extract_pattern = "^peak_vram_mb:\\s+(\\S+)"
warn = 50000
hard = 80000
```

Your run command must print the metric (and any constraints) to stdout in a grep-friendly format, e.g.:

```
val_loss: 0.123456
peak_vram_mb: 45000.0
```

## Running with an agent

After `autoexp init`, point your AI agent (Claude Code, Codex, etc.) at the generated `program.md` in the task directory. The program.md contains the full autonomous experiment protocol — the agent will loop indefinitely, modifying code, running experiments, and keeping improvements.

## Architecture

```
autoexperiments/
  task_config.py   — loads and validates task.toml
  runner.py        — executes experiments, enforces timeouts, extracts metrics
  tracker.py       — SQLite-backed experiment history with lineage tracing
  git_ops.py       — git operations for branching, committing, reverting
  program_gen.py   — generates agent program.md from task config
  cli.py           — CLI entry point (init, run, history, export)

tasks/
  gpt-pretraining/ — port of karpathy/autoresearch
  demo-sorting/    — simple CPU-only demo task
```

## Example tasks

### GPT pretraining (`tasks/gpt-pretraining/`)

The original autoresearch setup: single-GPU GPT pretraining with a 5-minute time budget. Requires an NVIDIA GPU and `uv`. Copy `prepare.py` and `train.py` from autoresearch into this directory.

### Sorting benchmark (`tasks/demo-sorting/`)

A simple CPU-only demo: optimize a sorting benchmark for maximum throughput. No GPU required — good for testing the framework.

## License

MIT
