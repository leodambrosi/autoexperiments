# autoexperiments

A framework for autonomous AI-driven experimentation. Inspired by [autoresearch](https://github.com/karpathy/autoresearch). A Gemini agent runs in a loop: modify code, run an experiment, measure the result, keep or discard, repeat.

You define the task (what to run, what to optimize, what files the agent can touch). The framework handles execution, timeout enforcement, metric extraction, result tracking, and the agent loop.

## Quick start

```bash
# Install
pip install -e .

# Set your Gemini API key
export GEMINI_API_KEY=your-key-here

# Define a task (see "Defining a task" below), then:

# 1. Initialize — validates config, generates program.md
autoexp init tasks/my-task

# 2. (Optional) Edit program.md to add strategy, tips, failed experiments

# 3. Run the agent
autoexp agent tasks/my-task

# 4. View history
autoexp history tasks/my-task

# 5. Export to TSV
autoexp export tasks/my-task
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

Your run command must print the metric (and any constraints) to stdout:

```
val_loss: 0.123456
peak_vram_mb: 45000.0
```

## Running the agent

```bash
# Default: gemini-3.1-pro-preview, 50 iterations
autoexp agent tasks/my-task

# Use a cheaper model
autoexp agent tasks/my-task -m gemini-3-flash-preview

# More iterations
autoexp agent tasks/my-task -n 100
```

The agent gets five tools:

| Tool | Description |
|------|-------------|
| `read_file` | Read files in the task directory |
| `edit_file` | Find-and-replace in mutable files |
| `run_experiment` | Run the task, extract metric, classify as keep/discard |
| `view_history` | Query past experiment results |
| `bash` | Shell commands (git commit, reset, diff, etc.) |

On each experiment, the framework automatically:
- Runs the command with a 2x time budget hard timeout
- Extracts the metric via regex
- Compares to the best kept result
- Logs to SQLite as `keep` (new best) or `discard`
- Returns `improved: true/false` to the agent

The agent is expected to `git reset --hard HEAD~1` when `improved: false`.

## Steering the agent

`program.md` is the agent's system prompt. `autoexp init` generates a default one from `task.toml`, but you should edit it to add:

- Current best metric and the config that achieved it
- A table of failed experiments so the agent doesn't repeat them
- Strategy guidance (what to try next, what phases to follow)
- Hardware constraints and gotchas

The agent reads this at startup. Update it between runs to refine strategy.

## Viewing results

```bash
# Last 20 experiments
autoexp history tasks/my-task

# Last 50
autoexp history tasks/my-task -n 50

# Export all to TSV
autoexp export tasks/my-task
```

Results are stored in `tasks/my-task/.autoexp/experiments.db` (SQLite).

## Architecture

```
autoexperiments/
  cli.py           — CLI: init, agent, history, export
  agent.py         — Gemini agent loop with tool dispatch
  runner.py        — run experiments, enforce timeouts, extract metrics, classify & log
  tracker.py       — SQLite experiment history
  task_config.py   — load and validate task.toml
  git_ops.py       — git branch, commit, reset, snapshot
  program_gen.py   — generate program.md from task config
```

## Example tasks

### LLM fine-tuning (`tasks/llm-finetune/`)

LoRA fine-tuning of Qwen3.5-2B with thinking traces. Runs on CUDA or Apple Silicon MPS. 5-minute time budget per experiment.

### Sorting benchmark (`tasks/demo-sorting/`)

CPU-only demo: optimize a sorting benchmark for throughput. No GPU required — good for testing the framework.

## License

MIT
