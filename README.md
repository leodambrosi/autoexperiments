# autoexperiments

A framework for autonomous AI-driven experimentation. Inspired by Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch). A Gemini agent runs in a loop: modify code, run an experiment, measure the result, keep or discard, repeat.

You define the task (what to run, what to optimize, what files the agent can touch). The framework handles execution, metric extraction, git commits, automatic reverts on failure, and result tracking.

## Quick start

```bash
# Install
pip install -e .

# Set your Gemini API key
export GEMINI_API_KEY=your-key-here

# One-time: initialize task and generate agent prompt
autoexp init tasks/llm-finetune

# Edit program.md to add strategy, tips, failed experiments (optional but recommended)

# Run the agent (starts autonomous experiment loop)
autoexp agent tasks/llm-finetune

# Monitor a running experiment in another terminal
tail -f tasks/llm-finetune/run.log

# View experiment history
autoexp history tasks/llm-finetune

# Export results to TSV
autoexp export tasks/llm-finetune
```

### Google Colab

```python
# Install from git
!pip install git+https://github.com/youruser/autoexperiments.git

# Auth
import os
from google.colab import userdata
os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")

# Run via Python API
from autoexperiments import Experiment

exp = Experiment("tasks/llm-finetune")
exp.run_agent(model="gemini-3.1-pro-preview", max_iterations=20)
exp.history()
```

## How it works

The agent loop is simple:

1. Agent edits mutable files with `edit_file`
2. Agent calls `run_experiment` with a description
3. Framework auto-commits, runs the command, extracts the metric
4. If improved → commit is kept, becomes the new baseline
5. If not improved → commit is automatically reverted
6. Agent reflects and tries the next idea

The agent never needs to run git commands. Commits and reverts are handled by `run_experiment`.

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
| `run_experiment` | Auto-commit, run, extract metric, auto-revert on failure |
| `view_history` | Query past experiment results |
| `bash` | Shell commands for inspecting files and environment |

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

# Monitor a running experiment
tail -f tasks/my-task/run.log

# Export all to TSV
autoexp export tasks/my-task
```

Results are stored in `tasks/my-task/.autoexp/experiments.db` (SQLite).

## Architecture

```
autoexperiments/
  cli.py           — CLI: init, agent, history, export
  agent.py         — Gemini agent loop with tool dispatch
  runner.py        — run experiments, stream output, extract metrics, classify & log
  tracker.py       — SQLite experiment history
  task_config.py   — load and validate task.toml
  git_ops.py       — git commit, reset, snapshot (optional, works without git)
  program_gen.py   — generate program.md from task config
```

## Example tasks

### LLM fine-tuning (`tasks/llm-finetune/`)

LoRA fine-tuning of Qwen3.5-2B with thinking traces. Runs on CUDA or Apple Silicon MPS. 5-minute time budget per experiment.

### Sorting benchmark (`tasks/demo-sorting/`)

CPU-only demo: optimize a sorting benchmark for throughput. No GPU required — good for testing the framework.

## License

MIT
