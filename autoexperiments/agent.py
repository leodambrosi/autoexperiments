"""
Autonomous experiment agent powered by Google Gemini.

Runs in a loop: reads code, proposes changes, runs experiments,
keeps improvements, discards failures. Uses tool calling to
interact with the codebase and experiment runner.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from google import genai
from google.genai import types

from .runner import run_and_record
from .task_config import TaskConfig
from .tracker import ExperimentTracker


# ---------------------------------------------------------------------------
# Tool definitions (JSON schemas for Gemini function calling)
# ---------------------------------------------------------------------------

TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="read_file",
        description="Read the contents of a file in the task directory.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the task directory, e.g. 'finetune.py'",
                },
            },
            "required": ["path"],
        },
    ),
    types.FunctionDeclaration(
        name="edit_file",
        description="Replace a specific string in a file with a new string. The old_string must appear exactly once in the file.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the task directory",
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact string to find and replace (must be unique in the file)",
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement string",
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
    ),
    types.FunctionDeclaration(
        name="run_experiment",
        description="Run the experiment command and return the results including metric, status, and constraint values.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Short description of what this experiment tests",
                },
            },
            "required": ["description"],
        },
    ),
    types.FunctionDeclaration(
        name="view_history",
        description="View the experiment history showing past results, metrics, and which experiments were kept.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "last_n": {
                    "type": "integer",
                    "description": "Number of recent experiments to show (default 10)",
                },
            },
        },
    ),
    types.FunctionDeclaration(
        name="bash",
        description="Run a shell command in the task directory. Use for git operations (commit, reset, diff, log) and inspecting files.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run",
                },
            },
            "required": ["command"],
        },
    ),
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _resolve_path(task_dir: Path, relative: str) -> Path | None:
    """Resolve a relative path safely within the task directory."""
    path = task_dir / relative
    if not path.resolve().is_relative_to(task_dir.resolve()):
        return None
    return path


def _tool_read_file(task_dir: Path, args: dict) -> dict:
    path = _resolve_path(task_dir, args["path"])
    if path is None:
        return {"error": "Path escapes task directory"}
    if not path.exists():
        return {"error": f"File not found: {args['path']}"}
    content = path.read_text()
    return {"content": content, "lines": len(content.splitlines())}


def _tool_edit_file(task_dir: Path, args: dict) -> dict:
    path = _resolve_path(task_dir, args["path"])
    if path is None:
        return {"error": "Path escapes task directory"}
    if not path.exists():
        return {"error": f"File not found: {args['path']}"}

    content = path.read_text()
    old = args["old_string"]
    new = args["new_string"]

    count = content.count(old)
    if count == 0:
        return {"error": f"old_string not found in {args['path']}"}
    if count > 1:
        return {"error": f"old_string found {count} times in {args['path']} — must be unique. Provide more context."}

    content = content.replace(old, new, 1)
    path.write_text(content)
    return {"success": True, "path": args["path"]}


def _tool_run_experiment(task_dir: Path, config: TaskConfig, tracker: ExperimentTracker, args: dict) -> dict:
    result, status, improved = run_and_record(
        config, task_dir, tracker,
        description=args.get("description", ""),
        log_path=task_dir / "run.log",
    )

    output = {
        "status": status,
        "wall_seconds": round(result.wall_seconds, 1),
        "improved": improved,
    }
    if result.metric is not None:
        output["metric"] = result.metric
        output["metric_name"] = config.metric.name
        best = tracker.best(direction=config.metric.direction)
        if best:
            output["best_so_far"] = best.metric_value
    if result.constraints:
        output["constraints"] = result.constraints
    if result.crashed:
        output["tail"] = result.tail[-2000:]
    return output


def _tool_view_history(task_dir: Path, config: TaskConfig, tracker: ExperimentTracker, args: dict) -> dict:
    last_n = args.get("last_n", 10)
    records = tracker.history(last_n=last_n)
    if not records:
        return {"message": "No experiments recorded yet."}

    fmt = config.metric.format
    rows = []
    for r in records:
        rows.append({
            "id": r.id,
            "commit": r.commit,
            "metric": f"{r.metric_value:{fmt}}" if r.status != "crash" else "---",
            "status": r.status,
            "wall_seconds": r.wall_seconds,
            "description": r.description,
        })

    best = tracker.best(direction=config.metric.direction)
    best_info = None
    if best:
        best_info = {"commit": best.commit, "metric": f"{best.metric_value:{fmt}}"}

    return {
        "experiments": rows,
        "best": best_info,
        "total": tracker.count(),
    }


def _tool_bash(task_dir: Path, args: dict) -> dict:
    command = args["command"]
    # Block dangerous commands
    dangerous = ["rm -rf /", "rm -rf ~", "sudo", "mkfs", "> /dev/"]
    for d in dangerous:
        if d in command:
            return {"error": f"Blocked dangerous command containing '{d}'"}

    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=task_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = {
            "returncode": proc.returncode,
            "stdout": proc.stdout[-3000:] if proc.stdout else "",
            "stderr": proc.stderr[-1000:] if proc.stderr else "",
        }
        return output
    except subprocess.TimeoutExpired:
        return {"error": "Command timed out after 30s"}


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

TOOL_DISPATCH = {
    "read_file": lambda td, cfg, tr, args: _tool_read_file(td, args),
    "edit_file": lambda td, cfg, tr, args: _tool_edit_file(td, args),
    "run_experiment": _tool_run_experiment,
    "view_history": _tool_view_history,
    "bash": lambda td, cfg, tr, args: _tool_bash(td, args),
}


def run_agent(
    task_dir: Path,
    config: TaskConfig,
    model: str = "gemini-3.1-pro-preview",
    max_iterations: int = 50,
    api_key: str | None = None,
) -> None:
    """Run the autonomous experiment agent loop."""

    # Setup
    task_dir = task_dir.resolve()
    db_path = task_dir / ".autoexp" / "experiments.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tracker = ExperimentTracker(db_path)

    # Load .env if present
    env_file = task_dir / ".env"
    if not env_file.exists():
        env_file = task_dir.parent / ".env"
    if not env_file.exists():
        env_file = task_dir.parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

    # Gemini client
    resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not resolved_key:
        raise RuntimeError("No API key. Set GEMINI_API_KEY env var, pass --api-key, or add to .env")
    client = genai.Client(api_key=resolved_key)

    # Read system prompt from program.md (source of truth, human-editable)
    program_path = task_dir / "program.md"
    if not program_path.exists():
        from .program_gen import write_program
        write_program(config, program_path)
        print(f"Generated initial {program_path}")
    system_prompt = program_path.read_text()
    print(f"Loaded program.md ({len(system_prompt)} chars)")

    # Tool config — manual function calling (we handle execution)
    tools = types.Tool(function_declarations=TOOL_DECLARATIONS)
    gen_config = types.GenerateContentConfig(
        tools=[tools],
        system_instruction=system_prompt,
        temperature=0.7,
    )

    # Initial message
    contents: list[types.Content] = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(
                text="Begin the autonomous experiment loop.\n\n"
                "Phase 1: Read ALL files (mutable + readonly) to deeply understand "
                "the model architecture, data pipeline, training loop, and evaluation. "
                "Then view experiment history to learn what's been tried and what worked.\n\n"
                "Phase 2: Based on your understanding, propose STRUCTURAL changes — "
                "not just hyperparameter tweaks. Think about different PEFT methods, "
                "custom loss functions, optimizer changes, learning rate schedules, "
                "training strategies. Write a hypothesis for each experiment explaining "
                "WHY it should help.\n\n"
                "Remember: if the run_experiment tool returns improved=false, you MUST "
                "git reset --hard HEAD~1 to revert. If improved=true, keep the commit "
                "and build on it."
            )],
        ),
    ]

    print(f"Agent started (model={model}, max_iterations={max_iterations})")
    print(f"Task: {config.name}")
    print(f"Metric: {config.metric.name} ({config.metric.direction})")
    print(f"Mutable files: {config.mutable_files}")
    print("-" * 60)

    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration}/{max_iterations} ---")

        # Call Gemini
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=gen_config,
            )
        except Exception as e:
            print(f"API error: {e}")
            time.sleep(5)
            continue

        if not response.candidates:
            print("Empty response from model, retrying...")
            time.sleep(2)
            continue

        model_content = response.candidates[0].content
        contents.append(model_content)

        # Check for text parts (model reasoning/commentary)
        for part in model_content.parts:
            if hasattr(part, "text") and part.text:
                print(f"\n[Agent]: {part.text[:500]}")

        # Check for function calls
        function_calls = [p for p in model_content.parts if hasattr(p, "function_call") and p.function_call]

        if not function_calls:
            # No tool calls — nudge toward action
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_text(
                    text="Don't just think — act. Make a concrete code change and run an experiment. "
                    "Try something STRUCTURAL: a different optimizer, a custom loss function, "
                    "a new learning rate schedule, layer-wise LoRA config, or a different PEFT method. "
                    "Write your hypothesis, edit the code, commit, and run."
                )],
            ))
            continue

        # Execute each tool call and collect responses
        response_parts = []
        for part in function_calls:
            fc = part.function_call
            tool_name = fc.name
            tool_args = dict(fc.args) if fc.args else {}

            print(f"  [Tool] {tool_name}({json.dumps(tool_args, default=str)[:200]})")

            handler = TOOL_DISPATCH.get(tool_name)
            if handler:
                result = handler(task_dir, config, tracker, tool_args)
            else:
                result = {"error": f"Unknown tool: {tool_name}"}

            # Truncate large results
            result_str = json.dumps(result, default=str)
            if len(result_str) > 10000:
                result = {"truncated": result_str[:10000] + "..."}

            print(f"  [Result] {json.dumps(result, default=str)[:300]}")

            response_parts.append(
                types.Part.from_function_response(
                    name=tool_name,
                    response=result,
                )
            )

        # Send tool results back
        contents.append(types.Content(role="user", parts=response_parts))

        # Trim conversation if it gets too long (keep system + last 40 turns)
        if len(contents) > 60:
            contents = contents[:1] + contents[-40:]

    tracker.close()
    print(f"\n{'='*60}")
    print(f"Agent finished after {iteration} iterations")
