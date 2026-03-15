"""
Git operations for the experiment loop: branching, committing,
snapshotting mutable files, and reverting on discard.

All functions gracefully handle missing git repos (for Colab / non-git use).
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


class GitError(Exception):
    pass


def _run(cmd: list[str], cwd: str | Path) -> str:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise GitError(f"Command {' '.join(cmd)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def has_git(repo: str | Path) -> bool:
    """Check if the directory is inside a git repo."""
    try:
        _run(["git", "rev-parse", "--git-dir"], repo)
        return True
    except (GitError, FileNotFoundError):
        return False


def current_branch(repo: str | Path) -> str:
    return _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo)


def current_commit(repo: str | Path, short: bool = True) -> str:
    """Return current git commit hash, or a unique timestamp-based ID if not in a git repo."""
    try:
        cmd = ["git", "rev-parse"]
        if short:
            cmd.append("--short")
        cmd.append("HEAD")
        return _run(cmd, repo)
    except (GitError, FileNotFoundError):
        # No git — hash the directory path + timestamp as a fallback ID
        import time
        tag = f"{Path(repo).resolve()}:{time.time()}"
        return hashlib.sha1(tag.encode()).hexdigest()[:7]


def branch_exists(repo: str | Path, branch: str) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", branch],
        cwd=repo, capture_output=True, text=True,
    )
    return result.returncode == 0


def create_branch(repo: str | Path, branch: str) -> None:
    _run(["git", "checkout", "-b", branch], repo)


def checkout(repo: str | Path, ref: str) -> None:
    _run(["git", "checkout", ref], repo)


def commit(repo: str | Path, files: list[str], message: str) -> str:
    """Stage files and commit. Returns the short commit hash."""
    for f in files:
        _run(["git", "add", f], repo)
    _run(["git", "commit", "-m", message], repo)
    return current_commit(repo, short=True)


def reset_to(repo: str | Path, commit_ref: str) -> None:
    """Hard reset to a specific commit. Used to discard failed experiments."""
    _run(["git", "reset", "--hard", commit_ref], repo)


def snapshot_files(repo: str | Path, files: list[str]) -> dict[str, str]:
    """Read the current contents of mutable files for the config snapshot."""
    snapshot = {}
    for f in files:
        p = Path(repo) / f
        if p.exists():
            snapshot[f] = p.read_text()
    return snapshot


def is_clean(repo: str | Path) -> bool:
    result = _run(["git", "status", "--porcelain"], repo)
    return result == ""
