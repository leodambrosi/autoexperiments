"""
Experiment tracker: SQLite-backed storage for experiment results with
history queries, lineage tracing, and TSV export.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentRecord:
    id: int
    timestamp: float
    commit: str
    parent_commit: str | None
    metric_name: str
    metric_value: float | None
    constraints: dict[str, float]
    status: str  # "keep", "discard", "crash", "timeout", "constraint_violated"
    description: str
    wall_seconds: float
    config_snapshot: str  # JSON blob of mutable file contents at time of run


DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    git_commit TEXT NOT NULL,
    parent_commit TEXT,
    metric_name TEXT NOT NULL,
    metric_value REAL,
    constraints_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    wall_seconds REAL NOT NULL DEFAULT 0.0,
    config_snapshot TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_commit ON experiments(git_commit);
"""


class ExperimentTracker:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(DB_SCHEMA)

    def log(
        self,
        commit: str,
        metric_name: str,
        metric_value: float | None,
        status: str,
        description: str = "",
        parent_commit: str | None = None,
        constraints: dict[str, float] | None = None,
        wall_seconds: float = 0.0,
        config_snapshot: dict[str, str] | None = None,
    ) -> int:
        """Log an experiment. Returns the experiment ID."""
        self._conn.execute(
            """INSERT INTO experiments
               (timestamp, git_commit, parent_commit, metric_name, metric_value,
                constraints_json, status, description, wall_seconds, config_snapshot)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(),
                commit,
                parent_commit,
                metric_name,
                metric_value,
                json.dumps(constraints or {}),
                status,
                description,
                wall_seconds,
                json.dumps(config_snapshot or {}),
            ),
        )
        self._conn.commit()
        return self._conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def best(self, direction: str = "minimize") -> ExperimentRecord | None:
        """Return the best kept experiment."""
        order = "ASC" if direction == "minimize" else "DESC"
        row = self._conn.execute(
            f"SELECT * FROM experiments WHERE status = 'keep' ORDER BY metric_value {order} LIMIT 1"
        ).fetchone()
        return self._row_to_record(row) if row else None

    def history(self, last_n: int = 20) -> list[ExperimentRecord]:
        """Return the most recent experiments."""
        rows = self._conn.execute(
            "SELECT * FROM experiments ORDER BY id DESC LIMIT ?", (last_n,)
        ).fetchall()
        return [self._row_to_record(r) for r in reversed(rows)]

    def lineage(self, commit: str) -> list[ExperimentRecord]:
        """Trace the chain of kept experiments leading to this commit."""
        chain = []
        current = commit
        while current:
            row = self._conn.execute(
                "SELECT * FROM experiments WHERE git_commit = ? AND status = 'keep' LIMIT 1",
                (current,),
            ).fetchone()
            if not row:
                break
            record = self._row_to_record(row)
            chain.append(record)
            current = record.parent_commit
        chain.reverse()
        return chain

    def count(self, status: str | None = None) -> int:
        if status:
            return self._conn.execute(
                "SELECT COUNT(*) FROM experiments WHERE status = ?", (status,)
            ).fetchone()[0]
        return self._conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]

    def export_tsv(self, path: str | Path) -> None:
        """Export all experiments to a TSV file."""
        rows = self._conn.execute("SELECT * FROM experiments ORDER BY id").fetchall()
        lines = ["commit\tmetric\tstatus\twall_seconds\tdescription"]
        for r in rows:
            lines.append(f"{r['git_commit']}\t{r['metric_value']}\t{r['status']}\t{r['wall_seconds']:.1f}\t{r['description']}")
        Path(path).write_text("\n".join(lines) + "\n")

    def close(self):
        self._conn.close()

    def _row_to_record(self, row: sqlite3.Row) -> ExperimentRecord:
        return ExperimentRecord(
            id=row["id"],
            timestamp=row["timestamp"],
            commit=row["git_commit"],
            parent_commit=row["parent_commit"],
            metric_name=row["metric_name"],
            metric_value=row["metric_value"],
            constraints=json.loads(row["constraints_json"]),
            status=row["status"],
            description=row["description"],
            wall_seconds=row["wall_seconds"],
            config_snapshot=row["config_snapshot"],
        )
