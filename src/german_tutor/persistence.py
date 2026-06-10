"""Durable learner profile & progress store (SQLite).

This is *separate* from the conversation transcript (which LangGraph keeps in its
own checkpointer). This store holds the cross-session learning state: current
level, per-unit mastery, a lesson pointer (so resume continues mid-lesson), the
spaced-repetition vocab deck, attempt history, errors, and session summaries.

Thread-safety: LangGraph runs node/tool work on worker threads and may execute
tool calls in parallel. A single shared sqlite3 connection used concurrently
raises SQLITE_MISUSE, so this Store hands each thread its own connection via
thread-local storage and uses WAL mode for concurrent read/write.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import date, datetime, timezone
from pathlib import Path

from .srs import SrsState, review

DEFAULT_DB_PATH = Path(".german_tutor/progress.db")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today() -> str:
    return date.today().isoformat()


SCHEMA = """
CREATE TABLE IF NOT EXISTS learners (
    id TEXT PRIMARY KEY,
    name TEXT,
    current_level TEXT DEFAULT 'A1',
    created_at TEXT,
    last_active TEXT
);
CREATE TABLE IF NOT EXISTS progress (
    learner_id TEXT,
    unit_id TEXT,
    status TEXT DEFAULT 'in_progress',
    mastery REAL DEFAULT 0.0,
    attempts INTEGER DEFAULT 0,
    last_seen TEXT,
    PRIMARY KEY (learner_id, unit_id)
);
CREATE TABLE IF NOT EXISTS lesson_pointer (
    learner_id TEXT PRIMARY KEY,
    unit_id TEXT,
    step INTEGER DEFAULT 0,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS cached_lessons (
    learner_id TEXT,
    unit_id TEXT,
    lesson_json TEXT,
    created_at TEXT,
    PRIMARY KEY (learner_id, unit_id)
);
CREATE TABLE IF NOT EXISTS vocab_cards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    learner_id TEXT,
    lemma TEXT,
    gloss TEXT,
    ease REAL DEFAULT 2.5,
    interval INTEGER DEFAULT 0,
    reps INTEGER DEFAULT 0,
    due_date TEXT,
    UNIQUE (learner_id, lemma)
);
CREATE TABLE IF NOT EXISTS attempts (
    learner_id TEXT,
    unit_id TEXT,
    exercise_id TEXT,
    correct INTEGER,
    score REAL,
    ts TEXT
);
CREATE TABLE IF NOT EXISTS errors (
    learner_id TEXT,
    category TEXT,
    example TEXT,
    correction TEXT,
    ts TEXT
);
CREATE TABLE IF NOT EXISTS session_log (
    learner_id TEXT,
    summary TEXT,
    ts TEXT
);
"""


class Store:
    """Thin SQLite wrapper. One connection per Store; safe for the CLI's single thread."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        # Bootstrap the schema once on the creating thread's connection.
        boot = self._new_conn()
        boot.executescript(SCHEMA)
        boot.commit()

    def _new_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        self._local.conn = conn
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        """A connection owned by the current thread (created on first use)."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._new_conn()
        return conn

    # ----- learner / level -------------------------------------------------
    def get_or_create_learner(self, learner_id: str, name: str | None = None) -> dict:
        row = self.conn.execute(
            "SELECT * FROM learners WHERE id = ?", (learner_id,)
        ).fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO learners (id, name, created_at, last_active) VALUES (?,?,?,?)",
                (learner_id, name, _now(), _now()),
            )
            self.conn.commit()
            row = self.conn.execute(
                "SELECT * FROM learners WHERE id = ?", (learner_id,)
            ).fetchone()
        else:
            self.conn.execute(
                "UPDATE learners SET last_active = ? WHERE id = ?", (_now(), learner_id)
            )
            self.conn.commit()
        return dict(row)

    def is_returning(self, learner_id: str) -> bool:
        row = self.conn.execute(
            "SELECT COUNT(*) AS c FROM attempts WHERE learner_id = ?", (learner_id,)
        ).fetchone()
        return bool(row["c"])

    def set_level(self, learner_id: str, level: str) -> None:
        self.conn.execute(
            "UPDATE learners SET current_level = ? WHERE id = ?", (level, learner_id)
        )
        self.conn.commit()

    def get_level(self, learner_id: str) -> str:
        row = self.conn.execute(
            "SELECT current_level FROM learners WHERE id = ?", (learner_id,)
        ).fetchone()
        return row["current_level"] if row else "A1"

    # ----- progress / mastery ---------------------------------------------
    def completed_unit_ids(self, learner_id: str) -> set[str]:
        rows = self.conn.execute(
            "SELECT unit_id FROM progress WHERE learner_id = ? AND status = 'completed'",
            (learner_id,),
        ).fetchall()
        return {r["unit_id"] for r in rows}

    def update_mastery(self, learner_id: str, unit_id: str, mastery: float, status: str) -> None:
        mastery = max(0.0, min(1.0, mastery))
        self.conn.execute(
            """INSERT INTO progress (learner_id, unit_id, status, mastery, attempts, last_seen)
               VALUES (?,?,?,?,1,?)
               ON CONFLICT(learner_id, unit_id) DO UPDATE SET
                 mastery = excluded.mastery,
                 status = excluded.status,
                 attempts = progress.attempts + 1,
                 last_seen = excluded.last_seen""",
            (learner_id, unit_id, status, mastery, _now()),
        )
        self.conn.commit()

    def progress_summary(self, learner_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT unit_id, status, mastery, attempts FROM progress WHERE learner_id = ? ORDER BY unit_id",
            (learner_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ----- lesson pointer (resume) ----------------------------------------
    def set_pointer(self, learner_id: str, unit_id: str, step: int) -> None:
        self.conn.execute(
            """INSERT INTO lesson_pointer (learner_id, unit_id, step, updated_at)
               VALUES (?,?,?,?)
               ON CONFLICT(learner_id) DO UPDATE SET
                 unit_id = excluded.unit_id, step = excluded.step, updated_at = excluded.updated_at""",
            (learner_id, unit_id, step, _now()),
        )
        self.conn.commit()

    def get_pointer(self, learner_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT unit_id, step FROM lesson_pointer WHERE learner_id = ?", (learner_id,)
        ).fetchone()
        return dict(row) if row else None

    # ----- cached lessons (deterministic resume) --------------------------
    def cache_lesson(self, learner_id: str, unit_id: str, lesson_json: str) -> None:
        self.conn.execute(
            """INSERT INTO cached_lessons (learner_id, unit_id, lesson_json, created_at)
               VALUES (?,?,?,?)
               ON CONFLICT(learner_id, unit_id) DO UPDATE SET lesson_json = excluded.lesson_json""",
            (learner_id, unit_id, lesson_json, _now()),
        )
        self.conn.commit()

    def get_cached_lesson(self, learner_id: str, unit_id: str) -> str | None:
        row = self.conn.execute(
            "SELECT lesson_json FROM cached_lessons WHERE learner_id = ? AND unit_id = ?",
            (learner_id, unit_id),
        ).fetchone()
        return row["lesson_json"] if row else None

    # ----- attempts / errors ----------------------------------------------
    def record_attempt(
        self, learner_id: str, unit_id: str, exercise_id: str, correct: bool, score: float
    ) -> None:
        self.conn.execute(
            "INSERT INTO attempts (learner_id, unit_id, exercise_id, correct, score, ts) VALUES (?,?,?,?,?,?)",
            (learner_id, unit_id, exercise_id, int(correct), score, _now()),
        )
        self.conn.commit()

    def log_error(self, learner_id: str, category: str, example: str, correction: str) -> None:
        self.conn.execute(
            "INSERT INTO errors (learner_id, category, example, correction, ts) VALUES (?,?,?,?,?)",
            (learner_id, category, example, correction, _now()),
        )
        self.conn.commit()

    def top_errors(self, learner_id: str, limit: int = 5) -> list[dict]:
        rows = self.conn.execute(
            """SELECT category, COUNT(*) AS n FROM errors WHERE learner_id = ?
               GROUP BY category ORDER BY n DESC LIMIT ?""",
            (learner_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ----- vocab / SRS -----------------------------------------------------
    def add_vocab(self, learner_id: str, lemma: str, gloss: str) -> None:
        self.conn.execute(
            """INSERT OR IGNORE INTO vocab_cards (learner_id, lemma, gloss, due_date)
               VALUES (?,?,?,?)""",
            (learner_id, lemma, gloss, _today()),
        )
        self.conn.commit()

    def due_vocab(self, learner_id: str, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            """SELECT id, lemma, gloss FROM vocab_cards
               WHERE learner_id = ? AND (due_date IS NULL OR due_date <= ?)
               ORDER BY due_date LIMIT ?""",
            (learner_id, _today(), limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def count_due_vocab(self, learner_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS c FROM vocab_cards WHERE learner_id = ? AND (due_date IS NULL OR due_date <= ?)",
            (learner_id, _today()),
        ).fetchone()
        return row["c"]

    def review_vocab(self, learner_id: str, card_id: int, quality: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM vocab_cards WHERE id = ? AND learner_id = ?", (card_id, learner_id)
        ).fetchone()
        if row is None:
            return None
        new = review(SrsState(ease=row["ease"], interval=row["interval"], reps=row["reps"]), quality)
        due = date.fromordinal(date.today().toordinal() + max(1, new.interval)).isoformat()
        self.conn.execute(
            "UPDATE vocab_cards SET ease=?, interval=?, reps=?, due_date=? WHERE id=?",
            (new.ease, new.interval, new.reps, due, card_id),
        )
        self.conn.commit()
        return {"lemma": row["lemma"], "interval": new.interval, "due_date": due}

    # ----- session log -----------------------------------------------------
    def log_session(self, learner_id: str, summary: str) -> None:
        self.conn.execute(
            "INSERT INTO session_log (learner_id, summary, ts) VALUES (?,?,?)",
            (learner_id, summary, _now()),
        )
        self.conn.commit()

    def last_session(self, learner_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT summary, ts FROM session_log WHERE learner_id = ? ORDER BY ts DESC LIMIT 1",
            (learner_id,),
        ).fetchone()
        return dict(row) if row else None

    def welcome_back(self, learner_id: str) -> str:
        """Human-readable 'where you left off' summary for startup."""
        level = self.get_level(learner_id)
        pointer = self.get_pointer(learner_id)
        due = self.count_due_vocab(learner_id)
        errs = self.top_errors(learner_id, limit=2)
        parts = [f"Level: {level}"]
        if pointer:
            parts.append(f"last lesson: {pointer['unit_id']} (step {pointer['step']})")
        if due:
            parts.append(f"{due} vocab card(s) due")
        if errs:
            parts.append("weak spots: " + ", ".join(e["category"] for e in errs))
        return " · ".join(parts)

    def close(self) -> None:
        self.conn.close()
