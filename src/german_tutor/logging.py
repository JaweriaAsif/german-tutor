"""Readable JSON conversation logs for the German tutor.

Each run writes a numbered log (logs/conversation-NNN.json) containing a `rounds`
list of {user, assistant} turns. These logs are handy for replaying a session and,
in particular, for feeding a real (mis)behaving session back into RELAI:

    relai learning-env create --log-file logs/conversation-007.json \\
        --feedback "the tutor kept asking 'do you want to continue?'"

so a learning environment can be recreated from an actual transcript instead of
hand-written.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from threading import Lock

DEFAULT_LOG_DIR = Path("logs")
LOG_FILE_PATTERN = re.compile(r"^conversation-(\d+)\.json$")
_DE_MARKER = re.compile(r"\[\[de:(.+?)\]\]", re.DOTALL)


def strip_markers(text: str) -> str:
    """Remove [[de:...]] audio markers so the log reads as plain text."""
    return _DE_MARKER.sub(lambda m: m.group(1).strip(), text)


def next_numbered_log_file(log_dir: Path = DEFAULT_LOG_DIR) -> Path:
    """Reserve and return the next logs/conversation-NNN.json path."""
    log_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        int(m.group(1))
        for p in log_dir.glob("conversation-*.json")
        if (m := LOG_FILE_PATTERN.match(p.name))
    ]
    n = max(existing, default=0) + 1
    while True:
        candidate = log_dir / f"conversation-{n:03d}.json"
        try:
            with candidate.open("x", encoding="utf-8"):
                pass
            return candidate
        except FileExistsError:
            n += 1


class ConversationLog:
    """Append-only JSON log of {user, assistant} rounds, plus session metadata."""

    def __init__(self, path: str | Path, *, learner_id: str | None = None) -> None:
        self.path = Path(path)
        self._lock = Lock()
        self._learner_id = learner_id

    def append_round(self, *, user: str, assistant: str | None) -> None:
        with self._lock:
            log = self._read()
            log["rounds"].append(
                {"user": user, "assistant": strip_markers(assistant or "")}
            )
            self._write(log)

    def _read(self) -> dict:
        if not self.path.exists() or not self.path.read_text(encoding="utf-8").strip():
            return {"learner_id": self._learner_id, "rounds": []}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("rounds"), list):
            return data
        raise ValueError(f"{self.path} is not a conversation log")

    def _write(self, log: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(log, ensure_ascii=False, indent=2)
        self.path.write_text(f"{payload}\n", encoding="utf-8")
