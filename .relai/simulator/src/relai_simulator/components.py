from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from german_tutor.cli import CHECKPOINT_DB
from german_tutor.persistence import DEFAULT_DB_PATH, Store
from german_tutor.tools_lc import make_tools

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LEARNER_ID = "relai-simulator"


def get_learner_state(*, learner_id: str | None = None, db_path: str | None = None) -> str:
    return _invoke_tool("get_learner_state", learner_id=learner_id, db_path=db_path)


def set_level(*, level: str, learner_id: str | None = None, db_path: str | None = None) -> str:
    return _invoke_tool("set_level", learner_id=learner_id, db_path=db_path, level=level)


def get_next_unit(*, learner_id: str | None = None, db_path: str | None = None) -> str:
    return _invoke_tool("get_next_unit", learner_id=learner_id, db_path=db_path)


def get_unit_details(*, unit_id: str, learner_id: str | None = None, db_path: str | None = None) -> str:
    return _invoke_tool(
        "get_unit_details",
        learner_id=learner_id,
        db_path=db_path,
        unit_id=unit_id,
    )


def save_lesson_pointer(
    *,
    unit_id: str,
    step: int,
    learner_id: str | None = None,
    db_path: str | None = None,
) -> str:
    return _invoke_tool(
        "save_lesson_pointer",
        learner_id=learner_id,
        db_path=db_path,
        unit_id=unit_id,
        step=step,
    )


def cache_lesson(
    *,
    unit_id: str,
    lesson_json: str,
    learner_id: str | None = None,
    db_path: str | None = None,
) -> str:
    return _invoke_tool(
        "cache_lesson",
        learner_id=learner_id,
        db_path=db_path,
        unit_id=unit_id,
        lesson_json=lesson_json,
    )


def get_cached_lesson(
    *,
    unit_id: str,
    learner_id: str | None = None,
    db_path: str | None = None,
) -> str:
    return _invoke_tool(
        "get_cached_lesson",
        learner_id=learner_id,
        db_path=db_path,
        unit_id=unit_id,
    )


def record_attempt(
    *,
    unit_id: str,
    exercise_id: str,
    correct: bool,
    score: float,
    learner_id: str | None = None,
    db_path: str | None = None,
) -> str:
    return _invoke_tool(
        "record_attempt",
        learner_id=learner_id,
        db_path=db_path,
        unit_id=unit_id,
        exercise_id=exercise_id,
        correct=correct,
        score=score,
    )


def log_error(
    *,
    category: str,
    example: str,
    correction: str,
    learner_id: str | None = None,
    db_path: str | None = None,
) -> str:
    return _invoke_tool(
        "log_error",
        learner_id=learner_id,
        db_path=db_path,
        category=category,
        example=example,
        correction=correction,
    )


def update_mastery(
    *,
    unit_id: str,
    mastery: float,
    status: str,
    learner_id: str | None = None,
    db_path: str | None = None,
) -> str:
    return _invoke_tool(
        "update_mastery",
        learner_id=learner_id,
        db_path=db_path,
        unit_id=unit_id,
        mastery=mastery,
        status=status,
    )


def add_vocab(
    *,
    lemma: str,
    gloss: str,
    learner_id: str | None = None,
    db_path: str | None = None,
) -> str:
    return _invoke_tool(
        "add_vocab",
        learner_id=learner_id,
        db_path=db_path,
        lemma=lemma,
        gloss=gloss,
    )


def get_due_vocab(*, learner_id: str | None = None, db_path: str | None = None) -> str:
    return _invoke_tool("get_due_vocab", learner_id=learner_id, db_path=db_path)


def review_vocab(
    *,
    card_id: int,
    quality: int,
    learner_id: str | None = None,
    db_path: str | None = None,
) -> str:
    return _invoke_tool(
        "review_vocab",
        learner_id=learner_id,
        db_path=db_path,
        card_id=card_id,
        quality=quality,
    )


def log_session_summary(
    *,
    summary: str,
    learner_id: str | None = None,
    db_path: str | None = None,
) -> str:
    return _invoke_tool(
        "log_session_summary",
        learner_id=learner_id,
        db_path=db_path,
        summary=summary,
    )


def simulator_checkpoint_db_path() -> str:
    return str(_resolve_path(os.getenv("RELAI_SIMULATOR_CHECKPOINT_DB"), CHECKPOINT_DB))


def _invoke_tool(tool_name: str, *, learner_id: str | None, db_path: str | None, **kwargs: Any) -> str:
    resolved_learner_id = learner_id or os.getenv("TUTOR_LEARNER_ID", DEFAULT_LEARNER_ID)
    resolved_db_path = _resolve_path(db_path or os.getenv("TUTOR_DB"), DEFAULT_DB_PATH)
    store = Store(resolved_db_path)
    try:
        store.get_or_create_learner(resolved_learner_id)
        registry = _tool_registry(store, resolved_learner_id)
        return registry[tool_name].invoke(kwargs)
    finally:
        store.close()


def _tool_registry(store: Store, learner_id: str) -> dict[str, object]:
    registry: dict[str, object] = {}
    for group in make_tools(store, learner_id).values():
        for tool in group:
            name = getattr(tool, "name", None)
            if isinstance(name, str) and name and name not in registry:
                registry[name] = tool
    return registry


def _resolve_path(raw_value: str | os.PathLike[str] | None, default_value: str | os.PathLike[str]) -> Path:
    path = Path(raw_value or default_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
