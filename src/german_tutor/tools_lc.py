"""LangChain tools that read/write durable learner progress.

Built by a factory bound to a specific (store, learner_id) via closures, so the
LLM never has to supply the learner id and can't address another learner. The
tools return JSON strings the model can parse.

Note: conversation/graph state is persisted separately by LangGraph's checkpointer
(see graph.py). These tools manage *domain* progress: level, mastery, the lesson
pointer, the spaced-repetition deck, and the error log.
"""

from __future__ import annotations

import json
import os

from langchain_core.tools import BaseTool, tool

from . import curriculum as curr
from .persistence import Store


def make_tools(store: Store, learner_id: str) -> dict[str, list[BaseTool]]:
    """Return tool groupings keyed by specialist role."""

    @tool
    def get_learner_state() -> str:
        """Return the learner's level, lesson pointer, progress, due-vocab count and weak spots."""
        pointer = store.get_pointer(learner_id)
        progress = store.progress_summary(learner_id)
        return json.dumps(
            {
                "level": store.get_level(learner_id),
                "pointer": pointer,
                "progress": progress,
                "due_vocab": store.count_due_vocab(learner_id),
                "weak_spots": store.top_errors(learner_id, limit=5),
                "last_session": store.last_session(learner_id),
                "is_absolute_beginner": pointer is None and not progress,
            }
        )

    @tool
    def set_level(level: str) -> str:
        """Set the learner's CEFR level. `level` must be one of A1, A2, B1, B2."""
        level = level.upper()
        if level not in curr.LEVELS:
            return json.dumps({"ok": False, "error": f"level must be one of {curr.LEVELS}"})
        store.set_level(learner_id, level)
        return json.dumps({"ok": True, "level": level})

    @tool
    def get_next_unit() -> str:
        """Return the next curriculum unit to study, given level and completed units."""
        unit = curr.next_unit(store.get_level(learner_id), store.completed_unit_ids(learner_id))
        return json.dumps(unit or {"done": True})

    @tool
    def get_unit_details(unit_id: str) -> str:
        """Return curriculum details (objectives, grammar, vocab theme) for a unit id."""
        return json.dumps(curr.get_unit(unit_id) or {"error": "unknown unit"})

    @tool
    def save_lesson_pointer(unit_id: str, step: int) -> str:
        """Save how far the learner has progressed in a lesson so it can be resumed."""
        store.set_pointer(learner_id, unit_id, step)
        return json.dumps({"ok": True})

    @tool
    def cache_lesson(unit_id: str, lesson_json: str) -> str:
        """Persist a generated lesson (as JSON) so resuming shows the identical lesson."""
        store.cache_lesson(learner_id, unit_id, lesson_json)
        return json.dumps({"ok": True})

    @tool
    def get_cached_lesson(unit_id: str) -> str:
        """Return the previously cached lesson JSON for a unit, or null if none exists."""
        cached = store.get_cached_lesson(learner_id, unit_id)
        return cached if cached else json.dumps(None)

    @tool
    def record_attempt(unit_id: str, exercise_id: str, correct: bool, score: float) -> str:
        """Record the result of one graded exercise attempt."""
        store.record_attempt(learner_id, unit_id, exercise_id, correct, score)
        return json.dumps({"ok": True})

    @tool
    def log_error(category: str, example: str, correction: str) -> str:
        """Log a learner mistake so it can drive targeted review later."""
        store.log_error(learner_id, category, example, correction)
        return json.dumps({"ok": True})

    @tool
    def update_mastery(unit_id: str, mastery: float, status: str) -> str:
        """Update mastery (0-1) and status ('in_progress'|'completed'|'review') for a unit."""
        if status not in ("in_progress", "completed", "review"):
            status = "in_progress"
        store.update_mastery(learner_id, unit_id, mastery, status)
        return json.dumps({"ok": True})

    @tool
    def add_vocab(lemma: str, gloss: str) -> str:
        """Add a vocab word to the spaced-repetition deck (include article for nouns)."""
        store.add_vocab(learner_id, lemma, gloss)
        return json.dumps({"ok": True})

    @tool
    def get_due_vocab() -> str:
        """Return vocab cards due for review today (id, lemma, gloss)."""
        return json.dumps(store.due_vocab(learner_id))

    @tool
    def review_vocab(card_id: int, quality: int) -> str:
        """Grade a vocab recall (quality 0-5) and reschedule the card via SM-2."""
        return json.dumps(store.review_vocab(learner_id, card_id, quality) or {"error": "not found"})

    @tool
    def log_session_summary(summary: str) -> str:
        """Save a one-line recap of this session for the next 'welcome back'."""
        store.log_session(learner_id, summary)
        return json.dumps({"ok": True})

    @tool
    def speak_german(text: str) -> str:
        """Generate spoken German audio for a word, phrase, or sentence so the learner
        can hear the pronunciation. Saves an audio file and (on macOS) plays it aloud.
        Use it when introducing a new word or a key example sentence. Pass natural
        German text only (no phonetic spelling)."""
        from .tts import play, synthesize, tts_available

        if not tts_available():
            return json.dumps(
                {"ok": False, "error": "TTS unavailable on this system (no macOS 'say')."}
            )
        path = synthesize(text)
        played = False
        if path is not None and os.getenv("TUTOR_TTS_AUTOPLAY", "1") != "0":
            played = play(path)
        return json.dumps(
            {"ok": path is not None, "audio_path": str(path) if path else None,
             "played": played, "text": text}
        )

    state = [get_learner_state, get_next_unit, get_unit_details]
    return {
        "state": state,
        "placement": [*state, set_level],
        "grammar": [*state, speak_german],
        "vocab": [*state, add_vocab, get_due_vocab, review_vocab, log_error, speak_german],
        "exercise": [*state, record_attempt, log_error],
        "lesson": [
            *state, get_cached_lesson, cache_lesson, save_lesson_pointer,
            add_vocab, record_attempt, log_error, update_mastery, speak_german,
        ],
        "conversation": [*state, log_error, speak_german],
        "progress": [*state, update_mastery, log_session_summary],
        "concierge": state,
    }
