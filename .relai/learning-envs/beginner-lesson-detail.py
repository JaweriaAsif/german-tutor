"""RELAI learning environment for detailed beginner lesson pacing."""

import json

from relai import (
    CodeEvaluator,
    EvaluationResult,
    LLMJudgeEvaluator,
    ModelSpec,
    PersonaInput,
    RELAIEnvironment,
)

TAGS = ["end-to-end", "absolute-beginner-lesson-detailed-and-paced"]
MODEL = ModelSpec(name="gpt-5.4")


def _ok_response(**payload):
    return json.dumps({"ok": True, **payload})


def mock_set_level(*args, **kwargs):
    """Mock for set_level: Writes the learner's CEFR level to persistent SQLite state."""
    return _ok_response(level=str(kwargs.get("level", "A1")).upper())


def mock_save_lesson_pointer(*args, **kwargs):
    """Mock for save_lesson_pointer: Writes lesson resume state to the persistent learner database."""
    return _ok_response()


def mock_cache_lesson(*args, **kwargs):
    """Mock for cache_lesson: Persists generated lesson content for future resume behavior."""
    return _ok_response()


def mock_record_attempt(*args, **kwargs):
    """Mock for record_attempt: Writes graded exercise attempts to the persistent learner database."""
    return _ok_response()


def mock_log_error(*args, **kwargs):
    """Mock for log_error: Writes learner mistakes to the durable error log used by later review flows."""
    return _ok_response()


def mock_update_mastery(*args, **kwargs):
    """Mock for update_mastery: Mutates persisted mastery and unit status values for the learner."""
    return _ok_response()


def mock_add_vocab(*args, **kwargs):
    """Mock for add_vocab: Writes new vocabulary cards into the learner's spaced-repetition deck."""
    return _ok_response()


def mock_review_vocab(*args, **kwargs):
    """Mock for review_vocab: Updates spaced-repetition scheduling state in the persistent learner database."""
    return json.dumps(
        {
            "id": 1,
            "lemma": "der Tisch",
            "gloss": "the table",
            "interval": 1,
            "quality": int(kwargs.get("quality", 4)),
        }
    )


def mock_log_session_summary(*args, **kwargs):
    """Mock for log_session_summary: Writes a session recap that changes later welcome-back behavior."""
    return _ok_response()


def _event_field(event, *names):
    for name in names:
        if isinstance(event, dict) and name in event:
            return event[name]
        value = getattr(event, name, None)
        if value is not None:
            return value
    return None


def _simulation_events(simulation_result):
    events = getattr(simulation_result, "events", None)
    if events is None and isinstance(simulation_result, dict):
        events = simulation_result.get("events")
    return list(events or [])


def _assistant_messages(simulation_result):
    messages = []
    for event in _simulation_events(simulation_result):
        event_type = _event_field(event, "type", "event_type", "kind", "name")
        if event_type not in {"agent_message", "assistant_message"}:
            continue
        content = _event_field(event, "content", "message", "text")
        if isinstance(content, str) and content.strip():
            messages.append(content.strip())
    return messages


def check_multi_turn_segment(simulation_result):
    assistant_messages = _assistant_messages(simulation_result)
    if len(assistant_messages) >= 2:
        return EvaluationResult(
            score=1.0,
            feedback=(
                f"Observed {len(assistant_messages)} tutor turns, so the evaluation covered "
                "a short multi-turn lesson segment rather than a single reply."
            ),
        )
    return EvaluationResult(
        score=0.0,
        feedback=(
            "Expected at least 2 tutor turns to evaluate a short lesson segment with a "
            f"checkpoint and follow-up, but observed {len(assistant_messages)} tutor turn(s)."
        ),
    )


environment = RELAIEnvironment(
    schema_version="relai.learning_environment.v1",
    id="beginner-lesson-detail",
    name="Detailed A1 Lesson",
    description="Tests whether the tutor gives a slow, detailed beginner lesson with a comprehension check before advancing.",
    tags=TAGS,
    input=PersonaInput(
        seed_message="/lesson",
        persona=(
            "You are an absolute beginner with no German knowledge. You want very slow, "
            "thorough teaching. If the tutor asks a checkpoint question, answer briefly "
            "in English based only on what was just taught. If the tutor moves on too "
            "quickly, ask it to slow down."
        ),
        intent=(
            "Get a beginner-friendly A1 lesson one tiny step at a time, with the tutor "
            "checking understanding before it advances."
        ),
        max_turns=3,
        model=MODEL,
    ),
    mocks={
        "set_level": mock_set_level,
        "save_lesson_pointer": mock_save_lesson_pointer,
        "cache_lesson": mock_cache_lesson,
        "record_attempt": mock_record_attempt,
        "log_error": mock_log_error,
        "update_mastery": mock_update_mastery,
        "add_vocab": mock_add_vocab,
        "review_vocab": mock_review_vocab,
        "log_session_summary": mock_log_session_summary,
    },
    evaluators=[
        CodeEvaluator(
            id="multi-turn-lesson-segment",
            description="Checks that the transcript includes a short multi-turn lesson segment.",
            evaluate=check_multi_turn_segment,
        ),
        LLMJudgeEvaluator(
            id="detailed-beginner-lesson-quality",
            description="Judges whether the tutor teaches an absolute beginner slowly, clearly, and accurately.",
            instructions="""
Judge the full observed transcript, not just the final reply.

Score only these criteria:
- The tutor teaches at A1 level for an absolute beginner and does not assume prior German knowledge.
- The lesson is detailed and paced slowly, one small step at a time, rather than terse or rushed.
- The tutor includes multiple concrete German examples with English glosses in the observed segment.
- New German words introduced in the segment are translated or explained in English.
- The tutor gives simple pronunciation help when useful for a newly introduced beginner form.
- The tutor checks understanding before advancing and responds appropriately to the learner's answer or confusion.
- The tutor does not invent grammar rules or make materially incorrect grammar claims.

If the transcript does not show at least two substantive assistant lesson turns, or it never includes a comprehension check before moving forward, cap the score at 0.4.

When deducting points, feedback must:
- name the failed criterion or rubric dimension,
- describe the observed issue that triggered the deduction,
- state what full-credit behavior would have required.

Keep feedback concise and concrete. If full credit is deserved, briefly say why.
""".strip(),
            model=MODEL,
        ),
    ],
)
