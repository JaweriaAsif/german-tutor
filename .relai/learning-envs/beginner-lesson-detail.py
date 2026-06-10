"""RELAI learning environment for detailed beginner German lessons."""

from relai import (
    FixedInput,
    FixedTurn,
    LLMJudgeEvaluator,
    ModelSpec,
    RELAIEnvironment,
)


TAGS = []


def mock_get_learner_state(*args, **kwargs):
    """Return a deterministic absolute-beginner learner profile."""
    return {
        "learner_id": "beginner-learner",
        "level": "A1",
        "lesson_pointer": None,
        "progress_summary": "Brand-new learner with no completed German lessons.",
        "due_vocab_count": 0,
        "due_vocab": [],
        "weak_spots": [],
        "last_session": None,
    }


def mock_set_level(*args, **kwargs):
    """Acknowledge level updates without touching persistent state."""
    return {"status": "ok"}


def mock_get_next_unit(*args, **kwargs):
    """Return the first beginner unit for a new learner."""
    return {
        "unit_id": "A1.U1",
        "title": "Greetings and introductions",
        "done": False,
    }


def mock_save_lesson_pointer(*args, **kwargs):
    """Acknowledge lesson-pointer writes without persistence."""
    return {"status": "ok"}


def mock_cache_lesson(*args, **kwargs):
    """Acknowledge lesson caching without persistence."""
    return {"status": "ok"}


def mock_get_cached_lesson(*args, **kwargs):
    """Return no cached lesson for a new learner."""
    return None


def mock_record_attempt(*args, **kwargs):
    """Acknowledge exercise-attempt writes without persistence."""
    return {"status": "ok"}


def mock_log_error(*args, **kwargs):
    """Acknowledge error logging without persistence."""
    return {"status": "ok"}


def mock_update_mastery(*args, **kwargs):
    """Acknowledge mastery updates without persistence."""
    return {"status": "ok"}


def mock_add_vocab(*args, **kwargs):
    """Acknowledge vocab-card creation without persistence."""
    return {"status": "ok"}


def mock_get_due_vocab(*args, **kwargs):
    """Return an empty review queue for a new learner."""
    return []


def mock_review_vocab(*args, **kwargs):
    """Return a plausible spaced-repetition update payload."""
    return {
        "status": "ok",
        "interval": 1,
        "reps": 1,
        "ease": 2.5,
        "due_date": "2026-06-11",
    }


def mock_log_session_summary(*args, **kwargs):
    """Acknowledge session-summary writes without persistence."""
    return {"status": "ok"}


BEGINNER_LESSON_JUDGE = LLMJudgeEvaluator(
    id="beginner-lesson-depth",
    description="Judges whether the tutor gives a slow, detailed, accurate A1 lesson for a complete beginner.",
    instructions=(
        "Score the assistant's final tutoring reply for this scenario only. The learner is an absolute "
        "beginner asking for a German lesson. Judge only these criteria:\n"
        "1. The lesson is detailed and beginner-friendly rather than terse or rushed.\n"
        "2. Explanations are step by step, assume no prior German knowledge, and keep German at A1 level.\n"
        "3. Grammar explanations are accurate and do not invent rules.\n"
        "4. The reply includes multiple concrete German examples, and each example has an English gloss or "
        "clear English translation.\n"
        "5. New words introduced in the lesson are translated and explained.\n"
        "6. The reply gives simple pronunciation hints where useful.\n"
        "7. Before closing or moving on, the tutor explicitly checks the learner's understanding.\n"
        "If you deduct points, the feedback must name the failed criterion, cite the observed issue, and say "
        "what full-credit behavior required. Keep feedback concise and specific."
    ),
    model=ModelSpec(name="gpt-5.4"),
)


environment = RELAIEnvironment(
    schema_version="relai.learning_environment.v1",
    id="beginner-lesson-detail",
    name="Detailed Beginner Lesson",
    description="Checks that the tutor gives a slow, detailed A1 German lesson for an absolute beginner and checks understanding before moving on.",
    tags=TAGS,
    input=FixedInput(
        turns=[
            FixedTurn(
                content=(
                    "I'm a complete beginner in German. Please teach me a detailed beginner lesson, explain "
                    "everything slowly, and assume I know nothing yet."
                )
            )
        ]
    ),
    mocks={
        "get_learner_state": mock_get_learner_state,
        "set_level": mock_set_level,
        "get_next_unit": mock_get_next_unit,
        "save_lesson_pointer": mock_save_lesson_pointer,
        "cache_lesson": mock_cache_lesson,
        "get_cached_lesson": mock_get_cached_lesson,
        "record_attempt": mock_record_attempt,
        "log_error": mock_log_error,
        "update_mastery": mock_update_mastery,
        "add_vocab": mock_add_vocab,
        "get_due_vocab": mock_get_due_vocab,
        "review_vocab": mock_review_vocab,
        "log_session_summary": mock_log_session_summary,
    },
    evaluators=[BEGINNER_LESSON_JUDGE],
)
