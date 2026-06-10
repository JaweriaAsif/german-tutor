"""RELAI environment for judging the first beginner lesson response."""

import json

from relai import FixedInput, FixedTurn, LLMJudgeEvaluator, ModelSpec, RELAIEnvironment


TAGS = ["end-to-end", "first-beginner-lesson-detailed-and-accurate"]


def mock_set_level(level="A1", *args, **kwargs):
    """Return a valid set_level tool payload without mutating learner state."""
    del args, kwargs
    normalized = str(level).upper()
    return json.dumps({"ok": True, "level": normalized})


def mock_save_lesson_pointer(*args, **kwargs):
    """Return a valid success payload for lesson pointer writes."""
    del args, kwargs
    return json.dumps({"ok": True})


def mock_cache_lesson(*args, **kwargs):
    """Return a valid success payload for lesson cache writes."""
    del args, kwargs
    return json.dumps({"ok": True})


def mock_record_attempt(*args, **kwargs):
    """Return a valid success payload for attempt logging."""
    del args, kwargs
    return json.dumps({"ok": True})


def mock_log_error(*args, **kwargs):
    """Return a valid success payload for error logging."""
    del args, kwargs
    return json.dumps({"ok": True})


def mock_update_mastery(*args, **kwargs):
    """Return a valid success payload for mastery updates."""
    del args, kwargs
    return json.dumps({"ok": True})


def mock_add_vocab(*args, **kwargs):
    """Return a valid success payload for vocab additions."""
    del args, kwargs
    return json.dumps({"ok": True})


def mock_review_vocab(*args, **kwargs):
    """Return a valid success payload for vocab review writes."""
    del args, kwargs
    return json.dumps({"ok": True})


def mock_log_session_summary(*args, **kwargs):
    """Return a valid success payload for session summary writes."""
    del args, kwargs
    return json.dumps({"ok": True})


environment = RELAIEnvironment(
    schema_version="relai.learning_environment.v1",
    id="beginner-lesson-detail",
    name="Detailed Beginner Lesson",
    description="Checks that the tutor opens a first A1 lesson with a slow, thorough, accurate beginner-friendly step.",
    tags=TAGS,
    input=FixedInput(
        turns=[
            FixedTurn(
                content=(
                    "Start my first German lesson. I'm a complete beginner with no prior "
                    "German, so please teach one small step at a time and explain everything "
                    "clearly in English."
                )
            )
        ]
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
        LLMJudgeEvaluator(
            id="beginner-lesson-detail",
            description="Judges whether the first lesson reply is detailed, slow, and accurate for an absolute beginner.",
            instructions="""
Evaluate only the assistant's first lesson reply in this environment.

This is a first-step lesson scenario, not a whole-unit lesson dump. Do not
penalize the tutor for teaching only one small concept in the first turn if the
step itself is thorough.

Score based only on these criteria:
1. Beginner-friendly detail and pacing: the reply teaches slowly, assumes no
prior German knowledge, and is not terse or rushed.
2. Concrete examples with English support: when the reply introduces German, it
uses multiple simple examples where appropriate and provides English glosses or
translations so a beginner can follow.
3. New-language explanation: it explains each new word or phrase it introduces
well enough for a complete beginner; pronunciation hints are a plus when useful
but are not required in every answer.
4. Check-before-moving-on behavior: it includes a simple comprehension check,
confirmation prompt, or beginner-friendly question before proceeding.
5. Accuracy and level control: the German stays at A1 level and the explanation
does not invent grammar rules or make materially false grammar claims.

Do not add unrelated criteria such as overall helpfulness, personality, tool
usage, or whether the tutor completed an entire lesson plan.

If you deduct points, the feedback must name the failed criterion, describe the
observed issue that caused the deduction, and state what full-credit behavior
would have required. Keep the feedback concise and specific.
""".strip(),
            model=ModelSpec(name="gpt-5.4"),
        )
    ],
)
