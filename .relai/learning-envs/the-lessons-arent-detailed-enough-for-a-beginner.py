"""RELAI learning environment for beginner lesson detail."""

import json

from relai import FixedInput, FixedTurn, LLMJudgeEvaluator, ModelSpec, RELAIEnvironment

MODEL = ModelSpec(name="gpt-5.4")
TAGS = ["end-to-end", "first-beginner-lesson-detailed-and-accurate"]


def _ok_json(**extra: object) -> str:
    return json.dumps({"ok": True, **extra})


def mock_set_level(level: str = "A1", *args, **kwargs) -> str:
    del args, kwargs
    return _ok_json(level=str(level).upper())


def mock_save_lesson_pointer(unit_id: str = "A1.U1", step: int = 0, *args, **kwargs) -> str:
    del args, kwargs
    return _ok_json(unit_id=unit_id, step=step)


def mock_cache_lesson(unit_id: str = "A1.U1", lesson_json: str = "{}", *args, **kwargs) -> str:
    del args, kwargs
    return _ok_json(unit_id=unit_id, cached=bool(lesson_json))


def mock_record_attempt(
    unit_id: str = "A1.U1",
    exercise_id: str = "ex-1",
    correct: bool = True,
    score: float = 1.0,
    *args,
    **kwargs,
) -> str:
    del args, kwargs
    return _ok_json(unit_id=unit_id, exercise_id=exercise_id, correct=bool(correct), score=score)


def mock_log_error(
    category: str = "grammar",
    example: str = "",
    correction: str = "",
    *args,
    **kwargs,
) -> str:
    del args, kwargs
    return _ok_json(category=category, example=example, correction=correction)


def mock_update_mastery(
    unit_id: str = "A1.U1",
    mastery: float = 0.5,
    status: str = "in_progress",
    *args,
    **kwargs,
) -> str:
    del args, kwargs
    return _ok_json(unit_id=unit_id, mastery=mastery, status=status)


def mock_add_vocab(lemma: str = "ich", gloss: str = "I", *args, **kwargs) -> str:
    del args, kwargs
    return _ok_json(lemma=lemma, gloss=gloss)


def mock_review_vocab(card_id: int = 1, quality: int = 4, *args, **kwargs) -> str:
    del args, kwargs
    return json.dumps(
        {
            "card_id": card_id,
            "quality": quality,
            "ease": 2.5,
            "interval": 1,
            "reps": 1,
            "due_date": "2099-01-01",
        }
    )


def mock_log_session_summary(summary: str = "", *args, **kwargs) -> str:
    del args, kwargs
    return _ok_json(summary=summary)


def mock_speak_german(text: str = "Hallo", *args, **kwargs) -> str:
    del args, kwargs
    return json.dumps(
        {
            "ok": True,
            "audio_path": "/tmp/relai-fake-audio.aiff",
            "played": False,
            "text": text,
        }
    )


environment = RELAIEnvironment(
    schema_version="relai.learning_environment.v1",
    id="the-lessons-arent-detailed-enough-for-a-beginner",
    name="Detailed First Beginner Lesson",
    description="Checks that the tutor starts a first German lesson with a slow, detailed, beginner-friendly step.",
    tags=TAGS,
    input=FixedInput(
        turns=[
            FixedTurn(
                content=(
                    "Start my first German lesson. I am a complete beginner with no prior "
                    "German, so teach one small step at a time and explain it clearly in English."
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
        "speak_german": mock_speak_german,
    },
    evaluators=[
        LLMJudgeEvaluator(
            id="beginner-lesson-detail",
            description="Judges whether the first lesson reply is detailed and understandable for a complete beginner.",
            instructions="""
Evaluate only the assistant's first lesson reply in this environment.

This is a first-step beginner lesson scenario. Do not penalize the tutor for
teaching only one small concept in the first turn if that step itself is
thorough.

Score only these criteria:
1. Beginner-friendly detail and pacing: the reply teaches slowly, assumes no
prior German knowledge, and is not terse or rushed.
2. Concrete examples with English support: the reply includes simple German
examples with English glosses or translations so a total beginner can follow.
3. Clear explanation of new language: when the reply introduces German words,
phrases, or a small grammar pattern, it explains what they mean in plain
English instead of assuming prior knowledge.
4. Check-before-moving-on behavior: it includes a simple comprehension check,
confirmation prompt, or one explicit beginner task before advancing.
5. Accuracy and level control: the German stays at beginner level and the reply
does not make materially false grammar claims.

Do not add unrelated criteria such as personality, tool usage, or whether the
assistant completed a whole lesson plan.

If you deduct points, the feedback must name the failed criterion, describe the
observed issue that caused the deduction, and state what full-credit behavior
would have required. Keep the feedback concise and specific.
""".strip(),
            model=MODEL,
        )
    ],
)
