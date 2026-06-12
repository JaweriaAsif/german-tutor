"""RELAI learning environment for example-backed explanation of introduced forms."""

import json

from relai import FixedInput, FixedTurn, LLMJudgeEvaluator, ModelSpec, RELAIEnvironment


MODEL = ModelSpec(name="gpt-5.4")
TAGS = ["end-to-end", "new-word-forms-explained-with-examples"]


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


def mock_add_vocab(lemma: str = "heissen", gloss: str = "to be called", *args, **kwargs) -> str:
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
    return _ok_json(audio_path="/tmp/fake-audio.aiff", played=False, text=text)


environment = RELAIEnvironment(
    schema_version="relai.learning_environment.v1",
    id="the-lessons-introduces-new-words-but-then-doesnt",
    name="Example-Backed Word Forms",
    description="Checks that lesson steps use enough examples to explain the specific new forms they introduce.",
    tags=TAGS,
    input=FixedInput(
        turns=[
            FixedTurn(
                content=(
                    "/lesson Teach me the next lesson step by step. When you introduce a new "
                    "word or form, make it clear with examples."
                )
            ),
            FixedTurn(content="yes, continue"),
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
            id="introduced-forms-need-examples",
            description="Judges whether introduced lesson forms are explained with enough examples to make those specific forms clear.",
            instructions=(
                "Evaluate the full lesson transcript for one narrow teaching behavior: whenever "
                "the tutor introduces a new word form, pattern, or inflected form in a lesson "
                "step, it should give enough examples to explain that specific form clearly.\n\n"
                "Use the clarification intent for this environment:\n"
                "1. Judge only the specific forms introduced in that lesson step, not a full "
                "paradigm unless the tutor itself introduces that broader paradigm.\n"
                "2. Treat this as a general lesson-teaching expectation whenever new words are "
                "introduced, not only as a first-beginner-opening rule.\n\n"
                "Full-credit behavior:\n"
                "1. When the tutor introduces a new form or small pattern, it supports that form "
                "with enough concrete examples in context for the learner to see how it works.\n"
                "2. The examples are relevant to the exact form being taught, not just loosely "
                "related vocabulary mentions.\n"
                "3. The tutor explains what to notice in those examples so the learner can connect "
                "the examples to the form.\n"
                "4. Do not require full verb tables, complete article systems, or broad grammar "
                "coverage unless the tutor explicitly chose to introduce them.\n"
                "5. It is acceptable for the lesson to stay small and paced; the issue is whether "
                "the introduced forms are example-backed enough to understand.\n\n"
                "Deduct points if the tutor introduces a new form but gives too few examples to "
                "make that form clear, if the examples do not actually illustrate the introduced "
                "form, or if it names a form without enough explanation for the learner to see the "
                "pattern. Do not deduct merely because the reply does not cover a wider paradigm "
                "that was never introduced. If you deduct points, name the failed criterion, "
                "describe the observed issue that triggered the deduction, and state what "
                "full-credit behavior would have required."
            ),
            model=MODEL,
        )
    ],
)
