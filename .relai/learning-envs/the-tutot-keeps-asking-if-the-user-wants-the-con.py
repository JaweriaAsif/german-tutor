"""RELAI learning environment for short lesson-continuation replies."""

import json

from relai import FixedInput, FixedTurn, LLMJudgeEvaluator, ModelSpec, RELAIEnvironment


MODEL = ModelSpec(name="gpt-5.4")
TAGS = ["end-to-end", "short-continuations-advance-lesson"]


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


def mock_add_vocab(lemma: str = "hallo", gloss: str = "hello", *args, **kwargs) -> str:
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


def mock_speak_german(text: str = "hallo", *args, **kwargs) -> str:
    del args, kwargs
    return _ok_json(audio_path="/tmp/fake-audio.aiff", played=False, text=text)


environment = RELAIEnvironment(
    schema_version="relai.learning_environment.v1",
    id="the-tutot-keeps-asking-if-the-user-wants-the-con",
    name="Brief Replies Continue Lessons",
    description="Checks that brief continuation replies keep an active lesson moving instead of resetting to a menu or confirmation prompt.",
    tags=TAGS,
    input=FixedInput(
        turns=[
            FixedTurn(content="/lesson Start my next lesson and teach me one small step at a time."),
            FixedTurn(content="yes"),
            FixedTurn(content="next"),
            FixedTurn(content="continue"),
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
            id="brief-continuation-keeps-momentum",
            description="Judges whether short continuation replies keep the lesson moving instead of resetting the interaction.",
            instructions=(
                "Evaluate the full transcript for one narrow behavior: when a lesson is already "
                "in progress, brief learner continuation replies such as 'yes', 'next', or "
                "'continue' should advance the current lesson rather than stall. Score only this "
                "lesson-momentum behavior; do not add unrelated criteria such as general "
                "helpfulness, tone, tool choice, or overall lesson quality.\n\n"
                "Full-credit behavior:\n"
                "1. The tutor clearly starts a lesson on the first turn.\n"
                "2. After each brief continuation reply, the tutor treats it as a signal to keep "
                "going with the same lesson.\n"
                "3. The follow-up turns show forward motion, such as teaching the next step, "
                "giving the next exercise, or giving feedback tied to the current lesson step.\n"
                "4. The tutor does not reset into concierge behavior, ask a generic 'do you want "
                "to continue?' question, or re-list an activity menu instead of progressing.\n"
                "5. The tutor does not simply repeat the same setup or stall without advancing "
                "the lesson.\n\n"
                "Deduct points when the tutor treats a brief continuation as vague small talk, "
                "asks the learner to choose from options again, asks whether to continue instead "
                "of continuing, or otherwise fails to move the lesson forward. If you deduct "
                "points, the feedback must name the failed criterion, describe the observed issue "
                "that triggered the deduction, and state what full-credit behavior would have "
                "required."
            ),
            model=MODEL,
        )
    ],
)
