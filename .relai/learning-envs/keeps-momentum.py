"""RELAI learning environment for lesson momentum after continuation replies."""

import json

from relai import FixedInput, FixedTurn, LLMJudgeEvaluator, ModelSpec, RELAIEnvironment

MODEL = ModelSpec(name="gpt-5.4")
TAGS = ["end-to-end", "lesson-continuations-keep-momentum"]


def _ok_json(**extra: object) -> str:
    return json.dumps({"ok": True, **extra})


def mock_set_level(level: str = "A1") -> str:
    return _ok_json(level=str(level).upper())


def mock_save_lesson_pointer(unit_id: str, step: int) -> str:
    return _ok_json(unit_id=unit_id, step=step)


def mock_cache_lesson(unit_id: str, lesson_json: str) -> str:
    return _ok_json(unit_id=unit_id, cached=bool(lesson_json))


def mock_record_attempt(unit_id: str, exercise_id: str, correct: bool, score: float) -> str:
    return _ok_json(unit_id=unit_id, exercise_id=exercise_id, correct=bool(correct), score=score)


def mock_log_error(category: str, example: str, correction: str) -> str:
    return _ok_json(category=category, example=example, correction=correction)


def mock_update_mastery(unit_id: str, mastery: float, status: str) -> str:
    return _ok_json(unit_id=unit_id, mastery=mastery, status=status)


def mock_add_vocab(lemma: str, gloss: str) -> str:
    return _ok_json(lemma=lemma, gloss=gloss)


def mock_review_vocab(card_id: int, quality: int) -> str:
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


def mock_log_session_summary(summary: str) -> str:
    return _ok_json(summary=summary)


def mock_speak_german(text: str) -> str:
    return json.dumps(
        {
            "ok": True,
            "audio_path": "/tmp/mock-audio.wav",
            "played": False,
            "text": text,
        }
    )


environment = RELAIEnvironment(
    schema_version="relai.learning_environment.v1",
    id="keeps-momentum",
    name="Lesson Momentum",
    description="Checks that brief continuation replies keep an in-progress lesson moving instead of restarting the menu.",
    tags=TAGS,
    input=FixedInput(
        turns=[
            FixedTurn(content="/lesson"),
            FixedTurn(content="yes"),
            FixedTurn(content="next"),
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
            id="keeps-momentum",
            description="Judges whether short continuation replies advance the lesson without stalling or re-listing options.",
            instructions=(
                "Evaluate the full interaction as an in-progress lesson momentum scenario. "
                "The learner starts a lesson, then replies with short continuation messages "
                "like 'yes' and 'next'. Score only the keep-momentum behavior required here. "
                "Full credit requires all of the following: (1) after each short continuation, "
                "the tutor moves directly into the next lesson step or the next single exercise "
                "instead of stalling; (2) it does not ask whether the learner wants to continue "
                "or otherwise pause for redundant confirmation; (3) it does not re-list the "
                "activity menu such as lesson, vocab, quiz, conversation, or progress unless "
                "the lesson is clearly finished or the learner explicitly asks what they can do; "
                "(4) each turn ends with the one concrete thing the tutor needs next, not a menu "
                "of options. Deduct points if the tutor restarts the menu, asks 'do you want to "
                "continue?' after a clear continuation, or otherwise fails to advance the lesson. "
                "If you deduct points, the feedback must name the failed criterion, describe the "
                "observed issue that triggered the deduction, and state what full-credit behavior "
                "would have required."
            ),
            model=MODEL,
        )
    ],
)
