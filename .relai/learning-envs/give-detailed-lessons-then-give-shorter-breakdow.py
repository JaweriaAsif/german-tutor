"""RELAI learning environment for lesson pacing from detailed teaching to shorter follow-up."""

import json

from relai import FixedInput, FixedTurn, LLMJudgeEvaluator, ModelSpec, RELAIEnvironment


MODEL = ModelSpec(name="gpt-5.4")
TAGS = ["end-to-end", "detailed-first-shorter-follow-up"]


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
    id="give-detailed-lessons-then-give-shorter-breakdow",
    name="Detailed Then Shorter Lessons",
    description="Checks that the tutor teaches a lesson point in detail first and only gives a shorter simpler restatement on a later follow-up turn.",
    tags=TAGS,
    input=FixedInput(
        turns=[
            FixedTurn(
                content=(
                    "/lesson Teach me my next lesson in detail first. Give me the full explanation "
                    "before simplifying anything."
                )
            ),
            FixedTurn(
                content=(
                    "Now give me a shorter, simpler breakdown of that same teaching point."
                )
            ),
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
            id="detailed-then-shorter-follow-up",
            description="Judges whether the tutor teaches in detail first and then gives a shorter simpler restatement of the same point only after the learner asks for it.",
            instructions=(
                "Evaluate one narrow behavior across the full transcript: the tutor should give a "
                "detailed lesson explanation first, and only on a later learner follow-up turn "
                "should it provide a shorter, simpler restatement of that same teaching point.\n\n"
                "Score only this sequencing-and-restatement behavior. Do not add unrelated criteria "
                "such as general helpfulness, tone, tool choice, or whether the tutor picked a "
                "specific grammar topic.\n\n"
                "Full-credit behavior:\n"
                "1. The first tutor lesson reply is genuinely detailed: it explains the current "
                "lesson point with enough substance, examples, or breakdown that it reads like the "
                "main teaching pass rather than a terse summary.\n"
                "2. The tutor does not try to do both layers at once in the opening reply. The "
                "shorter breakdown belongs in a later follow-up turn after the learner responds.\n"
                "3. After the learner asks for a shorter version, the tutor gives a visibly shorter "
                "and simpler restatement of the SAME teaching point from the prior lesson reply.\n"
                "4. The follow-up does not jump to a new topic, new exercise, or generic activity "
                "menu instead of restating the same point.\n"
                "5. This is a general lesson-pacing expectation, not a beginner-only rule. Do not "
                "require absolute-beginner framing unless the tutor itself chose it.\n\n"
                "Deduct points if the opening reply is already short, if it tries to combine the "
                "detailed lesson and shorter recap in the same turn, if the later reply is not "
                "meaningfully shorter or simpler, or if the later reply changes the teaching point. "
                "If you deduct points, name the failed criterion, describe the observed issue that "
                "triggered the deduction, and state what full-credit behavior would have required."
            ),
            model=MODEL,
        )
    ],
)
