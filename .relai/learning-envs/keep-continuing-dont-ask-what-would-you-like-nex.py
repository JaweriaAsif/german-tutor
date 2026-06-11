"""RELAI learning environment for lesson momentum after repeated learner answers."""

import json

from relai import LLMJudgeEvaluator, ModelSpec, PersonaInput, RELAIEnvironment


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
    category: str = "spelling",
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


def mock_add_vocab(lemma: str = "aufstehen", gloss: str = "to get up", *args, **kwargs) -> str:
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


def mock_speak_german(text: str = "Guten Morgen", *args, **kwargs) -> str:
    del args, kwargs
    return _ok_json(audio_path="/tmp/fake-audio.aiff", played=False, text=text)


environment = RELAIEnvironment(
    schema_version="relai.learning_environment.v1",
    id="keep-continuing-dont-ask-what-would-you-like-nex",
    name="Keep Lesson Moving",
    description="Checks that when a learner keeps answering lesson prompts, the tutor continues the lesson instead of asking what to do next.",
    tags=TAGS,
    input=PersonaInput(
        persona=(
            "You are an engaged beginner German learner following the tutor's lesson closely. "
            "You answer each prompt briefly and keep going without asking to switch activities."
        ),
        intent=(
            "Stay in the current lesson for several turns. After each exercise or correction, "
            "reply with a short answer in German when possible, or a brief continuation cue "
            "like 'weiter' or 'next' when the tutor is clearly prompting you to continue. "
            "Do not ask for a menu, do not ask to stop, and do not request a different activity."
        ),
        seed_message="/lesson Teach me my next lesson step by step and keep going as I answer.",
        max_turns=6,
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
        "speak_german": mock_speak_german,
    },
    evaluators=[
        LLMJudgeEvaluator(
            id="lesson-momentum-after-answers",
            description="Judges whether the tutor keeps advancing the lesson after the learner continues answering.",
            instructions=(
                "Evaluate one narrow behavior across the full transcript: once the learner is "
                "actively participating in a lesson and keeps answering, the tutor should keep "
                "the lesson moving instead of stopping to ask what the learner wants next.\n\n"
                "Full-credit behavior:\n"
                "1. The tutor starts or resumes a coherent lesson.\n"
                "2. After the learner answers or gives a brief continuation cue, the tutor "
                "responds with the next lesson step, next exercise, or feedback plus the next "
                "prompt in the same lesson.\n"
                "3. The tutor does not stall with concierge behavior such as 'What would you "
                "like next?', a generic activity menu, or a confirmation question about whether "
                "to continue while the learner is still clearly engaged in the lesson.\n"
                "4. If the tutor intentionally presents options, that should happen only after "
                "it has clearly completed a coherent lesson segment; do not deduct for that if "
                "the transcript shows a genuine, intentional wrap-up rather than premature "
                "stalling.\n\n"
                "Deduct points when the tutor breaks lesson momentum by re-routing to menu "
                "selection, asking the learner to choose what to do next, or otherwise failing "
                "to advance the ongoing lesson after clear continued participation. If you "
                "deduct, identify the failed criterion, describe the observed issue that caused "
                "the deduction, and state what full-credit behavior would have required."
            ),
            model=MODEL,
        )
    ],
)
