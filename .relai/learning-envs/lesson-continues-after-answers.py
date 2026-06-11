"""RELAI learning environment for short lesson-continuation replies."""

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
    id="lesson-continues-after-answers",
    name="Lesson Continues After Answers",
    description="Checks that after the learner answers a couple of lesson questions in a row, the tutor advances to the next step instead of stopping to ask 'do you want to continue?' or re-listing the menu.",
    tags=TAGS,
    input=PersonaInput(
        persona=(
            "An eager absolute-beginner German learner taking their very first lesson. "
            "Cooperative and a little unsure, but always tries to answer whatever the "
            "tutor asks with a short attempt in German."
        ),
        intent=(
            "Work through the lesson by answering each question the tutor asks, one at a "
            "time, for several questions in a row. Always give a brief German answer (or "
            "a short 'next'/'weiter' to move on); never ask to stop or change activity. "
            "The point is to see whether the tutor keeps presenting the next question or "
            "instead stalls by asking 'do you want to continue?' or re-listing the menu."
        ),
        seed_message="/lesson Start my first lesson and teach me one small step at a time.",
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
            id="lesson-continues-after-answers",
            description="Judges whether, after the learner answers a couple of questions, the tutor keeps moving instead of asking to continue or re-listing the menu.",
            instructions=(
                "Evaluate the full transcript for ONE narrow behavior: as the learner answers "
                "the tutor's questions turn after turn, the tutor must keep the lesson moving "
                "forward. Score only this lesson-momentum behavior; ignore unrelated criteria "
                "like tone, tool choice, or overall lesson quality.\n\n"
                "Full-credit behavior:\n"
                "1. The tutor starts a lesson and asks a first question/exercise.\n"
                "2. After EACH learner answer, the tutor gives brief feedback and then "
                "immediately presents the NEXT question or teaching step in the same lesson.\n"
                "3. Across the whole transcript the tutor NEVER stops to ask a standalone "
                "'do you want to continue?' / 'shall we continue?' / 'ready for the next one?' "
                "type question — it just continues.\n"
                "4. The tutor does NOT re-list the activity menu (lesson / vocab / quiz / "
                "conversation / progress) between questions.\n"
                "5. The tutor ends each turn with the single next thing it needs from the "
                "learner (the next question), not a menu or a yes/no confirmation.\n\n"
                "Deduct points for EACH turn where the tutor asks whether to continue, pauses "
                "for confirmation before the next question, or re-offers the menu after the "
                "learner has clearly been answering. Score lower the more often this happens. "
                "If you deduct, name the offending turn(s), quote the 'continue?'/menu text, and "
                "state that the tutor should have moved straight to the next question instead."
            ),
            model=MODEL,
        )
    ],
)
