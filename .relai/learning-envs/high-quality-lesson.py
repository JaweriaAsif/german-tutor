"""RELAI learning environment for a paced CEFR-aligned grammar lesson."""

import json

from relai import LLMJudgeEvaluator, ModelSpec, PersonaInput, RELAIEnvironment

MODEL = ModelSpec(name="gpt-5.4")
TAGS = ["end-to-end", "paced-level-appropriate-lesson"]


def _ok_json(**extra: object) -> str:
    return json.dumps({"ok": True, **extra})


def mock_set_level(level: str) -> str:
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


environment = RELAIEnvironment(
    schema_version="relai.learning_environment.v1",
    id="high-quality-lesson",
    name="Paced A1 Grammar Lesson",
    description="Tests that the tutor delivers a paced, level-appropriate grammar lesson with checks and one-at-a-time practice.",
    tags=TAGS,
    input=PersonaInput(
        persona=(
            "You are an A1 German learner who wants a careful grammar lesson and answers "
            "briefly when the tutor checks understanding."
        ),
        intent=(
            "Work through the next A1 grammar lesson step by step, let the tutor explain "
            "before moving on, and complete practice only one exercise at a time."
        ),
        seed_message="/lesson Please teach me the next A1 grammar point one step at a time.",
        max_turns=4,
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
        LLMJudgeEvaluator(
            id="lesson-quality",
            description="Judges whether the lesson is paced, level-appropriate, and pedagogically sound.",
            instructions=(
                "Evaluate the full interaction as a paced grammar lesson for an A1 learner. "
                "Assume the intended scenario is a curriculum-aligned grammar lesson at the "
                "learner's CEFR level, and use the transcript rather than only the final turn. "
                "Reward responses that: (1) state a clear learning objective near the start, "
                "(2) explain the grammar point step by step, (3) include multiple concrete German "
                "example sentences with English glosses, (4) keep German at A1 level and clearly "
                "flag any higher-level preview, (5) introduce relevant vocabulary, (6) check the "
                "learner's understanding before advancing, (7) present exercises one at a time "
                "with specific corrective feedback, and (8) avoid inventing grammar rules. "
                "Score down for terse or rushed lessons, for dumping too much content or multiple "
                "exercises at once, for level-inappropriate German, for missing glosses or "
                "understanding checks, or for dubious grammar explanations. If you deduct points, "
                "the feedback must name the failed rubric dimension, cite the observed issue that "
                "triggered the deduction, and state what full-credit behavior would have required."
            ),
            model=MODEL,
        )
    ],
)
