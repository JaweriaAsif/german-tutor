"""RELAI environment for a zero-knowledge German lesson response."""

import json

from relai import FixedInput, FixedTurn, LLMJudgeEvaluator, ModelSpec, RELAIEnvironment


MODEL = ModelSpec(name="gpt-5.4")
TAGS = ["end-to-end", "absolute-beginner-lesson-everything-explained"]


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


def mock_add_vocab(lemma: str = "der Name", gloss: str = "name", *args, **kwargs) -> str:
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


environment = RELAIEnvironment(
    schema_version="relai.learning_environment.v1",
    id="detailed-no-german-assumed",
    name="Zero-Knowledge Lesson Detail",
    description="Checks that a first German lesson fully explains everything for a learner with no prior German.",
    tags=TAGS,
    input=FixedInput(
        turns=[
            FixedTurn(
                content=(
                    "/lesson I know no German at all. Please give me a detailed first lesson "
                    "and explain every German word, phrase, and grammar idea in plain English."
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
            id="zero-knowledge-beginner-lesson",
            description="Judges whether the lesson is detailed enough for a learner with zero German knowledge.",
            instructions="""
Evaluate only the assistant's lesson reply in this environment.

This scenario tests a learner with absolutely no prior German knowledge. Score
only the zero-knowledge teaching behavior described below. Do not add unrelated
criteria such as personality, general helpfulness, tool use, or whether the
assistant completed a full multi-step curriculum plan.

Rubric:
1. No assumed German knowledge: the reply must be understandable to someone who
knows no German. If it leaves German words, phrases, or example sentences
untranslated or unexplained, deduct points.
2. Detailed explanation: each new teaching point is broken into small steps
instead of being terse or compressed. A complete beginner should be able to
follow without outside help.
3. Plain-English teaching of concepts: if the reply uses grammar terms or
concepts, it defines them in simple English before relying on them.
4. Worked beginner support: the reply should include multiple simple examples
with English glosses, and when nouns or pronunciation-relevant items appear, it
should clearly explain articles, gender, or pronunciation cues when useful.
5. No hidden prerequisites: the lesson must not rely on the learner already
knowing vocabulary, grammar conventions, or German classroom norms.

Score high only when the reply is genuinely detailed and fully beginner-safe.
Score low if it is terse, skips explanations, assumes prior knowledge, or
contains untranslated German that a new learner would not understand.

If you deduct points, the feedback must name the failed criterion, describe the
observed issue that caused the deduction, and state what full-credit behavior
would have required. Keep the feedback concise, specific, and evidence-based.
""".strip(),
            model=MODEL,
        )
    ],
)
