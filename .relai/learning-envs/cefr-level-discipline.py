"""RELAI environment for A1-level handling of advanced grammar requests."""

import json

from relai import FixedInput, FixedTurn, LLMJudgeEvaluator, ModelSpec, RELAIEnvironment


MODEL = ModelSpec(name="gpt-5.4")
TAGS = ["end-to-end", "a1-advanced-grammar-deferred-to-later-units"]


def _ok_json(**extra: object) -> str:
    return json.dumps({"ok": True, **extra})


def mock_set_level(level: str = "A1", *args, **kwargs) -> str:
    del args, kwargs
    return _ok_json(level=str(level).upper())


def mock_save_lesson_pointer(unit_id: str = "A1.U1", step: int = 0, *args, **kwargs) -> str:
    del unit_id, step, args, kwargs
    return _ok_json()


def mock_cache_lesson(unit_id: str = "A1.U1", lesson_json: str = "{}", *args, **kwargs) -> str:
    del unit_id, lesson_json, args, kwargs
    return _ok_json()


def mock_record_attempt(
    unit_id: str = "A1.U1",
    exercise_id: str = "ex-1",
    correct: bool = True,
    score: float = 1.0,
    *args,
    **kwargs,
) -> str:
    del unit_id, exercise_id, correct, score, args, kwargs
    return _ok_json()


def mock_log_error(
    category: str = "grammar",
    example: str = "",
    correction: str = "",
    *args,
    **kwargs,
) -> str:
    del category, example, correction, args, kwargs
    return _ok_json()


def mock_update_mastery(
    unit_id: str = "A1.U1",
    mastery: float = 0.5,
    status: str = "in_progress",
    *args,
    **kwargs,
) -> str:
    del unit_id, mastery, status, args, kwargs
    return _ok_json()


def mock_add_vocab(lemma: str = "der Satz", gloss: str = "sentence", *args, **kwargs) -> str:
    del lemma, gloss, args, kwargs
    return _ok_json()


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
    del summary, args, kwargs
    return _ok_json()


def mock_speak_german(text: str = "Hallo", *args, **kwargs) -> str:
    del args, kwargs
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
    id="cefr-level-discipline",
    name="A1 Advanced Grammar Deferral",
    description="Checks that the tutor defers advanced grammar requests from an A1 learner and redirects to A1-safe content.",
    tags=TAGS,
    input=FixedInput(
        turns=[
            FixedTurn(
                content=(
                    "I'm an A1 German learner. Can you teach me Konjunktiv II or complex "
                    "subordinate clauses?"
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
            id="a1-level-discipline",
            description="Judges whether the tutor keeps an advanced grammar request at an A1-appropriate level.",
            instructions="""
Evaluate only the assistant response to this A1 learner's advanced grammar request.

This environment is about level discipline, not general lesson quality. Do not add
unrelated criteria such as personality, exhaustive correctness across all German
grammar, tool usage, or whether the assistant completes a full lesson.

Score using only these criteria:
1. Explicit level control: the reply recognizes that Konjunktiv II or complex
subordinate clauses are above A1 and clearly defers substantive teaching of them
to later study, later units, or a higher level.
2. No advanced grammar dump: the reply must not launch into a detailed B1/B2-style
explanation, dense rule list, or several advanced example sentences that expect
the learner to absorb the advanced topic now.
3. A1-safe redirect: the reply should offer an immediate beginner-safe next step,
preview, or prerequisite topic the learner can study now instead of leaving them
with only a refusal.
4. Honest framing of any preview: if the reply briefly mentions the advanced topic,
it should clearly frame that mention as a short preview only, not as the learner's
current lesson content.

Do not require exact wording. A brief acknowledgement of the advanced topic is fine
if the tutor still defers it and redirects appropriately.

If you deduct points, the feedback must name the failed criterion, describe the
observed issue that triggered the deduction, and state what full-credit behavior
would have required. Keep the feedback concise, specific, and evidence-based.
""".strip(),
            model=MODEL,
        )
    ],
)
