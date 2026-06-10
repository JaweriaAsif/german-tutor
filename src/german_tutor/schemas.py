"""Structured (Pydantic) output models shared across the tutor agents.

Typed outputs make grading deterministic and make RELAI evaluators precise:
an exercise always carries its own expected answer, and a grading result
always carries an error breakdown.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


CEFRLevel = Literal["A1", "A2", "B1", "B2"]

ExerciseType = Literal[
    "cloze",          # fill-in-the-blank
    "conjugation",    # conjugate a verb
    "translation",    # EN -> DE or DE -> EN
    "reorder",        # put words in correct order
    "mcq",            # multiple choice
    "free_writing",   # open response (LLM-graded)
]


class Exercise(BaseModel):
    """A single exercise item with everything needed to grade it."""

    id: str = Field(description="Stable id, e.g. 'A2.U5.ex1'.")
    type: ExerciseType
    cefr_level: CEFRLevel
    prompt: str = Field(description="What the learner sees.")
    expected_answer: str = Field(description="Canonical correct answer.")
    acceptable_variants: list[str] = Field(
        default_factory=list,
        description="Other answers that should count as correct (case/space-insensitive).",
    )
    options: list[str] = Field(
        default_factory=list, description="Choices for mcq exercises."
    )
    hint: str | None = None
    grammar_point: str | None = None


class ExerciseSet(BaseModel):
    """A small batch of exercises produced for a lesson step or quiz."""

    unit_id: str
    cefr_level: CEFRLevel
    exercises: list[Exercise] = Field(min_length=1)


class GradedError(BaseModel):
    category: str = Field(description="e.g. 'wrong auxiliary', 'case error', 'word order'.")
    span: str = Field(description="The incorrect fragment from the learner's answer.")
    correction: str
    explanation_en: str


class GradingResult(BaseModel):
    """Result of grading one learner answer."""

    exercise_id: str
    is_correct: bool
    score: float = Field(ge=0.0, le=1.0)
    errors: list[GradedError] = Field(default_factory=list)
    feedback_en: str = Field(description="Encouraging, specific feedback in English.")
    feedback_de: str | None = Field(
        default=None, description="Optional level-appropriate feedback in German."
    )
    next_step_hint: str = Field(description="What to practice or do next.")


class LessonStep(BaseModel):
    kind: Literal["teach", "checkpoint", "exercise", "vocab", "wrapup"]
    title: str
    content: str = Field(description="Teaching prose or instruction for this step.")


class LessonPlan(BaseModel):
    """A paced lesson for one curriculum unit. Cached so resume is identical."""

    unit_id: str
    cefr_level: CEFRLevel
    title: str
    objectives: list[str]
    grammar_points: list[str]
    vocab_theme: str
    steps: list[LessonStep] = Field(min_length=1)


class VocabCard(BaseModel):
    lemma: str = Field(description="The German word, with article if a noun, e.g. 'der Tisch'.")
    gloss: str = Field(description="English meaning.")
    example_de: str | None = None
    cefr_level: CEFRLevel


class ProgressUpdate(BaseModel):
    """Returned by the progress tracker after a lesson or quiz."""

    unit_id: str
    mastery: float = Field(ge=0.0, le=1.0)
    recommendation: Literal["advance", "review", "repeat"]
    rationale: str
