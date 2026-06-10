"""SM-2 spaced-repetition scheduling for vocabulary cards.

Pure functions, no I/O, so they are trivially unit-testable and deterministic.
`quality` is the recall grade 0-5 (>=3 counts as a successful recall), per the
classic SuperMemo-2 algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SrsState:
    ease: float = 2.5
    interval: int = 0
    reps: int = 0


def review(state: SrsState, quality: int) -> SrsState:
    """Return the updated SRS state after a review of the given quality (0-5)."""
    quality = max(0, min(5, quality))

    if quality < 3:
        # Failed recall: reset reps and interval, keep (slightly reduced) ease.
        new_ease = max(1.3, state.ease - 0.2)
        return SrsState(ease=new_ease, interval=1, reps=0)

    reps = state.reps + 1
    if reps == 1:
        interval = 1
    elif reps == 2:
        interval = 6
    else:
        interval = round(state.interval * state.ease)

    new_ease = state.ease + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    new_ease = max(1.3, new_ease)
    return SrsState(ease=new_ease, interval=interval, reps=reps)
