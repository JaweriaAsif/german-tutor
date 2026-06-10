"""Load and query the static CEFR curriculum.

The curriculum is fixed, versioned data (curriculum/*.yaml). The model generates
the *delivery* of each unit, but the set of units, objectives, grammar points and
vocab themes is stable and authored here.
"""

from __future__ import annotations

import functools
from pathlib import Path

import yaml

LEVELS = ("A1", "A2", "B1", "B2")


def _curriculum_dir() -> Path:
    # repo_root/curriculum, resolved relative to this file (src/german_tutor/..)
    return Path(__file__).resolve().parents[2] / "curriculum"


@functools.lru_cache(maxsize=1)
def load_curriculum() -> dict[str, list[dict]]:
    """Return {level: [unit, ...]} for all levels, in curriculum order."""
    curriculum: dict[str, list[dict]] = {}
    base = _curriculum_dir()
    for level in LEVELS:
        path = base / f"{level.lower()}.yaml"
        if not path.exists():
            curriculum[level] = []
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        curriculum[level] = data.get("units", [])
    return curriculum


def all_units() -> list[dict]:
    """Flat, ordered list of every unit across levels."""
    flat: list[dict] = []
    for level in LEVELS:
        for unit in load_curriculum()[level]:
            flat.append({**unit, "level": level})
    return flat


def get_unit(unit_id: str) -> dict | None:
    for unit in all_units():
        if unit["id"] == unit_id:
            return unit
    return None


def units_for_level(level: str) -> list[dict]:
    return [{**u, "level": level} for u in load_curriculum().get(level, [])]


def next_unit(level: str, completed_unit_ids: set[str]) -> dict | None:
    """First not-yet-completed unit at the given level, else first of next level."""
    order = list(LEVELS)
    if level not in order:
        level = "A1"
    for lvl in order[order.index(level):]:
        for unit in units_for_level(lvl):
            if unit["id"] not in completed_unit_ids:
                return unit
    return None
