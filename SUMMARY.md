# German Tutor — Summary

## The agent

A multi-agent **CEFR German tutor (A1–B2)** built on **LangGraph**. A router
classifies each turn and dispatches to one specialist ReAct agent:

| Agent | Role |
|-------|------|
| placement | assess + set CEFR level |
| lesson | paced, resumable lesson for a unit |
| grammar | explain a grammar point |
| vocab | teach/review words (spaced repetition) |
| exercise | generate + grade exercises one at a time |
| conversation | role-play German dialogue |
| progress | update mastery, recommend next step |
| concierge | greet / menu / quick replies |
| *offtopic* | guardrail — refuses non-German requests |

**Persistence (two layers):**
- *Conversation/graph state* → LangGraph **SQLite checkpointer**, keyed by learner id → "resume where you left off."
- *Durable progress* → SQLite `Store`: level, per-unit mastery, lesson pointer, SM-2 vocab deck, error log.

**Audio:** macOS `say` TTS (German voice) for pronunciation; CLI renders 🔊 markers + `/play N`.

## Testing with RELAI

Behavior is validated/optimized with **RELAI learning environments** (persona
learners + LLM-judge evaluators). Three lesson-quality envs (tag `end-to-end`):

- `beginner-lesson-detail` — detailed, paced first lesson
- `detailed-no-german-assumed` — assumes zero prior German; everything explained
- `high-quality-lesson` — clear objective, examples, level-appropriate, checks understanding

Run them:
```sh
relai simulate --tags end-to-end --env-file .relai/simulator.env          # score current behavior
relai optimize --tags end-to-end --env-file .relai/simulator.env --total-rollouts 30   # improve
```
RELAI Python SDK **0.1.14** (backend-shipped), CLI **0.1.18**.

## How the optimizer improved the agent

A 30-rollout `relai optimize` run committed one accepted change
(`[RELAI optimized] Improve absolute-beginner lesson openings`):

- **`graph.py` (lesson prompt):** for a first lesson, teach for **zero prior
  knowledge** — gloss every German word inline in English, no unexplained
  "preview" German, explicit pronunciation hints and tasks.
- **`tools_lc.py`:** `get_learner_state` now returns an **`is_absolute_beginner`**
  flag (`pointer is None and not progress`) so the lesson agent detects a true
  first lesson and switches into the detailed beginner mode.

Net effect: beginner lessons are more thorough and never assume the learner
already knows German — directly raising the lesson-quality eval scores.

## Mocked data for testing

RELAI decides per tool whether to **mock** (a deterministic stub) or run **live**,
recorded in `.relai/mock-manifest.json`:

| Policy | Tools | Why |
|--------|-------|-----|
| **mock** (state-mutating) | `set_level`, `save_lesson_pointer`, `cache_lesson`, `record_attempt`, `log_error`, `update_mastery`, `add_vocab`, `review_vocab`, `log_session_summary` | writes to SQLite — mock so a simulation doesn't corrupt real learner state |
| **do_not_mock** (read-only) | `get_learner_state`, `get_next_unit`, `get_unit_details`, `get_cached_lesson`, `get_due_vocab` | safe local reads; run live so the agent sees real curriculum/state |

In a learning environment, supply deterministic outputs for the mocked tools via
the `mocks={}` field, e.g.:
```python
mocks={
    "get_next_unit": {"id": "A1.U1", "title": "Begrüßungen", "level": "A1", ...},
    "update_mastery": {"ok": True},
}
```
The LLM model calls in the agent run live against OpenAI (no mock); only the
project's own tools are mocked.

> Note: `speak_german` (TTS) was added after the manifest was generated, so it is
> not yet listed. Re-run `relai init` (or add a `mock` entry) so simulations don't
> invoke `say`/audio during rollouts.
