"""Deterministic tests for the non-LLM core: SRS, curriculum, persistence."""

from __future__ import annotations

from german_tutor import curriculum as curr
from german_tutor.persistence import Store
from german_tutor.srs import SrsState, review


def test_srs_successful_recall_grows_interval():
    state = SrsState()
    s1 = review(state, 5)
    assert s1.interval == 1 and s1.reps == 1
    s2 = review(s1, 5)
    assert s2.interval == 6 and s2.reps == 2
    s3 = review(s2, 5)
    assert s3.interval > 6 and s3.reps == 3


def test_srs_failure_resets():
    state = review(review(SrsState(), 5), 5)  # reps=2, interval=6
    failed = review(state, 1)
    assert failed.reps == 0
    assert failed.interval == 1
    assert failed.ease < state.ease


def test_curriculum_loads_all_levels():
    units = curr.all_units()
    ids = {u["id"] for u in units}
    assert {"A1.U1", "A2.U5", "B1.U3", "B2.U4"} <= ids
    assert curr.get_unit("A2.U5")["title"].startswith("Perfekt")


def test_next_unit_skips_completed():
    completed = {"A1.U1", "A1.U2"}
    nxt = curr.next_unit("A1", completed)
    assert nxt["id"] == "A1.U3"


def test_store_progress_and_resume(tmp_path):
    db = tmp_path / "p.db"
    store = Store(db)
    store.get_or_create_learner("alice", name="Alice")
    assert store.get_level("alice") == "A1"

    store.set_level("alice", "A2")
    store.set_pointer("alice", "A2.U5", 3)
    store.update_mastery("alice", "A2.U1", 0.9, "completed")
    store.record_attempt("alice", "A2.U5", "A2.U5.ex1", correct=False, score=0.0)
    store.log_error("alice", "wrong auxiliary", "Ich habe gegangen", "Ich bin gegangen")
    store.close()

    # Reopen: durable state survives across "sessions".
    store2 = Store(db)
    assert store2.get_level("alice") == "A2"
    assert store2.get_pointer("alice") == {"unit_id": "A2.U5", "step": 3}
    assert "A2.U1" in store2.completed_unit_ids("alice")
    assert store2.is_returning("alice")
    assert store2.top_errors("alice")[0]["category"] == "wrong auxiliary"
    store2.close()


def test_vocab_srs_reschedules(tmp_path):
    store = Store(tmp_path / "v.db")
    store.get_or_create_learner("bob")
    store.add_vocab("bob", "der Tisch", "the table")
    due = store.due_vocab("bob")
    assert len(due) == 1
    result = store.review_vocab("bob", due[0]["id"], quality=5)
    assert result["interval"] >= 1
    # After a good review it should no longer be due today.
    assert store.count_due_vocab("bob") == 0
    store.close()
