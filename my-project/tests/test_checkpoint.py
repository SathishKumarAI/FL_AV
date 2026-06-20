"""Tests for the checkpoint cadence predicate (should_checkpoint)."""
from my_project.task import should_checkpoint


def test_saves_every_round_when_save_every_1():
    assert all(should_checkpoint(r, 1, 3) for r in range(1, 4))


def test_respects_save_every_cadence():
    # save_every=2 over 5 rounds: rounds 2 and 4 by cadence, plus final round 5.
    got = [r for r in range(1, 6) if should_checkpoint(r, 2, 5)]
    assert got == [2, 4, 5]


def test_always_saves_final_round_off_cadence():
    # save_every=10 but only 3 rounds → never on cadence, still saves round 3.
    assert should_checkpoint(3, 10, 3) is True
    assert should_checkpoint(1, 10, 3) is False


def test_save_every_zero_is_treated_as_one():
    assert should_checkpoint(1, 0, 3) is True
