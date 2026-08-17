"""One learning-rate anneal across the run, not one per round.

Ultralytics decays lr0 to `lr0 * lrf` inside a call to train(), and every round calls
train() fresh. Six rounds was six independent anneals from full LR, so the fleet
replayed the first slice of a schedule six times and never reached the low-LR
consolidation phase that makes the last epochs of a centralised run count.
"""
import pytest

from my_project.client_app import LR0, LRF, round_lr


def test_each_round_starts_where_the_previous_one_ended():
    """That is what makes it one schedule rather than six. If the ends and starts do
    not meet, the LR jumps between rounds and the anneal is decorative."""
    rounds = 6
    for r in range(1, rounds):
        lr0, lrf = round_lr(r, rounds)
        next_lr0, _ = round_lr(r + 1, rounds)
        assert lr0 * lrf == pytest.approx(next_lr0, rel=1e-9), f"gap after round {r}"


def test_the_run_spans_the_whole_schedule_and_never_restarts_it():
    """Round 1 at full lr0, as a centralised run starts; the last round near lr0*lrf,
    which is where a centralised run finishes. Before this, every round started at
    0.01 -- the fleet's sixth round trained as aggressively as its first."""
    rounds = 6
    first_lr0, _ = round_lr(1, rounds)
    last_lr0, last_lrf = round_lr(rounds, rounds)

    assert first_lr0 == pytest.approx(LR0)
    assert last_lr0 * last_lrf == pytest.approx(LR0 * LRF, rel=1e-9)

    starts = [round_lr(r, rounds)[0] for r in range(1, rounds + 1)]
    assert starts == sorted(starts, reverse=True), "the LR must fall across the run"
    assert len(set(starts)) == rounds, "a repeated lr0 is a restarted schedule"


def test_a_single_round_run_is_a_whole_schedule_by_itself():
    """The demo profile and the CI smoke both run one or two rounds. A schedule that
    only makes sense at six would leave those training at a fraction of lr0 for no
    reason anyone could see in the config."""
    lr0, lrf = round_lr(1, 1)
    assert lr0 == pytest.approx(LR0)
    assert lr0 * lrf == pytest.approx(LR0 * LRF, rel=1e-9)


def test_a_round_number_outside_the_run_is_clamped_not_extrapolated():
    """A restarted or resumed federation can report a round beyond num_rounds. The
    linear factor goes negative past the end, and a negative learning rate is an
    optimiser walking uphill -- with nothing in any log that would say so."""
    assert round_lr(9, 6) == round_lr(6, 6)
    assert round_lr(0, 6) == round_lr(1, 6)
    for r in (-3, 0, 1, 5, 6, 99):
        lr0, lrf = round_lr(r, 6)
        assert lr0 > 0 and lrf > 0
