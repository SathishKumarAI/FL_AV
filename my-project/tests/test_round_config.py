"""What the server tells every client about the round it is about to run.

The pictures Ultralytics draws are not free and were being drawn six times to be kept
once. These assert the decision, not the drawing.
"""
import pytest

from my_project.server_app import round_config


def test_plots_are_drawn_on_the_round_whose_output_survives():
    """The client passes exist_ok=True, so every round overwrote the previous round's
    pictures in the same directory. Drawing them in rounds 1..n-1 cost GPU time for
    files that were destroyed before anything read them."""
    drew = [round_config(r, 6, 4)["plots"] for r in range(1, 7)]
    assert drew == [False, False, False, False, False, True]


def test_a_single_round_run_still_draws_its_pictures():
    """Off-by-one guard: with num_rounds=1 the first round IS the last one, and a
    dashboard with no pictures at all would look like a broken artifact server."""
    assert round_config(1, 1, 4)["plots"] is True


def test_a_run_that_overshoots_its_round_count_still_draws():
    """`>=`, not `==`. A strategy that runs an extra round must not silently produce
    a run with no diagnostics at all."""
    assert round_config(7, 6, 4)["plots"] is True


def test_plots_every_round_restores_the_old_behaviour():
    assert all(round_config(r, 6, 4, plots_every_round=True)["plots"]
               for r in range(1, 7))


def test_lr0_and_mosaic_default_to_sentinels_that_mean_leave_ultralytics_alone():
    """0.0 is a legitimate mosaic value meaning 'off', so it cannot double as
    'unspecified' -- a default silently flipped to its opposite is this project's
    signature bug. Negative is the sentinel; lr0 uses 0.0 because no run wants a
    learning rate of zero."""
    cfg = round_config(1, 6, 4)
    assert cfg["mosaic"] < 0, "mosaic sentinel must not collide with 'mosaic off'"
    assert cfg["lr0"] == 0.0
    assert cfg["optimizer"] == "auto"


def test_the_round_config_carries_nothing_per_vehicle():
    """FedAvg hands ONE FitIns to every client, so anything per-vehicle put here
    arrives as the last value written -- the B9 bug, where the whole fleet trained a
    single shard. batch_id is injected per client in configure_fit instead, and this
    asserts nobody moves it back."""
    cfg = round_config(3, 6, 4)
    assert "batch_id" not in cfg
    assert "proximal_mu" not in cfg


@pytest.mark.parametrize("rounds", [1, 2, 6, 12])
def test_exactly_one_round_draws_pictures(rounds):
    drew = [round_config(r, rounds, 1)["plots"] for r in range(1, rounds + 1)]
    assert sum(drew) == 1, f"{sum(drew)} of {rounds} rounds draw plots, expected 1"
