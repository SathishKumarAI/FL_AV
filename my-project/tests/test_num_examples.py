"""Tests that num_examples reflects real shard size (FedAvg aggregation weight)."""
from my_project.task import count_shard_examples


def test_train_count_matches_fixture():
    # batch_1 ships train.txt with 6308 image entries (verified fixture).
    assert count_shard_examples(1, "train") == 6308


def test_val_count_matches_fixture():
    assert count_shard_examples(1, "val") == 1010


def test_not_the_old_constant():
    # Guard against regression to the hardcoded 10.
    assert count_shard_examples(1, "train") != 10


def test_counts_differ_across_splits():
    assert count_shard_examples(1, "train") > count_shard_examples(1, "val")


def test_missing_batch_returns_at_least_one():
    # A non-existent shard must never yield 0 (would zero its FedAvg weight).
    assert count_shard_examples(99999, "train") >= 1
