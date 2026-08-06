"""Any Flower strategy must be reachable, and none of them silently.

The shard assignment, the aggregate checksum log and the checkpointing used to be
welded to FedAvg by inheritance. These assert the mixin composes with other
aggregators, that the project's behaviour still wins in the MRO, and that a name
nobody registered is refused rather than quietly served as FedAvg.
"""
import numpy as np
import pytest
from flwr.common import ndarrays_to_parameters

from my_project.server_app import (BatchAssignmentMixin, CustomBatchStrategy, STRATEGIES,
                                   build_strategy)


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    """Constructing a strategy truncates logs/metrics.csv and mkdirs checkpoints/."""
    monkeypatch.chdir(tmp_path)


def _kwargs():
    return dict(
        project_kwargs=dict(num_rounds=2, checkpoint_dir="checkpoints"),
        common_kwargs=dict(
            fraction_fit=1.0, min_fit_clients=2, min_evaluate_clients=2,
            min_available_clients=2,
            initial_parameters=ndarrays_to_parameters([np.zeros(4, dtype=np.float32)]),
        ),
    )


def test_every_registered_strategy_can_be_built():
    assert {"fedavg", "fedprox", "fedadam", "fedyogi", "fedadagrad"} <= set(STRATEGIES)
    for name in STRATEGIES:
        strategy = build_strategy(name, **_kwargs())
        mro = type(strategy).__mro__
        assert mro[1] is BatchAssignmentMixin, (
            f"{name}: project behaviour must resolve before the aggregator, or "
            f"configure_fit would not assign shards")
        assert issubclass(type(strategy), STRATEGIES[name])
        assert strategy.batch_id_range == (1, 10)


def test_an_unknown_strategy_is_refused_not_quietly_replaced():
    """A run labelled fedadam that is actually FedAvg is the failure mode this
    repo keeps producing. It must raise at server start instead."""
    with pytest.raises(ValueError) as excinfo:
        build_strategy("fedwhatever", **_kwargs())
    assert "fedwhatever" in str(excinfo.value)
    assert "fedavg" in str(excinfo.value), "the error should list what is available"


def test_tuning_arguments_reach_the_strategies_that_take_them():
    """FedAdam has eta; FedAvg does not. One call site, filtered by signature."""
    adam = build_strategy("fedadam", tuning_kwargs={"eta": 0.05}, **_kwargs())
    assert float(adam.eta) == pytest.approx(0.05)

    # Handing FedAvg the same kwarg must not raise -- it is dropped and logged.
    avg = build_strategy("fedavg", tuning_kwargs={"eta": 0.05}, **_kwargs())
    assert not hasattr(avg, "eta")


def test_fedprox_is_fedavg_plus_a_proximal_term_shipped_to_clients():
    proj = dict(num_rounds=2, checkpoint_dir="checkpoints", proximal_mu=0.1)
    kwargs = _kwargs()
    kwargs["project_kwargs"] = proj
    prox = build_strategy("fedprox", **kwargs)
    assert prox.proximal_mu == pytest.approx(0.1)
    assert build_strategy("fedavg", **_kwargs()).proximal_mu == 0.0


def test_the_pipeline_offers_exactly_what_the_server_registers():
    """pipeline/stages.py mirrors this list rather than importing it, so that the
    pipeline package stays free of flwr and ultralytics. Mirrors drift; this is the
    guard that says so before a user picks a name the server will reject."""
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from pipeline import stages

    assert set(stages.STRATEGIES) == set(STRATEGIES), (
        "pipeline.stages.STRATEGIES and server_app.STRATEGIES have drifted: "
        f"only in pipeline {sorted(set(stages.STRATEGIES) - set(STRATEGIES))}, "
        f"only in server {sorted(set(STRATEGIES) - set(stages.STRATEGIES))}")


def test_the_old_name_still_means_what_the_docs_say():
    """CustomBatchStrategy is named in the README, three docs and another test."""
    assert CustomBatchStrategy.__mro__[1] is BatchAssignmentMixin
    assert CustomBatchStrategy.__mro__[2] is STRATEGIES["fedavg"]
