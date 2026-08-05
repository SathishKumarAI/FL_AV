"""Each client must receive its OWN shard id.

Regression guard for a bug the server's own logs could not show: FedAvg builds a
single FitIns/EvaluateIns and hands the same object to every client, so writing
``config["batch_id"]`` per client mutated one shared dict. The server logged two
different assignments while both clients actually received the last one, and the
federation silently trained one shard twice.
"""
import pytest
from flwr.common import Parameters

from my_project.server_app import CustomBatchStrategy


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    """Run in a scratch dir.

    Constructing the strategy has side effects on relative paths: MetricsLogger
    truncates ``logs/metrics.csv`` and ``__init__`` mkdirs ``checkpoints/``. Without
    this the suite would wipe the artifacts of a real run.
    """
    monkeypatch.chdir(tmp_path)


class _Proxy:
    """Minimal ClientProxy stand-in — configure_fit only reads `cid`."""

    def __init__(self, cid):
        self.cid = cid


class _Manager:
    def __init__(self, n):
        self._clients = [_Proxy(f"client-{i}") for i in range(n)]

    def num_available(self):
        return len(self._clients)

    def sample(self, num_clients, min_num_clients=None, criterion=None):
        return self._clients

    def wait_for(self, num_clients, timeout=None):
        return True


def _strategy():
    return CustomBatchStrategy(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        initial_parameters=Parameters(tensors=[], tensor_type="numpy.ndarray"),
        batch_id_range=(1, 10),
    )


def _batch_ids(instructions):
    return [ins.config["batch_id"] for _, ins in instructions]


def test_configure_fit_gives_each_client_a_distinct_shard():
    ids = _batch_ids(
        _strategy().configure_fit(1, Parameters(tensors=[], tensor_type="numpy.ndarray"), _Manager(2))
    )
    assert len(ids) == 2
    assert len(set(ids)) == 2, f"clients share a shard: {ids}"


def test_configure_evaluate_gives_each_client_a_distinct_shard():
    ids = _batch_ids(
        _strategy().configure_evaluate(1, Parameters(tensors=[], tensor_type="numpy.ndarray"), _Manager(2))
    )
    assert len(set(ids)) == 2, f"clients share a shard: {ids}"


def test_a_client_keeps_its_shard_across_rounds():
    """FL data locality: the same client must get the same shard every round."""
    strategy = _strategy()
    params = Parameters(tensors=[], tensor_type="numpy.ndarray")
    manager = _Manager(2)
    first = _batch_ids(strategy.configure_fit(1, params, manager))
    second = _batch_ids(strategy.configure_fit(2, params, manager))
    assert first == second


def test_a_late_joining_client_does_not_get_a_held_shard():
    """A separate per-round `used_batch_ids` set was cleared each round while the
    client->shard map was not, so a client arriving in round 2 could be handed a
    shard another client was already training."""
    strategy = _strategy()
    params = Parameters(tensors=[], tensor_type="numpy.ndarray")
    held = _batch_ids(strategy.configure_fit(1, params, _Manager(2)))
    # Round 2: the original two plus a third that has never been seen.
    all_ids = _batch_ids(strategy.configure_fit(2, params, _Manager(3)))
    assert len(set(all_ids)) == 3, f"late joiner collided: {held} then {all_ids}"
