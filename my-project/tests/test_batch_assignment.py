"""Each client must receive its OWN shard id.

Regression guard for a bug the server's own logs could not show: FedAvg builds a
single FitIns/EvaluateIns and hands the same object to every client, so writing
``config["batch_id"]`` per client mutated one shared dict. The server logged two
different assignments while both clients actually received the last one, and the
federation silently trained one shard twice.
"""
from flwr.common import Code, Parameters, Status

from my_project.server_app import CustomBatchStrategy


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
