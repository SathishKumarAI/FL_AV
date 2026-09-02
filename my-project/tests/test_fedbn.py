"""FedBN: the BatchNorm layers stay this vehicle's own, the rest is the aggregate.

Why this method and not another: the fleet's non-IID axis is *condition* -- night,
rain, overcast, clear -- which is FEATURE shift, and feature statistics are exactly
what BatchNorm buffers encode. FedAvg averages a night vehicle's running_mean with a
clear-daylight one and produces normalisation that describes no vehicle's data.
FedProx, FedAdam and FedAvgM all address drift in parameter space instead.

The failure mode these guard is a *partial* FedBN: some BN tensors kept, some
overwritten. It trains, it logs, it looks like the method, and it is not.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from my_project.get_set_model import batchnorm_keys, get_weights, set_weights

from tests.test_weights import TinyBNNet, _train_a_step


def test_every_batchnorm_tensor_is_found_and_nothing_else_is():
    """Found by module TYPE. A substring match on "bn" would both miss a renamed
    layer and catch unrelated keys, and the result would be a partial FedBN."""
    model = TinyBNNet()
    keys = batchnorm_keys(model)

    assert keys == {"bn.weight", "bn.bias", "bn.running_mean", "bn.running_var",
                    "bn.num_batches_tracked"}, keys
    # Parameters AND buffers: FedBN keeps the whole normalisation layer local, not
    # only its statistics.
    assert "bn.running_var" in keys and "bn.weight" in keys
    # Nothing outside the BN module.
    assert not any(k.startswith(("conv.", "fc.")) for k in keys)


def test_a_model_with_no_batchnorm_reports_none_rather_than_guessing():
    keys = batchnorm_keys(nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU()))
    assert keys == set()


def test_batchnorm_stays_local_while_everything_else_becomes_the_aggregate():
    """The whole method, in one assertion pair."""
    local, incoming = TinyBNNet(), TinyBNNet()
    _train_a_step(local)                       # this vehicle's BN has diverged
    _train_a_step(incoming)
    with torch.no_grad():                      # make the aggregate unmistakably different
        for t in incoming.state_dict().values():
            if t.dtype.is_floating_point:
                t.add_(1.0)

    before = {k: v.detach().clone() for k, v in local.state_dict().items()}
    bn = batchnorm_keys(local)
    assert set_weights(local, get_weights(incoming), keep_local=bn)

    after = local.state_dict()
    for key in bn:
        assert torch.equal(after[key], before[key]), \
            f"{key} was overwritten by the aggregate; this is FedAvg, not FedBN"
    for key in after:
        if key not in bn:
            assert not torch.equal(after[key], before[key]), \
                f"{key} kept its local value; the aggregate did not arrive"


def test_without_the_filter_batchnorm_is_averaged_as_before():
    """The default path must be untouched: FedBN is opt-in, and a change of default
    would silently re-define every historical comparison."""
    local, incoming = TinyBNNet(), TinyBNNet()
    _train_a_step(local)
    _train_a_step(incoming)

    assert set_weights(local, get_weights(incoming))
    for key, value in incoming.state_dict().items():
        assert torch.equal(local.state_dict()[key], value), key


def test_the_wire_format_does_not_change():
    """Skipped tensors are discarded ON RECEIPT, not omitted on send. The transfer is
    a positional zip against state_dict().keys(); dropping entries would shift every
    key after the first BatchNorm and load garbage into the wrong layers -- with
    strict=True passing, because the shapes would still line up often enough."""
    model = TinyBNNet()
    full = get_weights(model)
    assert len(full) == len(model.state_dict())

    short = [a for k, a in zip(model.state_dict().keys(), full)
             if k not in batchnorm_keys(model)]
    assert set_weights(model, short, keep_local=batchnorm_keys(model)) is False, \
        "a short list was accepted; the count check is what stops a silent misalignment"


def test_kept_tensors_are_cloned_not_aliased():
    """load_state_dict copies into the live tensors. Handing it the model's own
    tensor as the source can make the write a no-op that reports success."""
    local, incoming = TinyBNNet(), TinyBNNet()
    _train_a_step(local)
    bn = batchnorm_keys(local)
    kept = {k: local.state_dict()[k].detach().clone() for k in bn}

    assert set_weights(local, get_weights(incoming), keep_local=bn)
    for key, value in kept.items():
        assert torch.equal(local.state_dict()[key], value), key


def test_num_batches_tracked_survives_the_integer_dtype_path():
    """An int64 buffer that round-trips through numpy is where dtype bugs live, and
    it is kept rather than copied under FedBN -- a different code path again."""
    local, incoming = TinyBNNet(), TinyBNNet()
    _train_a_step(local)
    n_before = local.state_dict()["bn.num_batches_tracked"].item()
    assert n_before > 0

    assert set_weights(local, get_weights(incoming), keep_local=batchnorm_keys(local))
    kept = local.state_dict()["bn.num_batches_tracked"]
    assert kept.item() == n_before
    assert kept.dtype == torch.int64
