"""The 13-class head starts as a detector, not as noise.

`YOLO(yaml).load(coco.pt)` transfers 349 of 355 tensors. The six it cannot are the
three classification convolutions, whose shapes differ between 80 and 13 classes, so
they stay randomly initialised. Round 1 of every federation this project has run was
spent teaching that head what a car is while backpropagating the noise into a backbone
that already knew.

Measured on 280 held-out BDD100K images with no training at all:

    random head    mAP50 = 0.0058
    warm-started   mAP50 = 0.2664

A toy Detect stand-in is used below rather than the real model, so these run without
the 22 MB checkpoint -- the same reason test_weights.py uses TinyBNNet.
"""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import yaml

from pathlib import Path

from my_project.get_set_model import (APPROXIMATE_FROM_COCO, BDD_CLASSES,
                                      warm_start_head)

SCALES = 3


class FakeDetect(nn.Module):
    """Detect's shape, not its behaviour: `nl` branches ending in a 1x1 per class."""

    def __init__(self, nc: int, width: int = 8):
        super().__init__()
        self.nc = nc
        self.cv3 = nn.ModuleList(
            nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.Conv2d(width, nc, 1))
            for _ in range(SCALES))


class FakeModel(nn.Module):
    """A DetectionModel's shape: `.model` is a Sequential ending in the head."""

    def __init__(self, nc: int, names=None, width: int = 8):
        super().__init__()
        self.model = nn.Sequential(nn.Identity(), FakeDetect(nc, width))
        self.names = names or {}


def _class_conv(model, scale):
    return model.model[-1].cv3[scale][-1]


def _coco_like(width: int = 8):
    """An 80-class head whose rows are all distinguishable from one another."""
    names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
             "truck", "boat", "traffic light", "fire hydrant", "stop sign"]
    names += [f"filler{i}" for i in range(80 - len(names))]
    coco = FakeModel(80, dict(enumerate(names)), width)
    with torch.no_grad():
        for s in range(SCALES):
            conv = _class_conv(coco, s)
            for k in range(80):
                conv.weight[k] = float(k + 1)      # row k is recognisably row k
                conv.bias[k] = float(-(k + 1))
    return coco, names


def test_the_shared_classes_are_copied_and_the_rest_are_left_random():
    """Nine of BDD's thirteen classes exist in COCO. The other four -- rider, trailer,
    other person, other vehicle -- have no counterpart, and inventing one would be
    worse than a random row: `rider` is left out on purpose because COCO's nearest
    label is `person`, the class BDD most often confuses it with."""
    coco, coco_names = _coco_like()
    ours = FakeModel(13)
    before = [_class_conv(ours, s).weight.detach().clone() for s in range(SCALES)]

    warmed = warm_start_head(ours, coco)

    assert warmed == ["person", "car", "truck", "bus", "train", "motorcycle",
                      "bicycle", "traffic light", "traffic sign"]
    index = {n: i for i, n in enumerate(coco_names)}
    for j, name in enumerate(BDD_CLASSES):
        source = name if name in index else APPROXIMATE_FROM_COCO.get(name)
        for s in range(SCALES):
            row = _class_conv(ours, s).weight[j]
            if name in warmed:
                assert torch.equal(row, _class_conv(coco, s).weight[index[source]]), name
                assert not torch.equal(row, before[s][j]), f"{name} did not actually move"
            else:
                assert torch.equal(row, before[s][j]), f"{name} should still be random"


def test_a_head_that_does_not_correspond_is_left_random_rather_than_guessed():
    """Different backbone width means the rows are not comparable at all. Copying
    anyway would put arbitrary numbers in a head that then looks pretrained."""
    coco, _ = _coco_like(width=8)
    ours = FakeModel(13, width=16)
    before = _class_conv(ours, 0).weight.detach().clone()

    assert warm_start_head(ours, coco) == []
    assert torch.equal(_class_conv(ours, 0).weight, before)


def test_the_class_list_is_the_one_the_shards_are_built_from():
    """The order IS the class index -- `pipeline/build_fleet.py` reads this same file
    to write every shard's data.yaml. A list that agrees on membership but not on
    order would warm-start `car` into the `truck` row and score plausibly."""
    src = Path(__file__).resolve().parents[1] / "batch" / "batch_1" / "data.yaml"
    declared = yaml.safe_load(src.read_text())
    assert declared["names"] == BDD_CLASSES
    assert declared["nc"] == len(BDD_CLASSES)


def test_warming_an_already_warm_head_changes_nothing():
    """The client warms its head and is then handed the server's, which was warmed
    the same way. Running it twice has to be a no-op or the two sides drift and
    set_weights' strict=True turns a silent asymmetry into a failed round."""
    coco, _ = _coco_like()
    ours = FakeModel(13)
    warm_start_head(ours, coco)
    once = [_class_conv(ours, s).weight.detach().clone() for s in range(SCALES)]
    warm_start_head(ours, coco)
    for s in range(SCALES):
        assert torch.equal(_class_conv(ours, s).weight, once[s])
