# The CPU reproduction path, containerised.
#
# What this image is FOR: running this project's checks on a machine that is not this
# machine. `docker build` then `docker run` gives you both test suites — 33 in
# my-project, 118 in pipeline — with no venv, no CUDA, and none of the Windows traps
# that make a fresh clone here cost an hour.
#
# What it is NOT for, stated up front so nobody discovers it by wasting an afternoon:
#
#   * It cannot train. There is no CUDA here and no GPU passthrough configured. Torch
#     is the CPU wheel deliberately — the whole point is a small image that runs
#     everywhere, and this project's real runs need a Blackwell card and cu128.
#   * It has no data. BDD100K is 7.6 GB in a kagglehub cache and this repo's first
#     hard rule is that data is never committed; `.dockerignore` enforces that the
#     build context cannot smuggle any in. A container that trained on nothing would
#     be worse than one that says it cannot train.
#   * It is therefore not a deployment. Flower's Deployment Engine (SuperLink +
#     SuperNodes over gRPC+TLS) is configured in my-project/pyproject.toml under
#     `[tool.flwr.federations.remote-deployment]` and is a different exercise.
#
# Size, measured rather than guessed: 2.05 GB. The CPU torch wheel is most of it and
# `my-project/batch/*/labels` — ten shards of committed label text, ~291 MB — is the
# rest. Those are real repo content, not data that slipped past .dockerignore: the
# image contains zero .jpg and zero .pt, asserted after the build. Trimming them would
# mean first establishing which tests read a real shard layout, and 291 MB of a 2 GB
# image is not worth finding that out wrong.
#
# Build and run:
#   docker build -t federated-yolov8:cpu .
#   docker run --rm federated-yolov8:cpu                     # both suites
#   docker run --rm federated-yolov8:cpu pytest pipeline/tests -q
#   docker run --rm -it federated-yolov8:cpu bash            # poke at it

FROM python:3.12-slim

# 3.12 and not 3.13 for the same reason CI pins it: flwr[simulation] pulls ray, whose
# dependency marker is python>=3.11,<3.13.

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # flwr >= 1.31 otherwise builds its own runtime env with `uv sync` and reinstalls
    # torch from PyPI, discarding the CPU wheel pinned below. On a real host that
    # silently costs 5.5x the wall clock; here it would just bloat the image.
    FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1

WORKDIR /repo

# Dependencies before source, so editing a .py file does not re-download torch.
# Mirrors CI's two install steps exactly: if this drifts from .github/workflows/ci.yml
# then a green container stops meaning a green CI.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir flwr numpy pyyaml pytest

COPY . /repo

# `pytest my-project/tests` imports `utils.*` and `my_project.*` by top-level name.
ENV PYTHONPATH=/repo/my-project

# Ultralytics is NOT installed: my-project/tests/conftest.py stubs it, and the
# pipeline package imports it lazily inside the two functions that score a
# checkpoint. Installing it would add ~2 GB for code these tests never reach.
CMD ["pytest", "my-project/tests", "pipeline/tests", "-q"]
