# FedProx Strategy

Optional FedProx-style proximal regularization on top of the FedAvg baseline,
selectable from config. Motivated by Roadmap item #1 — BDD100K shards are
**non-IID** (different scenes/time-of-day per client), where plain FedAvg can
diverge as local models drift apart.

## Why FedProx

FedProx ([Li et al., 2020](https://arxiv.org/abs/1812.06127)) adds a proximal
term to each client's objective:

```
min_w  F_k(w) + (mu/2) * || w - w_global ||^2
```

The `(mu/2)||w - w_global||²` term penalizes local models for straying far from
the global model, which stabilizes training under heterogeneous (non-IID) data
and partial participation.

## How it's implemented here

True FedProx adds the proximal term to the loss at **every SGD step**. This
project trains via Ultralytics' high-level `YOLO.train()`, which does not expose
a per-step loss hook. So the proximal term is applied as a **round-level
approximation in weight space**, after local training completes:

```
w_client  ←  w_client − mu · (w_client − w_global)
```

i.e. each client's update is shrunk toward the global model by factor `mu`
before being sent back for aggregation. `mu = 0` is exactly FedAvg; larger `mu`
pulls harder toward the global model.

- **Server** (`server_app.py`): `server_fn` reads `strategy` / `proximal_mu`
  from `run_config`; `CustomBatchStrategy` ships `proximal_mu` to clients in the
  fit config. Aggregation stays FedAvg (FedProx uses FedAvg aggregation).
- **Client** (`client_app.py`): snapshots the global weights, trains locally,
  then applies the shrinkage above when `proximal_mu > 0`.

> This is a pragmatic approximation, not per-step FedProx. For exact FedProx,
> swap in a custom Ultralytics trainer that adds the proximal term to the loss.

## Usage

Enable via `pyproject.toml`:

```toml
[tool.flwr.app.config]
strategy = "fedprox"
proximal_mu = 0.1
```

Or at runtime without editing the file:

```bash
flwr run . --run-config "strategy='fedprox' proximal_mu=0.1"
```

Typical `mu` values: `0.001`–`1.0`. Start at `0.1`; raise it if clients diverge,
lower it if convergence stalls. Clients that applied the term report
`proximal_mu` in their fit metrics.
