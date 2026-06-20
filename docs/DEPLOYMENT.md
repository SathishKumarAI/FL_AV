# Production Deployment (Flower Deployment Engine)

Moves FL_AV off the single-process simulation onto a real distributed topology:
one **SuperLink** (coordinator) and N **SuperNodes** (data owners), communicating
over gRPC with TLS. Each SuperNode trains YOLOv8 on its own shard and only model
weights cross the network.

> These are deployment **artifacts**; they have not been run end-to-end in CI
> (no SuperLink host / certs / GPU there). Pin image tags and the
> torch/torchvision/CUDA triad to your hardware before going live.

## Topology

```
                       flwr run . remote-deployment
                                  │ (Exec API :9093, TLS)
                          ┌───────▼────────┐
                          │   SuperLink     │  CPU
                          │  :9091 Driver   │
                          │  :9092 Fleet    │
                          │  :9093 Exec     │
                          └───────▲────────┘
                  gRPC+TLS :9092  │  (root cert = ca.crt)
            ┌─────────────────────┼─────────────────────┐
        ┌───▼────┐            ┌───▼────┐            ┌───▼────┐
        │SuperNode│ GPU       │SuperNode│ GPU       │SuperNode│ GPU
        │ part 0  │           │ part 1  │           │ part 2  │
        │ batch_1 │           │ batch_2 │           │ batch_3 │
        └─────────┘           └─────────┘           └─────────┘
```

## 1. Certificates (TLS)

```bash
cd my-project
bash scripts/gen_certs.sh <superlink-host>   # DNS/IP clients will dial
```

Writes `certs/` (gitignored): `ca.crt`, `server.pem`, `server.key`. The server
cert **SAN must match the address** clients dial or the TLS handshake fails. In
production use a real CA and rotate certs (a rotation requires a SuperLink
restart). Treat keys as secrets (Docker/K8s secrets), never commit them.

## 2. Federation config

`pyproject.toml` defines:

```toml
[tool.flwr.federations.remote-deployment]
address = "127.0.0.1:9093"            # SuperLink Exec API
root-certificates = "certs/ca.crt"
```

Point `address` at your SuperLink's Exec API host:port.

## 3. Bring up the cluster

```bash
cd my-project
docker compose up --build          # 1 SuperLink + 3 GPU SuperNodes (TLS)
```

- SuperNode image is CUDA-based; GPU passthrough needs the **NVIDIA Container
  Toolkit** on the host. It asserts `torch.cuda.is_available()` at startup and
  exits loudly if the GPU isn't visible.
- Each SuperNode mounts **only its own** `batch/batch_N` shard (data locality)
  and reads `FL_AV_DATA_ROOT=/data`.

Manual (no Docker) equivalent:

```bash
# SuperLink
flower-superlink --ssl-ca-certfile certs/ca.crt \
  --ssl-certfile certs/server.pem --ssl-keyfile certs/server.key
# Each SuperNode (own host/GPU)
flower-supernode --superlink <host>:9092 --root-certificates certs/ca.crt \
  --node-config "partition-id=0 num-partitions=3"
```

## 4. Run the federation

```bash
cd my-project
flwr run . remote-deployment
# override hyperparameters:
flwr run . remote-deployment --run-config "num_server_rounds=5 strategy='fedprox' proximal_mu=0.1"
```

The aggregated global model is written to `checkpoints/global_round_N.pt` and
`checkpoints/global_last.pt` on the SuperLink side (see [ARCHITECTURE.md](ARCHITECTURE.md)).

## Ports

| Port | API | Who connects |
|------|-----|--------------|
| 9091 | Driver / ServerApp | internal |
| 9092 | Fleet | SuperNodes |
| 9093 | Exec | `flwr run` |

## Production checklist

- [ ] Real CA-issued certs; SAN matches the dialed host; rotation plan.
- [ ] Pin `flwr/superlink` tag and torch/torchvision/CUDA to your driver.
- [ ] `FL_AV_DATA_ROOT` points at each node's mounted shard.
- [ ] Resource limits / GPU reservations per SuperNode.
- [ ] Ship `checkpoints/` to a model registry / object store (see Phase D).
- [ ] Secrets via Docker/K8s secrets, not images or `pyproject.toml`.
