# Security

## Reporting

Open a [private security advisory](https://github.com/SathishKumarAI/federated-yolov8-object-detection/security/advisories/new).
Do not open a public issue for anything exploitable. Expect a first reply within a week —
this is a research project maintained by one person, not a product with an on-call rota.

## What this project is, in security terms

A local research pipeline. It is worth being explicit about the boundaries, because
"federated" is a word that invites assumptions this code does not earn:

| | |
|---|---|
| **No credentials, anywhere** | kagglehub downloads anonymously. There is no `kaggle.json`, no `.env`, no token — not read, not copied, not logged. Hosted experiment trackers were rejected for this reason rather than on quality. A PR that introduces a credential of any kind will be refused |
| **No data leaves the machine** | BDD100K lives in the kagglehub cache and shards hardlink onto it. Nothing is committed, uploaded or sent anywhere. A test asserts the ignore rules still match real generated paths |
| **The dashboard binds to loopback only** | `127.0.0.1`, never `0.0.0.0`. It has no authentication and is not meant to have any: it can start a training run, so exposing it to a network would be handing out remote code execution. Do not put it behind a tunnel or a reverse proxy |
| **Every route that maps a URL onto a path is guarded** | one `safe_child` helper for traversal, plus an allowlist for the trainer's own output files. Both are covered by tests. If you find a way past either, that is a real report |
| **Federated learning is not privacy** | gradients leak, and this project has no differential-privacy wrapper today. Nothing here should be read as a claim that a vehicle's data is unrecoverable from its updates. Adding DP with a stated ε is phase 5 of `docs/PHASED_PLAN.md` and is not done |

## Dependencies

Floors are pinned in `my-project/pyproject.toml` and `pipeline/requirements.txt`, with a
comment saying why each floor is where it is. Upgrades are verified by CI on Windows and
Linux, plus a two-round federated smoke that asserts the aggregate checksum changes
between rounds — a dependency that silently breaks training fails that job rather than
producing a quieter wrong number.
