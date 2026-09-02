# Phase G — GitHub management: make the discipline a rule, not a document

**Date:** 2026-08-16 · **Phase:** G of [`docs/PHASED_PLAN.md`](../PHASED_PLAN.md) ·
**Backlog:** 96, 97, 87, 99, 100

## Goal

The project's process — branch per change, PR with numbers, squash merge, verify before
claiming — currently lives in `CONTRIBUTING.md` and in the author's habits. Move the
parts that can be enforced into the repository, so a tired session cannot skip them.

The specific thing this defends against: **every defect in this project's history is a
path, CWD, encoding or environment trap**, and CI runs on one OS. Seven such defects were
found in a single session merely by running the pipeline from a script instead of a shell.

## Hard constraints

- **No secrets, no hosted services.** Nothing here needs a token beyond `GITHUB_TOKEN`.
  Hosted trackers are rejected on the credentials rule, not on quality.
- **CI must not need the dataset.** BDD100K is 7.6 GB in a kagglehub cache. The smoke run
  uses the synthetic/tiny path the existing CI already uses.
- **A failing check blocks the merge.** A required check that can be dismissed is a
  comment with extra steps.
- Do not automate the backlog into issues wholesale. 100 auto-created issues is noise;
  one issue per phase with the items as a checklist is a tracker.

## Inputs

- `.github/workflows/ci.yml` — tests plus an end-to-end federation smoke on CPU that
  asserts the aggregate checksum changes between rounds
- `.github/workflows/label.yml`, `.github/labeler.yml`, `.github/PULL_REQUEST_TEMPLATE.md`
- `CONTRIBUTING.md` — the process, written down
- Branch `feat/pipeline-observability` is where recent work sits; `main` is reached only
  through a squash-merged PR

## Deliverables

| # | Item | What it prevents |
|---|---|---|
| 1 | **CI matrix: `windows-latest` + `ubuntu-latest`** on the test job | the entire class of defect this project actually ships. Expect it to go red the first time — that redness is the deliverable |
| 2 | **A cp1252 job**, or a step that runs the pipeline with a non-UTF-8 console encoding | two of the seven standalone defects were an unencodable character killing an output thread |
| 3 | **Nightly scheduled smoke on `main`** (backlog 97) | `main` is green at merge time and never checked again. Dependencies move; ultralytics 8.4 already removed `python -m ultralytics.cfg` under this project |
| 4 | **Branch protection on `main`**: PR required, CI required, squash-only, delete branch on merge, no force-push | makes `CONTRIBUTING.md` executable. Configure with `gh api` and commit the command in `scripts/` so it is reproducible, not clicked |
| 5 | **Required evidence block in the PR template** — the verification commands and their real output, the holdout numbers, the checksum sequence. Empty or `n/a` is a review failure | "should work" has cost this project GPU-days |
| 6 | **Run bundle as a CI artifact** (backlog 87) — config + fleet manifest + metrics + report, uploaded per smoke run | a number quoted in a PR becomes downloadable rather than asserted |
| 7 | **One GitHub issue per phase**, items as a checklist, linked from `docs/PHASED_PLAN.md`; `BACKLOG_100.md` stays the catalogue | a list in a markdown file cannot be assigned, closed, or referenced from a commit |
| 8 | **Labels matching the work, not the language** — `phase:1-runtime`, `phase:2-schedule`, `⚠ my-project`, `needs-gpu`, `docs` — wired into `.github/labeler.yml` by path | ⚠ changes need a different review; the label should say so before a human reads the diff |
| 9 | **ADRs** (backlog 100) in `docs/adr/`: assemble-don't-build, the `pipeline/` isolation rule, the partition design, the never-commit-data rule | each has been re-argued at least once. Written down, they stop being re-argued |
| 10 | **A `gh`-driven release per phase** — tag, notes generated from the squashed commits, the phase's result table pasted in | the result outlives the branch, which is the `exp/` rule applied to phases |

## Definition of done

```bash
gh workflow list
gh run list --limit 5 --json name,conclusion,headBranch
gh api repos/:owner/:repo/branches/main/protection | jq '.required_status_checks.contexts'
python -m pytest pipeline/tests -q
python -m pytest my-project/tests -q
```

In the commit body:

- the matrix run's result on both OSes, including whatever Windows-only failure it
  surfaced and whether it was fixed or filed (**a found bug is fixed or filed, never
  silently left** — say which)
- the protection settings as returned by the API, not as intended
- a link to the nightly run's first execution
- the artifact bundle's contents, listed

## Out of scope

Self-hosted GPU runners, release automation to PyPI, Dependabot beyond what is already
configured, any CI job that would need the 7.6 GB dataset, and code coverage gates —
this repo's tests are chosen to catch named failures, and a coverage percentage would
reward writing different ones.
