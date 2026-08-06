# Prompts

Every non-trivial piece of work in this repo is built the same way:

```
plan  ->  prompt  ->  code  ->  verify
```

The **prompt** is the artifact in between. It is the brief that the implementation is
written from: what to build, what not to build, which constraints are hard, and what
"done" means. Writing it down before coding is what stops a design decision from
quietly becoming an implementation accident — and it is what makes the work
reproducible by a different person, or a different model, later.

## The rule

**Save the prompt here before writing the code it describes.** Not after. A prompt
reconstructed after the fact documents what happened, not what was intended, and the
gap between those two is exactly the thing worth keeping.

## Naming

```
YYYY-MM-DD-<topic>-<kind>.md
```

`<kind>` is usually `build-prompt`, sometimes `review-prompt` or `debug-prompt`.

## What a prompt must contain

| Section | Why |
|---|---|
| Goal | one paragraph, the outcome not the steps |
| Hard constraints | the things that must not be violated, stated as prohibitions |
| Inputs | files, versions, data, and what is already verified |
| Deliverables | the exact files to produce |
| Definition of done | commands that must pass, with expected output |
| Out of scope | what to deliberately not build |

The constraint and out-of-scope sections matter most. Most wasted work comes from
building something nobody asked for, not from building the requested thing badly.

## Relationship to specs

`docs/superpowers/specs/` holds **designs** — what the system is and why it is shaped
that way, agreed before implementation. This folder holds the **briefs** written from
those designs. A design usually yields one prompt; a large design may yield several,
one per increment.

## Index

| Date | Prompt | Design it came from |
|---|---|---|
| 2026-08-05 | [pipeline + observability](2026-08-05-pipeline-observability-build-prompt.md) | [design](../superpowers/specs/2026-08-05-pipeline-observability-design.md) |
