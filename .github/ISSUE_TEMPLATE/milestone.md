---
name: Milestone or task
about: A unit of work on the torch-backend effort (fork only; created only when the owner says so)
title: "M<stage>.<n> — <short title>"
labels: ""
---

<!--
Fork-only template. Rules for the text (see the grace-torch-ticket skill): the issue must be implementable without opening anything else; use the same
section names as the existing milestone issues; sanitise everything (`python tools/board.py sanitise`); no personal data, local paths, e-mail addresses,
stray `#N`, `@mentions`, or links into other repositories. Delete every comment before submitting.
-->

**Depends on:** #<issue>, #<issue> · **Blocks:** none · **Related:** none

<!-- `Depends on` is hard (cannot start before these are Done); `Related` is soft (shared code or preferred order). `Blocks` is filled in when later issues name this one. -->

## Context

- Stage <0-8>: <stage name>
- Estimate: <n> focused days
- Priority: <P0-P3> (score <n>); slack <n> days; <on|off> the critical path
- Runs on: <local | HPC | GPU | user>
- Needs from outside the repository: <the owner's go, a download (name, source, size), HPC, GPU, or nothing>
- Upstream unit(s): <U-number, or none> (prepared locally as `pr/U*` branches; nothing is sent to ICAMS without a per-unit go)

## Work

<!-- What to do, concretely: the units, files and commands involved; the measured facts it rests on (numbers, with where they come from). -->

## Exit criterion

<!-- How anyone can tell it is finished: observable, with the commands that prove it and the numbers expected. -->

## Out of scope

<!-- What this issue must NOT do (binding): neighbouring work that belongs to another issue (name it), changes to tolerances, regenerating references, reformatting existing files. If it turns out bigger than its estimate, propose a split as a finding instead of expanding. -->

## Decisions it depends on

<!-- Inline the decided outcome of every decision (D<number>) this issue waits on, so the issue needs no other reading; or "none". -->

## Definition of Done

Tick each box as soon as a task of this issue satisfies it, with the evidence (`python tools/board.py check <id> <n> --note ...`); mark a box that cannot apply with `na` and a reason.

- [ ] Exit criterion met, `Verify` output pasted in the PR
- [ ] Every modified function or class was at 90% coverage or more **before** the change (tests committed first); `tools/check_touched_coverage.py` output pasted
- [ ] New code covered at 90% or more, with logic tests and, where physics applies, tests on physical values from an independent oracle; planted mutants caught
- [ ] `ruff` strict and `ty` clean on new code; legacy ratchet not risen; import-linter contracts green
- [ ] Public API has docstrings; no commented-out code, no `print`, no TODO without an issue link; error messages actionable and tested
- [ ] Diff is one concern within the size budget (about 400 lines without fixtures)
- [ ] Release notes updated if user-visible; divergence ledger row if an upstream file changed
- [ ] A pull request referencing this issue (`Refs #<this issue>`) is merged into `torch-backend` (the issue is resolved only then; `board.py done` checks it and ticks this box)

---
Background: the `Decisions` issue and the issues labelled `reference` ("Appendix A" to "Appendix L").
