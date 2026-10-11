---
name: grace-torch-pr
description: Open the pull request of a board issue into torch-backend - the branch and worktree conventions, the tidy-up pass before opening it, the checks to run, how to write the description (outcome, evidence, limits, review focus), commit messages, upstream units, and what to do after the merge. Use when preparing, opening or updating a pull request, when writing a commit message or PR text, and before pushing a branch.
---

# grace-torch-pr

Procedure for the pull request of one issue. The board side (picking the issue, findings, the Definition of Done, `board.py done`) is in `grace-torch-ticket`; the rules this depends on are in `CLAUDE.md` and `grace-torch`. Run `tools/board.py` from the repository root.

## One issue, one PR

```
branch: <id>-<slug>                e.g. clean1-triage-table   (the id in lower case)
base:   torch-backend              (the integration branch on the fork; never master, never ICAMS)
```

- When the work is ready (see `CLAUDE.md`), open a **draft** PR yourself and ask the user to review it; marking it ready, auto-merge and merging are never yours. Its description contains `Refs #<issue number>` (not `Closes`: the issue is closed by `board.py done` after the merge) and the Verify
  output and per-unit coverage. Then `python tools/board.py status CLEAN1 "PR Open"` (it refuses unless an open PR references the issue). Make sure your `torch-backend` is current first.
- **Work in your own worktree** (rule in `CLAUDE.md`): `git worktree add -b <id>-<slug> ../<repository>-<id> origin/torch-backend`, then `git branch --unset-upstream` in it: git sets the upstream
  to `origin/torch-backend`, and a plain `git push` would then try to update the integration branch itself instead of the PR branch. **When the PR is merged** and `board.py done` has run:
  `git worktree remove ../<repository>-<id>` (no `--force`; a refusal means uncommitted work, so report it) and `git branch -d <branch>` (it refuses when your checkout of `torch-backend` lacks the merge: pull it, or check ancestry with
  `git merge-base --is-ancestor <branch> origin/torch-backend`); the remote branch stays unless the user says to delete it. `git worktree list` shows what is still around.
- **The issue's scope is binding.** If it turns out bigger than its estimate, propose a split with a finding. Do not expand silently and do not fold in
  an unrelated fix you noticed. A tolerance change, regenerating a golden reference, or reformatting existing files is never part of a feature PR.
- **Units that could be offered upstream** (non-breaking, not about PyTorch, no new dependency) are cut from `upstream/master` as `pr/U<n>-<slug>` so the
  fork never waits; nothing is sent to ICAMS without an explicit go for that unit. `CLAUDE.md`, `.claude/`, `tools/` and `baselines/` never appear on such a branch:
  run `python tools/check_pr_branch.py` (it fails on any fork-only path in the diff against `upstream/master`). Draft PR text goes through `python tools/board.py sanitise`.
- Commit messages are imperative with `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`; test commits come before the change they protect;
  a refactor never shares a commit with a behaviour change. You may commit on the work branch of the issue; you push it to open the draft PR and to update that PR, and never commit to `torch-backend` or `master` directly.

## Before you open the PR

- **Tidy up your code first** (rule in `CLAUDE.md`): four reviews of the diff against `origin/torch-backend`, one per lens, run as parallel subagents or with the `simplify` skill. Give each the diff and the touched files, and tell it to report, not to edit.
  Reuse: an existing helper, base class or tool that does the same, duplicates the diff adds (`python tools/check_clones.py check`). Simplification: nesting, dead branches, parameters nobody varies, functions above complexity 10 or 50 statements.
  Efficiency: a cost that a measurement shows (rule R7), not a guess. Altitude: logic placed in the wrong layer or module, private names used across modules, an import against the direction of the layers.
  Apply what is right in `refactor:` commits (each fix of a unit below 90% coverage first gets its characterization test), run the checks again, and put the outcome per lens in the PR's Review focus: what was found, what was fixed, what was left and why.
  A finding outside the files of the PR is commented on the issue (`board.py finding`), not fixed in this PR.
- Run the Verify commands of the issue and go through the Definition of Done checklist in its body (every box ticked or marked not applicable): exit criterion met, touched units at 90% coverage **before** the
  change, new code covered with both test layers, `ruff`/`ty` clean on new code, import contracts green, docstrings, no commented-out code, single concern.
- Run the tests that can catch a behaviour change (from the repository root): the full suite with `-n 4 --dist load`, compared with `baselines/` through
  `tools/junit_outcomes.py compare`. Skipped suites are named, not reported as clean.
- **A GPU- or HPC-marked issue needs that hardware.** State what ran where; never report GPU or real-weight results from a local CPU run.

## Writing the PR summary

Pull requests into `torch-backend` use the fork's template, `.github/PULL_REQUEST_TEMPLATE/torch-backend.md`. Open it as a draft when the work is ready, then ask the user to review it (the user marks it ready and merges).

```bash
python tools/board.py pr-body CLEAN1 > body.md      # template pre-filled: Refs, the exit criterion, the Definition of Done state
# fill the sections, delete the comments
python tools/board.py pr-check CLEAN1 body.md       # must report 0 problems
gh pr create --repo akwarii/grace-tensorpotential --base torch-backend --draft --title "CLEAN1 — <issue title>" --body-file body.md
```

Rules for the text:

1. **Outcome first.** The Summary is two to four plain sentences: what is different after this PR and why it matters. No account of how you got there, no copy of the issue (link it
   with `Refs #<issue>`; never `Closes`, because `board.py done` resolves the issue after the merge).
2. **Evidence, not assertion.** Every claim ("tests pass", "behaviour-preserving", "coverage 93%") comes with the command and the number, from this task: counts against `baselines/`,
   per-unit coverage before and after for every touched unit, comparisons with the oracle, planted mutants caught. Avoid "should" and "probably" for anything you can run.
3. **Say what kind of change it is.** Behaviour-preserving, or exactly what changes. List the touched files and units, and what was deliberately left alone.
4. **Be honest about limits.** List failing or skipped tests with ids and causes, anything you could not run (GPU, HPC, real weights, the other pandas version) and why, and a
   partly met exit criterion as partly met.
5. **Unexpected findings** are linked to their issue comments, or "none". **Review focus** names one to three concrete places (the riskiest unit, the assertion that matters,
   the tolerance row used), what could go wrong, and how to undo it (usually: revert).
6. **Style.** Impersonal and concrete ("Adds ...", "Replaces ..."), short paragraphs, numbers with units, no first-person chatter, no marketing, no emoji. One concern per PR within the
   size budget (about 400 lines without fixtures), or the reason it is larger.
7. **Sanitised.** `pr-check` fails on unsanitised text (rules in `CLAUDE.md`).
8. **Keep it current.** When the diff changes after the description was written, update the description in the same step. After opening, `board.py status <ID> "PR Open"`.
   Attribution lines on fork PRs follow the environment's configuration; PR text for upstream (`pr/U*`) branches carries no AI attribution.
