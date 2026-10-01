---
name: grace-torch-ticket
description: Take a board issue (milestone M<stage>.<n> or gate G<n>) of the fork from Todo to Done - the session ritual, the board helper commands, recording unexpected findings, the PR into torch-backend, and the review gate. Use when starting, working on, or wrapping up any issue of the project board, and whenever you find something the issue did not predict.
---

# grace-torch-ticket

Procedure for one issue. The rules it depends on are in `grace-torch`; read that first. The board helper is `tools/board.py`; it can only touch
the fork. Run it from the repository root.

## Session ritual

1. **Pick.** `python tools/board.py next` lists issues whose hard dependencies are Done, best first. Take the one the user names if they name one.
2. **Read the context.** `python tools/board.py context M0.2` prints the issue, its comments, and the comments of its hard dependencies. Earlier
   agents recorded what surprised them there. Then read the files the issue names.
3. **Restate the exit criterion in your own words before writing any code.** If restating exposes an ambiguity, raise it on the issue
   (`finding`), not in the implementation.
4. **Start.** `python tools/board.py start M0.2` moves Status from **Todo to In Progress** and comments. It refuses when a hard dependency is not Done;
   say so and stop rather than building on unfinished work (`--force` only if the user said so).

## Record unexpected findings at once

An **unexpected finding** is anything different from what the issue or plan says: a count that is not the stated one, a tool or test that behaves
otherwise, a bug found on the way, a number that varies, a dependency that is missing, a decision you had to take. Do not keep it in your head or
in a final summary; a later agent will start from the issue.

```bash
python tools/board.py finding M0.2 "what you saw; the command and the numbers; what it changes for later work" --also M1.9 G0
```

Give the evidence (command, output, numbers), not only the conclusion, and name the later issues it affects with `--also`. Routine progress is not a
finding. If you discover that the board itself is wrong (a day estimate, a dependency, a priority, a missing issue), say so in a finding and tell the user; the
`Depends on` line of an issue and its native "blocked by" relations are the dependency data, and they are changed by the user or on the user's go.

## New skills need validation

Adding a line or a gotcha to an existing skill or to `CLAUDE.md` is part of the work. **Creating a new skill is not**: propose its name, description and outline to the
user first, and create it only after the user validates it.

## Keep the issues current

Issues go stale as the work moves: a file is renamed, a command changes, a count or a decision is superseded, a dependency is added. Fix the text of every issue
your change makes stale **in the same task**, not later, and run `python tools/board.py lint` before you finish; it reports removed files, unknown ids and
issue numbers, appendices without a reference issue, old counts and second-person wording. Fix every `STALE` finding, also the ones you did not cause, and
sanitise as you go (the helper does it for what it posts).

## Keep the Definition of Done current

The issue body ends with a **Definition of Done** checklist. **Every time you finish a task that belongs to the issue, tick the boxes it satisfies**, with the
evidence; do not leave the ticking to the end.

```bash
python tools/board.py dod M0.2                                   # numbered checklist
python tools/board.py check M0.2 1 4 --note "command and result"  # tick boxes 1 and 4, comment the evidence
python tools/board.py na M0.2 7 "why it does not apply"           # a box that cannot apply is marked, never left blank
python tools/board.py uncheck M0.2 4                              # if a later change invalidates it
```

Only tick what you verified in this session. `done` refuses while a box is open (`--waive "reason"` records an explicit exception on the issue).

## One issue, one PR

```
branch: <stage-id>-<slug>          e.g. m0-2-triage-table
base:   torch-backend              (the integration branch on the fork; never master, never ICAMS)
```

- Open a draft PR only when the user tells you to; committing on the work branch and preparing and checking the description beforehand are allowed, merging is never yours. Its description contains `Refs #<issue number>` (not `Closes`: the issue is closed by `board.py done` after the merge) and the Verify
  output and per-unit coverage. Then `python tools/board.py status M0.2 "PR Open"` (it refuses unless an open PR references the issue). Make sure your `torch-backend` is current first.
- When you create the PR branch with `git worktree add -b <branch> <dir> origin/torch-backend`, run `git branch --unset-upstream` in it: git sets the upstream to `origin/torch-backend`,
  and a plain `git push` would then try to update the integration branch itself instead of the PR branch.
- **The issue's scope is binding.** If it turns out bigger than its estimate, propose a split with a finding. Do not expand silently and do not fold in
  an unrelated fix you noticed. A tolerance change, regenerating a golden reference, or reformatting existing files is never part of a feature PR.
- **Units that could be offered upstream** (non-breaking, not about PyTorch, no new dependency) are cut from `upstream/master` as `pr/U<n>-<slug>` so the
  fork never waits; nothing is sent to ICAMS without an explicit go for that unit. `CLAUDE.md`, `.claude/`, `tools/` and `baselines/` never appear on such a branch:
  run `python tools/check_pr_branch.py` (it fails on any fork-only path in the diff against `upstream/master`). Draft PR text goes through `python tools/board.py sanitise`.
- Commit messages are imperative with `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`; test commits come before the change they protect;
  a refactor never shares a commit with a behaviour change. You may commit on the work branch of the issue; you push it only when told to push or to open the PR, and never commit to `torch-backend` or `master` directly.

## Creating an issue

New issues are created only when the user says so. Start from the fork's template, `.github/ISSUE_TEMPLATE/milestone.md` (fork only, never on a `pr/U*` branch): the same sections as the
existing milestone issues plus **Out of scope** (binding) and **Decisions it depends on** (the decided outcomes inlined, so the issue needs no other reading), and the Definition of
Done with the merged-PR box. The `Depends on` line names issue numbers (add the native "blocked by" relations), the Context lines carry the estimate and where it runs, and the text is
sanitised. Add the issue to the board with its fields (Stage, Runs on, Priority, Days, Kind), and run `python tools/board.py lint` afterwards.

## Writing the PR summary

Pull requests into `torch-backend` use the fork's template, `.github/PULL_REQUEST_TEMPLATE/torch-backend.md`. Open one only when the user tells you to (preparing the description earlier is fine).

```bash
python tools/board.py pr-body M0.2 > body.md      # template pre-filled: Refs, the exit criterion, the Definition of Done state
# fill the sections, delete the comments
python tools/board.py pr-check M0.2 body.md       # must report 0 problems
gh pr create --repo akwarii/grace-tensorpotential --base torch-backend --draft --title "M0.2 — <issue title>" --body-file body.md
```

Rules for the text:

1. **Outcome first.** The Summary is two to four plain sentences: what is different after this PR and why it matters. No account of how you got there, no copy of the issue (link it
   with `Refs #<issue>`; never `Closes`, because `board.py done` resolves the issue after the merge).
2. **Evidence, not assertion.** Every claim ("tests pass", "behaviour-preserving", "coverage 93%") comes with the command and the number, from this task: counts against `baselines/`,
   per-unit coverage before and after for every touched unit, comparisons with the oracle, planted mutants caught. Avoid "should" and "probably" for anything you can run.
3. **Say what kind of change it is.** Behaviour-preserving, or exactly what changes. List the touched files and units, and what was deliberately left alone.
4. **Be honest about limits.** List failing or skipped tests with ids and causes, anything you could not run (GPU, HPC, real weights, the other pandas version) and why, and a
   partly met exit criterion as partly met.
5. **Unexpected findings** are linked to their issue comments, or "none". **Review focus** names one to three concrete places: the riskiest unit, the assertion that matters,
   the tolerance row used.
6. **Style.** Impersonal and concrete ("Adds ...", "Replaces ..."), short paragraphs, numbers with units, no first-person chatter, no marketing, no emoji. One concern per PR within the
   size budget (about 400 lines without fixtures), or the reason it is larger.
7. **Sanitised.** `pr-check` fails on unsanitised text; the rules of `CLAUDE.md` apply (no personal data, local paths, e-mail addresses, stray `#N`, `@mentions`, links into other
   repositories).
8. **Keep it current.** When the diff changes after the description was written, update the description in the same step. After opening, `board.py status <ID> "PR Open"`.
   Attribution lines on fork PRs follow the environment's configuration; PR text for upstream (`pr/U*`) branches carries no AI attribution.

## Before you call it done

- Run the Verify commands of the issue and go through the Definition of Done checklist in its body (every box ticked or marked not applicable): exit criterion met, touched units at 90% coverage **before** the
  change, new code covered with both test layers, `ruff`/`ty` clean on new code, import contracts green, docstrings, no commented-out code, single concern.
- Run the tests that can catch a behaviour change (from `tests/`): the full suite with `-n 4 --dist load`, compared with `baselines/` through
  `tools/junit_outcomes.py compare`. Skipped suites are named, not reported as clean.
- **A GPU- or HPC-marked issue needs that hardware.** State what ran where; never report GPU or real-weight results from a local CPU run.
- **An issue is resolved only once its PR is merged into `torch-backend` on the fork** (the user merges). **Done** then requires the exit criterion to be met: `python tools/board.py done M0.2 "Verify output and numbers"` checks that no box is open, comments the
  evidence, sets Done and closes the issue. A gate is closed only when its pass criterion is met.
