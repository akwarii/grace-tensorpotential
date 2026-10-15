---
name: grace-torch-ticket
description: Take a board issue (a milestone such as CLEAN1 or TEST2, or a gate such as GATE-CLEAN) of the fork from Todo to Done - the session ritual, the board helper commands, recording unexpected findings, the Definition of Done, creating and closing issues. Use when starting, working on, or wrapping up any issue of the project board, and whenever you find something the issue did not predict.
---

# grace-torch-ticket

Procedure for one issue on the board: picking it, findings, the Definition of Done, creating and closing issues. The pull request itself (branch, tidy-up pass, description) is `grace-torch-pr`. The rules it depends on are in `grace-torch`; read that first. The board helper is `tools/board.py`; it can only touch
the fork. Run it from the repository root.

## Ids

An issue id is a theme and a number, written as in the issue title (`CLEAN1`, `SPEC3`, `TORCH2`, ...); the gates are `GATE-CLEAN`, `GATE-SPEC`, `GATE-MODEL`, `GATE-EQUIV` and `GATE-SIM`.
`python tools/board.py list [Status]` prints every id with its issue number and status. Numbers restart in each theme and follow the dependency order: an issue never depends on a higher number of its own theme (`lint` reminds you).
A new issue takes the next free number, and an insertion in the middle renumbers the later ones only when nobody is working on them (an id then resolves to a different issue, so tell the agents involved).
The first numbering (`M0.2`, `G0`, ...) is retired: the helper still accepts it (each issue has a `- Legacy id:` line, and `resolve` prints the new id), old commits, branches and pull requests keep using it,
and `lint` reports it in issue text. Write the new id everywhere new.

## Session ritual

1. **Pick.** `python tools/board.py next` lists issues whose hard dependencies are Done, best first. Take the one the user names if they name one.
2. **Read the context.** `python tools/board.py context CLEAN1` prints the issue, its comments, and the comments of its hard dependencies. Earlier
   agents recorded what surprised them there. Then read the files the issue names.
3. **Restate the exit criterion in your own words before writing any code.** If restating exposes an ambiguity, raise it on the issue
   (`finding`), not in the implementation.
4. **Start.** `python tools/board.py start CLEAN1` moves Status from **Todo to In Progress** and comments. It refuses when a hard dependency is not Done;
   say so and stop rather than building on unfinished work (`--force` only if the user said so).

## Record unexpected findings at once

An **unexpected finding** is anything different from what the issue or plan says: a count that is not the stated one, a tool or test that behaves
otherwise, a bug found on the way, a number that varies, a dependency that is missing, a decision you had to take. Do not keep it in your head or
in a final summary; a later agent will start from the issue.

```bash
python tools/board.py finding CLEAN1 "what you saw; the command and the numbers; what it changes for later work" --also FIX2 GATE-CLEAN
```

Give the evidence (command, output, numbers), not only the conclusion, and name the later issues it affects with `--also`. Routine progress is not a
finding. If you discover that the board itself is wrong (a day estimate, a dependency, a priority, a missing issue), say so in a finding and tell the user; the
`Depends on` line of an issue and its native "blocked by" relations are the dependency data, and they are changed by the user or on the user's go.

## Keep the issues current

Issues go stale as the work moves: a file is renamed, a command changes, a count or a decision is superseded, a dependency is added. Fix the text of every issue
your change makes stale **in the same task**, not later, and run `python tools/board.py lint` before you finish; it reports removed files, unknown ids and
issue numbers, appendices without a reference issue, old counts and second-person wording. Fix every `STALE` finding, also the ones you did not cause, and
sanitise as you go (the helper does it for what it posts).

## The project table

The table view shows Status, Stage, Priority, **Blocked by** (dependencies that are not Done), **PR** (pull requests whose description has `Refs #<issue>`, with their state). GitHub's own "Linked pull requests" column stays empty: tested on this fork, neither a closing keyword (`Closes #N`) nor a linked branch links a pull request whose base is not the default branch (`master`), and our pull requests target `torch-backend`; hence the PR column. `board.py status` and `board.py done`
refresh the two text columns by themselves; after anything else that changes them (a PR opened or closed, an issue created) run `python tools/board.py refresh` (it writes only the cells that changed).

## The GraphQL budget

GitHub allows 5,000 GraphQL points an hour for the whole account, shared by every agent and the desktop app, and every `gh issue`, `gh pr` and `gh project` command spends them. **`gh project item-list` (about 100 points per call) and `gh project field-list` (about 150) are expensive**, because they fetch every field, label and assignee as nested connections. Never call them, and never loop over `gh` calls (an issue list or a pull-request list costs 1 to 3). `board.py` reads the table with one narrow query and reads the issues, the pull requests and the table once per command; the project's id and columns are kept for a day in `<git-dir>/board-cache.json` (`BOARD_NO_CACHE=1` ignores it). `python tools/board.py budget` shows what is left and when it resets (`gh api rate_limit` lags behind the real count; the `X-Ratelimit-*` headers of `GH_DEBUG=api` are correct); `--verbose` prints the cost of each table query. When a call fails with "API rate limit exceeded", stop using `gh` until the reset time the message gives.

## Keep the Definition of Done current

The issue body ends with a **Definition of Done** checklist. **Every time you finish a task that belongs to the issue, tick the boxes it satisfies**, with the
evidence; do not leave the ticking to the end.

```bash
python tools/board.py dod CLEAN1                                   # numbered checklist
python tools/board.py check CLEAN1 1 4 --note "command and result"  # tick boxes 1 and 4, comment the evidence
python tools/board.py na CLEAN1 7 "why it does not apply"           # a box that does not apply is marked, never left blank; a box that cannot be met because of a planning gap is a question to the user (CLAUDE.md), not an `na`
python tools/board.py uncheck CLEAN1 4                              # if a later change invalidates it
```

Only tick what you verified in this session. `done` refuses while a box is open (`--waive "reason"` records an explicit exception on the issue).

## Creating an issue

New issues are created only when the user says so. Write the body from the fork's template, `.github/ISSUE_TEMPLATE/milestone.md` (fork only, never on a `pr/U*` branch): the same sections as the
existing milestone issues plus **Out of scope** (binding) and **Decisions it depends on** (the decided outcomes inlined, so the issue needs no other reading), and the Definition of
Done with the merged-PR box (`Refs #<this issue>`). The first line names the issue numbers it depends on, the Context lines carry the estimate and where it runs, and the text is sanitised.
Then create it with one command (without `--yes` it prints what it would do and writes nothing):

```bash
python tools/board.py create QUAL4 "Short title" --body body.md --stage 0 --days 1.5 --priority P3 [--runs-on local] [--label L] [--no-mirror] [--yes]
```

It checks the body against the template, creates the issue with its stage and priority labels, replaces the `Refs` placeholder by the number, puts it on the board with its fields (Status, Stage, Runs on, Priority,
Upstream, Kind, Days), adds the native "blocked by" relations, adds the new issue to the `Blocks` line of the open issues it depends on (each body is read again just before it is written, because another agent may be ticking boxes in it)
and refreshes the table columns. Run `python tools/board.py lint` afterwards.

## Closing the issue

- **An issue is resolved only once its PR is merged into `torch-backend` on the fork** (the user merges). **Done** then requires the exit criterion to be met: `python tools/board.py done CLEAN1 "Verify output and numbers"` checks that no box is open, comments the
  evidence, sets Done and closes the issue. A gate is closed only when its pass criterion is met.
- **Dropping an issue is the owner's decision**, and only an issue that blocks nothing can be dropped (owner, 2026-10-15): `python tools/board.py drop CLEAN1 "reason"` refuses while an issue that depends on it
  (its "Depends on" line or a native "blocked by" relation) is neither Done nor Dropped; otherwise it comments the reason, closes the issue as not planned and sets Status to **Dropped**.
  Record on the later issues what they need from the dropped work (`finding --also`), and leave its branch unless the owner says otherwise. `lint` reports an issue that still depends on a Dropped one.

The branch, the worktree, the tidy-up pass, the PR text and the commit conventions are in `grace-torch-pr`.
