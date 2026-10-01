<!--
Fork-only template for pull requests into `torch-backend` on this fork. Generate a filled copy with `python tools/board.py pr-body <ID>`.
Do not use it for upstream (ICAMS) pull requests: those use the upstream template and are sent only with an explicit go for that unit.
Delete every comment before submitting. Rules for the text: see the "Writing the PR summary" section of the grace-torch-ticket skill.
-->

## Summary

<!-- 2 to 4 sentences, plain words: what is different after this PR and why it matters. Lead with the outcome, not the process. -->

Refs #<issue number>

<!-- `Refs`, never `Closes`: the issue is resolved by `python tools/board.py done` after the merge. -->

## What changed

<!-- One line per area or unit (file, function, class). Say what was deliberately NOT changed. State whether behaviour changes: "behaviour-preserving" or what changes. -->

-

## Evidence

<!-- Commands you ran in this task and their results, with numbers: counts against `baselines/`, coverage of every touched unit before and after, comparisons with the oracle, planted mutants caught. Say what you could not run (GPU, HPC, real weights) and why. -->

| Check | Command | Result |
|---|---|---|
| | | |

## Unexpected findings

<!-- Findings recorded on the issue (link the comments) or "none". -->

## Review focus

<!-- Where a reviewer should look hardest (the risky unit, the assertion that matters, the tolerance used), what could go wrong, and how to undo it (usually: revert this PR). -->

## Definition of Done

<!-- Copy the checklist of the issue; tick only what is ticked on the issue. -->
