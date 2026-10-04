---
name: grace-torch
description: Shared working context for the GRACE PyTorch backend and the Stage 0 clean-up that precedes it. Use whenever a task touches the torch backend (`tensorpotential/torch_backend/`, `tensorpotential/core/`, `tests_torch/`), the TF-to-torch conversion, a milestone or gate issue of the fork's board, or a question about how the new code relates to the TensorFlow code. Load this before writing code for a milestone.
---

# grace-torch

The goal: load a TF-trained GRACE model (`model.yaml` plus checkpoint) in PyTorch and reproduce its energy, forces and stress, then serve it
through torch-sim. The TensorFlow code is the numerical oracle; the torch code is a twin of it, not a port line by line.

| Companion skill | Use it for |
|---|---|
| `grace-torch-ticket` | Taking a board issue from Todo to Done: the protocol, the board helper, findings, the PR |
| `grace-torch-tests` | The 90% coverage rule, the two test layers, planted mutants, parallel runs |
| `grace-torch-goldens` | The committed baselines and golden fixtures, tolerances, debugging a mismatch |
| `grace-torch-numerics` | Units, sign conventions, dtypes, derivatives, equivariance checks |

## Where the work comes from

Work is tracked as issues on the fork (`akwarii/grace-tensorpotential`) and on the project board
<https://github.com/users/akwarii/projects/2>. **The issue is the source of the work and of the exit criterion**; it names its dependencies
(`Depends on`), what it unblocks (`Blocks`), its estimate, and a Definition of Done. The comments of an issue and of its dependencies hold what earlier agents
found; read them. Fields: `Status` (Todo, In Progress, PR Open, Done), `Stage` (0-8), `Priority` (P0-P3, from a score), `Runs on` (local, HPC, GPU, user).
The decisions taken so far (D1-D19: packaging, fixtures, periodic-boundary policy, upstream policy, ...) are in the `Decisions` issue
(`python tools/board.py context Decisions`); a ticked item is binding. The evidence behind the milestones (measurements, the pandas 2-vs-3 check, the upstream-unit list, the cuEquivariance study, the MACE and fairchem survey) is in issues labelled `reference`, titled "Appendix A" to "Appendix L" and cited that way by the milestone issues (`gh issue list --label reference`). If an issue contradicts a decision, or the code contradicts an issue, report it as a finding on the issue.

## The rules that keep the work convergent

- **R1 - The TF classes are the oracle.** They change only when a milestone says so, and then the change is behaviour-preserving, covered at 90%,
  and compared with `baselines/`. The torch twins own their copy of the index-table ("plan") logic, guarded by table-equality tests against
  tables dumped from TF. Do not import TF from torch code.
- **R2 - Layering and import direction.** `core/` (pure numpy, no TF, no torch) -> plans (pure numpy tables) -> ops (torch maths) -> instructions ->
  model -> torch-sim wrapper. `weights` and `convert` are separate; the TF-side extractor lives outside `torch_backend/` (under `scripts/`), so
  `torch_backend/` and `core/` have a zero-TF rule without exceptions. An import-linter contract and a `sys.meta_path` blocker test enforce it.
- **R3 - Unsupported means an error.** A class, or a class option value, without a twin stops the load with an actionable message naming it.
  Nothing is skipped or approximated silently. Every option of the in-scope classes (including `dense_nbr` and `lm_first`) is ported; an option
  that changes layout and not the result is verified by a TF fixture pair (option off and on, same weights).
- **R4 - Model state is never a backend property.** A kernel backend can change speed, never tables, parameter counts, weight layout or results.
- **R5 - One place for each decision.** One module decides dtypes and casts, one table holds tolerances, one module holds conventions, one
  `errors` module holds the exception types. Duplicating TF plan logic in the twins is the single recorded exception to DRY.
- **R6 - Weights are data.** Weights travel as `npz` (pickle disabled) plus JSON metadata keyed by `<instruction>/<attribute>`; no pickle, no `eval`.
  Random-weight fixtures only locally; foundation-model weights only on the HPC.
- **R7 - Measure before optimising.** A performance idea is a hypothesis until a benchmark says otherwise, and GPU results need GPU access
  that must be asked for.
- **R8 - Public surface.** The fork is public: everything you post is sanitised (rules and the `board.py sanitise` command in `CLAUDE.md`).
- **R9 - The user and the issue take precedence** over the general rules here and in `CLAUDE.md`. An issue whose job is to change legacy code (clean-up, quality,
  deduplication, packaging) may do what the general text discourages, within its scope and with the safeguards it names.

## Layout (destination)

```
tensorpotential/core/            TF-free shared code: neighbour-list contract, process_cutoff_dict, graph schema, conventions
tensorpotential/torch_backend/   spec, registry, plans, ops, instructions/, model, weights, torchsim
tensorpotential/scripts/         TF-side extractor (variables -> npz), converter entry points
tests_torch/                     conftest without TF import; golden/ (tiny committed anchors), fixtures/ (local, git-ignored), oracle/
tools/                           baselines tools, coverage and clone checks
```

Check the tree before relying on a path: pieces appear milestone by milestone, and the board is the record of what exists.
