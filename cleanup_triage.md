# Triage of commented-out code (M0.2)

Status: **decisions recorded; awaiting sign-off in the pull request**. No source file was edited; this table is the input of M0.3 (removal) and, through the register below, of M0.5.

## How the table was made

- **Detectors.** (1) `ruff 0.16.7 check --isolated --select ERA001` over every tracked `.py` file except the fork-only `tools/` (607 lines flagged). (2) An AST census: a run of adjacent comment-only lines, or a single comment line, whose text after the leading `# ` parses with `ast.parse` into a statement that is more than a bare name or constant (adds 164 lines that ERA001 misses, mostly prose, labels and formulas).
- **Block.** A maximal run of adjacent comment-only lines flagged by either detector. Every ERA001 line belongs to exactly one block (771 lines in 258 blocks, 607 of them ERA001). Column `ERA` is `<ERA001 lines>/<lines>`; `0/n` means the AST census alone found it.
- **Id.** First 8 hex digits of SHA-256 over `<path>\n<the block's lines, each stripped, joined by \n>\n<n>`, where `n` counts earlier identical blocks in the same file (0 for almost all). It does not depend on line numbers, so it survives the shifts caused by earlier removals; the line range is only a locator for the current tree.
- **Kinds.** **A** dead code (superseded line, leftover, debug print); **B** archival data; **C** explanatory pseudo-code or detector false positive (formula, label, shape note, section banner); **D** disabled alternative with a reason nearby (a `TODO`, an inline reason or a label saying why); **E** disabled wiring of a documented or user-visible option, cross-checked against docs, tests and console scripts.
- **Actions.** `delete` for A and B, `keep` for C, D and E (the default of the issue). Kind A rows inside `compat/pace/` are `keep` because that package is excluded from every edit. A note that starts with `review:` marks a delete whose deleted text carries numerics or behaviour (not just a leftover); the owner looked at these rows and confirmed the deletion.

## Summary

| kind | action | blocks | lines |
|---|---|---:|---:|
| A | delete | 161 | 368 |
| A | keep | 22 | 26 |
| B | delete | 8 | 182 |
| C | keep | 37 | 37 |
| D | keep | 18 | 40 |
| E | keep | 12 | 118 |
| **total** | | **258** | **771** |

- `delete`: 169 blocks, 550 lines. `keep`: 89 blocks, 221 lines, of which 25 blocks in `compat/pace/`.
- 12 delete rows carry `review:`; 12 rows are class E (register below).

## Decisions

1. **Provenance comments.** Two tables are deleted (`2232b95c`, `7590fbab`, `tensorpotential/utils.py`). The live `DEFAULT_CUTOFF_1L` equals the deleted 1L table floored at 5.0 A for all 87 elements (checked), so M0.3 replaces `# equilibrium nn-dist + 3 A, crop(4,8).round(1)` with one line saying so. The live `DEFAULT_CUTOFF_2L` does not follow from the deleted 2L table (differences of 0.1 to 1.5 A; 58 of 87 exactly +0.5 A). **Decision (owner): the 2L table is kept in git history only.** M0.3 replaces the line above it (`nn-dist + 1.75 A`) with a comment that gives the commit before the removal, so the table can be recovered with `git show <commit>:tensorpotential/utils.py`; the same pointer may be added to the 1L comment. M0.3 fills in the hash.
2. **`review:` rows.** **Decision (owner): all 12 are deleted** like the other `delete` rows.
3. **Mixed blocks.** None is split: a block is deleted or kept whole. The only mixed case is `ba33c8d9` (an `exp(-i k.r)` note on the second line), kept whole as class D.
4. **compat/pace/** (rows with `keep`, kind A): untouched by the scope of the issue; M0.3 step 5 depends on decision D5.
5. **Test blocks of class E** (two commented-out integration tests with numeric references) need a decision, not a deletion: complete them (re-baseline on the HPC or locally) or retire them.

## Class E register (retire or complete)

Default action: keep. M0.5 copies this register into `cleanup_report.md`.

| id | location | what is disabled | evidence | decision to take |
|---|---|---|---|---|
| `5a978b0c` | `tensorpotential/cli/gracemaker.py:647-649` | LoRA finalisation before `--save-model` | docs/gracemaker/inputfile.md:74 'LORA (experimental, not supported)'; `enable_lora_adaptation` / `finalize_lora_update` live in functions/nn.py, instructions/base.py, instructions/compute.py, tpmodel.py | retire the LoRA path as a whole, or finish it |
| `8acc3b08` | `tensorpotential/cli/gracemaker.py:663-675` | `potential: lora` / `reduce_lora` input options (activation and reduction, with model re-save) | same docs lines 75-76 (the option examples); no test exercises it | same |
| `bfc0922a` | `tensorpotential/cli/gracemaker.py:745-747` | LoRA finalisation before the final save | same | same |
| `1c5d0a40` | `tensorpotential/functions/radial.py:99-101` | `GaussianRadialBasisFunction(normalized=...)` stores nothing and normalises nothing | functions/radial.py:72-129; reaches the class through the captured kwargs of `RadialBasis(basis_type='Gaussian', ...)`, hence it can be in saved model.yaml files | remove the argument (a model-format change) or restore the code |
| `ca55d8af` | `tensorpotential/functions/radial.py:112-118` | same, `build` | same | same |
| `caafca75` | `tensorpotential/functions/radial.py:124-129` | same, `compute_basis` | same | same |
| `cf89403b` | `tensorpotential/scripts/grace_preprocess.py:713` | `stress_units` in `grace_preprocess` | wired in cli/data.py:1093 (gracemaker) and documented in docs/gracemaker/inputfile.md:25; commented in scripts/grace_preprocess.py:713 with a TODO at :721 | wire it or document the eV/A3 limitation of `grace_preprocess` |
| `d1ac63dd` | `tensorpotential/scripts/grace_utils.py:491` | `--aux` selection of `grace_utils aux_model` (read of `args.aux`) | documented in docs/gracemaker/utilities.md:228 and :347 (`--aux energy_only parallel_2L`); `parser_aux_model.add_argument('--aux')` is commented (scripts/grace_utils.py:708) | restore the argument or correct the docs; today only `compute_energy` is added |
| `f0d17646` | `tensorpotential/scripts/grace_utils.py:504-505` | `compute_local` aux function (`ComputeStructureEnergyAndForcesAndVirial(local=True)`) | listed with `--aux` in the docs and in the argparse help text | same |
| `fb708616` | `tensorpotential/scripts/grace_utils.py:708-713` | the `--aux` argparse entry itself (default `parallel_2L energy_only compute_local`) | same | same |
| `94abfc6e` | `tests/test_integration_test.py:328-363` | test `test_MoNbTaW_MLP_switch_ef_lr_reduction_early_stop` (MLP model, ef/lr switch, early stop; 6 epochs of numeric references) | tests/MoNbTaW-MLP/input.yaml is tracked | re-baseline the references and re-enable, or retire with the input |
| `0bb0b394` | `tests/test_integration_test.py:501-537` | test `test_MoNbTaW_GRACE_2L_MP` (2L message-passing model; 3 epochs of numeric references) | tests/MoNbTaW-GRACE/input_2L_MP.yaml is tracked | same |

## Full table

Sorted by file and line. `n` is the number of lines in the block.

| id | location | ERA | kind | action | note |
|---|---|---:|:-:|---|---|
| `d6d36b4c` | `setup.py:4-5` | 2/2 | D | keep | disabled alternative (version read from pyproject.toml), labelled '(Optional)'; the packaging milestone decides |
| `659919fe` | `setup.py:7-10` | 4/4 | D | keep | disabled alternative (version read from pyproject.toml), labelled '(Optional)'; the packaging milestone decides |
| `a16d3d73` | `setup.py:15` | 1/1 | D | keep | disabled alternative (version read from pyproject.toml), labelled '(Optional)'; the packaging milestone decides |
| `48c555de` | `tensorpotential/calculator/asecalculator.py:918` | 1/1 | A | delete | kwarg commented out; the builder default applies |
| `1449288c` | `tensorpotential/calculator/foundation_models.py:139` | 1/1 | B | delete | old `dirname` values, no reader (the loader keys on MODEL_URL_KEY); the FS-OMAT and 2L-OMAT lines are copy-paste duplicates |
| `24af65e5` | `tensorpotential/calculator/foundation_models.py:150` | 1/1 | B | delete | old `dirname` values, no reader (the loader keys on MODEL_URL_KEY); the FS-OMAT and 2L-OMAT lines are copy-paste duplicates |
| `e8cda0e7` | `tensorpotential/calculator/foundation_models.py:161` | 1/1 | B | delete | old `dirname` values, no reader (the loader keys on MODEL_URL_KEY); the FS-OMAT and 2L-OMAT lines are copy-paste duplicates |
| `66336258` | `tensorpotential/calculator/foundation_models.py:175` | 1/1 | B | delete | old `dirname` values, no reader (the loader keys on MODEL_URL_KEY); the FS-OMAT and 2L-OMAT lines are copy-paste duplicates |
| `72f1a8a9` | `tensorpotential/calculator/foundation_models.py:185` | 1/1 | B | delete | old `dirname` values, no reader (the loader keys on MODEL_URL_KEY); the FS-OMAT and 2L-OMAT lines are copy-paste duplicates |
| `408b4999` | `tensorpotential/calculator/foundation_models.py:196` | 1/1 | B | delete | old `dirname` values, no reader (the loader keys on MODEL_URL_KEY); the FS-OMAT and 2L-OMAT lines are copy-paste duplicates |
| `0e3bbc19` | `tensorpotential/cli/data.py:257-259` | 1/3 | A | delete | fraction semantics of `limit_size`; docs describe an integer `train_size` only |
| `2453d793` | `tensorpotential/cli/data.py:280` | 1/1 | A | delete | shard shuffle disabled; no reason given |
| `584afd18` | `tensorpotential/cli/data.py:289` | 1/1 | A | delete | shard shuffle disabled; no reason given |
| `c25a8041` | `tensorpotential/cli/data.py:931` | 1/1 | A | delete | leftover assignment or debug print |
| `ba469085` | `tensorpotential/cli/data.py:1290` | 1/1 | A | delete | leftover assignment or debug print |
| `f99c922d` | `tensorpotential/cli/data.py:1315` | 1/1 | A | delete | refers to `tc.INPUT_USE_PER_SPECIE_N_NEI`, which does not exist; belongs with the per-specie neighbour statistics removed in databuilder.py |
| `097fd491` | `tensorpotential/cli/data.py:1441-1442` | 1/2 | A | delete | leftover assignment or debug print |
| `6328a9af` | `tensorpotential/cli/gracemaker.py:265-267` | 2/3 | A | delete | review: copying `shift` while loading a model was disabled while `scale` is live; no reason in the file, owner to confirm |
| `65606c17` | `tensorpotential/cli/gracemaker.py:337` | 0/1 | A | delete | stray `global` statement inside a prose comment |
| `5a978b0c` | `tensorpotential/cli/gracemaker.py:647-649` | 2/3 | E | keep | LoRA wiring: `lora` / `reduce_lora` are still described in docs/gracemaker/inputfile.md ('experimental, not supported') and the layer classes keep `enable_lora_adaptation` / `finalize_lora_update` |
| `383cf344` | `tensorpotential/cli/gracemaker.py:657` | 1/1 | D | keep | reason nearby: `TODO: first save with jit will convert function to JIT forever!` |
| `8acc3b08` | `tensorpotential/cli/gracemaker.py:663-675` | 8/13 | E | keep | LoRA wiring: `lora` / `reduce_lora` are still described in docs/gracemaker/inputfile.md ('experimental, not supported') and the layer classes keep `enable_lora_adaptation` / `finalize_lora_update` |
| `bfc0922a` | `tensorpotential/cli/gracemaker.py:745-747` | 2/3 | E | keep | LoRA wiring: `lora` / `reduce_lora` are still described in docs/gracemaker/inputfile.md ('experimental, not supported') and the layer classes keep `enable_lora_adaptation` / `finalize_lora_update` |
| `04c9b9b9` | `tensorpotential/cli/metrics.py:161` | 1/1 | A | delete | leftover assignment |
| `d4bf086f` | `tensorpotential/cli/metrics.py:179` | 1/1 | A | delete | leftover assignment |
| `bde07ffd` | `tensorpotential/cli/prepare.py:182` | 1/1 | A | delete | leftover assignment |
| `7a19be74` | `tensorpotential/cli/prepare.py:201` | 1/1 | A | delete | leftover assignment |
| `ffa9402d` | `tensorpotential/cli/prepare.py:326` | 1/1 | A | delete | replaced by the live `try: from tensorpotential.experimental import extra_losses` a few lines below |
| `9489ac5e` | `tensorpotential/cli/train.py:200` | 1/1 | D | keep | reason nearby: `TODO: decide the strategy for batches: add extra or remove tail?` |
| `afd4bd91` | `tensorpotential/cli/train.py:494` | 1/1 | A | delete | callback call disabled; no reason given |
| `14cb4fd3` | `tensorpotential/cli/train.py:941-944` | 1/4 | D | keep | reason nearby: `TODO: compute per-group metrics ...`; `compute_per_group_metrics` does not exist |
| `cc194d08` | `tensorpotential/cli/train_callbacks.py:224-226` | 2/3 | A | delete | review: superseded `alpha` formula, behaves differently when there is no warmup; no reason given |
| `604ce37f` | `tensorpotential/cli/wizard.py:286` | 0/1 | C | keep | section label inside a dataclass (AST false positive) |
| `f02084fc` | `tensorpotential/compat/pace/_version.py:39` | 0/1 | C | keep | versioneer comment; compat/pace is out of scope for every edit |
| `d3d06a4c` | `tensorpotential/compat/pace/_version.py:330` | 0/1 | C | keep | versioneer comment; compat/pace is out of scope for every edit |
| `d00b4bfe` | `tensorpotential/compat/pace/fit.py:111-113` | 1/3 | A | keep | dead code; kept only because compat/pace is out of scope for every edit (M0.3 step 5 / D5) |
| `93aaa467` | `tensorpotential/compat/pace/fit.py:135` | 1/1 | A | keep | dead code; kept only because compat/pace is out of scope for every edit (M0.3 step 5 / D5) |
| `77d742a7` | `tensorpotential/compat/pace/fit.py:188` | 1/1 | A | keep | dead code; kept only because compat/pace is out of scope for every edit (M0.3 step 5 / D5) |
| `32aa1d13` | `tensorpotential/compat/pace/fit.py:375` | 1/1 | A | keep | dead code; kept only because compat/pace is out of scope for every edit (M0.3 step 5 / D5) |
| `20ec2107` | `tensorpotential/compat/pace/fit.py:445` | 1/1 | A | keep | dead code; kept only because compat/pace is out of scope for every edit (M0.3 step 5 / D5) |
| `a12e9741` | `tensorpotential/compat/pace/fit.py:453` | 1/1 | A | keep | dead code; kept only because compat/pace is out of scope for every edit (M0.3 step 5 / D5) |
| `182a5dfa` | `tensorpotential/compat/pace/fit.py:484-485` | 0/2 | A | keep | dead code; kept only because compat/pace is out of scope for every edit (M0.3 step 5 / D5) |
| `21386057` | `tensorpotential/compat/pace/fit.py:496-497` | 2/2 | A | keep | dead code; kept only because compat/pace is out of scope for every edit (M0.3 step 5 / D5) |
| `37585d71` | `tensorpotential/compat/pace/fit.py:585` | 1/1 | A | keep | dead code; kept only because compat/pace is out of scope for every edit (M0.3 step 5 / D5) |
| `5bac6f3c` | `tensorpotential/compat/pace/fit.py:593` | 1/1 | A | keep | dead code; kept only because compat/pace is out of scope for every edit (M0.3 step 5 / D5) |
| `4051cbc5` | `tensorpotential/compat/pace/fit.py:599` | 1/1 | A | keep | dead code; kept only because compat/pace is out of scope for every edit (M0.3 step 5 / D5) |
| `b39edf07` | `tensorpotential/compat/pace/fit.py:677` | 1/1 | A | keep | dead code; kept only because compat/pace is out of scope for every edit (M0.3 step 5 / D5) |
| `938c8c2e` | `tensorpotential/compat/pace/fit.py:686` | 1/1 | A | keep | dead code; kept only because compat/pace is out of scope for every edit (M0.3 step 5 / D5) |
| `819d5451` | `tensorpotential/compat/pace/fit.py:693` | 1/1 | A | keep | dead code; kept only because compat/pace is out of scope for every edit (M0.3 step 5 / D5) |
| `4848a458` | `tensorpotential/compat/pace/fitmetrics.py:134` | 1/1 | A | keep | dead code; kept only because compat/pace is out of scope for every edit (M0.3 step 5 / D5) |
| `d5a5616c` | `tensorpotential/compat/pace/functions/spherical_harmonics.py:8` | 1/1 | A | keep | dead code; kept only because compat/pace is out of scope for every edit (M0.3 step 5 / D5) |
| `9926e83b` | `tensorpotential/compat/pace/functions/spherical_harmonics.py:76` | 1/1 | A | keep | dead code; kept only because compat/pace is out of scope for every edit (M0.3 step 5 / D5) |
| `b6f9d4b3` | `tensorpotential/compat/pace/functions/spherical_harmonics.py:214` | 1/1 | A | keep | dead code; kept only because compat/pace is out of scope for every edit (M0.3 step 5 / D5) |
| `a2720642` | `tensorpotential/compat/pace/potentials/ace.py:225` | 1/1 | C | keep | explains the normalisation constant; compat/pace is out of scope for every edit |
| `1efc4d83` | `tensorpotential/compat/pace/potentials/ace.py:969` | 1/1 | A | keep | dead code; kept only because compat/pace is out of scope for every edit (M0.3 step 5 / D5) |
| `2183a7c4` | `tensorpotential/compat/pace/potentials/ace.py:984` | 1/1 | A | keep | dead code; kept only because compat/pace is out of scope for every edit (M0.3 step 5 / D5) |
| `623b2456` | `tensorpotential/compat/pace/potentials/ace.py:1000` | 1/1 | A | keep | dead code; kept only because compat/pace is out of scope for every edit (M0.3 step 5 / D5) |
| `41d029d8` | `tensorpotential/compat/pace/utils/utilities.py:129` | 1/1 | A | keep | dead code; kept only because compat/pace is out of scope for every edit (M0.3 step 5 / D5) |
| `d8d3464c` | `tensorpotential/data/__init__.py:5-6` | 2/2 | A | delete | imports left out of the package namespace; both modules exist |
| `1006b0af` | `tensorpotential/data/databuilder.py:43` | 1/1 | A | delete | unused import (matscipy is the default neighbour list) |
| `42fac878` | `tensorpotential/data/databuilder.py:455-457` | 3/3 | A | delete | per-specie neighbour statistics (`nat_per_specie`, `total_nei_per_specie`) disabled in five places; delete together; debug print in 68daece5 |
| `262b08ae` | `tensorpotential/data/databuilder.py:459-462` | 3/4 | A | delete | per-specie neighbour statistics (`nat_per_specie`, `total_nei_per_specie`) disabled in five places; delete together; debug print in 68daece5 |
| `68daece5` | `tensorpotential/data/databuilder.py:466-469` | 2/4 | A | delete | per-specie neighbour statistics (`nat_per_specie`, `total_nei_per_specie`) disabled in five places; delete together; debug print in 68daece5 |
| `10a425bc` | `tensorpotential/data/databuilder.py:501-502` | 2/2 | A | delete | per-specie neighbour statistics (`nat_per_specie`, `total_nei_per_specie`) disabled in five places; delete together; debug print in 68daece5 |
| `2c6cfd74` | `tensorpotential/data/databuilder.py:551` | 0/1 | C | keep | names the keys handled by the lines below (AST false positive) |
| `c54fe126` | `tensorpotential/data/databuilder.py:586-596` | 8/11 | A | delete | per-specie neighbour statistics (`nat_per_specie`, `total_nei_per_specie`) disabled in five places; delete together; debug print in 68daece5 |
| `909db5d2` | `tensorpotential/data/process_df.py:362` | 1/1 | A | delete | leftover statement / log line |
| `405b7b4a` | `tensorpotential/data/process_df.py:442-477` | 25/36 | A | delete | whole function `compute_shifted_scaled_corrected_energy`, referenced nowhere else (git grep) |
| `27e92ec1` | `tensorpotential/data/weighting.py:339` | 1/1 | A | delete | leftover statement / log line |
| `0b38ed09` | `tensorpotential/export.py:102` | 0/1 | A | delete | leftover assertion / assignment |
| `879a4530` | `tensorpotential/export.py:196` | 1/1 | A | delete | leftover assertion / assignment |
| `70b935ee` | `tensorpotential/extra/gen_tensor/ewald.py:24` | 1/1 | C | keep | formula documenting the next line |
| `ffa7c31d` | `tensorpotential/extra/gen_tensor/ewald.py:35` | 1/1 | C | keep | formula documenting the next line |
| `ba33c8d9` | `tensorpotential/extra/gen_tensor/ewald.py:93-94` | 2/2 | D | keep | reason nearby: `This has to be unsorted_segment_sum with atoms_to_structure as index`; the commented lines are the per-structure variant |
| `67884d16` | `tensorpotential/extra/gen_tensor/ewald.py:97` | 1/1 | D | keep | reason nearby: `This has to be unsorted_segment_sum with atoms_to_structure as index`; the commented lines are the per-structure variant |
| `1528a190` | `tensorpotential/extra/gen_tensor/ewald.py:100-101` | 2/2 | D | keep | reason nearby: `This has to be unsorted_segment_sum with atoms_to_structure as index`; the commented lines are the per-structure variant |
| `a6c44370` | `tensorpotential/extra/gen_tensor/ewald.py:113-114` | 2/2 | A | delete | superseded alternative; the live lines follow |
| `600219b3` | `tensorpotential/extra/gen_tensor/ewald.py:122` | 1/1 | A | delete | superseded alternative; the live lines follow |
| `e2ce39da` | `tensorpotential/extra/gen_tensor/ewald.py:134` | 1/1 | A | delete | superseded alternative; the live lines follow |
| `3bac05a3` | `tensorpotential/extra/gen_tensor/loss.py:33` | 0/1 | C | keep | `TODO: Metric` marker, not code; no issue link (M0.5 counts TODO markers) |
| `be50b9ad` | `tensorpotential/extra/gen_tensor/model.py:118` | 1/1 | A | delete | kwarg commented out with an inline label |
| `d4bb17e8` | `tensorpotential/extra/gen_tensor/model.py:319` | 1/1 | A | delete | kwarg commented out with an inline label |
| `4d4fdec4` | `tensorpotential/extra/gen_tensor/wizard.py:20` | 0/1 | C | keep | section label inside a dataclass (AST false positive) |
| `ff0722c8` | `tensorpotential/extra/gen_tensor/wizard.py:36` | 0/1 | C | keep | section label inside a dataclass (AST false positive) |
| `e1abc2cf` | `tensorpotential/extra/gen_tensor/wizard.py:274` | 1/1 | C | keep | explains the table below (AST false positive) |
| `70f208e7` | `tensorpotential/functions/couplings.py:418` | 1/1 | C | keep | group label inside a coupling exclusion list (ERA001 false positive); `L = 0` ... `L = 5` and the AAA/AAAA level labels organise the entries |
| `b5b0b79b` | `tensorpotential/functions/couplings.py:751` | 1/1 | C | keep | group label inside a coupling exclusion list (ERA001 false positive); `L = 0` ... `L = 5` and the AAA/AAAA level labels organise the entries |
| `60c13060` | `tensorpotential/functions/couplings.py:1078` | 1/1 | C | keep | group label inside a coupling exclusion list (ERA001 false positive); `L = 0` ... `L = 5` and the AAA/AAAA level labels organise the entries |
| `a6c1c0ee` | `tensorpotential/functions/couplings.py:1415` | 1/1 | C | keep | group label inside a coupling exclusion list (ERA001 false positive); `L = 0` ... `L = 5` and the AAA/AAAA level labels organise the entries |
| `bea2f0a7` | `tensorpotential/functions/couplings.py:1690` | 1/1 | C | keep | group label inside a coupling exclusion list (ERA001 false positive); `L = 0` ... `L = 5` and the AAA/AAAA level labels organise the entries |
| `d7dbcf1a` | `tensorpotential/functions/couplings.py:1981` | 1/1 | C | keep | group label inside a coupling exclusion list (ERA001 false positive); `L = 0` ... `L = 5` and the AAA/AAAA level labels organise the entries |
| `147ab396` | `tensorpotential/functions/couplings.py:2230` | 0/1 | C | keep | prose of the provenance comment (AST false positive) |
| `7890d67d` | `tensorpotential/functions/couplings.py:2240` | 1/1 | C | keep | group label inside a coupling exclusion list (ERA001 false positive); `L = 0` ... `L = 5` and the AAA/AAAA level labels organise the entries |
| `020388b6` | `tensorpotential/functions/couplings.py:2243` | 1/1 | C | keep | group label inside a coupling exclusion list (ERA001 false positive); `L = 0` ... `L = 5` and the AAA/AAAA level labels organise the entries |
| `8e33b6de` | `tensorpotential/functions/couplings.py:2254` | 1/1 | C | keep | group label inside a coupling exclusion list (ERA001 false positive); `L = 0` ... `L = 5` and the AAA/AAAA level labels organise the entries |
| `b5faf829` | `tensorpotential/functions/couplings.py:2460` | 1/1 | A | delete | review: alternative CG normalisation (`sqrt(2L+1)`) next to the live `CG_normalization_dict`; numerics oracle |
| `88b86f36` | `tensorpotential/functions/couplings.py:2492` | 1/1 | A | delete | debug print |
| `11db0ac5` | `tensorpotential/functions/couplings.py:2559-2563` | 4/5 | A | delete | review: per-group norm column superseded by the sort below; numerics oracle |
| `4fe92b63` | `tensorpotential/functions/couplings.py:2585-2591` | 4/7 | A | delete | function `gen_CG_matrix` replaced by `gen_CG_matrix2` directly below |
| `8ed00499` | `tensorpotential/functions/nn.py:144` | 1/1 | A | delete | superseded or leftover line |
| `4c5be51f` | `tensorpotential/functions/nn.py:244` | 1/1 | C | keep | shape annotation |
| `979de542` | `tensorpotential/functions/nn.py:411` | 1/1 | A | delete | superseded or leftover line |
| `8175a2dd` | `tensorpotential/functions/radial.py:94` | 1/1 | A | delete | superseded formula, the live line follows |
| `1c5d0a40` | `tensorpotential/functions/radial.py:99-101` | 2/3 | E | keep | `GaussianRadialBasisFunction` still accepts `normalized` (it arrives through `RadialBasis(basis_type='Gaussian', **kwargs)`, whose kwargs are captured into saved model.yaml files) but the code that used it is commented out: the option is accepted and ignored |
| `ca55d8af` | `tensorpotential/functions/radial.py:112-118` | 3/7 | E | keep | `GaussianRadialBasisFunction` still accepts `normalized` (it arrives through `RadialBasis(basis_type='Gaussian', **kwargs)`, whose kwargs are captured into saved model.yaml files) but the code that used it is commented out: the option is accepted and ignored |
| `84f86f82` | `tensorpotential/functions/radial.py:122` | 1/1 | A | delete | superseded formula, the live line follows |
| `caafca75` | `tensorpotential/functions/radial.py:124-129` | 4/6 | E | keep | `GaussianRadialBasisFunction` still accepts `normalized` (it arrives through `RadialBasis(basis_type='Gaussian', **kwargs)`, whose kwargs are captured into saved model.yaml files) but the code that used it is commented out: the option is accepted and ignored |
| `90d73e3e` | `tensorpotential/functions/radial.py:171` | 1/1 | A | delete | superseded formula, the live line follows |
| `ab3e56f2` | `tensorpotential/functions/spherical_harmonics.py:180` | 1/1 | A | delete | superseded or leftover line |
| `9f0d593e` | `tensorpotential/functions/spherical_harmonics.py:258-270` | 10/13 | A | delete | review: loop version of the real-to-complex conversion, replaced by the live vectorised code above |
| `e61f8d0e` | `tensorpotential/functions/spherical_harmonics.py:279-280` | 1/2 | A | delete | superseded or leftover line |
| `f46f136d` | `tensorpotential/instructions/base.py:218` | 1/1 | A | delete | debug print / leftover line |
| `33163120` | `tensorpotential/instructions/base.py:471` | 1/1 | A | delete | debug print / leftover line |
| `0f1e972b` | `tensorpotential/instructions/base.py:516` | 1/1 | A | delete | debug print / leftover line |
| `e2177843` | `tensorpotential/instructions/compute.py:169-173` | 4/5 | A | delete | review: epsilon-regularised alternative of `BondLength` / `ScaledBondVector`; the live code does not use it |
| `745db008` | `tensorpotential/instructions/compute.py:189` | 1/1 | A | delete | superseded line, the live equivalent follows |
| `e4cf8987` | `tensorpotential/instructions/compute.py:209` | 1/1 | A | delete | review: epsilon-regularised alternative of `BondLength` / `ScaledBondVector`; the live code does not use it |
| `ba8ed46a` | `tensorpotential/instructions/compute.py:214` | 1/1 | A | delete | review: epsilon-regularised alternative of `BondLength` / `ScaledBondVector`; the live code does not use it |
| `64063127` | `tensorpotential/instructions/compute.py:220` | 1/1 | A | delete | review: epsilon-regularised alternative of `BondLength` / `ScaledBondVector`; the live code does not use it |
| `4d71837a` | `tensorpotential/instructions/compute.py:317-318` | 2/2 | A | delete | superseded line, the live equivalent follows |
| `04d6abb0` | `tensorpotential/instructions/compute.py:321` | 1/1 | A | delete | superseded line, the live equivalent follows |
| `552a19aa` | `tensorpotential/instructions/compute.py:627` | 1/1 | A | delete | superseded line, the live equivalent follows |
| `7ff08748` | `tensorpotential/instructions/compute.py:630` | 1/1 | A | delete | superseded line, the live equivalent follows |
| `71ad536d` | `tensorpotential/instructions/compute.py:657` | 1/1 | A | delete | superseded line, the live equivalent follows |
| `5ab35119` | `tensorpotential/instructions/compute.py:2067` | 1/1 | D | keep | reason nearby: `TODO: This must not be done`  |
| `5d042490` | `tensorpotential/instructions/compute.py:2075` | 1/1 | D | keep | reason nearby: `TODO: This must not be done`  |
| `b9faa0fc` | `tensorpotential/instructions/compute.py:2079-2080` | 2/2 | D | keep | reason nearby: `to be able to re-write it later, otherwise checkpoint will fail` |
| `6abbcea2` | `tensorpotential/instructions/compute.py:2228-2229` | 2/2 | D | keep | reason nearby: `to be able to re-write it later, otherwise checkpoint will fail` |
| `9a5a403b` | `tensorpotential/instructions/compute.py:3087` | 1/1 | A | delete | debug print / alternative label |
| `ec64d0f3` | `tensorpotential/instructions/compute.py:3170` | 1/1 | A | delete | debug print / alternative label |
| `2535cf7e` | `tensorpotential/instructions/compute.py:3448` | 1/1 | A | delete | debug print / alternative label |
| `d252dae5` | `tensorpotential/instructions/compute.py:3472-3476` | 5/5 | A | delete | remnants of the earlier reducing implementation of `CollectInvarBasis` (parameters, variables, rms normalisation); the class is now a plain collector. Delete as one group |
| `86992da0` | `tensorpotential/instructions/compute.py:3487-3488` | 2/2 | A | delete | remnants of the earlier reducing implementation of `CollectInvarBasis` (parameters, variables, rms normalisation); the class is now a plain collector. Delete as one group |
| `b800eb81` | `tensorpotential/instructions/compute.py:3492-3493` | 2/2 | A | delete | remnants of the earlier reducing implementation of `CollectInvarBasis` (parameters, variables, rms normalisation); the class is now a plain collector. Delete as one group |
| `365b1337` | `tensorpotential/instructions/compute.py:3496-3504` | 2/9 | A | delete | remnants of the earlier reducing implementation of `CollectInvarBasis` (parameters, variables, rms normalisation); the class is now a plain collector. Delete as one group |
| `7c1fac65` | `tensorpotential/instructions/compute.py:3547-3588` | 24/42 | A | delete | remnants of the earlier reducing implementation of `CollectInvarBasis` (parameters, variables, rms normalisation); the class is now a plain collector. Delete as one group |
| `7f0bb85b` | `tensorpotential/instructions/compute.py:3610-3611` | 2/2 | A | delete | remnants of the earlier reducing implementation of `CollectInvarBasis` (parameters, variables, rms normalisation); the class is now a plain collector. Delete as one group |
| `1e7c446d` | `tensorpotential/instructions/compute.py:3615-3616` | 2/2 | A | delete | remnants of the earlier reducing implementation of `CollectInvarBasis` (parameters, variables, rms normalisation); the class is now a plain collector. Delete as one group |
| `9cae9ef0` | `tensorpotential/instructions/compute.py:3618-3626` | 7/9 | A | delete | remnants of the earlier reducing implementation of `CollectInvarBasis` (parameters, variables, rms normalisation); the class is now a plain collector. Delete as one group |
| `bc8901e7` | `tensorpotential/instructions/compute.py:3703-3707` | 3/5 | A | delete | remnants of the earlier reducing implementation of `CollectInvarBasis` (parameters, variables, rms normalisation); the class is now a plain collector. Delete as one group |
| `a4b75a57` | `tensorpotential/instructions/compute.py:3738` | 1/1 | C | keep | explains the guard above it (the example named in the issue) |
| `c2c10548` | `tensorpotential/instructions/compute.py:4559` | 1/1 | C | keep | shape annotation |
| `65978d29` | `tensorpotential/instructions/compute.py:4710` | 1/1 | A | delete | superseded line |
| `e094b1e3` | `tensorpotential/instructions/output.py:194` | 1/1 | A | delete | alternative activation, the live code follows |
| `b4e5a9dd` | `tensorpotential/instructions/output.py:276` | 1/1 | A | delete | superseded initialisation / call |
| `89027b51` | `tensorpotential/instructions/output.py:301` | 1/1 | A | delete | superseded initialisation / call |
| `a78cc630` | `tensorpotential/instructions/output.py:377` | 1/1 | A | delete | superseded initialisation / call |
| `0cf2916e` | `tensorpotential/loss.py:21-30` | 9/10 | D | keep | labelled alternative: `# Or stricktly upper part` (upper-triangle mask variant of the live composition loss) |
| `5fca7c97` | `tensorpotential/loss.py:156-160` | 3/5 | A | delete | method `get_corresponding_metrics`, referenced nowhere else |
| `e1f1caa9` | `tensorpotential/loss.py:1170` | 0/1 | C | keep | column order of the Voigt virial |
| `dc4745aa` | `tensorpotential/metrics.py:129` | 1/1 | A | delete | leftover assignment |
| `0e553a47` | `tensorpotential/potentials/presets.py:215` | 1/1 | D | keep | reason inline: `not yet supported for export` |
| `9cd81993` | `tensorpotential/scripts/_kokkos_export.py:1530` | 1/1 | C | keep | formula documenting the code below |
| `df1fff73` | `tensorpotential/scripts/grace_predict.py:103` | 1/1 | A | delete | kwarg equal to the `TPCalculator` default (3) |
| `cf89403b` | `tensorpotential/scripts/grace_preprocess.py:713` | 1/1 | E | keep | `stress_units` is documented for the gracemaker input file (docs/gracemaker/inputfile.md) and wired in cli/data.py:1093, but `grace_preprocess` leaves it commented out (TODO at grace_preprocess.py:721), so it always preprocesses with the default eV/A3 |
| `d4146a02` | `tensorpotential/scripts/grace_preprocess.py:725` | 0/1 | D | keep | marker: a TODO line with code (`user_cutoff_dict`); no issue link |
| `d1ac63dd` | `tensorpotential/scripts/grace_utils.py:491` | 1/1 | E | keep | `--aux` of `grace_utils aux_model` is still described in docs/gracemaker/utilities.md (default: all three functions) but the argument is commented out, so only `compute_energy` is added; `compute_local` is disabled with it |
| `f0d17646` | `tensorpotential/scripts/grace_utils.py:504-505` | 2/2 | E | keep | `--aux` of `grace_utils aux_model` is still described in docs/gracemaker/utilities.md (default: all three functions) but the argument is commented out, so only `compute_energy` is added; `compute_local` is disabled with it |
| `fb708616` | `tensorpotential/scripts/grace_utils.py:708-713` | 4/6 | E | keep | `--aux` of `grace_utils aux_model` is still described in docs/gracemaker/utilities.md (default: all three functions) but the argument is commented out, so only `compute_energy` is added; `compute_local` is disabled with it |
| `15be09a1` | `tensorpotential/tensorpot.py:121` | 1/1 | D | keep | reason nearby: bare `TODO:` above it |
| `3348c336` | `tensorpotential/tensorpot.py:140-141` | 2/2 | A | delete | superseded or leftover |
| `d0f05304` | `tensorpotential/tensorpot.py:143-146` | 4/4 | A | delete | superseded or leftover |
| `94493242` | `tensorpotential/tensorpot.py:613` | 1/1 | A | delete | superseded or leftover |
| `05a2c6d8` | `tensorpotential/tensorpot.py:653` | 1/1 | A | delete | superseded or leftover |
| `9202913a` | `tensorpotential/tpmodel.py:359` | 1/1 | A | delete | superseded or debug line |
| `7960a326` | `tensorpotential/tpmodel.py:411` | 1/1 | A | delete | superseded or debug line |
| `7b34b4e6` | `tensorpotential/tpmodel.py:654` | 1/1 | A | delete | superseded or debug line |
| `e44e2f75` | `tensorpotential/tpmodel.py:845` | 1/1 | A | delete | superseded or debug line |
| `ae5dd4ae` | `tensorpotential/tpmodel.py:1027` | 1/1 | A | delete | superseded or debug line |
| `b30cb12f` | `tensorpotential/tpmodel.py:1130-1131` | 1/2 | A | delete | superseded or debug line |
| `6f78cf59` | `tensorpotential/tpmodel.py:1151-1153` | 1/3 | D | keep | marker: `TODO: make all non-trainable ??` inside the LoRA path; no issue link |
| `15e45e20` | `tensorpotential/uq/cli/build/thresholds.py:44` | 1/1 | C | keep | formula documenting the threshold rule (comment above the code) |
| `2232b95c` | `tensorpotential/utils.py:387-474` | 88/88 | B | delete | archival cutoff tables (87 elements each); the live 1L table equals the deleted one floored at 5.0 A for all 87 elements, the live 2L table does not follow from the deleted one (diffs 0.1 to 1.5 A). Keep one provenance line per table (see report) |
| `7590fbab` | `tensorpotential/utils.py:566-653` | 88/88 | B | delete | archival cutoff tables (87 elements each); the live 1L table equals the deleted one floored at 5.0 A for all 87 elements, the live 2L table does not follow from the deleted one (diffs 0.1 to 1.5 A). Keep one provenance line per table (see report) |
| `fc14b3d8` | `tensorpotential/utils.py:968` | 1/1 | A | delete | leftover line |
| `8bff3f6b` | `tests/MoNbTaW-CUSTOM/model.py:102` | 1/1 | A | delete | leftover line / debug print in a test helper |
| `37f83ecd` | `tests/data_distrib/check_stats.py:64-65` | 2/2 | A | delete | leftover line / debug print in a test helper |
| `a3b64872` | `tests/test_calculator.py:296` | 1/1 | A | delete | debug print, plot, or superseded set-up line in a test |
| `812ca93b` | `tests/test_calculator.py:308` | 1/1 | A | delete | debug print, plot, or superseded set-up line in a test |
| `dc8726e7` | `tests/test_calculator.py:337-340` | 1/4 | A | delete | debug print, plot, or superseded set-up line in a test |
| `4fc0bdff` | `tests/test_calculator.py:348` | 1/1 | A | delete | debug print, plot, or superseded set-up line in a test |
| `cda2b8c5` | `tests/test_calculator.py:356` | 1/1 | A | delete | debug print, plot, or superseded set-up line in a test |
| `9023fbd9` | `tests/test_calculator.py:364` | 1/1 | A | delete | debug print, plot, or superseded set-up line in a test |
| `a17a893b` | `tests/test_calculator.py:379` | 1/1 | A | delete | debug print, plot, or superseded set-up line in a test |
| `253a9f37` | `tests/test_calculator.py:388` | 1/1 | A | delete | debug print, plot, or superseded set-up line in a test |
| `bc02f399` | `tests/test_calculator.py:407` | 1/1 | A | delete | debug print, plot, or superseded set-up line in a test |
| `24935289` | `tests/test_calculator.py:416` | 1/1 | A | delete | debug print, plot, or superseded set-up line in a test |
| `5e6a7bd5` | `tests/test_calculator.py:429-430` | 2/2 | A | delete | debug print, plot, or superseded set-up line in a test |
| `3a5fe0af` | `tests/test_calculator.py:442-443` | 2/2 | A | delete | debug print, plot, or superseded set-up line in a test |
| `1cb59d09` | `tests/test_calculator.py:533` | 1/1 | C | keep | element label of the assertion group |
| `7476cd35` | `tests/test_calculator.py:537` | 1/1 | C | keep | element label of the assertion group |
| `50bfa473` | `tests/test_calculator.py:589` | 1/1 | A | delete | debug print, plot, or superseded set-up line in a test |
| `d4ad6c44` | `tests/test_calculator.py:602` | 1/1 | A | delete | debug print, plot, or superseded set-up line in a test |
| `57630f9f` | `tests/test_calculator.py:647` | 1/1 | A | delete | debug print, plot, or superseded set-up line in a test |
| `5b5c1360` | `tests/test_coupling_parity_bug.py:396` | 0/1 | C | keep | step label |
| `744e6c1d` | `tests/test_databuilder.py:61` | 1/1 | A | delete | debug print |
| `0976f947` | `tests/test_databuilder.py:77-79` | 2/3 | A | delete | review: disabled assertion `batch_tot_nat == 199`; the sibling test asserts 199 live, check it still holds before deleting |
| `ad49dc5b` | `tests/test_databuilder.py:112` | 1/1 | A | delete | debug print |
| `1b4bf380` | `tests/test_databuilder.py:226` | 1/1 | A | delete | debug print |
| `656933b2` | `tests/test_databuilder.py:242-243` | 1/2 | A | delete | superseded assertion (185), the live one asserts 199 |
| `e983006b` | `tests/test_distrib.py:65-66` | 2/2 | A | delete | debug redirect / disabled clean-up of the output directory |
| `2bf2f9ca` | `tests/test_distrib.py:98` | 1/1 | A | delete | debug redirect / disabled clean-up of the output directory |
| `06d23337` | `tests/test_export.py:46` | 1/1 | C | keep | explains the assertion or section below, or a section banner |
| `cc3182bb` | `tests/test_fm_shift_auto.py:161` | 1/1 | C | keep | explains the assertion or section below, or a section banner |
| `49369653` | `tests/test_fm_shift_auto.py:199` | 1/1 | C | keep | explains the assertion or section below, or a section banner |
| `7a5abedb` | `tests/test_instructions.py:64` | 1/1 | A | delete | debug print, plot, or superseded set-up line in a test |
| `0fb01235` | `tests/test_instructions.py:82-83` | 2/2 | A | delete | debug print, plot, or superseded set-up line in a test |
| `63c8ba2d` | `tests/test_instructions.py:88` | 1/1 | A | delete | debug print, plot, or superseded set-up line in a test |
| `4c28da12` | `tests/test_instructions.py:97-98` | 2/2 | A | delete | debug print, plot, or superseded set-up line in a test |
| `30112fbd` | `tests/test_instructions.py:105` | 1/1 | A | delete | debug print, plot, or superseded set-up line in a test |
| `ef725969` | `tests/test_instructions.py:115-116` | 2/2 | A | delete | debug print, plot, or superseded set-up line in a test |
| `e934c135` | `tests/test_instructions.py:145` | 1/1 | A | delete | debug print, plot, or superseded set-up line in a test |
| `36f09989` | `tests/test_instructions.py:151` | 1/1 | A | delete | debug print, plot, or superseded set-up line in a test |
| `5d3d5478` | `tests/test_instructions.py:173` | 1/1 | A | delete | debug print, plot, or superseded set-up line in a test |
| `88f69028` | `tests/test_instructions.py:175` | 1/1 | A | delete | review: disabled shape assertion; the live line indexes columns [0, 1, 4] so the asserted shape is stale |
| `a48a604e` | `tests/test_instructions.py:267-268` | 2/2 | A | delete | debug print, plot, or superseded set-up line in a test |
| `c2838926` | `tests/test_instructions.py:346` | 1/1 | A | delete | debug print, plot, or superseded set-up line in a test |
| `c56819e0` | `tests/test_instructions.py:362` | 1/1 | A | delete | debug print, plot, or superseded set-up line in a test |
| `42f07ccc` | `tests/test_instructions.py:414` | 1/1 | A | delete | debug print, plot, or superseded set-up line in a test |
| `e9487b41` | `tests/test_instructions.py:436` | 1/1 | A | delete | debug print, plot, or superseded set-up line in a test |
| `5efb000a` | `tests/test_instructions.py:440-442` | 3/3 | A | delete | debug print, plot, or superseded set-up line in a test |
| `8d2295c2` | `tests/test_instructions.py:468` | 1/1 | A | delete | debug print, plot, or superseded set-up line in a test |
| `1cd41ca8` | `tests/test_instructions.py:542` | 1/1 | A | delete | debug print, plot, or superseded set-up line in a test |
| `01fe1b16` | `tests/test_instructions.py:544-547` | 4/4 | A | delete | debug print, plot, or superseded set-up line in a test |
| `1a46d5e0` | `tests/test_instructions.py:592` | 1/1 | A | delete | debug print, plot, or superseded set-up line in a test |
| `a940bbd5` | `tests/test_instructions.py:594-597` | 4/4 | A | delete | debug print, plot, or superseded set-up line in a test |
| `03f5baa6` | `tests/test_instructions.py:624` | 1/1 | A | delete | debug print, plot, or superseded set-up line in a test |
| `8ac6a2b9` | `tests/test_instructions.py:628` | 1/1 | A | delete | debug print, plot, or superseded set-up line in a test |
| `7080ff85` | `tests/test_instructions.py:734` | 1/1 | A | delete | review: disabled rotation of the input vectors; the live assertion above checks the rotation relation |
| `b5fad0b0` | `tests/test_instructions.py:803` | 1/1 | A | delete | debug print, plot, or superseded set-up line in a test |
| `94f94f61` | `tests/test_instructions.py:823` | 1/1 | A | delete | debug print, plot, or superseded set-up line in a test |
| `b15cd4e6` | `tests/test_instructions.py:1075` | 1/1 | A | delete | debug print, plot, or superseded set-up line in a test |
| `b8f7bd10` | `tests/test_instructions.py:1079-1080` | 2/2 | A | delete | debug print, plot, or superseded set-up line in a test |
| `02863137` | `tests/test_instructions.py:1099-1100` | 1/2 | A | delete | debug print, plot, or superseded set-up line in a test |
| `2eefff38` | `tests/test_instructions.py:1410` | 1/1 | A | delete | debug print, plot, or superseded set-up line in a test |
| `e735e824` | `tests/test_instructions.py:1430` | 1/1 | A | delete | debug print, plot, or superseded set-up line in a test |
| `68b38a6e` | `tests/test_instructions.py:1446-1447` | 2/2 | A | delete | debug print, plot, or superseded set-up line in a test |
| `3b01f1dd` | `tests/test_instructions.py:1449-1450` | 2/2 | A | delete | debug print, plot, or superseded set-up line in a test |
| `7f204941` | `tests/test_instructions.py:1458-1459` | 2/2 | A | delete | debug print, plot, or superseded set-up line in a test |
| `f0c933da` | `tests/test_instructions.py:1464-1465` | 2/2 | A | delete | debug print, plot, or superseded set-up line in a test |
| `861a1793` | `tests/test_instructions.py:1468-1469` | 2/2 | A | delete | debug print, plot, or superseded set-up line in a test |
| `890d7f38` | `tests/test_instructions.py:1476-1477` | 2/2 | A | delete | debug print, plot, or superseded set-up line in a test |
| `df1275dd` | `tests/test_instructions.py:1485-1487` | 3/3 | A | delete | debug print, plot, or superseded set-up line in a test |
| `2773350b` | `tests/test_instructions.py:1497-1498` | 2/2 | A | delete | debug print, plot, or superseded set-up line in a test |
| `d471cc58` | `tests/test_instructions.py:1508-1509` | 2/2 | A | delete | debug print, plot, or superseded set-up line in a test |
| `7bc5421f` | `tests/test_instructions.py:1519-1520` | 2/2 | A | delete | debug print, plot, or superseded set-up line in a test |
| `0da88933` | `tests/test_instructions.py:1593` | 1/1 | A | delete | debug print, plot, or superseded set-up line in a test |
| `94abfc6e` | `tests/test_integration_test.py:328-363` | 10/36 | E | keep | commented-out integration tests with numeric references (`test_MoNbTaW_MLP_switch_ef_lr_reduction_early_stop` on tests/MoNbTaW-MLP/input.yaml, `test_MoNbTaW_GRACE_2L_MP` on tests/MoNbTaW-GRACE/input_2L_MP.yaml); both input files are tracked; complete (re-baseline the references) or retire |
| `0bb0b394` | `tests/test_integration_test.py:501-537` | 11/37 | E | keep | commented-out integration tests with numeric references (`test_MoNbTaW_MLP_switch_ef_lr_reduction_early_stop` on tests/MoNbTaW-MLP/input.yaml, `test_MoNbTaW_GRACE_2L_MP` on tests/MoNbTaW-GRACE/input_2L_MP.yaml); both input files are tracked; complete (re-baseline the references) or retire |
| `508db232` | `tests/test_loss_piecewise_linear.py:320` | 1/1 | C | keep | explains the assertion or section below, or a section banner |
| `308e13c5` | `tests/test_streaming_pipeline.py:378` | 0/1 | C | keep | explains the assertion or section below, or a section banner |
| `838e3ddc` | `tests/test_tp_model.py:20-22` | 1/3 | A | delete | unused import / kwarg / checkpoint call |
| `a421b5fd` | `tests/test_tp_model.py:484` | 1/1 | A | delete | unused import / kwarg / checkpoint call |
| `2fd026d3` | `tests/test_tp_model.py:487` | 1/1 | A | delete | unused import / kwarg / checkpoint call |
| `75e15173` | `tests/test_tp_model.py:510-511` | 2/2 | A | delete | unused import / kwarg / checkpoint call |
| `0389049a` | `tests/test_tp_model.py:515` | 1/1 | A | delete | unused import / kwarg / checkpoint call |
| `f3cd304d` | `tests/test_uq_common.py:87` | 0/1 | C | keep | explains the assertion or section below, or a section banner |
| `f66ee423` | `tests/test_uq_thresholds.py:30` | 1/1 | C | keep | explains the assertion or section below, or a section banner |
