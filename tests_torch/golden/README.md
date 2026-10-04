# Golden fixtures of the TensorFlow models

TensorFlow numbers that the torch twins are compared with: random weights, every instruction output, the index
tables, the results. Written by `tools/make_golden.py` and by nothing else; a fixture is never regenerated as a side
effect of another change, and a failing comparison is never fixed by regenerating one.

Two tiers (decision D8):

| | tiny anchors | faithful fixtures |
|---|---|---|
| where | `tests_torch/golden/` (committed) | `tests_torch/fixtures/` (the `.npz` files are git-ignored; manifests and yamls are committed) |
| model | the two 2L yamls with 4 elements and small widths | the two 2L yamls with 6 elements and the real widths |
| structures | `isolated_atom`, `dimer`, `self_image_cell`, `periodic_triple` | those and `fcc4`, `icosahedron_satellite`, `rattled_multi8` |
| size | each fixture under 1 MB, files together | each file under 25 MB |
| used for | CI without TensorFlow, the twin tests | the same tests on the real widths, run locally |

## What is in a fixture

For `omat_tiny_f64` (`<model>_<tier>_<dtype>`, model `omat` = `model_grace_2L_omat.yaml`, `large` =
`model_grace_2L_omat_large_base.yaml`, dtype `f64` or `f32` for the dtype of the parameters; the data are float64):

* `omat_tiny_f64.npz`: per structure `<case>/in/<key>` (the model input: `bond_vector`, `ind_i`, `ind_j`, `mu_i`, `mu_j`,
  `atomic_mu_i`, the batch maps), `<case>/ins/<instruction>` (the output of every instruction),
  `<case>/out_before/<instruction>` and `<case>/out_after/<instruction>` (the energy target before and after each output
  instruction: those overwrite `input_data[target.name]` in place, so a plain dump would keep only the last value),
  `<case>/res/{energy, atomic_energy, forces, virial, stress, pair_f}`; and `tables/<instruction>/<attribute>` (the
  `left_ind`, `right_ind`, `m_sum_ind`, `cg`, `w_tile_*`, `collect_*`, `norm_map` ... that each instruction builds; stored
  once under the instruction that owns it).
* `omat_tiny_f64.weights.npz`: every variable keyed `<instruction>/<attribute path>`, the place it has in the checkpoint
  (`A1/w_left`, `R/layers/0/w`, `Z/element_map_symbols`); never the TF variable name, which is not unique (the two
  `InvariantLayerRMSNorm.scale` of the large model are both `Variable:0`). Text is stored as a unicode array: no pickle
  anywhere (`np.load(..., allow_pickle=False)`).
* `omat_tiny_f64.json`: the manifest: versions of TensorFlow, numpy, pandas, ase and python, the seed, the git sha of the
  library and of the generator and whether either tree was modified, the module-level switch `_USE_GEMM_COUPLE`, the
  resolved constructor arguments of every instruction (in the order they run), the shape, dtype and trainability of every
  variable, the structures, the sha256 and size of each file.
* `yamls/omat_tiny.yaml`: the model, derived from the parent yaml (`python tools/make_golden.py yamls`); the fixtures
  and the yaml belong together.

Units: Angstrom, eV, eV/Angstrom; the stress is `-virial / volume` in ASE sign and Voigt order (xx, yy, zz, yz, xz, xy) and
zero for a structure without a cell; the virial is `sum_bonds r (x) f`, stored xx, yy, zz, xy, xz, yz.

## The weights

Every trainable floating variable is overwritten with seeded non-zero values (several layers start at zero:
`InvariantLayerRMSNorm.init`, the output norms), keyed by the variable's place in the checkpoint, so they do not depend on
the order the variables were created in: `N(0, 1/sqrt(shape[0]))` for matrices and `N(1, 0.1)` for vectors. The cutoff `rc`
and the element index tables are not trainable and keep their values. The float32 fixture has the float64 weights cast to
float32. No foundation weights are ever used.

## The structures and the species

The structures are those of `tests_torch/structures`. The model of a tier has only some elements, so a species of the
structure set that the model lacks is relabelled (geometry untouched, the model sees element indices only):

| tier | elements | relabelled |
|---|---|---|
| tiny | Cu, H, Mg, O | Au to H, Ca to Mg, Si to Cu |
| faithful | Ca, Cu, H, Mg, O, Si | Au to Ca |

The tiny set leaves out the structures with hundreds of bonds (the size budget: a dump of every instruction output grows
with the number of bonds), so unequal neighbour counts, partial periodicity and a many-element cell are covered by the
faithful tier only. `mineral` (2,680 bonds) and `slab` are in neither tier. The structures of the shared set that are small
enough for a tiny anchor are symmetric (`fcc4` and `self_image_cell` have an inversion centre on every atom or at the mid-point, so
their forces vanish to 1e-15; `dimer` is aperiodic), which would leave no tiny anchor with a force in a periodic cell:
`periodic_triple`, owned by the generator, is Cu, Mg and O at generic positions of a 4.4 x 4.7 x 5.0 A cell (66 bonds, forces
and stress not zero, atoms are neighbours of their own images).

## Option pairs

`<model>_<tier>_f64_dense` and `<model>_<tier>_f64_lm_first` are the same model and weights with `dense_nbr` or `lm_first`
switched on in every instruction that takes it; they hold no weights file (the manifest says `weights_from`) and store their own
inputs (the dense batch has its own layout). Their manifest records the comparison with the default layout:

* weights: the same keys, the same shapes, nothing to convert (`option_pair.weights`);
* results (energy, atomic energy, forces, virial, stress, pair forces): the same to the last digits; the largest absolute
  difference is 4e-21 (tiny) and 6e-17 (faithful), float64;
* `dense_nbr`: every node-level tensor and every table is equal to the default layout. A per-bond tensor (`bond_vector`, `ind_i`,
  `BondLength`, `R`, `Y`, `pair_f`, ...) is equal when every atom has the same number of neighbours (all the tiny structures);
  for `icosahedron_satellite` (12, 13 or 4 neighbours) the dense batch adds 18 padding bonds to the 164 real ones (each
  `ind_i = ind_j = 0`, with a bond vector beyond the cutoff and an exactly zero pair force; `n_neigh_real` is 164 in both layouts), so those
  tensors have 182 rows and the manifest lists them as `differs`. A twin of the
  dense layout has to reproduce that padding (issue DENSE1), and the energy, forces and stress are unaffected;
* `lm_first`: every tensor that has an angular axis is the default with that axis moved to the front (shape `[lm, ...]` for
  `[..., lm]`), and so is every Clebsch-Gordan table; all other tensors are equal.

`model_grace.yaml` is not in the set: its output instruction `MLPOut2ScalarTarget` has no `lm_first` transpose (recorded in
the oracle snapshot of SAFE2).

## Regenerating, verifying, comparing

```bash
python tools/make_golden.py write --tier tiny                      # tests_torch/golden/
python tools/make_golden.py write --tier faithful                  # tests_torch/fixtures/
python tools/make_golden.py verify tests_torch/golden              # reload every fixture in TF and compare with itself
python tools/make_golden.py compare DIR_A DIR_B                    # two runs, array by array
python tools/make_golden.py cells                                  # every option cell of the parents is still exercised
```

`write` uses one TensorFlow thread and deterministic ops, from a committed tree (the manifest records whether the tree was
modified; `tests_torch/test_golden_fixtures.py` fails on a modified one). The library imported is the one on `PYTHONPATH`,
which is how the generator runs from a worktree of the `pre-cleanup` tag: `PYTHONPATH=<worktree> python tools/make_golden.py write ...`
(in a worktree of this repository the editable install of the shared `.venv` points at the main checkout, so always set
`PYTHONPATH` to the tree whose library you mean; the first line printed is its location).

`verify` builds the model from the yaml next to the fixture with a seed other than the fixture's, assigns the stored weights, runs the
stored inputs and compares every stored array with the named float64 and float32 rows of `tools/oracle_snapshot.py`. Within one
process the result is exact; two processes can differ by one ulp in a few intermediates of the large model
(`baselines/README.md`).

## Reading a fixture without TensorFlow

```python
import json
import numpy as np

arrays = np.load("tests_torch/golden/omat_tiny_f64.npz", allow_pickle=False)
weights = np.load("tests_torch/golden/omat_tiny_f64.weights.npz", allow_pickle=False)
manifest = json.load(open("tests_torch/golden/omat_tiny_f64.json"))
energy = arrays["dimer/res/energy"]
a = arrays["self_image_cell/ins/A"]        # output of instruction A, shape [n_atoms, n_rad_max, (lmax+1)**2]
```

## Sizes (megabytes)

| fixture | arrays | weights | all files | arrays in the file |
|---|---|---|---|---|
| `omat_tiny_f64` | 0.37 | 0.14 | 0.53 | 325 |
| `omat_tiny_f32` | 0.23 | 0.07 | 0.33 | 325 |
| `large_tiny_f64` | 0.64 | 0.27 | 0.94 | 339 |
| `large_tiny_f32` | 0.38 | 0.14 | 0.55 | 339 |
| `omat_faithful_f64` | 11.58 | 8.66 | 20.26 | 425 |
| `omat_faithful_f32` | 5.82 | 4.20 | 10.05 | 425 |
| `large_faithful_f64` | 23.07 | 18.61 | 41.71 | 445 |
| `large_faithful_f32` | 11.41 | 9.03 | 20.47 | 445 |

The option pairs hold no weights: 0.40 to 0.67 MB (tiny) and 10.6 to 23.1 MB (faithful). The faithful target of 25 MB holds
for each file; the sum for `large_faithful_f64` is 41.7 MB because its float64 weights alone are 18.6 MB at the real widths.
