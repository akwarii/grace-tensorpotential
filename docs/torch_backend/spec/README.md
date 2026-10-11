# Class spec sheets (SPEC3)

One sheet per TensorFlow instruction class that the PyTorch twins must reproduce: the 17 classes of the two
2L foundation `model.yaml` files (`tests/model_grace_2L_omat.yaml`, `tests/model_grace_2L_omat_large_base.yaml`).
A sheet states what the TF class does, precisely enough to write the twin without reading the TF code again,
and says where in the TF code each statement comes from. The TF classes are the oracle (rule R1); where a sheet
and the code disagree, the code wins and the sheet is wrong.

## Pinned source

Every citation refers to `torch-backend` at commit `683ebd0` (TensorFlow 2.20.0 with legacy Keras 2; this branch was first
written on `ae456fe` and re-pinned after merging `683ebd0`, which added one import line to most of the cited modules). A citation reads
`instructions/compute.py:187-194`: a path **relative to `tensorpotential/`**, then the line range. Tests are
cited as `tests/test_compute.py:399` (relative to the repository root). Inside a sheet the directory is dropped
from the files that occur often, and the bare name stands for: `compute.py`, `base.py`, `output.py`
(`instructions/`); `radial.py`, `spherical_harmonics.py`, `nn.py`, `lora.py`, `couplings.py` (`functions/`);
`tpmodel.py`, `constants.py` (`tensorpotential/`). When the source moves, the line
numbers move with it; the sheets are re-pinned by the issue that changes the unit.

## Families and sheets

| Family | Sheets | Status |
|---|---|---|
| Geometry and radial | [BondLength](BondLength.md), [ScaledBondVector](ScaledBondVector.md), [RadialBasis](RadialBasis.md), [SphericalHarmonic](SphericalHarmonic.md), [MLPRadialFunction](MLPRadialFunction.md), [MLPRadialFunction_v2](MLPRadialFunction_v2.md) | draft; proposals accepted, review pending |
| Embedding and single-particle basis | [ScalarChemicalEmbedding](ScalarChemicalEmbedding.md), [SingleParticleBasisFunctionScalarInd](SingleParticleBasisFunctionScalarInd.md), [SingleParticleBasisFunctionEquivariantInd](SingleParticleBasisFunctionEquivariantInd.md) | draft, awaiting review |
| Product and reduce | [ProductFunction](ProductFunction.md), [FCRight2Left](FCRight2Left.md), [FunctionReduceN](FunctionReduceN.md) | draft, awaiting review |
| Norm and output | [InvariantLayerRMSNorm](InvariantLayerRMSNorm.md), [CreateOutputTarget](CreateOutputTarget.md), [LinMLPOut2ScalarTarget](LinMLPOut2ScalarTarget.md), [ConstantScaleShiftTarget](ConstantScaleShiftTarget.md), [TrainableShiftTarget](TrainableShiftTarget.md) | draft, awaiting review |

## Layout of a sheet

A header (class, source, base classes, families of the shipped yamls that use it, what it needs from the data) and
the eight sections of the template (Appendix C of the board):

1. Constructor arguments that affect inference: pinned default, supported values.
2. Derived tables: the numpy function, its inputs and output shapes, and where TF builds them.
3. Parameters: TF variable name, shape, dtype, trainable or not, zero-initialised or not.
4. Runtime constants and the dtype each is created in.
5. Forward formula: shapes, gathers, reductions, activation, epsilons.
6. Dtype and promotion behaviour: what TF does silently and what the twin has to do explicitly.
7. Options rejected as unsupported, with the reason.
8. Golden-fixture keys used to test the twin and the error achieved.

Three conventions:

- **[I]** marks a variable name, shape or dtype that is *inferred*: read from the code, or seen in an ad hoc run of
  the TF class. **[V]** marks one *verified* against a `probe_<model>.json` of SPEC2 (the TensorFlow probe of the
  variables and tensor attributes of the test yamls, `baselines/probes/`). The sheets were written before the probe
  existed and the tags have been flipped since: **[V]** now covers every name, shape, dtype and trainability that
  `model_grace`, `model_grace_2L_omat` or `model_grace_2L_omat_large_base` exercises, in both parameter dtypes
  (float32 and float64). **[I]** remains only for what no probed yaml exercises: the Gaussian and `RadSinBessel`
  bases, `Cheb` with `normalized`, `norm` in the radial MLPs, `MLPRadialFunction_v2` with `chem_embedding`,
  `chemical_embedding` in `SingleParticleBasisFunctionEquivariantInd` and `ConstantScaleShiftTarget`, a non-zero
  `shift` or an `atomic_shift_map`, `lm_first = True`, `out_norm`, `is_central_atom_type_dependent` for
  `FCRight2Left`, a dict `avg_n_neigh`, and `sep_lin_gate`. Each of those items says so where it occurs. The probe
  confirmed every name, shape and dtype that the sheets stated; it corrected one description (`FCRight2Left.norm_map`
  stays a float64 numpy array and `norm_out_factor` is the tensor used in the forward pass).
- **Measured** marks a behaviour obtained by running the TF class on the pinned commit, as opposed to a reading of
  the code. It is not a verification against a probe.
- **Proposal** marks a decision the reviewer may overrule; the closing list "For the reviewer" of a sheet repeats
  them.

## Facts every sheet relies on

These hold for all 17 classes and are not repeated in each sheet.

- **G1. Two dtypes.** The model has a parameter dtype (`param_dtype`, passed to every `build`:
  `tpmodel.py:831-839`) and a data dtype (`input_signature_float_dtype`, default `tf.float64`:
  `tpmodel.py:832`). Bond vectors and everything computed from them are float64 in every shipped use; the
  parameters are float64 or float32.
- **G2. Implicit promotion is on.** The first TensorFlow-side module of `tensorpotential` that is imported runs
  `tf.experimental.numpy.experimental_enable_numpy_behavior(dtype_conversion_mode="all")`
  (`_configure_tf_options`, `__init__.py:46-67`, applied through the import of `tensorpotential/_tf_options.py`, which
  every such module makes before `import tensorflow`; the package itself no longer imports TensorFlow), so a float32 tensor times a float64 tensor silently gives float64, and TF32 is
  disabled. Whether a result is float32 or float64 is therefore decided class by class, in section 6 of each
  sheet, and a twin that follows PyTorch's promotion rules by default will agree only where the sheet says so.
- **G3. Defaults are part of the model.** `capture_init_args` stores the constructor arguments, defaults
  included, in `model.yaml` (`instructions/base.py:158-225`, `to_dict` at 206-212). A twin reads the arguments
  from the yaml and must not apply its own defaults to a key that is present; for an absent key the pinned
  default of section 1 applies.
  The two shipped 2L yamls are in the old flat format (one entry per instruction, no `metadata` block, so no
  `param_dtype`); `load_instructions` reads that format, the wrapped one and the oldest list format
  (`instructions/base.py:274-325`). The twin reads the parameter dtype from the weights it is given, not from the yaml.
- **G4. Call convention.** `TPInstruction.__call__` runs `frwrd` and stores the result under the instruction's
  own name in the data dictionary; it fails if the name is already there (`instructions/base.py:375-383`).
  Instructions run in the order of the yaml. Consumers refer to producers by name (`bonds`, `basis`, `vhat`, ...).
- **G5. Data keys.** `bond_vector` `[n_bonds, 3]`, `mu_i` and `mu_j` (element index of the central and the
  neighbouring atom of each bond), `ind_i` and `ind_j` (atom index of each bond), `batch_tot_nat`
  (`constants.py:29-33`, `13-14`).
- **G6. Forces come from autograd.** `tape.watch(bond_vector)`, the atomic energies are differentiated with
  respect to it, and the negative gradient is the pair force (`tpmodel.py:293-297`). Every instruction between
  `bond_vector` and the energy has to be differentiable in the twin, and the constants that shape the gradient
  (the `1e-10` softening of the bond length, the `r == 0` substitutions of the radial bases) are part of the
  specification.
- **G7. Angular index.** A tensor with an `lm` axis lists `l = 0, 1, ...` in blocks, and `m = -l, ..., l`
  inside a block; the index is `l*l + l + m`, the axis has `(lmax + 1)**2` entries
  (`instructions/base.py:516-528`).
- **G8. Variable names carry the weight-decay flag.** `no_weight_decay=True` puts `no_decay` in a weight's name
  and `False` puts `_` (`functions/nn.py:40-43`, `272-275`). The flag has no effect at inference, but the
  checkpoint keys differ between the two spellings, and the extractor has to follow the names of the model.
- **G9. Variable names are not unique keys.** Some variables are created without a name scope or a name:
  the two `InvariantLayerRMSNorm.scale` of large_base are both called `Variable:0`. The extractor keys weights by the
  attribute path of the instruction (`<instruction>/<attribute>`), as rule R6 says, never by the TF variable name.
- **G10. The energy chain.** `CreateOutputTarget` makes `atomic_energy`; every output instruction after it reads that key
  and writes the result back under the same key (`instructions/output.py:68-75`). A dump of the data dictionary therefore
  holds only the last value; an intermediate value needs a snapshot before and after each output instruction. The model
  reads `atomic_energy` as `[n_atoms, 1]` (`tpmodel.py:296`).

## Unread keywords

Many classes end in `**kwargs`, and `capture_init_args` stores whatever keys the yaml has. Keys that TF accepts and
**never reads** therefore exist in shipped models. Measured on the two 2L yamls (every key that is not a constructor
parameter): `RadialBasis` (the basis keywords `nfunc`, `p`, `rcut`, `normalized`, which are read); `ProductFunction`
`n_out`, `chemical_embedding`, `downscale_embedding_size`; `FunctionReduceN` `n_in`, `chemical_embedding`,
`downscale_embedding_size`; `SingleParticleBasisFunctionEquivariantInd` `radia_basis` (sic), `n_out`;
`LinMLPOut2ScalarTarget` `full_origin_norm`, `init_norm` (omat yaml only). Policy for the twins (decided, owner, 2026-10-04): each class keeps an allow-list of keys that TF ignores, accepted with any value; every other
unknown key is an error naming it. The allow-list of a class is in its sheet.

## Dtype chain of a float32 model (Measured)

Both 2L yamls built with float32 parameters on the pinned commit and run on a 16-atom periodic structure with float64
data (`tools/oracle_snapshot.py` builders): the outputs of `BondLength`, `ScaledBondVector`, `RadialBasis` and
`SphericalHarmonic` are **float64**; the outputs of every instruction from the radial functions `R`, `R1` on (all
`SingleParticleBasisFunction*`, `FCRight2Left`, `ProductFunction`, `FunctionReduceN`, `InvariantLayerRMSNorm`, the
output instructions) are **float32**. With float64 parameters everything is float64. The energy of the float32 model
differs from the float64 one by `1.6e-7` eV and `5.3e-9` eV (omat, large_base; same seeded weights, one structure). Shapes
for that structure (16 atoms, 928 bonds): `R` `[928, 32, 25]` (omat) and `[928, 42, 25]` (large_base); the sheets give
the others.

## Existing TF characterisation

`tests/test_compute.py` and `tests/test_output.py` (issue TEST6) pin the behaviour of the 17 classes with
logic tests and with oracle-based physics tests. The sheets cite them for every behaviour they pin, and a comment
starting "Pins current behaviour (reported as a finding)" marks a TF defect that a twin must not copy by
accident. They are the first fixture of a twin: the TF numbers of a sheet are reproducible from there.

## Decisions of the review

The owner decided on 2026-10-04 (the PR for these sheets is the record); the sheets say **Proposal** only for what is
still open, and the "For the reviewer" list of each sheet states what was decided.

| Topic | Decision | Sheet |
|---|---|---|
| Unread keys | per-class allow-list of keys TF ignores; every other unknown key is an error naming it | every class; list in each sheet |
| `lora_config`, `is_per_atom`, `type: complex` | rejected at load | radial MLPs, embedding, `SphericalHarmonic` |
| All four radial bases, both Chebyshev kinds | ported | `RadialBasis` |
| `norm` and the chemical embeddings of the radial MLPs, the gate of `MLPRadialFunction_v2` | ported, `tanh` copied where it is, not reported upstream | `MLPRadialFunction`, `MLPRadialFunction_v2` |
| `rc` in two precisions | copied | `RadialBasis` |
| `dense_nbr` | `true` and `false` accepted, same sum | `SingleParticleBasisFunctionEquivariantInd` |
| `init_target_value = "ones"` | ported | `FunctionReduceN` |
| `simplify = True` | rejected | `FunctionReduceN` |
| `atomic_shift_map` keys other than `0 .. n-1` | rejected with a message | `ConstantScaleShiftTarget` |
| `return_hidden_target` | ported (extra output `concat([lin, h_last])`) | `LinMLPOut2ScalarTarget` |
| `l != 0` in the energy chain | rejected after a check: the scalar readouts return the same `[n_atoms, 1]` for any `l` | `CreateOutputTarget` |
| `local` with a `chem_embedding` | rejected explicitly (`NotImplementedError`) | `MLPRadialFunction_v2` |

**Disagreements with the option table of SPEC2.** `tensorpotential/torch_backend/spec/options.py` (and the generated
`option_matrix.md`) was written before these decisions and differs in four places, which the issue that builds the twins
has to reconcile (the sheets are the specification): `FunctionReduceN.init_target_value` accepts only `"zeros"` (decided:
`"ones"` too); `FunctionReduceN.simplify` accepts both booleans (decided: `True` rejected); `LinMLPOut2ScalarTarget`
`return_hidden_target` accepts only `null` (decided: ported); `ConstantScaleShiftTarget.atomic_shift_map` accepts only `null`
(the sheet ports a map with keys `0 .. n-1`; a dict is seen only in `GRACE-3L-OMAT-large`). This PR does not change the
option table.

## Registry and completeness checks (SPEC5)

`tensorpotential/torch_backend/spec/registry.py` holds one entry for every `TPInstruction` subclass of the TF source (39 today):
17 `supported` (these sheets, the option rules of `options.py`) and 22 `rejected` with a reason (`REJECTED_CLASSES`). The loader calls
`check_supported`, which lists every unsupported class and option in one `UnsupportedModelError`; its `problems` carry the kind:

| Kind | When | Message names |
|---|---|---|
| `unknown_class` | the `__cls__` path is not in the registry; classes match by exact name, so a subclass of a supported class is unknown too | instruction, class, the nearest registry class |
| `rejected_class` | the class has no twin | instruction, class, the reason |
| `unknown_option` | a key the class does not have (a key TF ignores is accepted) | instruction, class, the nearest valid key |
| `missing_option` | a parameter without a default that neither the yaml nor `DEFAULTS` gives | instruction, class, option |
| `unsupported_value` | a value outside the rule; for a default the yaml omits, the message says the default applies | instruction, class, option, what is supported |
| `rejected_value` | a value seen in a shipped yaml that has no twin | instruction, class, option, the reason |

`tools/instruction_ast.py` reads the same classes from the source with `ast` (no TensorFlow). `python tools/instruction_ast.py check` fails on an
instruction class without a registry entry, a registry entry for a class that no longer exists, a constructor parameter without an option rule,
a pinned default that differs from the source (`RadialBasis` takes its defaults from the Chebyshev basis it forwards to: `DEFAULTS_ORIGIN`), a
default that is not a literal, and a supported class that lost `capture_init_args`. `tests/test_registry.py` runs it on the tree and on copies
with a planted change; `.github/workflows/upstream-completeness.yml` runs it weekly on the current upstream master (it needs Actions enabled on
the fork and, for the schedule, the file on the default branch; `workflow_dispatch` runs it by hand). It passed on the local `upstream/master`
(`cc1bb38`, 2026-09-03). A new upstream class goes into `SUPPORTED_OPTIONS` (with a sheet) or `REJECTED_CLASSES` (with a reason).

`GOLDEN_TESTS` in `registry.py` maps a supported class to the test that compares its twin with the golden fixtures; it is empty, and
`EXPECTED_WITHOUT_GOLDEN` in `tests/test_registry.py` lists the 17 classes without one. A twin issue adds its classes to the first and removes
them from the second in the same pull request; the test fails in both directions, so the list can only shrink.
