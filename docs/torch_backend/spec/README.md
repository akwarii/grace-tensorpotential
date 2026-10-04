# Class spec sheets (SPEC1)

One sheet per TensorFlow instruction class that the PyTorch twins must reproduce: the 17 classes of the two
2L foundation `model.yaml` files (`tests/model_grace_2L_omat.yaml`, `tests/model_grace_2L_omat_large_base.yaml`).
A sheet states what the TF class does, precisely enough to write the twin without reading the TF code again,
and says where in the TF code each statement comes from. The TF classes are the oracle (rule R1); where a sheet
and the code disagree, the code wins and the sheet is wrong.

## Pinned source

Every citation refers to `torch-backend` at commit `ae456fe` (TensorFlow 2.20.0, `tf_keras`). A citation reads
`instructions/compute.py:186-193`: a path **relative to `tensorpotential/`**, then the line range. Tests are
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
| Norm and output | InvariantLayerRMSNorm, CreateOutputTarget, LinMLPOut2ScalarTarget, ConstantScaleShiftTarget, TrainableShiftTarget | not written |

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
  the TF class. **[V]** marks one *verified* against a `probe_<model>.json` of SPEC5 (the TensorFlow probe of the
  variables and tensor attributes of the test yamls). Every such item in these sheets is **[I]** until SPEC5
  delivers the probe files; the issue that does it flips the tags (SPEC5 states that as its exit criterion).
- **Measured** marks a behaviour obtained by running the TF class on the pinned commit, as opposed to a reading of
  the code. It is not a verification against a probe.
- **Proposal** marks a decision the reviewer may overrule; the closing list "For the reviewer" of a sheet repeats
  them.

## Facts every sheet relies on

These hold for all 17 classes and are not repeated in each sheet.

- **G1. Two dtypes.** The model has a parameter dtype (`param_dtype`, passed to every `build`:
  `tpmodel.py:830-838`) and a data dtype (`input_signature_float_dtype`, default `tf.float64`:
  `tpmodel.py:831`). Bond vectors and everything computed from them are float64 in every shipped use; the
  parameters are float64 or float32.
- **G2. Implicit promotion is on.** Importing `tensorpotential` calls
  `tf.experimental.numpy.experimental_enable_numpy_behavior(dtype_conversion_mode="all")`
  (`__init__.py:46-65`), so a float32 tensor times a float64 tensor silently gives float64, and TF32 is
  disabled. Whether a result is float32 or float64 is therefore decided class by class, in section 6 of each
  sheet, and a twin that follows PyTorch's promotion rules by default will agree only where the sheet says so.
- **G3. Defaults are part of the model.** `capture_init_args` stores the constructor arguments, defaults
  included, in `model.yaml` (`instructions/base.py:157-224`, `to_dict` at 206-212). A twin reads the arguments
  from the yaml and must not apply its own defaults to a key that is present; for an absent key the pinned
  default of section 1 applies.
  The two shipped 2L yamls are in the old flat format (one entry per instruction, no `metadata` block, so no
  `param_dtype`); `load_instructions` reads that format, the wrapped one and the oldest list format
  (`instructions/base.py:273-324`). The twin reads the parameter dtype from the weights it is given, not from the yaml.
- **G4. Call convention.** `TPInstruction.__call__` runs `frwrd` and stores the result under the instruction's
  own name in the data dictionary; it fails if the name is already there (`instructions/base.py:374-382`).
  Instructions run in the order of the yaml. Consumers refer to producers by name (`bonds`, `basis`, `vhat`, ...).
- **G5. Data keys.** `bond_vector` `[n_bonds, 3]`, `mu_i` and `mu_j` (element index of the central and the
  neighbouring atom of each bond), `ind_i` and `ind_j` (atom index of each bond), `batch_tot_nat`
  (`constants.py:29-33`, `13-14`).
- **G6. Forces come from autograd.** `tape.watch(bond_vector)`, the atomic energies are differentiated with
  respect to it, and the negative gradient is the pair force (`tpmodel.py:292-296`). Every instruction between
  `bond_vector` and the energy has to be differentiable in the twin, and the constants that shape the gradient
  (the `1e-10` softening of the bond length, the `r == 0` substitutions of the radial bases) are part of the
  specification.
- **G7. Angular index.** A tensor with an `lm` axis lists `l = 0, 1, ...` in blocks, and `m = -l, ..., l`
  inside a block; the index is `l*l + l + m`, the axis has `(lmax + 1)**2` entries
  (`instructions/base.py:515-527`).
- **G8. Variable names carry the weight-decay flag.** `no_weight_decay=True` puts `no_decay` in a weight's name
  and `False` puts `_` (`functions/nn.py:39-42`, `271-274`). The flag has no effect at inference, but the
  checkpoint keys differ between the two spellings, and the extractor has to follow the names of the model.

## Unread keywords

Many classes end in `**kwargs`, and `capture_init_args` stores whatever keys the yaml has. Keys that TF accepts and
**never reads** therefore exist in shipped models. Measured on the two 2L yamls (every key that is not a constructor
parameter): `RadialBasis` (the basis keywords `nfunc`, `p`, `rcut`, `normalized`, which are read); `ProductFunction`
`n_out`, `chemical_embedding`, `downscale_embedding_size`; `FunctionReduceN` `n_in`, `chemical_embedding`,
`downscale_embedding_size`; `SingleParticleBasisFunctionEquivariantInd` `radia_basis` (sic), `n_out`;
`LinMLPOut2ScalarTarget` `full_origin_norm`, `init_norm` (omat yaml only). Policy for the twins (**Proposal**, accepted
for the radial classes): each class keeps an allow-list of keys that TF ignores, accepted with any value; every other
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
