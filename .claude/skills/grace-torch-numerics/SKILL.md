---
name: grace-torch-numerics
description: Units, sign conventions, dtype policy, derivatives, equivariance and layout conventions for GRACE energies, forces, virial and stress. Use when writing or reviewing anything that computes or converts energies, forces, stress, neighbour-list vectors, tensor layouts or precision settings, in TensorFlow or torch code.
---

# grace-torch-numerics

## Conventions (fixed by the TensorFlow implementation; the torch code must match)

- Units: energy eV, forces eV/Angstrom, virial eV, stress eV/Angstrom^3.
- Geometry enters the model as pair data: `bond_vector = r_j - r_i + S . cell` for each directed pair (i -> j and j -> i both present, self-images
  kept, strict `d < rc`), with `ind_i`, `ind_j`, `mu_i`, `mu_j` (element indices), `atomic_mu_i`.
- **Forces** are `-dE/d(bond_vector)` per pair, accumulated `F = segment_sum(pair_f, ind_j) - segment_sum(pair_f, ind_i)`. **Virial** is
  `sum(pair_f (x) D)`; the **ASE stress** is `-virial / V` with the Voigt reorder `[0, 1, 2, 5, 4, 3]`. Fix the sign on a hand case, never by trial.
- The instruction graph runs in file order; ordinary instructions write `input_data[name]`, **output instructions overwrite `input_data[target.name]` in place**.
- Constructor defaults of every instruction are part of the saved `model.yaml` (`capture_init_args`); `param_dtype` has three code defaults that disagree
  (float64 without metadata in `metadata_utils.py`, float32 in `tensorpot.py` and `utils.py`), so pin it explicitly.
- Runtime scaling is applied at call time (for example a `1/sqrt(n_in)` on dense blocks), not folded into stored weights; the twins reproduce that.
- Internal torch layout of equal-multiplicity operands is `[atoms, lm, n]`; the TF default is `[atoms, n, lm]` and `lm_first` is `[lm, atoms, n]`.
  Convert the TF fixtures at the boundary rather than changing the math.

## Precision

- Parameter dtype, geometry dtype and every explicit cast are decided in **one module**; nothing else casts. A float32 constant or a Python scalar can silently
  promote or demote a tensor: audit creation and use of every constant, and test with both fp64 and fp32.
- fp32 parameters with fp64 geometry is the supported mixed path; its error against fp64 is measured, not assumed.
- TF32 is **scoped, never global**: a context manager around `forward` sets and restores it; model constructors never touch process-wide precision
  settings. Precision is owned by the caller.
- Scatter sums use `index_add_`/`segment_sum` on a fixed order in tests; summation order differs between neighbour-list backends, which shows up at the
  level of floating-point rounding, so compare canonicalised pair sets and use the tolerance table.

## Derivatives

- Forces by autograd with respect to `bond_vector` (the geometry input), not to positions, so the same graph gives the virial. Keep the graph needed for
  a second derivative only when asked (`create_graph`/`retain_graph` choices are explicit and tested).
- Always verify a derivative against a central finite difference of the energy in float64, with a step chosen from the energy scale; an analytic gradient
  checked only against another analytic gradient proves nothing.
- Stress: finite-strain derivative of the energy under `r -> (1 + eps) r`, cell strained together with positions.

## Equivariance and invariance checks

Rotate, translate and permute the input and compare: energy invariant, forces equivariant, stress rotating as a rank-2 tensor. Isolated atoms (no bonds) and
single bonds are edge cases the code must handle; a pair at exactly `rc` is not a neighbour. Padded inputs (for compilation) must equal unpadded ones:
padding bonds carry zero weight, a cutoff beyond `rc`, and no fictitious bond is added for an atom with no neighbours (the TF builder does add one beyond the
cutoff; comparisons drop bonds with `d >= rc`).

## Neighbour lists

matscipy is the default in the TF path; vesin and torch-sim's lists are alternatives. All must return the same set of `(i, j, S)` and the same vectors
on the contract's test structures (box smaller than 2 rc, triclinic two-atom cell with self-images, isolated atom, partial pbc, vacuum slab, left-handed
cell, positions outside the cell, mixed-size batches) and against a brute-force oracle. A backend that overflows must raise, never drop neighbours. The
contract is a pure function: it never edits the caller's `Atoms`. The TF calculator's `enforce_pbc` makes all axes periodic and edits the `Atoms` in
place; the contract keeps the all-periodic default for parity and offers per-axis `pbc` as an opt-in flag.

## Accelerated kernels (only after measurement)

cuEquivariance mapping of the products is verified on CPU with `method="naive"` and `math_dtype` set explicitly (the default may run fp32 math);
`uniform_1d` can silently fall back to `naive`, so assert the method actually selected. A backend may not change tables, parameter counts or weight layout.
