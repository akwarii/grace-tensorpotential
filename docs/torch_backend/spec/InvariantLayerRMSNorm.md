# InvariantLayerRMSNorm

| | |
|---|---|
| Source | `instructions/compute.py:3775-3871` (class at 3775), base `TPInstruction` (`instructions/base.py:328`) |
| Family | norm and output |
| Used in | the large_base yaml only: `I_nl_LN` (`inpt rho`, `type only_nonlin`) and `I_0_LN` (`inpt I2`, `type full`), both `init zeros`. Not in the omat yaml. |
| Reads | the entry of `inpt`, `[n_atoms, n_out, 1]` (invariant features); `atomic_mu_i`, `batch_tot_nat_real` |
| Writes | `<name>` `[n_atoms, n_out, 1]` |
| Variables | `scale` (and `lin_scale` for `sep_lin_gate`) |

A root-mean-square normalisation of the invariant channels of each atom, across the channel axis, with a learned
per-channel scale. In `only_nonlin` mode channel 0 (the linear, energy-like channel) passes through untouched and the
other channels are normalised.

## 1. Constructor arguments

`compute.py:3781-3799`. All stored in the yaml (G3).

| Argument | Default | Values | Effect |
|---|---|---|---|
| `inpt` | required | an instruction with `n_out` (stored as a name) | Source; `n_out` of this instruction is `inpt.n_out`. If `inpt.lm_first` is true its output is `[lm, n_atoms, n_out]` and is transposed first (`compute.py:3838-3840`). |
| `name` | required | any unique `str` | Key of the output. |
| `type` | `"only_nonlin"` | `"full"`, `"only_nonlin"`, `"sep_lin_gate"` | Section 5. The check is an `assert` in the constructor, so a value set after construction is not checked (TEST6 finding 15). |
| `init` | `"zeros"` | `"zeros"`, `"ones"`, `"random"`, `"near_zero"` | Training only (initial `scale`); an `assert`. |
| `**kwargs` | | | Swallowed; none in the shipped yamls. |

## 2. Derived tables

None. `n_out = inpt.n_out`; the channel count of the output equals that of the input.

## 3. Parameters

`scale` is **[V]** against `probe_model_grace_2L_omat_large_base.json`, both parameter dtypes (name `Variable:0`, parameter dtype,
trainable; shapes `[1, 17, 1]` and `[1, 16, 1]`; checkpoint keys `model/instructions/I_0_LN/scale/...` and
`.../I_nl_LN/scale/...`). `lin_scale` is **[I]**: no probed yaml uses `sep_lin_gate`. Trainable, `float_dtype`; the variables are created **unnamed** by `tf.Variable(...)`.

| Attribute | Shape | Present when | Init |
|---|---|---|---|
| `scale` | `[1, n_out, 1]` for `full`; `[1, n_out - 1, 1]` for `only_nonlin` and `sep_lin_gate` | always | per `init` (zeros, ones, `N(0, 1)`, `N(0, 1e-8)`); `zeros` is the default, so an untrained model outputs zero for the normalised channels |
| `lin_scale` | `[1, 1]` | `type = "sep_lin_gate"` | same |

**The TF name is just `Variable:0`** for every instance: `build` has no name scope (it is not decorated with
`tf.Module.with_name_scope`) and the variables carry no name, so the two instances of large_base (`[1, 17, 1]` and
`[1, 16, 1]`) have the **same** variable name (**Measured**, `model.variables`; **[V]**, `probe_model_grace_2L_omat_large_base.json`). A name-based extractor cannot tell them
apart; the extractor has to key by attribute path (`<instruction>/scale`, `<instruction>/lin_scale`) from the object
tree, never by variable name. Examples: large_base `I_nl_LN.scale` `[1, 16, 1]` (input `rho` has 17 channels),
`I_0_LN.scale` `[1, 17, 1]`.
`build` creates them only `if not self.is_built`, and there is no `else` for an unknown `init` (no variable is made,
finding 15).

## 4. Runtime constants

`epsilon = 1e-10`, a tensor in `float_dtype` (`compute.py:3833`), cast to the data dtype where used. Not in the checkpoint
and not configurable: hard-code it. (**[V]**: `epsilon` is a scalar tensor in the parameter dtype, no checkpoint key.)

## 5. Forward

`compute.py:3836-3871`. `x = data[inpt]` as `[n_atoms, n, w]` (here `w = 1`: `lm` has one entry). `n_real` is
`batch_tot_nat_real`, `n_total` the length of `atomic_mu_i`, and `real[a] = (a < n_real)`.

```
full:           rms = rsqrt( mean(x**2, axis=1, keepdims) + eps )            # [n_atoms, 1, w]; mean over the n channels
                rms = where(real, rms, 0)
                out = x * rms * scale                                         # scale [1, n, 1]
only_nonlin:    lin    = x[:, 0, :]                                           # [n_atoms, w]
sep_lin_gate:   lin    = x[:, 0, :] * lin_scale                               # lin_scale [1, 1]
                nonlin = x[:, 1:, :]                                          # [n_atoms, n-1, w]
                nl_rms = rsqrt( mean(nonlin**2, axis=1, keepdims) + eps )
                nl     = where(real, nonlin * nl_rms * scale, 0)              # scale [1, n-1, 1]
                out    = concat([lin[:, None, :], nl], axis=1)                # [n_atoms, n, w]
```

The mean is over the **channel** axis, per atom. The output is always `[n_atoms, n_out, w]` (never `lm_first`: the
class has no such attribute, so consumers read it as `[n_atoms, n, w]`).

**Padding.** The twin has no padded atoms: it passes `batch_tot_nat_real = n_atoms`, and every `where(real, ...)` is the
identity. If padding is ever used, **Measured** behaviour: in `full` mode a padding atom's output is exactly 0; in
`only_nonlin` and `sep_lin_gate` modes its normalised part is 0 but its linear channel 0 is **not masked** (TEST6
finding 15, pinned at `tests/test_compute.py:5017`).

The formula was re-implemented in numpy and compared with TF: the two shipped instances (seeded non-zero scales) and
synthetic instances of the three types: maximum difference `4.4e-16` (float64).

## 6. Dtype and promotion

The output has the dtype of `x` (the cast of `epsilon`, `scale` and `lin_scale` to `x.dtype` happens in the forward
pass): **Measured** float32 for float32 parameters (the input of the layer is float32 in that model). The twin casts the
three constants to the dtype of `x`.

## 7. Options rejected

- `type` outside the three values: rejected at load time (TF rejects it in the constructor).
- `init`: accepted and ignored by the twin (no effect on a loaded model). An unknown `init` is not an error in TF
  after construction; the twin does not need it.
- `full` and `sep_lin_gate` are **ported** although only `only_nonlin` and `full` occur in a shipped model; `sep_lin_gate`
  needs an option fixture.
- `local = True`: not applicable.

## 8. Golden-fixture keys

Fixture keys `I_nl_LN` and `I_0_LN`, both `[n_atoms, 17, 1]` (large_base). FIX2 produces the yaml fixtures; no achieved
error exists yet. TF oracle: a numpy re-computation of the three types, padding atoms, invariance of the normalised
channels under a rescaling of the input (`tests/test_compute.py:4875-5190`).

## For the reviewer

None beyond the family-wide proposals.
