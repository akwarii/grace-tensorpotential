# LinMLPOut2ScalarTarget

| | |
|---|---|
| Source | `instructions/output.py:311-439` (class at 312); `FullyConnectedMLP` in `functions/nn.py:347-433`, `scalar_rms_ln` in `functions/nn.py:464-470`; base `TPOutputInstruction` (`output.py:49-74`), `LORAInstructionMixin` (`instructions/base.py:393`) |
| Family | norm and output |
| Used in | omat: name `MLPOut2ScalarTarget`, `origin [I_out]`, `hidden_layers [64]`, `normalize layer`, `activation null`. large_base: name `LinMLPOut2ScalarTarget`, `origin [I_nl_LN, I_0_LN]`, `hidden_layers [64]`, `normalize null`, `activation tanh`. Both `n_out 1`, `l 0`, target `atomic_energy`. |
| Reads | the `origin` entries (`[n_atoms, n, 1]`), the target; `batch_tot_nat_real`, `batch_tot_nat` |
| Writes | **`atomic_energy`** (the target's name), `[n_atoms, 1]` (`TPOutputInstruction.__call__`, `output.py:67-74`) |
| Variables | the MLP weights; `scale` with `normalize` |

The energy readout: for each atom, channel 0 of the invariant features is added to the energy **linearly**, the other
channels (normalised if asked) go through a bias-free MLP, and the result is added to the running target.

## 1. Constructor arguments

`output.py:322-366`. All stored in the yaml (G3); `origin` and `target` are stored as names.

| Argument | Default | Values | Effect |
|---|---|---|---|
| `origin` | required | list of instructions with `n_out` and (if equivariant) `lmax` | The invariant sources; all have the same `n_out` (`assert`) and `lmax == 0` (`assert`; an instruction without `lmax` counts as 0). |
| `target` | required | a `CreateOutputTarget` | The accumulator, and the key that is overwritten. `target.l == l` (`assert`). |
| `hidden_layers` | `None` | list of ints or `None` | `None` means `[32]`; both yamls give `[64]`. |
| `name` | `"LinMLPOut2ScalarTarget"` | any unique `str` | Names the MLP layers and `scale`. The omat yaml names it `MLPOut2ScalarTarget`. |
| `n_out` | `1` | integer | MLP output width; `1` for an energy. |
| `normalize` | `None` | `None`, `"layer"` | `"layer"` normalises the non-linear channels per atom (section 5); any other string fails an `assert`. |
| `activation` | `None` | `None`, `"tanh"`, `"silu"`, `"sigmoid"` | `None`: `silu(x) * 1.6759` (`functions/nn.py:343-344`); a string: that plain function (`functions/nn.py:14`, `386-391`). |
| `l` | `0` | integer | Must equal `target.l`. |
| `lora_config` | `None` | dict or `None` | Section 7. |
| `return_hidden_target` | `None` | a `str` or `None` | Names an extra data key that receives the hidden features (section 7). Both yamls: absent. |
| `**kwargs` | | | Swallowed. The omat yaml stores `full_origin_norm: false` and `init_norm: zeros`, both **ignored** (allow-list of this class). |

## 2. Derived tables

None. `n_in = origin[0].n_out - 1` is the MLP input width; the MLP layer sizes are `[n_in] + hidden_layers + [n_out]`.

## 3. Parameters

Names, shapes and dtypes are **[I]**. Trainable, `float_dtype`.

| Attribute | TF name | Shape | Notes |
|---|---|---|---|
| `mlp.layer<i>.w` | `<name>_MLP_layer<i>/DenseLayer_<name>_MLP_layer<i>__:0` | `[sizes[i], sizes[i+1]]` | `N(0, 1)`, no bias; weight-decay flag `_` (the MLP is built with the default `no_weight_decay = False`). **Measured**: omat `[12, 64]`, `[64, 1]`; large_base `[16, 64]`, `[64, 1]`. |
| `scale` (only with `normalize`) | `<name>/scale:0` | `[1, n_in]` | `N(0, 1e-16)` at init, so an untrained layer-normalised branch is 0. **Measured**: omat `[1, 12]`. |

## 4. Runtime constants

- `DenseLayer.norm = 1 / sqrt(n_in_layer)` per MLP layer (`functions/nn.py:295`), not stored.
- `epsilon = 1e-16` in `scalar_rms_ln` (a Python float, `functions/nn.py:464`); not configurable.
- The `1.6759` of the default activation.

## 5. Forward

`output.py:400-439`. For each origin `s`, `x = data[s]` (transposed to `[n_atoms, n, w]` when `s.lm_first` is true),
`f = x[:, :, 0]` `[n_atoms, n]`:

```
transformed = sum_s f_s[:, 1:]                       # [n_atoms, n - 1]
lin         = sum_s f_s[:, 0:1]                      # [n_atoms, 1]
if normalize == "layer":
    rms         = rsqrt( mean(transformed**2, axis=-1, keepdims) + 1e-16 )      # padding atoms: rms = 0
    transformed = transformed * scale * rms                                      # scale [1, n - 1]
h  = mlp(transformed)                                 # [n_atoms, n_out]; hidden layers with act, the last without, no bias
atomic_energy = target + (h + lin)                    # the target is a scalar 0.0 or [n_atoms, 1]
```

The sums over origins come first, so with two origins (large_base) the channels `1 ..` of both are **added** before the
MLP and the two linear channels are added to each other. The MLP is `FullyConnectedMLP` as in the radial MLP (hidden
activation, none on the output; `use_bias False`).

**Padding.** The twin has no padding atoms (`batch_tot_nat_real = batch_tot_nat = n_atoms`), so the mask
`range(n_total) < n_real` is all true. For padded batches **Measured**: without `normalize` nothing masks padding atoms;
with `normalize = "layer"` they get `target + lin` (TEST6 finding 17, pinned at `tests/test_output.py:656`).

The formula was re-implemented in numpy from the instruction's weights and compared with the TF `atomic_energy` of both 2L
models (including the following scale or shift instruction): maximum difference `3.1e-17` on values up to `8e-2` (float64,
seeded weights).

## 6. Dtype and promotion

The origin features, the MLP and `scale` are all in the parameter dtype in a real model, so the energy has the
parameter dtype (**Measured** float32 for float32 parameters). The Python scalar `1e-16` of `scalar_rms_ln` takes the dtype of the tensor it is added to; the twin uses the same
constant in the dtype of the features.

## 7. Options rejected

- `lora_config` not `None`: **Proposal**, rejected (`output.py:378-379`, `383-398`).
- `return_hidden_target` not `None`: **Proposal**, rejected. It writes the hidden features into the data dictionary from
  inside the forward pass (a contract violation noted in the code, `output.py:429-434`) for the uncertainty tools; the
  twin has no GMM-UQ feature path (out of scope).
- `normalize` other than `None` and `"layer"`: rejected (TF asserts).
- Several origins, `lm_first` origins, `hidden_layers` of any length, `activation` in `{tanh, silu, sigmoid}`, `n_out`:
  ported (large_base has two origins; `lm_first` needs an option fixture).
- The unread keys `full_origin_norm`, `init_norm`: accepted and ignored.
- `local = True`: not applicable.

## 8. Golden-fixture keys

The key `atomic_energy` after the whole chain, `[n_atoms, 1]` (see the sheets of the target instructions); the oracle
for this class alone is the numpy re-computation of `tests/test_output.py:379-837` from seeded weights, including the
layer normalisation, padding atoms and the LoRA update. FIX2 produces the yaml fixtures; no achieved error exists yet.
A dump of the dictionary contains only the **last** value of `atomic_energy`, so a fixture of this class alone needs
the value before the shift instruction (a snapshot of the key before and after each output instruction).

## For the reviewer

None beyond the family-wide proposals (reject `lora_config`, plus `return_hidden_target`).
