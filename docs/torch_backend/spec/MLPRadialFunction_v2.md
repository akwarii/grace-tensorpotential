# MLPRadialFunction_v2

| | |
|---|---|
| Source | `instructions/compute.py:826-978` (class at 827); layers in `functions/nn.py:17-147` (`Linear`) |
| Family | geometry and radial |
| Bases | `TPInstruction`, `LORAInstructionMixin` (`instructions/base.py:327`, `393`) |
| Used in | the large_base yaml: `R` (`n_rad_max 42`) and `R1` (`n_rad_max 32`), both `lmax 4, hidden_layers [64, 64], activation [silu, silu], init_type normal, normalize true, chem_embedding null, embed_i false, embed_j true`. Not in the omat yaml. |
| Reads | the basis entry `[n_bonds, nfunc]`; with a chemical embedding also its table and `mu_i`, `mu_j` |
| Writes | `<name>` `[n_bonds, n_rad_max, (lmax + 1)**2]` |
| Variables | one weight matrix per layer; one more for the embedding transform |

The second generation of the radial MLP: a bias-free MLP built from `Linear` layers with a per-layer activation
list, and an optional multiplicative gate by the chemical embedding of the neighbour (and of the centre).
Output layout and shape are those of `MLPRadialFunction`.

## 1. Constructor arguments

`compute.py:833-922`. All stored in the yaml (G3).

| Argument | Default | Values | Effect |
|---|---|---|---|
| `n_rad_max` | required | integer `>= 1` | Radial functions per `l`. |
| `lmax` | required | integer `>= 0` | Highest `l`; `n_out = n_rad_max * (lmax + 1)`. |
| `name` | `"MLPRadialFunction"` | any unique `str` | Key of the output (same default as v1). |
| `basis` | `None` | `RadialBasis`, another `TPInstruction`, `str`, `None` | As for v1 (`compute.py:868-886`); `None` with an `input_shape` builds but never sets `basis_name` (finding 11). |
| `input_shape` | `None` | integer | Width of the basis when `basis` is a name. **No embedding width is added** (unlike v1). |
| `hidden_layers` | `None` | list of ints | `None` means `[64, 64]` (v1: `[64, 64, 64]`). |
| `activation` | `None` | `None`; a `str`; a list of `str` | `None` gives `"silu"` per hidden layer; a string is repeated for every hidden layer; a list must have one entry per hidden layer (`assert`, `compute.py:860-866`). Names are looked up in `ACTIVATION_DICT = {tanh, silu, sigmoid}` (`functions/nn.py:14`) **at the first forward pass**: an unknown name is a bare `KeyError` (finding 11, pinned at `tests/test_compute.py:1571`). `silu` here is the plain `tf.nn.silu`, without the `1.6759` of v1's default. |
| `no_weight_decay` | `True` | bool | Only the weight names (G8). |
| `init_type` | `"normal"` | `"normal"`, `"uniform"`, `"zeros"` | Initialiser of every `Linear` (`functions/nn.py:33-34`, an `assert`). No effect on a loaded model **except** through `normalize` below. |
| `normalize` | `True` | bool | Selects where the `1/sqrt(n_in)` lives: see sections 3 and 4. |
| `chem_embedding` | `None` | a `ScalarChemicalEmbedding` or `None` | Enables the gate. |
| `embed_i` | `False` | bool | Include the central atom's embedding in the gate. |
| `embed_j` | `True` | bool | Include the neighbour's embedding in the gate. |
| `lora_config` | `None` | dict or `None` | Fine-tuning state; see section 7. |

`embed_i` and `embed_j` have no effect without `chem_embedding`.

## 2. Derived tables

`l_tile` as in `MLPRadialFunction` (`compute.py:888-894`): `np.repeat(np.arange(lmax + 1), 2 * np.arange(lmax + 1) + 1)`,
int32, not a variable.

Layer widths `[input_shape] + hidden_layers + [n_rad_max * (lmax + 1)]`, one `Linear` per consecutive pair
(`compute.py:895-910`), so `len(hidden_layers) + 1` layers.

## 3. Parameters

All in `float_dtype`, trainable, created in `build` (`compute.py:924-934`; `functions/nn.py:46-107`). `use_bias` is
`False` (no bias). Weights are `[n_in, n_out]`.

| Attribute | Shape | TF name (**Measured**, `name = MLPRadialFunction`) |
|---|---|---|
| `layers[i].w`, `i = 0 .. len(hidden_layers)` | `[widths[i], widths[i+1]]`, e.g. `[8, 64]`, `[64, 64]`, `[64, 210]` | `MLPRadialFunction_Linear_<i>/Linear_MLPRadialFunction_Linear_<i>_no_decay:0` |
| `embed_transform.w` (only with `chem_embedding`) | `[embedding_size, n_out]` | `<name>_ChemEmb_Linear`, weight-decay flag `_` (this `Linear` is made with the default `no_weight_decay = False`) |

All names, shapes and dtypes of this section are **[I]**.

Initial distribution (training only; a loaded model overwrites it): `normal` is `N(0, s)` with `s = 1` if
`normalize` and `s = 1/sqrt(n_in)` if not; `uniform` is `U(-s, s)`; `zeros` is zero (`functions/nn.py:50-101`).

## 4. Runtime constants

(dtypes **[I]**) `Linear.norm` (`functions/nn.py:36-37`, `56-60`, `102`): `1 / sqrt(n_in)` if `normalize`, else `1.0`; a tensor in the
weight dtype, **not in the checkpoint**. The twin recomputes it from the layer's `n_in` and from `normalize`
(which it reads from the yaml). The embedding transform is always built with `normalize = True`, so its factor is
`1 / sqrt(embedding_size)`.

## 5. Forward

`compute.py:955-978`. `b` is the basis `[n_bonds, nfunc]`.

```
for i in 0 .. L-1:   b = act_i( b @ (w_i * norm_i) )            # L = len(hidden_layers); act_i from ACTIVATION_DICT
y = b @ (w_L * norm_L)                                           # last layer, no activation, [n_bonds, n_out]
```

`Linear.__call__` (`functions/nn.py:124-147`): `w * norm`, cast of the input to the weight dtype if different,
`matmul` over the flattened leading axes, no bias.

With `chem_embedding` (raises `NotImplementedError` if `local` is true, `compute.py:963-964`):

```
z         = embed_transform(data[chem_embedding.name])           # [n_elem, n_out]  (a Linear, no activation)
embedding = 1.0
if embed_j:  embedding = embedding * gather(z, mu_j)
if embed_i:  embedding = embedding * gather(z, mu_i);  embedding = tanh(embedding)   # tanh only here
y = y * embedding
```

The `tanh` sits inside the `if embed_i` block (`compute.py:970-973`): with `embed_j` alone the gate is the
plain gathered embedding, with `embed_i` it is the `tanh` of the product. This is the code as it is; TEST6 did not
list it as a finding, and no test pins the `embed_j`-only case with `tanh` absent. With both flags false the gate
is `1.0` (a no-op).

Then, as in v1:

```
y   = reshape(y, [-1, n_rad_max, lmax + 1])
out = gather(y, l_tile, axis=-1)                                 # [n_bonds, n_rad_max, (lmax + 1)**2]
```

Gate-less case, the shipped one: a plain MLP, `silu` on both hidden layers, no bias.

## 6. Dtype and promotion

- As for v1: the first `Linear` casts the float64 basis to the weight dtype, so with float32 parameters the output
  is float32 (**Measured** `(3, 42, 25)` float32; float64 for float64 parameters).
- The embedding table arrives in the parameter dtype (it is the output of `ScalarChemicalEmbedding`) and `z` is
  computed in it; there is **no cast to the data dtype** here (v1 casts the embedding up to float64). `y` and the
  gate are therefore both in the weight dtype.
- Constructor-supplied LoRA is initialised twice (the layers' `build`, then the instruction's); harmless
  (finding 12).

## 7. Options rejected

- `lora_config` not `None`: as for v1. **Proposal**: reject with a message that tells to merge first
  (`compute.py:931-932`, `936-953`).
- `init_type`: accepted and irrelevant to a loaded model; `zeros` and `uniform` parse. Not rejected.
- `basis = None`: rejected at load time (TF fails at the first forward pass).
- An activation name outside `{tanh, silu, sigmoid}`: rejected at load time (TF raises `KeyError` at the first
  forward pass).
- `local = True` with a `chem_embedding` (TF raises `NotImplementedError`, `compute.py:963-964`). `local` is the TF
  flag of the domain-decomposed execution path (`tpmodel.py:388-400`, `609-617`); the twin has no such mode, so the
  case does not arise. **Proposal**; a decision for the reviewer.

## 8. Golden-fixture keys

Fixture keys `R` (`[n_bonds, 42, 25]`) and `R1` (`[n_bonds, 32, 25]`) of the large_base yaml. Oracle: a numpy
re-computation of the forward pass from seeded weights, including the gate, and the LoRA update
(`tests/test_compute.py:1418-1743`). FIX2 produces the yaml fixtures; no achieved error exists yet. The
`embed_i` / `embed_j` / `normalize` option pairs have no yaml fixture: they need the option-pair fixtures of FIX2.

## For the reviewer

1. **The gate.** `chem_embedding` is not used by either 2L model. Proposal: port it (as the R3 rule says), with
   the `tanh` exactly where it is, and cover it by an option-pair fixture. Alternative: reject until a model needs it.
2. **The `tanh` placement** (inside `if embed_i`) looks accidental. The twin copies it (the TF classes are the
   oracle); say so if the owner wants it reported to upstream instead.
3. **Proposal (section 7)**: reject `lora_config`, unknown activation names and `basis = None` at load time; no
   `local` mode in the twin.
