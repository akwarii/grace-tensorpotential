# ScalarChemicalEmbedding

| | |
|---|---|
| Source | `instructions/compute.py:982-1123` (class at 982), bases `TPInstruction`, `LORAInstructionMixin`, `ElementsReduceInstructionMixin` (`instructions/base.py:328`, `394`, `418`) |
| Family | embedding and single-particle basis |
| Used in | both 2L yamls, name `Z`, `embedding_size 128`, `is_trainable true`, `init random`; read by both `SingleParticleBasisFunctionScalarInd` instructions (`A`, `B0`) |
| Reads | nothing |
| Writes | `Z` `[n_elements, embedding_size]` (the weight table itself) |
| Variables | `w`, plus two non-trainable element-map variables |

A learned vector per chemical element. Its output is the table, not a per-atom or per-bond tensor: consumers gather
rows from it with the element index of an atom or a bond.

## 1. Constructor arguments

`compute.py:997-1018`. All stored in the yaml (G3).

| Argument | Default | Values | Effect |
|---|---|---|---|
| `name` | required | any unique `str` | Key of the output, and the name consumers use to find the table. |
| `element_map` | required | dict `symbol -> int` | `n_elements = len(element_map)`; `element_map_symbols` and `element_map_index` are the keys and values in dict order. The values are the **row indices** of the table: the element index `mu` carried by `mu_i`, `mu_j` and `atomic_mu_i` addresses row `mu`, so a model needs the values to be `0 .. n_elements - 1`. The two shipped yamls have 89 elements. |
| `embedding_size` | required | integer `>= 1` | Width of the table (128 in both yamls). |
| `is_trainable` | `True` | bool | Training only. |
| `init` | `"random"` | `"random"`, `"zero"`, `"zeros"`, `"delta"` | Training only: `N(0, 1)`, zeros, or a rectangular identity (`compute.py:1034-1057`). Any other value raises `NotImplementedError` in `build`. No effect on a loaded model. |
| `lora_config` | `None` | dict or `None` | Fine-tuning state; section 7. |

The class has **no `is_per_atom` attribute**: every consumer reads it with `getattr(..., "is_per_atom", False)`
(`compute.py:728-729`, `1179`, `1340`), so the per-atom branches of the consumers are never taken for it.

## 2. Derived tables

None. The element map is data: `get_element_map()` returns `(symbols as str, index)` (`compute.py:1020-1025`) and
the calculators use it to turn atomic numbers into element indices and to find the number of elements.

## 3. Parameters

Names, shapes, dtypes and trainability are **[V]** against the three probe files, both parameter dtypes. The checkpoint keys are
`model/instructions/<name>/w/...`, `.../element_map_symbols/...` and `.../element_map_index/...` (here `<name> = Z`).

| Attribute | TF name | Shape | dtype | Trainable | Notes |
|---|---|---|---|---|---|
| `w` | `Z/ChemicalEmbedding:0` | `[n_elements, embedding_size]`, `[89, 128]` | `float_dtype` | `is_trainable` | created in `build` (`compute.py:1027-1066`) |
| `element_map_symbols` | `element_map_symbols:0` | `[n_elements]` | `string` | no | created in `__init__` (`compute.py:1009-1011`), so it exists before `build` and carries no `Z/` scope |
| `element_map_index` | `element_map_index:0` | `[n_elements]` | `int32` | no | same (`compute.py:1012-1014`) |

The two element-map variables are checkpoint data, not derivable constants: the twin reads them (rule R6;
strings go through the JSON metadata, indices through the `npz`).

## 4. Runtime constants

None.

## 5. Forward

`compute.py:1068-1074`:

```
Z = w                                   (LoRA inactive)
```

No gather, no input. The instruction writes the table under its name; the dependants gather.
(With an active LoRA the output is `w + lora_reconstruction(...)`, section 7.)

## 6. Dtype and promotion

The output has the parameter dtype (`float32` or `float64`). Consumers cast it to the dtype of the radial
function before use (`SingleParticleBasisFunctionScalarInd`: `tf.cast(z, r.dtype)`, `compute.py:1239`).

## 7. Options rejected

- `lora_config` not `None`, as for the radial MLPs (`compute.py:1062-1063`, `1115-1123`): **Proposal** the twin
  raises and says to merge first. `enable_lora_adaptation` with an unknown mode leaves a half-enabled instruction
  (TEST6 finding 12, pinned at `tests/test_compute.py:1908`); it is a training-side defect.
- `get_index_to_select`, `prepare_variables_for_selected_elements`, `upd_init_args_new_elements`
  (`compute.py:1076-1114`) restrict a model to a subset of elements. Decision D19: the model loads with its full
  element set, so none of them is part of the twin.
- `init` and `is_trainable` are accepted and ignored by the twin (no effect on a loaded model).

## 8. Golden-fixture keys

Fixture key `Z`, `[89, 128]`. It is a copy of a weight, so a fixture is exact (error 0) by construction. The
oracle of `tests/test_compute.py:1744-2254` (exact values of the `zeros` and `delta` initialisers, the standard
deviation of `random`, row selection) covers the TF side. FIX2 produces the yaml fixtures; no achieved error
exists yet.

## For the reviewer

None beyond the family-wide proposals of the README (reject `lora_config`).
