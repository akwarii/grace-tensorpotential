---
paths:
  - "tensorpotential/instructions/**"
  - "tensorpotential/tpmodel.py"
  - "tensorpotential/export.py"
  - "tensorpotential/metadata_utils.py"
  - "tensorpotential/torch_backend/spec/**"
  - "docs/torch_backend/spec/**"
  - "tests/test_registry.py"
  - "tests/test_instructions.py"
  - "tests/test_compute.py"
  - "tests/test_output.py"
  - "tests/test_base.py"
---

# Instruction classes, model.yaml and the registry

*Loaded when you read or edit a matching file (`paths` above); the rules that always apply are in `CLAUDE.md`.*

- `capture_init_args` writes constructor defaults into the saved `model.yaml`: a default is part of every saved model, so changing a constructor signature or default changes existing models (persisted API). It is a **class** decorator.
- **A new `TPInstruction` subclass needs a registry entry**, or `tests/test_registry.py` and `tools/instruction_ast.py check` fail (they also fail on a pinned default that differs from the source); how, and the scheduled upstream job: `docs/torch_backend/spec/README.md` (SPEC5).
  A class that inherits its constructor reports the parent's parameters.
- Output instructions overwrite `input_data[target.name]` **in place** (`instructions/output.py`); instruction order in the yaml matters and a dump must snapshot the key before and after.
- `tpmodel.__getattr__` and `instructions.__getattr__` look unused but saved yamls import classes through them; dead-code tools flag them, keep them.
- `lm_first=True` fails on yamls whose output is `MLPOut2ScalarTarget` (`model_grace.yaml`). `CollectInvarBasis` cannot be instantiated as shipped (abstract `upd_init_args_new_elements` missing); tests subclass it with a stub. `ConstantScaleShiftTarget` sorts `atomic_shift_map` by key, so keys must be element indices.

```bash
uv run --frozen --no-sync python tools/instruction_ast.py check   # every TPInstruction subclass has a registry entry; pinned defaults equal the source
```
