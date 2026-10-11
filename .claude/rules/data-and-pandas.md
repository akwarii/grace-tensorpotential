---
paths:
  - "tensorpotential/data/**"
  - "tensorpotential/calculator/**"
  - "tensorpotential/cli/data.py"
  - "tests/test_databuilder.py"
  - "tests/test_utils.py"
  - "tests/test_data.py"
  - "tests/neighbour_oracle.py"
  - "pyproject.toml"
---

# Data builder, neighbour lists and pandas

*Loaded when you read or edit a matching file (`paths` above); the rules that always apply are in `CLAUDE.md`.*

- `GeometricalDataBuilder.extract_from_ase_atoms` edits its argument (`enforce_pbc` gives a non-periodic `Atoms` a cell, centres it, sets `pbc=True`, also overriding `pbc=(T,T,F)`): pass a copy when the structure is reused. An atom with no neighbour gets one dummy bond `(i, 0)` far outside the cutoff; a zero third cell vector with partial `pbc` raises `LinAlgError`.
  Pinned by `tests/test_databuilder.py` and `tests/test_utils.py`; brute-force reference `tests/neighbour_oracle.py`.
- `pandas<3` stays pinned (decision D11) although the suite is green on 3.0.3 and 2.3.3; pickles written with pandas 3 cannot be read by pandas 2. A second pandas environment: `baselines/README.md`, with its `bin` on `PATH` (tests start `grace_preprocess` by name).
