---
paths:
  - "tests_torch/golden/**"
  - "tests_torch/fixtures/**"
  - "tools/make_golden.py"
  - "tools/tests/test_make_golden.py"
---

# Golden fixtures

*Loaded when you read or edit a matching file (`paths` above); the rules that always apply are in `CLAUDE.md`.*

- Golden fixtures: the library imported is the one on `PYTHONPATH` (read the `library:` line the generator prints); commit the generator before writing fixtures (the tests fail on a modified tree); never write the manifest with sorted keys. Other pitfalls: `tests_torch/golden/README.md`.
- The tolerance table `tests_torch/golden/tolerances.py` and every committed golden file are edit-locked (`LOCK.sha256`): after a regeneration or a row change, `python -m tests_torch.golden.lock record` in a commit of its own that says why; never widen a row to make a test pass.

```bash
# tests_torch/structures/structures.json/.xyz are written by python -m tests_torch.structures.build_structures (test_build_structures.py fails on a difference).
# The generator is the only thing allowed to write fixtures, from a committed tree (tiers, verify, compare, the generator's tests: tests_torch/golden/README.md)
uv run --frozen --no-sync python tools/make_golden.py write --tier tiny   # tests_torch/golden/ (committed); --tier faithful: tests_torch/fixtures/ (npz git-ignored)
```
