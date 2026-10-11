---
paths:
  - "tensorpotential/scripts/**"
  - "tensorpotential/cli/**"
  - "tensorpotential/uq/cli/**"
  - "tests/cli_help/**"
  - "tests/cli_help.py"
  - "tests/test_cli_help.py"
  - "docs/gracemaker/cli.md"
---

# The `grace` command and the console tools

*Loaded when you read or edit a matching file (`paths` above); the rules that always apply are in `CLAUDE.md`.*

- **The `grace` command and the console tools (CLI1).** `tensorpotential/scripts/grace.py` is the one table (`COMMANDS`, `UQ_COMMANDS`: name, module, function, one-line help, backend tag) and the dispatch; it is TF-free
  (import contract, `TF_FREE_MODULES`) and checks the backend of a row (`require_backend`) before it imports the module. A new console command is a row there, a `[project.scripts]` entry, and a `main(argv=None, *, prog=...)` that
  returns the exit code (every entry function takes `argv` first and a keyword `prog`: the usage line reads `grace uq build` through `grace`, `grace_uq build` through the old name; `tests/test_cli_help.py` checks the contract).
  The `--help` text of every command and sub-command is committed in `tests/cli_help/` (compared after folding the Python 3.13 option format and the wrapping); regenerate only from an unmodified tree (`python -m tests.cli_help write`).
  Output goes through `cli/console.py`: rich only styles (a message is a `Text`, never markup, never wrapped), so a pipe gets the same text as the `print` it replaced; a test that needs a terminal replaces `console.get_console`
  with a `make_console(force_terminal=True)` on a `StringIO`. The prompt helpers are in `cli/prompts.py` (`cli.wizard` re-exports the old names), so a test that fakes `_ask_*` for the dataset question patches `cli.prompts`, and a test of
  `extra/gen_tensor/wizard.py` patches both modules or types answers into `input()`.
