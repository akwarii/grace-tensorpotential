"""Size budgets of the files that agents load into their context.

``CLAUDE.md`` and the unscoped parts of ``.claude/`` are read at the start of every session; the skills and the
path-scoped rules are read on demand. Each file has a budget a little above its size when it was last compacted,
so a file cannot grow unnoticed. When a budget is hit, move what is tied to one module, tool or issue next to that
code (the section "Adding to this file" of ``CLAUDE.md``), delete what is fixed, or compact. Raising a budget is
a decision of the user, not of the agent that hit it.
"""

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CLAUDE_MD_BUDGET = 21_000
SKILL_BUDGET = 11_500
RULE_BUDGET = 3_200
TOTAL_BUDGET = 84_000  # CLAUDE.md, every skill and every rule together

FILES = [REPO / "CLAUDE.md", *sorted((REPO / ".claude" / "skills").glob("*/SKILL.md")), *sorted((REPO / ".claude" / "rules").glob("*.md"))]


def budget_of(path: Path) -> int:
    """Return the byte budget of an agent file."""
    if path.name == "CLAUDE.md":
        return CLAUDE_MD_BUDGET
    return SKILL_BUDGET if path.name == "SKILL.md" else RULE_BUDGET


def test_the_agent_files_exist():
    assert REPO / "CLAUDE.md" in FILES
    assert any(p.name == "SKILL.md" for p in FILES)
    assert any(p.parent.name == "rules" for p in FILES)


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(REPO)))
def test_an_agent_file_stays_within_its_budget(path):
    size = path.stat().st_size
    assert size <= budget_of(path), (
        f"{path.relative_to(REPO)} is {size} bytes, budget {budget_of(path)}: compact it or move what is tied to one "
        "module or issue next to that code (CLAUDE.md, 'Adding to this file'); raising the budget needs the user's say-so"
    )


def test_the_agent_files_together_stay_within_the_total_budget():
    total = sum(p.stat().st_size for p in FILES)
    assert total <= TOTAL_BUDGET, f"the agent files total {total} bytes, budget {TOTAL_BUDGET}"
