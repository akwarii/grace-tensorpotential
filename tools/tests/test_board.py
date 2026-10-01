"""Logic tests for tools/board.py.

`gh` (the GitHub command line, an external network boundary) is replaced by an in-memory fake that behaves like the few `gh` calls the
tool makes; everything else, including the sanitiser, the Definition-of-Done editing, the lint and the pull-request checks, is the real code.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import board

REPO_ROOT = Path(__file__).resolve().parents[2]


def issue_body(deps="none", score=81, blocks="none", extra=""):
    return f"""**Depends on:** {deps} · **Blocks:** {blocks} · **Related:** none

## Context

- Stage 0: Cleanup
- Priority: P0 (score {score}); slack 0 days; on the critical path

## Work

Do the work.{extra}

## Exit criterion

It works.

## Out of scope

- The Work of any other issue.

## Decisions it depends on

None specific to this issue.

## Definition of Done

Tick each box as soon as a task of this issue satisfies it.

- [ ] Exit criterion met
- [ ] Tests written
- [ ] A pull request referencing this issue (`Refs #{{n}}`) is merged into `torch-backend` (the issue is resolved only then)

---
Background: the `Decisions` issue.
"""


class FakeGH:
    """In-memory stand-in for the `gh` calls of board.py."""

    def __init__(self):
        self.calls = []
        self.comments = []
        self.closed = []
        self.issues = [
            {
                "number": 1,
                "title": "Decisions — D1 to D19 tracker",
                "body": "_Written for the project owner: you means the owner._\n- [x] D1",
                "state": "OPEN",
            },
            {
                "number": 2,
                "title": "M0.1 — Safety net",
                "body": issue_body(),
                "state": "OPEN",
            },
            {
                "number": 3,
                "title": "M0.2 — Triage",
                "body": issue_body("#2", 60, "#4"),
                "state": "OPEN",
            },
            {
                "number": 4,
                "title": "M0.3 — Remove code",
                "body": issue_body("#3", 70),
                "state": "OPEN",
            },
            {
                "number": 5,
                "title": "Appendix A — Measurements",
                "body": "reference text",
                "state": "OPEN",
            },
            {
                "number": 6,
                "title": "G0 — Stage 0 gate",
                "body": "**Depends on:** #4 · **Blocks:** none · **Related:** none\n\n## Pass criterion\n\nok\n",
                "state": "OPEN",
            },
        ]
        self.prs = [
            {
                "number": 10,
                "title": "M0.1 - safety net",
                "body": "Refs #2",
                "baseRefName": "torch-backend",
                "mergedAt": "2026-10-01T10:00:00Z",
                "state": "MERGED",
                "url": "u10",
            },
            {
                "number": 11,
                "title": "M0.2 triage",
                "body": "Refs #3",
                "baseRefName": "torch-backend",
                "mergedAt": None,
                "state": "OPEN",
                "url": "u11",
            },
            {
                "number": 12,
                "title": "other base",
                "body": "Refs #4",
                "baseRefName": "master",
                "mergedAt": "2026-10-01T11:00:00Z",
                "state": "MERGED",
                "url": "u12",
            },
        ]
        self.status = {
            "M0.1": "Todo",
            "M0.2": "Todo",
            "M0.3": "Todo",
            "G0": "Todo",
            "Decisions": "Todo",
        }
        self.options = {
            "opt-todo": "Todo",
            "opt-prog": "In Progress",
            "opt-pr": "PR Open",
            "opt-done": "Done",
        }

    def issue(self, number):
        return next(i for i in self.issues if i["number"] == number)

    def __call__(self, *args):
        self.calls.append(args)
        head = args[:2]
        if head == ("issue", "list"):
            return json.dumps(self.issues)
        if head == ("issue", "comment"):
            self.comments.append((int(args[2]), args[args.index("--body") + 1]))
            return ""
        if head == ("issue", "edit"):
            self.issue(int(args[2]))["body"] = Path(
                args[args.index("--body-file") + 1]
            ).read_text()
            return ""
        if head == ("issue", "close"):
            self.closed.append(int(args[2]))
            return ""
        if head == ("issue", "view"):
            return f"issue {args[2]} with comments"
        if head == ("project", "list"):
            return json.dumps({
                "projects": [
                    {"title": "Other", "number": 1, "id": "P0"},
                    {"title": "GRACE torch backend", "number": 2, "id": "P2"},
                ]
            })
        if head == ("project", "field-list"):
            options = [{"name": n, "id": i} for i, n in self.options.items()]
            return json.dumps({
                "fields": [
                    {"name": "Title", "id": "f0"},
                    {"name": "Status", "id": "fs", "options": options},
                ]
            })
        if head == ("project", "item-list"):
            items = []
            for issue in self.issues:
                key = board.item_id_of(str(issue["title"]))
                if key in self.status:
                    items.append({
                        "title": issue["title"],
                        "id": f"item-{key}",
                        "status": self.status[key],
                    })
            return json.dumps({"items": items})
        if head == ("api", "graphql"):
            query = args[3]
            item_match = re.search(r'itemId:"item-([^"]+)"', query)
            option_match = re.search(r'singleSelectOptionId:"([^"]+)"', query)
            assert item_match and option_match
            self.status[item_match.group(1)] = self.options[option_match.group(1)]
            return ""
        if head == ("pr", "list"):
            state = args[args.index("--state") + 1]
            if state == "merged":
                return json.dumps([p for p in self.prs if p["mergedAt"]])
            if state == "open":
                return json.dumps([p for p in self.prs if p["state"] == "OPEN"])
            return json.dumps(self.prs)
        raise AssertionError(f"unexpected gh call: {args}")


@pytest.fixture
def fake(monkeypatch):
    f = FakeGH()
    monkeypatch.setattr(board, "gh", f)  # external boundary: the GitHub CLI
    monkeypatch.setattr(board, "private_terms", lambda: [])
    return f


def ns(**kw):
    return argparse.Namespace(**kw)


# ---------------------------------------------------------------- sanitiser


def test_sanitise_keeps_known_issue_numbers_and_neutralises_others(capsys):
    out = board.sanitise("see #12 and #9999", known={12})
    assert out == "see #12 and No. 9999"
    assert "#N that is not an issue" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mail a.b@example.org now", "mail <email> now"),
        ("at /home/someone/work/file.py, then", "at <local path>, then"),
        ("see ~/work/notes.md", "see <local path>"),
        ("ping @someone", "ping @​someone"),
        ("https://github.com/ICAMS/grace-tensorpotential/pull/34", "(link removed)"),
        ("ICAMS/grace-tensorpotential#34", "(link removed)"),
        ("run `/tmp/scratch/x` here", "run `<local path>` here"),
    ],
)
def test_sanitise_rules(text, expected):
    assert board.sanitise(text) == expected


def test_sanitise_leaves_code_spans_and_clean_text_alone(capsys):
    text = "use `@decorator` and `#12` here and the number 7 and a/b paths"
    assert board.sanitise(text, known=set()) == text.replace("`#12`", "`#12`")
    assert capsys.readouterr().err == ""


def test_sanitise_does_not_mutate_its_input():
    text = "x @someone y"
    board.sanitise(text)
    assert text == "x @someone y"


def test_private_terms_are_read_from_the_git_dir_and_applied(
    tmp_path, monkeypatch, capsys
):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".git" / "board-private-terms.txt").write_text(
        "# comment\nSecretWord => generic\nlonger secret => other\n"
    )
    monkeypatch.chdir(tmp_path)
    assert board.private_terms() == [
        ("longer secret", "other"),
        ("SecretWord", "generic"),
    ]
    assert board.sanitise("a secretword and a Longer Secret") == "a generic and a other"
    assert "private term generalised" in capsys.readouterr().err


def test_private_terms_absent_file_or_no_repository(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # not a git repository
    assert board.private_terms() == []
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    assert board.private_terms() == []  # repository without the file


def test_icams_repository_is_refused():
    r = subprocess.run(
        [sys.executable, "-c", "import board"],
        cwd=Path(board.__file__).parent,
        env={"BOARD_REPO": "ICAMS/grace-tensorpotential", "PATH": ""},
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    assert "refusing to act on an ICAMS repository" in r.stderr


def test_gh_failure_exits_with_the_error(monkeypatch):
    monkeypatch.undo()  # use the real gh wrapper with a failing command
    monkeypatch.setattr(
        board.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "boom"),
    )
    with pytest.raises(SystemExit, match="boom"):
        board.gh("issue", "list", "--repo", "x")


def test_gh_success_returns_stripped_output(monkeypatch):
    monkeypatch.setattr(
        board.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, " out \n", ""),
    )
    assert board.gh("x") == "out"


# ---------------------------------------------------------------- reading the board


def test_item_ids():
    assert board.item_id_of("M0.1 — Safety net") == "M0.1"
    assert board.item_id_of("Decisions — D1 to D19 tracker") == "Decisions"
    assert board.item_id_of("Appendix E — pandas") == "Appendix E"


def test_issues_parse_dependencies_and_scores(fake):
    info = board.issues()
    assert info["M0.2"]["needs"] == ["M0.1"]
    assert info["M0.2"]["score"] == 60
    assert info["M0.1"]["needs"] == []
    assert info["Appendix A"]["score"] == 0
    assert info["G0"]["needs"] == ["M0.3"]


def test_resolve_is_case_insensitive_and_rejects_unknown(fake):
    info = board.issues()
    assert board.resolve("m0.1", info) == "M0.1"
    with pytest.raises(SystemExit, match="unknown id"):
        board.resolve("M9.9", info)


def test_project_and_statuses(fake):
    number, proj, field = board.project()
    assert (number, proj, field["name"]) == ("2", "P2", "Status")
    assert board.statuses()["M0.1"] == "Todo"


def test_project_missing(fake, monkeypatch):
    monkeypatch.setattr(board, "PROJECT_TITLE", "No such project")
    with pytest.raises(SystemExit, match="not found"):
        board.project()


def test_set_status_updates_the_board_and_validates(fake, capsys):
    board.set_status("M0.1", "In Progress")
    assert fake.status["M0.1"] == "In Progress"
    assert "Status -> In Progress" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="status must be one of"):
        board.set_status("M0.1", "Finished")


def test_known_numbers_include_issues_and_pull_requests(fake):
    assert {1, 2, 3, 10, 11, 12} <= board.known_numbers(board.issues())


def test_merged_pr_for_requires_merge_into_the_integration_branch(fake):
    info = board.issues()
    pr = board.merged_pr_for(info, "M0.1")
    assert pr is not None and pr["number"] == 10
    assert board.merged_pr_for(info, "M0.2") is None  # its PR is still open
    assert board.merged_pr_for(info, "M0.3") is None  # merged, but into another branch
    explicit = board.merged_pr_for(info, "M0.1", explicit=10)
    assert explicit is not None and explicit["number"] == 10
    assert board.merged_pr_for(info, "M0.1", explicit=99) is None


def test_merged_pr_reference_must_match_the_issue_number_exactly(fake):
    fake.prs[0]["body"] = "Refs #22 and #20"
    assert board.merged_pr_for(board.issues(), "M0.1") is None  # #2 is not #22 or #20


# ---------------------------------------------------------------- posting


def test_comment_is_sanitised_before_posting(fake):
    info = board.issues()
    board.comment(info, "M0.1", "see /home/x/y and #9999")
    number, text = fake.comments[-1]
    assert number == 2
    assert "<local path>" in text
    assert "No. 9999" in text


def test_put_body_is_sanitised_and_written(fake):
    info = board.issues()
    board.put_body(info, "M0.1", "new body for @someone")
    assert fake.issue(2)["body"] == "new body for @​someone"


# ---------------------------------------------------------------- Definition of Done


def test_dod_lines_only_inside_the_section():
    body = issue_body()
    lines = body.split("\n")
    idx = board.dod_lines(body)
    assert len(idx) == 3
    assert all(lines[i].startswith("- [") for i in idx)
    assert board.dod_lines("no checklist here") == []
    assert board.dod_lines(
        "## Definition of Done\n\n- [ ] one\n## Next\n- [ ] not counted"
    ) == [2]


def test_set_boxes_ticks_unticks_and_marks_not_applicable(fake, capsys):
    info = board.issues()
    board.set_boxes(info, "M0.1", [1], True)
    assert (
        board.dod_lines(fake.issue(2)["body"])
        and "- [x] Exit criterion met" in fake.issue(2)["body"]
    )
    info = board.issues()
    board.set_boxes(info, "M0.1", [1], False)
    assert "- [ ] Exit criterion met" in fake.issue(2)["body"]
    info = board.issues()
    board.set_boxes(info, "M0.1", [2], True, na_reason="no code")
    assert "- [x] ~~Tests written~~ (not applicable: no code)" in fake.issue(2)["body"]
    assert "as not applicable" in capsys.readouterr().out


def test_set_boxes_rejects_bad_numbers_and_missing_checklists(fake):
    info = board.issues()
    with pytest.raises(SystemExit, match="box number must be 1..3"):
        board.set_boxes(info, "M0.1", [4], True)
    with pytest.raises(SystemExit, match="no Definition of Done"):
        board.set_boxes(info, "Appendix A", [1], True)


def test_cmd_dod_prints_numbered_boxes(fake, capsys):
    board.cmd_dod(ns(id="m0.1"))
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("1. - [ ] Exit criterion met")
    assert len(out) == 3


def test_cmd_check_with_note_comments_the_evidence(fake):
    board.cmd_check(ns(id="M0.1", numbers=[1, 2], note="command and result"))
    assert "- [x] Exit criterion met" in fake.issue(2)["body"]
    assert any("command and result" in t for _, t in fake.comments)


def test_cmd_check_without_note_does_not_comment(fake):
    board.cmd_check(ns(id="M0.1", numbers=[1], note=None))
    assert fake.comments == []


def test_cmd_na_and_cmd_uncheck(fake):
    board.cmd_na(ns(id="M0.1", number=2, reason="irrelevant"))
    assert "not applicable: irrelevant" in fake.issue(2)["body"]
    board.cmd_check(ns(id="M0.1", numbers=[1], note=None))
    board.cmd_uncheck(ns(id="M0.1", numbers=[1]))
    assert "- [ ] Exit criterion met" in fake.issue(2)["body"]


def test_pr_box_number():
    assert board.pr_box_number(issue_body()) == 3
    assert board.pr_box_number("## Definition of Done\n\n- [ ] a\n") is None


# ---------------------------------------------------------------- protocol commands


def test_cmd_next_lists_only_ready_items_best_first(fake, capsys):
    board.cmd_next(ns())
    out = capsys.readouterr().out
    assert "M0.1" in out
    assert "M0.2" not in out  # blocked by M0.1, which is not Done
    fake.status["M0.1"] = "Done"
    board.cmd_next(ns())
    assert capsys.readouterr().out.split()[0] == "M0.2"


def test_cmd_next_when_nothing_is_ready(fake, capsys):
    for k in fake.status:
        fake.status[k] = "Done"
    board.cmd_next(ns())
    assert "no ready item" in capsys.readouterr().out


def test_cmd_list_filters_by_status(fake, capsys):
    fake.status["M0.1"] = "In Progress"
    board.cmd_list(ns(status="In Progress"))
    out = capsys.readouterr().out
    assert "M0.1" in out and "M0.2" not in out
    board.cmd_list(ns(status=None))
    assert "M0.2" in capsys.readouterr().out


def test_cmd_context_prints_the_issue_and_its_dependencies(fake, capsys):
    board.cmd_context(ns(id="M0.2"))
    out = capsys.readouterr().out
    assert "issue 3 with comments" in out and "issue 2 with comments" in out


def test_cmd_start_refuses_a_blocked_issue_unless_forced(fake, capsys):
    with pytest.raises(SystemExit, match="blocked by"):
        board.cmd_start(ns(id="M0.2", force=False))
    board.cmd_start(ns(id="M0.2", force=True))
    assert fake.status["M0.2"] == "In Progress"
    assert "forced" in fake.comments[-1][1]


def test_cmd_start_moves_a_ready_issue_to_in_progress(fake):
    board.cmd_start(ns(id="M0.1", force=False))
    assert fake.status["M0.1"] == "In Progress"
    assert fake.comments[-1][0] == 2


def test_cmd_finding_comments_on_the_issue_and_the_affected_ones(fake):
    board.cmd_finding(ns(id="M0.1", text="a surprise", also=["M0.2", "G0"]))
    assert [n for n, _ in fake.comments] == [2, 3, 6]
    assert all("a surprise" in t for _, t in fake.comments)
    board.cmd_finding(ns(id="M0.1", text="another", also=None))
    assert fake.comments[-1][0] == 2


def test_cmd_status_refuses_done_and_requires_an_open_pr_for_pr_open(fake):
    with pytest.raises(SystemExit, match="use `done`"):
        board.cmd_status(ns(id="M0.1", status="Done"))
    with pytest.raises(SystemExit, match="No open pull request"):
        board.cmd_status(ns(id="M0.1", status="PR Open"))  # its PR is merged, not open
    board.cmd_status(ns(id="M0.2", status="PR Open"))  # PR 11 is open and references #3
    assert fake.status["M0.2"] == "PR Open"
    board.cmd_status(ns(id="M0.2", status="In Progress"))
    assert fake.status["M0.2"] == "In Progress"


def test_cmd_done_refuses_without_a_merged_pr(fake):
    with pytest.raises(SystemExit, match="No pull request referencing #3 is merged"):
        board.cmd_done(ns(id="M0.2", evidence="e", waive=None, pr=None, waive_pr=None))
    assert fake.closed == []


def test_cmd_done_with_a_merged_pr_ticks_the_pr_box_then_needs_the_rest(fake):
    with pytest.raises(SystemExit, match="boxes still open: \\[1, 2\\]"):
        board.cmd_done(ns(id="M0.1", evidence="e", waive=None, pr=None, waive_pr=None))
    assert "- [x] A pull request referencing this issue" in fake.issue(2)["body"]
    assert fake.closed == []


def test_cmd_done_closes_the_issue_when_everything_is_satisfied(fake):
    board.cmd_check(ns(id="M0.1", numbers=[1, 2], note=None))
    board.cmd_done(
        ns(id="M0.1", evidence="all green", waive=None, pr=None, waive_pr=None)
    )
    assert fake.closed == [2]
    assert fake.status["M0.1"] == "Done"
    assert "Merged pull request: #10" in fake.comments[-1][1]


def test_cmd_done_with_waivers_records_the_reasons(fake):
    board.cmd_done(
        ns(
            id="M0.2",
            evidence="gate passed",
            waive="not applicable here",
            pr=None,
            waive_pr="owner said so",
        )
    )
    texts = [t for _, t in fake.comments]
    assert any(
        "Closed without a merged pull request. Reason: owner said so" in t
        for t in texts
    )
    assert any(
        "boxes [1, 2] open. Reason: not applicable here" in t or "boxes [1, 2, 3]" in t
        for t in texts
    ) or any("Closed with Definition of Done" in t for t in texts)
    assert fake.closed == [3]


def test_cmd_done_for_an_issue_without_a_checklist_with_waived_pr(fake):
    board.cmd_done(
        ns(id="G0", evidence="criterion met", waive=None, pr=None, waive_pr="gate")
    )
    assert fake.closed == [6]


# ---------------------------------------------------------------- lint


def lint(fake, capsys):
    try:
        board.cmd_lint(ns())
        code = 0
    except SystemExit as e:
        code = e.code
    return code, capsys.readouterr().out


def test_lint_clean_issues_pass(fake, capsys):
    for n in (2, 3, 4):
        fake.issue(n)["body"] = fake.issue(n)["body"].replace("{n}", str(n))
    code, out = lint(fake, capsys)
    assert code == 0 and "0 stale finding(s)" in out


def test_lint_reports_each_kind_of_stale_text(fake, capsys):
    fake.issue(2)["body"] = issue_body(
        extra=" See plan/ and M9.9, Appendix K and #777; 79 milestone issues; your go."
    )
    fake.issue(3)["body"] = fake.issue(3)["body"].replace("## Out of scope", "## Scope")
    fake.issue(4)["body"] = fake.issue(4)["body"].replace(
        "merged into `torch-backend`", "merged"
    )
    code, out = lint(fake, capsys)
    assert code == 1
    for expected in (
        "mentions removed `plan/`",
        "unknown id M9.9",
        "cites Appendix K",
        "unknown issue #777",
        "missing '## Out of scope'",
        "lacks the merged-PR box",
        "an old issue or milestone count",
        'second person ("your")',
    ):
        assert expected in out


def test_lint_second_person_is_fine_when_the_note_says_so(fake, capsys):
    fake.issue(1)["body"] += " your"
    code, out = lint(fake, capsys)
    assert "Decisions" not in "".join(
        line for line in out.splitlines() if "second person" in line
    )


# ---------------------------------------------------------------- pull-request text


def test_section_extraction():
    text = "## A\n\nfirst\n\n## B\n\nsecond\n"
    assert board.section(text, "A") == "first"
    assert board.section(text, "B") == "second"
    assert board.section(text, "C") == ""


def test_cmd_pr_body_fills_refs_exit_criterion_and_checklist(fake, capsys):
    board.cmd_pr_body(ns(id="M0.1"))
    captured = capsys.readouterr()
    assert "Refs #2" in captured.out
    assert "Exit criterion of the issue: It works." in captured.out
    assert "- [ ] Tests written" in captured.out
    assert "<!--" not in captured.out
    assert "Suggested PR title: M0.1 — Safety net" in captured.err


def filled_pr_body(fake, capsys, ident="M0.1"):
    board.cmd_pr_body(ns(id=ident))
    text = capsys.readouterr().out
    number = fake.issue(2)["number"]
    text = text.replace(
        f"Refs #{number}\n",
        f"Adds the safety net of the clean-up so that later changes can be compared with the untouched tree and shown to be neutral.\n\nRefs #{number}\n",
        1,
    )
    text = re.sub(
        r"## What changed\n+-\n",
        "## What changed\n\n- three small tools; behaviour-preserving\n",
        text,
        count=1,
    )
    text = text.replace("| | | |", "| tests | `pytest tools` | 40 passed |")
    for heading, content in (
        ("Unexpected findings", "none"),
        ("Review focus", "the comparison tolerances"),
        ("Risks and rollback", "revert the pull request"),
    ):
        text = re.sub(
            rf"## {heading}\n\n", f"## {heading}\n\n{content}\n\n", text, count=1
        )
    return text


def test_cmd_pr_check_accepts_a_filled_description(fake, capsys, tmp_path):
    f = tmp_path / "body.md"
    f.write_text(filled_pr_body(fake, capsys))
    board.cmd_pr_check(ns(id="M0.1", file=str(f)))
    assert "0 problem(s)" in capsys.readouterr().out


def test_cmd_pr_check_reports_every_rule(fake, capsys, tmp_path):
    f = tmp_path / "body.md"
    f.write_text(
        "<!-- left over -->\n## Summary\n\nshort\n\nCloses #2\n\n## What changed\n\n-\n\n## Evidence\n\n| | | |\n\nsee /home/someone/x\n"
    )
    with pytest.raises(SystemExit):
        board.cmd_pr_check(ns(id="M0.1", file=str(f)))
    out = capsys.readouterr().out
    for expected in (
        "missing `Refs #2`",
        "uses Closes/Fixes/Resolves",
        "template comments are still present",
        "section 'What changed' is missing or empty",
        "section 'Unexpected findings' is missing or empty",
        "the Summary is too short",
        "the Evidence table has an empty row",
        "text is not sanitised",
    ):
        assert expected in out


def test_cmd_sanitise_reads_a_file(fake, capsys, tmp_path):
    f = tmp_path / "t.md"
    f.write_text("see #2 and #9999 and @someone")
    board.cmd_sanitise(ns(file=str(f)))
    assert capsys.readouterr().out == "see #2 and No. 9999 and @​someone"


# ---------------------------------------------------------------- command line


@pytest.mark.parametrize(
    "argv",
    [
        ["list"],
        ["next"],
        ["dod", "M0.1"],
        ["check", "M0.1", "1", "--note", "n"],
        ["uncheck", "M0.1", "1"],
        ["na", "M0.1", "2", "why"],
        ["start", "M0.1", "--force"],
        ["finding", "M0.1", "text", "--also", "M0.2"],
        ["status", "M0.2", "PR Open"],
        ["context", "M0.1"],
        ["lint"],
    ],
)
def test_main_dispatches_every_command(fake, monkeypatch, capsys, argv):
    monkeypatch.setattr(sys, "argv", ["board.py", *argv])
    try:
        board.main()
    except SystemExit as e:  # lint may report findings on the tiny fixture
        assert argv[0] == "lint" and e.code == 1


def test_main_done_and_pr_commands(fake, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "board.py",
            "done",
            "G0",
            "evidence",
            "--waive-pr",
            "gate",
            "--waive",
            "no list",
        ],
    )
    board.main()
    assert fake.closed == [6]
    body = tmp_path / "b.md"
    body.write_text("x")
    monkeypatch.setattr(sys, "argv", ["board.py", "pr-check", "M0.1", str(body)])
    with pytest.raises(SystemExit):
        board.main()
    monkeypatch.setattr(sys, "argv", ["board.py", "pr-body", "M0.1"])
    board.main()
    monkeypatch.setattr(sys, "argv", ["board.py", "sanitise", str(body)])
    board.main()
    assert REPO_ROOT.joinpath(
        ".github", "PULL_REQUEST_TEMPLATE", "torch-backend.md"
    ).exists()
