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


def issue_body(deps="none", score=81, blocks="none", extra="", legacy=None):
    legacy_line = f"- Legacy id: {legacy}\n" if legacy else ""
    return f"""**Depends on:** {deps} · **Blocks:** {blocks} · **Related:** none

## Context

- Stage 0: Cleanup
{legacy_line}- Priority: P0 (score {score}); slack 0 days; on the critical path

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
                "title": "SAFE1 — Safety net",
                "body": issue_body(legacy="M0.1"),
                "state": "OPEN",
            },
            {
                "number": 3,
                "title": "CLEAN1 — Triage",
                "body": issue_body("#2", 60, "#4", legacy="M0.2"),
                "state": "OPEN",
            },
            {
                "number": 4,
                "title": "CLEAN2 — Remove code",
                "body": issue_body("#3", 70, legacy="M0.3"),
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
                "title": "GATE-CLEAN — Stage 0 gate",
                "body": "**Depends on:** #4 · **Blocks:** none · **Related:** none\n\n## Context\n\n- Legacy id: G0\n\n## Pass criterion\n\nok\n",
                "state": "OPEN",
            },
        ]
        self.prs = [
            {
                "number": 10,
                "title": "SAFE1 - safety net",
                "body": "Refs #2",
                "baseRefName": "torch-backend",
                "mergedAt": "2026-10-01T10:00:00Z",
                "state": "MERGED",
                "url": "u10",
            },
            {
                "number": 11,
                "title": "CLEAN1 triage",
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
            "SAFE1": "Todo",
            "CLEAN1": "Todo",
            "CLEAN2": "Todo",
            "GATE-CLEAN": "Todo",
            "Decisions": "Todo",
        }
        self.text_fields = False  # whether the project has the Blocked by / PR columns
        self.text = {}  # (item key, column name) -> text
        self.cell_writes = 0
        self.page_size = 100  # items per page of the table query
        self.remaining = 4990  # what the rate-limit fields of the fake report
        self.draft_items = 0  # table rows that are not issues
        self.options = {
            "opt-todo": "Todo",
            "opt-prog": "In Progress",
            "opt-pr": "PR Open",
            "opt-done": "Done",
        }

    def issue(self, number):
        return next(i for i in self.issues if i["number"] == number)

    def rate(self, cost):
        return {
            "limit": 5000,
            "cost": cost,
            "remaining": self.remaining,
            "used": 5000 - self.remaining,
            "resetAt": "2099-01-01T00:00:00Z",
        }

    def items_page(self, args):
        """The response of the table query: one node per issue on the board, in pages of `page_size`."""
        kv = dict(a.split("=", 1) for a in args[3::2] if "=" in a)
        nodes = []
        for issue in self.issues:
            key = board.item_id_of(str(issue["title"]))
            if key in self.status:
                node = {
                    "id": f"item-{key}",
                    "content": {"number": issue["number"], "title": issue["title"]},
                    "status": {"name": self.status[key]},
                }
                for alias, column in (("blocked", "Blocked by"), ("pr", "PR")):
                    if (key, column) in self.text:
                        node[alias] = {"text": self.text[key, column]}
                nodes.append(node)
        nodes += [{"id": f"draft-{n}", "content": {}} for n in range(self.draft_items)]
        start = int(kv.get("cursor", 0))
        end = start + self.page_size
        page = {
            "pageInfo": {"hasNextPage": end < len(nodes), "endCursor": str(end)},
            "nodes": nodes[start:end],
        }
        data = {"rateLimit": self.rate(cost=1), "node": {"items": page}}
        return json.dumps({"data": data})

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
            fields = [
                {"name": "Title", "id": "f0"},
                {"name": "Status", "id": "fs", "options": options},
            ]
            if self.text_fields:
                fields += [
                    {"name": "Blocked by", "id": "fb"},
                    {"name": "PR", "id": "fp"},
                ]
            return json.dumps({"fields": fields})
        if head == ("api", "graphql"):
            query = args[3]
            if "items(first" in query:
                return self.items_page(args)
            if "rateLimit" in query:
                return json.dumps({"data": {"rateLimit": self.rate(cost=1)}})
            if "$field" in query:  # a text cell: set, or cleared when there is no $text
                kv = dict(a.split("=", 1) for a in args[3::2])
                key = kv["item"].removeprefix("item-")
                column = {"fb": "Blocked by", "fp": "PR"}[kv["field"]]
                if "$text" in query:
                    self.text[key, column] = kv["text"]
                else:
                    self.text.pop((key, column), None)
                self.cell_writes += 1
                return ""
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
def fake(monkeypatch, tmp_path):
    f = FakeGH()
    monkeypatch.setattr(board, "gh", f)  # external boundary: the GitHub CLI
    monkeypatch.setattr(board, "private_terms", lambda: [])
    # the project cache lives in a temporary directory, never in the real git directory
    monkeypatch.setattr(board, "git_dir_file", lambda name: tmp_path / name)
    monkeypatch.delenv("BOARD_NO_CACHE", raising=False)
    board.reset_cache()
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
    assert board.item_id_of("SAFE1 — Safety net") == "SAFE1"
    assert board.item_id_of("Decisions — D1 to D19 tracker") == "Decisions"
    assert board.item_id_of("Appendix E — pandas") == "Appendix E"


def test_issues_parse_dependencies_and_scores(fake):
    info = board.issues()
    assert info["CLEAN1"]["needs"] == ["SAFE1"]
    assert info["CLEAN1"]["score"] == 60
    assert info["SAFE1"]["needs"] == []
    assert info["Appendix A"]["score"] == 0
    assert info["GATE-CLEAN"]["needs"] == ["CLEAN2"]


def test_resolve_is_case_insensitive_and_rejects_unknown(fake):
    info = board.issues()
    assert board.resolve("safe1", info) == "SAFE1"
    with pytest.raises(SystemExit, match="unknown id"):
        board.resolve("M9.9", info)


def test_legacy_ids_resolve_to_the_new_id(fake, capsys):
    info = board.issues()
    assert info["SAFE1"]["legacy"] == "M0.1"
    assert info["Appendix A"]["legacy"] is None
    assert board.resolve("M0.2", info) == "CLEAN1"
    assert board.resolve("g0", info) == "GATE-CLEAN"
    assert "[M0.2 is now CLEAN1]" in capsys.readouterr().err
    assert board.resolve("CLEAN1", info) == "CLEAN1"
    assert capsys.readouterr().err == ""


def test_legacy_id_gives_the_same_start_as_the_new_id(fake):
    board.cmd_start(ns(id="M0.1", force=False))
    assert fake.status["SAFE1"] == "In Progress"


def test_project_and_statuses(fake):
    number, proj, field = board.project()
    assert (number, proj, field["name"]) == ("2", "P2", "Status")
    assert board.statuses()["SAFE1"] == "Todo"


def test_project_missing(fake, monkeypatch):
    monkeypatch.setattr(board, "PROJECT_TITLE", "No such project")
    with pytest.raises(SystemExit, match="not found"):
        board.project()


def test_set_status_updates_the_board_and_validates(fake, capsys):
    board.set_status("SAFE1", "In Progress")
    assert fake.status["SAFE1"] == "In Progress"
    assert "Status -> In Progress" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="status must be one of"):
        board.set_status("SAFE1", "Finished")


def test_known_numbers_include_issues_and_pull_requests(fake):
    assert {1, 2, 3, 10, 11, 12} <= board.known_numbers(board.issues())


def test_merged_pr_for_requires_merge_into_the_integration_branch(fake):
    info = board.issues()
    pr = board.merged_pr_for(info, "SAFE1")
    assert pr is not None and pr["number"] == 10
    assert board.merged_pr_for(info, "CLEAN1") is None  # its PR is still open
    assert board.merged_pr_for(info, "CLEAN2") is None  # merged, but into another branch
    explicit = board.merged_pr_for(info, "SAFE1", explicit=10)
    assert explicit is not None and explicit["number"] == 10
    assert board.merged_pr_for(info, "SAFE1", explicit=99) is None


def test_merged_pr_reference_must_match_the_issue_number_exactly(fake):
    fake.prs[0]["body"] = "Refs #22 and #20"
    assert board.merged_pr_for(board.issues(), "SAFE1") is None  # #2 is not #22 or #20


# ---------------------------------------------------------------- posting


def test_comment_is_sanitised_before_posting(fake):
    info = board.issues()
    board.comment(info, "SAFE1", "see /home/x/y and #9999")
    number, text = fake.comments[-1]
    assert number == 2
    assert "<local path>" in text
    assert "No. 9999" in text


def test_put_body_is_sanitised_and_written(fake):
    info = board.issues()
    board.put_body(info, "SAFE1", "new body for @someone")
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
    board.set_boxes(info, "SAFE1", [1], True)
    assert (
        board.dod_lines(fake.issue(2)["body"])
        and "- [x] Exit criterion met" in fake.issue(2)["body"]
    )
    info = board.issues()
    board.set_boxes(info, "SAFE1", [1], False)
    assert "- [ ] Exit criterion met" in fake.issue(2)["body"]
    info = board.issues()
    board.set_boxes(info, "SAFE1", [2], True, na_reason="no code")
    assert "- [x] ~~Tests written~~ (not applicable: no code)" in fake.issue(2)["body"]
    assert "as not applicable" in capsys.readouterr().out


def test_set_boxes_rejects_bad_numbers_and_missing_checklists(fake):
    info = board.issues()
    with pytest.raises(SystemExit, match="box number must be 1..3"):
        board.set_boxes(info, "SAFE1", [4], True)
    with pytest.raises(SystemExit, match="no Definition of Done"):
        board.set_boxes(info, "Appendix A", [1], True)


def test_cmd_dod_prints_numbered_boxes(fake, capsys):
    board.cmd_dod(ns(id="safe1"))
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("1. - [ ] Exit criterion met")
    assert len(out) == 3


def test_cmd_check_with_note_comments_the_evidence(fake):
    board.cmd_check(ns(id="SAFE1", numbers=[1, 2], note="command and result"))
    assert "- [x] Exit criterion met" in fake.issue(2)["body"]
    assert any("command and result" in t for _, t in fake.comments)


def test_cmd_check_without_note_does_not_comment(fake):
    board.cmd_check(ns(id="SAFE1", numbers=[1], note=None))
    assert fake.comments == []


def test_cmd_na_and_cmd_uncheck(fake):
    board.cmd_na(ns(id="SAFE1", number=2, reason="irrelevant"))
    assert "not applicable: irrelevant" in fake.issue(2)["body"]
    board.cmd_check(ns(id="SAFE1", numbers=[1], note=None))
    board.cmd_uncheck(ns(id="SAFE1", numbers=[1]))
    assert "- [ ] Exit criterion met" in fake.issue(2)["body"]


def test_pr_box_number():
    assert board.pr_box_number(issue_body()) == 3
    assert board.pr_box_number("## Definition of Done\n\n- [ ] a\n") is None


# ---------------------------------------------------------------- protocol commands


def test_cmd_next_lists_only_ready_items_best_first(fake, capsys):
    board.cmd_next(ns())
    out = capsys.readouterr().out
    assert "SAFE1" in out
    assert "CLEAN1" not in out  # blocked by SAFE1, which is not Done
    fake.status["SAFE1"] = "Done"
    board.reset_cache()  # the next command line starts without what this one read
    board.cmd_next(ns())
    assert capsys.readouterr().out.split()[0] == "CLEAN1"


def test_cmd_next_when_nothing_is_ready(fake, capsys):
    for k in fake.status:
        fake.status[k] = "Done"
    board.cmd_next(ns())
    assert "no ready item" in capsys.readouterr().out


def test_cmd_list_filters_by_status(fake, capsys):
    fake.status["SAFE1"] = "In Progress"
    board.cmd_list(ns(status="In Progress"))
    out = capsys.readouterr().out
    assert "SAFE1" in out and "CLEAN1" not in out
    board.cmd_list(ns(status=None))
    assert "CLEAN1" in capsys.readouterr().out


def test_cmd_context_prints_the_issue_and_its_dependencies(fake, capsys):
    board.cmd_context(ns(id="CLEAN1"))
    out = capsys.readouterr().out
    assert "issue 3 with comments" in out and "issue 2 with comments" in out


def test_cmd_start_refuses_a_blocked_issue_unless_forced(fake, capsys):
    with pytest.raises(SystemExit, match="blocked by"):
        board.cmd_start(ns(id="CLEAN1", force=False))
    board.cmd_start(ns(id="CLEAN1", force=True))
    assert fake.status["CLEAN1"] == "In Progress"
    assert "forced" in fake.comments[-1][1]


def test_cmd_start_moves_a_ready_issue_to_in_progress(fake):
    board.cmd_start(ns(id="SAFE1", force=False))
    assert fake.status["SAFE1"] == "In Progress"
    assert fake.comments[-1][0] == 2


def test_cmd_finding_comments_on_the_issue_and_the_affected_ones(fake):
    board.cmd_finding(ns(id="SAFE1", text="a surprise", also=["CLEAN1", "GATE-CLEAN"]))
    assert [n for n, _ in fake.comments] == [2, 3, 6]
    assert all("a surprise" in t for _, t in fake.comments)
    board.cmd_finding(ns(id="SAFE1", text="another", also=None))
    assert fake.comments[-1][0] == 2


def test_cmd_status_refuses_done_and_requires_an_open_pr_for_pr_open(fake):
    with pytest.raises(SystemExit, match="use `done`"):
        board.cmd_status(ns(id="SAFE1", status="Done"))
    with pytest.raises(SystemExit, match="No open pull request"):
        board.cmd_status(ns(id="SAFE1", status="PR Open"))  # its PR is merged, not open
    board.cmd_status(ns(id="CLEAN1", status="PR Open"))  # PR 11 is open and references #3
    assert fake.status["CLEAN1"] == "PR Open"
    board.cmd_status(ns(id="CLEAN1", status="In Progress"))
    assert fake.status["CLEAN1"] == "In Progress"


def test_cmd_done_refuses_without_a_merged_pr(fake):
    with pytest.raises(SystemExit, match="No pull request referencing #3 is merged"):
        board.cmd_done(ns(id="CLEAN1", evidence="e", waive=None, pr=None, waive_pr=None))
    assert fake.closed == []


def test_cmd_done_with_a_merged_pr_ticks_the_pr_box_then_needs_the_rest(fake):
    with pytest.raises(SystemExit, match="boxes still open: \\[1, 2\\]"):
        board.cmd_done(ns(id="SAFE1", evidence="e", waive=None, pr=None, waive_pr=None))
    assert "- [x] A pull request referencing this issue" in fake.issue(2)["body"]
    assert fake.closed == []


def test_cmd_done_closes_the_issue_when_everything_is_satisfied(fake):
    board.cmd_check(ns(id="SAFE1", numbers=[1, 2], note=None))
    board.cmd_done(
        ns(id="SAFE1", evidence="all green", waive=None, pr=None, waive_pr=None)
    )
    assert fake.closed == [2]
    assert fake.status["SAFE1"] == "Done"
    assert "Merged pull request: #10" in fake.comments[-1][1]


def test_cmd_done_with_waivers_records_the_reasons(fake):
    board.cmd_done(
        ns(
            id="CLEAN1",
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
        ns(id="GATE-CLEAN", evidence="criterion met", waive=None, pr=None, waive_pr="gate")
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


def test_lint_flags_legacy_ids_but_not_the_legacy_line(fake, capsys):
    for n in (2, 3, 4):
        fake.issue(n)["body"] = fake.issue(n)["body"].replace("{n}", str(n))
    fake.issue(3)["body"] = issue_body("#2", 60, "#4", extra=" After M0.1 and G0.", legacy="M0.2").replace("{n}", "3")
    code, out = lint(fake, capsys)
    assert code == 1
    assert "uses the legacy id M0.1; write SAFE1" in out
    assert "uses the legacy id G0; write GATE-CLEAN" in out
    assert "legacy id M0.2" not in out  # the `- Legacy id:` line itself is allowed


def test_lint_flags_unknown_ids_of_the_new_scheme_but_not_ruff_codes(fake, capsys):
    fake.issue(2)["body"] = issue_body(
        extra=" Needs CLEAN9 and GATE-NOPE; ruff CLEAN401 and CLEAN1a are fine.", legacy="M0.1"
    )
    code, out = lint(fake, capsys)
    assert code == 1
    assert "unknown id CLEAN9" in out and "unknown id GATE-NOPE" in out
    assert "CLEAN401" not in out and "unknown id CLEAN1" not in out


def test_id_reference_pattern_uses_the_prefixes_in_use():
    pattern = board.id_reference_pattern({"CLEAN1", "TWIN10", "GATE-CLEAN", "Decisions"})
    text = "CLEAN1, TWIN10, TWIN123, SIM2, GATE-CLEAN, xTWIN1, CLEAN2a"
    assert pattern.findall(text) == ["CLEAN1", "TWIN10", "GATE-CLEAN", "CLEAN2"]
    assert board.id_reference_pattern({"Decisions"}).findall("TWIN1 GATE-X") == ["GATE-X"]


def test_lint_warns_when_a_number_precedes_its_dependency(fake, capsys):
    for n in (2, 3, 4):
        fake.issue(n)["body"] = fake.issue(n)["body"].replace("{n}", str(n))
    fake.issue(3)["body"] = fake.issue(3)["body"].replace("#2 ", "#4 ", 1)  # CLEAN1 now needs CLEAN2
    code, out = lint(fake, capsys)
    assert code == 0  # a reminder to check, not a stale finding
    assert "CLEAN1 (#3): depends on CLEAN2, which has a higher number" in out
    assert "SAFE1" not in "".join(l for l in out.splitlines() if "higher number" in l)


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
    board.cmd_pr_body(ns(id="SAFE1"))
    captured = capsys.readouterr()
    assert "Refs #2" in captured.out
    assert "Exit criterion of the issue: It works." in captured.out
    assert "- [ ] Tests written" in captured.out
    assert "<!--" not in captured.out
    assert "Suggested PR title: SAFE1 — Safety net" in captured.err


def filled_pr_body(fake, capsys, ident="SAFE1"):
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
        ("Review focus", "the comparison tolerances; revert the pull request to undo"),
    ):
        text = re.sub(
            rf"## {heading}\n\n", f"## {heading}\n\n{content}\n\n", text, count=1
        )
    return text


def test_cmd_pr_check_accepts_a_filled_description(fake, capsys, tmp_path):
    f = tmp_path / "body.md"
    f.write_text(filled_pr_body(fake, capsys))
    board.cmd_pr_check(ns(id="SAFE1", file=str(f)))
    assert "0 problem(s)" in capsys.readouterr().out


def test_cmd_pr_check_reports_every_rule(fake, capsys, tmp_path):
    f = tmp_path / "body.md"
    f.write_text(
        "<!-- left over -->\n## Summary\n\nshort\n\nCloses #2\n\n## What changed\n\n-\n\n## Evidence\n\n| | | |\n\nsee /home/someone/x\n"
    )
    with pytest.raises(SystemExit):
        board.cmd_pr_check(ns(id="SAFE1", file=str(f)))
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


# ---------------------------------------------------------------- table columns


@pytest.fixture
def columns(fake):
    fake.text_fields = True
    return fake


def test_item_key_lowercases_only_the_first_letter():
    assert board.item_key("PR") == "pR"
    assert board.item_key("Blocked by") == "blocked by"


def test_pr_label_states():
    base = {"number": 5, "mergedAt": None, "state": "OPEN", "isDraft": False}
    assert board.pr_label(base) == "#5 open"
    assert board.pr_label({**base, "isDraft": True}) == "#5 draft"
    assert board.pr_label({**base, "state": "CLOSED"}) == "#5 closed"
    assert board.pr_label({**base, "mergedAt": "2026-10-01"}) == "#5 merged"
    assert board.pr_label({k: v for k, v in base.items() if k != "isDraft"}) == "#5 open"


def test_link_columns_blockers_and_refs(columns):
    info = board.issues()
    sts = {"SAFE1": "Done", "CLEAN1": "Todo", "CLEAN2": "Todo", "GATE-CLEAN": "Todo"}
    got = board.link_columns(info, sts, columns.prs)
    assert got["SAFE1"] == ("", "#10 merged")
    assert got["CLEAN1"] == ("", "#11 open")  # SAFE1 is Done, so it no longer blocks
    assert got["CLEAN2"] == ("CLEAN1", "#12 merged")
    assert got["GATE-CLEAN"] == ("CLEAN2", "")
    assert "Decisions" not in got  # not in `sts`


def test_refresh_writes_only_the_cells_that_changed(columns, capsys):
    assert board.refresh_links() == 6
    assert columns.text["CLEAN1", "Blocked by"] == "SAFE1"
    assert columns.text["CLEAN1", "PR"] == "#11 open"
    assert ("SAFE1", "Blocked by") not in columns.text
    assert board.refresh_links() == 0  # nothing left to write
    assert columns.cell_writes == 6
    assert "refresh: 0 cell(s) written" in capsys.readouterr().out
    columns.status["SAFE1"] = "Done"  # a blocker is resolved: that cell is cleared
    board.reset_cache()  # by someone else, between two command lines
    assert board.refresh_links() == 1
    assert ("CLEAN1", "Blocked by") not in columns.text


def test_refresh_passes_text_as_a_variable_not_in_the_query(columns):
    columns.prs[1]["title"] = 'quote " and \\ backslash Refs #3'
    board.refresh_links()
    assert columns.text["CLEAN1", "PR"] == "#11 open"
    assert not any('quote' in str(c) and 'query=' in str(c) for c in columns.calls)


def test_refresh_without_the_columns(fake, capsys):
    with pytest.raises(SystemExit, match="no text column"):
        board.refresh_links()
    assert board.refresh_links(strict=False) == 0
    assert "refresh skipped" in capsys.readouterr().err
    assert fake.cell_writes == 0


def test_status_and_done_refresh_the_columns(columns):
    columns.status["SAFE1"] = "In Progress"
    board.cmd_status(ns(id="CLEAN1", status="PR Open"))
    assert columns.text["CLEAN1", "PR"] == "#11 open"
    board.cmd_done(ns(id="SAFE1", evidence="ok", waive="boxes not ticked in this fixture", pr=None, waive_pr=None))
    assert columns.status["SAFE1"] == "Done"
    assert ("CLEAN1", "Blocked by") not in columns.text


def test_cmd_refresh_dispatch(columns, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["board.py", "refresh"])
    board.main()
    assert "refresh: 6 cell(s) written" in capsys.readouterr().out


# ---------------------------------------------------------------- command line


@pytest.mark.parametrize(
    "argv",
    [
        ["list"],
        ["next"],
        ["dod", "SAFE1"],
        ["check", "SAFE1", "1", "--note", "n"],
        ["uncheck", "SAFE1", "1"],
        ["na", "SAFE1", "2", "why"],
        ["start", "SAFE1", "--force"],
        ["finding", "SAFE1", "text", "--also", "CLEAN1"],
        ["status", "CLEAN1", "PR Open"],
        ["context", "SAFE1"],
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
            "GATE-CLEAN",
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
    monkeypatch.setattr(sys, "argv", ["board.py", "pr-check", "SAFE1", str(body)])
    with pytest.raises(SystemExit):
        board.main()
    monkeypatch.setattr(sys, "argv", ["board.py", "pr-body", "SAFE1"])
    board.main()
    monkeypatch.setattr(sys, "argv", ["board.py", "sanitise", str(body)])
    board.main()
    assert REPO_ROOT.joinpath(
        ".github", "PULL_REQUEST_TEMPLATE", "torch-backend.md"
    ).exists()


# ---------------------------------------------------------------- GraphQL budget


def calls_of(fake, *head):
    return [c for c in fake.calls if c[: len(head)] == head]


def table_queries(fake):
    return [c for c in calls_of(fake, "api", "graphql") if "items(first" in c[3]]


def test_the_table_is_read_with_one_narrow_graphql_query_not_item_list(fake):
    items = board.board_items()
    assert not calls_of(fake, "project", "item-list")
    assert len(table_queries(fake)) == 1
    assert items["SAFE1"] == {
        "id": "item-SAFE1",
        "title": "SAFE1 — Safety net",
        "number": 2,
        "status": "Todo",
    }


def test_the_table_query_asks_only_for_the_columns_the_tool_reads():
    q = board.ITEMS_QUERY
    assert q.count("fieldValueByName") == 3  # no `fieldValues(first: ...)` connection, no labels, no assignees
    assert "fieldValues" not in q and "labels" not in q and "assignees" not in q
    for name in ("Status", board.BLOCKED_FIELD, board.PR_FIELD):
        assert f'name: "{name}"' in q


def test_the_table_is_read_in_pages_and_keeps_text_cells(columns):
    columns.page_size = 2
    columns.text["CLEAN1", "PR"] = "#11 open"
    items = board.board_items()
    assert set(items) == {"SAFE1", "CLEAN1", "CLEAN2", "GATE-CLEAN", "Decisions"}
    assert len(table_queries(columns)) == 3  # five items, two per page
    assert items["CLEAN1"][board.item_key("PR")] == "#11 open"
    assert board.item_key("Blocked by") not in items["CLEAN1"]  # an empty cell has no key


def test_rows_that_are_not_issues_are_skipped_and_a_missing_status_means_todo(fake):
    fake.draft_items = 3
    del fake.status["GATE-CLEAN"]
    items = board.board_items()
    assert not any(k.startswith("draft") for k in items) and len(items) == 4
    assert board.statuses()["SAFE1"] == "Todo"


def test_a_row_without_a_status_value_defaults_to_todo(fake, monkeypatch):
    real = fake.items_page

    def without_status(args):
        data = json.loads(real(args))
        for node in data["data"]["node"]["items"]["nodes"]:
            node["status"] = None
        return json.dumps(data)

    monkeypatch.setattr(fake, "items_page", without_status)
    assert set(board.statuses().values()) == {"Todo"}


def test_one_command_reads_each_thing_once(columns):
    columns.status["SAFE1"] = "In Progress"
    board.cmd_status(ns(id="CLEAN1", status="PR Open"))
    assert len(table_queries(columns)) == 1  # the status write and the column writes update the copy
    assert len(calls_of(columns, "pr", "list")) == 1  # "open" is filtered from the one read
    assert len(calls_of(columns, "issue", "list")) == 1
    assert len(calls_of(columns, "project", "list")) == 1
    assert len(calls_of(columns, "project", "field-list")) == 1


def test_done_reads_the_issues_again_only_after_it_edited_one(columns):
    columns.status["SAFE1"] = "In Progress"
    board.cmd_done(ns(id="SAFE1", evidence="ok", waive="boxes", pr=None, waive_pr=None))
    assert len(table_queries(columns)) == 1
    assert len(calls_of(columns, "pr", "list")) == 1
    assert len(calls_of(columns, "issue", "list")) == 2  # once, and once after the PR box was ticked


def test_start_reads_the_table_once(fake):
    board.cmd_start(ns(id="SAFE1", force=False))
    assert len(table_queries(fake)) == 1
    assert fake.status["SAFE1"] == "In Progress"


def test_put_body_invalidates_the_issues_and_set_status_updates_the_table_copy(fake):
    info = board.issues()
    assert board.issues() is info  # read once
    board.put_body(info, "SAFE1", "new body")
    assert board.issues()["SAFE1"]["body"] == "new body"
    board.set_status("SAFE1", "In Progress")
    assert board.statuses()["SAFE1"] == "In Progress"
    assert len(table_queries(fake)) == 1


def test_pull_requests_are_filtered_by_state_from_one_read(fake):
    assert [p["number"] for p in board.pull_requests()] == [10, 11, 12]
    assert [p["number"] for p in board.pull_requests("merged")] == [10, 12]
    assert [p["number"] for p in board.pull_requests("open")] == [11]
    assert len(calls_of(fake, "pr", "list")) == 1


def test_project_metadata_is_written_to_the_git_directory_and_reused(fake, tmp_path):
    board.project()
    assert (tmp_path / "board-cache.json").exists()
    assert len(calls_of(fake, "project", "list")) == len(calls_of(fake, "project", "field-list")) == 1
    board.reset_cache()  # the next command line
    number, proj, field = board.project()
    assert (number, proj, field["name"]) == ("2", "P2", "Status")
    assert len(calls_of(fake, "project", "list")) == 1  # read from the file


@pytest.mark.parametrize("how", ["stale", "other project", "corrupt", "disabled"])
def test_project_metadata_file_is_ignored_when_stale_foreign_corrupt_or_disabled(
    fake, tmp_path, monkeypatch, how
):
    board.project()
    path = tmp_path / "board-cache.json"
    saved = json.loads(path.read_text())
    if how == "stale":
        saved["saved"] -= board.CACHE_TTL + 1
    if how == "other project":
        saved["key"] = "someone/else|Other"
    path.write_text("{not json" if how == "corrupt" else json.dumps(saved))
    if how == "disabled":
        monkeypatch.setenv("BOARD_NO_CACHE", "1")
    board.reset_cache()
    board.project()
    assert len(calls_of(fake, "project", "list")) == 2


def test_project_metadata_without_a_git_directory_or_with_a_read_only_one(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(board, "git_dir_file", lambda name: None)
    assert board.project()[0] == "2"
    monkeypatch.setattr(board, "git_dir_file", lambda name: tmp_path / "missing-dir" / name)
    board.reset_cache()
    assert board.project()[0] == "2"  # the file cannot be written; nothing breaks


def test_a_status_option_added_after_the_file_was_written_is_found_by_reading_again(fake):
    board.project()
    fake.options["opt-new"] = "In Progress"  # the fake gains the option the cached copy lacks
    del fake.options["opt-prog"]
    board.project_meta()["fields"]["Status"]["options"] = [
        o for o in board.project_meta()["fields"]["Status"]["options"] if o["name"] != "In Progress"
    ]
    board.set_status("SAFE1", "In Progress")
    assert fake.status["SAFE1"] == "In Progress"
    assert len(calls_of(fake, "project", "list")) == 2


def test_a_column_created_after_the_file_was_written_is_found_by_reading_again(fake):
    board.project()  # written without the text columns
    fake.text_fields = True
    board.reset_cache()
    assert board.refresh_links() == 6
    assert len(calls_of(fake, "project", "list")) == 2


def test_the_budget_command_reports_points_and_reset(fake, capsys):
    board.cmd_budget(ns())
    out = capsys.readouterr().out
    assert "4990 of 5000 points left" in out and "resets at" in out


def test_verbose_prints_the_cost_and_low_budgets_warn_or_stop(fake, capsys, monkeypatch):
    monkeypatch.setattr(board, "VERBOSE", True)
    board.board_items()
    err = capsys.readouterr().err
    assert "this query cost 1, 4990 points left" in err
    monkeypatch.setattr(board, "VERBOSE", False)
    board.reset_cache()
    fake.remaining = board.LOW_BUDGET - 1
    board.board_items()
    warned = capsys.readouterr().err
    assert "warning" in warned and f"{board.LOW_BUDGET - 1} GraphQL points left" in warned
    board.reset_cache()
    fake.remaining = board.MIN_BUDGET - 1
    with pytest.raises(SystemExit, match="budget nearly spent"):
        board.board_items()
    board.note_rate(None)  # a query that reports nothing is silent


def test_the_budget_thresholds_are_exclusive(fake, capsys):
    fake.remaining = board.LOW_BUDGET  # exactly the limit: no warning yet
    board.board_items()
    assert capsys.readouterr().err == ""
    board.reset_cache()
    fake.remaining = board.MIN_BUDGET  # exactly the minimum: a warning, no stop
    board.board_items()
    assert "warning" in capsys.readouterr().err


def test_a_comfortable_budget_prints_nothing(fake, capsys):
    board.board_items()
    assert capsys.readouterr().err == ""


def test_main_accepts_the_verbose_flag_and_the_budget_command(fake, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["board.py", "--verbose", "budget"])
    monkeypatch.setattr(board, "VERBOSE", False)
    board.main()
    assert board.VERBOSE is True
    assert "GraphQL budget" in capsys.readouterr().out
    monkeypatch.setattr(board, "VERBOSE", False)


def test_each_command_line_run_starts_without_what_the_last_one_read(fake, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["board.py", "list", "Done"])
    board.main()
    assert "SAFE1" not in capsys.readouterr().out
    fake.status["SAFE1"] = "Done"  # changed by someone else between two runs
    board.main()
    assert "SAFE1" in capsys.readouterr().out


def test_a_rate_limit_failure_says_when_the_budget_resets(monkeypatch):
    now = 1_000_000
    monkeypatch.setattr(board.time, "time", lambda: now)

    def run(args, **kwargs):
        if "-i" in args:  # the hint reads the response headers
            return subprocess.CompletedProcess(args, 1, f"HTTP/2.0 200 OK\nX-Ratelimit-Reset: {now + 754}\n", "")
        return subprocess.CompletedProcess(args, 1, "", "GraphQL: API rate limit exceeded for user ID 1.")

    monkeypatch.setattr(board.subprocess, "run", run)
    with pytest.raises(SystemExit) as e:
        board.gh("issue", "list")
    text = str(e.value)
    assert "rate limit exceeded" in text and "in 12 min 34 s" in text and "nothing that needs GraphQL" in text


def test_a_rate_limit_failure_without_headers_still_gives_advice(monkeypatch):
    monkeypatch.setattr(
        board.subprocess,
        "run",
        lambda args, **k: subprocess.CompletedProcess(args, 1, "", "API rate limit exceeded"),
    )
    with pytest.raises(SystemExit, match="per hour"):
        board.gh("issue", "list")


def test_other_failures_do_not_query_the_rate_limit(monkeypatch):
    seen = []

    def run(args, **kwargs):
        seen.append(args)
        return subprocess.CompletedProcess(args, 1, "", "boom")

    monkeypatch.setattr(board.subprocess, "run", run)
    with pytest.raises(SystemExit, match="boom"):
        board.gh("issue", "list")
    assert len(seen) == 1
