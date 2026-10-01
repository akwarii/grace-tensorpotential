"""Board helper for agents and humans: read, start, comment on and move the milestone issues of the fork.

Standalone: everything is read live from GitHub (issue titles, dependency lines, the project board); no local file besides this script is needed.

usage (from anywhere inside the repository):
    python tools/board.py next                      ready items (all hard dependencies Done), best score first
    python tools/board.py list [STATUS]             every item with its Status
    python tools/board.py context ID                the issue, its comments, and the comments of its hard dependencies
    python tools/board.py start ID [--force]        Status -> In Progress, plus a comment; refuses if a hard dependency is not Done
    python tools/board.py finding ID "text" [--also ID ...]   comment an UNEXPECTED finding on ID (and on later issues it affects)
    python tools/board.py status ID STATUS          STATUS = Todo | In Progress | PR Open | Done
    python tools/board.py dod ID                    show the Definition of Done checklist of the issue, numbered
    python tools/board.py check ID N [N ...] [--note "evidence"]   tick DoD boxes (every time you finish a task of the issue)
    python tools/board.py na ID N "reason"          mark a DoD box not applicable, with the reason (never leave an inapplicable box blank)
    python tools/board.py uncheck ID N [N ...]      untick boxes
    python tools/board.py done ID "evidence" [--pr N] [--waive "reason"] [--waive-pr "reason"]
                                                    comment the evidence, Status -> Done, close the issue. Refuses while a DoD box is open, and
                                                    unless a pull request that references the issue (`Refs #ID-number`) is MERGED into the
                                                    integration branch of the fork (--waive-pr only when the user said so, e.g. for a gate)
    python tools/board.py pr-body ID                 print a pull-request description for ID from the fork's template, pre-filled (Refs, exit criterion, DoD state)
    python tools/board.py pr-check ID FILE          check a pull-request description against the rules (Refs #N, no Closes, sections filled, comments removed, sanitised)
    python tools/board.py lint                      find stale text in the issues (removed files, unknown ids, missing appendices, old counts, second person)
    python tools/board.py sanitise [FILE]           print FILE (or stdin) made safe for a public tracker; use it for any PR or issue text you write yourself

Everything this script posts is sanitised first (the repository is public): bare #N that is not an issue of this repository, @mentions outside
code spans, links to other repositories' issues and pull requests, absolute local paths and e-mail addresses are neutralised, and the private terms
listed in `<git-dir>/board-private-terms.txt` (lines `term => replacement`, never committed) are replaced. A notice says what was changed.

ID is a milestone or gate id such as CLEAN1 or GATE-CLEAN (or Decisions); the legacy ids of the first numbering (M0.2, G0, ...) are still accepted
and resolve to the new id (each issue carries a `- Legacy id:` line). The target repository is the fork below (override with the environment
variable BOARD_REPO); any ICAMS repository is refused. The project is found by its title.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = os.environ.get("BOARD_REPO", "akwarii/grace-tensorpotential")
INTEGRATION_BRANCH = os.environ.get("BOARD_BRANCH", "torch-backend")
OWNER = REPO.split("/")[0]
PROJECT_TITLE = "GRACE torch backend"
STATUSES = ["Todo", "In Progress", "PR Open", "Done"]
DOD_HEADING = "## Definition of Done"
LEGACY_LINE = re.compile(r"^- Legacy id: (\S+)[ \t]*\n?", re.M)
LEGACY_ID = re.compile(r"(?<![\w.])(?:M\d+\.\d+(?!\d)|G\d(?!\w))")

if REPO.lower().startswith("icams/"):
    sys.exit("refusing to act on an ICAMS repository")


def private_terms() -> list[tuple[str, str]]:
    """Terms to generalise, read from <git-dir>/board-private-terms.txt (kept out of the repository on purpose)."""
    r = subprocess.run(
        ["git", "rev-parse", "--git-dir"], capture_output=True, text=True
    )
    path = (
        Path(r.stdout.strip()) / "board-private-terms.txt"
        if r.returncode == 0
        else None
    )
    if path is None or not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if "=>" in line and not line.lstrip().startswith("#"):
            a, b = line.split("=>", 1)
            out.append((a.strip(), b.strip()))
    return sorted(out, key=lambda x: -len(x[0]))


def sanitise(text: str, known: set[int] | None = None) -> str:
    """Make text safe for a public issue tracker; prints a notice listing what changed."""
    known = known or set()
    changes: list[str] = []
    parts = re.split(r"(```.*?```|`[^`\n]*`)", text, flags=re.S)
    for i in range(0, len(parts), 2):
        s = parts[i]
        rules = [
            (
                r"https?://github\.com/[^\s)]+/(?:issues|pull|discussions)/\d+",
                "(link removed)",
                "link to another issue or pull request",
            ),
            (r"\b[\w.-]+/[\w.-]+#\d+", "(link removed)", "cross-repository reference"),
            (r"[\w.+-]+@[\w-]+\.[\w.-]+", "<email>", "e-mail address"),
            (
                r"(?<![\w.])(?:/home/|/tmp/|/mnt/|/Users/|~/)[^\s,;)`]+",
                "<local path>",
                "local path",
            ),
            (r"(?<![\w`@])@(?=[A-Za-z])", "@\u200b", "@mention"),
        ]
        for pat, repl, label in rules:
            new = re.sub(pat, repl, s)
            if new != s:
                changes.append(label)
                s = new
        new = re.sub(
            r"(?<![\w&])#(\d+)",
            lambda m: m.group(0) if int(m.group(1)) in known else f"No. {m.group(1)}",
            s,
        )
        if new != s:
            changes.append("#N that is not an issue of this repository")
            s = new
        parts[i] = s
    # absolute paths inside code spans are sanitised too
    out = "".join(parts)
    out2 = re.sub(
        r"(?<![\w.])(?:/home/|/tmp/|/mnt/|/Users/)[^\s,;)`]+", "<local path>", out
    )
    if out2 != out:
        changes.append("local path in code")
        out = out2
    for term, repl in private_terms():
        new = re.sub(re.escape(term), repl, out, flags=re.IGNORECASE)
        if new != out:
            changes.append("private term generalised")
            out = new
    if changes:
        print(f"[sanitised: {', '.join(sorted(set(changes)))}]", file=sys.stderr)
    return out


def gh(*args: str) -> str:
    r = subprocess.run(["gh", *args], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"gh {' '.join(args[:4])} failed: {r.stderr.strip()[:400]}")
    return r.stdout.strip()


def item_id_of(title: str) -> str:
    head = title.split(" — ")[0].strip()
    return "Decisions" if head.lower().startswith("decisions") else head


def issues() -> dict[str, dict]:
    """id -> {number, title, body, state, needs (ids), score}."""
    raw = json.loads(
        gh(
            "issue",
            "list",
            "--repo",
            REPO,
            "--state",
            "all",
            "--limit",
            "500",
            "--json",
            "number,title,body,state",
        )
    )
    by_num = {x["number"]: item_id_of(x["title"]) for x in raw}
    out = {}
    for x in raw:
        first = x["body"].split("\n", 1)[0]
        m = re.search(r"\*\*Depends on:\*\*\s*([^·]*)", first)
        needs = (
            [
                by_num[int(n)]
                for n in re.findall(r"#(\d+)", m.group(1))
                if int(n) in by_num
            ]
            if m
            else []
        )
        s = re.search(r"Priority: P\d \(score (\d+)\)", x["body"])
        legacy = LEGACY_LINE.search(x["body"])
        out[by_num[x["number"]]] = {
            **x,
            "needs": needs,
            "score": int(s.group(1)) if s else 0,
            "legacy": legacy.group(1) if legacy else None,
        }
    return out


def resolve(item_id: str, known: dict[str, dict]) -> str:
    """The id of the issue named by `item_id` (case-insensitive); a legacy id (M0.2, G0, ...) resolves to its new id."""
    wanted = next((k for k in known if k.lower() == item_id.lower()), None)
    if wanted is None:
        wanted = next(
            (
                k
                for k, v in known.items()
                if (v.get("legacy") or "").lower() == item_id.lower()
            ),
            None,
        )
        if wanted is not None:
            print(f"[{item_id} is now {wanted}]", file=sys.stderr)
    if wanted is None:
        sys.exit(
            f"unknown id {item_id}; known examples: {', '.join(sorted(known)[:8])} ..."
        )
    return wanted


def project() -> tuple[str, str, dict]:
    """(project number, project node id, Status field)."""
    projects = json.loads(gh("project", "list", "--owner", OWNER, "--format", "json"))[
        "projects"
    ]
    p = next((x for x in projects if x["title"] == PROJECT_TITLE), None)
    if p is None:
        sys.exit(f"project '{PROJECT_TITLE}' not found for {OWNER}")
    number = str(p["number"])
    fields = json.loads(
        gh("project", "field-list", number, "--owner", OWNER, "--format", "json")
    )["fields"]
    return number, p["id"], next(f for f in fields if f["name"] == "Status")


def board_items(number: str) -> dict[str, dict]:
    items = json.loads(
        gh(
            "project",
            "item-list",
            number,
            "--owner",
            OWNER,
            "--format",
            "json",
            "--limit",
            "500",
        )
    )["items"]
    return {item_id_of(i["title"]): i for i in items}


def statuses() -> dict[str, str]:
    number, _, _ = project()
    return {k: v.get("status", "Todo") for k, v in board_items(number).items()}


def set_status(item_id: str, status: str) -> None:
    if status not in STATUSES:
        sys.exit(f"status must be one of {STATUSES}")
    number, proj, field = project()
    item = board_items(number)[item_id]["id"]
    option = next(o["id"] for o in field["options"] if o["name"] == status)
    q = (
        f'mutation {{ updateProjectV2ItemFieldValue(input:{{projectId:"{proj}", itemId:"{item}", fieldId:"{field["id"]}", '
        f'value:{{singleSelectOptionId:"{option}"}}}}) {{ clientMutationId }} }}'
    )
    gh("api", "graphql", "-f", f"query={q}")
    print(f"{item_id}: Status -> {status}")


def pull_requests(state: str = "all") -> list[dict]:
    return json.loads(
        gh(
            "pr",
            "list",
            "--repo",
            REPO,
            "--state",
            state,
            "--limit",
            "300",
            "--json",
            "number,title,body,baseRefName,mergedAt,state,url",
        )
    )


def known_numbers(info: dict) -> set[int]:
    """Numbers of issues and pull requests of this repository (references to them are kept, others are neutralised)."""
    return {v["number"] for v in info.values()} | {p["number"] for p in pull_requests()}


def merged_pr_for(info: dict, item_id: str, explicit: int | None = None) -> dict | None:
    """A pull request merged into the integration branch that references the issue (or the explicitly given one)."""
    number = info[item_id]["number"]
    ref = re.compile(rf"(?<![\w&])#{number}(?!\d)")
    for pr in pull_requests("merged"):
        if pr["baseRefName"] != INTEGRATION_BRANCH or not pr["mergedAt"]:
            continue
        if explicit is not None and pr["number"] == explicit:
            return pr
        if explicit is None and ref.search(pr["title"] + "\n" + (pr["body"] or "")):
            return pr
    return None


def comment(info: dict, item_id: str, text: str) -> None:
    text = sanitise(text, known_numbers(info))
    gh("issue", "comment", str(info[item_id]["number"]), "--repo", REPO, "--body", text)
    print(f"comment added to {item_id} (#{info[item_id]['number']})")


def put_body(info: dict, item_id: str, body: str) -> None:
    body = sanitise(body, known_numbers(info))
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(body)
        path = f.name
    try:
        gh(
            "issue",
            "edit",
            str(info[item_id]["number"]),
            "--repo",
            REPO,
            "--body-file",
            path,
        )
    finally:
        Path(path).unlink(missing_ok=True)


def dod_lines(body: str) -> list[int]:
    """Line indices of the checkbox items under the Definition of Done heading."""
    lines = body.split("\n")
    if DOD_HEADING not in lines:
        return []
    out = []
    for i in range(lines.index(DOD_HEADING) + 1, len(lines)):
        if lines[i].startswith("## ") or lines[i] == "---":
            break
        if lines[i].startswith(("- [ ] ", "- [x] ")):
            out.append(i)
    return out


def set_boxes(
    info: dict,
    item_id: str,
    numbers: list[int],
    checked: bool,
    na_reason: str | None = None,
) -> None:
    body = info[item_id]["body"]
    lines = body.split("\n")
    idx = dod_lines(body)
    if not idx:
        sys.exit(f"{item_id} has no Definition of Done checklist")
    for n in numbers:
        if not 1 <= n <= len(idx):
            sys.exit(f"box number must be 1..{len(idx)}")
        i = idx[n - 1]
        text = re.sub(r"^- \[[ x]\] ", "", lines[i])
        lines[i] = (
            f"- [x] ~~{text}~~ (not applicable: {na_reason})"
            if na_reason is not None
            else f"- [{'x' if checked else ' '}] {text}"
        )
    put_body(info, item_id, "\n".join(lines))
    print(
        f"{item_id}: boxes {numbers} {'ticked' if checked else 'unticked'}"
        + (" as not applicable" if na_reason else "")
    )


def cmd_next(_a) -> None:
    info, sts = issues(), statuses()
    ready = [
        k
        for k, s in sts.items()
        if s == "Todo"
        and k in info
        and all(sts.get(d) == "Done" for d in info[k]["needs"])
    ]
    for k in sorted(ready, key=lambda x: -info[x]["score"]):
        print(f"{k:10} #{info[k]['number']:<4} score {info[k]['score']}")
    if not ready:
        print("no ready item (everything is done, in progress, or blocked)")


def cmd_list(a) -> None:
    info = issues()
    for k, s in statuses().items():
        if k in info and (not a.status or s == a.status):
            print(f"{k:10} #{info[k]['number']:<4} {s}")


def cmd_context(a) -> None:
    info = issues()
    i = resolve(a.id, info)
    for item in [i, *info[i]["needs"]]:
        print("=" * 100)
        print(
            gh("issue", "view", str(info[item]["number"]), "--repo", REPO, "--comments")
        )


def cmd_start(a) -> None:
    info, sts = issues(), statuses()
    i = resolve(a.id, info)
    blocked = [d for d in info[i]["needs"] if sts.get(d) != "Done"]
    if blocked and not a.force:
        sys.exit(
            f"{i} is blocked by {blocked} (not Done). Use --force only if the user said so."
        )
    set_status(i, "In Progress")
    comment(
        info,
        i,
        "**Started.** Status Todo -> In Progress."
        + (f" Hard dependencies not Done (forced): {blocked}." if blocked else ""),
    )


def cmd_finding(a) -> None:
    info = issues()
    i = resolve(a.id, info)
    comment(
        info,
        i,
        f"**Unexpected finding** (something the issue did not predict; read before you start).\n\n{a.text}",
    )
    for other in a.also or []:
        comment(
            info,
            resolve(other, info),
            f"**Unexpected finding from {i}**, relevant here.\n\n{a.text}",
        )


def cmd_status(a) -> None:
    info = issues()
    i = resolve(a.id, info)
    if a.status == "Done":
        sys.exit(
            "An issue is resolved only when a pull request is merged into the integration branch: use `done`, which checks it."
        )
    if a.status == "PR Open":
        open_prs = [
            p
            for p in pull_requests("open")
            if re.search(
                rf"(?<![\w&])#{info[i]['number']}(?!\d)",
                p["title"] + "\n" + (p["body"] or ""),
            )
        ]
        if not open_prs:
            sys.exit(
                f"No open pull request references #{info[i]['number']}. Open one (description: `Refs #{info[i]['number']}`) before setting PR Open."
            )
    set_status(i, a.status)


def cmd_dod(a) -> None:
    info = issues()
    body = info[resolve(a.id, info)]["body"]
    lines = body.split("\n")
    for n, i in enumerate(dod_lines(body), 1):
        print(f"{n}. {lines[i]}")


def cmd_check(a) -> None:
    info = issues()
    i = resolve(a.id, info)
    set_boxes(info, i, a.numbers, True)
    if a.note:
        comment(
            info,
            i,
            f"Definition of Done, boxes {a.numbers} ticked. Evidence:\n\n{a.note}",
        )


def cmd_na(a) -> None:
    info = issues()
    set_boxes(info, resolve(a.id, info), [a.number], True, na_reason=a.reason)


def cmd_uncheck(a) -> None:
    info = issues()
    set_boxes(info, resolve(a.id, info), a.numbers, False)


def pr_box_number(body: str) -> int | None:
    lines = body.split("\n")
    for n, k in enumerate(dod_lines(body), 1):
        if "merged into" in lines[k]:
            return n
    return None


def cmd_done(a) -> None:
    info = issues()
    i = resolve(a.id, info)
    number = info[i]["number"]
    pr = merged_pr_for(info, i, a.pr)
    if pr is None and not a.waive_pr:
        sys.exit(
            f"No pull request referencing #{number} is merged into {INTEGRATION_BRANCH}. An issue is resolved only once its PR is merged: "
            f"open the PR with `Refs #{number}` in its description, get it merged (by the user), then run `done` again. "
            "`--waive-pr REASON` only when the user said so (for example for a gate or a setting change)."
        )
    box = pr_box_number(info[i]["body"])
    if pr is not None and box is not None:
        set_boxes(info, i, [box], True)
        info = issues()
    body = info[i]["body"]
    lines = body.split("\n")
    open_boxes = [
        n for n, k in enumerate(dod_lines(body), 1) if lines[k].startswith("- [ ] ")
    ]
    if pr is None:
        open_boxes = [n for n in open_boxes if n != box]
    if open_boxes and not a.waive:
        sys.exit(
            f"Definition of Done boxes still open: {open_boxes}. Tick them with `check`, mark them `na` with a reason, or pass --waive REASON."
        )
    if open_boxes:
        comment(
            info,
            i,
            f"Closed with Definition of Done boxes {open_boxes} open. Reason: {a.waive}",
        )
    if pr is None:
        comment(info, i, f"Closed without a merged pull request. Reason: {a.waive_pr}")
    via = f" Merged pull request: #{pr['number']}." if pr else ""
    comment(info, i, f"**Done.** Exit criterion met.{via} Evidence:\n\n{a.evidence}")
    set_status(i, "Done")
    gh("issue", "close", str(number), "--repo", REPO)
    print(f"{i} closed")


def cmd_sanitise(a) -> None:
    text = Path(a.file).read_text() if a.file else sys.stdin.read()
    sys.stdout.write(sanitise(text, known_numbers(issues())))


REMOVED = (
    "plan/",
    "TORCH_BACKEND_PLAN",
    "milestones.yaml",
    "create_board",
    "analyze.py",
    "render_issues",
    "DEPENDENCIES.md",
)


def id_reference_pattern(ids: set[str]) -> re.Pattern:
    """Pattern for references to ids of the current scheme (PREFIX1 .. PREFIX99, GATE-NAME), built from the prefixes in use.

    One or two digits only: ruff codes (SIM102, PERF401, FIX001) share some prefixes but have three.
    """
    prefixes = sorted({m.group(0) for k in ids if (m := re.match(r"[A-Z]+(?=\d)", k))})
    parts = [r"GATE-[A-Z]+\b"]
    if prefixes:
        parts.append(rf"(?:{'|'.join(prefixes)})\d{{1,2}}(?!\d)")
    return re.compile(rf"(?<!\w)(?:{'|'.join(parts)})")


def cmd_lint(_a) -> None:
    """Report stale text in issue bodies. Hard findings should be fixed in the same task that made them stale."""
    info = issues()
    ids = set(info)
    appendices = {k.split()[1] for k in ids if k.startswith("Appendix ")}
    hard: list[str] = []
    soft: list[str] = []
    legacy = {v["legacy"]: k for k, v in info.items() if v.get("legacy")}
    new_ref = id_reference_pattern(ids)
    for k, v in info.items():
        body = v["body"]
        head = f"{k} (#{v['number']})"
        text = LEGACY_LINE.sub("", body)
        for token in REMOVED:
            if token in body:
                hard.append(f"{head}: mentions removed `{token}`")
        for ref in set(LEGACY_ID.findall(text)):
            if ref in legacy:
                hard.append(f"{head}: uses the legacy id {ref}; write {legacy[ref]}")
            else:
                hard.append(f"{head}: refers to unknown id {ref}")
        for ref in set(new_ref.findall(text)):
            if ref not in ids:
                hard.append(f"{head}: refers to unknown id {ref}")
        for a in set(re.findall(r"Appendix ([A-L])\b", body)):
            if a not in appendices and not k.startswith("Appendix "):
                hard.append(f"{head}: cites Appendix {a}, which has no reference issue")
        for n in set(re.findall(r"(?<![\w&])#(\d+)", "\n".join(body.split("\n")[1:]))):
            if int(n) not in {x["number"] for x in info.values()}:
                hard.append(f"{head}: refers to unknown issue #{n}")
        if DOD_HEADING in body:
            for needed in (
                "## Context",
                "## Work",
                "## Exit criterion",
                "## Out of scope",
                "## Decisions it depends on",
            ):
                if needed not in body:
                    hard.append(
                        f"{head}: does not follow the issue template (missing '{needed}')"
                    )
        if DOD_HEADING in body and "merged into `torch-backend`" not in body:
            hard.append(f"{head}: Definition of Done lacks the merged-PR box")
        if re.search(r"\b(?:76|79|82|85) (?:milestone|issues)\b", body):
            soft.append(f"{head}: an old issue or milestone count")
        if (
            re.search(r"\b[Yy]our\b", body)
            and "Written for the project owner" not in body
        ):
            soft.append(
                f'{head}: second person ("your"); the text is read by several people and agents'
            )
    for line in hard:
        print("STALE ", line)
    for line in soft:
        print("check ", line)
    print(
        f"{len(hard)} stale finding(s), {len(soft)} to check, {len(info)} issues scanned"
    )
    if hard:
        sys.exit(1)


def section(body: str, heading: str) -> str:
    m = re.search(
        rf"^## {re.escape(heading)}\n+(.*?)(?=^## |^---|\Z)", body, flags=re.S | re.M
    )
    return m.group(1).strip() if m else ""


PR_SECTIONS = [
    "Summary",
    "What changed",
    "Evidence",
    "Unexpected findings",
    "Review focus",
    "Risks and rollback",
    "Definition of Done",
    "Checklist",
]


def cmd_pr_body(a) -> None:
    info = issues()
    i = resolve(a.id, info)
    number = info[i]["number"]
    body = info[i]["body"]
    template = (
        Path(__file__).resolve().parent.parent
        / ".github"
        / "PULL_REQUEST_TEMPLATE"
        / "torch-backend.md"
    )
    text = re.sub(r"<!--.*?-->\n?", "", template.read_text(), flags=re.S)
    text = text.replace("Refs #<issue number>", f"Refs #{number}")
    lines = body.split("\n")
    dod = "\n".join(lines[k] for k in dod_lines(body))
    text = text.replace(
        "## Definition of Done\n\n", f"## Definition of Done\n\n{dod}\n\n", 1
    )
    exit_criterion = section(body, "Exit criterion")
    text = text.replace(
        "## Evidence\n\n",
        f"## Evidence\n\nExit criterion of the issue: {exit_criterion}\n\n",
        1,
    )
    print(f"Suggested PR title: {info[i]['title']}", file=sys.stderr)
    sys.stdout.write(sanitise(text, known_numbers(info)))


def cmd_pr_check(a) -> None:
    info = issues()
    i = resolve(a.id, info)
    number = info[i]["number"]
    text = Path(a.file).read_text()
    problems = []
    if not re.search(rf"\bRefs #{number}(?!\d)", text):
        problems.append(f"missing `Refs #{number}`")
    if re.search(r"\b(?:Closes|Fixes|Resolves) #\d+", text, flags=re.I):
        problems.append(
            "uses Closes/Fixes/Resolves; use `Refs #N` (the issue is closed by `board.py done` after the merge)"
        )
    if "<!--" in text:
        problems.append("template comments are still present")
    for s in PR_SECTIONS:
        content = section(text, s)
        if not content or content in ("-", "| | | |"):
            problems.append(f"section '{s}' is missing or empty")
    summary = section(text, "Summary").replace(f"Refs #{number}", "").strip()
    if len(summary.split()) < 12:
        problems.append(
            "the Summary is too short to say what changed and why (at least a couple of sentences)"
        )
    if "| | | |" in text:
        problems.append("the Evidence table has an empty row")
    clean = sanitise(text, known_numbers(info))
    if clean != text:
        problems.append("text is not sanitised (run `board.py sanitise FILE`)")
    for p_ in problems:
        print("PR-CHECK ", p_)
    print(f"{len(problems)} problem(s)")
    if problems:
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("next").set_defaults(fn=cmd_next)
    s = sub.add_parser("list")
    s.add_argument("status", nargs="?")
    s.set_defaults(fn=cmd_list)
    for name, fn in (("context", cmd_context), ("dod", cmd_dod)):
        s = sub.add_parser(name)
        s.add_argument("id")
        s.set_defaults(fn=fn)
    s = sub.add_parser("start")
    s.add_argument("id")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_start)
    s = sub.add_parser("finding")
    s.add_argument("id")
    s.add_argument("text")
    s.add_argument("--also", nargs="*")
    s.set_defaults(fn=cmd_finding)
    s = sub.add_parser("status")
    s.add_argument("id")
    s.add_argument("status")
    s.set_defaults(fn=cmd_status)
    s = sub.add_parser("check")
    s.add_argument("id")
    s.add_argument("numbers", type=int, nargs="+")
    s.add_argument("--note")
    s.set_defaults(fn=cmd_check)
    s = sub.add_parser("uncheck")
    s.add_argument("id")
    s.add_argument("numbers", type=int, nargs="+")
    s.set_defaults(fn=cmd_uncheck)
    s = sub.add_parser("na")
    s.add_argument("id")
    s.add_argument("number", type=int)
    s.add_argument("reason")
    s.set_defaults(fn=cmd_na)
    sub.add_parser("lint").set_defaults(fn=cmd_lint)
    s = sub.add_parser("pr-body")
    s.add_argument("id")
    s.set_defaults(fn=cmd_pr_body)
    s = sub.add_parser("pr-check")
    s.add_argument("id")
    s.add_argument("file")
    s.set_defaults(fn=cmd_pr_check)
    s = sub.add_parser("sanitise")
    s.add_argument("file", nargs="?")
    s.set_defaults(fn=cmd_sanitise)
    s = sub.add_parser("done")
    s.add_argument("id")
    s.add_argument("evidence")
    s.add_argument("--waive")
    s.add_argument("--pr", type=int)
    s.add_argument("--waive-pr")
    s.set_defaults(fn=cmd_done)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
