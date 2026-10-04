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
    python tools/board.py refresh                   write the "Blocked by" and "PR" columns of the project table (only the cells that changed)
    python tools/board.py budget                    what is left of the hourly GraphQL budget (5,000 points, shared by every agent and the app)
    python tools/board.py lint                      find stale text in the issues (removed files, unknown ids, missing appendices, old counts, second person)
    python tools/board.py sanitise [FILE]           print FILE (or stdin) made safe for a public tracker; use it for any PR or issue text you write yourself

Everything this script posts is sanitised first (the repository is public): bare #N that is not an issue of this repository, @mentions outside
code spans, links to other repositories' issues and pull requests, absolute local paths and e-mail addresses are neutralised, and the private terms
listed in `<git-dir>/board-private-terms.txt` (lines `term => replacement`, never committed) are replaced. A notice says what was changed.

ID is a milestone or gate id such as CLEAN1 or GATE-CLEAN (or Decisions); the legacy ids of the first numbering (M0.2, G0, ...) are still accepted
and resolve to the new id (each issue carries a `- Legacy id:` line). The target repository is the fork below (override with the environment
variable BOARD_REPO); any ICAMS repository is refused. The project is found by its title.

GitHub's GraphQL budget is 5,000 points an hour for the whole account. The project table is read with one narrow query (about the cost of one
`gh issue list`), the issues, the pull requests and the table once per command, and the project's id and columns are kept for a day in
`<git-dir>/board-cache.json` (BOARD_NO_CACHE=1 ignores it). `--verbose` (or BOARD_VERBOSE=1) prints the cost of each table query; below 200 points
left every table query warns, below 10 the command stops and says when the budget resets.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

REPO = os.environ.get("BOARD_REPO", "akwarii/grace-tensorpotential")
INTEGRATION_BRANCH = os.environ.get("BOARD_BRANCH", "torch-backend")
OWNER = REPO.split("/")[0]
PROJECT_TITLE = "GRACE torch backend"
STATUSES = ["Todo", "In Progress", "PR Open", "Done"]
DOD_HEADING = "## Definition of Done"
BLOCKED_FIELD = "Blocked by"  # text columns of the project table, written by `refresh`
PR_FIELD = "PR"
CACHE_TTL = 24 * 3600  # seconds the project's id, columns and Status options are kept in the git directory
LOW_BUDGET = 200  # GraphQL points left below which every query that reports them prints a warning
MIN_BUDGET = 10  # ... and below which a command stops instead of failing half-way
VERBOSE = bool(os.environ.get("BOARD_VERBOSE"))
RATE_QUERY = "{ rateLimit { limit cost remaining used resetAt } }"
# One request for the whole table, asking only for what the tool reads. `gh project item-list` fetches every field,
# label and assignee of every item as nested connections: 103 points per call on this 97-item, 22-field project, and
# `gh project field-list` about 110 to 150 (measured 2026-10-04 with a pause before each reading, the counter lags);
# this query costs 1, and the project metadata is kept in a file, so a command costs about 3 points.
ITEMS_QUERY = (
    "query($project: ID!, $cursor: String) { rateLimit { cost remaining resetAt } "
    "node(id: $project) { ... on ProjectV2 { items(first: 100, after: $cursor) { "
    "pageInfo { hasNextPage endCursor } nodes { id content { ... on Issue { number title } } "
    'status: fieldValueByName(name: "Status") { ... on ProjectV2ItemFieldSingleSelectValue { name } } '
    'blocked: fieldValueByName(name: "Blocked by") { ... on ProjectV2ItemFieldTextValue { text } } '
    'pr: fieldValueByName(name: "PR") { ... on ProjectV2ItemFieldTextValue { text } } } } } } }'
)
LEGACY_LINE = re.compile(r"^- Legacy id: (\S+)[ \t]*\n?", re.M)
LEGACY_ID = re.compile(r"(?<![\w.])(?:M\d+\.\d+(?!\d)|G\d(?!\w))")

if REPO.lower().startswith("icams/"):
    sys.exit("refusing to act on an ICAMS repository")


def git_dir_file(name: str) -> Path | None:
    """A file in the git directory (kept out of the repository on purpose), or None outside a repository."""
    r = subprocess.run(
        ["git", "rev-parse", "--git-dir"], capture_output=True, text=True
    )
    return Path(r.stdout.strip()) / name if r.returncode == 0 else None


def private_terms() -> list[tuple[str, str]]:
    """Terms to generalise, read from <git-dir>/board-private-terms.txt (kept out of the repository on purpose)."""
    path = git_dir_file("board-private-terms.txt")
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


def reset_text(epoch: float) -> str:
    """Local time of day of a reset, with the minutes left."""
    wait = max(0, int(epoch - time.time()))
    clock = time.strftime("%H:%M:%S", time.localtime(epoch))
    return f"{clock} (in {wait // 60} min {wait % 60} s)"


def rate_limit_hint() -> str:
    """When the GraphQL window resets, read from the response headers (`gh api rate_limit` was found to lag behind them)."""
    r = subprocess.run(
        ["gh", "api", "-i", "graphql", "-f", "query={ __typename }"],
        capture_output=True,
        text=True,
    )
    m = re.search(r"^x-ratelimit-reset:\s*(\d+)", r.stdout, re.I | re.M)
    if not m:
        return "The GraphQL budget is per hour; try again later."
    return (
        f"The GraphQL budget resets at {reset_text(int(m.group(1)))}; "
        "nothing that needs GraphQL (every `gh issue`, `gh pr` and `gh project` command) works until then."
    )


def gh(*args: str) -> str:
    r = subprocess.run(["gh", *args], capture_output=True, text=True)
    if r.returncode != 0:
        err = r.stderr.strip()
        hint = f"\n{rate_limit_hint()}" if "rate limit" in err.lower() else ""
        sys.exit(f"gh {' '.join(args[:4])} failed: {err[:400]}{hint}")
    return r.stdout.strip()


_CACHE: dict[str, object] = {}  # what this run has read; a command line is one run


def reset_cache() -> None:
    """Forget what this run has read (a caller that changed the board behind the tool's back)."""
    _CACHE.clear()


def cached(key: str, fetch):
    """Read once per run: the table, the issues and the pull requests are each needed by several steps of a command."""
    if key not in _CACHE:
        _CACHE[key] = fetch()
    return _CACHE[key]


def note_rate(rate: dict | None) -> None:
    """Report what a query cost and warn, or stop, when the hourly GraphQL budget is nearly spent."""
    if not rate:
        return
    reset = datetime.fromisoformat(rate["resetAt"].replace("Z", "+00:00")).timestamp()
    if VERBOSE:
        print(
            f"[graphql: this query cost {rate['cost']}, {rate['remaining']} points left, resets at {reset_text(reset)}]",
            file=sys.stderr,
        )
    if rate["remaining"] < MIN_BUDGET:
        sys.exit(
            f"GraphQL budget nearly spent ({rate['remaining']} points left); it resets at {reset_text(reset)}. Stop and wait."
        )
    if rate["remaining"] < LOW_BUDGET:
        print(
            f"[warning: {rate['remaining']} GraphQL points left, resets at {reset_text(reset)}; "
            "avoid anything that is not needed now]",
            file=sys.stderr,
        )


def item_id_of(title: str) -> str:
    head = title.split(" — ")[0].strip()
    return "Decisions" if head.lower().startswith("decisions") else head


def issues() -> dict[str, dict]:
    """id -> {number, title, body, state, needs (ids), score}; read once per run (`put_body` invalidates it)."""
    return cached("issues", fetch_issues)


def fetch_issues() -> dict[str, dict]:
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


def fetch_project_meta() -> dict:
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
    return {"number": number, "id": p["id"], "fields": {f["name"]: f for f in fields}}


def load_project_meta(refresh: bool) -> dict:
    """The project's number, node id and columns: from <git-dir>/board-cache.json when it is fresh, else from GitHub."""
    key = f"{REPO}|{PROJECT_TITLE}"
    path = git_dir_file("board-cache.json")
    use_file = path is not None and not os.environ.get("BOARD_NO_CACHE")
    if use_file and not refresh and path.exists():
        try:
            saved = json.loads(path.read_text())
            if saved["key"] == key and time.time() - saved["saved"] < CACHE_TTL:
                return saved["meta"]
        except (ValueError, KeyError, TypeError, OSError):
            pass  # unreadable or stale: read it again
    meta = fetch_project_meta()
    if use_file:
        try:
            path.write_text(json.dumps({"key": key, "saved": time.time(), "meta": meta}))
        except OSError:
            pass  # a read-only git directory only costs the next command two requests
    return meta


def project_meta(refresh: bool = False) -> dict:
    """Project metadata, read once per run; `refresh` ignores the file (a column or option was added since)."""
    if refresh:
        _CACHE.pop("meta", None)
    return cached("meta", lambda: load_project_meta(refresh))


def project() -> tuple[str, str, dict]:
    """(project number, project node id, Status field)."""
    m = project_meta()
    return m["number"], m["id"], m["fields"]["Status"]


def fetch_items() -> dict[str, dict]:
    """item id of the issue -> {id, title, number, status, <text columns>}; one request per 100 items."""
    project_id = project_meta()["id"]
    items: dict[str, dict] = {}
    cursor = None
    while True:
        args = ["api", "graphql", "-f", f"query={ITEMS_QUERY}", "-f", f"project={project_id}"]
        if cursor:
            args += ["-f", f"cursor={cursor}"]
        data = json.loads(gh(*args))["data"]
        note_rate(data.get("rateLimit"))
        page = data["node"]["items"]
        for node in page["nodes"]:
            content = node.get("content") or {}
            if "number" not in content:
                continue  # a draft item or a pull request
            row = {"id": node["id"], "title": content["title"], "number": content["number"]}
            for alias, key in (
                ("status", "status"),
                ("blocked", item_key(BLOCKED_FIELD)),
                ("pr", item_key(PR_FIELD)),
            ):
                value = node.get(alias)
                if value:
                    row[key] = value.get("name") or value.get("text") or ""
            items[item_id_of(content["title"])] = row
        if not page["pageInfo"]["hasNextPage"]:
            return items
        cursor = page["pageInfo"]["endCursor"]


def board_items() -> dict[str, dict]:
    """The project table, read once per run (`set_status` and the column writes keep it current)."""
    return cached("items", fetch_items)


def statuses() -> dict[str, str]:
    return {k: v.get("status", "Todo") for k, v in board_items().items()}


def set_status(item_id: str, status: str) -> None:
    if status not in STATUSES:
        sys.exit(f"status must be one of {STATUSES}")
    _, proj, field = project()
    if not any(o["name"] == status for o in field["options"]):
        project_meta(refresh=True)  # an option may have been added since the file was written
        _, proj, field = project()
    row = board_items()[item_id]
    item = row["id"]
    option = next(o["id"] for o in field["options"] if o["name"] == status)
    q = (
        f'mutation {{ updateProjectV2ItemFieldValue(input:{{projectId:"{proj}", itemId:"{item}", fieldId:"{field["id"]}", '
        f'value:{{singleSelectOptionId:"{option}"}}}}) {{ clientMutationId }} }}'
    )
    gh("api", "graphql", "-f", f"query={q}")
    row["status"] = status
    print(f"{item_id}: Status -> {status}")


SET_TEXT = (
    "mutation($project: ID!, $item: ID!, $field: ID!, $text: String!) { updateProjectV2ItemFieldValue("
    "input: {projectId: $project, itemId: $item, fieldId: $field, value: {text: $text}}) { clientMutationId } }"
)
CLEAR_FIELD = (
    "mutation($project: ID!, $item: ID!, $field: ID!) { clearProjectV2ItemFieldValue("
    "input: {projectId: $project, itemId: $item, fieldId: $field}) { clientMutationId } }"
)


def project_fields() -> dict[str, dict]:
    return project_meta()["fields"]


def item_key(field_name: str) -> str:
    """The key `gh project item-list` uses for a field: only the first letter is lower-cased ("PR" -> "pR")."""
    return field_name[:1].lower() + field_name[1:]


def set_text_field(proj: str, item: str, field: str, text: str) -> None:
    """Write (or, for an empty text, clear) a text cell of the project table."""
    ids = ["-f", f"project={proj}", "-f", f"item={item}", "-f", f"field={field}"]
    if text:
        gh("api", "graphql", "-f", f"query={SET_TEXT}", *ids, "-f", f"text={text}")
    else:
        gh("api", "graphql", "-f", f"query={CLEAR_FIELD}", *ids)


def pr_label(pr: dict) -> str:
    if pr["mergedAt"]:
        state = "merged"
    elif pr["state"] == "OPEN":
        state = "draft" if pr.get("isDraft") else "open"
    else:
        state = "closed"
    return f"#{pr['number']} {state}"


def link_columns(
    info: dict, statuses_: dict[str, str], prs: list[dict]
) -> dict[str, tuple[str, str]]:
    """item id -> (open blockers, pull requests that reference it with `Refs #N`)."""
    out = {}
    for item_id, v in info.items():
        if item_id not in statuses_:
            continue
        blocked = [d for d in v["needs"] if statuses_.get(d) != "Done"]
        ref = re.compile(rf"\bRefs #{v['number']}(?!\d)")
        mine = sorted(
            (p for p in prs if ref.search(p["title"] + "\n" + (p["body"] or ""))),
            key=lambda p: p["number"],
        )
        out[item_id] = (", ".join(blocked), ", ".join(pr_label(p) for p in mine))
    return out


def refresh_links(strict: bool = True) -> int:
    """Write the Blocked by and PR columns for every item whose cell is out of date; returns the number of cells written."""
    _, proj, _ = project()
    fields = project_fields()
    missing = [n for n in (BLOCKED_FIELD, PR_FIELD) if n not in fields]
    if missing:  # the column may have been created after the file was written
        fields = project_meta(refresh=True)["fields"]
        missing = [n for n in (BLOCKED_FIELD, PR_FIELD) if n not in fields]
    if missing:
        msg = f"project has no text column {missing}; create it (Text field) first"
        if strict:
            sys.exit(msg)
        print(f"[refresh skipped: {msg}]", file=sys.stderr)
        return 0
    items = board_items()
    info = issues()
    sts = {k: v.get("status", "Todo") for k, v in items.items()}
    written = 0
    for item_id, (blocked, prs) in link_columns(info, sts, pull_requests()).items():
        for name, key, value in (
            (BLOCKED_FIELD, item_key(BLOCKED_FIELD), blocked),
            (PR_FIELD, item_key(PR_FIELD), prs),
        ):
            if items[item_id].get(key, "") != value:
                set_text_field(proj, items[item_id]["id"], fields[name]["id"], value)
                if value:
                    items[item_id][key] = value
                else:
                    items[item_id].pop(key, None)
                written += 1
    print(f"refresh: {written} cell(s) written")
    return written


def pull_requests(state: str = "all") -> list[dict]:
    """The pull requests of the fork (`state`: all, open or merged), read once per run and filtered here."""
    everything = cached(
        "prs",
        lambda: json.loads(
            gh(
                "pr",
                "list",
                "--repo",
                REPO,
                "--state",
                "all",
                "--limit",
                "300",
                "--json",
                "number,title,body,baseRefName,mergedAt,state,url,isDraft",
            )
        ),
    )
    if state == "merged":
        return [p for p in everything if p["mergedAt"]]
    if state == "open":
        return [p for p in everything if p["state"] == "OPEN"]
    return everything


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
    _CACHE.pop("issues", None)  # the body just written is what the next read must show


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
    refresh_links(strict=False)


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
    refresh_links(strict=False)


def cmd_refresh(_a) -> None:
    refresh_links()


def cmd_budget(_a) -> None:
    """What is left of the hourly GraphQL budget (one point; `gh api rate_limit` lags behind the real count)."""
    r = json.loads(gh("api", "graphql", "-f", f"query={RATE_QUERY}"))["data"]["rateLimit"]
    reset = datetime.fromisoformat(r["resetAt"].replace("Z", "+00:00")).timestamp()
    print(f"GraphQL budget: {r['remaining']} of {r['limit']} points left, resets at {reset_text(reset)}")


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
        for dep in v["needs"]:
            a, b = re.fullmatch(r"([A-Z]+)(\d+)", dep), re.fullmatch(r"([A-Z]+)(\d+)", k)
            if a and b and a[1] == b[1] and int(a[2]) > int(b[2]):
                soft.append(
                    f"{head}: depends on {dep}, which has a higher number in the same theme (numbers follow the dependency order)"
                )
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
    "Definition of Done",
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
    p.add_argument("--verbose", action="store_true", help="print what each GraphQL table query cost and what is left")
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
    sub.add_parser("refresh").set_defaults(fn=cmd_refresh)
    sub.add_parser("budget").set_defaults(fn=cmd_budget)
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
    global VERBOSE
    VERBOSE = VERBOSE or a.verbose
    reset_cache()
    a.fn(a)


if __name__ == "__main__":
    main()
