"""Driver for one feature loop: decides which stage runs next. No model involved.

The lead runs it through scripts/loop.sh from the main checkout:

    loop.sh start <name>      prepare the worktree and the ledger (ticket on stdin)
    loop.sh next <name>       read the finished stage's report, decide the next stage
    loop.sh answer <name>     record answers to open questions (stdin) and continue
    loop.sh summary <name>    the loop's result in a few lines
    loop.sh status            one line per loop

Each stage is a new agent with a clean context. It reads reports/ledger.md in its
worktree, does the stage named under "Now", and writes reports/stages/<stage>.md.
The ledger is the loop's shared memory. Only this script writes it.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROLES = {"researcher", "planner", "executor", "qa"}


# ---------------- small helpers ----------------
def conf() -> dict[str, str]:
    """Return loop.conf as a dict of strings."""
    out = {}
    for line in (ROOT / "loop.conf").read_text(encoding="utf-8").splitlines():
        m = re.fullmatch(r'([A-Z_]+)=("?)(.*)\2', line.strip())
        if m:
            out[m.group(1)] = m.group(3)
    return out


def git(*args: str, cwd: Path = ROOT, check: bool = True) -> subprocess.CompletedProcess:
    """Run git and return the finished process."""
    return subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True, check=check)


def worktree(name: str) -> Path:
    """Return the worktree folder of a loop."""
    return ROOT.parent / f"pea-{name}"


def check_name(name: str) -> str:
    """Return the name if it is a safe ticket name, else exit."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name):
        sys.exit(f"error: ticket name {name!r} must be lower-case letters, digits and dashes")
    return name


def load(name: str) -> dict:
    """Return the saved state of a loop, or exit if it was never started."""
    path = worktree(name) / "reports" / "state.json"
    if not path.is_file():
        sys.exit(f"error: no loop for {name!r}. Run: ./scripts/loop.sh start {name}")
    return json.loads(path.read_text(encoding="utf-8"))


def save(state: dict) -> None:
    """Write the state file, the ledger and the launch file of the current stage."""
    wt = worktree(state["name"])
    reports = wt / "reports"
    (reports / "stages").mkdir(parents=True, exist_ok=True)
    (reports / "state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    (reports / "ledger.md").write_text(render(state), encoding="utf-8")
    cur = state["current"]
    if cur and state["status"] == "running":
        (reports / "next.env").write_text(
            f"STAGE_ROLE={cur['role']}\nSTAGE_MODEL={cur['model']}\nSTAGE_EFFORT={cur['effort']}\n",
            encoding="utf-8",
        )
    else:
        (reports / "next.env").unlink(missing_ok=True)


def field(text: str, key: str) -> str:
    """Return the value after "KEY:" on its own line, or an empty string."""
    m = re.search(rf"^{re.escape(key)}:[ \t]*(.*)$", text, flags=re.MULTILINE)
    return m.group(1).strip() if m else ""


def stage(state: dict, sid: str, role: str, task: str) -> dict:
    """Return a new stage with its role's model and effort from loop.conf."""
    c = conf()
    model = c[f"{role.upper()}_MODEL"] if role != "qa" else qa_model(state["name"])
    return {
        "id": sid,
        "role": role,
        "task": task,
        "model": model,
        "effort": c[f"{role.upper()}_EFFORT"],
    }


def qa_model(name: str) -> str:
    """Return the model QA runs on for this branch (scripts/qa_model.sh decides)."""
    r = subprocess.run(
        ["bash", str(ROOT / "scripts" / "qa_model.sh"), "main"],
        cwd=worktree(name),
        text=True,
        capture_output=True,
    )
    return r.stdout.strip() or conf()["QA_MODEL_CRITICAL"]


def next_id(state: dict, base: str) -> str:
    """Return base, base-2, base-3... for the first one not used yet in this loop."""
    used = {h["id"] for h in state["history"]} | {q["id"] for q in state["queue"]}
    if state["current"]:
        used.add(state["current"]["id"])
    if base not in used:
        return base
    n = 2
    while f"{base}-{n}" in used:
        n += 1
    return f"{base}-{n}"


def subtasks(state: dict) -> list[str]:
    """Return the subtask IDs in the plan file, in order."""
    plan = worktree(state["name"]) / state["plan"]
    if not plan.is_file():
        return []
    return re.findall(r"^### (S\d+):", plan.read_text(encoding="utf-8"), flags=re.MULTILINE)


# ---------------- the ledger ----------------
def render(state: dict) -> str:
    """Return the ledger text: ticket, answers, what each stage reported, what runs now."""
    wt = worktree(state["name"])
    out = [
        f"# Loop ledger: {state['name']}",
        "",
        "Shared memory for this loop. The loop driver writes it. Agents read it and",
        "write only their own stage report.",
        "",
        f"Branch: feature/{state['name']} (base: main)",
        f"Brief: {state['brief']}",
        f"Plan: {state['plan']}",
        "",
        "## Ticket",
        "",
        state["ticket"].strip(),
        "",
    ]
    if state["answers"]:
        out += ["## Answers to open questions", "", "Treat these as facts.", ""]
        out += [a.strip() + "\n" for a in state["answers"]]
    out += ["## Log", ""]
    last_qa = max((h["id"] for h in state["history"] if h["role"] == "qa"), default="", key=qa_no)
    for h in state["history"]:
        out.append(f"### {h['id']} ({h['role']}): {h['result']}")
        path = wt / "reports" / "stages" / f"{h['id']}.md"
        if h["role"] == "qa" and h["id"] != last_qa:
            out += [f"Superseded by a later QA round. Full text: reports/stages/{h['id']}.md", ""]
        elif path.is_file():
            out += ["", path.read_text(encoding="utf-8").strip(), ""]
    if not state["history"]:
        out += ["Nothing yet.", ""]
    out += ["## Now", ""]
    cur = state["current"]
    if state["status"] == "running" and cur:
        out += [
            f"Stage: {cur['id']}",
            f"Role: {cur['role']}",
            f"Task: {cur['task']}",
            f"Write your stage report to: reports/stages/{cur['id']}.md",
        ]
    else:
        out += [f"Loop status: {state['status'].upper()}. {state['reason']}".rstrip()]
    return "\n".join(out) + "\n"


def qa_no(sid: str) -> int:
    """Return the round number of a QA stage ID such as qa3."""
    m = re.fullmatch(r"qa(\d+)", sid)
    return int(m.group(1)) if m else 0


# ---------------- routing ----------------
def stop(state: dict, status: str, reason: str) -> None:
    """End the loop, or pause it for a decision."""
    state["status"], state["reason"] = status, reason
    if status != "decision":
        state["queue"] = []


def replan(state: dict, why: str) -> None:
    """Send the loop back to the planner once. A second time, the loop fails."""
    if state["replanned"]:
        stop(state, "failed", f"Still failing after one re-plan. {why}")
        return
    state["replanned"] = True
    task = (
        "Re-plan. The stage report above explains why. Edit the same plan file, and "
        "list the new or changed subtask IDs in your report's CHANGED line."
    )
    state["queue"] = [stage(state, next_id(state, "plan"), "planner", f"{task} Reason: {why}")]


def after_research(state: dict, text: str) -> None:
    """Route a researcher's report."""
    if field(text, "STATUS") == "NEEDS DECISION":
        stop(state, "decision", "The researcher has a blocking question (see its report).")


def after_plan(state: dict, text: str, cur: dict) -> None:
    """Route a planner's report: follow-up research, a human, or the build stages."""
    status = field(text, "STATUS")
    c = conf()
    again = "Write the plan from the brief and the answers above."
    if status == "NEEDS RESEARCH":
        if state["research_followups"] >= 1:
            stop(state, "decision", "The planner needs research a second time (see its report).")
            state["queue"] = [stage(state, next_id(state, "plan"), "planner", again)]
            return
        state["research_followups"] += 1
        q = field(text, "QUESTION")
        state["queue"] = [
            stage(
                state,
                next_id(state, "research"),
                "researcher",
                f"Follow-up. Answer this in a new section of the same brief: {q}",
            ),
            stage(state, next_id(state, "plan"), "planner", again),
        ]
        return
    if status == "NEEDS HUMAN":
        stop(state, "decision", "The plan needs a change only a person may make (see its report).")
        state["queue"] = [stage(state, next_id(state, "plan"), "planner", again)]
        return
    wt = worktree(state["name"])
    plan = wt / state["plan"]
    ids = subtasks(state)
    if not plan.is_file() or not ids:
        stop(state, "failed", "The planner reported DONE but the plan file has no subtasks.")
        return
    n_lines = len(plan.read_text(encoding="utf-8").splitlines())
    limit = int(c["PLAN_MAX_LINES"])
    if n_lines > limit:
        if state["plan_resends"] >= 1:
            stop(state, "failed", f"The plan is {n_lines} lines after one resend (limit {limit}).")
            return
        state["plan_resends"] += 1
        state["queue"] = [
            stage(
                state,
                next_id(state, "plan"),
                "planner",
                f"The plan is {n_lines} lines; the limit is {limit}. Cut everything an "
                "executor can decide. Keep the subtasks.",
            )
        ]
        return
    git("add", "--", state["brief"], state["plan"], cwd=wt, check=False)
    git(
        "commit",
        "-m",
        f"plan: {state['name']}",
        "--",
        state["brief"],
        state["plan"],
        cwd=wt,
        check=False,
    )
    todo = ids
    if cur["id"] != "plan" and state["replanned"]:
        changed = re.findall(r"S\d+", field(text, "CHANGED"))
        todo = [i for i in ids if i in changed] or ids
        for i in todo:
            state["fails"][i] = 0
    state["queue"] = [
        stage(state, next_id(state, i), "executor", f"Build subtask {i} from the plan.")
        for i in todo
    ]
    state["queue"].append(qa_stage(state))


def qa_stage(state: dict) -> dict:
    """Return a QA stage. It gets its round number when it starts."""
    return stage(
        state,
        "qa",
        "qa",
        "Review the whole branch against the plan and the brief. Base branch: main.",
    )


def after_build(state: dict, text: str, cur: dict) -> None:
    """Route an executor's report."""
    if field(text, "STATUS") == "BLOCKED":
        replan(state, f"Stage {cur['id']} is blocked.")


def after_qa(state: dict, text: str, cur: dict) -> None:
    """Route a QA verdict: done, fixes and another round, a re-plan, or failure."""
    c = conf()
    if field(text, "VERDICT") == "PASS":
        stop(state, "done", "")
        return
    unowned = field(text, "UNOWNED CHANGES")
    if unowned and unowned.lower() != "none":
        stop(state, "failed", f"The branch changes files no subtask owns: {unowned}")
        return
    failed = re.findall(r"^(S\d+): FAIL", text, flags=re.MULTILINE)
    for i in failed:
        state["fails"][i] = state["fails"].get(i, 0) + 1
    integration = field(text, "INTEGRATION")
    limit = int(c["MAX_SUBTASK_FAILS"])
    if (integration and integration.lower() != "none") or not failed:
        replan(
            state, f"QA round {state['qa_round']}: {integration or 'failed with no subtask named'}"
        )
    elif any(state["fails"][i] >= limit for i in failed):
        worst = [i for i in failed if state["fails"][i] >= limit]
        replan(state, f"{', '.join(worst)} failed QA {limit} times.")
    elif state["qa_round"] >= int(c["MAX_QA_ROUNDS"]):
        stop(state, "failed", f"Still failing after {state['qa_round']} QA rounds.")
    else:
        state["queue"] = [
            stage(
                state,
                next_id(state, f"{i}-fix{state['fails'][i]}"),
                "executor",
                f"Fix subtask {i}. QA stage {cur['id']} failed it; its report is in the Log. "
                "Fix only what it lists for this subtask.",
            )
            for i in failed
        ]
        state["queue"].append(qa_stage(state))


def result_of(role: str, text: str) -> str:
    """Return the one-word result a stage report states, or an empty string."""
    value = field(text, "VERDICT") if role == "qa" else field(text, "STATUS")
    return value if re.fullmatch(r"[A-Z][A-Z ]*", value) else ""


def advance(state: dict) -> list[str]:
    """Take in the current stage's report if there is one and return the lead's instructions."""
    name, cur = state["name"], state["current"]
    lines: list[str] = []
    if state["status"] == "running" and cur:
        path = worktree(name) / "reports" / "stages" / f"{cur['id']}.md"
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        result = result_of(cur["role"], text)
        if not result:
            why = "its report is unreadable" if text else "it has not written its report"
            return [f"WAIT {name}-{cur['id']}  ({why}: reports/stages/{cur['id']}.md)"]
        state["history"].append({"id": cur["id"], "role": cur["role"], "result": result})
        state["current"] = None
        lines.append(f"CLOSE {name}-{cur['id']}")
        if cur["role"] == "researcher":
            after_research(state, text)
        elif cur["role"] == "planner":
            after_plan(state, text, cur)
        elif cur["role"] == "executor":
            after_build(state, text, cur)
        else:
            after_qa(state, text, cur)
    if state["status"] == "running":
        if not state["current"]:
            if not state["queue"]:
                stop(state, "failed", "The loop ran out of stages without a QA pass.")
            else:
                state["current"] = state["queue"].pop(0)
                if state["current"]["role"] == "qa":  # the branch is complete only now
                    state["qa_round"] += 1
                    state["current"]["id"] = f"qa{state['qa_round']}"
                    state["current"]["model"] = qa_model(name)
        if state["current"]:
            lines.append(f"SPAWN {name}-{state['current']['id']}")
    if state["status"] != "running":
        word = {"done": "DONE", "failed": "FAILED", "decision": "DECISION"}[state["status"]]
        lines.append(f"{word} {name}  {state['reason']}".rstrip())
    save(state)
    return lines


# ---------------- commands ----------------
def cmd_start(name: str) -> None:
    """Prepare the worktree, write the ledger and print the first instructions."""
    ticket = sys.stdin.read().strip()
    if not ticket:
        sys.exit("error: give the ticket's notes on standard input")
    if git("branch", "--show-current").stdout.strip() != "main":
        sys.exit("error: run this from the main checkout, on main")
    wt, branch = worktree(name), f"feature/{name}"
    if not wt.exists():
        has_branch = git("rev-parse", "--verify", "--quiet", branch, check=False).returncode == 0
        args = (
            ["worktree", "add", str(wt), branch]
            if has_branch
            else ["worktree", "add", str(wt), "-b", branch]
        )
        r = git(*args, check=False)
        if r.returncode != 0:
            sys.exit(f"FAILED {name}  cannot create the worktree: {r.stderr.strip()}")
    else:
        r = git("merge", "main", "--no-edit", cwd=wt, check=False)
        if r.returncode != 0:
            git("merge", "--abort", cwd=wt, check=False)
            sys.exit(f"FAILED {name}  merging main into {branch} conflicts; resolve it by hand")
    # Borrow the authors' repo from this checkout instead of downloading it again.
    ref = str(ROOT / "third_party" / "Pain-axis")
    if git("submodule", "update", "--init", "--reference", ref, cwd=wt, check=False).returncode:
        git("submodule", "update", "--init", cwd=wt, check=False)
    shutil.rmtree(wt / "reports", ignore_errors=True)
    state = {
        "name": name,
        "ticket": ticket,
        "brief": f"research/feature-{name}.md",
        "plan": f"plans/feature-{name}.md",
        "status": "running",
        "reason": "",
        "answers": [],
        "history": [],
        "current": None,
        "queue": [],
        "qa_round": 0,
        "fails": {},
        "replanned": False,
        "research_followups": 0,
        "plan_resends": 0,
    }
    state["queue"] = [
        stage(state, "research", "researcher", "Research this ticket and write the brief."),
        stage(state, "plan", "planner", "Split the ticket into subtasks and write the plan."),
    ]
    print(f"PROJECT {wt}")
    print("\n".join(advance(state)))


def cmd_answer(name: str) -> None:
    """Record answers to the open questions and continue the loop."""
    state = load(name)
    text = sys.stdin.read().strip()
    if state["status"] != "decision":
        sys.exit(f"error: the loop for {name} is not waiting for a decision")
    if not text:
        sys.exit("error: give the answers on standard input")
    state["answers"].append(text)
    state["status"], state["reason"] = "running", ""
    print("\n".join(advance(state)))


def summary(name: str) -> str:
    """Return the loop's result in a few lines, for the lead's report and the pull request."""
    state = load(name)
    wt = worktree(name)
    log = git("log", "--format=%h %s", f"main..feature/{name}", check=False).stdout.strip()
    out = [
        f"STATUS: {state['status'].upper()}",
        f"TICKET: {name}",
        f"REQUEST: {state['ticket'].splitlines()[0][:200]}",
        f"BRANCH: feature/{name}",
        f"QA ROUNDS: {state['qa_round']}",
        f"STAGES: {', '.join(h['id'] + '=' + h['result'] for h in state['history']) or 'none'}",
    ]
    if state["reason"]:
        out.append(f"REASON: {state['reason']}")
    for h in state["history"]:
        path = wt / "reports" / "stages" / f"{h['id']}.md"
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        if state["status"] == "decision" and h is state["history"][-1]:
            out += ["OPEN QUESTIONS (last stage report):", text.strip()]
        if h["role"] == "researcher":
            m = re.search(r"^LATER:[ \t]*\n?((?:.+\n?)*)", text, flags=re.MULTILINE)
            if m and m.group(1).strip() and m.group(1).strip().lower() != "none":
                out += ["LATER:", m.group(1).strip()]
    out += ["COMMITS:", log or "none"]
    return "\n".join(out)


def cmd_status() -> None:
    """Print one line per loop that has a state file."""
    for path in sorted(ROOT.parent.glob("pea-*/reports/state.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        now = s["current"]["id"] if s["current"] and s["status"] == "running" else "-"
        print(f"{s['name']:24} {s['status'].upper():9} stage {now:12} QA rounds {s['qa_round']}")


def main() -> None:
    """Command line entry point."""
    p = argparse.ArgumentParser(description="Feature-loop driver.")
    p.add_argument("command", choices=["start", "next", "answer", "summary", "status"])
    p.add_argument("name", nargs="?")
    a = p.parse_args()
    if a.command == "status":
        cmd_status()
        return
    if not a.name:
        sys.exit("error: give the ticket name")
    name = check_name(a.name)
    if a.command == "start":
        cmd_start(name)
    elif a.command == "next":
        print("\n".join(advance(load(name))))
    elif a.command == "answer":
        cmd_answer(name)
    else:
        print(summary(name))


if __name__ == "__main__":
    main()
