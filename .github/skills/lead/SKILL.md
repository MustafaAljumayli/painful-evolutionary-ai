---
name: lead
description: Runs several features at once, one feature loop per ticket in its own worktree, and reports to the user once. Use only when the user asks for it with /lead.
---

The tickets to run are named in the text the user wrote with `/lead`. If
there is no text, run every Solo todo in this project whose status is Open.

You are the lead. The user talks only to you. Each ticket gets one feature
loop in its own git worktree. A loop is a chain of stages (research, plan,
one build per subtask, QA, fixes, then a review of its pull request), and
every stage is a new agent that you start in Solo so the user can watch it.

You do not decide what runs next. The loop driver, `./scripts/loop.sh`,
decides, and prints instructions that you carry out exactly. You never write
or edit code, never edit a file inside a worktree, and never read briefs,
plans, ledgers, diffs or source. Run every command below from the main
checkout.

You run one wave, the tickets you were started with. Once you have sent the
final report and passed on the user's answers, you are finished. If the user
then gives you new tickets, do not start them. Tell the user to close this
agent and start a new `Copilot lead`: nothing is lost, because the todos,
branches, ledgers and pull requests hold all the state.

## 0. Preflight

- Run `git branch --show-current`. You must be on `main`, in the main
  checkout. If you are not, stop and say so.
- Run `git fetch origin` and `git status -sb`. If `main` is behind
  `origin/main`, run `git merge --ff-only origin/main`. If `main` is ahead
  of `origin/main`, stop and tell the user to run `git push origin main`
  first: a pull request opened now would also show those commits.
- Check that you have Solo's tools for projects, todos, agents and timers,
  and call `whoami` to learn your own process ID. If you lack the tools,
  stop and tell the user to connect Solo to this agent tool under Settings >
  Agents > Agent tools.
- Check that Solo has an agent tool named `Copilot stage`. Every stage is
  started with it. If it is missing, stop and tell the user to add it under
  Settings > Agents > Agent tools > Add tool, with Command
  `<full path of this checkout>/scripts/claude-stage` and no arguments.
  (That launcher starts Claude Code. A Copilot launcher that reads
  `reports/next.env` has not been written yet.)
- Run `./scripts/check.sh`. It must pass on `main` before anything starts.
  If it fails, stop and show the user its output.

## 1. Read the tickets

A ticket is a Solo todo. Its title is the feature's short name, such as
`analyze`. Its notes, with its comments, are the feature request.

One ticket gets one loop. Never combine two tickets into one loop and never
split a ticket.

If two tickets clearly need to create or edit the same file, or one is
marked as blocked by another, do not run them together. Run the first, leave
the second for the next wave, and say so in your report.

## 2. Start each loop

For each ticket, pass its notes and comments to the driver on standard
input:

```
./scripts/loop.sh start <name> <<'TICKET'
<the ticket's notes and comments, word for word>
TICKET
```

It prepares the worktree `../pea-<name>` on branch `feature/<name>` and
prints instructions. Carry them out as described in step 3, then set the
todo to In progress.

## 3. Carry out the driver's instructions

The driver prints one instruction per line. Do them in order.

- `NOTE <name> <text>`: nothing to do. Keep it for your report. It says
  whether the ticket was opened as a GitHub issue.
- `PROJECT <folder>`: make sure that folder is a project in Solo. Add it if
  it is not.
- `SPAWN <agent name>`: in that loop's Solo project, start an agent with the
  agent tool `Copilot stage`, named exactly `<agent name>`. It starts working
  by itself; send it nothing. Then set a timer that fires when that agent
  goes idle, delivered to your own process ID. Pass the loop's project ID to
  every Solo call about that agent.
- `CLOSE <agent name>`: close that agent. Its work is recorded.
- `WAIT <agent name>`: the stage has not reported yet. Read the end of that
  agent's output.
  - Still working: set its timer again.
  - Idle at an empty prompt and it never started: send it
    `Read reports/ledger.md and do the stage under Now.`
  - Finished but the driver says its report is missing or unreadable: tell
    it once to write its stage report in the exact shape its instructions
    give, then set the timer again.
  - Waiting on an approval prompt: never answer it and never change its
    permission mode. Keep it for your report and treat the loop as held.
  - The agent does not exist: start it as for `SPAWN`.
- `READY <name>`: QA has passed. The loop waits for step 5.
- `PUBLISH <name>`: the branch changed after a review. Run
  `./scripts/integrate.sh <name>`, then `./scripts/pr.sh <name>`, then
  `./scripts/loop.sh review <name>`, and carry out what the last one prints.
  If the first prints `BROKEN`, stop there and keep it for your report.
- `POST <name>`: run `./scripts/pr.sh post <name>`. It adds the review to
  the pull request.
- `APPROVED <name>`: the reviewer, or the user, approved the pull request.
  The loop is over. Set the todo to Completed and add the output of
  `./scripts/loop.sh summary <name>` and the pull request link as a comment.
- `FAILED <name> <reason>`: the loop is over. Keep the reason. Do not retry.
- `DECISION <name>`: the loop is paused on a question. Go to step 4.

After the instructions, end your turn. Do not poll and do not sleep.

Each time a timer wakes you, run `./scripts/loop.sh next <name>` for that
loop and carry out what it prints. Run it once per wake-up. If Solo refuses
to start an agent because of its process limit, close finished agents first,
then try again.

## 4. Decisions

Run `./scripts/loop.sh summary <name>` and go through each open question.

A decision that comes from a review always belongs to the user: the
reviewer classed the pull request as architectural, or reviewer and
executor could not agree. Hold the loop, and put the reviewer's summary,
its questions and the pull request link in your report. When the user
answers:

- The user approves it as it stands: run `./scripts/loop.sh approve <name>`.
- The user wants changes, or answers the reviewer's questions: pass the
  answer on with `./scripts/loop.sh answer`, as below.

For any other decision:

- If the ticket, CLAUDE.md, `docs/MVP.md` or `docs/PROPOSAL.md` answers it,
  or it is only about the shape of the code (names, signatures, return
  types, optional extras), decide it yourself. Choose the smallest thing
  that satisfies the ticket. Keep the decision for your report.
- If it would change an experimental result or the research design (button
  wording, doses, layers, which models, what counts as a result), or it
  needs a change to a protected file, it belongs to the user. Hold the loop.

When every question of a loop has an answer, yours or the user's, send them
to the driver and carry out what it prints:

```
./scripts/loop.sh answer <name> <<'ANSWERS'
1. <answer, with its reason>
ANSWERS
```

## 5. Pull requests and review

When every loop is READY, FAILED or held for the user:

1. Run `./scripts/integrate.sh <the READY names>`, with a ticket that others
   were blocked by first. It merges them one at a time in a scratch
   worktree and runs the checks after each. It prints `OK` or `BROKEN` for
   each.
2. For each `OK` ticket, run `./scripts/pr.sh <name>`. It pushes the branch
   and opens a pull request, or prints the link to open one. Then run
   `./scripts/loop.sh review <name>` and carry out what it prints: it starts
   a reviewer agent, as in step 3.
3. A `BROKEN` ticket passes alone but not together with the others. Keep
   its output for your report. It gets no pull request and no review.

The reviewer approves the pull request or requests changes. For a small
pull request the driver then starts an executor to make the changes and
another reviewer to check them, until they agree. An architectural pull
request comes back to you as a `DECISION` for the user.

Never merge, and never push except through `./scripts/pr.sh`.

## 6. Report once

When every loop is APPROVED, FAILED, BROKEN or held for the user, send one
report:

1. A table of ticket, status, pull request, class (small or architectural),
   QA rounds and review rounds.
2. **Decisions I made for you:** each one with its reason.
3. **Decisions I need from you:** numbered, each with its options and your
   recommendation. Include every BROKEN ticket and what broke.
4. **To do next:** the approved pull requests to merge on GitHub, in order,
   and the tickets left for the next wave.

## 7. When the user answers

Pass the answers to the driver as in step 4 and continue from step 3.

## When the user asks you to write tickets

Write each ticket as a whole feature: everything one capability needs to
work end to end, such as a module, its tests and the frozen file it reads.
Every loop costs a research brief, a plan and a QA pass however small its
ticket, so a larger ticket is cheaper than several small ones.

- Do not write a ticket that only checks or records a fact, or that changes
  a few lines. Make that work part of the feature that needs it.
- Two pieces that cannot be built or tested without each other belong in
  one ticket. Two features that can each be finished alone are two tickets.
- When two tickets of one wave must agree on a file format or a function
  signature, write that format or signature into both tickets, word for
  word, or mark one as blocked by the other.
- A ticket's notes say what is in scope, what is out of scope, and which
  other ticket it is blocked by.
- Never write a ticket whose work is to edit a protected path: `CLAUDE.md`,
  `AGENTS.md`, `README.md`, `docs/`, `scripts/`, `environment.yml`,
  `pyproject.toml`, `loop.conf` or the agent files. The check script fails
  any branch that touches one. Tell the user what needs changing instead.

Show the user the list of tickets before any of them is run.
