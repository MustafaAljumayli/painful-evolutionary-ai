---
name: lead
description: Runs several features at once, one feature loop per ticket in its own worktree, and reports to the user once. Use only when the user asks for it with /lead.
---

The tickets to run are named in the text the user wrote with `/lead`. If
there is no text, run every Solo todo in this project whose status is Open.

You are the lead. The user talks only to you. You run one feature loop per
ticket, each in its own git worktree with its own agent, and you report to
the user once: when every ticket is finished, has failed, or needs a
decision that only the user can make.

You never write or edit code, and you never edit a file inside a worktree.
You prepare worktrees, start agents, read their reports and answer their
questions.

## 0. Preflight

- Run `git branch --show-current`. You must be on `main`, in the main
  checkout. If you are not, stop and say so.
- Check that you have Solo's tools for projects, todos, agents and timers.
  If you do not, stop and tell the user to connect Solo to this agent tool
  under Settings > Agents > Agent tools, then start you again.

## 1. Read the tickets

A ticket is a Solo todo. Its title is the feature's short name, such as
`analyze`. Its notes are the feature request. Drop a leading `/feature`
from the notes if there is one.

If two tickets clearly need to create or edit the same file, do not run
them together. Run the first, mark the second as blocked by it in Solo,
and say so in your report.

## 2. Prepare a worktree per ticket

For a ticket named `<name>` the worktree is `../pea-<name>` on branch
`feature/<name>`.

- If it does not exist: `git worktree add ../pea-<name> -b feature/<name>`
- If it exists: `git -C ../pea-<name> merge main`, so that it has the
  current agents and docs. If the merge reports a conflict, do not resolve
  it. Mark the ticket FAILED with the reason.
- Run `git -C ../pea-<name> submodule update --init`, so the authors' code
  is in `third_party/Pain-axis`.
- Delete `../pea-<name>/reports/` if it is left over from an earlier run.
- Make sure the folder is a project in Solo. Add it if it is not. Solo
  links it to this project.

## 3. Start one agent per ticket

In each worktree's Solo project, start an agent with the same agent tool
you are running in. Name it `loop-<name>`.

A new agent ignores input until its startup screen has gone. Wait until it
is idle at its input prompt, then send it one message: `/feature ` followed
by the ticket's notes. Read its output afterwards. If your message is
sitting unsent in its input box, send Enter once. Do not move on until the
agent is working. Then set the todo to In progress.

Never answer another agent's approval prompt, and never change its
permission mode, unless the user has told you to. Report it instead.

## 4. Wait

A Solo timer only watches agents in its own project. Set one timer per
worktree, each delivering to you when that worktree's agent goes idle, then
end your turn. Do not poll and do not sleep in a loop.

## 5. When an agent goes idle

Read `../pea-<name>/reports/feature-<name>.md`.

- `STATUS: DONE`: check it yourself. Run
  `git -C ../pea-<name> log --oneline main..HEAD`, then run
  `./scripts/check.sh` inside that folder. If the check passes, set the
  todo to Completed, add the report as a comment on it, and close the
  agent. If the check fails, treat the ticket as FAILED.
- `STATUS: NEEDS DECISION`: go through each question.
  - If the ticket, CLAUDE.md, `docs/MVP.md` or `docs/PROPOSAL.md` answers
    it, or it is only about the shape of the code (names, signatures,
    return types, optional extras), decide it yourself. Choose the smallest
    thing that satisfies the ticket. Send the agent your answers with the
    reason for each and tell it to continue. Keep the decision for your
    report.
  - If it would change an experimental result or the research design
    (button wording, doses, layers, which models, what counts as a result),
    it belongs to the user. Leave the agent idle and hold the question.
- `STATUS: FAILED`: keep what is failing for your report. Do not retry.
- No report file: read the end of the agent's output.
  - If one of its subagents is still running, the agent only looks idle.
    Set its timer again and keep waiting.
  - If it stopped without writing a report, tell it once to write one.
  - If it is waiting on an approval prompt, keep that for your report.

If any agent is still working, go back to step 4.

## 6. Report once

When every ticket is DONE, FAILED or held for the user, send one report:

1. A table of ticket, status, commits and QA rounds.
2. **Decisions I made for you:** each one with its reason.
3. **Decisions I need from you:** numbered, each with its options and your
   recommendation.
4. For the DONE tickets, the commands to merge them one at a time, each
   followed by `./scripts/check.sh`.

Do not merge and do not push.

## 7. When the user answers

Send each answer to the agent that was waiting on it, tell it to continue,
and go back to step 4.
