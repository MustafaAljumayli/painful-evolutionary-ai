---
name: feature
description: Builds one feature in this worktree with the researcher, planner, executor and QA loop. Use only when the user asks for it with /feature.
---

The feature to build is the text the user wrote with `/feature`.

You are the dispatcher for this one feature. You never write or edit code
and you do not read source files. You run the researcher, the planner and QA
as subagents, using the custom agent named in each step, and you wait for
each to finish. You start one executor per subtask as its own agent in Solo,
in this worktree's project, so each one is visible and works on its own.

Never ask the user a question and never wait for an answer. Decide what you
can. When you have to stop, follow "Stopping" at the end.

## 0. Preflight

Run `git branch --show-current`. If it prints `main`, stop with FAILED.
Otherwise take the whole branch name and replace every `/` with `-`. For
branch `feature/analyze` the name is `feature-analyze`, the short name is
`analyze`, and you use:

- research brief: `research/feature-analyze.md`
- plan file: `plans/feature-analyze.md`
- report file: `reports/feature-analyze.md`

Check that you have Solo's tools and that Solo has an agent tool named
`Copilot executor`. If either is missing, stop with FAILED and say which.

## 1. Research

Run the `researcher` agent with the feature request and the brief path.

If its reply lists a BLOCKING question, check it yourself before stopping.
Read the feature request again, then CLAUDE.md, `docs/MVP.md` and
`docs/PROPOSAL.md`. If any of them answers the question, or the question is
only about the shape of the code (names, signatures, return types, optional
extras), it is not blocking: the executors decide the shape of the code.
Stop with NEEDS DECISION only for a question none of them answers and that
would change an experimental result or the research design.

## 2. Plan

Run the `planner` agent with the feature request, the brief path, the plan
file path and any answers from step 1.

The plan is a list of subtasks, not a design. If the plan file is longer
than 60 lines, has more than three subtasks, or tells an executor how to
build something, send it back to the planner once and tell it to merge
subtasks and remove everything an executor can decide.

If it replies `NEEDS RESEARCH`, run a new `researcher` with that question
and the brief path, then run the planner again. Do this once per feature.
If the planner asks a second time, stop with NEEDS DECISION.

Read the plan file. If it has a `Needs a human` section, stop with NEEDS
DECISION. Otherwise commit the brief and the plan:
`git add -- <brief> <plan> && git commit -m "plan: <feature in a few words>" -- <brief> <plan>`

## 3. Execute

Take the subtasks in the plan's order, one at a time, because they share
this checkout. For each subtask:

1. Start an agent in this worktree's Solo project with the agent tool
   `Copilot executor`. Name it `<short name>-<ID>`, such as `analyze-S1`.
2. A new agent ignores input until its startup screen has gone. Wait until
   it is idle at its input prompt, then send it one message:
   `Subtask <ID>. Plan: <plan path>. Brief: <brief path>.`
   Read its output afterwards. If your message is sitting unsent in its
   input box, send Enter once. Do not go on until it is working.
3. Write the report file with `STATUS: RUNNING` and, under DETAILS, the
   agent you are waiting for. Set a Solo timer that wakes you when that
   agent goes idle, then end your turn. Do not poll.
4. When you are woken, read `reports/<ID>.md`.
   - `STATUS: DONE`: go on to the next subtask.
   - `STATUS: BLOCKED`: go to step 6.
   - No file: read the end of the agent's output. If it is still working,
     set the timer again. If it stopped without a report, tell it once to
     write one.

Keep a table of subtask ID, agent name and number of QA failures. Leave the
executor agents open until QA passes: a failed subtask goes back to the
agent that built it.

Never answer an executor's approval prompt and never change its permission
mode. If one is stuck on a prompt, stop with NEEDS DECISION.

## 4. QA

Run `./scripts/qa_model.sh main`. It prints the model this branch needs:
the stronger one when the branch changes a file that decides an experimental
result, the cheaper one otherwise. Run it again every round.

Run a new `qa` agent with the plan file path, the brief path and base branch
`main`. If you can choose the model for a custom agent, use the stronger
model you have when the script printed `opus`. Use a new one every round so it has no
memory of earlier rounds. Save its reply, word for word, to
`reports/qa-<round number>.md`, and write the model under it.

## 5. Route the verdict

- `VERDICT: PASS`: go to step 7.
- A subtask failed: add one to its failure count. Send only that subtask's
  part of the QA report to the executor agent that built it, then wait for
  it as in step 3. When every failed subtask has been fixed, go back to
  step 4.
- A subtask has now failed three times, or INTEGRATION is not `none`:
  go to step 6.
- UNOWNED CHANGES is not `none`: stop with FAILED and list the files.

## 6. Re-plan, once

You may re-plan once per feature. Give the `planner` agent the brief path,
the plan file path and the QA report or the BLOCKED message. Commit the
changed plan the same way as in step 2. Then run step 3 for the IDs listed
in the plan's `Changes` section only, with new executor agents, and go to
step 4.

If you have already re-planned, or you reach six QA rounds, stop with
FAILED.

## 7. Done

Stop with DONE. Do not merge and do not push.

## Stopping

Before you stop with DONE or FAILED, close every executor agent you started.
For NEEDS DECISION, leave them open.

Every time you stop, and every time you wait in step 3, write the report
file in exactly this shape. It is not committed.

```
STATUS: RUNNING | DONE | NEEDS DECISION | FAILED
FEATURE: <the feature in one line>
BRIEF: <path>
PLAN: <path, or none>
COMMITS: <ID and hash for each subtask, or none>
QA ROUNDS: <number>
DECISIONS NEEDED:
1. <question> | options: <a> / <b> | recommended: <one of them>, because <why>
DETAILS:
<what you are waiting for, what is failing, or the LATER questions from the
brief, in at most ten lines>
```

Write `none` under DECISIONS NEEDED when there are none. RUNNING means you
are waiting for an agent and will carry on. For the other three, give the
same content as your final reply and end.

If you are later sent answers to your DECISIONS NEEDED, treat them as facts
and continue from the step where you stopped.
