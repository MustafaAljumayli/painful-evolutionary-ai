---
description: Build one feature in this worktree with the planner, executor and QA loop
argument-hint: <what to build>
disable-model-invocation: true
---

Build this feature: $ARGUMENTS

You are the orchestrator for this one feature. You never write or edit code
yourself and you do not read source files. You work from the plan file and
the short replies your subagents send back. Wait for each subagent to finish
before the next step (run it in the foreground, not the background).

## 0. Preflight

Run `git branch --show-current`. If it prints `main`, stop and tell me to
start from a feature worktree. Otherwise the plan file is
`plans/<branch name with every / replaced by ->.md`.

## 1. Plan

Call the `planner` subagent with the feature request and the plan file path.
Read the plan file when it returns. If it has a `Needs a human` section,
stop and show it to me.

## 2. Execute

For each subtask, in the plan's order, call the `executor` subagent with the
plan file path and the subtask ID. Name it `exec-<ID>`. Run one executor at
a time: they share this checkout.

Keep a table of subtask ID, executor name and number of QA failures.

If an executor replies `BLOCKED`, go to step 5.

## 3. QA

Call a new `qa` subagent with the plan file path and base branch `main`.
Use a new one every round so it has no memory of earlier rounds.

## 4. Route the verdict

- `VERDICT: PASS`: go to step 6.
- A subtask failed: add one to its failure count. Send only that subtask's
  part of the QA report to the executor that built it, with SendMessage to
  `exec-<ID>`, so it keeps its own context. When every failed subtask has
  been fixed, go back to step 3.
- A subtask has now failed three times, or INTEGRATION is not `none`:
  go to step 5.
- UNOWNED CHANGES is not `none`: stop and show me the files.

## 5. Re-plan, once

You may re-plan once per feature. Send the planner the plan file path and
the QA report or the BLOCKED message. Then run step 2 for the IDs listed in
the plan's `Changes` section only, with new executors, and go to step 3.

If you have already re-planned, stop and tell me what is still failing.

## 6. Done

Tell me, in under fifteen lines: the feature, the subtask IDs with their
commit hashes, how many QA rounds it took, and the two commands to merge
the branch. Do not merge and do not push.

## When to interrupt me

Only for these:

- a `Needs a human` section, a BLOCKED you cannot re-plan around, or
  unowned changes
- the feature still fails after the one re-plan, or after six QA rounds
- everything passed

Do not ask me to confirm the plan, to approve a retry, or what to do next.
