---
name: feature
description: Builds one feature in this worktree with the researcher, planner, executor and QA loop. Use only when the user asks for it with /feature.
---

The feature to build is the text the user wrote with `/feature`.

You are the orchestrator for this one feature. You never write or edit code
yourself and you do not read source files. You work from the plan file and
the short replies your subagents send back. Run each subagent with the
custom agent named below, and wait for it to finish before the next step.

## 0. Preflight

Run `git branch --show-current`. If it prints `main`, stop and tell the user
to start from a feature worktree. Otherwise take the branch name, replace
every `/` with `-`, and use:

- research brief: `research/<name>.md`
- plan file: `plans/<name>.md`

## 1. Research

Run the `researcher` agent with the feature request and the brief path. If
its reply lists a BLOCKING question, stop and show the user the question.

## 2. Plan

Run the `planner` agent with the feature request, the brief path and the
plan file path.

If it replies `NEEDS RESEARCH`, run a new `researcher` with that question
and the brief path, then run the planner again. Do this once per feature.
If the planner asks a second time, stop and show the user its question.

Read the plan file. If it has a `Needs a human` section, stop and show it to
the user. Otherwise commit the brief and the plan:
`git add -- <brief> <plan> && git commit -m "plan: <feature in a few words>" -- <brief> <plan>`

## 3. Execute

For each subtask, in the plan's order, run the `executor` agent with the
plan file path and the subtask ID. Run one executor at a time: they share
this checkout.

Keep a table of subtask ID, the executor that built it, and its number of
QA failures.

If an executor replies `BLOCKED`, go to step 6.

## 4. QA

Run a new `qa` agent with the plan file path, the brief path and base branch
`main`. Use a new one every round so it has no memory of earlier rounds.

## 5. Route the verdict

- `VERDICT: PASS`: go to step 7.
- A subtask failed: add one to its failure count. Give only that subtask's
  part of the QA report to the executor that built it, if you can still
  message it. If you cannot, run a new `executor` with the plan file path,
  the subtask ID and that part of the report. When every failed subtask has
  been fixed, go back to step 4.
- A subtask has now failed three times, or INTEGRATION is not `none`:
  go to step 6.
- UNOWNED CHANGES is not `none`: stop and show the user the files.

## 6. Re-plan, once

You may re-plan once per feature. Give the `planner` agent the brief path,
the plan file path and the QA report or the BLOCKED message. Commit the
changed plan the same way as in step 2. Then run step 3 for the IDs listed
in the plan's `Changes` section only, with new executors, and go to step 4.

If you have already re-planned, stop and say what is still failing.

## 7. Done

Report in under fifteen lines: the feature, the brief and plan paths, the
subtask IDs with their commit hashes, how many QA rounds it took, any LATER
questions from the brief, and the two commands to merge the branch. Do not
merge and do not push.

## When to interrupt the user

Only for these:

- a BLOCKING question in the research brief, or a second `NEEDS RESEARCH`
- a `Needs a human` section, a BLOCKED you cannot re-plan around, or
  unowned changes
- the feature still fails after the one re-plan, or after six QA rounds
- everything passed

Do not ask the user to confirm the brief or the plan, to approve a retry,
or what to do next.
