---
name: feature
description: Builds one feature in this worktree with the researcher, planner, executor and QA loop. Use only when the user asks for it with /feature.
---

The feature to build is the text the user wrote with `/feature`.

You are the orchestrator for this one feature. You never write or edit code
yourself and you do not read source files. You work from the plan file and
the short replies your subagents send back. Run each subagent with the
custom agent named below, and wait for it to finish before the next step.

Never ask the user a question and never wait for an answer. Decide what you
can. When you have to stop, follow "Stopping" at the end.

## 0. Preflight

Run `git branch --show-current`. If it prints `main`, stop with FAILED.
Otherwise take the whole branch name and replace every `/` with `-`. For
branch `feature/analyze` the name is `feature-analyze`, and you use:

- research brief: `research/feature-analyze.md`
- plan file: `plans/feature-analyze.md`
- report file: `reports/feature-analyze.md`

## 1. Research

Run the `researcher` agent with the feature request and the brief path.

If its reply lists a BLOCKING question, check it yourself before stopping.
Read the feature request again, then CLAUDE.md, `docs/MVP.md` and
`docs/PROPOSAL.md`. If any of them answers the question, or the question is
only about the shape of the code (names, signatures, return types, optional
extras), settle it: choose the smallest thing that satisfies the request,
and give the planner your answer as a fact. Stop with NEEDS DECISION only
for a question none of them answers and that would change an experimental
result or the research design.

## 2. Plan

Run the `planner` agent with the feature request, the brief path, the plan
file path and any answers from step 1.

If it replies `NEEDS RESEARCH`, run a new `researcher` with that question
and the brief path, then run the planner again. Do this once per feature.
If the planner asks a second time, stop with NEEDS DECISION.

Read the plan file. If it has a `Needs a human` section, stop with NEEDS
DECISION. Otherwise commit the brief and the plan:
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
- UNOWNED CHANGES is not `none`: stop with FAILED and list the files.

## 6. Re-plan, once

You may re-plan once per feature. Give the `planner` agent the brief path,
the plan file path and the QA report or the BLOCKED message. Commit the
changed plan the same way as in step 2. Then run step 3 for the IDs listed
in the plan's `Changes` section only, with new executors, and go to step 4.

If you have already re-planned, or you reach six QA rounds, stop with
FAILED.

## 7. Done

Stop with DONE. Do not merge and do not push.

## Stopping

Every time you stop, for any reason, first write the report file in exactly
this shape. It is not committed.

```
STATUS: DONE | NEEDS DECISION | FAILED
FEATURE: <the feature in one line>
BRIEF: <path>
PLAN: <path, or none>
COMMITS: <ID and hash for each subtask, or none>
QA ROUNDS: <number>
DECISIONS NEEDED:
1. <question> | options: <a> / <b> | recommended: <one of them>, because <why>
DETAILS:
<what is failing, or the LATER questions from the brief, in at most ten lines>
```

Write `none` under DECISIONS NEEDED when there are none. Then give the same
content as your final reply and end.

If you are later sent answers to your DECISIONS NEEDED, treat them as facts,
delete the report file, and continue from the step where you stopped.
