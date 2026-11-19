---
name: solo
description: Control the Solo app (soloterm.com) from the shell with its `solo` command-line tool, without MCP. Use when you need to start, message, read, wait for or close agents and processes in Solo, or to read and update Solo projects, todos and scratchpads.
---

Solo is the desktop app that runs this project's agents and processes. You
control it with its own command-line tool, `solo`. Do not use, enable or ask
for Solo's MCP server: this setup works without MCP.

Every command below runs in the shell. Add `--json` when you need to read
IDs or fields from the result. If you are unsure of a flag, run
`solo help <group> <command>`, for example `solo help processes spawn`.

## Before the first command

Run `solo doctor`. If it fails, stop and tell the user:

1. Solo must be running.
2. In Solo, open Settings > Integrations > HTTP API & CLI, turn on HTTP API,
   and run the install snippet shown there so that `solo` is on the PATH.

## Who you are

When Solo started you, it set two environment variables:

- `SOLO_PROCESS_ID`: your own process ID.
- `SOLO_PROJECT_ID`: the ID of the project you run in.

Read them with `echo "$SOLO_PROCESS_ID $SOLO_PROJECT_ID"`. Never stop,
delete or send input to your own process.

## Commands

IDs are numbers. Most commands need `--project-id <id>`; a process in
another project is only found when you pass that project's ID.

| To do this | Run |
|---|---|
| List projects and their folders | `solo projects list --json` |
| Add a folder as a project | `solo projects create <name> <existing folder>` |
| List the agent tools and their IDs | `solo agents list --json` |
| List processes in a project | `solo processes list --project-id <p> --json` |
| See one process (kind, status, uptime) | `solo processes get <id> --json` |
| Start an agent | `solo processes spawn --project-id <p> --kind agent --agent-tool-id <tool id> --name <name> --json` |
| Start an agent with extra arguments | add `--arg <argument>` once per argument |
| Start a plain terminal | `solo processes spawn --project-id <p> --kind terminal --name <name>` |
| Read the end of a process's output | `solo processes output <id> --project-id <p> --lines 60` |
| Send a message to an agent | `solo processes input <id> --project-id <p> --text "<message>"` |
| Send a long message | `solo processes input <id> --project-id <p> --text - < message.txt` |
| Press Enter only | `solo processes input <id> --project-id <p> --bytes 13` |
| Press Ctrl+C | `solo processes input <id> --project-id <p> --bytes 3` |
| Stop a process | `solo processes stop <id>` |
| Close a process for good | `solo processes delete <id> --project-id <p> --confirm-stop-running` |
| List open todos | `solo todos list --project-id <p> --status open --json` |
| Read one todo | `solo todos get <todo id> --project-id <p> --json` |
| Create a todo | `solo todos create --project-id <p> --title "<title>" --body-file <file or ->` |
| Set a todo's status | `solo todos update <todo id> --project-id <p> --status <open, in_progress, backlog or completed>` |
| Complete a todo | `solo todos complete <todo id> --project-id <p>` |
| List scratchpads | `solo scratchpads list --project-id <p> --json` |
| Read a scratchpad | `solo scratchpads read <id> --project-id <p>` |
| Create a scratchpad | `solo scratchpads create --project-id <p> --name <name> --content-file <file or ->` |
| Add to a scratchpad | `solo scratchpads append <id> --project-id <p> --expected-revision <n> --newline --content "<text>"` |

A process's status is one of `stopped`, `starting`, `running`, `stopping`,
`exited` or `failed`. An agent that has finished its turn and waits at its
prompt is still `running`.

## Waiting for an agent

The command-line tool has no timers and does not say when an agent is idle.
To wait, run:

```
./scripts/solo-wait.sh [--project-id <p>] <process id> [<process id> ...]
```

It returns when one of the processes has printed nothing new for 30 seconds
or has exited, and prints `IDLE <id>` with the end of its output. After a
few minutes without that, it prints `STILL WORKING` and returns; run it
again. Do not write your own polling loop and do not use `sleep`.

Idle only means quiet. Read the output it prints before you decide what the
agent is doing: it may have finished, or it may be waiting for an answer.

## Rules

- A new agent ignores input until its startup screen has gone. After you
  spawn one, wait for it with `solo-wait.sh` before you send it a message,
  then read its output to check that the message was submitted. If the text
  sits unsent in its input box, press Enter once.
- Never answer another agent's approval prompt and never change its
  permission mode unless the user told you to. Report it instead.
- Do not send the same message twice. If a send reports that delivery is
  uncertain, read the agent's output before doing anything else.
- Close the agents you started when their work is recorded. Solo limits how
  many processes run at once.
