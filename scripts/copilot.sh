#!/usr/bin/env bash
# Starts Copilot CLI with the approvals the /feature loop needs, so it does
# not stop to ask before each file edit, check run, commit or source lookup.
# Extra arguments pass straight through, for example:
#   ./scripts/copilot.sh --model <id>
#
# --autopilot keeps Copilot working until the loop reports done. Remove it
# if you would rather it hand control back after every turn.
# The --allow-url lines are the sites the researcher reads the prior work from.
exec copilot \
  --autopilot \
  --max-autopilot-continues 20 \
  --allow-tool='write' \
  --allow-tool='shell(./scripts/check.sh)' \
  --allow-tool='shell(git add)' \
  --allow-tool='shell(git commit)' \
  --deny-tool='shell(git push)' \
  --allow-url='arxiv.org' \
  --allow-url='github.com' \
  --allow-url='raw.githubusercontent.com' \
  --allow-url='huggingface.co' \
  "$@"
