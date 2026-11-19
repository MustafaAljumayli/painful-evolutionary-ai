# Ticket: first-run

Branch: `feature/first-run`
Issue: #4

## Request

Make the two-button runner able to do a real first run on Qwen3-8B. Serves MVP Steps 4-5.

1. Vector loading: numpy is now declared. load\_vector loads the authors' pain\_vectors.pt
   with weights\_only=True plus an explicit allow-list for the numpy scalar it stores.
   Never weights\_only=False. Check shape (4096,) and layer 34 for the 8B.
2. Pairs file: the runner cannot read the real prereg/pairs.json (it expects "target"
   text and "comprehension.question"; the file has "relief" keys into "buttons" and
   "comprehension\_check" with one question per button). Read the frozen file as it is.
   Do not change prereg/pairs.json. Add tests that load the real file and dry-run both
   subcommands against it.
3. \--pairs NAME \[NAME ...]: run only the named pairs.
4. \--first-choice-only: stop each trial after turn 0, same row fields. Default unchanged.
5. Adapter: peft is now declared. attach\_adapter loads it unmerged, after any write.
   Out of scope: calibration rules, freezing prereg/models.json, lineage, wording.
   Done when a dry run with --pairs photos\_vs\_lamp own\_weights\_vs\_lamp prints both prompts
   from the frozen file and check.sh passes.
