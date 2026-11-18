# Plan: Record the local-hardware scope deviation for the MVP in the project docs

Brief: research/feature-scope-local-mvp.md

## Shared names
none

## Subtasks

### S1: Scope-deviation wording in the four context docs (ALREADY IMPLEMENTED)
- Owns: CLAUDE.md, AGENTS.md, docs/MVP.md, docs/PROPOSAL.md
- Depends on: none
- Goal: The edits already exist uncommitted in the working tree (made by the dispatcher). No executor writes new content; this subtask is only verified by QA and committed. The four files state the local-hardware scope deviation as the brief describes it. PROPOSAL.md Experiments 1-3 keep their original wording and only the added scope note differs.
- Read: Findings on the scope deviation and the figures, Gaps (1-4, now fixed), Traps
- Done when:
  - Every figure and size named in the four files matches the brief.
  - The scope wording appears in all four files and is consistent between them.
  - No change touches prereg/, button description strings, or any sourced number (paper, authors' logs, runs/).
  - PROPOSAL.md Experiments 1-3 are not reworded and the added scope note is kept.
  - CLAUDE.md and AGENTS.md are byte-identical, and `./scripts/check.sh` exits 0.
