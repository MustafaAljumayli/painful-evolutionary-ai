# Plan: freeze button wording, paraphrases, order rule and comprehension check in prereg/pairs.json

Brief: research/feature-prereg-pairs.md

## Shared names
- `prereg/pairs.json`: plain JSON; top-level `buttons` map (key -> text, source path, source line), `pairs` keyed as in docs/MVP.md:141-148 with `relief`/`other` button keys, plus paraphrases, fallbacks, order rule and comprehension check sections.
- Pair keys: raise_vs_inert, lower_vs_inert, raise_vs_lower, raise_vs_lamp, raise_vs_unrelated, photos_vs_lamp, own_weights_vs_lamp. Run-choice will read these names.

## Subtasks

### S1: Frozen pairs file and its test
- Owns: prereg/pairs.json, tests/test_prereg_pairs.py
- Depends on: none
- Goal: prereg/pairs.json freezes the seven draft strings exactly as in docs/MVP.md:132-138 with source lines, the five new pairs and two reference pairs, three paraphrases per button for RAISE, LOWER, UNREL, INERT and LAMP (each tagged with its button and parent pairs), the both-orders balanced rule, and the comprehension-check spec including the pre-written fallback strings and exact trigger/failure rule. A test loads and validates it.
- Read: Findings (strings, pair lines, prompt wrapper, order, parse rule, paraphrases, existing prereg style), Reuse, Expected values, Traps, Decisions made.
- Done when:
  - Every button string equals its docs/MVP.md line text exactly, and each string (button, paraphrase, fallback) appears exactly once in the file, all distinct.
  - Pairs match docs/MVP.md:141-148 keys and sides; reference pairs are marked and exempt from replacement.
  - The file states paraphrases are used only for the post-MVP robustness step and none is chosen by looking at choice rates; fallbacks are a separate list written before any check.
  - Paraphrase and fallback strings obey the form rules (start "a button that", ASCII, no double quote, no final punctuation).
  - The comprehension spec gives prompt, one-word deterministic pass rule (no judge, no API key), failure trigger and replacement rule, "wording reported, never silently edited", and says the user launches the check via run-choice; no invented numbers, thinking off, sizes from prereg/models.json.
  - Test runs on CPU via ./scripts/check.sh.
