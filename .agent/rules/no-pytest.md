# RÈGLE CRITIQUE : INTERDICTION FORMELLE D'EXÉCUTER LES TESTS UNITAIRES

Il est STRICTEMENT INTERDIT à Antigravity ainsi qu'à tout sous-agent ou script d'exécuter des tests unitaires (pytest, unittest, etc.) sans l'accord explicite et préalable d'Henri.

## Core Principle
> Agents are strictly forbidden from executing tests automatically or as a precautionary verification step during development, build, or audit tasks, unless explicitly requested by the user.

## Required Behavior ✅
* Skip running `pytest` or any other test suite automatically during code modifications or audits.
* Execute test suites ONLY when the user has explicitly requested to run them in a message.
* Rely on code structure, compilation checks, imports validation, and manual code inspection instead of running tests to verify changes during standard development steps.

## Forbidden Patterns ❌
* Running `pytest`, `python -m pytest`, or equivalent test commands within any subagent or shell process without explicit user instructions.
