#!/usr/bin/env bash
set -euo pipefail

# Enforce project policy: every commit must stage CHANGELOG.md.
staged="$(git diff --cached --name-only --diff-filter=ACMR)"
if [[ -z "$staged" ]]; then
  exit 0
fi

if ! grep -Eq '(^|/)CHANGELOG\.md$' <<<"$staged"; then
  echo "ERROR: CHANGELOG.md must be updated and staged for every commit." >&2
  echo "Hint: add a short entry under [Unreleased] before committing." >&2
  exit 1
fi
