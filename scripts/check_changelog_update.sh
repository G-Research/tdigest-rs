#!/usr/bin/env bash
set -euo pipefail

if git diff --cached --quiet; then
  exit 0
fi

if git diff --cached --quiet --diff-filter=AM -- CHANGELOG.md; then
  echo "Stage a change to the root CHANGELOG.md before committing." >&2
  exit 1
fi
