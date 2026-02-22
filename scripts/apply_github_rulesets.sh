#!/usr/bin/env bash
set -euo pipefail

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI is required. Install and authenticate first."
  exit 1
fi

REPO="${1:-}"
if [ -z "${REPO}" ]; then
  REPO="$(gh repo view --json nameWithOwner --jq .nameWithOwner)"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MASTER_SPEC="${ROOT_DIR}/.github/rulesets/master.json"
TAGS_SPEC="${ROOT_DIR}/.github/rulesets/tags_vstar.json"

if [ ! -f "${MASTER_SPEC}" ] || [ ! -f "${TAGS_SPEC}" ]; then
  echo "Ruleset spec files not found under .github/rulesets/"
  exit 1
fi

gh auth status -h github.com >/dev/null

echo "Applying repository merge settings on ${REPO} (squash-only)..."
gh api -X PATCH "repos/${REPO}" \
  -f allow_squash_merge=true \
  -f allow_merge_commit=false \
  -f allow_rebase_merge=false >/dev/null

apply_ruleset() {
  local name="$1"
  local target="$2"
  local spec="$3"
  local existing_id

  existing_id="$(gh api "repos/${REPO}/rulesets" --jq ".[] | select(.name==\"${name}\" and .target==\"${target}\") | .id" | head -n1 || true)"
  if [ -n "${existing_id}" ]; then
    gh api -X PUT "repos/${REPO}/rulesets/${existing_id}" --input "${spec}" >/dev/null
    echo "Updated ruleset: ${name} (id=${existing_id})"
  else
    existing_id="$(gh api -X POST "repos/${REPO}/rulesets" --input "${spec}" --jq .id)"
    echo "Created ruleset: ${name} (id=${existing_id})"
  fi
}

apply_ruleset "Protect master" "branch" "${MASTER_SPEC}"
apply_ruleset "Protect release tags v*" "tag" "${TAGS_SPEC}"

echo "Done."
gh ruleset check master -R "${REPO}" || true
