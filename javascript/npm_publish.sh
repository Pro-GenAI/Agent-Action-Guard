#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_JSON="${SCRIPT_DIR}/package.json"
DEFAULT_REGISTRY="https://registry.npmjs.org/"
REGISTRY="${NPM_REGISTRY:-$DEFAULT_REGISTRY}"
PACKAGE_MANAGER="${PACKAGE_MANAGER:-}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--dry-run] [--tag <tag>] [--access <public|restricted>]

Publishes the JavaScript package in:
  ${SCRIPT_DIR}

Options:
  --dry-run            Validate and pack, but do not publish
  --tag <tag>          Publish with the given dist-tag (default: latest)
  --access <value>     npm access level (default: public)
  -h, --help           Show this help text

Environment:
  NPM_TOKEN            Optional npm token. If set, this script writes a temporary
                       .npmrc in the package directory for this publish session.
  NPM_REGISTRY         Optional registry override
  PACKAGE_MANAGER      Optional package manager override (npm or pnpm)
EOF
}

DRY_RUN=0
DIST_TAG="latest"
ACCESS="public"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --tag)
      DIST_TAG="${2:-}"
      if [[ -z "$DIST_TAG" ]]; then
        echo "error: --tag requires a value" >&2
        exit 1
      fi
      shift 2
      ;;
    --access)
      ACCESS="${2:-}"
      if [[ -z "$ACCESS" ]]; then
        echo "error: --access requires a value" >&2
        exit 1
      fi
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$PACKAGE_MANAGER" ]]; then
  if [[ -f "$SCRIPT_DIR/pnpm-lock.yaml" ]] && command -v pnpm >/dev/null 2>&1; then
    PACKAGE_MANAGER="pnpm"
  else
    PACKAGE_MANAGER="npm"
  fi
fi

if ! command -v "$PACKAGE_MANAGER" >/dev/null 2>&1; then
  echo "error: ${PACKAGE_MANAGER} is not installed or not on PATH" >&2
  exit 1
fi

if [[ ! -f "$PACKAGE_JSON" ]]; then
  echo "error: package.json not found at ${PACKAGE_JSON}" >&2
  exit 1
fi

cd "$SCRIPT_DIR"

PACKAGE_NAME="$(node -p "require('./package.json').name")"
PACKAGE_VERSION="$(node -p "require('./package.json').version")"

echo "Package: ${PACKAGE_NAME}@${PACKAGE_VERSION}"
echo "Registry: ${REGISTRY}"
echo "Tag: ${DIST_TAG}"
echo "Access: ${ACCESS}"
echo "Package manager: ${PACKAGE_MANAGER}"

TEMP_NPMRC=""
cleanup() {
  if [[ -n "$TEMP_NPMRC" && -f "$TEMP_NPMRC" ]]; then
    rm -f "$TEMP_NPMRC"
  fi
}
trap cleanup EXIT

if [[ -n "${NPM_TOKEN:-}" ]]; then
  REGISTRY_HOST="${REGISTRY#https://}"
  REGISTRY_HOST="${REGISTRY_HOST#http://}"
  REGISTRY_HOST="${REGISTRY_HOST%/}"
  TEMP_NPMRC="${SCRIPT_DIR}/.npmrc.publish"
  cat >"$TEMP_NPMRC" <<EOF
registry=${REGISTRY}
//${REGISTRY_HOST}/:_authToken=${NPM_TOKEN}
EOF
  export NPM_CONFIG_USERCONFIG="$TEMP_NPMRC"
  echo "Using temporary npm auth from NPM_TOKEN"
fi

echo "Checking ${PACKAGE_MANAGER} authentication"
if ! "$PACKAGE_MANAGER" whoami --registry "$REGISTRY" >/dev/null 2>&1; then
  echo "error: ${PACKAGE_MANAGER} authentication failed. Run '${PACKAGE_MANAGER} login' or set NPM_TOKEN." >&2
  exit 1
fi

if [[ "$PACKAGE_MANAGER" == "pnpm" ]]; then
  echo "Installing dependencies from lockfile"
  pnpm install --frozen-lockfile

  echo "Running test suite"
  pnpm test

  echo "Previewing publish contents"
  pnpm publish --dry-run --no-git-checks
else
  echo "Installing dependencies from lockfile"
  npm ci

  echo "Running test suite"
  npm test

  echo "Previewing publish contents"
  npm pack --dry-run
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run complete. Package was validated but not published."
  exit 0
fi

echo "Publishing ${PACKAGE_NAME}@${PACKAGE_VERSION}"
if [[ "$PACKAGE_MANAGER" == "pnpm" ]]; then
  pnpm publish \
    --registry "$REGISTRY" \
    --tag "$DIST_TAG" \
    --access "$ACCESS" \
    --no-git-checks
else
  npm publish \
    --registry "$REGISTRY" \
    --tag "$DIST_TAG" \
    --access "$ACCESS"
fi

echo "Publish complete: ${PACKAGE_NAME}@${PACKAGE_VERSION}"
