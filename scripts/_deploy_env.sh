# Shared connection setup for the deployment scripts. Source, do not execute.
#
#   . "$(dirname "${BASH_SOURCE[0]}")/_deploy_env.sh"
#
# Afterwards: SSH_OPTS is an array of ssh arguments, SSH is a full ssh command
# array, and TARGET is user@host. Use them as "${SSH[@]}" so paths containing
# spaces survive.
#
# Exists because three scripts were each building these arguments inline and
# each carried the same two faults.
#
# First, a tilde read from .env is a literal character, not the home directory.
# The shell expands ~ only in source text, never in a value it read from a file,
# so FORECAST_DEPLOY_KEY=~/.ssh/id_ed25519 produced a path that cannot exist and
# an error naming a file the user can plainly see is there.
#
# Second, conditional expansions like ${VAR:+-i "$VAR"} rely on word splitting
# to become two arguments. bash splits; zsh does not, and zsh is the default
# shell on macOS, where these scripts are run. The result was a single mangled
# argument and an ssh usage message, which says nothing about the cause.

# shellcheck shell=bash

_expand_tilde() {
  case "$1" in
    "~")   printf '%s' "$HOME" ;;
    "~/"*) printf '%s' "$HOME/${1#\~/}" ;;
    *)     printf '%s' "$1" ;;
  esac
}

_deploy_env_init() {
  local repo_root
  repo_root="$(cd "$(dirname "${BASH_SOURCE[1]}")/.." && pwd)"

  if [ -f "$repo_root/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$repo_root/.env"
    set +a
  fi

  local missing=()
  [ -z "${FORECAST_DEPLOY_HOST:-}" ] && missing+=("FORECAST_DEPLOY_HOST")
  [ -z "${FORECAST_DEPLOY_USER:-}" ] && missing+=("FORECAST_DEPLOY_USER")
  [ -z "${FORECAST_DEPLOY_PATH:-}" ] && missing+=("FORECAST_DEPLOY_PATH")

  if [ ${#missing[@]} -gt 0 ]; then
    echo "Missing configuration: ${missing[*]}" >&2
    echo "Set them in ${repo_root}/.env, or in the environment." >&2
    echo >&2
    echo "Currently resolved:" >&2
    echo "  FORECAST_DEPLOY_HOST=${FORECAST_DEPLOY_HOST:-<unset>}" >&2
    echo "  FORECAST_DEPLOY_USER=${FORECAST_DEPLOY_USER:-<unset>}" >&2
    echo "  FORECAST_DEPLOY_PATH=${FORECAST_DEPLOY_PATH:-<unset>}" >&2
    return 1
  fi

  TARGET="${FORECAST_DEPLOY_USER}@${FORECAST_DEPLOY_HOST}"
  SSH_OPTS=(-p "${FORECAST_DEPLOY_PORT:-22}" -o StrictHostKeyChecking=accept-new)

  if [ -n "${FORECAST_DEPLOY_KEY:-}" ]; then
    local key
    key="$(_expand_tilde "$FORECAST_DEPLOY_KEY")"
    if [ ! -f "$key" ]; then
      echo "FORECAST_DEPLOY_KEY points at ${key}, which does not exist." >&2
      [ "$key" != "$FORECAST_DEPLOY_KEY" ] &&
        echo "(expanded from ${FORECAST_DEPLOY_KEY})" >&2
      return 1
    fi
    SSH_OPTS+=(-i "$key")
  fi

  SSH=(ssh "${SSH_OPTS[@]}" "$TARGET")
  return 0
}

_deploy_env_init || return 1
