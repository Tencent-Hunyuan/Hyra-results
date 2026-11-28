#!/usr/bin/env bash
# =============================================================================
# hyprover environment — source this in any shell that talks to the system.
#
#   source <repo>/scripts/env.sh
#
# Loads config/codex.env, config/hyprover.env and runtime/runtime.env
# in that order, then fills unset defaults and puts bin/ + the provisioned node +
# venv on PATH. The bin/ wrappers source this for you, so `hyprover` and `codex`
# work without you sourcing it manually. Sourcing twice is harmless
# (idempotent).
# =============================================================================

# Self-locate the repo root (this file lives at <repo>/scripts/env.sh).
HYPROVER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export HYPROVER_ROOT

# 0) codex backend (BYO OpenAI-compatible endpoint + key; gitignored)
if [ -f "$HYPROVER_ROOT/config/codex.env" ]; then
  set -a; . "$HYPROVER_ROOT/config/codex.env"; set +a
fi

# 1) user config (accounts / models / ports / toggles) — overrides the codex file
if [ -f "$HYPROVER_ROOT/config/hyprover.env" ]; then
  set -a; . "$HYPROVER_ROOT/config/hyprover.env"; set +a
fi

# 2) machine-derived paths written by bootstrap.sh (node, codex.js, venv)
if [ -f "$HYPROVER_ROOT/runtime/runtime.env" ]; then
  set -a; . "$HYPROVER_ROOT/runtime/runtime.env"; set +a
fi

# 3) defaults for anything still unset
export HYPROVER_RUNTIME="${HYPROVER_RUNTIME:-$HYPROVER_ROOT/runtime}"
export HYPROVER_AGENTS_ROOT="${HYPROVER_AGENTS_ROOT:-$HYPROVER_RUNTIME/projects}"
export VERIFIER_RESULTS_DIR="${VERIFIER_RESULTS_DIR:-$HYPROVER_RUNTIME/verify-runs}"
export CODEX_HOME="${CODEX_HOME:-$HYPROVER_RUNTIME/codex-home}"
export VERIFY_PORT="${VERIFY_PORT:-8091}"
export DASHBOARD_PORT="${DASHBOARD_PORT:-8099}"
export HYPROVER_VERIFY_URL="${HYPROVER_VERIFY_URL:-http://127.0.0.1:${VERIFY_PORT}/verify}"
export HYPROVER_MAIN_MODEL="${HYPROVER_MAIN_MODEL:-${HYPROVER_CODEX_MODEL:-${CODEX_API_MODEL:-gpt-6-astra}}}"
export HYPROVER_MAIN_EFFORT="${HYPROVER_MAIN_EFFORT:-${HYPROVER_CODEX_EFFORT:-max}}"
export HYPROVER_WORKER_MODEL="${HYPROVER_WORKER_MODEL:-$HYPROVER_MAIN_MODEL}"   # worker-only model (defaults to the neutral model above)
# back-compat: keep the old names populated for any shell consumer that still
# expands them (Python resolves via the aliases in hyprover.codex regardless).
export HYPROVER_CODEX_MODEL="$HYPROVER_MAIN_MODEL"
export HYPROVER_CODEX_EFFORT="$HYPROVER_MAIN_EFFORT"
export CODEX_BACKEND="${CODEX_BACKEND:-api}"            # api (BYO key) | chatgpt (your login)

# 4) PATH: bin wrappers first, then the provisioned node + venv (if bootstrapped)
_hyprover_path="$HYPROVER_ROOT/bin"
[ -n "${HYPROVER_NODE_BIN:-}" ] && [ -d "$HYPROVER_NODE_BIN" ] && _hyprover_path="$_hyprover_path:$HYPROVER_NODE_BIN"
[ -n "${HYPROVER_VENV:-}" ]     && [ -d "$HYPROVER_VENV/bin" ] && _hyprover_path="$_hyprover_path:$HYPROVER_VENV/bin"
case ":$PATH:" in *":$_hyprover_path:"*) : ;; *) export PATH="$_hyprover_path:$PATH" ;; esac

# 5) the python the engine runs on (venv if bootstrapped, else system python3)
if [ -n "${HYPROVER_VENV:-}" ] && [ -x "$HYPROVER_VENV/bin/python" ]; then
  export HYPROVER_PY="$HYPROVER_VENV/bin/python"
else
  export HYPROVER_PY="${HYPROVER_PY:-$(command -v python3 || true)}"
fi

# silent unless HYPROVER_ENV_VERBOSE=1
if [ "${HYPROVER_ENV_VERBOSE:-0}" = "1" ]; then
  echo "HYPROVER_ROOT=$HYPROVER_ROOT"
  echo "HYPROVER_PY=$HYPROVER_PY"
  echo "HYPROVER_AGENTS_ROOT=$HYPROVER_AGENTS_ROOT"
  echo "HYPROVER_VERIFY_URL=$HYPROVER_VERIFY_URL"
  echo "CODEX_HOME=$CODEX_HOME"
fi

# 6) verify-service identity probe (shared by doctor.sh / services.sh).
# A bare `/health` 200 does NOT prove the responder is OURS: on a shared host a
# second hyprover deployment can hold the same VERIFY_PORT, and its verifier would
# answer our probe (a false "up" that silently routes fact_submit to the wrong
# verifier). This classifies the port by matching the pid /health self-reports
# against our own runtime/run/verify.pid. Echoes one word + sets a code:
#   ours    (0)  -> healthy AND the responder is our pidfile's process
#   foreign (3)  -> something answers, but it is not ours (port collision)
#   stale   (4)  -> our pidfile process is dead and nothing answers the port
#   down    (5)  -> nothing answers and we have no pidfile process
hyprover_verify_health(){
  local url="http://127.0.0.1:${VERIFY_PORT}/health"
  local pf="$HYPROVER_RUNTIME/run/verify.pid"
  local our_pid resp health_pid
  our_pid="$(cat "$pf" 2>/dev/null || true)"
  resp="$(curl -s --max-time 5 "$url" 2>/dev/null || true)"
  if [ -z "$resp" ]; then
    if [ -n "$our_pid" ] && kill -0 "$our_pid" 2>/dev/null; then echo down; return 5; fi
    { [ -n "$our_pid" ]; } && { echo stale; return 4; }
    echo down; return 5
  fi
  # pull "pid": <n> out of the JSON without a json dep
  health_pid="$(printf '%s' "$resp" | sed -n 's/.*"pid"[[:space:]]*:[[:space:]]*\([0-9]\{1,\}\).*/\1/p')"
  if [ -n "$our_pid" ] && [ -n "$health_pid" ] && [ "$our_pid" = "$health_pid" ]; then
    echo ours; return 0
  fi
  # answered, but not by our pidfile's process (or an old build with no pid field)
  echo foreign; return 3
}
