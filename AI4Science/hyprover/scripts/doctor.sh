#!/usr/bin/env bash
# Diagnostic checks for config, runtime, dependencies and services.
# The API backend performs real model generation and appends health logs.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/../scripts/env.sh"
ok(){ printf '  \033[32mok\033[0m   %s\n' "$1"; }
no(){ printf '  \033[31mFAIL\033[0m %s\n' "$1"; }
wn(){ printf '  \033[33mwarn\033[0m %s\n' "$1"; }
echo "== hyprover doctor =="
echo "HYPROVER_ROOT=$HYPROVER_ROOT"
[ -f "$HYPROVER_ROOT/config/hyprover.env" ] && ok "config/hyprover.env present" || wn "config/hyprover.env missing (copy from .example)"
[ -f "$HYPROVER_ROOT/config/codex.env" ] && ok "config/codex.env present" || wn "config/codex.env missing (copy from .example, fill BYO key)"
[ -n "${HYPROVER_PY:-}" ] && [ -x "$HYPROVER_PY" ] && ok "python: $HYPROVER_PY" || no "no python (run bootstrap.sh)"
"$HYPROVER_PY" -c 'from hyprover._mcp import FastMCP' 2>/dev/null && ok "python dep: mcp" || no "python dep: mcp missing or incompatible"
(cd / && "$HYPROVER_PY" -c 'import hyprover' 2>/dev/null) && ok "python pkg: hyprover (importable from any cwd)" || no "hyprover package not installed in the venv — worker MCP gateway and bin/ wrappers fail off the repo root (run bootstrap.sh)"
"$HYPROVER_PY" -c 'import fastapi, uvicorn, pydantic' 2>/dev/null && ok "python deps: fastapi/uvicorn/pydantic" || no "verify-service deps missing"
if [ -n "${HYPROVER_NODE:-}" ] && [ -x "${HYPROVER_NODE:-/nonexist}" ]; then ok "node: $HYPROVER_NODE"; else wn "node not provisioned"; fi
if "$HYPROVER_ROOT/bin/codex" --version >/dev/null 2>&1; then ok "codex: $("$HYPROVER_ROOT/bin/codex" --version 2>/dev/null)"; else no "codex wrapper not working (run bootstrap.sh)"; fi
if [ "${CODEX_BACKEND:-api}" = "api" ]; then
  if [ -f "$CODEX_HOME/config.toml" ] && grep -q hyprover_api "$CODEX_HOME/config.toml" 2>/dev/null; then ok "codex backend: api provider configured"; else no "codex api provider missing (scripts/setup-codex.sh api)"; fi
  bash "$HYPROVER_ROOT/scripts/check-codex.sh" >/dev/null 2>&1 && ok "codex API live ping ok" || wn "codex API ping FAILED (bash scripts/check-codex.sh — endpoint down/rate-limited?)"
else
  env CODEX_HOME="$CODEX_HOME" "$HYPROVER_ROOT/bin/codex" login status >/dev/null 2>&1 && ok "codex login ok ($CODEX_HOME)" || wn "codex not logged in (scripts/setup-codex.sh login)"
fi
case "$(hyprover_verify_health)" in
  ours)    ok "verify service up :$VERIFY_PORT (ours)" ;;
  foreign) no "verify port :$VERIFY_PORT answered by a FOREIGN process — another deployment holds this port; set a distinct VERIFY_PORT/DASHBOARD_PORT in config/hyprover.env (fact_submit would post to the wrong verifier)" ;;
  stale)   wn "verify service down :$VERIFY_PORT (stale pidfile; scripts/services.sh up verify)" ;;
  *)       wn "verify service down :$VERIFY_PORT (scripts/services.sh up verify)" ;;
esac
echo "done."
