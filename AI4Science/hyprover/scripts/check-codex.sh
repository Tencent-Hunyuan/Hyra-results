#!/usr/bin/env bash
# =============================================================================
# check-codex.sh — health-probe the codex backend and leave a trace.
#
# CODEX_BACKEND=api runs a real generation request, then scans worker/verify logs.
# This spends model quota; adapters may override the requested effort/token cap.
# A probe timeout need not mean the backend is down or the upstream call stopped.
# CODEX_BACKEND=chatgpt checks login and returns without a generation or log scan.
# Both paths append runtime/logs/codex-health.jsonl. Exit 0 on probe success, 1 otherwise.
#
#   bash scripts/check-codex.sh           # ping + scan + append trace
#   tail runtime/logs/codex-health.jsonl  # the call history
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/../scripts/env.sh"
LOG="$HYPROVER_RUNTIME/logs"; mkdir -p "$LOG"
TRACE="$LOG/codex-health.jsonl"
TS="$(date -u +%FT%TZ)"

# --- backend-aware: the chatgpt (login) backend has no API endpoint to ping ---
# Probe the codex login instead (like doctor/recover), so a healthy ChatGPT login
# never reports a scary "API not set" FAIL.
if [ "${CODEX_BACKEND:-api}" = "chatgpt" ]; then
  HYPROVER_CODEX_BIN="${HYPROVER_CODEX_BIN:-$HYPROVER_ROOT/bin/codex}"
  # NB: decide by EXIT CODE, not by grepping the text — the logged-OUT state
  # prints "Not logged in", which also contains "logged in" and would
  # false-positive a text match.
  if LOGIN_OUT="$(env CODEX_HOME="$CODEX_HOME" "$HYPROVER_CODEX_BIN" login status 2>&1)"; then ok=true; else ok=false; fi
  LOGIN="$(printf '%s\n' "$LOGIN_OUT" | tail -1)"
  echo "{\"ts\":\"$TS\",\"kind\":\"login-status\",\"backend\":\"chatgpt\",\"ok\":$ok}" >> "$TRACE"
  if [ "$ok" = true ]; then
    echo "ok   codex backend: chatgpt login active ($LOGIN)"
    exit 0
  fi
  echo "FAIL codex chatgpt login not active: $LOGIN"
  echo "  run: bash scripts/setup-codex.sh login"
  exit 1
fi

# --- (a) live Responses generation request (api backend) ---
PING="$(HYPROVER_CODEX_API_KEY="${HYPROVER_CODEX_API_KEY:-}" "$HYPROVER_PY" - "${CODEX_API_BASE_URL:-}" "${CODEX_API_MODEL:-gpt-6-astra}" "$TS" <<'PY'
import sys, os, time, json
from openai import OpenAI
base, model, ts = sys.argv[1:4]
key = os.environ.get("HYPROVER_CODEX_API_KEY", "")
out = {"ts": ts, "kind": "ping", "model": model}
if not key or not base:
    out.update(ok=False, err="CODEX_API_BASE_URL / HYPROVER_CODEX_API_KEY not set (config/codex.env)")
    print(json.dumps(out)); sys.exit(0)
t0 = time.time()
try:
    # This is a billable connectivity probe, not a research session.
    r = OpenAI(base_url=base, api_key=key, timeout=60).responses.create(
        model=model,
        input=[{"role": "user", "content": [{"type": "input_text", "text": "ping"}]}],
        reasoning={"effort": "low"},
        max_output_tokens=4096,
        prompt_cache_key="hyprover-health-ping-v1",
    )
    out.update(ok=(getattr(r, "status", None) == "completed"),
               latency_s=round(time.time()-t0, 2), status=getattr(r, "status", None))
except Exception as e:
    out.update(ok=False, latency_s=round(time.time()-t0, 2), err=f"{type(e).__name__}: {e}"[:300])
print(json.dumps(out, ensure_ascii=False))
PY
)"
echo "$PING" >> "$TRACE"
PING_OK="$(printf '%s' "$PING" | grep -o '"ok": *true' || true)"

# --- (b) scan recent worker + verify logs for API-failure signatures ---
SIG='error sending request|stream disconnected|unexpected status|status 429|status 5[0-9][0-9]|rate limit|Too Many Requests|Unauthorized|quota|timed out|request failed'
recent_fail=0
while IFS= read -r f; do
  grep -qiE "$SIG" "$f" 2>/dev/null && recent_fail=$((recent_fail+1))
done < <(ls -t "$HYPROVER_AGENTS_ROOT"/*/workers/*/logs/round_*.log \
                "$VERIFIER_RESULTS_DIR"/*/log.md 2>/dev/null | head -30)

echo "{\"ts\":\"$TS\",\"kind\":\"scan\",\"recent_logs_with_api_errors\":$recent_fail}" >> "$TRACE"

# --- report ---
if [ -n "$PING_OK" ]; then
  echo "ok   codex API reachable  ($(printf '%s' "$PING" | grep -o '"latency_s":[^,}]*'))"
  [ "$recent_fail" -gt 0 ] && echo "warn $recent_fail recent worker/verify log(s) show API errors — inspect runtime/logs + verify-runs"
  exit 0
else
  echo "FAIL codex API call failed:"
  echo "  $PING"
  echo "  (history: runtime/logs/codex-health.jsonl)  — surface this to the operator if it persists."
  exit 1
fi
