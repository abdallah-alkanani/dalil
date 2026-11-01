#!/bin/sh
# Bring the local site up: check the data volume, start the API and wait for search to
# be ready, start the web app and the ngrok tunnel, then ask one question to confirm the
# site answers. Pieces already running are left as they are. Reads
# ~/.config/dalil/local.env (see local.env.example); logs go to ~/.local/state/dalil.
#
#     web/scripts/up.sh            # --no-ask: skip the test question
set -eu
. "$(dirname "$0")/common.sh"
say() { printf '%s\n' "$*"; }
die() { printf 'up: %s\n' "$*" >&2; exit 1; }

load_config || die "no config"
mkdir -p "$STATE"

missing=$(missing_stores)
[ "$missing" != volume ] || die "the data volume is not mounted: $DALIL_DATA_DIR/db is not there. Connect the drive, then run this again."
[ -z "$missing" ] || die "the data volume is mounted but $DALIL_DATA_DIR/db is missing: $missing"
say "data volume: $DALIL_DATA_DIR/db"

if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -n "${DALIL_KEYS_FROM:-}" ] && [ -f "$DALIL_KEYS_FROM" ]; then
  line=$(grep '^OPENROUTER_API_KEY=.' "$DALIL_KEYS_FROM" || true)
  if [ -n "$line" ]; then
    printf '%s\n' "$line" >> "$CONFIG" && chmod 600 "$CONFIG"
    say "keys: copied OPENROUTER_API_KEY from $DALIL_KEYS_FROM into $CONFIG"
    load_config
  fi
fi
for k in SITE_TOKEN ADMIN_TOKEN OLLAMA_API_KEY; do eval "[ -n \"\${$k:-}\" ]" || die "$k is not set in $CONFIG"; done
[ -n "${OPENROUTER_API_KEY:-}" ] || say "keys: OPENROUTER_API_KEY is not set; search will match words only, not meaning"

PY="$ROOT/web/backend/.venv/bin/python"
if [ ! -x "$PY" ]; then
  say "api python: creating web/backend/.venv"
  uv venv -q --python 3.12 "$ROOT/web/backend/.venv"
  uv pip install -q --python "$PY" -r "$ROOT/web/backend/requirements.txt"
fi
if [ -n "${GEOIP_DB:-}" ] && [ ! -f "$GEOIP_DB" ]; then
  "$PY" "$ROOT/web/backend/scripts/download_country_db.py" "$GEOIP_DB"
fi
case "${DATABASE_URL:-}" in sqlite:///*) mkdir -p "$(dirname "${DATABASE_URL#sqlite:///}")" ;; esac

if listening "$API_PORT" && curl -s -m 5 "$API/health" | health_state | grep -q '^failed'; then
  # Started while the volume was missing (or failed otherwise): it never retries, so restart it.
  say "api: restarting the API on $API_PORT, which failed to load search"
  kill $(lsof -t -nP -iTCP:"$API_PORT" -sTCP:LISTEN) 2>/dev/null || true
  for _ in $(seq 1 15); do listening "$API_PORT" || break; sleep 1; done
fi
if listening "$API_PORT"; then
  say "api: already running on $API_PORT"
else
  say "api: starting on $API_PORT (log: $STATE/api.log)"
  (cd "$ROOT/web/backend" && PYTHONPATH="$ROOT" nohup "$PY" -m uvicorn dalil_api.main:app \
    --host 127.0.0.1 --port "$API_PORT" >> "$STATE/api.log" 2>&1 & echo $! > "$STATE/api.pid")
fi
last=""
for _ in $(seq 1 900); do
  state=$(curl -s -m 5 "$API/health" | health_state 2>/dev/null || true)
  [ "$state" = "$last" ] || { [ -z "$state" ] || say "api: $state"; last=$state; }
  case "$state" in
    ready) break ;;
    failed*) tail -5 "$STATE/api.log" >&2; die "search failed to start: $state" ;;
  esac
  sleep 2
done
[ "$state" = ready ] || die "search was not ready after 30 minutes: $state"

ENV_LOCAL="$ROOT/web/frontend/.env.local"
(umask 077 && printf 'DALIL_API_URL=%s\nSITE_TOKEN=%s\nOPENUI_TELEMETRY_DISABLED=1\nDO_NOT_TRACK=1\nNEXT_TELEMETRY_DISABLED=1\n' \
  "$API" "$SITE_TOKEN" > "$ENV_LOCAL")
if listening "$SITE_PORT"; then
  say "site: already running on $SITE_PORT"
else
  head=$(git -C "$ROOT" rev-parse HEAD)
  if [ ! -f "$ROOT/web/frontend/.next/BUILD_ID" ] || [ "$(cat "$STATE/site_build" 2>/dev/null)" != "$head" ]; then
    say "site: building $(git -C "$ROOT" rev-parse --short HEAD)"
    (cd "$ROOT/web/frontend" && rm -rf .next && npm run build >> "$STATE/site.log" 2>&1) || die "site build failed (see $STATE/site.log)"
    echo "$head" > "$STATE/site_build"
  fi
  say "site: starting on $SITE_PORT (log: $STATE/site.log)"
  (cd "$ROOT/web/frontend" && OPENUI_TELEMETRY_DISABLED=1 DO_NOT_TRACK=1 NEXT_TELEMETRY_DISABLED=1 \
    nohup ./node_modules/.bin/next start --port "$SITE_PORT" --hostname 127.0.0.1 >> "$STATE/site.log" 2>&1 &)
  for _ in $(seq 1 30); do listening "$SITE_PORT" && break; sleep 1; done
fi

BASE=$SITE
if [ -n "${PUBLIC_URL:-}" ]; then
  if curl -s -m 5 http://127.0.0.1:4040/api/tunnels | grep -q "\"public_url\":\"${PUBLIC_URL%/}\""; then
    say "tunnel: already up: $PUBLIC_URL"
  else
    say "tunnel: starting $PUBLIC_URL -> $SITE_PORT"
    nohup ngrok http "127.0.0.1:$SITE_PORT" --url "$PUBLIC_URL" --log "$STATE/ngrok.log" > /dev/null 2>&1 &
    sleep 5
  fi
  BASE=${PUBLIC_URL%/}
fi

[ "${1:-}" = "--no-ask" ] && { say "up: $BASE (test question skipped)"; exit 0; }
jar=$(mktemp)
answer=$(curl -sN -m 180 -b "$jar" -c "$jar" -H 'ngrok-skip-browser-warning: 1' -H 'content-type: application/json' \
  -X POST "$BASE/api/ask" -d '{"question":"ما حكم صلاة الجماعة؟"}' || true)
rm -f "$jar"
if printf '%s' "$answer" | grep -q '"type": *"done"'; then
  say "up: $BASE answers questions"
else
  die "the site did not answer the test question: $(printf '%s' "$answer" | tail -1 | cut -c1-300)"
fi
