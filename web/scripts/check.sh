#!/bin/sh
# Say which piece of the local site is down: the data volume, the keys, the API, the
# web app or the tunnel. Reads ~/.config/dalil/local.env (see local.env.example).
#
#     web/scripts/check.sh
. "$(dirname "$0")/common.sh"
FIRST=""
ok()   { printf '  ok    %-13s %s\n' "$1" "$2"; }
down() { printf '  DOWN  %-13s %s\n' "$1" "$2"; [ -n "$FIRST" ] || FIRST="$1: $2"; }

load_config || exit 1
ok config "$CONFIG"

missing=$(missing_stores)
if [ "$missing" = volume ]; then down "data volume" "$DALIL_DATA_DIR/db is not there: connect the drive"
elif [ -n "$missing" ]; then down "data volume" "mounted, but missing: $missing"
else ok "data volume" "$DALIL_DATA_DIR/db"; fi

keys=""
for k in SITE_TOKEN ADMIN_TOKEN OPENROUTER_API_KEY OLLAMA_API_KEY; do eval "[ -n \"\${$k:-}\" ]" || keys="$keys $k"; done
if [ -n "$keys" ]; then down keys "not set in $CONFIG:$keys"; else ok keys "all set"; fi

if [ -x "$ROOT/web/backend/.venv/bin/python" ]; then ok "api python" "web/backend/.venv"
else down "api python" "web/backend/.venv is missing (up.sh creates it)"; fi

if ! listening "$API_PORT"; then down api "nothing listening on $API_PORT (run web/scripts/up.sh)"
else
  state=$(curl -s -m 5 "$API/health" | health_state)
  case "$state" in ready) ok api "search ready on $API_PORT" ;; *) down api "$state" ;; esac
fi

if ! listening "$SITE_PORT"; then down site "nothing listening on $SITE_PORT"
else
  state=$(curl -s -m 5 "$SITE/api/health" | health_state)
  case "$state" in
    ready) ok site "on $SITE_PORT, reaches the API" ;;
    unreachable*|offline*) down site "on $SITE_PORT, but cannot reach the API (DALIL_API_URL in web/frontend/.env.local?)" ;;
    *) down site "on $SITE_PORT, API says: $state" ;;
  esac
fi

if [ -n "${PUBLIC_URL:-}" ]; then
  target=$(curl -s -m 5 http://127.0.0.1:4040/api/tunnels | python3 -c 'import json,sys
url = sys.argv[1].rstrip("/")
try: ts = json.load(sys.stdin)["tunnels"]
except Exception: ts = []
print(next((t["config"]["addr"] for t in ts if t["public_url"].rstrip("/") == url), ""))' "$PUBLIC_URL")
  case "$target" in
    "") down tunnel "no ngrok tunnel for $PUBLIC_URL" ;;
    *":$SITE_PORT") ok tunnel "$PUBLIC_URL -> $target" ;;
    *) down tunnel "$PUBLIC_URL points at $target, not port $SITE_PORT" ;;
  esac
  code=$(curl -s -o /dev/null -m 15 -w '%{http_code}' -H 'ngrok-skip-browser-warning: 1' "$PUBLIC_URL/")
  if [ "$code" = 200 ]; then ok public "$PUBLIC_URL answers 200"; else down public "$PUBLIC_URL answers $code"; fi
fi

if [ -n "$FIRST" ]; then echo "down first: $FIRST"; exit 1; fi
echo "all up"
