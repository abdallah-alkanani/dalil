# Shared by up.sh and check.sh: the config, the paths, and small helpers.
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CONFIG=${DALIL_LOCAL_ENV:-$HOME/.config/dalil/local.env}
STATE=${DALIL_STATE_DIR:-$HOME/.local/state/dalil}
STORES="faiss_metadata.sqlite parents.sqlite bm25.sqlite dense_faiss_ar.index"

load_config() {
  [ -f "$CONFIG" ] || { echo "no config at $CONFIG (start from web/scripts/local.env.example)"; return 1; }
  set -a
  . "$CONFIG"
  set +a
  API_PORT=${API_PORT:-8000}
  SITE_PORT=${SITE_PORT:-3300}
  API="http://127.0.0.1:$API_PORT"
  SITE="http://127.0.0.1:$SITE_PORT"
}

# "ready", "failed: <detail>", or another state, from a /health JSON body on stdin.
health_state() {
  python3 -c 'import json,sys
raw = sys.stdin.read()
if not raw.strip(): print("starting"); raise SystemExit
try: d = json.loads(raw)
except Exception: print("unreadable"); raise SystemExit
s = d.get("search", "?")
print(s + (": " + d["detail"] if s != "ready" and d.get("detail") else "") + ("" if d.get("database", "ok") == "ok" else " | database " + d["database"]))'
}

listening() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

# The stores missing from the data directory, or "volume" when the directory itself is gone.
missing_stores() {
  [ -n "${DALIL_DATA_DIR:-}" ] && [ -d "$DALIL_DATA_DIR/db" ] || { echo volume; return; }
  for f in $STORES; do [ -f "$DALIL_DATA_DIR/db/$f" ] || printf '%s ' "$f"; done
}
