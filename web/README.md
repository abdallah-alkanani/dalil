# dalil on the web

| | |
|---|---|
| [`backend/`](backend/README.md) | FastAPI over `dalil.s8_search`: search, streamed cited answers, feedback and a daily cap for the whole site; who asks and from which countries. Docker and Railway config. |
| [`frontend/`](frontend/README.md) | Next.js app, open to everyone: answers stream as OpenUI Lang into cited components; quotes are shown in their passages' own text. |

Run the API on the stores in `data/db`, then the app against it:

```bash
cd web/backend
DALIL_DATA_DIR=../../data ADMIN_TOKEN=... SITE_TOKEN=... \
  PYTHONPATH=../.. .venv/bin/uvicorn dalil_api.main:app --port 8000

cd web/frontend && cp .env.example .env.local    # DALIL_API_URL, SITE_TOKEN
npm install && npm run dev                       # http://localhost:3300
```

## On one machine

`web/scripts/up.sh` brings the site up on a machine whose stores are on a drive that may be
unplugged: it stops with a clear message when the data volume is not mounted, starts the API
and waits for search to be ready, starts the web app and the ngrok tunnel, and asks one
question to confirm the site answers. `web/scripts/check.sh` says which piece is down. Both
read `~/.config/dalil/local.env` (template: `web/scripts/local.env.example`), which keeps the
keys, the app's SQLite database and the country database on the internal disk.

