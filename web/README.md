# dalil on the web

| | |
|---|---|
| [`backend/`](backend/README.md) | FastAPI over `dalil.s8_search`: search, streamed cited answers, feedback and daily caps for every visitor; who asks and from which countries. Docker and Railway config. |
| [`frontend/`](frontend/README.md) | Next.js app, open to everyone: answers stream as OpenUI Lang into cited components; quotes are shown in their passages' own text. |

Run the API on the stores in `data/db`, then the app against it:

```bash
cd web/backend && ./scripts/sync_dalil.sh        # copies the dalil package and the answering prompt into vendor/
DALIL_DATA_DIR=../../data ADMIN_TOKEN=... SITE_TOKEN=... \
  PYTHONPATH=vendor .venv/bin/uvicorn dalil_api.main:app --port 8000

cd web/frontend && cp .env.example .env.local    # DALIL_API_URL, SITE_TOKEN
npm install && npm run dev                       # http://localhost:3300
```
