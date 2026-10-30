# dalil on the web

| | |
|---|---|
| [`backend/`](backend/README.md) | FastAPI over `dalil.s8_search`: invite-only search, streamed cited answers, expert feedback, daily caps. Docker and Railway config. |
| [`frontend/`](frontend/README.md) | Next.js app: answers stream as OpenUI Lang into cited components; quotes are checked against their passages. |

Run the API on the stores in `data/db`, then the app against it:

```bash
cd web/backend && ./scripts/sync_dalil.sh        # copies the dalil package and the answering prompt into vendor/
DALIL_DATA_DIR=../../data ADMIN_TOKEN=... \
  PYTHONPATH=vendor .venv/bin/uvicorn dalil_api.main:app --port 8000

cd web/frontend && cp .env.example .env.local    # DALIL_API_URL
npm install && npm run dev                       # http://localhost:3300, sign in with an invite code
```
