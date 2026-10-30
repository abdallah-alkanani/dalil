# Daleel on the web

| | |
|---|---|
| [`backend/`](backend/README.md) | FastAPI over `dalil.s8_search`: invite-only search, cited answers, expert feedback, daily spend caps. Docker and Railway config. |
| [`frontend/`](frontend/README.md) | Next.js app: answers stream as OpenUI Lang into cited components; quotes are checked against their passages. |

Run the API on the stores in `../data/db`, then the app against it:

```bash
cd web/backend && ./scripts/sync_dalil.sh        # copies the dalil package into vendor/
DALIL_DATA_DIR=../../data DALIL_DENSE_COMPACT_DIMS=1024 ADMIN_TOKEN=... \
  PYTHONPATH=vendor .venv/bin/uvicorn daleel_api.main:app --port 8000

cd web/frontend && cp .env.example .env.local    # DALEEL_TOKEN from POST /auth/redeem, ANSWER_API_KEY
npm install && npm run dev                       # http://localhost:3300
```

The MIT license covers the code, not the books (see the main README), and many of the
books are by modern authors still in copyright, so keep any deployment invite-only.
