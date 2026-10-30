# Daleel API

FastAPI service over `dalil.s8_search`: search, cited answers, and expert feedback.
Invite-only, with per-user and global daily caps on OpenRouter spend.

## Endpoints

| | |
|---|---|
| `GET /health` | liveness; `search` is `downloading`, `loading`, `ready` or `failed` |
| `POST /admin/invites` | admin: mint an invite code `{label, role, max_uses}` |
| `POST /auth/redeem` | `{code, name, email?}` -> bearer token |
| `POST /search` | `{question, n_results, categories?, query_planning, rerank, expand_neighbours}` |
| `POST /ask` | `{question, categories?}` -> answer with numbered, cited passages |
| `POST /feedback` | `{query_id, rating, comment, corrected_answer, passages:[{parent_id, relevance, citation_ok}], missing_sources}` |
| `GET /queries/{id}` | a query with its answer and feedback |
| `GET /status`, `GET /admin/export` | admin: load state, today's spend; queries + answers + feedback for evals |

Every search and answer reports what actually ran in `status`: `retrieval`
(`hybrid`, or `bm25_fallback(<reason>)` when embeddings fail), `planner`, `rerank`.

## Variables

| variable | |
|---|---|
| `OPENROUTER_API_KEY` | embeddings, planner, answers |
| `ADMIN_TOKEN` | bearer token for the admin routes |
| `DATABASE_URL` | Postgres (Railway sets it); SQLite under the data dir when unset |
| `DALIL_DATA_DIR` | `/data` (the volume); stores live in `/data/db` |
| `DALIL_DENSE_COMPACT_DIMS` | `1024`: 1.9 GB first-pass index in RAM, exact rescore from disk |
| `DALIL_DENSE_RESCORE` | rows rescored exactly (default 1000) |
| `HF_DATA_REPO`, `HF_TOKEN`, `HF_DATA_PREFIX`, `HF_DATA_REVISION` | where missing stores are downloaded from on boot |
| `ANSWER_MODEL` | default `anthropic/claude-sonnet-5.5` |
| `DALIL_PLANNER_MODEL`, `DALIL_LISTWISE_MODEL` | default `anthropic/claude-haiku-5.5` |
| `USER_DAILY_ASKS`, `USER_DAILY_SEARCHES`, `GLOBAL_DAILY_ASKS`, `DAILY_SPEND_CAP_USD` | 30, 300, 500, 10 |
| `CORS_ORIGINS` | comma-separated origins of the feedback site |

Stores needed in `/data/db`: `faiss_metadata.sqlite`, `parents.sqlite`, `bm25.sqlite`,
`dense_faiss_ar.index`, and `dense_compact_1024_f16.index` (built with
`python -m dalil.s7_index.compact --dims 1024` in the dalil repo).

## Run and deploy

```bash
./scripts/sync_dalil.sh                      # copy the dalil package into vendor/
PYTHONPATH=vendor .venv/bin/python -m pytest tests
DALIL_DATA_DIR=../../data DALIL_DENSE_COMPACT_DIMS=1024 PYTHONPATH=vendor \
  .venv/bin/uvicorn daleel_api.main:app --port 8000
./scripts/sync_dalil.sh && railway up        # Dockerfile build on Railway
```
