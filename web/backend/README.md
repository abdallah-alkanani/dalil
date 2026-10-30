# dalil API

FastAPI service over `dalil.s8_search`: search, cited answers and expert feedback.
Invite-only, with per-user and global daily caps.

## Endpoints

| | |
|---|---|
| `GET /health` | liveness; `search` is `downloading`, `building`, `loading`, `ready` or `failed` |
| `POST /admin/invites` | admin: mint an invite code `{label, role, max_uses}` |
| `POST /auth/redeem` | `{code, name, email?}` -> bearer token |
| `GET /me` | the token's user |
| `POST /search` | `{question, n_results, categories?, query_planning, rerank, expand_neighbours}` |
| `POST /ask` | `{question, categories?}` -> the answer as NDJSON events, below |
| `POST /feedback` | `{query_id, rating, comment, corrected_answer, passages:[{parent_id, relevance, citation_ok}], missing_sources}` |
| `GET /queries/{id}` | a query with its answer and feedback |
| `GET /status`, `GET /admin/export` | admin: load state, today's use; queries + answers + feedback for evals |

`/ask` streams one JSON object per line: `retrieval` (`query_id`, `status`, the numbered
`passages`), then `delta` events with the answer's text, then `done` (`model`,
`cost_usd`, `latency_ms`) or `error`. The question, passages and answer are stored,
with the model, token counts, reported cost and a hash of the system prompt.

The system prompt is `prompts/answering_system_prompt.md` followed by
`dalil_api/answer_components.md`, the OpenUI component instructions that
`web/frontend` generates from its component library (`npm run prompt`). The model
writes OpenUI Lang.

Every search and answer reports what actually ran in `status`: `retrieval`
(`hybrid`, or `bm25_fallback(<reason>)` when the query cannot be embedded), `planner`,
`rerank`, `source_classifier`.

## Variables

| variable | |
|---|---|
| `OPENROUTER_API_KEY` | query embeddings, the source classifier and the planner |
| `ANSWER_BASE_URL` | OpenAI-compatible endpoint of the answer model; default `https://ollama.com/v1` |
| `ANSWER_API_KEY` | its key; `OLLAMA_API_KEY` is used when unset |
| `ANSWER_MODEL` | default `deepseek-v4.1-flash:1017` |
| `ANSWER_REASONING_EFFORT` | sent as `reasoning_effort`; default `none`, empty sends nothing |
| `ANSWER_MAX_TOKENS`, `ANSWER_PASSAGES`, `PASSAGE_MAX_CHARS` | 6000, 8, 3000 |
| `ADMIN_TOKEN` | bearer token for the admin routes |
| `DATABASE_URL` | Postgres (Railway sets it); SQLite under the data dir when unset |
| `DALIL_DATA_DIR` | `/data` (the volume); stores live in `/data/db` |
| `DALIL_DENSE_COMPACT_DIMS` | unset: exact dense search; `1024`: the compact mode in the main README |
| `DALIL_DENSE_RESCORE` | rows the compact mode rescores exactly (default 1000) |
| `DALIL_PLANNER_MODEL`, `DALIL_LISTWISE_MODEL` | see the main README |
| `HF_DATA_REPO`, `HF_TOKEN`, `HF_DATA_PREFIX`, `HF_DATA_REVISION` | where missing stores are downloaded from on boot |
| `USER_DAILY_ASKS`, `USER_DAILY_SEARCHES`, `GLOBAL_DAILY_ASKS`, `DAILY_SPEND_CAP_USD` | 30, 300, 500, 10 |
| `CORS_ORIGINS` | comma-separated origins allowed to call the API from a browser |

`DAILY_SPEND_CAP_USD` counts the cost the answer endpoint reports (OpenRouter does);
the per-user and global question caps apply whatever the endpoint.

Stores in `/data/db`: `faiss_metadata.sqlite`, `parents.sqlite`, `bm25.sqlite` and
`dense_faiss_ar.index`, downloaded from `HF_DATA_REPO` when missing. With
`DALIL_DENSE_COMPACT_DIMS=1024` the API also needs `dense_compact_1024_f16.index`. It is
not on the dataset: when it is missing, the API builds it on the host from
`dense_faiss_ar.index` on boot (`dalil.s7_index.compact`, one pass over the index;
`/health` shows `building`).

## Run and deploy

```bash
./scripts/sync_dalil.sh                      # copy the dalil package and the answering prompt into vendor/
PYTHONPATH=vendor .venv/bin/python -m pytest tests
DALIL_DATA_DIR=../../data PYTHONPATH=vendor .venv/bin/uvicorn dalil_api.main:app --port 8000
./scripts/sync_dalil.sh && railway up        # Dockerfile build on Railway
```
