# dalil API

FastAPI service over `dalil.s8_search`: search, cited answers and feedback. Open to every
visitor of the web app, with per-visitor and global daily caps.

## Visitors

There is no sign-in. The web app's server calls the API with `SITE_TOKEN` and the browser's
visitor id, a random value it keeps in a cookie; the API stores only its SHA-256 hash, and a
visitor exists from their first question, search or name. A visitor may give a name
(`POST /visitor`). Each question keeps the two-letter country it came from: the country the
host reports (Vercel, Cloudflare), or else the visitor's IP looked up in DB-IP's offline
country database (`GEOIP_DB`). No IP is stored or logged. `GET /admin/visitors` lists every
visitor with their name, questions, searches and countries, and every country with its
visitors and questions.

## Endpoints

| | |
|---|---|
| `GET /health` | liveness; `search` is `downloading`, `building`, `loading`, `ready` or `failed` |
| `GET /visitor`, `POST /visitor` | the visitor's name; `{name}` sets it, `""` takes it back |
| `POST /search` | `{question, n_results, categories?, query_planning, rerank, expand_neighbours}` |
| `POST /ask` | `{question, categories?}` -> the answer as NDJSON events, below |
| `POST /feedback` | `{query_id, rating, comment, corrected_answer, passages:[{parent_id, relevance, citation_ok}], missing_sources}` |
| `GET /queries/{id}` | a query with its answer and feedback |
| `GET /status`, `GET /admin/export` | admin: load state, today's use; queries + answers + feedback for evals |
| `GET /admin/visitors` | admin: who asks, and from which countries |

`/ask` streams one JSON object per line: `start` (the query id), one `search` per search the
answer model runs (its query, the new numbered passages and what the engine ran), `delta`
events with the answer's text (`reset` discards text streamed before a search), then `done`
(`model`, `cost_usd`, `latency_ms`) or `error`. The answer model searches with a `search`
tool, at most 3 times per question, and each search counts against the visitor's daily
searches. The question, every passage shown and the answer are stored, with the model, token
counts, reported cost and a hash of the system prompt.

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
| `ANSWER_MAX_TOKENS`, `ANSWER_PASSAGES`, `PASSAGE_MAX_CHARS` | 6000; passages per search, 8; characters per passage, 3000 |
| `ANSWER_PROMPT_PATH` | the answering prompt; default `prompts/answering_system_prompt.md` at the repository root, or its copy in the Docker image |
| `ADMIN_TOKEN` | bearer token for the admin routes |
| `SITE_TOKEN` | bearer token the web app's server sends with every visitor's call; the same value as the web app's `SITE_TOKEN` |
| `GEOIP_DB` | DB-IP country database (`.mmdb`) for questions whose host sends no country; the Docker image downloads it with `scripts/download_country_db.py`. [IP Geolocation by DB-IP](https://db-ip.com), CC BY 4.0 |
| `DATABASE_URL` | Postgres (Railway sets it); SQLite under the data dir when unset |
| `DALIL_DATA_DIR` | `/data` (the volume); stores live in `/data/db` |
| `DALIL_DENSE_COMPACT_DIMS` | unset: exact dense search; `1024`: the compact mode in the main README |
| `DALIL_DENSE_RESCORE` | rows the compact mode rescores exactly (default 1000) |
| `DALIL_PLANNER_MODEL`, `DALIL_LISTWISE_MODEL` | see the main README |
| `HF_DATA_REPO`, `HF_TOKEN`, `HF_DATA_PREFIX`, `HF_DATA_REVISION`, `HF_REPO_TYPE` | where missing stores are downloaded from on boot; `HF_REPO_TYPE` defaults to `dataset` |
| `USER_DAILY_ASKS`, `USER_DAILY_SEARCHES`, `GLOBAL_DAILY_ASKS`, `DAILY_SPEND_CAP_USD` | 30, 300, 500, 10; ollama.com reports no cost, so with the default endpoint `DAILY_SPEND_CAP_USD` never triggers |
| `CORS_ORIGINS` | comma-separated origins allowed to call the API from a browser |

Stores in `/data/db`: `faiss_metadata.sqlite`, `parents.sqlite`, `bm25.sqlite` and
`dense_faiss_ar.index`, downloaded from `HF_DATA_REPO` when missing. With
`DALIL_DENSE_COMPACT_DIMS=1024` the API also needs `dense_compact_1024_f16.index`. It is
not on the dataset: when it is missing, the API builds it on the host from
`dense_faiss_ar.index` on boot (`dalil.s7_index.compact`, one pass over the index;
`/health` shows `building`).

## Run and deploy

The API runs on the dalil package at the repository root:

```bash
PYTHONPATH=../.. .venv/bin/python -m pytest dalil_api
DALIL_DATA_DIR=../../data ADMIN_TOKEN=... SITE_TOKEN=... \
  PYTHONPATH=../.. .venv/bin/uvicorn dalil_api.main:app --port 8000
```

The image is built from the repository root, which holds the package and the answering
prompt:

```bash
docker build -f web/backend/Dockerfile -t dalil-api .    # from the repository root
```

On Railway the service builds from the repository root: leave Root Directory empty and set
the config file to `/web/backend/railway.json` (the Dockerfile and the health check), or set
`RAILWAY_DOCKERFILE_PATH=web/backend/Dockerfile`. Deploy from GitHub, or with `railway up`
from the repository root.
