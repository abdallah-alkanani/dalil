# Daleel frontend (OpenUI demo)

Next.js app over the Daleel API. Ask a question; the passages arrive first, then the
answer streams as [OpenUI Lang](https://openui.com/docs/openui-lang) and renders with
OpenUI's `Renderer` into Daleel's own components.

## How the answer stays faithful to the sources

- Generated components refer to passages **by number** (`Quote(2, "...")`, `[3]`,
  `View(..., [6, 7])`). Passage text comes from the search results
  (`lib/passages.tsx`), never from the model.
- `Quote` checks the model's words against passage n (`lib/arabic.ts`: diacritics,
  alef forms and punctuation ignored) and shows the **source's own span**, marked
  "مطابق للمصدر", or a warning when the words are not in the passage.
- The system prompt is generated from the component library
  (`scripts/gen-prompt.tsx` → `lib/system-prompt.generated.ts`, about 6.6k chars).

## Run

```bash
cp .env.example .env.local     # fill DALEEL_TOKEN (POST /auth/redeem) and ANSWER_API_KEY
npm install                    # .npmrc skips install scripts (OpenUI's telemetry postinstall)
npm run dev                    # http://localhost:3300; regenerates the prompt first
```

The API must be running (see `../backend/README.md`), for example:

```bash
cd ../backend && DALIL_DATA_DIR=../../data DALIL_DENSE_COMPACT_DIMS=1024 \
  ADMIN_TOKEN=... PYTHONPATH=vendor .venv/bin/uvicorn daleel_api.main:app --port 8000
```

## Layout

| | |
|---|---|
| `app/api/ask` | `/search` on the API, then the answer from any OpenAI-compatible endpoint, streamed as NDJSON |
| `app/api/feedback`, `app/api/health` | proxies to the API |
| `lib/library.tsx` | the OpenUI library: Answer, Lead, Prose, Quote, Views, Steps, Table, Gap, Referral, FollowUpBlock |
| `components/` | page shell, citation chips, source list, source drawer with expert relevance feedback |

## OpenUI notes

- OpenUI's `Table` and `FollowUpBlock` are replaced with same-signature versions:
  the first renders cells as plain text (raw `[n]`), the second hardcodes an English
  "Related Queries" header. `Tag`, `TextCallout`, `Steps`, `FollowUpItem`,
  `IconButton`, `Button`, `Skeleton` and `ThemeProvider` are used as they ship, and
  hold up in RTL.
- In development `@openuidev/react-lang` auto-mounts an inspector widget loaded from
  jsDelivr. `app/layout.tsx` turns it off; set `NEXT_PUBLIC_OPENUI_DEVTOOLS=1` to keep it.
- Telemetry: `OPENUI_TELEMETRY_DISABLED=1` and `DO_NOT_TRACK=1`.

## Answer model

DeepSeek 4.1 Flash on Ollama Cloud (`deepseek-v4.1-flash:1017`, the model Hermes runs),
with `reasoning_effort: none`. Measured on four questions (Ollama Cloud, 8 passages,
about 17k prompt tokens):

| model | model time | quotes verified |
|---|---|---|
| DeepSeek 4.1 Flash, reasoning none | 3.7 s | 4/4 |
| DeepSeek 4.1 Flash, reasoning low | 7.8 s | 6/6 |
| DeepSeek 4.1 Flash, default reasoning, 3000-token cap | returned nothing: reasoning used the whole cap | |
| Claude Sonnet 5.5 (OpenRouter, reasoning mandatory) | 20 s, about $0.08 | |

Switch with `ANSWER_BASE_URL`, `ANSWER_API_KEY`, `ANSWER_MODEL`, `ANSWER_REASONING_EFFORT`.

## Before production

- Move the answer step into the API (a streaming `/ask` that takes this prompt), so
  answers, cost and the spend caps are logged as `/ask` does today. Here the
  frontend calls the model directly and only `/search` and `/feedback` are recorded.
- Per-user login: the demo uses one server-side token for every visitor.
