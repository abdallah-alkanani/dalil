# dalil frontend

Next.js app over the dalil API, open to everyone. Each browser gets a random visitor id in
an httpOnly cookie, and a visitor may give a name. A question goes to the API's streaming
`/ask`, where the answer model searches (a line under the question says what it is doing, and
the passages appear as they arrive) and the answer streams as
[OpenUI Lang](https://openui.com/docs/openui-lang) into the components in
`lib/library.tsx`.

## Answers and their sources

- Components refer to passages by number (`Quote(2, "...")`, `[3]`). A `View` names who
  holds a view and carries their words as a `Quote`. Passage text comes from the search
  results (`lib/passages.tsx`), never from the model.
- `Quote` matches the model's words against passage n (`lib/arabic.ts`: diacritics,
  alef forms and punctuation ignored) and shows the passage's own text with the
  model's **bold** placed on it.
- `npm run prompt` writes the component instructions for the answer model to
  `web/backend/dalil_api/answer_components.md` (`scripts/gen_prompt.tsx`). Run it after
  changing a component's name, props or description.

## Run

```bash
cp .env.example .env.local     # DALIL_API_URL, SITE_TOKEN
npm install                    # .npmrc skips install scripts (OpenUI's telemetry postinstall)
npm run dev                    # http://localhost:3300
```

The API must be running (`../backend/README.md`) with the same `SITE_TOKEN`. The site sends
the API the country its host reports (`x-vercel-ip-country`, `cf-ipcountry`), or else the
visitor's IP for the API's offline lookup.

## Layout

| | |
|---|---|
| `app/api/ask` | the API's `/ask` stream, for this visitor |
| `app/api/visitor` | the visitor's name |
| `app/api/feedback`, `app/api/health` | the API's `/feedback` and `/health` |
| `lib/library.tsx` | the OpenUI library: Answer, Lead, Prose, Quote, Views, Steps, Table, Gap, Referral, FollowUpBlock |
| `components/` | `dalil.tsx` (the page), `cited_text.tsx`, `source_list.tsx`, `source_drawer.tsx` (the passage, with relevance feedback) |

## OpenUI

- `Table` and `FollowUpBlock` keep OpenUI's signatures but render `[n]` citations and an
  Arabic heading. `Tag`, `TextCallout`, `Steps`, `FollowUpItem`, `IconButton`, `Button`,
  `Skeleton` and `ThemeProvider` are used as they ship.
- In development `@openuidev/react-lang` mounts an inspector widget loaded from
  jsDelivr; `app/layout.tsx` keeps it off unless `NEXT_PUBLIC_OPENUI_DEVTOOLS=1`.
- Telemetry is off with `OPENUI_TELEMETRY_DISABLED=1` and `DO_NOT_TRACK=1`.
