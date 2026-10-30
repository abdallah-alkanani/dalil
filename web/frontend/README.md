# dalil frontend

Next.js app over the dalil API. Visitors sign in with an invite code; a question goes
to the API's streaming `/ask`, where the answer model searches (each search and its
passages appear as they run) and the answer streams as
[OpenUI Lang](https://openui.com/docs/openui-lang) into the components in
`lib/library.tsx`.

## Answers and their sources

- Components refer to passages by number (`Quote(2, "...")`, `[3]`). A `View` names who
  holds a view and carries their words as a `Quote`. Passage text comes from the search
  results (`lib/passages.tsx`), never from the model.
- `Quote` matches the model's words against passage n (`lib/arabic.ts`: diacritics,
  alef forms and punctuation ignored) and shows the passage's own text with the
  model's **bold** placed on it, marked "مطابق للمصدر", or a warning when the words
  are not in the passage.
- `npm run prompt` writes the component instructions for the answer model to
  `web/backend/dalil_api/answer_components.md` (`scripts/gen_prompt.tsx`). Run it after
  changing a component's name, props or description.

## Run

```bash
cp .env.example .env.local     # DALIL_API_URL
npm install                    # .npmrc skips install scripts (OpenUI's telemetry postinstall)
npm run dev                    # http://localhost:3300
```

The API must be running (`../backend/README.md`); an admin mints invite codes with
`POST /admin/invites`.

## Layout

| | |
|---|---|
| `app/api/ask` | the API's `/ask` stream, as the signed-in visitor |
| `app/api/auth/*` | redeem an invite (the token is kept in an httpOnly cookie), current user, sign out |
| `app/api/feedback`, `app/api/health` | the API's `/feedback` and `/health` |
| `lib/library.tsx` | the OpenUI library: Answer, Lead, Prose, Quote, Views, Steps, Table, Gap, Referral, FollowUpBlock |
| `components/` | `dalil.tsx` (sign-in and page), `cited_text.tsx`, `source_list.tsx`, `source_drawer.tsx` (the passage, with relevance feedback) |

## OpenUI

- `Table` and `FollowUpBlock` keep OpenUI's signatures but render `[n]` citations and an
  Arabic heading. `Tag`, `TextCallout`, `Steps`, `FollowUpItem`, `IconButton`, `Button`,
  `Skeleton` and `ThemeProvider` are used as they ship.
- In development `@openuidev/react-lang` mounts an inspector widget loaded from
  jsDelivr; `app/layout.tsx` keeps it off unless `NEXT_PUBLIC_OPENUI_DEVTOOLS=1`.
- Telemetry is off with `OPENUI_TELEMETRY_DISABLED=1` and `DO_NOT_TRACK=1`.
