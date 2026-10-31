# Status

- **Search** works over the database in `data/db`: 542 books, 349,633 passages, 927,744
  chunks. [books.md](books.md) lists the books.
- **Tests:** `pytest` runs 71 tests, offline. None of them need the data.
- **Compact dense mode** (`DALIL_DENSE_COMPACT_DIMS=1024`) is optional; exact search is the
  default. Its results against exact search are in [compact_check.md](compact_check.md).
- **The classifier and planner** run on `google/gemini-3.1-flash-lite` by default, checked in
  [model_check.md](model_check.md).
- **The web app** (`web/`): an API with search, streamed cited answers in which the answer
  model runs the search, feedback and daily caps; and a Next.js site, open to everyone, whose
  answers render as OpenUI components with quotes shown in their passages' own text. The admin
  sees each visitor's name (if given), questions and countries at `GET /admin/visitors`.
