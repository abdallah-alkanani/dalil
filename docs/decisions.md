# Decisions

Each decision names the code that implements it.

## Corpus shape

- **Three layers.** Elements hold the structure, parents are the units shown and cited, and
  retrieval views are the packs that get embedded. Each view has one parent.
- **Parents are cut only at boundaries in the text.** The boundaries are headings, speaker
  turns, inline headings, section markers and recorded semantic boundaries
  (`dalil/s4_extract/boundaries.py`, `sections.py`). A parent with no boundary stays whole and
  gets `needs_semantic_split`.
- **Size constants.** Parents target 15,000 words and are considered for cuts above 22,500
  (`PARENT_TARGET_WORDS`, `PARENT_SPLIT_ABOVE` in `boundaries.py`).
- **Speaker turns.** A text is split at speaker labels only if it has at least three turns and
  averages at most 2,000 words per turn. Splitting starts above 4,000 words; an exchange has
  a minimum of 250 words and a target of 1,200 (`dalil/s4_extract/dialogue.py`).
- **Pack sizes.** Sections are packed into paragraphs of at most 256 words by default
  (`emit_section` in `sections.py`). Plain-text core profiles pack at 420 words
  (`parse_text_profile`), tafsir commentary views at 360 words and tafsir segments at 1,400
  (`core_corpus.py`). Hadith full, sanad and matn views are whole up to 800
  words, then split into 360-word clauses. Fatwa answers are at most 800 words.
- **Inline headings.** A `[فصل ...]` heading inside a line counts only when the bracket opens
  with a structural keyword, so `[البقرة: 255]` is not a heading
  (`dalil/s4_extract/detectors/inline_headings.py`).
- **Embedding text.** Every view is embedded with a header: `الكتاب` (book title),
  `التصنيف` (category), `الموضع` (section path), `نوع النص` (text role), then `النص`
  (`dalil/s4_extract/records.py`).
- **Shamela footnotes** are added beside the body. Each becomes a `footnote` view
  (`{view_key}:fn:NNN`), and a `الحواشي:` block is appended to the parent's
  `text_for_synthesis`. A footnote is placed by the words around its marker, and the
  `[عند: «…»]` label quotes the six words before the marker
  (`dalil/s4_extract/sources/shamela_footnotes.py`).
- **Torn definite articles.** `ا لجواب` is joined back to `الجواب` only when the alif
  follows no Arabic letter and the next word does not already start with lam
  (`dalil/s4_extract/records.py`).
- **Empty parents.** An empty parent is dropped only when its title repeats the next
  parent's title (`drop_empty_duplicates` in `dalil/s4_extract/collection.py`).
- **Short verses.** A view with fewer than 3 Arabic letters is not embedded, except
  `ayah_exact` views (`too_little_arabic` in `dalil/s6_embed/embed_views.py`).

## Search

- **Shared normalisation.** `dalil/arabic/normalize.py` is used for both the index and the
  query. It strips tashkeel, folds `أ إ آ ٱ` to `ا`, and maps `ک` and `ی` to Arabic letters.
  It does not fold `ى`, `ؤ`, `ئ` or `ة`, because folding would merge different words (the
  module names `على`/`علي` and `رحمة`/`رحمه`). `records.normalize_arabic` folds them, and
  search does not use it.
- **Hybrid fusion.** BM25 and dense rankings are fused per parent with weighted reciprocal-rank
  fusion: k=60, dense weight 0.65, BM25 weight 0.35. A parent is scored by its best child
  (`dalil/s8_search/ranking.py`, `engine.py`).
- **After fusion.** A source-category prior, then MMR with lambda 0.7 over the top 80 parents.
  An optional listwise rerank covers the top 20 (`LISTWISE_DEPTH`). Listwise rerank is off by
  default.
- **Query planning** is off by default. When it is on, the exact question is always variant
  `Q0`, a critic rejects variants that drift to another question, and a planner failure falls
  back to the question (`dalil/s8_search/planning.py`).
- **Dense search** is an exact `IndexFlatIP`. The index is scored in blocks of 32,768 vectors
  (512 MB at 4,096 float32 each) with `faiss.knn`, and the block results are merged with
  `faiss.ResultHeap`. Blocks are read with O_DIRECT when the index does not fit in the page
  cache (`dalil/s8_search/dense.py`).
- **BM25 freshness** is checked against FTS5's `docsize` table, not by counting rows
  (`dalil/s7_index/bm25.py`). `append` refuses to run when `bm25.sqlite` is out of step.
- **Compact dense mode** is optional, for hosts with little RAM or a slow disk. The first 1,024
  dimensions of every vector (Qwen3-Embedding is Matryoshka-trained), as float16, are scanned in
  memory and the top 1,000 rescored exactly (`CompactDenseIndex`, `DALIL_DENSE_COMPACT_DIMS`).
  Exact search stays the default because the compact mode can differ: on the 724 dataset
  questions it returned the same top 50 for 723 (`docs/compact_check.md`).
- **Classifier and planner models** come from `DALIL_LISTWISE_MODEL` and `DALIL_PLANNER_MODEL`,
  default `google/gemini-3.1-flash-lite`, the model with the most correct classifications (tied)
  and the fastest valid planner on the 197 people-asked questions (`docs/model_check.md`).
  Reasoning models fail both: they spend the 20 and 900 output tokens before answering.
- **Search reports what ran.** `query()` returns `status` with the retrieval mode, planner,
  rerank and source classifier outcomes. A hybrid search whose question cannot be embedded
  falls back to BM25 and records why in `status.retrieval`.

## Web

- **One answer path.** The website asks the API's `/ask`, which streams the answer and stores
  the question, the passages, the answer, the model, its token counts and reported cost, and a
  hash of the system prompt. The per-user and global daily caps apply there
  (`web/backend/dalil_api/main.py`).
- **The answer model runs the search.** `prompts/answering_system_prompt.md` is written for a
  model that searches first and searches again when a search misses, so on the website the
  model gets a `search` tool, at most 3 calls per question; passage numbers continue across
  searches, and each search counts against the user's daily searches
  (`web/backend/dalil_api/answer.py`).
- **The system prompt** is the repository's answering prompt followed by the OpenUI component
  instructions, generated from the frontend's component library (`npm run prompt` writes
  `web/backend/dalil_api/answer_components.md`).
- **The answer model** is any OpenAI-compatible endpoint; the default is DeepSeek 4.1 Flash on
  Ollama Cloud with `reasoning_effort: none`, so the token budget goes to the answer. It returns
  tool calls on that endpoint.
- **Quotes are checked.** Components refer to passages by number. `Quote` matches the model's
  words against the passage, ignoring diacritics, alef forms and punctuation, and shows the
  passage's own text, or a warning when the words are not in it (`web/frontend/lib/arabic.ts`).
- **Invite-only.** Each visitor signs in with an invite code and gets their own API token, kept
  in an httpOnly cookie, so the per-user caps apply to every visitor.

## Pipeline

- **The database is rebuilt, not patched.** `run_build.sh` removes `db/` and appends every
  corpus into empty stores. `REUSE_DB` only supplies vectors for text that has not changed
  (`dalil/build/run_build.sh`, `dalil/s6_embed/reuse.py`).
- **Vectors are matched by hash.** A vector is reused when the 16-byte BLAKE2b hash of
  `embedding_text` matches (`dalil/s6_embed/reuse.py`).
- **Embedding is sharded.** A shard holds 2,000 views and is saved as it completes; completed
  shards are skipped on restart (`dalil/s6_embed/embed_views.py`).
- **Embedding engine.** `--engine vllm` runs the model in bf16 with `runner="pooling"`. It
  needs its own Python environment, given to `run_build.sh` as `EMBED_PY`.
- **An LLM judges structure, and code cuts.** Nothing in `dalil/s4_extract` imports
  `dalil.llm`. The parent policy and the boundary decisions are written to files, which the
  extractor reads.
- **Semantic boundaries** are asked in 8,000-word chunks (`--chunk-words`). A failed chunk is
  recorded as `partial: true` with `missing_ranges`, and a `--regap` run asks only for those
  ranges (`dalil/s3_structure/semantic_boundaries.py`).
- **Anchors** are matched on normalised tokens. A cut is applied only when its anchor is found
  in the text (`normalize_for_search`, `split_at_anchors`).
- **`max_tokens` on LLM calls.** The OpenRouter transport sends `max_tokens` only when a
  caller passes it (`dalil/llm/openrouter.py`). The extraction-side stages (`s3_structure`,
  `s5_audit`) pass none. The query planner and critic pass 900 and 500
  (`dalil/s8_search/engine.py`).
- **opencode CLI runs with every tool denied.** Each call gets a private configuration whose
  only rule denies all tools, and runs with `--standalone` (`dalil/llm/opencode_cli.py`).
