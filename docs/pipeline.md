# The pipeline

How a book goes from a source file to a retrievable, cited passage. Paths below are relative to
the repository root. Data lives under `data/`; `DALIL_DATA_DIR` overrides it.

```
acquire -> prepare -> structure -> extract -> audit -> embed -> index -> search
download   to clean    an LLM picks  cut into   check the   GPU     FAISS +  hybrid
           text        the policy    parents    result      vectors SQLite   search
                       (judgement)   (deterministic)
```

| stage | package | what it does |
|---|---|---|
| acquire | `dalil.s1_acquire` | download Ketabonline packages and Shamela 4 books |
| prepare | `dalil.s2_prepare` | turn a source format into clean text and elements; per-book repairs |
| structure | `dalil.s3_structure` | an LLM picks each book's parent policy; finds semantic boundaries in long parents |
| extract | `dalil.s4_extract` | cuts elements, parents and retrieval views (no LLM) |
| audit | `dalil.s5_audit` | junk, token, coherence and parent/child checks |
| embed | `dalil.s6_embed` | Qwen3-Embedding-8B vectors, sharded and resumable |
| index | `dalil.s7_index` | appends to the FAISS and SQLite stores; reading order; BM25 |
| search | `dalil.s8_search` | BM25 and dense search, RRF, MMR, optional query planning and listwise rerank |

---

## Building the database

`dalil/build/run_build.sh` builds the stores from `data/` and `recipe/`:

```bash
P=<project dir> bash dalil/build/run_build.sh prepare    # Ketabonline handoff, per-book repairs
P=<project dir> bash dalil/build/run_build.sh extract    # core corpus and collection, validated
P=<project dir> bash dalil/build/run_build.sh select     # drop books, garbage views, author ids
P=<project dir> bash dalil/build/run_build.sh plan       # views that need a vector, one part per GPU
P=<project dir> bash dalil/build/run_build.sh embed      # one process per GPU
P=<project dir> bash dalil/build/run_build.sh assemble
P=<project dir> bash dalil/build/run_build.sh index      # empty stores, every corpus appended, finalize
```

- Output goes to `$BUILD` (default `data/build`). The stores go to `$BUILD/db`.
- `GPUS` lists the GPUs (default `0 1 2 3`). `WORKERS` is the process count of each parallel
  step (default: the CPU count).
- `ENGINE=vllm` with `EMBED_PY=<a Python with vLLM and dalil>` runs embedding with vLLM. The
  default engine is `sentence-transformers`.
- `REUSE_DB=<stores>` keeps the vectors of every view whose text is unchanged in those stores.
  Without it, every view is embedded.
- `select` keeps every book except those in `recipe/decisions/drop_books.txt`.
- `finalize` sets the author ids, runs reading order and the BM25 check, and writes
  `BUILD_REPORT.json` into the stores directory.

---

## Adding a book

```bash
# 1. get the file and convert it to UTF-8 text

# 2. register it: one line in recipe/books/collection_manifest.jsonl and one in
#    recipe/books/collection_profiles.jsonl (see "Registering a book")

# 3. choose the parent policy (sets parent_policy in the profile)
python -m dalil.s3_structure.parent_policy --raw <book.txt> --out /tmp/policy.json

# 4. extract into its own build directory
python -m dalil.s4_extract.collection --output-root data/build/structured/<batch> \
    --book-id <book_id> --boundaries <boundaries.jsonl>

# 5. check it
python -m dalil.s4_extract.validate --root data/build/structured/<batch>
python -m dalil.s5_audit.junk_scan data/build/structured/<batch>/retrieval_views.jsonl
python -m dalil.s5_audit.parent_child --parents <batch>/parents.jsonl --views <batch>/retrieval_views.jsonl

# 6. sample pages and check the cuts
python -m dalil.s5_audit.sample_pages <batch>/parents.jsonl --books 5 --pages-per-book 3 --out /tmp/s.json
python -m dalil.s5_audit.text_quality /tmp/s.json --out /tmp/a.json
python -m dalil.s5_audit.coherence <batch>/parents.jsonl --out /tmp/coh.json --window 3 --samples 20

# 7. embed on a GPU machine (resumable)
python -m dalil.s6_embed.embed_views --views <batch>/retrieval_views.jsonl --out <shards_dir>

# 8. append to the live stores, then add reading order
python -m dalil.s7_index.append --corpus <batch> --embeddings <shards_dir> --db data/db --dry-run
python -m dalil.s7_index.append --corpus <batch> --embeddings <shards_dir> --db data/db
python -m dalil.s7_index.navigation --db data/db --corpus <batch>
```

`append` and `navigation` take `--dry-run`. `embed_views` does not. `append` writes the BM25
rows itself. Rebuild the lexical index only after a change to normalisation:
`python -m dalil.s7_index.bm25 --db data/db`. To create an empty set of stores:
`python -m dalil.s7_index.schema --db data/db`.

---

## Stages

### s1_acquire: download

- `dalil.s1_acquire.ketabonline` downloads author catalogs, or named books with `--book`.
  Only each book's `files.data` package is requested.
- `dalil.s1_acquire.shamela4 --shamela-id <id> --out <dir>` fetches one book from the
  AuthenticIlm/Shamela4_Full_DB dataset. It is pinned to one revision and each file is checked
  against its SHA-256.

Other sources are not downloaded by this package. Their files are in `data/books/original/`.

### s2_prepare: normalise to text

`dalil.s2_prepare.ketabonline` turns Ketabonline packages into `elements.jsonl`. Its `--raw`,
`--catalog`, `--download-manifest`, `--author slug=id` and `--no-db-overlap` options prepare other
package folders. `dalil.s2_prepare.ketabonline_handoff` re-emits the collection's Ketabonline
handoff from `recipe/books/ketabonline_handoff_manifest.jsonl`. `dalil.arabic.clean` holds the
shared cleaning profile.

Repairs run on a copy in `data/build/books/`, never on the download:

| module | what it repairs |
|---|---|
| `dalil.s2_prepare.albani_restore` | Ketabonline books 1423 and 6006 |
| `dalil.s2_prepare.question_marks` | run before Ketabonline book 92317 is prepared |
| `dalil.s2_prepare.manzuma` | the 054 `.doc` (converted with LibreOffice by `run_build.sh`) |

### s3_structure: decide how to cut

- `dalil.s3_structure.parent_policy` reads several windows of a book's raw text and asks an
  LLM for one `parent_policy` with a `confidence`. Put the result in the book's profile. After
  repeated failure it returns `needs_review` with `parent_policy: null`.
- `dalil.s3_structure.semantic_boundaries` handles long parents. It asks about 8,000-word chunks
  and records where the subject changes, as verbatim anchors and word offsets. A chunk that fails
  is recorded with `partial: true` and `missing_ranges`. `--regap` asks only for those ranges.
  `--transport` is one of `openrouter`, `ollama_cloud`, `ollama_cli` or `opencode_cli`.

### s4_extract: cut

`dalil.s4_extract.collection` (the collection) and `dalil.s4_extract.core_corpus` (the core
corpus) write three layers:

| layer | what it is | used for |
|---|---|---|
| **elements** | the structural tree (book, chapter, heading, paragraph) | provenance and page numbers |
| **parents** | one semantic unit | what is shown and cited |
| **retrieval_views** | parent text packed into views with a context header | what gets embedded |

They also write `quran_ref_index.jsonl` (Quran citations), `isnad_edges.jsonl` (narrator pairs)
and `parser_coverage_report.json`.

Every view's `embedding_text` starts with a header:

```
الكتاب: تبصير الناسك بأحكام المناسك
التصنيف: fiqh
الموضع: تبصير الناسك > مسألة
نوع النص: paragraph
النص: <the text>
```

Ids come from the content (SHA-1 truncated to 12 hex digits), so books are parsed in parallel
(`--workers`, default up to 8) with the same ids.

The LLM is used before extraction (to choose the policy) and after it (`s5_audit`). The
extractor calls no LLM, and applies the recorded boundaries only where their anchor is still
found in the text.

Module map:

| module | role |
|---|---|
| `sources/` | one parser per source format: `openiti`, `shamela`, `shamela_footnotes`, `ketabonline`, `ibn_baz`, `saaid` |
| `sections.py` | one unit to parent, element and views (`emit_section`) |
| `boundaries.py` | applies recorded and in-text boundaries to long sections |
| `dialogue.py`, `qanda.py` | speaker-turn and question-and-answer splitting |
| `text.py` | markup, symbol-font and honorific expansion, source-marker stripping |
| `quran_refs.py` | surah names and Quran citation extraction |
| `records.py` | record builders, content-derived ids, packing, sanad and matn, isnad edges |
| `validate.py` | schema, ids and cross-references |

#### Shamela footnotes

Shamela stores a page's footnotes apart from its body. After a book's parents are written,
`sources/shamela_footnotes.py` places each footnote with the parent that holds the words around
its marker. It appends the footnote to that parent's `text_for_synthesis` under `الحواشي:` and
adds a `footnote` view (`{view_key}:fn:NNN`).

### s5_audit: check before embedding

| tool | question |
|---|---|
| `junk_scan` | degenerate text: repeated characters, Latin, digit floods |
| `token_stats` | token counts and length distribution |
| `sample_pages` | random pages from random books |
| `text_quality` | an LLM reads sampled pages and reports damage |
| `coherence` | does one argument span several parents |
| `parse_check` | an LLM checks parser boundaries on sampled books |
| `parent_child` | every view is inside its parent |

### s6_embed: vectors

`dalil.s6_embed.embed_views` uses Qwen3-Embedding-8B, 4096 dimensions, float16 and L2-normalised.

- Shards hold 2,000 views (`--shard-size`). A completed shard is skipped on restart.
- Views with fewer than 3 Arabic letters are skipped, except `ayah_exact` views
  (`--min-arabic`).
- `--engine vllm` needs its own Python environment (`EMBED_PY` in `run_build.sh`). The
  default is `sentence-transformers`.

To reuse vectors when a re-extraction changes only some views:

```bash
python -m dalil.s6_embed.reuse plan --corpus <build> --db data/db --out <build>/to_embed.jsonl
python -m dalil.s6_embed.embed_views --views <build>/to_embed.jsonl --out <new>      # GPU
python -m dalil.s6_embed.reuse assemble --corpus <build> --db data/db \
    --new <new> --texts <build>/to_embed.jsonl --out <build>/embeddings
```

`plan` lists one view per text that has no stored vector. `assemble` takes each vector from the
stores or the new shards by the hash of its `embedding_text`. The book's old views must be
removed with `dalil.s7_index.prune` before its rebuilt views are appended, because `append`
refuses a `view_id` it already has.

### s7_index: the stores

`dalil.s7_index.append` appends to the four stores in `data/db` and does not renumber existing
rows. It stops when a `view_id` already exists, when the FAISS row count differs from the
metadata, or when `bm25.sqlite` is out of step. It copies the stores first unless `--no-backup`
is given.

### catalog: books and hadith grades

- `dalil.catalog.books` writes the book catalog (`--out`). `dalil.build.select` reads
  `recipe/books/catalog.json` for the author ids.
- `python -m dalil.catalog.hadith_grades <dir>` maps al-Albani's graded Sunan onto the
  fawazahmed0/hadith-api `ara-*.min.json` files in `<dir>`, and writes
  `data/catalog/hadith/sunan_grades.jsonl`. The reviewed result is shipped as
  `recipe/decisions/sunan_grades.jsonl.gz`, which `dalil.s4_extract.core_corpus` reads.

### s8_search: retrieval

`python -m dalil.s8_search "question"`. Defaults: hybrid retrieval, RRF k=60 with weights 0.65
dense and 0.35 BM25, MMR lambda 0.7 over the top 80 parents, listwise rerank off, query planning
off, `--parent-max-chars 12000`. The flow is in [design.md](design.md#retrieval).

---

## `dalil.arabic`: the shared text rules

`dalil/arabic/normalize.py` is the search normalisation. The BM25 index and the query use the
same rules.

| rule | applied |
|---|---|
| strip tashkeel and Quranic marks | yes |
| NFKC, tatweel, Arabic-Indic digits to ASCII, lowercase | yes |
| `أ إ آ ٱ` to `ا` | yes |
| `ک` to `ك`, `ی` to `ي` | yes |
| `ى` to `ي` | no |
| `ؤ` to `و`, `ئ` to `ي` | no |
| `ة` to `ه` | no |

The character classes are written with `\uXXXX` escapes. `records.normalize_arabic` folds `ى`
and `ة`, but only the extractor uses it.

---

## Registering a book

Two JSONL lines: one in `recipe/books/collection_manifest.jsonl` and one in
`recipe/books/collection_profiles.jsonl`.

`collection_manifest.jsonl` gives the identity and the location of the bytes, with paths relative
to `data/`:

```json
{"book_id":"book:<category>:<author_id>:<hash12>","work_id":"work:...",
 "title_ar":"...","author_id":"...","category":"fiqh","subcategory":null,
 "source_corpus":"saaid","file_format":"saaid_txt",
 "source_path":"books/original/<author>/<book>/<n>.txt",
 "source_sha256":"..."}
```

`collection_profiles.jsonl` gives how to cut it:

```json
{"book_id":"...","parser_profile":"saaid_txt_parser",
 "parent_policy":"fatwa_masala_parent_v1","view_policy":"paragraph_pack_view_v1",
 "audit_status":"approved_for_parser_implementation"}
```

A new value of `source_corpus`, `parser_profile`, `parent_policy` or `parent_type` must also be
added to the schemas in `dalil/s4_extract/schemas/`. `dalil.s4_extract.validate` checks the
output against them and reports the rows that fail.

---

## Things that will bite you

- **Page numbers** exist only for Ketabonline parsers: 68,267 of the 81,335 Ketabonline parents
  carry `page_number`. No other parser profile sets one. `order_in_book` is set for every parent.
- **OpenITI** strips its `PageV` markers, so its parents have no page numbers.
- **Symbol-font characters.** Some Word-derived text encodes Arabic symbols as Private Use Area
  codepoints. `PUA_EXPANSIONS` in `dalil/s4_extract/text.py` expands them (for example U+F065 to
  صلى الله عليه وسلم). Check `junk_scan` when adding a source of that kind.
