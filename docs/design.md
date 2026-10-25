# Design

How dalil turns the books in `data/books/` into searchable units, and how it searches them.
The books are listed in [books.md](books.md).

## Pipeline

Each book has a parser profile (a source reader, a parent policy and a view policy). The
profile turns the book into three layers:

- **Element**: a structural unit from the source (a surah, an ayah, a hadith, a heading block,
  a paragraph, a fatwa). Elements form a tree through `parent_element_id` and `ancestor_ids`,
  with one `book` element per book.
- **Parent**: the unit that is returned and cited, for example a full hadith, one ayah's
  commentary, a fatwa or a dialogue exchange.
- **Retrieval view**: a searchable piece of a parent. Every view has one `parent_id`. Views
  are embedded and indexed; parents are not. A search scores parents by their best view.

```text
recipe/books/*.jsonl (manifest + profile per book)
  dalil.s3_structure.parent_policy        chooses the parent policy (LLM, written to the profile)
  dalil.s3_structure.semantic_boundaries  records cut points for long parents (LLM)
        v
dalil.s4_extract.core_corpus / collection   elements -> parents -> retrieval views (no LLM)
        v
dalil.s4_extract.validate, dalil.s5_audit.*  schema, cross-reference and quality checks
        v
dalil.s6_embed.embed_views                  Qwen3-Embedding-8B over embedding_text
        v
dalil.s7_index.append, .navigation, .bm25   FAISS + SQLite stores in data/db
        v
dalil.s8_search                             BM25 + dense search, fused by parent
```

## Stores

The build writes `elements.jsonl`, `parents.jsonl`, `retrieval_views.jsonl`,
`quran_ref_index.jsonl`, `isnad_edges.jsonl` and `parser_coverage_report.json`. The stores in
`data/db`:

| File | Contents |
|---|---|
| `dense_faiss_ar.index` | `IndexFlatIP`, one row per embedded view |
| `faiss_metadata.sqlite` | table `child_metadata`: one row per vector (`faiss_row`, `view_id`, `parent_id`, `book_id`, `category`, `view_type`, `refs_json`, `section_path_json`, `display_text`, `embedding_text`, `qwen_tokens`) |
| `parents.sqlite` | table `parents`: full parent text (`text_original`, `text_for_synthesis`), citation, refs, section path, `order_in_book`, and `page_number`, `page_end`, `part_name`, `page_index` |
| `bm25.sqlite` | FTS5 table `child_fts` over normalised child text, `rowid = faiss_row + 1` |

`dalil.s7_index.schema` creates empty stores. `dalil.s7_index.append` adds a corpus without
renumbering existing rows.

## Schemas

The schemas are in `dalil/s4_extract/schemas/`. Their `schema_version` values are
`dalil.element.v1`, `dalil.parent.v1` and `dalil.retrieval_view.v1`.
Elements, parents and views each carry `parser_profile`, `parent_policy`, `view_policy` and
`quality_flags`.

Token counts are stored as a list of `{tokenizer_name, token_count}`. The extractor writes
`whitespace_ar_v1` counts. `dalil.s7_index.append` adds Qwen tokenizer counts (`qwen_tokens`).

Texts: `text_original` keeps the vocalised source for display and citation. The normalised
form is used only for matching.

## Id rules

Ids come from the source, so a rebuild of the same input gives the same ids.

| Record | Pattern |
|---|---|
| Book element | `elem:{book_id}:book` |
| Quran | `elem:quran:surah:{s}`, `elem:quran:{s}:{a}`, `parent:quran:{s}:{a}`, `view:quran:{s}:{a}:ayah_exact` |
| Hadith clause view | `…:clause:{NNN}` |
| Other parents | `parent:{book_id}:{h}`, where `h` is the first 12 hex digits of the SHA-1 of the unit id and its order |
| Split piece | `…:seg:{NNN}` |
| Shamela footnote view | `{view_key}:fn:{NNN}` |
| Quran reference | `qref:{source_id}:{surah}:{ayah_start}-{ayah_end}` |
| Isnad edge | `edge:{parent_id}:{NNN}` |

Book ids are source-prefixed for the core corpus (`hadith_api::…`, `tafsir_arabic::…`,
`openiti:…`, `shamela:…`, `saaid:…`, `quran_uthmani_ar`). The collection uses
`book:{category}:{author_id}:{hash12}`.

## Parser profiles and policies

A profile names a source reader and the policies for one book. The profiles are in
`recipe/books/core_profiles.jsonl` and `recipe/books/collection_profiles.jsonl`. The parser
names in use:

| Parser | Source |
|---|---|
| `quran_json_parser` | Quran JSON, one surah list |
| `hadith_json_parser`, `hadith_flat_json_parser` | hadith JSON, with chapters, or a flat list |
| `musnad_csv_parser` | Musnad CSV |
| `tafsir_json_parser` | one JSON file per surah |
| `shamela_txt_parser`, `shamela_json_records_parser` | Shamela text, or a folder of page records |
| `shamela4_json_parser` | Shamela 4 `pages.jsonl` and `toc.jsonl` |
| `openiti_txt_parser` | OpenITI mARkdown |
| `saaid_txt_parser` | plain text from `.doc` |
| `ketabonline_json_parser` | Ketabonline page packages |
| `ibn_baz_fatwa_jsonl_parser` | fatwa JSONL, one record per fatwa |

The parent policy is chosen per book. `dalil.s3_structure.parent_policy` asks an LLM to choose
one of seven generic policies:

| Policy | Meaning in the prompt |
|---|---|
| `heading_section_parent_v1` | short heading lines (باب, فصل, كتاب) delimit sections |
| `fatwa_masala_parent_v1` | each ruling opens with a marker such as مسألة |
| `qanda_pair_parent_v1` | explicit question and answer pairs |
| `numbered_item_parent_v1` | numbered items are the units |
| `paragraph_group_parent_v1` | continuous prose with no reliable marker |
| `monograph_prose_parent_v1` | one continuous argument with no internal divisions |
| `risala_collection_parent_v1` | a collection of separate short treatises |

If the model fails repeatedly, the result is `needs_review` with `parent_policy: null`. A
profile with `audit_status` `needs_review` or `hold` still extracts, and every record gets
`needs_review` or `book_level_hold`.

Structure detectors are named per book in the profiles (`recommended_detectors`). The
detector code is in `dalil/s4_extract/detectors/` (inline headings, section markers, speaker
turns) and in the source parsers under `dalil/s4_extract/sources/`.

## View types in the database

Counts from `faiss_metadata.sqlite`, 2026-10-25:

| view_type | views | view_type | views |
|---|---:|---|---:|
| paragraph_pack | 270,894 | tafsir_ayah_commentary | 16,784 |
| heading_title | 161,879 | commentary_paragraph | 7,581 |
| hadith_matn | 145,277 | ayah_exact | 6,236 |
| hadith_sanad | 74,128 | numbered_item_view | 3,447 |
| hadith_full | 62,572 | qanda_answer | 2,386 |
| footnote | 49,710 | poem_verse | 943 |
| tafsir_paragraph | 28,455 | qanda_question | 650 |
| hadith_commentary | 22,294 | hadith_clause | 254 |
| fatwa_title | 19,680 | fatwa_answer_clause | 50 |
| fatwa_answer | 19,610 | | |
| fatwa_question | 18,989 | | |
| qanda_pair | 15,925 | | |

## Size rules

- **Parents.** The target is 15,000 words (`PARENT_TARGET_WORDS`). A long parent is cut only at
  boundaries found in the text. The boundary pass asks about parents longer than 4,000 words by
  default (`--min-words`). The repair pass re-asks pieces above 22,500 words (`--regap-above`).
- **Speaker turns.** A text is split at speaker labels above 4,000 words, if it averages at most
  2,000 words per turn. Exchanges have a minimum of 250 words and a target of 1,200
  (`dalil/s4_extract/dialogue.py`).
- **Views.** Paragraph packs are at most 256 words by default (`emit_section`). Plain-text core
  profiles pack at 420 words, tafsir commentary views at 360. Hadith full, sanad and matn views
  are whole up to 800 words, then split into 360-word clauses. Fatwa answers are at most 800 words.
- **Semantic boundary questions** are asked in chunks of 8,000 words (`--chunk-words`). A cut is
  applied only where its recorded anchor is found in the text (`split_at_anchors`).

## Genre notes

- **Quran.** The parent is the ayah and the surah is an ancestor element. The one view is
  `ayah_exact`, with its header. Wider context comes from reading-order neighbours at query
  time (`--expand-neighbours`).
- **Hadith.** The parent is the full hadith (chain and text). The views are `hadith_full`,
  `hadith_sanad` and `hadith_matn`, plus `hadith_clause` for long texts. A parent returns whole
  when only its sanad or matn view matched. Sanad and matn splits carry the flags
  `sanad_start_anchored`, `sanad_matn_split_quote_boundary`, `sanad_matn_split_low_confidence`
  and `no_sanad_detected`.
- **Tafsir.** The parent is the commentary on one ayah (`tafsir_ayah_commentary`). Its views are
  `tafsir_ayah_commentary` and `tafsir_paragraph`.
- **Fatwas.** The views are `fatwa_title`, `fatwa_question` and `fatwa_answer`. A record with no
  extracted answer gets the flag `answerless_record`.
- **Lessons.** Speaker-turn splits are `dialogue_exchange` parents.
- **Isnad edges.** Adjacent narrator pairs are recorded with `extraction_confidence` 0.55 and the
  flag `heuristic_isnad_edge`.
- **Quran references.** Each reference has a `match_method`: `source_structural_ref` (the source's
  own structure) or `explicit_ref` (a citation in the text), which has confidence 0.75.

## Embedding

- **Model.** `Qwen/Qwen3-Embedding-8B`, 4096 dimensions, pinned to revision
  `4e423935c619ae4df87b646a3ce949610c66241c` (`dalil/s6_embed/embed_views.py`). Vectors are
  L2-normalised and stored as float16 in shards of 2,000 views.
- **Documents** are embedded without an instruction. **Queries** are embedded with the Arabic
  retrieval instruction `EMBED_INSTRUCTION_AR` (`dalil/s8_search/api.py`) through OpenRouter's
  `qwen/qwen3-embedding-8b`.
- **Input.** `embedding_text` carries the header described in [decisions.md](decisions.md).
- **Engines.** `sentence-transformers` (the default) or `vllm`.

## Retrieval

`python -m dalil.s8_search`, also `dalil-search`:

1. Optional query planning. The exact question is always variant `Q0`.
2. Dense search over all vectors. `k` is widened until enough distinct parents survive the filters.
3. BM25 over normalised child text (FTS5).
4. Each list is grouped by `parent_id` (score = best child). The two lists are fused with
   reciprocal-rank fusion: k=60, dense 0.65, BM25 0.35.
5. A source-category prior, from keywords or an LLM classifier.
6. MMR with lambda 0.7 over the top 80 parents.
7. Optional listwise LLM rerank of the top 20.
8. Parents are fetched. Long parents are windowed to `--parent-max-chars` (12,000) around the
   match and snapped to sentence boundaries. Reading-order neighbours are attached with
   `--expand-neighbours`.

Filters (category, book, author, source, view and text role, parent type, policy, quality flag,
Quran surah and ayah, hadith collection and number, section text) apply as SQL on the child and
parent stores.

Dense search (`dalil/s8_search/dense.py`) streams the flat index from disk in blocks of 32,768
vectors and scores each with `faiss.knn`. The blocks are merged with `faiss.ResultHeap`.
