# dalil

**dalil** (دليل, "evidence") is an Arabic retrieval system (the search half of a
RAG) over 542 classical Islamic books: the Quran, the hadith collections, tafsir,
fiqh, aqeedah and seerah. Ask a question and it returns the passages that address
it, each with its scholar, book and place, ready to hand to a language model as
cited evidence.

Everything it searches is ready to download from
[Hugging Face](https://huggingface.co/datasets/Alkanani/dalil): the books in their
original and cleaned form, the search databases, and 927,744 precomputed
Qwen3-Embedding-8B vectors in an exact FAISS index. Only the question is embedded
when you search, through OpenRouter or any OpenAI-compatible API; keyword search
alone needs no API key.

```
$ python -m dalil.s8_search "ما حكم الربا؟"

[1] العقيدة | الضياء اللامع من الخطب الجوامع، [الخطبة الرابعة في حكم التحيل على الربا]
    Section: الضياء اللامع من الخطب الجوامع > [الخطبة الرابعة في حكم التحيل على الربا]
    Parent: parent:book:aqeedah:ibn_uthaymeen_1347_1421:456db89c30dc:fd5694fffcd2 | View: paragraph_pack
الخطبة الرابعة في حكم التحيل على الربا الحمد لله الذي أباح لنا من المكاسب أحلها ...
```

The current database holds 542 books cut into 349,633 passages, searched through
927,744 embedded chunks.

## The books

- **ابن تيمية** (119 books): مجموع الفتاوى، جامع المسائل - المجموعة الرابعة، منهاج السنة النبوية، مجموعة الرسائل والمسائل - ط وهبة، and 115 more
- **ابن عثيمين** (108 books): فتاوى نور على الدرب للعثيمين، مجموع فتاوى ورسائل العثيمين، لقاء الباب المفتوح، اللقاء الشهري، and 104 more
- **الألباني** (107 books): صحيح وضعيف الجامع الصغير، تفريغ «سلسلة الهدى والنور» للشيخ الألباني - الإصدار 4، صحيح الجامع الصغير وزيادته، التعليقات الحسان على صحيح ابن حبان، and 103 more
- **ابن القيم** (61 books): إعلام الموقعين عن رب العالمين ت مشهور، التبيان في أيمان القرآن ط عالم الفوائد، الطرق الحكمية في السياسة الشرعية ط عالم الفوائد، التفسير القيم لابن القيم، and 57 more
- **ابن باز** (60 books): فتاوى نور على الدرب لابن باز، فتاوى الدروس لابن باز، مجموع فتاوى ومقالات متنوعة - ابن باز، فتاوى نور على الدرب لابن باز - بعناية الشويعر، and 56 more
- **وليد السعيدان** (50 books): تلخيص مسائل الحج على القول الراجح، تمام المنة ببعض ما اتفق عليه أهل السنة، إتحاف أهل الألباب بمعرفة التوحيد والعقيدة في سؤال وجواب، الشرح المبدي جمال منظومة الشيخ ابن سعدي، and 46 more
- **The Quran and the hadith collections:** القرآن الكريم، صحيح البخاري، صحيح مسلم، سنن أبي داود، جامع الترمذي، سنن النسائي، سنن ابن ماجه، موطأ مالك، مسند أحمد، الأربعون النووية، الأحاديث القدسية
- **Tafsir:** تفسير ابن كثير، جامع البيان (الطبري)، الجامع لأحكام القرآن (القرطبي)، التحرير والتنوير (ابن عاشور)
- and 22 more books by other scholars

The full list, scholar by scholar, is in [docs/books.md](docs/books.md).

## Install

```bash
git clone https://github.com/abdallah-alkanani/dalil && cd dalil
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest          # runs offline, needs no data
```

Put your API key in a `.env` file (copy `.env.example`):

```
OPENROUTER_API_KEY=...
```

Search uses OpenRouter by default. Any OpenAI-compatible API works too: give its
address with `--openrouter-url` and the name of the variable that holds its key
with `--api-key-env`, for example `--openrouter-url https://api.example.com/v1
--api-key-env MY_API_KEY`. It must serve Qwen3-Embedding-8B
(`--embed-model`), the model the database was embedded with.

## Data

The books, the search databases and the vectors are in the Hugging Face dataset
[Alkanani/dalil](https://huggingface.co/datasets/Alkanani/dalil). Search needs them in `data/`, and a rebuild also needs `recipe/`:

```bash
hf download Alkanani/dalil --repo-type dataset --exclude "recipe/*" --local-dir data
hf download Alkanani/dalil --repo-type dataset --include "recipe/*" --local-dir .
```

| folder | what it is |
|---|---|
| `data/db/` | the search databases (about 28 GB): `dense_faiss_ar.index`, `faiss_metadata.sqlite`, `parents.sqlite`, `bm25.sqlite` |
| `data/books/original/` | every book in its original form, one folder per author and per book |
| `data/books/cleaned/` | every book's full cleaned text, one `.txt` per book, in the same folders |
| `data/passages/` | every passage as Parquet (citation, book, scholar, category, place, text); also shown in the Hugging Face viewer and loadable with `datasets` |
| `recipe/` | which books to use, where each one is cut, hadith grades |

Set `DALIL_DATA_DIR` to keep `data/` somewhere else.

## Search

```bash
python -m dalil.s8_search "فضل الصدقة" --category hadith
python -m dalil.s8_search "حكم المسح على الخفين" --expand-neighbours 1 --json
python -m dalil.s8_search "إنما الأعمال بالنيات" --retrieval bm25   # no API key needed
python -m dalil.s8_search --list-sources
```

`dalil-search` is the same command once installed. `--help` lists every filter:
category, book, author, Quran surah and ayah, hadith collection and number,
section, and more.

What a search does:

1. Keyword search (BM25 over normalised Arabic) and vector search
   (Qwen3-Embedding-8B, every chunk compared exactly) run side by side.
2. The two rankings are merged by passage with weighted reciprocal-rank fusion:
   vector 0.65, keyword 0.35 (`--dense-weight`, `--bm25-weight`).
3. Passages from the source type the question is about (hadith, Quran, fiqh, ...)
   keep their score and the rest are multiplied by 0.9. The type comes from
   keywords in the question, then from an LLM; `--boost` sets it by hand.
4. MMR reorders the top 80 so the list is not near-copies of one passage
   (`--mmr-lambda 0.7`).
5. Optional: `--listwise` has an LLM rerank the top 20, and
   `--query-planning auto` also searches with LLM rewrites of the question.
   Both are off by default.

Every LLM call goes through OpenRouter with `OPENROUTER_API_KEY`. With
`--retrieval bm25`, a search makes no API call. The source-type classifier runs only
when no keyword in the question names a source type; it and `--listwise` use
`DALIL_LISTWISE_MODEL`, and `--query-planning` uses `DALIL_PLANNER_MODEL` (optional
preferred provider `DALIL_PLANNER_PROVIDER`); both default to
`google/gemini-3.1-flash-lite`. On the 197 questions asked by people, the keywords left 96
to the classifier; it named the question's source type, in at most two types, for 69, and
the planner returned a valid, Arabic plan with the critic for 30 of 30
([docs/model_check.md](docs/model_check.md), `dalil/s8_search/model_check.py`). If the
question cannot be embedded, a hybrid search falls back to BM25, and the result's
`status.retrieval` says so.

## Answering with a language model

dalil returns evidence, not answers. To have a model answer from it, give the
model `prompts/answering_system_prompt.md` as its system prompt. It requires
every statement to name the scholar, the book and the place, every view to be
given, and every quote to be pasted as it is in the source.

## How it works

```mermaid
flowchart LR
    A[Book] --> B[DeepSeek reads it<br/>and picks how to cut it]
    B --> C[Parents<br/>a hadith, a fatwa,<br/>a mas'ala, a section]
    C --> D[Chunks<br/>each with a header naming<br/>its book and place]
    D --> E[BM25 + exact<br/>vector search]
    E --> F[Fusion and MMR]
    F --> G[Whole parent<br/>+ its neighbours]
    G --> H[Cited answer]
```

### Books are cut where the book divides

Each book becomes three layers: **elements** (its chapters, headings,
paragraphs and pages), **parents** (the units that are shown and cited: a
hadith, a mas'ala, a fatwa, a question and its answer) and **chunks** (the
pieces of a parent that are embedded and searched). Nothing is cut at a fixed
word count. A fatwa keeps its question with its answer, a mas'ala stays whole,
and a transcribed lesson is split at its speaker turns.

DeepSeek reads each book and chooses how it is cut: by heading, by fatwa or
mas'ala, by question and answer, by numbered item, and others
(`dalil/s3_structure/parent_policy.py`). Counting markers is not
enough: on the al-Saeedan books, counting `مسألة`, `س)` and `فصل` suggested one
policy, but the text is divided by `الفرع الأول…` ordinals and `((كتاب الطهارة))`
headings, and one of those books would have become a single 34,000-word parent.

Some parents are still too long, and classical Arabic prose often has no
punctuation to cut at. For those, DeepSeek reads the text in 8,000-word slices
that cover every word and names, in the text's own words, where the subject
changes (`dalil/s3_structure/semantic_boundaries.py`). Its answer is checked: an
anchor that is not found exactly once in its slice is dropped, and no cut is
made there. There is no fallback cut at a word count. In the database, 146
parents in 51 books (7.4 million words; the longest 555,464 words) were cut this
way, at 6,502 places. One of them, in سد الذرائع وتحريم الحيل لابن القيم, as it is recorded in
`recipe/decisions/semantic_boundaries.jsonl`:

```json
{"word_index": 29994,
 "anchor": "تنبيه: وقد تقدم أن وسيلة المحرم محرمة",
 "reason": "Each anchor marks a clear shift: new section headings, new arguments, or new sub-topics like rulings"}
```

The model's choices are written to files, and the build only applies them, so
no model is called while building and the same input always gives the same
database. Every cut can be reviewed before anything is embedded.

### Search small, return whole

Chunks are small (paragraph packs of at most 256 words), and each is embedded
with a header that names its book, its category, its place in the book and the
kind of text it is. The answer is the whole parent the chunk belongs to. A long
parent is cut down around the match, ending on sentence boundaries
(`--parent-max-chars`), and `--expand-neighbours N` adds the parents before and
after it in the book, so an argument that runs over several parents can be read
in order.

A hadith is searched three ways: its full text, its isnad and its matn are
separate chunks, so a phrase from the matn finds the hadith however long its
chain is. In the four Sunan, 19,291 hadith carry al-Albani's grade.

This keyword search (`--retrieval bm25`, no API key) matched the matn chunk
of a hadith in سنن أبي داود:

```
$ python -m dalil.s8_search "من غش فليس مني" --retrieval bm25 --hadith-collection sunan_abi_dawud --expand-neighbours 1 --json
```

The chunk that matched, as it was embedded:

```
الكتاب: سنن أبي داود
التصنيف: hadith
الموضع: سنن أبي داود > كتاب الأدب > حديث 5249
نوع النص: matn
النص: قَالَ رَسُولُ اللَّهِ صلى الله عليه وسلم " اقْتُلُوا الْحَيَّاتِ كُلَّهُنَّ فَمَنْ خَافَ ثَأْرَهُنَّ فَلَيْسَ مِنِّي " .
```

What is returned, cited as `سنن أبي داود، حديث 5249`: the whole hadith with its
isnad and al-Albani's grade.

```
حَدَّثَنَا عَبْدُ الْحَمِيدِ بْنُ بَيَانٍ السُّكَّرِيُّ، عَنْ إِسْحَاقَ بْنِ يُوسُفَ، عَنْ شَرِيكٍ، عَنْ أَبِي إِسْحَاقَ، عَنِ الْقَاسِمِ بْنِ عَبْدِ الرَّحْمَنِ، عَنْ أَبِيهِ، عَنِ ابْنِ مَسْعُودٍ، قَالَ قَالَ رَسُولُ اللَّهِ صلى الله عليه وسلم " اقْتُلُوا الْحَيَّاتِ كُلَّهُنَّ فَمَنْ خَافَ ثَأْرَهُنَّ فَلَيْسَ مِنِّي " .

حكم الألباني: صحيح، المشكاة (٤١٤٠ / التحقيق الثاني)
```

Its neighbours, حديث 5248 and حديث 5250, are the hadith beside it in كتاب الأدب, on
the same subject.

### Exact search, from disk

By default every question is compared with all 927,744 vectors (`IndexFlatIP`); no
approximate index is used. The 15.2 GB index is not loaded into memory: it is
read from disk in large blocks and scored with FAISS's own exact kernel, so
vector search runs on a 16 GB machine (`dalil/s8_search/dense.py`).

An optional faster mode is for a host with little RAM or a slow disk.
`python -m dalil.s7_index.compact --dims 1024` writes a 1.9 GB float16 copy of the first
1,024 dimensions (Qwen3-Embedding is Matryoshka-trained). With
`DALIL_DENSE_COMPACT_DIMS=1024`, search scans that copy in memory and rescores its top
1,000 exactly from the full index (`DALIL_DENSE_RESCORE`). Its results can differ from
exact search: on the 724 questions of the questions dataset it returned the same top 50
for 723 and 99.84% of the top 200
([docs/compact_check.md](docs/compact_check.md), `dalil/s7_index/compact_check.py`).

### Cleaned books

Every book's full cleaned text is published next to its original, in the same
folders. The cleaning removes markup, page markers and invisible PDF
characters, and puts back text the sources lost:

- Symbol-font characters become the words they stood for (U+F065 is صلى الله
  عليه وسلم; `dalil/s4_extract/text.py`).
- In إتحاف أهل الألباب, 2,155 honorifics and Quran brackets had become `?`. Each
  one is put back from evidence: the honorifics from another copy of the same
  book, the Quran quotes by matching the text against the Quran
  (`dalil/s2_prepare/question_marks.py`).
- Text that Ketabonline's page layout leaves out of the book (9.8% of صحيح أبي
  داود - الأم) is recovered (`dalil/s2_prepare/albani_restore.py`).
- 49,710 Shamela footnotes are put back with the passage they belong to, and
  each is also searchable on its own.

Search normalises Arabic the same way for the books and the question, without
merging different words such as على and علي, or رحمة and رحمه
(`dalil/arabic/normalize.py`).

To rebuild the database from the books, run `dalil/build/run_build.sh` (it needs
a GPU to embed).

## Layout

```
dalil/
  s1_acquire/    download books (Ketabonline, Shamela 4)
  s2_prepare/    turn each source format into clean text; per-book repairs
  s3_structure/  ask an LLM how each book should be cut
  s4_extract/    cut books into elements, parents and chunks
  s5_audit/      quality checks before embedding
  s6_embed/      embed chunks on a GPU (Qwen3-Embedding-8B)
  s7_index/      build the FAISS and SQLite stores
  s8_search/     the search engine and its command line
  build/         run_build.sh: runs the whole build
  catalog/       the book catalog and al-Albani's grades of the four Sunan
  arabic/        Arabic normalisation and cleaning
  llm/           LLM connections (OpenRouter, Ollama, OpenCode) and API keys
docs/            how it works, why, and the list of books
prompts/         the system prompt for the answering model
tests/           offline tests
web/             a search API and a web app over it (web/README.md)
```

## Documentation

- [`docs/pipeline.md`](docs/pipeline.md): building the database, each stage, adding a book
- [`docs/design.md`](docs/design.md): the stores, schemas, ids, parser profiles and retrieval
- [`docs/decisions.md`](docs/decisions.md): why things are the way they are, with the code that does it
- [`docs/books.md`](docs/books.md): the books in the database
- [`docs/status.md`](docs/status.md): what works now
- [`docs/compact_check.md`](docs/compact_check.md): the compact dense mode against exact search, per question
- [`docs/model_check.md`](docs/model_check.md): the classifier and planner models, per question

## License

The code is under the [MIT License](LICENSE). It does not cover the books, which
are not part of this repository.
