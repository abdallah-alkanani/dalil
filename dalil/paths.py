"""Where dalil finds its data.

Everything large lives under one data directory: ``data/`` next to the package,
or ``$DALIL_DATA_DIR``.

    books/original/   every book in its original form: <author>/<book>/ (Arabic names)
    books/cleaned/    every book's full cleaned text, one .txt in the same <author>/<book>/
    build/            everything the pipeline generates; build/books/ holds the
                      repaired copies that prepare writes
    db/               the search stores

Paths recorded inside manifests, such as ``books/original/ابن القيم/زاد المعاد/...``, are
relative to the data directory.

What a rebuild needs besides the source texts (book manifests, parser profiles,
and decisions made by an LLM or by hand) is versioned in ``recipe/``.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DALIL_DATA_DIR") or REPO_ROOT / "data").expanduser().resolve()

ORIGINAL_DIR = DATA_DIR / "books" / "original"
CLEANED_DIR = DATA_DIR / "books" / "cleaned"
BUILD_DIR = DATA_DIR / "build"
BUILD_BOOKS_DIR = BUILD_DIR / "books"
DB_DIR = DATA_DIR / "db"

RECIPE_DIR = REPO_ROOT / "recipe"
RECIPE_BOOKS = RECIPE_DIR / "books"
RECIPE_DECISIONS = RECIPE_DIR / "decisions"
# al-Albani's grade and the standard printed number of each hadith of the four Sunan
# (made by dalil.catalog.hadith_grades, reviewed, then copied here).
HADITH_GRADES_PATH = RECIPE_DECISIONS / "sunan_grades.jsonl.gz"
SEMANTIC_BOUNDARIES_PATH = RECIPE_DECISIONS / "semantic_boundaries.jsonl"
