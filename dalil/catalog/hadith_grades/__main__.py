"""Run the five steps in order: extract -> align -> build -> external_check -> finalize.

    python -m dalil.catalog.hadith_grades <folder with ara-*.min.json> [--db DB] [--work WORK] [--out OUT]

The folder holds fawazahmed0/hadith-api ara-{abudawud,tirmidhi,nasai,ibnmajah}.min.json
(https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/<name>.min.json), used only by the check.
"""
from __future__ import annotations

import sys

from . import align, build, external_check, extract, finalize
from .common import path_args


def main(argv: list[str] | None = None) -> int:
    ap = path_args(__doc__, db=True)
    ap.add_argument("folder", help="fawazahmed0/hadith-api ara-*.min.json files")
    args = ap.parse_args(argv)
    paths = ["--work", str(args.work)]
    for step, extra in ((extract, ["--db", str(args.db)]), (align, []), (build, []),
                        (external_check, [args.folder]), (finalize, ["--out", str(args.out)])):
        if step.main(paths + extra) != 0:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
