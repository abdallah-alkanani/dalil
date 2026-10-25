"""al-Albani's grade and the standard printed number of each hadith of the four Sunan.

The plain hadith_api Sunan in the search DB are aligned with al-Albani's graded
copies of the same books (also in the DB); fawazahmed0/hadith-api is used only to
check the result. Output: sunan_grades.jsonl (read by dalil.s4_extract.core_corpus)
and graded_entry_numbers.jsonl. ``python -m dalil.catalog.hadith_grades <dir>``
runs the five steps in order.
"""
