# Compact dense search against exact search

`python -m dalil.s7_index.compact_check` on the 724 questions of the dalil questions dataset (197 asked by people, 513 by the research agent, 14 from eval files), compact index 1024 dims fp16, rescore 1000. Overlap of the top k retrieval views; per-question results in `compact_check.jsonl`.

### all (724 questions)

| | top 10 | top 50 | top 200 |
|---|---:|---:|---:|
| mean overlap | 99.99% | 100.00% | 99.84% |
| worst question | 9/10 | 49/50 | 187/200 |
| questions identical as a set | 723/724 | 723/724 | 633/724 |
| questions identical in order | 719/724 | 700/724 | 566/724 |

### source = agent (513 questions)

| | top 10 | top 50 | top 200 |
|---|---:|---:|---:|
| mean overlap | 99.98% | 100.00% | 99.86% |
| worst question | 9/10 | 50/50 | 189/200 |
| questions identical as a set | 512/513 | 513/513 | 450/513 |
| questions identical in order | 510/513 | 498/513 | 406/513 |

### source = eval_file (14 questions)

| | top 10 | top 50 | top 200 |
|---|---:|---:|---:|
| mean overlap | 100.00% | 100.00% | 100.00% |
| worst question | 10/10 | 50/50 | 200/200 |
| questions identical as a set | 14/14 | 14/14 | 14/14 |
| questions identical in order | 14/14 | 14/14 | 10/14 |

### source = user (197 questions)

| | top 10 | top 50 | top 200 |
|---|---:|---:|---:|
| mean overlap | 100.00% | 99.99% | 99.77% |
| worst question | 10/10 | 49/50 | 187/200 |
| questions identical as a set | 197/197 | 196/197 | 169/197 |
| questions identical in order | 195/197 | 188/197 | 150/197 |
