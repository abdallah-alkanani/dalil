# Classifier and planner models

`python -m dalil.s8_search.model_check` on the 197 questions asked by people in the questions dataset: the classifier on the 96 that no keyword assigns a source type (the only ones it sees at query time), the planner on the first 30 of those, with the critic. Per-question results in `model_check.jsonl`.

| model | classifier correct | classifier failed | classifier median s | planner valid | planner median s |
|---|---:|---:|---:|---:|---:|
| `google/gemma-4-26b-a4b-it` | 85/96 | 0 | 0.7 | 30/30 | 5.1 |
| `google/gemini-3.1-flash-lite` | 80/96 | 0 | 0.8 | 30/30 | 2.6 |
| `mistralai/mistral-small-2603` | 77/96 | 0 | 0.7 | 30/30 | 5.4 |
| `anthropic/claude-haiku-5.5` | 0/96 | 96 | 1.0 | 1/30 | 5.3 |
| `deepseek/deepseek-v4.1-flash` | 2/96 | 94 | 0.8 | 0/30 | 10.4 |
| `qwen/qwen3.5-flash-02-23` | 13/96 | 0 | 0.8 | 0/30 | 1.0 |

The defaults are `google/gemma-4-26b-a4b-it` for the classifier (`DALIL_LISTWISE_MODEL`, most
correct) and `google/gemini-3.1-flash-lite` for the planner (`DALIL_PLANNER_MODEL`, valid on all
30 and the fastest). The models that fail spend the classifier's 20 and the planner's 900
output tokens on reasoning before they answer.
