# Classifier and planner models

`python -m dalil.s8_search.model_check` on the 197 questions asked by people in the questions dataset: the classifier on the 96 that no keyword assigns a source type (the only ones it sees at query time), correct when it names the question's source type and at most 2 in all; the planner on the first 30 of those, with the critic. Per-question results, by line number in the questions dataset (`question_id`), in `model_check.jsonl`.

| model | classifier correct | types named, mean | classifier failed | classifier median s | planner valid | planner median s |
|---|---:|---:|---:|---:|---:|---:|
| `google/gemma-4-26b-a4b-it` | 69/96 | 2.0 | 0 | 0.8 | 30/30 | 4.8 |
| `google/gemini-3.1-flash-lite` | 69/96 | 2.0 | 0 | 0.8 | 30/30 | 2.5 |
| `mistralai/mistral-small-2603` | 59/96 | 1.9 | 0 | 0.6 | 30/30 | 4.9 |
| `anthropic/claude-haiku-5.5` | 0/96 | - | 96 | 1.0 | 0/30 | 5.4 |
| `deepseek/deepseek-v4.1-flash` | 6/96 | 1.0 | 89 | 0.7 | 1/30 | 5.5 |
| `qwen/qwen3.5-flash-02-23` | 13/96 | 0.0 | 0 | 0.9 | 0/30 | 1.1 |

`google/gemini-3.1-flash-lite` is the default for both (`DALIL_LISTWISE_MODEL`,
`DALIL_PLANNER_MODEL`): it ties gemma on the classifier and is the faster planner. The models that
fail spend the classifier's 20 and the planner's 900 output tokens on reasoning before they answer.
