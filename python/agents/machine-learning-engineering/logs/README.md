# Experiment Logs

Raw artifacts from MLE-STAR experiment runs on MLE-Bench-Lite Kaggle tasks.

## Structure

Each subdirectory corresponds to a `(model, configuration)` pair:

| Directory | Model | Configuration |
|---|---|---|
| `gptoss120b_skrub/` | GPT-OSS 120B | Skrub pipelines |
| `gptoss120b_skrub_rag/` | GPT-OSS 120B | Skrub + Hybrid RAG |
| `llama70b_skrub/` | Llama 3.3 70B | Skrub pipelines |
| `llama70b_skrub_rag/` | Llama 3.3 70B | Skrub + Hybrid RAG |
| `qwen3coder_baseline/` | Qwen3-Coder 480B | Default (no skrub) — submission CSVs only (no final_state/solution) |
| `qwen3coder_skrub/` | Qwen3-Coder 480B | Skrub pipelines |
| `qwen3coder_skrub_rag/` | Qwen3-Coder 480B | Skrub + Hybrid RAG |

## Artifacts per task

- **`<task>_submission.csv`** — Kaggle submission file (gitignored — jigsaw CSVs are ~22MB)
- **`<task>_final_state.json`** — Full agent state dump after pipeline completion (exec results, validation scores, debug cycles)
- **`<task>_final_solution.py`** — Final generated solution code

## Coverage

These logs cover a **4-task subset** (Spooky, Jigsaw, Nomad, Pizza) used during development and validation. The full 10-task evaluation reported in `EXPERIMENTAL_RESULTS.md` was conducted separately with the final pipeline configuration.

## Extracting metrics

These logs use flat naming (`<task>_final_state.json`) rather than the workspace subdirectory layout that `compute_run_metrics.py` expects. To parse metrics directly:

```python
import json
state = json.load(open("logs/gptoss120b_skrub/nomad2018-predict-transparent-conductors_final_state.json"))
exec_results = {k: v for k, v in state.items() if "exec_result" in k and isinstance(v, dict)}
scores = [(k, v["score"]) for k, v in exec_results.items() if "score" in v]
```

## Index

See [INDEX.md](INDEX.md) for a complete listing of all runs with Kaggle scores and medal status.
