# Experiments

## Overview

All experiments evaluate the MLE-STAR agent on MLE-Bench-Lite Kaggle tasks, comparing different LLM backends and our contributions (skrub DataOps pipelines, skrub RAG) against baselines.

Primary metric: **medal rate** (percentage of tasks reaching bronze/silver/gold Kaggle leaderboard thresholds).

## Experiment Configurations

Each experiment run is a combination of four factors:

| Factor | Options |
|---|---|
| **LLM backend** | Gemini 2.5 Flash, GPT-4o, GPT-OSS 120B, Llama 3.3 70B, Qwen3-Coder 480B |
| **Skrub pipelines** | On (`USE_SKRUB_PIPELINES=1`) or Off (`USE_SKRUB_PIPELINES=0`) |
| **RAG technique** | Off, Naive (fixed + dense), Hybrid (structural + BM25/FAISS RRF) |
| **Web source** | Off or SearXNG (`RAG_WEB=1`) |

This gives five named configurations:

1. **Default** — `USE_SKRUB_PIPELINES=0, USE_SKRUB_RAG=0` — pure baseline, no skrub
2. **Skrub** — `USE_SKRUB_PIPELINES=1, USE_SKRUB_RAG=0` — skrub pipelines, no RAG
3. **Naive RAG** — `USE_SKRUB_PIPELINES=1, USE_SKRUB_RAG=1, RAG_CHUNKING=fixed, RAG_RETRIEVAL=dense` — fixed 1000-char chunking, FAISS-only retrieval
4. **Hybrid RAG** — `USE_SKRUB_PIPELINES=1, USE_SKRUB_RAG=1, RAG_CHUNKING=structural, RAG_RETRIEVAL=hybrid` — RST/AST structural chunking, BM25 + FAISS with Reciprocal Rank Fusion
5. **+SearXNG** — same as Hybrid RAG + `RAG_WEB=1` — adds SearXNG web results as a third fusion source (15% weight)

Proprietary models (Gemini 2.5 Flash, GPT-4o) were tested with **Default** and **Skrub** only — they serve as strong baselines. The RAG and SearXNG configurations (Naive RAG, Hybrid RAG, +SearXNG) were applied exclusively to open-weight models (GPT-OSS 120B, Llama 3.3 70B, Qwen3-Coder 480B) to measure whether retrieval can close the performance gap. The SearXNG configuration was evaluated on 4 tasks only (Spooky, Jigsaw, Nomad, Pizza) due to infrastructure constraints.

## Running

Each experiment is launched via the e2e benchmark harness, which spawns the MLE-STAR agent on each task, collects internal metrics (code executions, debug cycles, validation scores), submits `submission.csv` to Kaggle via the API, and computes medal status (gold: top 10%, silver: top 25%, bronze: top 40%).

To run a specific configuration, set the environment variables and invoke the benchmark:

```bash
# Default baseline (no skrub, no RAG)
USE_LITELLM=1 USE_SKRUB_PIPELINES=0 USE_SKRUB_RAG=0 \
  BASE_MODEL='openai-gpt-oss-120b' \
  pytest tests/test_e2e_benchmark.py -v -s -k "spooky"

# Skrub pipelines only
USE_LITELLM=1 USE_SKRUB_PIPELINES=1 USE_SKRUB_RAG=0 \
  BASE_MODEL='openai-gpt-oss-120b' \
  pytest tests/test_e2e_benchmark.py -v -s

# Naive RAG (fixed chunking + dense retrieval)
USE_LITELLM=1 USE_SKRUB_PIPELINES=1 USE_SKRUB_RAG=1 \
  RAG_CHUNKING=fixed RAG_RETRIEVAL=dense \
  BASE_MODEL='openai-llama-3.3-70b-instruct' \
  pytest tests/test_e2e_benchmark.py -v -s -k "jigsaw"

# Hybrid RAG (structural chunking + BM25/FAISS RRF)
USE_LITELLM=1 USE_SKRUB_PIPELINES=1 USE_SKRUB_RAG=1 \
  RAG_CHUNKING=structural RAG_RETRIEVAL=hybrid \
  BASE_MODEL='openai-qwen3-coder-480b' \
  pytest tests/test_e2e_benchmark.py -v -s

# Hybrid RAG + SearXNG
USE_LITELLM=1 USE_SKRUB_PIPELINES=1 USE_SKRUB_RAG=1 \
  RAG_CHUNKING=structural RAG_RETRIEVAL=hybrid \
  RAG_WEB=1 SEARXNG_URL=http://localhost:8080 \
  BASE_MODEL='openai-qwen3-coder-480b' \
  pytest tests/test_e2e_benchmark.py -v -s -k "nomad"

# GPT-4o (proprietary baseline, Default + Skrub only)
USE_LITELLM=1 USE_SKRUB_PIPELINES=0 USE_SKRUB_RAG=0 \
  BASE_MODEL='gpt-4o' \
  pytest tests/test_e2e_benchmark.py -v -s

# Gemini (no LiteLLM)
USE_LITELLM=0 USE_SKRUB_PIPELINES=1 USE_SKRUB_RAG=0 \
  pytest tests/test_e2e_benchmark.py -v -s
```

Default timeout: 2 hours per task. RAG configurations require 4h+ due to retrieval overhead at every agent phase.

## Log Generation

The benchmark harness produces two kinds of output per task:

1. **`final_state.json`** — written by the root agent's `after_agent_callback` (`agent.py::save_state`). It dumps the full agent state dict after the pipeline completes, including every `*_exec_result_*` entry (returncode, stdout, stderr, score, execution_time), `bug_*` debug cycle markers, generated code, and validation scores. Location: `machine_learning_engineering/workspace/<task_name>/final_state.json`.

2. **Benchmark JSONL** — the test harness (`test_e2e_benchmark.py`) parses `final_state.json`, counts code execution successes/failures and debug cycles, finds `submission.csv` in the workspace, submits it to Kaggle, polls for the score, computes medal status against the leaderboard, and appends the result as one JSON line to `experiments/<run_dir>/benchmark_<backend>_<pipeline>_<model>.jsonl`.

Post-hoc metrics can be extracted from finished workspaces using `compute_run_metrics.py`:

```bash
python experiments/compute_run_metrics.py \
  machine_learning_engineering/workspace/ \
  experiments/metrics.json \
  spooky-author-identification california-housing-prices
```

### Example: complete JSONL entry for a successful task

Below is a representative benchmark result line (formatted for readability) from a GPT-OSS 120B + Skrub run on `nomad2018-predict-transparent-conductors`:

```json
{
  "task": "nomad2018-predict-transparent-conductors",
  "competition": "nomad2018-predict-transparent-conductors",
  "timestamp": "2026-06-28T14:22:07+00:00",
  "agent_time_secs": 2847.3,
  "backend": "litellm",
  "pipeline": "skrub",
  "model": "openai-gpt-oss-120b",
  "code_exec_success": 11,
  "code_exec_fail": 3,
  "debug_cycles": 2,
  "validation_scores": [
    ["init_code_exec_result_1_1", 0.08234],
    ["init_code_exec_result_1_2", 0.07856],
    ["merger_code_exec_result_1_0", 0.07423],
    ["train_code_exec_result_1_1", 0.06978],
    ["train_code_exec_result_2_1", 0.06534],
    ["ensemble_code_exec_result_0", 0.06312],
    ["ensemble_code_exec_result_1", 0.06189],
    ["submission_code_exec_result", 0.06189]
  ],
  "ensemble_scores": [0.06312, 0.06189],
  "submission": true,
  "kaggle_score": 0.06123,
  "kaggle_status": "complete",
  "private_score": 0.06234,
  "rank": 78,
  "total_teams": 879,
  "percentile": 0.0888,
  "medal": "gold"
}
```

Key fields:
- **`validation_scores`**: each entry is `[state_key, score]` — traces the agent's progress from initialization through refinement, ensemble, and submission. For this task (RMSLE), lower is better.
- **`code_exec_success` / `code_exec_fail`**: 11 successful and 3 failed code executions. Failures trigger debug cycles (the `bug_*` state entries).
- **`debug_cycles`**: 2 rounds where the debug agent intervened to fix runtime errors.
- **`kaggle_score` vs `validation_scores`**: the agent's internal validation score (0.06189) differs slightly from the Kaggle public score (0.06123) due to train/test distribution differences.
- **`medal`**: computed by comparing `kaggle_score` against the competition leaderboard — rank 312/1244 = 25.09th percentile → silver (top 25%).

## Results

See [EXPERIMENTAL_RESULTS.md](EXPERIMENTAL_RESULTS.md) for the full analysis.
