# MLE-STAR: Open-Source LLM Backend + Skrub DataOps for Automated Machine Learning

## Overview

This project extends **MLE-STAR** (Machine Learning Engineering Agent via Search and Targeted Refinement) — a multi-agent system that autonomously solves Kaggle-style ML competitions. The original system, described in [arxiv.org/abs/2506.15692](https://www.arxiv.org/abs/2506.15692), uses Gemini as its sole LLM backend and generates standard sklearn pipelines.

Our contributions add three capabilities:

1. **Open-source LLM backend** via LiteLLM, routing to models hosted on an in-house H100 cluster (Llama 3.3 70B, GPT-OSS 120B, Qwen3-Coder 480B) and OpenAI GPT-4o.
2. **Skrub DataOps pipeline generation** — all code-generation prompts have a skrub variant that instructs the model to use `TableVectorizer`, `GapEncoder`, `MinHashEncoder`, `DatetimeEncoder`, and `tabular_learner`.
3. **RAG over skrub documentation** — a FAISS + BM25 hybrid retrieval system over skrub docs, examples, and docstrings, queried at runtime by agents that need skrub API knowledge.

Every contribution is toggled via environment variables so each component's effect can be isolated in experiments.

**Research question:** Can open-source LLMs combined with targeted RAG and structured skrub pipelines close the gap with proprietary Gemini on automated ML benchmarks?

## Project Structure

```
machine-learning-engineering/
├── machine_learning_engineering/
│   ├── agent.py                          # Root agent (mle_frontdoor_agent)
│   ├── prompt.py                         # System & task prompts
│   ├── shared_libraries/
│   │   ├── config.py                     # Runtime config + LiteLLM setup
│   │   ├── skrub_rag.py                  # RAG system (FAISS + BM25 + SearXNG)
│   │   ├── code_util.py                  # Python code execution
│   │   ├── debug_util.py                 # Bug detection, debug agents, rollback
│   │   ├── common_util.py               # Text extraction, file I/O
│   │   ├── check_leakage_util.py        # Data leakage detection
│   │   ├── skrub_rag_index.faiss        # Pre-built dense index
│   │   ├── skrub_rag_bm25.pkl           # Pre-built BM25 index
│   │   └── skrub_rag_chunks*.pkl        # Chunk metadata
│   └── sub_agents/
│       ├── initialization/               # Phase 1: model search & generation
│       ├── refinement/                   # Phase 2: ablation & improvement
│       ├── ensemble/                     # Phase 3: voting/stacking
│       └── submission/                   # Phase 4: test inference & CSV output
├── experiments/                           # Experiment runner scripts & logs
│   ├── run_all_experiments.sh            # Main experiment orchestrator
│   ├── run_4tasks_dataops_experiment.sh  # 4-task sweep
│   ├── run_retry8h_skrub_rag.sh          # Extended 8h retry runs
│   ├── compute_run_metrics.py            # Metrics extraction
│   └── *.log                             # Experiment run logs
├── docs/                                  # Project documentation
│   ├── README.md                          # This file
│   ├── CONTRIBUTIONS.md                   # Per-member contribution breakdown
│   ├── EXPERIMENTS.md                     # Experiment configurations & scripts
│   └── EXPERIMENTAL_RESULTS.md            # Full results & analysis
├── tests/
│   └── test_e2e_benchmark.py             # E2E benchmark harness
├── machine_learning_engineering/tasks/    # MLE-Bench-Lite datasets (10 tasks)
├── .env.example                          # Configuration template
├── requirements.txt
└── pyproject.toml
```

## Agent Architecture

The system is a 4-phase sequential pipeline of nested agents:

```
mle_frontdoor_agent (root)
  └── mle_pipeline_agent [Sequential]
        ├── initialization_agent      — Web search for SOTA models, parallel
        │                               solution generation, candidate eval & ranking
        ├── refinement_agent          — Per-solution ablation studies, targeted
        │                               code block improvement
        ├── ensemble_agent            — Voting/stacking strategy across solutions
        └── submission_agent          — Test inference and submission.csv generation
```

Each code generation step wraps in a run-debug-rollback loop: generate code, execute in subprocess, parse validation metrics, and on failure retry with a debug agent (up to 10 retries).

See [EXPERIMENTS.md](EXPERIMENTS.md) for the experiment configurations and [EXPERIMENTAL_RESULTS.md](EXPERIMENTAL_RESULTS.md) for the full analysis.

## Our Contributions

### 1. LiteLLM Backend (`shared_libraries/config.py`)

Replaces the Gemini-only backend with a configurable LiteLLM router that supports any OpenAI-compatible endpoint. We route through an in-house H100 cluster to access:

- **Llama 3.3 70B** (`openai-llama-3.3-70b-instruct`)
- **GPT-OSS 120B** (`openai-gpt-oss-120b`)
- **Qwen3-Coder 480B** (`openai-qwen3-coder-480b`)
- **GPT-4o** via OpenAI API

Toggle: `USE_LITELLM=1` in `.env`.

### 2. Skrub DataOps Prompts (`sub_agents/*/prompt.py`)

All sub-agent prompts have dual-mode variants. When enabled, the system instructs the LLM to build solutions as skrub DataOps computation graphs instead of standard sklearn pipelines.

Toggle: `USE_SKRUB_PIPELINES=1` in `.env`.

### 3. Skrub RAG System (`shared_libraries/skrub_rag.py`)

A retrieval-augmented generation system providing skrub API knowledge to agents at runtime:

- **Chunking modes**: `fixed` (1000-char overlapping) or `structural` (RST sections + Python AST + numpydoc)
- **Retrieval modes**: `dense` (FAISS cosine) or `hybrid` (BM25 + dense with Reciprocal Rank Fusion)
- **Optional web source**: SearXNG integration as a third fusion source with configurable weights
- **Tool**: `search_skrub_docs(query)` injected into model retrieval, debug, and run agents

Toggle: `USE_SKRUB_RAG=1`, `RAG_CHUNKING=structural`, `RAG_RETRIEVAL=hybrid` in `.env`.

See [EXPERIMENTAL_RESULTS.md](EXPERIMENTAL_RESULTS.md) for the RAG technique analysis and full benchmark results.

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management

### Installation

```bash
git clone <repo-url>
cd python/agents/machine-learning-engineering
cp .env.example .env
# Fill in API keys in .env
uv sync
```

### Download Task Datasets

```bash
./download_tasks.sh
```

### Configuration

All settings are controlled via environment variables in the `.env` file. Copy `.env.example` and adjust:

```bash
cp .env.example .env
```

#### LLM Backend

| Variable | Values | Description |
|---|---|---|
| `USE_LITELLM` | `0` / `1` | `0` = Gemini via Google API (requires `GOOGLE_API_KEY`). `1` = any OpenAI-compatible endpoint via LiteLLM. |
| `BASE_MODEL` | model ID | LiteLLM model identifier, e.g. `openai-gpt-oss-120b`, `openai-llama-3.3-70b-instruct`, `openai-qwen3-coder-480b` |
| `API_ENDPOINT` | URL | vLLM / Ollama endpoint URL (e.g. `http://localhost:8000/v1/chat/completions`) |
| `API_KEY` | string | API key for the endpoint |
| `MAX_TOKENS` | int | Max output tokens per LLM call (default: 300 for fast models, increase for complex tasks) |
| `TEMPERATURE` | float | Sampling temperature (default: 0.5) |
| `REQUEST_TIMEOUT` | int | Seconds before LiteLLM times out a single call (default: 3600, needed for large-context models) |

When `USE_LITELLM=1`, `config.py` strips the `/chat/completions` suffix from `API_ENDPOINT` (LiteLLM appends it automatically), sets `OPENAI_API_KEY` and `OPENAI_API_BASE` in the process environment, and instantiates a `LiteLlm` model wrapper. Unsupported parameters (e.g. `temperature` for certain models) are silently dropped via `litellm.drop_params = True`.

#### Skrub DataOps Pipeline Toggle

| Variable | Values | Description |
|---|---|---|
| `USE_SKRUB_PIPELINES` | `0` / `1` | `0` = default sklearn/pytorch prompts. `1` = skrub DataOps prompts. |

**How prompt toggling works:** Every sub-agent prompt file (`sub_agents/*/prompt.py`) defines **two variants** of each instruction — a default version and a skrub version. For example, `initialization/prompt.py` has both `MODEL_RETRIEVAL_INSTR` (skrub) and `MODEL_RETRIEVAL_INSTR_DEFAULT` (sklearn). At runtime, each agent's instruction builder reads `use_skrub_pipelines` from the shared agent state and selects the appropriate prompt:

```python
# sub_agents/initialization/agent.py
use_skrub = context.state.get("use_skrub_pipelines", True)
instr = prompt.MODEL_RETRIEVAL_INSTR if use_skrub else prompt.MODEL_RETRIEVAL_INSTR_DEFAULT
```

This toggling happens in **all four phases**: initialization (model search + code generation), refinement (ablation studies), ensemble (stacking strategy), and submission (test inference). The skrub prompts instruct the LLM to build solutions as skrub DataOps computation graphs (`skrub.var(...)` → `.skb.mark_as_X()` → `.skb.apply(TableVectorizer())` → `.skb.make_learner()`) instead of standard `sklearn.pipeline.Pipeline`.

The toggle is set once at startup via `configure_pipeline_mode()` in `agent.py`, which writes the flag into the shared ADK agent state so all sub-agents read the same value.

#### Skrub RAG System

| Variable | Values | Description |
|---|---|---|
| `USE_SKRUB_RAG` | `0` / `1` | `0` = no RAG tool. `1` = inject `search_skrub_docs(query)` tool into agents. |
| `RAG_CHUNKING` | `fixed` / `structural` | `fixed` = 1000-char overlapping windows. `structural` = RST section boundaries + Python AST + numpydoc parsing. |
| `RAG_RETRIEVAL` | `dense` / `hybrid` | `dense` = FAISS cosine similarity only. `hybrid` = BM25 sparse + FAISS dense fused with Reciprocal Rank Fusion. |
| `RAG_TOKEN_BUDGET` | int | Max tokens of retrieved context (default: 2000). Controls how much documentation is injected per query. |
| `RAG_MIN_SCORE` | float | Cosine similarity floor for FAISS candidates (default: 0.2). Lower = more results, noisier. |

**How RAG works at runtime:** When `USE_SKRUB_RAG=1`, the `search_skrub_docs(query)` tool is injected into the model retrieval agent, the debug agent, and the refinement agents. Each time one of these agents generates or fixes code, it can call this tool to retrieve relevant skrub API documentation.

The retrieval pipeline:
1. **Chunking** (offline, at index build time): skrub documentation is split into chunks. `fixed` mode uses simple 1000-character windows with 200-char overlap. `structural` mode parses RST section headers, Python AST nodes (classes, functions, methods), and numpydoc parameter blocks to create semantically meaningful chunks that preserve API boundaries.
2. **Dense retrieval**: The query is embedded using the configured embedding model, then matched against a pre-built FAISS index (`skrub_rag_index.faiss`). Top candidates above `RAG_MIN_SCORE` are returned.
3. **BM25 retrieval** (hybrid mode only): The query is also matched against a BM25 index (`skrub_rag_bm25.pkl`) for keyword-level matching. This catches exact function names like `TableVectorizer` that dense embeddings might miss.
4. **Reciprocal Rank Fusion** (hybrid mode only): Dense and BM25 results are merged using RRF scoring (`1 / (k + rank)`) to combine semantic and keyword relevance.
5. **Token budget**: Results are truncated to fit within `RAG_TOKEN_BUDGET` tokens before being returned to the agent.

Since the RAG tool is called at **every code-generating phase** (initialization, debug loops, refinement), RAG configurations take approximately **2-3x longer** than non-RAG runs.

#### SearXNG Web Source

| Variable | Values | Description |
|---|---|---|
| `RAG_WEB` | `0` / `1` | `0` = internal docs only. `1` = add SearXNG web results as a third fusion source. |
| `SEARXNG_URL` | URL | Self-hosted SearXNG instance (e.g. `http://localhost:8080`). Must have JSON format enabled. |
| `RAG_WEB_TOP` | int | Number of web results to fetch before preprocessing (default: 10). |
| `RAG_WEIGHTS` | floats | Comma-separated source priority weights for RRF fusion: `internal(dense),keyword(BM25),web` (default: `0.60,0.25,0.15`). |

When enabled, SearXNG results are preprocessed (HTML stripped, chunked) and added as a third source in the RRF fusion step. The default weights (60% internal docs, 25% BM25, 15% web) prioritize precise local documentation while supplementing with up-to-date web examples.

#### Task Selection

| Variable | Values | Description |
|---|---|---|
| `TASK` | task name | Active Kaggle task directory name (e.g. `jigsaw-toxic-comment-classification-challenge`) |

#### Experiment Presets

Common configurations for reproducing experiments:

```bash
# Default baseline (no skrub, no RAG)
USE_LITELLM=1 USE_SKRUB_PIPELINES=0 USE_SKRUB_RAG=0

# Skrub only (pipelines, no RAG)
USE_LITELLM=1 USE_SKRUB_PIPELINES=1 USE_SKRUB_RAG=0

# Naive RAG (fixed chunking + dense retrieval)
USE_LITELLM=1 USE_SKRUB_PIPELINES=1 USE_SKRUB_RAG=1 RAG_CHUNKING=fixed RAG_RETRIEVAL=dense

# Hybrid RAG (structural chunking + BM25/FAISS RRF)
USE_LITELLM=1 USE_SKRUB_PIPELINES=1 USE_SKRUB_RAG=1 RAG_CHUNKING=structural RAG_RETRIEVAL=hybrid

# Hybrid RAG + SearXNG
USE_LITELLM=1 USE_SKRUB_PIPELINES=1 USE_SKRUB_RAG=1 RAG_CHUNKING=structural RAG_RETRIEVAL=hybrid RAG_WEB=1 SEARXNG_URL=http://localhost:8080
```

See [.env.example](../.env.example) for the full list of variables with inline documentation.

## Running

### Interactive Mode

```bash
uv run adk run machine_learning_engineering   # CLI
uv run adk web                                 # Web UI
```

### Running Benchmarks

```bash
# Single task
pytest tests/test_e2e_benchmark.py -v -s -k "spooky"

# All 10 tasks
pytest tests/test_e2e_benchmark.py -v -s

# Full experiment sweep
cd experiments && bash run_all_experiments.sh
```

## Benchmark Tasks

10 MLE-Bench-Lite Kaggle competitions:

| Task | Type | Metric |
|---|---|---|
| California Housing Prices | Tabular Regression | RMSE |
| Jigsaw Toxic Comment Classification | Multi-label Classification | Mean ROC AUC |
| Nomad 2018 Transparent Conductors | Multi-output Regression | RMSLE |
| Random Acts of Pizza | Binary Classification | ROC AUC |
| Spooky Author Identification | Multi-class Classification | Log Loss |
| Leaf Classification | Multi-class Classification | Accuracy |
| Aerial Cactus Identification | Image Classification | Accuracy |
| Denoising Dirty Documents | Image-to-Image Regression | MSE |
| Detecting Insults in Social Commentary | Binary Classification | Accuracy |
| Text Normalization (EN/RU) | Sequence Prediction | Accuracy |

## Reference

- MLE-STAR paper: [arxiv.org/abs/2506.15692](https://www.arxiv.org/abs/2506.15692)
- Google ADK: [github.com/google/adk-python](https://github.com/google/adk-python)
- skrub: [skrub-data.org](https://skrub-data.org/)
