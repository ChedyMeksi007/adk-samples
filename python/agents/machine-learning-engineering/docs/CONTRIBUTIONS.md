# Contributions

## Mohamed Chedy Meksi

- **LiteLLM backend integration**: Replaced the Gemini-only backend with a configurable LiteLLM router supporting open-source models via an in-house H100 cluster. Implemented the `LiteLlm` model wrapper, Vertex AI conditional loading, and API compatibility fixes (context window truncation, GPT-5 parameter handling).
  - `machine_learning_engineering/shared_libraries/config.py` — backend toggle, model instantiation, API endpoint configuration
  - `machine_learning_engineering/agent.py` — conditional model selection
  - `.env.example` — configuration template

- **Benchmark harness and metrics tooling**: Built the end-to-end benchmark test that runs the agent on all tasks, submits to Kaggle, polls scores, and computes medal status.
  - `tests/test_e2e_benchmark.py` — parametrized benchmark harness with Kaggle submission
  - `experiments/compute_run_metrics.py` — metrics extraction from `final_state.json`

- **Bug fixes and compatibility**: Fixed null text part crashes in response parsing, GPT-5 unsupported parameter errors, and context window overflow for GPT-4o.
  - `machine_learning_engineering/shared_libraries/common_util.py` — null text handling
  - `machine_learning_engineering/shared_libraries/config.py` — temperature/parameter guards

## Hachem Gaiech

- **Skrub RAG system**: Designed and implemented the retrieval-augmented generation pipeline for skrub documentation. Built the FAISS dense index, BM25 sparse index, structural chunking (RST + Python AST + numpydoc), and hybrid retrieval with Reciprocal Rank Fusion. Added SearXNG web source integration as a third fusion source with configurable weights.
  - `machine_learning_engineering/shared_libraries/skrub_rag.py` — full RAG pipeline (chunking, indexing, retrieval, fusion)
  - `machine_learning_engineering/shared_libraries/skrub_rag_index.faiss`, `skrub_rag_bm25.pkl`, `skrub_rag_chunks*.pkl` — pre-built indexes
  - `docs/EXPERIMENTAL_RESULTS.md` — experimental analysis including RAG technique comparison

- **Skrub DataOps prompt engineering**: Wrote the dual-mode prompt variants for all sub-agents, instructing the LLM to generate skrub DataOps computation graphs (`TableVectorizer`, `GapEncoder`, `MinHashEncoder`, `tabular_learner`) instead of plain sklearn pipelines. Later rewrote prompts to generate real DataOps plans.
  - `machine_learning_engineering/sub_agents/initialization/prompt.py` — skrub model retrieval and evaluation prompts
  - `machine_learning_engineering/sub_agents/refinement/prompt.py` — skrub ablation study prompts
  - `machine_learning_engineering/sub_agents/ensemble/prompt.py` — skrub ensemble strategy prompts
  - `machine_learning_engineering/prompt.py` — root system instruction (skrub DataOps focus)

- **Experiment execution and result collection**: Ran the open-source LLM benchmark experiments (GPT-OSS 120B, Llama 70B baselines and skrub configurations), collected and organized all benchmark result artifacts.

## Azer Mahjoub

- **Task dataset curation and pipeline**: Curated the 10 MLE-Bench-Lite Kaggle task datasets, wrote the download script, and set up the task directory structure with descriptions, train/test splits, and sample submissions.
  - `machine_learning_engineering/tasks/` — 10 task directories with data and descriptions
  - `download_tasks.sh` — automated Kaggle dataset downloader
  - `machine_learning_engineering/shared_libraries/check_leakage_util.py` — data leakage detection utility

- **Skrub pipeline mode integration**: Integrated the skrub pipeline toggle into the agent logic, wiring `USE_SKRUB_PIPELINES` into the initialization, refinement, and ensemble phases so the agent conditionally switches between sklearn and skrub code generation.
  - `machine_learning_engineering/sub_agents/initialization/agent.py` — conditional tool injection and skrub RAG wiring
  - `machine_learning_engineering/sub_agents/refinement/agent.py` — skrub-aware ablation agents
  - `machine_learning_engineering/shared_libraries/debug_util.py` — debug agent with skrub tool support

- **GPT-4o experiment runs**: Executed the GPT-4o baseline and skrub benchmark runs across all 10 tasks, including extended retry configurations.

## Shared / Joint Work

- **Project setup**: Environment configuration, dependency management (`pyproject.toml`, `requirements.txt`, `setup.sh`).
- **Experiment design**: Defined the three-factor experimental matrix (LLM backend x skrub pipelines x skrub RAG) and the evaluation protocol (medal rate on MLE-Bench-Lite).
- **Code review and debugging**: All members participated in iterative debugging, prompt tuning, and result analysis throughout the project.
