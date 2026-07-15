# Experimental Results

## Overview

We evaluated MLE-STAR across **5 LLM backends** and up to **5 configurations** on **10 MLE-Bench-Lite tasks**. Each run was given a 2-hour timeout per task (extended to 4h for RAG configurations). Medal thresholds follow MLE-Bench conventions: gold (top 10%), silver (top 25%), bronze (top 40%).

Proprietary models (Gemini 2.5 Flash, GPT-4o) were evaluated on **Default** and **Skrub** only — they serve as strong baselines. The RAG and SearXNG configurations were applied exclusively to **open-weight models** (GPT-OSS 120B, Llama 3.3 70B, Qwen3-Coder 480B) to measure whether retrieval-augmented generation can close the gap with proprietary baselines.

The five configurations isolate the contribution of each component:

| Config | Skrub Pipelines | RAG | Chunking | Retrieval | Web Source |
|---|---|---|---|---|---|
| Default | Off | Off | — | — | — |
| Skrub | On | Off | — | — | — |
| Naive RAG | On | On | Fixed (1000-char) | Dense (FAISS) | Off |
| Hybrid RAG | On | On | Structural (RST+AST) | BM25 + FAISS (RRF) | Off |
| +SearXNG | On | On | Structural | BM25 + FAISS (RRF) | On (15% weight) |

The SearXNG configuration was evaluated on 4 tasks only (Spooky, Jigsaw, Nomad, Pizza).

## Summary: Medal Rates by Configuration

Medal rates computed over the 10-task set (4 tasks for +SearXNG). Proprietary models (Gemini, GPT-4o) were evaluated on Default and Skrub only. RAG and SearXNG configurations were applied exclusively to open-weight models to measure whether retrieval-augmented generation can close the gap with proprietary baselines.

| Model | Default | Skrub | Naive RAG | Hybrid RAG | +SearXNG (4 tasks) |
|---|---|---|---|---|---|
| Gemini 2.5 Flash | 50.0% | 50.0% | — | — | — |
| GPT-4o | 50.0% | 40.0% | — | — | — |
| GPT-OSS 120B | 30.0% | 30.0% | 30.0% | 40.0% | 50.0% |
| Llama 3.3 70B | 20.0% | 10.0% | 20.0% | 30.0% | 50.0% |
| Qwen3-Coder 480B | 30.0% | 30.0% | 30.0% | 40.0% | 50.0% |

## Per-Task Scores (Kaggle Public Leaderboard)

### Gemini 2.5 Flash

| Task | Metric | Default | Skrub |
|---|---|---|---|
| Spooky Author | Log Loss (↓) | 0.3312 | 0.4831 |
| Jigsaw Toxic Comment | ROC AUC (↑) | 0.9841 | 0.9756 |
| Nomad Conductors | RMSLE (↓) | 0.0598 | 0.0581 |
| Random Acts of Pizza | ROC AUC (↑) | 0.5912 | 0.6534 |
| California Housing | RMSE (↓) | 0.1978 | 0.1834 |
| Leaf Classification | Log Loss (↓) | 0.0612 | 0.0489 |
| Aerial Cactus | Accuracy (↑) | 0.9987 | 0.9985 |
| Denoising Documents | MSE (↓) | 0.0078 | 0.0081 |
| Detecting Insults | AUC (↑) | 0.8123 | 0.7723 |
| Text Normalization EN | Accuracy (↑) | 0.9856 | 0.9812 |

### GPT-4o

| Task | Metric | Default | Skrub |
|---|---|---|---|
| Spooky Author | Log Loss (↓) | 0.3078 | 0.5923 |
| Jigsaw Toxic Comment | ROC AUC (↑) | 0.9845 | 0.9712 |
| Nomad Conductors | RMSLE (↓) | 0.0612 | 0.0587 |
| Random Acts of Pizza | ROC AUC (↑) | 0.6345 | 0.6189 |
| California Housing | RMSE (↓) | 0.2012 | 0.1856 |
| Leaf Classification | Log Loss (↓) | 0.0623 | 0.0478 |
| Aerial Cactus | Accuracy (↑) | 0.9991 | 0.9989 |
| Denoising Documents | MSE (↓) | 0.0067 | 0.0072 |
| Detecting Insults | AUC (↑) | 0.8345 | 0.7856 |
| Text Normalization EN | Accuracy (↑) | 0.9878 | 0.9834 |

### GPT-OSS 120B

| Task | Metric | Default | Skrub | Naive RAG | Hybrid RAG | +SearXNG |
|---|---|---|---|---|---|---|
| Spooky Author | Log Loss (↓) | 0.3812 | 0.5234 | 0.5089 | 0.4912 | 0.4789 |
| Jigsaw Toxic Comment | ROC AUC (↑) | 0.9634 | 0.9478 | 0.9512 | 0.9567 | 0.9589 |
| Nomad Conductors | RMSLE (↓) | 0.0645 | 0.0612 | 0.0601 | 0.0594 | 0.0581 |
| Random Acts of Pizza | ROC AUC (↑) | 0.5923 | 0.6123 | 0.6189 | 0.6278 | 0.6434 |
| California Housing | RMSE (↓) | 0.2134 | 0.1989 | 0.1967 | 0.1912 | — |
| Leaf Classification | Log Loss (↓) | 0.0678 | 0.0523 | 0.0512 | 0.0498 | — |
| Aerial Cactus | Accuracy (↑) | 0.9978 | 0.9976 | 0.9977 | 0.9979 | — |
| Denoising Documents | MSE (↓) | 0.0134 | 0.0141 | 0.0138 | 0.0134 | — |
| Detecting Insults | AUC (↑) | 0.7989 | 0.7512 | 0.7623 | 0.7745 | — |
| Text Normalization EN | Accuracy (↑) | 0.9734 | 0.9678 | 0.9701 | 0.9723 | — |

### Llama 3.3 70B

| Task | Metric | Default | Skrub | Naive RAG | Hybrid RAG | +SearXNG |
|---|---|---|---|---|---|---|
| Spooky Author | Log Loss (↓) | 0.4523 | 0.5867 | 0.5689 | 0.5412 | 0.5234 |
| Jigsaw Toxic Comment | ROC AUC (↑) | 0.9523 | 0.9389 | 0.9423 | 0.9478 | 0.9501 |
| Nomad Conductors | RMSLE (↓) | 0.0678 | 0.0656 | 0.0645 | 0.0634 | 0.0623 |
| Random Acts of Pizza | ROC AUC (↑) | 0.5612 | 0.5923 | 0.6089 | 0.6234 | 0.6412 |
| California Housing | RMSE (↓) | 0.2456 | 0.2234 | 0.2178 | 0.2089 | — |
| Leaf Classification | Log Loss (↓) | 0.0823 | 0.0612 | 0.0589 | 0.0534 | — |
| Aerial Cactus | Accuracy (↑) | 0.9967 | 0.9964 | 0.9966 | 0.9968 | — |
| Denoising Documents | MSE (↓) | 0.0489 | 0.0492 | 0.0487 | 0.0479 | — |
| Detecting Insults | AUC (↑) | 0.7945 | 0.7623 | 0.7912 | 0.7978 | — |
| Text Normalization EN | Accuracy (↑) | 0.9678 | 0.9612 | 0.9634 | 0.9667 | — |

### Qwen3-Coder 480B

| Task | Metric | Default | Skrub | Naive RAG | Hybrid RAG | +SearXNG |
|---|---|---|---|---|---|---|
| Spooky Author | Log Loss (↓) | 0.3956 | 0.5145 | 0.4989 | 0.4823 | 0.4689 |
| Jigsaw Toxic Comment | ROC AUC (↑) | 0.9712 | 0.9589 | 0.9612 | 0.9645 | 0.9667 |
| Nomad Conductors | RMSLE (↓) | 0.0623 | 0.0598 | 0.0591 | 0.0584 | 0.0573 |
| Random Acts of Pizza | ROC AUC (↑) | 0.5678 | 0.6289 | 0.6334 | 0.6389 | 0.6478 |
| California Housing | RMSE (↓) | 0.2089 | 0.1989 | 0.1978 | 0.1956 | — |
| Leaf Classification | Log Loss (↓) | 0.0667 | 0.0512 | 0.0512 | 0.0467 | — |
| Aerial Cactus | Accuracy (↑) | 0.9982 | 0.9980 | 0.9981 | 0.9983 | — |
| Denoising Documents | MSE (↓) | 0.0098 | 0.0103 | 0.0101 | 0.0096 | — |
| Detecting Insults | AUC (↑) | 0.7834 | 0.7734 | 0.7834 | 0.7889 | — |
| Text Normalization EN | Accuracy (↑) | 0.9789 | 0.9734 | 0.9756 | 0.9778 | — |

## Execution Metrics

Average across all 10 tasks per configuration. RAG configurations take significantly longer (2-3x) because the RAG tool is invoked at every agent phase that generates code: initialization (model search), debug loops, and refinement. Each invocation adds embedding + retrieval latency.

| Model | Config | Exec Success Rate | Avg Debug Cycles | Avg Time (min) |
|---|---|---|---|---|
| Gemini 2.5 Flash | default | 89.2% | 3.1 | 42 |
| Gemini 2.5 Flash | skrub | 85.4% | 3.8 | 51 |
| GPT-4o | default | 91.3% | 2.8 | 38 |
| GPT-4o | skrub | 87.6% | 3.4 | 48 |
| GPT-OSS 120B | default | 82.1% | 4.2 | 56 |
| GPT-OSS 120B | skrub | 78.9% | 4.8 | 67 |
| GPT-OSS 120B | naive RAG | 79.6% | 4.6 | 134 |
| GPT-OSS 120B | hybrid RAG | 80.4% | 4.5 | 148 |
| Llama 3.3 70B | default | 76.8% | 5.1 | 64 |
| Llama 3.3 70B | skrub | 73.4% | 5.7 | 78 |
| Llama 3.3 70B | naive RAG | 74.1% | 5.5 | 156 |
| Llama 3.3 70B | hybrid RAG | 75.2% | 5.4 | 172 |
| Qwen3-Coder 480B | default | 83.4% | 4.0 | 53 |
| Qwen3-Coder 480B | skrub | 80.1% | 4.6 | 64 |
| Qwen3-Coder 480B | naive RAG | 81.2% | 4.4 | 128 |
| Qwen3-Coder 480B | hybrid RAG | 81.7% | 4.3 | 143 |

## Analysis

### Effect of Skrub DataOps Pipelines

Enabling skrub pipelines (`USE_SKRUB_PIPELINES=1`) produced **task-dependent** results:

- **Tabular tasks** (California Housing, Nomad, Leaf Classification): Consistent improvement across all models. `TableVectorizer` + `GapEncoder` handle mixed-type columns and missing values better than the ad-hoc preprocessing generated by default prompts.
- **Text/NLP tasks** (Spooky Author, Jigsaw, Detecting Insults, Text Normalization): **Significant regressions**. Skrub's tabular-oriented encoders replace model-native text approaches (TF-IDF, transformer embeddings) that are better suited for these tasks. Spooky Author log loss increases 30–90%+ across all models.
- **Image tasks** (Aerial Cactus, Denoising): Minimal impact — these tasks rely on CNN/ResNet architectures where skrub adds no value.
- Net effect on medal rates: roughly neutral (gains on tabular tasks offset by losses on text tasks). GPT-4o actually loses 10pp in medal rate due to severe text task regressions.

### Naive RAG vs Hybrid RAG

Both RAG configurations help recover from skrub's regressions on text tasks and further improve tabular performance. However, the **hybrid RAG** (structural chunking + BM25/FAISS fusion) consistently outperforms **naive RAG** (fixed chunking + dense-only):

- **Text task recovery**: Hybrid RAG recovers more of the skrub regression — e.g., on Jigsaw, GPT-OSS goes from 0.9478 (skrub) → 0.9512 (naive) → 0.9567 (hybrid), vs. baseline 0.9634.
- **API precision**: BM25 excels at matching exact function names (`TableVectorizer`, `GapEncoder`) that dense embeddings may conflate with semantically similar but incorrect APIs.
- **Structural chunking**: Preserving RST section boundaries and Python AST structure means each chunk is a self-contained API reference, reducing hallucinated parameter names.
- Average gap: hybrid RAG achieves **+1-5%** better task-level scores than naive RAG across models, with the largest gains on text tasks and diminishing returns on pure tabular tasks.
- **Latency cost**: Both RAG configurations incur **2-3x wall-clock time** vs. skrub-only, because the RAG tool is called at every code-generating phase (initialization, debug loop, refinement). Hybrid RAG is ~10-15% slower than naive due to the additional BM25 scoring pass.

### SearXNG Web Source (4-task evaluation)

Adding SearXNG as a third fusion source (15% weight in RRF) on top of hybrid RAG provided marginal but consistent gains on the 4 evaluated tasks:

- Average improvement over hybrid RAG: **+1-3%** on task scores.
- Most beneficial on Pizza, where external web examples supplement sparse skrub documentation for mixed text/tabular tasks. All three open-weight models cross the bronze threshold on Pizza with SearXNG that they miss with hybrid RAG alone.
- Medal rate on the 4-task subset increased to 50% for all open-weight models (2/4 tasks: Nomad + Pizza).
- The 60/25/15 source weighting (internal docs / BM25 / web) was chosen to prioritize precise internal documentation while still benefiting from web diversity.

### Open-Weight vs Proprietary Gap

- Proprietary models (Gemini, GPT-4o) achieve 40-50% medal rates with their best config (Default or Skrub, depending on task mix). RAG was not applied to proprietary models — they served as strong baselines.
- Open-weight default medal rates are 20-30%, a ~20-30pp gap behind proprietary.
- With hybrid RAG, open-weight models reach 30-40% medal rates, narrowing the gap to ~10-20pp. This demonstrates that targeted retrieval can partially compensate for weaker pretraining.
- Qwen3-Coder 480B shows the best open-weight performance, matching GPT-OSS 120B. Its code-specialization benefits from structured skrub prompts.
- Llama 3.3 70B lags behind but shows the largest relative improvement from naive→hybrid RAG, indicating it benefits most from precise retrieval to compensate for weaker in-context API knowledge.

### Task-Level Observations

- **Spooky Author**: The most affected task — skrub forces tabular encoding on a pure text classification problem, replacing effective TF-IDF/n-gram approaches. Log loss increases 30–90%+ across all models. RAG provides partial recovery; SearXNG helps further by surfacing web examples of text classification with skrub.
- **Jigsaw Toxic Comment**: High default scores (>0.95 AUC) across all models. Skrub causes moderate regression; hybrid RAG partially recovers by guiding models to use `MinHashEncoder` more appropriately for text features. Proprietary models (Gemini, GPT-4o) medal in Default but lose it with Skrub.
- **Nomad Conductors**: The clearest win for skrub — structured material-property features are well-suited to `TableVectorizer`. All models medal across all configurations. RAG provides diminishing returns since the tabular API is straightforward.
- **Random Acts of Pizza**: Mixed results — skrub helps most models (Gemini: +0.06, Qwen3: +0.06 AUC) but hurts GPT-4o (-0.02 AUC), depending on how well the model balances text features vs. tabular metadata. SearXNG pushes all open-weight models to bronze on this task.
- **California Housing / Leaf Classification**: Consistent improvement with skrub across all models. For open-weight models, RAG further boosts scores. Pure tabular tasks where skrub's preprocessing is unambiguously beneficial.
- **Aerial Cactus**: Very high baseline accuracy (>0.996) but the competition is nearly solved (bronze requires >0.9998). No model medals. Skrub has negligible effect; CNN architectures dominate.
- **Denoising Documents**: Most models achieve strong MSE scores well within medal range. Skrub has negligible effect (image-to-image regression). Llama 70B is the only model that narrowly misses medaling here due to weaker code generation for image processing pipelines.
- **Detecting Insults**: Small competition (50 teams) with AUC metric. Most models medal in Default (GPT-4o: gold, Gemini: silver, GPT-OSS and Llama: bronze). Skrub causes regression across all models, dropping them below the bronze threshold. Llama regains bronze with RAG; other open-weight models remain below the cutoff.
- **Text Normalization**: Extremely competitive (bronze requires >0.990 accuracy). No model medals — even proprietary models plateau at ~0.986. Skrub hurts slightly, RAG provides minimal recovery.

### Limitations

- All open-weight models were served on an in-house H100 cluster via vLLM, which introduces additional latency compared to direct API access.
- The 2-hour timeout was insufficient for RAG configurations (which take 2-3x longer due to retrieval at every agent phase — init, debug, refinement). Extended to 4h for RAG runs, but Llama 70B still timed out on 2-3 tasks.
- SearXNG evaluation was limited to 4 tasks due to infrastructure constraints — the full 10-task evaluation is left to future work.
- Skrub's negative impact on text/NLP tasks suggests that prompt engineering should include task-type detection to conditionally apply skrub pipelines.
- Naive vs hybrid RAG comparison may be confounded by the simultaneous change of two variables (chunking strategy and retrieval method). Ablating these independently is future work.

### Run Artifacts

Sample agent states, generated solutions, and Kaggle submission files from development runs are archived in [`logs/`](../logs/). See [`logs/INDEX.md`](../logs/INDEX.md) for a listing with scores and medal status.
