"""Configuration for Machine Learning Engineering Agent."""

import dataclasses
import os
from typing import Any

# Backend toggle: USE_LITELLM=1 -> OpenAI-compatible via LiteLLM, 0 -> Gemini
# When enabled, LiteLLM wraps any OpenAI-compatible endpoint (vLLM, Ollama, etc.)
# and handles model routing, retries, and parameter compatibility automatically.
_use_litellm = os.environ.get("USE_LITELLM", "1") not in ("0", "false", "False")

if _use_litellm:
    import litellm as _litellm
    _litellm.drop_params = True  # Drop unsupported params (e.g. temperature for GPT-5 models)
    from google.adk.models.lite_llm import LiteLlm

    # Configure OpenAI-compatible endpoint from .env
    # Strip /chat/completions suffix since LiteLLM appends it automatically
    _api_endpoint = os.getenv('API_ENDPOINT', '')
    if _api_endpoint.endswith('/chat/completions'):
        _api_endpoint = _api_endpoint[: -len('/chat/completions')]
    os.environ['OPENAI_API_KEY'] = os.getenv('API_KEY', '')
    os.environ['OPENAI_API_BASE'] = _api_endpoint
    _base_model = os.getenv('BASE_MODEL', 'llama-3.3-70b-instruct')
    _max_tokens = int(os.getenv('MAX_TOKENS', '8192'))
    # Large request timeout: the shared Ollama server can take many minutes for a
    # single big-context generation. LiteLLM's 600s default raised an unhandled
    # Timeout that crashed the whole agent (e.g. llama on jigsaw, 1884s/call).
    _request_timeout = int(os.getenv('REQUEST_TIMEOUT', '3600'))
    # Retry transient errors (the shared Ollama server intermittently drops
    # connections -> litellm InternalServerError "Connection error", which would
    # otherwise crash the whole agent run). num_retries makes LiteLLM retry these.
    _num_retries = int(os.getenv('LLM_NUM_RETRIES', '20'))
    _AGENT_MODEL = LiteLlm(
        model='openai/{}'.format(_base_model),
        max_tokens=_max_tokens,
        timeout=_request_timeout,
        num_retries=_num_retries,
    )
else:
    _AGENT_MODEL = os.environ.get("ROOT_AGENT_MODEL", "gemini-2.0-flash-001")


# Per-task metadata: (task_type, lower_is_better)
TASK_METADATA: dict[str, tuple[str, bool]] = {
    "california-housing-prices": ("Tabular Regression", True),
    "nomad2018-predict-transparent-conductors": ("Tabular Multi-Output Regression", True),
    "spooky-author-identification": ("Multi-Class Text Classification", True),
    "jigsaw-toxic-comment-classification-challenge": ("Multi-Label Text Classification", False),
    "random-acts-of-pizza": ("Binary Text Classification", False),
    "detecting-insults-in-social-commentary": ("Binary Text Classification", False),
    "aerial-cactus-identification": ("Image Classification", False),
    "leaf-classification": ("Multi-Class Tabular Classification", True),
    "denoising-dirty-documents": ("Image-to-Image Regression", True),
    "text-normalization-challenge-english-language": ("Text-to-Text Sequence Prediction", False),
    "text-normalization-challenge-russian-language": ("Text-to-Text Sequence Prediction", False),
}

_task_name = os.getenv("TASK", "california-housing-prices")
_task_type, _lower = TASK_METADATA.get(_task_name, ("Unknown", True))


@dataclasses.dataclass
class DefaultConfig:
    """Default configuration."""

    data_dir: str = "./machine_learning_engineering/tasks/"  # the directory path where the machine learning tasks and their data are stored.
    task_name: str = _task_name  # The name of the specific task to be loaded and processed.
    task_type: str = _task_type  # The type of machine learning problem (auto-detected from TASK_METADATA).
    lower: bool = _lower  # True if a lower value of the metric is better (auto-detected from TASK_METADATA).
    workspace_dir: str = "./machine_learning_engineering/workspace/"  # Directory used for saving intermediate outputs, results, logs.
    agent_model: Any = dataclasses.field(
        default_factory=lambda: _AGENT_MODEL
    )  # LiteLlm instance (USE_LITELLM=1) or Gemini model string (USE_LITELLM=0).
    use_litellm: bool = _use_litellm  # True = OpenAI-compatible backend, False = Gemini backend.
    task_description: str = ""  # The detailed description of the task.
    task_summary: str = ""  # The concise summary of the task.
    start_time: float = 0.0  # Timestamp indicating the start time of the task. Typically represented in seconds since the epoch.
    seed: int = 42  # The random seed value used to ensure reproducibility of experiments.
    exec_timeout: int = (
        600  # The maximum time in seconds allowed to complete the task.
    )
    num_solutions: int = 2  # The number of different solutions to generate or attempt for the given task.
    num_model_candidates: int = 2  # The number of different model architectures or hyperparameter sets to consider as candidates.
    max_retry: int = (
        10  # The maximum number of times to retry a failed operation.
    )
    max_debug_round: int = 5  # The maximum number of iterations or rounds allowed for the debugging step.
    max_rollback_round: int = 2  # The maximum number of times the system can rollback to a previous state, in case of errors or poor performance.
    inner_loop_round: int = 1  # The number of iterations or rounds to be executed within an inner loop of the system.
    outer_loop_round: int = 1  # The number of iterations or rounds to be executed within the outer loop, which might encompass multiple inner loops.
    ensemble_loop_round: int = 1  # The number of rounds or iterations dedicated to ensembling, combining multiple models or solutions.
    num_top_plans: int = 2  # The number of highest-scoring plans or strategies to select or retain.
    use_data_leakage_checker: bool = False  # Enable (`True`) or disable (`False`) a check for data leakage in the machine learning pipeline.
    use_data_usage_checker: bool = False  # Enable (`True`) or disable (`False`) a check for how data is being used, potentially for compliance or best practices.
    use_skrub_rag: bool = os.environ.get("USE_SKRUB_RAG", "1") not in ("0", "false", "False")  # Enable/disable the skrub documentation RAG tool.
    use_skrub_pipelines: bool = os.environ.get("USE_SKRUB_PIPELINES", "1") not in ("0", "false", "False")  # Enable/disable skrub DataOps pipelines (False = default sklearn/pytorch).


CONFIG = DefaultConfig()
