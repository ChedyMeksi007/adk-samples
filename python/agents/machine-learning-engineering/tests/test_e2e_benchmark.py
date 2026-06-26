"""End-to-end benchmark: run MLE agent on each task, submit to Kaggle, collect scores.

The MLE-STAR agent reads its task from the TASK env var at import time, so each
task must run in a **separate subprocess** to get a fresh config.  This test
orchestrates that: it spawns one child process per task, collects results, and
prints a summary table.

Usage:
    # Run all tasks (Gemini backend, current .env settings)
    pytest tests/test_e2e_benchmark.py -v -s

    # Run a single task
    pytest tests/test_e2e_benchmark.py -v -s -k "spooky"

    # Run with LiteLLM backend
    USE_LITELLM=1 pytest tests/test_e2e_benchmark.py -v -s

    # Config check only (no agent run)
    pytest tests/test_e2e_benchmark.py -v -s -k "config"
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import dotenv
import pytest
from kaggle.api.kaggle_api_extended import KaggleApi

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = PROJECT_ROOT / "machine_learning_engineering" / "tasks"
WORKSPACE_DIR = PROJECT_ROOT / "machine_learning_engineering" / "workspace"
RESULTS_DIR = PROJECT_ROOT / "experiments"

# Task name -> (kaggle competition slug, lower_is_better)
TASK_KAGGLE_MAP: dict[str, tuple[str, bool]] = {
    "spooky-author-identification": ("spooky-author-identification", True),
    "jigsaw-toxic-comment-classification-challenge": ("jigsaw-toxic-comment-classification-challenge", False),
    "nomad2018-predict-transparent-conductors": ("nomad2018-predict-transparent-conductors", True),
    "random-acts-of-pizza": ("random-acts-of-pizza", False),
    "aerial-cactus-identification": ("aerial-cactus-identification", False),
    "leaf-classification": ("leaf-classification", True),
    "denoising-dirty-documents": ("denoising-dirty-documents", True),
    "detecting-insults-in-social-commentary": ("detecting-insults-in-social-commentary", False),
    "text-normalization-challenge-english-language": ("text-normalization-challenge-english-language", False),
    "text-normalization-challenge-russian-language": ("text-normalization-challenge-russian-language", False),
}

MEDAL_THRESHOLDS = {
    "gold": 0.10,
    "silver": 0.25,
    "bronze": 0.40,
}


# ---------------------------------------------------------------------------
# Kaggle helpers
# ---------------------------------------------------------------------------
def _get_kaggle_api() -> KaggleApi:
    api = KaggleApi()
    api.authenticate()
    return api


def _kaggle_submit(api: KaggleApi, competition: str, path: str, message: str) -> str:
    resp = api.competition_submit(
        file_name=path, message=message, competition=competition, quiet=True,
    )
    msg = resp.message or "submitted"
    print(f"  [kaggle submit] {msg}", flush=True)
    return msg


def _kaggle_poll_score(
    api: KaggleApi, competition: str, wait_secs: int = 20, max_retries: int = 10,
) -> dict:
    for attempt in range(max_retries):
        time.sleep(wait_secs)
        subs = api.competition_submissions(competition)
        if not subs:
            print(f"  [score] attempt {attempt + 1}/{max_retries}: no submissions")
            continue

        latest = subs[0]
        status = str(latest.status or "").lower()
        public_score = latest.public_score or ""

        if "complete" in status and public_score:
            return {
                "public_score": float(public_score),
                "private_score": float(latest.private_score) if latest.private_score else None,
                "status": "complete",
            }
        elif "error" in status:
            return {"public_score": None, "status": "error", "error": latest.error_description or ""}
        else:
            print(f"  [score] attempt {attempt + 1}/{max_retries}: {status}")

    return {"public_score": None, "status": "timeout"}


def _kaggle_get_medal(
    api: KaggleApi, competition: str, score: float, lower_is_better: bool,
) -> dict:
    entries = api.competition_leaderboard_view(competition, page_size=200) or []
    scores = []
    for e in entries:
        try:
            scores.append(float(e.score))
        except (ValueError, TypeError, AttributeError):
            continue

    if not scores:
        return {"rank": None, "total_teams": None, "percentile": None, "medal": None}

    total = len(scores)
    rank = (sum(1 for s in scores if s < score) + 1) if lower_is_better else (sum(1 for s in scores if s > score) + 1)
    percentile = min(rank / total, 1.0)

    medal = None
    for name, thresh in sorted(MEDAL_THRESHOLDS.items(), key=lambda x: x[1]):
        if percentile <= thresh:
            medal = name
            break

    return {"rank": rank, "total_teams": total, "percentile": round(percentile, 4), "medal": medal}


def _find_submission_csv(task_name: str) -> str | None:
    workspace = WORKSPACE_DIR / task_name
    if not workspace.exists():
        return None
    candidates = sorted(workspace.rglob("submission.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0]) if candidates else None


def _current_config_label() -> tuple[str, str, str]:
    backend = "litellm" if os.environ.get("USE_LITELLM", "0") not in ("0", "false", "False") else "gemini"
    pipeline = "skrub" if os.environ.get("USE_SKRUB_PIPELINES", "0") not in ("0", "false", "False") else "default"
    model = os.environ.get("BASE_MODEL", "llama-3.3-70b-instruct") if backend == "litellm" else os.environ.get("ROOT_AGENT_MODEL", "gemini-2.5-flash")
    return backend, pipeline, model


def _run_results_dir() -> Path:
    """Return a timestamped, config-specific results directory."""
    backend, pipeline, model = _current_config_label()
    model_safe = model.replace("/", "-").replace(" ", "_")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return RESULTS_DIR / f"{backend}_{pipeline}_{model_safe}_{ts}"


# ---------------------------------------------------------------------------
# Parse final_state.json for detailed metrics
# ---------------------------------------------------------------------------
def _parse_final_state(task_name: str) -> dict:
    """Extract metrics from the agent's final_state.json in the workspace.

    Returns a dict with:
      - validation_scores: list of (stage, score) tuples
      - code_exec_success: int count of successful code executions
      - code_exec_fail: int count of failed code executions
      - debug_cycles: int estimated debug/retry cycles
      - final_submission_code: str or None
      - ensemble_scores: list of scores from ensemble iterations
    """
    workspace = WORKSPACE_DIR / task_name
    state_path = workspace / "final_state.json"
    metrics: dict = {
        "validation_scores": [],
        "code_exec_success": 0,
        "code_exec_fail": 0,
        "debug_cycles": 0,
        "final_submission_code": None,
        "ensemble_scores": [],
    }

    if not state_path.exists():
        return metrics

    try:
        state = json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError):
        return metrics

    # Collect all code execution results and scores
    exec_result_pattern = re.compile(r"(.*_exec_result.*)")
    bug_pattern = re.compile(r"^bug_\w+_\d+$")

    for key, value in state.items():
        # Code execution results (contain returncode, score, stderr etc.)
        if exec_result_pattern.match(key) and isinstance(value, dict):
            returncode = value.get("returncode")
            if returncode == 0:
                metrics["code_exec_success"] += 1
            elif returncode is not None:
                metrics["code_exec_fail"] += 1

            score = value.get("score")
            if score is not None and score not in (1e9, 0):
                metrics["validation_scores"].append((key, score))

            # Ensemble scores
            if key.startswith("ensemble_code_exec_result_"):
                if score is not None and score not in (1e9, 0):
                    metrics["ensemble_scores"].append(score)

        # Count bug/debug entries as debug cycles
        if bug_pattern.match(key) and value:
            metrics["debug_cycles"] += 1

    # Find the submission code (look for the final solution key)
    submission_code = state.get("submission_code")
    if not submission_code:
        # Try to find the latest train code as fallback
        for task_id in range(1, 10):
            for step in range(10, -1, -1):
                code = state.get(f"train_code_{step}_{task_id}")
                if code:
                    submission_code = code
                    break
            if submission_code:
                break

    metrics["final_submission_code"] = submission_code
    return metrics


def _save_result(result: dict, run_dir: Path | None = None) -> None:
    target_dir = run_dir or RESULTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    backend, pipeline, model = _current_config_label()
    model_safe = model.replace("/", "-").replace(" ", "_")
    results_file = target_dir / f"benchmark_{backend}_{pipeline}_{model_safe}.jsonl"
    with open(results_file, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"  [saved] {results_file}")


def _save_submission_code(task_name: str, code: str | None, run_dir: Path) -> None:
    """Save the final submission code to the run results directory."""
    if not code:
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    code_file = run_dir / f"{task_name}_final_code.py"
    code_file.write_text(code)
    print(f"  [saved] final code -> {code_file}")


# ---------------------------------------------------------------------------
# Worker script — runs ONE task in a fresh process
# ---------------------------------------------------------------------------
_WORKER_SCRIPT = r'''
"""Worker: run MLE agent on a single task. Invoked as a subprocess."""
import asyncio
import os
import sys
import time

import dotenv
dotenv.load_dotenv()

# TASK env var is already set by the parent process before spawning us.
from google.adk.agents.run_config import RunConfig
from google.adk.runners import InMemoryRunner
from google.genai import types
from machine_learning_engineering.agent import root_agent

async def main():
    runner = InMemoryRunner(agent=root_agent, app_name="mle-benchmark")
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id="benchmark_user"
    )
    task_name = os.environ["TASK"]
    content = types.Content(
        parts=[types.Part(text=f"Execute the {task_name} task.")],
        role="user",
    )
    run_config = RunConfig(max_llm_calls=2000)

    start = time.time()
    async for _ in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=content,
        run_config=run_config,
    ):
        pass
    elapsed = time.time() - start
    print(f"AGENT_ELAPSED={elapsed:.1f}")

asyncio.run(main())
'''


def _run_agent_subprocess(task_name: str, timeout: int = 1800) -> float:
    """Spawn a subprocess that runs the MLE agent for a single task.

    Returns the elapsed time in seconds, or raises on failure.
    """
    env = os.environ.copy()
    env["TASK"] = task_name

    result = subprocess.run(
        [sys.executable, "-c", _WORKER_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(PROJECT_ROOT),
        timeout=timeout,
    )

    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print(f"  STDERR (last 2000 chars):\n{result.stderr[-2000:]}")
        raise RuntimeError(f"Agent subprocess failed for {task_name} (rc={result.returncode})")

    # Parse elapsed time from stdout
    for line in result.stdout.splitlines():
        if line.startswith("AGENT_ELAPSED="):
            return float(line.split("=")[1])
    return 0.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def load_env():
    dotenv.load_dotenv(PROJECT_ROOT / ".env")


@pytest.fixture(scope="session")
def kaggle_api():
    return _get_kaggle_api()


@pytest.fixture(scope="session")
def run_dir():
    """A single timestamped results directory shared across all tasks in this run."""
    return _run_results_dir()


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
def test_config_check(kaggle_api):
    """Verify the full stack is configured correctly before running tasks."""
    # Import here so .env is loaded first
    from machine_learning_engineering.shared_libraries import config

    errors = []

    # 1. Kaggle
    print("\n[config] Kaggle auth...")
    try:
        username = kaggle_api.config_values.get("username", None)
        print(f"  Kaggle user: {username}")
        assert username, "Kaggle username is empty"
    except Exception as e:
        errors.append(f"Kaggle auth failed: {e}")

    print("[config] Kaggle API connectivity...")
    try:
        subs = kaggle_api.competition_submissions("spooky-author-identification")
        print(f"  API reachable (existing submissions: {len(subs)})")
    except Exception as e:
        errors.append(f"Kaggle API unreachable: {e}")

    # 2. Backend
    print("[config] Agent backend...")
    backend, pipeline, model = _current_config_label()
    print(f"  USE_LITELLM:  {config.CONFIG.use_litellm}")
    print(f"  Backend:      {backend}")
    print(f"  Pipeline:     {pipeline}")
    print(f"  Model:        {model}")
    print(f"  agent_model:  {config.CONFIG.agent_model} ({type(config.CONFIG.agent_model).__name__})")

    if config.CONFIG.use_litellm:
        api_base = os.environ.get("OPENAI_API_BASE", "")
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_base:
            errors.append("USE_LITELLM=1 but API_ENDPOINT is empty")
        if not api_key:
            errors.append("USE_LITELLM=1 but API_KEY is empty")
    else:
        google_key = os.environ.get("GOOGLE_API_KEY", "")
        if not google_key:
            errors.append("USE_LITELLM=0 (Gemini) but GOOGLE_API_KEY is empty")
        print(f"  GOOGLE_API_KEY: {'set' if google_key else 'NOT SET'}")

    # 3. Skrub
    print(f"  use_skrub_pipelines: {config.CONFIG.use_skrub_pipelines}")
    print(f"  use_skrub_rag:       {config.CONFIG.use_skrub_rag}")

    # 4. google_search (Gemini mode only)
    if not config.CONFIG.use_litellm:
        print("[config] google_search tool...")
        try:
            from google.adk.tools.google_search_tool import google_search  # noqa: F401
            print("  google_search: available")
        except ImportError as e:
            errors.append(f"Gemini mode but google_search unavailable: {e}")

    # 5. Task data
    print("[config] Task data...")
    for task_name in TASK_KAGGLE_MAP:
        task_dir = TASKS_DIR / task_name
        has_desc = (task_dir / "task_description.txt").exists()
        has_data = any(task_dir.glob("train.*")) or any(task_dir.glob("*train*"))
        ok = has_desc and has_data
        if not ok:
            errors.append(f"Task data missing: {task_name}")
        print(f"  {task_name}: {'OK' if ok else 'MISSING'}")

    # 6. Agent starts (quick import check in subprocess)
    print("[config] Agent import check...")
    r = subprocess.run(
        [sys.executable, "-c", "import dotenv; dotenv.load_dotenv(); from machine_learning_engineering.agent import root_agent; print('OK')"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        env=os.environ.copy(), timeout=30,
    )
    if "OK" in r.stdout:
        print("  Agent imports: OK")
    else:
        errors.append(f"Agent import failed: {r.stderr[-500:]}")

    print()
    if errors:
        for err in errors:
            print(f"  ERROR: {err}")
        pytest.fail(f"Config check failed with {len(errors)} error(s):\n" + "\n".join(errors))
    else:
        print("  All checks passed.")


# ---------------------------------------------------------------------------
# Parametrized e2e test
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("task_name", list(TASK_KAGGLE_MAP.keys()))
def test_task_e2e(task_name: str, kaggle_api, run_dir):
    """Run the MLE agent on a task (subprocess), submit to Kaggle, check medal."""
    competition, lower_is_better = TASK_KAGGLE_MAP[task_name]
    task_dir = TASKS_DIR / task_name

    if not task_dir.exists() or not (task_dir / "task_description.txt").exists():
        pytest.skip(f"Task data missing for {task_name}")

    backend, pipeline, model = _current_config_label()

    print(f"\n{'='*60}")
    print(f"  TASK:        {task_name}")
    print(f"  COMPETITION: {competition}")
    print(f"  BACKEND:     {backend} | {pipeline} | {model}")
    print(f"  RESULTS DIR: {run_dir}")
    print(f"{'='*60}")

    # ---- 1. Run the agent in a subprocess (fresh TASK env) ----
    existing_submission = _find_submission_csv(task_name)
    if existing_submission:
        print(f"\n[1/4] Reusing existing submission: {existing_submission}")
        elapsed = 0.0
    else:
        print("\n[1/4] Running MLE agent (subprocess)...")
        start_time = time.time()
        try:
            elapsed = _run_agent_subprocess(task_name, timeout=int(os.environ.get("TASK_TIMEOUT", "7200")))
        except Exception as e:
            elapsed = time.time() - start_time
            fallback = _find_submission_csv(task_name)
            if fallback:
                print(f"  Agent errored but submission exists: {fallback}")
            else:
                agent_metrics = _parse_final_state(task_name)
                result = {
                    "task": task_name, "competition": competition,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "agent_time_secs": round(elapsed, 1),
                    "backend": backend, "pipeline": pipeline, "model": model,
                    "submission": False, "kaggle_score": None, "medal": None,
                    "agent_error": str(e),
                    "code_exec_success": agent_metrics["code_exec_success"],
                    "code_exec_fail": agent_metrics["code_exec_fail"],
                    "debug_cycles": agent_metrics["debug_cycles"],
                    "validation_scores": agent_metrics["validation_scores"],
                    "ensemble_scores": agent_metrics["ensemble_scores"],
                }
                _save_result(result, run_dir)
                _save_submission_code(task_name, agent_metrics["final_submission_code"], run_dir)
                pytest.fail(f"Agent failed for {task_name}: {e}")

        print(f"  Agent finished in {elapsed:.1f}s")

    # ---- 2. Parse agent metrics from final_state.json ----
    print("\n  Parsing agent metrics from final_state.json...")
    agent_metrics = _parse_final_state(task_name)
    print(f"    Code executions: {agent_metrics['code_exec_success']} ok / {agent_metrics['code_exec_fail']} failed")
    print(f"    Debug cycles:    {agent_metrics['debug_cycles']}")
    if agent_metrics["validation_scores"]:
        print(f"    Validation scores:")
        for stage, score in agent_metrics["validation_scores"]:
            print(f"      {stage}: {score:.5f}")
    if agent_metrics["ensemble_scores"]:
        print(f"    Ensemble scores: {[f'{s:.5f}' for s in agent_metrics['ensemble_scores']]}")

    # Save final submission code
    _save_submission_code(task_name, agent_metrics["final_submission_code"], run_dir)

    # ---- 3. Find submission file ----
    print("\n[2/4] Looking for submission.csv...")
    submission_path = _find_submission_csv(task_name)

    result = {
        "task": task_name, "competition": competition,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent_time_secs": round(elapsed, 1),
        "backend": backend, "pipeline": pipeline, "model": model,
        "code_exec_success": agent_metrics["code_exec_success"],
        "code_exec_fail": agent_metrics["code_exec_fail"],
        "debug_cycles": agent_metrics["debug_cycles"],
        "validation_scores": agent_metrics["validation_scores"],
        "ensemble_scores": agent_metrics["ensemble_scores"],
    }

    if not submission_path:
        result.update(submission=False, kaggle_score=None, medal=None)
        _save_result(result, run_dir)
        pytest.fail(f"No submission.csv produced for {task_name}")

    print(f"  Found: {submission_path}")
    result["submission"] = True

    # ---- 4. Submit to Kaggle ----
    print("\n[3/4] Submitting to Kaggle...")
    submit_msg = f"MLE-agent | {backend} | {pipeline} | {model}"
    try:
        _kaggle_submit(kaggle_api, competition, submission_path, submit_msg)
    except Exception as e:
        result.update(kaggle_score=None, kaggle_error=str(e), medal=None)
        _save_result(result, run_dir)
        pytest.fail(f"Kaggle submission failed: {e}")

    # ---- 5. Get score and medal ----
    print("\n[4/4] Waiting for Kaggle score...")
    score_info = _kaggle_poll_score(kaggle_api, competition)
    result["kaggle_status"] = score_info["status"]

    if score_info["public_score"] is not None:
        result["kaggle_score"] = score_info["public_score"]
        result["private_score"] = score_info.get("private_score")
        print(f"  Public score: {score_info['public_score']}")

        lb = _kaggle_get_medal(kaggle_api, competition, score_info["public_score"], lower_is_better)
        result.update(rank=lb["rank"], total_teams=lb["total_teams"], percentile=lb["percentile"], medal=lb["medal"])

        if lb["rank"]:
            print(f"  Rank: {lb['rank']}/{lb['total_teams']} ({lb['percentile']:.1%})")
        print(f"  Medal: {(lb['medal'] or 'none').upper()}")
    else:
        result.update(kaggle_score=None, medal=None)
        print(f"  Score not available: {score_info['status']}")

    _save_result(result, run_dir)

    print(f"\n  {task_name}: score={result.get('kaggle_score', 'N/A')} "
          f"medal={(result.get('medal') or 'none').upper()} time={result['agent_time_secs']}s")


# ---------------------------------------------------------------------------
# Summary (runs after all tests)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def print_summary():
    yield
    if not RESULTS_DIR.exists():
        return

    print(f"\n\n{'='*80}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*80}")

    for results_subdir in sorted(RESULTS_DIR.iterdir()):
        jsonl_files = list(results_subdir.glob("benchmark_*.jsonl")) if results_subdir.is_dir() else []
        # Also handle legacy flat .jsonl files
        if results_subdir.is_file() and results_subdir.suffix == ".jsonl":
            jsonl_files = [results_subdir]

        for f in jsonl_files:
            print(f"\n  [{f.parent.name}/{f.name}]" if f.parent != RESULTS_DIR else f"\n  [{f.name}]")
            rows = []
            for line in f.read_text().strip().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

            if not rows:
                continue

            # Per-task table
            print(f"  {'Task':<40} {'Score':<12} {'Rank':<12} {'Medal':<8} {'Time':<8} {'Exec OK':<8} {'Exec Fail':<10} {'Debug':<6}")
            print(f"  {'-'*40} {'-'*12} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*6}")

            total_tasks = len(rows)
            medal_counts = {"gold": 0, "silver": 0, "bronze": 0}
            total_exec_ok = 0
            total_exec_fail = 0
            total_debug = 0
            total_time = 0.0
            submissions = 0

            for r in rows:
                task = r.get("task", "?")[:39]
                score = f"{r['kaggle_score']:.5f}" if r.get("kaggle_score") is not None else "N/A"
                rank = f"{r.get('rank', '?')}/{r.get('total_teams', '?')}" if r.get("rank") else "N/A"
                medal = (r.get("medal") or "none").upper()
                t = f"{(r.get('agent_time_secs') or 0):.0f}s"
                exec_ok = r.get("code_exec_success", "-")
                exec_fail = r.get("code_exec_fail", "-")
                debug = r.get("debug_cycles", "-")
                print(f"  {task:<40} {score:<12} {rank:<12} {medal:<8} {t:<8} {str(exec_ok):<8} {str(exec_fail):<10} {str(debug):<6}")

                # Aggregate
                m = r.get("medal")
                if m in medal_counts:
                    medal_counts[m] += 1
                if r.get("submission"):
                    submissions += 1
                total_exec_ok += r.get("code_exec_success") or 0
                total_exec_fail += r.get("code_exec_fail") or 0
                total_debug += r.get("debug_cycles") or 0
                total_time += r.get("agent_time_secs") or 0

            # Aggregate summary
            any_medal = medal_counts["gold"] + medal_counts["silver"] + medal_counts["bronze"]
            print(f"\n  --- Aggregate ---")
            print(f"  Tasks: {total_tasks} | Submissions: {submissions}")
            print(f"  Medal rate (any):    {any_medal}/{total_tasks} ({any_medal/total_tasks*100:.1f}%)" if total_tasks else "")
            print(f"  Medal rate (gold):   {medal_counts['gold']}/{total_tasks} ({medal_counts['gold']/total_tasks*100:.1f}%)" if total_tasks else "")
            print(f"  Medal rate (silver): {medal_counts['silver']}/{total_tasks} ({medal_counts['silver']/total_tasks*100:.1f}%)" if total_tasks else "")
            print(f"  Medal rate (bronze): {medal_counts['bronze']}/{total_tasks} ({medal_counts['bronze']/total_tasks*100:.1f}%)" if total_tasks else "")
            total_exec = total_exec_ok + total_exec_fail
            if total_exec > 0:
                print(f"  Code exec success rate: {total_exec_ok}/{total_exec} ({total_exec_ok/total_exec*100:.1f}%)")
            print(f"  Total debug cycles: {total_debug}")
            print(f"  Total agent time:   {total_time:.0f}s ({total_time/60:.1f}min)")

    print(f"\n{'='*80}")
