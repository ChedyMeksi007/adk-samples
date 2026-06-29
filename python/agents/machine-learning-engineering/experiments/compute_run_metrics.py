#!/usr/bin/env python3
"""Extract per-task agent-internal metrics from a finished run's workspaces.

Metrics per task (derived from workspace/<task>/final_state.json + generated files):
  - exec_success_rate : fraction of code executions that returned rc==0
  - exec_total / exec_success : raw counts behind the rate
  - debug_cycles : number of debug-iteration code files (e.g. train0_1.py, *_improveN.py)
  - code_files_total : all generated .py files (proxy for total codegen iterations)
  - validation_score : best internal "Final Validation Performance" the agent reached
  - validation_score_final : score of the final ensemble/submission solution

Usage:
  python compute_run_metrics.py <workspace_root> <out_json> <task1> <task2> ...
"""
from __future__ import annotations
import json
import os
import re
import sys


def _is_exec_result(key: str) -> bool:
    return "exec_result" in key


def _task_metrics(workspace_root: str, task: str) -> dict:
    task_dir = os.path.join(workspace_root, task)
    state_path = os.path.join(task_dir, "final_state.json")
    m = {
        "final_state_found": False,
        "exec_total": 0,
        "exec_success": 0,
        "exec_success_rate": None,
        "debug_cycles": 0,
        "code_files_total": 0,
        "validation_score": None,
        "validation_score_final": None,
        "lower_is_better": None,
    }

    # ---- from final_state.json: exec success rate + validation scores ----
    # debug_cycles is counted here (from bug_* state keys) so it matches the
    # definition used by tests/test_e2e_benchmark.py::_parse_final_state, keeping
    # the per-run metrics.json comparable with the test's inline JSONL output.
    bug_pat = re.compile(r"^bug_\w+_\d+$")
    if os.path.exists(state_path):
        m["final_state_found"] = True
        try:
            with open(state_path) as f:
                state = json.load(f)
        except Exception as e:
            state = {}
            m["state_error"] = str(e)

        lower = bool(state.get("lower", True))
        m["lower_is_better"] = lower

        scores = []
        bug_cycles = 0
        for key, val in state.items():
            if bug_pat.match(key) and val:
                bug_cycles += 1
            if not _is_exec_result(key) or not isinstance(val, dict):
                continue
            if "returncode" not in val:
                continue
            m["exec_total"] += 1
            if val.get("returncode") == 0:
                m["exec_success"] += 1
            sc = val.get("score")
            # skip sentinel failure scores (1e9 / 0 used as placeholders)
            if isinstance(sc, (int, float)) and sc not in (1e9, 0):
                scores.append(sc)
            if key == "submission_code_exec_result" or key.startswith("ensemble_code_exec_result"):
                if isinstance(sc, (int, float)) and sc not in (1e9, 0):
                    m["validation_score_final"] = sc
        if m["exec_total"]:
            m["exec_success_rate"] = round(m["exec_success"] / m["exec_total"], 4)
        if scores:
            m["validation_score"] = min(scores) if lower else max(scores)
        m["debug_cycles"] = bug_cycles

    # ---- from generated files: debug cycles + total codegen ----
    py_files = []
    for root, _, files in os.walk(task_dir):
        if os.sep + "input" in root:  # skip copied input data
            continue
        for fn in files:
            if fn.endswith(".py"):
                py_files.append(fn)
    m["code_files_total"] = len(py_files)
    # Refinement files end in improveN.py. (debug_cycles is taken from the bug_*
    # state keys above, matching the test's _parse_final_state definition.)
    refine_pat = re.compile(r".*improve\d+\.py$")
    m["refine_steps"] = sum(1 for fn in py_files if refine_pat.match(fn))
    return m


def main():
    workspace_root = sys.argv[1]
    out_json = sys.argv[2]
    tasks = sys.argv[3:]

    out = {"workspace_root": workspace_root, "tasks": {}}
    agg_total = agg_success = 0
    for task in tasks:
        tm = _task_metrics(workspace_root, task)
        out["tasks"][task] = tm
        agg_total += tm["exec_total"]
        agg_success += tm["exec_success"]
    out["overall_exec_total"] = agg_total
    out["overall_exec_success"] = agg_success
    out["overall_exec_success_rate"] = round(agg_success / agg_total, 4) if agg_total else None

    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)

    # human-readable summary to stdout
    print(f"  [metrics] {out_json}")
    for task, tm in out["tasks"].items():
        print(f"    {task[:40]:<40} exec={tm['exec_success']}/{tm['exec_total']} "
              f"({tm['exec_success_rate']}) debug={tm['debug_cycles']} "
              f"val={tm['validation_score']} final={tm['validation_score_final']}")
    print(f"    OVERALL exec success rate: {out['overall_exec_success_rate']}")


if __name__ == "__main__":
    main()
