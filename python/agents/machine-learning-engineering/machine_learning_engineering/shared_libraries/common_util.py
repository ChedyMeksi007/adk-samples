"""Common utility functions."""

import os
import random
import re
import shutil

import numpy as np
import torch
from google.adk.models import llm_response


def get_text_from_response(
    response: llm_response.LlmResponse,
) -> str:
    """Extracts text from response.

    Skips reasoning/"thought" parts (e.g. gpt-oss exposes its chain-of-thought
    via a separate reasoning field that ADK converts into thought=True parts).
    Concatenating those would pollute generated code with prose.
    """
    final_text = ""
    if response.content and response.content.parts:
        for part in response.content.parts:
            if getattr(part, "thought", False):
                continue
            if getattr(part, "text", None) is not None:
                final_text += part.text
    return final_text


def extract_code(text: str) -> str:
    """Extracts Python code from an LLM response, robust to surrounding prose.

    Models like gpt-oss often wrap code in explanatory prose plus a fenced
    block ("Here is the script:\n```python\n...\n```"). A naive
    .replace("```python", "") leaves the prose prefix, which then fails to
    execute. This returns only the fenced code block(s) when present, else the
    text with stray fence markers stripped.
    """
    if not text:
        return ""
    blocks = re.findall(r"```(?:python|py)?\s*\n?(.*?)```", text, re.DOTALL)
    if blocks:
        # pick the largest block (the real program), not echoed snippets
        code = max(blocks, key=len).strip()
    else:
        code = text.replace("```python", "").replace("```", "").strip()
    # normalise "fancy" unicode some models emit (smart quotes, dashes) that
    # break compilation when they land in code positions
    for bad, good in (
        ("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"'),
        ("‑", "-"), ("–", "-"), ("—", "-"), (" ", " "),
    ):
        code = code.replace(bad, good)
    return code


def set_random_seed(seed: int) -> None:
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def copy_file(source_file_path: str, destination_dir: str) -> None:
    """Copies a file to the specified directory."""
    if not os.path.isdir(destination_dir):
        os.makedirs(destination_dir, exist_ok=True)
    shutil.copy2(source_file_path, destination_dir)
