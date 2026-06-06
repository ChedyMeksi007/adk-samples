# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Demonstration of Machine Learning Engineering Agent using Agent Development Kit."""

import json
import os

from google.adk import agents
from google.adk.agents import callback_context as callback_context_module
from google.adk.tools import tool_context as tool_context_module
from google.genai import types

from machine_learning_engineering import prompt
from machine_learning_engineering.shared_libraries import config
from machine_learning_engineering.shared_libraries.config import _use_litellm
from machine_learning_engineering.sub_agents.ensemble import (
    agent as ensemble_agent_module,
)
from machine_learning_engineering.sub_agents.initialization import (
    agent as initialization_agent_module,
)
from machine_learning_engineering.sub_agents.refinement import (
    agent as refinement_agent_module,
)
from machine_learning_engineering.sub_agents.submission import (
    agent as submission_agent_module,
)

def configure_pipeline_mode(
    use_skrub_pipelines: bool,
    use_skrub_rag: bool,
    tool_context: tool_context_module.ToolContext,
) -> dict:
    """Configure whether to use skrub DataOps pipelines and skrub RAG.

    Args:
        use_skrub_pipelines: If True, all agents will produce skrub DataOps
            plans (the deferred `.skb` computation graph: skrub.var ->
            .skb.mark_as_X/mark_as_y -> .skb.apply(...) -> .skb.make_learner).
            If False, agents use default sklearn/pytorch solutions.
        use_skrub_rag: If True, agents can search skrub documentation via RAG.
            If False, the skrub docs search tool is disabled.

    Returns:
        A dict with the current configuration.
    """
    tool_context.state["use_skrub_pipelines"] = use_skrub_pipelines
    tool_context.state["use_skrub_rag"] = use_skrub_rag
    return {
        "use_skrub_pipelines": use_skrub_pipelines,
        "use_skrub_rag": use_skrub_rag,
        "status": "Configuration updated",
    }


def save_state(
    callback_context: callback_context_module.CallbackContext,
) -> types.Content | None:
    """Prints the current state of the callback context."""
    workspace_dir = callback_context.state.get("workspace_dir", "")
    task_name = callback_context.state.get("task_name", "")
    run_cwd = os.path.join(workspace_dir, task_name)
    with open(os.path.join(run_cwd, "final_state.json"), "w") as f:
        json.dump(callback_context.state.to_dict(), f, indent=2)
    return None


mle_pipeline_agent = agents.SequentialAgent(
    name="mle_pipeline_agent",
    sub_agents=[
        initialization_agent_module.initialization_agent,
        refinement_agent_module.refinement_agent,
        ensemble_agent_module.ensemble_agent,
        submission_agent_module.submission_agent,
    ],
    description="Executes a sequence of sub-agents for solving the MLE task.",
    after_agent_callback=save_state,
)

# For ADK tools compatibility, the root agent must be named `root_agent`
def get_global_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the global instruction based on skrub pipeline setting."""
    use_skrub = context.state.get("use_skrub_pipelines", config.CONFIG.use_skrub_pipelines)
    return prompt.SYSTEM_INSTRUCTION if use_skrub else prompt.SYSTEM_INSTRUCTION_DEFAULT


# In Gemini mode, root agent uses ROOT_AGENT_MODEL (e.g. gemini-2.5-flash).
# In LiteLLM mode, root agent uses the same OpenAI-compatible model as sub-agents.
_root_model = config.CONFIG.agent_model if _use_litellm else os.getenv("ROOT_AGENT_MODEL", "gemini-2.5-flash")

root_agent = agents.Agent(
    model=_root_model,
    name="mle_frontdoor_agent",
    instruction=prompt.FRONTDOOR_INSTRUCTION,
    global_instruction=get_global_instruction,
    tools=[configure_pipeline_mode],
    sub_agents=[mle_pipeline_agent],
    generate_content_config=types.GenerateContentConfig(temperature=0.01),
)
