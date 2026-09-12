"""The builtin prompts must support informational tasks and show the final output.

The default prompts used to be written for code-modification tasks only and told the agent to
finish with a bare `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`, which submits an empty output.
They now explain how to submit a final answer so that questions/analysis tasks show their output.
"""

import re
from pathlib import Path

import pytest
import yaml
from jinja2 import StrictUndefined, Template

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel, make_output

BUILTIN_CONFIGS = ["mini.yaml", "default.yaml", "mini_textbased.yaml"]


def _render_instance_template(config_name: str) -> str:
    config = yaml.safe_load((Path("src/minisweagent/config") / config_name).read_text(encoding="utf-8"))
    return Template(config["agent"]["instance_template"], undefined=StrictUndefined).render(
        task="Analyze the last run for me",
        system="Linux",
        release="6.8",
        version="1",
        machine="x86_64",
    )


@pytest.mark.parametrize("config_name", BUILTIN_CONFIGS)
def test_prompts_cover_informational_tasks_and_ask_for_final_output(config_name):
    rendered = _render_instance_template(config_name).lower()
    assert "information tasks" in rendered
    assert "show the output" in rendered
    assert "final answer" in rendered
    assert "complete_task_and_submit_final_output" in rendered


@pytest.mark.parametrize("config_name", BUILTIN_CONFIGS)
def test_documented_submission_example_shows_the_full_output(config_name):
    r"""Executing the prompt's own example must submit the answer verbatim.

    A plain `printf '...'` example would interpret `%`/`\` and break on `'`, which silently
    truncated long answers (e.g. `>74%` became `>740n`).
    """
    rendered = _render_instance_template(config_name)
    blocks = re.findall(r"```(?:mswea_bash_command|bash)\n(.*?)\n```", rendered, re.DOTALL)
    example = next(block for block in blocks if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in block)
    answer = "100% done, it's a \\c literal & <tag>"
    command = example.replace("Your final answer here", answer)

    config = yaml.safe_load((Path("src/minisweagent/config") / config_name).read_text(encoding="utf-8"))["agent"]
    agent = DefaultAgent(
        DeterministicModel(outputs=[make_output("Submitting", [{"command": command}])]),
        LocalEnvironment(),
        **config,
    )

    info = agent.run("Analyze the last run for me")
    assert info["exit_status"] == "Submitted"
    assert info["submission"].strip() == answer
