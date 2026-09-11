"""Tests for resuming an agent from a saved trajectory."""

import json
from pathlib import Path

import pytest
import yaml

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel, make_output


@pytest.fixture
def agent_config():
    return yaml.safe_load(Path("src/minisweagent/config/default.yaml").read_text())["agent"]


def _make_interrupted_trajectory(path: Path, config: dict, *, cost: float = 0.5, n_calls: int = 2) -> None:
    """Save a non-terminal trajectory the way an interrupted run leaves it behind."""
    agent = DefaultAgent(DeterministicModel(outputs=[]), LocalEnvironment(), **config)
    agent.extra_template_vars["task"] = "Test task"
    agent.messages = [
        agent.model.format_message(role="system", content="system prompt"),
        agent.model.format_message(role="user", content="Please solve: Test task"),
        agent.model.format_message(role="assistant", content="working", extra={"actions": [{"command": "echo hi"}]}),
        agent.model.format_message(role="user", content="hi"),
    ]
    agent.cost = cost
    agent.n_calls = n_calls
    agent.save(path)


def _completing_agent(config: dict, submission: str = "finished") -> DefaultAgent:
    command = f"echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\necho {submission}"
    return DefaultAgent(
        DeterministicModel(outputs=[make_output("done", [{"command": command}])]),
        LocalEnvironment(),
        **config,
    )


def test_load_restores_state_and_targets_the_same_file(tmp_path, agent_config):
    path = tmp_path / "run.traj.json"
    _make_interrupted_trajectory(path, agent_config)
    agent = DefaultAgent(DeterministicModel(outputs=[]), LocalEnvironment(), **agent_config)

    data = agent.load(path)

    assert data["info"]["task"] == "Test task"
    assert agent.messages == json.loads(path.read_text())["messages"]
    assert agent.cost == 0.5
    assert agent.n_calls == 2
    assert agent.extra_template_vars["task"] == "Test task"
    assert agent.config.output_path == path


def test_run_continues_loaded_trajectory(tmp_path, agent_config):
    path = tmp_path / "run.traj.json"
    _make_interrupted_trajectory(path, agent_config)
    agent = _completing_agent(agent_config)
    agent.load(path)
    n_prefix = len(agent.messages)

    info = agent.run()

    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "finished\n"
    assert agent.n_calls == 3
    assert agent.cost == 1.5
    # The original prefix is kept, only the new assistant + exit messages are appended.
    assert len(agent.messages) == n_prefix + 2
    assert agent.messages[0]["content"] == "system prompt"


def test_load_rejects_a_trajectory_without_messages(tmp_path, agent_config):
    path = tmp_path / "empty.traj.json"
    path.write_text(json.dumps({"info": {}}))
    agent = DefaultAgent(DeterministicModel(outputs=[]), LocalEnvironment(), **agent_config)

    with pytest.raises(ValueError, match="does not contain any messages"):
        agent.load(path)


def test_resume_method_continues_and_writes_back(tmp_path, agent_config):
    path = tmp_path / "run.traj.json"
    _make_interrupted_trajectory(path, agent_config)

    info = _completing_agent(agent_config, submission="again").resume(path)

    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "again\n"
    saved = json.loads(path.read_text())
    assert saved["info"]["exit_status"] == "Submitted"
    assert saved["info"]["model_stats"] == {"instance_cost": 1.5, "api_calls": 3}


def test_resuming_a_finished_trajectory_does_not_call_the_model(tmp_path, agent_config):
    path = tmp_path / "run.traj.json"
    _make_interrupted_trajectory(path, agent_config)
    _completing_agent(agent_config).resume(path)

    agent = DefaultAgent(DeterministicModel(outputs=[]), LocalEnvironment(), **agent_config)
    agent.load(path)
    info = agent.run()

    assert info["exit_status"] == "Submitted"
    assert agent.n_calls == 3  # unchanged: run() returned without querying
