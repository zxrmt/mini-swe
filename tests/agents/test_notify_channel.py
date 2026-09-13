"""Tests for the configurable task-completion notification channel."""

import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import yaml

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.agents.interactive import InteractiveAgentConfig
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel, make_output


@pytest.fixture(autouse=True)
def _no_global_notify_channel(monkeypatch):
    """The real global .env may set the channel; isolate the default-behaviour tests from it."""
    monkeypatch.delenv("MSWEA_NOTIFY_CHANNEL", raising=False)
    monkeypatch.delenv("NOTIFY_CHANNEL", raising=False)


def _config() -> dict:
    return yaml.safe_load(Path("src/minisweagent/config/default.yaml").read_text())["agent"]


def _completing_agent(**overrides) -> DefaultAgent:
    model = DeterministicModel(
        outputs=[make_output("done", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho ok"}])]
    )
    return DefaultAgent(model, LocalEnvironment(), **{**_config(), **overrides})


def _run(agent: DefaultAgent) -> tuple[dict, str]:
    out = io.StringIO()
    with redirect_stdout(out):
        info = agent.run("task")
    return info, out.getvalue()


def test_notify_channel_defaults_to_none_and_stays_silent():
    assert AgentConfig(**_config()).notify_channel == "none"
    info, out = _run(_completing_agent())
    assert info["exit_status"] == "Submitted"
    assert "\a" not in out


def test_notify_channel_terminal_bell_rings_on_completion():
    _, out = _run(_completing_agent(notify_channel="terminal_bell"))
    assert "\a" in out


def test_notify_channel_env_var_rings_on_completion(monkeypatch):
    monkeypatch.setenv("MSWEA_NOTIFY_CHANNEL", "terminal_bell")
    _, out = _run(_completing_agent())
    assert "\a" in out


def test_explicit_channel_overrides_env(monkeypatch):
    monkeypatch.setenv("MSWEA_NOTIFY_CHANNEL", "terminal_bell")
    _, out = _run(_completing_agent(notify_channel="none"))
    assert "\a" not in out


def test_interactive_agent_config_inherits_notify_channel():
    assert issubclass(InteractiveAgentConfig, AgentConfig)
    assert InteractiveAgentConfig(**_config()).notify_channel == "none"
