"""Tests that the notify channel configuration reaches the agent from the CLI/config."""

from unittest.mock import Mock, patch

import pytest

from minisweagent.run.mini import DEFAULT_CONFIG_FILE, main


def _agent_config(**kwargs) -> dict:
    captured = {}

    def fake_get_agent(model, env, config, **extra):
        captured["config"] = config
        return Mock(run=Mock(return_value={}))

    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.get_agent", side_effect=fake_get_agent),
        patch("minisweagent.run.mini.get_model", return_value=Mock()),
        patch("minisweagent.run.mini.get_environment", return_value=Mock()),
    ):
        main(
            task="task",
            yolo=True,
            quiet=True,
            output=None,
            model_name="model",
            model_class=None,
            agent_class=None,
            environment_class=None,
            **kwargs,
        )
    return captured["config"]


@pytest.mark.parametrize(
    ("config_spec", "expected"),
    [
        ("agent.notify_channel=terminal_bell", "terminal_bell"),
        ("run.notify_channel=terminal_bell", "terminal_bell"),
        ("notify_channel=terminal_bell", "terminal_bell"),
        ("agent.notify_channel=none", "none"),
    ],
)
def test_notify_channel_specs_reach_agent_config(config_spec, expected):
    assert _agent_config(config_spec=[str(DEFAULT_CONFIG_FILE), config_spec])["notify_channel"] == expected


def test_notify_channel_cli_flag_reaches_agent_config():
    config = _agent_config(config_spec=[str(DEFAULT_CONFIG_FILE)], notify_channel="terminal_bell")
    assert config["notify_channel"] == "terminal_bell"
