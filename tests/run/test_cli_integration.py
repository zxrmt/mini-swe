import json
import os
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from minisweagent.run.mini import DEFAULT_CONFIG_FILE, app, main


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def _render_panel(panel) -> str:
    """Render a rich renderable to plain text (no ANSI) for assertions."""
    import io

    from rich.console import Console as RichConsole

    console = RichConsole(file=io.StringIO(), width=200, force_terminal=False)
    console.print(panel)
    return console.file.getvalue()


def test_import_mini_does_not_load_prompt_toolkit():
    """The interactive prompt dependency should only be imported when prompting is needed."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import minisweagent.run.mini; "
            'print(any(name == "prompt_toolkit" or name.startswith("prompt_toolkit.") for name in sys.modules))',
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert result.stdout.splitlines()[-1] == "False"


def test_configure_if_first_time_called():
    """Test that configure_if_first_time is called when running mini main."""
    with (
        patch("minisweagent.run.mini.configure_if_first_time") as mock_configure,
        patch("minisweagent.run.mini.get_agent") as mock_get_agent,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_environment") as mock_get_env,
        patch("minisweagent.run.mini.get_config_from_spec") as mock_get_config,
    ):
        # Setup mocks
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_environment = Mock()
        mock_get_env.return_value = mock_environment
        mock_get_config.return_value = {"agent": {"system_template": "test"}, "env": {}, "model": {}}

        # Setup mock agent instance
        mock_agent = Mock()
        mock_agent.run.return_value = {"exit_status": "Success", "submission": "Result"}
        mock_get_agent.return_value = mock_agent

        # Call main function
        main(
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            model_name="test-model",
            task="Test task",
            yolo=False,
            output=None,
            model_class=None,
            agent_class=None,
            environment_class=None,
        )

        # Verify configure_if_first_time was called
        mock_configure.assert_called_once()


def test_mini_command_calls_run_interactive():
    """Test that mini command creates agent via get_agent."""
    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.get_agent") as mock_get_agent,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_environment") as mock_get_env,
        patch("minisweagent.run.mini.get_config_from_spec") as mock_get_config,
    ):
        # Setup mocks
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_environment = Mock()
        mock_get_env.return_value = mock_environment
        mock_get_config.return_value = {"agent": {"system_template": "test", "mode": "confirm"}, "env": {}, "model": {}}

        # Setup mock agent instance
        mock_agent = Mock()
        mock_agent.run.return_value = {"exit_status": "Success", "submission": "Result"}
        mock_get_agent.return_value = mock_agent

        # Call main function with task provided (so prompt is not called)
        main(
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            model_name="test-model",
            task="Test task",
            yolo=False,
            output=None,
            model_class=None,
            agent_class=None,
            environment_class=None,
        )

        # Verify get_agent was called
        mock_get_agent.assert_called_once()
        args, kwargs = mock_get_agent.call_args
        assert args[0] == mock_model  # model
        assert args[1] == mock_environment  # env
        # Verify agent.run was called with the task
        mock_agent.run.assert_called_once_with("Test task")


def test_mini_configures_a_conversation_directory():
    """`mini` points the interactive agent at a directory of conversations for `/resume`."""
    from minisweagent import global_config_dir
    from minisweagent.run.mini import DEFAULT_CONVERSATIONS_DIR

    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.get_agent") as mock_get_agent,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_environment") as mock_get_env,
        patch("minisweagent.run.mini.get_config_from_spec", return_value={"agent": {}, "model": {}}),
    ):
        mock_get_model.return_value = Mock()
        mock_get_env.return_value = Mock()
        mock_get_agent.return_value = Mock(run=Mock(return_value={}))

        main(
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            model_name="test-model",
            task="Test task",
            yolo=False,
            output=None,
            model_class=None,
            agent_class=None,
            environment_class=None,
        )

    args, _ = mock_get_agent.call_args
    assert args[2]["conversation_dir"] == DEFAULT_CONVERSATIONS_DIR
    assert DEFAULT_CONVERSATIONS_DIR == global_config_dir / "conversations"


def test_mini_calls_prompt_when_no_task_provided():
    """Test that mini calls prompt when no task is provided."""
    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini._multiline_prompt") as mock_prompt,
        patch("minisweagent.run.mini.get_agent") as mock_get_agent,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_environment") as mock_get_env,
        patch("minisweagent.run.mini.get_config_from_spec") as mock_get_config,
    ):
        # Setup mocks
        mock_prompt.return_value = "User provided task"
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_environment = Mock()
        mock_get_env.return_value = mock_environment
        mock_get_config.return_value = {"agent": {"system_template": "test", "mode": "confirm"}, "env": {}, "model": {}}

        # Setup mock agent instance
        mock_agent = Mock()
        mock_agent.run.return_value = {"exit_status": "Success", "submission": "Result"}
        mock_get_agent.return_value = mock_agent

        # Call main function without task
        main(
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            model_name="test-model",
            task=None,  # No task provided
            yolo=False,
            output=None,
            model_class=None,
            agent_class=None,
            environment_class=None,
        )

        # Verify prompt was called
        mock_prompt.assert_called_once()

        # Verify get_agent was called
        mock_get_agent.assert_called_once()
        # Verify agent.run was called with the task from prompt
        mock_agent.run.assert_called_once_with("User provided task")


def test_model_not_loaded_before_task_prompt():
    """Startup must not pay the model-load cost before asking the user for a task."""
    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini._multiline_prompt") as mock_prompt,
        patch("minisweagent.run.mini.get_agent") as mock_get_agent,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_environment") as mock_get_env,
        patch("minisweagent.run.mini.get_config_from_spec") as mock_get_config,
    ):
        mock_get_model.return_value = Mock()
        mock_get_env.return_value = Mock()
        mock_get_agent.return_value = Mock()
        mock_get_config.return_value = {"agent": {}, "env": {}, "model": {}}

        prompt_calls = []

        def record_prompt():
            prompt_calls.append(mock_get_model.called)
            return "User provided task"

        mock_prompt.side_effect = record_prompt

        main(
            config_spec=[DEFAULT_CONFIG_FILE],
            model_name="test-model",
            task=None,
            yolo=False,
            quiet=False,
            output=None,
            exit_immediately=False,
            model_class=None,
            agent_class=None,
            environment_class=None,
        )

        assert prompt_calls == [False]


def test_mini_with_explicit_model():
    """Test that mini works with explicitly provided model."""
    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.get_agent") as mock_get_agent,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_environment") as mock_get_env,
        patch("minisweagent.run.mini.get_config_from_spec") as mock_get_config,
    ):
        # Setup mocks
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_environment = Mock()
        mock_get_env.return_value = mock_environment
        mock_get_config.return_value = {
            "agent": {"system_template": "test", "mode": "yolo"},
            "env": {},
            "model": {"default_config": "test"},
        }

        # Setup mock agent instance
        mock_agent = Mock()
        mock_agent.run.return_value = {"exit_status": "Success", "submission": "Result"}
        mock_get_agent.return_value = mock_agent

        # Call main function with explicit model
        main(
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            model_name="gpt-4",
            task="Test task with explicit model",
            yolo=True,
            output=None,
            model_class=None,
            agent_class=None,
            environment_class=None,
        )

        # Verify get_model was called (model name is merged into config)
        mock_get_model.assert_called_once()

        # Verify get_agent was called
        mock_get_agent.assert_called_once()
        # Verify agent.run was called
        mock_agent.run.assert_called_once_with("Test task with explicit model")


def test_reasoning_effort_sets_model_config():
    """`main(reasoning_effort=...)` forwards the value to the model config."""
    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.get_agent", return_value=Mock()),
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_environment", return_value=Mock()),
        patch("minisweagent.run.mini.get_config_from_spec") as mock_get_config,
    ):
        mock_get_config.return_value = {"agent": {}, "env": {}, "model": {}}
        main(
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            model_name="gpt-5",
            task="Test",
            yolo=True,
            output=None,
            model_class=None,
            agent_class=None,
            environment_class=None,
            reasoning_effort="high",
        )
        assert mock_get_model.call_args.kwargs["config"]["reasoning_effort"] == "high"


def test_reasoning_effort_flag_via_cli():
    """`mini --reasoning-effort high` is parsed and reaches the model config."""
    from typer.testing import CliRunner

    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.get_agent", return_value=Mock()),
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_environment", return_value=Mock()),
    ):
        result = CliRunner().invoke(
            app,
            ["--reasoning-effort", "high", "-m", "gpt-5", "-t", "test", "-c", str(DEFAULT_CONFIG_FILE)],
        )
        assert result.exit_code == 0, result.output
        assert mock_get_model.call_args.kwargs["config"]["reasoning_effort"] == "high"


@pytest.mark.parametrize(
    ("model_config", "expected"),
    [
        ({"reasoning_effort": "high"}, "high"),
        ({"model_kwargs": {"reasoning_effort": "medium"}}, "medium"),
        ({}, "default"),
        ({"reasoning_effort": "high", "model_kwargs": {"reasoning_effort": "low"}}, "low"),
    ],
)
def test_welcome_board_displays_reasoning_effort(model_config, expected):
    """The startup board shows the effective reasoning effort (model_kwargs wins over the shortcut)."""
    from minisweagent.run.mini import _welcome_board

    output = _render_panel(_welcome_board("gpt-5", ["mini.yaml"], {"mode": "yolo"}, model_config))
    assert "Reasoning" in output
    assert expected in output


def test_welcome_board_reasoning_effort_defaults_when_model_config_omitted():
    """The board still renders (showing the provider default) when no model config is passed."""
    from minisweagent.run.mini import _welcome_board

    output = _render_panel(_welcome_board("gpt-5", ["mini.yaml"], {"mode": "yolo"}))
    assert "Reasoning" in output
    assert "default" in output


def test_main_passes_reasoning_effort_to_welcome_board():
    """The reasoning effort set via the CLI reaches the welcome board."""
    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.get_agent", return_value=Mock()),
        patch("minisweagent.run.mini.get_model", return_value=Mock()),
        patch("minisweagent.run.mini.get_environment", return_value=Mock()),
        patch("minisweagent.run.mini.get_config_from_spec", return_value={"agent": {}, "env": {}, "model": {}}),
        patch("minisweagent.run.mini._welcome_board") as mock_board,
    ):
        main(
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            model_name="gpt-5",
            task="Test",
            yolo=True,
            output=None,
            model_class=None,
            agent_class=None,
            environment_class=None,
            reasoning_effort="high",
        )
        assert mock_board.call_args.args[3] == {"model_name": "gpt-5", "reasoning_effort": "high"}


def test_main_passes_env_reasoning_effort_to_welcome_board(monkeypatch):
    """MSWEA_REASONING_EFFORT is folded into the config shown by the startup board."""
    monkeypatch.setenv("MSWEA_REASONING_EFFORT", "high")
    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.get_agent", return_value=Mock()),
        patch("minisweagent.run.mini.get_model", return_value=Mock()),
        patch("minisweagent.run.mini.get_environment", return_value=Mock()),
        patch("minisweagent.run.mini.get_config_from_spec", return_value={"agent": {}, "env": {}, "model": {}}),
        patch("minisweagent.run.mini._welcome_board") as mock_board,
    ):
        main(
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            model_name="gpt-5",
            task="Test",
            yolo=True,
            output=None,
            model_class=None,
            agent_class=None,
            environment_class=None,
        )
        assert mock_board.call_args.args[3]["reasoning_effort"] == "high"


def test_reasoning_effort_env_var_sets_model_config(monkeypatch):
    """MSWEA_REASONING_EFFORT (e.g. from the global .env file) reaches the model config."""
    monkeypatch.setenv("MSWEA_REASONING_EFFORT", "high")
    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.get_agent", return_value=Mock()),
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_environment", return_value=Mock()),
        patch("minisweagent.run.mini.get_config_from_spec", return_value={"agent": {}, "env": {}, "model": {}}),
    ):
        main(
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            model_name="gpt-5",
            task="Test",
            yolo=True,
            output=None,
            model_class=None,
            agent_class=None,
            environment_class=None,
        )
        assert mock_get_model.call_args.kwargs["config"]["reasoning_effort"] == "high"


def test_yolo_mode_sets_correct_agent_config():
    """Test that yolo mode sets the correct agent configuration."""
    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.get_agent") as mock_get_agent,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_environment") as mock_get_env,
        patch("minisweagent.run.mini.get_config_from_spec") as mock_get_config,
    ):
        # Setup mocks
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_environment = Mock()
        mock_get_env.return_value = mock_environment
        mock_get_config.return_value = {"agent": {"system_template": "test"}, "env": {}, "model": {}}

        # Setup mock agent instance
        mock_agent = Mock()
        mock_agent.run.return_value = {"exit_status": "Success", "submission": "Result"}
        mock_get_agent.return_value = mock_agent

        # Call main function with yolo=True
        main(
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            model_name="test-model",
            task="Test yolo task",
            yolo=True,
            output=None,
            model_class=None,
            agent_class=None,
            environment_class=None,
        )

        # Verify get_agent was called with yolo mode in config
        mock_get_agent.assert_called_once()
        args, kwargs = mock_get_agent.call_args
        # The config (third positional arg) should contain the mode
        assert args[2].get("mode") == "yolo"
        # Verify agent.run was called
        mock_agent.run.assert_called_once_with("Test yolo task")


def test_confirm_mode_sets_correct_agent_config():
    """Test that when yolo=False, no explicit mode is set (defaults to None)."""
    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.get_agent") as mock_get_agent,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_environment") as mock_get_env,
        patch("minisweagent.run.mini.get_config_from_spec") as mock_get_config,
    ):
        # Setup mocks
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_environment = Mock()
        mock_get_env.return_value = mock_environment
        mock_get_config.return_value = {"agent": {"system_template": "test"}, "env": {}, "model": {}}

        # Setup mock agent instance
        mock_agent = Mock()
        mock_agent.run.return_value = {"exit_status": "Success", "submission": "Result"}
        mock_get_agent.return_value = mock_agent

        # Call main function with yolo=False (default)
        main(
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            model_name="test-model",
            task="Test confirm task",
            yolo=False,
            output=None,
            model_class=None,
            agent_class=None,
            environment_class=None,
        )

        # Verify get_agent was called with no explicit mode (defaults to None)
        mock_get_agent.assert_called_once()
        args, kwargs = mock_get_agent.call_args
        # The config (third positional arg) should not contain mode when yolo=False
        assert args[2].get("mode") is None
        # Verify agent.run was called
        mock_agent.run.assert_called_once_with("Test confirm task")


def test_mini_help():
    """Test that mini --help works correctly."""
    result = subprocess.run(
        [sys.executable, "-m", "minisweagent", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    # Strip ANSI color codes for reliable text matching
    clean_output = strip_ansi_codes(result.stdout)
    assert "Run mini-SWE-agent in your local environment." in clean_output
    assert "--help" in clean_output
    assert "--config" in clean_output
    assert "--model" in clean_output
    assert "--task" in clean_output
    assert "--yolo" in clean_output
    assert "--output" in clean_output


def test_mini_help_with_typer_runner():
    """Test help functionality using typer's test runner."""
    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    # Strip ANSI color codes for reliable text matching
    clean_output = strip_ansi_codes(result.stdout)
    assert "Run mini-SWE-agent in your local environment." in clean_output
    assert "--help" in clean_output
    assert "--config" in clean_output
    assert "--model" in clean_output
    assert "--task" in clean_output
    assert "--yolo" in clean_output
    assert "--output" in clean_output


def test_python_m_minisweagent_help():
    """Test that python -m minisweagent --help works correctly."""
    result = subprocess.run(
        [sys.executable, "-m", "minisweagent", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert "mini-SWE-agent" in result.stdout


def test_mini_script_help():
    """Test that the mini script entry point help works."""
    result = subprocess.run(
        ["mini", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert "mini-SWE-agent" in result.stdout


def test_mini_swe_agent_help():
    """Test that mini-swe-agent --help works correctly."""
    result = subprocess.run(
        ["mini-swe-agent", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    clean_output = strip_ansi_codes(result.stdout)
    assert "mini-SWE-agent" in clean_output


def test_mini_extra_help():
    """Test that mini-extra --help works correctly."""
    result = subprocess.run(
        ["mini-extra", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    clean_output = strip_ansi_codes(result.stdout)
    assert "central entry point for all extra commands" in clean_output
    assert "config" in clean_output
    assert "inspect" in clean_output
    assert "swebench" in clean_output


def test_mini_e_help():
    """Test that mini-e --help works correctly."""
    result = subprocess.run(
        ["mini-e", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    clean_output = strip_ansi_codes(result.stdout)
    assert "central entry point for all extra commands" in clean_output


@pytest.mark.parametrize(
    ("subcommand", "aliases"),
    [
        ("config", ["config"]),
        ("inspect", ["inspect", "i", "inspector"]),
        ("swebench", ["swebench"]),
        ("swebench-single", ["swebench-single"]),
    ],
)
def test_mini_extra_subcommand_help(subcommand: str, aliases: list[str]):
    """Test that mini-extra subcommands --help work correctly."""
    for alias in aliases:
        result = subprocess.run(
            ["mini-extra", alias, "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0
        # Just verify that help output is returned (content varies by subcommand)
        assert len(result.stdout) > 0


def test_mini_extra_config_help():
    """Test that mini-extra config --help works correctly."""
    result = subprocess.run(
        ["mini-extra", "config", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert len(result.stdout) > 0
    # Config command should have help output
    clean_output = strip_ansi_codes(result.stdout)
    assert "--help" in clean_output


def test_exit_immediately_flag_sets_confirm_exit_false():
    """Test that --exit-immediately flag sets confirm_exit to False in agent config."""
    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.get_agent") as mock_get_agent,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_environment") as mock_get_env,
        patch("minisweagent.run.mini.get_config_from_spec") as mock_get_config,
    ):
        # Setup mocks
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_environment = Mock()
        mock_get_env.return_value = mock_environment
        mock_get_config.return_value = {"agent": {"system_template": "test"}, "env": {}, "model": {}}

        # Create mock agent with config
        mock_agent = Mock()
        mock_agent.config.confirm_exit = False
        mock_agent.run.return_value = {"exit_status": "Success", "submission": "Result"}
        mock_get_agent.return_value = mock_agent

        # Call main function with --exit-immediately flag
        agent = main(
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            model_name="test-model",
            task="Test task",
            yolo=False,
            output=None,
            exit_immediately=True,  # This should set confirm_exit=False
            model_class=None,
            agent_class=None,
            environment_class=None,
        )

        # Verify the agent's config has confirm_exit set to False
        assert agent.config.confirm_exit is False


def test_no_exit_immediately_flag_sets_confirm_exit_true():
    """Test that when --exit-immediately flag is not used, confirm_exit defaults to True."""
    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.get_agent") as mock_get_agent,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_environment") as mock_get_env,
        patch("minisweagent.run.mini.get_config_from_spec") as mock_get_config,
    ):
        # Setup mocks
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_environment = Mock()
        mock_get_env.return_value = mock_environment
        mock_get_config.return_value = {"agent": {"system_template": "test"}, "env": {}, "model": {}}

        # Create mock agent with config
        mock_agent = Mock()
        mock_agent.config.confirm_exit = True
        mock_agent.run.return_value = {"exit_status": "Success", "submission": "Result"}
        mock_get_agent.return_value = mock_agent

        # Call main function without --exit-immediately flag (defaults to False)
        agent = main(
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            model_name="test-model",
            task="Test task",
            yolo=False,
            output=None,
            model_class=None,
            agent_class=None,
            environment_class=None,
        )

        # Verify the agent's config has confirm_exit set to True
        assert agent.config.confirm_exit is True


def test_exit_immediately_flag_with_typer_runner():
    """Test --exit-immediately flag using typer's test runner."""
    from typer.testing import CliRunner

    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.get_agent") as mock_get_agent,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_environment") as mock_get_env,
        patch("minisweagent.run.mini.get_config_from_spec") as mock_get_config,
    ):
        # Setup mocks
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_environment = Mock()
        mock_get_env.return_value = mock_environment
        mock_get_config.return_value = {"agent": {"system_template": "test"}, "env": {}, "model": {}}

        # Setup mock agent instance
        mock_agent = Mock()
        mock_agent.run.return_value = {"exit_status": "Success", "result": "Result"}
        mock_agent.messages = []
        mock_get_agent.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(app, ["--task", "Test task", "--exit-immediately", "--model", "test-model"])

        assert result.exit_code == 0
        mock_get_agent.assert_called_once()
        args, kwargs = mock_get_agent.call_args
        # The config (third positional arg) should contain confirm_exit
        assert args[2].get("confirm_exit") is False


def test_output_file_is_created(tmp_path):
    """Test that output trajectory file is created when --output is specified."""
    from typer.testing import CliRunner

    output_file = tmp_path / "test_traj.json"

    # Create a temporary config file
    config_file = tmp_path / "test_config.yaml"
    default_config_path = Path("src/minisweagent/config/default.yaml")
    config_file.write_text(default_config_path.read_text())

    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        # Keep the conversation archiver out of the developer's real
        # `~/.config/mini-swe-agent/conversations` directory (which `/resume` lists).
        patch("minisweagent.run.mini.DEFAULT_CONVERSATIONS_DIR", tmp_path / "conversations"),
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_environment") as mock_get_env,
        patch("minisweagent.agents.utils.prompt_user.prompt_session.prompt", return_value=""),
        patch("minisweagent.agents.utils.prompt_user._multiline_prompt_session.prompt", return_value=""),
    ):
        # Setup mocks
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.model_dump.return_value = {}
        mock_model.serialize.return_value = {
            "info": {
                "config": {"model": {}, "model_type": "MockModel"},
            }
        }
        mock_model.get_template_vars.return_value = {}
        mock_model.format_message.side_effect = lambda **kwargs: dict(**kwargs)
        # query now returns dict with extra["actions"]
        mock_model.query.side_effect = [
            {
                "role": "assistant",
                "content": "```mswea_bash_command\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\necho done\n```",
                "extra": {"actions": [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\necho done"}]},
            },
        ]
        # format_observation_messages returns observation messages
        mock_model.format_observation_messages.return_value = []
        mock_get_model.return_value = mock_model

        # Environment execute raises Submitted when COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT is seen
        from minisweagent.exceptions import Submitted

        def execute_side_effect(action):
            raise Submitted(
                {
                    "role": "exit",
                    "content": "done",
                    "extra": {"exit_status": "Submitted", "submission": "done"},
                }
            )

        mock_environment = Mock()
        mock_environment.config = Mock()
        mock_environment.config.model_dump.return_value = {}
        mock_environment.execute.side_effect = execute_side_effect
        mock_environment.get_template_vars.return_value = {
            "system": "TestOS",
            "release": "1.0",
            "version": "1.0.0",
            "machine": "x86_64",
        }
        mock_environment.serialize.return_value = {
            "info": {"config": {"environment": {}, "environment_type": "MockEnvironment"}}
        }
        mock_get_env.return_value = mock_environment

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--task",
                "Test task",
                "--model",
                "test-model",
                "--output",
                str(output_file),
                "--config",
                str(config_file),
            ],
        )

        if result.exit_code != 0:
            print(f"Error output: {result.output}")
        assert result.exit_code == 0
        assert output_file.exists(), f"Output file {output_file} was not created"


def _write_interrupted_trajectory(path: Path, *, remaining_submission: str = "resumed") -> None:
    """Write a non-terminal trajectory whose saved model config completes on resume."""
    import yaml

    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.models.test_models import DeterministicModel, make_output

    config = yaml.safe_load(Path("src/minisweagent/config/default.yaml").read_text())["agent"]
    agent = DefaultAgent(
        DeterministicModel(
            outputs=[
                make_output(
                    "done", [{"command": f"echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\necho {remaining_submission}"}]
                )
            ]
        ),
        LocalEnvironment(),
        **config,
    )
    agent.extra_template_vars["task"] = "Test task"
    agent.messages = [
        agent.model.format_message(role="system", content="system prompt"),
        agent.model.format_message(role="user", content="Please solve: Test task"),
    ]
    agent.cost = 0.5
    agent.n_calls = 2
    agent.save(path)


def test_cli_resume_continues_a_saved_trajectory(tmp_path):
    """`mini --resume <traj>` continues the run and writes the result back to the same file."""
    from typer.testing import CliRunner

    trajectory = tmp_path / "run.traj.json"
    _write_interrupted_trajectory(trajectory)

    with patch("minisweagent.run.mini.configure_if_first_time"):
        result = CliRunner().invoke(app, ["--resume", str(trajectory)])

    assert result.exit_code == 0, result.output
    saved = json.loads(trajectory.read_text())
    assert saved["info"]["exit_status"] == "Submitted"
    assert saved["info"]["submission"] == "resumed\n"
    assert saved["info"]["model_stats"] == {"instance_cost": 1.5, "api_calls": 3}


def test_cli_resume_missing_file_fails_cleanly(tmp_path):
    from typer.testing import CliRunner

    result = CliRunner().invoke(app, ["--resume", str(tmp_path / "missing.traj.json")])

    assert result.exit_code != 0
    assert "trajectory file not found" in result.output


def test_main_resume_loads_trajectory_and_skips_task_prompt(tmp_path):
    """Directly calling main with `resume=<path>` must load that path and run without prompting."""
    trajectory = tmp_path / "run.traj.json"
    _write_interrupted_trajectory(trajectory)
    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini._multiline_prompt") as mock_prompt,
        patch("minisweagent.run.mini.get_agent") as mock_get_agent,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_environment") as mock_get_env,
        patch("minisweagent.run.mini.get_config_from_spec") as mock_get_config,
    ):
        mock_get_model.return_value = Mock()
        mock_get_env.return_value = Mock()
        mock_agent = Mock()
        mock_agent.run.return_value = {"exit_status": "Submitted", "submission": "resumed"}
        mock_get_agent.return_value = mock_agent
        mock_get_config.return_value = {"agent": {}, "run": {}, "model": {}, "environment": {}}

        main(
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            model_name=None,
            model_class=None,
            agent_class=None,
            environment_class=None,
            task=None,
            output=None,
            resume=trajectory,
        )

    mock_agent.load.assert_called_once_with(trajectory)
    mock_agent.run.assert_called_once_with("")
    mock_prompt.assert_not_called()


def test_resume_requested_reasoning_effort_wins_over_saved(tmp_path, monkeypatch):
    """A reasoning effort requested for the resumed run must override the saved trajectory's value."""
    trajectory = tmp_path / "run.traj.json"
    trajectory.write_text(
        json.dumps(
            {
                "info": {
                    "config": {
                        "model": {
                            "model_name": "openai/deepseek-flash",
                            "reasoning_effort": "low",
                            "model_kwargs": {"drop_params": True, "reasoning_effort": "low"},
                        }
                    }
                },
                "messages": [{"role": "system", "content": "system prompt"}],
            }
        )
    )
    monkeypatch.setenv("MSWEA_REASONING_EFFORT", "max")
    captured = {}

    def fake_get_model(*args, **kwargs):
        captured["config"] = kwargs["config"]
        return Mock()

    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.get_agent", return_value=Mock()),
        patch("minisweagent.run.mini.get_model", side_effect=fake_get_model),
        patch("minisweagent.run.mini.get_environment", return_value=Mock()),
        patch("minisweagent.run.mini.get_config_from_spec", return_value={"agent": {}, "run": {}, "model": {}}),
    ):
        main(
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            model_name=None,
            model_class=None,
            agent_class=None,
            environment_class=None,
            task=None,
            output=None,
            resume=trajectory,
        )

    assert captured["config"]["reasoning_effort"] == "max"
    assert captured["config"]["model_kwargs"]["reasoning_effort"] == "max"


# --- /resume and /new at the initial "What do you want to do?" prompt ---


def _write_saved_conversation(path: Path, task: str = "Saved task") -> Path:
    """Write a trajectory file the way the conversation archiver leaves it behind."""
    path.write_text(
        json.dumps(
            {
                "info": {"task": task, "exit_status": "", "model_stats": {"api_calls": 1}},
                "messages": [
                    {"role": "system", "content": "system prompt"},
                    {"role": "user", "content": f"Please solve: {task}"},
                ],
            }
        )
    )
    return path


def _run_main_with_initial_prompt(prompt_returns, conversation_dir):
    """Run `main` without a task, faking both the task prompt and the conversation picker."""
    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.DEFAULT_CONVERSATIONS_DIR", conversation_dir),
        patch("minisweagent.run.mini._multiline_prompt", side_effect=prompt_returns),
        patch("minisweagent.agents.utils.prompt_user.prompt_session.prompt", return_value="1"),
        patch("minisweagent.run.mini.get_agent") as mock_get_agent,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_environment") as mock_get_env,
        patch("minisweagent.run.mini.get_config_from_spec", return_value={"agent": {}, "model": {}}),
    ):
        mock_get_model.return_value = Mock()
        mock_get_env.return_value = Mock()
        mock_agent = Mock(run=Mock(return_value={}))
        mock_get_agent.return_value = mock_agent

        main(
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            model_name="test-model",
            task=None,
            yolo=False,
            output=None,
            model_class=None,
            agent_class=None,
            environment_class=None,
        )
    return mock_agent


def test_initial_prompt_resume_selects_and_resumes(tmp_path):
    """`/resume` at the startup prompt lists saved conversations and loads the chosen one."""
    conversation = _write_saved_conversation(tmp_path / "saved.traj.json", "A saved task")

    mock_agent = _run_main_with_initial_prompt(["/resume"], tmp_path)

    mock_agent.resume_conversation.assert_called_once_with(conversation)
    mock_agent.run.assert_called_once_with("")


def test_initial_prompt_resume_inline_selection(tmp_path):
    """The conversation number can be passed on the `/resume` line to skip the picker."""
    conversation = _write_saved_conversation(tmp_path / "saved.traj.json", "A saved task")

    mock_agent = _run_main_with_initial_prompt(["/resume 1"], tmp_path)

    mock_agent.resume_conversation.assert_called_once_with(conversation)
    mock_agent.run.assert_called_once_with("")


def test_initial_prompt_resume_without_saved_conversations_reprompts(tmp_path):
    """`/resume` with nothing to resume explains itself and asks for a task again."""
    mock_agent = _run_main_with_initial_prompt(["/resume", "A real task"], tmp_path)

    mock_agent.resume_conversation.assert_not_called()
    mock_agent.run.assert_called_once_with("A real task")


def test_initial_prompt_new_with_inline_task(tmp_path):
    """`/new <task>` at the startup prompt runs `<task>` instead of the literal command."""
    mock_agent = _run_main_with_initial_prompt(["/new build a thing"], tmp_path)

    mock_agent.run.assert_called_once_with("build a thing")
    mock_agent.resume_conversation.assert_not_called()


@pytest.mark.parametrize(("flag",), [("--tokens",), ("--token",)])
def test_tokens_flag_replaces_anthropic_api_key(flag, tmp_path, monkeypatch):
    """`mini --tokens <key>` (and its `--token` alias) rewrites ANTHROPIC_API_KEY in the config file."""
    from typer.testing import CliRunner

    config_file = tmp_path / ".env"
    config_file.write_text("MSWEA_MODEL_NAME='gpt-5'\nANTHROPIC_API_KEY='sk-old'\n")
    monkeypatch.setattr("minisweagent.run.utilities.config.global_config_file", config_file)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-old")

    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.get_agent", return_value=Mock()),
        patch("minisweagent.run.mini.get_model"),
        patch("minisweagent.run.mini.get_environment", return_value=Mock()),
    ):
        result = CliRunner().invoke(
            app,
            [flag, "sk-new", "-m", "gpt-5", "-t", "test", "-c", str(DEFAULT_CONFIG_FILE)],
        )

    assert result.exit_code == 0, result.output
    content = config_file.read_text()
    assert "ANTHROPIC_API_KEY='sk-new'" in content
    assert "sk-old" not in content
    assert "MSWEA_MODEL_NAME='gpt-5'" in content
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-new"
    assert "ANTHROPIC_API_KEY" in result.output


def test_no_tokens_flag_leaves_config_file_alone(tmp_path, monkeypatch):
    """Without --tokens, the global config file is not modified."""
    from typer.testing import CliRunner

    config_file = tmp_path / ".env"
    config_file.write_text("ANTHROPIC_API_KEY='sk-old'\n")
    monkeypatch.setattr("minisweagent.run.utilities.config.global_config_file", config_file)

    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.get_agent", return_value=Mock()),
        patch("minisweagent.run.mini.get_model"),
        patch("minisweagent.run.mini.get_environment", return_value=Mock()),
    ):
        result = CliRunner().invoke(app, ["-m", "gpt-5", "-t", "test", "-c", str(DEFAULT_CONFIG_FILE)])

    assert result.exit_code == 0, result.output
    assert "ANTHROPIC_API_KEY='sk-old'" in config_file.read_text()
