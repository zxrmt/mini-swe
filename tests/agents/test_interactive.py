import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from minisweagent.agents.interactive import InteractiveAgent, _format_action_line, _observation_rows
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import (
    DeterministicModel,
    DeterministicResponseAPIToolcallModel,
    DeterministicToolcallModel,
    make_output,
    make_response_api_output,
    make_toolcall_output,
)
from minisweagent.models.utils.content_string import get_content_string


@contextmanager
def mock_prompts(side_effect):
    """Patch both single-line and multiline prompt sessions with shared side_effect."""
    if callable(side_effect):
        se = side_effect
    else:
        it = iter(side_effect)

        def se(*args, **kwargs):
            return next(it)

    with patch("minisweagent.agents.utils.prompt_user.prompt_session.prompt", side_effect=se):
        with patch("minisweagent.agents.utils.prompt_user._multiline_prompt_session.prompt", side_effect=se):
            yield


# --- Helper functions to abstract message format differences ---


def get_text(msg: dict) -> str:
    """Extract text content from a message regardless of format."""
    content = msg.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content:
        return content[0].get("text", "")
    return ""


# --- Model factory functions ---


def make_text_model(outputs_spec: list[tuple[str, list[dict]]], **kwargs) -> DeterministicModel:
    """Create a DeterministicModel from a list of (content, actions) tuples."""
    return DeterministicModel(outputs=[make_output(content, actions) for content, actions in outputs_spec], **kwargs)


def make_tc_model(outputs_spec: list[tuple[str, list[dict]]], **kwargs) -> DeterministicToolcallModel:
    """Create a DeterministicToolcallModel from a list of (content, actions) tuples."""
    outputs = []
    for i, (content, actions) in enumerate(outputs_spec):
        tc_actions = []
        tool_calls = []
        for j, action in enumerate(actions):
            tool_call_id = f"call_{i}_{j}"
            tc_actions.append({"command": action["command"], "tool_call_id": tool_call_id})
            tool_calls.append(
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": "bash", "arguments": f'{{"command": "{action["command"]}"}}'},
                }
            )
        outputs.append(make_toolcall_output(content, tool_calls, tc_actions))
    return DeterministicToolcallModel(outputs=outputs, **kwargs)


def make_response_api_model(
    outputs_spec: list[tuple[str, list[dict]]], **kwargs
) -> DeterministicResponseAPIToolcallModel:
    """Create a DeterministicResponseAPIToolcallModel from a list of (content, actions) tuples."""
    outputs = []
    for i, (content, actions) in enumerate(outputs_spec):
        api_actions = []
        for j, action in enumerate(actions):
            tool_call_id = f"call_resp_{i}_{j}"
            api_actions.append({"command": action["command"], "tool_call_id": tool_call_id})
        outputs.append(make_response_api_output(content, api_actions))
    return DeterministicResponseAPIToolcallModel(outputs=outputs, **kwargs)


def _make_model(outputs: list[tuple[str, list[dict]]], **kwargs) -> DeterministicModel:
    """Create a DeterministicModel from a list of (content, actions) tuples.

    Kept for backward compatibility with tests that don't need parametrization.
    """
    return make_text_model(outputs, **kwargs)


# --- Fixtures ---


@pytest.fixture
def default_config():
    """Load default agent config from config/default.yaml"""
    config_path = Path("src/minisweagent/config/default.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["agent"]


@pytest.fixture
def toolcall_config():
    """Load toolcall agent config from config/mini.yaml"""
    config_path = Path("src/minisweagent/config/mini.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["agent"]


@pytest.fixture(params=["text", "toolcall", "response_api"])
def model_factory(request, default_config, toolcall_config):
    """Parametrized fixture that returns (factory_fn, config) for all three model types."""
    if request.param == "text":
        return make_text_model, default_config
    elif request.param == "toolcall":
        return make_tc_model, toolcall_config
    else:  # response_api
        return make_response_api_model, toolcall_config


def test_successful_completion_with_confirmation(model_factory):
    """Test agent completes successfully when user confirms all actions."""
    factory, config = model_factory
    with mock_prompts(["", ""]):  # Confirm action with Enter, then no new task
        agent = InteractiveAgent(
            model=factory(
                [
                    ("Finishing", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'completed'"}]),
                ]
            ),
            env=LocalEnvironment(),
            **config,
        )

        info = agent.run("Test completion with confirmation")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "completed\n"
        assert agent.n_calls == 1


def test_action_rejection_and_recovery(model_factory):
    """Test agent handles action rejection and can recover."""
    factory, config = model_factory
    with mock_prompts(
        [
            "User rejected this action",  # Reject first action
            "",  # Confirm second action
            "",  # No new task when agent wants to finish
        ]
    ):
        agent = InteractiveAgent(
            model=factory(
                [
                    ("First try", [{"command": "echo 'first attempt'"}]),
                    ("Second try", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'recovered'"}]),
                ]
            ),
            env=LocalEnvironment(),
            **config,
        )

        info = agent.run("Test action rejection")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "recovered\n"
        assert agent.n_calls == 2
        # Should have rejection message in conversation
        rejection_messages = [msg for msg in agent.messages if "User rejected this action" in get_text(msg)]
        assert len(rejection_messages) == 1


def test_yolo_mode_activation(model_factory):
    """Test entering yolo mode disables confirmations."""
    factory, config = model_factory
    with mock_prompts(
        [
            "/y",  # Enter yolo mode
            "",  # This should be ignored since yolo mode is on
            "",  # No new task when agent wants to finish
        ]
    ):
        agent = InteractiveAgent(
            model=factory(
                [
                    ("Test command", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'yolo works'"}]),
                ]
            ),
            env=LocalEnvironment(),
            **config,
        )

        info = agent.run("Test yolo mode")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "yolo works\n"
        assert agent.config.mode == "yolo"


def test_help_command(model_factory):
    """Test help command shows help and continues normally."""
    factory, config = model_factory
    with mock_prompts(
        [
            "/h",  # Show help
            "",  # Confirm action after help
            "",  # No new task when agent wants to finish
        ]
    ):
        with patch("minisweagent.agents.interactive.console.print") as mock_print:
            agent = InteractiveAgent(
                model=factory(
                    [
                        ("Test help", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'help shown'"}]),
                    ]
                ),
                env=LocalEnvironment(),
                **config,
            )

            info = agent.run("Test help command")
            assert info["exit_status"] == "Submitted"
            assert info["submission"] == "help shown\n"
            # Check that help was printed
            help_calls = [call for call in mock_print.call_args_list if "/y" in str(call)]
            assert len(help_calls) > 0


def test_whitelisted_actions_skip_confirmation(model_factory):
    """Test that whitelisted actions don't require confirmation."""
    factory, config = model_factory
    with mock_prompts([""]):  # No new task when agent wants to finish
        agent = InteractiveAgent(
            model=factory(
                [
                    (
                        "Whitelisted",
                        [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'no confirmation needed'"}],
                    ),
                ]
            ),
            env=LocalEnvironment(),
            **{
                **config,
                "whitelist_actions": [r"echo.*"],
            },
        )

        info = agent.run("Test whitelisted actions")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "no confirmation needed\n"


def _test_interruption_helper(
    factory, config, interruption_input, expected_message_fragment, problem_statement="Test interruption"
):
    """Helper function for testing interruption scenarios."""
    agent = InteractiveAgent(
        model=factory(
            [
                ("Initial step", [{"command": "echo 'will be interrupted'"}]),
                (
                    "Recovery",
                    [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'recovered from interrupt'"}],
                ),
            ]
        ),
        env=LocalEnvironment(),
        **config,
    )

    # Mock the query to raise KeyboardInterrupt on first call, then work normally
    original_query = agent.query
    call_count = 0

    def mock_query(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise KeyboardInterrupt()
        return original_query(*args, **kwargs)

    # Mock console.input based on the interruption_input parameter
    input_call_count = 0

    def mock_input(prompt, **kwargs):
        nonlocal input_call_count
        input_call_count += 1
        if input_call_count == 1:
            return interruption_input  # For the interruption handling
        return ""  # Confirm all subsequent actions

    with mock_prompts(mock_input):
        with patch.object(agent, "query", side_effect=mock_query):
            info = agent.run(problem_statement)

    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "recovered from interrupt\n"
    # Check that the expected interruption message was added
    interrupt_messages = [msg for msg in agent.messages if expected_message_fragment in get_text(msg)]
    assert len(interrupt_messages) == 1

    return agent, interrupt_messages[0]


def test_interruption_handling_with_message(model_factory):
    """Test that interruption with user message is handled properly."""
    factory, config = model_factory
    agent, interrupt_message = _test_interruption_helper(factory, config, "User interrupted", "Interrupted by user")

    # Additional verification specific to this test
    assert "User interrupted" in get_text(interrupt_message)


def test_interruption_handling_empty_message(model_factory):
    """Test that interruption with empty input is handled properly."""
    factory, config = model_factory
    _test_interruption_helper(factory, config, "", "Temporary interruption caught")


def test_multiple_confirmations_and_commands(model_factory):
    """Test complex interaction with multiple confirmations and commands."""
    factory, config = model_factory
    with mock_prompts(
        [
            "reject first",  # Reject first action
            "/h",  # Show help for second action
            "/y",  # After help, enter yolo mode
            "",  # After yolo mode enabled, confirm (but yolo mode will skip future confirmations)
            "",  # No new task when agent wants to finish
        ]
    ):
        agent = InteractiveAgent(
            model=factory(
                [
                    ("First action", [{"command": "echo 'first'"}]),
                    (
                        "Second action",
                        [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'complex flow completed'"}],
                    ),
                ]
            ),
            env=LocalEnvironment(),
            **config,
        )

        info = agent.run("Test complex interaction flow")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "complex flow completed\n"
        assert agent.config.mode == "yolo"  # Should be in yolo mode
        assert agent.n_calls == 2


def test_non_whitelisted_action_requires_confirmation(model_factory):
    """Test that non-whitelisted actions still require confirmation."""
    factory, config = model_factory
    with mock_prompts(["", ""]):  # Confirm action, then no new task
        agent = InteractiveAgent(
            model=factory(
                [
                    (
                        "Non-whitelisted",
                        [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'confirmed'"}],
                    ),
                ]
            ),
            env=LocalEnvironment(),
            **{
                **config,
                "whitelist_actions": [r"ls.*"],  # Only ls commands whitelisted
            },
        )

        info = agent.run("Test non-whitelisted action")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "confirmed\n"


# New comprehensive mode switching tests


def test_human_mode_basic_functionality(model_factory):
    """Test human mode where user enters shell commands directly."""
    factory, config = model_factory
    with mock_prompts(
        [
            "echo 'user command'",  # User enters shell command
            "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'human mode works'",  # User enters final command
            "",  # No new task when agent wants to finish
        ]
    ):
        agent = InteractiveAgent(
            model=factory([]),  # LM shouldn't be called in human mode
            env=LocalEnvironment(),
            **{
                **config,
                "mode": "human",
            },
        )

        info = agent.run("Test human mode")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "human mode works\n"
        assert agent.config.mode == "human"
        assert agent.n_calls == 0  # LM should not be called


def test_human_mode_switch_to_yolo(model_factory):
    """Test switching from human mode to yolo mode."""
    factory, config = model_factory
    with mock_prompts(
        [
            "/y",  # Switch to yolo mode from human mode
            "",  # Confirm action in yolo mode (though no confirmation needed)
            "",  # No new task when agent wants to finish
        ]
    ):
        agent = InteractiveAgent(
            model=factory(
                [
                    (
                        "LM action",
                        [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'switched to yolo'"}],
                    ),
                ]
            ),
            env=LocalEnvironment(),
            **{
                **config,
                "mode": "human",
            },
        )

        info = agent.run("Test human to yolo switch")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "switched to yolo\n"
        assert agent.config.mode == "yolo"
        assert agent.n_calls == 1


def test_human_mode_switch_to_confirm(model_factory):
    """Test switching from human mode to confirm mode."""
    factory, config = model_factory
    with mock_prompts(
        [
            "/c",  # Switch to confirm mode from human mode
            "",  # Confirm action in confirm mode
            "",  # No new task when agent wants to finish
        ]
    ):
        agent = InteractiveAgent(
            model=factory(
                [
                    (
                        "LM action",
                        [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'switched to confirm'"}],
                    ),
                ]
            ),
            env=LocalEnvironment(),
            **{
                **config,
                "mode": "human",
            },
        )

        info = agent.run("Test human to confirm switch")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "switched to confirm\n"
        assert agent.config.mode == "confirm"
        assert agent.n_calls == 1


def test_confirmation_mode_switch_to_human_with_rejection(model_factory):
    """Test switching from confirm mode to human mode with /u command."""
    factory, config = model_factory
    with mock_prompts(
        [
            "/u",  # Switch to human mode and reject action
            "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'human command after rejection'",  # Human command
            "",  # No new task when agent wants to finish
        ]
    ):
        agent = InteractiveAgent(
            model=factory(
                [
                    ("LM action", [{"command": "echo 'first action'"}]),
                    ("Recovery action", [{"command": "echo 'recovery'"}]),
                ]
            ),
            env=LocalEnvironment(),
            **{
                **config,
                "mode": "confirm",
            },
        )

        info = agent.run("Test confirm to human switch")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "human command after rejection\n"
        assert agent.config.mode == "human"
        # Should have rejection message
        rejection_messages = [msg for msg in agent.messages if "Switching to human mode" in get_text(msg)]
        assert len(rejection_messages) == 1


def test_confirmation_mode_switch_to_yolo_and_continue(model_factory):
    """Test switching from confirm mode to yolo mode with /y and continuing with action."""
    factory, config = model_factory
    with mock_prompts(
        [
            "/y",  # Switch to yolo mode and confirm current action
            "",  # No new task when agent wants to finish
        ]
    ):
        agent = InteractiveAgent(
            model=factory(
                [
                    (
                        "LM action",
                        [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'switched and continued'"}],
                    ),
                ]
            ),
            env=LocalEnvironment(),
            **{
                **config,
                "mode": "confirm",
            },
        )

        info = agent.run("Test confirm to yolo switch")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "switched and continued\n"
        assert agent.config.mode == "yolo"


def test_mode_switch_during_keyboard_interrupt(model_factory):
    """Test mode switching during keyboard interrupt handling."""
    factory, config = model_factory
    agent = InteractiveAgent(
        model=factory(
            [
                ("Initial step", [{"command": "echo 'will be interrupted'"}]),
                (
                    "Recovery",
                    [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'recovered after mode switch'"}],
                ),
            ]
        ),
        env=LocalEnvironment(),
        **{
            **config,
            "mode": "confirm",
        },
    )

    # Mock the query to raise KeyboardInterrupt on first call
    original_query = agent.query
    call_count = 0

    def mock_query(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise KeyboardInterrupt()
        return original_query(*args, **kwargs)

    with mock_prompts(
        [
            "/y",  # Switch to yolo mode during interrupt
            "",  # Confirm subsequent actions (though yolo mode won't ask)
        ]
    ):
        with patch.object(agent, "query", side_effect=mock_query):
            info = agent.run("Test interrupt mode switch")

    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "recovered after mode switch\n"
    assert agent.config.mode == "yolo"
    # Should have interruption message
    interrupt_messages = [msg for msg in agent.messages if "Temporary interruption caught" in get_text(msg)]
    assert len(interrupt_messages) == 1


def test_already_in_mode_behavior(model_factory):
    """Test behavior when trying to switch to the same mode."""
    factory, config = model_factory
    with mock_prompts(
        [
            "/c",  # Try to switch to confirm mode when already in confirm mode
            "",  # Confirm action after the "already in mode" recursive prompt
            "",  # No new task when agent wants to finish
        ]
    ):
        agent = InteractiveAgent(
            model=factory(
                [
                    (
                        "Test action",
                        [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'already in mode'"}],
                    ),
                ]
            ),
            env=LocalEnvironment(),
            **{
                **config,
                "mode": "confirm",
            },
        )

        info = agent.run("Test already in mode")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "already in mode\n"
        assert agent.config.mode == "confirm"


def test_all_mode_transitions_yolo_to_others(model_factory):
    """Test transitions from yolo mode to other modes."""
    factory, config = model_factory
    with mock_prompts(
        [
            "/c",  # Switch from yolo to confirm
            "",  # Confirm action in confirm mode
            "",  # No new task when agent wants to finish
        ]
    ):
        agent = InteractiveAgent(
            model=factory(
                [
                    ("First action", [{"command": "echo 'yolo action'"}]),
                    (
                        "Second action",
                        [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'confirm action'"}],
                    ),
                ]
            ),
            env=LocalEnvironment(),
            **{
                **config,
                "mode": "yolo",
            },
        )

        # Trigger first action in yolo mode (should execute without confirmation)
        # Then interrupt to switch mode
        original_query = agent.query
        call_count = 0

        def mock_query(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Interrupt on second query
                raise KeyboardInterrupt()
            return original_query(*args, **kwargs)

        with patch.object(agent, "query", side_effect=mock_query):
            info = agent.run("Test yolo to confirm transition")

        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "confirm action\n"
        assert agent.config.mode == "confirm"


def test_all_mode_transitions_confirm_to_human(model_factory):
    """Test transition from confirm mode to human mode."""
    factory, config = model_factory
    with mock_prompts(
        [
            "/u",  # Switch from confirm to human (rejecting action)
            "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'human command'",  # User enters command in human mode
            "",  # No new task when agent wants to finish
        ]
    ):
        agent = InteractiveAgent(
            model=factory([("LM action", [{"command": "echo 'rejected action'"}])]),
            env=LocalEnvironment(),
            **{
                **config,
                "mode": "confirm",
            },
        )

        info = agent.run("Test confirm to human transition")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "human command\n"
        assert agent.config.mode == "human"


def test_help_command_from_different_contexts(model_factory):
    """Test help command works from different contexts (confirmation, interrupt, human mode)."""
    factory, config = model_factory
    # Test help during confirmation
    with mock_prompts(
        [
            "/h",  # Show help during confirmation
            "",  # Confirm after help
            "",  # No new task when agent wants to finish
        ]
    ):
        with patch("minisweagent.agents.interactive.console.print") as mock_print:
            agent = InteractiveAgent(
                model=factory(
                    [
                        (
                            "Test action",
                            [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'help works'"}],
                        ),
                    ]
                ),
                env=LocalEnvironment(),
                **{
                    **config,
                    "mode": "confirm",
                },
            )

            info = agent.run("Test help from confirmation")
            assert info["exit_status"] == "Submitted"
            assert info["submission"] == "help works\n"
            # Verify help was shown
            help_calls = [call for call in mock_print.call_args_list if "Current mode: " in str(call)]
            assert len(help_calls) > 0


def test_help_command_from_human_mode(model_factory):
    """Test help command works from human mode."""
    factory, config = model_factory
    with mock_prompts(
        [
            "/h",  # Show help in human mode
            "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'help in human mode'",  # User command after help
            "",  # No new task when agent wants to finish
        ]
    ):
        with patch("minisweagent.agents.interactive.console.print") as mock_print:
            agent = InteractiveAgent(
                model=factory([]),  # LM shouldn't be called
                env=LocalEnvironment(),
                **{
                    **config,
                    "mode": "human",
                },
            )

            info = agent.run("Test help from human mode")
            assert info["exit_status"] == "Submitted"
            assert info["submission"] == "help in human mode\n"
            # Verify help was shown
            help_calls = [call for call in mock_print.call_args_list if "Current mode: " in str(call)]
            assert len(help_calls) > 0


def test_complex_mode_switching_sequence(model_factory):
    """Test complex sequence of mode switches across different contexts."""
    factory, config = model_factory
    agent = InteractiveAgent(
        model=factory(
            [
                ("Action 1", [{"command": "echo 'action1'"}]),
                ("Action 2", [{"command": "echo 'action2'"}]),
                ("Action 3", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'final action'"}]),
            ]
        ),
        env=LocalEnvironment(),
        **{
            **config,
            "mode": "confirm",
        },
    )

    # Mock interruption on second query
    original_query = agent.query
    call_count = 0

    def mock_query(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise KeyboardInterrupt()
        return original_query(*args, **kwargs)

    with mock_prompts(
        [
            "/y",  # Confirm->Yolo during first action confirmation
            "/u",  # Yolo->Human during interrupt
            "/c",  # Human->Confirm in human mode
            "",  # Confirm final action
            "",  # No new task when agent wants to finish
            "",  # Extra empty input for any additional prompts
            "",  # Extra empty input for any additional prompts
        ]
    ):
        with patch.object(agent, "query", side_effect=mock_query):
            info = agent.run("Test complex mode switching")

    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "final action\n"
    assert agent.config.mode == "confirm"  # Should end in confirm mode


def test_limits_exceeded_multiple_times_with_continuation(model_factory):
    """Test that limits can be exceeded and updated multiple times."""
    factory, config = model_factory
    agent = InteractiveAgent(
        model=factory(
            [
                ("Step 1", [{"command": "echo 'step1'"}]),
                ("Step 2", [{"command": "echo 'step2'"}]),
                ("Step 3", [{"command": "echo 'step3'"}]),
                ("Step 4", [{"command": "echo 'step4'"}]),
                (
                    "Final",
                    [
                        {
                            "command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'completed after multiple increases'"
                        }
                    ],
                ),
            ],
            cost_per_call=1.0,  # Standard cost per call
        ),
        env=LocalEnvironment(),
        **{
            **config,
            "step_limit": 1,  # Will be exceeded after first step
            "mode": "yolo",
        },
    )

    # Mock input() to provide new limits multiple times (interactive terminal).
    # First limit increase: step_limit=2, then step_limit=10 when exceeded again
    with patch.object(InteractiveAgent, "_stdin_is_interactive", return_value=True):
        with patch("builtins.input", side_effect=["2", "10"]):
            with mock_prompts([""]):  # No new task
                with patch("minisweagent.agents.interactive.console.print"):
                    info = agent.run("Test multiple limit increases")

    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "completed after multiple increases\n"
    assert agent.n_calls == 5  # Should complete all 5 steps
    assert agent.config.step_limit == 10  # Should have final updated step limit


def test_limits_exceeded_non_interactive_stops_cleanly(model_factory):
    """Without a terminal (e.g. `--yolo` in CI / a sandbox), a breached limit must
    stop cleanly with a LimitsExceeded exit status instead of crashing on EOFError
    when trying to prompt for new limits."""
    factory, config = model_factory
    agent = InteractiveAgent(
        model=factory(
            [("Step 1", [{"command": "echo 'first step'"}])],
            cost_per_call=0.6,
        ),
        env=LocalEnvironment(),
        **{
            **config,
            "step_limit": 1,  # breached after the first step
            "mode": "yolo",
        },
    )

    with patch.object(InteractiveAgent, "_stdin_is_interactive", return_value=False):
        with patch("builtins.input", side_effect=AssertionError("input() must not be called")) as mock_in:
            with patch("minisweagent.agents.interactive.console.print"):
                info = agent.run("Test non-interactive limit stop")

    assert info["exit_status"] == "LimitsExceeded"
    assert agent.n_calls == 1  # one model call happened, the next was blocked by the limit
    mock_in.assert_not_called()


def test_time_exceeded_never_prompts(model_factory):
    """A wall-clock limit can't be lifted by raising the step limit, so it must
    always stop cleanly -- even with an interactive terminal -- rather than prompt
    (which would otherwise loop forever)."""
    factory, config = model_factory
    agent = InteractiveAgent(
        model=factory([("Step 1", [{"command": "echo 'first step'"}])]),
        env=LocalEnvironment(),
        **{**config, "step_limit": 10, "wall_time_limit_seconds": 1, "mode": "yolo"},
    )
    agent._start_time = 0  # force the wall-clock budget to be already exhausted

    with patch.object(InteractiveAgent, "_stdin_is_interactive", return_value=True):
        with patch("builtins.input", side_effect=AssertionError("input() must not be called")) as mock_in:
            with patch("minisweagent.agents.interactive.console.print"):
                info = agent.run("Test time-exceeded clean stop")

    assert info["exit_status"] == "TimeExceeded"
    assert agent.n_calls == 0  # limit tripped before any model call
    mock_in.assert_not_called()


def test_continue_after_completion_with_new_task(model_factory):
    """Test that user can provide a new task when agent wants to finish."""
    factory, config = model_factory
    with mock_prompts(
        [
            "",  # Confirm first action
            "Create a new file",  # Provide new task when agent wants to finish
            "",  # Confirm second action for new task
            "",  # Don't provide another task after second completion (finish)
        ]
    ):
        agent = InteractiveAgent(
            model=factory(
                [
                    (
                        "First task",
                        [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'first task completed'"}],
                    ),
                    (
                        "Second task",
                        [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'new task completed'"}],
                    ),
                ]
            ),
            env=LocalEnvironment(),
            **config,
        )

        info = agent.run("Complete the initial task")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "new task completed\n"
        assert agent.n_calls == 2
        # Should have the new task message in conversation
        new_task_messages = [
            msg for msg in agent.messages if "The user added a new task: Create a new file" in get_text(msg)
        ]
        assert len(new_task_messages) == 1


def test_continue_after_completion_does_not_replay_exit_message(model_factory):
    """Continuing a finished conversation must not pass the internal `exit` role to the model.

    Regression test: the `exit` marker that ends a run is not a valid chat role, so replaying it
    as history made the API reject the follow-up request ("unknown variant `exit`"). It must be
    dropped before the conversation continues.
    """
    factory, config = model_factory
    with mock_prompts(["", "Create a new file", "", ""]):
        agent = InteractiveAgent(
            model=factory(
                [
                    ("First", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'first done'"}]),
                    ("Second", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'second done'"}]),
                ]
            ),
            env=LocalEnvironment(),
            **config,
        )
        queried_roles = []
        original_query = agent.model.query

        def recording_query(messages, **kwargs):
            queried_roles.append([msg.get("role") for msg in messages])
            return original_query(messages, **kwargs)

        agent.model.query = recording_query
        info = agent.run("Complete the initial task")

    assert info["exit_status"] == "Submitted"
    # At least one follow-up query happened after the first completion...
    assert len(queried_roles) >= 2
    # ...and the model never saw the `exit` marker.
    assert not any("exit" in roles for roles in queried_roles)
    # Only the final terminating marker is kept in the saved history.
    assert [msg for msg in agent.messages if msg.get("role") == "exit"] == [agent.messages[-1]]


def test_continue_after_completion_without_new_task(model_factory):
    """Test that agent finishes normally when user doesn't provide a new task."""
    factory, config = model_factory
    with mock_prompts(
        [
            "",  # Confirm first action
            "",  # Don't provide new task when agent wants to finish (empty input)
        ]
    ):
        agent = InteractiveAgent(
            model=factory(
                [
                    (
                        "Task completion",
                        [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'original task completed'"}],
                    ),
                ]
            ),
            env=LocalEnvironment(),
            **config,
        )

        info = agent.run("Complete the task")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "original task completed\n"
        assert agent.n_calls == 1
        # Should not have any new task messages
        new_task_messages = [msg for msg in agent.messages if "The user added a new task" in get_text(msg)]
        assert len(new_task_messages) == 0


def test_continue_after_completion_multiple_cycles(model_factory):
    """Test multiple continuation cycles with new tasks."""
    factory, config = model_factory
    with mock_prompts(
        [
            "",  # Confirm first action
            "Second task",  # Provide first new task
            "",  # Confirm second action
            "Third task",  # Provide second new task
            "",  # Confirm third action
            "",  # Don't provide another task (finish)
        ]
    ):
        agent = InteractiveAgent(
            model=factory(
                [
                    ("First", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'first completed'"}]),
                    ("Second", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'second completed'"}]),
                    ("Third", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'third completed'"}]),
                ]
            ),
            env=LocalEnvironment(),
            **config,
        )

        info = agent.run("Initial task")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "third completed\n"
        assert agent.n_calls == 3
        # Should have both new task messages
        new_task_messages = [msg for msg in agent.messages if "The user added a new task" in get_text(msg)]
        assert len(new_task_messages) == 2
        assert "Second task" in get_text(new_task_messages[0])
        assert "Third task" in get_text(new_task_messages[1])


def test_continue_after_completion_in_yolo_mode(model_factory):
    """Test continuation when starting in yolo mode (no confirmations needed)."""
    factory, config = model_factory
    with mock_prompts(
        [
            "Create a second task",  # Provide new task when agent wants to finish
            "",  # Don't provide another task after second completion (finish)
        ]
    ):
        agent = InteractiveAgent(
            model=factory(
                [
                    ("First", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'first completed'"}]),
                    (
                        "Second",
                        [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'second task completed'"}],
                    ),
                ]
            ),
            env=LocalEnvironment(),
            **{
                **config,
                "mode": "yolo",  # Start in yolo mode
            },
        )

        info = agent.run("Initial task")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "second task completed\n"
        assert agent.config.mode == "yolo"
        assert agent.n_calls == 2
        # Should have the new task message
        new_task_messages = [msg for msg in agent.messages if "Create a second task" in get_text(msg)]
        assert len(new_task_messages) == 1


def test_confirm_exit_enabled_asks_for_confirmation(model_factory):
    """Test that when confirm_exit=True, agent asks for confirmation before finishing."""
    factory, config = model_factory
    with mock_prompts(["", ""]):  # Confirm action, then no new task (empty string to exit)
        agent = InteractiveAgent(
            model=factory(
                [
                    ("Finishing", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'completed'"}]),
                ]
            ),
            env=LocalEnvironment(),
            **{
                **config,
                "confirm_exit": True,  # Should ask for confirmation
            },
        )

        info = agent.run("Test confirm exit enabled")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "completed\n"
        assert agent.n_calls == 1


def test_confirm_exit_disabled_exits_immediately(model_factory):
    """Test that when confirm_exit=False, agent exits immediately without asking."""
    factory, config = model_factory
    with mock_prompts([""]):  # Only confirm action, no exit confirmation needed
        agent = InteractiveAgent(
            model=factory(
                [
                    ("Finishing", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'completed'"}]),
                ]
            ),
            env=LocalEnvironment(),
            **{
                **config,
                "confirm_exit": False,  # Should NOT ask for confirmation
            },
        )

        info = agent.run("Test confirm exit disabled")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "completed\n"
        assert agent.n_calls == 1


def test_confirm_exit_with_new_task_continues_execution(model_factory):
    """Test that when user provides new task at exit confirmation, agent continues."""
    factory, config = model_factory
    with mock_prompts(
        [
            "",  # Confirm first action
            "Please do one more thing",  # Provide new task instead of exiting
            "",  # Confirm second action
            "",  # No new task on second exit confirmation
        ]
    ):
        agent = InteractiveAgent(
            model=factory(
                [
                    ("First task", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'first done'"}]),
                    (
                        "Additional task",
                        [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'additional done'"}],
                    ),
                ]
            ),
            env=LocalEnvironment(),
            **{
                **config,
                "confirm_exit": True,
            },
        )

        info = agent.run("Test exit with new task")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "additional done\n"
        assert agent.n_calls == 2
        # Check that the new task was added to the conversation
        new_task_messages = [msg for msg in agent.messages if "Please do one more thing" in get_text(msg)]
        assert len(new_task_messages) == 1


def test_confirm_exit_config_field_defaults(model_factory):
    """Test that confirm_exit field has correct default value."""
    factory, config = model_factory
    agent = InteractiveAgent(
        model=factory([]),
        env=LocalEnvironment(),
        **config,
    )
    # Default should be True
    assert agent.config.confirm_exit is True


def test_confirm_exit_config_field_can_be_set(model_factory):
    """Test that confirm_exit field can be explicitly set."""
    factory, config = model_factory
    agent_with_confirm = InteractiveAgent(
        model=factory([]),
        env=LocalEnvironment(),
        **{
            **config,
            "confirm_exit": True,
        },
    )
    assert agent_with_confirm.config.confirm_exit is True

    agent_without_confirm = InteractiveAgent(
        model=factory([]),
        env=LocalEnvironment(),
        **{
            **config,
            "confirm_exit": False,
        },
    )
    assert agent_without_confirm.config.confirm_exit is False


def test_submission_help_then_human_mode(model_factory):
    """Test: agent submits → /h shows help and reprompts → /u switches to human → echo 'test' → see 'test'."""
    factory, config = model_factory
    with mock_prompts(
        [
            "/h",  # At submission prompt: show help, reprompt
            "/u",  # At submission prompt: switch to human mode
            "echo 'test'",  # In human mode: run command
            "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'",  # Submit from human mode
            "",  # Confirm exit
        ]
    ):
        with patch("minisweagent.agents.interactive.console.print") as mock_print:
            agent = InteractiveAgent(
                model=factory(
                    [("Finishing", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'initial'"}])],
                ),
                env=LocalEnvironment(),
                **{**config, "mode": "yolo"},
            )
            info = agent.run("Solve the issue")
    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "done\n"
    assert agent.config.mode == "human"
    assert agent.n_calls == 1
    # Help was shown
    assert any("/y" in str(c) for c in mock_print.call_args_list)
    # echo 'test' output is visible in the conversation
    assert any("test" in get_text(msg) for msg in agent.messages)


def test_submission_enter_quits(model_factory):
    """Test: agent submits → Enter → quit for real."""
    factory, config = model_factory
    with mock_prompts([""]):  # At submission prompt: Enter to quit
        agent = InteractiveAgent(
            model=factory(
                [("Finishing", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'completed'"}])],
            ),
            env=LocalEnvironment(),
            **{**config, "mode": "yolo"},
        )
        info = agent.run("Solve the issue")
    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "completed\n"
    assert agent.n_calls == 1


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("ls -la", "Bash(ls -la)"),
        ("cat <<'EOF' > f.py\nprint(1)\nEOF", "Bash(cat <<'EOF' > f.py…)"),
        ("x" * 200, f"Bash({'x' * 70}…)"),
    ],
)
def test_format_action_line(command, expected):
    assert _format_action_line(command) == expected


@pytest.mark.parametrize(
    ("content", "max_lines", "expected"),
    [
        ("", 0, (["(No output)"], False)),
        ("single", 3, (["single"], False)),
        ("a\nb\nc", 0, (["a", "b", "c"], False)),
        ("a\nb\nc\nd", 3, (["a", "b", "c", "… +1 lines"], True)),
        ("a\nb\nc", 3, (["a", "b", "c"], False)),
        ("x" * 160, 0, (["x" * 75, "x" * 75, "x" * 10], False)),
        ("x" * 160, 2, (["x" * 75, "x" * 75, "… +1 lines"], True)),
    ],
)
def test_observation_rows(content, max_lines, expected):
    assert _observation_rows(content, max_lines) == expected


def test_display_truncation_does_not_reach_the_model(toolcall_config, capsys):
    """Long outputs are elided on screen, but the model still gets all of them."""
    agent = InteractiveAgent(
        model=make_tc_model(
            [
                ("Counting", [{"command": "seq 1 20"}]),
                ("Finishing", [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}]),
            ],
            observation_template="{{ output.output }}",
        ),
        env=LocalEnvironment(),
        **{**toolcall_config, "mode": "yolo", "confirm_exit": False, "observation_display_lines": 3},
    )
    agent.run("Test truncation")
    output = capsys.readouterr().out
    assert "● Bash(seq 1 20)" in output
    assert "  ⎿  1\n     2\n     3\n     … +17 lines" in output
    assert "\n     20\n" not in output
    assert "20" in get_content_string(agent.messages[3])


def test_text_based_model_does_not_repeat_its_command(default_config, capsys):
    """Text-based models spell out the command in their reasoning, so no extra action line."""
    agent = InteractiveAgent(
        model=make_text_model(
            [
                ("Counting\n```bash\nseq 1 3\n```", [{"command": "seq 1 3"}]),
                ("Finishing", [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}]),
            ],
            observation_template="{{ output.output }}",
        ),
        env=LocalEnvironment(),
        **{**default_config, "mode": "yolo", "confirm_exit": False},
    )
    agent.run("Test no duplicate command")
    output = capsys.readouterr().out
    assert "Bash(" not in output
    assert "seq 1 3" in output and "  ⎿  " in output


def test_submission_is_shown_before_asking_for_a_new_task(toolcall_config, capsys):
    """The final output must be visible before the "Task Completed" prompt is shown."""
    agent = InteractiveAgent(
        model=make_tc_model(
            [("Finishing", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'THE FINAL ANSWER'"}])]
        ),
        env=LocalEnvironment(),
        **{**toolcall_config, "mode": "yolo", "confirm_exit": True},
    )
    with mock_prompts([""]):  # No new task: accept the completion prompt
        info = agent.run("Answer the question")
    output = capsys.readouterr().out
    assert info["submission"] == "THE FINAL ANSWER\n"
    assert "THE FINAL ANSWER" in output
    assert output.index("THE FINAL ANSWER") < output.index("Task Completed")


def test_multiline_task_is_collapsed_in_the_status_line(toolcall_config, capsys):
    """A multi-line task (e.g. pasted terminal output) must not flood the display with blank lines."""
    agent = InteractiveAgent(
        model=make_tc_model(
            [("Finishing", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'"}])]
        ),
        env=LocalEnvironment(),
        **{**toolcall_config, "mode": "yolo", "confirm_exit": False, "quiet": True},
    )
    agent.run("first line\n\n\n\nsecond line")
    output = capsys.readouterr().out
    assert "Current Task > first line second line" in output
    assert "first line\n\n" not in output


def test_multiline_new_task_is_collapsed(toolcall_config, capsys):
    """A multi-line task given at the completion prompt must not add blank lines to the display/history."""
    agent = InteractiveAgent(
        model=make_tc_model(
            [
                ("First", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'one'"}]),
                ("Second", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'two'"}]),
            ]
        ),
        env=LocalEnvironment(),
        **{**toolcall_config, "mode": "yolo", "confirm_exit": True, "quiet": True},
    )
    with mock_prompts(["a new\n\n\n\ntask", ""]):
        info = agent.run("initial task")
    output = capsys.readouterr().out
    assert info["submission"] == "two\n"
    assert "The user added a new task: a new task" in output
    assert "a new\n\n" not in output
    new_task = next(m for m in agent.messages if "The user added a new task" in get_text(m))
    assert get_text(new_task) == "The user added a new task: a new task"


# --- /new: starting a new conversation ---


def test_new_conversation_from_completion_prompt(model_factory):
    """`/new` at the completion prompt discards the history and starts fresh."""
    factory, config = model_factory
    with mock_prompts(
        [
            "",  # Confirm first action
            "/new",  # Start a new conversation
            "Brand new task",  # The new task prompted by /new
            "",  # Confirm the new conversation's action
            "",  # No further task after the new conversation completes
        ]
    ):
        agent = InteractiveAgent(
            model=factory(
                [
                    ("First", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'first done'"}]),
                    ("Second", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'second done'"}]),
                ]
            ),
            env=LocalEnvironment(),
            **config,
        )
        info = agent.run("Original task")

    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "second done\n"
    # The counters restart for the new conversation.
    assert agent.n_calls == 1
    contents = [get_text(msg) for msg in agent.messages]
    # The old conversation is gone...
    assert not any("Original task" in c for c in contents)
    assert not any("first done" in c for c in contents)
    # ...and the fresh conversation is present.
    assert any("Brand new task" in c for c in contents)
    # A new conversation is not a continuation of the old one.
    assert not any("The user added a new task" in c for c in contents)


def test_new_conversation_from_confirmation_prompt(model_factory):
    """`/new` can abandon a pending action and start a fresh conversation."""
    factory, config = model_factory
    with mock_prompts(
        [
            "/new",  # At the confirmation prompt: abandon and start over
            "Fresh task",  # The new task
            "",  # Confirm the fresh task's action
            "",  # No further task
        ]
    ):
        agent = InteractiveAgent(
            model=factory(
                [
                    ("First", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'first done'"}]),
                    ("Second", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'fresh done'"}]),
                ]
            ),
            env=LocalEnvironment(),
            **config,
        )
        info = agent.run("Original task")

    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "fresh done\n"
    contents = [get_text(msg) for msg in agent.messages]
    assert not any("Original task" in c for c in contents)
    assert not any("first done" in c for c in contents)
    assert any("Fresh task" in c for c in contents)


def test_new_conversation_with_inline_task(model_factory):
    """The task for the new conversation can be given on the same line as `/new`."""
    factory, config = model_factory
    with mock_prompts(
        [
            "",  # Confirm first action
            "/new Inline task",  # Start a new conversation with an inline task
            "",  # Confirm the new action
            "",  # No further task
        ]
    ):
        agent = InteractiveAgent(
            model=factory(
                [
                    ("First", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'first done'"}]),
                    ("Second", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'inline done'"}]),
                ]
            ),
            env=LocalEnvironment(),
            **config,
        )
        info = agent.run("Original task")

    assert info["submission"] == "inline done\n"
    assert any("Inline task" in get_text(msg) for msg in agent.messages)
    assert not any("Original task" in get_text(msg) for msg in agent.messages)


def test_new_conversation_in_human_mode(model_factory):
    """In human mode `/new` resets the conversation before the user takes over."""
    factory, config = model_factory
    with mock_prompts(
        [
            "/new",  # Human prompt: start fresh
            "Human mode task",  # The new task
            "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'human done'",  # Human command submits
            "",  # No further task
        ]
    ):
        agent = InteractiveAgent(model=factory([]), env=LocalEnvironment(), **{**config, "mode": "human"})
        info = agent.run("Original task")

    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "human done\n"
    assert not any("Original task" in get_text(msg) for msg in agent.messages)
    assert any("Human mode task" in get_text(msg) for msg in agent.messages)


def test_help_command_lists_new(model_factory):
    """The help text advertises the /new command."""
    factory, config = model_factory
    with mock_prompts(["/h", "", ""]):
        with patch("minisweagent.agents.interactive.console.print") as mock_print:
            agent = InteractiveAgent(
                model=factory(
                    [("Finishing", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'"}])]
                ),
                env=LocalEnvironment(),
                **config,
            )
            agent.run("Test help lists /new")
    assert any("/new" in str(call) for call in mock_print.call_args_list)


def test_new_conversation_after_keyboard_interrupt(model_factory):
    """`/new` entered after Ctrl+C still resets to a fresh conversation."""
    factory, config = model_factory
    agent = InteractiveAgent(
        model=factory(
            [
                ("Finishing", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'after interrupt'"}]),
            ]
        ),
        env=LocalEnvironment(),
        **{**config, "mode": "yolo"},
    )
    original_query = agent.query
    calls = {"n": 0}

    def query():
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt
        return original_query()

    agent.query = query
    with mock_prompts(
        [
            "/new",  # After Ctrl+C: start a new conversation
            "Interrupted new task",  # The new task
            "",  # No further task after completion
        ]
    ):
        info = agent.run("Original task")

    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "after interrupt\n"
    assert not any("Original task" in get_text(msg) for msg in agent.messages)
    assert any("Interrupted new task" in get_text(msg) for msg in agent.messages)


# --- /resume: selecting a previous conversation ---


def _save_conversation(
    path: Path,
    *,
    task: str = "Saved task",
    exit_status: str = "",
    api_calls: int = 1,
    mtime: float | None = None,
    finished: bool = False,
) -> Path:
    """Write a trajectory file the way an interrupted (or finished) run leaves it behind."""
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": f"Please solve: {task}"},
        {"role": "assistant", "content": "working on it", "extra": {"actions": [{"command": "echo hi"}]}},
        {"role": "user", "content": "observation"},
    ]
    if finished:
        messages.append(
            {
                "role": "exit",
                "content": "Submitted",
                "extra": {"exit_status": exit_status or "Submitted", "submission": "old submission\n"},
            }
        )
    path.write_text(
        json.dumps(
            {
                "info": {
                    "task": task,
                    "exit_status": exit_status,
                    "model_stats": {"api_calls": api_calls, "instance_cost": 0.0},
                },
                "messages": messages,
            }
        )
    )
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_list_conversations_orders_and_skips_broken_files(tmp_path):
    """Conversations are returned newest-first and broken/non-trajectory files are ignored."""
    from minisweagent.agents.interactive import list_conversations

    now = time.time()
    _save_conversation(tmp_path / "older.traj.json", task="Older task", mtime=now - 100)
    _save_conversation(tmp_path / "newer.traj.json", task="Newer task", mtime=now)
    (tmp_path / "broken.traj.json").write_text("{ this is not json")
    (tmp_path / "notes.txt").write_text("hello")

    conversations = list_conversations(tmp_path)

    assert [c.task for c in conversations] == ["Newer task", "Older task"]
    assert conversations[0].path == tmp_path / "newer.traj.json"


def test_resume_lists_and_continues_selected_conversation(model_factory, tmp_path, capsys):
    """`/resume` shows the saved conversations and continues the one the user picks."""
    factory, config = model_factory
    now = time.time()
    _save_conversation(tmp_path / "older.traj.json", task="Older task", api_calls=1, mtime=now - 100)
    _save_conversation(tmp_path / "newer.traj.json", task="Newer task", api_calls=2, mtime=now)
    with mock_prompts(
        [
            "/resume",  # Confirmation prompt: ask for the list of conversations
            "1",  # Pick the most recent one
            "",  # Confirm the resumed conversation's submitting action
            "",  # No further task
        ]
    ):
        agent = InteractiveAgent(
            model=factory(
                [
                    ("First", [{"command": "echo 'should not run'"}]),  # Consumed before the resume, then discarded
                    ("Second", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'resumed done'"}]),
                ]
            ),
            env=LocalEnvironment(),
            **{**config, "conversation_dir": tmp_path},
        )
        info = agent.run("Original task")

    output = capsys.readouterr().out
    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "resumed done\n"
    contents = [get_text(msg) for msg in agent.messages]
    # The selected (newest) conversation is now the active one...
    assert any("Newer task" in c for c in contents)
    assert not any("Older task" in c for c in contents)
    # ...and the discarded first conversation never ran its command.
    assert not any("should not run" in c for c in contents)
    assert agent.conversation_path == tmp_path / "newer.traj.json"
    # The list actually showed the conversations as selectable options.
    assert "Saved conversations" in output
    assert "Older task" in output and "Newer task" in output


def test_resume_with_inline_selection(model_factory, tmp_path):
    """The conversation number can be given on the same line as `/resume`, skipping the prompt."""
    factory, config = model_factory
    now = time.time()
    _save_conversation(tmp_path / "a.traj.json", task="A task", mtime=now)
    _save_conversation(tmp_path / "b.traj.json", task="B task", mtime=now - 100)
    with mock_prompts(["/resume 2", "", ""]):
        agent = InteractiveAgent(
            model=factory(
                [
                    ("First", [{"command": "echo 'nope'"}]),
                    ("Second", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'inline'"}]),
                ]
            ),
            env=LocalEnvironment(),
            **{**config, "conversation_dir": tmp_path},
        )
        info = agent.run("Original task")

    assert info["submission"] == "inline\n"
    assert any("B task" in get_text(msg) for msg in agent.messages)
    assert not any("A task" in get_text(msg) for msg in agent.messages)


def test_resume_finished_conversation_continues_it(model_factory, tmp_path):
    """Resuming a conversation that already finished lets the user keep going."""
    factory, config = model_factory
    _save_conversation(tmp_path / "finished.traj.json", task="Finished task", exit_status="Submitted", finished=True)
    with mock_prompts(["/resume", "1", "", ""]):
        agent = InteractiveAgent(
            model=factory(
                [
                    ("First", [{"command": "echo 'nope'"}]),  # Consumed before the resume, then discarded
                    ("Second", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'continued'"}]),
                ]
            ),
            env=LocalEnvironment(),
            **{**config, "conversation_dir": tmp_path},
        )
        info = agent.run("Original task")

    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "continued\n"
    assert any("Finished task" in get_text(msg) for msg in agent.messages)


def test_resume_without_saved_conversations_reprompts(model_factory, tmp_path, capsys):
    """`/resume` with an empty conversation directory explains itself and prompts again."""
    factory, config = model_factory
    with mock_prompts(["/resume", "", ""]):
        agent = InteractiveAgent(
            model=factory([("First", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'"}])]),
            env=LocalEnvironment(),
            **{**config, "conversation_dir": tmp_path},
        )
        info = agent.run("Original task")

    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "done\n"
    assert "No saved conversations found" in capsys.readouterr().out


def test_resume_invalid_selection_reprompts(model_factory, tmp_path, capsys):
    """An out-of-range selection is rejected and the user can still continue the run."""
    factory, config = model_factory
    _save_conversation(tmp_path / "saved.traj.json", task="Saved task")
    with mock_prompts(["/resume", "99", "", ""]):
        agent = InteractiveAgent(
            model=factory([("First", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'"}])]),
            env=LocalEnvironment(),
            **{**config, "conversation_dir": tmp_path},
        )
        info = agent.run("Original task")

    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "done\n"
    assert not any("Saved task" in get_text(msg) for msg in agent.messages)
    assert "Invalid selection" in capsys.readouterr().out


def test_conversations_are_archived_to_the_conversation_dir(model_factory, tmp_path):
    """Every conversation is written to the conversation directory so `/resume` can list it."""
    factory, config = model_factory
    with mock_prompts(["", ""]):
        agent = InteractiveAgent(
            model=factory([("First", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'"}])]),
            env=LocalEnvironment(),
            **{**config, "conversation_dir": tmp_path},
        )
        agent.run("Archive me")

    files = list(tmp_path.glob("*.traj.json"))
    assert len(files) == 1
    saved = json.loads(files[0].read_text())
    assert saved["info"]["task"] == "Archive me"
    assert saved["info"]["exit_status"] == "Submitted"


def test_conversation_archived_alongside_the_output_file(model_factory, tmp_path):
    """The regular output file keeps being written, and a copy lands in the conversation directory."""
    factory, config = model_factory
    output = tmp_path / "last_run.traj.json"
    conversation_dir = tmp_path / "conversations"
    with mock_prompts(["", ""]):
        agent = InteractiveAgent(
            model=factory([("First", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'"}])]),
            env=LocalEnvironment(),
            **{**config, "conversation_dir": conversation_dir, "output_path": output},
        )
        agent.run("Archive me twice")

    archived = list(conversation_dir.glob("*.traj.json"))
    assert len(archived) == 1
    assert json.loads(archived[0].read_text())["info"]["task"] == "Archive me twice"
    assert json.loads(output.read_text())["info"]["task"] == "Archive me twice"


def test_help_command_lists_resume(model_factory):
    """The help text advertises the /resume command."""
    factory, config = model_factory
    with mock_prompts(["/h", "", ""]):
        with patch("minisweagent.agents.interactive.console.print") as mock_print:
            agent = InteractiveAgent(
                model=factory(
                    [("Finishing", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'"}])]
                ),
                env=LocalEnvironment(),
                **config,
            )
            agent.run("Test help lists /resume")
    assert any("/resume" in str(call) for call in mock_print.call_args_list)


def test_status_line_shows_context_size(toolcall_config):
    """The `[step N] Current Task >` line is prefixed with the current context size."""
    with mock_prompts(["", ""]):
        agent = InteractiveAgent(
            model=make_tc_model(
                [("First", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'"}])]
            ),
            env=LocalEnvironment(),
            **toolcall_config,
        )
        with patch("minisweagent.agents.interactive.console.print") as mock_print:
            agent.run("Context size test")
    printed = " ".join(str(call) for call in mock_print.call_args_list)
    assert "Current Task >" in printed
    assert "ctx]" in printed


def test_context_tokens_prefers_provider_usage():
    """When the provider reports usage, that value is used directly (no counting)."""
    agent = InteractiveAgent.__new__(InteractiveAgent)
    msg = {
        "role": "assistant",
        "content": "x",
        "extra": {"response": {"usage": {"prompt_tokens": 1500, "completion_tokens": 25, "total_tokens": 1525}}},
    }
    assert agent._context_tokens(msg) == 1525
    assert (
        agent._context_tokens(
            {"role": "assistant", "extra": {"response": {"usage": {"prompt_tokens": 42, "completion_tokens": 8}}}}
        )
        == 50
    )


def test_context_tokens_falls_back_to_counting():
    """Providers that do not report usage fall back to counting the message history."""

    class FakeModel:
        class config:
            model_name = "gpt-4o"

    agent = InteractiveAgent.__new__(InteractiveAgent)
    agent.messages = [{"role": "user", "content": "hello world"}]
    agent.model = FakeModel()
    n = agent._context_tokens({"role": "assistant", "content": "x", "extra": {}})
    assert isinstance(n, int) and n > 0


def test_context_tokens_returns_none_when_uncountable():
    """If usage is missing and counting fails, the status line just omits the size."""

    class FakeModel:
        class config:
            model_name = "gpt-4o"

    agent = InteractiveAgent.__new__(InteractiveAgent)
    agent.messages = [{"role": "user", "content": "hello"}]
    agent.model = FakeModel()
    with patch("litellm.token_counter", side_effect=RuntimeError("nope")):
        assert agent._context_tokens({"role": "assistant", "content": "x", "extra": {}}) is None


def test_format_token_count():
    assert InteractiveAgent._format_token_count(832) == "832"
    assert InteractiveAgent._format_token_count(12_345) == "12.3k"
    assert InteractiveAgent._format_token_count(1_234_567) == "1.2M"
