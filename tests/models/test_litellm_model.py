from unittest.mock import MagicMock, patch

import pytest

from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_toolcall import BASH_TOOL


class TestLitellmModelConfig:
    def test_default_format_error_template(self):
        assert LitellmModelConfig(model_name="test").format_error_template == "{{ error }}"


def _mock_litellm_response(tool_calls):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.tool_calls = tool_calls
    mock_response.choices[0].message.model_dump.return_value = {"role": "assistant", "content": None}
    mock_response.model_dump.return_value = {}
    return mock_response


class TestLitellmModel:
    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_query_includes_bash_tool(self, mock_cost, mock_completion):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "echo test"}'
        tool_call.id = "call_1"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4")
        model.query([{"role": "user", "content": "test"}])

        mock_completion.assert_called_once()
        assert mock_completion.call_args.kwargs["tools"] == [BASH_TOOL]

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_query_text_returns_content_without_tools_or_tool_calls(self, mock_cost, mock_completion):
        """`query_text` asks for a plain-text answer (used by `/compact`) and never requires a tool call."""
        response = _mock_litellm_response(None)
        response.choices[0].message.model_dump.return_value = {"role": "assistant", "content": "a summary"}
        mock_completion.return_value = response
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4")
        message = model.query_text([{"role": "user", "content": "summarize"}])

        assert message["content"] == "a summary"
        assert message["extra"]["cost"] == 0.001
        # No bash tool is sent, so the model is free to reply with plain text.
        assert "tools" not in mock_completion.call_args.kwargs

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_parse_actions_valid_tool_call(self, mock_cost, mock_completion):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "ls -la"}'
        tool_call.id = "call_abc"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4")
        result = model.query([{"role": "user", "content": "list files"}])
        assert result["extra"]["actions"] == [{"command": "ls -la", "tool_call_id": "call_abc"}]

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_parse_actions_no_tool_calls_raises(self, mock_cost, mock_completion):
        mock_completion.return_value = _mock_litellm_response(None)
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4")
        with pytest.raises(FormatError):
            model.query([{"role": "user", "content": "test"}])

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_finish_reason_threaded_into_format_error_template(self, mock_cost, mock_completion):
        """The response finish_reason is exposed to format_error_template via template_kwargs, so a
        config can report a max_tokens truncation instead of the misleading "no tool call" error."""
        response = _mock_litellm_response(None)
        response.choices[0].finish_reason = "length"
        mock_completion.return_value = response
        mock_cost.return_value = 0.001

        model = LitellmModel(
            model_name="gpt-4",
            format_error_template="{% if finish_reason == 'length' %}cut off{% else %}{{ error }}{% endif %}",
        )
        with pytest.raises(FormatError) as exc:
            model.query([{"role": "user", "content": "test"}])
        assert exc.value.messages[0]["content"] == "cut off"

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_reasoning_effort_passed_to_api(self, mock_cost, mock_completion):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "echo test"}'
        tool_call.id = "call_1"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-5", reasoning_effort="high")
        model.query([{"role": "user", "content": "test"}])

        assert mock_completion.call_args.kwargs["reasoning_effort"] == "high"

    def test_reasoning_effort_explicit_model_kwargs_wins(self):
        model = LitellmModel(model_name="gpt-5", reasoning_effort="high", model_kwargs={"reasoning_effort": "low"})
        assert model.config.model_kwargs["reasoning_effort"] == "low"

    def test_reasoning_effort_omitted_by_default(self):
        model = LitellmModel(model_name="gpt-5")
        assert "reasoning_effort" not in model.config.model_kwargs

    def test_format_observation_messages(self):
        model = LitellmModel(model_name="gpt-4", observation_template="{{ output.output }}")
        message = {"extra": {"actions": [{"command": "echo test", "tool_call_id": "call_1"}]}}
        outputs = [{"output": "test output", "returncode": 0}]
        result = model.format_observation_messages(message, outputs)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_1"
        assert result[0]["content"] == "test output"

    def test_format_observation_messages_no_actions(self):
        model = LitellmModel(model_name="gpt-4")
        result = model.format_observation_messages({"extra": {}}, [])
        assert result == []


def _stream_chunks(*deltas):
    return [{"choices": [{"delta": delta}]} for delta in deltas]


def _bash_tool_call():
    tool_call = MagicMock()
    tool_call.function.name = "bash"
    tool_call.function.arguments = '{"command": "echo test"}'
    tool_call.id = "call_1"
    return tool_call


class _FakeClock:
    """Minimal stand-in for the ``time`` module so the model's clock is deterministic."""

    def __init__(self, monotonic_values):
        self._values = iter(monotonic_values)

    def monotonic(self):
        return next(self._values)

    @staticmethod
    def time():
        return 0.0


def test_query_measures_time_to_first_token_and_output_speed():
    """Streaming chunks are timed so the status line can show TTFT and tokens/s."""
    response = _mock_litellm_response([_bash_tool_call()])
    response.usage.completion_tokens = 100
    clock = _FakeClock([0.0, 0.5, 2.0])  # request start, first token, stream end
    with (
        patch("minisweagent.models.litellm_model.litellm.completion") as mock_completion,
        patch("minisweagent.models.litellm_model.litellm.stream_chunk_builder", return_value=response),
        patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost", return_value=0.001),
        patch("minisweagent.models.litellm_model.time", clock),
    ):
        mock_completion.return_value = iter(_stream_chunks({"role": "assistant"}, {"content": "hi"}, {"content": "!"}))
        msg = LitellmModel(model_name="gpt-4").query([{"role": "user", "content": "test"}])

    assert msg["extra"]["time_to_first_token"] == 0.5
    assert msg["extra"]["output_tokens"] == 100
    assert msg["extra"]["output_tokens_per_second"] == pytest.approx(100 / 1.5)


def test_query_without_stream_has_no_timing():
    """`stream: false` in model_kwargs keeps the non-streaming path and reports no timing."""
    response = _mock_litellm_response([_bash_tool_call()])
    with (
        patch("minisweagent.models.litellm_model.litellm.completion", return_value=response) as mock_completion,
        patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost", return_value=0.001),
    ):
        msg = LitellmModel(model_name="gpt-4", model_kwargs={"stream": False}).query(
            [{"role": "user", "content": "test"}]
        )

    assert "time_to_first_token" not in msg["extra"]
    assert "stream" not in mock_completion.call_args.kwargs


def test_query_handles_provider_ignoring_stream():
    """A backend that returns a complete response for stream=True falls back without timing."""
    response = _mock_litellm_response([_bash_tool_call()])
    with (
        patch("minisweagent.models.litellm_model.litellm.completion", return_value=response),
        patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost", return_value=0.001),
    ):
        msg = LitellmModel(model_name="gpt-4").query([{"role": "user", "content": "test"}])

    assert msg["extra"]["actions"] == [{"command": "echo test", "tool_call_id": "call_1"}]
    assert "time_to_first_token" not in msg["extra"]


def _tool_call_chunks(model="custom-model"):
    """A minimal OpenAI-style stream that ends in a single bash tool call."""
    return [
        {
            "id": "1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        },
        {
            "id": "1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "bash", "arguments": '{"command": "echo hi"}'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        },
        {
            "id": "1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        },
    ]


def test_streaming_query_suppresses_litellm_provider_list(capsys):
    """litellm's stream builder must not spam the 'Provider List' hint for unprefixed model names."""
    with (
        patch("minisweagent.models.litellm_model.litellm.completion", return_value=iter(_tool_call_chunks())),
        patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost", return_value=0.001),
    ):
        msg = LitellmModel(model_name="openai/custom-model").query([{"role": "user", "content": "hi"}])

    assert msg["extra"]["actions"] == [{"command": "echo hi", "tool_call_id": "call_1"}]
    assert msg["extra"]["output_tokens"] == 7
    assert "Provider List" not in capsys.readouterr().out
