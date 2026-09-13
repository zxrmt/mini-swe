import json
import logging
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import litellm
from pydantic import BaseModel

from minisweagent.exceptions import FormatError
from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.actions_toolcall import (
    BASH_TOOL,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)
from minisweagent.models.utils.anthropic_utils import _reorder_anthropic_thinking_blocks
from minisweagent.models.utils.cache_control import set_cache_control
from minisweagent.models.utils.openai_multimodal import expand_multimodal_content
from minisweagent.models.utils.retry import retry

logger = logging.getLogger("litellm_model")


@contextmanager
def _suppress_litellm_debug_info():
    """litellm prints provider hints for models that are not in its built-in cost map."""
    previous = litellm.suppress_debug_info
    litellm.suppress_debug_info = True
    try:
        yield
    finally:
        litellm.suppress_debug_info = previous


def _chunk_carries_output(chunk) -> bool:
    """Whether a streaming chunk holds the first generated token.

    Providers wrap the first token differently (``content``, ``reasoning_content`` or
    ``tool_calls``); a leading role-only delta does not count.
    """
    choices = chunk.get("choices") if isinstance(chunk, dict) else getattr(chunk, "choices", None)
    if not choices:
        return False
    choice = choices[0]
    delta = choice.get("delta") if isinstance(choice, dict) else getattr(choice, "delta", None)
    if delta is None:
        return False
    if isinstance(delta, dict):
        return bool(delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"))
    return bool(
        getattr(delta, "content", None)
        or getattr(delta, "reasoning_content", None)
        or getattr(delta, "tool_calls", None)
    )


class LitellmModelConfig(BaseModel):
    model_name: str
    """Model name. Highly recommended to include the provider in the model name, e.g., `anthropic/claude-sonnet-4-5-20250929`."""
    model_kwargs: dict[str, Any] = {}
    """Additional arguments passed to the API."""
    reasoning_effort: str | None = None
    """Reasoning effort to request from the model (e.g. `"low"`, `"medium"`, `"high"`). Shortcut for `model_kwargs["reasoning_effort"]`; an explicit `model_kwargs` entry takes precedence."""
    litellm_model_registry: Path | str | None = os.getenv("LITELLM_MODEL_REGISTRY_PATH")
    """Model registry for cost tracking and model metadata. See the local model guide (https://mini-swe-agent.com/latest/models/local_models/) for more details."""
    set_cache_control: Literal["default_end"] | None = None
    """Set explicit cache control markers, for example for Anthropic models"""
    cost_tracking: Literal["default", "ignore_errors"] = os.getenv("MSWEA_COST_TRACKING", "default")
    """Cost tracking mode for this model. Can be "default" or "ignore_errors" (ignore errors/missing cost info)"""
    format_error_template: str = "{{ error }}"
    """Template used when the LM's output is not in the expected format."""
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    """Template used to render the observation after executing an action."""
    multimodal_regex: str = ""
    """Regex to extract multimodal content. Empty string disables multimodal processing."""


class LitellmModel:
    abort_exceptions: list[type[Exception]] = [
        litellm.exceptions.UnsupportedParamsError,
        litellm.exceptions.NotFoundError,
        litellm.exceptions.PermissionDeniedError,
        litellm.exceptions.ContextWindowExceededError,
        litellm.exceptions.AuthenticationError,
        KeyboardInterrupt,
    ]

    def __init__(self, *, config_class: Callable = LitellmModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        if self.config.reasoning_effort is not None:
            # Keep a single source of truth: forward the shortcut into ``model_kwargs``.
            self.config.model_kwargs.setdefault("reasoning_effort", self.config.reasoning_effort)
        if self.config.litellm_model_registry and Path(self.config.litellm_model_registry).is_file():
            with _suppress_litellm_debug_info():
                litellm.utils.register_model(json.loads(Path(self.config.litellm_model_registry).read_text()))
        self._last_generation_timing: dict | None = None

    def _completion(self, messages: list[dict], *, tools: list[dict] | None = None, **kwargs):
        """Call ``litellm.completion`` with the request envelope shared by every query path."""
        return litellm.completion(
            model=self.config.model_name,
            messages=messages,
            **({"tools": tools} if tools else {}),
            **kwargs,
        )

    def _query(self, messages: list[dict[str, str]], **kwargs):
        tools = kwargs.pop("tools", [BASH_TOOL])
        try:
            response, self._last_generation_timing = self._query_with_timing(messages, tools=tools, **kwargs)
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e
        return response

    def _query_with_timing(self, messages: list[dict[str, str]], *, tools: list[dict] | None = None, **kwargs):
        """Run a completion, measuring time-to-first-token and output speed.

        Streaming is used by default so the first token can be observed. Backends that
        ignore ``stream=True`` (or test doubles that return a complete response) are handled
        transparently: the response is returned unchanged and the timing is ``None``.
        Opt out by setting ``stream: false`` in ``model_kwargs``.

        Returns ``(response, timing)`` where ``timing`` is a dict holding any of
        ``time_to_first_token`` (seconds), ``output_tokens_per_second`` and ``output_tokens``.
        """
        params = dict(self.config.model_kwargs | kwargs)
        if not params.pop("stream", True):
            return self._completion(messages, tools=tools, **params), None
        params.setdefault("stream_options", {"include_usage": True})
        # The stream builder re-derives the provider from the model name echoed back by the
        # API (often without its provider prefix), which makes litellm print its
        # "Provider List" hint on every step. It is only debug noise, so silence it here.
        with _suppress_litellm_debug_info():
            start = time.monotonic()
            stream = self._completion(messages, tools=tools, stream=True, **params)
            chunks: list = []
            first_token_time: float | None = None
            try:
                for chunk in stream:
                    if first_token_time is None and _chunk_carries_output(chunk):
                        first_token_time = time.monotonic() - start
                    chunks.append(chunk)
            except TypeError:
                if chunks:  # a real stream broke midway: let the caller's retry logic handle it
                    raise
                # Not a real stream: a provider or a test double returned a complete response.
                return stream, None
            except Exception:
                if chunks:  # already generated output: don't risk paying for it twice
                    raise
                # The backend rejected streaming (e.g. unsupported `stream_options`) before
                # producing anything, so retry as a plain request without any timing.
                params.pop("stream_options", None)
                return self._completion(messages, tools=tools, **params), None
            if not chunks:
                return stream, None
            response = litellm.stream_chunk_builder(chunks, messages=messages)
            if response is None:
                return stream, None
            end = time.monotonic()
        return response, self._timing_from_stream(response, start, end, first_token_time)

    @staticmethod
    def _timing_from_stream(response, start: float, end: float, first_token_time: float | None) -> dict | None:
        """Turn streaming timestamps into the timing dict stored on the message."""
        usage = getattr(response, "usage", None)
        completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
        timing: dict[str, float] = {}
        if first_token_time is not None:
            timing["time_to_first_token"] = first_token_time
            generation_seconds = end - start - first_token_time
            if completion_tokens and generation_seconds > 0:
                timing["output_tokens_per_second"] = completion_tokens / generation_seconds
        if completion_tokens:
            timing["output_tokens"] = completion_tokens
        return timing or None

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        prepared = [{k: v for k, v in msg.items() if k != "extra"} for msg in messages]
        prepared = _reorder_anthropic_thinking_blocks(prepared)
        return set_cache_control(prepared, mode=self.config.set_cache_control)

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        self._last_generation_timing = None
        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                response = self._query(self._prepare_messages_for_api(messages), **kwargs)
        timing = self._last_generation_timing
        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        # Note: all model.query() implementations must persist the response and cost on FormatError.
        try:
            actions = self._parse_actions(response)
        except FormatError as e:
            e.messages[0]["extra"].update(cost_output)
            try:
                e.messages[0]["extra"]["response"] = response.model_dump(mode="json")
            except Exception:
                # model_dump failed (e.g. unserializable object); fall back to repr
                # so the spec contract ("response MUST be persisted") holds unconditionally.
                e.messages[0]["extra"]["response"] = repr(response)
            raise
        message = response.choices[0].message.model_dump()
        message["extra"] = {
            "actions": actions,
            "response": response.model_dump(),
            **cost_output,
            "timestamp": time.time(),
        }
        if timing:
            message["extra"].update(timing)
        return message

    def query_text(self, messages: list[dict], **kwargs) -> dict:
        """Query the model for a plain-text answer without parsing a bash action.

        Used by the agent to summarize the conversation (``/compact``). Unlike :meth:`query`
        it sends no tools and does not require the reply to contain a tool call.
        """
        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                response = self._query(self._prepare_messages_for_api(messages), tools=[], **kwargs)
        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        message = response.choices[0].message.model_dump()
        message["extra"] = {**cost_output, "response": response.model_dump(), "timestamp": time.time()}
        return message

    def _calculate_cost(self, response) -> dict[str, float]:
        try:
            cost = litellm.cost_calculator.completion_cost(response, model=self.config.model_name)
            if cost <= 0.0:
                raise ValueError(f"Cost must be > 0.0, got {cost}")
        except Exception as e:
            cost = 0.0
            if self.config.cost_tracking != "ignore_errors":
                msg = (
                    f"Error calculating cost for model {self.config.model_name}: {e}, perhaps it's not registered? "
                    "You can ignore this issue from your config file with cost_tracking: 'ignore_errors' or "
                    "globally with export MSWEA_COST_TRACKING='ignore_errors'. "
                    "Alternatively check the 'Cost tracking' section in the documentation at "
                    "https://klieret.short.gy/mini-local-models. "
                    " Still stuck? Please open a github issue at https://github.com/SWE-agent/mini-swe-agent/issues/new/choose!"
                )
                logger.critical(msg)
                raise RuntimeError(msg) from e
        return {"cost": cost}

    def _parse_actions(self, response) -> list[dict]:
        """Parse tool calls from the response. Raises FormatError if unknown tool."""
        tool_calls = response.choices[0].message.tool_calls or []
        return parse_toolcall_actions(
            tool_calls,
            format_error_template=self.config.format_error_template,
            template_kwargs={"finish_reason": response.choices[0].finish_reason},
        )

    def format_message(self, **kwargs) -> dict:
        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Format execution outputs into tool result messages."""
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }
