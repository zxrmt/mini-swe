"""This file provides convenience functions for selecting models.
You can ignore this file completely if you explicitly set your model in your run script.
"""

import copy
import importlib
import os
import threading
import uuid

from minisweagent import Model


class GlobalModelStats:
    """Global model statistics tracker with optional call limit."""

    def __init__(self):
        self._cost = 0.0
        self._n_calls = 0
        self._lock = threading.Lock()
        self.call_limit = int(os.getenv("MSWEA_GLOBAL_CALL_LIMIT", "0"))
        if self.call_limit > 0 and not os.getenv("MSWEA_SILENT_STARTUP"):
            print(f"Global call limit: {self.call_limit}")

    def add(self, cost: float) -> None:
        """Add a model call with its cost, checking the call limit."""
        with self._lock:
            self._cost += cost
            self._n_calls += 1
        if 0 < self.call_limit < self._n_calls + 1:
            raise RuntimeError(f"Global call limit exceeded: {self._n_calls}")

    @property
    def cost(self) -> float:
        return self._cost

    @property
    def n_calls(self) -> int:
        return self._n_calls


GLOBAL_MODEL_STATS = GlobalModelStats()


def get_reasoning_effort(config: dict | None = None) -> str | None:
    """Resolve the reasoning effort for a model config.

    Precedence: an explicit ``model_kwargs.reasoning_effort``, then the top-level
    ``reasoning_effort`` shortcut, then the ``MSWEA_REASONING_EFFORT`` environment
    variable, so the global ``.env`` file can provide a default (just like
    ``MSWEA_MODEL_NAME``).
    """
    config = config or {}
    return (
        (config.get("model_kwargs") or {}).get("reasoning_effort")
        or config.get("reasoning_effort")
        or os.getenv("MSWEA_REASONING_EFFORT")
    )


def get_model(input_model_name: str | None = None, config: dict | None = None) -> Model:
    """Get an initialized model object from any kind of user input or settings."""
    resolved_model_name = get_model_name(input_model_name, config)
    if config is None:
        config = {}
    config = copy.deepcopy(config)
    config["model_name"] = resolved_model_name

    # ``reasoning_effort`` can be set as a top-level model key as a shortcut for
    # ``model_kwargs.reasoning_effort``. Every model class forwards ``model_kwargs``
    # to its backend, so this works for all of them. An explicit ``model_kwargs``
    # entry wins, which allows provider-specific overrides.
    if (reasoning_effort := get_reasoning_effort(config)) is not None:
        config.setdefault("reasoning_effort", reasoning_effort)
        model_kwargs = config.get("model_kwargs") or {}
        model_kwargs.setdefault("reasoning_effort", reasoning_effort)
        config["model_kwargs"] = model_kwargs

    # OpenCode's Go router (https://opencode.ai/docs/go/) rejects requests that don't carry a
    # stable `x-opencode-session` header, so inject one per conversation (user value wins).
    api_base = (
        (config.get("model_kwargs") or {}).get("api_base")
        or os.getenv("OPENAI_API_BASE")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("ANTHROPIC_BASE_URL")
        or ""
    )
    if "opencode.ai" in api_base and "/go" in api_base:
        config.setdefault("model_kwargs", {}).setdefault("extra_headers", {}).setdefault(
            "x-opencode-session", f"mini-swe-agent-{uuid.uuid4().hex}"
        )

    model_class = get_model_class(resolved_model_name, config.pop("model_class", ""))

    if (
        any(s in resolved_model_name.lower() for s in ["anthropic", "sonnet", "opus", "claude"])
        and "set_cache_control" not in config
    ):
        # Select cache control for Anthropic models by default
        config["set_cache_control"] = "default_end"

    return model_class(**config)


def get_model_name(input_model_name: str | None = None, config: dict | None = None) -> str:
    """Get a model name from any kind of user input or settings."""
    if config is None:
        config = {}
    if input_model_name:
        return input_model_name
    if from_config := config.get("model_name"):
        return from_config
    if from_env := os.getenv("MSWEA_MODEL_NAME"):
        return from_env
    raise ValueError("No default model set. Please run `mini-extra config setup` to set one.")


_MODEL_CLASS_MAPPING = {
    "litellm": "minisweagent.models.litellm_model.LitellmModel",
    "litellm_textbased": "minisweagent.models.litellm_textbased_model.LitellmTextbasedModel",
    "litellm_response": "minisweagent.models.litellm_response_model.LitellmResponseModel",
    "openrouter": "minisweagent.models.openrouter_model.OpenRouterModel",
    "openrouter_textbased": "minisweagent.models.openrouter_textbased_model.OpenRouterTextbasedModel",
    "openrouter_response": "minisweagent.models.openrouter_response_model.OpenRouterResponseModel",
    "portkey": "minisweagent.models.portkey_model.PortkeyModel",
    "portkey_response": "minisweagent.models.portkey_response_model.PortkeyResponseAPIModel",
    "requesty": "minisweagent.models.requesty_model.RequestyModel",
    "deterministic": "minisweagent.models.test_models.DeterministicModel",
}


def get_model_class(model_name: str, model_class: str = "") -> type:
    """Select the best model class.

    If a model_class is provided (as shortcut name, or as full import path,
    e.g., "anthropic" or "minisweagent.models.anthropic.AnthropicModel"),
    it takes precedence over the `model_name`.
    Otherwise, the model_name is used to select the best model class.
    """
    if model_class:
        full_path = _MODEL_CLASS_MAPPING.get(model_class, model_class)
        try:
            module_name, class_name = full_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (ValueError, ImportError, AttributeError):
            msg = f"Unknown model class: {model_class} (resolved to {full_path}, available: {_MODEL_CLASS_MAPPING})"
            raise ValueError(msg)

    # Default to LitellmModel
    from minisweagent.models.litellm_model import LitellmModel

    return LitellmModel
