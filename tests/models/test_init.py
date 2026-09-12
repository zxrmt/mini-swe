import os
from unittest.mock import patch

import pytest

from minisweagent.models import GlobalModelStats, get_model, get_model_class, get_model_name
from minisweagent.models.test_models import DeterministicModel, make_output


class TestGetModelName:
    # Common config used across tests - model_name should be direct, not nested under "model"
    CONFIG_WITH_MODEL_NAME = {"model_name": "config-model"}

    def test_input_model_name_takes_precedence(self):
        """Test that explicit input_model_name overrides all other sources."""
        with patch.dict(os.environ, {"MSWEA_MODEL_NAME": "env-model"}):
            assert get_model_name("input-model", self.CONFIG_WITH_MODEL_NAME) == "input-model"

    def test_config_takes_precedence_over_env(self):
        """Test that config takes precedence over environment variable."""
        with patch.dict(os.environ, {"MSWEA_MODEL_NAME": "env-model"}):
            assert get_model_name(None, self.CONFIG_WITH_MODEL_NAME) == "config-model"

    def test_env_var_fallback(self):
        """Test that environment variable is used when no config provided."""
        with patch.dict(os.environ, {"MSWEA_MODEL_NAME": "env-model"}):
            assert get_model_name(None, {}) == "env-model"

    def test_config_fallback(self):
        """Test that config model name is used when input and env are missing."""
        with patch.dict(os.environ, {}, clear=True):
            assert get_model_name(None, self.CONFIG_WITH_MODEL_NAME) == "config-model"

    def test_raises_error_when_no_model_configured(self):
        """Test that ValueError is raised when no model is configured anywhere."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="No default model set. Please run `mini-extra config setup` to set one."
            ):
                get_model_name(None, {})

            with pytest.raises(
                ValueError, match="No default model set. Please run `mini-extra config setup` to set one."
            ):
                get_model_name(None, None)


class TestGetModelClass:
    def test_anthropic_model_selection(self):
        """Test that anthropic-related model names return LitellmModel by default."""
        from minisweagent.models.litellm_model import LitellmModel

        for name in ["anthropic", "sonnet", "opus", "claude-sonnet", "claude-opus"]:
            assert get_model_class(name) == LitellmModel

    def test_litellm_model_fallback(self):
        """Test that non-anthropic model names return LitellmModel."""
        from minisweagent.models.litellm_model import LitellmModel

        for name in ["gpt-4", "gpt-3.5-turbo", "llama2", "random-model"]:
            assert get_model_class(name) == LitellmModel

    def test_partial_matches(self):
        """Test that partial string matches work correctly."""
        from minisweagent.models.litellm_model import LitellmModel

        assert get_model_class("my-anthropic-model") == LitellmModel
        assert get_model_class("sonnet-latest") == LitellmModel
        assert get_model_class("opus-v2") == LitellmModel
        assert get_model_class("gpt-anthropic-style") == LitellmModel
        assert get_model_class("totally-different") == LitellmModel

    def test_litellm_response_model_selection(self):
        """Test that litellm_response model class can be selected."""
        from minisweagent.models.litellm_response_model import LitellmResponseModel

        assert get_model_class("any-model", "litellm_response") == LitellmResponseModel


class TestGetModel:
    def test_config_deep_copy(self):
        """Test that get_model preserves original config via deep copy."""
        original_config = {"model_kwargs": {"api_key": "original"}, "outputs": [make_output("test", [])]}

        with patch("minisweagent.models.get_model_class") as mock_get_class:
            mock_get_class.return_value = lambda **kwargs: DeterministicModel(
                outputs=[make_output("test", [])], model_name="test"
            )
            get_model("test-model", original_config)
            assert original_config["model_kwargs"]["api_key"] == "original"
            assert "model_name" not in original_config

    def test_integration_with_compatible_model(self):
        """Test get_model works end-to-end with a model that handles extra kwargs."""
        with patch("minisweagent.models.get_model_class") as mock_get_class:
            hello_output = make_output("hello", [])

            def compatible_model(**kwargs):
                # Filter to only what DeterministicModel accepts, provide defaults
                config_args = {k: v for k, v in kwargs.items() if k in ["outputs", "model_name"]}
                if "outputs" not in config_args:
                    config_args["outputs"] = [make_output("default", [])]
                return DeterministicModel(**config_args)

            mock_get_class.return_value = compatible_model
            model = get_model("test-model", {"outputs": [hello_output]})
            assert isinstance(model, DeterministicModel)
            assert model.config.outputs == [hello_output]
            assert model.config.model_name == "test-model"

    def test_config_api_key_used_when_no_env_var(self):
        """Test that config api_key is used when env var is not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = {"model_kwargs": {"api_key": "config-key"}, "model_class": "litellm"}
            model = get_model("test-model", config)

            # LitellmModel stores the api_key in model_kwargs
            assert model.config.model_kwargs["api_key"] == "config-key"

    def test_no_api_key_when_none_provided(self):
        """Test that no api_key is set when neither env var nor config provide one."""
        with patch.dict(os.environ, {}, clear=True):
            config = {"model_class": "litellm"}
            model = get_model("test-model", config)

            # LitellmModel should not have api_key when none provided
            model_kwargs = getattr(model.config, "model_kwargs", {})
            assert "api_key" not in model_kwargs

    def test_reasoning_effort_shortcut_moved_into_model_kwargs(self):
        """A top-level reasoning_effort is forwarded through model_kwargs for every model class."""
        config = {"model_class": "litellm", "reasoning_effort": "high", "model_kwargs": {"drop_params": True}}
        model = get_model("test-model", config)
        assert model.config.model_kwargs["reasoning_effort"] == "high"
        # LitellmModelConfig also keeps the top-level value for introspection.
        assert model.config.reasoning_effort == "high"
        # The caller's config must not be mutated.
        assert config == {"model_class": "litellm", "reasoning_effort": "high", "model_kwargs": {"drop_params": True}}

    def test_reasoning_effort_shortcut_does_not_overwrite_model_kwargs(self):
        """An explicit model_kwargs entry wins, allowing provider-specific overrides."""
        config = {"model_class": "litellm", "reasoning_effort": "high", "model_kwargs": {"reasoning_effort": "low"}}
        model = get_model("test-model", config)
        assert model.config.model_kwargs["reasoning_effort"] == "low"

    def test_reasoning_effort_shortcut_works_for_non_litellm_class(self):
        """The shortcut is handled centrally, so it also applies to e.g. OpenRouter."""
        from minisweagent.models.openrouter_model import OpenRouterModel

        model = get_model("@openai/gpt-5", {"model_class": "openrouter", "reasoning_effort": "high"})
        assert isinstance(model, OpenRouterModel)
        assert model.config.model_kwargs["reasoning_effort"] == "high"

    def test_reasoning_effort_env_fallback(self):
        """MSWEA_REASONING_EFFORT provides the effort when the config does not set one."""
        with patch.dict(os.environ, {"MSWEA_REASONING_EFFORT": "high"}):
            model = get_model("test-model", {"model_class": "litellm"})
            assert model.config.model_kwargs["reasoning_effort"] == "high"

    def test_reasoning_effort_config_takes_precedence_over_env(self):
        """An explicit config effort wins over MSWEA_REASONING_EFFORT."""
        with patch.dict(os.environ, {"MSWEA_REASONING_EFFORT": "high"}):
            model = get_model("test-model", {"model_class": "litellm", "reasoning_effort": "low"})
            assert model.config.model_kwargs["reasoning_effort"] == "low"

    def test_reasoning_effort_env_works_for_non_litellm_class(self):
        """The env fallback is handled centrally, so it also applies to e.g. OpenRouter."""
        from minisweagent.models.openrouter_model import OpenRouterModel

        with patch.dict(os.environ, {"MSWEA_REASONING_EFFORT": "high"}):
            model = get_model("@openai/gpt-5", {"model_class": "openrouter"})
            assert isinstance(model, OpenRouterModel)
            assert model.config.model_kwargs["reasoning_effort"] == "high"

    def test_opencode_go_session_header_added(self):
        """OpenCode Go endpoints get an x-opencode-session header automatically."""
        with patch.dict(os.environ, {"OPENAI_API_BASE": "https://opencode.ai/zen/go/v1"}, clear=True):
            model = get_model("openai/deepseek-flash", {"model_class": "litellm"})
        assert model.config.model_kwargs["extra_headers"]["x-opencode-session"].startswith("mini-swe-agent-")

    @pytest.mark.parametrize(
        ("env_var", "api_base"),
        [
            ("OPENAI_BASE_URL", "https://opencode.ai/zen/go/v1"),
            ("OPENAI_API_BASE", "https://opencode.ai/zen/go/v1"),
            ("ANTHROPIC_BASE_URL", "https://opencode.ai/zen/go/v1"),
        ],
    )
    def test_opencode_go_session_header_for_each_base_url_env(self, env_var, api_base):
        """The header is injected whichever env var litellm reads the Go base URL from."""
        with patch.dict(os.environ, {env_var: api_base}, clear=True):
            model = get_model("some-model", {"model_class": "litellm"})
        assert "x-opencode-session" in model.config.model_kwargs["extra_headers"]

    def test_opencode_go_session_header_from_model_kwargs_api_base(self):
        """An explicit model_kwargs.api_base pointing at Go also triggers the header."""
        with patch.dict(os.environ, {}, clear=True):
            model = get_model(
                "some-model",
                {"model_class": "litellm", "model_kwargs": {"api_base": "https://opencode.ai/zen/go/v1"}},
            )
        assert "x-opencode-session" in model.config.model_kwargs["extra_headers"]

    @pytest.mark.parametrize(
        ("env_var", "api_base"),
        [
            ("OPENAI_BASE_URL", "https://api.deepseek.com"),
            ("OPENAI_BASE_URL", "https://opencode.ai/zen/v1"),  # zen, but not the Go router
            ("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
        ],
    )
    def test_no_session_header_for_other_providers(self, env_var, api_base):
        """Regular endpoints must not get the header injected."""
        with patch.dict(os.environ, {env_var: api_base}, clear=True):
            model = get_model("openai/some-model", {"model_class": "litellm"})
        assert "extra_headers" not in model.config.model_kwargs

    def test_opencode_go_user_session_header_wins(self):
        """A user-provided session header is preserved (e.g. restored when resuming a run)."""
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://opencode.ai/zen/go/v1"}, clear=True):
            model = get_model(
                "openai/deepseek-flash",
                {
                    "model_class": "litellm",
                    "model_kwargs": {"extra_headers": {"x-opencode-session": "my-session"}},
                },
            )
        assert model.config.model_kwargs["extra_headers"]["x-opencode-session"] == "my-session"

    def test_get_deterministic_model(self):
        """Test that get_model can instantiate DeterministicModel via model_class parameter."""
        outputs = [make_output("hello", []), make_output("world", [])]
        config = {"outputs": outputs, "cost_per_call": 2.0}
        model = get_model("test-model", config | {"model_class": "deterministic"})

        assert isinstance(model, DeterministicModel)
        assert model.config.outputs == outputs
        assert model.config.cost_per_call == 2.0
        assert model.config.model_name == "test-model"


class TestGlobalModelStats:
    def test_prints_call_limit_when_set(self, capsys):
        """Test that call limit is printed when MSWEA_GLOBAL_CALL_LIMIT is set."""
        with patch.dict(os.environ, {"MSWEA_GLOBAL_CALL_LIMIT": "10"}, clear=True):
            GlobalModelStats()
            captured = capsys.readouterr()
            assert "Global call limit: 10" in captured.out

    def test_no_print_when_silent_startup_set(self, capsys):
        """Test that limits are not printed when MSWEA_SILENT_STARTUP is set."""
        with patch.dict(
            os.environ,
            {"MSWEA_GLOBAL_CALL_LIMIT": "10", "MSWEA_SILENT_STARTUP": "1"},
            clear=True,
        ):
            GlobalModelStats()
            captured = capsys.readouterr()
            assert "Global call limit" not in captured.out

    def test_no_print_when_no_limits_set(self, capsys):
        """Test that nothing is printed when no limits are set."""
        with patch.dict(os.environ, {}, clear=True):
            GlobalModelStats()
            captured = capsys.readouterr()
            assert "Global call limit" not in captured.out

    def test_no_print_when_limits_are_zero(self, capsys):
        """Test that nothing is printed when limits are explicitly set to zero."""
        with patch.dict(os.environ, {"MSWEA_GLOBAL_CALL_LIMIT": "0"}, clear=True):
            GlobalModelStats()
            captured = capsys.readouterr()
            assert "Global call limit" not in captured.out
