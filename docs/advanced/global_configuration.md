# Global configuration

!!! abstract "Configuring mini"

    * This guide shows how to configure the `mini` agent's global settings (API keys, default model, etc.).
      Basically anything that is set as environment variables or similar.
    * You should already be familiar with the [quickstart guide](../quickstart.md).
    * For more agent specific settings, see the [yaml configuration file guide](yaml_configuration.md).

!!! tip "Setting up models"

    Setting up models is also covered in the [quickstart guide](../quickstart.md).

## Setting global configuration

All global configuration can be either set as environment variables, or in the `.env` file (the exact location is printed when you run `mini`).
Environment variables take precedence over variables set in the `.env` file.

We provide several helper functions to update the global configuration.

For example, to set the default model and API keys, you can run:

```bash
mini-extra config setup
```

or to update specific settings:

```
mini-extra config set KEY VALUE
# e.g.,
mini-extra config set MSWEA_MODEL_NAME "anthropic/claude-sonnet-4-5-20250929"
mini-extra config set ANTHROPIC_API_KEY "sk-..."
```

To replace your Anthropic API key when starting a run (e.g. when rotating keys), use the `mini` flag:

```bash
mini --tokens "sk-..."
```

which rewrites `ANTHROPIC_API_KEY` in the `.env` file and uses the new key for that run.

or to unset a key:

```bash
mini-extra config unset KEY
# e.g.,
mini-extra config unset ANTHROPIC_API_KEY
```

You can also edit the `.env` file directly and we provide a helper function for that:

```bash
mini-extra config edit
```

To set environment variables (recommended for temporary experimentation or API keys):

```bash
export KEY="value"
# windows:
setx KEY "value"
```

## Models, keys, costs

!!! tip "See also"

    Read the [quickstart guide](../quickstart.md) first—it already covers most of this.

```bash
# Default model name
# (default: not set)
MSWEA_MODEL_NAME="anthropic/claude-sonnet-4-5-20250929"

# Default reasoning effort passed to the model (e.g. "low", "medium", "high")
# (default: provider default)
MSWEA_REASONING_EFFORT="high"
```

To ignore errors from cost tracking checks (for example for free models), set:

```bash
# CAREFUL: This can lead to unmanaged spending!
MSWEA_COST_TRACKING="ignore_errors"
```

To register extra models to litellm (see [local models](../models/local_models.md) for more details), you can either specify the path in the agent file, or set

```bash
LITELLM_MODEL_REGISTRY_PATH="/path/to/your/model/registry.json"
```

Global call limit:

```bash
# Global limit on number of model calls (0 = no limit)
# (default: 0)
MSWEA_GLOBAL_CALL_LIMIT="100"

# Number of retry attempts for model API calls
# (default: 10)
MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT="10"
```

## Notifications

Get an audible alert (the terminal bell, `\a`) when a task finishes:

```bash
# Notification channel used when a task completes: "terminal_bell" or "none"
# (default: "none")
MSWEA_NOTIFY_CHANNEL="terminal_bell"
```

The same option can be set per run in the yaml config (`agent: notify_channel: terminal_bell`) or
with the `mini` flag `--notify-channel terminal_bell`.

## Blocking agent file access

Refuse the agent access to specific files or folders. Any command that references one of these paths is not
executed; instead the agent gets an "Access denied" error explaining that the path is off-limits. The value is a
colon-separated (``:``) list of absolute paths, where a folder also blocks everything inside it:

```bash
# Files/folders the agent is not allowed to access (default: empty = everything allowed)
MSWEA_DISABLE_AGENT_ACCESS="/home/user/.ssh:/home/user/.aws/credentials:/etc/shadow"
```

The same option can be set per run in the yaml config (``environment: disabled_access: ["/home/user/.ssh"]``).
Blocking is currently enforced by the local environment, which is the default.

## Default config files

```bash
# Set a custom directory for agent config files in addition to the builtin ones
# This allows to specify them by names
MSWEA_CONFIG_DIR="/path/to/your/own/config/dir"

# Config path for mini run script
# (default: package_dir / "config" / "mini.yaml")
MSWEA_MINI_CONFIG_PATH="/path/to/your/own/config"

# Custom style path for trajectory inspector
# (default: package_dir / "config" / "inspector.tcss")
MSWEA_INSPECTOR_STYLE_PATH="/path/to/your/inspector/style.tcss"
```

### Settings for environments

```bash
# Path/name to the singularity/apptainer executable
# (default: "singularity")
MSWEA_SINGULARITY_EXECUTABLE="singularity"

# Path/name to the docker executable
# (default: "docker")
MSWEA_DOCKER_EXECUTABLE="docker"

# Path/name to the bubblewrap executable
# (default: "bwrap")
MSWEA_BUBBLEWRAP_EXECUTABLE="bwrap"
```

## Default run files

```bash
# Default run script entry point for the main CLI
# (default: "minisweagent.run.mini")
MSWEA_DEFAULT_RUN="minisweagent.run.mini"
```

{% include-markdown "_footer.md" %}
