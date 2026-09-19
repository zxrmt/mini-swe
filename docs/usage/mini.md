# `mini`

!!! abstract "Overview"

    * `mini` is a REPL-style interactive command line interface for using mini-SWE-agent in the local environment (as opposed to workflows that require sandboxing or large scale batch processing).

<figure markdown="span">
  <div class="gif-container gif-container-styled" data-glightbox-disabled>
    <img src="https://github.com/SWE-agent/swe-agent-media/blob/main/media/mini/png/mini.png?raw=true"
         data-gif="https://github.com/SWE-agent/swe-agent-media/blob/main/media/mini/gif/mini.gif?raw=true"
         alt="mini" data-glightbox="false" width="600" />
  </div>
</figure>


## Command line options

Useful switches:

- `-h`/`--help`: Show help
- `-t`/`--task`: Specify a task to run (else you will be prompted)
- `-c`/`--config`: Specify a config file to use, else we will use [`mini.yaml`](https://github.com/swe-agent/mini-swe-agent/blob/main/src/minisweagent/config/mini.yaml) or the config `MSWEA_MINI_CONFIG_PATH` environment variable (see [global configuration](../advanced/global_configuration.md)).
  It's enough to specify the name of the config file, e.g., `-c mini.yaml` (see [global configuration](../advanced/global_configuration.md) for how it is resolved).
- `-m`/`--model`: Specify a model to use, else we will use the model `MSWEA_MODEL_NAME` environment variable (see [global configuration](../advanced/global_configuration.md))
- `--reasoning-effort`: Set the model's reasoning effort (e.g. `low`, `medium`, `high`). Equivalent to `model.reasoning_effort` in the config file (see [model settings](../models/quickstart.md)).
- `--tokens`/`--token`: Replace the `ANTHROPIC_API_KEY` in your global config file (`.env`, see [global configuration](../advanced/global_configuration.md)) with the given key and use it for this run.
  Handy for rotating keys: `mini --tokens sk-...`
- `-y`/`--yolo`: Start in `yolo` mode (see below)
- `-r`/`--resume`: Resume an interrupted run from its saved trajectory instead of starting a new task.
  Resumes the `-o`/`--output` file (or a trajectory passed as a positional argument).

## Resuming an interrupted run

`mini` persists the trajectory after every step, so an interrupted run (Ctrl+C, crash, closed terminal)
can be continued instead of restarted:

```bash
mini --resume                        # continue the last run (the default output file)
mini --resume path/to/run.traj.json  # continue a specific trajectory
mini -r -o path/to/run.traj.json     # same, using --output
```

Resuming restores the message history, the task, the accumulated cost/model calls and the model and
environment configuration, then continues from the last saved step and writes back to the same file.

## Modes of operation

`mini` provides three different modes of operation

- `confirm` (`/c`): The LM proposes an action and the user is prompted to confirm (press Enter) or reject (enter a rejection message)
- `yolo` (`/y`): The action from the LM is executed immediately without confirmation
- `human` (`/u`): The user takes over to type and execute commands

You can switch between the modes with the `/c`, `/y`, and `/u` commands that you can enter any time the agent is waiting for input.
You can also press `Ctrl+C` to interrupt the agent at any time, allowing you to switch between modes.

`mini` starts in `confirm` mode by default. To start in `yolo` mode, you can add `-y`/`--yolo` to the command line.

## Starting a new conversation

By default, adding a new task when a task is completed continues the *same* conversation, so the model
keeps the whole history as context. Type `/new` at any prompt to discard the current conversation and
start over with a fresh task and an empty context. You can pass the task on the same line, e.g.
`/new write a sudoku game`; otherwise `mini` prompts you for it.

## Resuming a previous conversation

Every conversation is saved to a `conversations` directory inside your global config directory after
each step. Type `/resume` at any prompt to list the saved conversations (most recent first, with their
task, status and number of model calls) and pick one by number.
You can also pass the choice on the same line, e.g. `/resume 2`.

Selecting a conversation only *loads* it: `mini` restores the full history, previews the last model
turn from that conversation so you can see where it left off, and then asks you for a message. That
message is sent to the model to continue the conversation, so you decide what happens next instead of
the model being queried automatically.

This differs from `-r`/`--resume` on the command line, which continues a single trajectory file that you
name (or the last run's default output file).

## Compacting the conversation

Long conversations eventually fill the model's context window. Type `/compact` at any prompt to
summarize the conversation with the very same model that `mini` is configured with (so no extra
setup or API key is needed) and replace the older messages with that summary. The task and the
system prompt are kept, so the agent can keep working with a much smaller context. The command is
also listed in the in-session `/h` help.

The summarization prompt is the `agent.compaction_template` config value; override it in your config
file to change the shape of the summary.

## Miscellaneous tips

- `mini` saves the full history of your last run to your global config directory.
  The path to the directory is printed when you start `mini`.

## Implementation

??? note "Default config"

    - [Read on GitHub](https://github.com/swe-agent/mini-swe-agent/blob/main/src/minisweagent/config/mini.yaml)

    ```yaml
    --8<-- "src/minisweagent/config/mini.yaml"
    ```

??? note "Run script"

    - [Read on GitHub](https://github.com/swe-agent/mini-swe-agent/blob/main/src/minisweagent/run/mini.py)
    - [API reference](../reference/run/mini.md)

    ```python
    --8<-- "src/minisweagent/run/mini.py"
    ```

??? note "Agent class"

    - [Read on GitHub](https://github.com/swe-agent/mini-swe-agent/blob/main/src/minisweagent/agents/interactive.py)
    - [API reference](../reference/agents/interactive.md)

    ```python
    --8<-- "src/minisweagent/agents/interactive.py"
    ```

{% include-markdown "../_footer.md" %}