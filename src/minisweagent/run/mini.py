#!/usr/bin/env python3

"""Run mini-SWE-agent in your local environment. This is the default executable `mini`."""
# Read this first: https://mini-swe-agent.com/latest/usage/mini/  (usage)

import json
import os
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from minisweagent import __version__, global_config_dir
from minisweagent.agents import get_agent
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.models import get_api_base, get_model, get_model_name, get_reasoning_effort
from minisweagent.run.utilities.config import configure_if_first_time
from minisweagent.utils.serialize import UNSET, recursive_merge

DEFAULT_CONFIG_FILE = Path(os.getenv("MSWEA_MINI_CONFIG_PATH", builtin_config_dir / "mini.yaml"))
DEFAULT_OUTPUT_FILE = global_config_dir / "last_mini_run.traj.json"
DEFAULT_CONVERSATIONS_DIR = global_config_dir / "conversations"


_HELP_TEXT = """Run mini-SWE-agent in your local environment.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/mini/[/bold green]
[/not dim]
"""

_CONFIG_SPEC_HELP_TEXT = """Path to config files, filenames, or key-value pairs.

[bold red]IMPORTANT:[/bold red] [red]If you set this option, the default config file will not be used.[/red]
So you need to explicitly set it e.g., with [bold green]-c mini.yaml <other options>[/bold green]

Multiple configs will be recursively merged.

Examples:

[bold red]-c model.model_kwargs.temperature=0[/bold red] [red]You forgot to add the default config file! See above.[/red]

[bold green]-c mini.yaml -c model.model_kwargs.temperature=0.5[/bold green]

[bold green]-c swebench.yaml agent.mode=yolo[/bold green]
"""

console = Console(highlight=False)
app = typer.Typer(rich_markup_mode="rich")


def _multiline_prompt() -> str:
    """Load the prompt only when a task actually needs to be requested."""
    from minisweagent.agents.utils.prompt_user import _multiline_prompt as prompt

    return prompt()


def _prompt_for_initial_task(conversation_dir: Path | None, mode: str = "confirm") -> tuple[str, Path | None]:
    """Ask the user for the first task, handling the `/h`, `/resume` and `/new` commands.

    Slash commands are handled here (rather than by the agent) because the model is only loaded
    after the user answers, keeping startup cheap. Returns the task to run and, when the user asked
    to resume, the trajectory path to load before running.
    """
    from minisweagent.agents.interactive import print_slash_commands_help, select_conversation

    conversation_dir = Path(conversation_dir) if conversation_dir else None
    while True:
        console.print("[bold yellow]What do you want to do?")
        console.print("[bold yellow]>[/bold yellow] ", end="")
        user_input = _multiline_prompt().strip()
        if user_input in ("/h", "/help"):
            print_slash_commands_help(mode)
            continue
        if user_input == "/resume" or user_input.startswith("/resume "):
            selected = select_conversation(conversation_dir, user_input[len("/resume") :].strip())
            if selected is not None:
                console.print(f"[bold green]Resuming conversation {escape(str(selected))}[/bold green]")
                return "", selected
            continue
        if user_input == "/new" or user_input.startswith("/new "):
            task = user_input[len("/new") :].strip()
            if task:
                console.print("[bold green]Starting a new conversation.[/bold green]")
                return task, None
            continue
        return user_input, None


def _welcome_board(
    model_name: str, config_spec: list[str], agent_config: dict, model_config: dict | None = None
) -> Panel:
    """What this run is about to do: which model, provider URL, configs, mode, and reasoning effort."""
    specs = "\n          ".join(str(spec) for spec in config_spec)
    mode = agent_config.get("mode", "confirm") + (" (quiet)" if agent_config.get("quiet") else "")
    model_config = model_config or {}
    # The effective effort: an explicit model_kwargs entry wins over the top-level shortcut.
    # `main` folds the MSWEA_REASONING_EFFORT default into ``model_config`` before calling us.
    reasoning_effort = (model_config.get("model_kwargs") or {}).get("reasoning_effort") or model_config.get(
        "reasoning_effort"
    )
    # `get_api_base` also checks the OPENAI_* / ANTHROPIC_* environment variables, so the
    # board shows the endpoint that `get_model` will actually use even when it is only set
    # in the global .env file.
    provider_url = get_api_base(model_config)
    provider_line = f"[bold]Provider[/bold]  {provider_url or 'default'}"
    if provider_url:
        provider_line += " [dim](Base URL)[/dim]"
    return Panel(
        f"[bold]Model[/bold]     [green]{model_name}[/green]\n"
        f"{provider_line}\n"
        f"[bold]Reasoning[/bold] {reasoning_effort or 'default'}\n"
        f"[bold]Config[/bold]    {specs}\n"
        f"[bold]Mode[/bold]      {mode}",
        title=f"mini-swe-agent {__version__}",
        title_align="left",
        border_style="green",
        expand=False,
    )


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    model_name: str | None = typer.Option(None, "-m", "--model", help="Model to use",),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use (e.g., 'litellm' or 'minisweagent.models.litellm_model.LitellmModel')", rich_help_panel="Advanced"),
    reasoning_effort: str | None = typer.Option(None, "--reasoning-effort", help="Reasoning effort passed to the model (e.g. 'low', 'medium', 'high')", rich_help_panel="Model"),
    notify_channel: str | None = typer.Option(None, "--notify-channel", help="Where to send a task-completed alert: 'terminal_bell' or 'none'", rich_help_panel="Advanced"),
    agent_class: str | None = typer.Option(None, "--agent-class", help="Agent class to use (e.g., 'interactive' or 'minisweagent.agents.interactive.InteractiveAgent')", rich_help_panel="Advanced"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment class to use (e.g., 'local' or 'minisweagent.environments.local.LocalEnvironment')", rich_help_panel="Advanced"),
    task: str | None = typer.Option(None, "-t", "--task", help="Task/problem statement", show_default=False),
    yolo: bool = typer.Option(True, "--yolo/--no-yolo", "-y", help="Run without confirmation", show_default=False),
    quiet: bool = typer.Option(True, "--quiet/--no-quiet", "-q", help="Hide the system prompt and the observation metadata", show_default=False),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help=_CONFIG_SPEC_HELP_TEXT),
    output: Path | None = typer.Option(DEFAULT_OUTPUT_FILE, "-o", "--output", help="Output trajectory file"),
    exit_immediately: bool = typer.Option(False, "--exit-immediately", help="Exit immediately when the agent wants to finish instead of prompting.", rich_help_panel="Advanced"),
    resume: bool = typer.Option(False, "-r", "--resume", help="Continue an interrupted run from its saved trajectory instead of starting a new task.", rich_help_panel="Advanced"),
    resume_path: Path | None = typer.Argument(None, help="Trajectory to resume from (defaults to the --output file)."),
) -> Any:
    # fmt: on
    configure_if_first_time()

    # Build the config from the command line arguments
    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({
        "run": {
            "task": task or UNSET,
        },
        "agent": {
            "agent_class": agent_class or UNSET,
            "mode": "yolo" if yolo else UNSET,
            "quiet": True if quiet else UNSET,
            "confirm_exit": False if exit_immediately else UNSET,
            "output_path": output or UNSET,
            "notify_channel": notify_channel if isinstance(notify_channel, str) else UNSET,
            "conversation_dir": DEFAULT_CONVERSATIONS_DIR,
        },
        "model": {
            "model_class": model_class or UNSET,
            "model_name": model_name or UNSET,
            "reasoning_effort": reasoning_effort if isinstance(reasoning_effort, str) else UNSET,
        },
        "environment": {
            "environment_class": environment_class or UNSET,
        },
    })
    config = recursive_merge(*configs)

    # A notification channel set at the top level or in the `run` section is forwarded to the
    # agent config, which owns the "task completed" alert; an explicit agent value still wins.
    for _source in (config.get("run"), config):
        if _source and (channel := _source.pop("notify_channel", UNSET)) is not UNSET:
            config.setdefault("agent", {}).setdefault("notify_channel", channel)

    # `MSWEA_REASONING_EFFORT` (e.g. set in the global .env file) is the fallback
    # when neither the config files nor the command line set a reasoning effort.
    model_config = config.setdefault("model", {}) or {}
    config["model"] = model_config
    if (effort := get_reasoning_effort(model_config)) is not None:
        model_config.setdefault("reasoning_effort", effort)

    # Capture the effort requested for *this* invocation (CLI/config/.env) before the
    # resume merge below, which may reintroduce a saved `model_kwargs.reasoning_effort`.
    # Since `model_kwargs` wins over the top-level shortcut, the saved entry would
    # otherwise silently shadow a newly requested effort.
    requested_reasoning_effort = get_reasoning_effort(model_config)

    # Resume when asked explicitly (--resume), when a trajectory path is passed (either as
    # `resume` or as the positional `resume_path`). Note that when `main` is called directly in
    # python, `resume`/`resume_path` are the typer defaults, so check the concrete types.
    resume_file = None
    if isinstance(resume, (str, Path)):
        resume_file = Path(resume)
    elif isinstance(resume_path, Path):
        resume_file = resume_path
    elif resume is True:
        resume_file = output if isinstance(output, Path) else DEFAULT_OUTPUT_FILE
    resume = resume_file is not None
    if resume:
        resume_file = resume_file or DEFAULT_OUTPUT_FILE
        if not resume_file.exists():
            raise typer.BadParameter(f"Cannot resume: trajectory file not found: {resume_file}")
        resume_data = json.loads(resume_file.read_text())
        if not isinstance(resume_data, dict) or "messages" not in resume_data:
            raise typer.BadParameter(f"Cannot resume: not a saved mini-swe-agent trajectory: {resume_file}")
        saved_config = (resume_data.get("info") or {}).get("config") or {}
        # Restore the model/environment/agent used by the run we resume, letting explicit CLI options win.
        config = recursive_merge(
            {
                "model": {**(saved_config.get("model") or {}), "model_class": saved_config.get("model_type", UNSET)},
                "environment": {
                    **(saved_config.get("environment") or {}),
                    "environment_class": saved_config.get("environment_type", UNSET),
                },
                "agent": {**(saved_config.get("agent") or {}), "agent_class": saved_config.get("agent_type", UNSET)},
            },
            config,
        )
        config.setdefault("agent", {})["output_path"] = resume_file

        # Let an effort requested for this run win over the trajectory's saved value,
        # so that e.g. `MSWEA_REASONING_EFFORT` or `--reasoning-effort` takes effect on resume.
        if requested_reasoning_effort is not None:
            merged_model_config = config.setdefault("model", {})
            merged_model_config["reasoning_effort"] = requested_reasoning_effort
            merged_model_config.setdefault("model_kwargs", {})["reasoning_effort"] = requested_reasoning_effort

    console.print(
        _welcome_board(
            get_model_name(config=config.get("model", {})),
            config_spec,
            config.get("agent", {}),
            config.get("model", {}),
        )
    )

    prompted_resume: Path | None = None
    if resume:
        run_task = ""
        console.print(f"Resuming interrupted run from [bold green]'{resume_file}'[/bold green]")
    elif (configured_task := config.get("run", {}).get("task", UNSET)) is not UNSET:
        run_task = configured_task
    else:
        run_task, prompted_resume = _prompt_for_initial_task(
            config.get("agent", {}).get("conversation_dir"), str(config.get("agent", {}).get("mode", "confirm"))
        )

    model = get_model(config=config.get("model", {}))
    env = get_environment(config.get("environment", {}), default_type="local")
    agent = get_agent(model, env, config.get("agent", {}), default_type="interactive")
    if resume:
        agent.load(resume_file)
    elif prompted_resume is not None:
        # Continue the conversation chosen with `/resume` at the initial prompt (`mini` uses the
        # interactive agent, whose `resume_conversation` also re-opens finished conversations).
        getattr(agent, "resume_conversation", agent.load)(prompted_resume)
    agent.run(run_task)
    if (output_path := config.get("agent", {}).get("output_path")):
        console.print(f"Saved trajectory to [bold green]'{output_path}'[/bold green]")
    return agent


if __name__ == "__main__":
    app()
