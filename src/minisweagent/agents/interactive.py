"""A small generalization of the default agent that puts the user in the loop.

There are three modes:
- human: commands issued by the user are executed immediately
- confirm: commands issued by the LM but not whitelisted are confirmed by the user
- yolo: commands issued by the LM are executed immediately without confirmation
"""

import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NoReturn

from rich.console import Console
from rich.markup import escape
from rich.rule import Rule

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.agents.utils.prompt_user import _multiline_prompt, prompt_session
from minisweagent.exceptions import InterruptAgentFlow, LimitsExceeded, Submitted, TimeExceeded, UserInterruption
from minisweagent.models.utils.content_string import get_content_string, get_reasoning_string

console = Console(highlight=False)

# Legacy consoles (cp1252 & friends) cannot encode the markers, so fall back to ASCII there.
BULLET, ELBOW, ELLIPSIS = (
    ("●", "⎿", "…") if (sys.stdout.encoding or "").lower().replace("-", "").startswith("utf") else ("*", "\\_", "...")
)


class NewConversation(InterruptAgentFlow):
    """Raised to discard the current conversation and start a fresh one with a new task."""

    def __init__(self, task: str):
        self.task = task
        super().__init__()


class ResumeConversation(InterruptAgentFlow):
    """Raised to load a previously saved conversation and continue it."""

    def __init__(self, path: Path):
        self.path = path
        super().__init__()


@dataclass
class SavedConversation:
    """A trajectory file that `/resume` can offer to the user."""

    path: Path
    task: str = ""
    exit_status: str = ""
    api_calls: int = 0
    mtime: float = 0.0


def list_conversations(directory: Path) -> list[SavedConversation]:
    """Return the saved conversations in ``directory``, most recently modified first."""
    conversations = []
    for path in directory.glob("*.traj.json"):
        try:
            info = json.loads(path.read_text()).get("info") or {}
        except (json.JSONDecodeError, OSError):  # unreadable file: skip instead of crashing /resume
            continue
        stats = info.get("model_stats") or {}
        conversations.append(
            SavedConversation(
                path=path,
                task=str(info.get("task") or ""),
                exit_status=str(info.get("exit_status") or ""),
                api_calls=int(stats.get("api_calls") or 0),
                mtime=path.stat().st_mtime,
            )
        )
    return sorted(conversations, key=lambda c: c.mtime, reverse=True)


def select_conversation(conversation_dir: Path | None, selection: str = "") -> Path | None:
    """List the saved conversations in ``conversation_dir`` and let the user pick one.

    ``selection`` is the optional choice typed on the same line as the command (``/resume 2``);
    when it is empty the user is prompted. Returns the chosen path, or ``None`` if the user
    cancelled or picked an invalid entry.
    """
    conversations = list_conversations(conversation_dir) if conversation_dir else []
    if not conversations:
        console.print("[bold yellow]No saved conversations found.[/bold yellow]")
        return None
    console.print("[bold]Saved conversations:[/bold]")
    for i, conversation in enumerate(conversations, 1):
        task = " ".join(conversation.task.split())[:60] or "(no task)"
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(conversation.mtime))
        status = conversation.exit_status or "in progress"
        console.print(
            f"  [bold green]{i}[/bold green]  [dim]{when}[/dim]  {escape(status)}  "
            f"[dim]{conversation.api_calls} steps[/dim]  {escape(task)}"
        )
    if not selection:
        console.print("[bold yellow]Select a conversation (number, Enter to cancel)[/bold yellow]")
        console.print("[bold yellow]>[/bold yellow] ", end="")
        selection = prompt_session.prompt("").strip()
    try:
        index = int(selection)
    except ValueError:
        index = 0
    if 1 <= index <= len(conversations):
        return conversations[index - 1].path
    if selection:
        console.print(f"[bold red]Invalid selection: {escape(selection)}[/bold red]")
    return None


def _format_action_line(command: str) -> str:
    """One-line command summary, elided to fit the terminal."""
    first_line, _, rest = command.partition("\n")
    limit = max(console.width - 10, 20)  # leave room for the bullet and "Bash(...)"
    return f"Bash({first_line[:limit]}{ELLIPSIS if rest or len(first_line) > limit else ''})"


def _observation_rows(content: str, max_lines: int) -> tuple[list[str], bool]:
    """Rows of a result block, and whether any were dropped.

    Long lines are chunked to the terminal width first, so ``max_lines`` bounds the rows
    that actually end up on screen rather than the newlines in the output.
    """
    width = max(console.width - 5, 20)  # the elbow marker and the continuation indent
    rows = [
        line[i : i + width] or ""
        for line in content.splitlines() or ["(No output)"]
        for i in range(0, max(len(line), 1), width)
    ]
    if 0 < max_lines < len(rows):
        return rows[:max_lines] + [f"{ELLIPSIS} +{len(rows) - max_lines} lines"], True
    return rows, False


class InteractiveAgentConfig(AgentConfig):
    mode: Literal["human", "confirm", "yolo"] = "confirm"
    """Whether to confirm actions."""
    whitelist_actions: list[str] = []
    """Never confirm actions that match these regular expressions."""
    confirm_exit: bool = True
    """If the agent wants to finish, do we ask for confirmation from user?"""
    quiet: bool = False
    """Hide the system/instance prompts and print observations without the returncode/key wrappers."""
    observation_display_lines: int = 3
    """Only display this many lines of every command output (0 = all). The model still sees all of it."""
    conversation_dir: Path | None = None
    """If set, save every conversation here and let `/resume` list them."""


class InteractiveAgent(DefaultAgent):
    _MODE_COMMANDS_MAPPING = {"/u": "human", "/c": "confirm", "/y": "yolo"}

    def __init__(self, *args, config_class=InteractiveAgentConfig, **kwargs):
        super().__init__(*args, config_class=config_class, **kwargs)
        self.cost_last_confirmed = 0.0
        self.conversation_path: Path | None = None
        self._awaiting_resume_message = False

    def run(self, task: str = "", **kwargs) -> dict:
        if not self.messages:  # fresh conversation: give it its own file so `/resume` can find it later
            self.conversation_path = self._new_conversation_path(task)
        return super().run(task, **kwargs)

    def save(self, path: Path | None = None, *extra_dicts) -> dict:
        # Write the regular output file, but also keep a copy in the conversations directory.
        data = super().save(path, *extra_dicts)
        primary = Path(path or self.config.output_path) if (path or self.config.output_path) else None
        if self.conversation_path and primary != self.conversation_path:
            self.conversation_path.parent.mkdir(parents=True, exist_ok=True)
            self.conversation_path.write_text(json.dumps(data, indent=2))
        return data

    def _interrupt(self, content: str, *, itype: str = "UserInterruption") -> NoReturn:
        raise UserInterruption({"role": "user", "content": content, "extra": {"interrupt_type": itype}})

    def add_messages(self, *messages: dict) -> list[dict]:
        # Extend supermethod to print messages
        for msg in messages:
            if self.config.quiet and not self.messages:  # system prompt & instance template
                continue
            self._print_message(msg)
        return super().add_messages(*messages)

    @staticmethod
    def _format_token_count(n: int) -> str:
        """Compact human-readable token count, e.g. ``832``, ``12.3k``, ``1.2M``."""
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n / 1_000:.1f}k"
        return str(n)

    @staticmethod
    def _format_output_speed(tokens_per_second: float) -> str:
        """Compact output speed for the status line, e.g. ``45 token/s``."""
        return f"{tokens_per_second:.0f} token/s"

    @staticmethod
    def _format_first_token_time(seconds: float) -> str:
        """Compact time-to-first-token for the status line, e.g. ``fTTFT 1.2s``."""
        return f"fTTFT {seconds:.1f}s"

    @staticmethod
    def _timing_stats(msg: dict) -> tuple[float | None, float | None]:
        """Return ``(time_to_first_token, output_tokens_per_second)`` for the status line.

        Timing is recorded by the model (``msg["extra"]``) when it streams the response;
        either value is ``None`` when the model did not report it.
        """
        extra = msg.get("extra") or {}
        timing = extra.get("timing") or extra
        ttft = timing.get("time_to_first_token")
        speed = timing.get("output_tokens_per_second")
        return (float(ttft) if ttft is not None else None, float(speed) if speed is not None else None)

    def _context_tokens(self, msg: dict) -> int | None:
        """Best-effort size of the current context (in tokens) for the status line.

        Prefers the provider-reported usage of the call that produced ``msg`` (authoritative
        and cheap). Falls back to counting the message history for providers that do not
        report usage. Returns ``None`` when neither is available.
        """
        usage = (msg.get("extra") or {}).get("response", {}).get("usage") or {}
        total = usage.get("total_tokens") or (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)
        if total:
            return int(total)
        try:
            import litellm

            from minisweagent.models.utils.actions_toolcall import BASH_TOOL

            # Providers differ in message format (e.g. Responses API `input_text` blocks), so
            # count a provider-neutral text rendering instead of the raw, provider-specific messages.
            messages = [{"role": m.get("role") or "assistant", "content": get_content_string(m)} for m in self.messages]
            return int(litellm.token_counter(model=self.model.config.model_name, messages=messages, tools=[BASH_TOOL]))
        except Exception:
            return None

    @staticmethod
    def _message_role(msg: dict) -> str:
        """Classify a message for display.

        Responses API messages (`get_content_string` already understands their ``output`` array)
        carry no ``role``/``type``; treat them as the assistant turn they are.
        """
        if role := msg.get("role"):
            return role
        if msg.get("object") == "response" or "output" in msg:
            return "assistant"
        return msg.get("type", "unknown")

    def _print_message(self, msg: dict) -> None:
        extra = msg.get("extra", {})
        reasoning = get_reasoning_string(msg)
        content = get_content_string(msg, quiet=self.config.quiet, skip_tool_calls=True, include_reasoning=False)
        if "returncode" in extra:  # command output belongs under the command that produced it
            rows, truncated = _observation_rows(content, self.config.observation_display_lines)
            failed = extra.get("returncode") or extra.get("exception_info")
            console.print(f"  [{'red' if failed else 'yellow' if truncated else 'green'}]{ELBOW}[/]  ", end="")
            console.print("\n     ".join(rows), markup=False)
            return
        if (role := self._message_role(msg)) == "assistant":
            task = " ".join(str(self.extra_template_vars.get("task", "")).split())[:100]
            context = self._context_tokens(msg)
            headline = escape(f"[step {self.n_calls}] Current Task >")
            parts = [f"[green]{BULLET}[/green]"]
            if context is not None:
                parts.append(f"[bold green]{escape(f'[{self._format_token_count(context)} ctx]')}[/bold green]")
            ttft, speed = self._timing_stats(msg)
            if speed is not None:
                parts.append(f"[bold green]{escape(f'({self._format_output_speed(speed)})')}[/bold green]")
            if ttft is not None:
                parts.append(f"[bold green]{escape(f'({self._format_first_token_time(ttft)})')}[/bold green]")
            parts.append(f"[bold green]{headline}[/bold green]")
            if task:
                parts.append(f"[dim cyan]{escape(task)}[/]")
            console.print("\n" + " ".join(parts), soft_wrap=True)
        else:
            console.print(f"\n[bold green]{BULLET}[/bold green] [bold green]{role.capitalize()}[/bold green]")
        if reasoning:
            console.print(reasoning, markup=False, style="dim grey50")
        if content:
            console.print(content, markup=False)
        for action in extra.get("actions", []):
            if "tool_call_id" in action:  # text-based models already show the command in their reasoning
                console.print(f"\n[green]{BULLET}[/green] ", end="")
                console.print(_format_action_line(action["command"]), markup=False)

    def query(self) -> dict:
        # Extend supermethod to handle human mode
        if self._awaiting_resume_message:
            # `/resume` only loads a conversation: let the user type the message that continues it
            # rather than querying the model straight away.
            self._awaiting_resume_message = False
            self.add_messages({"role": "user", "content": self._prompt_for_resume_message()})
        if self.config.mode == "human":
            match command := self._prompt_and_handle_slash_commands("[bold yellow]>[/bold yellow] "):
                case "/y" | "/c":
                    pass
                case _:
                    msg = {
                        "role": "user",
                        "content": f"User command: \n```bash\n{command}\n```",
                        "extra": {"actions": [{"command": command}]},
                    }
                    self.add_messages(msg)
                    return msg
        try:
            with console.status("Waiting for the LM to respond..."):
                return super().query()
        except TimeExceeded:
            # A wall-clock limit can't be lifted by raising the step limit
            # (the next query re-checks the clock and raises again), so prompting
            # would loop forever. Always stop cleanly instead.
            raise
        except LimitsExceeded:
            if not self._stdin_is_interactive():
                # No terminal to prompt for new limits -- e.g. an unattended
                # `--yolo` run, or any run with stdin redirected from /dev/null
                # (CI, a sandbox). Stop cleanly so the trajectory is saved with a
                # LimitsExceeded exit status, instead of crashing on EOFError when
                # reading input.
                raise
            console.print(
                f"Limits exceeded. Limits: {self.config.step_limit} steps.\n"
                f"Current spend: {self.n_calls} steps, ${self.cost:.2f}."
            )
            self.config.step_limit = int(input("New step limit: "))
            return super().query()

    @staticmethod
    def _stdin_is_interactive() -> bool:
        """Whether an interactive terminal is available to prompt the user.

        Returns False for unattended runs (e.g. `--yolo` in CI, or inside a
        sandbox with stdin redirected from /dev/null), where calling ``input()``
        would raise ``EOFError`` and crash the run.
        """
        try:
            return sys.stdin is not None and sys.stdin.isatty()
        except (ValueError, OSError):
            return False

    def step(self) -> list[dict]:
        # Override the step method to handle user interruption and new conversations
        try:
            console.print(Rule())
            return super().step()
        except NewConversation as e:
            return self._start_new_conversation(e.task)
        except ResumeConversation as e:
            return self.resume_conversation(e.path)
        except KeyboardInterrupt:
            try:
                interruption_message = self._prompt_and_handle_slash_commands(
                    "\n\n[bold yellow]Interrupted.[/bold yellow] "
                    "[green]Type a comment/command[/green] (/h for available commands)"
                    "\n[bold yellow]>[/bold yellow] "
                ).strip()
            except NewConversation as e:
                return self._start_new_conversation(e.task)
            except ResumeConversation as e:
                return self.resume_conversation(e.path)
            if not interruption_message or interruption_message in self._MODE_COMMANDS_MAPPING:
                interruption_message = "Temporary interruption caught."
            self._interrupt(f"Interrupted by user: {interruption_message}")

    def execute_actions(self, message: dict) -> list[dict]:
        # Override to handle user confirmation and confirm_exit, with try/finally to preserve partial outputs
        actions = message.get("extra", {}).get("actions", [])
        commands = [action["command"] for action in actions]
        outputs = []
        submitted = None
        try:
            self._ask_confirmation_or_interrupt(commands)
            for action in actions:
                outputs.append(self.env.execute(action))
        except Submitted as e:
            submitted = e
        finally:
            result = self.add_messages(
                *self.model.format_observation_messages(message, outputs, self.get_template_vars())
            )
        if submitted is not None:
            # Record and print the submission *before* asking what to do next, so the user
            # actually sees the final output of the task instead of only the "Task Completed" prompt.
            self.add_messages(*submitted.messages)
            self._check_for_new_task_or_submit(submitted)
        return result

    def _add_observation_messages(self, message: dict, outputs: list[dict]) -> list[dict]:
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))

    def _start_new_conversation(self, task: str) -> list[dict]:
        """Discard the current conversation and begin a fresh one for ``task``."""
        self.messages = []
        self.cost = 0.0
        self.n_calls = 0
        self.n_consecutive_format_errors = 0
        self.extra_template_vars = {"task": task}
        self._start_time = time.time()
        self._awaiting_resume_message = False
        self.conversation_path = self._new_conversation_path(task)
        return self.add_messages(
            self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
            self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
        )

    def _new_conversation_path(self, task: str) -> Path | None:
        """Where to archive a fresh conversation, so that `/resume` can find it later."""
        if not self.config.conversation_dir:
            return None
        self.config.conversation_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^A-Za-z0-9]+", "-", task).strip("-").lower()[:40] or "conversation"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = self.config.conversation_dir / f"{stamp}-{slug}.traj.json"
        counter = 2
        while path.exists():
            path = self.config.conversation_dir / f"{stamp}-{slug}-{counter}.traj.json"
            counter += 1
        return path

    def resume_conversation(self, path: Path) -> list[dict]:
        """Load a saved conversation and continue it in place of the current one.

        The model is not queried until the user types a message, so picking a conversation
        with `/resume` only loads it.
        """
        output_path = self.config.output_path
        self.load(path)
        # `load` retargets the output file to the loaded trajectory; keep the session's own output
        # file too (the resumed conversation is still archived under `conversation_path`).
        self.config.output_path = output_path
        self.conversation_path = path
        self._drop_exit_message()  # the saved run was finished: drop the exit so that run() keeps going
        self._print_resumed_conversation_preview()  # show where the conversation left off
        self._awaiting_resume_message = True  # wait for the user instead of querying the model
        return self.messages

    def _print_resumed_conversation_preview(self) -> None:
        """Show the tail of a just-resumed conversation so the user can preview where it left off."""
        task = " ".join(str(self.extra_template_vars.get("task", "")).split())
        title = "[bold]Resumed conversation[/bold]" + (f" [dim]- {escape(task)}[/dim]" if task else "")
        console.print(Rule(title))
        # The last model turn and everything after it (the observations it produced).
        last_assistant = max(
            (i for i, msg in enumerate(self.messages) if self._message_role(msg) == "assistant"), default=-1
        )
        for msg in self.messages[last_assistant:]:
            self._print_message(msg)

    def _select_conversation(self, selection: str) -> Path | None:
        return select_conversation(self.config.conversation_dir, selection)

    def _check_for_new_task_or_submit(self, e: Submitted) -> None:
        """Ask the user whether to add a new task (the submission has already been recorded and shown)."""
        if not self.config.confirm_exit:
            return None
        message = (
            "[bold yellow]Task Completed[/bold yellow] ([bold]/h[/bold] for commands)\n[bold yellow]>[/bold yellow] "
        )
        # Collapse whitespace so a multi-line task doesn't flood the terminal with blank lines.
        user_input = " ".join(self._prompt_and_handle_slash_commands(message).split())
        if user_input == "/u":  # directly continue
            self._drop_exit_message()
            self._interrupt("Switched to human mode.")
        elif user_input in self._MODE_COMMANDS_MAPPING:  # ask again
            return self._check_for_new_task_or_submit(e)
        elif user_input:
            self.extra_template_vars["task"] = user_input
            self._drop_exit_message()
            self._interrupt(f"The user added a new task: {user_input}", itype="UserNewTask")
        return None

    def _drop_exit_message(self) -> None:
        """Drop the trailing exit marker from a finished run so the conversation can continue.

        The ``exit`` role is an internal marker and is rejected by the LM API, so it must never
        be part of the history sent to the model.
        """
        if self.messages and self.messages[-1].get("role") == "exit":
            self.messages.pop()

    def _should_ask_confirmation(self, action: str) -> bool:
        return self.config.mode == "confirm" and not any(re.match(r, action) for r in self.config.whitelist_actions)

    def _ask_confirmation_or_interrupt(self, commands: list[str]) -> None:
        if not any(self._should_ask_confirmation(c) for c in commands):
            return
        prompt = (
            f"[bold yellow]Execute {len(commands)} action(s)?[/] [green][bold]Enter[/] to confirm[/], "
            "[red]type [bold]comment[/] to reject[/], or [blue][bold]/h[/] to show available commands[/]\n"
            "[bold yellow]>[/bold yellow] "
        )
        match user_input := self._prompt_and_handle_slash_commands(prompt).strip():
            case "" | "/y":
                pass  # confirmed, do nothing
            case "/u":  # Skip execution action and get back to query
                self._interrupt("Commands not executed. Switching to human mode", itype="UserRejection")
            case _:
                self._interrupt(
                    f"Commands not executed. The user rejected your commands with the following message: {user_input}",
                    itype="UserRejection",
                )

    def _prompt_for_resume_message(self) -> str:
        """Ask the user for the message that continues a conversation loaded with `/resume`."""
        prompt = (
            "[bold yellow]Resumed conversation.[/bold yellow] "
            "[green]Type a message for the model[/green] (/h for commands)\n[bold yellow]>[/bold yellow] "
        )
        while True:
            message = self._prompt_and_handle_slash_commands(prompt).strip()
            if message and message not in self._MODE_COMMANDS_MAPPING:
                return message

    def _prompt_and_handle_slash_commands(self, prompt: str, *, _multiline: bool = False) -> str:
        """Prompts the user, takes care of /h (followed by requery) and sets the mode. Returns the user input."""
        console.print(prompt, end="")
        if _multiline:
            return _multiline_prompt()
        user_input = prompt_session.prompt("")
        if user_input == "/m":
            return self._prompt_and_handle_slash_commands(prompt, _multiline=True)
        if user_input == "/h":
            console.print(
                f"Current mode: [bold green]{self.config.mode}[/bold green]\n"
                f"[bold green]/y[/bold green] to switch to [bold yellow]yolo[/bold yellow] mode (execute LM commands without confirmation)\n"
                f"[bold green]/c[/bold green] to switch to [bold yellow]confirmation[/bold yellow] mode (ask for confirmation before executing LM commands)\n"
                f"[bold green]/u[/bold green] to switch to [bold yellow]human[/bold yellow] mode (execute commands issued by the user)\n"
                f"[bold green]/m[/bold green] to enter multiline comment\n"
                f"[bold green]/new[/bold green] to start a new conversation (discards the current history)\n"
                f"[bold green]/resume[/bold green] to list saved conversations and continue one\n"
                f"[bold green]/compact[/bold green] to summarize the conversation into a smaller context",
            )
            return self._prompt_and_handle_slash_commands(prompt)
        if user_input == "/compact":
            before = len(self.messages)
            if self.compact() is None:
                console.print("[bold yellow]Nothing to compact yet.[/bold yellow]")
            else:
                console.print(
                    f"[bold green]Conversation compacted[/bold green] "
                    f"[dim]({before} messages -> {len(self.messages)})[/dim]"
                )
            return self._prompt_and_handle_slash_commands(prompt)
        if user_input == "/new" or user_input.startswith("/new "):
            task = user_input[len("/new") :].strip()
            console.print("[bold green]Starting a new conversation.[/bold green]")
            if not task:
                console.print("[bold yellow]What do you want to do?[/bold yellow]")
                console.print("[bold yellow]>[/bold yellow] ", end="")
                task = _multiline_prompt()
            raise NewConversation(task)
        if user_input == "/resume" or user_input.startswith("/resume "):
            if (path := self._select_conversation(user_input[len("/resume") :].strip())) is not None:
                console.print(f"[bold green]Resuming conversation {escape(str(path))}[/bold green]")
                raise ResumeConversation(path)
            return self._prompt_and_handle_slash_commands(prompt)
        if user_input in self._MODE_COMMANDS_MAPPING:
            if self.config.mode == self._MODE_COMMANDS_MAPPING[user_input]:
                return self._prompt_and_handle_slash_commands(
                    f"[bold red]Already in {self.config.mode} mode.[/bold red]\n{prompt}"
                )
            self.config.mode = self._MODE_COMMANDS_MAPPING[user_input]
            console.print(f"Switched to [bold green]{self.config.mode}[/bold green] mode.")
            return user_input
        return user_input
