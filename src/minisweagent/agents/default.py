"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation
or https://minimal-agent.com for a tutorial on the basic building principles.
"""

import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Literal

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from minisweagent import Environment, Model, __version__
from minisweagent.exceptions import FormatError, InterruptAgentFlow, LimitsExceeded, TimeExceeded
from minisweagent.models.utils.content_string import get_content_string
from minisweagent.utils.serialize import recursive_merge

COMPACTION_SYSTEM_PROMPT = (
    "You compress an AI coding agent's conversation history into a concise summary so that the agent "
    "can keep working with a much smaller context. You never run commands; you only reply with the summary."
)

COMPACTION_TEMPLATE = """\
The conversation below is getting long. Summarize it so the work can continue with a much smaller context.

<conversation>
{{ conversation }}
</conversation>

Reply with a concise Markdown summary using exactly these sections (keep every section, using "(none)" when empty):
## Objective
- What the user is trying to accomplish (one or two sentences).
## Important Details
- Decisions, constraints, preferences and facts needed to continue.
## Work State
### Completed
- Finished work and verified facts.
### Active
- Current work, partial changes and investigation state.
### Blocked
- Failing commands, errors and unknowns.
## Next Move
1. The immediate next action.
2. The following action, if known.
## Relevant Files
- Files or directories and why they matter.

Rules:
- Use terse bullets, not prose.
- Preserve exact file paths, symbols, commands, error strings, URLs and identifiers.
- Do not mention this summary process or that the context was compacted.
- Do not run any commands; reply with the summary only."""


class AgentConfig(BaseModel):
    """Check the config files in minisweagent/config for example settings."""

    system_template: str
    """Template for the system message (the first message)."""
    instance_template: str
    """Template for the first user message specifying the task (the second message overall)."""
    compaction_template: str = COMPACTION_TEMPLATE
    """Template used to summarize the conversation when the user runs ``/compact``.

    ``{{ conversation }}`` is replaced with the rendered message history. The model's reply
    replaces the older messages while the system and task messages are kept."""
    step_limit: int = 0
    """Maximum number of steps the agent can take."""
    wall_time_limit_seconds: int = 0
    """Stop agent after this many seconds of wall-clock time. 0 means no limit."""
    max_consecutive_format_errors: int = 3
    """Exit after this many format errors in a row (0 = no limit)."""
    output_path: Path | None = None
    """Save the trajectory to this path."""
    notify_channel: Literal["terminal_bell", "none"] = "none"
    """Where to send the "task completed" alert: ``terminal_bell`` rings the terminal bell, ``none`` is silent.

    An explicitly set value wins; otherwise the ``MSWEA_NOTIFY_CHANNEL`` (or ``NOTIFY_CHANNEL``) environment
    variable is used as the default."""


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        """See the `AgentConfig` class for permitted keyword arguments."""
        if "notify_channel" not in kwargs:
            channel = os.getenv("MSWEA_NOTIFY_CHANNEL") or os.getenv("NOTIFY_CHANNEL")
            if channel in ("terminal_bell", "none"):
                kwargs["notify_channel"] = channel
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.logger = logging.getLogger("agent")
        self.cost = 0.0
        self.n_calls = 0
        self.n_consecutive_format_errors = 0
        self._start_time = time.time()

    def get_template_vars(self, **kwargs) -> dict:
        return recursive_merge(
            self.config.model_dump(),
            self.env.get_template_vars(),
            self.model.get_template_vars(),
            {
                "n_model_calls": self.n_calls,
                "model_cost": self.cost,
                "elapsed_seconds": int(time.time() - self._start_time),
            },
            self.extra_template_vars,
            kwargs,
        )

    def _render_template(self, template: str) -> str:
        return Template(template, undefined=StrictUndefined).render(**self.get_template_vars())

    def _query_text(self, messages: list[dict]) -> dict:
        """Ask the model for a plain-text reply (no bash action), for example when summarizing."""
        if hasattr(self.model, "query_text"):
            return self.model.query_text(messages)
        return self.model.query(messages)

    def compact(self) -> str | None:
        """Summarize the conversation with the configured model and shrink the history.

        The system message and the original task message are kept verbatim; everything else is
        replaced with a single summary message written by the model, so the agent keeps working
        with a much smaller context. The summary is built from the task and the whole conversation.
        Returns the summary, or ``None`` when the conversation is too short to compact.
        """
        if len(self.messages) <= 2:
            return None
        # Include everything after the system prompt so the summary also captures the task.
        conversation = "\n\n".join(
            f"<{role}>\n{get_content_string(message)}\n</{role}>"
            for message in self.messages[1:]
            if (role := message.get("role") or "assistant")
        )
        prompt = Template(self.config.compaction_template, undefined=StrictUndefined).render(
            **recursive_merge(self.get_template_vars(), {"conversation": conversation})
        )
        response = self._query_text(
            [
                self.model.format_message(role="system", content=COMPACTION_SYSTEM_PROMPT),
                self.model.format_message(role="user", content=prompt),
            ]
        )
        self.cost += response.get("extra", {}).get("cost", 0.0)
        self.n_calls += 1
        # Only accept real text content: a model that ignores the instructions and calls a tool
        # (or only emits reasoning) must not turn its bash command into the summary.
        summary = get_content_string(response, skip_tool_calls=True, include_reasoning=False).strip()
        if not summary:
            return None
        self.messages = self.messages[:2] + [
            self.model.format_message(
                role="user",
                content=(
                    "The earlier conversation was compacted into the summary below. Continue the task "
                    "from here without repeating completed work.\n\n"
                    f"<summary>\n{summary}\n</summary>"
                ),
            )
        ]
        return summary

    def add_messages(self, *messages: dict) -> list[dict]:
        self.logger.debug(messages)  # set log level to debug to see
        self.messages.extend(messages)
        if any(m.get("role") == "exit" for m in messages) and self.config.notify_channel == "terminal_bell":
            self._notify_task_completed()
        return list(messages)

    def _notify_task_completed(self) -> None:
        """Ring the terminal bell (when configured) to alert that the agent has finished."""
        print("\a", end="", flush=True)

    def handle_uncaught_exception(self, e: Exception) -> list[dict]:
        return self.add_messages(
            self.model.format_message(
                role="exit",
                content=str(e),
                extra={
                    "exit_status": type(e).__name__,
                    "submission": "",
                    "exception_str": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
        )

    def run(self, task: str = "", **kwargs) -> dict:
        """Run step() until agent is finished. Returns dictionary with exit_status, submission keys.

        If the agent already holds messages (for example after `load`), those are continued instead
        of starting a new trajectory. Passing ``resume=<path>`` loads that trajectory first.
        """
        if (resume := kwargs.pop("resume", None)) is not None:
            self.load(None if resume is True else resume)
        if not self.messages:
            self.extra_template_vars |= {"task": task, **kwargs}
            self.add_messages(
                self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
                self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
            )
        while self.messages[-1].get("role") != "exit":
            try:
                self.step()
                self.n_consecutive_format_errors = 0  # reset on any clean step
            except FormatError as e:
                # The call was billed before parsing failed, so query() never got to charge it.
                self.cost += e.messages[0].get("extra", {}).get("cost", 0.0)
                self.n_consecutive_format_errors += 1
                if 0 < self.config.max_consecutive_format_errors <= self.n_consecutive_format_errors:
                    self.add_messages(
                        *e.messages,
                        {
                            "role": "exit",
                            "content": "RepeatedFormatError",
                            "extra": {"exit_status": "RepeatedFormatError", "submission": ""},
                        },
                    )
                else:
                    self.add_messages(*e.messages)
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            finally:
                self.save(self.config.output_path)
        return self.messages[-1].get("extra", {})

    def load(self, path: Path | None = None) -> dict:
        """Load a saved trajectory and restore the agent state so that `run` can continue it.

        This restores the message history, the model call/cost counters, the task and the wall-clock
        start time, and makes subsequent calls to `save` write back to `path` (defaults to the
        configured output path).
        """
        path = Path(path or self.config.output_path)
        data = json.loads(path.read_text())
        messages = data.get("messages")
        if not messages:
            raise ValueError(f"Trajectory {path} does not contain any messages")
        self.messages = messages
        info = data.get("info", {})
        stats = info.get("model_stats", {})
        self.cost = stats.get("instance_cost", self.cost)
        self.n_calls = stats.get("api_calls", self.n_calls)
        if task := info.get("task"):
            self.extra_template_vars["task"] = task
        self._start_time = time.time() - info.get("elapsed_seconds", 0)
        self.config.output_path = path
        return data

    def resume(self, path: Path | None = None) -> dict:
        """Load a saved trajectory and continue it from where it stopped."""
        self.load(path)
        return self.run()

    def step(self) -> list[dict]:
        """Query the LM, execute actions."""
        return self.execute_actions(self.query())

    def query(self) -> dict:
        """Query the model and return model messages. Override to add hooks."""
        if 0 < self.config.step_limit <= self.n_calls:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        if 0 < self.config.wall_time_limit_seconds <= int(time.time() - self._start_time):
            raise TimeExceeded(
                {
                    "role": "exit",
                    "content": "TimeExceeded",
                    "extra": {"exit_status": "TimeExceeded", "submission": ""},
                }
            )
        self.n_calls += 1
        message = self.model.query(self.messages)
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions in message, add observation messages, return them."""
        outputs = [self.env.execute(action) for action in message.get("extra", {}).get("actions", [])]
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))

    def serialize(self, *extra_dicts) -> dict:
        """Serialize agent state to a json-compatible nested dictionary for saving."""
        last_message = self.messages[-1] if self.messages else {}
        last_extra = last_message.get("extra", {})
        agent_data = {
            "info": {
                "model_stats": {
                    "instance_cost": self.cost,
                    "api_calls": self.n_calls,
                },
                "config": {
                    "agent": self.config.model_dump(mode="json"),
                    "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "mini_version": __version__,
                "task": self.extra_template_vars.get("task", ""),
                "elapsed_seconds": int(time.time() - self._start_time),
                "exit_status": last_extra.get("exit_status", ""),
                "submission": last_extra.get("submission", ""),
            },
            "messages": self.messages,
            "trajectory_format": "mini-swe-agent-1.1",
        }
        return recursive_merge(agent_data, self.model.serialize(), self.env.serialize(), *extra_dicts)

    def save(self, path: Path | None, *extra_dicts) -> dict:
        """Save the trajectory of the agent to a file if path is given. Returns full serialized data.
        You can pass additional dictionaries with extra data to be (recursively) merged into the output data.
        """
        data = self.serialize(*extra_dicts)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
        return data
