import os
import platform
import re
import shlex
import signal
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator

from minisweagent.exceptions import Submitted
from minisweagent.utils.serialize import recursive_merge

DISABLED_ACCESS_ENV_VAR = "MSWEA_DISABLE_AGENT_ACCESS"

# Path-like tokens inside a command (absolute paths, `~/...`, `./...`, `../...`). Embedded paths such as
# `open("/etc/passwd")` are found this way, because a plain argument split would return the whole expression.
_PATH_PATTERN = re.compile(r"(?:~|\.{1,2})?(?:/[^\s'\"<>|;&()]+)+")


class LocalEnvironmentConfig(BaseModel):
    cwd: str = ""
    env: dict[str, str] = {}
    timeout: int = 30
    disabled_access: list[str] = []
    """Paths (files or folders) the agent is not allowed to access. Commands that reference any of them are
    refused instead of being executed. When unset, the ``MSWEA_DISABLE_AGENT_ACCESS`` environment variable is
    used as a colon-separated default (e.g. ``/home/me/.ssh:/home/me/secrets``)."""

    @field_validator("disabled_access", mode="before")
    @classmethod
    def _split_disabled_access(cls, value: Any) -> Any:
        """Accept a colon-separated string (as read from the environment variable) or a list of paths."""
        if isinstance(value, str):
            return [p.strip() for p in re.split(rf"[{re.escape(os.pathsep)}\n]", value) if p.strip()]
        return value


class LocalEnvironment:
    def __init__(self, *, config_class: type = LocalEnvironmentConfig, **kwargs):
        """This class executes bash commands directly on the local machine."""
        if "disabled_access" not in kwargs:
            kwargs["disabled_access"] = os.getenv(DISABLED_ACCESS_ENV_VAR, "")
        self.config = config_class(**kwargs)

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the local environment and return the result as a dict."""
        command = action.get("command", "")
        cwd = cwd or self.config.cwd or os.getcwd()
        if blocked := self._disabled_access(command, cwd):
            output = {
                "output": (
                    f"Access denied: '{blocked}' is off-limits, so this command was not executed. "
                    "Reading, writing, or otherwise referencing this path is not allowed. "
                    "Do not try to access it again; continue with a different approach."
                ),
                "returncode": 1,
                "exception_info": f"Access to '{blocked}' is disabled by the {DISABLED_ACCESS_ENV_VAR} policy.",
                "extra": {"exception_type": "AccessDenied", "exception": blocked},
            }
        else:
            try:
                result = _run(command, cwd, os.environ | self.config.env, timeout or self.config.timeout)
                output = {"output": result.stdout, "returncode": result.returncode, "exception_info": ""}
            except Exception as e:
                raw_output = getattr(e, "output", None)
                raw_output = (
                    raw_output.decode("utf-8", errors="replace")
                    if isinstance(raw_output, bytes)
                    else (raw_output or "")
                )
                output = {
                    "output": raw_output,
                    "returncode": -1,
                    "exception_info": f"An error occurred while executing the command: {e}",
                    "extra": {"exception_type": type(e).__name__, "exception": str(e)},
                }
        self._check_finished(output)
        return output

    def _disabled_access(self, command: str, cwd: str) -> str | None:
        """Return the first disabled path referenced by ``command``, or ``None`` if the command is allowed."""
        if not self.config.disabled_access:
            return None
        disabled = [_resolve_path(p, cwd) for p in self.config.disabled_access]
        for candidate, candidate_cwd in _referenced_paths(command, cwd):
            resolved = _resolve_path(candidate, candidate_cwd)
            if any(resolved == d or resolved.is_relative_to(d) for d in disabled):
                return candidate
        return None

    def _check_finished(self, output: dict):
        """Raises Submitted if the output indicates task completion."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(self.config.model_dump(), platform.uname()._asdict(), os.environ, kwargs)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }


def _resolve_path(path: str, cwd: str) -> Path:
    """Expand ``~`` and resolve ``path`` to an absolute path, interpreting relative paths against ``cwd``."""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path(cwd) / resolved
    try:
        return resolved.resolve()
    except (OSError, RuntimeError):
        return resolved.absolute()


def _referenced_paths(command: str, cwd: str) -> Iterator[tuple[str, str]]:
    """Yield ``(path, cwd)`` for every path-like token ``command`` references, following ``cd`` changes.

    Both plain arguments and paths embedded in expressions (e.g. ``open("/etc/passwd")``) are reported.
    """
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars="();<>|&")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:  # unbalanced quotes; fall back to a plain whitespace split
        tokens = command.split()
    current, index = cwd, 0
    while index < len(tokens):
        token = tokens[index]
        if token == "cd" and index + 1 < len(tokens) and not tokens[index + 1].startswith("-"):
            current = str(_resolve_path(tokens[index + 1], current))
            index += 2
            continue
        cleaned = token.strip("<>|&;,")
        candidates = [cleaned] if cleaned and not cleaned.startswith("-") else []
        for candidate in candidates + _PATH_PATTERN.findall(token):
            yield candidate, current
        index += 1


def _run(command: str, cwd: str, env: dict[str, str], timeout: int) -> subprocess.CompletedProcess[str]:
    """Like subprocess.run, but kills the whole process group on timeout so no children are orphaned."""
    process = subprocess.Popen(
        command,
        shell=True,
        text=True,
        cwd=cwd,
        env=env,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=os.name == "posix",
    )
    try:
        stdout, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL) if os.name == "posix" else process.kill()
        stdout, _ = process.communicate()
        raise subprocess.TimeoutExpired(command, timeout, output=stdout)
    return subprocess.CompletedProcess(command, process.returncode, stdout=stdout)
