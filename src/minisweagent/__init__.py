"""
This file provides:

- Path settings for global config file & relative directories
- Version numbering
- Protocols for the core components of mini-swe-agent.
  By the magic of protocols & duck typing, you can pretty much ignore them,
  unless you want the static type checking.
"""

__version__ = "2.4.6"

import os
import sys
from pathlib import Path
from typing import Any, Protocol

import dotenv
from platformdirs import user_config_dir

from minisweagent.utils.log import logger

package_dir = Path(__file__).resolve().parent


# On macOS, use the XDG-style config dir for consistency with Linux instead of
# platformdirs' `~/Library/Application Support`.
default_config_dir = (
    Path.home() / ".config" / "mini-swe-agent" if sys.platform == "darwin" else Path(user_config_dir("mini-swe-agent"))
)
global_config_dir = Path(os.getenv("MSWEA_GLOBAL_CONFIG_DIR") or default_config_dir)
global_config_dir.mkdir(parents=True, exist_ok=True)
global_config_file = global_config_dir / ".env"

dotenv.load_dotenv(dotenv_path=global_config_file)
# Config left in the platformdirs location (macOS before the switch) still applies,
# but only for keys the file above doesn't set (`load_dotenv` doesn't override).
legacy_config_file = Path(user_config_dir("mini-swe-agent")) / ".env"
if legacy_config_file != global_config_file:
    dotenv.load_dotenv(dotenv_path=legacy_config_file)


# === Protocols ===
# You can ignore them unless you want static type checking.


class Model(Protocol):
    """Protocol for language models."""

    config: Any

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict: ...

    def query_text(self, messages: list[dict[str, str]], **kwargs) -> dict: ...

    def format_message(self, **kwargs) -> dict: ...

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]: ...

    def get_template_vars(self, **kwargs) -> dict[str, Any]: ...

    def serialize(self) -> dict: ...


class Environment(Protocol):
    """Protocol for execution environments."""

    config: Any

    def execute(self, action: dict, cwd: str = "") -> dict[str, Any]: ...

    def get_template_vars(self, **kwargs) -> dict[str, Any]: ...

    def serialize(self) -> dict: ...


class Agent(Protocol):
    """Protocol for agents."""

    config: Any

    def run(self, task: str, **kwargs) -> dict: ...

    def save(self, path: Path | None, *extra_dicts) -> dict: ...


__all__ = [
    "Agent",
    "Model",
    "Environment",
    "package_dir",
    "__version__",
    "global_config_file",
    "global_config_dir",
    "logger",
]
