"""Tests for blocking the agent's access to configured files/folders (``MSWEA_DISABLE_AGENT_ACCESS``)."""

from pathlib import Path

import pytest
import yaml

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import DISABLED_ACCESS_ENV_VAR, LocalEnvironment, LocalEnvironmentConfig
from minisweagent.models.test_models import DeterministicModel, make_output


@pytest.fixture(autouse=True)
def _no_global_disabled_access(monkeypatch):
    """The real global .env may set the variable; isolate tests from it unless they opt in."""
    monkeypatch.delenv(DISABLED_ACCESS_ENV_VAR, raising=False)


def _agent_config() -> dict:
    return yaml.safe_load(Path("src/minisweagent/config/default.yaml").read_text())["agent"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("/etc/passwd", ["/etc/passwd"]),
        ("/etc/passwd:/home/me/.ssh", ["/etc/passwd", "/home/me/.ssh"]),
        ("/etc/passwd:", ["/etc/passwd"]),
        ("", []),
        (["/a", "/b"], ["/a", "/b"]),
    ],
)
def test_disabled_access_accepts_string_and_list(value, expected):
    assert LocalEnvironmentConfig(disabled_access=value).disabled_access == expected


def test_disabled_access_reads_env_var(monkeypatch):
    monkeypatch.setenv(DISABLED_ACCESS_ENV_VAR, "/etc/passwd:/etc/shadow")
    env = LocalEnvironment()
    assert env.config.disabled_access == ["/etc/passwd", "/etc/shadow"]
    assert "Access denied" in env.execute({"command": "cat /etc/passwd"})["output"]


def test_explicit_disabled_access_overrides_env_var(monkeypatch):
    monkeypatch.setenv(DISABLED_ACCESS_ENV_VAR, "/etc/passwd")
    env = LocalEnvironment(disabled_access=[])
    assert env.config.disabled_access == []
    assert "Access denied" not in env.execute({"command": "cat /etc/passwd"})["output"]


def test_disabled_file_is_refused_without_leaking_content(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret content")
    env = LocalEnvironment(disabled_access=str(secret))

    result = env.execute({"command": f"cat {secret}"})

    assert result["returncode"] != 0
    assert "top secret content" not in result["output"]
    assert str(secret) in result["output"]
    assert "not allowed" in result["output"]
    assert result["extra"]["exception_type"] == "AccessDenied"


def test_disabled_folder_refuses_files_inside_it(tmp_path):
    folder = tmp_path / "private"
    folder.mkdir()
    (folder / "notes.txt").write_text("do not read me")
    env = LocalEnvironment(disabled_access=str(folder), cwd=str(tmp_path))

    assert "Access denied" in env.execute({"command": "cat private/notes.txt"})["output"]
    assert "Access denied" in env.execute({"command": f"cat {folder / 'notes.txt'}"})["output"]


def test_sibling_path_with_shared_prefix_is_allowed(tmp_path):
    blocked = tmp_path / "secret.txt"
    blocked.write_text("secret")
    allowed = tmp_path / "secret.txt.bak"
    allowed.write_text("backup")
    env = LocalEnvironment(disabled_access=str(blocked), cwd=str(tmp_path))

    result = env.execute({"command": f"cat {allowed}"})

    assert result["returncode"] == 0
    assert result["output"].strip() == "backup"


@pytest.mark.parametrize(
    "command",
    [
        "cat /etc/passwd",
        "cat '/etc/passwd'",
        "cat </etc/passwd",
        "tail -n 1 /etc/passwd | grep root",
        "python3 -c \"print(open('/etc/passwd').read())\"",
        "cd /etc && cat passwd",
    ],
)
def test_various_ways_of_referencing_a_disabled_path_are_refused(command):
    env = LocalEnvironment(disabled_access="/etc/passwd")

    result = env.execute({"command": command})

    assert result["returncode"] != 0, command
    assert "Access denied" in result["output"]


def test_unrelated_command_still_runs():
    env = LocalEnvironment(disabled_access=["/etc/passwd", "/etc/shadow"])
    result = env.execute({"command": "echo 'hello world'"})
    assert (result["returncode"], result["output"].strip()) == (0, "hello world")


def test_agent_receives_access_denied_as_observation(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret content")
    model = DeterministicModel(
        outputs=[
            make_output("reading", [{"command": f"cat {secret}"}]),
            make_output("done", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho ok"}]),
        ]
    )
    agent = DefaultAgent(model, LocalEnvironment(disabled_access=str(secret)), **_agent_config())

    info = agent.run("read the secret")

    assert info["exit_status"] == "Submitted"
    observation = agent.messages[3]["content"]
    assert "Access denied" in observation
    assert "top secret content" not in observation
