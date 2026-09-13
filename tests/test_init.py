"""Tests for minisweagent.__init__."""

import os
import subprocess
import sys

import pytest


def test_startup_banner_survives_non_utf8_stdout(tmp_path):
    """Importing the package must not crash when stdout can't encode the startup banner (e.g. Windows cp1252)."""
    env = {
        **os.environ,
        "PYTHONIOENCODING": "cp1252",
        "MSWEA_SILENT_STARTUP": "",
        "MSWEA_GLOBAL_CONFIG_DIR": str(tmp_path),
    }
    result = subprocess.run([sys.executable, "-c", "import minisweagent"], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(sys.platform != "darwin", reason="only macOS diverts from the platformdirs location")
def test_macos_xdg_config_dir_with_legacy_fallback(tmp_path):
    """macOS uses `~/.config/mini-swe-agent`; a config left in the platformdirs location still applies,
    but only for keys the new location doesn't set."""
    xdg_dir = tmp_path / ".config" / "mini-swe-agent"
    legacy_dir = tmp_path / "Library" / "Application Support" / "mini-swe-agent"
    xdg_dir.mkdir(parents=True)
    legacy_dir.mkdir(parents=True)
    (xdg_dir / ".env").write_text("MSWEA_TEST_MAIN=main\nMSWEA_TEST_SHARED=main\n")
    (legacy_dir / ".env").write_text("MSWEA_TEST_LEGACY=legacy\nMSWEA_TEST_SHARED=legacy\n")
    code = "\n".join([
        "import os",
        "from pathlib import Path",
        "from minisweagent import global_config_dir",
        "assert global_config_dir == Path.home() / '.config' / 'mini-swe-agent'",
        "assert os.environ['MSWEA_TEST_MAIN'] == 'main'",
        "assert os.environ['MSWEA_TEST_LEGACY'] == 'legacy'",
        "assert os.environ['MSWEA_TEST_SHARED'] == 'main', 'the ~/.config file must win over the legacy one'",
    ])
    env = {k: v for k, v in os.environ.items() if k != "MSWEA_GLOBAL_CONFIG_DIR"} | {"HOME": str(tmp_path)}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
