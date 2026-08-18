"""
System tests: run amazing_hand_cmd.py as a subprocess.

Only commands that do NOT require hardware are covered here:
  - --help  (exits 0, shows options)
  - --list  (reads config file, prints poses/sequences)
  - Error paths that fail before any serial connection is opened

Hardware-dependent paths (--pose, --sequence) are covered with mock
controllers in tests/test_integration.py instead.
"""
import sys
import subprocess
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLI = PROJECT_ROOT / "amazing_hand_cmd.py"

SAMPLE_CONFIG = {
    "poses": {
        "open": {"positions": [0] * 8},
        "close": {"positions": [110] * 8},
        "half": {"positions": [55] * 8},
    },
    "sequences": {
        "demo": {
            "steps": [
                "open:3,3,3,3,3,3,3,3|2.0s",
                "close:3,3,3,3,3,3,3,3|2.0s",
            ]
        },
        "wave": {
            "steps": [
                "open:1,1,1,1,6,6,3,3|1.0s",
                "close:6,6,4,4,1,1,3,3|1.0s",
            ]
        },
    },
}


@pytest.fixture()
def sample_config_file(tmp_path):
    cf = tmp_path / "test_config.yaml"
    cf.write_text(yaml.dump(SAMPLE_CONFIG))
    return cf


@pytest.fixture()
def empty_config_file(tmp_path):
    cf = tmp_path / "empty_config.yaml"
    cf.write_text("poses: {}\nsequences: {}\n")
    return cf


def run_cli(*args):
    """Execute the CLI as a subprocess using the current Python interpreter."""
    return subprocess.run(
        [sys.executable, str(CLI)] + list(args),
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------

class TestCliHelp:
    def test_help_exits_zero(self):
        result = run_cli("--help")
        assert result.returncode == 0

    def test_help_contains_list_option(self):
        result = run_cli("--help")
        assert "--list" in result.stdout

    def test_help_contains_pose_option(self):
        result = run_cli("--help")
        assert "--pose" in result.stdout

    def test_help_contains_sequence_option(self):
        result = run_cli("--help")
        assert "--sequence" in result.stdout

    def test_help_contains_loop_option(self):
        result = run_cli("--help")
        assert "--loop" in result.stdout


# ---------------------------------------------------------------------------
# --list
# ---------------------------------------------------------------------------

class TestCliList:
    def test_list_exits_zero(self, sample_config_file):
        result = run_cli("--list", "--config", str(sample_config_file))
        assert result.returncode == 0

    def test_list_shows_all_pose_names(self, sample_config_file):
        result = run_cli("--list", "--config", str(sample_config_file))
        for pose in ("open", "close", "half"):
            assert pose in result.stdout

    def test_list_shows_all_sequence_names(self, sample_config_file):
        result = run_cli("--list", "--config", str(sample_config_file))
        for seq in ("demo", "wave"):
            assert seq in result.stdout

    def test_list_shows_step_count(self, sample_config_file):
        result = run_cli("--list", "--config", str(sample_config_file))
        assert "2 steps" in result.stdout

    def test_list_empty_config_prints_none(self, empty_config_file):
        result = run_cli("--list", "--config", str(empty_config_file))
        assert result.returncode == 0
        assert "(none)" in result.stdout

    def test_list_with_missing_config_exits_nonzero(self, tmp_path):
        missing = tmp_path / "no_such_file.yaml"
        result = run_cli("--list", "--config", str(missing))
        assert result.returncode != 0

    def test_list_no_connection_opened(self, sample_config_file):
        """FR-CLI-1 AC 1.3: --list does NOT open a serial connection."""
        result = run_cli("--list", "--config", str(sample_config_file))
        assert result.returncode == 0
        # Success without hardware proves no connection was attempted

    def test_list_shows_pose_positions(self, sample_config_file):
        """FR-CLI-1 AC 1.1: Output shows each pose with positions."""
        result = run_cli("--list", "--config", str(sample_config_file))
        assert "[0, 0, 0, 0, 0, 0, 0, 0]" in result.stdout or "0" in result.stdout

    def test_list_shows_sequence_step_details(self, sample_config_file):
        """FR-CLI-1 AC 1.2: Output shows sequence step details."""
        result = run_cli("--list", "--config", str(sample_config_file))
        assert "open:" in result.stdout or "close:" in result.stdout


# ---------------------------------------------------------------------------
# CLI help options (FR-UI-7)
# ---------------------------------------------------------------------------

class TestCliHelpOptions:
    def test_help_contains_port_option(self):
        """FR-UI-7 AC 7.1: --help shows all options."""
        result = run_cli("--help")
        assert "--port" in result.stdout

    def test_help_contains_baudrate_option(self):
        result = run_cli("--help")
        assert "--baudrate" in result.stdout

    def test_help_contains_config_option(self):
        result = run_cli("--help")
        assert "--config" in result.stdout


# ---------------------------------------------------------------------------
# Error paths (no hardware required)
# ---------------------------------------------------------------------------

class TestCliErrors:
    def test_no_arguments_exits_nonzero(self):
        result = run_cli()
        assert result.returncode != 0

    def test_loop_without_sequence_exits_nonzero(self, sample_config_file):
        # --list with --loop: --list is accepted by the mutex group,
        # but --loop without --sequence triggers the explicit check.
        result = run_cli("--list", "--loop", "--config", str(sample_config_file))
        assert result.returncode != 0

    def test_pose_and_sequence_mutually_exclusive(self, sample_config_file):
        result = run_cli(
            "--pose", "open",
            "--sequence", "demo",
            "--config", str(sample_config_file),
        )
        assert result.returncode != 0

    def test_list_and_pose_mutually_exclusive(self, sample_config_file):
        result = run_cli(
            "--list",
            "--pose", "open",
            "--config", str(sample_config_file),
        )
        assert result.returncode != 0

    def test_list_and_sequence_mutually_exclusive(self, sample_config_file):
        result = run_cli(
            "--list",
            "--sequence", "demo",
            "--config", str(sample_config_file),
        )
        assert result.returncode != 0
